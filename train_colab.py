"""
Colab-ready training script -- Charbonnier + Sobel + VGG + FFT + MS-SSIM +
Gram matrix (texture) loss + low-weight adversarial, heavier augmentation
(flips + rotations + elastic + randomized synthetic re-degradation), and
disconnect-safe checkpointing so a Colab free-tier timeout doesn't lose
progress.

*** THIS VERSION ADDS --gram_weight FOR THE GRAM MATRIX TEXTURE LOSS ***
Everything else is unchanged from your original train_colab.py.

Colab setup (run these in cells before this script):

    from google.colab import drive
    drive.mount('/content/drive')

    !pip install -q pytorch-msssim

    # clone or upload your repo so `src/` is importable, then:
    %cd /content/TECHNOX-NoiseTrue-Adaptive

Usage (Gram matrix fine-tune, resuming from a prepped checkpoint seeded
from best.pth -- see prep_gram_finetune_ckpt.py):
    python train_colab.py \
        --gt_dir /content/semicon_train_data/semicon_train_data/semicon_train_data/GT \
        --noisy_dir /content/semicon_train_data/semicon_train_data/semicon_train_data/NoisyLR \
        --ckpt_dir /content/drive/MyDrive/semicon_gram_finetune_checkpoints \
        --epochs 15 --gram_weight 0.05 \
        --vgg_weight 0.05 --fft_weight 0.07 --msssim_weight 0.15

Re-running the exact same command after a disconnect automatically resumes
from the last saved checkpoint in --ckpt_dir -- nothing extra to pass.
"""

import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model_nafnet import NAFNetLiteAdaptive
from src.discriminator import PatchDiscriminator
from src.advanced_loss import VGGPerceptualLoss, GramMatrixLoss
from src.dataset_augmented_v2 import RestorationDatasetAugmentedV2
from src.losses_v3 import combined_loss_v3, combined_generator_loss, discriminator_loss


def parse_args():
    p = argparse.ArgumentParser(description="Train NoiseTrue-Adaptive, upgraded pipeline")
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--noisy_dir", required=True)
    p.add_argument("--ckpt_dir", required=True, help="Directory for resumable checkpoints (put this on Drive)")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--disc_lr", type=float, default=1e-4)
    p.add_argument("--base_ch", type=int, default=48)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synth_prob", type=float, default=0.3)
    p.add_argument("--elastic_prob", type=float, default=0.2)
    p.add_argument("--vgg_weight", type=float, default=0.05)
    p.add_argument("--fft_weight", type=float, default=0.05)
    p.add_argument("--msssim_weight", type=float, default=0.2)
    p.add_argument("--adv_weight", type=float, default=0.015)
    p.add_argument("--gram_weight", type=float, default=0.0,
                    help="Weight for the Gram matrix (texture) loss. 0.0 disables it "
                         "(fully backward compatible with old runs). Try ~0.03-0.06 -- "
                         "similar magnitude to vgg_weight, not zero, not dominant.")
    p.add_argument("--use_gan", action="store_true",
                    help="Enable the adversarial term. Off by default -- reconstruction-only "
                         "loss is the safer choice; turn this on once that's stable.")
    p.add_argument("--gan_start_epoch", type=int, default=20,
                    help="Warm up on reconstruction losses alone before introducing the "
                         "discriminator, so the generator has a sane starting point first.")
    return p.parse_args()


def save_checkpoint(path, epoch, model, disc, opt_g, opt_d, sched_g, best_val):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "disc": disc.state_dict() if disc is not None else None,
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict() if opt_d is not None else None,
        "sched_g": sched_g.state_dict(),
        "best_val": best_val,
    }, path)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("WARNING: no GPU detected -- in Colab, set Runtime > Change runtime type > GPU")

    torch.manual_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    latest_path = os.path.join(args.ckpt_dir, "latest.pth")
    best_path = os.path.join(args.ckpt_dir, "best.pth")

    dataset = RestorationDatasetAugmentedV2(
        args.gt_dir, args.noisy_dir, augment=True,
        synth_prob=args.synth_prob, elastic_prob=args.elastic_prob,
    )
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             num_workers=2, pin_memory=(device.type == "cuda"))
    print(f"Train pairs: {len(train_ds)} | Val pairs: {len(val_ds)}")

    model = NAFNetLiteAdaptive(base_ch=args.base_ch).to(device)
    print(f"Generator parameters: {sum(p.numel() for p in model.parameters()):,}")

    disc = PatchDiscriminator().to(device) if args.use_gan else None

    vgg_loss_fn = VGGPerceptualLoss().to(device)
    # Gram matrix loss reuses the SAME vgg module as VGGPerceptualLoss --
    # no double-loading of VGG weights. Only instantiated if gram_weight > 0.
    gram_loss_fn = GramMatrixLoss(vgg_loss_fn.vgg).to(device) if args.gram_weight > 0 else None
    if gram_loss_fn is not None:
        print(f"Gram matrix loss enabled, weight={args.gram_weight}")

    opt_g = torch.optim.Adam(model.parameters(), lr=args.lr)
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.disc_lr, betas=(0.5, 0.999)) if disc else None
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs)

    start_epoch = 0
    best_val = float("inf")

    # --- resume from the last checkpoint if one exists ---
    if os.path.exists(latest_path):
        print(f"Resuming from {latest_path}")
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt_g.load_state_dict(ckpt["opt_g"])
        sched_g.load_state_dict(ckpt["sched_g"])
        best_val = ckpt["best_val"]
        start_epoch = ckpt["epoch"] + 1
        if disc is not None and ckpt.get("disc") is not None:
            disc.load_state_dict(ckpt["disc"])
            opt_d.load_state_dict(ckpt["opt_d"])
        print(f"Resumed at epoch {start_epoch}, best_val so far: {best_val:.5f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        dataset.augment = True
        gan_active = args.use_gan and epoch >= args.gan_start_epoch
        if disc is not None:
            disc.train()

        total_g, total_d = 0.0, 0.0
        for noisy, gt in train_loader:
            noisy, gt = noisy.to(device), gt.to(device)

            pred = model(noisy)

            if gan_active:
                # --- discriminator step ---
                opt_d.zero_grad()
                disc_real = disc(gt)
                disc_fake = disc(pred.detach())
                d_loss = discriminator_loss(disc_real, disc_fake)
                d_loss.backward()
                opt_d.step()
                total_d += d_loss.item()

                # --- generator step, now including the adversarial term ---
                opt_g.zero_grad()
                disc_fake_for_g = disc(pred)
                g_loss = combined_generator_loss(
                    pred, gt, vgg_loss_fn, disc_fake_for_g, gram_loss_fn,
                    vgg_weight=args.vgg_weight, fft_weight=args.fft_weight,
                    msssim_weight=args.msssim_weight, adv_weight=args.adv_weight,
                    gram_weight=args.gram_weight,
                )
            else:
                # --- reconstruction-only step (warmup, or --use_gan not set) ---
                opt_g.zero_grad()
                g_loss = combined_loss_v3(
                    pred, gt, vgg_loss_fn, gram_loss_fn,
                    vgg_weight=args.vgg_weight, fft_weight=args.fft_weight,
                    msssim_weight=args.msssim_weight, gram_weight=args.gram_weight,
                )

            g_loss.backward()
            opt_g.step()
            total_g += g_loss.item()

        sched_g.step()

        model.eval()
        dataset.augment = False
        val = 0.0
        with torch.no_grad():
            for noisy, gt in val_loader:
                noisy, gt = noisy.to(device), gt.to(device)
                pred = model(noisy)
                # validation always scored on reconstruction loss only --
                # keeps "best checkpoint" comparable across the warmup/GAN boundary
                val += combined_loss_v3(
                    pred, gt, vgg_loss_fn, gram_loss_fn,
                    vgg_weight=args.vgg_weight, fft_weight=args.fft_weight,
                    msssim_weight=args.msssim_weight, gram_weight=args.gram_weight,
                ).item()
        val /= len(val_loader)

        gan_tag = f"d_loss: {total_d/len(train_loader):.5f} " if gan_active else ""
        print(f"Epoch {epoch+1}/{args.epochs} - g_loss: {total_g/len(train_loader):.5f} "
              f"- {gan_tag}- val_loss: {val:.5f} - lr: {sched_g.get_last_lr()[0]:.6f}", flush=True)

        is_best = val < best_val
        if is_best:
            best_val = val

        # save every epoch -- this is what makes Colab disconnects non-fatal
        save_checkpoint(latest_path, epoch, model, disc, opt_g, opt_d, sched_g, best_val)
        if is_best:
            torch.save(model.state_dict(), best_path)

    print(f"\nBest val_loss: {best_val:.5f}")
    print(f"Best model weights: {best_path}")
    print(f"Full resumable state: {latest_path}")


if __name__ == "__main__":
    main()
