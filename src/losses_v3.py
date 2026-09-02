"""
Adds four things to your existing Charbonnier + Sobel + VGG loss:

  - FFT magnitude loss: penalizes missing high-frequency content directly in
    the frequency domain -- complements Sobel (which is a spatial-domain
    edge proxy) by targeting the actual thing 2x downsampling destroys.
  - MS-SSIM loss: SSIM is one of the three scored metrics, so optimizing it
    directly (multi-scale, more stable gradient-wise than single-scale SSIM)
    should help more than only optimizing pixel/perceptual proxies for it.
  - A LOW-weight adversarial term (LSGAN, from a small PatchDiscriminator)
    to sharpen fine detail. Kept subordinate to the reconstruction losses by
    design -- this is training-only and costs nothing at inference.
  - Gram matrix (style/texture) loss: unlike VGG perceptual loss, which
    penalizes features being in the WRONG POSITION, Gram matrix loss
    penalizes the wrong TEXTURE STATISTICS regardless of position. This
    targets over-smoothing directly -- pixel/position losses reward
    averaging away texture the model is unsure about; Gram loss doesn't
    care about exact position, only whether the right kind of texture
    pattern is present, so it pushes the model to keep committing to real
    texture instead of blurring it out.

Needs: pip install pytorch-msssim
(pure functions, no custom MS-SSIM implementation -- less failure-prone
than hand-rolling the multi-scale Gaussian pyramid ourselves)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

from src.dataset_and_losses import charbonnier_loss, edge_loss
from src.advanced_loss import GramMatrixLoss


def fft_loss(pred, target):
    """L1 distance between FFT magnitudes. Operates per-image; pred/target
    are (B, 1, H, W)."""
    pred_fft = torch.fft.rfft2(pred, norm="ortho")
    target_fft = torch.fft.rfft2(target, norm="ortho")
    return F.l1_loss(pred_fft.abs(), target_fft.abs())


def ms_ssim_loss(pred, target):
    """1 - MS-SSIM, so it's a quantity to minimize like the other terms.
    data_range=1.0 assumes pred/target roughly in [0,1] at this point --
    true for pred since the model clips/sigmoids at output, and for target
    since GT is clean signal. If your GT isn't guaranteed in [0,1], pass the
    actual range explicitly."""
    return 1.0 - ms_ssim(pred, target, data_range=1.0, size_average=True)


def combined_loss_v3(pred, target, vgg_loss_fn, gram_loss_fn=None,
                      vgg_weight=0.05, fft_weight=0.05, msssim_weight=0.2,
                      gram_weight=0.0):
    """Reconstruction loss only -- no adversarial term here. Use this for
    warmup epochs, or always if you decide not to use the discriminator.

    gram_loss_fn is optional (defaults to None / gram_weight=0.0) so this
    function stays backward compatible with any existing calls that don't
    pass it -- the Gram term only activates if both are supplied."""
    pixel = charbonnier_loss(pred, target)
    edge = edge_loss(pred, target)
    perceptual = vgg_loss_fn(pred, target)
    freq = fft_loss(pred, target)
    msssim = ms_ssim_loss(pred, target)
    total = pixel + 0.1 * edge + vgg_weight * perceptual + fft_weight * freq + msssim_weight * msssim
    if gram_loss_fn is not None and gram_weight > 0:
        total = total + gram_weight * gram_loss_fn(pred, target)
    return total


def generator_adversarial_loss(disc_fake_logits):
    """LSGAN generator loss: push discriminator's judgment of fake patches
    toward 'real' (label 1), via MSE rather than BCE -- more stable on small
    datasets, standard choice for restoration GANs."""
    target = torch.ones_like(disc_fake_logits)
    return F.mse_loss(disc_fake_logits, target)


def discriminator_loss(disc_real_logits, disc_fake_logits):
    """LSGAN discriminator loss: real patches -> 1, fake patches -> 0."""
    real_loss = F.mse_loss(disc_real_logits, torch.ones_like(disc_real_logits))
    fake_loss = F.mse_loss(disc_fake_logits, torch.zeros_like(disc_fake_logits))
    return 0.5 * (real_loss + fake_loss)


def combined_generator_loss(pred, target, vgg_loss_fn, disc_fake_logits, gram_loss_fn=None,
                             vgg_weight=0.05, fft_weight=0.05, msssim_weight=0.2,
                             adv_weight=0.015, gram_weight=0.0):
    """Full generator loss used once GAN training is active: reconstruction
    terms + a low-weight adversarial term. adv_weight is intentionally small
    (0.01-0.02) so the discriminator sharpens detail without dominating and
    inventing texture -- the exact risk the original design avoided by
    skipping GAN loss altogether."""
    recon = combined_loss_v3(pred, target, vgg_loss_fn, gram_loss_fn,
                              vgg_weight, fft_weight, msssim_weight, gram_weight)
    adv = generator_adversarial_loss(disc_fake_logits)
    return recon + adv_weight * adv
