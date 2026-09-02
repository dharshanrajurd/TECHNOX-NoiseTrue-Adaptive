import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from dataset_and_losses import charbonnier_loss, edge_loss  # reuse what already works


class VGGPerceptualLoss(nn.Module):
    """Compares images in VGG feature space instead of raw pixels -- this is
    what actually targets 'does it look right', which is what LPIPS measures.
    Your training loss never included this before -- it's the most direct fix
    for the LPIPS gap you were trying to close."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights='DEFAULT').features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        # grayscale (1 channel) -> fake RGB (3 channel), VGG expects 3-channel input
        pred3 = pred.repeat(1, 3, 1, 1)
        target3 = target.repeat(1, 3, 1, 1)

        pred_norm = (pred3 - self.mean) / self.std
        target_norm = (target3 - self.mean) / self.std

        pred_feat = self.vgg(pred_norm)
        target_feat = self.vgg(target_norm)

        return F.l1_loss(pred_feat, target_feat)


class GramMatrixLoss(nn.Module):
    """Style/texture loss: instead of matching WHERE features are (like
    VGGPerceptualLoss does), this matches the STATISTICAL CORRELATION
    between feature channels -- i.e. 'does this image have the same kind
    of texture', regardless of exact spatial position.

    This targets the over-smoothing problem directly: pixel/feature-position
    losses reward the model for averaging away texture it's unsure about,
    since a slightly-misplaced texture is still numerically close. Gram
    matrix loss doesn't care about position -- it only cares whether the
    right KIND of texture statistics are present, which pushes the model
    to keep committing to real texture instead of blurring it out.

    Uses relu3_3 (index 15 in vgg16.features), a mid-deep layer -- shallow
    layers (relu1_x) pick up pixel/noise-level statistics, which is exactly
    what we DON'T want this loss keying on given the input is noisy."""

    def __init__(self, vgg_module):
        """
        vgg_module: pass in the SAME nn.Module used by VGGPerceptualLoss
        (i.e. an existing VGGPerceptualLoss instance's .vgg attribute) so
        VGG weights aren't loaded into memory twice.
        """
        super().__init__()
        # vgg_module is already features[:16] (up to relu3_3) from
        # VGGPerceptualLoss -- reuse directly, no separate loading
        self.vgg = vgg_module
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _gram_matrix(self, feat):
        b, c, h, w = feat.shape
        feat = feat.view(b, c, h * w)                     # [B, C, H*W]
        gram = torch.bmm(feat, feat.transpose(1, 2))       # [B, C, C]
        return gram / (c * h * w)                          # normalize by C*H*W

    def forward(self, pred, target):
        pred3 = pred.repeat(1, 3, 1, 1)
        target3 = target.repeat(1, 3, 1, 1)

        pred_norm = (pred3 - self.mean) / self.std
        target_norm = (target3 - self.mean) / self.std

        pred_feat = self.vgg(pred_norm)
        target_feat = self.vgg(target_norm)

        pred_gram = self._gram_matrix(pred_feat)
        target_gram = self._gram_matrix(target_feat)

        return F.mse_loss(pred_gram, target_gram)


def combined_loss_v2(pred, target, vgg_loss_fn, vgg_weight=0.05):
    """Charbonnier (pixel accuracy) + edge (structure) + VGG (perceptual quality).
    vgg_weight kept small since VGG loss values are naturally larger in scale
    than pixel losses -- 0.05 is a reasonable starting point, tunable if needed."""
    pixel = charbonnier_loss(pred, target)
    edge = edge_loss(pred, target)
    perceptual = vgg_loss_fn(pred, target)
    return pixel + 0.1 * edge + vgg_weight * perceptual
