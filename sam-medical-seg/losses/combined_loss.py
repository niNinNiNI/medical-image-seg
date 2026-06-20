"""
Combined Loss Functions for Medical Image Segmentation

Implements DiceLoss + BCEWithLogitsLoss (1:1 weighting) as the default.
Also provides FocalLoss and TverskyLoss as alternatives for class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.

    Dice = 2 * |pred ∩ target| / (|pred| + |target|)
    Loss = 1 - Dice
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Logits (B, 1, H, W) — sigmoid is applied inside.
            target: Binary mask (B, 1, H, W) — values in {0, 1}.

        Returns:
            Scalar loss tensor.
        """
        pred = torch.sigmoid(pred)
        # Flatten spatial dimensions
        pred_flat = pred.view(pred.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1 - dice).mean()


class FocalLoss(nn.Module):
    """
    Focal loss for addressing class imbalance.

    FL = -α * (1 - p_t)^γ * log(p_t)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Logits (B, 1, H, W).
            target: Binary mask (B, 1, H, W).

        Returns:
            Scalar loss tensor.
        """
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pred_sigmoid = torch.sigmoid(pred)

        p_t = target * pred_sigmoid + (1 - target) * (1 - pred_sigmoid)
        focal_weight = (1 - p_t) ** self.gamma

        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        loss = alpha_weight * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class TverskyLoss(nn.Module):
    """
    Tversky loss — generalization of Dice loss.
    Allows weighting of false positives vs false negatives.

    Tversky = TP / (TP + α*FP + β*FN)
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha  # FP weight
        self.beta = beta     # FN weight
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(pred.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)

        tp = (pred_flat * target_flat).sum(dim=1)
        fp = (pred_flat * (1 - target_flat)).sum(dim=1)
        fn = ((1 - pred_flat) * target_flat).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - tversky).mean()


class CombinedLoss(nn.Module):
    """
    Combined Dice + BCE loss for medical image segmentation.

    Loss = dice_weight * DiceLoss + bce_weight * BCEWithLogitsLoss

    Default: 1:1 weighting (dice_weight=0.5, bce_weight=0.5)
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth

        self.dice_loss_fn = DiceLoss(smooth=smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Logits (B, 1, H, W).
            target: Binary mask (B, 1, H, W).

        Returns:
            Scalar combined loss.
        """
        dice = self.dice_loss_fn(pred, target)
        bce = F.binary_cross_entropy_with_logits(pred, target)

        return self.dice_weight * dice + self.bce_weight * bce


def get_loss(loss_type: str = "combined", **kwargs):
    """
    Factory function to create a loss module.

    Args:
        loss_type: One of "combined", "dice", "bce", "focal", "tversky".
        **kwargs: Passed to the loss constructor.

    Returns:
        nn.Module loss function.
    """
    loss_registry = {
        "combined": CombinedLoss,
        "dice": DiceLoss,
        "bce": nn.BCEWithLogitsLoss,
        "focal": FocalLoss,
        "tversky": TverskyLoss,
    }

    if loss_type not in loss_registry:
        raise ValueError(
            f"Unknown loss type '{loss_type}'. "
            f"Choose from {list(loss_registry.keys())}"
        )

    return loss_registry[loss_type](**kwargs)
