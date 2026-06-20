"""
Standard U-Net for Medical Image Segmentation Baseline.

A classic 5-level U-Net with symmetric encoder-decoder and skip connections.
Used as a traditional method baseline for comparison with SAM-MedSeg.

Reference: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
Image Segmentation", MICCAI 2015.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """MaxPool → DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsample → DoubleConv (with skip connection)"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle size mismatch from odd dimensions
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                                     diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net for binary medical image segmentation.

    Architecture:
        Encoder: 5 levels of DoubleConv → MaxPool (3 → 64 → 128 → 256 → 512 → 1024)
        Bottleneck: DoubleConv (1024 → 1024)
        Decoder: 5 levels of Upsample → Concat(skip) → DoubleConv
        Output: 1×1 Conv → single-channel logits

    Args:
        n_channels: Input channels (3 for RGB, 1 for grayscale).
        n_classes: Output classes (1 for binary segmentation).
        bilinear: Use bilinear upsampling instead of transposed conv.
        base_channels: Number of channels after first conv (doubled each level).
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        bilinear: bool = True,
        base_channels: int = 64,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)

        # Decoder
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)

        # Output
        self.outc = nn.Conv2d(base_channels, n_classes, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Input image (B, C, H, W).

        Returns:
            Logits (B, n_classes, H, W).
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output
        logits = self.outc(x)
        return logits


def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    # Quick test
    model = UNet(n_channels=3, n_classes=1, base_channels=64)
    params = count_parameters(model)
    print(f"UNet: {params['total']:,} total params, {params['trainable']:,} trainable")

    x = torch.randn(1, 3, 1024, 1024)
    with torch.no_grad():
        out = model(x)
    print(f"Input:  {x.shape}  →  Output: {out.shape}")
