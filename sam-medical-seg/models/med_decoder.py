"""
Medical Segmentation Decoder (U-Net Style)

Fuses multi-scale intermediate features from the SAM image encoder
via skip connections and upsampling to produce a fine-grained
segmentation mask.

Architecture:
    feat3 (768, 64, 64) ────────────────────────────┐
    feat6 (768, 64, 64) ───────────────────┐         │
    feat9 (768, 64, 64) ──────────┐        │         │
    feat12 (768, 64, 64) ──┐      │        │         │
                            │      │        │         │
        1x1→64 → Up×2 → ConvBlock → Concat → ConvBlock
                          → Up×2 → ConvBlock → Concat → ConvBlock
                                    → Up×2 → ConvBlock → Concat → ConvBlock
                                              → Up×4 → 1x1 → Mask

Each ConvBlock = Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double 3×3 convolution block with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpConv(nn.Module):
    """Upsampling + double convolution block."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        scale_factor: int = 2,
    ):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MedDecoder(nn.Module):
    """
    U-Net style decoder for medical image segmentation.

    Fuses 4 intermediate feature maps from the SAM image encoder
    (layers 3, 6, 9, 12) via progressive upsampling and skip connections.

    Input features:
        feat3:  (B, 768, 64, 64)
        feat6:  (B, 768, 64, 64)
        feat9:  (B, 768, 64, 64)
        feat12: (B, 768, 64, 64)  ← deepest

    Output:
        (B, 1, 1024, 1024) segmentation mask (logits)
    """

    def __init__(
        self,
        in_channels: int = 768,
        base_channels: int = 64,
        inter_channels: Optional[List[int]] = None,
        output_size: tuple = (1024, 1024),
    ):
        """
        Args:
            in_channels: Channel dimension of each input feature map.
            base_channels: Base channel count after 1×1 projection.
            inter_channels: Channel counts for skip connections.
                           Default: [768, 768, 768, 768] (no change).
            output_size: Target output size (H, W).
        """
        super().__init__()
        self.in_channels = in_channels
        self.output_size = output_size
        inter_channels = inter_channels or [768, 768, 768, 768]

        # 1×1 projections to reduce channel dimension
        self.proj3 = nn.Conv2d(inter_channels[0], base_channels, kernel_size=1)
        self.proj6 = nn.Conv2d(inter_channels[1], base_channels, kernel_size=1)
        self.proj9 = nn.Conv2d(inter_channels[2], base_channels, kernel_size=1)
        self.proj12 = nn.Conv2d(inter_channels[3], base_channels, kernel_size=1)

        # Decoder path (deepest → shallow)
        # Stage 1: feat12 → up×2 → concat feat9
        self.up12_9 = UpConv(
            in_channels=base_channels,
            skip_channels=base_channels,
            out_channels=base_channels,
            scale_factor=2,
        )

        # Stage 2: → up×2 → concat feat6
        self.up9_6 = UpConv(
            in_channels=base_channels,
            skip_channels=base_channels,
            out_channels=base_channels,
            scale_factor=2,
        )

        # Stage 3: → up×2 → concat feat3
        self.up6_3 = UpConv(
            in_channels=base_channels,
            skip_channels=base_channels,
            out_channels=base_channels,
            scale_factor=2,
        )

        # Final upsampling: 512 → 1024 (2×)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ConvBlock(base_channels, base_channels // 2),
        )

        # Output projection
        self.out_conv = nn.Conv2d(base_channels // 2, 1, kernel_size=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for convolutions."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        intermediate_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass through the medical decoder.

        Args:
            intermediate_features: List of 4 feature tensors from SAM
                layers [3, 6, 9, 12]. Each: (B, C, H_feat, W_feat).

        Returns:
            Segmentation logits: (B, 1, H_out, W_out)

        Raises:
            ValueError: If intermediate_features does not contain exactly 4 tensors.
        """
        if len(intermediate_features) != 4:
            raise ValueError(
                f"Expected 4 intermediate features (layers 3,6,9,12), "
                f"got {len(intermediate_features)}"
            )

        feat3, feat6, feat9, feat12 = intermediate_features

        # Check for missing features
        for name, feat in zip(["layer3", "layer6", "layer9", "layer12"],
                               [feat3, feat6, feat9, feat12]):
            if feat is None:
                raise ValueError(
                    f"Feature '{name}' is None. Ensure SAM backbone hooks are "
                    f"registered and the forward pass completed."
                )

        # 1×1 projections
        p3 = self.proj3(feat3)    # (B, 64, 64, 64)
        p6 = self.proj6(feat6)    # (B, 64, 64, 64)
        p9 = self.proj9(feat9)    # (B, 64, 64, 64)
        p12 = self.proj12(feat12)  # (B, 64, 64, 64)

        # Progressive upsampling with skip connections
        # feat12 (64,64,64) → 128×128 → concat feat9
        x = self.up12_9(p12, p9)
        # → 256×256 → concat feat6
        x = self.up9_6(x, p6)
        # → 512×512 → concat feat3
        x = self.up6_3(x, p3)

        # Final upsampling: 512 → 1024
        x = self.final_up(x)

        # Resize to target output size if needed
        if x.shape[2:] != self.output_size:
            x = F.interpolate(
                x, size=self.output_size, mode="bilinear", align_corners=True
            )

        # Output: single-channel logits
        out = self.out_conv(x)

        return out

    def get_num_params(self) -> int:
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())
