"""
SAM Backbone Loader

Loads pretrained SAM ViT-B model with layer-wise freeze control.
Exposes intermediate features via forward hooks for the medical decoder.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from segment_anything import sam_model_registry
from segment_anything.modeling import Sam, ImageEncoderViT, MaskDecoder, PromptEncoder


class SAMBackbone(nn.Module):
    """
    Load pretrained SAM model with layer-wise freezing and feature extraction.

    Freeze strategy:
      - Default: freeze first 9 transformer layers of the image encoder
      - Last 3 layers + mask decoder are trainable
      - Prompt encoder is always frozen

    Forward hooks on layers [3, 6, 9, 12] extract intermediate features
    for the U-Net style medical decoder.
    """

    # Supported SAM model types
    MODEL_TYPES = {
        "vit_b": "sam_vit_b_01ec64.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_h": "sam_vit_h_4b8939.pth",
    }

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "vit_b",
        freeze_layers: int = 9,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            checkpoint_path: Path to SAM checkpoint file.
            model_type: SAM model variant (vit_b / vit_l / vit_h).
            freeze_layers: Number of transformer layers to freeze (0-11).
            device: Target device (auto-detect if None).
        """
        super().__init__()

        if model_type not in self.MODEL_TYPES:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Choose from {list(self.MODEL_TYPES.keys())}"
            )

        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.freeze_layers = freeze_layers
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load SAM model
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found at {checkpoint_path}. "
                f"Download from https://github.com/facebookresearch/segment-anything"
            )

        self.sam: Sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        self.sam.to(self.device)

        # Expose components
        self.image_encoder: ImageEncoderViT = self.sam.image_encoder
        self.prompt_encoder: PromptEncoder = self.sam.prompt_encoder
        self.mask_decoder: MaskDecoder = self.sam.mask_decoder

        # Apply freeze strategy
        self._apply_freeze()

        # Register hooks for intermediate features
        self._intermediate_features: Dict[int, torch.Tensor] = {}
        self._hooks = []
        self._register_hooks()

    def _apply_freeze(self) -> None:
        """Apply layer-wise freezing to the image encoder."""
        # Prompt encoder: always frozen
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        # Image encoder blocks
        if hasattr(self.image_encoder, "blocks"):
            blocks = self.image_encoder.blocks
        elif hasattr(self.image_encoder, "layers"):
            blocks = self.image_encoder.layers
        else:
            warnings.warn("Cannot identify transformer blocks in image encoder.")
            return

        for i, block in enumerate(blocks):
            if i < self.freeze_layers:
                for param in block.parameters():
                    param.requires_grad = False

        # Mask decoder: always trainable
        for param in self.mask_decoder.parameters():
            param.requires_grad = True

        # Count parameters
        total = sum(p.numel() for p in self.sam.parameters())
        trainable = sum(p.numel() for p in self.sam.parameters() if p.requires_grad)
        print(f"  SAM Backbone: {total:,} total params, "
              f"{trainable:,} trainable ({100*trainable/total:.1f}%)")

    def _register_hooks(self) -> None:
        """Register forward hooks on specified transformer layers."""
        target_layers = [3, 6, 9, 12]  # 1-indexed layer numbers

        if hasattr(self.image_encoder, "blocks"):
            blocks = self.image_encoder.blocks
        elif hasattr(self.image_encoder, "layers"):
            blocks = self.image_encoder.layers
        else:
            return

        def make_hook(layer_idx: int):
            def hook_fn(module, input, output):
                self._intermediate_features[layer_idx] = output
            return hook_fn

        for layer_num in target_layers:
            idx = layer_num - 1  # 0-indexed
            if idx < len(blocks):
                hook = blocks[idx].register_forward_hook(make_hook(layer_num))
                self._hooks.append(hook)

    def train(self, mode: bool = True) -> "SAMBackbone":
        """
        Override train() to ensure frozen layers remain in eval mode.
        """
        super().train(mode)

        if mode:
            # Force frozen blocks to eval mode
            if hasattr(self.image_encoder, "blocks"):
                blocks = self.image_encoder.blocks
            elif hasattr(self.image_encoder, "layers"):
                blocks = self.image_encoder.layers
            else:
                return self

            for i, block in enumerate(blocks):
                if i < self.freeze_layers:
                    block.eval()

            # Prompt encoder always in eval
            self.prompt_encoder.eval()

        return self

    def forward(
        self,
        image: torch.Tensor,
        return_intermediate: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the SAM image encoder.

        Args:
            image: Input image tensor (B, 3, H, W), normalized to [0, 1].
            return_intermediate: Whether to return intermediate features.

        Returns:
            Dict with keys:
                "image_embeddings": (B, 256, 64, 64) SAM image embeddings
                "intermediate_features": list of [feat3, feat6, feat9, feat12]
                    each of shape (B, 768, 64, 64)
        """
        # SAM expects pixel values in a specific range — normalize to [0, 255]
        # and apply SAM's preprocessing (mean/std normalization inside forward)
        if image.max() <= 1.0:
            image = image * 255.0

        # Pad to 1024x1024 if needed (SAM requires specific input size)
        _, _, h, w = image.shape
        if h != 1024 or w != 1024:
            image = nn.functional.interpolate(
                image, size=(1024, 1024), mode="bilinear", align_corners=False
            )

        # Forward through image encoder
        image_embeddings = self.image_encoder(image)

        result = {
            "image_embeddings": image_embeddings,
        }

        if return_intermediate:
            # Hook outputs are in (B, H, W, C) spatial form.
            # Convert to (B, C, H, W) for the medical decoder.
            intermediate = []
            for layer_num in [3, 6, 9, 12]:
                feat = self._intermediate_features.get(layer_num)
                if feat is not None:
                    # (B, H, W, C) → (B, C, H, W)
                    feat = feat.permute(0, 3, 1, 2).contiguous()
                intermediate.append(feat)
            result["intermediate_features"] = intermediate

        return result

    def enable_gradient_checkpointing(self) -> None:
        """
        Wrap each transformer block with gradient checkpointing to reduce
        memory during training. Essential for 1024×1024 resolution on 8GB GPU.

        Checkpointing trades ~30% more forward computation for ~60% less
        activation memory by not storing intermediate activations and
        recomputing them during backward.
        """
        import torch.utils.checkpoint as cp

        class CheckpointBlock(nn.Module):
            """Wrapper that applies checkpoint to a block's forward."""
            def __init__(self, block):
                super().__init__()
                self.block = block

            def forward(self, x):
                return cp.checkpoint(self.block, x, use_reentrant=False)

        blocks = getattr(self.image_encoder, "blocks", None)
        if blocks is not None:
            for i, block in enumerate(blocks):
                if not isinstance(block, CheckpointBlock):
                    blocks[i] = CheckpointBlock(block)
            print(f"  Gradient Checkpointing: enabled on {len(blocks)} blocks")

    def get_prompt_encoder(self) -> PromptEncoder:
        """Return the SAM prompt encoder."""
        return self.prompt_encoder

    def get_mask_decoder(self) -> MaskDecoder:
        """Return the SAM mask decoder."""
        return self.mask_decoder

    def remove_hooks(self) -> None:
        """Clean up forward hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._intermediate_features.clear()
