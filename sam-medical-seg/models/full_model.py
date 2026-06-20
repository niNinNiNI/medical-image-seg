"""
Full Model Integration: SAM-MedSeg

Combines all four modules into an end-to-end trainable model:
  1. SAMBackbone — pretrained SAM ViT-B with layer-wise freezing
  2. LoRAAdapter — low-rank adaptation on attention Q/V projections
  3. PromptOptimizer — automatic prompt point generation
  4. MedDecoder — U-Net style skip-connection decoder

Fusion: final_mask = α * SAM_mask + (1-α) * MedDecoder_mask
where α is a learnable parameter (initialized at 0.5).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sam_backbone import SAMBackbone
from .lora_adapter import LoRAAdapter
from .prompt_optimizer import PromptOptimizer
from .med_decoder import MedDecoder


class SAMMedSeg(nn.Module):
    """
    SAM-MedSeg: Few-shot medical image segmentation with improved SAM.

    Combines LoRA-adapted SAM with automatic prompt optimization and
    a medical skip-connection decoder via learnable fusion.
    """

    def __init__(
        self,
        sam_checkpoint: str,
        model_type: str = "vit_b",
        freeze_layers: int = 9,
        lora_r: int = 4,
        lora_alpha: int = 16,
        lora_target_modules: Optional[List[str]] = None,
        lora_dropout: float = 0.1,
        prompt_strategy: str = "grid_sampling",
        grid_size: int = 32,
        iou_threshold: float = 0.5,
        med_decoder_base_channels: int = 64,
        fusion_alpha: float = 0.5,
        image_size: int = 1024,
        use_checkpoint: bool = False,
        use_lora: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            sam_checkpoint: Path to SAM checkpoint file.
            model_type: SAM variant (vit_b / vit_l / vit_h).
            freeze_layers: Number of transformer layers to freeze.
            lora_r: LoRA rank.
            lora_alpha: LoRA scaling factor.
            lora_target_modules: Which attention projections to adapt.
            lora_dropout: LoRA dropout rate.
            prompt_strategy: "grid_sampling" or "coarse_seg".
            grid_size: N for N×N grid points.
            iou_threshold: IoU threshold for prompt filtering.
            med_decoder_base_channels: Base channels for medical decoder.
            fusion_alpha: Initial fusion weight (learnable parameter).
            image_size: Input image size (square assumed).
            use_checkpoint: Enable gradient checkpointing for memory efficiency.
            use_lora: Apply LoRA adapter (set False for med_decoder_only ablation).
            device: Target device.
        """
        super().__init__()

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.image_size = image_size

        # 1. SAM Backbone
        print("[SAM-MedSeg] Loading SAM backbone...")
        self.backbone = SAMBackbone(
            checkpoint_path=sam_checkpoint,
            model_type=model_type,
            freeze_layers=freeze_layers,
            device=self.device,
        )

        # 2. LoRA Adapter
        if use_lora:
            print("[SAM-MedSeg] Applying LoRA adapter...")
            self.lora_adapter = LoRAAdapter(
                image_encoder=self.backbone.image_encoder,
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules or ["qkv", "proj"],
                lora_dropout=lora_dropout,
            )
            # Apply LoRA (replaces self.backbone.image_encoder with PEFT model)
            self.backbone.sam.image_encoder = self.lora_adapter.apply()
        else:
            print("[SAM-MedSeg] Skipping LoRA adapter (use_lora=False).")
            self.lora_adapter = None

        # Enable gradient checkpointing for high-res training
        if use_checkpoint:
            print("[SAM-MedSeg] Enabling gradient checkpointing...")
            self.backbone.enable_gradient_checkpointing()

        # 3. Prompt Optimizer
        print("[SAM-MedSeg] Initializing prompt optimizer...")
        self.prompt_optimizer = PromptOptimizer(
            strategy=prompt_strategy,
            grid_size=grid_size,
            iou_threshold=iou_threshold,
            image_size=image_size,
        )

        # 4. Medical Decoder
        print("[SAM-MedSeg] Initializing medical decoder...")
        self.med_decoder = MedDecoder(
            in_channels=768,  # ViT-B embedding dim
            base_channels=med_decoder_base_channels,
            output_size=(image_size, image_size),
        )

        # 5. Learnable fusion parameter
        self.fusion_alpha = nn.Parameter(torch.tensor(fusion_alpha))

        self.to(self.device)

        # Count total parameters
        self._log_param_count()

    def _log_param_count(self) -> None:
        """Log total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"[SAM-MedSeg] Total: {total:,} params | "
              f"Trainable: {trainable:,} ({100*trainable/total:.2f}%) | "
              f"Frozen: {frozen:,}")

    def forward(
        self,
        image: torch.Tensor,
        gt_mask: Optional[torch.Tensor] = None,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            image: Input image (B, 3, H, W), normalized to [0, 1].
            gt_mask: Ground truth mask (optional, for training).
            return_all: If True, return intermediate outputs.

        Returns:
            Dict with keys:
                "final_mask": (B, 1, H, W) fused segmentation mask (logits)
                "sam_mask": (B, 1, H, W) SAM decoder output (if return_all)
                "med_mask": (B, 1, H, W) Medical decoder output (if return_all)
                "fusion_alpha": float (current fusion weight)
                "prompts": dict (if return_all)
        """
        B = image.shape[0]

        # 1. Encode image through SAM backbone
        features = self.backbone(image, return_intermediate=True)
        image_embeddings = features["image_embeddings"]
        inter_features = features.get("intermediate_features", [])

        # 2. Generate optimized prompts
        if gt_mask is not None and self.training:
            # Training: could use ground truth to sample oracle prompts
            # For now, still use automatic prompts
            prompts = self.prompt_optimizer(
                self.backbone, image, image_embeddings, gt_mask
            )
        else:
            with torch.no_grad():
                prompts = self.prompt_optimizer(
                    self.backbone, image, image_embeddings
                )

        # 3. SAM mask decoder path
        # NOTE: SAM's mask_decoder is NOT designed for multi-image batching.
        # It treats the batch dim as "multiple prompt sets for ONE image".
        # We must process each sample individually, then concatenate.
        points = prompts["points"]
        labels = prompts["labels"]

        image_pe = self.backbone.sam.prompt_encoder.get_dense_pe()

        sam_masks_list = []
        for b_idx in range(B):
            # Prepare per-sample sparse embeddings (1, N, 2) / (1, N)
            single_pts = points[b_idx:b_idx+1]
            single_labels = labels[b_idx:b_idx+1]

            sparse_emb, dense_emb = self.backbone.sam.prompt_encoder(
                points=(single_pts, single_labels),
                boxes=None,
                masks=None,
            )

            single_mask, _ = self.backbone.sam.mask_decoder(
                image_embeddings=image_embeddings[b_idx:b_idx+1],
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            sam_masks_list.append(single_mask)

        sam_masks = torch.cat(sam_masks_list, dim=0)

        # Resize SAM output from 256×256 to target size
        sam_mask = F.interpolate(
            sam_masks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        # 4. Medical decoder path
        if inter_features and all(f is not None for f in inter_features):
            med_mask = self.med_decoder(inter_features)
        else:
            # Fallback: if intermediate features unavailable, use zeros
            med_mask = torch.zeros(
                B, 1, self.image_size, self.image_size,
                device=image.device
            )

        # 5. Fusion
        alpha = torch.sigmoid(self.fusion_alpha)  # constrain to [0, 1]
        final_mask = alpha * sam_mask + (1 - alpha) * med_mask

        result = {
            "final_mask": final_mask,
        }

        if return_all:
            result.update({
                "sam_mask": sam_mask,
                "med_mask": med_mask,
                "fusion_alpha": alpha.item(),
                "prompts": prompts,
            })

        return result

    def set_fusion_alpha(self, alpha: float) -> None:
        """Manually set the fusion weight (for ablation studies)."""
        # Convert to logit space
        import math
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be in [0, 1]")
        # Handle edge cases: alpha=0 → logit=-10, alpha=1 → logit=+10
        if alpha <= 0.0:
            logit = -10.0
        elif alpha >= 1.0:
            logit = 10.0
        else:
            logit = math.log(alpha / (1 - alpha))
        self.fusion_alpha.data.fill_(logit)

    def get_fusion_alpha(self) -> float:
        """Get current fusion weight as a float in [0, 1]."""
        return torch.sigmoid(self.fusion_alpha).item()

    def merge_lora(self) -> None:
        """Merge LoRA weights for inference (eliminates adapter overhead)."""
        if self.lora_adapter is not None:
            self.lora_adapter.merge()

    def unmerge_lora(self) -> None:
        """Unmerge LoRA weights for continued training."""
        if self.lora_adapter is not None:
            self.lora_adapter.unmerge()

    def cleanup(self) -> None:
        """Clean up hooks and temporary resources."""
        self.backbone.remove_hooks()
