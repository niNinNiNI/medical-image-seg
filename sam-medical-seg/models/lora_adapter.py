"""
LoRA (Low-Rank Adaptation) Adapter Module

Wraps SAM image encoder with LoRA layers on Attention Q/V projections.
Uses HuggingFace PEFT for standard LoRA implementation.

Key properties:
- Adapts attention qkv projection (fused Q/K/V in SAM ViT)
- Optionally adapts attention output proj and MLP layers
- LoRA rank r=4, alpha=16 (adds ~0.1% parameters)
- Supports merge/unmerge for inference efficiency
"""

from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType


class LoRAAdapter:
    """
    LoRA adapter for SAM image encoder.

    Wraps the image encoder with PEFT LoRA, targeting attention Q/V projections.

    Usage:
        adapter = LoRAAdapter(sam_backbone.image_encoder, config)
        adapted_encoder = adapter.apply()  # returns PEFT-wrapped model
        # ... training ...
        adapter.merge()  # merge LoRA weights into base for fast inference
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        r: int = 4,
        lora_alpha: int = 16,
        target_modules: Optional[list] = None,
        lora_dropout: float = 0.1,
        bias: str = "none",
    ):
        """
        Args:
            image_encoder: SAM image encoder (nn.Module).
            r: LoRA rank (default 4).
            lora_alpha: LoRA scaling factor (default 16).
            target_modules: List of module name patterns to apply LoRA to.
            lora_dropout: LoRA dropout rate.
            bias: Bias handling ("none" / "all" / "lora_only").
        """
        self.base_model = image_encoder
        self.r = r
        self.lora_alpha = lora_alpha
        # SAM ViT uses fused QKV projection (attn.qkv), not separate Q_proj/V_proj
        self.target_modules = target_modules or ["qkv"]
        self.lora_dropout = lora_dropout
        self.bias = bias

        self.lora_config = None
        self.peft_model = None
        self._merged = False

    @property
    def config(self) -> LoraConfig:
        """Build or return cached LoraConfig."""
        if self.lora_config is None:
            self.lora_config = LoraConfig(
                r=self.r,
                lora_alpha=self.lora_alpha,
                target_modules=self.target_modules,
                lora_dropout=self.lora_dropout,
                bias=self.bias,
                task_type=TaskType.FEATURE_EXTRACTION,
            )
        return self.lora_config

    def apply(self) -> nn.Module:
        """
        Apply LoRA to the image encoder.

        Returns:
            PEFT-wrapped model (the adapted image encoder).
        """
        if self.peft_model is not None:
            return self.peft_model

        self.peft_model = get_peft_model(self.base_model, self.config)

        # Count trainable parameters
        total = sum(p.numel() for p in self.peft_model.parameters())
        trainable = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        ratio = 100 * trainable / total if total > 0 else 0
        print(f"  LoRA Adapter: {total:,} total → {trainable:,} trainable ({ratio:.2f}%)")

        return self.peft_model

    def merge(self) -> None:
        """Merge LoRA weights into base model for faster inference."""
        if self.peft_model is None:
            raise RuntimeError("LoRA not applied yet. Call apply() first.")
        if self._merged:
            return  # already merged

        self.peft_model.merge_adapter()
        self._merged = True
        print("  LoRA: weights merged into base model.")

    def unmerge(self) -> None:
        """Unmerge LoRA weights (e.g., for continued training)."""
        if self.peft_model is None:
            raise RuntimeError("LoRA not applied yet.")
        if not self._merged:
            return

        self.peft_model.unmerge_adapter()
        self._merged = False
        print("  LoRA: weights unmerged.")

    def merge_and_unload(self) -> nn.Module:
        """
        Merge LoRA weights and return the base model without PEFT wrapper.
        This eliminates inference latency from the adapter.

        Returns:
            Base image encoder with LoRA weights fused in.
        """
        if self.peft_model is None:
            raise RuntimeError("LoRA not applied yet. Call apply() first.")

        merged_model = self.peft_model.merge_and_unload()
        self._merged = True
        return merged_model

    def get_trainable_params(self) -> int:
        """Return count of trainable (LoRA) parameters."""
        if self.peft_model is None:
            # Estimate from config
            return 0
        return sum(
            p.numel() for p in self.peft_model.parameters() if p.requires_grad
        )

    def get_total_params(self) -> int:
        """Return total parameter count."""
        if self.peft_model is None:
            return sum(p.numel() for p in self.base_model.parameters())
        return sum(p.numel() for p in self.peft_model.parameters())

    def verify_consistency(
        self,
        input_tensor: torch.Tensor,
        tol: float = 1e-5,
    ) -> bool:
        """
        Verify that merge+unmerge produces consistent output.

        Args:
            input_tensor: Sample input for testing.
            tol: Tolerance for output difference.

        Returns:
            True if outputs are consistent within tolerance.
        """
        self.apply()
        self.base_model.eval()

        with torch.no_grad():
            out_before = self.peft_model(input_tensor)

        self.merge()
        with torch.no_grad():
            out_merged = self.peft_model(input_tensor)

        self.unmerge()
        with torch.no_grad():
            out_after = self.peft_model(input_tensor)

        # Check merge consistency
        diff = (out_before - out_after).abs().max().item()
        consistent = diff < tol

        print(f"  LoRA consistency check: max_diff={diff:.2e}, "
              f"passed={consistent}")
        return consistent
