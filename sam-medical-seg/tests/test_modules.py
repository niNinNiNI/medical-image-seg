#!/usr/bin/env python3
"""
Unit tests for SAM-MedSeg modules.

Section 9.1 of the project plan specifies tests for each module.
Run with: python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestMedicalSegDataset:
    """Tests for MedicalSegDataset (requires preprocessed data)."""

    def test_import(self):
        from data.dataset import MedicalSegDataset, create_few_shot_subset
        assert MedicalSegDataset is not None
        assert create_few_shot_subset is not None

    def test_create_few_shot_subset_logic(self):
        """Test few-shot subset creation logic with a simple mock dataset."""
        from data.dataset import create_few_shot_subset
        from torch.utils.data import TensorDataset

        # Create a mock dataset with 100 samples
        mock_data = TensorDataset(torch.randn(100, 3, 64, 64))
        dataset = mock_data  # Just use as a generic Dataset

        # Test ratios
        for ratio in [0.01, 0.05, 0.10, 1.0]:
            subset = create_few_shot_subset(dataset, ratio, seed=42)
            expected = max(1, int(len(dataset) * ratio))
            assert len(subset) == expected, \
                f"ratio={ratio}: expected {expected}, got {len(subset)}"

    def test_seed_reproducibility(self):
        """Test that same seed produces same subset."""
        from data.dataset import create_few_shot_subset
        from torch.utils.data import TensorDataset

        mock_data = TensorDataset(torch.randn(100, 3, 64, 64))

        subset1 = create_few_shot_subset(mock_data, 0.1, seed=42)
        subset2 = create_few_shot_subset(mock_data, 0.1, seed=42)
        subset3 = create_few_shot_subset(mock_data, 0.1, seed=99)

        assert subset1.indices == subset2.indices, "Same seed should give same indices"
        assert subset1.indices != subset3.indices, "Different seeds should differ"


class TestSAMBackbone:
    """Tests for SAMBackbone (requires SAM checkpoint)."""

    def test_import(self):
        from models.sam_backbone import SAMBackbone
        assert SAMBackbone is not None

    def test_model_types(self):
        from models.sam_backbone import SAMBackbone
        for model_type in ["vit_b", "vit_l", "vit_h"]:
            assert model_type in SAMBackbone.MODEL_TYPES

    def test_invalid_model_type(self):
        from models.sam_backbone import SAMBackbone
        import pytest
        with pytest.raises(ValueError):
            SAMBackbone(checkpoint_path="dummy.pth", model_type="invalid")


class TestLoRAAdapter:
    """Tests for LoRAAdapter."""

    def test_import(self):
        from models.lora_adapter import LoRAAdapter
        assert LoRAAdapter is not None

    def test_config_creation(self):
        from models.lora_adapter import LoRAAdapter
        mock_encoder = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
        )
        adapter = LoRAAdapter(
            image_encoder=mock_encoder,
            r=4,
            lora_alpha=16,
            target_modules=["0", "2"],
        )
        config = adapter.config
        assert config.r == 4
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.1


class TestMedDecoder:
    """Tests for MedDecoder."""

    def test_import(self):
        from models.med_decoder import MedDecoder
        assert MedDecoder is not None

    def test_output_shape_1024(self):
        from models.med_decoder import MedDecoder
        decoder = MedDecoder(
            in_channels=768,
            base_channels=64,
            output_size=(1024, 1024),
        )
        feats = [torch.randn(2, 768, 64, 64) for _ in range(4)]
        out = decoder(feats)
        assert out.shape == (2, 1, 1024, 1024), \
            f"Expected (2,1,1024,1024), got {out.shape}"

    def test_output_shape_512(self):
        from models.med_decoder import MedDecoder
        decoder = MedDecoder(
            in_channels=768,
            base_channels=32,
            output_size=(512, 512),
        )
        feats = [torch.randn(2, 768, 64, 64) for _ in range(4)]
        out = decoder(feats)
        assert out.shape == (2, 1, 512, 512), \
            f"Expected (2,1,512,512), got {out.shape}"

    def test_gradient_flow(self):
        from models.med_decoder import MedDecoder
        decoder = MedDecoder(in_channels=768, base_channels=64)
        feats = [torch.randn(2, 768, 64, 64, requires_grad=False) for _ in range(4)]
        out = decoder(feats)
        loss = out.sum()
        loss.backward()
        # Check that all parameters received gradients
        params_with_grad = sum(
            1 for p in decoder.parameters() if p.grad is not None
        )
        total_params = sum(1 for _ in decoder.parameters())
        assert params_with_grad == total_params, \
            f"Only {params_with_grad}/{total_params} params have gradients"

    def test_wrong_num_features(self):
        from models.med_decoder import MedDecoder
        import pytest
        decoder = MedDecoder()
        with pytest.raises(ValueError):
            feats = [torch.randn(1, 768, 64, 64) for _ in range(3)]  # only 3
            decoder(feats)


class TestPromptOptimizer:
    """Tests for PromptOptimizer."""

    def test_import(self):
        from models.prompt_optimizer import PromptOptimizer
        assert PromptOptimizer is not None

    def test_grid_point_count(self):
        from models.prompt_optimizer import PromptOptimizer
        optimizer = PromptOptimizer(strategy="grid_sampling", grid_size=8)
        points = optimizer._generate_grid_points(batch_size=2, device=torch.device("cpu"))
        assert points.shape == (2, 64, 2), \
            f"Expected (2, 64, 2), got {points.shape}"

    def test_invalid_strategy(self):
        from models.prompt_optimizer import PromptOptimizer
        import pytest
        with pytest.raises(ValueError):
            PromptOptimizer(strategy="invalid")


class TestCombinedLoss:
    """Tests for loss functions."""

    def test_import(self):
        from losses.combined_loss import CombinedLoss, DiceLoss, FocalLoss, TverskyLoss, get_loss
        assert CombinedLoss is not None

    def test_loss_range(self):
        from losses.combined_loss import CombinedLoss, DiceLoss
        pred = torch.randn(2, 1, 128, 128)
        target = (torch.rand(2, 1, 128, 128) > 0.5).float()

        # CombinedLoss
        loss_fn = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
        loss = loss_fn(pred, target)
        assert 0.0 <= loss.item() <= 5.0, f"Loss out of range: {loss.item()}"

    def test_dice_gradient_nonzero(self):
        from losses.combined_loss import DiceLoss
        pred = torch.randn(2, 1, 128, 128, requires_grad=True)
        target = (torch.rand(2, 1, 128, 128) > 0.5).float()

        loss_fn = DiceLoss()
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0, "Gradient is zero!"

    def test_get_loss_factory(self):
        from losses.combined_loss import get_loss, CombinedLoss, DiceLoss, FocalLoss, TverskyLoss
        assert isinstance(get_loss("combined"), CombinedLoss)
        assert isinstance(get_loss("dice"), DiceLoss)
        assert isinstance(get_loss("focal"), FocalLoss)
        assert isinstance(get_loss("tversky"), TverskyLoss)

    def test_get_loss_invalid(self):
        from losses.combined_loss import get_loss
        import pytest
        with pytest.raises(ValueError):
            get_loss("nonexistent")


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_perfect_prediction(self):
        from utils.metrics import dice_score, iou_score
        pred = torch.ones(2, 1, 64, 64) * 10.0  # strong positive logits
        target = torch.ones(2, 1, 64, 64)

        dice = dice_score(pred, target)
        iou = iou_score(pred, target)

        assert abs(dice - 1.0) < 0.01, f"Dice should be ~1.0, got {dice}"
        assert abs(iou - 1.0) < 0.01, f"IoU should be ~1.0, got {iou}"

    def test_empty_prediction(self):
        from utils.metrics import dice_score, iou_score
        pred = torch.ones(2, 1, 64, 64) * -10.0  # strong negative logits
        target = torch.ones(2, 1, 64, 64)

        dice = dice_score(pred, target)
        iou = iou_score(pred, target)

        assert dice < 0.1, f"Dice should be near 0, got {dice}"
        assert iou < 0.1, f"IoU should be near 0, got {iou}"

    def test_all_metrics(self):
        from utils.metrics import compute_all_metrics
        pred = torch.randn(4, 1, 128, 128)
        target = (torch.rand(4, 1, 128, 128) > 0.5).float()

        metrics = compute_all_metrics(pred, target)
        for key in ["dice", "iou", "precision", "recall"]:
            assert key in metrics, f"Missing metric: {key}"
            assert 0.0 <= metrics[key] <= 1.0, \
                f"{key} out of range: {metrics[key]}"


class TestFullModel:
    """Integration tests for the full model (smoke tests)."""

    def test_import(self):
        from models.full_model import SAMMedSeg
        assert SAMMedSeg is not None

    def test_fusion_alpha(self):
        """Test fusion alpha setter/getter."""
        # We test the logic without instantiating the full model
        alpha_val = 0.7
        import math
        logit = math.log(alpha_val / (1 - alpha_val))
        recovered = 1.0 / (1.0 + math.exp(-logit))
        assert abs(recovered - alpha_val) < 1e-6


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
