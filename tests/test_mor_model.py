"""
Test suite for MoR model components
"""

import os

# Import MoR components
import sys
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.advanced_mor import AdvancedMoRModel, LearnedThresholdRouter
from src.models.mor_model import MoRConfig, MoRModel, RecursiveTransformerLayer


class TestMoRConfig:
    """Test MoR configuration"""

    def test_default_config(self):
        config = MoRConfig()
        assert config.vocab_size == 50257
        assert config.hidden_size == 256
        assert config.num_attention_heads == 8
        assert config.max_recursion_depth == 3

    def test_custom_config(self):
        config = MoRConfig(
            hidden_size=512, num_attention_heads=16, max_recursion_depth=4
        )
        assert config.hidden_size == 512
        assert config.num_attention_heads == 16
        assert config.max_recursion_depth == 4


class TestRecursiveTransformerLayer:
    """Test recursive transformer layer"""

    @pytest.fixture
    def config(self):
        return MoRConfig(hidden_size=256, num_attention_heads=8)

    @pytest.fixture
    def layer(self, config):
        return RecursiveTransformerLayer(config)

    def test_layer_initialization(self, layer, config):
        assert isinstance(layer.attention, nn.MultiheadAttention)
        assert isinstance(layer.feed_forward, nn.Sequential)
        assert isinstance(layer.norm1, nn.LayerNorm)
        assert isinstance(layer.norm2, nn.LayerNorm)

    def test_forward_pass(self, layer):
        batch_size, seq_len, hidden_size = 2, 10, 256
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        output = layer(hidden_states)

        assert output.shape == (batch_size, seq_len, hidden_size)
        assert not torch.isnan(output).any()

    def test_forward_with_mask(self, layer):
        batch_size, seq_len, hidden_size = 2, 10, 256
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = layer(hidden_states, attention_mask=attention_mask)

        assert output.shape == (batch_size, seq_len, hidden_size)
        assert not torch.isnan(output).any()


class TestMoRModel:
    """Test complete MoR model"""

    @pytest.fixture
    def config(self):
        return MoRConfig(
            vocab_size=1000,  # Smaller for testing
            hidden_size=128,
            num_attention_heads=4,
            max_recursion_depth=3,
            num_recursive_layers=2,
        )

    @pytest.fixture
    def model(self, config):
        return MoRModel(config)

    def test_model_initialization(self, model, config):
        assert len(model.recursive_layers) == config.num_recursive_layers
        assert model.router is not None
        assert model.embedding is not None
        assert model.output_layer is not None

    def test_forward_pass(self, model):
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        outputs = model(input_ids)

        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, 1000)
        assert "recursion_depths" in outputs
        assert not torch.isnan(outputs["logits"]).any()

    @pytest.mark.slow
    def test_training_step(self, model):
        """Test that model can perform a training step"""
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        labels = torch.randint(0, 1000, (batch_size, seq_len))

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        outputs = model(input_ids)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs["logits"].view(-1, 1000), labels.view(-1))

        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert not torch.isnan(loss).any()


class TestLearnedThresholdRouter:
    """Test advanced router with learned thresholds"""

    @pytest.fixture
    def config(self):
        return MoRConfig(
            hidden_size=128, max_recursion_depth=3, use_learned_thresholds=True
        )

    @pytest.fixture
    def router(self, config):
        return LearnedThresholdRouter(config)

    def test_router_initialization(self, router, config):
        assert router.hidden_size == config.hidden_size
        assert router.max_depth == config.max_recursion_depth
        assert router.threshold_net is not None

    def test_router_forward(self, router):
        batch_size, seq_len, hidden_size = 2, 8, 128
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        depths = router(hidden_states, training=False)

        assert depths.shape == (batch_size, seq_len)
        assert depths.min() >= 1
        assert depths.max() <= 3
        assert depths.dtype == torch.long

    def test_router_training_mode(self, router):
        batch_size, seq_len, hidden_size = 2, 8, 128
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        depths = router(hidden_states, training=True)

        assert depths.shape == (batch_size, seq_len)
        assert depths.dtype == torch.long


class TestAdvancedMoRModel:
    """Test advanced MoR model with all features"""

    @pytest.fixture
    def config(self):
        return MoRConfig(
            vocab_size=1000,
            hidden_size=128,
            num_attention_heads=4,
            max_recursion_depth=3,
            use_learned_thresholds=True,
            use_efficiency_routing=True,
            use_multi_scale_attention=True,
        )

    @pytest.fixture
    def model(self, config):
        return AdvancedMoRModel(config)

    def test_advanced_model_initialization(self, model, config):
        assert model.config.use_learned_thresholds
        assert model.config.use_efficiency_routing
        assert model.config.use_multi_scale_attention

    def test_advanced_forward_pass(self, model):
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        outputs = model(input_ids)

        assert "logits" in outputs
        assert "recursion_depths" in outputs
        assert "efficiency_scores" in outputs
        assert not torch.isnan(outputs["logits"]).any()


@pytest.mark.integration
class TestMoRIntegration:
    """Integration tests for complete MoR pipeline"""

    def test_model_save_load(self, tmp_path):
        """Test model serialization"""
        config = MoRConfig(vocab_size=1000, hidden_size=64)
        model = MoRModel(config)

        # Save model
        save_path = tmp_path / "test_model.pth"
        torch.save(
            {"model_state_dict": model.state_dict(), "config": config.__dict__},
            save_path,
        )

        # Load model
        checkpoint = torch.load(save_path)
        new_config = MoRConfig(**checkpoint["config"])
        new_model = MoRModel(new_config)
        new_model.load_state_dict(checkpoint["model_state_dict"])

        # Test that loaded model works
        input_ids = torch.randint(0, 1000, (1, 5))
        outputs = new_model(input_ids)
        assert "logits" in outputs

    @pytest.mark.slow
    def test_efficiency_comparison(self):
        """Test that MoR is more efficient than standard transformer"""
        # This would require implementing a standard transformer baseline
        # For now, just test that MoR produces reasonable efficiency metrics
        config = MoRConfig(vocab_size=1000, hidden_size=128)
        model = MoRModel(config)

        input_ids = torch.randint(0, 1000, (4, 16))
        outputs = model(input_ids)

        depths = outputs["recursion_depths"]
        avg_depth = depths.float().mean()

        # Average depth should be less than max depth (showing efficiency)
        assert avg_depth < config.max_recursion_depth
        assert avg_depth >= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
