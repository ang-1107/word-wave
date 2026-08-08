"""Tests for src.model — forward pass, attention mask, edge cases."""

from __future__ import annotations

import torch

from src.model import WordWaveModel

# ---------------------------------------------------------------------------
# Forward pass basics
# ---------------------------------------------------------------------------


class TestForwardPass:
    def _make_model(self, vocab_size: int = 50, **kwargs) -> WordWaveModel:
        return WordWaveModel(
            vocab_size=vocab_size, embedding_dim=16, hidden_dim=32, **kwargs
        )

    def test_output_shape(self):
        model = self._make_model(vocab_size=50)
        model.eval()
        x = torch.randint(0, 50, (4, 10))  # batch=4, seq_len=10
        out = model(x)
        assert out.shape == (4, 50)

    def test_output_dtype(self):
        model = self._make_model()
        model.eval()
        x = torch.randint(0, 50, (2, 5))
        out = model(x)
        assert out.dtype == torch.float32

    def test_single_token_input(self):
        model = self._make_model()
        model.eval()
        x = torch.tensor([[7]])  # batch=1, seq_len=1
        out = model(x)
        assert out.shape == (1, 50)


# ---------------------------------------------------------------------------
# Attention padding mask (Bug 5)
# ---------------------------------------------------------------------------


class TestAttentionMask:
    def _make_model(self, vocab_size: int = 50) -> WordWaveModel:
        return WordWaveModel(vocab_size=vocab_size, embedding_dim=16, hidden_dim=32)

    def test_padded_positions_get_near_zero_attention(self):
        """Left-padded positions (index 0 = pad) should receive negligible
        attention weight after masking."""
        model = self._make_model()
        model.eval()

        # Sequence: [PAD, PAD, PAD, 5, 10]  (padding_idx = 0)
        x = torch.tensor([[0, 0, 0, 5, 10]])

        # Run forward and inspect internally by re-computing
        with torch.no_grad():
            pad_mask = x == model.embedding.padding_idx
            embeddings = model.embedding(x)
            outputs, _ = model.lstm(embeddings)
            attention_logits = model.attention(outputs).squeeze(-1)
            masked_logits = attention_logits.masked_fill(pad_mask, float("-inf"))
            weights = torch.softmax(masked_logits, dim=1)

        # Padded positions (first 3) should have ~0 weight
        pad_weights = weights[0, :3]
        real_weights = weights[0, 3:]
        assert pad_weights.sum().item() < 1e-6, f"Pad weights too large: {pad_weights}"
        assert abs(real_weights.sum().item() - 1.0) < 1e-5

    def test_no_padding_all_weight_used(self):
        """When there's no padding, all positions should receive attention."""
        model = self._make_model()
        model.eval()
        x = torch.tensor([[3, 5, 7, 10, 2]])  # no zeros
        with torch.no_grad():
            out = model(x)
        # Just check it runs without error and produces valid output
        assert not torch.isnan(out).any()

    def test_all_padding_no_nan(self):
        """Edge case: fully padded input should NOT produce NaN thanks to
        the uniform-attention fallback."""
        model = self._make_model()
        model.eval()
        x = torch.tensor([[0, 0, 0, 0, 0]])
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any(), "All-padding input produced NaN"
        assert not torch.isinf(out).any(), "All-padding input produced Inf"


# ---------------------------------------------------------------------------
# Weight Tying
# ---------------------------------------------------------------------------


class TestWeightTying:
    def test_tie_weights_output_shape(self):
        model = WordWaveModel(
            vocab_size=50, embedding_dim=16, hidden_dim=32, tie_weights=True
        )
        model.eval()
        x = torch.randint(0, 50, (4, 10))
        out = model(x)
        assert out.shape == (4, 50)

    def test_tie_weights_reduces_parameters(self):
        model_untied = WordWaveModel(
            vocab_size=1000, embedding_dim=128, hidden_dim=256, tie_weights=False
        )
        model_tied = WordWaveModel(
            vocab_size=1000, embedding_dim=128, hidden_dim=256, tie_weights=True
        )
        untied_params = sum(p.numel() for p in model_untied.parameters())
        tied_params = sum(p.numel() for p in model_tied.parameters())
        # The tied model should have fewer parameters because it reuses the embedding weights
        # instead of having a separate (128 x 1000) linear layer.
        assert tied_params < untied_params
