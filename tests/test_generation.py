"""Tests for src.generation — sampling, top-p filtering, beam search, BLEU."""

from __future__ import annotations

import pytest
import torch

from src.generation import (
    _apply_no_repeat_ngram,
    _apply_repetition_penalty,
    _filter_logits_with_top_k,
    _filter_logits_with_top_p,
    beam_search_decoder,
    evaluate_bleu,
    sample_decoder,
    sample_next_token,
)
from src.model import WordWaveModel
from src.tokenizer import Vocabulary

# ---------------------------------------------------------------------------
# sample_next_token
# ---------------------------------------------------------------------------


class TestSampleNextToken:
    def test_raises_on_zero_temperature(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="temperature"):
            sample_next_token(logits, temperature=0.0)

    def test_raises_on_negative_temperature(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="temperature"):
            sample_next_token(logits, temperature=-1.0)

    def test_returns_valid_index(self):
        logits = torch.tensor([1.0, 2.0, 3.0, 0.5])
        idx = sample_next_token(logits, temperature=1.0)
        assert 0 <= idx < 4

    def test_deterministic_with_dominant_logit(self):
        """A very dominant logit should almost always be selected."""
        logits = torch.tensor([-100.0, -100.0, 100.0, -100.0])
        results = {sample_next_token(logits, temperature=0.01) for _ in range(20)}
        assert results == {2}


# ---------------------------------------------------------------------------
# _filter_logits_with_top_p
# ---------------------------------------------------------------------------


class TestFilterLogitsWithTopP:
    def test_raises_on_invalid_top_p(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="top_p"):
            _filter_logits_with_top_p(logits, top_p=0.0)
        with pytest.raises(ValueError, match="top_p"):
            _filter_logits_with_top_p(logits, top_p=1.5)

    def test_top_p_1_keeps_all(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        filtered = _filter_logits_with_top_p(logits, top_p=1.0)
        # With top_p=1.0, nothing should be filtered out
        assert (filtered > float("-inf")).all()

    def test_top_p_filters_tail(self):
        # Create logits where one token dominates
        logits = torch.tensor([10.0, -5.0, -5.0, -5.0])
        filtered = _filter_logits_with_top_p(logits, top_p=0.5)
        # The dominant token (idx 0) should remain; others may be -inf
        assert filtered[0] > float("-inf")
        # At least one of the others should be filtered
        neg_inf_count = (filtered == float("-inf")).sum().item()
        assert neg_inf_count >= 1

    def test_first_token_always_kept(self):
        logits = torch.tensor([10.0, 5.0, 2.0, 1.0])
        # Even with top_p very small, the best token is kept to prevent NaNs
        filtered = _filter_logits_with_top_p(logits, top_p=0.0001)
        assert filtered[0].item() == 10.0
        assert torch.isneginf(filtered[1:]).all()


class TestFilterLogitsWithTopK:
    def test_top_k_keeps_top_k(self):
        logits = torch.tensor([10.0, 5.0, 2.0, 1.0])
        filtered = _filter_logits_with_top_k(logits, top_k=2)
        assert filtered[0].item() == 10.0
        assert filtered[1].item() == 5.0
        assert torch.isneginf(filtered[2])
        assert torch.isneginf(filtered[3])

    def test_top_k_zero_keeps_all(self):
        logits = torch.tensor([10.0, 5.0, 2.0, 1.0])
        filtered = _filter_logits_with_top_k(logits, top_k=0)
        assert torch.allclose(filtered, logits)

    def test_top_k_larger_than_vocab(self):
        logits = torch.tensor([10.0, 5.0, 2.0, 1.0])
        filtered = _filter_logits_with_top_k(logits, top_k=10)
        assert torch.allclose(filtered, logits)


class TestApplyRepetitionPenalty:
    def test_penalty_one_no_op(self):
        logits = torch.tensor([10.0, 5.0, -2.0, 1.0])
        filtered = _apply_repetition_penalty(logits.clone(), [1, 2], penalty=1.0)
        assert torch.allclose(filtered, logits)

    def test_penalizes_positive_and_negative_logits(self):
        logits = torch.tensor([10.0, 5.0, -2.0, 1.0])
        filtered = _apply_repetition_penalty(logits.clone(), [1, 2], penalty=2.0)
        # Token 1: 5.0 / 2.0 = 2.5
        # Token 2: -2.0 * 2.0 = -4.0
        assert filtered[1].item() == 2.5
        assert filtered[2].item() == -4.0
        assert filtered[0].item() == 10.0  # untouched
        assert filtered[3].item() == 1.0  # untouched


class TestApplyNoRepeatNgram:
    def test_no_op_if_n_zero(self):
        logits = torch.tensor([10.0, 5.0, 2.0, 1.0])
        filtered = _apply_no_repeat_ngram(logits.clone(), [1, 2, 3], n=0)
        assert torch.allclose(filtered, logits)

    def test_no_op_if_sequence_too_short(self):
        logits = torch.tensor([10.0, 5.0, 2.0, 1.0])
        # n=3 needs at least 2 tokens of context
        filtered = _apply_no_repeat_ngram(logits.clone(), [1], n=3)
        assert torch.allclose(filtered, logits)

    def test_masks_repeated_bigram(self):
        logits = torch.tensor([10.0, 5.0, 2.0, 1.0])
        # Sequence: [0, 1, 2, 1]
        # Current context: [1] (n=2 means context is 1 token)
        # Occurrences of [1]: index 1 -> followed by 2
        # Occurrences of [1]: index 3 -> end of sequence
        # So token 2 should be masked
        filtered = _apply_no_repeat_ngram(logits.clone(), [0, 1, 2, 1], n=2)
        assert torch.isneginf(filtered[2])
        assert not torch.isneginf(filtered[1])


# ---------------------------------------------------------------------------
# beam_search_decoder (tiny model)
# ---------------------------------------------------------------------------


class TestBeamSearchDecoder:
    @pytest.fixture()
    def tiny_setup(self):
        vocab = Vocabulary.build("the cat sat on a mat")
        model = WordWaveModel(vocab_size=len(vocab), embedding_dim=8, hidden_dim=16)
        model.eval()
        max_len = 5
        return model, vocab, max_len

    def test_returns_string(self, tiny_setup):
        model, vocab, max_len = tiny_setup
        result = beam_search_decoder(
            model, vocab, "the", beam_width=3, next_words=3, max_len=max_len
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_starts_with_seed(self, tiny_setup):
        model, vocab, max_len = tiny_setup
        result = beam_search_decoder(
            model, vocab, "the cat", beam_width=3, next_words=2, max_len=max_len
        )
        assert result.startswith("the cat")

    def test_beam_width_1(self, tiny_setup):
        """Beam width 1 should behave like greedy search."""
        model, vocab, max_len = tiny_setup
        result = beam_search_decoder(
            model, vocab, "the", beam_width=1, next_words=2, max_len=max_len
        )
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# sample_decoder (tiny model)
# ---------------------------------------------------------------------------


class TestSampleDecoder:
    @pytest.fixture()
    def tiny_setup(self):
        vocab = Vocabulary.build("the cat sat on a mat")
        model = WordWaveModel(vocab_size=len(vocab), embedding_dim=8, hidden_dim=16)
        model.eval()
        max_len = 5
        return model, vocab, max_len

    def test_returns_string(self, tiny_setup):
        model, vocab, max_len = tiny_setup
        result = sample_decoder(
            model,
            vocab,
            "the",
            next_words=3,
            max_len=max_len,
            temperature=1.0,
            top_p=0.9,
        )
        assert isinstance(result, str)

    def test_starts_with_seed(self, tiny_setup):
        model, vocab, max_len = tiny_setup
        result = sample_decoder(
            model,
            vocab,
            "cat sat",
            next_words=2,
            max_len=max_len,
        )
        assert result.startswith("cat sat")


# ---------------------------------------------------------------------------
# evaluate_bleu
# ---------------------------------------------------------------------------


class TestEvaluateBleu:
    def test_identical_sentences(self):
        score = evaluate_bleu("the cat sat", "the cat sat")
        assert score > 0.9

    def test_completely_different(self):
        score = evaluate_bleu("the cat sat", "dogs run fast")
        assert score < 0.1

    def test_partial_overlap(self):
        score = evaluate_bleu("the cat sat on the mat", "the cat on a mat")
        assert 0.0 < score < 1.0

    def test_empty_candidate(self):
        score = evaluate_bleu("the cat sat", "")
        assert score == 0.0
