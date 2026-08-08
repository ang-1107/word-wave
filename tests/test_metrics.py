"""Tests for src.metrics — corpus BLEU, ROUGE-L, distinct-n, evaluate_model_metrics."""

from __future__ import annotations

import torch

from src.metrics import (
    _lcs_length,
    compute_corpus_bleu,
    compute_distinct_n,
    compute_rouge_l,
    compute_rouge_l_corpus,
    evaluate_model_metrics,
)
from src.model import WordWaveModel

# ---------------------------------------------------------------------------
# _lcs_length
# ---------------------------------------------------------------------------


class TestLcsLength:
    def test_identical(self):
        assert _lcs_length(["a", "b", "c"], ["a", "b", "c"]) == 3

    def test_no_overlap(self):
        assert _lcs_length(["a", "b"], ["c", "d"]) == 0

    def test_partial_overlap(self):
        assert _lcs_length(["a", "b", "c", "d"], ["a", "c", "d"]) == 3

    def test_empty(self):
        assert _lcs_length([], ["a", "b"]) == 0
        assert _lcs_length(["a"], []) == 0


# ---------------------------------------------------------------------------
# compute_rouge_l
# ---------------------------------------------------------------------------


class TestRougeL:
    def test_identical(self):
        score = compute_rouge_l("the cat sat", "the cat sat")
        assert abs(score - 1.0) < 1e-6

    def test_no_overlap(self):
        score = compute_rouge_l("the cat", "dogs run")
        assert score == 0.0

    def test_partial_overlap(self):
        score = compute_rouge_l("the cat sat on the mat", "the cat on a mat")
        assert 0.0 < score < 1.0

    def test_empty_hypothesis(self):
        assert compute_rouge_l("the cat", "") == 0.0

    def test_empty_reference(self):
        assert compute_rouge_l("", "the cat") == 0.0


class TestRougeLCorpus:
    def test_average(self):
        refs = ["the cat sat", "dogs run fast"]
        hyps = ["the cat sat", "dogs run fast"]
        score = compute_rouge_l_corpus(refs, hyps)
        assert abs(score - 1.0) < 1e-6

    def test_empty_corpus(self):
        assert compute_rouge_l_corpus([], []) == 0.0


# ---------------------------------------------------------------------------
# compute_corpus_bleu
# ---------------------------------------------------------------------------


class TestCorpusBleu:
    def test_identical_corpus(self):
        refs = ["the cat sat on the mat", "dogs run in the park"]
        hyps = ["the cat sat on the mat", "dogs run in the park"]
        score = compute_corpus_bleu(refs, hyps)
        assert score > 0.9

    def test_no_overlap(self):
        refs = ["alpha beta gamma"]
        hyps = ["delta epsilon zeta"]
        score = compute_corpus_bleu(refs, hyps)
        # With smoothing, this may not be exactly 0 but should be very low
        assert score < 0.2

    def test_empty_hypothesis(self):
        refs = ["the cat sat"]
        hyps = [""]
        score = compute_corpus_bleu(refs, hyps)
        assert score < 0.1


# ---------------------------------------------------------------------------
# compute_distinct_n
# ---------------------------------------------------------------------------


class TestDistinctN:
    def test_all_unique_unigrams(self):
        texts = ["the cat sat on a mat"]
        score = compute_distinct_n(texts, n=1)
        assert abs(score - 1.0) < 1e-6  # 6 unique / 6 total

    def test_all_repeated(self):
        texts = ["the the the the"]
        score = compute_distinct_n(texts, n=1)
        assert abs(score - 0.25) < 1e-6  # 1 unique / 4 total

    def test_bigrams(self):
        texts = ["a b a b a b"]
        # bigrams: (a,b), (b,a), (a,b), (b,a), (a,b) → 2 unique / 5 total
        score = compute_distinct_n(texts, n=2)
        assert abs(score - 0.4) < 1e-6

    def test_multiple_texts(self):
        texts = ["hello world", "hello there"]
        # unigrams: hello, world, hello, there → 3 unique / 4 total
        score = compute_distinct_n(texts, n=1)
        assert abs(score - 0.75) < 1e-6

    def test_empty_text(self):
        assert compute_distinct_n([""], n=1) == 0.0
        assert compute_distinct_n([], n=1) == 0.0

    def test_single_token(self):
        # single token means 0 bigrams
        assert compute_distinct_n(["hello"], n=2) == 0.0


# ---------------------------------------------------------------------------
# evaluate_model_metrics (integration-style test with a tiny model)
# ---------------------------------------------------------------------------


class TestEvaluateModelMetrics:
    def test_returns_correct_shape(self):
        from torch.utils.data import DataLoader, TensorDataset

        vocab_size = 20
        model = WordWaveModel(vocab_size=vocab_size, embedding_dim=8, hidden_dim=16)
        model.eval()

        inputs = torch.randint(0, vocab_size, (16, 5))
        labels = torch.randint(0, vocab_size, (16,))
        dataset = TensorDataset(inputs, labels)
        loader = DataLoader(dataset, batch_size=8)

        top_k_acc, avg_loss, perplexity = evaluate_model_metrics(
            model, loader, torch.device("cpu"), top_k=5
        )
        assert 0.0 <= top_k_acc <= 1.0
        assert avg_loss > 0.0
        assert perplexity > 1.0
