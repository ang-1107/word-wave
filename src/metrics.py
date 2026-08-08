"""Evaluation helpers for WordWave predictions."""

from __future__ import annotations

import math
from typing import cast

import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def evaluate_model_metrics(
    model,
    loader: DataLoader,
    device: torch.device,
    top_k: int = 5,
):
    """Return top-k accuracy, average loss, and perplexity for a dataloader."""

    total_examples = 0
    correct_examples = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(
            loader, desc="Evaluating metrics", unit="batch", leave=True
        ):
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            total_loss += criterion(logits, labels).item()

            top_k_predictions = torch.topk(
                logits, k=min(top_k, logits.size(-1)), dim=-1
            ).indices
            correct_examples += (
                top_k_predictions.eq(labels.unsqueeze(-1)).any(dim=-1).sum().item()
            )
            total_examples += labels.size(0)

    if total_examples == 0:
        return 0.0, float("inf"), float("inf")

    top_k_accuracy = correct_examples / total_examples
    average_loss = total_loss / total_examples
    perplexity = math.exp(average_loss)
    return top_k_accuracy, average_loss, perplexity


# ---------------------------------------------------------------------------
# Corpus-level generation quality metrics
# ---------------------------------------------------------------------------


def compute_corpus_bleu(
    references: list[str],
    hypotheses: list[str],
) -> float:
    """Compute corpus-level BLEU with Chen & Cherry smoothing (method4).

    Each reference/hypothesis is a whitespace-separated string of tokens.
    """
    ref_tokenized = [[ref.split()] for ref in references]
    hyp_tokenized = [hyp.split() for hyp in hypotheses]
    smoothing = SmoothingFunction().method4
    return float(
        cast(
            float,
            corpus_bleu(ref_tokenized, hyp_tokenized, smoothing_function=smoothing),
        )
    )


def _lcs_length(x: list[str], y: list[str]) -> int:
    """Compute length of the longest common subsequence."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1 between a single reference and hypothesis."""
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs = _lcs_length(ref_tokens, hyp_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_rouge_l_corpus(references: list[str], hypotheses: list[str]) -> float:
    """Average ROUGE-L F1 over a corpus of reference/hypothesis pairs."""
    if not references:
        return 0.0
    scores = [
        compute_rouge_l(ref, hyp)
        for ref, hyp in zip(references, hypotheses, strict=True)
    ]
    return sum(scores) / len(scores)


def compute_distinct_n(texts: list[str], n: int) -> float:
    """Compute distinct-n: ratio of unique n-grams to total n-grams.

    A diversity metric — higher values indicate less repetition.
    """
    total_ngrams = 0
    unique_ngrams: set[tuple[str, ...]] = set()
    for text in texts:
        tokens = text.split()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            unique_ngrams.add(ngram)
            total_ngrams += 1
    if total_ngrams == 0:
        return 0.0
    return len(unique_ngrams) / total_ngrams
