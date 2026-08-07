"""Evaluation helpers for WordWave predictions."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.settings import load_settings


SETTINGS = load_settings()


def evaluate_model_metrics(
    model,
    evaluation_inputs,
    evaluation_labels,
    sample_size: int = SETTINGS.runtime.evaluation_sample_size,
    top_k: int = 5,
):
    """Return top-k accuracy and perplexity on the saved validation split."""

    if (
        evaluation_inputs is None
        or evaluation_labels is None
        or len(evaluation_inputs) == 0
    ):
        return 0.0, float("inf")

    device = next(model.parameters()).device
    dataset = TensorDataset(evaluation_inputs, evaluation_labels)
    sample_count = min(len(dataset), sample_size)
    loader = DataLoader(dataset, batch_size=min(128, sample_count), shuffle=False)

    total_examples = 0
    correct_examples = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
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
        return 0.0, float("inf")

    top_k_accuracy = correct_examples / total_examples
    perplexity = math.exp(total_loss / total_examples)
    return top_k_accuracy, perplexity
