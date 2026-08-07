"""Evaluation helpers for WordWave predictions."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.data import DataLoader


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
        return 0.0, float("inf"), float("inf")

    top_k_accuracy = correct_examples / total_examples
    average_loss = total_loss / total_examples
    perplexity = math.exp(average_loss)
    return top_k_accuracy, average_loss, perplexity
