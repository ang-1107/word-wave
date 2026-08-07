"""Dataset preparation utilities for WordWave training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.tokenizer import Vocabulary


@dataclass
class SequenceDataset:
    inputs: torch.Tensor
    labels: torch.Tensor


def build_training_sequences(token_ids: list[int], max_len: int, pad_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    inputs: list[list[int]] = []
    labels: list[int] = []

    for index in range(1, len(token_ids)):
        window = token_ids[max(0, index - max_len):index]
        padded = [pad_idx] * max(0, max_len - len(window)) + window[-max_len:]
        inputs.append(padded)
        labels.append(token_ids[index])

    if not inputs:
        raise ValueError("The training corpus is too small to build next-word sequences.")

    return torch.tensor(inputs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def split_train_validation(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[SequenceDataset, SequenceDataset]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(inputs), generator=generator)
    shuffled_inputs = inputs[indices]
    shuffled_labels = labels[indices]

    validation_size = max(1, int(len(shuffled_inputs) * validation_fraction))
    train_size = max(1, len(shuffled_inputs) - validation_size)

    train_dataset = SequenceDataset(
        inputs=shuffled_inputs[:train_size],
        labels=shuffled_labels[:train_size],
    )
    validation_dataset = SequenceDataset(
        inputs=shuffled_inputs[train_size:],
        labels=shuffled_labels[train_size:],
    )
    return train_dataset, validation_dataset


def build_vocabulary(text: str, max_vocab_size: int | None = None, min_freq: int = 1) -> Vocabulary:
    return Vocabulary.build(text, max_vocab_size=max_vocab_size, min_freq=min_freq)
