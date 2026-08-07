"""Dataset preparation utilities for WordWave training."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm

from src.corpus import iter_source_files
from src.settings import load_settings
from src.tokenizer import Vocabulary, tokenize_text

SETTINGS = load_settings()
SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class CorpusSplitFractions:
    train: float
    validation: float
    test: float


def get_corpus_split_fractions() -> CorpusSplitFractions:
    validation_fraction = float(SETTINGS.training.validation_fraction)
    test_fraction = float(SETTINGS.training.test_fraction)
    train_fraction = 1.0 - validation_fraction - test_fraction

    if train_fraction <= 0.0:
        raise ValueError("Training fraction must remain greater than zero.")

    return CorpusSplitFractions(
        train=train_fraction,
        validation=validation_fraction,
        test=test_fraction,
    )


def _split_bucket(identifier: str) -> float:
    digest = hashlib.blake2b(identifier.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64)


def _iter_source_lines_with_progress(
    source_files: list[Path], desc: str
) -> Iterator[tuple[Path, int, str]]:
    for file_path in tqdm(source_files, desc=desc, unit="file", leave=True):
        try:
            with file_path.open("r", encoding="utf-8", errors="ignore") as file_handle:
                for line_number, line in enumerate(file_handle, start=1):
                    cleaned_line = line.strip()
                    if cleaned_line:
                        yield file_path, line_number, cleaned_line
        except OSError:
            continue


def build_vocabulary_from_source(
    source_path: str,
    max_vocab_size: int | None = None,
    min_freq: int = 1,
) -> Vocabulary:
    source_files = list(
        iter_source_files(source_path, SETTINGS.runtime.allowed_extensions)
    )

    def token_stream() -> Iterator[str]:
        for _, _, line in _iter_source_lines_with_progress(
            source_files, "Building vocabulary"
        ):
            yield from tokenize_text(line)

    return Vocabulary.build_from_tokens(
        token_stream(), max_vocab_size=max_vocab_size, min_freq=min_freq
    )


class NextTokenIterableDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        source_path: str,
        vocabulary: Vocabulary,
        max_len: int,
        split: SplitName,
    ) -> None:
        super().__init__()
        self.source_path = source_path
        self.vocabulary = vocabulary
        self.max_len = max_len
        self.split = split
        self.split_fractions = get_corpus_split_fractions()

    def _belongs_to_split(self, bucket: float) -> bool:
        if self.split == "train":
            return bucket < self.split_fractions.train
        if self.split == "validation":
            return (
                self.split_fractions.train
                <= bucket
                < (self.split_fractions.train + self.split_fractions.validation)
            )
        return bucket >= (self.split_fractions.train + self.split_fractions.validation)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        source_files = list(
            iter_source_files(self.source_path, SETTINGS.runtime.allowed_extensions)
        )
        for file_path, line_number, line in _iter_source_lines_with_progress(
            source_files, f"{self.split.title()} dataset"
        ):
            token_ids = self.vocabulary.encode(line)
            if len(token_ids) < 2:
                continue

            line_id = f"{file_path}:{line_number}"
            if not self._belongs_to_split(_split_bucket(line_id)):
                continue

            for token_index in range(1, len(token_ids)):
                window = token_ids[max(0, token_index - self.max_len) : token_index]
                padded_window = self.vocabulary.pad_sequence(window, self.max_len)
                yield (
                    torch.tensor(padded_window, dtype=torch.long),
                    torch.tensor(token_ids[token_index], dtype=torch.long),
                )


def build_streaming_dataloaders(
    source_path: str,
    vocabulary: Vocabulary,
    max_len: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = NextTokenIterableDataset(source_path, vocabulary, max_len, "train")
    validation_dataset = NextTokenIterableDataset(
        source_path, vocabulary, max_len, "validation"
    )
    test_dataset = NextTokenIterableDataset(source_path, vocabulary, max_len, "test")

    return (
        DataLoader(train_dataset, batch_size=batch_size),
        DataLoader(validation_dataset, batch_size=batch_size),
        DataLoader(test_dataset, batch_size=batch_size),
    )
