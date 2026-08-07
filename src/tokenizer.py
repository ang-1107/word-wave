"""Vocabulary utilities for WordWave."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from pathlib import Path

import torch


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9']+")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def tokenize_text(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class Vocabulary:
    word_to_idx: dict[str, int]
    idx_to_word: dict[int, str]
    pad_token: str = PAD_TOKEN
    unk_token: str = UNK_TOKEN

    @classmethod
    def build(
        cls,
        text: str,
        max_vocab_size: int | None = None,
        min_freq: int = 1,
    ) -> "Vocabulary":
        counter = Counter(tokenize_text(text))
        ordered_tokens = sorted(counter.items(), key=lambda item: (-item[1], item[0]))

        word_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for word, frequency in ordered_tokens:
            if frequency < min_freq or word in word_to_idx:
                continue
            if max_vocab_size is not None and len(word_to_idx) >= max_vocab_size:
                break
            word_to_idx[word] = len(word_to_idx)

        idx_to_word = {idx: word for word, idx in word_to_idx.items()}
        return cls(word_to_idx=word_to_idx, idx_to_word=idx_to_word)

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "Vocabulary":
        word_to_idx = {
            str(word): int(idx) for word, idx in state["word_to_idx"].items()
        }
        idx_to_word = {
            int(idx): str(word) for idx, word in state["idx_to_word"].items()
        }
        return cls(
            word_to_idx=word_to_idx,
            idx_to_word=idx_to_word,
            pad_token=str(state.get("pad_token", PAD_TOKEN)),
            unk_token=str(state.get("unk_token", UNK_TOKEN)),
        )

    def to_state_dict(self) -> dict[str, object]:
        return {
            "word_to_idx": self.word_to_idx,
            "idx_to_word": self.idx_to_word,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
        }

    def save(self, path: str | Path) -> None:
        torch.save(self.to_state_dict(), Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        state = torch.load(Path(path), map_location="cpu")
        return cls.from_state_dict(state)

    @property
    def pad_idx(self) -> int:
        return self.word_to_idx[self.pad_token]

    @property
    def unk_idx(self) -> int:
        return self.word_to_idx[self.unk_token]

    def __len__(self) -> int:
        return len(self.word_to_idx)

    def encode(self, text: str) -> list[int]:
        return [
            self.word_to_idx.get(token, self.unk_idx) for token in tokenize_text(text)
        ]

    def texts_to_sequences(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]

    def pad_sequence(self, token_ids: list[int], max_len: int) -> list[int]:
        trimmed = token_ids[-max_len:]
        padding = [self.pad_idx] * max(0, max_len - len(trimmed))
        return padding + trimmed

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        words: list[str] = []
        for token_id in token_ids:
            word = self.idx_to_word.get(int(token_id), self.unk_token)
            if skip_special_tokens and word in {self.pad_token, self.unk_token}:
                continue
            words.append(word)
        return " ".join(words)


def load_vocabulary(path: str | Path) -> Vocabulary:
    return Vocabulary.load(path)
