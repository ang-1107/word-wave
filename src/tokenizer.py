"""Vocabulary utilities for WordWave using Byte-Pair Encoding (BPE)."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import torch

# Tokenize words and punctuation separately
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]+")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def tokenize_text(text: str) -> list[str]:
    """Pre-tokenize text into words and punctuation."""
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class Vocabulary:
    word_to_idx: dict[str, int]
    idx_to_word: dict[int, str]
    merges: dict[tuple[str, str], int] = field(default_factory=dict)
    pad_token: str = PAD_TOKEN
    unk_token: str = UNK_TOKEN

    @classmethod
    def build(
        cls,
        text: str,
        max_vocab_size: int = 10000,
        min_freq: int = 1,
    ) -> Vocabulary:
        return cls.build_from_tokens(
            tokenize_text(text), max_vocab_size=max_vocab_size, min_freq=min_freq
        )

    @classmethod
    def build_from_tokens(
        cls,
        tokens: Iterable[str],
        max_vocab_size: int = 10000,
        min_freq: int = 1,
    ) -> Vocabulary:
        word_freqs = Counter(token for token in tokens if token)

        # Initialize word representations with characters + </w> boundary
        splits = {word: list(word) + ["</w>"] for word in word_freqs}

        word_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}

        # Base vocabulary (all unique characters)
        for word in word_freqs:
            for char in splits[word]:
                if char not in word_to_idx:
                    word_to_idx[char] = len(word_to_idx)

        merges: dict[tuple[str, str], int] = {}

        # BPE Training Loop
        while len(word_to_idx) < max_vocab_size:
            pair_freqs: Counter[tuple[str, str]] = Counter()
            for word, freq in word_freqs.items():
                split = splits[word]
                if len(split) < 2:
                    continue
                for i in range(len(split) - 1):
                    pair_freqs[(split[i], split[i + 1])] += freq

            if not pair_freqs:
                break

            best_pair, max_freq = pair_freqs.most_common(1)[0]
            if max_freq < min_freq:
                break

            new_token = best_pair[0] + best_pair[1]
            word_to_idx[new_token] = len(word_to_idx)
            merges[best_pair] = len(merges)

            # Apply merge to all splits
            for word, split in splits.items():
                if len(split) < 2:
                    continue
                new_split: list[str] = []
                i = 0
                while i < len(split):
                    if i < len(split) - 1 and (split[i], split[i + 1]) == best_pair:
                        new_split.append(new_token)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                splits[word] = new_split

        idx_to_word = {idx: token for token, idx in word_to_idx.items()}
        return cls(word_to_idx=word_to_idx, idx_to_word=idx_to_word, merges=merges)

    def _encode_word(self, word: str) -> list[str]:
        """Apply learned BPE merges to a single word."""
        splits = list(word) + ["</w>"]
        while len(splits) > 1:
            pairs = [(splits[i], splits[i + 1]) for i in range(len(splits) - 1)]
            # Find the pair that was merged earliest (lowest rank)
            best_pair = min(pairs, key=lambda pair: self.merges.get(pair, float("inf")))

            if best_pair not in self.merges:
                break

            new_splits: list[str] = []
            i = 0
            while i < len(splits):
                if i < len(splits) - 1 and (splits[i], splits[i + 1]) == best_pair:
                    new_splits.append(splits[i] + splits[i + 1])
                    i += 2
                else:
                    new_splits.append(splits[i])
                    i += 1
            splits = new_splits

        return splits

    def encode(self, text: str) -> list[int]:
        """Encode text using the trained BPE model."""
        words = tokenize_text(text)
        token_ids: list[int] = []
        for word in words:
            subwords = self._encode_word(word)
            for subword in subwords:
                token_ids.append(self.word_to_idx.get(subword, self.unk_idx))
        return token_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode a sequence of BPE token IDs back into a string."""
        tokens: list[str] = []
        for token_id in token_ids:
            token_str = self.idx_to_word.get(int(token_id), self.unk_token)
            if skip_special_tokens and token_str in {self.pad_token, self.unk_token}:
                continue
            tokens.append(token_str)

        # Concatenate all subwords and then replace the word boundary markers with spaces
        raw_text = "".join(tokens)
        decoded_text = raw_text.replace("</w>", " ").strip()
        return decoded_text

    def texts_to_sequences(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]

    def pad_sequence(self, token_ids: list[int], max_len: int) -> list[int]:
        trimmed = token_ids[-max_len:]
        padding = [self.pad_idx] * max(0, max_len - len(trimmed))
        return padding + trimmed

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> Vocabulary:
        word_to_idx_state = cast(dict[object, object], state["word_to_idx"])
        idx_to_word_state = cast(dict[object, object], state["idx_to_word"])

        word_to_idx = {
            str(word): cast(int, idx) for word, idx in word_to_idx_state.items()
        }
        idx_to_word = {
            cast(int, idx): str(word) for idx, word in idx_to_word_state.items()
        }

        # Deserialize merges list back to dictionary
        merges_list = cast(list[tuple[str, str, int]], state.get("merges", []))
        merges = {(a, b): rank for a, b, rank in merges_list}

        return cls(
            word_to_idx=word_to_idx,
            idx_to_word=idx_to_word,
            merges=merges,
            pad_token=str(state.get("pad_token", PAD_TOKEN)),
            unk_token=str(state.get("unk_token", UNK_TOKEN)),
        )

    def to_state_dict(self) -> dict[str, object]:
        # Serialize merges as a list of tuples to avoid tuple-key dictionary issues
        merges_list = [(a, b, rank) for (a, b), rank in self.merges.items()]
        return {
            "word_to_idx": self.word_to_idx,
            "idx_to_word": self.idx_to_word,
            "merges": merges_list,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
        }

    def save(self, path: str | Path) -> None:
        torch.save(self.to_state_dict(), Path(path))

    @classmethod
    def load(cls, path: str | Path) -> Vocabulary:
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


def load_vocabulary(path: str | Path) -> Vocabulary:
    return Vocabulary.load(path)
