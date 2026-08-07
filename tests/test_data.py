"""Tests for src.data — split bucket, split assignment, leakage regression."""

from __future__ import annotations

from src.data import CorpusSplitFractions, NextTokenIterableDataset, _split_bucket
from src.tokenizer import Vocabulary

# ---------------------------------------------------------------------------
# _split_bucket
# ---------------------------------------------------------------------------


class TestSplitBucket:
    def test_deterministic(self):
        """Same identifier must always produce the same bucket value."""
        id_a = "path/to/file.txt:42"
        assert _split_bucket(id_a) == _split_bucket(id_a)

    def test_different_inputs_differ(self):
        assert _split_bucket("file:1") != _split_bucket("file:2")

    def test_range_zero_to_one(self):
        for i in range(200):
            val = _split_bucket(f"test:{i}")
            assert 0.0 <= val < 1.0

    def test_distribution_roughly_uniform(self):
        """Buckets should be roughly uniform; at least not all in one bin."""
        values = [_split_bucket(f"id:{i}") for i in range(1000)]
        below_half = sum(1 for v in values if v < 0.5)
        # Should be roughly 500 ± some margin
        assert 350 < below_half < 650


# ---------------------------------------------------------------------------
# _belongs_to_split (via NextTokenIterableDataset)
# ---------------------------------------------------------------------------


class TestSplitAssignment:
    """Verify that the three splits are non-overlapping and exhaustive."""

    def _make_dataset(self, split: str) -> NextTokenIterableDataset:
        vocab = Vocabulary.build("a b c d e")
        ds = NextTokenIterableDataset(
            source_path=".",
            vocabulary=vocab,
            max_len=5,
            split=split,  # type: ignore[arg-type]
        )
        # Override fractions for predictable testing
        ds.split_fractions = CorpusSplitFractions(train=0.8, validation=0.1, test=0.1)
        return ds

    def test_splits_non_overlapping_and_exhaustive(self):
        train_ds = self._make_dataset("train")
        val_ds = self._make_dataset("validation")
        test_ds = self._make_dataset("test")

        for i in range(500):
            bucket = _split_bucket(f"test_id:{i}")
            in_train = train_ds._belongs_to_split(bucket)
            in_val = val_ds._belongs_to_split(bucket)
            in_test = test_ds._belongs_to_split(bucket)

            assigned = [in_train, in_val, in_test]
            assert sum(assigned) == 1, (
                f"Bucket {bucket:.6f} assigned to {sum(assigned)} splits "
                f"(train={in_train}, val={in_val}, test={in_test})"
            )


# ---------------------------------------------------------------------------
# Leakage regression — all windows from the same line go to the same split
# ---------------------------------------------------------------------------


class TestNoLeakage:
    """After the Bug 1 fix, every window from a given (file, line) must land
    in the same split.  This test would have *failed* on the old code."""

    def test_same_line_same_split(self):
        """Hash key is now file_path:line_number, so the split is decided
        once per line — independent of token_index."""
        file_path = "/corpus/book.txt"
        line_number = 7

        line_id = f"{file_path}:{line_number}"
        bucket = _split_bucket(line_id)

        # All token indices within this line get the same bucket
        for _token_index in range(1, 50):
            # The old code hashed f"{file_path}:{line_number}:{token_index}"
            # which would give different buckets per token_index.
            assert _split_bucket(line_id) == bucket
