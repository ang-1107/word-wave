"""Tests for src.tokenizer — Vocabulary build, encode, decode, pad, serialization."""

from __future__ import annotations

import pytest

from src.tokenizer import PAD_TOKEN, UNK_TOKEN, Vocabulary, tokenize_text

# ---------------------------------------------------------------------------
# tokenize_text
# ---------------------------------------------------------------------------


class TestTokenizeText:
    def test_lowercases(self):
        assert tokenize_text("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self):
        assert tokenize_text("it's a test, right?") == ["it's", "a", "test", "right"]

    def test_empty_string(self):
        assert tokenize_text("") == []

    def test_only_punctuation(self):
        assert tokenize_text("!@#$%^&*()") == []

    def test_numbers_preserved(self):
        assert tokenize_text("year 2024") == ["year", "2024"]


# ---------------------------------------------------------------------------
# Vocabulary.build / build_from_tokens
# ---------------------------------------------------------------------------


class TestVocabularyBuild:
    def test_special_tokens_present(self):
        vocab = Vocabulary.build("hello world hello")
        assert vocab.word_to_idx[PAD_TOKEN] == 0
        assert vocab.word_to_idx[UNK_TOKEN] == 1

    def test_vocab_size_includes_specials(self):
        vocab = Vocabulary.build("a b c")
        # 3 words + <pad> + <unk>
        assert len(vocab) == 5

    def test_max_vocab_size(self):
        vocab = Vocabulary.build("a b c d e f", max_vocab_size=4)
        # 4 total = <pad> + <unk> + 2 real words
        assert len(vocab) == 4

    def test_min_freq_filters(self):
        vocab = Vocabulary.build("a a a b b c", min_freq=2)
        assert "a" in vocab.word_to_idx
        assert "b" in vocab.word_to_idx
        assert "c" not in vocab.word_to_idx

    def test_ordering_by_frequency(self):
        vocab = Vocabulary.build("rare common common common")
        # "common" should get a lower index than "rare"
        assert vocab.word_to_idx["common"] < vocab.word_to_idx["rare"]


# ---------------------------------------------------------------------------
# encode / decode
# ---------------------------------------------------------------------------


class TestEncodeDecode:
    @pytest.fixture()
    def vocab(self):
        return Vocabulary.build("the cat sat on the mat")

    def test_encode_known_tokens(self, vocab: Vocabulary):
        ids = vocab.encode("the cat")
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) == 2

    def test_encode_unknown_maps_to_unk(self, vocab: Vocabulary):
        ids = vocab.encode("the elephant")
        assert ids[1] == vocab.unk_idx

    def test_decode_roundtrip(self, vocab: Vocabulary):
        text = "the cat sat"
        ids = vocab.encode(text)
        decoded = vocab.decode(ids, skip_special_tokens=True)
        assert decoded == text

    def test_decode_skips_specials_by_default(self, vocab: Vocabulary):
        ids = [vocab.pad_idx, vocab.word_to_idx["the"]]
        decoded = vocab.decode(ids)
        assert PAD_TOKEN not in decoded

    def test_decode_keeps_specials_when_asked(self, vocab: Vocabulary):
        ids = [vocab.pad_idx, vocab.word_to_idx["the"]]
        decoded = vocab.decode(ids, skip_special_tokens=False)
        assert PAD_TOKEN in decoded


# ---------------------------------------------------------------------------
# pad_sequence
# ---------------------------------------------------------------------------


class TestPadSequence:
    @pytest.fixture()
    def vocab(self):
        return Vocabulary.build("a b c d")

    def test_left_pads_short_sequence(self, vocab: Vocabulary):
        ids = [2, 3]
        padded = vocab.pad_sequence(ids, max_len=5)
        assert len(padded) == 5
        assert padded[:3] == [vocab.pad_idx] * 3
        assert padded[3:] == ids

    def test_trims_long_sequence_keeps_last(self, vocab: Vocabulary):
        ids = [2, 3, 4, 5, 6]
        padded = vocab.pad_sequence(ids, max_len=3)
        assert len(padded) == 3
        assert padded == [4, 5, 6]

    def test_exact_length_unchanged(self, vocab: Vocabulary):
        ids = [2, 3, 4]
        padded = vocab.pad_sequence(ids, max_len=3)
        assert padded == ids

    def test_empty_list(self, vocab: Vocabulary):
        padded = vocab.pad_sequence([], max_len=3)
        assert padded == [vocab.pad_idx] * 3


# ---------------------------------------------------------------------------
# state_dict round-trip
# ---------------------------------------------------------------------------


class TestStateDictRoundTrip:
    def test_roundtrip(self):
        original = Vocabulary.build("alpha beta gamma delta")
        state = original.to_state_dict()
        restored = Vocabulary.from_state_dict(state)

        assert restored.word_to_idx == original.word_to_idx
        assert restored.idx_to_word == original.idx_to_word
        assert restored.pad_token == original.pad_token
        assert restored.unk_token == original.unk_token

    def test_save_load_file(self, tmp_path):
        original = Vocabulary.build("alpha beta gamma")
        path = tmp_path / "vocab.pt"
        original.save(path)
        restored = Vocabulary.load(path)

        assert len(restored) == len(original)
        assert restored.word_to_idx == original.word_to_idx
