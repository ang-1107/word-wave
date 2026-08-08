"""Tests for src.tokenizer — BPE Vocabulary build, encode, decode, pad, serialization."""

from __future__ import annotations

import pytest

from src.tokenizer import PAD_TOKEN, UNK_TOKEN, Vocabulary, tokenize_text

# ---------------------------------------------------------------------------
# tokenize_text
# ---------------------------------------------------------------------------


class TestTokenizeText:
    def test_lowercases_and_separates_punctuation(self):
        # We changed the tokenization regex to keep punctuation but separate it
        assert tokenize_text("Hello World!") == ["hello", "world", "!"]

    def test_handles_apostrophes(self):
        # We split by non-word characters now, so "it's" -> "it", "'", "s"
        assert tokenize_text("it's") == ["it", "'", "s"]

    def test_empty_string(self):
        assert tokenize_text("") == []

    def test_only_punctuation(self):
        assert tokenize_text("!@#") == ["!@#"]

    def test_numbers_preserved(self):
        assert tokenize_text("year 2024") == ["year", "2024"]


# ---------------------------------------------------------------------------
# Vocabulary.build / build_from_tokens (BPE Logic)
# ---------------------------------------------------------------------------


class TestVocabularyBuild:
    def test_special_tokens_present(self):
        vocab = Vocabulary.build("hello world hello")
        assert vocab.word_to_idx[PAD_TOKEN] == 0
        assert vocab.word_to_idx[UNK_TOKEN] == 1

    def test_vocab_initializes_with_characters(self):
        vocab = Vocabulary.build("abc", max_vocab_size=10)
        assert "a" in vocab.word_to_idx
        assert "b" in vocab.word_to_idx
        assert "c" in vocab.word_to_idx
        assert "</w>" in vocab.word_to_idx

    def test_bpe_merges_frequent_pairs(self):
        # "ab" appears 4 times, "bc" appears 2 times
        # BPE should merge "a" + "b" first if max_vocab_size allows
        vocab = Vocabulary.build("ab ab ab ab abc abc", max_vocab_size=15)
        # Check if the merged token "ab" is in the vocabulary
        assert "ab" in vocab.word_to_idx

    def test_max_vocab_size_respected(self):
        # "hello world" has 10 base tokens (8 unique chars + pad + unk)
        # If we set max_vocab_size=11, it should perform exactly 1 merge and stop.
        vocab = Vocabulary.build("hello world", max_vocab_size=11)
        assert len(vocab) == 11


# ---------------------------------------------------------------------------
# encode / decode
# ---------------------------------------------------------------------------


class TestEncodeDecode:
    @pytest.fixture()
    def vocab(self):
        # A simple corpus where "th" will likely be merged
        return Vocabulary.build("the cat sat on the mat", max_vocab_size=50)

    def test_encode_known_tokens(self, vocab: Vocabulary):
        ids = vocab.encode("the cat")
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) > 0

    def test_encode_unseen_word_breaks_into_subwords(self, vocab: Vocabulary):
        # "that" was not in training. But 't', 'h', 'a', 't' and maybe 'th' are.
        # It shouldn't just be an <unk> token if characters are known.
        ids = vocab.encode("that")
        assert vocab.unk_idx not in ids

    def test_encode_unknown_character_maps_to_unk(self, vocab: Vocabulary):
        # 'z' was not in training corpus
        ids = vocab.encode("z")
        assert vocab.unk_idx in ids

    def test_decode_roundtrip(self, vocab: Vocabulary):
        text = "the cat sat"
        ids = vocab.encode(text)
        decoded = vocab.decode(ids, skip_special_tokens=True)
        assert decoded == text

    def test_decode_skips_specials_by_default(self, vocab: Vocabulary):
        ids = [vocab.pad_idx] + vocab.encode("the")
        decoded = vocab.decode(ids)
        assert PAD_TOKEN not in decoded

    def test_decode_keeps_specials_when_asked(self, vocab: Vocabulary):
        ids = [vocab.pad_idx] + vocab.encode("the")
        decoded = vocab.decode(ids, skip_special_tokens=False)
        assert PAD_TOKEN in decoded


# ---------------------------------------------------------------------------
# pad_sequence
# ---------------------------------------------------------------------------


class TestPadSequence:
    @pytest.fixture()
    def vocab(self):
        return Vocabulary.build("a b c d", max_vocab_size=10)

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


# ---------------------------------------------------------------------------
# state_dict round-trip
# ---------------------------------------------------------------------------


class TestStateDictRoundTrip:
    def test_roundtrip(self):
        original = Vocabulary.build("alpha beta gamma delta", max_vocab_size=50)
        state = original.to_state_dict()
        restored = Vocabulary.from_state_dict(state)

        assert restored.word_to_idx == original.word_to_idx
        assert restored.idx_to_word == original.idx_to_word
        assert restored.merges == original.merges
        assert restored.pad_token == original.pad_token
        assert restored.unk_token == original.unk_token

    def test_save_load_file(self, tmp_path):
        original = Vocabulary.build("alpha beta gamma", max_vocab_size=30)
        path = tmp_path / "vocab.pt"
        original.save(path)
        restored = Vocabulary.load(path)

        assert len(restored) == len(original)
        assert restored.word_to_idx == original.word_to_idx
        assert restored.merges == original.merges
