from pathlib import Path

from src.corpus import is_text_like_file


class TestIsTextLikeFile:
    def test_directory_returns_false(self, tmp_path: Path):
        assert is_text_like_file(tmp_path) is False

    def test_missing_file_returns_false(self, tmp_path: Path):
        missing_file = tmp_path / "does_not_exist.txt"
        assert is_text_like_file(missing_file) is False

    def test_empty_file_returns_true(self, tmp_path: Path):
        empty_file = tmp_path / "empty.txt"
        empty_file.touch()
        assert is_text_like_file(empty_file) is True

    def test_valid_utf8_returns_true(self, tmp_path: Path):
        text_file = tmp_path / "valid.txt"
        text_file.write_text("Hello, world! 🌍", encoding="utf-8")
        assert is_text_like_file(text_file) is True

    def test_null_byte_returns_false(self, tmp_path: Path):
        binary_file = tmp_path / "binary.bin"
        # Write valid text but with a null byte inserted
        binary_file.write_bytes(b"Hello\x00World")
        assert is_text_like_file(binary_file) is False

    def test_invalid_utf8_returns_false(self, tmp_path: Path):
        legacy_file = tmp_path / "legacy.txt"
        # Write a file using windows-1252 encoding with special characters
        # that are not valid UTF-8
        legacy_file.write_bytes("Café".encode("windows-1252"))
        assert is_text_like_file(legacy_file) is False

    def test_large_file_only_reads_first_chunk(self, tmp_path: Path):
        large_file = tmp_path / "large.txt"
        # Write > 1024 bytes of valid text
        large_content = "A" * 2000
        large_file.write_text(large_content, encoding="utf-8")
        assert is_text_like_file(large_file) is True
