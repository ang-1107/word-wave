"""Corpus discovery and streaming helpers for WordWave training."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from src.settings import load_settings

SETTINGS = load_settings()


def is_text_like_file(path: Path) -> bool:
    """Dynamically determine if a file is plaintext using a combined heuristic."""
    if not path.is_file():
        return False

    try:
        with path.open("rb") as f:
            chunk = f.read(1024)

        if not chunk:
            return True  # Empty files are technically text

        # 1. Git Heuristic: If we find a NULL byte, it's definitively binary.
        if b"\x00" in chunk:
            return False

        # 2. Strict UTF-8 Check: Ensure the chunk decodes purely.
        # This prevents silently corrupting the vocabulary with latin-1
        # or binary files that coincidentally lack NULL bytes in their first KB.
        chunk.decode("utf-8", errors="strict")
        return True
    except (OSError, UnicodeDecodeError):
        return False


def iter_source_files(source_path: str | Path) -> Iterator[Path]:
    """Recursively yield all dynamically detected plaintext files."""
    path = Path(source_path)
    if path.is_file():
        if is_text_like_file(path):
            yield path
        return

    if not path.is_dir():
        raise FileNotFoundError(f"Training source path not found: {path}")

    for child_path in sorted(path.rglob("*")):
        if is_text_like_file(child_path):
            yield child_path


def iter_source_lines(source_path: str | Path) -> Iterator[tuple[Path, int, str]]:
    """Yield all non-empty lines from all plaintext files in the source path."""
    for file_path in iter_source_files(source_path):
        try:
            with file_path.open("r", encoding="utf-8", errors="ignore") as file_handle:
                for line_number, line in enumerate(file_handle, start=1):
                    cleaned_line = line.strip()
                    if cleaned_line:
                        yield file_path, line_number, cleaned_line
        except OSError:
            continue
