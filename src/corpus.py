"""Corpus discovery and streaming helpers for WordWave training."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from src.settings import load_settings

SETTINGS = load_settings()
DEFAULT_ALLOWED_EXTENSIONS = tuple(SETTINGS.runtime.allowed_extensions)


def is_text_like_file(
    path: Path, allowed_extensions: Iterable[str] | None = None
) -> bool:
    extensions = tuple(allowed_extensions or DEFAULT_ALLOWED_EXTENSIONS)
    return path.is_file() and path.suffix.lower() in extensions


def iter_source_files(
    source_path: str | Path,
    allowed_extensions: Iterable[str] | None = None,
) -> Iterator[Path]:
    path = Path(source_path)
    if path.is_file():
        if is_text_like_file(path, allowed_extensions):
            yield path
        return

    if not path.is_dir():
        raise FileNotFoundError(f"Training source path not found: {path}")

    for child_path in sorted(path.rglob("*")):
        if is_text_like_file(child_path, allowed_extensions):
            yield child_path


def iter_source_lines(
    source_path: str | Path,
    allowed_extensions: Iterable[str] | None = None,
) -> Iterator[tuple[Path, int, str]]:
    for file_path in iter_source_files(source_path, allowed_extensions):
        try:
            with file_path.open("r", encoding="utf-8", errors="ignore") as file_handle:
                for line_number, line in enumerate(file_handle, start=1):
                    cleaned_line = line.strip()
                    if cleaned_line:
                        yield file_path, line_number, cleaned_line
        except OSError:
            continue
