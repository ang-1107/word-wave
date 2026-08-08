"""Script to build a training corpus using the Wikitext-103 dataset."""

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

# URL for Wikitext-103 from fast.ai mirror
API_URL = "https://s3.amazonaws.com/fast-ai-nlp/wikitext-103.tgz"
MAX_FILE_SIZE_BYTES = 64 * 1024 * 1024  # 64 MB chunking limit


def reporthook(count, block_size, total_size):
    """Progress bar for urlretrieve."""
    if total_size > 0:
        progress = int(count * block_size * 100 / total_size)
        progress = min(progress, 100)
        sys.stdout.write(
            f"\rDownloading Wikitext-103 archive: {progress}% ({total_size / (1024 * 1024):.1f} MB)"
        )
        sys.stdout.flush()


def download_archive(output_path: Path) -> bool:
    """Downloads the Wikitext-103 archive."""
    try:
        urllib.request.urlretrieve(API_URL, str(output_path), reporthook)
        print("\nDownload complete.")
        return True
    except Exception as e:
        print(f"\nError downloading Wikipedia data: {e}", file=sys.stderr)
        return False


def build_dataset(target_mb: float, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    target_bytes = int(target_mb * 1024 * 1024)
    archive_path = output_dir / "wikitext-103.tgz"

    if not download_archive(archive_path):
        return

    print(f"Extracting up to {target_mb} MB of text into {output_dir}/...")

    total_written = 0
    file_index = 1

    current_file_path = output_dir / f"wiki_corpus_part{file_index}.txt"
    current_file = current_file_path.open("wb")
    current_file_size = 0

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            # Find the main training text file in the archive
            member = None
            for m in tar.getmembers():
                if "wiki.train.tokens" in m.name:
                    member = m
                    break

            if not member:
                print(
                    "Error: Could not find 'wiki.train.tokens' in archive.",
                    file=sys.stderr,
                )
                return

            f = tar.extractfile(member)
            if f is None:
                print("Error: Could not extract file.", file=sys.stderr)
                return

            chunk_size = 1024 * 1024  # 1 MB chunks for reading

            while total_written < target_bytes:
                bytes_to_read = min(chunk_size, target_bytes - total_written)
                data = f.read(bytes_to_read)
                if not data:
                    break  # End of file reached

                bytes_to_write = len(data)

                # Check chunk limit (split to a new file if this write would exceed max file size)
                if current_file_size + bytes_to_write > MAX_FILE_SIZE_BYTES:
                    current_file.close()
                    file_index += 1
                    current_file_path = output_dir / f"wiki_corpus_part{file_index}.txt"
                    current_file = current_file_path.open("wb")
                    current_file_size = 0

                current_file.write(data)
                current_file_size += bytes_to_write
                total_written += bytes_to_write

                progress = min(100.0, (total_written / target_bytes) * 100)
                sys.stdout.write(
                    f"\rProgress: {progress:.1f}% ({total_written / (1024 * 1024):.2f} MB / {target_mb:.2f} MB)"
                )
                sys.stdout.flush()

        print("\nDataset built successfully!")
    finally:
        current_file.close()
        # Clean up the downloaded archive
        if archive_path.exists():
            archive_path.unlink()
            print("Cleaned up temporary archive.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Wikipedia dataset.")
    parser.add_argument(
        "--size-mb", type=float, help="Target size of the dataset in megabytes"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory to save the dataset files",
    )
    args = parser.parse_args()

    # Load defaults from config if not provided
    if args.size_mb is None:
        try:
            # Add project root to path so we can import src.settings
            PROJECT_ROOT = Path(__file__).resolve().parent.parent
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            from src.settings import load_settings

            settings = load_settings()
            target_mb = float(settings.runtime.default_corpus_size_mb)
        except Exception as e:
            print(
                f"Warning: Could not load config.yaml ({e}). Defaulting to 64 MB.",
                file=sys.stderr,
            )
            target_mb = 64.0
    else:
        target_mb = args.size_mb

    build_dataset(target_mb, args.output_dir)


if __name__ == "__main__":
    main()
