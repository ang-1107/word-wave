"""Script to build a training corpus from random English Wikipedia articles."""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://en.wikipedia.org/w/api.php?action=query&generator=random&grnnamespace=0&prop=extracts&explaintext=1&format=json"
MAX_FILE_SIZE_BYTES = 64 * 1024 * 1024  # 64 MB chunking limit


def fetch_random_wiki_articles() -> list[str]:
    """Fetches a batch of random Wikipedia articles as plaintext."""
    req = urllib.request.Request(
        API_URL, headers={"User-Agent": "WordWave-Dataset-Builder/1.0"}
    )
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode("utf-8"))
            articles = []
            for page in data.get("query", {}).get("pages", {}).values():
                text = page.get("extract", "").strip()
                if text:
                    articles.append(text)
            return articles
    except urllib.error.HTTPError as e:
        # Wikipedia rate limits API requests. Back off for a bit if we hit it.
        if e.code == 429:
            time.sleep(2.0)
        else:
            time.sleep(1.0)
        return []
    except urllib.error.URLError:
        time.sleep(1.0)
        return []


def build_dataset(target_mb: float, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    target_bytes = int(target_mb * 1024 * 1024)
    total_written = 0
    file_index = 1

    print(f"Building {target_mb} MB Wikipedia corpus in {output_dir}/...")

    current_file_path = output_dir / f"wiki_corpus_part{file_index}.txt"
    current_file = current_file_path.open("w", encoding="utf-8")
    current_file_size = 0

    try:
        while total_written < target_bytes:
            articles = fetch_random_wiki_articles()
            for article in articles:
                # Add a separator between articles
                content = article + "\n\n"
                content_bytes = content.encode("utf-8")
                size = len(content_bytes)

                # Check chunk limit
                if current_file_size + size > MAX_FILE_SIZE_BYTES:
                    current_file.close()
                    file_index += 1
                    current_file_path = output_dir / f"wiki_corpus_part{file_index}.txt"
                    current_file = current_file_path.open("w", encoding="utf-8")
                    current_file_size = 0

                current_file.write(content)
                current_file_size += size
                total_written += size

                if total_written >= target_bytes:
                    break

            # Print progress
            progress = min(100.0, (total_written / target_bytes) * 100)
            print(
                f"\rProgress: {progress:.1f}% ({total_written / (1024 * 1024):.2f} MB / {target_mb:.2f} MB)",
                end="",
                flush=True,
            )

        print("\nDataset built successfully!")
    finally:
        current_file.close()


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
                sys.path.append(str(PROJECT_ROOT))

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
