"""Central configuration for WordWave runtime assets and defaults."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "word-wave.keras"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer.pkl"

DEFAULT_SEED_TEXT = "Artificial intelligence is"
DEFAULT_GENERATION_LENGTH = 10
MAX_GENERATION_LENGTH = 20
DEFAULT_BEAM_WIDTH = 5
EVALUATION_SAMPLE_SIZE = 5000
