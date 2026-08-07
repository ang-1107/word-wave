"""Load and cache model artifacts used by the app."""

from __future__ import annotations

from pathlib import Path

import torch

from src.model import WordWaveModel
from src.settings import load_settings
from src.tokenizer import Vocabulary, load_vocabulary

SETTINGS = load_settings()


def load_artifacts():
    """Load the trained PyTorch model and tokenizer from disk."""

    model_path = Path(SETTINGS.runtime.model_path)
    tokenizer_path = Path(SETTINGS.runtime.tokenizer_path)

    model_bundle = torch.load(model_path, map_location="cpu")
    vocabulary: Vocabulary = load_vocabulary(tokenizer_path)

    model = WordWaveModel(**model_bundle["model_config"])
    model.load_state_dict(model_bundle["state_dict"])
    model.eval()

    max_len = int(model_bundle["max_len"])
    metrics = dict(model_bundle.get("metrics", {}))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, vocabulary, max_len, metrics, device
