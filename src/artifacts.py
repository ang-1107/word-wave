"""Load and cache model artifacts used by the app."""

from __future__ import annotations

import pickle

from tensorflow.keras.models import load_model

from src.config import MODEL_PATH, TOKENIZER_PATH


def load_artifacts():
    """Load the trained model and tokenizer from disk."""

    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as file_handle:
        tokenizer = pickle.load(file_handle)
    max_len = model.input_shape[1]
    vocab_size = len(tokenizer.word_index) + 1
    return model, tokenizer, max_len, vocab_size
