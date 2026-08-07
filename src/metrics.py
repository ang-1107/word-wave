"""Evaluation helpers for WordWave predictions."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import top_k_accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.config import EVALUATION_SAMPLE_SIZE


def evaluate_model_metrics(model, tokenizer, max_len, sample_size=EVALUATION_SAMPLE_SIZE):
    """Return top-5 accuracy and perplexity for a subset of tokenizer samples."""

    x_samples = []
    y_true = []

    word_index = tokenizer.word_index

    for word, _ in list(word_index.items())[:sample_size]:
        tokenized = tokenizer.texts_to_sequences([word])[0]
        if len(tokenized) < 2:
            continue

        for i in range(1, len(tokenized)):
            sequence = tokenized[: i + 1]
            padded = pad_sequences([sequence[:-1]], maxlen=max_len, padding="pre")
            if padded.shape[1] == max_len:
                x_samples.append(padded[0])
                y_true.append(sequence[-1])
            if len(x_samples) >= sample_size:
                break
        if len(x_samples) >= sample_size:
            break

    if not x_samples:
        return 0.0, float("inf")

    x_eval = np.array(x_samples)
    y_true = np.array(y_true)

    y_pred_probs = model.predict(x_eval, verbose=0)
    top_5 = top_k_accuracy_score(y_true, y_pred_probs, k=5)

    log_probs = np.log(np.take_along_axis(y_pred_probs, y_true[:, None], axis=1).flatten() + 1e-10)
    perplexity = np.exp(-np.mean(log_probs))

    return top_5, perplexity
