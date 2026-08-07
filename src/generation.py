"""Text generation helpers for WordWave."""

from __future__ import annotations

import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from tensorflow.keras.preprocessing.sequence import pad_sequences


def beam_search_decoder(model, tokenizer, seed_text, beam_width=5, next_words=10, max_len=50):
    sequences = [(seed_text, 0.0)]

    for _ in range(next_words):
        all_candidates = []
        for sequence, score in sequences:
            tokenized = tokenizer.texts_to_sequences([sequence])[0]
            tokenized = pad_sequences([tokenized], maxlen=max_len, padding="pre")
            predictions = model.predict(tokenized, verbose=0)[0]
            top_indices = np.argsort(predictions)[-beam_width:]

            for idx in top_indices:
                word = tokenizer.index_word.get(idx, "")
                if not word:
                    continue
                candidate = (sequence + " " + word, score - np.log(predictions[idx] + 1e-10))
                all_candidates.append(candidate)

        if not all_candidates:
            break

        sequences = sorted(all_candidates, key=lambda item: item[1])[:beam_width]

    return sequences[0][0]


def evaluate_bleu(reference_sentence, generated_sentence):
    reference = [reference_sentence.split()]
    candidate = generated_sentence.split()
    return sentence_bleu(reference, candidate, weights=(0.5, 0.5))
