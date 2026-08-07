"""Text generation helpers for WordWave."""

from __future__ import annotations

import math

import torch
from nltk.translate.bleu_score import sentence_bleu

from src.tokenizer import Vocabulary


def beam_search_decoder(
    model,
    vocabulary: Vocabulary,
    seed_text: str,
    beam_width: int = 5,
    next_words: int = 10,
    max_len: int = 50,
):
    sequences = [(seed_text.strip(), 0.0)]
    device = next(model.parameters()).device

    for _ in range(next_words):
        all_candidates = []
        for sequence, score in sequences:
            token_ids = vocabulary.encode(sequence)
            padded_ids = vocabulary.pad_sequence(token_ids, max_len)
            input_ids = torch.tensor([padded_ids], dtype=torch.long, device=device)

            with torch.no_grad():
                logits = model(input_ids)[0]
                probabilities = torch.softmax(logits, dim=-1)

            top_k = min(beam_width, probabilities.numel())
            top_probabilities, top_indices = torch.topk(probabilities, k=top_k)

            for probability, idx in zip(top_probabilities.tolist(), top_indices.tolist()):
                word = vocabulary.idx_to_word.get(int(idx), "")
                if word in {"", vocabulary.pad_token, vocabulary.unk_token}:
                    continue
                candidate_sequence = f"{sequence} {word}".strip()
                candidate_score = score - math.log(probability + 1e-10)
                candidate = (candidate_sequence, candidate_score)
                all_candidates.append(candidate)

        if not all_candidates:
            break

        sequences = sorted(all_candidates, key=lambda item: item[1])[:beam_width]

    return sequences[0][0] if sequences else seed_text.strip()


def evaluate_bleu(reference_sentence, generated_sentence):
    reference = [reference_sentence.split()]
    candidate = generated_sentence.split()
    return sentence_bleu(reference, candidate, weights=(0.5, 0.5))
