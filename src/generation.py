"""Text generation helpers for WordWave."""

from __future__ import annotations

import math

import torch
from nltk.translate.bleu_score import sentence_bleu

from src.tokenizer import Vocabulary


def _prepare_input(
    vocabulary: Vocabulary, sequence: str, max_len: int, device
) -> torch.Tensor:
    token_ids = vocabulary.encode(sequence)
    padded_ids = vocabulary.pad_sequence(token_ids, max_len)
    return torch.tensor([padded_ids], dtype=torch.long, device=device)


def _filter_logits_with_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in the interval (0, 1].")

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
    indices_to_remove = cumulative_probabilities > top_p
    indices_to_remove[..., 0] = False

    filtered_logits = logits.clone()
    filtered_logits[sorted_indices[indices_to_remove]] = float("-inf")
    return filtered_logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> int:
    if temperature <= 0.0:
        raise ValueError("temperature must be greater than zero.")

    adjusted_logits = logits / temperature
    if top_p < 1.0:
        adjusted_logits = _filter_logits_with_top_p(adjusted_logits, top_p)

    probabilities = torch.softmax(adjusted_logits, dim=-1)
    token_index = torch.multinomial(probabilities, num_samples=1)
    return int(token_index.item())


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

            for probability, idx in zip(
                top_probabilities.tolist(), top_indices.tolist(), strict=False
            ):
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


def sample_decoder(
    model,
    vocabulary: Vocabulary,
    seed_text: str,
    next_words: int = 10,
    max_len: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> str:
    sequence = seed_text.strip()
    device = next(model.parameters()).device

    for _ in range(next_words):
        input_ids = _prepare_input(vocabulary, sequence, max_len, device)

        with torch.no_grad():
            logits = model(input_ids)[0]

        next_token_id = sample_next_token(logits, temperature=temperature, top_p=top_p)
        word = vocabulary.idx_to_word.get(next_token_id, "")
        if word in {"", vocabulary.pad_token, vocabulary.unk_token}:
            continue
        sequence = f"{sequence} {word}".strip()

    return sequence


def generate_text(
    model,
    vocabulary: Vocabulary,
    seed_text: str,
    next_words: int = 10,
    max_len: int = 50,
    beam_width: int = 5,
    temperature: float = 1.0,
    top_p: float = 1.0,
    strategy: str = "beam_search",
) -> str:
    normalized_strategy = strategy.strip().lower()
    if normalized_strategy == "beam_search":
        return beam_search_decoder(
            model,
            vocabulary,
            seed_text,
            beam_width=beam_width,
            next_words=next_words,
            max_len=max_len,
        )
    if normalized_strategy in {"top_p", "top-p", "nucleus"}:
        return sample_decoder(
            model,
            vocabulary,
            seed_text,
            next_words=next_words,
            max_len=max_len,
            temperature=temperature,
            top_p=top_p,
        )
    if normalized_strategy == "temperature":
        return sample_decoder(
            model,
            vocabulary,
            seed_text,
            next_words=next_words,
            max_len=max_len,
            temperature=temperature,
            top_p=1.0,
        )

    raise ValueError(f"Unsupported decoding strategy: {strategy}")


def evaluate_bleu(reference_sentence, generated_sentence):
    reference = [reference_sentence.split()]
    candidate = generated_sentence.split()
    return sentence_bleu(reference, candidate, weights=(0.5, 0.5))
