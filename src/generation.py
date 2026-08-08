"""Text generation helpers for WordWave."""

from __future__ import annotations

import math
from typing import cast

import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

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

    if top_p == 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
    indices_to_remove = cumulative_probabilities > top_p
    # Keep at least the most probable token
    indices_to_remove[..., 0] = False

    filtered_logits = logits.clone()
    filtered_logits[sorted_indices[indices_to_remove]] = float("-inf")
    return filtered_logits


def _filter_logits_with_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return logits

    top_k = min(top_k, logits.size(-1))
    top_values, _ = torch.topk(logits, top_k)
    min_value = top_values[-1]

    filtered_logits = torch.where(
        logits < min_value, torch.tensor(float("-inf"), device=logits.device), logits
    )
    return filtered_logits


def _apply_repetition_penalty(
    logits: torch.Tensor, sequence_ids: list[int], penalty: float
) -> torch.Tensor:
    if penalty == 1.0 or not sequence_ids:
        return logits

    device = logits.device
    seq_tensor = torch.tensor(sequence_ids, device=device, dtype=torch.long)

    score = torch.gather(logits, 0, seq_tensor)
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(0, seq_tensor, score)
    return logits


def _apply_no_repeat_ngram(
    logits: torch.Tensor, sequence_ids: list[int], n: int
) -> torch.Tensor:
    if n <= 0 or len(sequence_ids) < n - 1:
        return logits

    context = sequence_ids[-(n - 1) :] if n > 1 else []

    for i in range(len(sequence_ids) - n + 1):
        if n == 1 or sequence_ids[i : i + n - 1] == context:
            banned_token = sequence_ids[i + n - 1]
            logits[banned_token] = float("-inf")

    return logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    if temperature <= 0.0:
        raise ValueError("temperature must be greater than zero.")

    adjusted_logits = logits / temperature
    adjusted_logits = _filter_logits_with_top_k(adjusted_logits, top_k)
    adjusted_logits = _filter_logits_with_top_p(adjusted_logits, top_p)

    probabilities = torch.softmax(adjusted_logits, dim=-1)

    # Fallback to greedy if all valid tokens are masked (e.g. by filters/penalties)
    if torch.isnan(probabilities).any() or probabilities.sum() == 0:
        return int(torch.argmax(logits).item())

    token_index = torch.multinomial(probabilities, num_samples=1)
    return int(token_index.item())


def beam_search_decoder(
    model,
    vocabulary: Vocabulary,
    seed_text: str,
    beam_width: int = 5,
    next_words: int = 10,
    max_len: int = 50,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
):
    # (sequence_text, score, sequence_ids)
    initial_ids = vocabulary.encode(seed_text.strip())
    sequences = [(seed_text.strip(), 0.0, initial_ids)]
    device = next(model.parameters()).device

    for _ in range(next_words):
        all_candidates = []
        for sequence, score, seq_ids in sequences:
            padded_ids = vocabulary.pad_sequence(seq_ids, max_len)
            input_ids = torch.tensor([padded_ids], dtype=torch.long, device=device)

            with torch.no_grad():
                logits = model(input_ids)[0]

                # Apply repetition penalties on the logits directly
                logits = logits.clone()
                logits = _apply_repetition_penalty(logits, seq_ids, repetition_penalty)
                logits = _apply_no_repeat_ngram(logits, seq_ids, no_repeat_ngram_size)

                probabilities = torch.softmax(logits, dim=-1)

            top_k = min(beam_width, probabilities.numel())
            top_probabilities, top_indices = torch.topk(probabilities, k=top_k)

            for probability, idx in zip(
                top_probabilities.tolist(), top_indices.tolist(), strict=False
            ):
                token_id = int(idx)
                word = vocabulary.idx_to_word.get(token_id, "")
                if word in {"", vocabulary.pad_token, vocabulary.unk_token}:
                    continue

                candidate_sequence = f"{sequence} {word}".strip()
                # If probability is 0 (due to mask), candidate score is -inf
                candidate_score = score - math.log(probability + 1e-10)
                candidate_ids = seq_ids + [token_id]

                candidate = (candidate_sequence, candidate_score, candidate_ids)
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
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> str:
    sequence = seed_text.strip()
    seq_ids = vocabulary.encode(sequence)
    device = next(model.parameters()).device

    for _ in range(next_words):
        padded_ids = vocabulary.pad_sequence(seq_ids, max_len)
        input_ids = torch.tensor([padded_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_ids)[0]

            logits = logits.clone()
            logits = _apply_repetition_penalty(logits, seq_ids, repetition_penalty)
            logits = _apply_no_repeat_ngram(logits, seq_ids, no_repeat_ngram_size)

        next_token_id = sample_next_token(
            logits, temperature=temperature, top_k=top_k, top_p=top_p
        )

        seq_ids.append(next_token_id)
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
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
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
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
    if normalized_strategy == "sample":
        return sample_decoder(
            model,
            vocabulary,
            seed_text,
            next_words=next_words,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

    # Legacy fallback mapping for backward compatibility
    if normalized_strategy in {"top_p", "top-p", "nucleus"}:
        return sample_decoder(
            model,
            vocabulary,
            seed_text,
            next_words=next_words,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
    if normalized_strategy == "temperature":
        return sample_decoder(
            model,
            vocabulary,
            seed_text,
            next_words=next_words,
            max_len=max_len,
            temperature=temperature,
            top_k=0,
            top_p=1.0,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

    raise ValueError(f"Unsupported decoding strategy: {strategy}")


def evaluate_bleu(reference_sentence: str, generated_sentence: str) -> float:
    reference = [reference_sentence.split()]
    candidate = generated_sentence.split()
    smoothing = SmoothingFunction().method4
    return float(
        cast(
            float,
            sentence_bleu(
                reference, candidate, weights=(0.5, 0.5), smoothing_function=smoothing
            ),
        )
    )
