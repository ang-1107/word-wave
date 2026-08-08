"""Train the WordWave PyTorch model from a streaming text corpus."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.corpus import iter_source_files
from src.data import (
    _split_bucket,
    build_streaming_dataloaders,
    build_vocabulary_from_source,
    get_corpus_split_fractions,
)
from src.generation import generate_text
from src.metrics import (
    compute_corpus_bleu,
    compute_distinct_n,
    compute_rouge_l_corpus,
    evaluate_model_metrics,
)
from src.model import WordWaveModel
from src.settings import load_settings
from src.tokenizer import Vocabulary, tokenize_text

SETTINGS = load_settings()


def run_epoch(
    model: WordWaveModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    description: str,
    scaler: GradScaler | None = None,
    gradient_clip_norm: float | None = None,
) -> float:
    total_loss = 0.0
    total_examples = 0
    is_training = optimizer is not None
    use_mixed_precision = (
        scaler is not None and scaler.is_enabled() and device.type == "cuda"
    )

    model.train(mode=is_training)
    for inputs, labels in tqdm(loader, desc=description, unit="batch", leave=True):
        inputs = inputs.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        autocast_context = (
            torch.autocast("cuda", enabled=True)
            if use_mixed_precision
            else nullcontext()
        )
        with autocast_context:
            logits = model(inputs)
            loss = criterion(logits, labels)

        if is_training:
            if use_mixed_precision and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(1, total_examples)


def save_artifacts(
    model: WordWaveModel,
    model_config: dict[str, object],
    training_config: dict[str, object] | None,
    vocabulary_state: dict[str, object],
    max_len: int,
    metrics_payload: dict[str, float],
    vocabulary_path: str | Path = SETTINGS.runtime.tokenizer_path,
    model_path: str | Path = SETTINGS.runtime.model_path,
) -> None:
    model_path = Path(model_path)
    vocabulary_path = Path(vocabulary_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    vocabulary_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving tokenizer to {vocabulary_path}...")
    torch.save(vocabulary_state, vocabulary_path)
    print(f"Saving model to {model_path}...")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": model_config,
            "training_config": training_config or {},
            "max_len": max_len,
            "metrics": metrics_payload,
        },
        model_path,
    )
    print("Saved model and tokenizer artifacts.")


def _build_checkpoint_payload(
    model: WordWaveModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: GradScaler,
    epoch: int,
    best_validation_loss: float,
    best_state_dict: dict[str, torch.Tensor] | None,
    epochs_without_improvement: int,
    model_config: dict[str, object],
    training_config: dict[str, object],
    metrics_payload: dict[str, float] | None = None,
) -> dict[str, object]:
    checkpoint: dict[str, object] = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_validation_loss": best_validation_loss,
        "best_state_dict": best_state_dict,
        "epochs_without_improvement": epochs_without_improvement,
        "model_config": model_config,
        "training_config": training_config,
    }
    if metrics_payload is not None:
        checkpoint["metrics"] = metrics_payload
    return checkpoint


def _load_training_checkpoint(
    checkpoint_path: Path,
    model: WordWaveModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: GradScaler,
    device: torch.device,
) -> dict[str, object] | None:
    if not checkpoint_path.exists():
        return None

    print(f"Resuming from latest checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler_state:
        scaler.load_state_dict(scaler_state)
    return checkpoint


def _evaluate_generation_quality(
    model: WordWaveModel,
    vocabulary: Vocabulary,
    source_path: str,
    max_len: int,
    sample_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Sample lines from the test split, generate continuations, and compute
    corpus-level BLEU, ROUGE-L, and distinct-1/2 metrics."""

    source_files = list(
        iter_source_files(source_path, SETTINGS.runtime.allowed_extensions)
    )
    split_fractions = get_corpus_split_fractions()
    test_threshold = split_fractions.train + split_fractions.validation

    # Collect tokenized lines that fall in the test split
    test_lines: list[list[str]] = []
    for file_path in source_files:
        try:
            with file_path.open("r", encoding="utf-8", errors="ignore") as fh:
                for line_number, line in enumerate(fh, start=1):
                    cleaned = line.strip()
                    if not cleaned:
                        continue
                    line_id = f"{file_path}:{line_number}"
                    if _split_bucket(line_id) >= test_threshold:
                        tokens = tokenize_text(cleaned)
                        if len(tokens) >= 4:
                            test_lines.append(tokens)
        except OSError:
            continue

    if not test_lines:
        print("No test lines available for generation evaluation.")
        return {}

    if len(test_lines) > sample_size:
        test_lines = random.sample(test_lines, sample_size)

    references: list[str] = []
    hypotheses: list[str] = []

    model.eval()
    for tokens in tqdm(test_lines, desc="Generation evaluation", unit="sample"):
        split_point = max(1, len(tokens) // 2)
        seed = " ".join(tokens[:split_point])
        reference = " ".join(tokens[split_point:])
        num_to_generate = len(tokens) - split_point

        generated_full = generate_text(
            model,
            vocabulary,
            seed,
            next_words=num_to_generate,
            max_len=max_len,
            strategy="beam_search",
            beam_width=5,
        )
        # Extract only the generated continuation (remove seed prefix)
        continuation = generated_full[len(seed) :].strip()
        references.append(reference)
        hypotheses.append(continuation)

    result: dict[str, float] = {
        "corpus_bleu": compute_corpus_bleu(references, hypotheses),
        "rouge_l": compute_rouge_l_corpus(references, hypotheses),
        "distinct_1": compute_distinct_n(hypotheses, n=1),
        "distinct_2": compute_distinct_n(hypotheses, n=2),
    }
    print(
        f"Generation metrics — BLEU={result['corpus_bleu']:.4f}  "
        f"ROUGE-L={result['rouge_l']:.4f}  "
        f"distinct-1={result['distinct_1']:.4f}  "
        f"distinct-2={result['distinct_2']:.4f}"
    )
    return result


def train_model(
    data_path: str,
    max_len: int = SETTINGS.training.max_len,
    max_vocab_size: int | None = SETTINGS.training.max_vocab_size,
    embedding_dim: int = SETTINGS.training.embedding_dim,
    hidden_dim: int = SETTINGS.training.hidden_dim,
    num_layers: int = SETTINGS.training.num_layers,
    dropout: float = SETTINGS.training.dropout,
    batch_size: int = SETTINGS.training.batch_size,
    epochs: int = SETTINGS.training.epochs,
    learning_rate: float = SETTINGS.training.learning_rate,
    gradient_clip_norm: float = SETTINGS.training.gradient_clip_norm,
    lr_scheduler_patience: int = SETTINGS.training.lr_scheduler_patience,
    early_stopping_patience: int = SETTINGS.training.early_stopping_patience,
    early_stopping_min_delta: float = SETTINGS.training.early_stopping_min_delta,
    use_mixed_precision: bool = SETTINGS.training.use_mixed_precision,
    label_smoothing: float = SETTINGS.training.label_smoothing,
    lr_warmup_epochs: int = SETTINGS.training.lr_warmup_epochs,
    tie_weights: bool = SETTINGS.training.tie_weights,
    evaluation_sample_size: int = SETTINGS.training.evaluation_sample_size,
    vocabulary_path: str | Path = SETTINGS.runtime.tokenizer_path,
    model_path: str | Path = SETTINGS.runtime.model_path,
) -> dict[str, float]:
    seed = SETTINGS.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Building vocabulary from {data_path}...")
    vocabulary = build_vocabulary_from_source(data_path, max_vocab_size=max_vocab_size)
    print("Building streaming dataloaders...")
    train_loader, validation_loader, test_loader = build_streaming_dataloaders(
        data_path,
        vocabulary,
        max_len=max_len,
        batch_size=batch_size,
    )

    model = WordWaveModel(
        vocab_size=len(vocabulary),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        tie_weights=tie_weights,
    ).to(device)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        f"Model parameters: {total_parameters:,} total, {trainable_parameters:,} trainable"
    )

    model_config: dict[str, object] = {
        "vocab_size": len(vocabulary),
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "tie_weights": tie_weights,
    }

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=lr_scheduler_patience,
    )
    scaler = GradScaler("cuda", enabled=use_mixed_precision and device.type == "cuda")

    checkpoint_root = Path(model_path).parent
    latest_checkpoint_path = SETTINGS.runtime.latest_checkpoint_path
    best_checkpoint_path = SETTINGS.runtime.best_checkpoint_path
    if not latest_checkpoint_path.is_absolute():
        latest_checkpoint_path = checkpoint_root / latest_checkpoint_path
    if not best_checkpoint_path.is_absolute():
        best_checkpoint_path = checkpoint_root / best_checkpoint_path

    training_config: dict[str, object] = {
        "gradient_clip_norm": gradient_clip_norm,
        "lr_scheduler_patience": lr_scheduler_patience,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "use_mixed_precision": use_mixed_precision,
        "label_smoothing": label_smoothing,
        "lr_warmup_epochs": lr_warmup_epochs,
    }

    checkpoint = _load_training_checkpoint(
        latest_checkpoint_path,
        model,
        optimizer,
        scheduler,
        scaler,
        device,
    )

    best_validation_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0
    start_epoch = 1

    if checkpoint is not None:
        start_epoch = int(cast(int, checkpoint.get("epoch", 0))) + 1
        best_validation_loss = float(
            cast(float, checkpoint.get("best_validation_loss", best_validation_loss))
        )
        best_state_dict = cast(
            dict[str, torch.Tensor] | None, checkpoint.get("best_state_dict")
        )
        epochs_without_improvement = int(
            cast(int, checkpoint.get("epochs_without_improvement", 0))
        )
        print(f"Resuming training from epoch {start_epoch}.")
    else:
        print("No latest checkpoint found; starting a fresh training run.")

    # Per-epoch metric log
    log_path = checkpoint_root / "training_log.jsonl"

    for epoch in range(start_epoch, epochs + 1):
        # LR warmup: linearly ramp LR from a small fraction to the full value
        if lr_warmup_epochs > 0 and epoch <= lr_warmup_epochs:
            warmup_factor = epoch / lr_warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate * warmup_factor

        print(f"Starting epoch {epoch}/{epochs}...")
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            description=f"Training epoch {epoch}/{epochs}",
            scaler=scaler,
            gradient_clip_norm=gradient_clip_norm,
        )
        validation_loss = run_epoch(
            model,
            validation_loader,
            criterion,
            None,
            device,
            description=f"Validating epoch {epoch}/{epochs}",
        )

        # Only step the plateau scheduler after warmup completes
        if epoch > lr_warmup_epochs:
            scheduler.step(validation_loss)

        improvement = best_validation_loss - validation_loss
        if improvement > early_stopping_min_delta:
            best_validation_loss = validation_loss
            best_state_dict = {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
            print(f"New best validation loss: {best_validation_loss:.4f}")
            torch.save(
                _build_checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_validation_loss,
                    best_state_dict,
                    epochs_without_improvement,
                    model_config,
                    training_config,
                ),
                best_checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{epochs} — "
            f"train_loss={train_loss:.4f}  val_loss={validation_loss:.4f}  "
            f"lr={current_lr:.6f}"
        )

        # Persist per-epoch metrics
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "validation_loss": float(validation_loss),
            "lr": current_lr,
        }
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(epoch_metrics) + "\n")

        torch.save(
            _build_checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_validation_loss,
                best_state_dict,
                epochs_without_improvement,
                model_config,
                training_config,
                metrics_payload={
                    "train_loss": float(train_loss),
                    "validation_loss": float(validation_loss),
                },
            ),
            latest_checkpoint_path,
        )
        print(f"Saved latest checkpoint to {latest_checkpoint_path}")

        if epochs_without_improvement >= early_stopping_patience:
            print(
                "Early stopping triggered after "
                f"{epoch} epochs without sufficient validation improvement."
            )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # ---- Full evaluation on validation and test splits ----
    validation_top_k, validation_average_loss, validation_perplexity = (
        evaluate_model_metrics(model, validation_loader, device)
    )
    test_top_k, test_average_loss, test_perplexity = evaluate_model_metrics(
        model, test_loader, device
    )

    metrics_payload: dict[str, float] = {
        "validation_top_k": float(validation_top_k),
        "validation_loss": float(validation_average_loss),
        "validation_perplexity": float(validation_perplexity),
        "test_top_k": float(test_top_k),
        "test_loss": float(test_average_loss),
        "test_perplexity": float(test_perplexity),
    }

    # ---- Corpus-level generation evaluation ----
    if evaluation_sample_size > 0:
        generation_metrics = _evaluate_generation_quality(
            model, vocabulary, data_path, max_len, evaluation_sample_size, device
        )
        metrics_payload.update(generation_metrics)

    save_artifacts(
        model,
        model_config,
        training_config,
        vocabulary.to_state_dict(),
        max_len=max_len,
        metrics_payload=metrics_payload,
        vocabulary_path=vocabulary_path,
        model_path=model_path,
    )

    metrics_payload["best_validation_loss"] = float(best_validation_loss)
    return metrics_payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the WordWave PyTorch model.")
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to a plain-text file or directory of plaintext files for training.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=SETTINGS.training.max_len,
        help="Maximum sequence length used for training.",
    )
    parser.add_argument(
        "--max-vocab-size",
        type=int,
        default=SETTINGS.training.max_vocab_size,
        help="Maximum number of vocabulary tokens to keep.",
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=SETTINGS.training.embedding_dim
    )
    parser.add_argument("--hidden-dim", type=int, default=SETTINGS.training.hidden_dim)
    parser.add_argument("--num-layers", type=int, default=SETTINGS.training.num_layers)
    parser.add_argument("--dropout", type=float, default=SETTINGS.training.dropout)
    parser.add_argument("--batch-size", type=int, default=SETTINGS.training.batch_size)
    parser.add_argument("--epochs", type=int, default=SETTINGS.training.epochs)
    parser.add_argument(
        "--learning-rate", type=float, default=SETTINGS.training.learning_rate
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=SETTINGS.training.gradient_clip_norm,
    )
    parser.add_argument(
        "--lr-scheduler-patience",
        type=int,
        default=SETTINGS.training.lr_scheduler_patience,
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=SETTINGS.training.early_stopping_patience,
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=SETTINGS.training.early_stopping_min_delta,
    )
    parser.add_argument(
        "--use-mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=SETTINGS.training.use_mixed_precision,
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=SETTINGS.training.label_smoothing,
        help="Label smoothing factor for CrossEntropyLoss.",
    )
    parser.add_argument(
        "--lr-warmup-epochs",
        type=int,
        default=SETTINGS.training.lr_warmup_epochs,
        help="Number of linear LR warmup epochs before ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--tie-weights",
        action=argparse.BooleanOptionalAction,
        default=SETTINGS.training.tie_weights,
        help="Tie embedding and output projection weights.",
    )
    parser.add_argument(
        "--evaluation-sample-size",
        type=int,
        default=SETTINGS.training.evaluation_sample_size,
        help="Number of test lines to sample for generation evaluation (0 to skip).",
    )
    parser.add_argument("--model-path", default=str(SETTINGS.runtime.model_path))
    parser.add_argument(
        "--tokenizer-path", default=str(SETTINGS.runtime.tokenizer_path)
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    results = train_model(
        data_path=args.data_path,
        max_len=args.max_len,
        max_vocab_size=args.max_vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        gradient_clip_norm=args.gradient_clip_norm,
        lr_scheduler_patience=args.lr_scheduler_patience,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        use_mixed_precision=args.use_mixed_precision,
        label_smoothing=args.label_smoothing,
        lr_warmup_epochs=args.lr_warmup_epochs,
        tie_weights=args.tie_weights,
        evaluation_sample_size=args.evaluation_sample_size,
        vocabulary_path=args.tokenizer_path,
        model_path=args.model_path,
    )
    print(results)


if __name__ == "__main__":
    main()
