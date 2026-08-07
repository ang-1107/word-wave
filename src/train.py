"""Train the WordWave PyTorch model from a streaming text corpus."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data import build_streaming_dataloaders, build_vocabulary_from_source
from src.metrics import evaluate_model_metrics
from src.model import WordWaveModel
from src.settings import load_settings

SETTINGS = load_settings()


def run_epoch(
    model: WordWaveModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    description: str,
) -> float:
    total_loss = 0.0
    total_examples = 0
    is_training = optimizer is not None

    model.train(mode=is_training)
    for inputs, labels in tqdm(loader, desc=description, unit="batch", leave=True):
        inputs = inputs.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = criterion(logits, labels)

        if is_training:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(1, total_examples)


def save_artifacts(
    model: WordWaveModel,
    model_config: dict[str, object],
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
            "max_len": max_len,
            "metrics": metrics_payload,
        },
        model_path,
    )
    print("Saved model and tokenizer artifacts.")


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
    vocabulary_path: str | Path = SETTINGS.runtime.tokenizer_path,
    model_path: str | Path = SETTINGS.runtime.model_path,
) -> dict[str, float]:
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
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_validation_loss = float("inf")
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        print(f"Starting epoch {epoch}/{epochs}...")
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            description=f"Training epoch {epoch}/{epochs}",
        )
        validation_loss = run_epoch(
            model,
            validation_loader,
            criterion,
            None,
            device,
            description=f"Validating epoch {epoch}/{epochs}",
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state_dict = {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            }

        print(
            f"Epoch {epoch}/{epochs} - train_loss={train_loss:.4f} validation_loss={validation_loss:.4f}"
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss = run_epoch(
        model,
        test_loader,
        criterion,
        None,
        device,
        description="Evaluating test split",
    )
    validation_top_k, validation_average_loss, validation_perplexity = (
        evaluate_model_metrics(model, validation_loader, device)
    )

    save_artifacts(
        model,
        {
            "vocab_size": len(vocabulary),
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        },
        vocabulary.to_state_dict(),
        max_len=max_len,
        metrics_payload={
            "validation_top_k": float(validation_top_k),
            "validation_loss": float(validation_average_loss),
            "validation_perplexity": float(validation_perplexity),
            "test_loss": float(test_loss),
        },
        vocabulary_path=vocabulary_path,
        model_path=model_path,
    )

    return {
        "validation_top_k": float(validation_top_k),
        "validation_loss": float(validation_average_loss),
        "validation_perplexity": float(validation_perplexity),
        "test_loss": float(test_loss),
        "best_validation_loss": float(best_validation_loss),
    }


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
        vocabulary_path=args.tokenizer_path,
        model_path=args.model_path,
    )
    print(results)


if __name__ == "__main__":
    main()
