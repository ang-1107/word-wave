"""Train the WordWave PyTorch model from a plain text corpus."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import build_training_sequences, build_vocabulary, split_train_validation
from src.model import WordWaveModel
from src.settings import load_settings


SETTINGS = load_settings()


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def run_epoch(
    model: WordWaveModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    total_loss = 0.0
    total_examples = 0
    is_training = optimizer is not None

    model.train(mode=is_training)
    for inputs, labels in loader:
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
    evaluation_inputs: torch.Tensor,
    evaluation_labels: torch.Tensor,
    vocabulary_path: str | Path = SETTINGS.runtime.tokenizer_path,
    model_path: str | Path = SETTINGS.runtime.model_path,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": model_config,
            "max_len": max_len,
            "evaluation_inputs": evaluation_inputs.cpu(),
            "evaluation_labels": evaluation_labels.cpu(),
        },
        Path(model_path),
    )
    torch.save(vocabulary_state, Path(vocabulary_path))


def train_model(
    text: str,
    max_len: int = SETTINGS.training.max_len,
    max_vocab_size: int | None = SETTINGS.training.max_vocab_size,
    embedding_dim: int = SETTINGS.training.embedding_dim,
    hidden_dim: int = SETTINGS.training.hidden_dim,
    num_layers: int = SETTINGS.training.num_layers,
    dropout: float = SETTINGS.training.dropout,
    batch_size: int = SETTINGS.training.batch_size,
    epochs: int = SETTINGS.training.epochs,
    learning_rate: float = SETTINGS.training.learning_rate,
    validation_fraction: float = SETTINGS.training.validation_fraction,
    seed: int = SETTINGS.training.seed,
    vocabulary_path: str | Path = SETTINGS.runtime.tokenizer_path,
    model_path: str | Path = SETTINGS.runtime.model_path,
) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocabulary = build_vocabulary(text, max_vocab_size=max_vocab_size)
    token_ids = vocabulary.encode(text)
    inputs, labels = build_training_sequences(token_ids, max_len=max_len, pad_idx=vocabulary.pad_idx)
    train_dataset, validation_dataset = split_train_validation(inputs, labels, validation_fraction, seed)

    train_loader = DataLoader(TensorDataset(train_dataset.inputs, train_dataset.labels), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(TensorDataset(validation_dataset.inputs, validation_dataset.labels), batch_size=batch_size)

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
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        validation_loss = run_epoch(model, validation_loader, criterion, None, device)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}

        print(f"Epoch {epoch}/{epochs} - train_loss={train_loss:.4f} validation_loss={validation_loss:.4f}")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    vocabulary.save(vocabulary_path)
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
        evaluation_inputs=validation_dataset.inputs,
        evaluation_labels=validation_dataset.labels,
        vocabulary_path=vocabulary_path,
        model_path=model_path,
    )

    return {
        "train_examples": float(len(train_dataset.inputs)),
        "validation_examples": float(len(validation_dataset.inputs)),
        "best_validation_loss": float(best_validation_loss),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the WordWave PyTorch model.")
    parser.add_argument("--text-file", required=True, help="Path to a plain-text training corpus.")
    parser.add_argument("--max-len", type=int, default=SETTINGS.training.max_len, help="Maximum sequence length used for training.")
    parser.add_argument("--max-vocab-size", type=int, default=SETTINGS.training.max_vocab_size, help="Maximum number of vocabulary tokens to keep.")
    parser.add_argument("--embedding-dim", type=int, default=SETTINGS.training.embedding_dim)
    parser.add_argument("--hidden-dim", type=int, default=SETTINGS.training.hidden_dim)
    parser.add_argument("--num-layers", type=int, default=SETTINGS.training.num_layers)
    parser.add_argument("--dropout", type=float, default=SETTINGS.training.dropout)
    parser.add_argument("--batch-size", type=int, default=SETTINGS.training.batch_size)
    parser.add_argument("--epochs", type=int, default=SETTINGS.training.epochs)
    parser.add_argument("--learning-rate", type=float, default=SETTINGS.training.learning_rate)
    parser.add_argument("--validation-fraction", type=float, default=SETTINGS.training.validation_fraction)
    parser.add_argument("--seed", type=int, default=SETTINGS.training.seed)
    parser.add_argument("--model-path", default=str(SETTINGS.runtime.model_path))
    parser.add_argument("--tokenizer-path", default=str(SETTINGS.runtime.tokenizer_path))
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    text = read_text(args.text_file)
    results = train_model(
        text=text,
        max_len=args.max_len,
        max_vocab_size=args.max_vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        vocabulary_path=args.tokenizer_path,
        model_path=args.model_path,
    )
    print(results)


if __name__ == "__main__":
    main()
