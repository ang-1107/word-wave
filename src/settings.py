"""Load WordWave settings from the root YAML config file."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class RuntimeSettings:
    model_path: Path
    tokenizer_path: Path
    latest_checkpoint_path: Path
    best_checkpoint_path: Path
    default_seed_text: str
    default_generation_length: int
    max_generation_length: int
    default_beam_width: int
    default_sampling_temperature: float
    default_top_k: int
    default_top_p: float
    default_repetition_penalty: float
    default_no_repeat_ngram_size: int
    default_decoding_strategy: str
    default_corpus_size_mb: int


@dataclass(frozen=True)
class TrainingSettings:
    max_len: int
    max_vocab_size: int
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float
    batch_size: int
    epochs: int
    learning_rate: float
    gradient_clip_norm: float
    lr_scheduler_patience: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    use_mixed_precision: bool
    validation_fraction: float
    test_fraction: float
    seed: int
    evaluation_sample_size: int
    label_smoothing: float
    lr_warmup_epochs: int
    tie_weights: bool


@dataclass(frozen=True)
class Settings:
    runtime: RuntimeSettings
    training: TrainingSettings


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    runtime_data = data["runtime"]
    training_data = data["training"]

    return Settings(
        runtime=RuntimeSettings(
            model_path=_resolve_path(runtime_data["model_path"]),
            tokenizer_path=_resolve_path(runtime_data["tokenizer_path"]),
            latest_checkpoint_path=_resolve_path(
                runtime_data.get("latest_checkpoint_path", "word-wave.latest.pt")
            ),
            best_checkpoint_path=_resolve_path(
                runtime_data.get("best_checkpoint_path", "word-wave.best.pt")
            ),
            default_seed_text=str(runtime_data["default_seed_text"]),
            default_generation_length=int(runtime_data["default_generation_length"]),
            max_generation_length=int(runtime_data["max_generation_length"]),
            default_beam_width=int(runtime_data["default_beam_width"]),
            default_sampling_temperature=float(
                runtime_data["default_sampling_temperature"]
            ),
            default_top_k=int(runtime_data.get("default_top_k", 0)),
            default_top_p=float(runtime_data["default_top_p"]),
            default_repetition_penalty=float(
                runtime_data.get("default_repetition_penalty", 1.0)
            ),
            default_no_repeat_ngram_size=int(
                runtime_data.get("default_no_repeat_ngram_size", 0)
            ),
            default_decoding_strategy=str(runtime_data["default_decoding_strategy"]),
            default_corpus_size_mb=int(runtime_data.get("default_corpus_size_mb", 64)),
        ),
        training=TrainingSettings(
            max_len=int(training_data["max_len"]),
            max_vocab_size=int(training_data["max_vocab_size"]),
            embedding_dim=int(training_data["embedding_dim"]),
            hidden_dim=int(training_data["hidden_dim"]),
            num_layers=int(training_data["num_layers"]),
            dropout=float(training_data["dropout"]),
            batch_size=int(training_data["batch_size"]),
            epochs=int(training_data["epochs"]),
            learning_rate=float(training_data["learning_rate"]),
            gradient_clip_norm=float(training_data.get("gradient_clip_norm", 1.0)),
            lr_scheduler_patience=int(training_data.get("lr_scheduler_patience", 1)),
            early_stopping_patience=int(
                training_data.get("early_stopping_patience", 3)
            ),
            early_stopping_min_delta=float(
                training_data.get("early_stopping_min_delta", 0.0001)
            ),
            use_mixed_precision=bool(training_data.get("use_mixed_precision", True)),
            validation_fraction=float(training_data["validation_fraction"]),
            test_fraction=float(training_data["test_fraction"]),
            seed=int(training_data["seed"]),
            evaluation_sample_size=int(
                training_data.get("evaluation_sample_size", 500)
            ),
            label_smoothing=float(training_data.get("label_smoothing", 0.0)),
            lr_warmup_epochs=int(training_data.get("lr_warmup_epochs", 0)),
            tie_weights=bool(training_data.get("tie_weights", False)),
        ),
    )
