"""PyTorch model used by WordWave."""

from __future__ import annotations

import torch
from torch import nn


class WordWaveModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pad_mask = input_ids == self.embedding.padding_idx
        embeddings = self.embedding(input_ids)
        outputs, _ = self.lstm(embeddings)
        attention_logits = self.attention(outputs).squeeze(-1)
        attention_logits = attention_logits.masked_fill(pad_mask, float("-inf"))

        # Guard against all-padding inputs: if every position is masked,
        # softmax would produce NaN.  Fall back to uniform attention.
        all_masked = pad_mask.all(dim=1, keepdim=True)
        attention_weights = torch.softmax(attention_logits, dim=1)
        if all_masked.any():
            uniform = torch.ones_like(attention_weights) / attention_weights.size(1)
            attention_weights = torch.where(all_masked, uniform, attention_weights)

        context = torch.bmm(attention_weights.unsqueeze(1), outputs).squeeze(1)
        return self.classifier(context)
