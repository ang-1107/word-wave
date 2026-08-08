"""PyTorch model used by WordWave."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class WordWaveModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.2,
        tie_weights: bool = False,
    ) -> None:
        super().__init__()
        self.tie_weights_flag = tie_weights
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

        # When weight tying is enabled, the classifier projects to
        # embedding_dim and the final vocab projection reuses
        # self.embedding.weight — cutting parameters and coupling the
        # input/output representations (Press & Wolf, 2017).
        output_dim = embedding_dim if tie_weights else vocab_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
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
        logits = self.classifier(context)

        if self.tie_weights_flag:
            logits = F.linear(logits, self.embedding.weight)

        return logits
