import torch
import torch.nn as nn


class ChatbotNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        pad_idx: int,
        dropout: float = 0.2
    ):
        super().__init__()

        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids shape: [batch_size, max_len]
        embedded = self.embedding(input_ids)  # [batch_size, max_len, embed_dim]

        mask = (input_ids != self.pad_idx).unsqueeze(-1).float()  # [batch_size, max_len, 1]
        masked_embedded = embedded * mask

        token_count = mask.sum(dim=1).clamp(min=1.0)  # [batch_size, 1]
        pooled = masked_embedded.sum(dim=1) / token_count  # [batch_size, embed_dim]

        x = self.fc1(pooled)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits