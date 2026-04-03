import random

import torch
import torch.nn as nn


class Seq2SeqChatbot(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        pad_idx: int,
        dropout: float = 0.2
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.hidden_dim = hidden_dim

        self.encoder_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.decoder_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.encoder_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.decoder_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def encode(self, src_ids: torch.Tensor):
        embedded = self.dropout(self.encoder_embedding(src_ids))
        _, hidden = self.encoder_gru(embedded)
        return hidden

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        hidden = self.encode(src_ids)
        decoder_input = tgt_ids[:, 0].unsqueeze(1)
        outputs = []

        for step in range(1, tgt_ids.size(1)):
            embedded = self.dropout(self.decoder_embedding(decoder_input))
            decoder_output, hidden = self.decoder_gru(embedded, hidden)
            logits = self.output_layer(decoder_output.squeeze(1))
            outputs.append(logits.unsqueeze(1))

            teacher_force = random.random() < teacher_forcing_ratio
            next_input = tgt_ids[:, step] if teacher_force else logits.argmax(dim=-1)
            decoder_input = next_input.unsqueeze(1)

        return torch.cat(outputs, dim=1)

    def generate(
        self,
        src_ids: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_new_tokens: int
    ) -> list[int]:
        hidden = self.encode(src_ids)
        decoder_input = torch.full(
            (src_ids.size(0), 1),
            bos_idx,
            dtype=torch.long,
            device=src_ids.device
        )

        generated: list[int] = []

        for _ in range(max_new_tokens):
            embedded = self.decoder_embedding(decoder_input)
            decoder_output, hidden = self.decoder_gru(embedded, hidden)
            logits = self.output_layer(decoder_output.squeeze(1))
            next_token = logits.argmax(dim=-1)
            token_id = int(next_token.item())

            if token_id == eos_idx:
                break

            generated.append(token_id)
            decoder_input = next_token.unsqueeze(1)

        return generated
