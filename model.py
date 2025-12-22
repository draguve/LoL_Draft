import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # not a parameter

    def forward(self, x: torch.Tensor):
        S = x.size(1)
        return x + self.pe[:S].unsqueeze(0)  # broadcast to [1, S, E]


class LeagueModel(nn.Module):
    def __init__(
        self,
        max_seq_len: int = 150,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dim_ff: int = 128,
        dropout: float = 0.1,
        vocab_size: int = 1000,
        num_out_features: int = 1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)

        self.pos = PositionalEncoding(d_model, max_len=max_seq_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_out_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        h = self.token_emb(x)
        h = self.pos(h)
        h = self.encoder(h)
        h = self.head(h)
        h = self.sigmoid(h)
        return h


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randint(1, 1000, (8, 150))
    model = LeagueModel(max_seq_len=150)
    y = model(x)
    print("input:", x.shape)
    print("output:", y.shape)
