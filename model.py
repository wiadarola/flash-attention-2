import math

import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        p_drop: float,
        d_ff: int,
        d_vocab: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoding = LazyPositionalEncoding()
        self.encoders = nn.ModuleList(
            Encoder(d_model, n_heads, p_drop, d_ff) for _ in range(n_layers)
        )
        self.decoders = nn.ModuleList(
            Decoder(d_model, n_heads, p_drop, d_ff) for _ in range(n_layers)
        )
        self.linear = nn.Linear(d_model, d_vocab)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        encoded = self.encoders(x, attn_mask)
        decoded = self.decoders(x, encoded, attn_mask)
        x = self.linear(decoded)
        return x


class LazyPositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "encoding"):
            _, L, F = x.shape  # B L F
            self._instantiate(L, F)
            self.encoding = self.encoding.to(x.device, x.dtype, non_blocking=True)
        return x + self.encoding

    def _instantiate(self, seq_len: int, d_model: int):
        encoding = torch.zeros(seq_len, d_model)  # L F
        dimension = torch.arange(d_model).repeat(seq_len, 1)  # L F
        div_term = 10_000 ** (2 * dimension / d_model)
        position = torch.arange(seq_len).repeat(d_model, 1).T  # L F

        theta = position / div_term
        encoding[:, 0::2] = theta[:, 0::2].sin()
        encoding[:, 1::2] = theta[:, 1::2].cos()
        encoding = encoding.unsqueeze(0)

        self.register_buffer("encoding", encoding, persistent=True)  # 1 L F


class Encoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, p_drop: float, d_ff: int):
        super().__init__()
        self.attn = SelfAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(p_drop)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.ln1(x + self.attn(x, attn_mask)))
        x = self.drop2(self.ln2(x + self.ffn(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, p_drop: float, d_ff: int):
        super().__init__()
        self.self_attn = SelfAttention(d_model, n_heads)
        self.causal_mask = torch.triu(torch.full((128, 128), -torch.inf), diagonal=1)
        self.ln1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(p_drop)
        self.cross_attn = EncoderDecoderAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(p_drop)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.ln3 = nn.LayerNorm(d_model)
        self.drop3 = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor, e: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.ln1(x + self.self_attn(x, self.causal_mask + mask)))
        x = self.drop2(self.ln2(x + self.cross_attn(x, e, mask)))
        x = self.drop3(self.ln3(x + self.ffn(x)))
        return x


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.d_k = math.sqrt(d_model / n_heads)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).unflatten(-1, (self.n_heads, -1)).transpose(1, 2).chunk(3, -1)
        qk = (q @ k.transpose(-1, -2) / self.d_k + mask).softmax(-1)
        attn = (qk @ v).transpose(1, 2).flatten(2)
        x = self.out(attn)
        return x


class EncoderDecoderAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.d = math.sqrt(d_model / n_heads)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, e: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q = self.q(x).unflatten(-1, (self.n_heads, -1)).transpose(1, 2)
        k, v = self.kv(e).unflatten(-1, (self.n_heads, -1)).transpose(1, 2).chunk(2, -1)
        qk = (q @ k.transpose(-1, -2) / self.d + mask).softmax(-1)
        attn = (qk @ v).transpose(1, 2).flatten(2)
        x = self.out(attn)
        return x


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
