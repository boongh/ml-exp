import torch
import torch.nn as nn
import torch.nn.functional as F

class SineusoidalPositionalEncoding(nn.Module):
    def __init__(self, n_embed, dtype=torch.float32, sinusoidal_base = 10000.0, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, n_embed, dtype=dtype)
        self.gamma = nn.Parameter(torch.ones(1, n_embed, dtype=dtype))
        position = torch.arange(0, max_len, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embed, 2, dtype=dtype) * (-torch.log(torch.tensor(sinusoidal_base, dtype=dtype)) / n_embed))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape = (B, T, n_embed)
        # pe shape = (1, max_len, n_embed) -> sliced to match x.size(1)
        return x + self.gamma * self.pe[:, :x.size(1), :]


class SimplePositionalEncoding(nn.Module):
    def __init__(self, n_embed, dtype=torch.float32, max_len=5000):
        super().__init__()
        self.positional_encoding = nn.Embedding(max_len, n_embed, dtype=dtype)

    def forward(self, x):
        # x shape = (B, T, n_embed)
        return x + self.positional_encoding(torch.arange(x.size(1), device=x.device))


#class RoPE(nn.Module):
