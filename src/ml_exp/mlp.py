import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_Full(nn.Module):
    def __init__(self, n_embed, dtype=torch.float32, dropout_rate=0.1, activation=nn.GELU()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 2, dtype=dtype),
            activation,
            nn.Linear(n_embed * 2, n_embed, dtype=dtype),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        return self.net(x)

class MLP_Lowrank(nn.Module):
    def __init__(self, n_embed, rank, dtype=torch.float32, dropout_rate=0.1, activation=nn.GELU()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, rank, dtype=dtype),
            nn.Linear(rank, n_embed, dtype=dtype),
            activation,
            nn.Linear(n_embed, rank, dtype=dtype),
            nn.Linear(rank, n_embed, dtype=dtype),
            nn.Dropout(dropout_rate) # Automatically dissables during eval
        )
    
    def forward(self, x):
        return self.net(x)