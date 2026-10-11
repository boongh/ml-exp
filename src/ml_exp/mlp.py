import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_Full(nn.Module):
    def __init__(self, n_embed, dtype=torch.float32, dropout_rate=0.1, activation=nn.GELU()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed, dtype=dtype),
            activation,
            nn.Linear(n_embed, n_embed, dtype=dtype),
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

## MLP with fixed parameter count to baseline_n_embed ** 2
class MLP_fixed(nn.Module):
    def __init__(self, n_embed, mlp_rank, dtype=torch.float32, dropout_rate=0.1, activation=nn.GELU()):
        super().__init__()
        # Baseline parameter count = 2 * (n_embed ** 2 + n_embed (bias))
        # New parameter count = n_embed* rank + rank + rank * hidden_layer + hidden_layer + hidden_layer * rank + rank + rank * n_embed + n_embed
        # Solve for hidden_layer to get the hidden layer size that will give us the same parameter count as the baseline model with full-rank MLP
        # h = (2 n ^ 2 + n - 2nr - 2r) / (2r+1)

        hidden_layer = int((2 * n_embed ** 2 + n_embed - 2 * mlp_rank * n_embed - 2 * mlp_rank) / (2 * mlp_rank + 1))
        if hidden_layer <= 0:
            raise ValueError("Invalid hidden layer size. Please choose a larger mlp_rank.")

        self.net = nn.Sequential(
            nn.Linear(n_embed, mlp_rank, dtype=dtype),
            nn.Linear(mlp_rank, hidden_layer, dtype=dtype),
            activation,
            nn.Linear(hidden_layer, mlp_rank, dtype=dtype),
            nn.Linear(mlp_rank, n_embed, dtype=dtype),
            nn.Dropout(dropout_rate) # Automatically dissables during eval
        )
    
    def forward(self, x):
        return self.net(x)