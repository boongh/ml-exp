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
    def __init__(self, n_embed, mlp_rank=None, dtype=torch.float32, dropout_rate=0.1, activation=nn.GELU(), baseline_hidden=None):
        super().__init__()
        # baseline_hidden: hidden width H of the baseline MLP (n_embed -> H -> n_embed), defaults to n_embed.
        # mlp_rank=None builds that baseline MLP itself.
        baseline_hidden = n_embed if baseline_hidden is None else baseline_hidden
        if baseline_hidden <= 0:
            raise ValueError("baseline_hidden must be positive.")
        if mlp_rank is None:
            self.hidden_layer = baseline_hidden
            self.net = nn.Sequential(
                nn.Linear(n_embed, baseline_hidden, dtype=dtype),
                activation,
                nn.Linear(baseline_hidden, n_embed, dtype=dtype),
                nn.Dropout(dropout_rate)
            )
            return

        # Baseline parameter count = n_embed * H + H + H * n_embed + n_embed = 2 * n_embed * H + H + n_embed
        # New parameter count = n_embed* rank + rank + rank * hidden_layer + hidden_layer + hidden_layer * rank + rank + rank * n_embed + n_embed
        # Solve for hidden_layer to get the hidden layer size that will give us the same parameter count as the baseline MLP
        # h = (2 n H + H - 2nr - 2r) / (2r+1)
        # (H = n_embed gives the plain full-rank baseline: (2 n ^ 2 + n - 2nr - 2r) / (2r+1))

        hidden_layer = int((2 * n_embed * baseline_hidden + baseline_hidden - 2 * mlp_rank * n_embed - 2 * mlp_rank) / (2 * mlp_rank + 1))
        if hidden_layer <= 0:
            raise ValueError("Invalid hidden layer size. Please choose a larger mlp_rank.")
        self.hidden_layer = hidden_layer

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