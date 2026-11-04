import torch
import torch.nn.functional as F


class RNN(torch.nn.Module):
    def __init__(
        self,
        state_size: int,
        vocab_size: int,
        /,
        *,
        vocab_embedding_size: int = 256,
        hidden_size: int = 384,
        rnnchain_count: int = 20,
    ):
        super().__init__()
        self.state_size = state_size
        self.vocab_size = vocab_size
        self.vocab_embedding_size = vocab_embedding_size
        self.hidden_size = hidden_size
        self.rnnchain_count = rnnchain_count

        self.embedder_lin = torch.nn.Embedding(
            self.vocab_size, self.vocab_embedding_size
        )

        self.lmhead = torch.nn.Linear(self.vocab_embedding_size, self.vocab_size)

        self.rnnchain = torch.nn.Sequential(
            *[
                RNNState(
                    self.state_size,
                    self.vocab_embedding_size,
                    self.hidden_size,
                )
                for i in range(rnnchain_count)
            ]
        )

    def newstate(self):
        return torch.zeros((self.rnnchain_count, self.state_size))

    # x.shape() = [T]
    def forward(self, x: torch.Tensor, state: torch.Tensor, targets: torch.Tensor):
        b, t = x.size()
        if b != 1:
            raise Exception("Invalid token tensor size")

        if torch.min(x) < 0 or torch.max(x) >= self.vocab_size:
            raise Exception("Invalid token vocab range")

        logits = []
        embeddings = self.embedder_lin(x)
        for i in range(t):
            fused_state_token = torch.cat((state, embeddings[0, i, :]), dim=-1)
            state = state + self.update(fused_state_token)
            state = torch.tanh(state)
            extract = self.extract(state)
            logits.append(self.lmhead(extract))

        logits = torch.stack(logits, dim=0)

        if targets is not None:
            T, C = logits.shape
            logits_flat = logits.view(T, C)
            targets_flat = targets.view(T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        else:
            loss = None

        return logits, state, loss


class RNNState(torch.nn.Module):
    def __init__(self, state_size: int, embedding_size: int, hidden_size: int):
        super().__init__()
        self.state_size = state_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.project = torch.nn.Linear(self.embedding_size, self.hidden_size)
        self.update = torch.nn.Linear(
            self.state_size + self.hidden_size, self.state_size
        )
        self.extract = torch.nn.Linear(self.state_size, self.embedding_size)

        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, embedding, state):
        fused_state_token = torch.cat((state[], self.project(embedding)), dim=-1)
        state = state + self.update(fused_state_token)
        state = torch.tanh(state)
        extract = self.extract(state)

        return self.dropout(extract), state
