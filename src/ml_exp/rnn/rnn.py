import torch
import torch.nn.functional as F
from torch.multiprocessing.spawn import ProcessExitedException


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

        self.rnnchain = torch.nn.ModuleList(
            [
                RNNState(
                    self.state_size,
                    self.vocab_embedding_size,
                    self.hidden_size,
                )
                for i in range(rnnchain_count)
            ]
        )

    # Returns Batch new states
    def newstate(self, batch: int, device: str) -> torch.Tensor:
        return torch.zeros((batch, self.rnnchain_count, self.state_size), device=device)

    def forward(
        self, x: torch.Tensor, state: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates the sum of things and other things.

        Args:
            x (torch.Tensor): Input token batch of size (Batch, Time).
            state (torch.Tensor): State to use for batch inference size (Batch, Layer, State Size).
            targets (torch.Tensor): Target used for loss calculation size (Batch, Time)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: of logits, next_state, loss. Respectively.
            logits: Raw output logits of size (Batch, Time, Vocab Size)
            next_state: The state used for next forward pass size (Batch, Layer, State Size)
            loss:
        """

        # x.size() = (B, T)
        # state (L, B, S)
        bx, tx = x.size()
        state = state.transpose(0, 1)
        ls, bs, ss = state.size()

        if bs != bx:
            raise Exception("Inconsistent batch size")

        # Token can't be over vocab size limit
        if torch.min(x) < 0 or torch.max(x) >= self.vocab_size:
            raise Exception("Invalid token vocab range")

        # State check size.
        if ls != self.rnnchain_count:
            raise Exception("Invalid state array size")

        if ss != self.state_size:
            raise Exception("Invalid state size")

        # =====================================================

        logits = []
        embeddings: torch.Tensor = self.embedder_lin(x)  # (B, T, N)
        embeddings = embeddings.transpose(0, 1)  # Need (T, B, N)
        for i in range(tx):
            next_states = []

            carry_logits = embeddings[i]

            for layeri, rnnlayer in enumerate(self.rnnchain):
                # carry_logits = (B, N)
                # state = (L, B, S)
                # state[layeri].size() = (B, S)
                logit, next_state = rnnlayer(carry_logits, state[layeri])
                next_states.append(next_state)
                carry_logits = carry_logits + logit

            logits.append(carry_logits)
            state = torch.stack(next_states)

        logitsout = self.lmhead(torch.stack(logits, dim=0).transpose(0, 1))

        # print(
        #     {
        #         "logitsout": logitsout.size(),
        #         "embeddings": embeddings.size(),
        #         "x": x.size(),
        #         "state": state.size(),
        #         "targets": targets.size(),
        #     }
        # )

        if targets is not None:
            B, T, C = logitsout.shape
            logits_flat = logitsout.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        else:
            loss = None

        state = state.transpose(0, 1)

        # print(
        #     {"logitsout": logitsout.size(), "state": state.size(), "loss": loss.size()}
        # )

        return logitsout, state, loss


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

        # carry_logits = (B, N)
        # state = (L, B, S)
        # state[layeri].size() = (B, S)

        scaled_embedding = self.project(embedding)
        # print(f"size of state = {state.size()}")
        # print(f"size of embedding = {scaled_embedding.size()}")
        fused_state_token = torch.cat((state, scaled_embedding), dim=-1)
        state = state + self.update(fused_state_token)
        state = torch.tanh(state)
        extract = self.extract(state)

        return self.dropout(extract), self.dropout(state)
