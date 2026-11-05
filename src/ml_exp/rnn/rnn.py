from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class RNN(torch.nn.Module, ABC):
    def __init__(self, state_size: int, vocab_size: int):
        super().__init__()
        self.state_size = state_size
        self.vocab_size = vocab_size
        pass

    def _check_size(self, x, state, targets=None):
        # x.size() = (B, T)
        # state (L, B, S)
        bx, tx = x.size()
        bs, ls, ss = state.size()

        if bs != bx:
            raise Exception("Inconsistent batch size between x and state")

        if targets is not None:
            bt, tt = targets.size()
            if bt != bx:
                raise Exception("Inconsistent batch size between x and targets")

            if tx != tt:
                raise Exception("Inconsistent time size between x and targets")

        # Token can't be over vocab size limit
        if torch.min(x) < 0 or torch.max(x) >= self.vocab_size:
            raise Exception("Invalid token vocab range")

        if ss != self.state_size:
            raise Exception("Invalid state size")

    @abstractmethod
    def newstate(self, batch: int, device: str) -> torch.Tensor:
        pass

    def forward(
        self, x: torch.Tensor, state: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        self._check_size(x, state, targets)
        return self._do_forward(x, state, targets)

    @abstractmethod
    def _do_forward(
        self, x: torch.Tensor, state: torch.Tensor, targets: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        pass

    @abstractmethod
    def _state_catch_up(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        pass


class RNNLangModel(RNN):
    def __init__(
        self,
        state_size: int,
        vocab_size: int,
        /,
        *,
        dropout: float = 0.4,
        vocab_embedding_size: int = 256,
        hidden_size: int = 384,
        rnnchain_count: int = 20,
    ):
        super().__init__(state_size, vocab_size)
        self.vocab_embedding_size = vocab_embedding_size
        self.hidden_size = hidden_size
        self.rnnchain_count = rnnchain_count
        self.dropout = dropout

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
                    dropout=self.dropout,
                )
                for _ in range(rnnchain_count)
            ]
        )

    # Returns Batch new states
    def newstate(self, batch: int, device: str) -> torch.Tensor:
        return torch.zeros((batch, self.rnnchain_count, self.state_size), device=device)

    def _do_forward(
        self, x: torch.Tensor, state: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
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

        # bx, tx = x.size()
        # state = state.transpose(0, 1)
        # ls, bs, ss = state.size()

        # if bs != bx:
        #     raise Exception("Inconsistent batch size")

        # # Token can't be over vocab size limit
        # if torch.min(x) < 0 or torch.max(x) >= self.vocab_size:
        #     raise Exception("Invalid token vocab range")

        # # State check size.
        # if ls != self.rnnchain_count:
        #     raise Exception("Invalid state array size")

        # if ss != self.state_size:
        #     raise Exception("Invalid state size")

        # =====================================================
        # x.size() = (B, T)
        # state (L, B, S)

        logits = []
        embeddings: torch.Tensor = self.embedder_lin(x)  # (B, T, N)
        embeddings = embeddings.transpose(0, 1)  # Need (T, B, N)
        state = state.transpose(0, 1)
        bx, tx = x.size()
        for i in range(tx):
            carry_logits = embeddings[i]

            carry_logits, state = self._step(carry_logits, state)
            # for layeri, rnnlayer in enumerate(self.rnnchain):
            #     # carry_logits = (B, N)
            #     # state = (L, B, S)
            #     # state[layeri].size() = (B, S)
            #     # print(
            #     #     {
            #     #         "carry_logits": carry_logits.size(),
            #     #         "state": state.size(),
            #     #         "embeddings": embeddings.size(),
            #     #     }
            #     # )
            #     logit, next_state = rnnlayer(carry_logits, state[layeri])
            #     next_states.append(next_state)
            #     carry_logits = carry_logits + logit

            logits.append(carry_logits)

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

    def _state_catch_up(self, x, state) -> torch.Tensor:
        embeddings: torch.Tensor = self.embedder_lin(x)  # (B, T, N)
        embeddings = embeddings.transpose(0, 1)  # Need (T, B, N)
        tx, _, _ = embeddings.size()
        for i in range(tx):
            _, state = self._step(embeddings[i], state)

        return state

    def generate(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        max_token: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        super()._check_size(x, state)
        state = state.transpose(0, 1)
        state = self._state_catch_up(x[:, :-1], state)
        carry_logits = self.embedder_lin(x[:, -1])
        ids = []
        for _ in range(max_token):
            carry_logits, state = self._step(carry_logits, state)
            id_next = torch.multinomial(
                F.softmax(self.lmhead(carry_logits), dim=-1), num_samples=1
            ).squeeze(-1)
            ids.append(id_next)
            carry_logits = self.embedder_lin(id_next)

        return torch.stack(ids, dim=-1), state

    def _step(
        self, x: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_states = []

        carry_logits = x
        for layeri, rnnlayer in enumerate(self.rnnchain):
            # carry_logits = (B, N)
            # state = (L, B, S)
            # state[layeri].size() = (B, S)
            # print(
            #     {
            #         "carry_logits": carry_logits.size(),
            #         "state": state.size(),
            #     }
            # )
            out, next_state = rnnlayer(carry_logits, state[layeri])
            next_states.append(next_state)
            carry_logits = carry_logits + out

        state = torch.stack(next_states)

        return carry_logits, state


class RNNState(torch.nn.Module):
    def __init__(
        self,
        state_size: int,
        embedding_size: int,
        hidden_size: int,
        *,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.state_size = state_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.project = torch.nn.Linear(self.embedding_size, self.hidden_size)
        self.update = torch.nn.Linear(
            self.state_size + self.hidden_size, self.state_size
        )
        self.extract = torch.nn.Linear(self.state_size, self.embedding_size)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, embedding, state):

        # embedding = (B, N)
        # state = (L, B, S)
        # state[layeri].size() = (B, S)

        state = self.evolve_state(embedding, state)
        extract = self.extract(state)

        return extract, state

    def evolve_state(self, embedding, state):
        # embedding = (B, N)
        # state = (L, B, S)
        # state[layeri].size() = (B, S)

        scaled_embedding = self.project(embedding)
        # print(f"size of state = {state.size()}")
        # print(f"size of embedding = {scaled_embedding.size()}")
        fused_state_token = torch.cat((state, scaled_embedding), dim=-1)
        state = state + self.update(fused_state_token)
        state = torch.tanh(state)

        return self.dropout(state)
