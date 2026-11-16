from abc import ABC, abstractmethod
from enum import Enum
from typing import Self

import torch
import torch.nn.functional as F
from torch.nn import Module

from ml_exp.experiments.base_experimental import (
    BaseExperimentalModel,
    ParameterBreakdown,
    VarEnum,
)


class RNN(Module, ABC):
    def __init__(self, state_size: int, vocab_size: int):
        super().__init__()
        self.state_size = state_size
        self.vocab_size = vocab_size

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


class RNNLangModelHyperparam(Enum):
    state_size = "STATE_SIZE"
    vocab_embedding_size = "VOCAB_EMBEDDING"
    hidden_size = "HIDDEN_SIZE"
    rnn_layer_count = "LAYERCOUNT"


class RNNLangModel(RNN, BaseExperimentalModel[RNNLangModelHyperparam]):
    def __init__(
        self,
        state_size: int,
        vocab_size: int,
        /,
        *,
        dropout: float = 0.4,
        vocab_embedding_size: int = 256,
        hidden_size: int = 384,
        layer_count: int = 20,
    ):
        super().__init__(state_size, vocab_size)
        self.vocab_embedding_size = vocab_embedding_size
        self.hidden_size = hidden_size
        self.rnnchain_count = layer_count
        self.dropout = dropout

        self.embedder_lin = torch.nn.Embedding(
            self.vocab_size, self.vocab_embedding_size
        )

        self.rnnchain = torch.nn.ModuleList(
            [
                RNNState(
                    self.state_size,
                    self.vocab_embedding_size,
                    self.hidden_size,
                    dropout=self.dropout,
                )
                for _ in range(layer_count)
            ]
        )

        self.lmhead = torch.nn.Linear(self.vocab_embedding_size, self.vocab_size)

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
        state = state.transpose(0, 1)  # (L, B, S)
        bx, tx = x.size()

        # def evolve_state(self, embedding, state):
        #     # state[layeri].size() = (B, S)

        #     scaled_embedding = self.project(embedding)
        #     # print(f"size of state = {state.size()}")
        #     # print(f"size of embedding = {scaled_embedding.size()}")
        #     fused_state_token = torch.cat((state, scaled_embedding), dim=-1)
        #     state = state + self.update(fused_state_token)
        #     state = torch.tanh(state)
        #     extract = self.extract(self.dropout(state))

        # embedding = (T, B, N)
        # state = (L, B, S)
        #

        next_state = []

        embeddings = embeddings.transpose(0, 1)  # Need (T, B, N)
        for layeri, layer in enumerate(self.rnnchain):
            thisstate = state[layeri]  # (B, S) of layer L
            next_token = []
            hidden = layer.project(embeddings)  # (T, B, H)
            for embedding in hidden:  # (B, H)
                fused_state_token = torch.cat((thisstate, embedding), dim=-1)
                thisstate = layer.dropout(
                    torch.tanh(thisstate + layer.update(fused_state_token))
                )
                next_token.append(thisstate)
            state_sequence = torch.stack(next_token, dim=0)
            embeddings = embeddings + layer.extract(state_sequence)
            next_state.append(thisstate)

        final_state = torch.stack(next_state, dim=0)

        logitsout = self.lmhead(embeddings.transpose(0, 1))  # (B, T, E)

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

        state = final_state.transpose(0, 1)

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

    def copy_scaled_to_fixed_p(self, var: VarEnum, pcount: int) -> Self:
        """Return a freshly initialized model with only `var` resized.

        Choose the positive integer giving the closest parameter count, choosing
        the smaller size on ties. Targets below the minimum use size one.
        Preserve dropout, device, dtype and training mode, but do not copy weights.
        """
        from math import isqrt

        if isinstance(pcount, bool) or not isinstance(pcount, int) or pcount <= 0:
            raise ValueError("pcount must be a positive integer")
        if not isinstance(var, RNNLangModelHyperparam):
            raise ValueError(f"Unsupported scaling variable: {var!r}")

        S, E, H, L, V = (
            self.state_size,
            self.vocab_embedding_size,
            self.hidden_size,
            self.rnnchain_count,
            self.vocab_size,
        )
        if min(S, E, H, L, V) < 1:
            raise ValueError(
                "Scaling requires positive model dimensions and layer count"
            )

        # Embedding + head: 2VE + V (weights are not tied).
        # Each RNNState, including all biases:
        # project = EH + H; update = S(S + H) + S; extract = SE + E.
        # Thus P = 2VE + V + L[S² + S(H + E + 1) + EH + H + E].
        # Write P(x) = ax² + bx + c for the one variable being changed.
        a = 0
        if var is RNNLangModelHyperparam.state_size:
            # P(S) = LS² + L(H + E + 1)S + c,
            # c = 2VE + V + L(EH + H + E).
            # Inverse: S = (-b + sqrt(b² + 4L(pcount - c))) / (2L).
            a = L
            b = L * (H + E + 1)
            c = 2 * V * E + V + L * (E * H + H + E)
        elif var is RNNLangModelHyperparam.vocab_embedding_size:
            # P(E) = [2V + L(S + H + 1)]E + V + L(S² + SH + S + H).
            # Inverse: E = [pcount - V - L(S² + SH + S + H)]
            #              / [2V + L(S + H + 1)].
            b = 2 * V + L * (S + H + 1)
            c = V + L * (S * S + S * H + S + H)
        elif var is RNNLangModelHyperparam.hidden_size:
            # P(H) = L(E + S + 1)H + 2VE + V + L(S² + SE + S + E).
            # Inverse: H = [pcount - 2VE - V - L(S² + SE + S + E)]
            #              / [L(E + S + 1)].
            b = L * (E + S + 1)
            c = 2 * V * E + V + L * (S * S + S * E + S + E)
        else:  # RNNLangModelHyperparam.rnn_layer_count
            # P(L) = [S² + S(H + E + 1) + EH + H + E]L + 2VE + V.
            # Inverse: L = (pcount - 2VE - V)
            #              / [S² + S(H + E + 1) + EH + H + E].
            b = S * S + S * (H + E + 1) + E * H + H + E
            c = 2 * V * E + V

        # Integer arithmetic avoids rounding errors for large budgets. When the
        # unconstrained inverse is negative, the minimum valid size is optimal.
        remaining = max(0, pcount - c)
        lower = max(
            1,
            (isqrt(b * b + 4 * a * remaining) - b) // (2 * a) if a else remaining // b,
        )
        # P is strictly increasing for positive sizes: only the two integers
        # surrounding the inverse can minimize |P(x) - pcount|.
        size = min(
            (lower, lower + 1),
            key=lambda x: (abs(a * x * x + b * x + c - pcount), x),
        )
        scaled = type(self)(
            size if var is RNNLangModelHyperparam.state_size else S,
            V,
            dropout=self.dropout,
            vocab_embedding_size=(
                size if var is RNNLangModelHyperparam.vocab_embedding_size else E
            ),
            hidden_size=size if var is RNNLangModelHyperparam.hidden_size else H,
            layer_count=size if var is RNNLangModelHyperparam.rnn_layer_count else L,
        )
        reference = next(self.parameters())
        scaled.to(device=reference.device, dtype=reference.dtype)
        scaled.train(self.training)
        return scaled


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
