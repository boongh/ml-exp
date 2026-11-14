from enum import Enum
from typing import Self

import torch
import torch.nn.functional as F
from torch import nn

from ml_exp.experiments.base_experimental import BaseExperimentalModel
from ml_exp.transformer.attention import MultiHeadAttention
from ml_exp.transformer.mlp import MLP_fixed, MLP_Full
from ml_exp.transformer.pos_embs import SineusoidalPositionalEncoding


# Dimensions: (B, T, n_embed) in and out, where B = batch size, T = sequence length, n_embed = embedding dimension
class TransformerBlock(nn.Module):
    def __init__(
        self,
        head_size,
        n_embed,
        n_head,
        mlp_rank=1,
        dtype=torch.float32,
        causal=True,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            head_size,
            n_embed,
            n_head,
            dtype=dtype,
            causal=causal,
            dropout_rate=dropout_rate,
        )
        self.ffwdn = MLP_Full(
            n_embed, dtype=dtype, dropout_rate=dropout_rate, activation=nn.GELU()
        )
        self.layer_norm1 = nn.LayerNorm(n_embed, dtype=dtype)
        self.layer_norm2 = nn.LayerNorm(n_embed, dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(x)
        # x + self.attention() = residual connectiong
        # Norm before sublayer -> Pre norm formulation
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.ffwdn(self.layer_norm2(x))
        return x


# Transformer block with fixed MLP parameter count
# It calculates internal n_embed based on given mlp_rank
# so that the parameter count in the MLP is fixed to baseline_n_embed ** 2
# where baseline_n_embed is the embedding dimension of a baseline model with
# full-rank MLP
# Dimensions: (B, T, n_embed) in and out, where B = batch size, T = sequence length, n_embed = embedding dimension
class FixedSizeTransformerBlock(nn.Module):
    def __init__(
        self,
        head_size,
        n_embed,
        n_head,
        mlp_rank=None,
        baseline_hidden=None,
        dtype=torch.float32,
        causal=True,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            head_size,
            n_embed,
            n_head,
            dtype=dtype,
            causal=causal,
            dropout_rate=dropout_rate,
        )
        self.ffwdn = MLP_fixed(
            n_embed,
            mlp_rank,
            dtype=dtype,
            dropout_rate=dropout_rate,
            activation=nn.GELU(),
            baseline_hidden=baseline_hidden,
        )
        self.layer_norm1 = nn.LayerNorm(n_embed, dtype=dtype)
        self.layer_norm2 = nn.LayerNorm(n_embed, dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x) -> torch.Tensor:
        x = self.dropout(x)
        # x + self.attention() = residual connectiong
        # Norm before sublayer -> Pre norm formulation
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.ffwdn(self.layer_norm2(x))
        return x


## Test effect of increasing MLP parameter count and hidden layer without non-linearities in between
class MultiWidthLayerLinearTransformerBlock(nn.Module):
    def __init__(
        self,
        head_size,
        input_n_embed,
        hidden_n_embed,
        n_head,
        dtype=torch.float32,
        causal=True,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            head_size,
            input_n_embed,
            n_head,
            dtype=dtype,
            causal=causal,
            dropout_rate=dropout_rate,
        )

        # Previous experiments suggests that stacking affine layers without intervening non-linearities does not improve performance
        # So we are slightly confident that this approach of reducing the MLP rank will not be able to match the performance of a full-rank MLP with the same parameter count
        # This has to be done before attention since increasing attention dimension will have significant effect on performance, and we want to isolate the effect of MLP rank reduction on performance
        self.internal_projection_in = nn.Linear(
            input_n_embed, hidden_n_embed, dtype=dtype
        )
        self.internal_projection_out = nn.Linear(
            hidden_n_embed, input_n_embed, dtype=dtype
        )

        self.ffwdn = MLP_Full(
            input_n_embed, dtype=dtype, dropout_rate=dropout_rate, activation=nn.GELU()
        )
        self.layer_norm1 = nn.LayerNorm(input_n_embed, dtype=dtype)
        self.layer_norm2 = nn.LayerNorm(input_n_embed, dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(x)
        # x + self.attention() = residual connectiong
        # Norm before sublayer -> Pre norm formulation
        x = x + self.attention(self.layer_norm1(x))

        # Hidden projection to test with artificially increasing parameter count
        # while keeping the affine layer linear
        t = self.internal_projection_in(self.layer_norm2(x))
        t = self.internal_projection_out(t)

        x = x + self.ffwdn(t)
        return x


class LangModelHyperparam(Enum):
    head_size = "HEAD_SIZE"
    n_embed = "N_EMBED"
    baseline_hidden = "BASELINE_HIDDEN"
    transformer_blocks = "TRANSFORMER_BLOCKS"
    n_head = "N_HEAD"
    mlp_rank = "MLP_RANK"


class LangModel(BaseExperimentalModel[LangModelHyperparam]):
    variables = LangModelHyperparam

    def __init__(
        self,
        vocab_size,
        block_size,
        head_size=16,
        n_embed=32,
        baseline_hidden=None,
        transformer_blocks=1,
        n_head=4,
        mlp_rank=32,
        dtype=torch.float32,
        causal=True,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.input_n_embed = n_embed
        # Resolve the default once so varying n_embed holds MLP baseline width
        # fixed, rather than silently scaling a second experimental variable.
        self._scaling_config = dict(
            vocab_size=vocab_size,
            block_size=block_size,
            head_size=head_size,
            n_embed=n_embed,
            baseline_hidden=n_embed if baseline_hidden is None else baseline_hidden,
            transformer_blocks=transformer_blocks,
            n_head=n_head,
            mlp_rank=mlp_rank,
            causal=causal,
            dropout_rate=dropout_rate,
        )

        # Define model layers
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed, dtype=dtype)
        # self.position_embedding_table = nn.Embedding(block_size, n_embed, dtype=dtype)
        self.positional_encoding = SineusoidalPositionalEncoding(
            n_embed, dtype=dtype, max_len=block_size
        )

        transformer_chain = []

        for _ in range(transformer_blocks):
            transformer_chain.append(
                FixedSizeTransformerBlock(
                    head_size,
                    n_embed,
                    n_head,
                    mlp_rank=mlp_rank,
                    baseline_hidden=baseline_hidden,
                    dtype=dtype,
                    causal=causal,
                    dropout_rate=dropout_rate,
                )
            )
            transformer_chain.append(
                SineusoidalPositionalEncoding(
                    self.input_n_embed, dtype=dtype, max_len=block_size
                )
            )

        #### Transformer Block
        self.transformer_blocks = nn.Sequential(
            *transformer_chain,
        )

        # Projection layer to vocab size
        self.lm_head = nn.Linear(self.input_n_embed, vocab_size, dtype=dtype)

    def copy_scaled_to_fixed_p(self, var: LangModelHyperparam, pcount: int) -> Self:
        """Freshly initialize a copy with only the selected dimension changed.

        Minimize absolute parameter-count error; ties use the smaller dimension.
        As in RNNLangModel, the closest count may exceed the target. Preserve
        device, dtype and training mode, but not weights. Embedding widths must
        be even for the existing sinusoidal encoding. Rank scaling retains the
        low-rank architecture; mlp_rank=None therefore cannot be rank-scaled.
        """
        if isinstance(pcount, bool) or not isinstance(pcount, int) or pcount <= 0:
            raise ValueError("pcount must be a positive integer")
        if not isinstance(var, LangModelHyperparam):
            raise ValueError(f"Unsupported scaling variable: {var!r}")

        config = self._scaling_config.copy()
        V, E, H, L, K, N, R = (
            config[key] for key in (
                "vocab_size", "n_embed", "baseline_hidden", "transformer_blocks",
                "head_size", "n_head", "mlp_rank",
            )
        )
        if min(V, E, H, L, K, N) < 1 or (R is not None and R < 1):
            raise ValueError("Scaling requires positive dimensions and block count")

        # V=vocabulary, E=embedding, H=baseline hidden, L=blocks,
        # K=head size, N=head count, R=rank, h=actual low-rank hidden width.
        # Embedding/head: 2VE+V. Initial positional gamma: E.
        # Per block: attention 4EKN+E, LayerNorms 4E, positional gamma E,
        # MLP M=(2E+1)H+E (full) or 2R(E+1)+(2R+1)h+E (low-rank),
        # h=floor(((2E+1)H-2R(E+1))/(2R+1)). Thus:
        # P=2VE+V+E+L(4EKN+6E+M). All biases/gammas are included.
        def parameter_count(size):
            values = config | {var.name: size}
            e, h, l, k, n, r = (
                values[key] for key in (
                    "n_embed", "baseline_hidden", "transformer_blocks",
                    "head_size", "n_head", "mlp_rank",
                )
            )
            if r is None:
                mlp = (2 * e + 1) * h + e
            else:
                inner = ((2 * e + 1) * h - 2 * r * (e + 1)) // (2 * r + 1)
                mlp = 2 * r * (e + 1) + (2 * r + 1) * inner + e
            return 2 * V * e + V + e + l * (4 * e * k * n + 6 * e + mlp)

        # For the inverses below, C=2VE+V+E+L(4EKN+7E).
        minimum, step = 1, 1
        if var is LangModelHyperparam.head_size:
            # P(K)=4ELNK+c; inverse K=(pcount-c)/(4ELN).
            # c=2VE+V+E+L(6E+M).
            pass
        elif var is LangModelHyperparam.n_head:
            # P(N)=4ELKN+c; inverse N=(pcount-c)/(4ELK).
            # c=2VE+V+E+L(6E+M).
            pass
        elif var is LangModelHyperparam.transformer_blocks:
            # P(L)=(4EKN+6E+M)L+2VE+V+E.
            # Inverse L=(pcount-2VE-V-E)/(4EKN+6E+M).
            pass
        elif var is LangModelHyperparam.n_embed:
            # Full: P(E)=bE+c, b=2V+1+L(4KN+7+2H), c=V+LH.
            # Inverse E=(pcount-c)/b.
            # Low-rank, on a fixed-h plateau: b=2V+1+L(4KN+7+2R),
            # c=V+L(2R+(2R+1)h); inverse E=(pcount-c)/b.
            # h changes at floor boundaries; search the exact monotone count.
            minimum, step = 2, 2
            if R is not None:
                # Require h>=1: 2E(H-R)>=4R+1-H.
                if H <= R:
                    raise ValueError("No valid embedding width at this fixed rank")
                denominator = 2 * (H - R)
                minimum = max(2, (4 * R + 1 - H + denominator - 1) // denominator)
                minimum += minimum % 2
        elif var is LangModelHyperparam.baseline_hidden:
            # Full: P(H)=C+L(2E+1)H; inverse H=(pcount-C)/(L(2E+1)).
            # Low-rank: P(H)=C+L[2R(E+1)+(2R+1)floor(aH-d)],
            # a=(2E+1)/(2R+1), d=2R(E+1)/(2R+1).
            # Inverse plateau: h=(pcount-C-2LR(E+1))/(L(2R+1)),
            # then smallest H=ceil(((2R+1)h+2R(E+1))/(2E+1)).
            if R is not None:
                minimum = (2 * R * (E + 1) + 2 * R + 1 + 2 * E) // (2 * E + 1)
        else:  # LangModelHyperparam.mlp_rank
            # P(R)=C+L[2R(E+1)+(2R+1)h(R)]. For fixed h:
            # inverse R=(pcount-C-Lh)/(2L(E+h+1)).
            # h(R) decreases in steps, making P non-monotone. Check every
            # valid positive rank; h>=1 gives R<=((2E+1)H-1)/(2E+4).
            if R is None:
                raise ValueError("mlp_rank scaling requires a low-rank source model")
            maximum = ((2 * E + 1) * H - 1) // (2 * E + 4)
            if maximum < 1:
                raise ValueError("No valid positive MLP rank")
            size = min(
                range(1, maximum + 1),
                key=lambda x: (abs(parameter_count(x) - pcount), x),
            )

        if var is not LangModelHyperparam.mlp_rank:
            # Integer bisection implements the inverses without floating-point
            # rounding and handles MLP floor boundaries exactly. Find the first
            # admissible size whose count reaches the target.
            lo, hi = 0, 1
            while parameter_count(minimum + hi * step) < pcount:
                hi *= 2
            while lo < hi:
                mid = (lo + hi) // 2
                if parameter_count(minimum + mid * step) < pcount:
                    lo = mid + 1
                else:
                    hi = mid
            candidates = (minimum + max(0, lo - 1) * step, minimum + lo * step)
            size = min(candidates, key=lambda x: (abs(parameter_count(x) - pcount), x))
            # A low-rank baseline width can have equal-count plateaus. Return
            # the smallest width on the winning plateau, including below target.
            target = parameter_count(size)
            lo, hi = 0, (size - minimum) // step
            while lo < hi:
                mid = (lo + hi) // 2
                if parameter_count(minimum + mid * step) < target:
                    lo = mid + 1
                else:
                    hi = mid
            size = minimum + lo * step

        config[var.name] = size
        reference = next(self.parameters())
        scaled = type(self)(**config, dtype=reference.dtype)
        scaled.to(device=reference.device, dtype=reference.dtype)
        scaled.train(self.training)
        return scaled

    def forward(self, idx, targets=None):

        # idx shape = (B, T)
        B, T = idx.shape

        # tokens shape = (B, T, n_embed)
        tkn_embs = self.token_embedding_table(idx)
        # pos_embs = self.position_embedding_table(torch.arange(T, device=idx.device))
        tkn_embs = self.positional_encoding(tkn_embs)

        # Self-attention
        # # atten shape = (B, T, n_embed)
        # atten = self.self_attention_head(tkn_embs)

        # #FFN
        # ffwdn = self.ffwdn(atten)

        # Transformer blocks
        ffwdn = self.transformer_blocks(tkn_embs)

        # logits shape = (B, T, vocab_size)
        logits = self.lm_head(ffwdn)

        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        else:
            loss = None

        return logits, loss

    def generation(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        # for _ in range(max_new_tokens):
        #     logits, _ = self(idx)
        #     logits = logits[:, -1, :]
        #     probs = F.softmax(logits, dim=-1)
        #     idx_next = torch.multinomial(probs, num_samples=1)
        #     idx = torch.cat((idx, idx_next), dim=1)

        return idx
