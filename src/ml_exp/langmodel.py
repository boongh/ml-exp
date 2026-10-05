import torch
from torch import nn
import torch.nn.functional as F
from .pos_embs import SineusoidalPositionalEncoding, SimplePositionalEncoding
from .mlp import MLP_Full, MLP_Lowrank
from .attention import MultiHeadAttention

# Dimensions: (B, T, n_embed) in and out, where B = batch size, T = sequence length, n_embed = embedding dimension
class TransformerBlock(nn.Module):
    def __init__(self, head_size, n_embed, n_head, mlp_rank = 1, dtype=torch.float32, causal=True, dropout_rate=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(head_size, n_embed, n_head, dtype=dtype, causal=causal, dropout_rate=dropout_rate)
        self.ffwdn = MLP_Lowrank(n_embed, mlp_rank, dtype=dtype, dropout_rate=dropout_rate, activation=nn.GELU())
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
#! TODO: Need a way to fix total parameter count in a transformer block to stay as consistent as possible. 
#! Internal projection layers increases parameter count, haven't proved yet that it has no effect on performance, but it is a concern.
# Dimensions: (B, T, n_embed) in and out, where B = batch size, T = sequence length, n_embed = embedding dimension
class FixedSizeTransformerBlock(nn.Module):
    def __init__(self, head_size, n_embed, n_head, mlp_rank = 1, dtype=torch.float32, causal=True, dropout_rate=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(head_size, n_embed, n_head, dtype=dtype, causal=causal, dropout_rate=dropout_rate)
        self.ffwdn = MLP_Lowrank(n_embed, mlp_rank, dtype=dtype, dropout_rate=dropout_rate, activation=nn.GELU())
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


## Test effect of increasing MLP parameter count and hidden layer without non-linearities in between
class MultiWidthLayerLinearTransformerBlock(nn.Module):
    def __init__(self, head_size, input_n_embed, hidden_n_embed, n_head, mlp_rank = 1, dtype=torch.float32, causal=True, dropout_rate=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(head_size, input_n_embed, n_head, dtype=dtype, causal=causal, dropout_rate=dropout_rate)

        # Previous experiments suggests that stacking affine layers without intervening non-linearities does not improve performance
        # So we are slightly confident that this approach of reducing the MLP rank will not be able to match the performance of a full-rank MLP with the same parameter count
        # This has to be done before attention since increasing attention dimension will have significant effect on performance, and we want to isolate the effect of MLP rank reduction on performance
        self.internal_projection_in = nn.Linear(input_n_embed, hidden_n_embed, dtype=dtype)
        self.internal_projection_out = nn.Linear(hidden_n_embed, input_n_embed, dtype=dtype)

        self.ffwdn = MLP_Full(input_n_embed, dtype=dtype, dropout_rate=dropout_rate, activation=nn.GELU())
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

class LangModel(nn.Module):
    def __init__(self, vocab_size, block_size, head_size=16, n_embed=32, transformer_blocks=1, n_head=4, hidden_layer_size=256, mlp_rank=1, dtype=torch.float32, causal=True, dropout_rate=0.1):
        super().__init__()
        self.block_size = block_size

        # Define model layers
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed, dtype=dtype)
        # self.position_embedding_table = nn.Embedding(block_size, n_embed, dtype=dtype)
        self.positional_encoding = SineusoidalPositionalEncoding(n_embed, dtype=dtype, max_len=block_size)

        self.hiddenlayer_size = hidden_layer_size
        transformer_chain = []

        for _ in range(transformer_blocks):
            transformer_chain.append(MultiWidthLayerLinearTransformerBlock(head_size, self.hiddenlayer_size, n_head, mlp_rank=mlp_rank, dtype=dtype, causal=causal, dropout_rate=dropout_rate))
            transformer_chain.append(SineusoidalPositionalEncoding(self.hiddenlayer_size, dtype=dtype, max_len=block_size))

        #### Transformer Block
        self.transformer_blocks = nn.Sequential(
            nn.Linear(n_embed, self.hiddenlayer_size, dtype=dtype),
            *transformer_chain,
        )

        # Projection layer to vocab size
        self.lm_head = nn.Linear(self.hiddenlayer_size, vocab_size, dtype=dtype)


    def forward(self, idx, targets = None):

        #idx shape = (B, T)
        B, T = idx.shape

        #tokens shape = (B, T, n_embed)
        tkn_embs = self.token_embedding_table(idx)
        #pos_embs = self.position_embedding_table(torch.arange(T, device=idx.device))
        tkn_embs = self.positional_encoding(tkn_embs)

        #Self-attention
        # # atten shape = (B, T, n_embed)
        # atten = self.self_attention_head(tkn_embs)

        # #FFN
        # ffwdn = self.ffwdn(atten)
        
        #Transformer blocks
        ffwdn = self.transformer_blocks(tkn_embs)

        #logits shape = (B, T, vocab_size)
        logits = self.lm_head(ffwdn)

        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(
                logits_flat, 
                targets_flat
                )
        else:
            loss = None

        return logits, loss

    def generation(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
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