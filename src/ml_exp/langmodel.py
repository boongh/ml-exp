import torch
from torch import nn
import torch.nn.functional as F
from .pos_embs import SineusoidalPositionalEncoding, SimplePositionalEncoding
from .mlp import MLP_Full, MLP_Lowrank
from .attention import MultiHeadAttention

# Dimensions: (B, T, n_embed) in and out, where B = batch size, T = sequence length, n_embed = embedding dimension
class TransformerBlock(nn.Module):
    def __init__(self, head_size, n_embed, n_head, rank_devision = 1, dtype=torch.float32, causal=True, dropout_rate=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(head_size, n_embed, n_head, dtype=dtype, causal=causal, dropout_rate=dropout_rate)
        self.ffwdn = MLP_Full(n_embed, dtype=dtype, dropout_rate=dropout_rate, activation=nn.GELU())
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

class LangModel(nn.Module):
    def __init__(self, vocab_size, block_size, head_size=16, n_embed=32, transformer_blocks=1, n_head=4, hidden_layer_size=256, rank_devision=1, dtype=torch.float32, causal=True, dropout_rate=0.1):
        super().__init__()
        self.block_size = block_size

        # Define model layers
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed, dtype=dtype)
        # self.position_embedding_table = nn.Embedding(block_size, n_embed, dtype=dtype)
        self.positional_encoding = SineusoidalPositionalEncoding(n_embed, dtype=dtype, max_len=block_size)

        self.hiddenlayer_size = hidden_layer_size
        transformer_chain = []

        for _ in range(transformer_blocks):
            transformer_chain.append(TransformerBlock(head_size, self.hiddenlayer_size, n_head, rank_devision=rank_devision, dtype=dtype, causal=causal, dropout_rate=dropout_rate))
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