import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, n_embed, n_head, dtype=torch.float32, causal=True, dropout_rate=0.1):
        super().__init__()
        self.head_size = head_size
        self.n_head = n_head
        self.causal = causal
        self.qkv = nn.Linear(n_embed, 3 * n_head * head_size, bias=False, dtype=dtype)
        self.proj = nn.Linear(n_head * head_size, n_embed, dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention_dropout_rate = dropout_rate
    
    def forward(self, x, use_cache = None, position_ids=None):
        # x shape = (B, T, n_embed)
        batch_size, sequence_length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch_size, sequence_length, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.n_head, self.head_size).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=self.causal,
            dropout_p=self.attention_dropout_rate if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(batch_size, sequence_length, -1)
        out = self.dropout(out)
        return self.proj(out) # (B, T, n_embed)

class SingleSelfAttentionHead(nn.Module):
    # #key, query, value
    # k = self.key(tkn_embs)   # (B, T, head_size)
    # q = self.query(tkn_embs) # (B, T, head_size)
    # v = self.value(tkn_embs) # (B, T, head_size)

    # mask = self.mask_tril[:T, :T] # (T, T)
    
    # #wei shape = (T, T)
    # wei = (q @ k.transpose(-2, -1)) * self.head_size**-0.5 # (B, T, T)

    # wei = wei.masked_fill(~mask, float('-inf'))
    # wei = F.softmax(wei, dim=-1)

    # #atten shape = (B, T, n_embed)
    # atten =  wei @ v
    def __init__(self, intit_block_size, head_size, n_embed, dropout_rate=0.1, dtype=torch.float32, causal=True):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False, dtype=dtype)
        self.query = nn.Linear(n_embed, head_size, bias=False, dtype=dtype)
        self.value = nn.Linear(n_embed, head_size, bias=False, dtype=dtype)
        self.layer_dropout = nn.Dropout(dropout_rate)

        self.register_buffer(
            "mask_tril",
            torch.tril(torch.ones(intit_block_size, intit_block_size, dtype=torch.bool)),
            persistent=False,
        )
        self.causal = causal

    def forward(self, x, use_cache = None, position_ids=None):
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, T)

        if self.causal:
            T = wei.shape[-1]
            if T > self.mask_tril.shape[0]:
                print(f"Block size increase... {self.mask_tril.shape[0]} -> {self.mask_tril.shape[0] * 2}")
                self.register_buffer(
                    "mask_tril",
                    torch.tril(torch.ones(self.mask_tril.shape[0] * 2, self.mask_tril.shape[0] * 2, dtype=torch.bool)),
                    persistent=False,
                )
            mask = self.mask_tril[:T, :T]
            wei = wei.masked_fill(~mask, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.layer_dropout(wei)

        return wei @ v


#Class for computing KV with caching
class KV_Computer:
    def __init__(self, max_block_size, head_size, n_embed, n_head, dtype=torch.float32, device="cpu"):
        self.max_block_size = max_block_size
        self.head_size = head_size
        self.n_embed = n_embed
        self.dtype = dtype

    