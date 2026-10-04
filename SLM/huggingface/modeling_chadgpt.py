"""Inference architecture for ChadGPT; derived from SLM/chadgpt.py."""

import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x.pow(3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg["emb_dim"]
        self.layers = nn.Sequential(nn.Linear(dim, 4 * dim), GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        return self.layers(x)


class RoPEEmbedding(nn.Module):
    def __init__(self, emb_dim, n_heads, pi_scale, base=10000):
        super().__init__()
        head_dim = emb_dim // n_heads
        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("theta", theta)
        self.pi_scale = pi_scale

    def forward(self, x, start_pos=0):
        t = torch.arange(start_pos, start_pos + x.shape[-2], device=x.device).float()
        freqs = ((t * self.pi_scale).unsqueeze(1) * self.theta).unsqueeze(0).unsqueeze(0)
        cos = freqs.cos().repeat_interleave(2, dim=-1).to(x.dtype)
        sin = freqs.sin().repeat_interleave(2, dim=-1).to(x.dtype)
        rotated = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).flatten(-2)
        return x * cos + rotated * sin


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg["emb_dim"]
        self.n_heads = cfg["n_heads"]
        self.n_kv_heads = cfg["n_kv_heads"]
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = dim // self.n_heads
        self.W_query = nn.Linear(dim, dim, bias=cfg["qkv_bias"])
        self.W_key = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=cfg["qkv_bias"])
        self.W_value = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=cfg["qkv_bias"])
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.rope = RoPEEmbedding(dim, self.n_heads, cfg["pi_scale"], cfg["rope_base"])

    def forward(self, x, past_kv=None, start_pos=0):
        batch, length, dim = x.shape
        q = self.W_query(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, start_pos), self.rope(k, start_pos)
        past_length = 0
        if past_kv is not None:
            past_length = past_kv[0].shape[-2]
            k = torch.cat([past_kv[0], k], dim=-2)
            v = torch.cat([past_kv[1], v], dim=-2)
        new_kv = (k, v)
        k, v = k.repeat_interleave(self.n_rep, dim=1), v.repeat_interleave(self.n_rep, dim=1)
        mask = None
        # Cached chunks need their causal mask offset by the existing cache length.
        if past_length and length > 1:
            queries = torch.arange(length, device=x.device) + past_length
            keys = torch.arange(k.shape[-2], device=x.device)
            mask = keys.unsqueeze(0) <= queries.unsqueeze(1)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=length > 1 and past_length == 0,
        )
        out = out.transpose(1, 2).contiguous().view(batch, length, dim)
        return self.out_proj(out), new_kv


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(cfg)
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, past_kv=None, start_pos=0):
        attn, new_kv = self.att(self.norm1(x), past_kv, start_pos)
        x = x + self.drop_shortcut(attn)
        return x + self.drop_shortcut(self.ff(self.norm2(x))), new_kv


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])

    def forward(self, in_idx, past_key_values=None, start_pos=None):
        cached_length = past_key_values[0][0].shape[-2] if past_key_values is not None else 0
        start_pos = cached_length if start_pos is None else start_pos
        if start_pos < 0 or start_pos + in_idx.shape[1] > self.config["context_length"]:
            raise ValueError("Input and cached tokens exceed the 4096-token context window.")
        x = self.drop_emb(self.tok_emb(in_idx))
        new_key_values = []
        for i, block in enumerate(self.trf_blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, layer_kv = block(x, layer_past, start_pos)
            new_key_values.append(layer_kv)
        # Input embeddings and output projection share these exact weights.
        return self.final_norm(x) @ self.tok_emb.weight.T, new_key_values
