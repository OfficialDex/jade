import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class rmsnorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def rope_frequencies(head_dim, context, theta, device):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(context, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[: x.shape[-2]].unsqueeze(0).unsqueeze(0)
    sin = sin[: x.shape[-2]].unsqueeze(0).unsqueeze(0)
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config["heads"]
        self.kv_heads = config["kv_heads"]
        self.head_dim = config["hidden_size"] // config["heads"]
        self.query = nn.Linear(config["hidden_size"], self.heads * self.head_dim, bias=False)
        self.key = nn.Linear(config["hidden_size"], self.kv_heads * self.head_dim, bias=False)
        self.value = nn.Linear(config["hidden_size"], self.kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(self.heads * self.head_dim, config["hidden_size"], bias=False)

    def forward(self, x, cos, sin):
        batch, seq, _ = x.shape
        q = self.query(x).view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.kv_heads != self.heads:
            repeat = self.heads // self.kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(batch, seq, self.heads * self.head_dim)
        return self.out(out)


class feedforward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.up = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.down = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = rmsnorm(config["hidden_size"])
        self.attention = attention(config)
        self.feedforward_norm = rmsnorm(config["hidden_size"])
        self.feedforward = feedforward(config)

    def forward(self, x, cos, sin):
        x = x + self.attention(self.attention_norm(x), cos, sin)
        x = x + self.feedforward(self.feedforward_norm(x))
        return x


class jade(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.blocks = nn.ModuleList([block(config) for _ in range(config["layers"])])
        self.norm = rmsnorm(config["hidden_size"])
        self.head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        self.head.weight = self.embed.weight

        head_dim = config["hidden_size"] // config["heads"]
        cos, sin = rope_frequencies(head_dim, config["context"], config["rope_theta"], device="cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config["layers"]))
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids, targets=None):
        x = self.embed(ids)
        for layer in self.blocks:
            x = layer(x, self.rope_cos, self.rope_sin)
        x = self.norm(x)
        logits = self.head(x)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

