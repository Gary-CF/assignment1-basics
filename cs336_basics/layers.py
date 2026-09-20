import math

import torch
from torch import nn


def linear(x, weight):
    return x @ weight.T


def embedding(token_ids, weight):
    return weight[token_ids]


def rmsnorm(x, weight, eps=1e-5):
    in_dtype = x.dtype
    x = x.to(torch.float32)
    mean_square = x.square().mean(dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(mean_square + eps)
    return (normalized * weight).to(in_dtype)


def silu(x):
    return x * torch.sigmoid(x)


def swiglu(x, w1, w2, w3):
    gate = silu(linear(x, w1))
    up = linear(x, w3)
    return linear(gate * up, w2)


def softmax(x, dim=-1):
    shifted = x - x.max(dim=dim, keepdim=True).values
    exp_x = shifted.exp()
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    probabilities = softmax(scores, dim=-1)
    return probabilities @ V


class Linear(nn.Module):
    def __init__(self, d_in, d_out, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(d_out, d_in, device=device, dtype=dtype)
        )
        std = math.sqrt(2 / (d_in + d_out))
        nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        return linear(x, self.weight)


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids):
        return embedding(token_ids, self.weight)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        return rmsnorm(x, self.weight, self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        return swiglu(x, self.w1.weight, self.w2.weight, self.w3.weight)