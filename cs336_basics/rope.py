import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError("d_k must be a positive even integer")
        if theta <= 0 or max_seq_len <= 0:
            raise ValueError("theta and max_seq_len must be positive")
        self.d_k = d_k

        pair_ids = torch.arange(d_k // 2, device=device, dtype=torch.float32)
        inv_freq = theta ** (-2 * pair_ids / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = positions[:, None] * inv_freq[None, :]

        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    def forward(self, x, token_positions):
        if x.shape[-1] != self.d_k:
            raise ValueError("The last dimension of x must equal d_k")
        token_positions = token_positions.to(device=x.device)
        cos = self.cos_cache[token_positions].to(dtype=x.dtype)
        sin = self.sin_cache[token_positions].to(dtype=x.dtype)

        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)