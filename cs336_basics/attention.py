import torch
from torch import nn

from cs336_basics.layers import Linear, linear, scaled_dot_product_attention
from cs336_basics.rope import RotaryPositionalEmbedding


def multihead_self_attention(
    x, q_weight, k_weight, v_weight, o_weight,
    num_heads, rope=None, token_positions=None,
):
    d_model = x.shape[-1]
    if num_heads <= 0 or d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by a positive num_heads")
    d_head = d_model // num_heads
    seq_len = x.shape[-2]

    def split_heads(projected):
        return projected.reshape(
            *projected.shape[:-1], num_heads, d_head
        ).transpose(-3, -2)

    q = split_heads(linear(x, q_weight))
    k = split_heads(linear(x, k_weight))
    v = split_heads(linear(x, v_weight))

    if rope is not None:
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        head_positions = token_positions.unsqueeze(-2)
        q = rope(q, head_positions)
        k = rope(k, head_positions)

    mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
    attended = scaled_dot_product_attention(q, k, v, mask)
    merged = attended.transpose(-3, -2).reshape(*x.shape[:-1], d_model)
    return linear(merged, o_weight)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, d_model, num_heads, max_seq_len=None, theta=None,
        device=None, dtype=None,
    ):
        super().__init__()
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by a positive num_heads")
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = None
        if theta is not None:
            if max_seq_len is None:
                raise ValueError("max_seq_len is required when using RoPE")
            self.rope = RotaryPositionalEmbedding(
                theta, d_model // num_heads, max_seq_len, device=device
            )

    def forward(self, x, token_positions=None):
        return multihead_self_attention(
            x,
            self.q_proj.weight,
            self.k_proj.weight,
            self.v_proj.weight,
            self.output_proj.weight,
            self.num_heads,
            self.rope,
            token_positions,
        )