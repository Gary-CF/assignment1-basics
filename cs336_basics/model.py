import torch
from torch import nn

from cs336_basics.attention import MultiHeadSelfAttention
from cs336_basics.layers import Embedding, Linear, RMSNorm, SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model, num_heads, d_ff, context_length, rope_theta,
        device=None, dtype=None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads,
            max_seq_len=context_length, theta=rope_theta,
            device=device, dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self, vocab_size, context_length, d_model, num_layers,
        num_heads, d_ff, rope_theta=10000.0, device=None, dtype=None,
    ):
        super().__init__()
        if context_length <= 0 or num_layers <= 0:
            raise ValueError("context_length and num_layers must be positive")
        self.context_length = context_length

        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, context_length, rope_theta,
                device=device, dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids, token_positions=None):
        seq_len = token_ids.shape[-1]
        if not 0 < seq_len <= self.context_length:
            raise ValueError("Sequence length must be between 1 and context_length")
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=token_ids.device)

        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        return self.lm_head(x)