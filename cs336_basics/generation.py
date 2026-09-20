import math
from contextlib import nullcontext

import torch

from cs336_basics.layers import softmax


@torch.no_grad()
def sample_next_token(logits, temperature=1.0, top_p=1.0, generator=None):
    """Map (batch, vocab) logits to (batch, 1) next-token IDs."""
    if logits.ndim != 2 or logits.shape[-1] == 0:
        raise ValueError("Expected nonempty (batch, vocab) logits")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    # Subtract first so a very small temperature cannot overflow positive logits.
    shifted = logits.float() - logits.float().max(dim=-1, keepdim=True).values
    probabilities = softmax(shifted / temperature, dim=-1)
    if top_p == 1:
        return torch.multinomial(probabilities, 1, generator=generator)

    sorted_probs, sorted_ids = probabilities.sort(dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    previous_mass = torch.cat((torch.zeros_like(cumulative[:, :1]), cumulative[:, :-1]), dim=-1)
    keep = previous_mass < top_p
    filtered = sorted_probs * keep
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    sampled_rank = torch.multinomial(filtered, 1, generator=generator)
    return sorted_ids.gather(-1, sampled_rank)


@torch.no_grad()
def generate(
    model, prompt_ids, max_new_tokens=256, temperature=0.8, top_p=0.95,
    eos_token_id=None, seed=336, precision="fp32",
):
    """Return prompt plus completion as a list of IDs for one prompt."""
    if max_new_tokens < 0 or precision not in ("fp32", "bf16"):
        raise ValueError("Invalid max_new_tokens or precision")
    if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("Invalid temperature or top_p")
    device = next(model.parameters()).device
    ids = torch.as_tensor(prompt_ids, dtype=torch.long, device=device)
    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError("The prompt must be a nonempty 1D sequence of token IDs")
    ids = ids.unsqueeze(0)
    generator = torch.Generator(device=device).manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            context = ids[:, -model.context_length:]
            amp = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
            with amp:
                logits = model(context)[:, -1, :]
            next_id = sample_next_token(logits, temperature, top_p, generator)
            ids = torch.cat((ids, next_id), dim=-1)
            if eos_token_id is not None and next_id.item() == eos_token_id:
                break
    finally:
        model.train(was_training)
    return ids[0].tolist()
