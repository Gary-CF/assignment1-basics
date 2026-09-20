"""Reproduce the assignment's parameter/FLOP/memory arithmetic without a GPU."""
import json
from fractions import Fraction


def account(vocab_size, context_length, num_layers, d_model, num_heads, d_ff):
    V, T, L, D, H, F = vocab_size, context_length, num_layers, d_model, num_heads, d_ff
    assert D % H == 0
    parameters = {
        "token_embeddings": V * D,
        "lm_head": V * D,
        "attention_projections": 4 * L * D * D,
        "ffn": 3 * L * D * F,
        "rmsnorm": (2 * L + 1) * D,
    }
    flops = {
        "attention_projections": 8 * L * T * D * D,
        "attention_scores_and_values": 4 * L * T * T * D,
        "ffn": 6 * L * T * D * F,
        "lm_head": 2 * T * D * V,
    }
    P, forward = sum(parameters.values()), sum(flops.values())
    return {
        "config": dict(V=V, T=T, L=L, D=D, H=H, F=F),
        "parameters_by_component": parameters, "parameters": P,
        "parameter_bytes_fp32": 4 * P, "parameter_GiB_fp32": 4 * P / 2**30,
        "forward_flops_by_component": flops, "forward_flops": forward,
        "forward_percentages": {key: 100 * value / forward for key, value in flops.items()},
    }


def main():
    models = {"tinystories": account(10000, 256, 4, 512, 16, 1344)}
    for name, L, D, H in (("small", 12, 768, 12), ("medium", 24, 1024, 16),
                           ("large", 36, 1280, 20), ("xl", 48, 1600, 25)):
        F = 64 * round(Fraction(8 * D, 3 * 64))
        models[name] = account(50257, 1024, L, D, H, F)
    models["xl_long"] = account(50257, 16384, 48, 1600, 25, 4288)

    # AdamW accounting explicitly assumes F=8D/3, not the rounded F=4288 above.
    V, T, L, D, H = 50257, 1024, 48, 1600, 25
    P = 2 * V * D + L * (12 * D**2 + 2 * D) + D
    A_per_batch = L * (Fraction(56, 3) * T * D + 2 * H * T**2) + T * D + 2 * T * V
    activation_bytes = int(4 * A_per_batch)
    static_bytes = 16 * P
    memory = {
        "assumption": "All FP32; retain one output per listed operation; one BTV tensor for cross-entropy",
        "parameters": P, "parameter_bytes": 4 * P, "gradient_bytes": 4 * P,
        "optimizer_state_bytes": 8 * P, "static_bytes": static_bytes,
        "activation_bytes_per_batch_item": activation_bytes,
        "max_batch_80GB_decimal": (80_000_000_000 - static_bytes) // activation_bytes,
        "max_batch_80GiB": (80 * 2**30 - static_bytes) // activation_bytes,
    }
    # Use the rounded GPT-2 XL architecture from transformer_accounting for this estimate.
    total_training_flops = 3 * models["xl"]["forward_flops"] * 1024 * 400_000
    time_estimate = {
        "forward_flops_per_sequence": models["xl"]["forward_flops"],
        "batch_size": 1024, "steps": 400_000,
        "assumed_effective_flops_per_second": 495e12 * 0.5,
        "training_flops_excluding_adamw": total_training_flops,
        "hours": total_training_flops / (495e12 * 0.5) / 3600,
    }
    print(json.dumps({"models": models, "adamw_memory": memory, "h100_time": time_estimate}, indent=2))


if __name__ == "__main__":
    main()
