"""Generate a reproducible text sample from a run_train.py checkpoint."""
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import torch

from cs336_basics.generation import generate
from cs336_basics.model import TransformerLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=336)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--out", default="runs/sample.txt")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.device == "cuda" and args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable; use --precision fp32")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer_path = Path(args.tokenizer or checkpoint["config"]["tokenizer"])
    fingerprint = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    if fingerprint != checkpoint["data_signature"]["tokenizer_sha256"]:
        raise ValueError("Tokenizer does not match the training checkpoint")
    with tokenizer_path.open("rb") as stream:
        data = pickle.load(stream)
    # Use the student's already-tested production BPE implementation.
    from cs336_basics.tokenizer import Tokenizer
    tokenizer = Tokenizer(data["vocab"], data["merges"], data.get("special_tokens"))
    eos_ids = [i for i, value in data["vocab"].items() if value == b"<|endoftext|>"]
    if len(eos_ids) != 1:
        raise ValueError("Expected exactly one <|endoftext|> token")
    prompt_ids = list(tokenizer.encode(args.prompt))
    if not prompt_ids:
        raise ValueError("Please provide a nonempty prompt")
    model = TransformerLM(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)
    iteration = checkpoint["iteration"]
    model_config = checkpoint["model_config"]
    del checkpoint

    started = time.perf_counter()
    ids = generate(model, prompt_ids, args.max_new_tokens, args.temperature, args.top_p,
                   eos_ids[0], args.seed, args.precision)
    elapsed = time.perf_counter() - started
    generated_ids = ids[len(prompt_ids):]
    text = tokenizer.decode(ids)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()), "iteration": iteration,
        "model_config": model_config, "tokenizer_sha256": fingerprint,
        "prompt": args.prompt, "temperature": args.temperature, "top_p": args.top_p,
        "seed": args.seed, "device": args.device, "precision": args.precision,
        "torch_version": str(torch.__version__), "max_new_tokens": args.max_new_tokens,
        "generated_tokens": len(generated_ids), "generated_ids": generated_ids,
        "stopped_on_eos": bool(generated_ids and generated_ids[-1] == eos_ids[0]),
        "seconds": elapsed,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(text)
    print(f"\nGenerated {len(generated_ids)} tokens from step {iteration}; saved {output} and {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
