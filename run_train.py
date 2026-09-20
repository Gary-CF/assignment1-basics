"""Train on native-endian uint16 token files; run from the repository root."""
import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from cs336_basics.model import ABLATIONS,TransformerLM
from cs336_basics.training import AdamW, clip_grad_norm, cross_entropy, get_lr_cosine_schedule


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/tinystories_train.bin")
    parser.add_argument("--valid", default="data/tinystories_valid.bin")
    parser.add_argument("--tokenizer", default="data/tinystories_tokenizer.pkl")
    parser.add_argument("--out", default="runs/tinystories")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--ablation", choices=ABLATIONS, default="baseline")
    for name, default in {
        "context-length": 256, "d-model": 512, "num-layers": 4, "num-heads": 16,
        "d-ff": 1344, "micro-batch": 4, "grad-accum": 8, "steps": 40000,
        "warmup": 200, "eval-every": 100, "eval-examples": 80,
        "save-every": 200, "log-every": 10, "seed": 336, "cpu-threads": 4,
    }.items():
        parser.add_argument(f"--{name}", type=int, default=default)
    for name, default in {
        "rope-theta": 10000., "lr": 3e-4, "min-lr": 3e-5,
        "beta1": 0.9, "beta2": 0.95, "eps": 1e-8,
        "weight-decay": 0.1, "max-grad-norm": 1.,
    }.items():
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--stop-after", type=int, default=None,
                        help="Stop at this completed step without changing the LR schedule")
    parser.add_argument("--check-data-only", action="store_true")
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="offline")
    parser.add_argument("--wandb-project", default="cs336-a1")
    preliminary, _ = parser.parse_known_args()
    saved = None
    if preliminary.resume:
        saved = torch.load(preliminary.resume, map_location="cpu", weights_only=True)
        parser.set_defaults(**saved["config"])
        parser.set_defaults(resume=None, stop_after=None, check_data_only=False)
    args = parser.parse_args()
    positive = ("context_length", "d_model", "num_layers", "num_heads", "d_ff",
                "micro_batch", "grad_accum", "steps", "eval_every", "eval_examples",
                "save_every", "log_every", "cpu_threads")
    if any(getattr(args, key) <= 0 for key in positive):
        parser.error("Dimensions, batch sizes, step counts, and intervals must be positive")
    if not 0 <= args.warmup < args.steps:
        parser.error("Require 0 <= warmup < steps")
    if args.stop_after is not None and args.stop_after <= 0:
        parser.error("stop-after must be positive")
    if args.d_model % args.num_heads or (args.d_model // args.num_heads) % 2:
        parser.error("d_model must divide evenly into heads of even dimension")    
    if args.ablation == "silu" and args.d_ff != 4 * args.d_model:
        parser.error("--ablation silu requires --d-ff equal to 4 * --d-model")
    
    if saved is not None:
        if args.ablation != saved["config"].get("ablation", "baseline"):
            parser.error("Resume must preserve ablation; use a new run for another architecture")
        mutable = {"out", "resume", "stop_after", "check_data_only", "device", "cpu_threads",
                   "wandb_mode", "wandb_project", "log_every", "save_every", "micro_batch", "grad_accum"}
        changed = [key for key, value in saved["config"].items()
                   if key not in mutable and getattr(args, key) != value]
        old_batch = saved["config"]["micro_batch"] * saved["config"]["grad_accum"]
        if changed or args.micro_batch * args.grad_accum != old_batch:
            parser.error(f"Resume must preserve training settings and effective batch; changed: {changed}")
    return args, saved


def load_data(args):
    with open(args.tokenizer, "rb") as stream:
        tokenizer = pickle.load(stream)
    vocab = tokenizer["vocab"]
    if not vocab or set(vocab) != set(range(len(vocab))) or len(vocab) > 65536:
        raise ValueError("Expected contiguous token IDs fitting uint16")
    if not all(isinstance(value, bytes) for value in vocab.values()):
        raise ValueError("Vocabulary values must be bytes")
    signature = {"tokenizer_sha256": hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest()}
    datasets = []
    for name, filename in (("train", args.train), ("valid", args.valid)):
        path = Path(filename).resolve()
        stat = path.stat()
        if stat.st_size % 2 or stat.st_size // 2 <= args.context_length:
            raise ValueError(f"{path}: invalid uint16 file size or insufficient tokens")
        data = np.memmap(path, dtype=np.uint16, mode="r")
        low, high = 65535, 0
        for start in range(0, len(data), 1_000_000):
            chunk = data[start:start + 1_000_000]
            low, high = min(low, int(chunk.min())), max(high, int(chunk.max()))
        if high >= len(vocab):
            raise ValueError(f"{name}: token {high} is outside vocab_size={len(vocab)}")
        signature[name] = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        print(f"{name}: {len(data):,} tokens, ID range [{low}, {high}]", flush=True)
        print("preview:", b"".join(vocab[int(i)] for i in data[:80]).decode("utf-8", errors="replace"), flush=True)
        datasets.append(data)
    return datasets[0], datasets[1], len(vocab), signature


def sample_batch(data, batch_size, context_length, device, rng):
    # Only materialize the selected windows, never the full memmap.
    starts = rng.integers(0, len(data) - context_length, size=batch_size)
    indices = starts[:, None] + np.arange(context_length + 1)[None, :]
    tokens = np.asarray(data[indices], dtype=np.int64)
    tokens = torch.from_numpy(tokens).to(device)
    return tokens[:, :-1], tokens[:, 1:]


def autocast_context(args):
    if args.precision == "bf16":
        return torch.autocast(device_type=args.device, dtype=torch.bfloat16)
    return nullcontext()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(model, data, args, device):
    # Reset the validation RNG: comparable batches, independent of training RNG.
    rng = np.random.default_rng(args.seed + 1)
    was_training = model.training
    model.eval()
    total_loss = 0.
    try:
        for start in range(0, args.eval_examples, args.micro_batch):
            batch_size = min(args.micro_batch, args.eval_examples - start)
            x, y = sample_batch(data, batch_size, args.context_length, device, rng)
            with autocast_context(args):
                loss = cross_entropy(model(x), y)
            total_loss += float(loss) * batch_size
        value = total_loss / args.eval_examples
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite validation loss")
        return value
    finally:
        model.train(was_training)


def save_run(path, model, optimizer, step, args, model_config, signature, rng, elapsed, best):
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": step,
        "config": vars(args), "model_config": model_config, "data_signature": signature,
        "train_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if args.device == "cuda" else [],
        "elapsed_seconds": elapsed, "best_val_loss": best,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def prepare_log(path, resumed_step):
    if not path.exists():
        return
    if resumed_step is None:
        raise FileExistsError(f"{path} exists; use --resume or a new --out directory")
    # Preserve the old log before discarding records after the restored checkpoint.
    shutil.copy2(path, path.with_name(f"metrics.before-resume-{time.time_ns()}.jsonl"))
    kept = []
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # A killed process can leave a partial final line.
        if record["step"] <= resumed_step:
            kept.append(json.dumps(record))
    path.write_text("\n".join(kept) + ("\n" if kept else ""))


def main():
    args, saved = parse_args()
    torch.set_num_threads(args.cpu_threads)
    train, valid, vocab_size, signature = load_data(args)
    if args.check_data_only:
        return
    if saved is not None and signature != saved["data_signature"]:
        raise ValueError("Data files or tokenizer changed since the checkpoint")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; choose --device cpu --precision fp32")
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 unavailable; choose --precision fp32")
        print("GPU:", torch.cuda.get_device_name(device), flush=True)
    torch.manual_seed(args.seed)
    model_config = dict(
        vocab_size=vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        ablation=args.ablation,
    )
    # FP32 parameters and AdamW moments; autocast only selected forward operations.
    model = TransformerLM(**model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
                      eps=args.eps, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    step, elapsed_before, best = 0, 0., math.inf
    if saved is not None:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["train_rng"]
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        step, elapsed_before, best = saved["iteration"], saved["elapsed_seconds"], saved["best_val_loss"]
    stop = min(args.steps, args.stop_after if args.stop_after is not None else args.steps)
    if stop <= step:
        raise ValueError(f"Already at step {step}; requested stopping step {stop}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "metrics.jsonl"
    prepare_log(log_path, step if saved is not None else None)
    if saved is None and (out / "latest.pt").exists():
        raise FileExistsError("latest.pt already exists; use --resume or a new output directory")
    metadata = {"config": vars(args), "model_config": model_config, "data_signature": signature,
                "torch_version": str(torch.__version__),
                "parameters": sum(p.numel() for p in model.parameters())}
    (out / f"config-from-step-{step}.json").write_text(json.dumps(metadata, indent=2))
    tokens_per_step = args.micro_batch * args.grad_accum * args.context_length
    print(f"parameters={metadata['parameters']:,}; tokens/step={tokens_per_step:,}; "
          f"schedule tokens={args.steps * tokens_per_step:,}; steps {step} -> {stop}", flush=True)
    wb = None
    if args.wandb_mode != "disabled":
        import wandb
        wb = wandb.init(project=args.wandb_project, name=f"{out.name}-from-{step}",
                        mode=args.wandb_mode, dir=str(out), config=metadata)
        wb.define_metric("step")
        wb.define_metric("*", step_metric="step")
    began = time.perf_counter()

    def elapsed():
        return elapsed_before + time.perf_counter() - began

    def record(kind, **values):
        row = {"kind": kind, "step": step, "tokens": step * tokens_per_step,
               "wall_seconds": elapsed(), **values}
        with log_path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)
        if wb is not None:
            wb.log({key: value for key, value in row.items() if key != "kind"})

    def save(name):
        save_run(out / name, model, optimizer, step, args, model_config, signature, rng, elapsed(), best)

    phase = "initial checkpoint"
    try:
        # A complete step-0 checkpoint is available even if the first forward OOMs.
        save("latest.pt")
        phase = "initial evaluation"
        val_loss = evaluate(model, valid, args, device)
        record("eval", val_loss=val_loss)
        if val_loss < best:
            best = val_loss
            save("best.pt")
        save("latest.pt")
        window_loss, window_seconds, window_steps = 0., 0., 0
        while step < stop:
            phase = "forward/backward/update"
            model.train()
            optimizer.zero_grad(set_to_none=True)
            lr = get_lr_cosine_schedule(step, args.lr, args.min_lr, args.warmup, args.steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            total_loss = torch.zeros((), device=device)
            for _ in range(args.grad_accum):
                x, y = sample_batch(train, args.micro_batch, args.context_length, device, rng)
                with autocast_context(args):
                    loss = cross_entropy(model(x), y)
                total_loss += loss.detach() / args.grad_accum
                (loss / args.grad_accum).backward()
            grad_norm = clip_grad_norm(model.parameters(), args.max_grad_norm)
            loss_value, norm_value = float(total_loss), float(grad_norm)
            if not math.isfinite(loss_value) or not math.isfinite(norm_value):
                raise FloatingPointError("Nonfinite loss/gradient; update cancelled")
            optimizer.step()
            synchronize(device)
            step += 1  # Only count a fully completed optimizer update.
            duration = time.perf_counter() - started
            window_loss += loss_value
            window_seconds += duration
            window_steps += 1
            if step == 1 or step % args.log_every == 0 or step == stop:
                peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.
                record("train", train_loss=window_loss / window_steps, lr=lr, grad_norm=norm_value,
                       tokens_per_second=tokens_per_step * window_steps / window_seconds,
                       peak_allocated_gib=peak)
                window_loss, window_seconds, window_steps = 0., 0., 0
            if step % args.eval_every == 0 or step == stop:
                phase = "evaluation"
                val_loss = evaluate(model, valid, args, device)
                record("eval", val_loss=val_loss)
                if val_loss < best:
                    best = val_loss
                    save("best.pt")
            if step % args.save_every == 0 or step == stop:
                phase = "checkpoint"
                save("latest.pt")
    except (torch.OutOfMemoryError, FloatingPointError, KeyboardInterrupt) as error:
        # An optimizer update can fail halfway through. Never serialize that state as resumable.
        message = {"error": type(error).__name__, "message": str(error), "phase": phase,
                   "last_completed_step_in_memory": step, "resume_from": str(out / "latest.pt")}
        (out / "interrupted.json").write_text(json.dumps(message, indent=2))
        print(json.dumps(message), flush=True)
        if isinstance(error, torch.OutOfMemoryError):
            print("Reduce --micro-batch and increase --grad-accum to preserve their product; "
                  "resume from the last complete checkpoint.", flush=True)
        raise
    finally:
        if wb is not None:
            wb.finish()
    print(f"Finished at step {step}; best validation loss={best:.6f}; checkpoint={out / 'latest.pt'}", flush=True)


if __name__ == "__main__":
    main()

