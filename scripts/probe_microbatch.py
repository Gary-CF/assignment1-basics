"""Find a short-run CUDA micro-batch boundary using the existing training CLI.

Each trial runs in a new process, including forward, backward, gradient clipping
and AdamW updates. It is a capacity probe, not a learning-curve experiment.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


def search_boundary(test, cap):
    """Double until OOM, then bisect; assumes memory demand is monotonic."""
    low, candidate = 0, 1
    while test(candidate):
        low = candidate
        if low == cap:
            return low, None
        candidate = min(cap, candidate * 2)
    high = candidate
    while high - low > 1:
        middle = (low + high) // 2
        if test(middle):
            low = middle
        else:
            high = middle
    return low, high


def run_trial(trainer, root, output, batch, steps):
    trial = output / f"batch_{batch:04d}"
    trial.mkdir()  # Never reuse an existing trial directory.
    command = [
        sys.executable, str(trainer), "--out", str(trial),
        "--ablation", "baseline", "--device", "cuda", "--precision", "bf16",
        "--context-length", "256", "--d-model", "512", "--num-layers", "4",
        "--num-heads", "16", "--d-ff", "1344",
        "--micro-batch", str(batch), "--grad-accum", "1",
        "--steps", "40000", "--warmup", "200", "--stop-after", str(steps),
        "--lr", "1e-3", "--min-lr", "1e-4",
        "--beta1", "0.9", "--beta2", "0.95", "--eps", "1e-8",
        "--weight-decay", "0.1", "--max-grad-norm", "1", "--seed", "336",
        "--eval-examples", "1", "--eval-every", str(steps),
        "--save-every", str(steps), "--log-every", "1",
        "--wandb-mode", "disabled",
    ]
    console = trial / "console.log"
    print(f"Testing micro-batch={batch}, {steps} complete optimizer updates...", flush=True)
    with console.open("w") as stream:
        process = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
    result = {"micro_batch": batch, "returncode": process.returncode,
              "console": str(console), "command": command}
    interruption = trial / "interrupted.json"
    error_info = json.loads(interruption.read_text()) if interruption.exists() else {}
    if process.returncode == 0:
        rows = [json.loads(line) for line in (trial / "metrics.jsonl").read_text().splitlines()
                if line.strip()]
        train = [row for row in rows if row["kind"] == "train"]
        if not train or train[-1]["step"] != steps:
            raise RuntimeError(f"Trial did not complete {steps} steps: {console}")
        peak = max(row["peak_allocated_gib"] for row in train)
        if not math.isfinite(peak) or peak <= 0:
            raise RuntimeError(f"Invalid CUDA memory measurement: {console}")
        result.update(status="fits", peak_allocated_gib=peak)
    elif (error_info.get("error") == "OutOfMemoryError" or
          "CUDA out of memory" in console.read_text()):
        result.update(status="oom", phase=error_info.get("phase", "see console"))
    else:
        result.update(status="error", detail=error_info or console.read_text()[-3000:])
    # Keep logs/configs; remove only throwaway checkpoints from this new trial.
    if result["status"] in ("fits", "oom"):
        for filename in ("latest.pt", "best.pt", "latest.tmp", "best.tmp"):
            (trial / filename).unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/microbatch_probe"))
    parser.add_argument("--max-batch", type=int, default=256)
    parser.add_argument("--trial-steps", type=int, default=3)
    args = parser.parse_args()
    if args.max_batch < 1 or not 2 <= args.trial_steps <= 40000:
        parser.error("Require max-batch >= 1 and 2 <= trial-steps <= 40000")
    root = Path(__file__).resolve().parents[1]
    trainer = root / "run_train.py"
    if not trainer.exists():
        parser.error(f"Missing training script: {trainer}")
    output = args.out if args.out.is_absolute() else root / args.out
    if output.exists():
        parser.error(f"Output already exists; choose a new --out directory: {output}")
    output.mkdir(parents=True)
    report = {
        "status": "running", "trial_steps": args.trial_steps, "max_batch": args.max_batch,
        "precision": "bf16", "gradient_accumulation": 1,
        "scope": "Short training-step capacity; not a guarantee for long runs or generation",
        "search_assumption": "Memory demand is monotonic; other GPU usage remains stable",
        "trials": [],
    }
    summary = output / "summary.json"

    def save():
        temporary = summary.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(summary)

    def test(batch):
        result = run_trial(trainer, root, output, batch, args.trial_steps)
        report["trials"].append(result)
        save()
        print(f"micro-batch={batch}: {result['status']}"
              + (f", peak={result['peak_allocated_gib']:.3f} GiB"
                 if result["status"] == "fits" else ""), flush=True)
        if result["status"] == "error":
            raise RuntimeError(f"Non-OOM failure; inspect {result['console']}")
        return result["status"] == "fits"

    save()
    try:
        fits, oom = search_boundary(test, args.max_batch)
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="aborted", error=f"{type(error).__name__}: {error}")
        save()
        raise
    report.update(status="completed", largest_fitting_tested=fits, smallest_oom_tested=oom)
    save()
    print(f"Largest fitting tested: {fits}; smallest OOM tested: {oom}")
    if oom is None:
        print("Search cap reached without OOM; capacity is at least this batch size.")
    print(summary)


if __name__ == "__main__":
    main()
