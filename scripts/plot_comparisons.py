"""Plot fixed-token-budget validation curves without changing raw logs."""
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GROUPS = {
    "learning_rate": [
        ("baseline", "LR 3e-4"), ("lr_1e3", "LR 1e-3"),
        ("lr_3e3", "LR 3e-3"),
    ],
    "batch_size": [
        ("batch1", "Batch 1"),
        ("lr_1e3", "Batch 32"),
        ("batch64", "Batch 64"),
        ("batch128", "Batch 128"),
    ],
    "rmsnorm": [
        ("lr_1e3", "RMSNorm, LR 1e-3"),
        ("no_rmsnorm_1e3", "No RMSNorm, LR 1e-3"),
        ("baseline", "RMSNorm, LR 3e-4"),
        ("no_rmsnorm_3e4", "No RMSNorm, LR 3e-4"),
    ],
    "architecture": [
        ("lr_1e3", "Baseline"), ("postnorm_1e3", "Post-norm"),
        ("no_rope_1e3", "NoPE"), ("silu_1e3", "SiLU FFN"),
    ],
}


def read_evals(path, max_tokens):
    # Retain the last evaluation for each step, including resume-point repeats.
    # This is a plotting convention, not proof that concurrent writes were harmless.
    records = {}
    duplicates = conflicts = 0
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON; raw log unchanged") from error
        if row.get("kind") != "eval":
            continue
        step, tokens, loss = row["step"], row["tokens"], row["val_loss"]
        if (not isinstance(step, int) or step < 0 or
                not isinstance(tokens, int) or tokens < 0 or
                not math.isfinite(loss)):
            raise ValueError(f"{path}:{line_number}: invalid evaluation values")
        if step in records:
            duplicates += 1
            old = records[step]
            conflicts += (old["val_loss"] != loss or old["tokens"] != tokens)
        records[step] = row
    rows = sorted(
        (row for row in records.values() if row["tokens"] <= max_tokens),
        key=lambda row: row["step"],
    )
    if not rows or rows[-1]["tokens"] != max_tokens:
        raise ValueError(f"{path}: missing evaluation at exactly {max_tokens:,} tokens")
    if any(b["tokens"] <= a["tokens"] for a, b in zip(rows, rows[1:])):
        raise ValueError(f"{path}: token counts do not increase with steps")
    last = rows[-1]
    return {
        "source": str(path.resolve()),
        "duplicate_eval_rows": duplicates,
        "conflicting_duplicate_eval_rows": conflicts,
        "endpoint_step": last["step"],
        "endpoint_tokens": last["tokens"],
        "endpoint_val_loss": last["val_loss"],
        "eval_rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("reports/experiments"))
    parser.add_argument("--max-tokens", type=int, default=32_768_000)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    names = sorted({name for group in GROUPS.values() for name, _ in group})
    runs = {name: read_evals(args.runs / name / "metrics.jsonl", args.max_tokens)
            for name in names}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, result in runs.items():
        print(f"{name}: step={result['endpoint_step']}, "
              f"val_loss={result['endpoint_val_loss']:.6f}, "
              f"duplicate_evals={result['duplicate_eval_rows']}, "
              f"conflicting_evals={result['conflicting_duplicate_eval_rows']}")
    for group_name, members in GROUPS.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for name, label in members:
            rows = runs[name]["eval_rows"]
            for index, ax in enumerate(axes):
                selected = rows if index == 0 else [
                    row for row in rows if row["tokens"] >= args.max_tokens / 2
                ]
                ax.plot([row["tokens"] / 1e6 for row in selected],
                        [row["val_loss"] for row in selected],
                        label=label, linewidth=1.6)
        for index, ax in enumerate(axes):
            ax.set(xlabel="Training tokens (millions)",
                   ylabel="Validation cross-entropy (nats/token)",
                   title="Full comparison budget" if index == 0 else "Second half of budget")
            ax.set_xlim(0 if index == 0 else args.max_tokens / 2e6, args.max_tokens / 1e6)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle(group_name.replace("_", " ").title())
        for suffix in ("png", "svg"):
            output = args.out / f"{group_name}.{suffix}"
            fig.savefig(output, dpi=180)
            print(output)
        plt.close(fig)
    summary = args.out / "comparison.json"
    summary.write_text(json.dumps({
        "max_tokens": args.max_tokens,
        "duplicate_policy": "last evaluation per step; raw logs unchanged",
        "runs": runs,
    }, indent=2, allow_nan=False) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
