"""Usage: uv run --with matplotlib python plot_loss.py runs/tinystories/metrics.jsonl"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    args = parser.parse_args()
    rows = {}
    for line in args.metrics.read_text().splitlines():
        row = json.loads(line)
        rows[(row["kind"], row["step"])] = row
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for kind, key, color in (("train", "train_loss", "#2563eb"), ("eval", "val_loss", "#e76f24")):
        series = sorted((row for row in rows.values() if row["kind"] == kind), key=lambda row: row["step"])
        if not series:
            continue
        for ax, x_key, scale in ((axes[0], "step", 1), (axes[1], "wall_seconds", 60)):
            ax.plot([row[x_key] / scale for row in series], [row[key] for row in series],
                    label=kind, color=color, linewidth=1.6)
    for ax, label in zip(axes, ("Optimizer steps", "Run wall time (minutes)")):
        ax.set(xlabel=label, ylabel="Cross-entropy (nats/token)")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle(args.metrics.parent.name)
    output = args.metrics.with_name("loss.png")
    svg_output = output.with_suffix(".svg")
    fig.savefig(output, dpi=180)
    fig.savefig(svg_output)
    plt.close(fig)
    print(output)
    print(svg_output)


if __name__ == "__main__":
    main()
