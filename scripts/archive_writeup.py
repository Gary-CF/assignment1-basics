"""Archive existing small report evidence; never train or copy model/data binaries."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


REQUIRED_RUNS = (
    "baseline", "lr_1e3", "lr_3e3", "batch1", "batch64", "batch128",
    "no_rmsnorm_1e3", "no_rmsnorm_3e4", "postnorm_1e3",
    "no_rope_1e3", "silu_1e3", "owt_1e3",
)
FULL_RUNS = ("baseline", "lr_1e3", "owt_1e3")
GROUPS = ("learning_rate", "batch_size", "rmsnorm", "architecture")
ALLOWED_NAMES = ("metrics.jsonl", "loss.png", "loss.svg", "sample.txt", "sample.json")
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024


def archive(root: Path):
    root = root.resolve()
    required = [root / "runs" / name / "metrics.jsonl" for name in REQUIRED_RUNS]
    required += [root / "runs" / name / f"loss.{ext}"
                 for name in FULL_RUNS for ext in ("png", "svg")]
    required += [root / "runs" / name / f"sample.{ext}"
                 for name in ("lr_1e3", "owt_1e3") for ext in ("txt", "json")]
    in_place = [root / "reports/experiments" / f"{group}.{ext}"
                for group in GROUPS for ext in ("png", "svg")]
    in_place += [root / "reports/experiments/comparison.json",
                 root / "reports/tokenizer/summary.json"]
    required += in_place
    missing = [str(p.relative_to(root)) for p in required if not p.is_file()]
    if missing:
        raise ValueError("Missing required evidence; no files copied:\n  " + "\n  ".join(missing))

    copies = []
    # Immediate run directories only; no recursive copying of checkpoints or wandb.
    for metrics in sorted((root / "runs").glob("*/metrics.jsonl")):
        for name in ALLOWED_NAMES:
            source = metrics.parent / name
            if source.is_file():
                copies.append((source, root / "reports/training" / metrics.parent.name / name))
    for source, target in (
        ("data/owt_bpe_32k/summary.json", "reports/evidence/owt_bpe_summary.json"),
        ("runs/microbatch_probe/summary.json", "reports/evidence/microbatch_summary.json"),
    ):
        if (root / source).is_file():
            copies.append((root / source, root / target))

    sources = [source for source, _ in copies] + in_place
    for source in sources:
        if not source.resolve().is_relative_to(root):
            raise ValueError(f"Evidence resolves outside repository: {source}")
        size = source.stat().st_size
        if not 0 < size <= MAX_FILE_BYTES:
            raise ValueError(f"Empty or unexpectedly large evidence file: {source} ({size} bytes)")
    total = sum(source.stat().st_size for source in sources)
    if total > MAX_TOTAL_BYTES:
        raise ValueError(f"Evidence exceeds the 256 MiB archive limit: {total} bytes")
    # Refuse a destination symlink that would send a copy outside this repository.
    for _, target in copies:
        if not target.resolve().is_relative_to(root):
            raise ValueError(f"Destination resolves outside repository: {target}")

    entries = []
    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        entries.append({
            "source": source.relative_to(root).as_posix(),
            "path": target.relative_to(root).as_posix(),
            "bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        })
    for source in in_place:
        entries.append({
            "source": source.relative_to(root).as_posix(),
            "path": source.relative_to(root).as_posix(),
            "bytes": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        })
    target = root / "reports/evidence/manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "scope": "Existing small artifacts copied without modifying their contents; not a numerical audit",
        "files": entries,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Archived/indexed {len(entries)} files, {total / 1024**2:.2f} MiB")
    print(target.relative_to(root))
    print("Original logs remain unchanged. No datasets, checkpoints, or wandb directories copied.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        archive(args.repo)
    except ValueError as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
