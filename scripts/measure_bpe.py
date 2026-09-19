"""Measure BPE training in a fresh Linux/WSL process; preserve existing artifacts."""
import argparse
import cProfile
import hashlib
import json
import pickle
import pstats
import resource
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--compare-tokenizer", type=Path)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Missing input: {args.input}")
    if args.vocab_size < 257:
        parser.error("vocab-size must be at least 257")
    if args.compare_tokenizer and not args.compare_tokenizer.is_file():
        parser.error(f"Missing comparison tokenizer: {args.compare_tokenizer}")
    if args.out.exists():
        parser.error(f"Output exists; choose a new directory: {args.out}")

    # Support both `python scripts/measure_bpe.py` and module execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cs336_basics.bpe_tokenizer import train_bpe

    args.out.mkdir(parents=True)
    digest = hashlib.sha256()
    with args.input.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    special_tokens = ["<|endoftext|>"]
    profiler = cProfile.Profile() if args.profile else None
    print("Training BPE; this implementation prints no merge progress...", flush=True)
    started = time.perf_counter()
    if profiler:
        profiler.enable()
    try:
        vocab, merges = train_bpe(str(args.input), args.vocab_size, special_tokens)
    finally:
        if profiler:
            profiler.disable()
    elapsed = time.perf_counter() - started
    # Linux/WSL ru_maxrss is KiB, and includes process memory since startup.
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    special_bytes = {s.encode("utf-8") for s in special_tokens}
    ordinary = [(i, token) for i, token in vocab.items() if token not in special_bytes]
    longest_id, longest = max(ordinary, key=lambda item: (len(item[1]), item[1]))
    artifact = {"vocab": vocab, "merges": merges, "special_tokens": special_tokens}
    with (args.out / "tokenizer.pkl").open("xb") as stream:
        pickle.dump(artifact, stream)
    report = {
        "input": str(args.input.resolve()),
        "input_bytes": args.input.stat().st_size,
        "input_sha256": digest.hexdigest(),
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": len(vocab), "merges": len(merges),
        "training_wall_seconds": elapsed,
        "profiling_enabled": args.profile,
        "timing_scope": "train_bpe only; includes cProfile overhead if enabled",
        "peak_process_rss_kib": peak_rss_kib,
        "peak_process_rss_gib": peak_rss_kib / (1024 ** 2),
        "memory_scope": "Linux/WSL process high-water mark through training; excludes later pickle operations",
        "longest_ordinary_token": {
            "id": longest_id, "byte_length": len(longest),
            "bytes_repr": repr(longest), "hex": longest.hex(),
            "text_with_replacement": longest.decode("utf-8", errors="replace"),
        },
    }
    if args.compare_tokenizer:
        # Load only trusted local artifacts produced by this project.
        with args.compare_tokenizer.open("rb") as stream:
            previous = pickle.load(stream)
        report["comparison"] = {
            "path": str(args.compare_tokenizer),
            "vocab_equal": vocab == previous["vocab"],
            "merges_equal": merges == previous["merges"],
            "special_tokens_equal": special_tokens == previous.get("special_tokens"),
        }
    if profiler:
        profiler.dump_stats(str(args.out / "profile.prof"))
        with (args.out / "profile.txt").open("w") as stream:
            stats = pstats.Stats(profiler, stream=stream).strip_dirs()
            stats.sort_stats("cumulative").print_stats(25)
            stats.sort_stats("tottime").print_stats(25)
    output = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    (args.out / "summary.json").write_text(output + "\n", encoding="utf-8")
    print(output)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
