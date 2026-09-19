"""Ten documents per domain: compression, cross-domain encoding, and throughput."""
import argparse
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import pickle
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cs336_basics.tokenizer import Tokenizer
from train_bpe_disk import documents


def source_info(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_tokenizer(path, cache_size=0):
    # These pickle files must be trusted artifacts from this project.
    with Path(path).open("rb") as stream:
        artifact = pickle.load(stream)
    vocab = artifact["vocab"]
    if set(vocab) != set(range(len(vocab))) or not 257 <= len(vocab) <= 65536:
        raise ValueError("Require contiguous token IDs fitting uint16")
    if not all(type(v) is bytes for v in vocab.values()):
        raise ValueError("Vocabulary values must be bytes")
    if artifact["special_tokens"] != ["<|endoftext|>"]:
        raise ValueError("This data pipeline expects EOS as its only special token")
    tokenizer = Tokenizer(vocab, artifact["merges"], artifact["special_tokens"])
    if cache_size:
        original = tokenizer._encode_pretoken

        @lru_cache(maxsize=cache_size)
        def cached(pretoken):
            return tuple(original(pretoken))

        # Cache only short pretokens. Long, one-off strings cannot fill the cache.
        def encode_pretoken(pretoken):
            return cached(pretoken) if len(pretoken) <= 64 else original(pretoken)

        tokenizer._encode_pretoken = encode_pretoken
        tokenizer.clear_experiment_cache = cached.cache_clear
    return tokenizer


def sample_documents(path, seed, count=10):
    rng = random.Random(seed)
    reservoir = []
    seen = 0
    for index, (raw, end) in enumerate(documents(Path(path), 0, 4 * 1024 ** 2, 64 * 1024 ** 2)):
        if not raw.strip():
            continue
        seen += 1
        record = (index, raw)
        if len(reservoir) < count:
            reservoir.append(record)
        else:
            j = rng.randrange(seen)
            if j < count:
                reservoir[j] = record
    if seen < count:
        raise ValueError(f"Only {seen} nonempty documents in {path}")
    samples = [{"document_index": index, "utf8_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(), "text": raw.decode("utf-8")}
               for index, raw in sorted(reservoir)]
    return {"source": source_info(path), "seed": seed, "nonempty_documents_seen": seen,
            "sampling": "uniform reservoir over nonempty EOS-delimited documents; EOS excluded",
            "documents": samples}


def benchmark(tokenizer_path, sample, repeats, cache_size):
    docs = sample["documents"]
    tok = load_tokenizer(tokenizer_path)
    expected = [tok.encode(doc["text"]) for doc in docs]
    for doc, ids in zip(docs, expected):
        if tok.decode(ids) != doc["text"]:
            raise ValueError("Round-trip mismatch")
        if list(tok.encode_iterable(io.StringIO(doc["text"]))) != ids:
            raise ValueError("encode_iterable differs from encode on sampled document")
    total_bytes = sum(d["utf8_bytes"] for d in docs)
    total_tokens = sum(map(len, expected))
    timings = {}
    for name, capacity in [("original", 0), ("bounded_cache", cache_size)]:
        seconds = []
        for _ in range(repeats):
            tok = load_tokenizer(tokenizer_path, capacity)
            started = time.perf_counter()
            actual = [tok.encode(doc["text"]) for doc in docs]
            elapsed = time.perf_counter() - started
            if actual != expected:
                raise ValueError("Cached/timed IDs differ from original encoder")
            seconds.append(elapsed)
            if capacity:
                tok.clear_experiment_cache()
        median = statistics.median(seconds)
        throughput = total_bytes / median
        timings[name] = {
            "cache_entries": capacity, "seconds_per_repeat": seconds,
            "median_seconds": median, "bytes_per_second": throughput,
            "estimated_825GB_hours": (825 * 10 ** 9) / throughput / 3600,
            "scope": "encode only, documents already in RAM, fresh tokenizer/cache each repeat; single process",
        }
    return {
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest(),
        "utf8_bytes": total_bytes, "tokens": total_tokens,
        "bytes_per_token": total_bytes / total_tokens,
        "documents": [{"document_index": doc["document_index"], "bytes": doc["utf8_bytes"],
                       "tokens": len(ids), "ids": ids} for doc, ids in zip(docs, expected)],
        "timing": timings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny-input", type=Path, default=Path("data/TinyStoriesV2-GPT4-valid.txt"))
    parser.add_argument("--owt-input", type=Path, default=Path("data/owt_valid.txt"))
    parser.add_argument("--tiny-tokenizer", type=Path, default=Path("data/tinystories_tokenizer.pkl"))
    parser.add_argument("--owt-tokenizer", type=Path, default=Path("data/owt_bpe_32k/tokenizer.pkl"))
    parser.add_argument("--out", type=Path, default=Path("reports/tokenizer"))
    parser.add_argument("--seed", type=int, default=336)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cache-size", type=int, default=32768)
    args = parser.parse_args()
    if args.repeats < 1 or args.cache_size < 1:
        parser.error("Require positive repeats and cache-size")
    args.out.mkdir(parents=True, exist_ok=False)
    samples = {}
    for name, path, seed in [("tinystories", args.tiny_input, args.seed),
                             ("owt", args.owt_input, args.seed + 1)]:
        print(f"Sampling 10 {name} documents from the entire validation file...", flush=True)
        samples[name] = sample_documents(path, seed)
    (args.out / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    results = {}
    for name, path, domain in [("tiny_on_tiny", args.tiny_tokenizer, "tinystories"),
                               ("owt_on_owt", args.owt_tokenizer, "owt"),
                               ("tiny_on_owt", args.tiny_tokenizer, "owt")]:
        print(f"Benchmarking {name}...", flush=True)
        result = benchmark(path, samples[domain], args.repeats, args.cache_size)
        (args.out / f"{name}_ids.json").write_text(json.dumps(result.pop("documents")), encoding="utf-8")
        results[name] = result
    (args.out / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name, result in results.items():
        print(f"{name}: {result['bytes_per_token']:.4f} bytes/token")
        for mode, timing in result["timing"].items():
            print(f"  {mode}: {timing['bytes_per_second']:,.0f} bytes/s; "
                  f"825GB estimate={timing['estimated_825GB_hours']:.2f} hours")
    print(f"Saved: {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
