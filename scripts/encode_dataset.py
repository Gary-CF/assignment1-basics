"""Encode the complete EOS-delimited corpus into native-endian uint16, with resume."""
import argparse
from array import array
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

from tokenizer_experiments import load_tokenizer, source_info
from train_bpe_disk import documents, EOS


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def encode(args):
    if array("H").itemsize != 2:
        raise RuntimeError("Expected 16-bit unsigned short")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.resolve() == args.input.resolve():
        raise ValueError("Output cannot be the input file")
    lock = args.out.with_name(args.out.name + ".lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(args)
    finally:
        lock.close()


def run_locked(args):
    out = args.out
    partial = out.with_name(out.name + ".partial")
    state_path = out.with_name(out.name + ".state.json")
    summary_path = out.with_name(out.name + ".json")
    source = source_info(args.input)
    signature = hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()
    if args.resume:
        state = json.loads(state_path.read_text())
        if (state["source"] != source or state["tokenizer_sha256"] != signature
                or state["byteorder"] != sys.byteorder):
            raise ValueError("Input/tokenizer/byteorder changed; cannot resume")
        if state["phase"] == "complete" and summary_path.exists():
            if out.stat().st_size != state["tokens"] * 2:
                raise ValueError("Completed output size changed")
            print(summary_path.read_text(), flush=True)
            return
    else:
        if any(p.exists() for p in (out, partial, state_path, summary_path)):
            raise ValueError("Output artifacts already exist; use --resume or a new --out")
        state = {"phase": "encoding", "source": source,
                 "tokenizer": str(args.tokenizer.resolve()), "tokenizer_sha256": signature,
                 "dtype": "uint16", "byteorder": sys.byteorder,
                 "input_offset": 0, "tokens": 0, "documents": 0,
                 "min_id": None, "max_id": None, "elapsed_seconds": 0.0}
        partial.touch(exist_ok=False)
        atomic_json(state_path, state)
    tok = load_tokenizer(args.tokenizer, args.cache_size)
    eos_id = tok.id_of[EOS]
    if state["phase"] == "encoding":
        committed_bytes = state["tokens"] * 2
        if partial.stat().st_size < committed_bytes:
            raise ValueError("Partial file is shorter than its committed checkpoint")
        prior_elapsed = state["elapsed_seconds"]
        started = time.perf_counter()
        start_offset = state["input_offset"]
        offset = start_offset
        last_checkpoint = offset
        last_log = started
        additional_documents = 0
        with partial.open("r+b", buffering=1024 * 1024) as sink:
            # Discard only the uncommitted tail left by an interrupted invocation.
            sink.truncate(committed_bytes)
            sink.seek(committed_bytes)

            def checkpoint():
                sink.flush()
                os.fsync(sink.fileno())
                state["elapsed_seconds"] = prior_elapsed + time.perf_counter() - started
                atomic_json(state_path, state)

            for raw, end in documents(args.input, offset, 4 * 1024 ** 2,
                                      args.max_document_mib * 1024 ** 2):
                ids = tok.encode(raw.decode("utf-8"))
                # Preserve delimiters exactly, including consecutive EOS and a missing final EOS.
                delimiter_bytes = end - offset - len(raw)
                if delimiter_bytes == len(EOS):
                    ids.append(eos_id)
                elif delimiter_bytes != 0:
                    raise RuntimeError("Invalid document boundary")
                if ids:
                    low, high = min(ids), max(ids)
                    if low < 0 or high >= len(tok.vocab):
                        raise ValueError("Token ID outside the vocabulary")
                    state["min_id"] = low if state["min_id"] is None else min(state["min_id"], low)
                    state["max_id"] = high if state["max_id"] is None else max(state["max_id"], high)
                sink.write(array("H", ids).tobytes())
                offset = end
                state["input_offset"] = offset
                state["tokens"] += len(ids)
                state["documents"] += 1
                additional_documents += 1
                if offset - last_checkpoint >= args.checkpoint_mib * 1024 ** 2:
                    checkpoint()
                    last_checkpoint = offset
                now = time.perf_counter()
                if now - last_log >= 10:
                    rate = (offset - start_offset) / (now - started)
                    eta = (source["bytes"] - offset) / max(rate, 1) / 60
                    print(f"{offset / max(source['bytes'], 1):.1%}: {state['tokens']:,} tokens, "
                          f"{rate / 1e6:.2f} MB/s, estimated remaining {eta:.1f} min", flush=True)
                    last_log = now
                if args.stop_after_documents and additional_documents >= args.stop_after_documents:
                    break
            if offset == source["bytes"]:
                state["phase"] = "encoded"
            checkpoint()
        if state["phase"] == "encoding":
            print(f"Paused at byte {offset:,}; restart with --resume", flush=True)
            return
    # Recover even if a previous invocation stopped between renaming and reporting.
    if partial.exists():
        if out.exists():
            raise ValueError("Both partial and final files exist; inspect before proceeding")
        if partial.stat().st_size != state["tokens"] * 2:
            raise ValueError("Encoded file size mismatch")
        os.replace(partial, out)
    if out.stat().st_size != state["tokens"] * 2 or state["input_offset"] != source["bytes"]:
        raise ValueError("Final file does not match recorded completion")
    preview = array("H")
    with out.open("rb") as stream:
        preview.frombytes(stream.read(min(state["tokens"], 500) * 2))
    summary = dict(state, phase="complete", output=str(out.resolve()),
                   output_bytes=out.stat().st_size, cache_entries_this_process=args.cache_size,
                   current_process_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
                   preview_500_tokens=tok.decode(preview))
    atomic_json(summary_path, summary)
    atomic_json(state_path, dict(state, phase="complete"))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache-size", type=int, default=32768)
    parser.add_argument("--checkpoint-mib", type=int, default=16)
    parser.add_argument("--max-document-mib", type=int, default=64)
    parser.add_argument("--stop-after-documents", type=int,
                        help="Pause after this many additional documents in this invocation")
    args = parser.parse_args()
    if args.cache_size < 0 or args.checkpoint_mib < 1 or args.max_document_mib < 1:
        parser.error("Invalid cache or memory settings")
    if args.stop_after_documents is not None and args.stop_after_documents < 1:
        parser.error("stop-after-documents must be positive")
    encode(args)


if __name__ == "__main__":
    main()
