"""Exact full-corpus GPT-2 BPE with SQLite indexes; EOS is <|endoftext|>."""
import argparse
from array import array
from collections import Counter
import json
import os
from pathlib import Path
import pickle
import resource
import sqlite3
import sys
import time

import regex

EOS = b"<|endoftext|>"
PATTERN = regex.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")


def pack(values):
    a = array("I", values)
    if a.itemsize != 4:
        raise RuntimeError("Expected 32-bit unsigned integers")
    if sys.byteorder != "little":
        a.byteswap()
    return a.tobytes()


def unpack(blob):
    a = array("I")
    a.frombytes(blob)
    if sys.byteorder != "little":
        a.byteswap()
    return a


def documents(path, offset, chunk_bytes, max_document_bytes):
    """Split bytes only at complete EOS delimiters; never split a UTF-8 document."""
    pending = bytearray()
    consumed = offset
    with path.open("rb") as stream:
        stream.seek(offset)
        while True:
            block = stream.read(chunk_bytes)
            if not block:
                if pending:
                    if len(pending) > max_document_bytes:
                        raise ValueError("Document exceeds --max-document-mib")
                    yield bytes(pending), consumed + len(pending)
                return
            pending.extend(block)
            start = 0
            while True:
                end = pending.find(EOS, start)
                if end < 0:
                    break
                if end - start > max_document_bytes:
                    raise ValueError("Document exceeds --max-document-mib")
                next_start = end + len(EOS)
                yield bytes(pending[start:end]), consumed + next_start
                start = next_start
            if start:
                del pending[:start]
                consumed += start
            # A partial EOS at the chunk boundary is not part of the document.
            if len(pending) > max_document_bytes + len(EOS) - 1:
                raise ValueError("No EOS within --max-document-mib; inspect the corpus format")


def get_meta(db, key):
    return json.loads(db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()[0])


def set_meta(db, key, value):
    db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value)))


def initialize(db, source, vocab_size):
    db.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE words (id INTEGER PRIMARY KEY, raw BLOB UNIQUE NOT NULL,
                            n INTEGER NOT NULL, seq BLOB);
        CREATE TABLE pairs (a INTEGER, b INTEGER, av BLOB NOT NULL, bv BLOB NOT NULL,
                            n INTEGER NOT NULL CHECK(n>=0), PRIMARY KEY(a,b)) WITHOUT ROWID;
        CREATE INDEX best_pair ON pairs(n DESC, av DESC, bv DESC);
        CREATE TABLE occurrences (a INTEGER, b INTEGER, w INTEGER,
                                  PRIMARY KEY(a,b,w)) WITHOUT ROWID;
        CREATE TABLE tokens (id INTEGER PRIMARY KEY, value BLOB NOT NULL);
        CREATE TABLE merges (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER);
    """)
    with db:
        for key, value in {"source": source, "vocab_size": vocab_size, "phase": "counting",
                           "offset": 0, "pretoken_occurrences": 0,
                           "indexed_through": 0, "elapsed_seconds": 0.0}.items():
            set_meta(db, key, value)
        db.executemany("INSERT INTO tokens VALUES (?,?)",
                       [(0, EOS)] + [(b + 1, bytes([b])) for b in range(256)])


def train(args):
    path = args.input.resolve()
    stat = path.stat()
    source = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    database = args.out / "counts.sqlite3"
    if args.resume:
        if not database.is_file():
            raise ValueError("--resume requires an existing counts.sqlite3")
    else:
        args.out.mkdir(parents=True, exist_ok=False)
    db = sqlite3.connect(database)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA cache_size=-65536")  # 64 MiB page-cache target, not a process limit.
    db.execute("PRAGMA temp_store=FILE")
    try:
        if not args.resume:
            initialize(db, source, args.vocab_size)
        if get_meta(db, "source") != source or get_meta(db, "vocab_size") != args.vocab_size:
            raise ValueError("Resume requires the same input path, size, mtime and vocab-size")
        started = time.perf_counter()
        prior_elapsed = get_meta(db, "elapsed_seconds")

        def checkpoint_time():
            set_meta(db, "elapsed_seconds", prior_elapsed + time.perf_counter() - started)

        if get_meta(db, "phase") == "counting":
            counts = Counter()
            pending_count = 0
            total = get_meta(db, "pretoken_occurrences")
            offset = get_meta(db, "offset")
            last_print = time.perf_counter()

            def flush(end):
                nonlocal total, pending_count
                with db:
                    db.executemany("INSERT INTO words(raw,n) VALUES (?,?) "
                                   "ON CONFLICT(raw) DO UPDATE SET n=n+excluded.n",
                                   ((word.encode("utf-8"), n) for word, n in counts.items()))
                    total += pending_count
                    set_meta(db, "offset", end)
                    set_meta(db, "pretoken_occurrences", total)
                    checkpoint_time()
                counts.clear()
                pending_count = 0

            for raw, offset in documents(path, offset, args.chunk_bytes,
                                          args.max_document_mib * 1024 ** 2):
                for match in PATTERN.finditer(raw.decode("utf-8")):
                    counts[match.group()] += 1
                    pending_count += 1
                # Commit only at complete document boundaries, making resume exact.
                if pending_count >= 100_000:
                    flush(offset)
                if time.perf_counter() - last_print >= 10:
                    print(f"counting: {offset / max(stat.st_size, 1):.1%}, "
                          f"{total + pending_count:,} pretokens", flush=True)
                    last_print = time.perf_counter()
            flush(offset)
            with db:
                set_meta(db, "phase", "indexing")
                checkpoint_time()
            print("Counting complete; constructing disk pair indexes...", flush=True)

        vocab = dict(db.execute("SELECT id,value FROM tokens ORDER BY id"))

        def change_pair(a, b, delta):
            if delta < 0:
                changed = db.execute("UPDATE pairs SET n=n+? WHERE a=? AND b=?", (delta, a, b))
                if changed.rowcount != 1:
                    raise RuntimeError("Missing pair during frequency decrement")
            else:
                db.execute("INSERT INTO pairs VALUES (?,?,?,?,?) "
                           "ON CONFLICT(a,b) DO UPDATE SET n=n+excluded.n",
                           (a, b, vocab[a], vocab[b], delta))
            db.execute("DELETE FROM pairs WHERE a=? AND b=? AND n=0", (a, b))

        if get_meta(db, "phase") == "indexing":
            last = get_meta(db, "indexed_through")
            last_print = time.perf_counter()
            while True:
                rows = db.execute("SELECT id,raw,n FROM words WHERE id>? ORDER BY id LIMIT 256",
                                  (last,)).fetchall()
                if not rows:
                    break
                with db:
                    for wid, raw, freq in rows:
                        seq = array("I", (b + 1 for b in raw))
                        db.execute("UPDATE words SET seq=? WHERE id=?", (pack(seq), wid))
                        for (a, b), n in Counter(zip(seq, seq[1:])).items():
                            change_pair(a, b, n * freq)
                            db.execute("INSERT INTO occurrences VALUES (?,?,?)", (a, b, wid))
                        last = wid
                    set_meta(db, "indexed_through", last)
                    checkpoint_time()
                if time.perf_counter() - last_print >= 10:
                    print(f"indexing: {last:,} unique pretokens", flush=True)
                    last_print = time.perf_counter()
            with db:
                set_meta(db, "phase", "merging")
                checkpoint_time()
            print("Indexes complete; starting merges...", flush=True)

        merge_count = db.execute("SELECT COUNT(*) FROM merges").fetchone()[0]
        last_print = time.perf_counter()
        while get_meta(db, "phase") == "merging" and len(vocab) < args.vocab_size:
            best = db.execute("SELECT a,b,n FROM pairs ORDER BY n DESC,av DESC,bv DESC LIMIT 1").fetchone()
            if best is None:
                break
            a, b, frequency = best
            new_id = len(vocab)
            vocab[new_id] = vocab[a] + vocab[b]
            # Each complete merge is one transaction. Interruptions roll it back.
            with db:
                db.execute("INSERT INTO tokens VALUES (?,?)", (new_id, vocab[new_id]))
                last_word = 0
                while True:
                    rows = db.execute("SELECT w.id,w.n,w.seq FROM occurrences o "
                                      "JOIN words w ON w.id=o.w "
                                      "WHERE o.a=? AND o.b=? AND o.w>? ORDER BY o.w LIMIT 256",
                                      (a, b, last_word)).fetchall()
                    if not rows:
                        break
                    for wid, freq, blob in rows:
                        old = unpack(blob)
                        new = array("I")
                        i = 0
                        while i < len(old):
                            if i + 1 < len(old) and old[i] == a and old[i + 1] == b:
                                new.append(new_id)
                                i += 2
                            else:
                                new.append(old[i])
                                i += 1
                        before = Counter(zip(old, old[1:]))
                        after = Counter(zip(new, new[1:]))
                        for p in before.keys() | after.keys():
                            delta = (after[p] - before[p]) * freq
                            if delta:
                                change_pair(*p, delta)
                        for p in before.keys() - after.keys():
                            db.execute("DELETE FROM occurrences WHERE a=? AND b=? AND w=?", (*p, wid))
                        for p in after.keys() - before.keys():
                            db.execute("INSERT INTO occurrences VALUES (?,?,?)", (*p, wid))
                        db.execute("UPDATE words SET seq=? WHERE id=?", (pack(new), wid))
                        last_word = wid
                db.execute("INSERT INTO merges VALUES (?,?,?)", (merge_count, a, b))
                merge_count += 1
                checkpoint_time()
            if merge_count % 100 == 0 or time.perf_counter() - last_print >= 10:
                print(f"merging: vocab={len(vocab):,}/{args.vocab_size:,}, "
                      f"pair_frequency={frequency:,}", flush=True)
                last_print = time.perf_counter()
        with db:
            set_meta(db, "phase", "done")
            checkpoint_time()
        merges = [(vocab[a], vocab[b]) for a, b in db.execute("SELECT a,b FROM merges ORDER BY id")]
        artifact = {"vocab": vocab, "merges": merges, "special_tokens": [EOS.decode()]}
        temporary = args.out / "tokenizer.tmp"
        with temporary.open("wb") as stream:
            pickle.dump(artifact, stream)
        os.replace(temporary, args.out / "tokenizer.pkl")
        longest = max((v for k, v in vocab.items() if k != 0), key=lambda v: (len(v), v))
        report = {
            "source": source, "scope": "entire input file; no corpus subsampling",
            "requested_vocab_size": args.vocab_size, "actual_vocab_size": len(vocab),
            "merges": len(merges), "unique_pretokens": db.execute("SELECT COUNT(*) FROM words").fetchone()[0],
            "pretoken_occurrences": get_meta(db, "pretoken_occurrences"),
            "training_wall_seconds_across_sessions": get_meta(db, "elapsed_seconds"),
            "current_process_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
            "longest_ordinary_token": {"byte_length": len(longest), "bytes_repr": repr(longest),
                                       "hex": longest.hex(), "text_with_replacement": longest.decode("utf-8", "replace")},
        }
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        report["database_bytes"] = database.stat().st_size
        temporary = args.out / "summary.tmp"
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, args.out / "summary.json")
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 ** 2)
    parser.add_argument("--max-document-mib", type=int, default=64)
    args = parser.parse_args()
    if args.vocab_size < 257 or args.chunk_bytes < 1 or args.max_document_mib < 1:
        parser.error("Require vocab-size >=257, positive chunk-bytes and max-document-mib")
    train(args)


if __name__ == "__main__":
    main()
