"""TinyStories -> (vocab, merges, uint16 编码流 .bin)。一次性数据准备脚本。"""
import pickle
import time
from itertools import islice
from pathlib import Path

import numpy as np

from cs336_basics.bpe_tokenizer import train_bpe
from cs336_basics.tokenizer import Tokenizer

SPECIAL_TOKENS = ["<|endoftext|>"]
VOCAB_SIZE = 10_000          # 10k 规模；uint16 上限 65535，水位线以下
CHUNK = 1_000_000            # 每 100 万个 id 落一次盘（峰值内存 ≈ 2MB）

DATA = Path("data")


def save_tokenizer(vocab, merges, out: Path) -> None:
    """内部产物，pickle 最省事；将来要人读再换成 json/latin-1 文本格式。"""
    with open(out, "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges,
                     "special_tokens": SPECIAL_TOKENS}, f)


def encode_file_to_bin(tok: Tokenizer, text_path: Path, bin_path: Path) -> int:
    """流式编码：encode_iterable 进，uint16 裸流 .bin 出。返回 token 总数。"""
    total = 0
    with open(text_path) as fin, open(bin_path, "wb") as fout:
        it = tok.encode_iterable(fin)
        while True:
            chunk = np.fromiter(islice(it, CHUNK), dtype=np.uint16)  # 修复位：不传 count
            if chunk.size == 0:
                break
            chunk.tofile(fout)
            total += chunk.size
            print(f"  {bin_path.name}: {total:,} tokens...", flush=True)
    return total


def main() -> None:
    t0 = time.time()
    print("[1/3] 训练词表（10k merges，300MB 子集）...", flush=True)
    vocab, merges = train_bpe(str(DATA / "_vocab_300mb.txt"),   # 子集训词表，保内存
                              VOCAB_SIZE, SPECIAL_TOKENS)
    save_tokenizer(vocab, merges, DATA / "tinystories_tokenizer.pkl")
    print(f"  vocab={len(vocab)}, merges={len(merges)}, "
          f"{time.time()-t0:.0f}s", flush=True)

    tok = Tokenizer(vocab, merges, SPECIAL_TOKENS)

    print("[2/3] 编码 train（全量 1.7GB，流式）...", flush=True)
    n_train = encode_file_to_bin(tok, DATA / "TinyStoriesV2-GPT4-train.txt",
                                 DATA / "tinystories_train.bin")

    print("[3/3] 编码 valid...", flush=True)
    n_valid = encode_file_to_bin(tok, DATA / "TinyStoriesV2-GPT4-valid.txt",
                                 DATA / "tinystories_valid.bin")

    print(f"完成: train={n_train:,} tokens, valid={n_valid:,} tokens, "
          f"总计 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()