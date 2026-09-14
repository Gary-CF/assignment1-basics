import re
import regex

from collections import Counter

GPT2_PRETOKEN_PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)"   # 1 缩写尾巴: 's 'd 'm 't 'll 've 're
    r"| ?\p{L}+"              # 2 字母串（含单前导空格，汉字也算 \p{L}）
    r"| ?\p{N}+"              # 3 数字串
    r"| ?[^\s\p{L}\p{N}]+"    # 4 标点/符号串
    r"|\s+(?!\S)"             # 5 行尾/成段空白（不吞掉词前空格）
    r"|\s+"                   # 6 其余空白（词前的空白）
)

def load_text(path: str) ->str:
     """磁盘 bytes -> str。二进制读入再 decode，让编码问题暴露而不是静默出错。"""
     with open(path,"rb") as f:
          return f.read().decode("utf-8")

def split_special_tokens(text:str,special_tokens:list[str])->list[str]:
     """
    按特殊 token 切段，保留它们本身。
    返回"交替序列"：偶数位是普通文本段，奇数位是特殊 token 本体。
    """
     if not special_tokens:
          return [text]
     pattern = "(" + "|".join(re.escape(tok) for tok in special_tokens) + ")"
     return re.split(pattern, text)

def build_word_counts(segments: list[str], special_tokens:list[str])->dict[tuple[bytes,...],int]:
     """
    交替序列 -> 词频表（核心数据结构：dict[词的字节元组, 出现次数]）。
    """
     special_set = set(special_tokens)
     counts: dict[tuple[bytes,...],int]={}

     for seg in segments:
          if seg in special_set:
               # 特殊 token：作为整体进表（单 token 的词，对合并循环无影响）
               key=(seg.encode("utf-8"),)
               counts[key] = counts.get(key,0) + 1
               continue
           # finditer 流式切词：切一个计一个，不把全部词存成列表（PDF 原文要求）
          for m in regex.finditer(GPT2_PRETOKEN_PATTERN,seg):
                word = m.group()
                key=tuple(bytes([b]) for b in word.encode("utf-8"))

                counts[key] = counts.get(key,0)+1
     return counts

def count_pairs(
          word_counts:dict[tuple[bytes,...],int],
)->dict[tuple[bytes,bytes],int]:
     """遍历每个词元组的相邻 pair，按词频加权累加。"""
     pair_counts:dict[tuple[bytes,bytes],int] = {}
     for word, freq in word_counts.items():
          # zip 让"序列自己"和"自己错开一位的序列"并行：
         # (l,o,w) × (o,w) → (l,o), (o,w) —— 恰好是所有相邻对
         for a,b in zip(word,word[1:]):
              pair = (a,b)
              pair_counts[pair] = pair_counts.get(pair,0) + freq
     return pair_counts

def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    text = load_text(input_path)
    segments = split_special_tokens(text, special_tokens)
    word_counts = build_word_counts(segments, special_tokens)

    vocab: dict[int, bytes] = {}
    next_id = 0
    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1
    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1
    merges: list[tuple[bytes, bytes]] = []

    # ---- 三张索引表（一次性全量构建，之后只增量维护）----
    pair_to_freq: dict[tuple[bytes, bytes], int] = {}      # pair -> 频率
    pair_to_words: dict[tuple[bytes, bytes], set] = {}     # pair -> 含它的词（反查）
    freq_to_pairs: dict[int, set] = {}                     # 频率 -> 该频率的 pair 集合
    max_freq = 0
    for word, freq in word_counts.items():
        for i in range(len(word) - 1):
            p = (word[i], word[i + 1])
            pair_to_freq[p] = pair_to_freq.get(p, 0) + freq
            pair_to_words.setdefault(p, set()).add(word)
    for p, f in pair_to_freq.items():
        freq_to_pairs.setdefault(f, set()).add(p)
    if pair_to_freq:
        max_freq = max(pair_to_freq.values())

    def bump(p: tuple[bytes, bytes], delta: int) -> None:
        """pair 频率 += delta，同步迁移频率桶。"""
        nonlocal max_freq
        f = pair_to_freq.get(p, 0)
        if f:
            freq_to_pairs[f].discard(p)
        nf = f + delta
        if nf > 0:
            pair_to_freq[p] = nf
            freq_to_pairs.setdefault(nf, set()).add(p)
            if nf > max_freq:
                max_freq = nf
        else:
            pair_to_freq.pop(p, None)

    # ---- 主循环：选最优 → 只处理受影响词 ----
    while len(vocab) < vocab_size and pair_to_freq:
        # 选最优：桶顶取集合内 max，bytes 元组比较 = 字典序大者，C 速度
        while not freq_to_pairs.get(max_freq):
            max_freq -= 1
        a, b = max(freq_to_pairs[max_freq])
        merged = a + b

        # 快照迭代：取名单时不删索引，循环结束后 (a,b) 的集合必然已空
        affected = list(pair_to_words.get((a, b), ()))
        for word in affected:
            freq = word_counts.pop(word)
            # 词内 pair 去重计数：撤销按 出现次数×词频 扣，
            # 但反向索引里每词每 pair 只摘一次
            for p, cnt in Counter(zip(word, word[1:])).items():
                bump(p, -freq * cnt)
                s = pair_to_words[p]
                s.discard(word)
                if not s and p != (a, b):
                    del pair_to_words[p]
            # 替换（与朴素版同款双指针）
            nw: list[bytes] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    nw.append(merged)
                    i += 2
                else:
                    nw.append(word[i])
                    i += 1
            new_word = tuple(nw)
            word_counts[new_word] = word_counts.get(new_word, 0) + freq
            for p, cnt in Counter(zip(new_word, new_word[1:])).items():
                bump(p, freq * cnt)
                pair_to_words.setdefault(p, set()).add(new_word)
        pair_to_words.pop((a, b), None)  # 统一收尸

        merges.append((a, b))
        vocab[next_id] = merged
        next_id += 1

    return vocab, merges




if __name__ == "__main__":
    vocab, merges = train_bpe("tiny_corpus.txt", 270, ["<|endoftext|>"])
    print("len(vocab) =", len(vocab))          # 应为 270
    print("len(merges) =", len(merges))        # 应为 13
    print("前 3 条 merges:", merges[:3])
    print("后 3 条 merges:", merges[-3:])
    print("vocab 末尾:", list(vocab.items())[-3:])

     


               