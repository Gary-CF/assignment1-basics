import re
import regex

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
          input_path: str,vocab_size:int,special_tokens:list[str]
)->tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
     text = load_text(input_path)
     segments = split_special_tokens(text,special_tokens)
     word_counts = build_word_counts(segments,special_tokens)

     # id 分配约定：特殊 token 占 0 起的前几个，256 个字节紧随其后，
     # 合并产物从 next_id 继续——这就是为什么 merges 顺序就是规格
     vocab: dict[int,bytes] = {}
     next_id = 0
     for tok in special_tokens:
          vocab[next_id] = tok.encode("utf-8")
          next_id += 1
     for i in range(256):
          vocab[next_id] = bytes([i])
          next_id += 1

     merges: list[tuple[bytes,bytes]] = []

     # 合并主循环
     while len(vocab) < vocab_size:
          pair_counts = count_pairs(word_counts)

          if not pair_counts:
               break

          best_pair, _best_freq = max(
               pair_counts.items(), key=lambda kv: (kv[1], kv[0])
          )
          a,b = best_pair
          merged = a + b

          new_word_counts: dict[tuple[bytes,...],int]={}
          for word,freq in word_counts.items():
               new_word:list[bytes] = []
               i = 0
               while i < len(word):
                    if i < len(word) - 1 and word[i] == a and word[i+1]==b:
                         new_word.append(merged)
                         i+=2
                    else:
                         new_word.append(word[i])
                         i+=1
               new_key = tuple(new_word)
               new_word_counts[new_key] = new_word_counts.get(new_key,0)+freq
          word_counts = new_word_counts

          merges.append((a,b))
          vocab[next_id] = merged
          next_id += 1
     return vocab,merges




if __name__ == "__main__":
    vocab, merges = train_bpe("tiny_corpus.txt", 270, ["<|endoftext|>"])
    print("len(vocab) =", len(vocab))          # 应为 270
    print("len(merges) =", len(merges))        # 应为 13
    print("前 3 条 merges:", merges[:3])
    print("后 3 条 merges:", merges[-3:])
    print("vocab 末尾:", list(vocab.items())[-3:])

     


               