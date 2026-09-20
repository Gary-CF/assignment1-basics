import re
import regex

# GPT-2 预切分正则与训练端共享一份（从 bpe_tokenizer 导入）
from cs336_basics.bpe_tokenizer import GPT2_PRETOKEN_PATTERN


class Tokenizer:
    """BPE tokenizer：由训练产物 (vocab, merges) 驱动的编解码器。

    vocab: dict[int, bytes]     id → 字节序列（不要求 id 有序，测试会把
                                缺失的特殊 token 追加到末尾）
    merges: list[tuple[bytes, bytes]]  施工日志，下标即优先级
    special_tokens: 特殊 token 字符串列表，可为 None（不启用特殊匹配）
    """

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        # 反查表：bytes → id。encode 的终点、decode 的起点
        self.id_of = {tok: i for i, tok in vocab.items()}
        # rank 表：pair → 下标。merges 的倒排，编码机的燃料
        self.rank = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = list(special_tokens) if special_tokens else []
        if self.special_tokens:
            # 最长匹配：长的排前面，交替匹配时长者优先
            toks = sorted(self.special_tokens, key=len, reverse=True)
            self._special_re = re.compile(
                "(" + "|".join(re.escape(t) for t in toks) + ")"
            )
            self._special_set = set(self.special_tokens)
        else:
            self._special_re = None
            self._special_set = set()

    # ---------- 核心：rank 机 ----------
    def _encode_pretoken(self, pretoken: str) -> list[int]:
        """单个 pretoken → id 列表：反复焊当前最小 rank 的 pair。"""
        word = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
        while len(word) > 1:
            # 扫描相邻 pair，找 rank 最小者（平手天然取最左：严格 < 才更新）
            best_rank, best_i = None, -1
            for i in range(len(word) - 1):
                r = self.rank.get((word[i], word[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:          # 无可合并 pair → 停机
                break
            a, b = word[best_i], word[best_i + 1]
            merged = a + b
            # 双指针：该 pair 的全部出现，一次焊完（与训练替换同款语义）
            nw, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    nw.append(merged)
                    i += 2
                else:
                    nw.append(word[i])
                    i += 1
            word = tuple(nw)
        return [self.id_of[t] for t in word]

    # ---------- encode：三步流水线 ----------
    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        segments = re.split(self._special_re, text) if self._special_re else [text]
        for seg in segments:
            if seg in self._special_set:               # 特殊 token：整体直查 id
                ids.append(self.id_of[seg.encode("utf-8")])
                continue
            for m in regex.finditer(GPT2_PRETOKEN_PATTERN, seg):  # GPT-2 正则切词
                ids.extend(self._encode_pretoken(m.group()))      # rank 机
        return ids

    # ---------- decode：拼接 + 一次 UTF-8 解码 ----------
    def decode(self, ids) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8",errors="replace")

    # ---------- encode_iterable：流式 + 安全切点 ----------
    def encode_iterable(self, file_object, chunk_size: int = 32768):
        """文件对象进，id 逐个 yield；缓冲区有界（显存测试：RSS 增量 ≤1MB）。"""
        buf = ""
        while True:
            chunk = file_object.read(chunk_size)
            if not chunk:
                break
            buf += chunk
            # 安全切点 = 最后一个"纯空白 pretoken"的起点：
            # 它前面必是完整 pretoken（不同类不粘连），它本身可能延伸 → 留在缓冲
            cut = None
            for m in regex.finditer(GPT2_PRETOKEN_PATTERN, buf):
                if m.group().isspace():
                    cut = m.start()
            if cut is None:
                continue                       # 本块无可切点，继续读
            prefix, buf = buf[:cut], buf[cut:]
            for _id in self.encode(prefix):
                yield _id
        # 流末尾是天然边界，剩余全部 flush
        for _id in self.encode(buf):
            yield _id