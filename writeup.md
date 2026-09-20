# CS336 Assignment 1 实验报告草稿

状态：实现与本机全套测试已完成；baseline 正在训练。本文的理论题与已观测实验记录可用，标记“待补”的内容尚未完成，不能作为最终完整提交。

依据：本仓库 `cs336_assignment1_basics.pdf`。GPT-2 small/medium/large/XL 均指采用本作业架构的同规模模型，并非原版 GPT-2 的精确结构。算术复核：`uv run python scripts/resource_accounting.py`；SGD 实验复现：`uv run python scripts/sgd_sanity.py`。

## 1. 环境与验收证据

- 用户本机：NVIDIA GeForce RTX 5060 Laptop GPU，约 7.96 GiB 显存；PyTorch 2.11.0+cu130，uv 环境。
- 工作分支：`dev/bpe`。模型提交记录包括 `a554971`、`be0f84b`、`3f54a10`；训练工具及后续提交的最终 SHA 待填写。
- 本机已确认全套 pytest 通过；已知测试集合为 48 项，正常结果为 47 passed、1 expected xfail。提交报告时附最终终端摘要，不把 xfail 写成实现缺陷。
- `data/`、`.venv/`、`runs/`、W&B 运行产物不提交到源码仓库。报告中的精选曲线和样例需单独整理为提交材料。
- 沙盒额外验证：模型梯度、因果性、RoPE 广播、采样边界、CPU 断点恢复及模拟 OOM；本机已经完成 CUDA 训练烟测和实际 BPE 文本生成联调。

## 2. Unicode 与编码

### unicode1

(a) `chr(0)` 是 Unicode U+0000，即 NUL 控制字符。

(b) `repr(chr(0))` 显示转义形式 `'\x00'`；直接打印通常没有可见字形，但字符仍然存在。

(c) Python 字符串可以包含 NUL，拼接不会因此结束，`len("a" + chr(0) + "b") == 3`；显示效果与字符串中是否存在该字符是不同问题。

### unicode2

(a) UTF-8 对 ASCII 只用一个字节，适合英文占比较高的语料，也不存在 UTF-16/32 的字节序选择问题；它可以用固定 256 个字节符号表示 Unicode 文本。不过并非所有语言的 UTF-8 编码都比 UTF-16 更短。

(b) `"é".encode("utf-8") == b"\xc3\xa9"`。按单字节分别解码会在 `b"\xc3"` 上报错，因为一个 UTF-8 字符可以跨多个字节；应先拼接 token 对应的字节，再整体解码。

(c) `b"\xc3\x28"` 不是合法 UTF-8：`0xC3` 要求后面跟一个 `10xxxxxx` 的延续字节，而 `0x28` 不满足这一要求。

## 3. Tokenizer 实现与实验

### 已完成实现

- BPE：基础字节词表、特殊 token 隔离、正则预切分、按频率及规定的平局规则选择合并。
- 训练优化：增量 pair 计数、pair 到词的反查索引、频率桶。历史协作记录中的同输入计时为优化版约 0.43 s、朴素版约 2.93 s；这些是历史记录，不是本轮重新计时。
- 使用端：按 merge rank 执行合并，特殊 token 最长匹配，流式编码保留安全边界。
- 本机已通过参考 tokenizer 对拍和流式内存测试。单个 token 的字节序列未必是完整 UTF-8 字符，因此 decode 在拼接字节后使用 `errors="replace"`。

### train_bpe_tinystories

实际配置是使用约 300 MB TinyStories 子集训练 10,000 词表，然后编码完整训练/验证文本；不能把这个词表训练过程表述为“在全量训练文本上训练词表”。训练和使用时均包含 `<|endoftext|>`。

已确认产物：`tinystories_tokenizer.pkl` 保存 `vocab`、`merges`、`special_tokens`；两份 `.bin` 是无头部、原生字节序的 uint16 ID 流。

- 训练集：541,252,216 tokens，ID 范围 `[0, 9999]`。
- 验证集：5,466,269 tokens，ID 范围 `[0, 9999]`。
- 已检查解码预览为英文故事，验证集预览中可见 `<|endoftext|>`。
- 待补：BPE 训练实际耗时、峰值 RSS、最长 token、profiling 结果。不能用文件时间戳代替准确的进程耗时或内存测量。

### train_bpe_expts_owt / tokenizer_experiments

- 待补：OWT 32K tokenizer 与最长 token，和 TinyStories tokenizer 的比较。
- 待补：固定抽取各 10 个文档，报告 UTF-8 原始字节数除以 token 数，即 bytes/token；不要把字符数当字节数，也不要把 `.bin` 文件大小当原始文本大小。
- 待补：用 TinyStories tokenizer 编码 OWT 样本，测量跨域压缩变化。
- 待补：编码吞吐 `原始字节数/编码秒数`；若吞吐为 r bytes/s，则 825 GB 文本估计需 `825e9/r` 秒，须注明十进制 GB。
- uint16 可表示 ID 0 至 65,535，足够容纳 10K/32K 词表，每个 ID 占 2 字节。训练时再将小批量 ID 转换为 int64。

## 4. Transformer LM resource accounting

### 4.1 统一符号与计算口径

V 为词表大小，T 为序列长度，L 为层数，D 为模型维度，H 为注意力头数，F 为 FFN 隐藏维度，B 为 batch size，单头维度 d=D/H。

本实现所有 Linear 无 bias，输入 embedding 与 LM head 不共享权重，RoPE 无可训练参数。FP32 每元素 4 字节；1 GB=10^9 bytes，1 GiB=2^30 bytes。

矩阵乘法 `(m,n) @ (n,p)` 按 2mnp FLOPs 计。下文前向 FLOPs 只计题目要求的矩阵乘法，不计 RMSNorm、RoPE、SiLU、softmax、残差和索引等操作。普通 dense attention 仍计算完整 T×T 分数矩阵；因果 mask 不使本实现的矩阵乘法 FLOPs 减半。

### 4.2 参数量推导

| 模块 | 参数量 |
|---|---:|
| token embedding | VD |
| 每层 Q、K、V、输出投影 | 4D² |
| 每层 SwiGLU 三个矩阵 | 3DF |
| 每层两个 RMSNorm | 2D |
| 最终 RMSNorm | D |
| LM head | VD |

因此总参数量为：

$$P=2VD+L(4D^2+3DF+2D)+D.$$

当前 TinyStories 配置 V=10,000，T=256，L=4，D=512，H=16，F=1344：

| 部分 | 参数量 |
|---|---:|
| 输入 embedding | 5,120,000 |
| LM head | 5,120,000 |
| 所有注意力投影 | 4,194,304 |
| 所有 FFN | 8,257,536 |
| 所有 RMSNorm | 4,608 |
| 总计 | **22,696,448** |

FP32 参数本身占 90,785,792 bytes，约 0.08455 GiB。不能把它当成完整训练显存。

当 F≈8D/3 时，FFN 约有 8D² 参数，与 attention 的 4D² 相比约占单个 block 主要矩阵参数的 2/3。若误用 F=4D，则 FFN 变为 12D²，占比约为 3/4；两种口径不能混用。

### transformer_accounting (a)

代入 GPT-2 XL-shaped 配置 V=50,257，T=1024，L=48，D=1600，H=25，F=4288，可得 **1,640,452,800 个参数**。仅加载 FP32 参数需要 **6,561,811,200 bytes，即 6.562 GB 或 6.111 GiB**。

### transformer_accounting (b)

以下按 B=1 的一条完整序列计算；多个 batch 样本时整体乘 B。

| 矩阵乘法 | 每层/每次 FLOPs | GPT-2 XL 全模型 FLOPs |
|---|---:|---:|
| Q、K、V 与 attention 输出投影 | 每层 8TD² | 1,006,632,960,000 |
| QKᵀ 与 attention probabilities × V | 每层 4T²D | 322,122,547,200 |
| SwiGLU 三个投影 | 每层 6TDF | 2,023,332,249,600 |
| LM head | 最后一次 2TDV | 164,682,137,600 |
| 总计 | L(8TD²+4T²D+6TDF)+2TDV | **3,516,769,894,400** |

Embedding 是查表，不计作一个 VD 大小矩阵乘法；RoPE 是逐元素旋转，也不计入本题矩阵乘法项。

### transformer_accounting (c)

在 GPT-2 XL、T=1024 的设置中，FFN 占约 **57.53%**，是最大的矩阵乘法开销，其次是注意力的四个投影（约 **28.62%**）；QKᵀ 和加权求和合计约 **9.16%**。

### transformer_accounting (d)

保持 V=50,257、T=1024，按离 8D/3 最近的 64 的倍数选取 F。

| 模型 | L / D / H / F | 前向 FLOPs | attention 投影 | QKᵀ + PV | FFN | LM head |
|---|---|---:|---:|---:|---:|---:|
| small | 12 / 768 / 12 / 2048 | 291,648,307,200 | 19.88% | 13.25% | 39.76% | 27.10% |
| medium | 24 / 1024 / 16 / 2752 | 830,172,299,264 | 24.83% | 12.42% | 50.05% | 12.70% |
| large | 36 / 1280 / 20 / 3392 | 1,768,530,903,040 | 27.32% | 10.93% | 54.30% | 7.45% |
| XL | 48 / 1600 / 25 / 4288 | 3,516,769,894,400 | 28.62% | 9.16% | 57.53% | 4.68% |

当 D 和 L 增加且 T 固定时，层内投影/FFN 近似按 LD² 增长，占比上升；LM head 不随层数增加，因此占比下降。QKᵀ 与 PV 按 LD 增长，相对 LD² 项的占比下降。

### transformer_accounting (e)

把 XL 的 T 从 1024 增至 16,384 后，前向矩阵乘法共 **133,577,729,638,400 FLOPs**，约为原来的 **37.98 倍**。线性于 T 的项增大 16 倍，而 QKᵀ/PV 的 T² 项增大 256 倍，后者占比从 9.16% 升至 **61.73%**，成为最大项；FFN 占比降至约 24.24%。

## 5. 学习率 toy experiment

本节是 PDF 的 `learning_rate_tuning`，与后面的语言模型学习率 sweep 是两个不同任务。

按 PDF 的 10×10 参数矩阵、L(w)=mean(w²)、步长 α/√(t+1) 实验，固定 seed=336，并对所有 α 使用同一个初值。以下为沙盒 PyTorch 2.11.0 CPU 实测；原示例打印每次更新之前的 loss，表格额外列出完成 10 次更新后的 loss。

| α | 初始 loss | 10 次更新后的 loss |
|---:|---:|---:|
| 1 | 24.936384 | 20.374941 |
| 10 | 24.936384 | 2.940821 |
| 100 | 24.936384 | 2.60509e-24 |
| 1000 | 24.936384 | 6.55267e19 |

α=10 比 α=1 收敛更快；α=100 在这个特定二次问题上也迅速下降，α=1000 在前 10 步明显发散。原因可从梯度 2w/100 看出：每步将 w 乘以 `1−α/(50√(t+1))`；不要把这个 toy 实验的数值阈值直接迁移到语言模型上。

## 6. AdamW resource accounting

### adamw_accounting (a)：参数、梯度、状态与激活

本题明确采用 F=8D/3 的近似，所以参数量为：

$$P=2VD+L(12D^2+2D)+D.$$

参数占 4P bytes，梯度占 4P bytes，两份 AdamW moment 占 8P bytes，总静态存储为 16P bytes。

激活采用以下明确的简化口径：对题目列出的每个操作保留一份完整输出；不单独计算 residual、RoPE、归一化统计量、反向临时空间、优化器临时空间和 CUDA 工作区。交叉熵额外按一个 BTV 大小的张量估算。这是算术估计，不是对实际 autograd 峰值分配的精确模拟。

| 每层需要计入的激活 | 元素数 |
|---|---:|
| 两个 RMSNorm 输出 | 2BTD |
| Q、K、V 投影输出 | 3BTD |
| QKᵀ 和 softmax 输出 | 2BHT² |
| 加权 value 与 attention 输出投影 | 2BTD |
| FFN 的 W1、W3、SiLU、逐元素乘积 | 4BTF |
| FFN 的 W2 输出 | BTD |
| 每层合计 | 8BTD+4BTF+2BHT² |

再计入最终 RMSNorm 的 BTD、LM head 的 BTV、交叉熵的 BTV，得到激活总元素数：

$$A=B\left[L\left(\frac{56}{3}TD+2HT^2\right)+TD+2TV\right].$$

简化的训练峰值存储估算为：

$$M(B)=16P+4A\quad\text{bytes}.$$

梯度累积时，该式激活部分应代入 micro-batch，而不是有效 batch。参数、梯度和 AdamW 状态不会因为累积次数增加而复制 K 份。

### adamw_accounting (b)

仍按本节指定 F=8D/3，而非前节取整后的 4288，XL-shaped 参数量为 **1,635,537,600**。

$$M(B)=16,356,614,144\,B+26,168,601,600\quad\text{bytes}.$$

若 80 GB 指 80×10^9 bytes，则最大整数 B 为 **3**；若按 80 GiB 计算，结果也为 3。实际可用 batch 仍取决于内核、激活保存、精度与额外临时空间，不能把该简化结果当作实测显存保证。

### adamw_accounting (c)

AdamW 单次更新为 Θ(P)。若按 PDF 的调整学习率写法，m 更新约 3P、v 更新约 4P、直接写出的 weight decay 约 2P、最后 sqrt/add/divide/multiply/subtract 约 5P，共约 **14P FLOPs**，忽略每个参数张量的标量系数计算。

若先把 weight decay 合并成一个乘法，会减少约 P 操作；本项目显式计算 m_hat/v_hat 并使用合并 weight decay 的写法约 **15P**。sqrt/divide 在这里各按一次标量操作计，不代表它们与乘加具有相同执行耗时。

### adamw_accounting (d)

采用前节取整 F=4288 的 XL-shaped 前向计数，并遵循题设“反向为前向两倍”，batch=1024、400,000 步的矩阵乘法计算量为：

$$3\times3,516,769,894,400\times1024\times400,000
=4.32140684623872\times10^{21}\ \text{FLOPs}.$$

按题设峰值 495 TFLOP/s 和 MFU=50%，有效算力为 247.5×10^12 FLOP/s，因此需约 **4,850.06 小时（202.09 天）**。这是题设模型下的算术估计，不是当前 RTX 5060 Laptop 训练的 ETA；忽略数据读取、验证、checkpoint 与 AdamW 等额外时间。

## 7. 训练与生成：已观测记录

### 7.1 训练实现

`run_train.py` 支持 memmap、micro-batch 梯度累积、BF16 autocast/FP32 参数与 optimizer moments、余弦调度、梯度裁剪、独立固定验证抽样、JSONL 和 W&B 离线日志。

checkpoint 保存模型、optimizer、iteration、模型/运行配置、训练采样 RNG、PyTorch RNG、数据标识和累计运行时间。写入临时文件后原子替换；OOM 时保留上一次完整 checkpoint，不将可能部分更新的 optimizer 状态作为恢复点。

CPU 合成数据的独立验证中，连续训练 60 步与 23 步后恢复至 60 步的模型、optimizer 和采样 RNG 完全一致。该结论不能扩大为“所有 CUDA 内核均逐 bit 可复现”。

### 7.2 smoke run

配置：D=128，L=2，H=4，F=384，T=128，micro-batch=2，累积次数=2，100 步，共 51,200 tokens。

| 指标 | 本机观测 |
|---|---:|
| 初始验证 loss | 9.2244596481 |
| 第 100 步验证 loss | 7.1990871429 |
| 第 100 步训练日志 loss | 7.1768598557 |
| 第 100 步近期吞吐 | 25,478.35 tokens/s |
| 第 100 步峰值已分配显存 | 0.09934 GiB |

从 step 100 的 checkpoint，在 CPU 上用 temperature=0.8、top-p=0.95、prompt=`Once upon a time` 成功生成 32 个新 token，并保存 TXT/JSON。结果包含英文词和不合理的拼接，尚不连贯；它验证了编码→采样→解码→保存链路，不作为最终模型的流畅性证明，也不满足最终 256-token 样例要求。

### 7.3 baseline（训练中）

启动配置：V=10,000，T=256，D=512，L=4，H=16，F=1344；micro-batch=4，梯度累积=8，有效 batch=32；AdamW lr_max=3e-4，lr_min=3e-5，β=(0.9,0.95)，eps=1e-8，weight_decay=0.1；warmup=200，计划 40,000 步，即 327,680,000 tokens。以实际 `config-from-step-*.json` 为最终配置证据。

| step | 已处理 tokens | 验证 loss | 训练日志 loss |
|---:|---:|---:|---:|
| 20 | 163,840 | 8.9229203701 | 9.0752250671 |
| 40 | 327,680 | 8.0276911974 | 8.2892579079 |
| 3060 | 25,067,520 | **截图未提供** | 1.9631813452 |

step 3020–3060 的近期训练吞吐约为 45,154–50,049 tokens/s；峰值已分配显存约 0.7313 GiB。该显存指标来自 `torch.cuda.max_memory_allocated`，不等于 GPU 总占用或 allocator reserved memory。

训练 loss 是最近日志窗口内训练 batch 损失的平均，验证 loss 来自固定抽样的验证序列。二者来自不同数据，不能把训练 loss≈1.96 写成验证 loss≈1.96。

待补最终 step、最佳验证 loss 及所在步数、完整耗时、loss.png/loss.svg、checkpoint 路径和至少 256 个新 token（或生成 EOS 为止）的文本样例。尚不能声称达到了题目要求的 TinyStories validation loss≤1.45。

### 7.4 生成方法与待补评价

`sample_next_token` 支持 temperature=0 的贪心模式、正温度 softmax 与 top-p。Top-p 保留累计概率达到阈值所需的最小前缀，包含跨过阈值的 token，然后重新归一化。

`generate` 每步读取最后位置的 logits，追加新 token，遇到 EOS 或长度上限停止；超出 context_length 时使用最近的上下文窗口。样例元数据记录 checkpoint 步数、温度、top-p、seed 和生成 token IDs。

最终流畅性分析至少讨论两个因素：模型训练程度/数据分布，以及 temperature/top-p 对随机性和重复的影响；也可讨论上下文窗口限制。正式评价必须引用实际生成文本，不能以这些一般性因素代替观察结果。

## 8. 尚未完成的完整作业实验

| 作业项 | 当前状态 | 所需证据 |
|---|---|---|
| TinyStories tokenizer 资源、最长 token、profiling | 待补 | 测量结果与解释 |
| OWT 32K tokenizer 与跨域压缩实验 | 待补 | 两种词表和固定样本统计 |
| LM 学习率 sweep 与发散边界 | 仅 baseline lr=3e-4 在跑 | 多学习率曲线及至少一次发散证据 |
| batch size 对比 | 未开展正式对照 | 多有效 batch 的训练曲线 |
| 无 RMSNorm | 未实现/未运行 | 与 baseline 及较低 lr 对照 |
| post-norm | 未实现/未运行 | 与 pre-norm 对照 |
| NoPE | 未运行实验 | 与 RoPE 对照 |
| 无门控 SiLU FFN，F=4D | 未实现/未运行 | 与 SwiGLU 近似参数匹配对照 |
| OWT LM 实验 | 未运行 | OWT 曲线/生成，或明确说明低资源替代范围 |
| leaderboard | 未运行 | 单独说明是否参与，不声称提交 |

完整作业不能仅凭 pytest 全绿与一条 baseline 曲线宣称完成。短跑可用于筛选超参数，但最终对照应明确模型、数据、有效 batch、token 预算及学习率日程；不同预算的曲线不能直接归因于架构差异。

## 9. 最终提交前补齐

- 最终 pytest 输出与 git commit/tag；确认工作区和大文件管理。
- 最终实验配置、独立验证结果、训练曲线、生成文本及其 JSON 元数据。
- 上述所有待补实验的实际证据；若使用低资源替代，准确声明适用范围。
- 把本草稿中的“训练中”和“待补”逐项更新为真实结果后，再给最终版本打标签。
