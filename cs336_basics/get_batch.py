import numpy as np
import torch


def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str):
    """从 1D token 流随机采 (batch_size, context_length) 的 (输入, 标签) 对，y = x 右移一格。"""
    n = len(dataset)
    # 起点 ∈ [0, n - context_length - 1]：保证 y 的最后一个下标 start+context_length ≤ n-1
    starts = np.random.randint(0, n - context_length, size=batch_size)
    x = np.stack([dataset[s : s + context_length] for s in starts])
    y = np.stack([dataset[s + 1 : s + context_length + 1] for s in starts])
    return (torch.from_numpy(x.astype(np.int64)).to(device),
            torch.from_numpy(y.astype(np.int64)).to(device))