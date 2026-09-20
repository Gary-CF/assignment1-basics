import math

import torch
from torch.optim import Optimizer


def cross_entropy(logits, targets):
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    log_normalizer = shifted.exp().sum(dim=-1).log()
    correct_logits = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_normalizer - correct_logits).mean()


@torch.no_grad()
def clip_grad_norm(parameters, max_norm):
    if max_norm < 0:
        raise ValueError("max_norm must be nonnegative")
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    total_sq = torch.zeros((), device=grads[0].device, dtype=torch.float32)
    for grad in grads:
        total_sq.add_(grad.float().square().sum())
    total_norm = total_sq.sqrt()
    coefficient = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    for grad in grads:
        grad.mul_(coefficient)
    return total_norm


class AdamW(Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999),
        eps=1e-8, weight_decay=0.01,
    ):
        if lr < 0 or eps <= 0 or weight_decay < 0:
            raise ValueError("Invalid lr, eps, or weight_decay")
        if not all(0 <= beta < 1 for beta in betas) or len(betas) != 2:
            raise ValueError("betas must contain two values in [0, 1)")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p.mul_(1 - lr * weight_decay)
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
        return loss


def get_lr_cosine_schedule(
    it, max_learning_rate, min_learning_rate,
    warmup_iters, cosine_cycle_iters,
):
    if it < 0 or not 0 <= warmup_iters < cosine_cycle_iters:
        raise ValueError("Require it >= 0 and 0 <= warmup_iters < cosine_cycle_iters")
    if not 0 <= min_learning_rate <= max_learning_rate:
        raise ValueError("Require 0 <= min_learning_rate <= max_learning_rate")
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it >= cosine_cycle_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (
        max_learning_rate - min_learning_rate
    )