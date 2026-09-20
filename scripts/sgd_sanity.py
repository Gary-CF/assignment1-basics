"""The PDF's learning_rate_tuning toy experiment, with a shared seeded initial state."""
import json
import math

import torch


def main():
    torch.manual_seed(336)
    initial = 5 * torch.randn(10, 10)
    results = []
    for lr in (1., 10., 100., 1000.):
        weights = torch.nn.Parameter(initial.clone())
        losses = []
        for step in range(10):
            weights.grad = None
            loss = weights.square().mean()
            losses.append(float(loss.detach()))
            loss.backward()
            with torch.no_grad():
                weights.add_(weights.grad, alpha=-lr / math.sqrt(step + 1))
        results.append({"lr": lr, "printed_losses_before_updates": losses,
                        "loss_after_10_updates": float(weights.detach().square().mean())})
    print(json.dumps({"seed": 336, "results": results}, indent=2))


if __name__ == "__main__":
    main()
