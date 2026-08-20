"""Minimal ordinary-PyTorch training loop with UsuiTrack.

Shows the required split (UsuiTrack owns 2D matrix weights, a separate
optimizer owns everything else) and both gradient-lifecycle modes:

  --release-matrix-grads off (default): ordinary backward(), then step().
  --release-matrix-grads on: each matrix gradient is consumed and freed
      as soon as backward() produces it, via a post-accumulate-grad hook.
      This trades gradient accumulation for lower peak activation+gradient
      memory. Requires exactly one backward() per step().

Run: python examples/train_pytorch.py [--release-matrix-grads]
"""

from __future__ import annotations

import argparse

import torch
from torch import nn

from usuitrack import UsuiTrack


class ToyModel(nn.Module):
    def __init__(self, dim: int = 64, hidden: int = 128, depth: int = 3):
        super().__init__()
        layers = [nn.Linear(dim, hidden, bias=False), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden, bias=False), nn.LayerNorm(hidden), nn.GELU()]
        layers += [nn.Linear(hidden, dim, bias=False)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_optimizers(model: nn.Module, release_matrix_grads: bool) -> tuple[UsuiTrack, torch.optim.AdamW]:
    matrix_params = [p for p in model.parameters() if p.ndim == 2]
    other_params = [p for p in model.parameters() if p.ndim != 2]
    matrix_opt = UsuiTrack(
        matrix_params,
        lr=4e-4,
        rank=16,
        release_matrix_grads=release_matrix_grads,
    )
    other_opt = torch.optim.AdamW(other_params, lr=1e-4, betas=(0.9, 0.99))
    return matrix_opt, other_opt


def train(release_matrix_grads: bool, steps: int = 50) -> float:
    torch.manual_seed(0)
    model = ToyModel()
    matrix_opt, other_opt = build_optimizers(model, release_matrix_grads)

    loss = torch.tensor(float("nan"))
    for _ in range(steps):
        x = torch.randn(32, 64)
        target = torch.randn(32, 64)
        loss = nn.functional.mse_loss(model(x), target)
        loss.backward()

        # With release_matrix_grads=True, matrix gradients were already
        # consumed by the backward hook and param.grad is already None for
        # every matrix weight -- step() applies the retained pending work.
        matrix_opt.step()
        other_opt.step()
        matrix_opt.zero_grad()
        other_opt.zero_grad()

    return float(loss.detach())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-matrix-grads", action="store_true")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    final_loss = train(args.release_matrix_grads, args.steps)
    print(f"release_matrix_grads={args.release_matrix_grads} final_loss={final_loss:.4f}")
