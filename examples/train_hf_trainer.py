"""Hugging Face Trainer integration for UsuiTrack.

Trainer drives a single optimizer object, so UsuiTrack (matrix weights) and
a fallback AdamW (everything else) get wrapped behind one object that
presents Trainer's expected `torch.optim.Optimizer` surface and routes
each call to both.

`nn.Embedding` weights are 2D but stay out of UsuiTrack. Each row of an
embedding table is its own independent vector, not part of a shared linear
map, so a shared low-rank update doesn't describe what the table needs.
Route embeddings to the fallback optimizer alongside biases and norms.

Run: uv run --with transformers examples/train_hf_trainer.py [--release-matrix-grads]
"""

from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments

from usuitrack import UsuiTrack


class SplitOptimizer(torch.optim.Optimizer):
    """Combines a matrix optimizer and a fallback optimizer behind one
    Optimizer-shaped object so Trainer can drive both with a single call."""

    def __init__(self, matrix_opt: UsuiTrack, other_opt: torch.optim.Optimizer):
        self.matrix_opt = matrix_opt
        self.other_opt = other_opt

    @property
    def param_groups(self):
        return self.matrix_opt.param_groups + self.other_opt.param_groups

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        self.matrix_opt.step()
        self.other_opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.matrix_opt.zero_grad(set_to_none=set_to_none)
        self.other_opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"matrix": self.matrix_opt.state_dict(), "other": self.other_opt.state_dict()}

    def load_state_dict(self, state_dict) -> None:
        self.matrix_opt.load_state_dict(state_dict["matrix"])
        self.other_opt.load_state_dict(state_dict["other"])


def build_optimizer(model: nn.Module, release_matrix_grads: bool) -> SplitOptimizer:
    embedding_params = {id(p) for m in model.modules() if isinstance(m, nn.Embedding) for p in m.parameters()}
    matrix_params = [p for p in model.parameters() if p.ndim == 2 and id(p) not in embedding_params]
    other_params = [p for p in model.parameters() if p.ndim != 2 or id(p) in embedding_params]

    matrix_opt = UsuiTrack(matrix_params, lr=4e-4, rank=8, release_matrix_grads=release_matrix_grads)
    other_opt = torch.optim.AdamW(other_params, lr=1e-4, betas=(0.9, 0.99))
    return SplitOptimizer(matrix_opt, other_opt)


class RandomTokenDataset(Dataset):
    """Synthetic causal-LM dataset: fixed-length random token sequences."""

    def __init__(self, vocab_size: int, seq_len: int, size: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        torch.manual_seed(idx)
        ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": ids, "labels": ids.clone()}


def train(release_matrix_grads: bool, steps: int = 20) -> float:
    vocab_size, seq_len = 256, 32
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=seq_len,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=0,
        eos_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    optimizer = build_optimizer(model, release_matrix_grads)

    args = TrainingArguments(
        output_dir="/tmp/usuitrack-hf-example",
        per_device_train_batch_size=8,
        max_steps=steps,
        logging_steps=steps,
        report_to=[],
        # release_matrix_grads requires exactly one backward() per optimizer
        # step -- gradient accumulation is incompatible with it.
        gradient_accumulation_steps=1,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=RandomTokenDataset(vocab_size, seq_len, size=steps * 8),
        optimizers=(optimizer, None),
    )
    result = trainer.train()
    return result.training_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-matrix-grads", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    final_loss = train(args.release_matrix_grads, args.steps)
    print(f"release_matrix_grads={args.release_matrix_grads} training_loss={final_loss:.4f}")
