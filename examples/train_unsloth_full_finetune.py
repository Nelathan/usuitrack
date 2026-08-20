"""UsuiTrack under Unsloth's loader, full fine-tune, no LoRA, no bitsandbytes.

Unsloth's `FastModel` usually wraps the base model in a PEFT LoRA adapter
plus 4-bit quantization. Neither leaves UsuiTrack anything to attach to:
LoRA freezes the base weight and trains a small delta instead, and a
quantized weight isn't a plain differentiable 2D tensor. Pass
`full_finetuning=True, load_in_4bit=False` and Unsloth hands back an
ordinary bf16 model with real weights, exactly what UsuiTrack expects.

Verified end to end on `LiquidAI/LFM2.5-350M-Base` (Unsloth ships a patched
Lfm2 path): 5 real steps, loss 5.31 -> 3.65, 1.86 GB peak on an RTX 4070
Super. That number is this toy's batch of 4 short sentences, not a claim
about real training; measure your own shapes.

Run: uv run --with "unsloth" --with torch examples/train_unsloth_full_finetune.py
     Needs a CUDA GPU. Unsloth pulls its own compatible torch build.
"""

from __future__ import annotations

import argparse

from unsloth import FastModel
import torch
from torch import nn

from usuitrack import UsuiTrack


def build_optimizers(model: nn.Module, release_matrix_grads: bool) -> tuple[UsuiTrack, torch.optim.AdamW]:
    embedding_params = {id(p) for m in model.modules() if isinstance(m, nn.Embedding) for p in m.parameters()}
    matrix_params = [p for p in model.parameters() if p.ndim == 2 and id(p) not in embedding_params]
    other_params = [p for p in model.parameters() if p.ndim != 2 or id(p) in embedding_params]

    matrix_opt = UsuiTrack(matrix_params, lr=4e-4, rank=32, release_matrix_grads=release_matrix_grads)
    other_opt = torch.optim.AdamW(other_params, lr=1e-4, betas=(0.9, 0.99))
    return matrix_opt, other_opt


def train(release_matrix_grads: bool, steps: int = 5) -> float:
    model, tokenizer = FastModel.from_pretrained(
        model_name="LiquidAI/LFM2.5-350M-Base",
        max_seq_length=256,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=True,
    )
    matrix_opt, other_opt = build_optimizers(model, release_matrix_grads)

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Liquid neural networks blend continuous dynamics with attention.",
        "A rank-limited optimizer tracks a moving subspace of the gradient.",
        "Consumer GPUs benefit from optimizer state that does not scale with model size.",
    ]
    batch = tokenizer(texts, return_tensors="pt", padding=True).to("cuda")
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100

    model.train()
    loss = torch.tensor(float("nan"))
    for step in range(steps):
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=labels)
        loss = out.loss
        loss.backward()
        matrix_opt.step()
        other_opt.step()
        matrix_opt.zero_grad()
        other_opt.zero_grad()
        print(f"step {step} loss {loss.item():.4f}")

    return float(loss.detach())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-matrix-grads", action="store_true")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    final_loss = train(args.release_matrix_grads, args.steps)
    print(f"release_matrix_grads={args.release_matrix_grads} final_loss={final_loss:.4f}")
