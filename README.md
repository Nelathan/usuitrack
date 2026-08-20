# UsuiTrack

A PyTorch optimizer for matrix parameters that keeps a rank-limited, moving
low-rank basis instead of a full-size first or second moment. It targets full
fine-tuning and continued pretraining on consumer GPUs, where AdamW's two
full-size moments per matrix are often the thing standing between you and a
bigger model or a bigger batch.

UsuiTrack is not a LoRA replacement in the usual sense. It doesn't freeze a
low-rank parameterization of the weight; it tracks a moving rank-`r` subspace
of the *gradient* and updates the full weight through it, every step. The
subspace itself adapts as training proceeds.

## Result

Single controlled comparison, matched matrix-parameter set, matched optimizer
state budget, `LiquidAI/LFM2.5-350M-Base`, 1k steps:

| | target loss | source retention loss | peak allocated VRAM | matrix state |
|---|---|---|---|---|
| LoRA rank 44 + AdamW | 1.674265 | 3.084880 | 3.81 GB | 98.9 MB |
| UsuiTrack rank 128 | **1.672156** | **3.037444** | **2.88 GB** | 97.4 MB |

LoRA leads target loss early (step 100: 1.770 vs 1.876), UsuiTrack overtakes
around step 900, and LoRA runs about 1% faster per step. Full methodology and
the evidence trail this survived are in [`docs/SPEC.md`](docs/SPEC.md).

## Install

Not yet on PyPI:

```bash
uv add git+https://github.com/Nelathan/usuitrack
```

Requires `torch>=2.1`, Python `>=3.11`. Pure PyTorch: no other runtime
dependency.

## Use

UsuiTrack only accepts 2D matrix parameters. Give it every `Linear`-style
weight and route everything else, biases, norms, and *embeddings* (2D but
not what a rank-limited update wants), to a separate optimizer: the same
split Muon and other orthogonalized optimizers use.

```python
import torch
from torch import nn
from usuitrack import UsuiTrack

embedding_params = {id(p) for m in model.modules() if isinstance(m, nn.Embedding) for p in m.parameters()}
matrix_params = [p for p in model.parameters() if p.ndim == 2 and id(p) not in embedding_params]
other_params = [p for p in model.parameters() if p.ndim != 2 or id(p) in embedding_params]

matrix_opt = UsuiTrack(matrix_params, lr=4e-4, rank=128)
other_opt = torch.optim.AdamW(other_params, lr=1e-4, betas=(0.9, 0.99))

loss = model(**batch).loss
loss.backward()
matrix_opt.step()
other_opt.step()
matrix_opt.zero_grad()
other_opt.zero_grad()
```

An optional layerwise-release mode drops each matrix gradient right after
it's consumed instead of holding the whole backward's gradients in memory at
once:

```python
matrix_opt = UsuiTrack(matrix_params, lr=4e-4, rank=128, release_matrix_grads=True)
```

This requires one backward per `step()` call, no gradient accumulation, and
it isn't transactional: see `examples/` and `docs/SPEC.md` for the exact
contract.

See [`examples/`](examples) for a plain-PyTorch loop, an HF `Trainer`
integration, and a full-finetune run through Unsloth's `FastModel` loader
(no LoRA, no quantization: UsuiTrack drives the full weights directly).

## How it works

[`docs/SPEC.md`](docs/SPEC.md) is the precise mathematical specification:
coordinates, the per-step pipeline, and the design decisions with their
reasons. [`docs/LEGEND.md`](docs/LEGEND.md) is a long-form narrative account
of the same mechanism, written for intuition rather than reference. The two
are meant to be read together and are checked against each other.

In short: each matrix keeps a small orthonormal basis (`rank` columns) and a
projected first moment in that basis. The basis moves via an exact
Grassmann-geodesic step driven by an Oja covariance tangent, so momentum
transports through basis motion as a rigid rotation instead of being
re-projected and losing energy. Direction comes from a leverage-balanced
Newton-Schulz polar map (Aurora), using the optimal coefficient schedule from
Amsel, Persson, Musco, and Gower's ["The Polar
Express"](https://arxiv.org/abs/2505.16932); scale comes from the original
parameter shape, Muon-style.

## Status and limitations

This is an early release. The core algorithm is measured (see above); the
generic-PyTorch integration surface is not yet fully swept:

- Tested on a single GPU. No DDP, FSDP, or other distributed testing.
- `release_matrix_grads` requires exactly one backward per `step()`. Plain
  gradient accumulation (no release, multiple `backward()` calls before one
  `step()`) is tested and works.
- Not tested under `autocast`/`GradScaler`, `torch.compile`'s `capturable`/
  differentiable modes, or exotic tensor subclasses.
- Minimum PyTorch version (`2.1`) is inferred from the API surface used
  (`register_post_accumulate_grad_hook`), not empirically swept across
  versions.

If you hit one of these, open an issue. This list is the honest starting
point, not a promise every box will stay unchecked.

## License

MIT. See [`LICENSE`](LICENSE).
