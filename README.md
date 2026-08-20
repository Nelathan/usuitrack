# UsuiTrack

A PyTorch optimizer for matrix parameters that keeps a rank-limited, moving
low-rank basis instead of a full-size first or second moment. It targets
fine-tuning and continued pretraining on consumer GPUs, where AdamW's two
full-size moments per matrix are often the thing standing between you and a
bigger model or a bigger batch.

UsuiTrack is not a LoRA replacement in the usual sense — it doesn't freeze a
low-rank parameterization of the weight. It tracks a moving rank-`r` subspace
of the *gradient* and updates the full weight through it, every step. The
subspace itself adapts as training proceeds.

## Result

Single controlled comparison, matched matrix-parameter set, matched optimizer
state budget, `LiquidAI/LFM2.5-350M-Base`, 1k steps:

| | target loss | source retention loss | peak allocated VRAM | matrix state |
|---|---|---|---|---|
| LoRA rank 44 + AdamW | 1.674265 | 3.084880 | 3.81 GB | 98.9 MB |
| UsuiTrack rank 128 | **1.672156** | **3.037444** | **2.88 GB** | 97.4 MB |

LoRA leads target loss early (step 100: 1.770 vs 1.876) and UsuiTrack
overtakes around step 900. LoRA was about 1% faster per step. Full
methodology and the evidence trail this survived are in
[`docs/SPEC.md`](docs/SPEC.md).

## Install

Not yet on PyPI. Install from source:

```bash
uv add git+https://github.com/Nelathan/usuitrack
# or
pip install git+https://github.com/Nelathan/usuitrack
```

Requires `torch>=2.1`, Python `>=3.11`. Pure PyTorch — no other runtime
dependency.

## Use

UsuiTrack only accepts 2D matrix parameters. Give it every `Linear`/`Conv`-style
weight and route everything else (biases, norms, embeddings) to a separate
optimizer — exactly the split Muon and other orthogonalized optimizers use.

```python
import torch
from usuitrack import UsuiTrack

matrix_params = [p for n, p in model.named_parameters() if p.ndim == 2]
other_params = [p for n, p in model.named_parameters() if p.ndim != 2]

matrix_opt = UsuiTrack(matrix_params, lr=4e-4, rank=128)
other_opt = torch.optim.AdamW(other_params, lr=1e-4, betas=(0.9, 0.99))

loss = model(**batch).loss
loss.backward()
matrix_opt.step()
other_opt.step()
matrix_opt.zero_grad()
other_opt.zero_grad()
```

For transformer weights specifically, an explicit residual-facing `side`
policy per parameter group beats the shape-only `auto` default — see
[`docs/SPEC.md`](docs/SPEC.md#coordinates) for which side each weight wants.

An optional layerwise-release mode drops each matrix gradient right after
it's consumed, instead of holding the full backward's gradients in memory at
once:

```python
matrix_opt = UsuiTrack(matrix_params, lr=4e-4, rank=128, release_matrix_grads=True)
```

This requires one backward per `step()` call (no gradient accumulation) and
is not transactional — see `examples/` and `docs/SPEC.md` for the exact
contract.

## How it works

[`docs/SPEC.md`](docs/SPEC.md) is the precise mathematical specification:
coordinates, the per-step pipeline, and the design decisions with their
reasons. [`docs/LEGEND.md`](docs/LEGEND.md) is a long-form narrative account
of the same mechanism, written for intuition rather than reference — the two
are meant to be read together and are checked against each other.

In short: each matrix keeps a small orthonormal basis (`rank` columns) and a
projected first moment in that basis. The basis moves via an exact
Grassmann-geodesic step driven by an Oja covariance tangent, so momentum
transports through basis motion as a rigid rotation rather than being
re-projected and losing energy. Direction comes from a leverage-balanced
Newton-Schulz polar map (Aurora); scale comes from the original parameter
shape (Muon-style).

## Status and limitations

This is an early release. The core algorithm is measured (see above); the
generic-PyTorch integration surface is not yet swept:

- Tested on a single GPU. No DDP, FSDP, or other distributed testing.
- No gradient accumulation support (`release_matrix_grads` requires exactly
  one backward per `step()`; without it, standard accumulation works but is
  untested here).
- No sparse-gradient support.
- Not tested under `autocast`/`GradScaler`, `torch.compile`'s `capturable`/
  differentiable modes, or exotic tensor subclasses.
- Minimum PyTorch version (`2.1`) is inferred from the API surface used
  (`register_post_accumulate_grad_hook`), not empirically swept across
  versions.

If you hit one of these, open an issue — this list is the honest starting
point, not a promise every box will stay unchecked.

## License

MIT. See [`LICENSE`](LICENSE).
