# Facts

Measured numbers that neither state a rule (`SPEC.md`) nor frame an open question
(`PLAN.md`). Each carries where it came from. A fact here is a measurement at a
date, not a law; when a new read contradicts one, replace it and say so in PLAN.

Telemetry names are the ones logged at the time. `update_to_param_ratio` is
today's `relative_step`, the same quantity; `step_gain/<role>` was
`relative_step/<role>` over `lr`, and was dropped once the law made it
agreement times caution (2026-10-03).

## Anima lane

### What the model is made of

Base `circlestone-labs/Anima-Base-v1.0-Diffusers` transformer, by the roles
`scripts/usuitrack_base_distance.py` assigns (2026-09-14):

| role | tensors | params | base rms | trained by |
|---|---|---:|---:|---|
| attn1 q/k/v/out, attn2 q/out | 2D | 117M each | 0.0099-0.0116 | UsuiTrack |
| attn2 k/v | 2D | 58.7M each | 0.0094 / 0.0146 | UsuiTrack |
| ff_up, ff_down | 2D | 470M each | 0.0111 / 0.0114 | UsuiTrack |
| patch_embed, proj_out | 2D | 0.14M / 0.13M | 0.0158 / 0.0055 | UsuiTrack |
| adaln (`norm1.linear_1`/`linear_2`) | **2D** | 58.7M | 0.0259 | UsuiTrack |
| other (`norm2`/`norm3`/`norm_out` `linear_1`/`linear_2`) | **2D** | 119M | 0.0406 | UsuiTrack |
| other 1D (norm_q/norm_k gains) | 1D | 14k | 0.774 | AdamW fallback |
| time (linears / norm) | 2D / 1D | 16.8M / 2k | 0.0088 / 0.192 | frozen (`ignore_if_contains`) |

`adaln` and `other` are the same kind of module, the AdaLN-LoRA pairs of the
three block norms and the output norm: `linear_1` reads the SiLU'd timestep
embedding, one (B, 2048) vector per sample, and `linear_2` writes the shift,
scale and gate added onto the frozen `temb` base. ~178M parameters, ~9% of the
transformer. Under UsuiTrack since 2026-09-28; before that they were the
fallback's, which then carried 48% of the optimizer state for 9% of the
parameters. The block LayerNorms have no affine weights and the model no biases,
so the fallback now holds only the qk-norm gains.

### Per-step motion, in one unit

Relative update per step (update norm over weight norm), 2026-09-14:

| class | per-step relative motion |
|---|---:|
| UsuiTrack matrices, run 13 plateau (`update_to_param_ratio`) | 3.7e-6 |
| UsuiTrack matrices, run 12 plateau | 2.3e-6 |
| AdamW `adaln` ceiling at `fallback_lr 2e-6` (`lr / rms`) | 7.7e-5 |
| AdamW `other` ceiling at `2e-6` | 4.9e-5 |
| LFM best arm (cosine-zero, `lr 4e-4`, bs16 x 1024 tokens) | 7.6e-5 |

AdamW moves each coordinate by up to `lr` per step regardless of gradient scale,
so its relative motion is `lr / rms`: at `2e-6` the fallback's per-step budget is
~20x the matrices' realized step. A nominal lr ratio between the two optimizers
means nothing without the rms. Anima's matrix step is ~20x smaller than LFM's
best arm at roughly twice the tokens per step (bs4 x k4 x ~2304 latent tokens
at 768 px, assuming 8x VAE and 2x2 patches -- not checked).

### bf16 stochastic rounding walks the weights

All weights on this lane are bf16 and every write is stochastically rounded.
Run 13 checkpoints (plateau ends at step 432), per role, 2026-09-14:

| role | `|w400-w350| / |w350-base|` | cos(increment, net) | coords unchanged 350->400 |
|---|---:|---:|---:|
| every matrix role | 0.374-0.382 | 0.001-0.004 | 88-96% |
| adaln | 0.377 | 0.003 | 67% |
| other | 0.374 | 0.001 | 68% |
| patch_embed / proj_out | 0.437 / 0.404 | -0.062 / -0.016 | 84% / 53% |

A random walk predicts `sqrt(50/350) = 0.378` and cosine 0; coherent travel
predicts ~0.14 and a positive cosine.

Run 12 over a longer horizon (1400 -> 1600, plateau until 1728; k=1, step
`2.3e-6`): every matrix role reads ratio 0.369-0.383 against the walk's 0.378,
cosine 0.000-0.005, 80-93% of coordinates unchanged in 200 steps. For a
straight coherent component carrying fraction `f` of the displacement norm the
cosine is `f^2 * (t2-t1)/t1 / ratio`, so cosine <= 0.002 bounds `f` at ~7%.
The toy's fp32 *incoherent* stream (gradient noise through the moment) reads
cosine 0.045 on its own; the real 0.001 is below even that, so rounding noise
buries the moment's short-range correlation too. Read with
`scripts/usuitrack_walk_read.py` (ai-toolkit).

Toy on a real `attn1.to_q` (2048x2048), rank-78 update stream at relative
`3.7e-6` per step, `beta 0.9`:

| update stream | fp32 `|net|` at 350 | bf16+SR `|net|` | bf16 ratio / cos / unchanged |
|---|---:|---:|---|
| incoherent | 3.0e-4 | 2.13e-3 | 0.380 / 0.001 / 91% |
| 10% coherent | 6.2e-4 | 2.20e-3 | 0.366 / 0.026 / 91% |
| fully coherent | 1.30e-3 | 2.47e-3 | 0.331 / 0.119 / 91% |

The incoherent toy reproduces the real run's magnitude, ratio, cosine and
unchanged fraction. Rounding error is 7x the fp32 travel there, 1.6x even for a
perfectly straight path. A small toy (256x128, rms-1 weights, `lr 0.05`) moves
its weights 4.9x the intended update per step through the same mechanism.

Consequence: every distance-from-base figure on this lane -- the P13 run-10
monotone rise, the P18 fallback tables, the run 12/13 comparison below -- is
dominated by the walk. It remains the right read for "how far did the weights
go"; it is not a read of how far learning went.

### Walk-only control: what the walk does to samples

`anima_walk_only_control`: base weights plus a simulated walk
(`scripts/usuitrack_walk_only_model.py`) matched per role to run 13 @550 (whole
model 2.943e-3 against run 13's 2.944e-3), sampled at 1024 px, 30 steps, seed 42,
run 13's five prompts. Pixel RMS on [0,1] RGB against base (base samples are
bit-identical across runs, checked):

| prompt | walk-only | run 13 @50 | run 13 @550 |
|---|---:|---:|---:|
| 0 whale landscape | 0.018 | 0.078 | 0.128 |
| 1 armored warrior | 0.146 | 0.222 | 0.175 |
| 2 gown | 0.041 | 0.148 | 0.211 |
| 3 creature and bird | 0.036 | 0.138 | 0.181 |
| 4 ethereal figure | 0.018 | 0.050 | 0.116 |

Pixel RMS is crude and composition-sensitive; the images were also looked at
(agent's read, not the user's): the walk shifts composition on prompt 1, is near
identical to base on prompt 2, and carries none of run 13's style shift.

### Distance from base, run 12 @2200 vs run 13 @550

| role | run 12 | run 13 |
|---|---:|---:|
| ALL | 2.314e-3 | 2.944e-3 |
| adaln / other | 0 (frozen) | 4.691e-3 / 3.173e-3 |
| attn1 k / q / v / out | 4.71 / 4.28 / 3.30 / 3.76 e-3 | 2.99 / 2.73 / 2.15 / 2.54 e-3 |
| attn2 k / q / v / out | 4.53 / 3.57 / 3.79 / 2.73 e-3 | 3.31 / 2.41 / 2.60 / 1.96 e-3 |
| ff_up / ff_down | 3.55 / 2.36 e-3 | 2.29 / 1.57 e-3 |
| patch_embed / proj_out | 12.5 / 11.5 e-3 | 7.17 / 6.90 e-3 |

Every matrix role moved less in run 13; the whole-model excess is the fallback
class, frozen in run 12. Both columns are walk-dominated (above).

### Run 12 and run 13 telemetry

Bucket means over each run (steps are optimizer steps):

| read | run 12 (k=1, 2304 steps, lr 5e-5) | run 13 (k=4, 576 steps, lr 1e-4) |
|---|---|---|
| `update_to_param_ratio` plateau | 2.33e-6 | 3.70e-6 |
| `moment_persistence` | +5e-4 -> -3.5e-5 (mid) -> +5.5e-4 | -6.7e-4 (start) -> 1.1-1.7e-3 |
| `grad_moment_cosine` | +4.6e-3 -> -1.8e-4 (mid) -> +1.4e-3 | +3.4e-3 .. +6.6e-3, +1.0e-3 at the end |
| `micro_batch_agreement` | -- | 2.5-3.1e-3 plateau, 1.3e-3 then 3.0e-4 as lr decays to 2e-5 |
| `grad_capture` | 0.606 -> 0.655 | 0.646 -> 0.748 |
| `tangent_live_fraction` | 0.884 -> 0.902 | 0.857 -> 0.906 |
| `tangent_concentration` | 0.85 flat | 0.613 -> 0.573 |

Run 13 changed `k`, lr, rank table (sum 595 vs 256) and `fallback_lr` together.

### Run 14: `k` against the table

`anima_usuitrack_14_k1_contrast` is run 13 with `k=1` and `lr 5e-5`, 576 steps.
Bucket means by optimizer step (runs 12 and 13 at the same step indices):

| read, steps 200-431 | run 12 (k=1, table 256) | run 14 (k=1, table 595) | run 13 (k=4, table 595) |
|---|---:|---:|---:|
| `update_to_param_ratio` | 2.34e-6 | 3.64e-6 | 3.71e-6 |
| `grad_capture` | 0.625 | 0.701 | 0.738 |
| `tangent_live_fraction` | 0.889 | 0.552 | 0.891 |
| `tangent_concentration` | 0.847 | 0.849 | 0.577 |
| `moment_persistence` | 6.3e-4 | 6.1e-4 | 1.16e-3 |
| `grad_moment_cosine` | 2.6e-3 | 2.4e-3 | 4.5e-3 |

Per-role live fraction, steps >= 200, run 14 against run 13: attn1 k/q/v/out
0.552/0.569/0.636/0.493 (run 13 0.890/0.908/0.918/0.871); attn2 k/q/v/out
0.500/0.504/0.467/0.508 (0.818/0.920/0.821/0.812); ff up/down 0.659/0.557
(0.955/0.898); patch_embed/proj_out 0.797/0.840 (0.988/0.999). The k=4/k=1
ratio at table rank is 1.4-1.8 per role, against 2.1-3.6 at calibration rank 256.

### Runs 15 and 16: current code, and patch-equalized loss

Both run 13's config for 300 steps (decay from 225), `k=4`. Run 15 on the
2026-09-14 trainer (token-exact weights, final fold unrounded, fallback divided
before the clip); run 16 adds `loss_type: patch_equalized_mse` and
`timestep_type: linear`. Bucket means by optimizer step:

| read, steps 150-224 | run 13 | run 15 | run 16 |
|---|---:|---:|---:|
| `update_to_param_ratio` | 3.71e-6 | 3.69e-6 | 3.70e-6 |
| `grad_capture` | 0.717 | 0.719 | 0.723 |
| `tangent_live_fraction` | 0.883 | 0.885 | 0.888 |
| `tangent_concentration` | 0.585 | 0.578 | 0.572 |
| `moment_persistence` | 1.33e-3 | 1.71e-3 | 1.81e-3 |
| `grad_moment_cosine` | 5.3e-3 | 7.1e-3 | 6.7e-3 |
| `micro_batch_agreement` | 2.70e-3 | 2.88e-3 | 2.48e-3 |
| `micro_batch_agreement`, steps 0-49 | 1.93e-3 | 1.87e-3 | 3.28e-3 |

Run 15 tracks run 13 on capture, liveness and concentration. Its persistence and
overshoot cosine read higher; no noise floor exists for those reads on this lane,
so that is unresolved, not an effect of the code change. Wall clock 17.4 (run 15)
and 17.7 s/step (run 16), both including sampling.

### Rank calibration at `k=4`, and what the sized table reached

`anima_rankcal_r256_acc4` (rank 256, bs4 x k4, `lr 4e-5`, bf16 accumulators, 9
windows) against run 13's per-role `live_fraction` at its table, steps >= 200:

| role | mean | median | geomean | geomean/mean | table | run 13 frac |
|---|---:|---:|---:|---:|---:|---:|
| attn1_k | 114.4 | 120.1 | 105.6 | 0.92 | 106 | 0.890 |
| attn1_out | 76.5 | 84.3 | 65.7 | 0.86 | 66 | 0.871 |
| attn1_q | 82.4 | 84.0 | 78.4 | 0.95 | 78 | 0.908 |
| attn1_v | 42.2 | 43.9 | 38.5 | 0.91 | 39 | 0.918 |
| attn2_k | 56.4 | 66.1 | 45.1 | 0.80 | 45 | 0.818 |
| attn2_out | 38.7 | 32.9 | 27.6 | 0.71 | 28 | 0.812 |
| attn2_q | 47.7 | 50.5 | 45.2 | 0.95 | 45 | 0.920 |
| attn2_v | 43.8 | 49.1 | 34.2 | 0.78 | 34 | 0.821 |
| ff_down | 53.7 | 63.8 | 45.4 | 0.85 | 45 | 0.898 |
| ff_up | 46.9 | 47.7 | 45.1 | 0.96 | 45 | 0.955 |
| patch_embed / proj_out | shape-capped | | | | 34 / 32 | 0.988 / 0.999 |

`corr(geomean/mean, frac) = 0.92` over the ten uncapped roles. At equal rank 256,
k=4 live counts were 2.1-3.6x the k=1 calibration's (`PLAN.md` history in
`ARCHIVE.md`). Bf16 rounding cannot account for that: stochastically rounding a
2048x256 tangent with a power-law spectrum moves its live count by at most one
plane. No fp32 `k=4` calibration exists to compare -- that attempt OOMed at step
3, before its first window.

### Memory and throughput

| run | shape | s/step (wall, incl. sampling) | s/sample | memory |
|---|---|---:|---:|---|
| run 12 | bs4, k=1, table 256 | 4.67 | 1.17 | -- |
| run 13 | bs4, k=4, table 595 | 17.76 | 1.11 | wandb: 76% of 12.3 GB allocated |
| run 14 | bs4, k=1, table 595 | ~3.2-3.7 (progress bar) | | |
| run 15 | bs4, k=4, table 595, median interval | 14.78 | 0.92 | |
| run 12 | bs4, k=1, table 256, median interval | 3.93 | 0.98 | |
| probe | bs1, k=16, checkpointing, median of 2 intervals | 15.46 | 0.97 | |
| probe | bs1, k=16, no gradient checkpointing | | | OOM (3 retries) |
| rankcal r256 k=4, fp32 accumulators | | | | OOM at step 3 |
| rankcal r256 k=4, bf16 accumulators | | 13.9-15.3 | | needed `expandable_segments` |

Pinned PCIe on this box: 12.2 GB/s host-to-device, 12.6 GB/s device-to-host;
a CPU fp32 add runs ~0.06 s/GB (2026-09-14, measured beside a training run).

Accumulation costs nothing per sample; it costs optimizer steps per hour. The
card is at its limit at bs4 x k4 with the sized table -- an earlier "6.6 GB
peak" note was a different counter and should not be used for headroom.
Sampling: ~23 s per 1024x1024 image at 30 steps.

Telemetry's cost, 2026-10-03 (synthetic gradients, so a cost read only): 40
bf16 matrices at Anima's block shapes, eager, `rank_fraction 0.025`. The
optimizer step takes 55 ms with diagnostics off, 61 ms at `core`, 62 ms at
`full`, and peak transient VRAM is 45.5 MiB at every tier. That is ~0.16 ms per
matrix per step for `core`, almost all of it small kernel launches. Before the
update norm moved to frame coordinates and the non-finite count to the raw norm,
the step took 59 ms off and 70 ms at `core`; both were full-matrix passes.

### A real bs16 on 12 GB: offloaded checkpointing

2026-09-28, torch 2.14, bs16 x 768 (36,864 tokens). Held without checkpointing,
one block saves 5.69 GiB for backward: attn1 1.87 (three fp32 rope'd q/k copies,
288 MiB each), ff 1.33, attn2 1.22, norms 0.87, glue 0.42. Checkpointing leaves
28 block inputs; `activation_offloading` (ai-toolkit) keeps those in pinned host
memory under `save_on_cpu`, which does reach a non-reentrant checkpoint, and
checkpoints attn1 and attn2 once more inside the block, since the recompute of
one block is what then sets the peak.

| arm (every block compiled) | peak over weights | s/sample |
|---|---:|---:|
| plain bs4 | 2.02 GiB | 0.621 |
| offload bs16, no nesting | 4.23 GiB | 0.650 |
| offload bs16, attn1+attn2 nested | 2.58 GiB | 0.718 |

The whole-model compile of run 17 (0.616 s/sample at bs4) is not an option at
bs16: it graph-breaks at the saved-tensor hooks, and it holds every gradient
until its one backward graph ends, where per-block compile lets
`release_matrix_grads` free a block's gradients as its backward finishes. In
the trainer the unnested arm fits step 1 and OOMs from step 2, once the
optimizer state is resident; nested runs at ~12.4 s/step. At rank 0.05 on the
2026-10-31 tree both paths fit unnested ("Molegrad on Anima"). Gradients agree with
eager within the compile noise floor (relative grad-norm difference median
~1e-2, as for plain compile against eager).

After a step, with AdaLN still on the fallback: 5.48 GiB allocated, 8.32
reserved, 7.88 step peak. Parameters 3.89, UsuiTrack state ~0.76, the fallback's
AdamW moments for the AdaLN pairs ~0.68 -- 48% of optimizer state for 9% of the
parameters, gone since they moved to UsuiTrack (`optimizer.pt` 1.41 -> 0.97
GiB). `nvidia-smi` reads reserved allocator cache, not use: 11.4 GB at 20
steps, and the card is still near full -- a 570 MiB request failed once and
succeeded on the allocator's retry.

### Molegrad on Anima: 9% faster, the nesting no longer earns its time

2026-10-31, bs16 x 768, `rank_fraction 0.05`, offloaded checkpoints, per-block
compile, seeded pairs on one tree (usuitrack `projected-grads` = `lab` plus
molegrad; ai-toolkit `usuitrack-molegrad` on upstream `e9c72b7`). s/step over
shape-matched steady steps, peak allocated per step:

| arm | nested attn1/attn2 | s/step | peak GiB |
|---|---|---:|---:|
| `G`, `release_matrix_grads` | yes | 12.05 | 7.58 |
| molegrad | yes | 10.93 | 8.09 |
| `G`, `release_matrix_grads` | no | 11.0 | 9.24 |
| molegrad | no | 9.97 | 9.29 |

Unnested, molegrad costs 0.05 GiB over the release path: its probe sketches.
Both carry the projections from backward into the step, 0.61 GiB (`Z`,
tangent) against 0.67 (`Z`, aim, probes). Nested, molegrad costs 0.5 GiB more,
and that is not `G`: a CPU eager count of weight-shaped matmuls in a steady
backward under the trainer's checkpoints finds 452 for `G` and 0 for molegrad,
nested or flat. Which interaction of the custom Function with the nested
checkpoint holds it is not located. The nesting costs ~1 s/step on both paths.

The peak is the end of backward, inside block 0's recompute: resident memory
grows 0.66 GiB through backward (24 MB per block of held projections), and each
block's recompute and backward adds ~4.04 GiB on top (9.29 at block 0, 8.6 at
block 27); a block's forward peaks at 8.52. The offloaded checkpoint inputs
never return to the card together.

Whole-model compile does not reach Anima: its forward calls
`trainable_model.transformer`, not the `sd.unet` the trainer compiles, so the
arm runs eager (16.4 s/step, 8.65 GiB). Compiled in place, the graph breaks at
the offload's saved-tensor hooks and runs at the same speed; without offload at
bs8 x 2 it compiles, but a recompile took 22 minutes. Molegrad also compiles
slower per new bucket shape than `G` (44 s against 12 s). One 576 MiB request
per run fails and succeeds on the allocator's retry, as above.

### Accumulation beats offloading on Anima under molegrad

2026-10-31, bs16 per step, molegrad at rank 0.05, unnested, block compile,
eager weight write, 20 seeded steps each (median of steady steps):

| arm | s/step | peak GiB | reserved GiB |
|---|---:|---:|---:|
| bs16, offloaded | 10.78 | 9.29 | 9.97 |
| bs8 x 2, no offload | 9.88 | 9.12 | 9.57 |
| bs4 x 4, no offload | 10.40 | 7.16 | 7.52 |

Offloading pins and copies every checkpoint input across PCIe each step, and
that costs more than splitting the batch. Without it, bs8's inputs stay on the
card, ~2.1 GiB over 28 blocks, while the per-block recompute transient that
sets the peak (~4 GiB at bs16) halves: the two cancel. bs4 x 4 leaves 2.8 GiB
free at the batch Anima's buckets were sized for. The same arm with the
compiled write ran at 9.97-10.0 s/step (run 29, the probe above): eager costs
Anima ~0.8 s/step.

### `relative_step` under `lr` on Anima is AdaLN's empty planes

`unit_polar` scales the polar factor by `1/sqrt(r)`, as if the moment filled
every plane; five Newton-Schulz iterations leave a plane far below the top near
zero, so a moment that fills `k` of `r` planes steps at `sqrt(k/r)` of `lr`.
Polar norms of saved moments, by role (`optimizer.pt`, 2026-10-31):

| role | run 28, rank 0.025, step 960 | run 29, rank 0.05, step 192 |
|---|---:|---:|
| attention, FF, stream ends | 0.996-1.001 | 0.995 |
| AdaLN `linear_1` (85) | 0.690 (r 18) | 0.589 (r 36) |
| AdaLN `linear_2` (84) | 0.887 (r 31) | 0.695 (r 63) |
| fleet, predicted / read | 0.896 / 0.883-0.890 | 0.829 / 0.827 |

AdaLN's moment fills about 9-12 planes of `linear_1` and 24-30 of `linear_2`,
the low rank a batch of timestep embeddings gives ("Where the step lands, per role"), so a
higher rank only adds empty planes. AdaLN holds 52% of `sum ||W||^2` (its RMS is
2.5-4x the bulk's), so it sets the fleet read: per-role norms weighted by each
role's share of `sum ||W||^2` (run 29's checkpoint) predict the read to 0.2%.
Run 28's 12% gap is this. The `relative_step` read says it equals `lr` whenever
every matrix stepped; that holds only where every moment fills its rank.

### Run 20 against run 18: what the fleet reads say about lr

Run 20 (bs8, `beta 0.98`, caution, `lr 9e-4`) took a relative step of 2.0e-5
per step, 1.67x run 18's 1.2e-5 at half the batch: its lr was derived from
sensor run 19's step, never sample-judged, rather than from run 18's, which
was. Fleet reads, run 20 against run 18:

| read | run 18 | run 20, steps 200-1700 | run 20, last 50 steps (lr ~0.25x) |
|---|---:|---:|---:|
| `grad_moment_cosine` | +0.001 to +0.006 | -0.001 to -0.006 | +0.0027 |
| `moment_persistence` | 0.001 | 0.000 | 0.0003 |
| `tangent_concentration` | 0.881-0.888 | 0.892-0.907 | 0.895 |
| `projected_grad_norm`, mean / max | 0.012-0.014 / 0.023 | 0.012-0.020 / 0.078 | 0.0125 / 0.020 |
| `caution_kept` | | 0.533-0.535 | 0.534 |

The higher grad norm is mostly the batch: half the batch predicts sqrt(2) x
0.0125 = 0.0177. The negative cosine lean is the lr read -- the moment carrying
the weights past where the next gradient wants them -- and it turned positive
only in the anneal's last bucket, one bucket against a scatter of ~0.002. The
user's read mid-run: lower loss, degraded samples, not broken. Run
`anima_usuitrack_20_b098_caution_bs8` and `anima_usuitrack_18_bs16_rf010`,
`loss_log.db`, 2026-10-01.

### Runs 24 and 25: agreement inside the step against outside

bs16 (offloaded), `beta 0.95`, caution, `rank_fraction 0.025`, one epoch (192
steps), warmup 20, cosine decay from ~150. Run 24 takes the law's step (agreement
inside, `lr 7e-5`); run 25 renormalizes the masked moment to unit norm before
the lift (`scripts/usuitrack_outside_runner.py` in ai-toolkit), at `lr 6.84e-6`,
run 24's plateau step. Fleet `relative_step` matched to 1.3% through warmup,
plateau and decay. Means over 30-step bins:

| steps | cosine, 24 | cosine, 25 | `norm1.linear_1`, 24 / 25 | `norm3.linear_1`, 24 / 25 |
|---|---:|---:|---:|---:|
| 30-59 | +0.016 | +0.0095 | +0.082 / +0.022 | +0.066 / +0.003 |
| 60-89 | +0.0044 | -0.0005 | +0.012 / -0.007 | +0.001 / -0.013 |
| 90-119 | +0.0037 | -0.0018 | +0.013 / -0.013 | +0.003 / -0.013 |
| 120-149 | -0.0017 | +0.0027 | -0.007 / -0.003 | -0.012 / -0.001 |
| 180-191 (lr 1.2e-5) | +0.011 | +0.006 | +0.062 / +0.006 | +0.043 / +0.012 |

- At the same fleet step, outside crossed zero at ~60 and inside at ~120, and
  outside led in every role group, attention and FF included. Spread evenly,
  the step over-drives the modules whose agreement is at noise.
- Under the decay, inside's AdaLN `linear_1` recovered its cosine; outside's
  stayed near zero. Inside keeps the coherent modules coherent.
- Outside turned positive again at 120-149 at full lr while inside dipped; the
  data order is shared, so it is each run's own cycle, unexplained.
- Early (20-59) outside read *lower* AdaLN cosine though it gives AdaLN less
  step than inside. A hypothesis, unmeasured: AdaLN's gradient depends on every
  block below it, and outside moves the noisy blocks more. Testing it needs
  per-role `relative_step`.
- `grad_capture` 0.77 to 0.81 in both, with no trough: at this step the frame
  keeps up. `transport_speed` flat at 0.028.
- In run 24 a decay of lr from `7e-5` to `1.2e-5` took fleet cosine from
  -0.002 to +0.011 on fresh data: on Anima too, cutting lr restores cosine.
- `caution_kept` reads 0.556 in both, against 0.601 for an EMA of independent
  directions at `beta 0.95`; run 20 (`beta 0.98`) read 0.534 against 0.564.
  LFM sits 0.01-0.02 under the same formula at every beta (0.624, 0.588,
  0.555, 0.539 at 0.9-0.99). Anima's batch disagrees with the moment's history
  more than independence would: anti-correlated steps.

Samples, the user's: 25 moved less than 24; read at the end, different rather
than worse, inside still favoured. `loss_log.db`, 2026-10-03.

### Run 26: the fleet vote sits at zero from step ~100

Run 24's setup (bs16, `beta 0.95`, `rank_fraction 0.025`) with the polar mask
and the lr controller's second form (branch `cosine-controller`, `ARCHIVE.md`),
ceiling `1.4e-4`, no warmup, 960 steps = 5 epochs. Means over 48-step bins
(a quarter epoch):

| steps | vote | `lr_scale` (min) | `relative_step` |
|---|---:|---:|---:|
| 0-47 | +0.059 | 1.00 (0.97) | 1.28e-5 |
| 96-143 | -0.017 | 0.84 (0.70) | 1.25e-5 |
| 288-335 | -0.028 | 0.77 (0.64) | 1.08e-5 |
| 624-671 | -0.056 | 0.58 (0.47) | 8.1e-6 |
| 912-959 | -0.026 | 0.78 (0.64) | 1.07e-5 |

- The vote is positive only in the first quarter epoch, then sits between
  -0.01 and -0.06 for the rest of the run. After the first bin, fleet
  `grad_moment_cosine` stays within ±0.0022 of zero in every bin: after ~100
  steps the batch neither confirms nor opposes the moment, by much, where
  run 24's cosine crossed zero too.
- Per role, only `proj_out` (+0.009 to +0.019), `norm_out.linear_1` from
  step ~200 (+0.015 to +0.032) and early `norm3.linear_1` (+0.032) hold a
  clearly positive cosine; `patch_embed` stays negative (-0.006 to -0.019);
  attention and ff sit within ±0.004. `moment_persistence` orders the roles
  the same way, at 0.006 or below.
  Not normalised by rank, so not a per-role signal read yet (PLAN E9).
- `caution_kept` 0.592 against its floor of 0.601, under runs 24-25's 0.556:
  the polar mask moved it toward the floor, as on LFM.
- Fleet step averaged ~1.1e-5 for 960 steps, against run 24's 6.84e-6 for 192.

Samples, the user's: very stable, barely changed from base; general quality
arguably above base. Judged over the 5 epochs, it moved more than runs 24-25.
All runs share one seed and prompt set, which is good for comparing runs and
poor for judging fresh. `loss_log.db`, 2026-10-04.

## LFM2.5-350M lane

bs16 x seq1024, seed 1, 1k steps, broad-no-embeddings. Noise: ~1e-3 target,
2-6e-3 source at 1k ("Run-to-run noise"; withdraws the `3e-4` / `2e-3`
floors this line stated).

| arm | target | source | `update_to_param_ratio` |
|---|---:|---:|---:|
| global r128, `2e-4` | 1.669330 | 3.084276 | |
| per-role table (12,032 planes), `2e-4` | 1.667302 | 3.079098 | 1.66e-4 |
| calibrated table (11,886 planes) | 1.667920 | ~= | |
| `side=right`, r128 | 1.677500 | 3.089930 | |
| `beta 0.9`, `lr 4e-4` (cosine-zero) | **1.665708** | **3.040611** | 7.6e-5 |
| `beta 0.9`, `lr 8.7e-4` | 1.669194 | 3.086723 | 1.65e-4 |
| `beta 0.5`, `lr 1.6e-4` | 1.690248 | 3.013536 | 7.7e-5 |
| `beta 0.99`, `lr 1.29e-3` | 1.681707 | 3.138621 | 7.5e-5 |
| `beta 0`, `lr 2e-4` | 1.685989 | 3.020878 | 1.66e-4 |

`moment_persistence` within `1e-3` of zero in every arm. With no moment
(`beta 0`) target loss is far worse: the moment's temporal cancellation is doing
real work there.

### The turn matters while the aim moves, and not once it has settled

The frame is in effect an average of its aims over the last `~1/eta` updates, so
the turn it can take is set by two things only: how far its aim can be trusted
and how fast the model moves the aim. Early, the aim moves fast and lag
dominates: on LFM (bs16, r128, `lr 2e-4`, `beta 0.9`, seed 1) capture and target
loss improve steadily from `0.005` to `0.04` and are flat to `0.08` (target at
300 steps 1.7468 at `0.005`, 1.7448 at `0.01`, 1.7426 at `0.04` and `0.08`).
The harmonic start `max(0.01, 1/t)` lands between `0.02` and `0.04`: its old win
was the size of the turn, not the decay.

Once the aim settles, the turn stops mattering. Dropping `0.04` to `0.01` at
step 200 or 500 never beats constant `0.04` at 1k (1.68330 and 1.68301 against
1.68288; `0.01` throughout 1.68391), and capture does not rise after the drop.
The switch at 200 keeps ~60% of `0.04`'s lead, all of it banked before the
switch. A settled frame orbits -- net travel over 10 updates is ~30% of its path
at both `eta` -- and the energy aim is steady enough that the wider orbit at
`0.04` costs no measurable capture. Every late gap is at or below the `3e-4`
floor; one seed per arm.

On Anima (run 17 against run 15, `k=4`) the same shape: capture +0.057 and live
fraction +0.016 over the first 100 steps, gaps nearly gone by 200, and neither
loss nor samples separate the runs at 300. `--seed` seeds only the rounding
draws; the LFM seed pair differs by `1.6e-4`. Runs `e3_*` (lab wandb) and
`anima_usuitrack_17_eta040_300`, 2026-09-28.

### The energy aim is the right target; the write roles lack rank

At fixed weights, 1024 bs1 samples, 27 matrices: the mean gradient's subspace
(`M`, pair-estimated) against the top-`r` eigenspace of what the aim targets at
batch `B`, each scored on how much of the *held-out* half's mean-gradient
energy it captures.

- **The signal collapses as the model trains.** At base the mean gradient is
  3.5-10% of a sample's energy; after 300 steps it is 0.1-0.4%, and its
  subspace no longer replicates between two halves of 512 samples (overlap
  ~0.2-0.3).
- **So no signal-only target is estimable, and the energy aim beats one.** The
  mean gradient lives inside the directions where per-sample energy lives, so
  the Cov-dominated aim is a low-variance estimate of it. At the trained state
  the energy frame captures more held-out signal than the estimated signal
  frame in eight of nine roles, and the gap widens with rank, because the signal
  frame saturates at its estimation noise (0.75-0.93) while the energy frame
  keeps climbing.
- **Rank splits by what a role does to the residual stream.** Roles that read it
  (conv.in_proj, w1, w3, q, k, v) saturate by rank 64-128 at 0.94-0.99 capture.
  Roles that write it (conv.out_proj, w2, out_proj) keep climbing through 256:
  0.71 -> 0.89, 0.77 -> 0.86, 0.75 -> 0.89 from table rank to 256. The
  liveness-sized table gives exactly these roles the smallest ranks (20, 110,
  42). Liveness reads the steepness of the tangent spectrum, not the breadth of
  the signal.
- **A real batch aims better than accumulated micro-batches, in the write roles.**
  The per-micro-batch aim's target keeps the micro-batch's noise weight
  (`Cov/4` at bs4); one bs16 batch's target has `Cov/16`. At table rank that is
  0.71 against 0.77 on w2, and within a point on the read roles.

Caveats that remain: these score ideal top-`r` eigenspaces of the targets, not
the tracked `Q`; the trained state is one 300-step run at `eta 0.01`.
Scratchpad `subspace_read.py`, 2026-09-28.

### Rank is a fraction of the matrix's size

The calibrated LFM table, read as `r / sqrt(mn)`, is a shape rule for every role
that reads the residual stream: w1 .097, w3 .090, conv.in_proj .104, q .086,
k .083, v .105. The roles that write it sit at a half to a fifth of that
(w2 .051, out_proj .041, conv.out_proj .020) -- the roles liveness starves and
capture says want more. A uniform rank loses to the table because rank is also
the step's distribution: the update's norm carries `sqrt(r)`, so the table gives
the MLP read roles a larger share of the step and a uniform rank takes it away.
`sqrt(mn)` gives it back with one constant and no calibration run.

bs16, 500 steps, `eta 0.04`, `lr 2e-4`, seed 1:

| arm | planes | target | source | live | update/param |
|---|---:|---:|---:|---:|---:|
| calibrated table | 11,886 | 1.70995 | 3.00586 | .944 | 3.8e-5 |
| uniform r128 | 11,776 | 1.71370 | 3.00079 | .802 | 3.3e-5 |
| `0.078 sqrt(mn)` | 11,924 | 1.71037 | 3.00380 | .882 | 3.6e-5 |
| `0.095 sqrt(mn)` | ~14,500 | **1.70593** | **3.00358** | .822 | 4.0e-5 |

At equal planes the rule ties the table (target +4e-4, source -2.1e-3, both at
their floors); at `0.095` it wins both heads, which a larger step alone does not
do. At 1k (`0.1 sqrt(mn)` against `e3_eta040_1k_s1`, r128) it trades instead:
target 1.67426 against 1.68288, source 3.02628 against 3.02206, at 24% more
update per step -- the rank-step coupling makes that comparison partly an lr
comparison (`ARCHIVE.md`, "the step is relative to the weight").

**Liveness is not a sizing target.** It ranks these arms backwards: the best arm
has the second-lowest live fraction. A real batch raises it at any rank (bs4 to
bs16 at `lr` scaled by `sqrt` tokens: r128 .727 to .802, table .806 to .944),
but the table's target lead over r128 is the same at both batches (-3.2e-3 at
bs4, -3.7e-3 at bs16), so the batch that sets liveness does not set the table's
edge. `tangent_live_fraction` stays as a read of how much of the turn lands on
planes that carry signal. Runs `grid_*`, `shape_c0*`, `shape_c100_1k_s1` (lab
wandb), 2026-09-28.

### Where the step lands, per role

The relative step `lr x step_gain/<role>` (`SPEC.md`, telemetry; logged then as
`relative_step`) on the standard arm at 1k (`rank_fraction` 0.1, `lr 2e-4`, last
logging window):

| role | relative step |
|---|---:|
| `feed_forward.w3` | 5.6e-5 |
| `self_attn.k_proj` | 5.4e-5 |
| `self_attn.q_proj` | 4.3e-5 |
| `feed_forward.w1` | 4.2e-5 |
| `self_attn.v_proj` | 3.8e-5 |
| `self_attn.out_proj` | 3.4e-5 |
| `conv.in_proj` | 3.3e-5 |
| `conv.out_proj` | 2.9e-5 |
| `feed_forward.w2` | 2.7e-5 |

Fleet `update_to_param_ratio` `4.1e-5`. One `lr` spreads 2.1x across roles, and
the three write projections into the residual stream take the three smallest
relative steps. Whether that is the aspect factor, rank, or weight norm is not
separated. The same run is a third noise point for this arm: target 1.674156 against
1.674075 and 1.674264, source 3.024513 against 3.025343 and 3.026280 -- the
instrument does not move the trajectory, and accumulation's removal
(`lfm_1k_bs16_rf010_noaccum`) sits inside the spread. Runs `lfm_1k_d5_relstep`,
`lfm_1k_bs16_rf010_noaccum`, `shape_c100_1k_s1`, 2026-09-29.

On Anima (sensor run 19: run 18 at `lr 2e-4`, 150 steps, sampling off; steps
40-110), against the full-lift step `lr aspect sqrt(r) / ||W||` computed from
shapes and base weights:

| role | relative step | measured / full lift |
|---|---:|---:|
| `patch_embed.proj` | 2.35e-4 | 0.22 |
| `proj_out` | 1.64e-4 | 0.29 |
| `attn2.to_k` | 5.5e-5 | 0.22 |
| `ff.net.0` | 4.1e-5 | 0.23 |
| q, k, v of attn1 and attn2 | 2.8-3.2e-5 | 0.20-0.23 |
| `to_out` of attn1 and attn2 | 2.4-2.8e-5 | 0.20-0.23 |
| `ff.net.2` | 2.0e-5 | 0.23 |
| AdaLN `linear_2` (norm1-3, norm_out) | 1.7-2.4e-5 | 0.06-0.07 |
| AdaLN `linear_1` | 1.5-5.2e-6 | 0.05 |

Every role outside AdaLN sits at `w = sqrt((1-beta)/(1+beta)) = 0.229`
(`beta 0.9`): the norm of an average of uncorrelated unit directions, with
`moment_persistence` at 0.0012. The step of each role is therefore fixed at init
as `lr w aspect sqrt(r) / ||W||`. The 160x spread is shape and weight norm alone,
and the two ends of the stream take 5-10x any block role. AdaLN sits below the
floor: its moment populates about 12 of `linear_2`'s 125 planes and 3 of
`linear_1`'s 72, near the rank-16 gradient a bs16 batch of timestep embeddings
can give, so its rank above that does not reach the step. A `beta` change moves
every non-AdaLN step by `w`: x0.70 at 0.95, x0.44 at 0.98, x0.31 at 0.99, while
the moment stays white.

Tangent liveness is a different read: `0.11` says the aim has magnitude on one
plane in nine, not that the moment is low-rank. `transport_speed` follows it as
`eta sqrt(live)`: 0.038 predicted and 0.037 measured on run 17, 0.013 and 0.014
on run 18. Run `anima_usuitrack_19_sensor_lr2e4`, `loss_log.db`, 2026-09-29.

Run 20 (`beta 0.98`, caution, bs8, `rank_fraction 0.025`, `lr 9e-4`; steps
300-560) confirms the read at the new defaults and adds caution's share.
`step_gain` per role against the white lift `w aspect sqrt(r) / ||W||`
(`w = 0.100`), and its parts against one `attn1.to_q` (per-matrix `||W||` 20.6):

| role | aspect | `sqrt(r)` | `1/||W||` | measured / white | `caution_kept` |
|---|---:|---:|---:|---:|---:|
| `patch_embed.proj` (2048x68, r 9) | 5.5x | 0.42x | 3.5x | 0.63 | 0.584 |
| `proj_out` (64x2048, r 9) | 1x | 0.42x | 10.3x | 0.81 | 0.549 |
| attention and ff, 14 roles | 1-2x | | | 0.72-0.77 | 0.53-0.55 |
| AdaLN `linear_2` | | | | 0.31-0.32 | 0.516 |
| AdaLN `linear_1` | | | | 0.28-0.31 | 0.52-0.53 |

Per unit lr the two ends of the stream take 6.5x and 4.5x the bulk's relative
step, and the weight norm, not the rank, is what puts them there: a 0.14M-
parameter matrix has a small Frobenius norm, so a fixed-norm update is a large
fraction of it (per element 12x the bulk's update RMS on `patch_embed`). Every
non-AdaLN role sits at caution's 0.77 of white, LFM's own factor: the moment is
a white average on this lane at `0.98` as at `0.9`, so `beta` bought no
coherence and `lr w` only sets the size of the noise walk. The two instruments
disagree on the ends: run 13's checkpoint increments were anti-correlated with
net travel only there (cosine -0.062 and -0.016 against 0.001-0.004 for every
other role, "bf16 stochastic rounding walks the weights" above), yet
`caution_kept` puts `patch_embed` at the highest agreement of any role. Run
`anima_usuitrack_20_b098_caution_bs8`, `loss_log.db`, base weights, 2026-10-01.

### Beta buys target at a held step and pays in source

Standard arm at 1k (bs16, `rank_fraction` 0.1, seed 1, constant lr after 50
warmup steps); `update/param` is the fleet `update_to_param_ratio`:

| arm | update/param | target | source | cos | grad norm |
|---|---:|---:|---:|---:|---:|
| `beta 0.9`, `2e-4` | 4.1e-5 | 1.67416 | 3.0245 | +1.9e-3 | 6.49 |
| `beta 0.9`, `4e-4` | 8.1e-5 | 1.66161 | 3.0494 | -2.5e-5 | 6.36 |
| `beta 0.98`, `5e-4` | 4.4e-5 | **1.66081** | 3.0723 | +8.0e-4 | 6.17 |
| `beta 0.98`, `7e-4` | 6.2e-5 | 1.66459 | 3.0914 | +3.3e-4 | 6.01 |
| `beta 0.995`, `1e-3` | 4.4e-5 | 1.67819 | 3.1337 | +7.5e-4 | 5.85 |
| `beta 0.98` re-polared, `5e-5` | 4.4e-5 | 1.66172 | 3.0579 | +1.1e-3 | 6.21 |
| `beta 0.98` re-polared, `1e-4` | 8.8e-5 | 1.67503 | 3.1068 | +5e-5 | 5.87 |
| `beta 0.98` cautious, `5e-4` | 6.2e-5 | **1.65734** | 3.0726 | -1.2e-3 | 6.15 |

- The moment stays white at every `beta`: `step_gain` moves by `w` on all nine
  roles (0.43-0.44x at 0.98 against 0.437 predicted, 0.50x at 0.995 against
  0.501). What `beta` buys is the signal's share of a noise-dominated norm,
  which `step_gain` cannot see: at a held step 0.98 gains 0.013 target.
- Source worsens monotonically with `beta` at a held step (3.025, 3.072,
  3.134): forgetting grows with memory length, not travel. `beta 0.9` at twice
  the step reaches 0.98's target with 0.023 less source damage.
- More lr than the held step does not pay at 0.98, and the cosine does not
  find that optimum: it stays positive past the loss's best lr, since the
  moment holds only `1 - beta` of the current direction.
- 0.995 lags 0.98 by 0.041 at step 250, closing to 0.017 by 1000: part of it is
  the slow fill of a zero-initialised moment, the source cost is not.
- Gradient norm falls with `beta` and with lr, independent of loss: the
  implicit flatness pull of a long lever arm (heavy-ball's `lr/(1-beta)`
  penalty on `||grad||^2`), a hypothesis for polar directions.
- Re-polaring the moment before the lift (`REORTHOGONALIZE_MOMENT`, state
  untouched) trades 9e-4 target for 0.014 source at the held step, and fails
  at twice the step: a flat spectrum hands noise planes full length.
- Caution in the frame (`CAUTIOUS`) keeps 56% of the coordinates and its
  `numel/kept` rescale lifts the step 1.39x, which lands it on plain 0.98 at
  `7e-4`'s exact step: against that it is 0.0073 better on target and 0.019 on
  source, and against plain `5e-4` 0.0035 better on target at equal source.
  Masking what the batch contradicts is what lets the larger step pay. Target
  quantiles p10/p50/p90 1.322/1.650/1.927.

Frame reads (capture, live fraction 0.81-0.83, `transport_speed` 0.0386) do not
move with `beta` or lr. Runs `d4_*_1k_s1`, `lfm_1k_d5_relstep`, 2026-09-29.

### Caution: where it pays, and what its rescale was

Same arm (bs16, 1k, seed 1), caution on, 2026-10-01. Rescaled arms carry the
removed `numel / kept` factor; the release ships without it.

| arm | update/param | kept | target | source |
|---|---:|---:|---:|---:|
| `beta 0.9`, `2e-4`, rescaled | 5.5e-5 | 0.624 | 1.67111 | 3.0292 |
| `beta 0.95`, `2.9e-4`, rescaled | 5.7e-5 | 0.588 | 1.66352 | 3.0439 |
| `beta 0.98`, `5e-4`, rescaled (replicate) | 6.2e-5 | 0.555 | 1.65896 | 3.0782 |
| `beta 0.99`, `6.5e-4`, rescaled | 5.7e-5 | 0.539 | 1.66063 | 3.0965 |
| `beta 0.98`, `1e-3`, no rescale | 6.8e-5 | 0.555 | 1.65901 | 3.0823 |
| `beta 0.98`, `7e-4`, **no rescale** | 4.8e-5 | 0.555 | 1.65942 | **3.0657** |

lr per `beta` is the held step, `2e-4 x 0.229 / w(beta)`.

- `0.98` is caution's target optimum; source worsens monotonically with
  `beta`, as without caution. The kept fraction falls with `beta` (0.62 to
  0.54): a longer memory disagrees with the current batch more often.
- Caution's gain grows with `beta` at a matched step. At 0.9 it sits on the
  line between plain `2e-4` and `4e-4` (interpolated, not measured); at 0.98
  plain at the same step (`7e-4`) is 0.006-0.007 worse on target.
- The rescale was an lr multiplier and nothing else on LFM. Without it
  `step_gain` is 0.552-0.556 of the rescaled value in all nine roles, i.e. the
  kept fraction, uniform across roles; the masked step is 0.77 of the unmasked
  norm. Without rescale `1e-3` repeats rescaled `5e-4`, and `7e-4` holds the
  same target at 0.013-0.017 better source.
- **Run-to-run spread is wider than the floor this file states.** The
  rescaled `5e-4` arm repeated at identical config and seed reads 1.65896 /
  3.0782 against 1.65734 / 3.0726 on 2026-09-29: 1.6e-3 on target, 5.6e-3 on
  source. One pair, not a floor; differences under ~2e-3 target are not claims.

### Batch is not interchangeable with beta

Equal tokens (bs16 x 1k against bs8 x 2k), `beta 0.98`, caution rescaled,
2026-10-01:

| arm | update/param | target | source |
|---|---:|---:|---:|
| bs16, `5e-4` | 6.2e-5 | 1.65896 | 3.0782 |
| bs8, `3.5e-4` (`sqrt(tokens)` rule) | 4.3e-5 | 1.67088 | 3.0473 |
| bs8, `5e-4` (same lr) | 6.2e-5 | 1.67692 | 3.0692 |
| bs8, `beta 0.99`, `4.6e-4` | 4.0e-5 | 1.67376 | 3.0635 |

- bs8 loses 0.012 target at equal tokens. More lr makes it worse on both
  losses, so the `sqrt(tokens)` rule held and bs8 was not under-stepped.
- Raising `beta` to keep bs16's sample memory makes bs8 worse again. A batch
  averages at one parameter point; the moment averages over points the
  parameters have left, and pays that staleness.
- Gradient norm reads ~8.5 at bs8 against ~6.2 at bs16 in every arm: batch
  noise, not lr.

Runs `b9/b95/b99-caut-held-1k`, `b098-caut-bs16-1k`, `b098-caut-bs8-2k*`,
`b099-caut-bs8-2k`, `b098-caut-norescale-*`, 2026-10-01.

### The step relative to the weight: loss-neutral where agreement is uniform

The trust-ratio law (`SPEC.md` step 8) against the aspect rule, 1k, bs16,
`beta 0.98`, caution:

| arm | lr | fleet step | target | source |
|---|---:|---:|---:|---:|
| aspect rule | `7e-4` | 4.8e-5 | 1.659416 | 3.065650 |
| law, agreement inside | `6.2e-4` | 4.8e-5 | 1.659546 | 3.065264 |
| law, inside, 1.4x | `8.7e-4` | 6.8e-5 | 1.659409 | 3.080202 |
| law, agreement outside (unit step) | `4.8e-5` | 4.8e-5 | 1.660761 | 3.060560 |

At matched step the law moves neither head beyond the replicate spread, and it
flattens the per-role step from 2.1x (0.046-0.095) to 1.02x (0.077-0.078). The
lr that matches the aspect rule's step is `step / (agreement x caution)`, here
`4.8e-5 / 0.077`, exactly. Above it the familiar `0.98` trade returns: the same
target, source 0.015 worse. Outside and inside cannot separate here: every LFM
role carries the same agreement (`caution_kept` 0.550-0.557), so outside is
inside at another lr; they can differ only where agreement varies by role, which
on Anima is AdaLN. Runs `d5-inside-6.2e-4`, `d5-inside-8.7e-4`,
`d5-outside-4.8e-5`, 2026-10-01.

### The lr bracket under the law, and what a doubled batch wants

1k, bs16, `beta 0.98`, caution, law inside, seed 1; the bs32 arms run 500 steps
for the same tokens. Fleet step is the plateau `relative_step`; cosine and
persistence are the last logged interval. 2026-10-03.

| arm | lr | fleet step | target | source | cosine | persistence |
|---|---:|---:|---:|---:|---:|---:|
| bs16 | `6.2e-5` | 5e-6 | 1.740939 | 2.999729 | +0.019 | +0.0021 |
| bs16 | `2e-4` | 1.6e-5 | 1.687609 | 3.027565 | +0.0048 | +0.0005 |
| bs16 | `6.2e-4` | 4.8e-5 | 1.659546 | 3.065264 | -0.0008 | -0.0001 |
| bs32, `beta 0.98` | `8.8e-4` | 7.0e-5 | 1.675968 | 3.047561 | +0.0006 | +0.00004 |
| bs32, `beta 0.95` | `8.8e-4` | 1.15e-4 | 1.674347 | 3.032164 | -0.0037 | -0.0006 |

- At bs16, lr trades target for source monotonically across 10x: the low arm
  learns, less, and forgets least. Cosine crosses
  zero between `2e-4` and `6.2e-4`; persistence crosses with it at about a tenth
  of the magnitude, so it is the noisier read of the same thing. `caution_kept`
  does not move with lr (0.555 in all three).
- Cosine falls through every run, even the low arm's (0.265 early, 0.126
  mid-run, 0.021 at the end): part of the decline is the data being used up, not
  over-travel. Decaying lr beat constant on target (`ARCHIVE.md`, "the cosine
  lr controller").
- `grad_capture` dips after warmup and recovers slowly; the trough deepens with
  lr (0.772 at `6.2e-5`, 0.760 at `6.2e-4`, steps 75-125). The frame lags the
  step it is asked to follow. The fleet step peaks at step 75 in both, where
  warmup has finished and the moment is still coherent.
- bs32 at the `sqrt(tokens)` lr under-steps at `beta 0.98`: 0.016 worse target
  than bs16 at equal tokens, 0.018 better source, cosine still positive.
- At the same lr, `beta 0.95` steps 1.64x further, ties 0.98 on target
  (-0.0016) and forgets 0.015 less. A shorter memory is less stale, and
  staleness costs step size times memory length, so the larger the step, the
  more a short memory pays. Its cosine ended negative without a loss cost: on
  this read, zero is not shown to be the loss-optimal set point.
- At a held small step on bs16 (above, "Caution: where it pays", pre-law,
  rescaled) 0.95 lost 0.004 target to 0.98: the optimal beta falls as the step grows.

Runs `law-lr6.2e-5`, `law-lr2e-4-cos0`, `d5-inside-6.2e-4`,
`law-bs32-b098-lr8.8e-4`, `law-bs32-b095-lr8.8e-4`.

### `eta 0.02` against `0.04`, after the liveness fix

Standard arm (bs16, 1k, `beta 0.98`, `6.2e-4`, law inside), only
`GEODESIC_STEPSIZE` changed. 2026-10-03.

| eta | target | source | `transport_speed` | capture trough (75-125) | capture at 900 |
|---|---:|---:|---:|---:|---:|
| `0.04` | 1.659546 | 3.065264 | 0.0384 | 0.760 | 0.769 |
| `0.02` | 1.659738 | 3.059932 | 0.0193 | 0.750 | 0.769 |

Halving the turn halves `transport_speed` and deepens the capture trough by
0.010; by step 600 the two frames capture the same. Target ties; source is
5.3e-3 better at `0.02`, inside the one replicate pair's 5.6e-3. Cosine and
relative step match bin for bin. The turn shows only in the early lag, as
E3 found. Runs `eta0.02-lr6.2e-4`, `d5-inside-6.2e-4`.

### The mask against `polar(g)`: loss-neutral, and `caution_kept` lands on its floor

Standard arm (bs16, 1k, `beta 0.98`, `6.2e-4`, eta `0.04`), the mask's sign
reference swapped from the raw projected gradient to `direction` by runner
patch. 2026-10-03.

| reference | target | source | `caution_kept` | floor | `relative_step` |
|---|---:|---:|---:|---:|---:|
| raw `g` | 1.659546 | 3.065264 | 0.5553 | 0.564 | 4.8e-5 |
| `polar(g)` | 1.659703 | 3.064518 | 0.5632 | 0.564 | 4.9e-5 |

Both loss deltas sit under the noise floor. The read that moved is kept: against
`polar(g)` it sits on the independence floor, uniform across roles (0.561 to
0.564); against raw `g` it sat 0.009 under. The deficit under the floor was a
property of the reference, not anticorrelation between moment and batch: in
the space the moment is made of, a late-run batch is sign-independent of it.
Runs `polarmask-lr6.2e-4`, `d5-inside-6.2e-4`.

### `beta 0.95` against `0.98` at a matched step, bs16

Standard arm at eta `0.02`; `beta 0.95` at `3.8e-4`, the lr that matches
0.98's step at `6.2e-4`. 2026-10-03.

| beta | lr | target | source | `relative_step` | `caution_kept` (floor) |
|---|---:|---:|---:|---:|---:|
| 0.98 | `6.2e-4` | 1.659738 | 3.059932 | 4.8e-5 | 0.556 (0.564) |
| 0.95 | `3.8e-4` | 1.666523 | 3.040345 | 5.0e-5 | 0.588 (0.601) |

The step matched (4%). The shorter memory lost 6.8e-3 target and kept 0.020
source, both past the replicate pair's noise. At a matched bs16 step this is
the trade "Beta buys target at a held step" already reads, not a win: unlike
bs32, the step here is too small for staleness to dominate. Runs
`b095-lr3.8e-4`, `eta0.02-lr6.2e-4`.

### Moment first: the order ties on loss, the decay and beta 0.95 are the gain

Standard arm (bs16, 1k, eta `0.02`), torch 2.14. *Polar-first* is the design
before 2026-10-05: the moment averages `polar(Z_t)`, the cautious mask compares
it with `O_t`, and the step is `lr` times the masked moment. *Moment-first*
averages `Z_t / ||Z_t||` and steps `polar(M_t)`, so `relative_step` is `lr`.
Variants ran by runner patch. 2026-10-05.

Constant lr, step matched to the reference's:

| arm | beta | lr | target | source | `relative_step` |
|---|---:|---:|---:|---:|---:|
| polar-first + mask (reference) | 0.98 | `6.2e-4` | 1.659890 | 3.062551 | 4.9e-5 |
| moment-first | 0.98 | `4.8e-5` | 1.662030 | 3.065155 | 4.8e-5 |
| moment-first, mask before the polar map | 0.98 | `4.8e-5` | 1.662979 | 3.049522 | 4.8e-5 |
| moment-first, mask after the polar map | 0.98 | `4.8e-5` | 1.666439 | 3.036927 | 3.8e-5 |
| moment-first, step times `\|\|M\|\|` | 0.98 | `4.8e-4` | 1.659971 | 3.078005 | 4.7e-5 |
| moment-first | 0.98 | `3.4e-5` | 1.665105 | 3.043454 | 3.4e-5 |
| polar-first, `O_t` masked by `M` | 0.98 | `6.4e-5` | 1.669565 | 3.031216 | 5.1e-5 |

Linear decay to 20% of the peak after warmup (`--lr-final-ratio 0.2`):

| arm | beta | peak lr | seed | target | source | final `relative_step` |
|---|---:|---:|---:|---:|---:|---:|
| polar-first + mask | 0.98 | `1.24e-3` | 1 | 1.652506 | 3.086228 | 2.1e-5 |
| polar-first + mask | 0.98 | `1.24e-3` | 2 | 1.653680 | 3.084472 | 2.1e-5 |
| moment-first | 0.98 | `1e-4` | 1 | 1.650362 | 3.094063 | 2.1e-5 |
| moment-first | 0.95 | `1e-4` | 1 | 1.652291 | 3.060419 | 2.1e-5 |
| moment-first | 0.95 | `1e-4` | 2 | 1.653326 | 3.055182 | 2.1e-5 |

The peak `1.24e-3` puts polar-first's late step on moment-first's: both end at
2.1e-5. Seeds differ by 1.2e-3 target and 1.8e-3 source (polar-first) and
1.0e-3 and 5.2e-3 (moment-first), inside the replicate pair's spread.

- **The order does not decide target.** Two-seed means under the decay:
  moment-first at 0.95 1.652809, polar-first with the mask at 0.98 1.653093.
  At a matched constant step the two sit 2.1e-3 apart, at the edge of the
  spread.
- **The decay is the gain.** Linear to 20% beats each design's constant arm
  on target, by 6.8e-3 (polar-first, two-seed mean) and 0.012 (moment-first
  at 0.98, one seed).
- **Beta 0.95 buys the source.** Under the decay it holds target against 0.98
  (+1.9e-3, one seed) and keeps 0.034 more source, where at a constant matched
  step it lost 6.8e-3 target ("`beta 0.95` against `0.98`"). On one seed at
  0.98, moment-first forgot 7.8e-3 more than polar-first with the mask; the
  0.028 source gain over the old default is beta's, not the order's.
- **What the cut variants did.** The mask before the polar map kept 0.016 more
  source at equal target, which is the lr trade's direction. After it, it
  shortened the step to 3.8e-5 and landed where a lower lr does (the `3.4e-5`
  row). Scaling by `||M||` tied the reference's target and lost 0.015 source. Stepping `O_t`
  masked by the moment was worst on target.
- **The late moment.** Under the decay `grad_moment_cosine` ends at +0.007 and
  `moment_persistence` at +0.001; at constant lr both are negative (-0.009,
  -0.0008). A decaying step stops overshooting what the moment holds.

Runs `lab-ref-torch2.14`, `moment-first-mask-*-b098-lr4.8e-5`, `mf-none-*`,
`ema-judge-b098-lr6.4e-5`, `lab-lin1.24e-3-b098-s{1,2}`.

### The release cleanup reproduces the reference arm

2026-10-28, standard arm (`--beta 0.95 --usuitrack-lr 1e-4 --lr-final-ratio
0.2`, seed 1), on the restructured package with the compiled kernels and every
fallback path removed: `release-clean-s1` 1.652230 target / 3.057482 source
against `mf-branch-verify-s1`'s 1.652199 / 3.058867, inside the noise floor on
both. 1.133 s/step and 3.15 GB peak allocated, the harness's old checkpointing
repair (inside the compiled layer) included. With HF checkpointing outside it
(`b0aade0`), `release-trainmode-s1` reads 1.652065 / 3.059957 (inside the noise
floor), 0.840 s/step, 2.66 GB backward peak; why the old placement was 35%
slower is unmeasured. Last logged reads: `grad_capture` 0.768,
`tangent_live_fraction` 0.828, `grad_moment_cosine` 0.007, `relative_step`
2.1e-5 (the decayed lr).

### Stacking only the tangent eigh keeps most of the speed and none of the transient

2026-10-29, standard arm with the lag reads, seed 1. Stacking every same-shaped
matrix's polar and turn into one call (`release-trainmode-s1`) ran 0.840 s/step
at a 2.24 GB `optimizer.step` peak; everything per matrix
(`release-unstack-s1`) 0.969 s/step at 1.63 GB; per matrix with only the
`[r,r]` tangent-Gram `eigh` batched by rank (`release-eighstack-s1`) 0.887
s/step at 1.64 GB. Losses within 4e-5 target and 8e-4 source of each other,
inside the noise floor: stacking moves speed, not outcome. A microbench put
~0.8 ms/matrix on a solo `eigh` and ~0.4 ms on a solo polar, both launch-bound
at 350M. On LFM3 the stacked transient was the peak that pushed bs16 off the
card (below). The lag reads cost nothing measurable: `lfm25-cos2e5-s1` without
them 0.8870 s/step against 0.8871 with, same build.

### Supervised tokens per step vary 7.5%

2026-10-29, the standard arm's 1000 training batches (bs16 x seq1024, synth,
right-padded), counted on CPU (`outputs/tools/tokens.py`): 11,324 supervised
tokens per step at 350M, CV 0.073, p01-p99 9,298-13,165; LFM3's tokenizer
10,898, CV 0.076, 8,916-12,725. Mean step-to-step change 8%.

### lr at 350M: the plateau cosine changes sign between 2e-5 and 5e-5

2026-10-28, standard arm at `--lr-final-ratio 1` (PLAN L1's 350M half), seed 1;
initial losses 2.297055 target / 2.917386 source. `grad_moment_cosine` as bin
means over steps (wandb history):

| arm | 0-100 | 100-250 | 250-500 | 500-750 | 750-1000 | target | source |
|---|---|---|---|---|---|---|---|
| `lfm25-c5e5-s1` | +0.475 | +0.056 | +0.0018 | -0.0051 | -0.0066 | 1.664033 | 3.041894 |
| `lfm25-c2e5-s1` | +0.604 | +0.176 | +0.034 | +0.0083 | +0.0017 | 1.692191 | 3.010912 |
| `lfm25-cos2e5-s1` (2e-5, cosine to 0.2) | +0.604 | +0.177 | +0.039 | +0.0195 | +0.0173 | 1.717568 | 2.997595 |
| `lfm25-cos4e5-s1` (4e-5, cosine to 0.2) | +0.513 | +0.079 | +0.0106 | +0.0059 | +0.0080 | 1.680789 | 3.018701 |
| `lfm25-cos5e5-s1` (5e-5, cosine to 0.2) | +0.475 | +0.057 | +0.0055 | +0.0032 | +0.0059 | 1.671939 | 3.026230 |
| `release-clean-s1` (1e-4, decay 0.2) | +0.351 | +0.017 | -0.004 | -0.003 | +0.0002 | 1.652230 | 3.057482 |

At 5e-5 the step overshoots the moment from step ~500 on, as 1e-4 does before
its decay brings it back; at 2e-5 the moment leads throughout. Loss orders the
same way: 2e-5 gains 0.028 less target and forgets 0.031 less source than 5e-5.
`relative_step` reads `lr` on both (2e-5 reads 1.37e-5 in the first bin, the
warmup). `grad_capture` ~0.77 and `tangent_live_fraction` 0.81 at the end on
both: neither depends on lr. 1.12 s/step. Cosine decay to 0.2 of 2e-5
(2026-10-29) keeps the cosine positive and higher on the plateau; it gains
0.025 less target and forgets 0.013 less source than constant 2e-5, the
direction a lower constant lr would also take. Cosine decay of 5e-5
(2026-10-30) keeps the cosine positive throughout where constant 5e-5 turns
negative; it gains 0.008 less target and forgets 0.016 less source. Cosine
4e-5 sits between the two cosine arms on both losses, positive throughout;
0.90 s/step on torch 2.15.

## LFM3-1.7B lane

### bs16 fits once the polar runs per matrix

2026-10-29/30, `LiquidAI/LFM3-1.7B-Base` at revision `545fc8f`, bs16, seq 1024,
compile on, `release_matrix_grads`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. With the stacked polar the
peak was `optimizer.step`'s transient (9.03 GB allocated in a 20-step probe),
growing with the count of same-shaped matrices, so with depth; it failed bs16
twice. Per matrix (`f6ae127`) the peak is backward: 7.91 GB allocated, 10.2 GB
reserved, 4.22 s/step at `rank_fraction 0.1` (`lfm3-c5e5-s1`); 7.48 GB, 8.43
GB, 3.84 s/step at 0.05. Five 1k runs since, no failure, with Discord, zapzap
and Zed closed (they held ~1.4 GB of the card).

### lr on LFM3: the cosine crosses zero at a lower lr than at 350M

2026-10-29/30, the standard arm on LFM3 (above), seed 1, eval every 250 (500
on the two constant arms); initial losses 1.857030 target / 2.649378 source.
`grad_moment_cosine` as bin means over steps (wandb history):

| arm | rank | 0-100 | 100-250 | 250-500 | 500-750 | 750-1000 | target | source |
|---|---|---|---|---|---|---|---|---|
| `lfm3-c5e5-s1` (constant) | 0.1 | +0.394 | +0.037 | -0.0029 | -0.0070 | -0.0057 | 1.317608 | 2.703539 |
| `lfm3-c2e5-r05-s1` (constant) | 0.05 | +0.468 | +0.081 | +0.0069 | -0.0042 | -0.0048 | 1.333427 | 2.690231 |
| `lfm3-cos5e5-r05-s1` (cosine to 0.2) | 0.05 | +0.393 | +0.030 | -0.0022 | -0.0022 | +0.0023 | 1.319610 | 2.690622 |
| `lfm3-cos2e5-r05-s1-redo` (cosine to 0.2) | 0.05 | +0.469 | +0.082 | +0.0106 | +0.0035 | +0.0060 | 1.343179 | 2.681620 |
| `lfm3-cos4e5-s1-500` (cosine to 0.2, 500 steps) | 0.05 | +0.415 | +0.043 | +0.0086 | - | - | 1.348516 | 2.676394 |
| `lfm3-const4e5-s1` (constant) | 0.05 | +0.415 | +0.040 | -0.0033 | -0.0081 | -0.0069 | 1.321947 | 2.699859 |

At 350M constant 2e-5 leads throughout and cosine-decayed 5e-5 stays positive
(table above); on LFM3 the first turns negative from step 500 and the second
dips below zero from 250 to 750. At matched rank 0.1 and constant 5e-5, LFM3
turns negative a bin earlier than 350M (250-500: -0.0029 against +0.0018) and
deeper: LFM3 overshoots at a lower lr, so `lr` does not transfer as a
dynamics-preserving constant. How much rank 0.05 adds is unseparated.
`relative_step` reads `lr` on both models. Rank 0.05 against 0.1 at 5e-5 holds
`tangent_live_fraction` at 0.85 against 0.60 and costs capture, 0.70 against
0.77; neither moves with lr.

Loss: cosine 5e-5 matches constant 2e-5 on source (2.6906 / 2.6902) and gains
0.014 more target, so it dominates it; against cosine 2e-5 it gains 0.024 more
target and forgets 0.009 more source. At 350M the same cosine 5e-5 is not
dominant over constant 2e-5 (0.020 more target, 0.015 more forgetting).

`lfm3-const4e5-s1` is the standard arm and the reference for LFM3 reads
(constant `4e-5`, eval every 100; 3.89 s/step, 7.47 GB backward peak). A
shorter run compares at a shared step; at 500 it reads 1.340820 / 2.686102.
Against constant 2e-5 it gains 0.0115 target and forgets 0.0096 more source;
its cosine turns negative in the same bin as constant 5e-5, with capture 0.70
and `tangent_live_fraction` 0.85 on the plateau. The 500-step cosine arm,
annealed by step 500, trails it there by 0.0077 target and forgets 0.0097 less.

A spontaneous gradient burst. `lfm3-cos2e5-r05-s1`, the redo's first run at
the same seed and settings, had one window (steps 550-575) with
`projected_grad_norm` 21x its neighbours; the redo matches it to 4-5 digits
through step 550 and has no burst, so it is nondeterministic, not the data.
The burst left `tangent_live_fraction` 2.5 points and `grad_capture` 0.005 low
for one window, then raised the plateau cosine (bins 500-750 / 750-1000:
+0.0197 / +0.0118 against the redo's +0.0035 / +0.0060) for ~150 steps. Final
losses 1.345078 / 2.683056, 1.9e-3 / 1.4e-3 from the redo: at the noise floor.

### Input-side frames tracked from activations: 0.81 overlap, no eval cost

2026-10-30, PLAN S3, `--activation-frames` (`experiments/activation_frames.py`)
on the LFM3 cosine `2e-5` arm at rank 0.05, seed 1, against
`lfm3-cos2e5-r05-s1-redo` (1.343179 / 2.681620). Input-side matrices only (105
of 153); `X` masked past each sequence's last supervised label. Bin means over
steps (0-100 ... 750-1000):

| read | 0-100 | 100-250 | 250-500 | 500-750 | 750-1000 |
|---|---|---|---|---|---|
| `xframe_overlap` (shadow) | 0.639 | 0.750 | 0.797 | 0.810 | 0.814 |
| `grad_capture_input`, G-frame (shadow) | 0.801 | 0.778 | 0.780 | 0.780 | 0.778 |
| `grad_capture_xframe`, X-frame (shadow) | 0.799 | 0.769 | 0.767 | 0.766 | 0.764 |
| `act_capture_input`, G-frame (shadow) | 0.694 | 0.726 | 0.732 | 0.733 | 0.734 |
| `act_capture_xframe`, X-frame (shadow) | 0.729 | 0.744 | 0.745 | 0.745 | 0.746 |
| `grad_capture_input`, aimed from X (aim) | 0.800 | 0.770 | 0.767 | 0.766 | 0.764 |

The two frames share 81% of their span and still converge slowly; the
activation frame holds 1.4 points less of the input-side gradient than the
gradient's own frame, and the gradient frame 1.2 points less of the
activations than theirs. Aiming the optimizer's input-side frames from `X`
(`lfm3-s3aim-cos2e5-r05-s1`) lands exactly on the shadow's prediction (0.764),
moves overall `grad_capture` 0.709 -> 0.699, leaves `grad_moment_cosine`'s bins
unchanged, and reads 1.344375 / 2.681983: +1.2e-3 target, +3.6e-4 source, inside
the noise floor. `tangent_live_fraction` rises 0.852 -> 0.869 (the input-side
tangent spectrum is now X's). The shadow arm leaves training alone end to end:
1.343303 / 2.681469 against the redo, 1.2e-4 / 1.5e-4. Cost in the harness
(a graph break per hooked Linear): shadow 4.44 s/step against 3.87, aim 4.16;
at 350M shadow 1.13 against 0.89. A 300-step 350M shadow smoke read overlap
0.81, G-frame 0.854 against X-frame 0.825 of input-side `G`.

### Frames from both streams: `X` weighted by `||delta_t||^2` holds G's capture

2026-10-30, PLAN S4, `--stream-frames` (`experiments/stream_frames.py`) on the
standard arm (cosine `4e-5`, 500 steps, seed 1), `s4-streams-s1-500`, against
`lfm3-cos4e5-s1-500`. Every shadow starts as a copy of the optimizer's fitted
frame and turns by the same aim and turn; every read scores a frame on the
gradient it has not yet seen. Means over steps 250-500 (the 100-250 bin reads
the same to within 0.01, so the disagreements are steady, not convergence):

| side | frame tracked from | overlap with G-frame | capture of `G` |
|---|---|---|---|
| input (105) | `G` (the optimizer's) | - | 0.776 |
| | `X`: `X^T X Q` | 0.808 | 0.764 |
| | `X` weighted: `X^T diag(\|\|delta_t\|\|^2) X Q` | 0.928 | 0.775 |
| | agreement core: `X^T (delta Q_o) C` | 0.897 | 0.771 |
| | sum of trace-normalized `X` and norm-input-gradient Grams | 0.584 | 0.739 |
| output (48) | `G` (the optimizer's) | - | 0.541 |
| | `delta`: `delta^T delta Q` | 0.732 | 0.547 |

The diagonal of `delta delta^T` is what separates `X`'s frame from `G`'s: it
lifts overlap 0.81 -> 0.93 and closes the capture gap to 0.001. The
off-diagonal adds nothing the diagonal does not (agreement below weighted on
both reads); the residual-gradient Gram moves the frame away. On the output
side `delta` alone captures as much as `G`. Capture moves far less than
overlap: the 19% of span the `X` frame lacks carries 1.2 points of the next
gradient. One seed; capture gaps of 0.005 or less are unresolved.

Training untouched: 1.348438 / 2.676389 against 1.348516 / 2.676394. Cost of
the instrument (five shadows and the far frame per matrix, hooks that
exhaust dynamo's recompile limit): 5.19 s/step against 3.88, peak 8.39 GB
against 7.34.

### The compiled weight write saves 0.22 s/step

2026-10-30, standard arm, 100 steps, seed 1: `write_weight` compiled
(`cwrite-s1-100`) against run eager (`eagerwrite-s1-100`).

| write | s/step | optimizer peak | target @100 | source @100 |
|---|---|---|---|---|
| eager | 3.95 | 5.03 GB | 1.433068 | 2.671485 |
| compiled | 3.73 | 4.87 GB | 1.433122 | 2.671895 |

Eager makes ~10 passes over `W`; compiled, one fused pass after the norm
and the lift (0.38 against 1.21 ms per matrix). LFM3 compiles three graphs.
Backward peak 7.23 GB in both.

### Training without `dW`: the stream aim ties `G`'s; the input side costs source

2026-10-30, PLAN S2, `--projected-grads` (`experiments/projected_grads.py`) on
the standard arm (constant `4e-5`, 500 steps, seed 1), against
`lfm3-const4e5-s1` at step 500. The `x-side` arms put every tracked matrix on
its input side (a runner patching `resolve_tracked_side`); the others keep the
shape rule (input side on 105 matrices, output side on 48). Target / source:

| aim | shape rule | input side everywhere |
|---|---|---|
| `G` | 1.340820 / 2.686102 | 1.339945 / 2.695103 |
| streams, no `dW` | 1.340507 / 2.685977 | 1.339551 / 2.695242 |

The aim is loss-neutral: the stream arms match their `G` twins to 4e-4 at
every eval. The side is not: input side everywhere forgets 9.0e-3 more source
at 500 under both aims, behind at every eval from step 100 (+5e-3 to +9e-3),
for target that leads early (-8.7e-3 at 100) and ties by 500 (-0.9e-3). It
raises capture 0.703 -> 0.770 (250-500) and `projected_grad_norm` 11%.

Under the stream aim `tangent_live_fraction` reads 0.996 against 0.845: the
stream action has energy in every frame direction where `G`'s leaves 15%
dead, so the read means something else there. Its `grad_capture` divides by
the 16-probe estimate of `||G||` (published as `grad_capture_probe` since
2026-11-15). On the true bs16 gradients of an ordinary
run (`outputs/tools/probe_bias.py`, 20 draws per matrix at steps 1, 25, 100,
200), the estimate's ratio to `||G||` averages 0.990-1.004 on every weight
shape, spread 4-11% per matrix and step (3.7% on the FFN inputs at 200, 8% on
k/v), and inflates a read divided by it by 0.2-1.1% at step 200: about +0.003
of the +0.006 capture over the `G` arm.

Cost: 3.48 s/step against 3.89 (both sides), the 0.4 s the `dW` GEMMs cost;
backward peak 7.76 GB against 7.47, the persistent leaves and their grads
(full-size leaves; as expanded zeros they hold 0.44 GB less, "S2's leaves
hold no storage").

### The stream aim pulls on every plane; the frame does not travel further

2026-10-30, `lag-g-s1-300` against `lag-s2-s1-300` (standard arm, 300 steps,
`--basis-lag-diagnostic --basis-lag-interval 10`), means over steps 250-300:

| read | `G` aim | stream aim (S2) |
|---|---|---|
| `transport_speed` | 0.0198 | 0.0199 |
| `transport_lag` (10 steps) | 0.0610 | 0.0587 |
| `transport_curve` | 0.693 | 0.705 |
| `transport_correction` | 5.1e-4 | 6.6e-4 |
| `tangent_concentration` | 0.70 | 0.48 |

Speed is eta's under the constant turn; on the plateau both frames cover the
same net distance and churn alike, ~70% of the path cancelling within ten
steps, so the new aim asks for no eta of its own. The difference is the aim's
shape: its leading plane holds 0.48 of the tangent Gram against 0.70, the
pull spread over every plane (the 0.996 liveness), and the in-span
correction is 30% larger. Over steps 0-100 the S2 frame travels further and
straighter (lag 0.105 against 0.084, curve 0.47 against 0.55): a one-off move
from the `G`-fitted start toward the stream's own span. Eval at 300: 1.358110
against 1.358713.

### Where the S2 step's time goes: the weight write, then host overhead

2026-10-30, `torch.profiler` over three steady steps (`outputs/tools/
profile_step.py`), and 100-step speed reads from runners that patch the
release (`fused_write.py`, `fused_step.py`), seconds per step over steps
75-100:

| build | s/step |
|---|---|
| reference (`G`) | 3.89 |
| S2 | 3.49 |
| S2, weight write compiled | 3.285 |
| S2, write + blend + polar + turn compiled | 3.268 |

The weight write (SPEC step 8 with its stochastic round) is ~10 dense fp32
passes per parameter: 212 ms per step eager on all 153 matrices, 63 ms as one
`torch.compile` of the same math, 12 ms at the one-read-one-write bandwidth
floor (microbench `write_bench.py`). In the run it saved 0.20 s, losses equal
to 4e-5. Compiling the rest of phase two saved 17 ms: those functions were
never kernel-bound. With the write compiled the optimizer step costs 540 ms of
host time, ~3.5 ms of Python, dispatch and diagnostics per matrix, against 52
ms of kernels; the GPU span of the step is 209 ms because the host starts it
during the backward's tail. Over the whole step 480 ms (13%) of GPU time is
idle, in 5-100 us gaps between ~30k kernels, ~155 ms of it inside the
optimizer step and ~37 ms in `prepare`.

### Rank 0.1 against 0.05: capture without loss, under either aim

2026-10-30, standard arm at 500 steps, `r10-g-s1-500` and `r10-s2-s1-500`
against the rank-0.05 pair above; target / source at 500, reads over steps
250-500:

| aim | rank 0.05 | rank 0.1 |
|---|---|---|
| `G` | 1.340820 / 2.686102, live 0.85, capture 0.70 | 1.340731 / 2.684453, live 0.59, capture 0.77 |
| streams (S2) | 1.340507 / 2.685977, live 1.00, capture 0.71 | 1.340237 / 2.684781, live 0.91, capture 0.78 |

The four arms sit within 6e-4 target and 1.6e-3 source. The stream aim keeps
91% of the planes live at rank 0.1 where `G`'s keeps 59%, and neither turns
the extra capture into loss. Rank 0.1 starts slower (eval at 100: 1.4410 `G`,
1.4387 S2, against 1.4331) and catches up by 300. S2 at rank 0.1 runs 3.73
s/step against 4.03; its leaves scale with rank, backward peak 8.56 GB against
7.69 (+0.88, against +0.29 at rank 0.05), with full-size leaves ("S2's
leaves hold no storage").

### S2's leaves hold no storage

2026-10-30, S2 on the standard arm, 100 steps, seed 1: leaves as expanded
zeros (`s2-anchors-100`) against full-size (`s2-fusedwrite-100`).

| leaves | s/step | forward peak | backward peak | optimizer peak | target @100 | source @100 |
|---|---|---|---|---|---|---|
| full-size | 3.33 | 7.39 GB | 7.66 GB | 5.31 GB | 1.431788 | 2.671430 |
| expanded | 3.37 | 6.95 GB | 7.22 GB | 4.87 GB | 1.431734 | 2.671608 |

Every peak drops by the predicted 0.438 GB (0.88 at rank 0.1); the backward
peak now matches the `G` path's 7.23 GB. The 0.04 s/step is unresolved.

### S2 accumulates

2026-10-30, S2, bs16, 100 steps, seed 1, the same rows split into k
micro-batches. Each frame is fitted in its first backward from that
micro-batch's `G`; at k=1 that is the old first step.

| run | capture | backward peak | s/step | target @100 | source @100 |
|---|---|---|---|---|---|
| k=1, before the token weighting (`s2-fitbwd-100`) | 0.710 | 7.22 GB | 3.42 | 1.431798 | 2.671562 |
| k=1 (`s2-split1-100`) | 0.710 | 7.22 GB | 3.38 | 1.431749 | 2.671502 |
| k=2 (`s2-split2-100`) | 0.708 | 6.01 GB | 3.35 | 1.432345 | 2.670934 |

The k=1 pair sets the float noise, 5e-5. k=2 moves 6e-4 either way, under
the run-to-run floor: the frame fitted from half the batch, or `z` summed in
bf16, unseparated.

### The `[Q | P]` join: 40 ms in the microbench, inside the run's spread

2026-10-30. S2's backward work per step on LFM3 shapes, compiled
(`outputs/tools/s2_bwd_bench.py`): 247 ms, 207 ms with the probes riding the
frame's GEMM over each stream, against an 80 ms floor for reading `X` and
`delta` once at the measured 390-420 GB/s. In the run (`s2-join-100` against
`s2-split1-100`, standard arm, 100 steps, seed 1): target 1.431849 against
1.431749, capture 0.710; steps 75-100 at 3.290 s/step, where three runs of
the code before the join spread over 3.294-3.324.

### S2 ties `G` through 1k steps, 12% faster

2026-10-31, the standard arm at 1k steps, seed 1, with the compiled write and
the `[Q | P]` join: `lfm3-g-1k-s1` (`G`) against `lfm3-s2-1k-s1`
(`--projected-grads`). `G` reproduces the pre-compile reference
`lfm3-const4e5-s1` to 1.9e-4 target and 8.5e-4 source.

| arm | target @1k | source @1k | capture | s/step | peak |
|---|---|---|---|---|---|
| `G` | 1.322141 | 2.699005 | 0.704 | 3.71 | 7.47 GB |
| S2 | 1.322050 | 2.698746 | 0.703 | 3.25 | 7.46 GB |

S2 reads lower target than `G` at all ten evals: 1.4e-3 at step 100 and
6.2e-4 at 200, above the LFM3 null pair's 6e-4 ("Run-to-run noise"), then
within 4.4e-4 from 300 to 1000, inside it; source within 5.7e-4 either way.
The late plateau does not separate the aims; the early lead is a hint, not a
claim.

The global norm of all trainable parameters rises linearly and identically in
both arms, 906.629 at step 25 to 906.717 at 1k (+0.01%). It is the whole
model's norm, so it does not read whether a tracked role rises.

### Run-to-run noise: pairing cancels the batch wiggle, not the drift

2026-10-31, from the eval histories of 273 finished wandb runs
(`outputs/d6/`). Two kinds of noise, read two ways.

**Batch wiggle**, the eval-to-eval jitter around a curve's trend, read by
second differences of evals every 100 steps from step 300 (independent noise
`sigma` gives `Var(d2) = 6 sigma^2`). It is the response to the last batches,
so arms at the same seed share it: their second differences correlate
0.94-0.98.

| lane | wiggle sigma, target / source | left after pairing at one seed |
|---|---|---|
| LFM2.5, 10 runs | 1.3e-3 / 1.1e-2 | 4.7e-4 / 3.0e-3 |
| LFM3, 3 runs | 7.1e-4 / 1.7e-3 | 1.8e-4 / 3.9e-4 |

**Drift**, two runs separating smoothly, which second differences cannot see
and an endpoint comparison feels whole. Read from null pairs, `|delta|` at
the evals:

| pair | lane | target | source |
|---|---|---|---|
| same seed, rescaled `5e-4` arm repeated | LFM2.5 @1k | 1.6e-3 | 5.6e-3 |
| seed 2 against 1, `mf-none-lin1e-4-b095` | LFM2.5 @1k | 1.0e-3 | 5.2e-3 |
| seed 2 against 1, `lab-lin1.24e-3-b098` | LFM2.5 @1k | 1.2e-3 | 1.8e-3 |
| compiled write, other rounding draws (`lfm3-g-1k-s1` / `lfm3-const4e5-s1`) | LFM3, every eval to 1k | <= 5.5e-4 | <= 9.6e-4 |
| same seed, one with a gradient burst at 550 (`lfm3-cos2e5-r05-s1`) | LFM3 @500 / @1k | 1e-4 / 1.9e-3 | 7e-5 / 1.4e-3 |

A changed seed spreads the 1k endpoint no further than a same-seed replicate
does on LFM2.5: by 1k any perturbation has drifted to the same scale, ~1e-3
target and 2-6e-3 source. On LFM3 a null pair stays inside 6e-4 / 1e-3 to
1k unless an event (a burst) kicks it, which moves it 2e-3. Below those, a
same-seed delta at one eval is not a claim; a sign held across every eval
is weak evidence (molegrad sits 9e-5 to 1.4e-3 under `G` on target at all
ten, "S2 ties `G` through 1k steps"). LFM3 has no seed pair.

### UsuiTrack against a memory-matched LoRA: LoRA forgets twice as much

2026-11-01, PLAN R2. The standard arm at 1k steps, seed 1, release code (eager
write): `head-g-1k`, `head-mole-1k` (`--molegrad`), and LoRA at rank 56, alpha
56, on the same matrix set at `1e-4` and `2e-4` (`head-lora56-{1e4,2e4}-1k`).
Rank 56 makes adapters plus both Adam moments the size of rank 0.05's frame
plus moment. Target is SYNTH, source the FineWeb-mix shard; source starts at
2.649378 for every arm.

| arm | target @1k | source @1k | source rise | s/step | peak | state |
|---|---|---|---|---|---|---|
| `G` | 1.322008 | 2.699799 | 0.050 | 4.09 | 7.48 GB | 352.0 MB |
| molegrad | 1.321333 | 2.699158 | 0.050 | 3.66 | 7.47 GB | 352.0 MB |
| LoRA `1e-4` | 1.326224 | 2.737972 | 0.089 | 3.34 | 7.64 GB | 347.4 MB |
| LoRA `2e-4` | 1.323579 | 2.761907 | 0.113 | 3.33 | 7.64 GB | 347.4 MB |

Target by eval, `G` minus LoRA (negative: `G` ahead):

| step | 100 | 200 | 300 | 400 | 500 | 700 | 900 | 1000 |
|---|---|---|---|---|---|---|---|---|
| LoRA `1e-4` | +3.4e-2 | +7.9e-3 | +2.4e-3 | -6e-5 | -2.1e-3 | -3.6e-3 | -4.4e-3 | -4.2e-3 |
| LoRA `2e-4` | +4.9e-2 | +1.8e-2 | +9.9e-3 | +7.2e-3 | +5.1e-3 | +2.2e-3 | -4.7e-4 | -1.6e-3 |

LoRA leads target early and loses the lead, by step 400 at `1e-4` and 900 at
`2e-4`; at 1k `G` is ahead of both, by 4.2e-3 and 1.6e-3, against a 6e-4
floor. Source decides it: LoRA's source rise is 1.8x UsuiTrack's at `1e-4` and
2.2x at `2e-4`, 3.8e-2 and 6.2e-2 past it, 38 and 62 times the floor.
Doubling LoRA's lr bought 2.6e-3 of target for 2.4e-2 of source. LoRA takes
9% less time per step than molegrad and 18% less than `G`, and peaks 0.16 GB
higher: its adapters live in the activations, UsuiTrack's state only in the
optimizer.

Molegrad sits under `G` on target at all ten evals again, 2.1e-4 to 1.3e-3,
with other rounding draws than "S2 ties `G` through 1k steps" but the same
seed. Both arms run 0.38-0.41 s/step slower than in that section, which had
the compiled write (0.22 s/step, "The compiled weight write saves 0.22
s/step") on torch 2.15, against eager on 2.14 here; the rest is not split.
`G` reproduces its loss to 1.3e-4 target and 7.9e-4 source.

### The left aim's row weighting changes nothing

2026-11-13, PLAN S3, `s3r2-streams-s1-500`: the standard arm (constant `4e-5`,
500 steps, seed 1) under `--stream-frames`, which now carries a second
output-side shadow turned by the token diagonal of `G G^T`, `delta^T
diag(||x_t||^2) delta Q`, beside the `delta^T delta Q` shadow Molegrad's left
aim is. Both start as copies of the fitted frame and are read on the gradient
they have not yet seen. Means over steps 275-500, the 48 output-side matrices:

| frame tracked from | capture of `G` | overlap with G-frame | live fraction |
|---|---|---|---|
| `G` (the optimizer's) | 0.5444 | - | 0.567 |
| `delta` | 0.5485 | 0.738 | 0.998 |
| `delta` weighted by `\|\|x_t\|\|^2` | 0.5500 | 0.741 | 0.999 |

The weighting has something to act on, `||x_t||^2` over supervised tokens has
std/mean 0.89, and it moves capture by 0.0015 and overlap by 0.003, the same
sign at every logged step from 100 on and under the 0.005 this read resolves on
one seed. The mirror of the right aim is not a different frame here. On the
right the weighting is the whole effect (overlap 0.81 -> 0.93, "Frames from
both streams"): `||delta_t||^2` selects tokens, `||x_t||^2` does not.

By side, `G`'s own aim turns 0.972 of its planes on the 105 input-side matrices
and 0.567 on the 48 output-side ones; either `delta` shadow turns all of them
and holds as much of `G`. Training untouched: 1.340802 / 2.685818 against the
`G` arm's 1.340820 / 2.686102. The instrument costs 5.88 s/step.

### Output-side capture is low for rank on most writers: what persists is as wide as the frame or wider

2026-11-13, `oracle-capture-s1-125` (standard arm, seed 1, a runner,
`outputs/tools/oracle_capture.py`). Per matrix, on one step's own `G`: the
tracked frame's capture; the oracle, what the best rank-`r` frame on that side
holds (`G`'s top singular values); and the stale oracle, the previous step's
best frame on this step's `G`. Means of steps 101 and 121:

| matrices | `r` | tracked | oracle | stale oracle | random floor |
|---|---|---|---|---|---|
| left, 2048x2048 (24) | 102 | 0.49 | 0.92 | 0.42 | 0.22 |
| left, `w2` 2048x7680 (24) | 198 | 0.52 | 0.87 | 0.46 | 0.31 |
| right, 2048x2048 (11) | 102 | 0.84 | 0.94 | 0.80 | 0.22 |
| right, 512x2048 (22) | 51 | 0.85 | 0.94 | 0.81 | 0.16 |
| right, 6144x2048 (13) | 177 | 0.79 | 0.92 | 0.75 | 0.29 |
| right, `w1` 7680x2048 (48) | 198 | 0.74 | 0.89 | 0.69 | 0.31 |

One step's gradient is low-rank on either side (effective rank 8-30), and the
frame's rank would hold 0.87-0.94 of it. The best frame of one batch holds
0.42-0.46 of the next batch's gradient on the output side and 0.69-0.81 on the
input side, and the tracked frame beats that by 0.04-0.07 everywhere. One seed,
two steps.

Corrected 2026-11-14 (was titled "... because the span does not persist, not
for rank", and concluded rank is not what the output side lacks). That read one
batch. The Gram summed over the batches within 25 steps
(`outputs/tools/ceiling.py` on the recorded stream `record-stream-s1-300`,
means of steps 50, 100, 150, 200, 250), on the writers' output side:

| writer | effective rank | frame rank | energy in its top `r` |
|---|---|---|---|
| `o12` | 269 | 102 | 0.48 |
| `c11` | 91 | 102 | 0.72 |
| `w2` | 263 | 198 | 0.58 |
| `o2` | 106 | 102 | 0.65 |
| `o20` | 180 | 102 | 0.54 |

The three input-side matrices read 5-15. Each batch is low-rank in a different
place, so the best frame for many batches holds only 0.48-0.72 of their summed
energy on the output side. On `o12`, `w2` and `o20` what persists is wider than
the frame; on `c11` and `o2` it is about the frame's width. Withdrawn: "rank is
not what the output side lacks".

### The turn's `eigh` is 20 ms; a polar map of the tangent is another tracker

2026-11-13, PLAN R2 and R3. Timed on the tangents of `s3r2-streams-s1-500` at
step 400 (153 matrices, ranks 8 to 198; step 100 reads the same), between two
device syncs (`outputs/tools/r2_turn_bench.py`):

| | per step |
|---|---|
| the release's turn, whole | 72.6 ms |
| its `tangent_spectra`: Grams and one padded `eigh` per group | 19.9 ms |
| the same `eigh` stacked per rank instead of padded | 21.2 ms |
| Newton-Schulz on each tangent, then the geodesic | 69.1 ms |

`eigh` is 0.5% of the step and padding to the largest rank costs nothing
against per-rank stacks. The other 53 ms is `frame.turn`'s per-matrix
arithmetic, which a polar map does not remove.

Read in the run, each step from the held frame and tangent, none stored
(`outputs/tools/r2_polar_turn.py`), means over steps 275-500: the moment's
quintic on the tangent turns 0.98 of the planes against `eigh`'s 0.845 live,
travels 1.13 as far, and its moved span ends 0.33 of a step's travel from the
`eigh` turn's. The quintic has a soft knee near `1.5e-3 ||Delta||_F` where
`eigh` has a hard floor at `sqrt(r eps) sigma_max`, and 0.57 of the planes sit
within ten floors of the floor, so the knee decides which planes turn. This
measures difference from the `eigh` turn, not which frame tracks better.

### A warm frame matches an exact Gram EMA's; a faster tracker buys the first 100 steps only

2026-11-14. One real gradient stream recorded from the standard arm
(`record-stream-s1-300`, seed 1, 300 steps, eight matrices of layers 2, 11, 12
and 20, uncompiled; `outputs/tools/record_stream.py`), then any tracker replayed
against it offline (`outputs/tools/replay.py`), scored on gradients it has not
seen. Off-policy: the stream comes from training under today's turn. Two reads,
both on the next batch: `cap`, `||G Q|| / ||G||`, and `sig`, the share of the
energy two consecutive batches agree on that the frame holds,
`<G_t Q, G_t+1 Q> / <G_t, G_t+1>`. The replay of today's turn reproduces the
optimizer's recorded frames to three digits.

Steps 201-300, today's turn against the top-`r` eigenspace of an exact EMA of
the Gram (`gram/0.98`), the causal reference: the three input-side matrices
equal it (`cap` 0.83-0.89, `sig` 0.96-0.98); the five writers hold as much or
more `cap` (0.45-0.60) and trail by 0.005-0.05 on `sig`. A frame fitted to the
surrounding 50 steps, later batches included, holds more `sig` on `o12` and
`w2` (0.68-0.69 against the recorded frame's 0.58-0.59, steps 100 and 200), so
the reference is what a causal tracker reaches, not the most a frame could
hold. In steps 31-100 the writers lag: on `o12`, `sig` 0.44 against the
reference's 0.70. On the input side there is no such headroom: the same
non-causal frame (steps 100 and 200) holds `cap` 0.861 / 0.887 / 0.761 on `o12`,
`c11`, `w2` against today's turn's 0.853 / 0.883 / 0.746 over steps 101-200,
with the window Gram's effective rank 16.5 / 2.6 / 10.5 against a frame of 102 /
102 / 198 (`outputs/runs/ceiling-right.log`, `replay-right.log`).

The twin (`docs/topics/subspace-evolution.md`, "Candidates") against the turn,
writers `o12`, `c11`, `w2`, `sig` in steps 31-100 and 201-300:

| start, aim | `eta` 0.02 | `eta` 0.05 | twin 0.95 |
|---|---|---|---|
| fitted, `G` | 0.452, 0.568 | 0.560, 0.609 | 0.703, 0.613 |
| random, `G` | 0.208, 0.512 | 0.402, 0.580 | 0.695, 0.608 |
| fitted, `delta^T delta` | 0.472, 0.543 | 0.501, 0.542 | 0.489, 0.543 |

Readers tie in every row but one: from a random start `eta` 0.02 lags in steps
31-100 (`sig` 0.733 against 0.938 and 0.943). The twin is flat over `beta_f` 0.9-0.98 and `rho`
0.3-3. Under the diagonal aim every tracker ties, at less `sig` than `G`'s own
aim allows.

In loss, 500 steps, seed 1, target at 100 / 300 / 500 and source at 500:

| arm | 100 | 300 | 500 | source |
|---|---|---|---|---|
| standard (`lfm3-const4e5-s1`) | 1.433109 | 1.358706 | 1.340820 | 2.686102 |
| twin 0.95, `rho` 1 (`twin-b95-r1-s1`) | 1.429744 | 1.358707 | 1.341237 | 2.687229 |
| `eta` 0.05 (`eta05-s1`) | 1.430372 | 1.358027 | 1.340464 | 2.686215 |

A faster tracker leads by ~3e-3 at step 100 and by nothing from step 300;
`eta` 0.05 buys what the twin buys. At 500 the twin is 4e-4 behind on target
and 1.1e-3 behind on source, both at the noise floor. The replay's ranking of
the twin over `eta` 0.05 did not show in loss. One seed; the stream was
recorded uncompiled, the loss runs are compiled.

### Residual writers on the input side: the step's direction costs source, at every writer `lr` tried

2026-11-15, PLAN X5. The input side's frame cuts the dimension consistently
for writers too ("A warm frame matches an exact Gram EMA's"), and at equal `lr`
it tied target and forgot 9e-3 more source ("Training without `dW`"). The
question: is that forgetting the step's size or its direction?

At fixed weights (the replay's 8-batch target and source gradients, steps 200
and 250, today's turn), output side against input side: `gain`, the step's
cosine with each loss's mean gradient (x1e3; negative on source means the step
raises source loss), and `bite`, the unit step's K-FAC curvature,
`tr(D dW C dW^T)` over its isotropic value (`outputs/runs/replay-shift.log`,
`replay-right.log`):

| writer | target `gain` | source `gain` | target `bite` | source `bite` |
|---|---|---|---|---|
| `o12` | 14.9 / 22.0 | +0.9 / -2.8 | 35.0 / 25.3 | 15.3 / 15.2 |
| `c11` | 8.2 / 15.4 | -1.5 / -5.0 | 8.9 / 23.3 | 3.9 / 15.2 |
| `w2` | 13.8 / 19.4 | -0.9 / -4.3 | 12.1 / 34.4 | 3.7 / 23.7 |

The input-side step is the archetype that source and target share, so it aligns
with target more and works against source, where the output side's step is
close to orthogonal to source. On `c11` and `w2` it also bites ~2.7x harder, so
equal `lr` is a larger functional step there. Matching that step (the writers'
`lr` scaled by 0.6, `1/sqrt(2.7)`, an unmeasured heuristic, and by 0.35) tests
whether the size alone was the cost.

In loss, 500 steps, seed 1, target at 100 / 200 / 300 / 500, source at 500, and
the source excess at 500 over the standard side's own `lr` line (through the
two standard arms, at the arm's target):

| arm | 100 | 200 | 300 | 500 | source | over the line |
|---|---|---|---|---|---|---|
| standard, `4e-5` | 1.433109 | 1.377298 | 1.358706 | 1.340820 | 2.686102 | -- |
| standard, `2.8e-5` (`std-lr2.8e5-s1`) | 1.451076 | 1.390573 | 1.368870 | 1.348318 | 2.682384 | -- |
| all input side, writers x1 (`xside-w1-s1`) | 1.424580 | 1.372684 | 1.355639 | 1.339912 | 2.695234 | +8.7e-3 |
| writers x0.6 (`xside-w0.6-s1`) | 1.434084 | 1.377949 | 1.359030 | 1.341939 | 2.690363 | +4.8e-3 |
| writers x0.35 (`xside-w0.35-s1`) | 1.441191 | 1.382690 | 1.362381 | 1.344173 | 2.687367 | +2.9e-3 |
| only `o_proj` on the input side (`xside-oproj-s1`) | 1.431844 | 1.376551 | 1.358254 | 1.340516 | 2.688302 | +2.1e-3 |

Scaling the writers' `lr` down walks the input-side arms toward the standard
point, never under the standard side's line: at x0.35 the arm is behind the
standard arm on both losses (+3.4e-3 target, +1.3e-3 source), which needs no
interpolation. One scale for all writers is coarse: `o_proj` bites less on the
input side (25.3 against 35.0), so the scaled arms shrank its step where nothing
called for it. `o_proj` alone at full `lr`, the writer the fixed-weights ratio of
`gain^2` to `bite` favoured most on the input side (19.2 against 6.3; conv 10.2
against 7.6 also, `w2` 11.0 against 15.8 not), ties target within the floor from
step 200 and forgets 2.2e-3 more source, about twice the floor. So the cost is
the direction, not the size, as far as an `lr` scalar can match a step; for
finetuning the writers keep their output side. On target alone, the measure
for pretraining, the input side everywhere at full `lr` is ahead at every eval,
by 8.5e-3 at step 100 closing to 0.9e-3 at 500, one seed: faster arrival, not
yet a standing lead.
Capture reads
0.770-0.772 on the all-input arms and 0.725 on `o_proj` alone against 0.704: more
of `G` held, worse for the model. The line is linear through two points, one
seed; sources at 500 only. The standard row is from 2026-10-30's tree; a
same-tree rerun (`std-lr4e5-rerun-s1`) lands within 2.1e-4 at every eval and
6e-5 on source. Runners `outputs/tools/xside_lr.py`
(`WRITER_LR_SCALE`), `outputs/tools/xside_role.py` (`ROLE`).

### Cutting the input side's turn loses to the full polar turn every step; summing over 5 steps ties

2026-11-15, the replay of "A warm frame matches an exact Gram EMA's" (one pass
over the recorded stream, no run-to-run noise measured), the matrices on the
input side, aim `G`, fitted start. Each tracker against today's turn (polar
tangent of the live planes, `eta` 0.02, every step): the signed difference in
`cap` and `sig` on the next batch over steps 201-300 (x1e3), as a range over the
matrices named. `r` is the frame's rank, `p` a cut tracker's; "state" is what a
tracker holds beyond the frame.

| tracker | matrices | state | `cap` | `sig` |
|---|---|---|---|---|
| sum the normalized tangent, turn every 5 at `5 eta` (`every/5`) | 7 | `[n, r]` | -2 to +7 | -6 to -2 |
| the same every 10 at `10 eta` (`every/10`) | 7 | `[n, r]` | -5 to +3 | -16 to -11 |
| the twin, retracted every 10 (`shiftk/10`) | 7 | `[n, r]` | -5 to +5 | -8 to -3 |
| the tangent summed in a rank-32 state truncated every step, turned every 10 (`sketch/10/32`), `eta` 0.02 | 7 | `[n, 32] + [32, r]` | -39 to -10 | -40 to -13 |
| `sketch/10/32` at `eta` 0.0357 | `o12`, `c11` | `[n, 32] + [32, r]` | -26 to -23 | -43 to -39 |
| `sketch/10/16` at `eta` 0.0505 | `o12`, `c11` | `[n, 16] + [16, r]` | -49 to -46 | -87 to -76 |
| one step's tangent cut to its top 32 planes (`low/32`), `eta` 0.0357 | `o12`, `c11` | none | -5 | -7 to -3 |
| `low/16`, `eta` 0.0505 | `o12`, `c11` | none | -13 to -10 | -14 to -6 |
| the raw tangent, no polar, fixed angle 0.03 (best `cap` of 0.01 / 0.03 / 0.1) | `o12`, `q12`, `w1` | none | -20 to -7 | -54 to -11 |

The 7 are `o12`, `c11`, `w2`, `o2`, `o20`, `q12`, `w1`
(`outputs/runs/replay-every.log`, `replay-sketch.log`); the others are from
`replay-sketch-eqtravel.log` and `replay-diagG.log`. A rank-`p` turn at angle
`eta` travels `eta sqrt(p / r)`: at 0.02 a rank-32 cut on `r` 102 moves 0.0112
per step against today's 0.02. The raised angles (0.0357, 0.0505) restore 0.02
for `low` on `r` 102; the sketches reach 0.016-0.020.

Summing the full tangent over 5 steps ties today's turn; over 10 it keeps `cap`
and loses `sig`. A rank-`p` accumulator, the form that saves state, loses on
every matrix at both angles, by more as `p` halves, and raising its angle
makes it worse while it helps `low` (`low/32` on `o12`: `cap` -10 at 0.02, -5
at 0.0357). At equal travel the sketch loses `sig` ~1.7x faster than `cap`. One
step's tangent cut to 32 planes is within 7e-3 at equal travel, and loses more
at each halving. The raw tangent's `sig` is best at 0.01 on `w1` (-24). Under
Molegrad's diagonal aim (`o12` and `q12`, `replay-diagK.log`) the signs
are the same and the margins 1-3e-3 smaller: raw at 0.03 `cap` -4 and -9,
`low/32` at 0.02 `cap` -7 and -11. The replay's SVD of `old^T new` failed on
`w1` under the diagonal aim and on `o20` under the raised-angle sketches. Not beaten here:
the untruncated twin (`shift/0.95/1`, "A warm frame matches") holds `cap` +2 to
+9 over today's turn on `o12`, `w2`, `o2` and `o20` in 201-300, at `sig` -1 to -5.

### In replay under the diagonal aim, the twin's polar from Newton-Schulz reads as `eigh`'s; retracting every step is closest to today's turn

2026-11-15, three matrices, one stream, off-policy, no loss run: the replay of "A warm frame matches an exact Gram EMA's", the diagonal aim,
input side, fitted start, on the three matrices whose input-side diagonal aim the stream
records (`o12`, `q12`, `w1`; `outputs/runs/replay-shiftK.log`,
`replay-shiftK-readers.log`, `replay-shiftKns.log`). The twin (`beta_f` 0.95,
`rho` 1) with `Y` taking every batch and the frame retracted every `k` steps,
against today's turn: signed difference in `cap` / `sig` x1e3.

| tracker | 31-100 `o12` | `q12` | `w1` | 201-300 `o12` | `q12` | `w1` | travel 201-300 |
|---|---|---|---|---|---|---|---|
| twin, every step | +6 / -3 | +3 / -1 | +5 / -4 | +4 / -2 | +1 / -2 | +2 / -1 | 0.007-0.010 |
| twin, every 5 | -1 / -9 | -5 / -10 | -2 / -15 | +3 / -2 | 0 / -3 | +2 / -2 | 0.003-0.004 |
| twin, every 10 | -7 / -16 | -12 / -17 | -9 / -24 | +2 / -3 | -2 / -7 | 0 / -4 | 0.002-0.003 |
| the tangent summed, turned every 10 (`every/10`) | 0 / -9 | -2 / -7 | 0 / -8 | -1 / -11 | -4 / -15 | -3 / -11 | 0.020 |

Today's turn travels 0.020. At the fixed-weights steps (200, 250) the twin
every step holds more of the source mean gradient (`sig` 0.894 / 0.823 / 0.743
against 0.884 / 0.807 / 0.724) and ties on the target's (-4e-3 to +4e-3); every
10 loses 25e-3 of the target's on `q12`. The twin every step ties today's turn
late and is the nearest to it in 31-100; spacing its retraction out costs
arrival. Summing the tangent for a turn every 10 arrives better than the twin
every 10 and holds less `sig` from step 201. `w1` runs under the diagonal aim here, unlike
the sketch replays above.

The twin's polar factor from the release's Newton-Schulz, then one
retraction, in place of the Gram `eigh`, reads the same to three digits on
every row, every `k` and all three matrices. Over the run, Newton-Schulz alone
leaves the frame up to 0.06 off orthonormal (worst `|QᵀQ - I|`), and the
retraction brings it to 1.5e-4 (`o12`, `q12`, `w1` at `k` 1:
`outputs/tools/replay_ortho.py`, `outputs/runs/replay-ortho-ns.log`). No timing
was measured.

### The frame moment's in-span part needs fp32

2026-11-15, LFM3, native sides, aim `G`, the twin in the release (Newton-Schulz,
one retraction, `beta_f` 0.95). With the whole frame moment `Y` stored in bf16
by stochastic rounding, the standard arm's train loss read 1.8351 / 1.6832 at
steps 25 / 50, against 1.8016 / 1.6209 for the runner's twin with fp32 `Y`
(`outputs/runs/twin-ns-bf16y-s1-killed.log`, killed at 50). The rounding is
unbiased (mean error -9e-6 ulp), but a bf16 EMA built with it sits about 0.7%
rms off its fp32 value, and nearly all of `Y` lies in the frame's span: the
noise on the large in-span part swamps the small horizontal part that moves the
frame. On recorded gradients (`outputs/tools/twin_release_vs_runner.py`), the
bf16 release's frame drifts 4.6e-2 per plane from the runner's by step 40 and
keeps growing; with fp32 `Y` it holds at the bf16 floor. The replay, which
keeps everything fp32, did not see it: `Y` rounded to bf16 there reads the same
`cap`, `sig` and travel as fp32 on `o12`, `w2`, `q12`
(`outputs/runs/replay-storage-G.log`). Storage matters only under stochastic
rounding in the loop.

Stored split, `Y = Q S + H` against the new frame, with `S = QᵀY` `[r,r]` in
fp32 and `H` `[d,r]` in the param dtype, the release tracks the runner at the
floor: frame span difference 2.4-3.8e-3, frame 1-2e-3 and moment 1-5e-4 max-abs,
`|QᵀQ - I|` 5e-4 to 1.2e-3 through step 60 on all three matrices
(`outputs/runs/twin-split-vs-runner.log`). With fp32 `Y` the 500-step arm ends
at 1.341354 / 2.687026, against the runner's 1.341237 / 2.687229
(`outputs/runs/twin-ns-s1.log`, `twin-b95-r1-s1.log`), within the noise floor.
The split arm, stopped at step 250, reads target 1.429670 / 1.376295 at evals
100 / 200, against fp32 `Y`'s 1.429619 / 1.376283
(`outputs/runs/twin-split-s1-stopped250.log`).

Timing in the harness, paired: the tree before the twin (`1f2931d`) then the
twin (`063dc18`), 50 steps each back to back, standard arm with lab
diagnostics, one pass. Mean over steps 1-50: 4.040 against 4.094 s/step, +1.3%;
over 26-50, 4.06 against 4.15 (`outputs/runs/paired-old-1.log`,
`paired-new-1.log`). Unpaired runs had read 3.87-3.91 for the turn and 4.10-4.13
for the twin. The day-to-day drift is larger than the gap. The bench below has
the twin's `step()` 19 ms faster, without the lab diagnostics, so their
per-matrix `transport_correction` may carry the difference. That is not
measured.

Cost on LFM3's 153 matrices, bs16 gradients, `step()` alone
(`outputs/tools/lfm3_step_bench.py`, `outputs/lfm3_step_bench.log`): today's
turn 570 ms median with 334 MiB of matrix state; the twin with the split 551 ms
and 431 MiB (+29%). The Gram-side polar (Newton-Schulz on `[r,r]`, fewer FLOPs,
more launches) took 597 ms: on a launch-bound host the `[d,r]` iteration is
faster. Whole-`Y` fp32 would hold 501 MiB.

### The new defaults on LFM3: the twin trails the turn by 8e-4 target at 500, at the floor

2026-11-15, the standard arm on the defaults of fast-step chapter 1 (input side
everywhere, Molegrad on, the twin): target / source 1.340346 / 2.696886 at 500,
3.56 s/step, 482.5 MiB of matrix state
(`outputs/runs/defaults-xside-mole-s1.log`). The same setup under today's turn
(`s2-xside-s1-500.log`, not paired in time) read 1.339551 / 2.695242 at 3.49
s/step and 334 MiB. Target eval differences, twin minus turn, at 100-500:
-3.5e-4, +1.9e-4, +4.3e-4, +5.7e-4, +8.0e-4. That is a slow drift, up to just
over the LFM3 noise floor (6e-4 / 1e-3), and source +1.6e-3. The twin under `G`
on the input side (`xside-twin-b95-s1.log`) ends at 1.340393 / 2.696862, the
same, and the turn under `G` at 1.339945. So the gap is the tracker's on this
side, not Molegrad's. The matrix state is +44% on the input side, against +29%
on the native sides ("The frame moment's in-span part needs fp32"). Against the
native-side standard arm (1.340764 / 2.686162), the input side costs 1.1e-2 of
source, the ARCHIVE cost the user took.

On nanodit (d768x12, 77 matrices, tiny-dit's wiring with every group on the
input side, `outputs/tools/nanodit_xside_bench.py`, `outputs/nanodit_xside_bench.log`;
two passes, old tree then new, in one sitting) the training step reads 367-370 ms
under the turn (one 402 outlier) and 376-381 under the twin, with `step()`
stubbed 230-236 in both (one 259). The twin's `step()` costs about 9 ms more, about 7%. It
removes the host sync, which is what the CUDA graph needs, but not launches.
Per matrix it runs about as many small ops as the turn did.

### On a small deep model the step costs 38% of training; alone, the GPU idles 60% of it

2026-11-15, not an LFM lane: tiny-dit's nanodit d768x12 conv4 (77 tracked
matrices), 128 px, 4 x 64 micro-batches, compiled blocks, Molegrad and
UsuiTrack as `build_optimizers` wires them (trust step, left/auto sides, beta
0.99, rank 0.1, diagnostics on), bf16, synthetic tensors: timing only
(`outputs/tools/nanodit_step_bench.py`, `outputs/nanodit_step_bench.log`). A
training step takes 357 ms (358 on a repeat, 15 steps each), and 220 ms with
`UsuiTrack.step()` stubbed out, so the step costs 137 ms. The stub leaves
Molegrad's backward work in place and its projected gradients to be dropped at
`zero_grad`. The same 77 matrices and groups stepped alone on synthetic `.grad`
(`outputs/tools/nanodit_step_profile.py`, `outputs/nanodit_step_profile.log`)
take 127 ms unsynced, grad copy included, against 51 ms of summed self-CUDA
time over ~14.3k CUDA events per step (~186 per matrix, mean 3.6 µs): the GPU
idles 60% of the step, waiting on the host. That share is measured alone, not
inside the model. Phases timed between syncs (inflated by the syncs, not
additive; `round_into` runs both after the moment's rotation and inside the
weight write): polar of the moment 42 ms, turn 36, moment blend and diagnostics
29, weight write 26, prepare 19, `tangent_spectra` 12, `round_into` 10. On 72
of these shapes (min dim >= 64, all on the input side) the moment polar stacked
by shape (five stacks, the largest 33 x `[3072, 154]`, a 179 MiB transient)
takes 8.8 ms against 29.6 per matrix (`outputs/tools/polar_stack_bench.py`).

Stacked inside the model (runner patch `outputs/tools/stacked_step.py`, a
timing prototype: same shape in one batched call, chunks capped at 256 MiB of
fp32 work, which no stack here reaches), the training step reads 332 ms with
the moment polar stacked and 308 with the turn stacked too, against 355 per
matrix in the same process (`outputs/nanodit_step_bench_turn.log`; the
per-matrix arm 355.0-355.8 over three passes, 358.7-359.2 in another process):
`step()` falls from 135 to 88 ms. Wiring check, synthetic gradients, 3 steps
from one state on the 77 matrices (`outputs/tools/stacked_step_check.py`,
`outputs/stacked_step_check.log`), max-abs difference from the per-matrix step
over the tensor's max-abs, median / worst over matrices: polar stacked, weights
5.7e-4 / 4.5e-3 and moments and frames bit-identical; polar and turn stacked,
weights 2.4e-3 / 5.0e-3, moments 3.9e-3 / 6.7e-3, frames 3.1e-3 / 6.5e-3,
within one bf16 ulp of the tensor's largest element. No run-to-run baseline
for that difference is measured. Frames' worst `|QᵀQ - I|` 8.0e-4 against the
release's 8.4e-4. The step's peak allocation above its start: 15 MiB per
matrix, 119-121 MiB stacked; with an 8 MiB cap, every stack chunked, 33 MiB and
bit-identical results. Learning under the stacked step is not measured.
The optimizer's diagnostics accumulation, which nanodit runs on, costs 20-22
ms of the step either way (`outputs/nanodit_step_bench_diag.log`, one process,
`diagnostics` set to `None` on the live optimizer: 357.6-358.0 per matrix with
it, 335.8 without; 308.5 stacked with it, 288.2 without), so stacked and with
it off `step()` costs 68 ms. A lower bound on diagnostics' cost: Molegrad's
probe and the `raw_norm` pass in `_prepare` stay in, and the bench never drains
the reads (`pop_diagnostics`).

### Blocks take nanodit's `step()` from 140 to about 62 ms

2026-11-16, fast-step chapter 3: same-shaped matrices step batched in blocks whose
state is one buffer per entry (SPEC "Blocks"). nanodit, every group on the input side
(`outputs/tools/nanodit_xside_bench.py`), paired against chapter 2 (`af8ef61`) over
three passes, alternating trees (`outputs/nanodit_ch3_bench.log`): the training step
reads 370-404 ms per matrix and 291-293 in blocks (the first blocks pass 310-335,
warming); with `step()` stubbed both read 230-264, so `step()` falls from about 140
to about 62 ms, below the polar-and-turn prototype's 88. Same maths: from one state on
CUDA, 5 + 2 same-shaped matrices and Molegrad on 4 stacked Linears, fp32 weights agree
with the per-matrix tree to 1.5e-7 and state to 1e-5 of each tensor's max (summation
order); in bf16 weights and moments differ as much as two per-matrix runs with other
rounding draws, and frames and `H` by the fp32 order's last bits rounded to bf16.
The step's peak above its start, eight bf16 `[1024, 1024]` in one block: 16.7 MiB per
matrix, under the cap's estimate of 19.6. LFM3, standard arm, 100 steps
(`outputs/runs/blocks-s1-100.log`) against the new defaults' run at its step 100
(chapter 1, `defaults-xside-mole-s1`): target 1.423493 against 1.423241, inside the
floor; 3.41 against 3.88 s/step; optimizer peak 4.90 against 5.42 GB allocated,
backward peak 7.38 against 7.49.

### The CUDA write takes nanodit's `step()` from 79 to 45 ms, and LFM3's step 0.2-0.6 s faster

2026-11-16, fast-step chapter 6 (`8cba722`, SPEC step 8 "The write on CUDA"). TF32 in the lift costs nothing
measurable: with an fp32 (`ieee`) `tl.dot` the prototype's two passes cost 22 ms on nanodit (no tensor cores,
the lift formed twice at rank 77), with TF32 4.1-4.8 ms; at lr 2e-3 and 4e-5 the error's RMS against the eager
fp32 result matches `ieee`'s to 3 digits, and the realized step projects onto the intended one at 0.9995-1.0007
under both (`outputs/tools/fused_write_precision.py`, `outputs/fused_write_precision.log`).

nanodit's 77 matrices on synthetic `.grad`, bf16, `step()` alone (`outputs/tools/block_cap_sweep.py`; logs
`outputs/block_cap_sweep.log`, eager write, and `outputs/block_cap_sweep_cuda_write.log`), at the 256 MiB cap:

| write | blocks | wall ms | GPU ms | step peak |
|---|---|---|---|---|
| eager | 17 | 79-81 | 74-75 | +169 MiB |
| CUDA | 10 | 45 | 39-41 | +139 MiB |

The CUDA write holds no weight-sized transient, so its blocks are sized by rank-sized work alone and fewer form.
A rough split: at 10 blocks each, the eager write (1024 MiB cap) reads 69 ms of wall time against the kernel's 45,
so about 24 ms is the kernel's, and the rest of the 79 to 45 the fewer blocks'. Under the eager write the cap trades launches
for peak: 32 MiB forms 64 blocks at 192 ms wall and +14 MiB, 1024 MiB forms 10 at 69 ms and +684 MiB; summed GPU
time is not monotonic in the cap (59, 53, 96, 74, 64, 65 ms from 32 to 1024 MiB), unexplained. The user keeps 256.

LFM3, standard arm, 50 steps, two pairs back to back of `0718526` (eager write) against `8cba722`
(`outputs/runs/pair3-{eager,cuda}-50-{1,2}.log`), each pair matched on eval time: 4.00 against 3.78 s/step at
15.9 s of eval, 3.78 against 3.18 at 13.8 s. The machine changed state between the pairs, so the size reads
0.2-0.6 s, unsettled. Target 1.53115 and 1.53125 against 1.53085 twice, source within 6e-5, inside the floor;
the 100-step run (`outputs/runs/cudawrite-s1-100.log`) reads target 1.423217 against blocks' 1.423493, source
2.676723 against 2.677134. Optimizer peak 5.03 GB against 4.89 (larger blocks hold more rank-sized work),
under the backward's 7.38.

### One polar step cannot follow `Q+Y`; the polar on the Gram takes nanodit's `step()` from 39 to 29 ms

2026-11-16, the frame's polar (user: one Polar Express step on `Y` in place of five). Replay of the twin
(shiftkns/1/0.95/1, diagonal aim) on LFM3's `o12`, `q12` and `w1`, 300 steps
(`outputs/tools/replay_follow_polar.py`, logs `outputs/runs/replay-follow-*.log`). `T = Q+Y` is far from
orthonormal: its smallest singular value over its largest has median 0.029 and worst 0.018 over the stream,
a condition number of 35-55. A step's polynomial maps every singular value by one curve, so one step cannot
bring a spread of 35-55 to one; a warm start would not change `T`'s spread (reasoning, not a measured arm). The release's five steps plus a retraction land 1.8e-5 (mean) and 4.4e-5 (worst)
from the exact polar factor per `sqrt(r)`, and their capture reads as the exact factor's to the 3 digits printed. One quintic
step (the schedule's first) plus a retraction leaves the frame non-orthonormal (worst `|QᵀQ-I|` 1.0; its
distance from the exact factor reads inf, the reference degenerating on the trajectory's `T`) and capture
falls, over steps 201-300, from 0.565 to 0.361 (`o12`), 0.851 to 0.497 (`q12`) and 0.830 to 0.406 (`w1`); one
retraction step from unit mean singular value diverges on `o12` within the first steps (the run stopped there,
`q12` and `w1` unread).

Newton-Schulz's iterates on `T` stay `T C_k`, so the same iterations run on the `r x r` Gram `TᵀT`, the retraction
folded in, and end in one `T C`: about 2 passes over `d r²` against 12. On the stream it lands at most 3.7e-6 from
the iteration on `T`. As batched GEMMs the `r x r` rounds are launch-bound and lose at small `d` (0.81 against
1.15 ms at nanodit's `[12, 768, 38]`; `outputs/follow_polar_bench.log`); the release runs them as one Triton
program per matrix up to `r = 128` (SPEC step 7), for both maps, the frame's and the direction's.

Release kernel against the iteration on `X` (HEAD `9921daf`), nanodit's `step()` by CUDA events, median of the
timed steps, rank fraction at the ranks before the steps-of-32 rule (`outputs/nanodit_polar_release.log`,
`outputs/nanodit_polar_arms.log`; arms from `outputs/tools/nanodit_polar_arms.py`):

| nanodit | rank 0.05 | rank 0.1 |
|---|---|---|
| 12 layers, iteration on `X` | 39.3-39.5 ms | 48.4-49.5 ms |
| 12 layers, Gram, the kernel | 29.1-29.6 ms | 40.0-40.3 ms |
| 32 layers, iteration on `X` | 68.9, 71.9 ms | |
| 32 layers, Gram, the kernel | 59.7 ms | |

The prototype's 64-wide tiles ran rank 0.1 at 43.4-44.1 ms; the `r x r` rounds as batched `baddbmm` saved only
3 ms at rank 0.05 and lost 2 ms at 32 layers. On LFM3's shapes alone, a call (`outputs/tools/lfm3_polar_bench.py`,
`outputs/lfm3_polar_bench.log`): frames `[28, 2048, 102]` 1.75 against 1.76 ms, `[56, 2048, 198]` 10.9 against 4.1,
`[28, 7680, 198]` 19.8 against 4.3; moments `[28, 2048, 102]` 1.70 against 1.51, `[56, 7680, 198]` 34.0 against 8.6;
the `r > 128` ones on batched GEMMs. Every gap is under 5e-7. On real LFM3 moments (`o12`, `q12`, `w1`, `w2`, 300
steps; `outputs/tools/gram_moment_check.py`, `outputs/gram_moment_check.log`) the Gram squares a condition whose
smallest singular value is 4.9e-3 of the largest at worst; both routes sit 1.6-1.7e-2 from the exact polar factor
(the five-step schedule's own error) and agree to the 3 digits printed.

### On a small model Molegrad's backward is bandwidth, not FLOPs; aligned widths are the cheap gain

2026-11-16, nanodit (d768, 12 blocks, 4096 tokens per 64-sample micro-batch), every group on the input side, RTX
4070 SUPER. GPU time per call, mean of 50 back-to-back calls (`outputs/tools/molegrad_backward_bench.py`, its
log): per micro-batch the forward GEMMs take 13.8 ms, `grad_x` 13.9, the `dW` Molegrad replaces 13.8, and
Molegrad's body (`X[Q|P]`, `Z`, the norms, the diagonal aim) 11.6 at rank 0.1 and 10.5 at 0.05; that bench times
each piece alone, so its sums carry launch gaps. In GPU kernel time alone (profiler, `outputs/tools/mole_kernel_gpu.py`,
`outputs/mole_kernel_gpu.log`), at rank 0.05 the body is at the bandwidth floor of the passes it makes, `X` read
twice and `delta` twice: 0.059 ms at 768x768 against 25 MB at 504 GB/s, 0.050; 0.136 and 0.112 ms at 3072->768
and 768->3072 against 63 MB, 0.125 (the second under it, re-reads hitting the 48 MB L2). At rank 0.1 it sits above
the floor, 0.100 ms at 768x768, until padded (below). A Triton body that keeps those reads (two kernels, split-K
with atomics; `outputs/tools/mole_kernel.py`) was never faster than cuBLAS, up to 2.4x slower. Below the floor
is only one read of each, a single-pass kernel with atomics; not built.

Folding `X[Q|P]` into the forward's GEMM as rows of a stacked `[W; Qᵀ; Pᵀ]` (`outputs/tools/fold_forward_bench.py`,
`outputs/fold_forward_bench.log`) saves at best 1.35 ms per micro-batch at rank 0.1 and 0.2 at 0.05, with the
stack prebuilt, padded to 128 rows and `Y`'s row stride free downstream; unpadded it is slower than the two
GEMMs (23.2 against 16.2 ms). The forward GEMM is near compute-bound, so the added columns cost their FLOPs and
the saved read of `X` nearly cancels them. Under checkpointing the forward runs twice and the gain goes. Not
built.

Activation checkpointing per block does not pay on nanodit (`outputs/tools/nanodit_checkpoint_bench.py`,
`outputs/nanodit_checkpoint_bench.log`), forward plus backward per 256 samples, rank 0.1:

| micro-batch | unchecked | checkpointed |
|---|---|---|
| 64 (x4) | 246 ms, 2.21 GiB | 289 ms, 0.71 GiB |
| 128 (x2) | 244 ms, 3.99 GiB | 285 ms, 0.99 GiB |
| 256 (x1) | 243 ms, 7.46 GiB | 288 ms, 1.46 GiB |

The step is GPU-bound at micro-batch 64, so fewer micro-batches buy about 1%, and the recompute costs 18%.

Padding `[Q|P]` and the aim's factor with zero columns to a multiple of 16 (`caadbf3`): the body's GPU time at
rank 0.1 falls 0.100 to 0.081 ms at 768x768, 0.149 to 0.128 at 768->3072 (`outputs/mole_kernel_gpu.log`); at 0.05
by under 0.005. Same harness as the table, micro-batch 64, two A/B pairs (`outputs/molegrad_pad_ab.log`): rank
0.1 from 262 and 255 to 242 ms, and an unpadded 246 in the table's log, so 4-20 ms; rank 0.05 from 240 and 236 to
237 and 230, no claim against the unpadded pair's own 4 ms spread.

### The fused fallback takes nanodit's `StochasticAdamW` step from 12 to 0.5 ms

2026-11-16. nanodit's fallback is 70 tensors and 0.15M elements: 34 gains of 768, 12 of `[6, 768]`, 12 of 64,
9 depthwise convs `[768, 1, 3, 3]`, 3 others. The eager loop's ~20 small ops per tensor made it launch-bound:
11.67 ms wall, 11.65 GPU, on synthetic bf16 grads; one Triton launch over address tables takes 0.31 / 0.30
(`outputs/tools/nanodit_adamw_bench.py`, median of 30). In the training step it goes from 12.2 to 0.5 ms
(`outputs/nanodit_phase_wall_adamw.log` against `outputs/nanodit_phase_wall.log`). The same pair moved backward
178 to 138 ms and `step()` 49 to 42 at rank 0.1: Molegrad's padding, the polar on the Gram and ranks in steps of
32 between them, not separated. The training step, by phase sums, goes from 316 to 258 ms.

### Full space wins wall time at every shape, and GPU time against `r = 32` up to `s = 128`

2026-11-16, PLAN X5. One block's step on bf16 `[2048, s]` and `[s, 2048]`, synthetic `.grad`: tracked, the
release's `step()` with the rank patched; full, a hand-written loop of the step full space would run (`Z = G`
oriented to the short side, the blend, the unit polar, the CUDA write with frame `I` -- the lift the build skips
kept in). The full arm skips the optimizer's per-matrix bookkeeping, so its wall is a lower bound and the wall
gap is not all tracking's launches. Wall per step over 30 steps, GPU kernel time per step from the profiler
(`outputs/tools/full_space_bench.py`, `outputs/full_space_bench.log`; the full arm with the polar kernel at
widths up to 64 only, `outputs/full_space_cap64.log`). Wall / GPU ms, 12 matrices, the two orientations' range:

| s | full | tracked r 32 | r 64 | r 96 |
|---|---|---|---|---|
| 68 | 1.64 / 0.62-0.63 | 3.55 / 0.75-1.02 | | |
| 96 | 1.65-1.66 / 0.67-0.70 | 3.53-3.55 / 0.76-1.03 | | |
| 128 | 1.65-1.66 / 0.77-0.79 | 3.53-3.54 / 0.78-1.06 | 3.57-3.59 / 1.25-1.56 | |
| 192 | 1.64-1.68 / 1.33-1.37 | 3.55-3.61 / 0.84-1.13 | 3.60-3.62 / 1.32-1.64 | 3.66-3.67 / 2.28-2.61 |
| 256 | 1.90-1.95 / 1.85-1.90 | 3.56-3.59 / 0.91-1.19 | 3.58-3.63 / 1.39-1.71 | 3.63-3.70 / 2.37-2.74 |

On GPU time full space is under `r = 32` at `s = 68-96`, ties it at 128, and above crosses near `r = s/3` (ties
`r = 64` at 192, lies between `r = 64` and 96 at 256); at `r = s/4` above 128 tracking is cheaper, 1.39-1.71
against 1.85-1.90 at 256. The polar kernel at its cap of 96 runs the full arm at `s = 68` and 96 in 0.96-1.03 ms
wall and 0.94-1.01 GPU (the first log). One matrix alone: 1.3-1.8 ms wall and 0.18-0.28 GPU in both arms. The
frame on the long side costs the tracked arm 0.3 ms GPU more. State goes the other way: full space holds
`[long, s]` against `[long, r]` plus `2 s r + r²`, 2x at `patch_embed`'s 2048x68 under `r = 32`, 3.5x at
`s = 128`, the most the quarter admits at the 32 grid. The user takes full space from `r >= s/4`, wall over GPU
time and state: a rule, not a measured crossing.

Past the rank ceiling's quarter, one matrix, the same two arms against the rule's `r = s/4`
(`outputs/tools/full_space_wide.py`, `outputs/full_space_wide.log`), wall / GPU ms: nanodit's `ada` 4608x768,
full 2.70 / 2.45 against tracked `r = 192` 3.57 / 0.72; 4096x1024, full 4.91 / 5.60 against `r = 256` 3.51 /
1.11. At 768 full space still wins wall time at 3.4x the GPU time and 2.9x the state; at 1024, where the ceiling
puts the rule's edge, it loses both. One read each, unrepeated.

The `r x r` Gram kernel runs one program per matrix. CUDA events, back-to-back calls of `FOLLOW_SCHEDULE`
(`outputs/tools/gram_kernel_bench.py`, `outputs/gram_kernel_bench.log`): kernel 0.04 ms at width 32, 0.24 at 64,
0.67-0.75 at 68 and 96 at any stack of 1 to 56, 1.5 at 128 (2.2 from 28 matrices up); the eager loop ~1.0 ms
everywhere, its launches. GPU kernel time (profiler, `outputs/tools/gram_eager_gpu.py`,
`outputs/gram_eager_gpu.log`), 1 and 12 matrices: kernel 0.26, 0.75 and 1.69 ms at 64, 96 and 128; eager 0.15-0.24,
0.15-0.32 and 0.15-0.39. At 128 the kernel loses on both counts; at 96 it wins against the launches and costs
0.4-0.6 ms GPU more per call where the step is GPU-bound. The user caps it at 96.

## Run ledger (Anima)

| run | what changed | outcome |
|---|---|---|
| 8 `live_floor` | rank 64 uniform, `1e-5`, derived live floor | first non-destructive run |
| 9 | calibrated table rounded up, `2e-5` | aesthetic, coherent; ended close to base |
| 10 `inner_lr5e5` | measured table, inner-side cross-attn K/V, `5e-5`, fallback `5e-6`, time frozen | perturbed-then-settled; anatomical breakage in places |
| 11 `lr1e4_cosine` | `1e-4`, cosine from warmup | cooler for longer |
| 12 `freeze_fallback` | fallback frozen, resized table, constant turn, `5e-5` | stable, slightly harmed, bland: no creative/aesthetic advance |
| 13 `acc4` | k=4, table from k=4 calibration (sum 595), `1e-4`, fallback `2e-6` | samples good, wants more; undertrained |
| 14 `k1_contrast` | run 13 with k=1 and `5e-5` | half the table dead at k=1; worst smoothed loss of 13-16; warrior slightly broken |
| 15 `rebaseline300` | run 13's config on 2026-09-14 code, 300 steps | tracks run 13; best bird; at step ~300 it moved warrior and bird where run 13 @300 barely had |
| 16 `patch_equalized300` | run 15 + patch-equalized loss, no timestep weight | spectrum unchanged vs 15; samples moved less than 15 (C1 closed) |
| 17 `eta040_300` | run 15 at `eta 0.04` | much faster first 100 steps, then level with 15; slightly rougher early samples that recover |
| 18 `bs16_rf010` | real bs16, `rank_fraction` 0.1, AdaLN under UsuiTrack, no accumulation, `1e-4`, 960 steps = 5 epochs | first full success: stable, better step by step to the end; wants more lr, maybe more epochs |
| 20 `b098_caution_bs8` | bs8, `beta 0.98`, caution, `rank_fraction` 0.025, `9e-4` (1.67x run 18's relative step), anneal last 10%, 1920 steps = 5 epochs | lower loss, degraded samples, not broken (user, mid-run); final verdict pending; lr too high by the fleet reads |
| 21 `trust_ratio` | run 20 under the trust-ratio law (step relative to `||W||`), `3.2e-4` = run 20's bulk step; waists cut to the bulk's | many well-adapted samples, still many bugs, still too hot (user); stopped at 1099; fleet reads identical to run 20 |
| 22 `trust_lr15e5` | run 21 at `1.5e-4`, annealed over the last epoch to 10% | liked, but still hot and slowly degrading (user); stopped at 1130; `grad_moment_cosine` negative from ~128 |
| 23 `b095_lr5e5` | run 22 at `beta 0.95`, `5e-5`, warmup 100 | good quality, little change (user); cosine at or above zero to ~768, negative after; full 1920 |
| 24 `inside_bs16` | run 23 at a real bs16 (offloaded), `7e-5`, one epoch, cosine decay | moved more than 25; inside favoured (user); cosine zero at ~120, positive again under the decay |
| 25 `outside_bs16` | run 24 with agreement outside the step, `6.84e-6` = run 24's step | moved less; different rather than worse (user); cosine zero at ~60 |
| 26 `controller` | run 24 with the polar mask and the lr controller (`exp` form, closed), ceiling `1.4e-4`, 960 steps = 5 epochs | very stable, barely changed, arguably above base (user); vote ~0 from ~100 |
| 27 `last_polar` | bs16, `beta 0.95`, polar-first with the mean of this and the last polar where signs agree (branch `last-polar`), `2.8e-5`, cosine to 10%, 960 steps | improved (user); `relative_step` peaked ~1.3e-5, averaged 7.3e-6 |
| 28 `moment_first` | run 27's shape on moment-first (`lab` `27e09de`), `1.3e-5` flat, cosine over the last epoch to 10% | samples pending (user); `relative_step` 1.15e-5 on the plateau, 12% under `lr`: AdaLN's empty planes ("`relative_step` under `lr` on Anima"); fleet cosine -0.04 to +0.03, persistence within 0.003, capture 0.785 to 0.828; 3.2 h, no OOM |
| 29 `molegrad` | run 28's shape under molegrad, `rank_fraction 0.05`, no nested checkpoint, warmup 50 then linear `2e-5` to `1e-5`, 960 steps | clear win: samples evolve more stably and look better than 28's (user); 2.83 h, 10.0 s/step median, peak 9.30 GiB, no OOM, `optimizer.pt` 483 MB. Over steps 50-350 / 350-650 / 650-960: capture 0.868 / 0.873 / 0.879 against 28's 0.79-0.81; live fraction 0.52 / 0.56 / 0.58 against a flat 0.40, so more of the frame turns as the run goes; cosine -0.008 / -0.005 / -0.004, toward zero as `lr` decays; `relative_step` 0.83-0.84 `lr` (AdaLN, above). Rank, lr and aim all changed from 28 |

Sample reads above are the user's, from wandb, 2026-09-28. Run 15 against run 13
at step ~300 is not matched on schedule: run 15 decays from 225 and run 13 from
432, so run 15 moved more *at a lower lr*. Whether the 2026-09-14 trainer
(token-exact weights, unrounded final fold, fallback divided before the clip)
or something else did that is not separated. Whale, gown and ethereal barely
moved in any run. No run through 19 set `training_seed`, so batch order and
noise differ between every pair of them.

Run 18's step is 3.3x run 17's at the same `1e-4`: `update_to_param_ratio`
`1.20e-5` flat through the stable phase against `3.69e-6`, inside the 2-6x
derived from losing the accumulation shrink and the larger rank. Capture rose
from `.86` to `.91` over the run (run 17 `.75`). `tangent_live_fraction` is
`0.11` throughout against run 17's `0.90`: at 0.1 the aim resolves about one plane in
nine, fleet-wide. That is the E7 question's first number, not its answer --
liveness is not a sizing target (above), and the read is fleet-wide, so
which roles are dead is not known. The card is at its edge: 14 of 960 batches were
skipped on OOM, 3.75 h wall. Telemetry `loss_log.db`, sample read the user's,
2026-09-29.
