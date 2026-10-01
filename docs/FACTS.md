# Facts

Measured numbers that neither state a rule (`SPEC.md`) nor frame an open question
(`PLAN.md`). Each carries where it came from. A fact here is a measurement at a
date, not a law; when a new read contradicts one, replace it and say so in PLAN.

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
optimizer state is resident; nested runs at ~12.4 s/step. Gradients agree with
eager within the compile noise floor (relative grad-norm difference median
~1e-2, as for plain compile against eager).

After a step, with AdaLN still on the fallback: 5.48 GiB allocated, 8.32
reserved, 7.88 step peak. Parameters 3.89, UsuiTrack state ~0.76, the fallback's
AdamW moments for the AdaLN pairs ~0.68 -- 48% of optimizer state for 9% of the
parameters, gone since they moved to UsuiTrack (`optimizer.pt` 1.41 -> 0.97
GiB). `nvidia-smi` reads reserved allocator cache, not use: 11.4 GB at 20
steps, and the card is still near full -- a 570 MiB request failed once and
succeeded on the allocator's retry.

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

## LFM2.5-350M lane

bs16 x seq1024, seed 1, 1k steps, broad-no-embeddings. Noise floors: target
`3e-4`, source `2e-3` -- contradicted by a same-seed replicate at 1.6e-3 /
5.6e-3 (below, "Caution: where it pays").

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
comparison (`PLAN.md` D5).

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

`PLAN.md` D5's law (`SPEC.md` step 8) against the aspect rule, 1k, bs16,
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
| 21 `trust_ratio` | run 20 under D5's law (step relative to `||W||`), `3.2e-4` = run 20's bulk step; waists cut to the bulk's | many well-adapted samples, still many bugs, still too hot (user); stopped at 1099; fleet reads identical to run 20 |

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
