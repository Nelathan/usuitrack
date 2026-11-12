# Open design questions

What we are unsure about, how each would be settled, and the evidence that has
moved each answer. `SPEC.md` says what the optimizer does; `FACTS.md` holds the
numbers; `ARCHIVE.md` holds everything closed. See `AGENTS.md` for how this file
is worked.

Questions are grouped so a session can take **one group** and leave the rest
alone. Inside a group the items share runs, reads, or code. Nothing here has a
run yet; each "how to settle" is a proposal, not an agreed arm.

---

## Group N. Weight norm

### N1. Can the step force an ideal weight norm?

The step is relative to `||W||` (SPEC step 8), so a zero matrix never moves
and a small-norm init moves slowly in absolute terms, however much its gradient
says. Fine-tuning starts from trained norms; pretraining inits are non-zero but
far from the final norm. The user's question: since the design favours geometry,
can it target a norm rather than inherit one -- size the step against an ideal
`||W||` for the shape, or move the norm toward it explicitly? Open: what "ideal"
means (an init-scale rule, a measured trained-model norm per role), and whether
that is the optimizer's job at all. How to settle: a small model trained from
scratch, where the init is a free variable.
State: a group's `step_rule="fixed"` moves a zero matrix, and `norm_bounds`
clamps the norm into multiples of its init norm (SPEC step 8); nanodit pretrains
under a ceiling. Open still: the fixed rule's lr against the trust rule's, and a
target norm that is not the init's.

### N2. Rising weight norms, and weight decay

Weight norms rise over runs; the whole model's norm rises linearly and only
+0.01% over 1k LFM3 steps (FACTS, "S2 ties `G` through 1k steps"), which says
nothing per role. A step at a fixed
fraction of `||W||` compounds when it correlates with `W`, so the rise may be
N1's issue seen from the other end. Weight decay is not the remedy for a
Muon-type step (user, 2026-10-30), though `weight_decay` is in the release. How
to settle: `||W||` per role over a standard LFM arm first (cheap, a lab read).

---

## Group R. The first frame and `eigh`

### R1. Range finder against eigh for a finetune's first frame

The first frame is a range finder with one power step, not the exact top-`r`
eigenspace of the first Gram (SPEC step 2). On a flat spectrum it is approximate.
Open: how much `grad_capture` it gives up over the first steps of an LFM or Anima
finetune, where the first gradient carries the subspace. How to settle: a
first-steps read on the LFM lane against the eigh fit of `lab` at `be3f298`.

### R2. Does the turn need `eigh` at all?

The aim is closed-form; `eigh` is left only in the frame turn, one stacked call
on the `[r, r]` tangent Grams per step, which reads its status back to the host
(`frame.tangent_spectra`). Open: whether a polar map of the tangent (Newton-Schulz
or Polar Express, as the moment already uses) moves the frame as well, without
the sync. How to settle: on a branch from `lab`, the turn's angle and live planes
against `eigh` on real gradients, then a standard arm.

### R3. The cost of one padded `eigh`

`tangent_spectra` pads every Gram to the largest rank and runs one call, rather
than one per rank. Measured nowhere on a lane with mixed ranks (LFM, Anima),
where a small Gram pays for the largest. How to settle: step time on the standard
LFM arm against per-rank stacking.

---

## Group S. Subspace gradients

### S1. Accumulation under `release_matrix_grads`

Release frees each gradient in its hook, so it takes one backward per step; a
model whose weights plus gradients do not fit needs both release and
accumulation. Projecting per micro-batch is rank-sized, but the Oja tangent is
quadratic in `G` (`A = G^T G Q`), so summed per-micro-batch tangents are not the
summed gradient's tangent. The in-optimizer fold that did this was removed
(`ARCHIVE.md`, "accumulation is removed"); its open questions (B1-B4 there) are
this one's prior.

### S2. Weight gradients without `G` (molegrad)

The goal (user, 2026-10-30): a FLOP win under checkpointing, not a VRAM win.
Built as `Molegrad` in the package (SPEC, "Molegrad"), opt-in beside the `G`
path (user, 2026-10-31): `Z` exact and the diagonal aim from `X` and `delta`,
~0.1 of `dW`'s FLOPs at rank 0.05; design notes in
`docs/PROJECTED_ACTIVATION_2.md`. On LFM3 it ties `G` through 1k steps at 3.66
against 4.09 s/step (FACTS, "UsuiTrack against a memory-matched LoRA"). Open:

- Anima's batch under molegrad. Accumulation without offloading beats
  offloading: bs8 x 2 is 8% faster, bs4 x 4 3.5% faster at 2.1 GiB less peak
  (FACTS, "Accumulation beats offloading on Anima"). The next run takes
  bs4 x 4 (user, 2026-11-01) and the new `lr` schedule alone; `beta 0.9` is
  considered for after it (user).
  Open: whether bs4 x 4's headroom buys a larger batch, and whether
  the nesting goes from `anima.py` for the `G` path too (it fits unnested at
  rank 0.05). Batch is the noise lever: `beta` only buys staleness, lower `lr`
  is the other lever (user, 2026-10-31); for scale, lodestone trains Chroma at
  bs64.
- The early lead: target under `G` at all ten evals, 1.4e-3 at step 100, above
  the LFM3 null pair (FACTS, "Run-to-run noise"); repeated with other rounding
  draws at the same seed, 1.3e-3 at step 100 and under at all ten. A second
  seed of both arms says whether the diagonal aim buys it.
- The next saving (FACTS, "Where the S2 step's time goes"). The weight write
  is eager (ARCHIVE, "the release's weight write is eager"); a runner can
  compile it for 0.22 s/step. Past it the step is host-bound, ~155 ms of idle GPU in the optimizer and ~37 ms in `prepare`,
  which only a CUDA graph over the post-backward work reaches. That needs the
  release diagnostics to accumulate in place with static counts, `lr` as a
  device tensor, and molegrad's leaves to keep their grads; optional (user,
  2026-10-30: ~5%, clean code first). A Triton write grouped by shape is the
  other lever (user, 2026-11-01); it needs a GPU-only release or a second
  implementation. First, a profile of today's tree splitting host time into
  optimizer, lab diagnostics and molegrad's backward. Parked until the user
  picks it up.
- Accumulation works under molegrad (FACTS, "S2 accumulates"), 6e-4 from the
  unsplit step, holding no gradient: S1's answer for a user who opts in. S1
  stays open for the `G` path.

---

## Group T. Side

### T1. Anima under `auto`, with a few exceptions

ai-toolkit carries a hardcoded side table (residual-facing, plus cross-attention
K/V). Since liveness masking, `auto` may do as well with one or two exceptions.
Settling it needs the per-role reads the release no longer has: open is how the
ai-toolkit lane gets `LabDiagnostics`, which lives in this repo's `experiments/`.

---

## Parked at release

Moved whole to `ARCHIVE.md`, "PARKED at release"; reopen by moving the text
back.

- A1. Does the bf16 stochastic-rounding walk harm the model?
- A2. A walk-aware displacement read.
- D1. The matrix lr's peak and decay shape on each lane.
- D6. LFM run-to-run spread: three seeds of the default arm.
- E4. An arrival sensor for the constant turn.
- E5. The `min(m,n)/2` cap.
- E7. Anima's `rank_fraction`.
- E9. Which roles move, and should step or rank follow them?
- F1-F5. Per-matrix lr from the overshoot meter; magic numbers; device syncs;
  stratified timesteps; BFD packing on LFM.
- "Already tried. Do not repeat." is there too.
