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

### N2. Rising weight norms, and weight decay

Weight norms rise over runs (observed, unmeasured as a table). A step at a fixed
fraction of `||W||` compounds when it correlates with `W`, so the rise may be
N1's issue seen from the other end. `weight_decay` is in the release and
untested on either lane. How to settle: `||W||` per role over a standard LFM arm
first (cheap, a lab read), then a decay arm against it.

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
`Z` and the exact Oja action come from `X` and `delta` in four skinny GEMMs
wherever autograd would form `dW`, at ~0.1 of its FLOPs at rank 0.05. Design,
wiring candidates and the order of work: `docs/PROJECTED_ACTIVATION_2.md`.
The input-side frame tracks from `X` weighted by `delta`'s token norms
(ARCHIVE, "frames track from the streams"): its action `X^T (w^2 * X Q)` reuses
the `X Q` that `Z` needs, one GEMM where the exact action takes two. Built as
`--projected-grads` (FACTS, "Training without `dW`"): compiled, it collects
the whole `dW` cost and ties `G`'s aim on loss. Open:

- The next saving (FACTS, "Where the S2 step's time goes"). The weight write
  is compiled on `lab` (R1 asks about the release). Past it the step is
  host-bound, ~155 ms of idle GPU in the optimizer and ~37 ms in `prepare`,
  which only a CUDA graph over the post-backward work reaches. That needs the
  release diagnostics to accumulate in place with static counts, `lr` as a
  device tensor, and S2's leaves to keep their grads; optional (user,
  2026-10-30: ~5%, clean code first).
- Accumulation works under S2 (FACTS, "S2 accumulates"), which may answer
  S1 for the harness; whether it costs loss needs a matched-data pair.

---

## Group T. Side

### T1. Anima under `auto`, with a few exceptions

ai-toolkit carries a hardcoded side table (residual-facing, plus cross-attention
K/V). Since liveness masking, `auto` may do as well with one or two exceptions.
Settling it needs the per-role reads the release no longer has: open is how the
ai-toolkit lane gets `LabDiagnostics`, which lives in this repo's `experiments/`.

---

## Group R. Release

### R1. Does the compiled weight write enter the release?

Compiled on `lab` only (FACTS, "The compiled weight write saves 0.22
s/step"). Its cost to a user: inductor (Triton, or a C++ toolchain on CPU), a
first-step compile, and a budget of 8 graphs that a mixed bf16/fp32 model
fills, past which it runs eager silently.

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
