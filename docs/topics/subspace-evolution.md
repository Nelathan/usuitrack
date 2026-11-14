# Subspace evolution

Topic doc for PLAN group X, opened 2026-11-13. It holds this line's questions,
framing and candidate reads; measured numbers go to FACTS, closures to ARCHIVE.

## Status (2026-11-13)

Nothing measured, no instrument built, no arm agreed. Every "how to settle"
below is a candidate, not a choice.

## The goal

Two aims (user, 2026-11-13), the second the crucial one:

1. Can what this repo has learned break the wall the paper below ends at?
2. Can both sets of insight be used to make the frame's rank evolve faster?

The underlying problem, the user's *subspace evolution problem*: how to get the
tracker to explore the full-rank space, which is hard when activations cluster.

## Source

Eyler et al., "Stuck in a Loop: Runaway credit ratcheting in attention
matrices", 2026-09-09, https://hyperstition.cc/stuck-in-a-loop (read
2026-11-13). Motivation, not evidence: it says nothing on weight-gradient
rank, on optimizers past plain gradient descent, or on activations; every link
to the frame below is ours.

- For one head, `y_i = sum_j a_ij v_j`. The value gradient per key is
  `a_ij delta_i`; the query gradient is `(1/sqrt d) sum_j a_ij [delta_i .
  (v_j - y_i)] k_j`. Every term carries `a_ij`, so a key at attention `eps`
  gets `eps` of the signal however useful it is ("softmax deadlock"). No key
  gradient is given.
- Jacobian `dy_i/dq_i`, 128x128, Qwen2.5-1.5B, head 0 of layers 2, 14, 26, 56
  sequences: the text says 99% of tokens under effective rank 10 on the squared
  spectrum; its table's medians are 10.94 / 3.23 / 13.28 under a row labelled
  "normalized sigma". Unreconciled.
- Lock-in: attention at step 1k is closer to its own step-10k state than two
  runs' ends are to each other. No metric, model or seed count named.
- NATQ: each query's gradient preconditioned by the damped inverse Fisher of
  its attention row, then rescaled to its original norm; query gradient only.
  124M model, 5k steps: attention entropy 3.75 -> 4.69 nats, validation loss
  worse (a plot; 0.035-0.110 nats excess by group). No code.

## Three gates of one shape

Each scales a direction's learning signal by how much the model already uses
it.

1. **The softmax gate** (the paper's), inside attention: a key's signal is
   scaled by its attention weight.
2. **The activation gate**, the model's: a Linear's gradient is `delta^T X`, so
   its input side lies in the span of the batch's activations. Where no
   activation is, there is no gradient, even if one would improve the model
   (user). No optimizer sees past it.
3. **The frame gate**, the optimizer's: the step uses `Z = G Q` only, and the
   frame turns along the horizontal part of `G^T G Q`, toward what the batch's
   gradient couples to the frame, ranked by energy. A frame is selected; tails
   are not cut, but they are seen by the tracker alone.

What this repo knows that bears on them:

- **Polar after aggregation is the placement that works** (many variants
  tried; SPEC step 5). NATQ whitens per query, before any sum, and loses. "A
  healthy polar output can hide a sick input" (SPEC): equalizing cannot recover
  what fell under the noise before it.
- **Inside the frame nothing is rich-get-richer.** The moment blends
  `Z/||Z||` and the step is its unit polar factor, so an admitted low-energy
  direction steps at full length. Energy decides admission only.
- **Following energy is natural under noise** (user): the gradient is already
  very noisy, and high energy is where noise is the smaller fraction. An
  admission rule that leaves energy has to pay for that.
- **A settled frame is not evidence of lock-in.** It orbits by design (FACTS,
  "The turn matters while the aim moves, and not once it has settled"), and low
  speed with low curve does not separate settled from starved
  (`docs/notes/2026-10-28-diagnostics-tiers-and-transport-reads.md`).
- **Side decides what the frame can filter.** A square `W_Q`/`W_K` resolves to
  the input side, so `Z` keeps its output side, where a change to the attention
  score gradient lands, whole; with `d_head_total < d_model` under `auto` the
  frame sits on the output side and would filter it. Check per role before
  reading tiny-dit's backward-temperature experiment.

## Questions

### X1. Archetype or detail: where does the gradient's energy lie?

*Hypothesis* (user, 2026-11-13, stated as a guess): there is a large gradient
where there are large activations, and there may be a core strong activation
line that carries no nuance. If the energy lies in that archetype, an
energy-ranked frame spends its planes on it; if it lies in the details, energy
is doing the right thing. Unknown which.

How to settle: undecided. First reads belong on the `G` path, where the real
residual is in hand; Molegrad's probe sketch is not a residual read.

### X2. Settled or starved?

For a frame that has stopped travelling: is there gradient outside it that
persists across steps and is never admitted, or only batch noise?
`1 - grad_capture^2` says how much lies outside; no read says whether it is the
same directions each step. ARCHIVE's parked E4 (an arrival sensor) is adjacent.

One number bears on it (FACTS, "The left aim's row weighting changes nothing"):
under `G`'s own aim the 48 residual writers turn 0.567 of their planes a step
against 0.972 on the input side, and a frame aimed by `delta^T delta Q` turns
all of them and holds as much of `G`.

A second (FACTS, "Output-side capture is low for rank"): a rank-`r` frame
could hold 0.87-0.94 of one step's gradient on either side, but one batch's best
frame holds only 0.42-0.46 of the next batch's on the output side, 0.69-0.81 on
the input side. Averaged over 50 batches the output-side Gram has effective rank
95-293 against 5-15 on the input side: the persistent gradient there is wider
than the frame.

How to settle: undecided. One candidate, unendorsed: at fixed weights, the
agreement of two half-batches on the leading directions of `G (I - Q Q^T)`,
against the same number for `Z`.

### X3. Is energy the right admission rule?

Only if X1 or X2 finds something energy keeps out: an archetype that fills the
frame, or a persistent residual that never enters. Open then: what else could
rank a direction, and what it costs in capture and in noise. No mechanism is
proposed.

### X4. Does aggregate-then-polar reach the paper's wall?

The paper's fix equalizes per query and fails. A polar step on the aggregated
`W_Q`/`W_K` update is the same intent at the placement that works here, and
UsuiTrack already takes it. *Hypothesis:* a buried key's pull, consistent over
tokens, forms a direction of the aggregate and gets a full-length step, if it
clears the noise and the frame admits it. Whether attention then spreads, or
locks in later, is unmeasured. It needs pretraining, so the lane is tiny-dit.

## Dropped or elsewhere

- Frame rotation as a sensor for attention lock-in: withdrawn, it conflates
  frame evolution with q/k evolution.
- "Activation-side tracking explores worse because it is PCA of the inputs":
  withdrawn. Molegrad's right aim is weighted by `||delta_t||^2`.
- Consensus-gated boosting of buried keys (the paper's own rule): dropped by
  the user; a gate that requires agreement rewards redundancy, and the wish is
  diversity.
- Backward temperature on the attention score gradient, and Interleaved Head
  Attention as a priming device: tiny-dit's.
