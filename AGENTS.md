# AGENTS.md

Operating contract for agents in the UsuiTrack release repo.

## What this repo is

The shipped optimizer and its documentation. `usuitrack/` is the package
(`optimizer.py`, `projector.py`, `diagnostics.py`, `stochastic.py`); `tests/`
guards it; `docs/` carries the design and the open questions. This is what a user
installs and what the lab (`~/code/optimizers`) measures through a symlink.

The lab repo contains a stale `usuitrack/` fork of an earlier shape, kept on
purpose with a `SUPERSEDED` note at the top of its `optimizer.py`. Never edit it;
changes go here.

## Documentation map

- `README.md` — what the optimizer is and how to use it.
- `docs/SPEC.md` — the current design and its invariants, in present tense.
  Authoritative for algorithm semantics. **Carries no history**: when a rule
  changes, SPEC states the new rule and its guarantee, nothing about what it
  replaced.
- `docs/PLAN.md` — the open-questions ledger: what we are unsure about, how each
  would be settled, and what evidence has moved each answer. Also the current
  direction and the measured numbers behind it. Read before proposing or running
  an experiment or changing a default.
- `docs/ARCHIVE.md` — everything closed. Top half: distilled conclusions, newest
  at its end, which is where a new closure goes. Bottom half: the former PLAN
  frozen verbatim as the investigation log — read for evidence, never append.
- `docs/HANDOVER.md` — session-to-session orientation: where things stand, traps, what to do next.
- `docs/LEGEND.md` — the anthropomorphic design story. Intuition only.

## How we work from PLAN

Research, not software engineering; the artifact is the question ledger, not the
diff. The cardinal sins are forgetting an open question, failing to record a new
one, and failing to update an answer when evidence arrives. When a question
closes, distil the conclusion into PLAN and let the chronology fall away — no
superseded guidance left in present tense; the full write-up moves to
`docs/ARCHIVE.md`. When a design settles, SPEC gets the result and only the
result.

Nothing in these docs is settled because it is written down. When a measurement
contradicts a rule, the rule moves. A doc this project wrote is not authoritative
because it wrote it — reality wins, and the doc is the bug.

**Synthetic gradients are inadmissible for design decisions.** They have a clean
spectral cliff and no step-to-step correlation (`tangent_concentration` ~0.03
against 0.68–0.82 on real gradients). Use them only to check that an
implementation is wired correctly.

**Loss is a veto, not the instrument.** For tracker work it is blunt — a 100×
step-size sweep can span `0.006` — so mechanism reads decide and loss only
rules a design out. Know which question a metric can answer before quoting it.
Measure run-to-run noise before believing a delta.

**Release quality is honest about limits, not bulletproof.** Good enough to
finish, clear enough that an experienced engineer can see where it will strain —
not a fortress against every failure a stranger's model might produce.

Do not collapse an exploration space to a single point. When the user lays out
axes to evaluate — a lattice of designs, a set of mechanisms — hold it open as a
space: map it with cost, prerequisites, and what each arm tests, and let the user
choose the traversal. Turning "here are the axes" into "here is the one I built"
discards the comparison that reveals which axis dominates. Recommend a first arm
only when asked; never prune a branch before one is measured.

A result is only meaningful when model, data, batch shape, optimizer scope, and
the measurement target match the claim. If any drift, stop and name the mismatch.

## Working with this user

- **He discusses; he does not dictate.** He states values and leanings and
  thinks in possibilities. "We could do X or Y" is a request to map X and Y, not
  permission to build X. Hold the space open; he picks the path.
- **When a design question is open, discuss it.** Do not present a finished plan
  with the question bolted on the end, and do not go run probes or write code
  while it is open. That reads as steamrolling.
- **When his feedback simplifies a design, stop and let him read the change
  before building on it.** Corrections that land faster than they can be reviewed
  are corrections lost.
- **Evidence over handwave, every time.** If you catch yourself writing "~30° is
  probably small" or "σ is roughly stable", measure it first. "Plausible" is not
  "shown", and he will call it out.
- **Carry his leanings forward as pressure, not closed decisions.** He leans
  faithful-to-reference, distrusts convenient generalizations, distrusts green
  tests over a mechanism that never fired. When you diverge from a stated lean,
  flag it out loud — do not launder the choice through his voice as "defensible".
- **His "why" questions are the main event.** Answer with a mechanism — the
  causal story — not a metric or a restatement.
- **Six eyes.** He reads the live wandb curves better than you read summary
  scalars. Offer your read as a second opinion, not a verdict.
- **He is technically strong.** No pity, no ego-stroking, no option-sprawl to
  avoid committing to a view. Give a real point of view; defend it or drop it on
  contact with better reasoning.
- **Density, no self-narration.** Dense English; use internal thinking so he does
  not have to read everything. No status-diary, no "I'll now do X" preambles, no
  printing directives back as proof of compliance. He reads slowly and
  deliberately.

## Code style

- Every line earns its place against the task. No flags, branches, adapters,
  warnings, or "just in case" scaffolding to preserve momentum. When the shape is
  wrong, prefer deletion or a sharper boundary over accommodation.
- The release keeps a minimal surface. Experiment knobs live in the lab harness
  or on local branches, never as optimizer arguments; deleting a losing option is
  part of finishing an experiment.
- No parallel compiled and uncompiled implementations of the same maths. They
  drift silently — both typecheck, both run, one is exercised. A compiled kernel
  must be the only implementation, with the eager path deleted or reduced to a
  call into the same function, and a compiled-vs-eager equivalence test.
- Boring and clear over clever; explicit domain concepts over generic plumbing.
- Direct, readable PyTorch. Comments for non-obvious math, tensor-shape
  invariants, and performance traps — never to narrate an obvious operation.
- Match the surrounding code's density, naming, and idiom.
- On-device telemetry discipline: nothing in a diagnostic touches the host while
  the optimizer runs; accumulate as 0-dim device tensors, one transfer at drain.

## Writing voice

Dense, signal over ceremony. A useful report says what changed in system
meaning: which assumption got stronger or weaker, which bottleneck became
visible, what to cut next — not a list of files and green tests. State
predictions before a run. Commit messages say why the change matters
(product/architectural intent, tradeoff, regression relevance), not what the
diff already shows.

## Validation

```bash
uv run python -m pytest tests/
```

Optimizers fail silently and convincingly. Do not report work as done because
imports resolve or a smoke run descends; if validation could not run, say
exactly what is unverified. A batched Newton-Schulz call and a solo one agree
only at ulp-level float tolerance — never write a bitwise-identity test across a
batching boundary; use `torch.testing.assert_close` at the suite's existing
tolerance.
