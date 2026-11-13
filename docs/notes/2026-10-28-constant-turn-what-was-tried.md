# The constant turn: what was tried

Prose cut from `docs/SPEC.md` at the 2026-10 release cleanup. Verbatim: present tense is the time of writing, and "above" points into SPEC as of `d76ea99`.

The time axis was tried and did not deliver either. A gain scaled by how well
consecutive top-`k` aims agreed, divided by an attainable ceiling
`(\sum\lambda)^2/(\sum\lambda^2 k)`, shipped for several runs and is now
removed on two grounds: the ceiling is a magnitude functional gating a step whose
whole purpose is to be magnitude-free, and on Anima it annealed the turn by 1.21x
over 2100 steps while on LFM it looked like it worked. The anchor-free two-lag
form that would replace the ceiling is separately falsified -- the aim's
persistence *shape* is stationary while its level is not, so the ratio holds near
0.81 and anneals nothing (`ARCHIVE.md` on `lab`).

The shape of the idea remains right: a converged frame orbits on batch noise and
should stop turning. What does not exist is a sensor that can tell arrival from
batch noise, and at bs4 a spreading aim is what noise alone produces. Until one
exists the turn is constant, which at least makes frame motion a single term that
`transport_speed` and `transport_curve` read directly. `eta` is then the only
handle, and scheduling it on the prior that aiming gets harder as the surface
smooths is a knob, not a sensor.

There is no cold start. It existed because the gain had no history on the first
update and because a full turn at an `eta` chosen for a scale near 0.05 failed
`eigh`; masking dead planes removed that failure, whose cause was rounding
artifacts promoted to unit-norm directions by `1/sigma`, not step size.
