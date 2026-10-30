"""The S4 instrument in `experiments/stream_frames.py`: frames from `X` and `delta`."""

import math

import torch
from torch.utils.checkpoint import checkpoint

from experiments.lab_diagnostics import LabDiagnostics
from experiments.stream_frames import STREAM_READS, StreamFrames, lfm_sites
from usuitrack import UsuiTrack

IN, HIDDEN, RANK, BATCH, TOKENS = 64, 128, 4, 4, 32


class _FeedForward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Linear(IN, HIDDEN, bias=False)
        self.w2 = torch.nn.Linear(HIDDEN, IN, bias=False)


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn_norm = torch.nn.Identity()
        self.feed_forward = _FeedForward()

    def forward(self, h):
        return self.feed_forward.w2(self.feed_forward.w1(self.ffn_norm(h)))


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer()])

    def forward(self, h):
        return checkpoint(self.layers[0], h, use_reentrant=False)


class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Model()


def _train(streams, steps, seed=0):
    """`S` carries the head tokens, which the loss weighs fully; `P`, ten times
    the energy, the tail, which it weighs at 1e-2: `G`'s input span is `S`,
    `X`'s is `P`. A 0.1 isotropic floor keeps `S` from being an exact invariant
    subspace of `X^T X`, and a rank-`r` target puts `delta` on a span. Reads
    cover the last ten steps."""

    torch.manual_seed(seed)
    basis, _ = torch.linalg.qr(torch.randn(IN, 3 * RANK))
    spans, heavy, outputs = basis[:, :RANK], basis[:, RANK : 2 * RANK], basis[:, 2 * RANK :]
    toy = _Toy()
    layer = toy.model.layers[0].feed_forward
    groups = [{"params": [layer.w1.weight], "side": "right"}, {"params": [layer.w2.weight], "side": "left"}]
    optimizer = UsuiTrack(groups, lr=1e-3, beta=0.9, rank_fraction=RANK / math.sqrt(IN * HIDDEN))
    optimizer.diagnostics = LabDiagnostics()
    frames = StreamFrames(toy, optimizer, lfm_sites(toy)) if streams else None
    labels = torch.ones(BATCH, TOKENS, dtype=torch.long)
    target = torch.randn(BATCH, TOKENS, RANK) @ outputs.mT
    scale = torch.cat([torch.ones(TOKENS // 2), torch.full((TOKENS // 2,), 1e-2)]).reshape(1, TOKENS, 1)
    for step in range(steps):
        head = torch.randn(BATCH, TOKENS // 2, RANK) @ spans.mT
        tail = 10.0 * torch.randn(BATCH, TOKENS // 2, RANK) @ heavy.mT
        floor = 0.1 * torch.randn(BATCH, TOKENS, IN)
        if frames is not None:
            frames.set_batch(labels)
        out = toy.model((torch.cat([head, tail], dim=1) + floor).requires_grad_())
        (out * target * scale).sum().backward()
        optimizer.step()
        if step == steps - 11:
            optimizer.pop_diagnostics()
    return layer, optimizer


def test_lfm_sites_map_each_reader_to_its_norm():
    sites = lfm_sites(_Toy())
    assert sites == {"model.layers.0.feed_forward.w1": "model.layers.0.ffn_norm", "model.layers.0.feed_forward.w2": "model.layers.0.ffn_norm"}


def test_streams_read_every_variant_and_leave_training_alone():
    alone = _train(False, 20)[0]
    layer, optimizer = _train(True, 20)
    drained = optimizer.pop_diagnostics()
    assert set(STREAM_READS) <= set(drained)
    torch.testing.assert_close(layer.w1.weight, alone.w1.weight)
    torch.testing.assert_close(layer.w2.weight, alone.w2.weight)


def test_delta_weighting_holds_the_gradient_span_where_x_drifts_off_it():
    # Every shadow starts on the gradient's frame (S); the unweighted X frame
    # turns onto P, the heavy tail delta barely touches, the weighted and
    # agreement frames stay, and the delta frame holds delta's span.
    _layer, optimizer = _train(True, 150)
    drained = optimizer.pop_diagnostics()
    assert drained["stream_x_overlap"] < 0.05
    assert drained["stream_weighted_overlap"] > 0.9
    assert drained["stream_agreement_overlap"] > 0.95
    assert drained["stream_delta_overlap"] > 0.95
