"""The S3 instrument in `experiments/activation_frames.py`: tracking input-side frames from X."""

import math

import torch

from experiments.activation_frames import ACTIVATION_READS, ActivationFrames
from experiments.lab_diagnostics import LabDiagnostics
from usuitrack import UsuiTrack

IN, OUT, RANK, BATCH, TOKENS = 64, 128, 4, 4, 32


def _planted():
    """Two orthogonal `[IN, RANK]` spans: `S` carries the supervised tokens,
    `P` the unsupervised tail, at ten times the energy."""

    basis, _ = torch.linalg.qr(torch.randn(IN, 2 * RANK))
    return basis[:, :RANK], basis[:, RANK:]


def _train(mode, steps, supervised_until=TOKENS // 2, seed=0):
    torch.manual_seed(seed)
    spans, heavy = _planted()
    layer = torch.nn.Linear(IN, OUT, bias=False)
    optimizer = UsuiTrack([{"params": [layer.weight], "side": "right"}], lr=1e-3, beta=0.9, rank_fraction=RANK / math.sqrt(IN * OUT))
    optimizer.diagnostics = LabDiagnostics()
    frames = ActivationFrames(layer, optimizer, mode) if mode != "off" else None
    labels = torch.full((BATCH, TOKENS), -100)
    labels[:, 1:supervised_until] = 1
    target = torch.randn(BATCH, TOKENS // 2, OUT)
    for _ in range(steps):
        head = torch.randn(BATCH, TOKENS // 2, RANK) @ spans.mT
        tail = 10.0 * torch.randn(BATCH, TOKENS // 2, RANK) @ heavy.mT
        if frames is not None:
            frames.set_batch(labels)
        out = layer(torch.cat([head, tail], dim=1))
        # Only the head is supervised, so delta, and with it G, never sees the tail.
        (out[:, : TOKENS // 2] * target).sum().backward()
        optimizer.step()
    return layer, optimizer, spans, heavy


def _overlap(frame, span):
    return float((frame.float().mT @ span).square().sum() / span.shape[1])


def test_shadow_reads_both_frames_and_leaves_training_alone():
    alone = _train("off", 20)[0]
    layer, optimizer, spans, _heavy = _train("shadow", 20, supervised_until=TOKENS)
    drained = optimizer.pop_diagnostics()
    assert set(ACTIVATION_READS) <= set(drained)
    assert _overlap(optimizer.state[layer.weight]["frame"], spans) > 0.99
    # Every row is in the mask, so the activation frame sits on the heavy tail G never sees.
    assert drained["xframe_overlap"] < 0.05
    assert drained["grad_capture_input"] > 0.99 and drained["grad_capture_xframe"] < 0.05
    assert drained["act_capture_xframe"] > 0.9 > drained["act_capture_input"]
    torch.testing.assert_close(layer.weight, alone.weight)


def test_rows_past_the_last_supervised_label_never_reach_the_activation_frame():
    _, optimizer, _, _ = _train("shadow", 20)
    drained = optimizer.pop_diagnostics()
    assert drained["xframe_overlap"] > 0.99
    assert drained["grad_capture_xframe"] > 0.99


def test_aim_turns_the_optimizers_frame_onto_the_activations():
    # The frame is fitted to G, on S; aimed by unmasked X it turns at eta per
    # step onto P, a quarter turn away.
    layer, optimizer, spans, heavy = _train("aim", 150, supervised_until=TOKENS)
    stored = optimizer.state[layer.weight]["frame"]
    assert _overlap(stored, heavy) > 0.95
    assert _overlap(stored, spans) < 0.05
