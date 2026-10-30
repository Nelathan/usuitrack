"""PLAN S2 in `experiments/projected_grads.py`: weight gradients without `G`."""

import math

import torch
from torch.utils.checkpoint import checkpoint

from experiments.projected_grads import ProjectedGrads, ProjectedLinear
from usuitrack import UsuiTrack, frame

IN, HIDDEN, BATCH, TOKENS = 64, 128, 4, 32


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Linear(IN, HIDDEN, bias=False)
        self.w2 = torch.nn.Linear(HIDDEN, IN, bias=False)

    def forward(self, h):
        return checkpoint(lambda h: self.w2(torch.tanh(self.w1(h))), h, use_reentrant=False)


def _setup(seed=0):
    torch.manual_seed(seed)
    block = _Block()
    groups = [{"params": [block.w1.weight], "side": "right"}, {"params": [block.w2.weight], "side": "left"}]
    optimizer = UsuiTrack(groups, lr=1e-3, rank_fraction=8 / math.sqrt(IN * HIDDEN))
    return block, optimizer


def _batch(seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH, TOKENS, IN, generator=generator), torch.randn(BATCH, TOKENS, IN, generator=generator)


def _loss(block, batch):
    inputs, target = batch
    return (block(inputs) * target).sum()


def _projected_step_two(batches):
    """Step one on the ordinary path; then the leaves of step two's backwards."""

    block, optimizer = _setup()
    projected = ProjectedGrads(block, optimizer)
    _loss(block, _batch(0)).backward()
    optimizer.step()
    for batch in batches:
        _loss(block, batch).backward()
    projected.prepare()
    return block, optimizer


def test_switches_after_the_fit_and_freezes_the_weights():
    block, optimizer = _setup()
    ProjectedGrads(block, optimizer)
    _loss(block, _batch(0)).backward()
    assert type(block.w1) is torch.nn.Linear
    optimizer.step()
    assert isinstance(block.w1, ProjectedLinear) and isinstance(block.w2, ProjectedLinear)
    assert not block.w1.weight.requires_grad


def test_projection_is_exact_and_the_action_comes_from_the_streams():
    block, optimizer = _projected_step_two([_batch(1)])
    # The ordinary gradient of the same weights on the same batch, and the streams.
    twin = _Block()
    twin.load_state_dict(block.state_dict())
    inputs, target = _batch(1)
    hidden = torch.tanh(twin.w1(inputs))
    hidden.retain_grad()
    (twin.w2(hidden) * target).sum().backward()
    for module, twin_module, side in ((block.w1, twin.w1, "right"), (block.w2, twin.w2, "left")):
        prepared = optimizer._pending[module.weight]
        basis = optimizer.state[module.weight]["frame"]
        grad = frame.orient(twin_module.weight.grad, side)
        torch.testing.assert_close(prepared.projected, grad @ basis)
        estimate_ratio = (prepared.raw_norm / grad.norm()).item()
        assert 0.5 < estimate_ratio < 1.5
    # Input side: X^T diag(||delta_t||^2) X Q, with delta the gradient at w1's output.
    x = inputs.reshape(-1, IN)
    delta = (hidden.grad * (1 - hidden.detach().square())).reshape(-1, HIDDEN)
    basis = optimizer.state[block.w1.weight]["frame"]
    expected = frame.aim(x.mT @ (delta.norm(dim=1, keepdim=True).square() * (x @ basis)), basis)
    torch.testing.assert_close(optimizer._pending[block.w1.weight].tangent, expected)
    # Output side: delta^T delta Q, with delta the gradient at w2's output.
    delta = target.reshape(-1, IN)
    basis = optimizer.state[block.w2.weight]["frame"]
    expected = frame.aim(delta.mT @ (delta @ basis), basis)
    torch.testing.assert_close(optimizer._pending[block.w2.weight].tangent, expected)


def test_micro_batches_accumulate_to_the_whole_batch():
    first, second = _batch(1), _batch(2)
    _block, split = _projected_step_two([first, second])
    whole = (torch.cat([first[0], second[0]]), torch.cat([first[1], second[1]]))
    _block, joined = _projected_step_two([whole])
    for param, prepared in split._pending.items():
        other = joined._pending[next(p for p in joined._pending if p.shape == param.shape)]
        torch.testing.assert_close(prepared.projected, other.projected)
        torch.testing.assert_close(prepared.tangent, other.tangent)
        torch.testing.assert_close(prepared.raw_norm, other.raw_norm)


def test_training_runs_with_projected_grads():
    block, optimizer = _setup()
    projected = ProjectedGrads(block, optimizer)
    before = block.w1.weight.detach().clone()
    for step in range(5):
        _loss(block, _batch(step)).backward()
        projected.prepare()
        optimizer.step()
    assert not torch.equal(block.w1.weight, before)
    assert all(torch.isfinite(p).all() for p in block.parameters())
