"""Weight gradients without `G` (SPEC, "Projected gradients")."""

import copy
import math

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from usuitrack import ProjectedGrads, UsuiTrack, frame
from usuitrack.projected import ProjectedLinear

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
    """Step one, which fits the frames; then the leaves of step two's backwards."""

    block, optimizer = _setup()
    projected = ProjectedGrads(block, optimizer)
    _loss(block, _batch(0)).backward()
    optimizer.step()
    for batch in batches:
        _loss(block, batch).backward()
    projected.prepare()
    return block, optimizer


def _gradients(block, batch):
    """The ordinary weight gradients of `block`'s weights on `batch`, oriented to each frame's side."""

    twin = _Block()
    twin.load_state_dict(block.state_dict())
    _loss(twin, batch).backward()
    return frame.orient(twin.w1.weight.grad, "right"), frame.orient(twin.w2.weight.grad, "left")


def _span(basis):
    return basis @ basis.mT


def test_step_one_is_the_optimizers_own_first_step():
    """At one micro-batch per step the fit comes from the step's whole `G`, and
    the weights after step one are the ordinary optimizer's."""

    block, optimizer = _setup()
    projected = ProjectedGrads(block, optimizer)
    assert isinstance(block.w1, ProjectedLinear) and isinstance(block.w2, ProjectedLinear)
    assert not block.w1.weight.requires_grad
    twin, twin_optimizer = _setup()
    _loss(block, _batch(0)).backward()
    optimizer.step()
    _loss(twin, _batch(0)).backward()
    twin_optimizer.step()
    for module, twin_module in ((block.w1, twin.w1), (block.w2, twin.w2)):
        torch.testing.assert_close(_span(optimizer.state[module.weight]["frame"]), _span(twin_optimizer.state[twin_module.weight]["frame"]))
        torch.testing.assert_close(module.weight, twin_module.weight)


def test_under_accumulation_the_first_micro_batch_fits_the_frame():
    first, second = _batch(1), _batch(2)
    block, optimizer = _setup()
    projected = ProjectedGrads(block, optimizer)
    from_first = _gradients(block, first)
    from_second = _gradients(block, second)
    for batch in (first, second):
        _loss(block, batch).backward()
    projected.prepare()
    for module, grad_first, grad_second in zip((block.w1, block.w2), from_first, from_second, strict=True):
        basis = optimizer.state[module.weight]["frame"]
        torch.testing.assert_close(_span(basis), _span(frame.fit(grad_first, basis.shape[-1])))
        prepared = optimizer._pending[module.weight]
        torch.testing.assert_close(prepared.projected, (grad_first + grad_second) @ basis)
        assert prepared.tangent is None


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


def _train(block, optimizer, projected, steps):
    for step in steps:
        _loss(block, _batch(step)).backward()
        optimizer.step()


def test_a_resumed_run_continues_the_run_it_was_saved_from():
    block, optimizer = _setup()
    projected = ProjectedGrads(block, optimizer)
    _train(block, optimizer, projected, range(3))
    weights, state = copy.deepcopy(block.state_dict()), copy.deepcopy(optimizer.state_dict())
    _train(block, optimizer, projected, range(3, 5))
    # Wrapped before the load, as a trainer resumes, and after it.
    for wrap_first in (True, False):
        resumed, resumed_optimizer = _setup(seed=1)
        resumed.load_state_dict(weights)
        if wrap_first:
            resumed_projected = ProjectedGrads(resumed, resumed_optimizer)
        # A load shares tensors that already match, and steps turn frames in place.
        resumed_optimizer.load_state_dict(copy.deepcopy(state))
        if not wrap_first:
            resumed_projected = ProjectedGrads(resumed, resumed_optimizer)
        _train(resumed, resumed_optimizer, resumed_projected, range(3, 5))
        for module, resumed_module in ((block.w1, resumed.w1), (block.w2, resumed.w2)):
            torch.testing.assert_close(resumed_optimizer.state[resumed_module.weight]["frame"], optimizer.state[module.weight]["frame"])
            torch.testing.assert_close(resumed_module.weight, module.weight)


def test_a_closure_is_refused():
    block, optimizer = _setup()
    ProjectedGrads(block, optimizer)
    with pytest.raises(ValueError, match="closure"):
        optimizer.step(lambda: _loss(block, _batch(0)).backward())


def test_training_runs_with_projected_grads():
    block, optimizer = _setup()
    projected = ProjectedGrads(block, optimizer)
    before = block.w1.weight.detach().clone()
    for step in range(5):
        for micro in range(2):
            _loss(block, _batch(2 * step + micro)).backward()
        optimizer.step()
    assert not torch.equal(block.w1.weight, before)
    assert all(torch.isfinite(p).all() for p in block.parameters())
