"""Weight gradients without `G` (SPEC, "Molegrad")."""

import copy
import math

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from usuitrack import Molegrad, UsuiTrack, frame
from usuitrack.molegrad import MoleLinear

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
    molegrad = Molegrad(block, optimizer)
    _loss(block, _batch(0)).backward()
    optimizer.step()
    for batch in batches:
        _loss(block, batch).backward()
    molegrad.prepare()
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
    molegrad = Molegrad(block, optimizer)
    assert isinstance(block.w1, MoleLinear) and isinstance(block.w2, MoleLinear)
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
    molegrad = Molegrad(block, optimizer)
    from_first = _gradients(block, first)
    from_second = _gradients(block, second)
    for batch in (first, second):
        _loss(block, batch).backward()
    molegrad.prepare()
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


def _train(block, optimizer, molegrad, steps):
    for step in steps:
        _loss(block, _batch(step)).backward()
        optimizer.step()


def test_a_resumed_run_continues_the_run_it_was_saved_from():
    block, optimizer = _setup()
    molegrad = Molegrad(block, optimizer)
    _train(block, optimizer, molegrad, range(3))
    weights, state = copy.deepcopy(block.state_dict()), copy.deepcopy(optimizer.state_dict())
    _train(block, optimizer, molegrad, range(3, 5))
    # Wrapped before the load, as a trainer resumes, and after it.
    for wrap_first in (True, False):
        resumed, resumed_optimizer = _setup(seed=1)
        resumed.load_state_dict(weights)
        if wrap_first:
            resumed_molegrad = Molegrad(resumed, resumed_optimizer)
        # A load shares tensors that already match, and steps turn frames in place.
        resumed_optimizer.load_state_dict(copy.deepcopy(state))
        if not wrap_first:
            resumed_molegrad = Molegrad(resumed, resumed_optimizer)
        _train(resumed, resumed_optimizer, resumed_molegrad, range(3, 5))
        for module, resumed_module in ((block.w1, resumed.w1), (block.w2, resumed.w2)):
            torch.testing.assert_close(resumed_optimizer.state[resumed_module.weight]["frame"], optimizer.state[module.weight]["frame"])
            torch.testing.assert_close(resumed_module.weight, module.weight)


def test_a_bias_gets_its_ordinary_gradient():
    torch.manual_seed(0)
    layer = torch.nn.Linear(IN, HIDDEN)
    twin = torch.nn.Linear(IN, HIDDEN)
    twin.load_state_dict(layer.state_dict())
    optimizer = UsuiTrack([layer.weight], lr=1e-3, rank_fraction=8 / math.sqrt(IN * HIDDEN))
    molegrad = Molegrad(layer, optimizer)
    inputs, _target = _batch(1)
    target = torch.randn(BATCH, TOKENS, HIDDEN)
    for module in (layer, twin):
        (module(inputs) * target).sum().backward()
    molegrad.prepare()
    torch.testing.assert_close(layer.bias.grad, twin.bias.grad)
    basis = optimizer.state[layer.weight]["frame"]
    torch.testing.assert_close(optimizer._pending[layer.weight].projected, frame.orient(twin.weight.grad, "right") @ basis)


def test_what_it_cannot_project_is_refused():
    class Custom(torch.nn.Linear):
        pass

    tied = torch.nn.Sequential(torch.nn.Linear(IN, IN, bias=False), torch.nn.Linear(IN, IN, bias=False))
    tied[1].weight = tied[0].weight
    tied_to_a_table = torch.nn.Sequential(torch.nn.Embedding(HIDDEN, IN), torch.nn.Linear(IN, HIDDEN, bias=False))
    tied_to_a_table[1].weight = tied_to_a_table[0].weight
    cases = {
        "own forward": torch.nn.Sequential(Custom(IN, HIDDEN, bias=False)),
        "shared": tied,
        "shared ": tied_to_a_table,
        "not Linears": torch.nn.Sequential(torch.nn.Embedding(HIDDEN, IN)),
    }
    for match, model in cases.items():
        optimizer = UsuiTrack(model.parameters(), rank_fraction=0.1)
        with pytest.raises(ValueError, match=match.strip()):
            Molegrad(model, optimizer)


def test_a_closure_is_refused_and_none_is_no_closure():
    block, optimizer = _setup()
    Molegrad(block, optimizer)
    with pytest.raises(ValueError, match="closure"):
        optimizer.step(lambda: _loss(block, _batch(0)).backward())
    for step in (lambda: optimizer.step(None), lambda: optimizer.step(closure=None)):
        _loss(block, _batch(0)).backward()
        step()


def test_zero_grad_discards_an_unconsumed_backward():
    """A backward dropped by `zero_grad()` (an OOM retry, a skipped batch)
    leaves no trace in the next step."""

    runs = []
    for discard in (True, False):
        block, optimizer = _setup()
        Molegrad(block, optimizer)
        _loss(block, _batch(0)).backward()
        optimizer.step()
        if discard:
            _loss(block, _batch(1)).backward()
            with pytest.warns(UserWarning, match="dropped"):
                optimizer.zero_grad()
        _loss(block, _batch(2)).backward()
        optimizer.step()
        runs.append(block)
    torch.testing.assert_close(runs[0].state_dict(), runs[1].state_dict())


def test_a_linear_called_twice_before_its_fit_is_refused():
    layer = torch.nn.Linear(IN, IN, bias=False)
    Molegrad(layer, UsuiTrack([layer.weight], rank_fraction=0.1))
    with pytest.raises(RuntimeError, match="more than one call"):
        layer(layer(_batch(0)[0])).sum().backward()


def test_a_non_finite_first_gradient_fits_a_finite_frame():
    """One NaN row of `w2`'s gradient is sanitized and fitted; the NaN token
    poisons every entry of `w1`'s, which sanitizes to zero and fits nothing."""

    block, optimizer = _setup()
    Molegrad(block, optimizer)
    inputs, target = _batch(0)
    target[0, 0, 0] = torch.nan
    (block(inputs) * target).sum().backward()
    optimizer.step()
    assert torch.isfinite(optimizer.state[block.w2.weight]["frame"]).all()
    assert "frame" not in optimizer.state[block.w1.weight]
    assert all(torch.isfinite(module.weight).all() for module in (block.w1, block.w2))


def test_matrices_of_two_dtypes_train_together():
    # fp32 first: the bf16 matrix must not inherit its probes.
    model = torch.nn.ModuleDict({"fp32": torch.nn.Linear(IN, HIDDEN, bias=False), "bf16": torch.nn.Linear(IN, HIDDEN, bias=False).bfloat16()})
    optimizer = UsuiTrack(model.parameters(), rank_fraction=0.1)
    Molegrad(model, optimizer)
    inputs = _batch(0)[0]
    (model["bf16"](inputs.bfloat16()).float().sum() + model["fp32"](inputs).sum()).backward()
    optimizer.step()


def test_training_runs_with_molegrad():
    block, optimizer = _setup()
    molegrad = Molegrad(block, optimizer)
    before = block.w1.weight.detach().clone()
    for step in range(5):
        for micro in range(2):
            _loss(block, _batch(2 * step + micro)).backward()
        optimizer.step()
    assert not torch.equal(block.w1.weight, before)
    assert all(torch.isfinite(p).all() for p in block.parameters())


def _zero_target_batch(seed):
    """A batch whose loss gradient is zero: every weight gradient is exactly zero."""

    inputs, target = _batch(seed)
    return inputs, torch.zeros_like(target)


def test_a_zero_first_gradient_fits_no_frame_and_the_first_nonzero_one_fits_as_a_first_step():
    block, optimizer = _setup()
    Molegrad(block, optimizer)
    before = copy.deepcopy(block.state_dict())
    _loss(block, _zero_target_batch(0)).backward()
    optimizer.step()
    assert all(not optimizer.state[module.weight] for module in (block.w1, block.w2))
    assert all(module.basis.requires_grad for module in (block.w1, block.w2))
    torch.testing.assert_close(block.state_dict(), before, rtol=0, atol=0)

    twin, twin_optimizer = _setup()
    Molegrad(twin, twin_optimizer)
    for model, opt in ((block, optimizer), (twin, twin_optimizer)):
        _loss(model, _batch(1)).backward()
        opt.step()
    for module, twin_module in ((block.w1, twin.w1), (block.w2, twin.w2)):
        for key in ("frame", "moment"):
            assert torch.equal(optimizer.state[module.weight][key], twin_optimizer.state[twin_module.weight][key])
        assert torch.equal(module.weight, twin_module.weight)


def test_nothing_is_handed_to_the_optimizer_for_an_unfitted_matrix_without_gradient():
    block, optimizer = _setup()
    molegrad = Molegrad(block, optimizer)
    _loss(block, _zero_target_batch(0)).backward()
    molegrad.prepare()
    assert not optimizer._pending


def test_under_accumulation_a_later_micro_batch_fits_when_the_first_is_zero():
    """The fit comes from the first nonzero micro-batch; the zero one adds
    nothing to `Z`, the action or the probes, and the fitting step has no tangent."""

    first, second = _zero_target_batch(1), _batch(2)
    block, optimizer = _setup()
    molegrad = Molegrad(block, optimizer)
    from_second = _gradients(block, second)
    for batch in (first, second):
        _loss(block, batch).backward()
    molegrad.prepare()
    for module, grad in zip((block.w1, block.w2), from_second, strict=True):
        basis = optimizer.state[module.weight]["frame"]
        torch.testing.assert_close(_span(basis), _span(frame.fit(grad, basis.shape[-1])))
        prepared = optimizer._pending[module.weight]
        torch.testing.assert_close(prepared.projected, grad @ basis)
        assert prepared.tangent is None
        assert 0.5 < (prepared.raw_norm / grad.norm()).item() < 1.5

    # And the next step is an ordinary one, with a tangent.
    optimizer.step()
    _loss(block, _batch(3)).backward()
    molegrad.prepare()
    assert all(optimizer._pending[module.weight].tangent is not None for module in (block.w1, block.w2))


def test_under_accumulation_a_zero_micro_batch_after_the_fit_adds_nothing():
    first, second = _batch(1), _zero_target_batch(2)
    block, optimizer = _setup()
    molegrad = Molegrad(block, optimizer)
    from_first = _gradients(block, first)
    for batch in (first, second):
        _loss(block, batch).backward()
    molegrad.prepare()
    for module, grad in zip((block.w1, block.w2), from_first, strict=True):
        basis = optimizer.state[module.weight]["frame"]
        torch.testing.assert_close(_span(basis), _span(frame.fit(grad, basis.shape[-1])))
        prepared = optimizer._pending[module.weight]
        torch.testing.assert_close(prepared.projected, grad @ basis)
        assert prepared.tangent is None


def test_a_load_while_a_matrix_waits_leaves_it_waiting_for_its_first_nonzero_gradient():
    runs = []
    for load in (True, False):
        block, optimizer = _setup()
        Molegrad(block, optimizer)
        _loss(block, _zero_target_batch(0)).backward()
        optimizer.step()
        if load:
            optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        _loss(block, _batch(1)).backward()
        optimizer.step()
        runs.append((block, optimizer))
    (loaded, loaded_optimizer), (plain, plain_optimizer) = runs
    for module, plain_module in ((loaded.w1, plain.w1), (loaded.w2, plain.w2)):
        assert torch.equal(loaded_optimizer.state[module.weight]["frame"], plain_optimizer.state[plain_module.weight]["frame"])
        assert torch.equal(module.weight, plain_module.weight)


def test_a_zero_initialized_output_trains_and_the_matrices_upstream_fit_from_a_real_gradient():
    """The adaLN-zero setting: with the output Linear at zero, everything
    upstream has a zero gradient on step one and must not fit a frame from it."""

    def network():
        return torch.nn.Sequential(torch.nn.Linear(IN, HIDDEN, bias=False), torch.nn.GELU(), torch.nn.Linear(HIDDEN, IN, bias=False))

    def backward(net, step):
        inputs, target = _batch(step)
        (net(inputs) * target).sum().backward()

    torch.manual_seed(0)
    model = network()
    torch.nn.init.zeros_(model[2].weight)
    first, last = model[0].weight, model[2].weight
    optimizer = UsuiTrack([{"params": [first]}, {"params": [last], "step_rule": "fixed"}], lr=1e-2, rank_fraction=8 / math.sqrt(IN * HIDDEN))

    def twin_gradient(step):
        twin = network()
        twin.load_state_dict(model.state_dict())
        backward(twin, step)
        return twin[0].weight.grad

    assert twin_gradient(0).abs().max() == 0
    Molegrad(model, optimizer)
    backward(model, 0)
    optimizer.step()
    assert not optimizer.state[first]
    assert last.abs().max() > 0

    step_two = twin_gradient(1)
    assert step_two.abs().max() > 0
    backward(model, 1)
    optimizer.step()
    basis = optimizer.state[first]["frame"]
    torch.testing.assert_close(_span(basis), _span(frame.fit(step_two, basis.shape[-1])))
