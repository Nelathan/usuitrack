import copy

import pytest
import torch

from usuitrack import UsuiTrack


# A representative transformer-ish weight. Small toy shapes like (8, 4) put
# effective_rank() straight into its floor -- a quarter of 4 rounds to 1 -- so
# every test silently exercised a rank-1 basis instead of the regime the
# optimizer is designed for. RANK sits below the quarter cap of COLS so the
# configured rank is the rank actually used.
ROWS, COLS, RANK = 256, 128, 16


def test_rejects_non_matrix_and_excessive_rank():
    with pytest.raises(ValueError, match="only supports 2D"):
        UsuiTrack([torch.nn.Parameter(torch.randn(4))])
    # Over-cap rank is not fatal -- effective_rank() clamps it and warns.
    with pytest.warns(UserWarning, match="exceeds"):
        UsuiTrack([torch.nn.Parameter(torch.randn(ROWS, COLS))], rank=COLS)


def test_basis_moves_every_step_by_default():
    weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    optimizer = UsuiTrack([weight], lr=0.01, rank=RANK, side="right")

    weight.grad = torch.randn_like(weight)
    optimizer.step()
    initial_basis = optimizer.state[weight]["basis"].clone()

    weight.grad = torch.randn_like(weight)
    optimizer.step()
    assert not torch.equal(initial_basis, optimizer.state[weight]["basis"])
    assert optimizer.param_groups[0]["basis_update_step"] == 2


def test_basis_update_interval_gates_geodesic_not_moment():
    weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    optimizer = UsuiTrack([weight], lr=0.01, rank=RANK, side="right", basis_update_interval=2)

    weight.grad = torch.randn_like(weight)
    optimizer.step()
    initial_basis = optimizer.state[weight]["basis"].clone()
    assert optimizer.param_groups[0]["basis_update_step"] == 0

    weight.grad = torch.randn_like(weight)
    optimizer.step()
    assert not torch.equal(initial_basis, optimizer.state[weight]["basis"])
    assert optimizer.param_groups[0]["basis_update_step"] == 1
    assert "projected_exp_avg" in optimizer.state[weight]


def test_prepare_release_matches_ordinary_step_exactly():
    """Validates the release_matrix_grads=True fast path (README's early-release
    mode) produces bit-identical parameters to the ordinary step() path."""

    torch.manual_seed(0)
    ordinary_weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    released_weight = torch.nn.Parameter(ordinary_weight.detach().clone())
    ordinary = UsuiTrack([ordinary_weight], lr=0.01, rank=RANK, side="right")
    released = UsuiTrack([released_weight], lr=0.01, rank=RANK, side="right", release_matrix_grads=True)

    for _ in range(3):
        gradient = torch.randn_like(ordinary_weight)
        ordinary_weight.grad = gradient.clone()
        released_weight.grad = gradient.clone()
        released.prepare(released_weight)
        assert released_weight.grad is None
        ordinary.step()
        released.step()
        torch.testing.assert_close(ordinary_weight, released_weight, rtol=0, atol=0)


def test_plain_gradient_accumulation_without_release_matches_manual_grad_sum():
    """UsuiTrack has no accumulation-aware bookkeeping: it just reads
    param.grad at step() time. Two backward() calls before one step() (no
    release_matrix_grads, no zero_grad in between) must therefore behave
    identically to a single step() on the pre-summed gradient."""

    torch.manual_seed(2)
    accumulated = torch.nn.Linear(COLS, ROWS, bias=False)
    presummed = torch.nn.Linear(COLS, ROWS, bias=False)
    presummed.weight.data.copy_(accumulated.weight.data)

    x1, x2 = torch.randn(8, COLS), torch.randn(8, COLS)
    accumulated(x1).sum().backward()
    accumulated(x2).sum().backward()

    grad1 = torch.autograd.grad(presummed(x1).sum(), presummed.weight)[0]
    grad2 = torch.autograd.grad(presummed(x2).sum(), presummed.weight)[0]
    presummed.weight.grad = grad1 + grad2

    torch.testing.assert_close(accumulated.weight.grad, presummed.weight.grad, rtol=0, atol=0)

    UsuiTrack([accumulated.weight], lr=0.01, rank=RANK).step()
    UsuiTrack([presummed.weight], lr=0.01, rank=RANK).step()
    torch.testing.assert_close(accumulated.weight, presummed.weight, rtol=0, atol=0)


def test_prepare_is_exactly_once():
    weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    optimizer = UsuiTrack([weight], rank=RANK)
    weight.grad = torch.randn_like(weight)
    optimizer.prepare(weight)
    with pytest.raises(RuntimeError, match="already prepared"):
        optimizer.prepare(weight)
    optimizer.step()


def test_zero_grad_drops_pending_prepared_work():
    """zero_grad() is an escape hatch for callers that must bail out (e.g. an
    OOM-retry loop) after prepare() already mutated moving-average state for
    some params. It can't undo that mutation, but it must not block recovery:
    it just drops the pending update so the next prepare()/step() starts
    clean."""

    weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    optimizer = UsuiTrack([weight], rank=RANK)
    weight.grad = torch.randn_like(weight)
    optimizer.prepare(weight)
    optimizer.zero_grad()
    assert not optimizer._pending_matrix_updates
    assert weight.grad is None


def test_state_dict_resume_matches_uninterrupted_run():
    torch.manual_seed(1)
    first = torch.nn.Parameter(torch.randn(ROWS, COLS))
    first_optimizer = UsuiTrack([first], lr=0.01, rank=RANK)
    for _ in range(2):
        first.grad = torch.randn_like(first)
        first_optimizer.step()

    second = torch.nn.Parameter(first.detach().clone())
    second_optimizer = UsuiTrack([second], lr=0.01, rank=RANK)
    second_optimizer.load_state_dict(copy.deepcopy(first_optimizer.state_dict()))

    gradient = torch.randn_like(first)
    first.grad = gradient.clone()
    second.grad = gradient.clone()
    first_optimizer.step()
    second_optimizer.step()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
