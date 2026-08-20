import copy

import pytest
import torch

from usuitrack import UsuiTrack


def test_rejects_non_matrix_and_excessive_rank():
    with pytest.raises(ValueError, match="only supports 2D"):
        UsuiTrack([torch.nn.Parameter(torch.randn(4))])
    with pytest.raises(ValueError, match="exceeds"):
        UsuiTrack([torch.nn.Parameter(torch.randn(3, 4))], rank=4)


def test_basis_moves_every_step_by_default():
    weight = torch.nn.Parameter(torch.randn(8, 4))
    optimizer = UsuiTrack([weight], lr=0.01, rank=3, side="right")

    weight.grad = torch.randn_like(weight)
    optimizer.step()
    initial_basis = optimizer.state[weight]["basis"].clone()

    weight.grad = torch.randn_like(weight)
    optimizer.step()
    assert not torch.equal(initial_basis, optimizer.state[weight]["basis"])
    assert optimizer.param_groups[0]["basis_update_step"] == 2


def test_basis_update_interval_gates_geodesic_not_moment():
    weight = torch.nn.Parameter(torch.randn(8, 4))
    optimizer = UsuiTrack([weight], lr=0.01, rank=3, side="right", basis_update_interval=2)

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
    ordinary_weight = torch.nn.Parameter(torch.randn(8, 4))
    released_weight = torch.nn.Parameter(ordinary_weight.detach().clone())
    ordinary = UsuiTrack([ordinary_weight], lr=0.01, rank=3, side="right")
    released = UsuiTrack([released_weight], lr=0.01, rank=3, side="right", release_matrix_grads=True)

    for _ in range(3):
        gradient = torch.randn_like(ordinary_weight)
        ordinary_weight.grad = gradient.clone()
        released_weight.grad = gradient.clone()
        released.prepare(released_weight)
        assert released_weight.grad is None
        ordinary.step()
        released.step()
        torch.testing.assert_close(ordinary_weight, released_weight, rtol=0, atol=0)


def test_prepare_is_exactly_once_and_zero_grad_cannot_discard_pending_work():
    weight = torch.nn.Parameter(torch.randn(8, 4))
    optimizer = UsuiTrack([weight], rank=3)
    weight.grad = torch.randn_like(weight)
    optimizer.prepare(weight)
    with pytest.raises(RuntimeError, match="already prepared"):
        optimizer.prepare(weight)
    with pytest.raises(RuntimeError, match="cannot discard"):
        optimizer.zero_grad()
    optimizer.step()


def test_state_dict_resume_matches_uninterrupted_run():
    torch.manual_seed(1)
    first = torch.nn.Parameter(torch.randn(8, 4))
    first_optimizer = UsuiTrack([first], lr=0.01, rank=3)
    for _ in range(2):
        first.grad = torch.randn_like(first)
        first_optimizer.step()

    second = torch.nn.Parameter(first.detach().clone())
    second_optimizer = UsuiTrack([second], lr=0.01, rank=3)
    second_optimizer.load_state_dict(copy.deepcopy(first_optimizer.state_dict()))

    gradient = torch.randn_like(first)
    first.grad = gradient.clone()
    second.grad = gradient.clone()
    first_optimizer.step()
    second_optimizer.step()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
