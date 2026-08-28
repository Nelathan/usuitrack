import torch

from usuitrack import ProjectionSide, RoutingPolicy, route_parameter, route_parameters

# A policy shaped like a real one: the two semantic exclusions plus a
# residual-stream side rule.
POLICY = RoutingPolicy(
    exclude=("embed.weight", "pos_emb", "norm1.linear", "norm_out.linear"),
    track_right=("to_q", "to_k", "to_v", "net.0.proj."),
    track_left=("to_out.", "net.2."),
)


def _named(*names):
    return [(name, torch.zeros(8, 4)) for name in names]


def test_side_hints_resolve_and_unhinted_matrices_fall_through_to_auto():
    assert route_parameter("blocks.0.attn.to_q.weight", torch.zeros(8, 4), POLICY) is ProjectionSide.RIGHT
    assert route_parameter("blocks.0.attn.to_out.0.weight", torch.zeros(8, 4), POLICY) is ProjectionSide.LEFT
    assert route_parameter("blocks.0.something.weight", torch.zeros(8, 4), POLICY) is ProjectionSide.AUTO


def test_non_matrix_and_excluded_parameters_go_to_the_fallback():
    assert route_parameter("blocks.0.norm.weight", torch.zeros(8), POLICY) is None
    assert route_parameter("text_conditioner.embed.weight", torch.zeros(32128, 1024), POLICY) is None
    # Excluded despite being a perfectly ordinary matrix shape: an AdaLN gate
    # is a semantic exclusion, not a numerical one.
    assert route_parameter("blocks.13.norm1.linear.weight", torch.zeros(6144, 256), POLICY) is None


def test_exclude_wins_over_a_side_hint():
    policy = RoutingPolicy(exclude=("to_q",), track_right=("to_q",))
    assert route_parameter("attn.to_q.weight", torch.zeros(8, 4), policy) is None


def test_empty_policy_is_pure_shape_routing():
    routing = route_parameters(
        [("a.weight", torch.zeros(8, 4)), ("a.bias", torch.zeros(8))], RoutingPolicy()
    )
    assert routing.matrix == {ProjectionSide.AUTO: [("a.weight", routing.matrix[ProjectionSide.AUTO][0][1])]}
    assert [name for name, _ in routing.fallback] == ["a.bias"]


def test_route_parameters_groups_by_side_and_describes_itself():
    routing = route_parameters(
        _named("attn.to_q.weight", "attn.to_k.weight", "attn.to_out.0.weight", "mlp.fc.weight")
        + [("norm.weight", torch.zeros(8)), ("embed.weight", torch.zeros(64, 8))],
        POLICY,
    )
    assert [name for name, _ in routing.matrix[ProjectionSide.RIGHT]] == ["attn.to_q.weight", "attn.to_k.weight"]
    assert [name for name, _ in routing.matrix[ProjectionSide.LEFT]] == ["attn.to_out.0.weight"]
    assert [name for name, _ in routing.matrix[ProjectionSide.AUTO]] == ["mlp.fc.weight"]
    assert [name for name, _ in routing.fallback] == ["norm.weight", "embed.weight"]
    assert routing.describe() == "UsuiTrack routing: auto=1, left=1, right=2, fallback=2"
