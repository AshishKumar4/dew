"""tools/kernel_routes.py: every registered decoder family's CPU trace within each row of TARGETS that
names a kernel reports the route each choice takes there, and each route is the target's kernel or a
refusal a row or an inherent limit explains."""

import json

import pytest

from tools import kernel_routes


@pytest.mark.parametrize("family", kernel_routes.families())
def test_every_route_of_a_family_is_its_targets_kernel_or_explained(family):
    """The family's committed fixture, traced for every audited target and mode (`kernel_routes.cells`):
    each choice takes the target's preferred kernel, or leaves it by a refusal its kernel declares, a
    gap with its row or inherent; a mixer refusing the mode's cache layout names its ledger row
    (`LAYOUT_GAPS`); a bidirectional family has no causal mode. Any other route, refusal or a family
    with no fixture fails, each named."""
    cells = list(kernel_routes.cells([family]))
    assert all(cell["fixture"] is not None for cell in cells), cells[0].get("error")
    assert sorted((cell["target"], cell["mode"]) for cell in cells) == sorted(
        (target, mode) for target in kernel_routes.audited() for mode in kernel_routes.MODES)
    unexplained = [f"{cell['target']} {cell['mode']}: {why}" for cell in cells
                   for why in kernel_routes.unexplained(cell)]
    assert not unexplained, "\n".join(unexplained)


def test_a_family_traced_for_a_gpu_target_reports_each_choice_as_that_target_routes_it():
    """llama's training step traced for sm80 reports its attention as sm80 routes it: on the target's
    preferred kernel, or off it with the refusal's code judged."""
    config = json.loads((kernel_routes.fixtures()["llama"] / "config.json").read_text())
    traced = kernel_routes.traced(config, "sm80", "train")
    attention = [route for route in traced if route.choice == "attention"]
    assert attention
    for route in attention:
        assert route.target == "sm80"
        judged = kernel_routes.verdict(route)
        assert (judged == "kernel") == (route.kernel == route.preferred)
        assert judged == "kernel" or route.refusal is not None


def test_a_decode_steps_routes_are_its_own_not_the_parameters_or_the_caches():
    """llama's single-token decode step traced for sm80 reports the routes of that step alone: one
    attention decision per layer, its single query's, where the parameters' and the cache's own
    forward passes over the prompt used to add theirs to the cell."""
    config = json.loads((kernel_routes.fixtures()["llama"] / "config.json").read_text())
    attention = [route for route in kernel_routes.traced(config, "sm80", "decode")
                 if route.choice in ("attention", "decode")]
    assert len(attention) == config["num_hidden_layers"]
