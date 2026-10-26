"""tools/kernel_routes.py: every registered decoder family's CPU trace on each emulated device
(`kernel_routes.audited()`) reports the route each choice takes there, and each route is the device's kernel
or a refusal a row or an inherent limit explains."""

import json

import pytest

from tools import kernel_routes


@pytest.mark.parametrize("mode", kernel_routes.MODES)
@pytest.mark.parametrize("target", kernel_routes.audited())
@pytest.mark.parametrize("family", kernel_routes.families())
def test_every_route_of_a_family_is_its_targets_kernel_or_explained(family, target, mode):
    """The family's tiniest committed fixture, traced for one device and mode (`kernel_routes.cells`):
    each choice takes the target's preferred kernel, or leaves it by a refusal its kernel declares, a
    gap with its row or inherent; a mixer refusing the mode's cache layout names its ledger row
    (`LAYOUT_GAPS`). A bidirectional family prefills through its causal view and decodes a canvas over
    that prefix. Any other route, refusal or a family with no fixture fails, each named."""
    (cell,) = kernel_routes.cells([family], [target], [mode])
    assert cell["fixture"] is not None, cell.get("error")
    unexplained = kernel_routes.unexplained(cell)
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


def test_a_cell_whose_trace_took_no_kernel_choice_is_unexplained():
    """A family's trace that reports no route at all (every choice deleted or bypassed) fails, named, as
    any unexplained route does; a mode the family's layers rule out (`absent`) and a refusal its row names
    stay explained."""
    assert kernel_routes.unexplained({"family": "llama", "target": "sm80", "mode": "train", "routes": []})
    assert not kernel_routes.unexplained({"family": "llama", "target": "sm80", "mode": "decode",
                                          "absent": "its layers take no cache"})
