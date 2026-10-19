"""tools/kernel_routes.py: a family's CPU trace within a GPU target reports the route each choice takes there."""

import json

from dew.nn.kernels.generation import TARGETS
from tools import kernel_routes


def test_a_family_traced_for_a_gpu_target_reports_each_choice_as_that_target_routes_it():
    """Every audited row is a GPU target of the table, and llama's training step traced for sm80 reports its
    attention as sm80 routes it: on the target's preferred kernel, or off it with the refusal's code judged."""
    assert set(kernel_routes.AUDITED) <= set(TARGETS)
    config = json.loads((kernel_routes.fixtures()["llama"] / "config.json").read_text())
    attention = [route for route in kernel_routes.traced(config, "sm80", "train") if route.choice == "attention"]
    assert attention
    for route in attention:
        assert route.target == "sm80"
        judged = kernel_routes.verdict(route)
        assert (judged == "kernel") == (route.kernel == route.preferred)
        assert judged == "kernel" or route.refusal is not None
