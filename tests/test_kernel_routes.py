"""tools/kernel_routes.py: every registered decoder family's CPU trace on each emulated device
(`kernel_routes.DEVICES`) reports the route each choice takes there, and each route is the device's kernel
or a refusal a row or an inherent limit explains."""

import json
import zlib

import pytest

from tools import kernel_routes

PARTS = 8
"""How many files hold the census (tests/test_kernel_routes_<part>.py), so CI's tasks split it by family
from its first run, before any task has timed them: a family's cells run in one process, which traces its
model once for every device and mode (`kernel_routes.cells`)."""


def part(family: str) -> int:
    """The census file a family's cells run in: a hash of its name, stable as families come and go."""
    return zlib.crc32(family.encode()) % PARTS


def census(index: int):
    """The census of the families of part `index` (`part`) as a test, every device and mode of each."""

    @pytest.mark.parametrize("mode", kernel_routes.MODES)
    @pytest.mark.parametrize("target", kernel_routes.audited())
    @pytest.mark.parametrize("family", [name for name in kernel_routes.families() if part(name) == index])
    def test(family, target, mode):
        """The family's tiniest committed fixture, traced for one device and mode
        (`kernel_routes.cells`): each choice takes the target's preferred kernel, or leaves it by a
        refusal its kernel declares, a gap with its row or inherent; a mixer refusing the mode's cache
        layout names its ledger row (`LAYOUT_GAPS`). A bidirectional family prefills through its causal
        view and decodes a canvas over that prefix. Any other route, refusal or a family with no fixture
        fails, each named."""
        (cell,) = kernel_routes.cells([family], [target], [mode])
        assert cell["fixture"] is not None, cell.get("error")
        unexplained = kernel_routes.unexplained(cell)
        assert not unexplained, "\n".join(unexplained)

    return test


def test_every_part_of_the_census_has_its_file():
    """Each part's census runs from a file of its own, so every family's cells run (`census`)."""
    from pathlib import Path

    files = {path.name for path in Path(__file__).parent.glob("test_kernel_routes_*.py")}
    assert files == {f"test_kernel_routes_{index}.py" for index in range(PARTS)}


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
