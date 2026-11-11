"""The GPU tier's model flows (G14, the owner's rule: no unit tests on GPUs; row 5461's covering set): each
model family that takes a distinct default route on a GPU, its training step and decode loop on par with
the reference path's against float64 truths (tests/model_flows.py).

Each flow runs twice in the tier: in its part's prepare (`DEW_TIER_PREPARE=1`, on the node's CPU), which
computes its truths and compiles its programs for the node's GPU in a process of the flow's own, started
under its XLA flags (`model_flows.prepared_apart`), and in its part's run, which loads them and judges; a
flow whose prepare did not write them is red, naming what is missing."""

import model_flows
import pytest

from dew.nn.kernels import generation

pytestmark = [pytest.mark.gpu(reason="each family's own GPU kernels, from programs compiled for this GPU"),
              pytest.mark.prepares]


@pytest.mark.parametrize("name", [flow.name for flow in model_flows.FLOWS])
def test_a_model_trains_and_decodes_on_its_gpu_routes(name, monkeypatch, record_property):
    """The flow `name` (`model_flows.FLOWS`): prepared where this process prepares, else run, its device's
    peak bytes reported (`peak_bytes`)."""
    flow = model_flows.flow_named(name)
    if model_flows.preparing():
        if not model_flows.prepared_apart(flow):
            pytest.skip(f"{flow.name}: the node's GPU runs its routes on the portable math (`Flow.families`)")
        return
    if not flow.runs_on(generation.device_generation()):
        pytest.skip(f"{flow.name}: this GPU runs its routes on the portable math (`Flow.families`)")
    # Its programs, compiled ahead, compile nothing here: the flags a load holds them to (`BuildTarget.held`).
    monkeypatch.setenv("XLA_FLAGS", model_flows.flow_flags(flow))
    record_property("peak_bytes", model_flows.run(flow))
