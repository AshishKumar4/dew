"""The GPU tier's model flows (G14, the owner's rule: no unit tests on GPUs; row 5461's covering set): each
model family that takes a distinct default route on a GPU, its training step and decode loop on par with
the reference path's against float64 truths (tests/model_flows.py).

Each flow runs twice in the tier: in its part's prepare (`DEW_TIER_PREPARE=1`, on the node's CPU), which
computes its truths and compiles its programs for the node's GPU in a process of the flow's own, started
under its XLA flags (`model_flows.prepared_apart`), and in its part's run, which loads them and judges; a
flow whose prepare did not write them is red, naming what is missing. A variant is a process: XLA parses its
flags once a process, so the tier prepares and runs this file once in the lane's process and once in a
user's (`lane_environment.USERS_ARITHMETIC`), each flow in the process of its variant alone."""

import model_flows
import pytest

from dew.nn.kernels import generation

pytestmark = [pytest.mark.gpu(reason="each family's own GPU kernels, from programs compiled for this GPU"),
              pytest.mark.prepares]


@pytest.mark.parametrize("name", [
    pytest.param(flow.name, marks=() if flow.lane else pytest.mark.users_process)
    for flow in model_flows.FLOWS])
def test_a_model_trains_and_decodes_on_its_gpu_routes(name, monkeypatch, record_property):
    """The flow `name` (`model_flows.FLOWS`) in a process of its variant (a user's flow marked
    `users_process`, which runs where the process computes as a user's does alone, tests/conftest.py):
    prepared where this process prepares, what its prepare took reported (`prepare_seconds`,
    `prepare_peak_bytes`, `prepare_written_bytes`), else run, its process held to its flags and precision
    first (`model_flows.held_process`), its device's peak bytes reported (`peak_bytes`)."""
    flow = model_flows.flow_named(name)
    if model_flows.preparing():
        if (took := model_flows.prepared_apart(flow)) is None:
            pytest.skip(f"{flow.name}: the node's GPU runs its routes on the portable math (`Flow.families`)")
        for name, value in took.items():
            record_property(f"prepare_{name}", value)
        return
    if not flow.runs_on(generation.device_generation()):
        pytest.skip(f"{flow.name}: this GPU runs its routes on the portable math (`Flow.families`)")
    model_flows.held_process(flow)
    # The flags its programs were compiled under, which a load holds them to (`BuildTarget.held`): the
    # process's own but the host backend's, which compute nothing of the GPU's programs.
    monkeypatch.setenv("XLA_FLAGS", model_flows.flow_flags(flow))
    record_property("peak_bytes", model_flows.run(flow))
