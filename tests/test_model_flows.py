"""The GPU tier's model flows as a CPU reads them (tests/model_flows.py, row 5461): each flow's model
builds at its widths, its Dew arm takes the routes it is for as each GPU generation's target chooses them,
its reference arm the reference path alone, and its truth computes in float64; and check 1's prepared
truths (`reference_error.stored_truths`) judge as the truths a site computes itself, a missing or stale
one refused."""

import jax
import jax.numpy as jnp
import model_flows
import numpy as np
import pytest
from reference_error import MissingTruth, assert_as_exact, stored_truths

from dew.nn.kernels import generation


def _routes(flow, program, *, reference: bool = False) -> list:
    """The routes `program` (of the flow's `programs_of`) takes traced for this process's assumed target."""
    flat, shapes = program
    with generation.routes() as found:
        if reference:
            with generation.chosen(**model_flows.REFERENCE):
                jax.jit(flat).trace(*shapes)
        else:
            jax.jit(flat).trace(*shapes)
    return found


@pytest.mark.parametrize("target", ["sm80", "sm90"])
@pytest.mark.parametrize("name", [flow.name for flow in model_flows.FLOWS])
def test_a_flows_model_takes_the_routes_it_is_for_on_each_gpu(name, target, monkeypatch):
    """The flow's training step and decode loop traced as `target`'s GPU chooses (`generation.assumed`):
    every route the flow is for taken off the reference path, a library's kernel (FlashAttention) chosen
    there though a CPU lowers none (its route's `preferred`; the node's prepare holds the kernel itself,
    `model_flows.held_routes`); its reference arm on the reference path alone."""
    flow = model_flows.flow_named(name)
    monkeypatch.setenv("XLA_FLAGS", model_flows.flow_flags(flow))
    programs = model_flows.programs_of(flow)
    with generation.assumed(target):
        found = [route for kind in ("train", "decode") for route in _routes(flow, programs[f"{kind} dew"])]
        reference = [route for kind in ("train", "decode")
                     for route in _routes(flow, programs[f"{kind} reference"], reference=True)]
    for choice, kernel in flow.native.items():
        taken = {route.kernel if kernel == model_flows.NATIVE else route.preferred
                 for route in found if str(route.choice) == choice}
        held = taken - model_flows.PORTABLE if kernel == model_flows.NATIVE else taken & {kernel}
        assert held, f"{name}: its {choice} took {sorted(taken)} on {target}, not {kernel}"
    assert not {route.kernel for route in reference} - model_flows.PORTABLE, sorted(
        {(str(route.choice), route.kernel) for route in reference})


@pytest.mark.parametrize("name", [flow.name for flow in model_flows.FLOWS])
def test_a_flows_truth_computes_in_float64(name, tmp_path, monkeypatch):
    """The flow's truths prepared on a CPU as its node prepares them, its first draw's of each program held to
    float64 (check 1's `_of_float64`), every draw its two stages may take written to the store."""
    flow = model_flows.flow_named(name)
    monkeypatch.setenv(model_flows.TRUTHS, str(tmp_path))
    monkeypatch.setenv(model_flows.PREPARE, "1")
    model_flows.judge(flow, {})
    written = sorted((model_flows.store_of(flow) / "truths").iterdir())
    assert len(written) == 2


def _site():
    """A bf16 product against its fp32 reference, by seed: a check-1 site small enough to prepare, then
    judge."""
    def draw(seed):
        rng = np.random.default_rng([5461, seed])
        return (rng.standard_normal((16, 32)).astype(np.float32),
                rng.standard_normal((32, 8)).astype(np.float32))

    def dew(a, b):
        return jnp.dot(a.astype(jnp.bfloat16), b.astype(jnp.bfloat16), preferred_element_type=jnp.float32)

    def reference(a, b):
        narrow = (operand.astype(jnp.bfloat16).astype(jnp.float32) for operand in (a, b))
        return jnp.dot(*narrow)

    return dew, reference, draw


def test_a_prepared_site_judges_as_one_that_computes_its_truth_and_refuses_a_missing_or_stale_one(tmp_path):
    """A site prepared (`stored_truths(prepare=True)`) writes every draw's truth and runs no arm; judged from
    that store it passes as it passes computing its own; a store of another stamp, or none, is a red
    (`MissingTruth`), never a truth computed in its place."""
    dew, reference, draw = _site()
    with stored_truths(tmp_path, prepare=True, stamp="a"):
        assert_as_exact(None, reference, draw, "a prepared product", seeds=8)
    assert len(list(tmp_path.iterdir())) == 1
    assert_as_exact(dew, reference, draw, "a prepared product", seeds=8)
    with stored_truths(tmp_path, prepare=False, stamp="a"):
        assert_as_exact(dew, reference, draw, "a prepared product", seeds=8)
    with stored_truths(tmp_path, prepare=False, stamp="b"), pytest.raises(MissingTruth):
        assert_as_exact(dew, reference, draw, "a prepared product", seeds=8)
    with stored_truths(tmp_path, prepare=False, stamp="a"), pytest.raises(MissingTruth):
        assert_as_exact(dew, reference, draw, "another product", seeds=8)
