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


def _routes(program, *, reference: bool = False) -> list:
    """The routes `program` (of the flow's `programs_of`) takes traced for this process's assumed target, as
    the reference arm's where `reference` (`model_flows.reference_kernels`)."""
    flat, shapes = program
    with generation.routes() as found:
        if reference:
            with generation.chosen(**model_flows.reference_kernels()):
                jax.jit(flat).trace(*shapes)
        else:
            jax.jit(flat).trace(*shapes)
    return found


@pytest.mark.parametrize("target", ["sm80", "sm90"])
@pytest.mark.parametrize("name", [flow.name for flow in model_flows.FLOWS])
def test_a_flows_model_takes_the_routes_it_is_for_on_each_gpu(name, target, monkeypatch):
    """The flow's training step and decode loop traced as `target`'s GPU chooses (`generation.assumed`) and
    held as its prepare holds them: every route the flow is for taken off the reference path or by the
    kernel it names (`model_flows.held_routes`), its reference arm on the reference path alone
    (`model_flows.held_reference`)."""
    flow = model_flows.flow_named(name)
    monkeypatch.setenv("XLA_FLAGS", model_flows.flow_flags(flow))
    programs = model_flows.programs_of(flow)
    with generation.assumed(target):
        dew = [route for kind in ("train", "decode") for route in _routes(programs[f"{kind} dew"])]
        reference = [route for kind in ("train", "decode")
                     for route in _routes(programs[f"{kind} reference"], reference=True)]
        model_flows.held_routes(flow, dew)
        model_flows.held_reference(flow, reference)


def test_the_reference_arm_runs_each_choices_generic_kernel_that_takes_every_call():
    """The reference arm's choices (`model_flows.reference_kernels`) are every choice Dew's ops serve, each a
    generic kernel of its op that requires nothing (the reference path); attention's and the decode's are
    their plain-JAX reference, never jax.nn's xla attention, which hands sinks and softcaps on."""
    chosen = model_flows.reference_kernels()
    assert set(chosen) == set(model_flows.generic_kernels())
    assert all(kernel in model_flows.generic_kernels()[choice] for choice, kernel in chosen.items())
    assert chosen["attention"] == chosen["decode"] == "reference"


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


def _site(salt: int = 5461):
    """A bf16 product against its fp32 reference, by seed (of the stream `salt` names): a check-1 site small
    enough to prepare, then judge."""
    def draw(seed):
        rng = np.random.default_rng([salt, seed])
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
    # The same site over another seed stream: draws its prepare did not make.
    _, _, other = _site(5462)
    with stored_truths(tmp_path, prepare=False, stamp="a"), pytest.raises(MissingTruth):
        assert_as_exact(dew, reference, other, "a prepared product", seeds=8)


def test_a_prepared_site_sized_by_its_terms_reads_them_from_its_store(tmp_path):
    """A site whose truth holds rows that cancel (its second half a product less the same product summed in
    two halves, exactly zero in float64) is sized by its terms (check 1's `_cancelling_terms`): prepared, its
    store holds the terms its truth program gave beside the truth, and judged from the store it reads them
    as a site's own terms program (`_SiteTerms.of`), passing as it passes computing them."""
    def draw(seed):
        rng = np.random.default_rng([5463, seed])
        return (rng.standard_normal((16, 32)).astype(np.float32),
                rng.standard_normal((32, 8)).astype(np.float32))

    def cancelling(a, b):
        whole = a @ b
        return jnp.concatenate([whole, whole - (a[:, :16] @ b[:16] + a[:, 16:] @ b[16:])], axis=1)

    with stored_truths(tmp_path, prepare=True, stamp="a"):
        assert_as_exact(None, cancelling, draw, "a cancelling product", seeds=8)
    assert_as_exact(cancelling, cancelling, draw, "a cancelling product", seeds=8)
    with stored_truths(tmp_path, prepare=False, stamp="a") as store:
        assert callable(store.site("a cancelling product")[1])
        assert_as_exact(cancelling, cancelling, draw, "a cancelling product", seeds=8)
