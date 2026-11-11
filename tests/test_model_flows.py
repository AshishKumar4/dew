"""The GPU tier's model flows as a CPU reads them (tests/model_flows.py, row 5461): each flow's model
builds at its widths, its Dew arm takes the routes it is for as each GPU generation's target chooses them,
its reference arm the reference path alone, and its truth computes in float64; and check 1's prepared
truths (`reference_error.stored_truths`) judge as the truths a site computes itself, a missing or stale
one refused."""

import os
import sys
import tempfile
from pathlib import Path

import lane_environment
import jax
import jax.numpy as jnp
import model_flows
import numpy as np
import pytest
from process_support import relayed
from reference_error import MissingTruth, assert_as_exact, stored_truths

from dew.nn.kernels import generation
from dew.parity import GateFailure
from dew.parity_stages import staged_units


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
    kernel it names (`model_flows.held_routes`), but by the portable math where it is native on other GPUs
    alone (`Flow.families`); its reference arm on the reference path alone (`model_flows.held_reference`)."""
    flow = model_flows.flow_named(name)
    monkeypatch.setenv("XLA_FLAGS", model_flows.flow_flags(flow))
    programs = model_flows.programs_of(flow)
    with generation.assumed(target), jax.default_matmul_precision(flow.precision):
        dew = {kind: _routes(programs[f"{kind} dew"]) for kind in model_flows.KINDS}
        reference = [route for kind in model_flows.KINDS
                     for route in _routes(programs[f"{kind} reference"], reference=True)]
        model_flows.held_routes(flow, dew, target)
        model_flows.held_reference(flow, reference)


def test_a_training_step_on_the_templates_plain_jax_mainloop_does_not_reach_its_route(monkeypatch):
    """An attention flow whose training step runs the template on the plain-JAX mainloop
    (`generation.chosen`'s `attention_mainloop='portable'`, as a GPU's mainloops declining it would), its
    decode loop on the GPU's own, does not hold: each program is held to its own routes, and the template's
    every pass to a GPU's mainloop (`model_flows.EVERY`), so a training step whose one pass, its backward,
    took the plain-JAX one beside a native forward does not hold either."""
    flow = model_flows.flow_named("dense bf16, the lane's")
    monkeypatch.setenv("XLA_FLAGS", model_flows.flow_flags(flow))
    programs = model_flows.programs_of(flow)
    with generation.assumed("sm90"), jax.default_matmul_precision(flow.precision):
        native = {kind: _routes(programs[f"{kind} dew"]) for kind in model_flows.KINDS}
        with generation.chosen(attention_mainloop="portable"):
            portable = _routes(programs["train dew"])
    model_flows.held_routes(flow, native, "sm90")
    with pytest.raises(model_flows.Unreached):
        model_flows.held_routes(flow, {**native, "train": portable}, "sm90")
    one_pass = [route for route in portable if str(route.choice) == "attention_mainloop"][:1]
    assert one_pass
    with pytest.raises(model_flows.Unreached):
        model_flows.held_routes(flow, {**native, "train": [*native["train"], *one_pass]}, "sm90")


def test_the_reference_arm_runs_each_choices_generic_kernel_that_takes_every_call():
    """The reference arm's choices (`model_flows.reference_kernels`) are every choice Dew's ops serve, each a
    generic kernel of its op that requires nothing (the reference path); attention's and the decode's are
    their plain-JAX reference, never jax.nn's xla attention, which hands sinks and softcaps on."""
    chosen = model_flows.reference_kernels()
    assert set(chosen) == set(model_flows.generic_kernels())
    assert all(kernel in model_flows.generic_kernels()[choice] for choice, kernel in chosen.items())
    assert chosen["attention"] == chosen["decode"] == "reference"


@pytest.fixture
def store():
    """A tier's store for one test, removed as the test ends: a flow's store holds every draw's float64
    gradients, about 2 GB a flow, which a file's tests would otherwise all keep to its session's end
    (`tmp_path`), past a CPU task's disk (the candidate's whole CPU suite at 173b183813a)."""
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


@pytest.mark.parametrize("name", [flow.name for flow in model_flows.FLOWS])
def test_a_flows_truth_computes_in_float64(name, store, monkeypatch):
    """The flow's truths prepared on a CPU as its node prepares them, its first draw's of each program held to
    float64 (check 1's `_of_float64`), every draw its two stages may take written to the store."""
    flow = model_flows.flow_named(name)
    monkeypatch.setenv(model_flows.TRUTHS, str(store))
    monkeypatch.setenv(model_flows.PREPARE, "1")
    for kind in model_flows.KINDS:
        model_flows.judge(flow, {}, kind)
    written = sorted((model_flows.store_of(flow) / "truths").iterdir())
    assert len(written) == 2
    assert all(len(list(site.glob("*.truth"))) == staged_units(flow.seeds) for site in written)


@pytest.mark.parametrize("lane", [True, False], ids=["the lane's", "a user's"])
def test_a_flow_holds_in_a_process_of_its_variant_alone(lane):
    """A flow holds only in a process opened under its variant's XLA flags and matmul precision
    (`model_flows.held_process`): a process started with them, the lane's or a user's
    (`lane_environment.USERS_ARITHMETIC`), holds it, and this one, whose backend opened under neither, refuses
    it, as a GPU lane's process of the other variant would."""
    flow = next(flow for flow in model_flows.FLOWS if flow.lane == lane)
    with pytest.raises(model_flows.WrongProcess):
        model_flows.held_process(flow)
    environment = dict(os.environ) if lane else lane_environment.configured(
        lane_environment.users_process(os.environ))
    tests = Path(__file__).resolve().parent
    script = f"import model_flows; model_flows.held_process(model_flows.flow_named({flow.name!r}))"
    ran = relayed([sys.executable, "-c", script], {
        **environment, "XLA_FLAGS": model_flows.flow_flags(flow),
        "PYTHONPATH": os.pathsep.join((str(tests.parent / "src"), str(tests)))}, timeout=600)
    assert ran.returncode == 0, ran.output[-3000:]


def test_a_parts_prepare_compiles_a_flows_programs_for_its_nodes_gpu_and_writes_its_truths(store,
                                                                                          monkeypatch):
    """A GPU part's prepare as its node's CPU runs it (`DEW_TIER_PREPARE=1`): a flow prepared in a process
    of its own (`model_flows.prepared_apart`) over the node's serialized topology (here a compile-only
    A100's, serialized as a host serializes its own), its Dew and reference programs compiled for that GPU
    and its truths written to the tier's store, and what it took reported."""
    from jax.experimental import topologies

    from dew.training.ahead import compile_only_gpus

    topology = store / "topology.bin"
    topology.write_bytes(topologies.TopologyDescription(list(compile_only_gpus("sm80", 1))).serialize())
    monkeypatch.setenv(model_flows.TOPOLOGY, str(topology))
    monkeypatch.setenv(model_flows.TRUTHS, str(store / "tier"))
    monkeypatch.setenv(model_flows.PREPARE, "1")
    flow = model_flows.flow_named("selective scan")
    took = model_flows.prepared_apart(flow)
    store = model_flows.store_of(flow)
    assert took is not None and took["written_bytes"] > 0 and took["seconds"] > 0
    assert (store / "dew.programs").is_file() and (store / "reference.programs").is_file()
    assert len(list((store / "truths").iterdir())) == 2


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
        narrow = (operand.astype(jnp.bfloat16).astype(operand.dtype) for operand in (a, b))
        return jnp.dot(*narrow)

    return dew, reference, draw


def test_a_prepared_site_judges_as_one_that_computes_its_truth_and_refuses_a_missing_or_stale_one(tmp_path):
    """A site prepared (`stored_truths(prepare=True)`) writes every draw's truth and runs no arm; judged from
    that store it passes as it passes computing its own; a store of another stamp, or none, is a red
    (`MissingTruth`), never a truth computed in its place."""
    dew, reference, draw = _site()
    with stored_truths(tmp_path, prepare=True, stamp="a"):
        assert_as_exact(None, reference, draw, "a prepared product", seeds=8)
        with pytest.raises(ValueError):
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
    """A site whose truth holds rows that cancel (its last rows a product less the same product summed in
    two halves, zero in float64 but for its rounding) is sized by its terms (check 1's `_cancelling_terms`):
    prepared, its store holds the terms its truth program gave beside the truth, and judged from the store
    it reads them as a site's own terms program (`_SiteTerms.of`): it passes as it passes computing them,
    and an arm that leaves a term in a cancelling row fails by the same rule."""
    def draw(seed):
        rng = np.random.default_rng([5463, seed])
        return (rng.standard_normal((16, 32)).astype(np.float32),
                rng.standard_normal((32, 8)).astype(np.float32))

    def cancelling(a, b):
        whole = a @ b
        return jnp.concatenate([whole, whole - (a[:, :16] @ b[:16] + a[:, 16:] @ b[16:])])

    def leaving_a_term(a, b):
        return cancelling(a, b).at[16].add(a[0, 0] * b[0, :])

    with stored_truths(tmp_path, prepare=True, stamp="a"):
        assert_as_exact(None, cancelling, draw, "a cancelling product", seeds=8)
    assert_as_exact(cancelling, cancelling, draw, "a cancelling product", seeds=8)
    with pytest.raises(GateFailure) as computing:
        assert_as_exact(leaving_a_term, cancelling, draw, "a cancelling product", seeds=8)
    with stored_truths(tmp_path, prepare=False, stamp="a"):
        assert_as_exact(cancelling, cancelling, draw, "a cancelling product", seeds=8)
        with pytest.raises(GateFailure) as stored:
            assert_as_exact(leaving_a_term, cancelling, draw, "a cancelling product", seeds=8)
    # Sized by its terms the row fails the sized rows' rule; read without them it would fail the rule of
    # rows of float64 zero.
    assert [finding.rule for finding in stored.value.findings] == [
        finding.rule for finding in computing.value.findings]
