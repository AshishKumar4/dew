"""The GPU tier's model flows as a CPU reads them (tests/model_flows.py, row 5461): each flow's model
builds at its widths, its Dew arm takes the routes it is for as each GPU generation's target chooses them,
its reference arm the reference path alone, and its truth computes in float64; and check 1's truths
computed by programs compiled ahead (`truths_ahead.compiled_truths`) judge as the truths a site computes
itself, a missing or stale program refused."""

import dataclasses
import functools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import lane_environment
import jax
import jax.numpy as jnp
import model_flows
import numpy as np
import pytest
import tier_wire
from arm_forms import WITHOUT_YNN, without_ynn
from process_support import relayed
from reference_error import ahead_arm, assert_as_exact, over_windows, run_to_run, windows
from truths_ahead import UNPREPARED, MissingTruth, compiled_truths

from dew.nn.kernels import generation
from dew.nn.kernels.selection import Refusal
from dew.parity import FEWEST_SEEDS, GateFailure


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
    held as its prepare holds them: every route the flow is for that a native kernel claims there
    (`model_flows.claimed`, read from the routes) taken off the reference path or by the kernel it names
    (`model_flows.held_routes`); its reference arm on the reference path alone
    (`model_flows.held_reference`)."""
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
    """An attention flow whose training step took the plain-JAX mainloop in one pass, its backward, beside a
    GPU's own in the rest (the GPU's preferred mainloop turned down there, as one declining a call would),
    does not hold: the template's every pass is held to a GPU's mainloop where one claims it
    (`model_flows.EVERY`, `model_flows.claimed`). Forced onto the plain-JAX mainloop by the caller
    (`generation.chosen`), no GPU mainloop claims it and the flow holds, as it does on its own routes."""
    flow = model_flows.flow_named("dense bf16, the lane's")
    monkeypatch.setenv("XLA_FLAGS", model_flows.flow_flags(flow))
    programs = model_flows.programs_of(flow)
    with generation.assumed("sm90"), jax.default_matmul_precision(flow.precision):
        native = {kind: _routes(programs[f"{kind} dew"]) for kind in model_flows.KINDS}
        with generation.chosen(attention_mainloop="portable"):
            forced = _routes(programs["train dew"])
    model_flows.held_routes(flow, native, "sm90")
    model_flows.held_routes(flow, {**native, "train": forced}, "sm90")
    preferred = {str(route.choice): route.preferred for route in native["train"]}
    one_pass = [dataclasses.replace(route, preferred=preferred[str(route.choice)], forced=None,
                                    refusal=Refusal("declined", "the GPU's mainloop declines the call"))
                for route in forced if str(route.choice) == "attention_mainloop"][:1]
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
    """The flow's truths prepared on a CPU as its node prepares them (`model_flows.judge`, here compiled for
    this CPU, a node's for its GPU): each program's first draw's truth held to float64 (check 1's
    `_of_float64`, a narrow one refused) and every program its sites' truths and terms take compiled into the
    store, at the fewest seeds check 1 takes (`parity.FEWEST_SEEDS`)."""
    fewest = dict.fromkeys(model_flows.KINDS, FEWEST_SEEDS)
    flow = dataclasses.replace(model_flows.flow_named(name), seeds=fewest)
    monkeypatch.setenv(tier_wire.TRUTHS, str(store))
    monkeypatch.setenv(tier_wire.PREPARE, "1")
    for kind in model_flows.KINDS:
        model_flows.judge(flow, {}, kind)


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


def test_a_parts_prepare_compiles_a_flows_programs_and_its_truths_programs_for_its_nodes_gpu(store,
                                                                                          monkeypatch):
    """A GPU part's prepare as its node's CPU runs it (`DEW_TIER_PREPARE=1`): a flow prepared in a process
    of its own (`model_flows.prepared_apart`) over the node's serialized topology (here a compile-only
    A100's, serialized as a host serializes its own), its Dew and reference programs compiled for that GPU
    and its truths' programs compiled into the tier's store, and what it took reported."""
    from jax.experimental import topologies

    from dew.nn.kernels.generation import compile_only_gpus

    topology = store / "topology.bin"
    topology.write_bytes(topologies.TopologyDescription(list(compile_only_gpus("sm80", 1))).serialize())
    monkeypatch.setenv(model_flows.TOPOLOGY, str(topology))
    monkeypatch.setenv(tier_wire.TRUTHS, str(store / "tier"))
    monkeypatch.setenv(tier_wire.PREPARE, "1")
    monkeypatch.setenv(tier_wire.UNIT, "a selective scan's unit")
    flow = model_flows.flow_named("selective scan")
    took = model_flows.prepared_apart(flow)
    store = model_flows.store_of(flow)
    assert took is not None and took["written_bytes"] > 0 and took["seconds"] > 0
    assert (store / "dew.programs").is_file() and (store / "reference.programs").is_file()
    assert len(list((store / "truths").iterdir())) == 2
    # Its programs' needs in the allocator's admission terms (`dew.training.memory.admitted`), filed under
    # the unit the tier sizes a worker for.
    [needs] = (store.parent / "needs").iterdir()
    filed = json.loads(needs.read_text())
    stated = json.loads((store / model_flows.NEED).read_text())["programs"]
    assert (filed["unit"], filed["programs"]) == ("a selective scan's unit", stated)


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
    """A site prepared (`compiled_truths(prepare=True)`) compiles its truth's programs and its arms' for every
    shape its stages' draws take and runs no arm; judged from that store it passes as it passes computing
    its own; a store of another stamp, of other shapes than a draw's, or of another Dew arm than the run's is
    a red (`MissingTruth`), never a program compiled in its place; a site its prepare did not reach runs
    unprepared, every draw judged, recorded in the store (`UNPREPARED`)."""
    dew, reference, draw = _site()
    def scaled(factor):
        def arm(a, b):
            return dew(a, b) * factor

        return arm

    with compiled_truths(tmp_path, prepare=True, stamp="a"):
        assert_as_exact(scaled(1.0), reference, draw, "a prepared product", seeds=8)
        with pytest.raises(ValueError):
            assert_as_exact(scaled(1.0), reference, draw, "a prepared product", seeds=8)
    assert_as_exact(scaled(1.0), reference, draw, "a prepared product", seeds=8)
    with compiled_truths(tmp_path, prepare=False, stamp="a"):
        assert_as_exact(scaled(1.0), reference, draw, "a prepared product", seeds=8)
    with compiled_truths(tmp_path, prepare=False, stamp="b"), pytest.raises(MissingTruth):
        assert_as_exact(scaled(1.0), reference, draw, "a prepared product", seeds=8)
    # A site its prepare did not reach runs unprepared, every draw judged, recorded in the store.
    with compiled_truths(tmp_path, prepare=False, stamp="a"):
        assert_as_exact(scaled(1.0), reference, draw, "another product", seeds=8)
    assert (tmp_path / UNPREPARED).is_file()

    def wider(seed):
        a, b = draw(seed)
        return np.concatenate([a, a]), b

    with compiled_truths(tmp_path, prepare=False, stamp="a"), pytest.raises(MissingTruth):
        assert_as_exact(scaled(1.0), reference, wider, "a prepared product", seeds=8)
    # The same function closing over another factor: another computation, never the prepared one's.
    with compiled_truths(tmp_path, prepare=False, stamp="a"), pytest.raises(MissingTruth):
        assert_as_exact(scaled(1.0 + 2.0 ** -8), reference, draw, "a prepared product", seeds=8)


def test_a_compiled_sites_run_traces_none_of_its_truth(tmp_path):
    """Judged from programs compiled ahead (`compiled_truths(prepare=False)`), a site's run computes every
    draw's truth with them and runs none of its truth's code, which its prepare alone traced; computing its
    own truths, a site traces it."""
    dew, reference, draw = _site()
    traced = []

    def truth(a, b):
        traced.append(None)
        return jnp.dot(a, b)

    with compiled_truths(tmp_path, prepare=True, stamp="a"):
        assert_as_exact(dew, reference, draw, "a product's truth", seeds=8, truth_fn=truth)
    assert traced
    traced.clear()
    with compiled_truths(tmp_path, prepare=False, stamp="a"):
        assert_as_exact(dew, reference, draw, "a product's truth", seeds=8, truth_fn=truth)
    assert not traced
    assert_as_exact(dew, reference, draw, "a product's truth", seeds=8, truth_fn=truth)
    assert traced


ARM_FACTOR = 1.0
"""A module's flag an arm reads (`test_an_arm_reading_another_flag_than_its_prepares_is_a_red`), as a test's
backend read at import is one."""

ARM_OPTIONS = {"factor": 1.0}
"""A module's options an arm reads (`test_an_arm_reading_other_options_than_its_prepares_is_a_red`), as a
test's options read off its backend are."""


def test_an_arm_reading_another_flag_than_its_prepares_is_a_red(tmp_path, monkeypatch):
    """An arm whose module's flag reads otherwise in its run than in its prepare (a backend read at import:
    the prepare's CPU, the run's GPU) is another arm: its run finds none of it (`MissingTruth`), never the
    prepare's program in its place; read alike, it passes."""
    dew, reference, draw = _site()

    def flagged(a, b):
        return dew(a, b) * ARM_FACTOR

    with compiled_truths(tmp_path, prepare=True, stamp="a"):
        assert_as_exact(flagged, reference, draw, "a flagged product", seeds=8)
    with compiled_truths(tmp_path, prepare=False, stamp="a"):
        assert_as_exact(flagged, reference, draw, "a flagged product", seeds=8)
    monkeypatch.setattr(sys.modules[__name__], "ARM_FACTOR", 1.0 + 2.0 ** -8)
    with compiled_truths(tmp_path, prepare=False, stamp="a"), pytest.raises(MissingTruth):
        assert_as_exact(flagged, reference, draw, "a flagged product", seeds=8)


def test_an_arm_reading_other_options_than_its_prepares_is_a_red(tmp_path, monkeypatch):
    """An arm reading a dict whose contents are others in its run than in its prepare (options read off the
    backend: the prepare's CPU, the run's GPU, each process its own dict) is another arm: its run finds none
    of it (`MissingTruth`), never the prepare's program in its place; with the same contents it passes (row
    7674: a container stated by its type alone named one arm in both, and the run computed the prepare's
    program)."""
    dew, reference, draw = _site()

    def optioned(a, b):
        return dew(a, b) * ARM_OPTIONS["factor"]

    with compiled_truths(tmp_path, prepare=True, stamp="a"):
        assert_as_exact(optioned, reference, draw, "an optioned product", seeds=8)
    monkeypatch.setattr(sys.modules[__name__], "ARM_OPTIONS", {"factor": 1.0})
    with compiled_truths(tmp_path, prepare=False, stamp="a"):
        assert_as_exact(optioned, reference, draw, "an optioned product", seeds=8)
    monkeypatch.setattr(sys.modules[__name__], "ARM_OPTIONS", {"factor": 1.0 + 2.0 ** -8})
    with compiled_truths(tmp_path, prepare=False, stamp="a"), pytest.raises(MissingTruth):
        assert_as_exact(optioned, reference, draw, "an optioned product", seeds=8)


def test_a_record_a_trace_fills_is_named_as_first_found_inside_another_container(tmp_path, monkeypatch):
    """A list an arm's trace fills (a test's record of its calls), named first by one site's arm and read by a
    later site's arm inside a dict, beside an alias of itself, is stated there as it was first found, so the
    later arm's run, a process of its own whose earlier site's executable fills none of it, finds its program
    (the reviews of row 7674's fix: the dict stated the list as the prepare's trace had filled it, and then
    its alias inside another dict did)."""
    import truths_ahead

    dew, reference, draw = _site()
    calls: list = []

    def recording(a, b):
        calls.append(None)
        return dew(a, b)

    # The record inside the dict, and beside an alias of itself inside another (the review's).
    held = {"direct": calls, "nested": {"calls": calls}}

    def holding(a, b):
        return dew(a, b) * (1.0 + 0.0 * len(held))

    with compiled_truths(tmp_path, prepare=True, stamp="a"):
        assert_as_exact(recording, reference, draw, "a recorded product", seeds=8)
        assert_as_exact(holding, reference, draw, "a held product", seeds=8)
    # The run's own process: its record empty and every container first named in it.
    calls.clear()
    monkeypatch.setattr(truths_ahead, "_FIRST_NAMED", {})
    with compiled_truths(tmp_path, prepare=False, stamp="a"):
        assert_as_exact(recording, reference, draw, "a recorded product", seeds=8)
        assert_as_exact(holding, reference, draw, "a held product", seeds=8)


def test_a_prepared_sites_needs_are_filed_under_its_unit(tmp_path, monkeypatch):
    """A test's sites prepared under a GPU tier part's unit (`truths_ahead.prepared`, the suite's prepare of
    every test, `tier_wire.UNIT` and `TRUTHS` named) file their programs' needs under the unit as a flow's
    are (`tier_wire.unit_needs`, row 7900): one admission for each program its run calls, its arms' and its
    truths', every truth's with the arms' state held beside it, so the lane sizes the unit's worker by
    them; before it filed none, and the lane admitted the unit on no needs at all."""
    import tier_wire
    from jax.experimental import topologies
    from truths_ahead import compiled_memory, prepared

    from dew.nn.kernels.generation import compile_only_gpus

    topology = tmp_path / "topology.bin"
    topology.write_bytes(topologies.TopologyDescription(list(compile_only_gpus("sm80", 1))).serialize())
    monkeypatch.setenv(model_flows.TOPOLOGY, str(topology))
    monkeypatch.setenv(tier_wire.UNIT, "a unit")
    monkeypatch.setenv(tier_wire.TRUTHS, str(tmp_path))
    dew, reference, draw = _site()
    store = tmp_path / "sites"
    with prepared(store, stamp="a", target=model_flows.node_target()):
        assert_as_exact(dew, reference, draw, "a product's arms", seeds=8)
    arms, truths = compiled_memory(store, arms=True), compiled_memory(store)
    filed = json.loads(tier_wire.needs_file(tmp_path, "a unit").read_text())["programs"]
    assert arms and truths and len(filed) == len(arms) + len(truths)
    state = max(memory["argument"] for memory in arms)
    assert all(program["beside"] >= state for program in filed[len(arms):])


def test_a_compiled_sites_run_traces_none_of_its_arms(tmp_path):
    """Judged from programs compiled ahead (`compiled_truths(prepare=False)`), a site's run computes every
    draw's arms with them and runs none of its arms' code, which its prepare alone traced (a GPU run then
    traces, lowers and compiles none of them); computing its own, a site traces both. Its verdict is the
    one a site computing its arms reaches."""
    dew, reference, draw = _site()
    dew_traced, reference_traced = [], []

    def dew_arm(a, b):
        dew_traced.append(None)
        return dew(a, b)

    def reference_arm(a, b):
        reference_traced.append(None)
        return reference(a, b)

    with compiled_truths(tmp_path, prepare=True, stamp="a"):
        assert_as_exact(dew_arm, reference_arm, draw, "a product's arms", seeds=8)
    assert dew_traced and reference_traced
    dew_traced.clear()
    reference_traced.clear()
    with compiled_truths(tmp_path, prepare=False, stamp="a"):
        assert_as_exact(dew_arm, reference_arm, draw, "a product's arms", seeds=8)
    assert not dew_traced and not reference_traced
    assert_as_exact(dew_arm, reference_arm, draw, "a product's arms", seeds=8)
    assert dew_traced and reference_traced


def _rewriting() -> bool:
    """Whether the algebra pass is on where this is called (`dew.algebra.passes`): a function of this module,
    which an arm's statement names (`dew._stated.stated`), where the switch itself has no statable form."""
    from dew._cache import ALGEBRA

    return ALGEBRA.value.on


def test_an_arm_of_its_own_compiler_options_is_prepared_whole(tmp_path):
    """A site whose reference arm is a jit of its own compiler options (`without_ynn`, which on the CPU lane
    turns YNNPACK's fusions off by them: a library arm's cuBLAS on a GPU,
    `generation_support.cublas_compiled`) prepares it whole, lowered as itself, and its run computes it from
    that program, tracing none, with no test of it recorded unprepared (row 8469: traced inside the arms'
    program, the prepare refused it, 'compiler_options can only be passed to top-level jax.jit', and its test
    ran unprepared). Its truth its own float64 function, as such a site's is (`blockwise_matrix`'s cuBLASLt
    reference beside its truth): a truth is walked by its trace, which no jit of compiler options lives in.
    A reference, it is traced and lowered with the algebra pass off, as its call runs (`unrewritten`): held
    whole, it was traced with the pass on, the program it stored a rewrite of the reference (the review).
    Its draw's arrays JAX's, so no NumPy refusal runs it beside the gate's arm (`_refuse_numpy_arms`)."""
    dew, reference, numpy_draw = _site()
    traced: list[bool] = []

    def counted(a, b):
        traced.append(_rewriting())
        return reference(a, b)

    def draw(seed):
        return tuple(map(jnp.asarray, numpy_draw(seed)))

    # Its own compiler options, YNNPACK's off, whatever the census (`arm_forms.switched_off`).
    whole = without_ynn(counted, compiler_options=dict(WITHOUT_YNN))
    with compiled_truths(tmp_path, prepare=True, stamp="a"):
        assert_as_exact(dew, whole, draw, "a product of a compiled reference", seeds=8, truth_fn=reference)
    assert traced and not any(traced)
    traced.clear()
    with compiled_truths(tmp_path, prepare=False, stamp="a"):
        assert_as_exact(dew, whole, draw, "a product of a compiled reference", seeds=8, truth_fn=reference)
    assert not traced and not (tmp_path / UNPREPARED).exists()


def test_a_library_sites_arms_are_prepared_and_run_from_their_programs(tmp_path):
    """A site with a library arm (`library_fn`, the owner's exception under BIBLE section 7, run on Dew's own
    inputs beside the reference) prepares it as one of its arms, and its run computes it from that program,
    tracing none, with no test of it recorded unprepared (row 8468: a store refused every library site,
    which then compiled on its GPU, test_gemm_gather's cuBLAS product among them)."""
    dew, reference, draw = _site()
    traced: list[None] = []

    def library(a, b):
        traced.append(None)
        return jnp.dot(a, b)

    options = {"library_fn": library, "library_row": 8468, "library_leaves": [""]}
    with compiled_truths(tmp_path, prepare=True, stamp="a"):
        assert_as_exact(dew, reference, draw, "a product beside its library", seeds=8, **options)
    assert traced
    traced.clear()
    with compiled_truths(tmp_path, prepare=False, stamp="a"):
        assert_as_exact(dew, reference, draw, "a product beside its library", seeds=8, **options)
    assert not traced and not (tmp_path / UNPREPARED).exists()


@pytest.mark.parametrize("jitted", [jax.jit, lambda fn: without_ynn(fn, compiler_options=dict(WITHOUT_YNN))],
                         ids=["a plain jit", "a jit of its own compiler options"])
def test_a_site_free_arm_runs_its_prepared_program(tmp_path, jitted):
    """A computation with no check-1 site (a bitwise check's kernel, `reference_error.ahead_arm`) is
    prepared as an arm is: its prepare traces it and gives zeros of its outputs' shapes, its run computes it
    with the program its prepare compiled, tracing none, the same values bit for bit as computing it; one its
    prepare did not reach runs unprepared, recorded and named (bible section 8: compiling in the prepare,
    none on a held GPU; the GPU tier's site-free tests computed before any site there and compiled on it)."""
    traced: list[None] = []

    def scaled(a, b):
        traced.append(None)
        return a * b + 1.0

    a, b = (np.arange(6, dtype=np.float32).reshape(2, 3) + offset for offset in (1.0, 0.5))
    with compiled_truths(tmp_path, prepare=True, stamp="a"):
        prepared = ahead_arm(jitted(scaled), "a scaled product")(a, b)
    assert traced and np.asarray(prepared).tolist() == np.zeros((2, 3), np.float32).tolist()
    traced.clear()
    with compiled_truths(tmp_path, prepare=False, stamp="a"):
        stored = ahead_arm(jitted(scaled), "a scaled product")(a, b)
    assert not traced and np.asarray(stored).tolist() == (a * b + 1.0).tolist()
    # Another function's jit, which JAX's own cache of traces (of `scaled`, the prepare's) does not hold.
    with compiled_truths(tmp_path, prepare=False, stamp="a"):
        unreached = ahead_arm(jitted(lambda a, b: scaled(a, b)), "a product no prepare reached")(a, b)
    assert traced and np.asarray(unreached).tolist() == (a * b + 1.0).tolist()
    assert (tmp_path / UNPREPARED).is_file()


@pytest.mark.parametrize("jitted", [False, True], ids=["a plain reference", "a jitted reference"])
def test_a_qualification_run_sizes_a_prepared_site_through_its_stored_arms(tmp_path, monkeypatch, jitted):
    """A qualification run (`check1_wire.QUALIFY`) of a site prepared for it sizes the site's seeds and then
    judges it through the programs its prepare compiled, the sizing draws' among them (row 7606): no arm
    traces at its float32 inputs in its run, where clearing the store compiled both on its GPU, the
    reference traced only as its sizing's float64 truth, on the host, plain or jitted; the count it sizes is
    recorded."""
    from check1_wire import QUALIFY

    dew, reference, draw = _site()
    narrow: list[bool] = []

    def dew_arm(a, b):
        narrow.append(a.dtype == jnp.float32)
        return dew(a, b)

    def reference_arm(a, b):
        narrow.append(a.dtype == jnp.float32)
        return reference(a, b)

    reference_fn = jax.jit(reference_arm) if jitted else reference_arm
    record = tmp_path / "qualified.jsonl"
    monkeypatch.setenv(QUALIFY, str(record))
    with compiled_truths(tmp_path / "store", prepare=True, stamp="a"):
        assert_as_exact(dew_arm, reference_fn, draw, "a product's arms", seeds=8)
    narrow.clear()
    with compiled_truths(tmp_path / "store", prepare=False, stamp="a"):
        assert_as_exact(dew_arm, reference_fn, draw, "a product's arms", seeds=8)
    assert not any(narrow)
    assert json.loads(record.read_text().splitlines()[-1])["sized"] <= 8


def test_a_compiled_sites_window_loop_and_run_to_run_arms_are_traced_by_its_prepare_alone(tmp_path):
    """Arms over a site's windows (`over_windows`) and declared to vary from run to run (`run_to_run`), each
    a record of its arm: judged from programs compiled ahead, the run traces neither (their prepare did), as
    a plain arm's (train 1016's F2: such adapters went unstated, so unprepared)."""
    dew, reference, draw = _site()
    traced = []

    def dew_arm(a, b):
        traced.append(None)
        return dew(a, b)

    def windowed(seed):
        return tuple(np.stack([x, x]) for x in draw(seed))

    # A jitted arm under both adapters: each a record of its arm, never stated as the wrapper it holds.
    arm = run_to_run(over_windows(jax.jit(dew_arm)), "a product summed in arrival order")
    with compiled_truths(tmp_path, prepare=True, stamp="a"):
        assert_as_exact(arm, over_windows(reference), windowed, "a product's adapted arms", seeds=8)
    assert traced
    traced.clear()
    with compiled_truths(tmp_path, prepare=False, stamp="a"):
        assert_as_exact(arm, over_windows(reference), windowed, "a product's adapted arms", seeds=8)
    assert not traced
    assert not (tmp_path / UNPREPARED).exists()


def test_a_window_loops_truth_is_computed_window_by_window_from_its_windows_program(tmp_path):
    """A site over windows (`over_windows`) prepared compiles its truth's program at one window's shapes, and
    its terms' at the whole draw's (a program of its own, or its window function's walk, `terms=True`), and
    its run computes every window's truth with the one, as the loop runs them, and the terms with the other:
    it passes as it passes computing its own."""
    dew, reference, draw = _site()
    windowed = windows(draw, 2)

    def terms(a, b):
        # A product's terms, row by row: each output's products' root sum of squares.
        return jnp.sqrt(jnp.sum(jnp.square(a[..., :, :, None] * b[..., None, :, :]), axis=-2))

    def stacked(a, b):
        # The reference's product of the window's stack too, which a walk over the whole draw reads.
        return jnp.matmul(*(operand.astype(jnp.bfloat16).astype(operand.dtype) for operand in (a, b)))

    with compiled_truths(tmp_path, prepare=True, stamp="a"):
        assert_as_exact(over_windows(dew), over_windows(reference), windowed, "a product over windows",
                        seeds=8)
        assert_as_exact(over_windows(dew), over_windows(reference), windowed,
                        "a product over windows, its terms", seeds=8, terms=terms)
        assert_as_exact(over_windows(dew), over_windows(stacked), windowed, "a product over windows, walked",
                        seeds=8,
                        terms=True)
    with compiled_truths(tmp_path, prepare=False, stamp="a"):
        assert_as_exact(over_windows(dew), over_windows(reference), windowed, "a product over windows",
                        seeds=8)
        assert_as_exact(over_windows(dew), over_windows(reference), windowed,
                        "a product over windows, its terms", seeds=8, terms=terms)
        assert_as_exact(over_windows(dew), over_windows(stacked), windowed,
                        "a product over windows, walked", seeds=8, terms=True)


def test_a_prepared_site_sized_by_its_terms_reads_them_from_its_store(tmp_path):
    """A site whose truth holds rows that cancel (its last rows a product less the same product summed in
    two halves, zero in float64 but for its rounding) is sized by its terms (check 1's `_cancelling_terms`):
    prepared, its store holds its truth's float64 walk compiled beside its truth, which its run decides on
    at its first draw and sizes every draw's rows by: it passes as it passes computing them, and an arm that
    leaves a term in a cancelling row fails by the same rule."""
    def draw(seed):
        rng = np.random.default_rng([5463, seed])
        return (rng.standard_normal((16, 32)).astype(np.float32),
                rng.standard_normal((32, 8)).astype(np.float32))

    def cancelling(a, b):
        whole = a @ b
        return jnp.concatenate([whole, whole - (a[:, :16] @ b[:16] + a[:, 16:] @ b[16:])])

    def leaving_a_term(a, b):
        return cancelling(a, b).at[16].add(a[0, 0] * b[0, :])

    with compiled_truths(tmp_path, prepare=True, stamp="a"):
        assert_as_exact(cancelling, cancelling, draw, "a cancelling product", seeds=8)
        assert_as_exact(leaving_a_term, cancelling, draw, "a cancelling product, a term left", seeds=8)
    assert_as_exact(cancelling, cancelling, draw, "a cancelling product", seeds=8)
    with pytest.raises(GateFailure) as computing:
        assert_as_exact(leaving_a_term, cancelling, draw, "a cancelling product", seeds=8)
    with compiled_truths(tmp_path, prepare=False, stamp="a"):
        assert_as_exact(cancelling, cancelling, draw, "a cancelling product", seeds=8)
        with pytest.raises(GateFailure) as stored:
            assert_as_exact(leaving_a_term, cancelling, draw, "a cancelling product, a term left", seeds=8)
    # Sized by its terms the row fails the sized rows' rule; read without them it would fail the rule of
    # rows of float64 zero.
    assert [finding.rule for finding in stored.value.findings] == [
        finding.rule for finding in computing.value.findings]


@pytest.mark.parametrize("held", ["0", "0,1"], ids=["one GPU", "two GPUs"])
def test_a_gpu_tier_parts_prepare_runs_under_its_runs_flags(held):
    """A GPU tier part's prepare (`DEW_TIER_PREPARE`, on the node's CPU, which hides its GPUs from it) is
    configured under the XLA flags
    that decide what its run's programs compute (`lane_environment.configure_lane`'s cuda lane,
    `dew.training.ahead.computation_flags`), whatever GPUs its run holds, so the programs it compiles load
    where the run holds them to its own (`BuildTarget.held`: the prepare's CPU lane's devices and scheduler
    beside the run's deterministic GPU ops refused test_gpu_numerics' vocabulary cell); and its CPU client
    opens, with a host device (one configured for a GPU of none failed every GPU part's prepare: "Could not
    determine process_index").

    The prepare's environment is the one its node and lane give it, mirrored, as Dew reads nothing of dew-ops
    (dew-ops reads Dew's `tier_wire`, never the reverse): dew-ops' session hides every GPU from every job's
    prepare (dewops/node/session.py's prepare environment, `"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS":
    "cpu"`, line 2388 at ops 945f3032f), and the lane's prepare names itself (tools/armada/ci.py,
    `gpu_prepared`: `JAX_PLATFORMS` cpu and `tier_wire.PREPARE`)."""
    import lane_environment

    from dew.training.ahead import computation_flags

    prepare = lane_environment.configured(
        {"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "DEW_TIER_PREPARE": "1"})
    run = lane_environment.configured({"CUDA_VISIBLE_DEVICES": held, "JAX_PLATFORMS": "cuda"})
    assert computation_flags(prepare["XLA_FLAGS"]) == computation_flags(run["XLA_FLAGS"])
    opened = subprocess.run([sys.executable, "-c", "import jax; print(len(jax.devices('cpu')))"],
                            env={**os.environ, **prepare}, capture_output=True, text=True, timeout=300)
    assert (opened.returncode, opened.stdout.split()[-1:]) == (0, ["1"]), opened.stderr[-2000:]


def test_a_parts_prepare_runs_every_cell_its_nodes_gpu_runs(store, monkeypatch):
    """A GPU part's prepare on its node's CPU (`DEW_TIER_PREPARE=1`, the node an A100 by its serialized
    topology) runs every cell of a flow pinned to the node's generation, Mosaic's among them, as its run
    does (`gpu_flow.run_cells`, `generation_support.runnable` reading the node's generation): none skipped
    for a GPU the prepare's process holds none of (row 7868: the recurrence matrix's five cells skipped
    there, then compiled on the held GPU)."""
    import gpu_flow
    from jax.experimental import topologies

    from dew.nn.kernels.generation import compile_only_gpus

    topology = store / "topology.bin"
    topology.write_bytes(topologies.TopologyDescription(list(compile_only_gpus("sm80", 1))).serialize())
    monkeypatch.setenv(model_flows.TOPOLOGY, str(topology))
    monkeypatch.setenv(tier_wire.PREPARE, "1")
    ran: list[int] = []
    cells = [gpu_flow.Cell(f"cell {index}", functools.partial(ran.append, index), generation="sm80",
                           mosaic=index % 2 == 1) for index in range(5)]
    # The node's compile-only GPU the default device, as a part's prepare holds it (`compiled_ahead`).
    with jax.default_device(compile_only_gpus("sm80", 1)[0]):
        assert gpu_flow.run_cells(cells) == []
    assert ran == list(range(5))


@pytest.mark.mesh(devices=2)
def test_a_truth_run_as_given_computes_on_the_host_in_a_prepare(tmp_path, monkeypatch):
    """A site whose float64 truth the gate runs as given (`as_given`: host work, NumPy, torch) has its value
    read in its prepare (`_shaped_truth`) on the host's CPU, its inputs placed there, wherever the store's
    device is: a GPU part's prepare computes nothing on its default device, a compile-only GPU, where the
    truth's eager operations raised "'Executable' object has no attribute 'local_devices'" and the test ran
    unprepared (row 4881's class: a compensated primitive's site, the audio encoders' cells)."""
    from arm_forms import as_given
    from truths_ahead import CompiledTruths

    dew, reference, draw = _site()
    placed = []

    def truth(a, b):
        placed.append((jax.config.jax_default_device, {device for x in (a, b) for device in x.devices()}))
        return reference(a, b)

    # The store's device another than the host's first, as a GPU part's compile-only GPU is.
    monkeypatch.setattr(CompiledTruths, "device", lambda store: jax.devices()[1])
    with compiled_truths(tmp_path, prepare=True, stamp="a"):
        assert_as_exact(dew, reference, draw, "a product of a given truth", seeds=8, truth_fn=as_given(truth))
    host = jax.devices("cpu")[0]
    assert placed and all(device == host and on == {host} for device, on in placed)


@pytest.mark.mesh(devices=2)
def test_a_host_placed_arm_is_prepared_and_run_on_the_host(tmp_path, monkeypatch):
    """A site-free arm placed on the host's CPU (`ahead_arm(host=True)`: a GPU host's CPU-placed computation,
    row 8376's) is traced in its prepare placed on the host's CPU, whatever the store's device (a GPU part's
    compile-only GPU), and compiled for it; its run, in a process listing another device first as a GPU's
    does, computes it from that program, tracing none, the same values as computing it."""
    from truths_ahead import CompiledTruths

    traced = []

    def scaled(a, b):
        traced.append(jax.config.jax_default_device)
        return a * b + 1.0

    a, b = (np.arange(6, dtype=np.float32).reshape(2, 3) + offset for offset in (1.0, 0.5))
    host = jax.devices("cpu")[0]
    monkeypatch.setattr(CompiledTruths, "device", lambda store: jax.devices()[1])
    with compiled_truths(tmp_path, prepare=True, stamp="a"):
        ahead_arm(jax.jit(scaled), "a host-placed product", host=True)(a, b)
    assert traced == [host]
    traced.clear()
    listed = jax.devices
    monkeypatch.setattr(jax, "devices", lambda backend=None: listed(backend) if backend else listed()[1:])
    with compiled_truths(tmp_path, prepare=False, stamp="a"):
        ran = ahead_arm(jax.jit(scaled), "a host-placed product", host=True)(a, b)
    assert not traced and np.asarray(ran).tolist() == (a * b + 1.0).tolist()
