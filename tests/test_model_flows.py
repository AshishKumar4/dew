"""The GPU tier's model flows as a CPU reads them (tests/model_flows.py, row 5461): each flow's model
builds at its widths, its Dew arm takes the routes it is for as each GPU generation's target chooses them,
its reference arm the reference path alone, and its truth computes in float64; and check 1's truths
computed by programs compiled ahead (`truths_ahead.compiled_truths`) judge as the truths a site computes
itself, a missing or stale program refused."""

import contextlib
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
from generation_support import node_topology
from process_support import relayed
from reference_error import ahead_arm, assert_as_exact, over_windows, run_to_run, windows
from tool_support import load
from truths_ahead import AT_RUN, UNPREPARED, MissingTruth, compiled_truths

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
        dew = {kind: _routes(programs[f"{kind} dew"]) for kind in flow.kinds}
        reference = [route for kind in flow.kinds
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
        native = {kind: _routes(programs[f"{kind} dew"]) for kind in flow.kinds}
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
    named = model_flows.flow_named(name)
    flow = dataclasses.replace(named, seeds=dict.fromkeys(named.kinds, FEWEST_SEEDS))
    monkeypatch.setenv(tier_wire.TRUTHS, str(store))
    monkeypatch.setenv(tier_wire.PREPARE, "1")
    for kind in flow.kinds:
        model_flows.judge(flow, {}, kind)


def test_a_packed_flows_exact_banks_are_its_banks_values():
    """A packed flow's truths read each bank's exact matrix from its bits (`model_flows.exact_banks`, by
    integer arithmetic and casts, its products in the truth's float64): each value its bfloat16 decode (code
    times block scale over 2^8, exact) times its factor (2^8 times its expert's scale), multiplied on the
    host in float64, where their product is exact, bit for bit."""
    from flax.traverse_util import flatten_dict

    from dew.nn.kernels.formats import NvFp4
    from dew.quantization import QUANTIZATION

    flow = next(flow for flow in model_flows.FLOWS if isinstance(flow, model_flows.Flow) and flow.packed)
    drawn = flow.taken(flow.model.init(jax.random.key(0), *flow.initialized()))
    banks = flatten_dict(dict(drawn[QUANTIZATION]), is_leaf=lambda _, leaf: isinstance(leaf, NvFp4))

    def widened(leaf):
        # As check 1 widens a draw's floats for its truth.
        return jnp.asarray(leaf, jnp.float64) if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf

    with jax.enable_x64(new_val=True):
        exact = flatten_dict(model_flows.exact_banks(jax.tree.map(widened, drawn))[QUANTIZATION])
        assert exact.keys() == banks.keys()
        for path, matrix in exact.items():
            bank = banks[path]
            want = (np.asarray(bank.decoded(), np.float64)
                    * np.asarray(bank.factor, np.float64)[..., None, None])
            assert matrix.dtype == jnp.float64
            np.testing.assert_array_equal(np.asarray(matrix), want)


def test_a_packed_flows_training_moves_its_parameters_and_never_its_banks():
    """A flow whose experts are packed NVFP4 banks (`Flow.packed`, row 8492) trains its parameters alone: two
    SGD steps over its training step's gradients, on the CPU's portable route, move its parameters and leave
    every bank's words and scales bit for bit as drawn, the banks frozen beside them as a packed load holds
    them (`model_flows.packed_variables`)."""
    from dew.quantization import QUANTIZATION

    flow = next(flow for flow in model_flows.FLOWS if isinstance(flow, model_flows.Flow) and flow.packed)
    flow = dataclasses.replace(flow, rows=1, length=32, prefill=24)
    drawn = flow.taken(flow.model.init(jax.random.key(0), *flow.initialized()))
    # The flow's own training step (`programs_of`'s), its Dew arm's named kernel a GPU's alone.
    step = jax.jit(flow.program("train", flow.model))
    ids = np.random.default_rng(0).integers(0, flow.vocab, (flow.rows, flow.length), np.int32)
    trained = drawn
    for _ in range(2):
        _, gradients = step(trained, ids)
        assert jax.tree.structure(gradients) == jax.tree.structure(trained["params"])
        moved = jax.tree.map(lambda held, gradient: held - 0.1 * gradient, trained["params"], gradients)
        assert any(not np.array_equal(np.asarray(held), np.asarray(after)) for held, after in
                   zip(jax.tree.leaves(trained["params"]), jax.tree.leaves(moved), strict=True))
        trained = {**trained, "params": moved}
        for held, kept in zip(jax.tree.leaves(drawn[QUANTIZATION]), jax.tree.leaves(trained[QUANTIZATION]),
                              strict=True):
            assert np.array_equal(np.asarray(kept), np.asarray(held))


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
    and its truths' programs compiled into the tier's store, and what it took reported. The dense fp32
    decoder's flow, the cheapest of two kinds to prepare: with the selective scan's this test took 120 s on
    CI at main 4f7110c66 and 117 s at 84cba23c0, against a 120 s group target (row 9709); prepared apart
    on armada, the dense fp32 flow took 77 s and the selective scan's 152."""


    topology = node_topology(store / "topology.bin")
    monkeypatch.setenv(model_flows.TOPOLOGY, str(topology))
    monkeypatch.setenv(tier_wire.TRUTHS, str(store / "tier"))
    monkeypatch.setenv(tier_wire.PREPARE, "1")
    monkeypatch.setenv(tier_wire.UNIT, "a dense decoder's unit")
    flow = model_flows.flow_named("dense fp32")
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
    assert (filed["unit"], filed["programs"]) == ("a dense decoder's unit", stated)


def test_a_flows_prepare_leaves_a_convolution_to_its_run_and_says_so(store, monkeypatch):
    """A flow whose programs hold convolutions (the UNet denoiser's), prepared over a compile-only A100's
    topology, prepares none of them: a convolution compiled without its GPU leaves its algorithm to run
    time (`report.CompiledOnDevice`), so its store names each program its run compiles on its GPU by the
    refusal's reason (`model_flows.ON_DEVICE`) and holds none of them, its truths compiled all the same;
    its run names each in the run's summary as its prepare declared (`truths_ahead.AT_RUN`), as an at-run
    site's truths are (row 8636)."""


    topology = node_topology(store / "topology.bin")
    monkeypatch.setenv(model_flows.TOPOLOGY, str(topology))
    monkeypatch.setenv(tier_wire.TRUTHS, str(store / "tier"))
    monkeypatch.setenv(tier_wire.PREPARE, "1")
    monkeypatch.setenv(tier_wire.UNIT, "a UNet's unit")
    flow = model_flows.flow_named("UNet denoiser fp32")
    assert model_flows.prepared_apart(flow) is not None
    held = model_flows.store_of(flow)
    on_device = json.loads((held / model_flows.ON_DEVICE).read_text())["programs"]
    assert sorted(on_device) == sorted(f"{kind} {arm}" for kind in flow.kinds for arm in ("dew", "reference"))
    assert all("convolution compiled without its GPU" in why for why in on_device.values())
    assert not (held / "dew.programs").exists() and not (held / "reference.programs").exists()
    assert len(list((held / "truths").iterdir())) == len(flow.kinds)
    assert sorted(model_flows.loaded(flow)) == sorted(on_device)
    named = json.loads((held / "truths" / AT_RUN).read_text())["reasons"]
    assert sorted(reason["error"] for reason in named) == sorted(
        f"{flow.name} {name}: {why}" for name, why in on_device.items())


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
    from truths_ahead import compiled_memory, prepared


    topology = node_topology(tmp_path / "topology.bin")
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


def test_a_convolution_sites_truths_and_arms_compile_on_its_gpu_in_its_run(tmp_path, monkeypatch):
    """A site holding a convolution, prepared for a node's GPU (`truths_ahead.prepared`): no compile without
    that GPU holds the convolution's algorithm, which cuDNN then picks at run time, on an A100 not the same
    each run (`report.DEVICELESS_CONVOLUTION_PICKS`, row 7941). Its prepare writes its truths and its arms
    at run, compiling none, their run compiling both on its GPU, its float64 truths on its FP64 units, as its
    prepare declared, which the run names (`truths_ahead.AT_RUN`: its truths and each arm), nothing
    unprepared; before, the prepare compiled both for the GPU and its run refused their load, and then its
    truths for the host's CPU, dense float64 on the node's CPU in the run (row 8576)."""
    import tier_wire
    from truths_ahead import AT_RUN, prepared


    topology = node_topology(tmp_path / "topology.bin")
    monkeypatch.setenv(model_flows.TOPOLOGY, str(topology))
    monkeypatch.setenv(tier_wire.UNIT, "a unit")
    monkeypatch.setenv(tier_wire.TRUTHS, str(tmp_path))

    def draw(seed):
        rng = np.random.default_rng([7941, seed])
        return (rng.standard_normal((1, 8, 8, 4)).astype(np.float32),
                rng.standard_normal((3, 3, 4, 4)).astype(np.float32))

    def convolved(x, w):
        return jax.lax.conv_general_dilated(x, w, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))

    store = tmp_path / "sites"
    with prepared(store, stamp="a", target=model_flows.node_target()):
        assert_as_exact(jax.jit(convolved), jax.jit(convolved), draw, "a convolution", seeds=8)
    sites = [json.loads(site.read_text()) for site in store.glob("*/site.json")]
    # Its truths, its arms, and its reference's twin as written (`arm_forms._written_twin`), apart.
    assert sorted((site["arms"], site["host"], "at_run" in site) for site in sites) == [
        (False, False, True), (True, False, True), (True, False, True)], sites
    # Their memory read offline, by which the lane sizes the run's worker (`compiled_memory`).
    assert all(site["memory"] for site in sites), sites
    with compiled_truths(store, prepare=False, stamp="a"):
        assert_as_exact(jax.jit(convolved), jax.jit(convolved), draw, "a convolution", seeds=8)
    # Its truths, its two arms and its reference's twin, each named once.
    assert len(json.loads((store / AT_RUN).read_text())["reasons"]) == 4
    assert not (store / UNPREPARED).exists()


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

    The prepare's environment is the one its node and lane give it, mirrored, as Dew reads nothing of the GPU
    fleet's code (the fleet reads Dew's `tier_wire`, never the reverse): a GPU node's session hides every GPU
    from every job's prepare (its prepare environment, `"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"`),
    and the lane's prepare names itself (tools/armada/ci.py, `gpu_prepared`: `JAX_PLATFORMS` cpu and
    `tier_wire.PREPARE`)."""
    import lane_environment

    from dew.training.ahead import computation_flags

    prepare = lane_environment.configured(
        {"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "DEW_TIER_PREPARE": "1"})
    run = lane_environment.configured({"CUDA_VISIBLE_DEVICES": held, "JAX_PLATFORMS": "cuda"})
    assert computation_flags(prepare["XLA_FLAGS"]) == computation_flags(run["XLA_FLAGS"])
    opened = subprocess.run([sys.executable, "-c", "import jax; print(len(jax.devices('cpu')))"],
                            env={**os.environ, **prepare}, capture_output=True, text=True, timeout=300)
    assert (opened.returncode, opened.stdout.split()[-1:]) == (0, ["1"]), opened.stderr[-2000:]


def test_a_parts_prepare_collects_the_recurrence_tiers_files(tmp_path):
    """A GPU part's prepare as its lane starts it (`lane_environment.configured`: `DEW_TIER_PREPARE=1` on the
    node's CPU, the node an A100 by its serialized topology) collects the recurrence tier's files in a
    process of its own, their imports reading the node's generation as gpu_flow's does (row 9540)."""


    topology = node_topology(tmp_path / "topology.bin")
    prepare = lane_environment.configured({"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu",
                                            tier_wire.PREPARE: "1", model_flows.TOPOLOGY: str(topology),
                                            tier_wire.TRUTHS: str(tmp_path / "tier")})
    # As CI's pytest runs (`pytest_environment`): none of this process's selection or plugins inherited.
    environment = load("session", "tools/armada").pytest_environment(prepare)
    collected = relayed([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                         "tests/test_gpu_recurrence.py", "tests/test_native_certificates.py"],
                        environment, timeout=600)
    assert collected.returncode == 0, collected.output[-3000:]


def test_a_parts_prepare_runs_every_cell_its_nodes_gpu_runs(store, monkeypatch):
    """A GPU part's prepare on its node's CPU (`DEW_TIER_PREPARE=1`, the node an A100 by its serialized
    topology) runs every cell of a flow pinned to the node's generation, Mosaic's among them, as its run
    does (`gpu_flow.run_cells`, `generation_support.runnable` reading the node's generation): none skipped
    for a GPU the prepare's process holds none of (row 7868: the recurrence matrix's five cells skipped
    there, then compiled on the held GPU)."""
    import gpu_flow

    from dew.nn.kernels.generation import compile_only_gpus

    topology = node_topology(store / "topology.bin")
    monkeypatch.setenv(model_flows.TOPOLOGY, str(topology))
    monkeypatch.setenv(tier_wire.PREPARE, "1")
    ran: list[int] = []
    cells = [gpu_flow.Cell(f"cell {index}", functools.partial(ran.append, index), generation="sm80",
                           mosaic=index % 2 == 1) for index in range(5)]
    # The node's compile-only GPU the default device, as a part's prepare holds it (`compiled_ahead`).
    with jax.default_device(compile_only_gpus("sm80", 1)[0]):
        assert gpu_flow.run_cells(cells) == []
    assert ran == list(range(5))


def test_a_truth_run_as_given_computes_in_no_prepare_and_is_held_to_float64_in_its_run(tmp_path):
    """A site whose float64 truth the gate runs as given (`as_given`: host work, NumPy, torch) has nothing
    computed of it in its prepare, on no device: a GPU part's prepare computes no dense math on its node's
    CPU (bible section 8), and on its default device, a compile-only GPU, the truth's eager operations
    raised "'Executable' object has no attribute 'local_devices'" (row 4881's class); its run holds it to
    float64 on its first draw (`_of_float64`), a float32 truth refused there."""
    from arm_forms import as_given
    from reference_error import NarrowTruth

    dew, reference, draw = _site()
    called = []

    def truth(a, b):
        called.append(None)
        return reference(a, b)

    with compiled_truths(tmp_path, prepare=True, stamp="a"):
        assert_as_exact(dew, reference, draw, "a product of a given truth", seeds=8, truth_fn=as_given(truth))
    assert not called
    with compiled_truths(tmp_path, prepare=False, stamp="a"):
        assert_as_exact(dew, reference, draw, "a product of a given truth", seeds=8, truth_fn=as_given(truth))
    assert called

    def narrow(a, b):
        return np.asarray(reference(a, b), np.float32)

    for prepare in (True, False):
        with (compiled_truths(tmp_path / "narrow", prepare=prepare, stamp="a"),
              contextlib.nullcontext() if prepare else pytest.raises(NarrowTruth)):
            assert_as_exact(dew, reference, draw, "a product of a narrow truth", seeds=8,
                            truth_fn=as_given(narrow))


@pytest.mark.mesh(devices=2)
def test_a_host_placed_arm_is_prepared_and_run_on_the_host(tmp_path, monkeypatch):
    """A site-free arm placed on the host's CPU (`ahead_arm(host=True)`: a GPU host's CPU-placed computation,
    row 8376's) is traced in its prepare placed on the host's CPU, whatever the store's device (a GPU part's
    compile-only GPU), and compiled for it; its run, in a process listing another device first as a GPU's
    does, computes it from that program, tracing none, the same values as computing it; the memory a store
    files for its devices (`truths_ahead.compiled_memory`) holds none of it."""
    from truths_ahead import CompiledTruths, compiled_memory

    traced = []

    def scaled(a, b):
        traced.append(jax.config.jax_default_device)
        return a * b + 1.0

    a, b = (np.arange(6, dtype=np.float32).reshape(2, 3) + offset for offset in (1.0, 0.5))
    host = jax.devices("cpu")[0]
    monkeypatch.setattr(CompiledTruths, "device", lambda store: jax.devices()[1])
    with compiled_truths(tmp_path, prepare=True, stamp="a"):
        ahead_arm(jax.jit(scaled), "a host-placed product", host=True)(a, b)
        # Traced, and retraced where its compile lowers it, on the host's CPU alone.
        assert traced and set(traced) == {host}
        ahead_arm(jax.jit(lambda a, b: a * b), "a device product")(a, b)
    # The store's needs its devices' programs' alone: none of the host's arm (the review).
    assert len(compiled_memory(tmp_path, arms=True)) == 1
    traced.clear()
    listed = jax.devices
    monkeypatch.setattr(jax, "devices", lambda backend=None: listed(backend) if backend else listed()[1:])
    with compiled_truths(tmp_path, prepare=False, stamp="a"):
        ran = ahead_arm(jax.jit(scaled), "a host-placed product", host=True)(a, b)
    assert not traced and np.asarray(ran).tolist() == (a * b + 1.0).tolist()


CHILD_ARM = """
import json
import jax
import numpy as np
from reference_error import ahead_arm
from test_model_flows import CHILD_TRACED, child_scaled
from truths_ahead import of_the_parent

a = np.arange(6, dtype=np.float32)
with of_the_parent():
    made = ahead_arm(jax.jit(child_scaled), "a child's product")(a, a)
print(json.dumps({"traced": len(CHILD_TRACED), "made": np.asarray(made).tolist()}))
"""
"""A child process's site-free arm (`test_a_child_process_prepares_and_runs_from_its_tests_store`), a function
of this module, which its statement names (`truths_ahead.arm_role`)."""

CHILD_TRACED: list[None] = []
"""Each trace of `child_scaled` in its process."""


def child_scaled(a, b):
    """A child process's arm (`CHILD_ARM`), its traces counted."""
    CHILD_TRACED.append(None)
    return a * b + 1.0


def test_a_child_process_prepares_and_runs_from_its_tests_store(tmp_path):
    """A computation that needs a process of its own (a backend option set once a process: the approximate
    divider's), run in a child of its test, opens its test's store (`truths_ahead.for_a_child`,
    `of_the_parent`): in the test's prepare it traces and compiles its arm into that store, giving zeros;
    in the test's run it computes the arm from that program, tracing none, its sites the test's, none of
    them unprepared."""
    from process_support import fresh
    from truths_ahead import for_a_child

    a = np.arange(6, dtype=np.float32)
    with compiled_truths(tmp_path, prepare=True, stamp="a"):
        prepared = fresh(CHILD_ARM, variables=for_a_child())
    with compiled_truths(tmp_path, prepare=False, stamp="a"):
        ran = fresh(CHILD_ARM, variables=for_a_child())
    assert prepared["traced"] and prepared["made"] == [0.0] * 6
    assert ran == {"traced": 0, "made": (a * a + 1.0).tolist()} and not (tmp_path / UNPREPARED).exists()


def test_a_site_whose_prepare_ended_before_its_arms_runs_them_unprepared(tmp_path, monkeypatch):
    """A site whose prepare wrote its truths and then failed before its arms' (a flow's cell its prepare
    deferred, `gpu_flow.run_cells`, its store still serving the run) runs its arms unprepared in its run,
    recorded, judging the site as one computing its own does, where its run found no program
    (`MissingTruth`, the review)."""
    import arm_store

    dew, reference, draw = _site()
    with compiled_truths(tmp_path, prepare=True, stamp="a"), monkeypatch.context() as failing:
        def ended(*arguments, **options):
            raise RuntimeError("the prepare ended before the site's arms")

        failing.setattr(arm_store, "_compile_arms", ended)
        with pytest.raises(RuntimeError):
            assert_as_exact(dew, reference, draw, "a product whose arms went unprepared", seeds=8)
    with compiled_truths(tmp_path, prepare=False, stamp="a"):
        assert_as_exact(dew, reference, draw, "a product whose arms went unprepared", seeds=8)
    assert (tmp_path / UNPREPARED).is_file()


def test_a_tests_unprepared_record_names_every_reason(tmp_path):
    """A test's record that it ran unprepared (`truths_ahead.UNPREPARED`) holds every reason it gave, each
    once, its first's its own: two site-free arms no prepare reached, the first called twice, are two
    reasons, where the record kept the first alone and hid the rest (Main: their count shows a tier
    drifting from its prepares)."""
    a = np.arange(6, dtype=np.float32)
    with compiled_truths(tmp_path, prepare=False, stamp="a"):
        for label in ("an unreached product", "an unreached product", "another unreached product"):
            ahead_arm(jax.jit(lambda x: x * 2.0), label)(a)
    record = json.loads((tmp_path / UNPREPARED).read_text())
    assert len(record["reasons"]) == 2 and record["reasons"][0]["error"] == record["error"]


@pytest.mark.mesh(devices=2)
def test_a_walk_its_values_steer_runs_on_the_runs_device_named_at_run(tmp_path, monkeypatch):
    """A site sized by its truth's terms (`terms=True`) whose float64 truth takes a branch its values pick
    (`lax.cond` on its operands), its terms' walk held by its prepare as one its run walks
    (`arm_store._held_walk`), walks it in its run on the run's own device, a GPU's FP64 units, each
    output's terms completed there (`float64_terms._or_size`), and the run names the walk as compiled at run
    (`truths_ahead.AT_RUN`), nothing unprepared: a float64 truth's terms walked on the host's CPU in the run
    was dense float64 on the node's CPU (testing.md, bible section 8; row 8576)."""
    import float64_terms
    from truths_ahead import AT_RUN

    dew, reference, draw = _site()

    def steered(a, b):
        return jax.lax.cond(jnp.sum(a) > 0, reference, reference, a, b)

    walked, sized = [], []
    evaluated, or_size = float64_terms.evaluated, float64_terms._or_size

    def placed(*arguments, **options):
        walked.append(float64_terms.PLACED.get())
        return evaluated(*arguments, **options)

    def placed_size(energy, value):
        sized.append(jax.config.jax_default_device)
        return or_size(energy, value)

    monkeypatch.setattr(float64_terms, "evaluated", placed)
    monkeypatch.setattr(float64_terms, "_or_size", placed_size)
    with compiled_truths(tmp_path, prepare=True, stamp="a"):
        assert_as_exact(dew, reference, draw, "a steered product", seeds=8, truth_fn=steered, terms=True)
    assert not walked
    with compiled_truths(tmp_path, prepare=False, stamp="a"), jax.default_device(jax.devices()[1]):
        assert_as_exact(dew, reference, draw, "a steered product", seeds=8, truth_fn=steered, terms=True)
    assert walked and sized and set(walked) == set(sized) == {jax.devices()[1]}
    assert (tmp_path / AT_RUN).is_file() and not (tmp_path / UNPREPARED).exists()


def test_a_site_whose_truths_alone_are_at_run_computes_its_reference_as_its_truth(tmp_path, monkeypatch):
    """A site naming no truth whose prepare writes its truths at run (`compile_site`'s `at_run`: their
    programs hold a choice its GPU makes) and compiles its arms: its run computes its truth from its
    reference as compiled there (`_site_judged`), its arms from the store, the truths named at run alone,
    nothing unprepared; the truth fell back to the reference arm read from the store, called at float64
    inputs it holds no program for (`MissingTruth`, the review of rows 8576 and 8579)."""
    from truths_ahead import AT_RUN, CompiledTruths

    from dew.nn.kernels.templates import report

    dew, reference, draw = _site()
    real = CompiledTruths.compile_site

    def refusing_truths(store, label, programs, held, **options):
        # A choice its GPU makes, which no compile without that GPU holds, in its truths' programs alone.
        if programs and not options.get("arms", False) and options.get("at_run") is None:
            raise report.CompiledOnDevice(jax.jit(lambda x: x).lower(np.zeros(1, np.float32)), ["truth"])
        return real(store, label, programs, held, **options)

    with compiled_truths(tmp_path, prepare=True, stamp="a"), monkeypatch.context() as at_run:
        at_run.setattr(CompiledTruths, "compile_site", refusing_truths)
        assert_as_exact(dew, reference, draw, "a product whose truths are at run", seeds=8)
    with compiled_truths(tmp_path, prepare=False, stamp="a"):
        assert_as_exact(dew, reference, draw, "a product whose truths are at run", seeds=8)
    assert len(json.loads((tmp_path / AT_RUN).read_text())["reasons"]) == 1
    assert not (tmp_path / UNPREPARED).exists()
