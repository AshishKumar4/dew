"""A GPU flow's runner (tests/gpu_flow.py) on cells whose outcomes are known: every cell runs, every
failure is kept and raised together, a cell's own skip is reported and the cells after it still run,
and a flow of no cell that runs is skipped, never passed."""

import inspect

import gpu_flow
import pytest
from generation_support import node_topology
from gpu_flow import Cell, CellsFailed, cells_of, run_cells


def test_a_flow_runs_every_cell_and_raises_every_failure_together():
    """A failing cell is kept, the cells after it run, and every failure is raised together after the
    last, each by its label."""
    ran = []
    raised_at: list[int] = []

    def failing(name):
        def run():
            ran.append(name)
            raised_at.append(inspect.currentframe().f_lineno + 1)
            raise AssertionError(f"{name} is off")
        return run

    with pytest.raises(CellsFailed) as raised:
        run_cells([Cell("first", failing("first")), Cell("passing", lambda: ran.append("passing")),
                   Cell("last", failing("last"))])
    assert ran == ["first", "passing", "last"]
    assert raised.value.labels == ["first", "last"]
    # Each failure kept with the frame it was raised at, the cell's own (its report names where it failed).
    assert [raised.value.where[label][-1] for label in ("first", "last")] == [(__file__, line)
                                                                              for line in raised_at]


def test_a_cell_that_skips_itself_is_reported_and_the_cells_after_it_run():
    """A cell's own skip is reported by its label and ends neither the flow nor the cells after it."""
    ran = []
    skipped = run_cells([Cell("absent", lambda: pytest.skip("its package is missing")),
                         Cell("present", lambda: ran.append("present"))])
    assert ran == ["present"] and skipped == ["absent"]


def test_a_flow_of_no_cell_that_runs_is_skipped():
    """A flow none of whose cells ran is skipped, never passed."""
    with pytest.raises(pytest.skip.Exception):
        run_cells([Cell("absent", lambda: pytest.skip("its package is missing"))])


ran_with = []
TOO_WIDE = True


FORMS = {"p": ("plain", 1.0), "s": ("scaled", 2.0)}


@pytest.mark.parametrize("width", [64, pytest.param(128, marks=pytest.mark.skipif(TOO_WIDE, reason="wide"))])
@pytest.mark.parametrize(("form", "scale"), FORMS.values(), ids=FORMS)
def gridded(form, scale, width, fixture):
    """A cell function whose test kept its grid in its own marks."""
    ran_with.append((form, scale, width, fixture))


def test_a_cell_function_gives_a_cell_a_parameter_set_and_skips_where_its_set_does():
    """`cells_of`: the product of the function's own parametrize marks, each set run with its arguments and
    the fixtures given, a set whose skipif holds skipped by its reason."""
    cells = cells_of(gridded, fixture="given")
    assert sorted(cell.label for cell in cells) == ["gridded: 128-p", "gridded: 128-s", "gridded: 64-p",
                                                    "gridded: 64-s"]
    ran_with.clear()
    skipped = run_cells(cells)
    assert sorted(ran_with) == [("plain", 1.0, 64, "given"), ("scaled", 2.0, 64, "given")]
    assert sorted(skipped) == ["gridded: 128-p", "gridded: 128-s"]


def test_a_flow_reports_its_counts_once_however_its_cells_nest():
    """A flow's counts (`gpu_flow.REPORTED`, the JUnit property `cells` CI reads): an outermost flow's alone,
    a sweep's inner flow counted within its cell, and a flow none of whose cells ran reported with none
    passed before it skips, so it reads as unqualified rather than green."""
    gpu_flow.REPORTED.clear()
    run_cells([Cell("a sweep", lambda: run_cells([Cell("one", lambda: None), Cell("two", lambda: None)])),
               Cell("absent", lambda: pytest.skip("not here"))])
    with pytest.raises(pytest.skip.Exception):
        run_cells([Cell("absent", lambda: pytest.skip("not here"))])
    assert [{key: run[key] for key in ("passed", "failed", "skipped")} for run in gpu_flow.REPORTED] == [
        {"passed": 1, "failed": 0, "skipped": 1}, {"passed": 0, "failed": 0, "skipped": 1}]
    gpu_flow.REPORTED.clear()


def test_a_cell_failing_by_pytests_own_failures_ends_no_flow_and_is_counted():
    """A cell's `pytest.fail`, and a `pytest.raises` that saw nothing, are its failures (they derive from
    BaseException): each is kept and raised with the rest, the cells after it still run, and the flow's
    counts are reported (`gpu_flow.REPORTED`) however it ends."""
    ran = []

    def unraised():
        with pytest.raises(ValueError):
            ran.append("unraised")

    gpu_flow.REPORTED.clear()
    with pytest.raises(CellsFailed) as raised:
        run_cells([Cell("failed", lambda: pytest.fail("its pool never ended")), Cell("unraised", unraised),
                   Cell("after", lambda: ran.append("after"))])
    assert raised.value.labels == ["failed", "unraised"] and ran == ["unraised", "after"]
    assert [{key: run[key] for key in ("passed", "failed", "skipped")} for run in gpu_flow.REPORTED] == [
        {"passed": 1, "failed": 2, "skipped": 0}]
    gpu_flow.REPORTED.clear()


def test_a_repeated_step_holds_its_bits_and_a_changed_one_fails():
    """`gpu_flow.repeats_bit_for_bit`: a step that gives the same bits each run passes, compiled afresh after
    its first; one whose bits change between runs fails."""
    import jax.numpy as jnp

    gpu_flow.repeats_bit_for_bit(lambda x: {"sum": x + 1, "product": x * 2}, jnp.arange(4.0), runs=3)
    counter = iter(range(10))

    def drifting(x):
        return {"drift": x + next(counter) * 1e-6}

    with pytest.raises(AssertionError):
        gpu_flow.repeats_bit_for_bit(drifting, jnp.arange(4.0))


def test_an_isolated_cell_takes_its_own_directory_and_the_suites_per_test_settings(tmp_path_factory):
    """`gpu_flow.isolated`: a cell runs in a directory of its own, with its own rung records, data cache and
    Hub answers beneath it, the fixtures it names passed, and each setting undone after it."""
    import os

    seen = {}

    def check(tmp_path, monkeypatch):
        seen.update(directory=tmp_path, rungs=os.environ["DEW_RUNG_RECORDS"],
                    cache=os.environ["DEW_DATA_CACHE"], answers=os.environ["DEW_HUB_ANSWERS"],
                    patch=monkeypatch)

    before = os.environ.get("DEW_RUNG_RECORDS")
    gpu_flow.isolated(tmp_path_factory)(check, "own")()
    directory = seen["directory"]
    assert seen["rungs"] == str(directory / "rungs") and seen["cache"] == str(directory / "data-cache")
    assert seen["answers"] == str(directory / "hub-answers") and os.environ.get("DEW_RUNG_RECORDS") == before



def test_a_mosaic_cell_runs_only_where_the_device_lowers_its_generations_copies():
    """On an H100, a cell of an A100's Triton kernels runs, and one of its Mosaic GPU kernels is skipped:
    Mosaic GPU lowers copies by the device, TMA on an H100, so the A100's cp.async pipelines are not
    what it runs (`generation_support.lowers_as`). An H100's own Mosaic GPU cell runs."""
    from dew.nn.kernels import generation

    ran = []
    with generation.assumed("sm90"):
        skipped = run_cells([Cell("triton sm80", lambda: ran.append("triton sm80"), "sm80"),
                             Cell("mosaic sm80", lambda: ran.append("mosaic sm80"), "sm80", mosaic=True),
                             Cell("mosaic sm90", lambda: ran.append("mosaic sm90"), "sm90", mosaic=True)])
    assert ran == ["triton sm80", "mosaic sm90"]
    assert skipped == ["mosaic sm80"]


def test_a_classs_covering_holds_every_native_value_and_pair_and_every_named_fault():
    """A class's covering (`gpu_flow.covering`) keeps each cell named for a GPU-only fault, and of the rest
    the fewest whose native axes hold every value and every pair of values the class's cells hold, each
    generation apart; a cell no native axis and no fault distinguishes is the CPU's."""
    def cell(label, generation=None, fault=None, **native):
        return Cell(label, lambda: None, generation, native=native, fault=fault)

    cells = [cell("bf16 forward", launch="forward", dtype="bfloat16"),
             cell("bf16 forward, another shape", launch="forward", dtype="bfloat16"),
             cell("fp16 forward", launch="forward", dtype="float16"),
             cell("bf16 backward", launch="backward", dtype="bfloat16"),
             cell("fp16 backward", launch="backward", dtype="float16"),
             cell("fp16 backward, a hook", launch="backward", dtype="float16"),
             cell("an uneven walk", launch="forward", dtype="bfloat16", fault="uneven walks"),
             cell("an A100's forward", "sm80", launch="forward", dtype="bfloat16")]
    kept = {kept.label for kept in gpu_flow.covering(cells)}
    assert {"an uneven walk", "an A100's forward", "fp16 forward", "bf16 backward"} <= kept
    assert len(kept & {"fp16 backward", "fp16 backward, a hook"}) == 1
    assert not kept & {"bf16 forward", "bf16 forward, another shape"}
    mine = [cell for cell in cells if cell.generation is None]
    held = set().union(*(gpu_flow.held(cell.native) for cell in mine if cell.label in kept))
    assert held >= set().union(*(gpu_flow.held(cell.native) for cell in mine))


def test_a_matrix_case_pinned_to_the_nodes_generation_runs_in_its_prepare(tmp_path, monkeypatch):
    """A matrix case pinned to its generation (`generation_support.pinned`, as every *_matrix.py pins its
    cases) runs in a GPU part's prepare on the node's CPU when the node is that generation, as its run runs
    it, and skips there when it is another (rows 7868, 7886: pins read off the CPU at import skipped every
    pinned case in the prepare, then compiled them on the held GPU)."""
    import generation_support
    import model_flows
    import tier_wire


    topology = node_topology(tmp_path / "topology.bin")
    monkeypatch.setenv(model_flows.TOPOLOGY, str(topology))
    monkeypatch.setenv(tier_wire.PREPARE, "1")
    ran: list[str] = []

    # An unpinned case beside the pinned ones, so the flow runs and its test fails, never skips, where the
    # pins skip both.
    @pytest.mark.parametrize("case", [*(pytest.param(case, marks=generation_support.pinned(case))
                                        for case in ("sm80", "sm90")), "unpinned"])
    def a_case(case):
        ran.append(case)

    run_cells(cells_of(a_case))
    assert ran == ["sm80", "unpinned"]


def test_a_cell_that_computes_in_a_parts_prepare_is_deferred_to_its_run(tmp_path, monkeypatch):
    """In a GPU tier part's prepare (`tier_wire.preparing`, its store open), a cell that raises, computing
    where nothing computes, is recorded deferred, where it raised (`tier_wire.DEFERRED`), and the flow
    passes its prepare, its other cells run; outside a prepare the same cell fails its flow (rows 8505 and
    8506: such a cell beside a flow's prepared sites turned its test red in the prepare)."""
    import tier_wire
    from tier_wire import DEFERRED
    from truths_ahead import UNPREPARED, compiled_truths

    def computing():
        raise AttributeError("'jaxlib._jax.Executable' object has no attribute 'local_devices'")

    ran = []
    flow = [Cell("computes", computing), Cell("prepared", lambda: ran.append("prepared"))]
    monkeypatch.setenv(tier_wire.PREPARE, "1")
    with compiled_truths(tmp_path, prepare=True, stamp="a"):
        run_cells(flow)
    # Recorded deferred, never unprepared, which would take the test's run off its store (tests/conftest.py).
    assert ran == ["prepared"] and (tmp_path / DEFERRED).is_file() and not (tmp_path / UNPREPARED).exists()
    monkeypatch.delenv(tier_wire.PREPARE)
    with pytest.raises(CellsFailed) as raised:
        run_cells(flow)
    assert raised.value.labels == ["computes"]


def test_two_cells_preparing_one_site_label_fail_the_prepare(tmp_path, monkeypatch):
    """In a GPU tier part's prepare, a second site of a label another cell's prepare began
    (`truths_ahead.SiteCollision`) fails its flow there, never deferred to the run, which would read the
    first site's programs as its own (row 13030: two attention-stability cells of one label, the second's
    run a MissingTruth); the first cell's prepare passes."""
    import store_wire
    import tier_wire
    from truths_ahead import compiled_truths

    def site():
        store_wire.COMPILED.get().compile_site("one label", {}, {})

    monkeypatch.setenv(tier_wire.PREPARE, "1")
    with compiled_truths(tmp_path, prepare=True, stamp="a"), pytest.raises(CellsFailed) as raised:
        run_cells([Cell("first", site), Cell("second", site)])
    assert raised.value.labels == ["second"]


def test_a_cell_past_its_prepares_cap_is_deferred_to_its_run_and_not_run(tmp_path, monkeypatch):
    """In a part's prepare past its cap (`tier_wire.PREPARE_BY`), a flow runs the cells before it and defers
    each after, unrun, to its run (`tier_wire.DEFERRED`, where `tier_wire.CAPPED`), its flow passing its
    prepare: the cells it prepared kept, those it left named unprepared."""
    import json
    import time

    import tier_wire
    from truths_ahead import compiled_truths

    ran = []

    def first():
        ran.append("first")
        monkeypatch.setenv(tier_wire.PREPARE_BY, repr(time.monotonic() - 1.0))

    flow = [Cell("first", first), Cell("second", lambda: ran.append("second")),
            Cell("third", lambda: ran.append("third"))]
    monkeypatch.setenv(tier_wire.PREPARE, "1")
    monkeypatch.setenv(tier_wire.PREPARE_BY, repr(time.monotonic() + 600.0))
    with compiled_truths(tmp_path, prepare=True, stamp="a"):
        run_cells(flow)
    assert ran == ["first"]
    record = json.loads((tmp_path / tier_wire.DEFERRED).read_text())
    assert [(reason["where"], reason["cell"]) for reason in record["reasons"]] == [
        (tier_wire.CAPPED, "second"), (tier_wire.CAPPED, "third")]


def test_a_test_opening_its_own_store_past_its_prepares_cap_is_named_unprepared_and_not_run(tmp_path):
    """In a part's prepare past its cap, a test marked `prepares` (a flow's, which opens its own store, as
    tests/test_gpu_flows.py's does, and starts its compile in its body) is skipped before its body, named
    unprepared where its prepare's records are read (tests/conftest.py's `compiled_ahead`, where
    `tier_wire.CAPPED`), never run past the cap."""
    import json
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    import tier_wire
    from truths_ahead import UNPREPARED

    ran = tmp_path / "ran"
    (tmp_path / "test_own_store.py").write_text(
        "import pathlib\nimport pytest\n\n\n@pytest.mark.prepares\ndef test_flow():\n"
        f"    pathlib.Path({str(ran)!r}).write_text('ran')\n")
    tests = Path(__file__).parent
    # The node's one GPU, as its driver lists it (a prepare counts its run's devices by it).
    tools = tmp_path / "bin"
    tools.mkdir()
    (tools / "nvidia-smi").write_text("#!/bin/sh\necho 'GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-0)'\n")
    (tools / "nvidia-smi").chmod(0o755)
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "--rootdir", str(tmp_path),
         "--junitxml", str(tmp_path / "report.xml"),
         "-c", str(tests.parent / "pyproject.toml"), "-p", "conftest", str(tmp_path / "test_own_store.py")],
        env={**os.environ, "PATH": f"{tools}:{os.environ['PATH']}", "JAX_PLATFORMS": "cpu",
             "PYTHONPATH": str(tests), tier_wire.PREPARE: "1",
             tier_wire.TRUTHS: str(tmp_path / "truths"), tier_wire.PREPARE_BY: repr(time.monotonic() - 1.0)},
        capture_output=True, text=True, timeout=600, cwd=tests.parent)
    import xml.etree.ElementTree as ElementTree

    suite = ElementTree.parse(tmp_path / "report.xml").getroot().find("testsuite")
    assert done.returncode == 0 and suite is not None
    assert (suite.get("tests"), suite.get("skipped")) == ("1", "1")
    assert not ran.exists()
    [record] = list((tmp_path / "truths").rglob(UNPREPARED))
    held = json.loads(record.read_text())
    assert (held["test"].endswith("test_own_store.py::test_flow"),
            [reason["where"] for reason in held["reasons"]]) == (True, [tier_wire.CAPPED])


def test_a_cell_raising_after_its_prepare_compiled_its_programs_passes_it_unrecorded(tmp_path, monkeypatch):
    """In a GPU tier part's prepare, a cell that compiled a site and then raised, checking what its prepare
    gives in its programs' place (`ahead_arm`'s zeros), passes its prepare with nothing recorded, its run
    judging it (row 8555: every such cell was recorded deferred and its test named unprepared, falsely);
    a cell whose site's prepare did not finish, its compile raised, its arm's trace raised after its
    truths were written, or a second check's truth trace raised after a first was prepared whole, is
    deferred, its failure recorded where it raised (the review)."""
    import arm_store
    import jax
    import numpy as np
    import tier_wire
    from reference_error import ahead_arm, assert_as_exact
    from tier_wire import DEFERRED
    from truths_ahead import UNPREPARED, CompiledTruths, compiled_truths

    def checked():
        made = ahead_arm(jax.jit(lambda x: x + 1.0), "a checked product")(np.zeros(3, np.float32))
        assert np.asarray(made).tolist() == [1.0, 1.0, 1.0]

    monkeypatch.setenv(tier_wire.PREPARE, "1")
    with compiled_truths(tmp_path, prepare=True, stamp="a"):
        run_cells([Cell("checks its product", checked)])
    assert not (tmp_path / DEFERRED).exists() and not (tmp_path / UNPREPARED).exists()

    def unfinished(store, programs, compiled, host=False):
        raise RuntimeError("its compile failed")

    with monkeypatch.context() as failing:
        failing.setattr(CompiledTruths, "_compiled", unfinished)
        failed = ahead_arm(jax.jit(lambda x: x * 2.0), "a failed product")
        with compiled_truths(tmp_path / "unfinished", prepare=True, stamp="a"):
            run_cells([Cell("compiles nothing", lambda: failed(np.zeros(3, np.float32)))])
    assert (tmp_path / "unfinished" / DEFERRED).is_file()

    # A site whose truths are written and whose arm's trace then raises.
    def traced_badly(*arguments, **options):
        raise RuntimeError("its arm's trace failed")

    def site():
        assert_as_exact(jax.jit(lambda x: x * 2.0), jax.jit(lambda x: x * 2.0),
                        lambda seed: (np.full(3, seed, np.float32),), "a product whose arm failed", seeds=8)

    with monkeypatch.context() as failing:
        failing.setattr(arm_store, "_arm_program", traced_badly)
        with compiled_truths(tmp_path / "arms", prepare=True, stamp="a"):
            run_cells([Cell("traces its arm badly", site)])
    assert (tmp_path / "arms" / DEFERRED).is_file()

    # A check prepared whole, then a second check whose truth's trace raises before it compiles any site.
    def two_checks():
        ahead_arm(jax.jit(lambda x: x + 1.0), "a first product")(np.zeros(3, np.float32))

        def truth(x):
            raise RuntimeError("its truth's trace failed")

        assert_as_exact(jax.jit(lambda x: x * 3.0), jax.jit(lambda x: x * 3.0),
                        lambda seed: (np.full(3, seed, np.float32),), "a product whose truth failed", seeds=8,
                        truth_fn=truth)

    with compiled_truths(tmp_path / "second", prepare=True, stamp="a"):
        run_cells([Cell("prepares one check and fails the next", two_checks)])
    assert (tmp_path / "second" / DEFERRED).is_file()


def test_a_flow_module_importing_gpu_flow_first_imports_in_a_parts_prepare():
    """In a GPU tier part's prepare (`tier_wire.PREPARE`) a module importing gpu_flow first imports: gpu_flow
    read the generation helpers at its import, which in a prepare read model_flows' node target, whose
    module imports gpu_flow, a circular import that refused such a module in every prepare and left its
    sites unprepared (the native certificates on a GH200 tier)."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    import tier_wire

    done = subprocess.run([sys.executable, "-c", "import gpu_flow"], cwd=Path(__file__).parent,
                          env={**os.environ, tier_wire.PREPARE: "1"}, capture_output=True, text=True,
                          check=False)
    assert done.returncode == 0, done.stderr[-2000:]



def test_a_cell_compiling_at_its_run_is_passed_by_in_a_parts_prepare(tmp_path, monkeypatch):
    """A cell with no check-1 site that declares its programs compiled at its run (`Cell.at_run`: its GPU's
    driver facts, a fresh process) is passed by in a GPU tier part's prepare, neither run nor recorded
    deferred, and its flow passes its prepare; outside a prepare it runs, its failure its flow's (rows 8074,
    8635: every such cell was deferred and its test named unprepared)."""
    import tier_wire
    from tier_wire import DEFERRED
    from truths_ahead import compiled_truths

    ran = []

    def facts():
        ran.append("facts")
        raise AssertionError("no driver facts here")

    flow = [Cell("driver facts", facts, at_run="its GPU's driver facts")]
    monkeypatch.setenv(tier_wire.PREPARE, "1")
    with compiled_truths(tmp_path, prepare=True, stamp="a"):
        run_cells(flow)
    assert ran == [] and not (tmp_path / DEFERRED).exists()
    monkeypatch.delenv(tier_wire.PREPARE)
    with pytest.raises(CellsFailed) as raised:
        run_cells(flow)
    assert ran == ["facts"] and raised.value.labels == ["driver facts"]


def test_a_parts_prepare_chooses_kernels_as_its_nodes_gpu(tmp_path, monkeypatch):
    """In a GPU tier part's prepare (`truths_ahead.prepared`, its node an H100) and in a child process of its
    test's (`truths_ahead.of_the_parent`, its store the test's: `for_a_child`), a call reads the node's
    generation and facts, as its run on that GPU does, where its compile-only device, placed off the CPU
    backend, read the bare platform 'gpu' and every kernel that GPU takes refused (rows 8076, 10088; the
    child's, row 10765)."""
    import model_flows
    from generation_support import node_target
    from truths_ahead import for_a_child, of_the_parent, prepared

    from dew.nn.kernels.generation import device_generation, gpu_target

    def reads_the_nodes_gpu():
        target = gpu_target()
        return (device_generation(), target.generation, target.multiprocessors) == ("sm90", "sm90", 132)

    monkeypatch.setenv(model_flows.TOPOLOGY, str(node_topology(tmp_path / "topology.bin", gpu="sm90")))
    with prepared(tmp_path / "store", stamp="a", target=node_target()):
        assert reads_the_nodes_gpu()
        child = for_a_child()
    for name, value in child.items():
        monkeypatch.setenv(name, value)
    with of_the_parent():
        assert reads_the_nodes_gpu()


def test_a_parts_prepare_prepares_its_child_processs_arms(tmp_path, monkeypatch):
    """In a GPU tier part's prepare (its node an H100), a check run in a process of its own, the approximate
    divider's (`test_precision_policy.divides_as_ieee_division_on_an_approximate_divider`), prepares its arms
    in its test's store (`truths_ahead.for_a_child`, `of_the_parent`), its failure on their zeros passing the
    prepare: it was passed by, its run compiling every arm on its held GPU, unprepared and unrecorded, and its
    child computed in the prepare (row 10748)."""
    import model_flows
    import test_precision_policy
    import tier_wire
    from arm_store import armed
    from generation_support import node_target
    from truths_ahead import CompiledTruths, prepared

    monkeypatch.setenv(model_flows.TOPOLOGY, str(node_topology(tmp_path / "topology.bin", gpu="sm90")))
    monkeypatch.setenv(tier_wire.PREPARE, "1")
    with prepared(tmp_path / "store", stamp="a", target=node_target()):
        test_precision_policy.divides_as_ieee_division_on_an_approximate_divider()
    edges = test_precision_policy.EDGES + test_precision_policy.APPROXIMATELY
    assert CompiledTruths(tmp_path / "store", prepare=False, stamp="a").holds(armed(edges))


def test_a_parts_prepare_draws_an_nvfp4_tiles_values_on_the_host(tmp_path, monkeypatch):
    """In a GPU tier part's prepare (its node an H100), the NVFP4 tile of quantized values
    (`test_algebra_bits._quantized_values`) is drawn and its arms prepared, nothing computed on the
    compile-only GPU its default device is: its draw ran eagerly there and raised "'Executable' object has no
    attribute 'local_devices'", naming its test unprepared (row 8617)."""
    import model_flows
    import test_algebra_bits
    import tier_wire
    from arm_store import armed
    from generation_support import node_target
    from truths_ahead import CompiledTruths, prepared

    monkeypatch.setenv(model_flows.TOPOLOGY, str(node_topology(tmp_path / "topology.bin", gpu="sm90")))
    monkeypatch.setenv(tier_wire.PREPARE, "1")
    with prepared(tmp_path / "store", stamp="a", target=node_target()):
        assert test_algebra_bits._quantized_values().shape == (256, 64)
    # Each of its arms prepared in the store, as the run reads them (row 11747).
    store = CompiledTruths(tmp_path / "store", prepare=False, stamp="a")
    assert all(store.holds(armed(f"an NVFP4 tile's values, {step}")) for step in ("quantized", "decoded"))


def test_a_parts_prepare_prepares_the_apg_edges_arms(tmp_path, monkeypatch):
    """In a GPU tier part's prepare (its node an H100), the numerics flow's APG cell
    (`test_guidance.apg_holds_its_projection_at_its_ranges_edges`) prepares each edge's arm in the store: its
    arm closed over the guidance's process, an object no statement holds, and the GH200 prepare named the cell
    unprepared, "its arm has no statement of what it computes" (DC-numerics-vocab-1-gh200)."""
    import model_flows
    import test_guidance
    import tier_wire
    from arm_store import armed
    from generation_support import node_target
    from gpu_flow import Cell, run_cells
    from truths_ahead import CompiledTruths, prepared

    monkeypatch.setenv(model_flows.TOPOLOGY, str(node_topology(tmp_path / "topology.bin", gpu="sm90")))
    monkeypatch.setenv(tier_wire.PREPARE, "1")
    with prepared(tmp_path / "store", stamp="a", target=node_target()):
        run_cells([Cell("APG at its range's edges",
                        test_guidance.apg_holds_its_projection_at_its_ranges_edges)])
    store = CompiledTruths(tmp_path / "store", prepare=False, stamp="a")
    assert all(store.holds(armed(f"APG at {name}")) for name in test_guidance.APG_EDGES)


def test_a_parts_prepare_compiles_a_site_calling_the_host_at_its_run(tmp_path, monkeypatch):
    """In a GPU tier part's prepare (its node an H100, the process's backend a CPU's), a site whose program
    calls back to the host (a reference through `jax.pure_callback`, as Dory's torch gradients) is written to
    compile at its run on its GPU: a host callback is lowered by the backend it runs on, absent here, and the
    prepare raised "Unknown backend: 'gpu' requested" (row 11497)."""
    import jax
    import jax.numpy as jnp
    import model_flows
    import numpy as np
    from generation_support import node_target
    from truths_ahead import prepared

    shape = jax.ShapeDtypeStruct((3,), jnp.float32)

    def doubled(x):
        return jax.pure_callback(lambda value: np.asarray(value) * 2, shape, x)

    monkeypatch.setenv(model_flows.TOPOLOGY, str(node_topology(tmp_path / "topology.bin", gpu="sm90")))
    with prepared(tmp_path / "store", stamp="a", target=node_target()) as store:
        store.compile_site("a site calling the host", {"doubled": (doubled, [shape])}, {})
        assert store.at_run("a site calling the host") is not None


def test_a_parts_prepare_passes_by_its_cells_placed_on_the_hosts_cpu(tmp_path, monkeypatch):
    """In a GPU tier part's prepare (its node an H100), the numerics flows' cells placed on the host's CPU, a
    projection and the norm seams, are passed by, none recorded deferred: the prepare chooses every kernel as
    on its GPU, so their CPU programs took Triton kernels that do not lower on a CPU, raised, and named their
    tests unprepared (row 10764)."""
    import model_flows
    import test_gpu_numerics
    import tier_wire
    from generation_support import node_target
    from tier_wire import DEFERRED
    from truths_ahead import prepared

    monkeypatch.setenv(model_flows.TOPOLOGY, str(node_topology(tmp_path / "topology.bin", gpu="sm90")))
    monkeypatch.setenv(tier_wire.PREPARE, "1")
    with prepared(tmp_path / "store", stamp="a", target=node_target()):
        run_cells([*test_gpu_numerics.projection_cells(), *test_gpu_numerics.seams_cells(None)])
    assert not (tmp_path / "store" / DEFERRED).exists()


def test_a_prepared_sites_draw_whose_shape_reads_its_value_keys_its_own_program():
    """A site's prepare reads each draw's shapes traced abstractly and draws in full one seed of each shapes
    (`arm_store._first_draws`): the draws it keeps name exactly the programs every draw drawn in full names,
    each its first seed's draw. A draw whose shape reads a value it drew (a length from its seed's key) is
    drawn in full for its shapes, every seed of it, and warned of (`ConcreteDraws`), each seed of other
    shapes keying its own program; one of fixed shapes keeps its first seed's draw alone, warned of by none
    (row 12431: a Gemma vocabulary's 1,280 draws of a 64 MiB head made in full for one program)."""
    import warnings

    import jax
    import jax.numpy as jnp
    import numpy as np
    from arm_arithmetic import drawn_on_the_host
    from arm_store import ConcreteDraws, _first_draws
    from truths_ahead import named

    def fixed(seed):
        return jax.random.normal(jax.random.key(seed), (4, 3)), jnp.arange(2)

    def read(seed):
        return (jnp.ones((int(jax.random.randint(jax.random.key(seed), (), 1, 4)), 3)),)

    seeds = range(12)
    for draw, concrete in ((fixed, 0), (read, len(seeds))):
        every: dict[str, int] = {}
        for seed in seeds:
            every.setdefault(named("truth", drawn_on_the_host(draw, seed)), seed)
        with warnings.catch_warnings(record=True) as warned:
            warnings.simplefilter("always")
            kept = _first_draws(draw, seeds, "a site")
        programs: dict[str, tuple] = {}
        for inputs in kept:
            programs.setdefault(named("truth", inputs), inputs)
        assert list(programs) == list(every)
        for name, inputs in programs.items():
            jax.tree.map(np.testing.assert_array_equal, inputs, drawn_on_the_host(draw, every[name]))
        assert len(kept) == (concrete or 1) and (len(every) > 1 if concrete else len(every) == 1)
        assert [(item.category, f"{concrete} of its draws" in str(item.message)) for item in warned] == (
            [(ConcreteDraws, True)] if concrete else [])
