"""A GPU flow's runner (tests/gpu_flow.py) on cells whose outcomes are known: every cell runs, every
failure is kept and raised together, a cell's own skip is reported and the cells after it still run,
and a flow of no cell that runs is skipped, never passed."""

import inspect

import gpu_flow
import pytest
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
    from jax.experimental import topologies

    from dew.nn.kernels.generation import compile_only_gpus

    topology = tmp_path / "topology.bin"
    topology.write_bytes(topologies.TopologyDescription(list(compile_only_gpus("sm80", 1))).serialize())
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
    where nothing computes, is recorded deferred, where it raised (`truths_ahead.DEFERRED`), and the flow
    passes its prepare, its other cells run; outside a prepare the same cell fails its flow (rows 8505 and
    8506: such a cell beside a flow's prepared sites turned its test red in the prepare)."""
    import tier_wire
    from truths_ahead import DEFERRED, UNPREPARED, compiled_truths

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
