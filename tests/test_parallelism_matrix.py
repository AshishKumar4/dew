"""Every model family on every objective class it takes, split by the layouts
that split it differently, against one device on the eight-device CPU mesh.

Each cell is one test. edm2's cells (`RUN_ONCE`) run their layout once
(tools/layout_parity.py's `run` with no draws): the layout compiles and takes its first step, its
sample the one-device reference's, its loss and gradient finite where the
reference's are, and its FLOPs an even split, or it is refused as REFUSED
states. Every other cell's test judges that first step's loss and gradient against one device by the
floor judge, which states it works or is refused as well, so no second test runs it again
(`layout_parity.floored`): each leaf within
FLOOR_FACTOR of the reference's own deviation under reassociating its sums
(its rows reordered and pooled, and the step's distance from float64), and
its FLOPs an even split. The tool's check 1 over many draws (`run` with
seeds, its CLI's judge) is no matrix test's judge while a cell's draws
cannot shard across CI tasks: its cells ran past a task's target whole, so
accuracy here keeps the floor judge the matrix ran before check 1, under
its own test name. Each cell states what it does: it works, or it is
refused by design for the reason REFUSED gives. A supported cell that
refuses or mismatches fails, and so does a refused cell that runs, so a
refusal is never read as a pass and a lifted one is never left unclaimed.
CI spreads a slow model's cells over its tasks; the run-once cells one
task runs share their model's reference draw, kept on disk under its
content (`layout_parity.References`), and the floor cells share their
model's reference and floors in the process (`layout_parity.floored`),
so each is computed once a task. The models are the tool's zoo, sized
so every layout divides them. Each row lists the layouts that split
its model in a way no other row covers: a pair's second axis, a packed or
masked column under a sequence split, a bidirectional exchange, a vision tower
beside a pipelined decoder, a tensor or sequence axis over experts, a norm's
statistics over groups of a split width, an attention gate of its own
projection over split heads, and a data axis Explicit
(`MeshSpec.explicit`), alone and beside fsdp, wherever an operation is placed
from its rows rather than its operands: the embedding's gather and scatter-add,
GRPO's shifted scores, MDLM's mask draw, Mamba-1's and Mamba-2's states and
conv histories, unpatchify, JEPA's repeat and covariance, and a vision tower's
placed slots.

The tool draws every parameter an initializer leaves all zeros, so a
DiT's zeroed output and modulations do not hide the layers above them
from the first step's gradient (2 of its 70 leaves were judged before).

A layout's data axis takes the devices its other axes leave, so on eight
devices every pipeline below splits the rows over data. The rows of
CELLS_ON_FOUR run on four of the eight, where four stages, or two beside a
sequence axis of two, leave no axis for the rows: every device holds all of
them."""

import jax
import pytest
from tool_support import load

CELLS: dict[str, tuple[str, ...]] = {
    "dense_sft": ("fsdp4", "tensor2_sequence2"),
    "dense_dpo": ("fsdp4", "sequence4"),
    "dense_grpo": ("tensor4", "sequence4", "data4_explicit"),
    "dense_mdlm": ("sequence4", "fsdp2_tensor2", "fsdp2_explicit"),
    "grouped": ("fsdp2_tensor2",),
    "gated": ("tensor4",),
    "moe_grpo": ("expert4", "expert2_fsdp2", "tensor4", "sequence4"),
    "moe_exchange": ("expert4", "expert2_fsdp2"),
    "mla_dpo": ("fsdp4", "sequence4"),
    "hybrid_sft": ("sequence4", "stage4", "data4_explicit"),
    "routed_tail": ("stage4",),
    "shared_tail": ("stage2_fsdp2",),
    "mamba1_packed": ("fsdp2_explicit",),
    "dense_packed": ("sequence4",),
    "rigel_packed": ("sequence4",),
    "mmdit": ("tensor4", "fsdp2_tensor2"),
    "dit": ("sequence4", "tensor4", "fsdp2_explicit"),
    "unet": ("tensor4", "fsdp2_tensor2", "sequence4"),
    "edm2": ("tensor4", "fsdp2_tensor2", "sequence4"),
    "jepa": ("fsdp4", "sequence4", "tensor4", "data4_explicit"),
    "multimodal": ("sequence4", "stage4", "data4_explicit"),
}
CELLS_ON_FOUR: dict[str, tuple[str, ...]] = {
    "dense": ("stage4", "stage2_sequence2"),
}
REFUSED: dict[tuple[str, str], str] = {
    ("hybrid_sft", "stage4"): "its four layers alternate two kinds, so no run of four repeats one layer",
}
"""The cells refused by design, each with its reason."""


RUN_ONCE: frozenset[tuple[str, str]] = frozenset({
    ("edm2", "tensor4"), ("edm2", "fsdp2_tensor2"), ("edm2", "sequence4"),
})
"""The cells no accuracy test judges. edm2's zoo model joined the matrix with check 1 as its
accuracy judge; the floor judge has never judged it, and one floor-judged cell of it runs 180 s
alone on an armada CPU worker (measured 2026-10-28), past a task's target for a test no durations
record holds. Its layouts run in the run-once test, and its accuracy returns with check 1 when a
cell's draws shard across tasks, as every cell's does."""


@pytest.fixture(scope="session")
def references(tmp_path_factory):
    """The layout tool this task's cells run and the reference draws its run-once cells share, each
    kept on disk under its content (`layout_parity._cache_name`) when first computed, while the disk
    keeps 4 GiB free, and read back by every later cell of its model. Its tool module is the one each
    test runs: the states and references the tool holds in a process (`_initial`,
    `layout_parity._floor_held`) live in that module, and `load` executes its file afresh, so a cell
    that loaded its own would rebuild its model's initial state and floor reference itself, where its
    task's first cell of the model built them."""
    tool = load("layout_parity")
    return tool, tool.References(tmp_path_factory.mktemp("layout-draws"), prepare=True, reserve=4 << 30)


WHOLE_SUITE: frozenset[tuple[str, str]] = frozenset({("edm2", "tensor4"), ("edm2", "fsdp2_tensor2")})
"""The run-once cells the whole suite alone runs: compile-bound, each takes 123-132 s alone on an
armada 25/17 worker and 113.8 and 98.4 s in a warm task there (2026-10-29), about 154 and 133 s on
the fleet's 26/17 workers (1.35x), past a task's 120 s target, and no smaller model cuts the
compile."""


def cells(*, judged: bool) -> list:
    """The cells as a test's parameters, their layouts on every device but CELLS_ON_FOUR's, on four:
    for the accuracy test (`judged`) every cell but the run-once ones (`RUN_ONCE`), and for the
    run-once test those alone, WHOLE_SUITE's marked to run on the whole suite."""
    return [pytest.param(model, layout, devices, id=f"{model}-{layout}" + ("-on-four" if devices else ""),
                         marks=pytest.mark.whole_suite if (model, layout) in WHOLE_SUITE else ())
            for table, devices in ((CELLS, None), (CELLS_ON_FOUR, 4))
            for model, layouts in sorted(table.items()) for layout in layouts
            if judged != ((model, layout) in RUN_ONCE)]


def statuses(rows):
    """The rows' statuses, with what explains each, whole, where an assertion shows them."""
    shown = ("layout", "status", "worst_leaf", "worst_ratio", "loss_error", "loss_bound", "flops_ratio",
             "flops_bound", "reason", "error")
    return [row["status"] for row in rows], "\n".join(
        " ".join(f"{key}={row.get(key)}" for key in shown) for row in rows)


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize(("model", "layout", "devices"), cells(judged=False))
def test_each_layout_of_a_model_runs_or_is_refused_as_stated(model, layout, devices, references):
    """Each run-once layout (`RUN_ONCE`) compiles and takes its first step, its sample the one-device
    reference's and its FLOPs an even split, or is refused for the reason REFUSED gives: a refused cell
    that runs fails, as does a supported one that raises."""
    tool, draws = references
    with jax.enable_x64(new_val=True):
        rows = tool.run([model], (layout,), dtype="float32", seeds=0, mixture={}, objective={},
                        references=draws, speak=lambda line: None, keep=lambda rows: None,
                        devices=devices)
    found, shown = statuses(rows)
    assert found == ["refused" if (model, layout) in REFUSED else "works"], shown


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize(("model", "layout", "devices"), cells(judged=True))
def test_each_layout_of_a_model_matches_one_device_or_is_refused_as_stated(model, layout, devices,
                                                                           references):
    """Each layout's first step matches one device's by the floor judge (`layout_parity.floored`): its
    loss and each gradient leaf within FLOOR_FACTOR of the reference's own deviation under reassociating
    its sums, and its FLOPs an even split, or it is refused for the reason REFUSED gives."""
    tool, _ = references
    with jax.enable_x64(new_val=True):
        rows = [tool.floored(model, layout, dtype="float32", devices=devices)]
    found, shown = statuses(rows)
    assert found == ["refused" if (model, layout) in REFUSED else "works"], shown
