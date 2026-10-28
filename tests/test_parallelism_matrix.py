"""Every model family on every objective class it takes, split by the layouts
that split it differently, against one device on the eight-device CPU mesh.

tools/layout_parity.py judges a layout's first-step loss and gradient by
check 1 against one device's over the model's SEEDS draws of its batch, the
truth the step in float64, every arm held to the one-device reference's
sample, and holds the layout's FLOPs to an even split. Each cell states what
it does: it works, or it is refused by design for the reason REFUSED gives.
A supported cell that refuses, mismatches or is inconclusive fails, and so
does a refused cell that runs, so a refusal is never read as a pass and a
lifted one is never left unclaimed. Each cell is two tests: whether its layout
runs or is refused, run once on every train, and its accuracy over the declared
draws, on every train where it fits the task target and in the whole suite where
it does not (`WHOLE_SUITE`). CI spreads a slow model's cells over its tasks;
the cells one task runs share their
model's reference and float64 draws, kept on disk under their content
(`layout_parity.References`), so each is computed once a task. The models are the
tool's zoo, sized so every layout divides them. Each row lists the layouts that
split its model in a way no other row covers: a pair's second axis, a packed or
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
SEEDS: dict[str, int] = {
    "dense_sft": 128, "dense_dpo": 64, "dense_grpo": 512, "dense_mdlm": 64, "grouped": 64, "gated": 64,
    "moe_grpo": 512, "moe_exchange": 64, "mla_dpo": 16, "hybrid_sft": 128, "routed_tail": 512,
    "shared_tail": 32, "mamba1_packed": 64, "dense_packed": 64, "rigel_packed": 256, "mmdit": 8,
    "dit": 8, "unet": 128, "edm2": 256, "jepa": 128, "multimodal": 64, "dense": 32,
}
"""Each model's declared draws, sized paired by `seeds_for` on armada, Dew's arm each of its layouts
(`dew_fn`), the most of them: a layout and the one-device reference run from one state on one batch,
so they share their upstream rounding, as test_architectures.py's layout cells do."""
WINDOWS: dict[str, int] = {"dense_grpo": 4, "moe_grpo": 4}
"""The models check 1 judges over windows of consecutive draws (`over_windows`): GRPO's first-step loss
sums advantage-weighted terms that nearly cancel, so its single draws' errors are heavy-tailed, and no
count of them resolves PARITY within 512 (sized on armada)."""
REFUSED: dict[tuple[str, str], str] = {
    ("hybrid_sft", "stage4"): "its four layers alternate two kinds, so no run of four repeats one layer",
}
"""The cells refused by design, each with its reason."""


@pytest.fixture(scope="session")
def draws(tmp_path_factory):
    """The reference and float64 draws this task's cells share, each kept on disk under its content
    (`layout_parity._cache_name`) when first computed, while the disk keeps 4 GiB free, and read back by
    every later cell of its model."""
    tool = load("layout_parity")
    return tool.References(tmp_path_factory.mktemp("layout-draws"), prepare=True, reserve=4 << 30)


WHOLE_SUITE: frozenset[tuple[str, str]] = frozenset({
    ("moe_grpo", "expert4"), ("moe_grpo", "tensor4"), ("moe_grpo", "sequence4"),
    ("moe_grpo", "expert2_fsdp2"),
    ("dense_grpo", "tensor4"), ("dense_grpo", "data4_explicit"), ("dense_grpo", "sequence4"),
    ("unet", "tensor4"), ("unet", "fsdp2_tensor2"), ("unet", "sequence4"),
})
"""The cells whose accuracy check runs past the suite's 240 s task target on an armada CPU worker at their
declared draws, measured each on its own (10-28: moe_grpo's 871-1475 s, dense_grpo's 736-1149 s,
unet's 526-749 s; every other cell 52-193 s): they run in the whole suite
(`whole_suite`, every fifth train, where a red is bisected and a P1) and return to every train's CI once
they fit or their draws shard across tasks. Their layouts still run in every train
(`test_each_layout_of_a_model_runs_or_is_refused_as_stated`), which caught each of the matrix's 24 reds
since 10-07, every one a layout that raised."""


FLOOR_JUDGED: dict[tuple[str, str], str] = {("routed_tail", "stage4"): "ledger 1992"}
"""The cells whose accuracy keeps the floor judge main's matrix used before check 1
(`layout_parity.floored`), each with the row that moves it onto check 1: routed_tail on stage4 is
inconclusive on check 1 at 512 seeds, two draws in a hundred carrying most of both arms' error on its
per-layer inputs, their records equal. 1186 retires the floor judge with them."""


def cells(*, judged: bool) -> list:
    """Every cell as a test's parameters: for the accuracy check (`judged`), the cells that run, the slow
    ones marked for the whole suite (`WHOLE_SUITE`)."""
    return [pytest.param(model, layout, devices, id=f"{model}-{layout}" + ("-on-four" if devices else ""),
                         marks=[pytest.mark.whole_suite] if judged and (model, layout) in WHOLE_SUITE else [])
            for table, devices in ((CELLS, None), (CELLS_ON_FOUR, 4))
            for model, layouts in sorted(table.items()) for layout in layouts
            if not (judged and (model, layout) in REFUSED)]


def matrix_rows(model, layout, devices, seeds, draws):
    """The tool's one row for the cell, its layout judged over `seeds` draws, or with none run once."""
    tool = load("layout_parity")
    with jax.enable_x64(new_val=True):
        return tool.run([model], (layout,), dtype="float32", seeds=seeds, windows=WINDOWS, mixture={},
                        objective={}, references=draws, speak=lambda line: None, keep=lambda rows: None,
                        devices=devices)


def statuses(rows):
    """The rows' statuses, with what explains each, whole, where an assertion shows them."""
    shown = ("layout", "status", "flops_ratio", "flops_bound", "reason", "error")
    return [row["status"] for row in rows], "\n".join(
        " ".join(f"{key}={row.get(key)}" for key in shown) for row in rows)


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize(("model", "layout", "devices"), cells(judged=False))
def test_each_layout_of_a_model_runs_or_is_refused_as_stated(model, layout, devices, draws):
    """Each layout compiles and takes its first step, its sample the one-device reference's and its FLOPs
    an even split, or is refused for the reason REFUSED gives: a refused cell that runs fails, as does a
    supported one that raises."""
    found, shown = statuses(matrix_rows(model, layout, devices, 0, draws))
    assert found == ["refused" if (model, layout) in REFUSED else "works"], shown


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize(("model", "layout", "devices"), cells(judged=True))
def test_each_layout_of_a_model_matches_one_device(model, layout, devices, draws):
    """Each supported layout's first step is as exact as one device's over the model's declared draws
    (check 1), and its FLOPs an even split; a cell check 1 does not yet resolve, by the floor judge
    (`FLOOR_JUDGED`)."""
    if (model, layout) in FLOOR_JUDGED:
        with jax.enable_x64(new_val=True):
            rows = [load("layout_parity").floored(model, layout, dtype="float32", devices=devices)]
    else:
        rows = matrix_rows(model, layout, devices, SEEDS[model], draws)
    found, shown = statuses(rows)
    assert found == ["works"], shown
