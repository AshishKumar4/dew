"""Every model family on every objective class it takes, split by the layouts
that split it differently, against one device on the eight-device CPU mesh.

Each cell is one layout of one model, judged by the layout tool's check 1
(tools/layout_parity.py's `run` over the model's draws, `judged`): over
`SEEDS` draws of its global batch, each layout's first-step loss and
gradient leaves are as near the step computed in float64 as one device's are,
within PARITY, and its FLOPs are an even split; or the cell is refused by
design for the reason REFUSED gives. A supported cell that refuses,
mismatches or is inconclusive fails, and so does a refused cell that runs, so
a refusal is never read as a pass and a lifted one is never left unclaimed.
Each model's seeds are `seeds_for` paired over its layouts, its largest
(armada vq-layout-size, 11-06). A model whose draws take past a task's target
whole runs its cells as armada's map tasks and a judge (`check1_shards`), its
seeds its mark's. Every cell of one task shares its model's reference draws,
kept on disk under their content (`layout_parity.References`). The models are
the tool's zoo, sized so every layout divides them. Each row lists the layouts
that split its model in a way no other row covers: a pair's second axis, a
packed or masked column under a sequence split, a bidirectional exchange, a
vision tower beside a pipelined decoder, a tensor or sequence axis over
experts, a norm's statistics over groups of a split width, an attention gate
of its own projection over split heads, and a data axis Explicit
(`MeshSpec.explicit`), alone and beside fsdp, wherever an operation is placed
from its rows rather than its operands: the embedding's gather and
scatter-add, GRPO's shifted scores, MDLM's mask draw, Mamba-1's and Mamba-2's
states and conv histories, unpatchify, JEPA's repeat and covariance, and a
vision tower's placed slots.

The tool draws every parameter an initializer leaves all zeros, so a
DiT's zeroed output and modulations do not hide the layers above them
from the first step's gradient (2 of its 70 leaves were judged before).
Every layout's step starts from one device's initial state; a layout's own
draw of it, split over its mesh, is another program, which each model's
first layout of every split of its parameters holds to one device's draw by
check 1 (`layout_parity.held_draw`, docs/design/testing.md's Layout
invariance).

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
    "edm2_level": ("tensor4", "fsdp2_tensor2"),
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


SEEDS: dict[str, int] = {
    "dense_sft": 128, "dense_dpo": 64, "dense_mdlm": 128, "grouped": 64, "gated": 64, "moe_exchange": 64,
    "mla_dpo": 16, "hybrid_sft": 128, "shared_tail": 32, "mamba1_packed": 8, "dense_packed": 32,
    "rigel_packed": 128, "mmdit": 8, "dit": 8, "jepa": 128, "multimodal": 64, "dense": 64,
}
"""Each model's declared check-1 seeds whose cells run whole in one task: the largest of its layouts'
`seeds_for` paired (armada vq-layout-size, 11-06). The sharded models' seeds are their marks'."""


@pytest.fixture(scope="session")
def references(tmp_path_factory):
    """The layout tool this task's cells run and the reference draws its cells share, each kept on disk
    under its content (`layout_parity._cache_name`) when first computed, while the disk keeps 4 GiB free,
    and read back by every later cell of its model. Its tool module is the one each test runs: the states
    the tool holds in a process (`_initial`) live in that module, and `load` executes its file afresh, so a
    cell that loaded its own would rebuild its model's initial state, where its task's first cell of the
    model built it."""
    tool = load("layout_parity")
    return tool, tool.References(tmp_path_factory.mktemp("layout-draws"), prepare=True, reserve=4 << 30)


def cells(*models: str) -> list[tuple[str, str, int | None]]:
    """The cells of `models` (every model's by default) as a test's arguments, their layouts on every
    device but CELLS_ON_FOUR's, on four."""
    return [(model, layout, devices)
            for table, devices in ((CELLS, None), (CELLS_ON_FOUR, 4))
            for model, layouts in sorted(table.items()) for layout in layouts
            if not models or model in models]


PARAMETER_AXES = ("fsdp", "tensor", "expert", "stage")
"""The mesh axes that split a model's parameters; a data or sequence axis splits its rows alone."""
DRAW_SEEDS = 8
"""The keys each split's draw of a model's initial parameters is held over (`held_draw`)."""


def split_draws() -> list[tuple[str, str, int | None]]:
    """For each model, its first cell of every split of its parameters (`PARAMETER_AXES` at their sizes
    in the tool's `LAYOUTS`), the refused cells left out: the cells whose own draw of the initial
    parameters is a program no other cell of the model's draws."""
    layouts = load("layout_parity").LAYOUTS
    seen, chosen = set(), []
    for model, layout, devices in sorted(cells()):
        split = tuple(sorted((axis, size) for axis, size in layouts[layout].items()
                             if axis in PARAMETER_AXES and size > 1))
        if split and (model, layout) not in REFUSED and (model, split, devices) not in seen:
            seen.add((model, split, devices))
            chosen.append((model, layout, devices))
    return chosen


def named(cases) -> list[str]:
    """Each cell's test id: its model and layout, on four where it runs there."""
    return [f"{model}-{layout}" + ("-on-four" if devices else "") for model, layout, devices in cases]


def statuses(rows):
    """The rows' statuses, with what explains each, whole, where an assertion shows them."""
    shown = ("layout", "status", "flops_ratio", "flops_bound", "reason", "error")
    return [row["status"] for row in rows], "\n".join(
        " ".join(f"{key}={row.get(key)}" for key in shown) for row in rows)


def assert_matches_one_device(model, layout, devices, references, seeds: int, window: int = 1):
    """A cell of the matrix: `layout`'s first step on `model` over `seeds` draws (each a `window` of
    consecutive draws) on par with one device's (check 1, `layout_parity.judged`) and its FLOPs an even
    split, or refused for the reason REFUSED gives."""
    tool, draws = references
    with jax.enable_x64(new_val=True):
        rows = tool.run([model], (layout,), dtype="float32", seeds=seeds, windows=window, mixture={},
                        objective={}, references=draws, speak=lambda line: None, keep=lambda rows: None,
                        devices=devices)
    found, shown = statuses(rows)
    assert found == ["refused" if (model, layout) in REFUSED else "works"], shown


UNSHARDED = cells(*SEEDS)


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize(("model", "layout", "devices"), UNSHARDED, ids=named(UNSHARDED))
def test_each_layout_of_a_model_matches_one_device_or_is_refused(model, layout, devices, references):
    """Each layout's first step on par with one device's over its model's SEEDS draws (check 1), its FLOPs
    an even split, or refused for the reason REFUSED gives."""
    assert_matches_one_device(model, layout, devices, references, SEEDS[model])


def marked_seeds(request) -> int:
    """The seeds a sharded cell declares: its test's `check1_shards` mark's, which CI's plan reads."""
    return request.node.get_closest_marker("check1_shards").kwargs["seeds"]


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize(("model", "layout", "devices"), cells("dense_grpo"), ids=named(cells("dense_grpo")))
@pytest.mark.check1_shards(seeds=512, per_task=128)
def test_a_grpo_decoders_layouts_match_one_device(model, layout, devices, references, request):
    """The matrix's cells (`assert_matches_one_device`) of the dense GRPO decoder, its draws as map tasks:
    its tensor4 cell judged 512 draws in 65 s whole (armada vq-layout-judge, 11-06)."""
    assert_matches_one_device(model, layout, devices, references, marked_seeds(request))


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize(("model", "layout", "devices"), cells("moe_grpo"), ids=named(cells("moe_grpo")))
@pytest.mark.check1_shards(seeds=512, per_task=64)
def test_a_grpo_mixtures_layouts_match_one_device(model, layout, devices, references, request):
    """The matrix's cells (`assert_matches_one_device`) of the GRPO mixture of experts, its draws as map
    tasks."""
    assert_matches_one_device(model, layout, devices, references, marked_seeds(request))


ROUTED_TAIL = cells("routed_tail")


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize(("model", "layout", "devices"), ROUTED_TAIL, ids=named(ROUTED_TAIL))
@pytest.mark.check1_shards(seeds=32, per_task=2)
def test_a_routed_tails_pipeline_matches_one_device(model, layout, devices, references, request):
    """The matrix's cell (`assert_matches_one_device`) of the routed tail's four stages, its draws as map
    tasks: its 64-row batch (`layout_parity.PLACED_ROWS`) sized paired at 32 seeds in 1879 s (armada
    vq-layout-size4, 11-06), a pipeline's draws slow on a CPU."""
    assert_matches_one_device(model, layout, devices, references, marked_seeds(request))


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize(("model", "layout", "devices"), cells("unet"), ids=named(cells("unet")))
@pytest.mark.check1_shards(seeds=128, per_task=24)
def test_a_unets_layouts_match_one_device(model, layout, devices, references, request):
    """The matrix's cells (`assert_matches_one_device`) of the U-Net, its draws as map tasks: its tensor4
    cell judged 128 draws in 116 s whole (armada vq-layout-judge, 11-06)."""
    assert_matches_one_device(model, layout, devices, references, marked_seeds(request))


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize(("model", "layout", "devices"), cells("edm2"), ids=named(cells("edm2")))
@pytest.mark.check1_shards(seeds=512, per_task=64)
def test_edm2s_layouts_match_one_device(model, layout, devices, references, request):
    """The matrix's cells (`assert_matches_one_device`) of EDM2's magnitude-preserving U-Net, its draws as
    map tasks: its fsdp2_tensor2 cell judged 512 draws in 117 s whole (armada vq-layout-judge, 11-06), where
    the floor judge had left it out (rows 1323, 827)."""
    assert_matches_one_device(model, layout, devices, references, marked_seeds(request))


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize(("model", "layout", "devices"), cells("edm2_level"), ids=named(cells("edm2_level")))
@pytest.mark.check1_shards(seeds=256, per_task=64)
def test_edm2s_level_conditioned_layouts_match_one_device(model, layout, devices, references, request):
    """The matrix's cells (`assert_matches_one_device`) of EDM2's level-conditioned U-Net, its draws as map
    tasks."""
    assert_matches_one_device(model, layout, devices, references, marked_seeds(request))


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize(("model", "layout", "devices"), split_draws(), ids=named(split_draws()))
def test_each_split_draws_a_models_initial_parameters_as_one_device_does(model, layout, devices, references):
    """A layout's own draw of a model's initial parameters, split over its mesh, and one device's draw
    are each held by check 1 to their float64 truth, the one-device program replayed in float64 from
    its own random bits, which every layout draws alike (`layout_parity.held_draw`)."""
    tool, _ = references
    tool.held_draw(model, layout, seeds=DRAW_SEEDS, devices=devices)
