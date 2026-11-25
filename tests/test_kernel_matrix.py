"""Each template kernel family's check-1 matrix (tests/kernel_matrix.py) covers every pair of axis values its
route serves, and nothing it does not: main's GPU tier runs its cells after a publish that reaches the kernels
(Main, 2026-11-02, after the attention dK escape, where the tier had no GQA width-128 backward cell). A family
whose axes, or whose route's support, change fails here until its cases are regenerated
(`kernel_matrix.covering`)."""

import importlib
import subprocess
import sys
from pathlib import Path

import gpu_flow
import kernel_matrix
import pytest

from dew.nn.kernels import generation

TESTS = Path(__file__).resolve().parent


@pytest.mark.parametrize("family", sorted(kernel_matrix.AXES))
def test_a_familys_cells_cover_every_pair_of_axis_values_its_route_offers(family):
    """A family's cells name values of its axes alone and hold every pair of axis values its route's
    derivation offers (`kernel_matrix.supported`, the census of its derivation's offer, cheap enough for each
    commit; the pairs a run's certified launches serve are a tier's census). That each cell is a combination
    the route serves as a run takes it is each cell's own check
    (`test_a_declared_cell_is_a_combination_its_route_serves`)."""
    cells = kernel_matrix.covered(family)
    axes = kernel_matrix.AXES[family]
    unaxed = [label for label, combination in cells
              if set(combination) != set(axes) or any(combination[axis] not in axes[axis] for axis in axes)]
    assert not unaxed, f"{family}'s cells name values outside its axes: {unaxed}"
    held_pairs = set().union(*(gpu_flow.held(combination, values=False) for _, combination in cells))
    missing = kernel_matrix.supported_pairs(family, already=held_pairs) - held_pairs
    assert not missing, (f"{len(missing)} pairs of {family}'s axis values its route serves are in no cell, "
                         f"e.g. {sorted(sorted(pair) for pair in missing)[:10]}: regenerate its cases from "
                         f"kernel_matrix.covering({family!r})")


@pytest.mark.parametrize(("family", "label", "combination"),
                         [(family, label, combination) for family in sorted(kernel_matrix.AXES)
                          for label, combination in kernel_matrix.covered(family)],
                         ids=lambda value: value if isinstance(value, str) and ":" in value else None)
def test_a_declared_cell_is_a_combination_its_route_serves(family, label, combination):
    """Each declared cell covers a combination its family's route serves as a run takes it
    (`kernel_matrix.served`: an attention backward's launch form as certification holds it, compiled for the
    cell's GPU), a cell apiece so the certifications spread across groups."""
    axes = kernel_matrix.AXES[family]
    assert kernel_matrix.served(family, tuple((axis, combination[axis]) for axis in axes)), label


@pytest.mark.parametrize(("flows", "tier"), [(kernel_matrix.FLOWS, "gpu"), (kernel_matrix.TPU_FLOWS, "tpu")],
                         ids=["gpu", "tpu"])
def test_every_matrix_flow_is_a_test_its_tier_selects(flows, tier):
    """Each of a tier's flows (`FLOWS`, `TPU_FLOWS`: node ids) is a test pytest collects under the tier's
    selection (`-m gpu`, `-m tpu`), whole or in parts, so the tier runs what the matrix declares (review: a
    helper's name, or an unmarked test, would run nothing)."""
    command = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "-m", tier]
    collected = subprocess.run([*command, *flows], cwd=TESTS.parent, capture_output=True, text=True,
                               check=False)
    # A flow in parts (`gpu_flow.in_parts`) collects as its parts' ids, `[part0]` and on.
    found = {line.strip().split("[")[0] for line in collected.stdout.splitlines()}
    missing = sorted(set(flows) - found)
    assert not missing, (missing, collected.stdout[-2000:], collected.stderr[-2000:])


def test_every_kernel_choice_is_run_under_test_by_a_familys_matrix():
    """Every choice a dispatch reports (`generation.CHOICES`) is a kernel some family's matrix runs under
    test: the families' CHOICES, each its cell module's, together are `generation.CHOICES` (row 184), so a
    choice added without a matrix fails here. A fused kernel that runs inside another's (a GEMM's prologue
    in the product) may be two families'."""
    modules = (importlib.import_module(module) for module, _ in kernel_matrix.CELLS.values())
    declared = set().union(*(module.CHOICES for module in modules))
    assert declared == generation.CHOICES, (
        f"choices no family's matrix runs: {sorted(generation.CHOICES - declared)}; "
        f"declared, not choices: {sorted(declared - generation.CHOICES)}")


def test_every_kernel_an_op_serves_is_run_under_test_by_a_familys_matrix():
    """Every kernel a dispatch may select for an op (`selection.Op`'s kernels, read from the production
    modules, the generic path that takes what the others turn down left as the reference) is one some
    family's cells run under test, by the families' KERNELS, each cell module's own, which its cells assert
    their routes against (row 4513): a kernel added to an op without a cell fails here."""
    tested: dict[str, set[str]] = {}
    for module, _ in kernel_matrix.CELLS.values():
        for choice, kernels in importlib.import_module(module).KERNELS.items():
            tested.setdefault(choice, set()).update(kernels)
    served = {choice: {kernel.name for op in found for kernel in op.kernels if not kernel.generic}
              for choice, found in kernel_matrix.ops().items()}
    untested = {choice: sorted(kernels - tested.get(choice, set())) for choice, kernels in served.items()
                if kernels - tested.get(choice, set())}
    assert not untested, f"kernels no family's matrix runs under test: {untested}"


@pytest.mark.parametrize("family", sorted(kernel_matrix.SEEDED))
def test_a_gpu_runs_each_seeded_case_of_a_family_and_no_other(family):
    """The GPU tier's cells of a family (`kernel_matrix.seeded_cells`) are its seeded cases, each a GPU-only
    red a ledger row names, every one of them and no other: the rest of its pairs are the CPU's (row 5461's
    covering set)."""
    cells = dict(kernel_matrix.covered(family))
    kept = [tuple(cells[cell.label].values()) for cell in kernel_matrix.seeded_cells(family)]
    assert sorted(kept) == sorted({case for _, case in kernel_matrix.SEEDED[family]})


def test_an_adapted_steps_cell_names_its_kernels_over_its_whole_step():
    """The low-rank family's training step (`low_rank_matrix._step`) names its kernels over the whole step,
    its pullback too (`with_gradients` runs it outside the forward's call): traced abstractly as an sm90 cell,
    its routes run `low_rank` `fused` and every other choice at XLA, as its GPU cell asserts (row 12755: named
    in the forward alone, the fused form's backward took the grouped template on a GH200)."""
    import jax
    import low_rank_matrix as matrix
    import numpy as np
    from jax.sharding import Mesh

    case = dict(zip(kernel_matrix.AXES["low_rank"], ("sm90", "8", "none", "bfloat16"), strict=True))
    model, variables = matrix._adapted(case)
    factors, x, cotangent = jax.eval_shape(matrix._drawn(case, variables), 0)
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with (generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh), generation.routes() as routes):
        jax.eval_shape(matrix._step(model, variables, "fused"), factors, x, cotangent)
    recorded = {(route.choice, route.kernel) for route in routes}
    unnamed = sorted(pair for pair in recorded if pair[0] not in matrix.CHOICES and pair[1] != "xla")
    assert ("low_rank", "fused") in recorded and not unnamed, (sorted(recorded), unnamed)
