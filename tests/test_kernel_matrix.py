"""Each template kernel family's check-1 matrix (tests/kernel_matrix.py) covers every pair of axis values its
route serves, and nothing it does not: main's GPU tier runs its cells after a publish that reaches the kernels
(Main, 2026-11-02, after the attention dK escape, where the tier had no GQA width-128 backward cell). A family
whose axes, or whose route's support, change fails here until its cases are regenerated
(`kernel_matrix.covering`)."""

import importlib
import itertools
import subprocess
import sys
from pathlib import Path

import kernel_matrix
import pytest

from dew.nn.kernels import generation

TESTS = Path(__file__).resolve().parent


def pairs(combination: dict[str, str]) -> set[frozenset[tuple[str, str]]]:
    """Every pair of axis values `combination` holds, each without an order."""
    return {frozenset(pair) for pair in itertools.combinations(combination.items(), 2)}


@pytest.mark.parametrize("family", sorted(kernel_matrix.AXES))
def test_a_familys_cells_cover_every_pair_of_axis_values_its_route_serves_and_nothing_else(family):
    cells = kernel_matrix.covered(family)
    axes = kernel_matrix.AXES[family]
    unaxed = [label for label, combination in cells
              if set(combination) != set(axes) or any(combination[axis] not in axes[axis] for axis in axes)]
    assert not unaxed, f"{family}'s cells name values outside its axes: {unaxed}"
    unserved = [label for label, combination in cells
                if not kernel_matrix.supported(family, tuple((axis, combination[axis]) for axis in axes))]
    assert not unserved, f"{family}'s cells cover combinations its route does not serve: {unserved}"
    held = set().union(*(pairs(combination) for _, combination in cells))
    missing = {frozenset(pair) for pair in kernel_matrix.supported_pairs(family)} - held
    assert not missing, (f"{len(missing)} pairs of {family}'s axis values its route serves are in no cell, "
                         f"e.g. {sorted(sorted(pair) for pair in missing)[:10]}: regenerate its cases from "
                         f"kernel_matrix.covering({family!r})")


@pytest.mark.parametrize(("flows", "tier"), [(kernel_matrix.FLOWS, "gpu"), (kernel_matrix.TPU_FLOWS, "tpu")],
                         ids=["gpu", "tpu"])
def test_every_matrix_flow_is_a_test_its_tier_selects(flows, tier):
    """Each of a tier's flows (`FLOWS`, `TPU_FLOWS`: node ids) is a test pytest collects under the tier's
    selection (`-m gpu`, `-m tpu`), so the tier runs what the matrix declares (review: a helper's name, or an
    unmarked test, would run nothing)."""
    command = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "-m", tier]
    collected = subprocess.run([*command, *flows], cwd=TESTS.parent, capture_output=True, text=True,
                               check=False)
    found = {line.strip() for line in collected.stdout.splitlines()}
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
