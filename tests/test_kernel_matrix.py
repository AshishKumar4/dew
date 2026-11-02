"""Each template kernel family's check-1 matrix (tests/kernel_matrix.py) covers every pair of axis values its
route serves, and nothing it does not: main's GPU tier runs its cells after a publish that reaches the kernels
(Main, 2026-11-02, after the attention dK escape, where the tier had no GQA width-128 backward cell). A family
whose axes, or whose route's support, change fails here until its cases are regenerated
(`kernel_matrix.covering`)."""

import ast
import itertools
from pathlib import Path

import kernel_matrix
import pytest

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


def test_every_matrix_flow_is_a_test_its_file_defines():
    """Each of `FLOWS`, the tier's node ids, names a test function its file defines, so the tier runs what the
    matrix declares."""
    for flow in kernel_matrix.FLOWS:
        path, _, name = flow.partition("::")
        defined = {node.name for node in ast.parse((TESTS.parent / path).read_text()).body
                   if isinstance(node, ast.FunctionDef)}
        assert name in defined, f"{flow} names no test of {path}"
