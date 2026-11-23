"""The diagonal scan's check-1 matrix (tests/kernel_matrix.py): HGRN's recurrence on the template's kernels on
par with the portable math.

Its cells run generation.CHOICES' `diagonal_scan`: `diagonal_scan.diagonal_scan` over bf16 or fp32 writes
from a state, whole rows or packed ones, its kernel named (`generation.chosen`). Its output, final state and
the gradients of its writes, gates and state are held to the same call on the portable math ('xla', check 1),
the token recurrence in float64 the truth (test_diagonal_scan's `token_by_token`). What it serves is its
selector's (`diagonal_scan.DIAGONAL_SCAN`, as the call asks it).
"""

import functools

import generation_support
import jax
import jax.numpy as jnp
import pytest
from gpu_flow import cell
from kernel_matrix import AXES
from reference_error import assert_as_exact
from test_diagonal_scan import draws, token_by_token, with_gradients

from dew.nn.kernels import diagonal_scan, generation
from dew.nn.kernels.selection import KernelRefused

CHOICES = frozenset({diagonal_scan.DIAGONAL_SCAN.choice})
"""The generation.CHOICES entries the family's cells run (row 184): the diagonal scan's selector's."""

KERNELS = {diagonal_scan.DIAGONAL_SCAN.choice: frozenset(
    kernel.name for kernel in diagonal_scan.DIAGONAL_SCAN.kernels if not kernel.generic)}
"""The kernels each choice's cells run under test (row 4513), its selector's every kernel but the generic
one each is held to: the recurrence template's walk."""

(KERNEL,) = KERNELS[diagonal_scan.DIAGONAL_SCAN.choice]
"""The kernel each cell holds to the portable math: the recurrence template's walk
(`templates.recurrence.diagonal`)."""

LENGTH = 300
"""A cell's tokens: several spans a row at every generation's facts (`diagonal.walk`)."""

SEEDS = 8
"""A cell's declared draws against the portable math (check 1)."""


def _step(kernel: str):
    """The diagonal scan on `kernel` (`generation.chosen`): its output, final state and gradients."""
    def scan(x, g, state, starts):
        with generation.chosen(diagonal_scan=kernel):
            return diagonal_scan.diagonal_scan(x, g, state, starts=starts)

    return with_gradients(scan)


def _drawn(case: dict[str, str]):
    """A draw of `case`'s two rows by seed (`test_diagonal_scan.draws`)."""
    return draws(LENGTH, jnp.dtype(case["dtype"]), case["documents"] == "packed")


@functools.cache
def _taken(case: tuple[tuple[str, str], ...]) -> bool:
    """Whether the selector gives `case` the kernel under its generation, traced (`jax.eval_shape`)."""
    named = dict(case)
    with generation.assumed(named["generation"]), generation.routes() as found:
        try:
            jax.eval_shape(_step(KERNEL), *_drawn(named)(0))
        except KernelRefused:
            return False
    return any(route.kernel == KERNEL for route in found)


def supported(case: dict[str, str]) -> bool:
    """Whether the selector serves `case` on its GPU (tests/kernel_matrix.py, `_taken`)."""
    return _taken(tuple(case.items()))


CASES = (
    ('sm90', 'bfloat16', 'one'),
    ('sm90', 'float32', 'packed'),
    ('sm80', 'bfloat16', 'packed'),
    ('sm80', 'float32', 'one'),
)
"""The cells' cases, each its axis values in `AXES`' order: every pair of axis values the selector serves
(`kernel_matrix.covering("diagonal_scan")`, regenerated where the axes or the selector's support change)."""


@cell("diagonal_scan", covers=lambda case: dict(zip(AXES["diagonal_scan"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=generation_support.pinned(case[0]))
    for case in CASES], ids="-".join)
def a_diagonal_scans_training_step_is_on_par(case: tuple[str, ...]) -> None:
    """`case`'s diagonal scan on its GPU on the kernels, on par with the portable math (check 1, the token
    recurrence in float64 the truth): its output and final state and the gradients of its writes, gates and
    state. Its routes are its family's (`CHOICES`)."""
    named = dict(zip(AXES["diagonal_scan"], case, strict=True))
    with generation.routes() as found:
        assert_as_exact(_step(KERNEL), _step("xla"), _drawn(named), "-".join(case), seeds=SEEDS,
                        truth_fn=jax.jit(with_gradients(token_by_token)))
    assert {route.choice for route in found} <= CHOICES
