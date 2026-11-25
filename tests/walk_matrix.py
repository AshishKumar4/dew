"""The channel walks' check-1 matrix (tests/kernel_matrix.py): the recurrence template's walks
(`templates.recurrence.walks`) on par with their ops' portable math.

Its cells run generation.CHOICES' `diagonal_scan` and `wkv4`, each op's kernel named (`generation.chosen`):
HGRN's diagonal scan (`kernels.diagonal.diagonal_scan`) and RWKV4's WKV (`kernels.rwkv4.wkv4`), over 16-bit or
fp32 operands from an entering state, whole rows or packed ones. Each output and final state and every
operand's gradient is held to the same call on the portable math ('xla', check 1), the op's recurrence a token
at a time in float64 the truth. What each serves is its selector's (`diagonal.DIAGONAL_SCAN`, `rwkv4.WKV4`).
"""

import functools

import generation_support
import jax
import jax.numpy as jnp
import pytest
import test_diagonal_scan
import test_wkv4
from gpu_flow import cell
from kernel_matrix import AXES
from reference_error import assert_as_exact

from dew.nn.kernels import diagonal, generation, rwkv4
from dew.nn.kernels.selection import KernelRefused

OPS = {"diagonal_scan": diagonal.DIAGONAL_SCAN, "wkv4": rwkv4.WKV4}
"""Each op the walks serve, by its choice: its selector."""

CHOICES = frozenset(OPS)
"""The generation.CHOICES entries the family's cells run (row 184): the walks' selectors'."""

KERNELS = {choice: frozenset(kernel.name for kernel in op.kernels if not kernel.generic)
           for choice, op in OPS.items()}
"""The kernels each choice's cells run under test (row 4513), its selector's every kernel but the generic
one each is held to: the recurrence template's walk."""

KERNEL = "mosaic_gpu"
"""The kernel each cell holds to the portable math: the recurrence template's walk."""

LENGTH = 300
"""A cell's tokens: several spans a row at every generation's facts (`walks.walk`)."""

SEEDS = 8
"""A cell's declared draws against the portable math (check 1)."""


def _differentiated(choice: str):
    """The test module's `differentiated` and its truth, the op's recurrence a token at a time."""
    if choice == "diagonal_scan":
        return test_diagonal_scan.differentiated, test_diagonal_scan.token_by_token
    return test_wkv4.differentiated, rwkv4.recurrent_wkv4


def _step(choice: str, kernel: str):
    """`choice`'s op on `kernel` (`generation.chosen`): its outputs and gradients."""
    differentiated, _ = _differentiated(choice)

    def run(*operands):
        with generation.chosen(**{choice: kernel}):
            if choice == "diagonal_scan":
                return diagonal.diagonal_scan(*operands[:3], starts=operands[3])
            return rwkv4.wkv4(*operands)

    return differentiated(run)


def _drawn(case: dict[str, str]):
    """A draw of `case`'s two rows by seed (its test module's)."""
    dtype, packed = jnp.dtype(case["dtype"]), case["documents"] == "packed"
    if case["op"] == "diagonal_scan":
        return test_diagonal_scan.draws(LENGTH, dtype, packed)
    return test_wkv4.wide_draws(LENGTH, dtype, packed)


@functools.cache
def _taken(case: tuple[tuple[str, str], ...]) -> bool:
    """Whether the op's selector gives `case` the kernel under its generation, traced (`jax.eval_shape`)."""
    named = dict(case)
    with generation.assumed(named["generation"]), generation.routes() as found:
        try:
            jax.eval_shape(_step(named["op"], KERNEL), *_drawn(named)(0))
        except KernelRefused:
            return False
    return any(route.kernel == KERNEL for route in found)


def supported(case: dict[str, str]) -> bool:
    """Whether the op's selector serves `case` on its GPU (tests/kernel_matrix.py, `_taken`)."""
    return _taken(tuple(case.items()))


CASES = (
    ('sm90', 'diagonal_scan', 'bfloat16', 'one'),
    ('sm90', 'wkv4', 'float32', 'packed'),
    ('sm80', 'diagonal_scan', 'bfloat16', 'packed'),
    ('sm80', 'wkv4', 'float32', 'one'),
    ('sm90', 'diagonal_scan', 'float32', 'one'),
    ('sm90', 'wkv4', 'bfloat16', 'one'),
)
"""The cells' cases, each its axis values in `AXES`' order: every pair of axis values the selectors serve
(`kernel_matrix.covering("channel_walk")`, regenerated where the axes or the selectors' support change)."""


@cell("channel_walk", covers=lambda case: dict(zip(AXES["channel_walk"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=generation_support.pinned(case[0]))
    for case in CASES], ids="-".join)
def a_channel_walks_training_step_is_on_par(case: tuple[str, ...]) -> None:
    """`case`'s op on its GPU on the walk, on par with its portable math (check 1, its recurrence a token at a
    time in float64 the truth): its outputs and final state and every operand's gradient. Its routes are its
    family's (`CHOICES`)."""
    named = dict(zip(AXES["channel_walk"], case, strict=True))
    differentiated, truth = _differentiated(named["op"])
    with generation.routes() as found:
        assert_as_exact(_step(named["op"], KERNEL), _step(named["op"], "xla"), _drawn(named), "-".join(case),
                        seeds=SEEDS, truth_fn=jax.jit(differentiated(truth)))
    assert {route.choice for route in found} <= CHOICES
