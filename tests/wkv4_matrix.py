"""RWKV4's WKV's check-1 matrix (tests/kernel_matrix.py): the recurrence template's channel walk on par with
the portable scan.

Its cells run generation.CHOICES' `wkv4`: `kernels.rwkv4.wkv4` over bf16 or fp32 keys and values from a
carry, whole rows or packed ones, its kernel named (`generation.chosen`). Its output, final carry and every
operand's gradient are held to the same call on the portable scan ('xla', check 1), fla's recurrence a token
at a time in float64 the truth (`rwkv4.recurrent_wkv4`). What it serves is its selector's (`rwkv4.WKV4`, as
the call asks it).
"""

import functools

import generation_support
import jax
import jax.numpy as jnp
import pytest
from gpu_flow import cell
from kernel_matrix import AXES
from reference_error import assert_as_exact
from test_wkv4 import differentiated, wide_draws

from dew.nn.kernels import generation, rwkv4
from dew.nn.kernels.selection import KernelRefused

CHOICES = frozenset({rwkv4.WKV4.choice})
"""The generation.CHOICES entries the family's cells run (row 184): the WKV's selector's."""

KERNELS = {rwkv4.WKV4.choice: frozenset(kernel.name for kernel in rwkv4.WKV4.kernels if not kernel.generic)}
"""The kernels each choice's cells run under test (row 4513), its selector's every kernel but the generic
one each is held to: the recurrence template's walk."""

(KERNEL,) = KERNELS[rwkv4.WKV4.choice]
"""The kernel each cell holds to the portable scan: the recurrence template's walk
(`templates.recurrence.wkv4`)."""

LENGTH = 300
"""A cell's tokens: several spans a row at every generation's facts (`templates.recurrence.wkv4.walk`)."""

SEEDS = 8
"""A cell's declared draws against the portable scan (check 1)."""


def _step(kernel: str):
    """The WKV on `kernel` (`generation.chosen`): its output, final carry and gradients."""
    def scan(time_decay, time_first, key, value, state, starts):
        with generation.chosen(wkv4=kernel):
            return rwkv4.wkv4(time_decay, time_first, key, value, state, starts)

    return differentiated(scan)


def _drawn(case: dict[str, str]):
    """A draw of `case`'s two rows by seed (`test_wkv4.wide_draws`)."""
    return wide_draws(LENGTH, jnp.dtype(case["dtype"]), case["documents"] == "packed")


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
(`kernel_matrix.covering("wkv4")`, regenerated where the axes or the selector's support change)."""


@cell("wkv4", covers=lambda case: dict(zip(AXES["wkv4"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=generation_support.pinned(case[0]))
    for case in CASES], ids="-".join)
def a_wkvs_training_step_is_on_par(case: tuple[str, ...]) -> None:
    """`case`'s WKV on its GPU on the kernels, on par with the portable scan (check 1, fla's recurrence a
    token at a time in float64 the truth): its output and final carry and every operand's gradient. Its
    routes are its family's (`CHOICES`)."""
    named = dict(zip(AXES["wkv4"], case, strict=True))
    with generation.routes() as found:
        assert_as_exact(_step(KERNEL), _step("xla"), _drawn(named), "-".join(case), seeds=SEEDS,
                        truth_fn=jax.jit(differentiated(rwkv4.recurrent_wkv4)))
    assert {route.choice for route in found} <= CHOICES
