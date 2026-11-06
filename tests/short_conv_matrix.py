"""The short conv's check-1 matrix (tests/kernel_matrix.py): a delta net's conv at its training shapes on par
with XLA's conv.

Its cells run generation.CHOICES' `short_conv`: `linear.causal_conv1d` as a gated delta net's layer calls it,
bf16 projections into the bf16 pieces fla's rule takes and fp32 into fp32, whole rows or packed ones, its
kernel named (`generation.chosen`, 'pallas'). Its pieces and the gradients of the projection and the taps are
held to the same call on XLA's conv ('xla', check 1), the float64 conv a row at a time the truth
(test_short_conv's `row_truth`). What it serves is its selector's (`short_conv.CONV`, as the call asks it).
"""

import functools

import jax
import jax.numpy as jnp
import pytest
from fla_gdn_support import CONV_SHAPES
from gpu_flow import cell
from kernel_matrix import AXES
from reference_error import assert_as_exact
from test_short_conv import TRAINING_SEEDS, cotangents, documents, operands, row_truth, run

from dew.nn.kernels import generation
from dew.nn.kernels.selection import KernelRefused
from dew.nn.linear import causal_conv1d

CHOICES = frozenset({"short_conv"})
"""The generation.CHOICES entries the family's cells run (row 184)."""

KERNEL = "pallas"
"""The conv's kernel each cell holds to XLA's: Dew's Triton kernels (`short_conv.conv`)."""


def _routed(kernel: str):
    """`test_short_conv.run`'s conv through `causal_conv1d` on `kernel` (`generation.chosen`)."""
    def conv(x, weights, activation, shift, *, valid, segments, parts, dtype):
        with generation.chosen(short_conv=kernel):
            return causal_conv1d(x, weights, activation, shift, valid=valid, segments=segments, parts=parts,
                                 dtype=dtype)
    return conv


def _step(case: dict[str, str], kernel: str):
    """`case`'s conv on `kernel`: its pieces and the gradients of the projection and the taps."""
    (_, length), parts = CONV_SHAPES[case["model"]]
    segments = documents(2, length) if case["documents"] == "packed" else None

    def step(x, weights, grads):
        out, found = run(_routed(kernel), x, weights, None, None, parts, jnp.dtype(case["dtype"]), grads,
                         segments=segments)
        return out, found[:2]

    return step


def _drawn(case: dict[str, str]):
    """A draw of `case`'s two rows, taps and cotangents by seed (`test_short_conv.operands`)."""
    (_, length), parts = CONV_SHAPES[case["model"]]
    dtype = jnp.dtype(case["dtype"])

    def draw(seed: int):
        x, weights, _, _ = operands(2, length, parts, dtype=dtype, seed=[1724, seed])
        return x, weights, cotangents(x, parts, dtype, seed=[1725, seed])

    return draw


@functools.cache
def _taken(case: tuple[tuple[str, str], ...]) -> bool:
    """Whether the conv's selector gives `case` the kernel under its generation, traced (`jax.eval_shape`)."""
    named = dict(case)
    with generation.assumed(named["generation"]), generation.routes() as found:
        try:
            jax.eval_shape(_step(named, KERNEL), *_drawn(named)(0))
        except KernelRefused:
            return False
    return any(route.kernel == KERNEL for route in found)


def supported(case: dict[str, str]) -> bool:
    """Whether the conv's selector serves `case` on its GPU (tests/kernel_matrix.py, `_taken`)."""
    return _taken(tuple(case.items()))


CASES = (
    ('sm90', '0.8b', 'bfloat16', 'one'),
    ('sm90', '9b', 'float32', 'packed'),
    ('sm80', '0.8b', 'bfloat16', 'packed'),
    ('sm80', '9b', 'float32', 'one'),
    ('sm90', '0.8b', 'float32', 'one'),
    ('sm90', '9b', 'bfloat16', 'one'),
)
"""The cells' cases, each its axis values in `AXES`' order: every pair of axis values the selector serves
(`kernel_matrix.covering("short_conv")`, regenerated where the axes or the selector's support change)."""


@cell("short_conv", covers=lambda case: dict(zip(AXES["short_conv"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=pytest.mark.skipif(
    generation.device_generation() != case[0], reason=f"an {case[0]} cell runs on its own GPU alone"))
    for case in CASES], ids="-".join)
def a_short_convs_training_step_is_on_par(case: tuple[str, ...]) -> None:
    """`case`'s conv on its GPU on the kernels, on par with XLA's conv (check 1, the float64 conv a row at
    a time the truth): its pieces and the gradients of the projection and the taps. Its routes are its
    family's (`CHOICES`)."""
    named = dict(zip(AXES["short_conv"], case, strict=True))
    (_, _), parts = CONV_SHAPES[named["model"]]
    segments = documents(2, CONV_SHAPES[named["model"]][0][1]) if named["documents"] == "packed" else None

    def exact(x, weights, grads):
        pieces, gradients = row_truth(x, weights, parts, grads, segments)
        return pieces, gradients[:2]

    with generation.routes() as found:
        assert_as_exact(_step(named, KERNEL), _step(named, "xla"), _drawn(named), "-".join(case),
                        seeds=TRAINING_SEEDS[named["model"], named["dtype"], named["documents"] == "packed"],
                        truth_fn=exact)
    assert {route.choice for route in found} <= CHOICES
