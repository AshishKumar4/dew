"""The selective scan's check-1 matrix (tests/kernel_matrix.py): Mamba's selective scan on the template on par
with recorded mamba_ssm.

Its cells run generation.CHOICES' `selective_scan`: each recorded case (tools/mamba_selective_reference.py's
`SEEDED_CASES`: bf16 and fp32 operands as the mixer hands them, each regime of the decays and steps, packed
documents, 4096 tokens) through the scan's selector with the template named, under the run's fast forms, its
output and every input's gradient held to recorded mamba_ssm's on the case's declared draws (check 1,
`test_selective_template.the_template_is_as_exact_as_mamba_ssm`): the template sums at mamba_ssm's rounding,
which the portable scan compensates past, so the upstream's own numbers are its reference. What it serves is
the selector's (`recurrence_chunk.SELECTIVE`: the template qualified on an H100 alone).
"""

import functools

import jax
import jax.numpy as jnp
import pytest
from gpu_flow import cell
from kernel_matrix import AXES
from selective_support import SEEDED_CASES, selective_draw, selective_xla
from test_selective_template import the_template_is_as_exact_as_mamba_ssm

from dew.nn.kernels import generation, recurrence_chunk
from dew.nn.mixers.mamba import selective_scan as portable

CHOICES = frozenset({"selective_scan"})
"""The generation.CHOICES entries the family's cells run (row 184)."""


@functools.cache
def _taken(case: tuple[tuple[str, str], ...]) -> bool:
    """Whether the scan's selector, deciding (no kernel named: its measured qualification read), gives
    `case`'s recorded call the template under its generation, traced (`jax.eval_shape`) as the cell runs
    it."""
    named = dict(case)
    *_, dtype, _ = SEEDED_CASES[named["case"]]
    scan = functools.partial(recurrence_chunk.selective_scan, portable=portable)
    arrays = selective_draw(0, named["case"])
    with generation.assumed(named["generation"]), generation.routes() as found:
        jax.eval_shape(functools.partial(selective_xla, dtype=jnp.dtype(dtype), scan=scan), arrays)
    return any(route.kernel == "mosaic_gpu" for route in found)


def supported(case: dict[str, str]) -> bool:
    """Whether the scan's selector serves `case` on its GPU (tests/kernel_matrix.py, `_taken`)."""
    return _taken(tuple(case.items()))


CASES = ()
"""The cells' cases, each its axis values in `AXES`' order: every pair of axis values the selector serves
(`kernel_matrix.covering("selective_scan")`, regenerated where the axes or the selector's support change)."""


@cell("selective_scan", covers=lambda case: dict(zip(AXES["selective_scan"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=pytest.mark.skipif(
    generation.device_generation() != case[0], reason=f"an {case[0]} cell runs on its own GPU alone"))
    for case in CASES], ids="-".join)
def a_selective_scan_is_as_exact_as_mamba_ssm(case: tuple[str, ...]) -> None:
    """`case`'s recorded call on its GPU on the template, on par with recorded mamba_ssm (check 1,
    `the_template_is_as_exact_as_mamba_ssm`). Its routes are its family's (`CHOICES`)."""
    named = dict(zip(AXES["selective_scan"], case, strict=True))
    with generation.routes() as found:
        the_template_is_as_exact_as_mamba_ssm(named["case"])
    assert {route.choice for route in found} <= CHOICES
