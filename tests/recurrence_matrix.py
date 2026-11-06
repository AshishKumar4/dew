"""The recurrence's check-1 matrix (tests/kernel_matrix.py): the chunked recurrence's training step on par
with the upstream's, through each kernel its routes name.

Its cells run generation.CHOICES' `gated_delta_rule` and `ssd`. Each takes the chunked op
(`recurrence_chunk.chunk`) of a form, forward and backward in one program as a layer's training step takes
them, with the kernel the cell names (`generation.chosen`):
- the recurrence template ('mosaic_gpu');
- the portable math ('xla').
The portable math's derived forms ('derived') join once they hold check 1: a call names them alone, and in
fp32 a step write gate's key gradient sat 1.54 times the portable math's distance from float64 and its final
memory 1.95 times (40 seeds, armada, recurrence matrix dry run 11-06, row 4466).

Its output, final memory and every operand's gradient are held to its reference's (check 1): the form's
port where it takes the call, fla's kernels for the delta rule's forms ('fla'), mamba_ssm's for the rated scan
('pallas'), each on its upstream's numbers in the case's dtype, as a 16-bit call or the fast forms round their
products as fla and mamba_ssm do; else the portable math as the GPU computes it; the recurrence a token at a
time in float64 the truth. What a kernel serves is the route's own: the step traced under the cell's
generation, its selectors deciding (`recurrence_chunk.port_kernel`, `scan_kernel`), a named kernel insisting
on the call and refusing what it does not take (`KernelRefused`).
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import pytest
from gpu_flow import cell
from kernel_matrix import AXES
from reference_error import FEWEST_SEEDS, assert_as_exact, with_gradients
from test_recurrence_chunk import FORMED, chunk_on, draw, token_by_token

from dew.nn.kernels import generation
from dew.nn.kernels.selection import KernelRefused

FORMS = {"gated delta net": FORMED[0], "kimi delta attention": FORMED[2], "write gate": FORMED[1],
         "rated scan": FORMED[4]}
"""Each `form` axis value's hooks (test_recurrence_chunk's): Qwen3.5's gated delta net (its gates and l2
norm), Kimi Linear's decay a key dimension with its output norm, a step write gate without the delta rule,
and Mamba-2's rated scan with its skip."""

PORTS = {"gated_delta_rule": "fla", "ssd": "pallas"}
"""Each op's port, its cells' reference where it takes the call (`_reference`)."""

CHOICES = frozenset({"gated_delta_rule", "ssd"})
"""The generation.CHOICES entries the family's cells run (row 184): the chunked op's selectors'."""


def _op(hooks) -> str:
    """The choice a form's call takes its kernel from: the rated scan's (`ssd`) or the delta rule's."""
    return "ssd" if hooks.rated and not hooks.delta else "gated_delta_rule"




def _drawn(case: dict[str, str], seed: int):
    """`case`'s operands from `seed` (`draw`, its queries, keys and values and write gate in the case's
    dtype) and fp32 cotangents of its output and final memory."""
    hooks = FORMS[case["form"]]
    (operands,) = draw(hooks, packed=case["documents"] == "packed", seed=seed)
    narrow = jnp.dtype(case["dtype"])
    operands = dataclasses.replace(operands, starts=None, **{
        name: getattr(operands, name).astype(narrow) for name in ("query", "key", "value", "beta")
        if getattr(operands, name) is not None})
    batch, length, heads, width = operands.value.shape
    keys = jax.random.split(jax.random.fold_in(jax.random.key(seed), 101))
    return operands, (jax.random.normal(keys[0], (batch, length, heads, width)),
                      jax.random.normal(keys[1], (batch, heads, operands.key.shape[-1], width)))


def _starts(case: dict[str, str]):
    """`case`'s documents' first tokens (`draw`'s, the same for every seed), None for one sequence."""
    if case["documents"] != "packed":
        return None
    (operands,) = draw(FORMS[case["form"]], packed=True, seed=0)
    return operands.starts


def _step(case: dict[str, str], kernel: str):
    """`case`'s training step through the chunked op on `kernel`: the output and final memory, and every
    operand's gradient under their cotangents (`with_gradients`)."""
    hooks, starts = FORMS[case["form"]], _starts(case)

    def ours(operands):
        return _wide(chunk_on(kernel, hooks, dataclasses.replace(operands, starts=starts)))

    return with_gradients(ours)


def _port(case: dict[str, str]) -> str:
    """`case`'s form's port (`PORTS`)."""
    return PORTS[_op(FORMS[case["form"]])]


def _wide(results):
    """A call's output and final memory in at least fp32, as its cotangents are drawn (float64 in the truth,
    which widens them): the cast's pullback rounds them to the call's own dtypes."""
    return jax.tree.map(lambda x: x.astype(jnp.promote_types(x.dtype, jnp.float32)), results)


@functools.cache
def _routed(case: tuple[tuple[str, str], ...], kernel: str) -> frozenset | None:
    """The routes (`generation.routes`: choice and kernel) `case`'s training step on `kernel` takes under its
    generation, traced (`jax.eval_shape`), the route's own selectors deciding; None where a named kernel
    refuses it (`KernelRefused`)."""
    named = dict(case)
    with generation.assumed(named["generation"]), generation.routes() as found:
        try:
            jax.eval_shape(_step(named, kernel), *_drawn(named, 0))
        except KernelRefused:
            return None
    return frozenset((route.choice, route.kernel) for route in found)


def _reference(case: dict[str, str]) -> str:
    """The kernel `case`'s cell is held to: its form's port where the port takes the call (`PORTS`, the
    upstream's numbers, which a 16-bit call and the fast forms round as), else the portable math ('xla'), as
    the cell's GPU computes it at the run's precision (a step write gate's form, which no port runs)."""
    port = _port(case)
    taken = _routed(tuple(case.items()), port) or frozenset()
    return port if any(kernel == port for _, kernel in taken) else "xla"


def supported(case: dict[str, str]) -> bool:
    """Whether the route serves `case` (tests/kernel_matrix.py): its kernel takes the case's training call
    (`_routed`, the route's own selectors: not refused, nor handed to another kernel) and is not its own
    reference (`_reference`)."""
    taken = _routed(tuple(case.items()), case["kernel"])
    return (taken is not None and any(kernel == case["kernel"] for _, kernel in taken)
            and _reference(case) != case["kernel"])


CASES = (
    ('sm90', 'gated delta net', 'mosaic_gpu', 'bfloat16', 'one'),
    ('sm90', 'kimi delta attention', 'xla', 'float32', 'packed'),
    ('sm80', 'rated scan', 'mosaic_gpu', 'bfloat16', 'packed'),
    ('sm80', 'gated delta net', 'xla', 'float32', 'one'),
    ('sm90', 'write gate', 'mosaic_gpu', 'float32', 'one'),
    ('sm90', 'rated scan', 'xla', 'bfloat16', 'one'),
    ('sm80', 'kimi delta attention', 'xla', 'bfloat16', 'one'),
    ('sm90', 'gated delta net', 'mosaic_gpu', 'bfloat16', 'packed'),
    ('sm90', 'write gate', 'mosaic_gpu', 'float32', 'packed'),
    ('sm90', 'rated scan', 'mosaic_gpu', 'float32', 'one'),
)
"""The cells' cases, each its axis values in `AXES`' order: every pair of axis values the route serves
(`kernel_matrix.covering("recurrence")`, regenerated where the axes or the route's support change)."""


@cell("recurrence", covers=lambda case: dict(zip(AXES["recurrence"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=pytest.mark.skipif(
    generation.device_generation() != case[0], reason=f"an {case[0]} cell runs on its own GPU alone"))
    for case in CASES], ids="-".join)
def a_recurrences_training_step_is_on_par(case: tuple[str, ...]) -> None:
    """The chunked op's training step of `case` on its GPU, its kernel named, on par with its reference's
    (`_reference`; check 1, the recurrence a token at a time in float64 the truth): the output, the final
    memory and every operand's gradient. Its routes are its family's (`CHOICES`)."""
    named = dict(zip(AXES["recurrence"], case, strict=True))
    hooks, starts = FORMS[named["form"]], _starts(named)

    def truth(operands):
        return _wide(token_by_token(hooks, dataclasses.replace(operands, starts=starts)))

    with generation.routes() as found:
        assert_as_exact(jax.jit(_step(named, named["kernel"])), jax.jit(_step(named, _reference(named))),
                        functools.partial(_drawn, named), "-".join(case), seeds=FEWEST_SEEDS,
                        truth_fn=with_gradients(truth))
    assert {route.choice for route in found} <= CHOICES
