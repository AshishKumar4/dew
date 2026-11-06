"""The recurrence's check-1 matrix (tests/kernel_matrix.py): the chunked recurrence's training step on par
with the upstream's, through each kernel its routes name.

Its cells run generation.CHOICES' `gated_delta_rule` and `ssd`. Each takes the chunked op
(`recurrence_chunk.chunk`) of a form, forward and backward in one program as a layer's training step takes
them, with the kernel the cell names (`generation.chosen`):
- the recurrence template ('mosaic_gpu');
- the portable math ('xla');
- its derived forms ('derived', the delta rule's).

Its output, final memory and every operand's gradient are held to the form's port's (check 1): fla's kernels
for the delta rule's forms ('fla'), mamba_ssm's for the rated scan ('pallas'), each on its upstream's numbers
in the case's dtype, as a 16-bit call or the fast forms round their products as fla and mamba_ssm do; the
recurrence a token at a time in float64 the truth. What a kernel serves is the route's own: the step traced
under the cell's generation, its selectors deciding (`recurrence_chunk.port_kernel`, `scan_kernel`), a named
kernel insisting on the call and refusing what it does not take (`KernelRefused`).
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
"""Each op's port, its cells' reference."""


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
def _served(case: tuple[tuple[str, str], ...]) -> bool:
    """Whether the route takes `case`'s training step on its kernel under its generation: traced
    (`jax.eval_shape`), the route's own selectors deciding, a named kernel refusing what it does not take
    (`KernelRefused`)."""
    named = dict(case)
    if named["kernel"] == "derived" and _op(FORMS[named["form"]]) == "ssd":
        return False
    with generation.assumed(named["generation"]):
        try:
            for kernel in (named["kernel"], _port(named)):
                jax.eval_shape(_step(named, kernel), *_drawn(named, 0))
        except KernelRefused:
            return False
    return True


def supported(case: dict[str, str]) -> bool:
    """Whether the route serves `case` (tests/kernel_matrix.py): its kernel and its form's port take the
    case's training call (`_served`, the route's own selector)."""
    return _served(tuple(case.items()))


CASES = (
    ('sm90', 'gated delta net', 'mosaic_gpu', 'bfloat16', 'one'),
    ('sm90', 'kimi delta attention', 'port', 'float32', 'packed'),
    ('sm80', 'write gate', 'xla', 'bfloat16', 'packed'),
    ('sm80', 'gated delta net', 'derived', 'float32', 'one'),
    ('sm90', 'rated scan', 'xla', 'float32', 'one'),
    ('sm80', 'kimi delta attention', 'port', 'bfloat16', 'one'),
    ('sm80', 'rated scan', 'mosaic_gpu', 'bfloat16', 'packed'),
    ('sm90', 'write gate', 'mosaic_gpu', 'float32', 'one'),
    ('sm90', 'gated delta net', 'derived', 'bfloat16', 'packed'),
    ('sm90', 'gated delta net', 'port', 'bfloat16', 'one'),
    ('sm90', 'gated delta net', 'xla', 'bfloat16', 'one'),
    ('sm90', 'kimi delta attention', 'xla', 'bfloat16', 'one'),
    ('sm90', 'kimi delta attention', 'derived', 'bfloat16', 'one'),
    ('sm90', 'write gate', 'port', 'bfloat16', 'one'),
    ('sm90', 'write gate', 'derived', 'bfloat16', 'one'),
    ('sm90', 'rated scan', 'port', 'bfloat16', 'one'),
)
"""The cells' cases, each its axis values in `AXES`' order: every pair of axis values the route serves
(`kernel_matrix.covering("recurrence")`, regenerated where the axes or the route's support change)."""


@cell("recurrence", covers=lambda case: dict(zip(AXES["recurrence"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=pytest.mark.skipif(
    generation.device_generation() != case[0], reason=f"an {case[0]} cell runs on its own GPU alone"))
    for case in CASES], ids="-".join)
def a_recurrences_training_step_is_on_par(case: tuple[str, ...]) -> None:
    """The chunked op's training step of `case` on its GPU, its kernel named, on par with its form's port's
    (check 1, the recurrence a token at a time in float64 the truth): the output, the final memory and every
    operand's gradient."""
    named = dict(zip(AXES["recurrence"], case, strict=True))
    hooks, starts = FORMS[named["form"]], _starts(named)

    def truth(operands):
        return _wide(token_by_token(hooks, dataclasses.replace(operands, starts=starts)))

    assert_as_exact(jax.jit(_step(named, named["kernel"])), jax.jit(_step(named, _port(named))),
                    functools.partial(_drawn, named), "-".join(case), seeds=FEWEST_SEEDS,
                    truth_fn=with_gradients(truth))
