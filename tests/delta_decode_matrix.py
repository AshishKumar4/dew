"""The recurrent decode's check-1 matrix (tests/kernel_matrix.py): a gated delta net's decode tokens on par
with the portable step's.

Its cells run generation.CHOICES' `delta_decode` (`linear.DELTA_DECODE`): a gated delta net's tokens decoded
in turn after a prefill, through the layer, as a server decodes them, on the kernel the cell names: the fused
step ('fused', which takes the layer's conv, gates and gated norm) or the step alone ('step'), its recurrent
state held in fp32 or bf16. Each token's output, and the conv history and recurrent state it leaves, are held
to the portable step's ('xla', check 1), the layer in float64 the truth. The other choices a decode token
reaches run at their references in both arms (`REFERENCES`, named), so a cell measures its kernel alone.
What a kernel serves is the decode selector's, as the layer asks it.
"""

import functools

import jax
import jax.numpy as jnp
import pytest
from gpu_flow import cell
from kernel_matrix import AXES
from reference_error import FEWEST_SEEDS, assert_as_exact
from test_linear_attention import fused_case

from dew.nn.inputs import MixerCall
from dew.nn.kernels import generation
from dew.nn.kernels.selection import KernelRefused

CHOICES = frozenset({"delta_decode"})
"""The generation.CHOICES entries the family's cells run (row 184)."""

REFERENCES = {"grouped_matmul": "xla", "rms_norm": "xla", "short_conv": "xla", "delta_gates": "apart"}
"""The other choices a gated delta net's decode token reaches, each at its generic kernel: its projections,
its gated norm, its conv and its gates' join."""

DECODED = 4
"""The tokens a cell decodes in turn after its prefill."""


def _decoded(module, kernel: str):
    """`module`'s `DECODED` tokens decoded in turn on `kernel`: each token's output, and the conv history and
    recurrent state they leave."""
    def decode(variables, cache, tokens):
        outs = []
        with generation.chosen(delta_decode=kernel, **REFERENCES):
            for at in range(DECODED):
                out, after = module.apply({**variables, "cache": cache}, tokens[:, at:at + 1],
                                          MixerCall(decode=True), mutable=["cache"])
                outs.append(out)
                cache = after["cache"]
        return jnp.concatenate(outs, axis=1), cache["conv_state"], cache["recurrent_state"]
    return decode


def _drawn(held):
    """A draw of a layer's variables, its cache after a prefill and `DECODED` tokens by seed (`fused_case`,
    Qwen3.5's widths at four heads), its recurrent state held in `held`."""
    def draw(seed: int):
        _, variables, cache, token = fused_case(4, 128, 3, jnp.bfloat16, held, seed)
        later = jax.random.normal(jax.random.key(3000 + seed), (token.shape[0], DECODED - 1, token.shape[-1]))
        return variables, cache, jnp.concatenate([token, later.astype(token.dtype)], axis=1)
    return draw


@functools.cache
def _taken(case: tuple[tuple[str, str], ...]) -> bool:
    """Whether the decode selector gives `case`'s tokens its kernel under its generation, traced
    (`jax.eval_shape`), the layer asking it."""
    named = dict(case)
    held = jnp.dtype(named["state"])
    layer = fused_case(4, 128, 3, jnp.bfloat16, held)[0]
    with generation.assumed(named["generation"]), generation.routes() as found:
        try:
            jax.eval_shape(_decoded(layer, named["kernel"]), *_drawn(held)(0))
        except KernelRefused:
            return False
    return any(route == ("delta_decode", named["kernel"]) for route in _choices(found))


def _choices(found) -> set[tuple[str, str]]:
    """The recorded routes' choices and kernels."""
    return {(route.choice, route.kernel) for route in found}


def supported(case: dict[str, str]) -> bool:
    """Whether the decode selector serves `case` on its GPU (tests/kernel_matrix.py, `_taken`)."""
    return _taken(tuple(case.items()))


CASES = ()
"""The cells' cases, each its axis values in `AXES`' order: every pair of axis values the selector serves
(`kernel_matrix.covering("delta_decode")`, regenerated where the axes or the selector's support change)."""


@cell("delta_decode", covers=lambda case: dict(zip(AXES["delta_decode"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=pytest.mark.skipif(
    generation.device_generation() != case[0], reason=f"an {case[0]} cell runs on its own GPU alone"))
    for case in CASES], ids="-".join)
def a_decode_token_is_on_par(case: tuple[str, ...]) -> None:
    """A gated delta net's `DECODED` decode tokens of `case` on its GPU on its kernel, on par with the
    portable step's (check 1, the layer in float64 the truth: a bf16 layer on widened arguments fixes its
    projections and norm in bf16 and casts its state back, so only the stored state's rounding is the
    contract's). Its routes are its family's (`CHOICES`), every other choice's at its reference
    (`REFERENCES`)."""
    named = dict(zip(AXES["delta_decode"], case, strict=True))
    held = jnp.dtype(named["state"])
    layer = fused_case(4, 128, 3, jnp.bfloat16, held)[0]
    with generation.routes() as found:
        assert_as_exact(_decoded(layer, named["kernel"]), _decoded(layer, "xla"), _drawn(held),
                        "-".join(case), seeds=FEWEST_SEEDS,
                        truth_fn=_decoded(layer.clone(dtype=jnp.float64), "xla"))
    assert all(choice in CHOICES or REFERENCES.get(choice) == kernel for choice, kernel in _choices(found))
