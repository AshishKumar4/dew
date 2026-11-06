"""The recurrent decode's check-1 matrix (tests/kernel_matrix.py): a recurrent layer's decode tokens on par
with the portable step's.

Its cells run generation.CHOICES' `delta_decode` (`linear.DELTA_DECODE`): a layer's tokens decoded in turn
after a prefill, through the layer, as a server decodes them, on the kernel the cell names: the fused step
('fused', which takes the layer's conv and gates, and its gated norm where it norms a head) or the step alone
('step'), its recurrent state held in fp32 or bf16 where the layer holds it so. The layers: a gated delta
net, Kimi delta attention and Mamba-2, each at a model's widths. Each token's output, and the conv history
and recurrent state it leaves, are held to the portable step's ('xla', check 1), the layer in float64 the
truth. The other choices a decode token reaches run at their references in both arms (`REFERENCES`,
named), so a cell measures its kernel alone. What a kernel serves is the decode selector's, as the layer
asks it.
"""

import dataclasses
import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from gpu_flow import cell
from kernel_matrix import AXES
from reference_error import FEWEST_SEEDS, assert_as_exact
from test_linear_attention import fused_case

from dew.nn.inputs import MixerCall
from dew.nn.kda import KimiDeltaAttention
from dew.nn.kernels import generation
from dew.nn.kernels.selection import KernelRefused
from dew.nn.linear import decode_kernel
from dew.nn.mixers.mamba2 import Mamba2

CHOICES = frozenset({"delta_decode"})
"""The generation.CHOICES entries the family's cells run (row 184)."""

KERNELS = {"delta_decode": frozenset({"fused", "step"})}
"""The kernels each choice's cells run under test (row 4513): the fused step and the step."""

REFERENCES = {"grouped_matmul": "xla", "rms_norm": "xla", "short_conv": "xla", "delta_gates": "apart"}
"""The other choices a layer's decode token reaches, each at its generic kernel: its projections, its
norm, its conv and its gates' join."""

DECODED = 4
"""The tokens a cell decodes in turn after its prefill."""


@dataclasses.dataclass(frozen=True)
class Layer:
    """A matrix layer: its module at a dtype (`make`), its draw of variables, cache after a prefill and
    `DECODED` tokens by seed and held state's dtype (`drawn`), the cache's recurrent state's name, its
    state's `[rows, Hv, Dk, Dv]` shape, whether it holds a bf16 state and whether its fused step norms a
    head (`linear.decode_kernel`'s `normed`)."""

    make: Callable
    drawn: Callable
    state: str
    shape: tuple[int, ...]
    holds_bf16: bool
    normed: bool


def _prefilled(module, variables, seed: int, width: int):
    """`module`'s cache after a 3-token prefill of three rows and `DECODED` tokens after it, by seed."""
    x = jnp.asarray(np.random.default_rng(seed).standard_normal((3, 3 + DECODED, width)).astype(np.float32))
    step = MixerCall(decode=True)
    allocated = module.apply(variables, x[:, :1], step, mutable=["cache"])[1]
    cache = module.apply({**variables, **allocated}, x[:, :3], step, mutable=["cache"])[1]["cache"]
    return cache, x[:, 3:]


def _delta_net(seed: int, held):
    """A gated delta net's draw (`fused_case`, Qwen3.5's widths at four heads), its state held in `held`."""
    _, variables, cache, token = fused_case(4, 128, 3, jnp.bfloat16, held, seed)
    later = jax.random.normal(jax.random.key(3000 + seed), (token.shape[0], DECODED - 1, token.shape[-1]))
    return variables, cache, jnp.concatenate([token, later.astype(token.dtype)], axis=1)


def _kimi(dtype=jnp.float32):
    """Kimi delta attention at two heads of 128."""
    return KimiDeltaAttention(emb_features=64, num_heads=2, head_dim=128, conv_kernel=4, chunk_size=4,
                              dtype=dtype)


def _mamba2(dtype=jnp.float32):
    """Mamba-2 at four heads of 64 over a group of 128 states (Mamba-2 2.7B's head and state widths)."""
    return Mamba2(emb_features=64, num_heads=4, head_dim=64, state_size=128, n_groups=1, conv_kernel=4,
                  chunk_size=4, norm_eps=1e-5, dtype=dtype)


def _drawn_of(make):
    """A draw of `make()`'s variables by seed, its cache after a prefill and its tokens (`_prefilled`)."""
    def draw(seed: int, held):
        del held
        module = make()
        variables = nn.unbox(module.init(jax.random.key(seed), jnp.zeros((3, 1, 64))))
        return variables, *_prefilled(module, variables, seed, 64)
    return draw


LAYERS = {
    "gated delta net": Layer(lambda dtype=jnp.bfloat16: fused_case(4, 128, 3, dtype, jnp.float32)[0],
                             _delta_net, "recurrent_state", (3, 4, 128, 128), holds_bf16=True, normed=True),
    "kimi delta attention": Layer(_kimi, _drawn_of(_kimi), "recurrent_state", (3, 2, 128, 128),
                                  holds_bf16=False, normed=True),
    "mamba2": Layer(_mamba2, _drawn_of(_mamba2), "ssm_state", (3, 4, 128, 64), holds_bf16=False,
                    normed=False),
}
"""Each layer the decode matrix runs, by its `layer` axis value."""


def _decoded(module, kernel: str, state: str):
    """`module`'s `DECODED` tokens decoded in turn on `kernel`: each token's output, and the conv history and
    recurrent state (the cache's `state`) they leave."""
    def decode(variables, cache, tokens):
        outs = []
        with generation.chosen(delta_decode=kernel, **REFERENCES):
            for at in range(DECODED):
                out, after = module.apply({**variables, "cache": cache}, tokens[:, at:at + 1],
                                          MixerCall(decode=True), mutable=["cache"])
                outs.append(out)
                cache = after["cache"]
        return jnp.concatenate(outs, axis=1), cache["conv_state"], cache[state]
    return decode


@functools.cache
def _taken(case: tuple[tuple[str, str], ...]) -> bool:
    """Whether the decode selector gives `case`'s tokens its kernel under its generation, named as the cell
    names it, as the layer asks it (`linear.decode_kernel`), where the layer holds the case's state."""
    named = dict(case)
    layer = LAYERS[named["layer"]]
    if named["state"] == "bfloat16" and not layer.holds_bf16:
        return False
    state = jax.ShapeDtypeStruct(layer.shape, jnp.dtype(named["state"]))
    with generation.assumed(named["generation"]), generation.chosen(delta_decode=named["kernel"]):
        try:
            return decode_kernel(state, fuses=True, normed=layer.normed) == named["kernel"]
        except KernelRefused:
            return False


def _choices(found) -> set[tuple[str, str]]:
    """The recorded routes' choices and kernels."""
    return {(route.choice, route.kernel) for route in found}


def supported(case: dict[str, str]) -> bool:
    """Whether the decode selector serves `case` on its GPU (tests/kernel_matrix.py, `_taken`)."""
    return _taken(tuple(case.items()))


CASES = (
    ('sm90', 'gated delta net', 'fused', 'float32'),
    ('sm80', 'gated delta net', 'step', 'bfloat16'),
    ('sm90', 'kimi delta attention', 'step', 'float32'),
    ('sm80', 'mamba2', 'fused', 'float32'),
    ('sm90', 'gated delta net', 'fused', 'bfloat16'),
    ('sm90', 'mamba2', 'step', 'float32'),
    ('sm80', 'kimi delta attention', 'fused', 'float32'),
)
"""The cells' cases, each its axis values in `AXES`' order: every pair of axis values the selector serves
(`kernel_matrix.covering("delta_decode")`, regenerated where the axes or the selector's support change)."""


@cell("delta_decode", covers=lambda case: dict(zip(AXES["delta_decode"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=pytest.mark.skipif(
    generation.device_generation() != case[0], reason=f"an {case[0]} cell runs on its own GPU alone"))
    for case in CASES], ids="-".join)
def a_decode_token_is_on_par(case: tuple[str, ...]) -> None:
    """A layer's `DECODED` decode tokens of `case` on its GPU on its kernel, on par with the portable step's
    (check 1, the layer in float64 the truth: a bf16 layer on widened arguments fixes its projections and
    norm in bf16 and casts its state back, so only the stored state's rounding is the contract's). Its
    routes are its family's (`CHOICES`), every other choice's at its reference (`REFERENCES`)."""
    named = dict(zip(AXES["delta_decode"], case, strict=True))
    layer = LAYERS[named["layer"]]
    held = jnp.dtype(named["state"])
    module = layer.make()
    with generation.routes() as found:
        assert_as_exact(_decoded(module, named["kernel"], layer.state), _decoded(module, "xla", layer.state),
                        functools.partial(layer.drawn, held=held), "-".join(case), seeds=FEWEST_SEEDS,
                        truth_fn=_decoded(module.clone(dtype=jnp.float64), "xla", layer.state))
    assert all(choice in CHOICES or REFERENCES.get(choice) == kernel for choice, kernel in _choices(found))
    assert ("delta_decode", named["kernel"]) in _choices(found)
