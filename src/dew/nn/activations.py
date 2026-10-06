"""The elementwise activations, one arithmetic each, and the names that pick them.

The gelus and silu compute in at least float32 and round once to the input's
dtype, as torch's bf16 kernels do (docs/performance.md): a bf16 polynomial or
erf moves the negative tail before it rounds. The exact gelu has two forms,
each kept for a measured reason. `gelu_exact` is the erfc form, the closest
to float64, which the towers' 'gelu' and the gated 'geglu_exact' compute;
`gelu_exact_torch` is torch's `1 + erf`, which the decoders' ungated
'gelu_exact' and ModernBERT's head compute, since their converted
checkpoints were trained with its rounding, and DiT's exact MLP, where it
made an A100 training step 9.8% faster (`dew.nn.dit`). `gelu_tanh` is the tanh
approximation, torch's `gelu(approximate='tanh')`. `quick_gelu` stays in
the input's dtype, as transformers' `QuickGELUActivation` does, and relu and
relu2 are exact in any dtype but for relu2's one rounded product.

Two tables name them. `activation` reads Hugging Face's `hidden_act` names
(transformers 5.16.1 `activations.ACT2CLS`): 'gelu' is the erfc form there.
Every tower checks its own subset of names before asking, and refuses the
rest in its own words. `UNGATED` holds Dew's names for an ungated decoder
feed-forward, which saved run records store: 'gelu' is the tanh form there and
'gelu_exact' torch's erf form. The gated names map onto these in
`dew.nn.moe.gated_product`, which owns the product's rounding.
"""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp

from .precision import at_least_fp32


def gelu_exact(x: jax.Array) -> jax.Array:
    """`0.5 * x * erfc(-x / sqrt(2))`, `jax.nn.gelu(approximate=False)`'s arithmetic, in at
    least fp32, in `x`'s dtype. Over 2^20 normal fp32 inputs at scale 3 it lies 5.6e-8 RMS
    from float64 where the `1 + erf` form and torch's gelu lie 1.0e-7, and 4.5e-10 against
    1.3e-7 below x = -3. In bf16 both forms lie 1.76e-3 RMS from float64."""
    work = x.astype(at_least_fp32(x.dtype))
    return (.5 * work * jax.lax.erfc(-work * math.sqrt(.5))).astype(x.dtype)


def gelu_exact_torch(x: jax.Array) -> jax.Array:
    """Torch's `0.5 * x * (1 + erf(x / sqrt(2)))` in at least fp32, in `x`'s dtype, for converted
    decoders: in fp32 its `1 + erf` rounds to zero from x = -5.21 down (torch's from -5.54), where
    the erfc form keeps a tail that a trained layer amplifies (tests/test_hf_gpt_neox.py)."""
    work = x.astype(at_least_fp32(x.dtype))
    return (.5 * work * (1 + jax.lax.erf(work * math.sqrt(.5)))).astype(x.dtype)


def gelu_tanh(x: jax.Array) -> jax.Array:
    """`0.5 * x * (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3)))` in at least fp32, in `x`'s dtype."""
    return jax.nn.gelu(x.astype(at_least_fp32(x.dtype)), approximate=True).astype(x.dtype)


def quick_gelu(x: jax.Array) -> jax.Array:
    """CLIP's activation, `x * sigmoid(1.702 x)`, ACT2FN's `quick_gelu`."""
    return x * jax.nn.sigmoid(1.702 * x)


def silu(x: jax.Array) -> jax.Array:
    """`x * sigmoid(x)` in at least fp32, in `x`'s dtype."""
    return jax.nn.silu(x.astype(at_least_fp32(x.dtype))).astype(x.dtype)


def relu2(x: jax.Array) -> jax.Array:
    """The squared relu, Nemotron-H's and ACT2FN's `relu2`."""
    return jnp.square(jax.nn.relu(x))


Activation = Callable[[jax.Array], jax.Array]
_HIDDEN_ACT: dict[str, Activation] = {'gelu': gelu_exact, 'gelu_pytorch_tanh': gelu_tanh,
                                      'quick_gelu': quick_gelu, 'silu': silu, 'relu': jax.nn.relu}


def activation(name: str) -> Activation:
    """The activation Hugging Face's `hidden_act` `name` computes; callers check `name` first."""
    return _HIDDEN_ACT[name]


UNGATED: dict[str, Activation] = {
    'gelu': gelu_tanh, 'gelu_exact': gelu_exact_torch, 'relu': jax.nn.relu, 'relu2': relu2}
"""The activations an ungated feed-forward takes, by Dew's names."""


def ungated_activation(name: str, x: jax.Array) -> jax.Array:
    """`x` through the `UNGATED` activation `name`, in `x`'s dtype; callers check `name` first."""
    return UNGATED[name](x)
