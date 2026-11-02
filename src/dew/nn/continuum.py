"""A continuum level: the parameters a model's own task loss steps in context, a window at a time.

Nested Learning's continuum memory (Behrouz et al., arXiv 2512.24695, section 7.1) updates a block's
parameters every `C` tokens by the gradient of the task's own objective, equation 71:
theta_{i+1} = theta_i - sum_{t in the window} eta_t grad L(theta_t; x_t), at each window's end. The
level here is that block: a residual two-layer map `x + U act(D x)`, equation 89's memory form, read on
a decoder's final states before its head (`CausalTransformer.continuum`). There each token's loss
reads the level through its own state alone, so a window's gradient is the sum of its tokens' own,
equation 71's per-token term exactly. `U` starts at zero, so a model gains the level as the identity:
its logits are the model's without it. The objective that steps it in context is
`dew.objectives.lm.adaptation`; a plain one trains the level as any other parameters.
"""

import dataclasses
import math
from collections.abc import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.typing import Dtype, PrecisionLike

from dew.nn.activations import _SLOPES, activation
from dew.nn.dense import Dense
from dew.nn.mlp import GatedMLP
from dew.nn.precision import at_least_fp32
from dew.nn.sharding import logical_axes
from dew.quantization import InputQuantizer, quantized_products

Quantizers = Callable[[nn.Module], InputQuantizer | None]
"""A model's checkpoint quantizer of a layer's input (`CausalTransformer._quantizer`), None for none."""


@dataclasses.dataclass(frozen=True)
class ContinuumLevel:
    """A continuum level's shape: its `hidden` width and its `activation`, a Hugging Face
    `hidden_act` name (`dew.nn.activations.activation`)."""

    hidden: int
    activation: str = "silu"

    def __post_init__(self):
        if self.hidden < 1:
            raise ValueError(f"a continuum level is at least one unit wide, not {self.hidden}")
        activation(self.activation)


@logical_axes({
    ("continuum", "down"): ("embed", "mlp"),
    ("continuum", "up"): ("mlp", "embed"),
})
class Continuum(nn.Module):
    """The level `ContinuumLevel` describes over `[..., features]` states: `x + up(act(down(x)))`,
    `up` zero at its start."""

    level: ContinuumLevel
    features: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        # The ungated feed-forward `down`, `up`, the latter zero at its start; its first projection is
        # the feed-forward's `up_proj`, named `down` here, and its output `down_proj`, named `up`.
        self.mlp = GatedMLP(self.level.hidden, self.features, activation(self.level.activation),
                            output_init_std=0.0, linear=Dense, names=("down", "up"), dtype=self.dtype,
                            precision=self.precision)
        nn.share_scope(self, self.mlp)

    def __call__(self, x: jax.Array) -> jax.Array:
        # The residual sums in at least fp32 and rounds once to the states' dtype, the head's input.
        wide = at_least_fp32(x.dtype)
        return (x.astype(wide) + self.mlp(x).astype(wide)).astype(x.dtype)

    @property
    def down(self) -> nn.Module:
        """The level's first map, D: the feed-forward's `up_proj`, stored as `down`."""
        return self.mlp.up_proj

    @property
    def up(self) -> nn.Module:
        """The level's output map, U, zero at its start: the feed-forward's `down_proj`, stored as `up`."""
        return self.mlp.down_proj

    def applied_kernel(self, name: str) -> jax.Array:
        """The kernel the map `name` (`down`, `up`) multiplies its input by on this call: its own. A method
        interceptor that changes that answers for itself (`CausalTransformer.applied_kernel`): `dew.lora`'s
        adds its factors' update, and a frozen FP8 site's is its codes decoded."""
        return getattr(self, name).get_variable("params", "kernel")

    def nonlinear_input(self, name: str) -> bool:
        """Whether this call computes the map `name` other than as a linear map of its input: as the Qwix
        quantization the call runs under quantizes its activations or gradients
        (`dew.quantization.quantized_products`), and as a method interceptor answers for itself (a
        checkpoint's input quantization and frozen FP8 products, `dew.lora`'s dropout on a training
        forward)."""
        return quantized_products(self, name).inputs

    def _step_terms(self, x: jax.Array, parts: tuple[str, ...], quantizer: Quantizers
                   ) -> tuple[jax.Array, jax.Array, jax.Array, float, float]:
        """What bounds a damped step of the level's `parts` (`down`, `up`) at states `x` `[..., features]`
        (`dew.objectives.lm.adaptation.damping`): each state's |x|^2 and |f|^2, x and f = act(down(x)) as
        `down` and `up` multiply them (`quantizer`, a model's checkpoint quantizer of a map's input, None
        for none), the Frobenius norm of the kernel `up` multiplies by (`applied_kernel`, an adapter's
        update on it included), at least its spectral norm, in at least fp32; and the activation's bounds
        p1, p2 on |act'| and |act''| (`_SLOPES`).

        A map whose product rounds its kernel (Qwix's quantized weights) makes the loss piecewise constant
        in that kernel, and an `up` that computes other than the linear map of its input by its kernel
        (`nonlinear_input`, or its kernel rounded) makes the level's output piecewise constant in `down`,
        through its rounding of f: the gradient jumps at every rounding boundary, and no curvature bounds
        a step there, refused, as is a step of `down` through an activation whose act'' no number bounds
        (ReLU's, a point mass at zero). A quantizer of `down`'s input rounds the states, which no step
        moves."""
        p1, p2 = _SLOPES[activation(self.level.activation)]
        if "down" in parts and p2 is None:
            raise ValueError("a damped step of the level's `down` bounds its loss's curvature through its "
                             "activation's second derivative, which a ReLU has none of (a point mass at "
                             "zero), so no rate bounds it there: select=('up',), or a smooth activation")
        for name in parts:
            if quantized_products(self, name).weights:
                raise ValueError(f"a damped step of the level's `{name}` bounds the loss's curvature in its "
                                 "kernel, and this call's quantization rounds that kernel in its product, "
                                 "making the loss piecewise constant in it, which no curvature bounds: step "
                                 "it undamped")
        if "down" in parts and (self.nonlinear_input("up") or quantized_products(self, "up").weights):
            raise ValueError("a damped step of the level's `down` bounds the loss's curvature through the "
                             "map `up` applies to f = act(D x), and this call's `up` rounds its input or its "
                             "kernel, making the level's output piecewise constant in `down`, which no "
                             "curvature bounds: select=('up',), or step it undamped")

        def read(value: jax.Array, layer: nn.Module) -> jax.Array:
            rounding = quantizer(layer)
            return (value if rounding is None else rounding.qdq(value)).astype(at_least_fp32(x.dtype))

        fed = read(activation(self.level.activation)(self.down(x)), self.up)
        kernel = self.applied_kernel("up").astype(at_least_fp32(x.dtype))
        return (jnp.sum(jnp.square(read(x, self.down)), axis=-1), jnp.sum(jnp.square(fed), axis=-1),
                jnp.sqrt(jnp.sum(jnp.square(kernel))), p1, p2 or 0.0)


def _head_loss_bounds(table: jax.Array, softcap: float | None = None, transform: tuple | None = None
                     ) -> tuple[jax.Array, jax.Array]:
    """Bounds over every state on the curvature and the slope of a slot's cross entropy in the states a
    head reads, `table` the squared Frobenius norm |A|^2 of its rows A, at least the spectral: an affine
    head's are |A|^2 / 2 and sqrt 2 |A| (the cross entropy's Hessian in the logits is at most 1/2, its
    gradient p - y at most sqrt 2); a logit `softcap` c tanh(u / c), of slope at most 1, adds
    |A|^2 4 / (3 sqrt 3 c) of curvature, of either sign; a `transform` N(phi(D s)) before A, given as
    ((p1, p2), |D|^2, g, e) with N a norm of largest gain g and epsilon e and p1, p2 phi's `_SLOPES`, has a
    Jacobian of at most |g| p1 |D| / sqrt e and a second derivative of at most
    |g| (6 p1^2 / e + p2 / sqrt e) |D|^2 (a norm's are |g| / sqrt e and 6 |g| / e). Constants of the step,
    not differentiated (`dew.objectives.lm.adaptation.damping`)."""
    curvature, slope = table / 2, jnp.sqrt(2 * table)
    if softcap is not None:
        curvature = table * (0.5 + 4 / (3 * math.sqrt(3) * softcap))
    if transform is not None:
        (p1, p2), spread, gain, epsilon = transform
        jacobian = jnp.sqrt(gain ** 2 * p1 ** 2 * spread / epsilon)
        bent = gain * (6 * p1 ** 2 / epsilon + (p2 or 0) / math.sqrt(epsilon)) * spread
        curvature, slope = jacobian ** 2 * curvature + slope * bent, jacobian * slope
    return jax.lax.stop_gradient(curvature), jax.lax.stop_gradient(slope)


def _squared_norm(value: jax.Array) -> jax.Array:
    """The sum of `value`'s squares, in at least fp32: a matrix's squared Frobenius norm."""
    return jnp.sum(jnp.square(value.astype(at_least_fp32(value.dtype))))
