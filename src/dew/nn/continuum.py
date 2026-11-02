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

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct
from flax.typing import Dtype, PrecisionLike

from dew.nn.activations import _SLOPES, activation
from dew.nn.dense import Dense
from dew.nn.mlp import GatedMLP
from dew.nn.norms import NormBounds
from dew.nn.precision import at_least_fp32
from dew.quantization import InputQuantizer, LinearProducts, intercepted_nonlinear, rounding_growth

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


class Continuum(LinearProducts):
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
                            precision=self.precision,
                            named_axes=(("down", ("embed", "mlp")), ("up", ("mlp", "embed"))))
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

    def step_terms(self, x: jax.Array, parts: tuple[str, ...], quantizer: Quantizers
                   ) -> tuple[jax.Array, jax.Array, jax.Array, float, float, float]:
        """What bounds a damped step of the level's `parts` (`down`, `up`) at states `x` `[..., features]`
        (`dew.objectives.lm.adaptation._damping`): each state's |x|^2 and |f|^2, x and f = act(down(x)) as
        `down` and `up` multiply them (`quantizer`, a model's checkpoint quantizer of a map's input, None
        for none), the Frobenius norm of the kernel `up` holds (`applied_kernel`, an adapter's update on it
        included), at least its spectral norm, in at least fp32, and how much its product's rounding can grow
        any kernel's norm (1 unrounded); and the activation's bounds p1, p2 on |act'| and |act''|
        (`_SLOPES`).

        Where Qwix's quantization rounds a map's operands (`rounding_growth`), each norm here is the most its
        rounding grows it to, and a step follows Qwix's straight-through gradient: the bound holds for the
        straight-through surrogate's smooth part, and a step that crosses a rounding cell adds the
        quantization's own jump. A step of `down` through an `up` a method interceptor computes other than
        as a linear map (`intercepted_nonlinear`: an adapter's dropout, a frozen FP8 product, a
        checkpoint's quantizer of its input, apart from the quantizer `quantizer` reads) is refused, as is
        one through an activation whose act'' no number bounds (ReLU's, a point mass at zero). A quantizer
        of `down`'s input rounds the states, which no step moves."""
        p1, p2 = _SLOPES[activation(self.level.activation)]
        if "down" in parts and p2 is None:
            raise ValueError("a damped step of the level's `down` bounds its loss's curvature through its "
                             "activation's second derivative, which a ReLU has none of (a point mass at "
                             "zero), so no rate bounds it there: select=('up',), or a smooth activation")
        # `up`'s checkpoint quantizer is read exactly here (`read`); any other interceptor's rounding not.
        if "down" in parts and intercepted_nonlinear(self, "up", input_scales=False):
            raise ValueError("a damped step of the level's `down` bounds the loss's curvature through the "
                             "map `up` applies to f = act(D x), and this call computes `up` other than as a "
                             "linear map of f (an adapter's dropout or a frozen FP8 product), which no "
                             "curvature bounds: select=('up',), or step it undamped")
        # Each map's operands as Qwix's rounding can grow them: its input's, then its kernel's.
        operands = self.dtype or x.dtype
        down_input, _ = rounding_growth(self, "down", self.features, operands)
        up_input, up_kernel = rounding_growth(self, "up", self.level.hidden, operands)

        def read(value: jax.Array, layer: nn.Module) -> jax.Array:
            rounding = quantizer(layer)
            return (value if rounding is None else rounding.qdq(value)).astype(at_least_fp32(x.dtype))

        fed = up_input * read(activation(self.level.activation)(self.down(x)), self.up)
        kernel = self.applied_kernel("up").astype(at_least_fp32(x.dtype))
        return (jnp.sum(jnp.square(down_input * read(x, self.down)), axis=-1),
                jnp.sum(jnp.square(fed), axis=-1), jnp.sqrt(jnp.sum(jnp.square(kernel))), up_kernel, p1,
                p2 or 0.0)


@struct.dataclass
class HeadTransform:
    """What bounds a head transform N(phi(D s)) before a head's rows (`HeadTerms`): phi's `slopes` (p1, p2,
    `_SLOPES`), the Frobenius norm of D (`dense`), at least its spectral, and N's derivative bounds
    (`NormBounds`)."""

    slopes: tuple[float, float | None] = struct.field(pytree_node=False)
    dense: jax.Array
    norm: NormBounds


@struct.dataclass
class HeadTerms:
    """What bounds the curvature and the slope of a slot's cross entropy in the states a head reads
    (`CausalTransformer.head_bounds`, `dew.objectives.lm.adaptation`), constants of the step, not
    differentiated. With |A| the Frobenius norm of the head's rows, at least the spectral, the logits'
    `curvature` and `slope` are |A|^2 / 2 and sqrt 2 |A| (the cross entropy's Hessian in the logits is at
    most 1/2, its gradient p - y at most sqrt 2), a logit softcap c tanh(u / c), of slope at most 1,
    adding |A|^2 4 / (3 sqrt 3 c) of curvature, of either sign. A `transform` (`HeadTransform`) adds its
    own, read at each slot's `spread` of its norm's input at the level's start, `[B, S]`."""

    curvature: jax.Array
    slope: jax.Array
    spread: jax.Array
    transform: HeadTransform | None = None

    @classmethod
    def of(cls, table: jax.Array, softcap: float | None, spread: jax.Array,
           transform: HeadTransform | None = None) -> HeadTerms:
        """The terms of a head of squared Frobenius norm `table` and logit `softcap` (None for none)."""
        curvature = table * (0.5 if softcap is None else 0.5 + 4 / (3 * math.sqrt(3) * softcap))
        return cls(*jax.lax.stop_gradient((curvature, jnp.sqrt(2 * table), spread)), transform)

    def at(self, floor: jax.Array, moved: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Bounds on the curvature and the slope over every state within `moved` (2-norm) of the states of
        a window whose slots' least spread is `floor`, per row. Through a transform N(phi(D s)), its
        norm's input moves by at most p1 |D| `moved`, where the norm's Jacobian and second derivative are
        at most n1 and n2 (`NormBounds.at`), so the transform's Jacobian is at most J = n1 p1 |D| and its
        second derivative at most (n2 p1^2 + n1 p2) |D|^2: the curvature J^2 times the logits' plus their
        slope times that, the slope J times theirs."""
        if self.transform is None:
            return jnp.broadcast_to(self.curvature, jnp.shape(floor)), jnp.broadcast_to(self.slope,
                                                                                          jnp.shape(floor))
        (p1, p2), dense = self.transform.slopes, self.transform.dense
        first, second = self.transform.norm.at(floor, p1 * dense * moved)
        jacobian = first * p1 * dense
        bent = (second * p1 ** 2 + first * (p2 or 0.0)) * jnp.square(dense)
        return jnp.square(jacobian) * self.curvature + self.slope * bent, jacobian * self.slope


def _head_growth(model: LinearProducts, maps: dict[str | None, int], dtype) -> dict[str | None, float]:
    """How much Qwix's rounding can grow each head map's kernel's norm (`rounding_growth`), `maps` each map's
    contraction by its submodule's name (None for a tied table's product in the model's own scope), its
    operands of the model's compute `dtype`. A map
    a method interceptor computes other than as a linear map of its input (`intercepted_nonlinear`: a
    checkpoint's quantizer of its input, a frozen FP8 product, an adapter's dropout) makes the head
    piecewise constant in its states, whose gradient no Lipschitz bound holds: refused."""
    rounded = [name for name in maps if name is not None and intercepted_nonlinear(model, name)]
    if rounded:
        raise ValueError(f"a damped step bounds the head's curvature in its states, and the head's "
                         f"{rounded} round their inputs, making it piecewise constant there, which no "
                         f"curvature bounds: step it undamped")
    return {name: rounding_growth(model, name, contraction, dtype)[1] for name, contraction in maps.items()}


def _squared_norm(value: jax.Array) -> jax.Array:
    """The sum of `value`'s squares, in at least fp32: a matrix's squared Frobenius norm."""
    return jnp.sum(jnp.square(value.astype(at_least_fp32(value.dtype))))
