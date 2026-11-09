"""The normalized transformer's (nGPT, arXiv 2410.01131) arithmetic: hidden states on the unit
sphere, each sublayer's output joining the residual by a learned step along the sphere.

NVIDIA's Dory (vllm-project/vllm#60149, vllm/model_executor/models/dory.py at 55076950) is the
reference these follow, rounding for rounding: a unit norm reduces in fp32 and returns its input's
dtype (`justnorm`), and the residual's step norms both ends in that dtype before it takes the step
in fp32 (`slerp_norm`). Each fp32 is at least fp32 (`at_least_fp32`): a float64 model computes in
float64 throughout.
"""

import dataclasses
import functools
import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from dew.nn.derivatives import Algebra, contracted
from dew.nn.kernels.generation import linear_tangent, unbroadcast
from dew.nn.precision import L2NORM_EPS, at_least_fp32

__all__ = ["NGPT"]


@dataclasses.dataclass(frozen=True)
class NGPT:
    """A normalized transformer's constants, as a checkpoint states them: `std` the initializer's
    standard deviation (`initializer_range`), `alpha` the residual's initial step
    (`ngpt_alpha_init`) and `gate_logit` the attention output gate's initial logit scale
    (`attention_output_gate_logit_scale_init`, None for the square root of the model's width).

    nGPT stores each learned gain at its initializer's scale and multiplies it back by a constant
    these give: the step's `alpha / std`, the query's `sqrt(head_dim) / std`, the gate's
    `sqrt(gate_logit) / std`, the feed-forward's `sqrt(width)` and the logits' `1 / std`. Each
    starts at that scale (`start`): `std`, or 1 for the feed-forward's, so the step starts at
    `alpha`, the query's squared gain at `head_dim` (nGPT's `sqrt(head_dim)` logit scale over the
    usual `1 / sqrt(head_dim)`), the gate's at `gate_logit`, the feed-forward's at `sqrt(width)` and
    the logits' at 1, as nGPT's `init_scaling` starts them; a zero `alpha` or `gate_logit` is a gain
    that starts and stays at zero.

    A model under it keeps every hidden state on the unit sphere: no norm has a weight, the
    keys and the attention's and feed-forward's outputs are unit-normed
    (`dew.nn.precision.l2norm`), each sublayer joins the residual by `stepped`, and no norm
    precedes the head.
    """

    std: float
    alpha: float
    gate_logit: float | None = None

    @property
    def start(self) -> float:
        """The stored value every gain but the feed-forward's starts at."""
        return self.std

    @property
    def step(self) -> float:
        return self.alpha / self.std

    def query(self, head_dim: int) -> float:
        return math.sqrt(head_dim) / self.std

    def gate(self, width: int) -> float:
        return math.sqrt(math.sqrt(width) if self.gate_logit is None else self.gate_logit) / self.std

    @staticmethod
    def hidden(width: int) -> float:
        return math.sqrt(width)

    @property
    def logits(self) -> float:
        return 1 / self.std


def stepped(residual: jax.Array, branch: jax.Array, alpha: jax.Array, scale: float) -> jax.Array:
    """The residual after a sublayer's `branch` joins it: both ends unit-normed in their own dtypes,
    a step of `|alpha * scale|` from the residual toward the branch in at least fp32 of the residual,
    the branch and the rate, and the result unit-normed back to the residual's dtype (Dory's
    `slerp_alpha`).

    Its reverse reads the rows twice, from the sums of squares its forward keeps a row
    (`_stepped_backward`): once for every inner product, which XLA reduces in one pass, and once for
    the cotangents, where JAX's own transpose takes each unit norm apart, a reduction and a scaling
    each (ledger 1308: 4 ms of Dory 4B's step on an H100); forward mode differentiates the formula
    (`linear_tangent`)."""
    return contracted("test_ngpts_join_and_normed_hidden_hold_every_order")(
        linear_tangent(functools.partial(_stepped_forward, scale),
                       functools.partial(_stepped_backward, scale),
                       functools.partial(_stepped, scale), kind=Algebra()))(residual, branch, alpha)


def _squares(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`x` in at least fp32 and its rows' sums of squares, a unit norm's reduction
    (`dew.nn.precision.l2norm`)."""
    wide = x.astype(at_least_fp32(x.dtype))
    return wide, jnp.sum(jnp.square(wide), axis=-1, keepdims=True)


def _unit(wide: jax.Array, squares: jax.Array, dtype) -> jax.Array:
    """`l2norm` of the rows `wide` from their sums of squares, rounded to `dtype`."""
    return (wide * jax.lax.rsqrt(squares + L2NORM_EPS)).astype(dtype)


def _stepped(scale: float, residual, branch, alpha) -> jax.Array:
    return _stepped_forward(scale, residual, branch, alpha)[0]


def _stepped_forward(scale: float, residual, branch, alpha):
    """The joined rows, and the sums of squares of the residual's, the branch's and the step's rows."""
    wide = at_least_fp32(jnp.result_type(residual, branch, alpha))
    residual_wide, residual_squares = _squares(residual)
    branch_wide, branch_squares = _squares(branch)
    start = _unit(residual_wide, residual_squares, residual.dtype).astype(wide)
    end = _unit(branch_wide, branch_squares, branch.dtype).astype(wide)
    rate = jnp.abs(alpha.astype(wide) * scale)
    joined, joined_squares = _squares(start + rate * (end - start))
    return _unit(joined, joined_squares, residual.dtype), (residual_squares, branch_squares, joined_squares)


def _stepped_backward(scale: float, saved, operands, cotangent) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The residual's, the branch's and the rate's cotangents of the join (`stepped`), through each unit
    norm's transpose, `r (dy - u <u, dy>)` for a unit row `u = x r`: every inner product they read
    taken of the rows and the cotangent themselves, each end's through the step written out, so no
    reduction waits on another's."""
    residual, branch, alpha = operands
    wide = at_least_fp32(jnp.result_type(residual, branch, alpha))
    inverse = [jax.lax.rsqrt(squares + L2NORM_EPS) for squares in saved]
    rows, columns = residual.astype(wide), branch.astype(wide)
    # The ends as the forward rounded them, which the step reads; their norms' own transposes read the
    # unrounded rows.
    start = (rows * inverse[0]).astype(residual.dtype).astype(wide)
    end = (columns * inverse[1]).astype(branch.dtype).astype(wide)
    rate = jnp.abs(alpha.astype(wide) * scale)
    step = start + rate * (end - start)
    dy = cotangent.astype(wide)
    # Every sum of the rows themselves, each scaled by its rows' inverse norms after it, so none waits on
    # another and XLA takes them in one pass.
    along, kept_dy, kept_step, moved_dy, moved_step = (
        jnp.sum(left * right, axis=-1, keepdims=True)
        for left, right in ((step, dy), ((1 - rate) * rows, dy), ((1 - rate) * rows, step),
                            (rate * columns, dy), (rate * columns, step)))
    along = along * inverse[2]
    d_joined = inverse[2] * (dy - step * inverse[2] * along)
    d_residual = inverse[0] * ((1 - rate) * d_joined - rows * jnp.square(inverse[0]) * inverse[2]
                               * (kept_dy - along * inverse[2] * kept_step))
    d_branch = inverse[1] * (rate * d_joined - columns * jnp.square(inverse[1]) * inverse[2]
                             * (moved_dy - along * inverse[2] * moved_step))
    # The rate's and each end's cotangents through `|alpha * scale|` and the broadcasts the step took, as JAX
    # pulls them back: its derivative at zero and its sums over each broadcast axis.
    _, rated = jax.vjp(lambda alpha: jnp.broadcast_to(jnp.abs(alpha.astype(wide) * scale), step.shape), alpha)
    (d_alpha,) = rated(d_joined * (end - start))
    return (unbroadcast(d_residual, residual.shape).astype(residual.dtype),
            unbroadcast(d_branch, branch.shape).astype(branch.dtype), d_alpha.astype(alpha.dtype))


def normed_hidden(gate: jax.Array, up: jax.Array, gate_gain: jax.Array, up_gain: jax.Array, factor: float,
                  product: Callable[[jax.Array, jax.Array], jax.Array]) -> jax.Array:
    """nGPT's hidden activations: the gate and up projections times their gains (`gain` at `factor`) in
    at least fp32, back in their dtype, their gated `product` in at least fp32, unit-normed in the gate's
    dtype (dory.py:188-194). Its reverse reads the rows twice, from the sum of squares its forward keeps
    a row (`_hidden_backward`): once for the norm's inner product, once for the cotangents and the
    gains'; forward mode differentiates the formula (`linear_tangent`)."""
    return contracted("test_ngpts_join_and_normed_hidden_hold_every_order")(
        linear_tangent(functools.partial(_hidden_forward, factor, product),
                       functools.partial(_hidden_backward, factor, product),
                       functools.partial(_hidden, factor, product),
                       kind=Algebra()))(gate, up, gate_gain, up_gain)


def _gated(factor: float, gate, up, gate_gain, up_gain):
    """The gained gate and up rows as the forward rounds them, in at least fp32, and the gains."""
    wide = at_least_fp32(jnp.result_type(gate, up))
    gains = gain(gate_gain, factor, wide), gain(up_gain, factor, wide)
    gated = (gate.astype(wide) * gains[0]).astype(gate.dtype).astype(wide)
    upped = (up.astype(wide) * gains[1]).astype(up.dtype).astype(wide)
    return gated, upped, gains


def _hidden(factor: float, product, gate, up, gate_gain, up_gain) -> jax.Array:
    return _hidden_forward(factor, product, gate, up, gate_gain, up_gain)[0]


def _hidden_forward(factor: float, product, gate, up, gate_gain, up_gain):
    """The normed hidden rows and their sums of squares."""
    gated, upped, _ = _gated(factor, gate, up, gate_gain, up_gain)
    hidden, squares = _squares(product(gated, upped).astype(gate.dtype))
    return _unit(hidden, squares, gate.dtype), (squares,)


def _hidden_backward(factor: float, product, saved, operands, cotangent):
    """The gate's, up's and both gains' cotangents of the normed hidden (`normed_hidden`), through the
    norm's transpose (`_stepped_backward`'s) and the product's own pullback at the gained rows."""
    gate, up, gate_gain, up_gain = operands
    (squares,) = saved
    gated, upped, gains = _gated(factor, gate, up, gate_gain, up_gain)
    produced, pulled = jax.vjp(product, gated, upped)
    inverse = jax.lax.rsqrt(squares + L2NORM_EPS)
    hidden = produced.astype(gate.dtype).astype(gated.dtype)
    dy = cotangent.astype(gated.dtype)
    # The norm's inner product of the rows themselves, scaled after it, so it waits on no other sum.
    along = jnp.sum(hidden * dy, axis=-1, keepdims=True) * inverse
    d_gated, d_upped = pulled((inverse * (dy - hidden * inverse * along)).astype(produced.dtype))
    # The gained rows' cotangents in their own dtypes, as the forward rounded them.
    d_gated = d_gated.astype(gate.dtype).astype(gated.dtype)
    d_upped = d_upped.astype(up.dtype).astype(gated.dtype)
    leading = tuple(range(gate.ndim - 1))
    return ((d_gated * gains[0]).astype(gate.dtype), (d_upped * gains[1]).astype(up.dtype),
            (jnp.sum(d_gated * gate.astype(gated.dtype), axis=leading) * factor).astype(gate_gain.dtype),
            (jnp.sum(d_upped * up.astype(gated.dtype), axis=leading) * factor).astype(up_gain.dtype))


def gain(scale: jax.Array, factor: float, dtype: DTypeLike | None) -> jax.Array:
    """A learned per-feature gain stored at the initializer's scale, as the multiplier it stands for in
    at least fp32 of its own dtype and the computation's `dtype`: `scale * factor` (nGPT's s_qk, s_u,
    s_v and s_z, and Dory's output-gate scale), so a float64 run over a narrower stored gain takes its
    product in float64."""
    wide = scale.dtype if dtype is None else jnp.promote_types(scale.dtype, dtype)
    return scale.astype(at_least_fp32(wide)) * factor
