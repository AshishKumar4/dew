"""The normalized transformer's (nGPT, arXiv 2410.01131) arithmetic: hidden states on the unit
sphere, each sublayer's output joining the residual by a learned step along the sphere.

NVIDIA's Dory (vllm-project/vllm#60149, vllm/model_executor/models/dory.py at 55076950) is the
reference these follow, rounding for rounding: a unit norm reduces in fp32 and returns its input's
dtype (`justnorm`), and the residual's step norms both ends in that dtype before it takes the step
in fp32 (`slerp_norm`).
"""

import dataclasses

import jax
import jax.numpy as jnp

EPSILON = 1e-6
"""The floor under a unit norm's squared length (dory.py's EPS)."""


@dataclasses.dataclass(frozen=True)
class NGPT:
    """A normalized transformer's gains. nGPT stores each learned gain at the initializer's scale
    and multiplies it back by a constant (`gain`): these are those constants, Dory's in its terms
    (`initializer_range` std, `ngpt_alpha_init` a0, hidden width H, head width d).

    `step` scales a sublayer's residual step, `|alpha * step|` (a0 / std), and `step_init` is the
    step it starts at (a0). `query` scales the query's per-feature gain, squared over the unit
    query after its rotation (sqrt(d) / std). `gate` scales the attention output gate's, squared
    over the gate's logits (sqrt(g0) / std, g0 = sqrt(H) unless a checkpoint names it). `hidden`
    scales the feed-forward's gate and up gains (sqrt(H)), and `logits` the per-token logit gain
    (1 / std).

    A model under it keeps every hidden state on the unit sphere: no norm has a weight, the
    keys and the attention's and feed-forward's outputs are unit-normed (`unit`), each sublayer
    joins the residual by `stepped`, and no norm precedes the head.
    """

    step: float
    step_init: float
    query: float
    gate: float
    hidden: float
    logits: float


def unit(x: jax.Array) -> jax.Array:
    """`x` over its L2 length along the last axis, reduced in fp32 and returned in `x`'s dtype:
    `x * rsqrt(sum(x**2) + 1e-6)`, Dory's `justnorm`."""
    wide = x.astype(jnp.float32)
    return (wide * jax.lax.rsqrt(jnp.sum(jnp.square(wide), axis=-1, keepdims=True) + EPSILON)).astype(x.dtype)


def stepped(residual: jax.Array, branch: jax.Array, alpha: jax.Array, scale: float) -> jax.Array:
    """The residual after a sublayer's `branch` joins it: both ends unit-normed in the residual's
    dtype, a step of `|alpha * scale|` from the residual toward the branch in fp32, and the result
    unit-normed back to the residual's dtype (Dory's `slerp_alpha`)."""
    start = unit(residual).astype(jnp.float32)
    end = unit(branch).astype(jnp.float32)
    rate = jnp.abs(alpha.astype(jnp.float32) * scale)
    return unit(start + rate * (end - start)).astype(residual.dtype)



def gain(scale: jax.Array, factor: float) -> jax.Array:
    """A learned per-feature gain stored at the initializer's scale, as the fp32 multiplier it
    stands for: `scale * factor` (nGPT's s_qk, s_u, s_v and s_z, and Dory's output-gate scale)."""
    return scale.astype(jnp.float32) * factor
