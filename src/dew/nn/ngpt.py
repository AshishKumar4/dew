"""The normalized transformer's (nGPT, arXiv 2410.01131) arithmetic: hidden states on the unit
sphere, each sublayer's output joining the residual by a learned step along the sphere.

NVIDIA's Dory (vllm-project/vllm#60149, vllm/model_executor/models/dory.py at 55076950) is the
reference these follow, rounding for rounding: a unit norm reduces in fp32 and returns its input's
dtype (`justnorm`), and the residual's step norms both ends in that dtype before it takes the step
in fp32 (`slerp_norm`). Each fp32 is at least fp32 (`at_least_fp32`): a float64 model computes in
float64 throughout.
"""

import dataclasses
import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from dew.nn.precision import at_least_fp32, l2_normed

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
    `slerp_alpha`). Every derivative is JAX's of the formula, each unit norm's the norm's own (`l2_normed`,
    range-safe where JAX's of `rsqrt` flushes), its reverse's reductions the algebra pass's to arrange (row
    11940: no rule of its own written to restructure them)."""
    wide = at_least_fp32(jnp.result_type(residual, branch, alpha))
    start = l2_normed(residual, residual.dtype).astype(wide)
    end = l2_normed(branch, branch.dtype).astype(wide)
    rate = jnp.abs(alpha.astype(wide) * scale)
    return l2_normed(start + rate * (end - start), residual.dtype)


def normed_hidden(gate: jax.Array, up: jax.Array, gate_gain: jax.Array, up_gain: jax.Array, factor: float,
                  product: Callable[[jax.Array, jax.Array], jax.Array]) -> jax.Array:
    """nGPT's hidden activations: the gate and up projections times their gains (`gain` at `factor`) in
    at least fp32, back in their dtype, their gated `product` in at least fp32, unit-normed in the gate's
    dtype (dory.py:188-194). Every derivative is JAX's of the formula, its norm's the norm's own (row 11940,
    as `stepped`'s)."""
    wide = at_least_fp32(jnp.result_type(gate, up))
    gated = (gate.astype(wide) * gain(gate_gain, factor, wide)).astype(gate.dtype).astype(wide)
    upped = (up.astype(wide) * gain(up_gain, factor, wide)).astype(up.dtype).astype(wide)
    return l2_normed(product(gated, upped).astype(gate.dtype), gate.dtype)


def gain(scale: jax.Array, factor: float, dtype: DTypeLike | None) -> jax.Array:
    """A learned per-feature gain stored at the initializer's scale, as the multiplier it stands for in
    at least fp32 of its own dtype and the computation's `dtype`: `scale * factor` (nGPT's s_qk, s_u,
    s_v and s_z, and Dory's output-gate scale), so a float64 run over a narrower stored gain takes its
    product in float64."""
    wide = scale.dtype if dtype is None else jnp.promote_types(scale.dtype, dtype)
    return scale.astype(at_least_fp32(wide)) * factor
