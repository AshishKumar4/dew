"""Routed value experts: an attention layer whose values come from a mixture of projections.

K2 Horizon's MoVA (IFM/K2-Horizon-MoVA-36B-A4B, modeling_k2_horizon.py at cca48b66,
`K2HorizonMoVAAttention`, :351-512) replaces the value projection of its sparse layers with
`experts` projections of the hidden state, `top_k` of them per token chosen by a router: each chosen
expert's projection passes a SiLU and the results are summed under the router's weights, the
selection biased by the router's bias and the weights the unbiased scores', normalized and scaled
(`calc_router_weights` and `combine_routed_experts`, :136-200). The router is `dew.nn.moe.Router`, the
experts one stacked kernel run as grouped matmuls (`expert_dispatch`), as a routed MLP's are.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name

from .moe import ExpertLinear, Router, expert_compute_dtype, expert_dispatch, expert_projection
from .precision import at_least_fp32, float32_einsum

if TYPE_CHECKING:
    from .backbones.decoder_block import Mixture


@dataclasses.dataclass(frozen=True)
class ValueExperts:
    """A layer's routed value experts: `experts` projections, `top_k` per token, chosen and weighted by a
    router of `score_function` whose weights are normalized over the chosen (`norm_topk_prob`) and
    scaled by `scaling`, a learned `bias` entering the selection alone (`Router.expert_bias`)."""

    experts: int
    top_k: int
    score_function: str = 'sigmoid'
    norm_topk_prob: bool = True
    scaling: float = 1.0
    bias: bool = False

    def __post_init__(self):
        if not 0 < self.top_k <= self.experts:
            raise ValueError(f"value experts choose top_k of their experts, got {self.top_k} of "
                             f"{self.experts}")


VALUE_EXPERT_AXES = ('exp', 'embed', 'kv')
"""The stacked value kernels' axes: an expert, the hidden width, and the key/value heads' width."""


class RoutedValues(nn.Module):
    """`spec`'s value experts over hidden states `[..., in_features]`: each token's `top_k` chosen
    projections to `features`, each through a SiLU, summed under the router's weights in fp32, run as
    the layer's routed MLP runs its own (`mixture`, `dew.nn.backbones.decoder_block.Mixture`): its
    kernel, its dispatch and its capacity, so an expert-parallel model places both sets of experts
    alike."""

    spec: ValueExperts
    in_features: int
    features: int
    mixture: Mixture
    init_std: float | None = None
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        spec = self.spec
        self.router = Router(num_experts=spec.experts, in_features=self.in_features, top_k=spec.top_k,
                             score_function=spec.score_function, normalize_weights=spec.norm_topk_prob,
                             routed_scaling_factor=spec.scaling, expert_bias=spec.bias,
                             init_std=self.init_std, precision=self.precision, name='v_router')
        self.projection = ExpertLinear(num_experts=spec.experts, in_features=self.in_features,
                                       features=self.features, init_std=self.init_std, name='v_proj')

    def _project(self, tokens, sizes, _expert_ids, kernels, *, dtype):
        projected = expert_projection(tokens, kernels[0], sizes, dtype, self.mixture.implementation,
                                      self.precision)
        return jax.nn.silu(checkpoint_name(jnp.asarray(projected), 'v_proj'))

    def __call__(self, x: jax.Array) -> jax.Array:
        weights, indices = self.router(x)
        kernel = self.projection.kernel
        compute = expert_compute_dtype(x, kernel, dtype=self.dtype)
        slots = expert_dispatch(functools.partial(self._project, dtype=compute), x, indices, (kernel,),
                                (VALUE_EXPERT_AXES,), num_experts=self.spec.experts,
                                dispatch=self.mixture.dispatch, capacity_factor=self.mixture.capacity_factor,
                                initializing=self.is_initializing(), output_dtype=compute)
        wide = at_least_fp32(slots.dtype)
        return float32_einsum('...ke,...k->...e', slots.astype(wide), weights.astype(wide),
                              precision=self.precision).astype(slots.dtype)
