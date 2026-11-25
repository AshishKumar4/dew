"""Block attention residuals: Kimi K3's residual as a softmax over depth.

Kimi K3 (arXiv 2607.24653) cuts the layers into blocks of `block_size`, keeps
each block's output sum, and has every sublayer read a softmax mixture of the
finished blocks and the current partial sum, weighted by one learned
pseudo-query per site (`KimiDecoderLayer._forward_attn_residual`,
`_apply_attn_res`, modeling_kimi_linear.py:973-1088 of moonshotai/Kimi-K3 at
f831ab6):

    v      = [block_0, ..., block_{n-1}, partial]          # [B, S, n + 1, D]
    k      = v / sqrt(mean(v^2) + eps)                     # fp32, per source
    scores = k @ (norm.weight * proj.weight)               # [B, S, n + 1]
    h      = softmax(scores) @ v                           # fp32, then v's dtype

Layer `i` opens a block when `i % block_size == 0`: after reading its
attention input, the partial it received becomes a finished block and the
partial restarts from the attention output. Layer 0 reads the embeddings as
block 0, and after the last layer a model-level site mixes all blocks with
the final partial before the final norm (`_apply_output_attn_res`,
:1226-1233). The state is one `[B, S, blocks + 1, D]` array, unfilled slots
zero and the partial last, so each layer's slice is static through a scan or
pipeline. A site's `scale` `[D]` is `*_res_norm.weight` and `kernel` `[D, 1]`
`*_res_proj.weight`; a zero kernel averages.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from flax import linen as nn

from .dense import watched_or_owned
from .derivatives import portable_formed
from .flops import projection_flops
from .norms import unweighted_rmsnorm
from .precision import at_least_fp32, computed_einsum


@dataclasses.dataclass(frozen=True)
class AttentionResiduals:
    """The block size the depth softmax groups layers by, `attn_res_block_size`, and whether each depth
    mixture sums its sources as its keys read them, each at unit RMS (`normalize_values`, Inkling 2
    Mini's `attn_res_normalize_values`; Kimi K3's mix them as they are), and whether its scores are each
    query's dot product over the root of the width (`scaled`, Inkling 2's; Kimi K3's are the products)."""

    block_size: int
    normalize_values: bool = False
    scaled: bool = False

    def blocks(self, num_layers: int) -> int:
        """The finished blocks after `num_layers` layers: every layer opening one counts."""
        return -(-num_layers // self.block_size)

    def site(self, index: int) -> ResidualSite:
        """Layer `index`'s place in the blocks."""
        return ResidualSite(finished=-(-index // self.block_size), opens=index % self.block_size == 0,
                            normalize_values=self.normalize_values, scaled=self.scaled)


NORMALIZED_UNWRITTEN = ("the source's format mixes its attention residuals' values as they are, and this "
                        "decoder normalizes them (AttentionResiduals.normalize_values); a run's checkpoint "
                        "or release keeps it")
"""Why a decoder normalizing its residuals' values is written in no family's layout (`Pretrained.export`)."""


@dataclasses.dataclass(frozen=True)
class ResidualSite:
    """One layer's static place in the depth mixture.

    `finished` counts the blocks done before the layer runs, which its
    attention input mixes with the partial sum; `opens` makes the layer
    close the partial it received into block `finished` and restart it;
    `normalize_values` and `scaled` are its mixtures' (`AttentionResiduals`).
    """

    finished: int
    opens: bool
    normalize_values: bool = False
    scaled: bool = False


def sources(state, finished: int, partial):
    """The `finished` blocks of `state` with `partial` after them: what one site mixes."""
    return jnp.concatenate([state[:, :, :finished], partial[:, :, None, :]], axis=2)


class DepthAttention(nn.Module):
    """Mix `[B, S, n, D]` sources into `[B, S, D]` by one learned pseudo-query.

    Both contractions run in the sources' compute dtype under the run's
    precision (`depth_read`), their softmax and the keys' statistics in fp32.
    With `normalize_values` the mixture sums the sources as their keys read
    them, each at unit RMS, rather than as they are (Inkling 2 Mini's
    `attn_res_normalize_values`). Its parameters are declared in `setup`, so
    a site's tree holds them whether or not a call reads it.
    """

    emb_features: int
    norm_eps: float = 1e-5
    normalize_values: bool = False
    scaled: bool = False

    def setup(self):
        self.scale = self.param('scale', nn.initializers.ones, (self.emb_features,), jnp.float32)
        self.kernel = self.param('kernel', nn.initializers.zeros, (self.emb_features, 1), jnp.float32)

    @nn.nowrap
    def depth_flops(self, lengths, num_sources: int) -> float:
        """The pseudo-query score and weighted-value contraction for each source."""
        return 2 * self.depth_parameter_flops(lengths, num_sources)

    @nn.nowrap
    def depth_parameter_flops(self, lengths, num_sources: int) -> float:
        """Only the score contracts a learned matrix; value mixing uses its weights."""
        return projection_flops(lengths, self.emb_features, num_sources)

    def __call__(self, values):
        # The values' cotangent leaves only with the query's scale and kernel gradients formed, the
        # mixture recomputed from its operands in the reverse mode (`portable_formed`): XLA otherwise puts
        # those reductions over every token off to the step's end, holding the fp32 keys and values live
        # across layers and an exit-gated loop's unrolled passes (row 11713). A call a quantization may own
        # or a caller watches runs as it is (`watched_or_owned`).
        if watched_or_owned(self):
            return depth_attended(values, self.scale, self.kernel, norm_eps=self.norm_eps,
                                  normalize_values=self.normalize_values, scaled=self.scaled)

        def attended(values, parameters):
            return depth_attended(values, *parameters, norm_eps=self.norm_eps,
                                  normalize_values=self.normalize_values, scaled=self.scaled)
        return portable_formed(attended)(values, (self.scale, self.kernel))


def depth_attended(values: jax.Array, scale: jax.Array, kernel: jax.Array, *, norm_eps: float,
                   normalize_values: bool, scaled: bool = False) -> jax.Array:
    """`DepthAttention`'s mixture of `values` `[B, S, n, D]` by its query's `scale` `[D]` and `kernel`
    `[D, 1]`, in `values`' dtype (`depth_read`'s of one query)."""
    query = scale.astype(at_least_fp32(values.dtype)) * kernel[:, 0].astype(at_least_fp32(values.dtype))
    return depth_read(values, query[:, None], norm_eps=norm_eps, normalize_values=normalize_values,
                      scaled=scaled)[:, :, 0]


def depth_read(values: jax.Array, queries: jax.Array, *, norm_eps: float,
               normalize_values: bool, scaled: bool = False) -> jax.Array:
    """`values` `[B, S, n, D]` mixed over their `n` sources by each of `queries` `[D, q]` (each a
    `DepthAttention`'s pseudo-query, `scale * kernel`), `[B, S, q, D]` in `values`' dtype: the sources read
    and normalized once for every query (keys at unit RMS, their statistics in fp32), each query's softmax
    over the sources in fp32, and both contractions in `values`' compute dtype under the run's precision
    (`dew.nn.precision.computed_einsum`: a bf16 run's Tensor Cores, `ACCURATE`'s exact float32; row 12918).
    With `normalize_values` the mixture sums the sources as their keys read them; with `scaled` each score
    is over the root of the width."""
    keys = unweighted_rmsnorm(values, norm_eps, implementation='xla')
    mixed = depth_mixed(keys, queries, keys if normalize_values else values, scaled=scaled)
    return mixed.astype(values.dtype)


def depth_mixed(keys: jax.Array, queries: jax.Array, values: jax.Array, *, scaled: bool = False) -> jax.Array:
    """`values` `[B, S, n, D]` summed over their `n` sources by the softmax over the sources of `keys`
    (alike) against each of `queries` `[D, q]`, `[B, S, q, D]` in at least fp32: the scores and the sum
    `computed_einsum`'s, in the sources' compute dtype under the run's precision, each score over the root
    of the width `D` in at least fp32 where `scaled`, before its softmax."""
    compute = values.dtype
    scores = computed_einsum('bsnd,dq->bsnq', keys, queries, dtype=compute)
    if scaled:
        wide = at_least_fp32(scores.dtype)
        scores = scores.astype(wide) * jnp.asarray(keys.shape[-1] ** -0.5, wide)
    weights = jax.nn.softmax(scores, axis=2)
    return computed_einsum('bsnq,bsnd->bsqd', weights, values, dtype=compute)
