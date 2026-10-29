"""Multi-head latent attention: DeepSeek-V3 MLA and the V3.2 sparse indexer.

The reference is transformers 5.16.1 (deepseek_v3, deepseek_v32, and
glm_moe_dsa, the same block with the indexer rotating interleaved pairs and
IndexShare layers attending the previous full layer's selection). MLA
compresses keys and values into one low-rank latent per token plus a small
decoupled rotary head: `kv_a_proj_with_mqa` maps the hidden states to
`[kv_lora_rank + qk_rope_head_dim]`, the latent is normed, and `kv_b_proj`
expands it to every head's nope keys and values. Queries are low-rank the
same way when `q_lora_rank` is set. Decode caches the compressed latents as
V3 does, and the sparse variant the expanded keys and values as V3.2 does.
The rotary head rotates interleaved pairs, at the plain or YaRN-scaled
frequencies (`dew.nn.rope`).
"""

import dataclasses
import functools
import math
from collections.abc import Callable, Mapping
from typing import ClassVar

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name

from dew.mesh import LogicalAxes
from dew.nn.attention import (
    LayerNorm,
    RMSNorm,
    causal_attention_mask,
    chunk_mask,
    kernel_for_materialized_mask,
    max_attention_logits,
    scaled_dot_product_attention,
)
from dew.nn.dense import Dense
from dew.nn.documents import document_mask
from dew.nn.flops import MissingFlopsCount, attention_flops, projection_flops
from dew.nn.inputs import (
    PLAIN_CALL,
    MixerCall,
    PredictionPhase,
    coordinates_of,
    counted_places,
    document_reach,
)
from dew.nn.kv_cache import KVCache, cached_validity, open_kv_cache, write_cache

# dsa_kpool reads INDEXER and the expanded cache below. Reading the registry
# contracts from base avoids importing the kind hub back during this module's
# initialization, so either kind can be the first imported.
from dew.nn.mixer_base import ATTENTION_CONTEXT_RESIDUALS, MixerBase, MixerContext
from dew.nn.mixers.attention import call_visibility, dense_visibility, group_fields
from dew.nn.precision import (
    at_least_fp32,
    compensated_add,
    compensated_sum,
    float32_dot_general,
    float32_einsum,
)
from dew.nn.rope import (
    LongRopeScaling,
    Scaling,
    YarnScaling,
    deinterleaved,
    rotary_freqs,
    rotate,
    yarn_query_scale,
)
from dew.nn.sharding import RESIDUAL, constrain, down_projection, logical_axes
from dew.nn.sparse_selection import (
    index_agreements,
    index_scores,
    selection_mask,
    sparse_latent_attention,
    top_k_selection,
)
from dew.quantization import quantized_products


def open_latent_cache(module: nn.Module, key, value_dim: int, index_keys, max_seq_len: int, *, valid=None,
                      layout: KVCache):
    """Open a sparse latent attention's decode cache: its expanded keys and values, as DeepSeek V3.2
    stores them, through the attention cache (`open_kv_cache`) in `layout`, and the sparse indexer's
    keys beside them where `index_keys` is given (None on a layer that owns no indexer and reads
    another layer's selection), one compact row each, which a paged server keeps as a row's state.
    Returns the [B, S] slots, an `append` that writes the new rows and returns the whole cache, the
    index keys last or None, and the attention cache's own append, whose `query` puts a query in the
    basis the stored keys are in."""
    positions, written = open_kv_cache(module, key, max_seq_len, valid=valid, value_dim=value_dim,
                                       layout=layout)
    cached_index = None if index_keys is None else module.variable(
        "cache", "cached_index", jnp.zeros, (key.shape[0], max_seq_len, index_keys.shape[-1]),
        index_keys.dtype)

    def append(new_key, new_value, new_index_keys):
        full_key, full_value = written(new_key, new_value)
        if cached_index is not None and written.allocated:
            cached_index.value = write_cache(cached_index.value, new_index_keys, written.positions)
        return full_key, full_value, None if cached_index is None else cached_index.value

    return positions, append, written


INDEXER = 'indexer'
"""The indexer's name under its attention layer, which the parameter paths
of the indexer's weights carry and nothing else in the tree does."""

INDEXER_COLLECTION = 'indexer'
"""The collection an attention layer sows its per-query indexer KL under
when a caller opens it, as the LM objective does to train the indexer."""

SELECTIONS = 'selections'
"""The collection V3.2's lightning indexer sows its scores and its heads' agreements before their relu
under when a caller opens it (`SparseIndexer.scores`), as a selection's margin does to read how near a
tie each key sits; never at initialization, so no variable tree holds it."""

INDEXER_EPS = 1e-8
"""MaxText's `EPS` (src/maxtext/utils/globals.py at 538fe7a3), which its
`calculate_indexer_loss` adds to the target's normaliser and inside both
logarithms of the KL."""


def indexer_kl(scores, query, key, keep, scale: float):
    """Per query, KL of the indexer's softmax from the attention: `[B, S]`, at least fp32 (float64
    for float64 inputs, so a float64 truth computes it in float64).

    The indexer's objective (arXiv 2512.02556, eq. 3 and 4; MaxText
    `calculate_indexer_loss`): the target is the main attention's distribution
    summed over heads and L1-normalised, the indexer's the softmax of its
    `[B, S, T]` `scores`, both over the keys `keep` allows (the causal mask in
    the dense warm-up, the top-k selection in sparse training). `query` and
    `key` are the main heads, `scale` the kernel's logit scale. The target is
    detached as the reference detaches it, and heads are summed one at a time
    to keep the footprint at `[B, S, T]`. MaxText's `EPS` (1e-8) guards the
    L1 normalisation and both logarithms, as the reference adds it: a key the
    target gives no mass contributes zero, and the gradient is the
    reference's, which the epsilon moves beyond float32 rounding.
    """
    wide = at_least_fp32(jnp.result_type(scores, query, key))
    masked = jnp.finfo(wide).min
    batch, length, total = scores.shape
    keep = jnp.broadcast_to(keep, (batch, length, total))
    query = jax.lax.stop_gradient(query).astype(wide) * scale
    key = jax.lax.stop_gradient(key).astype(wide)

    def head(summed, projections):
        q, k = projections
        product = float32_einsum('bsd,btd->bst', q, k)
        probabilities = jax.nn.softmax(jnp.where(keep, product, masked), axis=-1)
        return compensated_add(summed, (probabilities, jnp.zeros_like(probabilities))), None

    # The heads, the target's normalization over the keys and each query's KL over them all sum
    # rounded once (`compensated_add`, `compensated_sum`): plain sums biased the KL upward, 1.67
    # times MaxText's error from float64, which sums its heads at once at H times this footprint.
    zeros = jnp.zeros((batch, length, total), wide)
    (high, low), _ = jax.lax.scan(head, (zeros, zeros), (jnp.moveaxis(query, 2, 0), jnp.moveaxis(key, 2, 0)))
    summed = high + low
    target = summed / (compensated_sum(summed, (summed.ndim - 1,))[..., None] + INDEXER_EPS)
    indexer = jax.nn.softmax(jnp.where(keep, scores.astype(wide), masked), axis=-1)
    divergence = target * (jnp.log(target + INDEXER_EPS) - jnp.log(indexer + INDEXER_EPS))
    return compensated_sum(divergence, (divergence.ndim - 1,))


def _rotate(interleave: bool, part, freqs_cos, freqs_sin):
    """Rotate a rope slice in interleaved pairs (DeepSeek's main head, GLM's
    indexer), left in the reference's half layout, or half-split ones (V3.2's
    indexer)."""
    return rotate(deinterleaved(part) if interleave else part, freqs_cos, freqs_sin)


class IndexerFront:
    """The lightning indexers' shared front end, mixed into each (V3.2's `SparseIndexer`, GLM's
    k-pool, DeepSeek V4's CSA2): the query residual projected by `wq_b` into `n_heads` heads of
    `head_dim`, the hidden state by `weights_proj` into a weight a head, and the queries scored
    against keys by `index_scores`, at least in fp32 and under the indexer's `precision`, the inputs
    detached as the references detach them. Each indexer declares its own keys (`wk` and its
    `k_norm`) beside these, turns its queries as its reference does, and selects as it does."""

    # The fields of the module this mixes into, which declares them; a mixin's annotations are no
    # dataclass fields of their own.
    n_heads: int
    head_dim: int
    dtype: Dtype | None
    precision: PrecisionLike

    def declare_front(self) -> Callable[..., Dense]:
        """Declare `wq_b` and `weights_proj`, from the indexer's `setup`, and return the bias-free
        projection at its `dtype` and `precision` they are, for the indexer's own keys."""
        dense = functools.partial(Dense, use_bias=False, dtype=self.dtype, precision=self.precision,
                                  dot_general=float32_dot_general)
        self.wq_b = dense(self.n_heads * self.head_dim, name='wq_b')
        self.weights_proj = dense(self.n_heads, name='weights_proj')
        return dense

    def queries(self, q_resid):
        """The indexer's queries `[B, S, n_heads, head_dim]` of the detached query residual."""
        batch, length = q_resid.shape[:2]
        return self.wq_b(jax.lax.stop_gradient(q_resid)).reshape(batch, length, self.n_heads, self.head_dim)

    def weighed(self, agreements, hidden):
        """Scores `[B, S, T]` of the heads' `agreements` (`index_agreements`), weighted by
        `weights_proj` of the detached `hidden` (`index_scores`)."""
        return index_scores(agreements, self.weights_proj(jax.lax.stop_gradient(hidden)), self.head_dim,
                            self.precision)

    def scored(self, query, keys, hidden):
        """`query`'s scores against `keys` `[B, S, T]` (`weighed`)."""
        return self.weighed(index_agreements(query, keys, self.precision), hidden)


@logical_axes({
    ("wq_b",): ("qlora", "index"),
    ("wk",): ("embed", "index"),
    # The key norm is a LayerNorm, so scale and bias share the indexer width.
    ("k_norm",): ("index",),
    # Maps the hidden states onto one weight per indexer head.
    ("weights_proj",): ("embed", "index"),
})
class SparseIndexer(IndexerFront, nn.Module):
    """DeepSeek sparse attention's lightning indexer: a score per query and key.

    `modeling_deepseek_v32.DeepseekV32Indexer`: `wq_b` reads the query residual,
    `wk` the hidden states into cached keys, and `weights_proj` weights the
    heads into one score per key; `select` keeps each query's top-k. The
    indexer reads its inputs detached, so the main loss and the indexer's own
    (`indexer_kl`) each train only their side. V3.2 rotates the indexer's rope
    slice as half-split pairs and GLM as interleaved ones
    (modeling_glm_moe_dsa.py:231-232), which `rope_interleave` selects.
    """

    n_heads: int
    head_dim: int
    rope_head_dim: int
    top_k: int | None = None
    rope_interleave: bool = False
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        if self.head_dim <= self.rope_head_dim:
            raise ValueError(
                f"the indexer rotates a {self.rope_head_dim}-wide rope slice "
                f"out of heads of width {self.head_dim}, so the heads have "
                "to be wider")
        dense = self.declare_front()
        self.wk = dense(self.head_dim, name='wk')
        # A LayerNorm with weight and bias at a hardcoded 1e-6: the reference
        # names no config for it (modeling_deepseek_v32.py, DeepseekV32Indexer).
        self.k_norm = LayerNorm(epsilon=1e-6, dtype=self.dtype, name='k_norm')

    def keys(self, hidden):
        """The indexer's keys for these hidden states: `[B, S, head_dim]`."""
        return self.k_norm(self.wk(jax.lax.stop_gradient(hidden)))

    def rotated_keys(self, keys, freqs_cos, freqs_sin):
        """`keys` with their rope slice rotated at their own positions.

        Rotation happens once, before caching, so a cached key keeps the
        angle of its position while later queries rotate at theirs, which is
        what the reference's `update_indexer` ordering does.
        """
        k_rot, k_pass = jnp.split(keys, [self.rope_head_dim], axis=-1)
        k_rot = _rotate(self.rope_interleave, k_rot[:, :, None, :], freqs_cos, freqs_sin)
        return jnp.concatenate([k_rot[:, :, 0, :], k_pass], axis=-1)

    def scores(self, hidden, q_resid, keys, freqs_cos, freqs_sin):
        """Index scores of every query against every key: `[B, S, T]`, fp32.

        `keys` are the rotated keys of every candidate (the cache on
        decode); the query side is computed on each call. Scores run in
        fp32: the head weighting multiplies by `n_heads ** -0.5` in fp32 in
        the reference, and the relu keeps only the positive agreements.
        """
        q_rot, q_pass = jnp.split(self.queries(q_resid), [self.rope_head_dim], axis=-1)
        query = jnp.concatenate(
            [_rotate(self.rope_interleave, q_rot, freqs_cos, freqs_sin), q_pass], axis=-1)
        agreements = index_agreements(query, keys, self.precision)
        scores = self.weighed(agreements, hidden)
        if not self.is_initializing():
            # Kept only where a caller opens the collection, as a selection's margin does.
            self.sow(SELECTIONS, "scores", scores)
            self.sow(SELECTIONS, "agreements", agreements)
        return scores

    def select(self, scores, keep):
        """The top-k keys of each query among those `keep` allows, as
        `top_k_selection` returns them."""
        if self.top_k is None:
            raise ValueError("selecting keys needs the indexer's top_k")
        return top_k_selection(scores, keep, self.top_k)


@logical_axes({
    ("q_a_proj",): ("embed", "qlora"),
    ("q_b_proj",): ("qlora", "attention"),
    # The output concatenates the latent and the decoupled rope head, which
    # ride into the cache together; the name picks the sharded side.
    ("kv_a_proj_with_mqa",): ("embed", "kvlora"),
    ("kv_b_proj",): ("kvlora", "attention"),
    ("o_proj",): ("attention", "embed"),
    ("q_a_layernorm",): ("qlora",),
    ("kv_a_layernorm",): ("kvlora",),
})
class MultiHeadLatentAttention(nn.Module):
    """DeepSeek's multi-head latent attention, dense or sparse.

    decode=True runs against the cache like the standard mixer; `positions` and
    `segment_ids` behave as there. `rope_scaling`, the layer's one rope scaling
    (`MixerContext.rope_scaling`: YaRN, Llama 3.1's ramp or LongRoPE),
    transforms `rope_theta`, the layer's base, at every rotation it makes, and
    a YaRN ramp's mscale pre-scales the query. The two latent norms are the model's RMSNorm, as the
    reference builds them from the layer's one norm class.

    `index_n_heads` and `index_head_dim` put the indexer beside the attention
    and `index_topk` makes it select, the released V3.2 layer; without a top-k
    the indexer only scores, V3.2's dense warm-up (arXiv 2512.02556, section
    2.1.1). With the `indexer` collection open, training sows `indexer_kl`
    under `kl` over the keys the attention used. `index_shared` is GLM's
    IndexShare layer (modeling_glm_moe_dsa.py:313-318, 432-446): no indexer of
    its own, it attends the selection its provider stashed in `kv_store`.

    The query width `qk_nope_head_dim + qk_rope_head_dim` (192 in the released
    V3 configs, past cudnn's tiles) decides the kernel, values padded to it
    (`widen_value_heads`). A mask materialized outside decode (packed
    documents, row validity, the selection) takes 'auto' and 'cudnn' to xla.
    """

    reads: ClassVar[frozenset[str]] = frozenset({"pairwise_mask", "key_positions", "key_runs",
                                                 "position_bias"})
    """Of the call's `MIXING` fields (`dew.nn.inputs.MixerCall`), a dense layer's pairwise mask or key
    runs, the key positions they are read at and a position bias, on an uncached call, as the standard
    attention reads them; an indexed layer narrows its keys by the documents, the rows' validity and
    its own selection alone, and refuses them (`_check_call`)."""

    emb_features: int
    num_heads: int
    max_seq_len: int
    q_lora_rank: int | None
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    causal: bool = True
    rope_theta: float = 10000.0
    rope_interleave: bool = True
    rope_scaling: Scaling | None = None
    # DeepSeek and GLM norm the latents at RMSNorm's 1e-6 default whatever
    # the trunk's rms_norm_eps says (modeling_deepseek_v3.py:392,404,
    # modeling_glm_moe_dsa.py:349,361).
    norm_eps: float = 1e-6
    scale_offset: bool = False
    scale_after_cast: bool = False
    attention_bias: bool = False
    index_topk: int | None = None
    index_n_heads: int | None = None
    index_head_dim: int | None = None
    index_rope_interleave: bool = False
    index_shared: bool = False
    sliding_window: int | None = None
    """A dense layer's window: each query sees the keys within this many of its place, itself included,
    counting its document's real tokens (`dew.nn.attention.visibility_mask`); a non-causal layer windows
    only under `bidirectional_window`. An indexed layer's candidates take no window."""
    bidirectional_window: bool = False
    """Whether a non-causal layer keeps its window on both sides of a query (the layer context's flag of
    the name); otherwise it reads its whole row."""
    attention_chunk: int | None = None
    """A dense layer's chunk: each query sees the keys in its own chunk of this many places
    (`dew.nn.attention.chunk_mask`), as Llama 4's chunked layers do."""
    bidirectional_blocks: bool = False
    """A causal dense layer reads each block of its call's `block_groups` whole, as the standard
    attention's flag of the name does, on an uncached call."""
    bidirectional_images: bool = False
    """A causal dense layer reads each image of its call's `image_groups` whole, on an uncached call."""
    kv_store_key: str | None = None
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl
    kv_cache: KVCache = dataclasses.field(default_factory=KVCache)
    """The decode cache's storage layout (`dew.nn.kv_cache`), paged or quantized, over the latents
    or, under the sparse layout, the expanded keys."""
    rotary: bool = True
    """False is Kimi K3's MLA (`mla_use_nope`, modeling_kimi_linear.py:396-437
    of moonshotai/Kimi-K3 at f831ab6): the decoupled rope head is kept as
    shared key dims but never rotated, so the layer has no position at all."""
    output_gate: bool = False
    """Kimi K3's `mla_use_output_gate`: the heads' output times
    `sigmoid(g_proj(x))` before `o_proj` (modeling_kimi_linear.py:398-401, 470-472)."""
    force_fp32_for_softmax: bool = True

    def setup(self):
        if self.qk_rope_head_dim % 2:
            raise ValueError(
                "rotary positions rotate pairs, so the rope head dim must be "
                f"even, got {self.qk_rope_head_dim}")
        if self.qk_nope_head_dim < 1 or self.v_head_dim < 1:
            raise ValueError(
                "the nope head dim and the value head dim must be positive, "
                f"got {self.qk_nope_head_dim} and {self.v_head_dim}")
        if (self.index_n_heads is None) != (self.index_head_dim is None):
            raise ValueError(
                "the indexer needs its head count and head dim together, "
                "both set or both unset")
        if self.index_topk is not None and self.index_n_heads is None:
            raise ValueError(
                "index_topk selects with the indexer, which index_n_heads "
                "and index_head_dim have to describe")
        if self.index_shared and (self.index_n_heads is not None or self.index_topk is not None):
            raise ValueError(
                "index_shared reads another layer's selection, so the layer "
                "carries no indexer of its own: index_n_heads, index_head_dim "
                "and index_topk have to be None")
        if self.index_shared and self.kv_store_key is None:
            raise ValueError(
                "index_shared reads the selection under kv_store_key, which "
                "names the layer type that stashes it")
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        dense = functools.partial(
            Dense, use_bias=self.attention_bias, dtype=self.dtype,
            precision=self.precision, dot_general=float32_dot_general)
        norm = functools.partial(
            RMSNorm, epsilon=self.norm_eps,
            scale_offset=self.scale_offset,
            scale_after_cast=self.scale_after_cast, dtype=self.dtype)
        if self.q_lora_rank is None:
            # The reference's plain q_proj takes no bias, whatever
            # attention_bias says; only the low-rank projections do.
            self.q_proj = Dense(
                self.num_heads * qk_head_dim, use_bias=False, dtype=self.dtype,
                precision=self.precision, name='q_proj', dot_general=float32_dot_general)
        else:
            self.q_a_proj = dense(self.q_lora_rank, name='q_a_proj')
            self.q_a_layernorm = norm(name='q_a_layernorm')
            self.q_b_proj = Dense(
                self.num_heads * qk_head_dim, use_bias=False, dtype=self.dtype,
                precision=self.precision, name='q_b_proj', dot_general=float32_dot_general)
        self.kv_a_proj_with_mqa = dense(
            self.kv_lora_rank + self.qk_rope_head_dim, name='kv_a_proj_with_mqa')
        self.kv_a_layernorm = norm(name='kv_a_layernorm')
        self.kv_b_proj = Dense(
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            use_bias=False, dtype=self.dtype, precision=self.precision,
            name='kv_b_proj', dot_general=float32_dot_general)
        self.o_proj = dense(self.emb_features, name='o_proj')
        if self.output_gate:
            self.g_proj = Dense(self.num_heads * self.v_head_dim, use_bias=False,
                                dtype=self.dtype, precision=self.precision, name='g_proj',
                                dot_general=float32_dot_general)
        if self.index_n_heads is not None and self.index_head_dim is not None:
            if self.q_lora_rank is None:
                raise ValueError(
                    "the indexer reads the query residual, which only exists "
                    "with a q_lora_rank")
            self.indexer = self.make_indexer()

    def make_indexer(self) -> nn.Module:
        """The indexer beside the attention, built in `setup`: V3.2's lightning indexer."""
        assert self.index_n_heads is not None and self.index_head_dim is not None
        return SparseIndexer(
            n_heads=self.index_n_heads, head_dim=self.index_head_dim,
            rope_head_dim=self.qk_rope_head_dim, top_k=self.index_topk,
            rope_interleave=self.index_rope_interleave,
            dtype=self.dtype, precision=self.precision, name=INDEXER)

    @property
    def indexed(self) -> bool:
        """Whether the V3.2 indexer scores the keys beside the attention."""
        return self.index_n_heads is not None

    @property
    def sparse(self) -> bool:
        """Whether each query attends a selection of the keys: this layer's
        indexer's, or the one it shares."""
        return self.index_topk is not None or self.index_shared

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """The latent and expanded projections plus training's QK and AV."""
        if self.indexed or self.index_shared:
            raise MissingFlopsCount("MultiHeadLatentAttention needs indexer selection geometry for train/mfu")
        query = self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
        if self.q_lora_rank is None:
            count = projection_flops(lengths, self.emb_features, query)
        else:
            count = (projection_flops(lengths, self.emb_features, self.q_lora_rank)
                     + projection_flops(lengths, self.q_lora_rank, query))
        count += projection_flops(lengths, self.emb_features, self.kv_lora_rank + self.qk_rope_head_dim)
        count += projection_flops(lengths, self.kv_lora_rank,
                                   self.num_heads * (self.qk_nope_head_dim + self.v_head_dim))
        count += projection_flops(lengths, self.num_heads * self.v_head_dim, self.emb_features)
        if self.output_gate:
            count += projection_flops(lengths, self.emb_features, self.num_heads * self.v_head_dim)
        return count + attention_flops(lengths, self.num_heads,
                                       self.qk_nope_head_dim + self.qk_rope_head_dim,
                                       self.v_head_dim, causal=self.causal)

    def _select(self, index_scores, keep, kv_store):
        """The keys each query attends among `keep`, as `top_k_selection`
        returns them: this layer's top-k, stashed for the layers sharing it,
        or the stashed selection read."""
        if self.index_shared:
            if kv_store is None or self.kv_store_key not in kv_store:
                raise ValueError(
                    f"layer shares the index selection under {self.kv_store_key!r} "
                    "but no earlier selecting layer stashed one; the model has to "
                    "pass one kv_store dict down its layer stack")
            indices = kv_store[self.kv_store_key]
            keep = jnp.broadcast_to(keep, (*indices.shape[:2], keep.shape[-1]))
            allowed = jnp.take_along_axis(keep, jnp.maximum(indices, 0), axis=-1)
            return jnp.where(allowed & (indices >= 0), indices, -1)
        selected = self.indexer.select(index_scores, keep)
        if kv_store is not None and self.kv_store_key is not None:
            kv_store[self.kv_store_key] = selected
        return selected

    @property
    def query_scale(self) -> float:
        """The softmax-scale ratio the query carries into the kernel.

        The kernel scales by `1 / sqrt(qk_head_dim)` itself; YaRN's mscale
        rides on the query, so any other rope carries exactly 1.0.
        """
        if not isinstance(self.rope_scaling, YarnScaling):
            return 1.0
        return yarn_query_scale(self.rope_scaling)

    def _queries(self, x, place: LogicalAxes):
        """`[B, S, H, nope+rope]` queries and the residual the indexer reads,
        the query latent computed where `place` puts it."""
        batch, length, _ = x.shape
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # The projections carry the names a remat policy saves or offloads
        # (mixer_base.RESIDUALS); kv_b_proj is the fused K/V one.
        if self.q_lora_rank is None:
            q_resid = None
            queries = self.q_proj(x)
        else:
            q_resid = constrain(self.q_a_layernorm(self.q_a_proj(constrain(x, place))), place)
            queries = self.q_b_proj(constrain(q_resid, RESIDUAL))
        queries = checkpoint_name(queries, 'q_proj')
        return queries.reshape(batch, length, self.num_heads, qk_head_dim), q_resid

    def _latents(self, x, place: LogicalAxes):
        """The normed KV latent and the raw decoupled rope head.

        The tensor axis splits the heads, not the latents, so every tensor
        shard computing the down-projections (and `_queries`' own) over every
        token was 13% more work a device at DeepSeek-V3's shape under
        tensor=8. Where `place` is `SPREAD` they run on each shard's own
        tokens, a slice of a residual every shard holds whole, and the
        latents are gathered for the head-split up-projections."""
        compressed = constrain(self.kv_a_proj_with_mqa(constrain(x, place)), place)
        latent, rot = jnp.split(compressed, [self.kv_lora_rank], axis=-1)
        return constrain(self.kv_a_layernorm(latent), RESIDUAL), constrain(rot, RESIDUAL)

    def _rotated(self, q_rot, rot, positions, reach=None):
        """Rotate the query's and the latent's rope heads at `positions`.

        Returns the rotated pair and the angles, which the indexer rotates
        its own keys with. The decoupled rope head is shared by every query
        head, so it rotates with a head axis added and taken away again. A
        layer without `rotary` returns the heads as they are and no angles.
        """
        if not self.rotary:
            return q_rot, rot, None, None
        freqs_cos, freqs_sin = rotary_freqs(
            positions, self.qk_rope_head_dim, self.rope_theta, rope_scaling=self.rope_scaling,
            dtype=at_least_fp32(q_rot.dtype), reach=reach)
        return (_rotate(self.rope_interleave, q_rot, freqs_cos, freqs_sin),
                _rotate(self.rope_interleave, rot[:, :, None, :], freqs_cos, freqs_sin)[:, :, 0, :],
                freqs_cos, freqs_sin)

    def _expand(self, latent, rot):
        """Latent and rope head into per-head keys and values."""
        batch, length = latent.shape[0], latent.shape[1]
        width = self.qk_nope_head_dim + self.v_head_dim
        kv = checkpoint_name(self.kv_b_proj(latent), 'kv_proj').reshape(
            batch, length, self.num_heads, width)
        nope, values = jnp.split(kv, [self.qk_nope_head_dim], axis=-1)
        rot = jnp.broadcast_to(
            rot[:, :, None, :], (batch, length, self.num_heads, self.qk_rope_head_dim))
        return jnp.concatenate([nope, rot], axis=-1), values

    @nn.compact
    def __call__(self, x, call: MixerCall = PLAIN_CALL, prediction_phase: PredictionPhase = "ordinary"):
        """Project, cache and attend through the latent shell with the kind's candidate selection."""
        batch, length, _ = x.shape
        self._check_call(call, batch, length)
        place = down_projection(x, (self.q_lora_rank or 0) + self.kv_lora_rank + self.qk_rope_head_dim)
        queries, q_resid = self._queries(x, place)
        latent, rot = self._latents(x, place)
        decode, attention_metadata = call.decode, call.metadata
        causal, mask, objective = self.causal, None, None
        unselected = None
        implementation = self.attention_impl
        tokens = coordinates_of(attention_metadata, batch, length)
        # Each token's logical place (`Coordinates.places`); a cached call's default is its slot.
        positions, segment_ids = tokens.positions if decode else tokens.places(), tokens.segments
        valid = None if attention_metadata is None else attention_metadata.valid
        row_valid = jnp.ones((batch, length), bool) if valid is None else jnp.asarray(valid, bool)
        logical_positions = positions
        # jnp splits at indices where torch splits into sizes: one cut point,
        # since the widths add up exactly.
        q_pass, q_rot = jnp.split(
            queries, [self.qk_nope_head_dim], axis=-1)
        query_basis, own = None, False
        if decode:
            q_rot, key, value, mask, query_basis = self._decode_step(
                x, q_rot, rot, latent, q_resid, logical_positions, valid, call, prediction_phase)
            if self.sparse:
                unselected = self._empty_selection(mask[:, 0], row_valid)
            causal = False
        else:
            positions = jnp.arange(length) if positions is None else jnp.asarray(positions)
            # Under LongRoPE a packed document picks its factors by its own furthest place
            # (`document_reach`), so it rotates as it does alone.
            reach = (document_reach(segment_ids, valid, positions)
                     if isinstance(self.rope_scaling, LongRopeScaling) and segment_ids is not None else None)
            q_rot, rot, freqs_cos, freqs_sin = self._rotated(q_rot, rot, positions, reach)
            # Windows, chunks and groups count each document's real tokens from its first place
            # (`counted_places`); rotation reads the places as given.
            counted = (jnp.broadcast_to(positions, (batch, length)) if logical_positions is None
                       else counted_places(segment_ids, valid, logical_positions))
            visible, own = self._dense_visibility(attention_metadata, positions, counted, segment_ids, valid,
                                                  causal, batch, length)
            if visible is not None:
                mask, causal = visible, False
            elif segment_ids is not None:
                inside = document_mask(segment_ids)[:, None]
                mask = inside
                if causal:
                    mask = jnp.logical_and(
                        inside, causal_attention_mask(
                            jnp.arange(length), length))
                causal = False
            if self.indexed or self.index_shared:
                # The keys a query may attend before selection: the rows'
                # causal order (packed positions restart per document, so
                # they order nothing here) and the packed base when there
                # is one. The indexer selects among those, and the KL
                # measures over whatever the attention then uses.
                keep = None if mask is None else mask[:, 0]
                if causal:
                    rows = causal_attention_mask(jnp.arange(length), length)[:, 0]
                    keep = rows if keep is None else jnp.logical_and(keep, rows)
                if keep is None:
                    keep = jnp.ones((1, length, length), jnp.bool_)
                # Candidates carry every row even when all its keys are real; pooled indexers
                # gather row-specific slot lists from this visibility.
                keep = jnp.logical_and(keep, row_valid[:, None, :])
                index_keys = (self._rotated_index_keys(self._index_keys(x, row_valid), freqs_cos, freqs_sin)
                              if self.indexed else None)
                selection, index_scores = self._selection(
                    x, q_resid, index_keys, keep, freqs_cos, freqs_sin, call, row_valid,
                    slots=positions, prediction_phase=prediction_phase)
                if self.sparse:
                    if self._attends_sparsely(selection, length):
                        return self._output(self._sparse_attention(
                            q_pass, q_rot, latent, rot, selection, index_scores), x)
                    keep = selection_mask(selection, length)
                    mask, causal = keep[:, None], False
                    unselected = self._empty_selection(keep, row_valid)
                if (index_scores is not None and not self.is_initializing()
                        and self.is_mutable_collection(INDEXER_COLLECTION)):
                    objective = (index_scores, keep)
            key, value = self._expand(latent, rot)
        if valid is not None and not own:
            # A call's own pairwise mask or key runs replace its validity, as the shared core's do.
            queries_valid = jnp.asarray(valid, bool)[:, None, :, None]
            mask = queries_valid if mask is None else mask & queries_valid
            if not decode:
                mask = mask & jnp.asarray(valid, bool)[:, None, None, :]
        query = self._scaled_query(q_pass, q_rot)
        if query_basis is not None:
            query = query_basis(query)
        if mask is not None and not decode:
            # Packed documents, row validity and the indexer's selection all
            # materialize a mask on the training path, so they take the xla
            # kernel, the way the standard mixer's own masks do. Decoding
            # runs no backward pass, so its cache mask keeps the fused path.
            implementation = kernel_for_materialized_mask(
                implementation, query, key, dtype=self.dtype, precision=self.precision,
                force_fp32_for_softmax=self.force_fp32_for_softmax)
        if objective is not None:
            self._sow_indexer_kl(*objective, query, key)
        bias = None if decode or attention_metadata is None else attention_metadata.position_bias
        if bias is not None:
            implementation = kernel_for_materialized_mask(
                implementation, query, key, dtype=self.dtype, precision=self.precision,
                force_fp32_for_softmax=self.force_fp32_for_softmax)
        return self._attend(query, key, value, x, mask, implementation, causal=causal, bias=bias,
                            unselected=unselected, force_fp32_for_softmax=self.force_fp32_for_softmax)

    def _dense_visibility(self, metadata, positions, places, segment_ids, valid, causal: bool, batch: int,
                          length: int) -> tuple[jax.Array | None, bool]:
        """An uncached dense call's visibility `[B, 1, Q, K]` as the shared attention core resolves it
        (`call_visibility`, `dense_visibility`), None where its causality and documents alone give it, and
        whether the call's own pairwise mask or key runs replaced its causality, documents, groups and
        validity. `positions` are its tokens' rotary places, which a call's own visibility reads, and
        `places` their places counted over their documents' real tokens, which the window, the chunk and
        the groups read."""
        own = None if metadata is None else call_visibility(
            metadata, positions, batch=batch, queries=length, keys=length, key_places=places,
            sliding_window=self.sliding_window)
        groups = {name: jnp.asarray(getattr(metadata, name), jnp.int32) for name in self._group_fields()
                  if metadata is not None and getattr(metadata, name) is not None}
        window = self.sliding_window if causal or self.bidirectional_window else None
        if own is None and not (groups or window is not None or self.attention_chunk is not None):
            return None, False
        if own is not None:
            mask, runs, window, key_places = own
            causal, segment_ids, groups, valid = False, None, {}, None
        else:
            mask = runs = None
            key_places = places
        return dense_visibility(mask, segment_ids, causal, window, valid, runs, places, key_places, length,
                                length, self.attention_chunk, **groups), own is not None

    def _check_call(self, call: MixerCall, batch: int, length: int) -> None:
        metadata = call.metadata
        if metadata is not None and metadata.rotary_positions is not None:
            raise ValueError("MLA does not implement multi-axis rotary positions")
        selecting = self.indexed or self.index_shared
        if selecting and (self.sliding_window is not None or self.attention_chunk is not None):
            raise ValueError("an indexed mla layer selects its candidates among every key before it; "
                             "a window or chunk narrows a dense layer's keys")
        own = None if metadata is None else [name for name in ("pairwise_mask", "key_runs", "position_bias")
                                             if getattr(metadata, name) is not None]
        if own and (selecting or call.decode):
            raise ValueError(f"an mla layer reads {', '.join(own)} on a dense layer's uncached call, its "
                             "keys the call's own")
        grouped = [name for name in self._group_fields()
                   if metadata is not None and getattr(metadata, name) is not None]
        if grouped and (selecting or call.decode):
            raise ValueError(f"an mla layer reads {', '.join(grouped)} whole on a dense layer's uncached "
                             "call")
        if call.decode and self.attention_chunk is not None and metadata is not None and (
                coordinates_of(metadata, batch, length).positions is not None):
            raise ValueError("a chunked mla layer decodes at its cache's slots, so it takes no places of "
                             "the call's own")

    def _group_fields(self) -> tuple[str, ...]:
        """The `AttentionMetadata` group fields the layer reads whole (`group_fields`)."""
        return group_fields(self.causal, images=self.bidirectional_images, blocks=self.bidirectional_blocks)

    def _index_keys(self, x, valid):
        return self.indexer.keys(x)

    def _rotated_index_keys(self, keys, freqs_cos, freqs_sin):
        return self.indexer.rotated_keys(keys, freqs_cos, freqs_sin)

    def _selection(self, x, q_resid, index_keys, keep, freqs_cos, freqs_sin, call, row_valid, *,
                    slots: jax.Array, allocated: bool = False, committed=None,
                    prediction_phase: PredictionPhase = "ordinary"):
        """Select candidates and retain the lightning scores for its optional training objective."""
        scores = (self.indexer.scores(x, q_resid, index_keys, freqs_cos, freqs_sin)
                  if self.indexed else None)
        return (self._select(scores, keep, call.kv_store) if self.sparse else None), scores

    def _empty_selection(self, selected, row_valid) -> jax.Array | None:
        return None

    def _decode_step(self, x, q_rot, rot, latent, q_resid, logical_positions, valid, call,
                      prediction_phase: PredictionPhase):
        """One decode step's rotated query heads, its keys and values with the
        cache's joined, the mask over them, and what puts the scaled query in the
        basis the cached keys are in (None for none): the expanded keys and the
        indexer's selection under the sparse layout, or the latents expanded
        after the append under the dense one. `kv_cache`, the cache's layout,
        stores either form paged or quantized."""
        # The cache hands out the slots first, because the rope heads
        # rotate at absolute positions: the queries at this step's slots
        # and the appended latents at theirs, while the cached ones keep
        # the angles of the slots they were written at.
        if self.sparse:
            allocated = self.has_variable("cache", "cache_index")
            committed = self.get_variable("cache", "cache_index")
            row_valid = jnp.ones(x.shape[:2], bool) if valid is None else jnp.asarray(valid, bool)
            # Allocate from the actual projection's dtype and retain its one interceptor/dropout
            # draw. Only the decoupled rope channels change once the cache supplies their slots.
            key, value = self._expand(latent, rot)
            index_keys = self._index_keys(x, row_valid) if self.indexed else None
            positions, append, basis = open_latent_cache(self, key, self.v_head_dim, index_keys,
                                                         self.max_seq_len, valid=valid, layout=self.kv_cache)
            q_rot, rot, freqs_cos, freqs_sin = self._rotated(
                q_rot, rot,
                positions if logical_positions is None else logical_positions)
            if self.rotary:
                key = key.at[..., self.qk_nope_head_dim:].set(rot[:, :, None, :])
            if index_keys is not None:
                index_keys = self._rotated_index_keys(index_keys, freqs_cos, freqs_sin)
            key, value, index_full = append(key, value, index_keys)
            keep = self._visible(positions, key.shape[1])[:, 0]
            selection, _ = self._selection(
                x, q_resid, index_full, keep, freqs_cos, freqs_sin, call, row_valid, slots=positions,
                allocated=allocated, committed=committed, prediction_phase=prediction_phase)
            mask = selection_mask(selection, key.shape[1])[:, None]
            query_basis = basis.query
        else:
            # The latent and its rotary part, one head each, as the attention cache's keys and values,
            # the latents read back in their own basis, which the expansion reads.
            positions, append = open_kv_cache(self, latent[:, :, None], self.max_seq_len, valid=valid,
                                              value_dim=self.qk_rope_head_dim, layout=self.kv_cache)
            q_rot, rot, _, _ = self._rotated(
                q_rot, rot,
                positions if logical_positions is None else logical_positions)
            latent, rot = append(latent[:, :, None], rot[:, :, None])
            latent, rot = append.store.plain(latent)[:, :, 0], rot[:, :, 0]
            key, value = self._expand(latent, rot)
            mask = self._visible(positions, key.shape[1])
            query_basis = None
        return q_rot, key, value, mask, query_basis

    def _visible(self, slots, total: int):
        """`[B, 1, T, total]`: which of the cache's `total` slots each query at `slots` sees, those up
        to its own slot that hold a token, within its window and chunk counted over the cache's
        slots, which hold each row's real tokens in order."""
        seen = causal_attention_mask(slots, total, key_valid=cached_validity(self, total))
        if self.sliding_window is None and self.attention_chunk is None:
            return seen
        slots = jnp.asarray(slots)
        keys = jnp.arange(total)
        if self.sliding_window is not None:
            seen = seen & (slots[..., None] - keys < self.sliding_window)[:, None]
        if self.attention_chunk is not None:
            seen = seen & chunk_mask(slots, jnp.broadcast_to(keys, (slots.shape[0], total)),
                                     self.attention_chunk)
        return seen

    def _attend(self, query, key, value, x, mask, implementation, *, causal: bool = False, bias=None,
                unselected=None, force_fp32_for_softmax: bool = True):
        """The attention over `key` and `value` and its output projection, sowing the per-head
        maxima QK-Clip reads where a caller opened the collection. A query `unselected` marks reads
        every key alike, the mean of the values, as a selection that chose nothing does in the
        reference, and `force_fp32_for_softmax` is a k-pool attention's own policy; a latent
        attention's kind refuses any but the default (`latent_refusals`)."""
        # The per-head maxima the QK-Clip reads, with the nope width the
        # clip needs to split the latent projections: computed only when a
        # caller opened the collection. Under a sparse selection the mask
        # holds the selected keys, so the maximum is over those.
        if self._qk_open():
            self.sow("qk", "max_logits", max_attention_logits(
                query, key, causal=causal, mask=mask, bias=bias))
            self.sow("qk", "qk_nope", jnp.asarray(self.qk_nope_head_dim))
        attention = scaled_dot_product_attention(
            query, key, value, dtype=self.dtype, precision=self.precision,
            force_fp32_for_softmax=force_fp32_for_softmax,
            implementation=implementation, causal=causal, mask=mask, bias=bias)
        if unselected is not None:
            wide = value.astype(at_least_fp32(value.dtype))
            uniform = jnp.mean(wide, axis=1, keepdims=True).astype(attention.dtype)
            attention = jnp.where(unselected, uniform, attention)
        return self._output(attention, x)

    def _qk_open(self) -> bool:
        return not self.is_initializing() and self.is_mutable_collection("qk")

    def _attends_sparsely(self, selection, total: int) -> bool:
        """Whether a whole-sequence pass runs `sparse_latent_attention`: when the
        selection is narrower than the keys. An open `qk` collection keeps the
        masked kernel for its dense maxima, and so does a `kv_b_proj` that is not
        linear in its input (`nonlinear_input`), since the absorbed path applies it
        to the identity, not to the latents.
        """
        return (selection.shape[-1] < total and not self._qk_open()
                and not self.nonlinear_input("kv_b_proj"))

    def nonlinear_input(self, name: str) -> bool:
        """Whether this call computes the submodule `name`, or a module under
        it, other than as a linear map of its input. The layer's own Dense is
        one unless the Qwix quantization the call runs under quantizes its
        activations or gradients (`dew.quantization.quantized_products`); a
        weight-only rule leaves it linear, by its quantized matrix, which the
        absorbed path reads. A method interceptor that changes it answers for
        itself: `dew.lora`'s dropout draws on its targets' inputs, and a
        checkpoint's input quantization (`dew.quantization`) rounds them."""
        return quantized_products(self, name).inputs

    def _scaled_query(self, q_pass, q_rot):
        """The whole query with YaRN's mscale on it, as every kernel reads it."""
        query = jnp.concatenate([q_pass, q_rot], axis=-1)
        scale = self.query_scale
        return query * scale if scale != 1.0 else query

    def _sow_indexer_kl(self, index_scores, index_keep, query, key):
        """The indexer's objective, per query, over the keys the attention
        itself attends: dense attention's causal set in the warm-up, the
        selected set under the top-k. The kernel scales the (yarn-prescaled)
        query by the head width, so the target does."""
        self.sow(INDEXER_COLLECTION, "kl", indexer_kl(
            index_scores, query, key, index_keep,
            1.0 / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim)))

    def _sparse_attention(self, q_pass, q_rot, latent, rot, selection, index_scores):
        """Attend the selection in the latent space (`sparse_latent_attention`):
        the heads' context, `[B, S, H, v_head_dim]`.

        `kv_b_proj` applied to the identity is its matrix as the layer
        computes it, dtype policy and any interceptor linear in its input
        (a LoRA adapter without dropout in this call, a rule that quantizes
        the weights alone) included, so the absorbed weights are the ones the
        expanded path multiplies by (`_attends_sparsely`). The indexer's objective, when open, still reads the
        expanded keys over the selection mask.
        """
        heads, nope = self.num_heads, self.qk_nope_head_dim
        matrix = self.kv_b_proj(jnp.eye(self.kv_lora_rank, dtype=latent.dtype)).reshape(
            self.kv_lora_rank, heads, nope + self.v_head_dim)
        key_weight, value_weight = jnp.split(matrix, [nope], axis=-1)
        scale = self.query_scale
        if (index_scores is not None and not self.is_initializing()
                and self.is_mutable_collection(INDEXER_COLLECTION)):
            key, _ = self._expand(latent, rot)
            self._sow_indexer_kl(index_scores, selection_mask(selection, latent.shape[1]),
                                 self._scaled_query(q_pass, q_rot), key)
        return sparse_latent_attention(
            q_pass * scale if scale != 1.0 else q_pass,
            q_rot * scale if scale != 1.0 else q_rot,
            latent, rot, key_weight, value_weight, selection,
            scale=1.0 / math.sqrt(nope + self.qk_rope_head_dim), precision=self.precision)

    def _output(self, attention, x):
        """o_proj over the heads' context, gated by `x` under `output_gate`."""
        batch, length = attention.shape[:2]
        context = checkpoint_name(attention, "context").reshape(
            batch, length, self.num_heads * self.v_head_dim
        )
        if self.output_gate:
            context = context * nn.sigmoid(self.g_proj(x))
        return checkpoint_name(self.o_proj(context), 'o_proj')


def latent_refusals(ctx: MixerContext) -> Mapping[str, object]:
    """What a latent attention does not run of its layer's context: the
    attention policies it is asked (`MixerContext.policies`), the standard
    attention's value norm, keys as values, score scale and partial rotation;
    a dense layer takes its window and chunk. Its cache takes the layer's layout (`kv_cache`), paged or
    quantized, over its latents or its expanded keys."""
    return {
        **ctx.policies(),
        "v_norm": ctx.v_norm,
        "k_eq_v": ctx.k_eq_v,
        "attention_scale": ctx.attention_scale,
        "partial_rotary_factor": ctx.partial_rotary_factor,
    }


@dataclasses.dataclass(frozen=True)
class MLAMixer(MixerBase):
    """The `mla` kind: DeepSeek's latent attention under the reference's names.

    `mixer={"name": "mla", "fields": {...}}` takes a DeepSeek config.json's fields, so
    translation renames nothing. The rope base and its scaling are the layer's
    (`MixerContext.rope_theta` and `rope_scaling`, the model's or its kind's). The context's GQA
    geometry and `qk_norm` are not read (the latent norms are always present).
    A `kv_shared` layer is GLM's IndexShare layer, which shares its provider's
    selection and so needs the kind to select. Dials this cannot honour (a
    values norm, a window, an attention scale, a partial rotary) are refused.
    """

    residuals = ATTENTION_CONTEXT_RESIDUALS

    q_lora_rank: int | None = None
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    rope_interleave: bool = True
    index_topk: int | None = None
    index_n_heads: int | None = None
    index_head_dim: int | None = None
    index_rope_interleave: bool = False
    mla_use_nope: bool = False
    mla_use_output_gate: bool = False
    bidirectional_blocks: bool = False
    """A causal dense layer reads each block of a call's `block_groups` whole (the standard kind's flag)."""
    bidirectional_images: bool = False
    """A causal dense layer reads each image of a call's `image_groups` whole."""

    @property
    def indexed(self) -> bool:
        """Whether the layer carries the indexer, selecting or not."""
        return self.index_n_heads is not None

    @property
    def reads_blocks(self) -> bool:
        return self.bidirectional_blocks

    @property
    def sparse(self) -> bool:
        """Whether the indexer selects the keys each query attends."""
        return self.index_topk is not None

    def build(self, ctx: MixerContext) -> Callable[..., nn.Module]:
        ctx.refuse("mla", latent_refusals(ctx),
                   "the latent attention scales by its head dims and the yarn mscale, rotates its rope head "
                   "whole, attends the whole sequence with an fp32 softmax and norms its latents, not its "
                   "values")
        if ctx.kv_shared and not self.sparse:
            raise ValueError(
                "a sharing mla layer attends the selection of an earlier "
                "layer's indexer, so the kind has to select: index_topk, "
                "index_n_heads and index_head_dim have to be set")
        return functools.partial(
            MultiHeadLatentAttention,
            emb_features=ctx.emb_features,
            num_heads=ctx.num_heads,
            max_seq_len=ctx.max_seq_len,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            causal=ctx.causal,
            rope_theta=ctx.rope_theta,
            rope_interleave=self.rope_interleave,
            rope_scaling=ctx.rope_scaling,
            scale_offset=ctx.scale_offset,
            scale_after_cast=ctx.scale_after_cast,
            attention_bias=ctx.attention_bias,
            index_topk=None if ctx.kv_shared else self.index_topk,
            index_n_heads=None if ctx.kv_shared else self.index_n_heads,
            index_head_dim=None if ctx.kv_shared else self.index_head_dim,
            index_rope_interleave=self.index_rope_interleave,
            index_shared=ctx.kv_shared,
            sliding_window=ctx.sliding_window,
            bidirectional_window=ctx.bidirectional_window,
            attention_chunk=ctx.attention_chunk,
            bidirectional_blocks=self.bidirectional_blocks,
            bidirectional_images=self.bidirectional_images,
            kv_store_key=ctx.kv_store_key,
            dtype=ctx.dtype,
            precision=ctx.precision,
            attention_impl=ctx.attention_impl,
            rotary=not self.mla_use_nope,
            output_gate=self.mla_use_output_gate,
            kv_cache=ctx.kv_cache)
