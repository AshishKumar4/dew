"""GLM-5.3-Flash's sparse attention: NoPE latent attention under a k-pool indexer.

The reference is transformers 5.16.1 models/glm5_next/modeling_glm5_next.py
(`Glm5NextTextAttention`, lines 1064-1256, and `Glm5NextTextIndexer`, lines
736-1024), read as the specification. The attention is DeepSeek V3.2's
(`dew.nn.mla`) with no rope head at all: the released config's
`qk_rope_head_dim` is 0 and the reference refuses anything else
(configuration_glm5_next.py:225-228), so the queries and keys are the
`qk_nope_head_dim` latents alone, scaled by that width, and positions enter
only through causality.

The indexer scores pools of `index_kpool` consecutive keys, not single keys.
A pool's key is a softmax average of its members' indexer keys, weighted by
learned gate scores plus a per-slot term. Each query picks
`index_topk // index_kpool` pools and attends every token in them. With
`index_kpool_always_select_tail`, the incomplete pool at the causal frontier
rides along, so a query always sees its most recent keys. Pools count from a
row's first real key, which makes a left-padded row pool like the same
tokens unpadded. The selection is integer indices under
`torch.no_grad` in the reference, so it is detached here as well: the main
loss reaches none of the indexer's weights.

Decode caches the expanded keys and values and the indexer's packed state
`[k | gate_scores | valid]` per slot, as the reference's indexed cache layers
do, in the fixed-capacity cache MLA's sparse variant uses; unused slots carry
a zero valid channel, which is what keeps their pools out of the candidates.

Opted-in prediction caches also retain the complete selected token list and
its physical origin. Extend publishes the last valid query; draft reuses it
until replay publishes another. Ordinary steps recompute and invalidate it.
An origin of -1 means no seed, distinct from a valid empty selection.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.dtypes import promote_dtype
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name

from .attention import (
    LayerNorm,
    RMSNorm,
    causal_attention_mask,
    kernel_for_materialized_mask,
    max_attention_logits,
    scaled_dot_product_attention,
)
from .inputs import AttentionMetadata, PredictionPhase
from .mixers import MixerBase, MixerContext, mixers
from .mla import INDEXER, open_expanded_cache
from .sharding import logical_axes


def selection_mask(indices, total: int):
    """`[B, S, N]` token indices, -1 for none, as the `[B, S, total]` bool mask
    the attention takes (`build_attention_mask_from_topk`,
    modeling_glm5_next.py:1218-1256): a key is visible to a query iff one of
    the query's indices names it, so out-of-range entries drop out."""
    batch, length, _ = indices.shape
    # A negative index would wrap around in jnp; sending it past the end
    # lets the scatter drop it, as it drops any index at or past `total`.
    slots = jnp.where(indices >= 0, indices, total)
    return jnp.zeros((batch, length, total), jnp.bool_).at[
        jnp.arange(batch)[:, None, None], jnp.arange(length)[None, :, None], slots
    ].set(True, mode='drop')


def _first_valid(valid, total: int):
    """Each row's first valid slot, or `total` on a row without one
    (modeling_glm5_next.py:940-944)."""
    return jnp.where(jnp.any(valid, axis=-1), jnp.argmax(valid.astype(jnp.int32), axis=-1), total)


# wq_b, wk, k_norm and weights_proj carry the V3.2 indexer's declarations
# under the same names; the pool compression's two tables have no side worth
# naming and take the shape heuristic.
@logical_axes({}, heuristic=(("index_kpool_compress_*",),))
class KPoolIndexer(nn.Module):
    """The k-pool indexer: which keys each query attends, by pools
    (`Glm5NextTextIndexer`, modeling_glm5_next.py:736-1024).

    `packed` is the per-token state the attention caches on decode,
    `[k | gate_scores | valid]`, and `select_indices` scores every pool of it for
    every query: the pooled keys against `wq_b` of the query residual, relu,
    the heads weighted by `weights_proj`, top `index_topk // index_kpool`
    pools among those whose last token the query sees, the tail appended.
    Parameter names are the checkpoint's under `indexer`, the compression
    tables in their torch layout: `index_kpool_compress_ape`
    `[kpool, head_dim]` and `index_kpool_compress_gate` `[head_dim, hidden]`.
    """

    emb_features: int
    n_heads: int
    head_dim: int
    top_k: int
    kpool: int
    always_select_tail: bool = True
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        if self.kpool < 1:
            raise ValueError(f"index_kpool must be positive, got {self.kpool}")
        if self.top_k % self.kpool:
            raise ValueError(
                f"the pool budget is index_topk // index_kpool whole pools, so index_topk "
                f"({self.top_k}) must be divisible by index_kpool ({self.kpool})")
        dense = functools.partial(nn.Dense, use_bias=False, dtype=self.dtype, precision=self.precision)
        self.wq_b = dense(self.n_heads * self.head_dim, name='wq_b')
        self.wk = dense(self.head_dim, name='wk')
        # A LayerNorm with weight and bias at a hardcoded 1e-6 (modeling_glm5_next.py:763).
        self.k_norm = LayerNorm(epsilon=1e-6, dtype=self.dtype, name='k_norm')
        self.weights_proj = dense(self.n_heads, name='weights_proj')
        self.index_kpool_compress_ape = self.param(
            'index_kpool_compress_ape', nn.initializers.zeros, (self.kpool, self.head_dim), jnp.float32)
        self.index_kpool_compress_gate = self.param(
            'index_kpool_compress_gate', nn.initializers.zeros, (self.head_dim, self.emb_features), jnp.float32)

    def packed(self, x, valid):
        """`[B, S, 2 * head_dim + 1]`: the indexer's keys, the pool gate scores
        and the validity channel (modeling_glm5_next.py:797-803)."""
        x = jax.lax.stop_gradient(x)
        keys = self.k_norm(self.wk(x))
        x, gate = promote_dtype(x, self.index_kpool_compress_gate, dtype=self.dtype)
        gate_scores = jnp.einsum('bsh,dh->bsd', x, gate, precision=self.precision)
        return jnp.concatenate([keys, gate_scores.astype(keys.dtype), valid.astype(keys.dtype)[..., None]], axis=-1)

    def _pools(self, packed):
        """The pools of a packed state, `[B, P]` of them: their keys
        `[B, P, head_dim]`, their token indices `[B, P, kpool]` (-1 on a slot
        outside the row) and whether every slot is valid
        (`get_pooled_states`, modeling_glm5_next.py:899-967)."""
        keys, gate_scores, valid = jnp.split(packed, [self.head_dim, 2 * self.head_dim], axis=-1)
        valid = valid[..., 0] > 0
        batch, total = valid.shape
        pools = -(-total // self.kpool)
        first = _first_valid(valid, total)
        indices = first[:, None, None] + jnp.arange(pools * self.kpool).reshape(1, pools, self.kpool)
        safe = jnp.clip(indices, 0, total - 1).reshape(batch, pools * self.kpool)
        grouped_keys = jnp.take_along_axis(keys, safe[..., None], axis=1).reshape(
            batch, pools, self.kpool, self.head_dim)
        grouped_gate = jnp.take_along_axis(gate_scores, safe[..., None], axis=1).reshape(
            batch, pools, self.kpool, self.head_dim)
        grouped_valid = jnp.take_along_axis(valid, safe, axis=1).reshape(batch, pools, self.kpool)
        grouped_valid = jnp.logical_and(grouped_valid, indices < total)
        pool_valid = jnp.all(grouped_valid, axis=-1)
        indices = jnp.where(grouped_valid, indices, -1)
        logits = grouped_gate.astype(jnp.float32) + self.index_kpool_compress_ape.astype(jnp.float32)
        logits = jnp.where(grouped_valid[..., None], logits, -jnp.inf)
        # A pool with no valid slot softmaxes to nan, which the reference
        # zeroes (modeling_glm5_next.py:964-966).
        probabilities = jnp.nan_to_num(jax.nn.softmax(logits, axis=2)).astype(keys.dtype)
        return jnp.sum(probabilities * grouped_keys, axis=2), indices, pool_valid

    def _tail(self, visible, valid):
        """Each query's incomplete pool as token indices, `[B, S, kpool - 1]`,
        -1 where there is none (`append_visible_tail`,
        modeling_glm5_next.py:974-1024): the `visible_count % kpool` tokens
        after the last complete pool the query sees."""
        _, _, total = visible.shape
        first = _first_valid(valid, total)
        visible_count = jnp.sum(visible, axis=-1, dtype=jnp.int32)
        tail_count = visible_count % self.kpool
        offsets = jnp.arange(self.kpool - 1)
        tail = (first[:, None] + visible_count - tail_count)[..., None] + offsets
        allowed = jnp.logical_and(offsets < tail_count[..., None], tail < total)
        tail_visible = jnp.take_along_axis(visible, jnp.clip(tail, 0, total - 1), axis=-1)
        return jnp.where(jnp.logical_and(allowed, tail_visible), tail, -1)

    def select_indices(self, x, q_resid, packed, visible):
        """Complete selected token indices `[B, S, N]`, -1 for an empty slot.

        `packed` is the state of every candidate (the cache on decode),
        `visible` `[B, S, T]` which of them the query may see (causality and
        validity). Scores run in fp32 as the reference's do, everything
        detached (modeling_glm5_next.py:773-877). Pools that score equally
        choose the lower token index, because `jax.lax.top_k` is stable;
        torch's `topk` promises no tie order, so a selection a tie decides
        may differ from the reference's. A pool the query cannot see is
        never chosen, whatever the tie rule.
        """
        x, q_resid, packed = (jax.lax.stop_gradient(part) for part in (x, q_resid, packed))
        batch, length, _ = x.shape
        total = packed.shape[1]
        valid = packed[..., -1] > 0
        query = self.wq_b(q_resid).reshape(batch, length, self.n_heads, self.head_dim)
        pool_keys, pool_indices, pool_valid = self._pools(packed)
        scores = jnp.einsum('bshd,bpd->bshp', query.astype(jnp.float32), pool_keys.astype(jnp.float32))
        scores = jnp.maximum(scores * (self.head_dim ** -0.5), 0)
        weights = self.weights_proj(x).astype(jnp.float32) * (self.n_heads ** -0.5)
        index_scores = jnp.einsum('bsh,bshp->bsp', weights, scores)
        # A pool is a candidate iff the query sees its last token
        # (modeling_glm5_next.py:832-839); its indices are clamped only so
        # the gather is legal, the validity test excludes it on its own.
        pool_end = jnp.clip(pool_indices[..., -1], 0, total - 1)
        pool_visible = jnp.take_along_axis(
            visible, jnp.broadcast_to(pool_end[:, None, :], (batch, length, pool_end.shape[-1])), axis=-1)
        candidates = jnp.logical_and(pool_visible, pool_valid[:, None])
        index_scores = jnp.where(candidates, index_scores, jnp.finfo(jnp.float32).min)
        # The reference drops the pools no row fills (:969-972) before this
        # minimum; a dropped pool is never a candidate, and a selected
        # non-candidate contributes no indices, so the selection is the same.
        select_k = min(self.top_k // self.kpool, index_scores.shape[-1])
        chosen = jax.lax.top_k(index_scores, select_k, is_stable=True)[1]
        chosen_valid = jnp.take_along_axis(candidates, chosen, axis=-1)
        chosen_indices = pool_indices[jnp.arange(batch)[:, None, None], chosen]
        tokens = jnp.where(chosen_valid[..., None], chosen_indices, -1).reshape(batch, length, -1)
        if self.always_select_tail and self.kpool > 1:
            tokens = jnp.concatenate([tokens, self._tail(visible, valid)], axis=-1)
        return tokens


# The latent projections and norms carry MLA's declarations under the same names.
class KPoolSparseAttention(nn.Module):
    """GLM-5.3-Flash's `deepseek_sparse_attention` layer
    (`Glm5NextTextAttention`, modeling_glm5_next.py:1064-1256).

    Low-rank queries (`q_a_proj`, `q_a_layernorm`, `q_b_proj`) and a latent
    the keys and values expand from (`kv_a_proj_with_mqa`, `kv_a_layernorm`,
    `kv_b_proj`), with no rope head: the width the logits scale by is
    `qk_nope_head_dim`. Each query attends the keys its `KPoolIndexer`
    selects, through an fp32 softmax over the boolean mask the reference's
    eager path builds; the indexer folds causality and row validity into the
    selection, so nothing else masks. Row validity (`attention_metadata.valid`)
    enters the indexer's valid channel and blanks the selection of an
    invalid query (modeling_glm5_next.py:801, 875); the hidden states are
    not zeroed, as the reference zeroes them for its linear layers only.

    decode=True runs against a fixed-capacity cache of `max_seq_len` slots
    of the expanded keys and values and the packed indexer state, the first
    call writing the prompt and each later call appending its tokens, with
    causality read off the cache slots. Packed documents have no reference
    form (pools run over one row's consecutive keys) and are refused. The
    latent norms are the model's RMSNorm at the model's epsilon
    (modeling_glm5_next.py:1103, 1116) under its `scale_offset` and
    `scale_after_cast`. `attention_impl` reaches the shared kernel path as
    MLA's does; the selection is a materialized mask, so outside decode
    'auto' and 'cudnn' run xla.
    """

    emb_features: int
    num_heads: int
    max_seq_len: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    v_head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    index_kpool: int
    index_kpool_always_select_tail: bool = True
    norm_eps: float = 1e-5
    scale_offset: bool = False
    scale_after_cast: bool = False
    attention_bias: bool = False
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl
    force_fp32_for_softmax: bool = True

    def setup(self):
        if self.qk_nope_head_dim < 1 or self.v_head_dim < 1:
            raise ValueError(
                "the nope head dim and the value head dim must be positive, "
                f"got {self.qk_nope_head_dim} and {self.v_head_dim}")
        if self.index_n_heads < 1 or self.index_head_dim < 1:
            raise ValueError(
                "the indexer needs heads of a positive width, got "
                f"{self.index_n_heads} of {self.index_head_dim}")
        dense = functools.partial(
            nn.Dense, use_bias=self.attention_bias, dtype=self.dtype, precision=self.precision)
        norm = functools.partial(
            RMSNorm, epsilon=self.norm_eps, scale_offset=self.scale_offset,
            scale_after_cast=self.scale_after_cast, dtype=self.dtype)
        # The reference's low-rank second halves take no bias, whatever
        # attention_bias says (modeling_glm5_next.py:1105-1121).
        self.q_a_proj = dense(self.q_lora_rank, name='q_a_proj')
        self.q_a_layernorm = norm(name='q_a_layernorm')
        self.q_b_proj = nn.Dense(
            self.num_heads * self.qk_nope_head_dim, use_bias=False, dtype=self.dtype,
            precision=self.precision, name='q_b_proj')
        self.kv_a_proj_with_mqa = dense(self.kv_lora_rank, name='kv_a_proj_with_mqa')
        self.kv_a_layernorm = norm(name='kv_a_layernorm')
        self.kv_b_proj = nn.Dense(
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), use_bias=False,
            dtype=self.dtype, precision=self.precision, name='kv_b_proj')
        self.o_proj = dense(self.emb_features, name='o_proj')
        self.indexer = KPoolIndexer(
            emb_features=self.emb_features, n_heads=self.index_n_heads, head_dim=self.index_head_dim,
            top_k=self.index_topk, kpool=self.index_kpool,
            always_select_tail=self.index_kpool_always_select_tail,
            dtype=self.dtype, precision=self.precision, name=INDEXER)

    @nn.compact
    def __call__(self, x, decode: bool = False, positions=None, segment_ids=None, kv_store=None,
                 attention_metadata: AttentionMetadata | None = None,
                 prediction_phase: PredictionPhase = "ordinary"):
        # No rope: the slot positions order the keys and nothing rotates.
        del positions, kv_store
        if segment_ids is not None:
            raise ValueError(
                "kpool_sparse_attention pools one row's consecutive keys, which a packed "
                "row of documents has no reference form for; pass one document per row")
        batch, length, _ = x.shape
        valid = None if attention_metadata is None else attention_metadata.valid
        if valid is not None:
            valid = jnp.asarray(valid, bool)
            if valid.shape != (batch, length):
                raise ValueError(f"row validity must be {(batch, length)}, got {valid.shape}")
        row_valid = jnp.ones((batch, length), bool) if valid is None else valid
        # The projections carry the names a remat policy saves or offloads
        # (causal_transformer.RESIDUALS); kv_b_proj is the fused K/V one.
        q_resid = self.q_a_layernorm(self.q_a_proj(x))
        query = checkpoint_name(self.q_b_proj(q_resid), 'q_proj').reshape(
            batch, length, self.num_heads, self.qk_nope_head_dim)
        latent = self.kv_a_layernorm(self.kv_a_proj_with_mqa(x))
        kv = checkpoint_name(self.kv_b_proj(latent), 'kv_proj').reshape(
            batch, length, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        key, value = jnp.split(kv, [self.qk_nope_head_dim], axis=-1)
        packed = self.indexer.packed(x, row_valid)
        implementation = self.attention_impl
        if decode:
            allocated = self.has_variable("cache", "cache_index")
            committed = self.get_variable("cache", "cache_index")
            slots, append = open_expanded_cache(self, key, value, packed, self.max_seq_len, valid=valid)
            key, value, packed = append(key, value, packed)
            # The reference's static cache: keys at every slot, causality
            # by slot against the query's, validity from the cache
            # (`get_visible_tokens`, modeling_glm5_next.py:879-897).
            visible = causal_attention_mask(
                slots, key.shape[1], key_valid=self.get_variable("cache", "cache_valid"))[:, 0]
            # SGLang 97c6978 index_topk_share.py:22-28,48-64 carries the complete
            # token list; draft hits must not append the new query's tail.
            if prediction_phase != "ordinary" or self.has_variable("cache", "selection_position"):
                if prediction_phase == "draft" and length != 1:
                    raise ValueError("draft index reuse accepts one candidate per row")
                count = min(self.index_topk // self.index_kpool, -(-self.max_seq_len // self.index_kpool))
                width = count * self.index_kpool + (self.index_kpool - 1 if self.index_kpool_always_select_tail else 0)
                saved = self.variable("cache", "selection_indices", jnp.full, (batch, width), -1, jnp.int32)
                origin = self.variable("cache", "selection_position", jnp.full, (batch,), -1, jnp.int32)
                active = jnp.any(row_valid, axis=1)
                if committed is not None:
                    origin.value = jnp.where(active & (origin.value >= committed), -1, origin.value)
                hit = (origin.value >= 0) & active if prediction_phase == "draft" else jnp.zeros((batch,), bool)
                frozen = jnp.broadcast_to(saved.value[:, None], (batch, length, width))
                if allocated and prediction_phase == "draft":
                    indices = nn.cond(
                        jnp.all(hit | ~active), lambda module: frozen,
                        lambda module: module.indexer.select_indices(x, q_resid, packed, visible), self)
                    indices = jnp.where(hit[:, None, None], frozen, indices)
                else:
                    indices = self.indexer.select_indices(x, q_resid, packed, visible)
                if allocated:
                    if prediction_phase == "ordinary":
                        origin.value = jnp.where(active, -1, origin.value)
                    else:
                        last = jnp.max(jnp.where(row_valid, jnp.arange(length), 0), axis=1)
                        publish = active & ~hit
                        saved.value = jnp.where(publish[:, None], indices[jnp.arange(batch), last], saved.value)
                        origin.value = jnp.where(publish, slots[jnp.arange(batch), last], origin.value)
            else:
                indices = self.indexer.select_indices(x, q_resid, packed, visible)
        else:
            visible = causal_attention_mask(jnp.arange(length), length, key_valid=row_valid)[:, 0]
            # The selection is always a mask, so training takes the kernel
            # a materialized mask runs on.
            implementation = kernel_for_materialized_mask(
                implementation, query, dtype=self.dtype, precision=self.precision,
                force_fp32_for_softmax=self.force_fp32_for_softmax)
            indices = self.indexer.select_indices(x, q_resid, packed, visible)
        selected = selection_mask(indices, key.shape[1])
        # An invalid query selects nothing (modeling_glm5_next.py:875).
        mask = jnp.logical_and(selected, row_valid[:, :, None])[:, None]
        # The per-head maxima the QK-Clip reads, over the selected keys,
        # with the nope width the clip splits the latent projections by.
        if not self.is_initializing() and self.is_mutable_collection("qk"):
            self.sow("qk", "max_logits", max_attention_logits(query, key, mask=mask))
            self.sow("qk", "qk_nope", jnp.asarray(self.qk_nope_head_dim))
        attention = scaled_dot_product_attention(
            query, key, value, dtype=self.dtype, precision=self.precision,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            implementation=implementation, mask=mask)
        return checkpoint_name(self.o_proj(checkpoint_name(attention, 'context').reshape(
            batch, length, self.num_heads * self.v_head_dim)), 'o_proj')


@mixers("kpool_sparse_attention")
@dataclasses.dataclass(frozen=True)
class KPoolSparseAttentionMixer(MixerBase):
    """The `kpool_sparse_attention` kind, by GLM-5.3-Flash's config fields
    (configuration_glm5_next.py:112-116, 134-136, 153-154).

    The record has no `qk_rope_head_dim`: the layer is NoPE by construction,
    as the reference requires (configuration_glm5_next.py:225-228), so a
    config that names one is refused by the registry as an unknown field.
    The latent norms take the model's `norm_eps`, which is where the
    reference points them (modeling_glm5_next.py:1103, 1116). The context's
    grouped-query geometry, rotary base and `qk_norm` are not read; the
    dials a standard attention honours and this cannot are refused, as is a
    sharing layer, since every released layer runs its own indexer.
    """

    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 256
    v_head_dim: int = 256
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    index_kpool: int = 16
    index_kpool_always_select_tail: bool = True

    def build(self, ctx: MixerContext):
        if not ctx.causal:
            raise ValueError(
                "kpool_sparse_attention requires causal=True; its indexer selects among "
                "the keys at or before each query")
        unsupported = {
            "v_norm": ctx.v_norm,
            "k_eq_v": ctx.k_eq_v,
            "sliding_window": ctx.sliding_window,
            "attention_chunk": ctx.attention_chunk,
            "attention_scale": ctx.attention_scale,
            "partial_rotary_factor": ctx.partial_rotary_factor,
            "kv_shared": ctx.kv_shared,
            "output_gate": ctx.output_gate,
        }
        asked = sorted(name for name, value in unsupported.items() if value)
        if asked:
            raise ValueError(
                f"the kpool_sparse_attention mixer has no {', '.join(asked)}: the layer "
                "scales by its nope head dim, rotates nothing, attends its own selection "
                "of the whole sequence and norms its latents, not its values")
        return functools.partial(
            KPoolSparseAttention,
            emb_features=ctx.emb_features,
            num_heads=ctx.num_heads,
            max_seq_len=ctx.max_seq_len,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            v_head_dim=self.v_head_dim,
            index_n_heads=self.index_n_heads,
            index_head_dim=self.index_head_dim,
            index_topk=self.index_topk,
            index_kpool=self.index_kpool,
            index_kpool_always_select_tail=self.index_kpool_always_select_tail,
            norm_eps=ctx.norm_eps,
            scale_offset=ctx.scale_offset,
            scale_after_cast=ctx.scale_after_cast,
            attention_bias=ctx.attention_bias,
            dtype=ctx.dtype,
            precision=ctx.precision,
            attention_impl=ctx.attention_impl,
            force_fp32_for_softmax=ctx.force_fp32_for_softmax)
