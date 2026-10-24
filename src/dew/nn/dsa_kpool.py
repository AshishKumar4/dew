"""GLM-5.3-Flash's sparse attention: NoPE latent attention under a k-pool indexer.

The reference is transformers 5.16.1 models/glm5_next/modeling_glm5_next.py
(`Glm5NextTextAttention`, 1064-1256; `Glm5NextTextIndexer`, 736-1024). The
attention is DeepSeek V3.2's (`dew.nn.mla`) with no rope head: the released
`qk_rope_head_dim` is 0 and the reference refuses anything else
(configuration_glm5_next.py:225-228), so positions enter only through
causality.

The indexer scores pools of `index_kpool` consecutive keys, each pool's key a
softmax average of its members' keys under learned gate scores plus a
per-slot term. Each query attends every token of its top `index_topk //
index_kpool` pools, plus the incomplete frontier pool with
`index_kpool_always_select_tail`. Pools count from a row's first real key,
so left padding pools like the unpadded tokens. The selection is detached,
as the reference's `torch.no_grad` indices are.

Decode caches the expanded keys and values and the packed indexer state
`[k | gate_scores | valid]` per slot in MLA's fixed-capacity sparse cache; a
zero valid channel keeps unused slots out of the candidates. Opted-in
prediction caches also keep the selected token list and its origin: extend
publishes the last valid query's, draft reuses it, ordinary steps recompute
and invalidate it, and an origin of -1 means no seed.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import ClassVar

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.dtypes import promote_dtype
from flax.typing import Dtype, PrecisionLike

from dew.batch import valid_order

from .attention import LayerNorm
from .inputs import MixerCall, PredictionPhase, coordinates_of
from .mixer_base import ATTENTION_CONTEXT_RESIDUALS, MixerBase, MixerContext
from .mla import INDEXER, IndexerFront, MultiHeadLatentAttention, latent_refusals
from .precision import at_least_fp32, float32_einsum
from .sparse_selection import top_k_selection


def _real_slots(valid, count: int):
    """The physical slot of each row's first `count` real tokens in order, the
    order the cache compacts them in (`valid_order`), -1 past a row's real
    tokens. A left-padded row's are its first valid slot onward
    (modeling_glm5_next.py:940-944); a row with a hole skips it, as its
    cached prefill does."""
    order, real = valid_order(valid)
    compact = jnp.arange(count)
    slots = order[:, jnp.minimum(compact, valid.shape[-1] - 1)]
    return jnp.where(compact[None] < real[:, None], slots, -1)


# wq_b, wk, k_norm and weights_proj carry the V3.2 indexer's declarations
# under the same names; the pool compression's two tables have no side worth
# naming and take the shape heuristic.
class KPoolIndexer(IndexerFront, nn.Module):
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
        dense = self.declare_front()
        self.wk = dense(self.head_dim, name='wk')
        # A LayerNorm with weight and bias at a hardcoded 1e-6 (modeling_glm5_next.py:763).
        self.k_norm = LayerNorm(epsilon=1e-6, dtype=self.dtype, name='k_norm')
        self.index_kpool_compress_ape = self.param(
            'index_kpool_compress_ape', nn.initializers.zeros, (self.kpool, self.head_dim), jnp.float32)
        self.index_kpool_compress_gate = self.param(
            "index_kpool_compress_gate",
            nn.initializers.zeros,
            (self.head_dim, self.emb_features),
            jnp.float32,
        )

    def packed(self, x, valid):
        """`[B, S, 2 * head_dim + 1]`: the indexer's keys, the pool gate scores
        and the validity channel (modeling_glm5_next.py:797-803)."""
        x = jax.lax.stop_gradient(x)
        keys = self.k_norm(self.wk(x))
        x, gate = promote_dtype(x, self.index_kpool_compress_gate, dtype=self.dtype)
        gate_scores = float32_einsum('bsh,dh->bsd', x, gate, precision=self.precision)
        return jnp.concatenate(
            [keys, gate_scores.astype(keys.dtype), valid.astype(keys.dtype)[..., None]], axis=-1
        )

    def _pools(self, packed):
        """The pools of a packed state, `[B, P]` of them: their keys
        `[B, P, head_dim]`, their token indices `[B, P, kpool]` (-1 on a slot
        outside the row) and whether every slot is valid
        (`get_pooled_states`, modeling_glm5_next.py:899-967)."""
        keys, gate_scores, valid = jnp.split(packed, [self.head_dim, 2 * self.head_dim], axis=-1)
        valid = valid[..., 0] > 0
        batch, total = valid.shape
        pools = -(-total // self.kpool)
        indices = _real_slots(valid, pools * self.kpool)
        safe = jnp.maximum(indices, 0)
        grouped_keys = jnp.take_along_axis(keys, safe[..., None], axis=1).reshape(
            batch, pools, self.kpool, self.head_dim)
        grouped_gate = jnp.take_along_axis(gate_scores, safe[..., None], axis=1).reshape(
            batch, pools, self.kpool, self.head_dim)
        indices = indices.reshape(batch, pools, self.kpool)
        grouped_valid = indices >= 0
        pool_valid = jnp.all(grouped_valid, axis=-1)
        wide = at_least_fp32(grouped_gate.dtype)
        logits = grouped_gate.astype(wide) + self.index_kpool_compress_ape.astype(wide)
        logits = jnp.where(grouped_valid[..., None], logits, -jnp.inf)
        # A pool with no valid slot softmaxes to nan, which the reference
        # zeroes (modeling_glm5_next.py:964-966).
        probabilities = jnp.nan_to_num(jax.nn.softmax(logits, axis=2)).astype(keys.dtype)
        return jnp.sum(probabilities * grouped_keys, axis=2), indices, pool_valid

    def _tail(self, visible, valid):
        """Each query's incomplete pool as token indices, `[B, S, kpool - 1]`,
        -1 where there is none (`append_visible_tail`,
        modeling_glm5_next.py:974-1024): the `visible_count % kpool` real
        tokens after the last complete pool the query sees, in the pools'
        order (`_real_slots`)."""
        batch, queries, total = visible.shape
        slots = _real_slots(valid, total)
        visible_count = jnp.sum(visible, axis=-1, dtype=jnp.int32)
        tail_count = visible_count % self.kpool
        offsets = jnp.arange(self.kpool - 1)
        compact = (visible_count - tail_count)[..., None] + offsets
        tail = jnp.take_along_axis(jnp.broadcast_to(slots[:, None], (batch, queries, total)),
                                   jnp.minimum(compact, total - 1), axis=-1)
        allowed = jnp.logical_and(offsets < tail_count[..., None], compact < total)
        tail_visible = jnp.take_along_axis(visible, jnp.maximum(tail, 0), axis=-1)
        return jnp.where(allowed & (tail >= 0) & tail_visible, tail, -1)

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
        pool_keys, pool_indices, pool_valid = self._pools(packed)
        scores = self.scored(self.queries(q_resid), pool_keys, x)
        # A pool is a candidate iff the query sees its last token
        # (modeling_glm5_next.py:832-839); its indices are clamped only so
        # the gather is legal, the validity test excludes it on its own.
        pool_end = jnp.clip(pool_indices[..., -1], 0, total - 1)
        pool_visible = jnp.take_along_axis(
            visible, jnp.broadcast_to(pool_end[:, None, :], (batch, length, pool_end.shape[-1])), axis=-1)
        candidates = jnp.logical_and(pool_visible, pool_valid[:, None])
        # The reference drops the pools no row fills (:969-972) before its
        # ranking; a dropped pool is never a candidate, and a selected
        # non-candidate contributes no indices, so the selection is the same.
        chosen = top_k_selection(scores, candidates, self.top_k // self.kpool)
        chosen_indices = pool_indices[jnp.arange(batch)[:, None, None], jnp.maximum(chosen, 0)]
        tokens = jnp.where(chosen[..., None] >= 0, chosen_indices, -1).reshape(batch, length, -1)
        if self.always_select_tail and self.kpool > 1:
            tokens = jnp.concatenate([tokens, self._tail(visible, valid)], axis=-1)
        return tokens


# The latent projections and norms carry MLA's declarations under the same names.
class KPoolSparseAttention(MultiHeadLatentAttention):
    """GLM-5.3-Flash's `deepseek_sparse_attention` layer
    (`Glm5NextTextAttention`, modeling_glm5_next.py:1064-1256): the latent
    attention (`MultiHeadLatentAttention`) with no rope head
    (`qk_rope_head_dim=0`, `rotary=False`), its projections, latent norms,
    expansion and output, under the k-pool indexer's selection.

    Each query attends its `KPoolIndexer` selection through an fp32 softmax
    over the reference's eager boolean mask; the indexer folds causality and
    row validity in, and an invalid query's selection is blank
    (modeling_glm5_next.py:801, 875). The logits scale by `qk_nope_head_dim`,
    the whole query width.

    decode=True runs against a `max_seq_len`-slot cache of expanded keys,
    values and packed indexer state, causality read off the slots. Packed
    documents have no reference form and are refused. The latent norms are the
    model's RMSNorm at its epsilon (modeling_glm5_next.py:1103, 1116). The
    selection is a materialized mask, so outside decode 'auto' and 'cudnn' run
    xla.
    """

    reads: ClassVar[frozenset[str]] = frozenset()
    """None of the call's `MIXING` fields (`dew.nn.inputs.MixerCall`): its pools read each document's own
    consecutive keys, with no mask or bias to narrow them."""

    rotary: bool = False
    norm_eps: float = 1e-5
    """The latent norms' epsilon, the model's rms_norm_eps (configuration_glm5_next.py's 1e-5)."""
    index_kpool: int = 16
    index_kpool_always_select_tail: bool = True

    def setup(self):
        if self.qk_rope_head_dim or self.rotary:
            raise ValueError("the k-pool layer has no rope head: qk_rope_head_dim=0 and rotary=False")
        if not self.causal or self.yarn is not None or self.index_shared:
            raise ValueError(
                "the k-pool layer attends causally, with no rope to scale and an indexer of its own: "
                "causal=True, yarn=None and index_shared=False")
        heads, width = self.index_n_heads, self.index_head_dim
        if self.q_lora_rank is None or heads is None or width is None or self.index_topk is None:
            raise ValueError(
                "the k-pool layer's queries are low rank and its indexer selects: q_lora_rank, "
                "index_n_heads, index_head_dim and index_topk have to be set")
        if heads < 1 or width < 1:
            raise ValueError(f"the indexer needs heads of a positive width, got {heads} of {width}")
        super().setup()

    def make_indexer(self) -> nn.Module:
        """The k-pool indexer, by GLM-5.3-Flash's fields (`setup` refused any unset)."""
        heads, width, top_k = self.index_n_heads, self.index_head_dim, self.index_topk
        assert heads is not None and width is not None and top_k is not None
        return KPoolIndexer(
            emb_features=self.emb_features, n_heads=heads, head_dim=width, top_k=top_k,
            kpool=self.index_kpool, always_select_tail=self.index_kpool_always_select_tail,
            dtype=self.dtype, precision=self.precision, name=INDEXER)

    def _check_call(self, call: MixerCall, batch: int, length: int) -> None:
        if coordinates_of(call.metadata, batch, length).segments is not None:
            raise ValueError(
                "kpool_sparse_attention pools one row's consecutive keys, which a packed "
                "row of documents has no reference form for; pass one document per row")
        valid = None if call.metadata is None else call.metadata.valid
        if valid is not None and valid.shape != (batch, length):
            raise ValueError(f"row validity must be {(batch, length)}, got {valid.shape}")

    def _index_keys(self, x, valid):
        return self.indexer.packed(x, valid)

    def _rotated_index_keys(self, keys, freqs_cos, freqs_sin):
        return keys

    def _attends_sparsely(self, selection, total: int) -> bool:
        return False

    def _selection(self, x, q_resid, index_keys, keep, freqs_cos, freqs_sin, call, row_valid, *,
                    slots: jax.Array, allocated: bool = False, committed=None,
                    prediction_phase: PredictionPhase = "ordinary"):
        """Select pools, retaining the complete token list for opted-in draft reuse."""
        if not call.decode or (prediction_phase == "ordinary"
                               and not self.has_variable("cache", "selection_position")):
            return self.indexer.select_indices(x, q_resid, index_keys, keep), None
        # SGLang 97c6978 index_topk_share.py:22-28,48-64 carries the complete token list;
        # a draft hit must not append the new query's tail.
        batch, length = x.shape[:2]
        if prediction_phase == "draft" and length != 1:
            raise ValueError("draft index reuse accepts one candidate per row")
        pools = (self.index_topk or 0) // self.index_kpool
        count = min(pools, -(-self.max_seq_len // self.index_kpool))
        width = count * self.index_kpool + (
            self.index_kpool - 1 if self.index_kpool_always_select_tail else 0
        )
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
                lambda module: module.indexer.select_indices(x, q_resid, index_keys, keep), self)
            indices = jnp.where(hit[:, None, None], frozen, indices)
        else:
            indices = self.indexer.select_indices(x, q_resid, index_keys, keep)
        if allocated:
            if prediction_phase == "ordinary":
                origin.value = jnp.where(active, -1, origin.value)
            else:
                last = jnp.max(jnp.where(row_valid, jnp.arange(length), 0), axis=1)
                publish = active & ~hit
                saved.value = jnp.where(publish[:, None], indices[jnp.arange(batch), last], saved.value)
                origin.value = jnp.where(publish, slots[jnp.arange(batch), last], origin.value)
        return indices, None

    def _empty_selection(self, selected, row_valid):
        # A valid query that selects no pool, one of the first tokens before
        # a pool completes where no tail is always selected, reads every key
        # alike in the reference, whose masked logits are all equal there;
        # attention itself reads nothing for a row that sees no key.
        return jnp.logical_and(row_valid, ~jnp.any(selected, -1))[:, :, None, None]


@dataclasses.dataclass(frozen=True)
class KPoolSparseAttentionMixer(MixerBase):
    """The `kpool_sparse_attention` kind, by GLM-5.3-Flash's config fields
    (configuration_glm5_next.py:112-116, 134-136, 153-154). It has no
    `qk_rope_head_dim` (NoPE by construction, :225-228), so a config naming one
    is refused as an unknown field; the latent norms take the model's
    `norm_eps`. The context's GQA geometry, rotary base and `qk_norm` are not
    read, and sharing layers are refused: every released layer has its own
    indexer.
    """

    residuals = ATTENTION_CONTEXT_RESIDUALS

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
        ctx.refuse("kpool_sparse_attention", {
            **latent_refusals(ctx),
            "force_fp32_for_softmax": False,  # forwarded
            "kv_shared": ctx.kv_shared,
        }, "the layer scales by its nope head dim, rotates nothing, attends its own selection of the "
           "whole sequence and norms its latents, not its values")
        return functools.partial(
            KPoolSparseAttention,
            emb_features=ctx.emb_features,
            num_heads=ctx.num_heads,
            max_seq_len=ctx.max_seq_len,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=0,
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
            output_gate=ctx.output_gate,
            dtype=ctx.dtype,
            precision=ctx.precision,
            attention_impl=ctx.attention_impl,
            force_fp32_for_softmax=ctx.force_fp32_for_softmax)
