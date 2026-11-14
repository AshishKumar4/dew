"""Block-selected attention: each query reads the keys of the key blocks it selects, the attention template's
indexed form (docs/design/kernel-templates.md §5; its §A rows 36, 37, 59, 60 and 64).

A selection is each query's key blocks `[B, R, Q, S]` (-1 where it selects none; `R` a GQA group's or a
query head's), from a scoring pass the model runs in XLA, selecting through
`dew.nn.sparse_selection.top_k_selection`: MoBA's gate over chunk-mean keys (`moba_selection`), NSA's
compressed branch's weights (`nsa_selection`), MiniMax's learned indexer max-pooled to blocks
(`pooled_selection`), InfLLM-V2's two-window pooling (`infllm_selection`) or a caller's block mask
(`masked_selection`). `dew.nn.sparse_selection.selected_attention` runs it, each selected block one of the
key runs `scaled_dot_product_attention` takes (`key_runs`), which the template's mainloops walk as the
selected tiles alone (`attention_mosaic_program.indexed_walk`): work a query block's selected blocks, not
its keys. A selection takes no gradient."""

import dataclasses
import functools
from collections.abc import Callable
from typing import ClassVar, Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax.ad_checkpoint import checkpoint_name

from dew.batch import coordinates
from dew.nn.attention import AttentionImpl, scaled_dot_product_attention
from dew.nn.documents import CrossDocuments
from dew.nn.inputs import numbered_blocks
from dew.nn.inspection import QK
from dew.nn.kv_cache import cached_validity
from dew.nn.mixer_base import MixerContext
from dew.nn.mixers.attention import AttentionMixer, CausalSelfAttention
from dew.nn.precision import float32_einsum
from dew.nn.residuals import Residual
from dew.nn.sparse_selection import block_maxima, selected_attention, top_k_selection


def row_blocks(keys: int, block: int) -> jax.Array:
    """`[K]`: each of `keys` keys' block of `block` along one sequence, the chunks a selection of blocks
    names (`dew.nn.sparse_selection.selected_runs`)."""
    return jnp.arange(keys, dtype=jnp.int32) // block


def _earlier_blocks(queries: int, blocks: int, block: int) -> jax.Array:
    """`[Q, Kb]`: whether key block j lies wholly before query i's own block (a causal selection's
    candidates)."""
    own = jnp.arange(queries)[:, None] // block
    return jnp.arange(blocks)[None, :] < own


def moba_selection(query: jax.Array, key: jax.Array, chunks: jax.Array, *, top_k: int, count: int,
                   slots: jax.Array | None = None, documents: jax.Array | None = None,
                   dtype: jnp.dtype | None = None) -> jax.Array:
    """MoBA's selection (arXiv 2502.13189; moba_naive.py, its varlen form chunking each sequence from its
    start): each query head's gate, its query against the mean key of each chunk of its document before its
    own, the `top_k - 1` highest beside its own chunk, which the reference forces into its `top_k`. `query`
    `[B, Q, H, D]`, `key` `[B, K, G, D]` (a head's gate against its group's keys); `chunks` `[B, K]` or `[K]`
    each key's chunk numbered along the row (-1 for none, below `count`: `row_blocks`, or each document's
    blocks of places, `dew.nn.inputs.numbered_blocks`); `slots` `[B, Q]` each query's own key (its index
    where None, -1 for a query with none); `documents` `[B, K]` each key's document (one where None). The
    gate's product takes the attention logits' arithmetic: its operands in `dtype` (the queries' where None,
    fp32 under `wide_logits`), the mean keys formed in fp32, its sums fp32, where moba_naive.py widens both
    operands to fp32 whatever the inputs. A selection of chunks `[B, H, Q, top_k]`, a head's own, -1 where
    fewer."""
    rows, queries, heads, width = query.shape
    keys, groups = key.shape[1], key.shape[2]
    chunks = jnp.broadcast_to(chunks, (rows, keys))
    slots = jnp.broadcast_to(jnp.arange(queries) if slots is None else slots, (rows, queries))
    documents = jnp.zeros((rows, keys), jnp.int32) if documents is None else documents
    numbered = jnp.where(chunks >= 0, chunks, count)
    sums = jax.vmap(lambda each, ids: jax.ops.segment_sum(each, ids, num_segments=count))(
        key.astype(jnp.float32), numbered)
    sizes = jax.vmap(lambda ids: jax.ops.segment_sum(jnp.ones_like(ids), ids, num_segments=count))(numbered)
    means = sums / jnp.maximum(sizes, 1)[..., None, None]
    held = jax.vmap(lambda each, ids: jax.ops.segment_max(each, ids, num_segments=count))(documents, numbered)
    taken = query.dtype if dtype is None else dtype
    grouped = query.reshape(rows, queries, groups, heads // groups, width).astype(taken)
    gate = float32_einsum("bqghd,bcgd->bghqc", grouped, means.astype(taken),
                          preferred_element_type=jnp.float32).reshape(rows, heads, queries, count)
    reached = slots >= 0
    at = jnp.maximum(slots, 0)
    own = jnp.where(reached, jnp.take_along_axis(chunks, at, axis=-1), -1)
    mine = jnp.take_along_axis(documents, at, axis=-1)
    keep = ((jnp.arange(count) < own[..., None]) & (held[:, None, :] == mine[..., None])
            & (sizes[:, None, :] > 0))
    forced = jnp.broadcast_to(own[:, None, :, None], (rows, heads, queries, 1))
    return top_k_selection(jax.lax.stop_gradient(gate), keep[:, None], top_k - 1, forced=forced)


def moba_attention(query: jax.Array, key: jax.Array, value: jax.Array, *, block: int, top_k: int,
                   implementation: AttentionImpl = "auto") -> jax.Array:
    """MoBA's causal attention over one sequence: each query head reads its own block up to it and the
    earlier blocks its gate selects (`moba_selection`, `top_k` of them with its own), the template's indexed
    form a head a row (`selected_attention`). `query` `[B, S, H, D]`, `key` and `value` `[B, S, G, D]`;
    `[B, S, H, Dv]` out."""
    chunks = row_blocks(key.shape[1], block)
    selected = moba_selection(query, key, chunks, top_k=top_k, count=-(-key.shape[1] // block))
    return selected_attention(query, key, value, selected, chunks, own=False, implementation=implementation)


def nsa_selection(compressed_weights: jax.Array, *, stride: int, length: int, block: int, top_k: int,
                  queries: int, initial: int = 1, local: int = 2) -> jax.Array:
    """NSA's selection (arXiv 2502.11089 §3.3.2): a GQA group's compressed branch's attention weights
    `[B, G, h, Q, C]` (its `h` heads' softmax over `C` compressed keys, each of `length` keys every `stride`,
    eq. 8) summed over the group's heads (eq. 10), spread onto key blocks of `block` by eq. 9 (block j's
    score the weights of compressed keys `(block / stride) j + m + n`, `m < block / stride`, `n < length /
    stride`; `stride` divides both), the `top_k` highest of the earlier blocks beside the first `initial`
    and the `local` ending at the query's own, forced (the paper's n counts them): a selection `[B, G, Q,
    top_k + initial + local]`."""
    assert block % stride == 0 and length % stride == 0, "NSA's eq. 9 takes a stride dividing both blocks"
    weights = jax.lax.stop_gradient(compressed_weights).sum(axis=2)
    compressed = weights.shape[-1]
    blocks = -(-queries // block)
    # Eq. 9's spread: how many (m, n) pairs take compressed key c to block j.
    offsets = (np.arange(block // stride)[:, None] + np.arange(length // stride)[None, :]).ravel()
    spread = np.zeros((compressed, blocks), np.float32)
    for j in range(blocks):
        for c in (block // stride) * j + offsets:
            if c < compressed:
                spread[c, j] += 1
    scores = jnp.einsum("bgqc,ck->bgqk", weights, jnp.asarray(spread, weights.dtype))
    own = jnp.arange(queries)[:, None] // block
    forced = jnp.concatenate([jnp.broadcast_to(jnp.arange(initial)[None], (queries, initial)),
                              own - jnp.arange(local)[None, ::-1]], axis=-1)
    forced = jnp.where(forced >= 0, forced, -1)
    return top_k_selection(scores, _earlier_blocks(queries, blocks, block), top_k,
                           forced=jnp.broadcast_to(forced, (*scores.shape[:2], *forced.shape)))


def nsa_attention(query: jax.Array, key: jax.Array, value: jax.Array, compressed_key: jax.Array,
                  compressed_value: jax.Array, gates: jax.Array, *, stride: int, length: int, block: int,
                  top_k: int, window: int, window_key: jax.Array | None = None,
                  window_value: jax.Array | None = None, initial: int = 1, local: int = 2,
                  implementation: AttentionImpl = "auto") -> jax.Array:
    """NSA's attention (arXiv 2502.11089 §3), three branches a GQA group, gated: the compressed branch over
    the model's compressed keys and values `[B, C, G, D]` (compressed key c the keys `[c * stride, c * stride
    + length)`, eq. 7, read by a query that reaches its last), the selected branch over `key` and `value`'s
    blocks its weights select (`nsa_selection`, `selected_attention`) and the sliding branch over a `window`
    of `window_key` and `window_value` (each branch its own keys and values, §3.3.3; the selected branch's
    where not given), combined by `gates` `[B, Q, H, 3]` in XLA (the gate combine a pointwise op,
    kernel-templates.md §A row 36). `query` `[B, Q, G * h, D]`, `key` and `value` `[B, K, G, D]`."""
    rows, queries, heads, width = query.shape
    groups, compressed = key.shape[2], compressed_key.shape[1]
    per = heads // groups
    # The compressed branch's weights (eq. 8), which the selection reads, over the compressed keys whose
    # last key a query reaches.
    reach = jnp.arange(compressed)[None, :] * stride + length <= jnp.arange(queries)[:, None] + 1
    grouped = query.reshape(rows, queries, groups, per, width).astype(jnp.float32)
    logits = jnp.einsum("bqghd,bcgd->bghqc", grouped, compressed_key.astype(jnp.float32)) * width ** -0.5
    logits = jnp.where(reach[None, None, None], logits, -jnp.inf)
    weights = jnp.where(reach[None, None, None], jax.nn.softmax(logits, axis=-1), 0.0)
    compressed_out = jnp.einsum("bghqc,bcgd->bqghd", weights, compressed_value.astype(jnp.float32))
    selected = nsa_selection(weights, stride=stride, length=length, block=block, top_k=top_k, queries=queries,
                             initial=initial, local=local)
    selected_out = selected_attention(query, key, value, selected, row_blocks(key.shape[1], block), own=False,
                                      implementation=implementation)
    sliding_out = scaled_dot_product_attention(
        query, key if window_key is None else window_key, value if window_value is None else window_value,
        causal=True, sliding_window=window, implementation=implementation)
    branches = jnp.stack([compressed_out.reshape(rows, queries, heads, -1).astype(selected_out.dtype),
                          selected_out, sliding_out], axis=-1)
    return jnp.sum(branches * gates[..., None, :].astype(branches.dtype), axis=-1)


def pooled_selection(index_scores: jax.Array, *, block: int, top_k: int, local: int = 1) -> jax.Array:
    """MiniMax Sparse Attention's selection (kernel-templates.md §A row 59): a learned token indexer's scores
    `[B, G, Q, K]` (a GQA group's, causal: -inf past a query) max-pooled to key blocks of `block`, the `top_k`
    highest of the earlier blocks beside the `local` ending at the query's own, forced: `[B, G, Q, top_k +
    local]`."""
    rows, groups, queries, _ = index_scores.shape
    pooled = block_maxima(index_scores, None, block)
    keep = _earlier_blocks(queries, pooled.shape[-1], block) & (pooled > -jnp.inf)
    own = jnp.arange(queries)[:, None] // block
    forced = own - jnp.arange(local)[None, ::-1]
    forced = jnp.broadcast_to(jnp.where(forced >= 0, forced, -1), (rows, groups, *forced.shape))
    return top_k_selection(pooled, keep, top_k, forced=forced)


def infllm_selection(query: jax.Array, key: jax.Array, *, block: int, top_k: int = 64, kernel: int = 32,
                     stride: int = 16, initial: int = 1, local: int = 1) -> jax.Array:
    """InfLLM-V2's selection (arXiv 2509.24663 §3.3): keys mean-pooled by a window of `kernel` every `stride`
    into compressed keys (eq. 4), each head's softmax of its query against the compressed keys whose window
    it reaches whole, at the attention's scale `D ** -0.5` (eq. 5, causal), summed over a GQA group's heads
    (eq. 6), max-pooled onto key blocks of `block` over `block / stride` compressed keys every `block /
    stride` and `kernel / stride - 1` more (eq. 7: 5 every 4 at the paper's 64, 32, 16), the `top_k`
    highest of the earlier blocks beside the first `initial` and the `local` ending at the query's own,
    forced (eqs. 2, 3). `query` `[B, Q, G * h, D]`, `key` `[B, K, G, D]`; `[B, G, Q, top_k + initial +
    local]`. Its pooling is `reduce_window`'s, each window a key's or a score's once."""
    assert block % stride == 0 and kernel % stride == 0, "InfLLM-V2's pooling takes a stride dividing both"
    rows, keys, groups, width = key.shape
    queries = query.shape[1]
    query = query.reshape(rows, queries, groups, query.shape[2] // groups, width)
    blocks = -(-keys // block)
    earlier = _earlier_blocks(queries, blocks, block)
    own = jnp.arange(queries)[:, None] // block
    forced = jnp.concatenate([jnp.broadcast_to(jnp.arange(initial)[None], (queries, initial)),
                              own - jnp.arange(local)[None, ::-1]], axis=-1)
    forced = jnp.broadcast_to(jnp.where(forced >= 0, forced, -1), (rows, groups, queries, initial + local))
    windows = (keys - kernel) // stride + 1 if keys >= kernel else 0
    if windows == 0:
        # No window holds `kernel` keys: the forced blocks alone.
        nothing = jnp.zeros((rows, groups, queries, blocks), bool)
        return top_k_selection(jnp.zeros(nothing.shape), nothing, top_k, forced=forced)
    summed = jax.lax.reduce_window(key.astype(jnp.float32), 0.0, jax.lax.add, (1, kernel, 1, 1),
                                   (1, stride, 1, 1), "VALID")
    pooled = summed / kernel
    logits = jnp.einsum("bqghd,bwgd->bghqw", query.astype(jnp.float32), pooled) * width ** -0.5
    # A query reaches a window whose last key is at or before its own.
    reach = jnp.arange(windows)[None, :] * stride + kernel <= jnp.arange(queries)[:, None] + 1
    logits = jnp.where(reach, logits, -jnp.inf)
    weights = jnp.where(reach, jax.nn.softmax(logits, axis=-1), 0.0).sum(axis=2)
    step = block // stride
    span = step + kernel // stride - 1
    padded = jnp.pad(weights, ((0, 0), (0, 0), (0, 0), (0, max(0, (blocks - 1) * step + span - windows))))
    scores = jax.lax.reduce_window(padded, -jnp.inf, jax.lax.max, (1, 1, 1, span), (1, 1, 1, step),
                                   "VALID")[..., :blocks]
    return top_k_selection(scores, earlier, top_k, forced=forced)


def masked_selection(mask: jax.Array, block_q: int, *, queries: int, most: int) -> jax.Array:
    """A caller's block mask `[B, Qb, Kb]` (FlagAttention's Hy3 block-mask prefill, kernel-templates.md §A
    row 64) as a selection `[B, Q, most]`: each query its query block's selected key blocks in order, at
    most `most` (static: the mask's density bound, the scheduler's bucket), -1 after."""
    blocks = mask.shape[-1]
    order = jnp.where(mask, jnp.arange(blocks)[None, None], blocks)
    chosen = jnp.sort(order, axis=-1)[..., :most]
    chosen = jnp.where(chosen < blocks, chosen, -1).astype(jnp.int32)
    return jnp.repeat(chosen, block_q, axis=1)[:, :queries]


class BlockSparseSelfAttention(CausalSelfAttention):
    """`CausalSelfAttention` whose each query reads its own block and the earlier blocks of its document its
    selection picks (`selection`: MoBA's, `moba_selection`), `top_k` of them with its own, a block `block`
    places of its document from its first (`numbered_blocks`, MoBA's varlen chunks), through the template's
    indexed form (`selected_attention`): a selection a query head's, over its group's keys. A cached call
    selects over its cache's keys at the places the cache keeps (`Append.places`), a document a row."""

    selection: Literal["moba"] = "moba"
    block: int = 512
    top_k: int = 3

    reads: ClassVar[frozenset[str]] = frozenset()

    @property
    def _reads_places(self) -> bool:
        # Its blocks count each document's places, which its cache keeps beside its keys.
        return True

    def _attended(self, masking, positions, append, sinks, train: bool, *, bias=None):
        """The attention over each query's selected blocks (`selected_attention`), its selection MoBA's over
        the call's keys numbered into each document's blocks (`numbered_blocks`): an uncached call's its own
        tokens, by their documents and validity; a cached call's its cache's filled slots
        (`cached_validity`), its queries at the slots they were written to."""
        del positions, sinks, train
        query, key, value = masking.query, masking.key, masking.value
        rows, keys = key.shape[:2]
        groups = masking.groups
        if (bias is not None or masking.runs is not None or masking.window is not None
                or groups.block_groups is not None or groups.image_groups is not None
                or masking.query_slots is not None
                or (append is None and (masking.mask is not None or keys != query.shape[1]))):
            raise ValueError(f"{self.name}: block-selected attention reads its causal blocks alone, no bias, "
                             "key runs, window, groups, mask or prefix of a call's")
        if not self.is_initializing() and self.is_mutable_collection(QK):
            raise ValueError(f"{self.name}: QK-Clip's logit maxima are a dense call's, which block-selected "
                             "attention forms none of")
        if append is None:
            documents = masking.documents
            if isinstance(documents, CrossDocuments):
                raise ValueError(f"{self.name}: block-selected attention's keys are its own tokens, their "
                                 "documents the queries'")
            slots, valid = None, masking.valid
            most = 1 if documents is None else masking.max_segments
        else:
            # A cache row holds one document (`BlockSparseAttentionMixer.packed_cache`), its filled slots.
            slots = jnp.asarray(append.positions).reshape(query.shape[:2])
            valid, documents, most = cached_validity(self, keys), None, 1
        places = jnp.broadcast_to(masking.key_places, (rows, keys))
        chunks = numbered_blocks(coordinates((rows, keys), segments=documents, valid=valid, positions=places),
                                 (rows, keys), self.block)
        # A row's chunks: each document's blocks, a partial one at either end, and its padding's.
        count = keys if most is None else min(keys, keys // self.block + 4 * (most + 1))
        # The gate's product widened where the layer's logits are (`moba_selection`'s `dtype`).
        widened = jnp.dtype(jnp.float32) if self.ctx.wide_logits else None
        selected = moba_selection(query, key, chunks, top_k=self.top_k, count=count, slots=slots,
                                  documents=documents, dtype=widened)
        return checkpoint_name(selected_attention(
            query, key, value, selected, chunks, slots=slots, own=False,
            implementation=self.ctx.attention_impl, dtype=self.ctx.dtype, precision=self.ctx.precision,
            force_fp32_for_softmax=self.ctx.force_fp32_for_softmax, wide_logits=self.ctx.wide_logits,
            softcap=self.ctx.attn_logit_softcap), Residual.CONTEXT)


@dataclasses.dataclass(frozen=True)
class BlockSparseAttentionMixer(AttentionMixer):
    """Block-selected attention (`BlockSparseSelfAttention`): each query reads its own block and the earlier
    blocks its `selection` picks, `top_k` with its own, blocks of `block` places of its document; MoBA's
    (arXiv 2502.13189, whose experiments take blocks of 512 and the top 3). Its selection reads the layer's
    own queries and keys, no parameter of its own, and a cached call's its cache's keys again, so it keeps no
    state beside them."""

    selection: Literal["moba"] = "moba"
    block: int = 512
    top_k: int = 3

    # It selects over each row's cache whole, one document a row.
    packed_cache = False
    reads_forks = False
    drops_attention = False

    def mixed_refusal(self, ctx: MixerContext) -> str | None:
        return ("block-selected attention reads each query's blocks as key runs, which a mixed serving step "
                "lays none of")

    def build(self, ctx: MixerContext) -> Callable[..., nn.Module]:
        if not ctx.causal:
            raise ValueError("block_sparse_attention requires causal=True: a query selects the blocks before "
                             "its own")
        if self.block < 1 or self.top_k < 1:
            raise ValueError(f"block_sparse_attention takes a positive block and top_k, got {self.block} and "
                             f"{self.top_k}")
        if self.alibi or self.bidirectional_images or self.bidirectional_blocks:
            raise ValueError("block_sparse_attention adds no ALiBi bias and reads no image or block whole")
        ctx.refuse("block_sparse_attention", {
            "attention_sinks": ctx.attention_sinks, "attention_dropout_rate": ctx.attention_dropout_rate,
            "sliding_window": ctx.sliding_window, "attention_chunk": ctx.attention_chunk,
            "kv_shared": ctx.kv_shared,
        }, "each query head's selection is a row of its call, whose sinks and dropout are a head's "
           "across rows, and it reads its document's blocks from their start, no window, chunk or other "
           "layer's keys")
        kind = {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}
        return functools.partial(BlockSparseSelfAttention, ctx=ctx, **kind)
