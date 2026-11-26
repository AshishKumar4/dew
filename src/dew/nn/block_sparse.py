"""Block-selected attention: each query reads the keys of the key blocks it selects, the attention template's
indexed form (docs/design/kernel-templates.md §5; its §A rows 36, 37, 59, 60 and 64).

A selection is each query's key blocks `[B, R, Q, S]` (-1 where it selects none; `R` a GQA group's or a
query head's), from a scoring pass the model runs in XLA, selecting through
`dew.nn.sparse_selection.top_k_selection`: MoBA's gate over chunk-mean keys (`moba_selection`), NSA's
compressed branch's weights (`nsa_selection`), MiniMax's learned indexer max-pooled to blocks
(`pooled_selection`; MSA's index branch `MsaIndexer`, its selection `msa_selection`, MiniMax M3's at its
forced first and local blocks, and its loss `msa_indexer_kl`), InfLLM-V2's two-window pooling
(`infllm_selection`; its attention, dense below 8192 keys, `infllm_attention`) or a caller's block mask
(`masked_selection`). `dew.nn.sparse_selection.selected_attention` runs it, each selected block one of the
key runs `scaled_dot_product_attention` takes (`key_runs`), which the template's mainloops walk as the
selected tiles alone (`attention_mosaic_program.indexed_walk`): work a query block's selected blocks, not
its keys. A selection takes no gradient."""

import dataclasses
import functools
from collections.abc import Callable
from typing import ClassVar, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name

from dew.nn.attention import AttentionImpl, scaled_dot_product_attention
from dew.nn.dense import Dense
from dew.nn.documents import CrossDocuments, DocumentWindows, document_windows
from dew.nn.inspection import QK
from dew.nn.kv_cache import cached_validity
from dew.nn.mixer_base import MixerContext
from dew.nn.mixers.attention import AttentionMixer, CausalSelfAttention
from dew.nn.mla import SELECTIONS
from dew.nn.precision import float32_einsum
from dew.nn.residuals import Residual
from dew.nn.sparse_selection import SELECTING, block_maxima, block_runs, selected_attention, top_k_selection


def _earlier_blocks(queries: int, blocks: int, block: int) -> jax.Array:
    """`[Q, Kb]`: whether key block j lies wholly before query i's own block (a causal selection's
    candidates)."""
    own = jnp.arange(queries)[:, None] // block
    return jnp.arange(blocks)[None, :] < own


class MobaChunks(NamedTuple):
    """A call's keys in MoBA's chunks (`moba_chunks`): each document's complete windows of `block` of its
    real tokens from its first (`dew.nn.documents.document_windows`), the candidates a query's gate ranks,
    and each query's own chunk, the keys of its document's chunk up to it."""

    windows: DocumentWindows
    rank: jax.Array
    """`[B, Q]`: each query's rank among its document's real keys."""
    document: jax.Array
    """`[B, Q]`: each query's document."""
    own: jax.Array
    """`[B, Q, 2]`: each query's own chunk's run, its first key to the key past the query's own."""


def moba_chunks(keys: int, block: int, *, segments: jax.Array | None = None, valid: jax.Array | None = None,
                slots: jax.Array | None = None, rows: int = 1) -> MobaChunks:
    """A call's `keys` keys in MoBA's chunks of `block` (`MobaChunks`; moba_naive.py's varlen form, each
    sequence chunked from its start): its keys' documents `segments` `[B, K]` (one a row where None), their
    real ones where `valid` `[B, K]` marks them (every one where None), each document a run of the row's
    slots (`dew.nn.inputs.document_starts`); `slots` `[B, Q]` each query's own key, its index where None (a
    call whose queries are its keys)."""
    segments = jnp.zeros((rows, keys), jnp.int32) if segments is None else jnp.asarray(segments)
    rows = segments.shape[0]
    windows = document_windows(segments, valid, block)
    real = jnp.ones((rows, keys), bool) if valid is None else jnp.broadcast_to(jnp.asarray(valid, bool),
                                                                               (rows, keys))
    slots = jnp.broadcast_to(jnp.arange(keys), (rows, keys)) if slots is None else slots
    at = jnp.maximum(slots, 0)
    rank, document = (jnp.take_along_axis(field, at, axis=-1)
                      for field in (windows.query_rank, windows.query_document))
    # Each key's chunk's first real key: the last that opens a chunk at or before it, in its own document,
    # its documents each a run of the row's slots.
    opens = real & (windows.query_rank % block == 0)
    first = jax.lax.cummax(jnp.where(opens, jnp.arange(keys), 0), axis=1)
    start = jnp.take_along_axis(first, at, axis=-1)
    own = jnp.stack([jnp.where(slots >= 0, start, 0), jnp.where(slots >= 0, slots + 1, 0)], axis=-1)
    return MobaChunks(windows, rank, document, own)


def window_runs(windows: DocumentWindows) -> jax.Array:
    """`[B, T, 2]`: each window's run, its first key to the key past its last, empty where it is no
    document's complete window (`DocumentWindows.live`)."""
    first, last = windows.sources[..., 0], windows.sources[..., -1]
    return jnp.stack([jnp.where(windows.live, first, 0), jnp.where(windows.live, last + 1, 0)], axis=-1)


def moba_gate(query: jax.Array, key: jax.Array, chunks: MobaChunks,
              block: int) -> tuple[jax.Array, jax.Array]:
    """MoBA's gate (arXiv 2502.13189; moba_naive.py, which widens both its operands to fp32): each query
    head's query `[B, Q, H, D]` against each complete window's mean key of its group (`key`
    `[B, K, G, D]`), fp32, `[B, H, Q, T]`, and the windows it may rank `[B, Q, T]`: its own document's
    before its own chunk."""
    rows, queries, heads, width = query.shape
    groups = key.shape[2]
    windows = chunks.windows
    flat = key.astype(jnp.float32).reshape(rows, key.shape[1], groups * width)
    means = jnp.mean(windows.gathered(flat), axis=2).reshape(rows, -1, groups, width)
    grouped = query.astype(jnp.float32).reshape(rows, queries, groups, heads // groups, width)
    with jax.named_scope(SELECTING):
        gate = float32_einsum("bqghd,btgd->bghqt", grouped, means).reshape(rows, heads, queries, -1)
    keep = (windows.live[:, None, :] & (windows.document[:, None, :] == chunks.document[..., None])
            & (windows.window[:, None, :] < (chunks.rank // block)[..., None]))
    return jax.lax.stop_gradient(gate), keep


def moba_selection(gate: jax.Array, keep: jax.Array, top_k: int) -> jax.Array:
    """MoBA's selection of `gate` `[B, H, Q, T]` among the windows `keep` `[B, Q, T]` lets each query rank
    (`moba_gate`): the `top_k - 1` highest, beside its own chunk, which moba_naive.py forces into its
    `top_k` (`MobaChunks.own`): `[B, H, Q, top_k - 1]`, -1 where fewer."""
    return top_k_selection(gate, keep[:, None], top_k - 1)


def moba_attention(query: jax.Array, key: jax.Array, value: jax.Array, *, block: int, top_k: int,
                   implementation: AttentionImpl = "auto") -> jax.Array:
    """MoBA's causal attention over one sequence a row: each query head reads its own block up to it and
    the earlier blocks its gate selects (`moba_selection`, `top_k` of them with its own), the template's
    indexed form a head a row (`selected_attention`). `query` `[B, S, H, D]`, `key` and `value`
    `[B, S, G, D]`; `[B, S, H, Dv]` out."""
    chunks = moba_chunks(key.shape[1], block, rows=key.shape[0])
    selected = moba_selection(*moba_gate(query, key, chunks, block), top_k)
    return selected_attention(query, key, value, selected, window_runs(chunks.windows), own=chunks.own,
                              implementation=implementation)


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
    selected_out = selected_attention(query, key, value, selected, block_runs(key.shape[1], block),
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


class MsaIndexer(nn.Module):
    """MiniMax Sparse Attention's index branch (arXiv 2606.13392 §3, eqs. 5, 6, 11): an index query head a GQA
    group (`wq`, `groups` heads of `width`) and one index key head every group shares (`wk`), both of the
    detached hidden states, so the main loss trains neither (its own, `msa_indexer_kl`, trains these alone).
    Its scores `[B, G, N, N]` (fp32) are each query's scaled products with every key, -inf past it (eq. 6's
    `j <= i`); `msa_selection` max-pools them onto blocks and takes each group's top blocks."""

    groups: int
    width: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, hidden: jax.Array) -> jax.Array:
        rows, length, _ = hidden.shape
        hidden = jax.lax.stop_gradient(hidden)
        dense = functools.partial(Dense, use_bias=False, dtype=self.dtype, precision=self.precision)
        query = dense(self.groups * self.width, name="wq", kernel_axes=("embed", "index"))(hidden)
        key = dense(self.width, name="wk", kernel_axes=("embed", "index"))(hidden)
        query = query.reshape(rows, length, self.groups, self.width)
        scores = float32_einsum("bngd,bmd->bgnm", query, key, precision=self.precision) * self.width ** -0.5
        place = jnp.arange(length)
        return jnp.where(place[None, :] <= place[:, None], scores, -jnp.inf)


def msa_selection(index_scores: jax.Array, *, block: int = 128, top_k: int = 16, initial: int = 0,
                  local: int = 1) -> jax.Array:
    """MSA's selection (arXiv 2606.13392 eq. 7, §A row 59) and MiniMax M3's (§A row 37a): index scores
    `[B, G, Q, K]` (-inf past each query) max-pooled onto blocks of `block` (eq. 6), each group's `top_k`
    blocks a query, the first `initial` and the `local` ending at the query's own forced among them:
    `[B, G, Q, top_k]`, -1 a slot past the blocks the query sees. MSA forces its own block alone (blocks of
    128, the top 16); M3 the first block and the two ending at its own, its top 16 counting them, a single
    index head every group shares (`G` 1; the M3 index kernels of github.com/dphnAI/sonar,
    models/minimax_m3/common/ops/index_topk.py: a page's score its tokens' maximum, the forced pages
    prioritized within `topk`)."""
    rows, groups, queries, _ = index_scores.shape
    assert top_k > initial + local, "the selection counts its forced blocks among its top"
    pooled = block_maxima(index_scores, None, block)
    keep = _earlier_blocks(queries, pooled.shape[-1], block) & (pooled > -jnp.inf)
    own = jnp.arange(queries)[:, None] // block
    forced = jnp.concatenate([jnp.broadcast_to(jnp.arange(initial)[None], (queries, initial)),
                              own - jnp.arange(local)[None, ::-1]], axis=-1)
    forced = jnp.broadcast_to(jnp.where(forced >= 0, forced, -1), (rows, groups, *forced.shape))
    return top_k_selection(pooled, keep, top_k - initial - local, forced=forced)


def msa_indexer_kl(index_scores: jax.Array, logits: jax.Array, selected: jax.Array, *,
                   block: int) -> jax.Array:
    """MSA's indexer loss (arXiv 2606.13392 eqs. 9, 10): over each query's visible tokens of its group's
    selected blocks, the KL divergence of the index branch's softmax (`index_scores` `[B, G, Q, K]`) from the
    detached mean of its group's heads' attention softmaxes (`logits` `[B, H, Q, K]`, the main branch's, `H`
    a multiple of `G`), averaged over the queries and groups. `selected` `[B, G, Q, S]` (`msa_selection`'s,
    -1 a slot of none)."""
    rows, groups, queries, keys = index_scores.shape
    per = logits.shape[1] // groups
    place = jnp.arange(keys)
    chosen = jnp.any(selected[..., None] == (place // block), axis=-2)
    tokens = chosen & (place[None, :] <= jnp.arange(queries)[:, None])
    teacher = jax.nn.softmax(jnp.where(tokens[:, :, None], logits.reshape(rows, groups, per, queries, keys
                                                                          ).astype(jnp.float32), -jnp.inf),
                             axis=-1)
    teacher = jax.lax.stop_gradient(jnp.where(tokens, teacher.mean(axis=2), 0.0))
    student = jax.nn.log_softmax(jnp.where(tokens, index_scores.astype(jnp.float32), -jnp.inf), axis=-1)
    divergence = jnp.where(tokens & (teacher > 0), teacher * (jnp.log(jnp.where(teacher > 0, teacher, 1.0))
                                                               - student), 0.0)
    return divergence.sum() / (rows * queries * groups)


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


INFLLM_DENSE_BELOW = 8192
"""The keys below which InfLLM-V2 attends densely (kernel-templates.md §A row 60), the plain causal call: its
short calls' path, a selection of 64 blocks of 64 near every key of them anyway."""


def infllm_attention(query: jax.Array, key: jax.Array, value: jax.Array, *, block: int = 64, top_k: int = 64,
                     kernel: int = 32, stride: int = 16, initial: int = 1, local: int = 1,
                     dense_below: int = INFLLM_DENSE_BELOW, implementation: AttentionImpl = "auto",
                     **options) -> jax.Array:
    """InfLLM-V2's attention (kernel-templates.md §A row 60): a call over fewer than `dense_below` keys the
    plain causal call; a longer one each query's selected blocks (`infllm_selection`: the `top_k` of its
    earlier blocks of `block` by its group's two-window pooled scores, the first `initial` and the `local`
    ending at its own forced) through the template's indexed form (`selected_attention`), reading those
    blocks' keys alone. `query` `[B, Q, G * h, D]`, `key` and `value` `[B, K, G, D]`, a causal call whose
    queries are its keys; `options` the call's other arguments. The selection takes no gradient."""
    keys = key.shape[1]
    if keys < dense_below:
        return scaled_dot_product_attention(query, key, value, causal=True, implementation=implementation,
                                            **options)
    selected = infllm_selection(query, key, block=block, top_k=top_k, kernel=kernel, stride=stride,
                                initial=initial, local=local)
    return selected_attention(query, key, value, selected, block_runs(keys, block),
                              implementation=implementation, **options)


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
    """`CausalSelfAttention` whose each query reads its own chunk and the earlier chunks of its document its
    selection picks (`selection`: MoBA's, `moba_selection`), `top_k` of them with its own, a chunk `block`
    of its document's real tokens from its first (`moba_chunks`, MoBA's varlen chunks), through the
    template's indexed form (`selected_attention`): a selection a query head's, over its group's keys. A
    cached call selects over its cache's keys, a document a row."""

    selection: Literal["moba"] = "moba"
    block: int = 512
    top_k: int = 3

    reads: ClassVar[frozenset[str]] = frozenset()

    def _attended(self, masking, positions, append, sinks, train: bool, *, bias=None):
        """The attention over each query's selected chunks (`selected_attention`), its selection MoBA's over
        the call's keys in each document's chunks (`moba_chunks`): an uncached call's its own tokens, by
        their documents and validity; a cached call's its cache's filled slots (`cached_validity`), its
        queries at the slots they were written to. The kernel reads the call's real keys alone, so a chunk
        whose run spans a key the call hides reads none of it. The gate's selection is sown (`SELECTIONS`)
        for a caller that opens the collection: its scores, the windows each query may take, its budget."""
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
        else:
            # A cache row holds one document (`BlockSparseAttentionMixer.packed_cache`), its filled slots.
            slots = jnp.asarray(append.positions).reshape(query.shape[:2])
            valid, documents = cached_validity(self, keys), None
        valid = None if valid is None else jnp.broadcast_to(jnp.asarray(valid, bool), (rows, keys))
        chunks = moba_chunks(keys, self.block, segments=documents, valid=valid, slots=slots, rows=rows)
        gate, keep = moba_gate(query, key, chunks, self.block)
        if not self.is_initializing():
            visible = jnp.broadcast_to(keep[:, None], gate.shape)
            self.sow(SELECTIONS, "selection", {"scores": gate, "visible": visible,
                                               "budget": jnp.asarray(self.top_k - 1)})
        selected = moba_selection(gate, keep, self.top_k)
        return checkpoint_name(selected_attention(
            query, key, value, selected, window_runs(chunks.windows), own=chunks.own, slots=slots,
            key_valid=valid, dtype=self.ctx.dtype,
            precision=self.ctx.precision, force_fp32_for_softmax=self.ctx.force_fp32_for_softmax,
            wide_logits=self.ctx.wide_logits, softcap=self.ctx.attn_logit_softcap), Residual.CONTEXT)


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
