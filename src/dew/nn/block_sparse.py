"""Block-selected attention: each query reads the keys of the key blocks it selects, the attention template's
indexed form (docs/design/kernel-templates.md §5; its §A rows 36, 37, 59, 60 and 64).

A selection is each query's key blocks `[B, R, Q, S]` (-1 where it selects none; `R` a GQA group's or a
query head's), from a scoring pass the model runs in XLA, selecting through
`dew.nn.sparse_selection.top_k_selection`: MoBA's gate over chunk-mean keys (`moba_selection`), NSA's
compressed branch's weights (`nsa_selection`), MiniMax's learned indexer max-pooled to blocks
(`pooled_selection`; MSA's index branch `MsaIndexer`, its selection `msa_selection`, MiniMax M3's at its
forced first and local blocks, and its loss `msa_query_kls`), InfLLM-V2's two-window pooling
(`infllm_selection`; its attention, dense below 8192 keys, `infllm_attention`) or a caller's block mask
(`masked_selection`). `dew.nn.sparse_selection.selected_attention` runs it, each selected block one of the
key runs `scaled_dot_product_attention` takes (`key_runs`), which the template's mainloops walk as the
selected tiles alone (`attention_mosaic_program.indexed_walk`): work a query block's selected blocks, not
its keys. A selection takes no gradient."""

import dataclasses
import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.dtypes import promote_dtype
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name

from dew.nn.attention import (
    AttentionImpl,
    attention_implementation,
    reference_scores,
    scaled_dot_product_attention,
    score_precision,
)
from dew.nn.dense import Dense
from dew.nn.documents import CrossDocuments, DocumentWindows, document_windows
from dew.nn.inspection import INDEXER_COLLECTION, QK
from dew.nn.kernels.templates import attention_route
from dew.nn.kv_cache import cached_validity
from dew.nn.mixer_base import MixerContext
from dew.nn.mixers.attention import AttentionMixer, CausalSelfAttention
from dew.nn.mla import indexer_log_target, sow_selection
from dew.nn.precision import at_least_fp32, compensated_sum, float32_einsum
from dew.nn.residuals import Residual
from dew.nn.sparse_selection import (
    SELECTING,
    block_maxima,
    block_runs,
    indexed_attention,
    indexed_call,
    indexed_route,
    selected_attention,
    selection_mask,
    top_k_selection,
)

if TYPE_CHECKING:
    from dew.nn.protocols import IndexedMixer


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
    detached hidden states, so the main loss trains neither (its own, `msa_query_kls`, trains these alone).
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


def msa_candidates(index_scores: jax.Array, *, block: int = 128, initial: int = 0, local: int = 1
                   ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """MSA's blocks as its selection ranks them (`msa_selection`): index scores `[B, G, Q, K]` max-pooled
    onto blocks of `block` (eq. 6), `[B, G, Q, Kb]`; the earlier blocks each query reaches by a finite
    score, which it takes by their scores, `[B, G, Q, Kb]`; and the blocks it forces, `[Q, Kb]` (the first
    `initial` and the `local` ending at its own, each once, those it reaches alone: up to its own), ranked
    ahead of every score."""
    pooled = block_maxima(index_scores, None, block)
    blocks = pooled.shape[-1]
    queries = pooled.shape[-2]
    keep = _earlier_blocks(queries, blocks, block) & (pooled > -jnp.inf)
    own = jnp.arange(queries)[:, None] // block
    index = jnp.arange(blocks)[None, :]
    forced = ((index < initial) | ((index > own - local) & (index <= own))) & (index <= own)
    return pooled, keep & ~forced, forced


def msa_selection(index_scores: jax.Array, *, block: int = 128, top_k: int = 16, initial: int = 0,
                  local: int = 1) -> jax.Array:
    """MSA's selection (arXiv 2606.13392 eq. 7, §A row 59) and MiniMax M3's (§A row 37a): index scores
    `[B, G, Q, K]` (-inf past each query) max-pooled onto blocks of `block` (eq. 6), each group's `top_k`
    blocks a query, the first `initial` and the `local` ending at the query's own forced among them (those
    it reaches): `[B, G, Q, top_k]` whatever its blocks, -1 a slot past the blocks the query sees. MSA
    forces its own block alone (blocks of 128, the top 16); M3 the first block and the two ending at its own,
    its top 16 counting them, a single index head every group shares (`G` 1; the M3 index kernels of
    github.com/dphnAI/sonar, models/minimax_m3/common/ops/index_topk.py: a page's score its tokens' maximum,
    the forced pages prioritized within `topk`)."""
    assert top_k >= 1, "a selection takes a block"
    return msa_selected(*msa_candidates(index_scores, block=block, initial=initial, local=local), top_k=top_k)


def msa_selected(pooled: jax.Array, keep: jax.Array, forced: jax.Array, *, top_k: int) -> jax.Array:
    """`msa_selection` of its candidates (`msa_candidates`): each query's forced blocks first, then the
    `top_k` of the rest by their pooled scores, `[B, G, Q, top_k]`, -1 past those it holds."""
    blocks = pooled.shape[-1]
    # Ranked ahead of every other visible block within the one `top_k`, as M3's index kernels prioritize
    # them: the forced first (lower blocks first), then the rest's top by score (`top_k_selection`), the
    # forced bit ahead of any score, an infinite one too.
    # The forced list is the call's geometry alone: ranked once `[Q, slots]`, every row and group its own.
    slots = min(top_k, blocks)
    first = jax.lax.top_k(forced.astype(jnp.int32), slots, is_stable=True)[1]
    first = jnp.where(jnp.take_along_axis(forced, first, axis=-1), first, -1)
    ordinary = top_k_selection(pooled, keep, slots)
    both = jnp.concatenate([jnp.broadcast_to(first, ordinary.shape), ordinary], axis=-1)
    chosen = jnp.take_along_axis(both, jnp.argsort(both < 0, axis=-1, stable=True)[..., :slots], axis=-1)
    # Its `top_k` slots whatever its blocks: -1 past those it holds.
    return jnp.pad(chosen, ((0, 0),) * 3 + ((0, top_k - chosen.shape[-1]),), constant_values=-1)


def _selected_tokens(selected: jax.Array, keys: int, block: int) -> jax.Array:
    """`[B, G, Q, K]`: whether a query's selection (`selected` `[B, G, Q, S]`, blocks of `block`, -1 none)
    names a key's block and the key is no later than the query: the blocks' mask
    (`sparse_selection.selection_mask`, the groups folded into the batch) widened to their keys."""
    batch, groups, queries, _ = selected.shape
    blocks = selection_mask(selected.reshape(batch * groups, queries, selected.shape[-1]), -(-keys // block))
    chosen = jnp.repeat(blocks, block, axis=-1)[..., :keys].reshape(batch, groups, queries, keys)
    return chosen & (jnp.arange(keys)[None, :] <= jnp.arange(queries)[:, None])


def msa_scores(dtype, precision, *, wide_logits: bool = False,
               route: AttentionImpl = 'reference') -> Callable:
    """The logits MSA's attention forms of one head's query `[B, S, D]` and keys `[B, T, D]` (`_msa`, its
    selected attention in jax.nn's form, `scale_logits`: each logit the fp32 sum of its product, then the
    scale) on the `route` its attention takes (`sparse_selection.indexed_route`): its teacher's
    (`msa_query_kls`), so the indexer trains toward the weights the attention gives (a teacher scaling its
    query first saw bf16's 0.70703125 where the attention scaled after, 0.70710677, train 1368's F1). On
    the template's walk, as that walk weighs them, its policy the template's own
    (`attention_route.softmax_logits`: the fast fold, its rewalk past its reach, or the accurate form's raw
    differences, over the keys the row weighs, `allowed`; train 1440's review); on any other, as the
    reference path forms them (`attention.reference_scores`, a logit at a time range-safe), at the call's
    `dtype`, logit width and `score_precision` of its precision, its fp32 sums (a bf16 teacher's 4096 and
    4097 kept apart, audit 2026-11-27 P01)."""
    def scores(query: jax.Array, key: jax.Array, allowed: jax.Array) -> jax.Array:
        query, key = promote_dtype(query, key, dtype=dtype)
        if route != 'template':
            return reference_scores(query, key, 'bsd,btd->bst', precision=score_precision(precision),
                                    wide_logits=wide_logits, scale_logits=True)
        return attention_route.softmax_logits(query, key, allowed, 'bsd,btd->bst')

    return scores


def msa_query_kls(index_scores: jax.Array, query: jax.Array, key: jax.Array, selected: jax.Array, *,
                  block: int, scale: float, teacher: Callable | None = None) -> jax.Array:
    """MSA's indexer loss per query (arXiv 2606.13392 eqs. 9, 10), `[B, Q]`, as latent attention's indexer
    objective takes it (the objective's `IndexerTraining`): each group's KL of the index branch's softmax over
    the query's selected tokens from its heads' detached attention there (`mla.indexer_log_target`: their
    softmaxes summed, normalized), averaged over the groups. Its logarithms are taken whole, no epsilon
    (MaxText's, `mla.indexer_kl`'s, caps a token the branch scores far below the attention's at -log(1e-8)
    and stills its gradient), over the selected tokens alone, a token the teacher gives no mass adding
    nothing. `query` `[B, Q, G * h, D]`, `key` `[B, K, G, D]`, `index_scores` `[B, G, Q, K]`. Its teacher's
    logits are `teacher`'s (`mla.indexer_log_target`'s `scores`; the mixer's: `msa_scores`, its attention's
    own)."""
    groups, keys = index_scores.shape[1], index_scores.shape[3]
    per = query.shape[2] // groups
    tokens = _selected_tokens(selected, keys, block)

    def kl(group: int) -> jax.Array:
        keep = tokens[:, group]
        wide = at_least_fp32(jnp.result_type(index_scores, query, key))
        # The teacher's mass in its logarithm (`mla.indexer_log_target`), normalized there: a key whose mass
        # is below fp32's least normal keeps it for its product with the branch's gap below (a teacher of
        # logits 0 and -110 against scores 0 and -2^126 weighs the second key's gap 1.44e-10, where its
        # mass rounded to 0 first weighed it nothing, train 1345's F2).
        logged = jnp.where(keep, indexer_log_target(query[:, :, group * per:(group + 1) * per],
                                                    jnp.repeat(key[:, :, group:group + 1], per, axis=2), keep,
                                                    scale, dtype=wide, scores=teacher), -jnp.inf)
        top = jnp.max(logged, axis=-1, keepdims=True)
        top = jnp.where(jnp.isfinite(top), top, 0)
        shares = jnp.exp(logged - top)
        total = compensated_sum(shares, (shares.ndim - 1,))[..., None]
        # A row of no mass none; a NaN total (an invalid teacher) propagates.
        log_target = jnp.where(total == 0, -jnp.inf, logged - top - jnp.log(jnp.where(total == 0, 1, total)))
        # A key of the teacher's mass, a NaN's too (a teacher's invalid logit surfaces, never read as none).
        taken = keep & (log_target != -jnp.inf)
        target = jnp.exp(log_target)
        # KL = sum t log t + sum t (m - s) + log sum exp(s - m) over the selected keys alone, m their peak:
        # each gap m - s taken before its weight, none rounded away by a weighted mean, and one past fp32's
        # range weighted first (t m - t s: [fp32's max, its least] at a uniform teacher is fp32's max -
        # log 2, finite); a key it does not select adds nothing, its score however low.
        scores = jnp.where(keep, index_scores[:, group].astype(wide), -jnp.inf)
        peak = jax.lax.stop_gradient(jnp.max(scores, axis=-1, keepdims=True))
        peak = jnp.where(jnp.isfinite(peak), peak, 0)
        spread = jnp.log(jnp.sum(jnp.where(keep, jnp.exp(scores - peak), 0), axis=-1))
        # Each term at its share of the groups' mean, taken as its product and then divided where the
        # product holds in fp32, its gap divided first where the product passes fp32's range, and its peak's
        # and score's shares apart where the gap itself does: no group's own KL past fp32's range is rounded
        # before the mean. A weight below fp32's least normal meets them in its quarters instead (`scaled`),
        # so a term its rounded weight would flush keeps its value (a teacher of logits 0 and -110 against
        # scores 0 and -2^126: 1.44e-10, train 1345's F2).
        held = jnp.where(taken, scores, 0)
        gap = peak - held
        tiny = log_target < jnp.log(jnp.finfo(wide).tiny)

        quarter = jnp.exp(log_target / 4)

        def scaled(x, factor=1.0):
            # `target * x / groups` of a weight below fp32's least normal: its weight's quarters each a normal
            # number wherever the product is (e^(L/4), an exact quarter of its logarithm L: a normal product
            # of a finite x holds L above -176, a quarter above -44), multiplied into x one at a time, so none
            # rounds to 0 first and each takes one rounding, its derivative alike. An infinite `x` its own
            # signed infinity where a quarter is 0 (its weight's slope 0 there), where the product would be
            # 0 * inf.
            # A `factor` (2, a halved gap's) taken into the last quarter, inside the chain, so neither the
            # product nor its transpose rounds or overflows apart from it.
            product = (((x * quarter) * quarter) * quarter) * (factor * quarter) / groups
            return jnp.where(jnp.isinf(x) & (quarter == 0), jax.lax.stop_gradient(x), product)

        product = target * gap
        direct = jnp.where(jnp.isfinite(gap),
                           jnp.where(jnp.isfinite(product), product / groups, target * (gap / groups)),
                           target * (peak / groups) - target * (held / groups))
        # A gap past fp32's range halved before it is scaled (`peak / 2 - held / 2`, finite), so its two
        # sides' terms cannot flush apart before their normal sum.
        quartered = jnp.where(jnp.isfinite(gap), scaled(gap), scaled(peak / 2 - held / 2, 2.0))
        gaps = jnp.where(taken, jnp.where(tiny, quartered, direct), 0)
        # The entropy's terms formed as the gaps' are, a tiny weight's between its halves, so equal teacher
        # and student cancel where a flushed weight dropped one side.
        logged_safe = jnp.where(taken, log_target, 0)
        entropy = jnp.where(taken, jnp.where(tiny, scaled(logged_safe), target * logged_safe / groups), 0)
        divergence = compensated_sum(gaps, (gaps.ndim - 1,)) + spread / groups + compensated_sum(
            entropy, (entropy.ndim - 1,))
        return jnp.where(jnp.any(keep, axis=-1), divergence, 0)

    # The groups' mean: each group's KL already at its share (`kl`'s `weight`), summed.
    return compensated_sum(jnp.stack([kl(group) for group in range(groups)]), (0,))


INFLLM_STRIDE = 16
"""InfLLM-V2's pooling stride (arXiv 2509.24663 §3.3: windows of 32 keys every 16)."""


INFLLM_INITIAL = 1
"""The leading blocks InfLLM-V2's selection forces every query (arXiv 2509.24663 §3.3)."""

INFLLM_LOCAL = 1
"""The blocks ending at a query's own that InfLLM-V2's selection forces it."""


def infllm_selection(query: jax.Array, key: jax.Array, *, block: int, top_k: int = 64, kernel: int = 32,
                     stride: int = INFLLM_STRIDE, initial: int = INFLLM_INITIAL, local: int = INFLLM_LOCAL,
                     key_valid: jax.Array | None = None) -> jax.Array:
    """InfLLM-V2's selection (arXiv 2509.24663 §3.3): keys mean-pooled by a window of `kernel` every `stride`
    into compressed keys (eq. 4), each head's softmax of its query against the compressed keys whose window
    it reaches whole, at the attention's scale `D ** -0.5` (eq. 5, causal), summed over a GQA group's heads
    (eq. 6), max-pooled onto key blocks of `block` over `block / stride` compressed keys every `block /
    stride` and `kernel / stride - 1` more (eq. 7: 5 every 4 at the paper's 64, 32, 16), the highest of
    the earlier blocks beside the first `initial` and the `local` ending at the query's own, forced, `top_k`
    blocks in all (eqs. 2, 3; the paper's 64 counting its forced blocks); a block no window the query
    reaches covers is no candidate. `query` `[B, Q, G * h, D]`, `key` `[B, K, G, D]`; `[B, G, Q, initial +
    local + min(top_k - initial - local, blocks)]`, -1 past the blocks a query holds (its optional slots no
    more than the call's blocks, `top_k_selection`). Its pooling is `reduce_window`'s, each window a key's
    or a score's once, a key `key_valid` `[B, K]` hides pooled as a zero."""
    assert block % stride == 0 and kernel % stride == 0, "InfLLM-V2's pooling takes a stride dividing both"
    if top_k < initial + local:
        raise ValueError(f"InfLLM-V2's selection forces each query's first {initial} and {local} ending at "
                         f"its own, and top_k counts them: a top_k of {initial + local} or more, not {top_k}")
    optional = top_k - initial - local
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
    if optional == 0:
        # No pick is optional: the forced blocks alone, no score formed.
        return forced
    if windows == 0:
        # No window holds `kernel` keys: the forced blocks alone.
        nothing = jnp.zeros((rows, groups, queries, blocks), bool)
        return top_k_selection(jnp.zeros(nothing.shape), nothing, optional, forced=forced)
    # A hidden key (`key_valid`) pooled as a zero, so no hidden token moves a pick; a window of none but
    # hidden keys scores nothing, and a block of none is no optional pick (the forced ones stand).
    pooled_keys = key.astype(jnp.float32)
    reach = jnp.arange(windows)[None, :] * stride + kernel <= jnp.arange(queries)[:, None] + 1
    if key_valid is not None:
        valid = jnp.broadcast_to(key_valid, (rows, keys))
        pooled_keys = jnp.where(valid[:, :, None, None], pooled_keys, 0.0)
        held = jax.lax.reduce_window(valid.astype(jnp.int32), np.int32(0), jax.lax.add, (1, kernel),
                                     (1, stride), "VALID") > 0
        reach = reach[None] & held[:, None, :]
        lived = jnp.pad(valid, ((0, 0), (0, blocks * block - keys))).reshape(rows, blocks, block).any(-1)
        earlier = earlier & lived[:, None, None, :]
    summed = jax.lax.reduce_window(pooled_keys, 0.0, jax.lax.add, (1, kernel, 1, 1),
                                   (1, stride, 1, 1), "VALID")
    pooled = summed / kernel
    logits = jnp.einsum("bqghd,bwgd->bghqw", query.astype(jnp.float32), pooled) * width ** -0.5
    # A query reaches a window whose last key is at or before its own (and, of a row with hidden keys, one
    # holding a real key: `reach` `[B, Q, W]`, at the logits' rows, groups and heads).
    reach = reach if reach.ndim == 2 else reach[:, None, None]
    logits = jnp.where(reach, logits, -jnp.inf)
    # Each window's weight summed over the group's heads, -inf at a window the query does not reach (and in
    # the padding past the last), so a block no reached window covers pools -inf and is no candidate (a
    # weight of 0 there took a slot where padding added a window no query reached, train 1359's F3), and
    # padding past a row's keys never moves its picks; a reached window of weight 0 stays a candidate.
    reached = reach if reach.ndim == 2 else reach[:, 0]
    weights = jnp.where(reached, jnp.where(reach, jax.nn.softmax(logits, axis=-1), 0.0).sum(axis=2), -jnp.inf)
    step = block // stride
    span = step + kernel // stride - 1
    padded = jnp.pad(weights, ((0, 0), (0, 0), (0, 0), (0, max(0, (blocks - 1) * step + span - windows))),
                     constant_values=-jnp.inf)
    scores = jax.lax.reduce_window(padded, -jnp.inf, jax.lax.max, (1, 1, 1, span), (1, 1, 1, step),
                                   "VALID")[..., :blocks]
    return top_k_selection(scores, earlier & (scores > -jnp.inf), optional, forced=forced)


INFLLM_DENSE_BELOW = 8192
"""The keys below which InfLLM-V2 attends densely (kernel-templates.md §A row 60), the plain causal call: its
short calls' path, a selection of 64 blocks of 64 near every key of them anyway."""


def infllm_attention(query: jax.Array, key: jax.Array, value: jax.Array, *, block: int = 64, top_k: int = 64,
                     kernel: int = 32, stride: int = INFLLM_STRIDE, initial: int = INFLLM_INITIAL,
                     local: int = INFLLM_LOCAL,
                     dense_below: int = INFLLM_DENSE_BELOW, implementation: AttentionImpl = "auto",
                     dtype=None, precision=None, force_fp32_for_softmax: bool = True,
                     wide_logits: bool = False, softcap: float | None = None,
                     scale_logits: bool = False, key_valid: jax.Array | None = None,
                     sinks: jax.Array | None = None) -> jax.Array:
    """InfLLM-V2's attention (kernel-templates.md §A row 60): a call over fewer than `dense_below` keys the
    plain causal call; a longer one each query's selected blocks (`infllm_selection`: `top_k` in all, the top
    of its earlier blocks of `block` by its group's two-window pooled scores, the first `initial` and the
    `local` ending at its own forced) through the template's indexed form (`selected_attention`), reading
    those
    blocks' keys alone. `query` `[B, Q, G * h, D]`, `key` and `value` `[B, K, G, D]`, a causal call whose
    queries are its keys, its rows and heads whole: the call's arguments past them its numerics alone
    (`dtype`, `precision`, `force_fp32_for_softmax`, `wide_logits`, `softcap`, `scale_logits`), which both
    branches take alike; an operand laid by its rows, heads or places it takes none of, the selected
    branch folding each group into the batch. `key_valid` `[B, K]` marks a row's real keys (a right-padded
    row's), which both branches read alone and whose count is the row's for the cut: a row of fewer real keys
    than `dense_below` attends densely within a longer call, as its tokens unpadded do. `sinks` (`[H]`, or
    `[B, Q, H]` a query's own) join both branches' softmaxes. The selection takes no gradient."""
    if top_k < initial + local:
        raise ValueError(f"InfLLM-V2's selection forces each query's first {initial} and {local} ending at "
                         f"its own, and top_k counts them: a top_k of {initial + local} or more, not {top_k}")
    keys = key.shape[1]
    numerics = {"dtype": dtype, "precision": precision, "force_fp32_for_softmax": force_fp32_for_softmax,
                "wide_logits": wide_logits, "softcap": softcap, "scale_logits": scale_logits, "sinks": sinks,
                "key_valid": key_valid}

    def dense():
        return scaled_dot_product_attention(query, key, value, causal=True, implementation=implementation,
                                            **numerics)

    def sparse():
        selected = infllm_selection(query, key, block=block, top_k=top_k, kernel=kernel, stride=stride,
                                    initial=initial, local=local, key_valid=key_valid)
        return selected_attention(query, key, value, selected, block_runs(keys, block),
                                  implementation=implementation, **numerics)

    if keys < dense_below:
        return dense()
    if key_valid is None:
        return sparse()
    # A row of fewer real keys than the cut attends densely, its branch its real keys' (train 1345's F3): each
    # branch runs where some row takes it alone.
    short = jnp.sum(jnp.broadcast_to(key_valid, (key.shape[0], keys)), axis=-1) < dense_below
    return jax.lax.cond(jnp.all(short), dense, lambda: jax.lax.cond(
        jnp.any(short), lambda: jnp.where(short[:, None, None, None], dense(), sparse()), sparse))


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

    selection: Literal["moba", "infllm", "msa"] = "moba"
    block: int = 512
    top_k: int = 3
    index_width: int = 64
    """MSA's index heads' width (`MsaIndexer.width`, the paper's d_idx)."""
    dense_below: int = INFLLM_DENSE_BELOW
    """InfLLM-V2's dense cut (`infllm_attention`): fewer keys attend densely."""

    reads: ClassVar[frozenset[str]] = frozenset()

    def _attended(self, masking, positions, append, sinks, train: bool, *, bias=None, hidden=None):
        """The attention over each query's selected chunks (`selected_attention`), its selection MoBA's over
        the call's keys in each document's chunks (`moba_chunks`): an uncached call's its own tokens, by
        their documents and validity; a cached call's its cache's filled slots (`cached_validity`), its
        queries at the slots they were written to. The kernel reads the call's real keys alone, so a chunk
        whose run spans a key the call hides reads none of it. The gate's selection is sown (`SELECTIONS`)
        for a caller that opens the collection: its scores, the windows each query may take, its budget.
        `sinks` (a head's, or a query's own projected) join each query's softmax over its selection."""
        del positions, train
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
        if self.selection == "infllm":
            return checkpoint_name(self._infllm(query, key, value, masking, append, sinks), Residual.CONTEXT)
        if self.selection == "msa":
            return checkpoint_name(self._msa(query, key, value, masking, append, hidden), Residual.CONTEXT)
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
        sow_selection(self, gate, None, keep[:, None], self.top_k - 1)
        selected = moba_selection(gate, keep, self.top_k)
        return checkpoint_name(selected_attention(
            query, key, value, selected, window_runs(chunks.windows), own=chunks.own, slots=slots,
            key_valid=valid, sinks=sinks, dtype=self.ctx.dtype,
            precision=self.ctx.precision, force_fp32_for_softmax=self.ctx.force_fp32_for_softmax,
            wide_logits=self.ctx.wide_logits, softcap=self.ctx.attn_logit_softcap), Residual.CONTEXT)


    def _causal_implies_right_padding(self, metadata, causal: bool) -> bool:
        # InfLLM-V2's dense cut reads the call's keys whole, its padding among them, so a right-padded row's
        # validity reaches it (and its refusal), unelided: a padded row would choose its branch apart from
        # the same real tokens unpadded.
        return self.selection != "infllm" and super()._causal_implies_right_padding(metadata, causal)

    def _infllm(self, query, key, value, masking, append, sinks):
        """InfLLM-V2's attention over the call's own tokens (`infllm_attention`: dense below its cut, its
        selected blocks of `block`, `top_k` of them in all, past it), a row one document: its hidden tokens
        (a right-padded row's, by its validity or its one document's ids) it reads as such, pooled as zeros
        and its cut taken at its real keys' count; a packed row's documents or a cache's slots it refuses.
        `sinks` join both branches."""
        if append is not None or (masking.documents is not None and masking.max_segments != 1):
            raise ValueError(f"{self.name}: InfLLM-V2's selection reads a row of one document's tokens, its "
                             "own: no cache or packed documents")
        # A row's real keys: its validity where stated, else its one document's ids, its padding 0 (a
        # right-padded row of `lm(..., segments=1)`'s, train 1359's F2).
        # Both where both are stated: a key the validity hides or the ids pad is no real key (a mask hiding
        # one real token beside padding the ids mark counted the padding, train 1359's review).
        marks = [jnp.broadcast_to(jnp.asarray(mark, bool), key.shape[:2]) for mark in (
            masking.valid, None if masking.documents is None else jnp.asarray(masking.documents) != 0)
            if mark is not None]
        valid = functools.reduce(jnp.logical_and, marks) if marks else None
        return infllm_attention(query, key, value, block=self.block, top_k=self.top_k,
                                dense_below=self.dense_below, key_valid=valid, sinks=sinks,
                                dtype=self.ctx.dtype,
                                precision=self.ctx.precision,
                                force_fp32_for_softmax=self.ctx.force_fp32_for_softmax,
                                wide_logits=self.ctx.wide_logits, softcap=self.ctx.attn_logit_softcap)


    def _msa(self, query, key, value, masking, append, hidden):
        """MSA's attention (arXiv 2606.13392 §3): its index branch over the layer's input (`MsaIndexer`, an
        index query head a key head's group beside one shared index key head, the input detached), each
        group's `top_k` blocks of `block` its own forced (`msa_selection`), through the template's indexed
        form (`selected_attention`). Its selection is sown (`SELECTIONS`, `mla.sow_selection`'s record), and
        where an objective trains an indexer its loss per query (`msa_query_kls`, `INDEXER_COLLECTION`),
        which trains the index branch alone. Over the call's own tokens, a row one document: a right-padded
        row's trailing padding it reads as keys past the real ones, which causality keeps every real query
        from (no dense cut to move)."""
        if append is not None or masking.documents is not None or masking.valid is not None:
            raise ValueError(f"{self.name}: MSA's selection reads a row of one document's real tokens, its "
                             "own: no cache, packed documents or hidden token")
        scores = MsaIndexer(groups=key.shape[2], width=self.index_width, dtype=self.ctx.dtype,
                            precision=self.ctx.precision, name="msa_indexer")(hidden)
        pooled, taken, forced = msa_candidates(scores, block=self.block)
        selected = msa_selected(pooled, taken, forced, top_k=self.top_k)
        # Its selection as the gate reads every one (`selection_margins`): each query's blocks by their pooled
        # scores among those it takes by score, at the slots its forced own block leaves.
        sow_selection(self, pooled, None, taken, min(self.top_k, pooled.shape[-1]) - 1)
        # Its indexed call prepared once and its route decided once where its dispatch decides it (the
        # selector's, at a sequence exchange's instance's geometry, the call range-safe so no route forming
        # its raw products whole takes it, `AttentionCase.range_safe`), the call run on that decision and
        # its teacher forming its logits as that route does (`msa_scores`), on any mesh.
        bounds = block_runs(key.shape[1], self.block)
        options = {"dtype": self.ctx.dtype, "precision": self.ctx.precision,
                   "force_fp32_for_softmax": self.ctx.force_fp32_for_softmax,
                   "wide_logits": self.ctx.wide_logits, "softcap": self.ctx.attn_logit_softcap,
                   "scale_logits": True, "range_safe": True}
        prepared = indexed_call(query, key, value, selected, bounds, **options)
        route = indexed_route(prepared)
        if not self.is_initializing() and self.is_mutable_collection(INDEXER_COLLECTION):
            # Its indexer's loss per query, as an objective training an indexer reads it
            # (`IndexerTraining`, `mla.indexer_kl`'s collection).
            teacher = msa_scores(self.ctx.dtype, self.ctx.precision, wide_logits=self.ctx.wide_logits,
                                 route=attention_implementation(route.kernel))
            self.sow(INDEXER_COLLECTION, "kl", msa_query_kls(
                scores, query, key, selected, block=self.block, scale=query.shape[-1] ** -0.5,
                teacher=teacher))
        # jax.nn's form (`scale_logits`), the one every route forms its logits in, on its decided route.
        return indexed_attention(prepared, route=route)


@dataclasses.dataclass(frozen=True)
class BlockSparseAttentionMixer(AttentionMixer):
    """Block-selected attention (`BlockSparseSelfAttention`): each query reads its own block and the earlier
    blocks its `selection` picks, `top_k` in all with its own (InfLLM-V2's first block forced among them),
    blocks of `block` places. MoBA's (the default; arXiv
    2502.13189, blocks of 512 and the top 3) reads the layer's own queries and keys, no parameter of its own,
    over each document's blocks and a cached call's cache. InfLLM-V2's (`"infllm"`, blocks of 16s, dense
    below `dense_below` keys) and MSA's (`"msa"`, an index branch of parameters over the layer's input,
    `index_width` wide, trained by its own loss) read a row of one document, uncached: a packed row, a hidden
    token within it or a cache they refuse."""

    selection: Literal["moba", "infllm", "msa"] = "moba"
    block: int = 512
    top_k: int = 3
    index_width: int = 64
    dense_below: int = INFLLM_DENSE_BELOW

    # It selects over each row's cache whole, one document a row.
    packed_cache = False
    reads_forks = False
    drops_attention = False

    def mixed_refusal(self, ctx: MixerContext) -> str | None:
        return ("block-selected attention reads each query's blocks as key runs, which a mixed serving step "
                "lays none of")

    @property
    def indexed_mixers(self) -> tuple["IndexedMixer", ...]:
        # MSA's index branch, which an objective trains (`IndexerTraining`), selecting the keys its queries
        # attend (`sparse`).
        return (self,) if self.selection == "msa" else ()

    @property
    def indexed(self) -> bool:
        return self.selection == "msa"

    @property
    def sparse(self) -> bool:
        return self.selection == "msa"

    def generation_refusal(self) -> str | None:
        # InfLLM-V2's and MSA's selections read a row of one document's own tokens, uncached (`_infllm`,
        # `_msa`): a generation step's cache they refuse.
        if self.selection != "moba":
            return (f"block_sparse_attention's {self.selection} selection reads a row's own tokens, "
                    "uncached, where generation reads a cache")
        return None

    def build(self, ctx: MixerContext) -> Callable[..., nn.Module]:
        if not ctx.causal:
            raise ValueError("block_sparse_attention requires causal=True: a query selects the blocks before "
                             "its own")
        if self.selection == "infllm" and self.block % INFLLM_STRIDE:
            raise ValueError(f"InfLLM-V2's selection pools keys every {INFLLM_STRIDE} places into whole "
                             f"blocks, so takes a block of them, got {self.block}")
        if self.selection == "msa" and ctx.attn_logit_softcap is not None:
            raise ValueError("MSA's indexer loss takes its heads' plain softmax as its target, which a logit "
                             "softcap would part from the attention it trains toward")
        if self.selection == "msa" and (ctx.attention_sinks or self.projected_sinks):
            raise ValueError("MSA's indexer loss takes its heads' plain softmax as its target, which sinks "
                             "would part from the attention it trains toward")
        if self.block < 1 or self.top_k < 1:
            raise ValueError(f"block_sparse_attention takes a positive block and top_k, got {self.block} and "
                             f"{self.top_k}")
        if self.selection == "infllm" and self.top_k < INFLLM_INITIAL + INFLLM_LOCAL:
            raise ValueError(f"InfLLM-V2's selection forces each query's first and own blocks, and top_k "
                             f"counts them: a top_k of {INFLLM_INITIAL + INFLLM_LOCAL} or more, not "
                             f"{self.top_k}")
        if self.alibi or self.bidirectional_images or self.bidirectional_blocks:
            raise ValueError("block_sparse_attention adds no ALiBi bias and reads no image or block whole")
        ctx.refuse("block_sparse_attention", {
            "attention_dropout_rate": ctx.attention_dropout_rate,
            "sliding_window": ctx.sliding_window, "attention_chunk": ctx.attention_chunk,
            "kv_shared": ctx.kv_shared,
        }, "each query head's selection is a row of its call, whose dropout draws a head's across rows, "
           "and it reads its document's blocks from their start, no window, chunk or other layer's keys")
        kind = {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}
        return functools.partial(BlockSparseSelfAttention, ctx=ctx, **kind)
