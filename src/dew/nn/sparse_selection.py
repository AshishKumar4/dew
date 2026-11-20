"""DeepSeek sparse attention's selection and its execution in the latent space.

A selection is `[B, S, K]` key indices per query, -1 in a slot that names
no key, which is how GLM's reference builds its top-k mask
(`build_attention_mask_from_topk`, modeling_glm5_next.py:1218-1256). The
V3.2 indexer (`dew.nn.mla`), GLM's k-pool indexer (`dew.nn.dsa_kpool`) and
DeepSeek V4's compressed-entry indexer (`dew.nn.deepseek_v4`) all score
their keys through `index_scores` and select through `top_k_selection`,
`selection_mask` turns it into the `[B, S, T]` mask the dense kernels
read, and `sparse_latent_attention` attends it without that mask.
"""

import dataclasses
import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from dew.nn.attention import AttentionImpl, scaled_dot_product_attention
from dew.nn.derivatives import KernelGlue, contracted
from dew.nn.inputs import sequence_blocks
from dew.nn.kernels import generation
from dew.nn.kernels.generation import INHERENT, Choice, backend
from dew.nn.kernels.selection import Kernel, Op, Requirement, select
from dew.nn.precision import accumulated, at_least_fp32, exact_products, float32_einsum
from dew.nn.scatter import DROPPED
from dew.nn.sharding import manual_map


def index_agreements(query, keys, precision=None):
    """Each lightning indexer head's agreement of each query with each key, `q_h . k` before its relu,
    at least in fp32 (`at_least_fp32`): queries `[B, S, H, D]` against keys `[B, T, D]` every query
    shares or `[B, S, T, D]` per query, as `[B, S, H, T]`, under `precision`. A key every head
    disagrees with scores zero (`index_scores`), however near zero its agreements sit."""
    wide = at_least_fp32(query.dtype)
    shared = 'bshd,btd->bsht' if keys.ndim == 3 else 'bshd,bstd->bsht'
    return float32_einsum(shared, query.astype(wide), keys.astype(wide), precision=precision)


def index_scores(agreements, weights, width: int, precision=None):
    """The lightning indexer's score of each query against each key,
    `sum_h w_h relu(q_h . k) / sqrt(head_dim) / sqrt(n_heads)`, at least in
    fp32, from its heads' `agreements` `[B, S, H, T]` (`index_agreements`) of
    queries `width` wide, weighted by `weights` `[B, S, H]`, as `[B, S, T]`;
    the head sum takes `precision`. The V3.2 indexer scores its keys, GLM's
    k-pool its pooled keys and DeepSeek V4 its compressed entries through it."""
    heads, wide = agreements.shape[-2], agreements.dtype
    scores = jnp.maximum(agreements, 0) * width ** -0.5
    # The head sum as a [1, H] by [H, T] product per query: written as the
    # einsum over the scores first, it ran a third slower forward and backward
    # on an H100 at the V3.2 indexer's 64 heads over 4096 keys.
    weighted = (weights.astype(wide) * heads ** -0.5)[..., None, :]
    return accumulated(functools.partial(jnp.matmul, precision=precision),
                       weighted, scores).squeeze(-2)


def top_k_selection(scores, keep, top_k: int, *, forced=None):
    """The `top_k` highest-scoring keys of each query among those `keep`
    allows: `[..., K]` indices with `K = min(top_k, T)`, -1 in a slot no
    allowed key fills, then each of `forced` `[..., F]` (-1 for none), taken
    whatever its score and `keep`, which the top k passes over.

    Exactly `top_k` keys where at least that many are allowed, and every
    allowed key where fewer are, so a sequence the top-k covers attends as
    the dense mixer does. Equal scores choose the lower key index, because
    `jax.lax.top_k` is stable; torch's `topk` promises no tie order, so a
    selection a tie decides may differ from the reference's. `keep` is the
    attention mask over `scores` `[..., T]` (a leading axis of one
    broadcasts), and a key it forbids is never selected whatever the tie
    rule. A block selection takes blocks for keys (`block_maxima`,
    `dew.nn.block_sparse`).
    """
    total = scores.shape[-1]
    keep = jnp.broadcast_to(keep, scores.shape)
    if forced is not None:
        keep = keep & ~jnp.any(jnp.arange(total)[:, None] == forced[..., None, :], axis=-1)
    ranked = jnp.where(keep, scores, jnp.finfo(at_least_fp32(scores.dtype)).min)
    chosen = jax.lax.top_k(ranked, min(top_k, total), is_stable=True)[1]
    chosen = jnp.where(jnp.take_along_axis(keep, chosen, axis=-1), chosen, -1)
    return chosen if forced is None else jnp.concatenate([chosen, forced.astype(chosen.dtype)], axis=-1)


def selection_mask(indices, total: int):
    """`[B, S, K]` token indices, -1 for none, as the `[B, S, total]` bool mask
    the attention takes: a key is visible to a query iff one of the query's
    indices names it, so out-of-range entries drop out."""
    batch, length, _ = indices.shape
    # A negative index would wrap around in jnp, and one at or past `total`
    # names no key; both go to `DROPPED`, which the scatter drops.
    slots = jnp.where((indices >= 0) & (indices < total), indices, DROPPED)
    return jnp.zeros((batch, length, total), jnp.bool_).at[
        jnp.arange(batch)[:, None, None], jnp.arange(length)[None, :, None], slots
    ].set(True, mode='drop')


def top_k_keys(scores, keep, top_k: int):
    """`top_k_selection` as the `[B, S, T]` mask the dense kernels read."""
    return selection_mask(top_k_selection(scores, keep, top_k), scores.shape[-1])


SPARSE_QUERY_BLOCK = 128
"""Queries `sparse_latent_attention` attends at once. A block holds its
queries' selected latents, `[B, block, K, kv_lora_rank + rope]`, so this
bounds the gather at 151M values per batch row for V3.2's K of 2048."""


def sparse_latent_attention(query_nope, query_rot, latent, rot, key_weight, value_weight,
                            indices, *, scale: float, precision=None,
                            block: int = SPARSE_QUERY_BLOCK):
    """Attend each query over the `K` keys it selected, in the latent space.

    DeepSeek sparse attention's execution (arXiv 2512.02556, section 2.1;
    FlashMLA's sparse decoding kernel): the keys are never expanded per
    head. The nope query absorbs the key half of `kv_b_proj`, so a logit is
    the absorbed query against the key's latent plus the rope query against
    its shared rope head, and the probabilities weight the selected latents
    before the value half expands the result. It is the same softmax over
    the same keys as the dense attention under the selection mask, in
    `O(S * K)` rather than `O(S * T)`.

    `query_nope` `[B, S, H, n]` and `query_rot` `[B, S, H, p]` are the
    rotated, pre-scaled query halves; `latent` `[B, T, r]` and `rot`
    `[B, T, p]` the normed latent and the rotated rope head; `key_weight`
    `[r, H, n]` and `value_weight` `[r, H, v]` the two halves of
    `kv_b_proj`, which carries no bias in any reference. `indices` is the
    selection, -1 where a slot names no key. `scale` is the kernel's
    `1 / sqrt(n + p)`. Returns `[B, S, H, v]`.

    Queries run in blocks of `block` under `jax.checkpoint`, so neither the
    forward nor the backward pass holds more than one block's gathered
    latents. A query whose selection names no key (padding) averages its
    slots where the dense kernel averages every key; both are outputs
    nothing reads.
    """
    absorbed = float32_einsum('bshn,rhn->bshr', query_nope, key_weight.astype(query_nope.dtype),
                              precision=precision)
    batch, length, heads, rank = absorbed.shape
    dtypes = tuple(jnp.dtype(x.dtype) for x in (absorbed, query_rot, latent, rot))
    case = SparseCase(heads, rank, query_rot.shape[-1], indices.shape[-1], dtypes, batch * length,
                      batch * latent.shape[1], scale, precision)
    if select(SPARSE, case).kernel == 'template':
        context = templated_context(scale, precision, block)(absorbed, query_rot, latent, rot, indices)
    else:
        context = blocked_context(absorbed, query_rot, latent, rot, indices, scale=scale, precision=precision,
                                  block=block)
    return float32_einsum('bshr,rhv->bshv', context, value_weight.astype(context.dtype),
                          precision=precision)


def blocked_context(absorbed, query_rot, latent, rot, indices, *, scale: float, precision=None,
                    block: int = SPARSE_QUERY_BLOCK):
    """`sparse_latent_attention`'s heads' context in the latent space `[B, S, H, r]`, in XLA: each block of
    `block` queries gathers its selection's latents and rope heads and attends them, under
    `jax.checkpoint`, so neither pass holds more than one block's gathered latents."""
    batch, length, heads, _ = absorbed.shape
    blocks = -(-length // block)

    def blocked(x):
        return sequence_blocks(x, block, blocks, leading=True)

    def gathered(table, where):
        return jax.vmap(lambda rows, at: rows[at])(table, where)

    @jax.checkpoint
    def attend(pieces):
        q_latent, q_rot, at = pieces
        allowed, at = at >= 0, jnp.maximum(at, 0)
        keys, rots = gathered(latent, at), gathered(rot, at)
        wide = at_least_fp32(q_latent.dtype)
        logits = (float32_einsum('bqhr,bqkr->bqhk', q_latent, keys, precision=precision,
                                 preferred_element_type=wide)
                  + float32_einsum('bqhp,bqkp->bqhk', q_rot, rots, precision=precision,
                                   preferred_element_type=wide)) * scale
        logits = jnp.where(allowed[:, :, None, :], logits, jnp.finfo(wide).min)
        weights = jax.nn.softmax(logits, axis=-1)
        return float32_einsum('bqhk,bqkr->bqhr', weights, keys.astype(wide), precision=precision,
                              preferred_element_type=wide).astype(keys.dtype)
    context = jax.lax.map(attend, tuple(blocked(x) for x in (absorbed, query_rot, indices)))
    return jnp.moveaxis(context, 0, 1).reshape(batch, blocks * block, heads, -1)[:, :length]


@dataclasses.dataclass(frozen=True)
class SparseCase:
    """A sparse latent attention call as its selector reads it (`SPARSE`): its `heads`, each over a latent
    `rank` wide and a rope head `rope` wide, the `top_k` slots each query selects, its four operands'
    `dtypes`, its `rows` (a row a query), its `pool_keys` (`B * T`), its `scale`, and its dot `precision`
    (`TEMPLATE_SPARSE_SUPPORTS`)."""

    heads: int
    rank: int
    rope: int
    top_k: int
    dtypes: tuple[jnp.dtype, ...]
    rows: int
    pool_keys: int
    scale: float
    precision: jax.lax.PrecisionLike = None


def _template_tiles(case: SparseCase) -> bool:
    """Whether the attention template's decode reads `case`'s selection on the GPU: its work list derives a
    launch (`attention_decode.planned`, as `templated_context` takes the call)."""
    from dew.nn.kernels.templates import attention as template, attention_decode

    if backend() != "gpu":
        return False
    dtype = case.dtypes[0]
    width = case.rank + case.rope
    query = jax.ShapeDtypeStruct((case.rows, 1, case.heads, width), dtype)
    pool = jax.ShapeDtypeStruct((1, case.pool_keys, 1, width), dtype)
    table = jax.ShapeDtypeStruct((case.rows, case.top_k), jnp.int32)
    derived = []

    def planned(query, pool, table):
        lengths = jnp.full((case.rows,), case.top_k, jnp.int32)
        spans = template.spans(1, case.top_k, case.rows, lengths=lengths)
        derived.append(attention_decode.planned(query, pool, pool[..., :case.rank], spans, template.Hooks(),
                                                scale=case.scale, table=table)[0])
        return query

    jax.eval_shape(planned, query, pool, table)
    return derived[0] is not None


TEMPLATE_SPARSE_SUPPORTS: tuple[Requirement[SparseCase], ...] = (
    Requirement('dtype',
                lambda case: len(set(case.dtypes)) == 1 and case.dtypes[0] in (jnp.bfloat16, jnp.float16),
                lambda case: f"the template's decode reads matching 16-bit operands, not {case.dtypes}",
                row=INHERENT),
    Requirement('precision', lambda case: exact_products(case.precision, case.dtypes[0]),
                lambda case: f"the template's decode runs exact 16-bit products, not {case.precision}",
                row=INHERENT),
    Requirement('tiles', _template_tiles,
                lambda case: f"the attention template's decode derives no launch over {case.top_k} selected "
                             f"latents {case.rank + case.rope} wide on this GPU", row=INHERENT),
)
"""What the attention template's decode needs to read a selection (`templated_context`)."""


def _sparse_preferred(case: SparseCase) -> str:
    """The attention template's decode over the selected latents on a GPU, the blocked XLA path elsewhere."""
    return 'template' if backend() == 'gpu' else 'xla'


SPARSE: Op[SparseCase] = Op(Choice.SPARSE_ATTENTION, (
    Kernel('template', TEMPLATE_SPARSE_SUPPORTS),
    Kernel('xla', (), generic=True),
), _sparse_preferred)
"""The sparse latent attention's selector: on a GPU the attention template's decode loads each query's
selected latents into its key tiles (`templated_context`), where its requirements hold
(`TEMPLATE_SPARSE_SUPPORTS`); else the blocked XLA path (`blocked_context`), which gathers them."""


def _template_forward(absorbed, query_rot, latent, rot, indices, *, scale: float):
    """`templated_context`'s value: a row a query, its heads its rows (`attention_decode.packed`), over its
    selection's latents as one-key pages of the pool of every row's latents and rope heads side by side
    (`[1, B * T, 1, r + p]`, its values the latents' `r` columns), loaded into the walk's key tiles by the
    page table of its selection's slots (`attention_mosaic_forward.paged_reads`), the slots that name a key
    first, in order, and its keys spanning them; a query naming none reads its first slot, an output
    nothing reads. Interpreted where the arrays lie off a GPU, whatever GPU the call is chosen for."""
    from dew.nn.kernels.templates import attention as template, attention_decode

    batch, length, heads, rank = absorbed.shape
    keys = latent.shape[1]
    query = jnp.concatenate([absorbed.astype(latent.dtype), query_rot.astype(latent.dtype)], axis=-1)
    pool = jnp.concatenate([latent, rot.astype(latent.dtype)], axis=-1).reshape(1, batch * keys, 1, -1)
    named = indices >= 0
    taken = jnp.take_along_axis(indices, jnp.argsort(~named, axis=-1, stable=True), axis=-1)
    table = (jnp.maximum(taken, 0) + keys * jnp.arange(batch)[:, None, None]).astype(jnp.int32)
    lengths = jnp.maximum(jnp.sum(named, axis=-1), 1).reshape(-1)
    def local(query, pool, table, lengths):
        spans = template.spans(1, indices.shape[-1], query.shape[0], lengths=lengths)
        return attention_decode.attend(query, pool, pool[..., :rank], spans, template.Hooks(),
                                       scale=scale, table=table,
                                       interpret=generation.placed_platform() != "gpu")

    out = manual_map(local, (P(), P(), P(), P()), P())(
        query.reshape(batch * length, 1, heads, -1), pool, table.reshape(batch * length, -1), lengths)
    return out.reshape(batch, length, heads, rank), ()


@functools.cache
def templated_context(scale: float, precision, block: int) -> Callable:
    """`blocked_context`'s heads' context under `scale`, its value the attention template's decode over the
    selected latents (`_template_forward`), its derivatives at every order the blocked XLA path's at
    `precision` and `block` (`generation.linear_tangent`): reverse mode its pullback, forward mode its
    tangent; a selection takes none."""
    portable = functools.partial(blocked_context, scale=scale, precision=precision, block=block)

    def backward(saved, operands, cotangent):
        return tuple(jax.vjp(portable, *operands)[1](cotangent))

    return contracted("test_the_sparse_latent_kernel_holds_every_order")(
        generation.linear_tangent(functools.partial(_template_forward, scale=scale), backward, portable,
                                  kind=KernelGlue()))


def block_maxima(scores, visible, block_size: int):
    """Each block of `block_size` entries' best score among those `visible`
    allows (every one where None), `[..., ceil(T / block_size)]`, -inf for a
    block with none: the
    Hierarchical Sparse Indexer's first level's pooling (`candidate_pool`),
    and MiniMax's indexer's onto its key blocks
    (`dew.nn.block_sparse.pooled_selection`)."""
    total = scores.shape[-1]
    count = -(-total // block_size)
    ranked = scores if visible is None else jnp.where(visible, scores, -jnp.inf)
    ranked = jnp.pad(ranked, [(0, 0)] * (ranked.ndim - 1) + [(0, count * block_size - total)],
                     constant_values=-jnp.inf)
    return jnp.max(ranked.reshape(*ranked.shape[:-1], count, block_size), axis=-1)


def candidate_pool(scores, visible, blocks: int, block_size: int):
    """The Hierarchical Sparse Indexer's first level (v41:583-610): the
    `blocks` blocks of `block_size` entries with the highest best score
    (`block_maxima`), the block holding the query's newest entry always
    among them (`top_k_selection`'s `forced`), as the entries they hold,
    `[B, S, min(blocks, ceil(T / block_size)) * block_size]` in ascending
    order, -1 for a pick past the reachable blocks or an entry past the
    last. No entries (a sequence shorter than its compressor's rate) or no
    blocks is an empty pool, which reserves no block for the newest."""
    total = scores.shape[-1]
    best = block_maxima(scores, visible, block_size)
    count = min(blocks, best.shape[-1])
    if count == 0:
        return jnp.full((*scores.shape[:-1], 0), -1, jnp.int32)
    newest = (jnp.sum(visible, axis=-1) - 1) // block_size
    chosen = top_k_selection(best, best > -jnp.inf, count - 1, forced=newest[..., None])
    chosen = jnp.repeat(jnp.sort(chosen, axis=-1), block_size, axis=-1)
    entries = chosen * block_size + jnp.tile(jnp.arange(block_size), chosen.shape[-1] // block_size)
    return jnp.where((chosen >= 0) & (entries < total), entries, -1)


SELECTING = "selecting"
"""The scope a selection's scoring runs under (`jax.named_scope`): products that rank keys rather than
weigh them, as MoBA's gate does in fp32 whatever the logits' arithmetic (`dew.nn.block_sparse.moba_gate`),
which a program's logits are told apart from. A selection's scoring alone, never a product the attention
weighs: a witness of the logits' arithmetic passes over all within it (tests/reference_error.py's
`logits_operands`)."""


def block_runs(keys: int, block: int) -> jax.Array:
    """`[ceil(K / block), 2]`: each block of `block` keys of a sequence of `keys`, its first key and the key
    past its last, the entries a selection of blocks names (`selected_runs`)."""
    starts = jnp.arange(0, keys, block, dtype=jnp.int32)
    return jnp.stack([starts, jnp.minimum(starts + block, keys)], axis=-1)


def selected_runs(selected, bounds, *, own=None, slots=None, causal: bool = True):
    """Key runs `[B, Q, S + 1, 2]` (`[B, Q, S, 2]` without `own`) of a selection `selected` `[B, Q, S]`, each
    query's entries of `bounds` `[B, T, 2]` or `[T, 2]` (each entry's first key and the key past its last, an
    empty run for none: a row's blocks, `block_runs`, or a document's windows), -1 for none, beside each
    query's `own` run `[B, Q, 2]` where given; under causality each up to the query's own key (`slots`
    `[B, Q]`, each query's key, its index where None; -1 for a query with none); sorted, repeats and empty
    runs an empty run after the rest, as `scaled_dot_product_attention`'s `key_runs` takes them, which reads
    a selection of no entries and no own run as no key (the span builder's `_read_runs`). A selection takes
    no gradient."""
    rows, queries, _ = selected.shape
    bounds = jnp.broadcast_to(bounds, (rows, *bounds.shape[-2:]))
    slots = jnp.broadcast_to(jnp.arange(queries) if slots is None else slots, (rows, queries))
    taken = selected.astype(jnp.int32)
    at = jnp.maximum(taken, 0).reshape(rows, -1)
    start, end = (jnp.where(taken >= 0, jnp.take_along_axis(bounds[..., side], at, axis=-1).reshape(
        taken.shape), 0) for side in (0, 1))
    if own is not None:
        start, end = (jnp.concatenate([mine, theirs[..., None]], axis=-1)
                      for mine, theirs in ((start, own[..., 0]), (end, own[..., 1])))
    if causal:
        end = jnp.minimum(end, slots[..., None] + 1)
    held = end > start
    # The held runs first, in order, a repeat of the run before it dropped, the empty after them.
    order = jnp.lexsort((end, jnp.where(held, start, jnp.iinfo(jnp.int32).max)), axis=-1)
    start, end, held = (jnp.take_along_axis(each, order, axis=-1) for each in (start, end, held))
    repeated = jnp.concatenate([jnp.zeros_like(held[..., :1]), (start[..., 1:] == start[..., :-1])
                                & (end[..., 1:] == end[..., :-1])], axis=-1)
    held = held & ~repeated
    order = jnp.argsort(~held, axis=-1, stable=True)
    start, end, held = (jnp.take_along_axis(each, order, axis=-1) for each in (start, end, held))
    runs = jnp.stack([jnp.where(held, start, 0), jnp.where(held, end, 0)], axis=-1)
    return jax.lax.stop_gradient(runs)


def selected_attention(query, key, value, selected, bounds, *, own=None, slots=None, key_valid=None,
                       causal: bool = True, implementation: AttentionImpl = 'auto', **options):
    """Attention whose each query reads the entries its selection names (`selected` `[B, R, Q, S]`, a
    selection a GQA group's or a query head's, `selected_runs` over `bounds`) and its `own` run `[B, Q, 2]`
    where given, each of the `R` selections a row of the call's batch (the key runs are a row's), the call's
    real keys alone where `key_valid` `[B, K]` marks them (a run spans a hole the validity hides): `query`
    `[B, Q, R * h, D]`, `key` and `value` `[B, K, G, D]` with `R` a multiple of `G` (a head's selection
    reads its group's keys), out `[B, Q, R * h, Dv]`, through the template's indexed form
    (`scaled_dot_product_attention`'s `key_runs`). `options` are the call's other arguments. A call whose
    queries are its keys (`slots` None) states its causality to the kernel too; a cache call's runs alone
    end each query's keys."""
    rows, queries, heads, _ = query.shape
    groups = selected.shape[1]
    per = heads // groups
    shared = groups // key.shape[2]

    def batched(array: jax.Array, held: int) -> jax.Array:
        # `[B, S, R * held, D]` as `[B * R, S, held, D]`, a selection a row.
        length, depth = array.shape[1], array.shape[-1]
        return array.reshape(rows, length, groups, held, depth).transpose(0, 2, 1, 3, 4).reshape(
            rows * groups, length, held, depth)

    def rowed(array):
        # A row's `[B, ...]` (or one every row shares) at each of its selections' rows.
        return None if array is None else jnp.repeat(jnp.broadcast_to(array, (rows, *array.shape[-1:])),
                                                     groups, axis=0)

    rowed_bounds = jnp.repeat(jnp.broadcast_to(bounds, (rows, *bounds.shape[-2:])), groups, axis=0)
    rowed_own = None if own is None else jnp.repeat(own, groups, axis=0)
    runs = selected_runs(selected.reshape(rows * groups, queries, -1), rowed_bounds, own=rowed_own,
                         slots=rowed(slots), causal=causal)
    key, value = (jnp.repeat(each, shared, axis=2) for each in (key, value))
    out = scaled_dot_product_attention(batched(query, per), batched(key, 1), batched(value, 1),
                                       causal=causal and slots is None, key_runs=runs,
                                       key_valid=rowed(key_valid), implementation=implementation, **options)
    out = out.reshape(rows, groups, queries, per, -1).transpose(0, 2, 1, 3, 4)
    return out.reshape(rows, queries, heads, -1)
