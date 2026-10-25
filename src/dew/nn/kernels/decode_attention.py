"""A decode step's attention over a row's own keys, as a Pallas kernel for CUDA.

A serving step decodes every slot of its cache. jax.nn's xla attention, and
the folded form that reads the cache in its own layout
(`dew.nn.decode_geometry.folded_attention`), compute every row over the cache's
whole capacity, idle rows and the slots past a row's length included.
Each program here takes one row, one key head and one of the slices its keys
are cut into (`Split`), and reads only the blocks of the row's span of
keys, from its first (a sliding window's, where the layer has one) to its
length, with an online softmax over them: flash decoding. A windowed layer's
decode had built a mask instead and run jax.nn's xla attention, which
transposed the whole cache at every step: 5.5 of an H100's 21.8 ms decode
step of a 4B model with 42 windowed layers at 64 slots. One slice a row
writes the output itself; more write each slice's fp32 sums, its running
maximum and its softmax's total, which XLA combines (`_combined`). A decode
step of a few rows over a long context launches rows x heads programs, two
for one row of Qwen3.5's attention on a GPU of 108 or 132 multiprocessors;
slices fill the GPU, as many as its multiprocessors hold (`_heuristic`). Six
of Qwen3.5-0.8B's 256-wide layers at a 384-slot capacity, 128 rows with 16
drawing, took 0.28 ms against the folded form's 1.12 and jax.nn's 2.40 on an
RTX 4080, and 0.20 against 0.33 and 0.41 at 32 rows (docs/performance.md).
Its body is whole-block arithmetic and dots, so a Mosaic GPU port takes it
as is.

The arithmetic is jax.nn's but for the order of its sums: fp32 logits of
the operands, the scale after, an fp32 softmax, its probabilities in the
value's dtype for the value product, which accumulates in fp32 and is
normalized at the end rather than before the product, the slices' sums
combined in fp32 before it. Both products name
the highest precision, so fp32 operands multiply in IEEE fp32 whatever the
default precision, where Triton's default is one TF32 product.

`attend_paged` is the same kernel over a paged cache: a program takes one
row and one key head with the query heads that share it, and reads the row's
pages through its page table. JAX's GPU paged kernel
(`jax.experimental.pallas.ops.gpu.paged_attention`) stores its unnormalized
sums and its split partials in the query's dtype and divides by a bf16
denominator, three roundings where this has one.

A row reads at least one key: an idle slot reads its first.
"""

import dataclasses
import functools
import math
from typing import TYPE_CHECKING, ClassVar

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from . import tuning
from .generation import INHERENT, gpu_target, triton_runs
from .selection import Requirement
from .templates import attention_algebra as algebra
from .templates.attention import NATURAL, capped, kernel_dot

if TYPE_CHECKING:
    # Named in quotes: tokamax reads its ops' `_fwd` annotations as types, which postponed ones are not.
    from dew.nn.attention_kernels import AttentionCase
    from dew.nn.kv_cache import PagedCase

BLOCK = 32
"""Keys a step of the loop reads."""
QUERIES = 16
"""Query positions a program holds: the fewest a tensor-core dot takes."""
WARPS = 4
DTYPES = (jnp.float32, jnp.bfloat16, jnp.float16)
"""The dtypes it attends in: its products' own precision, accumulated in fp32."""

SPLITS = (1, 2, 4, 8, 16, 32)
"""The slices a launch may cut each row's keys into (`Split`), each held to the float64 gate on the GPU."""

@dataclasses.dataclass(frozen=True)
class Split:
    """A decode launch's config: the slices each row's keys are cut into, a program each, in whole
    steps of the loop."""
    splits: int


def _widths(head_dim: int) -> bool:
    """Whether the kernels take heads `head_dim` wide: a power of two from 16 to 256."""
    return head_dim & (head_dim - 1) == 0 and 16 <= head_dim <= 256


def _power_width(case: "AttentionCase") -> bool:
    return _widths(case.head_dim)


_Need = Requirement["AttentionCase"]
"""A requirement of an `AttentionCase`, the decode selector's case."""

SUPPORTS = (
    _Need('capability', lambda case: triton_runs(), "the decode kernel needs a GPU of sm80 or later",
          row=INHERENT),
    _Need('queries', lambda case: case.heads % case.kv_heads == 0
          and case.queries * (case.heads // case.kv_heads) <= QUERIES,
          lambda case: f"the call's {case.queries} query positions of {case.heads} query heads over "
                       f"{case.kv_heads} key heads are past the {QUERIES} rows a program holds",
          row=INHERENT),
    _Need('width', _power_width,
          lambda case: f"the heads are {case.head_dim} wide, not a power of two from 16 to 256",
          row='roadmap: decode kernel widths'),
    _Need('capacity', lambda case: case.keys % BLOCK == 0,
          lambda case: f"the cache holds {case.keys} keys, not whole blocks of {BLOCK}",
          row='roadmap: decode kernel capacities'),
    _Need('dtype', lambda case: case.dtype == case.key_dtype and case.dtype in DTYPES,
          lambda case: f"the queries are {case.dtype} over {case.key_dtype} keys; the kernel takes one of "
                       "fp32, bf16 and fp16 on both, float64 going to the reference path", row=INHERENT),
    _Need('arithmetic', lambda case: not case.arithmetic,
          "the call asks for arithmetic only the reference path computes", row=INHERENT),
)
"""The decode kernel's requirements of a call the decode selector serves (`dew.nn.attention_kernels.DECODE`,
over an `AttentionCase`): a GPU Dew's Triton kernels run on, at most `QUERIES` positions, a
power-of-two width from 16 to 256, a capacity of whole blocks and one of its dtypes on both sides."""


def _spec(scale: float, softcap: float | None, natural: bool, dtype, sinks: bool) -> algebra.Forward:
    """The algebra's facts for a decode launch: each query block's keys its row's span (`spans`), no
    documents, the cap as its score hook, the logits `natural` or not (`algebra.Forward.base2`), and
    whether a row's sums of `dtype` values need a guard (`algebra.headroom`). A decode
    block's rows are a key head's query heads and positions (`grouped_query`), so the hook takes no query
    positions."""
    return algebra.Forward("spans", 0, ids=False, scale=scale,
                           score_mod=None if softcap is None else functools.partial(capped, softcap),
                           natural=natural, headroom=natural and algebra.headroom(dtype),
                           initial=sinks)


def _attend(query: jax.Array, span: tuple, read, block: int, spec: algebra.Forward, steps,
            interpret: bool, sink: jax.Array | None = None) -> algebra.Partial:
    """The decode mainloop: `query` `[QUERIES, D]` over the keys of `steps`, a range of the loop's steps
    of `block` keys, those of the row's `span` (its first key and its length) alone, `read(step)` giving
    a step's keys and values `[block, D]`, as a softmax state (`algebra.Partial`). A `sink` `[QUERIES]`
    (each query row's value-free logit, -inf for a slice that does not carry it) starts the softmax as
    its first term."""
    first_key, length = span
    rows = jnp.arange(QUERIES)
    spans = ((jnp.full((QUERIES,), first_key), jnp.full((QUERIES,), length)),)

    def body(step, state):
        key, value = read(step)
        coords = algebra.Coords(rows[:, None], step * block + jnp.arange(block)[None, :], spans)
        logits = algebra.modify(spec, kernel_dot(query, key.T, interpret), coords)
        keys = algebra.Keys(coords, logits, lambda: value, functools.partial(kernel_dot, interpret=interpret))
        return algebra.plain_step(spec, state, keys)

    def walk(live=None) -> algebra.Partial:
        # The plain walk where `live` is None, else the guarded one, those rows shifting.
        start = algebra.empty(QUERIES, query.shape[-1])
        if sink is not None:
            real = sink > -jnp.inf
            start = algebra.resumed(spec, sink, real.astype(jnp.float32), start.acc, live=live)
        elif live is not None:
            start = algebra.headroomed(start, live)
        first, last = steps
        return jax.lax.fori_loop(first, last, body, start)

    if not spec.headroom:
        return walk()
    # A slice's state carries a bound on its values (`algebra.headroomed`), so a merge of slices each
    # finite but whose join overflows shifts as it needs (`algebra.merge`).
    return algebra.headroomed(algebra.guarded(walk, walk))


def _slice(span: tuple, block: int, split, splits: int) -> tuple:
    """The loop's steps of slice `split` of `splits` over a row's `span` of keys (its first and its
    length): whole steps of `block` keys, as even as they cut, the last slices empty where the span
    holds fewer steps than slices. The counts take the span's integer type, as `pl.cdiv` asks of both
    its operands, which a Python int is not where x64 is on."""
    step, slices = (jnp.asarray(count, span[1].dtype) for count in (block, splits))
    first, last = span[0] // step, pl.cdiv(span[1], step)
    per = pl.cdiv(last - first, slices)
    begin = first + split * per
    return begin, jnp.minimum(begin + per, last)


def _write(out_refs, spec: algebra.Forward, state: algebra.Partial):
    """The row's output normalized where one slice reads it all, else the slice's state, which
    `_combined` merges."""
    if len(out_refs) == 1:
        out = algebra.finish(spec, state, initial=True)[0]
        out_refs[0][...] = out.astype(out_refs[0].dtype)
    else:
        # The state's sums, maxima and totals, with its shifts, bounds and small sums under headroom;
        # which rows shifted is the merge's own to find again (`algebra.merge`).
        for ref, value in zip(out_refs, algebra.exported(state), strict=True):
            ref[...] = value


def _first_slice_sink(sinks_ref) -> jax.Array | None:
    """The block's sinks where this program reads the row's first slice, which alone carries them, so a
    merge of the slices counts each once (`_combined`); -inf in every other."""
    if sinks_ref is None:
        return None
    return jnp.where(pl.program_id(2) == 0, sinks_ref[...], -jnp.inf)


def _kernel(spans_ref, query_ref, key_ref, value_ref, *refs, spec: algebra.Forward, splits: int,
            interpret: bool, sinks: bool):
    sinks_ref, out_refs = (refs[0], refs[1:]) if sinks else (None, refs)

    def read(step):
        keys = pl.ds(step * BLOCK, BLOCK)
        return key_ref[keys, :], value_ref[keys, :]

    span = spans_ref[0], spans_ref[1]
    steps = _slice(span, BLOCK, pl.program_id(2), splits)
    _write(out_refs, spec, _attend(query_ref[...], span, read, BLOCK, spec, steps, interpret,
                                   _first_slice_sink(sinks_ref)))


def _paged_kernel(spans_ref, table_ref, query_ref, key_ref, value_ref, *refs, pages: int,
                  spec: algebra.Forward, splits: int, interpret: bool, sinks: bool):
    sinks_ref, out_refs = (refs[0], refs[1:]) if sinks else (None, refs)
    page_size, width = key_ref.shape[-2:]

    def read(step):
        held = table_ref[pl.ds(step * pages, pages)]
        return (key_ref[held].reshape(pages * page_size, width),
                value_ref[held].reshape(pages * page_size, width))

    span = spans_ref[0], spans_ref[1]
    steps = _slice(span, pages * page_size, pl.program_id(2), splits)
    _write(out_refs, spec, _attend(query_ref[...], span, read, pages * page_size, spec, steps, interpret,
                                   _first_slice_sink(sinks_ref)))


def _combined(spec: algebra.Forward, *parts, axis: int):
    """The slices' states (`[..., splits, ..., QUERIES, D]` sums along `axis`, their maxima, totals,
    shifts and reaches) merged in slice order (`algebra.merge`), as the normalized output in fp32: an
    empty slice's state (a maximum of -inf) merges as nothing."""
    states = [algebra.imported(tuple(jnp.take(part, split, axis=axis) for part in parts))
              for split in range(parts[0].shape[axis])]
    return algebra.finish(spec, functools.reduce(functools.partial(algebra.merge, spec), states),
                          initial=True)[0]


def _outputs(shape: tuple[int, ...], dtype, splits: int, at: int, spec, summed, reduced, *, headroom: bool):
    """The launch's outputs and their blocks: the output `shape` in `dtype` for one slice, else the
    slices' fp32 states (`algebra.Partial`: sums, maxima and totals, with shifts, reaches and the small
    weights' sums under headroom) with a slices axis at `at` (`spec`, `summed` and `reduced` the blocks)."""
    if splits == 1:
        return jax.ShapeDtypeStruct(shape, dtype), spec
    split = (*shape[:at], splits, *shape[at:])
    sums, rows = jax.ShapeDtypeStruct(split, jnp.float32), jax.ShapeDtypeStruct(split[:-1], jnp.float32)
    if not headroom:
        return [sums, rows, rows], [summed, reduced, reduced]
    return [sums, rows, rows, rows, rows, sums, rows], [summed, *[reduced] * 4, summed, reduced]


def _space(programs: int, steps: int, processors: int | None) -> set[Split]:
    """The slices a launch of `programs` programs a slice over `steps` steps of the loop may take on a
    GPU of `processors` multiprocessors (`GpuTarget.multiprocessors`): one where the programs already
    fill most of them or none are known, as FlashAttention 2 rules (csrc/flash_attn/flash_api.cpp,
    `num_splits_heuristic`), else any of `SPLITS` up to one a step."""
    if not processors or programs >= 0.8 * processors:
        return {Split(1)}
    return {Split(splits) for splits in SPLITS if splits <= min(steps, processors)}


RESIDENT = 6
"""Programs a multiprocessor runs at once that a launch fills (`_heuristic`)."""


def _heuristic(programs: int, steps: int, processors: int | None) -> Split:
    """The most slices a launch may take (`_space`) whose programs fill at most `RESIDENT` a
    multiprocessor. A program of `WARPS` warps over `BLOCK` keys a step leaves a multiprocessor room
    for several, which hide each other's reads. Over 60 shapes of Qwen3-0.6B's and Qwen3.5-0.8B's dense
    and paged decode (1 to 32 rows, 8K to 128K keys) timed at every count of `SPLITS`, this rule's pick
    ran within 1.010x of the fastest on an H100 (geometric mean; 1.22x at worst) and 1.037x on an A100
    (1.83x), the same for any `RESIDENT` from 5 to 7; FlashAttention 2's rule (the fewest whose waves
    run within 85% of the best count's) within 1.35x (2.44x) and 1.06x (1.83x). On the H100, Qwen3-0.6B
    at 2 rows over 131072 keys took 0.220 ms at this rule's 32 slices against 0.527 at FA2's 8."""
    counts = sorted(split.splits for split in _space(programs, steps, processors))
    if processors is None:
        return Split(counts[0])
    return Split(max((splits for splits in counts if programs * splits <= RESIDENT * processors),
                     default=counts[0]))


def _whole(rows: int, keys: int) -> jax.Array:
    """`rows` spans of every one of `keys` keys."""
    return jnp.stack([jnp.zeros(rows, jnp.int32), jnp.full(rows, keys, jnp.int32)], axis=1)


@dataclasses.dataclass(frozen=True)
class DecodeAttention(tuning.KernelOp):
    """`attend`'s launch over a contiguous cache: programs a row, a head and a slice."""

    config_cls: ClassVar[type[Split]] = Split

    ROWS: ClassVar[tuple[str, ...]] = ('query', 'key', 'value', 'spans')

    def operands(self, specs: list[jax.ShapeDtypeStruct], seed: int) -> list[jax.Array]:
        """Normal floats, each row's span its whole capacity."""
        arrays = super().operands(specs, seed)
        return [*arrays[:3], _whole(specs[3].shape[0], specs[1].shape[1]), *arrays[4:]]

    def _fwd(self, query: jax.Array, key: jax.Array, value: jax.Array, spans: jax.Array,
             sinks: jax.Array | None = None, *, interpret: bool, softcap: float | None = None,
             natural: bool = NATURAL, config: Split, return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        batch, _, heads, width = query.shape
        splits = config.splits
        keyed = pl.BlockSpec((None, key.shape[1], None, width), lambda row, head, split: (row, 0, head, 0))
        queried = pl.BlockSpec((None, QUERIES, None, width), lambda row, head, split: (row, 0, head, 0))
        sunk = [] if sinks is None else [pl.BlockSpec((None, QUERIES), lambda row, head, split: (head, 0))]
        summed = pl.BlockSpec((None, None, QUERIES, None, width),
                              lambda row, head, split: (row, split, 0, head, 0))
        reduced = pl.BlockSpec((None, None, QUERIES, None), lambda row, head, split: (row, split, 0, head))
        spec = _spec(1 / math.sqrt(width), softcap, natural, value.dtype, sinks is not None)
        out_shape, out_specs = _outputs(query.shape, query.dtype, splits, 1, queried, summed, reduced,
                                        headroom=spec.headroom)
        out = pl.pallas_call(
            lambda *refs: _kernel(*refs, spec=spec, splits=splits, interpret=interpret,
                                  sinks=sinks is not None),
            grid=(batch, heads, splits),
            in_specs=[pl.BlockSpec((None, 2), lambda row, head, split: (row, 0)), queried, keyed, keyed,
                      *sunk],
            out_specs=out_specs,
            out_shape=out_shape,
            compiler_params=plgpu.CompilerParams(num_warps=WARPS, num_stages=2),
            interpret=interpret, name="decode_attention",
        )(spans, query, key, value, *([] if sinks is None else [sinks]))
        return (out if splits == 1 else _combined(spec, *out, axis=1).astype(query.dtype)), None

    def _get_heuristics_config(self, ba) -> Split:
        return _heuristic(*self._launch(ba))

    @staticmethod
    def _launch(ba) -> tuple[int, int, int | None]:
        """Its programs a slice at its rows' next power of two (`_rows`), a row's steps of the loop, and
        the GPU's multiprocessors."""
        query = ba.arguments['query']
        return (_rows(query) * query.shape[2], pl.cdiv(ba.arguments['key'].shape[1], BLOCK),
                gpu_target().multiprocessors)


@dataclasses.dataclass(frozen=True)
class PagedDecodeAttention(tuning.KernelOp):
    """`attend_paged`'s launch: programs a row, a key head and a slice."""

    config_cls: ClassVar[type[Split]] = Split

    ROWS: ClassVar[tuple[str, ...]] = ('query', 'table', 'spans')

    def operands(self, specs: list[jax.ShapeDtypeStruct], seed: int) -> list[jax.Array]:
        """Normal floats, each row's table naming pages of the pool in turn and its span the whole
        table."""
        arrays = super().operands(specs, seed)
        (rows, per_row), count, page_size = specs[3].shape, specs[1].shape[1], specs[1].shape[2]
        table = (jnp.arange(rows * per_row, dtype=jnp.int32) % count).reshape(rows, per_row)
        return [*arrays[:3], table, _whole(rows, per_row * page_size), *arrays[5:]]

    def _fwd(self, query: jax.Array, key_pages: jax.Array, value_pages: jax.Array, table: jax.Array,
             spans: jax.Array, sinks: jax.Array | None = None, *, interpret: bool,
             softcap: float | None = None, natural: bool = NATURAL, config: Split,
             return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        rows, kv_heads, _, width = query.shape
        count, page_size = key_pages.shape[1:3]
        splits, pages = config.splits, BLOCK // page_size
        pooled = pl.BlockSpec((None, count, page_size, width), lambda row, head, split: (head, 0, 0, 0))
        queried = pl.BlockSpec((None, None, QUERIES, width), lambda row, head, split: (row, head, 0, 0))
        summed = pl.BlockSpec((None, None, None, QUERIES, width),
                              lambda row, head, split: (row, head, split, 0, 0))
        reduced = pl.BlockSpec((None, None, None, QUERIES), lambda row, head, split: (row, head, split, 0))
        sunk = [] if sinks is None else [pl.BlockSpec((None, QUERIES), lambda row, head, split: (head, 0))]
        spec = _spec(1 / math.sqrt(width), softcap, natural, value_pages.dtype, sinks is not None)
        out_shape, out_specs = _outputs(query.shape, query.dtype, splits, 2, queried, summed, reduced,
                                        headroom=spec.headroom)
        out = pl.pallas_call(
            lambda *refs: _paged_kernel(*refs, pages=pages, spec=spec, splits=splits, interpret=interpret,
                                        sinks=sinks is not None),
            grid=(rows, kv_heads, splits),
            in_specs=[pl.BlockSpec((None, 2), lambda row, head, split: (row, 0)),
                      pl.BlockSpec((None, table.shape[1]), lambda row, head, split: (row, 0)),
                      queried, pooled, pooled, *sunk],
            out_specs=out_specs,
            out_shape=out_shape,
            compiler_params=plgpu.CompilerParams(num_warps=WARPS, num_stages=2),
            interpret=interpret, name="paged_decode_attention",
        )(spans, table, query, key_pages, value_pages, *([] if sinks is None else [sinks]))
        return (out if splits == 1 else _combined(spec, *out, axis=2).astype(query.dtype)), None

    def _get_heuristics_config(self, ba) -> Split:
        return _heuristic(*self._launch(ba))

    @staticmethod
    def _launch(ba) -> tuple[int, int, int | None]:
        """Its programs a slice at its rows' next power of two (`_rows`), a row's steps of the loop over
        its table, `BLOCK` keys a step, and the GPU's multiprocessors."""
        query = ba.arguments['query']
        steps = pl.cdiv(ba.arguments['table'].shape[1], BLOCK // ba.arguments['key_pages'].shape[2])
        return _rows(query) * query.shape[1], steps, gpu_target().multiprocessors


_DECODE = DecodeAttention()
_PAGED_DECODE = PagedDecodeAttention()


def _rows(query: jax.Array) -> int:
    """The query's rows at the next power of two, where a launch's slices are read (`_heuristic`), so one
    count of slices serves every batch up to it."""
    return pl.next_power_of_2(query.shape[0])


def _spans(lengths: jax.Array, starts: jax.Array | None) -> jax.Array:
    """Each row's first key and its length, `[rows, 2]` int32."""
    lengths = jnp.asarray(lengths, jnp.int32)
    starts = jnp.zeros_like(lengths) if starts is None else jnp.asarray(starts, jnp.int32)
    return jnp.stack([starts, lengths], axis=1)


def grouped_query(attend, query, key, value):
    """`attend` over `query` `[B, T, N, D]` with each key head's `N // K` query heads as its query
    positions, `[B, T * N // K, K, D]` over the `K` key heads, and its output back in the query's
    layout: a decode step's lone query, whose heads read the same keys (key lengths alone end a
    row's), reads its key head's keys once a group. cuDNN otherwise padded the lone query to two
    positions and ran each query head on its own, 4.7% slower a layer at 64 rows for the same bits
    (docs/performance.md), and the decode kernel takes as many query heads as key heads
    (`decode_attention.SUPPORTS`)."""
    batch, positions, heads, width = query.shape
    kv_heads = key.shape[-2]
    group = heads // kv_heads
    if group == 1:
        return attend(query, key, value)
    grouped = query.reshape(batch, positions, kv_heads, group, width).transpose(0, 1, 3, 2, 4)
    out = attend(grouped.reshape(batch, positions * group, kv_heads, width), key, value)
    out = out.reshape(batch, positions, group, kv_heads, out.shape[-1]).transpose(0, 1, 3, 2, 4)
    return out.reshape(batch, positions, heads, out.shape[-1])


def attend(query: jax.Array, key: jax.Array, value: jax.Array, lengths: jax.Array, starts=None, *,
           softcap: float | None = None, sinks: jax.Array | None = None, natural: bool = NATURAL,
           interpret: bool = False) -> jax.Array:
    """Attention of `query` `[B, T, N, D]` over the keys from `starts` `[B]`
    (0 where None) to `lengths` `[B]` of `key` and `value` `[B, S, K, D]`, a
    query head reading the key head it shares with `N // K` others, `[B, T, N,
    D]` out, `interpret` running it on the CPU. A row reads only the blocks its
    keys fill, so a window over a long cache costs the window. Each row's start
    is below its length. A key head's query heads are its query positions, so
    the call's positions times its group are at most `QUERIES`. A `softcap`
    bounds the scaled logits (`softcap · tanh(logit / softcap)`, Gemma 2's);
    `sinks` `[N]` put one value-free logit a head in each row's softmax;
    `natural` keeps the logits natural (`algebra.Forward.base2`). Its
    Pallas call has no derivative: `dew.nn.attention` differentiates it as the
    attention it stands in for."""
    sunk = None if sinks is None else _row_sinks(sinks, key.shape[2], query.shape[1])

    def launched(grouped, key, value):
        # A key head's query heads as its query positions, `t * group + g` for head `k * group + g`.
        rows = grouped.shape[1]
        padded = jnp.pad(grouped, ((0, 0), (0, QUERIES - rows), (0, 0), (0, 0)))
        out = _DECODE(padded, key, value, _spans(lengths, starts), sunk, interpret=interpret, softcap=softcap,
                      natural=natural)
        return out[:, :rows]

    return grouped_query(launched, query, key, value)


def _row_sinks(sinks: jax.Array, kv_heads: int, positions: int) -> jax.Array:
    """Each key head's query rows' sinks `[kv_heads, QUERIES]` from the query heads' `sinks` `[N]`: row
    `t * group + g` of key head `k` is query head `k * group + g`'s, a row past the call's -inf, which
    no softmax counts."""
    rows = jnp.tile(sinks.astype(jnp.float32).reshape(kv_heads, -1), (1, positions))
    return jnp.pad(rows, ((0, 0), (0, QUERIES - rows.shape[1])), constant_values=-jnp.inf)


def _paged_geometry(case: "PagedCase") -> bool:
    return (case.heads % case.kv_heads == 0 and case.heads // case.kv_heads <= QUERIES
            and _widths(case.head_dim) and case.page_size is not None
            and case.page_size & (case.page_size - 1) == 0 and case.page_size <= BLOCK)


PAGED_GEOMETRY = Requirement["PagedCase"](
    'geometry', _paged_geometry,
    "the paged decode kernel reads at most `QUERIES` query heads a key head, a power-of-two width from 16 "
    "to 256 and a power-of-two page of at most `BLOCK` keys", row='roadmap: decode kernel widths')
"""What `attend_paged` needs of a paged decode step (`dew.nn.kv_cache.PagedCase`), which the paged
selector's GPU kernel declares (`dew.nn.kv_cache.TRITON_PAGED_SUPPORTS`)."""


def attend_paged(query: jax.Array, key_pages: jax.Array, value_pages: jax.Array, table: jax.Array,
                 lengths: jax.Array, starts=None, *, softcap: float | None = None,
                 sinks: jax.Array | None = None, natural: bool = NATURAL,
                 interpret: bool = False) -> jax.Array:
    """Attention of one query a row `[rows, heads, D]` over the keys from
    `starts` `[rows]` (0 where None) to `lengths` `[rows]` of that row's pages,
    `table` `[rows, per_row]` naming them in the `[kv_heads, pages, page_size,
    D]` pools; a query head reads the key head it shares with `heads //
    kv_heads` others; a `softcap` bounds the scaled logits and `sinks` `[heads]`
    join each row's softmax and `natural` keeps the logits natural as `attend`'s.
    `[rows, heads, D]` out, `interpret`
    running it on the CPU."""
    rows, heads, width = query.shape
    kv_heads, _, page_size, _ = key_pages.shape
    group, pages = heads // kv_heads, BLOCK // page_size
    grouped = jnp.pad(query.reshape(rows, kv_heads, group, width),
                      ((0, 0), (0, 0), (0, QUERIES - group), (0, 0)))
    table = jnp.pad(table, ((0, 0), (0, -table.shape[1] % pages)))
    sunk = None if sinks is None else _row_sinks(sinks, kv_heads, 1)
    out = _PAGED_DECODE(grouped, key_pages, value_pages, table, _spans(lengths, starts), sunk,
                        interpret=interpret, softcap=softcap, natural=natural)
    return out[:, :, :group].reshape(rows, heads, width)
