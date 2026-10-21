"""The attention template's Pallas Triton implementations (`dew.nn.kernels.templates.attention`): the
forward (`TritonForward`, `forward`), over any `KeySpans` and hooks, query heads grouped over their key
heads.

A program takes `block_q` queries of one batch row and query head and walks the key blocks its query
block visits (`BlockMap`): the edges with the visibility predicate, the whole tiles between without it
(their ids still compared where the call packs documents). Its online softmax starts from the hooks'
initial state, `score_mod` traces into the score tile and `output_mod` into the normalized output tile,
so a hook adds its own ops and nothing else. Heads narrower than a power of two ride in zero-padded,
and the call's queries and keys pad to whole blocks, which their spans leave unread.
"""

import dataclasses
import functools
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import tokamax
from jax.experimental import pallas as pl

from dew.nn.kernels import generation
from dew.nn.kernels.templates import cost as costs
from dew.nn.kernels.templates.attention import Hooks, KeySpans, Kind, Param, block_map
from dew.nn.kernels.templates.base import Template

DTYPES = (jnp.float32, jnp.bfloat16, jnp.float16)
"""The operand dtypes the kernels take, one on queries, keys and values alike."""


@dataclasses.dataclass(frozen=True, order=True)
class Tiles:
    """A forward program's tile, `block_q` queries by `block_k` keys, its `warps` and the pipeline's
    `stages`."""

    block_q: int
    block_k: int
    warps: int
    stages: int


SPACE = tuple(Tiles(block_q, block_k, warps, stages) for block_q in (32, 64, 128) for block_k in (32, 64, 128)
              for warps in (4, 8) for stages in (2, 3))
"""Every config the forward is proven over (tests/test_attention_triton.py pins each)."""


def _dot(a: jax.Array, b: jax.Array, interpret: bool) -> jax.Array:
    """`a @ b` accumulated in fp32: bf16's and fp16's own product, IEEE fp32 for fp32 operands;
    interpreted, the CPU's own product."""
    if interpret:
        return jnp.dot(a, b, preferred_element_type=jnp.float32)
    from jax.experimental.pallas import triton as plgpu

    return plgpu.dot(a, b, precision=jax.lax.Precision.HIGHEST).astype(jnp.float32)


def _power(width: int) -> int:
    """The power of two from 16 that holds `width`."""
    return max(16, 1 << (width - 1).bit_length())


def _whole(count: int, block: int) -> int:
    return -(-count // block) * block


def _tile_param(ref, kind: Kind, head, queries: jax.Array, keys: jax.Array | None):
    """A hook parameter's values for a tile, as `kind` lays them out: the head's scalar, the tile's
    queries' `[block_q, 1]`, its keys' `[1, block_k]`, its entries' or its output rows'."""
    match kind:
        case Kind.HEAD:
            return ref[head]
        case Kind.QUERY:
            return ref[...][:, None]
        case Kind.KEY:
            return ref[keys][None, :]
        case Kind.TILE:
            return ref[:, keys]
        case Kind.OUTPUT:
            return ref[...]


def _forward_kernel(bounds_ref, start_ref, end_ref, *refs, ids: bool, score_kinds: tuple[Kind, ...],
                    output_kinds: tuple[Kind, ...], initial: bool, residuals: bool, score_mod, output_mod,
                    scale: float, block_k: int, interpret: bool):
    refs = list(refs)
    query_ids_ref, key_ids_ref = (refs.pop(0), refs.pop(0)) if ids else (None, None)
    query_ref, key_ref, value_ref = refs.pop(0), refs.pop(0), refs.pop(0)
    score_refs = [refs.pop(0) for _ in score_kinds]
    output_refs = [refs.pop(0) for _ in output_kinds]
    m0_ref, total0_ref, acc0_ref = (refs.pop(0), refs.pop(0), refs.pop(0)) if initial else (None, None, None)
    out_ref, *residual_refs = refs
    block_q = query_ref.shape[0]
    head = pl.program_id(2)
    first_query = pl.program_id(0) * block_q
    query = query_ref[...]
    start, end = start_ref[...], end_ref[...]
    query_ids = query_ids_ref[...] if ids else None
    q_pos = first_query + jnp.arange(block_q)[:, None]

    def step(index, carry, *, masked: bool):
        acc, peak, total = carry
        keys = pl.ds(index * block_k, block_k)
        key, value = key_ref[keys, :], value_ref[keys, :]
        logits = _dot(query, key.T, interpret) * scale
        k_pos = index * block_k + jnp.arange(block_k)[None, :]
        if score_mod is not None:
            params = [_tile_param(ref, kind, head, q_pos, keys) for ref, kind in zip(score_refs, score_kinds,
                                                                                     strict=True)]
            logits = score_mod(logits, q_pos, k_pos, *params)
        seen = None
        if masked:
            seen = (k_pos >= start[:, None]) & (k_pos < end[:, None])
        if ids:
            same = query_ids[:, None] == key_ids_ref[keys][None, :]
            seen = same if seen is None else seen & same
        if seen is not None:
            logits = jnp.where(seen, logits, -jnp.inf)
        new_peak = jnp.maximum(peak, logits.max(axis=-1))
        safe = jnp.where(new_peak == -jnp.inf, 0.0, new_peak)
        correction = jnp.exp(peak - safe)
        weights = jnp.exp(logits - safe[:, None])
        acc = correction[:, None] * acc + _dot(weights.astype(value.dtype), value, interpret)
        return acc, new_peak, correction * total + weights.sum(axis=-1)

    width = value_ref.shape[-1]
    if initial:
        carry = (acc0_ref[...].astype(jnp.float32), m0_ref[...].astype(jnp.float32),
                 total0_ref[...].astype(jnp.float32))
    else:
        carry = (jnp.zeros((block_q, width), jnp.float32), jnp.full((block_q,), -jnp.inf, jnp.float32),
                 jnp.zeros((block_q,), jnp.float32))
    lo, hi, full_lo, full_hi = bounds_ref[0], bounds_ref[1], bounds_ref[2], bounds_ref[3]
    carry = jax.lax.fori_loop(lo, full_lo, functools.partial(step, masked=True), carry)
    carry = jax.lax.fori_loop(full_lo, full_hi, functools.partial(step, masked=False), carry)
    acc, peak, total = jax.lax.fori_loop(full_hi, hi, functools.partial(step, masked=True), carry)
    out = jnp.where(total[:, None] > 0, acc / jnp.where(total > 0, total, 1.0)[:, None], 0.0)
    if output_mod is not None:
        params = [_tile_param(ref, kind, head, q_pos, None) for ref, kind in zip(output_refs, output_kinds,
                                                                               strict=True)]
        out = output_mod(out, *params)
    out_ref[...] = out.astype(out_ref.dtype)
    if residuals:
        # A query that saw nothing from no state holds 0 for both, as the reference's residuals do.
        residual_refs[0][...] = jnp.where(peak == -jnp.inf, 0.0, peak)
        residual_refs[1][...] = total


@dataclasses.dataclass(frozen=True)
class TritonForward(Template[Tiles]):
    """The template's forward in Pallas Triton over padded operands (`forward` pads them): queries
    `[B, Qp, H, Dp]` over keys `[B, Kp, Hk, Dp]` and values `[B, Kp, Hk, Dvp]`, each query's keys
    `start` to `end` `[B, Qp]` and, with documents, its id `[B, Qp]` against the keys' `[B, Kp]`; the
    score and output hooks' parameters (`score_params`, `output_params`, laid out by `score_kinds` and
    `output_kinds`) and the softmax's initial state. Returns the output `[B, Qp, H, Dvp]` and, with
    `residuals`, each query's running maximum and denominator `[B, H, Qp]`, fp32."""

    name = "triton"
    config_cls = Tiles
    score_mod: Callable | None = None
    output_mod: Callable | None = None
    score_kinds: tuple[Kind, ...] = ()
    output_kinds: tuple[Kind, ...] = ()
    scale: float = 1.0

    def _fwd(self, query, key, value, start, end, query_ids, key_ids, score_params, output_params, m0, total0,
             acc0, *, residuals: bool, interpret: bool, config: Tiles, return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        batch, queries, heads, width = query.shape
        keys, kv_heads, value_width = key.shape[1], key.shape[2], value.shape[-1]
        group = heads // kv_heads
        block_q, block_k = config.block_q, config.block_k
        spans = KeySpans(start, end, query_ids, key_ids, keys)
        blocks = block_map(spans, block_q, block_k)
        bounds = jnp.stack([blocks.lo, blocks.hi, blocks.full_lo, blocks.full_hi], axis=-1).astype(jnp.int32)
        ids = query_ids is not None
        initial = m0 is not None
        per_query = pl.BlockSpec((None, block_q), lambda i, b, h: (b, i))
        whole_keys = pl.BlockSpec((None, keys), lambda i, b, h: (b, 0))
        in_specs = [pl.BlockSpec((None, None, 4), lambda i, b, h: (b, i, 0)), per_query, per_query]
        arrays = [bounds, start, end]
        if ids:
            in_specs += [per_query, whole_keys]
            arrays += [query_ids, key_ids]
        in_specs += [pl.BlockSpec((None, block_q, None, width), lambda i, b, h: (b, i, h, 0)),
                     pl.BlockSpec((None, keys, None, width), lambda i, b, h: (b, 0, h // group, 0)),
                     pl.BlockSpec((None, keys, None, value_width), lambda i, b, h: (b, 0, h // group, 0))]
        arrays += [query, key, value]
        for param, kind in [*zip(score_params, self.score_kinds, strict=True),
                            *zip(output_params, self.output_kinds, strict=True)]:
            in_specs.append(_param_spec(kind, param, block_q, keys))
            arrays.append(param)
        if initial:
            in_specs += [pl.BlockSpec((None, None, block_q), lambda i, b, h: (b, h, i))] * 2
            in_specs.append(pl.BlockSpec((None, None, block_q, value_width), lambda i, b, h: (b, h, i, 0)))
            arrays += [m0, total0, acc0]
        out_shape = [jax.ShapeDtypeStruct((batch, queries, heads, value_width), query.dtype)]
        out_specs = [pl.BlockSpec((None, block_q, None, value_width), lambda i, b, h: (b, i, h, 0))]
        if residuals:
            out_shape += [jax.ShapeDtypeStruct((batch, heads, queries), jnp.float32)] * 2
            out_specs += [pl.BlockSpec((None, None, block_q), lambda i, b, h: (b, h, i))] * 2
        kernel = functools.partial(
            _forward_kernel, ids=ids, score_kinds=self.score_kinds, output_kinds=self.output_kinds,
            initial=initial, residuals=residuals, score_mod=self.score_mod, output_mod=self.output_mod,
            scale=self.scale, block_k=block_k, interpret=interpret)
        out = pl.pallas_call(
            kernel, grid=(queries // block_q, batch, heads), in_specs=in_specs, out_specs=out_specs,
            out_shape=out_shape, interpret=interpret, name="attention_forward",
            compiler_params=plgpu.CompilerParams(num_warps=config.warps, num_stages=config.stages),
        )(*arrays)
        return (tuple(out) if residuals else out[0]), None

    def space(self, ba: tokamax.BoundArguments) -> Sequence[Tiles]:
        queries, keys = ba.arguments["query"].shape[1], ba.arguments["key"].shape[1]
        return tuple(tiles for tiles in SPACE if queries % tiles.block_q == 0 and keys % tiles.block_k == 0)

    def staged_bytes(self, ba: tokamax.BoundArguments, config: Tiles) -> int:
        """The query tile, and each pipeline stage's key and value tiles, in the operands' dtype."""
        query, value = ba.arguments["query"], ba.arguments["value"]
        size = jnp.dtype(query.dtype).itemsize
        width, value_width = query.shape[-1], value.shape[-1]
        return (config.block_q * width + config.stages * config.block_k * (width + value_width)) * size

    def cost(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget) -> float:
        """Every key tile visited (the spans are traced), each a `block_q` by `block_k` product of the
        queries with the keys and of the weights with the values, and as many exponentials; the operands
        read once and the output written once."""
        query, key, value = (ba.arguments[name] for name in ("query", "key", "value"))
        batch, queries, heads, width = query.shape
        keys, value_width = key.shape[1], value.shape[-1]
        tiles = keys // config.block_k
        size = jnp.dtype(query.dtype).itemsize
        return costs.seconds(costs.Cost(
            programs=batch * heads * (queries // config.block_q),
            device_bytes=(query.size + key.size + value.size + batch * queries * heads * value_width) * size,
            program_flops=tiles * 2.0 * config.block_q * config.block_k * (width + value_width),
            program_staged=float(self.staged_bytes(ba, config)) * tiles,
            program_exponentials=float(tiles * config.block_q * config.block_k),
            mode='tf32' if query.dtype == jnp.float32 else '16bit'), target)

    def eligible(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
        query, key, value = (ba.arguments[name] for name in ("query", "key", "value"))
        return (generation.capability(target.generation) is not None
                and query.dtype == key.dtype == value.dtype and query.dtype in DTYPES)


def _param_spec(kind: Kind, param: jax.Array, block_q: int, keys: int) -> pl.BlockSpec:
    """How a program reads a hook parameter of `kind`: a head's scalar from the whole `[H]`, its
    queries' or its output rows' block, every key of its row and head."""
    match kind:
        case Kind.HEAD:
            return pl.BlockSpec(param.shape, lambda i, b, h: (0,))
        case Kind.QUERY:
            return pl.BlockSpec((None, None, block_q), lambda i, b, h: (b, h, i))
        case Kind.KEY:
            return pl.BlockSpec((None, None, keys), lambda i, b, h: (b, h, 0))
        case Kind.TILE:
            return pl.BlockSpec((None, None, block_q, keys), lambda i, b, h: (b, h, i, 0))
        case Kind.OUTPUT:
            return pl.BlockSpec((None, block_q, None, param.shape[-1]), lambda i, b, h: (b, i, h, 0))


def _padded(param: Param, queries: int, keys: int, value_width: int) -> jax.Array:
    """A hook parameter at the padded call's shape: zeros past its queries, keys and value width, and a
    head vector at a power of two."""
    value = param.value
    match param.kind:
        case Kind.HEAD:
            return jnp.pad(value, (0, _power(value.shape[0]) - value.shape[0]))
        case Kind.QUERY:
            return jnp.pad(value, ((0, 0), (0, 0), (0, queries - value.shape[2])))
        case Kind.KEY:
            return jnp.pad(value, ((0, 0), (0, 0), (0, keys - value.shape[2])))
        case Kind.TILE:
            return jnp.pad(value, ((0, 0), (0, 0), (0, queries - value.shape[2]), (0, keys - value.shape[3])))
        case Kind.OUTPUT:
            return jnp.pad(value, ((0, 0), (0, queries - value.shape[1]), (0, 0),
                                   (0, value_width - value.shape[3])))


def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, residuals: bool = False, interpret: bool = False):
    """The template's forward (`TritonForward`) on `query` `[B, Q, H, D]` over `key` and `value`
    `[B, K, Hk, D]` and `[B, K, Hk, Dv]` and the keys `spans` leaves each query: as
    `dew.nn.kernels.templates.attention.reference` computes it, with `residuals` its running maxima and
    denominators `[B, H, Q]` too. The operands pad to whole blocks of the largest tile and to power-of-two
    head widths; the padding is unread and sliced off. `interpret` runs it on the CPU."""
    hooks = Hooks() if hooks is None else hooks
    queries, width = query.shape[1], query.shape[3]
    keys, value_width = key.shape[1], value.shape[-1]
    block = max(max(tiles.block_q for tiles in SPACE), max(tiles.block_k for tiles in SPACE))
    padded_q, padded_k = _whole(queries, block), _whole(keys, block)
    padded_d, padded_dv = _power(width), _power(value_width)

    def pad(array, rows, columns):
        return jnp.pad(array, ((0, 0), (0, rows - array.shape[1]), (0, 0), (0, columns - array.shape[3])))

    def pad_rows(array, rows, fill=0):
        return None if array is None else jnp.pad(array, ((0, 0), (0, rows - array.shape[1])),
                                                    constant_values=fill)

    initial = hooks.initial
    m0 = total0 = acc0 = None
    if initial is not None:
        m0 = jnp.pad(initial.m, ((0, 0), (0, 0), (0, padded_q - queries)), constant_values=-jnp.inf)
        total0 = jnp.pad(initial.total, ((0, 0), (0, 0), (0, padded_q - queries)))
        acc0 = jnp.pad(initial.acc, ((0, 0), (0, 0), (0, padded_q - queries), (0, padded_dv - value_width)))
    op = TritonForward(score_mod=hooks.score_mod, output_mod=hooks.output_mod,
                       score_kinds=tuple(param.kind for param in hooks.score_params),
                       output_kinds=tuple(param.kind for param in hooks.output_params),
                       scale=width ** -0.5 if scale is None else scale)
    out = op(pad(query, padded_q, padded_d), pad(key, padded_k, padded_d), pad(value, padded_k, padded_dv),
             pad_rows(spans.start, padded_q), pad_rows(spans.end, padded_q),
             pad_rows(spans.query_ids, padded_q), pad_rows(spans.key_ids, padded_k),
             tuple(_padded(param, padded_q, padded_k, padded_dv) for param in hooks.score_params),
             tuple(_padded(param, padded_q, padded_k, padded_dv) for param in hooks.output_params),
             m0, total0, acc0, residuals=residuals, interpret=interpret)
    if not residuals:
        return out[:, :queries, :, :value_width]
    out, peak, total = out
    return out[:, :queries, :, :value_width], peak[:, :, :queries], total[:, :, :queries]

