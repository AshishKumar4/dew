"""The attention template's Pallas Triton implementations (`dew.nn.kernels.templates.attention`): the
forward (`TritonForward`, `forward`) and the backward (`TritonBackward`, `backward`), over any
`KeySpans` and hooks, query heads grouped over their key heads.

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

from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import cost as costs
from dew.nn.kernels.templates.attention import Hooks, Initial, KeySpans, Kind, Param, block_map
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


def _value_gradient(exponentials: jax.Array, weights: jax.Array, dout: jax.Array, total: jax.Array,
                    interpret: bool) -> jax.Array:
    """A tile's share of dV `[block_k, Dv]`, fp32: the unnormalized weights `exp(s - m)` narrowed to the
    operands' dtype as the reference's forward rounds them, against each query's `dout / total` in fp32,
    the reference's derivative's own order, so no small weight or share is lost to normalizing first.
    bf16 alone takes FlashAttention's form, its normalized weights `p` narrowed against the output's
    cotangent as it is, so the product stays on bf16's tensor cores: a query's cotangents that cancel
    across a key still cancel, and a normalized weight below bf16's least subnormal (about 9e-41) is
    dropped, which only a cotangent past about 1e28 would make count."""
    if dout.dtype == jnp.bfloat16:
        return _dot(weights.astype(dout.dtype).T, dout, interpret)
    scaled = dout.astype(jnp.float32) * _inverse(total)[:, None]
    return _dot(exponentials.astype(dout.dtype).astype(jnp.float32).T, scaled, interpret)


def _power(width: int) -> int:
    """The power of two from 16 that holds `width`."""
    return max(16, 1 << (width - 1).bit_length())


LOG2E = 1.4426950408889634
"""log2(e), which turns a natural exponent into a base-2 one."""


def _whole(count: int, block: int) -> int:
    return -(-count // block) * block


def _eligible(ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
    """Whether a launch runs on `target`: a CUDA GPU, one dtype of `DTYPES` on the queries, keys and
    values, and bf16 only from `generation.BF16_GPU` on, where Triton compiles its products."""
    query, key, value = (ba.arguments[name] for name in ("query", "key", "value"))
    capability = generation.capability(target.generation)
    return (capability is not None and query.dtype == key.dtype == value.dtype and query.dtype in DTYPES
            and (query.dtype != jnp.bfloat16 or 10 * capability[0] + capability[1] >= generation.BF16_GPU))


def _mode(dtype) -> str:
    """The arithmetic a launch's products take (`_dot`): IEEE fp32 for fp32 operands, else 16-bit
    products summed in fp32."""
    return 'fp32' if dtype == jnp.float32 else '16bit'


TRACED_TILE = 16
"""The tile a hook is traced at for its picks' key (`_hook_program`)."""


def _kept_dtype(dtype) -> str:
    """A hook parameter's dtype in the kernels: fp32 for a floating one (`_padded`), else its own."""
    return "float32" if jnp.issubdtype(dtype, jnp.floating) else jnp.dtype(dtype).name


def _hook_program(name: str, hook, kinds: tuple[Kind, ...], dtypes: tuple[str, ...], *,
                  differentiated: bool = False, width: int = TRACED_TILE) -> str | None:
    """What a score or output hook computes, as its launch's picks key it (`tuning.program`): a score
    hook traced at a `TRACED_TILE` square fp32 tile, an output hook at `TRACED_TILE` queries of the
    call's padded value `width`, the entries' indices and each parameter as its kind lays it out there,
    in its dtype in the kernels (`dtypes`); `differentiated`, its pullback too, so a hook whose
    derivative is its own rule (`jax.custom_vjp`) keys by that rule. None or an already traced key as
    it is."""
    if hook is None or isinstance(hook, str):
        return hook
    tile = TRACED_TILE
    layouts = {Kind.HEAD: (), Kind.QUERY: (tile, 1), Kind.KEY: (1, tile), Kind.TILE: (tile, tile),
               Kind.OUTPUT: (tile, width)}
    params = [jax.ShapeDtypeStruct(layouts[kind], dtype) for kind, dtype in zip(kinds, dtypes, strict=True)]
    square = jax.ShapeDtypeStruct((tile, tile), jnp.float32)
    if name == "output_mod":
        return tuning.program(hook, jax.ShapeDtypeStruct((tile, width), jnp.float32), *params)
    positions = (jax.ShapeDtypeStruct((tile, 1), jnp.int32), jax.ShapeDtypeStruct((1, tile), jnp.int32))
    if not differentiated:
        return tuning.program(hook, square, *positions, *params)
    floating = [index for index, param in enumerate(params) if jnp.issubdtype(param.dtype, jnp.floating)]

    def pulled(s, q_pos, k_pos, *values):
        def scored(s, *wide):
            merged = list(values)
            for index, value in zip(floating, wide, strict=True):
                merged[index] = value
            return hook(s, q_pos, k_pos, *merged)

        out, pull = jax.vjp(scored, s, *(values[index] for index in floating))
        return out, pull(jnp.ones_like(out))

    return tuning.program(pulled, square, *positions, *params)


def _spanned(specs: list[jax.ShapeDtypeStruct], arrays: list[jax.Array]) -> list[jax.Array]:
    """A launch's timing operands with its spans whole: of its integer operands, the first two each
    query's first key (0) and one past its last (every key, `specs[1]` being the keys), the rest its
    documents' ids (one document), so a timed launch visits every tile a dense call does."""
    integers = [index for index, spec in enumerate(specs) if jnp.issubdtype(spec.dtype, jnp.integer)]
    keys = specs[1].shape[1]
    for place, index in enumerate(integers):
        if place:
            arrays[index] = jnp.full(specs[index].shape, keys if place == 1 else 1, specs[index].dtype)
    return arrays


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


def _forward_kernel(*refs, form: str, keys: int, ids: bool, score_kinds: tuple[Kind, ...],
                    output_kinds: tuple[Kind, ...], initial: bool, residuals: bool, score_mod, output_mod,
                    scale: float, block_k: int, interpret: bool):
    """The forward's program, specialized at trace time to what the call holds (`KeySpans.form`): a
    full call walks every key tile with no mask, a causal one the tiles up to its queries' last with
    the diagonal's index mask, and only spans read their bounds and masks; no score hook, the logits
    stay in base 2 (log2(e) folded into the scale) for exp2; and only a call where a query can see
    nothing (spans, a score hook) guards its maxima and denominators against it."""
    refs = list(refs)
    spanned = form == "spans"
    bounds_ref, start_ref, end_ref = (refs.pop(0), refs.pop(0), refs.pop(0)) if spanned else (None,) * 3
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
    query_ids = query_ids_ref[...] if ids else None
    q_pos = first_query + jnp.arange(block_q)[:, None]
    base2 = score_mod is None
    guarded = spanned or score_mod is not None
    ragged = keys % block_k != 0

    def step(index, carry):
        acc, peak, total = carry
        tile = pl.ds(index * block_k, block_k)
        key, value = key_ref[tile, :], value_ref[tile, :]
        logits = _dot(query, key.T, interpret) * (scale * LOG2E if base2 else scale)
        k_pos = index * block_k + jnp.arange(block_k)[None, :]
        if score_mod is not None:
            params = [_tile_param(ref, kind, head, q_pos, tile) for ref, kind in zip(score_refs, score_kinds,
                                                                                     strict=True)]
            logits = score_mod(logits, q_pos, k_pos, *params)
        seen = None
        if spanned:
            seen = (k_pos >= start_ref[...][:, None]) & (k_pos < end_ref[...][:, None])
        elif form == "causal":
            seen = k_pos <= q_pos
        if ragged and not spanned:
            seen = k_pos < keys if seen is None else seen & (k_pos < keys)
        if ids:
            same = query_ids[:, None] == key_ids_ref[tile][None, :]
            seen = same if seen is None else seen & same
        if seen is not None:
            logits = jnp.where(seen, logits, -jnp.inf)
        new_peak = jnp.maximum(peak, logits.max(axis=-1))
        safe = jnp.where(new_peak == -jnp.inf, 0.0, new_peak) if guarded else new_peak
        power = jnp.exp2 if base2 else jnp.exp
        correction = power(peak - safe)
        weights = power(logits - safe[:, None])
        acc = correction[:, None] * acc + _dot(weights.astype(value.dtype), value, interpret)
        return acc, new_peak, correction * total + weights.sum(axis=-1)

    width = value_ref.shape[-1]
    if initial:
        m0 = m0_ref[...].astype(jnp.float32)
        carry = (acc0_ref[...].astype(jnp.float32), m0 * LOG2E if base2 else m0,
                 total0_ref[...].astype(jnp.float32))
    else:
        carry = (jnp.zeros((block_q, width), jnp.float32), jnp.full((block_q,), -jnp.inf, jnp.float32),
                 jnp.zeros((block_q,), jnp.float32))
    tiles = -(-keys // block_k)
    if spanned:
        lo, hi = bounds_ref[0], bounds_ref[1]
    elif form == "causal":
        lo, hi = 0, jnp.minimum(-(-(first_query + block_q) // block_k), tiles)
    else:
        lo, hi = 0, tiles
    acc, peak, total = jax.lax.fori_loop(lo, hi, step, carry)
    if base2:
        peak = peak / LOG2E
    if guarded or initial:
        out = jnp.where(total[:, None] > 0, acc / jnp.where(total > 0, total, 1.0)[:, None], 0.0)
    else:
        out = acc / total[:, None]
    if output_mod is not None:
        params = [_tile_param(ref, kind, head, q_pos, None) for ref, kind in zip(output_refs, output_kinds,
                                                                               strict=True)]
        out = output_mod(out, *params)
    out_ref[...] = out.astype(out_ref.dtype)
    if residuals:
        # A query that saw nothing from no state holds 0 for both, as the reference's residuals do.
        residual_refs[0][...] = jnp.where(peak == -jnp.inf, 0.0, peak) if guarded or initial else peak
        residual_refs[1][...] = total


@dataclasses.dataclass(frozen=True)
class TritonForward(Template[Tiles]):
    """The template's forward in Pallas Triton over padded operands (`forward` pads them): queries
    `[B, Qp, H, Dp]` over keys `[B, Kp, Hk, Dp]` and values `[B, Kp, Hk, Dvp]`, each query's keys
    `start` to `end` `[B, Qp]` and, with documents, its id `[B, Qp]` against the keys' `[B, Kp]`; the
    score and output hooks' parameters (`score_params`, `output_params`, laid out by `score_kinds` and
    `output_kinds`) and the softmax's initial state. Returns the output `[B, Qp, H, Dvp]` and, with
    `residuals`, each query's running maximum and denominator `[B, H, Qp]`, fp32."""

    HOOKS = ("score_mod", "output_mod")
    name = "triton"
    config_cls = Tiles
    score_mod: Callable | None = None
    output_mod: Callable | None = None
    form: str = "spans"
    keys: int = 0
    score_kinds: tuple[Kind, ...] = ()
    output_kinds: tuple[Kind, ...] = ()
    score_dtypes: tuple[str, ...] = ()
    output_dtypes: tuple[str, ...] = ()
    output_width: int = TRACED_TILE
    scale: float = 1.0

    def _fwd(self, query, key, value, start, end, query_ids, key_ids, score_params, output_params, m0, total0,
             acc0, *, residuals: bool, interpret: bool, config: Tiles, return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        batch, queries, heads, width = query.shape
        keys, kv_heads, value_width = key.shape[1], key.shape[2], value.shape[-1]
        group = heads // kv_heads
        block_q, block_k = config.block_q, config.block_k
        ids = query_ids is not None
        initial = m0 is not None
        per_query = pl.BlockSpec((None, block_q), lambda i, b, h: (b, i))
        whole_keys = pl.BlockSpec((None, keys), lambda i, b, h: (b, 0))
        in_specs, arrays = [], []
        if self.form == "spans":
            # Only spans read their bounds, which the block map gives each query block.
            blocks = block_map(KeySpans(start, end, query_ids, key_ids, keys), block_q, block_k)
            bounds = jnp.stack([blocks.lo, blocks.hi, blocks.full_lo, blocks.full_hi],
                               axis=-1).astype(jnp.int32)
            in_specs += [pl.BlockSpec((None, None, 4), lambda i, b, h: (b, i, 0)), per_query, per_query]
            arrays += [bounds, start, end]
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
            scale=self.scale, block_k=block_k, interpret=interpret, form=self.form, keys=self.keys or keys)
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
        """The shared memory a program holds: the query tile, and each pipeline stage's key and value
        tiles, in the operands' dtype."""
        query, value = ba.arguments["query"], ba.arguments["value"]
        size = jnp.dtype(query.dtype).itemsize
        width, value_width = query.shape[-1], value.shape[-1]
        return (config.block_q * width + config.stages * config.block_k * (width + value_width)) * size

    def cost(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget) -> float:
        """Every key tile of the call (a bound: the spans are traced, so a sparse call visits fewer),
        each staging its key and value tiles once and taking a `block_q` by `block_k` product of the
        queries with the keys and of the weights with the values, and as many exponentials; its inputs
        read once, the output written once in their dtype and the residuals in fp32."""
        query, key, value = (ba.arguments[name] for name in ("query", "key", "value"))
        batch, queries, heads, width = query.shape
        keys, value_width = key.shape[1], value.shape[-1]
        tiles = keys // config.block_k
        size = jnp.dtype(query.dtype).itemsize
        rows = 2 * batch * heads * queries * 4 if ba.arguments.get("residuals") else 0
        # Every array it reads at its dtype (operands, spans, hook parameters, initial state), and the
        # output and residuals it writes.
        read = sum(leaf.size * jnp.dtype(leaf.dtype).itemsize for leaf in jax.tree.leaves(dict(ba.arguments))
                   if hasattr(leaf, "dtype"))
        return costs.seconds(costs.Cost(
            programs=batch * heads * (queries // config.block_q),
            device_bytes=read + batch * queries * heads * value_width * size + rows,
            program_flops=tiles * 2.0 * config.block_q * config.block_k * (width + value_width),
            program_staged=float(config.block_q * width * size
                                 + tiles * config.block_k * (width + value_width) * size),
            program_exponentials=float(tiles * config.block_q * config.block_k),
            mode=_mode(query.dtype)), target)

    def eligible(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
        return _eligible(ba, target)

    def operands(self, specs: list[jax.ShapeDtypeStruct], seed: int) -> list[jax.Array]:
        return _spanned(specs, super().operands(specs, seed))

    def traced(self, name: str, value: object) -> object:
        if name == "output_mod":
            return _hook_program(name, value, self.output_kinds, self.output_dtypes, width=self.output_width)
        return _hook_program(name, value, self.score_kinds, self.score_dtypes)


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
    """A hook parameter at the padded call's shape, in fp32 as the kernels compute (its gradient returns
    in its own dtype): zeros past its queries, keys and value width, and a head vector at a power of
    two."""
    value = param.value
    if jnp.issubdtype(value.dtype, jnp.floating):
        value = value.astype(jnp.float32)
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


@dataclasses.dataclass(frozen=True)
class _Padded:
    """A call at the launches' padded shapes (`_padded_call`): its queries and keys at whole blocks of
    the largest tile (at least one, so a call of no keys still has a tile to trace), its head widths at
    powers of two, its spans' rows and its score and output hooks' parameters alike; the padding is
    unread and sliced off."""

    query: jax.Array
    key: jax.Array
    value: jax.Array
    start: jax.Array
    end: jax.Array
    query_ids: jax.Array | None
    key_ids: jax.Array | None
    score_params: tuple[jax.Array, ...]
    output_params: tuple[jax.Array, ...]
    queries: int
    keys: int

    def rows(self, array: jax.Array, fill: float = 0.0) -> jax.Array:
        """A `[B, H, Q, ...]` row statistic at the padded queries."""
        pad = [(0, 0), (0, 0), (0, self.query.shape[1] - array.shape[2]), *[(0, 0)] * (array.ndim - 3)]
        return jnp.pad(array, pad, constant_values=fill)


def _padded_call(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans,
                 hooks: Hooks) -> _Padded:
    """`forward`'s and `backward`'s one padding of a call (`_Padded`)."""
    queries, width = query.shape[1], query.shape[3]
    keys, value_width = key.shape[1], value.shape[-1]
    block = max(max(tiles.block_q for tiles in SPACE), max(tiles.block_k for tiles in SPACE))
    padded_q, padded_k = _whole(max(queries, 1), block), _whole(max(keys, 1), block)
    padded_d, padded_dv = _power(width), _power(value_width)

    def pad(array, rows, columns):
        return jnp.pad(array, ((0, 0), (0, rows - array.shape[1]), (0, 0), (0, columns - array.shape[3])))

    def pad_rows(array, rows):
        return None if array is None else jnp.pad(array, ((0, 0), (0, rows - array.shape[1])))

    return _Padded(pad(query, padded_q, padded_d), pad(key, padded_k, padded_d),
                   pad(value, padded_k, padded_dv), pad_rows(spans.start, padded_q),
                   pad_rows(spans.end, padded_q), pad_rows(spans.query_ids, padded_q),
                   pad_rows(spans.key_ids, padded_k),
                   tuple(_padded(param, padded_q, padded_k, padded_dv) for param in hooks.score_params),
                   tuple(_padded(param, padded_q, padded_k, padded_dv) for param in hooks.output_params),
                   queries, keys)


def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, residuals: bool = False, interpret: bool = False):
    """The template's forward (`TritonForward`) on `query` `[B, Q, H, D]` over `key` and `value`
    `[B, K, Hk, D]` and `[B, K, Hk, Dv]` and the keys `spans` leaves each query: as
    `dew.nn.kernels.templates.attention.reference` computes it, with `residuals` its running maxima and
    denominators `[B, H, Q]` too. The operands pad to whole blocks of the largest tile and to power-of-two
    head widths (`_padded_call`). `interpret` runs it on the CPU."""
    hooks = Hooks() if hooks is None else hooks
    call = _padded_call(query, key, value, spans, hooks)
    queries, value_width = call.queries, value.shape[-1]
    initial = hooks.initial
    m0 = total0 = acc0 = None
    if initial is not None:
        m0, total0 = call.rows(initial.m, -jnp.inf), call.rows(initial.total)
        acc0 = jnp.pad(call.rows(initial.acc), ((0, 0),) * 3 + ((0, call.value.shape[-1] - value_width),))
    op = TritonForward(score_mod=hooks.score_mod, output_mod=hooks.output_mod,
                       score_kinds=tuple(param.kind for param in hooks.score_params),
                       output_kinds=tuple(param.kind for param in hooks.output_params),
                       score_dtypes=tuple(_kept_dtype(param.value.dtype) for param in hooks.score_params),
                       output_dtypes=tuple(_kept_dtype(param.value.dtype) for param in hooks.output_params),
                       output_width=call.value.shape[-1],
                       scale=query.shape[3] ** -0.5 if scale is None else scale, form=spans.form,
                       keys=spans.keys)
    out = op(call.query, call.key, call.value, call.start, call.end, call.query_ids, call.key_ids,
             call.score_params, call.output_params, m0, total0, acc0, residuals=residuals,
             interpret=interpret)
    if not residuals:
        return out[:, :queries, :, :value_width]
    out, peak, total = out
    return out[:, :queries, :, :value_width], peak[:, :, :queries], total[:, :, :queries]


DQ_GRADS = (Kind.HEAD, Kind.QUERY, Kind.TILE)
"""The parameter kinds whose gradients the dQ program writes: a head's partial sum over its queries, its
queries' own, and its tiles' entries; a key parameter's the dK/dV program writes."""


def _inverse(total: jax.Array) -> jax.Array:
    """Each query's `1 / total`, zero where it saw nothing (total 0)."""
    return jnp.where(total > 0, 1.0 / jnp.where(total > 0, total, 1.0), 0.0)


def _scored(score_mod, logits, q_pos, k_pos, params, *, slopes: bool):
    """`score_mod` on the scaled logits, in fp32, and with `slopes` each entry's derivative of it along
    the logits and along each parameter, a tile apiece: `score_mod` being pointwise in the entries, they
    are the whole of its pullback (`_pulled`). They are its reverse-mode pullback's transpose applied to
    each input's ones, on the hook's own tile and layouts, so a hook through a `jax.custom_vjp`
    differentiates as the reference's does and a hook reads its arguments at the ranks it is called
    with. An integer parameter has no derivative: its slopes are zeros. Without a hook, the logits and
    no slopes."""
    if score_mod is None:
        return logits, None
    if not slopes:
        return score_mod(logits, q_pos, k_pos, *params).astype(jnp.float32), None
    floating = [index for index, param in enumerate(params) if jnp.issubdtype(param.dtype, jnp.floating)]

    def scored(s, *wide):
        values = list(params)
        for index, value in zip(floating, wide, strict=True):
            values[index] = value
        return score_mod(s, q_pos, k_pos, *values)

    inputs = (logits, *(params[index] for index in floating))
    out, pull = jax.vjp(scored, *inputs)
    along = jax.linear_transpose(pull, out)

    def slope(index):
        [entries] = along(tuple(jnp.ones_like(x) if place == index else jnp.zeros_like(x)
                                for place, x in enumerate(inputs)))
        return entries.astype(jnp.float32)

    differentiated = {index: slope(place + 1) for place, index in enumerate(floating)}
    return out.astype(jnp.float32), (slope(0), *(differentiated.get(index, jnp.zeros_like(logits))
                                                  for index in range(len(params))))


def _pulled(cotangents, slopes, seen, kinds: tuple[Kind, ...]):
    """The score cotangents `cotangents` through `score_mod`, its entries' derivatives `slopes`
    (`_scored`; None without a hook): the logits' tile, and each parameter's, a head's or a query's its
    queries' sums `[block_q]`, a key's its keys' `[block_k]`, a tile's the tile. Each entry is masked
    before any sum, so one the call never sees (padding, another document) adds nothing even where the
    hook's derivative there is not finite. fp32."""
    if slopes is None:
        return cotangents, ()
    logits = jnp.where(seen, cotangents * slopes[0], 0.0)
    pieces = []
    for slope, kind in zip(slopes[1:], kinds, strict=True):
        entries = jnp.where(seen, cotangents * slope, 0.0)
        pieces.append(entries.sum(axis=0) if kind == Kind.KEY else entries if kind == Kind.TILE
                      else entries.sum(axis=-1))
    return logits, tuple(pieces)


def _key_param(ref, kind: Kind, head, rows):
    """A hook parameter's values for a dK/dV program's tile, its queries `rows` of the whole row: the
    head's scalar, the rows' `[block_q, 1]`, its keys' `[1, block_k]`, the rows' entries."""
    match kind:
        case Kind.HEAD:
            return ref[head]
        case Kind.QUERY:
            return ref[rows][:, None]
        case Kind.KEY:
            return ref[...][None, :]
        case Kind.TILE:
            return ref[rows, :]
        case Kind.OUTPUT:
            raise ValueError("an output hook's parameter has no score tile")


def _tile_logits(score_mod, query, key, seen, q_pos, k_pos, params, *, scale: float, interpret: bool,
                 slopes: bool = False):
    """A tile's scaled logits under `score_mod`, -inf where unseen, and with `slopes` its entries'
    derivatives (`_scored`)."""
    logits, derivatives = _scored(score_mod, _dot(query, key.T, interpret) * scale, q_pos, k_pos, params,
                                  slopes=slopes)
    return jnp.where(seen, logits, -jnp.inf), derivatives


def _tile_scores(score_mod, query, key, value, dout, peak, total, anchor, delta, seen, q_pos, k_pos, params,
                 *, scale: float, interpret: bool):
    """A tile's backward through its scores: its unnormalized weights `exp(s - m)` and its weights `p`
    (zero where masked), the masked score cotangents `p * ((dP - c) - (D - c))`, `dP = dO V^T`, `c` each
    query's `anchor` and `delta` its `D - c`, and the entries' derivatives through `score_mod`
    (`_scored`)."""
    logits, slopes = _tile_logits(score_mod, query, key, seen, q_pos, k_pos, params, scale=scale,
                                  interpret=interpret, slopes=True)
    exponentials = jnp.exp(logits - peak[:, None])
    weights, products = exponentials * _inverse(total)[:, None], _dot(dout, value.T, interpret)
    cotangents = jnp.where(seen, weights * ((products - anchor[:, None]) - delta[:, None]), 0.0)
    return exponentials, weights, cotangents, slopes


def _dq_kernel(bounds_ref, start_ref, end_ref, *refs, ids: bool, score_kinds: tuple[Kind, ...], score_mod,
               scale: float, block_k: int, interpret: bool):
    refs = list(refs)
    query_ids_ref, key_ids_ref = (refs.pop(0), refs.pop(0)) if ids else (None, None)
    query_ref, key_ref, value_ref, dout_ref, peak_ref, total_ref, initial_ref = (refs.pop(0)
                                                                                 for _ in range(7))
    score_refs = [refs.pop(0) for _ in score_kinds]
    # A tile parameter's gradient enters as its zeros, aliased to its output (`_dq_launch`).
    for _ in (kind for kind in score_kinds if kind == Kind.TILE):
        refs.pop(0)
    dq_ref, delta_ref, anchor_ref, peak_out_ref, total_out_ref, mass_ref, *outputs = refs
    grad_refs = [outputs.pop(0) if kind in DQ_GRADS else None for kind in score_kinds]
    block_q = query_ref.shape[0]
    head = pl.program_id(2)
    q_pos = pl.program_id(0) * block_q + jnp.arange(block_q)[:, None]
    query, dout = query_ref[...], dout_ref[...]
    peak, total = peak_ref[...], total_ref[...]
    start, end = start_ref[...], end_ref[...]
    query_ids = query_ids_ref[...] if ids else None

    def tile(index):
        keys = pl.ds(index * block_k, block_k)
        k_pos = index * block_k + jnp.arange(block_k)[None, :]
        seen = (k_pos >= start[:, None]) & (k_pos < end[:, None])
        if ids:
            seen &= query_ids[:, None] == key_ids_ref[keys][None, :]
        params = [_tile_param(ref, kind, head, q_pos, keys)
                  for ref, kind in zip(score_refs, score_kinds, strict=True)]
        return keys, k_pos, seen, params

    def row_terms(index, carry):
        # The softmax again, online as the forward's, over the very tiles and products the main pass
        # computes: its maximum m, denominator T, and D's numerator about an anchor c, the product of a
        # key that holds the running maximum, sum_j exp(s_j - m) (dP_j - c). The weights exp(s - m) / T
        # sum to one over these logits whatever tiles the forward took, and dP_j - D is taken as
        # (dP_j - c) - (D - c): the key that holds a query's maximum enters it as exactly 0, so a query
        # that keeps one key gets 0, and one whose other keys and initial state weigh less than fp32's
        # resolution beside it gets their share in full.
        peak, anchor, total, numerator = carry
        keys, k_pos, seen, params = tile(index)
        logits, _ = _tile_logits(score_mod, query, key_ref[keys, :], seen, q_pos, k_pos, params, scale=scale,
                                 interpret=interpret)
        products = _dot(dout, value_ref[keys, :].T, interpret)
        tile_peak = logits.max(axis=-1)
        new_peak = jnp.maximum(peak, tile_peak)
        new_anchor = jnp.where(tile_peak > peak, jnp.max(jnp.where(logits == tile_peak[:, None], products,
                                                                   -jnp.inf), axis=-1), anchor)
        safe = jnp.where(new_peak == -jnp.inf, 0.0, new_peak)
        correction = jnp.exp(peak - safe)
        exponentials = jnp.exp(logits - safe[:, None])
        numerator = (correction * (numerator + (anchor - new_anchor) * total)
                     + jnp.sum(exponentials * (products - new_anchor[:, None]), axis=-1))
        return new_peak, new_anchor, correction * total + exponentials.sum(axis=-1), numerator

    # From the softmax's initial state (`backward`), anchored at 0.
    peak0, shares = peak, (total, initial_ref[...])
    peak, anchor, total, numerator = jax.lax.fori_loop(bounds_ref[0], bounds_ref[1], row_terms,
                                                       (peak, jnp.zeros_like(total), *shares))
    peak = jnp.where(peak == -jnp.inf, 0.0, peak)
    delta = jnp.where(total > 0, numerator / jnp.where(total > 0, total, 1.0), 0.0)
    summed = tuple(kind in (Kind.HEAD, Kind.QUERY) for kind in score_kinds)

    def step(index, carry):
        dq, residual, mass, grads, weighed = carry
        keys, k_pos, seen, params = tile(index)
        key, value = key_ref[keys, :], value_ref[keys, :]
        _, weights, cotangents, slopes = _tile_scores(score_mod, query, key, value, dout, peak, total, anchor,
                                                   delta, seen, q_pos, k_pos, params, scale=scale,
                                                   interpret=interpret)
        logits, pieces = _pulled(cotangents, slopes, seen, score_kinds)
        # Scaled before it narrows to the operands' dtype, so fp16 holds what fp32 does.
        dq = dq + _dot((logits * scale).astype(key.dtype), key, interpret)
        for kind, ref, piece in zip(score_kinds, grad_refs, pieces, strict=True):
            if kind == Kind.TILE:
                ref[:, keys] = piece
        grads = tuple(grad + piece if row else grad for grad, row, piece in zip(grads, summed, pieces,
                                                                               strict=True))
        if any(summed):
            _, pieces = _pulled(weights, slopes, seen, score_kinds)
            weighed = tuple(sums + piece if row else sums for sums, row, piece in zip(weighed, summed, pieces,
                                                                                    strict=True))
        return (dq, residual + cotangents.sum(axis=-1), mass + weights.sum(axis=-1), grads, weighed)

    nothing = jnp.zeros((block_q,), jnp.float32)
    rows = tuple(nothing for _ in score_kinds)
    dq, residual, mass, grads, weighed = jax.lax.fori_loop(bounds_ref[0], bounds_ref[1], step,
                                                           (jnp.zeros(query.shape, jnp.float32), nothing,
                                                            nothing, rows, rows))
    dq_ref[...] = dq
    # D refined once from the main pass's own sums: the score cotangents of a query and of its initial
    # state sum to 0 by the softmax's shift, and what they sum to here is D's rounding. Moving D by it over
    # the weights' total lets the dK/dV launch, the initial state's gradients and the head and query
    # parameters' (by the pullbacks of the weights, `weighed`) take the D whose cotangents do.
    inverse = jnp.where(total > 0, 1.0 / jnp.where(total > 0, total, 1.0), 0.0)
    carried = jnp.where(peak0 == -jnp.inf, 0.0, jnp.exp(peak0 - peak)) * inverse
    whole_residual = residual + (shares[1] - shares[0] * (anchor + delta)) * carried
    whole_mass = mass + shares[0] * carried
    refined = jnp.where(whole_mass > 0, whole_residual / jnp.where(whole_mass > 0, whole_mass, 1.0), 0.0)
    delta = delta + refined
    grads = tuple(grad - refined * sums if row else grad
                  for grad, row, sums in zip(grads, summed, weighed, strict=True))
    delta_ref[...] = delta
    anchor_ref[...] = anchor
    peak_out_ref[...] = peak
    total_out_ref[...] = total
    # The keys' share of the weights, which tells a query that read only its initial state.
    mass_ref[...] = mass
    for grad, kind, ref in zip(grads, score_kinds, grad_refs, strict=True):
        if kind == Kind.HEAD:
            ref[...] = jnp.broadcast_to(jnp.sum(grad), ref.shape)
        elif kind == Kind.QUERY:
            ref[...] = grad


def _dkv_kernel(ranges_ref, start_ref, end_ref, *refs, ids: bool, score_kinds: tuple[Kind, ...], score_mod,
                scale: float, block_q: int, interpret: bool):
    refs = list(refs)
    query_ids_ref, key_ids_ref = (refs.pop(0), refs.pop(0)) if ids else (None, None)
    query_ref, key_ref, value_ref, dout_ref, peak_ref, total_ref, anchor_ref, delta_ref = (refs.pop(0)
                                                                                         for _ in range(8))
    score_refs = [refs.pop(0) for _ in score_kinds]
    dk_ref, dv_ref, *outputs = refs
    grad_refs = [outputs.pop(0) if kind == Kind.KEY else None for kind in score_kinds]
    block_k = key_ref.shape[0]
    head = pl.program_id(2)
    k_pos = pl.program_id(0) * block_k + jnp.arange(block_k)[None, :]
    key, value = key_ref[...], value_ref[...]
    key_ids = key_ids_ref[...] if ids else None

    def step(index, carry):
        dk, dv, *grads = carry
        rows = pl.ds(index * block_q, block_q)
        query, dout = query_ref[rows, :], dout_ref[rows, :]
        q_pos = index * block_q + jnp.arange(block_q)[:, None]
        seen = (k_pos >= start_ref[rows][:, None]) & (k_pos < end_ref[rows][:, None])
        if ids:
            seen &= query_ids_ref[rows][:, None] == key_ids[None, :]
        params = [_key_param(ref, kind, head, rows)
                  for ref, kind in zip(score_refs, score_kinds, strict=True)]
        exponentials, weights, cotangents, slopes = _tile_scores(score_mod, query, key, value, dout,
                                                                 peak_ref[rows],
                                                   total_ref[rows], anchor_ref[rows], delta_ref[rows], seen,
                                                   q_pos, k_pos, params, scale=scale, interpret=interpret)
        logits, pieces = _pulled(cotangents, slopes, seen, score_kinds)
        dv = dv + _value_gradient(exponentials, weights, dout, total_ref[rows], interpret)
        # Scaled before it narrows to the operands' dtype, so fp16 holds what fp32 does.
        dk = dk + _dot((logits * scale).astype(query.dtype).T, query, interpret)
        updated = [grad + piece if kind == Kind.KEY else grad
                   for grad, kind, piece in zip(grads, score_kinds, pieces, strict=True)]
        return dk, dv, *updated

    sums = [jnp.zeros((block_k,), jnp.float32) if kind == Kind.KEY else jnp.zeros((), jnp.float32)
            for kind in score_kinds]
    zeros = jnp.zeros(key.shape, jnp.float32), jnp.zeros(value.shape, jnp.float32)
    dk, dv, *grads = jax.lax.fori_loop(ranges_ref[0], ranges_ref[1], step, (*zeros, *sums))
    dk_ref[...] = dk
    dv_ref[...] = dv
    for grad, kind, ref in zip(grads, score_kinds, grad_refs, strict=True):
        if kind == Kind.KEY:
            ref[...] = grad


def _spans_specs(per_rows, whole, ids: bool) -> list:
    """The specs of the spans' starts and ends and, with documents, the queries' and keys' ids."""
    return [per_rows, per_rows, *([per_rows, whole] if ids else [])]


@dataclasses.dataclass(frozen=True, order=True)
class BackwardTiles:
    """The backward's config: the tile both its launches take (`block_q` by `block_k`, so the dK/dV
    launch's logits are the dQ launch's to the bit), and each launch's own warps and pipeline stages."""

    block_q: int
    block_k: int
    dq_warps: int
    dq_stages: int
    dkv_warps: int
    dkv_stages: int


BACKWARD_SPACE = tuple(BackwardTiles(tiles.block_q, tiles.block_k, tiles.warps, tiles.stages, warps, stages)
                       for tiles in SPACE for warps in (4, 8) for stages in (2, 3))
"""Every config of the backward: each of `SPACE`'s tiles for its dQ launch, with each warps and stages of
its dK/dV launch."""


@dataclasses.dataclass(frozen=True)
class TritonBackward(Template[BackwardTiles]):
    """The template's backward over padded operands (`backward` pads them), one tuning decision for its
    two launches: the dQ launch (`_dq_launch`), then the dK/dV launch (`_dkv_launch`) under its softmax.
    Returns the dQ launch's outputs, then the dK/dV launch's."""

    HOOKS = ("score_mod",)
    name = "triton"
    config_cls = BackwardTiles
    score_mod: Callable | None = None
    score_kinds: tuple[Kind, ...] = ()
    score_dtypes: tuple[str, ...] = ()
    scale: float = 1.0

    def _fwd(self, query, key, value, dout, peak, total, initial_term, start, end, query_ids, key_ids,
             score_params, *, interpret: bool, config: BackwardTiles, return_residuals: bool = False):
        shared = {"score_mod": self.score_mod, "score_kinds": self.score_kinds, "scale": self.scale,
                  "block_q": config.block_q, "block_k": config.block_k, "interpret": interpret}
        spans = start, end, query_ids, key_ids, score_params
        dq_out = _dq_launch(query, key, value, dout, peak, total, initial_term, *spans, warps=config.dq_warps,
                            stages=config.dq_stages, **shared)
        delta, anchor, peak, total = dq_out[1:5]
        dkv_out = _dkv_launch(query, key, value, dout, peak, total, anchor, delta, *spans,
                              warps=config.dkv_warps, stages=config.dkv_stages, **shared)
        return (*dq_out, *dkv_out), None

    def space(self, ba: tokamax.BoundArguments) -> Sequence[BackwardTiles]:
        queries, keys = ba.arguments["query"].shape[1], ba.arguments["key"].shape[1]
        return tuple(tiles for tiles in BACKWARD_SPACE
                     if queries % tiles.block_q == 0 and keys % tiles.block_k == 0)

    def staged_bytes(self, ba: tokamax.BoundArguments, config: BackwardTiles) -> int:
        """The shared memory a program of either launch holds, the larger: each stage's query,
        output-gradient, key and value tiles, in the operands' dtype."""
        query, value = ba.arguments["query"], ba.arguments["value"]
        size = jnp.dtype(query.dtype).itemsize
        width, value_width = query.shape[-1], value.shape[-1]
        return (max(config.dq_stages, config.dkv_stages) * (config.block_q + config.block_k)
                * (width + value_width) * size)

    def cost(self, ba: tokamax.BoundArguments, config: BackwardTiles, target: generation.GpuTarget) -> float:
        """Both launches' (`_launch_cost`): every tile of the call (a bound: the spans are traced, so a
        sparse call visits fewer)."""
        return sum(_launch_cost(ba, config, target, self.score_kinds, dq=dq) for dq in (True, False))

    def eligible(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
        return _eligible(ba, target)

    def operands(self, specs: list[jax.ShapeDtypeStruct], seed: int) -> list[jax.Array]:
        return _spanned(specs, super().operands(specs, seed))

    def traced(self, name: str, value: object) -> object:
        return _hook_program(name, value, self.score_kinds, self.score_dtypes, differentiated=True)


def _launch_cost(ba: tokamax.BoundArguments, config: BackwardTiles, target: generation.GpuTarget,
                 kinds: tuple[Kind, ...], *, dq: bool) -> float:
    """One backward launch's seconds: its programs, each staging the tiles it walks once a visit and
    taking its products and exponentials over every tile, the dQ launch two passes of the logits and
    dP and then dS K, the dK/dV launch the logits, dP, P^T dO and dS^T Q; every array it reads at its
    dtype, its fp32 gradients and row statistics written, and its hook parameters' gradients."""
    query, key, value = (ba.arguments[name] for name in ("query", "key", "value"))
    batch, queries, heads, width = query.shape
    keys, value_width = key.shape[1], value.shape[-1]
    size = jnp.dtype(query.dtype).itemsize
    read = sum(leaf.size * jnp.dtype(leaf.dtype).itemsize for leaf in jax.tree.leaves(dict(ba.arguments))
               if hasattr(leaf, "dtype"))
    if dq:
        programs, tiles = batch * heads * (queries // config.block_q), keys // config.block_k
        products, exponentials = 3 * width + 2 * value_width, 2
        walked = 2 * config.block_k * (width + value_width)
        written = 4 * batch * queries * heads * (width + 5)
        for param, kind in zip(ba.arguments["score_params"], kinds, strict=True):
            # A head's gradient is written as partials a query block; a query's or a tile's whole.
            written += 4 * (queries // config.block_q * batch * heads if kind == Kind.HEAD
                            else param.size if kind in DQ_GRADS else 0)
    else:
        programs, tiles = batch * heads * (keys // config.block_k), queries // config.block_q
        products, exponentials = 2 * width + 2 * value_width, 1
        # Its row statistics are the dQ launch's four outputs, one more than that launch's inputs.
        read += 4 * batch * heads * queries
        walked = config.block_q * (width + value_width)
        written = 4 * batch * keys * heads * (width + value_width)
        written += 4 * sum(param.size for param, kind in zip(ba.arguments["score_params"], kinds, strict=True)
                           if kind == Kind.KEY)
    wide = 0
    if not dq and query.dtype == jnp.float16:
        # fp16's dV product runs in fp32 (`_value_gradient`), after the launch's 16-bit work.
        products, wide = products - value_width, value_width
    seconds = costs.seconds(costs.Cost(
        programs=programs,
        device_bytes=read + written,
        program_flops=tiles * 2.0 * config.block_q * config.block_k * products,
        program_staged=float(tiles * walked * size),
        program_exponentials=float(tiles * config.block_q * config.block_k * exponentials),
        mode=_mode(query.dtype)), target)
    if wide:
        seconds += costs.seconds(costs.Cost(
            programs=programs, device_bytes=0,
            program_flops=tiles * 2.0 * config.block_q * config.block_k * wide, program_staged=0.0,
            program_exponentials=0.0, mode='fp32'), target)
    return seconds


def _dq_launch(query, key, value, dout, peak, total, initial_term, start, end, query_ids, key_ids,
               score_params, *, score_mod, score_kinds: tuple[Kind, ...], scale: float, block_q: int,
               block_k: int, warps: int, stages: int, interpret: bool) -> tuple[jax.Array, ...]:
    """The backward's dQ launch over padded operands (`backward` pads them): a program per query block
    and query head walks the key blocks the block map gives it twice, recomputing each tile's logits:
    first for its queries' softmax, online from the initial state's maximum, denominator and share of
    `D`'s numerator (`peak`, `total`, `initial_term`), to its maxima `m`, denominators `T`, anchors `c`
    and `D - c` (`D = sum_j p dP`); then for dQ, under the weights `p = exp(s - m) / T`, refining `D - c`
    from its own sums. Returns dQ `[B, Qp, H, Dp]` fp32, `D - c`, `c`, `m`, `T` and the initial
    maximum's cotangent `[B, H, Qp]`, and the gradients of the score hook's head (`[nQb, B, H, 1]`
    partials), query (`[B, H, Qp]`) and tile (`[B, H, Qp, Kp]`) parameters."""
    from jax.experimental.pallas import triton as plgpu

    batch, queries, heads, width = query.shape
    keys, kv_heads, value_width = key.shape[1], key.shape[2], value.shape[-1]
    group = heads // kv_heads
    blocks = block_map(KeySpans(start, end, query_ids, key_ids, keys), block_q, block_k)
    bounds = jnp.stack([blocks.lo, blocks.hi], axis=-1).astype(jnp.int32)
    ids = query_ids is not None
    per_query = pl.BlockSpec((None, block_q), lambda i, b, h: (b, i))
    per_head = pl.BlockSpec((None, None, block_q), lambda i, b, h: (b, h, i))
    in_specs = [pl.BlockSpec((None, None, 2), lambda i, b, h: (b, i, 0)),
                *_spans_specs(per_query, pl.BlockSpec((None, keys), lambda i, b, h: (b, 0)), ids),
                pl.BlockSpec((None, block_q, None, width), lambda i, b, h: (b, i, h, 0)),
                pl.BlockSpec((None, keys, None, width), lambda i, b, h: (b, 0, h // group, 0)),
                pl.BlockSpec((None, keys, None, value_width), lambda i, b, h: (b, 0, h // group, 0)),
                pl.BlockSpec((None, block_q, None, value_width), lambda i, b, h: (b, i, h, 0)),
                per_head, per_head, per_head]
    arrays = [bounds, start, end, *([query_ids, key_ids] if ids else []), query, key, value, dout, peak,
              total, initial_term]
    for param, kind in zip(score_params, score_kinds, strict=True):
        in_specs.append(_param_spec(kind, param, block_q, keys))
        arrays.append(param)
    out_shape = [jax.ShapeDtypeStruct(query.shape, jnp.float32),
                 *[jax.ShapeDtypeStruct((batch, heads, queries), jnp.float32)] * 5]
    out_specs = [pl.BlockSpec((None, block_q, None, width), lambda i, b, h: (b, i, h, 0)),
                 *[per_head] * 5]
    aliases = {}
    for param, kind in zip(score_params, score_kinds, strict=True):
        match kind:
            case Kind.HEAD:
                out_shape.append(jax.ShapeDtypeStruct((queries // block_q, batch, heads, 1), jnp.float32))
                out_specs.append(pl.BlockSpec((None, None, None, 1), lambda i, b, h: (i, b, h, 0)))
            case Kind.QUERY:
                out_shape.append(jax.ShapeDtypeStruct(param.shape, jnp.float32))
                out_specs.append(per_head)
            case Kind.TILE:
                # Entries no program visits keep their zeros: the output aliases a zero input.
                aliases[len(arrays)] = len(out_shape)
                in_specs.append(pl.BlockSpec((None, None, block_q, keys), lambda i, b, h: (b, h, i, 0)))
                arrays.append(jnp.zeros(param.shape, jnp.float32))
                out_shape.append(jax.ShapeDtypeStruct(param.shape, jnp.float32))
                out_specs.append(pl.BlockSpec((None, None, block_q, keys), lambda i, b, h: (b, h, i, 0)))
    kernel = functools.partial(_dq_kernel, ids=ids, score_kinds=score_kinds,
                               score_mod=score_mod, scale=scale, block_k=block_k,
                               interpret=interpret)
    out = pl.pallas_call(
        kernel, grid=(queries // block_q, batch, heads), in_specs=in_specs, out_specs=out_specs,
        out_shape=out_shape, input_output_aliases=aliases, interpret=interpret, name="attention_dq",
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
    )(*arrays)
    return tuple(out)


def _dkv_launch(query, key, value, dout, peak, total, anchor, delta, start, end, query_ids, key_ids,
                score_params, *, score_mod, score_kinds: tuple[Kind, ...], scale: float, block_q: int,
                block_k: int, warps: int, stages: int, interpret: bool) -> tuple[jax.Array, ...]:
    """The backward's dK/dV launch over padded operands: a program per key block and query head walks the
    query blocks the block map gives it, under the dQ launch's softmax at the dQ launch's tiles (a
    product rounds by its shape, so the logits it weighs each entry by are the very ones the dQ launch
    normalized); returns the query head's dK and dV `[B, Kp, H, Dp]` fp32, which `backward` sums over
    each key head's group, and the gradients of the score hook's key parameters (`[B, H, Kp]`)."""
    from jax.experimental.pallas import triton as plgpu

    batch, queries, heads, width = query.shape
    keys, kv_heads, value_width = key.shape[1], key.shape[2], value.shape[-1]
    group = heads // kv_heads
    blocks = block_map(KeySpans(start, end, query_ids, key_ids, keys), block_q, block_k)
    ranges = jnp.stack([blocks.q_lo, blocks.q_hi], axis=-1).astype(jnp.int32)
    ids = query_ids is not None
    whole_queries = pl.BlockSpec((None, queries), lambda j, b, h: (b, 0))
    whole_head = pl.BlockSpec((None, None, queries), lambda j, b, h: (b, h, 0))
    in_specs = [pl.BlockSpec((None, None, 2), lambda j, b, h: (b, j, 0)),
                *_spans_specs(whole_queries, pl.BlockSpec((None, block_k), lambda j, b, h: (b, j)), ids),
                pl.BlockSpec((None, queries, None, width), lambda j, b, h: (b, 0, h, 0)),
                pl.BlockSpec((None, block_k, None, width), lambda j, b, h: (b, j, h // group, 0)),
                pl.BlockSpec((None, block_k, None, value_width), lambda j, b, h: (b, j, h // group, 0)),
                pl.BlockSpec((None, queries, None, value_width), lambda j, b, h: (b, 0, h, 0)),
                *[whole_head] * 4]
    arrays = [ranges, start, end, *([query_ids, key_ids] if ids else []), query, key, value, dout, peak,
              total, anchor, delta]
    for param, kind in zip(score_params, score_kinds, strict=True):
        in_specs.append(_key_param_spec(kind, param, queries, block_k))
        arrays.append(param)
    out_shape = [jax.ShapeDtypeStruct((batch, keys, heads, width), jnp.float32),
                 jax.ShapeDtypeStruct((batch, keys, heads, value_width), jnp.float32)]
    out_specs = [pl.BlockSpec((None, block_k, None, width), lambda j, b, h: (b, j, h, 0)),
                 pl.BlockSpec((None, block_k, None, value_width), lambda j, b, h: (b, j, h, 0))]
    for param, kind in zip(score_params, score_kinds, strict=True):
        if kind == Kind.KEY:
            out_shape.append(jax.ShapeDtypeStruct(param.shape, jnp.float32))
            out_specs.append(pl.BlockSpec((None, None, block_k), lambda j, b, h: (b, h, j)))
    kernel = functools.partial(_dkv_kernel, ids=ids, score_kinds=score_kinds,
                               score_mod=score_mod, scale=scale, block_q=block_q,
                               interpret=interpret)
    out = pl.pallas_call(
        kernel, grid=(keys // block_k, batch, heads), in_specs=in_specs, out_specs=out_specs,
        out_shape=out_shape, interpret=interpret, name="attention_dkv",
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
    )(*arrays)
    return tuple(out)


def _key_param_spec(kind: Kind, param: jax.Array, queries: int, block_k: int) -> pl.BlockSpec:
    """How a dK/dV program reads a hook parameter of `kind`: a head's scalar from the whole `[H]`, its
    head's whole row of queries, its key block, its key block's columns of every query."""
    match kind:
        case Kind.HEAD:
            return pl.BlockSpec(param.shape, lambda j, b, h: (0,))
        case Kind.QUERY:
            return pl.BlockSpec((None, None, queries), lambda j, b, h: (b, h, 0))
        case Kind.KEY:
            return pl.BlockSpec((None, None, block_k), lambda j, b, h: (b, h, j))
        case Kind.TILE:
            return pl.BlockSpec((None, None, queries, block_k), lambda j, b, h: (b, h, 0, j))
        case Kind.OUTPUT:
            raise ValueError("an output hook's parameter has no score tile")


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Gradients:
    """The backward's gradients: of the queries, keys and values, of each score hook parameter (the
    order `Hooks.score_params` gives them), and of the softmax's initial state where it has one."""

    query: jax.Array
    key: jax.Array
    value: jax.Array
    score_params: tuple[jax.Array, ...]
    initial: Initial | None


def backward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None,
             dout: jax.Array, *, scale: float | None = None, interpret: bool = False) -> Gradients:
    """The template's backward (`TritonBackward`) of `forward` under the output's cotangent `dout`
    `[B, Q, H, Dv]`: each gradient fp32, then rounded to its input's dtype. The dQ launch first takes each
    query's softmax again, online over the tiles and products its main pass takes, to its maximum,
    denominator and `D = sum_j p dP`, as the softmax's derivative takes it: the weights it differentiates
    sum to one over its own logits, and each `dP - D` is taken about the product of the key that holds
    the query's maximum, so no rounding of that key's product enters it; the main pass then refines `D`
    once from its own sums for the dK/dV launch and the head, query and initial state's gradients
    (`_dq_kernel`). It reads none of the forward's residuals. The output hook is no part of it: a
    differentiated call applies `output_mod` past the kernel, where its own derivative runs, and passes
    `dout` before it. Deterministic: every gradient element is written by one program, a head
    parameter's partial sums added in a fixed order and a key head's query heads summed in XLA."""
    hooks = Hooks() if hooks is None else hooks
    if hooks.output_mod is not None:
        raise ValueError("the backward takes the output before the output hook (`backward`'s docstring)")
    batch, queries, heads, width = query.shape
    keys, kv_heads, value_width = key.shape[1], key.shape[2], value.shape[-1]
    group = heads // kv_heads
    scale = width ** -0.5 if scale is None else scale
    # One cotangent for every part: the output's, in the operands' dtype as the reference takes it.
    dout = dout.astype(query.dtype)
    cotangent = jnp.swapaxes(dout.astype(jnp.float32), 1, 2)
    state = hooks.initial
    if state is None:
        nothing = jnp.zeros((batch, heads, queries), jnp.float32)
        starts = nothing - jnp.inf, nothing, nothing
    else:
        # The softmax's initial state: its maximum, denominator and share of D's numerator, in fp32.
        starts = (state.m.astype(jnp.float32), state.total.astype(jnp.float32),
                  jnp.sum(state.acc.astype(jnp.float32) * cotangent, axis=-1))
    call = _padded_call(query, key, value, spans, hooks)
    padded_dout = jnp.pad(dout, ((0, 0), (0, call.query.shape[1] - queries), (0, 0),
                                 (0, call.value.shape[-1] - value_width)))
    operands = (call.query, call.key, call.value, padded_dout)
    rows = (call.start, call.end, call.query_ids, call.key_ids, call.score_params)
    kinds = tuple(param.kind for param in hooks.score_params)
    dtypes = tuple(_kept_dtype(param.value.dtype) for param in hooks.score_params)
    outputs = TritonBackward(score_mod=hooks.score_mod, score_kinds=kinds, scale=scale, score_dtypes=dtypes)(
        *operands, *(call.rows(start) for start in starts), *rows, interpret=interpret)
    dq, softmax = outputs[0], outputs[1:6]
    grads_out = list(outputs[6:])
    dq_grads = [grads_out.pop(0) for kind in kinds if kind in DQ_GRADS]
    dk, dv, *key_grads = grads_out
    delta, anchor, peak, total, mass = (array[:, :, :queries] for array in softmax)
    delta = anchor + delta
    padded_k, padded_d, padded_dv = call.key.shape[1], call.key.shape[-1], call.value.shape[-1]
    dk = dk.reshape(batch, padded_k, kv_heads, group, padded_d).sum(axis=3)
    dv = dv.reshape(batch, padded_k, kv_heads, group, padded_dv).sum(axis=3)
    grads = []
    for param in hooks.score_params:
        match param.kind:
            case Kind.HEAD:
                grads.append(dq_grads.pop(0).sum(axis=(0, 1))[:param.value.shape[0], 0])
            case Kind.QUERY:
                grads.append(dq_grads.pop(0)[:, :, :queries])
            case Kind.TILE:
                grads.append(dq_grads.pop(0)[:, :, :queries, :keys])
            case Kind.KEY:
                grads.append(key_grads.pop(0)[:, :, :keys])
            case Kind.OUTPUT:
                raise ValueError("a score hook takes no output parameter")
    initial = None
    if state is not None:
        # The initial state as one more softmax term, its weight exp(m0 - m) / T and value acc0 / total0:
        # each gradient divides by T first and takes the factor exp(m0 - m) last, so a state that weighs
        # little beside the keys (a subnormal factor) is rounded once there, as the reference rounds it.
        factor = jnp.where((total > 0) & jnp.isfinite(starts[0]), jnp.exp(starts[0] - peak), 0.0)
        inverse = jnp.where(total > 0, 1.0 / jnp.where(total > 0, total, 1.0), 0.0)
        # Its maximum's cotangent, that term's score cotangent, taken whole as the reference's derivative
        # takes it; exactly 0 for a query that read only its state, whose output its maximum leaves unmoved.
        moved = (jnp.sum(state.acc.astype(jnp.float32) * cotangent, axis=-1)
                 - state.total.astype(jnp.float32) * delta) * inverse
        initial = Initial(m=jnp.where(mass > 0, moved * factor, 0.0).astype(state.m.dtype),
                          total=(-(delta * inverse) * factor).astype(state.total.dtype),
                          acc=((cotangent * inverse[..., None]) * factor[..., None]).astype(state.acc.dtype))
    return Gradients(dq[:, :queries, :, :width].astype(query.dtype),
                     dk[:, :keys, :, :width].astype(key.dtype),
                     dv[:, :keys, :, :value_width].astype(value.dtype),
                     tuple(grad.astype(param.value.dtype) for grad, param in zip(grads, hooks.score_params,
                                                                                 strict=True)),
                     initial)
