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
from dew.nn.kernels.templates import attention_algebra as algebra, cost as costs
from dew.nn.kernels.templates.attention import (
    DKV_GRADS,
    DQ_GRADS,
    Gradients,
    Hooks,
    KeySpans,
    Kind,
    Split,
    block_map,
    gradients,
    kept_dtype,
    padded_call,
)
from dew.nn.kernels.templates.base import Template, dividing

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


FORWARD_DEFAULT = Tiles(128, 64, 4, 2)
"""The forward config first in its space, which a pick of tied estimates takes (`Template.estimate`):
within 6% of the fastest at Qwen3-8B, causal and full, on an A100 and an H100 (ATT-ZERO3-a100 and
ATT-ZERO4-h100, 2026-10-21), where the space's first config otherwise, 32 by 32, ran 2-3 times it."""

SPACE = (FORWARD_DEFAULT, *(tiles for tiles in (Tiles(block_q, block_k, warps, stages)
                                                for block_q in (32, 64, 128) for block_k in (32, 64, 128)
                                                for warps in (4, 8) for stages in (2, 3))
                            if tiles != FORWARD_DEFAULT))
"""Every config the forward is proven over (tests/test_attention_triton.py pins each), its default
first."""

BLOCK = max(max(tiles.block_q for tiles in SPACE), max(tiles.block_k for tiles in SPACE))
"""The largest tile of `SPACE`, the block a call pads to (`padded_call`)."""


def _dot(a: jax.Array, b: jax.Array, interpret: bool) -> jax.Array:
    """`a @ b` accumulated in fp32: bf16's and fp16's own product, IEEE fp32 for fp32 operands;
    interpreted, the CPU's own product."""
    if interpret:
        return jnp.dot(a, b, preferred_element_type=jnp.float32)
    from jax.experimental.pallas import triton as plgpu

    return plgpu.dot(a, b, precision=jax.lax.Precision.HIGHEST).astype(jnp.float32)


def _masked_rows(ref, rows: jax.Array, mask: jax.Array) -> jax.Array:
    """`ref`'s rows `rows`, each loaded only where `mask` holds and zero elsewhere."""
    from jax.experimental.pallas import triton as plgpu

    return plgpu.load(ref.at[rows, :], mask=mask[:, None], other=0)


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
    floating = algebra.floating_params(params)

    def pulled(s, q_pos, k_pos, *values):
        hooked = algebra.differentiable(hook, q_pos, k_pos, values, floating)
        out, pull = jax.vjp(hooked, s, *(values[index] for index in floating))
        return out, pull(jnp.ones_like(out))

    return tuning.program(pulled, square, *positions, *params)


def _spanned(specs: list[jax.ShapeDtypeStruct], arrays: list[jax.Array]) -> list[jax.Array]:
    """A launch's timing operands with its spans whole: of its integer operands by row `[B, Q]` (the
    others, an anchor key `[B, H, Q]`, keep their draw), the first two each query's first key (0) and
    one past its last (every key, `specs[1]` being the keys), the rest its documents' ids (one
    document), so a timed launch visits every tile a dense call does."""
    rows = [index for index, spec in enumerate(specs)
            if jnp.issubdtype(spec.dtype, jnp.integer) and spec.ndim == 2]
    keys = specs[1].shape[1]
    for place, index in enumerate(rows):
        filled = (0, keys)[place] if place < 2 else 1
        arrays[index] = jnp.full(specs[index].shape, filled, specs[index].dtype)
    return arrays


def _tile_param(ref, kind: Kind, head, queries: jax.Array, keys: pl.Slice | None):
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


def _forward_kernel(*refs, spec: algebra.Forward, score_kinds: tuple[Kind, ...],
                    output_kinds: tuple[Kind, ...], initial: bool, residuals: bool, output_mod, block_k: int,
                    interpret: bool):
    """The forward's Triton mainloop: one program a query block and head walks the key tiles its form
    gives it (every tile, the causal diagonal's, or the spans' block map), issuing the tile's two
    products and calling the algebra between them (`attention_algebra`)."""
    refs = list(refs)
    narrowing, bounds, key_ids = {}, None, {}
    if spec.form == "spans":
        # Only spans read their block map's bounds and their queries' intervals.
        bounds_ref = refs.pop(0)
        bounds = bounds_ref[0], bounds_ref[1]
        narrowing = {"start": refs.pop(0)[...], "end": refs.pop(0)[...]}
    if spec.ids:
        narrowing["query_ids"] = refs.pop(0)[...]
        key_ids["ref"] = refs.pop(0)
    query_ref, key_ref, value_ref = refs.pop(0), refs.pop(0), refs.pop(0)
    score_refs = [refs.pop(0) for _ in score_kinds]
    output_refs = [refs.pop(0) for _ in output_kinds]
    initial_refs = [refs.pop(0) for _ in range(3 if initial else 0)]
    out_ref, *residual_refs = refs
    block_q = query_ref.shape[0]
    head = pl.program_id(2)
    first_query = pl.program_id(0) * block_q
    tiles = -(-spec.keys // block_k)
    if bounds is not None:
        lo, hi = bounds
    elif spec.form == "causal":
        lo, hi = 0, jnp.minimum(-(-(first_query + block_q) // block_k), tiles)
    else:
        lo, hi = 0, tiles
    query = query_ref[...]
    q_pos = first_query + jnp.arange(block_q)[:, None]

    def softmax(index, state: algebra.Partial):
        tile = pl.ds(index * block_k, block_k)
        coords = algebra.Coords(q_pos, index * block_k + jnp.arange(block_k)[None, :], **narrowing,
                                key_ids=key_ids["ref"][tile] if spec.ids else None)
        params = tuple(_tile_param(ref, kind, head, q_pos, tile)
                       for ref, kind in zip(score_refs, score_kinds, strict=True))
        logits = algebra.modify(spec, _dot(query, key_ref[tile, :].T, interpret), coords, params)
        m, alpha = algebra.peak(spec, state.m, logits)
        weights = algebra.weights(spec, logits, m)
        return tile, coords, logits, m, alpha, weights, algebra.total(state.total, alpha, weights)

    def step(index, state: algebra.Partial) -> algebra.Partial:
        tile, _, _, m, alpha, weights, denominator = softmax(index, state)
        # The values load after the softmax, where jax's mha loads them.
        value = value_ref[tile, :]
        acc = algebra.rescale(state.acc, alpha) + _dot(algebra.narrow(weights, value.dtype), value, interpret)
        return algebra.Partial(acc, m, denominator)

    def tracked(index, carry: tuple[algebra.Partial, algebra.Argmax, jax.Array, jax.Array]):
        # A training forward's step: the anchor key and the split sums the backward reads
        # (`algebra.split_accumulate`); its state's values are `B`, every key's but the anchor's.
        state, arg, own, others_total = carry
        tile, coords, logits, m, alpha, weights, denominator = softmax(index, state)
        anchor = algebra.argmax(arg, logits, coords.k_pos)
        moved = anchor.peak > arg.peak
        value = value_ref[tile, :]
        others, own_tile = algebra.anchor_split(spec, algebra.narrow(weights, value.dtype),
                                                coords.k_pos == anchor.key[:, None], anchor, m)
        product = _dot(others, value, interpret)
        # The old anchor's value row, read only where the anchor moved: after a row's first tiles it
        # rarely does, and a masked load fetches nothing for the rest.
        old_value = _masked_rows(value_ref, arg.key, moved)
        acc, own, others_total = algebra.split_accumulate(state.acc, own, others_total, alpha, moved,
                                                          old_value, product, others, own_tile)
        return algebra.Partial(acc, m, denominator), anchor, own, others_total

    start = (algebra.resumed(spec, *(ref[...] for ref in initial_refs)) if initial
             else algebra.empty(block_q, value_ref.shape[-1]))
    output_params = tuple(_tile_param(ref, kind, head, q_pos, None)
                          for ref, kind in zip(output_refs, output_kinds, strict=True))
    if residuals:
        nothing = jnp.zeros((block_q,), jnp.float32)
        carry = start, algebra.no_argmax(block_q), nothing, nothing
        split, arg, own, others_total = jax.lax.fori_loop(lo, hi, tracked, carry)
        residual_refs[2][...] = arg.key
        residual_refs[3][...] = split.acc
        residual_refs[4][...] = own
        residual_refs[5][...] = others_total
        state = split._replace(acc=algebra.split_output(split.acc, own, value_ref[arg.key, :]))
    else:
        state = jax.lax.fori_loop(lo, hi, step, start)
    out, m, denominator = algebra.finish(spec, state, initial=initial, output_mod=output_mod,
                                         output_params=output_params)
    out_ref[...] = out.astype(out_ref.dtype)
    if residuals:
        residual_refs[0][...] = m
        residual_refs[1][...] = denominator


@dataclasses.dataclass(frozen=True)
class TritonForward(Template[Tiles]):
    """The template's forward in Pallas Triton over padded operands (`forward` pads them): queries
    `[B, Qp, H, Dp]` over keys `[B, Kp, Hk, Dp]` and values `[B, Kp, Hk, Dvp]`, each query's keys
    `start` to `end` `[B, Qp]` and, with documents, its id `[B, Qp]` against the keys' `[B, Kp]`; the
    score and output hooks' parameters (`score_params`, `output_params`, laid out by `score_kinds` and
    `output_kinds`) and the softmax's initial state. Returns the output `[B, Qp, H, Dvp]` and, with
    `residuals`, each query's running maximum and denominator `[B, H, Qp]`, fp32, and the first key
    holding the keys' maximum `[B, H, Qp]` int32, the backward's anchor (`algebra.Argmax`)."""

    HOOKS = ("score_mod", "output_mod")
    name = "triton"
    config_cls = Tiles
    score_mod: Callable | None = None
    output_mod: Callable | None = None
    form: str = "spans"
    keys: int | None = None
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
            per_row = pl.BlockSpec((None, None, block_q), lambda i, b, h: (b, h, i))
            out_shape += [*[jax.ShapeDtypeStruct((batch, heads, queries), jnp.float32)] * 2,
                          jax.ShapeDtypeStruct((batch, heads, queries), jnp.int32),
                          jax.ShapeDtypeStruct((batch, heads, queries, value_width), jnp.float32),
                          *[jax.ShapeDtypeStruct((batch, heads, queries), jnp.float32)] * 2]
            per_value = pl.BlockSpec((None, None, block_q, value_width), lambda i, b, h: (b, h, i, 0))
            out_specs += [per_row, per_row, per_row, per_value, per_row, per_row]
        # A call's own key count, none included; an op built without one counts its launch's keys.
        spec = algebra.Forward(self.form, keys if self.keys is None else self.keys, ids, self.scale,
                               self.score_mod)
        kernel = functools.partial(
            _forward_kernel, spec=spec, score_kinds=self.score_kinds, output_kinds=self.output_kinds,
            initial=initial, residuals=residuals, output_mod=self.output_mod, block_k=block_k,
            interpret=interpret)
        out = pl.pallas_call(
            kernel, grid=(queries // block_q, batch, heads), in_specs=in_specs, out_specs=out_specs,
            out_shape=out_shape, interpret=interpret, name="attention_forward",
            compiler_params=plgpu.CompilerParams(num_warps=config.warps, num_stages=config.stages),
        )(*arrays)
        return (tuple(out) if residuals else out[0]), None

    def space(self, ba: tokamax.BoundArguments) -> Sequence[Tiles]:
        return _dividing(SPACE, ba)

    def staged_bytes(self, ba: tokamax.BoundArguments, config: Tiles) -> int:
        """The shared memory Triton gives a program, in the operands' dtype: the key and value tiles of
        every stage but the one in use, the query tile, and the weights' tile on its way to the second
        product, which shares the query tile's memory where the block is 128 queries of heads at most
        128 wide. Fitted to the bytes Triton asks for (SMEM-a100 and SMEM-a10, 2026-10-21: equal on all 48
        configs it refused on either GPU, and every config it compiled within the budget)."""
        query, value = ba.arguments["query"], ba.arguments["value"]
        size = jnp.dtype(query.dtype).itemsize
        width, value_width = query.shape[-1], value.shape[-1]
        streamed = (config.stages - 1) * config.block_k * (width + value_width) * size
        query_tile, weights = config.block_q * width * size, config.block_q * config.block_k * size
        shared = config.block_q >= 128 and width <= 128
        return streamed + (max(query_tile, weights) if shared else query_tile + weights)

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

    def traced(self, name: str, value: Callable | str | None) -> str | None:
        if name == "output_mod":
            return _hook_program(name, value, self.output_kinds, self.output_dtypes, width=self.output_width)
        return _hook_program(name, value, self.score_kinds, self.score_dtypes)


def _dividing[T: (Tiles, BackwardTiles)](space: tuple[T, ...], ba: tokamax.BoundArguments) -> tuple[T, ...]:
    """The configs of `space` whose tiles divide the launch's padded queries and keys."""
    return dividing(space, (ba.arguments["query"].shape[1], ba.arguments["key"].shape[1]),
                    lambda tiles: (tiles.block_q, tiles.block_k))


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


def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, residuals: bool = False, interpret: bool = False):
    """The template's forward (`TritonForward`) on `query` `[B, Q, H, D]` over `key` and `value`
    `[B, K, Hk, D]` and `[B, K, Hk, Dv]` and the keys `spans` leaves each query: as
    `dew.nn.kernels.templates.attention.reference` computes it, with `residuals` its running maxima and
    denominators `[B, H, Q]` too, and each query's anchor key `[B, H, Q]` int32 (`backward` reads them
    all). The operands pad to whole blocks of the largest tile and to power-of-two
    head widths (`padded_call`). `interpret` runs it on the CPU."""
    hooks = Hooks() if hooks is None else hooks
    call = padded_call(query, key, value, spans, hooks, BLOCK)
    queries, value_width = call.queries, value.shape[-1]
    initial = hooks.initial
    m0 = total0 = acc0 = None
    if initial is not None:
        m0, total0 = call.rows(initial.m, -jnp.inf), call.rows(initial.total)
        acc0 = jnp.pad(call.rows(initial.acc), ((0, 0),) * 3 + ((0, call.value.shape[-1] - value_width),))
    op = TritonForward(score_mod=hooks.score_mod, output_mod=hooks.output_mod,
                       score_kinds=tuple(param.kind for param in hooks.score_params),
                       output_kinds=tuple(param.kind for param in hooks.output_params),
                       score_dtypes=tuple(kept_dtype(param.value.dtype) for param in hooks.score_params),
                       output_dtypes=tuple(kept_dtype(param.value.dtype) for param in hooks.output_params),
                       output_width=call.value.shape[-1],
                       scale=query.shape[3] ** -0.5 if scale is None else scale, form=spans.form,
                       keys=spans.keys)
    out = op(call.query, call.key, call.value, call.start, call.end, call.query_ids, call.key_ids,
             call.score_params, call.output_params, m0, total0, acc0, residuals=residuals,
             interpret=interpret)
    if not residuals:
        return out[:, :queries, :, :value_width]
    out, peak, total, anchors, others, own, others_total = out
    return (out[:, :queries, :, :value_width], peak[:, :, :queries], total[:, :, :queries],
            anchors[:, :, :queries],
            Split(others[:, :, :queries, :value_width], own[:, :, :queries], others_total[:, :, :queries]))


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
    derivatives (`algebra.scored`)."""
    logits, derivatives = algebra.scored(score_mod, _dot(query, key.T, interpret) * scale, q_pos, k_pos,
                                         params, slopes=slopes)
    return jnp.where(seen, logits, -jnp.inf), derivatives


def _tile_scores(score_mod, query, key, value, dout, peak, total, anchor, delta, seen, q_pos, k_pos, params,
                 *, key_index, scale: float, interpret: bool):
    """A tile's backward through its scores: its unnormalized weights `exp(s - m)` and its weights `p` over
    `total` (zero where masked), the masked score cotangents `p * ((dP - c) - (D - c))`
    (`algebra.grad_scores`), `dP = dO V^T`, `c` each query's `anchor` and `delta` its `D - c`, and the
    entries' derivatives through `score_mod` (`algebra.scored`)."""
    logits, slopes = _tile_logits(score_mod, query, key, seen, q_pos, k_pos, params, scale=scale,
                                  interpret=interpret, slopes=True)
    exponentials, weights = algebra.normalized(logits, peak, total)
    cotangents = algebra.grad_scores(exponentials, _dot(dout, value.T, interpret), anchor, delta, seen,
                                     k_pos == key_index[:, None], algebra.inverse(total))
    return exponentials, weights, cotangents, slopes


def _dq_kernel(bounds_ref, start_ref, end_ref, *refs, ids: bool, score_kinds: tuple[Kind, ...], score_mod,
               scale: float, block_k: int, split: bool, interpret: bool):
    refs = list(refs)
    id_refs = [refs.pop(0), refs.pop(0)] if ids else []
    (query_ref, key_ref, value_ref, dout_ref, peak_ref, total_ref, initial_ref, maximum_ref, index_ref,
     anchor_value_ref) = (refs.pop(0) for _ in range(10))
    # The forward's split sums and denominator, where it gives them (`algebra.split_delta`).
    split_refs = [refs.pop(0) for _ in range(4 if split else 0)]
    score_refs = [refs.pop(0) for _ in score_kinds]
    dq_ref, delta_ref, anchor_ref, peak_out_ref, total_out_ref, mass_ref, index_out_ref, *outputs = refs
    grad_refs = {place: outputs.pop(0) for place, kind in enumerate(score_kinds) if kind in DQ_GRADS}
    block_q = query_ref.shape[0]
    head = pl.program_id(2)
    q_pos = pl.program_id(0) * block_q + jnp.arange(block_q)[:, None]
    query, dout = query_ref[...], dout_ref[...]
    peak, total = peak_ref[...], total_ref[...]
    start, end = start_ref[...], end_ref[...]
    query_ids = [ref[...] for ref in id_refs[:1]]

    def tile(index):
        keys = pl.ds(index * block_k, block_k)
        k_pos = index * block_k + jnp.arange(block_k)[None, :]
        seen = (k_pos >= start[:, None]) & (k_pos < end[:, None])
        if ids:
            seen &= query_ids[0][:, None] == id_refs[1][keys][None, :]
        params = [_tile_param(ref, kind, head, q_pos, keys)
                  for ref, kind in zip(score_refs, score_kinds, strict=True)]
        return keys, k_pos, seen, params

    maximum = maximum_ref[...]

    def row_terms(index, state):
        # Every weight against the forward's maximum m, so no running maximum rescales a sum
        # (`algebra.OffAnchor`); the weights are this launch's own, exp(s - m) over its logits, so they
        # sum to one over them whatever logits the forward took.
        keys, k_pos, seen, params = tile(index)
        logits, _ = _tile_logits(score_mod, query, key_ref[keys, :], seen, q_pos, k_pos, params, scale=scale,
                                 interpret=interpret)
        value = value_ref[keys, :]
        exponentials, rest, own = algebra.off_anchor_weights(logits, maximum, k_pos == key_index[:, None],
                                                             value.dtype)
        return algebra.off_anchor(state, exponentials, rest, own, _dot(rest, value, interpret))

    # From the softmax's initial state (`gradients`): its maximum, denominator and share of D's numerator.
    key_index = index_ref[...]
    peak0, shares = peak, (total, initial_ref[...])
    before, state_total, state_share = algebra.off_anchor_start(peak, total, shares[1], maximum,
                                                                value_ref.shape[-1])
    if split:
        # The forward took the first pass (`algebra.split_accumulate`).
        others_ref, own_ref, others_total_ref, forward_total_ref = split_refs
        prior = forward_total_ref[...]
        anchor, delta, key_index = algebra.split_delta(
            others_ref[...], own_ref[...], others_total_ref[...], prior, state_total, anchor_value_ref[...],
            dout, key_index)
    else:
        state = jax.lax.fori_loop(bounds_ref[0], bounds_ref[1], row_terms, before)
        prior = state.total
        anchor, delta, key_index = algebra.off_anchor_delta(state, anchor_value_ref[...], dout, state_total,
                                                            state_share, key_index)
    peak = maximum
    index_out_ref[...] = key_index
    summed = tuple(kind in (Kind.HEAD, Kind.QUERY) for kind in score_kinds)

    def step(index, carry):
        # Each cotangent over the prior denominator, in the reference's order, and taken to the pass's
        # own at the end (`algebra.dq_finish`); the weights' sums unnormalized.
        dq, residual, mass, grads, weighed = carry
        keys, k_pos, seen, params = tile(index)
        key, value = key_ref[keys, :], value_ref[keys, :]
        weights, _, cotangents, slopes = _tile_scores(score_mod, query, key, value, dout, peak, prior, anchor,
                                                   delta, seen, q_pos, k_pos, params, key_index=key_index,
                                                   scale=scale, interpret=interpret)
        logits, pieces = algebra.pulled(cotangents, slopes, seen, score_kinds)
        # Scaled before it narrows to the operands' dtype and meets the keys, as the reference's is.
        dq = dq + _dot((logits * scale).astype(key.dtype), key, interpret)
        grads = tuple(grad + piece if row else grad for grad, row, piece in zip(grads, summed, pieces,
                                                                               strict=True))
        if any(summed):
            _, pieces = algebra.pulled(weights, slopes, seen, score_kinds)
            weighed = tuple(sums + piece if row else sums for sums, row, piece in zip(weighed, summed, pieces,
                                                                                    strict=True))
        return (dq, residual + cotangents.sum(axis=-1), mass + weights.sum(axis=-1), grads, weighed)

    nothing = jnp.zeros((block_q,), jnp.float32)
    rows = tuple(nothing for _ in score_kinds)
    sums = algebra.DqSums(*jax.lax.fori_loop(bounds_ref[0], bounds_ref[1], step,
                                             (jnp.zeros(query.shape, jnp.float32), nothing, nothing, rows,
                                              rows)))
    dq, delta, total, mass, grads = algebra.dq_finish(sums, prior, state_total, peak0, shares, peak, anchor,
                                                      delta)
    dq_ref[...] = dq
    delta_ref[...] = delta
    anchor_ref[...] = anchor
    peak_out_ref[...] = peak
    total_out_ref[...] = total
    # The keys' share of the weights, which tells a query that read only its initial state.
    mass_ref[...] = mass
    for place, (grad, kind) in enumerate(zip(grads, score_kinds, strict=True)):
        if kind == Kind.HEAD:
            grad_refs[place][...] = jnp.broadcast_to(jnp.sum(grad), grad_refs[place].shape)
        elif kind == Kind.QUERY:
            grad_refs[place][...] = grad


def _dkv_kernel(ranges_ref, start_ref, end_ref, *refs, ids: bool, score_kinds: tuple[Kind, ...], score_mod,
                scale: float, block_q: int, interpret: bool):
    refs = list(refs)
    id_refs = [refs.pop(0), refs.pop(0)] if ids else []
    query_ref, key_ref, value_ref, dout_ref, peak_ref, total_ref, anchor_ref, delta_ref, index_ref = (
        refs.pop(0) for _ in range(9))
    score_refs = [refs.pop(0) for _ in score_kinds]
    # A tile parameter's gradient enters as its zeros, aliased to its output (`_dkv_launch`).
    for _ in (kind for kind in score_kinds if kind == Kind.TILE):
        refs.pop(0)
    dk_ref, dv_ref, *outputs = refs
    grad_refs = {place: outputs.pop(0) for place, kind in enumerate(score_kinds) if kind in DKV_GRADS}
    block_k = key_ref.shape[0]
    head = pl.program_id(2)
    k_pos = pl.program_id(0) * block_k + jnp.arange(block_k)[None, :]
    key, value = key_ref[...], value_ref[...]
    key_ids = [ref[...] for ref in id_refs[1:]]

    def step(index, carry):
        dk, dv, *grads = carry
        rows = pl.ds(index * block_q, block_q)
        query, dout = query_ref[rows, :], dout_ref[rows, :]
        q_pos = index * block_q + jnp.arange(block_q)[:, None]
        seen = (k_pos >= start_ref[rows][:, None]) & (k_pos < end_ref[rows][:, None])
        if ids:
            seen &= id_refs[0][rows][:, None] == key_ids[0][None, :]
        params = [_key_param(ref, kind, head, rows)
                  for ref, kind in zip(score_refs, score_kinds, strict=True)]
        exponentials, weights, cotangents, slopes = _tile_scores(
            score_mod, query, key, value, dout, peak_ref[rows], total_ref[rows], anchor_ref[rows],
            delta_ref[rows], seen, q_pos, k_pos, params, key_index=index_ref[rows], scale=scale,
            interpret=interpret)
        logits, pieces = algebra.pulled(cotangents, slopes, seen, score_kinds)
        dv = dv + _dot(*algebra.value_operands(exponentials, weights, dout, total_ref[rows]), interpret)
        # Scaled before it narrows to the operands' dtype, so fp16 holds what fp32 does.
        dk = dk + _dot((logits * scale).astype(query.dtype).T, query, interpret)
        for place, (kind, piece) in enumerate(zip(score_kinds, pieces, strict=True)):
            if kind == Kind.TILE:
                # A tile parameter's entries, under the refined D (`algebra.refinement`).
                grad_refs[place][rows, :] = piece
        updated = [grad + piece if kind == Kind.KEY else grad
                   for grad, kind, piece in zip(grads, score_kinds, pieces, strict=True)]
        return dk, dv, *updated

    sums = [jnp.zeros((block_k,), jnp.float32) if kind == Kind.KEY else jnp.zeros((), jnp.float32)
            for kind in score_kinds]
    zeros = jnp.zeros(key.shape, jnp.float32), jnp.zeros(value.shape, jnp.float32)
    dk, dv, *grads = jax.lax.fori_loop(ranges_ref[0], ranges_ref[1], step, (*zeros, *sums))
    dk_ref[...] = dk
    dv_ref[...] = dv
    for place, (grad, kind) in enumerate(zip(grads, score_kinds, strict=True)):
        if kind == Kind.KEY:
            grad_refs[place][...] = grad


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


BACKWARD_DEFAULT = BackwardTiles(64, 64, 4, 2, 4, 2)
"""The backward config first in its space, which a pick of tied estimates takes: within 7% of the fastest
joint config of both launches at a llama shape and at head width 64, causal, on an A100 and an H100
(DPROF3-a100 and DPROF3-h100, 2026-10-21)."""

BACKWARD_SPACE = (BACKWARD_DEFAULT, *(config for config in (
    BackwardTiles(tiles.block_q, tiles.block_k, tiles.warps, tiles.stages, warps, stages)
    for tiles in SPACE for warps in (4, 8) for stages in (2, 3)) if config != BACKWARD_DEFAULT))
"""Every config of the backward: each of `SPACE`'s tiles for its dQ launch, with each warps and stages of
its dK/dV launch, its default first."""


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
    split: bool = True

    def _fwd(self, query, key, value, dout, peak, total, initial_term, maximum, key_index, anchor_value,
             others, own, others_total, forward_total, start, end, query_ids, key_ids, score_params, *,
             interpret: bool, config: BackwardTiles, return_residuals: bool = False):
        shared = {"score_mod": self.score_mod, "score_kinds": self.score_kinds, "scale": self.scale,
                  "block_q": config.block_q, "block_k": config.block_k, "interpret": interpret}
        spans = start, end, query_ids, key_ids, score_params
        sums = (others, own, others_total, forward_total) if self.split else None
        dq_out = _dq_launch(query, key, value, dout, peak, total, initial_term, maximum, key_index,
                            anchor_value, sums, *spans, warps=config.dq_warps, stages=config.dq_stages,
                            **shared)
        delta, anchor, peak, total = dq_out[1:5]
        # Each query's anchor key as the dQ launch chose it, -1 where it took D about 0.
        dkv_out = _dkv_launch(query, key, value, dout, peak, total, anchor, delta, dq_out[6], *spans,
                              warps=config.dkv_warps, stages=config.dkv_stages, **shared)
        return (*dq_out, *dkv_out), None

    def space(self, ba: tokamax.BoundArguments) -> Sequence[BackwardTiles]:
        return _dividing(BACKWARD_SPACE, ba)

    def staged_bytes(self, ba: tokamax.BoundArguments, config: BackwardTiles) -> int:
        """The shared memory Triton gives a program of either launch, the larger, in the operands' dtype: a
        dQ program its query and output-gradient tiles and the key and value tiles of every stage but the
        one in use; a dK/dV program its key and value tiles and the query and output-gradient tiles of
        every stage but the one in use; and two `block_q` by `block_k` tiles on their way between
        products. A bound fitted to the bytes Triton asks for (SMEM-a100 and SMEM-a10, 2026-10-21: at or
        above them on every config it refused, within the budget on all but two of the A10's compiled
        ones)."""
        query, value = ba.arguments["query"], ba.arguments["value"]
        size = jnp.dtype(query.dtype).itemsize
        width, value_width = query.shape[-1], value.shape[-1]
        rows = max(config.block_q + (config.dq_stages - 1) * config.block_k,
                   config.block_k + (config.dkv_stages - 1) * config.block_q)
        return (rows * (width + value_width) + 2 * config.block_q * config.block_k) * size

    def cost(self, ba: tokamax.BoundArguments, config: BackwardTiles, target: generation.GpuTarget) -> float:
        """Both launches' (`_launch_cost`): every tile of the call (a bound: the spans are traced, so a
        sparse call visits fewer)."""
        return sum(_launch_cost(ba, config, target, self.score_kinds, dq=dq) for dq in (True, False))

    def eligible(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
        return _eligible(ba, target)

    def operands(self, specs: list[jax.ShapeDtypeStruct], seed: int) -> list[jax.Array]:
        return _spanned(specs, super().operands(specs, seed))

    def traced(self, name: str, value: Callable | str | None) -> str | None:
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
        # fp16's dV product runs in fp32 (`algebra.value_operands`), after the launch's 16-bit work.
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


def _geometry(query, key, value, start, end, query_ids, key_ids, block_q: int, block_k: int):
    """A backward launch's sizes (rows, padded queries, query heads, widths, padded keys, a key head's
    group) and the spans' block map at its tiles."""
    batch, queries, heads, width = query.shape
    keys, kv_heads, value_width = key.shape[1], key.shape[2], value.shape[-1]
    blocks = block_map(KeySpans(start, end, query_ids, key_ids, keys), block_q, block_k)
    return batch, queries, heads, width, keys, value_width, heads // kv_heads, blocks


def _dq_launch(query, key, value, dout, peak, total, initial_term, maximum, key_index, anchor_value, sums,
               start, end, query_ids, key_ids, score_params, *, score_mod, score_kinds: tuple[Kind, ...],
               scale: float, block_q: int, block_k: int, warps: int, stages: int,
               interpret: bool) -> tuple[jax.Array, ...]:
    """The backward's dQ launch over padded operands (`backward` pads them): a program per query block
    and query head takes each query's D about 0 or about the anchor key `key_index`'s `anchor_value · dO`,
    whichever is nearer, from the forward's split sums (`sums`, `algebra.split_delta`), or where None from
    a first walk of the key blocks the block map gives it, weighed at the forward's `maximum` from the
    initial state's maximum, denominator and share of `D`'s numerator (`peak`, `total`, `initial_term`;
    `algebra.off_anchor_delta`); then walks them for dQ under the weights `exp(s - m)`, normalized at the
    end by their own sum `T`, refining `D - c` from its own sums. Returns dQ
    `[B, Qp, H, Dp]` fp32, `D - c`, `c`, `m`, `T` and the keys' mass `[B, H, Qp]`, the anchor key it took
    (-1 where about 0) `[B, H, Qp]` int32, and the gradients of the score hook's head (`[nQb, B, H, 1]`
    partials) and query (`[B, H, Qp]`) parameters."""
    from jax.experimental.pallas import triton as plgpu

    batch, queries, heads, width, keys, value_width, group, blocks = _geometry(
        query, key, value, start, end, query_ids, key_ids, block_q, block_k)
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
                per_head, per_head, per_head, per_head, per_head,
                pl.BlockSpec((None, block_q, None, value_width), lambda i, b, h: (b, i, h, 0))]
    arrays = [bounds, start, end, *([query_ids, key_ids] if ids else []), query, key, value, dout, peak,
              total, initial_term, maximum, key_index, anchor_value]
    if sums is not None:
        # The forward's split sums `[B, H, Qp, Dvp]` and `[B, H, Qp]` twice, and its denominator.
        in_specs += [pl.BlockSpec((None, None, block_q, value_width), lambda i, b, h: (b, h, i, 0)),
                     per_head, per_head, per_head]
        arrays += list(sums)
    for param, kind in zip(score_params, score_kinds, strict=True):
        in_specs.append(_param_spec(kind, param, block_q, keys))
        arrays.append(param)
    out_shape = [jax.ShapeDtypeStruct(query.shape, jnp.float32),
                 *[jax.ShapeDtypeStruct((batch, heads, queries), jnp.float32)] * 5,
                 jax.ShapeDtypeStruct((batch, heads, queries), jnp.int32)]
    out_specs = [pl.BlockSpec((None, block_q, None, width), lambda i, b, h: (b, i, h, 0)),
                 *[per_head] * 6]
    for param, kind in zip(score_params, score_kinds, strict=True):
        match kind:
            case Kind.HEAD:
                out_shape.append(jax.ShapeDtypeStruct((queries // block_q, batch, heads, 1), jnp.float32))
                out_specs.append(pl.BlockSpec((None, None, None, 1), lambda i, b, h: (i, b, h, 0)))
            case Kind.QUERY:
                out_shape.append(jax.ShapeDtypeStruct(param.shape, jnp.float32))
                out_specs.append(per_head)
    kernel = functools.partial(_dq_kernel, ids=ids, score_kinds=score_kinds,
                               score_mod=score_mod, scale=scale, block_k=block_k, split=sums is not None,
                               interpret=interpret)
    out = pl.pallas_call(
        kernel, grid=(queries // block_q, batch, heads), in_specs=in_specs, out_specs=out_specs,
        out_shape=out_shape, interpret=interpret, name="attention_dq",
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
    )(*arrays)
    return tuple(out)


def _dkv_launch(query, key, value, dout, peak, total, anchor, delta, key_index, start, end, query_ids,
                key_ids, score_params, *, score_mod, score_kinds: tuple[Kind, ...], scale: float,
                block_q: int, block_k: int, warps: int, stages: int,
                interpret: bool) -> tuple[jax.Array, ...]:
    """The backward's dK/dV launch over padded operands: a program per key block and query head walks the
    query blocks the block map gives it, under the dQ launch's softmax at the dQ launch's tiles (a
    product rounds by its shape, so the logits it weighs each entry by are the very ones the dQ launch
    normalized); returns the query head's dK and dV `[B, Kp, H, Dp]` fp32, which `backward` sums over
    each key head's group, and the gradients of the score hook's key (`[B, H, Kp]`) and tile
    (`[B, H, Qp, Kp]`) parameters."""
    from jax.experimental.pallas import triton as plgpu

    batch, queries, heads, width, keys, value_width, group, blocks = _geometry(
        query, key, value, start, end, query_ids, key_ids, block_q, block_k)
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
                *[whole_head] * 5]
    arrays = [ranges, start, end, *([query_ids, key_ids] if ids else []), query, key, value, dout, peak,
              total, anchor, delta, key_index]
    for param, kind in zip(score_params, score_kinds, strict=True):
        in_specs.append(_key_param_spec(kind, param, queries, block_k))
        arrays.append(param)
    out_shape = [jax.ShapeDtypeStruct((batch, keys, heads, width), jnp.float32),
                 jax.ShapeDtypeStruct((batch, keys, heads, value_width), jnp.float32)]
    out_specs = [pl.BlockSpec((None, block_k, None, width), lambda j, b, h: (b, j, h, 0)),
                 pl.BlockSpec((None, block_k, None, value_width), lambda j, b, h: (b, j, h, 0))]
    aliases = {}
    for param, kind in zip(score_params, score_kinds, strict=True):
        if kind == Kind.KEY:
            out_shape.append(jax.ShapeDtypeStruct(param.shape, jnp.float32))
            out_specs.append(pl.BlockSpec((None, None, block_k), lambda j, b, h: (b, h, j)))
        elif kind == Kind.TILE:
            # Entries no program visits keep their zeros: the output aliases a zero input.
            columns = pl.BlockSpec((None, None, queries, block_k), lambda j, b, h: (b, h, 0, j))
            aliases[len(arrays)] = len(out_shape)
            in_specs.append(columns)
            arrays.append(jnp.zeros(param.shape, jnp.float32))
            out_shape.append(jax.ShapeDtypeStruct(param.shape, jnp.float32))
            out_specs.append(columns)
    kernel = functools.partial(_dkv_kernel, ids=ids, score_kinds=score_kinds,
                               score_mod=score_mod, scale=scale, block_q=block_q,
                               interpret=interpret)
    out = pl.pallas_call(
        kernel, grid=(keys // block_k, batch, heads), in_specs=in_specs, out_specs=out_specs,
        out_shape=out_shape, input_output_aliases=aliases, interpret=interpret, name="attention_dkv",
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


def _launch(*arrays, score_mod, score_kinds: tuple[Kind, ...], score_dtypes: tuple[str, ...], scale: float,
            split: bool, interpret: bool):
    """The Triton mainloop's two launches under one tuning decision (`TritonBackward`)."""
    op = TritonBackward(score_mod=score_mod, score_kinds=score_kinds, scale=scale, score_dtypes=score_dtypes,
                        split=split)
    return op(*arrays, interpret=interpret)


def backward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None,
             dout: jax.Array, *, residuals: tuple[jax.Array, jax.Array, jax.Array, Split] | None = None,
             split: bool = True, scale: float | None = None, interpret: bool = False) -> Gradients:
    """The template's backward on the Triton mainloop (`gradients`, `TritonBackward`), from the forward's
    `residuals`: its maxima, denominators, anchor keys and split sums (`forward`'s, which it takes again
    where None). With `split`, D comes from the split sums; without, the dQ launch takes its own first pass
    over the keys (`algebra.off_anchor`). `interpret` runs it on the CPU."""
    if residuals is None:
        plain = Hooks() if hooks is None else Hooks(hooks.score_mod, hooks.score_params, hooks.initial)
        _, maxima, totals, anchors, sums = forward(query, key, value, spans, plain, scale=scale,
                                                   residuals=True, interpret=interpret)
        residuals = maxima, totals, anchors, sums
    return gradients(functools.partial(_launch, split=split, interpret=interpret), query, key, value, spans,
                     hooks, dout, residuals, block=BLOCK, scale=scale)
