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
from typing import Literal, overload

import jax
import jax.numpy as jnp
import tokamax
from jax.experimental import pallas as pl

from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention_algebra as algebra
from dew.nn.kernels.templates.attention import (
    ACCURATE,
    DKV_GRADS,
    DQ_GRADS,
    NATURAL,
    TRACED_TILE,
    AttentionLaunch,
    Gradients,
    Hooks,
    KeySpans,
    Kind,
    Split,
    block_map,
    by_power,
    gradients,
    kept_dtype,
    kernel_dot,
    launch_arguments,
    padded_call,
    prescaled,
    within,
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
"""The forward config first in its space, the launch's own where it fits (`Template.fitting`'s first,
nothing ranked): within 6% of the fastest at Qwen3-8B, causal and full, on an A100 and an H100
(ATT-ZERO3-a100 and ATT-ZERO4-h100, 2026-10-21), where the space's first config otherwise, 32 by 32, ran
2-3 times it."""

SPACE = (FORWARD_DEFAULT, *(tiles for tiles in (Tiles(block_q, block_k, warps, stages)
                                                for block_q in (32, 64, 128) for block_k in (32, 64, 128)
                                                for warps in (4, 8) for stages in (2, 3))
                            if tiles != FORWARD_DEFAULT))
"""Every config the forward is proven over (tests/test_attention_triton.py pins each), its default
first."""

BLOCK = max(max(tiles.block_q for tiles in SPACE), max(tiles.block_k for tiles in SPACE))
"""The largest tile of `SPACE`, the block a call pads to (`padded_call`)."""


def _eligible(ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
    """Whether a launch runs on `target`: a CUDA GPU, one dtype of `DTYPES` on the queries, keys and
    values, and bf16 only from `generation.BF16_GPU` on, where Triton compiles its products; spans of one
    interval a query (`refused`)."""
    query, key, value = (ba.arguments[name] for name in ("query", "key", "value"))
    capability = generation.capability(target.generation)
    return (capability is not None and query.dtype == key.dtype == value.dtype and query.dtype in DTYPES
            and (query.dtype != jnp.bfloat16 or 10 * capability[0] + capability[1] >= generation.BF16_GPU)
            and ba.arguments["start"].ndim == 2)


def refused(spans: KeySpans, hooks: Hooks | None = None) -> str | None:
    """Why this mainloop takes no call over `spans` under `hooks`, or None: a call's runs, several key
    intervals a query (`KeySpans`), which the frozen mainloop does not walk, the Mosaic and plain-JAX
    mainloops do (ledger 956, `attention_route`); probability dropout (`Hooks.drop`), which it does not
    draw, the plain-JAX mainloop does (ledger row 1521)."""
    if spans.start.ndim == 3:
        return ("the Triton mainloop walks one key interval a query; a call's runs take the Mosaic or the "
                "plain-JAX mainloop")
    if hooks is not None and hooks.drop is not None:
        return ("the Triton mainloop is frozen and takes no probability dropout (the route runs a dropping "
                "call on the reference path, ledger row 1521)")
    return None


def unfitted(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
             *, scale: float | None = None, natural: bool = NATURAL, residuals: bool = False) -> str | None:
    """Why no config of `forward`'s launch, bound as it binds this call (`_forward_call`), fits the GPU: its
    own fitting (`Template.fits`: its padding, the configs dividing it, the registers they hold and the
    shared memory they stage), or None. The route then takes another mainloop rather than a launch that
    cannot run (`attention_route._mainloop`); the backward's own launches hold the same tiles' operands."""
    op, args, _ = _forward_call(query, key, value, spans, hooks, scale=scale, natural=natural)
    if op.fits(op.bind(*args, residuals=residuals, interpret=False), generation.gpu_target()):
        return None
    return "no config of the Triton mainloop's forward fits the GPU's block and registers"


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
                    interpret: bool, guard: str):
    """The forward's Triton mainloop: one program a query block and head walks the key tiles its form
    gives it (every tile, the causal diagonal's, or the spans' block map), issuing the tile's two
    products and calling the algebra between them (`attention_algebra`). Under headroom
    (`algebra.Forward.headroom`) the plain launch (`guard` 'flag') also writes which of its rows'
    weighted values overflowed, and a second launch (`guard` 'rewalk', the plain launch's outputs its
    own) walks again, each row shifting as it needs, only the programs one of whose rows did; the plain
    walk itself is unchanged (`guard` 'none' where no row can overflow)."""
    refs = list(refs)
    spans = bounds = ids = None
    if spec.form == "spans":
        # Only spans read their block map's bounds and their queries' intervals.
        bounds_ref = refs.pop(0)
        bounds = bounds_ref[0], bounds_ref[1]
        spans = refs.pop(0)[...], refs.pop(0)[...]
    if spec.ids:
        # The queries' ids, and the keys' ref a tile reads its own from.
        ids = refs.pop(0)[...], refs.pop(0)
    query_ref, key_ref, value_ref = refs.pop(0), refs.pop(0), refs.pop(0)
    score_refs = [refs.pop(0) for _ in score_kinds]
    output_refs = [refs.pop(0) for _ in output_kinds]
    initial_refs = [refs.pop(0) for _ in range(3 if initial else 0)]
    flags_ref = refs.pop(0) if guard == "rewalk" else None
    if guard == "rewalk":
        # The plain launch's outputs enter as inputs aliased to this launch's outputs, which follow them
        # one for one (`input_output_aliases`); the program writes the outputs and reads no input of them.
        aliased = len(refs) // 2
        refs = refs[aliased:]
    out_ref, *residual_refs = refs
    flag_ref = residual_refs.pop() if guard == "flag" else None
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

    def keys_at(index) -> algebra.Keys:
        # A key tile's logits; its values load where the step calls for them, after the softmax, where
        # jax's mha loads them.
        tile = pl.ds(index * block_k, block_k)
        coords = algebra.Coords(q_pos, index * block_k + jnp.arange(block_k)[None, :],
                                None if spans is None else (spans,),
                                None if ids is None else (ids[0], ids[1][tile]))
        params = tuple(_tile_param(ref, kind, head, q_pos, tile)
                       for ref, kind in zip(score_refs, score_kinds, strict=True))
        logits = algebra.modify(spec, kernel_dot(query, key_ref[tile, :].T, interpret), coords, params)
        return algebra.Keys(coords, logits, lambda: value_ref[tile, :],
                            functools.partial(kernel_dot, interpret=interpret))

    def rows(keys):
        # Each row's value row at its key (the anchor's).
        return value_ref[keys, :]

    def walk(live=None):
        # The plain walk where `live` is None, else the guarded one, those rows shifting.
        resumed = (initial_refs[0][...], initial_refs[1][...], initial_refs[2][...]) if initial else None
        start = algebra.started(spec, resumed, block_q, value_ref.shape[-1], live)
        if residuals:
            return algebra.training_walk(spec, lo, hi, keys_at, rows, start)
        return algebra.plain_walk(spec, lo, hi, keys_at, start)

    output_params = tuple(_tile_param(ref, kind, head, q_pos, None)
                          for ref, kind in zip(output_refs, output_kinds, strict=True))

    def normalized(walked):
        # A guarded training walk's live rows take their split sums from a normalized pass at the walk's
        # maximum and denominator; the plain walk takes none.
        resumed = (initial_refs[0][...], initial_refs[2][...]) if initial else None
        return algebra.normalized_pass(spec, walked, keys_at, rows, (lo, hi), resumed)

    written = functools.partial(_written, spec, rows=rows, out_ref=out_ref, residual_refs=residual_refs,
                                flag_ref=flag_ref, residuals=residuals, initial=initial,
                                output_mod=output_mod, output_params=output_params, normalized=normalized)

    if flags_ref is None:
        written(walk())
        return
    live = flags_ref[...] > 0

    @pl.when(jnp.max(live.astype(jnp.int32)) > 0)
    def _():
        # Only a program one of whose rows overflowed walks again: each of those rows shifts as it needs,
        # every other row's bits the plain walk's (`algebra.room`).
        written(walk(live))


def _written(spec, walked, *, rows, out_ref, residual_refs, flag_ref, residuals: bool, initial: bool,
             output_mod, output_params, normalized: Callable):
    """A forward walk's output and residuals written: which rows overflowed where the launch flags them,
    the anchors and split sums of a training walk (`algebra.split_written`, a guarded one's live rows'
    from the `normalized` pass), and the normalized output with its maximum and denominator."""
    if flag_ref is not None:
        weighted = algebra.joined(walked, rows).acc if residuals else walked.acc
        flag_ref[...] = algebra.overflowed(weighted).astype(jnp.float32)
    if residuals:
        walked, anchors, sums = algebra.split_written(walked, rows, functools.partial(normalized, walked))
        residual_refs[2][...] = anchors
        for ref, part in zip(residual_refs[3:7], sums, strict=True):
            ref[...] = part
    out, m, denominator = algebra.finish(spec, walked, initial=initial, output_mod=output_mod,
                                         output_params=output_params)
    out_ref[...] = out.astype(out_ref.dtype)
    if residuals:
        residual_refs[0][...] = m
        residual_refs[1][...] = denominator


@dataclasses.dataclass(frozen=True)
class TritonForward(AttentionLaunch, Template[Tiles]):
    """The template's forward in Pallas Triton over padded operands (`forward` pads them): queries
    `[B, Qp, H, Dp]` over keys `[B, Kp, Hk, Dp]` and values `[B, Kp, Hk, Dvp]`, each query's keys
    `start` to `end` `[B, Qp]` and, with documents, its id `[B, Qp]` against the keys' `[B, Kp]`; the
    score and output hooks' parameters (`score_params`, `output_params`, laid out by `score_kinds` and
    `output_kinds`) and the softmax's initial state. Returns the output `[B, Qp, H, Dvp]` and, with
    `residuals`, each query's running maximum and denominator `[B, H, Qp]`, fp32, and the first key
    holding the keys' maximum `[B, H, Qp]` int32, the backward's anchor (`algebra.Argmax`). `natural`
    keeps the logits natural (`algebra.Forward.base2`)."""

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
    natural: bool = NATURAL

    def _fwd(self, query, key, value, start, end, query_ids, key_ids, score_params, output_params, m0, total0,
             acc0, *, residuals: bool, interpret: bool, config: Tiles, return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        # The query taken by its scale's power of two before its products (`prescale`).
        query, scale, power = prescaled(query, self.scale)
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
                          *[jax.ShapeDtypeStruct((batch, heads, queries), jnp.float32)] * 3]
            per_value = pl.BlockSpec((None, None, block_q, value_width), lambda i, b, h: (b, h, i, 0))
            out_specs += [per_row, per_row, per_row, per_value, per_row, per_row, per_row]
        # A call's own key count, none included; an op built without one counts its launch's keys.
        # Every form guards its sums where its values can overflow them (`algebra.Forward.headroom`), as the
        # plain-JAX and Mosaic mainloops do: the template's production route runs this one on a GPU.
        spec = algebra.Forward(self.form, keys if self.keys is None else self.keys, ids, scale,
                               self.score_mod, self.natural, algebra.headroom(value.dtype), initial,
                               compensated=algebra.compensates(value.dtype), prescale=power)
        kernel = functools.partial(
            _forward_kernel, spec=spec, score_kinds=self.score_kinds, output_kinds=self.output_kinds,
            initial=initial, residuals=residuals, output_mod=self.output_mod, block_k=block_k,
            interpret=interpret)
        launch = functools.partial(
            pl.pallas_call, grid=(queries // block_q, batch, heads), interpret=interpret,
            compiler_params=plgpu.CompilerParams(num_warps=config.warps, num_stages=config.stages))
        if not spec.headroom:
            out = launch(functools.partial(kernel, guard="none"), in_specs=in_specs, out_specs=out_specs,
                         out_shape=out_shape, name="attention_forward")(*arrays)
            return (tuple(out) if residuals else out[0]), None
        # The plain launch writes which rows overflowed; the second, issued only where one did, walks again
        # only the programs with one, its outputs the plain launch's (`_forward_kernel`): ordinary inputs
        # take the one launch and a check.
        per_row = pl.BlockSpec((None, None, block_q), lambda i, b, h: (b, h, i))
        flagged = jax.ShapeDtypeStruct((batch, heads, queries), jnp.float32)
        *out, flags = launch(functools.partial(kernel, guard="flag"), in_specs=in_specs,
                             out_specs=[*out_specs, per_row], out_shape=[*out_shape, flagged],
                             name="attention_forward")(*arrays)
        first = len(arrays) + 1

        def rewalked(out):
            return tuple(launch(functools.partial(kernel, guard="rewalk"),
                                in_specs=[*in_specs, per_row, *out_specs], out_specs=out_specs,
                                out_shape=out_shape, name="attention_forward_headroom",
                                input_output_aliases={first + index: index for index in range(len(out))},
                                )(*arrays, flags, *out))

        out = jax.lax.cond(jnp.max(flags) > 0, rewalked, lambda out: out, tuple(out))
        return (tuple(out) if residuals else out[0]), None

    def space(self, ba: tokamax.BoundArguments) -> Sequence[Tiles]:
        return _dividing(SPACE, ba)

    def staged_bytes(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget) -> int:
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

    def registers(self, ba: tokamax.BoundArguments, config: Tiles) -> int:
        """The fp32 words a thread holds live across the walk, the program's spread over its threads: the
        block's weighted values `[block_q, Dv]` and a tile's logits `[block_q, block_k]`, in a compensated
        walk what the values' sums dropped beside them (`algebra.Partial.low`, fp32 values'), and in a
        training walk the anchor's value row, `[block_q, Dv]` (`algebra.training_step`). Measured where
        it passes a thread's registers: the training walk at 128 by 64 in four warps, 320 a thread, takes
        10x the plain forward's time on an A100 and 4.6x on an H100, at 64 by 64 2.1x and 2.0x
        (TRAINTILES3-a100, TRAINTILES2-h100)."""
        rows, value = config.block_q, ba.arguments["value"]
        sums = 2 if algebra.compensates(value.dtype) else 1
        words = sums * rows * value.shape[-1] + rows * config.block_k
        if ba.arguments.get("residuals"):
            words += rows * value.shape[-1]
        return words // (32 * config.warps)

    def eligible(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
        return _eligible(ba, target)


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


@overload
def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, residuals: Literal[False] = False, natural: bool = NATURAL,
            interpret: bool = False) -> jax.Array:
    ...


@overload
def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, residuals: Literal[True], natural: bool = NATURAL,
            interpret: bool = False) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, Split]:
    ...


def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, residuals: bool = False, natural: bool = NATURAL,
            interpret: bool = False):
    """The template's forward (`TritonForward`) on `query` `[B, Q, H, D]` over `key` and `value`
    `[B, K, Hk, D]` and `[B, K, Hk, Dv]` and the keys `spans` leaves each query: as
    `dew.nn.kernels.templates.attention.reference` computes it, with `residuals` its running maxima and
    denominators `[B, H, Q]` too, and each query's anchor key `[B, H, Q]` int32 (`backward` reads them
    all). The operands pad to whole blocks of the largest tile and to power-of-two
    head widths (`padded_call`). `natural` keeps the logits natural (`algebra.Forward.base2`).
    `interpret` runs it on the CPU. A call over runs or under dropout is refused (`refused`)."""
    if reason := refused(spans, hooks):
        raise NotImplementedError(reason)
    op, args, queries = _forward_call(query, key, value, spans, hooks, scale=scale, natural=natural)
    value_width = value.shape[-1]
    out = op(*args, residuals=residuals, interpret=interpret)
    if not residuals:
        return out[:, :queries, :, :value_width]
    out, peak, total, anchors, others, own, others_total, shift = out
    return (out[:, :queries, :, :value_width], peak[:, :, :queries], total[:, :, :queries],
            anchors[:, :, :queries],
            Split(others[:, :, :queries, :value_width], own[:, :, :queries], others_total[:, :, :queries],
                  shift[:, :, :queries]))


def _forward_call(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None,
                  *, scale: float | None, natural: bool) -> tuple[TritonForward, tuple, int]:
    """`forward`'s launch, its padded operands (`padded_call`) and the call's queries before padding."""
    hooks = Hooks() if hooks is None else hooks
    call = padded_call(query, key, value, spans, hooks, BLOCK)
    initial = hooks.initial
    m0 = total0 = acc0 = None
    if initial is not None:
        m0, total0 = call.rows(initial.m, -jnp.inf), call.rows(initial.total)
        acc0 = jnp.pad(call.rows(initial.acc),
                       ((0, 0),) * 3 + ((0, call.value.shape[-1] - value.shape[-1]),))
    op = TritonForward(score_mod=hooks.score_mod, output_mod=hooks.output_mod,
                       score_kinds=tuple(param.kind for param in hooks.score_params),
                       output_kinds=tuple(param.kind for param in hooks.output_params),
                       score_dtypes=tuple(kept_dtype(param.value.dtype) for param in hooks.score_params),
                       output_dtypes=tuple(kept_dtype(param.value.dtype) for param in hooks.output_params),
                       output_width=call.value.shape[-1],
                       scale=query.shape[3] ** -0.5 if scale is None else scale, form=spans.form,
                       keys=spans.keys, natural=natural)
    return op, (call.query, call.key, call.value, call.start, call.end, call.query_ids, call.key_ids,
                call.score_params, call.output_params, m0, total0, acc0), call.queries


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


def _tile_scores(score_mod, query, key, value, dout, peak, total, anchor, delta, seen, q_pos, k_pos, params,
                 *, key_index, scale: float, how: algebra.Recompute, interpret: bool):
    """A tile's backward through its scores: its unnormalized weights `exp(s - m)` and its weights `p` over
    `total` (zero where masked), the masked score cotangents `p * ((dP - c) - (D - c))`
    (`algebra.grad_scores`), `dP = dO V^T`, `c` each query's `anchor` and `delta` its `D - c`, and the
    entries' derivatives through `score_mod` (`algebra.scored`), its weights recomputed as its forward
    computed them (`how`)."""
    logits, slopes = algebra.tile_logits(score_mod, kernel_dot(query, key.T, interpret), seen, q_pos, k_pos,
                                         params, scale=scale if how.factor is None else 1.0, slopes=True)
    exponentials, weights = algebra.normalized(logits, peak, total, how)
    cotangents = algebra.grad_scores(exponentials, kernel_dot(dout, value.T, interpret), anchor, delta, seen,
                                     k_pos == key_index[:, None], algebra.inverse(total))
    return exponentials, weights, cotangents, slopes


def _dq_kernel(bounds_ref, start_ref, end_ref, *refs, ids: bool, score_kinds: tuple[Kind, ...], score_mod,
               scale: float, prescale: float, grad_scale: float, how: algebra.Recompute, block_k: int,
               split: bool, accurate: bool, interpret: bool):
    refs = list(refs)
    id_refs = [refs.pop(0), refs.pop(0)] if ids else []
    (query_ref, key_ref, value_ref, dout_ref, peak_ref, total_ref, initial_ref, maximum_ref, index_ref,
     anchor_value_ref, shift_ref) = (refs.pop(0) for _ in range(11))
    # The forward's split sums and denominator, where it gives them (`algebra.split_delta`).
    split_refs = [refs.pop(0) for _ in range(4 if split else 0)]
    score_refs = [refs.pop(0) for _ in score_kinds]
    dq_ref, delta_ref, anchor_ref, peak_out_ref, total_out_ref, mass_ref, index_out_ref, *outputs = refs
    grad_refs = {place: outputs.pop(0) for place, kind in enumerate(score_kinds) if kind in DQ_GRADS}
    block_q = query_ref.shape[0]
    head = pl.program_id(2)
    q_pos = pl.program_id(0) * block_q + jnp.arange(block_q)[:, None]
    # The logits from the query taken by its power, as the forward took them (`prescale`).
    query, dout = by_power(query_ref[...], prescale), dout_ref[...]
    peak, total = peak_ref[...], total_ref[...]
    product = functools.partial(kernel_dot, interpret=interpret)
    start, end = start_ref[...], end_ref[...]
    query_ids = [ref[...] for ref in id_refs[:1]]

    def tile(index):
        keys = pl.ds(index * block_k, block_k)
        k_pos = index * block_k + jnp.arange(block_k)[None, :]
        seen = within(k_pos, start[:, None], end[:, None],
                      (query_ids[0][:, None], id_refs[1][keys][None, :]) if ids else None)
        params = [_tile_param(ref, kind, head, q_pos, keys)
                  for ref, kind in zip(score_refs, score_kinds, strict=True)]
        return keys, k_pos, seen, params

    maximum = maximum_ref[...]

    def row_terms(index, state):
        # Every weight against the forward's maximum m, so no running maximum rescales a sum
        # (`algebra.OffAnchor`); the weights are this launch's own, exp(s - m) over its logits, so they
        # sum to one over them whatever logits the forward took.
        keys, k_pos, seen, params = tile(index)
        logits, _ = algebra.tile_logits(score_mod, kernel_dot(query, key_ref[keys, :].T, interpret), seen,
                                        q_pos, k_pos, params, scale=scale if how.factor is None else 1.0)
        value = value_ref[keys, :]
        exponentials, rest, own, small = algebra.off_anchor_weights(logits, maximum,
                                                                    k_pos == key_index[:, None], value.dtype,
                                                                    shift, how)
        return algebra.off_anchor(state, exponentials, rest, own, kernel_dot(rest, value, interpret), shift,
                                  small, kernel_dot(small, value, interpret))

    # From the softmax's initial state (`gradients`): its maximum, denominator and share of D's numerator.
    key_index = index_ref[...]
    peak0, shares = peak, (total, initial_ref[...])
    shift = shift_ref[...]
    before, state_total = algebra.off_anchor_start(peak, total, maximum, value_ref.shape[-1], shift)
    if split:
        # The forward took the first pass (`algebra.training_step`).
        others_ref, own_ref, others_total_ref, forward_total_ref = split_refs
        prior = forward_total_ref[...]
        anchor, delta, key_index = algebra.split_delta(
            others_ref[...], own_ref[...], others_total_ref[...], prior, state_total, anchor_value_ref[...],
            dout, key_index, shift)
    else:
        state = jax.lax.fori_loop(bounds_ref[0], bounds_ref[1], row_terms, before)
        prior = algebra.power2(state.total, shift)
        anchor, delta, key_index = algebra.off_anchor_delta(state, anchor_value_ref[...], dout, state_total,
                                                            shares[1], key_index, shift)
    peak = maximum
    index_out_ref[...] = key_index
    summed = tuple(kind in (Kind.HEAD, Kind.QUERY) for kind in score_kinds)

    def step(index, carry):
        # Each cotangent over the prior denominator, in the reference's order, and taken to the pass's
        # own at the end (`algebra.dq_finish`); the weights' sums unnormalized.
        keys, k_pos, seen, params = tile(index)
        key, value = key_ref[keys, :], value_ref[keys, :]
        weights, _, cotangents, slopes = _tile_scores(score_mod, query, key, value, dout, peak, prior, anchor,
                                                   delta, seen, q_pos, k_pos, params, key_index=key_index,
                                                   scale=scale, how=how, interpret=interpret)
        # The cotangents at the call's whole scale, its logits at the scale its query's power left
        # (`prescale`).
        return algebra.dq_step(carry, weights, cotangents, slopes, seen, key, prior, scale=grad_scale,
                               kinds=score_kinds, summed=summed, dot=product, accurate=accurate)

    sums = jax.lax.fori_loop(bounds_ref[0], bounds_ref[1], step,
                             algebra.no_dq_sums(block_q, query.shape[-1], len(score_kinds)))
    dq, delta, total, mass, grads = algebra.dq_finish(sums, prior, state_total, peak0, shares, peak, anchor,
                                                      delta, accurate=accurate)
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
                scale: float, prescale: float, grad_scale: float, how: algebra.Recompute, block_q: int,
                accurate: bool, interpret: bool):
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
    def transposed(a, b):
        # A query tile's share of the keys' gradients, `aᵀ @ b` (`algebra.cotangent_product`).
        return kernel_dot(a.T, b, interpret)

    def step(index, carry):
        dk, dv, *grads = carry
        rows = pl.ds(index * block_q, block_q)
        query, dout = query_ref[rows, :], dout_ref[rows, :]
        q_pos = index * block_q + jnp.arange(block_q)[:, None]
        seen = within(k_pos, start_ref[rows][:, None], end_ref[rows][:, None],
                      (id_refs[0][rows][:, None], key_ids[0][None, :]) if ids else None)
        params = [_key_param(ref, kind, head, rows)
                  for ref, kind in zip(score_refs, score_kinds, strict=True)]
        exponentials, weights, cotangents, slopes = _tile_scores(
            score_mod, by_power(query, prescale), key, value, dout, peak_ref[rows], total_ref[rows],
            anchor_ref[rows], delta_ref[rows], seen, q_pos, k_pos, params, key_index=index_ref[rows],
            scale=scale, how=how, interpret=interpret)
        logits, pieces = algebra.pulled(cotangents, slopes, seen, score_kinds)
        dv = dv + transposed(*algebra.value_operands(exponentials, weights, dout, total_ref[rows]))
        # Scaled before it meets the queries, in the backward's form (`algebra.cotangent_product`), at the
        # call's whole scale against the call's own queries, the logits' alone taken by the power.
        dk = dk + algebra.cotangent_product(logits * grad_scale, query, transposed, accurate=accurate)
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
"""The backward config first in its space, the launch's own where it fits: within 7% of the fastest
joint config of both launches at a llama shape and at head width 64, causal, on an A100 and an H100
(DPROF3-a100 and DPROF3-h100, 2026-10-21)."""

BACKWARD_SPACE = (BACKWARD_DEFAULT, *(config for config in (
    BackwardTiles(tiles.block_q, tiles.block_k, tiles.warps, tiles.stages, warps, stages)
    for tiles in SPACE for warps in (4, 8) for stages in (2, 3)) if config != BACKWARD_DEFAULT))
"""Every config of the backward: each of `SPACE`'s tiles for its dQ launch, with each warps and stages of
its dK/dV launch, its default first."""


@dataclasses.dataclass(frozen=True)
class TritonBackward(AttentionLaunch, Template[BackwardTiles]):
    """The template's backward over padded operands (`backward` pads them), one tuning decision for its
    two launches: the dQ launch (`_dq_launch`), then the dK/dV launch (`_dkv_launch`) under its softmax.
    Returns the dQ launch's outputs, then the dK/dV launch's."""

    HOOKS = ("score_mod",)
    DIFFERENTIATED = True
    SPANS = 15
    """Its arrays' spans after the queries, keys, values, the output's cotangent and the forward's eleven
    residual and state arrays (`_fwd`)."""
    name = "triton"
    config_cls = BackwardTiles
    score_mod: Callable | None = None
    score_kinds: tuple[Kind, ...] = ()
    score_dtypes: tuple[str, ...] = ()
    scale: float = 1.0
    split: bool = True
    how: algebra.Recompute = algebra.PLAIN
    accurate: bool = ACCURATE
    prescale: float = 1.0

    def _fwd(self, query, key, value, dout, peak, total, initial_term, maximum, key_index, anchor_value,
             shift, others, own, others_total, forward_total, start, end, query_ids, key_ids, score_params, *,
             interpret: bool, config: BackwardTiles, return_residuals: bool = False):
        shared = {"score_mod": self.score_mod, "score_kinds": self.score_kinds, "scale": self.scale,
                  "prescale": self.prescale, "grad_scale": self.scale * self.prescale, "how": self.how,
                  "block_q": config.block_q,
                  "block_k": config.block_k,
                  "accurate": self.accurate, "interpret": interpret}
        spans = start, end, query_ids, key_ids, score_params
        sums = (others, own, others_total, forward_total) if self.split else None
        dq_out = _dq_launch(query, key, value, dout, peak, total, initial_term, maximum, key_index,
                            anchor_value, shift, sums, *spans, warps=config.dq_warps,
                            stages=config.dq_stages, **shared)
        delta, anchor, peak, total = dq_out[1:5]
        # Each query's anchor key as the dQ launch chose it, -1 where it took D about 0.
        dkv_out = _dkv_launch(query, key, value, dout, peak, total, anchor, delta, dq_out[6], *spans,
                              warps=config.dkv_warps, stages=config.dkv_stages, **shared)
        return (*dq_out, *dkv_out), None

    def space(self, ba: tokamax.BoundArguments) -> Sequence[BackwardTiles]:
        return _dividing(BACKWARD_SPACE, ba)

    def staged_bytes(self, ba: tokamax.BoundArguments, config: BackwardTiles,
                     target: generation.GpuTarget) -> int:
        """The shared memory Triton gives a program of either launch, the larger, in the operands' dtype: a
        dQ program its query and output-gradient tiles and the key and value tiles of every stage but the
        one in use; a dK/dV program its key and value tiles and the query and output-gradient tiles of
        every stage but the one in use; and two `block_q` by `block_k` tiles on their way between
        products. fp32 operands, which Triton's products take as tf32, stage the one in use too. A bound
        fitted to the bytes Triton asks for (SMEM-a100 and SMEM-a10, 2026-10-21, 16-bit: at or above them on
        every config it refused, within the budget on all but two of the A10's compiled ones; fp32 at 64-
        and 128-wide heads on every config of the space compiled for an A100, 2026-10-28: at or above them,
        where the 16-bit count fell up to 19968 bytes short)."""
        query, value = ba.arguments["query"], ba.arguments["value"]
        size = jnp.dtype(query.dtype).itemsize
        width, value_width = query.shape[-1], value.shape[-1]
        held = 1 if size >= 4 else 0
        rows = max(config.block_q + (config.dq_stages - 1 + held) * config.block_k,
                   config.block_k + (config.dkv_stages - 1 + held) * config.block_q)
        return (rows * (width + value_width) + 2 * config.block_q * config.block_k) * size

    def registers(self, ba: tokamax.BoundArguments, config: BackwardTiles) -> int:
        """The larger of its two launches' registers a thread (`_backward_registers`)."""
        return max(_backward_registers(ba, config, dq=True), _backward_registers(ba, config, dq=False))

    def eligible(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
        return _eligible(ba, target)


def _backward_registers(ba: tokamax.BoundArguments, config: BackwardTiles, *, dq: bool) -> int:
    """The fp32 words a thread of a backward launch's program holds live across its walk, the program's
    spread over its threads: its gradient accumulators (dQ's `[block_q, D]`, or dK's and dV's) and three
    `block_q` by `block_k` tiles of the walk (the logits' weights, dP and dS)."""
    width, value_width = ba.arguments["query"].shape[-1], ba.arguments["value"].shape[-1]
    tile = 3 * config.block_q * config.block_k
    if dq:
        return (config.block_q * width + tile) // (32 * config.dq_warps)
    return (config.block_k * (width + value_width) + tile) // (32 * config.dkv_warps)


def _geometry(query, key, value, start, end, query_ids, key_ids, block_q: int, block_k: int):
    """A backward launch's sizes (rows, padded queries, query heads, widths, padded keys, a key head's
    group) and the spans' block map at its tiles."""
    batch, queries, heads, width = query.shape
    keys, kv_heads, value_width = key.shape[1], key.shape[2], value.shape[-1]
    blocks = block_map(KeySpans(start, end, query_ids, key_ids, keys), block_q, block_k)
    return batch, queries, heads, width, keys, value_width, heads // kv_heads, blocks


def _dq_launch(query, key, value, dout, peak, total, initial_term, maximum, key_index, anchor_value, shift,
               sums, start, end, query_ids, key_ids, score_params, *, score_mod,
               score_kinds: tuple[Kind, ...], scale: float, prescale: float, grad_scale: float,
               how: algebra.Recompute, block_q: int, block_k: int, warps: int, stages: int, accurate: bool,
               interpret: bool) -> tuple[jax.Array, ...]:
    """The backward's dQ launch over padded operands (`backward` pads them): a program per query block
    and query head takes each query's D about 0 or about the anchor key `key_index`'s `anchor_value · dO`,
    whichever is nearer, from the forward's split sums (`sums`, `algebra.split_delta`), or where None from
    a first walk of the key blocks the block map gives it, weighed at the forward's `maximum` from the
    initial state's maximum, denominator and its values' mean's term (`peak`, `total`, `initial_term`;
    `algebra.off_anchor_start`, `algebra.off_anchor_delta`); then walks them for dQ under the weights
    `exp(s - m)`, normalized at the end by their own sum `T`, refining `D - c` from its own sums. Returns dQ
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
                pl.BlockSpec((None, block_q, None, value_width), lambda i, b, h: (b, i, h, 0)), per_head]
    arrays = [bounds, start, end, *([query_ids, key_ids] if ids else []), query, key, value, dout, peak,
              total, initial_term, maximum, key_index, anchor_value, shift]
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
                               score_mod=score_mod, scale=scale, prescale=prescale, grad_scale=grad_scale,
                               how=how,
                               block_k=block_k,
                               split=sums is not None, accurate=accurate, interpret=interpret)
    out = pl.pallas_call(
        kernel, grid=(queries // block_q, batch, heads), in_specs=in_specs, out_specs=out_specs,
        out_shape=out_shape, interpret=interpret, name="attention_dq",
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
    )(*arrays)
    return tuple(out)


def _dkv_launch(query, key, value, dout, peak, total, anchor, delta, key_index, start, end, query_ids,
                key_ids, score_params, *, score_mod, score_kinds: tuple[Kind, ...], scale: float,
                prescale: float, grad_scale: float, how: algebra.Recompute, block_q: int, block_k: int,
                warps: int, stages: int, accurate: bool, interpret: bool) -> tuple[jax.Array, ...]:
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
                               score_mod=score_mod, scale=scale, prescale=prescale, grad_scale=grad_scale,
                               how=how,
                               block_q=block_q, accurate=accurate, interpret=interpret)
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


def _backward_op(*, score_mod, score_kinds: tuple[Kind, ...], score_dtypes: tuple[str, ...], scale: float,
                 prescale: float, split: bool, natural: bool, accurate: bool,
                 initial: bool) -> TritonBackward:
    """The Triton mainloop's two launches under one tuning decision (`TritonBackward`), recomputing the
    weights as a forward of the call's form computed them (`algebra.recompute`), from the call's own query
    taken by `prescale` (`attention.prescale`) for its logits alone."""
    how = algebra.recompute(algebra.Forward("full", 0, ids=False, scale=scale, score_mod=score_mod,
                                            natural=natural, initial=initial, prescale=prescale))
    return TritonBackward(score_mod=score_mod, score_kinds=score_kinds, scale=scale,
                          score_dtypes=score_dtypes, split=split, how=how, accurate=accurate,
                          prescale=prescale)


def _launch(*arrays, interpret: bool, **params):
    """`_backward_op`'s launches on `gradients`' arrays."""
    return _backward_op(**params)(*arrays, interpret=interpret)


def backward_unfitted(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans,
                      hooks: Hooks | None, dout: jax.Array, *, residuals: tuple, scale: float | None = None,
                      natural: bool = NATURAL, accurate: bool = ACCURATE) -> str | None:
    """Why no config of `backward`'s launches, bound as `gradients` binds this call (`launch_arguments`),
    fits the GPU (`Template.fits`), or None: the route then takes another mainloop (`attention_route`)."""
    hooks = Hooks() if hooks is None else hooks
    arrays, params, _ = launch_arguments(query, key, value, spans, hooks, dout, residuals, block=BLOCK,
                                         scale=scale)
    op = _backward_op(**params, split=True, natural=natural, accurate=accurate,
                      initial=hooks.initial is not None)
    if op.fits(op.bind(*arrays, interpret=False), generation.gpu_target()):
        return None
    return "no config of the Triton mainloop's backward fits the GPU's block and registers"


def backward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None,
             dout: jax.Array, *, residuals: tuple[jax.Array, jax.Array, jax.Array, Split] | None = None,
             split: bool = True, scale: float | None = None, natural: bool = NATURAL,
             accurate: bool = ACCURATE, interpret: bool = False, rounded: bool = True) -> Gradients:
    """The template's backward on the Triton mainloop (`gradients`, `TritonBackward`), from the forward's
    `residuals`: its maxima, denominators, anchor keys and split sums (`forward`'s, which it takes again
    where None). With `split`, D comes from the split sums; without, the dQ launch takes its own first pass
    over the keys (`algebra.off_anchor`). `natural` keeps the forward's logits natural; `accurate` takes
    the backward's accurate form (`ACCURATE`). `interpret` runs it on the CPU. A call over runs or under
    dropout is refused (`refused`)."""
    if reason := refused(spans, hooks):
        raise NotImplementedError(reason)
    if residuals is None:
        plain = Hooks() if hooks is None else Hooks(hooks.score_mod, hooks.score_params, hooks.initial)
        _, maxima, totals, anchors, sums = forward(query, key, value, spans, plain, scale=scale,
                                                   residuals=True, natural=natural, interpret=interpret)
        residuals = maxima, totals, anchors, sums
    initial = hooks is not None and hooks.initial is not None
    # A guarded forward's live rows carry normalized split sums (`algebra.split_residuals`), which the first
    # pass cannot recompute without the forward's denominator: D comes from them, in either form.
    split = split or algebra.headroom(value.dtype)
    launch = functools.partial(_launch, split=split, natural=natural, accurate=accurate, initial=initial,
                               interpret=interpret)
    return gradients(launch, query, key, value, spans, hooks, dout, residuals, block=BLOCK, scale=scale,
                     rounded=rounded)
