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
from collections.abc import Callable
from typing import Literal, overload

import jax
import jax.numpy as jnp
import tokamax
from jax.experimental import pallas as pl

from dew.nn.kernels import generation, launches, tuning
from dew.nn.kernels.selection import Refusal
from dew.nn.kernels.templates import (
    attention_algebra as algebra,
    attention_gradient as gradient,
    attention_program as program,
    axes,
    report,
    resources,
)
from dew.nn.kernels.templates.attention import (
    ACCURATE,
    DKV_GRADS,
    DQ_GRADS,
    NATURAL,
    TRACED_TILE,
    AttentionLaunch,
    Gradients,
    Hooks,
    Kept,
    KeySpans,
    Kind,
    Split,
    block_map,
    blocked,
    by_power,
    gradients,
    interpreted_dot,
    interpreted_rows,
    kept_dtype,
    kernel_dot,
    key_tiles,
    launch_arguments,
    output_split,
    padded_call,
    prescaled,
    prologues_taken,
    scored,
    within,
)
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


WARPS = 4
"""A program's warps: one warpgroup, its tile's rows over its four warps, FlashAttention 2's."""

STAGES = 2
"""A launch's pipeline stages: one tile loading beside the one consumed (`resources.Launch.floor`)."""

ROWS = 32
"""A warp's rows of the block: two mma.sync instructions' sixteen."""

BLOCK = WARPS * ROWS
"""The most rows or keys a tile takes, the block a call pads to (`padded_call`): a warp's rows a warp
(FlashAttention 2's 128 a program)."""

LEAST = 16
"""The fewest rows, and keys, a tile takes: one mma.sync instruction's sixteen, for a launch no tile of a
warp's rows holds (a scaled fp32 training walk at width 256, its small weights fp32 and their products' sums
beside its compensated sums: 272 registers a thread at 32 by 32, 136 at 16 by 32)."""

EXTENTS = tuple(LEAST << power for power in range((BLOCK // LEAST).bit_length()))[::-1]
"""The extents a tile takes along its queries or its keys, largest first: powers of two from `BLOCK` down to
`LEAST`, the extents the algebra's laws hold at (tests/test_algebra_laws.py's `ROWS` and `KEYS`)."""

WARPGROUP_ROWS = WARPS * 16
"""The rows a warpgroup's product takes at least (PTX ISA: wgmma's and tcgen05's M of 64, four warps of 16
rows), and the rows four warps' mma.sync products tile (sixteen a warp)."""

MBARRIER_BYTES = 8
"""An mbarrier object's bytes in shared memory (PTX ISA: a 64-bit opaque object)."""


@dataclasses.dataclass(frozen=True)
class _Staged:
    """How Triton stages a launch's products on a target, as its compiles allocate shared memory
    (et-tritonsweep3: 4320 launches compiled for an A100, an A10, an RTX 4090, an H100, a B200 and an RTX
    5090, the bytes each asked read beside the config it traced): a product of 16-bit operands of
    `WARPGROUP_ROWS` rows at least on warpgroup MMA where the target issues it (`warpgroup`: Hopper's wgmma,
    Blackwell's tcgen05, `tensor_memory`), each stream's tiles loaded by TMA a buffer every stage; else on
    mma.sync, loaded by cp.async a buffer every stage but the one in use; fp32 operands on mma.sync's tf32
    form (`fp32`)."""

    warpgroup: bool
    tensor_memory: bool
    fp32: bool

    @classmethod
    def of(cls, target: generation.GpuTarget, dtype, rows: int) -> "_Staged":
        issuer = (target.architecture.issuers or ("",))[0]
        fp32 = jnp.dtype(dtype).itemsize == 4
        warpgroup = not fp32 and rows >= WARPGROUP_ROWS and issuer in ("wgmma", "tcgen05")
        return cls(warpgroup, warpgroup and issuer == "tcgen05", fp32)

    def buffers(self, stages: int) -> int:
        """The tiles a stream of `stages` holds in shared memory."""
        return stages if self.warpgroup else stages - 1

    def barrier_bytes(self, products: int) -> int:
        """The bytes of the barriers a program's `products` wait on: tcgen05's two a product."""
        return 2 * MBARRIER_BYTES * products if self.tensor_memory else 0


@dataclasses.dataclass(frozen=True)
class Staging:
    """A launch's shared memory as its model counts it: its `bytes`, equal to what Triton allocates where
    `whole` is False, and where True at or above it, some tile of it counted whole that Triton's allocator
    overlaps with others by their liveness."""

    bytes: int
    whole: bool = False


def _eligible(ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
    """Whether a launch runs on `target`: a CUDA GPU, one dtype of `DTYPES` on the queries, keys and
    values, and bf16 only from `generation.BF16_GPU` on, where Triton compiles its products; spans of one
    interval a query (`refused`)."""
    query, key, value = (ba.arguments[name] for name in ("query", "key", "value"))
    capability = generation.capability(target.generation)
    return (capability is not None and query.dtype == key.dtype == value.dtype and query.dtype in DTYPES
            and (query.dtype != jnp.bfloat16 or 10 * capability[0] + capability[1] >= generation.BF16_GPU)
            and ba.arguments["start"].ndim == 2)


def refused(spans: KeySpans, hooks: Hooks | None = None) -> Refusal | None:
    """Why this mainloop takes no call over `spans` under `hooks`, or None: a call's runs, several key
    intervals a query (`KeySpans`), which this mainloop does not walk, the Mosaic and plain-JAX
    mainloops do (ledger 956, `attention_route`); probability dropout (`Hooks.drop`), which it does not
    draw, the plain-JAX mainloop does (ledger row 1521)."""
    if spans.start.ndim == 3:
        return Refusal('spans', "the Triton mainloop walks one key interval a query; a call's runs take the "
                       "Mosaic or the plain-JAX mainloop")
    if hooks is not None and hooks.drop is not None:
        return Refusal('dropout', "the Pallas-Triton mainloop takes no probability dropout (the route runs a "
                       "dropping call on the reference path, ledger row 1521)")
    if hooks is not None and hooks.rounds is not None:
        return Refusal('rounds', "the Pallas-Triton mainloop walks no maximum before its weights, so rounds "
                       "none (`Hooks.rounds`); the Mosaic and plain-JAX mainloops do")
    return None


def unfitted(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
             *, scale: float | None = None, natural: bool = NATURAL,
             residuals: bool | Kept = False) -> Refusal | None:
    """Why no config of `forward`'s launch, bound as it binds this call (`_forward_call`), fits the GPU: its
    own fitting (`Template.fits`: its padding, the configs dividing it, the registers they hold and the
    shared memory they stage), or None. The route then takes another mainloop rather than a launch that
    cannot run (`attention_route._mainloop`); the backward's own launches hold the same tiles' operands."""
    op, args, _ = _forward_call(query, key, value, spans, hooks, scale=scale, natural=natural)
    if op.fits(op.bind(*args, residuals=_kept(residuals), interpret=False), generation.gpu_target()):
        return None
    return Refusal('resources',
                   "no config of the Triton mainloop's forward fits the GPU's block and registers")


def _tile_param(ref, kind: Kind, head, queries: jax.Array, keys: pl.Slice | None):
    """A hook parameter's values for a tile, as `kind` lays them out (`Kind.axes`, its block
    `_param_spec`'s): the head's scalar, the tile's entries over its keys `keys` (a score tile's, a
    singleton along an axis it does not vary on, `scored`) or its output rows'."""
    del queries
    if kind is Kind.HEAD:
        return ref[head]
    tile = ref[axes.placed(_held(kind), {} if keys is None else {"keys": keys})]
    return tile if kind is Kind.OUTPUT else scored(tile, kind)


def _held(kind: Kind) -> tuple[str, ...]:
    """A parameter's axes a program's block holds (`_param_spec`, `_key_param_spec`): all but its batch row's
    and head's, which the block indexes away."""
    return tuple(axis for axis in kind.axes if axis not in ("batch", "heads"))


def _forward_kernel(*refs, spec: algebra.Forward, score_kinds: tuple[Kind, ...],
                    output_kinds: tuple[Kind, ...], initial: bool, residuals: Kept, output_mod, block_k: int,
                    interpret: bool, maxima: bool):
    """The forward's Triton mainloop: one program a query block and head walks the key tiles its form
    gives it (every tile, the causal diagonal's, or the spans' block map), issuing the tile's two
    products and calling the algebra between them (`attention_algebra`), its rows' sums scaled where its
    spec is (`algebra.SCALED_WALK`, the scaled op's, `TritonForward.scaled`); with `maxima` it also writes
    each row's running maximum, which the fold's guard over whole launches reads
    (`algebra.walked_launches`)."""
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
    out_ref, *residual_refs = refs
    # A plain launch's rows' running maxima follow its outputs.
    maxima_ref = residual_refs.pop() if maxima else None
    block_q = query_ref.shape[0]
    head = pl.program_id(2)
    first_query = pl.program_id(0) * block_q
    lo, hi = bounds if bounds is not None else key_tiles(spec.form, first_query, block_q, block_k, spec.keys)
    query = query_ref[...]
    q_pos = first_query + jnp.arange(block_q)[:, None]

    def keys_at(form: algebra.Forward, index) -> algebra.Keys:
        # A key tile's logits in `form` (the call's, or the fold's rewalk's); its values load where the step
        # calls for them, after the softmax, where jax's mha loads them.
        tile = pl.ds(index * block_k, block_k)
        coords = algebra.Coords(q_pos, index * block_k + jnp.arange(block_k)[None, :],
                                None if spans is None else (spans,),
                                None if ids is None else (ids[0], ids[1][tile]))
        params = tuple(_tile_param(ref, kind, head, q_pos, tile)
                       for ref, kind in zip(score_refs, score_kinds, strict=True))
        logits = algebra.modify(form, kernel_dot(query, key_ref[tile, :].T, interpret), coords, params)
        values = algebra.Operand((block_k, value_ref.shape[-1]), value_ref.dtype, lambda: value_ref[tile, :])
        return algebra.Keys(coords, logits, values, functools.partial(kernel_dot, interpret=interpret))

    def rows(keys):
        # Each row's value row at its key (the anchor's).
        return value_ref[keys, :]

    resumed = (initial_refs[0][...], initial_refs[1][...], initial_refs[2][...]) if initial else None
    training = residuals is Kept.SPLIT

    def walk(form: algebra.Forward):
        # The walk in `form`: the call's, or the fold's rewalk's.
        return program.walk(form, lo, hi, functools.partial(keys_at, form), rows, resumed, block_q,
                            value_ref.shape[-1], training=training)

    output_params = tuple(_tile_param(ref, kind, head, q_pos, None)
                          for ref, kind in zip(output_refs, output_kinds, strict=True))

    def finished(form: algebra.Forward, carry):
        # A walk's outputs in `form` (`program.finished`): the normalized output, its maximum and
        # denominator, and a training walk's (`Kept.SPLIT`) anchors and split sums.
        return program.finished(form, carry, rows=rows, resumed=resumed, training=training,
                                output_mod=output_mod, output_params=output_params)

    walked = walk(spec)
    if maxima_ref is not None:
        maxima_ref[...] = (walked if isinstance(walked, algebra.Partial) else walked[0]).m
    out, m, denominator, *kept = finished(spec, walked)
    out_ref[...] = out.astype(out_ref.dtype)
    if residuals is not Kept.NONE:
        residual_refs[0][...] = m
        residual_refs[1][...] = denominator
    for ref, part in zip(residual_refs[2:7], kept, strict=True):
        ref[...] = part


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
    scaled: bool = False
    """Whether its kernel walks scaled (`algebra.SCALED_WALK`): the op a call some row of which can be scaled
    launches in its common op's place (`_fwd`, `algebra.may_scale`), its own tiles derived from its own
    registers and shared memory (`registers`, `staged_bytes`)."""

    def _fwd(self, query, key, value, start, end, query_ids, key_ids, score_params, output_params, m0, total0,
             acc0, *, residuals: bool | Kept, interpret: bool, config: Tiles, return_residuals: bool = False):
        residuals = _kept(residuals)
        arguments = (query, key, value, start, end, query_ids, key_ids, score_params, output_params, m0,
                     total0, acc0)
        walk = functools.partial(self._walk, *arguments, residuals=residuals, interpret=interpret,
                                 config=config)
        if self.scaled or not algebra.headroom(value.dtype):
            return walk(scaled=self.scaled), None
        # A call some row of which can be scaled (`algebra.may_scale`) launches the scaled op's kernel (its
        # own tiles, `scaled`), every other call, every ordinary input's, this op's unscaled one, bit for bit
        # the scaled walk's there.
        scaled_op = dataclasses.replace(self, scaled=True)

        def rare():
            # A pin of this op's where the scaled op can take it, else its own tiles
            # (`tuning.pinned_where_taken`): its registers and shared memory are not the common op's.
            with tuning.pinned_where_taken(scaled_op, scaled_op.bind(*arguments, residuals=residuals,
                                                                     interpret=interpret)) as taken:
                return taken(*arguments, residuals=residuals, interpret=interpret)

        return jax.lax.cond(algebra.may_scale(value, keys_axis=1, initial=acc0), rare,
                            functools.partial(walk, scaled=False)), None

    def _walk(self, query, key, value, start, end, query_ids, key_ids, score_params, output_params, m0,
              total0, acc0, *, residuals: Kept, interpret: bool, config: Tiles, scaled: bool):
        """The call's launch at `config`, its walk `scaled` where its values can pass fp32's largest."""
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
        per_row = pl.BlockSpec((None, None, block_q), lambda i, b, h: (b, h, i))
        if residuals is not Kept.NONE:
            out_shape += [jax.ShapeDtypeStruct((batch, heads, queries), jnp.float32)] * 2
            out_specs += [per_row, per_row]
        if residuals is Kept.SPLIT:
            out_shape += [jax.ShapeDtypeStruct((batch, heads, queries), jnp.int32),
                          jax.ShapeDtypeStruct((batch, heads, queries, value_width), jnp.float32),
                          *[jax.ShapeDtypeStruct((batch, heads, queries), jnp.float32)] * 3]
            per_value = pl.BlockSpec((None, None, block_q, value_width), lambda i, b, h: (b, h, i, 0))
            out_specs += [per_row, per_value, per_row, per_row, per_row]
        # A call's own key count, none included; an op built without one counts its launch's keys. A scaled
        # walk scales its rows' sums where they can pass fp32's largest (`algebra.Forward.headroom`,
        # `algebra.SCALED_WALK`).
        headroom = scaled and algebra.headroom(value.dtype)
        spec = algebra.Forward(self.form, keys if self.keys is None else self.keys, ids, scale,
                               self.score_mod, self.natural, headroom, initial,
                               compensated=algebra.compensates(value.dtype), prescale=power)
        kernel = functools.partial(
            _forward_kernel, score_kinds=self.score_kinds, output_kinds=self.output_kinds, initial=initial,
            residuals=residuals, output_mod=self.output_mod, block_k=block_k, interpret=interpret)
        launch = functools.partial(
            launches.pallas_call, grid=(queries // block_q, batch, heads), interpret=interpret,
            compiler_params=plgpu.CompilerParams(num_warps=config.warps, num_stages=config.stages))
        rowed = jax.ShapeDtypeStruct((batch, heads, queries), jnp.float32)
        # A scaled walk's launch named apart: a call compiles it beside its unscaled form (`scaled`).
        name = report.named("attention_forward_scaled" if spec.headroom else "attention_forward")

        def launched(form: algebra.Forward, live: jax.Array | None, into=None):
            # The call in `form` (`algebra.walked_launches`' launch): its outputs and its rows' running
            # maxima, which the fold's guard reads.
            del live, into
            *out, maxima = launch(functools.partial(kernel, spec=form, maxima=True), in_specs=in_specs,
                                  out_specs=[*out_specs, per_row], out_shape=[*out_shape, rowed],
                                  name=name)(*arrays)
            return tuple(out), maxima, None

        if spec.reaches:
            out = algebra.walked_launches(spec, launched)
        else:
            out = launch(functools.partial(kernel, spec=spec, maxima=False), in_specs=in_specs,
                         out_specs=out_specs, out_shape=out_shape, name=name)(*arrays)
        return tuple(out) if residuals is not Kept.NONE else out[0]

    GENERATES = True

    def derived(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> Tiles | None:
        """Its tile from the facts (`_derived`): its queries held, its keys walked."""
        return _derived(self, ba, target, lambda rows, walk: Tiles(rows, walk, WARPS, STAGES),
                        held=ba.arguments["query"].shape[1], walked=ba.arguments["key"].shape[1])

    def staged_bytes(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget) -> int:
        return self.staging(ba, config, target).bytes

    def staging(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget) -> Staging:
        """The shared memory Triton gives a program, as its compiles allocate it (`_Staged`): the key and
        value tiles a stream buffers (an fp32 call's keys every stage, its tf32 products reading them where
        they land), the query tile, and the weights' tile where it reaches the second product through shared
        memory: an fp32 call's, and on mma.sync where Triton gives the two chained products their warps
        along the values (the values' product's, converted first, along its longer side: fewer query rows
        than value columns), the weights otherwise staying in registers; tcgen05's barriers. A training
        walk converts its anchor's value row `[bq, Dv]` through shared memory, Triton's tile of it at most
        four warps' rows (warpgroup MMA's at most 64 columns; tcgen05 stages a query tile a buffered stage
        and a row's word). Each equal to the bytes Triton asked on every launch compiled but one form: a
        training walk on mma.sync with fewer query rows than value columns, whose row's tile Triton
        allocates in part, counted whole. Counted whole too, where Triton's allocator overlaps them with
        other buffers by their liveness (et-tritonsweep3: at or above every compile): a scaled walk's small
        weights' parts and its weights at the scale fp32, a per-key and a per-tile score parameter's tile
        on its way to the logits' layout. tcgen05 stages a head's and a query's parameters, a word a row."""
        query, value = ba.arguments["query"], ba.arguments["value"]
        size = jnp.dtype(query.dtype).itemsize
        width, value_width = query.shape[-1], value.shape[-1]
        rows, keys, stages = config.block_q, config.block_k, config.stages
        staged = _Staged.of(target, query.dtype, rows)
        held = staged.buffers(stages)
        elements = (stages if staged.fp32 else held) * keys * width + held * keys * value_width + rows * width
        if staged.fp32 or (not staged.warpgroup and rows < value_width):
            elements += rows * keys
        extra, whole = staged.barrier_bytes(2), False
        if _split(ba) and not staged.fp32:
            if staged.tensor_memory:
                elements += (stages - 1) * rows * width
                extra += rows * 4
            elif staged.warpgroup:
                elements += WARPGROUP_ROWS * min(value_width, WARPGROUP_ROWS)
            else:
                elements += min(rows, WARPGROUP_ROWS) * value_width
                whole = rows < value_width
        if self._scaled(ba):
            extra += (_small_parts(value.dtype) * size + 4) * rows * keys
            whole = True
        for kind in self.score_kinds:
            extra += _param_staged(kind, rows, keys, stages, tensor_memory=staged.tensor_memory)
            whole |= kind in (Kind.KEY, Kind.TILE)
        return Staging(elements * size + extra, whole)

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
        if _split(ba):
            words += rows * value.shape[-1]
        if self._scaled(ba):
            # A scaled walk's small weights fp32 and their products' sums (`algebra.at_scale`).
            words += rows * config.block_k + rows * value.shape[-1]
        return words // (32 * config.warps)

    def _scaled(self, ba: tokamax.BoundArguments) -> bool:
        """Whether its kernel walks scaled (`scaled`, values that can pass fp32's largest), which its
        resources price."""
        return self.scaled and algebra.headroom(ba.arguments["value"].dtype)

    def eligible(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
        return _eligible(ba, target)


def _param_staged(kind: Kind, rows: int, keys: int, stages: int, *, tensor_memory: bool,
                  summed: bool = False) -> int:
    """The shared memory a score parameter of `kind` takes in a program of `rows` by `keys` streamed at
    `stages`, its fp32 words (`TritonForward.staging`): a per-key one its keys' vector and a per-tile one its
    tile, each on its way to the logits' layout, counted whole; where its gradient is `summed` in the
    program (dK/dV's), a key's its tile and a tile's two, its gradient's beside it. On tcgen05
    (`tensor_memory`) a head's word and a query's rows' each buffered stage (a dK/dV program streams its
    queries' with its rows, `_dkv_staging`)."""
    if kind is Kind.KEY:
        return 4 * (rows * keys if summed else keys)
    if kind is Kind.TILE:
        return 4 * rows * keys * (2 if summed else 1)
    if tensor_memory and kind is Kind.HEAD:
        return 4 * (stages - 1)
    if tensor_memory and kind is Kind.QUERY and not summed:
        return 4 * rows * (stages - 1)
    return 0


def _small_parts(dtype) -> int:
    """The parts a scaled walk's small weights take for their product with values of `dtype`, as the algebra
    forms them (`algebra.small_parts`, traced for its count)."""
    held = jax.eval_shape(lambda small: algebra.small_parts(small, dtype),
                          jax.ShapeDtypeStruct((8, 8), jnp.float32))
    return len(held)


def _kept(residuals: bool | Kept | None) -> Kept:
    """What a forward keeps (`Kept`), a bound `residuals` as `forward` takes it: True the training walk's
    split sums, False or None nothing."""
    if isinstance(residuals, Kept):
        return residuals
    return Kept.SPLIT if residuals else Kept.NONE


def _split(ba: tokamax.BoundArguments) -> bool:
    """Whether a bound forward walks the training walk's split sums (`Kept.SPLIT`), which its resources
    price."""
    return _kept(ba.arguments.get("residuals")) is Kept.SPLIT


def _derived[T: (Tiles, BackwardTiles)](launch: Template[T], ba: tokamax.BoundArguments,
                                       target: generation.GpuTarget, tiles: Callable[[int, int], T], *,
                                       held: int, walked: int) -> T | None:
    """A launch's tile from the facts (design §7, as the Mosaic mainloop's `attention_mosaic._derived`), no
    space ranked: the tile a program holds whole (a forward's queries, a backward's keys, the dK/dV launch's
    accumulated rows) `BLOCK`, four warps of `ROWS` each, FlashAttention 2's 128 a program, and
    the tile it walks the largest of `EXTENTS` whose program holds (`legal`: the block's shared memory
    and a thread's registers), stepping down the held tile only where no walked one does; each dividing the
    call's padded extent it lies along (`held`, `walked`; a longer tile holds padding alone). `tiles(held,
    walked)` builds the config. At width 128, bf16: a forward takes 128 by 64, the order measured within 6%
    of the fastest on an A100 and an H100 (ATT-ZERO3-a100, ATT-ZERO4-h100), a training walk 64 by 64, the
    tile measured fastest of it where its anchor row puts 128 rows past a thread's registers
    (TRAINTILES3-a100), and a backward 64 by 64, the measured order's first (DPROF3-a100, DPROF3-h100).
    Where its products issue on the FMA pipes rather than the tensor cores (an fp32 call's IEEE products,
    `kernel_dot`), the largest such tile a multiprocessor holds two programs of (`_resident`) first, eight
    warps to hide each other's latency: one program a multiprocessor left its warps without an eligible one
    92 to 94% of cycles (an H100's fp32 training step at 1 x 1024, 147,456 bytes a program at 64 by 64,
    M-ET-55), and two took that step from 26.1 to 9.5 ms on an H100 and 28.7 to 22.8 on an A100, every
    launch faster (M-ET-57). A tensor-core launch keeps its largest tile: a bf16 backward at 32 by 64, two
    programs, took 2.34 ms a launch on an H100 where 64 by 64, one, took 1.30 to 1.58 (M-ET-57). None where
    no tile holds."""
    legal = [config for rows in (extent for extent in EXTENTS if held % extent == 0)
             for config in (tiles(rows, walk) for walk in EXTENTS if walked % walk == 0)
             if launch.legal(ba, config, target)]
    if _Staged.of(target, ba.arguments["query"].dtype, held).fp32:
        return next((config for config in legal if _resident(launch, ba, config, target) >= 2),
                    next(iter(legal), None))
    return next(iter(legal), None)


def _resident[T: (Tiles, BackwardTiles)](launch: Template[T], ba: tokamax.BoundArguments, config: T,
                                        target: generation.GpuTarget) -> int:
    """The programs of a launch at `config` a multiprocessor holds at once: its threads' registers
    (`registers`, at the allocator's step), `WARPS` warps of them, within its register file, as many as it
    holds blocks, and its shared memory (`staged_bytes`) within the multiprocessor's, each program's
    reservation its own (`resources.shared_programs`)."""
    held = target.multiprocessor
    step = held.register_step
    allocated = -(-(launch.registers(ba, config) or step) // step) * step * WARPS * held.warp_threads
    most = min(held.registers // allocated, held.resident_blocks)
    return resources.shared_programs(most, launch.staged_bytes(ba, config, target),
                                     generation.shared_budget(target))


def _param_spec(kind: Kind, param: jax.Array, block_q: int, keys: int) -> pl.BlockSpec:
    """How a program reads a hook parameter of `kind` (`Kind.axes`): a head's scalar from the whole `[H]`,
    else its batch row's and head's block of its queries' block, every key and the whole width."""
    if kind is Kind.HEAD:
        return pl.BlockSpec(param.shape, lambda i, b, h: (0,))
    extents = {"batch": None, "heads": None, "queries": block_q, "keys": keys, "width": param.shape[-1]}
    return pl.BlockSpec(axes.named(kind.axes, extents),
                        lambda i, b, h: axes.placed(kind.axes, {"batch": b, "heads": h, "queries": i}, 0))


@overload
def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, residuals: Literal[False] = False, natural: bool = NATURAL,
            interpret: bool = False) -> jax.Array:
    ...


@overload
def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, residuals: Literal[True] | Kept, natural: bool = NATURAL,
            interpret: bool = False) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, Split]:
    ...


def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, residuals: bool | Kept = False, natural: bool = NATURAL,
            interpret: bool = False):
    """The template's forward (`TritonForward`) on `query` `[B, Q, H, D]` over `key` and `value`
    `[B, K, Hk, D]` and `[B, K, Hk, Dv]` and the keys `spans` leaves each query: as
    `dew.nn.kernels.templates.attention.reference` computes it, with `residuals` its running maxima and
    denominators `[B, H, Q]` too, and each query's anchor key `[B, H, Q]` int32 (`backward` reads them
    all): `True` or `Kept.SPLIT` the training walk's anchor and split sums, `Kept.OUTPUT` the plain walk's
    output in their place (`output_split`, `takes_output_residuals`: no anchor key, as the Mosaic training
    forward keeps where the call takes it). The operands pad to whole blocks of the largest tile and to
    power-of-two head widths (`padded_call`). `natural` keeps the logits natural (`algebra.Forward.base2`).
    `interpret` runs it on the CPU. A call over runs or under dropout is refused (`refused`)."""
    if reason := refused(spans, hooks):
        raise NotImplementedError(reason)
    # The prologues taken as XLA's program before the launch (`prologues_taken`).
    query, key, hooks = prologues_taken(query, key, hooks)
    op, args, queries = _forward_call(query, key, value, spans, hooks, scale=scale, natural=natural)
    value_width = value.shape[-1]
    kept = _kept(residuals)
    out = op(*args, residuals=kept, interpret=interpret)
    if kept is Kept.NONE:
        return out[:, :queries, :, :value_width]
    if kept is Kept.OUTPUT:
        out, peak, total = out
        out, peak, total = out[:, :queries, :, :value_width], peak[:, :, :queries], total[:, :, :queries]
        return out, peak, total, jnp.full(peak.shape, -1, jnp.int32), output_split(out)
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
        m0, total0 = call.rows(initial.m, algebra.LEAST), call.rows(initial.total)
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
    """A hook parameter's values for a dK/dV program's tile, its queries `rows` of the whole row
    (`_key_param_spec`'s block): the head's scalar, else the tile's entries, a singleton along an axis it
    does not vary on (`scored`)."""
    if kind is Kind.OUTPUT:
        raise ValueError("an output hook's parameter has no score tile")
    if kind is Kind.HEAD:
        return ref[head]
    return scored(ref[axes.placed(_held(kind), {"queries": rows})], kind)


def _issued(a: jax.Array, b: jax.Array, *, interpret: bool) -> jax.Array:
    """A product the backward's derived forms issue (`Derived.step`'s `product`) as a tensor-core dot takes
    it, `a @ b` summed in fp32: a float32 left operand (the forms' weights and cotangents) against a 16-bit
    one narrowed by its parts, each scaled into the other's range and taken back after its product
    (`algebra.cotangent_product`: a cotangent past a 16-bit dtype's largest stays finite), as the hand steps
    narrowed theirs; the accurate form issues its parts, each of the other's dtype already."""
    def dot(x: jax.Array, y: jax.Array) -> jax.Array:
        if not interpret:
            return kernel_dot(x, y, interpret=False)
        # The CPU runs no 16-bit product into fp32 of a transposed tile (XLA's DotThunk, as the forms' rows
        # walk forms them): its operands taken to fp32, where 16-bit products are exact, summed in fp32.
        return interpreted_dot(x.astype(jnp.float32), y.astype(jnp.float32))
    if jnp.dtype(a.dtype).itemsize > jnp.dtype(b.dtype).itemsize:
        return algebra.cotangent_product(a, b, dot, accurate=False)
    return dot(a, b.astype(a.dtype))


def _dq_kernel(bounds_ref, start_ref, end_ref, *refs, ids: bool, score_kinds: tuple[Kind, ...], score_mod,
               scale: float, prescale: float, grad_scale: float, how: algebra.Recompute, block_k: int,
               split: bool, accurate: bool, interpret: bool):
    refs = list(refs)
    id_refs = [refs.pop(0), refs.pop(0)] if ids else []
    (query_ref, key_ref, value_ref, dout_ref, peak_ref, total_ref, initial_ref, maximum_ref, index_ref,
     anchor_value_ref, shift_ref) = (refs.pop(0) for _ in range(11))
    # The forward's split sums and denominator, where it gives them (`gradient.split_delta`).
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
    product = functools.partial(_issued, interpret=interpret)
    # D's and the anchor's products with dO, summed as `product` sums dP (`gradient.off_anchor_delta`).
    row_products = interpreted_rows if interpret else gradient.summed_rows
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

    def formed(index):
        # Every weight against the forward's maximum m, so no running maximum rescales a sum
        # (`gradient.OffAnchor`); the weights are this launch's own, exp(s - m) over its logits, so they
        # sum to one over them whatever logits the forward took.
        keys, k_pos, seen, params = tile(index)
        logits, _ = gradient.tile_logits(score_mod, kernel_dot(query, key_ref[keys, :].T, interpret), seen,
                                        q_pos, k_pos, params, scale=scale if how.factor is None else 1.0)
        return k_pos, logits, value_ref[keys, :]

    # From the softmax's initial state (`gradients`): its maximum, denominator and share of D's numerator.
    key_index = index_ref[...]
    peak0, shares = peak, (total, initial_ref[...])
    shift = shift_ref[...]
    before, state_total = gradient.off_anchor_start(peak, total, maximum, value_ref.shape[-1], shift)
    if split:
        # The forward took the first pass (`algebra.training_step`).
        others_ref, own_ref, others_total_ref, forward_total_ref = split_refs
        prior = forward_total_ref[...]
        anchor, delta, key_index = gradient.split_delta(
            others_ref[...], own_ref[...], others_total_ref[...], prior, state_total, anchor_value_ref[...],
            dout, key_index, shift, rows=row_products)
    else:
        state = program.off_anchor_pass(bounds_ref[0], bounds_ref[1], formed, before, maximum, key_index,
                                        shift, how, product)
        prior = algebra.power2(state.total, shift)
        anchor, delta, key_index = gradient.off_anchor_delta(state, anchor_value_ref[...], dout, state_total,
                                                            shares[1], key_index, shift,
                                                            rows=row_products)
    peak = maximum
    index_out_ref[...] = key_index
    summed = tuple(kind in DQ_GRADS for kind in score_kinds)

    def weighed(index):
        # A key tile's operands as the derived dQ walk reads them (`attention_program.dq_sums`).
        keys, k_pos, seen, params = tile(index)
        key = key_ref[keys, :]
        logits, slopes = gradient.tile_logits(score_mod, kernel_dot(query, key.T, interpret), seen, q_pos,
                                             k_pos, params, scale=scale if how.factor is None else 1.0,
                                             slopes=True)
        return k_pos, seen, key, value_ref[keys, :], logits, slopes

    # Each cotangent over the prior denominator, taken to the pass's own at the end (`gradient.dq_finish`),
    # at the call's whole scale, its logits at the scale its query's power left (`prescale`).
    sums = program.dq_sums(weighed, query, dout, peak, anchor, delta, prior, bounds_ref[0], bounds_ref[1],
                           key_index, None, hooked=score_mod is not None, score_kinds=score_kinds,
                           summed=summed, prescale=prescale, grad_scale=grad_scale, how=how,
                           accurate=accurate, dot=product)
    dq, delta, total, mass, grads = gradient.dq_finish(sums, prior, state_total, peak0, shares, peak, anchor,
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

    def formed(index):
        # A query tile's operands as the derived dK/dV walk reads them (`attention_program.dkv_sums`).
        rows = pl.ds(index * block_q, block_q)
        query, dout = query_ref[rows, :], dout_ref[rows, :]
        q_pos = index * block_q + jnp.arange(block_q)[:, None]
        seen = within(k_pos, start_ref[rows][:, None], end_ref[rows][:, None],
                      (id_refs[0][rows][:, None], key_ids[0][None, :]) if ids else None)
        params = [_key_param(ref, kind, head, rows)
                  for ref, kind in zip(score_refs, score_kinds, strict=True)]
        # The logits from the queries taken by the power; dK at the call's whole scale against the call's
        # own queries (`attention_program.dkv_sums`'s `grad_scale`).
        products = kernel_dot(by_power(query, prescale), key.T, interpret)
        logits, slopes = gradient.tile_logits(score_mod, products, seen, q_pos, k_pos, params,
                                             scale=scale if how.factor is None else 1.0, slopes=True)
        tiles = {"logits": logits, "q": query, "dout": dout, "peak": peak_ref[rows], "total": total_ref[rows],
                 "anchor": anchor_ref[rows], "delta": delta_ref[rows], "seen": seen,
                 "at_anchor": k_pos == index_ref[rows][:, None]}
        if slopes is not None:
            tiles["slope"] = slopes[0]
        return tiles, slopes, seen

    def pulled(index, grads, pieces):
        # A query tile's hook pieces (`gradient.pulled`): a tile parameter's entries written at its rows,
        # a key parameter's summed.
        rows = pl.ds(index * block_q, block_q)
        for place, (kind, piece) in enumerate(zip(score_kinds, pieces, strict=True)):
            if kind == Kind.TILE:
                grad_refs[place][rows, :] = piece
        return tuple(grad + piece if kind == Kind.KEY else grad
                     for grad, kind, piece in zip(grads, score_kinds, pieces, strict=True))

    sums = tuple(jnp.zeros((block_k,), jnp.float32) if kind == Kind.KEY else jnp.zeros((), jnp.float32)
                 for kind in score_kinds)
    dk, dv, grads = program.dkv_sums(formed, key, value, ranges_ref[0], ranges_ref[1], pulled, sums,
                                     hooked=score_mod is not None, dropped=False, grad_scale=grad_scale,
                                     how=how, accurate=accurate, score_kinds=score_kinds,
                                     dot=functools.partial(_issued, interpret=interpret))
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

    GENERATES = True

    def derived(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> BackwardTiles | None:
        """Its tiles from the facts (`_derived`): the dK/dV launch's keys held, the queries both launches
        walk, a program of either launch holding them."""
        return _derived(self, ba, target,
                        lambda rows, walk: BackwardTiles(walk, rows, WARPS, STAGES, WARPS, STAGES),
                        held=ba.arguments["key"].shape[1], walked=ba.arguments["query"].shape[1])

    def staged_bytes(self, ba: tokamax.BoundArguments, config: BackwardTiles,
                     target: generation.GpuTarget) -> int:
        """The shared memory Triton gives a program of either launch, the larger, as its compiles allocate
        it (`_dq_staging`, `_dkv_staging`)."""
        return max(_dq_staging(ba, config, target, self.score_kinds).bytes,
                   _dkv_staging(ba, config, target, self.score_kinds).bytes)

    def registers(self, ba: tokamax.BoundArguments, config: BackwardTiles) -> int:
        """The larger of its two launches' registers a thread (`_backward_registers`)."""
        return max(_backward_registers(ba, config, dq=True), _backward_registers(ba, config, dq=False))

    def eligible(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
        return _eligible(ba, target)


def _dq_staging(ba: tokamax.BoundArguments, config: BackwardTiles, target: generation.GpuTarget,
                score_kinds: tuple[Kind, ...]) -> Staging:
    """The shared memory Triton gives a dQ program (`_Staged`, `Staging`): the key and value tiles a stream
    buffers, the query and output-gradient tiles, and the score cotangents' tile where it reaches dQ's
    product through shared memory (an fp32 call's; on mma.sync where the chained products' warps run along
    the values, as the forward's weights do, `TritonForward.staging`); tcgen05's barriers. Each equal to the
    bytes
    Triton asked on every bf16 and fp32 launch compiled on mma.sync and wgmma (et-tritonsweep3). Counted
    whole where Triton's allocator overlaps them by liveness (at or above every compile): an fp16 call's
    cotangents' tile, which Triton stages in part; tcgen05's key tile and cotangents' tile beside its
    streams; a per-key and per-tile score parameter's."""
    query, value = ba.arguments["query"], ba.arguments["value"]
    size = jnp.dtype(query.dtype).itemsize
    width, value_width = query.shape[-1], value.shape[-1]
    rows, keys, stages = config.block_q, config.block_k, config.dq_stages
    staged = _Staged.of(target, query.dtype, rows)
    elements = (staged.buffers(stages) * keys * (width + value_width) + rows * (width + value_width))
    if staged.fp32 or (not staged.warpgroup and rows < value_width):
        elements += rows * keys
    if staged.fp32 and blocked(width) and blocked(keys):
        # The key tile staged again for dQ's product, its blocks along the keys where the logits' product took
        # them along the width (`kernel_dot`): Triton asked 8192 bytes more at 32 x 32, width 64, and 32768 at
        # 64 x 64, width 128, on an A100, an H100 and a B200 (et-tsmem).
        elements += keys * width
    fp16 = jnp.dtype(query.dtype) == jnp.dtype(jnp.float16)
    if fp16:
        elements += rows * keys
    if staged.tensor_memory:
        elements += keys * width + rows * keys
    extra = staged.barrier_bytes(3) + sum(
        _param_staged(kind, rows, keys, stages, tensor_memory=staged.tensor_memory) for kind in score_kinds)
    whole = fp16 or staged.tensor_memory or any(kind in (Kind.KEY, Kind.TILE) for kind in score_kinds)
    return Staging(elements * size + extra, whole)


def _dkv_staging(ba: tokamax.BoundArguments, config: BackwardTiles, target: generation.GpuTarget,
                 score_kinds: tuple[Kind, ...]) -> Staging:
    """The shared memory Triton gives a dK/dV program (`_Staged`, its products' rows its keys): the query and
    output-gradient tiles a stream buffers, the key and value tiles, the weights' tile on its way to dV's
    product, and its query rows' words a buffered stage (`_dkv_launch`'s per-query operands: the spans'
    start and end, the forward's five rows, a query's ids and per-query score parameters, a word each).
    Equal to the bytes Triton asked on every bf16 launch compiled on mma.sync before Hopper and on a
    B200's (et-tritonsweep3). An fp16 or fp32 call's, Hopper's and warpgroup MMA's stage more tiles of
    the walk, which Triton's allocator overlaps by their liveness (fp16's on identical shapes up to 1.7
    times bf16's): counted whole, the walk's widest tile, its query rows by its keys or its widths, fp32
    (an fp32 call's within 2% of every compile, at or above it; an fp16 call's four score tiles and two of
    it); a per-key and per-tile score parameter's with its gradient's."""
    query, value = ba.arguments["query"], ba.arguments["value"]
    size = jnp.dtype(query.dtype).itemsize
    width, value_width = query.shape[-1], value.shape[-1]
    rows, keys, stages = config.block_q, config.block_k, config.dkv_stages
    staged = _Staged.of(target, query.dtype, keys)
    # The spans' start and end, the forward's peak, total, anchor, delta and key index, a query's ids and
    # each per-query score parameter.
    words = 7 + (ba.arguments.get("query_ids") is not None) + sum(kind is Kind.QUERY for kind in score_kinds)
    streamed = staged.buffers(stages) * rows * (width + value_width)
    elements = streamed + keys * (width + value_width) + rows * keys
    extra = 4 * words * (stages - 1) * rows
    issuers = target.architecture.issuers
    whole = jnp.dtype(query.dtype) != jnp.dtype(jnp.bfloat16) or staged.warpgroup or "wgmma" in issuers
    if whole:
        # The walk's widest tile between its products, its query rows by its keys or its widths, fp32 (an
        # fp16 call's four score tiles and two of it), and a query row's word a buffered stage.
        widest = rows * max(width, value_width, keys)
        fp16 = jnp.dtype(query.dtype) == jnp.dtype(jnp.float16)
        extra += 4 * ((4 * rows * keys + 2 * widest) if fp16 else widest) + 4 * 2 * (stages - 1) * rows
    # A head's parameter staged where a product of its query rows issues on tcgen05.
    queried = _Staged.of(target, query.dtype, rows).tensor_memory
    extra += sum(_param_staged(kind, rows, keys, stages, tensor_memory=queried, summed=True)
                 for kind in score_kinds)
    whole |= any(kind in (Kind.KEY, Kind.TILE) for kind in score_kinds)
    return Staging(elements * size + extra, whole)


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
    whichever is nearer, from the forward's split sums (`sums`, `gradient.split_delta`), or where None from
    a first walk of the key blocks the block map gives it, weighed at the forward's `maximum` from the
    initial state's maximum, denominator and its values' mean's term (`peak`, `total`, `initial_term`;
    `gradient.off_anchor_start`, `gradient.off_anchor_delta`); then walks them for dQ under the weights
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
    out = launches.pallas_call(
        kernel, grid=(queries // block_q, batch, heads), in_specs=in_specs, out_specs=out_specs,
        out_shape=out_shape, interpret=interpret, name=report.named("attention_dq"),
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
    out = launches.pallas_call(
        kernel, grid=(keys // block_k, batch, heads), in_specs=in_specs, out_specs=out_specs,
        out_shape=out_shape, input_output_aliases=aliases, interpret=interpret,
        name=report.named("attention_dkv"),
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
    )(*arrays)
    return tuple(out)


def _key_param_spec(kind: Kind, param: jax.Array, queries: int, block_k: int) -> pl.BlockSpec:
    """How a dK/dV program reads a hook parameter of `kind` (`Kind.axes`): a head's scalar from the whole
    `[H]`, else its batch row's and head's block of every query and its key block."""
    if kind is Kind.OUTPUT:
        raise ValueError("an output hook's parameter has no score tile")
    if kind is Kind.HEAD:
        return pl.BlockSpec(param.shape, lambda j, b, h: (0,))
    extents = {"batch": None, "heads": None, "queries": queries, "keys": block_k}
    return pl.BlockSpec(axes.named(kind.axes, extents),
                        lambda j, b, h: axes.placed(kind.axes, {"batch": b, "heads": h, "keys": j}, 0))


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
                      natural: bool = NATURAL, accurate: bool = ACCURATE) -> Refusal | None:
    """Why no config of `backward`'s launches, bound as `gradients` binds this call (`launch_arguments`),
    fits the GPU (`Template.fits`), or None: the route then takes another mainloop (`attention_route`)."""
    hooks = Hooks() if hooks is None else hooks
    arrays, params, _ = launch_arguments(query, key, value, spans, hooks, dout, residuals, block=BLOCK,
                                         scale=scale)
    op = _backward_op(**params, split=True, natural=natural, accurate=accurate,
                      initial=hooks.initial is not None)
    if op.fits(op.bind(*arrays, interpret=False), generation.gpu_target()):
        return None
    return Refusal('resources',
                   "no config of the Triton mainloop's backward fits the GPU's block and registers")


def backward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None,
             dout: jax.Array, *, residuals: tuple[jax.Array, jax.Array, jax.Array, Split] | None = None,
             split: bool = True, scale: float | None = None, natural: bool = NATURAL,
             accurate: bool = ACCURATE, interpret: bool = False, rounded: bool = True) -> Gradients:
    """The template's backward on the Triton mainloop (`gradients`, `TritonBackward`), from the forward's
    `residuals`: its maxima, denominators, anchor keys and split sums (`forward`'s, which it takes again
    where None). With `split`, D comes from the split sums; without, the dQ launch takes its own first pass
    over the keys (`gradient.off_anchor`). `natural` keeps the forward's logits natural; `accurate` takes
    the backward's accurate form (`ACCURATE`). `interpret` runs it on the CPU. A call over runs or under
    dropout is refused (`refused`). Its query's and key's gradients are under their prologues
    (`attention_route._pulled` takes them back)."""
    if reason := refused(spans, hooks):
        raise NotImplementedError(reason)
    # The prologues taken as XLA's program before the launch (`prologues_taken`).
    query, key, hooks = prologues_taken(query, key, hooks)
    if residuals is None:
        plain = Hooks() if hooks is None else Hooks(hooks.score_mod, hooks.score_params, hooks.initial)
        _, maxima, totals, anchors, sums = forward(query, key, value, spans, plain, scale=scale,
                                                   residuals=Kept.SPLIT, natural=natural, interpret=interpret)
        residuals = maxima, totals, anchors, sums
    initial = hooks is not None and hooks.initial is not None
    # Values whose sums the forward may scale (`algebra.headroom`, `algebra.SCALED_WALK`) take D from the
    # split sums at their rows' scale (`attention_gradient.split_delta`), the pass their tiles are derived
    # for, in either form.
    split = split or algebra.headroom(value.dtype)
    launch = functools.partial(_launch, split=split, natural=natural, accurate=accurate, initial=initial,
                               interpret=interpret)
    return gradients(launch, query, key, value, spans, hooks, dout, residuals, block=BLOCK, scale=scale,
                     rounded=rounded)
