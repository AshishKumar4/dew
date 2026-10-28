"""What the attention template's Mosaic GPU launches share (`attention_mosaic`, design §6): the bundles it
runs on (`bundle`), its configs (`Tiles`, `BackwardTiles`), its legality rule (`refused`), and what a
forward's and a backward's programs read alike: the call's padding to a launch's blocks, its spans' block map
and each tile's coordinates (`_Program`, `_Spans`, `_coords`), its hooks' parameters and state (`_Hooked`,
`_param`)."""

import dataclasses
import functools
import math
from typing import NamedTuple

import jax
import jax.extend.core as jax_core
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention_algebra as algebra, bundles, mosaic, mosaic_sm90
from dew.nn.kernels.templates.attention import (
    Hooks,
    KeySpans,
    Kind,
    PaddedCall,
    block_map,
    differentiable,
    floating_params,
    kept_dtype,
    power_of_two,
    whole,
)

QUALIFIED = frozenset({mosaic_sm90.WGMMA.name})
"""The instruction families the attention mainloop is qualified on: wgmma's. mma.sync's bundle
(`bundles.of`) does not yet pass Mosaic GPU's layout inference in the attention programs (its MMA layouts'
stores and transposed views), so its calls take the frozen Triton mainloop by legality."""


def bundle(generation_name: str | None, dtype) -> mosaic.Family | None:
    """The bundle the attention mainloop runs on a device of `generation_name` for operands of `dtype`: the
    shared registry's (`bundles.of`) where attention is qualified on its family (`QUALIFIED`); None elsewhere,
    whose calls another mainloop takes by legality."""
    family = None if generation_name is None else bundles.of(generation_name, dtype)
    return family if family is not None and family.name in QUALIFIED else None


def bundle_of(tiles: "Tiles | BackwardTiles", dtype) -> mosaic.Family:
    """The bundle `tiles` run on: the current device's (`generation.gpu_target`), whose family their legality
    matched (`MosaicForward.legal`)."""
    family = bundle(generation.gpu_target().generation, dtype)
    if family is None or family.name != tiles.family:
        raise ValueError(f"{tiles}: this device's attention bundle is {family.name if family else None}")
    return family
"""The instruction families this mainloop runs on (`generation.family`), each by its bundle: wgmma's. A
family without one (mma.sync's, until its thin mainloop) takes another mainloop (`MosaicForward.eligible`,
`MosaicBackward.eligible`)."""


@dataclasses.dataclass(frozen=True, order=True)
class Tiles:
    """A forward launch's config: each consumer's query block (`block_q`), the key tile (`block_k`), the key
    and value tiles in flight (`stages`), the consumers a program (`consumers`: on wgmma FlashAttention-3's
    two warpgroups take turns on the tensor cores, one's softmax under the other's products) and the
    instruction family whose bundle runs it (`family`, `bundle`)."""

    block_q: int
    block_k: int
    stages: int
    consumers: int
    family: str = mosaic_sm90.WGMMA.name

    def __post_init__(self):
        if self.stages < 2:
            raise ValueError(f"{self}: two stages or more")


@dataclasses.dataclass(frozen=True, order=True)
class BackwardTiles:
    """A backward's config, one for its launches: the query block a dQ consumer takes and the key block a
    dK/dV consumer takes (`block_q`, `block_k`, so both launches' logits are one tile's), the tiles in
    flight (`stages`), the consumers a program (`consumers`), whether the fast form runs as one launch
    (`fused`, `_launch_backward`) and the instruction family whose bundle runs it (`family`)."""

    block_q: int
    block_k: int
    stages: int
    consumers: int
    fused: bool = False
    family: str = mosaic_sm90.WGMMA.name

    def __post_init__(self):
        if self.stages < 2:
            raise ValueError(f"{self}: two stages or more")


def _width(width: int, family: mosaic.Family, dtype) -> int:
    """A head width as a launch stages it: a power of two (`padded_call`) in whole swizzled rows
    (`Family.column_unit`)."""
    return whole(power_of_two(width), family.column_unit(dtype))


def _block(config: Tiles | BackwardTiles) -> int:
    """The block a launch at `config` pads a call's queries and keys to: whole programs of query blocks and
    key tiles (a forward's), or of query blocks and key blocks (a backward's dQ and dK/dV programs)."""
    keys = config.consumers * config.block_k if isinstance(config, BackwardTiles) else config.block_k
    return math.lcm(config.consumers * config.block_q, keys)


def _padded(queries: int, keys: int, config: Tiles | BackwardTiles) -> tuple[int, int]:
    """A call's queries and keys as a launch at `config` pads them (`padded_call`, `_block`)."""
    block = _block(config)
    return whole(max(queries, 1), block), whole(max(keys, 1), block)


@functools.cache
def _parts(accurate: bool, dtype) -> int:
    """The parts a backward tile's score cotangents take to their products in the form `accurate` names
    (`algebra.cotangent_parts`: one in the fast form's bf16, three in the accurate form's)."""
    tile = jax.ShapeDtypeStruct((1, 1), jnp.float32)
    return len(jax.eval_shape(lambda c: algebra.cotangent_parts(c, dtype, accurate=accurate), tile))


DROPPING = ("the Mosaic mainloop takes no probability dropout yet (the route runs a dropping call on the "
            "reference path, ledger row 1521)")
"""Why this mainloop refuses a dropping call (`refused`): its ops carry no draw."""


def extents(family: mosaic.Family, dtype, extent: int) -> tuple[int, ...]:
    """A tile axis's extents on `family`, largest first: whole multiples of the rows a consumer's products
    cover (`Family.role_rows`) and of its swizzled rows (`Family.column_unit`; a tile's keys are its weights'
    product's depth, its queries the dK/dV launch's), at most what one copy moves (`Family.copy_extent`,
    where the family's copies bound it; else registers bound them, `fits`) and the call's `extent` in whole
    multiples (a longer tile holds padding alone)."""
    unit = math.lcm(family.role_rows, family.column_unit(dtype))
    top = -(-max(extent, 1) // unit)
    if family.copy_extent is not None:
        top = min(top, family.copy_extent // unit)
    return tuple(range(top * unit, 0, -unit))


def whole_tiles(family: mosaic.Family, tiles: "Tiles | BackwardTiles") -> bool:
    """Whether a program of `family` takes `tiles`: its query and key tiles positive whole multiples of the
    rows a role's products cover (`Family.rows`), and one to the family's count of consumers
    (`Family.consumers`), whose turns its walks are written for."""
    return (0 < tiles.consumers <= family.consumers and tiles.block_q > 0 and tiles.block_k > 0
            and tiles.block_q % family.rows == 0 and tiles.block_k % family.rows == 0)


def refused(dtype, width: int, value_width: int, hooks: Hooks | None, *, backward: bool,
            family: mosaic.Family) -> str | None:
    """Why this mainloop takes no call of `dtype` operands, head widths `width` and `value_width` and `hooks`
    (its forward, or with `backward` its backward) on `family`'s bundle, or None: its legality rule, which
    the ops' eligibility (`MosaicForward.eligible`, `MosaicBackward.eligible`), the route and the stability
    suite read."""
    hooks = Hooks() if hooks is None else hooks
    sixteen = (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16))
    if hooks.drop is not None:
        return DROPPING
    if backward and jnp.dtype(dtype) != jnp.dtype(jnp.bfloat16):
        return f"the Mosaic backward takes bf16 operands, not {jnp.dtype(dtype)}"
    if jnp.dtype(dtype) not in sixteen:
        return f"the Mosaic mainloop takes 16-bit operands, not {jnp.dtype(dtype)}"
    if backward and width != value_width:
        return "the Mosaic backward stages dK and dV through one buffer: equal head widths"
    widest = max(_width(width, family, dtype), _width(value_width, family, dtype))
    if family.copy_extent is not None and widest > family.copy_extent:
        return (f"the Mosaic mainloop loads a head {widest} wide in one copy, past the "
                f"{family.copy_extent} one moves (`Family.copy_extent`)")
    return None


APPROXIMATED = frozenset({"exp", "exp2", "log", "log2", "log1p", "tanh", "logistic", "sin", "cos", "sqrt",
                          "rsqrt", "pow"})
"""The operations Mosaic GPU computes in the hardware's approximate forms under `approx_math` (its lowering's
exp, exp2, log, log2, tanh, sin, cos, sqrt and rsqrt; jax's logistic, log1p and pow lower through them)."""


def approximate(hooks: Hooks, tile: tuple[int, int], value_width: int, *, differentiated: bool) -> bool:
    """Whether a launch running `hooks` on `tile` takes the hardware's approximate transcendentals
    (`Family.kernel`'s `approx_math`): only where no hook computes one Mosaic GPU approximates
    (`APPROXIMATED`), with `differentiated` the score hook's pullback too. The mainloop's own exponentials
    hold check 1 at ex2.approx; a hook's need not: on an H100 a softcap's tanh.approx (2^-11 relative) beside
    an initial state put the state's sums' gradient at 7.04 times the reference's error from float64, every
    hook on par with exact math (the hooked backward, HOOKS-FIX-h100)."""
    traced = _hook_jaxprs(hooks, tile, value_width, differentiated=differentiated)
    return not any(_primitives(jaxpr) & APPROXIMATED for jaxpr in traced)


def _hook_jaxprs(hooks: Hooks, tile: tuple[int, int], value_width: int, *,
                 differentiated: bool = False) -> list[jax_core.Jaxpr]:
    """`hooks` traced as a launch runs them on `tile`, its outputs `value_width` wide: the score hook (and,
    `differentiated`, its pullback along the logits and its floating parameters) and the output hook."""
    rows, columns = tile

    def entries(kind: Kind, dtype, width: int) -> jax.ShapeDtypeStruct:
        # A parameter as `_param` gives it: a head's scalar, else the tile's entries.
        return jax.ShapeDtypeStruct(() if kind == Kind.HEAD else (rows, width), kept_dtype(dtype))

    traced = []
    # A keyed launch's hook is its fingerprint (`tuning.program`), whose lowering its live call established.
    score_mod = hooks.score_mod if callable(hooks.score_mod) else None
    if score_mod is not None:
        positions = jax.ShapeDtypeStruct(tile, jnp.int32)
        params = [entries(param.kind, param.value.dtype, columns) for param in hooks.score_params]
        logits = jax.ShapeDtypeStruct(tile, jnp.float32)
        traced.append(jax.make_jaxpr(score_mod)(logits, positions, positions, *params).jaxpr)
        if differentiated:
            floating = floating_params(params)

            def pulled(s, q_pos, k_pos, *values):
                hooked = differentiable(score_mod, q_pos, k_pos, values, floating)
                out, pull = jax.vjp(hooked, s, *(values[index] for index in floating))
                return pull(jnp.ones_like(out))

            traced.append(jax.make_jaxpr(pulled)(logits, positions, positions, *params).jaxpr)
    if callable(hooks.output_mod):
        traced.append(jax.make_jaxpr(hooks.output_mod)(
            jax.ShapeDtypeStruct((rows, value_width), jnp.float32),
            *(entries(param.kind, param.value.dtype, value_width) for param in hooks.output_params)).jaxpr)
    return traced


def _primitives(jaxpr: jax_core.Jaxpr) -> set[str]:
    """The names of the operations of `jaxpr` and of every jaxpr inside it."""
    names = {eqn.primitive.name for eqn in jaxpr.eqns}
    for eqn in jaxpr.eqns:
        for inner in jax_core.jaxprs_in_params(eqn.params):
            names |= _primitives(inner)
    return names


def _columns(array: jax.Array, unit: int) -> jax.Array:
    """`array` with its last axis padded to whole rows of `unit` entries, as the family's swizzled tiles hold
    them (`Family.column_unit`)."""
    return jnp.pad(array, ((0, 0),) * (array.ndim - 1) + ((0, -array.shape[-1] % unit),))


class _Hooked(NamedTuple):
    """A call's hooks as its programs read them: the score hook's parameters and the output hook's, as their
    kinds lay them out (`Hooks.score_params`', `[H]`, `[B, H, Qp]`, `[B, H, Kp]`, `[B, H, Qp, Kp]` or
    `[B, Qp, H, Dv]`), and the softmax's initial state, its maximum and denominator `[B, H, Qp]` and sums
    `[B, H, Qp, Dv]` (None where it starts from none)."""

    score: tuple[jax.Array, ...]
    output: tuple[jax.Array, ...]
    initial: tuple[jax.Array, jax.Array, jax.Array] | None


def _hooked_of(hooks: Hooks, call: PaddedCall, unit: int) -> _Hooked:
    """`hooks` at the padded call's shapes (`padded_call`), the value widths of the output hook's parameters
    and the initial state's sums in whole rows of `unit` entries (`_columns`), as the launch's values are."""
    initial = None
    if hooks.initial is not None:
        state = hooks.initial
        widened = ((0, 0),) * 3 + ((0, call.value.shape[-1] - state.acc.shape[-1]),)
        acc = jnp.pad(call.rows(state.acc), widened)
        initial = call.rows(state.m, -jnp.inf), call.rows(state.total), _columns(acc, unit)
    output = tuple(_columns(value, unit) if param.kind == Kind.OUTPUT else value
                   for value, param in zip(call.output_params, hooks.output_params, strict=True))
    return _Hooked(call.score_params, output, initial)


class _Guard(NamedTuple):
    """A guarded launch's arrays (`algebra.walked_launches`): the rows that shift, `live` `[B, H, Qp]` (int32,
    nonzero where live), and each key's values' largest magnitude, `largest` `[B, Hk, Kp]`, which a step's
    `algebra.room` reads (its values lie in a slot's shared memory)."""

    live: jax.Array
    largest: jax.Array


def _param(family: mosaic.Family, ref, kind: Kind, row, head, first, like, k_first=None):
    """A hook parameter's values for a tile `like` of queries from `first` (and keys from `k_first`, a score
    tile's), as `kind` lays them out: the head's scalar, else the tile's entries (a score tile's `[bq, bk]`,
    an output tile's `[bq, Dv]`) in the accumulator's layout, its queries' and keys' broadcast. A score hook
    reads them and its coordinates as the tile's entries, where the reference and the other mainloops give a
    query's and a key's as `[bq, 1]` and `[1, bk]`: Mosaic GPU's layouts hold no singleton axis an elementwise
    operation takes (jax 0.11.2's layout inference finds none for a `[64, 1]` or `[1, 64]` array's convert or
    compare, a lowering probe for an H100), so a hook that reshapes or indexes an axis fails to lower here,
    never silently; a pointwise hook is the same function of either."""
    rows = pl.ds(first, like.shape[0])
    match kind:
        case Kind.HEAD:
            return ref[head]
        case Kind.QUERY:
            return algebra.expand(family.rows_of(ref.at[row, head, rows]), like)
        case Kind.KEY:
            return algebra.across(family.columns_of(ref.at[row, head, pl.ds(k_first, like.shape[1])]), like)
        case Kind.TILE:
            return family.tile_of(ref.at[row, head, rows, pl.ds(k_first, like.shape[1])])
        case Kind.OUTPUT:
            return family.tile_of(ref.at[row, rows, head])


class _Spans(NamedTuple):
    """A spans call's arrays as its programs read them: each query's interval `start`, `end` `[B, Qp]` (a
    call's runs `[B, R, Qp]`, `_spans_of`), and its block map at the launch's tiles (`attention.block_map`):
    the key tiles each query block reads, `lo`, `hi`, and sees whole, `full_lo`, `full_hi` (`[B, Qp //
    block_q]`), and the query tiles reading each key block, `q_lo`, `q_hi` (`[B, Kp // block_k]`); with
    documents, the queries' and keys' ids (`[B, Qp]`, `[B, Kp]`) and the blocks' uniform ones (`query_doc`,
    `key_doc`)."""

    start: jax.Array
    end: jax.Array
    lo: jax.Array
    hi: jax.Array
    full_lo: jax.Array
    full_hi: jax.Array
    q_lo: jax.Array
    q_hi: jax.Array
    query_ids: jax.Array | None = None
    key_ids: jax.Array | None = None
    query_doc: jax.Array | None = None
    key_doc: jax.Array | None = None


def reached(spec: algebra.Forward, queries: int) -> int | None:
    """The query and key pairs a call of `queries` queries (unpadded) reaches, by its form: every pair of a
    full call; a causal call's, each query's keys up to its own place (`algebra.seen`'s `k_pos <= q_pos`,
    the first key aligned to the first query); None for a spans call, whose pairs its arrays decide, which
    no launch states before it runs."""
    if spec.form == "full":
        return queries * spec.keys
    if spec.form == "causal":
        return sum(min(spec.keys, row + 1) for row in range(queries))
    return None


def stated(spec: algebra.Forward, batch: int, heads: int, queries: int, widths: int, *arrays
           ) -> pl.CostEstimate | None:
    """A launch's cost as it states it to XLA and to Dew's FLOP count (`dew.telemetry.instrumentation`):
    a multiply-add a reached pair (`reached`), head and row for each of `widths` product columns, the
    model's products the launch owns (the forward's `Q K^T` and `P V`, the backward's two transposes of
    each, as the library's fused attention counts them: a recomputed product is the hardware's alone), a
    transcendental a reached pair, and the bytes of each of `arrays` once; None for a spans call, whose
    pairs no launch knows (`reached`): its FLOPs count as unsized, never as 0."""
    pairs = reached(spec, queries)
    if pairs is None:
        return None
    return pl.CostEstimate(flops=2 * batch * heads * pairs * widths, transcendentals=batch * heads * pairs,
                           bytes_accessed=sum(math.prod(array.shape) * jnp.dtype(array.dtype).itemsize
                                              for array in arrays))


def _spans_of(spec: algebra.Forward, call: PaddedCall, block_q: int, block_k: int) -> _Spans | None:
    """A spans call's `_Spans` at `block_q` by `block_k` tiles over its padded keys (None for a full or
    causal call, whose programs take their tiles from their places). A call's runs (`KeySpans`) lie run by
    run, `[B, R, Qp]`, so a program reads each run's bounds as a row (`_coords`); its blocks walk their
    runs' envelopes (`attention.block_map`), each tile masked by the union of its queries' runs
    (`algebra.seen`)."""
    if spec.form != "spans":
        return None
    blocks = block_map(KeySpans(call.start, call.end, call.query_ids, call.key_ids, call.key.shape[1]),
                       block_q, block_k)
    start, end = (jnp.moveaxis(bound, 2, 1) if bound.ndim == 3 else bound for bound in (call.start, call.end))
    return _Spans(start, end, blocks.lo, blocks.hi, blocks.full_lo, blocks.full_hi, blocks.q_lo, blocks.q_hi,
                  call.query_ids, call.key_ids, blocks.query_doc, blocks.key_doc)


@dataclasses.dataclass(frozen=True)
class _Program[T: (Tiles, BackwardTiles)]:
    """A program's static geometry: its family's bundle, its tiles, the call's key tiles (`steps`), the slots
    a stream holds, its key count, whether it is causal, and its query count (`algebra.Forward.queries`; a
    tile holding padding rows is never whole)."""

    family: mosaic.Family
    tiles: T
    steps: int
    stages: int
    keys: int
    causal: bool
    queries: int | None = None

    def padded(self, first) -> jax.Array | bool:
        """Whether the query tile from `first` holds padding rows past the call's queries, some or all of
        them (a call padded to whole programs)."""
        if self.queries is None:
            return False
        return first + self.tiles.block_q > self.queries

    def query_tile(self, count: int):
        """This program's query tile of the grid's `count`: a causal call's longest first (its last query
        tiles see the most keys), so the grid's tail holds its shortest programs, as FlashAttention-3's
        scheduler orders them; a full call's in order."""
        index = lax.axis_index("q_tiles")
        return count - 1 - index if self.causal else index

    def __post_init__(self):
        if not whole_tiles(self.family, self.tiles):
            raise ValueError(f"{self.tiles}: query and key tiles of whole {self.family.rows} rows and "
                             f"{self.family.consumers} consumers at most ({self.family.name})")

    def walked(self, first):
        """The key tiles a query block from `first` reads: those holding the call's keys (none wholly past
        them, `algebra.seen`), up to its diagonal where causal."""
        held = min(-(-self.keys // self.tiles.block_k), self.steps)
        if not self.causal:
            return jnp.int32(held)
        return jnp.minimum(pl.cdiv(first + self.tiles.block_q, self.tiles.block_k), held)

    def whole(self, first, walked):
        """The tiles every query of the block from `first` sees whole: those wholly within the call's keys
        (the padded call's last tiles may all be past them), and below its diagonal."""
        whole = jnp.int32(self.keys // self.tiles.block_k)
        if self.causal:
            whole = jnp.minimum(lax.div(first, self.tiles.block_k), whole)
        return jnp.minimum(jnp.maximum(whole, 0), walked)

    def reach(self, first, spans: _Spans | None, row):
        """The key tiles a query block from `first` reads, `[lo, hi)`, and whether it sees tile `index` whole
        (`whole(index)`, a tile that takes no mask): a full or causal call's from its places (`walked`,
        `whole`), a spans call's from its block map (`spans`, batch row `row`), whole where the ids of
        both blocks agree too."""
        if spans is None:
            walked = self.walked(first)
            # A block holding padding rows is whole nowhere: its padding sees no key (`algebra.seen`).
            count = jnp.where(self.padded(first), 0, self.whole(first, walked))
            return jnp.int32(0), walked, lambda index: index < count
        block = lax.div(first, self.tiles.block_q)
        full_lo, full_hi = spans.full_lo[row, block], spans.full_hi[row, block]

        def whole(index):
            inside = (index >= full_lo) & (index < full_hi)
            if spans.query_doc is None or spans.key_doc is None:
                return inside
            doc = spans.query_doc[row, block]
            return inside & (doc != 0) & (doc == spans.key_doc[row, index])

        return spans.lo[row, block], spans.hi[row, block], whole

    def readers(self, k_first, q_steps: int, spans: _Spans | None, row):
        """The query tiles reading a key block from `k_first`, `[lo, hi)` of the call's `q_steps`, and
        whether tile `index` sees it whole (`whole(index)`): a full or causal call's from its places, from
        its diagonal and none for a block wholly past the keys; a spans call's from its block map, whole
        where the ids of both blocks agree too (`reach`)."""
        block_q, block_k = self.tiles.block_q, self.tiles.block_k
        if spans is None:
            # The first query tile that sees the block: every one, or those from its diagonal (a causal call
            # of more keys than queries leaves its last keys unseen).
            lo = jnp.minimum(lax.div(k_first, block_q), q_steps) if self.causal else jnp.int32(0)
            lo = jnp.where(k_first >= self.keys, q_steps, lo)
            # The tiles from which every query sees the block whole: past its diagonal, and none if it holds
            # padding.
            whole = jnp.minimum(pl.cdiv(k_first + block_k - 1, block_q), q_steps) if self.causal else lo
            whole = jnp.where(k_first + block_k > self.keys, q_steps, jnp.maximum(whole, lo))
            # A query tile holding padding rows sees no block whole (`reach`).
            return lo, jnp.int32(q_steps), lambda index: (index >= whole) & jnp.logical_not(
                self.padded(index * block_q))
        block = lax.div(k_first, block_k)

        def seen_whole(index):
            inside = (block >= spans.full_lo[row, index]) & (block < spans.full_hi[row, index])
            if spans.query_doc is None or spans.key_doc is None:
                return inside
            doc = spans.key_doc[row, block]
            return inside & (doc != 0) & (doc == spans.query_doc[row, index])

        return spans.q_lo[row, block], spans.q_hi[row, block], seen_whole


def _streamed(reaches: list) -> tuple[jax.Array, jax.Array]:
    """The tiles a program streams: those any of its consumers' `reaches` (`[lo, hi)` each) reads."""
    return (functools.reduce(jnp.minimum, [lo for lo, _ in reaches]),
            functools.reduce(jnp.maximum, [hi for _, hi in reaches]))


def _coords(family: mosaic.Family, spans: _Spans | None, row, first, k_first, shape: tuple[int, int],
            q_pos=None, columns=None) -> algebra.Coords:
    """A tile's coordinates (`algebra.Coords`): its queries' and keys' places from `first` and `k_first`, the
    tile's entries in the accumulator's layout (`q_pos` the query places and `columns` the tile's key
    offsets, where the caller holds them; a hook reads them so, `_param`), and a spans call's queries'
    intervals and, with documents, both blocks' ids."""
    q_pos = first + family.iota(shape, 0) if q_pos is None else q_pos
    k_pos = k_first + (family.iota(shape, 1) if columns is None else columns)
    if spans is None:
        return algebra.Coords(q_pos, k_pos)
    rows = pl.ds(first, shape[0])

    def bounds_of(*run):
        # A query's interval, or a call's run's, one row of bounds a run (`_spans_of`).
        return (family.rows_of(spans.start.at[(row, *run, rows)]),
                family.rows_of(spans.end.at[(row, *run, rows)]))

    runs = [(run,) for run in range(spans.start.shape[1])] if spans.start.ndim == 3 else [()]
    bounds = tuple(bounds_of(*run) for run in runs)
    ids = None
    if spans.query_ids is not None and spans.key_ids is not None:
        ids = (family.rows_of(spans.query_ids.at[row, rows]),
               family.columns_of(spans.key_ids.at[row, pl.ds(k_first, shape[1])]))
    return algebra.Coords(q_pos, k_pos, bounds, ids)


def _rows(family: mosaic.Family, ref, row, head, first, count: int):
    """A query block's per-row statistic `[count]` of a `[B, H, Q]` array, in a row's layout."""
    return family.rows_of(ref.at[row, head, pl.ds(first, count)])


def _views_of(refs: dict, row, head) -> dict:
    """The streams' views by name (`Family.opened`, `Family.stream`, `mosaic.viewed`): each `[B, S, H, D]`
    operand's `[S, D]` at batch row `row` and head `head`, step `j` its `j`-th block of rows."""
    return {name: (ref.at[row, :, head], lambda j: (j, 0)) for name, ref in refs.items()}
