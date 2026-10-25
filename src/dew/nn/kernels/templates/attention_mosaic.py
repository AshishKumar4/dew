"""The attention template's Mosaic GPU mainloop (design §6, one mainloop per overlap structure): the order of
an instruction family's bundle's steps (`mosaic.Family`: its roles, operand streams, products and layouts)
around the template's algebra (`attention_algebra`), FlashAttention-3's structure on the wgmma family
(`mosaic_sm90.WGMMA`). It holds no instruction, layout or stream barrier of its own: only the overlap
structure, the query block's load and the consumers' schedule.

One program a query head and the family's consumers' query blocks (two warpgroups on wgmma, each a block of
`Tiles.block_q` rows), beside the family's loads streaming the key and value tiles (`Family.stream`,
`Family.walk`), every slot released by every consumer. Two consumers take turns on the tensor cores (a
schedule barrier between them), so one's softmax runs under the other's products. A step issues S = Q·Kᵀ,
releases the key slot, and hands S to the algebra as a tile (`algebra.Keys`) whose values are the value slot
and whose product accumulates the narrowed weights from registers into the running sums in place; while it
runs the consumer waits for its next key tile. A block reads the key tiles its spans reach (a full or causal
call's from its places, a spans call's from its block map, `attention.block_map`) and masks only those it
does not see whole (`algebra.seen`); a consumer keeps releasing the slots its partner alone reads.

The forward over every call form of 16-bit operands under every hook (`Hooks`: the score hook and its
parameters on each tile, the softmax's initial state in the walk's start, the output hook on its output), the
base-2 fold and natural logits, under the default's guards over whole launches (`algebra.walked_launches`: a
program walks its keys once, so the call walks again in the natural form where some row's logits pass the
fold's reach, and its overflowed rows again shifting under headroom). Its legality rule is `refused`."""

import dataclasses
import functools
from typing import NamedTuple

import jax
import jax.extend.core as jax_core
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention_algebra as algebra, mosaic, mosaic_sm90
from dew.nn.kernels.templates.attention import (
    ACCURATE,
    NATURAL,
    Gradients,
    Hooks,
    KeySpans,
    Kind,
    PaddedCall,
    Split,
    block_map,
    gradients,
    kept_dtype,
    padded_call,
)

COLUMNS = 64
"""The head width a family's swizzled tiles hold whole rows of, in 16-bit entries (128 bytes)."""


def family_of(dtype) -> mosaic.Family:
    """The bundle of the device's instruction family (`generation.family`) for operands of `dtype`."""
    name = generation.family(generation.device_generation())
    del dtype
    if name == "wgmma":
        return mosaic_sm90.WGMMA
    raise NotImplementedError(f"the Mosaic attention mainloop has no bundle for the {name!r} family")


@dataclasses.dataclass(frozen=True, order=True)
class Tiles:
    """A launch's tiles: each consumer's query block (`block_q`, whole rows of the family's MMA), the key tile
    (`block_k`) and the key and value tiles in flight (`stages`)."""

    block_q: int
    block_k: int
    stages: int

    def __post_init__(self):
        if self.stages < 2:
            raise ValueError(f"{self}: two stages or more")


DEFAULTS = {"wgmma": Tiles(64, 128, 2)}
"""The forward's tiles by family: wgmma's FlashAttention-3's at width 128, a warpgroup's 64 queries over 128
keys in two stages, the fastest of 64 or 128 queries over 64 or 128 keys in two to four stages on an H100
(llama-d128 causal bf16, 2 x 4096: 0.49 ms; 128 queries a consumer spill, 2.9 to 4.6 ms)."""


def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, natural: bool = NATURAL, tiles: Tiles | None = None,
            family: mosaic.Family | None = None, interpret: bool = False) -> jax.Array:
    """The template's forward on this mainloop: `query` `[B, Q, H, D]` over `key` and `value` `[B, K, Hk, D]`
    and the keys `spans` leaves each query under `hooks`, as `dew.nn.kernels.templates.attention.reference`
    computes it, by `family`'s bundle (the device's, `family_of`) at `tiles` (its default, `DEFAULTS`).
    `natural` keeps the logits natural (`algebra.Forward.base2`); `interpret` runs it on the CPU (Mosaic
    GPU's interpret mode)."""
    family = family_of(query.dtype) if family is None else family
    tiles = DEFAULTS[family.name] if tiles is None else tiles
    hooks = Hooks() if hooks is None else hooks
    width, value_width = query.shape[3], value.shape[3]
    if reason := refused(query.dtype, width, value_width, hooks, backward=False,
                         tile=(tiles.block_q, tiles.block_k)):
        raise NotImplementedError(reason)
    block = max(family.consumers * tiles.block_q, tiles.block_k)
    call = padded_call(query, key, value, spans, hooks, block)
    # Every form guards its sums where its values can overflow them (`algebra.Forward.headroom`), and the
    # fold its logits' reach, over whole launches (`algebra.walked_launches`).
    spec = algebra.Forward(spans.form, spans.keys, ids=spans.query_ids is not None,
                           scale=width ** -0.5 if scale is None else scale, score_mod=hooks.score_mod,
                           natural=natural, headroom=algebra.headroom(value.dtype),
                           initial=hooks.initial is not None)
    query, key, value = (_columns(array) for array in (call.query, call.key, call.value))
    spans_of, hooked = _spans_of(spec, call, tiles.block_q, tiles.block_k), _hooked_of(hooks, call)

    def launch(form: algebra.Forward, live: jax.Array | None):
        # The guarded launch reads its live rows and its keys' values' largest magnitudes (`algebra.room`).
        guard = None if live is None else _Guard(live.astype(jnp.int32),
                                                 jnp.swapaxes(algebra.magnitudes(value), 1, 2))
        out, status = _launch(query, key, value, spans_of, hooked, guard, spec=form, hooks=hooks, tiles=tiles,
                              family=family, interpret=interpret)
        return out, status[0], status[1] > 0

    def kept(live: jax.Array, guarded: jax.Array, plain: jax.Array) -> jax.Array:
        return jnp.where(jnp.swapaxes(live, 1, 2)[..., None], guarded, plain)

    out = algebra.walked_launches(spec, launch, kept)
    return out[:, :call.queries, :, :value_width]


def _columns(array: jax.Array) -> jax.Array:
    """`array` with its last axis padded to whole rows of `COLUMNS` 16-bit entries, as the family's swizzled
    tiles hold them."""
    return jnp.pad(array, ((0, 0),) * (array.ndim - 1) + ((0, -array.shape[-1] % COLUMNS),))


def refused(dtype, width: int, value_width: int, hooks: Hooks | None, *, backward: bool,
            tile: tuple[int, int]) -> str | None:
    """Why this mainloop takes no call of `dtype` operands, head widths `width` and `value_width` and `hooks`
    (its forward, or with `backward` its backward) at `tile`'s queries by keys, or None: its legality rule,
    which `forward` and `backward` refuse by and the route's chooser and the stability suite read."""
    hooks = Hooks() if hooks is None else hooks
    sixteen = (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16))
    if hooks.drop is not None:
        return ("the Mosaic mainloop takes no probability dropout yet (the route runs a dropping call on the "
                "reference path, ledger row 1521)")
    if backward and hooks != Hooks():
        return "the Mosaic backward takes no hooks yet"
    if backward and jnp.dtype(dtype) != jnp.dtype(jnp.bfloat16):
        return f"the Mosaic backward takes bf16 operands, not {jnp.dtype(dtype)}"
    if jnp.dtype(dtype) not in sixteen:
        return f"the Mosaic mainloop takes 16-bit operands, not {jnp.dtype(dtype)}"
    if width != value_width:
        return ("the Mosaic mainloop's output reuses the query's stage, and its backward stages dK and dV "
                "through one buffer: equal head widths")
    # The hooks as the launch runs them: its tile, its values padded to whole `COLUMNS` rows (`_columns`).
    return unlowered(hooks, tile, -(-value_width // COLUMNS) * COLUMNS)


def unlowered(hooks: Hooks, tile: tuple[int, int], value_width: int) -> str | None:
    """Why Mosaic GPU cannot lower `hooks` on this mainloop's tiles (`tile`, its outputs `value_width` wide),
    or None: a hook that forms an array of a unit axis beside a longer one (`q_pos[:, :1]`, a reshape), whose
    every operation jax 0.11.2's Mosaic GPU layout inference finds no layout for. A legality rule, which
    the route's chooser reads to send such a call to another route, until the fork's layout inference
    takes unit axes (ledger row 1428)."""
    rows, columns = tile

    def entries(kind: Kind, dtype, width: int) -> jax.ShapeDtypeStruct:
        # A parameter as `_param` gives it: a head's scalar, else the tile's entries.
        return jax.ShapeDtypeStruct(() if kind == Kind.HEAD else (rows, width), kept_dtype(dtype))

    traced = []
    if hooks.score_mod is not None:
        positions = jax.ShapeDtypeStruct(tile, jnp.int32)
        traced.append(jax.make_jaxpr(hooks.score_mod)(
            jax.ShapeDtypeStruct(tile, jnp.float32), positions, positions,
            *(entries(param.kind, param.value.dtype, columns) for param in hooks.score_params)).jaxpr)
    if hooks.output_mod is not None:
        traced.append(jax.make_jaxpr(hooks.output_mod)(
            jax.ShapeDtypeStruct((rows, value_width), jnp.float32),
            *(entries(param.kind, param.value.dtype, value_width) for param in hooks.output_params)).jaxpr)
    if any(_unit_axis(jaxpr) for jaxpr in traced):
        return ("the Mosaic mainloop takes no hook that forms a unit axis beside a longer one (Mosaic GPU's "
                "layout inference finds no layout for it)")
    return None


def _unit_axis(jaxpr: jax_core.Jaxpr) -> bool:
    """Whether any operation of `jaxpr`, or of a jaxpr inside it, gives a two-dimensional array of a unit
    axis beside a longer one."""
    for eqn in jaxpr.eqns:
        shapes = [getattr(var.aval, "shape", ()) for var in eqn.outvars]
        if any(len(shape) == 2 and 1 in shape and max(shape) > 1 for shape in shapes):
            return True
        if any(_unit_axis(inner) for inner in jax_core.jaxprs_in_params(eqn.params)):
            return True
    return False


class _Hooked(NamedTuple):
    """A call's hooks as its programs read them: the score hook's parameters and the output hook's, as their
    kinds lay them out (`Hooks.score_params`', `[H]`, `[B, H, Qp]`, `[B, H, Kp]`, `[B, H, Qp, Kp]` or
    `[B, Qp, H, Dv]`), and the softmax's initial state, its maximum and denominator `[B, H, Qp]` and sums
    `[B, H, Qp, Dv]` (None where it starts from none)."""

    score: tuple[jax.Array, ...]
    output: tuple[jax.Array, ...]
    initial: tuple[jax.Array, jax.Array, jax.Array] | None


def _hooked_of(hooks: Hooks, call: PaddedCall) -> _Hooked:
    """`hooks` at the padded call's shapes (`padded_call`), the value widths of the output hook's parameters
    and the initial state's sums in whole `COLUMNS` rows, as the launch's values are."""
    initial = None
    if hooks.initial is not None:
        state = hooks.initial
        initial = (call.rows(state.m, -jnp.inf), call.rows(state.total),
                   _columns(jnp.pad(call.rows(state.acc),
                                    ((0, 0),) * 3 + ((0, call.value.shape[-1] - state.acc.shape[-1]),))))
    output = tuple(_columns(value) if param.kind == Kind.OUTPUT else value
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
    """A spans call's arrays as its programs read them: each query's interval `start`, `end` `[B, Qp]`, and
    its block map at the launch's tiles (`attention.block_map`): the key tiles each query block reads, `lo`,
    `hi`, and sees whole, `full_lo`, `full_hi` (`[B, Qp // block_q]`), and the query tiles reading each key
    block, `q_lo`, `q_hi` (`[B, Kp // block_k]`); with documents, the queries' and keys' ids (`[B, Qp]`,
    `[B, Kp]`) and the blocks' uniform ones (`query_doc`, `key_doc`)."""

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


def _spans_of(spec: algebra.Forward, call: PaddedCall, block_q: int, block_k: int) -> _Spans | None:
    """A spans call's `_Spans` at `block_q` by `block_k` tiles over its padded keys (None for a full or
    causal call, whose programs take their tiles from their places)."""
    if spec.form != "spans":
        return None
    blocks = block_map(KeySpans(call.start, call.end, call.query_ids, call.key_ids, call.key.shape[1]),
                       block_q, block_k)
    return _Spans(call.start, call.end, blocks.lo, blocks.hi, blocks.full_lo, blocks.full_hi, blocks.q_lo,
                  blocks.q_hi, call.query_ids, call.key_ids, blocks.query_doc, blocks.key_doc)


@dataclasses.dataclass(frozen=True)
class _Program:
    """A program's static geometry: its family's bundle, its tiles, the call's key tiles (`steps`), the slots
    a stream holds, its key count and whether it is causal."""

    family: mosaic.Family
    tiles: Tiles
    steps: int
    stages: int
    keys: int
    causal: bool

    def query_tile(self, count: int):
        """This program's query tile of the grid's `count`: a causal call's longest first (its last query
        tiles see the most keys), so the grid's tail holds its shortest programs, as FlashAttention-3's
        scheduler orders them; a full call's in order."""
        index = lax.axis_index("q_tiles")
        return count - 1 - index if self.causal else index

    def __post_init__(self):
        unit = self.family.rows
        if self.tiles.block_q % unit or self.tiles.block_k % unit:
            raise ValueError(f"{self.tiles}: query and key tiles of whole {unit} rows ({self.family.name})")

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
            count = self.whole(first, walked)
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
            return lo, jnp.int32(q_steps), lambda index: index >= whole
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
    rows = (row, pl.ds(first, shape[0]))
    bounds = family.rows_of(spans.start.at[rows]), family.rows_of(spans.end.at[rows])
    ids = None
    if spans.query_ids is not None and spans.key_ids is not None:
        ids = (family.rows_of(spans.query_ids.at[rows]),
               family.columns_of(spans.key_ids.at[row, pl.ds(k_first, shape[1])]))
    return algebra.Coords(q_pos, k_pos, bounds, ids)


def _consumer(program: _Program, spec: algebra.Forward, consumer, q_view, out_view, status_views, q_smem,
              q_loaded, row_smem, streams, turn, streamed: tuple, first, spans: _Spans | None,
              hooked: _Hooked, guard: _Guard | None, place: tuple, *, hooks: Hooks, interpret: bool):
    """A consumer's query block (batch row, head and key head `place`): its walk of the key tiles it reads
    through the family's streams (`algebra.plain_step` a step) from the hooks' initial state, the score hook
    on each tile, passing those of the program's `streamed` range its partner alone reads; then its output
    under the output hook stored through its query stage, and its rows' maxima and whether their weighted
    values overflowed through `row_smem` to `status_views` (`algebra.walked_launches`). The plain walk's
    running sums are the family's accumulator, issued into in place; the guarded walk (`guard`, its rows
    `live` shifting) carries them, each tile's product fresh (`algebra.stepped`)."""
    row, head, kv_head = place
    family, tiles = program.family, program.tiles
    shape = (tiles.block_q, tiles.block_k)
    lo, hi, whole = program.reach(first, spans, row)
    family.loaded(((q_view, q_smem, q_loaded),))
    q = family.held(q_smem)
    q_pos, columns = first + family.iota(shape, 0), family.iota(shape, 1)
    width = q_smem.shape[-1]
    # The weighted values' running sums live in the family's accumulator (`Family.walk`), from the initial
    # state's where there is one; the walk carries the maxima and denominators, the state's `acc` an empty
    # place the algebra passes through (`algebra.Keys.accumulate`), a constant no loop may carry.
    state_rows = (row, head, pl.ds(first, tiles.block_q))
    live = None if guard is None else family.rows_of(guard.live.at[state_rows]) > 0
    if hooked.initial is not None:
        m0, total0, acc0 = hooked.initial
        start = algebra.resumed(spec, family.rows_of(m0.at[state_rows]),
                                family.rows_of(total0.at[state_rows]), family.tile_of(acc0.at[state_rows]),
                                live=live)
    else:
        # `algebra.started`'s empty state, in the family's layouts.
        start = algebra.Partial(family.zeros((tiles.block_q, width)), family.full((tiles.block_q,), -jnp.inf),
                                family.zeros((tiles.block_q,)))
        start = start if live is None else algebra.headroomed(start, live)
    pl.when(consumer == 1)(turn)
    # The slots before this block's first tile its partner reads, the turns it would have taken.
    family.passed(streams, streamed[0], lo, first=streamed[0], between=lambda: (turn(), turn()))

    def step(index, slots, carry, accumulators):
        s = family.product(q, slots["keys"].transpose((1, 0)), shape, turn)
        slots.release("keys")
        k_first = index * tiles.block_k
        coords = _coords(family, spans, row, first, k_first, shape, q_pos, columns)
        params = tuple(_param(family, ref, param.kind, row, head, first, s, k_first)
                       for ref, param in zip(hooked.score, hooks.score_params, strict=True))
        # A tile every query sees whole takes no mask: one walk, the choice a tile's (a walk of each, these
        # rings' waits and arrivals in two loops, faulted on an H100 under jax 0.11.2's lowering).
        logits, products = algebra.logits_of(spec, s, coords, params, masked=jnp.logical_not(whole(index)))

        def load():
            # The value slot, once the partner's products are issued: its turn on the tensor cores is over
            # when this one's come.
            turn(wait=False)
            slot = slots["values"]
            turn(arrive=False)
            return slot

        def dot(lhs, rhs):
            return family.registers_product(lhs, rhs, interpret)

        if guard is not None:
            magnitudes = guard.largest

            def largest():
                return family.columns_of(magnitudes.at[row, kv_head, pl.ds(k_first, tiles.block_k)])

            keys = algebra.Keys(coords, logits, load, dot, products=products, largest=largest)
            return algebra.plain_step(spec, carry, keys), accumulators
        held = list(accumulators)

        def accumulate(scale, p, value):
            held[0] = family.rescaled_products(held[0], scale, ((p, None),), value, interpret)

        state = algebra.Partial(jnp.zeros((), jnp.float32), *carry)
        state = algebra.plain_step(spec, state, algebra.Keys(coords, logits, load, dot, accumulate, products))
        return (state.m, state.total), tuple(held)

    if guard is not None:
        state, _ = family.walk(streams, ((step, lo, hi),), start, (), interpret, first=streamed[0])
    else:
        rows, (acc,) = family.walk(streams, ((step, lo, hi),), (start.m, start.total),
                                   ((tiles.block_q, width),), interpret, first=streamed[0],
                                   initial=(start.acc,))
        state = algebra.Partial(acc, *rows)
    # The slots past this block's last tile (a causal block whose diagonal ends first) its partner reads.
    family.passed(streams, hi, streamed[1], first=streamed[0], between=lambda: (turn(), turn()))
    pl.when(consumer == 0)(turn)
    overflowed = jnp.where(algebra.overflowed(state.acc), 1.0, 0.0)
    family.stored((family.as_row(state.m), family.as_row(overflowed)), status_views, smem=row_smem)
    out_params = tuple(_param(family, ref, param.kind, row, head, first, state.acc)
                       for ref, param in zip(hooked.output, hooks.output_params, strict=True))
    out, _, _ = algebra.finish(spec, state, initial=hooked.initial is not None, output_mod=hooks.output_mod,
                               output_params=out_params)
    family.stored((out,), (out_view,), smem=q_smem)


def _launch(query, key, value, spans: _Spans | None, hooked: _Hooked, guard: _Guard | None, *,
            spec: algebra.Forward, hooks: Hooks, tiles: Tiles, family: mosaic.Family, interpret: bool):
    """The forward's launch in `spec`'s form (`guard`'s rows shifting where given): its output `[B, Qp, H,
    Dv]` and its rows' status `[2, B, H, Qp]` fp32, their maxima in the walk's units and 1 where their
    weighted values overflowed (`algebra.walked_launches`)."""
    batch, queries, heads, width = query.shape
    keys_count, kv_heads = key.shape[1], key.shape[2]
    group, consumers = heads // kv_heads, family.consumers
    block_q, block_k = tiles.block_q, tiles.block_k
    pair = consumers * block_q
    steps = keys_count // block_k
    # No more stages than key tiles: a slot past them would load past the keys.
    program = _Program(family, tiles, steps, min(tiles.stages, steps), spec.keys, spec.form == "causal")

    def body(q_ref, k_ref, v_ref, *refs, streams):
        read, (out_ref, status_ref, q_smem, row_smem, q_loaded, schedule) = (refs[:len(leaves)],
                                                                             refs[len(leaves):])
        spans_of, hooked_of, guard_of = treedef.unflatten(read)
        row, head = lax.axis_index("batch"), lax.axis_index("heads")
        views = _views_of({"keys": k_ref, "values": v_ref}, row, lax.div(head, group))
        streams = family.opened(streams, views)
        pair_first = program.query_tile(queries // pair) * pair
        streamed = _streamed([program.reach(pair_first + index * block_q, spans_of, row)[:2]
                              for index in range(consumers)])

        def turn(*, arrive: bool = True, wait: bool = True):
            # The consumers' turns on the tensor cores, where a program has two (a schedule barrier).
            if consumers > 1 and arrive:
                plgpu.barrier_arrive(schedule)
            if consumers > 1 and wait:
                plgpu.barrier_wait(schedule)

        def consumer(index):
            first = pair_first + index * block_q
            rows = (row, pl.ds(first, block_q), head)
            status = tuple(status_ref.at[place, row, head, pl.ds(first, block_q)] for place in range(2))
            _consumer(program, spec, index, q_ref.at[rows], out_ref.at[rows], status, q_smem.at[index],
                      q_loaded.at[index], row_smem.at[index], streams, turn, streamed, first, spans_of,
                      hooked_of, guard_of, (row, head, lax.div(head, group)), hooks=hooks,
                      interpret=interpret)

        def producer():
            family.stream(streams, views, *streamed)

        family.roles(consumers, consumer, producer)

    blocks = {"keys": ((block_k, width), key.dtype), "values": ((block_k, value.shape[-1]), value.dtype)}
    # The arrays the programs read beside the operands.
    leaves, treedef = jax.tree.flatten((spans, hooked, guard))

    def entry(*refs):
        family.scoped(
            functools.partial(body, *refs),
            plgpu.SMEM((consumers, block_q, width), query.dtype),
            plgpu.SMEM((consumers, block_q), jnp.float32),
            plgpu.Barrier(num_barriers=consumers), plgpu.Barrier(num_arrivals=consumers),
            streams=family.streams(blocks, program.stages, consumers))

    out_type = (jax.ShapeDtypeStruct((batch, queries, heads, value.shape[-1]), query.dtype),
                jax.ShapeDtypeStruct((2, batch, heads, queries), jnp.float32))
    return family.kernel(entry, out_type=out_type, grid=(heads, queries // pair, batch),
                         grid_names=("heads", "q_tiles", "batch"), consumers=consumers,
                         interpret=interpret)(query, key, value, *leaves)


BACKWARD_DEFAULTS = {"wgmma": Tiles(64, 64, 2)}
"""The backward's tiles by family: each consumer's 64 queries over 64-key tiles, two stages, so a dQ program's
key and value streams, its consumers' queries and output cotangents, and their fp32 sums fit a block's shared
memory."""
BACKWARD_CONSUMERS = {False: 2, True: 1}
"""The backward's consumer warpgroups a program, by form (`accurate`), each a block of its own, their products
interleaved on the tensor cores: the fast form's two (an H100, llama-d128 causal bf16: the dQ launch 1.10 ms
against one consumer's 1.41), the accurate form's one, whose staged cotangent parts two consumers' blocks
would hold past a block's shared memory."""


def _rows(family: mosaic.Family, ref, row, head, first, count: int):
    """A query block's per-row statistic `[count]` of a `[B, H, Q]` array, in a row's layout."""
    return family.rows_of(ref.at[row, head, pl.ds(first, count)])


def _views_of(refs: dict, row, head) -> dict:
    """The streams' views by name (`Family.opened`, `Family.stream`, `mosaic.viewed`): each `[B, S, H, D]`
    operand's `[S, D]` at batch row `row` and head `head`, step `j` its `j`-th block of rows."""
    return {name: (ref.at[row, :, head], lambda j: (j, 0)) for name, ref in refs.items()}


@dataclasses.dataclass(frozen=True)
class _Backward:
    """A backward launch's static choices: its program's geometry, its forward's (`spec`: the call's form, the
    logits' scale and units), its form, its consumers, whether it is interpreted, and whether the dK/dV
    program adds each tile's dQ too (`fused`: one launch, dQ by atomic adds, D the forward's anchored one
    unrefined)."""

    program: _Program
    spec: algebra.Forward
    accurate: bool
    consumers: int
    interpret: bool
    fused: bool

    @property
    def how(self) -> algebra.Recompute:
        """How the backward recomputes the forward's weights."""
        return algebra.recompute(self.spec)

    @property
    def scale(self) -> float:
        """The logits' scale."""
        return self.spec.scale


def _scores(walk: _Backward, coords: algebra.Coords, masked: bool, s, dp, peak, total, anchor, delta, chosen):
    """A backward tile's weights unnormalized and over `total`, its score cotangents (`algebra.grad_scores`),
    its entries' derivatives and its mask (`algebra.seen` where `masked`) from its products `s` and `dp`, its
    coordinates and its queries' statistics."""
    seen = algebra.seen(walk.spec, coords, s) if masked else None
    how = walk.how
    logits, slopes = algebra.tile_logits(None, s, seen, None, None, (),
                                         scale=walk.spec.scale if how.factor is None else 1.0, slopes=True)
    exponentials, weights = algebra.normalized(logits, peak, total, how)
    at_anchor = coords.k_pos == algebra.expand(chosen, coords.k_pos)
    cotangents = algebra.grad_scores(exponentials, dp, anchor, delta, seen, at_anchor, algebra.inverse(total))
    return exponentials, weights, cotangents, slopes, seen


def _dq_step(walk: _Backward, operands: tuple, stats: tuple, coords_of, whole, index, slots, rows,
             accumulators):
    """A dQ consumer's key tile `index`: its scores and score cotangents from the key and value slots
    (`slots`) and its queries and output cotangents (`operands`) under its rows' statistics (`stats`), the
    tile's coordinates (`coords_of(k_first)`) and whether its block sees it whole (`whole(index)`), the row
    sums carried (`rows`) and dQ's and the keyed sums' products issued into `accumulators` in place."""
    q, do = operands
    peak, prior, anchor, delta, chosen = stats
    family, tiles, empty = walk.program.family, walk.program.tiles, jnp.zeros((), jnp.float32)
    shape = (tiles.block_q, tiles.block_k)
    dq_acc, keyed_acc = accumulators
    k_slot = slots["keys"]
    s = family.product(q, k_slot.transpose((1, 0)), shape)
    dp = family.product(do, slots["values"].transpose((1, 0)), shape)
    slots.release("values")

    def scores(*, masked: bool):
        # A tile every query sees whole (`masked` False) takes no mask. Without a hook nothing past the
        # cotangents reads the mask (`algebra.pulled`).
        coords = coords_of(index * tiles.block_k)
        exponentials, _, cotangents, _, _ = _scores(walk, coords, masked, s, dp, peak, prior, anchor, delta,
                                                    chosen)
        return exponentials, cotangents

    # One walk, its tiles' masks chosen a tile at a time: a walk of two loops into one warpgroup accumulator,
    # the first empty, faults on an H100 under jax 0.11.2's Mosaic GPU lowering.
    exponentials, cotangents = lax.cond(whole(index), functools.partial(scores, masked=False),
                                        functools.partial(scores, masked=True))
    # The weights' sums unnormalized (`algebra.dq_finish` takes them to the pass's own denominator); dQ's and
    # the keyed sums' places in the step's sums empty, as they accumulate in place.
    residual, mass, keyed_mass, keyed_shift = rows
    sums = algebra.DqSums(empty, residual, mass, (), (), empty, keyed_mass, keyed_shift)
    sums, parts = algebra.dq_terms(sums, exponentials, cotangents, None, None, scale=walk.scale, kinds=(),
                                   summed=(), dtype=k_slot.dtype, accurate=walk.accurate)
    dq_acc = family.register_products(dq_acc, parts, k_slot, walk.interpret)
    if walk.accurate:
        sums, keyed = algebra.keyed_terms(sums, exponentials, prior, scale=walk.scale, dtype=k_slot.dtype)
        keyed_acc = family.register_products(keyed_acc, keyed, k_slot, walk.interpret)
    return (sums.residual, sums.mass, sums.keyed_mass, sums.keyed_shift), (dq_acc, keyed_acc)


def _dq_launch(query, key, value, dout, peak, prior, anchor, delta, key_index, spans: _Spans | None, *,
               walk: _Backward):
    """The dQ program's sums over its key tiles (`algebra.dq_step`), unfinished: dQ's and, in the accurate
    form, the keys' product with the weights (`DqSums.keyed`), `[B, Qp, H, D]` fp32 each, and the cotangents'
    and weights' row sums and the keyed sums' mass and exponent, `[B, H, Qp]` each; `dq_finish` takes them to
    dQ and the refined D past the launch. Each query block of a pair is a consumer's, the key and value tiles
    streamed through a ring each."""
    batch, queries, heads, width = query.shape
    kv_heads, value_width = key.shape[2], value.shape[-1]
    program, consumers = walk.program, walk.consumers
    family = program.family
    block_q, block_k = program.tiles.block_q, program.tiles.block_k
    pair = consumers * block_q

    leaves, treedef = jax.tree.flatten(spans)

    def body(*refs, streams):
        (q_ref, k_ref, v_ref, do_ref, peak_ref, prior_ref, anchor_ref, delta_ref, index_ref), refs = (
            refs[:9], refs[9:])
        spans_of = None if spans is None else treedef.unflatten(refs[:len(leaves)])
        (dq_ref, keyed_ref, rows_ref, q_smem, do_smem, out_smem, row_smem, q_loaded,
         do_loaded) = refs[len(leaves):]
        row, head = lax.axis_index("batch"), lax.axis_index("heads")
        views = _views_of({"keys": k_ref, "values": v_ref}, row, lax.div(head, heads // kv_heads))
        streams = family.opened(streams, views)
        pair_first = program.query_tile(queries // pair) * pair
        streamed = _streamed([program.reach(pair_first + index * block_q, spans_of, row)[:2]
                              for index in range(consumers)])

        def consumer(index):
            first = pair_first + index * block_q
            rows = (row, pl.ds(first, block_q), head)
            family.loaded(((q_ref.at[rows], q_smem.at[index], q_loaded.at[index]),
                           (do_ref.at[rows], do_smem.at[index], do_loaded.at[index])))
            stats = tuple(_rows(family, ref, row, head, first, block_q)
                          for ref in (peak_ref, prior_ref, anchor_ref, delta_ref, index_ref))
            lo, hi, whole = program.reach(first, spans_of, row)
            nothing = family.zeros((block_q,))
            operands = family.held(q_smem.at[index]), family.held(do_smem.at[index])
            shape = (block_q, block_k)
            q_pos, columns = first + family.iota(shape, 0), family.iota(shape, 1)
            coords_of = functools.partial(_coords, family, spans_of, row, first, shape=shape, q_pos=q_pos,
                                          columns=columns)
            step = functools.partial(_dq_step, walk, operands, stats, coords_of, whole)
            # The tiles before this block's first its partner reads.
            family.passed(streams, streamed[0], lo, first=streamed[0])
            sums, (dq, keyed) = family.walk(
                streams, ((step, lo, hi),), (nothing, nothing, nothing, nothing + algebra.NO_SHIFT),
                ((block_q, width), (block_q, width)), walk.interpret, first=streamed[0])
            # The tiles past this block's last (a causal block whose diagonal ends first) its partner reads.
            family.passed(streams, hi, streamed[1], first=streamed[0])
            # The keyed sums the accurate form's alone (`algebra.dq_terms`).
            written = ((dq, dq_ref.at[rows]), (keyed, keyed_ref.at[rows]))[:2 if walk.accurate else 1]
            family.stored(*zip(*written, strict=True), smem=out_smem.at[index])
            # Each row sum in a row's layout whatever its own (the keyed mass of a call without a hook is
            # a constant).
            family.stored(tuple(family.as_row(part) for part in sums),
                          tuple(rows_ref.at[place, row, head, pl.ds(first, block_q)] for place in range(4)),
                          smem=row_smem.at[index])

        def producer():
            family.stream(streams, views, *streamed)

        family.roles(consumers, consumer, producer)

    blocks = {"keys": ((block_k, width), key.dtype), "values": ((block_k, value_width), value.dtype)}

    def entry(*refs):
        family.scoped(
            functools.partial(body, *refs),
            plgpu.SMEM((consumers, block_q, width), query.dtype),
            plgpu.SMEM((consumers, block_q, value_width), dout.dtype),
            plgpu.SMEM((consumers, block_q, width), jnp.float32),
            plgpu.SMEM((consumers, block_q), jnp.float32),
            plgpu.Barrier(num_barriers=consumers), plgpu.Barrier(num_barriers=consumers),
            streams=family.streams(blocks, program.stages, consumers))

    sums = jax.ShapeDtypeStruct((batch, queries, heads, width), jnp.float32)
    return family.kernel(
        entry, out_type=(sums, sums, jax.ShapeDtypeStruct((4, batch, heads, queries), jnp.float32)),
        grid=(heads, queries // pair, batch), grid_names=("heads", "q_tiles", "batch"), consumers=consumers,
        interpret=walk.interpret)(query, key, value, dout, peak, prior, anchor, delta, key_index, *leaves)


def _dkv_step(walk: _Backward, resident: tuple, rows_of, buffers, coords_of, whole, add_dq, index, slots,
              carry, accumulators):
    """A dK/dV consumer's query tile `index`: its scores and score cotangents from the query and output
    cotangent slots (`slots`) and its resident keys and values (`resident`) under the tile's rows'
    statistics (`rows_of(first query)`), its coordinates (`coords_of(first query)`) and whether it sees the
    keys whole (`whole(index)`), dV's and dK's products issued into `accumulators` in place, each part staged
    in `buffers` as its transposed product's left operand."""
    k_smem, v_smem = resident
    family, tiles = walk.program.family, walk.program.tiles
    shape = (tiles.block_q, tiles.block_k)
    dk_acc, dv_acc = accumulators
    q_slot, do_slot = slots["queries"], slots["douts"]
    q_first = index * tiles.block_q
    peak, denominator, centre, rest, chosen = rows_of(q_first)
    s = family.product(q_slot, k_smem.transpose((1, 0)), shape)
    dp = family.product(do_slot, v_smem.transpose((1, 0)), shape)

    def scores(*, masked: bool):
        # A tile every query sees whole (`masked` False) takes no mask (`_dq_step`'s).
        exponentials, weights, cotangents, _, _ = _scores(walk, coords_of(q_first), masked, s, dp, peak,
                                                          denominator, centre, rest, chosen)
        return exponentials, weights, cotangents

    # The tiles a query block sees whole unmasked, the rest masked, chosen a tile at a time in one walk
    # (`_dq_step`).
    exponentials, weights, cotangents = lax.cond(whole(index), functools.partial(scores, masked=False),
                                                 functools.partial(scores, masked=True))
    narrowed = algebra.value_weights(exponentials, weights, do_slot.dtype)
    parts = algebra.cotangent_parts(cotangents * walk.scale, q_slot.dtype, accurate=walk.accurate)
    # Each staged part's transpose the left operand (`algebra.cotangent_product`'s `aᵀ @ b`).
    dv_acc = family.staged_products(dv_acc, ((narrowed, None),), do_slot, buffers.at[pl.ds(0, 1)],
                                    walk.interpret, transposed=True)
    dk_acc = family.staged_products(dk_acc, parts, q_slot, buffers.at[pl.ds(1, len(parts))], walk.interpret,
                                    transposed=True)
    if add_dq is not None:
        # The tile's dQ, its staged cotangents times the resident keys (`algebra.cotangent_product`'s
        # `a @ b`), added into dQ: the query tile's other key blocks add theirs.
        add_dq(q_first, family.product(buffers.at[1], k_smem, (tiles.block_q, k_smem.shape[-1])))
    # The staged parts' reads done (the walk's settling) before the next tile's overwrite them.
    return carry, (dk_acc, dv_acc)


def _dkv_launch(query, key, value, dout, peak, total, anchor, delta, key_index, spans: _Spans | None, *,
                walk: _Backward) -> tuple[jax.Array, ...]:
    """The dK/dV program under the dQ program's softmax: each consumer's `block_k` keys of a query head,
    resident, walked by the query tiles that see them, streamed with their output cotangents through a
    ring each; a tile's weights and score cotangents (the algebra's, rows the queries) staged in shared
    memory as the transposed products' left operands (`algebra.value_operands`, `cotangent_product`). Returns
    the query head's dK and dV `[B, Kp, H, D]` fp32; `fused`, dQ `[B, Qp, H, D]` fp32 before them, each tile's
    staged cotangents times the resident keys added into it (`Family.added`)."""
    batch, queries, heads, width = query.shape
    keys_padded, kv_heads, value_width = key.shape[1], key.shape[2], value.shape[-1]
    program, consumers = walk.program, walk.consumers
    family = program.family
    block_q, block_k = program.tiles.block_q, program.tiles.block_k
    pair = consumers * block_k
    q_steps = queries // block_q
    stages = min(program.tiles.stages, q_steps)
    leaves, treedef = jax.tree.flatten(spans)

    def body(*refs, streams):
        (q_ref, k_ref, v_ref, do_ref, peak_ref, total_ref, anchor_ref, delta_ref, index_ref), refs = (
            refs[:9], refs[9:])
        spans_of, refs = (None if spans is None else treedef.unflatten(refs[:len(leaves)]),
                          refs[len(leaves):])
        dq_ref, refs = (refs[0], refs[1:]) if walk.fused else (None, refs)
        dk_ref, dv_ref, k_smem, v_smem, staged, out_smem, k_loaded, v_loaded = refs
        row, head = lax.axis_index("batch"), lax.axis_index("heads")

        def add_dq(dq_ref, index, q_first, dq):
            # A tile's dQ through the consumer's output stage, free until its dK and dV are stored;
            # interpreted, into a copy of the consumer's own (`Family.added`).
            view = dq_ref.at[index if walk.interpret else 0, row, pl.ds(q_first, block_q), head]
            family.added(dq, view, walk.interpret, smem=out_smem.at[index, pl.ds(0, block_q)])

        views = _views_of({"queries": q_ref, "douts": do_ref}, row, head)
        streams = family.opened(streams, views)
        pair_first = lax.axis_index("k_tiles") * pair
        streamed = _streamed([program.readers(pair_first + index * block_k, q_steps, spans_of, row)[:2]
                              for index in range(consumers)])

        def rows_of(q_first):
            return tuple(_rows(family, ref, row, head, q_first, block_q)
                         for ref in (peak_ref, total_ref, anchor_ref, delta_ref, index_ref))

        def consumer(index):
            k_first = pair_first + index * block_k
            keys_view = (row, pl.ds(k_first, block_k), lax.div(head, heads // kv_heads))
            family.loaded(((k_ref.at[keys_view], k_smem.at[index], k_loaded.at[index]),
                           (v_ref.at[keys_view], v_smem.at[index], v_loaded.at[index])))
            lo, hi, whole = program.readers(k_first, q_steps, spans_of, row)
            # The tiles before this block's first reader its partner reads.
            family.passed(streams, streamed[0], lo, first=streamed[0])
            adds = None if dq_ref is None else functools.partial(add_dq, dq_ref, index)
            coords_of = functools.partial(_coords, family, spans_of, row, k_first=k_first,
                                          shape=(block_q, block_k))
            step = functools.partial(_dkv_step, walk, (k_smem.at[index], v_smem.at[index]), rows_of,
                                     staged.at[index], coords_of, whole, adds)
            _, (dk, dv) = family.walk(streams, ((step, lo, hi),), None,
                                      ((block_k, width), (block_k, value_width)), walk.interpret,
                                      first=streamed[0])
            # The tiles past this block's last reader its partner reads.
            family.passed(streams, hi, streamed[1], first=streamed[0])
            out = (row, pl.ds(k_first, block_k), head)
            family.stored((dk, dv), (dk_ref.at[out], dv_ref.at[out]),
                          smem=out_smem.at[index, pl.ds(0, block_k)])

        def producer():
            family.stream(streams, views, *streamed)

        family.roles(consumers, consumer, producer)

    blocks = {"queries": ((block_q, width), query.dtype), "douts": ((block_q, value_width), dout.dtype)}

    def entry(*refs):
        family.scoped(
            functools.partial(body, *refs),
            plgpu.SMEM((consumers, block_k, width), key.dtype),
            plgpu.SMEM((consumers, block_k, value_width), value.dtype),
            # A tile's parts staged for the transposed products: dV's weights, and dK's cotangents in as many
            # parts as the form takes (`algebra.cotangent_parts`, three in the accurate form's bf16).
            plgpu.SMEM((consumers, 4 if walk.accurate else 2, block_q, block_k), query.dtype),
            # dK's and dV's output stage, and each tile's dQ's (`fused`).
            plgpu.SMEM((consumers, max(block_q, block_k), width), jnp.float32),
            plgpu.Barrier(num_barriers=consumers), plgpu.Barrier(num_barriers=consumers),
            streams=family.streams(blocks, stages, consumers))

    gradient = jax.ShapeDtypeStruct((batch, keys_padded, heads, width), jnp.float32)
    launch = family.kernel(
        entry, out_type=(gradient, gradient),
        grid=(heads, keys_padded // pair, batch), grid_names=("heads", "k_tiles", "batch"),
        consumers=consumers, interpret=walk.interpret)
    operands = (query, key, value, dout, peak, total, anchor, delta, key_index, *leaves)
    if not walk.fused:
        return launch(*operands)
    # dQ added into zeros in place: one copy, or each consumer's own interpreted (`Family.added`).
    copies = consumers if walk.interpret else 1
    dq = jax.new_ref(jnp.zeros((copies, batch, queries, heads, width), jnp.float32))
    dk, dv = launch(*operands, dq)
    return jax.freeze(dq).sum(axis=0), dk, dv


def backward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None,
             dout: jax.Array, *, residuals: tuple[jax.Array, jax.Array, jax.Array, Split],
             scale: float | None = None, natural: bool = NATURAL, accurate: bool = ACCURATE,
             tiles: Tiles | None = None, consumers: int | None = None, fused: bool = False,
             family: mosaic.Family | None = None, interpret: bool = False) -> Gradients:
    """The template's backward on this mainloop (`gradients`) from a training forward's `residuals` (its
    maxima, denominators, anchor keys and split sums): D about each query's anchor from the split sums
    (`algebra.split_delta`), a dQ launch and a dK/dV launch, and `dq_finish`'s refinement between them, in
    the form `accurate` names, by `family`'s bundle (the device's, `family_of`) at `tiles` (its default,
    `BACKWARD_DEFAULTS`) and `consumers` a program (the form's, at most the family's); with `fused`, the fast
    form's one launch (`_launch_backward`), a schedule the chooser ranks beside the two launches. Every call
    form of bf16 operands, no hooks."""
    family = family_of(query.dtype) if family is None else family
    tiles = BACKWARD_DEFAULTS[family.name] if tiles is None else tiles
    if fused and accurate:
        raise NotImplementedError("the fused Mosaic backward is the fast form's: the accurate form's dK/dV "
                                  "takes the D the dQ program refines")
    if reason := refused(query.dtype, query.shape[3], value.shape[3], hooks, backward=True,
                         tile=(tiles.block_q, tiles.block_k)):
        raise NotImplementedError(reason)
    count = min(BACKWARD_CONSUMERS[accurate], family.consumers) if consumers is None else consumers
    factor = query.shape[3] ** -0.5 if scale is None else scale
    spec = algebra.Forward(spans.form, spans.keys, ids=spans.query_ids is not None, scale=factor,
                           natural=natural)
    launch = functools.partial(_launch_backward, spec=spec, accurate=accurate, tiles=tiles, consumers=count,
                               fused=fused, family=family, interpret=interpret)
    block = max(count * tiles.block_q, count * tiles.block_k)
    return gradients(launch, query, key, value, spans, hooks, dout, residuals, block=block, scale=factor)


def _launch_backward(query, key, value, dout, peak0, total0, initial_term, maximum, key_index, anchor_value,
                     shift, others, own, others_total, forward_total, start, end, query_ids, key_ids,
                     score_params, *, score_mod, score_kinds, score_dtypes, scale: float,
                     spec: algebra.Forward, accurate: bool, tiles: Tiles, consumers: int, fused: bool,
                     family: mosaic.Family, interpret: bool):
    """`gradients`' launch on this mainloop: D from the forward's split sums, each query's anchor key as the
    first pass chose it. Two launches: the dQ program's sums, their finish (`algebra.dq_finish`) and the
    dK/dV program under its refined D, whose cotangents sum to 0 where the first pass's do not (a small
    weight on a large value, bf16: dK's worst row 2.5e1 at the first pass's D, where check 1 allows
    1.55e-2), the accurate form's always. The fast form's one launch (`fused`): the dK/dV program adding each
    tile's dQ, under the first pass's D and the forward's denominator. Neither is the faster everywhere: on
    an H100 at llama-d128 bf16 2 x 4096, 32 heads, the one launch takes 2.53 ms causal and 4.87 full, two
    launches 2.67 and 4.35 (the one launch's dQ reductions, each consumer's 64 keys reducing its own tile into
    global memory, twice FlashAttention-3's traffic, cost most where every key block reads every query), and
    the one launch holds 384 MB of temporaries against 258 (its dQ in fp32), so the chooser ranks them."""
    del score_params, score_mod, score_kinds, score_dtypes, scale
    width, value_width = query.shape[-1], value.shape[-1]
    _, state_total = algebra.off_anchor_start(peak0, total0, maximum, value_width, shift)
    cotangent = jnp.swapaxes(dout, 1, 2).astype(jnp.float32)
    anchor, delta, chosen = algebra.split_delta(others, own, others_total, forward_total, state_total,
                                                jnp.swapaxes(anchor_value, 1, 2), cotangent, key_index, shift)
    # The family's swizzled tiles hold whole rows of `COLUMNS` 16-bit entries.
    query, key, value, dout = (jnp.pad(array, ((0, 0),) * 3 + ((0, -array.shape[3] % COLUMNS),))
                               for array in (query, key, value, dout))
    steps = key.shape[1] // tiles.block_k
    program = _Program(family, tiles, steps, min(tiles.stages, steps), spec.keys, spec.form == "causal")
    walk = _Backward(program, spec, accurate, consumers, interpret, fused)
    padded = PaddedCall(query, key, value, start, end, query_ids, key_ids, (), (), query.shape[1],
                        spec.keys)
    spans = _spans_of(spec, padded, tiles.block_q, tiles.block_k)
    if walk.fused:
        dq, dk, dv = _dkv_launch(query, key, value, dout, maximum, forward_total, anchor, delta, chosen,
                                 spans, walk=walk)
        mass = algebra.split_mass(own, others_total, forward_total, shift)
        return (dq[..., :width], delta, anchor, maximum, forward_total, mass, chosen, dk[..., :width],
                dv[..., :value_width])
    dq_sum, keyed, rows = _dq_launch(query, key, value, dout, maximum, forward_total, anchor, delta, chosen,
                                     spans, walk=walk)
    sums = algebra.DqSums(jnp.swapaxes(dq_sum, 1, 2), rows[0], rows[1], (), (), jnp.swapaxes(keyed, 1, 2),
                          rows[2], rows[3])
    dq, refined, denominator, mass, _ = algebra.dq_finish(sums, forward_total, state_total, peak0,
                                                          (total0, initial_term), maximum, anchor, delta,
                                                          accurate=accurate)
    dk, dv = _dkv_launch(query, key, value, dout, maximum, denominator, anchor, refined, chosen, spans,
                         walk=walk)
    return (jnp.swapaxes(dq, 1, 2)[..., :width], refined, anchor, maximum, denominator, mass, chosen,
            dk[..., :width], dv[..., :value_width])
