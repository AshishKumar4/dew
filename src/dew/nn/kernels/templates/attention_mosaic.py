"""The attention template's Mosaic GPU mainloop (design §6, one mainloop per overlap structure): the order of
an instruction family's bundle's steps (`mosaic.Family`: its roles, operand streams, products and layouts)
around the template's algebra (`attention_algebra`), FlashAttention-3's structure on the wgmma family
(`mosaic_sm90.WGMMA`) and FlashAttention-2's on mma.sync (`mosaic_sm80.Mma`, one consumer a program). It holds
no instruction, layout or stream barrier of its own: only the overlap structure, the query block's load and
the consumers' schedule.

One program a query head and the family's consumers' query blocks (two warpgroups on wgmma, each a block of
`Tiles.block_q` rows), beside the family's loads streaming the key and value tiles (`Family.stream`,
`Family.walk`), every slot released by every consumer. Two consumers take turns on the tensor cores (a
schedule barrier between them), so one's softmax runs under the other's products. A step issues S = Q·Kᵀ,
releases the key slot, and hands S to the algebra as a tile (`algebra.Keys`) whose values are the value slot
and whose product accumulates the narrowed weights from registers into the running sums in place; while it
runs the consumer waits for its next key tile. A causal block masks only the tiles past its diagonal's start,
and a consumer whose diagonal ends before its partner's keeps releasing the slots its partner still reads.

The forward over full and causal calls of 16-bit operands, the base-2 fold and the natural form without
headroom (fp16 values, which cannot overflow it), no hooks."""

import dataclasses
import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention_algebra as algebra, mosaic, mosaic_sm80, mosaic_sm90
from dew.nn.kernels.templates.attention import (
    ACCURATE,
    NATURAL,
    Gradients,
    Hooks,
    KeySpans,
    Split,
    gradients,
    padded_call,
)

COLUMNS = 64
"""The head width a family's swizzled tiles hold whole rows of, in 16-bit entries (128 bytes)."""


def family_of(dtype) -> mosaic.Family:
    """The bundle of the device's instruction family (`generation.family`) for operands of `dtype`."""
    name = generation.family(generation.device_generation())
    if name == "wgmma":
        return mosaic_sm90.WGMMA
    if name == "mma.sync":
        return mosaic_sm80.Mma(jnp.dtype(dtype))
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


DEFAULTS = {"wgmma": Tiles(64, 128, 2), "mma.sync": Tiles(64, 64, 3)}
"""The forward's tiles by family: wgmma's FlashAttention-3's at width 128, a warpgroup's 64 queries over 128
keys in two stages, the fastest of 64 or 128 queries over 64 or 128 keys in two to four stages on an H100
(llama-d128 causal bf16, 2 x 4096: 0.53 ms; 128 queries a consumer spill, 2.9 to 4.6 ms); mma.sync's a
warpgroup's 64 queries over 64 keys in three stages, as FlashAttention-2's on an A100."""


def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, natural: bool = NATURAL, tiles: Tiles | None = None,
            family: mosaic.Family | None = None, interpret: bool = False) -> jax.Array:
    """The template's forward on this mainloop: `query` `[B, Q, H, D]` over `key` and `value` `[B, K, Hk, D]`
    and the keys `spans` leaves each query (full or causal), as `dew.nn.kernels.templates.attention.reference`
    computes it, by `family`'s bundle (the device's, `family_of`) at `tiles` (its default, `DEFAULTS`).
    `natural` keeps the logits natural (`algebra.Forward.base2`); `interpret` runs it on the CPU (Mosaic
    GPU's interpret mode)."""
    family = family_of(query.dtype) if family is None else family
    tiles = DEFAULTS[family.name] if tiles is None else tiles
    if hooks is not None and hooks != Hooks():
        raise NotImplementedError("the Mosaic mainloop takes no hooks yet")
    if spans.form not in ("full", "causal"):
        raise NotImplementedError(f"the Mosaic mainloop takes full and causal calls, not {spans.form!r} ones")
    if jnp.dtype(query.dtype) not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
        raise NotImplementedError(f"the Mosaic mainloop takes 16-bit operands, not {jnp.dtype(query.dtype)}")
    width, value_width = query.shape[3], value.shape[3]
    if width != value_width:
        raise NotImplementedError("the Mosaic mainloop's output reuses the query's stage: equal head widths")
    block = max(family.consumers * tiles.block_q, tiles.block_k)
    call = padded_call(query, key, value, spans, Hooks(), block)
    spec = algebra.Forward(spans.form, spans.keys, ids=False, scale=width ** -0.5 if scale is None else scale,
                           natural=natural)
    if spec.natural and algebra.headroom(value.dtype):
        raise NotImplementedError("the Mosaic mainloop's natural form takes no headroom guard yet")
    # The family's swizzled tiles hold whole rows of `COLUMNS` 16-bit entries.
    query, key, value = (jnp.pad(array, ((0, 0),) * 3 + ((0, -array.shape[3] % COLUMNS),))
                         for array in (call.query, call.key, call.value))
    out = _launch(query, key, value, spec=spec, tiles=tiles, family=family, interpret=interpret)
    return out[:, :call.queries, :, :value_width]


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
        """The key tiles a query block from `first` reads: every one, or those up to its diagonal."""
        if not self.causal:
            return jnp.int32(self.steps)
        return jnp.minimum(pl.cdiv(first + self.tiles.block_q, self.tiles.block_k), self.steps)

    def whole(self, first, walked):
        """The tiles every query of the block from `first` sees whole: those wholly within the call's keys
        (the padded call's last tiles may all be past them), and below its diagonal."""
        whole = jnp.int32(self.keys // self.tiles.block_k)
        if self.causal:
            whole = jnp.minimum(lax.div(first, self.tiles.block_k), whole)
        return jnp.minimum(jnp.maximum(whole, 0), walked)


def _consumer(program: _Program, spec: algebra.Forward, consumer, q_view, out_view, q_smem, q_loaded, streams,
              turn, schedule, most, first, *, interpret: bool):
    """A consumer's query block: its walk of the key tiles through the family's streams (`algebra.plain_step`
    a step), then its output stored through its query stage."""
    family, tiles = program.family, program.tiles
    shape = (tiles.block_q, tiles.block_k)
    walked = program.walked(first)
    whole = program.whole(first, walked)
    family.loaded(((q_view, q_smem, q_loaded),))
    q = family.held(q_smem)
    q_pos, columns = first + family.iota(shape, 0), family.iota(shape, 1)
    width = q_smem.shape[-1]
    # The weighted values' running sums live in the family's accumulator (`Family.walk`); the walk carries
    # the maxima and denominators, the state's `acc` an empty place the algebra passes through
    # (`algebra.Keys.accumulate`), a constant no loop may carry.
    rows = (family.full((tiles.block_q,), -jnp.inf), family.zeros((tiles.block_q,)))
    pl.when(consumer == 1)(turn)

    def step(index, slots, rows, accumulators):
        s = family.product(q, slots["keys"].transpose((1, 0)), shape, turn)
        slots.release("keys")
        coords = algebra.Coords(q_pos, index * tiles.block_k + columns)
        # A tile every query sees whole takes no mask: one walk, the choice a tile's (a walk of each, these
        # rings' waits and arrivals in two loops, faulted on an H100 under jax 0.11.2's lowering).
        logits = lax.cond(index < whole, lambda: algebra.modify(spec, s, coords, masked=False),
                          lambda: algebra.modify(spec, s, coords, masked=True))

        def load():
            # The value slot, once the partner's products are issued: its turn on the tensor cores is over
            # when this one's come.
            turn(wait=False)
            slot = slots["values"]
            turn(arrive=False)
            return slot

        def dot(lhs, rhs):
            return family.registers_product(lhs, rhs, interpret)

        held = list(accumulators)

        def accumulate(scale, p, value):
            held[0] = family.rescaled_products(held[0], scale, ((p, None),), value, interpret)

        state = algebra.Partial(jnp.zeros((), jnp.float32), *rows)
        state = algebra.plain_step(spec, state, algebra.Keys(coords, logits, load, dot, accumulate))
        return (state.m, state.total), tuple(held)

    rows, (acc,) = family.walk(streams, ((step, 0, walked),), rows, ((tiles.block_q, width),), interpret)
    state = algebra.Partial(acc, *rows)
    # The slots a consumer whose diagonal ends first leaves its partner, the turns it would have taken.
    family.passed(streams, walked, most, between=lambda: (turn(), turn()))
    pl.when(consumer == 0)(turn)
    out, _, _ = algebra.finish(spec, state, initial=False)
    family.stored((out,), (out_view,), smem=q_smem)


def _launch(query, key, value, *, spec: algebra.Forward, tiles: Tiles, family: mosaic.Family,
            interpret: bool):
    batch, queries, heads, width = query.shape
    keys_count, kv_heads = key.shape[1], key.shape[2]
    group, consumers = heads // kv_heads, family.consumers
    block_q, block_k = tiles.block_q, tiles.block_k
    pair = consumers * block_q
    steps = keys_count // block_k
    # No more stages than key tiles: a slot past them would load past the keys.
    program = _Program(family, tiles, steps, min(tiles.stages, steps), spec.keys, spec.form == "causal")

    def body(q_ref, k_ref, v_ref, out_ref, q_smem, q_loaded, schedule, *, streams):
        row, head = lax.axis_index("batch"), lax.axis_index("heads")
        views = _views_of({"keys": k_ref, "values": v_ref}, row, lax.div(head, group), block_k)
        streams = family.opened(streams, views)
        pair_first = program.query_tile(queries // pair) * pair
        most = program.walked(pair_first + pair - block_q)

        def turn(*, arrive: bool = True, wait: bool = True):
            # The consumers' turns on the tensor cores, where a program has two (a schedule barrier).
            if consumers > 1 and arrive:
                plgpu.barrier_arrive(schedule)
            if consumers > 1 and wait:
                plgpu.barrier_wait(schedule)

        def consumer(index):
            first = pair_first + index * block_q
            rows = (row, pl.ds(first, block_q), head)
            _consumer(program, spec, index, q_ref.at[rows], out_ref.at[rows], q_smem.at[index],
                      q_loaded.at[index], streams, turn, schedule, most, first, interpret=interpret)

        def producer():
            family.stream(streams, views, 0, most)

        family.roles(consumers, consumer, producer)

    blocks = {"keys": ((block_k, width), key.dtype), "values": ((block_k, value.shape[-1]), value.dtype)}

    def entry(q_ref, k_ref, v_ref, out_ref):
        family.scoped(
            functools.partial(body, q_ref, k_ref, v_ref, out_ref),
            plgpu.SMEM((consumers, block_q, width), query.dtype),
            plgpu.Barrier(num_barriers=consumers), plgpu.Barrier(num_arrivals=consumers),
            streams=family.streams(blocks, program.stages, consumers))

    return family.kernel(entry, out_type=jax.ShapeDtypeStruct((batch, queries, heads, value.shape[-1]),
                                                              query.dtype),
                         grid=(heads, queries // pair, batch), grid_names=("heads", "q_tiles", "batch"),
                         consumers=consumers, interpret=interpret)(query, key, value)


BACKWARD_DEFAULTS = {"wgmma": Tiles(64, 64, 2), "mma.sync": Tiles(64, 64, 2)}
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


def _views_of(refs: dict, row, head, block: int) -> dict:
    """The streams' views by name (`Family.opened`, `Family.stream`): each `[B, S, H, D]` operand's `[S, D]`
    at batch row `row` and head `head`, its steps `block` rows of it."""
    return {name: (ref.at[row, :, head], block) for name, ref in refs.items()}


class _Tile(NamedTuple):
    """A backward tile's place: its first query and key, its shape, whether it takes a mask (`masked`), and
    the call's form (`causal`, its keys)."""

    q_first: jax.Array
    k_first: jax.Array
    shape: tuple[int, int]
    masked: bool
    causal: bool
    keys: int


def _scores(family: mosaic.Family, tile: _Tile, s, dp, peak, total, anchor, delta, chosen, *,
            how: algebra.Recompute, scale: float):
    """A backward tile's weights unnormalized and over `total`, its score cotangents (`algebra.grad_scores`),
    its entries' derivatives and its mask from its products `s` and `dp` and its queries' statistics."""
    k_pos = tile.k_first + family.iota(tile.shape, 1)
    seen = None
    if tile.masked:
        q_pos = tile.q_first + family.iota(tile.shape, 0)
        seen = (k_pos <= q_pos if tile.causal else k_pos >= 0) & (k_pos < tile.keys)
    logits, slopes = algebra.tile_logits(None, s, seen, None, None, (),
                                         scale=scale if how.factor is None else 1.0, slopes=True)
    exponentials, weights = algebra.normalized(logits, peak, total, how)
    at_anchor = k_pos == algebra.expand(chosen, k_pos)
    cotangents = algebra.grad_scores(exponentials, dp, anchor, delta, seen, at_anchor, algebra.inverse(total))
    return exponentials, weights, cotangents, slopes, seen


@dataclasses.dataclass(frozen=True)
class _Backward:
    """A backward launch's static choices: its program's geometry, how it recomputes the weights, the
    logits' scale, its form, its consumers and whether it is interpreted."""

    program: _Program
    how: algebra.Recompute
    scale: float
    accurate: bool
    consumers: int
    interpret: bool


def _dq_step(walk: _Backward, operands: tuple, stats: tuple, first, whole, index, slots, rows, accumulators):
    """A dQ consumer's key tile `index`: its scores and score cotangents from the key and value slots
    (`slots`) and its queries and output cotangents (`operands`) under its rows' statistics (`stats`), the
    row sums carried (`rows`) and dQ's and the keyed sums' products issued into `accumulators` in place."""
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
        tile = _Tile(first, index * tiles.block_k, shape, masked, walk.program.causal, walk.program.keys)
        exponentials, _, cotangents, _, _ = _scores(family, tile, s, dp, peak, prior, anchor, delta, chosen,
                                                    how=walk.how, scale=walk.scale)
        return exponentials, cotangents

    # One walk, its tiles' masks chosen a tile at a time: a walk of two loops into one warpgroup accumulator,
    # the first empty, faults on an H100 under jax 0.11.2's Mosaic GPU lowering.
    exponentials, cotangents = lax.cond(index < whole, functools.partial(scores, masked=False),
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


def _dq_launch(query, key, value, dout, peak, prior, anchor, delta, key_index, *, walk: _Backward):
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

    def body(q_ref, k_ref, v_ref, do_ref, peak_ref, prior_ref, anchor_ref, delta_ref, index_ref, dq_ref,
             keyed_ref, rows_ref, q_smem, do_smem, out_smem, row_smem, q_loaded, do_loaded, *, streams):
        row, head = lax.axis_index("batch"), lax.axis_index("heads")
        views = _views_of({"keys": k_ref, "values": v_ref}, row, lax.div(head, heads // kv_heads), block_k)
        streams = family.opened(streams, views)
        pair_first = program.query_tile(queries // pair) * pair
        most = program.walked(pair_first + pair - block_q)

        def consumer(index):
            first = pair_first + index * block_q
            rows = (row, pl.ds(first, block_q), head)
            family.loaded(((q_ref.at[rows], q_smem.at[index], q_loaded.at[index]),
                           (do_ref.at[rows], do_smem.at[index], do_loaded.at[index])))
            stats = tuple(_rows(family, ref, row, head, first, block_q)
                          for ref in (peak_ref, prior_ref, anchor_ref, delta_ref, index_ref))
            walked = program.walked(first)
            nothing = family.zeros((block_q,))
            operands = family.held(q_smem.at[index]), family.held(do_smem.at[index])
            step = functools.partial(_dq_step, walk, operands, stats, first, program.whole(first, walked))
            sums, (dq, keyed) = family.walk(
                streams, ((step, 0, walked),), (nothing, nothing, nothing, nothing + algebra.NO_SHIFT),
                ((block_q, width), (block_q, width)), walk.interpret)
            # The tiles a block whose diagonal ends first leaves its partner.
            family.passed(streams, walked, most)
            # The keyed sums the accurate form's alone (`algebra.dq_terms`).
            written = ((dq, dq_ref.at[rows]), (keyed, keyed_ref.at[rows]))[:2 if walk.accurate else 1]
            family.stored(*zip(*written, strict=True), smem=out_smem.at[index])
            # Each row sum in a row's layout whatever its own (the keyed mass of a call without a hook is
            # a constant).
            family.stored(tuple(family.as_row(part) for part in sums),
                          tuple(rows_ref.at[place, row, head, pl.ds(first, block_q)] for place in range(4)),
                          smem=row_smem.at[index])

        def producer():
            family.stream(streams, views, 0, most)

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
        interpret=walk.interpret)(query, key, value, dout, peak, prior, anchor, delta, key_index)


def _dkv_step(walk: _Backward, resident: tuple, rows_of, buffers, k_first, whole, index, slots, carry,
              accumulators):
    """A dK/dV consumer's query tile `index`: its scores and score cotangents from the query and output
    cotangent slots (`slots`) and its resident keys and values (`resident`) under the tile's rows'
    statistics (`rows_of(first query)`), dV's and dK's products issued into `accumulators` in place, each
    part staged in `buffers` as its transposed product's left operand."""
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
        tile = _Tile(q_first, k_first, shape, masked, walk.program.causal, walk.program.keys)
        exponentials, weights, cotangents, _, _ = _scores(family, tile, s, dp, peak, denominator, centre,
                                                          rest, chosen, how=walk.how, scale=walk.scale)
        return exponentials, weights, cotangents

    # The block's diagonal tiles masked, the rest whole, chosen a tile at a time in one walk (`_dq_step`).
    exponentials, weights, cotangents = lax.cond(index < whole, functools.partial(scores, masked=True),
                                                 functools.partial(scores, masked=False))
    narrowed = algebra.value_weights(exponentials, weights, do_slot.dtype)
    parts = algebra.cotangent_parts(cotangents * walk.scale, q_slot.dtype, accurate=walk.accurate)
    # Each staged part's transpose the left operand (`algebra.cotangent_product`'s `aᵀ @ b`).
    dv_acc = family.staged_products(dv_acc, ((narrowed, None),), do_slot, buffers.at[pl.ds(0, 1)],
                                    walk.interpret, transposed=True)
    dk_acc = family.staged_products(dk_acc, parts, q_slot, buffers.at[pl.ds(1, len(parts))], walk.interpret,
                                    transposed=True)
    # The staged parts' reads done (the walk's settling) before the next tile's overwrite them.
    return carry, (dk_acc, dv_acc)


def _dkv_launch(query, key, value, dout, peak, total, anchor, delta, key_index, *, walk: _Backward):
    """The dK/dV program under the dQ program's softmax: each consumer's `block_k` keys of a query head,
    resident, walked by the query tiles that see them, streamed with their output cotangents through a
    ring each; a tile's weights and score cotangents (the algebra's, rows the queries) staged in shared
    memory as the transposed products' left operands (`algebra.value_operands`, `cotangent_product`). Returns
    the query head's dK and dV `[B, Kp, H, D]` fp32."""
    batch, queries, heads, width = query.shape
    keys_padded, kv_heads, value_width = key.shape[1], key.shape[2], value.shape[-1]
    program, consumers = walk.program, walk.consumers
    family = program.family
    block_q, block_k = program.tiles.block_q, program.tiles.block_k
    pair = consumers * block_k
    q_steps = queries // block_q
    stages = min(program.tiles.stages, q_steps)

    def first_tile(k_first):
        # The first query tile that sees a key block from `k_first`: every one, or those from its diagonal,
        # at most the last (a causal call of more keys than queries leaves its last keys unseen).
        return jnp.minimum(lax.div(k_first, block_q), q_steps) if program.causal else jnp.int32(0)

    def body(q_ref, k_ref, v_ref, do_ref, peak_ref, total_ref, anchor_ref, delta_ref, index_ref, dk_ref,
             dv_ref, k_smem, v_smem, staged, out_smem, k_loaded, v_loaded, *, streams):
        row, head = lax.axis_index("batch"), lax.axis_index("heads")
        views = _views_of({"queries": q_ref, "douts": do_ref}, row, head, block_q)
        streams = family.opened(streams, views)
        pair_first = lax.axis_index("k_tiles") * pair
        least = first_tile(pair_first)

        def rows_of(q_first):
            return tuple(_rows(family, ref, row, head, q_first, block_q)
                         for ref in (peak_ref, total_ref, anchor_ref, delta_ref, index_ref))

        def consumer(index):
            k_first = pair_first + index * block_k
            keys_view = (row, pl.ds(k_first, block_k), lax.div(head, heads // kv_heads))
            family.loaded(((k_ref.at[keys_view], k_smem.at[index], k_loaded.at[index]),
                           (v_ref.at[keys_view], v_smem.at[index], v_loaded.at[index])))
            mine = first_tile(k_first)
            # The query tiles from which every query sees the block whole: past its diagonal, and none if
            # it holds padding.
            whole = (jnp.minimum(pl.cdiv(k_first + block_k - 1, block_q), q_steps) if program.causal
                     else mine)
            whole = jnp.where(k_first + block_k > program.keys, q_steps, jnp.maximum(whole, mine))
            # The tiles before this block's diagonal its partner reads.
            family.passed(streams, least, mine, first=least)
            step = functools.partial(_dkv_step, walk, (k_smem.at[index], v_smem.at[index]), rows_of,
                                     staged.at[index], k_first, whole)
            _, (dk, dv) = family.walk(streams, ((step, mine, q_steps),), None,
                                      ((block_k, width), (block_k, value_width)), walk.interpret, first=least)
            out = (row, pl.ds(k_first, block_k), head)
            family.stored((dk, dv), (dk_ref.at[out], dv_ref.at[out]), smem=out_smem.at[index])

        def producer():
            family.stream(streams, views, least, q_steps)

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
            plgpu.SMEM((consumers, block_k, width), jnp.float32),
            plgpu.Barrier(num_barriers=consumers), plgpu.Barrier(num_barriers=consumers),
            streams=family.streams(blocks, stages, consumers))

    gradient = jax.ShapeDtypeStruct((batch, keys_padded, heads, width), jnp.float32)
    return family.kernel(
        entry, out_type=(gradient, gradient),
        grid=(heads, keys_padded // pair, batch), grid_names=("heads", "k_tiles", "batch"),
        consumers=consumers, interpret=walk.interpret)(query, key, value, dout, peak, total, anchor, delta,
                                                       key_index)


def backward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None,
             dout: jax.Array, *, residuals: tuple[jax.Array, jax.Array, jax.Array, Split],
             scale: float | None = None, natural: bool = NATURAL, accurate: bool = ACCURATE,
             tiles: Tiles | None = None, consumers: int | None = None, family: mosaic.Family | None = None,
             interpret: bool = False) -> Gradients:
    """The template's backward on this mainloop (`gradients`) from a training forward's `residuals` (its
    maxima, denominators, anchor keys and split sums): D about each query's anchor from the split sums
    (`algebra.split_delta`), a dQ launch and a dK/dV launch, and `dq_finish`'s refinement between them, in
    the form `accurate` names, by `family`'s bundle (the device's, `family_of`) at `tiles` (its default,
    `BACKWARD_DEFAULTS`) and `consumers` a program (the form's, at most the family's). Full and causal calls
    of bf16 operands, no hooks."""
    family = family_of(query.dtype) if family is None else family
    tiles = BACKWARD_DEFAULTS[family.name] if tiles is None else tiles
    if hooks is not None and hooks != Hooks():
        raise NotImplementedError("the Mosaic backward takes no hooks yet")
    if spans.form not in ("full", "causal"):
        raise NotImplementedError(f"the Mosaic backward takes full and causal calls, not {spans.form!r} ones")
    if jnp.dtype(query.dtype) != jnp.dtype(jnp.bfloat16):
        raise NotImplementedError(f"the Mosaic backward takes bf16 operands, not {jnp.dtype(query.dtype)}")
    if query.shape[3] != value.shape[3]:
        raise NotImplementedError("the Mosaic backward stages dK and dV through one buffer: equal head "
                                  "widths, as the forward takes")
    count = min(BACKWARD_CONSUMERS[accurate], family.consumers) if consumers is None else consumers
    factor = query.shape[3] ** -0.5 if scale is None else scale
    how = algebra.recompute(algebra.Forward("full", 0, ids=False, scale=factor, natural=natural))
    launch = functools.partial(_launch_backward, how=how, causal=spans.form == "causal",
                               keys_count=spans.keys, accurate=accurate, tiles=tiles, consumers=count,
                               family=family, interpret=interpret)
    block = max(count * tiles.block_q, count * tiles.block_k)
    return gradients(launch, query, key, value, spans, hooks, dout, residuals, block=block, scale=factor)


def _launch_backward(query, key, value, dout, peak0, total0, initial_term, maximum, key_index, anchor_value,
                     shift, others, own, others_total, forward_total, start, end, query_ids, key_ids,
                     score_params, *, score_mod, score_kinds, score_dtypes, scale: float,
                     how: algebra.Recompute, causal: bool, keys_count: int, accurate: bool, tiles: Tiles,
                     consumers: int, family: mosaic.Family, interpret: bool):
    """`gradients`' launch on this mainloop: D from the forward's split sums, the dQ program's sums, their
    finish (`algebra.dq_finish`) and the dK/dV program under its refined D, each query's anchor key as the
    first pass chose it."""
    del start, end, query_ids, key_ids, score_params, score_mod, score_kinds, score_dtypes
    width, value_width = query.shape[-1], value.shape[-1]
    _, state_total = algebra.off_anchor_start(peak0, total0, maximum, value_width, shift)
    cotangent = jnp.swapaxes(dout, 1, 2).astype(jnp.float32)
    anchor, delta, chosen = algebra.split_delta(others, own, others_total, forward_total, state_total,
                                                jnp.swapaxes(anchor_value, 1, 2), cotangent, key_index, shift)
    # The family's swizzled tiles hold whole rows of `COLUMNS` 16-bit entries.
    query, key, value, dout = (jnp.pad(array, ((0, 0),) * 3 + ((0, -array.shape[3] % COLUMNS),))
                               for array in (query, key, value, dout))
    steps = key.shape[1] // tiles.block_k
    program = _Program(family, tiles, steps, min(tiles.stages, steps), keys_count, causal)
    walk = _Backward(program, how, scale, accurate, consumers, interpret)
    dq_sum, keyed, rows = _dq_launch(query, key, value, dout, maximum, forward_total, anchor, delta, chosen,
                                     walk=walk)
    sums = algebra.DqSums(jnp.swapaxes(dq_sum, 1, 2), rows[0], rows[1], (), (), jnp.swapaxes(keyed, 1, 2),
                          rows[2], rows[3])
    dq, refined, denominator, mass, _ = algebra.dq_finish(sums, forward_total, state_total, peak0,
                                                          (total0, initial_term), maximum, anchor, delta,
                                                          accurate=accurate)
    dk, dv = _dkv_launch(query, key, value, dout, maximum, denominator, anchor, refined, chosen, walk=walk)
    return (jnp.swapaxes(dq, 1, 2)[..., :width], refined, anchor, maximum, denominator, mass, chosen,
            dk[..., :width], dv[..., :value_width])
