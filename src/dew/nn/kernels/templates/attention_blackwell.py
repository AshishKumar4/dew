"""The attention template's Blackwell mainloop (Mosaic GPU, sm100), the design's first sm100 rung
(docs/design/kernel-templates.md §6), on the tcgen05 bundle (`mosaic_sm100`, the GEMM template's
mainloop on tcgen05 beside it): tcgen05's products with their accumulators in tensor memory, as
FlashAttention-4 and Tokamax's sm100 kernel hold them, and the template's algebra (`attention_algebra`)
between the products.

A program is a head's query block of `RINGS.rows` rows, in three warpgroups (`schedule`). The
issuing one's load warp fills a ring of key stages and one of value stages (the bundle's streams,
`mosaic_sm100.RINGS`: their slots released by the products' commit), and its
product warp issues each step's `S = Q Kᵀ` into tensor memory and, a step behind, `P V` into the output's
accumulator there, so a step's `S` is computed while the last step's weights multiply their values. The
softmax warpgroup reads `S` out of tensor memory, hands it to the algebra (`logits_of`, `peak`, `weights`,
`denominator_after`), hands each step's rescale of the state before it to the correction warpgroup through
shared memory, and writes the narrowed weights into one of two buffers in tensor memory, the other the last
step's, which its `P V` reads meanwhile (FlashAttention-4's), once the `P V` two steps back has read it.
The correction warpgroup, FlashAttention-4's, rescales the output's accumulator in place once the last
`P V` is in it (`rescale`, `CHUNK` columns at a time), which the next `P V` waits for, and runs the
epilogue from the softmax's final rows: the rescale is off the softmax's path, its exponentials beside the
accumulator's loads and stores. A causal call walks the key blocks below its block's diagonal unmasked and
the diagonal's masked.

The forward over full and causal calls of 16-bit operands, no hooks, one CTA a block. FlashAttention-4's
two-CTA products and lazy rescale follow as rungs."""

import dataclasses
import functools
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import generation
from dew.nn.kernels.selection import Refusal
from dew.nn.kernels.templates import (
    attention_algebra as algebra,
    attention_mosaic,
    bodies,
    handoffs,
    mosaic,
    mosaic_sm100,
    resources,
    swizzling,
)
from dew.nn.kernels.templates.attention import NATURAL, Hooks, KeySpans, padded_call, prescaled
from dew.nn.kernels.templates.mosaic_sm100 import ISSUING, RINGS, STORED, TMEM, Accumulator

CHUNK = 32
"""The output accumulator's columns the correction rescales at a time."""
ROLES = (handoffs.Role("softmax"), handoffs.Role("correction"), *ISSUING)
"""A program's roles: the softmax's warpgroup and the correction's, then the issuing one's load and product
warps."""


def schedule(rows: int, block_k: int) -> handoffs.Schedule:
    """A program's roles (`ROLES`) and the hand-offs between them, each's barriers counting the arrivals of
    the role that fills it and of those that give it back (`handoffs.HandOff`): `S`'s accumulator of `rows` x
    `block_k`, which the product warp fills and the softmax gives back once read; the weights' two buffers'
    signals, the softmax's once its tensor-memory stores are in; each `P V`'s, the product warp's commit;
    each step's rescale in two slots, the softmax's; the rescaled accumulator's, the correction's; and the
    softmax's final rows', the softmax's."""
    return handoffs.Schedule(ROLES, (
        handoffs.HandOff("scores", mosaic_sm100.Accumulators(1, rows, block_k), "mma", ("softmax",)),
        handoffs.HandOff("p_full", mosaic_sm100.Signals(count=2), "softmax"),
        handoffs.HandOff("o_done", mosaic_sm100.Signals(count=2), "mma"),
        handoffs.HandOff("scaled", mosaic_sm100.Signals(count=2, tensor_core=False), "softmax"),
        handoffs.HandOff("corrected", mosaic_sm100.Signals(), "correction"),
        handoffs.HandOff("finished", mosaic_sm100.Signals(tensor_core=False), "softmax")))
PART = 16
"""A step's tile's columns the softmax reads and weighs at a time: its chains over a row's whole tile spill
(the natural relaunch's, 7.9 KB a thread at 128 columns), 80 bytes at 32 columns, none at 16 (ptxas for a
B200, tests/test_attention_blackwell.py's compile at widths 64 and 128)."""


@dataclasses.dataclass(frozen=True)
class Tiles:
    """A launch's key tile (`block_k`, a multiple of 64) and the key and value tiles in flight
    (`stages`)."""

    block_k: int = 128
    stages: int = 2

    def __post_init__(self):
        if self.block_k % 64 or self.stages < 2:
            raise ValueError(f"{self}: a key tile of whole 64 rows, two stages or more")


TILES = Tiles()
"""The tiles a call takes where it names none."""


def refused(query: jax.Array, spans: KeySpans, hooks: Hooks | None) -> Refusal | None:
    """Why this rung takes no call of `query`'s operands over `spans` under `hooks`, or None: it takes full
    and causal calls of 16-bit operands with no hook."""
    if hooks is not None and hooks != Hooks():
        return Refusal('hooks', "the sm100 mainloop takes no hooks yet")
    if spans.form not in ("full", "causal"):
        return Refusal('spans', f"the sm100 mainloop takes full and causal calls, not {spans.form!r} ones")
    if jnp.dtype(query.dtype) not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
        return Refusal('dtype', f"the sm100 mainloop takes 16-bit operands, not {jnp.dtype(query.dtype)}")
    return None


def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, natural: bool = NATURAL, tiles: Tiles = TILES,
            interpret: bool = False) -> jax.Array:
    """The template's forward on this mainloop: `query` `[B, Q, H, D]` over `key` and `value`
    `[B, K, Hk, D]` and the keys `spans` leaves each query (full or causal), as
    `dew.nn.kernels.templates.attention.reference` computes it. `natural` keeps the logits natural
    (`algebra.Forward.base2`); `interpret` runs it on the CPU (Mosaic GPU's interpreter)."""
    if reason := refused(query, spans, hooks):
        raise NotImplementedError(reason)
    walked = functools.partial(_walked, query, key, value, spans, scale=scale, natural=natural, tiles=tiles,
                               interpret=interpret)
    if not algebra.headroom(value.dtype):
        return walked()
    # A call some row of which can be scaled (`algebra.may_scale`) takes the template's scaled Mosaic op
    # (`attention_mosaic.MosaicForward.scaled`, its own tcgen05 config): this mainloop's softmax warpgroup
    # holds no second product for a scaled tile's small weights. Every other call, every ordinary input's,
    # takes this launch, bit for bit the scaled walk's there.
    return lax.cond(algebra.may_scale(value, keys_axis=1),
                    lambda: attention_mosaic.forward(query, key, value, spans, scale=scale, natural=natural,
                                                     interpret=interpret, scaled=True),
                    walked)


def _walked(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, *, scale: float | None,
            natural: bool, tiles: Tiles, interpret: bool) -> jax.Array:
    """`forward`'s launch, unscaled, under the fold's guard over whole launches
    (`algebra.walked_launches`)."""
    value_width = value.shape[-1]
    # The query taken by its scale's power of two before its products (`prescaled`), so a logit finite at
    # the call's scale is finite in the product.
    query, scale, power = prescaled(query, scale)
    # Queries and keys padded to whole tiles of both: a program's query block and a walk's key tile.
    call = padded_call(query, key, value, spans, Hooks(), math.lcm(RINGS.rows, tiles.block_k))
    spec = algebra.Forward(spans.form, spans.keys, ids=False, scale=scale, natural=natural, prescale=power)
    # Swizzled tiles hold whole 64-element rows of 16-bit operands.
    call = dataclasses.replace(call, **{name: jnp.pad(array, ((0, 0),) * 3 + ((0, -array.shape[3] % 64),))
                                        for name, array in (("query", call.query), ("key", call.key),
                                                            ("value", call.value))})

    def launch(form: algebra.Forward, live: jax.Array | None, into=None):
        del live, into
        out, maxima = _forward(call.query, call.key, call.value, spec=form, tiles=tiles, interpret=interpret)
        return out, maxima, None

    out = algebra.walked_launches(spec, launch)
    return out[:, :call.queries, :, :value_width]


class _Scratch(NamedTuple):
    """A program's opened scratch: its query block and its load's barrier (`Family.load_barriers`), the key
    and value rings (`mosaic_sm100.CommittedRing`), `S`'s accumulator (`mosaic_sm100.Tensor`: the products
    commit it, the softmax releases it once read), the weights' two buffers in tensor memory (a step's the
    turn's slot of the signal publishing them), the output's accumulator and the signal each `P V` commits
    (`mosaic_sm100.Signal`, a turn a step, two slots: a step's weights wait for the one two back), each
    step's rescale in shared memory (two slots) and the signal the softmax gives it by, the signal the
    correction publishes the rescaled accumulator by, the softmax's final rows in shared memory and their
    signal, and the epilogue's stages, which it writes by item (`ref[...] = `, which no static type of JAX's
    refs declares)."""

    q_smem: mosaic.Ref
    q_loaded: mosaic.Ref
    keys: mosaic.Ring
    values: mosaic.Ring
    scores: mosaic_sm100.Tensor
    p_tmems: tuple[mosaic.Ref, mosaic.Ref]
    p_full: mosaic_sm100.Signal
    acc_tmem: mosaic.Ref
    o_done: mosaic_sm100.Signal
    scales: Any
    scaled: mosaic_sm100.Signal
    corrected: mosaic_sm100.Signal
    rows_smem: Any
    finished: mosaic_sm100.Signal
    o_smem: Any
    l_smem: Any


class _Block(NamedTuple):
    """A program's query block from the grid: its row, head and first query, and the key blocks it walks
    (`steps`) of the call's (`blocks`), each read where a role reads it, inside the role (a value one role's
    branch closes over from the program's body fails Mosaic GPU's interpreter)."""

    causal: bool
    blocks: int
    block_k: int

    @property
    def row(self):
        return lax.axis_index("batch")

    @property
    def head(self):
        return lax.axis_index("heads")

    @property
    def first(self):
        return lax.axis_index("q_blocks") * RINGS.rows

    @property
    def steps(self):
        if not self.causal:
            return self.blocks
        return jnp.minimum(pl.cdiv(self.first + RINGS.rows, self.block_k), self.blocks)


def _load(block: _Block, scratch: _Scratch, k_ref, v_ref, *, group: int, tiles: Tiles) -> None:
    """The load warp: each key block's keys and values into their rings."""
    kv_head = lax.div(block.head, group)

    @pl.loop(0, block.blocks)
    def _(index):
        @pl.when(index < block.steps)
        def _():
            tile = (block.row, pl.ds(index * tiles.block_k, tiles.block_k), kv_head)
            scratch.keys.load(scratch.keys.turn(index), k_ref.at[tile])
            scratch.values.load(scratch.values.turn(index), v_ref.at[tile])

    # Each slot's last releases, observed so its barriers end where they began (`mosaic.drain`).
    mosaic.drain((scratch.keys, scratch.values), block.steps)


def _products(block: _Block, scratch: _Scratch, q_ref) -> None:
    """The product warp: its query block loaded whole (the bundle's `loaded`, beside the load warp's rings),
    then each step's `S = Q Kᵀ` into tensor memory once the softmax has read the last one, and a step behind,
    `P V` into the output's accumulator once its weights are there and the correction has taken the
    accumulator to their maximum."""
    queries = q_ref.at[block.row, pl.ds(block.first, RINGS.rows), block.head]
    RINGS.loaded(((queries, scratch.q_smem, scratch.q_loaded.at[0]),))

    def values(index):
        turn = scratch.values.turn(index)
        v_smem = scratch.values.acquire(turn)
        scratch.p_full.wait(scratch.p_full.turn(index))
        pl.when(index > 0)(lambda: scratch.corrected.wait(scratch.corrected.turn(index - 1)))
        consumed = _released(scratch.values, turn)
        acc = Accumulator(scratch.acc_tmem, (RINGS.rows, int(scratch.acc_tmem.shape[-1])),
                          accumulate=index > 0)

        def product(p_tmem):
            # The weights the softmax wrote to tensor memory (tcgen05's TS form, `Tcgen05Issuer.products`).
            weighed = Accumulator(p_tmem, (RINGS.rows, block.block_k), dtype=v_smem.dtype)
            RINGS.products(acc, weighed, v_smem, released=consumed)

        for slot, p_tmem in enumerate(scratch.p_tmems):
            pl.when(lax.rem(index, 2) == slot)(functools.partial(product, p_tmem))
        scratch.o_done.committed(scratch.o_done.turn(index))

    @pl.loop(0, block.blocks)
    def _(index):
        @pl.when(index < block.steps)
        def _():
            turn = scratch.keys.turn(index)
            k_smem = scratch.keys.acquire(turn)
            scores = scratch.scores.turn(index)
            s_tmem = scratch.scores.acquire(scores)
            RINGS.products(Accumulator(s_tmem, (RINGS.rows, block.block_k)), scratch.q_smem, k_smem.T,
                           accumulate=False, released=_released(scratch.keys, turn))
            scratch.scores.issued(scores)
            pl.when(index > 0)(functools.partial(values, index - 1))

    values(block.steps - 1)
    scratch.scores.drain(block.steps)


def _released(ring: mosaic.Ring, turn: mosaic.Turn):
    """The barrier a product reading `turn`'s slot of `ring` releases it at, by its commit
    (`mosaic_sm100.CommittedRing`: the slot's consumed barrier, tensor-core ordered)."""
    return ring.consumed.at[turn.slot]


def _softmax(block: _Block, scratch: _Scratch, *, spec: algebra.Forward, tiles: Tiles, blocks: int,
             dtype) -> None:
    """The softmax warpgroup: each step's `S` through the algebra, the factor that takes the state before it
    to the step's maximum handed to the correction (`_correction`), and the narrowed weights into tensor
    memory once the last `P V` has read its own; then its rows' maxima and denominators to the correction's
    epilogue."""
    rows, block_k = RINGS.rows, tiles.block_k
    # A step's tile `PART` columns at a time, its places' layout stated: a full call reads neither, and an
    # iota nothing reads takes no inferred layout (2977).
    q_pos = block.first + plgpu.broadcasted_iota(jnp.int32, (rows, PART), 0, layout=TMEM)
    columns = plgpu.broadcasted_iota(jnp.int32, (rows, PART), 1, layout=TMEM)
    parts = block_k // PART

    def logits(read, index, part, masked):
        coords = algebra.Coords(q_pos, index * block_k + part * PART + columns, tile_keys=block_k)
        return algebra.logits_of(spec, read(part), coords, masked=masked)

    def step(index, carry: tuple[jax.Array, jax.Array], *, masked):
        m, total = carry
        scores = scratch.scores.turn(index)
        scratch.scores.ready(scores)
        read = functools.partial(_part, scratch.scores.slot(scores))
        # The row's new maximum over the tile's parts, and the factor the state before takes to it.
        new = m
        for part in range(parts):
            new = algebra.peak(spec, new, *logits(read, index, part, masked))[0]
        m, alpha = algebra.peaked(spec, m, new)

        @pl.when(index > 0)
        def _():
            # The factor to the correction, in the slot two steps back's: the correction read that one
            # before the `P V` this step's weights wait for (`_correction`).
            turn = scratch.scaled.turn(index - 1)
            scratch.scales.at[turn.slot][...] = alpha
            scratch.scaled.published(turn)

        # The `P V` two steps back done reading the buffer this step's weights overwrite; the last step's
        # reads the other meanwhile.
        pl.when(index > 1)(lambda: scratch.o_done.wait(scratch.o_done.turn(index - 2)))

        for part in range(parts):
            # The weights complete before their two readers, the sum below and the narrowed copy `P V` takes:
            # else the compiler fuses each one's last multiply into the sum's fma and sinks the sums past the
            # last part (a full call's parts are one block, no mask's branch between them), keeping both
            # operands of every earlier part's weights live to there, 24 registers a part, 250 of 255 and the
            # denominator spilled at width 64 (ptxas for a B200, tests/test_attention_blackwell.py's compile).
            weights = lax.optimization_barrier(algebra.weights(spec, logits(read, index, part, masked)[0], m))
            total = (algebra.denominator_after(total, alpha, weights.sum(axis=-1)) if part == 0
                     else total + weights.sum(axis=-1))
            narrowed = algebra.narrow(weights, dtype)
            for slot, p_tmem in enumerate(scratch.p_tmems):
                weighed = Accumulator(p_tmem, (rows, block_k), dtype=dtype)
                pl.when(lax.rem(index, 2) == slot)(functools.partial(
                    RINGS.write, weighed, narrowed, first=part * PART))
        # The next S may overwrite this one once it is read out (`Tensor.release`).
        scratch.scores.release(scratch.scores.turn(index))
        scratch.p_full.published(scratch.p_full.turn(index))
        return m, total

    whole = block.steps
    padded = bool(spec.keys % block_k)
    if spec.form == "causal" or padded:
        # The blocks every query of the block sees whole, below its diagonal and before padding.
        whole = jnp.maximum(jnp.minimum(lax.div(block.first, block_k), blocks - padded), 0)
    # Its layouts inferred from the row reductions the walk's carry meets.
    start = (jnp.full((rows,), algebra.LEAST, jnp.float32), jnp.zeros((rows,), jnp.float32))
    def walked(index, carry):
        # A block the query block walks, masked from its first that some query sees in part.
        return lax.cond(index < block.steps, lambda carry: step(index, carry, masked=index >= whole),
                        lambda carry: carry, carry)

    m, total = lax.fori_loop(0, blocks, walked, start)
    # The rows' maxima and denominators to the epilogue, through shared memory, which relayouts them.
    scratch.rows_smem.at[0][...] = m
    scratch.rows_smem.at[1][...] = total
    scratch.finished.published(scratch.finished.turn(0))


def _part(scores: Accumulator, part: int) -> jax.Array:
    """A step's `S` tile's columns `PART` at a time, from tensor memory in its own layout."""
    return RINGS.read(scores, first=part * PART, columns=PART, layout=TMEM)


def _correction(block: _Block, scratch: _Scratch, outs, *, spec: algebra.Forward, blocks: int) -> None:
    """The correction warpgroup: each step's rescale of the output's accumulator in tensor memory, once the
    last `P V` is in it, by the factor the softmax hands it (`CHUNK` columns at a time), published for the
    next `P V`; then the epilogue: the output, and each row's maximum (`algebra.walked_launches`' fold
    guard)."""
    out_ref, maxima_ref = outs
    rows = RINGS.rows
    value_width = int(scratch.acc_tmem.shape[1])
    output = Accumulator(scratch.acc_tmem, (rows, value_width))

    @pl.loop(1, blocks)
    def _(index):
        @pl.when(index < block.steps)
        def _():
            turn = scratch.scaled.turn(index - 1)
            scratch.scaled.wait(turn)
            alpha = plgpu.load(scratch.scales.at[turn.slot], layout=TMEM.reduce(1))
            scratch.o_done.wait(scratch.o_done.turn(index - 1))
            for start in range(0, value_width, CHUNK):
                RINGS.update(output, functools.partial(algebra.rescale, alpha=alpha), first=start,
                             columns=CHUNK)
            scratch.corrected.published(scratch.corrected.turn(index - 1))

    scratch.o_done.wait(scratch.o_done.turn(block.steps - 1))
    scratch.finished.wait(scratch.finished.turn(0))
    acc = RINGS.read(output)
    plgpu.wait_load_tmem()
    m, total = (plgpu.load(scratch.rows_smem.at[place], layout=STORED.reduce(1)) for place in (0, 1))
    out = algebra.finish(spec, algebra.Partial(acc, m, total), initial=False)[0]
    mosaic.stored((out,), (out_ref.at[block.row, pl.ds(block.first, rows), block.head],), smem=scratch.o_smem,
                  wait=False)
    places = (block.row, block.head, pl.ds(block.first, rows))
    # The rows' maxima through the denominator's stage, which the fold's guard reads; a row that saw no key
    # -inf (`algebra.reached`'s convention).
    m = jnp.where(algebra.no_keys(total), -jnp.inf, m)
    mosaic.stored((m,), (maxima_ref.at[places],), smem=scratch.l_smem)


def _forward(query, key, value, *, spec: algebra.Forward, tiles: Tiles, interpret: bool):
    """The launch at `tiles`: the output `[B, Qp, H, Dv]` and each row's running maximum in the walk's units
    `[B, H, Qp]` (`algebra.walked_launches`' plain launch)."""
    batch, queries, heads, width = query.shape
    kv_heads, value_width = key.shape[2], value.shape[-1]
    group = heads // kv_heads
    rows, block_k, dtype = RINGS.rows, tiles.block_k, query.dtype
    # The key tiles holding the call's keys, none of padding alone but a call of no keys' one (its zero
    # values weigh to a zero output, as `padded_call` pads it for).
    blocks = max(1, -(-spec.keys // block_k))
    # The GPU in effect, its tensor memory's columns the accumulators' bound (`Accumulators.scratch`).
    target = generation.gpu_target()
    # The keys' and values' rings, the bundle's streams (`mosaic_sm100.RINGS`), each block's rows at the
    # widest swizzle its contraction fills (`mosaic_sm100.swizzled`'s).
    rings = RINGS.streams(
        {"keys": ((block_k, width), dtype), "values": ((block_k, value_width), dtype)}, tiles.stages, 1,
        swizzles={"keys": resources.widest_swizzle(dtype, width),
                  "values": resources.widest_swizzle(dtype, value_width)})
    roles = schedule(rows, block_k)

    def kernel(q_ref, k_ref, v_ref, out_ref, maxima_ref, *refs):
        (q_smem, q_loaded, key_refs, value_refs, score_refs, p_even, p_odd, p_full, acc_tmem, o_done, scales,
         scaled, corrected, rows_smem, finished, o_smem, l_smem) = refs
        opened = RINGS.opened({name: dict(zip(rings[name], refs, strict=True))
                                            for name, refs in (("keys", key_refs), ("values", value_refs))})
        scores, p_full, o_done, scaled, corrected, finished = roles.opened(
            (score_refs, p_full, o_done, scaled, corrected, finished))
        scratch = _Scratch(q_smem, q_loaded, opened["keys"], opened["values"], scores, (p_even, p_odd),
                           p_full, acc_tmem, o_done, scales, scaled, corrected, rows_smem, finished, o_smem,
                           l_smem)
        block = _Block(spec.form == "causal", blocks, block_k)
        roles.run(
            softmax=lambda _: _softmax(block, scratch, spec=spec, tiles=tiles, blocks=blocks, dtype=dtype),
            correction=lambda _: _correction(block, scratch, (out_ref, maxima_ref), spec=spec, blocks=blocks),
            load=functools.partial(_load, block, scratch, k_ref, v_ref, group=group, tiles=tiles),
            mma=functools.partial(_products, block, scratch, q_ref))

    # The hand-offs' allocations (`schedule`), each where the program's shared and tensor memory have long
    # placed it; the rescales and the final rows staged plainly (`swizzling.plain`), stated rather than left
    # to layout inference, which tiled a raw buffer read at row offsets in the KDA backward.
    scores, p_full, o_done, scaled, corrected, finished = roles.scratch(target)
    scratch = (mosaic_sm100.operand((rows, width), dtype, width), RINGS.load_barriers(1),
               tuple(rings["keys"].values()), tuple(rings["values"].values()), scores,
               *(mosaic_sm100.operand_columns(rows, block_k, dtype) for _ in range(2)), p_full,
               mosaic_sm100.accumulator_columns(rows, value_width), o_done,
               swizzling.plain((2, rows), jnp.float32), scaled, corrected,
               swizzling.plain((2, rows), jnp.float32), finished,
               mosaic_sm100.operand((rows, value_width), dtype, value_width), mosaic_sm100.vector(rows))
    rows_type = jax.ShapeDtypeStruct((batch, heads, queries), jnp.float32)
    out, maxima = mosaic.kernel(
        kernel, out_type=(jax.ShapeDtypeStruct((batch, queries, heads, value_width), dtype), rows_type),
        grid=(heads, queries // rows, batch), grid_names=("heads", "q_blocks", "batch"), scratch=scratch,
        interpret=bodies.interpreted(interpret), warpgroups=roles.warpgroups(), family=RINGS.name,
        # Its register decision, its two raised roles at the registers the schedule raises them to and the
        # issuing warpgroup lowered (`report.Launch.kernels`), which the compile point judges its kernel by.
        model=resources.register_model(RINGS, roles.warpgroups() - 1, roles.registers()["softmax"]),
        # A name a form (the default's fold and its natural rewalk compile apart).
        name="attention_forward_tcgen05" + ("_natural" if spec.natural else ""),
    )(query, key, value)
    return out, maxima
