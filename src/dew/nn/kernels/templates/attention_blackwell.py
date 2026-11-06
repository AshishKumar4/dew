"""The attention template's Blackwell mainloop (Mosaic GPU, sm100), the design's first sm100 rung
(docs/design/kernel-templates.md §6), on the tcgen05 bundle (`mosaic_sm100`, the GEMM template's
`gemm.mosaic_blackwell` beside it): tcgen05's products with their accumulators in tensor memory, as
FlashAttention-4 and Tokamax's sm100 kernel hold them, and the template's algebra (`attention_algebra`)
between the products.

A program is a head's query block of `TCGEN05.rows` rows, in the bundle's two warpgroups
(`Tcgen05.roles`). The issuing one's load warp fills a ring of key stages and one of value stages
(`mosaic_sm100.Stages`), and its product warp issues each step's `S = Q Kᵀ` into tensor memory and, a step
behind, `P V` into the output's accumulator there, so a step's `S` is computed while the last step's
weights multiply their values. The reading warpgroup runs the softmax: it reads `S` out of tensor memory,
hands it to the algebra (`logits_of`, `peak`, `weights`, `denominator_after`), rescales the output's
accumulator in place once the last `P V` is in it (`rescale`, `CHUNK` columns at a time, so `S` and the
accumulator are never both in registers), writes the narrowed weights back for the next `P V`, and
finishes. A causal call walks the key blocks below its block's diagonal unmasked and the diagonal's masked.

The forward over full and causal calls of 16-bit operands, no hooks, one CTA a block: the rung's first
piece. FlashAttention-4's correction warpgroup, two-CTA products and lazy rescale follow as rungs."""

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
from dew.nn.kernels.templates import attention_algebra as algebra, attention_portable, mosaic, mosaic_sm100
from dew.nn.kernels.templates.attention import NATURAL, Hooks, KeySpans, padded_call, prescaled
from dew.nn.kernels.templates.mosaic_sm100 import STORED, TCGEN05, TMEM

CHUNK = 32
"""The output accumulator's columns the softmax rescales at a time."""
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
    value_width = value.shape[-1]
    # The query taken by its scale's power of two before its products (`prescaled`), so a logit finite at
    # the call's scale is finite in the product.
    query, scale, power = prescaled(query, scale)
    # Queries and keys padded to whole tiles of both: a program's query block and a walk's key tile.
    call = padded_call(query, key, value, spans, Hooks(), math.lcm(TCGEN05.rows, tiles.block_k))
    spec = algebra.Forward(spans.form, spans.keys, ids=False, scale=scale, natural=natural,
                           headroom=algebra.headroom(value.dtype), prescale=power)
    # Swizzled tiles hold whole 64-element rows of 16-bit operands.
    call = dataclasses.replace(call, **{name: jnp.pad(array, ((0, 0),) * 3 + ((0, -array.shape[3] % 64),))
                                        for name, array in (("query", call.query), ("key", call.key),
                                                            ("value", call.value))})

    def launch(form: algebra.Forward, live: jax.Array | None, again: bool = False):
        if live is None:
            return _forward(call.query, call.key, call.value, spec=form, tiles=tiles, interpret=interpret,
                            kept=again)
        # The rows whose weighted values overflowed, rewalked shifting on the plain-JAX mainloop over the
        # same padded call (the rare launch, `algebra.walked_launches`); its maxima and overflows are not
        # read.
        out, _ = attention_portable.padded_walk(form, call, Hooks(), block_q=TCGEN05.rows,
                                                block_k=tiles.block_k, residuals=False)
        return out, jnp.zeros(live.shape, jnp.float32), jnp.zeros_like(live)

    def kept(live: jax.Array, guarded: jax.Array, plain: jax.Array) -> jax.Array:
        return jnp.where(jnp.swapaxes(live, 1, 2)[..., None], guarded, plain)

    out = algebra.walked_launches(spec, launch, kept)
    return out[:, :call.queries, :, :value_width]


class _Scratch(NamedTuple):
    """A program's opened scratch: its query block and its load's signal, the key and value rings
    (`mosaic_sm100.Ring`), `S`'s accumulator (`mosaic_sm100.Tensor`: the products commit it, the softmax
    releases it once read), the weights in tensor memory and the signal publishing them, the output's
    accumulator and the signal each `P V` commits (`mosaic_sm100.Signal`, a turn a step), and the
    epilogue's stages, which it writes by item (`ref[...] = `, which no static type of JAX's refs
    declares)."""

    q_smem: mosaic.Ref
    q_loaded: mosaic_sm100.Signal
    keys: mosaic_sm100.Ring
    values: mosaic_sm100.Ring
    scores: mosaic_sm100.Tensor
    p_tmem: mosaic.Ref
    p_full: mosaic_sm100.Signal
    acc_tmem: mosaic.Ref
    o_done: mosaic_sm100.Signal
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
        return lax.axis_index("q_blocks") * TCGEN05.rows

    @property
    def steps(self):
        if not self.causal:
            return self.blocks
        return jnp.minimum(pl.cdiv(self.first + TCGEN05.rows, self.block_k), self.blocks)


def _load(block: _Block, scratch: _Scratch, q_ref, k_ref, v_ref, *, group: int, tiles: Tiles) -> None:
    """The load warp: the query block once, then each key block's keys and values into their rings."""
    queries = q_ref.at[block.row, pl.ds(block.first, TCGEN05.rows), block.head]
    scratch.q_loaded.loaded(scratch.q_loaded.turn(0), queries, scratch.q_smem)
    kv_head = lax.div(block.head, group)

    @pl.loop(0, block.blocks)
    def _(index):
        @pl.when(index < block.steps)
        def _():
            tile = (block.row, pl.ds(index * tiles.block_k, tiles.block_k), kv_head)
            scratch.keys.load(scratch.keys.turn(index), (k_ref.at[tile],))
            scratch.values.load(scratch.values.turn(index), (v_ref.at[tile],))

    # Each stage's last releases, observed so its barriers end where they began (`Ring.drain`).
    scratch.keys.drain(1, block.steps)
    scratch.values.drain(1, block.steps)


def _products(block: _Block, scratch: _Scratch) -> None:
    """The product warp: each step's `S = Q Kᵀ` into tensor memory once the softmax has read the last one,
    and a step behind, `P V` into the output's accumulator once its weights are there."""
    scratch.q_loaded.wait(scratch.q_loaded.turn(0))

    def values(index):
        (v_smem,), consumed = scratch.values.acquire(scratch.values.turn(index))
        scratch.p_full.wait(scratch.p_full.turn(index))
        TCGEN05.product(scratch.acc_tmem, scratch.p_tmem, v_smem, consumed, accumulate=index > 0)
        scratch.o_done.committed(scratch.o_done.turn(index))

    @pl.loop(0, block.blocks)
    def _(index):
        @pl.when(index < block.steps)
        def _():
            (k_smem,), consumed = scratch.keys.acquire(scratch.keys.turn(index))
            scores = scratch.scores.turn(index)
            s_tmem = scratch.scores.acquire(scores)
            TCGEN05.product(s_tmem, scratch.q_smem, k_smem.T, consumed, accumulate=False)
            scratch.scores.issued(scores)
            pl.when(index > 0)(functools.partial(values, index - 1))

    values(block.steps - 1)
    scratch.scores.drain(block.steps)


def _softmax(block: _Block, scratch: _Scratch, outs, *, spec: algebra.Forward, tiles: Tiles, blocks: int,
             dtype) -> None:
    """The reading warpgroup: each step's `S` through the algebra, the output's accumulator taken to the
    step's maximum once the last `P V` is in it, the narrowed weights into tensor memory; then the
    epilogue: the output, and each row's maximum and whether its weighted values overflowed
    (`algebra.walked_launches`' guards)."""
    out_ref, maxima_ref, over_ref = outs
    rows, block_k = TCGEN05.rows, tiles.block_k
    value_width = int(scratch.acc_tmem.shape[1])
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
        read = scratch.scores.read(scratch.scores.turn(index), PART, layout=TMEM)
        # The row's new maximum over the tile's parts, and the factor the state before takes to it.
        new = m
        for part in range(parts):
            new = algebra.peak(spec, new, *logits(read, index, part, masked))[0]
        m, alpha = algebra.peaked(spec, m, new)

        @pl.when(index > 0)
        def _():
            # The last step's P V is in the accumulator, its weights read: take the accumulator to this
            # step's maximum, `CHUNK` columns at a time, before this step's weights overwrite them.
            scratch.o_done.wait(scratch.o_done.turn(index - 1))
            for start in range(0, value_width, CHUNK):
                acc = TCGEN05.load_columns(scratch.acc_tmem, start, CHUNK, layout=TMEM)
                TCGEN05.store_columns(scratch.acc_tmem, start, algebra.rescale(acc, alpha))

        for part in range(parts):
            # The weights complete before their two readers, the sum below and the narrowed copy `P V` takes:
            # else the compiler fuses each one's last multiply into the sum's fma and sinks the sums past the
            # last part (a full call's parts are one block, no mask's branch between them), keeping both
            # operands of every earlier part's weights live to there, 24 registers a part, 250 of 255 and the
            # denominator spilled at width 64 (ptxas for a B200, tests/test_attention_blackwell.py's compile).
            weights = lax.optimization_barrier(algebra.weights(spec, logits(read, index, part, masked)[0], m))
            total = (algebra.denominator_after(total, alpha, weights) if part == 0
                     else total + weights.sum(axis=-1))
            TCGEN05.store_columns(scratch.p_tmem, part * PART, algebra.narrow(weights, dtype))
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
    # A row that saw no key, by its denominator in the walk's row layout (its maximum's).
    no_keys = algebra.no_keys(total)
    scratch.o_done.wait(scratch.o_done.turn(block.steps - 1))
    acc = TCGEN05.load_columns(scratch.acc_tmem, 0, value_width)
    TCGEN05.loaded()
    # The denominator to the epilogue's row layout through shared memory, which relayouts it.
    total = TCGEN05.relaid(total, scratch.l_smem, STORED.reduce(1))
    out, _, _ = algebra.finish(spec, algebra.Partial(acc, m, total), initial=False)
    mosaic.stored((out,), (out_ref.at[block.row, pl.ds(block.first, rows), block.head],), smem=scratch.o_smem,
                  wait=False)
    places = (block.row, block.head, pl.ds(block.first, rows))
    # The rows' maxima, and the overflows of their sums, through the denominator's stage.
    over = algebra.overflowed(acc).astype(jnp.float32)
    # A row that saw no key -inf (`algebra.reached`'s convention).
    m = jnp.where(no_keys, -jnp.inf, m)
    mosaic.stored((m, over), (maxima_ref.at[places], over_ref.at[places]), smem=scratch.l_smem)


def _forward(query, key, value, *, spec: algebra.Forward, tiles: Tiles, interpret: bool, kept: bool = False):
    """The launch at `tiles`: the output `[B, Qp, H, Dv]` and each row's running maximum in the walk's units
    and whether its weighted values overflowed, `[B, H, Qp]` each (`algebra.walked_launches`' plain
    launch)."""
    batch, queries, heads, width = query.shape
    kv_heads, value_width = key.shape[2], value.shape[-1]
    group = heads // kv_heads
    rows, block_k, dtype = TCGEN05.rows, tiles.block_k, query.dtype
    # The key tiles holding the call's keys, none of padding alone but a call of no keys' one (its zero
    # values weigh to a zero output, as `padded_call` pads it for).
    blocks = max(1, -(-spec.keys // block_k))
    target = generation.gpu_target("sm100")
    transforms = mosaic_sm100.swizzled(dtype, width)
    key_stages = mosaic_sm100.Stages(tiles.stages, (((block_k, width), dtype, transforms),))
    value_stages = mosaic_sm100.Stages(tiles.stages, (((block_k, value_width), dtype,
                                                       mosaic_sm100.swizzled(dtype, value_width)),))
    scores = mosaic_sm100.Accumulators(1, rows, block_k)

    def kernel(q_ref, k_ref, v_ref, out_ref, maxima_ref, over_ref, *refs):
        (q_smem, q_loaded, key_refs, value_refs, score_refs, p_tmem, p_full, acc_tmem, o_done, o_smem,
         l_smem) = refs
        scratch = _Scratch(q_smem, loads.opened(q_loaded), key_stages.opened(*key_refs),
                           value_stages.opened(*value_refs), scores.opened(*score_refs), p_tmem,
                           signals.opened(p_full), acc_tmem, signals.opened(o_done), o_smem, l_smem)
        block = _Block(spec.form == "causal", blocks, block_k)
        TCGEN05.roles(
            functools.partial(_load, block, scratch, q_ref, k_ref, v_ref, group=group, tiles=tiles),
            functools.partial(_products, block, scratch),
            functools.partial(_softmax, block, scratch, (out_ref, maxima_ref, over_ref), spec=spec,
                              tiles=tiles, blocks=blocks, dtype=dtype))

    # The query block's load a load's signal; the weights' and each `P V`'s the tensor core's.
    loads, signals = mosaic_sm100.Signals(tensor_core=False), mosaic_sm100.Signals()
    scratch = (mosaic_sm100.operand((rows, width), dtype, width), loads.scratch(),
               key_stages.scratch(), value_stages.scratch(), scores.scratch(target),
               mosaic_sm100.operand_columns(rows, block_k, dtype), signals.scratch(),
               mosaic_sm100.accumulator_columns(rows, value_width), signals.scratch(),
               mosaic_sm100.operand((rows, value_width), dtype, value_width), mosaic_sm100.vector(rows))
    rows_type = jax.ShapeDtypeStruct((batch, heads, queries), jnp.float32)
    out, maxima, over = TCGEN05.kernel(
        kernel, out_type=(jax.ShapeDtypeStruct((batch, queries, heads, value_width), dtype), rows_type,
                          rows_type),
        grid=(heads, queries // rows, batch), grid_names=("heads", "q_blocks", "batch"), scratch=scratch,
        interpret=mosaic.interpreted(interpret),
        # A name a form (the default's fold and its natural rewalk compile apart), and the guard's rerun of
        # the plain launch its own (`algebra.walked_launches`' `again`).
        name="attention_forward_tcgen05" + ("_natural" if spec.natural else "") + ("_kept" if kept else ""),
    )(query, key, value)
    return out, maxima, over > 0
