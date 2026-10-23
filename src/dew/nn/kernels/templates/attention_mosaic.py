"""The attention template's Mosaic GPU mainloop on the wgmma family (Hopper, sm90; design §6):
FlashAttention-3's structure, built from the shared toolkit (`mosaic`: the roles, the operand rings) and
the family's issue (`mosaic_sm90`), with the template's algebra (`attention_algebra`) between the products.

One program a query head and a pair of query blocks: two consumer warpgroups, each a block of `Tiles.block_q`
rows, and a producer warpgroup streaming the key and value tiles through a ring each (`mosaic.Ring`), every
slot released by both consumers. The consumers take turns on the tensor cores (a schedule barrier between
them), so one's softmax runs under the other's products. A step issues S = Q·Kᵀ, releases the key slot, and
hands S to the algebra as a tile (`algebra.Keys`) whose values are the value slot and whose product
accumulates the narrowed weights from registers into the running sums in place; while it runs the consumer
waits for its next key tile. A causal block masks only the tiles past its diagonal's start, and a consumer
whose diagonal ends before its partner's keeps releasing the slots its partner still reads.

The forward over full and causal calls of 16-bit operands, the base-2 fold and the natural form without
headroom (fp16 values, which cannot overflow it), no hooks."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import attention_algebra as algebra, mosaic, mosaic_sm90
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

CONSUMERS = 2
"""The consumer warpgroups of a program, each a query block (FlashAttention-3's two)."""
FAMILY = mosaic_sm90.FAMILY


@dataclasses.dataclass(frozen=True, order=True)
class Tiles:
    """A launch's tiles: each consumer's query block (`block_q`, whole warpgroup rows of the family), the key
    tile (`block_k`) and the key and value tiles in flight (`stages`)."""

    block_q: int
    block_k: int
    stages: int

    def __post_init__(self):
        if self.block_q % FAMILY.rows or self.block_k % FAMILY.rows or self.stages < 2:
            raise ValueError(f"{self}: query and key tiles of whole {FAMILY.rows} rows, two stages or more")


DEFAULT = Tiles(64, 128, 2)
"""The forward's tiles: FlashAttention-3's at width 128, a warpgroup's 64 queries over 128 keys in two stages,
the fastest of 64 or 128 queries over 64 or 128 keys in two to four stages on an H100 (llama-d128 causal bf16,
2 x 4096: 0.65 ms; 128 queries a consumer spill, 2.9 to 4.6 ms)."""


def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, natural: bool = NATURAL, tiles: Tiles = DEFAULT,
            interpret: bool = False) -> jax.Array:
    """The template's forward on this mainloop: `query` `[B, Q, H, D]` over `key` and `value` `[B, K, Hk, D]`
    and the keys `spans` leaves each query (full or causal), as `dew.nn.kernels.templates.attention.reference`
    computes it. `natural` keeps the logits natural (`algebra.Forward.base2`); `interpret` runs it on the CPU
    (Mosaic GPU's interpret mode)."""
    if hooks is not None and hooks != Hooks():
        raise NotImplementedError("the sm90 mainloop takes no hooks yet")
    if spans.form not in ("full", "causal"):
        raise NotImplementedError(f"the sm90 mainloop takes full and causal calls, not {spans.form!r} ones")
    if jnp.dtype(query.dtype) not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
        raise NotImplementedError(f"the sm90 mainloop takes 16-bit operands, not {jnp.dtype(query.dtype)}")
    width, value_width = query.shape[3], value.shape[3]
    if width != value_width:
        raise NotImplementedError("the sm90 mainloop's output reuses the query's stage: equal head widths")
    call = padded_call(query, key, value, spans, Hooks(), max(CONSUMERS * tiles.block_q, tiles.block_k))
    spec = algebra.Forward(spans.form, spans.keys, ids=False, scale=width ** -0.5 if scale is None else scale,
                           natural=natural)
    if spec.natural and algebra.headroom(value.dtype):
        raise NotImplementedError("the sm90 mainloop's natural form takes no headroom guard yet")
    # The family's swizzled tiles hold whole rows of `FAMILY.rows` 16-bit entries.
    query, key, value = (jnp.pad(array, ((0, 0),) * 3 + ((0, -array.shape[3] % FAMILY.rows),))
                         for array in (call.query, call.key, call.value))
    out = _launch(query, key, value, spec=spec, tiles=tiles, interpret=interpret)
    return out[:, :call.queries, :, :value_width]


@dataclasses.dataclass(frozen=True)
class _Program:
    """A program's static geometry: its tiles, the call's key tiles (`steps`), the slots a ring holds, its key
    count and whether it is causal."""

    tiles: Tiles
    steps: int
    stages: int
    keys: int
    causal: bool

    def walked(self, first):
        """The key tiles a query block from `first` reads: every one, or those up to its diagonal."""
        if not self.causal:
            return jnp.int32(self.steps)
        return jnp.minimum(pl.cdiv(first + self.tiles.block_q, self.tiles.block_k), self.steps)

    def whole(self, first, walked):
        """The tiles every query of the block from `first` sees whole: below its diagonal, before padding."""
        if not self.causal and not self.keys % self.tiles.block_k:
            return walked
        whole = jnp.minimum(lax.div(first, self.tiles.block_k),
                            self.steps - bool(self.keys % self.tiles.block_k))
        return jnp.maximum(whole if self.causal else jnp.int32(self.steps - 1), 0)


def _consumer(program: _Program, spec: algebra.Forward, wg, q_view, out_view, q_smem, q_loaded, keys, values,
              turn, schedule, most, first, *, interpret: bool):
    """A consumer warpgroup's query block: its walk of the key tiles through the rings (`algebra.plain_walk`),
    then its output stored through its query stage."""
    tiles = program.tiles
    shape = (tiles.block_q, tiles.block_k)
    walked = program.walked(first)
    whole = program.whole(first, walked)
    plgpu.copy_gmem_to_smem(q_view, q_smem, q_loaded)
    plgpu.barrier_wait(q_loaded)
    q_pos, columns = first + mosaic_sm90.iota(shape, 0), mosaic_sm90.iota(shape, 1)
    width = q_smem.shape[-1]
    state = algebra.Partial(mosaic_sm90.zeros((tiles.block_q, width)),
                            plgpu.layout_cast(jnp.full((tiles.block_q,), -jnp.inf, jnp.float32),
                                              mosaic_sm90.ROW),
                            mosaic_sm90.zeros((tiles.block_q,)))
    def awaited(index):
        @pl.when(index < walked)
        def _():
            keys.acquire(index)

    awaited(jnp.int32(0))
    pl.when(wg == 1)(turn)

    def keys_at(index) -> algebra.Keys:
        s = mosaic_sm90.product(q_smem, keys.at(index).transpose((1, 0)), shape, turn)
        keys.release(index)
        coords = algebra.Coords(q_pos, index * tiles.block_k + columns)
        # A tile every query sees whole takes no mask: one walk, the choice a tile's (a walk of each, these
        # rings' waits and arrivals in two loops, faulted on an H100 under jax 0.11.2's lowering).
        logits = lax.cond(index < whole, lambda: algebra.modify(spec, s, coords, masked=False),
                          lambda: algebra.modify(spec, s, coords, masked=True))

        def load():
            # The value slot, once the partner's products are issued: its turn on the tensor cores is over
            # when this one's come.
            plgpu.barrier_arrive(schedule)
            slot = values.acquire(index)
            plgpu.barrier_wait(schedule)
            return slot

        def dot(lhs, rhs):
            product = mosaic_sm90.registers_product(lhs, rhs, interpret)
            values.release(index)
            # The next key tile's wait, before its scores need it.
            awaited(index + 1)
            return product

        return algebra.Keys(coords, logits, load, dot)

    state = algebra.plain_walk(spec, 0, walked, keys_at, state)

    def released(index, carry):
        # The slots a consumer whose diagonal ends first leaves its partner, awaited and released as it
        # would have (each consumer waits on every load: a wait takes its barrier's next phase).
        keys.acquire(index)
        keys.release(index)
        values.acquire(index)
        values.release(index)
        turn()
        turn()
        return carry

    lax.fori_loop(walked, most, released, None)
    pl.when(wg == 0)(turn)
    out, _, _ = algebra.finish(spec, state, initial=False)
    q_smem[...] = out.astype(q_smem.dtype)
    plgpu.commit_smem()
    plgpu.copy_smem_to_gmem(q_smem, out_view)
    plgpu.wait_smem_to_gmem(0)


def _launch(query, key, value, *, spec: algebra.Forward, tiles: Tiles, interpret: bool):
    batch, queries, heads, width = query.shape
    keys_count, kv_heads = key.shape[1], key.shape[2]
    group = heads // kv_heads
    block_q, block_k = tiles.block_q, tiles.block_k
    pair = CONSUMERS * block_q
    steps = keys_count // block_k
    # No more stages than key tiles: a slot past them would load past the keys.
    program = _Program(tiles, steps, min(tiles.stages, steps), spec.keys, spec.form == "causal")

    def body(q_ref, k_ref, v_ref, out_ref, q_smem, q_loaded, schedule, key_ring, value_ring):
        keys, values = mosaic.Ring(**key_ring), mosaic.Ring(**value_ring)
        row, head, wg = lax.axis_index("batch"), lax.axis_index("heads"), lax.axis_index("wg")
        pair_first = lax.axis_index("q_tiles") * pair
        most = program.walked(pair_first + pair - block_q)

        def turn():
            plgpu.barrier_arrive(schedule)
            plgpu.barrier_wait(schedule)

        def consumer(wg):
            first = pair_first + wg * block_q
            rows = (row, pl.ds(first, block_q), head)
            _consumer(program, spec, wg, q_ref.at[rows], out_ref.at[rows], q_smem.at[wg], q_loaded.at[wg],
                      keys, values, turn, schedule, most, first, interpret=interpret)

        def producer():
            kv_head = lax.div(head, group)

            @pl.loop(0, most)
            def _(index):
                tile = (row, pl.ds(index * block_k, block_k), kv_head)
                keys.load(index, k_ref.at[tile])
                values.load(index, v_ref.at[tile])

            keys.drain(most)
            values.drain(most)

        mosaic.roles(wg, CONSUMERS, consumer, producer)

    def entry(q_ref, k_ref, v_ref, out_ref):
        pl.run_scoped(
            lambda *scoped, **rings: body(q_ref, k_ref, v_ref, out_ref, *scoped, **rings),
            plgpu.SMEM((CONSUMERS, block_q, width), query.dtype),
            plgpu.Barrier(num_barriers=CONSUMERS), plgpu.Barrier(num_arrivals=CONSUMERS),
            key_ring=mosaic.ring((block_k, width), key.dtype, program.stages, consumers=CONSUMERS),
            value_ring=mosaic.ring((block_k, value.shape[-1]), value.dtype, program.stages,
                                   consumers=CONSUMERS),
            collective_axes="wg")

    return plgpu.kernel(
        entry, out_type=jax.ShapeDtypeStruct((batch, queries, heads, value.shape[-1]), query.dtype),
        grid=(heads, queries // pair, batch), grid_names=("heads", "q_tiles", "batch"),
        num_threads=CONSUMERS + 1, thread_name="wg", compiler_params=mosaic.compiler_params(approx_math=True),
        interpret=plgpu.InterpretGPUParams() if interpret else None,
    )(query, key, value)


BACKWARD_DEFAULT = Tiles(64, 64, 2)
"""The backward's tiles: each consumer's 64 queries over 64-key tiles, two stages, so a dQ program's key and
value rings, its consumers' queries and output cotangents, and their fp32 sums fit a block's shared memory."""
BACKWARD_CONSUMERS = 1
"""The backward's consumer warpgroups a program: one, whose threads take every register a producer leaves
(two consumers' 232 a thread spilled 0.4 to 18 GB a launch on an H100)."""


def _rows(ref, row, head, first, count: int):
    """A query block's per-row statistic `[count]` of a `[B, H, Q]` array, in a row's layout."""
    return plgpu.load(ref.at[row, head, pl.ds(first, count)], (), layout=mosaic_sm90.ROW, optimized=False)


def _dq_launch(query, key, value, dout, peak, prior, anchor, delta, key_index, *, how: algebra.Recompute,
               scale: float, causal: bool, keys_count: int, tiles: Tiles, accurate: bool, consumers: int,
               interpret: bool):
    """The dQ program's sums over its key tiles (`algebra.dq_step`), unfinished: dQ's and, in the accurate
    form, the keys' product with the weights (`DqSums.keyed`), `[B, Qp, H, D]` fp32 each, and the cotangents'
    and weights' row sums and the keyed sums' mass and exponent, `[B, H, Qp]` each; `dq_finish` takes them to
    dQ and the refined D past the launch. Each query block of a pair is a consumer's, the key and value tiles
    streamed through a ring each."""
    batch, queries, heads, width = query.shape
    kv_heads, value_width = key.shape[2], value.shape[-1]
    group = heads // kv_heads
    block_q, block_k = tiles.block_q, tiles.block_k
    pair = consumers * block_q
    steps = key.shape[1] // block_k
    program = _Program(tiles, steps, min(tiles.stages, steps), keys_count, causal)
    shape = (block_q, block_k)

    def body(q_ref, k_ref, v_ref, do_ref, peak_ref, prior_ref, anchor_ref, delta_ref, index_ref, dq_ref,
             keyed_ref, rows_ref, q_smem, do_smem, out_smem, row_smem, staged, q_loaded, do_loaded, key_ring,
             value_ring):
        keys, values = mosaic.Ring(**key_ring), mosaic.Ring(**value_ring)
        row, head, wg = lax.axis_index("batch"), lax.axis_index("heads"), lax.axis_index("wg")
        pair_first = lax.axis_index("q_tiles") * pair
        most = program.walked(pair_first + pair - block_q)

        def consumer(wg):
            first = pair_first + wg * block_q
            rows = (row, pl.ds(first, block_q), head)
            plgpu.copy_gmem_to_smem(q_ref.at[rows], q_smem.at[wg], q_loaded.at[wg])
            plgpu.copy_gmem_to_smem(do_ref.at[rows], do_smem.at[wg], do_loaded.at[wg])
            plgpu.barrier_wait(q_loaded.at[wg])
            plgpu.barrier_wait(do_loaded.at[wg])
            peak, prior, anchor, delta, chosen = (_rows(ref, row, head, first, block_q)
                                                  for ref in (peak_ref, prior_ref, anchor_ref, delta_ref,
                                                              index_ref))
            walked = program.walked(first)
            whole = program.whole(first, walked)
            q_pos, columns = first + mosaic_sm90.iota(shape, 0), mosaic_sm90.iota(shape, 1)
            nothing = mosaic_sm90.zeros((block_q,))
            # The row sums carried; dQ and the keyed sums accumulate in place (`mosaic_sm90.accumulating`).
            start = algebra.DqSums(None, nothing, nothing, (), (), None, nothing, nothing + algebra.NO_SHIFT)
            parts = staged.at[wg]

            def step(index, sums, dq_acc, keyed_acc):
                k_slot, v_slot = keys.acquire(index), values.acquire(index)
                s = mosaic_sm90.product(q_smem.at[wg], k_slot.transpose((1, 0)), shape)
                dp = mosaic_sm90.product(do_smem.at[wg], v_slot.transpose((1, 0)), shape)
                values.release(index)
                k_pos = index * block_k + columns
                seen = lax.cond(index < whole, lambda: k_pos >= 0,
                                lambda: (k_pos <= q_pos if causal else k_pos >= 0) & (k_pos < keys_count))
                logits, slopes = algebra.tile_logits(None, s, seen, q_pos, k_pos, (),
                                                     scale=scale if how.factor is None else 1.0, slopes=True)
                exponentials, _ = algebra.normalized(logits, peak, prior, how)
                at_anchor = k_pos == algebra.expand(chosen, k_pos)
                cotangents = algebra.grad_scores(exponentials, dp, anchor, delta, seen, at_anchor,
                                                 algebra.inverse(prior))

                # The weights' sums unnormalized (`algebra.dq_finish` takes them to the pass's own
                # denominator); each product's parts issued into one accumulator.
                sums, products = algebra.dq_terms(sums, exponentials, cotangents, slopes, seen, prior,
                                                  scale=scale, kinds=(), summed=(), dtype=k_slot.dtype,
                                                  accurate=accurate)
                mosaic_sm90.staged_products(dq_acc, products.parts, k_slot,
                                            parts.at[pl.ds(0, len(products.parts))])
                if products.keyed is not None:
                    mosaic_sm90.staged_products(keyed_acc, products.keyed, k_slot,
                                                parts.at[pl.ds(len(products.parts), 1)])
                # The key slot's and the staged parts' reads done before either is reused.
                plgpu.wgmma_wait(0)
                keys.release(index)
                return sums

            def walk(dq_acc, keyed_acc):
                sums = lax.fori_loop(0, walked, lambda index, sums: step(index, sums, dq_acc, keyed_acc),
                                     start)
                return sums._replace(dq=dq_acc[...], keyed=keyed_acc[...])

            sums = mosaic_sm90.accumulating(walk, (block_q, width), (block_q, width))

            @pl.loop(walked, most)
            def _(index):
                # The tiles a block whose diagonal ends first leaves its partner, awaited and released
                # unread (each consumer waits on every load: a wait takes its barrier's next phase).
                keys.acquire(index)
                keys.release(index)
                values.acquire(index)
                values.release(index)

            out = (row, pl.ds(first, block_q), head)
            for target, part in ((dq_ref, sums.dq), *(((keyed_ref, sums.keyed),) if accurate else ())):
                out_smem.at[wg][...] = part
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(out_smem.at[wg], target.at[out])
                plgpu.wait_smem_to_gmem(0)
            for place, part in enumerate((sums.residual, sums.mass, sums.keyed_mass, sums.keyed_shift)):
                # A row's layout whatever the sum's (the keyed mass of a call without a hook is a constant).
                row_smem.at[wg, place][...] = plgpu.layout_cast(part, mosaic_sm90.ROW)
            plgpu.commit_smem()
            for place in range(4):
                plgpu.copy_smem_to_gmem(row_smem.at[wg, place],
                                        rows_ref.at[place, row, head, pl.ds(first, block_q)])
            plgpu.wait_smem_to_gmem(0)

        def producer():
            kv_head = lax.div(head, group)

            @pl.loop(0, most)
            def _(index):
                tile = (row, pl.ds(index * block_k, block_k), kv_head)
                keys.load(index, k_ref.at[tile])
                values.load(index, v_ref.at[tile])

            keys.drain(most)
            values.drain(most)

        mosaic.roles(wg, consumers, consumer, producer)

    def entry(*refs):
        pl.run_scoped(
            lambda *scoped, **rings: body(*refs, *scoped, **rings),
            plgpu.SMEM((consumers, block_q, width), query.dtype),
            plgpu.SMEM((consumers, block_q, value_width), dout.dtype),
            plgpu.SMEM((consumers, block_q, width), jnp.float32),
            plgpu.SMEM((consumers, 4, block_q), jnp.float32),
            # A tile's parts staged as the products' left operands: dQ's cotangents in as many parts as the
            # form takes (`algebra.cotangent_parts`, three in the accurate form's bf16), and its weights.
            plgpu.SMEM((consumers, 4 if accurate else 1, block_q, block_k), query.dtype),
            plgpu.Barrier(num_barriers=consumers), plgpu.Barrier(num_barriers=consumers),
            key_ring=mosaic.ring((block_k, width), key.dtype, program.stages, consumers=consumers),
            value_ring=mosaic.ring((block_k, value_width), value.dtype, program.stages,
                                   consumers=consumers),
            collective_axes="wg")

    sums = jax.ShapeDtypeStruct((batch, queries, heads, width), jnp.float32)
    return plgpu.kernel(
        entry, out_type=(sums, sums, jax.ShapeDtypeStruct((4, batch, heads, queries), jnp.float32)),
        grid=(heads, queries // pair, batch), grid_names=("heads", "q_tiles", "batch"),
        num_threads=consumers + 1, thread_name="wg",
        compiler_params=mosaic.compiler_params(approx_math=True),
        interpret=plgpu.InterpretGPUParams() if interpret else None,
    )(query, key, value, dout, peak, prior, anchor, delta, key_index)


def _dkv_launch(query, key, value, dout, peak, total, anchor, delta, key_index, *, how: algebra.Recompute,
                scale: float, causal: bool, keys_count: int, tiles: Tiles, accurate: bool, consumers: int,
                interpret: bool):
    """The dK/dV program under the dQ program's softmax: each consumer's `block_k` keys of a query head,
    resident, walked by the query tiles that see them, streamed with their output cotangents through a
    ring each; a tile's weights and score cotangents (the algebra's, rows the queries) staged in shared
    memory as the transposed products' left operands (`algebra.value_operands`, `cotangent_product`). Returns
    the query head's dK and dV `[B, Kp, H, D]` fp32."""
    batch, queries, heads, width = query.shape
    keys_padded, kv_heads, value_width = key.shape[1], key.shape[2], value.shape[-1]
    group = heads // kv_heads
    block_q, block_k = tiles.block_q, tiles.block_k
    pair = consumers * block_k
    q_steps = queries // block_q
    stages = min(tiles.stages, q_steps)
    shape = (block_q, block_k)

    def first_tile(k_first):
        # The first query tile that sees a key block from `k_first`: every one, or those from its diagonal.
        return lax.div(k_first, block_q) if causal else jnp.int32(0)

    def body(q_ref, k_ref, v_ref, do_ref, peak_ref, total_ref, anchor_ref, delta_ref, index_ref, dk_ref,
             dv_ref, k_smem, v_smem, staged, out_smem, k_loaded, v_loaded, query_ring, dout_ring):
        queries_ring, douts = mosaic.Ring(**query_ring), mosaic.Ring(**dout_ring)
        row, head, wg = lax.axis_index("batch"), lax.axis_index("heads"), lax.axis_index("wg")
        pair_first = lax.axis_index("k_tiles") * pair
        kv_head = lax.div(head, group)
        least = first_tile(pair_first)

        def consumer(wg):
            k_first = pair_first + wg * block_k
            keys_view = (row, pl.ds(k_first, block_k), kv_head)
            plgpu.copy_gmem_to_smem(k_ref.at[keys_view], k_smem.at[wg], k_loaded.at[wg])
            plgpu.copy_gmem_to_smem(v_ref.at[keys_view], v_smem.at[wg], v_loaded.at[wg])
            plgpu.barrier_wait(k_loaded.at[wg])
            plgpu.barrier_wait(v_loaded.at[wg])
            mine = first_tile(k_first)
            columns = k_first + mosaic_sm90.iota(shape, 1)
            rows_iota = mosaic_sm90.iota(shape, 0)


            @pl.loop(least, mine)
            def _(index):
                # The tiles before this block's diagonal its partner reads, awaited and released unread
                # (each ring's counter the program's tile count from its first, `least`).
                queries_ring.acquire(index - least)
                queries_ring.release(index - least)
                douts.acquire(index - least)
                douts.release(index - least)

            def step(index, carry, dk_acc, dv_acc):
                q_slot, do_slot = queries_ring.acquire(index - least), douts.acquire(index - least)
                q_first = index * block_q
                peak, denominator, centre, rest, chosen = (_rows(ref, row, head, q_first, block_q)
                                                           for ref in (peak_ref, total_ref, anchor_ref,
                                                                       delta_ref, index_ref))
                s = mosaic_sm90.product(q_slot, k_smem.at[wg].transpose((1, 0)), shape)
                dp = mosaic_sm90.product(do_slot, v_smem.at[wg].transpose((1, 0)), shape)
                q_pos = q_first + rows_iota
                seen = ((columns <= q_pos) if causal else columns >= 0) & (columns < keys_count)
                logits, slopes = algebra.tile_logits(None, s, seen, q_pos, columns, (),
                                                     scale=scale if how.factor is None else 1.0, slopes=True)
                exponentials, weights = algebra.normalized(logits, peak, denominator, how)
                cotangents = algebra.grad_scores(exponentials, dp, centre, rest, seen,
                                                 columns == algebra.expand(chosen, columns),
                                                 algebra.inverse(denominator))
                pulled, _ = algebra.pulled(cotangents, slopes, seen, ())
                narrowed, _ = algebra.value_operands(exponentials, weights, do_slot, denominator)
                cotangent_parts = algebra.cotangent_parts(pulled * scale, q_slot.dtype, accurate=accurate)
                buffers = staged.at[wg]
                # Each staged part's transpose the left operand (`algebra.cotangent_product`'s `aᵀ @ b`).
                mosaic_sm90.staged_products(dv_acc, ((narrowed, None),), do_slot, buffers.at[pl.ds(0, 1)],
                                            transposed=True)
                mosaic_sm90.staged_products(dk_acc, cotangent_parts, q_slot,
                                            buffers.at[pl.ds(1, len(cotangent_parts))], transposed=True)
                # The slots' and the staged parts' reads done before either is reused.
                plgpu.wgmma_wait(0)
                queries_ring.release(index - least)
                douts.release(index - least)
                return carry

            def walk(dk_acc, dv_acc):
                lax.fori_loop(mine, q_steps, lambda index, carry: step(index, carry, dk_acc, dv_acc), None)
                return dk_acc[...], dv_acc[...]

            dk, dv = mosaic_sm90.accumulating(walk, (block_k, width), (block_k, value_width))
            out = (row, pl.ds(k_first, block_k), head)
            for target, part in ((dk_ref, dk), (dv_ref, dv)):
                out_smem.at[wg][...] = part
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(out_smem.at[wg], target.at[out])
                plgpu.wait_smem_to_gmem(0)

        def producer():
            @pl.loop(least, q_steps)
            def _(index):
                rows = (row, pl.ds(index * block_q, block_q), head)
                queries_ring.load(index - least, q_ref.at[rows])
                douts.load(index - least, do_ref.at[rows])

            queries_ring.drain(q_steps - least)
            douts.drain(q_steps - least)

        mosaic.roles(wg, consumers, consumer, producer)

    def entry(*refs):
        pl.run_scoped(
            lambda *scoped, **rings: body(*refs, *scoped, **rings),
            plgpu.SMEM((consumers, block_k, width), key.dtype),
            plgpu.SMEM((consumers, block_k, value_width), value.dtype),
            # A tile's parts staged for the transposed products: dV's weights, or dK's cotangents in as many
            # parts as the form takes (`algebra.cotangent_parts`, three in the accurate form's bf16).
            plgpu.SMEM((consumers, 4 if accurate else 2, block_q, block_k), query.dtype),
            plgpu.SMEM((consumers, block_k, width), jnp.float32),
            plgpu.Barrier(num_barriers=consumers), plgpu.Barrier(num_barriers=consumers),
            query_ring=mosaic.ring((block_q, width), query.dtype, stages, consumers=consumers),
            dout_ring=mosaic.ring((block_q, value_width), dout.dtype, stages, consumers=consumers),
            collective_axes="wg")

    gradient = jax.ShapeDtypeStruct((batch, keys_padded, heads, width), jnp.float32)
    return plgpu.kernel(
        entry, out_type=(gradient, gradient),
        grid=(heads, keys_padded // pair, batch), grid_names=("heads", "k_tiles", "batch"),
        num_threads=consumers + 1, thread_name="wg",
        compiler_params=mosaic.compiler_params(approx_math=True),
        interpret=plgpu.InterpretGPUParams() if interpret else None,
    )(query, key, value, dout, peak, total, anchor, delta, key_index)


def backward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None,
             dout: jax.Array, *, residuals: tuple[jax.Array, jax.Array, jax.Array, Split],
             scale: float | None = None, natural: bool = NATURAL, accurate: bool = ACCURATE,
             tiles: Tiles = BACKWARD_DEFAULT, consumers: int = BACKWARD_CONSUMERS,
             interpret: bool = False) -> Gradients:
    """The template's backward on this mainloop (`gradients`) from a training forward's `residuals` (its
    maxima, denominators, anchor keys and split sums): D about each query's anchor from the split sums
    (`algebra.split_delta`), a dQ launch and a dK/dV launch, and `dq_finish`'s refinement between them, in
    the form `accurate` names. Full and causal calls of bf16 operands, no hooks."""
    if hooks is not None and hooks != Hooks():
        raise NotImplementedError("the sm90 backward takes no hooks yet")
    if spans.form not in ("full", "causal"):
        raise NotImplementedError(f"the sm90 backward takes full and causal calls, not {spans.form!r} ones")
    if jnp.dtype(query.dtype) != jnp.dtype(jnp.bfloat16):
        raise NotImplementedError(f"the sm90 backward takes bf16 operands, not {jnp.dtype(query.dtype)}")
    width = query.shape[3]
    scale = width ** -0.5 if scale is None else scale
    how = algebra.recompute(algebra.Forward("full", 0, ids=False, scale=scale, natural=natural))
    launch = functools.partial(_launch_backward, how=how, causal=spans.form == "causal",
                               keys_count=spans.keys, accurate=accurate, tiles=tiles, consumers=consumers,
                               interpret=interpret)
    block = max(consumers * tiles.block_q, consumers * tiles.block_k)
    return gradients(launch, query, key, value, spans, hooks, dout, residuals, block=block, scale=scale)


def _launch_backward(query, key, value, dout, peak0, total0, initial_term, maximum, key_index, anchor_value,
                     shift, others, own, others_total, forward_total, start, end, query_ids, key_ids,
                     score_params, *, score_mod, score_kinds, score_dtypes, scale: float,
                     how: algebra.Recompute, causal: bool, keys_count: int, accurate: bool, tiles: Tiles,
                     consumers: int, interpret: bool):
    """`gradients`' launch on this mainloop: D from the forward's split sums, the dQ program's sums, their
    finish (`algebra.dq_finish`) and the dK/dV program under its refined D, each query's anchor key as the
    first pass chose it."""
    del start, end, query_ids, key_ids, score_params, score_mod, score_kinds, score_dtypes
    width = query.shape[-1]
    # The family's swizzled tiles hold whole rows of `FAMILY.rows` 16-bit entries.
    padded = [jnp.pad(array, ((0, 0),) * 3 + ((0, -array.shape[3] % FAMILY.rows),))
              for array in (query, key, value, dout)]
    _, state_total = algebra.off_anchor_start(peak0, total0, maximum, value.shape[-1], shift)
    cotangent = jnp.swapaxes(dout, 1, 2).astype(jnp.float32)
    anchor, delta, chosen = algebra.split_delta(others, own, others_total, forward_total, state_total,
                                                jnp.swapaxes(anchor_value, 1, 2), cotangent, key_index, shift)
    shared = {"how": how, "scale": scale, "causal": causal, "keys_count": keys_count, "tiles": tiles,
              "accurate": accurate, "consumers": consumers, "interpret": interpret}
    dq_sum, keyed, rows = _dq_launch(*padded, maximum, forward_total, anchor, delta, chosen, **shared)
    sums = algebra.DqSums(jnp.swapaxes(dq_sum, 1, 2), rows[0], rows[1], (), (), jnp.swapaxes(keyed, 1, 2),
                          rows[2], rows[3])
    dq, refined, denominator, mass, _ = algebra.dq_finish(sums, forward_total, state_total, peak0,
                                                          (total0, initial_term), maximum, anchor, delta,
                                                          accurate=accurate)
    dk, dv = _dkv_launch(*padded, maximum, denominator, anchor, refined, chosen, **shared)
    return (jnp.swapaxes(dq, 1, 2)[..., :width], refined, anchor, maximum, denominator, mass, chosen,
            dk[..., :width], dv[..., :value.shape[-1]])
