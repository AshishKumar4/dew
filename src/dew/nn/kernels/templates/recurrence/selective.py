"""Mamba's selective scan on the recurrence template's Mosaic GPU machinery (ledger 1300): a decay each
channel and state, `h_t = exp(dt_t A) h_{t-1} + dt_t x_t B_t`, `y_t = C_t . h_t + D x_t`
(`dew.nn.mixers.mamba`'s `selective_scan`), which no chunk product computes: the decay between two tokens
depends on the channel and the state together, so a chunk's pairs cost `C` times the scan's work.

A program holds a row's block of `block` channels, one a thread of its consumer warpgroup, each thread its
channel's `N` states in registers, and walks the row's chunks in turn on its family's streams (`Family.walk`:
TMA rings on sm90, `emit_pipeline` on sm80), a chunk's tokens one by one. Its output is staged a token at a
time and stored a chunk at a time; with `entered`, each chunk's entering state too, which the backward
replays its chunk from."""

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import cost, mosaic, resources

LOG2E = math.log2(math.e)
"""`exp(z) = exp2(z log2(e))`: the scan's decays are exp2 of the rate scaled by it."""

RESET_WIDTH = 32
"""The columns a row's resets take in their stream, every column the token's flag: the narrowest an int8 tile
both families stage (an sm80 stage swizzles 32 bytes at least, a TMA box takes 16)."""


@dataclasses.dataclass(frozen=True)
class Scan:
    """A selective scan's geometry: `rows` sequences of `length` tokens (a whole number of `chunk`s) of
    `channels` channels and `states` states each, `block` channels a program, on `family`'s Mosaic GPU
    mainloop (interpreted where `interpret`); `packed` where tokens reset their state (a document's first),
    `entered` where each chunk's entering state is kept for the backward."""

    rows: int
    length: int
    channels: int
    states: int
    chunk: int
    block: int
    family: mosaic.Family
    interpret: bool = False
    packed: bool = False
    entered: bool = False

    @property
    def chunks(self) -> int:
        return self.length // self.chunk

    @property
    def blocks(self) -> int:
        return self.channels // self.block


def _down(i):
    """A stream's step `i`: its view's `i`-th block of rows, the whole width."""
    return i, 0


UNROLLED = 8
"""The tokens a chunk's loop takes a turn: each token's step has too little independent work to hide its
latencies alone. Forward at chunk 32, 16 states, against mamba_ssm's selective_scan_fn in the same job
(8x2048x5120, 2x4096x8192): on an H100 80GB HBM3 0.742 and 0.806 ms against 1.311 and 1.054 (rolled 0.784
and 0.927, JS-selbar2-h100-894dfdf30); on an A100 SXM4 1.389 and 1.998 against 2.970 and 2.132
(JS-selbar-a100-894dfdf30). 2 and 16 tokens a turn took longer, and the whole chunk longer still
(JS-selunroll-h100-894dfdf30)."""


PARAMETER_ROWS = 8
"""The rows a program's parameter tile takes past its rates: the skip's, padded to the 8-row tiles a stream's
rows come in."""


def forward(x, dt, b, c, parameters, state, resets, scan: Scan):
    """The scan's outputs from padded operands: `x`, `dt` `[R, T, D]`, `b`, `c` `[R, T, N]`, `parameters`
    `[N + PARAMETER_ROWS, D]` fp32 (the rates `A log2(e)` transposed, then the skip `D`), `state` `[R, N, D]`
    fp32 and `resets` `[R, T, RESET_WIDTH]` int8 (a token's flag in every column; ignored unless
    `scan.packed`). Returns `y` `[R, T, D]` in `x`'s dtype, the final state `[R, N, D]` fp32 and, with
    `scan.entered`, each chunk's entering state `[R, chunks, N, D]` fp32 (None without)."""
    family, chunk, block, states = scan.family, scan.chunk, scan.block, scan.states
    blocks = {"x": ((chunk, block), x.dtype), "dt": ((chunk, block), dt.dtype),
              "b": ((chunk, states), b.dtype), "c": ((chunk, states), c.dtype)}
    if scan.packed:
        blocks["resets"] = ((chunk, RESET_WIDTH), jnp.int8)
    once = {"parameters": ((states + PARAMETER_ROWS, block), jnp.float32),
            "entering": ((states, block), jnp.float32)}
    # Each tile read a row at a time, at a traced row: staged plain (`mosaic.UNSWIZZLED`).
    streams = {"walked": family.streams(blocks, 2, 1, swizzles=dict.fromkeys(blocks, mosaic.UNSWIZZLED)),
               "once": family.streams(once, 1, 1, swizzles=dict.fromkeys(once, mosaic.UNSWIZZLED))}
    out_type = [jax.ShapeDtypeStruct(x.shape, x.dtype),
                jax.ShapeDtypeStruct((scan.rows, states, scan.channels), jnp.float32)]
    if scan.entered:
        out_type.append(jax.ShapeDtypeStruct((scan.rows, scan.chunks, states, scan.channels), jnp.float32))

    def entry(x_ref, dt_ref, b_ref, c_ref, parameters_ref, state_ref, resets_ref, y_ref, final_ref,
              *entered_ref):
        row, part = lax.axis_index("rows"), lax.axis_index("blocks")
        columns = pl.ds(part * block, block)
        # Each view the program's own columns, its steps down its rows (an sm80 pipeline's index map takes
        # no traced block index).
        views = {"x": (x_ref.at[row, :, columns], _down), "dt": (dt_ref.at[row, :, columns], _down),
                 "b": (b_ref.at[row], _down), "c": (c_ref.at[row], _down)}
        if scan.packed:
            views["resets"] = (resets_ref.at[row], _down)
        once_views = {"parameters": (parameters_ref.at[:, columns], _down),
                      "entering": (state_ref.at[row, :, columns], _down)}

        def body(staged, *, streams):
            out_smem, state_smem = staged
            walked = family.opened(streams["walked"], views)
            given = family.opened(streams["once"], once_views)

            def consumer(wg):
                def taken(index, slots, carry, accumulators):
                    del index, carry
                    parameters, entering = slots["parameters"], slots["entering"]
                    return ([parameters[n, :] for n in range(states)], parameters[states, :],
                            [entering[n, :] for n in range(states)]), accumulators

                vector = jnp.zeros((block,), jnp.float32)
                (rates, skip, h0), _ = family.walk(family.fed(given, once_views, 0, 1, wg), ((taken, 0, 1),),
                                                   ([vector] * states, vector, [vector] * states), (),
                                                   scan.interpret)

                def step(index, slots, h, accumulators):
                    if scan.entered:
                        family.stored(list(h), [entered_ref[0].at[row, index, n, columns]
                                                for n in range(states)], smem=state_smem)
                    xs, dts, bs, cs = (slots[name] for name in ("x", "dt", "b", "c"))
                    resets = slots["resets"] if scan.packed else None

                    def token(t, h):
                        step_ = dts[t, :].astype(jnp.float32)
                        value = xs[t, :].astype(jnp.float32)
                        written = step_ * value
                        kept = 1.0 if resets is None else 1.0 - resets[t, 0].astype(jnp.float32)
                        h = [jnp.exp2(step_ * rates[n]) * kept * h[n] + written * bs[t, n].astype(jnp.float32)
                             for n in range(states)]
                        read = functools.reduce(jnp.add, [cs[t, n].astype(jnp.float32) * h[n]
                                                          for n in range(states)])
                        out_smem[t, :] = (read + skip * value).astype(out_smem.dtype)
                        return h

                    h = lax.fori_loop(0, chunk, token, h, unroll=UNROLLED)
                    family.stored([out_smem[...]], [y_ref.at[row, pl.ds(index * chunk, chunk), columns]],
                                  smem=out_smem)
                    return h, accumulators

                h, _ = family.walk(family.fed(walked, views, 0, scan.chunks, wg), ((step, 0, scan.chunks),),
                                   h0, (), scan.interpret)
                family.stored(list(h), [final_ref.at[row, n, columns] for n in range(states)],
                              smem=state_smem)
                family.store_wait(0)

            def producer():
                family.stream(given, once_views, 0, 1)
                family.stream(walked, views, 0, scan.chunks)

            family.roles(1, consumer, producer)

        allocations = (plgpu.SMEM((chunk, block), x.dtype), plgpu.SMEM((block,), jnp.float32))
        family.scoped(body, allocations, streams=streams)

    entry.__name__ = "selective_scan"
    launched = family.kernel(entry, out_type=tuple(out_type), grid=(scan.rows, scan.blocks),
                             grid_names=("rows", "blocks"), consumers=1, interpret=scan.interpret,
                             name="selective_scan")
    outs = launched(x, dt, b, c, parameters, state, resets)
    return (*outs, None) if not scan.entered else tuple(outs)


def padding(family: mosaic.Family, columns: int, dtype) -> int:
    """The elements each row of a shared tile `columns` wide of `dtype` pads by, so the family's load of
    it in its accumulator's layout (`Family.tile_of`) reads every bank once a wavefront: a warp's lanes read
    a row's fragment, as wide as the layout's last two-dimensional tile, and the rows one wavefront serves
    then start a fragment's bytes apart modulo the banks' width (`cost.SHARED_BYTES_PER_CLOCK`). A tile
    written a row at a traced token cannot be swizzled (Mosaic GPU's tiling takes no row index). Unpadded,
    a 128-column fp32 tile's rows sat 512 bytes apart, on the same banks: 251M of the backward's 608M
    shared-load wavefronts conflicted (an H100, mamba1-ncu-ref-h100-1d3b0f647)."""
    fragment = next(tile for tile in reversed(family.layout.to_mgpu().tiling.tiles) if len(tile) == 2)[-1]
    size = jnp.dtype(dtype).itemsize
    return (fragment - columns) * size % cost.SHARED_BYTES_PER_CLOCK // size


def group(scan: "Scan", streamed: int) -> int:
    """The states a backward chunk replays at once: the most, a power of two dividing the states, whose
    per-token states `[C + 1, group, block]` fp32 and one `[2 C group, block]` tile of the channels' terms
    (each token's and state's for the readout's and the write's gradients, its rows padded, `padding`), as
    many rows as whole tiles of the accumulator's layout, fit a program's share of the family's shared
    memory beside the chunk's streams (`streamed` bytes) and the tokens' two `[C, block]` gradients: two
    programs' shares where any group fits them (a program's walk stalls on its own barriers alone), else
    one's. The fewer the groups, the fewer the replays and sums."""
    family, chunk, block = scan.family, scan.chunk, scan.block
    rows = family.layout.to_mgpu().base_tile_shape[0]
    terms = 2 * chunk * (block + padding(family, block, jnp.float32)) * 4
    per_state = (chunk + 1) * block * 4 + terms
    fixed = streamed + 2 * chunk * block * 4 + block * 4
    held = [size for size in (8, 4, 2, 1) if scan.states % size == 0 and 2 * chunk * size % rows == 0]
    for programs in (2, 1):
        budget = resources.program_shared(family.most_shared, programs)
        fitting = [size for size in held if fixed + size * per_state <= budget]
        if fitting:
            return max(fitting)
    return min(held)


@dataclasses.dataclass(frozen=True)
class _Chunk:
    """A backward step's chunk: its streams' slots (`x`, `dt` `[C, block]`, `b`, `c` `[C, N]`, `dy`, and
    the resets where the scan is packed), the program's rates, and its shared tiles: each token's kept states
    `[C + 1, size, block]`, the channels' terms `[2 C size, block]` (the readout's rows, then the write's),
    and the tokens' write and decay gradients' sums `[C, block]`."""

    xs: object
    dts: object
    bs: object
    cs: object
    dys: object
    resets: object
    rates: list
    kept: object
    terms: object
    writes: object
    decays: object
    size: int

    @property
    def chunk(self) -> int:
        return self.xs.shape[0]

    def row(self, ref, t) -> jax.Array:
        """Token `t`'s row of a stream's slot, fp32."""
        return ref[t, :].astype(jnp.float32)

    def decay(self, t, n) -> jax.Array:
        """Token `t`'s decay of state `n`, `exp2(dt rate)`, zero at a document's first token."""
        kept = 1.0 if self.resets is None else 1.0 - self.resets[t, 0].astype(jnp.float32)
        return jnp.exp2(self.row(self.dts, t) * self.rates[n]) * kept

    def group(self, own: range, start: list, g: list, d_rates: list) -> tuple[list, list]:
        """The group `own`'s states through the chunk from `start`, each token's kept and its readout's
        terms `dy_t h_t` beside them, then walked back from the cotangents `g` leaving the chunk, `g_t = C_t
        dy_t + a_{t+1} g_{t+1}` for the decay `a_t`: the write's terms `g_t w_t`, each token's write and decay
        gradients' sums `sum_n g_t B_t` and `sum_n g_t h_{t-1} a_t rate`, and the rates' gradients `d_rates`
        summed. Returns the cotangents entering the chunk and the rates' gradients."""
        chunk, size = self.chunk, self.size

        def replay(t, h):
            written = self.row(self.dts, t) * self.row(self.xs, t)
            out_grad = self.row(self.dys, t)
            h = [self.decay(t, n) * h[i] + written * self.bs[t, n].astype(jnp.float32)
                 for i, n in enumerate(own)]
            for i in range(size):
                self.kept[t + 1, i, :] = h[i]
                self.terms[t * size + i, :] = out_grad * h[i]
            return h

        def back(i, carried):
            g, d_rates = carried
            t = chunk - 1 - i
            step = self.row(self.dts, t)
            written = step * self.row(self.xs, t)
            out_grad = self.row(self.dys, t)
            g = [g[j] + self.cs[t, n].astype(jnp.float32) * out_grad for j, n in enumerate(own)]
            writes, decays = self.writes[t, :], self.decays[t, :]
            for j, n in enumerate(own):
                a = self.decay(t, n)
                writes = writes + g[j] * self.bs[t, n].astype(jnp.float32)
                through = g[j] * self.kept[t, j, :] * a
                decays = decays + through * self.rates[n]
                d_rates[j] = d_rates[j] + through * step
                self.terms[(chunk + t) * size + j, :] = g[j] * written
                g[j] = g[j] * a
            self.writes[t, :] = writes
            self.decays[t, :] = decays
            return g, d_rates

        # The entering state the replay's first row, each token's a row past it.
        for i in range(size):
            self.kept[0, i, :] = start[i]
        lax.fori_loop(0, chunk, replay, start, unroll=UNROLLED)
        return lax.fori_loop(0, chunk, back, (g, d_rates))

    def tokens(self, skip, d_skip) -> jax.Array:
        """Each token's write and decay gradients in place of their sums (`exp2(dt rate)`'s derivative in
        `dt` is `rate ln(2)` times it), and the skip's gradient `d_skip` summed."""
        def token(t, d_skip):
            out_grad, value = self.row(self.dys, t), self.row(self.xs, t)
            writes, decays = self.writes[t, :], self.decays[t, :]
            self.writes[t, :] = skip * out_grad + writes * self.row(self.dts, t)
            self.decays[t, :] = writes * value + decays * math.log(2.0)
            return d_skip + out_grad * value

        return lax.fori_loop(0, self.chunk, token, d_skip, unroll=UNROLLED)


def backward(x, dt, b, c, parameters, entered, resets, dy, dfinal, scan: Scan):
    """The scan's gradients from `forward`'s padded operands, its chunks' entering states `entered` `[R,
    chunks, N, D]` fp32 and the cotangents `dy` `[R, T, D]`, `dfinal` `[R, N, D]` fp32: `(dx, ddt [R, T, D],
    sums [R, blocks, chunks, N // size, 2 C size], dA [R, N, D], dD [R, 8, D], dstate [R, N, D])`, fp32,
    `sums` each program's channels' sums of the readout's terms, then the write's (the caller sums the
    programs), `dA` and `dD` each row's. A program walks its row's chunks from the last, a group of `size`
    states at a time (`group`, `_Chunk.group`), each group's entering states read from global memory a group
    ahead; the rates, the skip and `dfinal` are read once into registers."""
    family, chunk, block, states = scan.family, scan.chunk, scan.block, scan.states
    blocks = {"x": ((chunk, block), x.dtype), "dt": ((chunk, block), dt.dtype),
              "b": ((chunk, states), b.dtype), "c": ((chunk, states), c.dtype),
              "dy": ((chunk, block), dy.dtype)}
    if scan.packed:
        blocks["resets"] = ((chunk, RESET_WIDTH), jnp.int8)
    # One stage of each stream: two put an H100's program past its shared memory beside the replay's tiles.
    streamed = sum(math.prod(shape) * jnp.dtype(dtype).itemsize for shape, dtype in blocks.values())
    size = group(scan, streamed)
    groups = states // size
    pad = padding(family, block, jnp.float32)
    streams = {"walked": family.streams(blocks, 1, 1, swizzles=dict.fromkeys(blocks, mosaic.UNSWIZZLED))}
    shape = (scan.rows, scan.length, scan.channels)
    partial = (scan.rows, scan.blocks, scan.chunks, groups, 2 * chunk * size)
    out_type = (jax.ShapeDtypeStruct(shape, jnp.float32), jax.ShapeDtypeStruct(shape, jnp.float32),
                jax.ShapeDtypeStruct(partial, jnp.float32),
                jax.ShapeDtypeStruct((scan.rows, states, scan.channels), jnp.float32),
                jax.ShapeDtypeStruct((scan.rows, PARAMETER_ROWS, scan.channels), jnp.float32),
                jax.ShapeDtypeStruct((scan.rows, states, scan.channels), jnp.float32))
    chunks = scan.chunks

    def reversed_steps(i):
        """A stream's step `i`: the chunk `chunks - 1 - i`'s rows, the whole width."""
        return chunks - 1 - i, 0

    def entry(x_ref, dt_ref, b_ref, c_ref, parameters_ref, entered_ref, resets_ref, dy_ref, dfinal_ref,
              dx_ref, ddt_ref, sums_ref, da_ref, dd_ref, dstate_ref):
        row, part = lax.axis_index("rows"), lax.axis_index("blocks")
        columns = pl.ds(part * block, block)
        views = {"x": (x_ref.at[row, :, columns], reversed_steps),
                 "dt": (dt_ref.at[row, :, columns], reversed_steps),
                 "b": (b_ref.at[row], reversed_steps), "c": (c_ref.at[row], reversed_steps),
                 "dy": (dy_ref.at[row, :, columns], reversed_steps)}
        if scan.packed:
            views["resets"] = (resets_ref.at[row], reversed_steps)

        def body(staged, *, streams):
            kept_smem, summed_smem, writes_smem, decays_smem, vector_smem = staged
            walked = family.opened(streams["walked"], views)
            terms = summed_smem.at[:, pl.ds(0, block)]

            def consumer(wg):
                # Each row's channels; the padding's columns stay zero, which the sums add.
                summed_smem[...] = jnp.zeros(summed_smem.shape, jnp.float32)
                rates = [parameters_ref[n, columns] for n in range(states)]
                skip = parameters_ref[states, columns]
                g0 = [dfinal_ref[row, n, columns] for n in range(states)]

                def step(index, slots, carry, accumulators):
                    g, d_rates, d_skip = carry
                    n_chunk = chunks - 1 - index
                    walk = _Chunk(*(slots[name] for name in ("x", "dt", "b", "c", "dy")),
                                  slots["resets"] if scan.packed else None, rates, kept_smem, terms,
                                  writes_smem, decays_smem, size)
                    writes_smem[...] = jnp.zeros(writes_smem.shape, jnp.float32)
                    decays_smem[...] = jnp.zeros(decays_smem.shape, jnp.float32)

                    def entering(k):
                        own = range(k * size, (k + 1) * size)
                        return [entered_ref[row, n_chunk, n, columns] for n in own]

                    # The next group's entering states read while this one replays and walks back.
                    ahead = entering(0)
                    for k in range(groups):
                        own = range(k * size, (k + 1) * size)
                        start = ahead
                        if k + 1 < groups:
                            ahead = entering(k + 1)
                        g_own, d_own = walk.group(own, start, [g[n] for n in own],
                                                  [d_rates[n] for n in own])
                        for j, n in enumerate(own):
                            g[n], d_rates[n] = g_own[j], d_own[j]
                        # The group's readout's and write's terms summed over the channels at once.
                        sums = family.tile_of(summed_smem).sum(axis=1)
                        plgpu.store(sums_ref.at[row, part, n_chunk, k], sums, optimized=False)

                    d_skip = walk.tokens(skip, d_skip)
                    span = pl.ds(n_chunk * chunk, chunk)
                    family.stored([writes_smem[...]], [dx_ref.at[row, span, columns]], smem=writes_smem)
                    family.stored([decays_smem[...]], [ddt_ref.at[row, span, columns]], smem=decays_smem)
                    return (g, d_rates, d_skip), accumulators

                vector = jnp.zeros((block,), jnp.float32)
                carry = (list(g0), [vector] * states, vector)
                (g, d_rates, d_skip), _ = family.walk(family.fed(walked, views, 0, chunks, wg),
                                                      ((step, 0, chunks),), carry, (), scan.interpret)
                family.stored(list(g), [dstate_ref.at[row, n, columns] for n in range(states)],
                              smem=vector_smem)
                # `exp(dt A)`'s derivative in `A` is `dt` times it: `d_rates` is `A`'s gradient.
                family.stored(list(d_rates), [da_ref.at[row, n, columns] for n in range(states)],
                              smem=vector_smem)
                family.stored([d_skip], [dd_ref.at[row, 0, columns]], smem=vector_smem)
                family.store_wait(0)

            def producer():
                family.stream(walked, views, 0, chunks)

            family.roles(1, consumer, producer)

        allocations = (plgpu.SMEM((chunk + 1, size, block), jnp.float32),
                       plgpu.SMEM((2 * chunk * size, block + pad), jnp.float32),
                       plgpu.SMEM((chunk, block), jnp.float32), plgpu.SMEM((chunk, block), jnp.float32),
                       plgpu.SMEM((block,), jnp.float32))
        family.scoped(body, allocations, streams=streams)

    entry.__name__ = "selective_scan_backward"
    launched = family.kernel(entry, out_type=out_type, grid=(scan.rows, scan.blocks),
                             grid_names=("rows", "blocks"), consumers=1, interpret=scan.interpret,
                             name="selective_scan_backward")
    return launched(x, dt, b, c, parameters, entered, resets, dy, dfinal)


def selective_scan(x, dt, A, B, C, D, state=None, starts=None, *, family: mosaic.Family, chunk: int = 32,
                   block: int = 128, interpret: bool = False):
    """`dew.nn.mixers.mamba.selective_scan` on the template: `x`, `dt` `[B, S, D]` (after the softplus),
    `A` `[D, N]`, `B`, `C` `[B, S, N]`, `D` `[D]`, `state` `[B, D, N]` or None, `starts` `[B, S]` or None.
    Returns `(y [B, S, D], final state [B, D, N])` in `x`'s dtype. The sequence is padded to whole chunks
    with a zero step, which neither decays nor writes."""
    rows, length, channels = x.shape
    states = A.shape[-1]
    padded = -(-length // chunk) * chunk

    def pad(t):
        return jnp.pad(t, ((0, 0), (0, padded - length)) + ((0, 0),) * (t.ndim - 2))

    resets = (jnp.zeros((rows, padded, RESET_WIDTH), jnp.int8) if starts is None else
              jnp.broadcast_to(pad(jnp.asarray(starts, jnp.int8))[..., None], (rows, padded, RESET_WIDTH)))
    entering = (jnp.zeros((rows, states, channels), jnp.float32) if state is None
                else jnp.swapaxes(jnp.asarray(state, jnp.float32), 1, 2))
    scan = Scan(rows, padded, channels, states, chunk, block, family, interpret, packed=starts is not None)
    y, final = _scan(pad(x), pad(dt), jnp.asarray(A, jnp.float32), pad(B), pad(C),
                     jnp.asarray(D, jnp.float32), entering, resets, scan)
    return y[:, :length], jnp.swapaxes(final, 1, 2).astype(x.dtype)


def _parameters(A, D, channels: int):
    """`forward`'s parameter tile: the rates `A log2(e)` transposed, then the skip, padded to the stream's
    rows."""
    return jnp.concatenate([(A * LOG2E).T, D[None], jnp.zeros((PARAMETER_ROWS - 1, channels), jnp.float32)])


@functools.partial(jax.custom_vjp, nondiff_argnums=(8,))
def _scan(x, dt, A, B, C, D, entering, resets, scan: Scan):
    """`forward` over padded operands, its gradient `backward`'s: `(y, final [R, N, D] fp32)`."""
    y, final, _ = forward(x, dt, B, C, _parameters(A, D, scan.channels), entering, resets, scan)
    return y, final


def _scan_forward(x, dt, A, B, C, D, entering, resets, scan: Scan):
    kept = dataclasses.replace(scan, entered=True)
    y, final, entered = forward(x, dt, B, C, _parameters(A, D, scan.channels), entering, resets, kept)
    return (y, final), (x, dt, A, B, C, D, entered, resets)


def _scan_backward(scan: Scan, residuals, grads):
    x, dt, A, B, C, D, entered, resets = residuals
    dy, dfinal = grads
    dx, ddt, sums, da, dd, dstate = backward(x, dt, B, C, _parameters(A, D, scan.channels), entered, resets,
                                             dy.astype(x.dtype), dfinal.astype(jnp.float32), scan)
    # Each group's readout's terms, then its write's.
    half = sums.shape[-1] // 2
    dc, db = sums[..., :half], sums[..., half:]

    def laid(partial):
        """Each program's `[chunks, N // size, C * size]` sums over its channels, summed over the programs,
        as `[R, T, N]`."""
        size = partial.shape[-1] // scan.chunk
        summed = partial.sum(1).reshape(scan.rows, scan.chunks, scan.states // size, scan.chunk, size)
        return jnp.swapaxes(summed, 2, 3).reshape(scan.rows, scan.length, scan.states)

    return (dx.astype(x.dtype), ddt.astype(dt.dtype), da.sum(0).T.astype(A.dtype), laid(db).astype(B.dtype),
            laid(dc).astype(C.dtype), dd[:, 0].sum(0).astype(D.dtype), dstate, None)


_scan.defvjp(_scan_forward, _scan_backward)
