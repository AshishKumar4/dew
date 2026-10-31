"""Mamba's selective scan on the recurrence template's Mosaic GPU machinery (ledger 1300): a decay each
channel and state, `h_t = exp(dt_t A) h_{t-1} + dt_t x_t B_t`, `y_t = C_t . h_t + D x_t`
(`dew.nn.mixers.mamba`'s `selective_scan`), which no chunk product computes: the decay between two tokens
depends on the channel and the state together, so a chunk's pairs cost `C` times the scan's work.

The forward's program holds a row's block of `block` channels, one a thread of its consumer warpgroup, each
thread its channel's `N` states in registers, and walks the row's chunks in turn on its family's streams
(`Family.walk`: TMA rings on sm90, `emit_pipeline` on sm80), a chunk's tokens one by one. Its output is
staged a token at a time and stored a chunk at a time; with `entered`, each chunk's entering state too,
which the backward starts its blocks from. The backward's program takes a row's channel, a warpgroup's
threads over a block of its tokens, each state scanned across them (`templates.scan`), as mamba_ssm's
backward does."""

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import mosaic, scan as block_scan
from dew.nn.kernels.templates.recurrence import mainloop

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


POSITIONS_PER_THREAD = 16
"""The tokens a backward thread takes, mamba_ssm's kNItems: its warpgroup's 2048 a block. ptxas fills the
registers a kernel leaves it (Mosaic GPU asks one program a multiprocessor, nvvm.minctasm 1, jaxlib's
mosaic/gpu/passes.cc:211), so fewer tokens hold no more programs: at 8 the backward needed 152 registers
spill-free and took 202, at 16 needed 224 and took 254, two programs a multiprocessor either way
(need-*-sm90 at 1213539d0 and 02c73b9f0). Three, mamba_ssm's kMinBlocks, would take a register cap
(`.maxnreg`, or minctasm 3) at 8 tokens, which jaxlib's compile does not take."""


def block_positions(length: int) -> int:
    """The tokens a backward block of a scan `length` tokens long takes: its threads'
    (`POSITIONS_PER_THREAD`) where the length is long enough, else the fewest whole tokens a thread covering
    it."""
    threads = block_scan.THREADS
    return min(threads * POSITIONS_PER_THREAD, -(-length // threads) * threads)


def backward(x, dt, b, c, parameters, entered, resets, dy, dfinal, scan: Scan):
    """The scan's gradients from `forward`'s operands, channels leading: `x`, `dt`, `dy` `[R, D, T]`, `b`,
    `c` `[R, N, T]`, `parameters` (`forward`'s), its chunks' entering states `entered` `[R, chunks, N, D]`
    fp32, `resets` `[R, T]` int8 and the final state's cotangent `dfinal` `[R, N, D]` fp32, `T` a whole
    number of blocks (`block_positions`). Returns `(dx, ddt [R, D, T], db, dc [R, N, T], dA [R, N, D],
    dD [R, D], dstate [R, N, D])`, fp32.

    A program takes a row and a channel, a warpgroup's threads its block's tokens (`block_scan.blocked`), and
    walks the row's blocks from the last, each state in turn: its states through the block from the
    block's entering state by a scan (`block_scan.linear_scan`, each token's state before it, `h_{t-1}`), then
    its cotangents back by a reverse one from the state's carried from the block after (`dfinal` past the
    last): `q_t = a_t (C_t dy_t + q_{t+1})`, the cotangent a token passes the state before it, so
    `g_t = C_t dy_t + q_{t+1}` the state's own, its decay `a_t = exp2(dt_t rate)` (zero at a document's
    first token). The channel's gradients sum over its states in registers; the states' B and C gradients
    sum over the channels in global memory, each program's added atomically (mamba_ssm's atomicAdd), a
    block's tokens across the threads so a warp's adds are adjacent (cub's BlockedToStriped)."""
    rows, channels, length = x.shape
    states, positions = b.shape[1], block_positions(length)
    blocks, per = length // positions, positions // scan.chunk
    interpret = scan.interpret
    layout = block_scan.blocked(positions)

    def entry(x_ref, dt_ref, b_ref, c_ref, parameters_ref, entered_ref, resets_ref, dy_ref, dfinal_ref,
              db_ref, dc_ref, dx_ref, ddt_ref, da_ref, dd_ref, dstate_ref, exchanged, staged, carried):
        row, channel = lax.axis_index("rows"), lax.axis_index("channels")

        def load(ref):
            return plgpu.load(ref, layout=layout, optimized=False).astype(jnp.float32)

        def added(ref, value):
            """`value` added to `ref` atomically, across the threads (`block_scan.to_striped`), a warp's adds
            adjacent."""
            if interpret:
                ref[...] = ref[...] + value
                return
            plgpu.atomic_add(ref, block_scan.to_striped(value, staged), optimized=False)

        def block(i, d_skip):
            k = blocks - 1 - i
            span = pl.ds(k * positions, positions)
            xs, dts, dys = (load(ref.at[row, channel, span]) for ref in (x_ref, dt_ref, dy_ref))
            kept = 1.0 - load(resets_ref.at[row, span]) if scan.packed else None

            def state(n, sums):
                # A state a turn, rolled: unrolled, the states' operands lived together and spilled 12 KB
                # a thread (255 registers, an H100's compile, scanbwd-ptxas-sm90 at 5de7abaa0).
                dx, ddt = sums
                rate = parameters_ref[n, channel]
                bs, cs = load(b_ref.at[row, n, span]), load(c_ref.at[row, n, span])
                a = jnp.exp2(dts * rate) if kept is None else jnp.exp2(dts * rate) * kept
                written = dts * xs * bs
                entering = entered_ref[row, k * per, n, channel]
                before, _ = block_scan.linear_scan(a, written, entering, exchanged, exclusive=True,
                                                   interpret=interpret)
                read = cs * dys
                after, carried[0, n] = block_scan.linear_scan(
                    a, a * read, carried[0, n], exchanged, reverse=True, exclusive=True, interpret=interpret)
                cotangent = read + after
                through = cotangent * before * a
                # The rate's gradient so far read before the sum's barriers, so no thread reads another's
                # write of it: read after, an A100's threads added the block's sum twice over (1.6e-2 of
                # dA from float64 against the rest's 4e-7, mamba1-scan-a100-7d3152837).
                so_far = carried[1, n]
                carried[1, n] = so_far + block_scan.summed(through * dts, exchanged, interpret=interpret)
                added(db_ref.at[row, n, span], cotangent * dts * xs)
                added(dc_ref.at[row, n, span], dys * (a * before + written))
                # `exp2(dt rate)`'s derivative in `dt` is `rate ln(2)` times it, in the rate `dt ln(2)`.
                return (dx + cotangent * bs * dts,
                        ddt + cotangent * bs * xs + through * (rate * math.log(2.0)))

            zeros = jnp.zeros((positions,), jnp.float32)
            dx, ddt = lax.fori_loop(0, states, state, (parameters_ref[states, channel] * dys, zeros))
            plgpu.store(dx_ref.at[row, channel, span], dx, optimized=False)
            plgpu.store(ddt_ref.at[row, channel, span], ddt, optimized=False)
            return d_skip + block_scan.summed(dys * xs, exchanged, interpret=interpret)

        # Each state's cotangent carried from block to block and its rate's gradient summed, in shared
        # memory, a state's a turn of the rolled loop.
        for n in range(states):
            carried[0, n] = dfinal_ref[row, n, channel]
            carried[1, n] = jnp.zeros((), jnp.float32)
        dd_ref[row, channel] = lax.fori_loop(0, blocks, block, jnp.zeros((), jnp.float32))
        for n in range(states):
            dstate_ref[row, n, channel] = carried[0, n]
            da_ref[row, n, channel] = carried[1, n]

    sized = jax.ShapeDtypeStruct
    out_type = (sized(x.shape, jnp.float32), sized(x.shape, jnp.float32),
                sized((rows, states, channels), jnp.float32), sized((rows, channels), jnp.float32),
                sized((rows, states, channels), jnp.float32))
    launched = mosaic.kernel(entry, out_type=out_type, grid=(rows, channels), grid_names=("rows", "channels"),
                             warpgroups=1, interpret=interpret, family=scan.family.name,
                             name="selective_scan_backward",
                             scratch=(block_scan.scratch(), block_scan.exchange_scratch(positions),
                                      plgpu.SMEM((2, states), jnp.float32)))
    db, dc = (jax.new_ref(jnp.zeros(b.shape, jnp.float32)) for _ in range(2))
    dx, ddt, da, dd, dstate = launched(x, dt, b, c, parameters, entered, resets, dy, dfinal, db, dc)
    return dx, ddt, jax.freeze(db), jax.freeze(dc), da, dd, dstate


def selective_scan(x, dt, A, B, C, D, state=None, starts=None, *, family: mosaic.Family, chunk: int = 32,
                   block: int = 128, interpret: bool = False):
    """`dew.nn.mixers.mamba.selective_scan` on the template: `x`, `dt` `[B, S, D]` (after the softplus),
    `A` `[D, N]`, `B`, `C` `[B, S, N]`, `D` `[D]`, `state` `[B, D, N]` or None, `starts` `[B, S]` or None.
    Returns `(y [B, S, D], final state [B, D, N])` in `x`'s dtype. The sequence is padded to whole chunks
    with a zero step, which neither decays nor writes."""
    rows, length, channels = x.shape
    states = A.shape[-1]
    padded = -(-length // chunk) * chunk

    pad = functools.partial(mainloop._padded, axis=1, size=padded)
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
    positions = block_positions(scan.length)
    padded = -(-scan.length // positions) * positions

    def leading(t):
        """`t` `[R, T, ...]` with its tokens last, padded to whole blocks: a zero token decays nothing,
        writes nothing and reads nothing."""
        return mainloop._padded(jnp.moveaxis(t, 1, -1), axis=t.ndim - 1, size=padded)

    dx, ddt, db, dc, da, dd, dstate = backward(
        leading(x), leading(dt), leading(B), leading(C), _parameters(A, D, scan.channels), entered,
        leading(resets[..., 0]), leading(dy.astype(x.dtype)), dfinal.astype(jnp.float32), scan)

    def trailing(t):
        """A gradient `[R, ..., T]` back as `[R, T, ...]`, the padding's tokens dropped."""
        return jnp.moveaxis(t[..., :scan.length], -1, 1)

    return (trailing(dx).astype(x.dtype), trailing(ddt).astype(dt.dtype), da.sum(0).T.astype(A.dtype),
            trailing(db).astype(B.dtype), trailing(dc).astype(C.dtype), dd.sum(0).astype(D.dtype), dstate,
            None)


_scan.defvjp(_scan_forward, _scan_backward)
