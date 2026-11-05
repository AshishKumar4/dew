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
from typing import ClassVar

import jax
import jax.numpy as jnp
import tokamax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import generation, recurrence_algebra as algebra, tuning
from dew.nn.kernels.templates import base, mosaic, report, resources, scan as block_scan
from dew.nn.kernels.templates.recurrence import mainloop

RESET_WIDTH = 32
"""The columns a row's resets take in their stream, every column the token's flag: the narrowest an int8 tile
both families stage (an sm80 stage swizzles 32 bytes at least, a TMA box takes 16)."""


@dataclasses.dataclass(frozen=True)
class Scan:
    """A selective scan's geometry: `rows` sequences of `length` tokens (a whole number of `chunk`s) of
    `channels` channels and `states` states each, `block` channels a program, on `family`'s Mosaic GPU
    mainloop (interpreted where `interpret`); `packed` where tokens reset their state (a document's first),
    `entered` where the entering state of each chunk opening a backward block (`block_positions`) is kept
    for the backward, which starts its blocks from them; the backward's tokens a thread and programs a
    multiprocessor (`backward_plan`); the launch tracing the forward (`report.launch_digest`), whose name the
    backward's kernel carries too (`report.relaunched`)."""

    rows: int
    length: int
    channels: int
    states: int
    chunk: int
    block: int
    family: mosaic.Family
    positions: int
    resident: int
    interpret: bool = False
    packed: bool = False
    entered: bool = False
    launch: str | None = None

    @property
    def chunks(self) -> int:
        return self.length // self.chunk

    @property
    def opening(self) -> int:
        """The chunks a backward block takes, its first's entering state the forward keeps (`entered`)."""
        return block_positions(self.length, self.positions) // self.chunk

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


def forward_scope(family: mosaic.Family, chunk: int, states: int, dtypes: tuple,
                  packed: bool) -> resources.Scope:
    """`forward`'s shared memory by lifetime at `chunk` tokens (`forward_allocations`), its streams where the
    family states they live (`mainloop.allocation_scope`, `Family.lifetimes`), no reduction across warps."""
    streams, allocations = forward_allocations(family, chunk, block_scan.THREADS, states, dtypes, packed)
    return mainloop.allocation_scope(family, streams, allocations, reduces=False)


def forward_allocations(family: mosaic.Family, chunk: int, block: int, states: int, dtypes: tuple,
                        packed: bool) -> tuple[dict, tuple]:
    """`forward`'s shared memory at `chunk` tokens and `block` channels a program: its streams (a ring of two
    chunks of `x`, `dt`, `B`, `C` and, `packed`, the resets, in `dtypes`' x, dt, B and C dtypes; the
    parameters and the entering state once) and its output chunk and state row, which `Selective` sizes its
    chunk by (`resources.shared_bytes`)."""
    x_dtype, dt_dtype, b_dtype, c_dtype = dtypes
    blocks = {"x": ((chunk, block), x_dtype), "dt": ((chunk, block), dt_dtype),
              "b": ((chunk, states), b_dtype), "c": ((chunk, states), c_dtype)}
    if packed:
        blocks["resets"] = ((chunk, RESET_WIDTH), jnp.int8)
    once = {"parameters": ((states + PARAMETER_ROWS, block), jnp.float32),
            "entering": ((states, block), jnp.float32)}
    # Each tile read a row at a time, at a traced row: staged plain (`mosaic.UNSWIZZLED`).
    streams = {"walked": family.streams(blocks, 2, 1, swizzles=dict.fromkeys(blocks, mosaic.UNSWIZZLED)),
               "once": family.streams(once, 1, 1, swizzles=dict.fromkeys(once, mosaic.UNSWIZZLED))}
    return streams, (plgpu.SMEM((chunk, block), x_dtype), plgpu.SMEM((block,), jnp.float32))


@tuning.lowered_once(("scan",))
def forward(x, dt, b, c, parameters, state, resets, scan: Scan):
    """The scan's outputs from padded operands: `x`, `dt` `[R, T, D]`, `b`, `c` `[R, T, N]`, `parameters`
    `[N + PARAMETER_ROWS, D]` fp32 (the rates `A log2(e)` transposed, then the skip `D`), `state` `[R, N, D]`
    fp32 and `resets` `[R, T, RESET_WIDTH]` int8 (a token's flag in every column; ignored unless
    `scan.packed`). Returns `y` `[R, T, D]` in `x`'s dtype, the final state `[R, N, D]` fp32 and, with
    `scan.entered`, the entering state of each chunk opening a backward block `[R, chunks / opening, N, D]`
    fp32 (`Scan.opening`; None without): every chunk's kept 64 times what 2048-token blocks of 32-token
    chunks read (4.3 GB at Falcon 7B's step, 2970)."""
    family, chunk, block, states = scan.family, scan.chunk, scan.block, scan.states
    streams, allocations = forward_allocations(family, chunk, block, states, (x.dtype, dt.dtype, b.dtype,
                                                                              c.dtype), scan.packed)
    out_type = [jax.ShapeDtypeStruct(x.shape, x.dtype),
                jax.ShapeDtypeStruct((scan.rows, states, scan.channels), jnp.float32)]
    if scan.entered:
        kept = -(-scan.chunks // scan.opening)
        out_type.append(jax.ShapeDtypeStruct((scan.rows, kept, states, scan.channels), jnp.float32))

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
                        @pl.when(index % scan.opening == 0)
                        def _():
                            kept = index // scan.opening
                            family.stored(list(h), [entered_ref[0].at[row, kept, n, columns]
                                                    for n in range(states)], smem=state_smem)
                    xs, dts, bs, cs = (slots[name] for name in ("x", "dt", "b", "c"))
                    resets = slots["resets"] if scan.packed else None

                    def token(t, h):
                        step_ = dts[t, :].astype(jnp.float32)
                        value = xs[t, :].astype(jnp.float32)
                        kept = None if resets is None else 1.0 - resets[t, 0].astype(jnp.float32)
                        h = algebra.selective_token(
                            h, [algebra.selective_decay(step_, rates[n], kept) for n in range(states)],
                            step_ * value, [bs[t, n].astype(jnp.float32) for n in range(states)])
                        c = [cs[t, n].astype(jnp.float32) for n in range(states)]
                        read = algebra.selective_read(h, c, skip, value)
                        out_smem[t, :] = read.astype(out_smem.dtype)
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

        family.scoped(body, allocations, streams=streams)

    entry.__name__ = "selective_scan"
    launched = family.kernel(entry, out_type=tuple(out_type), grid=(scan.rows, scan.blocks),
                             grid_names=("rows", "blocks"), consumers=1, interpret=scan.interpret,
                             name="selective_scan")
    outs = launched(x, dt, b, c, parameters, state, resets)
    return (*outs, None) if not scan.entered else tuple(outs)


PARTIALS = 32
"""The buffers B's and C's gradients are added into across the channels, a channel's its index modulo
them, then summed (XLA's reduction): each buffer adds a 32nd of the channels in turn, where one buffer's
sequential adds put B's gradient 1.95 times the portable scan's distance from float64 with a document
starting at every chunk (the interpreted regime test); the adds a buffer takes fewer, as contended."""


BACKWARD_VECTORS = 12
"""The token vectors a backward thread holds live at its state loop's peak, each its tokens' fp32 values: the
channel's sums over the states (dx, ddt), the block's operands (x, dt, dy), and the state's (B, C, its decay,
its write, the states before each token, C dy and the cotangent after each token), counted from `backward`."""

BACKWARD_FIXED = 56
"""The registers a backward thread holds apart from its token vectors: its refs' 64-bit addresses (20), the
block scan's running pair, lane and warp terms (8), the state loop's scalars (its rate, entering state,
carried cotangent, rate gradient so far and indices, 12) and a thread's control
(`resources.CONTROL_REGISTERS`). With `BACKWARD_VECTORS` it states ptxas's least spill-free cap on an H100 at
4, 8 and 16 tokens a thread, 96, 152 and 232, as 104, 152 and 248 (sel-need at f3040b154)."""

SCAN_STEPS = 16
"""The dependent steps of one block scan past its tokens' own (`templates.scan.linear_scan`): six shuffles
across the warp, three warpgroup barriers, a shared store and load of the warps' aggregates and four joins of
them, and the apply to the thread's tokens. The backward runs two a state, forward and reverse."""


def backward_registers(positions: int) -> int:
    """The registers a backward thread holds at `positions` tokens (`BACKWARD_VECTORS`, `BACKWARD_FIXED`)."""
    return BACKWARD_VECTORS * positions + BACKWARD_FIXED


def backward_programs(registers: int, target: generation.GpuTarget) -> int:
    """The programs of the backward, one warpgroup and no producer, a multiprocessor of `target` holds at once
    at `registers` a thread: the shared planner's residency (`resources.RegisterModel.resident`) within its
    resident threads."""
    held = target.architecture
    threads = held.resident_threads // block_scan.THREADS
    model = resources.RegisterModel(registers, held.thread_registers, 1, min(held.resident_blocks, threads),
                                    held.register_step, 0)
    return model.resident(registers)


def backward_plan(target: generation.GpuTarget) -> tuple[int, int]:
    """The backward's tokens a thread and the programs a multiprocessor holds of it at their registers
    (`backward_registers`, `backward_programs`), which its compile caps a thread's registers at
    (`mosaic.compiler_params`' `resident`): of the token counts whose registers a thread holds, the one whose
    dependent steps a token, its own two and its block's scans' `2 * SCAN_STEPS` shared by the thread's
    tokens, take the least time across the programs a multiprocessor holds at once, ties to more programs
    (a family whose threads issue their own steps hides their latency by resident warps first,
    `resources.hidden`). On an H100 the derived 8 tokens at three programs took Mamba-1's bf16 training step
    at 8x2048x5120 in 6.43 ms, 16 at two 6.48, 4 at four 7.48, and fewer programs than the registers hold
    longer at each (8 at two 8.10, 4 at three 8.53): ptxas filled the registers a cap left it, and the step
    slowed (sel-sweep-sxm at f3040b154)."""
    plans = []
    positions = 1
    while backward_registers(positions) <= target.architecture.thread_registers:
        programs = backward_programs(backward_registers(positions), target)
        if programs:
            plans.append(((2 * positions + 2 * SCAN_STEPS) / (positions * programs), -programs, positions))
        positions *= 2
    _, programs, positions = min(plans)
    return positions, -programs


def block_positions(length: int, positions: int) -> int:
    """The tokens a backward block of a scan `length` tokens long takes: its threads' `positions` each where
    the length is long enough, else the fewest whole tokens a thread covering it."""
    threads = block_scan.THREADS
    return min(threads * positions, -(-length // threads) * threads)


@tuning.lowered_once(("scan",))
def backward(x, dt, b, c, parameters, entered, resets, dy, dfinal, scan: Scan):
    """The scan's gradients from `forward`'s operands, channels leading: `x`, `dt`, `dy` `[R, D, T]`, `b`,
    `c` `[R, N, T]`, `parameters` (`forward`'s), its blocks' entering states `entered` `[R, blocks, N, D]`
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
    states, positions = b.shape[1], block_positions(length, scan.positions)
    blocks = length // positions
    interpret = scan.interpret
    layout = block_scan.blocked(positions)

    def entry(x_ref, dt_ref, b_ref, c_ref, parameters_ref, entered_ref, resets_ref, dy_ref, dfinal_ref,
              dbc_ref, dx_ref, ddt_ref, da_ref, dd_ref, dstate_ref, exchanged, staged, carried):
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
                a = algebra.selective_decay(dts, rate, kept)
                written = dts * xs * bs
                entering = entered_ref[row, k, n, channel]
                before, _ = block_scan.linear_scan(a, written, entering, exchanged, exclusive=True,
                                                   interpret=interpret)
                read = cs * dys
                after, carried[0, n] = block_scan.linear_scan(
                    a, a * read, carried[0, n], exchanged, reverse=True, exclusive=True, interpret=interpret)
                d_value, d_step, d_b, d_c, d_rate = algebra.selective_adjoint(
                    read + after, before, a, written, dts, xs, bs, dys, rate)
                # The rate's gradient so far read before the sum's barriers, so no thread reads another's
                # write of it: read after, an A100's threads added the block's sum twice over (1.6e-2 of
                # dA from float64 against the rest's 4e-7, mamba1-scan-a100-7d3152837).
                so_far = carried[1, n]
                carried[1, n] = so_far + block_scan.summed(d_rate, exchanged, interpret=interpret)
                partial = channel % PARTIALS
                # B's and C's gradients one ref, each its place along its leading axis, indexed once.
                added(dbc_ref.at[0, partial, row, n, span], d_b)
                added(dbc_ref.at[1, partial, row, n, span], d_c)
                return dx + d_value, ddt + d_step

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
                             name="selective_scan_backward", resident=scan.resident,
                             scratch=(block_scan.scratch(), block_scan.exchange_scratch(positions),
                                      plgpu.SMEM((2, states), jnp.float32)))
    def accumulated(refs):
        """The launch's outputs written to `refs`, B's and C's gradients added to theirs from zero, one ref
        of both (a leading axis of them: two refs of equal zeros are one value to XLA, which a fusion
        wrapping the launch may pass once for both, row 3180, `mosaic.checked`): state the run discharges
        (`pl.run_state`), as a kernel's derivative inside a custom JVP (`generation.linear_tangent`) can
        create no Ref of its own. A whole-array write of a launch's output XLA folds to the output itself."""
        dbc_ref, *outputs = refs
        found = launched(x, dt, b, c, parameters, entered, resets, dy, dfinal, dbc_ref)
        for ref, value in zip(outputs, found, strict=True):
            ref[...] = value

    zeros = (jnp.zeros((2, PARTIALS, *b.shape), jnp.float32),
             *(jnp.zeros(t.shape, t.dtype) for t in out_type))
    dbc, dx, ddt, da, dd, dstate = pl.run_state(accumulated)(zeros)
    return dx, ddt, dbc[0].sum(0), dbc[1].sum(0), da, dd, dstate


CHUNK = 32
"""The fewest tokens a forward chunk takes, a ring stage of its operands whose entering state the forward
keeps for each backward block it opens (`Scan.opening`), and the fewest a call takes (a decode step's are
the portable math's, `recurrence_chunk.SELECTIVE_SUPPORTS`)."""


def selective_scan(x, dt, A, B, C, D, state=None, starts=None, *, family: mosaic.Family, chunk: int = CHUNK,
                   block: int = block_scan.THREADS, interpret: bool = False,
                   plan: tuple[int, int] | None = None):
    """`dew.nn.mixers.mamba.selective_scan` on the template: `x`, `dt` `[B, S, D]` (after the softplus),
    `A` `[D, N]`, `B`, `C` `[B, S, N]`, `D` `[D]`, `state` `[B, D, N]` or None, `starts` `[B, S]` or None.
    Returns `(y [B, S, D], final state [B, D, N])` in `x`'s dtype. The sequence is padded to whole chunks
    with a zero step, which neither decays nor writes. `chunk` divides the backward's block
    (`block_positions`), which starts from a chunk's entering state the forward keeps."""
    # The backward's plan its caller derived (`Selective.derived`), else this device's, interpreted the
    # interpreted bundle's generation's (`generation.INTERPRETED_GENERATION`).
    positions, resident = plan or backward_plan(generation.gpu_target(
        generation.INTERPRETED_GENERATION if interpret else None))
    if block_scan.THREADS * positions % chunk:
        raise ValueError(f"a chunk of {chunk} tokens divides no backward block of "
                         f"{block_scan.THREADS * positions}: its blocks start at no kept state")
    rows, length, channels = x.shape
    states = A.shape[-1]
    padded = -(-length // chunk) * chunk

    pad = functools.partial(mainloop.zero_padded, axis=1, size=padded)
    resets = (jnp.zeros((rows, padded, RESET_WIDTH), jnp.int8) if starts is None else
              jnp.broadcast_to(pad(jnp.asarray(starts, jnp.int8))[..., None], (rows, padded, RESET_WIDTH)))
    entering = (jnp.zeros((rows, states, channels), jnp.float32) if state is None
                else jnp.swapaxes(jnp.asarray(state, jnp.float32), 1, 2))
    scan = Scan(rows, padded, channels, states, chunk, block, family, positions, resident, interpret,
                packed=starts is not None, launch=report.launch_digest())
    y, final = _scan(pad(x), pad(dt), jnp.asarray(A, jnp.float32), pad(B), pad(C),
                     jnp.asarray(D, jnp.float32), entering, resets, scan)
    return y[:, :length], jnp.swapaxes(final, 1, 2).astype(x.dtype)


def _parameters(A, D, channels: int):
    """`forward`'s parameter tile: the rates `A log2(e)` transposed, then the skip, padded to the stream's
    rows."""
    padding = jnp.zeros((PARAMETER_ROWS - 1, channels), jnp.float32)
    return jnp.concatenate([(A * algebra.LOG2_E).T, D[None], padding])


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
    positions = block_positions(scan.length, scan.positions)
    padded = -(-scan.length // positions) * positions

    def leading(t):
        """`t` `[R, T, ...]` with its tokens last, padded to whole blocks: a zero token decays nothing,
        writes nothing and reads nothing."""
        return mainloop.zero_padded(jnp.moveaxis(t, 1, -1), axis=t.ndim - 1, size=padded)

    # Its kernel named for the launch that traced the forward (`report.relaunched`), as JAX traces it after
    # that launch's scope ended: a spill of it refuses the launch and steps its tokens down (`Selective`).
    with report.relaunched(scan.launch):
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


@dataclasses.dataclass(frozen=True)
class Tiles:
    """The selective scan's launches' config: the forward's chunk of tokens, the backward's tokens a thread
    and the programs a multiprocessor holds of it (`backward_plan`)."""

    chunk: int
    positions: int
    resident: int


class Selective(base.Template[Tiles]):
    """Mamba's selective scan on the recurrence template (`selective_scan`), its config derived from the
    device's facts (`derived`), its launches named for the compile's reports (`report.launching`), so a launch
    whose kernels spilled steps its backward's tokens down (`report.spilled`). Its operands are
    `dew.nn.mixers.mamba.selective_scan`'s: `x`, `dt` `[B, S, D]`, `A` `[D, N]`, `B`, `C` `[B, S, N]`, `D`
    `[D]`, the entering state `[B, D, N]` or None and documents' first tokens `[B, S]` or None."""

    name: ClassVar[str] = "mosaic_gpu"
    config_cls: ClassVar[type[Tiles]] = Tiles
    GENERATES: ClassVar[bool] = True

    def _fwd(self, x, dt, A, B, C, D, state=None, starts=None, *, interpret: bool = False, config: Tiles,
             return_residuals: bool = False):
        from dew.nn.kernels.templates import bundles

        del return_residuals
        family = (bundles.INTERPRETED if interpret
                  else bundles.of(generation.device_generation(), jnp.result_type(x)))
        assert family is not None, "a bundle claims the device's family (`recurrence_chunk.selective_kernel`)"
        return selective_scan(x, dt, A, B, C, D, state, starts, family=family, chunk=config.chunk,
                              interpret=interpret, plan=(config.positions, config.resident)), None

    def derived(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> Tiles | None:
        """The call's config on `target`: the backward's tokens a thread and residency from its register model
        (`backward_plan`), past the token counts a compile found spilling (`report.spilled`), each a step down
        to half its tokens (None where a spill no token count clears); and the forward's chunk, from `CHUNK`
        the longest whose ring a block's shared memory holds (`staged_bytes`), up to a warpgroup's tokens (a
        chunk dividing every backward block), dividing the call's tokens padded to `CHUNK`'s (a longer chunk
        walks no more padding): on an H100 Mamba-1's bf16 forward at 8x2048x5120 took 1.21 ms at 16 tokens a
        chunk, 1.14 at 32 and 1.12 to 1.14 at 64 and 128, its training step 6.51, 6.39 to 6.43 and 6.34 to
        6.40 (sel-chunk-sxm at 5bf70d36b)."""
        facts = self._facts(ba, target)
        positions, resident = backward_plan(facts)
        chunk, length = CHUNK, ba.arguments["x"].shape[1]
        padded = -(-length // CHUNK) * CHUNK
        while (2 * chunk <= block_scan.THREADS and padded % (2 * chunk) == 0 and self.staged_bytes(
                ba, Tiles(2 * chunk, positions, resident), facts) <= generation.shared_budget(facts)):
            chunk *= 2
        tiles = Tiles(chunk, positions, resident)
        # A launch whose compile found a kernel spilling steps the tile that sets that kernel's registers: the
        # backward's tokens. The forward's registers no tile sets (its chunk's loop is rolled, `UNROLLED`), so
        # a forward spill, or a backward's at one token, leaves no config.
        while spilled := report.spilled(self, ba, tiles):
            backward = all(name.startswith("selective_scan_backward") for name in spilled)
            if not backward or tiles.positions == 1:
                return None
            fewer = tiles.positions // 2
            tiles = Tiles(tiles.chunk, fewer, backward_programs(backward_registers(fewer), facts))
        return tiles

    @staticmethod
    def _facts(ba: tokamax.BoundArguments, target: generation.GpuTarget) -> generation.GpuTarget:
        """The facts a call's launches run by: `target`'s, interpreted the interpreted bundle's generation's
        (`generation.INTERPRETED_GENERATION`)."""
        interpreted = ba.arguments.get("interpret", False)
        return generation.gpu_target(generation.INTERPRETED_GENERATION) if interpreted else target

    def legal(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget) -> bool:
        return super().legal(ba, config, self._facts(ba, target))

    def registers(self, ba: tokamax.BoundArguments, config: Tiles) -> int | None:
        """The backward thread's registers at the config's tokens (`backward_registers`)."""
        return backward_registers(config.positions)

    def register_budget(self, config: Tiles) -> int:
        """A thread's share of a multiprocessor's registers at the config's residency."""
        return resources.entry_registers(1, config.resident)

    def staged_bytes(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget) -> int:
        """The forward's shared memory at the config's chunk (`forward_allocations`), as the toolkit counts a
        program's (`resources.shared_bytes`)."""
        from dew.nn.kernels.templates import bundles

        arguments = ba.arguments
        x, dt, b, c = (arguments[name] for name in ("x", "dt", "B", "C"))
        family = bundles.INTERPRETED if arguments.get("interpret", False) else bundles.of(target.generation,
                                                                                            x.dtype)
        assert family is not None, f"no bundle of the template claims {target.generation} (`eligible`)"
        return resources.shared_bytes(family, forward_scope(family, config.chunk, b.shape[-1],
                                                            (x.dtype, dt.dtype, b.dtype, c.dtype),
                                                            arguments.get("starts") is not None))

    def eligible(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
        """Whether a bundle of the template claims the call's GPU (interpreted, the interpreted bundle)."""
        from dew.nn.kernels.templates import bundles

        return (ba.arguments.get("interpret", False)
                or bundles.of(target.generation, ba.arguments["x"].dtype) is not None)


SELECTIVE = Selective()
"""The selective scan's template launch (`Selective`)."""
