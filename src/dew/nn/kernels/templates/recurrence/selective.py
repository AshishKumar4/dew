"""Mamba's selective scan on the recurrence template's Mosaic GPU machinery (ledger 1300): a decay each channel
and state, `h_t = exp(dt_t A) h_{t-1} + dt_t x_t B_t`, `y_t = C_t . h_t + D x_t` (`dew.nn.mixers.mamba`'s
`selective_scan`), which no chunk product computes: the decay between two tokens depends on the channel and
the state together, so a chunk's pairs cost `C` times the scan's work (design: tmp design-r07-mamba1).

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

from dew.nn.kernels.templates import mosaic

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
    parameters = jnp.concatenate([(jnp.asarray(A, jnp.float32) * LOG2E).T, jnp.asarray(D, jnp.float32)[None],
                                  jnp.zeros((PARAMETER_ROWS - 1, channels), jnp.float32)])
    y, final, _ = forward(pad(x), pad(dt), pad(B), pad(C), parameters, entering, resets, scan)
    return y[:, :length], jnp.swapaxes(final, 1, 2).astype(x.dtype)
