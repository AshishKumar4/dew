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
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import tokamax
from jax import lax
from jax._src.pallas.mosaic_gpu import core as plgpu_core
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew._devices import deterministic_ops_requested
from dew.nn import derivatives
from dew.nn.kernels import generation, recurrence_algebra as algebra, tuning
from dew.nn.kernels.chunked_launch import AlignedDocuments, Forked, zero_padded
from dew.nn.kernels.templates import base, mosaic, report, resources, scan as block_scan, swizzling
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
    owned: bool = False
    """Whether the backward's programs own their partial sums, the run asking its reductions ordered
    (`backward`)."""
    count: int = 0
    """A forked call's given states a row (`Forked`; zero for none): its documents laid from chunks' edges
    (`chunked_launch.AlignedDocuments`), each chunk taking a document up from a slot of its table."""
    held: int = 0
    """A forked call's stored slots a row, each a document's final a later one takes up."""

    @property
    def forked(self) -> bool:
        return self.count > 0

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
    # Each tile read a row at a time, at a traced row: staged plain (`swizzling.UNSWIZZLED`).
    streams = {"walked": family.streams(blocks, 2, 1, swizzles=dict.fromkeys(blocks, swizzling.UNSWIZZLED)),
               "once": family.streams(once, 1, 1, swizzles=dict.fromkeys(once, swizzling.UNSWIZZLED))}
    return streams, (plgpu.SMEM((chunk, block), x_dtype), plgpu.SMEM((block,), jnp.float32))


def _slot_at(row, slot, slots: int):
    """Row `row`'s slot `slot` of its `slots`, in range: one-based, a given slot's a chunk's load
    (`AlignedDocuments.loads`, `slots` `Scan.count`), a stored slot's a store or a load past the given slots
    less their count (`slots` `Scan.held`)."""
    return row * slots + jnp.clip(slot - 1, 0, slots - 1)


def _entered(scan: Scan, row, load, given, stored):
    """The state a document taking slot `load` up enters, every leaf of it (`given(at)`, `stored(at)` its
    given and stored slots' at a slot's index): a given slot's, past the given slots a stored one's, and a
    zero state where `load` is zero. The forward's and the backward's one reading of a chunk's table."""
    state = given(_slot_at(row, load, scan.count))
    if scan.held:
        kept = stored(_slot_at(row, load - scan.count, scan.held))
        state = jax.tree.map(lambda kept, entered: jnp.where(load > scan.count, kept, entered), kept, state)
    return jax.tree.map(lambda leaf: jnp.where(load > 0, leaf, 0.0), state)


@dataclasses.dataclass(frozen=True)
class Documents:
    """A forked call's documents as a forward program walks its row (`Scan.forked`; kernel-templates.md
    section 3, the recurrence template's `table.Taking`): before a chunk that takes a document up
    (`AlignedDocuments.loads`: one-based, zero a zero state, -1 none, the chunk continuing the state before
    it), its state the table's slot's, given or stored by an earlier document of the row; after a chunk
    that ends a storing document (`AlignedDocuments.stores`), its state written to the row's stored slot,
    which a later chunk of the same program reads back (`Family.kept`). The program's columns of each slot's
    `[N, D]` state, a vector a state."""

    scan: Scan
    given: Any
    loads: Any
    stores: Any
    stored: Any
    """The row's stored slots (None where its documents store none, `Scan.held`)."""
    row: jax.Array
    columns: object
    family: mosaic.Family

    def _slot(self, ref, base) -> list:
        return [ref[base, n, self.columns] for n in range(self.scan.states)]

    def taken(self, index, h) -> list:
        """The state chunk `index` enters: `h`, or where it takes a document up, its slot's (`_entered`)."""
        load = self.loads[self.row, index]
        def take():
            return _entered(self.scan, self.row, load, functools.partial(self._slot, self.given),
                            functools.partial(self._slot, self.stored))

        return lax.cond(load >= 0, take, lambda: list(h))

    def kept(self, index, h, *, smem) -> None:
        """Chunk `index`'s final state `h` written to its stored slot where it ends a storing document."""
        stored = self.stored
        if stored is None:
            return
        slot = self.stores[self.row, index]

        @pl.when(slot > 0)
        def _():
            base = _slot_at(self.row, slot, self.scan.held)
            self.family.kept(list(h), [stored.at[base, n, self.columns] for n in range(self.scan.states)],
                             self.scan.interpret, smem=smem)


@tuning.lowered_once(("scan",))
def forward(x, dt, b, c, parameters, state, resets, scan: Scan, table: tuple = ()):
    """The scan's outputs from padded operands: `x`, `dt` `[R, T, D]`, `b`, `c` `[R, T, N]`, `parameters`
    `[N + PARAMETER_ROWS, D]` fp32 (the rates `A log2(e)` transposed, then the skip `D`), `state` `[R, N, D]`
    fp32 and `resets` `[R, T, RESET_WIDTH]` int8 (a token's flag in every column; ignored unless
    `scan.packed`). Returns `y` `[R, T, D]` in `x`'s dtype, the final state `[R, N, D]` fp32, with
    `scan.entered` the entering state of each chunk opening a backward block `[R, chunks / opening, N, D]`
    fp32 (`Scan.opening`; None without): every chunk's kept 64 times what 2048-token blocks of 32-token
    chunks read (4.3 GB at Falcon 7B's step, 2970); and a forked call's stored slots `[R * held, N, D]`
    fp32 (None without). A forked call's `table` (`scan.forked`) is its given states `[R * count, N, D]`
    fp32 and the slot each chunk takes up and stores (`AlignedDocuments.loads`, `stores` `[R, chunks]`),
    the walk from a zero state."""
    family, chunk, block, states = scan.family, scan.chunk, scan.block, scan.states
    streams, allocations = forward_allocations(family, chunk, block, states, (x.dtype, dt.dtype, b.dtype,
                                                                              c.dtype), scan.packed)
    out_type = [jax.ShapeDtypeStruct(x.shape, x.dtype),
                jax.ShapeDtypeStruct((scan.rows, states, scan.channels), jnp.float32)]
    if scan.entered:
        kept = -(-scan.chunks // scan.opening)
        out_type.append(jax.ShapeDtypeStruct((scan.rows, kept, states, scan.channels), jnp.float32))
    if scan.held:
        out_type.append(jax.ShapeDtypeStruct((scan.rows * scan.held, states, scan.channels), jnp.float32))

    def entry(x_ref, dt_ref, b_ref, c_ref, parameters_ref, state_ref, resets_ref, *refs):
        table_refs, (y_ref, final_ref, *outputs) = ((refs[:3], refs[3:]) if scan.forked else ((), refs))
        entered_ref = outputs[:1] if scan.entered else []
        stored_ref = outputs[-1] if scan.held else None
        row, part = lax.axis_index("rows"), lax.axis_index("blocks")
        columns = pl.ds(part * block, block)
        documents = (Documents(scan, *table_refs, stored=stored_ref, row=row, columns=columns, family=family)
                     if scan.forked else None)
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
                    if documents is not None:
                        h = documents.taken(index, h)
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
                    if documents is not None:
                        documents.kept(index, h, smem=state_smem)
                    return h, accumulators

                h, _ = family.walk(family.fed(walked, views, 0, scan.chunks, wg), ((step, 0, scan.chunks),),
                                   h0, (), scan.interpret)
                family.stored(list(h), [final_ref.at[row, n, columns] for n in range(states)],
                              smem=state_smem)
                family.store_wait(0)

            def producer():
                family.stream(given, once_views, 0, 1)
                family.stream(walked, views, 0, scan.chunks)

            family.scheduled(1, consumer, producer)

        family.scoped(body, allocations, streams=streams)

    entry.__name__ = "selective_scan"
    launched = family.kernel(entry, out_type=tuple(out_type), grid=(scan.rows, scan.blocks),
                             grid_names=("rows", "blocks"), consumers=1, interpret=scan.interpret,
                             name="selective_scan")
    outs = launched(x, dt, b, c, parameters, state, resets, *table)
    if not scan.entered:
        outs = (*outs[:2], None, *outs[2:])
    return tuple(outs) if scan.held else (*outs, None)


PARTIALS = 32
"""The buffers B's and C's gradients are added into across the channels, a channel's its index modulo
them, then summed (XLA's reduction): each buffer adds a 32nd of the channels in turn, where one buffer's
sequential adds put B's gradient 1.95 times the portable scan's distance from float64 with a document
starting at every chunk (the interpreted regime test); the adds a buffer takes fewer, as contended."""


BACKWARD_CARRIED = ("dx", "ddt", "x", "dt", "dy")
"""The token vectors a backward thread carries across its state loop, each its tokens' fp32 values: the
channel's sums over the states and the block's operands."""

BACKWARD_STATE = ("B", "C", "decay", "write", "states before", "C dy", "cotangent after")
"""The token vectors a state's turn of the backward's state loop holds beside `BACKWARD_CARRIED` at its peak:
the state's operands, its decay and write, the states before each token, C dy and the cotangent after each
token, counted from `backward`."""

BACKWARD_SCALARS = {"addresses": 20, "block scan": 8, "state loop": 12}
"""The registers a backward thread holds apart from its token vectors and a thread's control
(`resources.CONTROL_REGISTERS`): its refs' 64-bit addresses, the block scan's running pair, lane and warp
terms, the state loop's scalars (its rate, entering state, carried cotangent, rate gradient so far and
indices). With the vectors they state ptxas's least spill-free cap on an H100 at 4, 8 and 16 tokens a
thread, 96, 152 and 232, as 104, 152 and 248 (sel-need at f3040b154)."""

SCAN_STEPS = 16
"""The dependent steps of one block scan past its tokens' own (`templates.scan.linear_scan`): six shuffles
across the warp, three warpgroup barriers, a shared store and load of the warps' aggregates and four joins of
them, and the apply to the thread's tokens. The backward runs two a state, forward and reverse."""


def backward_registers(positions: int) -> int:
    """The registers a backward thread holds at `positions` tokens: its token vectors (`BACKWARD_CARRIED`,
    `BACKWARD_STATE`), each `positions` fp32 words, its scalars (`BACKWARD_SCALARS`) and a thread's control
    (`resources.CONTROL_REGISTERS`)."""
    vectors = len(BACKWARD_CARRIED) + len(BACKWARD_STATE)
    return vectors * positions + sum(BACKWARD_SCALARS.values()) + resources.CONTROL_REGISTERS


def backward_programs(registers: int, target: generation.GpuTarget) -> int:
    """The programs of the backward, one warpgroup and no producer, a multiprocessor of `target` holds at once
    at `registers` a thread: the shared planner's residency (`resources.RegisterModel.resident`) within its
    resident threads."""
    held = target.multiprocessor
    threads = held.resident_threads // block_scan.THREADS
    model = resources.RegisterModel(registers, held.thread_registers, 1, min(held.resident_blocks, threads),
                                    held.register_step, 0, file=held.registers)
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
    while backward_registers(positions) <= target.multiprocessor.thread_registers:
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


@dataclasses.dataclass(frozen=True)
class DocumentsBack:
    """A forked call's documents as a backward program walks a row's channel (`Scan.forked`; the design of
    row 5288, kernel-templates.md section 3). Documents start at chunks' edges
    (`AlignedDocuments`), where the forward took each one's state `L` up from its table. A block's states are
    scanned from its kept entering state, each document starting inside it from its `L` (`before`). Its
    cotangents are the token recurrence's `q_t = a_t (C_t dy_t + q_{t+1})` with each document isolated at its
    first token, where `q_{t0}`, the true decay's, is the gradient of `L` (`dm`), and the cotangent a
    document's last token takes from past its end is its final's, `G`: the sum of the `dm` of the documents
    that take its stored slot up, each later in the row, so resolved in reverse. A block resolves its
    documents in reverse from an isolated scan `q0` (no `G` inside the block) and the suffix products `P` of
    each document's decays to its end, `dm = a_{t0} (C dy + q0_{t0+1}) + a_{t0} P_{t0+1} G`, then scans the
    cotangents once more with each `G` written at the next document's first token. `P` is a product of
    decays in the scan's own order, never a ratio of prefix products, so no decay is divided by and none
    underflowing loses another; a document's `G` ending past the block flows through the carried cotangent,
    as one does without documents.
    A program owns its channel of the given slots' gradients and the stored slots' sums, one buffer
    (`dslots`: the given slots', then the stored ones' past them, so the launch holds no two mutable operands
    of one shape, `mosaic.checked`), every thread reading a sum before any writes it (`mosaic.synced`)."""

    scan: Scan
    given: Any
    stored: Any
    loads: Any
    stores: Any
    dslots: Any
    exchanged: Any
    """The block scan's exchange (`block_scan.exchange_scratch`)."""
    tokens: Any
    """The documents' token rows: the states entered, the lasts, the two parts of `dm` and the `G` each first
    token takes."""
    layout: plgpu_core.SomeLayout
    rows: int
    """The call's rows, whose given slots lead the slots' gradients (`dslots`)."""
    row: jax.Array
    channel: jax.Array

    def _slot(self, ref, base, n):
        return ref[base, n, self.channel]

    def _taken(self, load, n):
        """The state a document taking slot `load` up enters, at state `n` of the program's channel."""
        return _entered(self.scan, self.row, load, lambda at: self._slot(self.given, at, n),
                        lambda at: self._slot(self.stored, at, n))

    def _summed(self, slot):
        """Stored slot `slot`'s index (one-based, `_slot_at`'s) in the slots' gradients (`dslots`), past the
        given slots'."""
        return self.scan.count * self.rows + _slot_at(self.row, slot, self.scan.held)

    def _vector(self, index: int):
        return plgpu.load(self.tokens.at[index], layout=self.layout, optimized=False)

    def _cleared(self, *indices: int) -> None:
        zeros = plgpu.layout_cast(jnp.zeros(self.tokens.shape[1:], jnp.float32), self.layout)
        for index in indices:
            plgpu.store(self.tokens.at[index], zeros, optimized=False)

    def scans(self, k, n, firsts, a, written, read, entering, carried):
        """Block `k`'s states before each token, the cotangents each takes from past it, and the cotangent
        it passes the block before, at state `n`, from its documents' first tokens `firsts` (1 or 0)."""
        scan, interpret, loads = self.scan, self.scan.interpret, self.loads
        size = self.tokens.shape[1]
        opening = size // scan.chunk
        # The states documents starting inside the block enter, at their first tokens, and the block's tokens
        # each ending one (a chunk's last before such a first).
        self._cleared(0, 1)
        mosaic.synced(interpret)
        for j in range(1, opening):
            load = loads[self.row, k * opening + j]

            @pl.when(load >= 0)
            def _(j=j, load=load):
                self.tokens[0, j * scan.chunk] = self._taken(load, n)
                self.tokens[1, j * scan.chunk - 1] = jnp.float32(1.0)
        mosaic.synced(interpret)
        entered, lasts = self._vector(0), self._vector(1)
        index = plgpu.layout_cast(lax.broadcasted_iota(jnp.int32, (size,), 0), self.layout)
        inside = jnp.where(index > 0, firsts, 0.0)
        before = block_scan.linear_scan(a * (1.0 - inside), written + a * entered, entering, self.exchanged,
                                        exclusive=True, interpret=interpret)[0]
        before = jnp.where(inside > 0, entered, before)
        cut = 1.0 - firsts
        isolated = block_scan.linear_scan(a * cut, a * read * cut, carried, self.exchanged, reverse=True,
                                          exclusive=True, interpret=interpret)[0]
        # Each token's suffix product of its document's decays to the document's last token, inclusive: a
        # product of decays in the scan's order, one at the last token, a one-token document's its own decay.
        products = block_scan.linear_scan(a * (1.0 - lasts), a * lasts, jnp.zeros((), jnp.float32),
                                          self.exchanged, reverse=True, interpret=interpret)[0]
        # Each first token's parts of its document's `dm`, the isolated cotangent's and `G`'s factor.
        mosaic.synced(interpret)
        plgpu.store(self.tokens.at[2], a * (read + isolated) * firsts, optimized=False)
        plgpu.store(self.tokens.at[3], products * firsts, optimized=False)
        self._cleared(4)
        mosaic.synced(interpret)
        self._resolved(k, n, opening)
        injected = self._vector(4)
        after, passed = block_scan.linear_scan(a * cut, a * read * cut + injected, carried, self.exchanged,
                                               reverse=True, exclusive=True, interpret=interpret)
        return before, after, passed

    def _resolved(self, k, n, opening: int) -> None:
        """Block `k`'s documents resolved from its last at state `n`: each one's `dm` sent to the slot it took
        up (a given slot's gradient, or summed into a stored one's `G`), and the `G` of the document before
        each written at its first token (`tokens[4]`) for the cotangents' last scan."""
        scan, interpret = self.scan, self.scan.interpret
        loads, stores, dslots = self.loads, self.stores, self.dslots
        chunk = scan.chunk
        later = jnp.int32(-1)
        for j in reversed(range(opening)):
            chunked = k * opening + j
            load = loads[self.row, chunked]
            # The document starting here ends inside the block where one starts after it: its `G` the stored
            # sum of its slot, held at its last chunk's store.
            ends = later >= 0
            store = stores[self.row, jnp.maximum(later - 1, 0)]
            final = 0.0
            if scan.held:
                final = jnp.where(ends & (store > 0), self._slot(dslots, self._summed(store), n), 0.0)
            dm = self.tokens[2, j * chunk] + jnp.where(ends, self.tokens[3, j * chunk] * final, 0.0)
            taken = load >= 0
            # The slot `dm` is sent to: a given one's gradient, or a stored one's sum past the given slots.
            sent = [(taken & (load > 0) & (load <= scan.count), _slot_at(self.row, load, scan.count))]
            if scan.held:
                sent.append((taken & (load > scan.count), self._summed(load - scan.count)))
            sums = [(when, at, self._slot(dslots, at, n)) for when, at in sent]
            mosaic.synced(interpret)
            for when, at, summed in sums:
                @pl.when(when)
                def _(at=at, summed=summed, dm=dm):
                    dslots[at, n, self.channel] = summed + dm
            mosaic.synced(interpret)
            # The document before this one ends at its first token: its `G`, every later document's `dm`
            # summed into its stored slot by now.
            before = stores[self.row, jnp.maximum(chunked - 1, 0)]
            if scan.held:
                ended = jnp.where((chunked > 0) & (before > 0), self._slot(dslots, self._summed(before), n),
                                  0.0)

                @pl.when(taken)
                def _(j=j, ended=ended):
                    self.tokens[4, j * chunk] = ended
            later = jnp.where(taken, chunked, later)
        mosaic.synced(interpret)


@tuning.lowered_once(("scan",))
def backward(x, dt, b, c, parameters, entered, resets, dy, dfinal, scan: Scan, table: tuple = ()):
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
    block's tokens across the threads so a warp's adds are adjacent (cub's BlockedToStriped), into one of
    `PARTIALS` buffers, which XLA sums. The adds land in whatever order the programs run, so a step repeated
    on the same schedule differs in B's and C's gradients' last bits (row 4392), each run within check 1 of
    float64: speed over determinism (bible §1), as mamba_ssm's, the default and a declared contract (row
    4448). A run asking its reductions ordered (`--xla_gpu_deterministic_ops`, `deterministic_ops_requested`)
    takes the fixed order instead: a program owns a buffer and walks the channels it holds in their order,
    adding plainly, every run's bits the same, at 2.0 to 2.2 times the step on an A100 (rows x channels
    1x5120: 2.31 ms atomic, 4.70 owned, 6.17 summed in shared memory; 4x8192: 9.22, 21.1, 25.1;
    selective-ab-a100 at 8ca2795113)."""
    _, channels, length = x.shape
    states, positions = b.shape[1], block_positions(length, scan.positions)
    blocks = length // positions
    interpret = scan.interpret
    if scan.forked and positions % scan.chunk:
        raise ValueError(f"a backward block of {positions} tokens holds no whole chunks of {scan.chunk}")
    layout = block_scan.blocked(positions)
    owned = scan.owned
    partials = math.gcd(PARTIALS, channels) if owned else min(PARTIALS, channels)
    # A program's channels: its own where the adds are atomic, else every `partials`-th from its buffer's.
    turns = channels // partials if owned else 1

    def entry(x_ref, dt_ref, b_ref, c_ref, parameters_ref, entered_ref, resets_ref, dy_ref, dfinal_ref,
              *refs):
        row = lax.axis_index("rows")
        walked = None
        if scan.forked:
            (given_ref, stored_ref, loads_ref, stores_ref, dbc_ref, dslots_ref, dx_ref, ddt_ref, da_ref,
             dd_ref, dstate_ref, exchanged, staged, carried, tokens) = refs
            # A channel's walk of the row's documents.
            walked = functools.partial(DocumentsBack, scan, given_ref, stored_ref, loads_ref, stores_ref,
                                       dslots_ref, exchanged, tokens, layout, x.shape[0], row)
        else:
            dbc_ref, dx_ref, ddt_ref, da_ref, dd_ref, dstate_ref, exchanged, staged, carried = refs

        def load(ref):
            return plgpu.load(ref, layout=layout, optimized=False).astype(jnp.float32)

        def added(ref, value):
            """`value` added to `ref`: plainly where the program owns it or interpreted (whose programs run
            one at a time), else atomically across the threads (`block_scan.to_striped`), a warp's adds
            adjacent."""
            if interpret or owned:
                plgpu.store(ref, load(ref) + value, optimized=False)
                return
            plgpu.atomic_add(ref, block_scan.to_striped(value, staged), optimized=False)

        def block(i, d_skip, *, channel):
            k = blocks - 1 - i
            span = pl.ds(k * positions, positions)
            xs, dts, dys = (load(ref.at[row, channel, span]) for ref in (x_ref, dt_ref, dy_ref))
            kept = 1.0 - load(resets_ref.at[row, span]) if scan.packed else None
            documents, firsts = ((None, None) if walked is None else
                                 (walked(channel), load(resets_ref.at[row, span])))

            def state(n, sums):
                # A state a turn, rolled: unrolled, the states' operands lived together and spilled 12 KB
                # a thread (255 registers, an H100's compile, scanbwd-ptxas-sm90 at 5de7abaa0).
                dx, ddt = sums
                rate = parameters_ref[n, channel]
                bs, cs = load(b_ref.at[row, n, span]), load(c_ref.at[row, n, span])
                a = algebra.selective_decay(dts, rate, kept)
                written = dts * xs * bs
                entering = entered_ref[row, k, n, channel]
                read = cs * dys
                before, after, carried[0, n] = (
                    _block_scans(a, written, read, entering, carried[0, n], exchanged, interpret)
                    if documents is None else
                    documents.scans(k, n, firsts, a, written, read, entering, carried[0, n]))
                d_value, d_step, d_b, d_c, d_rate = algebra.selective_adjoint(
                    read + after, before, a, written, dts, xs, bs, dys, rate)
                # The rate's gradient so far read before the sum's barriers, so no thread reads another's
                # write of it: read after, an A100's threads added the block's sum twice over (1.6e-2 of
                # dA from float64 against the rest's 4e-7, mamba1-scan-a100-7d3152837).
                so_far = carried[1, n]
                carried[1, n] = so_far + block_scan.summed(d_rate, exchanged, interpret=interpret)
                partial = channel % partials
                # B's and C's gradients one ref, each its place along its leading axis, indexed once.
                added(dbc_ref.at[0, partial, row, n, span], d_b)
                added(dbc_ref.at[1, partial, row, n, span], d_c)
                return dx + d_value, ddt + d_step

            zeros = jnp.zeros((positions,), jnp.float32)
            dx, ddt = lax.fori_loop(0, states, state, (parameters_ref[states, channel] * dys, zeros))
            plgpu.store(dx_ref.at[row, channel, span], dx, optimized=False)
            plgpu.store(ddt_ref.at[row, channel, span], ddt, optimized=False)
            return d_skip + block_scan.summed(dys * xs, exchanged, interpret=interpret)

        def turned(turn, carry):
            channel = lax.axis_index("channels") + turn * partials
            # Each state's cotangent carried from block to block and its rate's gradient summed, in shared
            # memory, a state's a turn of the rolled loop.
            for n in range(states):
                carried[0, n] = dfinal_ref[row, n, channel]
                carried[1, n] = jnp.zeros((), jnp.float32)
            dd_ref[row, channel] = lax.fori_loop(0, blocks, functools.partial(block, channel=channel),
                                                 jnp.zeros((), jnp.float32))
            for n in range(states):
                dstate_ref[row, n, channel] = carried[0, n]
                da_ref[row, n, channel] = carried[1, n]
            return carry

        lax.fori_loop(0, turns, turned, None)

    return _backward_launched(entry, x, dt, b, c, parameters, entered, resets, dy, dfinal, scan, table,
                              programs=channels // turns, partials=partials)


def _block_scans(a, written, read, entering, carried, exchanged, interpret: bool):
    """A state's two scans through a backward block without documents (`DocumentsBack.scans` a forked
    call's): its states before each token from the block's `entering` one, and its cotangents after each
    from `carried`, the block after's; returns both and the cotangent the block passes to the one before."""
    before, _ = block_scan.linear_scan(a, written, entering, exchanged, exclusive=True, interpret=interpret)
    after, passed = block_scan.linear_scan(a, a * read, carried, exchanged, reverse=True, exclusive=True,
                                           interpret=interpret)
    return before, after, passed


def _backward_launched(entry, x, dt, b, c, parameters, entered, resets, dy, dfinal, scan: Scan, table: tuple,
                       *, programs: int, partials: int):
    """`backward`'s launch of its program `entry`, a row's `programs` a program each, and its outputs:
    `(dx, ddt, db, dc, dA, dD, dstate)`, B's and C's gradients summed over their `partials` buffers; a forked
    call's (`table`: its given states, its stored slots, the slots each chunk takes up and stores) the given
    states' gradient `[R * count, N, D]` after them, its stored slots' sums a scratch of the launch's."""
    rows, channels, _ = x.shape
    states, positions = b.shape[1], block_positions(x.shape[2], scan.positions)
    interpret = scan.interpret
    sized = jax.ShapeDtypeStruct
    out_type = (sized(x.shape, jnp.float32), sized(x.shape, jnp.float32),
                sized((rows, states, channels), jnp.float32), sized((rows, channels), jnp.float32),
                sized((rows, states, channels), jnp.float32))
    scratch = (block_scan.scratch(), block_scan.exchange_scratch(positions),
               plgpu.SMEM((2, states), jnp.float32))
    if scan.forked:
        # The documents' token rows (`DocumentsBack`): the states entered, the lasts, the two parts of `dm`
        # and the `G` each first token takes.
        scratch = (*scratch, plgpu.SMEM((5, positions), jnp.float32))
    launched = mosaic.kernel(entry, out_type=out_type, grid=(rows, programs),
                             grid_names=("rows", "channels"),
                             warpgroups=1, interpret=interpret, family=scan.family.name,
                             name="selective_scan_backward", resident=scan.resident, scratch=scratch)

    def accumulated(refs):
        """The launch's outputs written to `refs`, B's and C's gradients added to theirs from zero, one ref
        of both (a leading axis of them: two refs of equal zeros are one value to XLA, which a fusion
        wrapping the launch may pass once for both, row 3180, `mosaic.checked`): state the run discharges
        (`pl.run_state`), as a kernel's derivative inside a custom JVP (`generation.linear_tangent`) can
        create no Ref of its own. A whole-array write of a launch's output XLA folds to the output itself."""
        summed, outputs = refs[:1 + scan.forked], refs[1 + scan.forked:]
        found = launched(x, dt, b, c, parameters, entered, resets, dy, dfinal, *table, *summed)
        for ref, value in zip(outputs, found, strict=True):
            ref[...] = value

    sums = [jnp.zeros((2, partials, *b.shape), jnp.float32)]
    if scan.forked:
        # The given slots' gradients and the stored slots' sums one buffer (`DocumentsBack`).
        given = table[0]
        sums.append(jnp.zeros((rows * (scan.count + scan.held), *given.shape[1:]), jnp.float32))
    dbc, *found = pl.run_state(accumulated)((*sums, *(jnp.zeros(t.shape, t.dtype) for t in out_type)))
    dgiven = found[0][:rows * scan.count] if scan.forked else None
    dx, ddt, da, dd, dstate = found[scan.forked:]
    return dx, ddt, dbc[0].sum(0), dbc[1].sum(0), da, dd, dstate, dgiven


CHUNK = 32
"""The fewest tokens a forward chunk takes, a ring stage of its operands whose entering state the forward
keeps for each backward block it opens (`Scan.opening`), and the fewest a call takes (a decode step's are
the portable math's, `recurrence_chunk.SELECTIVE_SUPPORTS`)."""


def selective_scan(x, dt, A, B, C, D, state=None, starts=None, *, family: mosaic.Family, chunk: int = CHUNK,
                   block: int = block_scan.THREADS, interpret: bool = False,
                   plan: tuple[int, int] | None = None, forked: Forked | None = None):
    """`dew.nn.mixers.mamba.selective_scan` on the template: `x`, `dt` `[B, S, D]` (after the softplus),
    `A` `[D, N]`, `B`, `C` `[B, S, N]`, `D` `[D]`, `state` `[B, D, N]` or None, `starts` `[B, S]` or None.
    Returns `(y [B, S, D], final state [B, D, N])` in `x`'s dtype. The sequence is padded to whole chunks
    with a zero step, which neither decays nor writes. `chunk` divides the backward's block
    (`block_positions`), which starts from a chunk's entering state the forward keeps. A forked call
    (`forked`, `recurrence_chunk.Chunk.loads`; `state` `[B, S', D, N]` its table's given states) lays each
    document from a chunk's edge, each taking its state up there from its table and a storing one leaving
    its final in its slot (`Documents`, `AlignedDocuments`), the portable scan's arithmetic; its final state
    is the row's last."""
    # The backward's plan its caller derived (`Selective.derived`), else this device's, interpreted the
    # interpreted target's (`generation.interpreted_target`).
    positions, resident = plan or backward_plan(generation.interpreted_target() if interpret
                                                else generation.gpu_target())
    if block_scan.THREADS * positions % chunk:
        raise ValueError(f"a chunk of {chunk} tokens divides no backward block of "
                         f"{block_scan.THREADS * positions}: its blocks start at no kept state")
    rows, length, channels = x.shape
    states = A.shape[-1]
    table, count, held, aligned = (), 0, 0, None
    if forked is not None:
        aligned = AlignedDocuments.of(forked, length, chunk)
        x, dt, B, C = aligned.laid(x, dt, B, C)
        # The given states `[B, S', D, N]` as the walk reads a slot, `[N, D]`, rows leading.
        given = jnp.swapaxes(jnp.asarray(state, jnp.float32), 2, 3)
        count, held, starts, state = given.shape[1], forked.held, None, None
        table = (given.reshape(rows * count, states, channels), aligned.loads, aligned.stores)
    padded = -(-x.shape[1] // chunk) * chunk

    pad = functools.partial(zero_padded, axis=1, size=padded)
    if aligned is not None:
        # A forked call's documents' first tokens, each a chunk's first that takes one up: the backward's
        # isolation, the forward's take its table's (`scan.packed` off).
        starts = jnp.repeat(aligned.loads >= 0, chunk, axis=1) & (jnp.arange(padded) % chunk == 0)
    resets = (jnp.zeros((rows, padded, RESET_WIDTH), jnp.int8) if starts is None else
              jnp.broadcast_to(pad(jnp.asarray(starts, jnp.int8))[..., None], (rows, padded, RESET_WIDTH)))
    entering = (jnp.zeros((rows, states, channels), jnp.float32) if state is None
                else jnp.swapaxes(jnp.asarray(state, jnp.float32), 1, 2))
    scan = Scan(rows, padded, channels, states, chunk, block, family, positions, resident, interpret,
                packed=starts is not None and aligned is None, launch=report.launch_digest(),
                owned=deterministic_ops_requested(), count=count, held=held)
    y, final = _scan(pad(x), pad(dt), jnp.asarray(A, jnp.float32), pad(B), pad(C),
                     jnp.asarray(D, jnp.float32), entering, resets, scan, *table)
    y = y[:, :x.shape[1]] if aligned is None else aligned.unlaid(y)
    return y[:, :length], jnp.swapaxes(final, 1, 2).astype(x.dtype)


def _parameters(A, D, channels: int):
    """`forward`'s parameter tile: the rates `A log2(e)` transposed, then the skip, padded to the stream's
    rows."""
    padding = jnp.zeros((PARAMETER_ROWS - 1, channels), jnp.float32)
    return jnp.concatenate([(A * algebra.LOG2_E).T, D[None], padding])


def _scan_primal(x, dt, A, B, C, D, entering, resets, scan: Scan, *table):
    """`forward` over padded operands, its gradient `backward`'s: `(y, final [R, N, D] fp32)`; a forked
    call's `table` (`forward`'s)."""
    y, final, *_ = forward(x, dt, B, C, _parameters(A, D, scan.channels), entering, resets, scan, table)
    return y, final


def _scan_forward(x, dt, A, B, C, D, entering, resets, scan: Scan, *table):
    kept = dataclasses.replace(scan, entered=True)
    y, final, entered, stored = forward(x, dt, B, C, _parameters(A, D, scan.channels), entering, resets,
                                        kept, table)
    return (y, final), (x, dt, A, B, C, D, entered, resets, table, stored)


def _scan_backward(scan: Scan, residuals, grads):
    x, dt, A, B, C, D, entered, resets, table, stored = residuals
    dy, dfinal = grads
    # A forked call's table as the backward reads it: given, stored (one slot of zeros where none is held),
    # then each chunk's slots taken up and stored.
    positions = block_positions(scan.length, scan.positions)
    padded = -(-scan.length // positions) * positions
    tabled = ()
    if table:
        given, loads, stores = table
        chunks = padded // scan.chunk
        widened = (jnp.pad(loads, ((0, 0), (0, chunks - loads.shape[1])), constant_values=-1),
                   jnp.pad(stores, ((0, 0), (0, chunks - stores.shape[1]))))
        tabled = (given, stored if stored is not None else jnp.zeros((1, *given.shape[1:]), jnp.float32),
                  *widened)

    def leading(t):
        """`t` `[R, T, ...]` with its tokens last, padded to whole blocks: a zero token decays nothing,
        writes nothing and reads nothing."""
        return zero_padded(jnp.moveaxis(t, 1, -1), axis=t.ndim - 1, size=padded)

    # Its kernel named for the launch that traced the forward (`report.relaunched`), as JAX traces it after
    # that launch's scope ended: a spill of it refuses the launch and steps its tokens down (`Selective`).
    with report.relaunched(scan.launch):
        dx, ddt, db, dc, da, dd, dstate, dgiven = backward(
            leading(x), leading(dt), leading(B), leading(C), _parameters(A, D, scan.channels), entered,
            leading(resets[..., 0]), leading(dy.astype(x.dtype)), dfinal.astype(jnp.float32), scan, tabled)

    def trailing(t):
        """A gradient `[R, ..., T]` back as `[R, T, ...]`, the padding's tokens dropped."""
        return jnp.moveaxis(t[..., :scan.length], -1, 1)

    return (trailing(dx).astype(x.dtype), trailing(ddt).astype(dt.dtype), da.sum(0).T.astype(A.dtype),
            trailing(db).astype(B.dtype), trailing(dc).astype(C.dtype), dd.sum(0).astype(D.dtype), dstate,
            None, *(() if not table else (dgiven, None, None)))


_scan = derivatives.contracted("test_the_selective_scan_holds_its_orders")(
    derivatives.declared_derivative(derivatives.KernelGlue(), _scan_primal, _scan_forward, _scan_backward,
                                    nondiff_argnums=(8,)))


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

    def _fwd(self, x, dt, A, B, C, D, state=None, starts=None, *, forked: Forked | None = None,
             interpret: bool = False, config: Tiles, return_residuals: bool = False):
        from dew.nn.kernels.templates import bundles

        del return_residuals
        family = bundles.of((generation.interpreted_target().generation if interpret
                             else generation.device_generation()), jnp.result_type(x))
        assert family is not None, "a bundle claims the device's family (`recurrence_chunk.selective_kernel`)"
        return selective_scan(x, dt, A, B, C, D, state, starts, family=family, chunk=config.chunk,
                              interpret=interpret, plan=(config.positions, config.resident),
                              forked=forked), None

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
        """The facts a call's launches run by: `target`'s, interpreted the interpreted target's
        (`generation.interpreted_target`)."""
        interpreted = ba.arguments.get("interpret", False)
        return generation.interpreted_target() if interpreted else target

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
        family = bundles.of(self._facts(ba, target).generation, x.dtype)
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
