"""The templates' TMA ring family (design §6): `RingFamily`, a family whose loads a producer warpgroup issues
by TMA into `Ring`s beside consumer warpgroups at `setmaxnreg` budgets, sm90's and sm100's, on `mosaic`'s
toolkit (its `Family`, walk, loads and stores)."""

import dataclasses
import functools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import jax
from jax import numpy as jnp
from jax._src.lib.mlir.dialects import nvvm
from jax._src.pallas.mosaic_gpu import core as plgpu_core
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import generation
from dew.nn.kernels.templates import after_walk, handoffs, persistent, resources, stores, swizzling
from dew.nn.kernels.templates.cost import Stated
from dew.nn.kernels.templates.mosaic import (
    Family,
    Loop,
    Product,
    ahead_stages,
    ahead_streams,
    chunked,
    commits,
    kernel,
    loaded,
    stepped,
    stepping,
    stored,
)
from dew.nn.kernels.templates.rings import (
    JOINED,
    Ring,
    Slots,
    Turn,
    drain,
    loads,
    opened_joined,
    passed,
    released,
    streamed,
)
from dew.nn.kernels.templates.stores import Output, Scattered, output_staging, staged_chunks

if TYPE_CHECKING:
    from dew.nn.kernels.templates.fed import Feed


def _fenced(ctx) -> None:
    """An inline body (`plgpu.inline_mgpu`) ordering the thread's global writes the async proxy made (TMA's)
    before its generic loads (PTX's `fence.proxy.async.global`); it returns nothing, an inline body's
    results its returned values alone. A fence alone: no work a count prices."""
    del ctx
    nvvm.fence_proxy(nvvm.ProxyKind.async_global)


class RingFamily[Accumulator](Family[dict, dict[str, Ring], Accumulator]):
    """A family whose loads a producer warpgroup issues by TMA into `Ring`s, beside one or two consumer
    warpgroups at `setmaxnreg` budgets (`roles`): sm90's and sm100's. Its walk acquires each step's slots as
    the step reads them and releases them once its products are done (`Slots`). Its products read their
    operands from the rings, a block's rows staged at the widest swizzle, 128 bytes. Composed with an issuer
    (`Issuer`) its walks read the home of (`accumulated`, `settled`)."""
    bounded_loads = True
    fixes_up = True
    gathers_rows = True
    selects_sources = True
    takes_pingpong = True
    walks_once = True
    stores_narrowed = True
    stores_chunks = True
    # Mosaic GPU's default for a warpgroup (its CompilerParams' 128 x 4 x 4 bytes).
    warpgroup_reduction_scratch = 2048
    stores_through_registers = False
    copy_extent = 256

    def registers(self, consumers: int, programs: int = 1, producer: int | None = None,
                  joined: tuple = ()) -> int:
        return resources.consumer_registers(consumers, programs, producer, self.joined_shares(joined))

    def column_unit(self, dtype) -> int:
        return swizzling.swizzle_columns(dtype, swizzling.SWIZZLE_BYTES)

    def depths(self, dtype) -> tuple[int, ...]:
        return (swizzling.swizzle_columns(dtype, swizzling.SWIZZLE_BYTES),)

    def kernel(self, entry, *, out_type, grid: tuple, grid_names: tuple, consumers: int,
               interpret: bool, approx_math: bool = True, name: str | None = None,
               cluster: int = 1, cost: pl.CostEstimate | Stated | None = None,
               model_flops: int | None = None,
               reducing_warpgroups: int = 0, resident: int = 1, scalars: int = 0,
               prefetched: Callable | None = None, registers: int | None = None, reduces: bool = True,
               producer: int | None = None, shared: int | None = None, tmem: int = 0,
               moves: bool | None = None, joined: tuple = ()) -> Callable:
        # Its programs read a table from global memory as any operand.
        del scalars
        if cluster > self.cluster:
            raise ValueError(f"{self.name} launches a cluster of at most {self.cluster}, not {cluster}")
        compiled = self.compiled_with(approx_math=approx_math)
        assert isinstance(compiled, plgpu.CompilerParams), "a ring family's kernels are Mosaic GPU's"
        return kernel(entry, out_type=out_type, grid=grid, grid_names=grid_names,
                      warpgroups=self.warpgroups(consumers, joined), family=self.name, interpret=interpret,
                      name=name, cluster=cluster, cost=cost, model_flops=model_flops, prefetched=prefetched,
                      model=(None if registers is None
                             else resources.register_model(self, consumers, registers, producer=producer,
                                                           programs=resident, shared=shared, tmem=tmem,
                                                           moves=moves, joined=joined,
                                                           target=generation.gpu_target())),
                      # scratch, `reduction_scratch`), and each implicitly reducing warpgroup's share.
                      compiled=dataclasses.replace(
                          compiled, min_blocks_per_sm=resident,
                          reduction_scratch_bytes=((self.reduction_scratch if reduces else 0)
                                                   + self.warpgroup_reduction_scratch * reducing_warpgroups)))

    def program_scoped(self, body, *allocations, streams):
        return pl.run_scoped(lambda *refs, **rings: body(*refs, streams=rings), *allocations, **streams,
                             collective_axes="wg")

    def roles(self, consumers: int, producer: int | None = None) -> handoffs.Schedule:
        return handoffs.Schedule((handoffs.Role("consumer", consumers), handoffs.Role("producer", weight=0)),
                                 floor=producer)

    def streams(self, blocks: dict, stages: int, consumers: int, multicast: tuple[str, ...] = (),
                swizzles: dict[str, int] | None = None, copies: dict[str, int] | None = None,
                gathered: tuple[str, ...] = (), joined: tuple[str, ...] = ()) -> dict:
        from dew.nn.kernels.templates import gathered as gathered_rows  # it builds on `Ring`
        from dew.nn.kernels.templates.rings import joined as joined_rings, ring

        allocated = {name: gathered_rows.ring(shape, dtype, stages, consumers=consumers) if name in gathered
                     else ring(shape, dtype, stages, consumers=consumers, multicast=name in multicast,
                               swizzle=(swizzles or {}).get(name), copies=(copies or {}).get(name, 1))
                     for name, (shape, dtype) in blocks.items()}
        # The streams `joined` names, one barrier pair between them (`rings.joined`), in the place of its
        # first: each step's loads of them awaited once and their slots given back once.
        together = tuple(name for name in joined if name in allocated and name not in gathered)
        if len(together) < 2:
            return allocated
        key, pair = joined_rings({name: allocated[name] for name in together}, multicast)
        return {key if name == together[0] else name: pair if name == together[0] else value
                for name, value in allocated.items() if name not in together[1:]}

    def opened(self, allocated: dict, views: dict | None = None) -> dict[str, Ring]:
        from dew.nn.kernels.templates import gathered  # it builds on `Ring`

        del views
        # A gathered stream's load barriers are shared memory of its own (`gathered.ring`), initialized here:
        # a program opens its rings once, every warpgroup before the roles split (`Ring.initialized`; its
        # gathered rings' users, a forward's `_launched` and `persist`, open them so).
        rings = {}
        for name, scoped in allocated.items():
            if JOINED in name:
                rings.update(opened_joined(name, scoped))
            else:
                plain = isinstance(scoped['full'].dtype, plgpu_core.BarrierType)
                rings[name] = (Ring if plain else gathered.Ring)(**scoped)
        for ring_ in rings.values():
            ring_.initialized()
        return rings

    def held_once(self, allocated: dict, views: dict, once: dict, once_views: dict
                  ) -> tuple[dict[str, Ring], dict, dict[str, Ring]]:
        # Its once streams held apart, one deep, beside its rings.
        return self.opened(allocated, views), views, self.opened(once, once_views)

    def stream(self, streams, views: dict, lo, hi, *, again: "after_walk.Decided | None" = None) -> None:
        rings = tuple(streams.values())

        def tiles(index):
            return tuple(ring.viewed(views[name], index) for name, ring in zip(streams, rings, strict=True))

        streamed(rings, tiles, lo, hi)
        if again is not None:
            pl.when(again.taken())(lambda: streamed(rings, tiles, lo, hi, counted=hi - lo))

    def output_staging(self, outputs: tuple, consumers: int, chunks: tuple[int, int] | None = None) -> list:
        # Each consumer's share of each output in its chunk buffers, not the whole tile (`output_staging`):
        # a packed one's its chunk's stored columns (`stores.Output`), a transposed one's `[columns, rows]`.
        buffers = []
        for output in map(Output.of, outputs):
            (rows, columns) = output.shape
            packed = (output,) if output.divisor > 1 or output.transposed else ()
            chunk, count = output_staging(columns, (output.dtype,), chunks, piece=self.chunk_columns(columns),
                                          packed=packed)
            stored = chunk // output.divisor
            block = (stored, rows) if output.transposed else (rows, stored)
            buffers.append(plgpu.SMEM((consumers, count, *block), output.dtype))
        return buffers

    def persist(self, streams, tiles: tuple[int, ...], schedule: Callable, issue: Callable, finish: Callable,
                *, consumers: int, axis: str, outputs: tuple, interpret: bool, in_flight: int = 0,
                unrolled: bool = False, accumulators: tuple = (), producer: int | None = None,
                pingpong: bool = False, chunks: tuple[int, int] | None = None) -> None:
        del accumulators  # its consumers' walks carry their own
        self.pingponged(pingpong)
        # The producer loads every tile's steps into the rings at a counter running across the tiles, so
        # a tile's first loads go out while the consumers store the last one (`_tiled`).
        rings = self.opened(streams)
        staging = self.output_staging(outputs, consumers, chunks)
        forms = tuple(map(Output.of, outputs))
        # Ping-pong's turns on the tensor cores (`persistent.consumed`'s `turns`).
        turns = [self.meeting(consumers)] if pingpong else []

        def loading():
            loaded, _ = persistent.tiled(tiles, schedule, functools.partial(loads, rings), axis=axis)
            drain(rings.values(), loaded)

        def consuming(smems, turns):
            return self._tiled(rings, tiles, schedule, issue, finish, smems, axis=axis, interpret=interpret,
                               in_flight=in_flight, unrolled=unrolled, turns=turns, forms=forms)

        def scoped(*refs):
            smems, turn_refs = refs[:len(staging)], refs[len(staging):]
            consumer = consuming(smems, turn_refs[0] if turn_refs else None)
            self.scheduled(consumers, consumer, loading, floor=producer)

        pl.run_scoped(scoped, *staging, *turns, collective_axes="wg")

    def _tiled(self, rings: dict[str, Ring], tiles: tuple[int, ...], schedule: Callable, issue: Callable,
               finish: Callable, smems, *, axis: str, interpret: bool, in_flight: int,
               unrolled: bool, turns=None, forms: tuple[Output, ...] = ()) -> Callable:
        """`persist`'s consumers: each walks its tiles' steps at a counter running across the tiles
        (`_tile_walk`) and stores its outputs (`_stores`), each tile its owner's alone at `turns` (ping-pong,
        `persistent.consumed`'s); the rings are left as `_ended` leaves them."""
        def run(index):
            store = self._stores(smems, index, forms)

            def walking(counter, active, steps, views):
                # `counter`: the step every ring is at.
                return self._tile_walk(rings, counter, jnp.where(active, steps, 0), views, index,
                                       interpret=interpret, in_flight=in_flight, unrolled=unrolled)

            ended = persistent.consumed(self, tiles, schedule, issue, finish, store, index, walking,
                                        axis=axis, turns=turns)
            self._ended(rings, ended, index)
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)

        return run

    def _stores(self, smems, index, forms: tuple[Output, ...] = ()) -> Callable:
        """Consumer `index`'s `store(output, value, ref, place)` (`persist`'s): its share of each output
        staged in chunks (`output_staging`: two buffers of `staged_columns`, `smems`), written from its
        issuer's layout (`Issuer.written`) or by its form's packing (`forms`, `stores.Output.write`, packed
        `divisor` tile columns a stored column, perhaps transposed), and stored by TMA, waiting only for a
        buffer's last store to read it before it writes it again (`stores.staged_chunks`,
        `stores.buffer_waits`)."""
        def store(output, value, ref, place):
            form = forms[output] if forms else None
            if form is None:
                staged_chunks(smems[output].at[index], value, ref, place, write=self.written)
                return
            if form.write is not None and isinstance(ref, Scattered):
                raise ValueError("a scattered output's rows land a row at a time as computed, no write of "
                                 "its own")
            staged_chunks(smems[output].at[index], value, ref, place, write=form.write or self.written,
                          divisor=form.divisor, transposed=form.transposed, narrowed=form.narrowed)

        return store

    def _tile_walk(self, rings: dict[str, Ring], counter, steps, views: dict, wg, *, interpret: bool,
                   in_flight: int, unrolled: bool) -> Callable:
        """`persist`'s walk of one tile's `steps` for consumer `wg` (its consumer's `walk(step, start,
        shapes, initial=)`), the rings at `counter`, each step's index counted from the tile's first, over
        the streams the layout gives the tile (`_tile_streams`)."""
        streams = self._tile_streams(rings, counter, steps, views, wg)

        def walk(step, start, shapes, *, initial=None):
            def stepped(at, slots, carry, accumulators):
                return step(at - counter, slots, carry, accumulators)

            return self.walk(streams, ((stepped, counter, counter + steps),), start, shapes, interpret,
                             in_flight=in_flight, initial=initial, unrolled=unrolled)

        return walk

    def _tile_streams(self, rings: dict[str, Ring], counter, steps, views: dict,
                      wg) -> "dict[str, Ring] | Feed":
        """The streams a tile's walk reads (`_tile_walk`): the rings, which the producer loads from
        `views`."""
        del counter, steps, views, wg
        return rings

    def _ended(self, rings: dict[str, Ring], end, wg) -> None:
        """Consumer `wg`'s rings after `persist`'s last tile, at counter `end`: the producer drains them."""
        del rings, end, wg

    def walk[C](self, streams, segments, start: C, shapes: tuple, interpret: bool, *,
                first: jax.Array | int = 0, in_flight: int = 0, initial: tuple | None = None,
                unrolled: bool = False, products: Sequence[Product] = ()
                ) -> tuple[C, tuple[jax.Array, ...]]:
        return self._walked(streams, segments, start, shapes, interpret, first=first, in_flight=in_flight,
                            initial=initial, unrolled=unrolled, products=products)

    def _walked[C](self, streams: dict[str, Ring], segments, start: C, shapes: tuple, interpret: bool, *,
                   first: jax.Array | int = 0, in_flight: int = 0, initial: tuple | None = None,
                   freed: Callable | None = None, unrolled: bool = False,
                   products: Sequence[Product] = ()) -> tuple[C, tuple[jax.Array, ...]]:
        """`walk`, `freed(counter)` run after each step's slots are given back (step `counter`'s). Each
        segment walks its steps a step a trip or, `unrolled`, a rotation of the rings (`stepping`). A step
        issuing a product ahead (`Product.ahead`) holds the next step's slot of its stream beside its own,
        so the rings hold two steps' slots at least (`Tiles.stages`' floor, `ahead_stages`), and tests where
        it stands in its range (`Slots.first`, `more`): its own slots read where it is the first, the next's
        where more follow, each under its guard, which a count of its emission reads by the step's place
        (`emission`'s place guards). Each step marked (`stepped`). A walk of no rings (one whose every
        operand is a once stream, `Walk.once`) turns one slot deep."""
        segments = stepped(segments)
        ahead = ahead_streams(products)
        stages = min((ring.stages for ring in streams.values()), default=1)
        if not 0 <= in_flight < stages:
            raise ValueError(f"a walk keeps fewer steps' products in flight than its rings' {stages} slots, "
                             f"none or some, not {in_flight}")
        if ahead and (in_flight or stages < ahead_stages(products)):
            raise ValueError(f"a walk reading {ahead} a step ahead holds two steps' slots of rings of "
                             f"{stages}, and keeps no product in flight past its step ({in_flight})")
        def given(turn: Turn, *, release: bool = False) -> None:
            # `turn`'s slots given back (`release`: here, a step its products ran past), then `freed`.
            if release:
                released(streams, turn)
            if freed is not None:
                freed(turn.counter)

        def turned(step, lo, hi, index, turn: Turn, carry, accumulators, *, before: int = 0):
            slots = Slots(streams, turn, ahead, more=index + 1 < hi, first=index == lo)
            groups = commits()
            carry, accumulators = step(index, slots, carry, accumulators)
            # The step's product groups (its completion tokens): a step `in_flight` back is done once no
            # more than this step's groups times `in_flight` are pending, however many products a step
            # issues (a walk's steps are one function each, their groups alike).
            groups = commits() - groups
            if not in_flight:
                # The step's products done before its slots are given back.
                self.settled()
                slots.done()
                given(turn)
                return carry, accumulators
            # The step's products run on while the next steps' operands are waited for: the step
            # `in_flight` back, done now, gives back its slots.
            slots.held()
            self.settled(in_flight * groups)
            if before >= in_flight:
                # The step `in_flight` back is the range's too: a rotation's, past its first steps.
                given(turn.back(in_flight, stages), release=True)
            else:
                pl.when(index >= lo + in_flight)(lambda: given(turn.back(in_flight, stages), release=True))
            return carry, accumulators

        def drained(lo, hi) -> Callable[[], None]:
            # A segment's last `in_flight` steps' products done and their slots given back, oldest first,
            # before the next segment's steps wait for those slots: its own releases start at its own `lo`.
            def drain():
                self.settled()
                for back in reversed(range(in_flight)):
                    pl.when(hi - back > lo)(lambda back=back: given(Turn.of(hi - 1 - back - first, stages),
                                                                    release=True))

            return drain

        loops: list[Loop] = []
        for step, lo, hi in segments:
            run = stepping(functools.partial(turned, step, lo, hi), lo, hi, first, stages, unrolled=unrolled)
            loops += [*run[:-1], run[-1]._replace(then=drained(lo, hi) if in_flight else None)]
        return self.accumulated(tuple(loops), start, shapes, interpret, initial)

    def passed(self, streams, lo, hi, *, first: jax.Array | int = 0, between=None) -> None:
        passed(tuple(streams.values()), lo, hi, first=first, between=between)

    def loaded(self, copies, *, issued: jax.Array | bool = True, axes: tuple | None = None) -> None:
        del axes  # copied by TMA, bytes whatever their layout
        loaded(copies, issued)

    def stored(self, parts, views, *, smem, wait: bool = True) -> None:
        # An add still reading `smem` (`added`) done first, unless the caller defers and so waits itself
        # (`Family.stored`): waiting here would drain its other tiles' deferred stores too.
        if wait:
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        stored(parts, views, smem=smem, wait=wait)

    def store_wait(self, pending: int) -> None:
        plgpu.wait_smem_to_gmem(pending, wait_read_only=True)

    def visible(self, interpret: bool) -> None:
        # TMA wrote them in the async proxy; the threads' loads read in the generic one (PTX's proxy fence).
        if interpret:
            return
        plgpu.inline_mgpu()(_fenced)()

    def added(self, value: jax.Array, view, interpret: bool, *, smem, pending: int = 0) -> None:
        if interpret:
            # Mosaic GPU's interpreter (jax 0.11.2) takes no TMA reduction.
            view[...] = view[...] + value
            return
        # A TMA reduce from `smem` a chunk of its columns at a time (`chunked`), rank-2 as the one-buffer
        # case of a rank-3 `(buffers, rows, cols)` stage (`stores.buffer_waits`, row 5272), with `pending`
        # earlier adds reading their own buffers folded into chunk 0's wait.
        buffers = tuple(smem.at[i] for i in range(smem.shape[0])) if smem.ndim == 3 else (smem,)
        chunks = chunked(value, view, buffers[0])
        waits = stores.buffer_waits(len(chunks), len(buffers))
        for index, ((piece, place), wait) in enumerate(zip(chunks, waits, strict=True)):
            buf = buffers[index % len(buffers)]
            plgpu.wait_smem_to_gmem(wait + pending if index == 0 else wait, wait_read_only=True)
            buf[...] = piece
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(buf, place, reduction_op="add")
