"""The template toolkit's TPU bundle (`MXU`): the shared mainloops on a TPU's TensorCore, through Pallas TPU.

A TPU core has no warps, warpgroups or registers a kernel budgets: its products are the MXU's, read from and
accumulated into values Mosaic TPU keeps in vregs or spills to VMEM, and its operand blocks arrive in VMEM by
DMA. So its issuer (`MxuIssuer`) multiplies a step's VMEM blocks into an fp32 value, and its transport
(`VmemRings`) streams each step's blocks into a ring of VMEM slots by asynchronous copies a step ahead (two
slots, double buffered, each awaited at its DMA semaphore), stores an output by copies from VMEM in a ladder
of static extents (`mosaic.store_rows`), and runs a persistent launch's tiles round-robin over the launch's
programs, one a core. A program is one role: no producer runs beside it, and nothing is multicast."""

import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax
from jax._src.state import types as state_types
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from dew.nn.kernels import launches
from dew.nn.kernels.templates import mosaic

STAGES = 2
"""The slots a stream's ring holds: a step's blocks arrive while the step before it multiplies (Pallas
TPU's own pipeline's double buffering)."""


class Slots:
    """A step's operand slots by stream name (`mosaic.Slots`'s interface): the walk awaits a step's blocks
    before the step and refills a slot only after the step that read it, so releasing one early is
    nothing."""

    def __init__(self, refs: dict):
        self._refs = refs

    def __getitem__(self, name: str):
        return self._refs[name]

    def release(self, name: str) -> None:
        del name

    def acquire_all(self) -> None:
        """Every slot acquired: the walk awaits a step's blocks whole before the step."""


@dataclasses.dataclass(frozen=True)
class Streams:
    """A program's operand streams opened (`VmemRings.opened`): each stream's ring (its `slots`, `(STAGES,
    *block)` in VMEM, and their DMA semaphores, `sems`) and its view `(ref, block_indices)` by name, the
    operand in HBM its steps' blocks come from, block `block_indices(j)` of step `j`."""

    rings: dict
    views: dict

    def copy(self, name: str, step, slot):
        """Step `step`'s block of stream `name` into its ring's `slot`, by an asynchronous copy."""
        ref, block_indices = self.views[name]
        slots = self.rings[name]['slots']
        block = slots.shape[1:]
        source = ref.at[tuple(pl.ds(index * size, size)
                              for index, size in zip(block_indices(step), block, strict=True))]
        return pltpu.make_async_copy(source, slots.at[slot], self.rings[name]['sems'].at[slot])


@dataclasses.dataclass(frozen=True)
class MxuIssuer(mosaic.Issuer[jax.Array]):
    """A TPU core's issuer and home: products by the MXU from VMEM blocks into an fp32 value, synchronous,
    the value Mosaic TPU's to hold in vregs or VMEM; no register layout, so a row or a column of a tile is
    the tile's own axis."""

    name = "mxu"

    rows = 128  # an MXU's side on a v5e, the least rows a product fills (a v6e's is 256)

    role_rows = 128

    operands_in_registers = False

    products_overlap = False

    column_scratch = 0  # its column sums take no scratch

    _VALUES = "a TPU core holds no register layout: its values are Mosaic TPU's, in vregs or VMEM"

    @property
    def layout(self):
        raise NotImplementedError(self._VALUES)

    @property
    def row(self):
        raise NotImplementedError(self._VALUES)

    @property
    def column(self):
        raise NotImplementedError(self._VALUES)

    @property
    def lane_columns(self) -> int:
        raise NotImplementedError(self._VALUES)

    # A tile, a row and a column are values: the members the GPU's issuers lay out are plain reads.
    def zeros(self, shape: tuple[int, ...]) -> jax.Array:
        return jnp.zeros(shape, jnp.float32)

    def iota(self, shape: tuple[int, ...], dimension: int) -> jax.Array:
        return lax.broadcasted_iota(jnp.int32, shape, dimension)

    def column_offsets(self, shape: tuple[int, int]) -> tuple[jax.Array, jax.Array]:
        # No lane decides a column: the index is its own offset.
        return jnp.zeros((), jnp.int32), self.iota(shape, 1)

    def rows_of(self, ref) -> jax.Array:
        return ref[...]

    def columns_of(self, ref) -> jax.Array:
        return ref[...]

    def tile_of(self, ref) -> jax.Array:
        return ref[...]

    def key_magnitudes(self, tile) -> jax.Array:
        return jnp.max(jnp.abs(tile[...]), axis=1).astype(jnp.float32)

    def column_sums(self, value: jax.Array, scratch, interpret: bool) -> jax.Array:
        del scratch, interpret
        return value.sum(axis=0)

    def as_row(self, vector: jax.Array) -> jax.Array:
        return vector

    def as_tile(self, value: jax.Array) -> jax.Array:
        return value

    def written(self, smem, value: jax.Array) -> None:
        smem[...] = value.astype(smem.dtype)

    def products(self, acc: jax.Array, lhs, rhs, interpret: bool, *, accumulate: bool = True) -> jax.Array:
        del interpret
        return acc + _product(lhs, rhs) if accumulate else _product(lhs, rhs)

    def settled(self, pending: int = 0) -> None:
        # Its products are synchronous: none is pending.
        del pending

    def held(self, ref):
        return ref

    def added_into(self, acc: jax.Array, value: jax.Array, interpret: bool) -> jax.Array:
        del interpret
        return acc + value.astype(jnp.float32)

    def product(self, lhs, rhs, shape: tuple[int, int], then=None, *, interpret: bool = False) -> jax.Array:
        del interpret
        if then is not None:
            then()
        return _product(lhs, rhs)

    def registers_product(self, lhs: jax.Array, rhs, interpret: bool) -> jax.Array:
        del interpret
        return _product(lhs, rhs)

    def register_products(self, acc: jax.Array, parts, rhs, interpret: bool) -> jax.Array:
        del interpret
        return acc + sum(_product(part, rhs) for part in parts)

    def rescaled_products(self, acc: jax.Array, scale, parts, rhs, interpret: bool) -> jax.Array:
        return self.register_products(acc * scale, parts, rhs, interpret)


class VmemRings(mosaic.Family[dict, Streams, jax.Array]):
    """A TPU core's transport: each stream's blocks DMA'd into a ring of VMEM slots a step ahead
    (`STAGES`), its program one role, its outputs stored from VMEM by copies of static extents. Composed
    with the MXU's issuer (`MXU`)."""

    consumers = 1

    bounded_loads = False  # a DMA past its operand's rows is refused, not filled with zeros

    stores_through_registers = False

    copies_in_pieces = False

    copies_vectors = True

    adds_vectors = False

    walks_scoped = False

    def registers(self, consumers: int, programs: int = 1) -> int:
        raise NotImplementedError("a TPU core budgets no registers: its values spill to VMEM")

    def column_unit(self, dtype) -> int:
        # A block's minor dimension in whole lanes (Mosaic TPU's tiling).
        del dtype
        return 128

    def depths(self, dtype) -> tuple[int, ...]:
        # A step's depth in whole lanes, the MXU's side on a v5e.
        del dtype
        return (128,)

    def kernel(self, entry, *, out_type, grid: tuple, grid_names: tuple, consumers: int,
               interpret: bool, approx_math: bool = True, name: str | None = None,
               cluster: int = 1, cost: pl.CostEstimate | None = None, model_flops: int | None = None,
               reducing_warpgroups: int = 0, resident: int = 1, scalars: int = 0) -> Callable:
        # A program a core, its operands and outputs left in HBM for its walks' copies; no cluster.
        del grid_names, approx_math, reducing_warpgroups, resident, model_flops
        _one(consumers, clustered=cluster > 1)

        def launched(*operands):
            # A table its programs read an element at a time in scalar memory, every other operand in HBM.
            spaces = [pltpu.SMEM if at < scalars else pl.ANY for at in range(len(operands))]
            return launches.pallas_call(
                entry, name=name or self.name, out_shape=out_type, grid=grid,
                in_specs=[pl.BlockSpec(memory_space=space) for space in spaces],
                out_specs=jax.tree.map(lambda _: pl.BlockSpec(memory_space=pl.ANY), out_type),
                compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",) * len(grid)),
                interpret=pltpu.InterpretParams() if interpret else False, cost_estimate=cost)(*operands)

        return launched

    def program_scoped(self, body, *allocations, streams):
        return pl.run_scoped(lambda *refs, **rings: body(*refs, streams=rings), *allocations, **streams)

    def roles(self, consumers: int, consumer, producer) -> None:
        _one(consumers)
        del producer
        consumer(0)

    def streams(self, blocks: dict, stages: int, consumers: int, multicast: tuple[str, ...] = (),
                swizzles: dict[str, int] | None = None, copies: dict[str, int] | None = None) -> dict:
        # Each stream's ring of `STAGES` slots, whatever the stages asked: a DMA a step ahead hides a step's.
        del stages, swizzles
        _one(consumers, clustered=bool(multicast))
        if any(count > 1 for count in (copies or {}).values()):
            raise ValueError("a TPU ring loads a step's block in one copy, not in pieces")
        return {name: {'slots': pltpu.VMEM((STAGES, *shape), dtype),
                       'sems': pltpu.SemaphoreType.DMA((STAGES,))}
                for name, (shape, dtype) in blocks.items()}

    def lifetimes(self, streams: dict, steps: int | None = None) -> tuple[tuple, tuple]:
        del steps
        return tuple(ring['slots'] for ring in streams.values()), ()

    def output_staging(self, outputs: tuple, consumers: int) -> list:
        # Each output's tile staged whole in VMEM.
        _one(consumers)
        return [pltpu.VMEM(shape, dtype) for shape, dtype in outputs]

    def opened(self, allocated, views: dict | None = None):
        if views is None:
            raise ValueError("a TPU ring's streams take their views where they open (its walk copies)")
        return Streams(allocated, views)

    def stream(self, streams, views, lo, hi, *, adds: tuple | None = None) -> None:
        # No producer role: the walk issues its own copies.
        del streams, views, lo, hi, adds

    def persist(self, streams, tiles: tuple[int, ...], schedule: Callable, issue: Callable, finish: Callable,
                *, consumers: int, axis: str, outputs: tuple, interpret: bool, in_flight: int = 0,
                unrolled: bool = False, accumulators: tuple = ()) -> None:
        # A launch's tiles round-robin over its programs, each tile's outputs staged in VMEM and copied out
        # before the next tile's walk.
        del axis, in_flight, unrolled, accumulators
        _one(consumers)
        program, programs = pl.program_id(0), pl.num_programs(0)
        total = 1
        for extent in tiles:
            total *= extent

        def run(*staging):
            def store(output, value, ref, place):
                staged = staging[output]
                staged[...] = value.astype(staged.dtype)
                mosaic.store_rows(staged, ref, *place, copy=_copied)

            def visit(turn, carry):
                index = _unravel(program + turn * programs, tiles)
                active, steps, views, tile = schedule(index)

                def walk(step, start, shapes, *, initial=None):
                    return self.walk(self.opened(streams, views), ((step, 0, steps),), start, shapes,
                                     interpret, initial=initial)

                def run_tile(walk, *, active: bool):
                    finish(0, tile, issue(0, tile, walk, active=active), store, active=active)

                pl.when(active)(lambda: run_tile(walk, active=True))
                pl.when(~active)(lambda: run_tile(mosaic.unwalked(self), active=False))
                return carry

            lax.fori_loop(0, (total - program + programs - 1) // programs, visit, 0)

        pl.run_scoped(run, *self.output_staging(outputs, consumers))

    def walk[C](self, streams, segments, start: C, shapes: tuple, interpret: bool, *,
                first: jax.Array | int = 0, in_flight: int = 0, initial: tuple | None = None,
                unrolled: bool = False, held_past: tuple[str, ...] = (), finish: Callable | None = None
                ) -> tuple[C, tuple[jax.Array, ...]]:
        # Each step's blocks awaited before it and the next step's copied while it multiplies; a segment
        # of no steps copies nothing. Its products are synchronous: no stage outlives its step.
        del first, in_flight, unrolled, interpret
        if held_past or finish is not None:
            raise NotImplementedError(f"{self.name}'s steps give their stages back as each ends")
        names = tuple(streams.views)
        carry = (start, self.started(shapes, initial))
        for step, lo, hi in segments:
            @pl.when(hi > lo)
            def _(lo=lo):
                for name in names:
                    streams.copy(name, lo, 0).start()

            def body(index, carry, step=step, lo=lo, hi=hi):
                slot = (index - lo) % STAGES

                @pl.when(index + 1 < hi)
                def _():
                    for name in names:
                        streams.copy(name, index + 1, 1 - slot).start()

                for name in names:
                    streams.copy(name, index, slot).wait()
                slots = Slots({name: streams.rings[name]['slots'].at[slot] for name in names})
                return step(index, slots, *carry)

            carry = lax.fori_loop(lo, hi, body, carry)
        return carry

    def passed(self, streams, lo, hi, *, first: jax.Array | int = 0, between=None) -> None:
        # Its walk copies only the steps it reads: a passed step loads nothing.
        del streams, lo, hi, first
        if between is not None:
            raise NotImplementedError("a TPU walk passes no steps between others")

    def loaded(self, copies, *, issued: jax.Array | bool = True, axes: tuple | None = None) -> None:
        # Each copied whole into its VMEM ref, at its semaphore, and awaited.
        del issued, axes
        for view, ref, semaphore in copies:
            pltpu.sync_copy(view, ref) if semaphore is None else _awaited(view, ref, semaphore)

    def load_barriers(self, count: int):
        return pltpu.SemaphoreType.DMA((count,))

    def staged_memory(self, shape: tuple[int, ...], dtype):
        # Its one role's staging in VMEM (`staging` keeps the GPU's up to two roles' shape, so a template
        # indexes it alike), its copies' DMA semaphores its `load_barriers`.
        return pltpu.VMEM(shape, dtype)

    def meet(self, halves) -> None:
        raise NotImplementedError("a TPU program is one role: it meets no other")

    def published(self) -> None:
        # Its products read VMEM by its own loads: nothing to publish.
        pass

    def stored(self, parts, views, *, smem, wait: bool = True) -> None:
        del wait
        for part, view in zip(parts, views, strict=True):
            smem[...] = part.astype(smem.dtype)
            pltpu.sync_copy(smem, view)

    def added(self, value: jax.Array, view, interpret: bool, *, smem, pending: int = 0) -> None:
        del pending
        raise NotImplementedError("a TPU core adds into HBM by no atomic: a split contraction's partial "
                                  "sums take a place each")


@dataclasses.dataclass(frozen=True)
class Mxu(MxuIssuer, VmemRings):
    """A TPU core's bundle: the MXU's issuer on VMEM rings."""


MXU = Mxu()
"""The TPU bundle (`bundles.of` on a TPU)."""


def _one(consumers: int, *, clustered: bool = False) -> None:
    """A TPU program is one role, and no cluster shares its blocks."""
    if consumers != 1 or clustered:
        raise NotImplementedError(f"a TPU program runs one role, not {consumers}, and no cluster")


def _operand(operand) -> tuple[jax.Array, bool]:
    """A product's operand as the MXU reads it: a value, or a VMEM block's (a ref, or a view of one read as
    the transpose of its stored block, its last transform `.transpose((1, 0))`), and whether it is that
    transpose, which the product contracts over rather than moving."""
    if isinstance(operand, jax.Array):
        return operand, False
    transforms = operand.transforms if isinstance(operand, state_types.TransformedRef) else ()
    if transforms and isinstance(transforms[-1], state_types.TransposeTransform):
        if transforms[-1].permutation != (1, 0):
            raise NotImplementedError(f"a TPU product reads a block or its transpose, not {transforms[-1]}")
        return state_types.TransformedRef(operand.ref, transforms[:-1])[...], True
    return operand[...], False


def _product(lhs, rhs) -> jax.Array:
    """`lhs @ rhs` in fp32 on the MXU, either operand read as the transpose of its stored block where it is
    one (a contraction over its stored block's rows or columns, as the MXU takes it)."""
    left, left_transposed = _operand(lhs)
    right, right_transposed = _operand(rhs)
    contracting = ((0 if left_transposed else 1,), (1 if right_transposed else 0,))
    return lax.dot_general(left, right, (contracting, ((), ())), preferred_element_type=jnp.float32)


def _unravel(flat, extents: tuple[int, ...]) -> tuple:
    """`flat`'s index in a grid of `extents`, its last axis fastest."""
    index = []
    for extent in reversed(extents):
        index.append(flat % extent)
        flat = flat // extent
    return tuple(reversed(index))


def _copied(source, destination) -> None:
    """`source` (in VMEM) to `destination` (in HBM): `mosaic.store_rows`' copy on a TPU."""
    pltpu.sync_copy(source, destination)


def _awaited(view, ref, semaphore) -> None:
    """`view` copied into `ref` at `semaphore`, and awaited."""
    copy = pltpu.make_async_copy(view, ref, semaphore)
    copy.start()
    copy.wait()
