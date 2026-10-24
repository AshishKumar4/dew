"""The mma.sync instruction family's bundle (Ampere and every 8.x and 12.x GPU; `generation.family`
'mma.sync') for the recurrence template's Mosaic mainloop (`mainloop`), under the toolkit's `mosaic.Family`:
one warpgroup a program, its operand tiles streamed into swizzled shared memory by cp.async (no TMA before
Hopper), each product a synchronous warp-level MMA of 16-bit operands loaded from shared memory into its
operand layouts (ldmatrix) and accumulated in registers. A register operand reaches a product through shared
memory (`staged`): Mosaic GPU's layout inference finds no cast from an accumulator's layout to an
operand's."""

import contextlib
import contextvars

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import mosaic
from dew.nn.kernels.templates.mosaic_sm80 import swizzled

_INTERPRETED = contextvars.ContextVar("interpreted", default=False)
"""Whether the kernel being traced runs in Mosaic GPU's interpreter, which lowers no mma.sync and no cp.async:
its products are then the dots they compute, and its loads plain copies."""


@contextlib.contextmanager
def _interpreted(interpret: bool):
    token = _INTERPRETED.set(interpret)
    try:
        yield
    finally:
        _INTERPRETED.reset(token)


class _Stream:
    """One operand's stream: its slots in shared memory, and its view `(ref, block)` each step copies `block`
    rows of into them (`Family.opened`'s), which the walk copies (one role loads and computes)."""

    def __init__(self, smem, view):
        self.smem, self.view = smem, view

    @property
    def stages(self) -> int:
        return int(self.smem.shape[0])


class _Slots:
    """A walk step's slots by stream name (`mosaic.Slots`' reading side): each step's tiles are in shared
    memory before the step runs, and a slot is reused only after every warp has passed the next step's wait.
    """

    def __init__(self, streams: dict, slot):
        self._streams, self._slot = streams, slot

    def __getitem__(self, name: str):
        return self._streams[name].smem.at[self._slot]

    def release(self, name: str) -> None:
        del name


class MmaSync(mosaic.Family[dict, dict, jax.Array]):
    """mma.sync's bundle: a warpgroup's four warps down a 64-row accumulator (Mosaic GPU's `MMA_ACC`),
    operands of 16 bits, loads by cp.async in `stages`-deep rings each walk waits on in turn. Its products are
    synchronous, so a product's operands are read once it returns, and its accumulators are register
    arrays."""

    name = "mma.sync"
    rows = 64
    consumers = 1

    @property
    def layout(self):
        return plgpu.Layout.MMA_ACC(jnp.bfloat16)

    @property
    def row(self):
        return self.layout.reduce(1)

    @property
    def column(self):
        return self.layout.reduce(0)

    def kernel(self, entry, *, out_type, grid: tuple, grid_names: tuple, consumers: int, interpret: bool):
        if consumers != 1:
            raise NotImplementedError("mma.sync's programs are one warpgroup")
        def traced(*refs):
            with _interpreted(interpret):
                entry(*refs)

        traced.__name__ = entry.__name__

        return plgpu.kernel(traced, out_type=out_type, grid=grid, grid_names=grid_names,
                            compiler_params=mosaic.compiler_params(),
                            interpret=plgpu.InterpretGPUParams() if interpret else None)

    def scoped(self, body, *allocations, streams) -> None:
        pl.run_scoped(lambda *refs, **slots: body(*refs, streams=slots), *allocations, **streams)

    def roles(self, consumers: int, consumer, producer) -> None:
        # One role: the producer's loads are the consumer's walk's.
        del consumers
        producer()
        consumer(0)

    def streams(self, blocks: dict, stages: int, consumers: int) -> dict:
        del consumers
        return {name: plgpu.SMEM((stages, *shape), dtype, transforms=swizzled(shape, dtype))
                for name, (shape, dtype) in blocks.items()}

    def opened(self, allocated: dict, views: dict | None = None) -> dict:
        return {name: _Stream(smem, None if views is None else views[name])
                for name, smem in allocated.items()}

    def stream(self, streams, views: dict, lo, hi) -> None:
        # The walk copies each step's tiles itself.
        del streams, views, lo, hi

    def walk[C](self, streams, segments, start: C, shapes: tuple, interpret: bool, *,
                first: jax.Array | int = 0, in_flight: int = 0) -> tuple[C, tuple[jax.Array, ...]]:
        # A product is done when it returns: no step's stays in flight past it.
        del interpret, in_flight
        names = tuple(streams)
        stages = min(stream.stages for stream in streams.values())
        ahead = stages - 1

        def copy(index, slot):
            for name in names:
                ref, block = streams[name].view
                view = ref.at[pl.ds(index * block, block)]
                if _INTERPRETED.get():
                    streams[name].smem.at[slot][...] = view[...]
                    continue
                plgpu.copy_gmem_to_smem(view, streams[name].smem.at[slot], None,
                                        oob_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS)

        both = (start, tuple(self.zeros(shape) for shape in shapes))
        for step, lo, hi in segments:
            # A copy past the segment loads its last step again into a slot no step reads (cp.async takes no
            # predicate), so every step commits one group a stream and the groups in flight past a step's
            # are always the steps ahead's. A segment takes at least one step.
            for back in range(ahead):
                copy(jnp.minimum(lo + back, hi - 1), lax.rem(lo + back - first, stages))

            def stepped(index, both, step=step, hi=hi):
                if not ahead:
                    # One slot: the last step's reads done (the wait's barrier, no copy in flight) before
                    # its next load.
                    if not _INTERPRETED.get():
                        plgpu.wait_gmem_to_smem(0)
                    copy(index, 0)
                # The wait's barrier also ends every warp's reads of the slot the next copy refills.
                if not _INTERPRETED.get():
                    plgpu.wait_gmem_to_smem(ahead * len(names))
                if ahead:
                    copy(jnp.minimum(index + ahead, hi - 1), lax.rem(index + ahead - first, stages))
                return step(index, _Slots(streams, lax.rem(index - first, stages)), *both)

            both = lax.fori_loop(lo, hi, stepped, both)
        return both

    def passed(self, streams, lo, hi, *, first: jax.Array | int = 0, between=None) -> None:
        del streams, first
        if between is not None:
            pl.loop(lo, hi)(lambda _: between())

    def loaded(self, copies) -> None:
        for view, ref, _ in copies:
            if _INTERPRETED.get():
                ref[...] = view[...]
            else:
                plgpu.copy_gmem_to_smem(view, ref, None, oob_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS)
        if not _INTERPRETED.get():
            plgpu.wait_gmem_to_smem(0)

    def stored(self, parts, views, *, smem) -> None:
        # Registers to global memory directly: no TMA before Hopper.
        del smem
        for part, view in zip(parts, views, strict=True):
            view[...] = part.astype(view.dtype)

    def added(self, value: jax.Array, view, interpret: bool, *, smem) -> None:
        raise NotImplementedError("mma.sync's bundle adds into global memory by no TMA reduction (sm90's)")

    def held(self, ref):
        return ref

    def operand(self, ref, side: str):
        """A shared-memory tile `ref` (a transposed view too) in the left or right operand's layout."""
        layout = plgpu.Layout.MMA_LHS if side == "lhs" else plgpu.Layout.MMA_RHS
        return plgpu.load(ref, layout=layout(ref.dtype))

    def product(self, lhs, rhs, shape: tuple[int, int], then=None) -> jax.Array:
        out = self.products(self.zeros(shape), lhs, rhs, _INTERPRETED.get())
        if then is not None:
            then()
        return out

    def products(self, acc, lhs, rhs, interpret: bool) -> jax.Array:
        # Mosaic GPU's interpreter lowers no mma.sync: the dot it computes.
        del interpret
        if _INTERPRETED.get():
            return acc + jnp.dot(lhs[...], rhs[...], preferred_element_type=jnp.float32)
        return plgpu.mma(acc, self.operand(lhs, "lhs"), self.operand(rhs, "rhs"))

    def registers_product(self, lhs: jax.Array, rhs, interpret: bool) -> jax.Array:
        acc = self.zeros((lhs.shape[0], rhs.shape[1]))
        return self.register_products(acc, ((lhs, None),), rhs, interpret)

    def register_products(self, acc, parts, rhs, interpret: bool):
        del interpret
        mosaic.unscaled(parts)
        for part, _ in parts:
            part = plgpu.layout_cast(part, plgpu.Layout.MMA_LHS(part.dtype))
            acc = plgpu.mma(acc, part, self.operand(rhs, "rhs"))
        return acc

    def staged_products(self, acc, parts, rhs, buffers, interpret: bool, *, transposed: bool = False):
        mosaic.unscaled(parts)
        for place, (part, _) in enumerate(parts):
            lhs = self.staged(part, buffers.at[place])
            acc = self.products(acc, lhs.T if transposed else lhs, rhs, interpret)
        return acc

    def rescaled_products(self, acc, scale, parts, rhs, interpret: bool):
        return self.register_products(scale(acc), parts, rhs, interpret)

    def settled(self, pending: int = 0) -> None:
        # The products are synchronous: each has read its operands when it returns.
        del pending

MMA_SYNC = MmaSync()
"""The mma.sync family's bundle."""
