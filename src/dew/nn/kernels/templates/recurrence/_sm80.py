"""The mma.sync instruction family's bundle (Ampere and every 8.x and 12.x GPU; `generation.family`
'mma.sync') for the recurrence template's Mosaic mainloop (`mainloop`), under the toolkit's `mosaic.Family`:
one warpgroup a program, its operand tiles streamed into swizzled shared memory by cp.async (no TMA before
Hopper), each product a synchronous warp-level MMA of 16-bit operands loaded from shared memory into its
operand layouts (ldmatrix) and accumulated in registers. A register operand reaches a product through shared
memory (`stage`): Mosaic GPU's layout inference finds no cast from an accumulator's layout to an operand's."""

import contextlib
import contextvars

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import mosaic


def swizzled(rows: int, columns: int, dtype) -> tuple:
    """The shared-memory transforms of a `[rows, columns]` operand tile: 8-row tiles as wide as the swizzle
    its rows' bytes allow, swizzled, as Mosaic GPU's MMA operand loads (ldmatrix) read them."""
    swizzle = plgpu.find_swizzle(columns * jnp.dtype(dtype).itemsize * 8)
    return plgpu.TilingTransform((8, swizzle // jnp.dtype(dtype).itemsize)), plgpu.SwizzleTransform(swizzle)


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
    """One operand's stream: its slots in shared memory, and the views a step copies into them, which the
    producer's `stream` names and the consumer's `walk` copies (one role does both)."""

    def __init__(self, smem):
        self.smem = smem
        self.views = None

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


class MmaSync(mosaic.Family):
    """mma.sync's bundle: a warpgroup's four warps down a 64-row accumulator (Mosaic GPU's `MMA_ACC`),
    operands of 16 bits, loads by cp.async in `stages`-deep rings each walk waits on in turn. Its products are
    synchronous, so a product's operands are read once it returns."""

    name = "mma.sync"
    rows = 64
    layout = plgpu.Layout.MMA_ACC(jnp.bfloat16)
    row = layout.reduce(1)
    column = layout.reduce(0)

    def kernel(self, entry, *, out_type, grid: tuple, grid_names: tuple, consumers: int, interpret: bool):
        if consumers != 1:
            raise NotImplementedError("mma.sync's programs are one warpgroup")
        def traced(*refs):
            with _interpreted(interpret):
                entry(*refs)

        return plgpu.kernel(traced, out_type=out_type, grid=grid, grid_names=grid_names,
                            compiler_params=mosaic.compiler_params(),
                            interpret=plgpu.InterpretGPUParams() if interpret else None)

    def roles(self, wg, consumers: int, consumer, producer) -> None:
        # One role: the producer names its views, and the consumer's walk copies them.
        del wg, consumers
        producer()
        consumer(0)

    def streams(self, blocks: dict, stages: int, consumers: int) -> dict:
        del consumers
        return {name: plgpu.SMEM((stages, *shape), dtype, transforms=swizzled(*shape, dtype))
                for name, (shape, dtype) in blocks.items()}

    def opened(self, allocated: dict) -> dict:
        return {name: _Stream(smem) for name, smem in allocated.items()}

    def stream(self, streams: dict, views, lo, hi) -> None:
        del lo, hi
        for stream in streams.values():
            stream.views = views

    def walk[C](self, streams: dict, segments, start: C, shapes: tuple, interpret: bool, *,
                first: jax.Array | int = 0) -> tuple[C, tuple[jax.Array, ...]]:
        del interpret
        names = tuple(streams)
        stages = min(stream.stages for stream in streams.values())
        ahead = stages - 1

        def copy(index, slot):
            views = next(iter(streams.values())).views(index)
            for name in names:
                if _INTERPRETED.get():
                    streams[name].smem.at[slot][...] = views[name][...]
                    continue
                plgpu.copy_gmem_to_smem(views[name], streams[name].smem.at[slot], None,
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
                    # One slot: the last step's reads done before its next load.
                    self.settled()
                    copy(index, 0)
                # The wait's barrier also ends every warp's reads of the slot the next copy refills.
                if not _INTERPRETED.get():
                    plgpu.wait_gmem_to_smem(ahead * len(names))
                if ahead:
                    copy(jnp.minimum(index + ahead, hi - 1), lax.rem(index + ahead - first, stages))
                return step(index, _Slots(streams, lax.rem(index - first, stages)), *both)

            both = lax.fori_loop(lo, hi, stepped, both)
        return both

    def passed(self, streams: dict, lo, hi, *, first: jax.Array | int = 0, between=None) -> None:
        del streams, first
        if between is not None:
            pl.loop(lo, hi)(lambda _: between())

    def operand(self, ref, side: str):
        """A shared-memory tile `ref` (a transposed view too) in the left or right operand's layout."""
        layout = plgpu.Layout.MMA_LHS if side == "lhs" else plgpu.Layout.MMA_RHS
        return plgpu.load(ref, layout=layout(ref.dtype))

    def product(self, lhs, rhs, shape: tuple[int, int], then=None) -> jax.Array:
        if _INTERPRETED.get():
            out = jnp.dot(lhs[...], rhs[...], preferred_element_type=jnp.float32)
        else:
            out = plgpu.mma(self.zeros(shape), self.operand(lhs, "lhs"), self.operand(rhs, "rhs"))
        if then is not None:
            then()
        return out

    def registers_product(self, lhs: jax.Array, rhs, interpret: bool) -> jax.Array:
        del interpret
        lhs = plgpu.layout_cast(lhs, plgpu.Layout.MMA_LHS(lhs.dtype))
        return plgpu.mma(self.zeros((lhs.shape[0], rhs.shape[1])), lhs, self.operand(rhs, "rhs"))

    def register_products(self, acc, parts, rhs, interpret: bool):
        del interpret
        for part, exponent in parts:
            if exponent is not None:
                raise NotImplementedError("a scaled part's product is no in-place accumulation (fp16's)")
            part = plgpu.layout_cast(part, plgpu.Layout.MMA_LHS(part.dtype))
            acc = plgpu.mma(acc, part, self.operand(rhs, "rhs"))
        return acc

    def staged_products(self, acc, parts, rhs, buffers, interpret: bool, *, transposed: bool = False):
        del interpret
        for place, (part, exponent) in enumerate(parts):
            if exponent is not None:
                raise NotImplementedError("a scaled part's product is no in-place accumulation (fp16's)")
            lhs = self.stage(part, buffers.at[place])
            acc = plgpu.mma(acc, self.operand(lhs.T if transposed else lhs, "lhs"), self.operand(rhs, "rhs"))
        return acc

    def stage(self, value, buffer):
        """`value` in the shared-memory tile `buffer`, in its dtype, once every warp's reads of `buffer` are
        done, for a product to read: a register operand's way to an MMA."""
        self.settled()
        buffer[...] = value.astype(buffer.dtype)
        plgpu.commit_smem()
        return buffer

    def settled(self) -> None:
        # The products are synchronous; the warpgroup's reads of shared memory are done at its barrier.
        plgpu.commit_smem()

    def zeros(self, shape: tuple[int, ...]) -> jax.Array:
        return self.full(shape, 0.0)

    def full(self, shape: tuple[int, ...], value: float) -> jax.Array:
        layout = self.layout if len(shape) == 2 else self.row
        return plgpu.layout_cast(jnp.full(shape, value, jnp.float32), layout)

    def iota(self, shape: tuple[int, int], dimension: int) -> jax.Array:
        return plgpu.broadcasted_iota(jnp.int32, shape, dimension, layout=self.layout)

    def rows_of(self, ref) -> jax.Array:
        return plgpu.load(ref, layout=self.row, optimized=False)

    def columns_of(self, ref) -> jax.Array:
        """A column vector read from `ref` (`[columns]`) in a column's layout. Two loads of one view in a
        row's and a column's layouts are one load to Mosaic GPU's layout inference, which then finds none for
        it."""
        return plgpu.load(ref, layout=self.column, optimized=False)

    def as_row(self, vector: jax.Array) -> jax.Array:
        return plgpu.layout_cast(vector, self.row)


MMA_SYNC = MmaSync()
"""The mma.sync family's bundle."""
