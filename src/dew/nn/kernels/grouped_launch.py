"""Dew: a grouped product's launches' shared base (`_GroupedLaunch`): its tile, interpretation and
stage facts, and the sizes it benchmarks, for `grouped_matmul`'s launches."""

import dataclasses
import math
from collections.abc import Callable, Hashable
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import mosaic_gpu as plgpu

from . import tuning
from .templates import base
from .templates.axes import held as held_axes
from .templates.gemm import mosaic_gpu
from .templates.gemm.hooks import NO_HOOKS, Hooks


@dataclasses.dataclass(frozen=True)
class Tile:
    """One Triton launch's output tile, its depth step, and its warps and pipeline stages."""
    block_m: int
    block_n: int
    block_k: int
    warps: int
    stages: int


INTERPRETED = plgpu.InterpretGPUParams(out_of_bounds_reads="uninitialized")
"""How a CPU interprets the template's launches on a GPU family: a read past an operand's rows, which a TMA
load zeroes, gives the poison every buffer starts with (NaN), as an output a kernel fails to store does, so
the interpreter shows both. A TPU core's interprets them as a TPU runs them, its DMAs and VMEM
(`mosaic_tpu.VmemRings.kernel`)."""


MOST_STAGES = 8
"""The deepest pipeline a Triton tile's derivation tries: past it a block of any GPU holds no 16-row step's
stages at the narrowest output tile."""

EXACT_STEP = 64
"""The most of the contraction one step of a launch with `exact_sums` takes: the tensor cores truncate
each add within a step's MMA chain, and exact sums promote only between steps."""


def _side_tile_bytes(operands, axes_of, extents: dict) -> int:
    """The bytes of one tile of each side operand at the block `extents` per launch axis, one group's of an
    operand by group (`axes.held`)."""
    return sum(math.prod(extents[axis] for axis in held_axes(axes)) * jnp.dtype(operand.dtype).itemsize
               for operand, axes in zip(operands, axes_of, strict=True))


def _sizes(specs: list[jax.ShapeDtypeStruct], seed: int, rows: int) -> jax.Array:
    """Group sizes summing to `rows` over the groups the third spec (the launch's group sizes) names,
    drawn unevenly, the first left empty where there are several, as a step routes them; a dense call's
    one group every row (emptied, its share's sum was zero and its sizes NaN cast to integers, W28-a100-2)."""
    groups = specs[2].shape[0]
    shares = np.random.default_rng(seed).dirichlet(np.ones(groups))
    if groups > 1:
        shares[0] = 0.0
    sizes = np.floor(shares / shares.sum() * rows).astype(np.int32)
    sizes[-1] += rows - sizes.sum()
    return jnp.asarray(sizes)


TRACED_TILE = 16
"""The square tile a launch's hooks are traced at to key its picks (`_GroupedLaunch.traced`)."""


class _GroupedLaunch[Config: Hashable](base.Template[Config]):
    """A grouped launch: its space is a declared tuple, its group sizes drawn as a step routes them."""
    # Annotations stay evaluated (no postponed annotations in this module):
    # tokamax serializes a launch's arguments into the program by their types.
    HOOKS: ClassVar[tuple[str, ...]] = ('hooks',)

    def bind(self, *args, **kwargs):
        """tokamax's binding, its hooks stating the dtypes a kernel traces them at where the call left them
        unstated (a launch called itself): the prologue's at the tokens' and its side operands' dtypes, the
        epilogue's at the fp32 accumulator's and its side operands', as `templates.gemm` states them, so
        every bound launch's hooks key alike (`traced`)."""
        ba = super().bind(*args, **kwargs)
        hooks = ba.arguments.get('hooks', NO_HOOKS)
        if not isinstance(hooks, Hooks):  # a stored launch's, already keyed (`traced`)
            return ba

        def named(*arrays) -> tuple[str, ...]:
            return tuple(jnp.dtype(array.dtype).name for array in arrays)

        stated = dataclasses.replace(
            hooks, prologue_dtypes=hooks.prologue_dtypes or (() if hooks.prologue is None else named(
                ba.arguments['tokens'], *ba.arguments.get('prologue_operands', ()))),
            epilogue_dtypes=hooks.epilogue_dtypes or (() if hooks.epilogue is None else (
                'float32', *named(*ba.arguments.get('epilogue_operands', ())))),
            accumulators=1 if ba.arguments.get('paired') is None else 2)
        return ba if stated == hooks else type(ba)(ba.op, {**ba.arguments, 'hooks': stated})

    ROWS: ClassVar[tuple[str, ...]]
    """The operands whose rows the launch's groups split: a product's `tokens`, a kernel gradient's
    `tokens` and `cotangent`."""

    def padded(self, ba, config: Config) -> float | None:
        """A dense call's (`dense`) output grid, its rows tiled by `block_m` and its columns by `block_n`
        (`tuning.KernelOp.padded`), the output's sides its launch's (`output_sides`); None for a grouped
        call, whose groups' rows a step routes, for a tile of no such blocks (a TPU's), and for a launch
        that states no sides."""
        sides = self.output_sides(ba)
        if not ba.arguments.get('dense') or not isinstance(config, Tile | mosaic_gpu.Tiles) or sides is None:
            return None
        rows, width = sides
        grid = -(-rows // config.block_m) * config.block_m * -(-width // config.block_n) * config.block_n
        return 1 - rows * width / grid if grid else None

    def output_sides(self, ba) -> tuple[int, int] | None:
        """The rows and columns of a dense call's output its tiles cover (`padded`): a product's (`Gmm`,
        `MosaicGmm`) and a kernel gradient's (`Tgmm`, `MosaicTgmm`) each their own; None here."""
        return None

    def prepared(self, name: str, value):
        """The launch's `Hooks` as every launch takes them (`tuning.KernelOp.prepared`): its prologue and
        epilogue each prepared, an explicit caller's and an inferred fusion's alike (`tuning.prepared`)."""
        if not isinstance(value, Hooks):
            return super().prepared(name, value)
        return dataclasses.replace(value, prologue=tuning.prepared(value.prologue),
                                   epilogue=tuning.prepared(value.epilogue))

    def traced(self, name: str, value: Callable | str | None) -> str | None:
        """The launch's `Hooks` as its picks key them (`tuning.KernelOp.traced`): each hook by what it
        computes (`tuning.program`) as a kernel traces it (`Hooks.before`, `Hooks.after`), at a
        `TRACED_TILE` square tile and each side tile as its axes lay it out there, in the dtypes the hooks
        state, beside their axes, squeezed positions and output dtypes; an already traced key as it is."""
        if not isinstance(value, Hooks):
            return value if isinstance(value, str) else None

        def program(hook, axes_of: tuple, dtypes: tuple, tiles: int = 1) -> str | None:
            if hook is None:
                return None
            tile, *sides = dtypes
            return tuning.program(hook, *(jax.ShapeDtypeStruct((TRACED_TILE, TRACED_TILE), tile),) * tiles,
                                  *(jax.ShapeDtypeStruct((TRACED_TILE,) * len(axes), dtype)
                                    for axes, dtype in zip(axes_of, sides, strict=True)))

        return "hooks:" + repr((program(value.before(), value.prologue_axes, value.prologue_dtypes),
                                program(value.after(), value.epilogue_axes, value.epilogue_dtypes,
                                        value.accumulators),
                                value.prologue_axes, value.epilogue_axes, value.out_dtypes,
                                value.prologue_broadcast, value.epilogue_broadcast, value.accumulators))

    def keyed(self, ba):
        """`tuning.KernelOp.keyed`, a call's tables (`gather`, `scatter`) bucketed as the tokens' rows are,
        and each side operand's dimension that follows the rows bucketed with the product's rows (the
        tokens', `ROWS`, or a gathered call's table's, `_rows_of`), so the launch a key measures loads side
        tiles over all its rows, and a kernel gradient's table meets its ungathered operand at one count."""
        keyed = super().keyed(ba)
        arguments = dict(keyed.arguments)
        for name in ('gather', 'scatter'):
            if arguments.get(name) is not None:
                arguments[name] = jax.tree.map(tuning.bucketed, arguments[name])
        rows = next(iter(jax.tree.leaves(arguments.get('gather'))), arguments['tokens']).shape[0]
        hooks = ba.arguments.get('hooks', NO_HOOKS)
        for name, axes_of in (('prologue_operands', hooks.prologue_axes),
                              ('epilogue_operands', hooks.epilogue_axes)):
            if name in arguments:
                arguments[name] = tuple(
                    jax.ShapeDtypeStruct(tuple(rows if axis == 'rows' else extent
                                               for axis, extent in zip(axes, side.shape, strict=True)),
                                         side.dtype)
                    for side, axes in zip(arguments[name], axes_of, strict=True))
        return type(keyed)(keyed.op, arguments)

    def operands(self, specs: list[jax.ShapeDtypeStruct], seed: int) -> list[jax.Array]:
        """`tuning.KernelOp.operands`, the group sizes (every launch's third argument) drawn as a step
        routes them over its rows (a table's, the last: a gathered or scattered product's, its zero entries
        reading and writing row 0, `gmm_call`'s `gather` and `scatter`), an integer side operand drawn as any
        other operand."""
        arrays = super().operands(specs, seed)
        last = specs[-1]
        gathered = len(specs) > 3 and last.ndim == 1 and jnp.issubdtype(last.dtype, jnp.integer)
        arrays[2] = _sizes(specs, seed, (last if gathered else specs[0]).shape[0])
        return arrays
