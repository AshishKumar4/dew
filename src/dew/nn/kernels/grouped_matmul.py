"""The grouped matmul's kernel launches, which the GEMM template (`kernels.gemm`) runs and
differentiates (`dew.nn.moe.expert_projection` through it), and XLA's portable projection.

Three products, in MaxText's megablox shape: the forward `gmm`, the input gradient `gmm` against the
transposed kernel, and the kernel gradient `tgmm`. Each is a template launch (`templates.base`) with
its implementations taken per call in a fixed order among those that run it: the template's Pallas Mosaic
kernels (`templates.gemm.mosaic_gpu`, route 'mosaic') on their instruction family's bundle, a GPU's or a
TPU core's (`templates.mosaic_tpu.MXU`), then JAX's Pallas Triton kernels (`ragged_dot.py`, vendored), on
every CUDA GPU from sm80 and on a CPU that interprets them, for a call the template does not take; and
CUTLASS's grouped GEMMs (`kernels/cutlass`, the dew-cutlass wheel) where it is installed, a vendored
kernel that runs on its declared cells (`DECLARED`) and by name (`generation.chosen`). The grouped
selector (`GROUPED`) reads which products they compute exactly (the product's declaration,
`templates.product.declared`) and which launch takes a call. They run only for their target's
lowering, CUDA's or a TPU's (`_on_platform`), chosen where the call lowers
(`jax.lax.platform_dependent`), so a CPU computation on a GPU host lowers to `jax.lax.ragged_dot`; an
explicit request interprets them on the CPU, as the CPU suite checks them.

Each launch's config is the one it derives for the GPU from facts, the same on every run; nothing ranks
configs (bible §6). JAX's one tile for every shape left a wide expert's products at half an A100's cuBLAS
rate (a 4096 by 14336 expert's forward 10.4 ms where 5.3 ms is the derived tile's, the kernel gradient 18.3
against 8.4).
"""

import abc
import dataclasses
import functools
import importlib
import importlib.util
import math
from collections.abc import Callable, Hashable
from typing import ClassVar, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.dtypes import promote_dtype
from flax.typing import Dtype, PrecisionLike
from jax._src.pallas.mosaic import core as tpu_core, lowering as mosaic_tpu_lowering
from jax._src.pallas.mosaic_gpu import (
    core as plgpu_core,  # the lowering's semantics (`_lowers`)
    lowering as mosaic_gpu_lowering,
)
from jax.custom_derivatives import SymbolicZero
from jax.experimental.pallas import mosaic_gpu as plgpu

from ..derivatives import Protected, contracted, declared_tangent, marked_call_p
from ..precision import accumulated, at_least_fp32, rounded_operand
from ..programs import equations
from . import generation, selection, tuning
from .generation import INHERENT
from .templates import across, base, bundles, mosaic, report, resources, swizzling
from .templates.axes import held as held_axes
from .templates.gemm import mosaic_gpu
from .templates.product import EIGHT, SCALE_DEPTH, Product


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


@dataclasses.dataclass(frozen=True)
class Hooks:
    """A launch's prologue and epilogue (`templates.gemm`): each a JAX function of a tile and its side
    tiles, or None, with each side operand's launch axes ('rows', 'depth', 'columns'), the positions of
    each side operand's dimensions of one the launch squeezed out (the hook gets them back), the dtypes of
    the epilogue's outputs where it returns several, and the dtypes a kernel traces each hook at (its tile's
    then each side tile's: `prologue_dtypes`, `epilogue_dtypes`). The side operands themselves are the
    launch's `prologue_operands` and `epilogue_operands`. A launch's picks key each hook by what it
    computes (`_GroupedLaunch.traced`), never by the function object. A `paired` product's epilogue takes its
    two accumulators before its side tiles (`gemm.gemm`'s `paired`)."""
    prologue: object = None
    prologue_axes: tuple = ()
    epilogue: object = None
    epilogue_axes: tuple = ()
    out_dtypes: tuple | None = None
    prologue_broadcast: tuple = ()
    epilogue_broadcast: tuple = ()
    prologue_dtypes: tuple = ()
    epilogue_dtypes: tuple = ()
    accumulators: int = 1
    """The accumulator tiles the epilogue takes before its side tiles: a launch's binding states them from
    its bound operands (two where a second matrix is bound, `gmm_call`'s `paired`; `_GroupedLaunch.bind`),
    never a caller."""

    def before(self):
        """The prologue as a kernel traces it: on the squeezed side tiles."""
        return _unsqueezing(self.prologue, self.prologue_broadcast)

    def after(self):
        """The epilogue as a kernel traces it: on the squeezed side tiles."""
        return _unsqueezing(self.epilogue, self.epilogue_broadcast, self.accumulators)

    def kernel_options(self) -> dict:
        return {'prologue': self.before(), 'prologue_axes': self.prologue_axes, 'epilogue': self.after(),
                'epilogue_axes': self.epilogue_axes, 'out_dtypes': self.out_dtypes}


def _unsqueezing(hook, positions: tuple, tiles: int = 1):
    """`hook` given its side tiles' dimensions of one back at `positions` (one tuple per side operand), its
    first `tiles` arguments the launch's tiles."""
    if hook is None or not any(positions):
        return hook
    return functools.partial(_unsqueezed, hook, positions, tiles)


def _unsqueezed(hook, positions: tuple, tiles: int, *arguments):
    """`hook` on its first `tiles` arguments and the side tiles after them with their dimensions of one
    back at `positions`, broadcast across its first tile's extent there (aligned to its trailing axes, as
    the hook's own arithmetic broadcasts them): Mosaic's layout inference lays out a broadcast of a held
    vector to a tile, as a side by group's (`axes.by_rows`), and no dimension of one (GMMGATE-h100-8)."""
    shape = arguments[0].shape

    def restored(side, where):
        if not where:
            return side
        rank = side.ndim + len(where)
        offset = len(shape) - rank
        if offset < 0:
            raise ValueError(f"a side tile of {rank} dimensions given back beside a tile of {len(shape)}")
        kept = [axis for axis in range(rank) if axis not in where]
        target = [0] * rank
        for axis in where:
            target[axis] = shape[offset + axis]
        for place, axis in enumerate(kept):
            target[axis] = side.shape[place]
        return jax.lax.broadcast_in_dim(side, tuple(target), tuple(kept))

    sides = arguments[tiles:]
    return hook(*arguments[:tiles], *(restored(side, where)
                                      for side, where in zip(sides, positions, strict=True)))


NO_HOOKS = Hooks()

MOST_STAGES = 8
"""The deepest pipeline a Triton tile's derivation tries: past it a block of any GPU holds no 16-row step's
stages at the narrowest output tile."""

EXACT_STEP = 64
"""The most of the contraction one step of a launch with `exact_sums` takes: the tensor cores truncate
each add within a step's MMA chain, and exact sums promote only between steps."""


def _lowers(hook, tiles: tuple, sides, rules) -> bool:
    """Whether a Pallas lowering's `rules` hold a rule for every primitive of `hook` on its `tiles` and its
    side tiles: Mosaic GPU's, at warpgroup semantics, have none for a transpose, which Triton's lowering
    takes. A marked span (`derivatives.marked_call_p`) lowers as its body, whose primitives are read."""
    if hook is None:
        return True
    return all(equation.primitive in rules or equation.primitive is marked_call_p
               for equation in equations(jax.make_jaxpr(hook)(*tiles, *sides).jaxpr))


def _hooks_lower(ba, dtype, rules) -> bool:
    """Whether a lowering's `rules` take the launch's hooks (`_lowers`), traced on 128-wide tiles: the
    primitives a hook makes do not depend on its tile's extents."""
    hooks = ba.arguments['hooks']

    def tiles(name, axes_of):
        return [jax.ShapeDtypeStruct((128,) * len(axes), side.dtype)
                for side, axes in zip(ba.arguments.get(name, ()), axes_of, strict=True)]

    return (_lowers(hooks.before(), (jax.ShapeDtypeStruct((128, 128), dtype),),
                    tiles('prologue_operands', hooks.prologue_axes), rules)
            and _lowers(hooks.after(), (jax.ShapeDtypeStruct((128, 128), jnp.float32),) * hooks.accumulators,
                        tiles('epilogue_operands', hooks.epilogue_axes), rules))


def _evaluates_exponential(ba) -> bool:
    """Whether the launch's epilogue evaluates a function of the special-function units
    (`dew.algebra.cost.EXPONENTIALS`), traced on 128-wide tiles as `_hooks_lower` traces it."""
    from dew.algebra.cost import EXPONENTIALS

    hooks = ba.arguments['hooks']
    if hooks.epilogue is None:
        return False
    sides = [jax.ShapeDtypeStruct((128,) * len(axes), side.dtype)
             for side, axes in zip(ba.arguments.get('epilogue_operands', ()), hooks.epilogue_axes,
                                   strict=True)]
    accumulators = (jax.ShapeDtypeStruct((128, 128), jnp.float32),) * hooks.accumulators
    return any(equation.primitive.name in EXPONENTIALS
               for equation in equations(jax.make_jaxpr(hooks.after())(*accumulators, *sides).jaxpr))


def _output_bytes(ba) -> int:
    """The bytes of one output element of each of the launch's outputs, together."""
    return sum(jnp.dtype(dtype).itemsize
               for dtype in (ba.arguments['hooks'].out_dtypes or (ba.arguments['out_dtype'],)))


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


class _Triton(_GroupedLaunch[Tile]):
    """JAX's Pallas Triton kernels (`ragged_dot`), on every CUDA GPU from sm80, and on a CPU where the
    call interprets them."""
    name: ClassVar[str] = 'triton'
    config_cls: ClassVar[type[Tile]] = Tile
    GENERATES: ClassVar[bool] = True

    def eligible(self, ba, target) -> bool:
        # Every CUDA GPU the template sends the kernels to (`kernels.gemm`), and
        # a CPU, which interprets them, or traces the CUDA branch a GPU's lowering takes; never a TPU (a
        # paired product the selector hands on apart: `_takes_pair`).
        return generation.capability(target.generation) is not None and self.derived(ba, target) is not None

    def derived(self, ba, target) -> Tile | None:
        """The one tile the launch runs on `target`, from facts (the ranked set and its cost model deleted,
        bible §6): eight warps, CUTLASS's and cuBLAS's Ampere block; the widest output tile from 128 x 256,
        halved columns first then rows, whose fp32 accumulator (two with exact sums) its threads hold
        (`legal`); the first contraction step the launch takes (`steps`) at which a block holds three stages,
        two loading ahead of the one consumed, else two; and the most stages that step's block holds
        (`staged_bytes`). A dense call takes the first such tile dividing its depth and width where one
        does: the kernel pads to one that does not. On an A100 a product's tile is 128 x 256 a step of 64
        at four stages: at that step three stages took 0.790 (forward) and 0.793 (input gradient) of two's
        time, a geomean over 6 grouped and 4 dense rows, where the ranked set's 128-deep step held two
        (PINS2-triton); PINS3 times the rule's tiles against the ranked set's."""
        rows, columns = 128, 256
        shapes = [(rows, columns)]
        while rows > 16 or columns > 16:
            rows, columns = (rows, columns // 2) if columns > rows else (rows // 2, columns)
            shapes.append((rows, columns))
        extents = self._dense_extents(ba) if ba.arguments.get('dense') else None
        found = None
        for output in shapes:
            for floor in (3, 2):
                for step in self.steps(ba):
                    tile = next((tile for stages in range(MOST_STAGES, floor - 1, -1)
                                 if self.legal(ba, tile := self.tile(output, step, stages), target)), None)
                    if tile is None:
                        continue
                    if extents is None or (extents[0] % self.contraction(tile) == 0
                                           and extents[1] % output[1] == 0):
                        return tile
                    found = found or tile
        return found

    def legal(self, ba, config: Tile, target) -> bool:
        """`Template.legal`, its accumulator (two with exact sums) in its threads' registers, an exactly
        summed step at most `EXACT_STEP` deep (a step's own MMA chain truncates its adds; exact sums
        promote between steps: H100's 128-row step left 1.55x IEEE's error, a 64-row one 1.03x,
        GMMGATE-h100-5), and on Hopper each warpgroup's share of the rows a multiple of wgmma's 64 (a
        32-row share falls back to Ampere's MMA, and an accumulator of 256 registers a thread spills: 1.2 to
        8.5 times slower on an H100, GMM-h100-2)."""
        rows, columns = self.output(config)
        accumulators = 2 if ba.arguments.get('exact_sums') else 1
        held = accumulators * rows * columns // (32 * config.warps) < target.multiprocessor.thread_registers
        short = not ba.arguments.get('exact_sums') or self.contraction(config) <= EXACT_STEP
        shared = not (target.wgmma and rows // max(config.warps // 4, 1) % 64)
        return held and short and shared and super().legal(ba, config, target)

    @staticmethod
    @abc.abstractmethod
    def steps(ba) -> tuple[int, ...]:
        """The contraction steps the launch tries, in order."""

    @staticmethod
    @abc.abstractmethod
    def tile(output: tuple[int, int], step: int, stages: int) -> Tile:
        """The tile of an `output` (rows, columns) tile, a contraction `step` deep, at `stages`, eight
        warps."""

    @staticmethod
    @abc.abstractmethod
    def _dense_extents(ba) -> tuple[int, int]:
        """The depth and width a dense call's tiles divide."""

    @staticmethod
    @abc.abstractmethod
    def output(tile: Tile) -> tuple[int, int]:
        """The rows and columns of a program's output tile, the dot's M and N."""

    @staticmethod
    @abc.abstractmethod
    def contraction(tile: Tile) -> int:
        """The contraction one step of `tile` takes, the dot's K."""

    def staged_bytes(self, ba, config: Tile, target) -> int:
        # Pallas Triton's pipeline holds `stages - 1` depth steps where its MMA reads its operands from
        # registers, and `stages` where it reads them from shared memory as it runs (the device's family's
        # issuer, `mosaic.Issuer.operands_in_registers`: Hopper's wgmma, Blackwell's tcgen05), one step read
        # while the next loads; an A100 and an H100 refuse exactly those bytes, and a B200 refused 262,160
        # bytes for a dense fp32 tile counted at `stages - 1` (gate-2738-gemm-b200, row 4449). The
        # prologue's side tiles load in the same steps.
        # A GPU outside every template family (a Turing part) issues no MMA reading shared memory: its
        # operands are registers' too.
        dtype, issuer = ba.arguments['tokens'].dtype, generation.family(target.generation)
        held = int(issuer is None or bundles.named(issuer, jnp.bfloat16).operands_in_registers)
        step = self.loaded(config) * jnp.dtype(dtype).itemsize + _side_tile_bytes(
            ba.arguments['prologue_operands'], ba.arguments['hooks'].prologue_axes, self.extents(config))
        # A contraction of one step has no loop whose dead stages the output's layout conversion reuses:
        # it takes a 16-bit output tile beside them (an L4 refused 114688 bytes for a 64 x 128 tile's
        # 98304 at a depth of one step, either output dtype, GEN6-gpu-l4; at four steps it asked the
        # stages' alone, TSMEM-l4).
        contracted = self._contracted(ba)
        single = contracted is not None and contracted <= self.contraction(config)
        rows, columns = self.output(config)
        # A hooked call converts its hooks' tiles' layouts through shared memory beside the stages: an A100
        # refused 181504 bytes for a relu producer's tile counted at 147456 (ALIGNED-gpu-a100), within a
        # step's more.
        hooked = ba.arguments['hooks'] != NO_HOOKS
        return step * (config.stages - held + hooked) + (2 * rows * columns if single else 0)

    @staticmethod
    def _contracted(ba) -> int | None:
        """The contraction a call's every tile steps over, None where its groups' rows set it."""
        return None

    @staticmethod
    @abc.abstractmethod
    def extents(tile: Tile) -> dict:
        """The block each launch axis takes at `tile`."""


    @staticmethod
    @abc.abstractmethod
    def loaded(tile: Tile) -> int:
        """The elements of both operand tiles one depth step loads."""


def _product_sides(ba) -> tuple[int, int]:
    """A product's output: its tokens by its kernel's width."""
    kernel = ba.arguments['kernel']
    return ba.arguments['tokens'].shape[0], kernel.shape[1 if ba.arguments['trans_rhs'] else 2]


def _gradient_sides(ba) -> tuple[int, int]:
    """A kernel gradient's output: its tokens' width (its depth) by its cotangent's, the tokens the
    contraction."""
    return ba.arguments['tokens'].shape[1], ba.arguments['cotangent'].shape[1]


@dataclasses.dataclass(frozen=True)
class Gmm(_Triton):
    """Each group's rows of `tokens` `[rows, in]` times its matrix of
    `kernel` `[groups, in, out]`, or of its transpose with `trans_rhs`."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens',)

    def output_sides(self, ba) -> tuple[int, int]:
        return _product_sides(ba)

    @staticmethod
    def _dense_extents(ba) -> tuple[int, int]:
        """The depth and width of a dense call's product, which `block_k` and `block_n` tile."""
        kernel = ba.arguments['kernel']
        return (kernel.shape[2], kernel.shape[1]) if ba.arguments['trans_rhs'] else kernel.shape[1:]

    @staticmethod
    def _contracted(ba) -> int | None:
        return Gmm._dense_extents(ba)[0]

    @staticmethod
    def steps(ba) -> tuple[int, ...]:
        # The deepest first: a dense depth, read a step at a time, synchronizes its pipeline fewer times a
        # tile the deeper the step (at 128 x 256 on an H100, a step of 64 took 1.56 ms where 32 took 1.78,
        # GMM-h100-2).
        del ba
        return 128, 64, 32, 16

    @staticmethod
    def tile(output: tuple[int, int], step: int, stages: int) -> Tile:
        return Tile(output[0], output[1], step, 8, stages)

    @staticmethod
    def loaded(tile: Tile) -> int:
        return (tile.block_m + tile.block_n) * tile.block_k

    @staticmethod
    def output(tile: Tile) -> tuple[int, int]:
        return tile.block_m, tile.block_n

    @staticmethod
    def contraction(tile: Tile) -> int:
        return tile.block_k

    @staticmethod
    def extents(tile: Tile) -> dict:
        return {'rows': tile.block_m, 'depth': tile.block_k, 'columns': tile.block_n}


    def _fwd(self, tokens: jax.Array, kernel: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), epilogue_operands: tuple = (), paired: jax.Array | None = None,
             first: jax.Array | None = None, gather: jax.Array | None = None,
             scatter: jax.Array | None = None, *, out_dtype: str, trans_rhs: bool, interpret: bool,
             hooks: Hooks = NO_HOOKS, exact_sums: bool = False,
             dense: bool = False, config: Tile, return_residuals: bool = False):
        del paired, gather, scatter  # a second matrix and table rows are Mosaic GPU's alone (`tables`)
        # Imported at first use: the kernels need jax's Pallas Triton backend,
        # which jax 0.11.2 deprecates, and importing a model must not load it.
        if dense:
            # One group of every row: the plain tiled kernel, which finds no group and masks no row.
            from .templates.gemm import triton_dense
            options = hooks.kernel_options()
            outs = triton_dense.gmm(tokens, kernel[0], tile=config, transpose_rhs=trans_rhs,
                                    out_dtypes=tuple(jnp.dtype(d) for d in (options.pop('out_dtypes')
                                                                            or (out_dtype,))),
                                    interpret=interpret, prologue_operands=prologue_operands,
                                    epilogue_operands=epilogue_operands, exact_sums=exact_sums, **options)
            return (outs if hooks.out_dtypes else outs[0]), None
        from . import ragged_dot
        return ragged_dot.gmm(tokens, kernel, sizes, block_m=config.block_m, block_k=config.block_k,
                              block_n=config.block_n, num_warps=config.warps, num_stages=config.stages,
                              trans_rhs=trans_rhs, interpret=interpret, compute_dtype=tokens.dtype,
                              out_dtype=jnp.dtype(out_dtype), first=first,
                              prologue_operands=prologue_operands, epilogue_operands=epilogue_operands,
                              exact_sums=exact_sums, **hooks.kernel_options()), None


@dataclasses.dataclass(frozen=True)
class Tgmm(_Triton):
    """Each group's `tokens` rows transposed times its `cotangent` rows: the
    kernel gradient `[groups, in, out]`."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens', 'cotangent')

    def output_sides(self, ba) -> tuple[int, int]:
        return _gradient_sides(ba)

    @staticmethod
    def _dense_extents(ba) -> tuple[int, int]:
        """The depth and width of a dense call's gradient, which `block_k` and `block_n` tile."""
        return ba.arguments['tokens'].shape[1], ba.arguments['cotangent'].shape[1]

    @staticmethod
    def loaded(tile: Tile) -> int:
        return tile.block_m * (tile.block_k + tile.block_n)

    @staticmethod
    def steps(ba) -> tuple[int, ...]:
        # Its steps run over a group's rows, each group's padded to whole steps: a step of 64 rows keeps
        # that padding to half of 64 a group.
        del ba
        return 64, 32, 16

    @staticmethod
    def tile(output: tuple[int, int], step: int, stages: int) -> Tile:
        return Tile(step, output[1], output[0], 8, stages)

    @staticmethod
    def output(tile: Tile) -> tuple[int, int]:
        return tile.block_k, tile.block_n

    @staticmethod
    def contraction(tile: Tile) -> int:
        return tile.block_m

    @staticmethod
    def extents(tile: Tile) -> dict:
        return {'rows': tile.block_m, 'depth': tile.block_k}


    def _fwd(self, tokens: jax.Array, cotangent: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), gather: tuple | None = None, *, out_dtype: str, interpret: bool,
             hooks: Hooks = NO_HOOKS,
             exact_sums: bool = False, dense: bool = False, config: Tile, return_residuals: bool = False):
        del gather  # table rows are Mosaic GPU's alone (`tables`)
        if dense:
            # One group of every row: the plain tiled kernel (`Gmm._fwd`).
            from .templates.gemm import triton_dense
            return triton_dense.tgmm(tokens, cotangent, tile=config, out_dtype=jnp.dtype(out_dtype),
                                     interpret=interpret, prologue=hooks.before(),
                                     prologue_operands=prologue_operands, prologue_axes=hooks.prologue_axes,
                                     exact_sums=exact_sums)[None], None
        from . import ragged_dot
        return ragged_dot.tgmm(tokens, cotangent, sizes, block_m=config.block_m, block_k=config.block_k,
                               block_n=config.block_n, num_warps=config.warps, num_stages=config.stages,
                               interpret=interpret, compute_dtype=tokens.dtype,
                               out_dtype=jnp.dtype(out_dtype), prologue=hooks.before(),
                               prologue_operands=prologue_operands, prologue_axes=hooks.prologue_axes,
                               exact_sums=exact_sums), None


_PaddedArgument = tuple[str, "jax.Array | jax.ShapeDtypeStruct | tuple", "dict | tuple"]
"""An argument `_Mosaic._aligned` pads: its name, its value (or tuple of them) and its dims (`_grown`'s)."""


class _Mosaic(_GroupedLaunch[mosaic_gpu.Tiles]):
    """JAX's Pallas Mosaic GPU kernels (`templates.gemm.mosaic_gpu`) on their instruction family's bundle
    (Hopper's warpgroup MMA, Ampere's mma.sync), one 16-bit dtype, their one tile generated from the
    family's facts, the device's and the call's extents (`mosaic_gpu.derived`). Each runs one warpgroup a
    multiprocessor, or two consumers beside a producer (a cooperative tile), persistent over the output
    tiles in a planar snake, so the rows a band of `grid_block_n` column tiles shares and the matrix block
    the row tiles in flight share come from the L2."""
    name: ClassVar[str] = 'mosaic'
    config_cls: ClassVar[type[mosaic_gpu.Tiles]] = mosaic_gpu.Tiles
    GENERATES: ClassVar[bool] = True
    EXACT_SUMS: ClassVar[bool] = False
    """Whether the kernel adds each depth step's product apart (`exact_sums`); one that does not is not
    eligible where the call asks for it."""
    STAGED_ROWS: ClassVar[bool] = False
    """Whether a consumer prepares its share of each step's rows stage itself (the kernel gradient's
    prologue and group zeroing, `registers`)."""
    K_MAJOR: ClassVar[bool] = False
    """Whether its operands lie rows-contiguous, step-scaled (`MosaicStepScaledTgmm`): whole steps of one
    group, nothing staged to rewrite."""
    FP8: ClassVar[bool] = False
    """Whether the launch multiplies FP8 operands (`_fp8`): a forward product's alone."""
    STEPS: ClassVar[tuple[int, ...] | None] = None
    """The depths its stage takes where its stage is its own (a packed matrix's decode,
    `MosaicPackedGmm`); None: its family's (`mosaic.Family.depths`)."""

    def eligible(self, ba, target) -> bool:
        first, second = (value.dtype for name, value in ba.arguments.items()
                         if name in ('tokens', 'kernel', 'cotangent'))
        # A TPU's launch plans its VMEM against the scoped budget a measurement states
        # (`generation.shared_budget`).
        budgeted = not generation.tpu(target) or target.scoped_vmem is not None
        return (first == second and budgeted and bool(self._families(ba, target))
                and self._hooks_lower(ba, first, target)
                and (self.EXACT_SUMS or not ba.arguments.get('exact_sums'))
                and self.derived(ba, target) is not None)

    def registers(self, ba, config: mosaic_gpu.Tiles) -> int | None:
        # A consumer thread's live set (`resources.tile_registers`: the accumulator it carries, a step's
        # fragments, an output's copy), and beside the accumulator the larger of its phases, each done
        # before the next starts: the rows' stage a kernel gradient's consumer prepares itself, and the
        # epilogue's side values (`_epilogue_registers`); under exact sums the running sum beside each
        # step's own product. Each hook is priced by what it materializes (`_prologue_registers`,
        # `_epilogue_registers`); a run's compile steps a spilling tile down (`report.compiled`). The
        # compile gate tests the actual
        # launch and residency caps: ptxas's spill decisions are not monotone, so a least spill-free cap
        # cannot decide a launch.
        ba, config = self._ran(ba, config)
        dtype = ba.arguments['tokens'].dtype
        family = mosaic_gpu.family_of(config, dtype)
        # The tile's columns its accumulators hold: a pair's both where its products span it.
        columns = mosaic_gpu.tile_columns(config, family)
        block = (config.block_m, columns, config.block_k)
        if not family.budgets_registers or generation.tpu(generation.gpu_target()):
            # A TPU core states none, whatever tile a pin names: its values are Mosaic TPU's, in VMEM
            # (`mosaic.Issuer.value_scratch`).
            return None
        hooks = ba.arguments['hooks']
        stored = hooks.out_dtypes or (ba.arguments['out_dtype'],)
        plain = resources.tile_registers(family, config.warpgroups, block, dtype, stored=stored,
                                        rhs_transposed=self._matrix_transposed(ba))
        accumulator = -(-config.block_m * columns // config.warpgroups // resources.THREADS)
        # A kernel gradient's consumer prepares its depth half of each step's rows stage in registers (the
        # neighbouring groups' rows zeroed: `mosaic_gpu._staged_halves`), apart from the products, which
        # read shared memory on such a family; mma.sync's products hold that stage in registers already.
        # sm90's kernel gradient needs 24 more than the forward at 128 x 192 and 128 x 256 (148 and 181,
        # DECIDE1).
        # Its rewrite and its zeroing address the stage's tiles themselves: its depth half of the rows, a
        # rows atom past it, and the cotangent's atom it loads and stores (`_staged_halves`; 3117's census,
        # 8 registers at 256 x 64 exact normed).
        staged = rewritten = 0
        # A stage read as loaded (`unstaged`) holds none of it.
        if self.STAGED_ROWS and not family.operands_in_registers and not self.unstaged(ba):
            words = -(-config.block_k * config.block_m // config.warpgroups * jnp.dtype(dtype).itemsize
                      // 4 // resources.THREADS)
            atom = (config.block_k, swizzling.swizzle_columns(dtype, swizzling.SWIZZLE_BYTES))
            tiles = [((config.block_k, config.block_m // config.warpgroups), dtype)] + [(atom, dtype)] * 3
            rewritten = family.scalar_registers(tiles) - family.scalar_registers()
            staged = max(0, words - family.held_registers) + rewritten
        prologue = self._prologue_registers(ba, config)
        prologue += rewritten if prologue else 0
        # A dense call's last wave balanced over the programs merges a slice of another program's partial at a
        # time beside its own (`mosaic.Family.fixed_up`, `across.merge_spans`).
        merged = (min(accumulator, across.MERGED_REGISTERS)
                  if ba.arguments.get('dense') and mosaic_gpu.fixes_up(config, dtype) else 0)
        phases = [staged, prologue, merged, self._epilogue_registers(ba, config, stored)]
        # A paired product's second accumulator, carried beside the first (`mosaic_gpu.gmm`'s `paired`): in
        # registers its whole share, in tensor memory the epilogue chunk read beside the first's
        # (`Issuer.chunk_columns`).
        held_second = -(-config.block_m * family.chunk_columns(config.block_n) // config.warpgroups
                        // resources.THREADS)
        plain += held_second if ba.arguments.get('paired') is not None else 0
        if ba.arguments.get('exact_sums'):
            # The running sum, a second fp32 accumulator (`mosaic_gpu.tgmm`'s exact sums), is carried as a
            # loop value beside each step's product (`mosaic.Issuer.summed_apart`), awaited before its add:
            # the two accumulators and the walk's control, no step's descriptors beside them
            # (`resources.CONTROL_REGISTERS`), else the staged rows above them (FS-K01 model2: 144 at 64 a
            # thread over stages 2-5, 208 at 96). A family whose products read registers holds each step's
            # fragments beside both (mma.sync's, one step's product at a time: 201 at its 128 x 128 forward
            # tile, NSDBG-5).
            held = plain + (accumulator if family.operands_in_registers else 0)
            return max(2 * accumulator + resources.CONTROL_REGISTERS, held + max(phases))
        return plain + max(phases)

    def _prologue_registers(self, ba, config: mosaic_gpu.Tiles) -> int:
        """A prologue's transform of a consumer's share of the rows' stage: that share and each side tile of
        its shape widened to fp32 as the hook computes them, and its result in the stage's dtype (HOOKS1,
        sm90: a norm's +46..53 and a cast's +47 at a 32-word share, SwiGLU's gate tile +87..91). Vector
        sides are a share's broadcasts, held within it. The forward's products take the result from
        registers (`mosaic_gpu._gmm_step`), each step's held until its products are done, so beside it each
        step its family keeps in flight (`Family.in_flight`) still holds its own; the kernel gradient
        rewrites its stage (`STAGED_ROWS`) and holds none."""
        hooks = ba.arguments['hooks']
        if hooks.prologue is None:
            return 0
        tiles = sum(len(held_axes(axes)) > 1 for axes in hooks.prologue_axes)
        share = config.block_m * config.block_k // config.warpgroups
        stored = jnp.dtype(ba.arguments['tokens'].dtype).itemsize
        family = mosaic_gpu.family_of(config, ba.arguments['tokens'].dtype)
        in_flight = 0 if self.STAGED_ROWS else family.in_flight
        # A family whose products read registers casts the result into its left operand's fragments
        # (`MmaIssuer._left`) beside the result itself: a cast forward's least spill-free cap on sm80 was 136
        # at 64 x 128 x 64, its model 127 without the cast's copy (EM-least4).
        copies = 1 + in_flight + (family.operands_in_registers and not self.STAGED_ROWS)
        return -(-share * ((1 + tiles) * 4 + copies * stored) // (4 * resources.THREADS))

    @staticmethod
    def _epilogue_registers(ba, config: mosaic_gpu.Tiles, stored: tuple) -> int:
        """The epilogue's values beyond the mainloop's held set: a family that stages its stores through
        shared memory holds one staged chunk of the output converted (`mosaic.output_staging`) and the
        store's control (`resources.CONTROL_REGISTERS`); then its side vectors, each side tile of the
        accumulator's shape in fp32 (a residual's: +94 at a 96-word accumulator, HOOKS1) and each output but
        the last held in its stored dtype while the next is computed (a second output: +45..57 at 128
        words)."""
        hooks = ba.arguments['hooks']
        family = mosaic_gpu.family_of(config, ba.arguments['tokens'].dtype)
        share, columns = config.block_m // config.warpgroups, mosaic_gpu.tile_columns(config, family)
        staging = 0
        if not family.stores_through_registers:
            chunk, _ = mosaic.output_staging(columns, stored)
            widest = max(jnp.dtype(dtype).itemsize for dtype in stored)
            staging = -(-share * chunk * widest // (4 * resources.THREADS)) + resources.CONTROL_REGISTERS
        if hooks.epilogue is None:
            return max(0, staging - family.held_registers)
        # Column vectors beside the accumulator, in its column layout (`Family.column_registers`), and a
        # paired product's combination of its two accumulators, which holds a row of its results, one fp32
        # column vector's worth (its SwiGLU spilled 32 bytes at 128 x 192 at 232, PAIRED-9, and fit 224 at
        # 384 x 64, PAIRED-11): each with the walk's control alone (`resources.CONTROL_REGISTERS`), their
        # excess over the mainloop's held set.
        vectors = sum(family.column_registers(columns, side.dtype)
                      for side, axes in zip(ba.arguments.get('epilogue_operands', ()), hooks.epilogue_axes,
                                            strict=True) if held_axes(axes) == ('columns',))
        vectors += family.column_registers(columns, jnp.float32) if hooks.accumulators > 1 else 0
        # A side tile and an output held beside the next are an epilogue chunk's (`Issuer.chunk_columns`: the
        # whole accumulator's where its registers hold it, tensor memory's chunk on tcgen05).
        accumulator = -(-config.block_m * family.chunk_columns(columns) // config.warpgroups
                        // resources.THREADS)
        tiles = sum(len(held_axes(axes)) > 1 for axes in hooks.epilogue_axes) * accumulator
        held = sum(-(-accumulator * jnp.dtype(dtype).itemsize // 4)
                   for dtype in (hooks.out_dtypes or ())[:-1])
        # An exponential the epilogue evaluates over the chunk (`dew.algebra.cost.EXPONENTIALS`) holds the
        # values its exact lowering keeps live at its peak (`resources.EXACT_EXPONENTIAL_LIVE`), the kernels
        # compiling without approximate math: a paired SwiGLU's sigmoid over its 128 x 128 epilogue spilled
        # 96 bytes beside a model without them (FS-SWIGLU-BEFORE2).
        exponential = resources.EXACT_EXPONENTIAL_LIVE * accumulator if _evaluates_exponential(ba) else 0
        stage = max(staging, resources.CONTROL_REGISTERS) + vectors
        return max(0, stage - family.held_registers) + tiles + held + exponential

    def register_budget(self, config: mosaic_gpu.Tiles) -> int:
        # Its family's budget a consumer, whatever the operands' dtype (`mosaic.Family.registers`); none on a
        # core that budgets no registers.
        family = mosaic_gpu.family_of(config, jnp.bfloat16)
        return family.registers(config.warpgroups) if family.budgets_registers else 0

    def _granular(self, ba, config: mosaic_gpu.Tiles, family: mosaic.Family) -> bool:
        """Whether `config` is in `family`'s granularity (`candidate`): at most its consumers, at most the
        programs its registers hold, a ring of two stages at least, whole roles of rows for each consumer,
        whole column units, one of its depths, and each streamed block's rows in its consumers' equal
        shares (`Family.shares_rows`)."""
        dtype = ba.arguments['tokens'].dtype
        return not (config.warpgroups > min(family.consumers, mosaic_gpu.CONSUMERS)
                    or config.programs > family.programs(config.warpgroups) or config.stages < 2
                    or config.block_m % (family.role_rows * config.warpgroups)
                    or config.block_n % family.column_unit(dtype)
                    or config.block_k not in (self.STEPS or family.depths(dtype))
                    or not all(family.shares_rows(rows, config.warpgroups)
                               for rows in self._streamed_rows(ba, config)))

    def _streamed_rows(self, ba, config: mosaic_gpu.Tiles) -> tuple[int, ...]:
        """The rows of each block a step streams, each its leading axis as it lies (its consumers' shares
        of them, `Family.shares_rows`): the rows' tile's (`block_m`, or `block_k` read transposed) and the
        matrix's (`block_k`, or `block_n` read transposed)."""
        return config.block_m, config.block_k, *((config.block_n,) if self._matrix_transposed(ba) else ())

    def _matrix_transposed(self, ba) -> bool:
        """Whether the matrix's blocks lie transposed, `[width, depth]` (`trans_rhs`), as its stream reads
        them."""
        return bool(ba.arguments.get('trans_rhs'))

    def _fp8(self, ba, dtype, family: mosaic.Family) -> bool:
        """Whether an FP8 call (`product.EIGHT`) runs here: a forward product on warpgroup MMA, its matrix
        K-major (`trans_rhs`, as Hopper's FP8 tensor cores read both operands; an N-major FP8 matrix is
        refused, row 3620), with no hook, its running sum promoted every step (`exact_sums`)."""
        return (jnp.dtype(dtype) in EIGHT and self.FP8 and family.takes_fp8
                and ba.arguments['trans_rhs']
                and ba.arguments['hooks'] == NO_HOOKS and ba.arguments.get('paired') is None
                and bool(ba.arguments.get('exact_sums')))

    def _families(self, ba, target) -> tuple[mosaic.Family, ...]:
        """The bundles the launch may run on `target`, in preference order: the most capable issuer's whose
        bundle takes the call (`bundles.groups`, its kernels stating their tiles' accumulators,
        `mosaic_gpu.gmm`'s and `tgmm`'s; `_takes`), so a call tcgen05's bundle refuses runs on mma.sync's
        (row 4727), a TPU core's its MXU's (`mosaic_tpu.MXU`); none where no issuer's does (a CPU interprets
        a GPU's or a TPU's only where `generation.assumed` names it)."""
        found = bundles.groups(target.generation, ba.arguments['tokens'].dtype, stated=True,
                               tpu=generation.tpu(target))
        return next((group for group in found if self._takes(ba, group[0])), ())

    def _takes(self, ba, family: mosaic.Family) -> bool:
        """Whether `family`'s bundle runs the call: 16-bit operands, or FP8 ones it takes (`_fp8`); gathered
        rows where its producer gathers them beside the matrix's whole loads (`Transport.gathers_rows`,
        `gathers_alone`); a prologue over a side tile where its kernels lower one (`takes_prologue_tiles`); a
        paired product where it takes a pair (`takes_pairs`); a step that prepares its operands (a prologue's
        rewrite, a consumer's own share of the rows' stage, an exactly summed step's product
        apart, but where its persistent walk sums the windows itself, `sums_in_persist`, or rewrites the
        stage in a role of its own, `rewrites_in_persist`) where its consumers issue their products."""
        dtype = ba.arguments['tokens'].dtype
        summed = bool(ba.arguments.get('exact_sums'))
        prepared = (ba.arguments['hooks'].prologue is not None
                    or (self.STAGED_ROWS and not self.unstaged(ba) and not self.rewritten(ba, family))
                    or (summed and not family.sums_in_persist))
        # Gathered rows (`gmm_call`'s `gather`) are a ring family's producer's gather, no prologue rewriting
        # them (`mosaic_gpu._gather_refused`).
        gathered = ba.arguments.get('gather') is not None
        tabled = gathered or ba.arguments.get('scatter') is not None
        multiplied = family.operand_dtypes is None or jnp.dtype(dtype) in family.operand_dtypes
        return (((jnp.dtype(dtype).itemsize == 2 and multiplied) or self._fp8(ba, dtype, family))
                and (not tabled or (family.gathers_rows and family.gathers_alone))
                and (not gathered or ba.arguments['hooks'].prologue is None)
                and (family.issues_in_consumers or not prepared)
                and (family.takes_prologue_tiles or not _prologue_tiled(ba))
                and (family.takes_pairs or ba.arguments.get('paired') is None))

    @abc.abstractmethod
    def _generated_extents(self, ba) -> tuple[int, int | None, int]:
        """A group's output rows (`rows`, on average), the depth the steps run over (None: a group's rows,
        padded, the output's rows then whole, not a group's) and the output's width, as
        `mosaic_gpu.derived` reads them."""

    def derived(self, ba, target) -> mosaic_gpu.Tiles | None:
        return self._derived(ba, target)

    def _derived(self, ba, target, admits=None) -> mosaic_gpu.Tiles | None:
        """The derivation's tile of `ba` on `target` (`mosaic_gpu.derived`), each candidate also one `admits`
        takes where given (a swapped call's own rule, `MosaicGmm._left_fits`), on the call as the launch runs
        it (`_aligned`)."""
        ba = self._aligned(ba)
        layouts = self._families(ba, target)
        if not layouts:
            return None
        rows, depth, width = self._generated_extents(ba)
        return mosaic_gpu.derived(layouts, rows=rows, depth=depth, width=width,
                                  dtype=ba.arguments['tokens'].dtype,
                                  multiprocessors=_multiprocessors(target), target=target,
                                  candidate=lambda tile: (self.candidate(ba, tile, target)
                                                          and (admits is None or admits(tile))),
                                  launch_of=lambda tile: self.launch_of(ba, tile), grouped=depth is not None,
                                  depths=self.STEPS, ragged=self._ragged(ba, layouts[0]),
                                  units=self._vector_units(ba, layouts[0]))

    def _vector_units(self, ba, family: mosaic.Family) -> tuple[int, int, int]:
        """The units a tile's rows, columns and step take from the call's side vectors (`candidate`'s, each
        block along a vector's axis whole `mosaic.Transport.vector_unit`s), which the derivation shapes its
        tiles in (`mosaic_gpu.derived`'s `units`)."""
        by_axis = self.BLOCKS
        units = {'block_m': 1, 'block_n': 1, 'block_k': 1}
        hooks = ba.arguments['hooks']
        sides = (('prologue_operands', hooks.prologue_axes), ('epilogue_operands', hooks.epilogue_axes))
        for name, axes_of in sides:
            for side, side_axes in zip(ba.arguments.get(name, ()), axes_of, strict=True):
                if len(held_axes(side_axes)) == 1 and held_axes(side_axes)[0] in by_axis:
                    block = by_axis[held_axes(side_axes)[0]]
                    units[block] = math.lcm(units[block], family.vector_unit(side.dtype))
        return units['block_m'], units['block_n'], units['block_k']

    def _ragged(self, ba, family: mosaic.Family) -> bool:
        """Whether the launch's tiles may leave its output's rows a partial last tile (`mosaic_gpu.derived`'s
        `ragged`): none but a kernel gradient's (`MosaicTgmm`)."""
        del ba, family
        return False

    def neighbours(self, ba, config: mosaic_gpu.Tiles) -> tuple[mosaic_gpu.Tiles, ...]:
        # The derived tile with one of its choices moved a step either way: a stage more or fewer, the
        # band halved or doubled, the columns or the rows the next block either side (a column unit, a
        # role of rows).
        family = mosaic_gpu.family_of(config, ba.arguments['tokens'].dtype)
        unit = family.column_unit(ba.arguments['tokens'].dtype)
        role = family.role_rows * config.warpgroups
        band = config.grid_block_n
        moves = [{'stages': config.stages - 1}, {'stages': config.stages + 1},
                 {'grid_block_n': max(1, band // 2)}, {'grid_block_n': band * 2},
                 {'block_n': config.block_n - unit}, {'block_n': config.block_n + unit},
                 {'block_m': config.block_m - role}, {'block_m': config.block_m + role}]
        found = []
        for move in moves:
            if min(move.values()) <= 0:
                continue
            try:
                moved = dataclasses.replace(config, **move)
            except ValueError:
                continue  # a shape no Mosaic tile takes (`Tiles.__post_init__`)
            if moved != config:
                found.append(moved)
        return tuple(found)

    def legal(self, ba, config: mosaic_gpu.Tiles, target) -> bool:
        """Whether `config` runs the call on `target` (`Template.legal`): a candidate (`candidate`) whose
        ring of `config.stages` and registers the planner holds at its programs (`resources.plan`, as the
        derivation sizes them). A cluster's legality is each multiprocessor's: its staged bytes and
        registers are a tile's alone. A tile is judged as the call it runs (`_ran`)."""
        ba, config = self._ran(ba, config)
        if not self.candidate(ba, config, target):
            return False
        family = mosaic_gpu.family_of(config, ba.arguments['tokens'].dtype)
        launch = dataclasses.replace(self.launch_of(ba, config), floor=config.stages, most=config.stages)
        return resources.plan([launch], family, target, programs=config.programs) is not None

    def launch_of(self, ba, config: mosaic_gpu.Tiles) -> resources.Launch:
        """The launch the planner sizes at `config`'s shape (`resources.plan`): its allocations at a ring
        depth (`_allocations`), its consumers, its registers a consumer thread (`registers`), and at most
        the stages its program's share of the GPU's block (`generation.shared_budget`) holds of a step's
        two operand blocks at an element a byte and its contraction has steps, at least two: the call's a
        launch at `config` runs (`_ran`), the launch naming `config`."""
        launched = config
        ba, config = self._ran(ba, config)
        family = mosaic_gpu.family_of(config, ba.arguments['tokens'].dtype)
        depth = self._generated_extents(ba)[1]
        # An exactly summed walk's steps are a traced count (`mosaic_gpu._gmm_scheduled`'s `traced`): its
        # pipeline allocates its whole ring, however few steps the call has.
        steps = None if depth is None or ba.arguments.get('exact_sums') else -(-depth // config.block_k)

        def scope(stages: int) -> resources.Scope:
            staged = dataclasses.replace(config, stages=stages)
            return self._allocations(ba, staged, family).scope(family, staged, steps,
                                                               rewritten=self.rewritten(ba, family))

        shared = resources.program_shared(generation.shared_budget(generation.gpu_target()), config.programs)
        most = shared // ((config.block_m + config.block_n) * config.block_k)
        if steps is not None:
            # A ring deeper than its contraction's steps holds no more of them.
            most = min(most, max(2, steps))
        # A family fixing its rings' slots streams them whatever the stages (`mosaic.Transport.ring_slots`).
        most = family.ring_slots or most
        # A family rewriting the staged rows in a role of its own runs it beside the consumers
        # (`mosaic_gpu.tgmm`'s `roles`).
        roles = config.warpgroups + self.rewritten(ba, family)
        return resources.Launch(scope, roles, self.registers(ba, config), most=most, tile=launched)

    def rewritten(self, ba, family: mosaic.Family) -> bool:
        """Whether the launch's staged rows are rewritten by `family`'s own role (`rewrites_in_persist`): its
        own loads' rows, never rows its tables gather (`mosaic_gpu.tgmm`'s `gather`, which a consumer's
        cp.async gather stages: the fed rings'), nor rows read as loaded (`unstaged`)."""
        tabled = ba.arguments.get('gather') is not None or ba.arguments.get('scatter') is not None
        return (self.STAGED_ROWS and not self.K_MAJOR and family.rewrites_in_persist and not tabled
                and not self.unstaged(ba))

    def unstaged(self, ba) -> bool:
        """Whether a kernel gradient's rows stage is read as loaded (`mosaic_gpu.rows_unstaged`): a dense call
        of no prologue or table, laid rows-major, which no consumer or role prepares."""
        tabled = ba.arguments.get('gather') is not None or ba.arguments.get('scatter') is not None
        return (self.STAGED_ROWS and not self.K_MAJOR
                and mosaic_gpu.rows_unstaged(dense=bool(ba.arguments.get('dense')),
                                             prologue=ba.arguments['hooks'].prologue, tabled=tabled))

    def candidate(self, ba, config: mosaic_gpu.Tiles, target) -> bool:
        """Whether `config` is a tile of the call on `target` apart from its resources: of the launch's
        family there (a layout of it, `bundles.layouts`), in its granularity (whole roles of rows for each
        consumer, exactly the family's accumulator rows where it fixes them, whole column units, one of
        its depths, at most its consumers, at most the programs its registers hold, a pair's tile on a family
        whose products span one, `mosaic_gpu.pair_takes`), its accumulators held in the tensor memory a
        family keeps them in (`mosaic.Issuer.tensor_columns`), with a ring of two
        stages at least (a walk keeping one step's products in flight holds its slot while the next step's
        loads fill another), its blocks dividing the call's extents, its cluster at most the family's
        (`mosaic.Family.cluster`), and a one-dimensional side operand's block, copied untiled, at most one
        copy's extent along its axis (`mosaic.Family.copy_extent`); none a compile of this process found
        spilling (`report.refused`), so the derivation takes the next tile by its own rule."""
        family = next((bundle for bundle in self._families(ba, target) if bundle.name == config.family), None)
        if (family is None or config.cluster > family.cluster or not mosaic_gpu.pair_takes(config, family)
                or not self._divides(ba, config) or report.refused(self, ba, config)):
            return False
        # A family fixing its accumulator's rows takes a tile of exactly them a consumer, and one holding its
        # accumulators in tensor memory a tile whose accumulators it holds (a paired product's two).
        accumulators = 1 if ba.arguments.get('paired') is None else 2
        if (family.accumulator_rows not in (None, config.block_m // config.warpgroups)
                or family.tensor_columns(mosaic_gpu.tile_columns(config, family), accumulators,
                                         bool(ba.arguments.get('exact_sums'))) > (target.tmem_columns or 0)):
            return False
        if not self._granular(ba, config, family):
            return False
        hooks = ba.arguments['hooks']
        sides = (('prologue_operands', hooks.prologue_axes), ('epilogue_operands', hooks.epilogue_axes))
        vectors = {(axis, jnp.dtype(side.dtype)) for name, axes_of in sides
                   for side, side_axes in zip(ba.arguments.get(name, ()), axes_of, strict=True)
                   if len(held_axes(side_axes)) == 1 for axis in held_axes(side_axes)}
        # A vector's block along its axis is at most one copy's and whole units of its family's
        # (`mosaic.Transport.vector_unit`).
        extent = family.copy_extent
        return all((extent is None or self.extents(config)[axis] <= extent)
                   and self.extents(config)[axis] % family.vector_unit(dtype) == 0
                   for axis, dtype in vectors)

    def _ran(self, ba, config: mosaic_gpu.Tiles) -> tuple:
        """The call and tile a launch at `config` runs, which its legality, registers and launch read: the
        call as it runs it (`_aligned`) by default."""
        return self._aligned(ba), config

    def _aligned(self, ba):
        """The call as the launch runs it (`_fwd`'s `_padded`): its depth and width each padded with zeros to
        whole units of its family's tiles where they are not (its depth to the least step,
        `Family.depths`, or a tile's rows where the depth is a tile's; its width to whole column units,
        `Family.column_unit`), the output sliced back; the call itself where they are. A width of 100
        columns, which no tile divides, so runs on 128, the template's where Triton's masked tiles ran it.
        The padding stands where the loads cannot bound a partial tile: Mosaic GPU's TMA view of a partial
        dimension is refused ("dimension 1 must have size smaller or a multiple of its tiling 64, but got
        100", a022d3abb83), which the fork's row 5299 takes; its cost is the padded operands' copy."""
        layouts = self._families(ba, generation.gpu_target())
        if not layouts:
            return ba
        family, dtype = layouts[0], ba.arguments['tokens'].dtype
        # The width in whole column units and whole units of the call's column vectors (`_vector_units`: an
        # mma.sync bias's 64, where its column unit is 32; a 200-wide input gradient padded to 224 took no
        # tile, EM-unal2).
        width_unit = math.lcm(family.column_unit(dtype), self._vector_units(ba, family)[1])
        units = {'depth': self._depth_unit(ba, family, dtype), 'width': width_unit}
        extents = self._padded_extents(ba)
        grown = {axis: -(-extent // units[axis]) * units[axis] for axis, extent in extents.items()}
        if grown == extents:
            return ba
        padded = {name: _grown(value, dims, grown) for name, value, dims in self._padded_dims(ba)}
        return type(ba)(ba.op, {**ba.arguments, **padded})

    @abc.abstractmethod
    def _depth_unit(self, ba, family: mosaic.Family, dtype) -> int:
        """The unit the call's depth takes (`_aligned`)."""

    @abc.abstractmethod
    def _padded_extents(self, ba) -> dict[str, int]:
        """The call's depth and width, as `_aligned` pads them."""

    @abc.abstractmethod
    def _padded_dims(self, ba) -> list[_PaddedArgument]:
        """Each argument `_aligned` pads, its value, and its dimensions by padded axis ('depth', 'width')."""

    @abc.abstractmethod
    def _divides(self, ba, config: mosaic_gpu.Tiles) -> bool:
        """Whether `config`'s blocks divide the call's extents as the kernel requires."""

    @staticmethod
    def _hooks_lower(ba, dtype, target) -> bool:
        """Whether the target's Mosaic lowering takes the launch's hooks (`_hooks_lower`): a TPU core's by its
        TensorCore rules; a GPU's with warpgroup semantics, whose rules a hook's operations need. A side
        operand the launch squeezes comes back broadcast across its tile (`_unsqueezed`)."""
        if generation.tpu(target):
            return _hooks_lower(ba, dtype, mosaic_tpu_lowering.lowering_rules[tpu_core.CoreType.TC])
        return _hooks_lower(ba, dtype, mosaic_gpu_lowering.mosaic_lowering_rules[plgpu_core.WGxWG_SEMANTICS])

    def staged_bytes(self, ba, config: mosaic_gpu.Tiles, target) -> int:
        # The kernel's own allocations at `config` (`mosaic_gpu.Allocations`), counted as Mosaic's resource
        # estimate counts them (`resources.shared_bytes`): its streams' stages, its side operands' staging,
        # its outputs' staging as its family stages them, and every barrier (at a block's edge, a Hopper
        # tile's barriers alone took it 128 bytes past). A scoped walk and the output staged after it
        # reuse shared memory; persistent rings and output staging remain live together.
        del target
        family = mosaic_gpu.family_of(config, ba.arguments['tokens'].dtype)
        return resources.shared_bytes(family, self.launch_of(ba, config).scope(config.stages))

    @abc.abstractmethod
    def _allocations(self, ba, config: mosaic_gpu.Tiles, family: mosaic.Family) -> mosaic_gpu.Allocations:
        """The kernel's allocations at `config` for the call `ba`."""

    BLOCKS: ClassVar[dict[str, str]]
    """Each launch axis's block field of a tile (`extents`, `_vector_units`)."""

    @classmethod
    def extents(cls, tile: mosaic_gpu.Tiles) -> dict:
        """The block each launch axis takes at `tile`."""
        return {axis: getattr(tile, block) for axis, block in cls.BLOCKS.items()}


    def program_shared(self, config: mosaic_gpu.Tiles, target) -> int:
        # The programs a multiprocessor runs at once share its shared memory.
        return resources.program_shared(super().program_shared(config, target), config.programs)

    def _programs(self, config: mosaic_gpu.Tiles) -> int:
        """The programs the persistent kernel's schedule spreads its tiles over: the tile's programs on each
        multiprocessor (`Tiles.programs`)."""
        return _multiprocessors(generation.gpu_target()) * config.programs

    def _launched(self, config: mosaic_gpu.Tiles, interpret: bool) -> int:
        """The programs the persistent kernel launches: its schedule's (`_programs`); the interpreter runs
        one cluster (jax/_src/pallas/mosaic_gpu/interpret), one program alone or a pair's two
        (`Tiles.cluster`)."""
        return config.cluster if interpret else self._programs(config)


SWAPPED_ROWS = 128
"""The most rows of the matrix a swapped decode tile takes (`MosaicGmm.derived`): its matrix's stream is the
walk, which more, shorter tiles spread over more programs, where the derivation's fewest loads a FLOP took
the widest rows its registers hold: an A100's 2048 x 2048 decode product with the matrix lying `[width,
depth]` derived 384 x 32 x 32 at three stages, 2.1x the time of the other layout's 128 x 32 x 64 at four at
the same 96 programs, 4.2x XLA's (W50-a100-1-M-NS-19-a100)."""


@dataclasses.dataclass(frozen=True)
class MosaicGmm(_Mosaic):
    """`Gmm` through the template's mainloop (`mosaic_gpu.gmm`) on a GPU family's bundle or a TPU core's,
    exact sums too: an FP8 product's running sum promoted into fp32 every step (`product.EIGHT`)."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens',)
    EXACT_SUMS: ClassVar[bool] = True
    FP8: ClassVar[bool] = True
    STEP_SCALED: ClassVar[bool] = False
    """Whether its kernel streams a step's rows' scales beside its ring (`MosaicStepScaledGmm`)."""
    BLOCK_SCALED: ClassVar[bool] = False
    """Whether its kernel streams an MX product's block scales beside its rings (`MosaicMxGmm`)."""

    def output_sides(self, ba) -> tuple[int, int]:
        return _product_sides(ba)

    def _allocations(self, ba, config: mosaic_gpu.Tiles, family: mosaic.Family) -> mosaic_gpu.Allocations:
        hooks = ba.arguments['hooks']
        return mosaic_gpu.gmm_allocations(
            config, family, dtype=ba.arguments['tokens'].dtype, rhs_dtype=ba.arguments['kernel'].dtype,
            transpose_rhs=ba.arguments['trans_rhs'], prologue_axes=hooks.prologue_axes,
            prologue_dtypes=tuple(side.dtype for side in ba.arguments['prologue_operands']),
            epilogue_axes=hooks.epilogue_axes,
            epilogue_dtypes=tuple(side.dtype for side in ba.arguments.get('epilogue_operands', ())),
            dtypes=hooks.out_dtypes or (ba.arguments['out_dtype'],),
            paired=ba.arguments.get('paired') is not None, step_scaled=self.STEP_SCALED,
            block_scaled=self.BLOCK_SCALED)

    def _generated_extents(self, ba) -> tuple[int, int | None, int]:
        # A group's rows on average over the call's groups, which a stack read from its `first` matrix
        # holds fewer of than matrices.
        tokens, kernel = ba.arguments['tokens'], ba.arguments['kernel']
        depth, width = (kernel.shape[2], kernel.shape[1]) if ba.arguments['trans_rhs'] else kernel.shape[1:]
        return -(-tokens.shape[0] // max(ba.arguments['sizes'].shape[0], 1)), depth, width

    def _divides(self, ba, config: mosaic_gpu.Tiles) -> bool:
        kernel = ba.arguments['kernel']
        depth, width = (kernel.shape[2], kernel.shape[1]) if ba.arguments['trans_rhs'] else kernel.shape[1:]
        columns = mosaic_gpu.tile_columns(config, mosaic_gpu.family_of(config, ba.arguments['tokens'].dtype))
        return depth % config.block_k == 0 and width % columns == 0

    def _depth_unit(self, ba, family: mosaic.Family, dtype) -> int:
        # The least step its stage takes, in whole units of a prologue's depth vectors (`candidate`'s, the
        # depth mask's among them, `_aligned`): sm89's mma.sync pads 200 to 224 by its least step alone, which
        # no step of its 64-element vector unit divides (EM-unal).
        step = min(self.STEPS or family.depths(dtype))
        if ba.arguments['hooks'].prologue is None:
            return step
        return math.lcm(step, family.vector_unit(jnp.float32), self._vector_units(ba, family)[2])

    def _aligned(self, ba):
        # A prologue never sees the depth's padding: its result zeroed past the call's depth by a side vector
        # over the depth (`mosaic_gpu.depth_masked`), bound with the padded call so its planner, its register
        # model and its kernel stage the vector alike.
        aligned = super()._aligned(ba)
        hooks = aligned.arguments['hooks']
        depth = self._padded_extents(ba).get('depth')
        padded = self._padded_extents(aligned).get('depth')
        if hooks.prologue is None or depth is None or depth == padded:
            return aligned
        abstract = isinstance(ba.arguments['tokens'], jax.ShapeDtypeStruct)
        valid = (jax.ShapeDtypeStruct((padded,), jnp.float32) if abstract
                 else (jnp.arange(padded) < depth).astype(jnp.float32))
        masked = dataclasses.replace(
            hooks, prologue=functools.partial(mosaic_gpu.depth_masked, hooks.prologue),
            prologue_axes=(*hooks.prologue_axes, ('depth',)),
            prologue_dtypes=(*hooks.prologue_dtypes, 'float32') if hooks.prologue_dtypes else (),
            prologue_broadcast=(*hooks.prologue_broadcast, ()) if hooks.prologue_broadcast else ())
        return type(aligned)(aligned.op, {**aligned.arguments, 'hooks': masked, 'prologue_operands': (
            *aligned.arguments['prologue_operands'], valid)})

    def _padded_extents(self, ba) -> dict[str, int]:
        # A packed, step-scaled or block-scaled matrix's layout is its own: such a call is not padded.
        if self.STEPS is not None or self.STEP_SCALED or self.BLOCK_SCALED:
            return {}
        kernel = ba.arguments['kernel']
        depth, width = (kernel.shape[2], kernel.shape[1]) if ba.arguments['trans_rhs'] else kernel.shape[1:]
        return {'depth': depth, 'width': width}

    def _padded_dims(self, ba) -> list[_PaddedArgument]:
        matrix = {1: 'width', 2: 'depth'} if ba.arguments['trans_rhs'] else {1: 'depth', 2: 'width'}
        hooks = ba.arguments['hooks']
        dims = [('tokens', ba.arguments['tokens'], {1: 'depth'}), ('kernel', ba.arguments['kernel'], matrix),
                ('prologue_operands', ba.arguments['prologue_operands'],
                 _side_dims(ba.arguments['prologue_operands'], hooks.prologue_axes, {'depth': 'depth'})),
                ('epilogue_operands', ba.arguments.get('epilogue_operands', ()),
                 _side_dims(ba.arguments.get('epilogue_operands', ()), hooks.epilogue_axes,
                            {'columns': 'width'}))]
        if ba.arguments.get('paired') is not None:
            dims.append(('paired', ba.arguments['paired'], matrix))
        return dims

    BLOCKS: ClassVar[dict[str, str]] = {'rows': 'block_m', 'depth': 'block_k', 'columns': 'block_n'}


    def derived(self, ba, target) -> mosaic_gpu.Tiles | None:
        # A dense product of fewer rows than a role's (a decode step's) runs swapped: its matrix takes the
        # MMA's rows and its rows the columns, whose unit is narrower, so its tile multiplies no padding rows
        # (CUTLASS's and cuBLAS's swap of A and B; megabake's 4 x 2048 x 2048 product went from 10.0 to
        # 4.7 us swapped, cuBLAS's time, row 1762). It is derived as the swapped call's own tile.
        swapped = self._swapped(ba, target)
        if swapped is None:
            return self._decode_split(ba, super().derived(ba, target), target)
        tile = self._derived(swapped, target, admits=lambda tile: (self._left_fits(ba, tile)
                                                                    and tile.block_m <= SWAPPED_ROWS))
        if tile is None:
            return None
        # Its few output tiles leave a wave's programs idle: its depth splits where the busiest program then
        # walks shorter, its fp32 partials' passes and their sum's launch counted (`mosaic_gpu.split_count`,
        # the kernel gradient's rule; rows 1762 and 483), every run in the one launch (`mosaic_gpu.gmm`).
        weight, padded = swapped.arguments['tokens'], swapped.arguments['kernel']
        splits = 1 if ba.arguments['exact_sums'] else mosaic_gpu.split_count(
            tile, groups=1, rows=weight.shape[1], depth=weight.shape[0], width=padded.shape[1],
            itemsizes=(weight.dtype.itemsize, padded.dtype.itemsize),
            multiprocessors=_multiprocessors(target), step_seconds=mosaic_gpu.step_seconds(tile, target),
            bandwidth=target.memory_rate, launch_seconds=target.launch_seconds)
        swapped_tile = dataclasses.replace(tile, swapped=True)
        split = dataclasses.replace(swapped_tile, splits=splits)
        return split if splits > 1 and self.legal(ba, split, target) else swapped_tile

    def _decode_split(self, ba, tile: mosaic_gpu.Tiles | None, target) -> mosaic_gpu.Tiles | None:
        """`tile`, a grouped call's of decode-sized groups (each group's rows within one row tile: a MoE
        decode step's experts) its depth split where the busiest program then walks shorter
        (`mosaic_gpu.split_count` over each active expert's one row tile, its matrix's stream the walk): its
        one wave of tiles streamed each expert's matrix at a program's share of the memory rate, 1.28-1.46x
        Triton's time at the same DRAM bytes, 48% of the peak rate against Triton's 80% (FS-3622-NCU1,
        FS-3622-WAVES1, row 3622). None, a dense call, exact sums (`legal`), hooks or a pair keep it
        unsplit."""
        hooks = ba.arguments['hooks']
        if (tile is None or ba.arguments['dense'] or ba.arguments['exact_sums']
                or ba.arguments.get('paired') is not None or hooks.prologue is not None
                or hooks.epilogue is not None):
            return tile
        tokens, kernel = ba.arguments['tokens'], ba.arguments['kernel']
        groups = ba.arguments['sizes'].shape[0]
        if tokens.shape[0] > groups * tile.block_m:
            return tile
        _, depth, width = self._generated_extents(ba)
        assert depth is not None, "a product's steps run over its depth"
        # A decode step's rows bound its active experts, each one row tile, and its partials hold its rows.
        active = min(groups, tokens.shape[0])
        splits = mosaic_gpu.split_count(
            tile, groups=active, rows=depth * active, depth=tile.block_m, width=width,
            partial_rows=-(-tokens.shape[0] // mosaic_gpu.TMA_ROWS) * mosaic_gpu.TMA_ROWS,
            itemsizes=(tokens.dtype.itemsize, kernel.dtype.itemsize),
            multiprocessors=_multiprocessors(target), step_seconds=mosaic_gpu.step_seconds(tile, target),
            bandwidth=target.memory_rate, launch_seconds=target.launch_seconds)
        split = dataclasses.replace(tile, splits=splits)
        return split if splits > 1 and self.legal(ba, split, target) else tile

    def _left_fits(self, ba, config: mosaic_gpu.Tiles) -> bool:
        """Whether a swapped tile reads `ba`'s matrix as its products' left operand: a matrix lying `[depth,
        width]` is read transposed, whose stage casts to the products' layout in whole row atoms on a family
        reading registers (`Family.transposed_rows`, as the kernel gradient's transposed rows)."""
        family = mosaic_gpu.family_of(config, ba.arguments['tokens'].dtype)
        return (ba.arguments['trans_rhs'] or not family.operands_in_registers
                or config.block_k % family.transposed_rows == 0)

    def _swapped(self, ba, target):
        """The dense product `ba` as its swapped tile runs it: its matrix `[width, depth]` the rows, however
        it lies, and its rows a matrix of them, padded to whole column units of every family the target
        runs, read transposed (`mosaic_gpu.gmm`'s `transpose_lhs` where the matrix lies `[depth, width]`).
        None where it runs as it is: a call with hooks, a paired, FP8 or grouped one, one whose rows fill
        a role (`Family.role_rows`), or one on a family whose products read shared memory themselves (not
        `operands_in_registers`, warpgroup MMA's: its swapped tile ran 1.606x the unswapped one's time on an
        H100, geomean of 24 decode rows, NS-SWAP2-h100, where mma.sync's ran 0.48-0.63x on an A100's narrow
        outputs, NS-SWAP-a100)."""
        families = self._families(ba, target)
        tokens, kernel = ba.arguments['tokens'], ba.arguments['kernel']
        if (not families or not all(family.operands_in_registers for family in families)
                or not ba.arguments.get('dense') or ba.arguments['hooks'] != NO_HOOKS
                or ba.arguments.get('paired') is not None or jnp.dtype(tokens.dtype).itemsize != 2
                or tokens.shape[0] >= min(family.role_rows for family in families)):
            return None
        depth, width = (kernel.shape[2], kernel.shape[1]) if ba.arguments['trans_rhs'] else kernel.shape[1:]
        unit = max(family.column_unit(tokens.dtype) for family in families)
        abstract = functools.partial(jax.ShapeDtypeStruct, dtype=tokens.dtype)
        options = {name: value for name, value in ba.arguments.items()
                   if name not in ('tokens', 'kernel', 'sizes', 'trans_rhs')}
        return self.bind(abstract((width, depth)), abstract((1, -(-tokens.shape[0] // unit) * unit, depth)),
                         ba.arguments['sizes'], **options, trans_rhs=True)

    def _ran(self, ba, config: mosaic_gpu.Tiles) -> tuple:
        """`_Mosaic._ran`: a swapped tile runs the call swapped (`_swapped`), the tile unswapped, on the
        target being selected for (`generation.gpu_target`); a grouped split's runs write fp32 partials, each
        at the tile unsplit (`_decode_split`)."""
        ba = self._aligned(ba)
        if not config.swapped:
            if config.splits == 1:
                return ba, config
            return (self.bind(*ba.args, **{**ba.kwargs, 'out_dtype': 'float32'}),
                    dataclasses.replace(config, splits=1))
        swapped = self._swapped(ba, generation.gpu_target())
        assert swapped is not None, "a swapped tile is a swappable call's (`legal`)"
        if config.splits > 1:
            # A split's runs write fp32 partials (`mosaic_gpu.gmm`), each at the tile unsplit.
            swapped = self.bind(*swapped.args, **{**swapped.kwargs, 'out_dtype': 'float32'})
        return swapped, dataclasses.replace(config, swapped=False, splits=1)

    def legal(self, ba, config: mosaic_gpu.Tiles, target) -> bool:
        if config.swapped and (self._swapped(ba, target) is None or not self._left_fits(ba, config)):
            return False
        # A split is a swapped dense call's or a grouped call's (a decode step's), each run some steps, of no
        # exact sums: their runs sum exactly (`mosaic_gpu.gmm`), but an A100's exact input gradient split four
        # ways at 64 x 128 x 32 (six stages, two programs) spilled 24 bytes where its register model held 187
        # (the GEMM compile gate, EM-G1), its runs' traced steps beside the running sum.
        depth = ba.arguments['tokens'].shape[1]
        if config.splits > 1 and ((ba.arguments['dense'] and not config.swapped) or ba.arguments['exact_sums']
                                  or not mosaic_gpu.split_runs(depth // config.block_k, config.splits)):
            return False
        return super().legal(ba, config, target)

    def _fwd(self, tokens: jax.Array, kernel: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), epilogue_operands: tuple = (), paired: jax.Array | None = None,
             first: jax.Array | None = None, gather: jax.Array | None = None,
             scatter: jax.Array | None = None, *, out_dtype: str, trans_rhs: bool, interpret: bool,
             hooks: Hooks = NO_HOOKS, exact_sums: bool = False,
             dense: bool = False, config: mosaic_gpu.Tiles, return_residuals: bool = False):
        del return_residuals
        return self._forward(tokens, kernel, sizes, prologue_operands, epilogue_operands, paired, first,
                             gather, scatter, out_dtype=out_dtype, trans_rhs=trans_rhs, interpret=interpret,
                             hooks=hooks, exact_sums=exact_sums, dense=dense, config=config, formats={})

    def _forward(self, tokens, kernel, sizes, prologue_operands, epilogue_operands, paired, first, gather,
                 scatter, *, out_dtype: str, trans_rhs: bool, interpret: bool, hooks: Hooks, exact_sums: bool,
                 dense: bool, config: mosaic_gpu.Tiles, formats: dict):
        """`_fwd`'s one forward, every product launch's: the call as the launch runs it, its depth and width
        padded to whole tiles (`_aligned`), its outputs sliced back; `formats` the format operands a
        subclass's binding takes (`MosaicStepScaledGmm`'s step scales, `MosaicMxGmm`'s block scales), passed
        to the mainloop as they are."""
        width = kernel.shape[1] if trans_rhs else kernel.shape[2]
        call = self._aligned(self.bind(tokens, kernel, sizes, prologue_operands, epilogue_operands, paired,
                                       first, gather, scatter, out_dtype=out_dtype, trans_rhs=trans_rhs,
                                       interpret=interpret, hooks=hooks, exact_sums=exact_sums,
                                       dense=dense)).arguments
        tokens, kernel, paired, hooks = call['tokens'], call['kernel'], call.get('paired'), call['hooks']
        prologue_operands, epilogue_operands = call['prologue_operands'], call.get('epilogue_operands', ())
        out, residuals = self._launched_call(tokens, kernel, sizes, prologue_operands, epilogue_operands,
                                             paired, first, gather, scatter, out_dtype=out_dtype,
                                             trans_rhs=trans_rhs, interpret=interpret, hooks=hooks,
                                             exact_sums=exact_sums, dense=dense, config=config,
                                             formats=formats)
        return jax.tree.map(lambda each: each[:, :width], out), residuals

    def _streamed_matrix(self, kernel, trans_rhs: bool) -> tuple:
        """The matrix the mainloop streams, whether its blocks lie transposed, and the format operands its
        stages take besides (`mosaic_gpu.gmm`'s): the call's own matrix, as it lies, none."""
        return kernel, trans_rhs, {}

    def _matrix_transposed(self, ba) -> bool:
        return self._streamed_matrix(ba.arguments['kernel'], ba.arguments['trans_rhs'])[1]

    def _launched_call(self, tokens, kernel, sizes, prologue_operands, epilogue_operands, paired, first,
                       gather, scatter, *, out_dtype: str, trans_rhs: bool, interpret: bool, hooks: Hooks,
                       exact_sums: bool, dense: bool, config: mosaic_gpu.Tiles, formats: dict):
        """`_fwd`'s launch of the call as it runs it (`_aligned`)."""
        if config.swapped:
            # The product transposed: the matrix's rows times the rows, padded to whole column tiles and
            # read transposed, the output transposed back.
            rows = tokens.shape[0]
            padded = jnp.pad(tokens, ((0, -rows % config.block_n), (0, 0)))
            tile = dataclasses.replace(config, swapped=False)
            out = mosaic_gpu.gmm(kernel[0], padded[None], sizes, tiles=tile, transpose_rhs=True,
                                 transpose_lhs=not trans_rhs, out_dtype=out_dtype,
                                 multiprocessors=self._launched(tile, interpret),
                                 interpret=INTERPRETED if interpret else False, dense=True,
                                 exact_sums=exact_sums)
            assert isinstance(out, jax.Array), "one output (`out_dtypes` is not passed)"
            return out[:, :rows].T, None
        matrix, transposed, stages = self._streamed_matrix(kernel, trans_rhs)
        return mosaic_gpu.gmm(tokens, matrix, sizes, first=first, tiles=config, transpose_rhs=transposed,
                              out_dtype=out_dtype, multiprocessors=self._launched(config, interpret),
                              interpret=INTERPRETED if interpret else False,
                              prologue_operands=prologue_operands, epilogue_operands=epilogue_operands,
                              dense=dense, paired=paired, exact_sums=exact_sums, gather=gather,
                              scatter=scatter, **stages, **formats, **hooks.kernel_options()), None


class _PackedMatrix(Protocol):
    """A packed matrix's fields as the packed launch streams them (`gemm.NvFp4`'s)."""

    words: jax.Array
    scales: jax.Array


@dataclasses.dataclass(frozen=True)
class MosaicPackedGmm(MosaicGmm):
    """`MosaicGmm` of an NVFP4 matrix (`gemm.nvfp4_product`'s packed kernel, row 3621): bfloat16 rows times
    its `kernel`, the bank as Dew stores it (`gemm.NvFp4`: its words and block scales, K-major, its shape and
    dtype the bfloat16 matrix it decodes to, `[groups, depth, width]`), streamed as its words and block
    scales and each stage decoded in place into the consumer's buffers (`mosaic_gpu.gmm`'s `scales`), so its
    tile takes `mosaic_gpu.NVFP4_STEP`-deep steps, whole 64-column tiles, the decode's shared memory and
    registers, and no cluster, split, swap or hook."""

    STEPS: ClassVar[tuple[int, ...] | None] = (mosaic_gpu.NVFP4_STEP,)

    def _streamed_matrix(self, kernel: "_PackedMatrix", trans_rhs: bool) -> tuple:
        # Its words, K-major whatever orientation its decoded shape states, and its block scales, which each
        # stage decodes (`mosaic_gpu.gmm`'s `scales`).
        del trans_rhs
        return kernel.words, True, {'scales': kernel.scales}

    def _allocations(self, ba, config: mosaic_gpu.Tiles, family: mosaic.Family) -> mosaic_gpu.Allocations:
        return mosaic_gpu.gmm_allocations(
            config, family, dtype=ba.arguments['tokens'].dtype, rhs_dtype=jnp.uint32, transpose_rhs=True,
            prologue_axes=(), prologue_dtypes=(), epilogue_axes=(), epilogue_dtypes=(),
            dtypes=(ba.arguments['out_dtype'],), packed=True)

    def candidate(self, ba, config: mosaic_gpu.Tiles, target) -> bool:
        return (config.cluster == 1 and config.splits == 1 and not config.swapped and config.block_n % 64 == 0
                and super().candidate(ba, config, target))

    def _swapped(self, ba, target):
        # Its matrix is packed: no swapped orientation reads it.
        del ba, target

    def registers(self, ba, config: mosaic_gpu.Tiles) -> int | None:
        # Beside the plain product's live set, a step's decode holds a 128-depth group's words, the code
        # being decoded and the stage's block scales, each a `[block_n, 16]` register tile over a role's
        # threads (`mosaic_gpu._stage_decoded`).
        return _beside(super().registers(ba, config), 3 * config.block_n * 16 // resources.THREADS)


@dataclasses.dataclass(frozen=True)
class MosaicTgmm(_Mosaic):
    """`Tgmm` through the template's mainloop (`mosaic_gpu.tgmm`) on a GPU family's bundle or a TPU core's:
    `block_m` x `block_n` tiles of each group's gradient, its rows `block_k` at a time."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens', 'cotangent')

    def output_sides(self, ba) -> tuple[int, int]:
        return _gradient_sides(ba)

    EXACT_SUMS: ClassVar[bool] = True
    STAGED_ROWS: ClassVar[bool] = True

    def _allocations(self, ba, config: mosaic_gpu.Tiles, family: mosaic.Family) -> mosaic_gpu.Allocations:
        return mosaic_gpu.tgmm_allocations(
            config, family, lhs_dtype=ba.arguments['tokens'].dtype,
            rhs_dtype=ba.arguments['cotangent'].dtype, prologue_axes=ba.arguments['hooks'].prologue_axes,
            prologue_dtypes=tuple(side.dtype for side in ba.arguments['prologue_operands']),
            out_dtype=ba.arguments['out_dtype'], k_major=self.K_MAJOR, step_scaled=self.K_MAJOR,
            unstaged=self.unstaged(ba))

    def _generated_extents(self, ba) -> tuple[int, int | None, int]:
        # A gradient tile's rows are the tokens' width; its steps run over a group's rows, padded to whole
        # steps (`mosaic_gpu.tgmm`).
        return ba.arguments['tokens'].shape[1], None, ba.arguments['cotangent'].shape[1]

    def _depth_unit(self, ba, family: mosaic.Family, dtype) -> int:
        # A tile's rows take whole roles of the depth, but where its last row tile may be partial
        # (`_ragged`): there whole swizzled rows of the output's, which a TMA store's tiled view takes no
        # fewer of ("dimension 1 must have size smaller or a multiple of its tiling 64, but got 200", a
        # 200-deep kernel gradient on sm90 and sm120, the compile gate at c717a2439b4).
        if self._ragged(ba, family):
            return math.lcm(mosaic_gpu.TMA_ROWS, swizzling.swizzle_columns(dtype, swizzling.SWIZZLE_BYTES))
        return family.role_rows

    def _padded_extents(self, ba) -> dict[str, int]:
        if self.K_MAJOR:  # a step-scaled call's operands lie as its scales do: not padded
            return {}
        depth, _, width = self._generated_extents(ba)
        return {'depth': depth, 'width': width}

    def _padded_dims(self, ba) -> list[_PaddedArgument]:
        hooks = ba.arguments['hooks']
        return [('tokens', ba.arguments['tokens'], {1: 'depth'}),
                ('cotangent', ba.arguments['cotangent'], {1: 'width'}),
                ('prologue_operands', ba.arguments['prologue_operands'],
                 _side_dims(ba.arguments['prologue_operands'], hooks.prologue_axes, {'depth': 'depth'}))]

    def derived(self, ba, target) -> mosaic_gpu.Tiles | None:
        # The derived tile, its contraction split where its output tiles underfill a wave
        # (`mosaic_gpu.split_count`) at the tile the split's own call, writing fp32 partials, derives. An
        # exact sum splits too, its runs summed exactly and their partials by IEEE adds (`mosaic_gpu.tgmm`):
        # unsplit, Qwen3.5-4B's key kernel gradient ran 2.03-2.06x the tensor cores' split sums on a GH200
        # (W123-gh200, W124-gh200).
        tile = super().derived(ba, target)
        if tile is None or self._balances(ba, tile, target) or self._splits(ba, tile, target) == 1:
            return tile
        partials = self._partials(ba)
        sliced = super().derived(partials, target)
        if sliced is None or (splits := self._splits(partials, sliced, target)) == 1:
            return tile
        split = dataclasses.replace(sliced, splits=splits)
        # A split a compile of this process found spilling (`report.refused`) gives way to the call's own.
        return tile if report.refused(self, ba, split) else split

    def _balances(self, ba, tile: mosaic_gpu.Tiles, target) -> bool:
        """Whether the call's last wave runs balanced over the programs (`tail.balanced`), which it
        takes no split beside: a dense one on a family that fixes a tile up across programs (its exact
        partials added by IEEE adds, `mosaic_gpu.tgmm`), its tail balanced."""
        if not (ba.arguments['dense'] and mosaic_gpu.fixes_up(tile, ba.arguments['tokens'].dtype)):
            return False
        depth, _, width = self._generated_extents(ba)
        tokens, cotangent = ba.arguments['tokens'], ba.arguments['cotangent']
        steps = -(-tokens.shape[0] // tile.block_k)
        return mosaic_gpu.balanced_tail(tile, (-(-depth // tile.block_m), width // tile.block_n), steps,
                                        (tokens.dtype.itemsize, cotangent.dtype.itemsize), target,
                                        _multiprocessors(target) * tile.programs) is not None

    def _splits(self, ba, tile: mosaic_gpu.Tiles, target) -> int:
        depth, _, width = self._generated_extents(ba)
        tokens, cotangent = ba.arguments['tokens'], ba.arguments['cotangent']
        return mosaic_gpu.split_count(tile, groups=ba.arguments['sizes'].shape[0],
                                      rows=_rows_of(_TGMMS, ba.args), depth=depth, width=width,
                                      itemsizes=(tokens.dtype.itemsize, cotangent.dtype.itemsize),
                                      multiprocessors=_multiprocessors(target),
                                      step_seconds=mosaic_gpu.step_seconds(tile, target),
                                      bandwidth=target.memory_rate, launch_seconds=target.launch_seconds)

    def _partials(self, ba):
        """The call `ba` writing fp32 partials, as a split's slices do (`mosaic_gpu.tgmm`): each slice a group
        of its own, so no dense call's (its stages staged, `unstaged`)."""
        return self.bind(*ba.args, **{**ba.kwargs, 'out_dtype': 'float32', 'dense': False})

    def _ran(self, ba, config: mosaic_gpu.Tiles) -> tuple:
        """The call and tile a launch at `config` runs (`_aligned`): a split one's fp32 partials, unsplit."""
        ba = self._aligned(ba)
        if config.splits == 1:
            return ba, config
        return self._partials(ba), dataclasses.replace(config, splits=1)

    def _divides(self, ba, config: mosaic_gpu.Tiles) -> bool:
        rows, _, width = self._generated_extents(ba)
        family = mosaic_gpu.family_of(config, ba.arguments['tokens'].dtype)
        return (rows % config.block_m == 0 or self._ragged(ba, family)) and width % config.block_n == 0

    def _ragged(self, ba, family: mosaic.Family) -> bool:
        # A depth no cooperative tile's rows divide takes a partial last row tile where the family's loads
        # past the depth read zeros (TMA's, `Family.bounded_loads`; row 2295: GPT-OSS's 2880, which held one
        # consumer's 64 x 192 tile alone, 1.79x Triton's time on an H100, where every other model's kernel
        # gradient, a cooperative tile, ran 0.78-0.92 of it, FSCELLS1).
        # A depth some cooperative tile divides keeps its whole tiles.
        rows, _, _ = self._generated_extents(ba)
        cooperative = family.role_rows * min(family.consumers, mosaic_gpu.CONSUMERS)
        prologue = ba.arguments['hooks'].prologue
        return (family.bounded_loads and rows % cooperative != 0
                and mosaic_gpu.ragged_rows(rows, prologue=prologue, k_major=self.K_MAJOR))

    def _streamed_rows(self, ba, config: mosaic_gpu.Tiles) -> tuple[int, ...]:
        # Both operands stream a step's rows (the tokens): `block_k` of them each.
        del ba
        return (config.block_k,)

    def candidate(self, ba, config: mosaic_gpu.Tiles, target) -> bool:
        """`_Mosaic.candidate` on one multiprocessor a tile (`mosaic_gpu.tgmm` takes no pair), a family
        whose products read registers loading its transposed stage in whole row atoms of its products (the
        step, `block_k`, in `Family.transposed_rows`)."""
        if config.cluster != 1 or not super().candidate(ba, config, target):
            return False
        family = mosaic_gpu.family_of(config, ba.arguments['tokens'].dtype)
        return not (family.operands_in_registers and config.block_k % family.transposed_rows)

    BLOCKS: ClassVar[dict[str, str]] = {'rows': 'block_k', 'depth': 'block_m'}


    def _fwd(self, tokens: jax.Array, cotangent: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), gather: tuple | None = None, *, out_dtype: str, interpret: bool,
             hooks: Hooks = NO_HOOKS,
             exact_sums: bool = False, dense: bool = False, config: mosaic_gpu.Tiles,
             return_residuals: bool = False):
        # The call as the launch runs it, its depth and width padded to whole tiles (`_aligned`), its output
        # sliced back.
        depth, width = tokens.shape[1], cotangent.shape[1]
        call = self._aligned(self.bind(tokens, cotangent, sizes, prologue_operands, gather,
                                       out_dtype=out_dtype, interpret=interpret, hooks=hooks,
                                       exact_sums=exact_sums, dense=dense)).arguments
        out = mosaic_gpu.tgmm(call['tokens'], call['cotangent'], sizes, tiles=config, out_dtype=out_dtype,
                              multiprocessors=self._programs(config),
                              interpret=INTERPRETED if interpret else False, prologue=hooks.before(),
                              prologue_operands=call['prologue_operands'], prologue_axes=hooks.prologue_axes,
                              exact_sums=exact_sums, dense=dense, gather=gather)
        return out[:, :depth, :width], None


@functools.cache
def cutlass_architectures() -> tuple[str, ...]:
    """The GPU generations the installed dew_cutlass holds code for, or none where it is not installed."""
    if importlib.util.find_spec('dew_cutlass') is None:
        return ()
    return importlib.import_module('dew_cutlass').ARCHITECTURES


@functools.cache
def _cutlass_registered() -> None:
    """dew_cutlass's FFI targets, registered with jax once, at the first launch that runs one."""
    for target, capsule in importlib.import_module('dew_cutlass').targets().items():
        jax.ffi.register_ffi_target(target, capsule, platform="CUDA")


@dataclasses.dataclass(frozen=True)
class _ScaledGmm(MosaicGmm):
    """`MosaicGmm` of a product whose rows' and matrix's scales its binding takes as one operand (`scales`),
    reaching the mainloop as the keyword of `mosaic_gpu.gmm`'s its subclass names (`SCALES`)."""

    SCALES: ClassVar[str]

    def _fwd(self, tokens: jax.Array, kernel: jax.Array, sizes: jax.Array, prologue_operands: tuple = (),
             epilogue_operands: tuple = (), paired: jax.Array | None = None, first: jax.Array | None = None,
             gather: jax.Array | None = None, scatter: jax.Array | None = None, *, out_dtype: str,
             trans_rhs: bool, interpret: bool, hooks: Hooks = NO_HOOKS, exact_sums: bool = False,
             dense: bool = False, scales: tuple = (), config: mosaic_gpu.Tiles,
             return_residuals: bool = False):
        del return_residuals
        return self._forward(tokens, kernel, sizes, prologue_operands, epilogue_operands, paired, first,
                             gather, scatter, out_dtype=out_dtype, trans_rhs=trans_rhs, interpret=interpret,
                             hooks=hooks, exact_sums=exact_sums, dense=dense, config=config,
                             formats={self.SCALES: tuple(scales)})


@dataclasses.dataclass(frozen=True)
class MosaicStepScaledGmm(_ScaledGmm):
    """`MosaicGmm` of a blockwise-scaled FP8 product (`gemm.blockwise_fp8_product`, row 4518), planned on the
    call it is: each exactly summed `SCALE_DEPTH`-deep step's partial times its rows' and
    columns' scales before its add (`mosaic_gpu.gmm`'s `step_scales`), its rows' scales streamed beside its
    ring (`gmm_allocations`' `step_scaled`), a column tile within one weight block's columns, and no split,
    swap or hook."""

    STEPS: ClassVar[tuple[int, ...] | None] = (SCALE_DEPTH,)
    STEP_SCALED: ClassVar[bool] = True
    SCALES: ClassVar[str] = 'step_scales'

    def candidate(self, ba, config: mosaic_gpu.Tiles, target) -> bool:
        within = SCALE_DEPTH % config.block_n == 0
        return config.splits == 1 and not config.swapped and within and super().candidate(ba, config, target)

    def _swapped(self, ba, target):
        # Its scales are its rows' and columns': no swapped orientation reads them.
        del ba, target

    def registers(self, ba, config: mosaic_gpu.Tiles) -> int | None:
        # Beside the exact product's live set, a step's column scales.
        return _beside(super().registers(ba, config), _column_scale_registers(ba, config))


@dataclasses.dataclass(frozen=True)
class MosaicMxGmm(_ScaledGmm):
    """`MosaicGmm` of a block-scaled product (`gemm.mxfp8_product`, `gemm.nvfp4_block_product`, row 4519),
    planned on the call it is: FP8 or E2M1 rows times a K-major matrix of theirs, each block of each scaled
    by its scale in the products (`mosaic_gpu.block_format`)
    (`mosaic_gpu.gmm`'s `mx_scales`, `mosaic.Issuer.scaled_products`), its block scales streamed beside its
    rings (`gmm_allocations`' `block_scaled`) on a family whose products take them (tcgen05's), its tiles
    whole 128s of rows, columns and depth, and no split, swap, hook or exact sums."""

    BLOCK_SCALED: ClassVar[bool] = True
    SCALES: ClassVar[str] = 'mx_scales'

    def _fp8(self, ba, dtype, family: mosaic.Family) -> bool:
        # Its products scale each block in the instruction, which sums in fp32 itself: no step promoted. Its
        # operands FP8 (MXFP8) or E2M1 (NVFP4), `mosaic_gpu.block_format`'s.
        return (jnp.dtype(dtype) in (*EIGHT, jnp.dtype(jnp.float4_e2m1fn)) and family.takes_block_scales
                and ba.arguments['trans_rhs'] and ba.arguments['hooks'] == NO_HOOKS
                and ba.arguments.get('paired') is None and not ba.arguments.get('exact_sums'))

    def candidate(self, ba, config: mosaic_gpu.Tiles, target) -> bool:
        whole = all(extent % 128 == 0 for extent in (config.block_m, config.block_n, config.block_k))
        return config.splits == 1 and not config.swapped and whole and super().candidate(ba, config, target)

    def _swapped(self, ba, target):
        # Its scales are its rows' and columns': no swapped orientation reads them.
        del ba, target


@dataclasses.dataclass(frozen=True)
class MosaicStepScaledTgmm(MosaicTgmm):
    """`MosaicTgmm` of a blockwise FP8 recipe's kernel gradient (`gemm.blockwise_fp8_kernel_gradient`, row
    4518), planned on the call it is: FP8 operands rows-contiguous (`mosaic_gpu.tgmm`'s `k_major`, as
    Hopper's FP8 tensor cores read both), each exactly summed 128-row step's partial times its depth rows' and
    columns' scales, its rows' scales streamed beside its ring, groups of whole steps, no split or hook."""

    STEPS: ClassVar[tuple[int, ...] | None] = (SCALE_DEPTH,)
    FP8: ClassVar[bool] = True
    K_MAJOR: ClassVar[bool] = True

    def _fwd(self, tokens: jax.Array, cotangent: jax.Array, sizes: jax.Array, prologue_operands: tuple = (),
             gather: tuple | None = None, *, out_dtype: str, interpret: bool, hooks: Hooks = NO_HOOKS,
             exact_sums: bool = False, dense: bool = False, step_scales: tuple = (),
             config: mosaic_gpu.Tiles, return_residuals: bool = False):
        # Bound as a kernel gradient states its operands, rows first: the rows-contiguous operands' transposes
        # (`gemm.blockwise_fp8_kernel_gradient`), which XLA folds into the kernel's own operands, read back as
        # they lie (`mosaic_gpu.tgmm`'s `k_major`); its scales (`step_scales`, the binding's operand) its
        # mainloop's `step_scales`.
        del prologue_operands, gather, hooks, dense, return_residuals
        out = mosaic_gpu.tgmm(tokens.T, cotangent.T, sizes, tiles=config, out_dtype=out_dtype,
                              multiprocessors=self._launched(config, interpret),
                              interpret=INTERPRETED if interpret else False, exact_sums=exact_sums,
                              k_major=True, step_scales=tuple(step_scales))
        return out, None

    def _fp8(self, ba, dtype, family: mosaic.Family) -> bool:
        # Its operands lie rows-contiguous, as Hopper's FP8 tensor cores read both.
        return (jnp.dtype(dtype) in EIGHT and family.takes_fp8
                and ba.arguments['hooks'] == NO_HOOKS and bool(ba.arguments.get('exact_sums')))

    def candidate(self, ba, config: mosaic_gpu.Tiles, target) -> bool:
        return config.splits == 1 and super().candidate(ba, config, target)

    def registers(self, ba, config: mosaic_gpu.Tiles) -> int | None:
        # Beside the exact gradient's live set, a step's column scales.
        return _beside(super().registers(ba, config), _column_scale_registers(ba, config))


def _beside(held: int | None, more: int) -> int | None:
    """A launch's registers `held` with `more` beside them; none where it states none (a TPU core's)."""
    return None if held is None else held + more


def _column_scale_registers(ba, config: mosaic_gpu.Tiles) -> int:
    """The registers a step-scaled walk holds beside the exact one's: a step's fp32 column scales in its
    accumulator's column layout (`Family.column_registers`)."""
    family = mosaic_gpu.family_of(config, ba.arguments['tokens'].dtype)
    return family.column_registers(config.block_n, jnp.float32)


def _cutlass_call(target: str, out: jax.ShapeDtypeStruct) -> Callable:
    """dew_cutlass's FFI call `target` giving `out`, a mapped call running its maps in turn, one launch
    each (row 4389: JAX's FFI batching refuses a call that states no rule)."""
    return jax.ffi.ffi_call(target, out, vmap_method='sequential')


def cutlass_tile(architecture: str, out_dtype) -> mosaic_gpu.Tiles:
    """Dew's record of the kernel the installed dew_cutlass instantiates for `architecture` writing
    `out_dtype`, as Dew's records name a launch (row 4323): the build's own statement of its compiled types
    (`dew_cutlass.KERNELS`: its tile, stages, consumer threads, cluster rows and instruction family), a
    fact of that binary, not a derivation. Its kernels band no column tiles."""
    kernel = importlib.import_module('dew_cutlass').KERNELS[(architecture, jnp.dtype(out_dtype).name)]
    rows, columns, depth = kernel.tile
    return mosaic_gpu.Tiles(rows, columns, depth, kernel.stages, 1,
                            warpgroups=kernel.consumers // resources.THREADS, cluster=kernel.cluster,
                            family=kernel.instructions)


def cutlass_built_for(generation_name: str, built: tuple[str, ...]) -> str | None:
    """The architecture of the dew_cutlass kernels `built` (`cutlass_architectures`, the installed wheel's) a
    GPU of `generation_name` runs (kernels/cutlass's ffi.cu dispatch): its own, or an sm80 build on any
    capability 8.x, whose plain SASS every later minor of its major runs (CUDA's binary compatibility);
    sm90a's kernels, of Hopper's arch-specific instructions, run on 9.0 alone. None where `built` holds none
    it runs."""
    if generation_name in built:
        return generation_name
    capability = generation.capability(generation_name)
    return 'sm80' if capability is not None and capability[0] == 8 and 'sm80' in built else None


class _Cutlass(_GroupedLaunch[mosaic_gpu.Tiles]):
    """CUTLASS's grouped GEMMs (`kernels/cutlass`, the dew-cutlass wheel) as XLA FFI calls: on Ampere and
    every 8.x part, CUTLASS 2's grouped kernel; on Hopper, CUTLASS 3's pointer-array cooperative kernel on a
    pair of multiprocessors; bf16 operands and bf16 or fp32 out, each group's shape set up on the device. A
    vendored production route on the cells where it measured faster than the template (`DECLARED`), and by
    name elsewhere, at the one kernel its build holds for the GPU and output (`tile`, the build's own record
    of it, `cutlass_tile`)."""
    name: ClassVar[str] = 'cutlass'
    config_cls: ClassVar[type[mosaic_gpu.Tiles]] = mosaic_gpu.Tiles

    def eligible(self, ba, target) -> bool:
        return cutlass_built_for(target.generation, cutlass_architectures()) is not None and self.takes(ba)

    def tile(self, ba) -> mosaic_gpu.Tiles | None:
        """The record of the kernel the build holds for the GPU being selected for (`generation.gpu_target`,
        `cutlass_built_for`) writing the call's output (`cutlass_tile`); None where it holds none."""
        built = cutlass_built_for(generation.gpu_target().generation, cutlass_architectures())
        return None if built is None else cutlass_tile(built, ba.arguments['out_dtype'])

    @staticmethod
    def takes(ba) -> bool:
        """Whether its kernels take the call wherever dew_cutlass holds them (`eligible` adds the installed
        library's generations): bf16 operands, bf16 or fp32 out, int32 sizes, every extent a multiple of 8
        elements (16-byte aligned rows of every operand and output, as its TMA copies and each group's
        pointers need), no hooks or exact sums, compiled (no interpreter), its stack read whole."""
        operands = [value for name, value in ba.arguments.items()
                    if name in ('tokens', 'kernel', 'cotangent')]
        extents = {extent for operand in operands for extent in operand.shape[1:]}
        return (all(operand.dtype == jnp.bfloat16 for operand in operands)
                and jnp.dtype(ba.arguments['out_dtype']) in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32))
                and ba.arguments['sizes'].dtype == jnp.int32 and all(extent % 8 == 0 for extent in extents)
                and ba.arguments['hooks'] == NO_HOOKS and not ba.arguments.get('exact_sums')
                and not ba.arguments['interpret'] and ba.arguments.get('first') is None)

    def space(self, ba) -> tuple[mosaic_gpu.Tiles, ...]:
        tile = self.tile(ba)
        return () if tile is None else (tile,)

    def fitting(self, ba, target) -> tuple[mosaic_gpu.Tiles, ...]:
        # Its build's kernel is the one it runs (`cutlass_tile`), sized to every part of its architecture.
        return self.space(ba)


@dataclasses.dataclass(frozen=True)
class CutlassGmm(_Cutlass):
    """`Gmm` through CUTLASS (`dew_cutlass_grouped_gemm`)."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens',)

    def _fwd(self, tokens: jax.Array, kernel: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), epilogue_operands: tuple = (), paired: jax.Array | None = None,
             first: jax.Array | None = None, gather: jax.Array | None = None,
             scatter: jax.Array | None = None, *, out_dtype: str, trans_rhs: bool, interpret: bool,
             hooks: Hooks = NO_HOOKS, exact_sums: bool = False,
             dense: bool = False, config: mosaic_gpu.Tiles, return_residuals: bool = False):
        # No hooks, side operands, second matrix, stack read from a first matrix or exact sums: the launch is
        # not eligible for them (`_Cutlass.eligible`); a dense call is one group, as any.
        del prologue_operands, epilogue_operands, paired, first, gather, scatter, interpret, hooks, exact_sums
        del dense, config
        _cutlass_registered()
        width = kernel.shape[1] if trans_rhs else kernel.shape[2]
        out = jax.ShapeDtypeStruct((tokens.shape[0], width), jnp.dtype(out_dtype))
        call = _cutlass_call('dew_cutlass_grouped_gemm', out)
        return call(tokens, kernel, sizes, transpose_w=trans_rhs), None


@dataclasses.dataclass(frozen=True)
class CutlassTgmm(_Cutlass):
    """`Tgmm` through CUTLASS (`dew_cutlass_grouped_gemm_k`): each group's gradient a GEMM over its rows,
    an empty group's zero, its pair splitting the gradient's rows."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens', 'cotangent')

    def tile(self, ba) -> mosaic_gpu.Tiles | None:
        # Dew's record of a kernel gradient pairs no row tiles (`mosaic_gpu.Tiles.cluster`: `tgmm` takes no
        # pair), whatever cluster the binary's kernel runs on.
        tile = super().tile(ba)
        return None if tile is None else dataclasses.replace(tile, cluster=1)

    def _fwd(self, tokens: jax.Array, cotangent: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), gather: tuple | None = None, *, out_dtype: str, interpret: bool,
             hooks: Hooks = NO_HOOKS,
             exact_sums: bool = False, dense: bool = False, config: mosaic_gpu.Tiles,
             return_residuals: bool = False):
        del prologue_operands, gather, interpret, hooks, exact_sums, dense, config  # as `CutlassGmm._fwd`'s
        _cutlass_registered()
        out = jax.ShapeDtypeStruct((sizes.shape[0], tokens.shape[1], cotangent.shape[1]),
                                   jnp.dtype(out_dtype))
        return _cutlass_call('dew_cutlass_grouped_gemm_k', out)(tokens, cotangent, sizes), None


def _grown(value, dims: dict | tuple, extents: dict):
    """`value` (an array, a stand-in for one, or a tuple of them, each with its own `dims`) with zeros past
    its own along each dimension `dims` names (by padded axis, `_Mosaic._aligned`) to `extents`."""
    if isinstance(value, tuple) or isinstance(dims, tuple):
        return tuple(_grown(each, each_dims, extents) for each, each_dims in zip(value, dims, strict=True))
    shape = list(value.shape)
    for dim, axis in dims.items():
        shape[dim] = extents[axis]
    if isinstance(value, jax.ShapeDtypeStruct):
        return jax.ShapeDtypeStruct(tuple(shape), value.dtype)
    return jnp.pad(value, tuple((0, size - held) for size, held in zip(shape, value.shape, strict=True)))


def _side_dims(sides: tuple, axes_of: tuple, padded: dict[str, str]) -> tuple:
    """Each side operand's dimensions by padded axis (`_grown`'s `dims`): those along a launch axis `padded`
    maps to one ('depth', or 'columns' to the width)."""
    return tuple({dim: padded[axis] for dim, axis in enumerate(axes) if axis in padded}
                 for side, axes in zip(sides, axes_of, strict=True))


def _prologue_tiled(ba) -> bool:
    """Whether the call's prologue reads a side operand's tile (`mosaic.Family.takes_prologue_tiles`)."""
    return any(len(held_axes(axes)) > 1 for axes in ba.arguments['hooks'].prologue_axes)


def _multiprocessors(target: generation.GpuTarget) -> int:
    """`target`'s multiprocessors, which the persistent Mosaic GPU launches spread their tiles over."""
    if target.multiprocessors is None:
        raise ValueError(f"{target.generation} states no multiprocessors, which the Mosaic GPU launches read")
    return target.multiprocessors


@dataclasses.dataclass(frozen=True)
class GroupedCase:
    """A grouped product as the grouped selector reads it (`GROUPED`): whether the kernels run on its
    target (`placed`: a GPU of sm80 on, or a TPU; the template's dispatch runs in the branch its platform
    places), whether they see its operands whole (`whole`,
    `dew.nn.sharding.seen_whole`), the `product` it declares, its `rows`, its launches bound to it
    (`call`) and the `kind` of call it is ('grouped forward', 'dense input_gradient', 'grouped
    kernel_gradient exact', as `gmm_call` and `tgmm_call` state it), which with the device's instruction
    family is its cell (`DECLARED`), an FP8 call's kind its own (' fp8')."""

    placed: bool
    whole: bool
    product: Product
    rows: int
    call: base.Call
    kind: str
    paired: bool = False
    """Whether it multiplies its rows by a second matrix beside the first (`gmm_call`'s `paired`)."""
    gathered: bool = False
    """Whether its rows are gathered from the tokens by a table (`gmm_call`'s `gather`)."""
    scattered: bool = False
    """Whether its rows land where a table says (`gmm_call`'s `scatter`)."""

    @property
    def cell(self) -> tuple[str | None, str]:
        return generation.family(self.call.target.generation), self.kind


_Need = selection.Requirement[GroupedCase]
"""A requirement of a `GroupedCase`."""

SUPPORTS = (
    _Need('capability', lambda case: case.placed, "the kernels need a GPU of sm80 or later", row=INHERENT),
    _Need('placement', lambda case: case.whole,
          "a mesh axis splits the operands, or a shard_map that checks varying axes holds them, and the "
          "kernels see whole arrays", row='ledger 1306'),
    _Need('product', lambda case: case.product.refusal is None, lambda case: case.product.refusal or '',
          row='ledger 1295'),
    _Need('step_sums', lambda case: not (case.product.promoted and generation.tpu(case.call.target)),
          lambda case: f"{case.call.target.generation}'s MXUs add within a step, which no measurement states "
                       "IEEE", row=INHERENT),
    _Need('rows', lambda case: case.rows > 0, "the call has no rows", row=INHERENT),
)
"""What every launch needs of a grouped product: a process that runs it, whole operands, a product the
kernels compute exactly on the target (its declaration's refusal, the target's tensor rate for its
dtype, no step-by-step sums on a TPU, whose launches do not add them apart), and rows to launch."""


def _launch_kernel(name: str, *measured: _Need) -> selection.Kernel[GroupedCase]:
    """`GROUPED`'s kernel for the launches implementation `name` runs: `SUPPORTS`, then its own
    eligibility for the call (`Template.eligible`), then what the targets measured of it (`measured`, which
    a call naming it overrides)."""
    return selection.Kernel(name, (*SUPPORTS, _Need(
        'pair', lambda case: not case.paired or _takes_pair(name, case),
        lambda case: f"the {name} launches take no paired product on {case.call.target.generation}, whose "
                     "two products then run apart", row='ledger 2100', kind='handoff'), _Need(
        # A paired call's capability is the pairing launch's alone: every other hands it on (`pair`).
        'launch', lambda case: ((case.paired and not _takes_pair(name, case))
                                or case.call.eligible(name)),
        lambda case: f"the {name} launches take no call of these dtypes, widths, side operands or hooks on "
                     f"{case.call.target.generation}", row='ledger 1295'), _Need(
        'tables', lambda case: not (case.gathered or case.scattered) or name == MosaicGmm.name,
        lambda case: f"the {name} launches read and write no rows through a table: Mosaic GPU's gathers and "
                     "scatters them", row='ledger 1376'), *measured))


DECLARED: dict[tuple[str | None, str], tuple[str, str]] = {
    ('mma.sync', 'dense forward'):
        ('xla', "DENSE1 (A100, a55c8ffb2, one program a multiprocessor): Mosaic 1.923x and Triton 1.947x "
                "XLA's dot (cuBLAS), geomean of 10 projections at 8192 tokens"),
    ('mma.sync', 'dense input_gradient'):
        ('xla', "DENSE1 (A100, one program a multiprocessor): Mosaic 2.881x, Triton 1.839x XLA's dot"),
    ('mma.sync', 'dense kernel_gradient'):
        ('xla', "DENSE1 (A100, one program a multiprocessor): Mosaic 1.964x, Triton 1.735x XLA's dot"),
    ('wgmma', 'grouped forward'):
        ('cutlass', "PAIR1-h100 (c771912a6, H100 SXM, a pair of multiprocessors a cluster multicasting each "
                    "group's matrix): the template 1.019x CUTLASS's time, geomean of 7 grouped MoE rows "
                    "(worst 1.18x, deepseek-v3 down; 0.951x its unpaired time)"),
    ('wgmma', 'grouped input_gradient'):
        ('cutlass', "PAIR1-h100: 1.018x CUTLASS's time, geomean of 7 grouped rows (worst 1.05x; 0.918x "
                    "its unpaired time)"),
    ('wgmma', 'dense forward'):
        ('xla', "DENSE1 (H100, a55c8ffb2): Mosaic 1.067x and Triton 1.616x XLA's dot (cuBLAS), geomean of "
                "10 projections at 8192 tokens"),
    ('wgmma', 'dense input_gradient'): ('xla', "DENSE1 (H100): Mosaic 1.048x, Triton 1.738x XLA's dot"),
    ('wgmma', 'dense kernel_gradient'): ('xla', "DENSE1 (H100): Mosaic 1.049x, Triton 1.149x XLA's dot"),
}
"""The grouped GEMM's declared cells (`selection.Op.declared`), where an old route measured faster than the
template, each with its receipt (owner, bible da64ae5): (instruction family, kind of call) to (the old
route, the receipt). The template is the default on every other cell; a cell leaves the list once the
template measures at least as fast on it (geomean over the receipt's rows). Nothing is ranked at runtime. A
call the template cannot run (its legality) takes the next route in `GROUPED`'s order, which is no cell of
this list."""

DECLARED_ONLY = _Need('declared', lambda case: DECLARED.get(case.cell, ('',))[0] == 'cutlass',
                      "CUTLASS's launches run on the cells where they measured faster than the template "
                      "(`DECLARED`), and where a call names them", row='bible da64ae5', kind='measured')
"""What keeps CUTLASS's launches, a vendored production route where they win (bible §6, as FA3), off every
other cell: the template, then Triton by legality, run those."""


DENSE_MEASURED = _Need('dense', lambda case: not case.kind.startswith('dense'),
                      "Triton's dense launches measured slower than XLA's dot on every generation timed "
                      "(DENSE1, 10 projections at 8192 tokens: A100 1.735x to 1.947x, H100 1.149x to 1.738x)",
                      row='bible da64ae5', kind='measured')
"""What keeps Triton's launches off a dense product the template's own launches cannot run: XLA's dot runs it,
as it measured faster than Triton's on every generation timed; a call naming Triton still runs them."""


IMPLEMENTATIONS = ((MosaicGmm(), MosaicTgmm()), (Gmm(), Tgmm()),
                   (CutlassGmm(), CutlassTgmm()))
"""Each implementation of the grouped product, in `GROUPED`'s order: its launch of a product and its launch
of a kernel gradient, one name for both."""

_GMMS = tuple(product for product, _ in IMPLEMENTATIONS)
_TGMMS = tuple(gradient for _, gradient in IMPLEMENTATIONS)

_NEEDS = {'cutlass': (DECLARED_ONLY,), 'triton': (DENSE_MEASURED,)}
"""Each old route's requirements beyond its launch's own: CUTLASS on its declared cells alone, Triton off the
dense products the template's launches cannot run."""

def _takes_pair(name: str, case: GroupedCase) -> bool:
    """Whether the launches `name` take the paired product `case` whole: Mosaic GPU's where a bundle takes it
    (`MosaicGmm._takes`); every other hands it on (`APART`)."""
    bound = case.call.bound.get(name)
    if name != MosaicGmm.name or bound is None:
        return False
    return bool(MosaicGmm()._families(bound[1], case.call.target))


class Apart:
    """`launched`'s answer for a paired product no launch takes whole: its two products run apart
    (`gemm._apart`), each through the launches that take it, as the unfused program runs
    them."""

    name = 'apart'


APART = Apart()

GROUPED: selection.Op[GroupedCase] = selection.Op(generation.Choice.GROUPED_MATMUL, (
    *(_launch_kernel(launch.name, *_NEEDS.get(launch.name, ())) for launch in _GMMS),
    selection.Kernel[GroupedCase](APART.name, (_Need('single', lambda case: case.paired,
                                                "the call is one product", row=INHERENT),), generic=True),
    selection.Kernel[GroupedCase]('xla', (), generic=True)), declared=DECLARED)
"""The grouped selector, in a fixed order, the template first (owner, bible da64ae5): Mosaic GPU's launch
where it takes the call, the TPU's launch on a TPU, Triton's where they refuse, else XLA's program, but a
declared cell runs its old route (`DECLARED`); CUTLASS's launches on their declared cells and by name
alone (`DECLARED_ONLY`). Every implementation shares the template's signature, so each binds the call's
own arguments, `_GMMS`' for a product, `_TGMMS`' for a kernel gradient."""


def grouped_case(launches, arguments, options, *, product: Product, whole: bool) -> GroupedCase:
    """The `GroupedCase` of `launches` called with `arguments` and `options` (`gmm_call`, `tgmm_call`),
    its kind of call as they state it: a kernel gradient (exactly summed or not) from `tgmm_call`'s
    launches, an input gradient or a forward from `gmm_call`'s, dense or grouped; and whether the kernels
    run on its target (`placed`): a TPU's, or a CUDA GPU's from compute capability 8.0
    (`generation.BF16_GPU`, JAX's bound for its Pallas lowerings: a T4 fails to compile them), never a
    CPU's (a CPU process interprets a GPU's only as the GPU it states, `generation.assumed`)."""
    if launches is _TGMMS:
        made = 'kernel_gradient exact' if options['exact_sums'] else 'kernel_gradient'
    else:
        made = 'input_gradient' if options['trans_rhs'] else 'forward'
    # An FP8 call's cells are its own: `DECLARED`'s receipts timed 16-bit products, and XLA's dot of FP8
    # operands sums narrower than the FP8 contract holds (row 1625), so the template takes them by default.
    eight = ' fp8' if any(jnp.dtype(argument.dtype) in EIGHT for argument in arguments[:2]) else ''
    # A gathered call's cells are its own: no old route reads gathered rows, nor did a receipt time one. A
    # kernel gradient's tables (`tgmm_call`'s) are its fifth argument, a product's (`gmm_call`'s) its eighth.
    tabled = 4 if launches is _TGMMS else 7
    gathered = len(arguments) > tabled and arguments[tabled] is not None
    scattered = launches is not _TGMMS and len(arguments) > 8 and arguments[8] is not None
    tables = f"{' gathered' if gathered else ''}{' scattered' if scattered else ''}"
    kind = f"{'dense' if options['dense'] else 'grouped'} {made}{eight}{tables}"
    call = base.call(launches, *arguments, **options)
    cuda = generation.capability(call.target.generation)
    floor = cuda is not None and cuda[0] * 10 + cuda[1] >= generation.BF16_GPU
    placed = generation.tpu(call.target) or floor
    paired = any(bound.arguments.get('paired') is not None for _, bound in call.bound.values())
    rows = _rows_of(launches, arguments)
    return GroupedCase(placed, whole, product, rows, call, kind, paired=paired, gathered=gathered,
                       scattered=scattered)


def _rows_of(launches, arguments) -> int:
    """The rows a call's product walks: its tokens', or its gather table's where it gathers its tokens (a
    kernel gradient's tokens' table, else its cotangent's, the tokens then whole)."""
    if launches is _TGMMS:
        tables = (arguments[4] if len(arguments) > 4 else None) or (None, None)
        return next((table for table in tables if table is not None), arguments[0]).shape[0]
    return (arguments[7] if len(arguments) > 7 and arguments[7] is not None else arguments[0]).shape[0]


def launched(launches, arguments, options, *, product: Product, whole: bool):
    """The launch of `launches` that runs the call (`arguments`, `options`) as `GROUPED` selects it,
    ready to call, `APART` where a paired call's products run apart, or None where XLA's program takes it:
    on the call's target, a CPU process's own (whose
    kernels run nowhere, so XLA's program takes every call and a kernel `generation.chosen` names raises)
    or the GPU it states (`generation.assumed`). Every branch insists on a named kernel."""
    case = grouped_case(launches, arguments, options, product=product, whole=whole)
    route = selection.select(GROUPED, case)
    if route.kernel in ('xla', APART.name):
        return None if route.kernel == 'xla' else APART
    return functools.partial(case.call.implementation(route.kernel), *arguments, **options)


def gmm_call(tokens, kernel, sizes, prologue_operands: tuple = (), epilogue_operands: tuple = (),
             paired: jax.Array | None = None, first: jax.Array | None = None, gather: jax.Array | None = None,
             scatter: jax.Array | None = None, *, out_dtype, trans_rhs: bool, hooks: Hooks = NO_HOOKS,
             interpret: bool, promoted: bool = False, dense: bool = False) -> tuple:
    """The launches, arguments and options of each group's rows of `tokens` times its matrix of `kernel`,
    both of one dtype, with the template's hooks (`gemm.gemm`). `promoted`,
    the declared product's (`templates.product.Product`), adds each depth step's partial apart
    (`exact_sums`); `dense`, one group of every row (a dense product), the plain tiled kernel where Triton
    runs it; `paired`, a second matrix of `kernel`'s shape multiplied by the same rows (Mosaic GPU's
    launch alone takes it); `first`, an int32 scalar, the matrix of `kernel`, a stack of more matrices
    than `sizes` has groups, that the first group reads, group `g` reading `first + g` where it lies (a
    banked layer's experts, `gemm.gemm`'s `layer`); `gather`, int32 `[rows]`, the rows of `tokens` the
    product's rows are, in order (a mixture of experts' tokens read where they lie, row 1376); `scatter`,
    int32 `[rows]` a permutation, the output row each product row lands at (its slots written in token
    order); Mosaic GPU's launch alone takes either."""
    optional = (paired, first, gather, scatter)
    last = max((place for place, value in enumerate(optional) if value is not None), default=-1)
    arguments = (tokens, kernel, sizes, tuple(prologue_operands), tuple(epilogue_operands),
                 *optional[:last + 1])
    options = {'out_dtype': jnp.dtype(out_dtype).name, 'trans_rhs': trans_rhs, 'hooks': hooks,
               'interpret': interpret, 'exact_sums': promoted, 'dense': dense}
    return _GMMS, arguments, options


def tgmm_call(tokens, cotangent, sizes, prologue_operands: tuple = (), gather: tuple | None = None, *,
              out_dtype, hooks: Hooks = NO_HOOKS, interpret: bool, promoted: bool = False,
              dense: bool = False) -> tuple:
    """The launches, arguments and options of each group's rows of `tokens` transposed times its rows of
    `cotangent`, both of one dtype, with the template's prologue
    (`gemm.gemm_t`); `promoted` and `dense` as `gmm_call`'s; `gather`, a table for either operand
    (`(tokens_table, cotangent_table)`, int32 `[rows]` or None), that operand's rows at its entries
    (`gmm_call`'s `gather`: a mixture of experts' tokens and a scattered product's cotangent, read where
    they lie), Mosaic GPU's launch's alone."""
    arguments = (tokens, cotangent, sizes, tuple(prologue_operands),
                 *(() if gather is None else (tuple(gather),)))
    options = {'out_dtype': jnp.dtype(out_dtype).name, 'hooks': hooks, 'interpret': interpret,
               'exact_sums': promoted, 'dense': dense}
    return _TGMMS, arguments, options


def _on_platform(kernel, fallback, interpret_on_cpu: bool):
    """`kernel` where the call lowers for its target's platform (CUDA, or a TPU's), and `fallback`
    everywhere else: a TPU's program, which never runs on CUDA, holds no CUDA branch, and a GPU's no TPU
    branch. A test's `interpret_on_cpu` interprets the kernel in a CPU process, as the GPU or TPU it
    emulates (`generation.assumed`), which it must state: interpreted for no device's facts, a launch
    would have none to derive its tile from, and the call would quietly run `fallback`. A new family
    member runs its template after the hardware-free checks, its native qualification an open item until
    measured (bible §6)."""
    target = generation.gpu_target()
    tpu = generation.tpu(target)
    branches = {'tpu' if tpu else 'cuda': functools.partial(kernel, interpret=False)}
    if interpret_on_cpu and jax.default_backend() == 'cpu':
        if not tpu and generation.capability(target.generation) is None:
            raise ValueError("interpret_on_cpu interprets the kernel as the GPU or TPU this CPU process "
                             "emulates; state it with generation.assumed(...)")
        branches['cpu'] = functools.partial(kernel, interpret=True)
    return branches, fallback


def _as_multiplied(tokens: jax.Array, kernel: jax.Array, precision: PrecisionLike) -> tuple:
    """`tokens` and `kernel` as XLA's portable product multiplies them, and its precision (`xla_ragged_dot`,
    `xla_dot`)."""
    if any(operand.dtype in EIGHT for operand in (tokens, kernel)):
        # FP8 codes are each exact in bfloat16: the product is theirs there, summed in the preferred type
        # (the FP8 contract's reference arm, testing.md, FP8 products). XLA's GPU compiler rewrites no
        # ragged dot of widened FP8 codes into its narrower-summing FP8 GEMM (sm90's lowering of this
        # product holds no F8 GEMM, FS-1625-HLO, row 1625).
        return tokens.astype(jnp.bfloat16), kernel.astype(jnp.bfloat16), jax.lax.Precision.DEFAULT
    if all(jnp.finfo(operand.dtype).bits == 16 for operand in (tokens, kernel)):
        return tokens, kernel, jax.lax.Precision.DEFAULT
    if tokens.dtype != kernel.dtype:
        # A 16-bit operand beside a wider one would reach the TPU kernel
        # 16-bit at HIGHEST too; widened, it is the same values (exactly),
        # and the wider operand keeps the precision the call asked for.
        wide = jnp.promote_types(tokens.dtype, kernel.dtype)
        return tokens.astype(wide), kernel.astype(wide), precision
    return tokens, kernel, precision


def xla_dot(tokens: jax.Array, matrix: jax.Array, *, precision: PrecisionLike = None,
            preferred_element_type: Dtype | None = None) -> jax.Array:
    """A dense product, every row through one matrix (`[depth, width]`), as `xla_ragged_dot` multiplies a
    group's rows (`_as_multiplied`), as XLA's own dot: no group, so no row masked, and no ragged dot to
    trace or to expand (a dense call's portable branch, `gemm`'s), summed as `precision.accumulated` sums a
    contraction (a CPU's float32 product as a vectorized dot sums it, row 4863)."""
    tokens, matrix, precision = _as_multiplied(tokens, matrix, precision)
    return accumulated(functools.partial(jax.lax.dot_general, dimension_numbers=(((1,), (0,)), ((), ())),
                                         precision=precision),
                       tokens, matrix, preferred_element_type=preferred_element_type)


def xla_ragged_dot(tokens: jax.Array, kernel: jax.Array, group_sizes: jax.Array, *,
                   precision: PrecisionLike = None,
                   preferred_element_type: Dtype | None = None) -> jax.Array:
    """`jax.lax.ragged_dot` as every backend runs it alike.

    A 16-bit product is exact at any precision, and XLA's TPU ragged-dot
    kernel refuses 16-bit operands at HIGHEST ("Bad lhs type"), so two of
    them multiply at DEFAULT: the same products, summed in the preferred
    type. A 16-bit operand beside a wider one is widened to it.
    XLA's TPU kernel writes values into the rows past the groups, where
    other backends write zeros, and its lhs cotangent is the same kernel.
    Those rows are zeroed going in and coming out, so neither the output
    nor a dropped row's gradient carries them. Its sums are `precision.accumulated`'s (a CPU's float32
    product as a vectorized dot sums it, row 4863).
    """
    tokens, kernel, precision = _as_multiplied(tokens, kernel, precision)
    # bincount widens under x64; TPU ragged-dot only lowers int32 counts.
    # Keep the wider domain for CPU calls whose row count actually needs it.
    index_dtype = jnp.int32 if tokens.shape[0] <= np.iinfo(np.int32).max else jnp.int64
    if group_sizes.dtype == jnp.int64:
        group_sizes = group_sizes.astype(index_dtype)
    grouped = jnp.arange(tokens.shape[0], dtype=index_dtype)[:, None] < jnp.sum(
        group_sizes, dtype=index_dtype
    )
    product = functools.partial(jax.lax.ragged_dot_general, ragged_dot_dimension_numbers=_GROUPED_ROWS,
                                precision=precision,
                                out_sharding=ragged_placement(((tokens, 0), (kernel, kernel.ndim - 1))))
    out = accumulated(product, jnp.where(grouped, tokens, 0), kernel, group_sizes,
                      preferred_element_type=preferred_element_type)
    return jnp.where(grouped, out, 0)


_GROUPED_ROWS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((1,), (1,)), ((), ())), lhs_ragged_dimensions=(0,), rhs_group_dimensions=(0,))
"""`jax.lax.ragged_dot`'s own: each group's rows of `[rows, depth]` times its `[depth, width]` matrix."""


def ragged_placement(sources: tuple) -> jax.sharding.NamedSharding | None:
    """Where a ragged product lands where an operand's type places it on a mesh with Explicit axes, split or
    not (`out_sharding`: JAX infers no placement for a ragged dot, "Explicit sharding inference for
    ragged_dot_general is not currently implemented"): each output dimension placed as the operand
    dimension it comes from (`sources`, each `(operand, axis)`, or None for one no operand dimension gives: a
    group axis), a contraction's sum whole, a mesh axis placing two dimensions placing the first alone; None
    where no operand's type names such a mesh."""
    from dew.mesh import mesh_axes

    def placed(operand):
        sharding = jax.typeof(operand).sharding
        explicit = isinstance(sharding, jax.sharding.NamedSharding) and sharding.mesh.explicit_axes
        return sharding if explicit else None

    shardings = [None if source is None else placed(source[0]) for source in sources]
    mesh = next((each.mesh for each in shardings if each is not None), None)
    if mesh is None:
        return None
    spec, used = [], set()
    for source, each in zip(sources, shardings, strict=True):
        axis = 0 if source is None else source[1] % source[0].ndim
        entry = None if each is None or axis >= len(each.spec) else each.spec[axis]
        axes = set(mesh_axes(entry)) if entry is not None else set()
        spec.append(entry if axes and not axes & used else None)
        used |= axes if spec[-1] is not None else set()
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))


def step_scaled(tokens: jax.Array, kernel: jax.Array, group_sizes: jax.Array, row_scales: jax.Array,
                column_scales: jax.Array, *, precision: PrecisionLike = None) -> jax.Array:
    """Each group's rows of `tokens` `[rows, depth]` times its matrix of `kernel` `[groups, depth, width]`,
    each `SCALE_DEPTH`-deep block's fp32 partial times its rows' scales (`row_scales`, `[depth / 128,
    rows]`) and its group's column scales (`column_scales`, `[groups, depth / 128, width / c]`, each
    scale `c` columns': 128 a weight block's, 1 a column's) before it is added into the fp32 sum, in the
    blocks' order: a blockwise-scaled FP8 product's sum (DeepSeek-V3's recipe, row 4518), the GEMM
    template's portable branch. The rows past the groups are zero."""
    blocks, rows = row_scales.shape
    groups, _, scaled = column_scales.shape
    width = kernel.shape[2]
    if tokens.shape[1] != blocks * SCALE_DEPTH or width % scaled:
        raise ValueError(f"scales {row_scales.shape} and {column_scales.shape} do not cover a "
                         f"{tokens.shape} x {kernel.shape} product in {SCALE_DEPTH}-deep blocks")
    owner = jnp.minimum(jnp.repeat(jnp.arange(groups), group_sizes, total_repeat_length=rows), groups - 1)

    def block(index, total):
        lhs = jax.lax.dynamic_slice_in_dim(tokens, index * SCALE_DEPTH, SCALE_DEPTH, axis=1)
        rhs = jax.lax.dynamic_slice_in_dim(kernel, index * SCALE_DEPTH, SCALE_DEPTH, axis=1)
        partial = xla_ragged_dot(lhs, rhs, group_sizes, precision=precision,
                                 preferred_element_type=jnp.float32)
        columns = jnp.repeat(column_scales[:, index][owner], width // scaled, axis=1)
        return total + partial * row_scales[index][:, None].astype(jnp.float32) * columns.astype(jnp.float32)

    return jax.lax.fori_loop(0, blocks, block, jnp.zeros((rows, width), jnp.float32))


def _portable_projection(x: jax.Array, kernel: jax.Array, group_sizes: jax.Array,
                dtype: Dtype | None, precision: PrecisionLike, out_dtype: Dtype | None = None) -> jax.Array:
    """The grouped projection on `xla_ragged_dot`, in every differentiation mode: `x`'s rows `[rows, in]`
    through each group's matrix of `kernel` `[groups, in, out]`, both cast to `dtype` (flax's promotion
    where None), summed in at least fp32 and rounded once. Its tangent is `dx @ Q(kernel) + Q(x) @
    dkernel`, `Q` the rounded operands, summed over the widest of the compute, input and kernel dtypes,
    so an input gradient keeps its input's dtype and a kernel gradient its master's: the portable
    projection `dew.nn.moe.expert_projection`'s XLA route and the GEMM template's portable branch run.
    `out_dtype`, where given, is the dtype its sum rounds to in place of the products' (the template's
    fp32 accumulator, which its epilogue reads). A protected span (`portable_projection`,
    `dew.nn.derivatives.declared_tangent`): each rounded operand and barrier holds a rounding the algebra
    pass must not move."""
    x, kernel = promote_dtype(x, kernel, dtype=dtype)
    accumulated = xla_ragged_dot(x, kernel, group_sizes, precision=precision,
                                 preferred_element_type=at_least_fp32(x.dtype))
    return accumulated.astype(out_dtype or x.dtype)


def _portable_projection_jvp(dtype: Dtype | None, precision: PrecisionLike, out_dtype: Dtype | None,
                    primals: tuple[jax.Array, jax.Array, jax.Array],
                    tangents: tuple[jax.Array, jax.Array, jax.Array]
                    ) -> tuple[jax.Array, jax.Array | SymbolicZero]:
    x, kernel, group_sizes = primals
    dx, dkernel, _ = tangents
    output = jnp.asarray(portable_projection(x, kernel, group_sizes, dtype, precision, out_dtype))
    work = at_least_fp32(jnp.result_type(output.dtype, x.dtype, kernel.dtype))
    compute = jnp.dtype(dtype) if dtype is not None else jnp.result_type(x.dtype, kernel.dtype)
    # A symbolic zero tangent contributes no term, so none is formed.
    live = [(name, tangent) for name, tangent in (('x', dx), ('kernel', dkernel))
            if not isinstance(tangent, SymbolicZero)]
    if not live:
        return output, SymbolicZero(jax.typeof(output))
    # The tangents keep their dtypes to this point whatever produced them, so an activation's bf16 cotangent
    # is not fused into a wider expression: XLA's excess-precision simplification (`SimplifyFPConversions`,
    # on under `xla_allow_excess_precision`, its default) drops a round trip through bf16 a fusion forms.
    held = dict(zip((name for name, _ in live),
                    jax.lax.optimization_barrier(tuple(tangent for _, tangent in live)),
                    strict=True))
    terms = []
    # Each rounded operand is held in the work dtype, which holds the rounding whatever the operand's own
    # (an fp16 master's largest value rounds to bf16's 65536, which fp16 does not hold).
    if 'x' in held:
        matrix = jnp.asarray(rounded_operand(kernel.astype(work), compute))
        terms.append(xla_ragged_dot(
            held['x'].astype(work), matrix.astype(work), group_sizes, precision=precision,
            preferred_element_type=work))
    if 'kernel' in held:
        inputs = jnp.asarray(rounded_operand(x.astype(work), compute))
        terms.append(xla_ragged_dot(
            inputs.astype(work), held['kernel'].astype(work), group_sizes, precision=precision,
            preferred_element_type=work))
    total = terms[0] if len(terms) == 1 else terms[0] + terms[1]
    tangent = jax.lax.optimization_barrier(total.astype(output.dtype))
    return output, tangent


portable_projection = contracted("test_a_portable_grouped_projection_holds_every_order")(
    declared_tangent(Protected(), _portable_projection, _portable_projection_jvp,
                     nondiff_argnums=(3, 4, 5), symbolic_zeros=True))


