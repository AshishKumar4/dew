"""The grouped matmul's kernel launches, which the GEMM template (`kernels.gemm`) runs and
differentiates (`dew.nn.moe.expert_projection` through it), and XLA's portable projection.

Three products, in MaxText's megablox shape: the forward `gmm`, the input gradient `gmm` against the
transposed kernel, and the kernel gradient `tgmm`. Each is a template launch (`templates.base`) with
its implementations taken per call in a fixed order among those that run it: Dew's Pallas Mosaic GPU
kernels (`templates.gemm.mosaic_gpu`) on their instruction family's bundle, the TPU's Mosaic kernels
(`templates.gemm.mosaic_tpu`), then JAX's Pallas Triton kernels (`ragged_dot.py`, vendored), on every
CUDA GPU from sm80 and on a CPU that interprets them, for a call neither Mosaic launch takes; and
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
from typing import ClassVar

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
from jax.experimental.pallas import mosaic_gpu as plgpu, tpu as pltpu

from ..derivatives import Protected, declared_tangent, marked_call_p
from ..precision import at_least_fp32, rounded_operand
from ..programs import equations
from . import generation, selection, tuning
from .generation import INHERENT
from .templates import base, bundles, mosaic, report, resources, tail as tail_of
from .templates.axes import GROUPS, held as held_axes
from .templates.gemm import mosaic_gpu, mosaic_tpu
from .templates.product import EIGHT, Product


@dataclasses.dataclass(frozen=True)
class Tile:
    """One Triton launch's output tile, its depth step, and its warps and pipeline stages."""
    block_m: int
    block_n: int
    block_k: int
    warps: int
    stages: int


def _doubling(least: int, most: int) -> tuple[int, ...]:
    """`least` and its doublings up to `most`, ascending."""
    return tuple(least << power for power in range(max(most // least, 1).bit_length())
                 if least << power <= most)


def _covering(extent: int) -> int:
    """The least power of two at least `extent`."""
    return 1 << (max(extent, 1) - 1).bit_length()


INTERPRETED = plgpu.InterpretGPUParams(out_of_bounds_reads="uninitialized")
"""How a CPU interprets the Mosaic GPU launches: a read past an operand's rows, which a TMA load zeroes,
gives the poison every buffer starts with (NaN), as an output a kernel fails to store does, so the
interpreter shows both."""

TPU_INTERPRETED = pltpu.InterpretParams()
"""How a CPU interprets the TPU launches: as a TPU runs them, its DMAs and VMEM, every buffer starting
as NaN and a read out of bounds raising."""


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
    back at `positions`."""
    sides = arguments[tiles:]
    return hook(*arguments[:tiles], *(jnp.expand_dims(side, where) if where else side
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
    drawn unevenly, one group left empty, as a step routes them."""
    groups = specs[2].shape[0]
    shares = np.random.default_rng(seed).dirichlet(np.ones(groups))
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
        """`tuning.KernelOp.keyed`, each side operand's dimension that follows the rows bucketed with the
        tokens' (`ROWS`), so the launch a key measures loads side tiles over all its rows."""
        keyed = super().keyed(ba)
        rows = keyed.arguments['tokens'].shape[0]
        hooks = ba.arguments.get('hooks', NO_HOOKS)
        arguments = dict(keyed.arguments)
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
        routes them, an integer side operand drawn as any other operand."""
        arrays = super().operands(specs, seed)
        arrays[2] = _sizes(specs, seed, specs[0].shape[0])
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
        held = accumulators * rows * columns // (32 * config.warps) < resources.facts().thread_registers
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
        # Pallas Triton's pipeline holds `stages - 1` depth steps, and on Hopper `stages` where it
        # pipelines a wgmma, which reads one step while the next loads; an A100 and an H100 refuse
        # exactly those bytes, so a Hopper tile is sized for the larger. The prologue's side tiles load
        # in the same steps.
        held = 0 if target.wgmma else 1
        step = self.loaded(config) * jnp.dtype(ba.arguments['tokens'].dtype).itemsize + _side_tile_bytes(
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
             *, out_dtype: str, trans_rhs: bool, interpret: bool, hooks: Hooks = NO_HOOKS,
             exact_sums: bool = False,
             dense: bool = False, config: Tile, return_residuals: bool = False):
        del paired  # a second matrix is Mosaic GPU's alone (`eligible`)
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
                              out_dtype=jnp.dtype(out_dtype), prologue_operands=prologue_operands,
                              epilogue_operands=epilogue_operands, exact_sums=exact_sums,
                              **hooks.kernel_options()), None


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
             prologue_operands: tuple = (), *, out_dtype: str, interpret: bool, hooks: Hooks = NO_HOOKS,
             exact_sums: bool = False, dense: bool = False, config: Tile, return_residuals: bool = False):
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


class _Mosaic(_GroupedLaunch[mosaic_gpu.Tiles]):
    """JAX's Pallas Mosaic GPU kernels (`templates.gemm.mosaic_gpu`) on their instruction family's bundle
    (Hopper's warpgroup MMA, Ampere's mma.sync), one 16-bit dtype, their one tile generated from the
    family's facts, the device's and the call's extents (`mosaic_gpu.derived`). Each runs one warpgroup a
    multiprocessor, or two consumers beside a producer (a cooperative tile), persistent over the output
    tiles in a planar snake, so the rows a band of `grid_block_n` column tiles shares and the matrix block
    the row tiles in flight share come from the L2."""
    name: ClassVar[str] = 'mosaic_gpu'
    config_cls: ClassVar[type[mosaic_gpu.Tiles]] = mosaic_gpu.Tiles
    GENERATES: ClassVar[bool] = True
    EXACT_SUMS: ClassVar[bool] = False
    """Whether the kernel adds each depth step's product apart (`exact_sums`); one that does not is not
    eligible where the call asks for it."""
    STAGED_ROWS: ClassVar[bool] = False
    """Whether a consumer prepares its share of each step's rows stage itself (the kernel gradient's
    prologue and group zeroing, `registers`)."""
    FP8: ClassVar[bool] = False
    """Whether the launch multiplies FP8 operands (`_fp8`): a forward product's alone."""

    def eligible(self, ba, target) -> bool:
        first, second = (value.dtype for name, value in ba.arguments.items()
                         if name in ('tokens', 'kernel', 'cotangent'))
        # A hooked or exactly summed call runs where its bundle's register model counts what they hold; a
        # paired product's epilogue over its two accumulators alone every family's counts (`registers`).
        family = self._family(ba, target)
        special = ((ba.arguments['hooks'] != NO_HOOKS or ba.arguments.get('exact_sums'))
                   and not (family is not None and _pair_alone(ba, family)))
        # A prologue's side operand by group streams with no group's origin here; Triton's and the TPU's
        # launches read it at their tile's group (`axes.at_group`).
        grouped = any(GROUPS in axes for axes in ba.arguments['hooks'].prologue_axes)
        return (first == second and (jnp.dtype(first).itemsize == 2 or self._fp8(ba, first, family))
                and family is not None and not grouped
                and (family.counts_hooks or not special)
                and self._hooks_lower(ba, first, target)
                and (self.EXACT_SUMS or not ba.arguments.get('exact_sums'))
                and self.derived(ba, target) is not None)

    def registers(self, ba, config: mosaic_gpu.Tiles) -> int:
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
        family, block = mosaic_gpu.family_of(config, dtype), (config.block_m, config.block_n, config.block_k)
        hooks = ba.arguments['hooks']
        stored = hooks.out_dtypes or (ba.arguments['out_dtype'],)
        plain = resources.tile_registers(family, config.warpgroups, block, dtype, stored=stored,
                                        rhs_transposed=ba.arguments.get('trans_rhs', False))
        accumulator = -(-config.block_m * config.block_n // config.warpgroups // resources.THREADS)
        # A kernel gradient's consumer prepares its depth half of each step's rows stage in registers (the
        # neighbouring groups' rows zeroed: `mosaic_gpu._staged_halves`), apart from the products, which
        # read shared memory on such a family; mma.sync's products hold that stage in registers already.
        # sm90's kernel gradient needs 24 more than the forward at 128 x 192 and 128 x 256 (148 and 181,
        # DECIDE1).
        # Its rewrite and its zeroing address the stage's tiles themselves: its depth half of the rows, a
        # rows atom past it, and the cotangent's atom it loads and stores (`_staged_halves`; 3117's census,
        # 8 registers at 256 x 64 exact normed).
        staged = rewritten = 0
        if self.STAGED_ROWS and not family.operands_in_registers:
            words = -(-config.block_k * config.block_m // config.warpgroups * jnp.dtype(dtype).itemsize
                      // 4 // resources.THREADS)
            atom = (config.block_k, mosaic.swizzle_columns(dtype, mosaic.SWIZZLE_BYTES))
            tiles = [((config.block_k, config.block_m // config.warpgroups), dtype)] + [(atom, dtype)] * 3
            rewritten = family.scalar_registers(tiles) - family.scalar_registers()
            staged = max(0, words - family.held_registers) + rewritten
        prologue = self._prologue_registers(ba, config)
        prologue += rewritten if prologue else 0
        phases = [staged, prologue, self._epilogue_registers(ba, config, stored)]
        # A paired product's second accumulator, carried beside the first (`mosaic_gpu.gmm`'s `paired`).
        plain += accumulator if ba.arguments.get('paired') is not None else 0
        if ba.arguments.get('exact_sums'):
            # The running sum, a second fp32 accumulator (`mosaic_gpu.tgmm`'s exact sums), is carried as a
            # loop value beside each step's scoped product, awaited before its add: the two accumulators and
            # the walk's control, no step's descriptors beside them (`resources.CONTROL_REGISTERS`), else the
            # staged rows above them (FS-K01 model2: 144 at 64 a thread over stages 2-5, 208 at 96).
            return max(2 * accumulator + resources.CONTROL_REGISTERS, plain + max(phases))
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
        return -(-share * ((1 + tiles) * 4 + (1 + in_flight) * stored) // (4 * resources.THREADS))

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
        share = config.block_m // config.warpgroups
        staging = 0
        if not family.stores_through_registers:
            chunk, _ = mosaic.output_staging(config.block_n, stored)
            widest = max(jnp.dtype(dtype).itemsize for dtype in stored)
            staging = -(-share * chunk * widest // (4 * resources.THREADS)) + resources.CONTROL_REGISTERS
        if hooks.epilogue is None:
            return max(0, staging - family.held_registers)
        # Column vectors beside the accumulator, in its column layout (`Family.column_registers`), and a
        # paired product's combination of its two accumulators, which holds a row of its results, one fp32
        # column vector's worth (its SwiGLU spilled 32 bytes at 128 x 192 at 232, PAIRED-9, and fit 224 at
        # 384 x 64, PAIRED-11): each with the walk's control alone (`resources.CONTROL_REGISTERS`), their
        # excess over the mainloop's held set.
        vectors = sum(family.column_registers(config.block_n, side.dtype)
                      for side, axes in zip(ba.arguments.get('epilogue_operands', ()), hooks.epilogue_axes,
                                            strict=True) if held_axes(axes) == ('columns',))
        vectors += family.column_registers(config.block_n, jnp.float32) if hooks.accumulators > 1 else 0
        accumulator = -(-config.block_m * config.block_n // config.warpgroups // resources.THREADS)
        tiles = sum(len(held_axes(axes)) > 1 for axes in hooks.epilogue_axes) * accumulator
        held = sum(-(-accumulator * jnp.dtype(dtype).itemsize // 4)
                   for dtype in (hooks.out_dtypes or ())[:-1])
        stage = max(staging, resources.CONTROL_REGISTERS) + vectors
        return max(0, stage - family.held_registers) + tiles + held

    def register_budget(self, config: mosaic_gpu.Tiles) -> int:
        # Its family's budget a consumer, whatever the operands' dtype (`mosaic.Family.registers`).
        return mosaic_gpu.family_of(config, jnp.bfloat16).registers(config.warpgroups)

    def _stored_rows(self, ba, config: mosaic_gpu.Tiles) -> int:
        """The multiple a consumer's stored rows of an output tile come in: a grouped call's ragged to a
        row, a dense call's whole `TMA_ROWS` (its rows padded to them, its tiles starting at whole ones)."""
        del config
        return mosaic_gpu.TMA_ROWS if ba.arguments.get('dense') else 1

    def _granular(self, ba, config: mosaic_gpu.Tiles, family: mosaic.Family) -> bool:
        """Whether `config` is in `family`'s granularity (`candidate`): at most its consumers, at most the
        programs its registers hold, a ring of two stages at least, whole roles of rows for each consumer,
        whole column units, rows it stores evenly, one of its depths, and each streamed block's rows in
        its consumers' equal shares (`Family.shares_rows`)."""
        dtype = ba.arguments['tokens'].dtype
        return not (config.warpgroups > family.consumers
                    or config.programs > family.programs(config.warpgroups) or config.stages < 2
                    or config.block_m % (family.role_rows * config.warpgroups)
                    or config.block_n % family.column_unit(dtype)
                    or not family.stores_columns(config.block_n, self._stored_rows(ba, config))
                    or config.block_k not in family.depths(dtype)
                    or not all(family.shares_rows(rows, config.warpgroups)
                               for rows in self._streamed_rows(ba, config)))

    def _streamed_rows(self, ba, config: mosaic_gpu.Tiles) -> tuple[int, ...]:
        """The rows of each block a step streams, each its leading axis as it lies (its consumers' shares
        of them, `Family.shares_rows`): the rows' tile's (`block_m`, or `block_k` read transposed) and the
        matrix's (`block_k`, or `block_n` read transposed)."""
        return config.block_m, config.block_k, *((config.block_n,) if ba.arguments.get('trans_rhs') else ())

    def _fp8(self, ba, dtype, family: mosaic.Family | None) -> bool:
        """Whether an FP8 call (`product.EIGHT`) runs here: a forward product on warpgroup MMA, its matrix
        K-major (`trans_rhs`, as Hopper's FP8 tensor cores read both operands; an N-major FP8 matrix is
        refused, row 3620), with no hook, its running sum promoted every step (`exact_sums`)."""
        return (jnp.dtype(dtype) in EIGHT and self.FP8 and family is not None and family.takes_fp8
                and ba.arguments['trans_rhs']
                and ba.arguments['hooks'] == NO_HOOKS and ba.arguments.get('paired') is None
                and bool(ba.arguments.get('exact_sums')))

    def _family(self, ba, target) -> mosaic.Family | None:
        """The bundle the launch runs on `target`: its instruction family's (`bundles.of`), none where the
        family has none (a TPU's launches are their own; a CPU interprets a GPU's only where
        `generation.assumed` names it)."""
        return bundles.of(target.generation, ba.arguments['tokens'].dtype)

    def _families(self, ba, target) -> tuple[mosaic.Family, ...]:
        """The bundles the launch may run on `target`, in preference order (`bundles.layouts`)."""
        return bundles.layouts(target.generation, ba.arguments['tokens'].dtype)

    @abc.abstractmethod
    def _generated_extents(self, ba) -> tuple[int, int | None, int]:
        """A group's output rows (`rows`, on average), the depth the steps run over (None: a group's rows,
        padded, the output's rows then whole, not a group's) and the output's width, as
        `mosaic_gpu.derived` reads them."""

    def derived(self, ba, target) -> mosaic_gpu.Tiles | None:
        return self._derived(ba, target)

    def _derived(self, ba, target, admits=None) -> mosaic_gpu.Tiles | None:
        """The derivation's tile of `ba` on `target` (`mosaic_gpu.derived`), each candidate also one `admits`
        takes where given (a swapped call's own rule, `MosaicGmm._left_fits`)."""
        layouts = self._families(ba, target)
        if not layouts:
            return None
        rows, depth, width = self._generated_extents(ba)
        return mosaic_gpu.derived(layouts, rows=rows, depth=depth, width=width,
                                  dtype=ba.arguments['tokens'].dtype,
                                  multiprocessors=_multiprocessors(target), target=target,
                                  candidate=lambda tile: (self.candidate(ba, tile, target)
                                                          and (admits is None or admits(tile))),
                                  launch_of=lambda tile: self.launch_of(ba, tile), grouped=depth is not None)

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
        steps = None if depth is None else -(-depth // config.block_k)

        def scope(stages: int) -> resources.Scope:
            staged = dataclasses.replace(config, stages=stages)
            return self._allocations(ba, staged, family).scope(family, staged, steps)

        shared = resources.program_shared(generation.shared_budget(generation.gpu_target()), config.programs)
        most = shared // ((config.block_m + config.block_n) * config.block_k)
        if steps is not None:
            # A ring deeper than its contraction's steps holds no more of them.
            most = min(most, max(2, steps))
        return resources.Launch(scope, config.warpgroups, self.registers(ba, config), most=most,
                                tile=launched)

    def candidate(self, ba, config: mosaic_gpu.Tiles, target) -> bool:
        """Whether `config` is a tile of the call on `target` apart from its resources: of the launch's
        family there (a layout of it, `bundles.layouts`), in its granularity (whole roles of rows for each
        consumer, whole column units, one of
        its depths, at most its consumers, at most the programs its registers hold) with a ring of two
        stages at least (a walk keeping one step's products in flight holds its slot while the next step's
        loads fill another), its blocks dividing the call's extents, its cluster at most the family's
        (`mosaic.Family.cluster`), and a one-dimensional side operand's block, copied untiled, at most one
        copy's extent along its axis (`mosaic.Family.copy_extent`); none a compile of this process found
        spilling (`report.refused`), so the derivation takes the next tile by its own rule."""
        family = next((bundle for bundle in self._families(ba, target) if bundle.name == config.family), None)
        if (family is None or config.cluster > family.cluster or not self._divides(ba, config)
                or report.refused(self, ba, config)):
            return False
        if not self._granular(ba, config, family):
            return False
        hooks = ba.arguments['hooks']
        axes = {axis for name, axes_of in (('prologue_operands', hooks.prologue_axes),
                                           ('epilogue_operands', hooks.epilogue_axes))
                for _, side_axes in zip(ba.arguments.get(name, ()), axes_of, strict=True)
                if len(held_axes(side_axes)) == 1 for axis in held_axes(side_axes)}
        # A vector the family reads from global memory instead (`copies_vectors`) loads in a role's row
        # tiles: its block along its axis whole roles (ns-regs: a 32-element depth vector on sm86 failed
        # to cast to the 64-row layout).
        extent, least = family.copy_extent, 1 if family.copies_vectors else family.role_rows
        return all((extent is None or self.extents(config)[axis] <= extent)
                   and self.extents(config)[axis] % least == 0 for axis in axes)

    def _ran(self, ba, config: mosaic_gpu.Tiles) -> tuple:
        """The call and tile a launch at `config` runs, which its legality, registers and launch read: the
        call itself by default."""
        return ba, config

    @abc.abstractmethod
    def _divides(self, ba, config: mosaic_gpu.Tiles) -> bool:
        """Whether `config`'s blocks divide the call's extents as the kernel requires."""

    @staticmethod
    def _hooks_lower(ba, dtype, target) -> bool:
        """Whether Mosaic GPU's lowering takes the launch's hooks (`_hooks_lower`). A side operand the
        launch squeezes and gives back with its dimension of one (`Hooks.prologue_broadcast`) has no
        layout Mosaic's inference finds for the size-one dimension, so a hook taking one does not lower
        there (GMMGATE-h100-8), where Triton's lowering takes it."""
        # Both bundles lower with warpgroup semantics, whose rules a hook's operations need.
        del target
        hooks = ba.arguments['hooks']
        if hooks.prologue_broadcast or hooks.epilogue_broadcast:
            return False
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

    @staticmethod
    @abc.abstractmethod
    def extents(tile: mosaic_gpu.Tiles) -> dict:
        """The block each launch axis takes at `tile`."""


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


@dataclasses.dataclass(frozen=True)
class MosaicGmm(_Mosaic):
    """`Gmm` through the Mosaic GPU kernels (`mosaic_gpu.gmm`), exact sums too: an FP8 product's running sum
    promoted into fp32 every step (`product.EIGHT`)."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens',)
    EXACT_SUMS: ClassVar[bool] = True
    FP8: ClassVar[bool] = True

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
            paired=ba.arguments.get('paired') is not None)

    def _generated_extents(self, ba) -> tuple[int, int | None, int]:
        tokens, kernel = ba.arguments['tokens'], ba.arguments['kernel']
        depth, width = (kernel.shape[2], kernel.shape[1]) if ba.arguments['trans_rhs'] else kernel.shape[1:]
        return -(-tokens.shape[0] // max(kernel.shape[0], 1)), depth, width

    def _divides(self, ba, config: mosaic_gpu.Tiles) -> bool:
        kernel = ba.arguments['kernel']
        depth, width = (kernel.shape[2], kernel.shape[1]) if ba.arguments['trans_rhs'] else kernel.shape[1:]
        return depth % config.block_k == 0 and width % config.block_n == 0

    @staticmethod
    def extents(tile: mosaic_gpu.Tiles) -> dict:
        return {'rows': tile.block_m, 'depth': tile.block_k, 'columns': tile.block_n}


    def derived(self, ba, target) -> mosaic_gpu.Tiles | None:
        # A dense product of fewer rows than a role's (a decode step's) runs swapped: its matrix takes the
        # MMA's rows and its rows the columns, whose unit is narrower, so its tile multiplies no padding rows
        # (CUTLASS's and cuBLAS's swap of A and B; megabake's 4 x 2048 x 2048 product went from 10.0 to
        # 4.7 us swapped, cuBLAS's time, row 1762). It is derived as the swapped call's own tile.
        swapped = self._swapped(ba, target)
        if swapped is None:
            return super().derived(ba, target)
        tile = self._derived(swapped, target, admits=lambda tile: self._left_fits(ba, tile))
        return None if tile is None else dataclasses.replace(tile, swapped=True)

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
        None where it runs as it is: a call with hooks, a paired, FP8 or grouped one, or one whose rows fill
        a role (`Family.role_rows`)."""
        families = self._families(ba, target)
        tokens, kernel = ba.arguments['tokens'], ba.arguments['kernel']
        if (not families or not ba.arguments.get('dense') or ba.arguments['hooks'] != NO_HOOKS
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
        target being selected for (`generation.gpu_target`)."""
        if not config.swapped:
            return ba, config
        swapped = self._swapped(ba, generation.gpu_target())
        assert swapped is not None, "a swapped tile is a swappable call's (`legal`)"
        return swapped, dataclasses.replace(config, swapped=False)

    def legal(self, ba, config: mosaic_gpu.Tiles, target) -> bool:
        if config.swapped and (self._swapped(ba, target) is None or not self._left_fits(ba, config)):
            return False
        return super().legal(ba, config, target)

    def _fwd(self, tokens: jax.Array, kernel: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), epilogue_operands: tuple = (), paired: jax.Array | None = None,
             *, out_dtype: str, trans_rhs: bool, interpret: bool, hooks: Hooks = NO_HOOKS,
             exact_sums: bool = False,
             dense: bool = False, config: mosaic_gpu.Tiles, return_residuals: bool = False):
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
        return mosaic_gpu.gmm(tokens, kernel, sizes, tiles=config, transpose_rhs=trans_rhs,
                              out_dtype=out_dtype, multiprocessors=self._launched(config, interpret),
                              interpret=INTERPRETED if interpret else False,
                              prologue_operands=prologue_operands, epilogue_operands=epilogue_operands,
                              dense=dense, paired=paired, exact_sums=exact_sums,
                              **hooks.kernel_options()), None


@dataclasses.dataclass(frozen=True)
class MosaicTgmm(_Mosaic):
    """`Tgmm` on Hopper (`mosaic_gpu.tgmm`): `block_m` x `block_n` tiles of each group's gradient, its rows
    `block_k` at a time."""

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
            out_dtype=ba.arguments['out_dtype'])

    def _generated_extents(self, ba) -> tuple[int, int | None, int]:
        # A gradient tile's rows are the tokens' width; its steps run over a group's rows, padded to whole
        # steps (`mosaic_gpu.tgmm`).
        return ba.arguments['tokens'].shape[1], None, ba.arguments['cotangent'].shape[1]

    def derived(self, ba, target) -> mosaic_gpu.Tiles | None:
        # The derived tile, its contraction split where its output tiles underfill a wave
        # (`mosaic_gpu.split_count`) at the tile the split's own call, writing fp32 partials, derives; an
        # exact sum's order is its contract, so it takes no split.
        tile = super().derived(ba, target)
        if (tile is None or ba.arguments['exact_sums'] or self._balances(ba, tile, target)
                or self._splits(ba, tile, target) == 1):
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
        takes no split beside: a dense one, not of exact sums, on a family that fixes a tile up across
        programs, its tail balanced."""
        family = mosaic_gpu.family_of(tile, ba.arguments['tokens'].dtype)
        if not (ba.arguments['dense'] and not ba.arguments['exact_sums'] and family.fixes_up):
            return False
        depth, _, width = self._generated_extents(ba)
        tiles = depth // tile.block_m * (width // tile.block_n)
        steps = -(-ba.arguments['tokens'].shape[0] // tile.block_k)
        return tail_of.balanced(tiles, steps, _multiprocessors(target),
                                handoff_steps=mosaic_gpu.handoff_steps(tile, target)) is not None

    def _splits(self, ba, tile: mosaic_gpu.Tiles, target) -> int:
        depth, _, width = self._generated_extents(ba)
        return mosaic_gpu.split_count(tile, groups=ba.arguments['sizes'].shape[0],
                                      rows=ba.arguments['tokens'].shape[0], depth=depth, width=width,
                                      multiprocessors=_multiprocessors(target),
                                      step_seconds=mosaic_gpu.step_seconds(tile, target),
                                      bandwidth=target.memory_rate, launch_seconds=target.launch_seconds)

    # A split tile runs the fp32 partials' call at its tile unsplit (`mosaic_gpu.tgmm`, `_ran`): its
    # legality, registers and launch are that call's; exact sums take no split.
    def legal(self, ba, config: mosaic_gpu.Tiles, target) -> bool:
        if config.splits > 1 and ba.arguments['exact_sums']:
            return False
        return super().legal(ba, config, target)

    def _partials(self, ba):
        """The call `ba` writing fp32 partials, as a split's slices do (`mosaic_gpu.tgmm`)."""
        return self.bind(*ba.args, **{**ba.kwargs, 'out_dtype': 'float32'})

    def _ran(self, ba, config: mosaic_gpu.Tiles) -> tuple:
        """The call and tile a launch at `config` runs: a split one's fp32 partials, unsplit."""
        if config.splits == 1:
            return ba, config
        return self._partials(ba), dataclasses.replace(config, splits=1)

    def _divides(self, ba, config: mosaic_gpu.Tiles) -> bool:
        rows, _, width = self._generated_extents(ba)
        return rows % config.block_m == 0 and width % config.block_n == 0

    def _stored_rows(self, ba, config: mosaic_gpu.Tiles) -> int:
        # Each consumer stores its depth half of a tile whole (`mosaic_gpu.tgmm`).
        del ba
        return config.block_m // config.warpgroups

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

    @staticmethod
    def extents(tile: mosaic_gpu.Tiles) -> dict:
        return {'rows': tile.block_k, 'depth': tile.block_m}


    def _fwd(self, tokens: jax.Array, cotangent: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), *, out_dtype: str, interpret: bool, hooks: Hooks = NO_HOOKS,
             exact_sums: bool = False, dense: bool = False, config: mosaic_gpu.Tiles,
             return_residuals: bool = False):
        return mosaic_gpu.tgmm(tokens, cotangent, sizes, tiles=config, out_dtype=out_dtype,
                               multiprocessors=self._programs(config),
                               interpret=INTERPRETED if interpret else False, prologue=hooks.before(),
                               prologue_operands=prologue_operands, prologue_axes=hooks.prologue_axes,
                               exact_sums=exact_sums, dense=dense), None


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


def _cutlass_call(target: str, out: jax.ShapeDtypeStruct) -> Callable:
    """dew_cutlass's FFI call `target` giving `out`, a mapped call running its maps in turn, one launch
    each (row 4389: JAX's FFI batching refuses a call that states no rule)."""
    return jax.ffi.ffi_call(target, out, vmap_method='sequential')


CUTLASS_TILE = mosaic_gpu.Tiles(128, 256, 64, 4, 8, warpgroups=2, cluster=2, family='wgmma')
"""The dew-cutlass wheel's one instantiated kernel, as Dew's records name a launch (row 4323): a fact of
that binary, not a derivation. kernels/cutlass/grouped_sm90.cu's `Kernel`, which every one of its targets
instantiates, is CUTLASS 3's pointer-array cooperative kernel (two consumer warpgroups) at a
`Shape<_128, _256, _64>` tile on a `Shape<_2, _1, _1>` cluster; its stages are CUTLASS's own, fitted to the
shared memory its epilogue leaves (`StageCountAutoCarveout`), and it has no band of column tiles, so the
record's `stages` and `grid_block_n` describe no choice of the binary's."""


class _Cutlass(_GroupedLaunch[mosaic_gpu.Tiles]):
    """CUTLASS's grouped GEMMs (`kernels/cutlass`, the dew-cutlass wheel) as XLA FFI calls: on Hopper,
    CUTLASS 3's pointer-array cooperative kernel, a 128 x 256 x 64 tile on a pair of multiprocessors, its
    stages the shared memory its epilogue leaves, bf16 operands and bf16 or fp32 out, each group's shape
    set up on the device. A vendored production route on the cells where it measured faster than the
    template (`DECLARED`), and by name elsewhere, at its one `TILE`."""
    name: ClassVar[str] = 'cutlass'
    config_cls: ClassVar[type[mosaic_gpu.Tiles]] = mosaic_gpu.Tiles
    TILE: ClassVar[mosaic_gpu.Tiles]

    def eligible(self, ba, target) -> bool:
        operands = [value for name, value in ba.arguments.items()
                    if name in ('tokens', 'kernel', 'cotangent')]
        # 16-byte aligned rows of every operand and output, as its TMA copies and each group's pointers
        # need: every extent a multiple of 8 elements.
        extents = {extent for operand in operands for extent in operand.shape[1:]}
        built = target.generation
        return (built in cutlass_architectures()
                and all(operand.dtype == jnp.bfloat16 for operand in operands)
                and jnp.dtype(ba.arguments['out_dtype']) in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32))
                and ba.arguments['sizes'].dtype == jnp.int32 and all(extent % 8 == 0 for extent in extents)
                and ba.arguments['hooks'] == NO_HOOKS and not ba.arguments.get('exact_sums')
                and not ba.arguments['interpret'])

    def space(self, ba) -> tuple[mosaic_gpu.Tiles, ...]:
        return (self.TILE,)

    def fitting(self, ba, target) -> tuple[mosaic_gpu.Tiles, ...]:
        # CUTLASS's builder fits the stages to the shared memory (`StageCountAutoCarveout`).
        return self.space(ba)

    def _get_heuristics_config(self, ba) -> mosaic_gpu.Tiles:
        return self.TILE


@dataclasses.dataclass(frozen=True)
class CutlassGmm(_Cutlass):
    """`Gmm` through CUTLASS (`dew_cutlass_grouped_gemm`)."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens',)

    TILE: ClassVar[mosaic_gpu.Tiles] = CUTLASS_TILE

    def _fwd(self, tokens: jax.Array, kernel: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), epilogue_operands: tuple = (), paired: jax.Array | None = None,
             *, out_dtype: str, trans_rhs: bool, interpret: bool, hooks: Hooks = NO_HOOKS,
             exact_sums: bool = False,
             dense: bool = False, config: mosaic_gpu.Tiles, return_residuals: bool = False):
        # No hooks, side operands, second matrix or exact sums: the launch is not eligible for them; a dense
        # call is one group, as any.
        del prologue_operands, epilogue_operands, paired, interpret, hooks, exact_sums, dense, config
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

    # Dew's record of a kernel gradient pairs no row tiles (`mosaic_gpu.Tiles.cluster`: `tgmm` takes no pair),
    # whatever cluster the binary's kernel runs on.
    TILE: ClassVar[mosaic_gpu.Tiles] = dataclasses.replace(CUTLASS_TILE, cluster=1)

    def _fwd(self, tokens: jax.Array, cotangent: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), *, out_dtype: str, interpret: bool, hooks: Hooks = NO_HOOKS,
             exact_sums: bool = False, dense: bool = False, config: mosaic_gpu.Tiles,
             return_residuals: bool = False):
        del prologue_operands, interpret, hooks, exact_sums, dense, config  # as `CutlassGmm._fwd`'s
        _cutlass_registered()
        out = jax.ShapeDtypeStruct((sizes.shape[0], tokens.shape[1], cotangent.shape[1]),
                                   jnp.dtype(out_dtype))
        return _cutlass_call('dew_cutlass_grouped_gemm_k', out)(tokens, cotangent, sizes), None


class _MosaicTpu(_GroupedLaunch[mosaic_tpu.Blocks]):
    """The TPU mainloops (`templates.gemm.mosaic_tpu`, Megablox's schedule): bf16 operands, which the
    MXUs multiply exactly and sum in fp32, widths a vreg's lanes divide, and a TPU whose scoped VMEM a
    measurement states; a CPU interprets them as the TPU runs them, and a TPU's own lowering runs them
    (`_on_platform`'s TPU branch), their native qualification open until a scoreboard row measures them."""
    name: ClassVar[str] = 'mosaic_tpu'
    config_cls: ClassVar[type[mosaic_tpu.Blocks]] = mosaic_tpu.Blocks
    GENERATES: ClassVar[bool] = True
    ROW_AXIS: ClassVar[int]
    """The block axis (`block_m`, `block_n`, `block_k`: 0, 1, 2) that walks the grouped rows."""

    def eligible(self, ba, target) -> bool:
        # Not for exact sums: the MXUs add within a step, in fp32 by Google's account, IEEE by none
        # (`GROUPED`'s 'step_sums' routes such a product to XLA).
        first, second = (value.dtype for name, value in ba.arguments.items()
                         if name in ('tokens', 'kernel', 'cotangent'))
        rules = mosaic_tpu_lowering.lowering_rules[tpu_core.CoreType.TC]
        return (generation.tpu(target) and target.scoped_vmem is not None
                and first == second == jnp.bfloat16
                and self.derived(ba, target) is not None and _hooks_lower(ba, first, rules)
                and not ba.arguments.get('exact_sums'))

    @abc.abstractmethod
    def spans(self, ba) -> tuple[int, int, int]:
        """The extents the `block_m`, `block_n` and `block_k` axes span at `ba`."""

    def derived(self, ba, target) -> mosaic_tpu.Blocks | None:
        """The one block the launch takes on `target` (docs/design/kernel-templates.md §7), from Mosaic
        TPU's facts and the call's extents alone: each axis the widest doubling of a vreg's lanes
        (`mosaic_tpu.LANES`) that divides its extent, the grouped rows' at most a group's rows doubled
        to cover them (a block a group boundary crosses runs once for each group), each at least what
        its one-dimensional side operands lay out at (`_lays_out`); then, until its buffers fit the
        scoped VMEM (`legal`), its largest axis halved, the step's before the output's, never below that
        least: the output block as square as the budget leaves it, which reads each operand fewest times.
        None where no block is legal."""
        spans, groups = self.spans(ba), ba.arguments['sizes'].shape[0]
        least = self._least_blocks(ba)
        blocks = []
        for axis, extent in enumerate(spans):
            if axis == self.ROW_AXIS:
                widest = max(_covering(-(-extent // max(groups, 1))), least[axis])
            else:
                dividing = [block for block in _doubling(mosaic_tpu.LANES, extent) if extent % block == 0]
                if not dividing:
                    return None
                widest = dividing[-1]
            blocks.append(widest)
        while not self.legal(ba, config := mosaic_tpu.Blocks(*blocks), target):
            halved = [axis for axis in (2, 0, 1) if blocks[axis] // 2 >= least[axis]]
            if not halved:
                return None
            # Half of a block that divides its extent divides it too.
            axis = max(halved, key=lambda axis: blocks[axis])
            blocks[axis] //= 2
        return config

    def neighbours(self, ba, config: mosaic_tpu.Blocks) -> tuple[mosaic_tpu.Blocks, ...]:
        # The derived blocks with one axis halved or doubled.
        return tuple(dataclasses.replace(config, **{name: value})
                     for name in ('block_m', 'block_n', 'block_k')
                     for value in (getattr(config, name) // 2, getattr(config, name) * 2) if value > 0)

    def _least_blocks(self, ba) -> list[int]:
        """The least block of each axis (`block_m`, `block_n`, `block_k`): a vreg's lanes, or the row of
        128 32-bit lanes a one-dimensional side operand along it lays out at short of its whole extent
        (`_lays_out`: 256 elements of a 16-bit one)."""
        least = [mosaic_tpu.LANES] * 3
        index = self.extents(mosaic_tpu.Blocks(0, 1, 2))
        hooks = ba.arguments['hooks']
        for name, axes_of in (('prologue_operands', hooks.prologue_axes),
                              ('epilogue_operands', hooks.epilogue_axes)):
            for side, axes in zip(ba.arguments.get(name, ()), axes_of, strict=True):
                if len(held_axes(axes)) == 1:
                    axis = index[held_axes(axes)[0]]
                    least[axis] = max(least[axis], 128 * 4 // jnp.dtype(side.dtype).itemsize)
        return least

    def legal(self, ba, config: mosaic_tpu.Blocks, target) -> bool:
        """Whether `config`'s blocks are whole vregs' lanes dividing their extents (the grouped rows' any),
        lay out every side operand (`_lays_out`) and, double-buffered with the fp32 accumulator, fit the
        TPU's scoped VMEM (`Template.legal`'s rule, VMEM for a block's shared memory)."""
        budget = target.scoped_vmem
        blocks = (config.block_m, config.block_n, config.block_k)
        return (budget is not None
                and all(block % mosaic_tpu.LANES == 0 and (axis == self.ROW_AXIS or extent % block == 0)
                        for axis, (extent, block) in enumerate(zip(self.spans(ba), blocks, strict=True)))
                and self._lays_out(ba, config) and self.staged_bytes(ba, config, target) <= budget)

    def fitting(self, ba, target) -> tuple[mosaic_tpu.Blocks, ...]:
        """The derived block where it is legal (`Template.fitting`), else raises with the TPU's scoped
        VMEM for the budget."""
        fits = self.fits(ba, target)
        if not fits:
            raise base.Unfitted(type(self).__name__, None, target.scoped_vmem or 0,
                                target.generation)
        return fits

    def staged_bytes(self, ba, config: mosaic_tpu.Blocks, target) -> int:
        # Pallas's pipeline holds two of each operand's and output's blocks (one in flight), beside the
        # fp32 accumulator; Mosaic TPU may hold each operand block's value and the product's fp32 tile as
        # temporaries beside them, at most (a v5e refused 20.33 and 20.95 MiB at a 1024 x 512 x 2048
        # product and kernel gradient, 16.12 at a 256 x 512 x 4096 gradient, whose buffers alone counted
        # 16.00, 16.00 and 13.0: the compile gate's libtpu runs).
        size = jnp.dtype(ba.arguments['tokens'].dtype).itemsize
        hooks = ba.arguments['hooks']
        extents = self.extents(config)
        outs = _output_bytes(ba)
        sides = (_side_tile_bytes(ba.arguments['prologue_operands'], hooks.prologue_axes, extents)
                 + _side_tile_bytes(ba.arguments.get('epilogue_operands', ()), hooks.epilogue_axes, extents))
        operands = self.loaded(config) * size + sides
        rows, columns = self.output(config)
        temporaries = operands + 4 * rows * columns
        return 2 * (operands + outs * rows * columns) + 4 * rows * columns + temporaries

    def _lays_out(self, ba, blocks: mosaic_tpu.Blocks) -> bool:
        """Whether Mosaic TPU lays out every side operand's block at `blocks`: a one-dimensional block
        (a power of two here) holds the operand's whole extent, the rows padded to their block, or at
        least a row of 128 lanes of 32-bit words (`_check_block_mappings` in jax's Pallas TPU lowering,
        256 of a 16-bit operand); a block of two dimensions or more here is of whole multiples of 128."""
        hooks = ba.arguments['hooks']
        extents = self.extents(blocks)
        for name, axes_of in (('prologue_operands', hooks.prologue_axes),
                              ('epilogue_operands', hooks.epilogue_axes)):
            for side, axes in zip(ba.arguments.get(name, ()), axes_of, strict=True):
                if len(held_axes(axes)) != 1:
                    continue
                axis, = held_axes(axes)
                extent = side.shape[axes.index(axis)]
                block = extents[axis]
                # A side operand by group lays out as its groups' vectors end to end (`mosaic_tpu._laid`).
                whole = (-(-extent // block) * block if axis == 'rows'
                         else math.prod(side.shape) if GROUPS in axes else extent)
                if block != whole and block < 128 * 4 // jnp.dtype(side.dtype).itemsize:
                    return False
        return True

    @staticmethod
    @abc.abstractmethod
    def extents(blocks: mosaic_tpu.Blocks) -> dict:
        """The block each launch axis takes at `blocks`."""

    @staticmethod
    @abc.abstractmethod
    def loaded(blocks: mosaic_tpu.Blocks) -> int:
        """The elements of both operand blocks one grid step loads."""

    @staticmethod
    @abc.abstractmethod
    def output(blocks: mosaic_tpu.Blocks) -> tuple[int, int]:
        """The rows and columns of an output block."""


@dataclasses.dataclass(frozen=True)
class MosaicTpuGmm(_MosaicTpu):
    """`Gmm` on a TPU (`mosaic_tpu.gmm`)."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens',)

    ROW_AXIS: ClassVar[int] = 0

    def spans(self, ba) -> tuple[int, int, int]:
        kernel = ba.arguments['kernel']
        depth, width = (kernel.shape[2], kernel.shape[1]) if ba.arguments['trans_rhs'] else kernel.shape[1:]
        return ba.arguments['tokens'].shape[0], width, depth

    @staticmethod
    def extents(blocks: mosaic_tpu.Blocks) -> dict:
        return {'rows': blocks.block_m, 'depth': blocks.block_k, 'columns': blocks.block_n}

    @staticmethod
    def loaded(blocks: mosaic_tpu.Blocks) -> int:
        return (blocks.block_m + blocks.block_n) * blocks.block_k

    @staticmethod
    def output(blocks: mosaic_tpu.Blocks) -> tuple[int, int]:
        return blocks.block_m, blocks.block_n


    def _fwd(self, tokens: jax.Array, kernel: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), epilogue_operands: tuple = (), paired: jax.Array | None = None,
             *, out_dtype: str, trans_rhs: bool, interpret: bool, hooks: Hooks = NO_HOOKS,
             exact_sums: bool = False,
             dense: bool = False, config: mosaic_tpu.Blocks, return_residuals: bool = False):
        # Never set: the launch is not eligible for exact sums or a second matrix; a dense call is one group,
        # as any.
        del exact_sums, dense, paired
        return mosaic_tpu.gmm(tokens, kernel, sizes, blocks=config, transpose_rhs=trans_rhs,
                              out_dtype=out_dtype, interpret=TPU_INTERPRETED if interpret else False,
                              prologue_operands=prologue_operands, epilogue_operands=epilogue_operands,
                              **hooks.kernel_options()), None


@dataclasses.dataclass(frozen=True)
class MosaicTpuTgmm(_MosaicTpu):
    """`Tgmm` on a TPU (`mosaic_tpu.tgmm`): `block_m` x `block_n` blocks of each group's gradient, its rows
    `block_k` at a time."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens', 'cotangent')

    ROW_AXIS: ClassVar[int] = 2

    def spans(self, ba) -> tuple[int, int, int]:
        tokens = ba.arguments['tokens']
        return tokens.shape[1], ba.arguments['cotangent'].shape[1], tokens.shape[0]

    @staticmethod
    def extents(blocks: mosaic_tpu.Blocks) -> dict:
        return {'rows': blocks.block_k, 'depth': blocks.block_m}

    @staticmethod
    def loaded(blocks: mosaic_tpu.Blocks) -> int:
        return blocks.block_k * (blocks.block_m + blocks.block_n)

    @staticmethod
    def output(blocks: mosaic_tpu.Blocks) -> tuple[int, int]:
        return blocks.block_m, blocks.block_n


    def _fwd(self, tokens: jax.Array, cotangent: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), *, out_dtype: str, interpret: bool, hooks: Hooks = NO_HOOKS,
             exact_sums: bool = False, dense: bool = False, config: mosaic_tpu.Blocks,
             return_residuals: bool = False):
        del exact_sums, dense  # never set (`eligible`); a dense call is one group, as any
        return mosaic_tpu.tgmm(tokens, cotangent, sizes, blocks=config, out_dtype=out_dtype,
                               interpret=TPU_INTERPRETED if interpret else False, prologue=hooks.before(),
                               prologue_operands=prologue_operands, prologue_axes=hooks.prologue_axes), None


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
                     f"{case.call.target.generation}", row='ledger 1295'), *measured))


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


IMPLEMENTATIONS = ((MosaicGmm(), MosaicTgmm()), (MosaicTpuGmm(), MosaicTpuTgmm()), (Gmm(), Tgmm()),
                   (CutlassGmm(), CutlassTgmm()))
"""Each implementation of the grouped product, in `GROUPED`'s order: its launch of a product and its launch
of a kernel gradient, one name for both."""

_GMMS = tuple(product for product, _ in IMPLEMENTATIONS)
_TGMMS = tuple(gradient for _, gradient in IMPLEMENTATIONS)

_NEEDS = {'cutlass': (DECLARED_ONLY,), 'triton': (DENSE_MEASURED,)}
"""Each old route's requirements beyond its launch's own: CUTLASS on its declared cells alone, Triton off the
dense products the template's launches cannot run."""

def _takes_pair(name: str, case: GroupedCase) -> bool:
    """Whether the launches `name` take the paired product `case` whole: Mosaic GPU's on a family whose
    register model counts its hooks (`mosaic.Family.counts_hooks`: warpgroup MMA), or on any family where
    its only hook is the epilogue over its two accumulators (`_pair_alone`: mma.sync's SwiGLU, 254 to 255
    registers and no spill at Qwen3-30B-A3B's and Qwen3.5-4B's gate|up on sm80, row 2100); every other
    hands it on (`APART`)."""
    family = bundles.of(case.call.target.generation, jnp.bfloat16)
    if name != MosaicGmm.name or family is None:
        return False
    bound = case.call.bound.get(name)
    return bound is not None and (family.counts_hooks or _pair_alone(bound[1], family))


def _pair_alone(ba, family: mosaic.Family) -> bool:
    """Whether `ba` is a paired product whose only hook is the epilogue over its two accumulators (no
    prologue, no side operand, no exact sums) on a family whose model counts it: one whose consumers issue
    no ring's loads (`Family.feeds_itself`: sm120's fed mma.sync spilled 208 and 280 bytes at the paired
    128 x 64 x 64 tile its model admitted, NS2100-1)."""
    hooks = ba.arguments['hooks']
    return (ba.arguments.get('paired') is not None and hooks.prologue is None and not family.feeds_itself
            and not ba.arguments.get('epilogue_operands') and not ba.arguments.get('exact_sums'))

class Apart:
    """`launched`'s answer for a paired product no launch takes whole: its two products run apart
    (`gemm._apart`), each through the launches that take it, as the unfused program runs
    them."""

    name = 'apart'


APART = Apart()

GROUPED: selection.Op[GroupedCase] = selection.Op('grouped_matmul', (
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
    kind = f"{'dense' if options['dense'] else 'grouped'} {made}{eight}"
    call = base.call(launches, *arguments, **options)
    cuda = generation.capability(call.target.generation)
    floor = cuda is not None and cuda[0] * 10 + cuda[1] >= generation.BF16_GPU
    placed = generation.tpu(call.target) or floor
    paired = any(bound.arguments.get('paired') is not None for _, bound in call.bound.values())
    return GroupedCase(placed, whole, product, arguments[0].shape[0], call, kind, paired=paired)


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
             paired: jax.Array | None = None, *, out_dtype, trans_rhs: bool, hooks: Hooks = NO_HOOKS,
             interpret: bool, promoted: bool = False, dense: bool = False) -> tuple:
    """The launches, arguments and options of each group's rows of `tokens` times its matrix of `kernel`,
    both of one dtype, with the template's hooks (`gemm.gemm`). `promoted`,
    the declared product's (`templates.product.Product`), adds each depth step's partial apart
    (`exact_sums`); `dense`, one group of every row (a dense product), the plain tiled kernel where Triton
    runs it; `paired`, a second matrix of `kernel`'s shape multiplied by the same rows (Mosaic GPU's
    launch alone takes it)."""
    arguments = (tokens, kernel, sizes, tuple(prologue_operands), tuple(epilogue_operands),
                 *(() if paired is None else (paired,)))
    options = {'out_dtype': jnp.dtype(out_dtype).name, 'trans_rhs': trans_rhs, 'hooks': hooks,
               'interpret': interpret, 'exact_sums': promoted, 'dense': dense}
    return _GMMS, arguments, options


def tgmm_call(tokens, cotangent, sizes, prologue_operands: tuple = (), *, out_dtype,
              hooks: Hooks = NO_HOOKS, interpret: bool, promoted: bool = False, dense: bool = False) -> tuple:
    """The launches, arguments and options of each group's rows of `tokens` transposed times its rows of
    `cotangent`, both of one dtype, with the template's prologue
    (`gemm.gemm_t`); `promoted` and `dense` as `gmm_call`'s."""
    arguments = (tokens, cotangent, sizes, tuple(prologue_operands))
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
    nor a dropped row's gradient carries them.
    """
    if any(operand.dtype in EIGHT for operand in (tokens, kernel)):
        # FP8 codes are each exact in bfloat16: the product is theirs there, summed in the preferred type
        # (the FP8 contract's reference arm, testing.md, FP8 products). XLA's GPU compiler rewrites no
        # ragged dot of widened FP8 codes into its narrower-summing FP8 GEMM (sm90's lowering of this
        # product holds no F8 GEMM, FS-1625-HLO, row 1625).
        tokens, kernel = tokens.astype(jnp.bfloat16), kernel.astype(jnp.bfloat16)
        precision = jax.lax.Precision.DEFAULT
    elif all(jnp.finfo(operand.dtype).bits == 16 for operand in (tokens, kernel)):
        precision = jax.lax.Precision.DEFAULT
    elif tokens.dtype != kernel.dtype:
        # A 16-bit operand beside a wider one would reach the TPU kernel
        # 16-bit at HIGHEST too; widened, it is the same values (exactly),
        # and the wider operand keeps the precision the call asked for.
        wide = jnp.promote_types(tokens.dtype, kernel.dtype)
        tokens, kernel = tokens.astype(wide), kernel.astype(wide)
    # bincount widens under x64; TPU ragged-dot only lowers int32 counts.
    # Keep the wider domain for CPU calls whose row count actually needs it.
    index_dtype = jnp.int32 if tokens.shape[0] <= np.iinfo(np.int32).max else jnp.int64
    if group_sizes.dtype == jnp.int64:
        group_sizes = group_sizes.astype(index_dtype)
    grouped = jnp.arange(tokens.shape[0], dtype=index_dtype)[:, None] < jnp.sum(
        group_sizes, dtype=index_dtype
    )
    out = jax.lax.ragged_dot(
        jnp.where(grouped, tokens, 0), kernel, group_sizes, precision=precision,
        preferred_element_type=preferred_element_type)
    return jnp.where(grouped, out, 0)


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
    # A symbolic zero tangent contributes no term. Only the others pass the
    # barrier: jax 0.11.2 cannot transpose one that holds a constant zero.
    live = [(name, tangent) for name, tangent in (('x', dx), ('kernel', dkernel))
            if not isinstance(tangent, SymbolicZero)]
    if not live:
        return output, SymbolicZero(jax.typeof(output))
    # The tangents keep their dtypes to this point whatever produced them, so
    # an activation's bf16 cotangent is not fused into a wider expression.
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


portable_projection = declared_tangent(Protected(), _portable_projection, _portable_projection_jvp,
                                       nondiff_argnums=(3, 4, 5), symbolic_zeros=True)


