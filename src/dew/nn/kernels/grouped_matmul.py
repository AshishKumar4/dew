"""The grouped matmul's kernel launches, which the GEMM template (`templates.gemm.template`) runs and
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
from jax.extend import core as jax_core

from ..precision import at_least_fp32, rounded_operand
from . import generation, selection, tuning
from .generation import INHERENT
from .templates import base, bundles, cost, mosaic, resources, tail as tail_of
from .templates.gemm import mosaic_gpu, mosaic_tpu
from .templates.product import Product


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
    two accumulators before its side tiles (`templates.gemm.template.gemm`'s `paired`)."""
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


def _equations(jaxpr):
    """Every equation of `jaxpr` and of the programs nested in it, at any depth."""
    yield from jaxpr.eqns
    for inner in jax_core.subjaxprs(jaxpr):
        yield from _equations(inner)


def _priced(hooks: Hooks) -> bool:
    """Whether the Mosaic launches' register model prices `hooks`, so a derivation decides their tiles as the
    GPU's compile does (the compile gate holds it to that): no prologue, an epilogue whose side operands are
    vectors (a bias's, a column or row scale's: `Family.column_registers`), one output (a paired product's
    epilogue's one). A prologue's rewrite (its nonlinear primitives' lowering workspace, the allocation
    cliff between stages), an epilogue's side tile and a second output the model does not yet price, off
    by 16 to 48 registers either way at the GPU's compile (FS-K01 model2, CAUSE-HOOK, CAUSE-SSA, K01-2):
    such a call takes another launch until its row prices it."""
    return (hooks.prologue is None and len(hooks.out_dtypes or ()) <= 1
            and all(len(axes) == 1 for axes in hooks.epilogue_axes))


def _lowers(hook, tiles: tuple, sides, rules) -> bool:
    """Whether a Pallas lowering's `rules` hold a rule for every primitive of `hook` on its `tiles` and its
    side tiles: Mosaic GPU's, at warpgroup semantics, have none for a transpose, which Triton's lowering
    takes."""
    if hook is None:
        return True
    return all(equation.primitive in rules
               for equation in _equations(jax.make_jaxpr(hook)(*tiles, *sides).jaxpr))


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
    """The bytes of one tile of each side operand at the block `extents` per launch axis."""
    return sum(math.prod(extents[axis] for axis in axes) * jnp.dtype(operand.dtype).itemsize
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
        # Every CUDA GPU the template sends the kernels to (`templates.gemm.template`), and
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
        held = accumulators * rows * columns // (32 * config.warps) < cost.THREAD_REGISTERS
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


@dataclasses.dataclass(frozen=True)
class Gmm(_Triton):
    """Each group's rows of `tokens` `[rows, in]` times its matrix of
    `kernel` `[groups, in, out]`, or of its transpose with `trans_rhs`."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens',)

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

    def eligible(self, ba, target) -> bool:
        first, second = (value.dtype for name, value in ba.arguments.items()
                         if name in ('tokens', 'kernel', 'cotangent'))
        # A hooked or exactly summed call runs where its bundle's register model counts what they hold.
        special = ba.arguments['hooks'] != NO_HOOKS or ba.arguments.get('exact_sums')
        family = self._family(ba, target)
        return (first == second and jnp.dtype(first).itemsize == 2
                and family is not None
                and (family.counts_hooks or not special)
                and self._hooks_lower(ba, first, target)
                and _priced(ba.arguments['hooks'])
                and (self.EXACT_SUMS or not ba.arguments.get('exact_sums'))
                and self.derived(ba, target) is not None)

    def registers(self, ba, config: mosaic_gpu.Tiles) -> int:
        # A consumer thread's live set (`resources.tile_registers`: the accumulator it carries, a step's
        # fragments, an output's copy), and beside the accumulator the larger of its phases, each done
        # before the next starts: the rows' stage a kernel gradient's consumer prepares itself, and the
        # epilogue's side values (`_epilogue_registers`); under exact sums the running sum beside each
        # step's own product. The hooks it prices are `_priced`'s. The compile gate tests the actual
        # launch and residency caps: ptxas's spill decisions are not monotone, so a least spill-free cap
        # cannot decide a launch.
        dtype = ba.arguments['tokens'].dtype
        family, block = mosaic_gpu.family_of(config, dtype), (config.block_m, config.block_n, config.block_k)
        hooks = ba.arguments['hooks']
        stored = hooks.out_dtypes or (ba.arguments['out_dtype'],)
        plain = resources.tile_registers(family, config.warpgroups, block, dtype, stored=stored)
        accumulator = -(-config.block_m * config.block_n // config.warpgroups // resources.THREADS)
        # A kernel gradient's consumer prepares its depth half of each step's rows stage in registers (the
        # neighbouring groups' rows zeroed: `mosaic_gpu._staged_halves`), apart from the products, which
        # read shared memory on such a family; mma.sync's products hold that stage in registers already.
        # sm90's kernel gradient needs 24 more than the forward at 128 x 192 and 128 x 256 (148 and 181,
        # DECIDE1).
        staged = 0
        if self.STAGED_ROWS and not family.operands_in_registers:
            words = -(-config.block_k * config.block_m // config.warpgroups * jnp.dtype(dtype).itemsize
                      // 4 // resources.THREADS)
            staged = max(0, words - family.held_registers)
        phases = [staged, self._epilogue_registers(ba, config)]
        # A paired product's second accumulator, carried beside the first (`mosaic_gpu.gmm`'s `paired`).
        plain += accumulator if ba.arguments.get('paired') is not None else 0
        if ba.arguments.get('exact_sums'):
            # The running sum, a second fp32 accumulator (`mosaic_gpu.tgmm`'s exact sums), is carried as a
            # loop value beside each step's scoped product, awaited before its add: the two accumulators and
            # the walk's control, no step's descriptors beside them (`resources.CONTROL_REGISTERS`), else the
            # staged rows above them (FS-K01 model2: 144 at 64 a thread over stages 2-5, 208 at 96).
            return max(2 * accumulator + resources.CONTROL_REGISTERS, plain + max(phases))
        return plain + max(phases)

    @staticmethod
    def _epilogue_registers(ba, config: mosaic_gpu.Tiles) -> int:
        """The epilogue's side values beyond the mainloop's held set, its side operands vectors
        (`_priced`)."""
        hooks = ba.arguments['hooks']
        if hooks.epilogue is None:
            return 0
        family = mosaic_gpu.family_of(config, ba.arguments['tokens'].dtype)
        # Column vectors beside the accumulator, in its column layout (`Family.column_registers`), and a
        # paired product's combination of its two accumulators, which holds a row of its results, one fp32
        # column vector's worth (its SwiGLU spilled 32 bytes at 128 x 192 at 232, PAIRED-9, and fit 224 at
        # 384 x 64, PAIRED-11): each with the walk's control alone (`resources.CONTROL_REGISTERS`), their
        # excess over the mainloop's held set.
        vectors = sum(family.column_registers(config.block_n, side.dtype)
                      for side, axes in zip(ba.arguments.get('epilogue_operands', ()), hooks.epilogue_axes,
                                            strict=True) if axes == ('columns',))
        vectors += family.column_registers(config.block_n, jnp.float32) if hooks.accumulators > 1 else 0
        return max(0, vectors + resources.CONTROL_REGISTERS - family.held_registers)

    def register_budget(self, config: mosaic_gpu.Tiles) -> int:
        # Its family's budget a consumer, whatever the operands' dtype (`mosaic.Family.registers`).
        return mosaic_gpu.family_of(config, jnp.bfloat16).registers(config.warpgroups)

    def _stored_rows(self, ba, config: mosaic_gpu.Tiles) -> int:
        """The multiple a consumer's stored rows of an output tile come in: a grouped call's ragged to a
        row, a dense call's whole `TMA_ROWS` (its rows padded to them, its tiles starting at whole ones)."""
        del config
        return mosaic_gpu.TMA_ROWS if ba.arguments.get('dense') else 1

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
        layouts = self._families(ba, target)
        if not layouts:
            return None
        rows, depth, width = self._generated_extents(ba)
        return mosaic_gpu.derived(layouts, rows=rows, depth=depth, width=width,
                                  dtype=ba.arguments['tokens'].dtype,
                                  multiprocessors=_multiprocessors(target), target=target,
                                  candidate=lambda tile: self.candidate(ba, tile, target),
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
        registers are a tile's alone."""
        if not self.candidate(ba, config, target):
            return False
        family = mosaic_gpu.family_of(config, ba.arguments['tokens'].dtype)
        launch = dataclasses.replace(self.launch_of(ba, config), floor=config.stages, most=config.stages)
        return resources.plan([launch], family, target, programs=config.programs) is not None

    def launch_of(self, ba, config: mosaic_gpu.Tiles) -> resources.Launch:
        """The launch the planner sizes at `config`'s shape (`resources.plan`): its allocations at a ring
        depth (`_allocations`), its consumers, its registers a consumer thread (`registers`), and at most
        the stages its program's share of the family's shared memory holds of a step's two operand blocks
        at an element a byte and its contraction has steps, at least two."""
        family = mosaic_gpu.family_of(config, ba.arguments['tokens'].dtype)
        depth = self._generated_extents(ba)[1]
        steps = None if depth is None else -(-depth // config.block_k)

        def scope(stages: int) -> resources.Scope:
            staged = dataclasses.replace(config, stages=stages)
            return self._allocations(ba, staged, family).scope(family, staged, steps)

        shared = resources.program_shared(family.most_shared, config.programs)
        most = shared // ((config.block_m + config.block_n) * config.block_k)
        if steps is not None:
            # A ring deeper than its contraction's steps holds no more of them.
            most = min(most, max(2, steps))
        return resources.Launch(scope, config.warpgroups, self.registers(ba, config), most=most, tile=config)

    def candidate(self, ba, config: mosaic_gpu.Tiles, target) -> bool:
        """Whether `config` is a tile of the call on `target` apart from its resources: of the launch's
        family there (a layout of it, `bundles.layouts`), in its granularity (whole roles of rows for each
        consumer, whole column units, one of
        its depths, at most its consumers, at most the programs its registers hold) with a ring of two
        stages at least (a walk keeping one step's products in flight holds its slot while the next step's
        loads fill another), its blocks dividing the call's extents, its cluster at most the family's
        (`mosaic.Family.cluster`), and a one-dimensional side operand's block, copied untiled, at most one
        copy's extent along its axis (`mosaic.Family.copy_extent`)."""
        family = next((bundle for bundle in self._families(ba, target) if bundle.name == config.family), None)
        if family is None or config.cluster > family.cluster or not self._divides(ba, config):
            return False
        dtype = ba.arguments['tokens'].dtype
        if (config.warpgroups > family.consumers or config.programs > family.programs(config.warpgroups)
                or config.stages < 2
                or config.block_m % (family.role_rows * config.warpgroups)
                or config.block_n % family.column_unit(dtype)
                or not family.stores_columns(config.block_n, self._stored_rows(ba, config))
                or config.block_k not in family.depths(dtype)):
            return False
        hooks = ba.arguments['hooks']
        axes = {axis for name, axes_of in (('prologue_operands', hooks.prologue_axes),
                                           ('epilogue_operands', hooks.epilogue_axes))
                for _, side_axes in zip(ba.arguments.get(name, ()), axes_of, strict=True)
                if len(side_axes) == 1 for axis in side_axes}
        # A vector the family reads from global memory instead (`copies_vectors`) loads in a role's row
        # tiles: its block along its axis whole roles (ns-regs: a 32-element depth vector on sm86 failed
        # to cast to the 64-row layout).
        extent, least = family.copy_extent, 1 if family.copies_vectors else family.role_rows
        return all((extent is None or self.extents(config)[axis] <= extent)
                   and self.extents(config)[axis] % least == 0 for axis in axes)

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
        # tile's barriers alone took it 128 bytes past). mma.sync's output staging is scoped beside its
        # pipeline's walk, so the two add (an L4 refused 114688 bytes for a 98304-byte pipeline and its
        # 16384-byte output, GEN4-gpu-l4).
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
    """`Gmm` through the Mosaic GPU kernels (`mosaic_gpu.gmm`)."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens',)

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


    def _fwd(self, tokens: jax.Array, kernel: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), epilogue_operands: tuple = (), paired: jax.Array | None = None,
             *, out_dtype: str, trans_rhs: bool, interpret: bool, hooks: Hooks = NO_HOOKS,
             exact_sums: bool = False,
             dense: bool = False, config: mosaic_gpu.Tiles, return_residuals: bool = False):
        # Exact sums are ineligible for this launch.
        del exact_sums
        return mosaic_gpu.gmm(tokens, kernel, sizes, tiles=config, transpose_rhs=trans_rhs,
                              out_dtype=out_dtype, multiprocessors=self._launched(config, interpret),
                              interpret=INTERPRETED if interpret else False,
                              prologue_operands=prologue_operands, epilogue_operands=epilogue_operands,
                              dense=dense, paired=paired, **hooks.kernel_options()), None


@dataclasses.dataclass(frozen=True)
class MosaicTgmm(_Mosaic):
    """`Tgmm` on Hopper (`mosaic_gpu.tgmm`): `block_m` x `block_n` tiles of each group's gradient, its rows
    `block_k` at a time."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens', 'cotangent')

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
        return dataclasses.replace(sliced, splits=splits)

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
        return tail_of.balanced(tiles, steps, _multiprocessors(target)) is not None

    def _splits(self, ba, tile: mosaic_gpu.Tiles, target) -> int:
        depth, _, width = self._generated_extents(ba)
        peak, multiprocessors = target.bf16_peak, _multiprocessors(target)
        flops = 2 * tile.block_m * tile.block_n * tile.block_k
        step = None if peak is None else flops / (peak / multiprocessors)
        return mosaic_gpu.split_count(tile, groups=ba.arguments['sizes'].shape[0],
                                      rows=ba.arguments['tokens'].shape[0], depth=depth, width=width,
                                      multiprocessors=multiprocessors, step_seconds=step,
                                      bandwidth=target.memory_bandwidth)

    # A split tile runs the fp32 partials' call at its tile unsplit (`mosaic_gpu.tgmm`): its legality,
    # registers and launch are that call's.
    def legal(self, ba, config: mosaic_gpu.Tiles, target) -> bool:
        if config.splits > 1 and ba.arguments['exact_sums']:
            return False
        ran, tile = self._ran(ba, config)
        return super().legal(ran, tile, target)

    def registers(self, ba, config: mosaic_gpu.Tiles) -> int:
        ran, tile = self._ran(ba, config)
        return super().registers(ran, tile)

    def launch_of(self, ba, config: mosaic_gpu.Tiles) -> resources.Launch:
        ran, tile = self._ran(ba, config)
        return dataclasses.replace(super().launch_of(ran, tile), tile=config)

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

    TILE: ClassVar[mosaic_gpu.Tiles] = mosaic_gpu.Tiles(128, 256, 64, 4, 8, warpgroups=2, cluster=2)

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
        return jax.ffi.ffi_call('dew_cutlass_grouped_gemm', out)(tokens, kernel, sizes,
                                                                 transpose_w=trans_rhs), None


@dataclasses.dataclass(frozen=True)
class CutlassTgmm(_Cutlass):
    """`Tgmm` through CUTLASS (`dew_cutlass_grouped_gemm_k`): each group's gradient a GEMM over its rows,
    an empty group's zero, its pair splitting the gradient's rows."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens', 'cotangent')

    TILE: ClassVar[mosaic_gpu.Tiles] = mosaic_gpu.Tiles(128, 256, 64, 4, 8, warpgroups=2)

    def _fwd(self, tokens: jax.Array, cotangent: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), *, out_dtype: str, interpret: bool, hooks: Hooks = NO_HOOKS,
             exact_sums: bool = False, dense: bool = False, config: mosaic_gpu.Tiles,
             return_residuals: bool = False):
        del prologue_operands, interpret, hooks, exact_sums, dense, config  # as `CutlassGmm._fwd`'s
        _cutlass_registered()
        out = jax.ShapeDtypeStruct((sizes.shape[0], tokens.shape[1], cotangent.shape[1]),
                                   jnp.dtype(out_dtype))
        return jax.ffi.ffi_call('dew_cutlass_grouped_gemm_k', out)(tokens, cotangent, sizes), None


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
                if len(axes) == 1:
                    axis = index[axes[0]]
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
                if len(axes) != 1:
                    continue
                block = extents[axes[0]]
                whole = -(-side.shape[0] // block) * block if axes[0] == 'rows' else side.shape[0]
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
    family is its cell (`DECLARED`)."""

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
        'pair', lambda case: not case.paired or _takes_pair(name, case.call.target),
        lambda case: f"the {name} launches take no paired product on {case.call.target.generation}, whose "
                     "two products then run apart", row='ledger 2100', kind='handoff'), _Need(
        # A paired call's capability is the pairing launch's alone: every other hands it on (`pair`).
        'launch', lambda case: ((case.paired and not _takes_pair(name, case.call.target))
                                or case.call.eligible(name)),
        lambda case: f"the {name} launches take no call of these dtypes, widths, side operands or hooks on "
                     f"{case.call.target.generation}", row='ledger 1295'), *measured))


DECLARED: dict[tuple[str | None, str], tuple[str, str]] = {
    ('mma.sync', 'grouped input_gradient'):
        ('triton', "FSCELLS1-a100 and FSCELLS2-a100 (base/kernels 9e1fd259c and 4c02f0c28, A100-SXM4-40GB, "
                   "the routed experts' products of Qwen3-30B-A3B, Qwen3-235B-A22B, DeepSeek-V3, Mixtral "
                   "8x7B and Kimi Linear, gate and down apart, groups balanced): the template 1.014x "
                   "Triton's time, geomean of 10 rows (worst 2.00x, Mixtral's down; best 0.59x, "
                   "Qwen3-30B-A3B's gate); its kernel gradient 0.880x and forward below 0.8x on those rows"),
    ('mma.sync', 'dense forward'):
        ('xla', "DENSE1 (A100, 2d1bbd051, one program a multiprocessor): Mosaic 1.923x and Triton 1.947x "
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
        ('xla', "DENSE1 (H100, 2d1bbd051): Mosaic 1.067x and Triton 1.616x XLA's dot (cuBLAS), geomean of "
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


_GMMS = (MosaicGmm(), MosaicTpuGmm(), Gmm(), CutlassGmm())
_TGMMS = (MosaicTgmm(), MosaicTpuTgmm(), Tgmm(), CutlassTgmm())

_NEEDS = {'cutlass': (DECLARED_ONLY,), 'triton': (DENSE_MEASURED,)}
"""Each old route's requirements beyond its launch's own: CUTLASS on its declared cells alone, Triton off the
dense products the template's launches cannot run."""

def _takes_pair(name: str, target) -> bool:
    """Whether the launches `name` take a paired product whole on `target`: Mosaic GPU's on a family whose
    register model counts the second accumulator (`mosaic.Family.counts_hooks`: warpgroup MMA); every other
    hands it on (`APART`)."""
    family = bundles.of(target.generation, jnp.bfloat16)
    return name == MosaicGmm.name and family is not None and family.counts_hooks

class Apart:
    """`launched`'s answer for a paired product no launch takes whole: its two products run apart
    (`templates.gemm.template._apart`), each through the launches that take it, as the unfused program runs
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
    kind = f"{'dense' if options['dense'] else 'grouped'} {made}"
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
    both of one dtype, with the template's hooks (`templates.gemm.template.gemm`). `promoted`,
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
    (`templates.gemm.template.gemm_t`); `promoted` and `dense` as `gmm_call`'s."""
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
    if all(jnp.finfo(operand.dtype).bits == 16 for operand in (tokens, kernel)):
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


@functools.partial(jax.custom_jvp, nondiff_argnums=(3, 4, 5))
def portable_projection(x: jax.Array, kernel: jax.Array, group_sizes: jax.Array,
                dtype: Dtype | None, precision: PrecisionLike, out_dtype: Dtype | None = None) -> jax.Array:
    """The grouped projection on `xla_ragged_dot`, in every differentiation mode: `x`'s rows `[rows, in]`
    through each group's matrix of `kernel` `[groups, in, out]`, both cast to `dtype` (flax's promotion
    where None), summed in at least fp32 and rounded once. Its tangent is `dx @ Q(kernel) + Q(x) @
    dkernel`, `Q` the rounded operands, summed over the widest of the compute, input and kernel dtypes,
    so an input gradient keeps its input's dtype and a kernel gradient its master's: the portable
    projection `dew.nn.moe.expert_projection`'s XLA route and the GEMM template's portable branch run.
    `out_dtype`, where given, is the dtype its sum rounds to in place of the products' (the template's
    fp32 accumulator, which its epilogue reads)."""
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


portable_projection.defjvp(_portable_projection_jvp, symbolic_zeros=True)


