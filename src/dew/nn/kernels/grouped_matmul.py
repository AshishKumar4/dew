"""The grouped matmul's kernel launches, which the GEMM template (`kernels.gemm`) runs and
differentiates (`dew.nn.moe.expert_projection` through it); XLA's portable products are
`kernels.xla_products`.

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
lowering, CUDA's or a TPU's (`xla_products._on_platform`), chosen where the call lowers
(`jax.lax.platform_dependent`), so a CPU computation on a GPU host lowers to `jax.lax.ragged_dot`; an
explicit request interprets them on the CPU, as the CPU suite checks them.

Each launch's config is the one it derives for the GPU from facts, the same on every run; nothing ranks
configs (bible §6). JAX's one tile for every shape left a wide expert's products at half an A100's cuBLAS
rate (a 4096 by 14336 expert's forward 10.4 ms where 5.3 ms is the derived tile's, the kernel gradient 18.3
against 8.4).
"""

import abc
import dataclasses
import enum
import functools
import importlib
import importlib.util
import math
from collections.abc import Callable, Hashable
from typing import ClassVar, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.pallas.mosaic import core as tpu_core, lowering as mosaic_tpu_lowering
from jax._src.pallas.mosaic_gpu import (
    core as plgpu_core,  # the lowering's semantics (`_lowers`)
    lowering as mosaic_gpu_lowering,
)
from jax.experimental.pallas import mosaic_gpu as plgpu

from ...algebra import bits
from ..derivatives import marked_call_p
from ..programs import equations
from . import generation, selection, tuning
from .generation import INHERENT
from .templates import across, base, bundles, mosaic, report, resources, swizzling
from .templates.axes import held as held_axes
from .templates.gemm import mosaic_gpu, packed, pricing
from .templates.gemm.hooks import NO_HOOKS, Hooks
from .templates.product import SCALE_DEPTH, Product, eight


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
    PRODUCER_GATHERS: ClassVar[bool] = False
    """Whether its producer gathers the rows its table names (`gmm`'s `gather`); a kernel gradient's tables
    are its consumers' copies (`mosaic_gpu.tgmm`'s `gather`)."""
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
        family = pricing.family_of(config, dtype)
        # The tile's columns its accumulators hold: a pair's both where its products span it.
        columns = pricing.tile_columns(config, family)
        block = (config.block_m, columns, config.block_k)
        if not family.budgets_registers or generation.tpu(generation.gpu_target()):
            # A TPU core states none, whatever tile a pin names: its values are Mosaic TPU's, in VMEM
            # (`mosaic.Issuer.value_scratch`).
            return None
        hooks = ba.arguments['hooks']
        stored = hooks.out_dtypes or (ba.arguments['out_dtype'],)
        plain = resources.tile_registers(family, config.sharing, block, dtype, stored=stored,
                                        rhs_transposed=self._matrix_transposed(ba),
                                        lhs_retained=ba.arguments.get('paired') is not None)
        # A consumer's accumulators: its share of the tile's rows (the whole tile's under ping-pong).
        accumulator = -(-config.share * columns // resources.THREADS)
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
                  if ba.arguments.get('dense') and pricing.fixes_up(config, dtype) else 0)
        phases = [staged, prologue, merged, self._epilogue_registers(ba, config, stored)]
        # A paired product's second accumulator, carried beside the first (`mosaic_gpu.gmm`'s `paired`): in
        # registers its whole share, in tensor memory the epilogue chunk read beside the first's
        # (`Issuer.chunk_columns`).
        held_second = -(-config.share * family.chunk_columns(config.block_n) // resources.THREADS)
        second = held_second if ba.arguments.get('paired') is not None else 0
        plain += second
        if ba.arguments.get('exact_sums') and family.sums_in_persist:
            # A family promoting each window's partials into running sums in its accumulators' own home
            # (tcgen05's tensor memory, `persist`'s `summed`) holds a chunk of a partial and the running sum's
            # beside it a promote at a time (`_promoted`, each product's in turn), not two register
            # accumulators: a 128 x 128 tile 2 x 64 words and the walk's control (B6, where two whole
            # accumulators charged 272 refused it past 248).
            chunk = -(-config.share * family.chunk_columns(columns) // resources.THREADS)
            return max(2 * chunk + resources.CONTROL_REGISTERS, plain + max(phases))
        if ba.arguments.get('exact_sums'):
            # Each product's running sum, a second fp32 accumulator (`mosaic_gpu.tgmm`'s exact sums), is
            # carried as a loop value beside its step's product (`mosaic.Issuer.summed_apart`), awaited before
            # its add: a pair's two running sums beside its two steps' products (the gathered cells' paired
            # product into fp32 spilled 324 bytes on a GH200 where one product's two were counted, M-EM-19c),
            # and the walk's control, no step's descriptors beside them (`resources.CONTROL_REGISTERS`), else
            # the staged rows above them (FS-K01 model2: 144 at 64 a thread over stages 2-5, 208 at 96). A
            # family whose products read registers holds each step's fragments beside both (mma.sync's, one
            # step's product at a time: 201 at its 128 x 128 forward tile, NSDBG-5).
            held = plain + (accumulator if family.operands_in_registers else 0)
            return max(2 * (accumulator + second) + resources.CONTROL_REGISTERS, held + max(phases))
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
        share = config.share * config.block_k
        stored = jnp.dtype(ba.arguments['tokens'].dtype).itemsize
        family = pricing.family_of(config, ba.arguments['tokens'].dtype)
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
        family = pricing.family_of(config, ba.arguments['tokens'].dtype)
        share, columns = config.share, pricing.tile_columns(config, family)
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
        accumulator = -(-config.share * family.chunk_columns(columns) // resources.THREADS)
        tiles = sum(len(held_axes(axes)) > 1 for axes in hooks.epilogue_axes) * accumulator
        held = sum(-(-accumulator * jnp.dtype(dtype).itemsize // 4)
                   for dtype in (hooks.out_dtypes or ())[:-1])
        stage = max(staging, resources.CONTROL_REGISTERS) + vectors
        return max(0, stage - family.held_registers) + tiles + held

    def register_budget(self, config: mosaic_gpu.Tiles) -> int:
        # Its family's budget a consumer, whatever the operands' dtype (`mosaic.Family.registers`); none on a
        # core that budgets no registers.
        family = pricing.family_of(config, jnp.bfloat16)
        return family.registers(config.warpgroups) if family.budgets_registers else 0

    def _granular(self, ba, config: mosaic_gpu.Tiles, family: mosaic.Family) -> bool:
        """Whether `config` is in `family`'s granularity (`candidate`): at most its consumers, at most the
        programs its registers hold, a ring of two stages at least, whole roles of rows for each consumer
        that shares a tile's rows (`Tiles.sharing`: one under ping-pong, whose consumers take whole tiles),
        whole column units, and one of its depths."""
        dtype = ba.arguments['tokens'].dtype
        return not (config.warpgroups > min(family.consumers, mosaic_gpu.CONSUMERS)
                    or config.programs > family.programs(config.warpgroups) or config.stages < 2
                    or config.block_m % (family.role_rows * config.sharing)
                    or config.block_n % family.column_unit(dtype)
                    or config.block_k not in (self.STEPS or family.depths(dtype)))

    def _matrix_transposed(self, ba) -> bool:
        """Whether the matrix's blocks lie transposed, `[width, depth]` (`trans_rhs`), as its stream reads
        them."""
        return bool(ba.arguments.get('trans_rhs'))

    def _fp8(self, ba, dtype, family: mosaic.Family) -> bool:
        """Whether an FP8 call (`product.eight`) runs here: a forward product on warpgroup MMA, its matrix
        streamed K-major (`_streamed_matrix`: as Hopper's FP8 tensor cores read both operands), with no hook,
        its running sum promoted every step (`exact_sums`)."""
        return (jnp.dtype(dtype) in family.fp8_formats and self.FP8
                and ba.arguments['hooks'] == NO_HOOKS and ba.arguments.get('paired') is None
                and bool(ba.arguments.get('exact_sums')))

    def _families(self, ba, target) -> tuple[mosaic.Family, ...]:
        """The bundles the launch may run on `target`, in preference order: the most capable issuer's whose
        bundle takes the call (`bundles.groups`, its kernels stating their tiles' accumulators,
        `mosaic_gpu.gmm`'s and `tgmm`'s; `_takes`), so a call tcgen05's bundle refuses runs on mma.sync's (row
        4727), a TPU core's its MXU's (`mosaic_tpu.MXU`); of that issuer's, each bundle that takes the call (a
        tcgen05 pair's, which gathers and scatters no rows, `Transport.gathers_rows`, not for a routed call);
        none where no issuer's does (a CPU interprets a GPU's or a TPU's only where `generation.assumed` names
        it)."""
        return next(iter(self._groups(ba, target)), ())

    def _groups(self, ba, target) -> tuple[tuple[mosaic.Family, ...], ...]:
        """Each issuer's bundles that take the call (`_takes`), in `bundles.groups`' preference order, a group
        whose issuer's bundle takes it: the derivation's, each group's tiles tried before the next's, so a
        call the most capable issuer takes but derives no tile of (a B200's paired product of three outputs,
        whose register model holds none on tcgen05) runs on the next issuer's (mma.sync's)."""
        found = bundles.groups(target.generation, ba.arguments['tokens'].dtype, stated=True,
                               tpu=generation.tpu(target))
        return tuple(tuple(family for family in group if self._takes(ba, family))
                     for group in found if self._takes(ba, group[0]))

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
        takes where given (a swapped call's own rule, `MosaicGmm._left_fits`), on the call padded to whole
        tiles (`_aligned`'s `whole_tiles`): the tile a launch runs whether its depth and width are padded or
        run partial (`_partial_axes`)."""
        ba = self._aligned(ba, whole_tiles=True)
        rows, depth, width = self._generated_extents(ba)
        # Each issuer's group in turn (`_groups`), a pair's layouts first where the rows hold its tile
        # (`mosaic_gpu.paired_first`), each its own derivation, the first that holds a tile taken.
        for tried in (tried for layouts in self._groups(ba, target)
                      for tried in mosaic_gpu.paired_first(layouts, rows)):
            tile = mosaic_gpu.derived(tried, rows=rows, depth=depth, width=width,
                                      dtype=ba.arguments['tokens'].dtype,
                                      multiprocessors=_multiprocessors(target), target=target,
                                      candidate=lambda tile: (self.candidate(ba, tile, target)
                                                              and (admits is None or admits(tile))),
                                      launch_of=lambda tile: self.launch_of(ba, tile),
                                      grouped=depth is not None, depths=self.STEPS,
                                      ragged=self._ragged(ba, tried[0]),
                                      units=self._vector_units(ba, tried[0]))
            if tile is not None:
                return tile
        return None

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
        family = pricing.family_of(config, ba.arguments['tokens'].dtype)
        unit = family.column_unit(ba.arguments['tokens'].dtype)
        role = family.role_rows * config.sharing
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
        family = pricing.family_of(config, ba.arguments['tokens'].dtype)
        launch = dataclasses.replace(self.launch_of(ba, config), floor=config.stages, most=config.stages)
        return resources.plan([launch], family, target, programs=config.programs) is not None

    def _stated(self, ba, config: mosaic_gpu.Tiles) -> dict:
        """The register model a launch at `config` states its mainloop: `registers`, `shared` bytes staged."""
        registers = self.registers(ba, config)
        return {'registers': registers,
                'shared': registers and self.staged_bytes(ba, config, generation.gpu_target())}

    def launch_of(self, ba, config: mosaic_gpu.Tiles) -> resources.Launch:
        """The launch the planner sizes at `config`'s shape (`resources.plan`): its allocations at a ring
        depth (`_allocations`), its consumers, its registers a consumer thread (`registers`), and at most
        the stages its program's share of the GPU's block (`generation.shared_budget`) holds of a step's
        two operand blocks at an element a byte and its contraction has steps, at least two: the call's a
        launch at `config` runs (`_ran`)."""
        ba, config = self._ran(ba, config)
        family = pricing.family_of(config, ba.arguments['tokens'].dtype)
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
        # A producer whose issue loop holds more than its family's least states its own (a tcgen05 load warp
        # gathering rows, `mosaic_gpu.gmm_producer`), as its kernel lowers it.
        gathers = self.PRODUCER_GATHERS and ba.arguments.get('gather') is not None
        producer = mosaic_gpu.gmm_producer(family, gathers)
        return resources.Launch(scope, roles, self.registers(ba, config), most=most, producer=producer)

    def rewritten(self, ba, family: mosaic.Family) -> bool:
        """Whether the launch's staged rows are rewritten by `family`'s own role (`rewrites_in_persist`): its
        own loads' rows, never rows its tables gather (`mosaic_gpu.tgmm`'s `gather`, which a consumer's
        cp.async gather stages: the fed rings'), nor rows read as loaded (`unstaged`)."""
        tables = (ba.arguments.get('gather'), ba.arguments.get('scatter'))
        return self.STAGED_ROWS and mosaic_gpu.rewrites(family, self.K_MAJOR, tables, self.unstaged(ba))

    def unstaged(self, ba) -> bool:
        """Whether a kernel gradient's rows stage is read as loaded (`mosaic_gpu.rows_unstaged`): a dense call
        (or its split's runs, `whole_steps`) of no prologue or table, laid rows-major, which no consumer or
        role prepares."""
        tabled = ba.arguments.get('gather') is not None or ba.arguments.get('scatter') is not None
        whole = bool(ba.arguments.get('dense')) or bool(ba.arguments.get('whole_steps'))
        return (self.STAGED_ROWS and not self.K_MAJOR
                and mosaic_gpu.rows_unstaged(dense=whole,
                                             prologue=ba.arguments['hooks'].prologue, tabled=tabled))

    def candidate(self, ba, config: mosaic_gpu.Tiles, target) -> bool:
        """Whether `config` is a tile of the call on `target` apart from its resources: of the launch's
        family there (a layout of it, `bundles.layouts`), in its granularity (whole roles of rows for each
        consumer, exactly the family's accumulator rows where it fixes them, whole column units, one of
        its depths, at most its consumers, at most the programs its registers hold, a compiled pair's tile on
        a family whose products span one, `mosaic_gpu.pair_takes`, none interpreted), its accumulators and
        its streams' tensor memory held in the tensor memory a family keeps them in
        (`mosaic.Issuer.tensor_columns`, `stream_tensor_columns`), with a ring of two
        stages at least (a walk keeping one step's products in flight holds its slot while the next step's
        loads fill another), its blocks dividing the call's extents, its cluster at most the family's
        (`mosaic.Family.cluster`), and a one-dimensional side operand's block, copied untiled, at most one
        copy's extent along its axis (`mosaic.Family.copy_extent`); none a compile of this process found
        spilling (`report.refused`), so the derivation takes the next tile by its own rule."""
        family = next((bundle for group in self._groups(ba, target) for bundle in group
                       if bundle.name == config.family), None)
        if (family is None or config.cluster > family.cluster or not mosaic_gpu.pair_takes(config, family)
                or (family.pairs_product and ba.arguments['interpret'])
                or self._pair_refuses(ba, config, family) or self._pingpong_refuses(config, family)
                or not self._divides(ba, config) or report.refused(self, ba, config)):
            return False
        # A family fixing its accumulator's rows takes a tile of exactly them a consumer, and one holding its
        # accumulators in tensor memory a tile whose accumulators it holds (a paired product's two).
        # Its tensor memory is every allocation live in the kernel's one scope: the accumulators and its
        # streams' (an MX product's block-scale rings, B7: a 128 x 256 MXFP8 tile's 512 columns of
        # accumulators passed beside its rings' 24 more, which the allocator refused).
        accumulators = 1 if ba.arguments.get('paired') is None else 2
        tensor = family.tensor_columns(pricing.tile_columns(config, family), accumulators,
                                       bool(ba.arguments.get('exact_sums')))
        if tensor:
            blocks = self._allocations(ba, config, family).blocks
            tensor += family.stream_tensor_columns(blocks, config.stages)
        if family.accumulator_rows not in (None, config.share) or tensor > (target.tmem_columns or 0):
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

    @property
    def _scaled(self) -> bool:
        """Whether the launch's call carries scales beside its matrix (a packed matrix's, step or block
        scales), its layout its own: none of `_Mosaic`'s."""
        return False

    def _pingpong_refuses(self, config: mosaic_gpu.Tiles, family: mosaic.Family) -> bool:
        """Whether a ping-pong tile (`Tiles.pingpong`) is one the launch does not run: on a family that takes
        none (`Family.takes_pingpong`: a producer of its own loads every tile), a kernel gradient's (its
        consumers prepare their halves of each step's rows stage, `STAGED_ROWS`), a scaled or swapped call
        (a packed decode's buffers and a swapped tile's rows are each consumer's half)."""
        return config.pingpong and (not family.takes_pingpong or self.STAGED_ROWS or self._scaled
                                    or config.swapped)

    def _pair_refuses(self, ba, config: mosaic_gpu.Tiles, family: mosaic.Family) -> bool:
        """Whether a pair's product refuses the call at `config` (`mosaic_gpu.pair_refuses`, the statement
        `gmm`, `tgmm` refuse by): its prologue, scales, gathered rows, exact sums, rewritten stage."""
        return mosaic_gpu.pair_refuses(config, family, transpose_lhs=False,
                                       exact_sums=bool(ba.arguments.get('exact_sums')), scaled=self._scaled,
                                       parts=(ba.arguments['hooks'].prologue, ba.arguments.get('gather')),
                                       rewritten=self.rewritten(ba, family))

    def _ran(self, ba, config: mosaic_gpu.Tiles) -> tuple:
        """The call and tile a launch at `config` runs, which its legality, registers and launch read: the
        call as it runs it (`_aligned`) by default."""
        return self._aligned(ba), config

    def _aligned(self, ba, *, whole_tiles: bool = False):
        """The call as the launch runs it (`_fwd`'s `_padded`): its depth and width each padded with zeros to
        whole units of its family's tiles where they are not (its depth to the least step,
        `Family.depths`, or a tile's rows where the depth is a tile's; its width to whole column units,
        `Family.column_unit`), the output sliced back; the call itself where they are. A width of 100
        columns, which no tile divides, so runs on 128, the template's where Triton's masked tiles ran it.
        An axis the launch runs with its last step or column tile partial (`_partial_axes`: TMA's bounded
        loads, row 5299) is not padded, so no copy is made of its operands; `whole_tiles`, every axis padded,
        the call the tile is derived on (`_derived`)."""
        layouts = self._families(ba, generation.gpu_target())
        if not layouts:
            return ba
        family = layouts[0]
        extents, whole = self._whole_extents(ba, family)
        if whole == extents:
            return ba
        depth_partial, width_partial = (False, False) if whole_tiles else self._partial_axes(ba, family)
        partial = {'depth': depth_partial, 'width': width_partial}
        grown = {axis: extent if partial.get(axis) else whole[axis] for axis, extent in extents.items()}
        if grown == extents:
            return ba
        padded = {name: _grown(value, dims, grown) for name, value, dims in self._padded_dims(ba)}
        return type(ba)(ba.op, {**ba.arguments, **padded})

    def _whole_extents(self, ba, family: mosaic.Family) -> tuple[dict[str, int], dict[str, int]]:
        """The call's depth and width (`_padded_extents`) and each in whole units of `family`'s tiles, as
        `_aligned` pads them: its depth to the least step (`_depth_unit`), its width to whole column units
        and whole units of the call's column vectors (`_vector_units`: an mma.sync bias's 64, where its
        column unit is 32; a 200-wide input gradient padded to 224 took no tile, EM-unal2)."""
        dtype = ba.arguments['tokens'].dtype
        width_unit = math.lcm(family.column_unit(dtype), self._vector_units(ba, family)[1])
        units = {'depth': self._depth_unit(ba, family, dtype), 'width': width_unit}
        extents = self._padded_extents(ba)
        return extents, {axis: -(-extent // units[axis]) * units[axis] for axis, extent in extents.items()}

    def _partial_axes(self, ba, family: mosaic.Family) -> tuple[bool, bool]:
        """Whether the call's depth, and its width, run their last step and column tile partial on `family`
        rather than padded (`_aligned`): a product's alone (`MosaicGmm._partial_axes`)."""
        del ba, family
        return False, False

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
        family = pricing.family_of(config, ba.arguments['tokens'].dtype)
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
    exact sums too: an FP8 product's running sum promoted into fp32 every step (`product.eight`)."""

    PRODUCER_GATHERS: ClassVar[bool] = True
    ROWS: ClassVar[tuple[str, ...]] = ('tokens',)
    EXACT_SUMS: ClassVar[bool] = True
    FP8: ClassVar[bool] = True
    STEP_SCALED: ClassVar[bool] = False
    """Whether its kernel streams a step's rows' scales beside its ring (`MosaicStepScaledGmm`)."""
    BLOCK_SCALED: ClassVar[bool] = False
    """Whether its kernel streams an MX product's block scales beside its rings (`MosaicMxGmm`)."""

    @property
    def _scaled(self) -> bool:
        # A packed matrix's words (`STEPS`), a step's row scales or an MX product's block scales.
        return self.STEPS is not None or self.STEP_SCALED or self.BLOCK_SCALED

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
            block_scaled=self.BLOCK_SCALED, gathered=ba.arguments.get('gather') is not None,
            interpret=bool(ba.arguments['interpret']))

    def _generated_extents(self, ba) -> tuple[int, int | None, int]:
        # A group's rows on average over the call's groups, which a stack read from its `first` matrix
        # holds fewer of than matrices.
        tokens, kernel = ba.arguments['tokens'], ba.arguments['kernel']
        depth, width = (kernel.shape[2], kernel.shape[1]) if ba.arguments['trans_rhs'] else kernel.shape[1:]
        return -(-tokens.shape[0] // max(ba.arguments['sizes'].shape[0], 1)), depth, width

    def _divides(self, ba, config: mosaic_gpu.Tiles) -> bool:
        # A depth or width the tile does not divide where its last step or column tile runs partial
        # (`_partial_axes`, at the tile `mosaic_gpu.partial_runs`), no deeper or wider than the call.
        kernel, dtype = ba.arguments['kernel'], ba.arguments['tokens'].dtype
        depth, width = (kernel.shape[2], kernel.shape[1]) if ba.arguments['trans_rhs'] else kernel.shape[1:]
        family = pricing.family_of(config, dtype)
        columns = pricing.tile_columns(config, family)
        runs = mosaic_gpu.partial_runs(config, dtype, ba.arguments['interpret'],
                                       transpose_rhs=ba.arguments['trans_rhs'],
                                       gathered=ba.arguments.get('gather') is not None,
                                       scattered=ba.arguments.get('scatter') is not None)
        depth_partial, width_partial = (each and run for each, run in
                                        zip(self._partial_axes(ba, family), runs, strict=True))
        return ((depth % config.block_k == 0 or (depth_partial and config.block_k <= depth))
                and (width % columns == 0 or (width_partial and columns <= width)))

    def _partial_axes(self, ba, family: mosaic.Family) -> tuple[bool, bool]:
        """`_Mosaic._partial_axes`: each axis the call does not fill with whole tiles that the launch may run
        partial (`_partial_admitted`) where the tile it derives on the padded call (`_derived`, read once
        through `derived`) runs it partial as it is (`mosaic_gpu.partial_runs`): the same tile, its operands
        and output uncopied, so it runs no slower than padded. A partial axis that would take a narrower tile
        is padded instead: a GH200's grouped 4096 x 200 x 136 product ran its partial tiles one swizzled tile
        deep and wide 1.07x and its input gradient 1.20x their padded time, where 8192 x 1000 x 1400's ran
        0.86x and 0.93x (M-EM-22b)."""
        extents, whole = self._whole_extents(ba, family)
        short = (extents.get('depth') != whole.get('depth'), extents.get('width') != whole.get('width'))
        admitted = tuple(each and grows for each, grows in zip(self._partial_admitted(ba, family), short,
                                                                strict=True))
        if not any(admitted):
            return False, False
        tile = self.derived(self._aligned(ba, whole_tiles=True), generation.gpu_target())
        if tile is None:
            return False, False
        dtype = ba.arguments['tokens'].dtype
        runs = mosaic_gpu.partial_runs(tile, dtype, ba.arguments['interpret'],
                                       transpose_rhs=ba.arguments['trans_rhs'],
                                       gathered=ba.arguments.get('gather') is not None,
                                       scattered=ba.arguments.get('scatter') is not None)
        # The tile no deeper and no wider than the call (`mosaic_gpu.gmm`'s refusal): a last tile partial,
        # never a first.
        within = (tile.block_k <= extents['depth'],
                  pricing.tile_columns(tile, pricing.family_of(tile, dtype)) <= extents['width'])
        depth_partial, width_partial = (each and run and fits for each, run, fits
                                        in zip(admitted, runs, within, strict=True))
        return depth_partial, width_partial

    def _partial_admitted(self, ba, family: mosaic.Family) -> tuple[bool, bool]:
        """Whether the launch may run the call's depth, and its width, partial: a plain matrix's product (no
        packed, step- or block-scaled one) on a
        family whose loads past an operand are bounded (`Family.bounded_loads`, TMA's) and whose products
        read its blocks by descriptor (`mosaic_gpu.partial_runs`), compiled, not swapped
        (`_swapped`), each axis at least a swizzled tile long where every operand and output along it lies in
        rows of whole 16 bytes, as a TMA descriptor's strides must (a 100-wide bf16 matrix's 200-byte rows are
        padded: "Async copies require all GMEM strides except the last one to be a multiple of 16 bytes"), and
        no side tile runs along it (a side's copy spans its own dtype's tiles, which the fork's untiled view
        of a partial extent takes one of: a float32 side's 64 columns are two); the depth's where the rows are
        not gathered and the width's where they are not scattered (a gather's and a scatter's piecewise copies
        are unbounded)."""
        if (self._scaled or not family.bounded_loads
                or family.operands_in_registers or ba.arguments['interpret']
                or self._swapped(ba, generation.gpu_target()) is not None):
            return False, False
        tokens, kernel, hooks = ba.arguments['tokens'], ba.arguments['kernel'], ba.arguments['hooks']
        depth, width = (kernel.shape[2], kernel.shape[1]) if ba.arguments['trans_rhs'] else kernel.shape[1:]
        # The matrix as its stream reads it (`_matrix_transposed`): its rows run along the depth or the width.
        trans_rhs = self._matrix_transposed(ba)

        def rows_of(extent: int, *dtypes) -> bool:
            # A call no narrower than a swizzled tile (`mosaic_gpu.partial_step`), which its last tile
            # spans partial; a narrower one is padded to its tile, as a whole tile's rows are.
            return (extent >= mosaic_gpu.partial_step(tokens.dtype)
                    and all(extent * jnp.dtype(dtype).itemsize % 16 == 0 for dtype in dtypes))

        def tiled(operands: tuple, axes_of: tuple, axis: str) -> bool:
            # A side tile along the axis (a vector's copy is one tile of its own, its load past it zeros).
            return any(len(axes) > 1 and axis in axes for axes in axes_of[:len(operands)])

        outputs = hooks.out_dtypes or (ba.arguments['out_dtype'],)
        depth_partial = (rows_of(depth, tokens.dtype, *((kernel.dtype,) if trans_rhs else ()))
                         and not tiled(ba.arguments['prologue_operands'], hooks.prologue_axes, 'depth')
                         and ba.arguments.get('gather') is None)
        width_partial = (rows_of(width, *outputs, *(() if trans_rhs else (kernel.dtype,)))
                         and not tiled(ba.arguments.get('epilogue_operands', ()), hooks.epilogue_axes,
                                       'columns')
                         and ba.arguments.get('scatter') is None)
        return depth_partial, width_partial

    def _depth_unit(self, ba, family: mosaic.Family, dtype) -> int:
        # The least step its stage takes, in whole units of a prologue's depth vectors (`candidate`'s, the
        # depth mask's among them, `_aligned`): sm89's mma.sync pads 200 to 224 by its least step alone, which
        # no step of its 64-element vector unit divides (EM-unal).
        step = min(self.STEPS or family.depths(dtype))
        if ba.arguments['hooks'].prologue is None:
            return step
        return math.lcm(step, family.vector_unit(jnp.float32), self._vector_units(ba, family)[2])

    def _aligned(self, ba, *, whole_tiles: bool = False):
        # A prologue never sees the depth's padding: its result zeroed past the call's depth by a side vector
        # over the depth (`mosaic_gpu.depth_masked`), bound with the padded call so its planner, its register
        # model and its kernel stage the vector alike.
        aligned = super()._aligned(ba, whole_tiles=whole_tiles)
        hooks = aligned.arguments['hooks']
        depth = self._padded_extents(ba).get('depth')
        padded = self._padded_extents(aligned).get('depth')
        # A depth run with its last step partial (`_partial_axes`) is masked too, its mask a vector of the
        # depth whose last step's load past it reads zeros.
        layouts = self._families(ba, generation.gpu_target())
        steps = self.STEPS or (layouts[0].depths(ba.arguments['tokens'].dtype) if layouts else ())
        whole = depth is not None and any(depth % step == 0 for step in steps)
        # A binding aligned once keeps its mask (planning aligns the bound call again: `_ran`, `registers`).
        if hooks.prologue is None or depth is None or (depth == padded and whole) or hooks.depth_masked:
            return aligned
        abstract = isinstance(ba.arguments['tokens'], jax.ShapeDtypeStruct)
        valid = (jax.ShapeDtypeStruct((padded,), jnp.float32) if abstract
                 else (jnp.arange(padded) < depth).astype(jnp.float32))
        masked = dataclasses.replace(
            hooks, prologue=functools.partial(mosaic_gpu.depth_masked, hooks.prologue), depth_masked=True,
            prologue_axes=(*hooks.prologue_axes, ('depth',)),
            prologue_dtypes=(*hooks.prologue_dtypes, 'float32') if hooks.prologue_dtypes else (),
            prologue_broadcast=(*hooks.prologue_broadcast, ()) if hooks.prologue_broadcast else ())
        return type(aligned)(aligned.op, {**aligned.arguments, 'hooks': masked, 'prologue_operands': (
            *aligned.arguments['prologue_operands'], valid)})

    def _padded_extents(self, ba) -> dict[str, int]:
        # A packed, step-scaled or block-scaled matrix's layout is its own: such a call is not padded.
        if self._scaled:
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
        # A dense product of fewer rows than a role's (a decode step's) may run swapped: its matrix takes the
        # MMA's rows and its rows the columns, whose unit is narrower, so its tile multiplies no padding rows
        # (CUTLASS's and cuBLAS's swap of A and B; megabake's 4 x 2048 x 2048 product went from 10.0 to
        # 4.7 us swapped, cuBLAS's time, row 1762), derived as the swapped call's own tile, and taken where
        # its walk prices below the unswapped tile's (row 5874: an NVFP4 decode's swap won on wide outputs
        # and lost 25-31% on square and narrow ones, M-EM-26-gh200).
        plain = self._decode_split(ba, super().derived(ba, target), target)
        swapped = self._swapped(ba, target)
        tile = None if swapped is None else self._derived(swapped, target, admits=lambda tile: (
            self._left_fits(ba, tile) and tile.block_m <= SWAPPED_ROWS))
        if tile is None:
            return plain
        # Its few output tiles leave a wave's programs idle: its depth splits where the busiest program then
        # walks shorter, its fp32 partials' passes and their sum's launch counted (`pricing.split_count`,
        # the kernel gradient's rule; rows 1762 and 483), every run in the one launch (`mosaic_gpu.gmm`).
        tile = dataclasses.replace(tile, swapped=True)
        splits = 1 if ba.arguments['exact_sums'] else pricing.split_count(
            dataclasses.replace(tile, swapped=False), **self._contraction(ba, tile, target))
        split = dataclasses.replace(tile, splits=splits)
        tile = split if splits > 1 and self.legal(ba, split, target) else tile
        walked = {each: pricing.walk_seconds(each, self._contraction(ba, each, target))
                  for each in (tile, plain) if each is not None}
        return plain if plain is not None and walked[plain] <= walked[tile] else tile

    def _contraction(self, ba, tile: mosaic_gpu.Tiles, target) -> dict:
        """A decode-sized dense call's walk at `tile` (`pricing.decode_contraction`): swapped, its matrix
        the rows (`_swapped`), its operands' bytes and decoded forms the launch's (`_operand_bytes`)."""
        _, depth, width = self._generated_extents(ba)
        assert depth is not None, "a product's steps run over its depth"
        rows = ba.arguments['tokens'].shape[0]
        swapped = self._swapped(ba, target) if tile.swapped else None
        return pricing.decode_contraction(
            tile, target, rows=rows, groups=ba.arguments['sizes'].shape[0], depth=depth, width=width,
            columns=None if swapped is None else swapped.arguments['kernel'].shape[1],
            # A decode step's partials hold its rows, a TMA descriptor's whole 8s of them.
            held=-(-rows // mosaic_gpu.TMA_ROWS) * mosaic_gpu.TMA_ROWS,
            operands=self._operand_bytes(ba, tile), multiprocessors=_multiprocessors(target))

    def _operand_bytes(self, ba, tile: mosaic_gpu.Tiles) -> tuple[tuple[float, float], tuple]:
        """The bytes an element of the products' left and right operands moves from device memory at `tile`
        (swapped, the matrix the left) and each one's decode (`pricing.step_shared_bytes`): none."""
        tokens, kernel = ba.arguments['tokens'], ba.arguments['kernel']
        rows, matrix = jnp.dtype(tokens.dtype).itemsize, jnp.dtype(kernel.dtype).itemsize
        return ((matrix, rows) if tile.swapped else (rows, matrix)), (None, None)

    def _decode_split(self, ba, tile: mosaic_gpu.Tiles | None, target) -> mosaic_gpu.Tiles | None:
        """`tile`, a call's of decode-sized groups (each group's rows within one row tile: a MoE decode
        step's experts, or a dense product's decode step run unswapped, its one group) its depth split where
        the busiest program then walks shorter (`pricing.split_count` over each active group's one row
        tile, its matrix's stream the walk): one wave of tiles streamed each expert's matrix at a program's
        share of the memory rate, 1.28-1.46x Triton's time at the same DRAM bytes, 48% of the peak rate
        against Triton's 80% (FS-3622-NCU1, FS-3622-WAVES1, row 3622); a dense decode on a GH200, unsplit,
        ran Llama-3-8B's 4096 x 4096 at one row on 16 programs of 132, 4.3x XLA's time (M-EM-38). None,
        exact sums (`legal`), hooks or a pair keep it unsplit."""
        hooks = ba.arguments['hooks']
        if (tile is None or ba.arguments['exact_sums'] or ba.arguments.get('paired') is not None
                or hooks.prologue is not None or hooks.epilogue is not None):
            return tile
        if ba.arguments['tokens'].shape[0] > ba.arguments['sizes'].shape[0] * tile.block_m:
            return tile
        splits = pricing.split_count(tile, **self._contraction(ba, tile, target))
        split = dataclasses.replace(tile, splits=splits)
        return split if splits > 1 and self.legal(ba, split, target) else tile

    def _left_fits(self, ba, config: mosaic_gpu.Tiles) -> bool:
        """Whether a swapped tile reads `ba`'s matrix as its products' left operand: a matrix lying `[depth,
        width]` is read transposed, whose stage casts to the products' layout in whole row atoms on a family
        reading registers (`Family.transposed_rows`, as the kernel gradient's transposed rows)."""
        family = pricing.family_of(config, ba.arguments['tokens'].dtype)
        return (ba.arguments['trans_rhs'] or not family.operands_in_registers
                or config.block_k % family.transposed_rows == 0)

    def _swapped(self, ba, target):
        """The dense product `ba` as its swapped tile runs it: its matrix `[width, depth]` the rows, however
        it lies, and its rows a matrix of them, padded to whole column units of every family the target
        runs, read transposed (`mosaic_gpu.gmm`'s `transpose_lhs` where the matrix lies `[depth, width]`).
        None where it runs as it is: a call with hooks, a paired or grouped one, one whose rows fill a role
        (`Family.role_rows`), or one on a family it does not swap on (`_swaps_on`)."""
        families = self._families(ba, target)
        tokens, kernel = ba.arguments['tokens'], ba.arguments['kernel']
        if (not families or not all(self._swaps_on(family, tokens.dtype) for family in families)
                or not ba.arguments.get('dense') or ba.arguments['hooks'] != NO_HOOKS
                or ba.arguments.get('paired') is not None
                or tokens.shape[0] >= min(family.role_rows for family in families)):
            return None
        depth, width = (kernel.shape[2], kernel.shape[1]) if ba.arguments['trans_rhs'] else kernel.shape[1:]
        unit = max(family.column_unit(tokens.dtype) for family in families)
        abstract = functools.partial(jax.ShapeDtypeStruct, dtype=tokens.dtype)
        swapped = self.SWAPPED_OPTIONS
        options = {name: value for name, value in ba.arguments.items()
                   if name not in ('tokens', 'kernel', 'sizes', 'trans_rhs', *swapped)}
        return self.bind(abstract((width, depth)), abstract((1, -(-tokens.shape[0] // unit) * unit, depth)),
                         ba.arguments['sizes'], **options, trans_rhs=True, **swapped)

    SWAPPED_OPTIONS: ClassVar[dict] = {}
    """The options a swapped call binds beside its orientation (`_swapped`): a packed matrix's
    `packed_rows`."""

    def _swaps_on(self, family: mosaic.Family, dtype) -> bool:
        """Whether a call of `dtype` rows runs swapped on `family` (`_swapped`): its products read their
        operands from registers (warpgroup MMA's, reading shared memory itself, ran its swapped tile 1.606x
        the unswapped one's time on an H100, geomean of 24 decode rows, NS-SWAP2-h100, where mma.sync's ran
        0.48-0.63x on an A100's narrow outputs, NS-SWAP-a100) and the rows are 16-bit."""
        return family.operands_in_registers and jnp.dtype(dtype).itemsize == 2

    def _swapped_launch(self, matrix, tokens, sizes, *, out_dtype: str, interpret: bool, exact_sums: bool,
                        config: mosaic_gpu.Tiles, **matrix_options) -> tuple:
        """A swapped tile's launch (`_swapped`): `matrix` the products' rows times the rows, padded to whole
        column tiles and read transposed, at the tile unswapped, the output transposed back;
        `matrix_options`, how the launch reads the matrix (`transpose_lhs`, a packed one's scales)."""
        rows = tokens.shape[0]
        padded = jnp.pad(tokens, ((0, -rows % config.block_n), (0, 0)))
        tile = dataclasses.replace(config, swapped=False)
        out = mosaic_gpu.gmm(matrix, padded[None], sizes, tiles=tile, transpose_rhs=True, out_dtype=out_dtype,
                             multiprocessors=self._launched(tile, interpret),
                             interpret=INTERPRETED if interpret else False, dense=True, exact_sums=exact_sums,
                             **matrix_options)
        assert isinstance(out, jax.Array), "one output (`out_dtypes` is not passed)"
        return out[:, :rows].T, None

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
        # A split is a decode step's (a dense call's, swapped or not, or a grouped one's), each run some
        # steps, of no exact sums: their runs sum exactly (`mosaic_gpu.gmm`), but an A100's exact input
        # gradient split four ways at 64 x 128 x 32 (six stages, two programs) spilled 24 bytes where its
        # register model held 187 (the GEMM compile gate, EM-G1), its runs' traced steps beside the running
        # sum.
        depth = ba.arguments['tokens'].shape[1]
        if config.splits > 1 and (ba.arguments['exact_sums']
                                  or not mosaic_gpu.split_runs(-(-depth // config.block_k), config.splits)):
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
        to the mainloop as they are beside the register model its launch states (`_stated`)."""
        width = kernel.shape[1] if trans_rhs else kernel.shape[2]
        call = (bound := self._aligned(self.bind(
            tokens, kernel, sizes, prologue_operands, epilogue_operands, paired, first, gather, scatter,
            out_dtype=out_dtype, trans_rhs=trans_rhs, interpret=interpret, hooks=hooks, exact_sums=exact_sums,
            dense=dense))).arguments
        tokens, kernel, paired, hooks = call['tokens'], call['kernel'], call.get('paired'), call['hooks']
        prologue_operands, epilogue_operands = call['prologue_operands'], call.get('epilogue_operands', ())
        out, residuals = self._launched_call(tokens, kernel, sizes, prologue_operands, epilogue_operands,
                                             paired, first, gather, scatter, out_dtype=out_dtype,
                                             trans_rhs=trans_rhs, interpret=interpret, hooks=hooks,
                                             exact_sums=exact_sums, dense=dense, config=config,
                                             formats={**formats, **self._stated(bound, config)})
        return jax.tree.map(lambda each: each[:, :width], out), residuals

    def _streamed_matrix(self, kernel, trans_rhs: bool) -> tuple:
        """The matrix the mainloop streams, whether its blocks lie transposed, and the format operands its
        stages take besides (`mosaic_gpu.gmm`'s): the call's own matrix, as it lies, none; an N-major FP8
        matrix transposed by XLA to the K-major one Hopper's FP8 tensor cores read (no TMA or wgmma
        transposes an 8-bit block), a copy of the matrix a call, where the grouped FP8 forward otherwise had
        no launch but XLA's, 40.0 ms against Triton's broken lowering at an H100's refused-call shape
        (GH200-EM-8c-TRITON)."""
        if eight(kernel.dtype) and not trans_rhs:
            return jnp.swapaxes(kernel, 1, 2), True, {}
        return kernel, trans_rhs, {}

    def _matrix_transposed(self, ba) -> bool:
        # Read from the binding's facts alone, which a derivation binds abstract (`_streamed_matrix`).
        return bool(ba.arguments['trans_rhs']) or eight(ba.arguments['kernel'].dtype)

    def _launched_call(self, tokens, kernel, sizes, prologue_operands, epilogue_operands, paired, first,
                       gather, scatter, *, out_dtype: str, trans_rhs: bool, interpret: bool, hooks: Hooks,
                       exact_sums: bool, dense: bool, config: mosaic_gpu.Tiles, formats: dict):
        """`_fwd`'s launch of the call as it runs it (`_aligned`), `formats` its mainloop's keywords."""
        if config.swapped:
            return self._swapped_launch(kernel[0], tokens, sizes, out_dtype=out_dtype, interpret=interpret,
                                        exact_sums=exact_sums, config=config, transpose_lhs=not trans_rhs,
                                        **formats)
        matrix, transposed, stages = self._streamed_matrix(kernel, trans_rhs)
        return mosaic_gpu.gmm(tokens, matrix, sizes, first=first, tiles=config, transpose_rhs=transposed,
                              out_dtype=out_dtype, multiprocessors=self._launched(config, interpret),
                              interpret=INTERPRETED if interpret else False,
                              prologue_operands=prologue_operands, epilogue_operands=epilogue_operands,
                              dense=dense, paired=paired, exact_sums=exact_sums, gather=gather,
                              scatter=scatter, **stages, **formats, **hooks.kernel_options()), None


class _PackedMatrix(Protocol):
    """A packed matrix's fields as the packed launch streams them (`formats.NvFp4`'s)."""

    words: jax.Array
    scales: jax.Array


@dataclasses.dataclass(frozen=True)
class MosaicPackedGmm(MosaicGmm):
    """`MosaicGmm` of an NVFP4 matrix (`gemm.nvfp4_product`'s packed kernel, row 3621): bfloat16 rows times
    its `kernel`, the bank as Dew stores it (`formats.NvFp4`: its words and block scales, K-major, its shape
    and dtype the bfloat16 matrix it decodes to, `[groups, depth, width]`), streamed as its words and block
    scales and each stage decoded in place into the consumer's buffers (`mosaic_gpu.gmm`'s `scales`), so its
    tile takes `packed.NVFP4_STEP`-deep steps, whole 64-column tiles, the decode's shared memory and
    registers, and no cluster, split or hook. A decode-sized dense call runs swapped (`_swapped`, row 5387):
    the matrix the products' rows, each stage of it decoded in registers into their left operand, the call's
    rows the columns."""

    STEPS: ClassVar[tuple[int, ...] | None] = (packed.NVFP4_STEP,)

    def _fwd(self, tokens: jax.Array, kernel: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), epilogue_operands: tuple = (), paired: jax.Array | None = None,
             first: jax.Array | None = None, gather: jax.Array | None = None,
             scatter: jax.Array | None = None, *, out_dtype: str, trans_rhs: bool, interpret: bool,
             hooks: Hooks = NO_HOOKS, exact_sums: bool = False, dense: bool = False,
             config: mosaic_gpu.Tiles, return_residuals: bool = False, packed_rows: bool = False):
        # `packed_rows` states the swapped call's orientation (`_swapped`), which its derivation reads: the
        # call itself runs as `config` says (`_launched_call`).
        del packed_rows
        return super()._fwd(tokens, kernel, sizes, prologue_operands, epilogue_operands, paired, first,
                            gather, scatter, out_dtype=out_dtype, trans_rhs=trans_rhs, interpret=interpret,
                            hooks=hooks, exact_sums=exact_sums, dense=dense, config=config,
                            return_residuals=return_residuals)

    def _streamed_matrix(self, kernel: "_PackedMatrix", trans_rhs: bool) -> tuple:
        # Its words, K-major whatever orientation its decoded shape states, and its block scales, which each
        # stage decodes (`mosaic_gpu.gmm`'s `scales`).
        del trans_rhs
        return kernel.words, True, {'scales': kernel.scales}

    def _matrix_transposed(self, ba) -> bool:
        # Its words K-major, or a swapped call's rows read transposed (`_swapped`): its matrix streams
        # transposed either way, whatever the binding holds (an abstract matrix where a derivation binds it).
        del ba
        return True

    def _allocations(self, ba, config: mosaic_gpu.Tiles, family: mosaic.Family) -> mosaic_gpu.Allocations:
        dtype = ba.arguments['tokens'].dtype
        rows = ba.arguments['packed_rows']
        return mosaic_gpu.gmm_allocations(
            config, family, dtype=dtype, rhs_dtype=dtype if rows else jnp.uint32, transpose_rhs=True,
            prologue_axes=(), prologue_dtypes=(), epilogue_axes=(), epilogue_dtypes=(),
            dtypes=(ba.arguments['out_dtype'],), packed=True, packed_rows=rows)

    def candidate(self, ba, config: mosaic_gpu.Tiles, target) -> bool:
        # A swapped call's consumers each decode 64 rows of the matrix, its fragment's; the matrix's columns
        # take whole 64-column fragments otherwise (`packed.rows_taken`, `packed.taken`).
        whole = (config.block_m // config.warpgroups == 64 if ba.arguments['packed_rows']
                 else config.block_n % 64 == 0)
        return (config.cluster == 1 and config.splits == 1 and not config.swapped and whole
                and super().candidate(ba, config, target))

    SWAPPED_OPTIONS: ClassVar[dict] = {'packed_rows': True}

    def _operand_bytes(self, ba, tile: mosaic_gpu.Tiles) -> tuple[tuple[float, float], tuple]:
        # Its matrix moves its words and block scales (`formats.NvFp4`): decoded into the stage the products
        # read (`gmm`'s `scales`, its decoded dtype's: one its consumers share or each one's own,
        # `packed.decoding`), or swapped, in registers into them, each consumer its rows (`packed_rows`).
        tokens, kernel = ba.arguments['tokens'], ba.arguments['kernel']
        stored = sum(leaf.size * jnp.dtype(leaf.dtype).itemsize
                     for leaf in (kernel.words, kernel.scales)) / math.prod(kernel.shape)
        rows, stage = jnp.dtype(tokens.dtype).itemsize, jnp.dtype(kernel.dtype).itemsize
        if tile.swapped:
            return (stored, rows), ((0, 1), None)
        copies = 1 if packed.shared_decode(tile, interpret=False) else tile.warpgroups
        return (rows, stored), (None, (stage, copies))

    def _swaps_on(self, family: mosaic.Family, dtype) -> bool:
        # Swapped, its matrix's stages decode in registers into the products' left operand
        # (`mosaic_gpu.gmm`'s `packed_rows`, row 5387), so no decoded stage passes through shared memory:
        # on a family whose products take a left operand from registers in the decode's layout
        # (`packed.rows_taken`: the warpgroup MMA's fragment).
        del dtype
        return family.layout == plgpu.Layout.WGMMA

    def _launched_call(self, tokens, kernel, sizes, prologue_operands, epilogue_operands, paired, first,
                       gather, scatter, *, out_dtype: str, trans_rhs: bool, interpret: bool, hooks: Hooks,
                       exact_sums: bool, dense: bool, config: mosaic_gpu.Tiles, formats: dict):
        if not config.swapped:
            return super()._launched_call(tokens, kernel, sizes, prologue_operands, epilogue_operands,
                                          paired, first, gather, scatter, out_dtype=out_dtype,
                                          trans_rhs=trans_rhs, interpret=interpret, hooks=hooks,
                                          exact_sums=exact_sums, dense=dense, config=config, formats=formats)
        # The matrix's words and scales the rows, decoded in registers (`_swaps_on`).
        return self._swapped_launch(kernel.words[0], tokens, sizes, out_dtype=out_dtype, interpret=interpret,
                                    exact_sums=exact_sums, config=config, scales=kernel.scales[0],
                                    packed_rows=True, **formats)

    def registers(self, ba, config: mosaic_gpu.Tiles) -> int | None:
        # Beside the plain product's live set, a step's decode holds a 128-depth group's words, the code
        # being decoded and the stage's block scales, each a `[columns, 16]` register tile over a role's
        # threads (`packed.lanes`, its columns the tile's, or a consumer's rows where the call runs swapped),
        # and a word's decode in flight, its form's live registers (`bits.BitForm.live`); swapped, the
        # decoded stage too, the products' left operand (`packed.rows_decoded`, bfloat16 a half register).
        if ba.arguments['packed_rows']:
            share = config.block_m // config.warpgroups
            decoded = 3 * share * 16 + share * config.block_k // 2
        else:
            decoded = 3 * config.block_n * 16
        return _beside(super().registers(ba, config),
                       decoded // resources.THREADS + bits.NVFP4_BFLOAT16_FORM.live)


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
        # The derived tile, split where its output tiles underfill a wave (`_split_tile`). An exact sum splits
        # too, its runs summed exactly and their partials by IEEE adds (`mosaic_gpu.tgmm`): unsplit,
        # Qwen3.5-4B's key kernel gradient ran 2.03-2.06x the tensor cores' split sums on a GH200
        # (W123-gh200, W124-gh200).
        tile = self._split_tile(ba, target)
        # A call whose programs fill less than a wave (its output tiles, each split as deep as its ring
        # allows) takes as long as its busiest program's walk, not its reads a FLOP, which the derived width
        # balances for a full wave: its width narrows while that walk, its partials' passes and their sum's
        # launch take less (`pricing.SplitWalk`). An A100's 64 x 256 gradient over 4096 rows split 16
        # ways at 64 x 128 ran 1.06x its time at 64 x 64 split 11 ways, and four groups' 200 x 100 ones over
        # 1024 rows a group 1.10x unsplit at 64 x 128 against 64 x 64 split 3 ways (M-JS-88-gemm-picks-a100).
        while tile is not None and not self._balances(ba, tile, target):
            walk = self._walk(ba, tile, target)
            if walk is None or not walk.underfills(tile.splits):
                break
            narrower = self._split_tile(ba, target, below=tile.block_n)
            if narrower is None:
                break
            walked = self._walk(ba, narrower, target)
            if walked is None or walked.seconds(narrower.splits) >= walk.seconds(tile.splits):
                break
            tile = narrower
        return tile

    def _split_tile(self, ba, target, below: int | None = None) -> mosaic_gpu.Tiles | None:
        """The derived tile (narrower than `below` columns where given), its contraction split where its
        output tiles underfill a wave (`pricing.split_count`) at the tile the split's own call, writing
        fp32 partials, derives, one a split runs on (a pair's refuses one, `mosaic_gpu.pair_refuses`)."""
        admits = None if below is None else (lambda tile: tile.block_n < below)
        tile = self._derived(ba, target, admits)
        if tile is None or self._balances(ba, tile, target) or self._splits(ba, tile, target) == 1:
            return tile
        partials, d = self._partials(ba), ba.arguments['tokens'].dtype
        sliced = self._derived(partials, target, lambda each: (admits is None or admits(each)) and not (
            self._pair_refuses(partials, dataclasses.replace(each, splits=2), pricing.family_of(each, d))))
        if sliced is None or (splits := self._splits(partials, sliced, target)) == 1:
            return tile
        split = dataclasses.replace(sliced, splits=splits)
        # A split a compile of this process found spilling (`report.refused`) gives way to the call's own.
        return tile if report.refused(self, ba, split) else split

    def _balances(self, ba, tile: mosaic_gpu.Tiles, target) -> bool:
        """Whether the call's last wave runs balanced over the programs (`tail.balanced`), which it
        takes no split beside: a dense one on a family that fixes a tile up across programs (its exact
        partials added by IEEE adds, `mosaic_gpu.tgmm`), its tail balanced."""
        if not (ba.arguments['dense'] and pricing.fixes_up(tile, ba.arguments['tokens'].dtype)):
            return False
        ba = self._aligned(ba)
        depth, _, width = self._generated_extents(ba)
        tokens, cotangent = ba.arguments['tokens'], ba.arguments['cotangent']
        steps = -(-tokens.shape[0] // tile.block_k)
        return pricing.balanced_tail(tile, (-(-depth // tile.block_m), width // tile.block_n), steps,
                                        (tokens.dtype.itemsize, cotangent.dtype.itemsize), target,
                                        _multiprocessors(target) * tile.programs) is not None

    def _splits(self, ba, tile: mosaic_gpu.Tiles, target) -> int:
        return pricing.split_count(tile, **self._contraction(ba, tile, target))

    def _walk(self, ba, tile: mosaic_gpu.Tiles, target) -> pricing.SplitWalk | None:
        """The walk `tile` runs the call in (`pricing.split_walk`): a split tile's over the partials' call
        its slices run (`_partials`)."""
        call = ba if tile.splits == 1 else self._partials(ba)
        return pricing.split_walk(tile, **self._contraction(call, tile, target))

    def _contraction(self, ba, tile: mosaic_gpu.Tiles, target) -> dict:
        """The call's contraction as the split prices it at `tile` (`pricing.split_walk`'s facts): the call
        as the launch runs it (`_aligned`), its depth and width padded to whole tiles (a 100-wide gradient's
        128 columns)."""
        ba = self._aligned(ba)
        depth, _, width = self._generated_extents(ba)
        tokens, cotangent = ba.arguments['tokens'], ba.arguments['cotangent']
        itemsizes = (tokens.dtype.itemsize, cotangent.dtype.itemsize)
        return {'groups': ba.arguments['sizes'].shape[0], 'rows': _rows_of(_TGMMS, ba.args), 'depth': depth,
                'width': width, 'itemsizes': itemsizes, 'multiprocessors': _multiprocessors(target),
                'step_seconds': pricing.step_seconds(tile, target, itemsizes),
                'bandwidth': target.memory_rate,
                'launch_seconds': target.launch_seconds}

    def _partials(self, ba):
        """The call `ba` writing fp32 partials, as a split's slices do (`mosaic_gpu.tgmm`): each slice a group
        of its own, so no dense call's; a dense call's slices whole steps from its first row (`whole_steps`),
        their stages read as loaded as its own (`unstaged`)."""
        return self.bind(*ba.args, **{**ba.kwargs, 'out_dtype': 'float32', 'dense': False,
                                      'whole_steps': bool(ba.arguments.get('dense'))})

    def _ran(self, ba, config: mosaic_gpu.Tiles) -> tuple:
        """The call and tile a launch at `config` runs (`_aligned`): a split one's fp32 partials, unsplit."""
        ba = self._aligned(ba)
        if config.splits == 1:
            return ba, config
        return self._partials(ba), dataclasses.replace(config, splits=1)

    def _divides(self, ba, config: mosaic_gpu.Tiles) -> bool:
        rows, _, width = self._generated_extents(ba)  # a pair's tile both CTAs' (`pricing.tile_rows`)
        family = pricing.family_of(config, ba.arguments['tokens'].dtype)
        whole = (rows % pricing.tile_rows(config, family) == 0
                 or (self._ragged(ba, family) and not family.pairs_product))
        return whole and width % pricing.tile_columns(config, family) == 0

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

    def candidate(self, ba, config: mosaic_gpu.Tiles, target) -> bool:
        """`_Mosaic.candidate` on one multiprocessor a tile or a pair (`tgmm` multicasts nothing), a family
        whose products read registers loading its transposed stage in whole row atoms of its products (the
        step, `block_k`, in `Family.transposed_rows`)."""
        if not super().candidate(ba, config, target):
            return False
        family = pricing.family_of(config, ba.arguments['tokens'].dtype)
        return ((config.cluster == 1 or family.pairs_product)
                and not (family.operands_in_registers and config.block_k % family.transposed_rows))

    BLOCKS: ClassVar[dict[str, str]] = {'rows': 'block_k', 'depth': 'block_m'}


    def _fwd(self, tokens: jax.Array, cotangent: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), gather: tuple | None = None, *, out_dtype: str, interpret: bool,
             hooks: Hooks = NO_HOOKS,
             exact_sums: bool = False, dense: bool = False, config: mosaic_gpu.Tiles,
             return_residuals: bool = False, whole_steps: bool = False):
        # The call as the launch runs it, its depth and width padded to whole tiles (`_aligned`), its output
        # sliced back. `whole_steps`, a dense call's split runs (`_partials`), as `mosaic_gpu.tgmm`'s.
        depth, width = tokens.shape[1], cotangent.shape[1]
        call = (bound := self._aligned(self.bind(
            tokens, cotangent, sizes, prologue_operands, gather, out_dtype=out_dtype, interpret=interpret,
            hooks=hooks, exact_sums=exact_sums, dense=dense, whole_steps=whole_steps))).arguments
        out = mosaic_gpu.tgmm(call['tokens'], call['cotangent'], sizes, tiles=config, out_dtype=out_dtype,
                              multiprocessors=self._programs(config),
                              interpret=INTERPRETED if interpret else False, prologue=hooks.before(),
                              prologue_operands=call['prologue_operands'], prologue_axes=hooks.prologue_axes,
                              exact_sums=exact_sums, dense=dense, gather=gather, whole_steps=whole_steps,
                              **self._stated(bound, config))
        return out[:, :depth, :width], None


@functools.cache
def cutlass_registered() -> None:
    """dew_cutlass's FFI targets, registered with jax once: at the first launch that runs one, or before a
    program compiled ahead is loaded (`dew.training.ahead.Executable.loaded`), which traces none."""
    for target, capsule in importlib.import_module('dew_cutlass').targets().items():
        jax.ffi.register_ffi_target(target, capsule, platform="CUDA")


@dataclasses.dataclass(frozen=True)
class _ScaledGmm(MosaicGmm):
    """`MosaicGmm` of a product whose rows' and matrix's scales its binding takes as one operand (`scales`),
    reaching the mainloop as the keyword of `mosaic_gpu.gmm`'s its subclass names (`SCALES`)."""

    SCALES: ClassVar[str]

    def _swapped(self, ba, target):
        # Its scales are its rows' and columns': no swapped orientation reads them.
        del ba, target

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

    def registers(self, ba, config: mosaic_gpu.Tiles) -> int | None:
        # Beside the exact product's live set, a step's column scales.
        return _beside(super().registers(ba, config), _column_scale_registers(ba, config))


@dataclasses.dataclass(frozen=True)
class MosaicMxGmm(_ScaledGmm):
    """`MosaicGmm` of a block-scaled product (`gemm.mxfp8_product`, `gemm.nvfp4_block_product`, row 4519),
    planned on the call it is: FP8 or E2M1 rows times a K-major matrix of theirs, each block of each scaled
    by its scale in the products (`product.block_format`)
    (`mosaic_gpu.gmm`'s `mx_scales`, `mosaic.Issuer.scaled_products`), its block scales streamed beside its
    rings (`gmm_allocations`' `block_scaled`) on a family whose products take them (tcgen05's), its tiles
    whole 128s of rows, columns and depth, and no split, swap, hook or exact sums."""

    BLOCK_SCALED: ClassVar[bool] = True
    SCALES: ClassVar[str] = 'mx_scales'

    def _fp8(self, ba, dtype, family: mosaic.Family) -> bool:
        # Its products scale each block in the instruction, which sums in fp32 itself: no step promoted. Its
        # operands FP8 (MXFP8) or E2M1 (NVFP4), `product.block_format`'s.
        return (jnp.dtype(dtype) in family.block_formats
                and ba.arguments['trans_rhs'] and ba.arguments['hooks'] == NO_HOOKS
                and ba.arguments.get('paired') is None and not ba.arguments.get('exact_sums'))

    def candidate(self, ba, config: mosaic_gpu.Tiles, target) -> bool:
        whole = all(extent % 128 == 0 for extent in (config.block_m, config.block_n, config.block_k))
        return config.splits == 1 and not config.swapped and whole and super().candidate(ba, config, target)


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
             config: mosaic_gpu.Tiles, return_residuals: bool = False, whole_steps: bool = False):
        # Bound as a kernel gradient states its operands, rows first: the rows-contiguous operands' transposes
        # (`gemm.blockwise_fp8_kernel_gradient`), which XLA folds into the kernel's own operands, read back as
        # they lie (`mosaic_gpu.tgmm`'s `k_major`); its scales (`step_scales`, the binding's operand) its
        # mainloop's `step_scales`. Its layout's steps are whole by the caller's (`k_major`), unsplit.
        del prologue_operands, gather, hooks, dense, return_residuals, whole_steps
        bound = self.bind(tokens, cotangent, sizes, out_dtype=out_dtype, interpret=interpret,
                          exact_sums=exact_sums, step_scales=step_scales)
        out = mosaic_gpu.tgmm(tokens.T, cotangent.T, sizes, tiles=config, out_dtype=out_dtype,
                              multiprocessors=self._launched(config, interpret),
                              interpret=INTERPRETED if interpret else False, exact_sums=exact_sums,
                              k_major=True, step_scales=tuple(step_scales), **self._stated(bound, config))
        return out, None

    def _fp8(self, ba, dtype, family: mosaic.Family) -> bool:
        # Its operands lie rows-contiguous, as Hopper's FP8 tensor cores read both.
        return (jnp.dtype(dtype) in family.fp8_formats
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
    family = pricing.family_of(config, ba.arguments['tokens'].dtype)
    return family.column_registers(config.block_n, jnp.float32)


def _cutlass_call(target: int, out: jax.ShapeDtypeStruct) -> Callable:
    """dew_cutlass's FFI call giving `out` of the `target`-th of the targets the wheel names
    (`dew_cutlass.TARGETS`: its grouped GEMM, then its weight gradient), a mapped call running its maps in
    turn, one launch each (row 4389: JAX's FFI batching refuses a call that states no rule)."""
    name = importlib.import_module('dew_cutlass').TARGETS[target]
    return jax.ffi.ffi_call(name, out, vmap_method='sequential')


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


def cutlass_built_for(generation_name: str) -> str | None:
    """The generation of the installed dew_cutlass's kernels a GPU of `generation_name` runs, as the wheel
    dispatches on its compute capability (`dew_cutlass.architecture`, the one statement of it); None where
    the wheel is not installed or runs none there."""
    return generation.cutlass_built().get(generation_name)


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
        return cutlass_built_for(target.generation) is not None and self.takes(ba)

    def tile(self, ba) -> mosaic_gpu.Tiles | None:
        """The record of the kernel the build holds for the GPU being selected for (`generation.gpu_target`,
        `cutlass_built_for`) writing the call's output (`cutlass_tile`); None where it holds none."""
        built = cutlass_built_for(generation.gpu_target().generation)
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
    """`Gmm` through CUTLASS (the wheel's grouped GEMM target, `dew_cutlass.TARGETS`)."""

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
        cutlass_registered()
        width = kernel.shape[1] if trans_rhs else kernel.shape[2]
        out = jax.ShapeDtypeStruct((tokens.shape[0], width), jnp.dtype(out_dtype))
        call = _cutlass_call(0, out)
        return call(tokens, kernel, sizes, transpose_w=trans_rhs), None


@dataclasses.dataclass(frozen=True)
class CutlassTgmm(_Cutlass):
    """`Tgmm` through CUTLASS (the wheel's weight-gradient target, `dew_cutlass.TARGETS`): each group's
    gradient a GEMM over its rows, an empty group's zero, its pair splitting the gradient's rows."""

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
        cutlass_registered()
        out = jax.ShapeDtypeStruct((sizes.shape[0], tokens.shape[1], cotangent.shape[1]),
                                   jnp.dtype(out_dtype))
        return _cutlass_call(1, out)(tokens, cotangent, sizes), None


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


class Made(enum.Enum):
    """What a grouped call makes (`Kind`): a forward product, its input gradient (the matrix read transposed),
    or the matrices' kernel gradient, summed by the tensor cores or exactly (`exact_sums`)."""

    FORWARD = 'forward'
    INPUT_GRADIENT = 'input_gradient'
    KERNEL_GRADIENT = 'kernel_gradient'
    EXACT_KERNEL_GRADIENT = 'kernel_gradient exact'


class Kind(NamedTuple):
    """The kind of call a grouped product is (`GroupedCase.kind`), which with the device's instruction family
    is its cell (`DECLARED`): dense (one matrix, every row) or grouped, what it makes, the operands no
    receipt of a 16-bit call times: FP8 ones (XLA's dot of them sums narrower than the FP8 contract holds,
    row 1625), rows gathered through a table or stored through one (no old route reads them); and, for a
    dense product whose matrix a receipt timed on its family (`MEASURED`), that matrix's `[in, out]` extents,
    a cell of its own, which the receipt's winner for it holds (None for every other product, the kind's
    cell)."""

    dense: bool
    made: Made
    fp8: bool = False
    gathered: bool = False
    scattered: bool = False
    matrix: tuple[int, int] | None = None

    def __str__(self) -> str:
        parts = ('fp8',) * self.fp8 + ('gathered',) * self.gathered + ('scattered',) * self.scattered
        timed = () if self.matrix is None else ('x'.join(map(str, self.matrix)),)
        return ' '.join(('dense' if self.dense else 'grouped', self.made.value, *parts, *timed))


@dataclasses.dataclass(frozen=True)
class GroupedCase:
    """A grouped product as the grouped selector reads it (`GROUPED`): whether the kernels run on its
    target (`placed`: a GPU of sm80 on, or a TPU; the template's dispatch runs in the branch its platform
    places), whether they see its operands whole (`whole`,
    `dew.nn.sharding.seen_whole`), the `product` it declares, its `rows`, its launches bound to it
    (`call`) and the `kind` of call it is (`Kind`, as `gmm_call` and `tgmm_call` state it), which with the
    device's instruction family is its cell (`DECLARED`)."""

    placed: bool
    whole: bool
    product: Product
    rows: int
    call: base.Call
    kind: Kind
    paired: bool = False
    """Whether it multiplies its rows by a second matrix beside the first (`gmm_call`'s `paired`)."""

    @property
    def gathered(self) -> bool:
        """Whether its rows are gathered from the tokens by a table (`gmm_call`'s `gather`)."""
        return self.kind.gathered

    @property
    def scattered(self) -> bool:
        """Whether its rows land where a table says (`gmm_call`'s `scatter`)."""
        return self.kind.scattered

    @property
    def cell(self) -> tuple[str | None, Kind]:
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


DECLARED_ONLY = _Need('declared', lambda case: DECLARED.get(case.cell, (None,))[0] is CUTLASS,
                      "CUTLASS's launches run on the cells where they measured faster than the template "
                      "(`DECLARED`), and where a call names them", row='bible da64ae5', kind='measured')
"""What keeps CUTLASS's launches, a vendored production route where they win (bible §6, as FA3), off every
other cell: the template, then Triton by legality, run those."""


DENSE_MEASURED = _Need('dense', lambda case: not case.kind.dense,
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

MOSAIC = _launch_kernel(MosaicGmm.name)
TRITON = _launch_kernel(Gmm.name, DENSE_MEASURED)
"""Triton's launches, off the dense products the template's launches cannot run (`DENSE_MEASURED`)."""
CUTLASS = _launch_kernel(CutlassGmm.name, DECLARED_ONLY)
"""CUTLASS's launches, on their declared cells alone (`DECLARED_ONLY`)."""
XLA = selection.Kernel[GroupedCase]('xla', (), generic=True)

FOUR_B = ((2560, 9216), (9216, 2560), (2560, 1024), (2560, 64))
"""Qwen3.5-4B's dense projections' matrices `[in, out]`: its MLP's up and down, its attention's key, its
gated delta net's gates, which M-EM-36-gh200 and em35d timed at 24,576 tokens, each product of each."""

MEASURED: dict[tuple[str | None, Made], frozenset[tuple[int, int]]] = {
    (family, made): frozenset(FOUR_B) for family in ('wgmma', 'tcgen05')
    for made in (Made.FORWARD, Made.INPUT_GRADIENT)}
"""The dense matrices each instruction family's receipts timed, by what the product makes: a product of one is
a cell of its own (`Kind.matrix`), which its receipt's winner holds; every other product its kind's. A kernel
gradient has none: XLA's dot of one is deferred to the backward's end, holding every layer's cotangents
live (the 4B's up, gate, key and value on a B200: 22.25 GB of the step's temporaries against 11.06 GB on
the template, compile-only, so a four-row step needed 148 GiB against 95), which no per-call time
prices."""

M_EM_36 = ("M-EM-36-gh200 (a GH200, base/kernels df82480cb5c, CUPTI medians of 5 rounds, cuBLASLt autotuned "
           "at run, 24,576 tokens)")
EM35D = ("em35d (a B200, base/kernels ce6eff7f0be, CUPTI medians of 5 rounds, cuBLASLt autotuned at run, "
         "24,576 tokens)")
B200_GRID = ("lab-b200-20261113-grid (a B200, base/kernels 5ac163abbec, medians of 5 rounds' same-round "
             "ratios, cuBLASLt autotuned at run, 24,576 tokens; its down kernel gradient the template's, "
             "0.833x)")

DECLARED: dict[tuple[str | None, Kind], tuple[selection.Kernel[GroupedCase], str]] = {
    ('mma.sync', Kind(dense=True, made=Made.FORWARD)):
        (XLA, "DENSE1 (A100, a55c8ffb2, one program a multiprocessor): Mosaic 1.923x and Triton 1.947x "
             "XLA's dot (cuBLAS), geomean of 10 projections at 8192 tokens; M-FS-9 (RTX PRO 6000, "
             "d7014761a, mma.sync.fed at two 64 x 64 programs a multiprocessor): Mosaic 1.99x, Triton "
             "1.09x, geomean of 13"),
    ('mma.sync', Kind(dense=True, made=Made.INPUT_GRADIENT)):
        (XLA, "DENSE1 (A100, one program a multiprocessor): Mosaic 2.881x, Triton 1.839x XLA's dot; "
             "M-FS-9 (RTX PRO 6000): Mosaic 3.00x, Triton 1.10x"),
    ('mma.sync', Kind(dense=True, made=Made.KERNEL_GRADIENT)):
        (XLA, "DENSE1 (A100, one program a multiprocessor): Mosaic 1.964x, Triton 1.735x XLA's dot; "
             "M-FS-9 (RTX PRO 6000): Mosaic 1.72x, Triton 1.28x"),
    ('mma.sync', Kind(dense=False, made=Made.FORWARD)):
        (CUTLASS, "M-EM-2 (A100-SXM4-40GB at 1095 MHz, base/kernels 59b96a9b4c2, 400 role-swapped pairs, A/A "
                 "within 0.993-1.008): the template 1.07x CUTLASS's time, geomean of 5 grouped MoE rows "
                 "(Qwen3-30B-A3B up 1.193, down 1.090; Mixtral up 0.795, down 1.090; GPT-OSS-20B up 1.244)"),
    ('wgmma', Kind(dense=True, made=Made.FORWARD)):
        (XLA, "DENSE1 (H100, a55c8ffb2): Mosaic 1.067x and Triton 1.616x XLA's dot (cuBLAS), geomean of "
             "10 projections at 8192 tokens"),
    ('wgmma', Kind(dense=True, made=Made.INPUT_GRADIENT)):
        (XLA, "DENSE1 (H100): Mosaic 1.048x, Triton 1.738x XLA's dot"),
    ('wgmma', Kind(dense=True, made=Made.FORWARD, matrix=(2560, 9216))): (
        MOSAIC, f"{M_EM_36}: the template 0.866x XLA's dot"),
    ('wgmma', Kind(dense=True, made=Made.FORWARD, matrix=(9216, 2560))): (
        XLA, f"{M_EM_36}: the template 1.01x XLA's dot"),
    ('wgmma', Kind(dense=True, made=Made.FORWARD, matrix=(2560, 1024))): (
        MOSAIC, f"{M_EM_36}: the template 0.96x XLA's dot"),
    ('wgmma', Kind(dense=True, made=Made.FORWARD, matrix=(2560, 64))): (
        XLA, f"{M_EM_36}: the template 1.294x XLA's dot"),
    ('wgmma', Kind(dense=True, made=Made.INPUT_GRADIENT, matrix=(2560, 9216))): (
        MOSAIC, f"{M_EM_36}: the template 0.943x XLA's dot"),
    ('wgmma', Kind(dense=True, made=Made.INPUT_GRADIENT, matrix=(9216, 2560))): (
        MOSAIC, f"{M_EM_36}: the template 0.912x XLA's dot"),
    ('wgmma', Kind(dense=True, made=Made.INPUT_GRADIENT, matrix=(2560, 1024))): (
        MOSAIC, f"{M_EM_36}: the template 0.949x XLA's dot"),
    ('wgmma', Kind(dense=True, made=Made.INPUT_GRADIENT, matrix=(2560, 64))): (
        MOSAIC, f"{M_EM_36}: the template 0.93x XLA's dot"),
    ('tcgen05', Kind(dense=True, made=Made.FORWARD, matrix=(2560, 64))): (
        XLA, f"{B200_GRID}: the template 1.216x XLA's dot (em35d 1.225x)"),
    ('tcgen05', Kind(dense=True, made=Made.INPUT_GRADIENT, matrix=(2560, 9216))): (
        XLA, f"{EM35D}: the template 1.009x XLA's dot"),
}
"""The grouped GEMM's declared cells (`selection.Op.declared`), where a route measured faster than the
template's default on a generation, each with its receipt (owner, bible da64ae5; "whatever is the
best/fastest remains the route until our templates catch up", ops 361caffc): (instruction family, kind of
call) to (that route's kernel, the receipt). A product of a matrix a receipt timed (`MEASURED`) is a cell of
its own, which that receipt's winner holds: the template's where it won (on Hopper, over its kind's XLA
cell), XLA's where it lost (on Blackwell, where the template is the kind's default); one timed whose cell is
undeclared takes the template. A cell leaves the list once the template measures at least as fast on it.
Its price includes the step memory its route holds (`HELD`), which no per-call time sees.
Nothing is ranked at runtime. A
call the template cannot run (its legality) takes the next route in `GROUPED`'s order, which is no cell of
this list."""

HELD: dict[tuple[str | None, Kind], tuple[int, str]] = {
    ('mma.sync', Kind(dense=True, made=Made.KERNEL_GRADIENT)): (
        1, "XLA's dot of each layer's kernel gradient deferred to the backward's end, each layer's cotangent "
           "held to it: Llama-3-8B's MLP at 8192 tokens 32 x 2 x 8192 x 14336 bf16, 15 GiB"),
}
"""The step memory a declared cell's route holds beyond the template's, which its receipt states beside its
time (row 5972's P0: the B200's and GH200's kernel-gradient cells held the 4B's every layer's dgate and dup to
the backward's end, a four-row step's temporaries 22.25 GB against 11.06 on the template): (instruction
family, kind of call) to (the outputs of its call each layer holds, the receipt's statement). A cell absent
holds none (tests/test_declared_cells_memory.py)."""


GROUPED: selection.Op[GroupedCase] = selection.Op(generation.Choice.GROUPED_MATMUL, (
    MOSAIC, TRITON, CUTLASS,
    selection.Kernel[GroupedCase](APART.name, (_Need('single', lambda case: case.paired,
                                                "the call is one product", row=INHERENT),), generic=True),
    XLA), declared=DECLARED)
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
        made = Made.EXACT_KERNEL_GRADIENT if options['exact_sums'] else Made.KERNEL_GRADIENT
    else:
        made = Made.INPUT_GRADIENT if options['trans_rhs'] else Made.FORWARD
    # An FP8 call's cells are its own: `DECLARED`'s receipts timed 16-bit products, and XLA's dot of FP8
    # operands sums narrower than the FP8 contract holds (row 1625), so the template takes them by default.
    fp8 = any(eight(argument.dtype) for argument in arguments[:2])
    # A gathered call's cells are its own: no old route reads gathered rows, nor did a receipt time one. A
    # kernel gradient's tables (`tgmm_call`'s) are its fifth argument, a product's (`gmm_call`'s) its eighth.
    tabled = 4 if launches is _TGMMS else 7
    gathered = len(arguments) > tabled and arguments[tabled] is not None
    scattered = launches is not _TGMMS and len(arguments) > 8 and arguments[8] is not None
    call = base.call(launches, *arguments, **options)
    matrix = _matrix_of(launches, arguments) if options['dense'] else None
    timed = matrix in MEASURED.get((generation.family(call.target.generation), made), ())
    kind = Kind(bool(options['dense']), made, fp8=fp8, gathered=gathered, scattered=scattered,
                matrix=matrix if timed else None)
    cuda = generation.capability(call.target.generation)
    floor = cuda is not None and cuda[0] * 10 + cuda[1] >= generation.BF16_GPU
    placed = generation.tpu(call.target) or floor
    paired = any(bound.arguments.get('paired') is not None for _, bound in call.bound.values())
    rows = _rows_of(launches, arguments)
    return GroupedCase(placed, whole, product, rows, call, kind, paired=paired)


def _matrix_of(launches, arguments) -> tuple[int, int]:
    """A dense product's matrix as it lies (`Kind.matrix`): a forward's kernel `[1, in, out]`, its input
    gradient's the same matrix read transposed, its kernel gradient's output (its rows' width by its
    cotangent's), one tuple for a projection's three products."""
    if launches is _TGMMS:
        return arguments[0].shape[1], arguments[1].shape[1]
    return tuple(arguments[1].shape[1:])


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
