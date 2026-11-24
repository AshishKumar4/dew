"""Dew: a grouped product's CUTLASS launches (`CutlassGmm`, `CutlassTgmm`: the dew-cutlass wheel's grouped
GEMM and weight gradient as XLA FFI calls), `grouped_matmul`'s, and the record of the kernel its build holds
(`cutlass_tile`, `cutlass_built_for`)."""

from __future__ import annotations

import dataclasses
import importlib
from typing import ClassVar

import jax
import jax.numpy as jnp

from dew.nn import sharding

from . import generation
from .grouped_launch import _GroupedLaunch
from .templates.axes import GROUPS
from .templates.gemm import tiling
from .templates.gemm.hooks import NO_HOOKS, Hooks


def _cutlass_call(target: int, out: jax.ShapeDtypeStruct, *operands, **attributes) -> jax.Array:
    """dew_cutlass's FFI call of the `target`-th of the targets the wheel names (`dew_cutlass.TARGETS`: its
    grouped GEMM, then its weight gradient) on `operands`, giving `out`, a mapped call running its maps in
    turn, one launch each (row 4389: JAX's FFI batching refuses a call that states no rule). Under a
    checked `shard_map` its operands vary alike and its result varies as they do (`sharding.varied_alike`,
    `sharding.computed_from`): an FFI call requires the first and states its result's type as given, so a
    layer's matrices sliced at each device's own layer (`CutlassGmm._fwd`) were typed the same on every
    device (#3976's review P0)."""
    name = importlib.import_module('dew_cutlass').TARGETS[target]
    call = jax.ffi.ffi_call(name, sharding.computed_from(out, *operands), vmap_method='sequential')
    return call(*sharding.varied_alike(operands), **attributes)


def cutlass_tile(architecture: str, out_dtype) -> tiling.Tiles:
    """Dew's record of the kernel the installed dew_cutlass instantiates for `architecture` writing
    `out_dtype`, as Dew's records name a launch (row 4323): the build's own statement of its compiled types
    (`dew_cutlass.KERNELS`: its tile, stages, consumer threads, cluster rows and instruction family), a
    fact of that binary, not a derivation. Its kernels band no column tiles."""
    from .templates import resources

    kernel = importlib.import_module('dew_cutlass').KERNELS[(architecture, jnp.dtype(out_dtype).name)]
    rows, columns, depth = kernel.tile
    return tiling.Tiles(rows, columns, depth, kernel.stages, 1,
                            warpgroups=kernel.consumers // resources.THREADS, cluster=kernel.cluster,
                            family=kernel.instructions)


def cutlass_built_for(generation_name: str) -> str | None:
    """The generation of the installed dew_cutlass's kernels a GPU of `generation_name` runs, as the wheel
    dispatches on its compute capability (`dew_cutlass.architecture`, the one statement of it); None where
    the wheel is not installed or runs none there."""
    return generation.cutlass_built().get(generation_name)


class _Cutlass(_GroupedLaunch[tiling.Tiles]):
    """CUTLASS's grouped GEMMs (`kernels/cutlass`, the dew-cutlass wheel) as XLA FFI calls: on Ampere and
    every 8.x part, CUTLASS 2's grouped kernel; on Hopper, CUTLASS 3's pointer-array cooperative kernel on a
    pair of multiprocessors; bf16 operands and bf16 or fp32 out, each group's shape set up on the device. A
    vendored production route on the cells where it measured faster than the template (`DECLARED`), and by
    name elsewhere, at the one kernel its build holds for the GPU and output (`tile`, the build's own record
    of it, `cutlass_tile`)."""
    name: ClassVar[str] = 'cutlass'
    config_cls: ClassVar[type[tiling.Tiles]] = tiling.Tiles

    def eligible(self, ba, target) -> bool:
        return cutlass_built_for(target.generation) is not None and self.in_contract(ba)

    def tile(self, ba) -> tiling.Tiles | None:
        """The record of the kernel the build holds for the GPU being selected for (`generation.gpu_target`,
        `cutlass_built_for`) writing the call's output (`cutlass_tile`); None where it holds none."""
        built = cutlass_built_for(generation.gpu_target().generation)
        return None if built is None else cutlass_tile(built, ba.arguments['out_dtype'])

    @staticmethod
    def in_contract(ba) -> bool:
        """Whether its kernels take the call wherever dew_cutlass holds them (`eligible` adds the installed
        library's generations): bf16 operands, bf16 or fp32 out, int32 sizes, every extent a multiple of 8
        elements (16-byte aligned rows of every operand and output, as its TMA copies and each group's
        pointers need), no epilogue or exact sums, compiled (no interpreter). A stack's layer it reads as a
        slice of its matrices, and rows a prologue forms (its sides by row or depth, none by group) as XLA
        forms them before the call (`_formed`)."""
        hooks = ba.arguments['hooks']
        operands = [value for name, value in ba.arguments.items()
                    if name in ('tokens', 'kernel', 'cotangent')]
        extents = {extent for operand in operands for extent in operand.shape[1:]}
        return (all(operand.dtype == jnp.bfloat16 for operand in operands)
                and jnp.dtype(ba.arguments['out_dtype']) in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32))
                and ba.arguments['sizes'].dtype == jnp.int32 and all(extent % 8 == 0 for extent in extents)
                and hooks.epilogue is None and hooks.out_dtypes is None
                and not any(GROUPS in axes for axes in hooks.prologue_axes)
                and not ba.arguments.get('exact_sums')
                and not ba.arguments['interpret'])

    def space(self, ba) -> tuple[tiling.Tiles, ...]:
        tile = self.tile(ba)
        return () if tile is None else (tile,)

    def offering(self, ba, target) -> tuple[tiling.Tiles, ...]:
        # Its build's kernel is the one it runs (`cutlass_tile`), sized to every part of its architecture.
        return self.space(ba)

    def legal(self, ba, config: tiling.Tiles, target) -> bool:
        """Whether `config` runs the call: its build's one kernel (`space`), which CUTLASS sized to every part
        of its architecture, its shared memory and registers its own; a pin or a tuned pick names it."""
        del target
        return config in self.space(ba)


@dataclasses.dataclass(frozen=True)
def _formed(tokens: jax.Array, prologue_operands: tuple, hooks: Hooks) -> jax.Array:
    """The rows a CUTLASS launch multiplies: `tokens` through the call's prologue where it has one, formed by
    XLA before the call in the tokens' dtype, as `gemm`'s portable program forms them (the copy a call naming
    CUTLASS takes where the template forms them tile by tile: an input-scaled MoE's rows times their routing
    weights, models-28's review). Rows read through tables hand on before a CUTLASS launch
    (`grouped_matmul`'s 'tables' need)."""
    if hooks.prologue is None:
        return tokens
    return hooks.before()(tokens, *prologue_operands).astype(tokens.dtype)


class CutlassGmm(_Cutlass):
    """`Gmm` through CUTLASS (the wheel's grouped GEMM target, `dew_cutlass.TARGETS`)."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens',)

    def _fwd(self, tokens: jax.Array, kernel: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), epilogue_operands: tuple = (), paired: jax.Array | None = None,
             first: jax.Array | None = None, gather: jax.Array | None = None,
             scatter: jax.Array | None = None, joined: jax.Array | None = None, beside: tuple = (),
             chained: tuple = (), *, out_dtype: str,
             trans_rhs: bool, interpret: bool,
             hooks: Hooks = NO_HOOKS, exact_sums: bool = False,
             dense: bool = False, config: tiling.Tiles, return_residuals: bool = False):
        # No epilogue, second matrix or exact sums: the launch is not eligible for them (`_Cutlass.eligible`);
        # a dense call is one group, as any.
        del epilogue_operands, paired, gather, scatter, interpret, exact_sums, dense, config, joined, beside
        del chained
        tokens = _formed(tokens, prologue_operands, hooks)
        if first is not None:
            # A stacked layer's call (a scanned stack's, `gemm._grouped`) reads its groups' matrices from the
            # stack's `first`: the library takes one array of them, so it gets the layer's as a slice, the
            # copy a stack read whole made before the scanned default (row 4074's review).
            kernel = jax.lax.dynamic_slice_in_dim(kernel, first, sizes.shape[0])
        importlib.import_module('dew_cutlass').registered()
        width = kernel.shape[1] if trans_rhs else kernel.shape[2]
        out = jax.ShapeDtypeStruct((tokens.shape[0], width), jnp.dtype(out_dtype))
        return _cutlass_call(0, out, tokens, kernel, sizes, transpose_w=trans_rhs), None


@dataclasses.dataclass(frozen=True)
class CutlassTgmm(_Cutlass):
    """`Tgmm` through CUTLASS (the wheel's weight-gradient target, `dew_cutlass.TARGETS`): each group's
    gradient a GEMM over its rows, an empty group's zero, its pair splitting the gradient's rows."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens', 'cotangent')

    def tile(self, ba) -> tiling.Tiles | None:
        # Dew's record of a kernel gradient pairs no row tiles (`tiling.Tiles.cluster`: `tgmm` takes no
        # pair), whatever cluster the binary's kernel runs on.
        tile = super().tile(ba)
        return None if tile is None else dataclasses.replace(tile, cluster=1)

    def _fwd(self, tokens: jax.Array, cotangent: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), gather: tuple | None = None, *, out_dtype: str, interpret: bool,
             hooks: Hooks = NO_HOOKS,
             exact_sums: bool = False, dense: bool = False, config: tiling.Tiles,
             return_residuals: bool = False):
        del gather, interpret, exact_sums, dense, config  # as `CutlassGmm._fwd`'s
        tokens = _formed(tokens, prologue_operands, hooks)
        importlib.import_module('dew_cutlass').registered()
        out = jax.ShapeDtypeStruct((sizes.shape[0], tokens.shape[1], cotangent.shape[1]),
                                   jnp.dtype(out_dtype))
        return _cutlass_call(1, out, tokens, cotangent, sizes), None
