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

from __future__ import annotations

import abc
import contextlib
import contextvars
import dataclasses
import functools
import importlib
import importlib.util
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, Protocol

import jax
import jax.numpy as jnp

from . import generation, selection
from .generation import INHERENT
from .grouped_cells import (
    B200_GRID,
    EM35D,
    M_EM_36,
    M_EM_52,
    M_EM_65,
    M_EM_67,
    MEASURED,
    NORMED_08B,
    NORMED_4B_8B,
    TOGETHER,
    Kind,
    Made,
)
from .grouped_launch import INTERPRETED, _GroupedLaunch
from .grouped_triton import Gmm, Tgmm, _gradient_sides, _product_sides
from .templates import base
from .templates.gemm import packed, tiling
from .templates.gemm.hooks import NO_HOOKS, Hooks
from .templates.product import SCALE_DEPTH, Product, eight

if TYPE_CHECKING:
    import tokamax

    from .templates import mosaic, resources
    from .templates.gemm import mosaic_gpu, pricing


def _stored_bytes(ba, rows: int, width: int) -> float:
    """The bytes a product's epilogue moves an element of its `rows` x `width` output
    (`pricing.split_walk`'s `stored`): each output's dtype's, and its epilogue operands' bytes over the
    output's elements (an [rows, columns] operand's an element, a column vector's a row's share)."""
    from .templates import cost

    hooks = ba.arguments['hooks']
    written = sum(jnp.dtype(dtype).itemsize for dtype in hooks.out_dtypes or (ba.arguments['out_dtype'],))
    return written + cost.stored(*ba.arguments.get('epilogue_operands', ())) / max(rows * width, 1)


def _matrix_extents(ba) -> tuple[int, int]:
    """A bound product call's depth and width as its matrix lies (`trans_rhs`'s `[width, depth]`), its width
    every matrix's side by side with the kernel's (`gmm_call`'s `beside`)."""
    kernel, transposed = ba.arguments['kernel'], ba.arguments['trans_rhs']
    depth, width = (kernel.shape[2], kernel.shape[1]) if transposed else kernel.shape[1:]
    return depth, width + sum(each.shape[1 if transposed else 2] for each in ba.arguments.get('beside') or ())


_PaddedArgument = tuple[str, "jax.Array | jax.ShapeDtypeStruct | tuple", "dict | tuple"]
"""An argument `_Mosaic._aligned` pads: its name, its value (or tuple of them) and its dims (`_grown`'s)."""


class _Mosaic(_GroupedLaunch[tiling.Tiles]):
    """JAX's Pallas Mosaic GPU kernels (`templates.gemm.mosaic_gpu`) on their instruction family's bundle
    (Hopper's warpgroup MMA, Ampere's mma.sync), one 16-bit dtype, their one tile generated from the
    family's facts, the device's and the call's extents (`derivation.derived`). Each runs one warpgroup a
    multiprocessor, or two consumers beside a producer (a cooperative tile), persistent over the output
    tiles in a planar snake, so the rows a band of `grid_block_n` column tiles shares and the matrix block
    the row tiles in flight share come from the L2."""
    name: ClassVar[str] = 'mosaic'
    config_cls: ClassVar[type[tiling.Tiles]] = tiling.Tiles
    GENERATES: ClassVar[bool] = True
    EXACT_SUMS: ClassVar[bool] = False
    """Whether the kernel adds each depth step's product apart (`exact_sums`); one that does not is not
    eligible where the call asks for it."""
    KERNEL: ClassVar[str] = tiling.GMM
    """Its kernel's name, which its tile's producer registers are kept by (`tiling.Tiles.producers`)."""
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

    def registers(self, ba, config: tiling.Tiles, target=None) -> int | None:
        return self._tail(ba, config, generation.gpu_target() if target is None else target)[0]

    def balanced(self, ba, config: tiling.Tiles, target=None) -> bool:
        """Whether the launch at `config` walks its last wave balanced over the programs (`tail.balanced`),
        each merging a slice of another program's partial beside its own (`mosaic.Family.fixed_up`): where
        its schedule would (`pricing.merges`: a dense call of one product, its tile alone on its
        multiprocessor and its depth unsplit, on a family fixing tiles up) and its registers hold the merge
        as its placement holds them (`_holds`). A tile that cannot holds its accumulator unbalanced: an A100's
        128 x 128 mma.sync tile, 280 registers a thread merging past the 255 and 216 unbalanced, ran
        1.07-1.09x cuBLAS's time on four dense products where the 192 x 64 tile the merge left its derivation
        ran 1.32-2.37x (M-FS-36, row 5271)."""
        return self._tail(ba, config, generation.gpu_target() if target is None else target)[1] is not None

    def _tail(self, ba, config: tiling.Tiles, target) -> tuple[int | None, object]:
        """A consumer thread's registers at `config` and its balanced `Tail` plan (or None)."""
        ran, tile = self._ran(ba, config, target)
        plan = None if config.splits > 1 or config.swapped else self._tail_plan(ran, tile, target)
        if plan is not None:
            merging = self._registers(ran, tile, merging=True, target=target)
            if self._holds(ran, tile, merging):
                return merging, plan
        return self._registers(ran, tile, merging=False, target=target), None

    def _tail_programs(self, ba, config: tiling.Tiles, target) -> int:
        return (self._programs(config, target) if self.STAGED_ROWS
                else self._launched(config, bool(ba.arguments['interpret']), target))

    def _tail_plan(self, ran, config: tiling.Tiles, target):
        from .templates.gemm import pricing

        call = ran.arguments
        dense = (bool(call.get('dense')) and call.get('paired') is None
                 and not call.get('beside') and not self._scaled)
        dtype = call['tokens'].dtype
        if not pricing.merges(config, dtype, dense=dense) or config.swapped:
            return None
        m, depth, n = self._generated_extents(ran)
        tokens = call['tokens']
        rhs = call['cotangent' if self.STAGED_ROWS else 'kernel']
        k = tokens.shape[0] if self.STAGED_ROWS else depth
        assert k is not None
        itemsizes = (jnp.dtype(tokens.dtype).itemsize, jnp.dtype(rhs.dtype).itemsize)
        return pricing.dense_tail(config, dtype, (m, n, k), itemsizes, target,
                                  self._tail_programs(ran, config, target), dense=dense)

    def _merges(self, ba, config: tiling.Tiles, target) -> bool:
        ran, tile = self._ran(ba, config, target)
        return config.splits == 1 and not config.swapped and self._tail_plan(ran, tile, target) is not None

    def _holds(self, ba, config: tiling.Tiles, registers: int | None) -> bool:
        """Whether a consumer thread's `registers` hold at `config`, the call as it runs (`_ran`), as its
        placement holds them (`resources.plan`): within its family's cap at its consumer roles and producer,
        at its residency."""
        from .templates.gemm import pricing

        family = pricing.family_of(config, ba.arguments['tokens'].dtype)
        if registers is None or not family.budgets_registers:
            return True
        roles, producer = config.warpgroups + self.rewritten(ba, family), config.producers.of(self.KERNEL)
        return (registers <= family.registers(roles, producer=producer)
                and config.programs <= family.programs(roles, registers, producer))

    def _registers(self, ba, config: tiling.Tiles, *, merging: bool, target) -> int | None:
        # A consumer thread's live set (`resources.tile_registers`: the accumulator it carries, a step's
        # fragments, an output's copy), and beside the accumulator the larger of its phases, each done
        # before the next starts: the rows' stage a kernel gradient's consumer prepares itself, and the
        # epilogue's side values (`_epilogue_registers`); under exact sums the running sum beside each
        # step's own product. Each hook is priced by what it materializes (`_prologue_registers`,
        # `_epilogue_registers`); a run's compile steps a spilling tile down (`report.compiled`). The
        # compile gate tests the actual
        # launch and residency caps: ptxas's spill decisions are not monotone, so a least spill-free cap
        # cannot decide a launch.
        from .templates import across, resources, stores, swizzling
        from .templates.gemm import pricing

        dtype = ba.arguments['tokens'].dtype
        family = pricing.family_of(config, dtype)
        # The tile's columns its accumulators hold: a pair's both where its products span it.
        columns = pricing.tile_columns(config, family)
        block = (config.block_m, columns, config.block_k)
        if not family.budgets_registers or generation.tpu(target):
            # A TPU core states none, whatever tile a pin names: its values are Mosaic TPU's, in VMEM
            # (`mosaic.Issuer.value_scratch`).
            return None
        hooks = ba.arguments['hooks']
        stored = hooks.out_dtypes or (ba.arguments['out_dtype'],)
        plain = resources.tile_registers(family, config.sharing, block, dtype, stored=stored,
                                        rhs_transposed=self._matrix_transposed(ba),
                                        lhs_retained=ba.arguments.get('paired') is not None)
        plain += pricing.prefetch_registers(family, config, ba=ba, staged_rows=self.STAGED_ROWS,
                                            unstaged=self.unstaged(ba), rewritten=self.rewritten(ba, family))
        plain += pricing.bounded_registers(family, config, dtype, dense=bool(ba.arguments.get('dense')),
                                           staged_rows=self.STAGED_ROWS)
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
        if self.STAGED_ROWS and not family.operands_in_registers and not family.tensor_columns(columns, 1):
            chunk = stores.staged_columns(columns, stored)
            tgmm_tiles = [((1,), jnp.int32)] * 6 + [((config.share, chunk), stored[0])]
            plain += family.scalar_registers(tgmm_tiles) - family.scalar_registers()
        if self.STAGED_ROWS and not family.operands_in_registers and not self.unstaged(ba):
            words = -(-config.block_k * config.block_m // config.warpgroups * jnp.dtype(dtype).itemsize
                      // 4 // resources.THREADS)
            atom = (config.block_k, swizzling.swizzle_columns(dtype, swizzling.SWIZZLE_BYTES))
            tiles = [((config.block_k, config.block_m // config.warpgroups), dtype)] + [(atom, dtype)] * 3
            rewritten = family.scalar_registers(tiles) - family.scalar_registers()
            staged = max(0, words - family.scalar_registers()) + rewritten
        prologue = self._prologue_registers(ba, config)
        prologue += rewritten if prologue else 0
        # A last wave balanced over the programs (`balanced`) merges a slice of another program's partial at a
        # time beside its own (`mosaic.Family.fixed_up`, `across.merge_spans`).
        merged = min(accumulator, across.MERGED_REGISTERS) if merging else 0
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

    def _prologue_registers(self, ba, config: tiling.Tiles) -> int:
        """A prologue's transform of a consumer's share of the rows' stage: that share and each side tile of
        its shape widened to fp32 as the hook computes them, and its result in the stage's dtype (HOOKS1,
        sm90: a norm's +46..53 and a cast's +47 at a 32-word share, SwiGLU's gate tile +87..91). Vector
        sides are a share's broadcasts, held within it. The forward's products take the result from
        registers (`mosaic_gpu._gmm_step`), each step's held until its products are done, so beside it each
        step its family keeps in flight (`Family.in_flight`) still holds its own; the kernel gradient
        rewrites its stage (`STAGED_ROWS`) and holds none; a family's own role rewriting it
        (`_prologue_rewritten`) transforms a role's rows of the stage at a time (`mosaic_gpu._prologue_stage`)
        and holds none past it."""
        from .templates import resources
        from .templates.axes import held as held_axes
        from .templates.gemm import pricing

        hooks = ba.arguments['hooks']
        if hooks.prologue is None:
            return 0
        tiles = sum(len(held_axes(axes)) > 1 for axes in hooks.prologue_axes)
        stored = jnp.dtype(ba.arguments['tokens'].dtype).itemsize
        family = pricing.family_of(config, ba.arguments['tokens'].dtype)
        rewritten = self._prologue_rewritten(ba, family)
        share = (family.rewriting.role_rows if rewritten else config.share) * config.block_k
        in_flight = 0 if self.STAGED_ROWS or rewritten else family.in_flight
        # A family whose products read registers casts the result into its left operand's fragments
        # (`MmaIssuer._left`) beside the result itself: a cast forward's least spill-free cap on sm80 was 136
        # at 64 x 128 x 64, its model 127 without the cast's copy (EM-least4).
        copies = 1 + in_flight + (family.operands_in_registers and not self.STAGED_ROWS)
        return -(-share * ((1 + tiles) * 4 + copies * stored) // (4 * resources.THREADS))

    @staticmethod
    def _epilogue_registers(ba, config: tiling.Tiles, stored: tuple) -> int:
        """The epilogue's values beyond the mainloop's held set, delegated to `pricing.epilogue_registers`
        (row 11982)."""
        from .templates.axes import held as held_axes
        from .templates.gemm import pricing

        hooks = ba.arguments['hooks']
        family = pricing.family_of(config, ba.arguments['tokens'].dtype)
        columns = pricing.tile_columns(config, family)
        if hooks.epilogue is None:
            return pricing.epilogue_registers(family, config.share, columns, stored, has_hook=False)
        vectors = sum(family.column_registers(columns, side.dtype)
                      for side, axes in zip(ba.arguments.get('epilogue_operands', ()), hooks.epilogue_axes,
                                            strict=True) if held_axes(axes) == ('columns',))
        vectors += family.column_registers(columns, jnp.float32) if hooks.accumulators > 1 else 0
        tile_count = sum(len(held_axes(axes)) > 1 for axes in hooks.epilogue_axes)
        out_dtypes = hooks.out_dtypes or stored
        return pricing.epilogue_registers(family, config.share, columns, out_dtypes,
                                          vectors=vectors, tile_count=tile_count, has_hook=True)

    def register_budget(self, config: tiling.Tiles) -> int:
        # Its family's budget a consumer, whatever the operands' dtype (`mosaic.Family.registers`); none on a
        # core that budgets no registers.
        from .templates.gemm import pricing

        family = pricing.family_of(config, jnp.bfloat16)
        return family.registers(config.warpgroups) if family.budgets_registers else 0

    def _granular(self, ba, config: tiling.Tiles, family: mosaic.Family, target) -> bool:
        """Whether `config` meets `family`'s granularity, residency, and TMEM budget (`pricing.granular_tile`,
        row 11982)."""
        from .templates.gemm import pricing

        return pricing.granular_tile(
            family, config, ba.arguments['tokens'].dtype, target=target, steps=self.STEPS,
            accumulators=1 if ba.arguments.get('paired') is None else 2,
            exact_sums=bool(ba.arguments.get('exact_sums')),
            stream_blocks=lambda: self._allocations(ba, config, family).blocks)

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
        from .templates import bundles

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
        prepared = ((ba.arguments['hooks'].prologue is not None and not self._prologue_rewritten(ba, family))
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
        `derivation.derived` reads them."""

    def derived(self, ba, target) -> tiling.Tiles | None:
        from .templates.gemm import producers
        tile = self._tile(ba, target)
        if tile is None:
            return None
        with generation.as_target(target):
            return producers.settled(self, ba, tile)

    def _tile(self, ba, target) -> tiling.Tiles | None:
        return self._derived(ba, target)

    def completed(self, ba, config):
        from .templates.gemm import producers
        return producers.started(self, ba, config) if isinstance(config, tiling.Tiles) else config

    def rungs(self, ba, config):
        from .templates.gemm import producers
        return producers.rungs(self, ba, config) if isinstance(config, tiling.Tiles) else iter(())

    def pinnable(self, ba, config) -> bool:
        from .templates import report

        return super().pinnable(ba, config) and not report.refused(self, ba, config, self.launch_target())

    def _refused(self, ba, config: tiling.Tiles, target) -> bool:
        # Refused on `target` past its producer's ladder: spilling at the rung it settles on
        # (`producers.settled`, settled under `target` until it takes one, row 11885).
        from .templates import report
        from .templates.gemm import producers
        with generation.as_target(target):
            return report.refused(self, ba, producers.settled(self, ba, config), target)

    def _derived(self, ba, target, admits=None) -> tiling.Tiles | None:
        """The derivation's tile of `ba` on `target` (`derivation.derived`), each candidate also one `admits`
        takes where given (a swapped call's own rule, `MosaicGmm._left_fits`), on the call padded to whole
        tiles (`_aligned`'s `whole_tiles`): the tile a launch runs whether its depth and width are padded or
        run partial (`_partial_axes`)."""
        from .templates.gemm import derivation

        ba = self._aligned(ba, target, whole_tiles=True)
        rows, depth, width = self._generated_extents(ba)
        # Each issuer's group in turn (`_groups`), a pair's layouts first where the rows hold its tile
        # (`derivation.paired_first`), each its own derivation, the first that holds a tile taken.
        for tried in (tried for layouts in self._groups(ba, target)
                      for tried in derivation.paired_first(layouts, rows)):
            tile = derivation.derived(tried, rows=rows, depth=depth, width=width,
                                      dtype=ba.arguments['tokens'].dtype,
                                      multiprocessors=_multiprocessors(target), target=target,
                                      candidate=lambda tile: (self.candidate(ba, tile, target)
                                                              and (admits is None or admits(tile))),
                                      launch_of=lambda tile: self.launch_of(ba, tile, target),
                                      grouped=depth is not None, depths=self.STEPS,
                                      dense=bool(ba.arguments.get('dense')),
                                      ragged=self._ragged(ba, tried[0]),
                                      units=self._vector_units(ba, tried[0]))
            if tile is not None:
                return tile
        return None

    def _vector_units(self, ba, family: mosaic.Family) -> tuple[int, int, int]:
        """The units a tile's rows, columns and step take from the call's side vectors (`candidate`'s, each
        block along a vector's axis whole `mosaic.Transport.vector_unit`s), which the derivation shapes its
        tiles in (`derivation.derived`'s `units`)."""
        from .templates.axes import held as held_axes

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
        """Whether the launch's tiles may leave its output's rows a partial last tile (`derivation.derived`'s
        `ragged`): none but a kernel gradient's (`MosaicTgmm`)."""
        del ba, family
        return False

    def neighbours(self, ba, config: tiling.Tiles) -> tuple[tiling.Tiles, ...]:
        # The derived tile with one of its choices moved a step either way: a stage more or fewer, the
        # band halved or doubled, the columns or the rows the next block either side (a column unit, a
        # role of rows).
        from .templates.gemm import pricing

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

    def legal(self, ba, config: tiling.Tiles, target) -> bool:
        """Whether `config` runs the call on `target` (`Template.legal`): a candidate (`candidate`) whose
        ring of `config.stages` and registers the planner holds at its programs (`resources.plan`, as the
        derivation sizes them). A cluster's legality is each multiprocessor's: its staged bytes and
        registers are a tile's alone. A tile is judged as the call it runs (`_ran`)."""
        return self._held(ba, config, target, self.candidate)

    def _held(self, ba, config: tiling.Tiles, target, candidate: Callable) -> bool:
        """`legal`'s judgment with `candidate` its candidate rule: the call as `config` runs it (`_ran`), a
        candidate whose ring and registers the planner holds."""
        from .templates import resources
        from .templates.gemm import pricing

        ran, tile = self._ran(ba, config, target)
        if not candidate(ran, tile, target):
            return False
        family = pricing.family_of(tile, ran.arguments['tokens'].dtype)
        launch = dataclasses.replace(self._launch(ran, tile, target,
                                                  registers=self.registers(ba, config, target)),
                                     floor=tile.stages, most=tile.stages)
        return resources.plan([launch], family, target, programs=tile.programs) is not None

    def _stated(self, ba, config: tiling.Tiles, target) -> dict:
        """The register model and planned balanced tail (`Tail` or False) a launch states its mainloop."""
        registers, plan = self._tail(ba, config, target)
        return {'registers': registers, 'balanced': False if plan is None else plan,
                'shared': registers and self.staged_bytes(ba, config, target)}

    def launch_of(self, ba, config: tiling.Tiles, target=None) -> resources.Launch:
        """The launch the planner sizes at `config`'s shape (`resources.plan`): its allocations at a ring
        depth (`_allocations`), its consumers, its registers a consumer thread (`registers`), and at most
        the stages its program's share of the GPU's block (`generation.shared_budget`) holds of a step's
        two operand blocks at an element a byte and its contraction has steps, at least two: the call's a
        launch at `config` runs (`_ran`)."""
        target = generation.gpu_target() if target is None else target
        ran, tile = self._ran(ba, config, target)
        return self._launch(ran, tile, target, registers=self.registers(ba, config, target))

    def _launch(self, ba, config: tiling.Tiles, target, *, registers: int | None) -> resources.Launch:
        """`launch_of`'s launch for the call `ba` as a launch at `config` runs it (`_ran`), a consumer thread
        at `registers`, the register model's for the call as given (`registers`)."""
        from .templates import resources
        from .templates.gemm import pricing

        family = pricing.family_of(config, ba.arguments['tokens'].dtype)
        steps = self._steps(ba, config)
        scope = self._scope(ba, config, family, steps)
        shared = resources.program_shared(generation.shared_budget(target), config.programs)
        most = shared // ((config.block_m + config.block_n) * config.block_k)
        if steps is not None:
            # A ring deeper than its contraction's steps holds no more of them.
            most = min(most, max(2, steps))
        # A family fixing its rings' slots streams them whatever the stages (`mosaic.Transport.ring_slots`).
        most = family.ring_slots or most
        # A family rewriting the staged rows in a role of its own runs it beside the consumers
        # (`mosaic_gpu.tgmm`'s `roles`).
        roles = config.warpgroups + self.rewritten(ba, family)
        # Its producer at the registers its tile states, as its kernel lowers it: a derived tile's its
        # ladder's first rung (`producers.started`), a pin's as pinned.
        return resources.Launch(scope, roles, registers, most=most, producer=config.producers.of(self.KERNEL))

    def rewritten(self, ba, family: mosaic.Family) -> bool:
        """Whether the launch's staged rows are rewritten by `family`'s own role (`rewrites_in_persist`): its
        own loads' rows, gathered or not (`mosaic_gpu.rewrites`), never rows read as loaded (`unstaged`); or a
        product's rows its prologue rewrites there (`_prologue_rewritten`)."""
        from .templates.gemm import mosaic_gpu

        return ((self.STAGED_ROWS and mosaic_gpu.rewrites(family, self.K_MAJOR, self.unstaged(ba)))
                or self._prologue_rewritten(ba, family))

    def _prologue_rewritten(self, ba, family: mosaic.Family) -> bool:
        """Whether a product's prologue rewrites its rows stage in `family`'s own role
        (`mosaic_gpu.prologue_rewritten`: tcgen05's rewriting warpgroup); a kernel gradient's prologue, whose
        consumers prepare their halves of each step's rows (`STAGED_ROWS`), never."""
        from .templates.gemm import mosaic_gpu

        return not self.STAGED_ROWS and mosaic_gpu.prologue_rewritten(family, ba.arguments['hooks'].prologue)

    def unstaged(self, ba) -> bool:
        """Whether a kernel gradient's rows stage is read as loaded (`mosaic_gpu.rows_unstaged`): a dense call
        (or its split's runs, `whole_steps`) of no prologue or table, laid rows-major, which no consumer or
        role prepares."""
        from .templates.gemm import mosaic_gpu

        tabled = ba.arguments.get('gather') is not None or ba.arguments.get('scatter') is not None
        whole = bool(ba.arguments.get('dense')) or bool(ba.arguments.get('whole_steps'))
        return (self.STAGED_ROWS and not self.K_MAJOR
                and mosaic_gpu.rows_unstaged(dense=whole,
                                             prologue=ba.arguments['hooks'].prologue, tabled=tabled))

    def candidate(self, ba, config: tiling.Tiles, target) -> bool:
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
        from .templates.axes import held as held_axes
        from .templates.gemm import mosaic_gpu
        family = next((bundle for group in self._groups(ba, target) for bundle in group
                       if bundle.name == config.family), None)
        if (family is None or config.cluster > family.cluster or not mosaic_gpu.pair_takes(config, family)
                or (family.pairs_product and ba.arguments['interpret'])
                or self._pair_refuses(ba, config, family) or self._pingpong_refuses(config, family)
                or not self._divides(ba, config, target) or self._refused(ba, config, target)):
            return False
        if not self._granular(ba, config, family, target):
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

    def _pingpong_refuses(self, config: tiling.Tiles, family: mosaic.Family) -> bool:
        """Whether a ping-pong tile (`Tiles.pingpong`) is one the launch does not run: on a family that takes
        none (`Family.takes_pingpong`: a producer of its own loads every tile), a kernel gradient's (its
        consumers prepare their halves of each step's rows stage, `STAGED_ROWS`), a scaled or swapped call
        (a packed decode's buffers and a swapped tile's rows are each consumer's half)."""
        return config.pingpong and (not family.takes_pingpong or self.STAGED_ROWS or self._scaled
                                    or config.swapped)

    def _pair_refuses(self, ba, config: tiling.Tiles, family: mosaic.Family) -> bool:
        """Whether a pair's product refuses the call at `config` (`mosaic_gpu.pair_refuses`, the statement
        `gmm`, `tgmm` refuse by): its prologue, scales, gathered rows, exact sums, rewritten stage."""
        from .templates.gemm import mosaic_gpu

        return mosaic_gpu.pair_refuses(config, family, transpose_lhs=False,
                                       exact_sums=bool(ba.arguments.get('exact_sums')), scaled=self._scaled,
                                       parts=(ba.arguments['hooks'].prologue, ba.arguments.get('gather')),
                                       rewritten=self.rewritten(ba, family))

    def _ran(self, ba, config: tiling.Tiles, target) -> tuple:
        """The call and tile a launch at `config` runs, which its legality, registers and launch read: the
        call as it runs it (`_aligned`) by default."""
        return self._aligned(ba, target), config

    def _aligned(self, ba, target, *, whole_tiles: bool = False):
        """The call as the launch runs it (`_fwd`'s `_padded`): its depth and width each padded with zeros to
        whole units of its family's tiles where they are not (its depth to the least step,
        `Family.depths`, or a tile's rows where the depth is a tile's; its width to whole column units,
        `Family.column_unit`), the output sliced back; the call itself where they are. A width of 100
        columns, which no tile divides, so runs on 128, the template's where Triton's masked tiles ran it.
        An axis the launch runs with its last step or column tile partial (`_partial_axes`: TMA's bounded
        loads, row 5299) is not padded, so no copy is made of its operands; `whole_tiles`, every axis padded,
        the call the tile is derived on (`_derived`)."""
        layouts = self._families(ba, target)
        if not layouts:
            return ba
        family = layouts[0]
        extents, whole = self._whole_extents(ba, family)
        if ba.arguments.get('chained'):
            # Chained products meet at whole steps (`chained_refusal`): their depths run as bound, neither
            # padded nor partial, so selection and the launch judge the same operands (c3b97e20f F3: a first
            # depth of 80 padded to 128 was derived on, and refused as bound).
            whole = {**whole, 'depth': extents['depth']}
        if whole == extents:
            return ba
        depth_partial, width_partial = ((False, False) if whole_tiles
                                        else self._partial_axes(ba, family, target))
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

    def _partial_axes(self, ba, family: mosaic.Family, target) -> tuple[bool, bool]:
        """Whether the call's depth, and its width, run their last step and column tile partial on `family`
        rather than padded (`_aligned`): a product's alone (`MosaicGmm._partial_axes`)."""
        del ba, family, target
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
    def _divides(self, ba, config: tiling.Tiles, target) -> bool:
        """Whether `config`'s blocks divide the call's extents as the kernel requires."""

    @staticmethod
    def _hooks_lower(ba, dtype, target) -> bool:
        """Whether the target's Mosaic lowering takes the launch's hooks (`hooks.hooks_lower`). A side operand
        the launch squeezes comes back broadcast across its tile (`_unsqueezed`)."""
        from .templates.gemm.hooks import hooks_lower

        return hooks_lower(ba, dtype, target)

    def staged_bytes(self, ba, config: tiling.Tiles, target) -> int:
        # The kernel's own allocations at `config` (`mosaic_gpu.Allocations`), counted as Mosaic's resource
        # estimate counts them (`resources.shared_bytes`): its streams' stages, its side operands' staging,
        # its outputs' staging as its family stages them, and every barrier (at a block's edge, a Hopper
        # tile's barriers alone took it 128 bytes past). A scoped walk and the output staged after it
        # reuse shared memory; persistent rings and output staging remain live together.
        from .templates import resources
        from .templates.gemm import pricing

        family = pricing.family_of(config, ba.arguments['tokens'].dtype)
        ran, tile = self._ran(ba, config, target)
        ran_family = pricing.family_of(tile, ran.arguments['tokens'].dtype)
        return resources.shared_bytes(family, self._scope(ran, tile, ran_family, self._steps(ran, tile))(
            config.stages))

    def _scope(self, ba, config: tiling.Tiles, family: mosaic.Family, steps: int | None) -> Callable:
        """The launch's allocations at a ring depth (`_allocations`) on `family`, `steps` deep, the call as
        it runs (`_ran`): what `launch_of` plans and `staged_bytes` counts, without the register model."""

        def scope(stages: int) -> resources.Scope:
            staged = dataclasses.replace(config, stages=stages)
            return self._allocations(ba, staged, family).scope(family, staged, steps,
                                                               rewritten=self.rewritten(ba, family))
        return scope

    def _steps(self, ba, config: tiling.Tiles) -> int | None:
        """The steps the call as it runs (`_ran`) takes at `config`: none where an exactly summed walk's are a
        traced count (`mosaic_gpu._gmm_scheduled`'s `traced`), whose pipeline allocates its whole ring however
        few steps the call has."""
        depth = self._generated_extents(ba)[1]
        return None if depth is None or ba.arguments.get('exact_sums') else -(-depth // config.block_k)

    @abc.abstractmethod
    def _allocations(self, ba, config: tiling.Tiles, family: mosaic.Family) -> mosaic_gpu.Allocations:
        """The kernel's allocations at `config` for the call `ba`."""

    BLOCKS: ClassVar[dict[str, str]]
    """Each launch axis's block field of a tile (`extents`, `_vector_units`)."""

    @classmethod
    def extents(cls, tile: tiling.Tiles) -> dict:
        """The block each launch axis takes at `tile`."""
        return {axis: getattr(tile, block) for axis, block in cls.BLOCKS.items()}


    def program_shared(self, config: tiling.Tiles, target) -> int:
        # The programs a multiprocessor runs at once share its shared memory.
        from .templates import resources

        return resources.program_shared(super().program_shared(config, target), config.programs)

    def _programs(self, config: tiling.Tiles, target) -> int:
        """The programs the persistent kernel's schedule spreads its tiles over: the tile's programs on each
        multiprocessor (`Tiles.programs`)."""
        return _multiprocessors(target) * config.programs

    def _launched(self, config: tiling.Tiles, interpret: bool, target) -> int:
        """The programs the persistent kernel launches: its schedule's (`_programs`); the interpreter runs
        one cluster (jax/_src/pallas/mosaic_gpu/interpret), one program alone or a pair's two
        (`Tiles.cluster`)."""
        return config.cluster if interpret else self._programs(config, target)


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

    ROWS: ClassVar[tuple[str, ...]] = ('tokens', 'joined')
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

    def _allocations(self, ba, config: tiling.Tiles, family: mosaic.Family) -> mosaic_gpu.Allocations:
        from .templates.gemm import mosaic_gpu

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
            interpret=bool(ba.arguments['interpret']), joined=ba.arguments.get('joined') is not None)

    def _generated_extents(self, ba) -> tuple[int, int | None, int]:
        # A group's rows on average over the call's groups, which a stack read from its `first` matrix
        # holds fewer of than matrices.
        tokens = ba.arguments['tokens']
        depth, width = _matrix_extents(ba)
        # Chained products' depths joined end to end: one contraction the launch walks (`gmm`'s `chained`).
        depth += sum(rows.shape[1] for rows, _ in ba.arguments.get('chained') or ())
        return -(-tokens.shape[0] // max(ba.arguments['sizes'].shape[0], 1)), depth, width

    def _divides(self, ba, config: tiling.Tiles, target) -> bool:
        # A depth or width the tile does not divide where its last step or column tile runs partial
        # (`_partial_axes`, at the tile `mosaic_gpu.partial_runs`), no deeper or wider than the call.
        from .templates.gemm import mosaic_gpu, pricing

        dtype = ba.arguments['tokens'].dtype
        depth, width = _matrix_extents(ba)
        family = pricing.family_of(config, dtype)
        columns = pricing.tile_columns(config, family)
        runs = mosaic_gpu.partial_runs(config, dtype, ba.arguments['interpret'],
                                       transpose_rhs=ba.arguments['trans_rhs'],
                                       gathered=ba.arguments.get('gather') is not None,
                                       scattered=ba.arguments.get('scatter') is not None)
        depth_partial, width_partial = (each and run for each, run in
                                        zip(self._partial_axes(ba, family, target), runs, strict=True))
        return ((depth % config.block_k == 0 or (depth_partial and config.block_k <= depth))
                and (width % columns == 0 or (width_partial and columns <= width)))

    def _partial_axes(self, ba, family: mosaic.Family, target) -> tuple[bool, bool]:
        """`_Mosaic._partial_axes`: each axis the call does not fill with whole tiles that the launch may run
        partial (`_partial_admitted`) where the tile it derives on the padded call (`_derived`, read once
        through `derived`) runs it partial as it is (`mosaic_gpu.partial_runs`): the same tile, its operands
        and output uncopied, so it runs no slower than padded. A partial axis that would take a narrower tile
        is padded instead: a GH200's grouped 4096 x 200 x 136 product ran its partial tiles one swizzled tile
        deep and wide 1.07x and its input gradient 1.20x their padded time, where 8192 x 1000 x 1400's ran
        0.86x and 0.93x (M-EM-22b)."""
        from .templates.gemm import mosaic_gpu, pricing

        extents, whole = self._whole_extents(ba, family)
        short = (extents.get('depth') != whole.get('depth'), extents.get('width') != whole.get('width'))
        admitted = tuple(each and grows for each, grows in zip(self._partial_admitted(ba, family, target),
                                                               short, strict=True))
        if not any(admitted):
            return False, False
        tile = self.derived(self._aligned(ba, target, whole_tiles=True), target)
        if tile is None:
            return False, False
        dtype = ba.arguments['tokens'].dtype
        runs = mosaic_gpu.partial_runs(tile, dtype, ba.arguments['interpret'],
                                       transpose_rhs=ba.arguments['trans_rhs'],
                                       gathered=ba.arguments.get('gather') is not None,
                                       scattered=ba.arguments.get('scatter') is not None)
        # The tile no deeper and no wider than the call (`mosaic_gpu.gmm`'s refusal): a last tile partial,
        # never a first; a width never padded (matrices side by side', `_padded_extents`) never partial.
        columns = pricing.tile_columns(tile, pricing.family_of(tile, dtype))
        within = (tile.block_k <= extents['depth'], columns <= extents.get('width', columns))
        depth_partial, width_partial = (each and run and fits for each, run, fits
                                        in zip(admitted, runs, within, strict=True))
        return depth_partial, width_partial

    def _partial_admitted(self, ba, family: mosaic.Family, target) -> tuple[bool, bool]:
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
        from .templates.gemm import mosaic_gpu

        if (self._scaled or not family.bounded_loads
                or family.operands_in_registers or ba.arguments['interpret']
                or self._swapped(ba, target) is not None):
            return False, False
        tokens, kernel, hooks = ba.arguments['tokens'], ba.arguments['kernel'], ba.arguments['hooks']
        depth, width = _matrix_extents(ba)
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

    def _aligned(self, ba, target, *, whole_tiles: bool = False):
        # A prologue never sees the depth's padding: its result zeroed past the call's depth by a side vector
        # over the depth (`mosaic_gpu.depth_masked`), bound with the padded call so its planner, its register
        # model and its kernel stage the vector alike.
        from .templates.gemm import mosaic_gpu

        aligned = super()._aligned(ba, target, whole_tiles=whole_tiles)
        hooks = aligned.arguments['hooks']
        depth = self._padded_extents(ba).get('depth')
        padded = self._padded_extents(aligned).get('depth')
        # A depth run with its last step partial (`_partial_axes`) is masked too, its mask a vector of the
        # depth whose last step's load past it reads zeros.
        layouts = self._families(ba, target)
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
        depth, width = _matrix_extents(ba)
        # Matrices side by side are each whole column tiles (`_takes_beside`): their depth alone padded.
        return {'depth': depth} if ba.arguments.get('beside') else {'depth': depth, 'width': width}

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
        if ba.arguments.get('joined') is not None:
            dims.append(('joined', ba.arguments['joined'], {1: 'depth'}))
        if ba.arguments.get('beside'):
            # Each matrix side by side its depth alone, as the kernel's (`_padded_extents`).
            depth = {dim: axis for dim, axis in matrix.items() if axis == 'depth'}
            dims = [(name, value, depth if name == 'kernel' else each) for name, value, each in dims]
            dims.append(('beside', ba.arguments['beside'], (depth,) * len(ba.arguments['beside'])))
        if ba.arguments.get('chained'):
            # Each product's matrix its width padded as the kernel's; their depths whole steps (`legal`).
            width = {dim: axis for dim, axis in matrix.items() if axis == 'width'}
            dims.append(('chained', ba.arguments['chained'], (({}, width),) * len(ba.arguments['chained'])))
        return dims

    BLOCKS: ClassVar[dict[str, str]] = {'rows': 'block_m', 'depth': 'block_k', 'columns': 'block_n'}


    def _tile(self, ba, target) -> tiling.Tiles | None:
        # A dense product of fewer rows than a role's (a decode step's) may run swapped: its matrix takes the
        # MMA's rows and its rows the columns, whose unit is narrower, so its tile multiplies no padding rows
        # (CUTLASS's and cuBLAS's swap of A and B; megabake's 4 x 2048 x 2048 product went from 10.0 to
        # 4.7 us swapped, cuBLAS's time, row 1762), derived as the swapped call's own tile, and taken where
        # its walk prices below the unswapped tile's (row 5874: an NVFP4 decode's swap won on wide outputs
        # and lost 25-31% on square and narrow ones, M-EM-26-gh200).
        from .templates.gemm import pricing

        plain = self._decode_split(ba, super()._tile(ba, target), target)
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
        tile = split if splits > 1 and self.takes(ba, split, target) else tile
        walked = {each: pricing.walk_seconds(each, self._contraction(ba, each, target))
                  for each in (tile, plain) if each is not None}
        return plain if plain is not None and walked[plain] <= walked[tile] else tile

    def _contraction(self, ba, tile: tiling.Tiles, target) -> dict:
        """A decode-sized dense call's walk at `tile` (`pricing.decode_contraction`): swapped, its matrix
        the rows (`_swapped`), its operands' bytes and decoded forms the launch's (`_operand_bytes`)."""
        from .templates.gemm import pricing, row_tiles

        _, depth, width = self._generated_extents(ba)
        assert depth is not None, "a product's steps run over its depth"
        rows = ba.arguments['tokens'].shape[0]
        swapped = self._swapped(ba, target) if tile.swapped else None
        return pricing.decode_contraction(
            tile, target, rows=rows, groups=ba.arguments['sizes'].shape[0], depth=depth, width=width,
            columns=None if swapped is None else swapped.arguments['kernel'].shape[1],
            # A decode step's partials hold its rows, a TMA descriptor's whole 8s of them.
            held=-(-rows // row_tiles.TMA_ROWS) * row_tiles.TMA_ROWS,
            operands=self._operand_bytes(ba, tile), stored=_stored_bytes(ba, rows, width),
            multiprocessors=_multiprocessors(target))

    def _operand_bytes(self, ba, tile: tiling.Tiles) -> tuple[tuple[float, float], tuple]:
        """The bytes an element of the products' left and right operands moves from device memory at `tile`
        (swapped, the matrix the left) and each one's decode (`pricing.step_shared_bytes`): none."""
        tokens, kernel = ba.arguments['tokens'], ba.arguments['kernel']
        rows, matrix = jnp.dtype(tokens.dtype).itemsize, jnp.dtype(kernel.dtype).itemsize
        return ((matrix, rows) if tile.swapped else (rows, matrix)), (None, None)

    def _decode_split(self, ba, tile: tiling.Tiles | None, target) -> tiling.Tiles | None:
        """`tile`, a call's of decode-sized groups (each group's rows within one row tile: a MoE decode
        step's experts, or a dense product's decode step run unswapped, its one group) its depth split where
        the busiest program then walks shorter (`pricing.split_count` over each active group's one row
        tile, its matrix's stream the walk): one wave of tiles streamed each expert's matrix at a program's
        share of the memory rate, 1.28-1.46x Triton's time at the same DRAM bytes, 48% of the peak rate
        against Triton's 80% (FS-3622-NCU1, FS-3622-WAVES1, row 3622); a dense decode on a GH200, unsplit,
        ran Llama-3-8B's 4096 x 4096 at one row on 16 programs of 132, 4.3x XLA's time (M-EM-38). None,
        exact sums (`legal`), hooks or a pair keep it unsplit."""
        from .templates.gemm import pricing

        hooks = ba.arguments['hooks']
        if (tile is None or ba.arguments['exact_sums'] or ba.arguments.get('paired') is not None
                or hooks.prologue is not None or hooks.epilogue is not None):
            return tile
        if ba.arguments['tokens'].shape[0] > ba.arguments['sizes'].shape[0] * tile.block_m:
            return tile
        splits = pricing.split_count(tile, **self._contraction(ba, tile, target))
        split = dataclasses.replace(tile, splits=splits)
        return split if splits > 1 and self.takes(ba, split, target) else tile

    def _left_fits(self, ba, config: tiling.Tiles) -> bool:
        """Whether a swapped tile reads `ba`'s matrix as its products' left operand: a matrix lying `[depth,
        width]` is read transposed, whose stage casts to the products' layout in whole row atoms on a family
        reading registers (`Family.transposed_rows`, as the kernel gradient's transposed rows)."""
        from .templates.gemm import pricing

        family = pricing.family_of(config, ba.arguments['tokens'].dtype)
        return (ba.arguments['trans_rhs'] or not family.operands_in_registers
                or config.block_k % family.transposed_rows == 0)

    def _swapped(self, ba, target):
        """The dense product `ba` as its swapped tile runs it: its matrix `[width, depth]` the rows, however
        it lies, and its rows a matrix of them, padded to whole column units of every family the target
        runs, read transposed (`mosaic_gpu.gmm`'s `transpose_lhs` where the matrix lies `[depth, width]`).
        None where it runs as it is: a call with hooks, a paired or grouped one, matrices side by side (a
        swapped call's matrix is its products' one left operand), one whose rows fill a role
        (`Family.role_rows`), or one on a family it does not swap on (`_swaps_on`)."""
        families = self._families(ba, target)
        tokens = ba.arguments['tokens']
        if (not families or not all(self._swaps_on(family, tokens.dtype) for family in families)
                or not ba.arguments.get('dense') or ba.arguments['hooks'] != NO_HOOKS
                or ba.arguments.get('paired') is not None or ba.arguments.get('beside')
                or ba.arguments.get('chained')
                or tokens.shape[0] >= min(family.role_rows for family in families)):
            return None
        depth, width = _matrix_extents(ba)
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
                        config: tiling.Tiles, target, **matrix_options) -> tuple:
        """A swapped tile's launch (`_swapped`): `matrix` the products' rows times the rows, padded to whole
        column tiles and read transposed, at the tile unswapped, the output transposed back;
        `matrix_options`, how the launch reads the matrix (`transpose_lhs`, a packed one's scales)."""
        from .templates.gemm import mosaic_gpu

        rows = tokens.shape[0]
        padded = jnp.pad(tokens, ((0, -rows % config.block_n), (0, 0)))
        tile = dataclasses.replace(config, swapped=False)
        out = mosaic_gpu.gmm(matrix, padded[None], sizes, tiles=tile, transpose_rhs=True, out_dtype=out_dtype,
                             multiprocessors=self._launched(tile, interpret, target),
                             interpret=INTERPRETED if interpret else False, dense=True, exact_sums=exact_sums,
                             **matrix_options)
        assert isinstance(out, jax.Array), "one output (`out_dtypes` is not passed)"
        return out[:, :rows].T, None

    def _ran(self, ba, config: tiling.Tiles, target) -> tuple:
        """`_Mosaic._ran`: a swapped tile runs the call swapped (`_swapped`), the tile unswapped, on
        `target`; a grouped split's runs write fp32 partials, each at the tile unsplit (`_decode_split`)."""
        ba = self._aligned(ba, target)
        if not config.swapped:
            if config.splits == 1:
                return ba, config
            return (self.bind(*ba.args, **{**ba.kwargs, 'out_dtype': 'float32'}),
                    dataclasses.replace(config, splits=1))
        swapped = self._swapped(ba, target)
        assert swapped is not None, "a swapped tile is a swappable call's (`legal`)"
        if config.splits > 1:
            # A split's runs write fp32 partials (`mosaic_gpu.gmm`), each at the tile unsplit.
            swapped = self.bind(*swapped.args, **{**swapped.kwargs, 'out_dtype': 'float32'})
        return swapped, dataclasses.replace(config, swapped=False, splits=1)

    def legal(self, ba, config: tiling.Tiles, target) -> bool:
        from .templates.gemm import mosaic_gpu

        if config.swapped and (self._swapped(ba, target) is None or not self._left_fits(ba, config)):
            return False
        # Matrices side by side judged at the tile as derived, its split and swap whole (`_ran` runs a
        # split's runs, and a swapped call, at a tile of neither, which `candidate` then reads).
        if ba.arguments.get('beside') and not self._takes_beside(ba, config):
            return False
        if ba.arguments.get('chained') and not self._takes_chained(ba, config):
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

    def candidate(self, ba, config: tiling.Tiles, target) -> bool:
        # Matrices side by side, or chained products, take a tile of their own (`_takes_beside`,
        # `_takes_chained`), the derivation taking the next.
        return ((not ba.arguments.get('beside') or self._takes_beside(ba, config))
                and (not ba.arguments.get('chained') or (self._takes_chained(ba, config)
                                                         and self._replays(ba, config, target)))
                and super().candidate(ba, config, target))

    def _replays(self, ba, config: tiling.Tiles, target) -> bool:
        """Whether a chained launch at `config` has its replay (`chained.guarded_walk`: its flagged tiles
        walked again at a narrower column tile, an accumulator a product, `chained.replay_tile`) at a tile the
        launch runs: legal as every tile is (`_Mosaic.legal`: its granularity and the planner's resources), so
        the two geometries are derived and validated before the chained launch is taken, and a call whose
        replay no tile holds takes its products apart (c3b97e20f F2)."""
        from .templates.gemm import pricing
        from .templates.gemm.chained import replay_refusal, replay_tile

        dtype, products = ba.arguments['tokens'].dtype, 1 + len(ba.arguments['chained'])
        if replay_refusal(config, products, pricing.family_of(config, dtype), dtype) is not None:
            return False

        def replayed(ba, config, target):
            # The replay's own tile, judged as any tile (its chaining judged at the call's, `_takes_chained`),
            # its own replay none: it flags nothing.
            return super(MosaicGmm, self).candidate(ba, config, target)

        return self._held(ba, replay_tile(config, products), target, replayed)

    def _takes_chained(self, ba, config: tiling.Tiles) -> bool:
        """Whether `config` joins the call's products end to end (`gmm_call`'s `chained`), as the launch
        refuses them (`mosaic_gpu.chained_refusal`): the call's parts, its swap and its scales among them."""
        from .templates.gemm import pricing
        from .templates.gemm.chained import chained_refusal

        arguments, hooks = ba.arguments, ba.arguments['hooks']
        matrix, transposed = self._streamed_matrix(arguments['kernel'], arguments['trans_rhs'])[:2]
        chained = tuple((rows, self._streamed_matrix(each, arguments['trans_rhs'])[0])
                        for rows, each in arguments['chained'])
        parts = (hooks.prologue, hooks.epilogue, arguments.get('paired'), arguments.get('joined'),
                 arguments.get('gather'), arguments.get('scatter'), arguments.get('first'),
                 arguments['exact_sums'], arguments.get('beside') or None, config.swapped or None,
                 self._scaled or None)
        family = pricing.family_of(config, arguments['tokens'].dtype)
        return chained_refusal(arguments['tokens'], matrix, chained, config, family,
                                          transpose_rhs=transposed, parts=parts) is None

    def _takes_beside(self, ba, config: tiling.Tiles) -> bool:
        """Whether `config` runs the call's matrices side by side (`gmm_call`'s `beside`), as the launch
        refuses them (`mosaic_gpu.beside_refusal`): the call's parts, its swap and its scales among them."""
        from .templates.gemm import mosaic_gpu, pricing

        arguments, hooks = ba.arguments, ba.arguments['hooks']
        widths = tuple(each.shape[1 if arguments['trans_rhs'] else 2]
                       for each in (arguments['kernel'], *arguments['beside']))
        parts = (hooks.prologue, hooks.epilogue, arguments.get('paired'), arguments.get('joined'),
                 arguments.get('gather'), arguments.get('scatter'), config.swapped or None,
                 self._scaled or None)
        family = pricing.family_of(config, arguments['tokens'].dtype)
        return mosaic_gpu.beside_refusal(config, family, widths, parts=parts) is None

    def _fwd(self, tokens: jax.Array, kernel: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), epilogue_operands: tuple = (), paired: jax.Array | None = None,
             first: jax.Array | None = None, gather: jax.Array | None = None,
             scatter: jax.Array | None = None, joined: jax.Array | None = None, beside: tuple = (),
             chained: tuple = (), *, out_dtype: str,
             trans_rhs: bool, interpret: bool,
             hooks: Hooks = NO_HOOKS, exact_sums: bool = False,
             dense: bool = False, config: tiling.Tiles, return_residuals: bool = False):
        del return_residuals
        return self._forward(tokens, kernel, sizes, prologue_operands, epilogue_operands, paired, first,
                             gather, scatter, joined, beside, chained, out_dtype=out_dtype,
                             trans_rhs=trans_rhs, interpret=interpret, hooks=hooks, exact_sums=exact_sums,
                             dense=dense, config=config, formats={})

    def _forward(self, tokens, kernel, sizes, prologue_operands, epilogue_operands, paired, first, gather,
                 scatter, joined=None, beside: tuple = (), chained: tuple = (), *, out_dtype: str,
                 trans_rhs: bool,
                 interpret: bool, hooks: Hooks, exact_sums: bool, dense: bool, config: tiling.Tiles,
                 formats: dict):
        """`_fwd`'s one forward, every product launch's: the call as the launch runs it, its depth and width
        padded to whole tiles (`_aligned`), its outputs sliced back; `formats` the format operands a
        subclass's binding takes (`MosaicStepScaledGmm`'s step scales, `MosaicMxGmm`'s block scales), passed
        to the mainloop as they are beside the register model its launch states (`_stated`); matrices
        `beside` the kernel (`mosaic_gpu.gmm`'s), whole tiles each (`legal`), their outputs each its own;
        `chained` products (`mosaic_gpu.gmm`'s), each matrix streamed as the kernel and its width padded as
        the call's, their depths whole steps each (`legal`)."""
        target = generation.gpu_target()
        width = kernel.shape[1] if trans_rhs else kernel.shape[2]
        call = (bound := self._aligned(self.bind(
            tokens, kernel, sizes, prologue_operands, epilogue_operands, paired, first, gather, scatter,
            joined, tuple(beside or ()), tuple(chained), out_dtype=out_dtype, trans_rhs=trans_rhs,
            interpret=interpret, hooks=hooks, exact_sums=exact_sums, dense=dense), target)).arguments
        tokens, kernel, paired, hooks = call['tokens'], call['kernel'], call.get('paired'), call['hooks']
        joined = call.get('joined')
        prologue_operands, epilogue_operands = call['prologue_operands'], call.get('epilogue_operands', ())
        if beside:
            # Each as the mainloop streams the kernel (`_streamed_matrix`), its depth padded as the call's.
            formats = {**formats, 'beside': tuple(self._streamed_matrix(matrix, trans_rhs)[0]
                                                  for matrix in call['beside'])}
        if chained:
            # Each product's matrix as the mainloop streams the kernel (`_streamed_matrix`).
            formats = {**formats, 'chained': tuple((rows, self._streamed_matrix(matrix, trans_rhs)[0])
                                                   for rows, matrix in call['chained'])}
        out, residuals = self._launched_call(tokens, kernel, sizes, prologue_operands, epilogue_operands,
                                             paired, first, gather, scatter, joined, out_dtype=out_dtype,
                                             trans_rhs=trans_rhs, interpret=interpret, hooks=hooks,
                                             exact_sums=exact_sums, dense=dense, config=config,
                                             formats={**formats, **self._stated(bound, config, target)},
                                             target=target)
        if beside:
            return out, residuals  # their columns unpadded (`_padded_extents`)
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
                       gather, scatter, joined=None, *, out_dtype: str, trans_rhs: bool, interpret: bool,
                       hooks: Hooks, exact_sums: bool, dense: bool, config: tiling.Tiles, formats: dict,
                       target):
        """`_fwd`'s launch of the call as it runs it (`_aligned`), `formats` its mainloop's keywords."""
        from .templates.gemm import mosaic_gpu

        if config.swapped:
            return self._swapped_launch(kernel[0], tokens, sizes, out_dtype=out_dtype, interpret=interpret,
                                        exact_sums=exact_sums, config=config, target=target,
                                        transpose_lhs=not trans_rhs, **formats)
        matrix, transposed, stages = self._streamed_matrix(kernel, trans_rhs)
        return mosaic_gpu.gmm(tokens, matrix, sizes, first=first, tiles=config, transpose_rhs=transposed,
                              out_dtype=out_dtype, multiprocessors=self._launched(config, interpret, target),
                              interpret=INTERPRETED if interpret else False,
                              prologue_operands=prologue_operands, epilogue_operands=epilogue_operands,
                              dense=dense, paired=paired, joined=joined, exact_sums=exact_sums, gather=gather,
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
             scatter: jax.Array | None = None, joined: jax.Array | None = None, beside: tuple = (),
             chained: tuple = (), *, out_dtype: str,
             trans_rhs: bool, interpret: bool,
             hooks: Hooks = NO_HOOKS, exact_sums: bool = False, dense: bool = False,
             config: tiling.Tiles, return_residuals: bool = False, packed_rows: bool = False):
        # `packed_rows` states the swapped call's orientation (`_swapped`), which its derivation reads: the
        # call itself runs as `config` says (`_launched_call`).
        del packed_rows, chained
        return super()._fwd(tokens, kernel, sizes, prologue_operands, epilogue_operands, paired, first,
                            gather, scatter, joined, out_dtype=out_dtype, trans_rhs=trans_rhs,
                            interpret=interpret,
                            hooks=hooks, exact_sums=exact_sums, dense=dense, config=config,
                            return_residuals=return_residuals)

    def _streamed_matrix(self, kernel: _PackedMatrix, trans_rhs: bool) -> tuple:
        # Its words, K-major whatever orientation its decoded shape states, and its block scales, which each
        # stage decodes (`mosaic_gpu.gmm`'s `scales`).
        del trans_rhs
        return kernel.words, True, {'scales': kernel.scales}

    def _matrix_transposed(self, ba) -> bool:
        # Its words K-major, or a swapped call's rows read transposed (`_swapped`): its matrix streams
        # transposed either way, whatever the binding holds (an abstract matrix where a derivation binds it).
        del ba
        return True

    def _allocations(self, ba, config: tiling.Tiles, family: mosaic.Family) -> mosaic_gpu.Allocations:
        from .templates.gemm import mosaic_gpu

        dtype = ba.arguments['tokens'].dtype
        rows = ba.arguments['packed_rows']
        return mosaic_gpu.gmm_allocations(
            config, family, dtype=dtype, rhs_dtype=dtype if rows else jnp.uint32, transpose_rhs=True,
            prologue_axes=(), prologue_dtypes=(), epilogue_axes=(), epilogue_dtypes=(),
            dtypes=(ba.arguments['out_dtype'],), packed=True, packed_rows=rows)

    def candidate(self, ba, config: tiling.Tiles, target) -> bool:
        # A swapped call's consumers each decode 64 rows of the matrix, its fragment's; the matrix's columns
        # take whole 64-column fragments otherwise (`packed.rows_taken`, `packed.taken`).
        whole = (config.block_m // config.warpgroups == 64 if ba.arguments['packed_rows']
                 else config.block_n % 64 == 0)
        return (config.cluster == 1 and config.splits == 1 and not config.swapped and whole
                and super().candidate(ba, config, target))

    SWAPPED_OPTIONS: ClassVar[dict] = {'packed_rows': True}

    def _operand_bytes(self, ba, tile: tiling.Tiles) -> tuple[tuple[float, float], tuple]:
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
        from .templates.gemm import packed
        del dtype
        return packed.rows_supported(family)

    def _launched_call(self, tokens, kernel, sizes, prologue_operands, epilogue_operands, paired, first,
                       gather, scatter, joined=None, *, out_dtype: str, trans_rhs: bool, interpret: bool,
                       hooks: Hooks, exact_sums: bool, dense: bool, config: tiling.Tiles, formats: dict,
                       target):
        if not config.swapped:
            return super()._launched_call(tokens, kernel, sizes, prologue_operands, epilogue_operands,
                                          paired, first, gather, scatter, joined, out_dtype=out_dtype,
                                          trans_rhs=trans_rhs, interpret=interpret, hooks=hooks,
                                          exact_sums=exact_sums, dense=dense, config=config, formats=formats,
                                          target=target)
        # The matrix's words and scales the rows, decoded in registers (`_swaps_on`).
        return self._swapped_launch(kernel.words[0], tokens, sizes, out_dtype=out_dtype, interpret=interpret,
                                    exact_sums=exact_sums, config=config, target=target,
                                    scales=kernel.scales[0], packed_rows=True, **formats)

    def _registers(self, ba, config: tiling.Tiles, *, merging: bool, target) -> int | None:
        # Beside the plain product's live set, a step's decode holds a 128-depth group's words, the code
        # being decoded and the stage's block scales, each a `[columns, 16]` register tile over a role's
        # threads (`packed.lanes`, its columns the tile's, or a consumer's rows where the call runs swapped),
        # and a word's decode in flight, its form's live registers (`bits.BitForm.live`); swapped, the
        # decoded stage too, the products' left operand (`packed.rows_decoded`, bfloat16 a half register).
        from ...algebra import bits
        from .templates import resources

        if ba.arguments['packed_rows']:
            share = config.block_m // config.warpgroups
            decoded = 3 * share * 16 + share * config.block_k // 2
        else:
            decoded = 3 * config.block_n * 16
        return _beside(super()._registers(ba, config, merging=merging, target=target),
                       decoded // resources.THREADS + bits.NVFP4_BFLOAT16_FORM.live)


@dataclasses.dataclass(frozen=True)
class MosaicTgmm(_Mosaic):
    """`Tgmm` through the template's mainloop (`mosaic_gpu.tgmm`) on a GPU family's bundle or a TPU core's:
    `block_m` x `block_n` tiles of each group's gradient, its rows `block_k` at a time."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens', 'cotangent')
    KERNEL: ClassVar[str] = tiling.TGMM

    def output_sides(self, ba) -> tuple[int, int]:
        return _gradient_sides(ba)

    EXACT_SUMS: ClassVar[bool] = True
    STAGED_ROWS: ClassVar[bool] = True

    def _allocations(self, ba, config: tiling.Tiles, family: mosaic.Family) -> mosaic_gpu.Allocations:
        from .templates.gemm import mosaic_gpu

        return mosaic_gpu.tgmm_allocations(
            config, family, lhs_dtype=ba.arguments['tokens'].dtype,
            rhs_dtype=ba.arguments['cotangent'].dtype, prologue_axes=ba.arguments['hooks'].prologue_axes,
            prologue_dtypes=tuple(side.dtype for side in ba.arguments['prologue_operands']),
            out_dtype=ba.arguments['out_dtype'], k_major=self.K_MAJOR, step_scaled=self.K_MAJOR,
            unstaged=self.unstaged(ba), interpret=bool(ba.arguments['interpret']),
            gathered=tuple(table is not None for table in ba.arguments.get('gather') or (None, None)))

    def _generated_extents(self, ba) -> tuple[int, int | None, int]:
        # A gradient tile's rows are the tokens' width; its steps run over a group's rows, padded to whole
        # steps (`mosaic_gpu.tgmm`).
        return ba.arguments['tokens'].shape[1], None, ba.arguments['cotangent'].shape[1]

    def _depth_unit(self, ba, family: mosaic.Family, dtype) -> int:
        # A tile's rows take whole roles of the depth, but where its last row tile may be partial
        # (`_ragged`): there whole swizzled rows of the output's, which a TMA store's tiled view takes no
        # fewer of ("dimension 1 must have size smaller or a multiple of its tiling 64, but got 200", a
        # 200-deep kernel gradient on sm90 and sm120, the compile gate at c717a2439b4).
        from .templates import swizzling
        from .templates.gemm import row_tiles

        if self._ragged(ba, family):
            return math.lcm(row_tiles.TMA_ROWS, swizzling.swizzle_columns(dtype, swizzling.SWIZZLE_BYTES))
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

    def _tile(self, ba, target) -> tiling.Tiles | None:
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

    def _split_tile(self, ba, target, below: int | None = None) -> tiling.Tiles | None:
        """The derived tile for `ba` at `target`, split across depth where unsplit underfills a wave and the
        single-pass split walk (`pricing.best_splits`) prices cheaper."""
        from .templates.gemm import pricing

        admits = None if below is None else (lambda tile: tile.block_n < below)
        tile = self._derived(ba, target, admits)
        if tile is None or self._balances(ba, tile, target) or self._splits(ba, tile, target) == 1:
            return tile
        partials, d = self._partials(ba), ba.arguments['tokens'].dtype
        sliced = self._derived(partials, target, lambda each: (admits is None or admits(each)) and not (
            self._pair_refuses(partials, dataclasses.replace(each, splits=2), pricing.family_of(each, d))))
        if sliced is None:
            return tile
        walk = self._walk(ba, sliced, target)
        if (splits := pricing.best_splits(walk, sliced.stages)) == 1:
            return tile
        split = dataclasses.replace(sliced, splits=splits)
        return tile if self._refused(ba, split, target) else split

    def _balances(self, ba, tile: tiling.Tiles, target) -> bool:
        """Whether the call's last wave runs balanced (`balanced`: `pricing.dense_tail` + register fit)."""
        return self.balanced(ba, tile, target)

    def _splits(self, ba, tile: tiling.Tiles, target) -> int:
        from .templates.gemm import pricing

        return pricing.split_count(tile, **self._contraction(ba, tile, target))

    def _walk(self, ba, tile: tiling.Tiles, target) -> pricing.SplitWalk | None:
        """The walk `tile` runs the call in (`pricing.split_walk`), split or not: its output stored at the
        call's own dtype, a split's fp32 partials priced by their passes (`SplitWalk.passes`)."""
        from .templates.gemm import pricing

        return pricing.split_walk(tile, **self._contraction(ba, tile, target))

    def _contraction(self, ba, tile: tiling.Tiles, target) -> dict:
        """The call's contraction as the split prices it at `tile` (`pricing.split_walk`'s facts): the call
        as the launch runs it (`_aligned`), its depth and width padded to whole tiles (a 100-wide gradient's
        128 columns)."""
        from .templates.gemm import pricing

        ba = self._aligned(ba, target)
        depth, _, width = self._generated_extents(ba)
        tokens, cotangent = ba.arguments['tokens'], ba.arguments['cotangent']
        itemsizes = (tokens.dtype.itemsize, cotangent.dtype.itemsize)
        return {'groups': ba.arguments['sizes'].shape[0], 'rows': _rows_of(_TGMMS, ba.args), 'depth': depth,
                'width': width, 'itemsizes': itemsizes,
                'stored': jnp.dtype(ba.arguments['out_dtype']).itemsize,
                'multiprocessors': _multiprocessors(target),
                'step_seconds': pricing.step_seconds(tile, target, itemsizes),
                'bandwidth': target.memory_rate,
                'launch_seconds': target.launch_seconds}

    def _partials(self, ba):
        """The call `ba` writing fp32 partials, as a split's slices do (`mosaic_gpu.tgmm`): each slice a group
        of its own, so no dense call's; a dense call's slices whole steps from its first row (`whole_steps`),
        their stages read as loaded as its own (`unstaged`)."""
        return self.bind(*ba.args, **{**ba.kwargs, 'out_dtype': 'float32', 'dense': False,
                                      'whole_steps': bool(ba.arguments.get('dense'))})

    def _ran(self, ba, config: tiling.Tiles, target) -> tuple:
        """The call and tile a launch at `config` runs (`_aligned`): a split one's fp32 partials, unsplit."""
        ba = self._aligned(ba, target)
        if config.splits == 1:
            return ba, config
        return self._partials(ba), dataclasses.replace(config, splits=1)

    def _divides(self, ba, config: tiling.Tiles, target) -> bool:
        from .templates.gemm import pricing

        del target

        rows, _, width = self._generated_extents(ba)  # a pair's tile both CTAs' (`pricing.tile_rows`)
        family = pricing.family_of(config, ba.arguments['tokens'].dtype)
        whole = (rows % pricing.tile_rows(config, family) == 0
                 or (self._ragged(ba, family) and config.cluster == 1))
        return whole and width % pricing.tile_columns(config, family) == 0

    def _ragged(self, ba, family: mosaic.Family) -> bool:
        # A depth no cooperative tile's rows divide takes a partial last row tile where the family's loads
        # past the depth read zeros (TMA's, `Family.bounded_loads`; row 2295: GPT-OSS's 2880, which held one
        # consumer's 64 x 192 tile alone, 1.79x Triton's time on an H100, where every other model's kernel
        # gradient, a cooperative tile, ran 0.78-0.92 of it, FSCELLS1).
        # A depth some cooperative tile divides keeps its whole tiles.
        from .templates.gemm import mosaic_gpu

        rows, _, _ = self._generated_extents(ba)
        cooperative = family.role_rows * min(family.consumers, tiling.CONSUMERS)
        prologue = ba.arguments['hooks'].prologue
        return (family.bounded_loads and rows % cooperative != 0
                and mosaic_gpu.ragged_rows(rows, prologue=prologue, k_major=self.K_MAJOR))

    def candidate(self, ba, config: tiling.Tiles, target) -> bool:
        """`_Mosaic.candidate` on a family whose products read registers loading its transposed stage in whole
        row atoms of its products (the step, `block_k`, in `Family.transposed_rows`)."""
        from .templates.gemm import pricing

        if not super().candidate(ba, config, target):
            return False
        family = pricing.family_of(config, ba.arguments['tokens'].dtype)
        return not (family.operands_in_registers and config.block_k % family.transposed_rows)

    BLOCKS: ClassVar[dict[str, str]] = {'rows': 'block_k', 'depth': 'block_m'}


    def _fwd(self, tokens: jax.Array, cotangent: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), gather: tuple | None = None, *, out_dtype: str, interpret: bool,
             hooks: Hooks = NO_HOOKS,
             exact_sums: bool = False, dense: bool = False, config: tiling.Tiles,
             return_residuals: bool = False, whole_steps: bool = False):
        # The call as the launch runs it, its depth and width padded to whole tiles (`_aligned`), its output
        # sliced back. `whole_steps`, a dense call's split runs (`_partials`), as `mosaic_gpu.tgmm`'s.
        from .templates.gemm import mosaic_gpu

        target = generation.gpu_target()
        depth, width = tokens.shape[1], cotangent.shape[1]
        call = (bound := self._aligned(self.bind(
            tokens, cotangent, sizes, prologue_operands, gather, out_dtype=out_dtype, interpret=interpret,
            hooks=hooks, exact_sums=exact_sums, dense=dense, whole_steps=whole_steps), target)).arguments
        out = mosaic_gpu.tgmm(call['tokens'], call['cotangent'], sizes, tiles=config, out_dtype=out_dtype,
                              multiprocessors=self._programs(config, target),
                              interpret=INTERPRETED if interpret else False, prologue=hooks.before(),
                              prologue_operands=call['prologue_operands'], prologue_axes=hooks.prologue_axes,
                              exact_sums=exact_sums, dense=dense, gather=gather, whole_steps=whole_steps,
                              **self._stated(bound, config, target))
        return out[:, :depth, :width], None


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
             gather: jax.Array | None = None, scatter: jax.Array | None = None,
             joined: jax.Array | None = None, beside: tuple = (), chained: tuple = (), *, out_dtype: str,
             trans_rhs: bool, interpret: bool, hooks: Hooks = NO_HOOKS, exact_sums: bool = False,
             dense: bool = False, scales: tuple = (),
             config: tiling.Tiles,
             return_residuals: bool = False):
        del return_residuals, chained
        return self._forward(tokens, kernel, sizes, prologue_operands, epilogue_operands, paired, first,
                             gather, scatter, joined, out_dtype=out_dtype, trans_rhs=trans_rhs,
                             interpret=interpret, hooks=hooks, exact_sums=exact_sums, dense=dense,
                             config=config, formats={self.SCALES: tuple(scales)})


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

    def candidate(self, ba, config: tiling.Tiles, target) -> bool:
        within = SCALE_DEPTH % config.block_n == 0
        return config.splits == 1 and not config.swapped and within and super().candidate(ba, config, target)

    def _registers(self, ba, config: tiling.Tiles, *, merging: bool, target) -> int | None:
        # Beside the exact product's live set, a step's column scales.
        return _beside(super()._registers(ba, config, merging=merging, target=target),
                       _column_scale_registers(ba, config))


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

    def candidate(self, ba, config: tiling.Tiles, target) -> bool:
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
             config: tiling.Tiles, return_residuals: bool = False, whole_steps: bool = False):
        # Bound as a kernel gradient states its operands, rows first: the rows-contiguous operands' transposes
        # (`gemm.blockwise_fp8_kernel_gradient`), which XLA folds into the kernel's own operands, read back as
        # they lie (`mosaic_gpu.tgmm`'s `k_major`); its scales (`step_scales`, the binding's operand) its
        # mainloop's `step_scales`. Its layout's steps are whole by the caller's (`k_major`), unsplit.
        from .templates.gemm import mosaic_gpu

        del prologue_operands, gather, hooks, dense, return_residuals, whole_steps
        target = generation.gpu_target()
        bound = self.bind(tokens, cotangent, sizes, out_dtype=out_dtype, interpret=interpret,
                          exact_sums=exact_sums, step_scales=step_scales)
        out = mosaic_gpu.tgmm(tokens.T, cotangent.T, sizes, tiles=config, out_dtype=out_dtype,
                              multiprocessors=self._launched(config, interpret, target),
                              interpret=INTERPRETED if interpret else False, exact_sums=exact_sums,
                              k_major=True, step_scales=tuple(step_scales),
                              **self._stated(bound, config, target))
        return out, None

    def _fp8(self, ba, dtype, family: mosaic.Family) -> bool:
        # Its operands lie rows-contiguous, as Hopper's FP8 tensor cores read both.
        return (jnp.dtype(dtype) in family.fp8_formats
                and ba.arguments['hooks'] == NO_HOOKS and bool(ba.arguments.get('exact_sums')))

    def candidate(self, ba, config: tiling.Tiles, target) -> bool:
        return config.splits == 1 and super().candidate(ba, config, target)

    def _registers(self, ba, config: tiling.Tiles, *, merging: bool, target) -> int | None:
        # Beside the exact gradient's live set, a step's column scales.
        return _beside(super()._registers(ba, config, merging=merging, target=target),
                       _column_scale_registers(ba, config))


def _beside(held: int | None, more: int) -> int | None:
    """A launch's registers `held` with `more` beside them; none where it states none (a TPU core's)."""
    return None if held is None else held + more


def _column_scale_registers(ba, config: tiling.Tiles) -> int:
    """The registers a step-scaled walk holds beside the exact one's: a step's fp32 column scales in its
    accumulator's column layout (`Family.column_registers`)."""
    from .templates.gemm import pricing

    family = pricing.family_of(config, ba.arguments['tokens'].dtype)
    return family.column_registers(config.block_n, jnp.float32)


def _cutlass_call(target: int, out: jax.ShapeDtypeStruct) -> Callable:
    """dew_cutlass's FFI call giving `out` of the `target`-th of the targets the wheel names
    (`dew_cutlass.TARGETS`: its grouped GEMM, then its weight gradient), a mapped call running its maps in
    turn, one launch each (row 4389: JAX's FFI batching refuses a call that states no rule)."""
    name = importlib.import_module('dew_cutlass').TARGETS[target]
    return jax.ffi.ffi_call(name, out, vmap_method='sequential')


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
        pointers need), no hooks or exact sums, compiled (no interpreter), its stack read whole."""
        operands = [value for name, value in ba.arguments.items()
                    if name in ('tokens', 'kernel', 'cotangent')]
        extents = {extent for operand in operands for extent in operand.shape[1:]}
        return (all(operand.dtype == jnp.bfloat16 for operand in operands)
                and jnp.dtype(ba.arguments['out_dtype']) in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32))
                and ba.arguments['sizes'].dtype == jnp.int32 and all(extent % 8 == 0 for extent in extents)
                and ba.arguments['hooks'] == NO_HOOKS and not ba.arguments.get('exact_sums')
                and not ba.arguments['interpret'] and ba.arguments.get('first') is None)

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
        # No hooks, side operands, second matrix, stack read from a first matrix or exact sums: the launch is
        # not eligible for them (`_Cutlass.eligible`); a dense call is one group, as any.
        del prologue_operands, epilogue_operands, paired, first, gather, scatter, interpret, hooks, exact_sums
        del dense, config, joined, beside, chained
        importlib.import_module('dew_cutlass').registered()
        width = kernel.shape[1] if trans_rhs else kernel.shape[2]
        out = jax.ShapeDtypeStruct((tokens.shape[0], width), jnp.dtype(out_dtype))
        call = _cutlass_call(0, out)
        return call(tokens, kernel, sizes, transpose_w=trans_rhs), None


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
        del prologue_operands, gather, interpret, hooks, exact_sums, dense, config  # as `CutlassGmm._fwd`'s
        importlib.import_module('dew_cutlass').registered()
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
    from .templates.axes import held as held_axes

    return any(len(held_axes(axes)) > 1 for axes in ba.arguments['hooks'].prologue_axes)


def _multiprocessors(target: generation.GpuTarget) -> int:
    """`target`'s multiprocessors, which the persistent Mosaic GPU launches spread their tiles over."""
    if target.multiprocessors is None:
        raise ValueError(f"{target.generation} states no multiprocessors, which the Mosaic GPU launches read")
    return target.multiprocessors


@dataclasses.dataclass(frozen=True)
class GroupedCase:
    """A grouped product as the grouped selector reads it (`GROUPED`): whether the kernels run on its target
    (`placed`: a GPU of sm80 on, or a TPU; the template's dispatch runs in the branch its platform places),
    whether they see its operands whole (`whole`: no automatic mesh axis splits them,
    `dew.nn.sharding.automatic_axes`), the `product` it declares, its `extents`, its launches bound to it
    (`call`) and the `kind` of call it is (`Kind`, as `gmm_call` and `tgmm_call` state it), which with the
    device's instruction family is its cell (`DECLARED`)."""

    placed: bool
    whole: bool
    product: Product
    call: base.Call
    kind: Kind
    extents: tuple[int, int, int]
    """Its product's rows, depth and width (`_extents_of`): a kernel gradient's rows the ones it sums over,
    a product's those it walks (`_rows_of`)."""
    paired: bool = False
    """Whether it is several products in one launch (`_compound`): a pair's (`gmm_call`'s `paired`), matrices
    side by side (`beside`) or products chained (`chained`), which Mosaic GPU's launch alone takes."""

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
    _Need('extents', lambda case: 0 not in case.extents, "the call has no rows, depth or width",
          row=INHERENT),
)
"""What every launch needs of a grouped product: a process that runs it, whole operands, a product the
kernels compute exactly on the target (its declaration's refusal, the target's tensor rate for its
dtype, no step-by-step sums on a TPU, whose launches do not add them apart), and some extent on each axis
(a product of no rows, depth or width has no tile to launch: XLA's program, its sum of nothing)."""


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

PRODUCTS_APART = selection.Kernel[GroupedCase](
    APART.name, (_Need('single', lambda case: case.paired, "the call is one product", row=INHERENT),),
    generic=True)
"""A paired call's or products' in one launch: each product its own call (`APART`), the launches that take
each its own."""


DECLARED: dict[tuple[str | None, Kind], tuple[selection.Kernel[GroupedCase], str]] = {
    **{(receipt.family, Kind(dense=True, made=receipt.made, together=True, matrix=matrix)): (
        {MOSAIC.name: MOSAIC, APART.name: PRODUCTS_APART}[receipt.winner], receipt.evidence)
       for receipt in TOGETHER for matrix in receipt.timed.matrices},
    **{(family, Kind(dense=True, made=made, matrix=matrix, formed=True)): (MOSAIC, M_EM_65)
       for family in ('mma.sync', 'wgmma') for made in (Made.FORWARD, Made.KERNEL_GRADIENT)
       for matrix in NORMED_08B},
    **{('wgmma', Kind(dense=True, made=made, matrix=matrix, formed=True)): (MOSAIC, M_EM_67)
       for made in (Made.FORWARD, Made.KERNEL_GRADIENT) for matrix in NORMED_4B_8B},
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
    ('wgmma', Kind(dense=True, made=Made.KERNEL_GRADIENT)):
        (XLA, "DENSE1 (H100): Mosaic 1.049x, Triton 1.149x XLA's dot"),
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
    ('wgmma', Kind(dense=True, made=Made.KERNEL_GRADIENT, matrix=(2560, 9216))): (
        XLA, f"{M_EM_52}: the template 1.042x XLA's dot (1.559 against 1.496 ms)"),
    ('wgmma', Kind(dense=True, made=Made.KERNEL_GRADIENT, matrix=(9216, 2560))): (
        MOSAIC, f"{M_EM_52}: the template 0.755x XLA's dot (1.493 against 1.977 ms)"),
    ('wgmma', Kind(dense=True, made=Made.KERNEL_GRADIENT, matrix=(2560, 1024))): (
        MOSAIC, f"{M_EM_52}: the template 0.784x XLA's dot (0.189 against 0.241 ms)"),
    ('wgmma', Kind(dense=True, made=Made.KERNEL_GRADIENT, matrix=(2560, 64))): (
        MOSAIC, f"{M_EM_52}: the template 0.39x XLA's dot (0.053 against 0.136 ms)"),
    ('tcgen05', Kind(dense=True, made=Made.FORWARD, matrix=(2560, 64))): (
        XLA, f"{B200_GRID}: the template 1.216x XLA's dot (em35d 1.225x)"),
    ('tcgen05', Kind(dense=True, made=Made.INPUT_GRADIENT, matrix=(2560, 9216))): (
        XLA, f"{EM35D}: the template 1.009x XLA's dot"),
    ('tcgen05', Kind(dense=True, made=Made.KERNEL_GRADIENT, matrix=(2560, 9216))): (
        XLA, f"{B200_GRID}: the template 1.355x XLA's dot (em35d 1.303x)"),
    ('tcgen05', Kind(dense=True, made=Made.KERNEL_GRADIENT, matrix=(2560, 1024))): (
        XLA, f"{B200_GRID}: the template 1.148x XLA's dot (em35d 1.326x)"),
}
"""The grouped GEMM's declared cells (`selection.Op.declared`), where a route measured faster than the
template's default on a generation, each with its receipt (owner, bible da64ae5; "whatever is the
best/fastest remains the route until our templates catch up", ops 361caffc): (instruction family, kind of
call) to (that route's kernel, the receipt). A product of a matrix a receipt timed (`MEASURED`) is a cell of
its own, which that receipt's winner holds: the template's where it won (on Hopper, over its kind's XLA
cell), XLA's where it lost (on Blackwell, where the template is the kind's default); one timed whose cell is
undeclared takes the template. A cell leaves the list once the template measures at least as fast on it.
A cell's route holds no more of a step's memory than the template's: each product's reverse mode returns
its rows' cotangent only with its kernel gradient formed (`derivatives.with_gradients_formed`, row 6027),
whichever route forms it (tests/test_declared_cells_memory.py). Nothing is ranked at runtime. A
call the template cannot run (its legality) takes the next route in `GROUPED`'s order, which is no cell of
this list."""

GROUPED: selection.Op[GroupedCase] = selection.Op(generation.Choice.GROUPED_MATMUL, (
    MOSAIC, TRITON, CUTLASS, PRODUCTS_APART, XLA), declared=DECLARED)
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
    extents = _extents_of(launches, arguments, options)
    rows = extents[0]
    # A left operand a prologue forms (a norm's, fused by the algebra's recognition) is a cell of its own
    # where a receipt timed it formed; elsewhere it is its matrix's cell as any call's (train 977 F1: the
    # 4B's normed projections keep the template FOUR_B's receipts measured).
    family = generation.family(call.target.generation)

    def timed(*, formed: bool) -> bool:
        return matrix is not None and any(timed.holds(matrix, rows)
                                          for timed in MEASURED.get((family, made, formed), ()))

    formed = options['hooks'].prologue is not None and timed(formed=True)
    # Products side by side or chained are one launch's, a cell of their own, keyed by their joined matrix
    # where a receipt timed it (`TOGETHER`), whatever their first matrix's.
    compound = _compound(call)
    together = _together(compound, matrix)
    if together is not None:
        # Of the arithmetic the receipt timed: its operands' dtype too (base review b68fc2382 F4: an fp16
        # chain inherited the bf16 win).
        arithmetic = (family, made, jnp.dtype(arguments[0].dtype).name)
        timed_together = any(receipt.timed.holds(together, rows) for receipt in TOGETHER
                             if (receipt.family, receipt.made, receipt.dtype) == arithmetic)
        matrix_of = together if timed_together else None
    else:
        matrix_of = matrix if formed or timed(formed=False) else None
    kind = Kind(bool(options['dense']), made, fp8=fp8, gathered=gathered, scattered=scattered,
                matrix=matrix_of, formed=formed, together=together is not None)
    cuda = generation.capability(call.target.generation)
    floor = cuda is not None and cuda[0] * 10 + cuda[1] >= generation.BF16_GPU
    placed = generation.tpu(call.target) or floor
    # A pair, matrices side by side or chained products: Mosaic GPU's launch alone takes them (`_takes_pair`).
    return GroupedCase(placed, whole, product, call, kind, extents, paired=compound is not None)


def _compound(call) -> tuple[str, tuple] | None:
    """Where `call` is several products in one launch, its form and its further products, as its launches
    bind them: `beside`, matrices side by side; `chained`, `(rows, matrix)` products over joined depths;
    `paired`, a pair's second matrix (`gmm_call`'s), alone; else None. Read once a case, for its cell
    (`_together`) and its launches (`GroupedCase.paired`)."""
    for _, bound in call.bound.values():
        for form in ('beside', 'chained'):
            if others := bound.arguments.get(form):
                return form, tuple(others)
        if (paired := bound.arguments.get('paired')) is not None:
            return 'paired', (paired,)
    return None


def _together(compound: tuple[str, tuple] | None, matrix: tuple[int, int] | None) -> tuple | None:
    """Where a call is products side by side or chained (`compound`, `_compound`'s), their geometry as a
    receipt keys it (`TOGETHER`): the launch's form (`beside`, matrices side by side; `chained`, products over
    joined depths), the first's `matrix`'s shared extent (`_matrix_of`: `[in, out]`; 0 where it is no dense
    product's) and each product's own width in the launch's order, the replay's widths reading them so (base
    review a1ad2b4ca F4: a sum of widths aliased two products and three); else None (a pair's is one
    product's cell)."""
    if compound is None or compound[0] == 'paired':
        return None
    form, others = compound
    depth, width = matrix or (0, 0)
    matrices = others if form == 'beside' else tuple(each for _, each in others)
    return form, depth, (width, *(each.shape[-1] for each in matrices))


def _matrix_of(launches, arguments) -> tuple[int, int]:
    """A dense product's matrix as it lies (`Kind.matrix`): a forward's kernel `[1, in, out]`, its input
    gradient's the same matrix read transposed, its kernel gradient's output (its rows' width by its
    cotangent's), one tuple for a projection's three products."""
    if launches is _TGMMS:
        return arguments[0].shape[1], arguments[1].shape[1]
    return tuple(arguments[1].shape[1:])


def _extents_of(launches, arguments, options) -> tuple[int, int, int]:
    """A call's product's rows, depth and width (`GroupedCase.extents`): a product's walked rows
    (`_rows_of`), its tokens' depth and its matrix's width; a kernel gradient's summed rows, its tokens'
    and its cotangent's columns."""
    rows = _rows_of(launches, arguments)
    if launches is _TGMMS:
        return rows, arguments[0].shape[1], arguments[1].shape[1]
    kernel = arguments[1]
    return rows, arguments[0].shape[1], kernel.shape[1 if options['trans_rhs'] else 2]


def _rows_of(launches, arguments) -> int:
    """The rows a call's product walks: its tokens', or its gather table's where it gathers its tokens (a
    kernel gradient's tokens' table, else its cotangent's, the tokens then whole)."""
    if launches is _TGMMS:
        tables = (arguments[4] if len(arguments) > 4 else None) or (None, None)
        return next((table for table in tables if table is not None), arguments[0]).shape[0]
    return (arguments[7] if len(arguments) > 7 and arguments[7] is not None else arguments[0]).shape[0]


def launched(launches, arguments, options, *, product: Product, whole: bool,
             selected: generation.Route | None = None):
    """The launch of `launches` that runs the call (`arguments`, `options`) as `GROUPED` selects it,
    ready to call, `APART` where a paired call's products run apart, or None where XLA's program takes it:
    on the call's target, a CPU process's own (whose
    kernels run nowhere, so XLA's program takes every call and a kernel `generation.chosen` names raises)
    or the GPU it states (`generation.assumed`). Every branch insists on a named kernel. `selected`, the
    route a caller's own selection of this call found unreported (a reverse pass's one launch for several
    products, `gemm._one_launch`): taken and reported as it is, the call selected once (row 8735)."""
    case = grouped_case(launches, arguments, options, product=product, whole=whole)
    route = selection.select(GROUPED, case) if selected is None else generation.published(selected)
    if (collected := _WEIGHED.get()) is not None:
        collected.append((case.call, route.kernel))
    if route.kernel in ('xla', APART.name):
        return None if route.kernel == 'xla' else APART
    return functools.partial(case.call.implementation(route.kernel), *arguments, **options)


_WEIGHED: contextvars.ContextVar[list | None] = contextvars.ContextVar('weighed', default=None)
"""The calls `launched` selects for while `weighed` collects them."""


@contextlib.contextmanager
def weighed() -> Iterator[list[tuple[base.Call, str]]]:
    """Each call traced in the block that `launched` selects for, in order, as its templates' bindings
    (`base.Call`) and the launch it selected (a template's name, 'xla' or `APART`'s): what a pass weighing a
    region reads of the launches it would make (`fused_gemm.analysis`), whether or not they are built."""
    found: list[tuple[base.Call, str]] = []
    token = _WEIGHED.set(found)
    try:
        yield found
    finally:
        _WEIGHED.reset(token)


def resident(hooked: Sequence[tuple[base.Call, str]],
             unhooked: Callable[[], Sequence[tuple[base.Call, str]]]) -> bool:
    """Whether a product's Mosaic launch with its hooks keeps the residency the same launch takes without
    them, each the last call `weighed` collected, `unhooked`'s weighed only where `hooked`'s is a Mosaic
    launch (kernel-templates §4: a neighbour joins the kernel where the kernel with it still holds its
    residency and stays spill-free; KERNEL-TEMPLATES-13, row 9032): the programs a multiprocessor of the
    tile it derives (`Template.derived`, its plan's: the residency `mosaic.Family.kernel` launches at, its
    registers within the family's budget) at least those of the tile it derives for the product unhooked,
    whatever launch that call selects. A hooked launch that derives no tile holds none; any other launch,
    or no product unhooked, is not judged. On an A100 an epilogue reading four float32 arrays of the
    product's shape derived 64 x 64 at one program a multiprocessor, where the product alone takes two
    (FS-HOOKED-RES)."""
    launched = _mosaic_launch(hooked)
    if launched is None:
        return True
    kernel, template, ba = launched
    target = generation.gpu_target()
    config = template.derived(ba, target)
    if config is None:
        return False
    bare = unhooked()
    if not bare:
        return True
    plain_template, plain_ba = bare[-1][0].bound[kernel]
    plain = plain_template.derived(plain_ba, target)
    return plain is None or config.programs >= plain.programs


def special_forms(hooked: Sequence[tuple[base.Call, str]]) -> Mapping | None:
    """The special functions' forms the last launch `weighed` collected compiles its hooks' in, where it is a
    Mosaic template's at a tile it derives: those of the compiler parameters its family compiles with
    (`mosaic.Transport.compiled_with` at `mosaic_gpu.APPROX_MATH`), read as a kernel equation's are
    (`emission.forms_of`); None for any other launch, or none (one MUFU an element,
    `emission.operation_work`)."""
    from .templates import emission
    from .templates.gemm import mosaic_gpu, pricing

    launched = _mosaic_launch(hooked)
    if launched is None:
        return None
    _, template, ba = launched
    config = template.derived(ba, generation.gpu_target())
    if config is None:
        return None
    family = pricing.family_of(config, ba.arguments['tokens'].dtype)
    return emission.forms_of(family.compiled_with(approx_math=mosaic_gpu.APPROX_MATH))


def _mosaic_launch(hooked: Sequence[tuple[base.Call, str]]
                   ) -> tuple[str, _Mosaic, tokamax.BoundArguments] | None:
    """The last launch `weighed` collected where it is a Mosaic template's (`_Mosaic`): its kernel's name, its
    template and its call's bindings; None for any other launch, or none."""
    if not hooked:
        return None
    call, kernel = hooked[-1]
    if kernel not in call.bound:
        return None
    template, ba = call.bound[kernel]
    if not isinstance(template, _Mosaic):
        return None
    return kernel, template, ba


def gmm_call(tokens, kernel, sizes, prologue_operands: tuple = (), epilogue_operands: tuple = (),
             paired: jax.Array | jax.ShapeDtypeStruct | None = None, first: jax.Array | None = None,
             gather: jax.Array | None = None, scatter: jax.Array | None = None,
             joined: jax.Array | None = None, beside: tuple = (), chained: tuple = (), *, out_dtype,
             trans_rhs: bool, hooks: Hooks = NO_HOOKS, interpret: bool, promoted: bool = False,
             dense: bool = False) -> tuple:
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
    order); Mosaic GPU's launch alone takes either; `joined`, rows of `tokens`' shape that `paired` multiplies
    into the one product (`mosaic_gpu.gmm`'s `joined`: a paired product's input gradient, row 6783);
    `beside`, matrices of `kernel`'s groups and depth side by side with it, each its own output (Mosaic
    GPU's launch alone takes them: `mosaic_gpu.gmm`'s `beside`, row 6794); `chained`, further `(rows,
    matrix)` products summed with it over their contractions joined end to end (Mosaic GPU's launch alone:
    `mosaic_gpu.gmm`'s `chained`, a multi-matrix input gradient, row 8282)."""
    optional = (paired, first, gather, scatter, joined, tuple(beside) or None,
                tuple(map(tuple, chained)) if chained else None)
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
