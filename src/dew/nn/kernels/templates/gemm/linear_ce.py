"""GEMM-fused linear cross-entropy on the shared Mosaic toolkit (docs/design/kernel-templates.md Appendix C
row 25, row 6103): computes per-tile online softmax summaries `(m, l, target_logit)` directly from each
`[block_m, block_n]` fp32 logit accumulator tile without ever materializing `[M, V]` logits in global memory,
then merges column-tile summaries via `dew.nn.categorical.rescaled` and `dew.nn.categorical.target_loss`."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from typing import ClassVar

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn import categorical
from dew.nn.kernels import generation, grouped_launch, selection, tuning, xla_products
from dew.nn.kernels.templates import (
    bundles,
    cost as unit_cost,
    mosaic,
    outputs,
    report,
    resources,
    rings,
    tail as tail_of,
)
from dew.nn.kernels.templates.gemm import derivation
from dew.nn.kernels.templates.gemm.mosaic_gpu import (
    APPROX_MATH,
    MOVES,
    Allocations,
    _oriented,
    _typed,
    _walked_products,
    multicast,
    pair_takes,
)
from dew.nn.kernels.templates.gemm.pricing import (
    TilesTemplate,
    epilogue_registers,
    family_of,
    granular_tile,
    tile_columns,
    tile_rows,
)
from dew.nn.kernels.templates.gemm.tiling import CONSUMERS, Tiles
from dew.nn.precision import exact_products
from dew.nn.sharding import automatic_axes

LINEAR_CE = "dew_linear_ce"
"""Kernel name for GEMM-fused linear cross-entropy launches."""


def supports_dtype(dtype) -> bool:
    """Whether `linear_ce` accepts `dtype` inputs (16-bit real floats for native Mosaic MMA)."""
    return jnp.dtype(dtype) in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16))


def linear_ce_allocations(tiles: Tiles, family: mosaic.Family, *, dtype, transpose_rhs: bool,
                          has_bias: bool = False, interpret: bool = False) -> Allocations:
    """Operand ring and side-vector staging allocations for `linear_ce`: rows `[block_m, block_k]`, vocab
    weights `[block_n, block_k]` (transposed) or `[block_k, block_n]`, staged `targets` / optional `bias`,
    and 1D per-row summary staging (`family.vector_stage`)."""
    del interpret
    block_m, block_n, block_k = tiles.block_m, tiles.block_n, tiles.block_k
    columns = tile_columns(tiles, family)
    rhs_block = (block_n, block_k) if transpose_rhs else (block_k, block_n)
    blocks = {"rows": ((block_m, block_k), dtype), "matrix": (rhs_block, dtype)}
    side_specs = [((tiles.share,), _typed(jnp.int32))]
    if has_bias:
        side_specs.append(((columns,), _typed(jnp.float32)))
    sides = family.staging("side", side_specs)
    row_stage = tuple(family.vector_stage((tiles.warpgroups, tiles.share), jnp.float32))
    return Allocations(blocks, (sides["side_smems"], sides["side_barriers"], row_stage), (), ("matrix",), ())


def linear_ce_registers(tiles: Tiles, dtype, *, has_bias: bool, target) -> int | None:
    """Consumer thread register count for `linear_ce` (`resources.tile_registers` + shared
    `pricing.epilogue_registers`)."""
    family = family_of(tiles, dtype, target)
    if not family.budgets_registers or generation.tpu(target):
        return None
    columns = tile_columns(tiles, family)
    plain = resources.tile_registers(family, tiles.sharing, (tiles.block_m, columns, tiles.block_k), dtype,
                                     stored=(jnp.float32,), rhs_transposed=True)
    # 4 running row vectors (m, l, target_logit, rest) + targets_row (+ optional bias column vector).
    vectors = 5 * family.row_registers(tiles.share, jnp.float32)
    if has_bias:
        vectors += family.column_registers(columns, jnp.float32)
    epilogue = epilogue_registers(family, tiles.share, columns, (jnp.float32,), vectors=vectors,
                                  has_hook=True)
    return plain + epilogue


def linear_ce_launch_of(tiles: Tiles, shape: tuple[int, int, int], dtype, *, transpose_rhs: bool,
                        has_bias: bool, target, interpret: bool = False) -> resources.Launch:
    """Resource planner `Launch` for `linear_ce` at `tiles` on `target` (`resources.plan`)."""
    _, _, k = shape
    family = family_of(tiles, dtype, target)
    steps = -(-k // tiles.block_k)

    def scope(stages: int) -> resources.Scope:
        staged = dataclasses.replace(tiles, stages=stages)
        return linear_ce_allocations(
            staged, family, dtype=dtype, transpose_rhs=transpose_rhs, has_bias=has_bias,
            interpret=interpret).scope(family, staged, steps, reduces=True,
                                       reducing_warpgroups=tiles.warpgroups)

    shared = resources.program_shared(generation.shared_budget(target), tiles.programs)
    most = min(shared // ((tiles.block_m + tiles.block_n) * tiles.block_k), max(2, steps))
    most = family.ring_slots or most
    regs = linear_ce_registers(tiles, dtype, has_bias=has_bias, target=target)
    return resources.Launch(scope, tiles.warpgroups, regs, most=most, producer=tiles.producers.of(LINEAR_CE))


def _linear_ce_finish_tile(wg, tile, accs, *, targets_gmem, bias_gmem, side_smems, side_barriers, row_stage,
                           out_m, out_l, out_t, out_r, vocab: int, softcap: float | None, columns: int,
                           family: mosaic.Family, tiles: Tiles, round_logits: bool = False) -> None:
    """Reduce one `[share, columns]` fp32 logit tile into per-row `(m, l, target_logit, rest)` summaries
    through the toolkit's native side-operand staging (`family.loaded`/`side_tile`) and vector store
    (`family.stored`/`as_row`); with `round_logits` each logit rounds to bf16 before its bias and softmax,
    as a bf16 head's product does (`precision.bf16_product`)."""
    row_start, ni, col_tile = tile
    part_row = row_start + tiles.share_row(wg)
    row_slice = pl.ds(part_row, tiles.share)
    loads = [(targets_gmem.at[row_slice], side_smems[0].at[wg], side_barriers[0].at[wg])]
    load_axes: tuple[tuple[str, ...], ...] = (("rows",),)
    if bias_gmem is not None:
        loads.append((bias_gmem.at[pl.ds(col_tile, columns)], side_smems[1].at[wg], side_barriers[1].at[wg]))
        load_axes = (("rows",), ("columns",))
    family.loaded(loads, axes=load_axes)
    targets_row = family.side_tile(side_smems[0].at[wg], ("rows",))

    m_acc = l_acc = t_acc = r_acc = None
    for start, (chunk,) in family.chunks(accs):
        z = chunk.astype(jnp.bfloat16).astype(jnp.float32) if round_logits else chunk
        if bias_gmem is not None:
            bias_col = family.side_tile(side_smems[1].at[wg].at[slice(start, start + z.shape[1])],
                                        ("columns",)).astype(jnp.float32)
            z = z + jax.lax.broadcast_in_dim(bias_col, z.shape, (1,))
        if softcap is not None:
            cap = jnp.float32(softcap)
            z = cap * jnp.tanh(z / cap)
        col_global = col_tile + start + family.iota(z.shape, 1)
        in_vocab = col_global < vocab
        z_masked = jnp.where(in_vocab, z, jnp.float32(-jnp.inf))
        m_chunk = jnp.max(z_masked, axis=1)
        peak_chunk = jnp.where(jnp.isfinite(m_chunk), m_chunk, jnp.float32(0.0))
        exp_z = jnp.exp(z_masked - jax.lax.broadcast_in_dim(peak_chunk, z.shape, (0,)))
        l_chunk = jnp.sum(exp_z, axis=1)
        hit = in_vocab & (col_global == jax.lax.broadcast_in_dim(targets_row, z.shape, (0,)))
        t_chunk = jnp.sum(jnp.where(hit, z, jnp.float32(0.0)), axis=1)
        r_chunk = jnp.sum(jnp.where(hit, jnp.float32(0.0), exp_z), axis=1)
        if m_acc is None:
            m_acc, l_acc, t_acc, r_acc = m_chunk, l_chunk, t_chunk, r_chunk
        else:
            assert l_acc is not None and t_acc is not None and r_acc is not None
            merged_m = jnp.maximum(m_acc, m_chunk)
            merged_peak = jnp.where(jnp.isfinite(merged_m), merged_m, jnp.float32(0.0))
            l_acc = (categorical.rescaled(m_acc, l_acc, merged_peak)
                     + categorical.rescaled(m_chunk, l_chunk, merged_peak))
            r_acc = (categorical.rescaled(m_acc, r_acc, merged_peak)
                     + categorical.rescaled(m_chunk, r_chunk, merged_peak))
            m_acc = merged_m
            t_acc = t_acc + t_chunk

    assert m_acc is not None and l_acc is not None and t_acc is not None and r_acc is not None
    row_smem = row_stage[0].at[wg] if row_stage else None
    family.stored((family.as_row(m_acc), family.as_row(l_acc), family.as_row(t_acc), family.as_row(r_acc)),
                  (out_m.at[ni, row_slice], out_l.at[ni, row_slice],
                   out_t.at[ni, row_slice], out_r.at[ni, row_slice]),
                  smem=row_smem)


def _merge_summaries(out_m: jax.Array, out_l: jax.Array, out_t: jax.Array, out_r: jax.Array
                     ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Merge per-column-tile `(m, l, target_logit, rest)` summaries `[col_tiles, M]` into exact
    `(loss, peak, l, rest)` via `dew.nn.categorical.rescaled` and `dew.nn.categorical.target_loss`."""
    m_all = jnp.max(out_m, axis=0)
    peak = jnp.where(jnp.isfinite(m_all), m_all, jnp.float32(0.0))
    l_all = jnp.sum(categorical.rescaled(out_m, out_l, peak[None, :]), axis=0)
    r_all = jnp.sum(categorical.rescaled(out_m, out_r, peak[None, :]), axis=0)
    t_all = jnp.sum(out_t, axis=0)
    return categorical.target_loss(peak, l_all, t_all), peak, l_all, r_all


def linear_ce(x: jax.Array, weight: jax.Array, targets: jax.Array, *, tiles: Tiles,
              transpose_rhs: bool = True, bias: jax.Array | None = None, softcap: float | None = None,
              round_logits: bool = False, multiprocessors: int,
              interpret: bool | plgpu.InterpretGPUParams = False, registers: int | None = None,
              shared: int | None = None) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """GEMM-fused linear cross-entropy forward returning `(loss, peak, l, rest)` each `[M]` fp32 without
    materializing `[M, V]` logits in global memory (row 6103)."""
    if x.ndim != 2 or weight.ndim != 2 or targets.ndim != 1:
        raise ValueError(f"linear_ce expects 2D x/weight and 1D targets, got "
                         f"{x.shape}, {weight.shape}, {targets.shape}")
    if tiles.staging is not None:
        raise ValueError(f"linear_ce stages its summaries at its family's own chunks, not {tiles.staging}")
    m, k = x.shape
    v, kw = weight.shape if transpose_rhs else (weight.shape[1], weight.shape[0])
    if kw != k or targets.shape[0] != m:
        raise ValueError(
            f"linear_ce shape mismatch: x={x.shape}, weight={weight.shape}, targets={targets.shape}")
    if bias is not None and bias.shape != (v,):
        raise ValueError(f"linear_ce bias must have shape ({v},), got {bias.shape}")
    if jnp.dtype(x.dtype) != jnp.dtype(weight.dtype):
        raise ValueError(
            f"native linear_ce requires matching x and weight dtypes, got {x.dtype} and {weight.dtype}")

    family = family_of(tiles, x.dtype)
    m_unit, n_unit, k_unit = tile_rows(tiles, family), tile_columns(tiles, family), tiles.block_k
    m_pad = -(-m // m_unit) * m_unit
    v_pad = -(-v // n_unit) * n_unit
    k_pad = -(-k // k_unit) * k_unit
    if m_pad != m or k_pad != k:
        x = jnp.pad(x, ((0, m_pad - m), (0, k_pad - k)))
    if v_pad != v or k_pad != k:
        w_pad = ((0, v_pad - v), (0, k_pad - k)) if transpose_rhs else ((0, k_pad - k), (0, v_pad - v))
        weight = jnp.pad(weight, w_pad)
    targets_i32 = targets.astype(jnp.int32)
    if m_pad != m:
        targets_i32 = jnp.pad(targets_i32, ((0, m_pad - m),), constant_values=jnp.int32(-1))
    bias_f32 = None if bias is None else bias.astype(jnp.float32)
    if bias_f32 is not None and v_pad != v:
        bias_f32 = jnp.pad(bias_f32, ((0, v_pad - v),))
    out_m, out_l, out_t, out_r = _linear_ce_mainloop(
        x, weight, targets_i32, bias_f32, (m_pad, v_pad, k_pad, v), tiles=tiles,
        transpose_rhs=transpose_rhs, softcap=softcap, round_logits=round_logits,
        multiprocessors=multiprocessors, interpret=interpret, has_bias=bias_f32 is not None,
        registers=registers, shared=shared)
    if m_pad != m:
        out_m, out_l, out_t, out_r = out_m[:, :m], out_l[:, :m], out_t[:, :m], out_r[:, :m]
    return _merge_summaries(out_m, out_l, out_t, out_r)


@tuning.lowered_once(("shape", "tiles", "transpose_rhs", "softcap", "round_logits", "multiprocessors",
                      "interpret", "has_bias", "registers", "shared"))
def _linear_ce_mainloop(x: jax.Array, weight: jax.Array, targets: jax.Array, bias: jax.Array | None,
                        shape: tuple[int, int, int, int], *, tiles: Tiles, transpose_rhs: bool,
                        softcap: float | None, round_logits: bool = False, multiprocessors: int, interpret,
                        has_bias: bool = False, registers: int | None = None, shared: int | None = None
                        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compiled/interpreted persistent Mosaic GEMM + online-softmax summary mainloop."""
    m, v_pad, k, vocab = shape
    family = family_of(tiles, x.dtype)
    block_m, block_n, block_k = tiles.block_m, tile_columns(tiles, family), tiles.block_k
    consumers, pair, part = tiles.warpgroups, tiles.cluster, (tiles.share, block_n)
    steps = k // block_k
    grid = (m // (block_m * pair), v_pad // block_n)
    total_tiles = grid[0] * grid[1]
    allocations = linear_ce_allocations(tiles, family, dtype=x.dtype, transpose_rhs=transpose_rhs,
                                        has_bias=has_bias, interpret=bool(interpret))

    def body(*gmems):
        if bias is not None:
            x_gmem, w_gmem, targets_gmem, bias_gmem, out_m, out_l, out_t, out_r = gmems
        else:
            x_gmem, w_gmem, targets_gmem, out_m, out_l, out_t, out_r = gmems
            bias_gmem = None

        def schedule(index):
            mi, ni = tail_of.planar_snake(index[0], grid, 1, tiles.grid_block_n)
            row_start = mi * (block_m * pair) + mosaic.cluster_index(pair) * block_m
            col_start = ni * block_n
            if family.pairs_product:
                pair_rows = pl.ds(mi * (block_m * pair), block_m * pair)
                views = {
                    "rows": (x_gmem.at[pair_rows], lambda j: rings.Paired((0, j), 0)),
                    "matrix": (w_gmem.at[pl.ds(col_start, block_n)] if transpose_rhs
                               else w_gmem.at[:, pl.ds(col_start, block_n)],
                               lambda j: rings.Paired((0, j) if transpose_rhs else (j, 0),
                                                      0 if transpose_rhs else 1)),
                }
            else:
                views = {
                    "rows": (x_gmem.at[pl.ds(row_start, block_m)], lambda j: (0, j)),
                    "matrix": (w_gmem.at[pl.ds(col_start, block_n)] if transpose_rhs
                               else w_gmem.at[:, pl.ds(col_start, block_n)],
                               (lambda j: (0, j)) if transpose_rhs else (lambda j: (j, 0))),
                }
            return jnp.ones((), jnp.bool_), steps, views, (row_start, ni, col_start)

        def scope(side_smems, side_barriers, row_stage, *, streams):
            def issue(wg, tile, walk, *, active):
                del tile, active
                mine_rows = pl.ds(tiles.share_row(wg), tiles.share)

                def step(index, slots, carry, accs):
                    del index
                    oriented = _oriented({"matrix": slots["matrix"]}, {"matrix": "rows"}, transpose_rhs)
                    lhs_ref = slots["rows"].at[mine_rows]
                    return carry, (family.products(accs[0], lhs_ref, oriented["matrix"]),)
                return _walked_products(family, walk, step, part, 1, exact_sums=False, prefetch=None)
            def finish(wg, tile, accs, store, *, active):
                del store, active
                _linear_ce_finish_tile(
                    wg, tile, accs, targets_gmem=targets_gmem, bias_gmem=bias_gmem,
                    side_smems=side_smems, side_barriers=side_barriers, row_stage=row_stage,
                    out_m=out_m, out_l=out_l, out_t=out_t, out_r=out_r, vocab=vocab, softcap=softcap,
                    columns=block_n, family=family, tiles=tiles, round_logits=round_logits)
            family.persist(streams, (total_tiles,), schedule, issue, finish, consumers=consumers,
                           axis="programs", outputs=allocations.outputs, interpret=bool(interpret),
                           in_flight=family.in_flight, unrolled=True, accumulators=(part,),
                           producer=tiles.producers.of(LINEAR_CE))

        side_specs = [((tiles.share,), targets_gmem)]
        if bias_gmem is not None:
            side_specs.append(((block_n,), bias_gmem))
        sides = family.staging("side", side_specs)
        row_stage = family.vector_stage((consumers, tiles.share), jnp.float32)
        family.scoped(scope, sides["side_smems"], sides["side_barriers"], row_stage,
                      streams=family.streams(allocations.blocks, tiles.stages, tiles.sharing,
                                             multicast=multicast(tiles, family, allocations.shared)))

    summary_type = (outputs.output(jnp.float32, grid[1], m),) * 4
    operands = (x, weight, targets, bias) if bias is not None else (x, weight, targets)
    programs = max(total_tiles * pair, pair) if family.walks_scoped and not interpret else multiprocessors
    flops = 2 * m * v_pad * k
    kernel = family.kernel(body, out_type=summary_type, grid=(programs // pair,), grid_names=("programs",),
                           consumers=consumers, interpret=interpret, approx_math=APPROX_MATH, name=LINEAR_CE,
                           cluster=pair, resident=tiles.programs, scalars=0, registers=registers,
                           reduces=True, reducing_warpgroups=consumers,
                           producer=tiles.producers.of(LINEAR_CE), shared=shared, moves=MOVES,
                           cost=unit_cost.estimate(flops, *operands, *summary_type))
    return kernel(*operands)


def derive_linear_ce(shape: tuple[int, int, int], dtype, target, *, transpose_rhs: bool = True,
                     has_bias: bool = False, interpret: bool = False,
                     refused: Callable[[Tiles], bool] | None = None, ba=None) -> Tiles | None:
    """Derive a legal `Tiles` for `linear_ce` on `target` via shared `derivation.derived` and
    `MOSAIC_LINEAR_CE.candidate`."""
    if not supports_dtype(dtype):
        return None
    m, v, k = shape
    groups = bundles.groups(target.generation, dtype, stated=True, tpu=generation.tpu(target))
    if not groups:
        return None
    if ba is None:
        w_shape = (v, k) if transpose_rhs else (k, v)
        bias_spec = jax.ShapeDtypeStruct((v,), jnp.float32) if has_bias else None
        ba = MOSAIC_LINEAR_CE.bind(jax.ShapeDtypeStruct((m, k), dtype),
                                   jax.ShapeDtypeStruct(w_shape, dtype),
                                   jax.ShapeDtypeStruct((m,), jnp.int32), bias_spec,
                                   transpose_rhs=transpose_rhs, softcap=None, interpret=interpret)
    multiprocessors = target.multiprocessors or 1
    for tried in (tried for layouts in groups for tried in derivation.paired_first(layouts, m)):
        family0 = tried[0]
        row_unit = family0.role_rows * min(family0.consumers, CONSUMERS) * family0.cluster
        col_unit = family0.column_unit(dtype) * (family0.cluster if family0.pairs_product else 1)
        step_unit = family0.depths(dtype)[-1]
        aligned_m = -(-m // row_unit) * row_unit
        aligned_v = -(-v // col_unit) * col_unit
        aligned_k = -(-k // step_unit) * step_unit

        def candidate(tile: Tiles) -> bool:
            if refused is not None and refused(tile):
                return False
            return MOSAIC_LINEAR_CE.candidate(ba, tile, target)

        def launch_of(tile: Tiles, *, rm: int = aligned_m, rv: int = aligned_v,
                      rk: int = aligned_k) -> resources.Launch:
            return linear_ce_launch_of(tile, (rm, rv, rk), dtype, transpose_rhs=transpose_rhs,
                                       has_bias=has_bias, target=target, interpret=interpret)

        tile = derivation.derived(tried, rows=aligned_m, depth=aligned_k, width=aligned_v, dtype=dtype,
                                  multiprocessors=multiprocessors, target=target, candidate=candidate,
                                  launch_of=launch_of, grouped=False, dense=True, ragged=True)
        if tile is not None:
            return tile
    return None


@dataclasses.dataclass(frozen=True)
class MosaicLinearCE(TilesTemplate):
    """GEMM-fused linear cross-entropy on the shared `TilesTemplate`/`KernelOp` certification path
    (row 6103)."""

    KERNEL: ClassVar[str] = LINEAR_CE
    config_cls: ClassVar[type[Tiles]] = Tiles

    def _supports_ba(self, ba) -> bool:
        x, weight = ba.arguments["x"], ba.arguments["weight"]
        return supports_dtype(x.dtype) and jnp.dtype(x.dtype) == jnp.dtype(weight.dtype)
    def _shape(self, ba) -> tuple[int, int, int]:
        x, w = ba.arguments["x"], ba.arguments["weight"]
        v = w.shape[0] if ba.arguments["transpose_rhs"] else w.shape[1]
        return x.shape[0], v, x.shape[1]

    def launch_of(self, ba, config: Tiles, target=None) -> resources.Launch:
        target = generation.gpu_target() if target is None else target
        return linear_ce_launch_of(config, self._shape(ba), ba.arguments["x"].dtype,
                                   transpose_rhs=ba.arguments["transpose_rhs"],
                                   has_bias=ba.arguments["bias"] is not None, target=target,
                                   interpret=ba.arguments["interpret"])

    def registers(self, ba, config: Tiles, target=None) -> int | None:
        target = generation.gpu_target() if target is None else target
        return linear_ce_registers(config, ba.arguments["x"].dtype,
                                   has_bias=ba.arguments["bias"] is not None, target=target)

    def candidate(self, ba, config: Tiles, target) -> bool:
        x, weight = ba.arguments["x"], ba.arguments["weight"]
        if (not supports_dtype(x.dtype) or jnp.dtype(x.dtype) != jnp.dtype(weight.dtype)
                or report.refused(self, ba, config, target)):
            return False
        groups = bundles.groups(target.generation, x.dtype, stated=True, tpu=generation.tpu(target))
        family = next((b for group in groups for b in group if b.name == config.family), None)
        if family is None or config.cluster > family.cluster or not pair_takes(config, family):
            return False
        if (config.splits != 1 or config.pingpong or config.swapped
                or (family.pairs_product and ba.arguments["interpret"])):
            return False
        return granular_tile(
            family, config, x.dtype, target=target,
            stream_blocks=lambda: linear_ce_allocations(
                config, family, dtype=x.dtype, transpose_rhs=ba.arguments["transpose_rhs"],
                has_bias=ba.arguments["bias"] is not None,
                interpret=ba.arguments["interpret"]).blocks)

    def derived(self, ba, target) -> Tiles | None:
        return derive_linear_ce(self._shape(ba), ba.arguments["x"].dtype, target,
                                transpose_rhs=ba.arguments["transpose_rhs"],
                                has_bias=ba.arguments["bias"] is not None,
                                interpret=ba.arguments["interpret"],
                                refused=lambda candidate: report.refused(self, ba, candidate, target), ba=ba)

    def _fwd(self, x: jax.Array, weight: jax.Array, targets: jax.Array, bias: jax.Array | None = None, *,
             transpose_rhs: bool = True, softcap: float | None = None, round_logits: bool = False,
             interpret: bool = False, config: Tiles, return_residuals: bool = False):
        del return_residuals
        target = generation.gpu_target()
        bound = self.bind(x, weight, targets, bias, transpose_rhs=transpose_rhs, softcap=softcap,
                          round_logits=round_logits, interpret=interpret)
        mp = config.cluster if interpret else unit_cost.wave(target.multiprocessors, config.programs)
        regs = self.registers(bound, config, target)
        shared = regs and self.staged_bytes(bound, config, target)
        out = linear_ce(x, weight, targets, tiles=config, transpose_rhs=transpose_rhs, bias=bias,
                        softcap=softcap, round_logits=round_logits, multiprocessors=mp,
                        interpret=grouped_launch.INTERPRETED if interpret else False,
                        registers=regs, shared=shared)
        return out, None


MOSAIC_LINEAR_CE = MosaicLinearCE()
"""The singleton Mosaic GEMM-fused linear cross-entropy launch on `TilesTemplate`/`KernelOp`."""

LINEAR_CE_ROW = "ledger 6103"


@dataclasses.dataclass(eq=False)
class LinearCECase:
    """Facts deciding whether `chunked._forward` runs Mosaic GEMM-fused CE or chunked XLA: its operands'
    dtypes, sharding and precision, and its tile (`plan`'s), planned only once the requirements before
    `launch` take the call, so a call the `measured` gate keeps on chunked XLA plans no tile."""
    whole: bool
    exact_products: bool
    dtype_ok: bool
    plan: Callable[[], Tiles | None]

    @functools.cached_property
    def tile(self) -> Tiles | None:
        return self.plan()


LINEAR_CE_OP: selection.Op[LinearCECase] = selection.Op(generation.Choice.LINEAR_CE, (
    selection.Kernel("mosaic", (
        selection.Requirement(
            "dtype", lambda case: case.dtype_ok,
            "Mosaic linear-CE requires matching bfloat16/float16 operand dtypes", LINEAR_CE_ROW),
        selection.Requirement(
            "sharding", lambda case: case.whole,
            "automatic mesh axes keep chunked XLA outside per-shard launch", LINEAR_CE_ROW),
        selection.Requirement(
            "precision", lambda case: case.exact_products,
            "precision does not match bf16/fp16 exact fp32 accumulation", LINEAR_CE_ROW),
        selection.Requirement(
            "measured", lambda case: False,
            "no receipt yet has Mosaic linear-CE beat chunked XLA (pending M-NG-6)",
            LINEAR_CE_ROW, kind="measured"),
        selection.Requirement(
            "launch", lambda case: case.tile is not None,
            "no legal Mosaic linear-CE tile holds this shape/dtype on the target", LINEAR_CE_ROW),
    )),
    selection.Kernel("xla", (), generic=True),
))
"""The LM loss's fused linear cross-entropy selector, its own choice (`linear_ce`, row 12457): naming the
grouped products' kernel (`grouped_matmul`) leaves the loss on its chunked XLA program, and only `linear_ce`
named `mosaic` forces the fused kernel, whose refusals (precision, launch) are then the run's to see."""


def mosaic_linear_ce(x: jax.Array, weight: jax.Array, targets: jax.Array,
                     bias: jax.Array | None = None, *, transpose_rhs: bool = True,
                     precision: jax.lax.PrecisionLike = None
                     ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array] | None:
    """Run `MOSAIC_LINEAR_CE` when `LINEAR_CE_OP` selects `'mosaic'` (returning `(loss, peak, mass,
    rest)`), else `None` so `dew.objectives.lm.chunked._forward` runs its portable chunked XLA path
    untouched (rows 6103, 12136). A bf16 head (`precision` the `BF16_BF16_F32` algorithm) rounds each logit
    to bf16 before its bias and softmax, as `chunked._tile_logits` does forward and backward."""
    from dew.nn.kernels import gemm

    dtype_ok = supports_dtype(x.dtype) and jnp.dtype(x.dtype) == jnp.dtype(weight.dtype)
    exact = exact_products(precision, x.dtype) if dtype_ok else False
    # A bf16 head's algorithm (`chunked.BF16`), whose every logit rounds to bf16.
    round_logits = precision is jax.lax.DotAlgorithmPreset.BF16_BF16_F32
    with (xla_products.committed_context((x, weight, targets)),
          xla_products.interpreted_context(interpret_on_cpu=True)):
        interpreting = xla_products.interpreted(interpret_on_cpu=True)

        def plan() -> Tiles | None:
            if interpreting:
                return derive_linear_ce(
                    (x.shape[0], weight.shape[0] if transpose_rhs else weight.shape[1], x.shape[1]),
                    x.dtype, generation.interpreted_target(), transpose_rhs=transpose_rhs,
                    has_bias=bias is not None, interpret=True)
            return gemm._planned(MOSAIC_LINEAR_CE, MOSAIC_LINEAR_CE.bind(
                jax.ShapeDtypeStruct(x.shape, x.dtype), jax.ShapeDtypeStruct(weight.shape, weight.dtype),
                jax.ShapeDtypeStruct(targets.shape, jnp.int32),
                None if bias is None else jax.ShapeDtypeStruct(bias.shape, bias.dtype),
                transpose_rhs=transpose_rhs, softcap=None, round_logits=round_logits, interpret=False))

        case = LinearCECase(not automatic_axes(), exact, dtype_ok, plan)
        if selection.select(LINEAR_CE_OP, case).kernel != "mosaic":
            return None
        return MOSAIC_LINEAR_CE(x, weight, targets, bias, transpose_rhs=transpose_rhs, softcap=None,
                                round_logits=round_logits, interpret=interpreting)
