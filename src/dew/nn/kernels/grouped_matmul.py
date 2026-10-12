"""The MoE grouped matmul on JAX's Pallas kernels, `gmm` and `tgmm`.

`ragged_dot.py` vendors the kernels; this module adds which products they
compute exactly (`ragged_dot_refusal`) and the custom VJP that trains through
them (`grouped_projection`), in MaxText's megablox shape: forward `gmm`,
input gradient `gmm` against the transposed kernel, kernel gradient `tgmm`.

They are Triton kernels, compiled only for a CUDA lowering, chosen where the
call lowers (`jax.lax.platform_dependent`), so a CPU computation on a GPU
host lowers to `jax.lax.ragged_dot`; an explicit request interprets them on
the CPU, as the CPU suite checks them. Each launch's tile is measured per
GPU model and shape (`Gmm`, `Tgmm`, `dew.nn.kernels.tuning`); JAX's
`_hyperparam_selection_rule` is the tile without a measurement. They stay on sm80 to sm89 despite
jax 0.11.2 deprecating Pallas Triton (it warns at each lowering, left to the
user's filters): JAX's Mosaic GPU grouped matmul needs wgmma, which those
lack, and tokamax's sm80 Mosaic config overflows an Ada card's shared memory
and has no backward (`dew.nn.kernels.generation.triton_runs`).
"""

from __future__ import annotations

import dataclasses
import functools
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from flax.typing import Dtype, PrecisionLike

from ..precision import asks_default_precision
from . import generation, tuning


def ragged_dot_refusal(compute: Dtype, operands: tuple[Dtype, ...],
                       precision: PrecisionLike) -> str | None:
    """Why the kernels do not compute the product a caller asked for, or None where they do.

    The kernels multiply in `compute`, accumulate in fp32 and ignore
    `precision`. With 16-bit compute that gives exact products summed in
    fp32, which any precision asks for. With fp32 compute it gives TF32,
    which only the default precision asks for (explicitly or through
    `jax_default_matmul_precision`). An operand or master wider than fp32
    needs its gradient summed wider than the kernels sum, and x64 widens
    their int32 group offsets, so both are refused.
    """
    if jax.config.jax_enable_x64:
        return "x64 widens the kernels' int32 group offsets"
    if jnp.result_type(compute, *operands, jnp.float32) != jnp.dtype(jnp.float32):
        return "an operand is wider than fp32, and the kernels sum gradients in fp32"
    compute = jnp.dtype(compute)
    if compute in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
        return None
    if compute != jnp.dtype(jnp.float32):
        return f"the kernels do not multiply in {compute.name}"
    return None if asks_default_precision(precision, configured=True) else (
        "fp32 products above the default precision, and the kernels multiply in TF32")


@dataclasses.dataclass(frozen=True)
class TileConfig:
    """A grouped matmul's tile, rows by contraction by columns, and its warps
    and pipeline stages (None: Triton's own)."""
    block_m: int
    block_k: int
    block_n: int
    warps: int | None
    stages: int | None


@dataclasses.dataclass(frozen=True)
class _Grouped(tuning.KernelOp):
    """A grouped matmul launch over `[rows, ...]` tokens and `[groups]` sizes:
    JAX's tile without a measurement, and the tiles of FrontDamselfly's A100
    sweep that fit the GPU's shared memory with their stages."""

    config_cls: ClassVar[type[TileConfig]] = TileConfig

    def _get_heuristics_config(self, ba) -> TileConfig:
        # Imported at first use: the kernels need jax's Pallas Triton backend,
        # which jax 0.11.2 deprecates, and importing a model must not load it.
        from . import ragged_dot
        return TileConfig(**ragged_dot._hyperparam_selection_rule(np.dtype(ba.arguments['tokens'].dtype)),
                          warps=None, stages=None)

    def _get_autotuning_configs(self, ba) -> set[TileConfig]:
        size = np.dtype(ba.arguments['tokens'].dtype).itemsize
        shared = generation.gpu_target().shared_limit or 0
        return {TileConfig(m, k, n, warps, stages) for m in (64, 128, 256) for k in (32, 64, 128)
                for n in (128, 256) for warps in (4, 8) for stages in (2, 3, 4)
                if stages * (m * k + k * n) * size <= shared}

    def operands(self, specs: list[jax.ShapeDtypeStruct], seed: int) -> list[jax.Array]:
        """Normal floats, and group sizes of a Dirichlet draw over the rows
        with the last group empty, as routing leaves one."""
        *arrays, sizes = super().operands(specs, seed)
        rng = np.random.default_rng(seed)
        groups, rows = specs[-1].shape[0], specs[0].shape[0]
        counts = (np.append(rng.multinomial(rows, rng.dirichlet(np.ones(groups - 1))), 0) if groups > 1
                  else np.array([rows]))
        return [*arrays, jnp.asarray(counts, sizes.dtype)]


@dataclasses.dataclass(frozen=True)
class Gmm(_Grouped):
    """`ragged_dot.gmm`: each group's rows of `tokens` against its matrix of
    `kernel`, `[k, n]`, or `[n, k]` transposed (`trans_rhs`, the input
    gradient)."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens',)

    def _fwd(self, tokens: jax.Array, kernel: jax.Array, sizes: jax.Array, *, out_dtype: str,
             trans_rhs: bool, interpret: bool, config: TileConfig, return_residuals: bool = False):
        from . import ragged_dot
        return ragged_dot.gmm(tokens, kernel, sizes, block_m=config.block_m, block_k=config.block_k,
                              block_n=config.block_n, num_warps=config.warps, num_stages=config.stages,
                              trans_rhs=trans_rhs, interpret=interpret, compute_dtype=tokens.dtype,
                              out_dtype=jnp.dtype(out_dtype)), None


@dataclasses.dataclass(frozen=True)
class Tgmm(_Grouped):
    """`ragged_dot.tgmm`: each group's `[k, n]` sum of its rows' outer
    products of `tokens` and `cotangent`, the kernel gradient."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens', 'cotangent')

    def _fwd(self, tokens: jax.Array, cotangent: jax.Array, sizes: jax.Array, *, out_dtype: str,
             interpret: bool, config: TileConfig, return_residuals: bool = False):
        from . import ragged_dot
        return ragged_dot.tgmm(tokens, cotangent, sizes, block_m=config.block_m, block_k=config.block_k,
                               block_n=config.block_n, num_warps=config.warps, num_stages=config.stages,
                               interpret=interpret, compute_dtype=tokens.dtype,
                               out_dtype=jnp.dtype(out_dtype)), None


_GMM = Gmm()
_TGMM = Tgmm()


def _gmm(tokens, kernel, sizes, out_dtype, *, trans_rhs: bool, interpret: bool):
    return _GMM(tokens, kernel.astype(tokens.dtype), sizes, out_dtype=jnp.dtype(out_dtype).name,
                trans_rhs=trans_rhs, interpret=interpret)


def _tgmm(tokens, cotangent, sizes, out_dtype, *, interpret: bool):
    return _TGMM(tokens, cotangent.astype(tokens.dtype), sizes, out_dtype=jnp.dtype(out_dtype).name,
                 interpret=interpret)


def _on_platform(kernel, fallback, interpret_on_cpu: bool):
    """`kernel` where the call lowers for CUDA, interpreted on the CPU when
    asked for, and `fallback` everywhere else."""
    branches = {'cuda': functools.partial(kernel, interpret=False)}
    if interpret_on_cpu:
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


def _forward(inputs, matrix, sizes, out_dtype, interpret_on_cpu):
    def kernel(inputs, matrix, sizes, *, interpret):
        return _gmm(inputs, matrix, sizes, out_dtype, trans_rhs=False, interpret=interpret)

    def fallback(inputs, matrix, sizes):
        return xla_ragged_dot(inputs, matrix, sizes, preferred_element_type=out_dtype)

    branches, default = _on_platform(kernel, fallback, interpret_on_cpu)
    return jax.lax.platform_dependent(inputs, matrix, sizes, default=default, **branches)


def _backward(inputs, matrix, sizes, cotangent, work, interpret_on_cpu):
    def kernel(inputs, matrix, sizes, cotangent, *, interpret):
        return (_gmm(cotangent, matrix, sizes, work, trans_rhs=True, interpret=interpret),
                _tgmm(inputs, cotangent, sizes, work, interpret=interpret))

    def fallback(inputs, matrix, sizes, cotangent):
        d_inputs = xla_ragged_dot(cotangent, jnp.swapaxes(matrix, 1, 2), sizes,
                                  preferred_element_type=work)
        _, pullback = jax.vjp(lambda matrix: xla_ragged_dot(
            inputs, matrix, sizes, preferred_element_type=work), matrix.astype(work))
        return d_inputs, pullback(cotangent.astype(work))[0]

    branches, default = _on_platform(kernel, fallback, interpret_on_cpu)
    return jax.lax.platform_dependent(inputs, matrix, sizes, cotangent, default=default,
                                      **branches)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def grouped_projection(x: jax.Array, kernel: jax.Array, group_sizes: jax.Array,
                       compute: Dtype, interpret_on_cpu: bool) -> jax.Array:
    """Compute `dew.nn.moe.expert_projection` with the kernels, differentiable in first-order reverse mode.

    The operands are cast to `compute`, and the product accumulates in fp32
    and is rounded once to `compute`. The input gradient takes `x`'s dtype
    and the kernel gradient takes the kernel's. Each is summed in fp32 from
    the compute-dtype cotangent and the rounded operands, which with 16-bit
    compute means exact products. Only a CUDA lowering runs the kernels
    (`interpret_on_cpu` adds the CPU's interpreter). Every other lowering
    runs `jax.lax.ragged_dot` under the same contract.
    """
    return _projection_fwd(x, kernel, group_sizes, compute, interpret_on_cpu)[0]


def _projection_fwd(x, kernel, group_sizes, compute, interpret_on_cpu):
    inputs, matrix = x.astype(compute), kernel.astype(compute)
    sizes = group_sizes.astype(jnp.int32)
    output = _forward(inputs, matrix, sizes, jnp.promote_types(compute, jnp.float32),
                      interpret_on_cpu)
    # The residuals are the rounded operands, the values the forward
    # multiplied; the dtypes of the originals are what the gradients take.
    residuals = (inputs, matrix, sizes, jnp.zeros((0,), x.dtype), jnp.zeros((0,), kernel.dtype))
    return output.astype(compute), residuals


def _projection_bwd(compute, interpret_on_cpu, residuals, cotangent):
    del compute
    inputs, matrix, sizes, x_like, kernel_like = residuals
    work = jnp.result_type(inputs.dtype, x_like.dtype, kernel_like.dtype, jnp.float32)
    d_inputs, d_matrix = _backward(inputs, matrix, sizes, cotangent.astype(inputs.dtype), work,
                                   interpret_on_cpu)
    return (d_inputs.astype(x_like.dtype), d_matrix.astype(kernel_like.dtype),
            np.zeros(sizes.shape, jax.dtypes.float0))


grouped_projection.defvjp(_projection_fwd, _projection_bwd)
