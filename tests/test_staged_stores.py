"""A consumer's register values staged into shared memory for its products to read
(`dew.nn.kernels.templates.mosaic.Family.staged`, `staged_together`): each value in its own buffer, an
mma.sync issuer's split-column stores bracketed by Mosaic GPU's own barriers in the order a reused buffer
needs, and a staged block fenced for asynchronous product readers alone (rows 5260, 13053)."""

import contextlib
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import COMPILES_ON_CPU, INTERPRETS
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from reference_error import equations

from dew.nn.kernels.templates import bundles, mosaic_sm80

pytestmark = pytest.mark.kernel

INTERPRETED = plgpu.InterpretGPUParams()
"""Mosaic GPU's own interpreter, which models warpgroups, barriers and shared memory."""


def _staged_three(x, *, interpret: bool, batched: bool):
    """`x` (16 x 64), twice and thrice it, staged in bf16 by an mma.sync issuer whose warps split a tile's
    columns (its stores bracketed by its own barriers), published together (`Family.staged_together`) or a
    store apiece (`Family.staged`), and each read back."""
    family = bundles.named("mma.sync", jnp.bfloat16, None).specialized(16, 64)

    def body(x_gmem, *outs):
        def scoped(*buffers):
            value = x_gmem[...] if interpret else family.zeros((16, 64))
            pairs = [(value * times, buffer) for times, buffer in zip((1, 2, 3), buffers, strict=True)]
            with family.staged_together() if batched else contextlib.nullcontext():
                for each, buffer in pairs:
                    family.staged(each, buffer)
            if interpret:
                for out, buffer in zip(outs, buffers, strict=True):
                    out[...] = family.tile_of(buffer).astype(jnp.float32)

        pl.run_scoped(scoped, *(plgpu.SMEM((16, 64), jnp.bfloat16) for _ in range(3)))

    out = jax.ShapeDtypeStruct((16, 64), jnp.float32)
    return family.kernel(body, out_type=(out,) * 3, grid=(1,), grid_names=("programs",), consumers=1,
                         interpret=interpret, name="staged")(x)


@INTERPRETS
def test_a_staged_batch_holds_each_value_in_its_buffer():
    """Interpreted: three values staged together (`Family.staged_together`) read back exactly from their own
    buffers."""
    x = jax.random.randint(jax.random.key(7), (16, 64), -4, 4).astype(jnp.float32)
    got = _staged_three(x, interpret=INTERPRETED, batched=True)
    for times, value in zip((1, 2, 3), got, strict=True):
        np.testing.assert_array_equal(np.asarray(value), times * np.asarray(x))


def _accesses(call) -> str:
    """The typed module `call` lowers to for an A100, in program order, as its warpgroup barriers ('B',
    `nvvm.barrier`), shared-memory stores ('W') and shared-memory reads ('R', `ldmatrix` or a load)."""
    import jax.experimental.mosaic.gpu as mgpu
    from jax._src.lib.mlir import ir
    from jax._src.lib.mlir.dialects import llvm
    from jax.experimental.mosaic.gpu import utils as mgpu_utils
    from kernel_compile import lowered_for

    from dew.nn.kernels.lowered import kernel_calls

    lowered = lowered_for(call, (jax.ShapeDtypeStruct((16, 64), jnp.float32),), "sm80")
    (kernel,) = [each for each in kernel_calls(lowered) if each.kind == "mosaic_gpu"]
    module = mgpu.deserialized(kernel.ir)
    found: list[str] = []

    def in_shared(value) -> bool:
        # A shared-memory ref, or an LLVM pointer into shared memory (`ldmatrix`'s).
        kind = value.type
        if isinstance(kind, ir.MemRefType):
            return mgpu_utils.is_smem_ref(kind)
        shared = mgpu_utils.WORKGROUP_NVPTX_ADDRESS_SPACE
        return isinstance(kind, llvm.PointerType) and kind.address_space == shared

    def visit(operation) -> ir.WalkResult:
        name = operation.name
        shared = any(in_shared(each) for each in operation.operands)
        if name == "nvvm.barrier":
            found.append("B")
        elif shared and name.endswith("store"):
            found.append("W")
        elif shared and (name.endswith("load") or name == "nvvm.ldmatrix"):
            found.append("R")
        return ir.WalkResult.ADVANCE

    with module.context:
        module.operation.walk(visit, ir.WalkOrder.PRE_ORDER)
    return "".join(found)


def _barriers(call) -> int:
    """The warpgroup barriers (`nvvm.barrier`) in the typed module `call` lowers to for an A100."""
    return _accesses(call).count("B")


@COMPILES_ON_CPU
@pytest.mark.parametrize("batched", [False, True], ids=["apiece", "together"])
def test_three_mma_sync_staged_stores_take_a_barrier_at_each_end_and_one_between_each_two(batched):
    """Lowered for an A100: three tiles staged by an mma.sync issuer whose warps split a tile's columns, each
    store bracketed by its own barriers, published apiece (`staged`) or together (`staged_together`), take
    four warpgroup barriers, one before the first store, one after the last and one between each two: the
    fork drops a collective barrier that repeats one before it (`core._drop_repeated_barriers`), where each
    store's own pair and the staging's beside it took 12 apiece and a batch's own pair 8 (a Kimi intra
    step's 20 a trip, FS-INTRA-BAR1; row 13053)."""
    assert _barriers(functools.partial(_staged_three, interpret=False, batched=batched)) == 4


def _staged_twice(x):
    """An mma.sync issuer whose warps split a tile's columns stages a tile into one shared buffer, reads it,
    stages another into the same buffer and reads that."""
    family = bundles.named("mma.sync", jnp.bfloat16, None).specialized(16, 64)

    def body(x_gmem, first, second):
        def scoped(buffer):
            # Both reads held in registers and written out past the second: no global store, and its own
            # automatic barriers, between the buffer's read and the store over it.
            value = family.zeros((16, 64))
            family.staged(value + 1, buffer)
            read = family.tile_of(buffer)
            family.staged(value + 2, buffer)
            reread = family.tile_of(buffer)
            first[...] = read.astype(jnp.float32)
            second[...] = reread.astype(jnp.float32)

        pl.run_scoped(scoped, plgpu.SMEM((16, 64), jnp.bfloat16))

    out = jax.ShapeDtypeStruct((16, 64), jnp.float32)
    return family.kernel(body, out_type=(out, out), grid=(1,), grid_names=("programs",), consumers=1,
                         interpret=False, name="staged_twice")(x)


@COMPILES_ON_CPU
def test_an_mma_sync_staged_buffer_read_and_overwritten_meets_a_barrier_between_each_two():
    """Lowered for an A100: a buffer an mma.sync issuer's split-column warps stage into, read, stage into
    again and read, holds a warpgroup barrier between every store and the read past it (RAW) and between
    every read and the store over it (WAR), its stores' brackets Mosaic GPU's own (row 13053)."""
    accesses = _accesses(_staged_twice)
    assert accesses.count("W") >= 2 and accesses.count("R") == 2, accesses
    # Between two barriers, stores alone or reads alone: no store and read of the buffer unordered.
    assert not [run for run in accesses.split("B") if "W" in run and "R" in run], accesses



@pytest.mark.parametrize("family", [bundles.MmaFed(jnp.bfloat16), bundles.MmaProduced(jnp.bfloat16),
                                    mosaic_sm80.Mma(jnp.bfloat16), "wgmma"],
                         ids=["mma.sync fed", "mma.sync produced", "mma.sync on cp.async", "wgmma"])
def test_a_staged_batch_is_fenced_for_asynchronous_product_readers_alone(family):
    """A consumer's staged tiles published together (`Family.staged_together`) take the async proxy's fence
    (`plgpu.commit_smem`) where its products read shared memory asynchronously (wgmma's), and none where
    they read it through the warpgroup's own ldmatrix, mma.sync's on any movement: TMA rings its consumer
    or a producer fills as on cp.async (a Kimi intra walk's three fences a step on a GH200, M-JS-272)."""
    from jax._src.pallas.mosaic_gpu import primitives as gpu

    from dew.nn.kernels.templates import mosaic_sm90

    family = mosaic_sm90.WGMMA if family == "wgmma" else family

    def body(out_ref, source, staging):
        value = plgpu.load(source, layout=family.layout, optimized=False)
        with family.staged_together():
            family.staged(value, staging)

    tile = plgpu.SMEM((64, 64), jnp.bfloat16)
    launch = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((1,), jnp.float32), num_threads=1,
                          thread_name="wg", scratch_types=[tile, tile])
    fences = [equation for equation in equations(jax.make_jaxpr(launch)().jaxpr)
              if equation.primitive is gpu.commit_smem_p]
    assert bool(fences) == (family is mosaic_sm90.WGMMA), fences
