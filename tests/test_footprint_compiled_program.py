"""The derived peak of an expert decoder's gradient is at most what XLA compiles it to (row 2750): the walk
reads the jaxpr, XLA compiles the program JAX lowers it to, and four terms the walk counted are not in it.
Each test states one, with the rule that removes it, on a program small enough to read."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from dew.training.footprint import Workspace, gemm_workspace, gradient_peak, peak_bytes


def test_a_shard_map_over_one_device_holds_what_it_would_inline():
    """A `shard_map` whose manual axes are all of size one is its body inline in the program XLA compiles
    (`_shard_map_lowering_shardy`, jax/_src/shard_map.py: no manual computation is made), so a gradient
    through one holds what it would hold inlined. Each layer's silu leaves its pullback three elementwise
    values (the activation, the logistic and its derivative) that XLA's fusion recomputes in each consumer
    (`PriorityFusionQueue::CanFuseWithAllNonBitcastUsers`, xla/backends/gpu/transforms/priority_fusion.cc); as
    outputs of a call they were buffers (the expert decoders' six f32[1024,128] values, 3 MiB, at the gate's
    peak)."""
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",))
    rows = jax.ShapeDtypeStruct((64, 128), jnp.float32)
    params = {"layers": jax.ShapeDtypeStruct((3, 128, 128), jnp.float32),
              "head": jax.ShapeDtypeStruct((128, 128), jnp.float32)}

    def layer(weights, hidden):
        return jax.nn.silu(hidden @ weights)

    def through(call):
        def loss(params, rows):
            hidden = rows
            for index in range(3):
                hidden = call(params["layers"][index], hidden)
            return jnp.sum((hidden @ params["head"]) ** 2), ()
        return loss

    mapped = through(jax.shard_map(layer, mesh=mesh, in_specs=(P(), P()), out_specs=P()))
    for size in (0, 1 << 20):
        workspace = Workspace(size, size)
        assert gradient_peak(mapped, params, (rows,), platform="gpu", workspace=workspace) == gradient_peak(
            through(layer), params, (rows,), platform="gpu", workspace=workspace)


def test_a_pallas_kernel_holds_its_outputs_and_no_workspace():
    """A Pallas kernel is one custom call to XLA, `__gpu$xla.gpu.triton`, whose results are the kernel's
    outputs alone (`pallas_call_lowering`, jax/_src/pallas/triton/pallas_call_registration.py: the body is
    serialized into the call's configuration). The matrix product in its body is Triton's, no HLO dot for
    `GemmRewriter` to hand cuBLAS, and a workspace is added to cuBLAS custom calls alone
    (`GemmWorkspaceRewriteVisitor::HandleCustomCall`, xla/backends/gpu/transforms/gemm_rewriter.cc); the
    values of its body live in registers and shared memory: the call holds its output."""
    from jax.experimental import pallas as pl

    shape = jax.ShapeDtypeStruct((64, 64), jnp.float32)

    def kernel(left, right, out):
        out[...] = left[...] @ right[...]

    def call(left, right):
        return pl.pallas_call(kernel, out_shape=shape, interpret=True)(left, right)

    program = jax.make_jaxpr(call)(shape, shape).jaxpr
    assert [equation.primitive.name for equation in program.eqns] == ["pallas_call"]
    assert peak_bytes(program, "gpu", Workspace(1 << 20, 1 << 20)) == 64 * 64 * 4


def test_a_platform_dependent_program_holds_the_branch_its_platform_lowers():
    """A `platform_dependent` conditional lowers to the branch of the platform it is lowered for
    (`_cond_lowering`, jax/_src/lax/control_flow/conditionals.py keeps `branches_kept`), so XLA never sees the
    other branches: a GPU holds the CUDA branch's values, a CPU the default's, whichever is larger. The
    platform index, an int32, is the program's own beside the branch."""
    values = jax.ShapeDtypeStruct((256, 256), jnp.float32)

    def small(values):
        return values * 2

    def large(values):
        return jnp.tanh(jnp.exp(values) @ values) @ values

    def peak(platform, **branches):
        program = jax.make_jaxpr(lambda values: jax.lax.platform_dependent(values, **branches))(values).jaxpr
        return peak_bytes(program, platform, Workspace(0, 0))

    alone = peak_bytes(jax.make_jaxpr(small)(values).jaxpr, "gpu", Workspace(0, 0))
    assert peak("gpu", cuda=small, default=large) - alone <= 4
    assert peak("cpu", cuda=large, default=small) - alone <= 4


def compiled_workspaces(gpu, function, *specs) -> list[int]:
    """The workspaces of the cuBLAS matmuls `function`'s gradient compiles to on the compile-only `gpu`,
    with XLA's Triton GEMMs off, as the planner compiles a step, so every product is a cuBLAS call."""
    sharding = NamedSharding(Mesh(np.asarray([gpu]), ("data",)), P())
    placed = [jax.ShapeDtypeStruct(spec.shape, spec.dtype, sharding=sharding) for spec in specs]
    compiled = jax.jit(jax.grad(lambda *arguments: function(*arguments)[0])).lower(*placed).compile(
        compiler_options={"xla_gpu_enable_triton_gemm": False})
    return [int(size) for size in re.findall(r"s8\[(\d+)\]", compiled.as_text())]


@pytest.mark.parametrize("generation", ["sm80", "sm90"])
def test_a_compile_only_gpus_gemm_holds_the_workspace_its_deviceless_compile_gives_it(generation):
    """With no device to autotune on, XLA gives each cuBLASLt matmul the default config's 4 MiB workspace
    (`ConfigAssigner` takes no autotuning without a device, `GetConfigAssignerOptions`, xla/service/gpu/
    autotuning/config_assigner_pass.cc, and `CublasLtBackend::GetDefaultConfig` and `ApplyConfig`,
    xla/backends/gpu/autotuner/cublaslt.cc, resize the workspace), replacing the 32 MiB at Hopper that
    `GemmWorkspaceRewriteVisitor` (gemm_rewriter.cc) handed it: a compile-only GPU's matmul holds 4 MiB on
    either generation."""
    from dew.training.ahead import compile_only_gpus

    (gpu,) = compile_only_gpus(generation, 1, 1 << 30)
    weights, rows = jax.ShapeDtypeStruct((64, 64), jnp.float32), jax.ShapeDtypeStruct((512, 64), jnp.float32)

    def loss(weights, rows):
        return jnp.sum((rows @ weights) ** 2), ()

    (compiled,) = set(compiled_workspaces(gpu, loss, weights, rows))
    derived = (gradient_peak(loss, weights, (rows,), platform="gpu", workspace=gemm_workspace(gpu)).bytes
               - gradient_peak(loss, weights, (rows,), platform="gpu", workspace=Workspace(0, 0)).bytes)
    assert derived == compiled


@pytest.mark.parametrize("generation", ["sm80", "sm90"])
def test_a_matmul_fused_with_its_loops_slice_keeps_the_workspace_its_rewrite_gave_it(generation):
    """A matmul whose operand a loop slices per turn (a contiguous, aligned dynamic slice, its offset the
    loop's counter's: `DynamicSliceFusionRewriterV2`, xla/backends/gpu/transforms/
    dynamic_slice_fusion_rewriter_v2.cc, `AnalyzeDynamicSlice`, dynamic_slice_analysis.cc) is fused with
    the slice, and the config assigner, which visits no fusion's computation, leaves it the workspace
    `GemmWorkspaceRewriteVisitor` gave it: 32 MiB at Hopper, 4 MiB before, where the loop's other
    matmuls hold the default config's 4 MiB. The derivation charges each its own."""
    from dew.training.ahead import compile_only_gpus

    (gpu,) = compile_only_gpus(generation, 1, 1 << 30)
    table, rows = jax.ShapeDtypeStruct((256, 64), jnp.float32), jax.ShapeDtypeStruct((512, 64), jnp.float32)

    def loss(table, rows):
        def tile(index, total):
            # Each turn's tile of the table, its offset the counter's: the slice the matmul is fused with.
            return total + jnp.sum(jnp.tanh(rows @ jax.lax.dynamic_slice_in_dim(table, index * 64, 64).T))

        return jax.lax.fori_loop(0, 4, tile, jnp.zeros(())), ()

    compiled = compiled_workspaces(gpu, loss, table, rows)
    workspace = gemm_workspace(gpu)
    assert sorted(set(compiled)) == sorted({workspace.each, workspace.sliced})
    derived = (gradient_peak(loss, table, (rows,), platform="gpu", workspace=workspace).bytes
               - gradient_peak(loss, table, (rows,), platform="gpu", workspace=Workspace(0, 0)).bytes)
    assert derived <= max(compiled)
