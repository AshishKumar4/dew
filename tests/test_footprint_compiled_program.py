"""The derived peak of an expert decoder's gradient is at most what XLA compiles it to (row 2750): the walk
reads the jaxpr, XLA compiles the program JAX lowers it to, and four terms the walk counted are not in it.
Each test states one, with the rule that removes it, on a program small enough to read."""


import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from dew.training.compiled_program import compiled, deviceless, lowering_platform
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


def test_a_pallas_kernel_holds_what_its_lowering_holds():
    """A native Pallas kernel is one custom call to XLA, `__gpu$xla.gpu.triton`, whose results are the
    kernel's outputs alone (`pallas_call_lowering`, jax/_src/pallas/triton/pallas_call_registration.py):
    its products are Triton's, no cuBLAS call's, its body's values in registers and shared memory, so
    the call holds its output. An interpreted one lowers to the HLO interpreter's program
    (`pallas_call_hlo_interpret`, jax/_src/pallas/hlo_interpreter.py): the kernel discharged of its
    state and run over its grid in a loop carrying its whole output, each block's product a cuBLAS call
    whose workspace is live beside it, within what XLA's compile of it on a compile-only Hopper holds."""
    from jax.experimental import pallas as pl

    from dew.training.ahead import compile_only_gpus

    (gpu,) = compile_only_gpus("sm90", 1, 1 << 30)
    workspace = gemm_workspace(gpu)
    output = jax.ShapeDtypeStruct((4096, 4096), jnp.float32)
    specs = (jax.ShapeDtypeStruct((4096, 64), jnp.float32), jax.ShapeDtypeStruct((64, 4096), jnp.float32))

    def kernel(left, right, out):
        out[...] = left[...] @ right[...]

    def call(interpret: bool):
        return pl.pallas_call(kernel, out_shape=output, grid=(8, 8), interpret=interpret,
                              in_specs=[pl.BlockSpec((512, 64), lambda row, column: (row, 0)),
                                        pl.BlockSpec((64, 512), lambda row, column: (0, column))],
                              out_specs=pl.BlockSpec((512, 512), lambda row, column: (row, column)))

    def peak(interpret: bool) -> int:
        program = jax.make_jaxpr(call(interpret))(*specs).jaxpr
        return peak_bytes(compiled(program, "cuda"), "gpu", workspace)

    sharding = NamedSharding(Mesh(np.asarray([gpu]), ("data",)), P())
    placed = [jax.ShapeDtypeStruct(spec.shape, spec.dtype, sharding=sharding) for spec in specs]
    held = jax.jit(call(interpret=True)).lower(*placed).compile(
        compiler_options={"xla_gpu_enable_triton_gemm": False}).memory_analysis()
    assert held is not None, "the compile states no memory"
    assert peak(interpret=False) == 4096 * 4096 * 4
    assert 4096 * 4096 * 4 + workspace.each <= peak(interpret=True) <= (
        held.output_size_in_bytes + held.temp_size_in_bytes)


@pytest.mark.parametrize("order", ["cuda-first", "rocm-first"])
def test_a_platform_dependent_program_holds_the_branch_its_platform_lowers(order):
    """A `platform_dependent` conditional lowers to the branch of the platform it is lowered for, inline,
    no conditional around it (`_cond_lowering`, jax/_src/lax/control_flow/conditionals.py keeps the
    branches whose platforms hold the lowering's and lowers a sole one as the computation itself), so XLA
    never sees the other branches, whichever order they are named in, and fuses the kept one with what
    reads it: CUDA holds its branch's values, a CPU the default's, as the branch alone in the program
    would."""
    values = jax.ShapeDtypeStruct((256, 256), jnp.float32)

    def small(values):
        return jnp.sin(values)

    def large(values):
        return jnp.tanh(jnp.exp(values) @ values) @ values

    def program(function):
        return jax.make_jaxpr(lambda values: jnp.sum(function(values)))(values).jaxpr

    def selected(**branches):
        return lambda values: jax.lax.platform_dependent(values, **branches)

    alone = peak_bytes(program(small), "gpu", Workspace(0, 0))
    named = ({"cuda": small, "rocm": large} if order == "cuda-first" else {"rocm": large, "cuda": small})
    lowered_for = (("cuda", {**named, "default": large}), ("cpu", {"cuda": large, "default": small}))
    for lowering, branches in lowered_for:
        lowered = compiled(program(selected(**branches)), lowering)
        assert "cond" not in {equation.primitive.name for equation in lowered.eqns}
        assert peak_bytes(lowered, "gpu", Workspace(0, 0)) == alone


def compiled_workspaces(gpu, function, *specs) -> list[int]:
    """The workspaces of the cuBLAS matmuls `function`'s gradient compiles to on the compile-only `gpu`,
    with XLA's Triton GEMMs off, as the planner compiles a step, so every product is a cuBLAS call."""
    sharding = NamedSharding(Mesh(np.asarray([gpu]), ("data",)), P())
    placed = [jax.ShapeDtypeStruct(spec.shape, spec.dtype, sharding=sharding) for spec in specs]
    program = jax.jit(jax.grad(lambda *arguments: function(*arguments)[0])).lower(*placed).compile(
        compiler_options={"xla_gpu_enable_triton_gemm": False}).as_text()
    assert program, "the compile states no program"
    return [int(size) for line in program.splitlines() if 'custom_call_target="__cublas' in line
            for size in re.findall(r"s8\[(\d+)\]", line)]


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
    loop's induction variable's: `DynamicSliceFusionRewriterV2`, xla/backends/gpu/transforms/
    dynamic_slice_fusion_rewriter_v2.cc, `AnalyzeDynamicSlice`, dynamic_slice_analysis.cc), its counter's
    or a scan's turn of its stacked inputs, is fused with the slice, and the config assigner, which
    visits no fusion's computation, leaves it the workspace `GemmWorkspaceRewriteVisitor` gave it: 32 MiB
    at Hopper, 4 MiB before, where other matmuls hold the default config's 4 MiB."""
    from dew.training.ahead import compile_only_gpus

    (gpu,) = compile_only_gpus(generation, 1, 1 << 30)
    table, rows = jax.ShapeDtypeStruct((256, 64), jnp.float32), jax.ShapeDtypeStruct((512, 64), jnp.float32)

    def loss(table, rows):
        def tile(index, total):
            # Each turn's tile of the table, its offset the counter's: the slice the matmul is fused with.
            return total + jnp.sum(jnp.tanh(rows @ jax.lax.dynamic_slice_in_dim(table, index * 64, 64).T))

        # Squared, so the loop's cotangent is a value each turn scales its residual by, not the seed.
        return jax.lax.fori_loop(0, 4, tile, jnp.zeros(())) ** 2, ()

    # XLA fuses the forward's matmul with the table's tile, which keeps the rewriter's workspace, and not
    # the backward's, whose operand is the residual's turn scaled, which takes the default config's.
    workspace = gemm_workspace(gpu)
    assert sorted(compiled_workspaces(gpu, loss, table, rows)) == sorted([workspace.each, workspace.sliced])

    def peak(workspace: Workspace) -> int:
        return gradient_peak(loss, table, (rows,), platform="gpu", workspace=workspace).bytes

    # The derivation charges each matmul as XLA does: one of either.
    vast = 1 << 30
    assert vast <= peak(Workspace(vast, 0)) and vast <= peak(Workspace(0, vast))
    assert peak(Workspace(vast, vast)) < 2 * vast


def test_a_compile_only_target_is_deviceless_and_lowers_for_cuda():
    """A compile-only GPU is no device of a running backend (`deviceless`) and JAX lowers for it as CUDA;
    a CPU of this process is a running backend's, lowered for as a CPU."""
    from dew.training.ahead import compile_only_gpus

    (gpu,) = compile_only_gpus("sm90", 1, 1 << 30)
    cpu = jax.devices("cpu")[0]
    assert deviceless(gpu) and not deviceless(cpu)
    assert (lowering_platform(gpu), lowering_platform(cpu)) == ("cuda", "cpu")


def scaled(product):
    """A loss of `product`'s matmuls scaled by a scalar, its gradient's first argument: the gradient's
    program holds the forward's matmuls alone, as the scale's cotangent reads them."""
    def loss(scale, table, rows):
        return scale * jnp.sum(jnp.tanh(product(table, rows))), ()

    return loss


def over_turns(tile):
    """`tile(index, table, rows)`'s products summed over four turns of a `fori_loop`."""
    def product(table, rows):
        return jax.lax.fori_loop(0, 4, lambda index, total: total + jnp.sum(tile(index, table, rows)),
                                 jnp.zeros(()))

    return product


def returned_update(table, rows):
    """Each turn's product, of values computed (no slice), updated into the loop's carried buffer, which
    the loop returns."""
    def turn(index, buffer):
        product = jnp.tanh(rows[:64]) @ jnp.tanh(table[:64]).T
        return jax.lax.dynamic_update_slice_in_dim(buffer, product, index * 64, axis=0)

    return jax.lax.fori_loop(0, 4, turn, jnp.zeros((256, 64)))


def discarded_update(table, rows):
    """Each turn's product, of values computed (no slice), updated into a scratch buffer reduced away,
    which the loop never returns."""
    def turn(index, total):
        product = jnp.tanh(rows[:64]) @ jnp.tanh(table[:64]).T
        scratch = jax.lax.dynamic_update_slice_in_dim(jnp.zeros((256, 64)), product, index * 64, axis=0)
        return total + jnp.sum(scratch)

    return jax.lax.fori_loop(0, 4, turn, jnp.zeros(()))


def shared_product(table, rows):
    """Each turn's product updated into the returned buffer and also reduced: two readers of it."""
    def turn(index, carried):
        buffer, total = carried
        product = jnp.tanh(rows[:64]) @ jnp.tanh(table[:64]).T
        return (jax.lax.dynamic_update_slice_in_dim(buffer, product, index * 64, axis=0),
                total + jnp.sum(product))

    buffer, total = jax.lax.fori_loop(0, 4, turn, (jnp.zeros((256, 64)), jnp.zeros(())))
    return total + jnp.sum(buffer)


def stacked_output(table, rows):
    """Each turn's product a scan's stacked output, read by nothing else."""
    def turn(total, tile):
        return total, jnp.tanh(rows[:64]) @ jnp.tanh(tile).T

    total, products = jax.lax.scan(turn, jnp.zeros(()), table.reshape(4, 64, 64))
    return total + jnp.sum(products)


def shared_output(table, rows):
    """Each turn's product a scan's stacked output and also reduced into its carry."""
    def turn(total, tile):
        product = jnp.tanh(rows[:64]) @ jnp.tanh(tile).T
        return total + jnp.sum(product), product

    total, products = jax.lax.scan(turn, jnp.zeros(()), table.reshape(4, 64, 64))
    return total + jnp.sum(products)


PROGRAMS = {
    # A static aligned contiguous slice, outside any loop.
    "static-slice": lambda table, rows: rows @ table[:64].T,
    # A tile its counter offsets evenly, the rows it cuts whole.
    "counted-rows": over_turns(lambda index, table, rows: rows @ jax.lax.dynamic_slice_in_dim(
        table, index * 64, 64).T),
    # A tile cut in two axes: no contiguous run of the table's bytes.
    "two-axis-cut": over_turns(lambda index, table, rows: rows[:, :32] @ jax.lax.dynamic_slice(
        table, (index * 32, 0), (64, 32)).T),
    # Offsets the counter squares: no linear progression of bytes.
    "squared-offsets": over_turns(lambda index, table, rows: rows @ jax.lax.dynamic_slice_in_dim(
        table, index * index * 16, 64).T),
    # A tile transposed and reshaped across its axes before the product: no bitcast of the slice.
    "mixed-reshape": over_turns(lambda index, table, rows: rows[:, :32] @ jax.lax.dynamic_slice_in_dim(
        table, index * 64, 64).T.reshape(32, 128)),
    "returned-update": returned_update,
    "discarded-update": discarded_update,
    "shared-product": shared_product,
    "stacked-output": stacked_output,
    "shared-output": shared_output,
}


@pytest.mark.parametrize("program", sorted(PROGRAMS))
def test_a_deviceless_matmul_holds_the_workspace_its_slice_fusion_decides(program):
    """On a compile-only Hopper, a cuBLAS matmul XLA fuses with a slice of its operand or of its
    result's update (`DynamicSliceFusionRewriterV2`) keeps the rewriter's 32 MiB, and any other the
    default config's 4 MiB: the derivation charges each matmul of a program the workspace XLA's own
    compile gives it, slices fused and slices not alike."""
    from dew.training.ahead import compile_only_gpus

    (gpu,) = compile_only_gpus("sm90", 1, 1 << 30)
    workspace = gemm_workspace(gpu)
    loss = scaled(PROGRAMS[program])
    specs = (jax.ShapeDtypeStruct((), jnp.float32), jax.ShapeDtypeStruct((256, 64), jnp.float32),
             jax.ShapeDtypeStruct((512, 64), jnp.float32))
    held = {"each" if size == workspace.each else "sliced" for size in compiled_workspaces(gpu, loss, *specs)}
    vast = 1 << 30

    def peak(charged: Workspace) -> int:
        # Under the compile-only mesh, as the planner derives a step for it.
        with jax.set_mesh(Mesh(np.asarray([gpu]), ("data",))):
            return gradient_peak(loss, specs[0], specs[1:], platform="gpu", workspace=charged).bytes

    charged = {kind for kind, charge in (("each", Workspace(vast, 0)), ("sliced", Workspace(0, vast)))
               if peak(charge) >= vast}
    assert charged == held


def test_deriving_a_loops_offsets_runs_no_callback_of_it():
    """XLA's evaluator computes an offset from the loop's counter only through operations of no side
    effect and no custom call (`GetLeafDependencies`, dynamic_slice_analysis.cc), and computes no
    operation the offset does not read: deriving a peak runs none of a loop's callbacks, neither one
    beside its offsets nor one an offset reads, and a slice whose offset a callback computes is read
    as fused with nothing."""
    called = []

    def offset(index):
        called.append(index)
        return index * 64

    def tile(index, table, rows):
        jax.debug.callback(called.append, index)
        start = jax.pure_callback(offset, jax.ShapeDtypeStruct((), jnp.int32), index)
        return rows @ jax.lax.dynamic_slice_in_dim(table, start, 64).T

    loss = scaled(over_turns(tile))
    specs = (jax.ShapeDtypeStruct((), jnp.float32), jax.ShapeDtypeStruct((256, 64), jnp.float32),
             jax.ShapeDtypeStruct((512, 64), jnp.float32))
    vast = 1 << 30
    sliced = gradient_peak(loss, specs[0], specs[1:], platform="gpu", workspace=Workspace(0, vast)).bytes
    assert called == []
    assert sliced < vast



def updated_buffer(loop: str):
    """A loop updating a 16 MiB buffer it starts from zeros, turn by turn, which it returns."""
    def program(rows):
        def turn(index, buffer):
            return jax.lax.dynamic_update_slice_in_dim(buffer, jnp.tanh(rows) + index, index * 64, axis=0)

        start = jnp.zeros((4096, 1024))
        if loop == "scan":
            return jax.lax.fori_loop(0, 64, turn, start)
        return jax.lax.while_loop(lambda carried: carried[0] < 64,
                                  lambda carried: (carried[0] + 1, turn(*carried)), (0, start))[1]

    return program


def discarded_carry(rows):
    """A scan carrying its input, which it updates each turn, its final carry discarded."""
    def turn(carried, _):
        updated = jnp.tanh(carried)
        return updated, jnp.sum(updated)

    return jax.lax.scan(turn, rows, None, length=4)[1]


def one_start_twice(rows):
    """A scan carrying one product and a view of it as two values, each updated apart, its final carries
    discarded."""
    product = rows.T @ rows

    def turn(carried, _):
        square, flat = carried
        return (jnp.tanh(square), jnp.exp(flat)), jnp.sum(square) + jnp.sum(flat)

    return jax.lax.scan(turn, (product, product.reshape(-1)), None, length=4)[1]


CARRIES = {
    # The 16 MiB buffer, held once from its zeros to its result.
    "scan": (updated_buffer("scan"), 4096 * 1024 * 4),
    "while": (updated_buffer("while"), 4096 * 1024 * 4),
    # The carry's own buffer, a copy of the input it starts from, though its result is discarded.
    "discarded": (discarded_carry, 64 * 1024 * 4),
    # Two 4 MiB buffers, one carry's each, though both start from one.
    "one-start-twice": (one_start_twice, 2 * 1024 * 1024 * 4),
}


@pytest.mark.parametrize("carried", sorted(CARRIES))
def test_a_loops_carry_is_one_buffer_from_its_start_to_its_result(carried):
    """XLA keeps a loop's carried value in one buffer from the buffer its initial value is computed into
    to the loop's result, its body's parameter and returned value of it among them
    (`ComputeAliasedValues`, xla/hlo/analysis/hlo_alias_analysis.cc), and copies the initial value into
    a buffer of the loop's where that is an input, or another carry's start (copy insertion,
    xla/service/copy_insertion.cc): each loop holds its carries' buffers, the least it can, and no more
    than XLA's compile of it on a compile-only Hopper holds."""
    from dew.training.ahead import compile_only_gpus

    (gpu,) = compile_only_gpus("sm90", 1, 1 << 30)
    program, least = CARRIES[carried]
    rows = jax.ShapeDtypeStruct((64, 1024), jnp.float32)
    derived = peak_bytes(compiled(jax.make_jaxpr(program)(rows).jaxpr, "cuda"), "gpu", Workspace(0, 0))
    sharding = NamedSharding(Mesh(np.asarray([gpu]), ("data",)), P())
    placed = jax.ShapeDtypeStruct(rows.shape, rows.dtype, sharding=sharding)
    held = jax.jit(program).lower(placed).compile().memory_analysis()
    assert held is not None, "the compile states no memory"
    assert least <= derived <= held.output_size_in_bytes + held.temp_size_in_bytes
