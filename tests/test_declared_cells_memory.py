"""A declared cell's route holds no more of a training step's memory than its receipt states (row 5972's P0:
the B200's and GH200's kernel-gradient cells sent the 4B's dW to XLA's dot, which XLA deferred to the
backward's end, every layer's cotangents held, a four-row step's temporaries 22.25 GB against 11.06 GB on
the template). Each generation's dense products in a stack of layers, forward and reverse, on a compile-only
device: their temporaries under the declared cells no more than the template's named in their place."""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from kernel_compile import lowered_for

from dew.distributed.placement import Layout
from dew.mesh import DEFAULT_RULES, MESH_AXES
from dew.nn import activations
from dew.nn.dense import Dense
from dew.nn.kernels import gemm, generation
from dew.nn.sharding import placing

LAYERS, ROWS, WIDTH = 4, 4096, 2048
FAMILIES = {family: name for name in reversed(generation.TARGETS)
            if (family := generation.family(name)) is not None and family != "tpu"}
"""Each instruction family a GPU target claims (`generation.TARGETS`, `generation.family`), at its first
generation in the table: a family added there is a family the contract's tests run on."""


def _step_temp(generation_name: str, chosen: dict, memory: int = 2**25) -> int:
    """The temporaries a `LAYERS`-deep stack of dense products takes, its forward and its reverse, on a
    compile-only device of `generation_name` stating `memory`: by default 32 MiB, below the step's need, so
    XLA's scheduler holds the least it can and the temporaries are the route's alone (an H100's 80 GB let it
    hold the declared cells' 117 MB against the template's 67 MB; a compile-only GPU stated 0 bytes before
    row 8768), its products routed as `chosen` names (`generation.chosen`)."""
    def loss(x, weights):
        for w in weights:
            x = jnp.tanh(gemm.gemm(x, w, None, out_dtype=jnp.bfloat16))
        return jnp.sum(x.astype(jnp.float32))

    shapes = (jax.ShapeDtypeStruct((ROWS, WIDTH), jnp.bfloat16),
              tuple(jax.ShapeDtypeStruct((WIDTH, WIDTH), jnp.bfloat16) for _ in range(LAYERS)))
    with generation.chosen(**chosen):
        return lowered_for(jax.grad(loss, argnums=(0, 1)), shapes, generation_name, memory=memory).compile(
            ).memory_analysis().temp_size_in_bytes


@pytest.mark.parametrize("generation_name", sorted(FAMILIES.values()))
def test_a_generations_declared_cells_hold_no_more_of_a_steps_memory_than_their_receipts_state(
        generation_name):
    """A stack of `LAYERS` dense bf16 products, `ROWS` x `WIDTH` by `WIDTH` x `WIDTH`, forward and reverse:
    its temporaries under its generation's declared cells (`grouped_matmul.DECLARED`) no larger than the
    template's named in their place (`generation.chosen(grouped_matmul='mosaic')`), past one output of slack:
    no layer's output held past its layer by a route the cells take."""
    output = ROWS * WIDTH * 2
    cells = _step_temp(generation_name, {})
    template = _step_temp(generation_name, {"grouped_matmul": "mosaic"})
    assert cells <= template + output, (cells, template)


@pytest.mark.parametrize("generation_name", sorted(FAMILIES.values()))
def test_the_templates_launches_hold_no_more_of_a_step_than_xlas_dot(generation_name):
    """The template's launches hold no more of a step's memory than XLA's dot in their place
    (kernel-templates.md §1: on par in peak memory): a split's fp32 partials and a balanced tail's within
    the workspace XLA hands cuBLAS (`GpuTarget.reference_workspace`), where an A100's dense kernel gradients
    split two ways held 80 MiB of a four-layer step's temporaries against XLA's 64 (row 12850). On a device
    stating more than the step needs (16 GiB), where XLA rematerializes nothing of either; below it, XLA
    recomputes its own dots and no launch's."""
    template = _step_temp(generation_name, {"grouped_matmul": "mosaic"}, memory=2**34)
    xla = _step_temp(generation_name, {"grouped_matmul": "xla"}, memory=2**34)
    assert template <= xla, (template, xla)


GATED_ROWS, GATED_WIDTH, HIDDEN = 4096, 1024, 2048

GATED = {"one device": (1, False), "two devices under a layout": (2, True), "two devices": (2, False)}
"""The gated stack's configurations: its devices (a data axis splitting its rows), and whether its program
traces under a layout (`placing`, as a trainer's step does), which maps its projections onto each device's
rows where a launch takes them (`dew.nn.kernels.projections.mapped_shards`)."""


class _GatedLayer(nn.Module):
    """A gated MLP's layer as a model's is (Dew's `Dense` projections, declared axes, the generation's SwiGLU
    product, `activations.gated_product`) with its residual added: the 4B's MLP at a smaller width."""

    @nn.compact
    def __call__(self, x):
        product = activations.gated_product("swiglu")
        dense = functools.partial(Dense, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, use_bias=False)
        hidden = product(dense(HIDDEN, kernel_axes=("embed", "mlp"), name="gate_proj")(x),
                         dense(HIDDEN, kernel_axes=("embed", "mlp"), name="up_proj")(x))
        return x + dense(GATED_WIDTH, kernel_axes=("mlp", "embed"), name="down_proj")(hidden)


def _gated_temp(generation_name: str, layers: int, configuration: str) -> int:
    """The temporaries a device's share of a `layers`-deep gated stack takes, forward and reverse, under the
    generation's declared cells, on compile-only devices of `generation_name` (`GATED`'s configuration),
    each layer rematerialized (`jax.checkpoint`, as a model's `remat` keeps its input alone)."""
    from jax.sharding import PartitionSpec
    from kernel_compile import _device

    devices, layout = GATED[configuration]
    module = _GatedLayer()
    rows = (GATED_ROWS // 8, 8, GATED_WIDTH)  # sequences of eight tokens
    params = jax.eval_shape(lambda: module.init(jax.random.key(0), jnp.zeros(rows, jnp.bfloat16)))

    def loss(x, stack):
        for variables in stack:
            x = jax.checkpoint(module.apply)(variables, x)
        return jnp.sum(x.astype(jnp.float32))

    # Dew's role axes, the rules' (`DEFAULT_RULES`), each of one device but the data axis splitting the rows.
    mesh = _device(generation_name, tuple((name, devices if name == "data" else 1) for name in MESH_AXES))

    placed = functools.partial(_placed, mesh)
    stack = tuple(jax.tree.map(placed, params) for _ in range(layers))
    x = placed(jax.ShapeDtypeStruct(rows, jnp.bfloat16), PartitionSpec("data"))
    with (generation.assumed(generation_name), jax.set_mesh(mesh), nn.logical_axis_rules(DEFAULT_RULES),
          placing(Layout() if layout else None)):
        return jax.jit(jax.grad(loss, argnums=(0, 1))).lower(x, stack).compile(
            ).memory_analysis().temp_size_in_bytes


@pytest.mark.parametrize("configuration", list(GATED))
@pytest.mark.parametrize("generation_name", sorted(FAMILIES.values()))
def test_a_gated_stacks_backward_holds_no_layers_outputs_past_its_layer(generation_name, configuration):
    """A rematerialized gated stack (each layer's gate and up products, their SwiGLU product and its down
    product, the 4B's MLP), forward and reverse under its generation's declared cells, on one device and on
    two splitting its rows, under a layout and under none: a device's temporaries grow by no more than its
    rows of each layer's input (the residual its rematerialization keeps) and the layer's weight gradients
    (the step's own outputs, which a data axis sums: XLA's all-reduce combiner stages every layer's in one
    buffer up to its threshold, `wrapped_concatenate` and its all-reduce, 2 x 25 MB at two layers on two
    B200s) a layer, from two layers to four, past one hidden output of slack. At four times the rows XLA put
    no kernel gradient off on base: its scheduler weighs the cotangents' size, so the rows are a model's
    shard's few thousand, where the hold the test reads was three cotangents a layer (29 MB a device).
    Row 6027: the template's input gradient beside XLA's kernel-gradient dot made XLA put that dot off to
    the step's end, every layer's fused SwiGLU backward pair (`[rows, hidden]` twice) held to it, a
    four-row 4B step's temporaries 80.50 GB on a compile-only GH200 against the
    parent's 20.42; on two devices with every product XLA's, three cotangents a layer: each product's
    reverse mode leaves its rows' cotangent only with its kernel gradient formed (`gemm.in_layer`)."""
    devices = GATED[configuration][0]
    residual, hidden = GATED_ROWS // devices * GATED_WIDTH * 2, GATED_ROWS // devices * HIDDEN * 2
    weights = 3 * GATED_WIDTH * HIDDEN * 2
    grown = _gated_temp(generation_name, 4, configuration) - _gated_temp(generation_name, 2, configuration)
    assert grown <= 2 * (residual + weights) + hidden, (grown, residual, weights, hidden)


def _placed(mesh, leaf, spec=None):
    """`leaf`'s shape and dtype placed on `mesh` by `spec`, whole where None."""
    from jax.sharding import NamedSharding, PartitionSpec

    spec = PartitionSpec() if spec is None else spec
    return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=NamedSharding(mesh, spec))


def _cublas_dtypes(compiled) -> list[str]:
    """The output element type of each cuBLAS product of `compiled`, read from its module's schema
    (`hlo_instructions`)."""
    from dew.telemetry.instrumentation import hlo_instructions

    return [instruction.arrays[0] for instruction in hlo_instructions(compiled).values()
            if "cublas" in instruction.target]


def _dot_dtypes(lowered) -> list[str]:
    """The result element type of each dot of `lowered`'s program, read from its operations."""
    from jax.extend.mlir import ir

    from dew.nn.kernels.lowered import read_operations

    return read_operations(lowered, lambda operation: str(ir.RankedTensorType(
        operation.results[0].type).element_type) if operation.name == "stablehlo.dot_general" else None)


def test_a_primal_dense_bf16_xla_fallback_stores_bf16_as_xlas_own_dot_does():
    """A dense bf16 product the template leaves to XLA, lowered for an H100 with no derivative taken
    (`gemm_xla.direct_dtype`, row 12898): its cuBLAS product stores bfloat16, as XLA's own dense dot (Dense's)
    does, no fp32 temporary beside it; a product of one row, an explicit dot algorithm, and a contraction a
    mesh axis splits keep the fp32 product and its one rounding."""
    from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec

    options = {"xla_gpu_enable_triton_gemm": False}
    x, w = jax.ShapeDtypeStruct((2048, 512), jnp.bfloat16), jax.ShapeDtypeStruct((512, 2048), jnp.bfloat16)
    one_row = jax.ShapeDtypeStruct((1, 512), jnp.bfloat16)

    def product(x, w):
        return gemm.gemm(x, w, None, out_dtype=jnp.bfloat16)

    def gradient(a, b):
        return gemm.gemm_t(a, b, None, out_dtype=jnp.bfloat16)

    def stored(fn, *shapes) -> list[str]:
        return _cublas_dtypes(lowered_for(fn, shapes, "sm90").compile(compiler_options=options))

    with generation.chosen(grouped_matmul="xla"):
        assert set(stored(product, x, w)) == {"bf16"}
        assert set(stored(gradient, x, x)) == {"bf16"}
        # A one-row product XLA reduces as a vector: its dot's own dtype, not cuBLAS's store.
        assert set(_dot_dtypes(lowered_for(product, (one_row, w), "sm90"))) == {"f32"}
        with jax.default_matmul_precision("F32_F32_F32"):
            assert set(stored(gradient, x, x)) == {"f32"}
    devices = generation.compile_only_gpus("sm90", 2, 2**36)
    mesh = Mesh(devices, ("tp",))
    split = (jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=NamedSharding(mesh, PartitionSpec(None, "tp"))),
             jax.ShapeDtypeStruct(w.shape, w.dtype, sharding=NamedSharding(mesh, PartitionSpec("tp", None))))
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="xla"), jax.set_mesh(mesh):
        compiled = jax.jit(product).lower(*split).compile(compiler_options=options)
    assert set(_cublas_dtypes(compiled)) == {"f32"}
    # Two rows an explicit data axis splits: each device's product is one row's, a vector's, kept wide.
    rows = Mesh(devices, ("data",), axis_types=(AxisType.Explicit,))
    pair = (jax.ShapeDtypeStruct((2, 512), jnp.bfloat16, sharding=NamedSharding(rows, PartitionSpec("data"))),
            jax.ShapeDtypeStruct(w.shape, w.dtype, sharding=NamedSharding(rows, PartitionSpec())))
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="xla"), jax.set_mesh(rows):
        assert set(_dot_dtypes(jax.jit(product).trace(*pair).lower())) == {"f32"}


def a_dense_bf16_products_tangent_sums_its_terms_before_rounding(product: str) -> None:
    """`jax.jvp` of a dense bf16 product (`gemm` or `gemm_t`, 4 x 2 by 2 x 4, no extent of one: a shape the
    XLA fallback stores bf16 directly where no derivative is taken) whose two tangent terms cancel to below a
    bf16 ulp of either (x = 129/128, w = 1, dx = 1, dw = -255/256 in each entry's one live term): every entry
    of the tangent is -127/32768, both terms summed in fp32 and rounded once (`gemm`'s portable program,
    every derivative's), never a bf16 rounding a term."""
    bf16 = jnp.bfloat16
    left, right = np.zeros((4, 2), np.float32), np.zeros((2, 4), np.float32)
    left[:, 0], right[0] = 129 / 128, 1.0
    d_left, d_right = np.zeros((4, 2), np.float32), np.zeros((2, 4), np.float32)
    d_left[:, 0], d_right[0] = 1.0, -255 / 256
    if product == "gemm":
        def fn(x, w):
            return gemm.gemm(x, w, None, out_dtype=bf16)
        operands, tangents = (left, right), (d_left, d_right)
    else:
        def fn(x, w):
            return gemm.gemm_t(x, w, None, out_dtype=bf16)[0]
        operands, tangents = (left.T, right), (d_left.T, d_right)

    def tangent(*arrays):
        with generation.chosen(grouped_matmul="xla"):
            return jax.jvp(fn, arrays[:2], arrays[2:])[1]

    found = jax.jit(tangent)(*(jnp.asarray(each, bf16) for each in (*operands, *tangents)))
    np.testing.assert_array_equal(np.asarray(found, np.float32), np.full((4, 4), -127 / 32768, np.float32))


@pytest.mark.parametrize("product", ["gemm", "gemm_t"])
def test_a_dense_bf16_products_tangent_sums_its_terms_before_rounding(product):
    """`a_dense_bf16_products_tangent_sums_its_terms_before_rounding` on this process's device (a GPU's
    native cell runs it on CUDA's fallback, tests/test_gpu_gemm.py)."""
    a_dense_bf16_products_tangent_sums_its_terms_before_rounding(product)
