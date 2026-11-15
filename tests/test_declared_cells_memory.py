"""A declared cell's route holds no more of a training step's memory than its receipt states (row 5972's P0:
the B200's and GH200's kernel-gradient cells sent the 4B's dW to XLA's dot, which XLA deferred to the
backward's end, every layer's cotangents held, a four-row step's temporaries 22.25 GB against 11.06 GB on
the template). Each generation's dense products in a stack of layers, forward and reverse, on a compile-only
device: their temporaries under the declared cells against the template's named in their place, past what
the cells' receipts state they hold (`grouped_matmul.HELD`)."""
import functools

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn
from kernel_compile import lowered_for

from dew.distributed.placement import Layout
from dew.mesh import DEFAULT_RULES, MESH_AXES
from dew.nn import activations
from dew.nn.dense import Dense
from dew.nn.kernels import gemm, generation, grouped_matmul
from dew.nn.sharding import placing

LAYERS, ROWS, WIDTH = 4, 4096, 2048
FAMILIES = {"sm80": "mma.sync", "sm90": "wgmma", "sm100": "tcgen05"}


def _step_temp(generation_name: str, chosen: dict) -> int:
    """The temporaries a `LAYERS`-deep stack of dense products takes, its forward and its reverse, on a
    compile-only device of `generation_name`, its products routed as `chosen` names (`generation.chosen`)."""
    def loss(x, weights):
        for w in weights:
            x = jnp.tanh(gemm.gemm(x, w, None, out_dtype=jnp.bfloat16))
        return jnp.sum(x.astype(jnp.float32))

    shapes = (jax.ShapeDtypeStruct((ROWS, WIDTH), jnp.bfloat16),
              tuple(jax.ShapeDtypeStruct((WIDTH, WIDTH), jnp.bfloat16) for _ in range(LAYERS)))
    with generation.chosen(**chosen):
        return lowered_for(jax.grad(loss, argnums=(0, 1)), shapes, generation_name).compile(
            ).memory_analysis().temp_size_in_bytes


@pytest.mark.parametrize("generation_name", list(FAMILIES))
def test_a_generations_declared_cells_hold_no_more_of_a_steps_memory_than_their_receipts_state(
        generation_name):
    """A stack of `LAYERS` dense bf16 products, `ROWS` x `WIDTH` by `WIDTH` x `WIDTH`, forward and reverse:
    its temporaries under its generation's declared cells (`grouped_matmul.DECLARED`) no larger than the
    template's named in their place (`generation.chosen(grouped_matmul='mosaic')`) by more than its cells'
    stated holds (`HELD`: outputs a layer each), past one output of slack."""
    family = FAMILIES[generation_name]
    held = sum(outputs for (cell_family, kind), (outputs, _) in grouped_matmul.HELD.items()
               if cell_family == family and kind.dense and kind.matrix is None)
    output = ROWS * WIDTH * 2
    cells = _step_temp(generation_name, {})
    template = _step_temp(generation_name, {"grouped_matmul": "mosaic"})
    assert cells <= template + (held * LAYERS + 1) * output, (cells, template, held)


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
    from jax.sharding import NamedSharding, PartitionSpec
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

    def placed(leaf, spec=None):
        spec = PartitionSpec() if spec is None else spec
        return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=NamedSharding(mesh, spec))

    stack = tuple(jax.tree.map(placed, params) for _ in range(layers))
    x = placed(jax.ShapeDtypeStruct(rows, jnp.bfloat16), PartitionSpec("data"))
    with (generation.assumed(generation_name), jax.set_mesh(mesh), nn.logical_axis_rules(DEFAULT_RULES),
          placing(Layout() if layout else None)):
        return jax.jit(jax.grad(loss, argnums=(0, 1))).lower(x, stack).compile(
            ).memory_analysis().temp_size_in_bytes


@pytest.mark.parametrize("configuration", list(GATED))
@pytest.mark.parametrize("generation_name", list(FAMILIES))
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
