"""A projection over rows a mesh splits runs per shard where a launch takes a device's rows' product
(`dew.nn.kernels.projections.mapped_shards`, rows 1306 and 1923): each device's rows its own product, each
weight entering where the traced program's layout places it (`placing`, `Layout.leaf_spec`) and gathered
whole in the map, its gradient reduce-scattered back to that placement; a joined projection's kernels
entering apart, each one's gradient summed at its own shape. A mesh that splits more than the rows, an
Explicit axis, no layout traced under, or a product no launch takes leaves the product to the partitioner."""
import contextlib
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from jax.sharding import Mesh, PartitionSpec
from reference_error import with_gradients
from sharded import MOVES, assert_runs_per_shard, collectives

from dew.distributed.placement import Layout
from dew.mesh import DEFAULT_RULES
from dew.nn.dense import Dense
from dew.nn.joined import joined_dense
from dew.nn.kernels import generation
from dew.nn.sharding import boxed, placing

LAYOUT = Layout(min_shard=1)
"""The layout the cases' programs trace under: Dew's own, every leaf split however small (its least share
one element), so the cases' small kernels are placed as a model's are."""

KERNEL_AXES = ("embed", "mlp")
"""The projections' kernels' logical axes, as a model's projections declare them (tests/test_mesh_axes.py)."""

def _declared():
    """The table a trainer's step traces under (`Trainer._declared`): the axes the projections declare."""
    return boxed({("params", name, leaf): KERNEL_AXES if leaf == "kernel" else KERNEL_AXES[-1:]
                  for name in ("first", "second") for leaf in ("kernel", "bias")})

SEEDS = 8
ROWS, LENGTH, FEATURES, WIDTH = 4, 8, 64, 64

CASES = {
    "dense-data": {"axis": "data"},
    "dense-fsdp": {"axis": "fsdp"},
    "joined-data": {"axis": "data", "join": -1},
    "joined-fsdp": {"axis": "fsdp", "join": -1},
    "joined-stacked-fsdp": {"axis": "fsdp", "join": -2},
    "rank2-fsdp": {"axis": "fsdp", "rank": 2},
    "fp32-stored-fsdp": {"axis": "fsdp", "stored": jnp.float32},
    "joined-fp32-stored-fsdp": {"axis": "fsdp", "join": -1, "stored": jnp.float32},
    "indivisible-fsdp": {"axis": "fsdp", "features": 63},
    "tensor": {"axis": "tensor"},
}
"""Each case's mesh axis (two devices), its projections joined on that output axis where `join` names one,
its input of rank 2 where `rank` says so, its parameters stored at `stored` (fp32: the layer's bf16 product
and fp32 bias rounded once, `dense._rounds_once`), and its input width (`features`, one fsdp cannot split)."""


class _Projections(nn.Module):
    """A bf16 Dense of `width`, or two joined (`joined_dense`) on output axis `join`, its parameters stored
    at `stored`."""

    width: int
    join: int | None = None
    stored: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        dense = [Dense(self.width, dtype=jnp.bfloat16, param_dtype=self.stored, kernel_axes=KERNEL_AXES,
                       name=name)
                 for name in (("first", "second") if self.join else ("first",))]
        if self.join is None:
            return dense[0](x)
        for projection in dense:
            projection(x)
        return joined_dense(x, dense, axis=self.join)


def _call(case: dict):
    """`case`'s projections as a function of its input, kernels and biases."""
    names = ("first", "second") if case.get("join") else ("first",)

    def call(x, *parameters):
        module = _Projections(parameters[0].shape[1], case.get("join"), case.get("stored", jnp.bfloat16))
        half = len(parameters) // 2
        params = {name: {"kernel": kernel, "bias": bias}
                  for name, kernel, bias in zip(names, parameters[:half], parameters[half:], strict=True)}
        return module.apply({"params": params}, x)

    return call


def _exact(join: int | None):
    """The projections' float64 arithmetic: each kernel's product with the input plus its bias, stacked on
    output axis `join` where joined."""
    def exact(x, *parameters):
        half = len(parameters) // 2
        outs = [x @ kernel + bias for kernel, bias in zip(parameters[:half], parameters[half:], strict=True)]
        return outs[0] if join is None else jnp.stack(outs, axis=join)

    def truth(*arguments):
        return with_gradients(exact)(*(argument.astype(jnp.float64) for argument in arguments))

    return truth


def _shapes(case: dict) -> tuple[tuple, tuple, tuple]:
    """`case`'s input, output and kernel shapes."""
    features = case.get("features", FEATURES)
    rows = (ROWS * LENGTH,) if case.get("rank") == 2 else (ROWS, LENGTH)
    join = case.get("join")
    out = (*rows, WIDTH) if join is None else (*rows, *((2, WIDTH) if join == -2 else (WIDTH, 2)))
    return (*rows, features), out, (features, WIDTH)


def _draw(case: dict):
    x_shape, out_shape, kernel_shape = _shapes(case)
    kernels = 2 if case.get("join") else 1
    stored = case.get("stored", jnp.bfloat16)

    def draw(seed):
        rng = np.random.default_rng(seed)
        x = jnp.asarray(rng.normal(size=x_shape), jnp.bfloat16)
        weights = [jnp.asarray(rng.normal(size=kernel_shape) / 8, stored) for _ in range(kernels)]
        biases = [jnp.asarray(rng.normal(size=(WIDTH,)), stored) for _ in range(kernels)]
        return (x, *weights, *biases, jnp.asarray(rng.normal(size=out_shape), jnp.bfloat16))

    return draw


def _kernel(name: str, shape: tuple, dtype) -> PartitionSpec:
    """Where `LAYOUT` places the projection `name`'s kernel of `shape` on the context's mesh."""
    path = tuple(jax.tree_util.DictKey(key) for key in ("params", name, "kernel"))
    return LAYOUT.leaf_spec(jax.sharding.get_abstract_mesh(), path, jax.ShapeDtypeStruct(shape, dtype))


def _specs(case: dict, mesh: Mesh) -> list:
    """Where `case`'s arguments live on `mesh`: the rows on a data or fsdp axis, each kernel where the
    layout places it, the biases whole; on a tensor axis, the kernels' outputs and the cotangent's."""
    axis, names = case["axis"], ("first", "second") if case.get("join") else ("first",)
    _, out_shape, kernel_shape = _shapes(case)
    if axis == "tensor":
        return [PartitionSpec(), *(PartitionSpec(None, axis),) * len(names), *(PartitionSpec(),) * len(names),
                PartitionSpec(*(None,) * (len(out_shape) - 1), axis)]
    with (jax.sharding.use_abstract_mesh(mesh.abstract_mesh), nn.logical_axis_rules(DEFAULT_RULES),
          _declared()):
        kernels = [_kernel(name, kernel_shape, case.get("stored", jnp.bfloat16)) for name in names]
    return [PartitionSpec(axis), *kernels, *(PartitionSpec(),) * len(names), PartitionSpec(axis)]


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("name", list(CASES))
def test_a_projection_on_a_two_device_mesh_is_on_par_with_one_device(name):
    """A bf16 Dense, or two joined, on a two-device mesh (`CASES`) under `LAYOUT`, the GEMM template's
    kernel named and interpreted as an H100's (each device's launch on its own rows), its forward and
    reverse: on par with one device's (check 1) from float64, its compiled step moving no operand between
    devices but an fsdp weight's gather."""
    case = CASES[name]
    mesh = Mesh(np.asarray(jax.devices()[:2]), (case["axis"],))
    gathers = case["axis"] == "fsdp"
    # The template named where the rows are split alone; a tensor axis's product is the partitioner's.
    named = {} if case["axis"] == "tensor" else {"grouped_matmul": "mosaic"}
    with (generation.assumed("sm90", interpreted=True), generation.chosen(**named),
          nn.logical_axis_rules(DEFAULT_RULES), placing(LAYOUT), _declared()):
        assert_runs_per_shard(with_gradients(_call(case)), _draw(case), _specs(case, mesh), mesh, name,
                              seeds=SEEDS, truth_fn=_exact(case.get("join")),
                              moves=("all-gather",) if gathers else ())


@pytest.mark.mesh(devices=2)
def test_a_projection_on_an_explicit_mesh_with_replicated_rows_runs():
    """A bf16 Dense under an Explicit two-device data mesh whose input is replicated (its type carrying its
    placement): it runs, on par with one device's from float64, the product the partitioner's as before."""
    case = {"axis": "data"}
    mesh = jax.make_mesh((2,), ("data",), axis_types=(jax.sharding.AxisType.Explicit,),
                         devices=jax.devices()[:2])
    specs = [PartitionSpec()] * 4
    with nn.logical_axis_rules(DEFAULT_RULES), placing(LAYOUT), _declared():
        assert_runs_per_shard(with_gradients(_call(case)), _draw(case), specs, mesh, "explicit", seeds=SEEDS,
                              truth_fn=_exact(None))


@pytest.mark.parametrize("axis", ["data", "fsdp"])
def test_a_projections_weight_gradient_leaves_its_map_as_its_placement_states(axis):
    """A bf16 Dense's step at a 4B model's up projection (8 sequences of 4,096 tokens, 2,560 by 9,728) over
    rows a two-device mesh splits, under `LAYOUT` with the GEMM template named, lowered for two compile-only
    H100s: its forward gathers the weight where the layout splits it and moves nothing on a data axis; its
    step reduce-scatters the weight's gradient back on fsdp and sums it whole on a data axis, moving no
    activation between devices (the template's kernel an opaque call the partitioner would read whole);
    and its temporaries are no more than the partitioner's step of the same product (`jnp.dot`, no map)."""
    from kernel_compile import _device

    mesh = _device("sm90", ((axis, 2),))
    shapes = ((8, 4096, 2560), (2560, 9728), (9728,), (8, 4096, 9728))
    with (jax.sharding.use_abstract_mesh(mesh.abstract_mesh), nn.logical_axis_rules(DEFAULT_RULES),
          _declared()):
        weight = _kernel("first", shapes[1], jnp.bfloat16)
    specs = (PartitionSpec(axis), weight, PartitionSpec(), PartitionSpec(axis))
    placed = [jax.ShapeDtypeStruct(shape, jnp.bfloat16, sharding=jax.sharding.NamedSharding(mesh, spec))
              for shape, spec in zip(shapes, specs, strict=True)]
    dense = _call({"axis": axis})

    def compiled(fn, arguments):
        with (generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"), jax.set_mesh(mesh),
              nn.logical_axis_rules(DEFAULT_RULES), placing(LAYOUT), _declared()):
            return jax.jit(fn).lower(*arguments).compile()

    def ops(executable):
        return {op for op, _ in collectives(executable.as_text())}

    def partitioned(x, kernel, bias):
        return (jnp.dot(x, kernel) + bias).astype(jnp.bfloat16)

    split = any(weight)
    assert ops(compiled(dense, placed[:3])) == ({"all-gather"} if split else set())
    summed = {"all-gather", "reduce-scatter"} if split else {"all-reduce"}
    step = compiled(with_gradients(dense), placed)
    assert summed <= ops(step) <= summed | {"all-reduce"}, ops(step)
    moved = [(op, shapes) for op, shapes in collectives(step.as_text()) if op in MOVES]
    # The weight's gather alone, whatever dimension XLA gathers it on: its whole elements.
    assert all(op == "all-gather" and [math.prod(each) for each in found] == [math.prod(shapes[1])]
               for op, found in moved), moved
    partitioner = compiled(with_gradients(partitioned), placed).memory_analysis().temp_size_in_bytes
    assert step.memory_analysis().temp_size_in_bytes <= partitioner


def test_a_projection_no_launch_takes_runs_as_it_ran_before():
    """A bf16 Dense on a two-device fsdp mesh whose product no launch takes (XLA named, as a declared cell or
    a CPU leaves it): its step under `LAYOUT` is the program it was traced under no layout, the partitioner's,
    no map around it (train 816 F3)."""
    from kernel_compile import _device

    case = {"axis": "fsdp"}
    mesh = _device("sm90", (("fsdp", 2),))
    x, *rest = _draw(case)(0)
    arguments = [jax.ShapeDtypeStruct(each.shape, each.dtype) for each in (x, *rest)]

    def lowered(*contexts):
        with (generation.assumed("sm90"), generation.chosen(grouped_matmul="xla"), jax.set_mesh(mesh),
              nn.logical_axis_rules(DEFAULT_RULES), contextlib.ExitStack() as stack):
            for context in contexts:
                stack.enter_context(context)
            return jax.jit(with_gradients(_call(case))).lower(*arguments).as_text()

    assert lowered(placing(LAYOUT)) == lowered()
