"""A projection over rows a mesh splits runs per shard (`dew.nn.kernels.projections.per_shard`, rows 1306 and
1923): each device's rows its own product, each weight whole in the map, gathered by the partitioner from its
layout's placement and its gradient returned there (reduce-scattered where fsdp splits it, summed whole on a
data axis); a joined projection's kernels entering apart, each one's gradient summed at its own shape. A mesh
that splits more than the rows, or an Explicit axis, leaves the product to the partitioner."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from jax.sharding import Mesh, PartitionSpec
from reference_error import with_gradients
from sharded import MOVES, assert_runs_per_shard, collectives

from dew.mesh import DEFAULT_RULES
from dew.nn.dense import Dense
from dew.nn.joined import joined_dense
from dew.nn.kernels import generation

SEEDS = 8
ROWS, LENGTH, FEATURES, WIDTH = 4, 8, 64, 32

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
    # fsdp splits the kernel's outputs where it cannot split its inputs, as the layout places it.
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
        dense = [Dense(self.width, dtype=jnp.bfloat16, param_dtype=self.stored, name=name)
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


def _specs(case: dict) -> list:
    """Where `case`'s arguments live: the rows on a data or fsdp axis, the kernels' inputs on fsdp where it
    divides them and their outputs where it does not (the layout's placement), the kernels' outputs and the
    cotangent's on a tensor axis."""
    axis, kernels = case["axis"], 2 if case.get("join") else 1
    _, out_shape, kernel_shape = _shapes(case)
    if axis == "tensor":
        return [PartitionSpec(), *(PartitionSpec(None, axis),) * kernels, *(PartitionSpec(),) * kernels,
                PartitionSpec(*(None,) * (len(out_shape) - 1), axis)]
    weight = (PartitionSpec() if axis != "fsdp" else PartitionSpec(axis) if kernel_shape[0] % 2 == 0
              else PartitionSpec(None, axis))
    return [PartitionSpec(axis), *(weight,) * kernels, *(PartitionSpec(),) * kernels, PartitionSpec(axis)]


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("name", list(CASES))
def test_a_projection_on_a_two_device_mesh_is_on_par_with_one_device(name):
    """A bf16 Dense, or two joined, on a two-device mesh (`CASES`), its forward and reverse: on par with one
    device's (check 1) from float64, its compiled step moving no operand between devices but an fsdp
    weight's gather."""
    case = CASES[name]
    mesh = Mesh(np.asarray(jax.devices()[:2]), (case["axis"],))
    gathers = case["axis"] == "fsdp"
    with nn.logical_axis_rules(DEFAULT_RULES):
        assert_runs_per_shard(with_gradients(_call(case)), _draw(case), _specs(case), mesh, name, seeds=SEEDS,
                              truth_fn=_exact(case.get("join")), moves=("all-gather",) if gathers else ())


@pytest.mark.mesh(devices=2)
def test_a_projection_on_an_explicit_mesh_with_replicated_rows_runs():
    """A bf16 Dense under an Explicit two-device data mesh whose input is replicated (its type carrying its
    placement): it runs, on par with one device's from float64, the product the partitioner's as before."""
    case = {"axis": "data"}
    mesh = jax.make_mesh((2,), ("data",), axis_types=(jax.sharding.AxisType.Explicit,),
                         devices=jax.devices()[:2])
    specs = [PartitionSpec()] * 4
    with nn.logical_axis_rules(DEFAULT_RULES):
        assert_runs_per_shard(with_gradients(_call(case)), _draw(case), specs, mesh, "explicit", seeds=SEEDS,
                              truth_fn=_exact(None))


@pytest.mark.parametrize("axis", ["data", "fsdp"])
def test_a_projections_weight_gradient_leaves_its_map_as_its_placement_states(axis):
    """A bf16 Dense's step at a 4B model's up projection (8 sequences of 4,096 tokens, 2,560 by 9,728) over
    rows a two-device mesh splits, lowered for two compile-only H100s: its forward gathers the weight where
    fsdp splits it and moves nothing on a data axis; its step reduce-scatters the weight's gradient back on
    fsdp and sums it whole on a data axis, and on the template's kernel (an opaque call the partitioner
    cannot split) moves no activation between devices; and its temporaries are no more than the
    partitioner's step of the same product (`jnp.dot`, no map)."""
    from kernel_compile import _device

    mesh = _device("sm90", ((axis, 2),))
    weight = PartitionSpec(axis) if axis == "fsdp" else PartitionSpec()
    shapes = ((8, 4096, 2560), (2560, 9728), (9728,), (8, 4096, 9728))
    specs = (PartitionSpec(axis), weight, PartitionSpec(), PartitionSpec(axis))
    placed = [jax.ShapeDtypeStruct(shape, jnp.bfloat16, sharding=jax.sharding.NamedSharding(mesh, spec))
              for shape, spec in zip(shapes, specs, strict=True)]
    dense = _call({"axis": axis})

    def compiled(fn, arguments, **chosen):
        with (generation.assumed("sm90"), generation.chosen(**chosen), jax.set_mesh(mesh),
              nn.logical_axis_rules(DEFAULT_RULES)):
            return jax.jit(fn).lower(*arguments).compile()

    def ops(executable):
        return {op for op, _ in collectives(executable.as_text())}

    def partitioned(x, kernel, bias):
        return (jnp.dot(x, kernel) + bias).astype(jnp.bfloat16)

    gathered = {"all-gather"} if axis == "fsdp" else set()
    assert ops(compiled(dense, placed[:3])) == gathered
    summed = {"all-gather", "reduce-scatter"} if axis == "fsdp" else {"all-reduce"}
    step = compiled(with_gradients(dense), placed)
    assert summed <= ops(step) <= summed | {"all-reduce"}, ops(step)
    kernel = compiled(with_gradients(dense), placed, grouped_matmul="mosaic")
    moved = [(op, shapes) for op, shapes in collectives(kernel.as_text()) if op in MOVES]
    gather = ("all-gather", "fsdp", [(2560, 9728)])
    weights = [(op, shapes) for op, shapes in moved if (op, axis, shapes) == gather]
    assert moved == weights, moved
    partitioner = compiled(with_gradients(partitioned), placed).memory_analysis().temp_size_in_bytes
    assert step.memory_analysis().temp_size_in_bytes <= partitioner
