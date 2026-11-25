"""A projection over rows a mesh splits runs per shard where a launch takes a device's rows' product
(`dew.nn.kernels.projections.mapped_shards`, rows 1306 and 1923): each device's rows its own product, each
weight entering where the traced program's layout places it (`placing`, `Layout.leaf_spec`) and gathered
whole in the map, its gradient reduce-scattered back to that placement; a joined projection's kernels
entering apart, each one's gradient summed at its own shape. A mesh that splits more than the rows, an
Explicit axis, no layout traced under, or a product no launch takes leaves the product to the partitioner."""
import contextlib
import math
import re

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
from dew.telemetry.instrumentation import hlo_instructions
from dew.telemetry.trace import collective

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
        if self.is_initializing():
            # Each projection's parameters made; an apply runs the joined product alone.
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


def unlaunched() -> list[str]:
    """The cases no GEMM kernel launches per shard (`gemm_matrix.launched`): the layer's own (a tensor axis's
    product, the partitioner's; a weight stored in fp32, rounded once). Every other case is a kernel's
    per-shard cell (`gemm_matrix.per_shard`, tests/test_per_shard.py)."""
    from gemm_matrix import launched

    taken = {name for name, _ in launched()}
    return [name for name in CASES if name not in taken]


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("name", unlaunched())
def test_a_projection_no_kernel_launches_per_shard_is_on_par_with_one_device(name):
    """A bf16 Dense, or two joined, on a two-device mesh (`unlaunched`) under `LAYOUT`: on par with one
    device's (check 1) from float64, its compiled step moving no operand between devices but an fsdp
    weight's gather."""
    case = CASES[name]
    mesh = Mesh(np.asarray(jax.devices()[:2]), (case["axis"],))
    with (generation.assumed("sm90", interpreted=True), nn.logical_axis_rules(DEFAULT_RULES), placing(LAYOUT),
          _declared()):
        # An fsdp weight gathered, its gradient reduce-scattered back (the gather's transpose).
        fsdp = case["axis"] == "fsdp"
        assert_runs_per_shard(with_gradients(_call(case)), _draw(case), _specs(case, mesh), mesh, name,
                              seeds=SEEDS, truth_fn=_exact(case.get("join")),
                              moves=("all-gather",) if fsdp else (),
                              reductions=(("reduce-scatter", ("fsdp",)),) if fsdp else ())


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("axis", ["data", "fsdp"])
def test_a_projection_read_apart_from_its_layer_runs_per_shard_as_its_layer_does(axis):
    """A projection read apart from its layer (`protocols.Projection`, as a vocabulary head reads
    `project_out`, through `dense.plain_product`), its kernel the parameter at its layer's path, on a
    two-device mesh splitting its rows, the template's kernel named and interpreted as an H100's: each
    device's launch on its own rows, on par with one device's (check 1) from float64, its compiled step
    moving no operand between devices but an fsdp weight's gather (row 6047: it went to the partitioner)."""
    from dew.nn.protocols import Projection

    path = tuple(jax.tree_util.DictKey(key) for key in ("params", "first", "kernel"))
    mesh = Mesh(np.asarray(jax.devices()[:2]), (axis,))

    def call(x, kernel):
        return Projection(kernel, path=path)(x)

    def draw(seed):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(rng.normal(size=(ROWS, LENGTH, FEATURES)), jnp.bfloat16),
                jnp.asarray(rng.normal(size=(FEATURES, WIDTH)) / 8, jnp.bfloat16),
                jnp.asarray(rng.normal(size=(ROWS, LENGTH, WIDTH)), jnp.bfloat16))

    def truth(x, kernel, cotangent):
        return with_gradients(lambda x, kernel: x @ kernel)(
            *(argument.astype(jnp.float64) for argument in (x, kernel, cotangent)))

    with (jax.sharding.use_abstract_mesh(mesh.abstract_mesh), nn.logical_axis_rules(DEFAULT_RULES),
          _declared()):
        kernel = _kernel("first", (FEATURES, WIDTH), jnp.bfloat16)
    with (generation.assumed("sm90", interpreted=True), generation.chosen(grouped_matmul="mosaic"),
          nn.logical_axis_rules(DEFAULT_RULES), placing(LAYOUT), _declared()):
        assert_runs_per_shard(with_gradients(call), draw, [PartitionSpec(axis), kernel, PartitionSpec(axis)],
                              mesh, f"a projection apart from its layer {axis}", seeds=SEEDS, truth_fn=truth,
                              moves=("all-gather",) if axis == "fsdp" else (),
                              reductions=(("reduce-scatter", ("fsdp",)),) if axis == "fsdp" else ())


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
    and its temporaries are no more than the partitioner's step of the same product (`jnp.dot`, no map),
    each compiled for H100s stating 128 MiB, below the step's need, so XLA's scheduler holds the least it can
    and the temporaries are the route's alone (an H100's 80 GB lets it hold the step's twice over, 99.6 MB
    against the partitioner's 49.8 MB; a compile-only GPU stated 0 bytes before row 8768)."""
    from kernel_compile import _device

    mesh = _device("sm90", ((axis, 2),), memory=2**27)
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
        return {op for op, _ in collectives(executable)}

    def partitioned(x, kernel, bias):
        return (jnp.dot(x, kernel) + bias).astype(jnp.bfloat16)

    split = any(weight)
    assert ops(compiled(dense, placed[:3])) == ({"all-gather"} if split else set())
    summed = {"all-gather", "reduce-scatter"} if split else {"all-reduce"}
    step = compiled(with_gradients(dense), placed)
    assert summed <= ops(step) <= summed | {"all-reduce"}, ops(step)
    moved = [(op, shapes) for op, shapes in collectives(step) if op in MOVES]
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


def test_a_one_member_batched_projection_runs_as_the_dense_product_it_is():
    """A bf16 `DenseGeneral` batched over one member (`[1, 32, 64]` rows, a `[1, 64, 48]` kernel) on an
    assumed B200, nothing named: its launch is the dense product's (`dense` bound), the one its probe asked
    about, where the probe asked the dense product's cells and the call ran one group's grouped launch (head
    review of 0099a4701 F3)."""
    from dew.nn.dense import DenseGeneral
    from dew.nn.kernels import tuning

    layer = DenseGeneral(48, batch_dims=(0,), axis=-1, use_bias=False, dtype=jnp.bfloat16)
    x = jax.ShapeDtypeStruct((1, 32, 64), jnp.bfloat16)
    params = jax.eval_shape(layer.init, jax.random.key(0), x)
    with generation.assumed("sm100"), tuning.seen() as launches:
        jax.eval_shape(layer.apply, params, x)
    assert launches.taken and all(bound.arguments["dense"] for _, bound in launches.taken), launches.taken


def test_a_batched_projection_runs_its_members_as_one_grouped_gemm():
    """A bf16 `DenseGeneral` batched over 4 groups (DeepSeek V4's grouped `o_a_proj`: `[4, 32, 64]` rows, a
    `[4, 64, 48]` kernel) on an assumed H100, the GEMM template named and interpreted: its members run as one
    grouped GEMM's uniform groups (`projections.grouped_rows`, Appendix C row 4), the template's route taken,
    and its output and the gradients of its rows and kernel are XLA's batched dot's within
    tests/reference_error.py's rule from float64 (check 1), where the batched call bypassed the template
    (spec alignment SA20261119pm F36)."""
    from reference_error import assert_as_exact

    from dew.nn.dense import DenseGeneral

    layer = DenseGeneral(48, batch_dims=(0,), axis=-1, use_bias=False, dtype=jnp.bfloat16)

    def draw(seed):
        keys = jax.random.split(jax.random.key(seed), 3)
        kernel = jax.random.normal(keys[0], (4, 64, 48)).astype(jnp.bfloat16)
        return ({"params": {"kernel": kernel}}, jax.random.normal(keys[1], (4, 32, 64)).astype(jnp.bfloat16),
                jax.random.normal(keys[2], (4, 32, 48)).astype(jnp.bfloat16))

    def under(kernel, run):
        def arm(*operands):
            with generation.chosen(grouped_matmul=kernel):
                return run(*operands)
        return arm

    truth = under("xla", with_gradients(layer.clone(dtype=jnp.float64).apply))
    with generation.assumed("sm90", interpreted=True), generation.routes() as routes:
        run = with_gradients(layer.apply)
        assert_as_exact(under("mosaic", run), under("xla", run), draw, "a batched projection's grouped GEMM",
                        seeds=8, truth_fn=truth)
    assert ("grouped_matmul", "mosaic") in {(route.choice, route.kernel) for route in routes}, routes


@pytest.mark.parametrize("axis", ["data", "fsdp"])
def test_a_mapped_stacks_weight_reductions_complete_per_layer(axis):
    """A rematerialized stack of four bf16 Dense layers mapped per shard (`on_shards`, `mapped_shards`)
    under the trainer's collective combiner (`dew.training.memory.COLLECTIVES`) on two compile-only H100s:
    each layer's weight-gradient collective (`all-reduce` on `data`, `reduce-scatter` on `fsdp`) completes
    before its input cotangent enters the layer below (`sharding.on_shards` tying them with
    `formed_together`, row 12898): every layer above the first has a collective of its own whose result the
    layers below's products read, where XLA's combiner had merged all four layers' unreduced weight
    gradients into one collective at the step's end, which no product waits on."""
    from kernel_compile import _device

    from dew.training.memory import COLLECTIVES

    layers = 4
    mesh = _device("sm90", ((axis, 2),))
    x_shape = (2, 1024, 512)
    w_shape = (512, 512)
    with (jax.sharding.use_abstract_mesh(mesh.abstract_mesh), nn.logical_axis_rules(DEFAULT_RULES),
          _declared()):
        weight = _kernel("first", w_shape, jnp.bfloat16)

    def loss(x, kernels):
        def block(h, kernel):
            return h + Dense(512, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, use_bias=False,
                             kernel_axes=KERNEL_AXES, name="first").apply({"params": {"kernel": kernel}}, h)

        for kernel in kernels:
            x = jax.checkpoint(block)(x, kernel)
        return jnp.sum(x.astype(jnp.float32))

    x_spec = jax.ShapeDtypeStruct(x_shape, jnp.bfloat16,
                                  sharding=jax.sharding.NamedSharding(mesh, PartitionSpec(axis)))
    w_spec = jax.ShapeDtypeStruct(w_shape, jnp.bfloat16,
                                  sharding=jax.sharding.NamedSharding(mesh, weight))
    kernels = (w_spec,) * layers
    with (generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"), jax.set_mesh(mesh),
          nn.logical_axis_rules(DEFAULT_RULES), placing(LAYOUT), _declared()):
        step = jax.jit(jax.grad(loss, argnums=(0, 1))).lower(x_spec, kernels).compile(
            compiler_options=COLLECTIVES)
    expected = "reduce-scatter" if any(weight) else "all-reduce"
    w_elems = math.prod(w_shape) // (2 if any(weight) else 1)
    instructions = hlo_instructions(step)
    users: dict[str, list[str]] = {}
    for name, instruction in instructions.items():
        for operand in instruction.operands:
            users.setdefault(operand, []).append(name)
    # The tie's barrier is a control dependency once XLA expands it (`OptimizationBarrierExpander`).
    for line in step.as_text().splitlines():
        if (ordered := re.match(r"\s*(?:ROOT )?%?(\S+) = .*control-predecessors=\{([^}]*)\}", line)):
            for operand in ordered[2].split(","):
                users.setdefault(operand.strip().lstrip("%"), []).append(ordered[1])

    def read_by_a_product(name: str) -> bool:
        seen, frontier = set(), list(users.get(name, ()))
        while frontier:
            if (each := frontier.pop()) not in seen:
                seen.add(each)
                if instructions[each].flops:
                    return True
                frontier.extend(users.get(each, ()))
        return False

    reduces = [name for name, instruction in instructions.items()
               if collective(instruction.op) == expected and not instruction.op.endswith("-start")
               and sum(math.prod(shape) for shape in instruction.shapes) >= w_elems]
    awaited = [name for name in reduces if read_by_a_product(name)]
    assert len(awaited) == layers - 1, (expected, reduces, awaited)
