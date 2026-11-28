"""Which program each mode of differentiation runs through `generation.linear_tangent`: forward mode the
portable arithmetic's tangent, reverse mode the backward kernel, however a jit, a shard_map, a scan, a
remat or a vmap stages the call."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec
from reference_error import assert_as_exact

from dew.nn.derivatives import KernelGlue
from dew.nn.kernels import generation


def tripled(x, index):
    """Three times `x`, whose backward kernel returns four times the cotangent, not the portable
    arithmetic's pullback, so a gradient says by its value which program ran: 4 the backward kernel, 3
    the portable arithmetic's. The forward kernel has no derivative of its own (a stop_gradient), so a
    tangent of 3 is the portable arithmetic's. `index`, an integer operand, takes float0's cotangent."""
    return generation.linear_tangent(
        lambda x, index: (jax.lax.stop_gradient(3 * x), ()),
        lambda saved, operands, cotangent: (4 * cotangent, np.zeros(operands[1].shape, jax.dtypes.float0)),
        lambda x, index: 3 * x, kind=KernelGlue())(x, index)


def summed(x, index):
    return tripled(x, index).sum()


def reversed_through(staging):
    """The gradient of `summed` at every element, reached through `staging`."""
    mesh = Mesh(np.array(jax.devices()), ("rows",))
    rows = PartitionSpec("rows")
    if staging == "vjp":
        return jax.grad(summed)
    if staging == "jit":
        return jax.jit(jax.grad(summed))
    if staging == "shard_map":
        # shard_map transposes the jvp of its body rather than taking a vjp.
        return jax.jit(jax.grad(lambda x, index: jax.shard_map(
            tripled, mesh=mesh, in_specs=(rows, rows), out_specs=rows)(x, index).sum()))
    if staging == "scan":
        def scanned(x, index):
            return jax.lax.scan(lambda carry, row: (carry, tripled(*row)), None,
                                (x.reshape(2, -1), index.reshape(2, -1)))[1].sum()

        return jax.grad(scanned)
    if staging == "remat":
        return jax.grad(jax.checkpoint(summed))
    if staging == "linearize":
        def transposed(x, index):
            linear = jax.linearize(lambda x: tripled(x, index), x)[1]
            return jax.linear_transpose(linear, x)(jnp.ones_like(x))[0]

        return transposed
    if staging == "vmap of a gradient":
        return lambda x, index: jax.vmap(jax.grad(summed))(x.reshape(2, -1), index.reshape(2, -1)).reshape(-1)
    if staging == "gradient of a vmap":
        return jax.grad(lambda x, index: jax.vmap(tripled)(x.reshape(2, -1), index.reshape(2, -1)).sum())
    # An operand two rows share: its cotangent sums the rows'.
    def shared(x, index):
        return jax.vmap(tripled, in_axes=(None, 0))(x, jnp.stack([index, index])).sum() / 2

    return jax.jit(jax.grad(shared))


@pytest.mark.parametrize("staging", ["vjp", "jit", "shard_map", "scan", "remat", "linearize",
                                     "vmap of a gradient", "gradient of a vmap", "shared across a vmap"])
def test_reverse_mode_runs_the_backward_kernel_however_the_call_is_staged(staging):
    """A vjp, or a jvp that a shard_map, a scan or `jax.linear_transpose` then transposes, of the call or of
    a batch of it: each runs the backward kernel (`generation.linear_map`)."""
    x = jnp.arange(8.0 * jax.device_count())
    index = jnp.arange(x.size)
    np.testing.assert_array_equal(np.asarray(reversed_through(staging)(x, index)), np.full(x.shape, 4.0))


@pytest.mark.parametrize("staging", ["jvp", "jit", "linearize", "vmap"])
def test_forward_mode_takes_the_portable_tangent_however_the_call_is_staged(staging):
    """`jax.jvp` and `jax.linearize`, of the call or of a batch of it, staged by a jit or not, evaluate the
    portable arithmetic's tangent."""
    x, index = jnp.arange(8.0), jnp.arange(8)
    direction = jnp.ones_like(x)

    def tangent(x):
        if staging == "linearize":
            return jax.linearize(lambda x: tripled(x, index), x)[1](direction)
        if staging == "vmap":
            return jax.jvp(lambda x: jax.vmap(tripled)(x.reshape(2, -1), index.reshape(2, -1)), (x,),
                           (direction,))[1].reshape(-1)
        return jax.jvp(lambda x: tripled(x, index), (x,), (direction,))[1]

    run = jax.jit(tangent) if staging == "jit" else tangent
    np.testing.assert_array_equal(np.asarray(run(x)), np.full(x.shape, 3.0))


def test_forward_over_reverse_differentiates_the_portable_pullback():
    """`jax.jvp` of a gradient: the backward kernel's tangent is the portable pullback's, which for a
    linear map is zero, its integer operand's tangent float0's along the way."""
    x, index = jnp.arange(8.0), jnp.arange(8)
    tangent = jax.jvp(lambda x: jax.grad(summed)(x, index), (x,), (jnp.ones_like(x),))[1]
    np.testing.assert_array_equal(np.asarray(tangent), np.zeros(x.shape))


def squared(x):
    """`x` squared through a kernel whose derivative runs its forward held still (`nested` off): its forward
    `x^2` saving `x`, its backward twice the saved `x` times the cotangent, its portable arithmetic `x^2`."""
    return generation.linear_tangent(
        lambda x: (jax.lax.stop_gradient(x * x), jax.lax.stop_gradient(x)),
        lambda saved, operands, cotangent: (2 * saved * cotangent,),
        lambda x: x * x, kind=KernelGlue(), nested=False)(x)


def _orders(square):
    """Each higher-order derivative of the loss `sum(square(x)^2)`, nonlinear in `square`'s output, along a
    direction: the Hessian's product with it by `jax.jvp` of the gradient and by the gradient of the
    gradient's product with it, `12 x^2` times it, and the third derivative twice along it, `24 x` times its
    square."""
    def loss(x):
        return jnp.sum(square(x) ** 2)

    def forward_over_reverse(x, direction):
        return jax.jvp(jax.grad(loss), (x,), (direction,))[1]

    return {
        "forward over reverse": forward_over_reverse,
        "reverse over reverse": lambda x, direction: jax.grad(
            lambda x: jnp.vdot(jax.grad(loss)(x), direction))(x),
        "third order": lambda x, direction: jax.jvp(
            lambda x: forward_over_reverse(x, direction), (x,), (direction,))[1],
    }


ORDERS = {label: (dew, _orders(lambda x: x * x)[label]) for label, dew in _orders(squared).items()}
"""Each order's Dew arm, through `squared`'s kernel, beside its reference, through the portable `x * x`."""

SEEDS = {"forward over reverse": 8, "reverse over reverse": 16, "third order": 8}
"""Each order's declared seeds, sized paired (`seeds_for` with its Dew arm) at 1d187ee02f6 on armada."""


def held_still_draw(draw: int) -> tuple[jax.Array, jax.Array]:
    """A draw's point and direction, 257 values each: the point's magnitudes spread over two decades, so the
    curvature `12 x^2` and the third derivative `24 x` round at every scale."""
    point = jax.random.normal(jax.random.fold_in(jax.random.key(81), draw), (257,), jnp.float32)
    direction = jax.random.normal(jax.random.fold_in(jax.random.key(82), draw), (257,), jnp.float32)
    return point * 10.0 ** jax.random.uniform(jax.random.fold_in(jax.random.key(83), draw), (257,),
                                              minval=-1.0, maxval=1.0), direction


@pytest.mark.parametrize("order", list(ORDERS))
def test_a_held_still_kernels_higher_orders_keep_its_loss_curvature(order):
    """Through a kernel whose derivative runs its forward held still (`nested` off), the loss `sum(x^4)`'s
    Hessian along a direction, by forward over reverse and by reverse over reverse, and its third derivative
    twice along it, are on par with the portable composition's (check 1, truth float64): its curvature `J^T H
    J` among them, where the output held still carried no tangent of its own and both second orders gave `4
    x^2` (row 8648), and the third order lost its part, 32 at two where `24 x` is 48 (row 8573's head
    review)."""
    dew, reference = ORDERS[order]
    assert_as_exact(jax.jit(dew), jax.jit(reference), held_still_draw, order, seeds=SEEDS[order])


def test_a_linear_maps_batched_pullback_sums_after_it_pulls_back():
    """A linear map that quarters its operand (`generation.linear_map`), shared by two lanes of a `jax.vmap`,
    transposed at float32 cotangents of `2^127` each: `2^126`, each lane's cotangent pulled back before the
    batch's sum, where summed first the sum overflowed to infinity before its quarter (base review 7fa8fe6c2
    F1); a narrowing pullback asks for the sum first itself (`summed_first`, an exact widening's)."""
    from dew.nn.kernels.generation import linear_map

    def quartered(x):
        return linear_map(lambda _, moving: moving * 0.25, lambda _, cotangent: cotangent * 0.25, (), x)

    lanes = jax.vmap(quartered, in_axes=None, axis_size=2)
    (pulled,) = jax.linear_transpose(lanes, jnp.float32(1.0))(jnp.full((2,), 2.0 ** 127, jnp.float32))
    assert pulled.dtype == jnp.float32 and float(pulled) == 2.0 ** 126


def test_a_linear_maps_batched_pullback_sums_each_cotangent_in_its_own_type():
    """A linear map of a float64 operand (`generation.linear_map`), shared by two lanes of a `jax.vmap`,
    transposed after each lane pulls back (not `summed_first`): the batch's float64 cotangents of `2^127`
    summed as float64 to `2^128` whether the pullback runs in x64 or outside it, never as float32 first (an
    infinity; row 14083's review); and a bfloat16 lane's sum in bfloat16, bit for bit JAX's own transpose
    of the batch's broadcast (`lax.reduce_sum`, Train 1474's review)."""
    from dew.nn.kernels.generation import linear_map

    def kept(x):
        return linear_map(lambda _, moving: moving, lambda _, cotangent: cotangent, (), x)

    with jax.enable_x64(new_val=True):
        _, pull = jax.vjp(jax.vmap(kept, in_axes=None, axis_size=2), jnp.float64(1.0))
        lanes = jnp.full((2,), 2.0 ** 127, jnp.float64)
        (inside,) = pull(lanes)
    (outside,) = pull(lanes)
    for found in (inside, outside):
        assert found.dtype == jnp.float64 and float(found) == 2.0 ** 128
    small = jnp.asarray([1.0, 2.0 ** -8, 2.0 ** -8], jnp.bfloat16)
    (ours,), (theirs,) = (jax.linear_transpose(jax.vmap(f, in_axes=None, axis_size=3), small[0])(small)
                          for f in (kept, lambda x: x))
    assert ours.dtype == jnp.bfloat16
    assert np.asarray(ours).view(np.uint16) == np.asarray(theirs).view(np.uint16)


@pytest.mark.mesh(devices=2)
def test_a_linear_maps_replicated_operand_takes_its_split_residuals_summed_cotangent():
    """A linear map `r · x` (`generation.linear_map`) of an `x` every device holds alike by residuals `r` a
    checked `shard_map` splits by rows: its pullback, `Σ r · c` over each device's rows, is that device's
    partial of `x`'s cotangent, which the map sums as `x` varies (none of the map's axes), so the gradient
    through the map is the unsplit call's, its pullback summing nothing itself; unsummed, the map's
    transpose handed the accumulator a cotangent varying over the axis (train 1511's K-F2)."""
    from jax.sharding import AxisType, PartitionSpec as P

    from dew.nn.kernels.generation import linear_map

    mesh = jax.make_mesh((2,), ("data",), axis_types=(AxisType.Auto,), devices=jax.devices()[:2])
    rows = jnp.arange(24, dtype=jnp.float32).reshape(6, 4)

    def scaled(rows, x):
        return linear_map(lambda rows, x: rows * x, lambda rows, cotangent: jnp.sum(rows * cotangent, axis=0),
                          rows, x)

    local = jax.shard_map(scaled, mesh=mesh, in_specs=(P("data"), P()), out_specs=P("data"))
    out, pull = jax.vjp(lambda x: local(rows, x), jnp.ones((4,), jnp.float32))
    (pulled,) = pull(jnp.ones_like(out))
    np.testing.assert_array_equal(np.asarray(pulled), np.asarray(rows).sum(axis=0))


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize(("rows", "summed", "names"), [
    ([2.0 ** 127, 2.0 ** 127, -(2.0 ** 127), -(2.0 ** 127)], 0.0, (None,)),
    ([1.0, 2.0, 3.0, 4.0], 10.0, ("data",)),
    ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 36.0, ("data", None))],
    ids=["cancelling", "named as the mesh", "nested in one named as the mesh"])
def test_a_batched_linear_maps_shared_operand_sums_each_lane_across_devices_first(rows, summed, names):
    """A linear map `r · x` batched (`jax.vmap`, two lanes a device, a level of `names`, outermost first)
    over residuals a checked `shard_map` splits, `x` shared by the lanes and every device: each lane's
    cotangent of `x` summed across the devices before the lanes add (`generation._batched_pullback`), as the
    map's unbatched transpose sums it, so a lane's partials `2^127` on one device and `-2^127` on the other
    cancel to 0, where the lanes added first read `2^128` and `-2^128`, inf on each device, and their sum NaN
    (train 1511's review); and under a batch named as the mesh's axis, or an anonymous one within it,
    summed over the mesh's devices, not the batch's lanes, as the gradient of `Σ r` (train 1511's second
    and third reviews)."""
    from jax.sharding import AxisType, PartitionSpec as P

    from dew.nn.kernels.generation import linear_map

    mesh = jax.make_mesh((2,), ("data",), axis_types=(AxisType.Auto,), devices=jax.devices()[:2])
    residuals = jnp.asarray(rows, jnp.float32)[:, None]

    def lane(row, x):
        return linear_map(lambda row, x: row * x, lambda row, cotangent: row * cotangent, row, x)

    def scaled(rows, x):
        mapped = lane
        for name in reversed(names):
            mapped = jax.vmap(mapped, in_axes=(0, None), axis_name=name)
        return mapped(rows.reshape(*(2,) * len(names), 1), x).reshape(-1, 1)

    local = jax.shard_map(scaled, mesh=mesh, in_specs=(P("data"), P()), out_specs=P("data"))
    out, pull = jax.vjp(lambda x: local(residuals, x), jnp.ones((1,), jnp.float32))
    (pulled,) = pull(jnp.ones_like(out))
    np.testing.assert_array_equal(np.asarray(pulled), np.full((1,), summed, np.float32))


def test_a_named_batchs_pullback_reads_its_lanes_name():
    """A linear map `x (1 + i)` of each lane `i` of a batch named "lane" (`lax.axis_index` in its apply and
    its pullback alike): the gradient of its sum over two lanes `[1, 2]`, the batch's name kept for its
    pullback (`generation._batched_pullback`), where its lanes mapped unnamed read no such axis (train 1515's
    review F2)."""
    from dew.nn.kernels.generation import linear_map

    def scaled(x):
        return x * (1.0 + jax.lax.axis_index("lane"))

    def lane(x):
        return linear_map(lambda _, x: scaled(x), lambda _, cotangent: scaled(cotangent), x, x)

    gradient = jax.grad(lambda x: jax.vmap(lane, axis_name="lane")(x).sum())(jnp.ones((2,), jnp.float32))
    np.testing.assert_array_equal(np.asarray(gradient), [1.0, 2.0])
