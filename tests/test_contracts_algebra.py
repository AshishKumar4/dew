"""The algebra engine's custom derivatives each hold their contract (docs/design/testing.md, derivative
contracts): every order finite wherever its float64 truth is a normal float32 and its plain float32 formula is
finite."""

import jax
import jax.numpy as jnp
import pytest
from contract_support import EDGES, ORDER_SETS, assert_contract, normal_draw

from dew.algebra import means, moments, scan, windows


def _plain(outputs, totals, anchors):
    """The convex mean as written: each state's mass at the joined anchor, its outputs weighed by it."""
    masses = totals * jnp.exp(anchors - jnp.max(anchors, axis=0, keepdims=True))
    return jnp.sum(masses[..., None] * outputs, axis=0) / jnp.sum(masses, axis=0)[..., None]


def _merged(outputs, totals, anchors):
    """The merge of finished states the contract holds, one function at every edge, so the cell's programs
    compiled at one edge serve the others (`contract_support._mesh_program`)."""
    return means.merged_output(outputs, totals, anchors, windows.Units(), 0)


@pytest.mark.parametrize("taken", ORDER_SETS)
def test_the_convex_mean_holds_every_order(taken):
    """The merge of finished states (`means.merged_output`, its derivative the law's own over scaled values,
    `dew.algebra.scaled`'s `applied` and `_split` within it): every order, forward and reverse, under a nested
    jit and a scan, in the outputs, the totals and the anchors, finite wherever its own float64 run is a
    normal float32 and the plain mean is, its outputs and totals at each edge (`EDGES`): the law runs in
    float64 too, its construct of its operands' format. The order set `taken` (`ORDER_SETS`), at every edge,
    each order's programs compiled once a cell (`contract_support._mesh_program`): a cell an order set, where
    sets sharing a cell took 246 to 264 s (row 11720)."""
    outputs, totals, anchors = normal_draw(6096, (3, 4, 8), (3, 4), (3, 4))
    along = normal_draw(6097, (3, 4, 8), (3, 4), (3, 4))
    (cotangent,) = normal_draw(6098, (4, 8))
    for edge, scale in EDGES.items():
        primals = (outputs * scale, (jnp.abs(totals) + 0.25) * scale, anchors)
        direction = (along[0] * scale, along[1] * scale, along[2])
        assert_contract(_merged, primals, direction, cotangent, f"the convex mean {edge}", plain_fn=_plain,
                        taking=ORDER_SETS[taken])


def _chunked_weighted(x, w):
    return moments.chunked_moments(x, w, tile=4, mode="walk")


def _chunked_cov(x, y, w):
    return moments.chunked_covariance(x, y, w, tile=4, mode="merge")


# The complete reverse-over-reverse program peaks above 9 GiB cold on the standard CI host.
@pytest.mark.parametrize("taken", [pytest.param(order, marks=pytest.mark.fresh_process)
                                  if order == "grad of grad" else order for order in ORDER_SETS])
def test_chunked_moments_holds_every_order(taken):
    """Chunked weighted central moments (`moments.chunked_moments`, walk) and covariance
    (`moments.chunked_covariance`, merge) hold every derivative order across float32's exponent edges.
    Use the plain mathematical declarations in float64 as independent truth, not the same chunked AD rule.
    x, y, w = normal_draw(9045, (2, 8), (2, 8), (2, 8))
    dx, dy, dw = normal_draw(9046, (2, 8), (2, 8), (2, 8))
    c_mean, c_my, c_cov = normal_draw(9047, (2,), (2,), (2,))
    for edge, scale in EDGES.items():
        w_pos = ((jnp.abs(w) + 0.25) * scale).at[:, 0].set(0.0)
        assert_contract(_chunked_weighted, (x * scale, w_pos), (dx * scale, dw * scale), (c_mean, c_cov),
                        f"chunked moments {edge}", plain_fn=moments.weighted_moments,
                        truth_fn=moments.weighted_moments, taking=ORDER_SETS[taken])
        assert_contract(_chunked_cov, (x * scale, y * scale, w_pos), (dx * scale, dy * scale, dw * scale),
                        (c_mean, c_my, c_cov), f"chunked covariance {edge}",
                        plain_fn=moments.weighted_covariance, truth_fn=moments.weighted_covariance,
                        taking=ORDER_SETS[taken])


def test_a_decayed_pair_holds_its_derivatives():
    keys = jnp.asarray([[2.0, 0.0], [0.0, 0.0]], jnp.float32)
    decay = jnp.zeros((2, 2, 2), jnp.float32).at[1, 0, 0].set(0.5)

    def paired(x):
        rows = jnp.zeros_like(keys).at[1, 0].set(x)
        return scan._paired(rows, keys, decay)[1, 0]

    assert jax.jit(jax.grad(jax.grad(lambda x: jnp.square(paired(x)))))(jnp.float32(3)) == 2

    # The user input is finite; the read it produces overflows, but d(read)/dx and the held factors do not.
    value, gradient = jax.jit(jax.value_and_grad(lambda x: paired(2 * x)))(jnp.float32(2.0 ** 127))
    assert jnp.isposinf(value)
    assert gradient == 2


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_a_normal_draw_is_read_where_a_gpu_parts_prepare_places_its_calls(dtype):
    """`contract_support.normal_draw` made where a GPU tier part's prepare places its calls (a compile-only
    GPU, on which nothing computes or is read) is read there, each value bit for bit what JAX's conversion
    of the drawn value to `dtype` is, at float32 and at float64 under x64 (row 10552: a draw placed on the
    prepare's GPU could not be read, and its cell went unprepared)."""
    import jax
    import numpy as np
    from contract_support import normal_draw

    from dew.nn.kernels.generation import compile_only_gpus

    def converted():
        rng = np.random.default_rng(10552)
        return [np.asarray(jnp.asarray(rng.standard_normal(shape) * 0.3, dtype)) for shape in ((3, 5), (7,))]

    (gpu,) = compile_only_gpus("sm90", 1, 80 * 2**30)
    for x64 in (False, True):
        with jax.enable_x64(x64):
            expected = converted()
            with jax.default_device(gpu):
                drawn = [np.asarray(each)
                         for each in normal_draw(10552, (3, 5), (7,), scale=0.3, dtype=dtype)]
            for mine, theirs in zip(drawn, expected, strict=True):
                bits = np.uint16 if mine.itemsize == 2 else np.uint32
                np.testing.assert_array_equal(mine.view(bits), theirs.view(bits))
