"""Chunked central moments hold their derivative laws at every float32 exponent edge."""

import jax.numpy as jnp
import pytest
from contract_support import EDGES, ORDER_SETS, assert_contract, normal_draw

from dew.algebra import moments

# Give each first order a task of its own. Keep reverse-over-reverse split at the placement boundary.
_ORDERS = (
    *(pytest.param((order,), "both", id=order) for order in ORDER_SETS["first orders"]),
    *(pytest.param(orders, "both", id=name) for name, orders in ORDER_SETS.items()
      if name not in ("first orders", "grad of grad")),
    pytest.param(("grad of grad",), "local", marks=pytest.mark.fresh_process, id="grad of grad-local"),
    pytest.param(("grad of grad",), "mesh", marks=pytest.mark.fresh_process, id="grad of grad-mesh"),
)


def _weighted(x, w):
    return moments.chunked_moments(x, w, tile=4, mode="walk")


def _covariance(x, y, w):
    return moments.chunked_covariance(x, y, w, tile=4, mode="merge")


def _weights(weights, scale):
    return ((jnp.abs(weights) + 0.25) * scale).at[:, 0].set(0.0)


@pytest.mark.parametrize(("orders", "placement"),
                         [case for case in _ORDERS if case.id != "grad of grad-local"])
def test_the_weighted_moments_walk_holds_every_order(orders, placement):
    _weighted_contract(orders, placement, EDGES)


def _value_gradient(result):
    value, _ = result
    return value


def _weight_gradient(result):
    _, weight = result
    return weight


@pytest.mark.fresh_process
@pytest.mark.parametrize("edge", EDGES)
@pytest.mark.parametrize("result_leaf", [_value_gradient, _weight_gradient], ids=["value", "weight"])
def test_the_local_weighted_walk_holds_reverse_over_reverse_at_every_scale(edge, result_leaf):
    _weighted_contract(("grad of grad",), "local", {edge: EDGES[edge]}, result_leaf=result_leaf)


def _weighted_contract(orders, placement, edges, *, result_leaf=None):
    x, _, w = normal_draw(9045, (2, 8), (2, 8), (2, 8))
    dx, _, dw = normal_draw(9046, (2, 8), (2, 8), (2, 8))
    c_mean, _, c_var = normal_draw(9047, (2,), (2,), (2,))
    for edge, scale in edges.items():
        assert_contract(_weighted, (x * scale, _weights(w, scale)), (dx * scale, dw * scale), (c_mean, c_var),
                        f"chunked moments {edge}", plain_fn=moments.weighted_moments,
                        truth_fn=moments.weighted_moments, taking=orders, placement=placement,
                        result_leaf=result_leaf)


@pytest.mark.parametrize(("orders", "placement"), _ORDERS)
def test_the_covariance_merge_holds_every_order(orders, placement):
    x, y, w = normal_draw(9045, (2, 8), (2, 8), (2, 8))
    dx, dy, dw = normal_draw(9046, (2, 8), (2, 8), (2, 8))
    c_mx, c_my, c_cov = normal_draw(9047, (2,), (2,), (2,))
    for edge, scale in EDGES.items():
        assert_contract(_covariance, (x * scale, y * scale, _weights(w, scale)),
                        (dx * scale, dy * scale, dw * scale), (c_mx, c_my, c_cov),
                        f"chunked covariance {edge}", plain_fn=moments.weighted_covariance,
                        truth_fn=moments.weighted_covariance,
                        taking=orders, placement=placement)
