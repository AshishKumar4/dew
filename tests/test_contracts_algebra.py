"""The algebra engine's custom derivatives each hold their contract (docs/design/testing.md, derivative
contracts): every order finite wherever its float64 truth is a normal float32 and its plain float32 formula is
finite."""

import jax.numpy as jnp
import pytest
from contract_support import EDGES, assert_contract, normal_draw

from dew.algebra import means


def _plain(outputs, totals, anchors):
    """The convex mean as written: each state's mass at the joined anchor, its outputs weighed by it."""
    masses = totals * jnp.exp(anchors - jnp.max(anchors, axis=0, keepdims=True))
    return jnp.sum(masses[..., None] * outputs, axis=0) / jnp.sum(masses, axis=0)[..., None]


def _merged(outputs, totals, anchors):
    """The merge of finished states the contract holds, one function at every edge, so the cell's programs
    compiled at one edge serve the others (`contract_support._mesh_program`)."""
    return means.merged_output(outputs, totals, anchors, means.Units(), 0)


ORDERS = {"its gradient's gradient": ("grad of grad",),
          "its forward derivatives of the gradient": ("jvp of grad", "second jvp", "jvp of grad, remat",
                                                      "second jvp, remat"),
          "its first orders, nested and scanned": ("primal", "jvp", "vjp", "jvp of grad, nested jit",
                                                  "jvp of grad, scanned")}
"""The contract's orders (`contract_support.orders`) in three cells, each about a third of its compiling,
every edge in each (rows 9806, 9807, 10132, 10163: a cell an edge compiled every order cold, three times a
train, each past the train's bound; an order's programs are alike at every edge, so a cell compiles each
once)."""


@pytest.mark.parametrize("taken", ORDERS)
def test_the_convex_mean_holds_every_order(taken):
    """The merge of finished states (`means.merged_output`, its derivative the law's own over scaled values,
    `dew.algebra.scaled`'s `applied` and `_split` within it): every order, forward and reverse, under a nested
    jit and a scan, in the outputs, the totals and the anchors, finite wherever its own float64 run is a
    normal float32 and the plain mean is, its outputs and totals at each edge (`EDGES`): the law runs in
    float64 too, its construct of its operands' format. The orders `taken` names (`ORDERS`), at every edge."""
    outputs, totals, anchors = normal_draw(6096, (3, 4, 8), (3, 4), (3, 4))
    along = normal_draw(6097, (3, 4, 8), (3, 4), (3, 4))
    (cotangent,) = normal_draw(6098, (4, 8))
    for edge, scale in EDGES.items():
        primals = (outputs * scale, (jnp.abs(totals) + 0.25) * scale, anchors)
        direction = (along[0] * scale, along[1] * scale, along[2])
        assert_contract(_merged, primals, direction, cotangent, f"the convex mean {edge}", plain_fn=_plain,
                        taking=ORDERS[taken])
