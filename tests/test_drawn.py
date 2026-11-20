"""How a draw outside any trace is kept and read again (`dew.objectives.drawn`): one trace for each held tree,
keyed by the tree as it is (`held_signature`)."""

import enum

import jax
import jax.numpy as jnp
from flax import struct


def test_a_kept_draw_over_trees_apart_by_an_empty_collection_draws_each_tree_its_own():
    """Two held trees apart only by an empty collection (a model's `batch_stats` it holds none of yet) are
    two kept draws (`dew.objectives.drawn.held_signature`): each call builds its own tree, the empty
    collection kept, where keyed by its leaves' paths alone the second replayed the first's (row 9548)."""
    from dew.objectives.drawn import WHOLE, drawn_once

    def doubled(tree, key):
        return jax.tree.map(lambda leaf: 2 * leaf, tree)

    variables = {"params": {"w": jnp.full(3, 1.0)}}
    key = jax.random.key(0)
    assert sorted(drawn_once(doubled, variables, key, WHOLE, kept="stats")) == ["params"]
    drawn = drawn_once(doubled, {**variables, "batch_stats": {}}, key, WHOLE, kept="stats")
    assert sorted(drawn) == ["batch_stats", "params"] and drawn["batch_stats"] == {}, drawn


class Mode(enum.Enum):
    """Two modes a node may hold as its static data."""

    FIRST = 1
    SECOND = 2


@struct.dataclass
class Tagged:
    """An array beside a mode its node holds as static data, not a leaf."""

    value: jax.Array
    mode: Mode = struct.field(pytree_node=False)


def test_a_kept_draw_over_trees_apart_by_a_nodes_enum_draws_each_tree_its_own():
    """Two held trees apart only by an enum member a node holds as static data are two kept draws: each call
    builds its own tree, its own member kept (row 9633: every member of one enum keyed alike, by its kind,
    and the second call replayed the first's)."""
    from dew.objectives.drawn import WHOLE, drawn_once

    def doubled(tree, key):
        return jax.tree.map(lambda leaf: 2 * leaf, tree)

    key = jax.random.key(0)
    assert drawn_once(doubled, Tagged(jnp.ones(2), Mode.FIRST), key, WHOLE, kept="modes").mode is Mode.FIRST
    assert drawn_once(doubled, Tagged(jnp.ones(2), Mode.SECOND), key, WHOLE, kept="modes").mode is Mode.SECOND
