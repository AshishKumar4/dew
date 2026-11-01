"""TREAD's token routing in `SimpleDiT` against CompVis/tread's `Router`.

The gather and scatter match the official router's on its own draw
(`tools/tread_reference.py`); the routed forward is then rebuilt block by
block from the bound model with them, on the indices the model drew.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.backbones.dit import SimpleDiT, scatter_tokens
from dew.nn.dit import ROPE_THETA, rope_for_scan
from dew.nn.rope import rotary_freqs
from dew.nn.scatter import gather_positions

ROUTER = np.load(Path(__file__).resolve().parent / "fixtures" / "tread" / "router.npz")
ROUTES = ((0.5, 1, 2), (0.25, 3, 3))


def test_the_gather_and_scatter_are_the_routers():
    """Its draw keeps `tokens - int(tokens * ratio)`, which `kept_tokens` keeps too."""
    tokens, kept = jnp.asarray(ROUTER["tokens"]), jnp.asarray(ROUTER["kept"])
    assert kept.shape[1] == 16 - int(16 * 0.3)
    np.testing.assert_array_equal(np.asarray(gather_positions(tokens, kept)), ROUTER["gathered"])
    np.testing.assert_array_equal(np.asarray(scatter_tokens(tokens, kept, jnp.asarray(ROUTER["processed"]))),
                                  ROUTER["scattered"])


def model(routes=ROUTES, **fields):
    return SimpleDiT(patch_size=2, emb_features=16, num_layers=5, num_heads=2, mlp_ratio=1,
                     routes=routes, **fields)


def drawn(network, x, t):
    """Variables with every leaf drawn, so adaLN-Zero's zeroed gates and
    output let each block reach the output."""
    variables = network.init(jax.random.PRNGKey(1), x, t)
    leaves, tree = jax.tree.flatten(variables)
    keys = jax.random.split(jax.random.PRNGKey(4), len(leaves))
    return jax.tree.unflatten(
        tree, [0.3 * jax.random.normal(key, leaf.shape) for key, leaf in zip(keys, leaves, strict=True)]
    )


def inputs():
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 8, 8, 3))
    return x, jnp.asarray([0.1, 0.5, 0.9])


def test_a_routed_forward_skips_each_span_on_the_drawn_tokens():
    network = model()
    x, t = inputs()
    variables = drawn(network, x, t)
    rngs = {"dropout": jax.random.PRNGKey(2)}
    output, sown = network.apply(variables, x, t, train=True, rngs=rngs, mutable=["intermediates"])

    bound = network.bind(variables, rngs=rngs)
    tokens, order = bound.embed(x)
    condition = bound.conditioning(t, None)
    full = rope_for_scan(tokens, 8, "raster")
    rotation, held, kept = full, None, None
    for index, block in enumerate(bound.blocks):
        if index in (1, 3):
            kept = sown["intermediates"][f"route_{index}"][0]
            assert kept.shape[1] == 16 - int(16 * {1: 0.5, 3: 0.25}[index])
            held, tokens = tokens, gather_positions(tokens, kept)
            rotation = rotary_freqs(kept, 8, ROPE_THETA, dtype=jnp.float32)
        tokens = block(tokens, condition, rotation, train=True)
        if index in (2, 3):
            tokens, rotation = scatter_tokens(held, kept, tokens), full
    np.testing.assert_array_equal(np.asarray(output), np.asarray(bound.output(tokens, order, 8, 8)))
    # Routing is structural (other tokens skip the spans), so any routed output is another computation.
    plain = model(()).apply(variables, x, t, train=True, rngs=rngs)
    assert not np.array_equal(np.asarray(output), np.asarray(plain))


def test_sampling_runs_every_token_through_every_block():
    x, t = inputs()
    variables = drawn(model(), x, t)
    np.testing.assert_array_equal(np.asarray(model().apply(variables, x, t)),
                                  np.asarray(model(()).apply(variables, x, t)))
