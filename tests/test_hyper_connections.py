"""Manifold-constrained hyper-connections against a float64 NumPy oracle.

The oracle below is `Glm5NextTextHyperConnection.forward` and the decoder
layer's mixing (modeling_glm5_next.py:267-295, 1316-1318) transcribed
line for line into NumPy at float64, with `DeepseekV4HyperHead.forward`
(modeling_deepseek_v4.py:958-962) for the weighted head; the two references
share the mapping token for token (modeling_deepseek_v4.py:915-943).
Nothing here imports torch, so the oracle holds the module to the math and
not to another implementation's rounding.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.hyper_connections import (
    HyperConnection,
    HyperConnections,
    HyperHead,
    collapse_streams,
    expand_streams,
    mix_streams,
    sinkhorn,
)

BOUND = 1e-4
H, D, B, S = 3, 8, 2, 5


def oracle_mapping(streams, fn, base, scale, eps, iters, norm_eps):
    """(post, comb, collapsed) of one site, modeling_glm5_next.py:277-295."""
    hc = streams.shape[2]
    flat = streams.reshape(*streams.shape[:2], hc * streams.shape[-1])
    flat = flat / np.sqrt(np.mean(np.square(flat), axis=-1, keepdims=True) + norm_eps)
    mixes = flat @ fn.T
    pre_w, post_w, comb_w = np.split(mixes, [hc, 2 * hc], axis=-1)
    pre_b, post_b, comb_b = np.split(base, [hc, 2 * hc])
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    pre = sigmoid(pre_w * scale[0] + pre_b) + eps
    post = 2 * sigmoid(post_w * scale[1] + post_b)
    logits = comb_w.reshape(*comb_w.shape[:-1], hc, hc) * scale[2] + comb_b.reshape(hc, hc)
    exp = np.exp(logits - logits.max(-1, keepdims=True))
    comb = exp / exp.sum(-1, keepdims=True) + eps
    comb = comb / (comb.sum(-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdims=True) + eps)
        comb = comb / (comb.sum(-2, keepdims=True) + eps)
    collapsed = (pre[..., None] * streams).sum(2)
    return post, comb, collapsed


def oracle_mix(post, comb, output, streams):
    """`post * output + comb^T @ residual`, modeling_glm5_next.py:1316-1318."""
    return post[..., None] * output[..., None, :] + np.swapaxes(comb, -1, -2) @ streams


def oracle_weighted_head(streams, fn, base, scale, eps, norm_eps):
    """modeling_deepseek_v4.py:958-962."""
    hc = streams.shape[2]
    flat = streams.reshape(*streams.shape[:2], hc * streams.shape[-1])
    flat = flat / np.sqrt(np.mean(np.square(flat), axis=-1, keepdims=True) + norm_eps)
    pre = 1 / (1 + np.exp(-(flat @ fn.T * scale + base))) + eps
    return (pre[..., None] * streams).sum(2)


def scaled(actual, wanted) -> float:
    return float(np.max(np.abs(np.asarray(actual, np.float64) - wanted)) / np.max(np.abs(wanted)))


@pytest.fixture
def case():
    rng = np.random.RandomState(0)
    spec = HyperConnections(hc_mult=H, hc_eps=1e-6, hc_sinkhorn_iters=5)
    streams = rng.randn(B, S, H, D)
    output = rng.randn(B, S, D)
    params = {"fn": rng.randn((2 + H) * H, H * D) * 0.3, "base": rng.randn((2 + H) * H) * 0.5,
              "scale": 1 + rng.rand(3)}
    return spec, streams, output, params


def site(spec, params, streams, output):
    """The module's site over `streams` mixed with `output`, in fp32."""
    module = HyperConnection(spec=spec, emb_features=D, norm_eps=1e-5)
    variables = {"params": jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float32), params)}
    outputs = module.apply(variables, jnp.asarray(streams, jnp.float32))
    if not isinstance(outputs, tuple) or len(outputs) != 3:
        raise TypeError("a hyper-connection site returns (post, comb, collapsed)")
    post, comb, collapsed = outputs
    return collapsed, mix_streams(post, comb, jnp.asarray(output, jnp.float32), jnp.asarray(streams, jnp.float32))


def test_the_site_mapping_and_mixing_match_the_oracle(case):
    spec, streams, output, params = case
    post, comb, collapsed = oracle_mapping(streams, params["fn"], params["base"], params["scale"],
                                           spec.hc_eps, spec.hc_sinkhorn_iters, 1e-5)
    wanted = oracle_mix(post, comb, output, streams)

    ours_collapsed, ours = site(spec, params, streams, output)

    assert scaled(ours_collapsed, collapsed) < BOUND
    assert scaled(ours, wanted) < BOUND
    # The last Sinkhorn step normalises the columns, so those sum to one to
    # the floor whatever the iteration count; the rows converge with it.
    assert np.allclose(comb.sum(-2), 1, atol=1e-4)


def test_the_mixing_direction_is_the_references(case):
    """`comb` is applied transposed. The other orientation is a different
    residual mix of the same Sinkhorn output, which the oracle separates."""
    spec, streams, output, params = case
    post, comb, _ = oracle_mapping(streams, params["fn"], params["base"], params["scale"],
                                   spec.hc_eps, spec.hc_sinkhorn_iters, 1e-5)
    other = post[..., None] * output[..., None, :] + comb @ streams

    _, ours = site(spec, params, streams, output)

    assert scaled(ours, other) > 1e-2


def test_the_gradients_match_central_differences_of_the_oracle(case):
    """Through the mapping, the Sinkhorn loop and the mix, into the streams
    and every parameter."""
    spec, streams, output, params = case
    cotangent = np.random.RandomState(1).randn(B, S, H, D)

    def loss_oracle(streams, fn, base, scale):
        post, comb, _ = oracle_mapping(streams, fn, base, scale, spec.hc_eps, spec.hc_sinkhorn_iters, 1e-5)
        return float(np.sum(oracle_mix(post, comb, output, streams) * cotangent))

    def loss_module(streams, params):
        _, mixed = site(spec, params, streams, output)
        return jnp.sum(mixed * jnp.asarray(cotangent, jnp.float32))

    grads = jax.grad(loss_module, argnums=(0, 1))(jnp.asarray(streams, jnp.float32),
                                                   jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float32), params))
    ours = {"streams": grads[0], **grads[1]}
    arguments = {"streams": streams, **params}
    for name, value in arguments.items():
        wanted = np.zeros_like(value)
        step = 1e-6
        for index in np.ndindex(value.shape):
            bumped = {**arguments, name: value.copy()}
            bumped[name][index] += step
            up = loss_oracle(bumped["streams"], bumped["fn"], bumped["base"], bumped["scale"])
            bumped[name][index] -= 2 * step
            down = loss_oracle(bumped["streams"], bumped["fn"], bumped["base"], bumped["scale"])
            wanted[index] = (up - down) / (2 * step)
        assert scaled(ours[name], wanted) < BOUND, name


def test_the_weighted_head_matches_the_oracle_and_the_mean_head_is_the_mean(case):
    spec, streams, _, _ = case
    rng = np.random.RandomState(2)
    params = {"hc_fn": rng.randn(H, H * D) * 0.3, "hc_base": rng.randn(H), "hc_scale": 1 + rng.rand(1)}
    head = HyperHead(spec=spec, emb_features=D, norm_eps=1e-5)
    variables = {"params": jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float32), params)}

    weighted = head.apply(variables, jnp.asarray(streams, jnp.float32))
    wanted = oracle_weighted_head(streams, params["hc_fn"], params["hc_base"], params["hc_scale"],
                                  spec.hc_eps, 1e-5)

    assert scaled(weighted, wanted) < BOUND
    assert scaled(collapse_streams(jnp.asarray(streams, jnp.float32), None), streams.mean(2)) < BOUND
    first = jnp.asarray(streams[:, :, 0], jnp.float32)
    assert np.array_equal(expand_streams(first, H), np.repeat(np.asarray(first)[:, :, None], H, axis=2))


def test_sinkhorn_compiles_its_repeats_as_one_loop():
    """Sinkhorn's repeats lower to a loop, so the gradient program, and the
    time XLA takes to compile it, does not grow with the iteration count:
    unrolled, 20 repeats took an L4 71 s to compile for one site's gradient,
    where the loop takes 0.5 s."""
    comb = jnp.full((2, 4, 4), 0.25, jnp.float32)

    def program(iters):
        gradient = jax.grad(lambda comb: jnp.sum(sinkhorn(comb, iters, 1e-6) ** 2))
        return jax.jit(gradient).lower(comb).as_text()

    twenty, forty = program(20), program(40)
    assert "stablehlo.while" in twenty
    assert twenty.count("stablehlo.divide") == forty.count("stablehlo.divide")


def test_the_record_refuses_what_the_references_cannot_build():
    with pytest.raises(ValueError, match="hc_mult"):
        HyperConnections(hc_mult=0)
    with pytest.raises(ValueError, match="hc_sinkhorn_iters"):
        HyperConnections(hc_sinkhorn_iters=0)
    with pytest.raises(ValueError, match="head"):
        HyperConnections(head="sum")
