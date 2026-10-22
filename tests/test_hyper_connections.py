"""Manifold-constrained hyper-connections on par with transformers 5.16.1, run live.

The reference is `Glm5NextTextHyperConnection` and the decoder layer's mixing
(modeling_glm5_next.py:267-295, 1316-1318), and `DeepseekV4HyperHead` for the
weighted head (modeling_deepseek_v4.py:958-962); the two references share the
mapping token for token (modeling_deepseek_v4.py:915-943). Each runs in
float32 and, as the truth, in float64 with its float32 pins widened, over drawn
streams, block outputs and parameters, and Dew's module is on par with the
float32 reference from that truth (check 1, tests/reference_error.py): the
mapping's outputs, the mix, and the gradient into the streams and every
parameter.
"""

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import torch
from reference_error import assert_as_exact

from dew.nn.hyper_connections import (
    HyperConnection,
    HyperConnections,
    HyperHead,
    collapse_streams,
    expand_streams,
    mix_streams,
)

H, D, B, S = 3, 8, 2, 5
SPEC = HyperConnections(hc_mult=H, hc_eps=1e-6, hc_sinkhorn_iters=5)
SEEDS = {"site": 256, "head": 8, "mean": 16}
"""Check 1's declared seeds for each site of the file, sized by `seeds_for` on armada."""


def site_draw(seed):
    """Streams, a block output, the site's parameters and a cotangent, drawn from `seed`."""
    rng = np.random.default_rng(seed)
    params = {"fn": rng.standard_normal(((2 + H) * H, H * D)) * 0.3,
              "base": rng.standard_normal((2 + H) * H) * 0.5, "scale": 1 + rng.random(3)}
    return (rng.standard_normal((B, S, H, D)), rng.standard_normal((B, S, D)), params,
            rng.standard_normal((B, S, H, D)))


def reference_site(wide: bool):
    """The reference site and mix as a function of a draw, in float32, or in float64 where `wide`: the
    collapsed input, the mix, and the gradient of `sum(mix * cotangent)` into the streams and each
    parameter."""
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperConnection

    from tools.diffusers_wan_reference import float64

    dtype = torch.float64 if wide else torch.float32

    def run(streams, output, params, cotangent):
        with float64() if wide else contextlib.nullcontext():
            config = Glm5NextTextConfig(hidden_size=D, hc_mult=H, hc_sinkhorn_iters=SPEC.hc_sinkhorn_iters,
                                        hc_eps=SPEC.hc_eps, rms_norm_eps=1e-5)
            module = Glm5NextTextHyperConnection(config).to(dtype)
            module.load_state_dict({name: torch.tensor(value, dtype=dtype)
                                    for name, value in params.items()})
            hidden = torch.tensor(streams, dtype=dtype, requires_grad=True)
            post, comb, collapsed = module(hidden)
            mixed = post.unsqueeze(-1) * torch.tensor(output, dtype=dtype).unsqueeze(-2) + torch.matmul(
                comb.transpose(-1, -2), hidden)
            grads = torch.autograd.grad((mixed * torch.tensor(cotangent, dtype=dtype)).sum(),
                                        [hidden, module.fn, module.base, module.scale])
        return {"collapsed": collapsed.detach().numpy(), "mixed": mixed.detach().numpy(),
                "grad": dict(zip(("streams", "fn", "base", "scale"), (grad.numpy() for grad in grads),
                                 strict=True))}
    return run


@jax.jit
def dew_site(streams, output, params, cotangent):
    """Dew's site and mix over a draw: what `reference_site` returns."""
    module = HyperConnection(spec=SPEC, emb_features=D, norm_eps=1e-5)

    def loss(streams, params):
        post, comb, collapsed = module.apply({"params": params}, streams)
        mixed = mix_streams(post, comb, output, streams)
        return jnp.sum(mixed * cotangent), (collapsed, mixed)

    (_, (collapsed, mixed)), (into_streams, into_params) = jax.value_and_grad(
        loss, argnums=(0, 1), has_aux=True)(streams, params)
    return {"collapsed": collapsed, "mixed": mixed, "grad": {"streams": into_streams, **into_params}}


def single(streams, output, params, cotangent):
    """A draw as float32 JAX arrays, as Dew computes in."""
    return jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float32), (streams, output, params, cotangent))


def test_the_site_mixes_and_differentiates_as_the_reference():
    """Through the mapping, the Sinkhorn loop and the mix, the outputs and the gradients into the
    streams and every parameter on par with the reference's float32 (check 1)."""
    assert_as_exact(lambda *draw: dew_site(*single(*draw)), reference_site(wide=False), site_draw,
                    "hyper-connection site", seeds=SEEDS["site"], truth_fn=reference_site(wide=True))


def head_draw(seed):
    rng = np.random.default_rng(seed)
    params = {"hc_fn": rng.standard_normal((H, H * D)) * 0.3, "hc_base": rng.standard_normal(H),
              "hc_scale": 1 + rng.random(1)}
    return rng.standard_normal((B, S, H, D)), params


def reference_head(wide: bool):
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperHead

    from tools.diffusers_wan_reference import float64

    dtype = torch.float64 if wide else torch.float32

    def run(streams, params):
        with float64() if wide else contextlib.nullcontext(), torch.no_grad():
            config = DeepseekV4Config(hidden_size=D, hc_mult=H, hc_eps=SPEC.hc_eps, rms_norm_eps=1e-5)
            head = DeepseekV4HyperHead(config).to(dtype)
            head.load_state_dict({name: torch.tensor(value, dtype=dtype) for name, value in params.items()})
            return head(torch.tensor(streams, dtype=dtype)).numpy()
    return run


def test_the_weighted_head_collapses_as_the_reference():
    """The weighted head on par with `DeepseekV4HyperHead`'s float32 (check 1)."""
    head = HyperHead(spec=SPEC, emb_features=D, norm_eps=1e-5)

    @jax.jit
    def dew(streams, params):
        return head.apply({"params": params}, streams)

    assert_as_exact(lambda streams, params: dew(*jax.tree.map(lambda x: jnp.asarray(x, jnp.float32),
                                                              (streams, params))),
                    reference_head(wide=False), head_draw, "weighted hyper head", seeds=SEEDS["head"],
                    truth_fn=reference_head(wide=True))


def test_the_mean_head_is_the_mean_and_expansion_repeats_a_stream():
    """Without a weighted head the streams collapse to their mean, on par with NumPy's float32 mean
    from the float64 one (check 1); expanding a stream repeats it, bit for bit."""
    def draw(seed):
        return (np.random.default_rng(seed).standard_normal((B, S, H, D)),)

    assert_as_exact(lambda streams: collapse_streams(jnp.asarray(streams, jnp.float32), None),
                    lambda streams: np.asarray(streams, np.float32).mean(2), draw, "mean hyper head",
                    seeds=SEEDS["mean"], truth_fn=lambda streams: np.asarray(streams, np.float64).mean(2))
    first = jnp.asarray(draw(0)[0][:, :, 0], jnp.float32)
    assert np.array_equal(expand_streams(first, H), np.repeat(np.asarray(first)[:, :, None], H, axis=2))
