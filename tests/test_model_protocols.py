"""Model capabilities (`dew.nn.protocols`), proven on the models themselves.

A family comes from Dew's model registry, built at a tiny size by
test_precision_policy.py's `build_model` and `tiny_inputs`, which every
registered family needs for that file's registry-wide parametrization to
run. A test here asserts what a capability does, not which classes have it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_precision_policy import build_model, tiny_inputs

from dew.nn.autoencoders.dc_ae import DCAE, DCAutoencoder
from dew.nn.autoencoders.flux2 import Flux2Autoencoder
from dew.nn.autoencoders.kl import AutoencoderKL, posterior_latent
from dew.nn.autoencoders.sd_vae import StableDiffusionVAE
from dew.nn.protocols import IntervalModel, RequiresText, TimeScaled
from dew.registry import models

# Autoencoders and denoisers: the KL posterior an objective trains an
# autoencoder through, the text a denoiser cannot run without, and the
# interval and time scale it reads its time in.


def kl_module(dtype=jnp.float32) -> AutoencoderKL:
    return AutoencoderKL(channels=(8, 16), latent_channels=4, blocks_per_level=1, norm_groups=4, dtype=dtype)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["float32", "bfloat16"])
def test_a_kl_autoencoder_gives_the_posterior_end_to_end_tuning_trains_through(dtype):
    """REPA-E's step (objective.py's `_end_to_end_latents`) read the
    posterior and decoded its draw by applying the autoencoder's module
    itself. `moments` and `decode_raw` give the same arrays and the same
    gradients, bit for bit, and `encode_batch`'s latent is a draw from that
    posterior, which is what lets the step renormalize the draw itself."""
    module = kl_module(dtype)
    params = module.init(jax.random.PRNGKey(5), jnp.zeros((1, 8, 8, 3)))["params"]
    autoencoder = StableDiffusionVAE(model=module, params=params, dtype=dtype, latent_shift=0.1,
                                     latent_scale=0.8)
    images = jax.random.uniform(jax.random.PRNGKey(1), (8, 8, 8, 3), minval=-1.0, maxval=1.0)
    key = jax.random.PRNGKey(2)

    def module_applied(tree):
        weights = {"params": tree}
        moments = module.apply(weights, images, method=module.moments)
        return moments, module.apply(weights, posterior_latent(moments, key), method=module.decode)

    def asked(tree):
        moments = autoencoder.moments(tree, images)
        return moments, autoencoder.decode_raw(tree, posterior_latent(moments, key))

    def loss(read):
        return lambda tree: sum(jnp.sum(jnp.square(part.astype(jnp.float32))) for part in read(tree))

    for transform in (lambda read: read, lambda read: jax.grad(loss(read))):
        got, expected = (jax.jit(transform(read))(params) for read in (asked, module_applied))
        for leaf, reference in zip(jax.tree.leaves(got), jax.tree.leaves(expected), strict=True):
            np.testing.assert_array_equal(leaf, reference)
    drawn = jax.jit(lambda tree: posterior_latent(autoencoder.moments(tree, images), key))(params)
    np.testing.assert_array_equal(autoencoder.encode_batch(params, images, key), drawn)


@pytest.mark.parametrize("autoencoder", [
    Flux2Autoencoder(model=kl_module(), params={}, mean=np.zeros(16), variance=np.ones(16), epsilon=1e-4),
    DCAutoencoder(model=DCAE(), params={}, latent_scale=1.0),
], ids=lambda autoencoder: type(autoencoder).__name__)
def test_an_autoencoder_whose_latent_is_no_kl_draw_refuses_by_name(autoencoder):
    """Two autoencoders that inherit the base refusal. FLUX.2's latent folds
    2x2 blocks of its AutoencoderKL's draw and normalizes them by batch-norm
    statistics, and a DC-AE encodes without a posterior: neither latent is a
    draw an objective may renormalize, so each refuses the posterior and its
    decode, naming itself."""
    name = type(autoencoder).__name__
    with pytest.raises(TypeError, match=name):
        autoencoder.moments(autoencoder.params, jnp.zeros((1, 8, 8, 3)))
    with pytest.raises(TypeError, match=name):
        autoencoder.decode_raw(autoencoder.params, jnp.zeros((1, 2, 2, 4)))


def denoiser(architecture: str, rng):
    """The registered `architecture` at a tiny size in float32, its sample,
    time and the rest of its tiny inputs; None where those are not a batch
    of samples and one time per row, a denoiser's."""
    inputs = tiny_inputs(architecture, rng)
    (sample, *args), conditions = inputs if isinstance(inputs[0], tuple) else (inputs, {})
    if not args or sample.ndim < 4 or jnp.shape(args[0]) != sample.shape[:1]:
        return None
    return build_model(architecture, "float32"), sample, args[0], tuple(args[1:]), conditions


def perturbed(variables, rng):
    """`variables` with noise on every parameter, so a zero-initialized head
    hides nothing a change of input does."""
    leaves, tree = jax.tree.flatten(variables["params"])
    noisy = [leaf + 0.1 * jax.random.normal(key, leaf.shape, leaf.dtype)
             for leaf, key in zip(leaves, jax.random.split(rng, len(leaves)), strict=True)]
    return {**variables, "params": jax.tree.unflatten(tree, noisy)}


@pytest.mark.parametrize("architecture", [architecture for architecture in sorted(models)
                                          if denoiser(architecture, jax.random.PRNGKey(0))])
def test_a_denoiser_reads_the_text_interval_and_time_it_declares(architecture, rng):
    """What the diffusion recipe relies on, held to every registered denoiser's declarations.

    An unconditional run builds and calls its denoiser on the sample and
    time alone: one declaring `RequiresText` refuses that, naming the keyword
    its text arrives under, and runs on that keyword's text; every other
    runs. An interval process hands its model each step's `duration`: cloned
    with `interval` set, an `IntervalModel` embeds it, a missing one as the
    zero duration, and cloned without refuses one, as every other denoiser
    does. MeanFlow and sCM slow a `TimeScaled` denoiser's time features
    through `time_scale`, which is the time's unit: at 16 on t and d it
    computes bit for bit what it computes at 2 on 8t and 8d, powers of two
    keeping both sides exact.
    """
    model, sample, time, rest, conditions = denoiser(architecture, rng)
    if isinstance(model, RequiresText):
        variables = model.init(rng, sample, time, *rest, **conditions)
        with pytest.raises(TypeError, match=model.text_keyword):
            model.apply(variables, sample, time)
        text = conditions.get(model.text_keyword, rest[0] if rest else None)
        denoised = model.apply(variables, sample, time, **{model.text_keyword: text})
    else:
        denoised = model.apply(model.init(rng, sample, time), sample, time)
    assert denoised.shape == sample.shape and bool(jnp.all(jnp.isfinite(denoised)))

    def built(model, scale=None):
        model = model if scale is None else model.clone(time_scale=scale)
        variables = perturbed(model.init(rng, sample, time, *rest, **conditions), rng)
        return lambda at, **duration: model.apply(variables, sample, at, *rest, **duration, **conditions)

    duration = jnp.full_like(time, 0.25)
    if isinstance(model, IntervalModel):
        with pytest.raises(ValueError, match="duration"):
            built(model.clone(interval=False))(time, duration=duration)
        model = model.clone(interval=True)
        call = built(model)
        np.testing.assert_array_equal(call(time), call(time, duration=jnp.zeros_like(time)))
        assert not np.array_equal(call(time), call(time, duration=duration))
    else:
        with pytest.raises(TypeError, match="duration"):
            built(model)(time, duration=duration)
    if isinstance(model, TimeScaled):
        spanned = isinstance(model, IntervalModel)
        fast, slow = (built(model, scale)(unit * time, **({"duration": unit * duration} if spanned else {}))
                      for scale, unit in ((16.0, 1.0), (2.0, 8.0)))
        assert not np.array_equal(fast, jnp.zeros_like(fast))
        np.testing.assert_array_equal(fast, slow)
