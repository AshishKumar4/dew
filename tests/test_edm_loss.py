"""EDM's and EDM2's training losses against NVlabs' own.

tests/fixtures/edm_loss/loss.npz holds NVlabs/edm's `EDMLoss` through
`EDMPrecond` and NVlabs/edm2's `EDM2Loss` through `Precond` with its logvar
head (tools/edm_loss_reference.py), each run on the same images, the same
standard normal behind every training sigma and the same noise, around an
affine stand-in network. Dew's EDM process meets them piece by piece, as
`DiffusionObjective.loss` composes the pieces (its own tests hold that
composition): the training time is the standard normal draw and its sigma
`schedule.sigmas(t)`, the forward process, the preconditioned read-out, the
model time the network and the uncertainty head are conditioned on, the
lambda weight and the learned-uncertainty term. Dew's L2 is half of NVlabs'
squared error, so twice Dew's per-element loss is theirs. The per-element
values, the gradient of their mean in the network's output (the backward
through the read-out, weight and uncertainty; the stand-in's own weights
take linear maps of it) and in the uncertainty head's weight are held to
the float64 rule.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.diffusion.presets import EDM
from dew.diffusion.schedules import expand
from dew.diffusion.transforms import broadcast_rates
from dew.nn.mp import MP_KERNEL, Uncertainty

REFERENCE = dict(np.load(Path(__file__).parent / "fixtures" / "edm_loss" / "loss.npz"))


def nhwc(array):
    return jnp.asarray(np.transpose(array, (0, 2, 3, 1)))


def per_element(process, weights, uncertainty=None, offset=0.0):
    """Twice Dew's per-element loss on the fixture's draws, NCHW like NVlabs',
    with `offset` added to the network's output."""
    schedule, prediction = process.schedule, process.prediction
    images, noise = nhwc(REFERENCE["images"]), nhwc(REFERENCE["noise"])
    t = jnp.asarray(REFERENCE["normals"])
    rates = broadcast_rates(schedule, t, images)
    noisy, c_in, target = prediction.forward_diffusion(images, noise, rates)
    time = schedule.model_time(t)
    raw = weights["a"] * (noisy * c_in) + weights["b"] * expand(time, noisy) + weights["c"] + offset
    loss = optax.l2_loss(prediction.pred_transform(noisy, raw, rates, t), target)
    loss = loss * expand(process.weight(t), loss)
    if uncertainty is not None:
        logvar = expand(uncertainty.apply(weights["head"], time), loss)
        loss = loss * jnp.exp(-logvar) + logvar / 2
    return jnp.transpose(2 * loss, (0, 3, 1, 2))


def stand_in() -> dict:
    return {name: jnp.asarray(REFERENCE[name]) for name in ("a", "b", "c")}


@pytest.mark.parametrize("regime,prefix,model", [("pixel", "edm", "model"), ("latent", "edm2", "unet")])
def test_the_training_sigmas_are_nvlabs(regime, prefix, model):
    """EDM's exp(N(-1.2, 1.2^2)) for pixels and EDM2's exp(N(-0.4, 1)) for
    latents, on the same standard normals NVlabs' loss drew."""
    schedule = EDM(regime=regime)().schedule
    sigma = schedule.sigmas(jnp.asarray(REFERENCE["normals"]))
    assert_as_exact_as_the_reference(sigma, REFERENCE[f"{prefix}/sigma"], REFERENCE[f"{prefix}/sigma_f64"],
                                     f"{prefix} sigmas")


def test_the_pixel_loss_and_its_gradient_are_edmloss():
    """NVlabs/edm's EDMLoss at its defaults (P_mean -1.2, P_std 1.2,
    sigma_data 0.5) through EDMPrecond, against `EDM(regime="pixel")`."""
    process = EDM(regime="pixel")()
    loss = per_element(process, stand_in())
    output = jax.grad(lambda offset: jnp.mean(per_element(process, stand_in(), offset=offset)))(
        jnp.zeros(nhwc(REFERENCE["images"]).shape))
    assert_as_exact_as_the_reference(loss, REFERENCE["edm/loss"], REFERENCE["edm/loss_f64"], "EDM loss")
    assert_as_exact_as_the_reference(jnp.transpose(output, (0, 3, 1, 2)),
                                     REFERENCE["edm/grad/network_output"],
                                     REFERENCE["edm/grad/network_output_f64"], "EDM gradient in F")


def test_the_latent_loss_with_learned_uncertainty_and_its_gradient_are_edm2loss():
    """NVlabs/edm2's EDM2Loss at its defaults (P_mean -0.4, P_std 1.0) through
    Precond's logvar head, u(sigma) of Eq. 21, against
    `EDM(regime="latent")` with `dew.nn.mp.Uncertainty` holding the same
    Fourier features and weight. The gradient reaches the head's weight as
    it reaches the network's."""
    process = EDM(regime="latent")()
    head = Uncertainty(channels=REFERENCE["logvar/freqs"].shape[0])
    weights = {**stand_in(), "head": {
        "params": {"linear": {MP_KERNEL: jnp.asarray(REFERENCE["logvar/weight"]).T}},
        "constants": {"fourier": {"frequencies": jnp.asarray(REFERENCE["logvar/freqs"]),
                                  "phases": jnp.asarray(REFERENCE["logvar/phases"])}}}}
    loss = per_element(process, weights, head)

    def mean(weights, offset):
        return jnp.mean(per_element(process, weights, head, offset))

    gradient, output = jax.grad(mean, argnums=(0, 1))(weights, jnp.zeros(nhwc(REFERENCE["images"]).shape))
    assert_as_exact_as_the_reference(loss, REFERENCE["edm2/loss"], REFERENCE["edm2/loss_f64"], "EDM2 loss")
    assert_as_exact_as_the_reference(jnp.transpose(output, (0, 3, 1, 2)),
                                     REFERENCE["edm2/grad/network_output"],
                                     REFERENCE["edm2/grad/network_output_f64"], "EDM2 gradient in F")
    assert_as_exact_as_the_reference(gradient["head"]["params"]["linear"][MP_KERNEL].T,
                                     REFERENCE["edm2/grad/logvar_weight"],
                                     REFERENCE["edm2/grad/logvar_weight_f64"],
                                     "EDM2 gradient in the logvar head")
