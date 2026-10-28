"""EDM's and EDM2's training losses against NVlabs' own.

tests/fixtures/edm_loss/seeds.npz holds, over seeds, NVlabs/edm's `EDMLoss` through
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
take linear maps of it) and in the uncertainty head's weight are held on
par with NVlabs' own by check 1, over the seeds the recording holds
(tests/fixtures/edm_loss/seeds.npz, each seed's inputs drawn again here).
"""
import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from reference_error import as_given, assert_as_exact, declared_seeds, indexed_recording, recording
from tool_support import run_from

from dew.diffusion.presets import EDM
from dew.diffusion.schedules import expand
from dew.diffusion.transforms import broadcast_rates
from dew.nn.mp import MP_KERNEL, Uncertainty

SEEDS = Path(__file__).parent / "fixtures" / "edm_loss" / "seeds.npz"


def nhwc(array):
    return jnp.asarray(np.transpose(array, (0, 2, 3, 1)))


def per_element(process, inputs, weights, uncertainty=None, offset=0.0):
    """Twice Dew's per-element loss on a draw's inputs, NCHW like NVlabs', with `offset` added to the
    network's output."""
    schedule, prediction = process.schedule, process.prediction
    images, noise = nhwc(inputs["images"]), nhwc(inputs["noise"])
    t = jnp.asarray(inputs["normals"])
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


@functools.cache
def generator():
    """tools/edm_loss_reference.py, whose seeded draw the recorded inputs are drawn again by, loaded as
    a script from its directory."""
    with run_from("edm_loss_reference", "tools") as module:
        return module


def recorded_site(site: str):
    """`site`'s draw, recorded reference and truth over its declared seeds (`indexed_recording` of a
    summarized recording), each seed's inputs drawn again (tools/edm_loss_reference.py's `draw`) and its
    truth Dew's loss in float64 (`edm_loss`, `edm2_loss`), held to the recording's sketch of NVlabs' own
    float64 twin, and the count."""
    seeds = declared_seeds(SEEDS)[site]
    truth_fn = {"edm": edm_loss, "edm2": edm2_loss}[site]
    return (*indexed_recording(recording(SEEDS), site, seeds, inputs_of=generator().draw, truth_fn=truth_fn),
            seeds)


def edm_loss(inputs):
    """Dew's pixel loss on a draw: its sigmas, per-element loss and the gradient of its mean in F."""
    process = EDM(regime="pixel")()
    weights = {name: jnp.asarray(inputs[name]) for name in ("a", "b", "c")}
    output = jax.grad(lambda offset: jnp.mean(per_element(process, inputs, weights, offset=offset)))(
        jnp.zeros(nhwc(inputs["images"]).shape))
    return {"sigma": process.schedule.sigmas(jnp.asarray(inputs["normals"]))[:, None],
            "loss": per_element(process, inputs, weights),
            "grad/network_output": jnp.transpose(output, (0, 3, 1, 2))}


def edm2_loss(inputs):
    """Dew's latent loss with learned uncertainty on a draw, its gradient in F and in the head."""
    process = EDM(regime="latent")()
    head = Uncertainty(channels=inputs["logvar/freqs"].shape[0])
    weights = {**{name: jnp.asarray(inputs[name]) for name in ("a", "b", "c")}, "head": {
        "params": {"linear": {MP_KERNEL: jnp.asarray(inputs["logvar/weight"]).T}},
        "constants": {"fourier": {"frequencies": jnp.asarray(inputs["logvar/freqs"]),
                                  "phases": jnp.asarray(inputs["logvar/phases"])}}}}

    def mean(weights, offset):
        return jnp.mean(per_element(process, inputs, weights, head, offset))

    gradient, output = jax.grad(mean, argnums=(0, 1))(weights, jnp.zeros(nhwc(inputs["images"]).shape))
    return {"sigma": process.schedule.sigmas(jnp.asarray(inputs["normals"]))[:, None],
            "loss": per_element(process, inputs, weights, head),
            "grad/network_output": jnp.transpose(output, (0, 3, 1, 2)),
            "grad/logvar_weight": gradient["head"]["params"]["linear"][MP_KERNEL].T}


@pytest.mark.parametrize("regime,site", [("pixel", "edm"), ("latent", "edm2")])
def test_the_training_sigmas_are_nvlabs(regime, site):
    """EDM's exp(N(-1.2, 1.2^2)) for pixels and EDM2's exp(N(-0.4, 1)) for
    latents, on the same standard normals NVlabs' loss drew, on par with
    them over the recorded seeds (check 1)."""
    schedule = EDM(regime=regime)().schedule
    draw, recorded, truth, seeds = recorded_site(site)
    assert_as_exact(lambda index, inputs: {"sigma": schedule.sigmas(jnp.asarray(inputs["normals"]))[:, None]},
                    as_given(lambda index, inputs: {"sigma": recorded(index, inputs)["sigma"]}), draw,
                    f"{site} sigmas", seeds=seeds,
                    truth_fn=as_given(lambda index, inputs: {"sigma": truth(index, inputs)["sigma"]}))


def test_the_pixel_loss_and_its_gradient_are_edmloss():
    """NVlabs/edm's EDMLoss at its defaults (P_mean -1.2, P_std 1.2,
    sigma_data 0.5) through EDMPrecond, against `EDM(regime="pixel")`, over
    the recorded seeds (check 1)."""
    draw, recorded, truth, seeds = recorded_site("edm")
    compiled = jax.jit(edm_loss)
    assert_as_exact(lambda index, inputs: compiled(inputs), recorded, draw, "EDM loss", seeds=seeds,
                    truth_fn=truth)


def test_the_latent_loss_with_learned_uncertainty_and_its_gradient_are_edm2loss():
    """NVlabs/edm2's EDM2Loss at its defaults (P_mean -0.4, P_std 1.0) through
    Precond's logvar head, u(sigma) of Eq. 21, against
    `EDM(regime="latent")` with `dew.nn.mp.Uncertainty` holding the same
    Fourier features and weight, over the recorded seeds (check 1). The
    gradient reaches the head's weight as it reaches the network's."""
    draw, recorded, truth, seeds = recorded_site("edm2")
    compiled = jax.jit(edm2_loss)
    assert_as_exact(lambda index, inputs: compiled(inputs), recorded, draw, "EDM2 loss", seeds=seeds,
                    truth_fn=truth)
