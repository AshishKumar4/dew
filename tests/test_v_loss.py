"""The Cosine preset's training loss against google-research's diffusion_distillation.

tests/fixtures/v_loss/loss.npz holds `Model.training_losses` of
diffusion_distillation/dpm.py (tools/v_loss_reference.py): a v-prediction
model under its `constant` weighting, the squared error of the x it implies,
in discrete time over improved-diffusion's 1000-entry cosine table, around a
stand-in network v = a * z + c that ignores time. Dew's `Cosine` preset meets
it as `DiffusionObjective.loss` composes the pieces: the fixture's index is
Dew's time, the forward process and v target, and the P2 weight 1 / (1 + SNR)
that turns the v error into the x error. Dew's L2 is half the squared error,
so twice Dew's per-element loss, averaged over each example, is theirs. The
per-example loss and the gradient of the batch mean in the network's output
are held to the float64 rule.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from reference_error import assert_as_exact_as_the_reference

from dew.diffusion.presets import Cosine
from dew.diffusion.schedules import expand
from dew.diffusion.transforms import broadcast_rates

REFERENCE = dict(np.load(Path(__file__).parent / "fixtures" / "v_loss" / "loss.npz"))


def per_example(offset):
    """Twice Dew's per-element loss on the fixture's draws, averaged over each example."""
    process = Cosine()()
    schedule, prediction = process.schedule, process.prediction
    images, noise = jnp.asarray(REFERENCE["images"]), jnp.asarray(REFERENCE["noise"])
    t = jnp.asarray(REFERENCE["indices"])
    rates = broadcast_rates(schedule, t, images)
    noisy, c_in, target = prediction.forward_diffusion(images, noise, rates)
    raw = jnp.asarray(REFERENCE["a"]) * (noisy * c_in) + jnp.asarray(REFERENCE["c"]) + offset
    loss = optax.l2_loss(prediction.pred_transform(noisy, raw, rates, t), target)
    loss = loss * expand(process.weight(t), loss)
    return jnp.mean(2 * loss, axis=(1, 2, 3))


def test_the_cosine_presets_loss_is_diffusion_distillations_x_loss_for_a_v_model():
    offset = jnp.zeros(REFERENCE["images"].shape, jnp.float32)
    assert_as_exact_as_the_reference(per_example(offset), REFERENCE["loss"], REFERENCE["loss_f64"], "loss")
    gradient = jax.grad(lambda o: jnp.mean(per_example(o)))(offset)
    assert_as_exact_as_the_reference(gradient, REFERENCE["grad"], REFERENCE["grad_f64"], "gradient")
