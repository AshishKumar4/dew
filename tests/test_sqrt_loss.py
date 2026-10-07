"""The Sqrt preset's training loss against Diffusion-LM's own.

tests/fixtures/sqrt_loss/loss.npz holds `GaussianDiffusion.training_losses`
of Diffusion-LM's gaussian_diffusion.py (tools/sqrt_loss_reference.py): an
x_0 model under the plain MSE loss on its 2000-step sqrt table, around a
stand-in network x_0 = a * x_t + c that ignores time. Dew's `Sqrt` preset
meets it as `DiffusionObjective.loss` composes the pieces: its continuous
time at Diffusion-LM's step k of T is (k + 1) / T, the forward process, the
x_0 target and the weight of one. Dew's L2 is half the squared error, so
twice Dew's per-element loss, averaged over each example, is theirs. The
per-example loss and the gradient of the batch mean in the network's output
are held to the float64 rule.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from reference_error import assert_as_exact_as_the_reference

from dew.diffusion.presets import Sqrt
from dew.diffusion.schedules import expand
from dew.diffusion.transforms import broadcast_rates

REFERENCE = dict(np.load(Path(__file__).parent / "fixtures" / "sqrt_loss" / "loss.npz"))
STEPS = 2000


def nhwc(array):
    return jnp.asarray(np.transpose(array, (0, 2, 3, 1)))


def per_example(offset):
    """Twice Dew's per-element loss on the fixture's draws, averaged over each example."""
    process = Sqrt()()
    schedule, prediction = process.schedule, process.prediction
    images, noise = nhwc(REFERENCE["images"]), nhwc(REFERENCE["noise"])
    t = jnp.asarray((REFERENCE["indices"] + 1) / STEPS, jnp.float32)
    rates = broadcast_rates(schedule, t, images)
    noisy, c_in, target = prediction.forward_diffusion(images, noise, rates)
    a, c = (jnp.asarray(np.transpose(REFERENCE[name], (1, 2, 0))) for name in ("a", "c"))
    raw = a * (noisy * c_in) + c + offset
    loss = optax.l2_loss(prediction.pred_transform(noisy, raw, rates, t), target)
    loss = loss * expand(process.weight(t), loss)
    return jnp.mean(2 * loss, axis=(1, 2, 3))


def test_the_sqrt_presets_loss_is_diffusion_lms_x0_mse():
    offset = jnp.zeros(nhwc(REFERENCE["images"]).shape, jnp.float32)
    assert_as_exact_as_the_reference(per_example(offset), REFERENCE["loss"], REFERENCE["loss_f64"], "loss")
    gradient = jax.grad(lambda o: jnp.mean(per_example(o)))(offset)
    assert_as_exact_as_the_reference(jnp.transpose(gradient, (0, 3, 1, 2)), REFERENCE["grad"],
                                     REFERENCE["grad_f64"], "gradient")
