"""JiT: a clean-sample prediction scored in velocity space, and its
bottleneck patch embedding against LTH14/JiT (`tools/jit_reference.py`)."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from reference_error import assert_as_exact_as_the_reference

from dew.diffusion import FlowMatchingScheduler, Process, VelocityLoss, expand, presets
from dew.inputs import Field, InputSpec, unit_range
from dew.nn.dit import PatchEmbedding
from dew.objectives.base import Step, scalar_loss
from dew.objectives.diffusion import DiffusionObjective
from dew.sampling import Euler

EMBED = np.load(Path(__file__).resolve().parent / "fixtures" / "jit" / "bottleneck.npz")


class Clean(nn.Module):
    """A clean-sample predictor with some dependence on the input and time."""

    @nn.compact
    def __call__(self, x, time, train=False):
        return jnp.tanh(x) * 0.6 + expand(time, x) * 1e-3


def test_the_loss_is_the_references_velocity_error():
    """LTH14/JiT's `Denoiser.forward`, written out in its clean-at-one time
    t' = 1 - sigma: z = t' x + (1 - t') e, v = (x - z) / max(1 - t', t_eps),
    v_pred = (net(z) - z) / max(1 - t', t_eps), loss mean((v - v_pred)^2);
    Dew's L2 is half of that."""
    process = presets.JiT()()
    objective = DiffusionObjective(Clean(), process, InputSpec(Field("image", (4, 4, 3))), guidance=None,
                                   sampler=Euler(), steps=2, ema_decay=None)
    params = objective.init(jax.random.PRNGKey(0))
    batch = {"image": np.asarray(jax.random.randint(jax.random.PRNGKey(1), (6, 4, 4, 3), 0, 256), np.uint8)}
    step = Step(step=jnp.asarray(0), key=jax.random.PRNGKey(2), ema=None)
    loss, _ = scalar_loss(objective, params, batch, step)

    _, _, time_key, noise_key, _ = jax.random.split(step.key, 5)
    x = np.asarray(unit_range(batch["image"]), np.float64)
    t = process.schedule.sample_t(time_key, 6)
    sigma = np.asarray(t, np.float64)[:, None, None, None]
    time = np.asarray(process.schedule.model_time(t), np.float64)[:, None, None, None]
    e = np.asarray(jax.random.normal(noise_key, x.shape), np.float64)
    clean_time = 1 - sigma
    z = clean_time * x + (1 - clean_time) * e
    v = (x - z) / np.maximum(1 - clean_time, 0.05)
    net = np.tanh(z) * 0.6 + time * 1e-3
    v_pred = (net - z) / np.maximum(1 - clean_time, 0.05)
    reference = np.mean(np.mean((v - v_pred) ** 2, axis=(1, 2, 3)))
    # Dew runs in float32: a relative 2e-5 is rounding, while a missed
    # clamp or a wrong time direction moves the loss by orders more.
    assert 2 * float(loss) == pytest.approx(reference, rel=2e-5)


def test_the_training_times_are_the_references_logit_normal_mirrored():
    """JiT draws t' = sigmoid(-0.8 + 0.8 n) clean-at-one; Dew's draw on the
    same normals mirrored is 1 - t'."""
    key = jax.random.PRNGKey(3)
    sigma = np.asarray(presets.JiT()().schedule.sample_t(key, 1000), np.float64)
    normals = np.asarray(jax.random.normal(key, (1000,)), np.float64)
    reference = 1 / (1 + np.exp(-(-0.8 + 0.8 * -normals)))
    np.testing.assert_allclose(sigma, 1 - reference, rtol=0, atol=1e-6)


def test_the_velocity_loss_refuses_another_prediction():
    with pytest.raises(ValueError, match="clean-sample prediction"):
        Process(FlowMatchingScheduler(), presets.Flow()().prediction, VelocityLoss()).weight(jnp.ones((1,)))


def test_the_bottleneck_embedding_is_jits():
    embed = PatchEmbedding(patch_size=8, embedding_dim=12, bottleneck=5)
    variables = {"params": {"Conv_0": {"kernel": jnp.asarray(EMBED["Conv_0.kernel"], jnp.float32)},
                            "Dense_0": {"kernel": jnp.asarray(EMBED["Dense_0.kernel"], jnp.float32),
                                        "bias": jnp.asarray(EMBED["Dense_0.bias"], jnp.float32)}}}
    tokens = embed.apply(variables, jnp.asarray(EMBED["pixels"]))
    assert_as_exact_as_the_reference(tokens, EMBED["tokens32"], EMBED["tokens"], "bottleneck")
