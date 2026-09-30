"""REPA-E: the autoencoder's regularizer and latent batch norm against the
official code (`tools/repae_reference.py`), and the gradient split of one
end-to-end step: the diffusion loss trains the model and not the
autoencoder, and the autoencoder's own loss trains it and not the model."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn

from dew.diffusion import presets
from dew.inputs import Field, InputSpec
from dew.nn.autoencoders.kl import AutoencoderKL
from dew.nn.autoencoders.sd_vae import StableDiffusionVAE
from dew.objectives.base import Step, scalar_loss
from dew.objectives.diffusion import Alignment, DiffusionObjective
from dew.objectives.diffusion.end_to_end import AUTOENCODER, LATENT_STATS, EndToEnd
from dew.registry import models
from dew.sampling import Euler, TextToImage
from dew.training import Trainer

CASE = np.load(Path(__file__).resolve().parent / "fixtures" / "repae" / "regularizer.npz")


def test_the_regularizer_is_repa_es():
    total, terms = EndToEnd().regularizer(jnp.asarray(CASE["images"], jnp.float32),
                                          jnp.asarray(CASE["reconstruction"], jnp.float32),
                                          jnp.asarray(CASE["moments"], jnp.float32))
    # float32 sums of a few hundred O(1) terms: 1e-5 relative is rounding.
    np.testing.assert_allclose(float(terms["kl"]), float(CASE["kl"]), rtol=1e-5)
    np.testing.assert_allclose(float(total), float(CASE["regularizer"]), rtol=1e-5)


def test_the_latent_batch_norm_is_repa_es():
    """Started from `init_bn`'s statistics, one training batch normalizes by
    its own and moves the running ones; evaluation then reads those."""
    end_to_end = EndToEnd()
    latents = jnp.asarray(CASE["latents"], jnp.float32)
    trained, statistics = end_to_end.batch_normalized(latents, end_to_end.initial_statistics(0.1, 0.8, 4))
    np.testing.assert_allclose(np.asarray(trained), CASE["trained"], rtol=0, atol=2e-6)
    np.testing.assert_allclose(np.asarray(statistics["mean"]), CASE["running_mean"], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(statistics["var"]), CASE["running_var"], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(end_to_end.normalized(latents, statistics)), CASE["evaluated"],
                               rtol=0, atol=2e-6)


class Patches(nn.Module):
    @nn.compact
    def __call__(self, pixels):
        return nn.Conv(5, (4, 4), strides=(4, 4))(pixels)


def objective(end_to_end: EndToEnd) -> DiffusionObjective:
    module = AutoencoderKL(channels=(8, 16), latent_channels=4, blocks_per_level=1, norm_groups=4)
    autoencoder = StableDiffusionVAE(
        model=module, params=module.init(jax.random.PRNGKey(5), jnp.zeros((1, 8, 8, 3)))["params"],
        dtype=jnp.float32, latent_shift=0.1, latent_scale=0.8)
    encoder = Patches()
    alignment = Alignment(encoder, encoder.init(jax.random.PRNGKey(9), jnp.zeros((1, 8, 8, 3))),
                          "dit_block_0", width=8)
    model = models.SimpleDiT(patch_size=2, emb_features=16, num_layers=2, num_heads=2, mlp_ratio=1,
                             output_channels=4)
    return DiffusionObjective(model, presets.Flow()(), InputSpec(Field("image", (8, 8, 3))), guidance=None,
                              sampler=Euler(), steps=2, autoencoder=autoencoder, alignment=alignment,
                              end_to_end=end_to_end, ema_decay=None)


BATCH = {"image": np.asarray(jax.random.randint(jax.random.PRNGKey(1), (4, 8, 8, 3), 0, 256), np.uint8)}
STEP = Step(step=jnp.asarray(0), key=jax.random.PRNGKey(2), ema=None)


def gradients(end_to_end: EndToEnd):
    task = objective(end_to_end)
    params = task.init(jax.random.PRNGKey(0))
    return params, jax.grad(lambda tree: scalar_loss(task, {**params, "params": tree}, BATCH, STEP)[0])(
        params["params"])


def total(tree) -> float:
    return float(sum(jnp.abs(leaf).sum() for leaf in jax.tree.leaves(tree)))


def test_each_loss_trains_its_own_side():
    """With the autoencoder's own loss weighted to zero, nothing reaches it:
    the denoising and alignment losses read a detached latent. With it on,
    the model's and projector's gradients are unchanged: the autoencoder's
    alignment reads them frozen."""
    params, joint = gradients(EndToEnd())
    _, model_only = gradients(EndToEnd(align_weight=0.0, reconstruction_weight=0.0, kl_weight=0.0))
    assert total(model_only[AUTOENCODER]) == 0
    assert total(joint[AUTOENCODER]) > 0
    for name in joint:
        if name != AUTOENCODER:
            for got, want in zip(jax.tree.leaves(joint[name]), jax.tree.leaves(model_only[name]), strict=True):
                np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    _, aligned_only = gradients(EndToEnd(reconstruction_weight=0.0, kl_weight=0.0))
    assert total(aligned_only[AUTOENCODER]) > 0
    assert set(params) >= {LATENT_STATS} and "autoencoder" not in params


def test_a_step_moves_the_running_statistics_and_the_task_decodes_with_them():
    task = objective(EndToEnd())
    trainer = Trainer(task, optax.adam(1e-3), key=jax.random.PRNGKey(3))
    state = trainer.initial_state()
    before = state.params[LATENT_STATS]
    state, *_ = trainer.compile(state, BATCH)(state, BATCH)
    after = state.params[LATENT_STATS]
    assert not np.allclose(np.asarray(after["mean"]), np.asarray(before["mean"]))

    published = TextToImage.from_objective(task, state.params)
    assert AUTOENCODER not in published.params["params"] and LATENT_STATS not in published.params
    for got, want in zip(jax.tree.leaves(published.params["autoencoder"]),
                         jax.tree.leaves(state.params["params"][AUTOENCODER]), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    np.testing.assert_allclose(np.asarray(published.autoencoder.latent_scale),
                               1 / np.sqrt(np.asarray(after["var"]) + 1e-4), rtol=1e-6)


def test_end_to_end_needs_alignment_and_a_kl_autoencoder():
    task = objective(EndToEnd())
    with pytest.raises(ValueError, match="needs `alignment`"):
        DiffusionObjective(task.model, task.process, task.inputs, autoencoder=task.autoencoder,
                           end_to_end=EndToEnd())
