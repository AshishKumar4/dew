"""Step-by-step sampler trajectories used by numerical reference tests."""

import jax
import jax.numpy as jnp
import numpy as np

from dew.sampling.sample import sample


def walk(solver, process, model, x_T, times, key=0):
    """Each solver interval's new state, without the closing denoise, through `sample`'s own walk with
    its folded per-step keys (`sample(record=...)`)."""
    params = model.init(jax.random.PRNGKey(1), jnp.ones((1, *x_T.shape[1:])), jnp.ones((1,)))
    denoise = process.denoiser(model, params, {})
    with jax.ensure_compile_time_eval():
        times = jnp.asarray(times, jnp.float32)
    _, latents = sample(denoise, x_T, solver=solver, key=key, times=times, final_denoise=False,
                        record=lambda x, state: x)
    return latents


def image_batch(loaded, prompts) -> dict:
    """One image a device, its pixels a ramp over the source's sample shape, with `prompts` in turn,
    so a device count that is odd, one GPU included, still gets a prompt for every image."""
    height, width, channels = loaded.inputs.sample.shape
    rows = jax.device_count()
    pixels = np.tile(np.arange(height * width * channels, dtype=np.uint8).reshape(1, height, width, channels),
                     (rows, 1, 1, 1))
    return {"image": pixels, **loaded.inputs.tokenize([prompts[row % len(prompts)] for row in range(rows)])}


def trained_step(objective, batch, checkpoints=None):
    """One SGD step of a published source's `objective` over `batch` through the public trainer, at a
    fixed key: the loss falls, the transformer's output projection moves, and the loaded encoders and
    autoencoder, which are state rather than weights, keep every leaf. With `checkpoints`, the stepped
    state saves and places back bit for bit. Returns the initial and the stepped states."""
    import optax
    from affine_run import raw_leaf

    from dew.objectives import Step
    from dew.training import Trainer

    trainer = Trainer(objective, optax.sgd(1e-2), key=jax.random.PRNGKey(3), checkpoints=checkpoints)
    initial = trainer.initial_state()
    fixed = Step(jnp.asarray(0), jax.random.PRNGKey(5), None)

    def value(params) -> float:
        loss, _ = objective.loss(params, batch, fixed)
        return float(loss.total / loss.mass)

    before = value(initial.variables)
    state, _, _, _, accepted, _ = trainer.compile(initial, batch)(initial, batch)
    assert bool(accepted)
    assert value(state.variables) < before
    assert not np.allclose(state.variables["params"]["proj_out"]["kernel"],
                           initial.variables["params"]["proj_out"]["kernel"])
    for held in ("encoders", "autoencoder"):
        for got, want in zip(jax.tree.leaves(state.variables[held]), jax.tree.leaves(initial.variables[held]),
                             strict=True):
            np.testing.assert_array_equal(got, want)
    if checkpoints is not None:
        checkpoints.save(1, state, None, {})
        checkpoints.wait()
        restored, _, _ = trainer.place()
        for got, want in zip(jax.tree.leaves(restored), jax.tree.leaves(state), strict=True):
            np.testing.assert_array_equal(raw_leaf(got), raw_leaf(want))
    return initial, state
