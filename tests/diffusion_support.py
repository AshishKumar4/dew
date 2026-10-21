"""What the diffusion suites share: a sampler's step-by-step walk, a published source's one train step
and its batch, and a student stepped once and saved as a run."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from interop_support import scaled_gap

from dew.diffusion.schedules import expand
from dew.sampling.sample import sample


class SourceOracle(nn.Module):
    """The model `tools/diffusers_source_reference.py` runs on both sides: a
    bounded nonlinear function of the scaled input and the model time, so a
    trajectory and its Jacobian depend on every scheduler decision the config
    carries and on nothing else."""

    @nn.compact
    def __call__(self, x, temb):
        return jnp.sin(x) * 0.07 + expand(temb, x) * 0.001



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


def relative_gap(actual, expected) -> float:
    """The largest per leading index of `scaled_gap`, so a small row is held to its own magnitude."""
    return max(scaled_gap(a, e) for a, e in zip(np.asarray(actual), np.asarray(expected), strict=True))


def images_of(url: str, size: int, batch: int = 32):
    """A run's data of `size` squares from the images `url` names (`dew.data.presets.images`), in batches
    of `batch`: what a diffusion or JEPA run's config trains on, which most tests only build from."""
    from dew.data import presets
    from dew.data.sources import source

    return source(url).pipe(presets.images, size, batch=batch)


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
    state saves and places back bit for bit. Returns the initial state, read to the host before
    the step donated it, and the stepped state."""
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
    # The step donates the state it is given, the loaded source's own arrays among its leaves, so
    # what it is compared with is read first.
    started = jax.device_get(initial)
    state, _, _, _, accepted, _ = trainer.compile(initial, batch)(initial, batch)
    assert bool(accepted)
    assert value(state.variables) < before
    assert not np.allclose(state.variables["params"]["proj_out"]["kernel"],
                           started.variables["params"]["proj_out"]["kernel"])
    for held in ("encoders", "autoencoder"):
        for got, want in zip(jax.tree.leaves(state.variables[held]), jax.tree.leaves(started.variables[held]),
                             strict=True):
            np.testing.assert_array_equal(got, want)
    if checkpoints is not None:
        checkpoints.save(1, state, None, {})
        checkpoints.wait()
        restored, _, _ = trainer.place()
        for got, want in zip(jax.tree.leaves(restored), jax.tree.leaves(state), strict=True):
            np.testing.assert_array_equal(raw_leaf(got), raw_leaf(want))
    return started, state


def saved_after_one_step(config, batch, directory, rate: float):
    """`config`'s task stepped once by Adam at `rate` over `batch` and saved as a run in `directory`,
    with the record its inference reads. Returns the task and its stepped state."""
    import optax

    from dew.training import Checkpoints, Trainer

    task = config.build()
    trainer = Trainer(task, optax.adam(rate), key=jax.random.PRNGKey(3))
    state = trainer.initial_state()
    state, *_ = trainer.compile(state, batch)(state, batch)
    checkpoints = Checkpoints(str(directory))
    checkpoints.save(1, state, None, artifact=task.inference_record())
    checkpoints.wait()
    config.save(str(directory))
    return task, state
