"""What the diffusion suites share: a sampler's step-by-step walk, a published source's one train step
and its batch, a student stepped once and saved as a run, and a trainer's update held to its
objective's gradient by check 1 (`moved_weights`, `applied`, `objective_gradient`)."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from interop_support import scaled_gap
from reference_error import as_given

from dew.artifacts import Representations
from dew.diffusion.schedules import expand
from dew.inputs import unit_range
from dew.nn.backbones.dit import SimpleDiT
from dew.objectives.base import Aux, EMASpec, Objective
from dew.sampling.sample import sample


class SourceOracle(nn.Module):
    """The model `tools/diffusers_source_reference.py` runs on both sides: a
    bounded nonlinear function of the scaled input and the model time, so a
    trajectory and its Jacobian depend on every scheduler decision the config
    carries and on nothing else."""

    @nn.compact
    def __call__(self, x, temb):
        return jnp.sin(x) * 0.07 + expand(temb, x) * 0.001


class ConditionalSourceOracle(nn.Module):
    """`SourceOracle` offset by its condition, so a guided walk reads back the combination that was
    applied (`tools/guidance_reference.py` walks APG over it)."""

    @nn.compact
    def __call__(self, x, temb, label):
        return SourceOracle()(x, temb) + expand(label, x)



def apg_walked(case, label, x_T):
    """A `tools/guidance_reference.py` APG case walked by Dew: `case["steps"]` Euler steps of the linear
    path over `ConditionalSourceOracle` at `label`, APG guiding the guider's steps
    [int(start N), int(stop N)), which is `interval=(first / N, (after - 1) / N)`."""
    from dew.diffusion import FlowMatchingScheduler, FlowMatchPredictionTransform, Process
    from dew.sampling import APG, Euler

    steps = case["steps"]
    first, after = int(case["start"] * steps), int(case["stop"] * steps)
    rows = x_T.shape[0]
    process = Process(FlowMatchingScheduler(), FlowMatchPredictionTransform())
    denoise = process.denoiser(ConditionalSourceOracle(), {}, {"label": jnp.full((rows,), label)},
                               {"label": jnp.zeros((rows,))})
    guidance = APG(case["scale"], eta=case["eta"], norm_threshold=case["norm_threshold"],
                   momentum=case["momentum"], interval=(first / steps, (after - 1) / steps))
    return sample(denoise, x_T, steps + 1, solver=Euler(), guidance=guidance, key=jax.random.PRNGKey(0),
                  final_denoise=False)


def source_final(schedule, process, times, rescale, label, scale):
    """The last latent of a published scheduler's walk as `tools/diffusers_source_reference.py`
    walks it, as Dew's function of the initial one: `schedule`'s own solver over `process`'s grid
    `times` on `SourceOracle`, or, where the case guides (`rescale`, CFG's rescale, not None),
    classifier-free guidance at `scale` on `ConditionalSourceOracle` at `label`."""
    from dew.sampling import CFG

    if rescale is None:
        model = SourceOracle()
        return lambda value: walk(schedule.solver, process, model, value, times)[-1]
    model = ConditionalSourceOracle()

    def final(value):
        rows = value.shape[0]
        params = model.init(jax.random.PRNGKey(1), jnp.ones((1, *value.shape[1:])), jnp.ones((1,)),
                            jnp.ones((1,)))
        denoise = process.denoiser(model, params, {"label": jnp.full((rows,), label)},
                                   {"label": jnp.zeros((rows,))})
        return sample(denoise, value, solver=schedule.solver, guidance=CFG(scale, rescale=rescale),
                      key=jax.random.PRNGKey(0), times=times, final_denoise=False)

    return final


def walk(solver, process, model, x_T, times, key=0):
    """Each solver interval's new state, without the closing denoise, through `sample`'s own walk with
    its folded per-step keys (`sample(record=...)`)."""
    params = model.init(jax.random.PRNGKey(1), jnp.ones((1, *x_T.shape[1:])), jnp.ones((1,)))
    denoise = process.denoiser(model, params, {})
    with jax.ensure_compile_time_eval():
        times = jnp.asarray(times, jnp.float32)
    _, latents = sample(denoise, x_T, solver=solver, key=key, times=times, final_denoise=False,
                        record=lambda x, state, denoised: x)
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


class TanhLayer(nn.Module):
    """A reference stand-in network's layer: tanh of the tokens through one square kernel, zero at init."""

    @nn.compact
    def __call__(self, tokens):
        return jnp.tanh(tokens @ self.param("kernel", nn.initializers.zeros, (tokens.shape[-1],) * 2))


def image_batch(owner, prompts, size: int | None = None) -> dict:
    """One image a device, its pixels a ramp over `owner`'s (a loaded source's or an objective's) sample
    shape, `size` square where given, with `prompts` in turn, so a device count that is odd, one GPU
    included, still gets a prompt for every image."""
    height, width, channels = owner.inputs.sample.shape[-3:]
    if size is not None:
        height = width = size
    rows = jax.device_count()
    pixels = np.tile(np.arange(height * width * channels, dtype=np.uint8).reshape(1, height, width, channels),
                     (rows, 1, 1, 1))
    return {"image": pixels, **owner.inputs.tokenize([prompts[row % len(prompts)] for row in range(rows)])}


def source_objective(source):
    """A published pipeline source (`source`/pipeline) loaded in float32, and the objective that trains
    its transformer through the whole source: its autoencoder and text towers as state, no
    unconditional drop, no EMA, two steps."""
    from dew.interop.pretrained import Pretrained
    from dew.objectives.diffusion import DiffusionObjective

    loaded = Pretrained.load(str(source / "pipeline"), dtype="float32", attention_impl="xla")
    return loaded, DiffusionObjective(loaded.model, loaded.process, loaded.inputs,
                                      autoencoder=loaded.autoencoder, variables=loaded.variables,
                                      unconditional_prob=0.0, ema_decay=None, steps=2)


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


def saved_after_one_step(config, batch, directory, rate: float, key: int = 3):
    """`config`'s task stepped once by Adam at `rate` from the state `key` initializes, over `batch`
    (or the batch `batch(task)` makes), and saved as a run in `directory`, with the record its
    inference reads. Returns the task, its stepped state and the batch."""
    import optax

    from dew.training import Checkpoints, Trainer

    task = config.build()
    batch = batch(task) if callable(batch) else batch
    trainer = Trainer(task, optax.adam(rate), key=jax.random.PRNGKey(key))
    state = trainer.initial_state()
    state, *_ = trainer.compile(state, batch)(state, batch)
    checkpoints = Checkpoints(str(directory))
    checkpoints.save(1, state, None, artifact=task.inference_record())
    checkpoints.wait()
    config.save(str(directory))
    return task, state, batch


def moved_weights(variables):
    """Check 1's `draw(seed)` over a trainer's `variables`: each trained leaf moved by the seed's draw,
    so zero-initialized layers take a gradient too."""
    host = jax.tree.map(np.asarray, variables)

    def draw(seed):
        keys = iter(jax.random.split(jax.random.key(seed), len(jax.tree.leaves(host["params"]))))

        def moved_leaf(leaf):
            return (leaf + 0.02 * np.asarray(jax.random.normal(next(keys), leaf.shape))).astype(leaf.dtype)

        return ({**host, "params": jax.tree.map(moved_leaf, host["params"])},)

    return draw


def applied(before, after):
    """sgd(1)'s update applied, in float64: the gradient with the update's float32 rounding."""
    return np.asarray(before, np.float64) - np.asarray(after, np.float64)


def objective_gradient(objective, variables, batch, step):
    """`objective`'s gradient of its trained params over `batch` at `step`'s draws."""
    def loss(params):
        return objective.scalar_loss({**variables, "params": params}, batch, step)[0]

    return jax.grad(loss)(variables["params"])


def as_a_step(trainer, variables, micro, gradient):
    """`gradient` jitted with the compiler options `trainer`'s step of a `micro` microbatch takes over
    its mesh, its devices and row shards (`dew.training.memory.step_compiler_options`, as
    `Trainer._planned` derives them), so a check of that step against it compiles both arms alike: on
    a GPU the step keeps dots that share an input apart, where XLA's default merges them into one
    GEMM, which rounds otherwise."""
    from dew.nn.sharding import batch_axes
    from dew.objectives.base import FROZEN
    from dew.training.memory import device_tokens, step_compiler_options

    mesh = trainer.device_mesh
    shards = math.prod(mesh.shape[axis] for axis in batch_axes(mesh))
    options = step_compiler_options(trainer.objective, device_tokens(trainer.objective, micro, shards),
                                    FROZEN in variables, mesh.devices.size)
    return jax.jit(gradient, compiler_options=options)


def float32_update(gradient):
    """The update a trainer's sgd(1) step is held to (`applied`): `gradient(variables)`, the objective's
    own, applied in float32 as the trainer applies it."""
    @as_given
    def update(variables):
        grads = jax.tree.map(np.asarray, gradient(variables))
        return jax.tree.map(lambda before, grad: applied(before, (before - grad).astype(np.float32)),
                            variables["params"], grads)

    return update


def float64_gradient(twin, step):
    """Check 1's truth of an sgd(1) update at `step`'s draws, `truth(variables, batch)`: `twin`'s
    gradient, the objective over its model in float64, of the variables and the batch widened,
    compiled once as each arm is."""
    from reference_error import widened

    wide = jax.jit(lambda variables, batch: objective_gradient(twin, variables, batch, step))

    def truth(variables, batch):
        with jax.enable_x64(new_val=True):
            return jax.tree.map(np.asarray, wide(widened(variables), widened(batch)))

    return truth


class DeterministicObjective(Objective):
    """Squared error against the input, straight through the real DiT.

    No noise level, no dropout, no unconditional mask: the loss depends only
    on the parameters and the batch. That is what makes two accumulation
    regimes comparable, and two topologies: every micro-gradient in a window
    is identical, which is exactly the "one big batch" an accumulated update
    stands in for, so a k-accumulated run and a plain run must trace the same
    parameters, and a partitioned run the same losses as a single-device one.
    """

    def __init__(self, decay=None, emb_features=32):
        self.model = SimpleDiT(patch_size=4, emb_features=emb_features, num_layers=1,
                               num_heads=2, mlp_ratio=1)
        self.ema = EMASpec(decay=optax.constant_schedule(0.999) if decay is None else decay)

    def fresh_variables(self, key, held):
        return self.model.init(key, jnp.ones((1, 8, 8, 3)), jnp.zeros((1,)))

    def loss(self, variables, batch, step):
        data = unit_range(batch["image"])
        preds = self.model.apply(variables, data, jnp.zeros((data.shape[0],), jnp.float32))
        return jnp.mean((preds - data) ** 2), Aux({})

    def evaluate(self, params, batch, step):
        """The EMA copy's reconstruction, pooled per example: reads the
        sharded EMA parameters, which the training step alone never does."""
        data = unit_range(batch["image"])
        preds = self.model.apply(step.ema, data, jnp.zeros((data.shape[0],), jnp.float32))
        return Representations(features=jnp.mean(preds, axis=(1, 2)),
                               labels=jnp.zeros((data.shape[0],), jnp.int32))
