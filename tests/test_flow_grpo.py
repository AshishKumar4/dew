"""Flow-GRPO against Gaussian algebra and the authors' SDE convention.

Reference: arXiv:2505.05470v5, equations 8-9, and
https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py
The reference averages coordinate log densities for its policy ratio. These
checks retain the joint density until the objective performs that reduction.
Each density, loss and gradient is on par with its closed form, or a rollout's
rescoring with its recording, from the float64 evaluation over drawn inputs
(check 1, tests/reference_error.py), the closed form evaluated on the walk's
own states; what is one computation, or exact arithmetic, is held exactly.
"""

import itertools
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from recording import RecordingTracker
from reference_error import assert_as_exact, declared_seeds, indexed_recording, recording, widened

from dew.diffusion import FlowMatchingScheduler, FlowMatchPredictionTransform, Process
from dew.diffusion.presets import JiT
from dew.diffusion.transforms import ConsistencyBoundary, EpsilonPredictionTransform, VPredictionTransform
from dew.inputs import Field, InputSpec
from dew.nn.autoencoders.api import AutoEncoder
from dew.objectives.base import Step
from dew.objectives.rl.flow import FlowGRPOObjective, FlowRollout
from dew.sampling import CFG, sample
from dew.sampling.flow import FlowSDE, flow_transition
from dew.telemetry.records import RECORD_TYPES
from dew.training import Trainer
from tools.parity_flow_grpo import RECORDED, dew_transition, transition_inputs

SEEDS = {'guided walk': 8, 'converted': 8, 'rescored walk': 8, 'clipping': 8, 'kl': 8, 'rollout': 8,
         'inpainting': 8, 'live policy': 8, 'mixed precision': 8, 'prompt evaluation': 8}
"""Check 1's declared seeds for each site of the file, sized by `seeds_for` (on armada)."""
ROWS = 64
"""A drawn batch's rows: enough that a few seeds resolve a site."""
BATCHES = 16
"""The batches a draw of a site holds whose outputs are one number a batch."""


def as_float32(tree):
    """`tree`'s float leaves at float32, as the arm under test took them, its other leaves as they are:
    a float64 truth evaluates the arm's own float32 states."""
    return jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float32)
                        if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating) else leaf, tree)


def walked_in_float32(walk, *inputs):
    """`walk` on `inputs` at float32 with x64 off, as the arm under test walked them."""
    with jax.enable_x64(new_val=False):
        return walk(*as_float32(inputs))


def drawn_rows(*shapes, scale=0.5):
    """Check 1's draw of normal arrays of `shapes`, ROWS rows each, scaled by `scale`."""
    def draw(seed):
        keys = jax.random.split(jax.random.fold_in(jax.random.key(0xF10), seed), len(shapes))
        return tuple(scale * jax.random.normal(key, (ROWS, *shape))
                     for key, shape in zip(keys, shapes, strict=True))
    return draw


class AffineVelocity(nn.Module):
    @nn.compact
    def __call__(self, x, temb, offset=0):
        gain = self.param("gain", nn.initializers.constant(0.25), ())
        return gain * x + (temb / 1000)[:, None] * 0.1 + offset


def guided_density(trajectory, offset, dtype):
    """The authors' SDE log density of each recorded transition of a walk of AffineVelocity under
    FlowMatchingScheduler(shift=3) and CFG(2) inside (0.1, 0.8), in NumPy at `dtype`, from the walk's
    own states."""
    offset = np.asarray(offset, dtype)
    found = []
    for index in range(4):
        t, s = np.asarray(trajectory.times[index:index + 2], dtype)
        sigma, following = 3 * t / (1 + 2 * t), 3 * s / (1 + 2 * s)
        latent = np.asarray(trajectory.states[:, index], dtype)
        action = np.asarray(trajectory.states[:, index + 1], dtype)
        scale = 2 if 0.1 <= index / 4 <= 0.8 else 1
        velocity = 0.25 * latent + 0.1 * sigma + scale * offset
        dt = following - sigma
        g_squared = 0.5**2 * sigma / (1 - (following if t == 1 else sigma))
        mean = latent + (velocity + g_squared / (2 * sigma) * (latent + (1 - sigma) * velocity)) * dt
        variance = g_squared * -dt
        found.append((-0.5 * ((action - mean)**2 / variance + np.log(2 * math.pi * variance))).sum(1))
    return np.stack(found, 1)


def test_recorded_trajectory_rescores_under_shifted_guided_process():
    """The walk's recorded densities, guidance and shift included, on par with the authors' Gaussian
    on the walk's own states from its float64 evaluation over drawn starts and offsets (check 1); the
    walk's samples are the sampler's, bit for bit."""
    process = Process(FlowMatchingScheduler(shift=3), FlowMatchPredictionTransform())
    model = AffineVelocity()
    x = jnp.asarray([[0.2, -0.5], [-0.1, 0.7]])
    offset = jnp.asarray([[-0.4], [0.8]])
    variables = model.init(jax.random.key(0), x, jnp.ones(2), offset=offset)
    solver, guidance, key = FlowSDE(noise_level=0.5), CFG(2, interval=(0.1, 0.8)), jax.random.key(1)

    def denoiser(offset):
        return process.denoiser(model, variables, {"offset": offset}, {"offset": jnp.zeros_like(offset)})

    walk = jax.jit(lambda start, offset: solver.trajectory(denoiser(offset), start, 5, guidance=guidance,
                                                           key=key))
    trajectory = walk(x, offset)
    ordinary = sample(denoiser(offset), x, 5, solver=solver, guidance=guidance, key=key)
    np.testing.assert_array_equal(trajectory.samples, ordinary)

    def truth(start, offset):
        return guided_density(walked_in_float32(walk, start, offset), offset, np.float64)

    assert_as_exact(lambda start, offset: walk(start, offset).log_probs,
                    lambda start, offset: guided_density(walk(start, offset), offset, np.float32),
                    drawn_rows((2,), (1,)), "guided walk", seeds=SEEDS['guided walk'], truth_fn=truth)
    deterministic = FlowSDE(0).trajectory(denoiser(offset), x, 5, key=key)
    assert np.isnan(deterministic.log_probs).all()
    assert not np.asarray(deterministic.stochastic).any()



def trajectory_batch(trajectory, advantages, mask=None):
    count, points = trajectory.states.shape[:2]
    return {
        "latents": trajectory.states[:, :-1],
        "next_latents": trajectory.states[:, 1:],
        "timesteps": jnp.broadcast_to(trajectory.times[:-1], (count, points - 1)),
        "next_timesteps": jnp.broadcast_to(trajectory.times[1:], (count, points - 1)),
        "rollout_steps": jnp.full((count,), points, jnp.int32),
        "old_log_probs": trajectory.log_probs,
        "transition_mask": trajectory.stochastic if mask is None else jnp.asarray(mask),
        "advantages": jnp.asarray(advantages),
    }


@pytest.mark.parametrize("prediction", ["clean", "v"])
def test_flow_objective_uses_converted_predictions(prediction):
    """JiT and other parameterizations share the rectified transition and its rescored density: the
    walk's recorded densities and the objective's rescoring of them, on par with the rectified
    Gaussian on the walk's own states from its float64 evaluation over drawn starts (check 1)."""
    process = JiT() if prediction == "clean" else Process(FlowMatchingScheduler(), VPredictionTransform())
    model = AffineVelocity()
    objective = FlowGRPOObjective(model, process, InputSpec(Field("image", (2,))),
                                  sde=FlowSDE(0.5), guidance=None)
    variables = objective.init(jax.random.key(30))
    denoise = objective.process.denoiser(model, variables, {})
    walk = jax.jit(lambda start: objective.sde.trajectory(denoise, start, 5, key=31))
    rescore = jax.jit(objective.log_probs)

    def rectified(trajectory, dtype):
        found = []
        for index, (t, s) in enumerate(zip(trajectory.times[:-1], trajectory.times[1:], strict=True)):
            t, s = np.asarray(t, dtype), np.asarray(s, dtype)
            x = np.asarray(trajectory.states[:, index], dtype)
            raw = 0.25 * x + 0.1 * t
            if prediction == "clean":
                velocity = (x - raw) / t
            else:
                variance = (1 - t)**2 + t**2
                velocity = ((2 * t - 1) * x + np.sqrt(variance) * raw) / variance
            dt = s - t
            g_squared = 0.5**2 * t / (1 - (s if t == 1 else t))
            mean = x + (velocity + g_squared / (2 * t) * (x + (1 - t) * velocity)) * dt
            variance = g_squared * -dt
            action = np.asarray(trajectory.states[:, index + 1], dtype)
            found.append(-0.5 * (((action - mean)**2 / variance) + np.log(2 * math.pi * variance)).sum(1))
        density = np.stack(found, 1)
        return {"recorded": density, "rescored": density}

    def scored(start):
        trajectory = walk(start)
        return {"recorded": trajectory.log_probs,
                "rescored": rescore(variables, trajectory_batch(trajectory, np.ones(ROWS, np.float32)))}

    assert_as_exact(scored, lambda start: rectified(walk(start), np.float32), drawn_rows((2,)),
                    f"{prediction} converted", seeds=SEEDS['converted'],
                    truth_fn=lambda start: rectified(walked_in_float32(walk, start), np.float64))
    trajectory = walk(jnp.asarray([[0.2, -0.5], [-0.1, 0.7]]))
    batch = trajectory_batch(trajectory, [-1, 1])
    loss, _ = objective.scalar_loss(variables, batch, Step(jnp.asarray(0), jax.random.key(32), None))
    assert np.isfinite(float(loss))


@pytest.mark.parametrize("prediction", [EpsilonPredictionTransform(),
                                        ConsistencyBoundary(EpsilonPredictionTransform())],
                         ids=["epsilon", "consistency_epsilon"])
def test_a_prediction_undefined_at_pure_noise_is_refused_only_where_a_walk_evaluates_it(prediction):
    """Epsilon's clean sample divides by alpha, inside a consistency boundary or not, and a
    rectified flow's alpha is 0 at pure noise. A trajectory from there, and a rollout that starts
    there, are refused before they run, rather than rolling out non-finite states.
    From t = 0.9 the same prediction walks compiled, and each transition rescores, compiled on its
    own, on par with the density the walk recorded over drawn starts (check 1), the truth the
    rescoring in float64 on the walk's own states."""
    process = Process(FlowMatchingScheduler(), prediction)
    model = AffineVelocity()
    x = jnp.asarray([[0.2, -0.5], [-0.1, 0.7]])
    variables = model.init(jax.random.key(0), x, jnp.ones(2))
    denoise = process.denoiser(model, variables, {})
    solver = FlowSDE(0.5)
    with pytest.raises(ValueError):
        solver.trajectory(denoise, x, 5, key=1)
    objective = FlowGRPOObjective(model, process, InputSpec(Field("image", (2,))), sde=solver, guidance=None)
    with pytest.raises(ValueError):
        FlowRollout(objective, lambda images, batch: np.zeros(len(images)), steps=5)
    times = jnp.linspace(0.9, 0.0, 5)

    @jax.jit
    def walk(start):
        _, (states, log_probs) = sample(
            denoise, start, solver=solver, key=jax.random.key(1), times=times, final_denoise=False,
            record=lambda following, state: (following, state[0]))
        return jnp.concatenate((start[None], states)), log_probs

    assert np.isfinite(np.asarray(walk(x)[0])).all()

    def rescoring(variables):
        denoise = process.denoiser(model, variables, {})

        @jax.jit
        def rescored(states):
            def one(latent, action, t, s):
                t, s = jnp.full((latent.shape[0],), t), jnp.full((latent.shape[0],), s)
                clean, noise = denoise(latent, t)
                return solver.transition(latent, t, s, clean, noise, process).log_prob(action)
            return jnp.stack([one(states[index], states[index + 1], times[index], times[index + 1])
                              for index in range(4)])
        return rescored

    rescored, wide = rescoring(variables), rescoring(widened(variables))

    def truth(start):
        states = np.asarray(walked_in_float32(walk, start)[0], np.float64)
        with jax.enable_x64(new_val=True):
            return wide(jnp.asarray(states))

    assert_as_exact(lambda start: rescored(walk(start)[0]), lambda start: walk(start)[1], drawn_rows((2,)),
                    f"{type(prediction).__name__} rescored walk", seeds=SEEDS['rescored walk'],
                    truth_fn=truth)


def clipped_objective(gain, states, times, advantages, mask, *, beta):
    """Flow-GRPO's clipped loss of a policy of `gain` against the walk's `states` under gain 0.25 and
    a reference of gain 0.1 (`beta` its KL's weight), its means in closed form, averaged over `mask`'s
    transitions, in jnp at `gain`'s dtype: the clip range 0.001, the advantages clipped to 2."""
    dtype = jnp.asarray(gain).dtype
    latent, following = jnp.asarray(states[:, :-1], dtype), jnp.asarray(states[:, 1:], dtype)
    t, s = jnp.asarray(times[:-1], dtype), jnp.asarray(times[1:], dtype)
    g2 = 0.6**2 * t / (1 - jnp.where(t == 1, s, t))
    variance = (g2 * (t - s))[None, :]
    coefficient = ((1 + g2 * (1 - t) / (2 * t)) * (s - t))[None, :, None]
    offset = latent * (1 + g2 / (2 * t) * (s - t))[None, :, None]

    def mean(gain):
        return offset + (gain * latent + 0.1 * t[None, :, None]) * coefficient

    old, current, reference = mean(jnp.asarray(0.25, dtype)), mean(gain), mean(jnp.asarray(0.1, dtype))
    log_ratio = (-jnp.square(following - current) + jnp.square(following - old)).mean(-1) / (2 * variance)
    ratio = jnp.exp(log_ratio)
    advantage = jnp.clip(jnp.asarray(advantages, dtype), -2, 2)[:, None]
    policy = jnp.maximum(-advantage * ratio, -advantage * jnp.clip(ratio, 0.999, 1.001))
    kl = jnp.square(current - reference).mean(-1) / (2 * variance)
    weights = jnp.asarray(mask, dtype)
    return jnp.sum(weights * (policy + beta * kl)) / jnp.sum(weights)


def test_flow_objective_matches_clipping_and_conditional_gaussian_kl():
    """The clipped objective's loss and its gradient in the policy, with and without the KL to a
    frozen reference, on par with the closed form's from its float64 evaluation on the walk's own
    states over drawn starts, advantages and masks (check 1); the clip binds on some transitions and
    not all, and a KL without its reference is refused."""
    process = Process(FlowMatchingScheduler(), FlowMatchPredictionTransform())
    model = AffineVelocity()
    objective = FlowGRPOObjective(model, process, InputSpec(Field("image", (2,))),
                                  sde=FlowSDE(0.6), guidance=None, beta=0.2,
                                  clip_range=0.001, adv_clip_max=2)
    without_reference = FlowGRPOObjective(model, process, objective.inputs, sde=objective.sde,
                                           guidance=None, clip_range=0.001, adv_clip_max=2)
    old = objective.init(jax.random.key(2))
    current = {**old, "params": {"gain": jnp.asarray(0.9)}}
    reference = {**old, "params": {"gain": jnp.asarray(0.1)}}
    step = Step(jnp.asarray(0), jax.random.key(4), reference)
    walk = jax.jit(lambda start: objective.sde.trajectory(process.denoiser(model, old, {}), start, 4,
                                                          key=jax.random.key(3)))

    def draw(seed):
        # Sixteen batches a draw: each loss and gradient is one number a batch.
        keys = jax.random.split(jax.random.fold_in(jax.random.key(5), seed), 3)
        return (0.5 * jax.random.normal(keys[0], (BATCHES, ROWS, 2)),
                jax.random.uniform(keys[1], (BATCHES, ROWS), minval=-3, maxval=3),
                jax.random.bernoulli(keys[2], 0.6, (BATCHES, ROWS, 3)))

    def batch_of(start, advantages, mask):
        batch = trajectory_batch(walk(start), advantages, mask)
        return {**batch, "old_log_probs": jnp.where(mask, batch["old_log_probs"], jnp.nan)}

    @jax.jit
    @jax.vmap
    def scored(start, advantages, mask):
        batch = batch_of(start, advantages, mask)

        def loss(gain, objective, step):
            return objective.scalar_loss({**current, "params": {"gain": gain}}, batch, step)[0]

        value, gradient = jax.value_and_grad(loss)(current["params"]["gain"], objective, step)
        return {"loss": value, "gradient": gradient,
                "policy": loss(current["params"]["gain"], without_reference, step.replace(ema=None))}

    def closed(states, times, advantages, mask, gain):
        terms = (states, times, advantages, mask)
        value, gradient = jax.value_and_grad(clipped_objective)(gain, *terms, beta=0.2)
        return {"loss": value, "gradient": gradient,
                "policy": clipped_objective(gain, states, times, advantages, mask, beta=0.0)}

    def reference_arm(start, advantages, mask):
        trajectory = jax.vmap(walk)(start)
        return jax.vmap(closed, (0, 0, 0, 0, None))(trajectory.states, trajectory.times, advantages, mask,
                                                    jnp.float32(0.9))

    def truth(start, advantages, mask):
        trajectory = walked_in_float32(jax.vmap(walk), start)
        with jax.enable_x64(new_val=True):
            return jax.vmap(closed, (0, 0, 0, 0, None))(
                np.asarray(trajectory.states, np.float64), np.asarray(trajectory.times, np.float64),
                advantages, mask, jnp.float64(0.9))

    assert_as_exact(scored, reference_arm, draw, "clipping", seeds=SEEDS['clipping'], truth_fn=truth)

    rescore = jax.jit(objective.log_probs)

    def rescored(variables):
        return lambda start: rescore(variables, trajectory_batch(walk(start), np.ones(ROWS, np.float32)))

    def wide_rescored(start):
        trajectory = walked_in_float32(walk, start)
        with jax.enable_x64(new_val=True):
            return rescore(widened(old), widened(trajectory_batch(trajectory, np.ones(ROWS, np.float32))))

    assert_as_exact(rescored(old), lambda start: walk(start).log_probs, drawn_rows((2,)),
                    "rescored clipping walk", seeds=SEEDS['rescored walk'], truth_fn=wide_rescored)

    start, advantages, mask = (value[0] for value in draw(0))
    trajectory = walk(start)
    latent = np.asarray(trajectory.states[:, :-1], np.float64)
    following = np.asarray(trajectory.states[:, 1:], np.float64)
    t, s = (np.asarray(times, np.float64) for times in (trajectory.times[:-1], trajectory.times[1:]))
    g2 = 0.6**2 * t / (1 - np.where(t == 1, s, t))
    coefficient = ((1 + g2 * (1 - t) / (2 * t)) * (s - t))[None, :, None]
    offset = latent * (1 + g2 / (2 * t) * (s - t))[None, :, None]
    means = [offset + (gain * latent + 0.1 * t[None, :, None]) * coefficient for gain in (0.25, 0.9)]
    ratio = np.exp((-np.square(following - means[1]) + np.square(following - means[0])).mean(-1)
                   / (2 * g2 * (t - s)))
    advantage = np.clip(np.asarray(advantages, np.float64), -2, 2)[:, None]
    binds = np.maximum(-advantage * ratio, -advantage * np.clip(ratio, 0.999, 1.001)) != -advantage * ratio
    # The clip binds on some masked transitions and not on all of them.
    bound = int(np.count_nonzero(binds & np.asarray(mask)))
    assert bound not in {0, int(np.count_nonzero(mask))}
    stats, _ = objective.loss(current, batch_of(start, advantages, mask), step)
    assert float(stats.mass) == np.count_nonzero(mask)
    with pytest.raises(ValueError):
        objective.scalar_loss(current, batch_of(start, advantages, mask), step.replace(ema=None))


def test_the_conditional_gaussian_kl_is_the_closed_form():
    """Two transitions of one variance whose velocities differ by 0.3: the KL between them on par with
    the closed form's from its float64 evaluation over drawn states, velocities and times (check 1)."""
    noise = 0.4

    def draw(seed):
        keys = jax.random.split(jax.random.fold_in(jax.random.key(6), seed), 3)
        t = jax.random.uniform(keys[1], (ROWS,), minval=0.2, maxval=0.9)
        return (jax.random.normal(keys[0], (ROWS, 2, 2)) / 3, t,
                t * jax.random.uniform(keys[2], (ROWS,), minval=0.1, maxval=0.9))

    def kl(x, t, s):
        velocity = jnp.full_like(x, 0.2)
        transition = flow_transition(x, velocity, t, s, noise_level=noise)
        return transition.kl(flow_transition(x, velocity + 0.3, t, s, noise_level=noise).mean)

    def closed(x, t, s):
        g_squared = noise**2 * t / (1 - t)
        coefficient = (1 + g_squared * (1 - t) / (2 * t)) * (s - t)
        return 4 * jnp.square(0.3 * coefficient) / (2 * g_squared * (t - s))

    assert_as_exact(jax.jit(kl), jax.jit(closed), draw, "kl", seeds=SEEDS['kl'])


def test_deterministic_rollout_cannot_contribute_policy_gradient():
    process = Process(FlowMatchingScheduler(), FlowMatchPredictionTransform())
    model = AffineVelocity()
    objective = FlowGRPOObjective(model, process, InputSpec(Field("image", (2,))),
                                  sde=FlowSDE(0), guidance=None, beta=0.1)
    variables = objective.init(jax.random.key(7))
    x = jnp.asarray([[0.1, -0.2], [0.3, 0.4]])
    trajectory = objective.sde.trajectory(process.denoiser(model, variables, {}), x, 3,
                                          key=jax.random.key(8))
    batch = trajectory_batch(trajectory, [-1, 1], np.ones((2, 2), bool))
    step = Step(jnp.asarray(0), jax.random.key(9), variables)
    stats, _ = objective.loss(variables, batch, step)
    value, active = objective.reduce_loss(stats)
    assert float(stats.mass) == 0 and float(value) == 0 and not bool(active)
    gradient = jax.grad(lambda gain: objective.scalar_loss(
        {**variables, "params": {"gain": gain}}, batch, step)[0])(variables["params"]["gain"])
    assert float(gradient) == 0



def test_flow_rollout_groups_rewards_selects_steps_and_preserves_likelihoods():
    """Rescoring reproduces the rollout's likelihoods, on par with them over
    rollouts (check 1), guidance included: the rollout walks 4 points and
    guides its step 1 of 3, inside (0.3, 0.6), while the objective's own walk
    of 7 points would not guide a step 1 of 6, so the rescoring reads the
    rollout's step count off the batch. Its group advantages are the float64
    normalization's, rounded once, and its timesteps the 4-point walk's first
    two, bit for bit."""
    from dew.inputs import CharTable, Condition
    from dew.nn.backbones.dit import SimpleDiT

    process = Process(FlowMatchingScheduler(shift=2), FlowMatchPredictionTransform())
    inputs = InputSpec(Field("image", (4, 4, 1)), {
        "textcontext": Condition(CharTable.from_pretrained(tokens=3, features=4))})
    model = SimpleDiT(output_channels=1, patch_size=2, emb_features=8,
                      num_layers=1, num_heads=2, mlp_ratio=2)
    guidance = CFG(1.5, interval=(0.3, 0.6))
    initial = FlowGRPOObjective(model, process, inputs, guidance=guidance, steps=7).init(jax.random.key(20))
    # The DiT starts with its output zeroed, which no condition moves; noise in
    # every weight lets a guided transition part from an unguided one.
    leaves, tree = jax.tree.flatten(initial["params"])
    noise = jax.random.split(jax.random.key(22), len(leaves))
    params = jax.tree.unflatten(tree, [leaf + 0.1 * jax.random.normal(key, leaf.shape, leaf.dtype)
                                       for leaf, key in zip(leaves, noise, strict=True)])
    objective = FlowGRPOObjective(model, process, inputs, guidance=guidance, steps=7,
                                  variables={**initial, "params": params})

    optimizer = optax.sgd(1e-3)
    state = Trainer(objective, optimizer, key=jax.random.key(21)).initial_state()
    variables = state.variables
    prompts = {**inputs.tokenize(["red", "blue"]), "target": np.asarray([-0.3, 0.6], np.float32)}

    def reward(images, context):
        return -np.square(images.mean(axis=(1, 2, 3)) - context["target"])

    rollout = FlowRollout(objective, reward, groups=3, steps=4, train_steps=2)
    batch = rollout(state, prompts, jax.random.key(23))
    rewards = np.asarray(batch["rewards"], np.float64).reshape(2, 3)
    expected = (rewards - rewards.mean(axis=1, keepdims=True)) / (rewards.std(axis=1, keepdims=True) + 1e-4)
    np.testing.assert_array_equal(batch["advantages"], expected.reshape(-1).astype(np.float32))
    np.testing.assert_array_equal(batch["timesteps"], np.broadcast_to(process.times(4)[:2], (6, 2)))

    def rolled(seed):
        return rollout(state, prompts, jax.random.fold_in(jax.random.key(23), seed))

    assert_rescores_as_recorded(objective, variables, rolled, "rollout", SEEDS['rollout'])
    loss, _ = objective.scalar_loss(variables, batch, Step(jnp.asarray(0), jax.random.key(24), None))
    gradients = jax.grad(lambda p: objective.scalar_loss({**variables, "params": p}, batch,
        Step(jnp.asarray(0), jax.random.key(24), None))[0])(variables["params"])
    norm = float(optax.tree.norm(gradients))
    assert np.isfinite(float(loss)) and np.isfinite(norm) and norm > 1e-5


def assert_rescores_as_recorded(objective, variables, rollout_of, label: str, seeds: int):
    """Check 1: the objective's rescoring of a rollout's transitions (`rollout_of(seed)`, a batch) on
    par with the densities the rollout recorded, the truth the rescoring in float64 of the rollout's
    own states, over rollouts drawn from their keys."""
    rescore = jax.jit(objective.log_probs)

    def truth(batch):
        with jax.enable_x64(new_val=True):
            return rescore(widened(variables), widened(batch))

    assert_as_exact(lambda batch: rescore(variables, batch), lambda batch: batch["old_log_probs"],
                    lambda seed: (rollout_of(seed),), label, seeds=seeds, truth_fn=truth)


class Inpainting(nn.Module):
    """A velocity of the latents beside an inpainting model's masked-image inputs, refused
    without them as an inpainting UNet's input-channel check refuses."""

    @nn.compact
    def __call__(self, x, temb, mask=None, masked_image=None):
        if mask is None or masked_image is None:
            raise ValueError("an inpainting model reads mask and masked_image")
        gain = self.param("gain", nn.initializers.constant(0.25), ())
        return gain * x + 0.3 * masked_image * (1 - mask) + (temb / 1000)[:, None, None, None] * 0.1


class Pooled(AutoEncoder):
    """Latents of the 2x2 means of the pixels' first channel, decoded by repeating them."""

    def __init__(self):
        self.params = {}

    def encode_batch(self, params, x, key=None, noise=None):
        rows, height, width, _ = x.shape
        return x[..., :1].reshape(rows, height // 2, 2, width // 2, 2, 1).mean(axis=(2, 4))

    def decode_batch(self, params, z):
        return jnp.repeat(jnp.repeat(jnp.repeat(z, 2, axis=1), 2, axis=2), 3, axis=-1)

    @property
    def downscale_factor(self):
        return 2

    @property
    def latent_channels(self):
        return 1


def test_an_inpainting_rollout_rescores_with_its_masked_image_inputs():
    """Flow-GRPO samples an inpainting model with the masked image and mask its inputs give,
    records them with each transition and rescores the transitions with them: the recorded
    likelihoods come back, on par over rollouts (check 1), and a policy step through them has a
    gradient."""
    process = Process(FlowMatchingScheduler(), FlowMatchPredictionTransform())
    inputs = InputSpec(Field("image", (4, 4, 3)), mask=Field("mask", (4, 4, 1)))
    objective = FlowGRPOObjective(Inpainting(), process, inputs, autoencoder=Pooled(), guidance=None, steps=4)
    state = Trainer(objective, optax.sgd(1e-3), key=jax.random.key(40)).initial_state()
    pixels = np.asarray(jax.random.randint(jax.random.key(41), (2, 4, 4, 3), 0, 256), np.uint8)
    mask = np.zeros((2, 4, 4, 1), np.float32)
    mask[:, :, 2:] = 1

    def reward(images, context):
        return -np.square(images.mean(axis=(1, 2, 3)))

    rollout = FlowRollout(objective, reward, groups=2, steps=4, train_steps=2)
    batch = rollout(state, {"image": pixels, "mask": mask}, jax.random.key(42))
    variables = state.variables
    def rolled(seed):
        return rollout(state, {"image": pixels, "mask": mask}, jax.random.fold_in(jax.random.key(42), seed))

    assert_rescores_as_recorded(objective, variables, rolled, "inpainting", SEEDS['inpainting'])
    step = Step(jnp.asarray(0), jax.random.key(43), None)
    gradients = jax.grad(lambda p: objective.scalar_loss({**variables, "params": p}, batch, step)[0])(
        variables["params"])
    assert np.isfinite(float(optax.tree.norm(gradients)))


def test_transition_matches_released_sampler_and_torch_distribution():
    """The transition's mean, variance, mean log density, that density's velocity gradient and its
    conditional KL, on par with the pinned author code's and torch.distributions' from their float64
    run over the recorded seeds (check 1, tools/parity_flow_grpo.py), the truth Dew's own transition
    in float64, held at every seed to the recording's sketch."""
    seeds = declared_seeds(RECORDED)["transition"]
    draw, recorded, truth = indexed_recording(recording(RECORDED), "transition", seeds,
                                              inputs_of=transition_inputs, truth_fn=jax.jit(dew_transition))
    dew = jax.jit(dew_transition)
    assert_as_exact(lambda index, inputs: dew(inputs), recorded, draw, "released transition", seeds=seeds,
                    truth_fn=truth)



@pytest.mark.mesh
def test_multihost_flow_rollout_reassembles_owned_groups(tmp_path):
    import subprocess
    import sys

    from test_multiprocess import (
        assert_same_parameters,
        dumped_params,
        free_port,
        report_of,
        terminate,
        worker_env,
    )

    worker = Path(__file__).with_name("flow_grpo_worker.py")
    coordinator = f"127.0.0.1:{free_port()}"
    outputs = [tmp_path / f"rank{rank}.json" for rank in range(2)]
    running = [subprocess.Popen(
        [sys.executable, str(worker), str(rank), "2", coordinator, str(output)],
        env={**worker_env(1), "JAX_ENABLE_X64": "0"}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True) for rank, output in enumerate(outputs)]
    try:
        reports = [report_of(process, output, timeout=120)
                   for process, output in zip(running, outputs, strict=True)]
    finally:
        for process in running:
            if process.poll() is None:
                terminate(process)
    baseline_path = tmp_path / "single.json"
    baseline_process = subprocess.Popen(
        [sys.executable, str(worker), "0", "1", "unused", str(baseline_path)],
        env={**worker_env(1), "JAX_ENABLE_X64": "0"}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True)
    baseline = report_of(baseline_process, baseline_path, timeout=120)
    local_rewards = np.concatenate([report["local_rewards"] for report in reports])
    assert local_rewards.shape == (12,)
    raw_callback = np.asarray(reports[0]["callback_rewards"], np.float64)
    np.testing.assert_array_equal(local_rewards, raw_callback)
    raw_groups = raw_callback.reshape(4, 3)
    expected_advantages = ((raw_groups - raw_groups.mean(axis=1, keepdims=True))
                           / (raw_groups.std(axis=1, keepdims=True) + 1e-4)).reshape(-1)
    # A row's arithmetic is its own on one process or two: the two runs agree bit for bit.
    np.testing.assert_array_equal(raw_callback, baseline["callback_rewards"])
    assert reports[0]["metric_rows"] == 4 and reports[0]["preview_rows"] == 4
    assert reports[1]["metric_rows"] == 0 and reports[1]["preview_rows"] == 0
    assert reports[0]["validation_mean"] == baseline["validation_mean"]
    for report, output in zip(reports, outputs, strict=True):
        np.testing.assert_array_equal(np.asarray(report["global_advantages"], np.float32),
                                      expected_advantages.astype(np.float32))
        np.testing.assert_array_equal(report["global_rewards"], raw_callback.astype(np.float32))
        assert not report["x64_enabled"]
        # The reassembled batch rescores as the one process's does, and keeps its recorded densities.
        for name in ("global_scores", "global_old_log_probs"):
            np.testing.assert_array_equal(report[name], baseline[name])
        assert report["updates"] == 1 and report["parameter_change"] > 0
        assert report["reference_unchanged"]
        assert_same_parameters(dumped_params(output), dumped_params(baseline_path))
        # The same rows as each process's own arrays, which process 0 would score for every sample.
        assert "held by this process alone" in report["local_refusal"], report["local_refusal"]



class ConstantVelocity(nn.Module):
    @nn.compact
    def __call__(self, x, temb):
        speed = self.param("speed", nn.initializers.constant(0.1), ())
        return jnp.full_like(x, speed)


def test_flow_evaluation_and_preview_use_live_policy_with_frozen_reference():
    """Evaluation and preview sample the live policy, a constant velocity, not the frozen reference:
    each image is the noise walked by the velocity through the shifted schedule's steps, one in all,
    clipped, on par with that Euler walk in NumPy from its float64 evaluation over drawn speeds and
    keys (check 1)."""
    from dew.diffusion.presets import Flow

    preset = Flow(shift=3)
    process = preset()
    # Sixteen channels: check 1's rows run along the last axis, and a row of one value is sized by
    # that value alone.
    shape = (2, 2, 16)
    objective = FlowGRPOObjective(ConstantVelocity(), preset, InputSpec(Field("image", shape)),
                                  beta=0.1, guidance=None, steps=3)
    reference = objective.init(jax.random.key(51))
    batch = {"image": np.zeros((ROWS, *shape), np.uint8)}

    def draw(seed):
        keys = jax.random.split(jax.random.fold_in(jax.random.key(52), seed))
        return jax.random.uniform(keys[0], (), minval=0.2, maxval=0.9), jax.random.key_data(keys[1])

    def sampled(speed, key):
        step = Step(jnp.asarray(0), jax.random.wrap_key_data(key), reference)
        live = {**reference, "params": {"speed": speed}}
        return {"evaluated": objective.evaluate(live, batch, step).images,
                "previewed": objective.preview(live, batch, step).images}

    def walked(speed, key):
        with jax.enable_x64(new_val=False):
            noise_key, _ = jax.random.split(jax.random.wrap_key_data(key))
            noise = np.asarray(process.noise(noise_key, (ROWS, *shape)))
            times = np.asarray(process.times(objective.steps))
        dtype = np.asarray(speed).dtype
        sigmas = [3 * t / (1 + 2 * t) for t in times.astype(dtype)]
        images = noise.astype(dtype)
        for sigma, following in itertools.pairwise(sigmas):
            images = images + (following - sigma) * speed
        images = np.clip(images, -1, 1)
        return {"evaluated": images, "previewed": images[:4]}

    assert sampled(*draw(0))["previewed"].shape == (4, *shape)
    assert_as_exact(sampled, walked, draw, "live policy", seeds=SEEDS['live policy'])



def test_mixed_precision_transition_keeps_density_arithmetic_in_float32():
    """A bf16 transition samples and scores in float32: its draws and their log densities on par with
    the Gaussian's in float32 from its float64 evaluation on the same normals, over drawn means,
    variances and keys (check 1); a zero variance draws its mean exactly and has no density."""
    from dew.sampling import GaussianTransition

    def draw(seed):
        keys = jax.random.split(jax.random.fold_in(jax.random.key(61), seed), 3)
        return (jax.random.normal(keys[0], (ROWS, 2)).astype(jnp.bfloat16),
                jax.random.uniform(keys[1], (ROWS,), minval=0.05, maxval=0.5).astype(jnp.bfloat16),
                jax.random.key_data(keys[2]))

    @jax.jit
    def transitioned(mean, variance, key):
        transition = GaussianTransition(mean, variance)
        sampled = transition.sample(jax.random.wrap_key_data(key))
        return {"sample": sampled, "log_prob": transition.log_prob(sampled)}

    def gaussian(mean, variance, key):
        with jax.enable_x64(new_val=False):
            normal = np.asarray(jax.random.normal(jax.random.wrap_key_data(key), np.shape(mean)))
        dtype = np.float64 if np.asarray(mean).dtype == np.float64 else np.float32
        mean, variance = np.asarray(mean, dtype), np.asarray(variance, dtype)
        sampled = mean + np.sqrt(variance)[:, None] * normal.astype(dtype)
        residual = sampled - mean
        density = (-0.5 * (residual**2 / variance[:, None] + np.log(2 * np.pi * variance[:, None]))).sum(-1)
        return {"sample": sampled, "log_prob": density}

    assert_as_exact(transitioned, gaussian, draw, "mixed precision", seeds=SEEDS['mixed precision'])
    mean = jnp.asarray([[0.2, -0.3], [0.7, 0.4]], jnp.bfloat16)
    transition = GaussianTransition(mean, jnp.asarray([0, 0.25], jnp.bfloat16))
    sampled = transition.sample(jax.random.key(61))
    np.testing.assert_array_equal(sampled[0], mean[0].astype(jnp.float32))
    assert sampled.dtype == jnp.float32
    assert np.isnan(float(transition.log_prob(sampled)[0]))



def test_float64_callback_distinctions_reach_a_real_policy_update():
    from dew.data.dataset import Loader
    from dew.nn.backbones.dit import SimpleDiT

    count = jax.device_count()
    raw = np.tile(np.asarray([1e6 + 0.01, 1e6 + 0.02, 1e6 + 0.03], np.float64), count)
    groups = raw.reshape(count, 3)
    oracle = ((groups - groups.mean(axis=1, keepdims=True))
              / (groups.std(axis=1, keepdims=True) + 1e-4)).reshape(-1)
    model = SimpleDiT(output_channels=1, patch_size=2, emb_features=8,
                      num_layers=1, num_heads=2, mlp_ratio=2)
    objective = FlowGRPOObjective(model, Process(FlowMatchingScheduler(), FlowMatchPredictionTransform()),
                                  InputSpec(Field("image", (4, 4, 1))), guidance=None)
    rollout = FlowRollout(objective, lambda images, context: raw, groups=3, steps=3)
    trainer = Trainer(objective, optax.sgd(1e-3), key=jax.random.key(91), rollout=rollout)
    initial = trainer.place()[0]
    source = {"image": np.zeros((count, 4, 4, 1), np.uint8)}
    collected = rollout(initial, source, jax.random.key(92))
    data = Loader(train=lambda partition: itertools.repeat(source), val=None, records=count, batch=count)
    final = trainer.fit(data, steps=1, log_every=1)
    assert int(final.updates) == 1
    change = float(optax.tree.norm(jax.tree.map(
        lambda a, b: a - b, final.variables["params"], initial.variables["params"])))
    assert np.isfinite(change) and change > 1e-6
    np.testing.assert_array_equal(collected["advantages"], oracle.astype(np.float32))
    np.testing.assert_array_equal(collected["rewards"], raw)
    assert collected["rewards"].dtype == np.float64
    np.testing.assert_array_equal(collected["transition_mask"],
                                   np.broadcast_to((oracle != 0)[:, None], (count * 3, 2)))



def test_conditioned_prompt_only_evaluation_preview_and_trainer_consumers():
    from dew.artifacts import ImageGrid
    from dew.data.dataset import Loader
    from dew.inputs import CharTable, Condition
    from dew.nn.backbones.dit import SimpleDiT

    class PixelMean:
        name = "pixel_mean"
        reads = ImageGrid

        def __init__(self):
            self.images = []

        def __call__(self, artifact, batch):
            images = np.asarray(artifact.images)
            self.images.append(images.copy())
            return float(images.sum(dtype=np.float64)), images.size

        def merge(self, left, right):
            return left[0] + right[0], left[1] + right[1]

        def finalize(self, values):
            return values[0] / values[1]

    count = jax.device_count()
    inputs = InputSpec(Field("image", (4, 4, 1)), {
        "textcontext": Condition(CharTable.from_pretrained(tokens=3, features=4))})
    prompts = inputs.tokenize([f"p{row}" for row in range(count)])
    model = SimpleDiT(output_channels=1, patch_size=2, emb_features=8,
                      num_layers=1, num_heads=2, mlp_ratio=2)
    process = Process(FlowMatchingScheduler(shift=2), FlowMatchPredictionTransform())
    objective = FlowGRPOObjective(model, process, inputs, guidance=CFG(1.5), beta=0.1, steps=3)
    rollout = FlowRollout(objective, lambda images, batch: images.mean((1, 2, 3)), groups=2, steps=3)
    tracker, metric = RecordingTracker(), PixelMean()
    trainer = Trainer(objective, optax.sgd(1e-3), key=jax.random.key(101), rollout=rollout, tracker=tracker)
    initial = trainer.place()[0]
    step = Step(initial.microstep, jax.random.key(102), initial.averaged)
    evaluated = objective.evaluate(initial.variables, prompts, step)
    previewed = objective.preview(initial.variables, prompts, step)
    assert evaluated.images.shape == (count, 4, 4, 1)
    many = inputs.tokenize([f"p{row}" for row in range(16)])
    many_tokens = {keyword: many[condition.field] for keyword, condition in inputs.conditions.items()}

    def draw(seed):
        return (jax.random.key_data(jax.random.fold_in(jax.random.key(102), seed)),)

    def evaluated_with(key):
        step = Step(initial.microstep, jax.random.wrap_key_data(key), initial.averaged)
        return objective.evaluate(initial.variables, many, step).images

    def sampled(variables):
        def run(key):
            encoders = variables["encoders"]
            denoiser = process.denoiser(model, objective.model_variables(variables),
                                        objective.encode(encoders, many_tokens), objective.encode(encoders))
            with jax.enable_x64(new_val=False):
                noise_key, sample_key = jax.random.split(jax.random.wrap_key_data(key))
                noise = np.asarray(process.noise(noise_key, (16, *objective.latent_shape)))
            dtype = jax.tree.leaves(variables)[0].dtype
            images = sample(denoiser, jnp.asarray(noise, dtype), objective.steps, solver=objective.solver,
                            guidance=objective.guidance, key=sample_key)
            return jnp.clip(images, -1, 1)
        return run

    def truth(key):
        with jax.enable_x64(new_val=True):
            return sampled(widened(initial.variables))(key)

    assert_as_exact(evaluated_with, sampled(initial.variables), draw, "prompt evaluation",
                    seeds=SEEDS['prompt evaluation'], truth_fn=truth)
    assert previewed.images.shape == (min(4, count), 4, 4, 1)
    assert len(previewed.captions) == min(4, count)
    data = Loader(
        train=lambda partition: itertools.repeat(prompts),
        val=lambda partition: iter((prompts, prompts)),
        records=count,
        batch=count,
    )
    final = trainer.fit(data, steps=1, log_every=1, eval_every=1, metrics=(metric,), preview=True)
    assert int(final.updates) == 1
    observed = np.concatenate(metric.images)
    assert observed.shape == (count * 2, 4, 4, 1)
    logged = {name: value for _, scalars in tracker.scalars for name, value in scalars.items()}
    # The metric's own sums, batch by batch in float64, merged in order, over its count.
    totals = [float(images.sum(dtype=np.float64)) for images in metric.images]
    assert logged["val/pixel_mean"] == sum(totals) / observed.size
    assert logged["evaluation/records"] == count * 2
    previews = [value for _, value in tracker.artifacts if not isinstance(value, RECORD_TYPES)]
    assert previews[0].images.shape == (min(4, count), 4, 4, 1)
