"""Flow-GRPO against Gaussian algebra and the authors' SDE convention.

Reference: arXiv:2505.05470v5, equations 8-9, and
https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py
The reference averages coordinate log densities for its policy ratio. These
checks retain the joint density until the objective performs that reduction.
Each density, loss and gradient is on par with its closed form from the
float64 evaluation over drawn inputs (check 1, tests/reference_error.py), the
closed form evaluated on the walk's own states from the policy's own
velocities, a walk's recorded densities and their rescoring alike; what is one
computation, or exact arithmetic, is held exactly.
"""

import itertools
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from certified import assert_certified
from flax import linen as nn
from recording import RecordingTracker
from reference_error import (
    as_given,
    assert_as_exact,
    assert_as_exact_as_the_reference,
    assert_fp32_reduction_bound,
    chain_roundings,
    declared_seeds,
    indexed_recording,
    recording,
    widened,
    with_gradients,
)

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

SEEDS = {'guided walk': 8, 'converted': 8, 'rescored walk': 16, 'clipping': 32, 'kl': 8, 'rollout': 64,
         'inpainting': 32, 'live policy': 64, 'mixed precision': 32, 'prompt evaluation': 16}
"""Check 1's declared seeds for each site of the file, sized by `seeds_for` (on armada)."""
ROWS = 64
"""A drawn batch's rows: enough that a few seeds resolve a site."""
BATCHES = 16
"""The batches a draw of a site holds whose outputs are one number a batch."""


def walked_in_float32(walk, *inputs):
    """`walk` on `inputs`, float arrays, at float32 with x64 off, as the arm under test walked them: a
    float64 truth evaluates the arm's own float32 states."""
    with jax.enable_x64(new_val=False):
        return walk(*(jnp.asarray(value, jnp.float32) for value in inputs))


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


def sde_density(x, action, sigma, following, velocity, noise, dtype):
    """The authors' SDE log density of a transition from `x` to `action` [rows, ...] between the noise
    rates `sigma` and `following` (one each, or one a row) at the policy's `velocity` there, its noise
    level `noise`, in NumPy at `dtype`: their `sde_step_with_logprob`, summed over a state's values."""
    x, action, velocity = (np.asarray(value, dtype) for value in (x, action, velocity))
    shape = (-1,) + (1,) * (x.ndim - 1)
    sigma, following = (np.asarray(rate, dtype).reshape(shape) for rate in (sigma, following))
    dt = following - sigma
    g_squared = noise**2 * sigma / (1 - np.where(sigma == 1, following, sigma))
    mean = x + (velocity + g_squared / (2 * sigma) * (x + (1 - sigma) * velocity)) * dt
    variance = g_squared * -dt
    density = -0.5 * ((action - mean)**2 / variance + np.log(2 * math.pi * variance))
    return density.sum(tuple(range(1, x.ndim)))


def walk_density(trajectory, rate, velocity, noise, dtype):
    """`sde_density` of each recorded transition of a walk, from its own states, at the noise rate
    `rate(t)` of each of its times and the velocity `velocity(index, x, sigma)` at transition `index`'s
    state `x`, in NumPy at `dtype`."""
    found = []
    for index in range(trajectory.states.shape[1] - 1):
        t, s = np.asarray(trajectory.times[index:index + 2], dtype)
        sigma, following = rate(t), rate(s)
        x = np.asarray(trajectory.states[:, index], dtype)
        found.append(sde_density(x, trajectory.states[:, index + 1], sigma, following,
                                 velocity(index, x, sigma), noise, dtype))
    return np.stack(found, 1)


def guided_density(trajectory, offset, dtype):
    """`walk_density` of a walk of AffineVelocity under FlowMatchingScheduler(shift=3) and CFG(2) inside
    (0.1, 0.8)."""
    offset = np.asarray(offset, dtype)

    def velocity(index, x, sigma):
        scale = 2 if 0.1 <= index / 4 <= 0.8 else 1
        return 0.25 * x + 0.1 * sigma + scale * offset

    return walk_density(trajectory, lambda t: 3 * t / (1 + 2 * t), velocity, 0.5, dtype)


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
                    as_given(lambda start, offset: guided_density(walk(start, offset), offset, np.float32)),
                    drawn_rows((2,), (1,)), "guided walk", seeds=SEEDS['guided walk'], truth_fn=truth)
    deterministic = FlowSDE(0).trajectory(denoiser(offset), x, 5, key=key)
    assert np.isnan(deterministic.log_probs).all()
    assert not np.asarray(deterministic.stochastic).any()



def recorded_and_rescored(walk, rescore, variables):
    """A walk's recorded densities and their rescoring under `variables` (`rescore`, the objective's
    `log_probs`), from a start."""
    def scored(start):
        trajectory = walk(start)
        return {"recorded": trajectory.log_probs,
                "rescored": rescore(variables, trajectory_batch(trajectory, np.ones(ROWS, np.float32)))}
    return scored


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

    def velocity(index, x, t):
        raw = 0.25 * x + 0.1 * t
        if prediction == "clean":
            return (x - raw) / t
        variance = (1 - t)**2 + t**2
        return ((2 * t - 1) * x + np.sqrt(variance) * raw) / variance

    def rectified(trajectory, dtype):
        density = walk_density(trajectory, lambda t: t, velocity, 0.5, dtype)
        return {"recorded": density, "rescored": density}

    assert_as_exact(recorded_and_rescored(walk, rescore, variables),
                    as_given(lambda start: rectified(walk(start), np.float32)), drawn_rows((2,)),
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
    From t = 0.9 the same prediction walks compiled, and the densities the walk recorded and each
    transition's rescoring, compiled on its own, are on par with the authors' Gaussian at the
    prediction's velocity on the walk's own states from its float64 evaluation over drawn starts
    (check 1)."""
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
            record=lambda following, state, denoised: (following, state[0]))
        return jnp.concatenate((start[None], states)), log_probs

    assert np.isfinite(np.asarray(walk(x)[0])).all()

    @jax.jit
    def rescored(states):
        def one(latent, action, t, s):
            t, s = jnp.full((latent.shape[0],), t), jnp.full((latent.shape[0],), s)
            clean, noise = denoise(latent, t)
            return solver.transition(latent, t, s, clean, noise, process).log_prob(action)
        return jnp.stack([one(states[index], states[index + 1], times[index], times[index + 1])
                          for index in range(4)])

    def velocity(x, t):
        # AffineVelocity's epsilon at gain 0.25, its clean sample, and through the boundary
        # (timestep_scaling 10, sigma_data 0.5) the clean sample and noise it returns instead.
        raw = 0.25 * x + 0.1 * t
        clean = (x - t * raw) / (1 - t)
        if isinstance(prediction, ConsistencyBoundary):
            scaled = 10 * t
            clean = scaled / np.sqrt(scaled**2 + 0.25) * clean + 0.25 / (scaled**2 + 0.25) * x
            return (x - clean * (1 - t)) / t - clean
        return raw - clean

    def closed(states, dtype):
        grid = np.asarray(times, dtype)
        density = np.stack([sde_density(states[index], states[index + 1], grid[index], grid[index + 1],
                                        velocity(np.asarray(states[index], dtype), grid[index]), 0.5, dtype)
                            for index in range(4)])
        return {"recorded": density, "rescored": density}

    def scored(start):
        states, log_probs = walk(start)
        return {"recorded": log_probs, "rescored": rescored(states)}

    assert_as_exact(scored, as_given(lambda start: closed(walk(start)[0], np.float32)), drawn_rows((2,)),
                    f"{type(prediction).__name__} rescored walk", seeds=SEEDS['rescored walk'],
                    truth_fn=lambda start: closed(walked_in_float32(walk, start)[0], np.float64))


def clipped_terms(gain, frozen, states, times, advantages):
    """Flow-GRPO's clipped policy term of each transition of a policy of `gain` against the walk's
    `states` under gain 0.25, its KL to a reference of gain `frozen`, its likelihood ratio and its
    advantage, its means in closed form, in jnp at `gain`'s dtype: the clip range 0.001, the
    advantages clipped to 2."""
    dtype = jnp.asarray(gain).dtype
    latent, following = jnp.asarray(states[:, :-1], dtype), jnp.asarray(states[:, 1:], dtype)
    t, s = jnp.asarray(times[:-1], dtype), jnp.asarray(times[1:], dtype)
    g2 = 0.6**2 * t / (1 - jnp.where(t == 1, s, t))
    variance = (g2 * (t - s))[None, :]
    coefficient = ((1 + g2 * (1 - t) / (2 * t)) * (s - t))[None, :, None]
    offset = latent * (1 + g2 / (2 * t) * (s - t))[None, :, None]

    def mean(gain):
        return offset + (gain * latent + 0.1 * t[None, :, None]) * coefficient

    old, current, reference = mean(jnp.asarray(0.25, dtype)), mean(gain), mean(jnp.asarray(frozen, dtype))
    log_ratio = (-jnp.square(following - current) + jnp.square(following - old)).mean(-1) / (2 * variance)
    ratio = jnp.exp(log_ratio)
    advantage = jnp.clip(jnp.asarray(advantages, dtype), -2, 2)[:, None]
    policy = jnp.maximum(-advantage * ratio, -advantage * jnp.clip(ratio, 0.999, 1.001))
    kl = jnp.square(current - reference).mean(-1) / (2 * variance)
    return policy, kl, ratio, advantage


def clipped_objective(gain, frozen, states, times, advantages, mask, *, beta):
    """`clipped_terms`' loss, `beta` its KL's weight, averaged over `mask`'s transitions."""
    policy, kl, _, _ = clipped_terms(gain, frozen, states, times, advantages)
    weights = jnp.asarray(mask, policy.dtype)
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
        # Each batch's loss pulls back one, the cotangent of a loss, held in the draw as the gate takes it.
        return (0.5 * jax.random.normal(keys[0], (BATCHES, ROWS, 2)),
                jax.random.uniform(keys[1], (BATCHES, ROWS), minval=-3, maxval=3),
                jax.random.bernoulli(keys[2], 0.6, (BATCHES, ROWS, 3)), jnp.ones(BATCHES, jnp.float32))

    def batch_of(start, advantages, mask):
        batch = trajectory_batch(walk(start), advantages, mask)
        return {**batch, "old_log_probs": jnp.where(mask, batch["old_log_probs"], jnp.nan)}

    @jax.jit
    @jax.vmap
    def scored(start, advantages, mask, one):
        batch = batch_of(start, advantages, mask)

        def loss(gain, objective, step):
            return objective.scalar_loss({**current, "params": {"gain": gain}}, batch, step)[0]

        scoring = with_gradients(lambda gain: loss(gain, objective, step))
        value, gradient = scoring(current["params"]["gain"], one)
        return {"loss": value, "gradient": gradient,
                "policy": loss(current["params"]["gain"], without_reference, step.replace(ema=None))}

    # The policies' gains as their parameters hold them, at float32 or widened from it.
    gains = (current["params"]["gain"], reference["params"]["gain"])

    def closed(states, times, advantages, mask, one, gain, frozen):
        terms = (frozen, states, times, advantages, mask)
        value, gradient = with_gradients(lambda gain: clipped_objective(gain, *terms, beta=0.2))(gain, one)
        return {"loss": value, "gradient": gradient,
                "policy": clipped_objective(gain, *terms, beta=0.0)}

    def reference_arm(start, advantages, mask, one):
        trajectory = jax.vmap(walk)(start)
        return jax.vmap(closed, (0, 0, 0, 0, 0, None, None))(trajectory.states, trajectory.times, advantages,
                                                             mask, one, *gains)

    @as_given
    def truth(start, advantages, mask, one):
        trajectory = walked_in_float32(jax.vmap(walk), start)
        with jax.enable_x64(new_val=True):
            return jax.vmap(closed, (0, 0, 0, 0, 0, None, None))(
                np.asarray(trajectory.states, np.float64), np.asarray(trajectory.times, np.float64),
                advantages, mask, one, *widened(gains))

    assert_as_exact(scored, reference_arm, draw, "clipping", seeds=SEEDS['clipping'], truth_fn=truth)

    rescore = jax.jit(objective.log_probs)

    def walked(trajectory, dtype):
        density = walk_density(trajectory, lambda t: t, lambda index, x, t: 0.25 * x + 0.1 * t, 0.6, dtype)
        return {"recorded": density, "rescored": density}

    assert_as_exact(recorded_and_rescored(walk, rescore, old),
                    as_given(lambda start: walked(walk(start), np.float32)),
                    drawn_rows((2,)), "rescored clipping walk", seeds=SEEDS['rescored walk'],
                    truth_fn=lambda start: walked(walked_in_float32(walk, start), np.float64))

    start, advantages, mask, _ = (value[0] for value in draw(0))
    trajectory = walk(start)
    with jax.enable_x64(new_val=True):
        policy, _, ratio, advantage = clipped_terms(
            *widened(gains), np.asarray(trajectory.states, np.float64),
            np.asarray(trajectory.times, np.float64), np.asarray(advantages, np.float64))
        binds = np.asarray(policy != -advantage * ratio)
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
    """The rollout's recorded likelihoods and their rescoring are on par with
    the authors' Gaussian at the DiT's own guided velocities over rollouts
    (check 1), guidance included: the rollout walks 4 points and guides its
    step 1 of 3, inside (0.3, 0.6), while the objective's own walk of 7 points
    would not guide a step 1 of 6, so the rescoring reads the rollout's step
    count off the batch. Its group advantages are the float64 normalization's,
    rounded once, and its timesteps the 4-point walk's first two, bit for
    bit."""
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

    def guided(variables, batch, dtype):
        weights, encoders = objective.model_variables(variables), variables["encoders"]
        given = objective.encode(encoders, {keyword: batch[condition.field]
                                            for keyword, condition in inputs.conditions.items()})
        blank = jax.tree.map(lambda null, value: jnp.broadcast_to(null, value.shape),
                             objective.encode(encoders), given)
        steps = int(batch["rollout_steps"][0]) - 1
        # The DiT computing at `dtype`: its own float32 floor would round a float64 truth's times.
        apply = jax.jit(model.clone(dtype=dtype).apply)

        def velocity(column, x, sigma):
            # The rollout's step of this column, off its time on the rollout's own grid of `steps` steps.
            index = int(np.rint((1 - float(batch["timesteps"][0, column])) * steps))
            scale = 1.5 if 0.3 <= index / steps <= 0.6 else 1.0
            time = jnp.asarray(sigma.reshape(-1) * 1000)
            conditional, unconditional = (
                np.asarray(apply(weights, jnp.asarray(x), time, **conditions), dtype)
                for conditions in (given, blank))
            return unconditional + scale * (conditional - unconditional)

        return recorded_density(batch, lambda t: 2 * t / (1 + t), velocity, 0.7, dtype)

    assert_rescores_as_walked(objective, variables, rolled, guided, "rollout", SEEDS['rollout'])
    loss, _ = objective.scalar_loss(variables, batch, Step(jnp.asarray(0), jax.random.key(24), None))
    gradients = jax.grad(lambda p: objective.scalar_loss({**variables, "params": p}, batch,
        Step(jnp.asarray(0), jax.random.key(24), None))[0])(variables["params"])
    norm = float(optax.tree.norm(gradients))
    assert np.isfinite(float(loss)) and np.isfinite(norm) and norm > 1e-5


def recorded_density(batch, rate, velocity, noise, dtype):
    """`sde_density` of each transition a rollout's batch records, from its own states and times, at the
    noise rate `rate(t)` of each time and the velocity `velocity(column, x, sigma)` at column `column`'s
    state `x`, `sigma` its rates shaped against it, in NumPy at `dtype`."""
    columns = []
    for column in range(np.shape(batch["timesteps"])[1]):
        x = np.asarray(batch["latents"][:, column], dtype)
        shape = (-1,) + (1,) * (x.ndim - 1)
        sigma, following = (rate(np.asarray(batch[name][:, column], dtype)).reshape(shape)
                            for name in ("timesteps", "next_timesteps"))
        columns.append(sde_density(x, batch["next_latents"][:, column], sigma, following,
                                   velocity(column, x, sigma), noise, dtype))
    return np.stack(columns, 1)


def assert_rescores_as_walked(objective, variables, rollout_of, density, label: str, seeds: int):
    """Check 1: the densities a rollout recorded (`rollout_of(seed)`, a batch) and the objective's
    rescoring of its transitions, on par with `density(variables, batch, dtype)`, the authors' Gaussian
    at the policy's own velocities on the rollout's own states, from its float64 evaluation over
    rollouts drawn from their keys."""
    rescore = jax.jit(objective.log_probs)

    def closed(variables, dtype):
        @as_given
        def run(batch):
            found = density(variables, batch, dtype)
            return {"recorded": found, "rescored": found}
        return run

    # The recorded log probabilities beside the rescored ones, and the widened weights' truth, on the host.
    assert_as_exact(as_given(lambda batch: {"recorded": batch["old_log_probs"],
                                            "rescored": rescore(variables, batch)}),
                    closed(variables, np.float32), lambda seed: (rollout_of(seed),), label, seeds=seeds,
                    truth_fn=as_given(lambda batch: closed(widened(variables), np.float64)(batch)))


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
    likelihoods and their rescoring are on par with the authors' Gaussian at the model's velocity of
    those inputs over rollouts (check 1), and a policy step through them has a gradient."""
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

    def inpainted(variables, batch, dtype):
        gain = np.asarray(objective.model_variables(variables)["params"]["gain"], dtype)
        spatial = {name: np.asarray(value, dtype) for name, value in batch["spatial"].items()}
        held = 0.3 * spatial["masked_image"] * (1 - spatial["mask"])
        return recorded_density(batch, lambda t: t, lambda column, x, sigma: gain * x + held + 0.1 * sigma,
                                0.7, dtype)

    assert_rescores_as_walked(objective, variables, rolled, inpainted, "inpainting", SEEDS['inpainting'])
    step = Step(jnp.asarray(0), jax.random.key(43), None)
    gradients = jax.grad(lambda p: objective.scalar_loss({**variables, "params": p}, batch, step)[0])(
        variables["params"])
    assert np.isfinite(float(optax.tree.norm(gradients)))


def test_transition_matches_released_sampler_and_torch_distribution():
    """The transition's mean, variance, mean log density, that density's velocity gradient and its
    conditional KL, on par with the pinned author code's and torch.distributions' from their float64
    run over the recorded seeds (check 1, tools/parity_flow_grpo.py), the truth Dew's own transition
    in float64, held at every seed to the recording's sketch; and every operation that computes them
    within its own operation's bound on each of those draws (check 2, tests/certified.py)."""
    seeds = declared_seeds(RECORDED)["transition"]
    draw, recorded, truth = indexed_recording(recording(RECORDED), "transition", seeds,
                                              inputs_of=transition_inputs, truth_fn=jax.jit(dew_transition))
    dew = jax.jit(dew_transition)
    assert_as_exact(lambda index, inputs: dew(inputs), recorded, draw, "released transition", seeds=seeds,
                    truth_fn=truth)
    # Check 2: every operation of the transition and its velocity gradient, one at a time, within its
    # own contract on every draw check 1 judged.
    assert_certified(dew_transition, lambda seed: (transition_inputs(seed),), "released transition",
                     seeds=seeds)



@pytest.mark.mesh
def test_a_clip_reward_scores_each_sample_against_its_prompts_caption():
    """A rollout's reward reads the batch's captions as host text beside its placed conditions
    (`split_fields`), each repeated into its prompt's group as the conditions are: a CLIP reward
    (`CLIPScore`) scores each sample against its own prompt's caption through CLIP's own tokenizer,
    exactly as the metric scores those images and captions, whatever tokenizer the conditions read."""
    from dew.artifacts import ImageGrid
    from dew.eval import CLIPScore
    from dew.inputs import CharTable, Condition
    from dew.nn.backbones.dit import SimpleDiT

    process = Process(FlowMatchingScheduler(shift=2), FlowMatchPredictionTransform())
    inputs = InputSpec(Field("image", (8, 8, 3)), {
        "textcontext": Condition(CharTable.from_pretrained(tokens=3, features=4))})
    model = SimpleDiT(output_channels=3, patch_size=2, emb_features=8, num_layers=1, num_heads=2, mlp_ratio=2)
    objective = FlowGRPOObjective(model, process, inputs, steps=3)
    state = Trainer(objective, optax.sgd(1e-3), key=jax.random.key(21)).initial_state()
    clip = CLIPScore(str(Path(__file__).resolve().parent / "fixtures" / "clip" / "tiny"))
    seen = {}

    def reward(images, context):
        seen["images"] = np.asarray(images)
        seen["captions"] = [str(caption) for caption in context["captions"]]
        return np.asarray(clip.fn(ImageGrid(images), context), np.float64)

    captions = ["red", "blue"]
    prompts = {**inputs.tokenize(captions), "captions": np.asarray(captions)}
    batch = FlowRollout(objective, reward, groups=2, steps=3)(state, prompts, jax.random.key(3))
    assert seen["captions"] == ["red", "red", "blue", "blue"]
    scored = np.asarray(clip.fn(ImageGrid(seen["images"]), {"captions": np.asarray(seen["captions"])}))
    np.testing.assert_array_equal(np.asarray(batch["rewards"]), scored.astype(np.float32))


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
    # Rank zero's reward reads every sample's caption, each prompt's group in order, gathered from the
    # host that holds it, as the one process reads its own.
    grouped = [caption for caption in ("red", "blue", "green", "gold") for _ in range(3)]
    assert reports[0]["callback_captions"] == baseline["callback_captions"] == grouped
    assert reports[0]["metric_rows"] == 4 and reports[0]["preview_rows"] == 4
    assert reports[1]["metric_rows"] == 0 and reports[1]["preview_rows"] == 0
    assert reports[0]["validation_mean"] == baseline["validation_mean"]
    for report, output in zip(reports, outputs, strict=True):
        np.testing.assert_array_equal(np.asarray(report["global_advantages"], np.float32),
                                      expected_advantages.astype(np.float32))
        np.testing.assert_array_equal(report["global_rewards"], raw_callback.astype(np.float32))
        assert not report["x64_enabled"]
        # The reassembled batch rescores as the one process's does, and keeps its recorded densities;
        # the one process's rescoring and recording of this model and process are check 1's, against
        # the authors' Gaussian (test_flow_rollout_groups_rewards_selects_steps_and_preserves_likelihoods).
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

    @as_given
    def sampled(speed, key):
        step = Step(jnp.asarray(0), jax.random.wrap_key_data(key), reference)
        live = {**reference, "params": {"speed": speed}}
        return {"evaluated": objective.evaluate(live, batch, step).images,
                "previewed": objective.preview(live, batch, step).images}

    @as_given
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

    @as_given
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
    untrained = FlowGRPOObjective(model, process, inputs, guidance=CFG(1.5), beta=0.1, steps=3).init(
        jax.random.key(100))
    # Noise in every weight, the zeroed output head's too, so the evaluation reads its prompts.
    leaves, tree = jax.tree.flatten(untrained["params"])
    noise = jax.random.split(jax.random.key(103), len(leaves))
    params = jax.tree.unflatten(tree, [leaf + 0.1 * jax.random.normal(key, leaf.shape, leaf.dtype)
                                       for leaf, key in zip(leaves, noise, strict=True)])
    objective = FlowGRPOObjective(model, process, inputs, guidance=CFG(1.5), beta=0.1, steps=3,
                                  variables={**untrained, "params": params})
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
        # Each image one row of check 1, its 16 values.
        return objective.evaluate(initial.variables, many, step).images.reshape(16, -1)

    def walked(variables, dtype):
        """The live policy's Euler walk in NumPy at `dtype` from the evaluation's noise, the DiT's own
        raw outputs guided by CFG(1.5) at every step of the shifted grid, clipped."""
        weights, encoders = objective.model_variables(variables), variables["encoders"]
        given = objective.encode(encoders, many_tokens)
        blank = jax.tree.map(lambda null, value: jnp.broadcast_to(null, value.shape),
                             objective.encode(encoders), given)
        # The DiT computing at `dtype`: its own float32 floor would round a float64 truth's times.
        apply = jax.jit(model.clone(dtype=dtype).apply)

        @as_given
        def run(key):
            with jax.enable_x64(new_val=False):
                noise_key, _ = jax.random.split(jax.random.wrap_key_data(key))
                images = np.asarray(process.noise(noise_key, (16, *objective.latent_shape)), dtype)
            times = 1 - np.arange(objective.steps, dtype=dtype) / (objective.steps - 1)
            sigmas = 2 * times / (1 + times)
            for sigma, following in itertools.pairwise(sigmas):
                time = jnp.full((16,), sigma * 1000)
                conditional, unconditional = (
                    np.asarray(apply(weights, jnp.asarray(images), time, **conditions), dtype)
                    for conditions in (given, blank))
                images = images + (following - sigma) * (unconditional + 1.5 * (conditional - unconditional))
            return np.clip(images, -1, 1).reshape(16, -1)
        return run

    assert_as_exact(evaluated_with, walked(initial.variables, np.float32), draw, "prompt evaluation",
                    seeds=SEEDS['prompt evaluation'],
                    truth_fn=lambda key: walked(widened(initial.variables), np.float64)(key))
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


def test_a_transitions_density_and_kl_hold_at_float32s_largest_variances():
    """A transition's density and KL stay finite and on SciPy's float64 values at a variance near float32's
    largest, where `2 pi` or 2 times it overflowed to infinity, and its KL at one near its least, where
    an error over it overflows and half the quotient fits: the density of a state at its mean under a
    variance of 1e38 is `-n / 2 (log 2 pi + log 1e38)` (JAX's isotropic `multivariate_normal.logpdf`), and
    the KL to a mean `d` away is `d**2 / (2 variance)`."""
    from scipy.stats import multivariate_normal as reference

    from dew.sampling.flow import GaussianTransition

    variance = np.asarray([1e38, 2.5e38], np.float32)
    mean = np.zeros((2, 3, 4), np.float32)
    transition = GaussianTransition(jnp.asarray(mean), jnp.asarray(variance))
    density = np.asarray(transition.log_prob(jnp.asarray(mean)))
    assert np.all(np.isfinite(density))
    truth = np.asarray([reference.logpdf(np.zeros(12), np.zeros(12), float(value)) for value in variance])
    # The oracle rule for a fixed input: within float32's rounding over each call's chain of roundings of
    # its float64 value's terms (`_held_by_its_roundings`).
    half_log = 6 * (abs(math.log(2 * math.pi)) + np.abs(np.log(variance.astype(np.float64))))
    _held_by_its_roundings(density, truth, half_log, transition.log_prob, jnp.asarray(mean))
    away = jnp.full_like(jnp.asarray(mean), 1e18)
    kl = np.asarray(transition.kl(away))
    assert np.all(np.isfinite(kl))
    wide = 12 * 1e36 / (2 * variance.astype(np.float64))
    _held_by_its_roundings(kl, wide, wide, transition.kl, away)
    # A variance near float32's least: the KL, about 2e38, fits, where its twice the quotient would not.
    narrow = np.asarray([2e-38, 3e-38], np.float32)
    near = GaussianTransition(jnp.asarray(mean), jnp.asarray(narrow))
    step = jnp.full_like(jnp.asarray(mean), np.float32(np.sqrt(8 / 12)))
    kl = np.asarray(near.kl(step))
    assert np.all(np.isfinite(kl))
    small = 12 * float(np.float32(np.sqrt(8 / 12))) ** 2 / (2 * narrow.astype(np.float64))
    _held_by_its_roundings(kl, small, small, near.kl, step)


def _held_by_its_roundings(found, truth, magnitude, fn, *arguments):
    """The oracle rule for a fixed input, which no draw varies: `found` within float32's rounding of
    `magnitude` (the float64 value's terms' size) over `fn`'s longest chain of roundings
    (`chain_roundings`), and finite."""
    assert np.all(np.isfinite(found))
    assert_fp32_reduction_bound(found, truth, np.broadcast_to(magnitude, np.shape(truth)),
                                chain_roundings(jax.make_jaxpr(fn)(*arguments)))


def _kl_truth(mean, reference, variance, mean_tangent, reference_tangent, variance_tangent, cotangent):
    """The KL, its JVP and its VJP to the mean and the variance in float64, on the float32 inputs."""
    wide = [value.astype(np.float64) for value in
            (mean, reference, variance, mean_tangent, reference_tangent, variance_tangent, cotangent)]
    difference, moving = wide[0] - wide[1], wide[3] - wide[4]
    divergence = np.sum(difference**2, axis=1) / (2 * wide[2])
    return {"KL": divergence,
            "JVP": np.sum(difference * moving, axis=1) / wide[2] - divergence * wide[5] / wide[2],
            "VJP to the mean": wide[6][:, None] * difference / wide[2][:, None],
            "VJP to the variance": -wide[6] * divergence / wide[2]}


def _plain_kl(mean, reference, variance, mean_tangent, reference_tangent, variance_tangent, cotangent):
    """The reference arm (the gate's ruling on rows 1722 and 1778): the KL, its JVP along every tangent at
    once and its VJPs as the plain float32 textbook formulas compute them, `|m - r|**2 / (2 v)` as
    written, `sum((m - r) * (t_m - t_r)) / v - KL * t_v / v`, `c (m - r) / v` and `-c KL / v`, with no
    alignment or scaling of any kind, compiled as Dew's arm is."""
    def plain(m, r, v, tm, tr, tv, ct):
        d = m - r
        axes = tuple(range(1, d.ndim))
        kl = jnp.square(d).sum(axis=axes) / (2 * v)
        return {"KL": kl, "JVP": (d * (tm - tr)).sum(axis=axes) / v - kl * tv / v,
                "VJP to the mean": ct[:, None] * d / v[:, None], "VJP to the variance": -ct * kl / v}

    inputs = (mean, reference, variance, mean_tangent, reference_tangent, variance_tangent, cotangent)
    return {name: np.asarray(value) for name, value in jax.jit(plain)(*map(jnp.asarray, inputs)).items()}


def _kl_bounds(mean, reference, variance, mean_tangent, reference_tangent, variance_tangent, cotangent):
    """Each output's derived bound where the plain float32 arm overflows and has no reference (the
    gate's ruling), check 2's form (`assert_fp32_reduction_bound`): the float64 magnitude of its terms
    and the rounding count of its plain formula, a KL of n coordinates n + 2 (each difference, each
    square, the n - 1 additions, the division), a VJP to the mean three, the variance's the KL's and
    two more, and a JVP twice the KL's and six more over its terms' absolute sum, so a sum that
    cancels is bounded by what it cancelled. Each as `(magnitudes, roundings)`."""
    wide = [np.asarray(value, np.float64) for value in
            (mean, reference, variance, mean_tangent, reference_tangent, variance_tangent, cotangent)]
    difference, moving = wide[0] - wide[1], wide[3] - wide[4]
    count = difference.shape[1]
    divergence = np.sum(difference**2, axis=1) / (2 * wide[2])
    terms = np.sum(np.abs(difference * moving), axis=1) / wide[2] + divergence * np.abs(wide[5]) / wide[2]
    return {"KL": (divergence, count + 2), "JVP": (terms, 2 * count + 10),
            "VJP to the mean": (np.abs(wide[6][:, None] * difference / wide[2][:, None]), 3),
            "VJP to the variance": (np.abs(wide[6] * divergence / wide[2]), count + 4)}


def _kl_derivatives(mean, reference, variance, mean_tangent, reference_tangent, variance_tangent, cotangent):
    """A transition's KL, its JVP along every tangent at once, and its VJP to the mean and the variance."""
    from dew.sampling.flow import GaussianTransition

    def kl(at, reference, spread):
        return GaussianTransition(at, spread).kl(reference)

    value, moved = jax.jvp(kl, (mean, reference, variance),
                           (mean_tangent, reference_tangent, variance_tangent))
    _, pullback = jax.vjp(lambda at, spread: kl(at, reference, spread), mean, variance)
    to_mean, to_variance = pullback(cotangent)
    return {"KL": value, "JVP": moved, "VJP to the mean": to_mean, "VJP to the variance": to_variance}


EDGE = 2.0**-16
"""How near float32's least normal or its largest an exact value lies, relatively, where a few
roundings may take a computed one across that bound (to a flushed subnormal or an infinity)."""


def _kinds(exact):
    """Each float64 value's place in float32's range: "normal" from a few roundings above the least
    normal, its whole lowest binade [2**-126, 2**-125) but that sliver, to a few below the largest;
    "over" from 2**128, past every float32; "below" zero or under half the least normal, which rounds
    to zero or a subnormal; and "edge" the slivers between, where a few roundings may cross a bound
    (`EDGE`)."""
    size = np.abs(exact)
    tiny, largest = float(np.finfo(np.float32).tiny), float(np.finfo(np.float32).max)
    return np.where(size >= 2.0**128, "over", np.where(
        (size >= tiny * (1 + EDGE)) & (size <= largest * (1 - EDGE)), "normal", np.where(
            (size == 0) | (size < tiny / 2), "below", "edge")))


def _below_normal(values):
    """Whether each float32 is zero or subnormal: its exponent field is zero."""
    return (values.astype(np.float32).view(np.uint32) >> 23 & 0xFF) == 0


BAND = 16
"""The binades one check-1 leaf of the KL's outputs spans: rows whose truths lie further apart are
judged as leaves of their own, so a small row is not read as a large one's float64 zero."""


def _assert_kl_outputs(dew, plain, truth, bounds, label):
    """Each output of `_kl_derivatives` against the float64 truth, element by element as the gate rules
    (rows 1722 and 1778): where the plain float32 arm (`_plain_kl`) is finite, on par with it (the grid's
    one-draw rule, `assert_as_exact_as_the_reference` at FACTOR until the gate's switch-over to PARITY;
    unscaled, its rows whole and its zero rows included, each band of `BAND` binades of the rows' sizes a
    leaf of its own, an element the plain arm leaves out of a row taking the truth in both arms); where
    it overflows, finite wherever the truth is a normal float32 and within the derived bound of the
    truth (`_kl_bounds`); infinite of its sign past the largest float32, zero or subnormal below the
    least, by the truth's place in float32's range (`_kinds`); and near either end (`edge`), the bound's
    or the end's: infinite of its sign above, zero or subnormal below."""
    for name, exact in truth.items():
        value, reference, kinds = np.asarray(dew[name]), np.asarray(plain[name]), _kinds(exact)
        # A truth below float32's least normal, or near either end, is held by its kind below.
        held = np.isfinite(reference) & ((kinds == "normal") | (exact == 0))
        mine, theirs = np.where(held, value, exact), np.where(held, reference, exact)
        axes = tuple(range(1, exact.ndim))
        size = np.max(np.abs(exact), axis=axes) if axes else np.abs(exact)
        band = np.frexp(size)[1] // BAND
        # Where the plain arm and Dew both give the float64 value there is no error to compare.
        apart = (mine != exact) | (theirs != exact)
        measured = np.any(apart, axis=axes) if axes else apart
        for level in np.unique(band[measured]):
            rows = measured & (band == level)
            assert_as_exact_as_the_reference(mine[rows], theirs[rows], exact[rows],
                                             f"{label}: {name}, rows near 2**{level * BAND}")
        magnitudes, roundings = bounds[name]
        unheld = ~np.isfinite(reference) & ((kinds == "normal") | (exact == 0))
        assert np.all(np.isfinite(value[unheld])), name
        assert_fp32_reduction_bound(value[unheld], exact[unheld], magnitudes[unheld], roundings)
        over = kinds == "over"
        assert np.all(np.isinf(value[over]) & (np.sign(value[over]) == np.sign(exact[over]))), name
        assert np.all(_below_normal(value[kinds == "below"])), name
        edge = kinds == "edge"
        top = edge & (np.abs(exact) > 1)
        bounded = top & ~(np.isinf(value) & (np.sign(value) == np.sign(exact)))
        assert np.all(np.isfinite(value[bounded])), name
        assert_fp32_reduction_bound(value[bounded], exact[bounded], magnitudes[bounded], roundings)
        bottom = edge & ~top & ~_below_normal(value)
        assert np.all(np.isfinite(value[bottom])), name
        assert_fp32_reduction_bound(value[bottom], exact[bottom], magnitudes[bottom], roundings)


KL_LEVELS = (-126, -125, -100, -63, -26, -1, 0, 1, 26, 63, 100, 126, 127)
"""The exponents a grid row's operands take: both of float32's edge binades, its middle, and between."""

FIXED_ROWS = (
    # (mean, reference, variance, mean tangent, reference tangent, variance tangent, cotangent), three
    # coordinates: each a review's or the second opinion's input. A JVP of products [2**120, -2**120,
    # 2**-100], which a plain float32 sum keeps and aligning to the largest flushed (Main, 1778).
    ([2.0**60, 2.0**60, 2.0**-100], [0, 0, 0], 1, [2.0**60, -2.0**60, 1], [0, 0, 0], 0, 1),
    ([2.0**60, 2.0**60, 1], [0, 0, 0], 1, [2.0**60, -2.0**60, 2.0**-5], [0, 0, 0],
     (1 + 2.0**-23) * 2.0**-125, 1),
    ([2.0**60, 2.0**60, 0], [2.0**-100, 0, 0], 1, [1, -1, 0], [0, 0, 0], 0, 1),
    ([1, 0, 0], [0, 0, 0], 1, [2.0**100, 0, 0], [2.0**-100, 0, 0], 2.0**101, 1),
    # Variances near float32's largest, whose double overflows, and near its least.
    ([1e18, 0, 0], [0, 0, 0], 2.5e38, [4, 0, 0], [4, 0, 0], 1e30, 0.7),
    ([0.5, 0, 0], [0, 0, 0], 2.0**-126, [4, 0, 0], [-4, 0, 0], 2.0**-126, 2.0**-126),
    ([2.0**127, 0, 0], [-2.0**127, 0, 0], float(np.finfo(np.float32).max), [1, 0, 0], [0, 0, 0], 0, 1),
    # A product the plain sum flushes beside largest ones that cancel (train 208, 1806): Dew may lose
    # it as plain does, and no more.
    ([2.0**45, 2.0**45, 2.0**-73], [0, 0, 0], 2.0**-21, [2.0**81, -2.0**81, 2.0**-73], [0, 0, 0], 0, 1),
    # A KL just under float32's largest (train 208), and two rows of far different sizes, the second's
    # VJP not hiding the first's.
    ([2.0**64, 0, 0], [0, 0, 0], 0.5 + 2.0**-24, [1, 0, 0], [0, 0, 0], 0, 1),
    ([1, 1, 0], [0, 0, 0], 1, [1, 0, 0], [0, 0, 0], 0, 1),
    ([2.0**100, 0, 0], [0, 0, 0], 3, [1, 0, 0], [0, 0, 0], 0, 1),
)
"""Rows every grid holds, beside its drawn ones."""


def _kl_grid():
    """Rows of three coordinates over float32's exponent range in every operand and tangent: each value
    a signed significand in [1, 2) at a level of `KL_LEVELS`, or zero, drawn apart per coordinate, every
    level met in every operand; in a third of the rows the mean's largest coordinate is the
    reference's (a cancelling maximum, its survivors at their own levels), in another third the
    tangents' largest cancel likewise; and `FIXED_ROWS` and `REVIEWED`'s rows. Every input is a normal
    float32 or zero: XLA:CPU reads a subnormal input as zero, which no float64 truth of the input's own
    value does."""
    rng = np.random.default_rng(1778)
    rows = 3 * 4096

    def drawn(shape, zeros=0.2):
        levels = rng.choice(KL_LEVELS, shape)
        values = np.ldexp(rng.uniform(1, 2, shape) * rng.choice((-1.0, 1.0), shape), levels)
        return np.where(rng.random(shape) < zeros, 0.0, values).astype(np.float32)

    mean, reference, mean_tangent, reference_tangent = (drawn((rows, 3)) for _ in range(4))
    variance = np.abs(drawn((rows,), zeros=0)).astype(np.float32)
    variance_tangent, cotangent = drawn((rows,)), drawn((rows,), zeros=0.05)
    cancelling, tangents = slice(0, rows // 3), slice(rows // 3, 2 * rows // 3)
    mean[cancelling, 0] = np.ldexp(np.float32(1.5), rng.choice((126, 127, 100, 0), rows // 3))
    reference[cancelling, 0] = mean[cancelling, 0]
    mean_tangent[tangents, 0] = np.ldexp(np.float32(1.25), rng.choice((127, 126, 63), rows // 3))
    reference_tangent[tangents, 0] = mean_tangent[tangents, 0]
    # Every level in every operand at least once, in a row of its own.
    for index, level in enumerate(KL_LEVELS):
        for operand in (mean, reference, mean_tangent, reference_tangent):
            operand[2 * rows // 3 + index, index % 3] = np.ldexp(np.float32(1.75), level)
        for operand in (variance, variance_tangent, cotangent):
            operand[2 * rows // 3 + index] = np.ldexp(np.float32(1.75), level)
    def padded(case):
        """A reviewed input's one row in three coordinates."""
        return tuple(np.pad(np.asarray(value, np.float64)[0], (0, 3 - len(value[0])))
                     if isinstance(value[0], list) else value[0] for value in case)

    rows_given = FIXED_ROWS + tuple(padded(case) for case in REVIEWED.values())
    fixed = [np.asarray(column, np.float32) for column in zip(*rows_given, strict=True)]
    drawn_inputs = (mean, reference, variance, mean_tangent, reference_tangent, variance_tangent, cotangent)
    return tuple(np.concatenate([draws, row]) for draws, row in zip(drawn_inputs, fixed, strict=True))


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_a_transitions_kl_and_its_derivatives_hold_over_float32s_exponent_range(compiled):
    """Over a grid of float32's exponents in every input (`_kl_grid`), cancelling and not, with every
    review's input among its rows, a transition's KL, its JVP along the mean's, the reference's and
    the variance's tangents at once, and its VJPs to the mean and the variance hold the float64 truth
    as the gate rules (`_assert_kl_outputs`), eagerly and compiled."""
    inputs = _kl_grid()
    run = jax.jit(_kl_derivatives) if compiled else _kl_derivatives
    _assert_kl_outputs(run(*(jnp.asarray(value) for value in inputs)), _plain_kl(*inputs),
                       _kl_truth(*inputs), _kl_bounds(*inputs), "a KL over float32's exponents")


REVIEWED = {
    # (mean, reference, variance, mean tangent, reference tangent, variance tangent, cotangent), each
    # input one row, the KL's JVP and VJP each exact in float32.
    "train 192's cancelling tangent": (
        [[0.5, 0.5]], [[0, 0]], [2.0**-126], [[16, -16]], [[0, 0]], [0], [1]),
    "train 195's overflowing contraction": (
        [[2.0**63, 0]], [[0, 0]], [2.0**32], [[2.0**66, 0]], [[0, 0]], [0], [1]),
    "train 195's overflowing variance pullback": (
        [[1, 1]], [[0, 0]], [1], [[0, 0]], [[0, 0]], [1], [2.0**127]),
    "a variance tangent over the variance first": (
        [[2.0**63, 0]], [[0, 0]], [2.0**32], [[0, 0]], [[0, 0]], [2.0**-126], [1]),
    "two terms past the range whose sum is in it": (
        [[1]], [[0]], [2.0**-126], [[4 + 2.0**-21]], [[0]], [2.0**-123], [1]),
    "a difference past the largest": (
        [[2.0**127]], [[-2.0**127]], [np.finfo(np.float32).max], [[1]], [[0]], [0], [1]),
    "train 201's cancelling maximum": (
        [[2.0**126, 2.0**-26]], [[2.0**126, 0]], [2.0**-52], [[0, 2.0**-26]], [[0, 0]], [0], [1]),
}


@pytest.mark.parametrize("case", list(REVIEWED))
def test_a_transitions_kl_and_its_derivatives_hold_at_each_reviewed_input(case):
    """Each input a review found leaving float32's range (`REVIEWED`), eagerly and compiled, as the
    gate rules (`_assert_kl_outputs`)."""
    inputs = tuple(np.asarray(value, np.float32) for value in REVIEWED[case])
    for run in (_kl_derivatives, jax.jit(_kl_derivatives)):
        _assert_kl_outputs(run(*(jnp.asarray(value) for value in inputs)), _plain_kl(*inputs),
                           _kl_truth(*inputs), _kl_bounds(*inputs), case)




def test_a_transitions_density_keeps_a_residual_whose_square_overflows():
    """A residual of 2**64 under a variance of 2**126: its square passes float32's largest, its square
    over the variance is 4, and the density is finite as the reference's is, the half-Mahalanobis term
    taken range-safe as the KL takes it (ledger 1874)."""
    from dew.sampling.flow import GaussianTransition

    transition = GaussianTransition(jnp.zeros((1, 1), jnp.float32), jnp.asarray([2.0 ** 126], jnp.float32))
    residual = jnp.full((1, 1), 2.0 ** 64, jnp.float32)
    density = np.asarray(transition.log_prob(residual))
    # The density on the standardized square, 4, in float64, held by the oracle rule for its fixed input.
    half_log = 0.5 * (math.log(2 * math.pi) + 126 * math.log(2))
    _held_by_its_roundings(density, np.asarray([-(2 + half_log)]), 2 + half_log, transition.log_prob,
                           residual)
    # Differentiable at every order, as the plain density: at unit variance its Hessian in the mean is -I.
    unit = GaussianTransition(jnp.zeros((1, 3), jnp.float32), jnp.ones((1,), jnp.float32))
    def density(mean):
        return GaussianTransition(mean, unit.variance).log_prob(jnp.ones((1, 3), jnp.float32)).sum()

    for second in (jax.jacfwd(jax.jacrev(density)), jax.jacrev(jax.jacfwd(density))):
        np.testing.assert_array_equal(np.asarray(second(unit.mean)).reshape(3, 3), -np.eye(3))
    # Finite wherever the term is: a residual of 2**64 under unit variance, half its square 2**127, and
    # a value of 2**127 beside a mean of -2**127 under float32's largest variance.
    for value, mean, variance, term in ((2.0 ** 64, 0.0, 1.0, 2.0 ** 127),
                                        (2.0 ** 127, -(2.0 ** 127), float(np.finfo(np.float32).max),
                                         2.0 ** 255 / float(np.finfo(np.float32).max))):
        edge = GaussianTransition(jnp.full((1, 1), mean, jnp.float32), jnp.asarray([variance], jnp.float32))
        density = edge.log_prob(jnp.full((1, 1), value, jnp.float32))
        truth = -(term + 0.5 * (math.log(2 * math.pi) + math.log(variance)))
        # The term rounds once, a quotient of powers of two over a significand, and the log terms fall
        # below its last place: the float32 rounding of the float64 density.
        np.testing.assert_array_equal(np.asarray(density), [np.float32(truth)])
        # Its derivative in the mean is finite there too, the residual over the variance.
        slope = jax.grad(lambda centre, edge=edge, value=value: edge.replace(mean=centre).log_prob(
            jnp.full((1, 1), value, jnp.float32)).sum())(edge.mean)
        np.testing.assert_array_equal(np.asarray(slope), [[np.float32((value - mean) / variance)]])
    # A residual of 2**-30 under a variance of 2**-120: the variance's derivative, -term / variance, passes
    # float32's largest, and the density's tangent along the mean alone, in forward mode and reverse,
    # is the residual over the variance, 2**90, finite, no inf times zero (ledger 1916).
    small = GaussianTransition(jnp.zeros((1, 1), jnp.float32), jnp.asarray([2.0 ** -120], jnp.float32))

    def along_mean(centre):
        return small.replace(mean=centre).log_prob(jnp.full((1, 1), 2.0 ** -30, jnp.float32)).sum()

    _, moved = jax.jvp(along_mean, (small.mean,), (jnp.ones((1, 1), jnp.float32),))
    np.testing.assert_array_equal(np.asarray(moved), 2.0 ** 90)
    np.testing.assert_array_equal(np.asarray(jax.grad(along_mean)(small.mean)), [[2.0 ** 90]])
