"""Flow-GRPO against Gaussian algebra and the authors' SDE convention.

Reference: arXiv:2505.05470v5, equations 8-9, and
https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py
The reference averages coordinate log densities for its policy ratio. These
checks retain the joint density until the objective performs that reduction.
"""

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import optax
from dew.training import Trainer

from flax import linen as nn

from dew.diffusion import FlowMatchingScheduler, FlowMatchPredictionTransform, Process
from dew.sampling import CFG, sample
from dew.sampling.flow import FlowSDE, flow_transition, sample_trajectory
from dew.inputs import Field, InputSpec
from dew.objectives.base import Step, scalar_loss
from dew.objectives.rl.flow import FlowGRPOObjective, FlowRollout


def test_transition_density_and_velocity_gradient_match_gaussian_algebra():
    x = np.asarray([[0.4, -0.7], [0.1, 0.8], [-0.3, 0.2]], np.float32)
    velocity = np.asarray([[0.2, 0.3], [-0.4, 0.5], [0.1, -0.2]], np.float32)
    action = np.asarray([[0.7, -0.6], [0.0, 0.4], [-0.2, 0.9]], np.float32)
    t = np.asarray([0.7, 0.4, 1.0], np.float32)
    s = np.asarray([0.5, 0.1, 0.9], np.float32)
    noise = 0.6
    dt = (s - t).astype(np.float64)
    denominator = 1 - np.where(t == 1, s, t).astype(np.float64)
    diffusion_squared = noise**2 * t / denominator
    correction = diffusion_squared / (2 * t)
    mean = x * (1 + correction[:, None] * dt[:, None]) + velocity * (
        1 + correction[:, None] * (1 - t[:, None])) * dt[:, None]
    variance = diffusion_squared * -dt
    expected = np.sum(-0.5 * ((action - mean)**2 / variance[:, None]
                              + np.log(2 * math.pi * variance[:, None])), axis=1)

    transition = flow_transition(x, velocity, t, s, noise_level=noise)
    actual = transition.log_prob(jnp.asarray(action))
    np.testing.assert_allclose(transition.mean, mean, atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(transition.variance, variance, atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)

    actual_gradient = jax.grad(lambda v: flow_transition(
        x, v, t, s, noise_level=noise).log_prob(jnp.asarray(action)).sum())(jnp.asarray(velocity))
    mean_derivative = (1 + correction * (1 - t)) * dt
    expected_gradient = (action - mean) / variance[:, None] * mean_derivative[:, None]
    np.testing.assert_allclose(actual_gradient, expected_gradient, atol=2e-6, rtol=2e-6)



def test_deterministic_endpoints_have_no_gaussian_density():
    x = jnp.asarray([[0.2, -0.3], [0.7, 0.4]])
    velocity = jnp.asarray([[0.4, 0.1], [-0.1, 0.8]])
    ode = flow_transition(x, velocity, 0.8, 0.2, noise_level=0)
    np.testing.assert_allclose(ode.sample(jax.random.key(3)), x - 0.6 * velocity, atol=1e-7)
    assert np.isnan(ode.log_prob(ode.mean)).all()
    np.testing.assert_array_equal(ode.stochastic, [False, False])
    np.testing.assert_array_equal(ode.kl(ode.mean), [0, 0])
    assert np.isinf(ode.kl(ode.mean + 1)).all()
    endpoint = flow_transition(x, velocity, jnp.asarray([0., 1.]), jnp.asarray([0., 1.]))
    np.testing.assert_array_equal(endpoint.sample(jax.random.key(4)), x)
    assert np.isnan(endpoint.log_prob(x)).all()


def test_gaussian_kl_uses_elapsed_time_and_all_sample_dimensions():
    x = jnp.arange(8, dtype=jnp.float32).reshape(2, 2, 2) / 10
    velocity = jnp.full_like(x, 0.2)
    t, s, noise = jnp.asarray([0.6, 0.8]), jnp.asarray([0.2, 0.7]), 0.4
    transition = flow_transition(x, velocity, t, s, noise_level=noise)
    reference = flow_transition(x, velocity + 0.3, t, s, noise_level=noise)
    dt = np.asarray(s - t)
    g_squared = noise**2 * np.asarray(t) / (1 - np.asarray(t))
    coefficient = (1 + g_squared * (1 - np.asarray(t)) / (2 * np.asarray(t))) * dt
    expected = 4 * np.square(0.3 * coefficient) / (2 * g_squared * -dt)
    np.testing.assert_allclose(transition.kl(reference.mean), expected, atol=2e-6, rtol=2e-6)


class AffineVelocity(nn.Module):
    @nn.compact
    def __call__(self, x, temb, offset=0):
        gain = self.param("gain", nn.initializers.constant(0.25), ())
        return gain * x + (temb / 1000)[:, None] * 0.1 + offset


def test_recorded_trajectory_rescores_under_shifted_guided_process():
    process = Process(FlowMatchingScheduler(shift=3), FlowMatchPredictionTransform())
    model = AffineVelocity()
    x = jnp.asarray([[0.2, -0.5], [-0.1, 0.7]])
    offset = jnp.asarray([[-0.4], [0.8]])
    variables = model.init(jax.random.key(0), x, jnp.ones(2), offset=offset)
    denoise = process.denoiser(model, variables, {"offset": offset}, {"offset": jnp.zeros_like(offset)})
    solver, guidance, key = FlowSDE(noise_level=0.5), CFG(2, interval=(0.1, 0.8)), jax.random.key(1)
    trajectory = jax.jit(lambda start: sample_trajectory(
        denoise, start, 5, solver=solver, guidance=guidance, key=key))(x)
    ordinary = sample(denoise, x, 5, solver=solver, guidance=guidance, key=key)
    np.testing.assert_array_equal(trajectory.samples, ordinary)
    for index in range(4):
        t, s = np.asarray(trajectory.times[index:index + 2], np.float64)
        sigma, following = 3 * t / (1 + 2 * t), 3 * s / (1 + 2 * s)
        latent = np.asarray(trajectory.states[:, index], np.float64)
        action = np.asarray(trajectory.states[:, index + 1], np.float64)
        scale = 2 if 0.1 <= 1 - t <= 0.8 else 1
        velocity = 0.25 * latent + 0.1 * sigma + scale * np.asarray(offset)
        dt = following - sigma
        g_squared = 0.5**2 * sigma / (1 - (following if t == 1 else sigma))
        mean = latent + (velocity + g_squared / (2 * sigma) * (
            latent + (1 - sigma) * velocity)) * dt
        variance = g_squared * -dt
        expected = (-0.5 * ((action - mean)**2 / variance + np.log(2 * math.pi * variance))).sum(1)
        np.testing.assert_allclose(trajectory.log_probs[:, index], expected, atol=3e-6, rtol=2e-6)
    deterministic = sample_trajectory(denoise, x, 5, solver=FlowSDE(0), key=key)
    assert np.isnan(deterministic.log_probs).all()
    assert not np.asarray(deterministic.stochastic).any()



def trajectory_batch(trajectory, advantages, mask=None):
    count, points = trajectory.states.shape[:2]
    return {
        "latents": trajectory.states[:, :-1],
        "next_latents": trajectory.states[:, 1:],
        "timesteps": jnp.broadcast_to(trajectory.times[:-1], (count, points - 1)),
        "next_timesteps": jnp.broadcast_to(trajectory.times[1:], (count, points - 1)),
        "old_log_probs": trajectory.log_probs,
        "transition_mask": trajectory.stochastic if mask is None else jnp.asarray(mask),
        "advantages": jnp.asarray(advantages),
    }


def test_flow_objective_matches_clipping_and_conditional_gaussian_kl():
    process = Process(FlowMatchingScheduler(), FlowMatchPredictionTransform())
    model = AffineVelocity()
    objective = FlowGRPOObjective(model, process, InputSpec(Field("image", (2,))),
                                  sde=FlowSDE(0.6), guidance=None, beta=0.2,
                                  clip_range=0.001, adv_clip_max=2)
    old = objective.init(jax.random.key(2))
    x = jnp.asarray([[0.1, -0.7], [0.5, 0.8], [-0.9, 0.2], [0.3, -0.4]])
    trajectory = sample_trajectory(process.denoiser(model, old, {}), x, 4,
                                   solver=objective.sde, key=jax.random.key(3))
    mask = np.asarray([[1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 1, 1]], bool)
    advantages = np.asarray([-3, 1, 2.5, -1], np.float32)
    batch = trajectory_batch(trajectory, advantages, mask)
    np.testing.assert_allclose(objective.log_probs(old, batch), trajectory.log_probs, atol=3e-6)
    batch["old_log_probs"] = jnp.where(mask, batch["old_log_probs"], jnp.nan)
    current = {**old, "params": {"gain": jnp.asarray(0.9)}}
    reference = {**old, "params": {"gain": jnp.asarray(0.1)}}
    step = Step(jnp.asarray(0), jax.random.key(4), reference)
    loss, _ = scalar_loss(objective, current, batch, step)
    gradient = jax.grad(lambda gain: scalar_loss(
        objective, {**current, "params": {"gain": gain}}, batch, step)[0])(current["params"]["gain"])

    expected_total, expected_gradient, clipped = 0.0, 0.0, 0
    expected_policy = 0.0
    for i in range(4):
        for j in range(3):
            if not mask[i, j]:
                continue
            t, s = np.asarray(trajectory.times[j:j + 2], np.float64)
            latent = np.asarray(trajectory.states[i, j], np.float64)
            following = np.asarray(trajectory.states[i, j + 1], np.float64)
            g2 = 0.6**2 * t / (1 - (s if t == 1 else t))
            variance = g2 * (t - s)
            coefficient = (1 + g2 * (1 - t) / (2 * t)) * (s - t)
            offset = latent * (1 + g2 / (2 * t) * (s - t))
            means = [offset + (gain * latent + 0.1 * t) * coefficient for gain in (0.25, 0.9, 0.1)]
            old_mean, mean, reference_mean = means
            log_ratio = (-np.square(following - mean) + np.square(following - old_mean)).mean() / (2 * variance)
            ratio = np.exp(log_ratio)
            advantage = np.clip(advantages[i], -2, 2)
            unclipped = -advantage * ratio
            bounded = -advantage * np.clip(ratio, 0.999, 1.001)
            kl = np.square(mean - reference_mean).mean() / (2 * variance)
            expected_total += max(unclipped, bounded) + 0.2 * kl
            expected_policy += max(unclipped, bounded)
            mean_derivative = latent * coefficient
            log_derivative = ((following - mean) * mean_derivative).mean() / variance
            if unclipped >= bounded:
                expected_gradient += -advantage * ratio * log_derivative
            else:
                clipped += 1
            expected_gradient += 0.2 * ((mean - reference_mean) * mean_derivative).mean() / variance
    assert 0 < clipped < mask.sum()
    assert float(loss) == pytest.approx(expected_total / mask.sum(), abs=3e-6)
    assert float(gradient) == pytest.approx(expected_gradient / mask.sum(), abs=3e-6)
    stats, _ = objective.loss(current, batch, step)
    assert float(stats.mass) == mask.sum()
    without_reference = FlowGRPOObjective(model, process, objective.inputs, sde=objective.sde,
                                           guidance=None, clip_range=0.001, adv_clip_max=2)
    policy_only, _ = scalar_loss(without_reference, current, batch, step.replace(ema=None))
    assert float(policy_only) == pytest.approx(expected_policy / mask.sum(), abs=3e-6)
    with pytest.raises(ValueError, match="reference"):
        scalar_loss(objective, current, batch, step.replace(ema=None))


def test_deterministic_rollout_cannot_contribute_policy_gradient():
    process = Process(FlowMatchingScheduler(), FlowMatchPredictionTransform())
    model = AffineVelocity()
    objective = FlowGRPOObjective(model, process, InputSpec(Field("image", (2,))),
                                  sde=FlowSDE(0), guidance=None, beta=0.1)
    variables = objective.init(jax.random.key(7))
    x = jnp.asarray([[0.1, -0.2], [0.3, 0.4]])
    trajectory = sample_trajectory(process.denoiser(model, variables, {}), x, 3,
                                   solver=objective.sde, key=jax.random.key(8))
    batch = trajectory_batch(trajectory, [-1, 1], np.ones((2, 2), bool))
    step = Step(jnp.asarray(0), jax.random.key(9), variables)
    stats, _ = objective.loss(variables, batch, step)
    value, active = objective.reduce_loss(stats)
    assert float(stats.mass) == 0 and float(value) == 0 and not bool(active)
    gradient = jax.grad(lambda gain: scalar_loss(
        objective, {**variables, "params": {"gain": gain}}, batch, step)[0])(variables["params"]["gain"])
    assert float(gradient) == 0



def test_flow_rollout_groups_rewards_selects_steps_and_preserves_likelihoods():
    from dew.inputs import CharTable, Condition
    from dew.nn.backbones.dit import SimpleDiT



    process = Process(FlowMatchingScheduler(shift=2), FlowMatchPredictionTransform())
    inputs = InputSpec(Field("image", (4, 4, 1)), {
        "textcontext": Condition(CharTable.from_pretrained(tokens=3, features=4))})
    model = SimpleDiT(output_channels=1, patch_size=2, emb_features=8,
                      num_layers=1, num_heads=2, mlp_ratio=2)
    objective = FlowGRPOObjective(model, process, inputs, guidance=CFG(1.5), steps=4)

    optimizer = optax.sgd(1e-3)
    state = Trainer(objective, optimizer, key=jax.random.key(21)).initial_state()
    variables = state.params
    prompts = {**inputs.tokenize(["red", "blue"]), "target": np.asarray([-0.3, 0.6], np.float32)}

    def reward(images, context):
        return -np.square(images.mean(axis=(1, 2, 3)) - context["target"])

    rollout = FlowRollout(objective, reward, groups=3, steps=4, train_steps=2)
    batch = rollout(state, prompts, jax.random.key(23))
    rewards = np.asarray(batch["rewards"]).reshape(2, 3)
    expected = (rewards - rewards.mean(axis=1, keepdims=True)) / (rewards.std(axis=1, keepdims=True) + 1e-4)
    np.testing.assert_allclose(batch["advantages"], expected.reshape(-1), atol=2e-6)
    np.testing.assert_allclose(batch["timesteps"], np.broadcast_to([1, 2/3], (6, 2)), atol=1e-7)
    np.testing.assert_allclose(objective.log_probs(variables, batch), batch["old_log_probs"], atol=5e-6)
    loss, _ = scalar_loss(objective, variables, batch, Step(jnp.asarray(0), jax.random.key(24), None))
    gradients = jax.grad(lambda p: scalar_loss(
        objective, {**variables, "params": p}, batch,
        Step(jnp.asarray(0), jax.random.key(24), None))[0])(variables["params"])
    norm = float(optax.tree.norm(gradients))
    assert np.isfinite(float(loss)) and np.isfinite(norm) and norm > 1e-5


def test_nonfinite_reward_is_refused_before_advantage_construction():

    model = AffineVelocity()
    process = Process(FlowMatchingScheduler(), FlowMatchPredictionTransform())
    objective = FlowGRPOObjective(model, process, InputSpec(Field("image", (2,))), guidance=None)

    state = Trainer(objective, optax.sgd(1e-3), key=jax.random.key(31)).initial_state()
    rollout = FlowRollout(objective, lambda images, context: np.full(images.shape[0], np.nan), steps=3)
    with pytest.raises(ValueError, match="finite"):
        rollout(state, {"image": np.zeros((2, 2), np.float32)}, jax.random.key(33))



def test_zero_reward_variance_has_no_training_support():
    model = AffineVelocity()
    process = Process(FlowMatchingScheduler(), FlowMatchPredictionTransform())
    objective = FlowGRPOObjective(model, process, InputSpec(Field("image", (2,))),
                                  guidance=None, beta=0.1)
    state = Trainer(objective, optax.sgd(1e-3), key=jax.random.key(41)).initial_state()
    rollout = FlowRollout(objective, lambda images, context: np.ones(images.shape[0]),
                          groups=3, steps=4)
    batch = rollout(state, {"image": np.zeros((2, 2), np.float32)}, jax.random.key(42))
    assert batch["latents"].shape[:2] == (6, 3)
    assert np.isfinite(batch["old_log_probs"]).all()
    params = {**state.params, "params": {"gain": jnp.asarray(0.9)}}
    stats, _ = objective.loss(params, batch, Step(state.microstep, jax.random.key(43), state.averaged))
    value, active = objective.reduce_loss(stats)
    assert float(stats.mass) == 0 and float(value) == 0 and not bool(active)



def test_transition_matches_released_sampler_and_torch_distribution():
    """CPU fp32 maxima: mean 0, variance 2.98e-8, mean log density 3.58e-7,
    velocity gradient 2.38e-7, conditional KL 4.77e-7. Bounds allow fp32
    square-root and reduction rounding across backends.
    """
    fixture = Path(__file__).parent / "fixtures/rl/flow_transition.npz"
    with np.load(fixture, allow_pickle=False) as ref:
        transition = flow_transition(ref["x"], ref["velocity"], ref["t"], ref["t_next"],
                                     noise_level=float(ref["noise_level"]))
        dimensions = math.prod(ref["x"].shape[1:])
        gradient = jax.grad(lambda velocity: flow_transition(
            ref["x"], velocity, ref["t"], ref["t_next"], noise_level=float(ref["noise_level"]))
            .log_prob(ref["action"]).sum() / dimensions)(jnp.asarray(ref["velocity"]))
        np.testing.assert_allclose(transition.mean, ref["mean"], atol=3e-7, rtol=2e-6)
        np.testing.assert_allclose(transition.variance, ref["variance"], atol=3e-7, rtol=2e-6)
        np.testing.assert_allclose(transition.log_prob(ref["action"]) / dimensions,
                                   ref["mean_log_prob"], atol=3e-6, rtol=2e-6)
        np.testing.assert_allclose(gradient, ref["velocity_grad"], atol=3e-6, rtol=2e-6)
        np.testing.assert_allclose(transition.kl(ref["reference_mean"]), ref["conditional_kl"],
                                   atol=3e-6, rtol=2e-6)



@pytest.mark.mesh
def test_multihost_flow_rollout_reassembles_owned_groups(tmp_path):
    import subprocess
    import sys
    from test_multiprocess import free_port, report_of, terminate, worker_env, dumped_params, assert_same_parameters

    worker = Path(__file__).with_name("flow_grpo_worker.py")
    coordinator = f"127.0.0.1:{free_port()}"
    outputs = [tmp_path / f"rank{rank}.json" for rank in range(2)]
    running = [subprocess.Popen(
        [sys.executable, str(worker), str(rank), "2", coordinator, str(output)],
        env=worker_env(1), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True) for rank, output in enumerate(outputs)]
    try:
        reports = [report_of(process, output, timeout=120)
                   for process, output in zip(running, outputs)]
    finally:
        for process in running:
            if process.poll() is None:
                terminate(process)
    baseline_path = tmp_path / "single.json"
    baseline_process = subprocess.Popen(
        [sys.executable, str(worker), "0", "1", "unused", str(baseline_path)],
        env=worker_env(1), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True)
    baseline = report_of(baseline_process, baseline_path, timeout=120)
    local_rewards = np.concatenate([report["local_rewards"] for report in reports])
    assert local_rewards.shape == (12,)
    np.testing.assert_allclose(local_rewards, baseline["global_rewards"], atol=3e-6)
    for report, output in zip(reports, outputs):
        np.testing.assert_allclose(report["global_rewards"], baseline["global_rewards"], atol=3e-6)
        assert report["density_error"] < 1e-5
        assert report["updates"] == 1 and report["parameter_change"] > 0
        assert report["reference_unchanged"]
        assert_same_parameters(dumped_params(output), dumped_params(baseline_path))



class ConstantVelocity(nn.Module):
    @nn.compact
    def __call__(self, x, temb):
        speed = self.param("speed", nn.initializers.constant(0.1), ())
        return jnp.full_like(x, speed)


def test_flow_evaluation_and_preview_use_live_policy_with_frozen_reference():
    process = Process(FlowMatchingScheduler(shift=3), FlowMatchPredictionTransform())
    objective = FlowGRPOObjective(ConstantVelocity(), process, InputSpec(Field("image", (2, 2, 1))),
                                  beta=0.1, guidance=None, steps=3)
    reference = objective.init(jax.random.key(51))
    live = {**reference, "params": {"speed": jnp.asarray(0.8)}}
    step = Step(jnp.asarray(0), jax.random.key(52), reference)
    batch = {"image": np.zeros((3, 2, 2, 1), np.uint8)}
    noise_key, _ = jax.random.split(step.key)
    expected = np.clip(np.asarray(process.noise(noise_key, (3, 2, 2, 1))) - 0.8, -1, 1)
    evaluated = objective.evaluate(live, batch, step)
    previewed = objective.preview(live, batch, step)
    np.testing.assert_allclose(evaluated.images, expected, atol=3e-7)
    np.testing.assert_allclose(previewed.images, expected, atol=3e-7)



def test_mixed_precision_transition_keeps_density_arithmetic_in_float32():
    from dew.sampling import GaussianTransition
    mean = jnp.asarray([[0.2, -0.3], [0.7, 0.4]], jnp.bfloat16)
    variance = jnp.asarray([0, 0.25], jnp.bfloat16)
    transition = GaussianTransition(mean, variance)
    key = jax.random.key(61)
    expected = mean.astype(jnp.float32) + jnp.sqrt(variance.astype(jnp.float32))[:, None] * jax.random.normal(key, mean.shape)
    sampled = transition.sample(key)
    # The compiled draw and eager oracle differ by one fp32 ULP (1.19e-7).
    np.testing.assert_allclose(sampled, expected, atol=2e-7, rtol=1e-6)
    np.testing.assert_array_equal(sampled[0], mean[0].astype(jnp.float32))
    assert sampled.dtype == jnp.float32
    log_probs = transition.log_prob(sampled)
    residual = np.asarray(sampled[1] - mean[1].astype(jnp.float32))
    expected_density = (-0.5 * (residual**2 / 0.25 + np.log(2 * np.pi * 0.25))).sum()
    assert np.isnan(float(log_probs[0]))
    assert float(log_probs[1]) == pytest.approx(expected_density, abs=1e-6)



def test_group_normalization_preserves_small_differences_on_large_reward_offset():
    model = AffineVelocity()
    process = Process(FlowMatchingScheduler(), FlowMatchPredictionTransform())
    objective = FlowGRPOObjective(model, process, InputSpec(Field("image", (2,))), guidance=None)
    state = Trainer(objective, optax.sgd(1e-3), key=jax.random.key(81)).initial_state()
    rollout = FlowRollout(objective, lambda images, context: 1_000_000 + images.mean(axis=1),
                          groups=3, steps=4)
    batch = rollout(state, {"image": np.zeros((2, 2), np.float32)}, jax.random.key(82))
    rewards = np.asarray(batch["rewards"], np.float64).reshape(2, 3)
    expected = (rewards - rewards.mean(axis=1, keepdims=True)) / (rewards.std(axis=1, keepdims=True) + 1e-4)
    np.testing.assert_allclose(batch["advantages"], expected.reshape(-1), atol=2e-6)









