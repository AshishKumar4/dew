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
import optax
import pytest
from flax import linen as nn
from recording import RecordingTracker

from dew.diffusion import FlowMatchingScheduler, FlowMatchPredictionTransform, Process
from dew.inputs import Field, InputSpec
from dew.objectives.base import Step
from dew.objectives.rl.flow import FlowGRPOObjective, FlowRollout
from dew.sampling import CFG, sample
from dew.sampling.flow import FlowSDE, flow_transition
from dew.telemetry.records import RECORD_TYPES
from dew.training import Trainer


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
    trajectory = jax.jit(lambda start: solver.trajectory(denoise, start, 5, guidance=guidance, key=key))(x)
    ordinary = sample(denoise, x, 5, solver=solver, guidance=guidance, key=key)
    np.testing.assert_array_equal(trajectory.samples, ordinary)
    for index in range(4):
        t, s = np.asarray(trajectory.times[index:index + 2], np.float64)
        sigma, following = 3 * t / (1 + 2 * t), 3 * s / (1 + 2 * s)
        latent = np.asarray(trajectory.states[:, index], np.float64)
        action = np.asarray(trajectory.states[:, index + 1], np.float64)
        scale = 2 if 0.1 <= index / 4 <= 0.8 else 1
        velocity = 0.25 * latent + 0.1 * sigma + scale * np.asarray(offset)
        dt = following - sigma
        g_squared = 0.5**2 * sigma / (1 - (following if t == 1 else sigma))
        mean = latent + (velocity + g_squared / (2 * sigma) * (
            latent + (1 - sigma) * velocity)) * dt
        variance = g_squared * -dt
        expected = (-0.5 * ((action - mean)**2 / variance + np.log(2 * math.pi * variance))).sum(1)
        np.testing.assert_allclose(trajectory.log_probs[:, index], expected, atol=3e-6, rtol=2e-6)
    deterministic = FlowSDE(0).trajectory(denoise, x, 5, key=key)
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


def test_flow_objective_matches_clipping_and_conditional_gaussian_kl():
    process = Process(FlowMatchingScheduler(), FlowMatchPredictionTransform())
    model = AffineVelocity()
    objective = FlowGRPOObjective(model, process, InputSpec(Field("image", (2,))),
                                  sde=FlowSDE(0.6), guidance=None, beta=0.2,
                                  clip_range=0.001, adv_clip_max=2)
    old = objective.init(jax.random.key(2))
    x = jnp.asarray([[0.1, -0.7], [0.5, 0.8], [-0.9, 0.2], [0.3, -0.4]])
    trajectory = objective.sde.trajectory(process.denoiser(model, old, {}), x, 4, key=jax.random.key(3))
    mask = np.asarray([[1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 1, 1]], bool)
    advantages = np.asarray([-3, 1, 2.5, -1], np.float32)
    batch = trajectory_batch(trajectory, advantages, mask)
    np.testing.assert_allclose(objective.log_probs(old, batch), trajectory.log_probs, atol=3e-6)
    batch["old_log_probs"] = jnp.where(mask, batch["old_log_probs"], jnp.nan)
    current = {**old, "params": {"gain": jnp.asarray(0.9)}}
    reference = {**old, "params": {"gain": jnp.asarray(0.1)}}
    step = Step(jnp.asarray(0), jax.random.key(4), reference)
    loss, _ = objective.scalar_loss(current, batch, step)
    gradient = jax.grad(lambda gain: objective.scalar_loss(
        {**current, "params": {"gain": gain}}, batch, step)[0])(current["params"]["gain"])

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
            log_ratio = (-np.square(following - mean) + np.square(following - old_mean)).mean() / (
                2 * variance
            )
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
    policy_only, _ = without_reference.scalar_loss(current, batch, step.replace(ema=None))
    assert float(policy_only) == pytest.approx(expected_policy / mask.sum(), abs=3e-6)
    with pytest.raises(ValueError, match="reference"):
        objective.scalar_loss(current, batch, step.replace(ema=None))


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
    """Rescoring reproduces the rollout's likelihoods, guidance included: the
    rollout walks 4 points and guides its step 1 of 3, inside (0.3, 0.6),
    while the objective's own walk of 7 points would not guide a step 1 of
    6, so the rescoring reads the rollout's step count off the batch."""
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
    rewards = np.asarray(batch["rewards"]).reshape(2, 3)
    expected = (rewards - rewards.mean(axis=1, keepdims=True)) / (rewards.std(axis=1, keepdims=True) + 1e-4)
    np.testing.assert_allclose(batch["advantages"], expected.reshape(-1), atol=2e-6)
    np.testing.assert_allclose(batch["timesteps"], np.broadcast_to([1, 2/3], (6, 2)), atol=1e-7)
    np.testing.assert_allclose(objective.log_probs(variables, batch), batch["old_log_probs"], atol=5e-6)
    loss, _ = objective.scalar_loss(variables, batch, Step(jnp.asarray(0), jax.random.key(24), None))
    gradients = jax.grad(lambda p: objective.scalar_loss({**variables, "params": p}, batch,
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
    params = {**state.variables, "params": {"gain": jnp.asarray(0.9)}}
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
    np.testing.assert_allclose(raw_callback, baseline["callback_rewards"], rtol=0, atol=5e-8)
    assert reports[0]["metric_rows"] == 4 and reports[0]["preview_rows"] == 4
    assert reports[1]["metric_rows"] == 0 and reports[1]["preview_rows"] == 0
    assert reports[0]["validation_mean"] == pytest.approx(baseline["validation_mean"], abs=1e-6)
    for report, output in zip(reports, outputs, strict=True):
        np.testing.assert_allclose(report["global_advantages"], expected_advantages, atol=2e-5)
        np.testing.assert_array_equal(report["global_rewards"], raw_callback.astype(np.float32))
        assert not report["x64_enabled"]
        assert report["density_error"] < 1e-5
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
    from dew.diffusion.presets import Flow

    preset = Flow(shift=3)
    process = preset()
    objective = FlowGRPOObjective(ConstantVelocity(), preset, InputSpec(Field("image", (2, 2, 1))),
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
    expected = mean.astype(jnp.float32) + jnp.sqrt(variance.astype(jnp.float32))[:, None] * jax.random.normal(
        key, mean.shape
    )
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
    callback_values = []

    def reward(images, context):
        values = 1_000_000 + images.mean(axis=1)
        callback_values.append(values.copy())
        return values

    rollout = FlowRollout(objective, reward,
                          groups=3, steps=4)
    batch = rollout(state, {"image": np.zeros((2, 2), np.float32)}, jax.random.key(82))
    rewards = np.asarray(callback_values[0], np.float64).reshape(2, 3)
    expected = (rewards - rewards.mean(axis=1, keepdims=True)) / (rewards.std(axis=1, keepdims=True) + 1e-4)
    np.testing.assert_allclose(batch["advantages"], expected.reshape(-1), atol=2e-6)



def test_float64_callback_distinctions_reach_a_real_policy_update():
    import itertools

    from dew.data import Dataset
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
    data = Dataset(train=lambda partition: itertools.repeat(source), val=None, records=count, batch=count)
    final = trainer.fit(data, steps=1, log_every=1)
    assert int(final.updates) == 1
    change = float(optax.tree.norm(jax.tree.map(
        lambda a, b: a - b, final.variables["params"], initial.variables["params"])))
    assert np.isfinite(change) and change > 1e-6
    np.testing.assert_allclose(collected["advantages"], oracle, atol=2e-6)
    np.testing.assert_array_equal(collected["rewards"], raw)
    assert collected["rewards"].dtype == np.float64
    np.testing.assert_array_equal(collected["transition_mask"],
                                   np.broadcast_to((oracle != 0)[:, None], (count * 3, 2)))



def test_conditioned_prompt_only_evaluation_preview_and_trainer_consumers():
    import itertools

    from dew.artifacts import ImageGrid
    from dew.data import Dataset
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
    tokens = {keyword: prompts[condition.field] for keyword, condition in inputs.conditions.items()}
    conditions = objective.encode(initial.variables["encoders"], tokens)
    denoiser = process.denoiser(model, objective.model_variables(initial.variables), conditions,
                               objective.encode(initial.variables["encoders"]))
    noise_key, sample_key = jax.random.split(step.key)
    expected = sample(denoiser, process.noise(noise_key, (count, *objective.latent_shape)),
                      objective.steps, solver=objective.solver, guidance=objective.guidance, key=sample_key)
    np.testing.assert_allclose(evaluated.images, np.clip(expected, -1, 1), atol=2e-6)
    assert previewed.images.shape == (min(4, count), 4, 4, 1)
    assert len(previewed.captions) == min(4, count)
    data = Dataset(
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
    assert logged["val/pixel_mean"] == pytest.approx(observed.mean(dtype=np.float64), abs=1e-8)
    assert logged["evaluation/records"] == count * 2
    previews = [value for _, value in tracker.artifacts if not isinstance(value, RECORD_TYPES)]
    assert previews[0].images.shape == (min(4, count), 4, 4, 1)


def test_flow_sde_takes_any_rectified_flow_schedule():
    """FlowSDE's step holds on the rectified-flow path, which a published
    source's flow grid walks as Dew's own schedule does, so either is taken
    and a VP schedule is refused; a timestep shift that is not finite and
    positive is refused where the schedule is built."""
    from dew.diffusion.schedules import CosineNoiseScheduler
    from dew.diffusion.schedules.source_grids import FlowGrid

    velocity = FlowMatchPredictionTransform()
    grid = FlowGrid(np.linspace(1.0, 0.0, 9), np.linspace(1000.0, 0.0, 9), 1.0)
    for schedule in (FlowMatchingScheduler(shift=3.0), grid):
        FlowSDE().validate(Process(schedule, velocity))
    with pytest.raises(ValueError, match="rectified-flow schedule"):
        FlowSDE().validate(Process(CosineNoiseScheduler(1000), velocity))
    for shift in (0.0, float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            FlowMatchingScheduler(shift=shift)

