"""Acceptance runs of Dew's environment, world-model and flow-RL stack, each writing a JSON report.

    python tools/rl_acceptance.py ppo OUT        PPO on Playground's CartpoleBalance: the training
                                                 episodes' return against wall-clock time, and the
                                                 greedy policy's return at the end.
    python tools/rl_acceptance.py dreamer OUT    DreamerV3 on CartpoleBalance at 100K environment
                                                 steps, against the 990 its paper reports at 500K
                                                 (arXiv 2301.04104v2, "Proprioceptive control scores").
    python tools/rl_acceptance.py nft OUT        DiffusionNFT against Flow-GRPO under matched
                                                 conditions: one pretrained flow model, one reward,
                                                 the same prompts, samples and updates.
    python tools/rl_acceptance.py dreamer4 OUT   Dreamer 4's world model at small scale: shortcut
                                                 forcing against diffusion forcing on bouncing-ball
                                                 video, open-loop prediction error at 4 and 64 steps.

Each writes OUT/<run>.json. Playground's environments need the `playground` distribution.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew.data.dataset import Loader


class _Tracker:
    """Each logged step's scalars with the wall-clock seconds since the run began."""

    def __init__(self) -> None:
        self.began, self.scalars = time.perf_counter(), []

    def log(self, scalars, step) -> None:
        """The training scalars by their own names, `train/` dropped."""
        named = {name.removeprefix("train/"): float(value) for name, value in scalars.items()}
        self.scalars.append({"step": int(step), "seconds": time.perf_counter() - self.began, **named})

    def artifact(self, value, step) -> None:
        del value, step


def _idle(rows: int) -> Loader:
    """A loader whose batches a rollout replaces whole."""
    return Loader(
        train=lambda partition: itertools.repeat({"step": np.zeros((rows,), np.float32)}),
        val=None,
        records=rows,
        batch=rows,
    )


def ppo(
    out: Path,
    *,
    task: str = "CartpoleBalance",
    envs: int = 2048,
    collections: int = 20,
    target: float = 900.0,
) -> dict:
    """Brax's DM Control PPO settings (Playground's dm_control_suite_params): 2048 environments,
    unrolls of 30, 16 passes of 32 minibatches over each collection of 16 unrolls, discount 0.995,
    rewards scaled by 10, entropy 1e-2, clip 0.3, Adam at 1e-3."""
    from dew.objectives.rl.control import ActorCritic, ControlPPOObjective, ControlRollout, episode_return
    from dew.rl.playground import Playground, playground
    from dew.training import Trainer

    probe = Playground.load(task)
    observation_size = int(probe.env.observation_size)
    env = playground(task, envs)
    objective = ControlPPOObjective(
        ActorCritic(probe.action_size),
        observation_size,
        discount=0.995,
        reward_scaling=10.0,
        entropy_cost=1e-2,
        clip=0.3,
    )
    rollout = ControlRollout(objective, env, unroll_length=30, unrolls=16, epochs=16, minibatches=32)
    tracker = _Tracker()
    trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.key(0), rollout=rollout, tracker=tracker)
    updates = collections * rollout.epochs * rollout.minibatches
    final = trainer.fit(
        _idle(len(jax.devices())), steps=updates, log_every=rollout.epochs * rollout.minibatches
    )
    returns = [
        (row["seconds"], row["episode_return"])
        for row in tracker.scalars
        if "episode_return" in row and np.isfinite(row["episode_return"])
    ]
    reached = next((seconds for seconds, value in returns if value >= target), None)
    greedy = episode_return(objective, playground(task, 128), final.variables, jax.random.key(1), 1000)
    report = {
        "task": task,
        "environment_steps": collections * 16 * 30 * envs,
        "updates": updates,
        "returns": returns,
        "target": target,
        "seconds_to_target": reached,
        "greedy_return": greedy,
        "device": str(jax.devices()[0]),
    }
    (out / "ppo.json").write_text(json.dumps(report, indent=1))
    return report


def laprop(lr: float = 4e-5, warmup: int = 1000) -> optax.GradientTransformation:
    """DreamerV3's optimizer (dreamerv3/agent.py:342-379): adaptive gradient clipping at 0.3, the RMS
    of 0.999 with epsilon 1e-20, momentum 0.9 bias-corrected, the rate ramped over `warmup` updates."""

    def clip_by_agc(clip: float = 0.3, floor: float = 1e-3) -> optax.GradientTransformation:
        def update(updates, state, params=None):
            def one(param, update):
                limit = clip * jnp.maximum(floor, jnp.linalg.norm(param.ravel()))
                return update / jnp.maximum(1.0, jnp.linalg.norm(update.ravel()) / limit)

            return jax.tree.map(one, params, updates), state

        return optax.GradientTransformation(lambda params: optax.EmptyState(), update)

    schedule = optax.join_schedules(
        [optax.linear_schedule(0.0, lr, warmup), optax.constant_schedule(lr)], [warmup]
    )
    return optax.chain(
        clip_by_agc(),
        optax.scale_by_rms(0.999, eps=1e-20, bias_correction=True),
        optax.trace(0.9),
        optax.scale_by_learning_rate(schedule),
    )


def dreamer(
    out: Path, *, task: str = "CartpoleBalance", envs: int = 16, environment_steps: int = 100_000
) -> dict:
    """DreamerV3's DM Control proprio preset (configs.yaml `dmc_proprio`): the 1M model, 16
    environments, windows of 64 in batches of 16, a train ratio of 1024."""
    from dew.objectives.rl.dreamer import DreamerRollout, DreamerV3Objective, WorldModel, episodes_return
    from dew.rl.playground import Playground, playground
    from dew.training import Trainer

    probe = Playground.load(task)
    env = playground(task, envs)
    objective = DreamerV3Objective(WorldModel(int(probe.env.observation_size), probe.action_size))
    rollout = DreamerRollout(
        objective,
        env,
        capacity=environment_steps // envs + 1000,
        batch=16,
        length=64,
        train_ratio=1024.0,
        prefill=64,
    )
    tracker = _Tracker()
    trainer = Trainer(objective, laprop(), key=jax.random.key(0), rollout=rollout, tracker=tracker)
    updates = int(environment_steps * rollout.train_ratio / (rollout.batch * rollout.length))
    final = trainer.fit(_idle(len(jax.devices())), steps=updates, log_every=1000)
    evaluated = episodes_return(objective, playground(task, envs), final.variables, jax.random.key(1), 1001)
    report = {
        "task": task,
        "environment_steps": rollout.environment_steps,
        "updates": updates,
        "return": evaluated,
        "paper_return_at_500k": 990,
        "gap_to_paper": 990 - evaluated,
        "training": [
            {key: row[key] for key in ("step", "seconds", "episode_return") if key in row}
            for row in tracker.scalars
        ],
        "device": str(jax.devices()[0]),
    }
    (out / "dreamer.json").write_text(json.dumps(report, indent=1))
    return report


def _shapes(rows: int, size: int, seed: int) -> np.ndarray:
    """`rows` images `[size, size, 1]` in [0, 255]: a bright rectangle at a random place and size on
    a dark ground."""
    rng = np.random.default_rng(seed)
    images = np.zeros((rows, size, size, 1), np.uint8)
    for image in images:
        h, w = rng.integers(3, size // 2, 2)
        y, x = rng.integers(0, size - h), rng.integers(0, size - w)
        image[y : y + h, x : x + w] = rng.integers(128, 256)
    return images


def nft(out: Path, *, pretrain: int = 3000, updates: int = 300, groups: int = 8, prompts: int = 16) -> dict:
    """One unconditional flow DiT pretrained on rectangles, then trained apart by Flow-GRPO and by
    DiffusionNFT toward images whose mean is 0.5 in [-1, 1], each over the same 10-step sampler,
    groups, prompts and updates at the same rate: the rewards each reached, update by update."""
    from dew.diffusion import presets
    from dew.inputs import Field, InputSpec
    from dew.nn.backbones.dit import SimpleDiT
    from dew.objectives.diffusion.objective import DiffusionObjective
    from dew.objectives.rl.flow import FlowGRPOObjective, FlowRollout
    from dew.objectives.rl.nft import DiffusionNFTObjective, NFTRollout
    from dew.training import Trainer

    inputs = InputSpec(Field("image", (16, 16, 1)))
    model = SimpleDiT(
        output_channels=1, patch_size=2, emb_features=128, num_layers=4, num_heads=4, mlp_ratio=4
    )
    images = _shapes(4096, 16, 0)
    data = Loader(
        train=lambda partition: ({"image": images[i : i + 64]} for i in itertools.cycle(range(0, 4096, 64))),
        val=None,
        records=4096,
        batch=64,
    )
    base = Trainer(
        DiffusionObjective(model, presets.Flow(), inputs, guidance=None),
        optax.adam(3e-4),
        key=jax.random.key(0),
    ).fit(data, steps=pretrain, log_every=pretrain)
    start = jax.device_get(base.variables)

    def reward(images, context):
        return -np.abs(images.mean(axis=(1, 2, 3)) - 0.5)

    source = {"image": np.zeros((prompts, 16, 16, 1), np.uint8)}
    curves = {}
    for name in ("flow_grpo", "diffusion_nft"):
        if name == "flow_grpo":
            objective = FlowGRPOObjective(
                model, presets.Flow(), inputs, guidance=None, steps=10, variables=start
            )
            rollout = FlowRollout(objective, reward, groups=groups, steps=11)
        else:
            objective = DiffusionNFTObjective(model, presets.Flow(), inputs, steps=10, variables=start)
            rollout = NFTRollout(objective, reward, groups=groups, steps=11)
        tracker = _Tracker()
        Trainer(objective, optax.adam(1e-4), key=jax.random.key(1), rollout=rollout, tracker=tracker).fit(
            Loader(
                train=lambda partition: itertools.repeat(source), val=None, records=prompts, batch=prompts
            ),
            steps=updates,
            log_every=1,
        )
        curves[name] = [
            (row["step"], row["seconds"], row["reward"]) for row in tracker.scalars if "reward" in row
        ]
    report = {
        "pretrain_steps": pretrain,
        "updates": updates,
        "groups": groups,
        "prompts": prompts,
        "reward": "minus |image mean - 0.5|",
        "curves": curves,
        "final_reward": {
            name: float(np.mean([r for _, _, r in curve[-20:]])) for name, curve in curves.items()
        },
        "device": str(jax.devices()[0]),
    }
    (out / "nft.json").write_text(json.dumps(report, indent=1))
    return report


def bouncing(rows: int, frames: int, size: int, seed: int) -> np.ndarray:
    """`rows` videos `[frames, size, size, 1]` in [-1, 1] of two balls of radius 2.5 bouncing off the
    walls at constant speeds."""
    rng = np.random.default_rng(seed)
    grid = np.stack(np.meshgrid(np.arange(size), np.arange(size), indexing="ij"), -1) + 0.5
    videos = np.full((rows, frames, size, size, 1), -1.0, np.float32)
    for video in videos:
        position = rng.uniform(3, size - 3, (2, 2))
        velocity = rng.uniform(-1.5, 1.5, (2, 2))
        for frame in video:
            for ball in range(2):
                inside = np.sum(np.square(grid - position[ball]), -1) <= 2.5**2
                frame[inside, 0] = 1.0
            position = position + velocity
            bounced = (position < 2.5) | (position > size - 2.5)
            velocity = np.where(bounced, -velocity, velocity)
            position = np.clip(position, 2.5, size - 2.5)
    return videos


def dreamer4(
    out: Path, *, tokenizer_steps: int = 4000, dynamics_steps: int = 6000, context: int = 4, horizon: int = 8
) -> dict:
    """The tokenizer trained on bouncing balls, then two dynamics models of one architecture on its
    latents, one by shortcut forcing and one by diffusion forcing, the same steps each; their open-loop
    predictions of `horizon` frames after `context` held-out frames, decoded and compared with the
    truth, at 4 sampling steps and at 64."""
    from dew.objectives.world_model import CausalTokenizer, Dynamics, TokenizerObjective, WorldModelObjective
    from dew.training import Trainer

    frames = context + horizon
    train, held = bouncing(2048, frames, 32, 0), bouncing(64, frames, 32, 1)

    def loader(videos):
        return Loader(
            train=lambda partition: (
                {"video": videos[i : i + 32]} for i in itertools.cycle(range(0, len(videos), 32))
            ),
            val=None,
            records=len(videos),
            batch=32,
        )

    tokenizer = CausalTokenizer((32, 32, 1), patch=4, width=128, depth=3, heads=4, latents=16, bottleneck=8)
    tokens = Trainer(TokenizerObjective(tokenizer), optax.adam(3e-4), key=jax.random.key(0)).fit(
        loader(train), steps=tokenizer_steps, log_every=tokenizer_steps
    )
    tokenizer_variables = jax.device_get(tokens.variables)
    reconstruction = float(np.mean(np.square(np.asarray(tokenizer.apply(tokenizer_variables, held)) - held)))
    dynamics = Dynamics(
        latents=16, bottleneck=8, spatial=16, width=256, depth=6, heads=4, levels=7, frames=frames
    )
    errors = {}
    for shortcut in (True, False):
        objective = WorldModelObjective(dynamics, tokenizer, tokenizer_variables, shortcut=shortcut)
        trained = Trainer(objective, optax.adam(3e-4), key=jax.random.key(1)).fit(
            loader(train), steps=dynamics_steps, log_every=dynamics_steps
        )
        latents = objective.latents(jnp.asarray(held))
        for steps in (4, 64):
            predicted = objective.generate(
                trained.variables, latents[:, :context], horizon, jax.random.key(2), steps=steps
            )
            decoded = np.asarray(
                tokenizer.apply(tokenizer_variables, predicted, method=CausalTokenizer.decode)
            )
            errors[f"{'shortcut' if shortcut else 'diffusion'}_forcing_{steps}_steps"] = float(
                np.mean(np.square(decoded - held[:, context:]))
            )
    report = {
        "tokenizer_reconstruction_mse": reconstruction,
        "open_loop_mse": errors,
        "context": context,
        "horizon": horizon,
        "tokenizer_steps": tokenizer_steps,
        "dynamics_steps": dynamics_steps,
        "device": str(jax.devices()[0]),
    }
    (out / "dreamer4.json").write_text(json.dumps(report, indent=1))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run", choices=("ppo", "dreamer", "nft", "dreamer4"))
    parser.add_argument("out", type=Path)
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    report = {"ppo": ppo, "dreamer": dreamer, "nft": nft, "dreamer4": dreamer4}[arguments.run](arguments.out)
    print(json.dumps({key: value for key, value in report.items() if not isinstance(value, (list, dict))}))


if __name__ == "__main__":
    main()
