"""Acceptance runs of Dew's environment, world-model and flow-RL stack, each writing a JSON report.

    python tools/rl_acceptance.py ppo OUT        PPO on Playground's CartpoleBalance: the training
                                                 episodes' return against wall-clock time, and the
                                                 greedy policy's return at the end.
    python tools/rl_acceptance.py dreamer OUT    DreamerV3 on CartpoleBalance at 100K environment
                                                 steps, against the 990 its paper reports at 500K
                                                 (arXiv 2301.04104v2, "Proprioceptive control scores").
    python tools/rl_acceptance.py nft OUT        DiffusionNFT against Flow-GRPO under matched
                                                 conditions: one flowers DiT, one reward model (CLIP),
                                                 the same prompts, samples and 500 updates each.
    python tools/rl_acceptance.py dreamer4 OUT   Dreamer 4's world model at small scale: shortcut
                                                 forcing against diffusion forcing on bouncing-ball
                                                 video, open-loop prediction error at 4 and 64 steps.

Each writes OUT/<run>.json. Playground's environments need the `playground` distribution, and the
flowers study Oxford Flowers 102 prepared as TFDS ArrayRecords (examples/train_flowers_tpu.py).
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
        """The training and rollout scalars by their own names, `train/` and `rollout/` dropped (a
        rollout's `episode_return` or `reward` is logged as `rollout/<name>`)."""
        named = {
            name.removeprefix("train/").removeprefix("rollout/"): float(value)
            for name, value in scalars.items()
        }
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
    # The RMS divides outside its root (opt.py:126-143); the momentum is an EMA, bias-corrected.
    return optax.chain(
        clip_by_agc(),
        optax.scale_by_rms(0.999, eps=1e-20, eps_in_sqrt=False, bias_correction=True),
        optax.ema(0.9, debias=True),
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
    evaluated = episodes_return(objective, playground(task, envs), final.variables, jax.random.key(1), 1000)
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


def flowers(size: int, batch: int, data: str) -> Loader:
    """Oxford Flowers 102's every image, at `size` pixels a side in batches of `batch`, prepared as
    TFDS ArrayRecords under `data` (examples/train_flowers.py)."""
    from dew.data import TFDS, presets

    return (
        TFDS("oxford_flowers102", split="all", path=data)
        .shuffle(seed=0, window=1024)
        .repeat()
        .pipe(presets.images, size, batch=batch, augment=True)
    )


def nft(
    out: Path,
    *,
    data: str = "~/.cache/dew/datasets/oxford_flowers102/2.1.1",
    pretrain: int = 30_000,
    updates: int = 500,
    groups: int = 8,
    prompts: int = 16,
    caption: str = "a photo of a red flower",
) -> dict:
    """The roadmap's comparison (docs/roadmap.md, [533]) as its planned acceptance runs it: one
    flowers DiT pretrained by flow matching, then fine-tuned apart by Flow-GRPO and by DiffusionNFT
    against one reward model, CLIP's score of
    each image against `caption`, `updates` steps each over the same 10-step sampler, prompts, groups
    and rate: each reward curve, and the steps each needed to reach the highest reward both reached."""
    from dew.artifacts import ImageGrid
    from dew.diffusion import presets
    from dew.eval.images import CLIPScore
    from dew.inputs import Field, InputSpec
    from dew.nn.backbones.dit import SimpleDiT
    from dew.objectives.diffusion.objective import DiffusionObjective
    from dew.objectives.rl.flow import FlowGRPOObjective, FlowRollout
    from dew.objectives.rl.nft import DiffusionNFTObjective, NFTRollout
    from dew.training import Trainer

    size = 64
    inputs = InputSpec(Field("image", (size, size, 3)))
    model = SimpleDiT(patch_size=4, emb_features=256, num_layers=8, num_heads=8, mlp_ratio=4)
    base = Trainer(
        DiffusionObjective(model, presets.Flow(), inputs, guidance=None),
        optax.adam(2e-4),
        key=jax.random.key(0),
    ).fit(flowers(size, 64, str(Path(data).expanduser())), steps=pretrain, log_every=pretrain // 10)
    start = jax.device_get(base.variables)
    metric = CLIPScore()

    def reward(images, context):
        captions = {"captions": np.asarray([caption] * len(images), object)}
        return np.asarray(metric.fn(ImageGrid(images), captions), np.float64)

    source = {"image": np.zeros((prompts, size, size, 3), np.uint8)}
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
        Trainer(objective, optax.adam(3e-5), key=jax.random.key(1), rollout=rollout, tracker=tracker).fit(
            Loader(
                train=lambda partition: itertools.repeat(source), val=None, records=prompts, batch=prompts
            ),
            steps=updates,
            log_every=1,
        )
        curves[name] = [
            (row["step"], row["seconds"], row["reward"]) for row in tracker.scalars if "reward" in row
        ]

    def smoothed(curve):
        return np.convolve(np.asarray([r for _, _, r in curve]), np.ones(10) / 10, mode="valid")

    common = min(float(smoothed(curve).max()) for curve in curves.values())
    reached = {name: int(np.argmax(smoothed(curve) >= common)) + 10 for name, curve in curves.items()}
    report = {
        "pretrain_steps": pretrain,
        "updates": updates,
        "groups": groups,
        "prompts": prompts,
        "reward": f"CLIP score against {caption!r}",
        "curves": curves,
        "common_reward": common,
        "steps_to_common_reward": reached,
        "first_reward": {name: float(np.mean([r for _, _, r in c[:10]])) for name, c in curves.items()},
        "final_reward": {name: float(np.mean([r for _, _, r in c[-20:]])) for name, c in curves.items()},
        "device": str(jax.devices()[0]),
    }
    (out / "nft.json").write_text(json.dumps(report, indent=1))
    return report


def bouncing(rows: int, frames: int, size: int, seed: int, radius: float = 2.5) -> np.ndarray:
    """`rows` videos `[frames, size, size, 1]` in [-1, 1] of two balls of `radius` bouncing off the
    walls at constant speeds."""
    rng = np.random.default_rng(seed)
    grid = np.stack(np.meshgrid(np.arange(size), np.arange(size), indexing="ij"), -1) + 0.5
    videos = np.full((rows, frames, size, size, 1), -1.0, np.float32)
    for video in videos:
        position = rng.uniform(radius + 0.5, size - radius - 0.5, (2, 2))
        velocity = rng.uniform(-1.5, 1.5, (2, 2))
        for frame in video:
            for ball in range(2):
                inside = np.sum(np.square(grid - position[ball]), -1) <= radius**2
                frame[inside, 0] = 1.0
            position = position + velocity
            bounced = (position < radius) | (position > size - radius)
            velocity = np.where(bounced, -velocity, velocity)
            position = np.clip(position, radius, size - radius)
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
