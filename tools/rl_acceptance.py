"""Acceptance runs of Dew's environment, world-model and flow-RL stack, each writing a JSON report.

    python tools/rl_acceptance.py ppo OUT        PPO on Playground's CartpoleBalance: the training
                                                 episodes' return against wall-clock time, and the
                                                 greedy policy's return at the end.
    python tools/rl_acceptance.py dreamer OUT    DreamerV3 on CartpoleBalance at 100K environment
                                                 steps, against the 990 its paper reports at 500K
                                                 (arXiv 2301.04104v2, "Proprioceptive control scores"):
                                                 the agent's return, its policy's draws, and beside it
                                                 the greedy policy's.
    python tools/rl_acceptance.py nft OUT        DiffusionNFT against Flow-GRPO under matched
                                                 conditions: one flowers DiT, one reward model (CLIP),
                                                 the same prompts, samples and 500 updates each.
    python tools/rl_acceptance.py dreamer4 OUT   Dreamer 4's world model at small scale: shortcut
                                                 forcing against diffusion forcing on bouncing-ball
                                                 video, open-loop prediction error at 4 and 64 steps.

Each writes OUT/<run>.json. PPO and DreamerV3 train as environment runs (`EnvRunConfig`, the `ppo_run` and
`dreamer_run` records) in the directories OUT/ppo and OUT/dreamer, and Dreamer 4's tokenizer and its two
dynamics models as world-model runs (`WorldModelRunConfig`, the `dreamer4_stages` records) in OUT/dreamer4-
tokenizer, -shortcut and -diffusion over the clips stored in OUT/dreamer4-videos. A rerun refuses them: a
run that held its steps would read on from them, and its report would join two runs' curves. Playground's
environments need the `playground` distribution, and the flowers study Oxford Flowers 102 prepared as TFDS
ArrayRecords (examples/train_flowers_tpu.py).
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

from dew.config import ObjectiveConfig, TrainerConfig
from dew.config.world_model import TokenizerConfig, WorldModelRunConfig
from dew.data.dataset import Loader
from dew.data.pipeline import Pipeline
from dew.objectives.rl.config import EnvConfig, EnvRunConfig, EpisodesConfig, RolloutConfig
from dew.optim import Linear, OptimConfig
from dew.registry import ModelConfig
from dew.training import LocalTracker

CONTROL, DREAMER = "dew.objectives.rl.control", "dew.objectives.rl.dreamer"
"""The modules of the agents' networks and objectives, which a record names by import path."""
GREEDY, DRAWN = "greedy", "drawn"
"""The validation splits the acceptance runs score on, which log `greedy/return` and `drawn/return`: the
policy's mean, and the draws it acts with in training, which a DreamerV3 return is published as."""


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


def tracked(directory: Path, began: float, *splits: str) -> tuple[list[dict], tuple[float, ...]]:
    """The training reports of the environment run in `directory`, each its step, the seconds since the
    wall-clock second `began` and its scalars by their own names, `train/` and `rollout/` dropped (a
    rollout's `episode_return` is logged as `rollout/episode_return`), and the return each of `splits`
    scored once, at the run's end (`<split>/return`)."""
    with LocalTracker(directory / "tracking") as journal:
        reports = journal.timed()
    curve = []
    for step, at, scalars in reports:
        if "train/loss" in scalars:
            named = {
                name.removeprefix("train/").removeprefix("rollout/"): value
                for name, value in scalars.items()
            }
            curve.append({"step": step, "seconds": at - began, **named})
    returns = []
    for split in splits:
        (scored,) = [scalars[f"{split}/return"] for _, _, scalars in reports if f"{split}/return" in scalars]
        returns.append(scored)
    return curve, tuple(returns)


def ppo_run(
    out: Path, *, task: str = "CartpoleBalance", envs: int = 2048, collections: int = 20
) -> EnvRunConfig:
    """Brax's DM Control PPO settings (Playground's dm_control_suite_params): 2048 environments,
    unrolls of 30, 16 passes of 32 minibatches over each collection of 16 unrolls, discount 0.995,
    rewards scaled by 10, entropy 1e-2, clip 0.3, Adam at 1e-3, each episode cut after 1000 steps; scored
    greedy over 1000 steps of 128 environments. It logs once a collection."""
    rollout = {"unroll_length": 30, "unrolls": 16, "epochs": 16, "minibatches": 32}
    every = rollout["epochs"] * rollout["minibatches"]
    playground = {"name": task}
    return EnvRunConfig(
        env=EnvConfig("playground", playground, count=envs, episode_length=1000),
        model=ModelConfig(f"{CONTROL}:ActorCritic"),
        objective=ObjectiveConfig(
            f"{CONTROL}:ControlPPOObjective",
            {"discount": 0.995, "reward_scaling": 10.0, "entropy_cost": 1e-2, "clip": 0.3},
        ),
        rollout=RolloutConfig("control", rollout),
        optim=OptimConfig(optimizer="adam", learning_rate=1e-3),
        trainer=TrainerConfig(
            checkpoint_dir=str(out),
            steps=collections * every,
            log_every=every,
            eval_every=None,
            checkpoint_every=None,
            resume="never",
        ),
        episodes={
            GREEDY: EpisodesConfig(1000, EnvConfig("playground", playground, count=128, episode_length=1000))
        },
    )


def ppo(
    out: Path,
    *,
    task: str = "CartpoleBalance",
    envs: int = 2048,
    collections: int = 20,
    target: float = 900.0,
) -> dict:
    """`ppo_run` trained and reported: the training episodes' return against the seconds the run took, the
    seconds to `target`, and the greedy policy's return at the end."""
    prepared = ppo_run(out, task=task, envs=envs, collections=collections).prepare()
    began = time.time()
    prepared.train("ppo")
    curve, (greedy,) = tracked(out / "ppo", began, GREEDY)
    returns = [
        (row["seconds"], row["episode_return"])
        for row in curve
        if "episode_return" in row and np.isfinite(row["episode_return"])
    ]
    reached = next((seconds for seconds, value in returns if value >= target), None)
    report = {
        "task": task,
        "environment_steps": collections * 16 * 30 * envs,
        "updates": prepared.run.trainer.steps,
        "returns": returns,
        "target": target,
        "seconds_to_target": reached,
        "greedy_return": greedy,
        "device": str(jax.devices()[0]),
    }
    (out / "ppo.json").write_text(json.dumps(report, indent=1))
    return report


def dreamer_run(
    out: Path, *, task: str = "CartpoleBalance", envs: int = 16, environment_steps: int = 100_000
) -> EnvRunConfig:
    """DreamerV3's DM Control proprio preset (configs.yaml `dmc_proprio`): the 1M model, 16
    environments cut after 1000 steps, windows of 64 in batches of 16, a train ratio of 1024, LaProp at
    4e-5 ramped over the first 1000 updates; scored over 1000 steps of the same environments, by the
    agent's draws and by its greedy policy. It logs every 1000 updates."""
    rollout = {
        "capacity": environment_steps // envs + 1000,
        "batch": 16,
        "length": 64,
        "train_ratio": 1024.0,
        "prefill": 64,
    }
    updates = int(environment_steps * rollout["train_ratio"] / (rollout["batch"] * rollout["length"]))
    return EnvRunConfig(
        env=EnvConfig("playground", {"name": task}, count=envs, episode_length=1000),
        model=ModelConfig(f"{DREAMER}:WorldModel"),
        objective=ObjectiveConfig(f"{DREAMER}:DreamerV3Objective"),
        rollout=RolloutConfig("dreamer", rollout),
        optim=OptimConfig(
            optimizer="laprop", learning_rate=Linear(peak_value=4e-5, warmup_steps=1000, end_value=4e-5)
        ),
        trainer=TrainerConfig(
            checkpoint_dir=str(out),
            steps=updates,
            log_every=1000,
            eval_every=None,
            checkpoint_every=None,
            resume="never",
        ),
        episodes={DRAWN: EpisodesConfig(1000, drawn=True), GREEDY: EpisodesConfig(1000)},
    )


def dreamer(
    out: Path, *, task: str = "CartpoleBalance", envs: int = 16, environment_steps: int = 100_000
) -> dict:
    """`dreamer_run` trained and reported: the agent's return, its draws', against the paper's, the greedy
    policy's beside it, and the training episodes' return along the run."""
    from dew.objectives.rl.dreamer import DreamerRollout

    prepared = dreamer_run(out, task=task, envs=envs, environment_steps=environment_steps).prepare()
    rollout = prepared.rollout
    assert isinstance(rollout, DreamerRollout)
    began = time.time()
    prepared.train("dreamer")
    curve, (drawn, greedy) = tracked(out / "dreamer", began, DRAWN, GREEDY)
    report = {
        "task": task,
        "environment_steps": rollout.environment_steps,
        "updates": prepared.run.trainer.steps,
        "return": drawn,
        "greedy_return": greedy,
        "paper_return_at_500k": 990,
        "gap_to_paper": 990 - drawn,
        "training": [
            {key: row[key] for key in ("step", "seconds", "episode_return") if key in row} for row in curve
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


BALL = 2.5
"""A bouncing ball's radius in pixels."""


def bouncing(rows: int, frames: int, size: int, seed: int) -> np.ndarray:
    """`rows` videos `[frames, size, size, 1]` in [-1, 1] of two balls of radius `BALL`, their edges
    anti-aliased, bouncing off the walls at constant speeds."""
    radius = BALL
    rng = np.random.default_rng(seed)
    grid = np.stack(np.meshgrid(np.arange(size), np.arange(size), indexing="ij"), -1) + 0.5
    videos = np.full((rows, frames, size, size, 1), -1.0, np.float32)
    for video in videos:
        position = rng.uniform(radius + 0.5, size - radius - 0.5, (2, 2))
        velocity = rng.uniform(-1.5, 1.5, (2, 2))
        for frame in video:
            for ball in range(2):
                # Each pixel lit by the share of it the ball covers, its edge anti-aliased over a pixel.
                distance = np.sqrt(np.sum(np.square(grid - position[ball]), -1))
                frame[..., 0] = np.maximum(frame[..., 0], 2 * np.clip(radius + 0.5 - distance, 0, 1) - 1)
            position = position + velocity
            bounced = (position < radius) | (position > size - radius)
            velocity = np.where(bounced, -velocity, velocity)
            position = np.clip(position, radius, size - radius)
    return videos


def bouncing_clips(directory: Path, frames: int) -> Pipeline:
    """The 2048 training videos `bouncing` makes (seed 0) of `frames` frames of 32 pixels, stored as one
    ArrayRecord shard under `directory`, whose export states their shape, and read as a chain of batches of
    32, in the order they are stored, repeating: what a run over data records, which a run of in-memory
    arrays cannot, so a stage reads the saved tokenizer's run back from its record."""
    from dew.data import Files, Memory
    from dew.data.sinks import export

    videos = bouncing(2048, frames, 32, 0)
    export(
        Memory({"video": list(videos)}),
        str(directory),
        format="array_record",
        rows_per_shard=len(videos),
        metadata={"fields": {"video": [frames, 32, 32, 1]}},
    )
    return Files(str(directory)).repeat().batch(32)


def dreamer4_stages(
    out: Path,
    clips: Pipeline,
    *,
    frames: int,
    tokenizer_steps: int = 4000,
    dynamics_steps: int = 6000,
) -> dict[str, WorldModelRunConfig]:
    """Dreamer 4's world model at small scale as world-model runs over `clips` of `frames` frames, each
    under Adam at 3e-4: the tokenizer ("tokenizer", `tokenizer_steps` steps, run key 0), then two dynamics
    models of one architecture over the tokenizer's latents at its last step, one trained by shortcut
    forcing ("shortcut") and one by diffusion forcing ("diffusion"), `dynamics_steps` steps each (run key
    1)."""
    world = "dew.objectives.world_model"
    adam = OptimConfig(optimizer="adam", learning_rate=3e-4)

    def trainer(steps: int, key: int) -> TrainerConfig:
        return TrainerConfig(
            checkpoint_dir=str(out),
            steps=steps,
            log_every=steps,
            eval_every=None,
            checkpoint_every=None,
            key=key,
            resume="never",
        )

    tokenizer = ModelConfig(
        f"{world}:CausalTokenizer",
        {"patch": 4, "width": 128, "depth": 3, "heads": 4, "latents": 16, "bottleneck": 8},
    )
    stages = {
        "tokenizer": WorldModelRunConfig(
            data=clips, model=tokenizer, optim=adam, trainer=trainer(tokenizer_steps, 0)
        )
    }
    dynamics = ModelConfig(
        f"{world}:Dynamics",
        {"spatial": 16, "width": 256, "depth": 6, "heads": 4, "levels": 7, "frames": frames},
    )
    saved = TokenizerConfig(str(out / "dreamer4-tokenizer"), tokenizer_steps)
    for name, shortcut in (("shortcut", True), ("diffusion", False)):
        stages[name] = WorldModelRunConfig(
            data=clips,
            model=dynamics,
            objective=ObjectiveConfig(f"{world}:WorldModelObjective", {"shortcut": shortcut}),
            tokenizer=saved,
            optim=adam,
            trainer=trainer(dynamics_steps, 1),
        )
    return stages


def dreamer4(
    out: Path, *, tokenizer_steps: int = 4000, dynamics_steps: int = 6000, context: int = 4, horizon: int = 8
) -> dict:
    """The tokenizer trained on bouncing balls, then two dynamics models of one architecture on its
    latents, one by shortcut forcing and one by diffusion forcing, the same steps each (the world-model
    runs `dreamer4_stages` records, under OUT/dreamer4-<stage>); their open-loop predictions of `horizon`
    frames after `context` held-out frames, decoded and compared with the truth, at 4 sampling steps and
    at 64."""
    from dew.objectives.world_model import CausalTokenizer, TokenizerObjective, WorldModelObjective

    frames = context + horizon
    held = bouncing(64, frames, 32, 1)
    stages = dreamer4_stages(
        out,
        bouncing_clips(out / "dreamer4-videos", frames),
        frames=frames,
        tokenizer_steps=tokenizer_steps,
        dynamics_steps=dynamics_steps,
    )
    prepared = stages["tokenizer"].prepare()
    tokens = prepared.train("dreamer4-tokenizer")
    assert isinstance(prepared.objective, TokenizerObjective)
    tokenizer = prepared.objective.model
    tokenizer_variables = jax.device_get(tokens.variables)
    reconstruction = float(np.mean(np.square(np.asarray(tokenizer.apply(tokenizer_variables, held)) - held)))
    errors = {}
    for name in ("shortcut", "diffusion"):
        prepared = stages[name].prepare()
        trained = prepared.train(f"dreamer4-{name}")
        objective = prepared.objective
        assert isinstance(objective, WorldModelObjective)
        latents = objective.latents(trained.variables, jnp.asarray(held))
        for steps in (4, 64):
            predicted = objective.generate(
                trained.variables, latents[:, :context], horizon, jax.random.key(2), steps=steps
            )
            decoded = np.asarray(
                tokenizer.apply(tokenizer_variables, predicted, method=CausalTokenizer.decode)
            )
            errors[f"{name}_forcing_{steps}_steps"] = float(np.mean(np.square(decoded - held[:, context:])))
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
