"""The base check of the FPV pipeline: SkyDreamer's recipe (Verraest et al., arXiv 2510.14783) learning online
from scratch on its race (`dew.rl.skydreamer`) from the camera's RGB image, by collective thrust and body
rates, its world model decoding the privileged state; an environment run
(`dew.objectives.rl.config.EnvRunConfig`) under OUT, which the tool run again goes on from, judged greedy
every `--every` updates into OUT/base.json.

    python tools/skydreamer_race.py OUT --assets A [--updates 125000] [--every 12500]

The recipe at SkyDreamer's settings (Sec III-A): DreamerV3's size12m model (an RSSM of 2048, 256 hidden, 16
classes; 16 deep convolutions and 256 units) over a 64 x 64 image beside the body rates, rotor speeds and
flight plan, decoding the `state` and those three; `train_ratio` 128 over 16 environments, a replay of 10M
steps, imagined returns bootstrapped from the slow critic (`slowtar`) and the policy's mean smoothed at 0.002
(Eq 11), LaProp at 4e-5 after a 1000-update warmup. 125,000 updates replay 128 steps for each of 1M taken.
Not SkyDreamer's: its `replay_context` of 16 (each window here starts its latent afresh), its later phases
(a batch length of 256 after 8M steps, the entropy and rate lowered after 13M) and the race's departures
(`dew.rl.skydreamer`: the track, the camera's tilt, the actions, the image)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
import numpy as np

from dew.checkpoints import Checkpoints, Keep
from dew.config import ObjectiveConfig, TrainerConfig
from dew.objectives.rl.config import EnvConfig, EnvRunConfig, RolloutConfig
from dew.optim import Linear, OptimConfig
from dew.registry import ModelConfig

ENVIRONMENTS, PIXELS = 16, 64
SENSED = (("rates", 3), ("motors", 4), ("plan", 24))
"""What the encoder reads beside the image, each entry and its width."""
DECODED = (("state", 16, 256), *((name, width, 256) for name, width in SENSED))
"""What the world model decodes besides the image (the informed decoder): the privileged state and the
sensed entries, each two layers of 256 then its width."""


def run(out: Path, assets: Path, drones: int = ENVIRONMENTS) -> EnvRunConfig:
    """The base check's run under OUT: SkyDreamer's recipe (the module docstring) over `drones` drones seen
    through HyperLux's cameras over `assets`."""
    renderer = {"class": "dew.rl.hyperlux:HyperLuxRenderer",
                "fields": {"count": drones, "assets": str(assets), "size": PIXELS}}
    return EnvRunConfig(
        env=EnvConfig("skydreamer", {"renderer": renderer}),
        model=ModelConfig("dew.objectives.rl.dreamer:WorldModel", {
            "reads": "image", "proprio": [list(entry) for entry in SENSED],
            "auxiliary": [list(head) for head in DECODED], "deter": 2048, "hidden": 256, "stoch": 32,
            "classes": 16, "blocks": 8, "units": 256, "conv_depth": 16, "dtype": "bfloat16"}),
        objective=ObjectiveConfig("dew.objectives.rl.dreamer:DreamerV3Objective",
                                  {"slow_targets": True, "smoothness": 0.002}),
        rollout=RolloutConfig("dreamer", {"batch": 16, "length": 64, "train_ratio": 128.0,
                                          "capacity": 10_000_000 // drones, "prefill": 64}),
        optim=OptimConfig(optimizer="laprop", learning_rate=Linear(peak_value=4e-5, warmup_steps=1000,
                                                                   end_value=4e-5)),
        trainer=TrainerConfig(checkpoint_dir=str(out), log_every=500, eval_every=None, checkpoint_every=None,
                              keep=Keep(latest=2)))


def scored(simulator, objective, variables, key: jax.Array) -> dict:
    """Each drone's first episode flying `objective`'s greedy policy from fresh episodes of `simulator`
    for one episode's steps: its return, gates passed, whether it ended (a crash: the race ends no episode
    otherwise but by its cut) and its steps, averaged over the drones."""
    from dew.rl.simulator import DrivenSimulator, simulator_rollout, simulator_start

    driven = DrivenSimulator(simulator)
    steps = simulator.sky.steps
    _, taken = simulator_rollout(driven, objective.greedy, variables,
                                 simulator_start(driven, key, objective.memory(simulator.count)), steps,
                                 reset=False)
    observed = jax.device_get(taken.following.observation)
    last, terminal = np.asarray(observed["last"]), np.asarray(observed["terminal"])
    ended = np.cumsum(last, 0) - last > 0
    reward = np.where(ended, 0, np.asarray(observed["reward"]))
    gates = np.where(ended, 0, np.asarray(observed["observation"]["passed"])).max(0)
    return {"return": float(reward.sum(0).mean()), "gates": float(gates.mean()),
            "gates_max": int(gates.max()), "crashed": float((terminal & ~ended).any(0).mean()),
            "steps": float((~ended).sum(0).mean())}


def trained(out: Path, assets: Path, updates: int, every: int) -> dict:
    """The base check trained to `updates` updates in chunks of `every`, going on from its checkpoints (its
    replay made again), each chunk's agent judged greedy on a race of its own into OUT/base.json."""
    from dew.objectives.rl.dreamer import DreamerRollout, DreamerV3Objective

    path = out / "base.json"
    report = json.loads(path.read_text()) if path.exists() else {"points": []}
    done = 0
    if (out / "base").is_dir():
        with Checkpoints(str(out / "base")) as checkpoints:
            done = checkpoints.latest or 0
    prepared = run(out, assets).prepare()
    rollout, agent = prepared.rollout, prepared.objective
    assert isinstance(rollout, DreamerRollout) and isinstance(agent, DreamerV3Objective)
    judged = EnvConfig("skydreamer", {"renderer": {"class": "dew.rl.hyperlux:HyperLuxRenderer", "fields": {
        "count": ENVIRONMENTS, "assets": str(assets), "size": PIXELS}}}).build()
    began = time.perf_counter()
    while done < updates:
        reached = min(updates, (done // every + 1) * every)
        prepared = dataclasses.replace(prepared, run=prepared.run.override({"trainer.steps": reached}))
        state = prepared.train("base")
        done = int(state.step)
        point = {"updates": done, "environment_steps": rollout.environment_steps,
                 "seconds": time.perf_counter() - began, "episode_return": rollout.metrics["episode_return"],
                 **scored(judged, agent, state.variables, jax.random.key(done))}
        report["points"].append(point)
        print(json.dumps(point), flush=True)
        path.write_text(json.dumps(report, indent=1))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=125_000)
    parser.add_argument("--every", type=int, default=12_500)
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    print(json.dumps(trained(arguments.out, arguments.assets, arguments.updates, arguments.every)))


if __name__ == "__main__":
    main()
