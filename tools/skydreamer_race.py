"""The base check of the FPV pipeline: SkyDreamer's recipe (Verraest et al., arXiv 2510.14783) learning online
from scratch on its race (`dew.rl.skydreamer`) from the camera's RGB image, by collective thrust and body
rates, its world model decoding the privileged state; an environment run
(`dew.objectives.rl.config.EnvRunConfig`) under OUT, which the tool run again goes on from, judged greedy
every `--every` updates into OUT/base.json against the untrained policy judged alike (`untrained`): on
Table III's evaluation starts (each before the first gate) and gates (1 m in size), and on held-out
episodes of the training task, each from the same seed at every judgement.

    python tools/skydreamer_race.py OUT --assets A [--updates 125000] [--every 12500] [--realism]

The recipe at SkyDreamer's settings (Sec III-A): DreamerV3's size12m model (an RSSM of 2048, 256 hidden, 16
classes; 16 deep convolutions and 256 units) over a 64 x 64 image beside the body rates, rotor speeds and
flight plan, decoding the `state`, the rates and the rotor speeds in the image's place; `train_ratio` 128 over
16 environments, a replay of `REPLAY` steps, imagined returns bootstrapped from the slow critic (`slowtar`)
and the policy's mean smoothed at 0.002 (Eq 11), LaProp at 4e-5 after a 1000-update warmup. 125,000 updates
replay 128 steps for each of 1M taken. Not SkyDreamer's: its replay of 10M steps (of 64 x 64 masks, 4 KB a
step; this image's 12 KB a step holds 2M in 25 GB of the GPU's memory, and 10M would take 123 GB), its
`replay_context` of 16 (each window here starts its latent afresh), its later phases (a batch length of 256
after 8M steps, the entropy and rate lowered after 13M) and the race's departures (`dew.rl.skydreamer`: the
track, the camera's tilt, the actions, the image).

`--realism` trains the same recipe toward a real FPV quadrotor's flight (`REALISM`): an equidistant fisheye
lens 120 degrees across (`dew.rl.hyperlux`), drawn 96 pixels square from a 160-pixel pinhole so a gate 10 m
off still spans about 7 pixels; SkyDreamer's 44 ms of image and action delay as the command's latency; its
inverted loop and ladder tracks beside the circle, an episode's drawn alike; and randomization by the race's
curriculum (`ramp`), from none, raised a tenth whenever the evaluation's flights crash under a third of the
time. Each judgement is made at the level the run trains at, on each track alone, the untrained policy
judged alike at each level the run reaches. Its replay holds 1M steps of the larger image (28 GB)."""

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
from dew.rl.hyperlux import HyperLuxRenderer
from dew.rl.skydreamer import Sky, SkyDreamerRace

ENVIRONMENTS = 16
EVALUATION_SEED, HELD_OUT_SEED = 0xE7A1, 0x5EED
"""The seeds every judgement's episodes start from, apart from the training's (the run's key)."""


@dataclasses.dataclass(frozen=True)
class Setting:
    """What a run of the recipe races: its `sky`, its images `pixels` square, a `fisheye` lens (its field of
    view, the source pinhole's pixels) or a pinhole, its replay's steps, and whether a curriculum raises its
    randomization (`curriculum`); `judging`, how its report's points are judged, which a run going on in its
    OUT must share."""

    sky: Sky
    pixels: int
    fisheye: tuple[float, int] | None
    replay: int
    curriculum: bool
    judging: str

    def renderer(self, assets: Path, drones: int = ENVIRONMENTS) -> dict:
        """The record of the setting's renderer of `drones` cameras over `assets`."""
        return {"class": "dew.rl.hyperlux:HyperLuxRenderer",
                "fields": {"count": drones, "assets": str(assets), "size": self.pixels, **self._lens}}

    def race(self, assets: Path, sky: Sky, ramp: float) -> SkyDreamerRace:
        """A race of its own over `sky` at `ramp`, its renderer the setting's, to judge on."""
        renderer = HyperLuxRenderer(ENVIRONMENTS, assets, size=self.pixels, **self._lens)
        return SkyDreamerRace(renderer, sky, ramp=ramp)

    @property
    def _lens(self) -> dict:
        return {} if self.fisheye is None else {"fisheye": self.fisheye[0], "source": self.fisheye[1]}


BASE = Setting(Sky(), pixels=64, fisheye=None, replay=2_000_000, curriculum=False,
               judging="untrained at each level beside each point; Table III evaluation and held-out "
                       "training episodes, fixed seeds")
"""The base check: the circle seen through a 64-pixel pinhole, nothing randomized and no latency."""
REALISM = Setting(Sky(tracks=("circle", "inverted_loop", "ladder"), fov=120.0, latency=0.044), pixels=96,
                  fisheye=(120.0, 160), replay=1_000_000, curriculum=True,
                  judging="untrained at each level; each track's Table III evaluation and held-out training "
                          "episodes at the level trained at, fixed seeds")
"""Toward a real FPV quadrotor's flight (the module docstring)."""
SENSED = (("rates", 3), ("motors", 4), ("plan", 24))
"""What the encoder reads beside the image, each entry and its width."""
DECODED = (("state", 16, 256), ("rates", 3, 256), ("motors", 4, 256))
"""What the informed world model decodes, in the image's place (SkyDreamer Sec II-B: the information variable
leaves the image out, and the flight plan is no observation of the drone): the privileged state and the
sensed rates and rotor speeds, each two layers of 256 then its width."""


def run(out: Path, assets: Path, drones: int = ENVIRONMENTS, setting: Setting = BASE) -> EnvRunConfig:
    """The recipe's run under OUT (the module docstring) over `drones` drones seen through HyperLux's cameras
    over `assets`, racing the `setting`'s race."""
    # The race's fields set otherwise than Sky's defaults, as a record holds them (a tuple as a list).
    sky = {f.name: list(value) if isinstance(value, tuple) else value for f in dataclasses.fields(Sky)
           if f.name != "quadrotor" and (value := getattr(setting.sky, f.name)) != f.default}
    race = {"renderer": setting.renderer(assets, drones),
            **({"sky": {"class": "dew.rl.skydreamer:Sky", "fields": sky}} if sky else {})}
    return EnvRunConfig(
        env=EnvConfig("skydreamer", race),
        model=ModelConfig("dew.objectives.rl.dreamer:WorldModel", {
            "reads": "image", "proprio": [list(entry) for entry in SENSED],
            "auxiliary": [list(head) for head in DECODED], "deter": 2048, "hidden": 256, "stoch": 32,
            "classes": 16, "blocks": 8, "units": 256, "conv_depth": 16, "dtype": "bfloat16"}),
        objective=ObjectiveConfig("dew.objectives.rl.dreamer:DreamerV3Objective",
                                  {"slow_targets": True, "smoothness": 0.002, "scales": {"decoder": 0.0}}),
        rollout=RolloutConfig("dreamer", {"batch": 16, "length": 64, "train_ratio": 128.0,
                                          "capacity": setting.replay // drones, "prefill": 64}),
        optim=OptimConfig(optimizer="laprop", learning_rate=Linear(peak_value=4e-5, warmup_steps=1000,
                                                                   end_value=4e-5)),
        trainer=TrainerConfig(checkpoint_dir=str(out), log_every=500, eval_every=None, checkpoint_every=None,
                              keep=Keep(latest=2)))


def scored(simulator, objective, variables, key: jax.Array) -> dict:
    """Each drone's first episode flying `objective`'s greedy policy from fresh episodes of `simulator`
    for one episode's steps: its return, gates passed, whether it ended (a crash: the race ends no episode
    otherwise but by its cut) and its steps, averaged over the drones."""
    from dew.rl.env import view, views
    from dew.rl.evaluation import first_episode_rows
    from dew.rl.simulator import DrivenSimulator, simulator_rollout, simulator_start

    driven = DrivenSimulator(simulator)
    steps = simulator.sky.steps
    _, taken = simulator_rollout(driven, objective.greedy, variables,
                                 simulator_start(driven, key, objective.memory(simulator.count)), steps,
                                 reset=False)
    following = taken.following
    # A driven step holds the environment's observation, whose `passed` view counts the gates.
    passed = view(views(following.observation)["observation"], "passed")
    rows = jax.device_get(first_episode_rows(following, {"passed": passed}))
    gates = np.asarray(rows["passed"])
    return {"return": float(rows["return"].mean()), "gates": float(gates.mean()),
            "gates_max": int(gates.max()), "crashed": float(rows["terminated"].mean()),
            "steps": float(rows["steps"].mean())}


def trained(out: Path, assets: Path, updates: int, every: int, setting: Setting = BASE) -> dict:
    """The recipe trained to `updates` updates in chunks of `every`, going on from its checkpoints (its replay
    made again), each chunk's agent judged greedy (`judged`) into OUT/base.json beside the untrained policy
    judged alike: on the base check's race, on Table III's evaluation and held-out training episodes; with a
    curriculum, on each track alone at the level the chunk trained at, the level then raised a tenth, to 1,
    where the evaluation's flights crashed under a third of the time, the untrained judged at each level
    reached. A run going on goes on at the level its report reached."""
    from dew.objectives.rl.dreamer import DreamerRollout, DreamerV3Objective

    path = out / "base.json"
    report: dict = (json.loads(path.read_text()) if path.exists()
                    else {"points": [], "judging": setting.judging, "level": 0.0})
    if report.get("judging") != setting.judging:
        raise ValueError(f"{path} was judged otherwise than this run judges ({setting.judging}); go on in a "
                         "fresh OUT")
    done = 0
    if (out / "base").is_dir():
        with Checkpoints(str(out / "base")) as checkpoints:
            done = checkpoints.latest or 0
    prepared = run(out, assets, setting=setting).prepare()
    rollout, agent = prepared.rollout, prepared.objective
    assert isinstance(rollout, DreamerRollout) and isinstance(agent, DreamerV3Objective)
    assert isinstance(rollout.env, SkyDreamerRace)
    level = rollout.env.ramp = report["level"]
    tracks = setting.sky.tracks if setting.curriculum else (None,)

    def judged(variables, level: float) -> dict:
        """Each track's (or the race's) Table III evaluation (every start before the first gate, the gates'
        size 1 m) and held-out training episodes at `level`."""
        scores = {}
        for track in tracks:
            sky = setting.sky if track is None else dataclasses.replace(setting.sky, tracks=(track,))
            evaluation = setting.race(assets, dataclasses.replace(sky, anywhere=0.0, size=1.0), level)
            held_out = setting.race(assets, sky, level)
            scores[track or "race"] = {
                "evaluation": scored(evaluation, agent, variables, jax.random.key(EVALUATION_SEED)),
                "held_out": scored(held_out, agent, variables, jax.random.key(HELD_OUT_SEED))}
        return scores if setting.curriculum else scores["race"]

    untrained = report.setdefault("untrained", {})
    began, state = time.perf_counter(), None
    scored_at = {point["updates"] for point in report["points"]}
    while state is None or done < updates:
        if str(level) not in untrained:
            untrained[str(level)] = judged(agent.init(jax.random.key(0)), level)
            print(json.dumps({"untrained": level, **untrained[str(level)]}), flush=True)
        # A chunk checkpointed and stopped before its score is restored there and scored before the next.
        pending = state is None and done > 0 and done not in scored_at
        reached = done if pending else min(updates, (done // every + 1) * every)
        prepared = dataclasses.replace(prepared, run=prepared.run.override({"trainer.steps": reached}))
        state = prepared.train("base")
        done = int(state.step)
        if done in scored_at:
            continue
        point = {"updates": done, "environment_steps": rollout.environment_steps, "level": level,
                 "seconds": time.perf_counter() - began, "episode_return": rollout.metrics["episode_return"],
                 **judged(state.variables, level)}
        report["points"].append(point)
        scored_at.add(done)
        print(json.dumps(point), flush=True)
        crashed = [scores["evaluation"]["crashed"] for scores in point.values() if isinstance(scores, dict)]
        if setting.curriculum and np.mean(crashed) < 1 / 3 and level < 1.0:
            level = rollout.env.ramp = round(min(1.0, level + 0.1), 1)
        report["level"] = level
        # Written aside and moved over, so a stop mid-write leaves the last report whole.
        aside = path.with_suffix(".json.part")
        aside.write_text(json.dumps(report, indent=1))
        aside.replace(path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=125_000)
    parser.add_argument("--every", type=int, default=12_500)
    parser.add_argument("--realism", action="store_true", help="train toward a real FPV quadrotor's flight")
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    setting = REALISM if arguments.realism else BASE
    print(json.dumps(trained(arguments.out, arguments.assets, arguments.updates, arguments.every, setting)))


if __name__ == "__main__":
    main()
