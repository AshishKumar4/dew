"""The base check of the FPV pipeline: SkyDreamer's recipe (Verraest et al., arXiv 2510.14783) learning online
from scratch on its race (`dew.rl.skydreamer`) from the camera's RGB image, by collective thrust and body
rates, its world model decoding the privileged state; an environment run
(`dew.config.env.EnvRunConfig`) under OUT, which the tool run again goes on from, trained to
`--steps` environment steps and judged greedy about every `--every` into OUT/base.json against the
untrained policy judged alike (`untrained`): on Table III's evaluation starts (each before the first gate)
and gates (1 m in size), and on held-out episodes of the training task, each from the same seed at every
judgement.

    python tools/skydreamer_race.py OUT --assets A [--steps 1000000] [--every 100000] [--realism]

The recipe at SkyDreamer's settings (Sec III-A): DreamerV3's size12m model (an RSSM of 2048, 256 hidden, 16
classes; 16 deep convolutions and 256 units) over a 64 x 64 image beside the body rates, rotor speeds and
flight plan, decoding the `state`, the rates and the rotor speeds in the image's place; `train_ratio` 128 over
16 environments, a replay of `REPLAY` steps, imagined returns bootstrapped from the slow critic (`slowtar`)
and the policy's mean smoothed at 0.002 (Eq 11), LaProp at 4e-5 after a 1000-update warmup. 125,000 updates
replay 128 steps for each of 1M taken. Not SkyDreamer's: its replay of 10M steps (of 64 x 64 masks, 4 KB a
step; this image's 12 KB a step holds 2M in 25 GB of the GPU's memory, and 10M would take 123 GB), its
`replay_context` of 16 (each window here starts its latent afresh), its later phases (a batch length of 256
after 8M of its 17M steps, the entropy and rate lowered after 13M), which the realism run takes, and the
race's departures (`dew.rl.skydreamer`: the track, the camera's tilt, the actions, the image).

`--realism` trains the same recipe toward a real FPV quadrotor's flight (`REALISM`): an equidistant fisheye
lens 120 degrees across (`dew.rl.hyperlux`), drawn 96 pixels square from a 160-pixel pinhole so a gate 10 m
off still spans about 7 pixels; SkyDreamer's 44 ms of image and action delay as the command's latency; its
inverted loop and ladder tracks beside the circle, an episode's drawn alike; and randomization by the race's
curriculum (`ramp`), each track's level from none, raised a tenth whenever that track's evaluation flights
crash under a third of the time, so a hard track holds back its own level alone. Each judgement is made
on each track alone at the level it trains at, the untrained policy judged alike at each level a track
reaches. Its replay holds 1M steps of the larger image (28 GB). It trains in SkyDreamer's three phases
(`PHASES`), their starts scaled from its 17M steps to the 4M this run is trained to.

A phase begins at the first judgement at or past its start; each window draws a phase's `length` steps,
so an update takes `length / 8` environment steps at the train ratio of 128 over 16 windows."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import jax
import numpy as np

from dew.checkpoints import Checkpoints, Keep
from dew.config import ObjectiveConfig, TrainerConfig
from dew.config.env import EnvConfig, EnvRunConfig, RolloutConfig
from dew.optim import Linear, OptimConfig
from dew.registry import ModelConfig
from dew.rl.hyperlux import HyperLuxRenderer
from dew.rl.skydreamer import Sky, SkyDreamerRace

ENVIRONMENTS = 16
WINDOWS, TRAIN_RATIO = 16, 128.0
"""An update's windows and the steps each environment step is replayed (DreamerV3's `train_ratio`)."""
EVALUATION_SEED, HELD_OUT_SEED = 0xE7A1, 0x5EED
"""The seeds every judgement's episodes start from, apart from the training's (the run's key)."""


@dataclasses.dataclass(frozen=True)
class Phase:
    """A stage of the recipe's training from `steps` environment steps on: the `length` of the windows its
    world model trains on, the policy's `entropy` coefficient and the learning rate."""

    steps: int
    length: int
    entropy: float
    learning_rate: float


WARMUP = Phase(0, length=64, entropy=3e-4, learning_rate=4e-5)
"""SkyDreamer's first phase, DreamerV3's defaults (Sec III-A)."""
PHASES = (WARMUP, Phase(1_880_000, length=256, entropy=3e-4, learning_rate=4e-5),
          Phase(3_060_000, length=256, entropy=1e-5, learning_rate=2e-6))
"""SkyDreamer's phases (Sec III-A): windows of 256 steps from its 8M steps on, for the long dependencies
identifying the drone's parameters needs, then the entropy and learning rate lowered from 13M on to fine-tune
the policy; their starts scaled from its 17M steps to 4M (8/17 and 13/17 of them)."""


@dataclasses.dataclass(frozen=True)
class Setting:
    """What a run of the recipe races: its `sky`, its images `pixels` square, a `fisheye` lens (its field of
    view, the source pinhole's pixels) or a pinhole, its replay's steps, whether a curriculum raises each
    track's randomization (`curriculum`) and the `phases` it trains in, the first from the start; `judging`,
    how its report's points are judged, which a run going on in its OUT must share."""

    sky: Sky
    pixels: int
    fisheye: tuple[float, int] | None
    replay: int
    curriculum: bool
    judging: str
    phases: tuple[Phase, ...] = (WARMUP,)

    def phase(self, steps: int) -> Phase:
        """The phase a run trains in once it has taken `steps` environment steps."""
        return [phase for phase in self.phases if phase.steps <= steps][-1]

    def renderer(self, assets: Path, drones: int = ENVIRONMENTS) -> dict:
        """The record of the setting's renderer of `drones` cameras over `assets`."""
        return {"class": "dew.rl.hyperlux:HyperLuxRenderer",
                "fields": {"count": drones, "assets": str(assets), "size": self.pixels, **self._lens}}

    def judge_renderer(self, assets: Path) -> HyperLuxRenderer:
        """A renderer of the setting's cameras over `assets`, which every race judged on shares in turn."""
        return HyperLuxRenderer(ENVIRONMENTS, assets, size=self.pixels, **self._lens)

    @property
    def _lens(self) -> dict:
        return {} if self.fisheye is None else {"fisheye": self.fisheye[0], "source": self.fisheye[1]}


BASE = Setting(Sky(), pixels=64, fisheye=None, replay=2_000_000, curriculum=False,
               judging="untrained beside each point; Table III evaluation and held-out training episodes, "
                       "fixed seeds")
"""The base check: the circle seen through a 64-pixel pinhole, nothing randomized and no latency."""
REALISM = Setting(Sky(tracks=("circle", "inverted_loop", "ladder"), fov=120.0, latency=0.044, frames=True),
                  pixels=96,
                  fisheye=(120.0, 160), replay=1_000_000, curriculum=True,
                  judging="untrained at each level; each track's Table III evaluation and held-out training "
                          "episodes at the level trained at, fixed seeds", phases=PHASES)
"""Toward a real FPV quadrotor's flight (the module docstring)."""
SENSED = ("rates", "motors", "plan")
"""What the encoder reads beside the image, each entry's width the race's."""
DECODED = (("state", 256), ("rates", 256), ("motors", 256))
"""What the informed world model decodes, in the image's place (SkyDreamer Sec II-B: the information variable
leaves the image out, and the flight plan is no observation of the drone): the privileged state and the
sensed rates and rotor speeds, each two layers of its units then its entry's width."""


def run(out: Path, assets: Path, drones: int = ENVIRONMENTS, setting: Setting = BASE,
        phase: Phase = WARMUP) -> EnvRunConfig:
    """The recipe's run under OUT (the module docstring) over `drones` drones seen through HyperLux's cameras
    over `assets`, racing the `setting`'s race in its `phase`."""
    # The race's fields set otherwise than Sky's defaults, as a record holds them (a tuple as a list).
    sky = {f.name: list(value) if isinstance(value, tuple) else value for f in dataclasses.fields(Sky)
           if f.name != "quadrotor" and (value := getattr(setting.sky, f.name)) != f.default}
    race = {"renderer": setting.renderer(assets, drones),
            **({"sky": {"class": "dew.rl.skydreamer:Sky", "fields": sky}} if sky else {})}
    return EnvRunConfig(
        env=EnvConfig("skydreamer", race),
        model=ModelConfig("dew.objectives.rl.dreamer:WorldModel", {
            "reads": "image", "proprio": list(SENSED),
            "auxiliary": [list(head) for head in DECODED], "deter": 2048, "hidden": 256, "stoch": 32,
            "classes": 16, "blocks": 8, "units": 256, "conv_depth": 16, "dtype": "bfloat16"}),
        objective=ObjectiveConfig("dew.objectives.rl.dreamer:DreamerV3Objective",
                                  {"slow_targets": True, "smoothness": 0.002, "scales": {"decoder": 0.0},
                                   "entropy": phase.entropy}),
        # The prefill holds one window, of the phase's length.
        rollout=RolloutConfig("dreamer", {"batch": WINDOWS, "length": phase.length,
                                          "train_ratio": TRAIN_RATIO, "capacity": setting.replay // drones,
                                          "prefill": phase.length}),
        optim=OptimConfig(optimizer="laprop", learning_rate=Linear(peak_value=phase.learning_rate,
                                                                   warmup_steps=1000,
                                                                   end_value=phase.learning_rate)),
        trainer=TrainerConfig(checkpoint_dir=str(out), log_every=500, eval_every=None, checkpoint_every=None,
                              keep=Keep(latest=2)))


def scored(simulator, objective, variables, key: jax.Array) -> dict:
    """Each drone's first episode flying `objective`'s greedy policy from fresh episodes of `simulator`
    for one episode's steps (`dew.rl.evaluation.flights`): its return, gates passed, whether it ended (a
    crash: the race ends no episode otherwise but by its cut) and its steps, averaged over the drones."""
    from dew.rl.env import view
    from dew.rl.evaluation import first_episode_rows, flights
    from dew.rl.simulator import DrivenSimulator

    driven = DrivenSimulator(simulator)
    flown = flights(objective.greedy, driven, simulator.sky.steps, memory=objective.memory)
    following = flown(variables, key)
    rows = jax.device_get(first_episode_rows(following, {"passed": view(following.observation, "passed")}))
    gates = np.asarray(rows["passed"])
    return {"return": float(rows["return"].mean()), "gates": float(gates.mean()),
            "gates_max": int(gates.max()), "crashed": float(rows["terminated"].mean()),
            "steps": float(rows["steps"].mean())}


def chunk_end(done: int, taken: int, steps: int, every: int, phase: Phase, *, pending: bool) -> int | None:
    """The update a run at `done` updates and `taken` environment steps trains its next chunk to: a chunk
    checkpointed and stopped before its score (`pending`) is restored where it stands and scored first;
    short of `steps`, the update that takes the steps to the next judgement (`every` steps apart) at the
    phase's pace; None once the run has taken `steps`, so a finished run run again trains nothing."""
    if pending:
        return done
    if taken >= steps:
        return None
    judgement = min(steps, (taken // every + 1) * every)
    return done + max(1, math.ceil((judgement - taken) / (WINDOWS * phase.length / TRAIN_RATIO)))


def judged_races(setting: Setting, assets: Path) -> dict[str, dict[str, SkyDreamerRace]]:
    """The races a run of `setting` is judged on, each track's (or the race's) Table III evaluation (every
    start before the first gate, the gates' size 1 m) and held-out training episodes, sharing one renderer
    over `assets`."""
    renderer = setting.judge_renderer(assets)
    races = {}
    for track in setting.sky.tracks if setting.curriculum else (None,):
        sky = setting.sky if track is None else dataclasses.replace(setting.sky, tracks=(track,))
        races[track or "race"] = {
            "evaluation": SkyDreamerRace(renderer, dataclasses.replace(sky, anywhere=0.0, size=1.0)),
            "held_out": SkyDreamerRace(renderer, sky)}
    return races


def trained(out: Path, assets: Path, steps: int, every: int, setting: Setting = BASE) -> dict:
    """The recipe trained to `steps` environment steps in chunks of about `every`, going on from its
    checkpoints (its replay made again), each chunk's agent judged greedy (`judged`) into OUT/base.json beside
    the untrained policy judged alike: on the base check's race, on Table III's evaluation and held-out
    training episodes; with a curriculum, on each track alone at the level it trained at, each track's level
    then raised a tenth, to 1, where its evaluation flights crashed under a third of the time, and the
    untrained judged at each level a track reaches. Each chunk trains in the setting's phase for the steps
    taken before it. A run going on goes on at the levels and in the phase its report reached."""
    from dew.objectives.rl.dreamer import DreamerRollout, DreamerV3Objective

    path = out / "base.json"
    report: dict = (json.loads(path.read_text()) if path.exists()
                    else {"points": [], "judging": setting.judging})
    if report.get("judging") != setting.judging:
        raise ValueError(f"{path} was judged otherwise than this run judges ({setting.judging}); go on in a "
                         "fresh OUT")
    done = 0
    if (out / "base").is_dir():
        with Checkpoints(str(out / "base")) as checkpoints:
            done = checkpoints.latest or 0
    points = report["points"]
    taken = points[-1]["environment_steps"] if points else 0
    races = judged_races(setting, assets)
    seeds = {"evaluation": EVALUATION_SEED, "held_out": HELD_OUT_SEED}
    # Each judged race's level, the training race's in `sky.tracks`' order (the base check's one race's).
    levels: dict[str, float] = report.setdefault("levels", dict.fromkeys(races, 0.0))

    def phased(phase: Phase) -> tuple:
        """The run prepared in `phase`, its race at the levels reached."""
        prepared = run(out, assets, setting=setting, phase=phase).prepare()
        rollout, agent = prepared.rollout, prepared.objective
        assert isinstance(rollout, DreamerRollout) and isinstance(agent, DreamerV3Objective)
        assert isinstance(rollout.env, SkyDreamerRace)
        rollout.env.ramp = tuple(levels.values())
        return prepared, rollout, agent

    phase = setting.phase(taken)
    prepared, rollout, agent = phased(phase)

    def judged(variables, names) -> dict:
        """Each named race judged at its level: the flights of `variables`' greedy policy, from fixed
        seeds."""
        scores = {}
        for name in names:
            for split, race in races[name].items():
                race.ramp = levels[name]
                key = jax.random.key(seeds[split])
                scores.setdefault(name, {})[split] = scored(race, agent, variables, key)
        return scores

    def baseline() -> None:
        """The untrained policy judged at each race's level, once a level (once without a curriculum)."""
        if not setting.curriculum:
            if "untrained" not in report:
                report["untrained"] = judged(agent.init(jax.random.key(0)), races)["race"]
            return
        for name, level in levels.items():
            at = report.setdefault("untrained", {}).setdefault(str(level), {})
            if name not in at:
                at.update(judged(agent.init(jax.random.key(0)), [name]))
                print(json.dumps({"untrained": level, name: at[name]}), flush=True)

    def written() -> None:
        # Written aside and moved over, so a stop mid-write leaves the last report whole.
        aside = path.with_suffix(".json.part")
        aside.write_text(json.dumps(report, indent=1))
        aside.replace(path)

    baseline()
    written()
    began, state = time.perf_counter(), None
    scored_at = {point["updates"] for point in points}
    while True:
        reached = chunk_end(done, taken, steps, every, phase,
                            pending=state is None and done > 0 and done not in scored_at)
        if reached is None:
            break
        prepared = dataclasses.replace(prepared, run=prepared.run.override({"trainer.steps": reached}))
        state = prepared.train("base")
        done, taken = int(state.step), rollout.environment_steps
        if done in scored_at:
            continue
        point = {"updates": done, "environment_steps": taken, "levels": dict(levels), "length": phase.length,
                 "seconds": time.perf_counter() - began, "episode_return": rollout.metrics["episode_return"],
                 **judged(state.variables, races)}
        if not setting.curriculum:
            point.update(point.pop("race"))
        points.append(point)
        scored_at.add(done)
        print(json.dumps(point), flush=True)
        if setting.curriculum:
            for name in races:
                if point[name]["evaluation"]["crashed"] < 1 / 3 and levels[name] < 1.0:
                    levels[name] = round(min(1.0, levels[name] + 0.1), 1)
            rollout.env.ramp = tuple(levels.values())
            baseline()
        written()
        if setting.phase(taken) != phase:
            # The next phase's run, its rollout going on from the checkpoint at its own pace. Its replay is
            # made at its first update, once the last phase's is let go: the GPU holds one.
            phase = setting.phase(taken)
            prepared, rollout, agent = phased(phase)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1_000_000, help="environment steps to train to")
    parser.add_argument("--every", type=int, default=100_000, help="environment steps between judgements")
    parser.add_argument("--realism", action="store_true", help="train toward a real FPV quadrotor's flight")
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    setting = REALISM if arguments.realism else BASE
    print(json.dumps(trained(arguments.out, arguments.assets, arguments.steps, arguments.every, setting)))


if __name__ == "__main__":
    main()
