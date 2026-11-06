"""The FPV gate race's training phases (`dew.rl.flight` in JAX; images by HyperLux, `dew.rl.hyperlux`), each
an environment run (`dew.config.env.EnvRunConfig`) under OUT, which the phase run again goes on
from, beside a JSON report.

    python tools/fpv_race.py check OUT --assets A        The renderer on this GPU: frames of the race, and
                                                          frames a second at 16, 256 and 1024 drones, and
                                                          the race's steps a second without images.
    python tools/fpv_race.py teacher OUT [--seconds S]    The privileged teacher, OUT/teacher: PPO over the
              [--warm W]                                  teacher's view, 4096 drones in JAX, for S seconds;
                                                          its greedy episodes scored as it trains, each
                                                          score's step kept, the best named (teacher.json).
                                                          Its curriculum's level, 0.2 at first, rises by 0.1
                                                          to 1 at each score whose crashes fall under a third
                                                          (Song et al. 2021 raised theirs on the crash rate):
                                                          the courses' difficulty, the drones' randomization
                                                          and, from 0.3, chained tracks beside loops. From W,
                                                          a teacher of the first race's view (`warmed`).
    python tools/fpv_race.py wrap OUT --teacher W         A teacher's weights (an earlier teacher.msgpack) as
                                                          the teacher phase's run of no updates, OUT/teacher,
                                                          its record the phase's own at --level (1).
    python tools/fpv_race.py world OUT --teacher T        The world model's stages from the teacher's run T
              --assets A [--teacher-step N]               (its step N, its latest unless named), each started
              [--no-highlight]                            from the one before, its replay begun with the
                                                          teacher's demonstrations: pretrain, DreamerV3
                                                          pretrained on them with behaviour cloning; dagger,
                                                          the agent flying and the teacher labelling its steps
                                                          (DAgger); online, at FlyDreamer's train ratio. The
                                                          teacher and each stage's agent scored alike.

`--hz` sets the control rate (60 by default). The world model's windows and imagination span the same
seconds at any rate: 64 steps and 15 imagined at 30 Hz, 128 and 30 at 60. `--assets` is the folder
`tools/fpv_assets.py` writes.

An agent is scored on its drones' first episodes, each run to its end or to `Racing.max_steps`, on courses
of the families it trains on and, held out, on the published layouts and figure-eights. The score records
the return, the gates passed, the share of episodes that ended in a crash, the share that finished an open
track, and the steps flown. The teacher reads its privileged view. The world model and its agent read the
camera's image alone; the state, the next gates' corners, the drone's airframe and physics and the nearest
pillars are targets for its auxiliary heads. HyperLux is the `hyperlux` extra and needs an NVIDIA GPU with
OptiX.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization, traverse_util

from dew.checkpoints import Checkpoints, Keep
from dew.config import ObjectiveConfig, TrainerConfig
from dew.config.env import EnvConfig, EnvRunConfig, RolloutConfig, TeacherConfig
from dew.optim import Linear, OptimConfig
from dew.registry import ModelConfig

AUXILIARY = (("state", 512), ("gates", 512), ("airframe", 256), ("physics", 256), ("obstacles", 256))
"""The world model's auxiliary heads: each privileged view and its head's units, its width the race's."""
CONTROL, DREAMER = "dew.objectives.rl.control", "dew.objectives.rl.dreamer"
"""The modules the teacher's and the agent's networks and objectives are named from."""
TRAINING = (("loop", 0.4), ("chain", 0.6))
"""The course families a teacher and an agent train on."""
HELD_OUT = (("layout", 0.5), ("figure8", 0.5))
"""The course families held out of training, for scoring alone."""


VIEWS = ("image", "state", "gates", "airframe", "physics", "obstacles", "passed", "finishing", "teacher")
"""What the world phase's races show: the camera's image, which the agent reads, and the privileged views."""
TEACHER_RUN, STAGES = "teacher", ("pretrain", "dagger", "online")
"""The runs under OUT: the teacher's, and the world phase's stages, each started from the one before."""
SCORED_EVERY = 10
"""The teacher's collections between its scores."""


def families_at(level: float):
    """The course families a teacher trains on at the curriculum's `level`: loops, and chained tracks beside
    them from 0.3."""
    return TRAINING if level >= 0.3 else (("loop", 1.0),)


def racing_record(hz: float, families, *, highlight: bool = True) -> dict:
    """The race at `hz` over `families` (`racing_at`) as an environment run's record holds it."""
    return {"hz": hz, "families": [[name, weight] for name, weight in families], "highlight": highlight}


def racing_at(hz: float, families=TRAINING, *, highlight: bool = True):
    """The race at `hz` over `families`."""
    from dew.rl.flight import Racing

    return Racing(hz=hz, families=families, highlight=highlight)


def race(drones: int, views: tuple[str, ...], *, hz: float = 60.0, families=TRAINING,
         assets: Path | None = None, highlight: bool = True):
    """A gate race of `drones` drones at `hz` over `families`, its images drawn by HyperLux over `assets`
    where it shows them."""
    from dew.rl.flight import GateRace

    renderer = None
    if "image" in views:
        from dew.rl.hyperlux import HyperLuxRenderer

        if assets is None:
            raise ValueError("a race that shows images draws them over assets (tools/fpv_assets.py)")
        renderer = HyperLuxRenderer(drones, assets)
    return GateRace(renderer, racing_at(hz, families, highlight=highlight), views=views, count=drones)


def teacher_run(out: Path, *, hz: float, level: float, drones: int = 4096) -> EnvRunConfig:
    """The teacher phase's run at the curriculum's `level` (`teacher`): PPO over the teacher's view
    (`dew.rl.flight.TEACHER` wide) of `drones` drones in JAX (`FlightEnv`), on the courses of its level
    (`families_at`) at its difficulty and randomization, its policy's spread a fifth of the actor-critic's, so
    the first rollouts fly rather than tumble; at `hz` control steps a second, its discount and its
    advantages' decay those of 0.99 and 0.95 a step at 60 Hz, so the teacher looks as many seconds ahead at
    any rate. Each step it scores at (`SCORED_EVERY` collections) is kept."""
    flight = {"racing": racing_record(hz, families_at(level)), "views": ["teacher"], "difficulty": level,
              "ramp": level}
    rollout = {"unroll_length": round(32 * hz / 30), "epochs": 4, "minibatches": 32}
    every = SCORED_EVERY * rollout["epochs"] * rollout["minibatches"]
    return EnvRunConfig(
        env=EnvConfig("flight", flight, count=drones),
        model=ModelConfig(f"{CONTROL}:ActorCritic",
                          {"policy_layers": [256] * 3, "value_layers": [256] * 4, "spread": 0.2}),
        objective=ObjectiveConfig(f"{CONTROL}:ControlPPOObjective", {
            "discount": 0.99 ** (60 / hz), "lam": 0.95 ** (60 / hz), "clip": 0.2, "entropy_cost": 1e-4,
            "reads": ["teacher"]}),
        rollout=RolloutConfig("control", rollout), optim=OptimConfig(optimizer="adam", learning_rate=3e-4,
                                                                     clip_grads=1.0),
        trainer=TrainerConfig(checkpoint_dir=str(out), steps=every, log_every=every, eval_every=None,
                              checkpoint_every=None, keep=Keep(latest=1, every=every)))


def warmed(path: Path, objective):
    """A teacher's weights for `objective`, the race's teacher, from a teacher of the race's view or of its
    first race's (`dew.rl.flight.INTERFACE` entries) written whole (`flax.serialization.to_bytes`): every
    weight its own, each first layer's weights from the view's new entries zero, so it acts as it did."""
    new = jax.device_get(objective.init(jax.random.key(0)))
    previous = traverse_util.flatten_dict(serialization.msgpack_restore(path.read_bytes()))
    fresh = traverse_util.flatten_dict(serialization.to_state_dict(new))
    for name, value in fresh.items():
        held = np.asarray(previous[name])
        if held.shape == value.shape:
            fresh[name] = held
        else:
            grown = np.zeros(value.shape, held.dtype)
            grown[:held.shape[0]] = held
            fresh[name] = grown
    return serialization.from_state_dict(new, traverse_util.unflatten_dict(fresh))


def _latest(directory: Path) -> int:
    """The last step the run in `directory` checkpointed, 0 where it has none."""
    if not directory.is_dir():
        return 0
    with Checkpoints(str(directory)) as checkpoints:
        return checkpoints.latest or 0


def _written(path: Path, text: str) -> None:
    """`text` at `path`, written whole beside it before it replaces what was there, so a run stopped at any
    point leaves either file whole."""
    path.with_name(path.name + ".part").write_text(text)
    path.with_name(path.name + ".part").replace(path)


def scored(simulator, policy, params, memory, key: jax.Array, steps: int, *,
           film: Path | None = None) -> dict:
    """Each drone's first episode under `policy` (reading the driven step whole) over `steps` steps from
    fresh episodes of `simulator`, an evaluation race of its own (`dew.rl.evaluation.flights`,
    `_first_episodes`); with `film`, the first four drones' camera images after each step written there
    `[steps, 4, H, W, 3]`. `memory` is the policy's starting state over the race's drones."""
    from dew.rl.env import view
    from dew.rl.evaluation import flights
    from dew.rl.simulator import DrivenSimulator

    following = flights(policy, DrivenSimulator(simulator), steps, memory=lambda count: memory)(params, key)
    if film is not None:
        np.savez_compressed(film, frames=np.asarray(view(following.observation, "image"))[:, :4])
    return _first_episodes(following, view(following.observation, "passed"), simulator.racing.crash_penalty)


def _first_episodes(following, passed, crash_penalty: float) -> dict:
    """Each drone's first episode from the steps that followed its actions and the gates passed so far
    `[steps, drones]` (`dew.rl.evaluation.first_episode_rows`): its return, gates passed, whether it crashed
    or finished an open track (an end whose step paid less than half the crash penalty is a crash) and its
    steps, averaged over the drones."""
    from dew.rl.evaluation import first_episode_rows

    rows = jax.device_get(first_episode_rows(following, {"passed": passed}))
    gates = np.asarray(rows["passed"])
    crashed = rows["terminated"] & (rows["last_reward"] < -crash_penalty / 2)
    return {"return": float(rows["return"].mean()), "gates": float(gates.mean()),
            "gates_max": int(gates.max()), "crashed": float(crashed.mean()),
            "finished": float((rows["terminated"] & ~crashed).mean()), "steps": float(rows["steps"].mean())}


def flown(env, objective, params, key: jax.Array, steps: int, crash_penalty: float) -> dict:
    """`scored` of a teacher's greedy flight in a batched JAX race (`dew.rl.flight.FlightEnv`, its views the
    teacher's, `passed` and `gates`), the whole flight one jitted rollout (`dew.rl.env.env_rollout`); and
    the share of its first episodes' steps whose next gate lies in the camera's view (`in_view`), which a
    policy of the camera alone needs of the flights it learns from."""
    from dew.rl.env import env_rollout, start
    from dew.rl.evaluation import first_episodes
    from dew.rl.flight import mounted

    def rolled(params, key):
        carry = start(env, key)
        camera = carry.state.camera
        taken = env_rollout(env, objective.greedy, params, carry, steps)[1]
        return taken, jax.vmap(mounted)(camera), camera.fov

    taken, mounts, fov = jax.device_get(jax.jit(rolled)(params, key))
    following = taken.following
    alive = np.asarray(first_episodes(following))
    seen = in_view(taken.observation["gates"], mounts, fov)
    return {**_first_episodes(following, following.observation["passed"], crash_penalty),
            "in_view": float(seen[alive].mean())}


def in_view(gates: np.ndarray, mounts: np.ndarray, fov: np.ndarray) -> np.ndarray:
    """Whether the next gate's centre lies in each drone's camera's field of view `[steps, drones]`, from the
    `gates` view (its corners in the body frame), each drone's camera mount `[drones, 3, 3]`
    (`dew.rl.flight.mounted`) and field of view `[drones]` (radians, a square image's): in front of the
    camera, its angles across and up from the camera's axis within half the field of view. A drone's mount
    holds through its first episode, the steps this scores."""
    center = np.asarray(gates, np.float64)[..., :12].reshape(*np.shape(gates)[:-1], 4, 3).mean(-2)
    local = np.einsum("dji,sdj->sdi", np.asarray(mounts, np.float64), center)
    depth, half = local[..., 0], np.asarray(fov, np.float64) / 2
    across = np.arctan2(np.abs(local[..., 1]), np.maximum(depth, 1e-9))
    rise = np.arctan2(np.abs(local[..., 2]), np.maximum(depth, 1e-9))
    return (depth > 0) & (across < half) & (rise < half)


def teacher_policy(objective):
    """The teacher's greedy action from a driven step, its environment's observation."""
    def policy(params, observed, key, memory):
        action, _, memory = objective.greedy(params, observed["observation"], key, memory)
        return action, {}, memory

    return policy


def check(out: Path, *, assets: Path, hz: float = 60.0) -> dict:
    """The race's first frames, and frames a second with images and without."""
    report = {"device": str(jax.devices()[0])}
    sample = race(16, ("image", "passed"), hz=hz, assets=assets)
    frames = [np.asarray(sample.reset(0)["image"])]
    for _ in range(30):
        stepped = sample.step(jnp.zeros((16, 4)))
    frames.append(np.asarray(stepped.observation["image"]))
    frames = np.concatenate(frames)
    np.save(out / "frames.npy", frames)
    report["frames"] = {"shape": list(frames.shape), "mean": float(frames.mean()), "std": float(frames.std())}
    del sample
    for drones, views in ((16, ("image",)), (256, ("image",)), (1024, ("image",)), (4096, ("teacher",))):
        timed = race(drones, views, hz=hz, assets=assets)
        timed.reset(0)
        actions = jnp.zeros((drones, 4))
        for _ in range(5):
            stepped = timed.step(actions)
        jax.block_until_ready(stepped.reward)
        began = time.perf_counter()
        for _ in range(50):
            stepped = timed.step(actions)
        jax.block_until_ready(stepped.reward)
        name = f"{drones}{'' if 'image' in views else '-without-images'}_steps_per_second"
        report[name] = 50 * drones / (time.perf_counter() - began)
        del timed
    (out / "check.json").write_text(json.dumps(report, indent=1))
    return report


def teacher(out: Path, *, seconds: float = 6600.0, drones: int = 4096, hz: float = 60.0,
            warm: Path | None = None, level: float = 0.2) -> dict:
    """The teacher phase's run (`teacher_run`, OUT/teacher) trained for `seconds` over `drones` drones in JAX,
    going on from where its checkpoints and teacher.json hold it, scored every `SCORED_EVERY` collections on
    1024 drones of the training families and of the held-out ones, the step of the best greedy return at the
    highest level reached named (teacher.json's `best`); its curriculum's level starts at `level` and rises by
    0.1, to 1, at each score whose crashes fall under a third, each chunk between scores the run at its level.
    A fresh run starts from `warm`, a teacher of the first race's view (`warmed`), where given."""
    from dew.rl.env import Batched
    from dew.rl.flight import FlightEnv

    def judged_at(level: float, families) -> Batched:
        shown = ("teacher", "passed", "gates")
        return Batched(FlightEnv(racing_at(hz, families), shown, difficulty=level, ramp=level), 1024)

    path = out / "teacher.json"
    report = json.loads(path.read_text()) if path.exists() else {
        "drones": drones, "hz": hz, "warm": None if warm is None else str(warm), "curve": [], "best": None,
        "level": level}
    level, began = report["level"], time.perf_counter()
    horizon, penalty = round(20 * hz), racing_at(hz).crash_penalty
    while time.perf_counter() - began < seconds:
        done = _latest(out / TEACHER_RUN)
        run = teacher_run(out, hz=hz, level=level, drones=drones)
        prepared = run.override({"trainer.steps": done + run.trainer.steps}).prepare()
        if warm is not None and not done:
            prepared.objective.variables = warmed(warm, prepared.objective)
        state = prepared.train(TEACHER_RUN)
        updates, objective = int(state.step), prepared.objective
        epochs, minibatches, length = (prepared.run.rollout.arguments[name]
                                       for name in ("epochs", "minibatches", "unroll_length"))
        point = {"updates": updates, "seconds": round(time.perf_counter() - began, 1),
                 "environment_steps": updates // (epochs * minibatches) * drones * length, "level": level,
                 **flown(judged_at(level, families_at(level)), objective, state.variables, jax.random.key(1),
                         horizon, penalty),
                 "held_out": flown(judged_at(level, HELD_OUT), objective, state.variables, jax.random.key(1),
                                   horizon, penalty)}
        report["curve"].append(point)
        print(json.dumps(point), flush=True)
        best = report["best"]
        if best is None or (level, point["return"]) > (best["level"], best["return"]):
            report["best"] = {"level": level, "return": point["return"], "step": updates}
        if point["crashed"] < 1 / 3 and level < 1.0:
            level = round(min(1.0, level + 0.1), 1)
        report["level"] = level
        _written(path, json.dumps(report, indent=1))
    best = report["best"]
    saved = EnvRunConfig.load(str(out / TEACHER_RUN))
    with Checkpoints(str(out / TEACHER_RUN)) as checkpoints:
        variables = checkpoints.variables(step=best["step"], ema=False)
    final = judged_at(1.0, TRAINING), judged_at(1.0, HELD_OUT)
    objective = saved.agent(final[0], variables)
    names = ("trained", "held_out")
    report["final"] = {name: flown(judged, objective, variables, jax.random.key(2), racing_at(hz).max_steps,
                                   penalty) for name, judged in zip(names, final, strict=True)}
    _written(path, json.dumps(report, indent=1))
    return report


def wrap(out: Path, *, weights: Path, hz: float = 60.0, level: float = 1.0, drones: int = 4096) -> dict:
    """A teacher's weights written whole (an earlier teacher phase's teacher.msgpack, of the race's view or
    its first race's, `warmed`) as the teacher phase's run, OUT/teacher, of no updates: its record the phase's
    own at `level` (`teacher_run`), its env, model and objective those a trained teacher's are, and its step 0
    checkpoint the weights, which `RunConfig.train` writes as a run's last step, so the world phase reads it
    as it reads a trained teacher, and the teacher phase goes on from it. The record cannot name the file the
    weights came from; the report (teacher.json, the teacher phase's, unscored) does."""
    prepared = teacher_run(out, hz=hz, level=level, drones=drones).override({"trainer.steps": 0}).prepare()
    prepared.objective.variables = warmed(weights, prepared.objective)
    state = prepared.train(TEACHER_RUN)
    # The teacher phase's own report, unscored, so the phase goes on from the wrapped run.
    report = {"drones": drones, "hz": hz, "warm": str(weights), "curve": [], "best": None, "level": level,
              "wrapped": {"run": str(out / TEACHER_RUN), "step": int(state.step)}}
    _written(out / "teacher.json", json.dumps(report, indent=1))
    return report


@dataclasses.dataclass(frozen=True)
class World:
    """The world phase's settings (`world`): FlyDreamer's agent at its size (TransDreamer's dynamics,
    bfloat16) on the image alone in races of `drones` drones at `hz`, its images drawn by HyperLux over
    `assets`, its teacher the run `teacher` at `teacher_step` (its latest where None). Without `highlight` the
    next gate is not lit and the courses are loops alone (a comparison). Each stage's replay holds
    `capacity_seconds` a drone (16 GB of 96-pixel images over 16 drones at 60 Hz) and begins with
    `demonstration_seconds` a drone of the teacher's flight, its policy's draws (DART, Laskey et al., arXiv
    1703.09327: a student cloned from greedy flight alone missed the teacher's actions on fresh flights by
    three times its training error and crashed within two seconds). A DAgger stage replays `dagger_ratio`
    steps for each it takes. Each stage writes its checkpoint every `saved_every` updates (about ten minutes
    on a G4)."""

    out: Path
    teacher: Path
    assets: Path
    teacher_step: int | None = None
    drones: int = 16
    hz: float = 60.0
    highlight: bool = True
    demonstration_seconds: float = 600.0
    capacity_seconds: float = 600.0
    dagger_ratio: float = 64.0
    saved_every: int = 2000

    @property
    def length(self) -> int:
        """The windows' steps, 64 at 30 Hz: the same seconds at any rate."""
        return round(64 * self.hz / 30)

    def run(self, stage: str, level: float) -> EnvRunConfig:
        """The stage's run (`STAGES`) at the curriculum's `level`. Every stage's replay begins with the
        teacher's demonstrations. `pretrain` clones them, the world model at FlyDreamer's `pretrain_wm` scales
        and the actor imitating the actions taken, training on them alone (`train_ratio` inf); `dagger`, from
        it, has the agent fly and imitate the teacher's label of each of its steps; `online`, from it, learns
        by imagination at FlyDreamer's train ratio of 128."""
        families = TRAINING if self.highlight else (("loop", 1.0),)
        renderer = {"class": "dew.rl.hyperlux:HyperLuxRenderer",
                    "fields": {"count": self.drones, "assets": str(self.assets)}}
        race = {"renderer": renderer, "racing": racing_record(self.hz, families, highlight=self.highlight),
                "views": list(VIEWS), "difficulty": level, "ramp": level}
        imagination = round(15 * self.hz / 30)
        cloned = {"imagination": imagination, "imitation": 1.0, "scales": {
            "policy": 0.0, "value": 0.0, "replay_value": 0.0, **{name: 1.0 for name, _ in AUXILIARY}}}
        objective = {"pretrain": cloned, "dagger": {**cloned, "imitates": "label"},
                     "online": {"imagination": imagination}}[stage]
        ratio = {"pretrain": math.inf, "dagger": self.dagger_ratio, "online": 128.0}[stage]
        rollout = {"batch": 16, "length": self.length, "train_ratio": ratio,
                   "prefill": round(self.demonstration_seconds * self.hz),
                   "capacity": round(self.capacity_seconds * self.hz)}
        # DreamerV3's rate, 4e-5 after a linear warmup over its first 1000 updates.
        rate = Linear(peak_value=4e-5, warmup_steps=1000, end_value=4e-5)
        run = EnvRunConfig(
            env=EnvConfig("gate_race", race),
            objective=ObjectiveConfig(f"{DREAMER}:DreamerV3Objective", objective),
            rollout=RolloutConfig("dreamer", rollout),
            teacher=TeacherConfig(str(self.teacher), self.teacher_step, demonstrates=True),
            optim=OptimConfig(optimizer="laprop", learning_rate=rate),
            trainer=TrainerConfig(checkpoint_dir=str(self.out), key=STAGES.index(stage), log_every=1000,
                                  eval_every=None, checkpoint_every=self.saved_every, keep=Keep(latest=1)))
        if stage != STAGES[0]:
            return dataclasses.replace(run, pretrained=str(self.out / STAGES[STAGES.index(stage) - 1]))
        dynamics = {"action_size": 4, "stoch": 32, "classes": 32, "width": 512, "layers": 4, "heads": 8,
                    "context": self.length, "dtype": "bfloat16"}
        model = {"conv_depth": 64, "units": 1024, "reads": "image", "dtype": "bfloat16",
                 "auxiliary": [list(head) for head in AUXILIARY], "dynamics": dynamics}
        return dataclasses.replace(run, model=ModelConfig(f"{DREAMER}:WorldModel", model))


def _staged(world: World, stage: str, updates: int, every: int | None, report: dict, judge, races: tuple,
            level: float):
    """The world phase's `stage` (`World.run`) trained to `updates` updates under OUT/<stage>, going on from
    its checkpoints, its replay made again; with `every`, in chunks of that many updates, each chunk's agent
    scored (`judge`) into report[stage] and its curriculum raised from `level` as the teacher's was, up a
    tenth, to 1, at each score whose crashes fall under a third, the stage's race and `races` drawn at it. The
    stage is prepared once and each chunk trains it on, its rollout's environments and replay kept between
    them. A chunk's checkpoint is written as it ends, then its score and the level the next chunk trains at
    (world.json's `level`), where a stopped stage goes on; one stopped between the two is restored at that
    checkpoint and scored before the next chunk trains. Returns the stage's last state, its agent and the
    level it ended at."""
    from dew.objectives.rl.dreamer import DreamerRollout, DreamerV3Objective

    done = _latest(world.out / stage)
    points = [point for point in report.get(stage, []) if point["updates"] <= done]
    prepared = world.run(stage, level).prepare()
    rollout, agent = prepared.rollout, prepared.objective
    assert isinstance(rollout, DreamerRollout) and isinstance(agent, DreamerV3Objective)
    began, before = time.perf_counter(), points[-1]["seconds"] if points else 0.0
    state = None
    while state is None or done < updates:
        # A chunk checkpointed and stopped before its score is restored there and scored before the next.
        pending = (state is None and every is not None and done > 0 and done % every == 0
                   and all(point["updates"] != done for point in points))
        reached = done if pending else updates if every is None else min(updates, (done // every + 1) * every)
        prepared = dataclasses.replace(prepared, run=prepared.run.override(
            {"trainer.steps": reached, "env.difficulty": level, "env.ramp": level}))
        state = prepared.train(stage)
        done = int(state.step)
        if every is None or (points and points[-1]["updates"] == done):
            continue
        point = {"updates": done, "environment_steps": rollout.environment_steps, "level": level,
                 "seconds": before + time.perf_counter() - began,
                 "episode_return": rollout.metrics["episode_return"],
                 **judge(agent.greedy, state.variables, agent.memory(world.drones), jax.random.key(done))}
        points.append(point)
        report[stage] = points
        print(json.dumps(point), flush=True)
        if point["crashed"] < 1 / 3 and level < 1.0:
            level = round(min(1.0, level + 0.1), 1)
            for raced in (rollout.env, *races):
                raced.difficulty = raced.ramp = level
        # The level the next chunk trains at, beside the score that earned it: a resumed stage goes on there.
        report["level"] = level
        _written(world.out / "world.json", json.dumps(report, indent=1))
    return state, agent, level


def world(world: World, *, pretrain_updates: int = 50_000, dagger_updates: int = 10_000,
          environment_seconds: float = 33_333.0, level: float = 1.0) -> dict:
    """The world phase's stages (`World.run`) in turn, each going on from its checkpoints: pretraining for
    `pretrain_updates` updates, DAgger for `dagger_updates`, scored every 2,000, and online training for
    `environment_seconds` of flight over all its drones (FlyDreamer's 1M steps at 30 Hz), scored every 10,000;
    the teacher and each stage's last agent scored alike, each flying greedy (world.json), the cloned, DAgger
    and online agents' first drones filmed (cloned.npz, dagger.npz, video.npz), each on an evaluation race
    of its own, on the training families and, held out, the others. Every race is at the curriculum's `level`
    (its courses' difficulty and its drones' randomization), which the scored stages raise as the teacher's
    curriculum did (`teacher`) toward 1."""
    families = TRAINING if world.highlight else (("loop", 1.0),)
    path = world.out / "world.json"
    report = json.loads(path.read_text()) if path.exists() else {
        "drones": world.drones, "hz": world.hz, "length": world.length, "highlight": world.highlight,
        "level": level, "teacher_run": str(world.teacher)}
    # A phase going on goes on at the level its last committed chunk reached.
    level = report["level"]

    def raced(families):
        raced = race(world.drones, VIEWS, hz=world.hz, families=families, assets=world.assets,
                     highlight=world.highlight)
        raced.difficulty = raced.ramp = level
        return raced

    judged, held_out = raced(families), raced(HELD_OUT)
    horizon = judged.racing.max_steps

    def judge(policy, params, memory, key, film=None):
        return {**scored(judged, policy, params, memory, key, horizon, film=film),
                "held_out": scored(held_out, policy, params, memory, key, horizon)}

    if "teacher" not in report:
        saved = EnvRunConfig.load(str(world.teacher))
        with Checkpoints(str(world.teacher)) as checkpoints:
            variables = checkpoints.variables(step=world.teacher_step, ema=False)
        report["teacher"] = judge(teacher_policy(saved.agent(judged, variables)), variables, None,
                                  jax.random.key(4))
        _written(path, json.dumps(report, indent=1))
    collecting = world.run("online", level).rollout.arguments
    online = int(round(environment_seconds * world.hz) * collecting["train_ratio"]
                 / (collecting["batch"] * collecting["length"]))
    filmed = {"pretrain": "cloned.npz", "dagger": "dagger.npz", "online": "video.npz"}
    for stage, updates, every in zip(STAGES, (pretrain_updates, dagger_updates, online), (None, 2000, 10_000),
                                     strict=True):
        state, agent, level = _staged(world, stage, updates, every, report, judge, (judged, held_out), level)
        report["level"] = level
        report[f"{stage}_agent"] = judge(agent.greedy, state.variables, agent.memory(world.drones),
                                         jax.random.key(7 + STAGES.index(stage)),
                                         film=world.out / filmed[stage])
        print(json.dumps({key: value for key, value in report.items() if not isinstance(value, list)}),
              flush=True)
        _written(path, json.dumps(report, indent=1))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("check", "teacher", "wrap", "world"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument("--seconds", type=float, default=6600.0)
    parser.add_argument("--teacher", type=Path, help="wrap: the teacher's weights (teacher.msgpack); world: "
                        "the teacher's run (OUT/teacher)")
    parser.add_argument("--teacher-step", type=int)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--warm", type=Path)
    parser.add_argument("--level", type=float, help="the curriculum's level: the teacher's first (0.2), a "
                        "wrapped teacher's and the world phase's every race's (1.0)")
    parser.add_argument("--no-highlight", action="store_true")
    parser.add_argument("--pretrain-updates", type=int, default=50_000)
    parser.add_argument("--dagger-updates", type=int, default=10_000)
    parser.add_argument("--dagger-ratio", type=float, default=64.0)
    parser.add_argument("--environment-seconds", type=float, default=33_333.0)
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    if arguments.phase == "check":
        report = check(arguments.out, assets=arguments.assets, hz=arguments.hz)
    elif arguments.phase == "teacher":
        report = teacher(arguments.out, seconds=arguments.seconds, hz=arguments.hz, warm=arguments.warm,
                         level=0.2 if arguments.level is None else arguments.level)
    elif arguments.phase == "wrap":
        report = wrap(arguments.out, weights=arguments.teacher, hz=arguments.hz,
                      level=1.0 if arguments.level is None else arguments.level)
    else:
        report = world(World(arguments.out, arguments.teacher, arguments.assets, arguments.teacher_step,
                             hz=arguments.hz, highlight=not arguments.no_highlight,
                             dagger_ratio=arguments.dagger_ratio),
                       pretrain_updates=arguments.pretrain_updates, dagger_updates=arguments.dagger_updates,
                       environment_seconds=arguments.environment_seconds,
                       level=1.0 if arguments.level is None else arguments.level)
    print(json.dumps({key: value for key, value in report.items() if not isinstance(value, list)}))


if __name__ == "__main__":
    main()
