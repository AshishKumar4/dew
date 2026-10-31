"""The FPV gate race's training phases (`dew.rl.flight` in JAX; images by HyperLux, `dew.rl.hyperlux`). Each
phase writes a JSON report and its weights to OUT.

    python tools/fpv_race.py check OUT --assets A        The renderer on this GPU: frames of the race, and
                                                          frames a second at 16, 256 and 1024 drones, and
                                                          the race's steps a second without images.
    python tools/fpv_race.py teacher OUT [--seconds S]    The privileged teacher: PPO over the teacher's
              [--warm W]                                  view, 4096 drones in JAX, for S seconds; its greedy
                                                          episodes scored as it trains, the best kept. Its
                                                          curriculum's level, 0.2 at first, rises by 0.1 to 1
                                                          at each score whose crashes fall under a third (Song
                                                          et al. 2021 raised theirs on the crash rate): the
                                                          courses' difficulty, the drones' randomization and,
                                                          from 0.3, chained tracks beside loops. From W, a
                                                          teacher of the first race's view (`warmed`).
    python tools/fpv_race.py world OUT --teacher W        The world model's phases: the teacher's
              --assets A [--no-highlight]                 demonstrations, DreamerV3 pretrained on them with
                                                          behaviour cloning, then trained online at
                                                          FlyDreamer's train ratio; the teacher, the cloned
                                                          agent and the online agent scored alike.

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
import itertools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization, traverse_util

from dew.data.dataset import Loader

AUXILIARY = (("state", 13, 512), ("gates", 24, 512), ("airframe", 6, 256), ("dynamics", 20, 256),
             ("obstacles", 12, 256))
"""The world model's auxiliary heads: each privileged view, its width and its head's units."""
TRAINING = (("loop", 0.4), ("chain", 0.6))
"""The course families a teacher and an agent train on."""
HELD_OUT = (("layout", 0.5), ("figure8", 0.5))
"""The course families held out of training, for scoring alone."""


def _idle() -> Loader:
    """A loader whose batches a rollout replaces whole."""
    return Loader(train=lambda partition: itertools.repeat({"step": np.zeros((8,), np.float32)}), val=None,
                  records=8, batch=8)


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


def teacher_objective(observation_size: int | None = None):
    """PPO over the teacher's view (`dew.rl.flight.TEACHER` wide, or `observation_size`), its policy's spread
    a fifth of the actor-critic's, so the first rollouts fly rather than tumble."""
    from dew.objectives.rl.control import ActorCritic, ControlPPOObjective
    from dew.rl.flight import TEACHER

    networks = ActorCritic(4, (256, 256, 256), (256, 256, 256, 256), spread=0.2)
    return ControlPPOObjective(networks, observation_size or TEACHER, discount=0.99, lam=0.95, clip=0.2,
                               entropy_cost=1e-4, reads=("teacher",))


def warmed(path: Path):
    """A teacher's weights for the race's view from a teacher of its first race's (`dew.rl.flight.INTERFACE`
    entries): every weight its own, each first layer's weights from the view's new entries zero, so it acts
    as it did."""
    from dew.rl.flight import INTERFACE

    old = serialization.from_bytes(teacher_objective(INTERFACE).init(jax.random.key(0)), path.read_bytes())
    new = jax.device_get(teacher_objective().init(jax.random.key(0)))
    previous = traverse_util.flatten_dict(serialization.to_state_dict(old))
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


def scored(simulator, policy, params, memory, key: jax.Array, steps: int, *,
           film: Path | None = None) -> dict:
    """Each drone's first episode under `policy` (reading the driven step whole) over `steps` steps from
    fresh episodes of `simulator`, an evaluation race of its own: its return, gates passed, whether it
    crashed or finished an open track (an end whose step paid less than half the crash penalty is a crash)
    and its steps, averaged over the drones, each read from the transitions the steps reach up to the
    episode's last; with `film`, the first four drones' camera images after each step written there
    `[steps, 4, H, W, 3]`."""
    from dew.rl.simulator import DrivenSimulator, simulator_rollout, simulator_start

    driven = DrivenSimulator(simulator)
    _, taken = simulator_rollout(driven, policy, params, simulator_start(driven, key, memory), steps,
                                 reset=False)
    observed = jax.device_get(taken.following.observation)
    if film is not None:
        np.savez_compressed(film, frames=observed["observation"]["image"][:, :4])
    ended = np.cumsum(observed["last"], 0) - observed["last"] > 0
    reward = np.where(ended, 0, observed["reward"])
    passed = np.where(ended, 0, observed["observation"]["passed"]).max(0)
    terminal = observed["terminal"] & ~ended
    crashed = terminal & (observed["reward"] < -simulator.racing.crash_penalty / 2)
    return {"return": float(reward.sum(0).mean()), "gates": float(passed.mean()),
            "gates_max": int(passed.max()), "crashed": float(crashed.any(0).mean()),
            "finished": float((terminal & ~crashed).any(0).mean()), "steps": float((~ended).sum(0).mean())}


def teacher_policy(objective):
    """The teacher's greedy action from a driven step."""
    def greedy(params, observed, key, memory):
        return objective.greedy(params, observed["observation"], key, memory)

    return greedy


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
    """PPO's teacher for `seconds` over 4096 drones in JAX (`dew.rl.flight.FlightEnv`), scored every ten
    collections on 1024 drones of the training families and of the held-out ones, the weights of the best
    greedy return at the highest level reached kept (teacher.msgpack); its curriculum's level starts at
    `level` and rises by 0.1, to 1, at each score whose crashes fall under a third: the courses' difficulty
    and the drones' randomization its level, chained tracks beside loops from 0.3. From `warm`, a teacher of
    the first race's view (`warmed`)."""
    from dew.objectives.rl.control import ControlRollout
    from dew.rl.env import Batched
    from dew.rl.flight import FlightEnv
    from dew.training import Trainer

    def env_at(level: float):
        families = TRAINING if level >= 0.3 else (("loop", 1.0),)
        return Batched(FlightEnv(racing_at(hz, families), ("teacher",), difficulty=level, ramp=level), drones)

    def judged_at(level: float, families) -> object:
        judged = race(1024, ("teacher", "passed"), hz=hz, families=families)
        judged.difficulty = judged.ramp = level
        return judged

    objective = teacher_objective()
    if warm is not None:
        objective.variables = warmed(warm)
    rollout = ControlRollout(objective, env_at(level), unroll_length=round(32 * hz / 30), epochs=4,
                             minibatches=32)
    trainer = Trainer(objective, optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4)),
                      key=jax.random.key(0), rollout=rollout)
    state = trainer.place()[0]
    report = {"drones": drones, "hz": hz, "warm": None if warm is None else str(warm), "curve": []}
    best, began, updates = (-1.0, -np.inf), time.perf_counter(), 0
    every = 10 * rollout.epochs * rollout.minibatches
    horizon = round(20 * hz)
    trained, held_out = judged_at(level, TRAINING), judged_at(level, HELD_OUT)
    while time.perf_counter() - began < seconds:
        state = trainer.fit(_idle(), steps=updates + every, log_every=every, state=state)
        updates += every
        policy = teacher_policy(objective)
        point = {"updates": updates, "seconds": round(time.perf_counter() - began, 1),
                 "environment_steps": updates // (rollout.epochs * rollout.minibatches) * drones
                 * rollout.unroll_length, "level": level,
                 **scored(trained, policy, state.variables, None, jax.random.key(1), horizon),
                 "held_out": scored(held_out, policy, state.variables, None, jax.random.key(1), horizon)}
        report["curve"].append(point)
        print(json.dumps(point), flush=True)
        if (level, point["return"]) > best:
            best = level, point["return"]
            (out / "teacher.msgpack").write_bytes(serialization.to_bytes(jax.device_get(state.variables)))
        if point["crashed"] < 1 / 3 and level < 1.0:
            level = round(min(1.0, level + 0.1), 1)
            rollout.env = env_at(level)
            trained, held_out = judged_at(level, TRAINING), judged_at(level, HELD_OUT)
        (out / "teacher.json").write_text(json.dumps(report, indent=1))
    kept = (out / "teacher.msgpack").read_bytes()
    variables = serialization.from_bytes(jax.device_get(state.variables), kept)
    final = judged_at(1.0, TRAINING), judged_at(1.0, HELD_OUT)
    names = ("trained", "held_out")
    report["best"] = {"level": best[0], **{name: scored(judged, teacher_policy(objective), variables, None,
                                                         jax.random.key(2), final[0].racing.max_steps)
                                           for name, judged in zip(names, final, strict=True)}}
    (out / "teacher.json").write_text(json.dumps(report, indent=1))
    return report


def demonstrations(simulator, objective, variables, steps: int, capacity: int):
    """The teacher's greedy flight from every drone of `simulator` for `steps` steps, recorded in the
    agent's convention, as a replay of `capacity` steps a drone the online agent goes on writing."""
    from dew.rl.replay import ReplayBuffer
    from dew.rl.simulator import DrivenSimulator, simulator_rollout, simulator_start

    driven = DrivenSimulator(simulator)
    _, taken = simulator_rollout(driven, teacher_policy(objective), variables,
                                 simulator_start(driven, jax.random.key(3)), steps, reset=False)
    stream = {**taken.observation, "action": taken.action}
    return ReplayBuffer.empty(jax.tree.map(lambda leaf: leaf[0], stream), capacity).add(stream)


def world(out: Path, *, teacher_weights: Path, assets: Path, drones: int = 16, hz: float = 60.0,
          demonstration_seconds: float = 300.0, pretrain_updates: int = 50_000,
          environment_seconds: float = 33_333.0, capacity_seconds: float = 600.0,
          highlight: bool = True) -> dict:
    """FlyDreamer's agent at its size (TransDreamer's dynamics, bfloat16) on the image alone: pretrained on
    the teacher's demonstrations (FlyDreamer's `pretrain_wm` scales and behaviour cloning), then trained
    online at train ratio 128, starting from the demonstrations' replay; the teacher, the cloned agent and
    the online agent scored alike (world.json, pretrained.msgpack, world.msgpack), the online agent every
    10,000 updates, and the cloned and the final agents' first drones filmed (cloned.npz, video.npz), each
    on an evaluation race of its own, on the training families and, held out, the others. Without
    `highlight` the next gate is not lit and the courses are loops alone (a comparison). Each drone flies
    `demonstration_seconds` for the demonstrations; the agent flies `environment_seconds` online over all
    its drones (FlyDreamer's 1M steps at 30 Hz); the replay holds `capacity_seconds` a drone (16 GB of
    96-pixel images over 16 drones at 60 Hz)."""
    from rl_acceptance import laprop

    from dew.objectives.rl.dreamer import DreamerRollout, DreamerV3Objective, WorldModel
    from dew.objectives.rl.tssm import TSSM
    from dew.training import Trainer

    scale = hz / 30
    length, imagination = round(64 * scale), round(15 * scale)
    views = ("image", "state", "gates", "airframe", "dynamics", "obstacles", "passed", "finishing", "teacher")
    families = TRAINING if highlight else (("loop", 1.0),)

    def raced(families):
        return race(drones, views, hz=hz, families=families, assets=assets, highlight=highlight)

    simulator, judged, held_out = raced(families), raced(families), raced(HELD_OUT)
    horizon = simulator.racing.max_steps
    taught = teacher_objective()
    held = serialization.from_bytes(taught.init(jax.random.key(0)), teacher_weights.read_bytes())

    def judge(policy, params, memory, key, film=None):
        return {**scored(judged, policy, params, memory, key, horizon, film=film),
                "held_out": scored(held_out, policy, params, memory, key, horizon)}

    report = {"drones": drones, "hz": hz, "length": length, "imagination": imagination,
              "highlight": highlight, "teacher": judge(teacher_policy(taught), held, None, jax.random.key(4))}
    began = time.perf_counter()
    replay = demonstrations(simulator, taught, held, round(demonstration_seconds * hz),
                            round(capacity_seconds * hz))
    report["demonstration_seconds"] = time.perf_counter() - began
    print(json.dumps(report), flush=True)

    dynamics = TSSM(action_size=4, stoch=32, classes=32, width=512, layers=4, heads=8, context=length,
                    dtype=jnp.bfloat16)
    model = WorldModel(0, 4, image=(96, 96, 3), conv_depth=64, units=1024, reads="image", auxiliary=AUXILIARY,
                       dynamics=dynamics, dtype=jnp.bfloat16)
    pretraining = DreamerV3Objective(model, imagination=imagination, imitation=1.0,
                                     scales={"policy": 0.0, "value": 0.0, "replay_value": 0.0,
                                             **{name: 1.0 for name, _, _ in AUXILIARY}})
    keys = iter(jax.random.split(jax.random.key(5), pretrain_updates + 1))
    sample = jax.jit(lambda key: replay.sample(key, 16, length))
    windows = Loader(train=lambda partition: (jax.device_get(sample(next(keys))) for _ in itertools.count()),
                     val=None, records=16, batch=16)
    began = time.perf_counter()
    pretrained = Trainer(pretraining, laprop(), key=jax.random.key(6)).fit(windows, steps=pretrain_updates,
                                                                           log_every=1000)
    report["pretrain_seconds"] = time.perf_counter() - began
    (out / "pretrained.msgpack").write_bytes(serialization.to_bytes(jax.device_get(pretrained.variables)))
    agent = DreamerV3Objective(model, imagination=imagination, variables=jax.device_get(pretrained.variables))
    report["cloned"] = judge(agent.act, pretrained.variables, agent.memory(drones), jax.random.key(7),
                             film=out / "cloned.npz")
    print(json.dumps(report), flush=True)
    (out / "world.json").write_text(json.dumps(report, indent=1))

    rollout = DreamerRollout(agent, simulator, batch=16, length=length, train_ratio=128.0, replay=replay)
    trainer = Trainer(agent, laprop(), key=jax.random.key(8), rollout=rollout)
    updates = int(environment_seconds * hz * rollout.train_ratio / (rollout.batch * rollout.length))
    state, report["online"], began = trainer.place()[0], [], time.perf_counter()
    for done in range(10_000, updates + 10_000, 10_000):
        state = trainer.fit(_idle(), steps=min(done, updates), log_every=1000, state=state)
        point = {"updates": min(done, updates), "environment_steps": rollout.environment_steps,
                 "seconds": time.perf_counter() - began, "episode_return": rollout.metrics["episode_return"],
                 **judge(agent.act, state.variables, agent.memory(drones), jax.random.key(done))}
        report["online"].append(point)
        print(json.dumps(point), flush=True)
        (out / "world.msgpack").write_bytes(serialization.to_bytes(jax.device_get(state.variables)))
        (out / "world.json").write_text(json.dumps(report, indent=1))
    report["online_agent"] = judge(agent.act, state.variables, agent.memory(drones), jax.random.key(9),
                                   film=out / "video.npz")
    (out / "world.json").write_text(json.dumps(report, indent=1))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("check", "teacher", "world"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument("--seconds", type=float, default=6600.0)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--warm", type=Path)
    parser.add_argument("--level", type=float, default=0.2)
    parser.add_argument("--no-highlight", action="store_true")
    parser.add_argument("--pretrain-updates", type=int, default=50_000)
    parser.add_argument("--environment-seconds", type=float, default=33_333.0)
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    if arguments.phase == "check":
        report = check(arguments.out, assets=arguments.assets, hz=arguments.hz)
    elif arguments.phase == "teacher":
        report = teacher(arguments.out, seconds=arguments.seconds, hz=arguments.hz, warm=arguments.warm,
                         level=arguments.level)
    else:
        report = world(arguments.out, teacher_weights=arguments.teacher, assets=arguments.assets,
                       hz=arguments.hz, pretrain_updates=arguments.pretrain_updates,
                       environment_seconds=arguments.environment_seconds,
                       highlight=not arguments.no_highlight)
    print(json.dumps({key: value for key, value in report.items() if not isinstance(value, list)}))


if __name__ == "__main__":
    main()
