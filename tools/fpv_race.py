"""The FPV gate race's training phases on HyperDrone (`dew.rl.flight`, `dew.rl.hyperdrone`). Each phase
writes a JSON report and its weights to OUT.

    python tools/fpv_race.py check OUT                  HyperDrone's renderer on this GPU: frames of the
                                                         race, and frames a second at 1024 and 4096 drones.
    python tools/fpv_race.py teacher OUT [--seconds S]  The privileged teacher: PPO over the teacher's
                                                         view, 4096 drones, for S seconds; its greedy
                                                         episodes scored as it trains, the best kept, the
                                                         courses' difficulty raised as its crashes fall.
    python tools/fpv_race.py world OUT --teacher W      The world model's phases: the teacher's
                                                         demonstrations, DreamerV3 pretrained on them with
                                                         behaviour cloning, then trained online at
                                                         FlyDreamer's train ratio; the teacher, the cloned
                                                         agent and the online agent scored alike.

`--hz` sets the control rate (60 by default, physics at 120 Hz; at 30, physics at 90). The world model's
windows and imagination span the same seconds at any rate: 64 steps and 15 imagined at 30 Hz, 128 and 30
at 60.

An agent is scored on its drones' first episodes, each run to its end or to `Racing.max_steps`. The score
records the return, the gates passed, the share of episodes that ended in a crash, and the steps flown.
The teacher reads its privileged view. The world model and its agent read the camera's image alone; the
state, the next gates' corners and the drone's airframe are targets for its auxiliary heads.
HyperDrone is the `hyperdrone` extra, Dew's build `hyperdrone==0.1.10.post1`, and needs an NVIDIA GPU
with OptiX.
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
from flax import serialization

from dew.data.dataset import Loader

TEACHER_SIZE = 43
"""The teacher's view's width (`dew.rl.flight.privileged`)."""
AUXILIARY = (("state", 13, 512), ("gates", 24, 512), ("airframe", 6, 256))
"""The world model's auxiliary heads: each privileged view, its width and its head's units."""


def _idle() -> Loader:
    """A loader whose batches a rollout replaces whole."""
    return Loader(train=lambda partition: itertools.repeat({"step": np.zeros((8,), np.float32)}), val=None,
                  records=8, batch=8)


def race(drones: int, views: tuple[str, ...], *, hz: float = 60.0, size: int = 96):
    """A gate race of `drones` drones on HyperDrone at `hz`, its physics at 90 Hz or the nearest rate a
    whole number of steps a control step reaches."""
    from dew.rl.flight import GATES, GateRace, Racing
    from dew.rl.hyperdrone import HyperDroneFlight

    racing = Racing(hz=hz)
    backend = HyperDroneFlight(drones, GATES, size=size, hz=hz, substeps=max(round(90 / hz), 1),
                               rate_limit=racing.rate_limit, headroom=1 + racing.spread,
                               opening=racing.opening, frame=racing.frame)
    return GateRace(backend, racing, views=views)


def teacher_objective():
    """PPO over the teacher's view, its policy's spread a fifth of the actor-critic's at the start, so the
    first rollouts fly rather than tumble."""
    from dew.objectives.rl.control import ActorCritic, ControlPPOObjective

    class Steady(ActorCritic):
        def policy(self, observation):
            location, spread = super().policy(observation)
            return location, spread / 5

    return ControlPPOObjective(Steady(4, (256, 256, 256), (256, 256, 256, 256)), TEACHER_SIZE, discount=0.99,
                               lam=0.95, clip=0.2, entropy_cost=1e-4, reads=("teacher",))


def scored(simulator, policy, params, memory, key: jax.Array, steps: int) -> dict:
    """Each drone's first episode under `policy` (reading the driven step whole): its return, gates
    passed, whether it crashed and its steps, averaged over the drones."""
    from dew.rl.simulator import DrivenSimulator, simulator_rollout, simulator_start

    driven = DrivenSimulator(simulator)
    _, taken = simulator_rollout(driven, policy, params, simulator_start(driven, key, memory), steps,
                                 reset=False)
    observed = jax.device_get(taken.observation)
    ended = np.cumsum(observed["last"], 0) - observed["last"] > 0
    reward = np.where(ended, 0, observed["reward"])
    passed = np.where(ended, 0, observed["observation"]["passed"]).max(0)
    return {"return": float(reward.sum(0).mean()), "gates": float(passed.mean()),
            "gates_max": int(passed.max()),
            "crashed": float((observed["terminal"] & ~ended).any(0).mean()),
            "steps": float((~ended).sum(0).mean() - 1)}


def teacher_policy(objective):
    """The teacher's greedy action from a driven step."""
    def greedy(params, observed, key, memory):
        return objective.greedy(params, observed["observation"], key, memory)

    return greedy


def check(out: Path, *, hz: float = 60.0) -> dict:
    """The race's first frames, and frames a second with images and without."""
    report = {"device": str(jax.devices()[0])}
    sample = race(16, ("image", "passed"), hz=hz)
    frames = [np.asarray(sample.reset(0)["image"])]
    for _ in range(30):
        stepped = sample.step(jnp.zeros((16, 4)))
    frames.append(np.asarray(stepped.observation["image"]))
    frames = np.concatenate(frames)
    np.save(out / "frames.npy", frames)
    report["frames"] = {"shape": list(frames.shape), "mean": float(frames.mean()), "std": float(frames.std())}
    del sample
    for drones, views in ((1024, ("image",)), (4096, ("image",)), (4096, ("teacher",))):
        timed = race(drones, views, hz=hz)
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


def teacher(out: Path, *, seconds: float = 6600.0, drones: int = 4096, hz: float = 60.0) -> dict:
    """PPO's teacher for `seconds`, scored every ten collections, the weights of the best greedy return at
    the highest difficulty reached kept (teacher.msgpack); the courses' difficulty starts at 0.2 and rises
    by 0.1, to 1, at each score whose crashes fall under a third (Song et al. 2021 raised it on the crash
    rate)."""
    from dew.objectives.rl.control import ControlRollout
    from dew.training import Trainer

    simulator = race(drones, ("teacher", "passed"), hz=hz)
    simulator.difficulty = 0.2
    objective = teacher_objective()
    rollout = ControlRollout(objective, simulator, unroll_length=round(32 * hz / 30), epochs=4,
                             minibatches=32)
    trainer = Trainer(objective, optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4)),
                      key=jax.random.key(0), rollout=rollout)
    state, report, best = trainer.place()[0], {"drones": drones, "hz": hz, "curve": []}, (-1.0, -np.inf)
    began, updates, every = time.perf_counter(), 0, 10 * rollout.epochs * rollout.minibatches
    horizon = round(20 * hz)
    while time.perf_counter() - began < seconds:
        state = trainer.fit(_idle(), steps=updates + every, log_every=every, state=state)
        updates += every
        point = {"updates": updates, "seconds": round(time.perf_counter() - began, 1),
                 "environment_steps": updates // (rollout.epochs * rollout.minibatches) * drones
                 * rollout.unroll_length, "difficulty": simulator.difficulty,
                 **scored(simulator, teacher_policy(objective), state.variables, None, jax.random.key(1),
                          horizon)}
        report["curve"].append(point)
        print(json.dumps(point), flush=True)
        if (simulator.difficulty, point["return"]) > best:
            best = simulator.difficulty, point["return"]
            (out / "teacher.msgpack").write_bytes(serialization.to_bytes(jax.device_get(state.variables)))
        if point["crashed"] < 1 / 3:
            simulator.difficulty = min(1.0, simulator.difficulty + 0.1)
        (out / "teacher.json").write_text(json.dumps(report, indent=1))
    kept = (out / "teacher.msgpack").read_bytes()
    variables = serialization.from_bytes(jax.device_get(state.variables), kept)
    report["best"] = {"difficulty": best[0], **scored(simulator, teacher_policy(objective), variables, None,
                                                      jax.random.key(2), simulator.racing.max_steps)}
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


def world(out: Path, *, teacher_weights: Path, drones: int = 16, hz: float = 60.0,
          demonstration_seconds: float = 546.0, pretrain_updates: int = 50_000,
          environment_seconds: float = 33_333.0, capacity_seconds: float = 1092.0) -> dict:
    """FlyDreamer's agent at its size (TransDreamer's dynamics, bfloat16) on the image alone: pretrained on
    the teacher's demonstrations (FlyDreamer's `pretrain_wm` scales and behaviour cloning), then trained
    online at train ratio 128, starting from the demonstrations' replay; the teacher, the cloned agent and
    the online agent scored alike (world.json, pretrained.msgpack, world.msgpack). Each drone flies
    `demonstration_seconds` for the demonstrations; the agent flies `environment_seconds` online over all
    its drones (FlyDreamer's 1M steps at 30 Hz); the replay holds `capacity_seconds` a drone."""
    from rl_acceptance import laprop

    from dew.objectives.rl.dreamer import DreamerRollout, DreamerV3Objective, WorldModel
    from dew.objectives.rl.tssm import TSSM
    from dew.training import Trainer

    scale = hz / 30
    length, imagination = round(64 * scale), round(15 * scale)
    simulator = race(drones, ("image", "state", "gates", "airframe", "passed", "teacher"), hz=hz)
    horizon = simulator.racing.max_steps
    taught = teacher_objective()
    held = serialization.from_bytes(taught.init(jax.random.key(0)), teacher_weights.read_bytes())
    report = {"drones": drones, "hz": hz, "length": length, "imagination": imagination,
              "teacher": scored(simulator, teacher_policy(taught), held, None, jax.random.key(4), horizon)}
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
    report["cloned"] = scored(simulator, agent.act, pretrained.variables, agent.memory(drones),
                              jax.random.key(7), horizon)
    print(json.dumps(report), flush=True)
    (out / "world.json").write_text(json.dumps(report, indent=1))

    rollout = DreamerRollout(agent, simulator, batch=16, length=length, train_ratio=128.0, replay=replay)
    trainer = Trainer(agent, laprop(), key=jax.random.key(8), rollout=rollout)
    updates = int(environment_seconds * hz * rollout.train_ratio / (rollout.batch * rollout.length))
    state, report["online"], began = trainer.place()[0], [], time.perf_counter()
    for done in range(10_000, updates + 10_000, 10_000):
        state = trainer.fit(_idle(), steps=min(done, updates), log_every=1000, state=state)
        point = {"updates": min(done, updates), "environment_steps": rollout.environment_steps,
                 "seconds": time.perf_counter() - began, "episode_return": rollout.metrics["episode_return"]}
        report["online"].append(point)
        print(json.dumps(point), flush=True)
        (out / "world.msgpack").write_bytes(serialization.to_bytes(jax.device_get(state.variables)))
        (out / "world.json").write_text(json.dumps(report, indent=1))
    report["online_agent"] = scored(simulator, agent.act, state.variables, agent.memory(drones),
                                    jax.random.key(9), horizon)
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
    parser.add_argument("--pretrain-updates", type=int, default=50_000)
    parser.add_argument("--environment-seconds", type=float, default=33_333.0)
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    if arguments.phase == "check":
        report = check(arguments.out, hz=arguments.hz)
    elif arguments.phase == "teacher":
        report = teacher(arguments.out, seconds=arguments.seconds, hz=arguments.hz)
    else:
        report = world(arguments.out, teacher_weights=arguments.teacher, hz=arguments.hz,
                       pretrain_updates=arguments.pretrain_updates,
                       environment_seconds=arguments.environment_seconds)
    print(json.dumps({key: value for key, value in report.items() if not isinstance(value, list)}))


if __name__ == "__main__":
    main()
