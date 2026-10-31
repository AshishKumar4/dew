"""The FPV gate race's training phases on HyperDrone (`dew.rl.flight`, `dew.rl.hyperdrone`), each writing
a JSON report and its weights to OUT.

    python tools/fpv_race.py check OUT                  HyperDrone's renderer on this GPU: frames of the
                                                         race, and frames a second at 1024 and 4096 drones.
    python tools/fpv_race.py teacher OUT [--seconds S]  The privileged teacher: PPO over the drones' state
                                                         and goal, 4096 drones, for S seconds; its greedy
                                                         episodes scored as it trains, the best kept.
    python tools/fpv_race.py world OUT --teacher W      The world model's phases: the teacher's
                                                         demonstrations, DreamerV3 pretrained on them with
                                                         behaviour cloning, then trained online at
                                                         FlyDreamer's train ratio; the teacher, the cloned
                                                         agent and the online agent scored alike.

An agent's score is its drones' first episodes, each to its end or `Racing.max_steps`: the return, the
gate bonuses earned, the share that ended in a collision, and the steps flown. The teacher reads the
privileged state and goal; the world model and its agent read the camera's image alone, the state and
goal its auxiliary heads' targets (FlyDreamer's loss scales). HyperDrone is the `hyperdrone` extra, on an
NVIDIA GPU with OptiX.
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

GATES = 12
TEACHER_SCALE = (30, 30, 5, 10, 10, 10, 1, 1, 1, 1, 4, 4, 4, 1, 1, 1, 30)
TEACHER_CENTER = (30, -30, 5) + (0,) * 14
"""The teacher's inputs, the state and goal joined, each shifted and divided to about unit size: the
course's extent and height, the speed, the attitude, the rate limit and the distance to a gate."""


def _idle() -> Loader:
    """A loader whose batches a rollout replaces whole."""
    return Loader(train=lambda partition: itertools.repeat({"step": np.zeros((8,), np.float32)}), val=None,
                  records=8, batch=8)


def race(drones: int, views: tuple[str, ...], size: int = 96):
    """A gate race of `drones` drones on HyperDrone, its throttle's hover the drone's own."""
    from dew.rl.flight import GateRace, Racing
    from dew.rl.hyperdrone import HyperDroneFlight

    backend = HyperDroneFlight(drones, GATES, size=size)
    return GateRace(backend, Racing(hover_throttle=backend.hover_throttle), views=views)


def teacher_objective():
    """PPO over the state and goal (`TEACHER_SCALE`), its policy's spread a fifth of the actor-critic's
    at the start, so the first rollouts fly rather than tumble."""
    from dew.objectives.rl.control import ActorCritic, ControlPPOObjective

    scale, center = jnp.asarray(TEACHER_SCALE, jnp.float32), jnp.asarray(TEACHER_CENTER, jnp.float32)

    class Steady(ActorCritic):
        def policy(self, observation):
            location, spread = super().policy(observation)
            return location, spread / 5

    class Teacher(ControlPPOObjective):
        def seen(self, observation):
            return (super().seen(observation) - center) / scale

    return Teacher(Steady(4, (256, 256, 256), (256, 256, 256, 256)), len(TEACHER_SCALE), discount=0.99,
                   lam=0.95, clip=0.2, reward_scaling=0.1, entropy_cost=1e-4, reads=("state", "goal"))


def scored(simulator, policy, params, memory, key: jax.Array, steps: int) -> dict:
    """Each drone's first episode under `policy` (reading the driven step whole): its return, gate
    bonuses, whether it collided and its steps, averaged over the drones."""
    from dew.rl.simulator import DrivenSimulator, simulator_rollout, simulator_start

    driven = DrivenSimulator(simulator)
    _, taken = simulator_rollout(driven, policy, params, simulator_start(driven, key, memory), steps,
                                 reset=False)
    observed = jax.device_get(taken.observation)
    ended = np.cumsum(observed["last"], 0) - observed["last"] > 0
    reward = np.where(ended, 0, observed["reward"])
    return {"return": float(reward.sum(0).mean()), "gate_bonuses": float((reward > 10).sum(0).mean()),
            "collided": float((observed["terminal"] & ~ended).any(0).mean()),
            "steps": float((~ended).sum(0).mean() - 1)}


def teacher_policy(objective):
    """The teacher's greedy action from a driven step."""
    def greedy(params, observed, key, memory):
        return objective.greedy(params, observed["observation"], key, memory)

    return greedy


def check(out: Path) -> dict:
    """The race's first frames, and frames a second with images and without."""
    report = {"device": str(jax.devices()[0])}
    sample = race(16, ("image", "state", "goal"))
    frames = np.asarray(sample.reset(0)["image"])
    for _ in range(30):
        stepped = sample.step(jnp.zeros((16, 4)))
    frames = np.concatenate([frames, np.asarray(stepped.observation["image"])])
    np.save(out / "frames.npy", frames)
    report["frames"] = {"shape": list(frames.shape), "mean": float(frames.mean()), "std": float(frames.std())}
    del sample
    for drones, views in ((1024, ("image", "state", "goal")), (4096, ("image", "state", "goal")),
                          (4096, ("state", "goal"))):
        timed = race(drones, views)
        timed.reset(0)
        actions = jnp.zeros((drones, 4))
        for _ in range(5):
            stepped = timed.step(actions)
        jax.block_until_ready(stepped.reward)
        began = time.perf_counter()
        for _ in range(50):
            stepped = timed.step(actions)
        jax.block_until_ready(stepped.reward)
        report[f"{drones}{'' if 'image' in views else '-physics'}_frames_per_second"] = 50 * drones / (
            time.perf_counter() - began)
        del timed
    (out / "check.json").write_text(json.dumps(report, indent=1))
    return report


def teacher(out: Path, *, seconds: float = 6600.0, drones: int = 4096) -> dict:
    """PPO's teacher for `seconds`, scored every ten collections; the best greedy return's weights kept
    (teacher.msgpack)."""
    from dew.objectives.rl.control import ControlRollout
    from dew.training import Trainer

    simulator = race(drones, ("state", "goal"))
    objective = teacher_objective()
    rollout = ControlRollout(objective, simulator, unroll_length=32, epochs=4, minibatches=32)
    trainer = Trainer(objective, optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4)),
                      key=jax.random.key(0), rollout=rollout)
    state, report, best = trainer.place()[0], {"drones": drones, "curve": []}, -np.inf
    began, updates, every = time.perf_counter(), 0, 10 * rollout.epochs * rollout.minibatches
    while time.perf_counter() - began < seconds:
        state = trainer.fit(_idle(), steps=updates + every, log_every=every, state=state)
        updates += every
        point = {"updates": updates, "seconds": round(time.perf_counter() - began, 1),
                 "environment_steps": updates // (rollout.epochs * rollout.minibatches) * drones * 32,
                 **scored(simulator, teacher_policy(objective), state.variables, None, jax.random.key(1),
                          600)}
        report["curve"].append(point)
        print(json.dumps(point), flush=True)
        if point["return"] > best:
            best = point["return"]
            (out / "teacher.msgpack").write_bytes(serialization.to_bytes(jax.device_get(state.variables)))
        (out / "teacher.json").write_text(json.dumps(report, indent=1))
    kept = (out / "teacher.msgpack").read_bytes()
    variables = serialization.from_bytes(jax.device_get(state.variables), kept)
    report["best"] = scored(simulator, teacher_policy(objective), variables, None, jax.random.key(2), 2000)
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


def world(out: Path, *, teacher_weights: Path, drones: int = 16, demonstration_steps: int = 16384,
          pretrain_updates: int = 50_000, environment_steps: int = 1_000_000, capacity: int = 32_768) -> dict:
    """FlyDreamer's agent at its size (TransDreamer's dynamics, bfloat16) on the image alone: pretrained on
    the teacher's demonstrations (FlyDreamer's `pretrain_wm` scales and behaviour cloning), then trained
    online at train ratio 128, from the demonstrations' replay on; the teacher, the cloned agent and the
    online agent scored alike (world.json, pretrained.msgpack, world.msgpack)."""
    from rl_acceptance import laprop

    from dew.objectives.rl.dreamer import DreamerRollout, DreamerV3Objective, WorldModel
    from dew.objectives.rl.tssm import TSSM
    from dew.training import Trainer

    simulator = race(drones, ("image", "state", "goal"))
    taught = teacher_objective()
    held = taught.init(jax.random.key(0))
    held = serialization.from_bytes(held, teacher_weights.read_bytes())
    report = {"drones": drones, "teacher": scored(simulator, teacher_policy(taught), held, None,
                                                  jax.random.key(4), 2000)}
    began = time.perf_counter()
    replay = demonstrations(simulator, taught, held, demonstration_steps, capacity)
    report["demonstration_seconds"] = time.perf_counter() - began
    print(json.dumps(report), flush=True)

    dynamics = TSSM(action_size=4, stoch=32, classes=32, width=512, layers=4, heads=8, context=64,
                    dtype=jnp.bfloat16)
    model = WorldModel(0, 4, image=(96, 96, 3), conv_depth=64, units=1024, reads="image",
                       auxiliary=(("state", 13, 512), ("goal", 4, 256)), dynamics=dynamics,
                       dtype=jnp.bfloat16)
    pretraining = DreamerV3Objective(model, scales={"policy": 0.0, "value": 0.0, "replay_value": 0.0,
                                                    "state": 1.0, "goal": 1.0}, imitation=1.0)
    keys = iter(jax.random.split(jax.random.key(5), pretrain_updates + 1))
    sample = jax.jit(lambda key: replay.sample(key, 16, 64))
    windows = Loader(train=lambda partition: (jax.device_get(sample(next(keys))) for _ in itertools.count()),
                     val=None, records=16, batch=16)
    began = time.perf_counter()
    pretrained = Trainer(pretraining, laprop(), key=jax.random.key(6)).fit(windows, steps=pretrain_updates,
                                                                           log_every=1000)
    report["pretrain_seconds"] = time.perf_counter() - began
    (out / "pretrained.msgpack").write_bytes(serialization.to_bytes(jax.device_get(pretrained.variables)))
    agent = DreamerV3Objective(model, variables=jax.device_get(pretrained.variables))
    report["cloned"] = scored(simulator, agent.act, pretrained.variables, agent.memory(drones),
                              jax.random.key(7), 2000)
    print(json.dumps(report), flush=True)
    (out / "world.json").write_text(json.dumps(report, indent=1))

    rollout = DreamerRollout(agent, simulator, batch=16, length=64, train_ratio=128.0, replay=replay)
    trainer = Trainer(agent, laprop(), key=jax.random.key(8), rollout=rollout)
    updates = int(environment_steps * rollout.train_ratio / (rollout.batch * rollout.length))
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
                                    jax.random.key(9), 2000)
    (out / "world.json").write_text(json.dumps(report, indent=1))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("check", "teacher", "world"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--seconds", type=float, default=6600.0)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--pretrain-updates", type=int, default=50_000)
    parser.add_argument("--environment-steps", type=int, default=1_000_000)
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    if arguments.phase == "check":
        report = check(arguments.out)
    elif arguments.phase == "teacher":
        report = teacher(arguments.out, seconds=arguments.seconds)
    else:
        report = world(arguments.out, teacher_weights=arguments.teacher,
                       pretrain_updates=arguments.pretrain_updates,
                       environment_steps=arguments.environment_steps)
    print(json.dumps({key: value for key, value in report.items() if not isinstance(value, list)}))


if __name__ == "__main__":
    main()
