"""FlyDreamer's FPV task (AshishKumar4/FlyDreamer at 043c9ef382048278ccf26849e54814ef5510a737,
embodied/envs/fpv) run on drawn inputs in float64: the oracle of tests/test_flight.py, recorded by
tools/venv_reference.py.

    PYTHONPATH=<checkout> python tools/flydreamer_worker.py SITE    `IN OUT` lines on stdin, each `done`

The task is built as FlyDreamer's run builds it (factory.py's builders over its checkout's
dreamerv3/configs.yaml `env.fpv`), with a case's overrides. A case is in Dew's frame (x forward, y left,
z up, the body forward, left, up); the worker turns it to FlyDreamer's NED and FRD, half a turn about x:
a point or velocity (x, -y, -z), a quaternion (w, x, -y, -z), body rates (p, -q, -r), and a course back.

SITE `course` places a case's `course` `[G, 3]` by FlyDreamer's waypoint randomization
(waypoints/base.py `_apply_randomization`), its draws the case's `angle`, `radius` and `height` `[G]`;
OUT gets the course placed. SITE `race` flies each of a case's drones through FPVEnv (env.py) on the
config's course, its randomization off and its episodes cut at the case's `max_steps`: the backend
answers each step with the drone's next recorded state (`position`, `velocity`, `orientation`,
`rates` `[D, T, ...]`) and whether it `collided` `[D, T]`, a reset with the drone's next start
(`starts` `[D, E, 13]`), and each step takes the case's `actions` `[T, D, 4]`; OUT gets each step's
`reward`, whether it was `terminal` or the episode's `last`, its privileged `state` and `goal` turned
back to Dew's frame, and the `command` the backend was given.
OUT holds `<case>.truth.<leaf>`.

FlyDreamer computes in float32 where it casts (types.py, navigation.py, waypoints/base.py, env.py and
factory.py each `np.float32`); the worker reads each module's float32 as float64 (`Wide`). FlyDreamer's
packages import its simulators and its agent; the worker loads the task's modules alone (`task`).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

TURN = np.asarray([1.0, -1.0, -1.0])
"""Half a turn about x, between Dew's frame and FlyDreamer's NED and FRD."""


class Wide:
    """NumPy with its float32 read as float64, and `random` replaced where given."""

    float32 = np.float64

    def __init__(self, random=None):
        if random is not None:
            self.random = random

    def __getattr__(self, name):
        return getattr(np, name)


class Drawn:
    """A NumPy generator whose `uniform` answers given values in order."""

    def __init__(self, values):
        self.values = iter(values)

    def uniform(self, low, high):
        return next(self.values)


def task():
    """FlyDreamer's task modules, its packages' own imports skipped, and the run's `env.fpv` config."""
    import yaml

    checkout = next(Path(entry) for entry in sys.path if (Path(entry) / "embodied" / "envs" / "fpv").is_dir())
    for name in ("embodied", "embodied.envs", "embodied.envs.fpv", "embodied.envs.fpv.backends",
                 "embodied.envs.fpv.rewards", "embodied.envs.fpv.waypoints"):
        package = types.ModuleType(name)
        package.__path__ = [str(checkout.joinpath(*name.split(".")))]
        sys.modules[name] = package
    sys.modules["embodied"].Env = object
    sys.modules["elements"] = types.ModuleType("elements")
    from embodied.envs.fpv import env, factory, types as fpv_types
    from embodied.envs.fpv.rewards import base as reward_base, navigation
    from embodied.envs.fpv.waypoints import base as waypoint_base

    for module in (env, factory, fpv_types, reward_base, navigation, waypoint_base):
        module.np = Wide()
    with open(checkout / "dreamerv3" / "configs.yaml") as file:
        config = yaml.safe_load(file)["defaults"]["env"]["fpv"]
    return types.SimpleNamespace(env=env, factory=factory, types=fpv_types, navigation=navigation,
                                 waypoints=waypoint_base, config=config)


def course(fpv, case):
    """The case's course placed by FlyDreamer's randomization, from its draws turned to NED."""
    config = fpv.factory.build_randomization_config(fpv.config)
    draws = np.stack([-case["angle"], case["radius"], -case["height"]], -1).ravel()
    manager = types.SimpleNamespace(_config=types.SimpleNamespace(randomization=config))
    fpv.waypoints.np = Wide(random=types.SimpleNamespace(default_rng=lambda seed: Drawn(draws)))
    placed = fpv.waypoints.WaypointManager._apply_randomization(manager, list(case["course"] * TURN))
    fpv.waypoints.np = Wide()
    return {"course": np.stack(placed) * TURN}


def race(fpv, case):
    """Each drone flown through FPVEnv over its recorded states, its actions the case's."""
    config = {**fpv.config, "max_steps": int(case["max_steps"]),
              "waypoint_randomization": {**fpv.config["waypoint_randomization"], "enabled": False}}

    class Course(fpv.waypoints.WaypointManager):
        def initialize_visuals(self, backend):
            pass

        def update_visuals(self, backend, reached_index):
            pass

        def cleanup_visuals(self, backend):
            pass

    def state(position, velocity, orientation, rates):
        return fpv.types.DroneState(position * TURN, velocity * TURN, orientation * np.r_[1.0, TURN],
                                    rates * TURN)

    class Replay:
        def __init__(self, drone):
            self.drone, self.step_index, self.episode, self.commands = drone, 0, 0, []

        def reset(self):
            start = case["starts"][self.drone, self.episode]
            self.episode += 1
            return state(start[:3], start[3:6], start[6:10], start[10:])

        def step(self, command, dt):
            self.commands.append(command.to_array())
            at = self.drone, self.step_index
            self.step_index += 1
            reached = (case[name][at] for name in ("position", "velocity", "orientation", "rates"))
            return state(*reached), bool(case["collided"][at])

        def get_images(self, size, include_depth=True):
            return fpv.types.ImageData.empty(size, include_depth)

    drones, steps = case["collided"].shape
    out = {name: np.zeros((drones, steps), dtype) for name, dtype in
           (("reward", np.float64), ("terminal", bool), ("last", bool))}
    out |= {"command": np.zeros((drones, steps, 4)), "state": np.zeros((drones, steps, 13)),
            "goal": np.zeros((drones, steps, 4))}
    for drone in range(drones):
        backend = Replay(drone)
        build = fpv.factory
        env = fpv.env.FPVEnv(backend, fpv.navigation.NavigationReward(build.build_reward_config(config)),
                             Course(build.build_waypoint_config(config, fpv.types.MarkerConfig())),
                             build.build_env_config(config), build.build_control_config(config))
        env.step({"reset": True})
        for step in range(steps):
            stepped = env.step({"action": case["actions"][step, drone], "reset": False})
            out["reward"][drone, step] = stepped["reward"]
            out["terminal"][drone, step] = stepped["is_terminal"]
            out["last"][drone, step] = stepped["is_last"]
            out["state"][drone, step] = stepped["state"] * np.r_[TURN, TURN, 1.0, TURN, TURN]
            out["goal"][drone, step] = stepped["goal"] * np.r_[TURN, 1.0]
            if stepped["is_last"]:
                env.step({"reset": True})
        out["command"][drone] = np.stack(backend.commands)
    return out


def main():
    site = {"course": course, "race": race}[sys.argv[1]]
    fpv = task()
    for line in sys.stdin:
        given, answer = line.split()
        with np.load(given) as loaded:
            case = {key.removeprefix("case."): loaded[key] for key in loaded.files}
        np.savez(answer, **{f"case.truth.{leaf}": value for leaf, value in site(fpv, case).items()})
        print("done", flush=True)


if __name__ == "__main__":
    main()
