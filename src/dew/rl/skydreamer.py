"""SkyDreamer's race (Verraest et al., arXiv 2510.14783) flown from the camera's RGB image by collective
thrust and body rates: its task, reward, termination and starts, with Dew's quadrotor and renderer.

The task (Sec II-C, Table III): a track's gates passed in order, each through its pre-gate, itself and its
post-gate, planes `depth` / 2 before and after it. The reward is 5 progress - rate + 30 gate. Progress is the
step's approach to the next pre-gate, zero from the pre-gate through the post-gate. The rate penalty is
(exp(min(|rates|_1, 17)) - 1) over 2 f_c 1e5. The gate reward at each of the three planes is 1 - max(|y|, |z|)
/ `size` where the drone crosses it. An episode ends, its last reward zero, when the drone crosses one of
those planes beyond `size` of the gate's centre, or when it is within 0.5 m of the ground while descending
faster than 1 m/s or rolled or pitched past 60 degrees, and is cut after `steps` steps (2000, 22 s: the
length of SkyDreamer's flights in Fig 5; the paper states no other). An episode starts 2 to 4 m before a
gate (any gate in `anywhere` of starts, else the first), within 1 m to its side and up to 1.3 m below its
centre (0.7 to 1.3 m before the first), at rest, rolled, pitched and turned from the gate's heading up to
20 degrees, its body rates up to 0.1 rad/s and its rotors at a quarter to half of their top speed.

The drone (`QUADCOPTER`) is Dew's quadrotor (`dew.rl.quadrotor`, its flight controller tracking collective
thrust and body rates) with SkyDreamer's Table II where Dew's model has the term: thrust `k_w` per unit mass
and the rotors' speeds from 341.75 to 3100 rad/s (a thrust-to-weight ratio of 6.1), their 30 ms time
constant, and the drag along the body's x and y, linear in the rotors' speed and quadratic in the velocity;
no vertical drag, ground effect or battery sag. An action is FlyDreamer's and Dream to Fly's command (Romero
et al., arXiv 2501.14377, Sec III-B): its first entry the throttle, zero the drone's hover thrust, -1 none and
1 its full thrust; the rest the body rates over `rate_limit`, sent at once.

The policy reads the camera's image (`image`, the renderer's, `SkyDreamerRace`) and, as SkyDreamer's does,
the body rates, the rotors' speeds and the flight plan (Sec II-H, Eq 12: for the next three gates, the change
in position and yaw from the gate before, then their positions and yaws). The privileged entries a world
model decodes (`state`, Sec II-B) are the position and velocity in the world and in the gate headed for, and
the orientation; with randomization off the camera's extrinsics and the dynamics' parameters are constants,
left out. The camera is tilted up `uptilt`, 30 degrees and fixed (the owner's, where SkyDreamer's is 45 to
55), its field of view `fov` that of SkyDreamer's focal length, 25 pixels across 64. Randomization is off:
no parameter spread, disturbances or noise.

Not SkyDreamer's: the track. Its tracks are given only in figures, so `CIRCLE` stands four gates on a circle
in its 8 x 8 m arena. The world frame is x forward, y left, z up (`dew.rl.flight`'s).
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from dew.rl.env import TimeStep
from dew.rl.flight import GATES, OBSTACLES, Camera, Renderer, Scene, Seen, quaternion
from dew.rl.quadrotor import GRAVITY, Body, Flight, Quadrotor, drawn_body, flown, hovering

CIRCLE = (((3.0, 0.0, 1.5), 90.0), ((0.0, 3.0, 1.5), 180.0), ((-3.0, 0.0, 1.5), 270.0),
          ((0.0, -3.0, 1.5), 0.0))
"""Four gates 1.5 m up on a circle of 3 m about the arena's centre, each its centre and heading (degrees),
facing along the circle: a closed track."""
_MASS = Quadrotor().mass
QUADCOPTER = Quadrotor(thrust_coefficient=1.55e-6 * _MASS,
                       drag_coefficient=1.55e-6 * _MASS * Quadrotor().drag_coefficient
                       / Quadrotor().thrust_coefficient,
                       top_speed=3100.0, idle_speed=341.75, motor_lag=(0.03, 0.03),
                       rotor_drag=5.37e-5 * _MASS, quadratic_drag=(4.10e-3 * _MASS, 1.51e-2 * _MASS),
                       vertical_drag=0.0, ground_effect=(0.0, 0.0), discharge=0.0, resistance=(0.0, 0.0))
"""Dew's quadrotor at its nominal mass with SkyDreamer's Table II thrust, rotor speeds, motor time constant
and drag (each per unit mass there, so times the mass here), its yaw torque's ratio to thrust Dew's."""
VIEWS = ("image", "rates", "motors", "plan", "state", "passed")
"""What the race shows: the camera's image, the body rates and rotor speeds, the flight plan, and privileged,
the `state` and the gates `passed`."""
PLANES = (-1, 0, 1)
"""The planes a gate is passed through, in its depth's halves along its normal: pre-gate, gate, post-gate."""


@dataclasses.dataclass(frozen=True)
class Sky:
    """SkyDreamer's race (the module docstring): its control rate `hz`, its gates (`track`: each centre and
    heading in degrees, the track closed), its gate rewards' `size` and the pre- and post-gates' span `depth`
    (Table III), the share of starts before any gate (`anywhere`), the body-rate command's `rate_limit`
    (rad/s), the camera's `uptilt` and horizontal field of view `fov` (degrees), the `quadrotor`, and the
    episode's `steps`."""

    hz: float = 90.0
    track: tuple[tuple[tuple[float, float, float], float], ...] = CIRCLE
    size: float = 0.8
    depth: float = 0.8
    anywhere: float = 0.7
    rate_limit: float = 10.0
    uptilt: float = 30.0
    fov: float = 2 * math.degrees(math.atan(32 / 25))
    quadrotor: Quadrotor = QUADCOPTER
    steps: int = 2000

    @property
    def centers(self) -> np.ndarray:
        """The gates' centres `[G, 3]`."""
        return np.asarray([center for center, _ in self.track], np.float32)

    @property
    def headings(self) -> np.ndarray:
        """The gates' headings `[G]` (radians)."""
        return np.deg2rad(np.asarray([heading for _, heading in self.track], np.float32))


@struct.dataclass
class Lap:
    """One drone's episode: its drawn `body` and its `flight`, its `camera`, the gate it heads for (`index`)
    and the plane of it it crosses next (`stage`, into `PLANES`), its distance to that gate's pre-gate after
    the last step, the gates passed and the steps taken."""

    body: Body
    flight: Flight
    camera: Camera
    index: jax.Array
    stage: jax.Array
    distance: jax.Array
    passed: jax.Array
    steps: jax.Array


def _axes(heading: jax.Array) -> tuple[jax.Array, jax.Array]:
    """A gate's normal (its passing direction) and its left, level."""
    cos, sin, zero = jnp.cos(heading), jnp.sin(heading), jnp.zeros_like(heading)
    return jnp.stack([cos, sin, zero], -1), jnp.stack([-sin, cos, zero], -1)


def in_gate(sky: Sky, index: jax.Array, point: jax.Array) -> jax.Array:
    """`point` in gate `index`'s frame: along its normal, to its left and up from its centre."""
    normal, left = _axes(jnp.asarray(sky.headings)[index])
    offset = point - jnp.asarray(sky.centers)[index]
    return jnp.stack([offset @ normal, offset @ left, offset[2]])


def pre_gate(sky: Sky, index: jax.Array) -> jax.Array:
    """Gate `index`'s pre-gate's centre."""
    normal, _ = _axes(jnp.asarray(sky.headings)[index])
    return jnp.asarray(sky.centers)[index] - sky.depth / 2 * normal


def plan(sky: Sky, index: jax.Array) -> jax.Array:
    """The flight plan `[24]` at gate `index` (Eq 12): for it and the two after it, the change in position
    and heading from the gate before (a half turn either way), then their positions and headings."""
    gates = (index + jnp.arange(-1, 3)) % len(sky.track)
    centers, headings = jnp.asarray(sky.centers)[gates], jnp.asarray(sky.headings)[gates]
    change = headings[1:] - headings[:-1]
    turned = jnp.arctan2(jnp.sin(change), jnp.cos(change))
    changes = jnp.concatenate([centers[1:] - centers[:-1], turned[:, None]], -1).reshape(-1)
    return jnp.concatenate([changes, jnp.concatenate([centers[1:], headings[1:, None]], -1).reshape(-1)])


def roll_pitch(orientation: jax.Array) -> tuple[jax.Array, jax.Array]:
    """An orientation's roll and pitch (its yaw, pitch, roll Euler angles')."""
    w, x, y, z = orientation
    roll = jnp.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return roll, jnp.arcsin(jnp.clip(2 * (w * y - z * x), -1.0, 1.0))


@struct.dataclass
class SkyRace:
    """One drone racing `sky`'s track as a `dew.rl.env.Env`: actions `[4]` in [-1, 1] (the module docstring);
    observations a dict of the `views` asked for, each but the image (`VIEWS`, which `SkyDreamerRace` adds).
    It cuts its own episodes (`Sky.steps`)."""

    sky: Sky = struct.field(pytree_node=False, default_factory=Sky)
    views: tuple[str, ...] = struct.field(pytree_node=False, default=VIEWS[1:])

    action_size = 4

    def __post_init__(self):
        if not self.views or not set(self.views) <= set(VIEWS[1:]):
            raise ValueError(f"a SkyDreamer race shows some of {VIEWS[1:]}, not {self.views}")

    def observed(self, lap: Lap) -> dict[str, jax.Array]:
        """The views of `lap` asked for."""
        sky, flight = self.sky, lap.flight
        normal, left = _axes(jnp.asarray(sky.headings)[lap.index])
        up = jnp.asarray([0.0, 0.0, 1.0])
        in_its = jnp.stack([flight.velocity @ axis for axis in (normal, left, up)])
        orientation = flight.orientation * jnp.where(flight.orientation[0] < 0, -1.0, 1.0)
        state = jnp.concatenate([flight.position, in_gate(sky, lap.index, flight.position), flight.velocity,
                                 in_its, orientation])
        shown = {"rates": flight.angular_velocity, "motors": flight.speed, "plan": plan(sky, lap.index),
                 "state": state, "passed": lap.passed}
        return {view: shown[view] for view in self.views}

    def scene(self, lap: Lap) -> Scene:
        """What a renderer draws of `lap`: the track's gates, none lit, no pillars."""
        sky = self.sky
        count = len(sky.track)
        center = np.resize(sky.centers, (GATES, 3))
        heading = np.resize(sky.headings, GATES)
        return Scene(jnp.asarray(center), jnp.asarray(heading), jnp.asarray(count), jnp.asarray(-1),
                     jnp.zeros((OBSTACLES, 4)), lap.camera.look)

    def reset(self, key: jax.Array) -> tuple[Lap, TimeStep]:
        """A start of Table III's (the module docstring)."""
        sky = self.sky
        keys = iter(jax.random.split(key, 9))
        anywhere = jax.random.uniform(next(keys)) < sky.anywhere
        index = jnp.where(anywhere, jax.random.randint(next(keys), (), 0, len(sky.track)), 0)
        low = jnp.where(anywhere, 0.0, 0.7)
        along = jax.random.uniform(next(keys), (), minval=-4.0, maxval=-2.0)
        aside = jax.random.uniform(next(keys), (), minval=-1.0, maxval=1.0)
        below = jax.random.uniform(next(keys), (), minval=low, maxval=1.3)
        normal, left = _axes(jnp.asarray(sky.headings)[index])
        position = jnp.asarray(sky.centers)[index] + along * normal + aside * left - below * jnp.asarray(
            [0.0, 0.0, 1.0])
        turned = jax.random.uniform(next(keys), (3,), minval=-math.pi / 9, maxval=math.pi / 9)
        orientation = quaternion(turned[0], turned[1], jnp.asarray(sky.headings)[index] + turned[2])
        body = drawn_body(sky.quadrotor, next(keys), jnp.zeros(()))
        rates = jax.random.uniform(next(keys), (3,), minval=-0.1, maxval=0.1)
        speeds = sky.quadrotor.top_speed * jax.random.uniform(next(keys), (4,), minval=0.25, maxval=0.5)
        flight = dataclasses.replace(hovering(sky.quadrotor, body, position, orientation), speed=speeds,
                                     angular_velocity=rates, measured=rates)
        zero = jnp.zeros(())
        camera = Camera(uptilt=jnp.radians(jnp.asarray(sky.uptilt)), fov=jnp.radians(jnp.asarray(sky.fov)),
                        mount=jnp.zeros(2), exposure=jnp.ones(()), white_balance=jnp.ones(3),
                        gamma=jnp.ones(()), noise=zero, shutter=zero, look=jnp.zeros(4, jnp.int32))
        lap = Lap(body=body, flight=flight, camera=camera, index=index, stage=jnp.zeros((), jnp.int32),
                  distance=jnp.linalg.norm(position - pre_gate(sky, index)), passed=jnp.zeros((), jnp.int32),
                  steps=jnp.zeros((), jnp.int32))
        done = jnp.zeros((), bool)
        return lap, TimeStep(self.observed(lap), zero, done, done)

    def commanded(self, lap: Lap, action: jax.Array) -> tuple[jax.Array, jax.Array]:
        """An action as the flight controller takes it: the collective thrust (N), the throttle's share of
        the hover below zero and of the span from the hover to the full thrust above; the body rates (rad/s),
        its share of `rate_limit`."""
        action = jnp.clip(action, -1, 1)
        hover = lap.body.mass * GRAVITY
        full = self.sky.quadrotor.nominal_thrust
        collective = hover + action[0] * jnp.where(action[0] >= 0, full - hover, hover)
        return collective, action[1:] * self.sky.rate_limit

    def step(self, lap: Lap, action: jax.Array) -> tuple[Lap, TimeStep]:
        """The drone flown a control step, then the planes it crossed in order, the reward and whether the
        episode ended (the module docstring)."""
        sky = self.sky
        before = lap.flight.position
        collective, rates = self.commanded(lap, action)
        flight, _ = flown(sky.quadrotor, lap.body, lap.flight, collective, rates, jax.random.key(0), sky.hz,
                          jnp.zeros(()))
        after = flight.position
        index, stage, approaching = lap.index, lap.stage, lap.stage == 0
        crossing, collided = jnp.ones((), bool), jnp.zeros((), bool)
        gate, passed = jnp.zeros(()), lap.passed
        for _ in PLANES:
            # The next plane: crossed this step where the step's path meets it from behind.
            plane = sky.depth / 2 * (stage - 1)
            start, end = in_gate(sky, index, before)[0], in_gate(sky, index, after)[0]
            crossed = crossing & (start < plane) & (end >= plane)
            where = before + (plane - start) / jnp.where(crossed, end - start, 1.0) * (after - before)
            local = in_gate(sky, index, where)
            off = jnp.maximum(jnp.abs(local[1]), jnp.abs(local[2])) / sky.size
            collided = collided | (crossed & (off > 1))
            gate = gate + jnp.where(crossed & (off <= 1), 1 - off, 0.0)
            through = crossed & (stage == len(PLANES) - 1)
            passed = passed + through.astype(passed.dtype)
            index = jnp.where(through, (index + 1) % len(sky.track), index)
            stage = jnp.where(crossed, (stage + 1) % len(PLANES), stage)
            crossing = crossed
        # Progress toward the pre-gate headed for, none from it through the post-gate.
        approached = lap.distance - jnp.linalg.norm(after - pre_gate(sky, lap.index))
        progress = jnp.where(approaching, approached, 0.0)
        rate = jnp.expm1(jnp.minimum(jnp.sum(jnp.abs(flight.angular_velocity)), 17.0)) / (2 * sky.hz * 1e5)
        roll, pitch = roll_pitch(flight.orientation)
        grounded = (after[2] < 0.5) & ((flight.velocity[2] < -1.0) | (jnp.abs(roll) > math.pi / 3)
                                       | (jnp.abs(pitch) > math.pi / 3))
        ended = collided | grounded
        reward = jnp.where(ended, 0.0, 5 * progress - rate + 30 * gate)
        lap = dataclasses.replace(lap, flight=flight, index=index, stage=stage,
                                  distance=jnp.linalg.norm(after - pre_gate(sky, index)), passed=passed,
                                  steps=lap.steps + 1)
        return lap, TimeStep(self.observed(lap), reward, ended, ~ended & (lap.steps >= sky.steps))


class SkyDreamerRace(Seen):
    """A batch of `renderer.count` drones racing `sky`'s track (`SkyRace`, stepped on device), seen through
    `renderer`'s cameras (`dew.rl.flight.Seen`), as a `dew.rl.simulator.Simulator`: observations a dict of
    the `views` asked for (`VIEWS`), each `[B, ...]`."""

    def __init__(self, renderer: Renderer | None, sky: Sky | None = None, views: tuple[str, ...] = VIEWS,
                 count: int | None = None):
        if not views or not set(views) <= set(VIEWS):
            raise ValueError(f"a SkyDreamer race shows some of {VIEWS}, not {views}")
        if renderer is None and count is None:
            raise ValueError("a race without a renderer is told its drones' count")
        self.sky = Sky() if sky is None else sky
        shown = tuple(view for view in views if view != "image")
        count = renderer.count if renderer is not None else int(count or 0)
        super().__init__(renderer, SkyRace(self.sky, shown if shown else ("passed",)), views, count)


__all__ = ["CIRCLE", "PLANES", "QUADCOPTER", "VIEWS", "Lap", "Sky", "SkyDreamerRace", "SkyRace", "in_gate",
           "plan", "pre_gate", "roll_pitch"]
