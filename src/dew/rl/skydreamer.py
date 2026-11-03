"""SkyDreamer's race (Verraest et al., arXiv 2510.14783) flown from the camera's RGB image by collective
thrust and body rates: its task, reward, termination and starts, with Dew's quadrotor and renderer.

The task (Sec II-C, Table III): a track's gates passed in order, each through its pre-gate, itself and its
post-gate, planes `depth` / 2 before and after it. The reward is 5 progress - rate + 30 gate. Progress is the
step's approach to the next pre-gate, zero from the pre-gate through the post-gate: a step that crosses the
pre-gate approaches it up to where it crosses. The rate penalty is (exp(min(|rates|_1, 17)) - 1) over 2 f_c
1e5. The gate reward at each of the three planes is 1 - max(|y|, |z|) / `size` where the drone crosses it. An
episode ends, its last reward zero, when the drone crosses one of those planes beyond `size` of the gate's
centre, where the race has `frames` strikes the frame of a gate it is not passing (the race's opening and
frame, `Racing`), or is within 0.5 m of the ground while descending faster than 1 m/s or rolled or pitched
past 60 degrees; it is cut after `steps` steps (2000, 22 s: the length of SkyDreamer's flights in Fig 5; the
paper states no other). An episode starts on one of the race's `tracks`, drawn alike, 2 to 4 m before a gate
(any gate in `anywhere` of starts, else the first), within 1 m to its side and up to 1.3 m below its centre
(0.7 to 1.3 m before the first), at rest, rolled, pitched and turned from the gate's heading up to 20 degrees,
its body rates up to 0.1 rad/s and its rotors at a quarter to half of their top speed.

A track's gates are seen or not: SkyDreamer adds gates no one sees to its tracks in training "to enforce the
correct flight maneuvers" (Sec III-A), a gate above one to fly over it. Its two tracks, given only in figures
(Figs 4 and 6, its 8 x 8 m arena), are reconstructed as `INVERTED_LOOP` and `LADDER`; `CIRCLE` stands four
gates on a circle in the same arena.

The drone (`QUADCOPTER`) is Dew's quadrotor (`dew.rl.quadrotor`, its flight controller tracking collective
thrust and body rates) with SkyDreamer's Table II where Dew's model has the term: thrust `k_w` per unit mass
and the rotors' speeds from 341.75 to 3100 rad/s (a thrust-to-weight ratio of 6.1), their 30 ms time
constant, and the drag along the body's x and y, linear in the rotors' speed and quadratic in the velocity;
no vertical drag, ground effect or battery sag. An action is FlyDreamer's and Dream to Fly's command (Romero
et al., arXiv 2501.14377, Sec III-B): its first entry the throttle, zero the drone's hover thrust, -1 none and
1 its full thrust; the rest the body rates over the drone's rate limit; it reaches the flight controller
`latency` seconds later (`delays`; SkyDreamer's image and action delays together are 33 and 11 ms).

The policy reads the camera's image (`image`, the renderer's, `SkyDreamerRace`) and, as SkyDreamer's does, the
body rates, the rotors' speeds (each a share of the top speed, so a decoder's squared error over them is of
the others' size) and the flight plan (Sec II-H, Eq 12: for the next three gates, the change in position and
yaw from the gate before, then their positions and yaws). The plan moves on as the drone crosses a gate's
plane: SkyDreamer moves it at a random point from the pre-gate to the post-gate in training, and where its
world model's decoded position passes 0.15 m before the gate in flight; here it moves at the gate. The
privileged entries a world model decodes (`state`, Sec II-B) are the position and velocity in the world and in
the gate headed for, and the orientation, the roll, pitch and yaw in the world and the yaw from the gate's
heading; the camera's extrinsics are fixed and the dynamics' parameters are left out. The camera is tilted up
30 degrees, fixed as a real FPV quadrotor's camera is mounted (`Racing`'s; SkyDreamer's is 45 to 55), its
horizontal field of view `fov`.

Randomization is `ramp`'s, a race's curriculum from 0 (none) to 1, as `dew.rl.flight`'s, one level for
every track or each track's own, so a curriculum may raise each track's as its own flights allow: the
drone's body and its rotors' noise (`dew.rl.quadrotor.drawn_body`, `flown`), its rate limit, the camera's
image and the scene's looks (`dew.rl.flight.drawn_camera`), and the latency's spread, half its nominal
either way at 1.
The world frame is x forward, y left, z up (`dew.rl.flight`'s).
"""

from __future__ import annotations

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from dew.rl.env import TimeStep
from dew.rl.flight import (
    GATES,
    OBSTACLES,
    Camera,
    Racing,
    Renderer,
    Scene,
    Seen,
    commanded,
    drawn_camera,
    gate_axes,
    quaternion,
    shown,
)
from dew.rl.quadrotor import GRAVITY, Body, Flight, Quadrotor, drawn_body, flown, hovering

type Gate = tuple[tuple[float, float, float], float, bool]
"""A track's gate: its centre, its heading (degrees, the way it is passed) and whether it is seen."""
CIRCLE: tuple[Gate, ...] = (((3.0, 0.0, 1.5), 90.0, True), ((0.0, 3.0, 1.5), 180.0, True),
                            ((-3.0, 0.0, 1.5), 270.0, True), ((0.0, -3.0, 1.5), 0.0, True))
"""Four gates 1.5 m up on a circle of 3 m about the arena's centre, facing along the circle."""
INVERTED_LOOP: tuple[Gate, ...] = (((3.0, 0.0, 1.35), -90.0, True), ((-2.1, 0.0, 4.0), 90.0, False),
                                   ((-2.0, 0.0, 1.35), -90.0, True), ((-2.1, 0.0, 4.0), 90.0, False))
"""SkyDreamer's inverted loop (Fig 6, its frame North-East-Down turned to this one's): through the first gate,
over the second, through it the other way after a split-S, and back over it; the gates' centres 1.35 m up
(its 2.7 m gates standing), the unseen gates 4 m up above the second."""
LADDER: tuple[Gate, ...] = (((3.0, 0.0, 1.35), -90.0, True), ((3.0, 0.0, 4.0), -90.0, False),
                            ((-2.1, 0.0, 4.0), 90.0, False), ((-2.0, 0.0, 1.35), -90.0, True),
                            ((-2.1, 0.0, 4.0), 90.0, False))
"""SkyDreamer's ladder inverted loop (Fig 4): through the first gate and, after a full turn, over it the same
way, then the inverted loop's second gate."""
TRACKS = {"circle": CIRCLE, "inverted_loop": INVERTED_LOOP, "ladder": LADDER}
"""The tracks a race is drawn over, by name."""
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
    """SkyDreamer's race (the module docstring): its control rate `hz`, the `tracks` an episode is drawn
    over (`TRACKS`' names), its gate rewards' `size` and the pre- and post-gates' span `depth` (Table III),
    the share of starts before any gate (`anywhere`), the camera's horizontal field of view `fov` (degrees),
    the command's nominal `latency` (seconds), the `quadrotor`, the episode's `steps`, and whether the frames
    of gates the drone is not passing are struck (`frames`)."""

    hz: float = 90.0
    tracks: tuple[str, ...] = ("circle",)
    size: float = 0.8
    depth: float = 0.8
    anywhere: float = 0.7
    fov: float = 2 * math.degrees(math.atan(32 / 25))
    latency: float = 0.0
    quadrotor: Quadrotor = QUADCOPTER
    steps: int = 2000
    frames: bool = False

    def __post_init__(self):
        # A record holds the names as a list.
        object.__setattr__(self, "tracks", tuple(self.tracks))
        if not self.tracks or not set(self.tracks) <= set(TRACKS):
            raise ValueError(f"a SkyDreamer race is drawn over some of {sorted(TRACKS)}, not {self.tracks}")
        if self.latency < 0:
            raise ValueError("a command's latency is at least none")

    @functools.cached_property
    def racing(self) -> Racing:
        """The gate race's settings this race shares (`dew.rl.flight.Racing`): the camera's (its fixed tilt
        and this field of view, its image's randomization), the rate limit and its spread, the gates'
        opening and frame."""
        return Racing(fov=self.fov)

    @functools.cached_property
    def layout(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Each track's gates `[T, G, ...]`, a shorter track's padded by repeating its first: their centres
        `[T, G, 3]`, headings `[T, G]` (radians) and whether each is seen `[T, G]`, and each track's gate
        count `[T]`."""
        tracks = [TRACKS[name] for name in self.tracks]
        width = max(len(track) for track in tracks)
        padded = [track + (track[0],) * (width - len(track)) for track in tracks]
        centers = np.asarray([[center for center, _, _ in track] for track in padded])
        headings = np.deg2rad(np.asarray([[heading for _, heading, _ in track] for track in padded]))
        seen = np.asarray([[gate[2] and index < len(track) for index, gate in enumerate(padded_track)]
                           for track, padded_track in zip(tracks, padded, strict=True)])
        return centers, headings, seen, np.asarray([len(track) for track in tracks])

    def delays(self, ramp: float) -> tuple[int, int, int]:
        """The latency's range in whole control steps at `ramp`, its nominal spread half its size either way
        at a ramp of 1, and the longest at any ramp, which the commands pending hold."""
        nominal = self.latency * self.hz
        return (round(nominal * (1 - ramp / 2)), round(nominal * (1 + ramp / 2)), round(nominal * 1.5))


@struct.dataclass
class Lap:
    """One drone's episode: its drawn `body`, `flight` and `camera`, its rate limit and command latency (whole
    steps) with the commands `pending` (`dew.rl.flight.delayed`), the track it races (into `Sky.tracks`), the
    gate it heads for (`index`) and the plane of it it crosses next (`stage`, into `PLANES`), its distance to
    that gate's pre-gate after the last step, the gates passed, the steps taken, and the key its rotors'
    noise is drawn from."""

    body: Body
    flight: Flight
    camera: Camera
    rate_limit: jax.Array
    delay: jax.Array
    pending: jax.Array
    track: jax.Array
    index: jax.Array
    stage: jax.Array
    distance: jax.Array
    passed: jax.Array
    steps: jax.Array
    key: jax.Array


def gate_of(sky: Sky, track: jax.Array, index: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Gate `index` of track `track`: its centre and heading."""
    centers, headings, _, _ = sky.layout
    return jnp.asarray(centers)[track, index], jnp.asarray(headings)[track, index]


def in_gate(sky: Sky, track: jax.Array, index: jax.Array, point: jax.Array) -> jax.Array:
    """`point` in gate `index`'s frame: along its normal, to its left and up from its centre."""
    center, heading = gate_of(sky, track, index)
    normal, left = gate_axes(heading)
    offset = point - center
    return jnp.stack([offset @ normal, offset @ left, offset[2]])


def pre_gate(sky: Sky, track: jax.Array, index: jax.Array) -> jax.Array:
    """Gate `index`'s pre-gate's centre."""
    center, heading = gate_of(sky, track, index)
    normal, _ = gate_axes(heading)
    return center - sky.depth / 2 * normal


def plan(sky: Sky, track: jax.Array, index: jax.Array) -> jax.Array:
    """The flight plan `[24]` at gate `index` (Eq 12): for it and the two after it, the change in position
    and heading from the gate before (a half turn either way), then their positions and headings."""
    gates = (index + jnp.arange(-1, 3)) % jnp.asarray(sky.layout[3])[track]
    centers, headings = gate_of(sky, track, gates)
    change = headings[1:] - headings[:-1]
    turned = jnp.arctan2(jnp.sin(change), jnp.cos(change))
    changes = jnp.concatenate([centers[1:] - centers[:-1], turned[:, None]], -1).reshape(-1)
    return jnp.concatenate([changes, jnp.concatenate([centers[1:], headings[1:, None]], -1).reshape(-1)])


def struck(sky: Sky, track: jax.Array, before: jax.Array, after: jax.Array, passing: jax.Array) -> jax.Array:
    """Whether the step from `before` to `after` crosses the frame of a seen gate of `track` other than those
    `passing` `[G]` (the gates it heads for, whose planes decide alone): crosses its plane between its opening
    and its frame's outer edge (`Racing`'s opening and frame)."""
    centers, headings, seen, _ = sky.layout
    centers, headings = jnp.asarray(centers)[track], jnp.asarray(headings)[track]
    seen = jnp.asarray(seen)[track]
    normal, left = gate_axes(headings)
    start, end = jnp.sum((before - centers) * normal, -1), jnp.sum((after - centers) * normal, -1)
    crossed = (start * end <= 0) & (start != end)
    where = before + (start / jnp.where(crossed, start - end, 1.0))[:, None] * (after - before) - centers
    across = jnp.maximum(jnp.abs(jnp.sum(where * left, -1)), jnp.abs(where[:, 2]))
    half = sky.racing.opening / 2
    framed = (across > half) & (across <= half + sky.racing.frame)
    return jnp.any(seen & ~passing & crossed & framed)


def heading_of(orientation: jax.Array) -> jax.Array:
    """An orientation's yaw (its yaw, pitch, roll Euler angles')."""
    w, x, y, z = orientation
    return jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def roll_pitch(orientation: jax.Array) -> tuple[jax.Array, jax.Array]:
    """An orientation's roll and pitch (its yaw, pitch, roll Euler angles')."""
    w, x, y, z = orientation
    roll = jnp.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return roll, jnp.arcsin(jnp.clip(2 * (w * y - z * x), -1.0, 1.0))


@struct.dataclass
class SkyRace:
    """One drone racing `sky`'s tracks as a `dew.rl.env.Env` at randomization `ramp` (the module docstring),
    one level for every track or each of `sky.tracks`' own in turn, an episode flown at its track's:
    actions `[4]` in [-1, 1]; observations a dict of the `views` asked for, each but the image (`VIEWS`,
    which `SkyDreamerRace` adds). It cuts its own episodes (`Sky.steps`); a curriculum changes its `ramp`,
    its programs compiled again."""

    sky: Sky = struct.field(pytree_node=False, default_factory=Sky)
    views: tuple[str, ...] = struct.field(pytree_node=False, default=VIEWS[1:])
    ramp: float | tuple[float, ...] = struct.field(pytree_node=False, default=0.0)

    action_size = 4

    def __post_init__(self):
        shown(self.views, VIEWS[1:], "a SkyDreamer race")

    def observed(self, lap: Lap) -> dict[str, jax.Array]:
        """The views of `lap` asked for."""
        sky, flight = self.sky, lap.flight
        _, heading = gate_of(sky, lap.track, lap.index)
        normal, left = gate_axes(heading)
        up = jnp.asarray([0.0, 0.0, 1.0])
        in_its = jnp.stack([flight.velocity @ axis for axis in (normal, left, up)])
        roll, pitch = roll_pitch(flight.orientation)
        yaw = heading_of(flight.orientation)
        from_gate = yaw - heading
        angles = jnp.stack([roll, pitch, yaw, jnp.arctan2(jnp.sin(from_gate), jnp.cos(from_gate))])
        state = jnp.concatenate([flight.position, in_gate(sky, lap.track, lap.index, flight.position),
                                 flight.velocity, in_its, angles])
        motors = flight.speed / sky.quadrotor.top_speed
        # Past the gate's plane, the plan shows the gates after it.
        planned = (lap.index + (lap.stage == len(PLANES) - 1)) % jnp.asarray(sky.layout[3])[lap.track]
        shown = {"rates": flight.angular_velocity, "motors": motors, "plan": plan(sky, lap.track, planned),
                 "state": state, "passed": lap.passed}
        return {view: shown[view] for view in self.views}

    def scene(self, lap: Lap) -> Scene:
        """What a renderer draws of `lap`: its track's seen gates, none lit, no pillars, in its camera's
        looks."""
        centers, headings, seen, _ = self.sky.layout
        drawn = [np.flatnonzero(row) for row in seen]
        center = np.asarray([np.resize(centers[track][gates], (GATES, 3))
                             for track, gates in enumerate(drawn)])
        heading = np.asarray([np.resize(headings[track][gates], GATES) for track, gates in enumerate(drawn)])
        count = jnp.asarray([len(gates) for gates in drawn])[lap.track]
        return Scene(jnp.asarray(center)[lap.track], jnp.asarray(heading)[lap.track], count, jnp.asarray(-1),
                     jnp.zeros((OBSTACLES, 4)), lap.camera.look)

    @property
    def levels(self) -> np.ndarray:
        """Each track's randomization, `[len(sky.tracks)]`, in `sky.tracks`' order."""
        return np.broadcast_to(np.asarray(self.ramp, float), (len(self.sky.tracks),))

    def reset(self, key: jax.Array) -> tuple[Lap, TimeStep]:
        """A start of Table III's (the module docstring) at its track's level of the race's `ramp`."""
        sky, racing = self.sky, self.sky.racing
        keys = iter(jax.random.split(key, 14))
        track = jax.random.randint(next(keys), (), 0, len(sky.tracks))
        ramp = jnp.asarray(self.levels)[track]
        anywhere = jax.random.uniform(next(keys)) < sky.anywhere
        gates = jnp.asarray(sky.layout[3])[track]
        index = jnp.where(anywhere, jax.random.randint(next(keys), (), 0, gates), 0)
        low = jnp.where(anywhere, 0.0, 0.7)
        along = jax.random.uniform(next(keys), (), minval=-4.0, maxval=-2.0)
        aside = jax.random.uniform(next(keys), (), minval=-1.0, maxval=1.0)
        below = jax.random.uniform(next(keys), (), minval=low, maxval=1.3)
        center, heading = gate_of(sky, track, index)
        normal, left = gate_axes(heading)
        position = center + along * normal + aside * left - below * jnp.asarray([0.0, 0.0, 1.0])
        turned = jax.random.uniform(next(keys), (3,), minval=-math.pi / 9, maxval=math.pi / 9)
        orientation = quaternion(turned[0], turned[1], heading + turned[2])
        body = drawn_body(sky.quadrotor, next(keys), ramp)
        rates = jax.random.uniform(next(keys), (3,), minval=-0.1, maxval=0.1)
        speeds = sky.quadrotor.top_speed * jax.random.uniform(next(keys), (4,), minval=0.25, maxval=0.5)
        flight = dataclasses.replace(hovering(sky.quadrotor, body, position, orientation), speed=speeds,
                                     angular_velocity=rates, measured=rates)
        rate_limit = racing.rate_limit * (1 + ramp * racing.rate_spread
                                          * jax.random.uniform(next(keys), (), minval=-1.0, maxval=1.0))
        # Each track's latency range at its level, and the longest at any level, which every lap holds.
        shortest, longest, held = np.asarray([sky.delays(float(level)) for level in self.levels]).T
        # Until its first command arrives, the drone holds its hover.
        hover = jnp.concatenate([(body.mass * GRAVITY)[None], jnp.zeros(3)])
        lap = Lap(body=body, flight=flight, camera=drawn_camera(racing, next(keys), ramp),
                  rate_limit=rate_limit,
                  delay=jax.random.randint(next(keys), (), jnp.asarray(shortest)[track],
                                           jnp.asarray(longest)[track] + 1),
                  pending=jnp.tile(hover, (int(held[0]) + 1, 1)), track=track, index=index,
                  stage=jnp.zeros((), jnp.int32),
                  distance=jnp.linalg.norm(position - pre_gate(sky, track, index)),
                  passed=jnp.zeros((), jnp.int32), steps=jnp.zeros((), jnp.int32), key=next(keys))
        done = jnp.zeros((), bool)
        return lap, TimeStep(self.observed(lap), jnp.zeros(()), done, done)

    def step(self, lap: Lap, action: jax.Array) -> tuple[Lap, TimeStep]:
        """The drone flown a control step, then the planes it crossed in order, the frames it struck, the
        reward and whether the episode ended (the module docstring)."""
        sky = self.sky
        before = lap.flight.position
        lap, collective, rates = commanded(sky.quadrotor, lap, action)
        key, noise = jax.random.split(lap.key)
        flight, _ = flown(sky.quadrotor, lap.body, lap.flight, collective, rates, noise, sky.hz,
                          jnp.asarray(self.levels)[lap.track])
        after = flight.position
        track, index, stage, approaching = lap.track, lap.index, lap.stage, lap.stage == 0
        crossing, collided = jnp.ones((), bool), jnp.zeros((), bool)
        gate, passed, reached = jnp.zeros(()), lap.passed, after
        for _ in PLANES:
            # The next plane: crossed this step where the step's path meets it from behind, inside the gate's
            # size to go on; outside it, the episode ends there.
            plane = sky.depth / 2 * (stage - 1)
            start, end = in_gate(sky, track, index, before)[0], in_gate(sky, track, index, after)[0]
            crossed = crossing & (start < plane) & (end >= plane)
            where = before + (plane - start) / jnp.where(crossed, end - start, 1.0) * (after - before)
            local = in_gate(sky, track, index, where)
            off = jnp.maximum(jnp.abs(local[1]), jnp.abs(local[2])) / sky.size
            inside = crossed & (off <= 1)
            collided = collided | (crossed & (off > 1))
            reached = jnp.where(crossed & (stage == 0), where, reached)
            gate = gate + jnp.where(inside, 1 - off, 0.0)
            through = inside & (stage == len(PLANES) - 1)
            passed = passed + through.astype(passed.dtype)
            index = jnp.where(through, (index + 1) % jnp.asarray(sky.layout[3])[track], index)
            stage = jnp.where(inside, (stage + 1) % len(PLANES), stage)
            crossing = inside
        slots = jnp.arange(sky.layout[0].shape[1])
        if sky.frames:
            collided = collided | struck(sky, track, before, after, (slots == lap.index) | (slots == index))
        # Progress toward the pre-gate headed for, up to where it is crossed; none from it to the post-gate.
        approached = lap.distance - jnp.linalg.norm(reached - pre_gate(sky, track, lap.index))
        progress = jnp.where(approaching, approached, 0.0)
        rate = jnp.expm1(jnp.minimum(jnp.sum(jnp.abs(flight.angular_velocity)), 17.0)) / (2 * sky.hz * 1e5)
        roll, pitch = roll_pitch(flight.orientation)
        grounded = (after[2] < 0.5) & ((flight.velocity[2] < -1.0) | (jnp.abs(roll) > math.pi / 3)
                                       | (jnp.abs(pitch) > math.pi / 3))
        ended = collided | grounded
        reward = jnp.where(ended, 0.0, 5 * progress - rate + 30 * gate)
        distance = jnp.linalg.norm(after - pre_gate(sky, track, index))
        lap = dataclasses.replace(lap, flight=flight, index=index, stage=stage, distance=distance,
                                  passed=passed, steps=lap.steps + 1, key=key)
        return lap, TimeStep(self.observed(lap), reward, ended, ~ended & (lap.steps >= sky.steps))


class SkyDreamerRace(Seen):
    """A batch of `renderer.count` drones racing `sky`'s tracks (`SkyRace`, stepped on device), seen through
    `renderer`'s cameras (`dew.rl.flight.Seen`), as a `dew.rl.simulator.Simulator`: observations a dict of
    the `views` asked for (`VIEWS`), each `[B, ...]`. An episode is flown at its track's level of the race's
    `ramp` (`SkyRace`), which a race starts at and which may change between steps. The renderer's gates
    must be the race's, and a renderer's fisheye lens (`dew.rl.hyperlux.HyperLuxRenderer`) the cameras'
    field of view."""

    def __init__(self, renderer: Renderer | None, sky: Sky | None = None, views: tuple[str, ...] = VIEWS,
                 count: int | None = None, ramp: float | tuple[float, ...] = 0.0):
        from dew.rl.hyperlux import HyperLuxRenderer

        self.sky, self.ramp = (Sky() if sky is None else sky), ramp
        racing = self.sky.racing
        if renderer is not None and tuple(renderer.gate) != (racing.opening, racing.frame, racing.depth):
            raise ValueError(f"the renderer draws gates of opening, frame and depth {tuple(renderer.gate)}, "
                             f"the race's are {(racing.opening, racing.frame, racing.depth)}")
        if isinstance(renderer, HyperLuxRenderer) and renderer.fisheye not in (None, self.sky.fov):
            raise ValueError(f"the renderer's fisheye lens is {renderer.fisheye} degrees across, the "
                             f"cameras' {self.sky.fov}: it would sample a pinhole of another field of view")
        race = SkyRace(self.sky, Seen.privileged(views))
        super().__init__(renderer, race, views, VIEWS, "a SkyDreamer race", count)

    def _env_now(self) -> SkyRace:
        # Held as plain floats: the race's programs are keyed on it.
        return dataclasses.replace(self._env, ramp=tuple(float(level) for level in np.ravel(self.ramp)))


__all__ = ["CIRCLE", "INVERTED_LOOP", "LADDER", "PLANES", "QUADCOPTER", "TRACKS", "VIEWS", "Gate", "Lap",
           "Sky", "SkyDreamerRace", "SkyRace", "gate_of", "heading_of", "in_gate", "plan", "pre_gate",
           "roll_pitch", "struck"]
