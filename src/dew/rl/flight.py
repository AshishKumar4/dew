"""First-person gate racing for a batch of quadrotors, any flight simulator behind it.

The task is FlyDreamer's (AshishKumar4/FlyDreamer at 043c9ef382048278ccf26849e54814ef5510a737,
embodied/envs/fpv), its values its run's (dreamerv3/configs.yaml `fpv`): a course of waypoints the drone
must reach in order, each within `gate_radius`, paid by its navigation reward (rewards/navigation.py)
and a bonus at each one reached, commanded in collective throttle and body rates (env.py
`_action_to_command`). The policy sees the camera's image alone; the drone's `state` and its `goal`, the
waypoint it heads for, are privileged (env.py `_compute_privileged_obs`), for a world model's auxiliary
heads and a teacher. Each episode's course is drawn afresh from its own key (FlyDreamer reseeded the
same course every reset).

Where FlyDreamer marked its waypoints with balloons in a scene that detected its own collisions, each
waypoint here stands in a square gate facing along the course, solid as the ground is: the drone's path
through each physics step is tested against both (`struck`), so a fast drone cannot pass through a
frame between two steps.

The world frame is x forward, y left, z up, the body frame forward, left, up: FlyDreamer's NED and FRD
frames turned half a turn about x, under which each term of its reward is unchanged (each reads a
distance, an absolute offset, or squared or absolute body rates), its course's offsets are drawn alike,
and its course (`FLYDREAMER_COURSE`) is (x, -y, -z).

A simulator supplies the physics and the camera (`FlightBackend`); `GateRace` makes it a
`dew.rl.simulator.Simulator`, the course, reward and collisions computed in JAX.
"""

from __future__ import annotations

import dataclasses
from functools import partial
from typing import Any, Protocol

import jax
import jax.numpy as jnp
from flax import struct

from dew.rl.env import TimeStep
from dew.rl.simulator import chosen, received

FLYDREAMER_COURSE = ((8.0, 0.0, 3.0), (18.0, 0.0, 5.0), (30.0, 0.0, 5.0), (42.0, -3.0, 5.0),
                     (52.0, -8.0, 5.0), (58.0, -15.0, 5.0), (58.0, -30.0, 5.0), (58.0, -44.0, 5.0),
                     (52.0, -54.0, 5.0), (42.0, -60.0, 5.0), (28.0, -60.0, 5.0), (14.0, -60.0, 5.0))
"""FlyDreamer's twelve waypoints in metres (configs.yaml `static_gates`), turned from NED: (x, -y, -z)."""


@dataclasses.dataclass(frozen=True)
class Racing:
    """The task, FlyDreamer's run's (configs.yaml `fpv`). An episode's course is drawn about `course`'s
    waypoints (waypoints/base.py `_apply_randomization`): each offset by a momentum-smoothed draw
    (`smoothing`) of a point uniform in radius to `xy_radius` and in angle, and a height uniform in
    `z_range`; then its height held to [`min_height`, `max_height`] and the waypoint pushed out to
    `spacing` from the one before where nearer. The reward's terms are `raced`'s; an action maps to a
    command by `commands`, `hover_throttle` the share of the drone's full thrust that holds it up; an
    episode starts at `start`, heading `start_yaw`, and is cut after `max_steps` steps.

    Each waypoint stands in a square gate, its opening `gate_opening` wide in a frame `gate_frame` wide,
    facing along the course; with `solid`, a drone of `radius` that crosses a frame or sinks to the
    ground collides (`struck`), and a backend that detects its own collisions passes False."""

    course: tuple[tuple[float, float, float], ...] = FLYDREAMER_COURSE
    xy_radius: float = 5.0
    z_range: tuple[float, float] = (-2.0, 2.0)
    min_height: float = 3.0
    max_height: float = 10.0
    spacing: float = 8.0
    smoothing: float = 0.6
    loop_start: int = 0
    gate_radius: float = 1.6
    gate_bonus: float = 60.0
    gate_size: float = 1.6
    gate_bonus_max_scale: float = 3.0
    collision_penalty: float = -25.0
    progress_scale: float = 5.0
    progress_clip: tuple[float, float] = (-1.0, 1.0)
    survival_bonus: float = 0.1
    survival_decay: float = 200.0
    time_penalty: float = -0.002
    time_penalty_growth: float = 300.0
    rate_penalty_scale: float = 0.3
    rate_yaw_weight: float = 1.0
    jerk_penalty_scale: float = 0.001
    visibility_penalty: float = 0.5
    camera_fov: float = 100.0
    max_steps: int = 2000
    hover_throttle: float = 0.35
    effective_range: float = 0.5
    rate_scales: tuple[float, float, float] = (2.0, 2.0, 2.0)
    start: tuple[float, float, float] = (0.0, 0.0, 1.0)
    start_yaw: float = 0.0
    gate_opening: float = 1.6
    gate_frame: float = 0.25
    radius: float = 0.2
    solid: bool = True

    @property
    def rate_limit(self) -> float:
        """The body rate a full action commands, in radians a second."""
        return max(self.rate_scales) / self.effective_range


@struct.dataclass
class Kinematics:
    """A batch of drones' states: `position` `[B, 3]` in metres and `velocity` `[B, 3]` in metres a second
    in the world frame, `orientation` `[B, 4]` the body-to-world rotation as a unit quaternion (w, x, y,
    z), and `angular_velocity` `[B, 3]` the body rates in radians a second."""

    position: jax.Array
    velocity: jax.Array
    orientation: jax.Array
    angular_velocity: jax.Array


@struct.dataclass
class Track:
    """Each drone's progress on its course `[B, G, 3]`: the waypoint it heads for, its distance from it
    after the last step (the progress reward's reference), the body rates then and whether there were
    any (the jerk penalty's), and the steps taken."""

    course: jax.Array
    index: jax.Array
    distance: jax.Array
    rates: jax.Array
    rated: jax.Array
    steps: jax.Array


@struct.dataclass
class Gates:
    """Each drone's gates as a renderer shows them: their centres `[B, G, 3]`, the yaw of each one's facing
    `[B, G]`, and the gate each drone heads for `[B]`."""

    center: jax.Array
    yaw: jax.Array
    target: jax.Array


def placed(racing: Racing, offsets: jax.Array) -> jax.Array:
    """A course `[G, 3]` about `racing.course` from each waypoint's drawn offset `[G, 3]`, as FlyDreamer
    places it (waypoints/base.py:66-108)."""

    def place(carry, inputs):
        momentum, previous = carry
        waypoint, offset, following = inputs
        momentum = racing.smoothing * momentum + (1 - racing.smoothing) * offset
        point = waypoint + momentum
        point = point.at[2].set(jnp.clip(point[2], racing.min_height, racing.max_height))
        away = point - previous
        gap = jnp.linalg.norm(away)
        pushed = previous + away / jnp.where(gap > 1e-6, gap, 1.0) * racing.spacing
        point = jnp.where(following & (gap < racing.spacing) & (gap > 1e-6), pushed, point)
        return (momentum, point), point

    course = jnp.asarray(racing.course, offsets.dtype)
    following = jnp.arange(course.shape[0]) > 0
    return jax.lax.scan(place, (jnp.zeros_like(offsets[0]), course[0]), (course, offsets, following))[1]


def drawn_course(racing: Racing, key: jax.Array) -> jax.Array:
    """An episode's course `[G, 3]` drawn from `key` (`placed`)."""
    count = len(racing.course)
    angle_key, radius_key, height_key = jax.random.split(key, 3)
    angle = jax.random.uniform(angle_key, (count,), maxval=2 * jnp.pi)
    radius = jax.random.uniform(radius_key, (count,), maxval=racing.xy_radius)
    height = jax.random.uniform(height_key, (count,), minval=racing.z_range[0], maxval=racing.z_range[1])
    return placed(racing, jnp.stack([radius * jnp.cos(angle), radius * jnp.sin(angle), height], -1))


def _following(racing: Racing, index: jax.Array, count: int) -> jax.Array:
    """The waypoint after each `index`, the course looping to `loop_start` (waypoints/base.py `advance`)."""
    return jnp.where(index + 1 == count, racing.loop_start, index + 1)


def gate_yaws(racing: Racing, courses: jax.Array) -> jax.Array:
    """Each gate's facing `[B, G]`: the yaw of the level line from the point before it on the course (the
    start, for the first) to the one after it (the course looping to `loop_start`)."""
    count = courses.shape[1]
    start = jnp.broadcast_to(jnp.asarray(racing.start, courses.dtype), courses[:, :1].shape)
    before = jnp.concatenate([start, courses[:, :-1]], 1)
    after = courses[:, _following(racing, jnp.arange(count), count)]
    along = after - before
    return jnp.arctan2(along[..., 1], along[..., 0])


def struck(racing: Racing, courses: jax.Array, path: jax.Array) -> jax.Array:
    """Whether each drone's path through a step `[B, S + 1, 3]`, its positions at each physics step's end
    and the one before, struck the ground or a gate's frame `[B]`: a position within `radius` of the
    ground, or a segment crossing a gate's plane where the drone, `radius` about the crossing, overlaps the
    frame. A segment is the straight line between two physics steps; the ground being flat, its ends
    alone can come nearest it."""
    grounded = jnp.any(path[:, 1:, 2] < racing.radius, -1)
    yaw = gate_yaws(racing, courses)[:, None]
    normal = jnp.stack([jnp.cos(yaw), jnp.sin(yaw), jnp.zeros_like(yaw)], -1)
    lateral = jnp.stack([-jnp.sin(yaw), jnp.cos(yaw)], -1)
    begin, end = path[:, :-1, None], path[:, 1:, None]
    center = courses[:, None]
    before, after = (jnp.sum((point - center) * normal, -1) for point in (begin, end))
    crossing = (before * after <= 0) & (before != after)
    share = before / jnp.where(before != after, before - after, 1.0)
    offset = begin + share[..., None] * (end - begin) - center
    across = jnp.maximum(jnp.abs(jnp.sum(offset[..., :2] * lateral, -1)), jnp.abs(offset[..., 2]))
    half = racing.gate_opening / 2
    framed = (across >= half - racing.radius) & (across <= half + racing.gate_frame + racing.radius)
    return grounded | jnp.any(crossing & framed, (1, 2))


def commands(racing: Racing, action: jax.Array) -> jax.Array:
    """Actions `[B, 4]` in [-1, 1] as commands `[B, 4]` (env.py `_action_to_command`): the collective
    throttle in [0, 1], the hover throttle at zero, then the roll, pitch and yaw rates in radians a
    second, each `rate_scales` at `effective_range` of the action and twice that at its end."""
    action = jnp.clip(action, -1, 1) / racing.effective_range
    hover = racing.hover_throttle
    throttle = jnp.clip(hover + action[:, 0] * jnp.where(action[:, 0] >= 0, 1 - hover, hover), 0, 1)
    rates = action[:, 1:] * jnp.asarray(racing.rate_scales, action.dtype)
    return jnp.concatenate([throttle[:, None], rates], -1)


def rotation(orientation: jax.Array) -> jax.Array:
    """The body-to-world rotation matrices `[..., 3, 3]` of unit quaternions (w, x, y, z)."""
    w, x, y, z = jnp.moveaxis(orientation, -1, 0)
    return jnp.stack([
        jnp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        jnp.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        jnp.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def started(kinematics: Kinematics, courses: jax.Array) -> Track:
    """Each drone's track as an episode starts on its course `[B, G, 3]` (env.py `_reset`)."""
    count = courses.shape[0]
    distance = jnp.linalg.norm(courses[:, 0] - kinematics.position, axis=-1)
    return Track(courses, jnp.zeros(count, jnp.int32), distance, jnp.zeros_like(kinematics.angular_velocity),
                 jnp.zeros(count, bool), jnp.zeros(count, jnp.int32))


def raced(racing: Racing, track: Track, kinematics: Kinematics,
          collided: jax.Array) -> tuple[Track, jax.Array, jax.Array, jax.Array]:
    """A step's track, reward, and whether its episode `terminated` or was `truncated`, from the drones'
    states after it and whether each `collided` (env.py `step`, rewards/navigation.py). A drone nearer
    its waypoint than `gate_radius` reaches it: a bonus of `gate_bonus`, shrunk as the larger of its
    lateral and vertical offsets from the waypoint nears `gate_size` and grown toward
    `gate_bonus_max_scale` times along the course; it heads for the next, its progress reference that
    one's distance. Then the reward toward the waypoint headed for: a decaying survival bonus, the
    progress made times `progress_scale` clipped, a log penalty on the body rates eased while
    progressing, a penalty on their change, a growing time penalty, and a penalty while the waypoint is
    out of the camera's view; a collision pays `collision_penalty` in their place and ends the episode,
    and an episode is cut after `max_steps` steps."""
    rows = jnp.arange(track.index.shape[0])
    count = track.course.shape[1]
    position, rates = kinematics.position, kinematics.angular_velocity
    offset = position - track.course[rows, track.index]
    reached = jnp.linalg.norm(offset, axis=-1) < racing.gate_radius
    widest = jnp.maximum(jnp.abs(offset[:, 1]), jnp.abs(offset[:, 2]))
    decay = jnp.maximum(0.0, 1.0 - widest / racing.gate_size)
    along = track.index.astype(position.dtype) / max(count - 1, 1)
    grown = 1 + (racing.gate_bonus_max_scale - 1) * along ** 2
    bonus = jnp.where(reached, racing.gate_bonus * decay * grown, 0.0)
    index = jnp.where(reached, _following(racing, track.index, count), track.index)
    to_go = track.course[rows, index] - position
    distance = jnp.linalg.norm(to_go, axis=-1)
    progress = jnp.where(reached, distance, track.distance) - distance

    steps = track.steps.astype(distance.dtype)
    survival = racing.survival_bonus * jnp.exp(-steps / racing.survival_decay)
    gained = jnp.clip(progress * racing.progress_scale, *racing.progress_clip)
    weighted = jnp.abs(rates[:, 0]) + jnp.abs(rates[:, 1]) + jnp.abs(rates[:, 2]) * racing.rate_yaw_weight
    gate = jnp.maximum(0.0, 1.0 / (1.0 + jnp.exp(-5.0 * progress)) - 0.5) * 2.0
    rate = -racing.rate_penalty_scale * jnp.log(1.0 + weighted ** 2) * (1.0 - 0.5 * gate)
    change = jnp.sum(jnp.square(rates - track.rates), -1)
    jerk = jnp.where(track.rated, -racing.jerk_penalty_scale * change, 0.0)
    time = racing.time_penalty * (1 + steps / racing.time_penalty_growth)
    ahead = jnp.einsum("bji,bj->bi", rotation(kinematics.orientation), to_go)
    off_axis = jnp.arctan2(jnp.linalg.norm(ahead[:, 1:], axis=-1), ahead[:, 0])
    unseen = (ahead[:, 0] <= 0) | (off_axis > jnp.deg2rad(racing.camera_fov / 2))
    visibility = jnp.where(unseen, -racing.visibility_penalty, 0.0)
    reward = jnp.where(collided, racing.collision_penalty,
                       survival + gained + rate + jerk + time + visibility) + bonus

    taken = track.steps + 1
    following = Track(track.course, index, distance, rates, jnp.ones_like(track.rated), taken)
    return following, reward, collided, ~collided & (taken >= racing.max_steps)


def privileged(track: Track, kinematics: Kinematics) -> dict[str, jax.Array]:
    """FlyDreamer's privileged views (env.py `_compute_privileged_obs`): the `state` `[B, 13]`, position,
    velocity, orientation and body rates, and the `goal` `[B, 4]`, the direction to the waypoint headed
    for (zero within a micrometre of it) and its distance."""
    to_go = track.course[jnp.arange(track.index.shape[0]), track.index] - kinematics.position
    distance = jnp.linalg.norm(to_go, axis=-1, keepdims=True)
    direction = jnp.where(distance > 1e-6, to_go / jnp.where(distance > 1e-6, distance, 1.0), 0.0)
    state = jnp.concatenate([kinematics.position, kinematics.velocity, kinematics.orientation,
                             kinematics.angular_velocity], -1)
    return {"state": state, "goal": jnp.concatenate([direction, distance], -1)}


class FlightBackend(Protocol):
    """A flight simulator for a batch of `count` drones. `reset(mask, position, orientation)` puts the
    drones `mask` `[B]` selects at rest at their poses (`[B, 3]`, `[B, 4]`) and answers every drone's
    `Kinematics`; `step(commands)` advances every drone by a control step under `[B, 4]` commands
    (`commands`) and answers their `Kinematics`, their paths through it `[B, S + 1, 3]` (their positions
    before it and after each of its `S` physics steps), and whether each collided with what the backend
    itself detects (None for a backend that detects nothing); `render(gates)` answers each drone's
    camera image `[B, H, W, C]` uint8 over its `Gates`. Arrays cross by DLPack
    (`dew.rl.simulator.Arrays`)."""

    count: int

    def reset(self, mask: jax.Array, position: jax.Array, orientation: jax.Array) -> Kinematics: ...

    def step(self, commands: jax.Array) -> tuple[Kinematics, Any, Any]: ...

    def render(self, gates: Gates) -> Any: ...  # noqa: ANN401


VIEWS = ("image", "state", "goal")
"""What a gate race can show: the camera's image, and the privileged `state` and `goal`
(`privileged`)."""


class GateRace:
    """`backend` racing `racing`'s courses as a `dew.rl.simulator.Simulator`: actions `[B, 4]` in [-1, 1]
    (`commands`); observations a dict of the `views` asked for (`VIEWS`), each `[B, ...]`; an episode
    restarted at `racing.start` on a course drawn afresh."""

    action_size = 4

    def __init__(self, backend: FlightBackend, racing: Racing | None = None, views: tuple[str, ...] = VIEWS):
        if not views or not set(views) <= set(VIEWS):
            raise ValueError(f"a gate race shows some of {VIEWS}, not {views}")
        racing = Racing() if racing is None else racing
        self.backend, self.racing, self.views, self.count = backend, racing, views, backend.count
        self._courses = jax.jit(lambda key: jax.vmap(partial(drawn_course, racing))(
            jax.random.split(key, self.count)))
        self._commands = jax.jit(partial(commands, racing))
        self._struck = jax.jit(partial(struck, racing))
        self._raced = jax.jit(partial(raced, racing))
        self._started = jax.jit(started)
        self._privileged = jax.jit(privileged)
        self._gates = jax.jit(lambda track: Gates(track.course, gate_yaws(racing, track.course), track.index))
        yaw = racing.start_yaw / 2
        self._pose = (jnp.tile(jnp.asarray(racing.start), (self.count, 1)),
                      jnp.tile(jnp.asarray([jnp.cos(yaw), 0.0, 0.0, jnp.sin(yaw)]), (self.count, 1)))
        self._key = self._track = None

    def _observed(self, kinematics: Kinematics) -> dict[str, jax.Array]:
        shown = self._privileged(self._track, kinematics)
        if "image" in self.views:
            shown["image"] = received(self.backend.render(self._gates(self._track)))
        return {view: shown[view] for view in self.views}

    def _started_on(self, mask: jax.Array, courses: jax.Array) -> Kinematics:
        kinematics = received(self.backend.reset(mask, *self._pose))
        fresh = self._started(kinematics, courses)
        self._track = fresh if self._track is None else chosen(mask, fresh, self._track)
        return kinematics

    def reset(self, seed: int) -> dict[str, jax.Array]:
        self._key, key = jax.random.split(jax.random.key(seed))
        self._track = None
        return self._observed(self._started_on(jnp.ones(self.count, bool), self._courses(key)))

    def restart(self, mask: jax.Array) -> dict[str, jax.Array]:
        self._key, key = jax.random.split(self._key)
        mask = jnp.asarray(mask, bool)
        courses = jnp.where(mask[:, None, None], self._courses(key), self._track.course)
        return self._observed(self._started_on(mask, courses))

    def step(self, action: jax.Array) -> TimeStep:
        kinematics, path, collided = received(self.backend.step(self._commands(action)))
        collided = jnp.zeros(self.count, bool) if collided is None else collided.astype(bool)
        if self.racing.solid:
            collided = collided | self._struck(self._track.course, path)
        self._track, reward, terminated, truncated = self._raced(self._track, kinematics, collided)
        return TimeStep(self._observed(kinematics), reward, terminated, truncated)


__all__ = ["FLYDREAMER_COURSE", "VIEWS", "FlightBackend", "GateRace", "Gates", "Kinematics", "Racing",
           "Track", "commands", "drawn_course", "gate_yaws", "placed", "privileged", "raced", "rotation",
           "started", "struck"]
