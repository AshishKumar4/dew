"""First-person drone gate racing in JAX: one drone's race as a `dew.rl.env.Env` (`FlightEnv`), on device and
jittable whole, flown by `dew.rl.quadrotor`'s physics; and a batch of them seen through a renderer's camera
as a `dew.rl.simulator.Simulator` (`GateRace`).

The reward is the one drone-racing RL has converged on: each step, the progress made toward the next gate's
centre (Kaufmann et al., Nature 2023, Swift, Eq 8; Song et al., Sci. Robot. 2023, Eq 5; SkyDreamer, arXiv
2510.14783, Sec II-C), less a cost on the body rates (Song 2023, b = 0.01 a step at 100 Hz, here 1 a second)
and on the action's change (Swift's lambda5); a bonus on passing a gate, the more for passing it centred
(SkyDreamer's r_gate); and a crash's penalty, which ends the episode (Song 2023's -10). A gate counts only
when its plane is crossed through its opening in its direction; crossing the next gate's plane beside it
misses it, a crash (SkyDreamer; Ferede et al., arXiv 2504.21586). The ground, the gates' frames and the
obstacles are solid, the drone's path through each physics step tested against them (`struck`), so a fast
drone cannot pass through one between two steps. An open track ends, a success, on passing its last gate.

Each episode is flown on a course of its own (`drawn_course`), of a `family`:
- `layout`: a published racing layout (uzh-rpg/agile_flight, MIT: Split-S and its real-world variant,
  AlphaPilot; and FlyDreamer's loop), each gate moved by up to `jitter` (Song et al., IROS 2021, Sec IV-C);
- `loop`: a closed loop of gates about an ellipse;
- `chain`: Song et al. 2021's gate primitive chained into an open track, G_{j+1} = f(G_j, p, R) (Eq 6): each
  gate a relative turn, climb and distance from the last, with split-S (a gate below the last, facing back)
  and dive primitives; the paper gives no ranges, so these match its tracks' 110 to 150 m and climbs up to
  17 m (Fig 7), and its curriculum from near-straight (Sec III-C.3);
- `figure8`: a closed figure-eight whose lobes cross at different heights.
The `difficulty`, from 0 to 1, scales the jitter, turns, climbs and primitives, and the obstacles' number, a
curriculum (Song 2021 raised it on the crash rate). Obstacles are vertical pillars beside or on the racing
line.
An episode starts anywhere on its course: hovering before a random gate (Swift; SkyDreamer's Table III), or
already flying toward it from along the segment before it.

The policy sees the camera's image alone, the next gate lit (`highlight`: its colours and pattern drawn per
episode, as AI Grand Prix 2026 lights its gates; FlyDreamer highlights its target): which gate is next is in
the image. The state, the next gates' corners, the obstacles and the drone's drawn parameters are privileged,
for a world model's auxiliary heads (SkyDreamer's privileged decoding) and a teacher (Swift's and Song 2023's
observation with the parameters). An action keeps FlyDreamer's semantics (env.py `_action_to_command`): its
first entry is the throttle, zero the drone's hover thrust, -1 none and 1 the nominal drone's full thrust;
the rest the body rates over the drone's rate limit. A command reaches the flight controller after the drone's
latency (Swift measured 40 ms; SkyDreamer's 11 ms action delay).

The world frame is x forward, y left, z up (FLU), the body frame forward, left, up.
"""

from __future__ import annotations

import dataclasses
from functools import partial
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from dew.nn.precision import divided
from dew.rl.env import TimeStep, chosen
from dew.rl.quadrotor import GRAVITY, Body, Flight, Quadrotor, drawn_body, flown, hovering, rotation
from dew.rl.simulator import Handed, received

LAYOUTS = {
    "split_s": ((-0.9, -1.27, 3.48), (9.09, 6.26, 1.08), (9.27, -3.46, 1.17), (-4.5, -6.14, 3.4),
                (-4.48, -5.94, 1.05), (4.45, -0.8, 1.09), (-2.65, 6.51, 1.3)),
    "real_track": ((-0.6, -0.86, 3.48), (9.0, 6.45, 1.05), (8.85, -3.8, 1.05), (-4.3, -5.6, 3.3),
                   (-4.3, -5.6, 1.25), (4.5, -0.45, 1.05), (-1.95, 6.81, 1.05)),
    "alpha_pilot": ((17.25, 3.63, 6.31), (-10.23, 7.77, 2.02), (-10.05, 30.62, 2.9), (16.75, 39.57, 6.76),
                    (2.33, 27.87, 2.55), (2.2, 9.0, 1.99), (-7.31, -12.14, 3.23), (-7.61, -28.47, 2.64),
                    (-0.01, -33.91, 2.1), (5.87, -28.44, 2.57), (7.25, -11.79, 2.55)),
    "flydreamer": ((8.0, 0.0, 3.0), (18.0, 0.0, 5.0), (30.0, 0.0, 5.0), (42.0, -3.0, 5.0), (52.0, -8.0, 5.0),
                   (58.0, -15.0, 5.0), (58.0, -30.0, 5.0), (58.0, -44.0, 5.0), (52.0, -54.0, 5.0),
                   (42.0, -60.0, 5.0), (28.0, -60.0, 5.0), (14.0, -60.0, 5.0)),
}
"""Racing layouts' gate centres in metres, FLU: uzh-rpg/agile_flight at
03dc1179c011f414aba1264bb6db92679be66b4f (envsim/parameters/tracks/SplitS.yaml, real_track.yaml,
AlphaPilot.yaml, MIT) and FlyDreamer's loop (AshishKumar4/FlyDreamer dreamerv3/configs.yaml `static_gates`,
turned from NED). Each gate faces along the course (`facings`): the layouts' own gate rotations follow their
simulator's model's axes, which the files do not state."""
FAMILIES = ("layout", "loop", "chain", "figure8")
"""The course families (`drawn_course`)."""
GATES = 16
"""The most gates a course holds; a shorter one's last are inactive."""
OBSTACLES = 8
"""The most obstacles a course holds; an inactive one has no radius."""
ASPECTS = (6.0, 12.0, 24.0)
"""A pillar's height over its radius, one of these: a renderer draws each as its shape scaled."""
NEAREST = 3
"""The obstacles a teacher is shown, the nearest first."""


@dataclasses.dataclass(frozen=True)
class Racing:
    """The race. `families` weighs the course families drawn (`FAMILIES`), `layouts` names the layouts a
    `layout` course is drawn from, `jitter` the most a layout's gate moves across and up at full difficulty;
    `obstacles` the most obstacles a course holds at full difficulty and `on_line` the share placed on the
    racing line. The reward's weights are `raced`'s; `hz` the control rate, 60 Hz (a teacher at 60 Hz passed
    27% more gates than at 30 in the same time; Swift's and Song's run at 90 to 100 Hz); an episode is cut
    after `episode_seconds` (Swift's 1500 steps). Gates' openings are `opening` square (1.5 m, SkyDreamer's
    and Ferede's) in a frame `frame` wide and `depth` deep; the drone collides as a cube of half-width
    `radius` aligned with the gate it meets and as a sphere of `radius` with the ground and obstacles
    (`struck`). `rate_limit` is the nominal body-rate command's limit and `rate_spread` its randomization,
    `latency` the command latency's range in seconds; the camera's `uptilt` above the body's forward axis in
    degrees, its horizontal field of view `fov` and `fov_spread`, its mount's `mount_spread`; the image's
    `exposure`, `white_balance`, `gamma`, `noise` (read noise's std, of full scale) and `shutter` (seconds)
    ranges; `highlight` whether the next gate is lit; `flying_share` the share of starts already flying and
    `start_speed` their fastest; and the `quadrotor` flown."""

    families: tuple[tuple[str, float], ...] = (("loop", 0.4), ("chain", 0.6))
    layouts: tuple[str, ...] = tuple(LAYOUTS)
    jitter: tuple[float, float] = (1.5, 0.75)
    obstacles: int = OBSTACLES
    on_line: float = 0.3
    hz: float = 60.0
    episode_seconds: float = 25.0
    opening: float = 1.5
    frame: float = 0.25
    depth: float = 0.1
    radius: float = 0.2
    progress_scale: float = 1.0
    rate_cost: float = 1.0
    smoothness_cost: float = 0.005
    gate_bonus: float = 6.0
    crash_penalty: float = 10.0
    miss_radius: float = 3.0
    ceiling: float = 25.0
    rate_limit: float = 10.0
    rate_spread: float = 0.2
    latency: tuple[float, float] = (0.01, 0.06)
    uptilt: tuple[float, float] = (15.0, 35.0)
    fov: float = 100.0
    fov_spread: float = 5.0
    mount_spread: float = 5.0
    exposure: tuple[float, float] = (0.5, 1.5)
    white_balance: float = 0.1
    gamma: tuple[float, float] = (0.8, 1.25)
    noise: tuple[float, float] = (0.0, 0.03)
    shutter: tuple[float, float] = (0.002, 0.008)
    highlight: bool = True
    flying_share: float = 0.5
    start_speed: float = 8.0
    quadrotor: Quadrotor = dataclasses.field(default_factory=Quadrotor)

    def __post_init__(self):
        names = [name for name, _ in self.families]
        unknown = (set(names) - set(FAMILIES)) | (set(self.layouts) - set(LAYOUTS))
        if unknown or not names or min(weight for _, weight in self.families) <= 0:
            raise ValueError(f"a race draws families of {FAMILIES} (layouts of {sorted(LAYOUTS)}) at "
                             f"positive weights, not {self.families} ({self.layouts})")

    @property
    def dt(self) -> float:
        return 1.0 / self.hz

    @property
    def max_steps(self) -> int:
        """The steps an episode is cut after."""
        return round(self.episode_seconds * self.hz)

    @property
    def delays(self) -> tuple[int, int]:
        """The latency's range in whole control steps."""
        return round(self.latency[0] * self.hz), max(round(self.latency[1] * self.hz), 1)


@struct.dataclass
class Course:
    """A course: its gates' centres `[G, 3]` and facings `[G]` (the yaw of their passing direction), how many
    it holds, whether it is closed (a loop) or open (a track ending at its last gate), its `family`'s index in
    `FAMILIES`, and its obstacles `[O, 4]`, each a vertical pillar's centre's x and y, radius and height (an
    inactive one's radius 0)."""

    center: jax.Array
    facing: jax.Array
    count: jax.Array
    closed: jax.Array
    family: jax.Array
    obstacles: jax.Array


@struct.dataclass
class Camera:
    """A drone's camera for an episode: its uptilt and horizontal field of view (radians), its mount's roll
    and yaw, and its image's exposure, white balance `[3]`, gamma, read noise and shutter (seconds); and the
    looks its renderer draws (`look` `[4]`: the gates', the lit gate's, the ground's and the sky's, each an
    index the renderer takes modulo its variants)."""

    uptilt: jax.Array
    fov: jax.Array
    mount: jax.Array
    exposure: jax.Array
    white_balance: jax.Array
    gamma: jax.Array
    noise: jax.Array
    shutter: jax.Array
    look: jax.Array


@struct.dataclass
class Race:
    """One drone's race: its course, the gate it heads for and its distance from it after the last step, the
    steps taken and gates passed; the last action and the commands awaiting their latency `[L, 4]`
    (collective thrust in N and body rates in rad/s), its latency in control steps and its rate limit; its
    drawn `body` and its `flight`; its camera; the key its physics draws from; and the curriculum it was
    drawn at (`difficulty`, `ramp`)."""

    course: Course
    index: jax.Array
    distance: jax.Array
    steps: jax.Array
    passed: jax.Array
    action: jax.Array
    pending: jax.Array
    delay: jax.Array
    rate_limit: jax.Array
    body: Body
    flight: Flight
    camera: Camera
    key: jax.Array
    difficulty: jax.Array
    ramp: jax.Array


def facings(center: jax.Array, count: jax.Array, closed: jax.Array) -> jax.Array:
    """Each gate's facing `[G]`: the yaw of the level line from the gate before it to the gate after it, on a
    closed course of `count` gates wrapping round, on an open one the first gate's from itself and the last's
    to itself."""
    order = jnp.arange(center.shape[0])
    before = jnp.where(closed, (order - 1) % count, jnp.maximum(order - 1, 0))
    after = jnp.where(closed, (order + 1) % count, jnp.minimum(order + 1, count - 1))
    along = center[after] - center[before]
    return jnp.arctan2(along[:, 1], along[:, 0])


def _layout_courses() -> tuple[np.ndarray, np.ndarray]:
    """Every layout, centred on its gates' mean across, its last gates repeating its first up to `GATES`,
    and its gate count."""
    courses, counts = [], []
    for name in LAYOUTS:
        gates = np.asarray(LAYOUTS[name])
        gates = gates - np.r_[gates[:, :2].mean(0), 0.0]
        courses.append(np.concatenate([gates, np.repeat(gates[:1], GATES - len(gates), 0)]))
        counts.append(len(gates))
    return np.stack(courses), np.asarray(counts, np.int32)


def _layout(racing: Racing, key: jax.Array, difficulty: jax.Array) -> tuple[jax.Array, jax.Array]:
    """A published layout of `racing.layouts`, each gate moved by up to `jitter` times the difficulty and held
    at least 0.75 m high; its gates and count."""
    which_key, moved_key = jax.random.split(key)
    kept = np.asarray([name in racing.layouts for name in LAYOUTS])
    courses, counts = _layout_courses()
    which = jax.random.choice(which_key, len(LAYOUTS), p=jnp.asarray(kept / kept.sum()))
    reach = jnp.asarray([racing.jitter[0], racing.jitter[0], racing.jitter[1]])
    moved = jax.random.uniform(moved_key, (GATES, 3), minval=-1.0, maxval=1.0)
    center = jnp.asarray(courses)[which] + difficulty * moved * reach
    return center.at[:, 2].set(jnp.maximum(center[:, 2], 0.75)), jnp.asarray(counts)[which]


def _loop(key: jax.Array, difficulty: jax.Array) -> tuple[jax.Array, jax.Array]:
    """A closed loop of 6 to 12 gates about an ellipse of radii 12 to 30 m, each moved along it by up to a
    quarter of their spacing and across it by a tenth of its radius times the difficulty, 1.5 m high plus up
    to 4.5 m times the difficulty, turned and flown either way round; its gates and count."""
    keys = iter(jax.random.split(key, 7))
    count = jax.random.randint(next(keys), (), 6, 13)
    radii = jax.random.uniform(next(keys), (2,), minval=12.0, maxval=30.0)
    along = jax.random.uniform(next(keys), (GATES,), minval=-0.25, maxval=0.25)
    across = jax.random.uniform(next(keys), (GATES,), minval=-0.1, maxval=0.1)
    height = jax.random.uniform(next(keys), (GATES,), maxval=4.5)
    turn = jax.random.uniform(next(keys), (), maxval=2 * jnp.pi)
    order = jnp.arange(GATES)
    angle = (order + difficulty * along) * (2 * jnp.pi / count)
    angle = jnp.where(jax.random.bernoulli(next(keys)), angle, -angle) + turn
    spread = 1 + difficulty * across
    center = jnp.stack([radii[0] * spread * jnp.cos(angle), radii[1] * spread * jnp.sin(angle),
                        1.5 + difficulty * height], -1)
    return jnp.where((order < count)[:, None], center, center[0]), count


def _chain(key: jax.Array, difficulty: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Song et al. 2021's gate primitive chained (Eq 6): 8 to `GATES` gates, the first 1.5 to 4 m high, each
    next one a turn of up to 90 degrees, a climb or descent of up to 3 m and 7 to 16 m on from the last, all
    times the difficulty but the distance; at a share of the difficulty's quarter a split-S (the next gate
    2.5 to 4 m below the last, a metre behind it, facing back) and as often a dive or climb of 4 to 7 m.
    Heights stay between 1 and 15 m; where the chain crosses itself, a later gate within 3 m across and 2.5 m
    in height of an earlier one climbs 3 m above it, up to 20 m. Its gates, count and facings."""
    keys = iter(jax.random.split(key, 8))
    count = jax.random.randint(next(keys), (), 8, GATES + 1)
    turns = difficulty * jax.random.uniform(next(keys), (GATES,), minval=-1.0, maxval=1.0) * jnp.pi / 2
    climbs = difficulty * jax.random.uniform(next(keys), (GATES,), minval=-3.0, maxval=3.0)
    distances = jax.random.uniform(next(keys), (GATES,), minval=7.0, maxval=16.0)
    primitive = jax.random.uniform(next(keys), (GATES,))
    split = primitive < difficulty / 4
    dive = (primitive >= difficulty / 4) & (primitive < difficulty / 2)
    plunge = jax.random.uniform(next(keys), (GATES,), minval=2.5, maxval=4.0)
    steep = jax.random.uniform(next(keys), (GATES,), minval=4.0, maxval=7.0) * jnp.sign(climbs + 1e-6)
    first = jax.random.uniform(next(keys), (), minval=1.5, maxval=4.0)

    def placed(carry, step):
        position, heading = carry
        turn, climb, distance, is_split, is_dive, drop, deep = step
        # A split-S's gate stays at least a metre up, and its frame clear of the last's.
        is_split = is_split & (position[2] - drop >= 1.0)
        direction = heading + turn / 2
        ahead = position + distance * jnp.stack([jnp.cos(direction), jnp.sin(direction), 0.0])
        ahead = ahead.at[2].add(jnp.where(is_dive, deep, climb))
        below = position.at[2].add(-drop) - 1.0 * jnp.stack([jnp.cos(heading), jnp.sin(heading), 0.0])
        nxt = jnp.where(is_split, below, ahead)
        nxt = nxt.at[2].set(jnp.clip(nxt[2], 1.0, 15.0))
        heading = jnp.where(is_split, heading + jnp.pi, heading + turn)
        return (nxt, heading), (nxt, heading)

    start = jnp.stack([0.0, 0.0, first])
    _, (rest, headings) = jax.lax.scan(placed, (start, jnp.zeros(())),
                                       (turns, climbs, distances, split, dive, plunge, steep))
    center = jnp.concatenate([start[None], rest[:-1]])
    facing = jnp.concatenate([jnp.zeros(1), headings[:-1]])
    order = jnp.arange(GATES)

    def separated(center: jax.Array, gate: jax.Array) -> tuple[jax.Array, None]:
        # A gate within 3 m across and 2.5 m in height of an earlier one (where the chain crosses itself)
        # climbs 3 m above the highest of them; a split-S's pair stands 2.5 m apart in height or more.
        point = center[gate]
        across = jnp.linalg.norm(center[:, :2] - point[:2], axis=-1)
        crowded = (order < gate) & (across < 3.0) & (jnp.abs(center[:, 2] - point[2]) < 2.5)
        lifted = jnp.max(jnp.where(crowded, center[:, 2], -jnp.inf)) + 3.0
        return center.at[gate, 2].set(jnp.where(crowded.any(), jnp.minimum(lifted, 20.0), point[2])), None

    for _ in range(3):
        center, _ = jax.lax.scan(separated, center, order)
    center = center - jnp.concatenate([center[:, :2].mean(0), jnp.zeros(1)])
    return (jnp.where((order < count)[:, None], center, center[0]), count,
            jnp.where(order < count, facing, facing[0]))


def _figure8(key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """A closed figure-eight (Gerono's lemniscate) of 8 to 12 gates spanning 24 to 40 m by 12 to 20 m, its two
    lobes 2.5 to 4 m apart in height where they cross (their gates' frames clear of each other), turned; its
    gates and count."""
    keys = iter(jax.random.split(key, 5))
    count = jax.random.randint(next(keys), (), 8, 13)
    width = jax.random.uniform(next(keys), (), minval=12.0, maxval=20.0)
    depth = jax.random.uniform(next(keys), (), minval=6.0, maxval=10.0)
    lift = jax.random.uniform(next(keys), (), minval=2.5, maxval=4.0)
    turn = jax.random.uniform(next(keys), (), maxval=2 * jnp.pi)
    order = jnp.arange(GATES)
    t = (order + 0.5) * (2 * jnp.pi / count)
    x, y = width * jnp.cos(t), depth * jnp.sin(2 * t)
    z = 3.0 + lift * (0.5 + 0.5 * jnp.sin(t))
    c, s = jnp.cos(turn), jnp.sin(turn)
    center = jnp.stack([c * x - s * y, s * x + c * y, z], -1)
    return jnp.where((order < count)[:, None], center, center[0]), count


def _obstacles(racing: Racing, key: jax.Array, center: jax.Array, count: jax.Array, closed: jax.Array,
               difficulty: jax.Array) -> jax.Array:
    """Up to `racing.obstacles` times the difficulty vertical pillars `[O, 4]` (x, y, radius, height): each
    at a random point 30 to 70% along a segment between consecutive gates, on the line at the share `on_line`
    and else 1.5 to 4 m beside it, 0.2 to 0.6 m in radius, its height its radius times one of `ASPECTS` (1.2
    to 14.4 m: a post a drone may clear, or a tower); one within 3 m of a gate is dropped (its radius 0)."""
    keys = iter(jax.random.split(key, 9))
    segments = jnp.where(closed, count, count - 1)
    segment = jax.random.randint(next(keys), (OBSTACLES,), 0, jnp.maximum(segments, 1))
    begin, end = center[segment], center[(segment + 1) % count]
    along = jax.random.uniform(next(keys), (OBSTACLES,), minval=0.3, maxval=0.7)
    point = begin + along[:, None] * (end - begin)
    heading = jnp.arctan2(end[:, 1] - begin[:, 1], end[:, 0] - begin[:, 0])
    on_line = jax.random.uniform(next(keys), (OBSTACLES,)) < racing.on_line
    aside = jnp.where(on_line, jax.random.uniform(next(keys), (OBSTACLES,), minval=-0.3, maxval=0.3),
                      jax.random.uniform(next(keys), (OBSTACLES,), minval=1.5, maxval=4.0)
                      * jnp.sign(jax.random.uniform(next(keys), (OBSTACLES,), minval=-1.0, maxval=1.0)))
    xy = point[:, :2] + aside[:, None] * jnp.stack([-jnp.sin(heading), jnp.cos(heading)], -1)
    radius = jax.random.uniform(next(keys), (OBSTACLES,), minval=0.2, maxval=0.6)
    height = radius * jnp.asarray(ASPECTS)[jax.random.randint(next(keys), (OBSTACLES,), 0, len(ASPECTS))]
    active = jnp.arange(OBSTACLES) < jnp.floor(difficulty * racing.obstacles
                                                * jax.random.uniform(next(keys), (), minval=0.25, maxval=1.0))
    gates = jnp.arange(GATES) < count
    apart = jnp.linalg.norm(xy[:, None] - center[None, :, :2], axis=-1)
    near = jnp.min(jnp.where(gates[None], apart, jnp.inf), -1)
    active = active & (near > 3.0 + radius) & (segments > 0)
    return jnp.concatenate([xy, jnp.where(active, radius, 0.0)[:, None], height[:, None]], -1)


def drawn_course(racing: Racing, key: jax.Array, difficulty: jax.Array) -> Course:
    """An episode's course (`Course`): its family drawn at `racing.families`' weights, then its gates
    (`_layout`, `_loop`, `_chain`, `_figure8`), facings and obstacles."""
    family_key, layout_key, loop_key, chain_key, eight_key, obstacle_key = jax.random.split(key, 6)
    weights = np.asarray([dict(racing.families).get(name, 0.0) for name in FAMILIES])
    family = jax.random.choice(family_key, len(FAMILIES), p=jnp.asarray(weights / weights.sum()))
    layout, layout_count = _layout(racing, layout_key, difficulty)
    loop, loop_count = _loop(loop_key, difficulty)
    chain, chain_count, chain_facing = _chain(chain_key, difficulty)
    eight, eight_count = _figure8(eight_key)
    center = jnp.stack([layout, loop, chain, eight])[family]
    count = jnp.stack([layout_count, loop_count, chain_count, eight_count])[family]
    closed = family != FAMILIES.index("chain")
    facing = jnp.where(closed, facings(center, count, closed), chain_facing)
    obstacles = _obstacles(racing, obstacle_key, center, count, closed, difficulty)
    return Course(center, facing, count, closed, family, obstacles)


def quaternion(roll: jax.Array, pitch: jax.Array, yaw: jax.Array) -> jax.Array:
    """Unit quaternions (w, x, y, z) of a yaw, then a pitch, then a roll."""
    cr, sr, cp, sp, cy, sy = (f(angle / 2) for angle in (roll, pitch, yaw) for f in (jnp.cos, jnp.sin))
    return jnp.stack([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                      cr * cp * sy - sr * sp * cy], -1)


def _gate_axes(facing: jax.Array) -> tuple[jax.Array, jax.Array]:
    """A gate's normal (its passing direction) and its left `[..., 3]`, level, from its facing."""
    zero = jnp.zeros_like(facing)
    normal = jnp.stack([jnp.cos(facing), jnp.sin(facing), zero], -1)
    return normal, jnp.stack([-jnp.sin(facing), jnp.cos(facing), zero], -1)


def drawn_camera(racing: Racing, key: jax.Array, ramp: jax.Array) -> Camera:
    """A drone's camera for an episode (`Camera`), each quantity uniform in its range, the image's spreads
    about neutral scaled by `ramp`."""
    keys = iter(jax.random.split(key, 10))

    def within(low: float, high: float, shape: tuple[int, ...] = ()) -> jax.Array:
        return jax.random.uniform(next(keys), shape, minval=low, maxval=high)

    def about(low: float, high: float, neutral: float) -> jax.Array:
        return neutral + ramp * (within(low, high) - neutral)

    return Camera(uptilt=jnp.deg2rad(within(*racing.uptilt)),
                  fov=jnp.deg2rad(within(racing.fov - racing.fov_spread, racing.fov + racing.fov_spread)),
                  mount=jnp.deg2rad(within(-racing.mount_spread, racing.mount_spread, (2,))),
                  exposure=about(*racing.exposure, neutral=1.0),
                  white_balance=1 + ramp * within(-racing.white_balance, racing.white_balance, (3,)),
                  gamma=about(*racing.gamma, neutral=1.0), noise=ramp * within(*racing.noise),
                  shutter=ramp * within(*racing.shutter),
                  look=jax.random.randint(next(keys), (4,), 0, 1 << 16))


def _start(racing: Racing, course: Course, key: jax.Array, difficulty: jax.Array) -> tuple[
        jax.Array, jax.Array, jax.Array, jax.Array]:
    """Where an episode starts on its course: the gate it heads for; its position, orientation and velocity.
    At `flying_share` it is already flying toward a random gate from 20 to 70% along the segment before it
    (before an open track's first gate, 6 to 14 m out along its facing), at up to `start_speed` times the
    difficulty; else it hovers 2 to 4 m before it, up to 1 m aside and 0.5 m up or down (Swift starts at a
    random gate; SkyDreamer's Table III). It faces its gate to within 20 degrees of yaw and 10 of roll and
    pitch, and starts no lower than 0.5 m."""
    gate_key, place_key, turn_key, flying_key, out_key, along_key, speed_key = jax.random.split(key, 7)
    last = jnp.where(course.closed, course.count, course.count - 1)
    index = jax.random.randint(gate_key, (), 0, jnp.maximum(last, 1))
    target = course.center[index]
    normal, left = _gate_axes(course.facing[index])
    back, aside, up = jax.random.uniform(place_key, (3,), minval=jnp.asarray([2.0, -1.0, -0.5]),
                                         maxval=jnp.asarray([4.0, 1.0, 0.5]))
    hovering_at = target - back * normal + aside * left + jnp.asarray([0.0, 0.0, 1.0]) * up
    previous = jnp.where(course.closed | (index > 0), course.center[(index - 1) % course.count],
                         target - jax.random.uniform(out_key, (), minval=6.0, maxval=14.0) * normal)
    share = jax.random.uniform(along_key, (), minval=0.2, maxval=0.7)
    flying_at = previous + share * (target - previous)
    flying = jax.random.uniform(flying_key, ()) < racing.flying_share
    position = jnp.where(flying, flying_at, hovering_at)
    position = position.at[2].set(jnp.maximum(position[2], 0.5))
    toward = target - position
    direction = toward / jnp.maximum(jnp.linalg.norm(toward), 1e-6)
    speed = jnp.where(flying, difficulty * racing.start_speed * jax.random.uniform(speed_key, ()), 0.0)
    roll, pitch, yaw = jnp.deg2rad(jax.random.uniform(turn_key, (3,), minval=-1.0, maxval=1.0)
                                   * jnp.asarray([10.0, 10.0, 20.0]))
    heading = jnp.where(flying, jnp.arctan2(direction[1], direction[0]), course.facing[index])
    return index, position, quaternion(roll, pitch, heading + yaw), speed * direction


def started(racing: Racing, key: jax.Array, difficulty: jax.Array, ramp: jax.Array) -> Race:
    """A drone's episode (`Race`) at a curriculum's `difficulty` and `ramp`: its course, its start on it, its
    drawn body (`dew.rl.quadrotor.drawn_body`) and camera, its latency and rate limit; its rotors at the
    speed that holds its weight, its commands awaiting their latency its hover; no pillar within 1.5 m of
    its start."""
    course_key, start_key, body_key, camera_key, delay_key, rate_key, key = jax.random.split(key, 7)
    course = drawn_course(racing, course_key, difficulty)
    index, position, orientation, velocity = _start(racing, course, start_key, difficulty)
    # No pillar stands within 1.5 m of where the drone starts.
    pillars = course.obstacles
    clear = jnp.linalg.norm(pillars[:, :2] - position[:2], axis=-1) > pillars[:, 2] + racing.radius + 1.5
    course = dataclasses.replace(course, obstacles=pillars.at[:, 2].multiply(clear))
    body = drawn_body(racing.quadrotor, body_key, ramp)
    flight = dataclasses.replace(hovering(racing.quadrotor, body, position, orientation), velocity=velocity)
    shortest, longest = racing.delays
    hover = jnp.concatenate([(body.mass * GRAVITY)[None], jnp.zeros(3)])
    rate_limit = racing.rate_limit * (1 + ramp * racing.rate_spread
                                      * jax.random.uniform(rate_key, (), minval=-1.0, maxval=1.0))
    return Race(course=course, index=index, distance=jnp.linalg.norm(course.center[index] - position),
                steps=jnp.zeros((), jnp.int32), passed=jnp.zeros((), jnp.int32), action=jnp.zeros(4),
                pending=jnp.tile(hover, (longest + 1, 1)),
                delay=jax.random.randint(delay_key, (), shortest, longest + 1), rate_limit=rate_limit,
                body=body, flight=flight, camera=drawn_camera(racing, camera_key, ramp), key=key,
                difficulty=difficulty, ramp=ramp)


def commanded(racing: Racing, race: Race, action: jax.Array) -> tuple[Race, jax.Array, jax.Array]:
    """A drone's action `[4]` in [-1, 1] as the flight controller takes it after the drone's latency: the
    collective thrust (N), the throttle's share of the hover below zero and of the span from the hover to the
    nominal drone's full thrust above; the body rates (rad/s), its share of the drone's rate limit."""
    action = jnp.clip(action, -1, 1)
    hover = race.body.mass * GRAVITY
    full = racing.quadrotor.nominal_thrust
    collective = hover + action[0] * jnp.where(action[0] >= 0, full - hover, hover)
    command = jnp.concatenate([collective[None], action[1:] * race.rate_limit])
    pending = jnp.concatenate([command[None], race.pending[:-1]])
    sent = pending[race.delay]
    return dataclasses.replace(race, pending=pending), sent[0], sent[1:]


def bars(racing: Racing) -> np.ndarray:
    """A gate's four frame bars `[4, 3, 2]`, each a box's extent along the gate's normal, its left and up from
    its centre, grown by the drone's `radius` on every side: where the drone's centre meets the frame."""
    half, deep = racing.opening / 2, racing.depth / 2
    edge = half + racing.frame
    boxes = np.asarray([[(-deep, deep), (-edge, edge), (half, edge)],
                        [(-deep, deep), (-edge, edge), (-edge, -half)],
                        [(-deep, deep), (half, edge), (-half, half)],
                        [(-deep, deep), (-edge, -half), (-half, half)]])
    return boxes + np.asarray([-racing.radius, racing.radius])


def _pillar_struck(racing: Racing, begin: jax.Array, end: jax.Array, obstacles: jax.Array) -> jax.Array:
    """Whether a segment from `begin` to `end` `[S, 3]` brings the drone's sphere into a pillar `[O, 4]`: its
    closest approach across to the pillar's axis within the pillar's radius plus the drone's, below the
    pillar's top grown by the drone's radius there. A physics step's segment is far shorter than a pillar's
    and the drone's radii together, so its closest approach decides."""
    start, along = begin[:, None, :2], (end - begin)[:, None, :2]
    toward = obstacles[None, :, :2] - start
    length = jnp.sum(along * along, -1)
    t = jnp.clip(jnp.sum(toward * along, -1) / jnp.where(length > 0, length, 1.0), 0.0, 1.0)
    gap = jnp.linalg.norm(start + t[..., None] * along - obstacles[None, :, :2], axis=-1)
    height = begin[:, None, 2] + t * (end - begin)[:, None, 2]
    hit = (gap < obstacles[None, :, 2] + racing.radius) & (height < obstacles[None, :, 3] + racing.radius)
    return jnp.any(hit & (obstacles[None, :, 2] > 0))


def struck(racing: Racing, race: Race, path: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Whether a drone's path through a step `[S + 1, 3]` (its positions before it and after each physics
    step, each pair joined by a straight segment) crashed, passed the gate it heads for, and how far across
    from its centre it passed. It crashes on coming within `radius` of the ground or climbing past `ceiling`,
    on a segment entering a gate's frame bar grown by `radius` (`bars`), on coming within `radius` of a
    pillar, or on crossing the next gate's plane in its direction outside the frame within `miss_radius` (a
    missed gate). It passes that gate on crossing its plane in its direction with its centre within `radius`
    of the opening's inside, the larger of its offsets across and up from the centre no more than half the
    opening less `radius`."""
    course = race.course
    grounded = jnp.any((path[1:, 2] < racing.radius) | (path[1:, 2] > racing.ceiling))
    normal, left = _gate_axes(course.facing)
    up = jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0], normal.dtype), normal.shape)
    frame = jnp.stack([normal, left, up], -1)  # [G, 3 (world), 3 (gate)]
    begin, end = path[:-1, None], path[1:, None]
    near, far = (jnp.einsum("sgi,gij->sgj", point - course.center[None], frame) for point in (begin, end))
    # The segment meets a bar's open box where the parameters at which it is inside each axis's open interval
    # overlap (Liang-Barsky's clipping).
    boxes = jnp.asarray(bars(racing), path.dtype)
    start, along = near[..., None, :], (far - near)[..., None, :]
    still = along == 0
    step = jnp.where(still, 1.0, along)
    low, high = (boxes[..., 0] - start) / step, (boxes[..., 1] - start) / step
    entered = jnp.max(jnp.where(still, -jnp.inf, jnp.minimum(low, high)), -1)
    left_at = jnp.min(jnp.where(still, jnp.inf, jnp.maximum(low, high)), -1)
    inside = jnp.all(~still | ((boxes[..., 0] < start) & (start < boxes[..., 1])), -1)
    met = inside & (jnp.maximum(entered, 0.0) < jnp.minimum(left_at, 1.0))
    active = jnp.arange(GATES) < course.count
    framed = jnp.any(met, -1) & active[None]
    before, after = near[..., 0], far[..., 0]
    crossing = (before * after <= 0) & (before != after)
    share = before / jnp.where(before != after, before - after, 1.0)
    offset = near + share[..., None] * (far - near)
    across = jnp.maximum(jnp.abs(offset[..., 1]), jnp.abs(offset[..., 2]))
    half = racing.opening / 2
    inner, outer = half - racing.radius, half + racing.frame + racing.radius
    target = (jnp.arange(GATES) == race.index)[None]
    forward = crossing & target & (before < 0)
    passed = forward & (across <= inner)
    missed = forward & (across >= outer) & (across < racing.miss_radius)
    pillar = _pillar_struck(racing, path[:-1], path[1:], course.obstacles)
    crashed = grounded | jnp.any(framed | missed) | pillar
    passed_any = jnp.any(passed)
    centring = jnp.max(jnp.where(passed, across, -jnp.inf))
    return crashed, passed_any & ~crashed, jnp.where(passed_any, centring, 0.0)


def raced(racing: Racing, race: Race, action: jax.Array, path: jax.Array) -> tuple[
        Race, jax.Array, jax.Array, jax.Array]:
    """A step's race, reward, and whether its episode `terminated` or was `truncated`, from the action taken
    and the drone's path through the step to its flight after it. The reward is the progress toward the gate
    headed for, its distance before the step less its distance after (times `progress_scale`), less
    `rate_cost` times the body rates' norm over the step, less `smoothness_cost` times the squared change of
    the action; on passing that gate, `gate_bonus` times one less its offset's share of half the opening, the
    drone heading for the next; on a crash (`struck`), less `crash_penalty`, which ends the episode. Passing
    an open track's last gate ends it, a success. An episode is cut after `max_steps` steps."""
    crashed, passed, across = struck(racing, race, path)
    course, flight = race.course, race.flight
    distance = jnp.linalg.norm(course.center[race.index] - flight.position)
    progress = racing.progress_scale * (race.distance - distance)
    bonus = jnp.where(passed, racing.gate_bonus * (1 - across / (racing.opening / 2)), 0.0)
    rate = racing.rate_cost * jnp.linalg.norm(flight.angular_velocity) * racing.dt
    action = jnp.clip(action, -1, 1)
    smoothness = racing.smoothness_cost * jnp.sum(jnp.square(action - race.action))
    following = jnp.where(course.closed, (race.index + 1) % course.count, race.index + 1)
    finished = passed & ~course.closed & (following >= course.count)
    index = jnp.where(passed & ~finished, following, race.index)
    distance = jnp.linalg.norm(course.center[index] - flight.position)
    reward = progress + bonus - rate - smoothness - jnp.where(crashed, racing.crash_penalty, 0.0)
    taken = race.steps + 1
    after = dataclasses.replace(race, index=index, distance=distance, steps=taken, action=action,
                                passed=race.passed + passed)
    ended = crashed | finished
    return after, reward, ended, ~ended & (taken >= racing.max_steps)


def corners(racing: Racing, race: Race, ahead: int = 2) -> jax.Array:
    """The corners of the openings of the gate headed for and of the `ahead - 1` after it (on an open track,
    its last gate in place of any beyond it), in the body frame `[12 * ahead]` (Song 2023's track
    observation)."""
    course, flight = race.course, race.flight
    half = racing.opening / 2
    world_to_body = rotation(flight.orientation).T
    shown = []
    for step in range(ahead):
        index = jnp.where(course.closed, (race.index + step) % course.count,
                          jnp.minimum(race.index + step, course.count - 1))
        _, left = _gate_axes(course.facing[index])
        center = course.center[index]
        up = jnp.asarray([0.0, 0.0, 1.0], center.dtype)
        for sideways, upward in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            shown.append(world_to_body @ (center + (sideways * left + upward * up) * half - flight.position))
    return jnp.concatenate(shown)


def nearest(race: Race) -> jax.Array:
    """The `NEAREST` active pillars nearest the drone, each its axis's closest point's offset in the body
    frame (at the drone's height, held within the pillar) and its radius `[4 * NEAREST]`, an absent one
    zeros."""
    flight, pillars = race.flight, race.course.obstacles
    height = jnp.clip(flight.position[2], 0.0, pillars[:, 3])
    axis = jnp.concatenate([pillars[:, :2], height[:, None]], -1)
    active = pillars[:, 2] > 0
    reach = jnp.where(active, jnp.linalg.norm(axis - flight.position, axis=-1), jnp.inf)
    order = jax.lax.top_k(-reach, NEAREST)[1]
    offset = (rotation(flight.orientation).T @ (axis[order] - flight.position).T).T
    shown = jnp.concatenate([offset, pillars[order, 2:3]], -1)
    return jnp.where(active[order][:, None], shown, 0.0).reshape(-1)


def airframe(racing: Racing, race: Race) -> jax.Array:
    """The drone's parameters a world model learns to read off its latent and a teacher is shown `[6]`: its
    thrust-to-weight at its start's charge over the nominal drone's, its rate limit over nominal, its motors'
    response a control step (one less the decay of their lag over it), its latency in control steps, and its
    camera's uptilt and field of view."""
    q = racing.quadrotor
    nominal = q.nominal_thrust / (q.mass * GRAVITY)
    return jnp.stack([race.body.thrust_to_weight(q) / nominal, race.rate_limit / racing.rate_limit,
                      1 - jnp.exp(-racing.dt / race.body.motor_lag), race.delay.astype(race.rate_limit.dtype),
                      race.camera.uptilt, race.camera.fov])


def dynamics(racing: Racing, race: Race) -> jax.Array:
    """The drone's physics a teacher is shown and a world model learns to read off `[20]`, each over its
    nominal: its mass, inertia `[3]`, thrust and drag coefficients and top speed; its rotor, quadratic `[2]`
    and vertical drag; its ground effect's strength; its battery's voltage now over full and its charge drawn
    over the pack's; and its constant force over its mass times the disturbance's limit and its moment over
    the nominal inertia times its limit `[3]` each, within [-1, 1]. A term a quadrotor turns off (its nominal
    or limit zero, so every drone's zero too) reads as the nominal's, 1, and an absent disturbance as 0."""
    q, body, flight = racing.quadrotor, race.body, race.flight
    return jnp.concatenate([
        jnp.stack([body.mass / q.mass]), body.inertia / jnp.asarray(q.inertia),
        jnp.stack([body.thrust_coefficient / q.thrust_coefficient,
                   _over(body.drag_coefficient, q.drag_coefficient), body.top_speed / q.top_speed,
                   _over(body.rotor_drag, q.rotor_drag)]),
        _over(body.quadratic_drag, q.quadratic_drag),
        jnp.stack([_over(body.vertical_drag, q.vertical_drag), body.ground_effect,
                   flight.voltage / q.full_voltage, flight.drawn / q.capacity]),
        _over(body.force / body.mass, q.bias_force, 0.0),
        _over(body.moment, np.asarray(q.inertia) * q.bias_moment, 0.0)])


def _over(value: jax.Array, nominal: float | tuple[float, ...] | np.ndarray, off: float = 1.0) -> jax.Array:
    """`value` over a quadrotor's `nominal`, and `off` where that nominal is zero (the term turned off)."""
    nominal = np.asarray(nominal, np.float64)
    return jnp.where(nominal == 0, off, value / np.where(nominal == 0, 1.0, nominal).astype(value.dtype))


def privileged(racing: Racing, race: Race) -> dict[str, jax.Array]:
    """The privileged views: the `state` `[13]` (position, velocity, orientation, body rates), the next two
    `gates`' corners in the body frame `[24]`, the drone's `airframe` `[6]` and `dynamics` `[20]`, the
    `obstacles` nearest `[12]`, the gates `passed` so far, whether the gate headed for is an open track's
    last (`finishing`), and the `teacher`'s observation `[80]`: the velocity and the direction of gravity in
    the body frame, the body rates, the corners, the last action and the airframe (the first 43, its
    interface since FlyDreamer's first race), then the dynamics, the obstacles and the finishing flag, each
    about unit size."""
    flight = race.flight
    world_to_body = rotation(flight.orientation).T
    velocity = world_to_body @ flight.velocity
    down = world_to_body @ jnp.asarray([0.0, 0.0, -1.0], velocity.dtype)
    gates = corners(racing, race)
    frame = airframe(racing, race)
    physics = dynamics(racing, race)
    obstacles = nearest(race)
    course = race.course
    finishing = (~course.closed & (race.index == course.count - 1)).astype(velocity.dtype)
    state = jnp.concatenate([flight.position, flight.velocity, flight.orientation, flight.angular_velocity])
    rates = flight.angular_velocity
    scaled = obstacles / jnp.asarray([10.0, 10.0, 10.0, 1.0] * NEAREST)
    teacher = jnp.concatenate([velocity / 10, down, rates / 10, gates / 10, race.action, frame, physics,
                               scaled, finishing[None]])
    return {"state": state, "gates": gates, "airframe": frame, "dynamics": physics, "obstacles": obstacles,
            "passed": race.passed, "finishing": finishing, "teacher": teacher}


TEACHER = 43 + 20 + 4 * NEAREST + 1
"""The width of the teacher's observation (`privileged`)."""
INTERFACE = 43
"""The width of the teacher's observation in its first race: its first entries keep their meaning."""
VIEWS = ("image", "state", "gates", "airframe", "dynamics", "obstacles", "passed", "finishing", "teacher")
"""What a gate race can show: the camera's image, and the privileged views (`privileged`)."""


@struct.dataclass
class FlightEnv:
    """One drone racing `racing`'s courses as a `dew.rl.env.Env`: actions `[4]` in [-1, 1] (`commanded`);
    observations a dict of the privileged `views` asked for (`privileged`). An episode is drawn at its
    curriculum, `difficulty` and `ramp` (`started`); a curriculum changes the env, its programs compiled again
    (a few times a run)."""

    racing: Racing = struct.field(pytree_node=False)
    views: tuple[str, ...] = struct.field(pytree_node=False, default=("teacher",))
    difficulty: float = struct.field(pytree_node=False, default=1.0)
    ramp: float = struct.field(pytree_node=False, default=1.0)

    action_size = 4

    def __post_init__(self):
        if not self.views or not set(self.views) <= set(VIEWS[1:]):
            raise ValueError(f"a flight env shows some of the privileged views {VIEWS[1:]}, not {self.views}")

    def _observed(self, race: Race) -> dict[str, jax.Array]:
        shown = privileged(self.racing, race)
        return {view: shown[view] for view in self.views}

    def reset(self, key: jax.Array) -> tuple[Race, TimeStep]:
        race = started(self.racing, key, jnp.asarray(self.difficulty), jnp.asarray(self.ramp))
        zero = jnp.zeros((), bool)
        return race, TimeStep(self._observed(race), jnp.zeros(()), zero, zero)

    def flown(self, race: Race, action: jax.Array) -> tuple[Race, jax.Array]:
        """The race after the drone flies a control step under `action`, and its path through it."""
        race, collective, rates = commanded(self.racing, race, action)
        key, noise = jax.random.split(race.key)
        flight, path = flown(self.racing.quadrotor, race.body, race.flight, collective, rates, noise,
                             self.racing.hz, race.ramp)
        return dataclasses.replace(race, flight=flight, key=key), path

    def step(self, race: Race, action: jax.Array) -> tuple[Race, TimeStep]:
        race, path = self.flown(race, action)
        race, reward, terminated, truncated = raced(self.racing, race, action, path)
        return race, TimeStep(self._observed(race), reward, terminated, truncated)


@struct.dataclass
class Scene:
    """What a renderer draws for each drone: its course's gates `[B, G, 3]`, facings `[B, G]` and count `[B]`,
    the gate lit (`lit` `[B]`, -1 for none), its obstacles `[B, O, 4]`, and its camera's looks `[B, 4]`
    (`Camera.look`)."""

    center: jax.Array
    facing: jax.Array
    count: jax.Array
    lit: jax.Array
    obstacles: jax.Array
    look: jax.Array


class Renderer(Protocol):
    """A renderer of a batch of `count` drones' cameras: `render(scene, cameras, opened)` answers each drone's
    image `[B, H, W, 3]` uint8 of its `Scene` from its camera at the shutter's close `[B, 12]` (`cameras`)
    and at its open (for motion blur). Its `aspect` is its images' width over height, and `gate` the opening,
    frame and depth of the gates it draws, which a race's must be. Arrays cross by DLPack
    (`dew.rl.simulator.Handed`)."""

    @property
    def count(self) -> int: ...

    @property
    def aspect(self) -> float: ...

    @property
    def gate(self) -> tuple[float, float, float]: ...

    def render(self, scene: Scene, cameras: jax.Array, opened: jax.Array) -> Handed: ...


def cameras(race: Race, aspect: float, before: float = 0.0) -> jax.Array:
    """A drone's camera `[12]` as HyperLux takes it (`set_cameras`: the position, the ray to the image's first
    pixel's corner and the rays across a pixel row and down a column), its mount tilted up by the camera's
    uptilt and turned by its mount's roll and yaw, its field of view the camera's: HyperDrone's
    `camera_base_one` (hyperdrone/_native/dynamics/impl.cpp) of the mounted camera; `before` seconds earlier,
    the drone moved back along its velocity and turned back by its body rates (a shutter's open)."""
    camera, flight = race.camera, race.flight
    position = flight.position - before * flight.velocity
    turned = rotation(flight.orientation)
    rates = -before * flight.angular_velocity
    angle = jnp.linalg.norm(rates)
    axis = rates / jnp.maximum(angle, 1e-12)
    back = rotation(jnp.concatenate([jnp.cos(angle / 2)[None], jnp.sin(angle / 2) * axis]))
    mount = rotation(quaternion(camera.mount[0], -camera.uptilt, camera.mount[1]))
    axes = turned @ back @ mount
    forward, left, up = axes[:, 0], axes[:, 1], axes[:, 2]
    scale = 2 * jnp.tan(camera.fov / 2)
    across, down = -left * scale, -up * scale / aspect
    first = forward - 0.5 * across - 0.5 * down
    return jnp.concatenate([position, first, across, down])


def developed(image: jax.Array, camera: Camera, key: jax.Array) -> jax.Array:
    """A rendered image `[H, W, 3]` uint8 as a drone's camera develops it, in the camera's dtype: its
    exposure, white balance and gamma, and read noise of the camera's std (full scale)."""
    dtype = camera.exposure.dtype
    light = divided(image.astype(dtype), 255) * camera.exposure * camera.white_balance
    light = jnp.power(jnp.clip(light, 0.0, 1.0), camera.gamma)
    light = light + camera.noise * jax.random.normal(key, light.shape, dtype)
    return jnp.round(jnp.clip(light, 0.0, 1.0) * 255).astype(jnp.uint8)


class GateRace:
    """A batch of `renderer.count` drones racing `racing`'s courses (`FlightEnv`, stepped on device) and seen
    through `renderer`'s cameras, as a `dew.rl.simulator.Simulator`: actions `[B, 4]` in [-1, 1];
    observations a dict of the `views` asked for (`VIEWS`), each `[B, ...]`. An episode restarts on a course
    drawn afresh at the race's curriculum, `difficulty` and `ramp`, which may change between steps."""

    action_size = 4
    _key: jax.Array
    _races: Race

    def __init__(self, renderer: Renderer | None, racing: Racing | None = None,
                 views: tuple[str, ...] = VIEWS, count: int | None = None):
        if not views or not set(views) <= set(VIEWS):
            raise ValueError(f"a gate race shows some of {VIEWS}, not {views}")
        if renderer is None and ("image" in views or count is None):
            raise ValueError("a gate race without a renderer shows no image and is told its drones' count")
        racing = Racing() if racing is None else racing
        if renderer is not None and tuple(renderer.gate) != (racing.opening, racing.frame, racing.depth):
            raise ValueError(f"the renderer draws gates of opening, frame and depth {tuple(renderer.gate)}, "
                             f"the race's are {(racing.opening, racing.frame, racing.depth)}: its drones "
                             "would see gates they do not fly through")
        self.renderer, self.racing, self.views = renderer, racing, views
        self.count = renderer.count if renderer is not None else int(count or 0)
        self.difficulty, self.ramp = 1.0, 1.0
        shown = tuple(view for view in views if view != "image")
        env = FlightEnv(racing, shown if shown else ("passed",))
        self._env = env
        self._started = jax.jit(lambda env, key: jax.vmap(env.reset)(jax.random.split(key, self.count)))
        self._stepped = jax.jit(lambda env, races, action: jax.vmap(env.step)(races, action))
        aspect = 1.0 if renderer is None else renderer.aspect
        self._shown = jax.jit(jax.vmap(lambda race, image, key: developed(image, race.camera, key)))
        self._observe = jax.jit(jax.vmap(env._observed))
        self._scene = jax.jit(jax.vmap(partial(_scene, racing)))
        self._cameras = jax.jit(jax.vmap(lambda race: (cameras(race, aspect),
                                                       cameras(race, aspect, race.camera.shutter))))

    def _env_now(self) -> FlightEnv:
        return dataclasses.replace(self._env, difficulty=float(self.difficulty), ramp=float(self.ramp))

    def _observed(self, observation: dict[str, jax.Array]) -> dict[str, jax.Array]:
        shown = dict(observation)
        if "image" in self.views:
            renderer = self.renderer
            if renderer is None:
                raise ValueError("a gate race without a renderer shows no image")
            closed, opened = self._cameras(self._races)
            image = received(renderer.render(self._scene(self._races), closed, opened))
            self._key, key = jax.random.split(self._key)
            shown["image"] = self._shown(self._races, image, jax.random.split(key, self.count))
        return {view: shown[view] for view in self.views}

    def reset(self, seed: int) -> dict[str, jax.Array]:
        self._key, key = jax.random.split(jax.random.key(seed))
        self._races, step = self._started(self._env_now(), key)
        return self._observed(step.observation)

    def restart(self, mask: jax.Array) -> dict[str, jax.Array]:
        mask = jnp.asarray(mask, bool)
        self._key, key = jax.random.split(self._key)
        fresh, _ = self._started(self._env_now(), key)
        self._races = chosen(mask, fresh, self._races)
        return self._observed(self._observe(self._races))

    def step(self, action: jax.Array) -> TimeStep:
        self._races, step = self._stepped(self._env_now(), self._races, jnp.asarray(action))
        return TimeStep(self._observed(step.observation), step.reward, step.terminated, step.truncated)


def _scene(racing: Racing, race: Race) -> Scene:
    """What a renderer draws of a drone's race (`Scene`): its gate headed for lit where the race lights it."""
    course = race.course
    lit = jnp.where(racing.highlight, race.index, -1)
    return Scene(course.center, course.facing, course.count, lit, course.obstacles, race.camera.look)


__all__ = ["ASPECTS", "FAMILIES", "GATES", "INTERFACE", "LAYOUTS", "NEAREST", "OBSTACLES", "TEACHER",
           "VIEWS", "Camera", "Course", "FlightEnv", "GateRace", "Race", "Racing", "Renderer", "Scene",
           "airframe", "bars", "cameras", "commanded", "corners", "developed", "drawn_camera",
           "drawn_course", "dynamics", "facings", "nearest", "privileged", "quaternion", "raced",
           "started", "struck"]
