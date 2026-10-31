"""First-person gate racing for a batch of quadrotors, any flight simulator behind it.

The reward is the one drone-racing RL has converged on: each step, the progress made toward the next gate's
centre (Kaufmann et al., Nature 2023, Swift, Eq 8; Song et al., Sci. Robot. 2023, Eq 5; Romero et al., arXiv
2501.14377, Eq 1; Verraest et al., arXiv 2510.14783, SkyDreamer, Sec II-C), less a cost on the body rates
(Song 2023, b = 0.01 a step at 100 Hz, here 1 a second) and on the action's change (Swift's lambda5, its
weight here over actions in [-1, 1]); a bonus on passing a gate, the more for passing it centred
(SkyDreamer's r_gate); and a crash's penalty, which ends the episode (Song 2023's -10). A gate counts only
when its plane is crossed through its opening in its direction; crossing the next gate's plane beside it
misses it, a crash (SkyDreamer; Ferede et al., arXiv 2504.21586). Gates' frames and the ground are solid,
the drone's path through each physics step tested against them (`struck`), so a fast drone cannot pass
through a frame between two steps.

Each episode is flown on a course of its own (`drawn_track`): a published racing layout (uzh-rpg/agile_flight,
MIT: Split-S and its real-world variant, AlphaPilot; and FlyDreamer's loop) with each gate moved by up to
`jitter` (Song et al., IROS 2021, Sec IV-C), or a random closed loop; the `difficulty`, from 0 to 1, scales
the jitter and the random loops' turns and climbs, a curriculum (Song 2021 raised it on the crash rate). An
episode starts before a random gate of its course, facing it (Swift; SkyDreamer's Table III).

A drone is randomized each episode for transfer from simulation (`Airframe`): its thrust-to-weight and rate
limit within `spread` of nominal (SkyDreamer's Table III, Ferede), a first-order lag on the commanded rates
and a latency of whole control steps (Swift measured 40 ms; Song 2023 simulates it), its camera's uptilt,
field of view and mount, and its image's brightness (HyperDrone, arXiv 2610.28950, Table 5). An action keeps
FlyDreamer's semantics (env.py `_action_to_command`): its first entry is the throttle, zero the hover and the
ends no thrust and full thrust; the rest the body rates over the drone's limit. The policy sees the camera's
image alone; the state, the next gates' corners and the drone's randomized parameters are privileged, for a
world model's auxiliary heads (SkyDreamer's privileged decoding) and a teacher (`TEACHER`, Swift's and Song
2023's observation with the parameters).

The world frame is x forward, y left, z up (FLU), the body frame forward, left, up. A simulator supplies the
physics and renders the camera (`FlightBackend`); `GateRace` makes it a `dew.rl.simulator.Simulator`.
"""

from __future__ import annotations

import dataclasses
from functools import partial
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from dew.rl.env import TimeStep, chosen
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
AlphaPilot.yaml, MIT) and FlyDreamer's loop
(AshishKumar4/FlyDreamer dreamerv3/configs.yaml `static_gates`, turned from NED). Each gate faces along the
course (`facings`): the layouts' own gate rotations follow their simulator's model's axes, which the files
do not state."""
GATES = max(map(len, LAYOUTS.values()))
"""The most gates a course holds; a shorter one's last are inactive."""


@dataclasses.dataclass(frozen=True)
class Racing:
    """The race. `layouts` names the layouts drawn (`LAYOUTS`) and `random_share` the share of courses that
    are random loops; `jitter` the most a layout's gate moves across and up, at full `difficulty`. The
    reward's weights are `raced`'s; `hz` the control rate, 60 Hz (a teacher at 60 Hz passed 27% more gates
    than at 30 in the same time and crashed less; Swift's and Song's run at 90 to 100 Hz); an episode is
    cut after `episode_seconds` (Swift's 1500 steps). Gates' openings are `opening` square (1.5 m,
    SkyDreamer's and Ferede's) in a frame `frame` wide and `depth` deep; the drone collides as a cube of
    half-width `radius` aligned with the gate it meets (`struck`). `spread` is each
    drone parameter's randomization about nominal, `latency` the command latency's range in seconds, `lag`
    the rate response's time constant's, `uptilt` the camera's tilt above the body's forward axis in
    degrees, `fov` its horizontal field of view and `fov_spread` and `mount_spread` their randomization,
    `brightness` the image's gain's range."""

    layouts: tuple[str, ...] = tuple(LAYOUTS)
    random_share: float = 0.5
    jitter: tuple[float, float] = (1.5, 0.75)
    difficulty: float = 1.0
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
    ceiling: float = 20.0
    spread: float = 0.2
    rate_limit: float = 10.0
    latency: tuple[float, float] = (0.02, 0.05)
    lag: tuple[float, float] = (0.02, 0.04)
    uptilt: tuple[float, float] = (15.0, 35.0)
    fov: float = 100.0
    fov_spread: float = 5.0
    mount_spread: float = 5.0
    brightness: tuple[float, float] = (0.5, 1.5)

    def __post_init__(self):
        unknown = set(self.layouts) - set(LAYOUTS)
        if unknown or not (self.layouts or self.random_share == 1) or not 0 <= self.random_share <= 1:
            raise ValueError(f"a race draws layouts of {sorted(LAYOUTS)}, not {sorted(unknown)}, or random "
                             "loops alone (random_share 1)")

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
class Kinematics:
    """A batch of drones' states: `position` `[B, 3]` in metres and `velocity` `[B, 3]` in metres a second
    in the world frame, `orientation` `[B, 4]` the body-to-world rotation as a unit quaternion (w, x, y,
    z), and `angular_velocity` `[B, 3]` the body rates in radians a second."""

    position: jax.Array
    velocity: jax.Array
    orientation: jax.Array
    angular_velocity: jax.Array


@struct.dataclass
class Airframe:
    """Each drone's randomized parameters for an episode: its thrust-to-weight's and rate limit's factors
    over nominal, its rate response's smoothing factor a control step (from its lag), its latency in
    control steps, its camera's uptilt and horizontal field of view (radians) and mount's roll and yaw, and
    its image's brightness."""

    thrust: jax.Array
    rates: jax.Array
    smoothing: jax.Array
    delay: jax.Array
    uptilt: jax.Array
    fov: jax.Array
    mount: jax.Array
    brightness: jax.Array

    def parameters(self) -> jax.Array:
        """The parameters a world model learns to read off its latent `[B, 6]`."""
        return jnp.stack([self.thrust, self.rates, self.smoothing, self.delay.astype(self.thrust.dtype),
                          self.uptilt, self.fov], -1)


@struct.dataclass
class Track:
    """Each drone's episode: its course's gates `[B, G, 3]`, their facings `[B, G]` (radians) and count
    `[B]`; the gate it heads for, its distance from it after the last step, the steps taken and the gates
    passed; the last action, the rates' command after its lag, and the commands awaiting their latency
    `[B, L, 4]` (each a collective acceleration in g and body rates in radians a second, hovering at the
    start); and its `Airframe`."""

    course: jax.Array
    facing: jax.Array
    count: jax.Array
    index: jax.Array
    distance: jax.Array
    steps: jax.Array
    passed: jax.Array
    action: jax.Array
    rates: jax.Array
    pending: jax.Array
    airframe: Airframe


@struct.dataclass
class Gates:
    """Each drone's gates as a renderer shows them: their centres `[B, G, 3]`, facings `[B, G]` and how many
    are its course's `[B]`."""

    center: jax.Array
    facing: jax.Array
    count: jax.Array


def facings(course: jax.Array, count: jax.Array) -> jax.Array:
    """Each gate's facing `[G]`: the yaw of the level line from the gate before it to the gate after it on
    a course of `count` gates `[G, 3]`, which loops."""
    order = jnp.arange(course.shape[0])
    before = course[(order - 1) % count]
    after = course[(order + 1) % count]
    along = after - before
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


@struct.dataclass
class CourseDraw:
    """An episode's course's draws, each uniform in its range: the layout (`which`, its index in `LAYOUTS`)
    and each of its gates' move `[G, 3]` in [-1, 1]; whether the course is a random loop instead; and the
    loop's gate count (6 to `GATES`), its ellipse's radii `[2]` (12 to 30 m), its turn (0 to 2 pi), each
    gate's move along the ellipse `[G]` (-1/4 to 1/4 of the gates' spacing) and across it (-0.1 to 0.1 of
    its radius), and height `[G]` (0 to 4.5 m), and whether it is flown anticlockwise."""

    which: jax.Array
    moved: jax.Array
    looped: jax.Array
    count: jax.Array
    radii: jax.Array
    turn: jax.Array
    along: jax.Array
    across: jax.Array
    height: jax.Array
    anticlockwise: jax.Array


def course_draws(racing: Racing, key: jax.Array) -> CourseDraw:
    """An episode's course's draws (`CourseDraw`): its layout one of `racing.layouts`, alike likely, and a
    random loop with `random_share`."""
    kept = np.asarray([name in racing.layouts for name in LAYOUTS])
    keys = iter(jax.random.split(key, 10))

    def within(low, high, shape=()):
        return jax.random.uniform(next(keys), shape, minval=low, maxval=high)

    return CourseDraw(which=jax.random.choice(next(keys), len(LAYOUTS), p=jnp.asarray(kept / kept.sum())),
                      moved=within(-1.0, 1.0, (GATES, 3)), looped=within(0.0, 1.0) < racing.random_share,
                      count=jax.random.randint(next(keys), (), 6, GATES + 1), radii=within(12.0, 30.0, (2,)),
                      turn=within(0.0, 2 * jnp.pi), along=within(-0.25, 0.25, (GATES,)),
                      across=within(-0.1, 0.1, (GATES,)), height=within(0.0, 4.5, (GATES,)),
                      anticlockwise=jax.random.bernoulli(next(keys)))


def course_of(racing: Racing, drawn: CourseDraw, difficulty: jax.Array) -> tuple[jax.Array, jax.Array]:
    """An episode's course `[GATES, 3]` and gate count from its draws at `difficulty`: its layout, each
    gate moved by its move times `jitter` across and up, times the difficulty, and held at least 0.75 m
    high; or its random loop, gates spaced evenly about an ellipse (each moved along and across it by its
    moves times the difficulty), turned, flown its way round, and 1.5 m high plus its height times the
    difficulty, the course's unused gates its first."""
    courses, counts = _layout_courses()
    reach = jnp.asarray([racing.jitter[0], racing.jitter[0], racing.jitter[1]])
    layout = jnp.asarray(courses)[drawn.which] + difficulty * drawn.moved * reach
    layout = layout.at[:, 2].set(jnp.maximum(layout[:, 2], 0.75))
    order = jnp.arange(GATES)
    angle = (order + difficulty * drawn.along) * (2 * jnp.pi / drawn.count)
    angle = jnp.where(drawn.anticlockwise, angle, -angle) + drawn.turn
    spread = 1 + difficulty * drawn.across
    loop = jnp.stack([drawn.radii[0] * spread * jnp.cos(angle), drawn.radii[1] * spread * jnp.sin(angle),
                      1.5 + difficulty * drawn.height], -1)
    loop = jnp.where((order < drawn.count)[:, None], loop, loop[0])
    return (jnp.where(drawn.looped, loop, layout),
            jnp.where(drawn.looped, drawn.count, jnp.asarray(counts)[drawn.which]))


def drawn_track(racing: Racing, key: jax.Array, difficulty: jax.Array) -> tuple[jax.Array, jax.Array]:
    """An episode's course `[GATES, 3]` and gate count drawn from `key` (`course_draws`, `course_of`)."""
    return course_of(racing, course_draws(racing, key), difficulty)


def quaternion(roll: jax.Array, pitch: jax.Array, yaw: jax.Array) -> jax.Array:
    """Unit quaternions (w, x, y, z) of a yaw, then a pitch, then a roll."""
    cr, sr, cp, sp, cy, sy = (f(angle / 2) for angle in (roll, pitch, yaw) for f in (jnp.cos, jnp.sin))
    return jnp.stack([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                      cr * cp * sy - sr * sp * cy], -1)


def rotation(orientation: jax.Array) -> jax.Array:
    """The body-to-world rotation matrices `[..., 3, 3]` of unit quaternions (w, x, y, z)."""
    w, x, y, z = jnp.moveaxis(orientation, -1, 0)
    return jnp.stack([
        jnp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        jnp.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        jnp.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def _gate_axes(facing: jax.Array) -> tuple[jax.Array, jax.Array]:
    """A gate's normal (its passing direction) and its left `[..., 3]`, level, from its facing."""
    zero = jnp.zeros_like(facing)
    normal = jnp.stack([jnp.cos(facing), jnp.sin(facing), zero], -1)
    return normal, jnp.stack([-jnp.sin(facing), jnp.cos(facing), zero], -1)


def drawn_airframe(racing: Racing, key: jax.Array) -> Airframe:
    """A drone's parameters for an episode (`Airframe`), each uniform in its range."""
    keys = iter(jax.random.split(key, 9))

    def within(low, high):
        return jax.random.uniform(next(keys), (), minval=low, maxval=high)

    lag = within(*racing.lag)
    shortest, longest = racing.delays
    mount = jnp.deg2rad(jax.random.uniform(next(keys), (2,), minval=-racing.mount_spread,
                                           maxval=racing.mount_spread))
    return Airframe(thrust=within(1 - racing.spread, 1 + racing.spread),
                    rates=within(1 - racing.spread, 1 + racing.spread),
                    smoothing=1 - jnp.exp(-racing.dt / lag),
                    delay=jax.random.randint(next(keys), (), shortest, longest + 1),
                    uptilt=jnp.deg2rad(within(*racing.uptilt)),
                    fov=jnp.deg2rad(within(racing.fov - racing.fov_spread, racing.fov + racing.fov_spread)),
                    mount=mount, brightness=within(*racing.brightness))


def started(racing: Racing, key: jax.Array, difficulty: jax.Array) -> tuple[Track, jax.Array, jax.Array]:
    """A drone's episode (`Track`) and its start's position and orientation: 2 to 4 m before a random gate of
    a drawn course, up to 1 m aside and 0.5 m up or down, facing it to within 20 degrees of yaw and 10 of
    roll and pitch (Swift starts at a random gate; SkyDreamer's Table III)."""
    course_key, airframe_key, gate_key, place_key, turn_key = jax.random.split(key, 5)
    course, count = drawn_track(racing, course_key, difficulty)
    facing = facings(course, count)
    index = jax.random.randint(gate_key, (), 0, count)
    normal, left = _gate_axes(facing[index])
    back, aside, up = jax.random.uniform(place_key, (3,), minval=jnp.asarray([2.0, -1.0, -0.5]),
                                         maxval=jnp.asarray([4.0, 1.0, 0.5]))
    position = course[index] - back * normal + aside * left + jnp.asarray([0.0, 0.0, 1.0]) * up
    position = position.at[2].set(jnp.maximum(position[2], 0.5))
    roll, pitch, yaw = jnp.deg2rad(jax.random.uniform(turn_key, (3,), minval=-1.0, maxval=1.0)
                                   * jnp.asarray([10.0, 10.0, 20.0]))
    orientation = quaternion(roll, pitch, facing[index] + yaw)
    airframe = drawn_airframe(racing, airframe_key)
    longest = racing.delays[1]
    hovering = jnp.tile(jnp.asarray([1.0, 0.0, 0.0, 0.0]), (longest + 1, 1))
    track = Track(course, facing, count, index, jnp.linalg.norm(course[index] - position),
                  jnp.zeros((), jnp.int32), jnp.zeros((), jnp.int32), jnp.zeros(4), jnp.zeros(3), hovering,
                  airframe)
    return track, position, orientation


def commanded(racing: Racing, track: Track, action: jax.Array, thrust_to_weight: float,
              rate_limit: float) -> tuple[Track, jax.Array]:
    """Each drone's action `[B, 4]` in [-1, 1] as the command the simulator steps, `[B, 4]` in [-1, 1] of its
    collective acceleration's range (0 to `thrust_to_weight` g) and its rate range (`rate_limit`), and the
    track after it. The throttle is a share of the drone's own full thrust, `racing`'s nominal times its
    factor, the hover at zero; the rates its limit's share, through the drone's lag; the command then waits
    out the drone's latency (`Track.pending`), the drone hovering until its first command arrives."""
    action = jnp.clip(action, -1, 1)
    airframe = track.airframe
    strongest = airframe.thrust * thrust_to_weight / (1 + racing.spread)
    hover = 1 / strongest
    throttle = hover + action[:, 0] * jnp.where(action[:, 0] >= 0, 1 - hover, hover)
    wanted = action[:, 1:] * (airframe.rates * racing.rate_limit)[:, None]
    rates = track.rates + airframe.smoothing[:, None] * (wanted - track.rates)
    command = jnp.concatenate([(throttle * strongest)[:, None], rates], -1)
    pending = jnp.concatenate([command[:, None], track.pending[:, :-1]], 1)
    sent = pending[jnp.arange(pending.shape[0]), airframe.delay]
    normalized = jnp.concatenate([2 * sent[:, :1] / thrust_to_weight - 1, sent[:, 1:] / rate_limit], -1)
    return dataclasses.replace(track, rates=rates, pending=pending), jnp.clip(normalized, -1, 1)


def bars(racing: Racing) -> np.ndarray:
    """A gate's four frame bars `[4, 3, 2]`, each a box's extent along the gate's normal, its left and up
    from its centre, grown by the drone's `radius` on every side: where the drone's centre meets the
    frame."""
    half, deep = racing.opening / 2, racing.depth / 2
    edge = half + racing.frame
    boxes = np.asarray([[(-deep, deep), (-edge, edge), (half, edge)],
                        [(-deep, deep), (-edge, edge), (-edge, -half)],
                        [(-deep, deep), (half, edge), (-half, half)],
                        [(-deep, deep), (-edge, -half), (-half, half)]])
    return boxes + np.asarray([-racing.radius, racing.radius])


def struck(racing: Racing, track: Track, path: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Where each drone's path through a step `[B, S + 1, 3]` (its positions before it and after each
    physics step, each pair joined by a straight segment) crashed, passed the gate it heads for, and how far
    across from its centre it passed, `[B]` each. It crashes on coming within `radius` of the ground or
    climbing past `ceiling`, on a segment entering the inside of a gate's frame bar grown by `radius`
    (`bars`), or on crossing the next gate's plane in its direction outside the opening's frame within
    `miss_radius` (a missed gate). It passes that gate on crossing its plane in its direction with its
    centre within `radius` of the opening's inside, the larger of its offsets across and up from the
    centre (`across`) no more than half the opening less `radius`."""
    grounded = jnp.any((path[:, 1:, 2] < racing.radius) | (path[:, 1:, 2] > racing.ceiling), -1)
    normal, left = _gate_axes(track.facing[:, None])
    up = jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0], normal.dtype), normal.shape)
    frame = jnp.stack([normal, left, up], -1)  # [B, 1, G, 3 (world), 3 (gate)]
    begin, end = path[:, :-1, None], path[:, 1:, None]
    center = track.course[:, None]
    near, far = (jnp.einsum("bsgi,bsgij->bsgj", point - center, frame) for point in (begin, end))
    # The segment meets a bar's open box where the parameters at which it is inside each axis's open
    # interval overlap (Liang-Barsky's clipping).
    boxes = jnp.asarray(bars(racing), path.dtype)
    start, along = near[..., None, :], (far - near)[..., None, :]
    still = along == 0
    step = jnp.where(still, 1.0, along)
    low, high = (boxes[..., 0] - start) / step, (boxes[..., 1] - start) / step
    entered = jnp.max(jnp.where(still, -jnp.inf, jnp.minimum(low, high)), -1)
    left_at = jnp.min(jnp.where(still, jnp.inf, jnp.maximum(low, high)), -1)
    inside = jnp.all(~still | ((boxes[..., 0] < start) & (start < boxes[..., 1])), -1)
    met = inside & (jnp.maximum(entered, 0.0) < jnp.minimum(left_at, 1.0))
    active = (jnp.arange(track.course.shape[1]) < track.count[:, None])[:, None]
    framed = jnp.any(met, -1) & active
    before, after = near[..., 0], far[..., 0]
    crossing = (before * after <= 0) & (before != after)
    share = before / jnp.where(before != after, before - after, 1.0)
    offset = near + share[..., None] * (far - near)
    across = jnp.maximum(jnp.abs(offset[..., 1]), jnp.abs(offset[..., 2]))
    half = racing.opening / 2
    inner, outer = half - racing.radius, half + racing.frame + racing.radius
    target = jax.nn.one_hot(track.index, track.course.shape[1], dtype=bool)[:, None]
    forward = crossing & target & (before < 0)
    passed = forward & (across <= inner)
    missed = forward & (across >= outer) & (across < racing.miss_radius)
    crashed = grounded | jnp.any(framed | missed, (1, 2))
    passed_any = jnp.any(passed, (1, 2))
    centring = jnp.max(jnp.where(passed, across, -jnp.inf), (1, 2))
    return crashed, passed_any & ~crashed, jnp.where(passed_any, centring, 0.0)


def raced(racing: Racing, track: Track, action: jax.Array, kinematics: Kinematics, path: jax.Array,
          collided: jax.Array) -> tuple[Track, jax.Array, jax.Array, jax.Array]:
    """A step's track, reward, and whether its episode `terminated` or was `truncated`, from the action
    taken, the drones' states and paths after it and whether the simulator saw each collide. The reward is
    the progress toward the gate headed for, its distance before the step less its distance after (times
    `progress_scale`), less `rate_cost` times the body rates' norm over the step, less `smoothness_cost`
    times the squared change of the action; on passing that gate, `gate_bonus` times one less its offset's
    share of half the opening, the drone heading for the next; and on a crash (`struck`, or the
    simulator's), less `crash_penalty`, which ends the episode. An episode is cut after `max_steps` steps
    (`episode_seconds`)."""
    crashed, passed, across = struck(racing, track, path)
    crashed = crashed | collided
    # A crash voids a passage in the same step, the simulator's as the race's own.
    passed = passed & ~crashed
    rows = jnp.arange(track.index.shape[0])
    position = kinematics.position
    distance = jnp.linalg.norm(track.course[rows, track.index] - position, axis=-1)
    progress = racing.progress_scale * (track.distance - distance)
    bonus = jnp.where(passed, racing.gate_bonus * (1 - across / (racing.opening / 2)), 0.0)
    rate = racing.rate_cost * jnp.linalg.norm(kinematics.angular_velocity, axis=-1) * racing.dt
    action = jnp.clip(action, -1, 1)
    smoothness = racing.smoothness_cost * jnp.sum(jnp.square(action - track.action), -1)
    index = jnp.where(passed, (track.index + 1) % track.count, track.index)
    distance = jnp.linalg.norm(track.course[rows, index] - position, axis=-1)
    reward = progress + bonus - rate - smoothness - jnp.where(crashed, racing.crash_penalty, 0.0)
    taken = track.steps + 1
    following = dataclasses.replace(track, index=index, distance=distance, steps=taken, action=action,
                                    passed=track.passed + passed)
    return following, reward, crashed, ~crashed & (taken >= racing.max_steps)


def corners(racing: Racing, track: Track, kinematics: Kinematics, ahead: int = 2) -> jax.Array:
    """The corners of the openings of the gate headed for and of the `ahead - 1` after it, in the body frame
    `[B, 12 * ahead]` (Song 2023's track observation)."""
    rows = jnp.arange(track.index.shape[0])
    half = racing.opening / 2
    shown = []
    world_to_body = jnp.swapaxes(rotation(kinematics.orientation), -1, -2)
    for step in range(ahead):
        index = (track.index + step) % track.count
        _, left = _gate_axes(track.facing[rows, index])
        center = track.course[rows, index]
        up = jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0], center.dtype), center.shape)
        for sideways, upward in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            corner = center + (sideways * left + upward * up) * half - kinematics.position
            shown.append(jnp.einsum("bij,bj->bi", world_to_body, corner))
    return jnp.concatenate(shown, -1)


def privileged(racing: Racing, track: Track, kinematics: Kinematics) -> dict[str, jax.Array]:
    """The privileged views: the `state` `[B, 13]` (position, velocity, orientation, body rates), the next
    two `gates`' corners in the body frame `[B, 24]`, the drone's randomized `airframe` `[B, 6]`, the gates
    `passed` so far in the episode `[B]`, and the `teacher`'s observation `[B, 43]`: the velocity and the
    direction of gravity in the body frame, the body rates, the corners, the last action and the airframe,
    each about unit size."""
    world_to_body = jnp.swapaxes(rotation(kinematics.orientation), -1, -2)
    velocity = jnp.einsum("bij,bj->bi", world_to_body, kinematics.velocity)
    down = jnp.einsum("bij,j->bi", world_to_body, jnp.asarray([0.0, 0.0, -1.0], velocity.dtype))
    gates = corners(racing, track, kinematics)
    airframe = track.airframe.parameters()
    state = jnp.concatenate([kinematics.position, kinematics.velocity, kinematics.orientation,
                             kinematics.angular_velocity], -1)
    rates = kinematics.angular_velocity
    teacher = jnp.concatenate([velocity / 10, down, rates / 10, gates / 10, track.action, airframe], -1)
    return {"state": state, "gates": gates, "airframe": airframe, "passed": track.passed, "teacher": teacher}


def cameras(track: Track, kinematics: Kinematics, aspect: float) -> jax.Array:
    """Each drone's camera `[B, 12]` as HyperLux takes it (`set_cameras`: the position, the ray to the image's
    first pixel's corner and the rays across a pixel row and down a column), its mount tilted up by the
    drone's uptilt and turned by its mount's roll and yaw, its field of view the drone's: HyperDrone's
    `camera_base_one` (hyperdrone/_native/dynamics/impl.cpp) of the mounted camera."""
    airframe = track.airframe
    mount = rotation(quaternion(airframe.mount[:, 0], -airframe.uptilt, airframe.mount[:, 1]))
    axes = jnp.einsum("bij,bjk->bik", rotation(kinematics.orientation), mount)
    forward, left, up = axes[..., 0], axes[..., 1], axes[..., 2]
    scale = 2 * jnp.tan(airframe.fov / 2)[:, None]
    across, down = -left * scale, -up * scale / aspect
    first = forward - 0.5 * across - 0.5 * down
    return jnp.concatenate([kinematics.position, first, across, down], -1)


class FlightBackend(Protocol):
    """A flight simulator for a batch of `count` drones stepped at `hz`, whose commands span
    `thrust_to_weight` g of collective acceleration and `rate_limit` radians a second of body rates, the
    race's nominal drone's with its randomization's headroom (`Racing.spread`). `reset(mask, position,
    orientation)` puts the drones `mask` `[B]` selects at rest at their poses (`[B, 3]`, `[B, 4]`) and
    answers every drone's `Kinematics`; `step(commands)` advances every drone by a control step under
    `[B, 4]` commands in [-1, 1] and answers their `Kinematics`, their paths through it `[B, S + 1, 3]`
    (their positions before it and after each of its `S` physics steps), and whether each collided with
    what the backend itself detects (None for a backend that detects nothing); `render(gates, cameras)`
    answers each drone's image `[B, H, W, C]` uint8 of its `Gates` from its camera `[B, 12]` (`cameras`).
    Arrays cross by DLPack (`dew.rl.simulator.Handed`)."""

    @property
    def count(self) -> int: ...

    @property
    def hz(self) -> float: ...

    @property
    def thrust_to_weight(self) -> float: ...

    @property
    def rate_limit(self) -> float: ...

    @property
    def aspect(self) -> float: ...

    def reset(self, mask: jax.Array, position: jax.Array, orientation: jax.Array) -> Kinematics: ...

    def step(self, commands: jax.Array) -> tuple[Kinematics, Handed, Handed | None]: ...

    def render(self, gates: Gates, cameras: jax.Array) -> Handed: ...


VIEWS = ("image", "state", "gates", "airframe", "passed", "teacher")
"""What a gate race can show: the camera's image, and the privileged views (`privileged`)."""


class GateRace:
    """`backend` racing `racing`'s courses as a `dew.rl.simulator.Simulator`: actions `[B, 4]` in [-1, 1]
    (`commanded`); observations a dict of the `views` asked for (`VIEWS`), each `[B, ...]`. An episode
    restarts before a random gate of a course drawn afresh, at the race's `difficulty`, which a curriculum
    may change between steps."""

    action_size = 4
    _key: jax.Array
    _track: Track

    def __init__(self, backend: FlightBackend, racing: Racing | None = None, views: tuple[str, ...] = VIEWS):
        if not views or not set(views) <= set(VIEWS):
            raise ValueError(f"a gate race shows some of {VIEWS}, not {views}")
        racing = Racing() if racing is None else racing
        fastest = racing.rate_limit * (1 + racing.spread)
        if backend.hz != racing.hz or backend.rate_limit < fastest * (1 - 1e-6):
            raise ValueError(f"a race at {racing.hz} Hz and up to {fastest} rad/s needs a backend stepping "
                             f"at it, not {backend.hz} Hz and {backend.rate_limit} rad/s")
        self.backend, self.racing, self.views, self.count = backend, racing, views, backend.count
        self.difficulty = racing.difficulty
        self._started = jax.jit(lambda key, difficulty: jax.vmap(partial(started, racing), (0, None))(
            jax.random.split(key, self.count), difficulty))
        self._commanded = jax.jit(partial(commanded, racing, thrust_to_weight=backend.thrust_to_weight,
                                          rate_limit=backend.rate_limit))
        self._raced = jax.jit(partial(raced, racing))
        self._privileged = jax.jit(partial(privileged, racing))
        self._cameras = jax.jit(partial(cameras, aspect=backend.aspect))
        self._shown = jax.jit(lambda image, brightness: jnp.clip(
            image.astype(jnp.float32) * brightness[:, None, None, None], 0, 255).astype(jnp.uint8))

    def _observed(self, kinematics: Kinematics) -> dict[str, jax.Array]:
        shown = self._privileged(self._track, kinematics)
        if "image" in self.views:
            track = self._track
            image = received(self.backend.render(Gates(track.course, track.facing, track.count),
                                                 self._cameras(track, kinematics)))
            shown["image"] = self._shown(image, track.airframe.brightness)
        return {view: shown[view] for view in self.views}

    def _drawn(self, mask: jax.Array) -> tuple[Track, Kinematics]:
        """Fresh episodes for the drones `mask` selects, put at their starts: every drone's fresh track and
        every drone's state."""
        self._key, key = jax.random.split(self._key)
        fresh, position, orientation = self._started(key, jnp.asarray(self.difficulty, jnp.float32))
        return fresh, received(self.backend.reset(mask, position, orientation))

    def reset(self, seed: int) -> dict[str, jax.Array]:
        self._key = jax.random.key(seed)
        self._track, kinematics = self._drawn(jnp.ones(self.count, bool))
        return self._observed(kinematics)

    def restart(self, mask: jax.Array) -> dict[str, jax.Array]:
        mask = jnp.asarray(mask, bool)
        fresh, kinematics = self._drawn(mask)
        self._track = chosen(mask, fresh, self._track)
        return self._observed(kinematics)

    def step(self, action: jax.Array) -> TimeStep:
        self._track, command = self._commanded(self._track, action)
        kinematics, path, collided = self.backend.step(command)
        kinematics, path = received(kinematics), received(path)
        collided = jnp.zeros(self.count, bool) if collided is None else received(collided).astype(bool)
        raced = self._raced(self._track, action, kinematics, path, collided)
        self._track, reward, terminated, truncated = raced
        return TimeStep(self._observed(kinematics), reward, terminated, truncated)


__all__ = ["GATES", "LAYOUTS", "VIEWS", "Airframe", "CourseDraw", "FlightBackend", "GateRace", "Gates",
           "Kinematics", "Racing", "Track", "bars", "cameras", "commanded", "corners", "course_draws",
           "course_of", "drawn_airframe", "drawn_track", "facings", "privileged", "quaternion", "raced",
           "rotation", "started", "struck"]
