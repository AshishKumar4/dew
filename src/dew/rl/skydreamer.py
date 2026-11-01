"""SkyDreamer's race (Verraest et al., arXiv 2510.14783), a `dew.rl.env.Env` in JAX: a quadcopter flown by
four motor commands at 90 Hz through a track's gates, seen through a binary mask of the gates.

The quadcopter is SkyDreamer's model (Sec II-D, Eq 3-6, Table II), in its frames: the world North-East-Down,
the body Forward-Right-Down. Its state is the position and velocity in the world, the orientation (a unit
quaternion, body to world), the body rates and the four motors' speeds (rad/s). Its specific forces are drag
along the body's x and y, linear in the rotors' summed speed and quadratic in the velocity, and thrust along
-z, `k_w` times the summed squared speeds, grown by the propeller's angle of attack and advance ratio over the
propeller radius `radius` (MonoRace, arXiv 2601.15222, Eq 6: 0.0485775 m, which SkyDreamer's Table II leaves
out). Its angular accelerations are each motor's squared speed weighted per axis, the yaw's linear in the
speeds and their changes, plus the gyroscopic terms. Each motor moves toward its steady speed for its command
with time constant `tau`. The model is integrated by fourth-order Runge-Kutta in `substeps` steps a control
step (SkyDreamer's 2.2 ms; 5 steps of 2.22 ms at 90 Hz).

The policy's inputs (Sec II-B, II-E, II-H): a 64 x 64 mask of the gates' frames, cast from the camera
(`masked`: focal length 25 pixels at 64, Eq 7, so 104 degrees across; tilted up by `uptilt`), the body rates,
the motor speeds, and the flight plan (`plan`, Eq 12): for the next three gates, the change in position and
yaw from the gate before, then their positions and yaws. A gate is passed when the drone crosses its plane,
and the plan then moves on. The privileged entries a world model decodes (`state`, Sec II-B) are the drone's
position and velocity in the world and in the gate headed for, and its orientation; with randomization off,
the camera's extrinsics and the dynamics' parameters are constants and left out.

The reward (Sec II-C, Eq 1-2) is 5 progress - rate + 30 gate. Progress is the step's approach to the next
pre-gate, zero from the pre-gate through the post-gate. The rate penalty is (exp(min(|rates|_1, 17)) - 1)
over 2 f_c 1e5. The gate reward, at each of a gate's pre-gate, gate and post-gate planes (`depth` / 2 before
and after it), is 1 - max(|y|, |z|) / `size` where the drone crosses. An episode ends with a final reward of
zero when the drone crosses one of those planes outside `size`, or when it is within 0.5 m of the ground
while descending faster than 1 m/s or rolled or pitched past 60 degrees. An episode starts (Table III) 2 to
4 m before a gate (any gate in 70% of starts, the first in the rest), at rest, slightly turned, its motors
between a quarter and half of their top speed.

Not SkyDreamer's: the track. Its tracks are given only in figures, so `CIRCLE` stands four of its gates on a
circle in its 8 x 8 m arena. Randomization is off (the owner's baseline): no parameter spread, no
disturbances, no motor-command noise, no mask augmentation, and the flight plan moves on exactly at a gate;
its camera is tilted up 30 degrees, fixed, where SkyDreamer's is 45 to 55.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from dew.rl.env import TimeStep
from dew.rl.quadrotor import rotation

GRAVITY = 9.81
"""The acceleration of gravity, m/s^2, along the world's z (down)."""
CIRCLE = (((3.0, 0.0, -1.5), 90.0), ((0.0, 3.0, -1.5), 180.0), ((-3.0, 0.0, -1.5), 270.0),
          ((0.0, -3.0, -1.5), 0.0))
"""Four gates 1.5 m up on a circle of 3 m about the arena's centre, each its centre (NED) and yaw (degrees),
facing along the circle: a closed track."""
VIEWS = ("mask", "rates", "motors", "plan", "state", "passed")
"""What the race shows: the gates' mask, the body rates and motor speeds, the flight plan, and privileged,
the `state` and the gates `passed`."""
PLANES = (-1, 0, 1)
"""The planes a gate is passed through, in its depths' halves along its normal: pre-gate, gate, post-gate."""


@dataclasses.dataclass(frozen=True)
class Sky:
    """SkyDreamer's race (the module docstring): its quadcopter's parameters (Table II; `radius`, MonoRace's
    propeller radius), its control rate `hz` and integration `substeps`, its gates (`track`: each centre in
    NED and yaw in degrees, the track closed), each `inner` and `outer` wide (its orange gates, Sec III-A),
    its gate rewards' `size` and the pre- and post-gates' span `depth` (Table III), its camera's `uptilt`
    (degrees) and focal length `focal` in pixels of its `pixels`-square mask, and the share of starts before
    any gate (`anywhere`, 70%)."""

    k_w: float = 1.55e-6
    k_x: float = 5.37e-5
    k_y: float = 5.37e-5
    k_x2: float = 4.10e-3
    k_y2: float = 1.51e-2
    k_angle: float = 3.145
    k_hor: float = 7.245
    k_v2: float = 0.0
    gyro: tuple[float, float, float] = (-0.89, 0.96, -0.34)
    omega_min: float = 341.75
    omega_max: float = 3100.0
    k: float = 0.5
    tau: float = 0.03
    k_p: tuple[float, float, float, float] = (4.99e-5, 3.78e-5, 4.82e-5, 3.83e-5)
    k_q: tuple[float, float, float, float] = (2.05e-5, 2.46e-5, 2.02e-5, 2.57e-5)
    k_r: float = 3.38e-3
    k_r_dot: float = 3.24e-4
    radius: float = 0.0485775
    hz: float = 90.0
    substeps: int = 5
    track: tuple[tuple[tuple[float, float, float], float], ...] = CIRCLE
    inner: float = 1.5
    outer: float = 2.7
    size: float = 0.8
    depth: float = 0.8
    uptilt: float = 30.0
    focal: float = 25.0
    pixels: int = 64
    anywhere: float = 0.7

    @property
    def centers(self) -> np.ndarray:
        """The gates' centres `[G, 3]` (NED)."""
        return np.asarray([center for center, _ in self.track], np.float32)

    @property
    def yaws(self) -> np.ndarray:
        """The gates' yaws `[G]` (radians)."""
        return np.deg2rad(np.asarray([yaw for _, yaw in self.track], np.float32))


@struct.dataclass
class Craft:
    """The quadcopter's state: position and velocity (NED world), orientation (body to world), body rates
    (FRD) and motor speeds `[4]` (rad/s)."""

    position: jax.Array
    velocity: jax.Array
    orientation: jax.Array
    rates: jax.Array
    motors: jax.Array


@struct.dataclass
class Flight:
    """One drone's episode: its `craft`, the gate it heads for (`index`) and the plane of it it crosses next
    (`stage`, into `PLANES`), its distance to that gate's pre-gate after the last step, and the gates
    passed."""

    craft: Craft
    index: jax.Array
    stage: jax.Array
    distance: jax.Array
    passed: jax.Array


def _product(a: jax.Array, b: jax.Array) -> jax.Array:
    """The Hamilton product of quaternions (w, x, y, z)."""
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return jnp.stack([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1, w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                      w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1, w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1])


def euler_quaternion(roll: jax.Array, pitch: jax.Array, yaw: jax.Array) -> jax.Array:
    """The orientation of yaw, then pitch, then roll (Z-Y-X), as a unit quaternion."""
    cr, sr, cp, sp = jnp.cos(roll / 2), jnp.sin(roll / 2), jnp.cos(pitch / 2), jnp.sin(pitch / 2)
    cy, sy = jnp.cos(yaw / 2), jnp.sin(yaw / 2)
    return jnp.stack([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                      cr * cp * sy - sr * sp * cy])


def roll_pitch(orientation: jax.Array) -> tuple[jax.Array, jax.Array]:
    """An orientation's roll and pitch (Z-Y-X Euler angles)."""
    w, x, y, z = orientation
    roll = jnp.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return roll, jnp.arcsin(jnp.clip(2 * (w * y - z * x), -1.0, 1.0))


def commanded(sky: Sky, commands: jax.Array) -> jax.Array:
    """Each motor's steady speed for its command in [0, 1] (Eq 6)."""
    command = jnp.clip(commands, 0.0, 1.0)
    span = sky.omega_max - sky.omega_min
    return span * jnp.sqrt(sky.k * command ** 2 + (1 - sky.k) * command) + sky.omega_min


def derivative(sky: Sky, craft: Craft, commands: jax.Array) -> Craft:
    """The state's rate of change under motor commands `[4]` (Eq 3-6)."""
    turned = rotation(craft.orientation)
    v = turned.T @ craft.velocity
    motors, total = craft.motors, jnp.sum(craft.motors)
    squared = jnp.square(motors)
    alpha = jnp.arctan(v[2] / (sky.radius * total))
    mu = jnp.arctan((v[0] ** 2 + v[1] ** 2) / (sky.radius * total))
    force = jnp.stack([-sky.k_x * v[0] * total - sky.k_x2 * v[0] * jnp.abs(v[0]),
                       -sky.k_y * v[1] * total - sky.k_y2 * v[1] * jnp.abs(v[1]),
                       -sky.k_w * (1 + sky.k_angle * alpha + sky.k_hor * mu) * jnp.sum(squared)
                       - sky.k_v2 * v[2] * jnp.abs(v[2])])
    accelerating = (commanded(sky, commands) - motors) / sky.tau
    p, q, r = craft.rates
    k_p, k_q = jnp.asarray(sky.k_p), jnp.asarray(sky.k_q)
    spin = jnp.asarray([-1.0, 1.0, 1.0, -1.0])
    moment = jnp.stack([jnp.asarray([-1.0, -1.0, 1.0, 1.0]) @ (k_p * squared) + sky.gyro[0] * q * r,
                        jnp.asarray([-1.0, 1.0, -1.0, 1.0]) @ (k_q * squared) + sky.gyro[1] * p * r,
                        spin @ (sky.k_r * motors + sky.k_r_dot * accelerating) + sky.gyro[2] * p * q])
    turning = 0.5 * _product(craft.orientation, jnp.concatenate([jnp.zeros(1), craft.rates]))
    return Craft(position=craft.velocity, velocity=turned @ force + jnp.asarray([0.0, 0.0, GRAVITY]),
                 orientation=turning, rates=moment, motors=accelerating)


def flown(sky: Sky, craft: Craft, commands: jax.Array) -> Craft:
    """The state after a control step under motor commands held through it: `substeps` steps of
    fourth-order Runge-Kutta, the orientation renormalized after each."""
    dt = 1 / (sky.hz * sky.substeps)

    def moved(craft: Craft, rate: Craft, by: float) -> Craft:
        return jax.tree.map(lambda x, dx: x + by * dx, craft, rate)

    def step(craft: Craft, _) -> tuple[Craft, None]:
        k1 = derivative(sky, craft, commands)
        k2 = derivative(sky, moved(craft, k1, dt / 2), commands)
        k3 = derivative(sky, moved(craft, k2, dt / 2), commands)
        k4 = derivative(sky, moved(craft, k3, dt), commands)
        rate = jax.tree.map(lambda a, b, c, d: (a + 2 * b + 2 * c + d) / 6, k1, k2, k3, k4)
        craft = moved(craft, rate, dt)
        return craft.replace(orientation=craft.orientation / jnp.linalg.norm(craft.orientation)), None

    return jax.lax.scan(step, craft, None, length=sky.substeps)[0]


def _axes(yaw: jax.Array) -> tuple[jax.Array, jax.Array]:
    """A gate's normal (its frame's x, along the flight) and its frame's y, in the world (NED)."""
    zero = jnp.zeros_like(yaw)
    cos, sin = jnp.cos(yaw), jnp.sin(yaw)
    return jnp.stack([cos, sin, zero], -1), jnp.stack([-sin, cos, zero], -1)


def in_gate(sky: Sky, index: jax.Array, point: jax.Array) -> jax.Array:
    """`point` in gate `index`'s frame: along its normal, to its right and down from its centre."""
    normal, right = _axes(jnp.asarray(sky.yaws)[index])
    offset = point - jnp.asarray(sky.centers)[index]
    return jnp.stack([offset @ normal, offset @ right, offset[2]])


def _pre_gate(sky: Sky, index: jax.Array) -> jax.Array:
    normal, _ = _axes(jnp.asarray(sky.yaws)[index])
    return jnp.asarray(sky.centers)[index] - sky.depth / 2 * normal


def masked(sky: Sky, craft: Craft) -> jax.Array:
    """The gates' binary mask `[P, P, 1]` uint8 (255 on a gate's frame), cast from the camera through each
    pixel's centre: a gate's frame is the square annulus between its `inner` and `outer` widths, and every
    gate on the track is drawn; nothing else is, so a pixel is a gate's wherever its ray meets any frame."""
    angle = math.radians(sky.uptilt)
    forward = np.asarray([math.cos(angle), 0.0, -math.sin(angle)])
    right, down = np.asarray([0.0, 1.0, 0.0]), np.asarray([math.sin(angle), 0.0, math.cos(angle)])
    offsets = (np.arange(sky.pixels) + 0.5 - sky.pixels / 2) / sky.focal
    rays = (forward + offsets[None, :, None] * right + offsets[:, None, None] * down).astype(np.float32)
    rays = jnp.asarray(rays) @ rotation(craft.orientation).T
    normal, side = _axes(jnp.asarray(sky.yaws))
    toward = jnp.asarray(sky.centers) - craft.position
    along = rays @ normal.T
    distance = jnp.where(jnp.abs(along) > 1e-9, (toward * normal).sum(-1) / along, -1.0)
    hit = rays[..., None, :] * distance[..., None] - toward
    across = jnp.maximum(jnp.abs((hit * side).sum(-1)), jnp.abs(hit[..., 2]))
    framed = (distance > 0) & (across >= sky.inner / 2) & (across <= sky.outer / 2)
    return (jnp.any(framed, -1)[..., None] * 255).astype(jnp.uint8)


def plan(sky: Sky, index: jax.Array) -> jax.Array:
    """The flight plan `[24]` at gate `index` (Eq 12): for it and the two after it, the change in position
    and yaw from the gate before, then their positions and yaws."""
    count = len(sky.track)
    gates = (index + jnp.arange(-1, 3)) % count
    centers, yaws = jnp.asarray(sky.centers)[gates], jnp.asarray(sky.yaws)[gates]
    change = yaws[1:] - yaws[:-1]
    turned = jnp.arctan2(jnp.sin(change), jnp.cos(change))
    changes = jnp.concatenate([centers[1:] - centers[:-1], turned[:, None]], -1).reshape(-1)
    return jnp.concatenate([changes, jnp.concatenate([centers[1:], yaws[1:, None]], -1).reshape(-1)])


@struct.dataclass
class SkyRace:
    """One drone racing `sky`'s track as a `dew.rl.env.Env`: actions `[4]` in [-1, 1], each motor's command
    in [0, 1] as (a + 1) / 2; observations a dict of the `views` asked for (`VIEWS`). Its episodes are cut by
    a `TimeLimit` the run gives it."""

    sky: Sky = struct.field(pytree_node=False, default_factory=Sky)
    views: tuple[str, ...] = struct.field(pytree_node=False, default=VIEWS)

    action_size = 4

    def __post_init__(self):
        if not self.views or not set(self.views) <= set(VIEWS):
            raise ValueError(f"a SkyDreamer race shows some of {VIEWS}, not {self.views}")

    def observed(self, flight: Flight) -> dict[str, jax.Array]:
        """The views of `flight` asked for."""
        sky, craft = self.sky, flight.craft
        shown = {"rates": craft.rates, "motors": craft.motors, "passed": flight.passed}
        if "mask" in self.views:
            shown["mask"] = masked(sky, craft)
        if "plan" in self.views:
            shown["plan"] = plan(sky, flight.index)
        if "state" in self.views:
            normal, right = _axes(jnp.asarray(sky.yaws)[flight.index])
            down = jnp.asarray([0.0, 0.0, 1.0])
            gate_velocity = jnp.stack([craft.velocity @ axis for axis in (normal, right, down)])
            orientation = craft.orientation * jnp.where(craft.orientation[0] < 0, -1.0, 1.0)
            shown["state"] = jnp.concatenate([craft.position, in_gate(sky, flight.index, craft.position),
                                              craft.velocity, gate_velocity, orientation])
        return {view: shown[view] for view in self.views}

    def reset(self, key: jax.Array) -> tuple[Flight, TimeStep]:
        """A start of Table III's: before any gate (`anywhere` of starts) or the first, 2 to 4 m before it,
        within 1 m to its side and up to 1.3 m below its centre (0.7 m below, before the first), at rest,
        rolled, pitched and turned from its yaw up to 20 degrees, its rates up to 0.1 rad/s and its motors at
        a quarter to half of their top speed."""
        sky = self.sky
        keys = iter(jax.random.split(key, 8))
        count = len(sky.track)
        anywhere = jax.random.uniform(next(keys)) < sky.anywhere
        index = jnp.where(anywhere, jax.random.randint(next(keys), (), 0, count), 0)
        low = jnp.where(anywhere, 0.0, 0.7)
        offset = jnp.stack([jax.random.uniform(next(keys), (), minval=-4.0, maxval=-2.0),
                            jax.random.uniform(next(keys), (), minval=-1.0, maxval=1.0),
                            jax.random.uniform(next(keys), (), minval=low, maxval=1.3)])
        normal, right = _axes(jnp.asarray(sky.yaws)[index])
        position = (jnp.asarray(sky.centers)[index] + offset[0] * normal + offset[1] * right
                    + offset[2] * jnp.asarray([0.0, 0.0, 1.0]))
        turned = jax.random.uniform(next(keys), (3,), minval=-math.pi / 9, maxval=math.pi / 9)
        orientation = euler_quaternion(turned[0], turned[1], jnp.asarray(sky.yaws)[index] + turned[2])
        craft = Craft(position=position, velocity=jnp.zeros(3), orientation=orientation,
                      rates=jax.random.uniform(next(keys), (3,), minval=-0.1, maxval=0.1),
                      motors=sky.omega_max * jax.random.uniform(next(keys), (4,), minval=0.25, maxval=0.5))
        flight = Flight(craft=craft, index=index, stage=jnp.zeros((), jnp.int32),
                        distance=jnp.linalg.norm(position - _pre_gate(sky, index)),
                        passed=jnp.zeros((), jnp.int32))
        zero = jnp.zeros((), bool)
        return flight, TimeStep(self.observed(flight), jnp.zeros(()), zero, zero)

    def step(self, flight: Flight, action: jax.Array) -> tuple[Flight, TimeStep]:
        """The drone flown a control step (`flown`), then the planes it crossed in order, the reward and
        whether the episode ended (the module docstring)."""
        sky = self.sky
        before = flight.craft.position
        craft = flown(sky, flight.craft, (jnp.clip(action, -1, 1) + 1) / 2)
        after = craft.position
        index, stage, approaching = flight.index, flight.stage, flight.stage == 0
        crossing, collided = jnp.ones((), bool), jnp.zeros((), bool)
        gate, passed = jnp.zeros(()), flight.passed
        for _ in PLANES:
            # The next plane: crossed this step where the path meets it from behind.
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
        approached = flight.distance - jnp.linalg.norm(after - _pre_gate(sky, flight.index))
        progress = jnp.where(approaching, approached, 0.0)
        distance = jnp.linalg.norm(after - _pre_gate(sky, index))
        rate = jnp.expm1(jnp.minimum(jnp.sum(jnp.abs(craft.rates)), 17.0)) / (2 * sky.hz * 1e5)
        roll, pitch = roll_pitch(craft.orientation)
        grounded = (after[2] > -0.5) & ((craft.velocity[2] > 1.0) | (jnp.abs(roll) > math.pi / 3)
                                        | (jnp.abs(pitch) > math.pi / 3))
        ended = collided | grounded
        reward = jnp.where(ended, 0.0, 5 * progress - rate + 30 * gate)
        flight = Flight(craft=craft, index=index, stage=stage, distance=distance, passed=passed)
        return flight, TimeStep(self.observed(flight), reward, ended, jnp.zeros((), bool))


__all__ = ["CIRCLE", "GRAVITY", "PLANES", "VIEWS", "Craft", "Flight", "Sky", "SkyRace", "commanded",
           "derivative", "euler_quaternion", "flown", "in_gate", "masked", "plan", "roll_pitch"]
