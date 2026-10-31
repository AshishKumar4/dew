"""A racing quadrotor's physics in JAX, one drone at a time (`vmap` for a batch), every parameter drawn per
episode for transfer from simulation.

The drone is a rigid body under four rotors (Kaufmann, Bauersfeld and Scaramuzza, ICRA 2022, arXiv 2202.10796,
Eq 1-3 and Table II: a 5-inch racer of 0.768 kg, inertia diag(2.5, 2.1, 4.3) g m^2, rotor thrust c_l w^2 with
c_l = 1.563e-6 N s^2, and yaw torque c_d w^2 with c_d = 1.909e-8 N m s^2), its rotors at Flightmare's arm of
0.17 m in an X (uzh-rpg/flightmare flightlib/configs/quadrotor_env.yaml), their order and spins L2F's (front
right, rear right, rear left, front left; yaw torques -, +, -, +; rl_tools rl/environments/l2f/multirotor.h).
At its nominal top speed of 2400 rad/s it lifts 4.8 times its weight (Swift's 4.1, Agilicious's 5.0; every
published 5-inch build is 4 to 5).

Each rotor's speed follows its steady speed through a first-order lag (NeuroBEM 33 ms, Agilicious 39.1 ms,
Ferede et al. arXiv 2504.21586 40 ms); its steady speed is its ESC duty times its top speed times the
battery's voltage over its full charge's (Swift's Methods Eq 6 makes the steady speed a function of voltage
and command; a DC motor's speed is proportional to its voltage). The battery is `cells` LiPo cells: each
cell's open-circuit voltage falls from its charge at the start as the pack discharges, and its internal
resistance sags it under the motors' current (Bauersfeld and Scaramuzza, RA-L 2022, Eq 10-16: a resistance of
at least 4.5 mOhm a cell), the current the rotors' shaft power, sum c_d w^3, over `efficiency` (an assumption:
no source states a 5-inch drive's).

The flight controller is Betaflight's in form (L2F's rate controller, rl/environments/l2f/operations_generic.h
132-190), run every physics step: a PD loop on the body rates to angular accelerations, times the nominal
inertia, then a mixer to each rotor's thrust with the commanded collective, then each rotor's duty from the
nominal thrust map. Its gains, 100 per second and 2 (the derivative on the measured rate's change over a
physics step of 1/240 s), give a 90% rise in 46 to 62 ms with at most 8% overshoot over the motor lags drawn
(a step response computed for this choice; no source publishes a racer's gains in these units).

The air resists the body: rotor drag along the body's x and y, proportional to the rotors' summed speed and
the body's velocity (Ferede et al. Eq 2; SkyDreamer, arXiv 2510.14783, Sec II-D, k = 5.37e-5 N s / m rad), a
quadratic drag along x and y (SkyDreamer Table II: 4.10e-3 and 1.51e-2 N s^2 / m^2), and a linear drag along z
(Kaufmann Table II: 0.15 N s / m). Near the ground the rotors lift more (Cheeseman and Bennett: by 1 / (1 - s
(R / 4z)^2), its strength s from 0.75, Garofano-Soldado et al. 2022's measured, to 1, for 5-inch rotors of R =
63.5 mm). Each episode adds a constant force and moment (SkyDreamer Table III: up to 2 m/s^2 and 2 rad/s^2),
and every control step's duties are noised (SkyDreamer's 0.2 of the normalized command).

Every drawn quantity scales its spread about nominal by `ramp`, from 0 (the nominal drone, no disturbance) to
1, a curriculum. The world frame is x forward, y left, z up; the body frame forward, left, up."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

ARM = 0.17 * 0.5 ** 0.5
"""A rotor's offset along x and y from the centre, metres: Flightmare's 0.17 m arm in an X."""
POSITIONS = ((ARM, -ARM), (-ARM, -ARM), (-ARM, ARM), (ARM, ARM))
"""The rotors' x and y in the body frame: front right, rear right, rear left, front left (L2F's order)."""
SPINS = (-1.0, 1.0, -1.0, 1.0)
"""Each rotor's yaw torque's sign per unit of its drag torque (L2F's)."""
GRAVITY = 9.81
"""The acceleration of gravity, m/s^2."""


@dataclasses.dataclass(frozen=True)
class Quadrotor:
    """The nominal racer and each parameter's randomization (a drawn value is nominal times a factor
    uniform within `1 +- spread`, or uniform in its range, each spread times the ramp): its `mass` (kg),
    `inertia` diagonal (kg m^2), rotor thrust and drag coefficients `thrust_coefficient` and
    `drag_coefficient`, `top_speed` and `idle_speed` (rad/s), `motor_lag` range (s), rotor drag
    `rotor_drag` (N s / m rad), quadratic drag `quadratic_drag` along body x and y (N s^2 / m^2), linear
    drag `vertical_drag` along body z (N s / m), ground effect's `ground_effect` strength range and rotor
    `rotor_radius` (m); the battery's `cells`, each cell's charge at the start `charged` (V) and full
    `full_voltage`, `discharge` (V over the pack's charge), `resistance` range (Ohm a cell), `capacity`
    (Ah) and the drive's `efficiency`; the constant `bias_force` (m/s^2) and `bias_moment` (rad/s^2) at most,
    `duty_noise` at most; the rate controller's `rate_gain` (1/s) and `rate_damping`; and the physics'
    `substeps` a control step."""

    mass: float = 0.768
    mass_spread: float = 0.2
    inertia: tuple[float, float, float] = (2.5e-3, 2.1e-3, 4.3e-3)
    inertia_spread: float = 0.3
    thrust_coefficient: float = 1.563e-6
    drag_coefficient: float = 1.909e-8
    coefficient_spread: float = 0.15
    top_speed: float = 2400.0
    top_spread: float = 0.1
    idle_speed: float = 150.0
    motor_lag: tuple[float, float] = (0.02, 0.06)
    rotor_drag: float = 5.37e-5
    quadratic_drag: tuple[float, float] = (4.10e-3, 1.51e-2)
    vertical_drag: float = 0.15
    drag_spread: float = 0.3
    ground_effect: tuple[float, float] = (0.75, 1.0)
    rotor_radius: float = 0.0635
    cells: int = 4
    charged: tuple[float, float] = (3.7, 4.2)
    full_voltage: float = 4.2
    discharge: float = 0.6
    resistance: tuple[float, float] = (0.003, 0.012)
    capacity: float = 1.5
    efficiency: float = 0.75
    bias_force: float = 2.0
    bias_moment: float = 2.0
    duty_noise: float = 0.2
    rate_gain: float = 100.0
    rate_damping: float = 2.0
    substeps: int = 4

    @property
    def nominal_thrust(self) -> float:
        """The nominal drone's full thrust at full charge, N."""
        return 4 * self.thrust_coefficient * self.top_speed ** 2


@struct.dataclass
class Body:
    """One drone's drawn parameters for an episode (`Quadrotor`'s, each its own), its battery's charge at the
    start and resistance, and its constant disturbance: the `force` (N, world frame) and `moment` (N m,
    body frame)."""

    mass: jax.Array
    inertia: jax.Array
    thrust_coefficient: jax.Array
    drag_coefficient: jax.Array
    top_speed: jax.Array
    motor_lag: jax.Array
    rotor_drag: jax.Array
    quadratic_drag: jax.Array
    vertical_drag: jax.Array
    ground_effect: jax.Array
    charged: jax.Array
    resistance: jax.Array
    force: jax.Array
    moment: jax.Array

    def thrust_to_weight(self, quadrotor: Quadrotor) -> jax.Array:
        """Its full thrust over its weight at its charge at the start."""
        top = self.top_speed * self.charged / quadrotor.full_voltage
        return 4 * self.thrust_coefficient * top ** 2 / (self.mass * GRAVITY)


@struct.dataclass
class Flight:
    """One drone's state: `position` and `velocity` (world frame, m and m/s), `orientation` the body-to-world
    rotation as a unit quaternion (w, x, y, z), `angular_velocity` the body rates (rad/s), each rotor's
    `speed` (rad/s), the charge `drawn` from the battery (Ah), the cell's voltage under the last load (V),
    and the body rates the flight controller last measured, a physics step before."""

    position: jax.Array
    velocity: jax.Array
    orientation: jax.Array
    angular_velocity: jax.Array
    speed: jax.Array
    drawn: jax.Array
    voltage: jax.Array
    measured: jax.Array


def drawn_body(quadrotor: Quadrotor, key: jax.Array, ramp: jax.Array) -> Body:
    """A drone's parameters for an episode (`Body`), each spread about nominal by `ramp`'s share of its
    randomization."""
    keys = iter(jax.random.split(key, 16))

    def factor(spread: float, shape: tuple[int, ...] = ()) -> jax.Array:
        return 1 + ramp * spread * jax.random.uniform(next(keys), shape, minval=-1.0, maxval=1.0)

    def between(low: float, high: float, nominal: float) -> jax.Array:
        drawn = jax.random.uniform(next(keys), (), minval=low, maxval=high)
        return nominal + ramp * (drawn - nominal)

    q = quadrotor
    mass = q.mass * factor(q.mass_spread)
    return Body(mass=mass, inertia=jnp.asarray(q.inertia) * factor(q.inertia_spread, (3,)),
                thrust_coefficient=q.thrust_coefficient * factor(q.coefficient_spread),
                drag_coefficient=q.drag_coefficient * factor(q.coefficient_spread),
                top_speed=q.top_speed * factor(q.top_spread),
                motor_lag=between(*q.motor_lag, nominal=sum(q.motor_lag) / 2),
                rotor_drag=q.rotor_drag * factor(q.drag_spread),
                quadratic_drag=jnp.asarray(q.quadratic_drag) * factor(q.drag_spread, (2,)),
                vertical_drag=q.vertical_drag * factor(q.drag_spread),
                ground_effect=between(*q.ground_effect, nominal=q.ground_effect[1]),
                charged=between(*q.charged, nominal=q.full_voltage),
                resistance=between(*q.resistance, nominal=q.resistance[0]),
                force=mass * ramp * q.bias_force
                * jax.random.uniform(next(keys), (3,), minval=-1.0, maxval=1.0),
                moment=jnp.asarray(q.inertia) * ramp * q.bias_moment
                * jax.random.uniform(next(keys), (3,), minval=-1.0, maxval=1.0))


def rotation(orientation: jax.Array) -> jax.Array:
    """The body-to-world rotation matrix `[3, 3]` of a unit quaternion (w, x, y, z)."""
    w, x, y, z = orientation
    return jnp.asarray([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _mixer(quadrotor: Quadrotor, dtype: jnp.dtype) -> jax.Array:
    """The nominal map `[4, 4]` from (collective thrust, roll, pitch and yaw torques) to each rotor's
    thrust: the inverse of the rotors' thrusts' sums, moments and drag torques, the nominal drone's, a
    constant of the program in `dtype` (inverted once as it traces, not by the device each step)."""
    kappa = quadrotor.drag_coefficient / quadrotor.thrust_coefficient
    rows = np.asarray([[1.0] * 4, [y for _, y in POSITIONS], [-x for x, _ in POSITIONS],
                       [spin * kappa for spin in SPINS]])
    return jnp.asarray(np.linalg.inv(rows), dtype)


def hovering(quadrotor: Quadrotor, body: Body, position: jax.Array, orientation: jax.Array) -> Flight:
    """A drone at rest at `position` and `orientation`, its rotors at the speed that holds its weight, its
    battery as charged."""
    hover = jnp.sqrt(body.mass * GRAVITY / (4 * body.thrust_coefficient))
    zero = jnp.zeros(3, position.dtype)
    return Flight(position=position, velocity=zero, orientation=orientation, angular_velocity=zero,
                  speed=jnp.full(4, hover, position.dtype), drawn=jnp.zeros((), position.dtype),
                  voltage=jnp.asarray(body.charged, position.dtype), measured=zero)


def duties(quadrotor: Quadrotor, flight: Flight, collective: jax.Array, rates: jax.Array,
           dt: float) -> tuple[jax.Array, jax.Array]:
    """The flight controller's ESC duties `[4]` in [0, 1] for a `collective` thrust (N) and body `rates`
    (rad/s), and the rates it measured: the rate loop's angular accelerations times the nominal inertia,
    mixed with the collective into each rotor's thrust, each turned into a duty by the nominal thrust map
    at full charge."""
    q = quadrotor
    measured = flight.angular_velocity
    wanted = q.rate_gain * (rates - measured) - q.rate_damping * (measured - flight.measured) / dt
    torque = jnp.asarray(q.inertia) * wanted
    wrench = jnp.concatenate([collective[None], torque])
    thrusts = _mixer(q, wrench.dtype) @ wrench
    speed = jnp.sqrt(jnp.maximum(thrusts, 0.0) / q.thrust_coefficient)
    return jnp.clip((speed - q.idle_speed) / (q.top_speed - q.idle_speed), 0.0, 1.0), measured


def _accelerations(quadrotor: Quadrotor, body: Body, flight: Flight) -> tuple[jax.Array, jax.Array]:
    """The linear (world frame) and angular (body frame) accelerations of a drone at `flight`."""
    turned = rotation(flight.orientation)
    lift = body.thrust_coefficient * jnp.square(flight.speed)
    height = jnp.maximum(flight.position[2], quadrotor.rotor_radius / 2)
    ground = 1 / (1 - body.ground_effect * jnp.square(quadrotor.rotor_radius / (4 * height)))
    moving = turned.T @ flight.velocity
    drag = -jnp.concatenate([body.rotor_drag * jnp.sum(flight.speed) * moving[:2]
                             + body.quadratic_drag * moving[:2] * jnp.abs(moving[:2]),
                             body.vertical_drag * moving[2:]])
    thrust = jnp.asarray([0.0, 0.0, 1.0]) * jnp.sum(lift) * ground
    weight = jnp.asarray([0.0, 0.0, -GRAVITY]) * body.mass
    linear = (turned @ (thrust + drag) + weight + body.force) / body.mass
    positions = jnp.asarray(POSITIONS)
    torque = jnp.stack([positions[:, 1] @ lift, -positions[:, 0] @ lift,
                        jnp.asarray(SPINS) @ (body.drag_coefficient * jnp.square(flight.speed))])
    torque = torque + body.moment
    rates = flight.angular_velocity
    angular = (torque - jnp.cross(rates, body.inertia * rates)) / body.inertia
    return linear, angular


def _turned(orientation: jax.Array, rates: jax.Array, dt: float) -> jax.Array:
    """`orientation` turned by body `rates` held for `dt` (the exponential map), renormalized."""
    angle = jnp.linalg.norm(rates) * dt
    axis = rates / jnp.maximum(jnp.linalg.norm(rates), 1e-12)
    half = jnp.concatenate([jnp.cos(angle / 2)[None], jnp.sin(angle / 2) * axis])
    w0, x0, y0, z0 = orientation
    w1, x1, y1, z1 = half
    product = jnp.stack([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1, w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                         w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1, w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1])
    return product / jnp.linalg.norm(product)


def flown(quadrotor: Quadrotor, body: Body, flight: Flight, collective: jax.Array, rates: jax.Array,
          key: jax.Array, hz: float, ramp: jax.Array) -> tuple[Flight, jax.Array]:
    """A drone's state after a control step of `1 / hz` seconds under a `collective` thrust (N) and body
    `rates` (rad/s), and its path `[substeps + 1, 3]` (its positions before the step and after each physics
    step). Each physics step the flight controller sets the duties (`duties`, Betaflight's loop runs far
    faster than a policy), noised by `ramp`'s share of `duty_noise` drawn once a control step; the rotors
    move toward their steady speeds, the body flies by semi-implicit Euler and the battery drains."""
    q = quadrotor
    dt = 1 / (hz * q.substeps)
    noise = ramp * q.duty_noise * jax.random.uniform(key, (4,), minval=-1.0, maxval=1.0)
    decay = jnp.exp(-dt / body.motor_lag)

    def physics(flight: Flight, _) -> tuple[Flight, jax.Array]:
        commanded, measured = duties(q, flight, collective, rates, dt)
        top = body.top_speed * flight.voltage / q.full_voltage
        steady = q.idle_speed + jnp.clip(commanded + noise, 0.0, 1.0) * (top - q.idle_speed)
        speed = steady + (flight.speed - steady) * decay
        flight = dataclasses.replace(flight, speed=speed, measured=measured)
        linear, angular = _accelerations(q, body, flight)
        velocity = flight.velocity + linear * dt
        rates_now = flight.angular_velocity + angular * dt
        power = jnp.sum(body.drag_coefficient * speed ** 3) / q.efficiency
        current = power / (q.cells * flight.voltage)
        drawn = flight.drawn + current * dt / 3600
        open_circuit = body.charged - q.discharge * drawn / q.capacity
        voltage = jnp.maximum(open_circuit - body.resistance * current, 0.1)
        flight = dataclasses.replace(flight, position=flight.position + velocity * dt, velocity=velocity,
                                     orientation=_turned(flight.orientation, rates_now, dt),
                                     angular_velocity=rates_now, drawn=drawn, voltage=voltage)
        return flight, flight.position

    start = flight.position
    flight, path = jax.lax.scan(physics, flight, None, length=q.substeps)
    return flight, jnp.concatenate([start[None], path])


__all__ = ["ARM", "GRAVITY", "POSITIONS", "SPINS", "Body", "Flight", "Quadrotor", "drawn_body", "duties",
           "flown", "hovering", "rotation"]
