"""A racing quadrotor's physics (`dew.rl.quadrotor`): a control step of drawn drones under drawn commands held
to a float64 NumPy transcription of the physics, written from the sources apart from Dew's code (the rigid
body, the rotors' lag and the battery's voltage, the drag, the ground effect, the disturbances and the flight
controller); a nominal drone holding its hover, an equilibrium of that physics; and a drone's drawn
parameters about nominal."""

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import assert_computes_the_oracle, chain_roundings

from dew.rl.quadrotor import Body, Flight, Quadrotor, drawn_body, flown, hovering

QUADROTOR = Quadrotor()
HZ = 60.0
NOMINAL_RATIO = 4 * 1.563e-6 * 2400 ** 2 / (0.768 * 9.81)
"""The nominal racer's thrust over its weight at full charge, from Kaufmann et al.'s Table II: four rotors of
1.563e-6 N s^2 at 2400 rad/s under 0.768 kg."""


def quaternion_matrix(q: np.ndarray) -> np.ndarray:
    """A unit quaternion's (w, x, y, z) rotation matrix, body to world."""
    w, x, y, z = q
    return np.array([[w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z]])


def oracle_step(q: Quadrotor, body: dict, flight: dict, collective: float, rates: np.ndarray,
                noise: np.ndarray, hz: float) -> tuple[dict, np.ndarray]:
    """One control step of one drone in float64 loops: every physics step, the controller's PD on the rates,
    the X mixer's inverse, the duties from the nominal thrust map, the rotors' first-order lag to the duty's
    speed at the battery's voltage, the forces (thrust with the ground effect; rotor, quadratic and vertical
    drag in the body frame; gravity; the bias force) and moments (the rotors' arms, their drag torques, the
    bias moment, the gyroscopic term), semi-implicit Euler, the exponential map, and the battery's charge
    drawn and voltage."""
    arm = 0.17 / np.sqrt(2)
    xs, ys = np.array([arm, -arm, -arm, arm]), np.array([-arm, -arm, arm, arm])
    spins = np.array([-1.0, 1.0, -1.0, 1.0])
    kappa = q.drag_coefficient / q.thrust_coefficient
    allocation = np.array([np.ones(4), ys, -xs, spins * kappa])
    dt = 1 / (hz * q.substeps)
    flight = {name: np.array(value, np.float64) for name, value in flight.items()}
    path = [flight["position"].copy()]
    for _ in range(q.substeps):
        omega, previous = flight["angular_velocity"], flight["measured"]
        accel = q.rate_gain * (rates - omega) - q.rate_damping * (omega - previous) / dt
        torque = np.array(q.inertia) * accel
        thrusts = np.linalg.solve(allocation, np.r_[collective, torque])
        duty = (np.sqrt(np.maximum(thrusts, 0) / q.thrust_coefficient) - q.idle_speed) \
            / (q.top_speed - q.idle_speed)
        duty = np.clip(np.clip(duty, 0, 1) + noise, 0, 1)
        top = body["top_speed"] * flight["voltage"] / q.full_voltage
        steady = q.idle_speed + duty * (top - q.idle_speed)
        speed = steady + (flight["speed"] - steady) * np.exp(-dt / body["motor_lag"])
        lift = body["thrust_coefficient"] * speed ** 2
        height = max(flight["position"][2], q.rotor_radius / 2)
        ground = 1 / (1 - body["ground_effect"] * (q.rotor_radius / (4 * height)) ** 2)
        turned = quaternion_matrix(flight["orientation"])
        v_body = turned.T @ flight["velocity"]
        drag = np.zeros(3)
        for axis in range(2):
            drag[axis] = -(body["rotor_drag"] * speed.sum() * v_body[axis]
                           + body["quadratic_drag"][axis] * v_body[axis] * abs(v_body[axis]))
        drag[2] = -body["vertical_drag"] * v_body[2]
        weight = np.array([0, 0, -9.81 * body["mass"]])
        force = turned @ (np.array([0, 0, lift.sum() * ground]) + drag) + weight + body["force"]
        linear = force / body["mass"]
        yaw = spins @ (body["drag_coefficient"] * speed ** 2)
        moment = np.array([ys @ lift, -(xs @ lift), yaw]) + body["moment"]
        inertia = body["inertia"]
        angular = (moment - np.cross(omega, inertia * omega)) / inertia
        velocity = flight["velocity"] + linear * dt
        new_rates = omega + angular * dt
        angle = np.linalg.norm(new_rates) * dt
        axis = new_rates / max(np.linalg.norm(new_rates), 1e-12)
        w1, x1, y1, z1 = np.cos(angle / 2), *(np.sin(angle / 2) * axis)
        w0, x0, y0, z0 = flight["orientation"]
        turned_q = np.array([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1, w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                             w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1, w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1])
        power = (body["drag_coefficient"] * speed ** 3).sum() / q.efficiency
        current = power / (q.cells * flight["voltage"])
        drawn = flight["drawn"] + current * dt / 3600
        voltage = max(body["charged"] - q.discharge * drawn / q.capacity - body["resistance"] * current, 0.1)
        flight.update(position=flight["position"] + velocity * dt, velocity=velocity,
                      orientation=turned_q / np.linalg.norm(turned_q), angular_velocity=new_rates,
                      speed=speed, drawn=drawn, voltage=voltage, measured=omega)
        path.append(flight["position"].copy())
    return flight, np.stack(path)


def drawn_case(seed: int, count: int):
    """`count` drones drawn at full randomization, each mid-flight (moving, turned, rotors and battery
    away from rest), with a collective and rates to fly."""
    rng = np.random.default_rng([532, 20, seed])
    keys = jax.random.split(jax.random.key(seed), count)
    bodies = jax.vmap(lambda key: drawn_body(QUADROTOR, key, jnp.asarray(1.0)))(keys)
    orientation = rng.normal(size=(count, 4))
    orientation /= np.linalg.norm(orientation, axis=-1, keepdims=True)
    flights = Flight(position=np.c_[rng.uniform(-5, 5, (count, 2)), rng.uniform(0.05, 4, count)],
                     velocity=rng.normal(0, 6, (count, 3)), orientation=orientation,
                     angular_velocity=rng.normal(0, 3, (count, 3)), speed=rng.uniform(300, 2400, (count, 4)),
                     drawn=rng.uniform(0, 0.3, count), voltage=rng.uniform(3.4, 4.2, count),
                     measured=rng.normal(0, 3, (count, 3)))
    collective = rng.uniform(0, 30, count)
    rates = rng.uniform(-10, 10, (count, 3))
    return bodies, flights, collective, rates


def test_a_control_step_is_the_physics_written_out():
    """512 drones drawn at full randomization, mid-flight under drawn commands, each flown a control step:
    every state and path position the float64 transcription's (`oracle_step`), to float64 rounding."""
    with jax.enable_x64(new_val=True):
        bodies, flights, collective, rates = drawn_case(0, 512)
        bodies = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), bodies)
        flights = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), flights)
        keys = jax.random.split(jax.random.key(9), 512)
        ramp = jnp.asarray(1.0, jnp.float64)

        def step(body, flight, collective, rates, key):
            return flown(QUADROTOR, body, flight, collective, rates, key, HZ, ramp)

        stepped = jax.jit(jax.vmap(step))
        inputs = (bodies, flights, jnp.asarray(collective), jnp.asarray(rates), keys)
        computed_flight, computed_path = stepped(*inputs)
        noise = np.asarray(jax.vmap(lambda key: QUADROTOR.duty_noise * jax.random.uniform(
            key, (4,), minval=-1.0, maxval=1.0, dtype=jnp.float64))(keys))
        expected_paths, expected = [], {name: [] for name in Flight.__dataclass_fields__}
        for drone in range(512):
            body = {name: np.asarray(getattr(bodies, name)[drone]) for name in Body.__dataclass_fields__}
            flight = {name: np.asarray(getattr(flights, name)[drone]) for name in Flight.__dataclass_fields__}
            after, path = oracle_step(QUADROTOR, body, flight, collective[drone], rates[drone], noise[drone],
                                      HZ)
            expected_paths.append(path)
            for name in expected:
                expected[name].append(after[name])
        roundings = chain_roundings(jax.make_jaxpr(stepped)(*inputs))
        assert_computes_the_oracle(np.asarray(computed_path), np.stack(expected_paths), "a step's paths",
                                   roundings=roundings)
        for name, values in expected.items():
            assert_computes_the_oracle(np.asarray(getattr(computed_flight, name)), np.stack(values),
                                       f"a step's {name}", roundings=roundings)


def nominal(position=(0.0, 0.0, 2.0), quadrotor: Quadrotor = QUADROTOR) -> tuple[Body, Flight]:
    """The nominal drone, undisturbed, at rest at `position`, level."""
    body = drawn_body(quadrotor, jax.random.key(0), jnp.asarray(0.0))
    return body, hovering(quadrotor, body, jnp.asarray(position), jnp.asarray([1.0, 0.0, 0.0, 0.0]))


def test_a_nominal_drone_holds_its_hover():
    """The nominal drone at full charge, where there is no ground effect and its battery holds its voltage (no
    resistance, no discharge), at rest and level two metres up, commanded its weight and no rates for two
    seconds in float64: the flight controller's duties give each rotor its share of the weight, so its
    position, velocity, orientation, body rates and rotors' speeds stay as they started, to float64
    rounding."""
    steady = Quadrotor(resistance=(0.0, 0.0), discharge=0.0, ground_effect=(0.0, 0.0))
    with jax.enable_x64(new_val=True):
        body, start = nominal(quadrotor=steady)
        weight = body.mass * 9.81

        def hover(flight):
            def step(flight, key):
                return flown(steady, body, flight, weight, jnp.zeros(3), key, HZ, jnp.asarray(0.0))[0], None

            return jax.lax.scan(step, flight, jax.random.split(jax.random.key(1), int(2 * HZ)))[0]

        held = jax.jit(hover)(start)
        roundings = chain_roundings(jax.make_jaxpr(hover)(start))

    def stacked(flight: Flight, names: tuple[str, ...]) -> np.ndarray:
        return np.concatenate([np.ravel(getattr(flight, name)) for name in names])

    for names in (("position", "velocity"), ("orientation", "angular_velocity", "measured"), ("speed",)):
        assert_computes_the_oracle(stacked(held, names), stacked(start, names), f"a hover's {names}",
                                   roundings=roundings)


def test_drawn_drones_spread_about_the_nominal_racer():
    """At no randomization every drone is the nominal racer (4.8 times its weight at full charge, no
    disturbance); at full randomization 4096 drones' masses, top speeds, lags and charges span their ranges
    and their thrust-to-weight ratios lie within the ranges' extremes."""
    keys = jax.random.split(jax.random.key(3), 4096)

    def at_no_randomization(key):
        body = drawn_body(QUADROTOR, key, jnp.asarray(0.0))
        return body.thrust_to_weight(QUADROTOR), body.force, body.moment

    with jax.enable_x64(new_val=True):
        ratio, force, moment = jax.jit(jax.vmap(at_no_randomization))(keys)
        roundings = chain_roundings(jax.make_jaxpr(jax.vmap(at_no_randomization))(keys))
        assert_computes_the_oracle(np.asarray(ratio), np.full(4096, NOMINAL_RATIO),
                                   "the nominal racer's thrust-to-weight ratio", roundings=roundings)
    assert not np.asarray(force).any() and not np.asarray(moment).any()
    drawn = jax.vmap(lambda key: drawn_body(QUADROTOR, key, jnp.asarray(1.0)))(keys)
    mass, top = np.asarray(drawn.mass), np.asarray(drawn.top_speed)
    assert mass.min() < 0.768 * 0.82 and mass.max() > 0.768 * 1.18
    assert top.min() < 2400 * 0.92 and top.max() > 2400 * 1.08
    lag, charged = np.asarray(drawn.motor_lag), np.asarray(drawn.charged)
    assert lag.min() < 0.022 and lag.max() > 0.058 and charged.min() < 3.72 and charged.max() > 4.18
    ratio = np.asarray(jax.vmap(lambda body: body.thrust_to_weight(QUADROTOR))(drawn))
    weakest = NOMINAL_RATIO * 0.85 * (0.9 * 3.7 / 4.2) ** 2 / 1.2
    strongest = NOMINAL_RATIO * 1.15 * 1.1 ** 2 / 0.8
    assert weakest < ratio.min() and ratio.max() < strongest
