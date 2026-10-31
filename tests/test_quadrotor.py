"""A racing quadrotor's physics (`dew.rl.quadrotor`): a control step of drawn drones under drawn commands held
to a float64 NumPy transcription of the physics, written from the sources apart from Dew's code (the rigid
body, the rotors' lag and the battery's voltage, the drag, the ground effect, the disturbances and the flight
controller); a nominal drone holding its hover, an equilibrium of that physics; and the drones' drawn
parameters, the racer's randomization written out over the draws."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_computes_the_oracle, chain_roundings

from dew.rl.quadrotor import Body, Flight, Quadrotor, drawn_body, flown, hovering

QUADROTOR = Quadrotor()
HZ = 60.0
NOMINAL = {
    "mass": 0.768, "mass_spread": 0.2, "inertia": (2.5e-3, 2.1e-3, 4.3e-3), "inertia_spread": 0.3,
    "thrust_coefficient": 1.563e-6, "drag_coefficient": 1.909e-8, "coefficient_spread": 0.15,
    "top_speed": 2400.0, "top_spread": 0.1, "motor_lag": (0.02, 0.06), "rotor_drag": 5.37e-5,
    "quadratic_drag": (4.10e-3, 1.51e-2), "vertical_drag": 0.15, "drag_spread": 0.3,
    "ground_effect": (0.75, 1.0), "charged": (3.7, 4.2), "full_voltage": 4.2, "resistance": (0.003, 0.012),
    "bias_force": 2.0, "bias_moment": 2.0,
}
"""The nominal racer and its parameters' randomization as `dew.rl.quadrotor`'s documentation gives them
(Kaufmann et al.'s Table II racer, the drag, ground-effect, battery and disturbance figures it cites, and its
own spreads and ranges), each under the name of the `Quadrotor` field it is."""
RETUNED = {
    "mass": 0.55, "mass_spread": 0.11, "inertia": (1.6e-3, 1.4e-3, 2.9e-3), "inertia_spread": 0.13,
    "thrust_coefficient": 1.2e-6, "drag_coefficient": 1.5e-8, "coefficient_spread": 0.17,
    "top_speed": 2900.0, "top_spread": 0.07, "motor_lag": (0.03, 0.05), "rotor_drag": 4.4e-5,
    "quadratic_drag": (3.3e-3, 1.9e-2), "vertical_drag": 0.21, "drag_spread": 0.23,
    "ground_effect": (0.6, 0.9), "charged": (3.5, 4.0), "full_voltage": 4.1, "resistance": (0.004, 0.009),
    "bias_force": 1.3, "bias_moment": 2.9,
}
"""A second racer with every parameter `drawn_body` reads changed, and no two spreads or biases alike."""


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


def body_draws(key: jax.Array) -> dict:
    """Every draw `drawn_body` makes from `key`, uniform in [0, 1), by the keys it splits off in the order it
    takes them, named for the parameter it sets."""
    keys = jax.random.split(key, 16)
    return {"mass": jax.random.uniform(keys[0]), "inertia": jax.random.uniform(keys[1], (3,)),
            "thrust": jax.random.uniform(keys[2]), "drag": jax.random.uniform(keys[3]),
            "top": jax.random.uniform(keys[4]), "lag": jax.random.uniform(keys[5]),
            "rotor_drag": jax.random.uniform(keys[6]), "quadratic": jax.random.uniform(keys[7], (2,)),
            "vertical": jax.random.uniform(keys[8]), "ground": jax.random.uniform(keys[9]),
            "charged": jax.random.uniform(keys[10]), "resistance": jax.random.uniform(keys[11]),
            "force": jax.random.uniform(keys[12], (3,)), "moment": jax.random.uniform(keys[13], (3,))}


def written_body(racer: dict, ramp: float, draws: dict) -> dict[str, np.ndarray]:
    """The parameters of the drones whose draws are `draws` (`body_draws`), written out from a racer's table
    at a ramp: a scale parameter is its nominal times a factor within 1 +- the ramp times its spread; a
    parameter in a range is its nominal moved the ramp's share of the way to its draw in the range (the
    nominal lag the middle of its range, the nominal ground effect its strongest, the nominal charge the
    full voltage, the nominal resistance its least); the disturbance force is the drawn mass and the moment
    the nominal inertia, each times the ramp, its bias and a draw within +-1."""

    def scaled(nominal, spread: float, unit: np.ndarray) -> np.ndarray:
        return np.asarray(nominal) * (1 + ramp * spread * (2 * unit - 1))

    def moved(span: tuple[float, float], nominal: float, unit: np.ndarray) -> np.ndarray:
        return nominal + ramp * (span[0] + (span[1] - span[0]) * unit - nominal)

    r, unit = racer, draws
    mass = scaled(r["mass"], r["mass_spread"], unit["mass"])
    return {
        "mass": mass,
        "inertia": scaled(r["inertia"], r["inertia_spread"], unit["inertia"]),
        "thrust_coefficient": scaled(r["thrust_coefficient"], r["coefficient_spread"], unit["thrust"]),
        "drag_coefficient": scaled(r["drag_coefficient"], r["coefficient_spread"], unit["drag"]),
        "top_speed": scaled(r["top_speed"], r["top_spread"], unit["top"]),
        "motor_lag": moved(r["motor_lag"], sum(r["motor_lag"]) / 2, unit["lag"]),
        "rotor_drag": scaled(r["rotor_drag"], r["drag_spread"], unit["rotor_drag"]),
        "quadratic_drag": scaled(r["quadratic_drag"], r["drag_spread"], unit["quadratic"]),
        "vertical_drag": scaled(r["vertical_drag"], r["drag_spread"], unit["vertical"]),
        "ground_effect": moved(r["ground_effect"], max(r["ground_effect"]), unit["ground"]),
        "charged": moved(r["charged"], r["full_voltage"], unit["charged"]),
        "resistance": moved(r["resistance"], min(r["resistance"]), unit["resistance"]),
        "force": mass[:, None] * ramp * r["bias_force"] * (2 * unit["force"] - 1),
        "moment": np.asarray(r["inertia"]) * ramp * r["bias_moment"] * (2 * unit["moment"] - 1),
    }


def written_ratio(racer: dict, body: dict[str, np.ndarray]) -> np.ndarray:
    """A drone's full thrust over its weight at its charge at the start: its four rotors' thrust at its top
    speed scaled by its charge's share of the full voltage, over its mass times 9.81."""
    top = body["top_speed"] * body["charged"] / racer["full_voltage"]
    return 4 * body["thrust_coefficient"] * top ** 2 / (body["mass"] * 9.81)


@pytest.mark.parametrize("ramp", [0.0, 0.5, 1.0])
@pytest.mark.parametrize(("quadrotor", "racer"), [(QUADROTOR, NOMINAL), (Quadrotor(**RETUNED), RETUNED)],
                         ids=["nominal", "retuned"])
def test_drawn_drones_are_the_randomization_written_out(quadrotor, racer, ramp):
    """4096 drones drawn at a ramp, in float64, of the nominal racer and of a retuned one (each parameter
    changed, so each is read from the quadrotor drawn from): every field of a drone's body is the
    randomization written out over the draws `drawn_body` makes (`written_body`) to float64 rounding, and at
    no randomization exactly the racer's own, with no disturbance; and the drone's thrust-to-weight ratio is
    the written formula's."""
    keys = jax.random.split(jax.random.key(3), 4096)

    def drawn(key):
        body = drawn_body(quadrotor, key, jnp.asarray(ramp))
        return body, body.thrust_to_weight(quadrotor)

    with jax.enable_x64(new_val=True):
        bodies, ratio = jax.jit(jax.vmap(drawn))(keys)
        draws = jax.tree.map(np.asarray, jax.jit(jax.vmap(body_draws))(keys))
        roundings = chain_roundings(jax.make_jaxpr(jax.vmap(drawn))(keys))
    expected = written_body(racer, ramp, draws)
    for name in Body.__dataclass_fields__:
        shown = np.asarray(getattr(bodies, name))
        if ramp:
            assert_computes_the_oracle(shown, expected[name], f"a drone's {name}", roundings=roundings)
        else:
            np.testing.assert_array_equal(shown, expected[name])
    assert_computes_the_oracle(np.asarray(ratio), written_ratio(racer, expected),
                               "a drone's thrust-to-weight ratio", roundings=roundings)
