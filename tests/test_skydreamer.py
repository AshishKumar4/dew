"""SkyDreamer's race (`dew.rl.skydreamer`) held to the paper written out in float64 NumPy: the race's rewards
and ends over drawn flights (Eq 1-2, Sec II-C), its starts (Table III), the flight plan (Eq 12) and the
commands its actions are; a world model reading proprioceptive entries beside its image; the policy-mean
smoothness term (Eq 11); and the base check's run (`tools/skydreamer_race.py`) read back from its record."""

import dataclasses
import itertools
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import assert_computes_the_oracle

from dew.objectives.rl.dreamer import DreamerV3Objective, WorldModel
from dew.rl.quadrotor import GRAVITY, hovering
from dew.rl.skydreamer import CIRCLE, Lap, Sky, SkyRace, plan

SKY = Sky()


def pre_gate_of(sky: Sky, index: int) -> np.ndarray:
    center, heading = sky.track[index]
    heading = math.radians(heading)
    return np.asarray(center) - sky.depth / 2 * np.asarray([math.cos(heading), math.sin(heading), 0.0])


def written_step(sky: Sky, before: np.ndarray, after: np.ndarray, velocity: np.ndarray, rates: np.ndarray,
                 orientation: np.ndarray, held: tuple) -> tuple:
    """A step of the race written from Sec II-C, from the drone's position before and after it: the planes
    crossed in order on the step's straight path (pre-gate, gate, post-gate, then the next gate's), each
    crossing's gate reward or, outside the gate's size, the end there; progress toward the pre-gate headed
    for up to where the step crosses it; the rate penalty; the shaped ground contact; a final reward of zero
    at an end."""
    index, stage, distance, passed = held
    approaching, aimed, gate, collided, reached = stage == 0, index, 0.0, False, after
    for _ in range(3):
        center, heading = sky.track[index]
        heading = math.radians(heading)
        normal = np.asarray([math.cos(heading), math.sin(heading), 0.0])
        left = np.asarray([-math.sin(heading), math.cos(heading), 0.0])
        plane = sky.depth / 2 * (stage - 1)
        start, end = (before - center) @ normal, (after - center) @ normal
        if not start < plane <= end:
            break
        crossing = before + (plane - start) / (end - start) * (after - before)
        where = crossing - np.asarray(center)
        off = max(abs(where @ left), abs(where[2])) / sky.size
        if stage == 0:
            reached = crossing
        if off > 1:
            collided = True
            break
        gate += 1 - off
        if stage == 2:
            passed, index = passed + 1, (index + 1) % len(sky.track)
        stage = (stage + 1) % 3
    progress = distance - np.linalg.norm(reached - pre_gate_of(sky, aimed)) if approaching else 0.0
    rate = math.expm1(min(np.abs(rates).sum(), 17.0)) / (2 * sky.hz * 1e5)
    w, x, y, z = orientation
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    tipped = abs(roll) > math.pi / 3 or abs(pitch) > math.pi / 3
    grounded = after[2] < 0.5 and (velocity[2] < -1.0 or tipped)
    ended = collided or grounded
    reward = 0.0 if ended else 5 * progress - rate + 30 * gate
    return reward, ended, (index, stage, np.linalg.norm(after - pre_gate_of(sky, index)), passed)


def test_a_race_rewards_and_ends_as_skydreamer_writes():
    """64 drones started as the race starts them, flown up to 200 steps in float64 by a throttle near hover
    and smoothly drawn rates: each step's reward is the race written out from the drone's path, to float64
    rounding, and its end, the gate headed for, the plane next and the gates passed are the written race's
    exactly, to each drone's first end; some drones pass a plane, some a whole gate, and some end."""
    race = SkyRace(views=("passed",))
    rng = np.random.default_rng([532, 3001, 11])
    planes = gates = ends = 0
    with jax.enable_x64(new_val=True):
        step = jax.jit(race.step)
        for drone, key in enumerate(jax.random.split(jax.random.key(5), 64)):
            lap, _ = race.reset(key)
            assert isinstance(lap, Lap)
            held = (int(lap.index), 0, float(lap.distance), 0)
            drift = np.cumsum(rng.normal(0, 0.02, (200, 4)), 0)
            got, want = [], []
            for t in range(200):
                # A throttle of zero holds the hover's thrust.
                wobble = 0.05 * np.sin(t / 20 + drone) + drift[t, 1:]
                action = np.clip(np.r_[0.05 + drift[t, 0], wobble], -1, 1)
                before = np.asarray(lap.flight.position)
                lap, taken = step(lap, jnp.asarray(action))
                flight = lap.flight
                reward, ended, held = written_step(
                    SKY, before, np.asarray(flight.position), np.asarray(flight.velocity),
                    np.asarray(flight.angular_velocity), np.asarray(flight.orientation), held)
                # Each reward beside the distance its progress is a difference of: rounding at the course's
                # size, which a small reward alone would not show.
                got += [float(taken.reward), float(lap.distance)]
                want += [reward, held[2]]
                assert (bool(taken.terminated), int(lap.index), int(lap.stage), int(lap.passed)) == (
                    ended, held[0], held[1], held[3])
                planes += held[1] != 0
                gates += held[3] > 0
                if ended:
                    ends += 1
                    break
            assert_computes_the_oracle(np.asarray(got), np.asarray(want), f"drone {drone}'s rewards",
                                       roundings=256)
    assert planes and gates and ends


def test_a_step_through_the_pre_gate_approaches_it_up_to_where_it_crosses():
    """A drone hovering level along the first gate's normal at 3.6 m/s, 0.41 m before the gate (the pre-gate
    0.4 m before it), crosses the pre-gate within a step: it earns the 0.01 m it approached the pre-gate,
    none for the way past it, and the pre-gate's reward at its centre, 30, so 30.05; started 0.9 m to the
    gate's side instead, it crosses the pre-gate beyond the gate's 0.8 m and ends there, its reward 0."""
    race = SkyRace(views=("passed",))
    center, heading = CIRCLE[0]
    heading = math.radians(heading)
    normal = np.asarray([math.cos(heading), math.sin(heading), 0.0])
    left = np.asarray([-math.sin(heading), math.cos(heading), 0.0])
    level = np.asarray([math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2)])
    with jax.enable_x64(new_val=True):
        lap, _ = race.reset(jax.random.key(0))
        for aside, reward, ended in ((0.0, 30.05, False), (0.9, 0.0, True)):
            start = np.asarray(center) - 0.41 * normal + aside * left
            flight = dataclasses.replace(
                hovering(SKY.quadrotor, lap.body, jnp.asarray(start), jnp.asarray(level)),
                velocity=jnp.asarray(3.6 * normal))
            here = dataclasses.replace(lap, flight=flight, index=jnp.asarray(0), stage=jnp.asarray(0),
                                       distance=jnp.asarray(np.hypot(0.01, aside)))
            after, taken = race.step(here, jnp.zeros(4))
            assert (bool(taken.terminated), int(after.stage)) == (ended, 0 if ended else 1)
            assert_computes_the_oracle(np.asarray([float(taken.reward)]), np.asarray([reward]),
                                       f"crossing {aside} m aside", roundings=64)


def test_a_race_starts_before_a_gate_at_rest():
    """Every start (Table III) is at rest, 2 to 4 m before its gate along its normal, within 1 m to its side
    and from its centre to 1.3 m below it (0.7 m below before the first gate, where 30% of starts are),
    turned from the gate's heading at most 20 degrees, its rotors at a quarter to half of their top speed."""
    race = SkyRace(views=("passed",))
    first = 0
    with jax.enable_x64(new_val=True):
        laps = jax.vmap(lambda key: race.reset(key)[0])(jax.random.split(jax.random.key(9), 512))
        laps = jax.tree.map(np.asarray, laps)
    for drone in range(512):
        lap = jax.tree.map(lambda leaf, drone=drone: leaf[drone], laps)
        center, heading = CIRCLE[int(lap.index)]
        heading = math.radians(heading)
        offset = lap.flight.position - np.asarray(center)
        along = offset @ np.asarray([math.cos(heading), math.sin(heading), 0.0])
        aside = offset @ np.asarray([-math.sin(heading), math.cos(heading), 0.0])
        assert -4 <= along <= -2 and abs(aside) <= 1 and -1.3 <= offset[2] <= 0
        assert not lap.flight.velocity.any()
        w, x, y, z = lap.flight.orientation
        turned = math.atan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z)) - heading
        assert abs(math.atan2(math.sin(turned), math.cos(turned))) <= math.pi / 9
        first += int(lap.index) == 0
    # The rotors' speeds, a share of their top speed, span the quarter to the half.
    shares = laps.flight.speed / SKY.quadrotor.top_speed
    assert shares.min() >= 0.25 and shares.max() <= 0.5
    # A start before the first gate is drawn 30% of the time, plus its share of the others.
    assert 0.3 * 512 < first < 0.6 * 512


def test_the_flight_plan_is_the_next_three_gates_and_their_changes():
    """The flight plan at each gate of the track is Eq 12 written out: for it and the two after it, the
    change in position and in heading (a half turn either way) from the gate before, then their positions
    and headings, the track closed."""
    count = len(CIRCLE)
    for index in range(count):
        gates = [CIRCLE[(index + offset) % count] for offset in range(-1, 3)]
        changes, places = [], []
        for before, gate in itertools.pairwise(gates):
            turn = math.radians(gate[1] - before[1])
            changes += [*np.subtract(gate[0], before[0]), math.atan2(math.sin(turn), math.cos(turn))]
            places += [*gate[0], math.radians(gate[1])]
        with jax.enable_x64(new_val=True):
            got = np.asarray(plan(SKY, jnp.asarray(index)))
        assert_computes_the_oracle(got, np.asarray(changes + places), f"gate {index}'s plan", roundings=8)


def test_an_action_commands_collective_thrust_and_body_rates():
    """An action's throttle commands the hover's thrust at zero, none at -1 and the full thrust at 1,
    linearly between; its rates `rate_limit` times it; each clipped to [-1, 1] first."""
    race = SkyRace(views=("passed",))
    hover, full = SKY.quadrotor.mass * GRAVITY, SKY.quadrotor.nominal_thrust
    rates = np.asarray([0.3, -0.7, 1.4])
    with jax.enable_x64(new_val=True):
        lap, _ = race.reset(jax.random.key(0))
        for throttle in (-1.5, -1.0, -0.3, 0.0, 0.4, 1.0, 2.0):
            collective, commanded = race.commanded(lap, jnp.asarray(np.r_[throttle, rates]))
            clipped = min(max(throttle, -1.0), 1.0)
            expected = hover + clipped * (full - hover if clipped >= 0 else hover)
            assert_computes_the_oracle(np.r_[float(collective), np.asarray(commanded)],
                                       np.r_[expected, np.clip(rates, -1, 1) * SKY.rate_limit],
                                       f"throttle {throttle}",
                                       roundings=4)


def test_a_world_model_reads_its_proprioceptive_entries_beside_its_image():
    """An agent whose world model reads proprioceptive entries beside its image acts apart on two drones that
    see the same image and differ in those entries alone, where one reading the image alone acts alike; and
    its loss over windows of such observations, the posterior scanned over each, is finite."""
    from dew.objectives.base import Step

    proprio = (("rates", 3), ("plan", 5))
    shared = {"image": (16, 16, 3), "reads": "image", "deter": 32, "hidden": 8, "stoch": 4, "classes": 4,
              "units": 16, "conv_depth": 4}
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, (16, 16, 3))
    seen = {"image": jnp.asarray(np.stack([image, image]), jnp.uint8),
            "rates": jnp.asarray([[0.1, 0.2, 0.3]] * 2), "plan": jnp.asarray([[0.0] * 5, [1.0] * 5])}
    step = {"observation": seen, "reward": jnp.zeros(2), "first": jnp.ones(2, bool),
            "last": jnp.zeros(2, bool), "terminal": jnp.zeros(2, bool)}
    windows = {"observation": {"image": jnp.asarray(rng.integers(0, 255, (2, 6, 16, 16, 3)), jnp.uint8),
                               "rates": jnp.asarray(rng.normal(size=(2, 6, 3))),
                               "plan": jnp.asarray(rng.normal(size=(2, 6, 5)))},
               "action": jnp.asarray(rng.uniform(-1, 1, (2, 6, 4))),
               "reward": jnp.asarray(rng.normal(size=(2, 6))),
               "first": jnp.zeros((2, 6), bool), "last": jnp.zeros((2, 6), bool),
               "terminal": jnp.zeros((2, 6), bool)}
    for reading, apart in ((proprio, True), ((), False)):
        objective = DreamerV3Objective(WorldModel(0, 4, proprio=reading, **shared), imagination=3)
        variables = objective.init(jax.random.key(0))
        acted, _, _ = objective.greedy(variables, step, jax.random.key(1), objective.memory(2))
        assert (not np.array_equal(np.asarray(acted[0]), np.asarray(acted[1]))) == apart
        if reading:
            step_of = Step(jnp.zeros((), jnp.int32), jax.random.key(2), None)
            ratio, _ = objective.loss(variables, windows, step_of)
            assert np.isfinite(float(ratio.total))


def test_the_smoothness_term_is_the_squared_change_of_the_imagined_means():
    """The actor's imagined loss with a smoothness weight is the loss without it plus the weight times each
    step's squared change of the policy's mean from the step before, in float64 to rounding."""
    rng = np.random.default_rng([532, 11])
    mean, std = rng.normal(size=(6, 9, 4)), rng.uniform(0.1, 1, (6, 9, 4))
    actions, advantage = rng.normal(size=(6, 9, 4)), rng.normal(size=(6, 8))
    weight = rng.uniform(0, 1, (6, 9))
    model = WorldModel(4, 4)
    with jax.enable_x64(new_val=True):
        plain = DreamerV3Objective(model)._imagined_policy((jnp.asarray(mean), jnp.asarray(std)),
                                                           jnp.asarray(actions), jnp.asarray(advantage),
                                                           jnp.asarray(weight))
        smoothed = DreamerV3Objective(model, smoothness=0.002)._imagined_policy(
            (jnp.asarray(mean), jnp.asarray(std)), jnp.asarray(actions), jnp.asarray(advantage),
            jnp.asarray(weight))
    change = np.sum(np.square(mean[:, 1:] - mean[:, :-1]), -1)
    assert_computes_the_oracle(np.asarray(smoothed), np.asarray(plain) + 0.002 * change, "smoothness",
                               roundings=16)


def test_the_base_checks_run_reads_back_as_itself():
    """The base check's run (`tools/skydreamer_race.py`) reads back from its record as itself."""
    from tool_support import load

    from dew.objectives.rl.config import EnvRunConfig

    run = load("skydreamer_race").run(Path("/runs/base"), Path("/assets"))
    assert EnvRunConfig.from_dict(json.loads(json.dumps(run.to_dict()))) == run
