"""SkyDreamer's race (`dew.rl.skydreamer`) held to the paper written out in float64 NumPy: the race's rewards
and ends over drawn flights (Eq 1-2, Sec II-C), its starts (Table III), the flight plan (Eq 12) and the
commands its actions are; a world model reading proprioceptive entries beside its image; the policy-mean
smoothness term (Eq 11); and the base check's run (`tools/skydreamer_race.py`) read back from its record."""

import itertools
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import assert_computes_the_oracle

from dew.objectives.rl.dreamer import DreamerV3Objective, WorldModel
from dew.rl.quadrotor import GRAVITY
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
    crossing's gate reward or, outside the gate's size, the end; progress toward the pre-gate headed for
    while before it; the rate penalty; the shaped ground contact; a final reward of zero at an end."""
    index, stage, distance, passed = held
    approaching, aimed, gate, collided = stage == 0, index, 0.0, False
    for _ in range(3):
        center, heading = sky.track[index]
        heading = math.radians(heading)
        normal = np.asarray([math.cos(heading), math.sin(heading), 0.0])
        left = np.asarray([-math.sin(heading), math.cos(heading), 0.0])
        plane = sky.depth / 2 * (stage - 1)
        start, end = (before - center) @ normal, (after - center) @ normal
        if not start < plane <= end:
            break
        where = before + (plane - start) / (end - start) * (after - before) - np.asarray(center)
        off = max(abs(where @ left), abs(where[2])) / sky.size
        if off > 1:
            collided = True
        else:
            gate += 1 - off
        if stage == 2:
            passed, index = passed + 1, (index + 1) % len(sky.track)
        stage = (stage + 1) % 3
    progress = distance - np.linalg.norm(after - pre_gate_of(sky, aimed)) if approaching else 0.0
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
                got.append(float(taken.reward))
                want.append(reward)
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
                                       np.r_[expected, rates * SKY.rate_limit], f"throttle {throttle}",
                                       roundings=4)


def test_a_world_model_reads_its_proprioceptive_entries_beside_its_image():
    """A world model reading proprioceptive entries beside its image sets two observations apart that differ
    in them alone, and reads an image alike whatever they are when it reads none; its posterior over a
    window of such observations trains (a finite loss) as an image's alone does."""
    proprio = (("rates", 3), ("plan", 5))
    shared = {"image": (16, 16, 3), "reads": "image", "deter": 32, "hidden": 8, "stoch": 4, "classes": 4,
              "units": 16, "conv_depth": 4}
    image = np.random.default_rng(1).integers(0, 255, (16, 16, 3))
    images = jnp.asarray(np.stack([image, image]), jnp.uint8)
    observed = {"image": images, "rates": jnp.asarray([[0.1, 0.2, 0.3]] * 2),
                "plan": jnp.asarray([[0.0] * 5, [1.0] * 5])}
    for reading, apart in ((proprio, True), ((), False)):
        model = WorldModel(0, 4, proprio=reading, **shared)
        objective = DreamerV3Objective(model)
        variables = objective.init(jax.random.key(0))
        bound = model.bind(variables)
        posterior = np.asarray(bound.posterior(jnp.zeros((2, 32)), bound.embed(bound.seen(observed))))
        assert (not np.array_equal(posterior[0], posterior[1])) == apart


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
