"""SkyDreamer's race (`dew.rl.skydreamer`) held to the paper written out in float64 NumPy: the race's rewards
and ends over drawn flights on each track (Eq 1-2, Sec II-C), the frames of gates passed by, its starts
(Table III), the flight plan (Eq 12), the commands its actions are and their latency; the fisheye lens's
resampling (`dew.rl.hyperlux`); a world model reading proprioceptive entries beside its image; the
policy-mean smoothness term (Eq 11); and the recipe's runs (`tools/skydreamer_race.py`) read back from their
records."""

import dataclasses
import itertools
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_computes_the_oracle

from dew.objectives.rl.dreamer import DreamerV3Objective, WorldModel
from dew.rl.flight import commanded
from dew.rl.hyperlux import fisheye_map, fisheyed
from dew.rl.quadrotor import GRAVITY, hovering
from dew.rl.skydreamer import CIRCLE, TRACKS, Lap, Sky, SkyRace, plan

SKY = Sky()


def axes_of(heading: float) -> tuple[np.ndarray, np.ndarray]:
    heading = math.radians(heading)
    return (np.asarray([math.cos(heading), math.sin(heading), 0.0]),
            np.asarray([-math.sin(heading), math.cos(heading), 0.0]))


def pre_gate_of(sky: Sky, gates, index: int) -> np.ndarray:
    center, heading, _ = gates[index]
    return np.asarray(center) - sky.depth / 2 * axes_of(heading)[0]


def written_step(sky: Sky, gates, before: np.ndarray, after: np.ndarray, velocity: np.ndarray,
                 rates: np.ndarray, orientation: np.ndarray, held: tuple) -> tuple:
    """A step of the race over `gates` written from Sec II-C, from the drone's position before and after it:
    the planes crossed in order on the step's straight path (pre-gate, gate, post-gate, then the next gate's),
    each crossing's gate reward or, outside the gate's size, the end there; where the race strikes frames,
    any seen gate's frame (between its 1.5 m opening and its 2 m outer edge) crossed but those headed for, the
    end; progress toward the pre-gate headed for up to where the step crosses it; the rate penalty; the
    shaped ground contact; a final reward of zero at an end."""
    index, stage, distance, passed = held
    approaching, aimed, gate, collided, reached = stage == 0, index, 0.0, False, after
    for _ in range(3):
        center, heading, _ = gates[index]
        normal, left = axes_of(heading)
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
            passed, index = passed + 1, (index + 1) % len(gates)
        stage = (stage + 1) % 3
    for other, (center, heading, seen) in enumerate(gates):
        normal, left = axes_of(heading)
        start, end = (before - center) @ normal, (after - center) @ normal
        if not sky.frames or not seen or other in (aimed, index) or start * end > 0 or start == end:
            continue
        where = before + start / (start - end) * (after - before) - np.asarray(center)
        across = max(abs(where @ left), abs(where[2]))
        opening, edge = sky.racing.opening / 2, sky.racing.opening / 2 + sky.racing.frame
        collided = collided or opening < across <= edge
    progress = distance - np.linalg.norm(reached - pre_gate_of(sky, gates, aimed)) if approaching else 0.0
    rate = math.expm1(min(np.abs(rates).sum(), 17.0)) / (2 * sky.hz * 1e5)
    w, x, y, z = orientation
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    tipped = abs(roll) > math.pi / 3 or abs(pitch) > math.pi / 3
    grounded = after[2] < 0.5 and (velocity[2] < -1.0 or tipped)
    ended = collided or grounded
    reward = 0.0 if ended else 5 * progress - rate + 30 * gate
    return reward, ended, (index, stage, np.linalg.norm(after - pre_gate_of(sky, gates, index)), passed)


@pytest.mark.parametrize("track", sorted(TRACKS))
def test_a_race_rewards_and_ends_as_skydreamer_writes(track):
    """64 drones started on a track as the race starts them, flown up to 200 steps in float64: half by a
    throttle near hover and smoothly drawn rates, half set level, hovering and flying at 5 m/s toward their
    gate. Each step's reward is the race written out from the drone's path, to float64 rounding, held beside
    the distance its progress differences, and its end, the gate headed for, the plane next and the gates
    passed are the written race's exactly, to each drone's first end; some drones pass a plane, some a whole
    gate, and some end."""
    sky, gates = Sky(tracks=(track,), frames=True), TRACKS[track]
    race = SkyRace(sky, views=("passed",))
    rng = np.random.default_rng([532, 3001, 11])
    planes = passes = ends = 0
    with jax.enable_x64(new_val=True):
        step = jax.jit(race.step)
        for drone, key in enumerate(jax.random.split(jax.random.key(5), 64)):
            lap, _ = race.reset(key)
            assert isinstance(lap, Lap)
            straight = drone % 2 == 1
            if straight:
                heading = math.radians(gates[int(lap.index)][1])
                level = jnp.asarray([math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2)])
                flight = hovering(sky.quadrotor, lap.body, lap.flight.position, level)
                lap = dataclasses.replace(lap, flight=dataclasses.replace(
                    flight, velocity=jnp.asarray([5 * math.cos(heading), 5 * math.sin(heading), 0.0])))
            held = (int(lap.index), 0, float(lap.distance), 0)
            drift = np.zeros((200, 4)) if straight else np.cumsum(rng.normal(0, 0.02, (200, 4)), 0)
            got, want = [], []
            for t in range(200):
                # A throttle of zero holds the hover's thrust.
                wobble = np.zeros(3) if straight else 0.05 * np.sin(t / 20 + drone) + drift[t, 1:]
                action = np.clip(np.r_[0.0 if straight else 0.05 + drift[t, 0], wobble], -1, 1)
                before = np.asarray(lap.flight.position)
                lap, taken = step(lap, jnp.asarray(action))
                flight = lap.flight
                reward, ended, held = written_step(
                    sky, gates, before, np.asarray(flight.position), np.asarray(flight.velocity),
                    np.asarray(flight.angular_velocity), np.asarray(flight.orientation), held)
                got += [float(taken.reward), float(lap.distance)]
                want += [reward, held[2]]
                assert (bool(taken.terminated), int(lap.index), int(lap.stage), int(lap.passed)) == (
                    ended, held[0], held[1], held[3])
                planes += held[1] != 0
                passes += held[3] > 0
                if ended:
                    ends += 1
                    break
            assert_computes_the_oracle(np.asarray(got), np.asarray(want), f"drone {drone}'s rewards",
                                       roundings=256)
    assert planes and passes and ends


def test_a_frame_flown_into_ends_a_flight_and_an_opening_does_not():
    """On the inverted loop, a drone headed for the gate over the second, flying level at 3.6 m/s across the
    second gate's plane, ends its flight where it crosses the frame (0.9 m from the centre) and flies on
    through the opening (0.3 m from it) or past the frame's outer edge (1.2 m), where the race strikes frames;
    where it does not, as in the base check, it flies on everywhere."""
    sky = Sky(tracks=("inverted_loop",), frames=True)
    race = SkyRace(sky, views=("passed",))
    center, heading, _ = TRACKS["inverted_loop"][2]
    normal, left = axes_of(heading)
    level = np.asarray([math.cos(math.radians(heading) / 2), 0.0, 0.0, math.sin(math.radians(heading) / 2)])
    with jax.enable_x64(new_val=True):
        lap, _ = race.reset(jax.random.key(0))
        for aside, ended in ((0.9, True), (0.3, False), (1.2, False)):
            start = np.asarray(center) - 0.02 * normal + aside * left
            flight = dataclasses.replace(
                hovering(sky.quadrotor, lap.body, jnp.asarray(start), jnp.asarray(level)),
                velocity=jnp.asarray(3.6 * normal))
            # Headed for the unseen gate over the second, 2.65 m above this crossing.
            here = dataclasses.replace(lap, flight=flight, track=jnp.asarray(0), index=jnp.asarray(1),
                                       stage=jnp.asarray(0), distance=jnp.asarray(5.0))
            _, taken = race.step(here, jnp.zeros(4))
            unframed = SkyRace(dataclasses.replace(sky, frames=False), views=("passed",))
            _, unstruck = unframed.step(here, jnp.zeros(4))
            assert (bool(taken.terminated), bool(unstruck.terminated)) == (ended, False)


def test_a_race_starts_before_a_gate_at_rest():
    """Every start (Table III) is at rest, 2 to 4 m before its gate along its normal, within 1 m to its side
    and from its centre to 1.3 m below it (0.7 m below before the first gate, where 30% of starts are),
    turned from the gate's heading at most 20 degrees, its rotors at a quarter to half of their top speed."""
    race = SkyRace(views=("passed",))
    first = 0
    with jax.enable_x64(new_val=True):
        laps = jax.vmap(lambda key: race.reset(key)[0])(jax.random.split(jax.random.key(9), 512))
        laps = jax.tree.map(np.asarray, dataclasses.replace(laps, key=jax.random.key_data(laps.key)))
    for drone in range(512):
        lap = jax.tree.map(lambda leaf, drone=drone: leaf[drone], laps)
        center, heading, _ = CIRCLE[int(lap.index)]
        normal, left = axes_of(heading)
        offset = lap.flight.position - np.asarray(center)
        assert -4 <= offset @ normal <= -2 and abs(offset @ left) <= 1 and -1.3 <= offset[2] <= 0
        assert not lap.flight.velocity.any()
        w, x, y, z = lap.flight.orientation
        turned = math.atan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z)) - math.radians(heading)
        assert abs(math.atan2(math.sin(turned), math.cos(turned))) <= math.pi / 9
        first += int(lap.index) == 0
    # The rotors' speeds, a share of their top speed, span the quarter to the half.
    shares = laps.flight.speed / SKY.quadrotor.top_speed
    assert shares.min() >= 0.25 and shares.max() <= 0.5
    # A start before the first gate is drawn 30% of the time, plus its share of the others.
    assert 0.3 * 512 < first < 0.6 * 512


@pytest.mark.parametrize("track", sorted(TRACKS))
def test_the_flight_plan_is_the_next_three_gates_and_their_changes(track):
    """The flight plan at each gate of each track, seen or not, is Eq 12 written out: for it and the two
    after it, the change in position and in heading (a half turn either way) from the gate before, then their
    positions and headings, the track closed."""
    gates = TRACKS[track]
    sky = Sky(tracks=(track,))
    for index in range(len(gates)):
        ahead = [gates[(index + offset) % len(gates)] for offset in range(-1, 3)]
        changes, places = [], []
        for before, gate in itertools.pairwise(ahead):
            turn = math.radians(gate[1] - before[1])
            changes += [*np.subtract(gate[0], before[0]), math.atan2(math.sin(turn), math.cos(turn))]
            places += [*gate[0], math.radians(gate[1])]
        with jax.enable_x64(new_val=True):
            got = np.asarray(plan(sky, jnp.asarray(0), jnp.asarray(index)))
        assert_computes_the_oracle(got, np.asarray(changes + places), f"gate {index}'s plan", roundings=8)


def test_an_action_commands_collective_thrust_and_body_rates_after_its_latency():
    """An action's throttle commands the hover's thrust at zero, none at -1 and the full thrust at 1,
    linearly between; its rates the drone's rate limit times it; each clipped to [-1, 1] first. With 44 ms of
    latency at 90 Hz, each command reaches the flight controller four steps after its action, the drone
    holding its hover until the first does; with none, at once."""
    rates = np.asarray([0.3, -0.7, 1.4])
    throttles = (-1.5, -1.0, -0.3, 0.0, 0.4, 1.0, 2.0)
    for latency, delay in ((0.0, 0), (0.044, 4)):
        race = SkyRace(Sky(latency=latency), views=("passed",))
        with jax.enable_x64(new_val=True):
            lap, _ = race.reset(jax.random.key(0))
            hover, full = float(lap.body.mass) * GRAVITY, SKY.quadrotor.nominal_thrust
            limit = float(lap.rate_limit)
            wanted = []
            for throttle in throttles:
                clipped = min(max(throttle, -1.0), 1.0)
                wanted.append(np.r_[hover + clipped * (full - hover if clipped >= 0 else hover),
                                    np.clip(rates, -1, 1) * limit])
            held = [np.r_[hover, 0.0, 0.0, 0.0]] * delay
            for at, throttle in enumerate(throttles):
                lap, collective, rates_sent = commanded(race.sky.quadrotor, lap,
                                                        jnp.asarray(np.r_[throttle, rates]))
                assert_computes_the_oracle(np.r_[float(collective), np.asarray(rates_sent)],
                                           (held + wanted)[at], f"throttle {throttle}", roundings=4)


def test_a_curriculum_draws_the_latency_and_randomizes_the_drone_and_camera_only_above_none():
    """At no randomization every drone's command waits its nominal four steps, its rate limit is the
    nominal 10 rad/s and its image neutral; at full randomization the latency spans two to six steps, each
    drawn, and the rate limits and images vary from drone to drone."""
    sky = Sky(latency=0.044)
    for ramp, delays in ((0.0, {4}), (1.0, {2, 3, 4, 5, 6})):
        race = SkyRace(sky, views=("passed",), ramp=ramp)
        laps = jax.vmap(lambda key, race=race: race.reset(key)[0])(jax.random.split(jax.random.key(3), 512))
        assert set(np.unique(np.asarray(laps.delay)).tolist()) == delays
        varies = [np.ptp(np.asarray(value)) > 0 for value in (laps.rate_limit, laps.camera.exposure,
                                                              laps.camera.gamma, laps.body.mass)]
        assert varies == [ramp > 0] * 4


def test_a_race_of_a_level_per_track_flies_each_tracks_episodes_at_its_own():
    """A race over the circle at no randomization and the inverted loop at full: the circle's episodes wait
    their nominal four steps and fly the nominal drone with no rotor noise, the loop's span two to six steps
    and vary; a circle's lap stepped from two keys flies alike at the circle's level of none, and apart
    once the circle's level is full."""
    sky = Sky(tracks=("circle", "inverted_loop"), latency=0.044)
    race = SkyRace(sky, views=("passed",), ramp=(0.0, 1.0))
    laps = jax.vmap(lambda key: race.reset(key)[0])(jax.random.split(jax.random.key(3), 512))
    track = np.asarray(laps.track)
    for index, delays, varies in ((0, {4}, False), (1, {2, 3, 4, 5, 6}, True)):
        on = track == index
        assert set(np.unique(np.asarray(laps.delay)[on]).tolist()) == delays
        spread = [np.ptp(np.asarray(value)[on]) > 0 for value in (laps.rate_limit, laps.body.mass)]
        assert spread == [varies] * 2
    circle = jax.tree.map(lambda leaf: leaf[int(np.flatnonzero(track == 0)[0])], laps)
    action = jnp.asarray([0.3, 0.1, -0.2, 0.05])
    for ramp, apart in (((0.0, 1.0), False), ((1.0, 0.0), True)):
        stepped = SkyRace(sky, views=("passed",), ramp=ramp)
        speeds = [np.asarray(stepped.step(circle.replace(key=jax.random.key(seed)), action)[0].flight.speed)
                  for seed in (5, 6)]
        assert (not np.array_equal(*speeds)) == apart


def written_fisheye(image: np.ndarray, size: int, fov: float) -> np.ndarray:
    """An equidistant fisheye `size` square of horizontal field of view `fov` sampled a pixel at a time from
    the pinhole `image` `[S, S, 3]` of the same field of view: each pixel's angle from the axis its radius
    times half the field of view, its direction projected onto the pinhole and read bilinearly; black where
    the pinhole does not reach."""
    source, half = image.shape[0], math.radians(fov) / 2
    out = np.zeros((size, size, 3))
    for row in range(size):
        for column in range(size):
            x, y = (column + 0.5) / size * 2 - 1, (row + 0.5) / size * 2 - 1
            radius = math.hypot(x, y)
            theta = half * radius
            if theta >= math.pi / 2:
                continue
            scale = math.tan(theta) / math.tan(half) / radius if radius else 0.0
            u, v = x * scale, y * scale
            if abs(u) > 1 or abs(v) > 1:
                continue
            c, r = (u + 1) / 2 * source - 0.5, (v + 1) / 2 * source - 0.5
            c0, r0 = math.floor(c), math.floor(r)
            for dr, wr in ((0, 1 - (r - r0)), (1, r - r0)):
                for dc, wc in ((0, 1 - (c - c0)), (1, c - c0)):
                    rr, cc = min(max(r0 + dr, 0), source - 1), min(max(c0 + dc, 0), source - 1)
                    out[row, column] += wr * wc * image[rr, cc]
    return out


def test_a_fisheye_lens_samples_the_pinhole_at_each_pixels_direction():
    """A 48-pixel equidistant fisheye 120 degrees across, resampled from drawn 80-pixel pinholes of the same
    field of view, is the fisheye written a pixel at a time, run in float64; its centre is held by the
    pinhole and its corners, beyond the pinhole's reach, black."""
    images = np.random.default_rng([532, 7]).integers(0, 256, (2, 80, 80, 3)).astype(np.uint8)
    corners, weights, held = fisheye_map(48, 80, 120.0)
    with jax.enable_x64(new_val=True):
        got = np.asarray(fisheyed(jnp.asarray(images), jnp.asarray(corners), jnp.asarray(weights)))
    written = np.stack([written_fisheye(image.astype(np.float64), 48, 120.0) for image in images])
    assert_computes_the_oracle(got, written, "the fisheye", roundings=8)
    assert held[24, 24] and not held[0, 0] and not held[-1, -1]


def test_a_world_model_reads_its_proprioceptive_entries_beside_its_image():
    """A world model reading proprioceptive entries beside its image sets apart, in its posterior's
    log-probabilities, two drones that see the same image and differ in those entries alone, where one
    reading the image alone holds them alike; its agent acts on such steps, and its loss over windows of them,
    the posterior scanned over each, is finite."""
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
        bound = objective.model.bind(variables)
        tokens = bound.embed(bound.seen(step["observation"]))
        posterior = np.asarray(bound.posterior(jnp.zeros((2, 32)), tokens))
        assert (not np.array_equal(posterior[0], posterior[1])) == apart
        acted, _, _ = objective.greedy(variables, step, jax.random.key(1), objective.memory(2))
        assert np.isfinite(np.asarray(acted)).all() and np.asarray(acted).shape == (2, 4)
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


def test_a_recipes_run_trains_nothing_past_its_steps_and_scores_a_chunk_left_unscored_first():
    """The recipe's run (`tools/skydreamer_race.py`) run again once it has taken its steps, or asked for fewer
    than it took, trains no update; a chunk checkpointed before its score is restored and scored where it
    stands; otherwise its next chunk ends at the update reaching the next judgement 100k steps on, at 8
    environment steps an update with windows of 64 and 32 with the later phases' 256."""
    from tool_support import load

    tool = load("skydreamer_race")
    warmup, longer = tool.PHASES[0], tool.PHASES[1]
    for steps in (4_000_000, 1_000_000):
        assert tool.chunk_end(300_000, 4_000_000, steps, 100_000, longer, pending=False) is None
    assert tool.chunk_end(300_000, 4_000_000, 4_000_000, 100_000, longer, pending=True) == 300_000
    assert tool.chunk_end(200_000, 1_602_048, 4_000_000, 100_000, warmup, pending=False) == 200_000 + 12_244
    assert tool.chunk_end(200_000, 1_602_048, 4_000_000, 100_000, longer, pending=False) == 200_000 + 3_061


@pytest.mark.parametrize("setting", ["BASE", "REALISM"])
def test_a_recipes_run_reads_back_as_itself(setting):
    """The recipe's run in each setting (`tools/skydreamer_race.py`: the base check and the realism run)
    and each of its phases reads back from its record as itself."""
    from tool_support import load

    from dew.config.env import EnvRunConfig

    tool = load("skydreamer_race")
    chosen = getattr(tool, setting)
    for phase in chosen.phases:
        run = tool.run(Path("/runs/base"), Path("/assets"), setting=chosen, phase=phase)
        assert EnvRunConfig.from_dict(json.loads(json.dumps(run.to_dict()))) == run
