"""First-person gate racing (`dew.rl.flight`), each rule held to a float64 NumPy transcription of it apart
from Dew's code: a step's reward, passages, crashes and termination over drawn paths through, beside and
around the gates; the commands' throttle, rate limit, lag and latency; the courses' layouts and loops; and
the camera as HyperDrone builds it from a mounted frame. A race over a backend flying straight at each
gate passes it."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_computes_the_oracle, chain_roundings

from dew.rl.flight import (
    GATES,
    LAYOUTS,
    GateRace,
    Kinematics,
    Racing,
    cameras,
    commanded,
    drawn_track,
    quaternion,
    raced,
    rotation,
    started,
)

RACING = Racing()
DRONES = 512


def tracks(racing: Racing, seed: int, count: int = DRONES):
    """`count` drones' episodes as a race starts them, and their starts' positions."""
    keys = jax.random.split(jax.random.key(seed), count)
    return jax.vmap(lambda key: started(racing, key, jnp.asarray(1.0)))(keys)


def axes(facing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (np.stack([np.cos(facing), np.sin(facing), np.zeros_like(facing)], -1),
            np.stack([-np.sin(facing), np.cos(facing), np.zeros_like(facing)], -1))


def oracle_step(racing: Racing, track, action, position, rates, path, collided):
    """One step of the race's rules for each drone, in float64 loops: its reward, whether it crashed, the
    gate it heads for after and whether its episode was cut."""
    half, inner = racing.opening / 2, racing.opening / 2 - racing.radius
    outer = half + racing.frame + racing.radius
    rewards, crashes, indices, cuts = [], [], [], []
    for drone in range(len(position)):
        course, facing, count = track.course[drone], track.facing[drone], int(track.count[drone])
        target = int(track.index[drone])
        crashed, passed, centring = bool(collided[drone]), False, 0.0
        for begin, end in zip(path[drone, :-1], path[drone, 1:], strict=True):
            if end[2] < racing.radius or end[2] > racing.ceiling:
                crashed = True
            for gate in range(count):
                normal, left = axes(facing[gate])
                before, after = (begin - course[gate]) @ normal, (end - course[gate]) @ normal
                if before * after > 0 or before == after:
                    continue
                point = begin + before / (before - after) * (end - begin) - course[gate]
                across = max(abs(point @ left), abs(point[2]))
                if inner < across < outer:
                    crashed = True
                if gate == target and before < 0:
                    if across <= inner:
                        passed, centring = True, max(centring, across)
                    elif outer <= across < racing.miss_radius:
                        crashed = True
        passed = passed and not crashed
        distance = np.linalg.norm(course[target] - position[drone])
        reward = racing.progress_scale * (track.distance[drone] - distance)
        if passed:
            reward += racing.gate_bonus * (1 - centring / half)
            target = (target + 1) % count
        reward -= racing.rate_cost * np.linalg.norm(rates[drone]) / racing.hz
        reward -= racing.smoothness_cost * np.sum((np.clip(action[drone], -1, 1) - track.action[drone]) ** 2)
        reward -= racing.crash_penalty if crashed else 0.0
        rewards.append(reward)
        crashes.append(crashed)
        indices.append(target)
        cuts.append(not crashed and track.steps[drone] + 1 >= racing.max_steps)
    return np.asarray(rewards), np.asarray(crashes), np.asarray(indices), np.asarray(cuts)


def drawn_paths(racing: Racing, track, rng: np.random.Generator):
    """Each drone's path through a step, three physics steps of a straight line, by its kind: through its
    gate's opening, through its frame, beside it within the miss radius, far beside it, short of it, or
    dipping to the ground; and the kinds."""
    count = len(track.index)
    rows = np.arange(count)
    center = track.course[rows, track.index]
    normal, left = axes(track.facing[rows, track.index])
    half = racing.opening / 2
    inner, outer = half - racing.radius, half + racing.frame + racing.radius
    kind = rng.integers(0, 6, count)
    miss = racing.miss_radius
    bounds = np.asarray([(0.0, inner - 0.02), (inner + 0.02, outer - 0.02), (outer + 0.02, miss - 0.02),
                         (miss + 0.5, miss + 2.0), (0.0, inner), (0.0, inner)])
    across = rng.uniform(bounds[kind, 0], bounds[kind, 1])
    sideways = rng.random(count) < 0.5
    other = rng.uniform(-1, 1, count) * across
    widest = across * rng.choice([-1, 1], count)
    offset = (np.where(sideways, widest, other)[:, None] * left
              + np.where(sideways, other, widest)[:, None] * np.asarray([0.0, 0.0, 1.0]))
    lead = rng.uniform(0.3, 1.2, count)
    times = np.asarray([-1.5, -0.5, 0.5, 1.5]) + np.where(kind == 4, -3.0, 0.0)[:, None]
    path = (center + offset)[:, None] + times[..., None] * (lead[:, None] * normal)[:, None]
    dipped = path.copy()
    dipped[:, 2, 2] = rng.uniform(0, 2 * racing.radius, count)
    return np.where((kind == 5)[:, None, None], dipped, path), kind


def test_a_step_pays_passes_and_ends_as_the_race_rules():
    """Drones flown through their next gate's opening, its frame, beside it within the miss radius and far
    beside it, short of it, and down to the ground: each step's reward (progress, the gate's centring
    bonus, the rates' and the action's costs, a crash's penalty), each crash, each gate headed for after the
    step and each cut held to the race's rules in float64 loops: rewards to float64 rounding, the rest
    exactly."""
    rng = np.random.default_rng([532, 5])
    with jax.enable_x64(new_val=True):
        racing = Racing(episode_seconds=0.05)
        track = jax.tree.map(np.asarray, tracks(racing, 0)[0])
        track = track.replace(steps=rng.integers(0, 3, DRONES), action=rng.uniform(-1, 1, (DRONES, 4)))
        path, kind = drawn_paths(racing, track, rng)
        position = path[:, -1]
        rates = rng.normal(0, 3, (DRONES, 3))
        kinematics = Kinematics(position, rng.normal(0, 5, (DRONES, 3)),
                                quaternion(*rng.normal(0, 0.3, (3, DRONES))), rates)
        action = rng.uniform(-1.2, 1.2, (DRONES, 4))
        collided = rng.random(DRONES) < 0.05
        stepped = jax.jit(lambda *inputs: raced(racing, *inputs))
        inputs = (jax.tree.map(jnp.asarray, track), jnp.asarray(action),
                  jax.tree.map(jnp.asarray, kinematics), jnp.asarray(path), jnp.asarray(collided))
        following, reward, terminated, truncated = stepped(*inputs)
        expected, crashed, index, cut = oracle_step(racing, track, action, position, rates, path, collided)
        roundings = chain_roundings(jax.make_jaxpr(stepped)(*inputs))
        assert_computes_the_oracle(np.asarray(reward), expected, "a step's rewards", roundings=roundings)
        np.testing.assert_array_equal(np.asarray(terminated), crashed)
        np.testing.assert_array_equal(np.asarray(following.index), index)
        np.testing.assert_array_equal(np.asarray(truncated), cut)
        np.testing.assert_array_equal(np.asarray(following.passed), track.passed + (index != track.index))
    moved = index != track.index
    assert moved[kind == 0].mean() > 0.9 and not moved[(kind >= 1) & (kind <= 4)].any()
    assert crashed[kind == 1].mean() > 0.9 and crashed[kind == 2].mean() > 0.9 and crashed[kind == 5].any()


def test_commands_carry_each_drones_throttle_rates_lag_and_latency():
    """Ten steps of drawn actions as the simulator's commands, held to the race's rules in float64 loops:
    the throttle a share of the drone's own full thrust about its hover, the rates its limit's share
    through its lag, each command sent after the drone's latency, all over the simulator's ranges."""
    rng = np.random.default_rng([532, 6])
    thrust_to_weight, rate_limit = 14.5, 12.0
    with jax.enable_x64(new_val=True):
        track = tracks(RACING, 1, 64)[0]
        actions = rng.uniform(-1.3, 1.3, (10, 64, 4))
        step = jax.jit(lambda track, action: commanded(RACING, track, action, thrust_to_weight, rate_limit))
        sent = []
        for action in actions:
            track, command = step(track, jnp.asarray(action))
            sent.append(np.asarray(command))
        airframe = jax.tree.map(np.asarray, track.airframe)
        expected = np.zeros((10, 64, 4))
        for drone in range(64):
            strongest = airframe.thrust[drone] * thrust_to_weight / (1 + RACING.spread)
            rates, queue = np.zeros(3), [np.zeros(4)] * (RACING.delays[1] + 1)
            for index, action in enumerate(np.clip(actions[:, drone], -1, 1)):
                hover = 1 / strongest
                throttle = hover + action[0] * ((1 - hover) if action[0] >= 0 else hover)
                wanted = action[1:] * airframe.rates[drone] * RACING.rate_limit
                rates = rates + airframe.smoothing[drone] * (wanted - rates)
                command = np.r_[2 * throttle * strongest / thrust_to_weight - 1, rates / rate_limit]
                queue = [command, *queue[:-1]]
                expected[index, drone] = np.clip(queue[airframe.delay[drone]], -1, 1)
        roundings = chain_roundings(jax.make_jaxpr(step)(track, jnp.asarray(actions[0])))
        assert_computes_the_oracle(np.stack(sent), expected, "the commands", roundings=10 * roundings)
    assert set(np.unique(airframe.delay)) == set(range(RACING.delays[0], RACING.delays[1] + 1))


def test_courses_are_their_layouts_moved_or_random_loops():
    """Courses drawn at no difficulty are the layouts as published, centred, or random loops of 6 to 12
    gates between 1.5 and 6 m high, and every layout and loops are drawn; at full difficulty each layout's
    gate moves within the jitter and every gate stays at least 0.75 m above the ground."""
    racing = Racing(random_share=0.3)
    keys = jax.random.split(jax.random.key(2), 2048)
    draw = jax.jit(jax.vmap(lambda key, difficulty: drawn_track(racing, key, difficulty), (0, None)))
    layouts = {name: np.asarray(gates) - np.r_[np.asarray(gates)[:, :2].mean(0), 0.0]
               for name, gates in LAYOUTS.items()}
    reach = np.r_[racing.jitter[0], racing.jitter[0], racing.jitter[1]] + 1e-4

    def loop(gates):
        return 6 <= len(gates) <= GATES and np.all((gates[:, 2] >= 1.5 - 1e-5) & (gates[:, 2] <= 6 + 1e-5))

    seen = set()
    for course, count in zip(*map(np.asarray, draw(keys, 0.0)), strict=True):
        gates = course[:count]
        named = [name for name, layout in layouts.items()
                 if len(layout) == count and np.allclose(gates, layout, atol=1e-5)]
        assert len(named) == 1 or (not named and loop(gates))
        seen.update(named or ["loop"])
    assert seen == {*LAYOUTS, "loop"}
    for course, count in zip(*map(np.asarray, draw(keys, 1.0)), strict=True):
        gates = course[:count]
        floor = np.r_[-np.inf, -np.inf, 0.75]
        moved = [np.abs(gates - np.maximum(layout, floor)) for layout in layouts.values()
                 if len(layout) == count]
        assert loop(gates) or any(np.all(each <= reach) for each in moved)
        assert np.all(gates[:, 2] >= 0.75 - 1e-5)


def test_a_camera_is_hyperdrones_of_its_mounted_frame():
    """Each drone's camera is HyperDrone's camera basis (hyperdrone/_native/dynamics/impl.cpp
    `camera_base_one`, transcribed here in float64) of the body frame turned by the drone's mount: rolled,
    tilted up by its uptilt and turned, its field of view the drone's."""
    rng = np.random.default_rng([532, 7])
    with jax.enable_x64(new_val=True):
        track = tracks(RACING, 3, 64)[0]
        kinematics = Kinematics(jnp.asarray(rng.normal(0, 10, (64, 3))), jnp.zeros((64, 3)),
                                quaternion(*jnp.asarray(rng.normal(0, 1, (3, 64)))), jnp.zeros((64, 3)))
        computed = jax.jit(lambda track, kinematics: cameras(track, kinematics, 1.0))
        airframe = jax.tree.map(np.asarray, track.airframe)
        body = np.asarray(rotation(kinematics.orientation))
        expected = []
        for drone in range(64):
            roll, yaw = airframe.mount[drone]
            pitch = -airframe.uptilt[drone]
            cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)
            turned = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            tilted = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            mount = turned @ tilted @ np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            frame = body[drone] @ mount
            scale = 2 * np.tan(airframe.fov[drone] / 2)
            du, dv = -frame[:, 1] * scale, frame[:, 2] * scale
            first = frame[:, 0] - 0.5 * du + 0.5 * dv
            expected.append(np.r_[np.asarray(kinematics.position)[drone], first, du, -dv])
        roundings = chain_roundings(jax.make_jaxpr(computed)(track, kinematics))
        assert_computes_the_oracle(np.asarray(computed(track, kinematics)), np.stack(expected), "the cameras",
                                   roundings=roundings)


class Straight:
    """A flight backend whose drones fly straight at 0.4 m a control step, in three physics steps, ignoring
    the commands, each toward the point it is aimed at (`aim`)."""

    hz, thrust_to_weight, rate_limit, aspect = 60.0, 14.5, 12.0, 1.0

    def __init__(self, count: int):
        self.count = count

    def reset(self, mask, position, orientation):
        self.position = np.asarray(position)
        still = np.zeros((self.count, 3))
        return Kinematics(self.position, still, np.asarray(orientation), still)

    def aim(self, points):
        heading = np.asarray(points) - self.position
        self.heading = heading / np.linalg.norm(heading, axis=-1, keepdims=True)

    def step(self, commands):
        travelled = np.asarray([0.0, 0.4 / 3, 0.8 / 3, 0.4])[:, None]
        path = self.position[:, None] + travelled * self.heading[:, None]
        self.position = path[:, -1]
        upright = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (self.count, 1))
        still = np.zeros((self.count, 3))
        return Kinematics(self.position.copy(), 12 * self.heading, upright, still), path, None


def test_a_race_over_a_backend_flying_at_each_gate_passes_it():
    """A gate race (`GateRace`) over a backend flying each drone straight through its first gate's centre
    pays each its progress and passes the gate, its `passed` counting it, within the steps its distance
    takes and with no crash before."""
    backend = Straight(64)
    race = GateRace(backend, Racing(), views=("passed",))
    race.reset(5)
    track = race._track
    backend.aim(np.asarray(track.course)[np.arange(64), np.asarray(track.index)])
    passed, rewards = [], []
    for _ in range(12):
        step = race.step(jnp.zeros((64, 4)))
        assert not np.asarray(step.terminated).any()
        passed.append(np.asarray(step.observation["passed"]))
        rewards.append(np.asarray(step.reward))
    passed, rewards = np.stack(passed), np.stack(rewards)
    assert (passed[-1] == 1).all()
    first = passed.argmax(0)
    assert (rewards[first, np.arange(64)] > 0.9 * RACING.gate_bonus).all()
    assert (rewards[:first.min()] > 0.3).all()


@pytest.mark.parametrize("asked", [{"layouts": ("nowhere",)}, {"layouts": (), "random_share": 0.5}])
def test_a_race_of_unknown_or_no_courses_is_refused(asked):
    with pytest.raises(ValueError):
        Racing(**asked)
