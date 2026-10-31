"""First-person gate racing (`dew.rl.flight`), each rule held to a float64 NumPy transcription of it apart
from Dew's code: a step's reward, passages, crashes and termination over drawn paths through, beside,
alongside and around the gates, and a race's over a backend's flight; the commands' throttle, rate limit,
lag and latency; the courses from their draws; and the camera as HyperDrone builds it from a mounted
frame. Hand-built steps near a gate hold the passing and striking rules themselves."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_computes_the_oracle, chain_roundings

from dew.rl.flight import (
    GATES,
    LAYOUTS,
    CourseDraw,
    GateRace,
    Kinematics,
    Racing,
    cameras,
    commanded,
    course_draws,
    course_of,
    quaternion,
    raced,
    rotation,
    started,
    struck,
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


def meets(begin: np.ndarray, end: np.ndarray, box) -> bool:
    """Whether the segment from `begin` to `end` enters the open box `box` (each axis's (low, high)): the
    parameters at which it is inside each axis's interval, intersected, leave some."""
    first, last = 0.0, 1.0
    for start, stop, (low, high) in zip(begin, end, box, strict=True):
        if start == stop:
            if not low < start < high:
                return False
            continue
        entered, left = sorted([(low - start) / (stop - start), (high - start) / (stop - start)])
        first, last = max(first, entered), min(last, left)
    return first < last


def frame_bars(racing: Racing) -> list:
    """A gate's four bars as boxes along its normal, its left and up from its centre, each grown by the
    drone's radius."""
    half, deep, grown = racing.opening / 2, racing.depth / 2, racing.radius
    edge = half + racing.frame
    top = [(-deep, deep), (-edge, edge), (half, edge)]
    bottom = [(-deep, deep), (-edge, edge), (-edge, -half)]
    left = [(-deep, deep), (half, edge), (-half, half)]
    right = [(-deep, deep), (-edge, -half), (-half, half)]
    return [[(low - grown, high + grown) for low, high in box] for box in (top, bottom, left, right)]


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
                local = [np.asarray([(point - course[gate]) @ normal, (point - course[gate]) @ left,
                                     point[2] - course[gate][2]]) for point in (begin, end)]
                if any(meets(*local, box) for box in frame_bars(racing)):
                    crashed = True
                before, after = local[0][0], local[1][0]
                if before * after > 0 or before == after:
                    continue
                point = local[0] + before / (before - after) * (local[1] - local[0])
                across = max(abs(point[1]), abs(point[2]))
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
    gate's opening, through its frame, beside it within the miss radius, far beside it, short of it,
    dipping to the ground, or alongside its plane sweeping across its frame's edge within the frame's
    depth and the drone's radius of the plane, or beyond them; and the kinds."""
    count = len(track.index)
    rows = np.arange(count)
    center = track.course[rows, track.index]
    normal, left = axes(track.facing[rows, track.index])
    half = racing.opening / 2
    inner, outer = half - racing.radius, half + racing.frame + racing.radius
    kind = rng.integers(0, 8, count)
    miss = racing.miss_radius
    bounds = np.asarray([(0.0, inner - 0.02), (inner + 0.02, outer - 0.02), (outer + 0.02, miss - 0.02),
                         (miss + 0.5, miss + 2.0), (0.0, inner), (0.0, inner), (0.0, inner), (0.0, inner)])
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
    path = np.where((kind == 5)[:, None, None], dipped, path)
    reach = racing.depth / 2 + racing.radius
    aside = np.where(kind == 6, rng.uniform(0, reach - 0.02, count), rng.uniform(reach + 0.02, 1.0, count))
    aside *= rng.choice([-1, 1], count)
    reached = rng.uniform(half + 0.02, outer + 0.5, count)
    swept = np.linspace(rng.uniform(0.0, inner - 0.02, count), reached, 4, -1)
    along = np.where(sideways[:, None, None], left[:, None], np.asarray([0.0, 0.0, 1.0]))
    way = rng.choice([-1, 1], (count, 1, 1))
    alongside = (center + aside[:, None] * normal)[:, None] + swept[..., None] * along * way
    return np.where((kind >= 6)[:, None, None], alongside, path), kind


def test_a_step_pays_passes_and_ends_as_the_race_rules():
    """Drones flown through their next gate's opening, its frame, beside it within the miss radius and far
    beside it, short of it, down to the ground, and alongside its plane across its frame's edge near the
    plane and away from it: each step's reward (progress, the gate's centring
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
    assert moved[kind == 0].any() and not moved[(kind >= 1) & (kind <= 4) | (kind >= 6)].any()
    assert crashed[kind == 1].any() and crashed[kind == 2].any() and crashed[kind == 5].any()
    assert crashed[kind == 6].any() and not crashed[kind == 7].all()


@pytest.mark.parametrize(("begin", "end", "crashes", "passes"), [
    ((-0.1, 0.5, 0.0), (-0.1, 0.9, 0.0), True, False),
    ((-0.4, 0.5, 0.0), (-0.4, 0.9, 0.0), False, False),
    ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), False, True),
    ((-1.0, 0.5, -0.5), (1.0, 0.5, -0.5), False, True),
    ((-1.0, 0.6, 0.0), (1.0, 0.6, 0.0), True, False),
    ((-1.0, 0.0, -1.1), (1.0, 0.0, -1.1), True, False),
    ((-1.0, 1.3, 1.3), (1.0, 1.3, 1.3), True, False),
    ((-1.0, 3.5, 0.0), (1.0, 3.5, 0.0), False, False),
    ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), False, False),
], ids=["alongside-into-a-bar", "alongside-clear", "centre", "inside", "inside-the-edge", "through-a-bar",
        "missed", "far-beside", "backwards"])
def test_a_gate_is_passed_or_struck_where_the_drone_meets_it(begin, end, crashes, passes):
    """A straight step near a gate facing along x (opening 1.5 m, frame 0.25 m wide and 0.1 m deep, the
    drone 0.2 m), its ends given from the gate's centre (along its normal, left and up): alongside the
    plane into a bar, within the frame's half depth and the drone's radius of it, it strikes the bar, and
    beyond them it does not; through the opening in its direction it passes, more than its radius inside
    the opening's edge, and within it it strikes the frame; through a bar it strikes it; beside the frame
    within the miss radius it misses the gate, a crash; far beside or through the opening backwards
    nothing happens."""
    racing = Racing()
    track = tracks(racing, 0, 1)[0]
    far = [[1000.0 + 10 * gate, 1000.0, 50.0] for gate in range(GATES - 1)]
    course = np.concatenate([[[10.0, 0.0, 2.0]], far])
    track = track.replace(course=jnp.asarray(course[None]), facing=jnp.zeros((1, GATES)),
                          count=jnp.asarray([GATES]), index=jnp.zeros(1, jnp.int32))
    path = jnp.asarray(course[0] + np.asarray([begin, end]))[None]
    crashed, passed, _ = struck(racing, track, path)
    assert (bool(crashed[0]), bool(passed[0])) == (crashes, passes)


def test_commands_carry_each_drones_throttle_rates_lag_and_latency():
    """Ten steps of drawn actions as the simulator's commands, held to the race's rules in float64 loops:
    the throttle a share of the drone's own full thrust about its hover, the rates its limit's share
    through its lag, each command sent after the drone's latency, the drone hovering until its first
    arrives, all over the simulator's ranges."""
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
            # The commands awaiting their latency, as the acceleration in g and rates the drone hovers at.
            rates, queue = np.zeros(3), [np.asarray([1.0, 0.0, 0.0, 0.0])] * (RACING.delays[1] + 1)
            for index, action in enumerate(np.clip(actions[:, drone], -1, 1)):
                hover = 1 / strongest
                throttle = hover + action[0] * ((1 - hover) if action[0] >= 0 else hover)
                wanted = action[1:] * airframe.rates[drone] * RACING.rate_limit
                rates = rates + airframe.smoothing[drone] * (wanted - rates)
                queue = [np.r_[throttle * strongest, rates], *queue[:-1]]
                sent_g, *sent_rates = queue[airframe.delay[drone]]
                expected[index, drone] = np.clip(np.r_[2 * sent_g / thrust_to_weight - 1,
                                                       np.asarray(sent_rates) / rate_limit], -1, 1)
        roundings = chain_roundings(jax.make_jaxpr(step)(track, jnp.asarray(actions[0])))
        assert_computes_the_oracle(np.stack(sent), expected, "the commands", roundings=10 * roundings)
    assert set(np.unique(airframe.delay)) == set(range(RACING.delays[0], RACING.delays[1] + 1))


def drawn_courses(rng: np.random.Generator, count: int) -> CourseDraw:
    """`count` courses' draws, each uniform in the range `CourseDraw` gives it."""
    return CourseDraw(which=rng.integers(0, len(LAYOUTS), count), moved=rng.uniform(-1, 1, (count, GATES, 3)),
                      looped=rng.random(count) < 0.5, count=rng.integers(6, GATES + 1, count),
                      radii=rng.uniform(12, 30, (count, 2)), turn=rng.uniform(0, 2 * np.pi, count),
                      along=rng.uniform(-0.25, 0.25, (count, GATES)),
                      across=rng.uniform(-0.1, 0.1, (count, GATES)),
                      height=rng.uniform(0, 4.5, (count, GATES)), anticlockwise=rng.random(count) < 0.5)


def oracle_course(racing: Racing, drawn: CourseDraw, row: int, difficulty: float) -> tuple[np.ndarray, int]:
    """A course of the draws' `row` by the race's rules, in float64 loops: its layout centred across, each
    gate moved and held 0.75 m high; or its loop's gates about the ellipse, the unused ones its first."""
    if not drawn.looped[row]:
        name = list(LAYOUTS)[drawn.which[row]]
        gates = np.asarray(LAYOUTS[name])
        centre = gates[:, :2].mean(0)
        course = []
        for index in range(GATES):
            x, y, z = gates[index] if index < len(gates) else gates[0]
            dx, dy, dz = drawn.moved[row, index] * difficulty
            course.append([x - centre[0] + dx * racing.jitter[0], y - centre[1] + dy * racing.jitter[0],
                           max(z + dz * racing.jitter[1], 0.75)])
        return np.asarray(course), len(gates)
    count = int(drawn.count[row])
    course = []
    for index in range(GATES):
        gate = index if index < count else 0
        angle = (gate + difficulty * drawn.along[row, gate]) * 2 * np.pi / count
        angle = (angle if drawn.anticlockwise[row] else -angle) + drawn.turn[row]
        spread = 1 + difficulty * drawn.across[row, gate]
        radius_x, radius_y = drawn.radii[row] * spread
        height = 1.5 + difficulty * drawn.height[row, gate]
        course.append([radius_x * np.cos(angle), radius_y * np.sin(angle), height])
    return np.asarray(course), count


@pytest.mark.parametrize("difficulty", [0.0, 0.4, 1.0])
def test_a_course_is_its_layout_moved_or_its_loop(difficulty):
    """Courses from drawn draws at a difficulty (`course_of`), each a published layout centred across, its
    gates moved by the jitter times the difficulty and held 0.75 m high, or a loop about a drawn ellipse,
    held to the race's rules in float64 loops: the gates to float64 rounding, their counts exactly."""
    rng = np.random.default_rng([532, 8])
    racing = Racing()
    with jax.enable_x64(new_val=True):
        drawn = drawn_courses(rng, 256)
        placed = jax.jit(jax.vmap(lambda drawn: course_of(racing, drawn, jnp.asarray(difficulty))))
        inputs = jax.tree.map(jnp.asarray, drawn)
        courses, counts = map(np.asarray, placed(inputs))
        expected = [oracle_course(racing, drawn, row, difficulty) for row in range(256)]
        roundings = chain_roundings(jax.make_jaxpr(placed)(inputs))
        assert_computes_the_oracle(courses, np.stack([course for course, _ in expected]), "the courses",
                                   roundings=roundings)
        np.testing.assert_array_equal(counts, [count for _, count in expected])


def test_courses_are_drawn_from_the_layouts_asked_for_and_loops():
    """A race's course draws take every layout it names and none other, and random loops at its share;
    each draw within its range."""
    names = ("split_s", "alpha_pilot")
    racing = Racing(layouts=names, random_share=0.3)
    drawn = jax.tree.map(np.asarray, jax.jit(jax.vmap(lambda key: course_draws(racing, key)))(
        jax.random.split(jax.random.key(2), 4096)))
    assert set(np.unique(drawn.which)) == {list(LAYOUTS).index(name) for name in names}
    assert drawn.looped.any() and not drawn.looped.all()
    assert set(np.unique(drawn.count)) == set(range(6, GATES + 1))
    for field, (low, high) in (("moved", (-1, 1)), ("radii", (12, 30)), ("turn", (0, 2 * np.pi)),
                               ("along", (-0.25, 0.25)), ("across", (-0.1, 0.1)), ("height", (0, 4.5))):
        values = getattr(drawn, field)
        assert np.all((values >= low) & (values <= high)), field


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
    the commands, each toward the point it is aimed at (`aim`); it keeps its last step's paths."""

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
        self.path = self.position[:, None] + travelled * self.heading[:, None]
        self.position = self.path[:, -1]
        upright = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (self.count, 1))
        still = np.zeros((self.count, 3))
        return Kinematics(self.position.copy(), 12 * self.heading, upright, still), self.path, None


def test_a_race_pays_and_passes_a_backends_flight_as_the_rules_do():
    """A gate race (`GateRace`) over a backend flying each drone straight through its first gate's centre:
    each step's reward, crash, and gates passed so far held to the race's rules in float64 loops
    (`oracle_step`) up to each drone's first crash, the rewards to float64 rounding and the rest exactly;
    every drone passes its gate."""
    count, rows = 64, np.arange(64)
    with jax.enable_x64(new_val=True):
        racing = Racing()
        backend = Straight(count)
        race = GateRace(backend, racing, views=("passed",))
        race.reset(5)
        track = jax.tree.map(np.asarray, race._track)
        backend.aim(track.course[rows, track.index])
        index, distance, passed, alive = track.index, track.distance, track.passed, np.ones(count, bool)
        computed, expected, roundings = [], [], 0
        for taken in range(12):
            current = track.replace(index=index, distance=distance, steps=np.full(count, taken),
                                    action=np.zeros((count, 4)))
            step = race.step(jnp.zeros((count, 4)))
            reward, crashed, index_after, _ = oracle_step(
                racing, current, np.zeros((count, 4)), backend.position, np.zeros((count, 3)), backend.path,
                np.zeros(count, bool))
            if taken == 0:
                inputs = (jax.tree.map(jnp.asarray, current), jnp.zeros((count, 4)),
                          Kinematics(*(jnp.zeros((count, size)) for size in (3, 3, 4, 3))),
                          jnp.asarray(backend.path), jnp.zeros(count, bool))
                roundings = chain_roundings(jax.make_jaxpr(lambda *inputs: raced(racing, *inputs))(*inputs))
            passed = passed + (index_after != index)
            computed.append(np.asarray(step.reward)[alive])
            expected.append(reward[alive])
            np.testing.assert_array_equal(np.asarray(step.terminated)[alive], crashed[alive])
            np.testing.assert_array_equal(np.asarray(step.observation["passed"])[alive], passed[alive])
            index = index_after
            distance = np.linalg.norm(track.course[rows, index] - backend.position, axis=-1)
            alive = alive & ~crashed
        assert_computes_the_oracle(np.concatenate(computed), np.concatenate(expected), "a flight's rewards",
                                   roundings=roundings)
    assert (passed >= 1).all()


@pytest.mark.parametrize("asked", [{"layouts": ("nowhere",)}, {"layouts": (), "random_share": 0.5}])
def test_a_race_of_unknown_or_no_courses_is_refused(asked):
    with pytest.raises(ValueError):
        Racing(**asked)
