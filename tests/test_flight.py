"""First-person gate racing (`dew.rl.flight`), each rule held to a float64 NumPy transcription of it apart
from Dew's code: a step's reward, passages, crashes, finishes and termination over drawn paths through,
beside, alongside and around the gates and the pillars; the commands' throttle, rates and latency; the
camera as HyperDrone builds it from a mounted frame, and its image's development; and the teacher's view
of the next gates and the nearest pillars. Hand-built steps near a gate and a pillar hold the passing and
striking rules themselves; the courses keep their families' shapes; a drone left to hover flies and is not
cut short; and a batch of races restarts only the drones it is told to."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_computes_the_oracle, chain_roundings

from dew.rl.env import env_rollout, start
from dew.rl.flight import (
    ASPECTS,
    FAMILIES,
    GATES,
    INTERFACE,
    LAYOUTS,
    NEAREST,
    OBSTACLES,
    TEACHER,
    FlightEnv,
    GateRace,
    Racing,
    cameras,
    commanded,
    corners,
    developed,
    drawn_course,
    nearest,
    privileged,
    quaternion,
    raced,
    started,
    struck,
)
from dew.rl.quadrotor import GRAVITY, rotation

RACING = Racing()
DRONES = 256


def races(racing: Racing, seed: int, count: int = DRONES, difficulty: float = 1.0):
    """`count` drones' episodes drawn at `difficulty` and full randomization."""
    keys = jax.random.split(jax.random.key(seed), count)
    return jax.vmap(lambda key: started(racing, key, jnp.asarray(difficulty), jnp.asarray(1.0)))(keys)


def numpy_of(tree):
    """A tree's arrays in NumPy, its keys as they are."""
    return jax.tree.map(lambda leaf: leaf if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key)
                        else np.asarray(leaf), tree)


def axes(facing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (np.stack([np.cos(facing), np.sin(facing), np.zeros_like(facing)], -1),
            np.stack([-np.sin(facing), np.cos(facing), np.zeros_like(facing)], -1))


def meets(begin: np.ndarray, end: np.ndarray, box) -> bool:
    """Whether the segment from `begin` to `end` enters the open box `box` (each axis's (low, high)): the
    parameters at which it is inside each axis's interval, intersected, leave some."""
    first, last = 0.0, 1.0
    for start_, stop, (low, high) in zip(begin, end, box, strict=True):
        if start_ == stop:
            if not low < start_ < high:
                return False
            continue
        entered, left = sorted([(low - start_) / (stop - start_), (high - start_) / (stop - start_)])
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


def pillar_hit(racing: Racing, begin: np.ndarray, end: np.ndarray, pillar: np.ndarray) -> bool:
    """Whether the segment's closest approach across to a pillar's axis comes within the radii, below its
    top grown by the drone's radius there."""
    x, y, radius, height = pillar
    if radius <= 0:
        return False
    along = end[:2] - begin[:2]
    length = along @ along
    t = 0.0 if length == 0 else min(max((np.asarray([x, y]) - begin[:2]) @ along / length, 0.0), 1.0)
    point = begin + t * (end - begin)
    return np.hypot(point[0] - x, point[1] - y) < radius + racing.radius and point[2] < height + racing.radius


def oracle_step(racing: Racing, race, path):
    """One step of the race's rules for each drone, in float64 loops: its reward, whether it ended (a crash
    or an open track's last gate passed), the gate it heads for after, and whether its episode was cut."""
    half, inner = racing.opening / 2, racing.opening / 2 - racing.radius
    outer = half + racing.frame + racing.radius
    rewards, ends, indices, cuts = [], [], [], []
    for drone in range(len(race.index)):
        center, facing = race.course.center[drone], race.course.facing[drone]
        count, closed = int(race.course.count[drone]), bool(race.course.closed[drone])
        target = int(race.index[drone])
        crashed, passed, centring = False, False, 0.0
        for begin, end in zip(path[drone, :-1], path[drone, 1:], strict=True):
            if end[2] < racing.radius or end[2] > racing.ceiling:
                crashed = True
            if any(pillar_hit(racing, begin, end, pillar) for pillar in race.course.obstacles[drone]):
                crashed = True
            for gate in range(count):
                normal, left = axes(facing[gate])
                local = [np.asarray([(point - center[gate]) @ normal, (point - center[gate]) @ left,
                                     point[2] - center[gate][2]]) for point in (begin, end)]
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
        position = race.flight.position[drone]
        reward = racing.progress_scale * (race.distance[drone] - np.linalg.norm(center[target] - position))
        finished = False
        if passed:
            reward += racing.gate_bonus * (1 - centring / half)
            if closed:
                target = (target + 1) % count
            elif target + 1 >= count:
                finished = True
            else:
                target += 1
        reward -= racing.rate_cost * np.linalg.norm(race.flight.angular_velocity[drone]) / racing.hz
        reward -= racing.smoothness_cost * np.sum((np.clip(race.action_taken[drone], -1, 1)
                                                    - race.action[drone]) ** 2)
        reward -= racing.crash_penalty if crashed else 0.0
        rewards.append(reward)
        ends.append(crashed or finished)
        indices.append(target)
        cuts.append(not (crashed or finished) and race.steps[drone] + 1 >= racing.max_steps)
    return np.asarray(rewards), np.asarray(ends), np.asarray(indices), np.asarray(cuts)


@dataclasses.dataclass
class Drawn:
    """A batch of races in NumPy, with the action each drone took this step."""

    course: object
    index: np.ndarray
    distance: np.ndarray
    steps: np.ndarray
    action: np.ndarray
    flight: object
    action_taken: np.ndarray


def drawn_paths(racing: Racing, race, rng: np.random.Generator):
    """Each drone's path through a step, four physics steps of a straight line, by its kind: through its
    gate's opening, through its frame, beside it within the miss radius, far beside it, short of it,
    dipping to the ground, alongside its plane across its frame's edge within the frame's half depth and
    the drone's radius of the plane or beyond them, or through a pillar it is moved onto; and the kinds
    and the pillars, each drone's first moved onto its path where its kind is the last."""
    count = len(race.index)
    rows = np.arange(count)
    center = race.course.center[rows, race.index]
    normal, left = axes(race.course.facing[rows, race.index])
    half = racing.opening / 2
    inner, outer = half - racing.radius, half + racing.frame + racing.radius
    kind = rng.integers(0, 9, count)
    miss = racing.miss_radius
    bounds = np.asarray([(0.0, inner - 0.02), (inner + 0.02, outer - 0.02), (outer + 0.02, miss - 0.02),
                         (miss + 0.5, miss + 2.0), (0.0, inner), (0.0, inner), (0.0, inner), (0.0, inner),
                         (0.0, inner)])
    across = rng.uniform(bounds[kind, 0], bounds[kind, 1])
    sideways = rng.random(count) < 0.5
    other = rng.uniform(-1, 1, count) * across
    widest = across * rng.choice([-1, 1], count)
    offset = (np.where(sideways, widest, other)[:, None] * left
              + np.where(sideways, other, widest)[:, None] * np.asarray([0.0, 0.0, 1.0]))
    lead = rng.uniform(0.3, 1.2, count)
    times = np.asarray([-1.5, -0.5, 0.5, 1.0, 1.5]) + np.where(kind == 4, -3.0, 0.0)[:, None]
    path = (center + offset)[:, None] + times[..., None] * (lead[:, None] * normal)[:, None]
    dipped = path.copy()
    dipped[:, 2, 2] = rng.uniform(0, 2 * racing.radius, count)
    path = np.where((kind == 5)[:, None, None], dipped, path)
    reach = racing.depth / 2 + racing.radius
    aside = np.where(kind == 6, rng.uniform(0, reach - 0.02, count), rng.uniform(reach + 0.02, 1.0, count))
    aside *= rng.choice([-1, 1], count)
    reached = rng.uniform(half + 0.02, outer + 0.5, count)
    swept = np.linspace(rng.uniform(0.0, inner - 0.02, count), reached, 5, axis=-1)
    along = np.where(sideways[:, None, None], left[:, None], np.asarray([0.0, 0.0, 1.0]))
    way = rng.choice([-1, 1], (count, 1, 1))
    alongside = (center + aside[:, None] * normal)[:, None] + swept[..., None] * along * way
    path = np.where(((kind == 6) | (kind == 7))[:, None, None], alongside, path)
    obstacles = race.course.obstacles.copy()
    hit = kind == 8
    obstacles[hit, 0, :2] = path[hit, 2, :2] + rng.uniform(-0.3, 0.3, (hit.sum(), 2))
    obstacles[hit, 0, 2:] = np.c_[rng.uniform(0.2, 0.6, hit.sum()), path[hit, 2, 2] + 2.0]
    return path, kind, obstacles


def test_a_step_pays_passes_ends_and_finishes_as_the_race_rules():
    """Drones on drawn courses flown through their next gate's opening, its frame, beside it within the miss
    radius and far beside it, short of it, down to the ground, alongside its plane near and away from its
    frame, and into a pillar: each step's reward (progress, the gate's centring bonus, the rates' and the
    action's costs, a crash's penalty), each end (a crash, or an open track's last gate passed), each gate
    headed for after the step and each cut held to the race's rules in float64 loops: rewards to float64
    rounding, the rest exactly."""
    rng = np.random.default_rng([532, 5])
    with jax.enable_x64(new_val=True):
        racing = Racing(episode_seconds=0.05)
        race = numpy_of(races(racing, 0))
        # Some drones head for an open track's last gate, to finish it.
        last = ~race.course.closed & (rng.random(DRONES) < 0.3)
        index = np.where(last, race.course.count - 1, race.index)
        drawn = Drawn(race.course, index, rng.uniform(0, 4, DRONES), rng.integers(0, 3, DRONES),
                      rng.uniform(-1, 1, (DRONES, 4)), race.flight, rng.uniform(-1.2, 1.2, (DRONES, 4)))
        path, kind, obstacles = drawn_paths(racing, drawn, rng)
        drawn.course = dataclasses.replace(race.course, obstacles=obstacles)
        flight = dataclasses.replace(race.flight, position=path[:, -1],
                                     angular_velocity=rng.normal(0, 3, (DRONES, 3)))
        drawn.flight = flight
        inputs = dataclasses.replace(race, course=drawn.course, index=index, distance=drawn.distance,
                                     steps=drawn.steps, action=drawn.action, flight=flight)
        stepped = jax.jit(jax.vmap(lambda race, action, path: raced(racing, race, action, path)))
        arguments = (jax.tree.map(jnp.asarray, inputs), jnp.asarray(drawn.action_taken), jnp.asarray(path))
        following, reward, ended, truncated = stepped(*arguments)
        expected, ends, indices, cut = oracle_step(racing, drawn, path)
        roundings = chain_roundings(jax.make_jaxpr(stepped)(*arguments))
        assert_computes_the_oracle(np.asarray(reward), expected, "a step's rewards", roundings=roundings)
        np.testing.assert_array_equal(np.asarray(ended), ends)
        np.testing.assert_array_equal(np.asarray(following.index), indices)
        np.testing.assert_array_equal(np.asarray(truncated), cut)
    moved = indices != index
    finished = ends & last & (kind == 0)
    assert moved[kind == 0].any() and finished.any()
    assert ends[kind == 1].any() and ends[kind == 2].any() and ends[kind == 5].any() and ends[kind == 8].any()
    assert ends[kind == 6].any() and not ends[kind == 7].all()


def one_gate(begin, end, pillar=(0.0, 0.0, 0.0, 0.0)):
    """A drone's race on a course whose first gate stands 2 m up at (10, 0) facing along x, its others far
    off, with one pillar `pillar` (x, y, radius, height), and its path from `begin` to `end` given from the
    gate's centre."""
    race = jax.tree.map(lambda leaf: leaf[0], races(RACING, 0, 1))
    far = [[1000.0 + 10 * gate, 1000.0, 50.0] for gate in range(GATES - 1)]
    center = np.concatenate([[[10.0, 0.0, 2.0]], far])
    obstacles = np.zeros((OBSTACLES, 4))
    obstacles[0] = pillar
    course = dataclasses.replace(race.course, center=jnp.asarray(center), facing=jnp.zeros(GATES),
                                 count=jnp.asarray(GATES), closed=jnp.asarray(data=True),
                                 obstacles=jnp.asarray(obstacles))
    race = dataclasses.replace(race, course=course, index=jnp.asarray(0))
    return race, jnp.asarray(center[0] + np.asarray([begin, end]))


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
    drone 0.2 m), its ends given from the gate's centre (along its normal, left and up): alongside the plane
    into a bar, within the frame's half depth and the drone's radius of it, it strikes the bar, and beyond
    them it does not; through the opening in its direction it passes, more than its radius inside the
    opening's edge, and within it it strikes the frame; through a bar it strikes it; beside the frame within
    the miss radius it misses the gate, a crash; far beside or through the opening backwards nothing
    happens."""
    race, path = one_gate(begin, end)
    crashed, passed, _ = struck(RACING, race, path)
    assert (bool(crashed), bool(passed)) == (crashes, passes)


@pytest.mark.parametrize(("pillar", "crashes"), [
    ((5.0, 0.0, 0.3, 6.0), True),
    ((5.0, 0.6, 0.3, 6.0), False),
    ((5.0, 0.4, 0.3, 6.0), True),
    ((5.0, 0.0, 0.3, 1.5), False),
    ((5.0, 0.0, 0.3, 1.9), True),
], ids=["through", "beside", "grazing", "over-the-top", "skimming-the-top"])
def test_a_pillar_is_struck_where_the_drone_meets_it(pillar, crashes):
    """A straight step 2 m up from 4 to 6 m along x past a pillar at x = 5 (radius 0.3, the drone 0.2):
    through it or within the radii of its axis it strikes it, beside it it does not; over a pillar 1.5 m
    high it passes, over one 1.9 m high (within the drone's radius of its top) it strikes it."""
    race, _ = one_gate((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), pillar)
    crashed, _, _ = struck(RACING, race, jnp.asarray([[4.0, 0.0, 2.0], [6.0, 0.0, 2.0]]))
    assert bool(crashed) == crashes


def test_commands_carry_each_drones_throttle_rates_and_latency():
    """Ten steps of drawn actions as the flight controller's commands, held to the race's rules in float64
    loops: the collective thrust a share of the drone's hover below zero and of the span to the nominal
    drone's full thrust above, the rates their share of the drone's rate limit, each command sent after the
    drone's latency, the drone hovering until its first arrives."""
    rng = np.random.default_rng([532, 6])
    with jax.enable_x64(new_val=True):
        race = races(RACING, 1, 64)
        actions = rng.uniform(-1.3, 1.3, (10, 64, 4))
        step = jax.jit(jax.vmap(lambda race, action: commanded(RACING, race, action)))
        sent = []
        for action in actions:
            race, collective, rates = step(race, jnp.asarray(action))
            sent.append(np.c_[np.asarray(collective), np.asarray(rates)])
        kept = numpy_of(race)
        full = RACING.quadrotor.nominal_thrust
        expected = np.zeros((10, 64, 4))
        for drone in range(64):
            hover = kept.body.mass[drone] * GRAVITY
            queue = [np.asarray([hover, 0.0, 0.0, 0.0])] * (RACING.delays[1] + 1)
            for index, action in enumerate(np.clip(actions[:, drone], -1, 1)):
                collective = hover + action[0] * ((full - hover) if action[0] >= 0 else hover)
                queue = [np.r_[collective, action[1:] * kept.rate_limit[drone]], *queue[:-1]]
                expected[index, drone] = queue[kept.delay[drone]]
        roundings = chain_roundings(jax.make_jaxpr(step)(race, jnp.asarray(actions[0])))
        assert_computes_the_oracle(np.stack(sent), expected, "the commands", roundings=roundings)
    assert set(np.unique(kept.delay)) == set(range(RACING.delays[0], RACING.delays[1] + 1))


def test_a_camera_is_hyperdrones_of_its_mounted_frame():
    """Each drone's camera is HyperDrone's camera basis (hyperdrone/_native/dynamics/impl.cpp
    `camera_base_one`, transcribed here in float64) of the body frame turned by the drone's mount: rolled,
    tilted up by its uptilt and turned, its field of view the drone's; at its shutter's open, of the drone
    moved back along its velocity and turned back by its body rates."""
    rng = np.random.default_rng([532, 7])
    with jax.enable_x64(new_val=True):
        race = races(RACING, 3, 64)
        flight = dataclasses.replace(race.flight, position=jnp.asarray(rng.normal(0, 10, (64, 3))),
                                     velocity=jnp.asarray(rng.normal(0, 8, (64, 3))),
                                     orientation=quaternion(*jnp.asarray(rng.normal(0, 1, (3, 64)))),
                                     angular_velocity=jnp.asarray(rng.normal(0, 4, (64, 3))))
        race = dataclasses.replace(race, flight=flight)
        computed = jax.jit(jax.vmap(lambda race: (cameras(race, 1.0),
                                                  cameras(race, 1.0, race.camera.shutter))))
        camera = jax.tree.map(np.asarray, race.camera)
        moved = jax.tree.map(np.asarray, flight)
        expected = []
        for before in (False, True):
            for drone in range(64):
                roll, yaw = camera.mount[drone]
                pitch = -camera.uptilt[drone]
                cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
                cy, sy = np.cos(yaw), np.sin(yaw)
                turned = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
                tilted = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
                mount = turned @ tilted @ np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
                body = np.asarray(rotation(jnp.asarray(moved.orientation[drone])))
                position = moved.position[drone]
                if before:
                    shutter = camera.shutter[drone]
                    position = position - shutter * moved.velocity[drone]
                    back = -shutter * moved.angular_velocity[drone]
                    angle = np.linalg.norm(back)
                    x, y, z = back / angle
                    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                    body = body @ (np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k)
                frame = body @ mount
                scale = 2 * np.tan(camera.fov[drone] / 2)
                du, dv = -frame[:, 1] * scale, frame[:, 2] * scale
                first = frame[:, 0] - 0.5 * du + 0.5 * dv
                expected.append(np.r_[position, first, du, -dv])
        roundings = chain_roundings(jax.make_jaxpr(computed)(race))
        assert_computes_the_oracle(np.concatenate([np.asarray(each) for each in computed(race)]),
                                   np.stack(expected), "the cameras", roundings=roundings)


def test_an_image_is_developed_as_its_camera_does():
    """A drawn image developed by drawn cameras: exposure and white balance, then gamma, then read noise, the
    result rounded to 8 bits, held to a float32 transcription: every pixel within one level of it."""
    rng = np.random.default_rng([532, 8])
    race = races(RACING, 4, 16)
    image = rng.integers(0, 256, (16, 12, 12, 3), np.uint8)
    keys = jax.random.split(jax.random.key(5), 16)
    shown = np.asarray(jax.jit(jax.vmap(lambda race, image, key: developed(image, race.camera, key)))(
        race, jnp.asarray(image), keys))
    camera = jax.tree.map(np.asarray, race.camera)
    for drone in range(16):
        noise = np.asarray(jax.random.normal(keys[drone], (12, 12, 3)))
        light = image[drone].astype(np.float32) / 255 * camera.exposure[drone] * camera.white_balance[drone]
        light = np.clip(light, 0, 1) ** camera.gamma[drone] + camera.noise[drone] * noise
        expected = np.round(np.clip(light, 0, 1) * 255)
        assert np.abs(shown[drone].astype(np.float64) - expected).max() <= 1


def test_the_teacher_sees_the_next_gates_and_nearest_pillars_from_its_body():
    """The corners of each drone's next two gates' openings and its nearest three pillars (each its axis's
    closest point at the drone's height and its radius), in the body frame, held to a float64 transcription;
    the teacher's observation keeps its first race's 43 entries first."""
    with jax.enable_x64(new_val=True):
        race = races(RACING, 5, 64)
        shown = jax.jit(jax.vmap(lambda race: (corners(RACING, race), nearest(race),
                                               privileged(RACING, race))))
        gates, pillars, views = shown(race)
        gates, pillars, views = np.asarray(gates), np.asarray(pillars), jax.tree.map(np.asarray, views)
        kept = numpy_of(race)
        half = RACING.opening / 2
        expected_gates, expected_pillars = [], []
        for drone in range(64):
            to_body = np.asarray(rotation(jnp.asarray(kept.flight.orientation[drone]))).T
            position = kept.flight.position[drone]
            count, closed = int(kept.course.count[drone]), bool(kept.course.closed[drone])
            seen = []
            for step in range(2):
                gate = ((kept.index[drone] + step) % count if closed
                        else min(kept.index[drone] + step, count - 1))
                _, left = axes(kept.course.facing[drone, gate])
                for sideways, upward in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
                    offset = (sideways * left + upward * np.r_[0, 0, 1]) * half
                    corner = kept.course.center[drone, gate] + offset
                    seen.append(to_body @ (corner - position))
            expected_gates.append(np.concatenate(seen))
            active = [pillar for pillar in kept.course.obstacles[drone] if pillar[2] > 0]
            axis = [np.r_[pillar[:2], min(max(position[2], 0.0), pillar[3])] for pillar in active]
            order = np.argsort([np.linalg.norm(point - position) for point in axis], kind="stable")[:NEAREST]
            near = [np.r_[to_body @ (axis[k] - position), active[k][2]] for k in order]
            near += [np.zeros(4)] * (NEAREST - len(near))
            expected_pillars.append(np.concatenate(near))
        roundings = chain_roundings(jax.make_jaxpr(shown)(race))
        assert_computes_the_oracle(gates, np.stack(expected_gates), "the gates' corners", roundings=roundings)
        assert_computes_the_oracle(pillars, np.stack(expected_pillars), "the pillars", roundings=roundings)
    teacher = views["teacher"]
    assert teacher.shape == (64, TEACHER)
    np.testing.assert_array_equal(teacher[:, 33:37], np.asarray(race.action))
    np.testing.assert_array_equal(teacher[:, 37:INTERFACE], views["airframe"])


@pytest.mark.parametrize("difficulty", [0.0, 0.5, 1.0])
def test_courses_keep_their_families_shapes(difficulty):
    """4096 courses drawn at a difficulty from every family: each layout course is a published layout
    (centred across) moved by at most the jitter times the difficulty and held 0.75 m high; each loop closed,
    6 to 12 gates; each chained track open, 8 to 16 gates, each 7 to 16 m on from the last (a split-S's 2.5
    to 4 m below and a metre behind, at first) and between 1 and 20 m high, no two of its gates within 3 m
    across and 2.5 m in height; each figure-eight closed, 8 to 12 gates; the gates of a loop or a
    figure-eight at least 2 m apart; every pillar at least 3 m and its radius from every gate, its height its
    radius times one of the aspects."""
    racing = Racing(families=tuple((family, 1.0) for family in FAMILIES))
    keys = jax.random.split(jax.random.key(6), 4096)
    courses = jax.tree.map(np.asarray, jax.jit(jax.vmap(
        lambda key: drawn_course(racing, key, jnp.asarray(difficulty))))(keys))
    layouts = {name: np.asarray(gates) - np.r_[np.asarray(gates)[:, :2].mean(0), 0.0]
               for name, gates in LAYOUTS.items()}
    reach = difficulty * np.r_[racing.jitter[0], racing.jitter[0], racing.jitter[1]] + 1e-4
    floor = np.r_[-np.inf, -np.inf, 0.75]
    seen = set()
    for row in range(4096):
        family = FAMILIES[courses.family[row]]
        seen.add(family)
        count = int(courses.count[row])
        gates = courses.center[row, :count]
        assert bool(courses.closed[row]) == (family != "chain")
        if family == "layout":
            assert any(len(layout) == count and np.all(np.abs(gates - np.maximum(layout, floor)) <= reach)
                       for layout in layouts.values())
        elif family == "loop":
            assert 6 <= count <= 12
        elif family == "figure8":
            assert 8 <= count <= 12
        else:
            assert 8 <= count <= GATES and np.all((gates[:, 2] >= 1 - 1e-5) & (gates[:, 2] <= 20 + 1e-5))
            steps = np.linalg.norm(np.diff(gates[:, :2], axis=0), axis=-1)
            split = np.abs(steps - 1.0) < 1e-4
            assert np.all(split | ((steps >= 7 - 1e-4) & (steps <= 16 + 1e-4)))
            across = np.linalg.norm(gates[:, None, :2] - gates[None, :, :2], axis=-1)
            height = np.abs(gates[:, None, 2] - gates[None, :, 2])
            assert not np.any((across < 3.0) & (height < 2.5) & ~np.eye(count, dtype=bool))
        if family in ("loop", "figure8"):
            apart = np.linalg.norm(gates[:, None] - gates[None], axis=-1)
            assert np.all(apart[~np.eye(count, dtype=bool)] > 2.0)
        pillars = courses.obstacles[row][courses.obstacles[row, :, 2] > 0]
        aspect = pillars[:, 3] / pillars[:, 2]
        assert np.all(np.min(np.abs(aspect[:, None] - np.asarray(ASPECTS)[None]), -1) < 1e-4)
        if len(pillars):
            across = np.linalg.norm(pillars[:, None, :2] - gates[None, :, :2], axis=-1)
            assert np.all(across > 3.0 + pillars[:, 2:3])
    assert seen == set(FAMILIES)


def test_a_hovering_drone_flies_and_is_cut_at_its_time():
    """Drones at no randomization started hovering on simple loops and left to hover (zero action) for an
    episode's steps: none crashes before its episode is cut, each cut exactly at its time."""
    racing = Racing(families=(("loop", 1.0),), flying_share=0.0, obstacles=0, episode_seconds=1.0)
    env = FlightEnv(racing, ("state",), difficulty=0.0, ramp=0.0)

    def still(params, observation, key, memory):
        return jnp.zeros(4), {}, memory

    batched = jax.vmap(lambda key: env_rollout(env, still, None, start(env, key), racing.max_steps,
                                               reset=False))
    _, steps = jax.jit(batched)(jax.random.split(jax.random.key(7), 32))
    ended = np.asarray(steps.following.terminated | steps.following.truncated)
    assert not np.asarray(steps.following.terminated).any()
    np.testing.assert_array_equal(ended.argmax(-1), racing.max_steps - 1)


def test_a_race_restarts_only_the_drones_it_is_told_to():
    """A batch of 16 races without images, stepped, then told to restart half its drones: those start fresh
    episodes (their steps none, new courses) while the others' episodes are unchanged."""
    race = GateRace(None, Racing(), views=("state", "passed"), count=16)
    race.reset(0)
    race.step(jnp.zeros((16, 4)))
    before = race._races
    mask = np.arange(16) % 2 == 0
    race.restart(jnp.asarray(mask))
    after = race._races
    np.testing.assert_array_equal(np.asarray(after.steps), np.where(mask, 0, np.asarray(before.steps)))
    kept, held = np.asarray(after.course.center), np.asarray(before.course.center)
    np.testing.assert_array_equal(kept[~mask], held[~mask])
    assert not np.array_equal(kept[mask], held[mask])


@pytest.mark.parametrize("asked", [{"families": (("nowhere", 1.0),)}, {"layouts": ("nowhere",)},
                                   {"families": ()}, {"families": (("loop", 0.0),)}])
def test_a_race_of_unknown_or_no_courses_is_refused(asked):
    """A race of an unknown family or layout, of no family, or of a family at no weight is refused."""
    with pytest.raises(ValueError):
        Racing(**asked)


def test_a_race_without_a_renderer_shows_no_image():
    """A gate race without a renderer asked for images is refused."""
    with pytest.raises(ValueError):
        GateRace(None, Racing(), views=("image",), count=4)
