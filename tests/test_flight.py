"""First-person gate racing (`dew.rl.flight`) against FlyDreamer's own task (AshishKumar4/FlyDreamer at
043c9ef382048278ccf26849e54814ef5510a737, embodied/envs/fpv, built from its run's config), its float64
run recorded on drawn inputs (tools/flydreamer_worker.py through tools/venv_reference.py, an oracle):
courses placed from the same draws, and drones flown through recorded states, paid, ended, shown and
commanded as FlyDreamer pays, ends, shows and commands them; and each drone's path struck against the
ground and its gates' frames."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import assert_computes_the_oracle, chain_roundings
from reference_runs import Oracle

from dew.rl.flight import (
    FLYDREAMER_COURSE,
    GateRace,
    Kinematics,
    Racing,
    commands,
    placed,
    privileged,
    raced,
    started,
    struck,
)
from dew.rl.simulator import simulator_rollout, simulator_start

FIXTURES = Path(__file__).parent / "fixtures" / "flydreamer"
DRONES, STEPS, MAX_STEPS = 4, 240, 100
"""A race's drones and steps, and its episodes' cut."""


def draw_course(seed: int) -> dict[str, np.ndarray]:
    """A course of twelve waypoints, each a step of 3 to 12 m from the last at a height of 0 to 13 m (some
    nearer than the spacing, some outside the heights), and the randomization's draws for each."""
    rng = np.random.default_rng([532, 2, seed])
    heading = rng.uniform(0, 2 * np.pi, 12)
    steps = rng.uniform(3, 12, (12, 1)) * np.stack([np.cos(heading), np.sin(heading)], -1)
    course = np.concatenate([np.cumsum(steps, 0), rng.uniform(0, 13, (12, 1))], -1)
    return {"course": course, "angle": rng.uniform(0, 2 * np.pi, 12), "radius": rng.uniform(0, 5, 12),
            "height": rng.uniform(-2, 2, 12)}


def quaternion(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Unit quaternions (w, x, y, z) of yaw, then pitch, then roll."""
    cr, sr, cp, sp, cy, sy = (f(angle / 2) for angle in (roll, pitch, yaw) for f in (np.cos, np.sin))
    return np.stack([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy], -1)


def flown(rng: np.random.Generator, start: np.ndarray, length: int) -> np.ndarray:
    """An episode's states `[length, 13]` (position, velocity, orientation, body rates) flying FlyDreamer's
    course from `start` at a drawn speed, wandering from it by a drawn reach, facing about its way."""
    points = np.concatenate([start[None], np.tile(np.asarray(FLYDREAMER_COURSE), (3, 1))])
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=-1))])
    distance = rng.uniform(0.8, 1.6) * np.arange(1, length + 1)
    path = np.stack([np.interp(distance, along, points[:, axis]) for axis in range(3)], -1)
    frequency, phase = rng.uniform(0.05, 0.3, (3, 3)), rng.uniform(0, 2 * np.pi, (3, 3))
    wander = np.sin(np.arange(length)[:, None, None] * frequency + phase).sum(-1) / np.sqrt(1.5)
    position = path + rng.uniform(0.2, 2.5) * wander
    velocity = 30 * np.diff(np.concatenate([start[None], position]), axis=0)
    yaw = np.arctan2(velocity[:, 1], velocity[:, 0]) + rng.normal(0, 0.5, length)
    orientation = quaternion(rng.normal(0, 0.3, length), rng.normal(0, 0.3, length), yaw)
    return np.concatenate([position, velocity, orientation, rng.normal(0, 2, (length, 3))], -1)


def draw_race(seed: int) -> dict[str, np.ndarray]:
    """Each drone's episodes flown one after another (`flown`), each from a start near the origin and
    ended by a collision at a drawn step or cut at `MAX_STEPS`, and every step's action in [-1.3, 1.3]."""
    rng = np.random.default_rng([532, 3, seed])
    states, collisions, starts = [], [], []
    for _ in range(DRONES):
        flights, ended, begun = [], [], []
        while sum(map(len, flights)) < STEPS:
            start = np.concatenate([rng.uniform([-1, -1, 0.2], [1, 1, 1.5]), np.zeros(3),
                                    quaternion(0.0, 0.0, rng.uniform(-0.3, 0.3)), np.zeros(3)])
            collides = rng.random() < 0.5
            length = int(rng.integers(1, MAX_STEPS + 1)) if collides else MAX_STEPS
            begun.append(start)
            flights.append(flown(rng, start[:3], length))
            ended.append(np.arange(length) == length - 1 if collides else np.zeros(length, bool))
        begun.append(begun[-1])
        states.append(np.concatenate(flights)[:STEPS])
        collisions.append(np.concatenate(ended)[:STEPS])
        starts.append(np.stack(begun))
    stacked = np.stack(states)
    most = max(map(len, starts))
    return {"position": stacked[..., :3], "velocity": stacked[..., 3:6], "orientation": stacked[..., 6:10],
            "rates": stacked[..., 10:], "collided": np.stack(collisions),
            "starts": np.stack([np.concatenate([held, np.repeat(held[-1:], most - len(held), 0)])
                                for held in starts]),
            "actions": rng.uniform(-1.3, 1.3, (STEPS, DRONES, 4)), "max_steps": np.asarray(MAX_STEPS)}


COURSE = Oracle(FIXTURES / "course.npz", draw_course, 8)
RACE = Oracle(FIXTURES / "race.npz", draw_race, 3)


def test_courses_are_placed_as_flydreamer_places_them():
    """Each course's waypoints offset by the momentum-smoothed draws, held to the heights and pushed out to
    the spacing as FlyDreamer's randomization places them from the same draws, to float64 rounding."""
    with jax.enable_x64(new_val=True):
        for seed, recorded in enumerate(COURSE.recorded()):
            drawn = draw_course(seed)
            racing = Racing(course=tuple(map(tuple, drawn["course"].tolist())))
            radius, angle = drawn["radius"], drawn["angle"]
            offsets = jnp.stack([radius * np.cos(angle), radius * np.sin(angle), drawn["height"]], -1)
            computed = jax.jit(lambda offsets, racing=racing: placed(racing, offsets))
            roundings = chain_roundings(jax.make_jaxpr(computed)(offsets))
            assert_computes_the_oracle(np.asarray(computed(offsets)), recorded["course"], f"course {seed}",
                                       roundings=roundings)


class Replay:
    """A flight backend answering recorded states (`draw_race`): each step every drone's next, each reset
    the selected drones' next starts; it keeps the commands it is given."""

    def __init__(self, drawn: dict[str, np.ndarray]):
        self.drawn, self.count = drawn, len(drawn["collided"])
        self.steps, self.episodes, self.commands, self.held = 0, np.zeros(self.count, int), [], None

    def kinematics(self) -> Kinematics:
        return Kinematics(*np.split(self.held, [3, 6, 10], axis=-1))

    def reset(self, mask, position, orientation):
        mask = np.asarray(mask)
        starts = self.drawn["starts"][np.arange(self.count), self.episodes]
        self.held = starts if self.held is None else np.where(mask[:, None], starts, self.held)
        self.episodes = self.episodes + mask
        return self.kinematics()

    def step(self, commands):
        self.commands.append(np.asarray(commands))
        before, at = self.held[:, :3], self.steps
        self.steps += 1
        names = ("position", "velocity", "orientation", "rates")
        self.held = np.concatenate([self.drawn[name][:, at] for name in names], -1)
        return self.kinematics(), np.stack([before, self.held[:, :3]], 1), self.drawn["collided"][:, at]


def test_a_gate_race_pays_ends_shows_and_commands_as_flydreamer():
    """Drones flown through recorded states, their courses FlyDreamer's run's and their collisions the
    recorded ones, as a simulator rolled out (`GateRace` over a backend replaying them,
    `simulator_rollout`): every step's reward FlyDreamer's (gates reached and their bonuses, progress, the
    rates' and their changes' penalties, time, the waypoint out of view, collisions) to float64 rounding,
    its episodes ended and cut on the same steps, each ended drone restarted on a fresh track, its
    privileged state FlyDreamer's bit for bit and its goal to float64 rounding, and each action
    commanded as FlyDreamer commands it."""
    racing = Racing(xy_radius=0.0, z_range=(0.0, 0.0), max_steps=MAX_STEPS, solid=False)
    with jax.enable_x64(new_val=True):
        for seed, recorded in enumerate(RACE.recorded()):
            drawn = draw_race(seed)
            backend = Replay(drawn)
            actions = jnp.asarray(drawn["actions"])

            def replayed(params, observation, key, memory, actions=actions):
                return actions[memory], {}, memory + 1

            race = GateRace(backend, racing, views=("state", "goal"))
            carry = simulator_start(race, jax.random.key(seed), jnp.zeros((), jnp.int32))
            _, steps = simulator_rollout(race, replayed, None, carry, STEPS)
            reached = jax.tree.map(lambda leaf: np.swapaxes(np.asarray(leaf), 0, 1), steps.following)
            kinematics = Kinematics(*(jnp.zeros((DRONES, size)) for size in (3, 3, 4, 3)))
            track = started(kinematics, jnp.zeros((DRONES, len(FLYDREAMER_COURSE), 3)))
            roundings = chain_roundings(jax.make_jaxpr(lambda track, kinematics: raced(
                racing, track, kinematics, jnp.zeros(DRONES, bool)))(track, kinematics))
            assert_computes_the_oracle(reached.reward, recorded["reward"], f"race {seed}'s rewards",
                                       roundings=roundings)
            np.testing.assert_array_equal(reached.terminated, recorded["terminal"])
            np.testing.assert_array_equal(reached.truncated, recorded["last"] & ~recorded["terminal"])
            np.testing.assert_array_equal(reached.observation["state"], recorded["state"])
            roundings = chain_roundings(jax.make_jaxpr(privileged)(track, kinematics))
            assert_computes_the_oracle(reached.observation["goal"], recorded["goal"], f"race {seed}'s goals",
                                       roundings=roundings)
            roundings = chain_roundings(jax.make_jaxpr(lambda action: commands(racing, action))(actions[0]))
            assert_computes_the_oracle(np.stack(backend.commands, 1), recorded["command"],
                                       f"race {seed}'s commands", roundings=roundings)


def test_a_path_through_a_frame_or_into_the_ground_strikes_it():
    """Drones crossing their gates' planes at drawn points, each crossing in the middle of a path whose
    ends lie a control step apart on either side: a crossing within the frame (from the opening's edge to
    the frame's outer edge, widened by the drone's radius either way) strikes it, one through the opening
    or wide of the frame does not, nor does a path that stays on one side; a path dipping within the
    drone's radius of the ground strikes it. Each gate faces along the course, the level line from the
    waypoint before it (the start, for the first) to the one after it."""
    racing = Racing()
    rng = np.random.default_rng([532, 4])
    count, gates = 256, len(FLYDREAMER_COURSE)
    heading = rng.uniform(-0.6, 0.6, (count, 1)) + np.cumsum(rng.uniform(-0.25, 0.25, (count, gates)), 1)
    steps = 12 * np.stack([np.cos(heading), np.sin(heading)], -1)
    courses = np.concatenate([np.cumsum(steps, 1), rng.uniform(6, 10, (count, gates, 1))], -1)
    gate = rng.integers(0, gates, count)
    rows = np.arange(count)
    before = np.where((gate == 0)[:, None], np.asarray(racing.start), courses[rows, gate - 1])
    after = courses[rows, (gate + 1) % gates]
    normal = np.concatenate([(after - before)[:, :2], np.zeros((count, 1))], -1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    lateral = np.stack([-normal[:, 1], normal[:, 0], np.zeros(count)], -1)
    # Kind 0 crosses through the opening, 1 the frame, 2 wide of it; 3 stays on one side of the plane;
    # 4 flies below the gates, dipping toward the ground.
    kind = rng.integers(0, 5, count)
    half, edge, margin = racing.gate_opening / 2, racing.gate_opening / 2 + racing.gate_frame, 0.02
    bounds = [(0.0, half - racing.radius - margin),
              (half - racing.radius + margin, edge + racing.radius - margin),
              (edge + racing.radius + margin, 3.0)]
    low, high = (np.asarray([bounds[min(each, 2)][side] for each in kind]) for side in (0, 1))
    across = rng.uniform(low, high)
    sideways = rng.random(count) < 0.5
    other = rng.uniform(-1, 1, count) * across
    widest = across * rng.choice([-1, 1], count)
    offset = (np.where(sideways, widest, other)[:, None] * lateral
              + np.where(sideways, other, widest)[:, None] * np.asarray([0.0, 0.0, 1.0]))
    velocity = rng.uniform(0.5, 1.5, (count, 1)) * normal * rng.choice([-1, 1], (count, 1))
    velocity += rng.normal(0, 0.1, (count, 3))
    times = np.asarray([-1.5, -0.5, 0.5, 1.5])[:, None]
    path = (courses[rows, gate] + offset)[:, None] + times * velocity[:, None]
    aside = 3 * np.abs(np.sum(velocity * normal, -1))
    path = np.where((kind == 3)[:, None, None], path + (aside[:, None] * normal)[:, None], path)
    dip = rng.uniform(0.0, 2 * racing.radius, count)
    below = np.repeat(path[:, :1], 4, 1) * np.asarray([1.0, 1.0, 0.0]) + np.asarray([0.0, 0.0, 2.5])
    below[:, 1, 2] = dip
    path = np.where((kind == 4)[:, None, None], below, path)
    expected = (kind == 1) | ((kind == 4) & (dip < racing.radius))
    computed = struck(racing, jnp.asarray(courses), jnp.asarray(path))
    np.testing.assert_array_equal(np.asarray(computed), expected)
    assert expected.any() and not expected.all()
