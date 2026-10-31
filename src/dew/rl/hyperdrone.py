"""HyperDrone's parts as a flight simulator for `dew.rl.flight`'s gate race (Dew's build of
rl-tools/hyperdrone 0.1.10, `hyperdrone==0.1.10.post1`, the `hyperdrone` extra, on an NVIDIA GPU with
OptiX).

HyperDrone's own environment holds one gate an instance and detects no collisions, so the race composes
its two parts as its examples/drone_flythrough.py does: L2F's dynamics (`hyperdrone.dynamics.Sim`) under
collective thrust and body rates with ideal actuation, on the GPU (Dew's build compiles that control's
CUDA kernels, which 0.1.10's did not), and HyperLux's batch ray tracer drawing each drone's front camera
over a ground and its own course's gates, placed on the device each frame. The states, commands and
cameras cross between them and JAX by DLPack on the device. The course, reward and collisions are the
race's, in JAX.

The drone is L2F's `flightmare` quadrotor, the nearest it holds to a five-inch racer: 0.73 kg on arms
of 0.17 m, its four rotors lifting 12.1 times its weight at full command (parameters/dynamics/
flightmare.h). Ideal actuation reads its thrust-to-weight from its own parameters, which 0.1.10 derives
from the default model whatever model is loaded, so it is set from the loaded model's rotors
(`thrust_to_weight`), with the race's headroom for its randomized drones.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from dew.rl.flight import Gates, Kinematics

GROUND = (0.35, 0.45, 0.3), (0.45, 0.55, 0.35)
"""The ground's two checker colours, linear RGB."""
SKY = (0.55, 0.7, 0.9)
"""The background's colour, linear RGB."""
GATE = (0.95, 0.35, 0.1)
"""The gates' colour, every gate alike, as a race's are."""
SUNK = (0.0, 0.0, -1000.0)
"""Where a course's unused gates stand, out of every camera's sight."""


def thrust_to_weight(model: dict) -> float:
    """A loaded L2F model's full-command thrust over its weight (L2F's `ideal_rates`, multirotor.h), from
    its parameters' `dynamics.dynamics`."""
    command = model["action_limit"]["max"]
    thrust = sum(direction[2] * (low + linear * command + square * command ** 2)
                 for (low, linear, square), direction in zip(model["rotor_thrust_coefficients"],
                                                             model["rotor_thrust_directions"], strict=True))
    return float(thrust / (model["mass"] * np.linalg.norm(model["gravity"])))


def _gate(opening: float, frame: float) -> Any:  # noqa: ANN401
    """A square gate facing along x: four bars `frame` wide about an opening `opening` wide."""
    import hyperlux
    from hyperlux.primitives import box

    assembly, half, depth = hyperlux.ObjectAssembly(), opening / 2 + frame / 2, 0.1
    outer = opening + 2 * frame
    for size, position in (((depth, outer, frame), (0, 0, half)), ((depth, outer, frame), (0, 0, -half)),
                           ((depth, frame, opening), (0, half, 0)), ((depth, frame, opening), (0, -half, 0))):
        assembly.add_part(box(size, color=GATE), hyperlux.make_transform(position=position))
    return assembly


def _ground(extent: float, tile: float) -> Any:  # noqa: ANN401
    """A checkered square of ground `extent` metres a side about the origin, its tiles `tile` wide."""
    import hyperlux

    ground = hyperlux.Object(name="ground")
    count = int(extent / tile)
    for row in range(count):
        for column in range(count):
            x, y = -extent / 2 + row * tile, -extent / 2 + column * tile
            vertices = np.asarray([[x, y, 0], [x + tile, y, 0], [x + tile, y + tile, 0], [x, y + tile, 0]],
                                  np.float32)
            ground.add_mesh(hyperlux.Mesh(vertices, np.asarray([[0, 1, 2], [0, 2, 3]], np.int32),
                                          color=GROUND[(row + column) % 2]))
    return ground


class HyperDroneFlight:
    """`count` drones of `model` flown by HyperDrone (`dew.rl.flight.FlightBackend`), each control step of
    `1 / hz` seconds `substeps` physics steps, commanded up to the model's thrust-to-weight and
    `rate_limit` radians a second, each times `headroom` (the race's randomization's, 1 + `Racing.spread`);
    each drone's camera `size` pixels square over a ground of `extent` metres a side and its course's up to
    `gates` gates of `opening` and `frame` (the race's `Racing.opening` and `frame`). Its image is
    `[count, size, size, 3]` uint8."""

    aspect = 1.0

    def __init__(self, count: int, gates: int, *, size: int = 96, model: str = "flightmare",
                 hz: float = 30.0, substeps: int = 3, rate_limit: float = 10.0, headroom: float = 1.2,
                 opening: float = 1.5, frame: float = 0.25, extent: float = 200.0, tile: float = 5.0,
                 seed: int = 0, device: str = "cuda"):
        from hyperdrone import dynamics

        self.count, self.gates, self.hz, self.substeps = count, gates, hz, substeps
        shared = {"dynamics": {"model": model}, "integration": {"dt": 1 / (hz * substeps)}}
        loaded = dynamics.Sim(1, parameters={"dynamics": shared}, device=device, control="ctbr",
                              actuation="ideal")
        self.thrust_to_weight = thrust_to_weight(loaded.parameters["dynamics"]["dynamics"]) * headroom
        self.rate_limit = rate_limit * headroom
        ideal = {"thrust_to_weight": self.thrust_to_weight, "rate_limit": [self.rate_limit] * 3}
        self.sim = dynamics.Sim(count, parameters={"dynamics": {**shared, "ideal_rates": ideal}}, seed=seed,
                                device=device, control="ctbr", actuation="ideal")
        self._drawn = size, opening, frame, extent, tile
        self.renderer = None
        self._bases = jax.jit(self._placed)

    def _built(self) -> None:
        """The renderer, built the first time an image is asked for (a race that shows none needs none):
        the ground and sky, and each drone's gates as placements of its own overlay, its camera's alone."""
        import hyperlux
        from hyperlux.camera import RGB, CameraSpec

        size, opening, frame, extent, tile = self._drawn
        scene = hyperlux.Scene()
        scene.add_object(_ground(extent, tile))
        scene.add_light(hyperlux.SceneLight.directional(direction=(0.3, 0.2, -1.0), color=(1.0, 1.0, 1.0)))
        scene.add_light(hyperlux.SceneLight.directional(direction=(-0.5, 0.4, -0.3), color=(0.4, 0.4, 0.5)))
        scene.set_environment(ambient=(0.3, 0.3, 0.3), background=SKY)
        pool = hyperlux.AssetPool()
        gate = pool.add_assembly(_gate(opening, frame))
        self._rgb = RGB("rgba8")
        self.renderer = hyperlux.HyperLux(
            size, size, batch_size=self.count, fidelity="medium", num_probes=0,
            cameras=(CameraSpec("front", (self._rgb,)),), num_overlays=self.count,
            max_overlay_instances=4 * self.gates, max_overlays_per_camera=1)
        self.renderer.init(scene, asset_pool=pool)
        self._scene, self._pool = scene, pool
        placements = []
        for drone in range(self.count):
            self.renderer.attach(drone, drone)
            placements += [self.renderer.spawn(drone, gate) for _ in range(self.gates)]
        self._placements = self.renderer.placement_batch(placements)

    def _placed(self, gates: Gates) -> jax.Array:
        """Each placement's base `[count * gates, 12]`: a drone's course's gates where they stand, facing
        their way, the rest sunk out of sight."""
        yaw = gates.facing.astype(jnp.float32)
        cos, sin, zero, one = jnp.cos(yaw), jnp.sin(yaw), jnp.zeros_like(yaw), jnp.ones_like(yaw)
        rotation = jnp.stack([jnp.stack([cos, -sin, zero], -1), jnp.stack([sin, cos, zero], -1),
                              jnp.stack([zero, zero, one], -1)], -2)
        used = jnp.arange(self.gates)[None] < gates.count[:, None]
        center = jnp.where(used[..., None], gates.center.astype(jnp.float32), jnp.asarray(SUNK))
        return jnp.concatenate([rotation, center[..., None]], -1).reshape(-1, 12)

    def _state(self, name: str) -> jax.Array:
        """The sim's `name` state of every drone, on the default device: a GPU sim's copied out of the
        buffer it lends by DLPack before its next step overwrites it."""
        if self.sim.device == "cuda":
            return jnp.array(jax.dlpack.from_dlpack(self.sim.state_dlpack(name)), copy=True)
        return jnp.asarray(self.sim.state[name])

    def _kinematics(self) -> Kinematics:
        return Kinematics(*(self._state(name) for name in ("position", "linear_velocity", "orientation",
                                                           "angular_velocity")))

    def reset(self, mask: jax.Array, position: jax.Array, orientation: jax.Array) -> Kinematics:
        mask = np.asarray(mask, bool)
        self.sim.reset(mask)
        still = np.zeros((self.count, 3))
        for name, value in (("position", position), ("orientation", orientation), ("linear_velocity", still),
                            ("angular_velocity", still)):
            placed = np.where(mask[:, None], np.asarray(value, np.float32), self.sim.state[name])
            self.sim.state[name] = placed.astype(np.float32)
        return self._kinematics()

    def step(self, commands: jax.Array) -> tuple[Kinematics, jax.Array, None]:
        commands = jnp.asarray(commands, jnp.float32)
        commands = jax.block_until_ready(commands) if self.sim.device == "cuda" else np.asarray(commands)
        path = [self._state("position")]
        for _ in range(self.substeps):
            self.sim.step(commands)
            path.append(self._state("position"))
        return self._kinematics(), jnp.stack(path, 1), None

    def render(self, gates: Gates, cameras: jax.Array) -> jax.Array:
        import hyperlux

        if self.renderer is None:
            self._built()
        bases = jax.block_until_ready(self._bases(gates))
        self.renderer.set_pose(self._placements, hyperlux.Pose(bases))
        self.renderer.update()
        self.renderer.set_cameras(jax.block_until_ready(jnp.asarray(cameras, jnp.float32)))
        self.renderer.render()
        blocks = [jax.dlpack.from_dlpack(block) for block in self.renderer.outputs[self._rgb].blocks]
        return jnp.concatenate(blocks)[..., :3]


__all__ = ["HyperDroneFlight", "thrust_to_weight"]
