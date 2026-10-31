"""HyperLux's batch ray tracer (rl-tools' hyperlux 0.1.10, the `hyperlux` extra, on an NVIDIA GPU with OptiX)
drawing each drone's front camera of the gate race (`dew.rl.flight.Renderer`), its scene built from the GLB
assets `tools/fpv_assets.py` writes.

HyperLux draws one scene for every camera, its environment and lights shared; a camera's own content is the
overlay attached to it, whose placements move each frame (README; `HyperLux.attach`, `spawn`, `set_pose`).
Each drone's overlay holds every look of its scene: the gates in each of their looks and the lit gate in each
of its own, the grounds, the skies, the pillars of each aspect in each material and the props; a frame shows
the looks its drone's episode drew (`dew.rl.flight.Camera.look`) and sinks the rest out of sight, shrunk.
Its gates stand where its course's do, facing their way, the lit one in its lit look; its ground and sky are
centred on its course, the sky turned; its pillars stand where its course's do, each its aspect's shape at its
radius; its props stand outside the course, placed by its look. A shutter's open and close cameras blur the
image by the drone's motion (`motion_blur` samples, at least two; HyperDrone's paper, Sec 4.1, finds two
enough). At "medium" fidelity HyperLux draws base-color textures without lights; exposure, white balance,
gamma and noise are the race's (`dew.rl.flight.developed`)."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dew.rl.flight import ASPECTS, GATES, OBSTACLES, Scene

SUNK = (0.0, 0.0, -1000.0)
"""Where a placement not shown stands, shrunk to a thousandth, out of every camera's sight."""
PROPS = 6
"""Each prop's placements a drone's scene holds."""


def _yawed(yaw: jax.Array, position: jax.Array) -> jax.Array:
    """Placements' bases `[..., 12]`: turned by `yaw` about z, at `position` `[..., 3]` (HyperLux's (3, 4)
    pose, row by row)."""
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    zero, one = jnp.zeros_like(yaw), jnp.ones_like(yaw)
    rows = [jnp.stack([c, -s, zero, position[..., 0]], -1), jnp.stack([s, c, zero, position[..., 1]], -1),
            jnp.stack([zero, zero, one, position[..., 2]], -1)]
    return jnp.concatenate(rows, -1)


class HyperLuxRenderer:
    """`count` drones' front cameras `size` pixels square (`dew.rl.flight.Renderer`) over the assets in
    `assets` (`tools/fpv_assets.py`'s manifest), at HyperLux's `fidelity`, blurred by `motion_blur` samples
    (1 for none). Its images are `[count, size, size, 3]` uint8."""

    aspect = 1.0

    def __init__(self, count: int, assets: str | Path, *, size: int = 96, fidelity: str = "medium",
                 motion_blur: int = 2):
        import hyperlux
        from hyperlux.camera import RGB, CameraSpec

        folder = Path(assets)
        manifest = json.loads((folder / "manifest.json").read_text())
        if tuple(manifest["aspects"]) != ASPECTS:
            raise ValueError(f"the assets' pillars are of aspects {manifest['aspects']}, the race's "
                             f"{ASPECTS}")
        self.count, self.motion_blur = count, motion_blur
        pool = hyperlux.AssetPool()

        def loaded(name: str) -> int:
            return pool.add_object(hyperlux.load_object(folder / name, fidelity=fidelity))

        self._kinds = {kind: [loaded(name) for name in manifest[kind]]
                       for kind in ("gate", "lit", "ground", "sky", "prop")}
        self._pillars = [[loaded(name) for name in variants] for variants in manifest["pillar"]]
        scene = hyperlux.Scene()
        anchor = hyperlux.Object(name="anchor")
        corner = np.asarray([[0, 0, -5000], [0.1, 0, -5000], [0, 0.1, -5000]], np.float32)
        anchor.add_mesh(hyperlux.Mesh(corner, np.asarray([[0, 1, 2]], np.int32)))
        scene.add_object(anchor)
        scene.set_environment(ambient=(0.3, 0.3, 0.3), background=(0.55, 0.7, 0.9))
        self._rgb = RGB("rgba8")
        layout = self._layout()
        self.renderer = hyperlux.HyperLux(
            size, size, batch_size=count, fidelity=fidelity, num_probes=0,
            cameras=(CameraSpec("front", (self._rgb,)),), num_overlays=count,
            max_overlay_instances=len(layout), max_overlays_per_camera=1, motion_blur_samples=motion_blur)
        self.renderer.init(scene, asset_pool=pool)
        placements = []
        for drone in range(count):
            self.renderer.attach(drone, drone)
            placements += [self.renderer.spawn(drone, asset) for asset in layout]
        self._placements = self.renderer.placement_batch(placements)
        self._placed = jax.jit(jax.vmap(self._poses))

    def _layout(self) -> list[int]:
        """Each placement's asset in a drone's overlay, in `_poses`' order."""
        kinds = self._kinds
        pillars = [asset for variants in self._pillars for asset in variants for _ in range(OBSTACLES)]
        return ([asset for asset in kinds["gate"] for _ in range(GATES)] + kinds["lit"] + kinds["ground"]
                + kinds["sky"] + pillars + [asset for asset in kinds["prop"] for _ in range(PROPS)])

    def _poses(self, scene: Scene) -> tuple[jax.Array, jax.Array]:
        """One drone's placements' bases `[P, 12]` and scales `[P]` (`_layout`'s order)."""
        kinds, pillars = self._kinds, self._pillars
        looks = scene.look
        center, facing = scene.center.astype(jnp.float32), scene.facing.astype(jnp.float32)
        sunk = jnp.asarray(SUNK, jnp.float32)
        bases, scales = [], []

        def placed(show: jax.Array, yaw: jax.Array, position: jax.Array, scale: jax.Array | float = 1.0):
            show = jnp.asarray(show)
            bases.append(_yawed(jnp.where(show, yaw, 0.0), jnp.where(show[..., None], position, sunk)))
            sized = jnp.broadcast_to(jnp.asarray(scale, jnp.float32), show.shape)
            scales.append(jnp.where(show, sized, 1e-3))

        gates = jnp.arange(GATES)
        shown = (gates < scene.count) & (gates != scene.lit)
        for variant in range(len(kinds["gate"])):
            placed(shown & (looks[0] % len(kinds["gate"]) == variant), facing, center)
        lit = jnp.clip(scene.lit, 0, GATES - 1)
        for variant in range(len(kinds["lit"])):
            chosen = (scene.lit >= 0) & (looks[1] % len(kinds["lit"]) == variant)
            placed(chosen, facing[lit], center[lit])
        middle = jnp.sum(jnp.where((gates < scene.count)[:, None], center, 0.0), 0) / scene.count
        ground = middle * jnp.asarray([1.0, 1.0, 0.0])
        for variant in range(len(kinds["ground"])):
            placed(looks[2] % len(kinds["ground"]) == variant, jnp.zeros(()), ground)
        turn = (looks[3] // len(kinds["sky"]) % 360).astype(jnp.float32) * jnp.pi / 180
        for variant in range(len(kinds["sky"])):
            placed(looks[3] % len(kinds["sky"]) == variant, turn, ground)
        obstacles = scene.obstacles.astype(jnp.float32)
        radius = obstacles[:, 2]
        shape = jnp.argmin(jnp.abs(obstacles[:, 3:4] / jnp.maximum(radius[:, None], 1e-6)
                                   - jnp.asarray(ASPECTS)[None]), -1)
        material = looks[0] // len(kinds["gate"]) % len(pillars[0])
        foot = jnp.concatenate([obstacles[:, :2], jnp.zeros((OBSTACLES, 1))], -1)
        for aspect in range(len(pillars)):
            for variant in range(len(pillars[aspect])):
                show = (radius > 0) & (shape == aspect) & (material == variant)
                placed(show, jnp.zeros(OBSTACLES), foot, radius)
        spread = jnp.linalg.norm(center[:, :2] - middle[:2], axis=-1)
        reach = jnp.max(jnp.where(gates < scene.count, spread, 0.0))
        key = jax.random.fold_in(jax.random.key(532), looks[1])
        for _ in kinds["prop"]:
            key, angle_key, out_key, yaw_key, size_key = jax.random.split(key, 5)
            angle = jax.random.uniform(angle_key, (PROPS,), maxval=2 * jnp.pi)
            out = reach + jax.random.uniform(out_key, (PROPS,), minval=8.0, maxval=60.0)
            where = ground + jnp.stack([out * jnp.cos(angle), out * jnp.sin(angle), jnp.zeros(PROPS)], -1)
            placed(jnp.ones(PROPS, bool), jax.random.uniform(yaw_key, (PROPS,), maxval=2 * jnp.pi), where,
                   jax.random.uniform(size_key, (PROPS,), minval=0.7, maxval=1.6))
        return jnp.concatenate([jnp.atleast_2d(base) for base in bases]), jnp.concatenate(
            [jnp.atleast_1d(scale) for scale in scales])

    def render(self, scene: Scene, cameras: jax.Array, opened: jax.Array) -> jax.Array:
        import hyperlux

        bases, scales = self._placed(scene)
        bases = jax.block_until_ready(bases.reshape(-1, 12))
        scales = jax.block_until_ready(scales.reshape(-1))
        self.renderer.set_pose(self._placements, hyperlux.Pose(bases, scale=scales))
        self.renderer.update()
        closed = jax.block_until_ready(jnp.asarray(cameras, jnp.float32))
        if self.motion_blur > 1:
            begun = jax.block_until_ready(jnp.asarray(opened, jnp.float32))
            self.renderer.set_motion_blur_cameras(begun, closed)
        else:
            self.renderer.set_cameras(closed)
        self.renderer.render()
        blocks = [jax.dlpack.from_dlpack(block) for block in self.renderer.outputs[self._rgb].blocks]
        return jnp.concatenate(blocks)[..., :3]


__all__ = ["PROPS", "SUNK", "HyperLuxRenderer"]
