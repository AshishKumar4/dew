"""The gate race seen through HyperLux (`dew.rl.hyperlux`), as a user reaches it: a `GateRace` over a
`HyperLuxRenderer` of the assets `tools/fpv_assets.py` builds, on an NVIDIA GPU with OptiX (the `hyperlux`
extra). Each drone's camera draws its own episode's course and moves with the drone, and a fisheye lens
samples the pinhole it draws, its outer corners, which no pinhole of its field of view holds, black."""

import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from dew.rl.flight import GateRace, Racing
from dew.rl.hyperlux import HyperLuxRenderer, fisheye_map

pytestmark = [pytest.mark.gpu(reason="HyperLux draws on an NVIDIA GPU with OptiX"),
              pytest.mark.network]

DRONES, SIZE = 2, 64
"""The race's drones and each camera's pixels across: two episodes, each its own course and looks."""


@pytest.fixture(scope="module")
def assets(tmp_path_factory) -> Path:
    """The renderer's assets as tools/fpv_assets.py builds them (Poly Haven's CC0 textures, fetched)."""
    out = tmp_path_factory.mktemp("fpv-assets")
    tool = Path(__file__).resolve().parents[1] / "tools" / "fpv_assets.py"
    subprocess.run([sys.executable, str(tool), str(out)], check=True)
    return out


def test_each_drones_camera_draws_its_own_course_and_moves_with_it(assets):
    """A gate race of two drones over HyperLux's renderer shows each drone an image `[2, 64, 64, 3]` of
    uint8: a drawn scene, not one colour; each drone's its own episode's (its course and looks drawn
    apart from the other's), and a drone's next image another once it has flown on."""
    race = GateRace(HyperLuxRenderer(DRONES, assets, size=SIZE), views=("image",))
    first = np.asarray(race.reset(0)["image"])
    assert first.shape == (DRONES, SIZE, SIZE, 3) and first.dtype == np.uint8
    assert all(np.unique(image.reshape(-1, 3), axis=0).shape[0] > 1 for image in first)
    assert not np.array_equal(first[0], first[1])
    # Full throttle, level: each drone climbs, its camera with it.
    later = first
    for _ in range(10):
        later = np.asarray(race.step(jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * DRONES)).observation["image"])
    assert all(not np.array_equal(before, after) for before, after in zip(first, later, strict=True))


def test_a_fisheye_lens_samples_the_pinhole_it_draws(assets):
    """A fisheye lens of the race's field of view (`fisheye_map`) draws each drone's image from a pinhole
    `source` pixels square: the pixels no pinhole of that field of view holds, its outer corners, are
    black, and every other pixel is a bilinear sample of the pinhole HyperLux drew, which a renderer of no
    lens at the source's size draws for the same race."""
    fov, source = Racing().fov, 96
    lensed = GateRace(HyperLuxRenderer(DRONES, assets, size=SIZE, fisheye=fov, source=source),
                      views=("image",))
    pinhole = GateRace(HyperLuxRenderer(DRONES, assets, size=source), views=("image",))
    fisheye, drawn = (np.asarray(race.reset(0)["image"]) for race in (lensed, pinhole))
    corners, weights, held = fisheye_map(SIZE, source, fov)
    assert held.any() and not held.all()
    np.testing.assert_array_equal(fisheye[:, ~held], 0)
    flat = drawn.reshape(DRONES, -1, 3).astype(np.float32)
    sampled = np.einsum("bnkc,nk->bnc", flat[:, corners.reshape(-1, 4)],
                        weights.reshape(-1, 4).astype(np.float32)).reshape(DRONES, SIZE, SIZE, 3)
    expected = np.round(np.clip(sampled, 0, 255)).astype(np.uint8)
    np.testing.assert_array_equal(fisheye[:, held], expected[:, held])
