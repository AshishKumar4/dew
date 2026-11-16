"""The gate race seen through HyperLux (`dew.rl.hyperlux`), as a user reaches it: a `GateRace` over a
`HyperLuxRenderer` of the assets `tools/fpv_assets.py` builds, on an NVIDIA GPU with OptiX (the `hyperlux`
extra). Each drone's camera draws its own episode's course and moves with the drone, and a fisheye lens
samples the pinhole it draws, its outer corners, which no pinhole of its field of view holds, black where
the camera develops the image as drawn.

A GPU node's driver serves no OptiX of its own: the fleet's `optix` reference, which a file naming
DEW_OPTIX_LIB takes, holds the driver's OptiX libraries at its version, loaded by path before HyperLux
asks for them by name (`optix`)."""

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import as_given, assert_as_exact

from dew.rl.flight import GateRace, Racing
from dew.rl.hyperlux import HyperLuxRenderer, fisheye_map, fisheyed

pytestmark = [pytest.mark.gpu(reason="HyperLux draws on an NVIDIA GPU with OptiX"),
              pytest.mark.network]

DRONES, SIZE = 2, 64
"""The race's drones and each camera's pixels across: two episodes, each its own course and looks."""


OPTIX = ("libnvidia-gpucomp.so.1", "libnvidia-rtcore.so.1", "libnvoptix.so.1")
"""The driver's OptiX libraries, in the order each needs the one before it."""


@pytest.fixture(scope="module", autouse=True)
def optix():
    """The fleet's OptiX libraries (DEW_OPTIX_LIB) loaded by path and made global, so the OptiX stubs' later
    load of libnvoptix.so.1 by name finds them, as a process's library path, read once as it starts, would
    not."""
    held = os.environ.get("DEW_OPTIX_LIB")
    if not held:
        raise FileNotFoundError("HyperLux draws through OptiX, which a GPU node's driver serves none of: set "
                                "DEW_OPTIX_LIB to the fleet's optix reference, the driver's OptiX libraries")
    for name in OPTIX:
        ctypes.CDLL(os.path.join(held, name), mode=ctypes.RTLD_GLOBAL)


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
    apart from the other's), and a drone's next image another once it has flown on, its camera developing
    no read noise, which would tell apart images the renderer drew alike."""
    # No read noise, which would tell images apart that the renderer drew alike.
    race = GateRace(HyperLuxRenderer(DRONES, assets, size=SIZE), Racing(noise=(0.0, 0.0)), views=("image",))
    first = np.asarray(race.reset(0)["image"])
    assert first.shape == (DRONES, SIZE, SIZE, 3) and first.dtype == np.uint8
    assert all(np.unique(image.reshape(-1, 3), axis=0).shape[0] > 1 for image in first)
    assert not np.array_equal(first[0], first[1])
    # Full throttle, level: each drone climbs, its camera with it.
    later = first
    for _ in range(10):
        later = np.asarray(race.step(jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * DRONES)).observation["image"])
    assert all(not np.array_equal(before, after) for before, after in zip(first, later, strict=True))


def lens_of(fov: float, source: int):
    """The lens a renderer of `fov` drawing `source`-square pinholes runs past its draw (`fisheye_map`,
    `fisheyed`, then each sample's nearest level), over pinhole images `[B, source, source, 3]`, as
    `HyperLuxRenderer.render` runs it: the resampling one program, its rounding apart."""
    corners, weights, _ = fisheye_map(SIZE, source, fov)
    lens = (jnp.asarray(corners, jnp.int32), jnp.asarray(weights, jnp.float32))
    sampled = jax.jit(fisheyed)

    def levels(images):
        return jnp.round(jnp.clip(sampled(jnp.asarray(images), *lens), 0, 255)).astype(jnp.uint8)
    return levels


def test_a_fisheye_lens_samples_the_pinhole_it_draws(assets):
    """A fisheye lens of the race's field of view (`fisheye_map`) draws each drone's image from a pinhole
    `source` pixels square: the pixels no pinhole of that field of view holds, its outer corners, are
    black, and the image is its lens's resampling (`lens_of`) of the pinhole HyperLux drew, which a
    renderer of no lens at the source's size draws for the same race. Both races start at the curriculum's
    first rung (`ramp` 0), whose cameras develop an image as it is drawn (`developed`: no exposure, white
    balance, gamma or read noise of their own), so each image is its lens's."""
    fov, source = Racing().fov, 96
    lensed = GateRace(HyperLuxRenderer(DRONES, assets, size=SIZE, fisheye=fov, source=source),
                      views=("image",), ramp=0.0)
    pinhole = GateRace(HyperLuxRenderer(DRONES, assets, size=source), views=("image",), ramp=0.0)
    fisheye, drawn = (np.asarray(race.reset(0)["image"]) for race in (lensed, pinhole))
    _, _, held = fisheye_map(SIZE, source, fov)
    assert held.any() and not held.all()
    np.testing.assert_array_equal(fisheye[:, ~held], 0)
    np.testing.assert_array_equal(fisheye, np.asarray(lens_of(fov, source)(drawn)))


LENS_SEEDS = 8
"""Check 1's declared seeds for the lens's samples against NumPy's: the two arms take one input, so their
errors move together and their ratio resolves at the fewest (`seeds_for`)."""


def test_a_fisheye_lens_samples_each_pixel_as_float32_does():
    """The lens's resampling (`fisheyed`, the program the renderer runs past its pinhole draw) of a pinhole
    image on par (check 1) with NumPy's float32 bilinear sample of it by the lens's float32 weights, from the
    sample in float64, before either rounds to levels, at JAX's default precision of products, as a user's
    process computes it."""
    fov, source = Racing().fov, 96
    corners, weights, _ = fisheye_map(SIZE, source, fov)
    taps, weighed = corners.reshape(-1, 4), weights.reshape(-1, 4).astype(np.float32)
    lens = (jnp.asarray(corners, jnp.int32), jnp.asarray(weights, jnp.float32))
    sampled = jax.jit(fisheyed)

    def sample(images, dtype):
        flat = np.asarray(images).astype(dtype).reshape(DRONES, -1, 3)
        return np.einsum("bnkc,nk->bnc", flat[:, taps], weighed.astype(dtype)).reshape(DRONES, SIZE, SIZE, 3)

    def draw(seed):
        return (np.random.default_rng(seed).integers(0, 256, (DRONES, source, source, 3)).astype(np.uint8),)

    def lensed(images):
        # As a user's process computes it, at JAX's default precision of products, whatever the lane states.
        with jax.default_matmul_precision("default"):
            return np.asarray(sampled(jnp.asarray(images), *lens), np.float64)

    # The lens runs its own program, as the renderer runs it; NumPy's arms are NumPy's.
    assert_as_exact(as_given(lensed),
                    as_given(lambda images: sample(images, np.float32).astype(np.float64)),
                    draw, "a fisheye lens's samples", seeds=LENS_SEEDS,
                    truth_fn=as_given(lambda images: sample(images, np.float64)))
