"""Per-record image draws and their host/device implementations.

The crop is an integer (top, left, height, width) rectangle, resized with
half-pixel bilinear interpolation and clamped edges. Both implementations
apply flip and then torchvision's float ColorJitter (brightness, contrast
and saturation, `torchvision.transforms.v2.functional` at 0.29) in the
supplied order, each clamped to the pixel range as torchvision's `_blend`
clamps it, and their caller rounds once. This is not OpenCV's area/cubic resize
used to make variable-size decoded records stackable before augmentation.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates
from jax.typing import ArrayLike


class ImageParameters(NamedTuple):
    """One example's crop, flip, brightness/contrast/saturation and order."""

    crop: ArrayLike
    flip: ArrayLike
    factors: ArrayLike
    order: ArrayLike


_GREY = (0.2989, 0.587, 0.114)
"""torchvision's `rgb_to_grayscale` weights, which its contrast and
saturation blend against."""
_LOW = (0.8, 0.95, 0.8)
_HIGH = (1.2, 1.05, 1.2)


def draw_device(key: jax.Array, shape: tuple[int, ...], *, flip: bool, jitter: bool,
                crop_scale: tuple[float, float]) -> ImageParameters:
    """Draw independently from one record's data key, never its batch index."""
    area, origin, mirror, colour, order = jax.random.split(key, 5)
    # Data draws stay fp32/int32 even when a model enables float64 globally.
    fraction = jnp.sqrt(jax.random.uniform(area, (), dtype=jnp.float32,
                                         minval=crop_scale[0], maxval=crop_scale[1]))
    sides = jnp.asarray(shape[:2], jnp.int32)
    extent = jnp.maximum(1, jnp.rint(sides * fraction).astype(jnp.int32))
    start = jax.random.randint(origin, (2,), 0, sides - extent + 1, dtype=jnp.int32)
    factors = (
        jax.random.uniform(
            colour,
            (3,),
            dtype=jnp.float32,
            minval=jnp.asarray(_LOW, jnp.float32),
            maxval=jnp.asarray(_HIGH, jnp.float32),
        )
        if jitter
        else jnp.ones(3, jnp.float32)
    )
    return ImageParameters(jnp.concatenate([start, extent]),
                           jax.random.bernoulli(mirror, p=jnp.float32(0.5)) if flip else jnp.asarray(a=False),
                           factors, jax.random.permutation(order, jnp.arange(3, dtype=jnp.int32))
                           if jitter else jnp.arange(3, dtype=jnp.int32))


def draw_host(rng: np.random.Generator, shape: tuple[int, ...], *, flip: bool, jitter: bool,
              crop_scale: tuple[float, float]) -> ImageParameters:
    """Draw the same parameter ranges with Grain's host record generator."""
    fraction = np.sqrt(rng.uniform(*crop_scale))
    extent = np.maximum(1, np.rint(np.asarray(shape[:2]) * fraction).astype(np.int32))
    start = rng.integers(0, np.asarray(shape[:2]) - extent + 1)
    return ImageParameters(np.concatenate([start, extent]),
                           np.asarray(flip and rng.random() < 0.5),
                           rng.uniform(_LOW, _HIGH) if jitter else np.ones(3),
                           rng.permutation(3) if jitter else np.arange(3))


def apply_device(image: ArrayLike, parameters: ImageParameters, size: int) -> jax.Array:
    """Crop/resize/flip/jitter one image in floating point, before quantization."""
    pixels = jnp.asarray(image, dtype=jnp.result_type(image, jnp.float32))
    top, left, height, width = jnp.asarray(parameters.crop)
    row = jnp.clip((jnp.arange(size, dtype=pixels.dtype) + 0.5) * height / size - 0.5, 0, height - 1)
    col = jnp.clip((jnp.arange(size, dtype=pixels.dtype) + 0.5) * width / size - 0.5, 0, width - 1)
    # A crop's shape is a dynamic per-record draw, while image.resize needs
    # static shapes. Sampling its half-pixel coordinates keeps a fixed output
    # shape and clamps to the crop's edges, not the surrounding image's.
    pixels = jax.vmap(lambda channel: map_coordinates(
        channel, [(top + row)[:, None], (left + col)[None, :]], order=1, mode="nearest"),
        in_axes=2, out_axes=2)(pixels)
    pixels = jnp.where(jnp.asarray(parameters.flip), pixels[:, ::-1], pixels)
    brightness, contrast, saturation = jnp.asarray(parameters.factors, dtype=pixels.dtype)
    grey = jnp.asarray(_GREY, dtype=pixels.dtype)

    def colour(index, current):
        return jnp.clip(jax.lax.switch(jnp.asarray(parameters.order)[index], (
            lambda x: x * brightness,
            lambda x: x * contrast + jnp.mean(jnp.sum(x * grey, axis=-1)) * (1 - contrast),
            lambda x: x * saturation + jnp.sum(x * grey, axis=-1, keepdims=True) * (1 - saturation),
        ), current), 0, 255)

    return jax.lax.fori_loop(0, 3, colour, pixels)


def jitter_host(pixels: np.ndarray, factors: ArrayLike, order: ArrayLike) -> np.ndarray:
    """torchvision's float ColorJitter on `[..., 3]` pixels in [0, 255]: the
    brightness, contrast and saturation `factors` applied in `order`, each a
    blend with black, the mean grey or each pixel's grey, clamped to the
    pixel range."""
    import cv2

    brightness, contrast, saturation = np.asarray(factors, dtype=pixels.dtype)
    grey = np.asarray(_GREY, dtype=pixels.dtype)
    pixels = np.array(pixels, copy=True)
    for index in np.asarray(order):
        if index == 0:
            np.multiply(pixels, brightness, out=pixels)
        elif index == 1:
            # The mean grey of the image as it stands at its place in the order.
            mean = cv2.transform(pixels, grey[None]).mean()
            np.multiply(pixels, contrast, out=pixels)
            np.add(pixels, mean * (1 - contrast), out=pixels)
        else:
            # Saturation as one matrix: out_c = s * x_c + (1 - s) * grey.
            matrix = np.eye(3, dtype=pixels.dtype) * saturation + np.outer(
                np.ones(3, pixels.dtype), (1 - saturation) * grey)
            pixels = cv2.transform(pixels, matrix)
        np.clip(pixels, 0, 255, out=pixels)
    return pixels


def apply_host(image: np.ndarray, parameters: ImageParameters, size: int) -> np.ndarray:
    """OpenCV bilinear resize and NumPy colour operations for the same draws."""
    import cv2

    dtype = np.result_type(image.dtype, np.float32)
    top, left, height, width = np.asarray(parameters.crop, dtype=np.int32)
    pixels = cv2.resize(image[top:top+height, left:left+width].astype(dtype),
                        (size, size), interpolation=cv2.INTER_LINEAR)
    if parameters.flip:
        pixels = pixels[:, ::-1]
    return jitter_host(pixels, parameters.factors, parameters.order)


def augment_batch(images: ArrayLike, raw_keys: ArrayLike, *, size: int, flip: bool, jitter: bool,
                  crop_scale: tuple[float, float]) -> jax.Array:
    """Augment a dense uint8 batch on its device, one Threefry key per record."""
    images = jnp.asarray(images)
    keys = jax.random.wrap_key_data(jnp.asarray(raw_keys, jnp.uint32), impl="threefry2x32")

    def one(image, key):
        parameters = draw_device(key, image.shape, flip=flip, jitter=jitter, crop_scale=crop_scale)
        return jnp.clip(jnp.rint(apply_device(image, parameters, size)), 0, 255).astype(jnp.uint8)

    return jax.vmap(one)(images, keys)
