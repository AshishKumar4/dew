"""LPIPS, the learned perceptual image distance, on VGG16.

Zhang et al. 2018, "The Unreasonable Effectiveness of Deep Features as a
Perceptual Metric": both images, in [-1, 1], are shifted and scaled per
channel, run through VGG16's convolutions, and at each of its five stages'
last ReLU (relu1_2, relu2_2, relu3_3, relu4_3, relu5_3) the features are
unit-normalized along the channels, their squared difference weighted per
channel by a learned 1x1 convolution and averaged over the positions; the
distance is the sum over the stages. It is richzhang/PerceptualSimilarity's
`lpips.LPIPS(net='vgg')` at v0.1, the network taming-transformers and
REPA-E copy for their perceptual losses.

The published weights are torchvision's ImageNet VGG16 (`IMAGENET1K_V1`,
vgg16-397923af.pth) and v0.1's linear heads (lpips/weights/v0.1/vgg.pth,
whose MD5 is the d507d734... taming-transformers and REPA-E check). Both
are PyTorch pickles: `LPIPSNetwork.published()` downloads each once,
checks its SHA-256, and converts it through `dew.interop.pickles`, which
needs torch for that first conversion only.

`LPIPS` is the same distance as an image metric: the mean over the sampled
frames of their distance to the batch's, lower being better.
"""

from __future__ import annotations

import functools
import hashlib
import os
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.artifacts import ImageGrid
from dew.objectives.base import Variables
from dew.registry import metrics

from .common import ImageMetric, paired
from .psnr import frame_batch

SHIFT = (-0.030, -0.088, -0.188)
SCALE = (0.458, 0.448, 0.450)
"""LPIPS's `ScalingLayer`: the per-channel shift and scale of [-1, 1] pixels."""

STAGES = ((64, 64), (128, 128), (256, 256, 256), (512, 512, 512), (512, 512, 512))
"""VGG16's convolution widths, stage by stage; a 2x2 max pool opens every
stage after the first."""

TORCHVISION_LAYERS = (0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28)
"""Where each convolution sits in torchvision's `vgg16().features`."""

VGG16_WEIGHTS = ("https://download.pytorch.org/models/vgg16-397923af.pth",
                 "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0")
LINEAR_WEIGHTS = ("https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/"
                  "082bb24f84c091ea94de2867d34c4544f68e0963/lpips/weights/v0.1/vgg.pth",
                  "a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868")
"""The published files and their SHA-256."""


def _unit(features: jax.Array) -> jax.Array:
    """`normalize_tensor`: each position's features over their norm plus 1e-10."""
    return features / (jnp.sqrt(jnp.sum(jnp.square(features), axis=-1, keepdims=True)) + 1e-10)


class VGG16(nn.Module):
    """VGG16's convolutions, returning each stage's last ReLU, channels last."""

    @nn.compact
    def __call__(self, pixels: jax.Array) -> list[jax.Array]:
        outputs, hidden, layer = [], pixels, 0
        for stage, widths in enumerate(STAGES):
            if stage:
                hidden = nn.max_pool(hidden, (2, 2), strides=(2, 2))
            for width in widths:
                hidden = nn.relu(nn.Conv(width, (3, 3), padding=1, name=f"conv_{layer}")(hidden))
                layer += 1
            outputs.append(hidden)
        return outputs


class LPIPSNetwork(nn.Module):
    """LPIPS between two `[B, H, W, 3]` image batches in [-1, 1], one
    distance per image."""

    @nn.compact
    def __call__(self, images: jax.Array, references: jax.Array) -> jax.Array:
        shift, scale = jnp.asarray(SHIFT, images.dtype), jnp.asarray(SCALE, images.dtype)
        vgg = VGG16(name="vgg")
        distance = jnp.zeros(images.shape[0], images.dtype)
        features = zip(vgg((images - shift) / scale), vgg((references - shift) / scale), strict=True)
        for stage, (left, right) in enumerate(features):
            weighted = nn.Conv(1, (1, 1), use_bias=False, name=f"lin_{stage}")(
                jnp.square(_unit(left) - _unit(right)))
            distance = distance + jnp.mean(weighted, axis=(1, 2, 3))
        return distance

    @staticmethod
    def published() -> tuple[LPIPSNetwork, Variables]:
        """The network and the published weights, downloaded and converted
        on first use."""
        return LPIPSNetwork(), _published_variables()


def _fetched(url: str, digest: str) -> Path:
    """`url` in Dew's cache, downloaded once and checked against `digest`."""
    from dew.telemetry.instrumentation import dew_cache_dir

    path = Path(dew_cache_dir()) / "lpips" / url.rsplit("/", 1)[-1]
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        with urllib.request.urlopen(url) as response, open(partial, "wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
        os.replace(partial, path)
    with open(path, "rb") as handle:
        found = hashlib.file_digest(handle, "sha256").hexdigest()
    if found != digest:
        raise ValueError(f"{url} hashes to {found}, not the {digest} LPIPS was written against")
    return path


def variables_from_torch(vgg: dict[str, np.ndarray], linear: dict[str, np.ndarray]) -> Variables:
    """The network's variables from torchvision's `vgg16()` state dict and
    v0.1's linear heads, kernels moved from OIHW to HWIO."""
    params: dict[str, object] = {"vgg": {f"conv_{layer}": {
        "kernel": jnp.asarray(np.transpose(vgg[f"features.{index}.weight"], (2, 3, 1, 0))),
        "bias": jnp.asarray(vgg[f"features.{index}.bias"])}
        for layer, index in enumerate(TORCHVISION_LAYERS)}}
    for stage in range(len(STAGES)):
        params[f"lin_{stage}"] = {"kernel": jnp.asarray(
            np.transpose(linear[f"lin{stage}.model.1.weight"], (2, 3, 1, 0)))}
    return {"params": params}


@functools.cache
def _published_variables() -> Variables:
    from dew.interop.pickles import converted
    from dew.interop.safetensors_io import WEIGHTS_FILE, read_file

    tables = []
    for url, digest in (VGG16_WEIGHTS, LINEAR_WEIGHTS):
        path = _fetched(url, digest)
        tables.append(dict(read_file(converted(path.parent, [path.name]) / WEIGHTS_FILE)[0]))
    return variables_from_torch(*tables)


@functools.cache
def _distance():
    network, variables = LPIPSNetwork.published()
    return jax.jit(lambda images, references: network.apply(variables, images, references))


@metrics("lpips")
class LPIPS(ImageMetric):
    """Mean LPIPS between the sampled frames and the batch's, lower is better.

    Both sides are in [-1, 1], the range the published network reads. The
    weights load on the first measurement, not at construction.
    """

    def __init__(self, field: str = "image", reads: type = ImageGrid):
        def measure(artifact, batch):
            samples, targets = paired(artifact, batch, field)
            return _distance()(frame_batch(samples), frame_batch(targets))

        super().__init__(name="lpips", measure=measure, better="lower", reads=reads)

