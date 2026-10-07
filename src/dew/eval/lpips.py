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
whose MD5 is the d507d734... taming-transformers and REPA-E check).
`LPIPSNetwork.published()` reads them from safetensors copies on the Hub,
whose tensors are bitwise those files', each pinned by revision and
SHA-256 through `dew.interop.inception_fid.fetch`.

`LPIPS` is the same distance as an image metric: the mean over the sampled
frames of their distance to the batch's, lower being better.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.artifacts import ImageGrid
from dew.objectives.base import Variables

from .common import ImageMetric, paired
from .psnr import frame_batch

SHIFT = tuple(np.asarray((-0.030, -0.088, -0.188), np.float32).tolist())
SCALE = tuple(np.asarray((0.458, 0.448, 0.450), np.float32).tolist())
"""LPIPS's `ScalingLayer`: the per-channel shift and scale of [-1, 1] pixels,
as the float32 numbers its `torch.Tensor` buffers hold, at any precision."""

STAGES = ((64, 64), (128, 128), (256, 256, 256), (512, 512, 512), (512, 512, 512))
"""VGG16's convolution widths, stage by stage; a 2x2 max pool opens every
stage after the first."""

TORCHVISION_LAYERS = (0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28)
"""Where each convolution sits in torchvision's `vgg16().features`."""

VGG16_WEIGHTS = ("timm/vgg16.tv_in1k", "model.safetensors", "b8d8aa2dd860af9233c8c67385a8097fd6c35d3f",
                 "57b026918159a6bf9faf8405c3a551903768e7138989d9c6224a14227203fad8")
LINEAR_WEIGHTS = ("vivym/lpips", "vgg_lpips_linear.safetensors", "270571f1fb2a2c4c5f920cec96cd87838c345982",
                  "0c6387bf2e51e434dedbc0ca4c5894f605c415b8d128bfe720e13456eb4333ac")
"""The Hub copies' repo, file, revision and SHA-256. timm's `features.*` are
bitwise vgg16-397923af.pth's, and vivym's `{stage}.weight` v0.1's
`lin{stage}.model.1.weight`."""


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
    """Computes LPIPS between two `[B, H, W, 3]` image batches in [-1, 1], one distance per image."""

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
        """Return the network and the published weights, downloading them on first use."""
        return LPIPSNetwork(), _published_variables()


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
    from dew.interop.inception_fid import fetch
    from dew.interop.safetensors_io import read_file

    vgg, linear = (read_file(fetch(*source))[0] for source in (VGG16_WEIGHTS, LINEAR_WEIGHTS))
    return variables_from_torch(vgg, {f"lin{stage}.model.1.weight": linear[f"{stage}.weight"]
                                      for stage in range(len(STAGES))})


@functools.cache
def _distance():
    network, variables = LPIPSNetwork.published()
    return jax.jit(lambda images, references: network.apply(variables, images, references))


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

