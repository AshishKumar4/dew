"""CLIP metrics on generated images.

`clip_score(images, prompts)` scores a set of images against the prompts they
were sampled from, and the registered metrics take the same cosine over a
validation pass.

Both metrics run the vendored towers in `dew.nn.text_encoders`, since
transformers 5 ships no `FlaxCLIPModel`, and preprocess with the checkpoint's
own PIL image processor, which transformers 5 ships. A score here is
what the reference computes for the same pixels and tokens;
`tests/test_metrics.py` states the tolerance and the difference observed.
"""

import functools
import logging
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

from dew.artifacts import ImageGrid
from dew.objectives.base import Batch
from dew.registry import metrics

from .common import ImageMetric, metric_device

_log = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/clip-vit-large-patch14"


@functools.cache
def _get_clip(modelname: str):
    """The vendored CLIP towers and the checkpoint's image processor, loaded
    once per model name. CLIP-L/14 is about 600 MB in HBM, and every metric
    built from this module shares the copy."""
    from transformers import CLIPImageProcessorPil

    from dew.nn.text_encoders import CLIPModel
    _log.info("loading CLIP model %r (cached for reuse)", modelname)
    return CLIPModel.from_pretrained(modelname), CLIPImageProcessorPil.from_pretrained(modelname)


@functools.cache
def _get_tokenizer(modelname: str):
    """The checkpoint's prompt tokenizer, loaded once per model name. It pads
    and truncates to the model's context exactly as the loader does, so a
    prompt scored here is the prompt a run's batch would carry."""
    from dew.data.processors import AutoTextTokenizer
    _log.info("loading CLIP tokenizer %r (cached for reuse)", modelname)
    return AutoTextTokenizer(tensor_type="np", modelname=modelname)


def _equal_counts(images: int, prompts: int) -> None:
    """CLIP pairs one prompt with one image, so the two counts have to agree."""
    if images != prompts:
        raise ValueError(f"CLIP scored {images} images against {prompts} prompts; "
                         "equal counts are required")


def _uint8_pixels(images: ArrayLike) -> NDArray[np.uint8]:
    """A sampler's [-1, 1] pixels as uint8, nearest value and clipped, because
    a sample can leave the range."""
    return np.clip(np.round((np.asarray(images) + 1.0) * 127.5), 0, 255).astype(np.uint8)


def clip_image_text_cosine(images: ArrayLike, input_ids: ArrayLike, attention_mask: ArrayLike, *,
                           modelname: str = DEFAULT_MODEL) -> jax.Array:
    """Per-image cosine between uint8 [N, H, W, 3] images and tokenized prompts.

    The images go through the checkpoint's own processor, so the embeddings are
    the ones the reference produces for these pixels and tokens.
    """
    model, processor = _get_clip(modelname)
    pixels = np.asarray(images)
    tokenized = np.shape(input_ids)[0]
    _equal_counts(pixels.shape[0], tokenized)
    if np.shape(attention_mask)[0] != tokenized:
        raise ValueError(f"CLIP got {tokenized} prompts and "
                         f"{np.shape(attention_mask)[0]} attention masks")
    pixel_values = processor(images=pixels, return_tensors="np")["pixel_values"]
    image_embeds = model.get_image_features(pixel_values)
    text_embeds = model.get_text_features(input_ids, attention_mask)
    image_embeds = image_embeds / jnp.linalg.norm(image_embeds, axis=-1, keepdims=True)
    text_embeds = text_embeds / jnp.linalg.norm(text_embeds, axis=-1, keepdims=True)
    return jnp.einsum('nd,nd->n', image_embeds, text_embeds)


def clip_score(images: ArrayLike, prompts: Sequence[str], *, modelname: str = DEFAULT_MODEL,
               batch_size: int = 64) -> float:
    """CLIPScore of uint8 [N, H, W, 3] images against one prompt each.

    100 * mean(max(cos(image, prompt), 0)), higher is better; typical T2I
    models score around 25-35 on natural prompts. The images are scored
    `batch_size` rows at a time, and the prompts are tokenized the way a run's
    batch carries them.
    """
    if batch_size < 1:
        raise ValueError(f"clip_score: a batch holds at least one image, got batch_size={batch_size}")
    pixels = np.asarray(images)
    if pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError(f"clip_score: expected uint8 [N, H, W, 3] images, got "
                         f"{pixels.dtype} {list(pixels.shape)}")
    _equal_counts(pixels.shape[0], len(prompts))
    if pixels.shape[0] == 0:
        raise ValueError("clip_score: no images to score")
    tokens = _get_tokenizer(modelname)(list(prompts))
    total = 0.0
    with metric_device():
        for start in range(0, pixels.shape[0], batch_size):
            stop = start + batch_size
            cosine = clip_image_text_cosine(pixels[start:stop], tokens["input_ids"][start:stop],
                                            tokens["attention_mask"][start:stop],
                                            modelname=modelname)
            # Summed the way a pass sums it, so a whole set in one batch is
            # the number the metric reports for the same images.
            total += float(np.asarray(100.0 * jnp.maximum(cosine, 0.0), dtype=np.float64).sum())
    return total / pixels.shape[0]


def _artifact_cosine(artifact: ImageGrid, batch: Batch, field: str, modelname: str) -> jax.Array:
    """The per-image cosine for one sampled grid and the prompts of its batch."""
    text = batch[field]
    return clip_image_text_cosine(_uint8_pixels(artifact.images), text["input_ids"],
                                  text["attention_mask"], modelname=modelname)


@metrics("clip")
def clip(modelname: str = DEFAULT_MODEL, field: str = "text") -> ImageMetric:
    """CLIP distance, mean(1 - cos(image, text)), lower is better. It logs as
    val/clip_similarity; `clip_score` is the standard number for a new run.
    """

    def measure(artifact, batch):
        return 1.0 - _artifact_cosine(artifact, batch, field, modelname)

    return ImageMetric(name="clip_similarity", measure=measure)


@metrics("clip_score")
def clip_score_metric(modelname: str = DEFAULT_MODEL, field: str = "text") -> ImageMetric:
    """Standard CLIPScore over a validation pass, the same number `clip_score`
    reports for the images and prompts the pass consumed.
    """

    def measure(artifact, batch):
        return 100.0 * jnp.maximum(_artifact_cosine(artifact, batch, field, modelname), 0.0)

    return ImageMetric(name="clip_score", measure=measure)
