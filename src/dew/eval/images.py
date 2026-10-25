"""Measure CLIP metrics on generated images.

`CLIPScore().score(images, prompts)` scores a set of images against the
prompts they were sampled from, `CLIPScore().scores(...)` each image on its
own, and the validation metrics take the same cosine over a validation pass.
`clip_image_features` is the image tower's embedding, for heads that read it
(an aesthetic predictor, say).

Both metrics run the vendored towers in `dew.nn.text_encoders`, since
transformers 5 ships no `FlaxCLIPModel`, and preprocess with the checkpoint's
own PIL image processor, which transformers 5 ships. A score here is
what the reference computes for the same pixels and tokens;
`tests/test_clip_metrics.py` states the tolerance and the difference observed.
"""

import functools
import logging
from collections.abc import Iterator, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from dew.artifacts import ImageGrid, uint8_pixels
from dew.nn.precision import accumulated
from dew.nn.text_encoders import DEFAULT_MODEL as CLIP_CHECKPOINT
from dew.objectives.base import Batch

from .common import ImageMetric, metric_device

_log = logging.getLogger(__name__)

@functools.cache
def _get_clip(modelname: str):
    """Load the vendored CLIP towers and the checkpoint's image processor,
    once per model name. CLIP-L/14 is about 600 MB in HBM, and every metric
    built from this module shares the copy."""
    from transformers import CLIPImageProcessorPil

    from dew.nn.text_encoders import CLIPModel
    _log.info("loading CLIP model %r (cached for reuse)", modelname)
    return CLIPModel.from_pretrained(modelname), CLIPImageProcessorPil.from_pretrained(modelname)


def _equal_counts(images: int, prompts: int) -> None:
    """Refuse a prompt count and an image count that disagree.

    CLIP pairs one prompt with one image.
    """
    if images != prompts:
        raise ValueError(f"CLIP scored {images} images against {prompts} prompts; "
                         "equal counts are required")


def clip_image_features(images: ArrayLike, *, modelname: str = CLIP_CHECKPOINT) -> jax.Array:
    """Return CLIP's image embedding [N, D] of uint8 [N, H, W, 3] images, unnormalized.

    The images go through the checkpoint's own processor, so the embeddings are
    the ones the reference produces for these pixels.
    """
    model, processor = _get_clip(modelname)
    pixel_values = processor(images=np.asarray(images), return_tensors="np")["pixel_values"]
    return model.get_image_features(pixel_values)


def clip_image_text_cosine(images: ArrayLike, input_ids: ArrayLike, attention_mask: ArrayLike, *,
                           modelname: str = CLIP_CHECKPOINT) -> jax.Array:
    """Return the per-image cosine between uint8 [N, H, W, 3] images and prompts."""
    model, _ = _get_clip(modelname)
    pixels = np.asarray(images)
    tokenized = np.shape(input_ids)[0]
    _equal_counts(pixels.shape[0], tokenized)
    if np.shape(attention_mask)[0] != tokenized:
        raise ValueError(f"CLIP got {tokenized} prompts and "
                         f"{np.shape(attention_mask)[0]} attention masks")
    image_embeds = clip_image_features(pixels, modelname=modelname)
    text_embeds = model.get_text_features(input_ids, attention_mask)
    image_embeds = image_embeds / jnp.linalg.norm(image_embeds, axis=-1, keepdims=True)
    text_embeds = text_embeds / jnp.linalg.norm(text_embeds, axis=-1, keepdims=True)
    return accumulated(functools.partial(jnp.einsum, 'nd,nd->n'), image_embeds, text_embeds)


def _clip_score(cosine: jax.Array) -> jax.Array:
    """Return each image's CLIPScore term, `100 * max(cos, 0)`, the one
    transform a validation pass and `CLIPScore.score` share."""
    return 100.0 * jnp.maximum(cosine, 0.0)


def clip_tokens(captions: Sequence[str], *,
                modelname: str = CLIP_CHECKPOINT) -> tuple[np.ndarray, np.ndarray]:
    """`captions` as CLIP's own tokenizer reads them, its ids and attention mask, each truncated to
    CLIP's context rather than overrunning its position table: what every CLIP metric scores."""
    from dew.data.text import load_tokenizer

    tokenizer = load_tokenizer(modelname)
    tokens = tokenizer(list(captions), padding="max_length", max_length=tokenizer.model_max_length,
                       truncation=True, return_tensors="np")
    return np.asarray(tokens["input_ids"]), np.asarray(tokens["attention_mask"])


def _artifact_cosine(artifact: ImageGrid, batch: Batch, field: str, modelname: str) -> jax.Array:
    """Return the per-image cosine for one sampled grid and its batch's prompts, the host captions in
    `field` (`EncodeCaptions(keep=True)`) tokenized by CLIP's own tokenizer (`clip_tokens`), whatever
    tokenizer the model's conditions read. A field of anything but text is refused (ValueError)."""
    captions = np.asarray(batch[field]) if not isinstance(batch[field], Mapping) else None
    if captions is None or captions.ndim != 1 or captions.dtype.kind not in "OUS":
        raise ValueError(f"CLIP scores the batch's captions as text in {field!r}, which holds "
                         f"{type(batch[field]).__name__} instead; keep them (EncodeCaptions(keep=True))")
    input_ids, attention_mask = clip_tokens([str(caption) for caption in captions], modelname=modelname)
    return clip_image_text_cosine(uint8_pixels(artifact.images), input_ids, attention_mask,
                                  modelname=modelname)


class CLIPDistance(ImageMetric):
    """Measures the CLIP distance, `mean(1 - cos(image, text))`; lower is better.

    It logs as `val/clip_similarity`. `CLIPScore` is the standard number to
    report for a new run.
    """

    def __init__(self, modelname: str = CLIP_CHECKPOINT, field: str = "caption"):
        def measure(artifact, batch):
            return 1.0 - _artifact_cosine(artifact, batch, field, modelname)

        super().__init__(name="clip_similarity", measure=measure, better="lower")


class CLIPScore(ImageMetric):
    """Measures CLIPScore for an image set against its prompts (`score`), or over a validation pass.

    Over a validation pass it is the mean over the sampled images and the
    batch's captions in `field`, as host text CLIP's own tokenizer reads
    (`clip_tokens`). `modelname` names the CLIP checkpoint.
    """

    def __init__(self, modelname: str = CLIP_CHECKPOINT, field: str = 'caption'):
        def measure(artifact, batch):
            return _clip_score(_artifact_cosine(artifact, batch, field, self.modelname))
        super().__init__(name='clip_score', measure=measure)
        self.modelname = modelname

    def score(self, images: ArrayLike, prompts: Sequence[str], *, batch_size: int = 64) -> float:
        """Return the CLIPScore of uint8 [N, H, W, 3] images, each against its own prompt.

        The score is `100 * mean(max(cos(image, prompt), 0))`, and higher is
        better. Typical T2I models score around 25-35 on natural prompts. The
        images are scored `batch_size` rows at a time, and the prompts are
        tokenized the way a run's batch holds them, truncated to CLIP's
        context.
        """
        # Summed the way a pass sums it, batch by batch with `+`, so a whole set in
        # one batch is the number the metric reports for the same images.
        total = 0.0
        for batch in self._batches(images, prompts, batch_size):
            total += float(batch.sum())
        return total / np.shape(images)[0]

    def scores(self, images: ArrayLike, prompts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
        """Return each image's CLIPScore against its own prompt, `100 * max(cos, 0)`, as float64 [N]:
        `score`'s terms, for ranking samples one by one."""
        return np.concatenate(list(self._batches(images, prompts, batch_size)))

    def _batches(self, images: ArrayLike, prompts: Sequence[str], batch_size: int) -> Iterator[np.ndarray]:
        """Each batch's per-image scores, float64, `batch_size` images at a time."""
        if batch_size < 1:
            raise ValueError(f"clip_score: a batch holds at least one image, got batch_size={batch_size}")
        pixels = np.asarray(images)
        if pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[-1] != 3:
            raise ValueError(f"clip_score: expected uint8 [N, H, W, 3] images, got "
                             f"{pixels.dtype} {list(pixels.shape)}")
        _equal_counts(pixels.shape[0], len(prompts))
        if pixels.shape[0] == 0:
            raise ValueError("clip_score: no images to score")
        input_ids, attention_mask = clip_tokens(prompts, modelname=self.modelname)
        with metric_device():
            for start in range(0, pixels.shape[0], batch_size):
                stop = start + batch_size
                cosine = clip_image_text_cosine(pixels[start:stop], input_ids[start:stop],
                                                attention_mask[start:stop], modelname=self.modelname)
                yield np.asarray(_clip_score(cosine), dtype=np.float64)

