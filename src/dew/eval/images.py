"""CLIP metrics on generated images.

Both metrics run the vendored towers in `dew.nn.text_encoders`, since
transformers 5 ships no `FlaxCLIPModel`, and preprocess with the checkpoint's
own PIL image processor, which transformers 5 still ships. A score here is
what the reference computes for the same pixels and tokens;
`tests/test_metrics.py` states the tolerance and the difference observed.
"""

import functools

import jax.numpy as jnp
import numpy as np

from dew.registry import metrics
from .common import ImageMetric


@functools.lru_cache(maxsize=None)
def _get_clip(modelname: str):
    """The vendored CLIP towers and the checkpoint's image processor, loaded
    once per model name: CLIP-L/14 is about 600 MB in HBM, and every metric
    built from this module shares the copy."""
    from transformers import CLIPImageProcessorPil
    from dew.nn.text_encoders import CLIPModel
    print(f"[metrics] Loading CLIP model '{modelname}' (cached for reuse)...")
    return CLIPModel.from_pretrained(modelname), CLIPImageProcessorPil.from_pretrained(modelname)


def _clip_image_text_cosine(model, processor, artifact, batch, field):
    """Per-sample cos(image, text) of shape [N], normalized the way
    `CLIPModel.forward` normalizes before its logits.

    An objective samples a fixed few rows of a batch, so the prompts are the
    leading rows of the batch's, row for row with the samples, as `paired`
    takes them for the pixel metrics.
    """
    # The sampler's [-1, 1] floats as uint8 pixels: nearest value, and clipped
    # because a sample can leave the range.
    images = np.clip(np.round((np.asarray(artifact.images) + 1.0) * 127.5), 0, 255).astype(np.uint8)
    count = images.shape[0]
    text = batch[field]
    if np.shape(text["input_ids"])[0] < count:
        raise ValueError(
            f"the artifact holds {count} rows and batch[{field!r}] only "
            f"{np.shape(text['input_ids'])[0]}; a metric that pairs an image with its "
            "prompt needs at least as many prompts as samples")
    pixel_values = processor(images=images, return_tensors="np")["pixel_values"]
    image_embeds = model.get_image_features(pixel_values)
    text_embeds = model.get_text_features(text["input_ids"][:count],
                                          text["attention_mask"][:count])
    image_embeds = image_embeds / jnp.linalg.norm(image_embeds, axis=-1, keepdims=True)
    text_embeds = text_embeds / jnp.linalg.norm(text_embeds, axis=-1, keepdims=True)
    return jnp.einsum('nd,nd->n', image_embeds, text_embeds)


@metrics("clip")
def clip(modelname: str = "openai/clip-vit-large-patch14", field: str = "text") -> ImageMetric:
    """CLIP distance, mean(1 - cos(image, text)), lower is better. Older runs
    were ranked by it as val/clip_similarity; `clip_score` is the standard
    number for a new run.
    """

    def measure(artifact, batch):
        model, processor = _get_clip(modelname)
        return jnp.mean(1.0 - _clip_image_text_cosine(model, processor, artifact, batch, field))

    return ImageMetric(name="clip_similarity", measure=measure)


@metrics("clip_score")
def clip_score(modelname: str = "openai/clip-vit-large-patch14",
               field: str = "text") -> ImageMetric:
    """Standard CLIPScore: 100 * max(cos(img, text), 0), higher is better.
    Typical T2I models score around 25-35 on natural prompts.
    """

    def measure(artifact, batch):
        model, processor = _get_clip(modelname)
        cos = _clip_image_text_cosine(model, processor, artifact, batch, field)
        return jnp.mean(100.0 * jnp.maximum(cos, 0.0))

    return ImageMetric(name="clip_score", measure=measure)
