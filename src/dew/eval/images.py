"""CLIP metrics on generated images.

Both metrics run the vendored towers in `dew.nn.text_encoders`, since
transformers 5 ships no `FlaxCLIPModel`, and preprocess with the checkpoint's
own PIL image processor, which transformers 5 still ships. A score here is
what the reference computes for the same pixels and tokens;
`tests/test_metrics.py` states the tolerance and the difference observed.
"""

from .common import EvaluationMetric
import jax.numpy as jnp
import numpy as np


# Cache the CLIP model so multiple metrics share one copy of the weights
# instead of loading CLIP-L/14 into HBM once per metric (~600MB each).
_clip_cache: dict = {}


def _get_clip(modelname: str):
    """Cached (model, processor) pair for the given CLIP modelname."""
    if modelname not in _clip_cache:
        from transformers import CLIPImageProcessorPil
        from dew.nn.text_encoders import CLIPModel
        print(f"[metrics] Loading CLIP model '{modelname}' (cached for reuse)...")
        _clip_cache[modelname] = (CLIPModel.from_pretrained(modelname),
                                  CLIPImageProcessorPil.from_pretrained(modelname))
    return _clip_cache[modelname]


def _clip_image_text_cosine(model, processor, generated, batch):
    """Per-sample cos(image, text) of shape [B], normalized the way
    `CLIPModel.forward` normalizes before its logits."""
    text = batch['text']
    # The sampler's [-1, 1] floats as uint8 pixels: nearest value, and clipped
    # because a sample can leave the range.
    images = np.clip(np.round((np.asarray(generated) + 1.0) * 127.5), 0, 255).astype(np.uint8)
    pixel_values = processor(images=images, return_tensors="np")["pixel_values"]
    image_embeds = model.get_image_features(pixel_values)
    text_embeds = model.get_text_features(text['input_ids'], text['attention_mask'])
    image_embeds = image_embeds / jnp.linalg.norm(image_embeds, axis=-1, keepdims=True)
    text_embeds = text_embeds / jnp.linalg.norm(text_embeds, axis=-1, keepdims=True)
    return jnp.einsum('bd,bd->b', image_embeds, text_embeds)


def get_clip_metric(
    modelname: str = "openai/clip-vit-large-patch14",
):
    """Old CLIP distance metric: mean(1 - cos(image, text)), lower is better.
    Kept so older runs ranked by best_val/clip_similarity stay comparable;
    prefer get_clip_score_metric for new runs.
    """
    model, processor = _get_clip(modelname)

    def clip_metric(generated: jnp.ndarray, batch):
        cos = _clip_image_text_cosine(model, processor, generated, batch)
        return jnp.mean(1.0 - cos)

    return EvaluationMetric(function=clip_metric, name='clip_similarity')


def get_clip_score_metric(
    modelname: str = "openai/clip-vit-large-patch14",
):
    """Standard CLIPScore: 100 * max(cos(img, text), 0), higher is better.
    Typical T2I models score around 25-35 on natural prompts.
    """
    model, processor = _get_clip(modelname)

    def clip_score_metric(generated: jnp.ndarray, batch):
        cos = _clip_image_text_cosine(model, processor, generated, batch)
        return jnp.mean(100.0 * jnp.maximum(cos, 0.0))

    return EvaluationMetric(
        function=clip_score_metric,
        name='clip_score',
        higher_is_better=True,
    )
