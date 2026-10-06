"""The CLIP metrics on the tiny checkpoint under tests/fixtures/clip, scored
against the cosines of the reference's own embeddings."""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from dew.artifacts import ImageGrid
from dew.eval import CLIPDistance, CLIPScore

CLIP_TINY = Path(__file__).resolve().parent / "fixtures" / "clip" / "tiny"


def clip_fixture():
    """The reference's inputs and embeddings, and the sampler-shaped batch
    that carries them: images in [-1, 1] and the tokenized captions."""
    reference = np.load(CLIP_TINY / "reference.npz")
    recipe = json.loads((CLIP_TINY / "prompts.json").read_text())["images"]
    images = np.random.RandomState(recipe["seed"]).randint(
        0, 256, tuple(recipe["shape"]), dtype=np.uint8)
    generated = jnp.asarray(images, jnp.float32) / 127.5 - 1.0
    batch = {"text": {"input_ids": reference["input_ids"],
                      "attention_mask": reference["attention_mask"]}}
    image = reference["image_embeds"].astype(np.float64)
    text = reference["text_embeds"].astype(np.float64)
    cosine = ((image * text).sum(-1)
              / np.linalg.norm(image, axis=-1) / np.linalg.norm(text, axis=-1))
    return generated, batch, cosine


# One fp32 cosine differs by ~1e-7 across devices; CLIPScore is a hundred
# cosines, so its bound is the same relative one.
CLIP_TOLERANCE = 1e-5


CLIP_SCORE_TOLERANCE = 1e-3


def test_clip_metric_scores_the_reference_cosine():
    """The factory builds on the vendored towers, the images go through the
    checkpoint's own processor, and the score is the reference's
    mean(1 - cos): observed 3.2e-08 off it against a tolerance of 1e-5. Before
    the towers were vendored, the factory raised ImportError on
    `FlaxCLIPModel`, which transformers 5 removed."""
    metric = CLIPDistance(modelname=str(CLIP_TINY))
    assert metric.name == 'clip_similarity' and metric.reads is ImageGrid
    generated, batch, cosine = clip_fixture()

    score = metric.finalize(metric(ImageGrid(generated), batch))

    expected = np.mean(1.0 - cosine)
    assert abs(score - expected) < CLIP_TOLERANCE, f"{score} against {expected}"


def test_a_run_ranked_by_clip_distance_keeps_its_lowest():
    """CLIP distance is 1 - cos, so the best checkpoint is the closest one:
    the metric declares lower is better, and a selection by it minimizes."""
    from dew.training.selection import Best
    from dew.training.trainer import Trainer

    metric = CLIPDistance(modelname="never/downloaded")
    assert metric.shown.better == "lower"
    assert Trainer._best_selection(Best(metric), [metric]).mode == "min"


def test_clip_score_metric_clamps_the_reference_cosine():
    """CLIPScore is 100 * max(cos, 0) averaged; the fixture holds one negative
    cosine (-0.072) among three positive ones, so the clamp does work here.
    Observed 6.1e-06 off the reference on CPU and 1.0e-05 on an RTX 4080,
    against a tolerance of 1e-3 on a score of order 15."""
    metric = CLIPScore(modelname=str(CLIP_TINY))
    assert metric.name == 'clip_score'
    generated, batch, cosine = clip_fixture()
    assert (cosine < 0).any() and (cosine > 0).any()

    score = metric.finalize(metric(ImageGrid(generated), batch))

    expected = np.mean(100.0 * np.maximum(cosine, 0.0))
    assert abs(score - expected) < CLIP_SCORE_TOLERANCE, f"{score} against {expected}"
    assert score != pytest.approx(np.mean(100.0 * cosine), abs=1e-3)


def test_clip_score_over_images_and_prompts_is_the_metric_number():
    """`CLIPScore.score` takes uint8 images and prompt strings, with no artifact
    and no tokenized batch, and tokenizes them the way a run's loader does. It
    lands on the metric's value for the same fixture, and both land on the
    reference's own cosines: a different padding or truncation would move the
    score off them, since one of the four cosines is negative and the clamp
    reads it."""
    prompts = json.loads((CLIP_TINY / "prompts.json").read_text())["prompts"]
    images = np.load(CLIP_TINY / "reference.npz")["images"]
    generated, batch, cosine = clip_fixture()
    metric = CLIPScore(modelname=str(CLIP_TINY))

    score = CLIPScore(str(CLIP_TINY)).score(images, prompts)

    pooled = metric.finalize(metric(ImageGrid(generated), batch))
    assert score == pytest.approx(pooled, abs=1e-9), f"{score} against {pooled}"
    expected = np.mean(100.0 * np.maximum(cosine, 0.0))
    assert abs(score - expected) < CLIP_SCORE_TOLERANCE, f"{score} against {expected}"


def test_clip_score_batches_a_set_into_the_score_of_the_whole_set():
    """`batch_size` splits the images and their prompts together. The score is
    a mean over images, so the split cannot move it: a misaligned slice would
    pair the wrong prompt with the wrong image and change the number."""
    prompts = json.loads((CLIP_TINY / "prompts.json").read_text())["prompts"]
    images = np.load(CLIP_TINY / "reference.npz")["images"]

    whole = CLIPScore(str(CLIP_TINY)).score(images, prompts)
    batched = CLIPScore(str(CLIP_TINY)).score(images, prompts, batch_size=3)

    assert batched == pytest.approx(whole, rel=1e-6)
    with pytest.raises(ValueError, match="equal counts"):
        CLIPScore(str(CLIP_TINY)).score(images, prompts[:2])


def test_clip_score_truncates_an_overlong_caption_to_the_text_context():
    """A caption past CLIP's context is cut to it, so 500 words and 600 score
    alike instead of overrunning the position table."""
    images = np.load(CLIP_TINY / "reference.npz")["images"][:1]

    long = CLIPScore(str(CLIP_TINY)).score(images, [" ".join(["word"] * 500)])

    assert np.isfinite(long)
    assert CLIPScore(str(CLIP_TINY)).score(images, [" ".join(["word"] * 600)]) == long


def test_a_sample_outside_the_pixel_range_is_clipped_not_wrapped():
    """A sampler does not promise [-1, 1]. Casting 1.2 straight to uint8 wraps
    it to a dark pixel, which the old metric did; the score of an overshooting
    white image has to be the score of a white one."""
    metric = CLIPScore(modelname=str(CLIP_TINY))
    _, batch, _ = clip_fixture()
    white = jnp.ones((4, 16, 12, 3), jnp.float32)

    assert metric.finalize(metric(ImageGrid(1.2 * white), batch)) == metric.finalize(
        metric(ImageGrid(white), batch)
    )
    assert metric.finalize(metric(ImageGrid(-1.2 * white), batch)) == metric.finalize(
        metric(ImageGrid(-white), batch)
    )
    assert metric.finalize(metric(ImageGrid(white), batch)) != metric.finalize(
        metric(ImageGrid(-white), batch)
    )
