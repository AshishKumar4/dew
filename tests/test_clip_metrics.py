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


def clip_fixture(conditions=None):
    """The reference's inputs and embeddings, and the sampler-shaped batch
    that carries them: images in [-1, 1] and the captions as text, beside
    the fields a run's `conditions` tokenized them into."""
    reference = np.load(CLIP_TINY / "reference.npz")
    fixture = json.loads((CLIP_TINY / "prompts.json").read_text())
    recipe = fixture["images"]
    images = np.random.RandomState(recipe["seed"]).randint(
        0, 256, tuple(recipe["shape"]), dtype=np.uint8)
    generated = jnp.asarray(images, jnp.float32) / 127.5 - 1.0
    batch = {"caption": np.asarray(fixture["prompts"], object),
             **({} if conditions is None else {"text": conditions(fixture["prompts"])})}
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


@pytest.mark.parametrize("conditions", ["t5/tiny", "tokenizers/gpt2"])
def test_a_validation_pass_scores_its_captions_in_clips_own_words(conditions):
    """A run whose conditions read the captions through another tokenizer (T5's SentencePiece, GPT-2's
    byte-level BPE) holds them in that tokenizer's ids; the CLIP metrics score the captions
    themselves, as `CLIPScore.score` does a set of prompts, so the condition's ids never reach
    CLIP."""
    from dew.data.text import load_tokenizer

    tokenizer = load_tokenizer(str(Path(__file__).resolve().parent / "fixtures" / conditions))
    generated, batch, _ = clip_fixture(
        lambda prompts: dict(tokenizer(prompts, padding=True, return_tensors="np")))
    metric = CLIPScore(str(CLIP_TINY))
    pooled = metric.finalize(metric(ImageGrid(generated), batch))
    prompts = json.loads((CLIP_TINY / "prompts.json").read_text())["prompts"]
    images = np.load(CLIP_TINY / "reference.npz")["images"]
    assert pooled == pytest.approx(metric.score(images, prompts), abs=1e-9)


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
    with pytest.raises(ValueError):
        CLIPScore(str(CLIP_TINY)).score(images, prompts[:2])


def test_each_images_clip_score_is_a_term_of_the_sets():
    """`scores` gives each image's own score, in order and whatever the batch,
    and `score` is their mean: one image paired with another's prompt would
    move both."""
    prompts = json.loads((CLIP_TINY / "prompts.json").read_text())["prompts"]
    images = np.load(CLIP_TINY / "reference.npz")["images"]
    metric = CLIPScore(str(CLIP_TINY))
    each = metric.scores(images, prompts)
    _, _, cosine = clip_fixture()

    assert each.shape == (len(prompts),) and each.dtype == np.float64
    np.testing.assert_allclose(each, 100.0 * np.maximum(cosine, 0.0), atol=CLIP_SCORE_TOLERANCE)
    # A batch of another size compiles its own kernels, which round the last digit apart.
    np.testing.assert_allclose(metric.scores(images, prompts, batch_size=3), each, rtol=1e-6)
    assert metric.score(images, prompts) == pytest.approx(each.mean(), abs=1e-9)
    assert metric.scores(images[::-1], prompts[::-1])[::-1] == pytest.approx(each, abs=1e-9)
