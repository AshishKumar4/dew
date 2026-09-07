"""Raw/prepared image tasks and snapshots through real native model computation."""

import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.inference import BlockGeneration, DenoisingInputs, TextToImage
from dew.interop import load_pretrained
from dew.nn.inputs import ModelInputs
from dew.sampling import CFG, Heun
from test_inference import make_run

FIXTURE = Path(__file__).parent / "fixtures/hf/diffusion-gemma-workflow"


def test_trained_image_task_accepts_raw_and_prepared_inputs_and_immutable_rebinding(tmp_path):
    objective, state = make_run(tmp_path)
    task = TextToImage.from_objective(objective, state.params)
    key = jax.random.key(23)
    raw = task(["flower", "tree"], steps=3, sampler=Heun(), guidance=CFG(2.0), key=key).host().images
    prepared = task.prepare(["flower", "tree"], key=key)
    assert isinstance(prepared, DenoisingInputs) and prepared.rows == 2
    again = task(prepared, steps=3, sampler=Heun(), guidance=CFG(2.0), key=key).host().images
    np.testing.assert_array_equal(again, raw)
    loaded = TextToImage.from_run(str(tmp_path), ema=False)
    np.testing.assert_allclose(loaded(["flower", "tree"], steps=3, sampler=Heun(), guidance=CFG(2.0), key=key).host().images,
                               raw, atol=2e-6, rtol=2e-6)
    mutable = jax.tree.map(lambda leaf: leaf, task.params.unfreeze())
    bound = task.bind(mutable)
    mutable["params"] = jax.tree.map(lambda leaf: leaf + 0.05, mutable["params"])
    np.testing.assert_array_equal(bound(prepared, steps=3, sampler=Heun(), guidance=CFG(2.0), key=key).host().images, raw)
    changed = task.bind(mutable)(prepared, steps=3, sampler=Heun(), guidance=CFG(2.0), key=key).host().images
    assert np.max(np.abs(changed - raw)) > 1e-4
    np.testing.assert_allclose(task("flower", steps=3, sampler=Heun(), key=key).host().images,
                               task(["flower"], steps=3, sampler=Heun(), key=key).host().images, atol=0, rtol=0)
    with pytest.raises(ValueError, match="initial noise"):
        task(replace(prepared, noise=prepared.noise[:, :-1]), steps=3, key=key)


def test_canvas_raw_media_processing_reaches_the_real_conditioner():
    source = load_pretrained(FIXTURE, dtype="float32", attention_impl="xla", max_seq_len=32)
    with np.load(FIXTURE / "reference.npz") as reference:
        prompt = np.asarray(reference["image_prompt"])
        pixels = np.asarray(reference["pixels"])
        expected = np.asarray(reference["image_tokens"][:, :12])

    class PixelProcessor:
        """This fixture's image payloads are already normalized pixel tensors."""
        def __call__(self, text, *, images=None):
            if len(text) != len(prompt) or images is None:
                raise ValueError("each fixture prompt needs its image")
            return ModelInputs(jnp.asarray(prompt), {"image_indices": jnp.where(prompt == 60, 0, -1)},
                               {"pixel_values": jnp.asarray(images)})

        def decode(self, ids):
            return [" ".join(map(str, row)) for row in np.asarray(ids)]

    processor = PixelProcessor()
    task = replace(source.block_generation(), processor=processor)
    generated = task(["image A", "image B"], 7, images=pixels, key=jax.random.key(11))
    np.testing.assert_array_equal(generated.tokens, expected)
    prepared = processor(["image A", "image B"], images=pixels)
    direct = task(prepared, 7, key=jax.random.key(11))
    np.testing.assert_array_equal(direct.tokens, generated.tokens)
    different = task(["image A", "image B"], 7, images=-pixels, key=jax.random.key(11))
    assert not np.array_equal(different.tokens, generated.tokens)
    for invalid in ([[1.9, 2.8]], [[1, 2], "drop this"], [[True, 1]]):
        with pytest.raises(ValueError):
            task(invalid, 2, key=jax.random.key(1))
