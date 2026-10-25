"""Multimodal wrappers: config translation and weight maps with parity.

translate_wrapper_config turns a gemma3, llama4, gemma4 or qwen3_5 wrapper
into its decoder, tower and projector records; translate_wrapper_weights
routes the released model.* nesting into the three trees, whose prefixes the
translation tests write down per family. Tower and projector parity rides the
tiny wrapper fixtures, whose vision halves come from the shared systems in
tools/hf_reference.py at a fresh seed with these observed differences, fp32
on CPU:

- gemma3-tiny-mm tower: max |difference| 9.6e-07, projector 4.8e-07,
  tolerance 1e-4. Prefixes model.vision_tower.* and
  model.multi_modal_projector.*.
- llama4-tiny-mm tower: max |difference| 7.0e-06, projector 1.5e-06,
  tolerance 1e-4. Bare vision_model.* and multi_modal_projector.* with no
  model. prefix.
- gemma4-tiny-mm tower: max |difference| 2.5e-05, projector 9.6e-07,
  tolerance 1e-4. Prefixes model.vision_tower.* and model.embed_vision.*.
- qwen35-tiny-mm tower: max |difference| 7.4e-06, projector 3.9e-06,
  tolerance 1e-4. Prefixes model.visual.* with the merger under
  model.visual.merger.*.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from safetensors.numpy import load_file

from dew.interop.hf_decoders import translate_wrapper_config, translate_wrapper_weights
from dew.registry import projectors, towers

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"


@pytest.mark.parametrize("family", ["gemma3-tiny-mm", "gemma4-tiny-mm"])
def test_a_gemma_wrapper_keeps_image_and_source_block_visibility_together(tmp_path, family):
    import shutil

    from dew.interop import Pretrained

    directory = tmp_path / "source"
    shutil.copytree(FIXTURES / family, directory)
    config = json.loads((directory / "config.json").read_text())
    config["modeling_overrides"] = {"attention": {"sliding_attention": "block_bidirectional"}}
    (directory / "config.json").write_text(json.dumps(config))
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    tokens = jnp.asarray([[10, 11, 12, 13]], jnp.int32)
    altered = tokens.at[0, 1].set(14)
    groups = jnp.asarray([[0, 0, -1, -1]], jnp.int32)

    def first(ids, **metadata):
        return np.asarray(loaded.model.apply(loaded.variables, ids, **metadata))[0, 0]

    np.testing.assert_array_equal(first(tokens), first(altered))
    for field in ("image_groups", "block_groups"):
        assert np.any(first(tokens, **{field: groups}) != first(altered, **{field: groups})), field


def test_an_explicit_causal_source_override_exports_through_the_native_decoder(tmp_path):
    import shutil

    from dew.interop import Pretrained
    from dew.interop.pretrained import PretrainedDecoder

    directory = tmp_path / "source"
    shutil.copytree(FIXTURES / "llama-tiny", directory)
    config = json.loads((directory / "config.json").read_text())
    config["modeling_overrides"] = {"attention": {"full_attention": "causal"}}
    (directory / "config.json").write_text(json.dumps(config))
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    exported = tmp_path / "derived"
    PretrainedDecoder.from_model(loaded.model, loaded.variables).save(exported)
    restored = Pretrained.load(exported, dtype="float32", attention_impl="reference")
    tokens = jnp.asarray([[1, 2, 3, 4]], jnp.int32)
    np.testing.assert_array_equal(restored.model.apply(restored.variables, tokens),
                                  loaded.model.apply(loaded.variables, tokens))


def test_a_wrapper_dtype_change_reaches_the_decoder_that_computes_its_logits():
    """Changing the multimodal model's run dtype changes its native text decoder too, while the
    fp32 master variables remain reusable by the original model."""
    from dew.interop import Pretrained
    from dew.registry import ModelConfig

    loaded = Pretrained.load(FIXTURES / "qwen35-tiny-mm", dtype="float32", attention_impl="reference")
    changed = ModelConfig.from_model(loaded.model).with_dtype("bfloat16").build()
    ids = jnp.asarray([[1, 2, 3]], jnp.int32)
    original = loaded.model.apply(loaded.variables, ids)
    narrowed = changed.apply(loaded.variables, ids)
    hidden = changed.apply(loaded.variables, ids, method=changed.hidden_states)
    assert original.dtype == narrowed.dtype == jnp.float32
    assert hidden.dtype == jnp.bfloat16
    assert np.isfinite(np.asarray(narrowed, np.float32)).all()
    np.testing.assert_array_equal(loaded.model.apply(loaded.variables, ids), original)


def test_released_block_attention_matches_transformers_against_float64():
    """The release's own mask adapter over independent block layouts, including causal-only
    rows, judged against the paired fp32 and float64 transformers recordings."""
    import jax
    from reference_error import assert_as_exact, declared_seeds, indexed_recording, recording

    from dew.interop import Pretrained

    directory = FIXTURES / "qwen35-blocks-tiny"
    source = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    path = directory / "reference.npz"
    seeds = declared_seeds(path)["blocks"]
    draw, reference, truth = indexed_recording(recording(path), "blocks", seeds)

    @jax.jit
    def forward(index, inputs):
        return {"logits": source.model.apply(
            source.variables, inputs["input_ids"], block_groups=inputs["block_groups"])}

    assert_as_exact(forward, reference, draw, "Qwen3.5 released block attention",
                    seeds=seeds, truth_fn=truth)


def test_wrapper_block_overrides_survive_export_and_admit_their_future_block_mate(tmp_path):
    """A source declaring block attention changes visibility only where runtime block IDs admit
    it. A future block mate can affect the earlier token, and source export preserves that rule."""
    import shutil

    from dew.interop import Pretrained

    directory = tmp_path / "source"
    shutil.copytree(FIXTURES / "qwen35-tiny-mm", directory)
    config = json.loads((directory / "config.json").read_text())
    config["modeling_overrides"] = {"attention": {
        "full_attention": "block_bidirectional", "linear_attention": "causal"}}
    (directory / "config.json").write_text(json.dumps(config))
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    ids = jnp.asarray([[1, 2, 3, 4]], jnp.int32)
    altered = ids.at[0, 1].set(7)
    blocked = jnp.asarray([[0, 0, -1, -1]], jnp.int32)

    def first(bundle, tokens, groups=None):
        return np.asarray(bundle.model.apply(bundle.variables, tokens, block_groups=groups))[0, 0]

    np.testing.assert_array_equal(first(loaded, ids), first(loaded, altered))
    assert np.any(first(loaded, ids, blocked) != first(loaded, altered, blocked))
    loaded.save(tmp_path / "export")
    restored = Pretrained.load(tmp_path / "export", dtype="float32", attention_impl="reference")
    np.testing.assert_array_equal(first(restored, ids, blocked), first(loaded, ids, blocked))
    np.testing.assert_array_equal(first(restored, ids), first(loaded, ids))


def load_wrapper(name):
    directory = FIXTURES / name
    record = translate_wrapper_config(
        json.loads((directory / "config.json").read_text()))
    variables = translate_wrapper_weights(
        load_file(str(directory / "model.safetensors")), record)
    return directory, record, variables


def wrapper_pixels(directory, record):
    """Fixture pixels as the tower reads them: the Gemma 4 fixture stores
    processor patches, which fold back into the image row-major."""
    pixels = np.load(directory / "pixels.npy")
    if record["tower"]["class"] != "gemma4":
        return pixels
    batch, count, _ = pixels.shape
    grid = int(count ** 0.5)
    patch = int(record["tower"]["fields"]["patch_size"])
    side = grid * patch
    return pixels.reshape(batch, grid, grid, patch, patch, 3).transpose(
        0, 5, 1, 3, 2, 4).reshape(batch, 3, side, side)


def test_wrapper_tower_and_projector_match_the_reference():
    """The vision halves of all four wrapper fixtures against their committed
    reference outputs."""
    for name, tolerance in (("gemma3-tiny-mm", 1e-4), ("llama4-tiny-mm", 1e-4),
                            ("gemma4-tiny-mm", 1e-4), ("qwen35-tiny-mm", 1e-4)):
        directory, record, variables = load_wrapper(name)
        pixels = wrapper_pixels(directory, record)
        tower_ref = np.load(directory / "tower_ref.npy")
        projector_ref = np.load(directory / "projector_ref.npy")
        tower = towers.from_record(record["tower"]).build()
        assert np.max(np.abs(np.asarray(
            tower.apply(variables["tower"], pixels)) - tower_ref)) < tolerance
        projector = projectors.from_record(record["projector"])
        assert np.max(np.abs(np.asarray(
            projector.apply(variables["projector"], tower_ref))
            - projector_ref)) < tolerance


def test_every_stored_spelling_of_a_media_tensor_writes_back_from_its_one_leaf(tmp_path):
    """A checkpoint may store a vision tensor under two names that load to one leaf; a trained
    leaf writes back under both, so the export reloads as the trained model, neither name kept
    at its stale source value."""
    import dataclasses
    from shutil import copyfile

    import jax
    from safetensors.numpy import save_file

    from dew.interop import Pretrained

    source = tmp_path / "source"
    source.mkdir()
    for name in ("config.json",):
        copyfile(FIXTURES / "gemma3-tiny-mm" / name, source / name)
    tensors = load_file(FIXTURES / "gemma3-tiny-mm" / "model.safetensors")
    spelled = "model.vision_tower.embeddings.patch_embedding.bias"
    tensors[spelled.removeprefix("model.")] = tensors[spelled]
    save_file(tensors, source / "model.safetensors")
    loaded = Pretrained.load(source, dtype="float32", attention_impl="reference")
    trained = dataclasses.replace(loaded, variables=jax.tree.map(lambda leaf: leaf + 0.25, loaded.variables))
    trained.save(tmp_path / "export")
    written = load_file(tmp_path / "export" / "model.safetensors")
    np.testing.assert_array_equal(written[spelled], tensors[spelled] + 0.25)
    np.testing.assert_array_equal(written[spelled.removeprefix("model.")], written[spelled])
    Pretrained.load(tmp_path / "export", dtype="float32", attention_impl="reference")
