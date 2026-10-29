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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import respelled
from safetensors.numpy import load_file

from dew.interop.hf_decoders import translate_wrapper_config, translate_wrapper_weights
from dew.registry import projectors, towers

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"


@pytest.mark.parametrize("family", ["gemma3-tiny-mm", "gemma4-tiny-mm"])
def test_a_gemma_wrapper_keeps_image_and_source_block_visibility_together(tmp_path, family):

    from dew.interop import Pretrained

    directory = respelled(FIXTURES / family, tmp_path / "source", lambda config: {
        **config, "modeling_overrides": {"attention": {"sliding_attention": "block_bidirectional"}}})
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

    from dew.interop import Pretrained
    from dew.interop.pretrained import PretrainedDecoder

    directory = respelled(FIXTURES / "llama-tiny", tmp_path / "source", lambda config: {
        **config, "modeling_overrides": {"attention": {"full_attention": "causal"}}})
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

    from dew.interop import Pretrained

    directory = respelled(FIXTURES / "qwen35-tiny-mm", tmp_path / "source", lambda config: {
        **config, "modeling_overrides": {"attention": {"full_attention": "block_bidirectional",
                                                       "linear_attention": "causal"}}})
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


def _without(source: Path, target: Path, dropped) -> Path:
    """A copy of `source` whose weights omit every tensor `dropped` names."""
    import shutil

    from safetensors.numpy import save_file

    shutil.copytree(source, target)
    tensors = load_file(target / "model.safetensors")
    save_file({name: array for name, array in tensors.items() if not dropped(name)},
              target / "model.safetensors", metadata={"format": "pt"})
    return target


def test_a_text_only_load_is_the_wrappers_language_model_as_its_text_checkpoint(tmp_path):
    """`text_only` loads Qwen3.5's language model alone: a checkpoint without its vision tower loads as
    a decoder holding no media part, it computes the wrapper's text logits bit for bit, and it saves as
    Qwen3.5's text checkpoint (`Qwen3_5ForCausalLM`, no tower), which loads back to the same logits."""
    from dew.interop import Pretrained, PretrainedDecoder
    from dew.nn.backbones.causal_transformer import CausalTransformer

    source = FIXTURES / "qwen35-native-tiny"
    blind = _without(source, tmp_path / "blind", lambda name: name.startswith("model.visual."))
    text = Pretrained.load(blind, dtype="float32", attention_impl="reference", text_only=True)
    wrapper = Pretrained.load(source, dtype="float32", attention_impl="reference")
    assert isinstance(text, PretrainedDecoder) and isinstance(text.model, CausalTransformer)
    ids = jnp.asarray(np.load(source / "input_ids.npy"))
    ids = jnp.where(ids < text.model.vocab_size, ids, 0)
    logits = np.asarray(text.model.apply(text.variables, ids))
    np.testing.assert_array_equal(logits, np.asarray(wrapper.model.apply(wrapper.variables, ids)))

    text.save(tmp_path / "export")
    config = json.loads((tmp_path / "export" / "config.json").read_text())
    assert (config["model_type"], config["architectures"]) == ("qwen3_5_text", ["Qwen3_5ForCausalLM"])
    assert not any(name.startswith(("model.visual.", "visual.")) for name in load_file(
        tmp_path / "export" / "model.safetensors"))
    reloaded = Pretrained.load(tmp_path / "export", dtype="float32", attention_impl="reference")
    np.testing.assert_array_equal(np.asarray(reloaded.model.apply(reloaded.variables, ids)), logits)


def test_a_text_only_load_takes_a_checkpoint_without_its_prediction_depth():
    """A Qwen3.5 release whose config names one prediction depth but whose checkpoint holds the trunk
    alone (re-saved by transformers, which builds no depth) loads text-only as that trunk."""
    from dew.interop import Pretrained

    config = json.loads((FIXTURES / "qwen35-blocks-tiny" / "config.json").read_text())
    assert config["text_config"]["mtp_num_hidden_layers"] == 1
    text = Pretrained.load(FIXTURES / "qwen35-blocks-tiny", dtype="float32", attention_impl="reference",
                           text_only=True)
    assert text.model.num_nextn_predict_layers == 0
    assert not any(str(name).startswith("mtp_") for name in text.variables["params"])


def _misshapen_depths(source: Path, target: Path, depth: str) -> Path:
    """A copy of `source` whose tensors of the prediction depth `depth` names are of a shape no decoder
    reads, so a load that builds or reads that depth refuses it."""
    import shutil

    from safetensors.numpy import save_file

    shutil.copytree(source, target)
    for shard in target.glob("*.safetensors"):
        tensors = load_file(shard)
        save_file({name: np.zeros((3,), array.dtype) if name.startswith(depth) else array
                   for name, array in tensors.items()}, shard, metadata={"format": "pt"})
    return target


@pytest.mark.parametrize(("fixture", "depth", "text_only"), [("glm4-moe-tiny", "model.layers.2.", False),
                                                             ("qwen38-dense-tiny", "mtp.", True),
                                                             ("qwen38-dense-tiny", "mtp.", False)],
                         ids=["glm", "qwen-text", "qwen-wrapper"])
def test_a_load_builds_and_reads_only_the_prediction_depths_it_keeps(tmp_path, fixture, depth, text_only):
    """`prediction_depths=0` builds a decoder's trunk alone, a wrapper's around it: its depth is neither
    built nor read, so a copy whose depth no decoder could read loads as the source does, to the same
    logits, and saves as a checkpoint of its trunk, reloading to the same model. A count at least the
    checkpoint's keeps every depth it holds."""
    from dew.interop import Pretrained

    def load(directory, **options):
        return Pretrained.load(directory, dtype="float32", attention_impl="reference", text_only=text_only,
                               **options)

    def decoder(loaded):
        return loaded.model if text_only or fixture.startswith("glm") else loaded.model.language_model

    source = FIXTURES / fixture
    whole = load(source)
    assert decoder(whole).num_nextn_predict_layers == 1
    assert load(source, prediction_depths=2).model == whole.model
    misshapen = _misshapen_depths(source, tmp_path / "misshapen", depth)
    with pytest.raises(ValueError):
        load(misshapen)
    loaded = load(misshapen, prediction_depths=0)
    assert decoder(loaded).num_nextn_predict_layers == 0
    leaves = jax.tree_util.tree_flatten_with_path(loaded.variables)[0]
    paths = [jax.tree_util.keystr(path) for path, _ in leaves]
    assert not any("mtp_" in path for path in paths)
    ids = jnp.arange(8, dtype=jnp.int32)[None] % decoder(loaded).vocab_size
    np.testing.assert_array_equal(np.asarray(loaded.model.apply(loaded.variables, ids)),
                                  np.asarray(whole.model.apply(whole.variables, ids)))
    loaded.save(tmp_path / "saved")
    reloaded = load(tmp_path / "saved")
    assert decoder(reloaded).num_nextn_predict_layers == 0
    np.testing.assert_array_equal(np.asarray(reloaded.model.apply(reloaded.variables, ids)),
                                  np.asarray(loaded.model.apply(loaded.variables, ids)))


@pytest.mark.parametrize("text_only", [False, True], ids=["wrapper", "text"])
def test_a_wrappers_nested_prediction_layer_is_the_depth_a_load_drops(tmp_path, text_only):
    """Qwen3.5's prediction layer stored under the wrapper's `model.language_model.mtp.*`, a spelling
    the wrapper reads as its bare `mtp.*`, is the depth `prediction_depths=0` drops: the copy so
    stored loads to the source's trunk, leaf for leaf."""
    import shutil

    from safetensors.numpy import save_file

    from dew.interop import Pretrained

    def load(directory, **options):
        return Pretrained.load(directory, dtype="float32", attention_impl="reference", text_only=text_only,
                               **options)

    source = FIXTURES / "qwen38-dense-tiny"
    nested = tmp_path / "nested"
    shutil.copytree(source, nested)
    for shard in nested.glob("*.safetensors"):
        tensors = load_file(shard)
        assert any(name.startswith("mtp.") for name in tensors)
        save_file({f"model.language_model.{name}" if name.startswith("mtp.") else name: array
                   for name, array in tensors.items()}, shard, metadata={"format": "pt"})
    ours, theirs = load(nested, prediction_depths=0).variables, load(source, prediction_depths=0).variables
    leaves = [jax.tree_util.tree_flatten_with_path(tree)[0] for tree in (ours, theirs)]
    assert [path for path, _ in leaves[0]] == [path for path, _ in leaves[1]]
    assert not any("mtp_" in jax.tree_util.keystr(path) for path, _ in leaves[0])
    for (_, mine), (_, other) in zip(*leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(other))


@pytest.mark.parametrize("text_only", [False, True], ids=["wrapper", "text"])
def test_a_discarded_depths_conflicting_spellings_are_never_compared(tmp_path, text_only):
    """A depth `prediction_depths=0` drops is read by none, its spellings neither: a copy of Qwen3.5
    spelling one of its prediction layer's tensors twice, as `mtp.*` and `model.language_model.mtp.*`,
    unequal, is refused where the load keeps the depth, and loads to the source's trunk where it
    drops it, leaf for leaf."""
    import shutil

    from safetensors.numpy import save_file

    from dew.interop import Pretrained

    def load(directory, **options):
        return Pretrained.load(directory, dtype="float32", attention_impl="reference", text_only=text_only,
                               **options)

    source = FIXTURES / "qwen38-dense-tiny"
    spelled = tmp_path / "spelled"
    shutil.copytree(source, spelled)
    for shard in spelled.glob("*.safetensors"):
        tensors = load_file(shard)
        depth = next((name for name in tensors if name.startswith("mtp.")), None)
        if depth is not None:
            tensors[f"model.language_model.{depth}"] = tensors[depth] + 1
            save_file(tensors, shard, metadata={"format": "pt"})
    with pytest.raises(ValueError):
        load(spelled)
    ours, theirs = load(spelled, prediction_depths=0).variables, load(source, prediction_depths=0).variables
    leaves = [jax.tree_util.tree_flatten_with_path(tree)[0] for tree in (ours, theirs)]
    assert [path for path, _ in leaves[0]] == [path for path, _ in leaves[1]]
    for (_, mine), (_, other) in zip(*leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(other))


@pytest.mark.parametrize("fixture", ["deepseek-v4-dspark-tiny", "deepseek-v41-tiny"])
def test_a_drafter_is_kept_whole_or_not_at_all(tmp_path, fixture):
    """A DSpark drafter's stages draft as one, its heads on its last: a count of all its stages keeps
    it whole, `prediction_depths=0` builds none of it and saves a checkpoint of the trunk alone, which
    reloads to the trunk, and a count between is refused."""
    from dew.interop import Pretrained

    def load(directory, **options):
        return Pretrained.load(directory, dtype="float32", attention_impl="reference", **options)

    def drafted(loaded) -> int:
        dspark = loaded.model.dspark
        return 0 if dspark is None else dspark.stages

    source = FIXTURES / fixture
    assert drafted(load(source)) == drafted(load(source, prediction_depths=3)) == 3
    with pytest.raises(ValueError):
        load(source, prediction_depths=1)
    loaded = load(source, prediction_depths=0)
    assert drafted(loaded) == 0
    leaves = jax.tree_util.tree_flatten_with_path(loaded.variables)[0]
    assert not any("dspark_" in jax.tree_util.keystr(path) for path, _ in leaves)
    loaded.save(tmp_path / "saved")
    assert drafted(load(tmp_path / "saved")) == 0


@pytest.mark.parametrize("depths", [-1, True])
def test_a_prediction_depth_count_below_zero_is_refused(depths):
    from dew.interop import Pretrained

    with pytest.raises(ValueError):
        Pretrained.load(FIXTURES / "glm4-moe-tiny", dtype="float32", prediction_depths=depths)


def test_a_text_only_load_keeps_what_the_wrapper_declares_for_its_language_model(tmp_path):
    """What a wrapper declares at its own scope governs its language model alone: block reading its
    modeling overrides turn on still admits a future block mate, and its end-of-text id still ends
    generation; and a vision tensor stored under two disagreeing names, which the full load refuses,
    is never compared."""

    from safetensors.numpy import save_file

    from dew.interop import Pretrained
    from dew.interop.generation_config import eos_ids

    directory = respelled(FIXTURES / "qwen35-tiny-mm", tmp_path / "source", lambda config: {
        **config, "modeling_overrides": {"attention": {"full_attention": "block_bidirectional",
                                                       "linear_attention": "causal"}},
        "eos_token_id": 3, "text_config": {**config["text_config"], "eos_token_id": None}})
    tensors = load_file(directory / "model.safetensors")
    spelled = next(name for name in tensors if name.startswith("model.visual."))
    tensors[spelled.removeprefix("model.")] = tensors[spelled] + 1
    save_file(tensors, directory / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(ValueError):
        Pretrained.load(directory, dtype="float32", attention_impl="reference")
    text = Pretrained.load(directory, dtype="float32", attention_impl="reference", text_only=True)
    ids = jnp.asarray([[1, 2, 3, 4]], jnp.int32)
    altered, blocked = ids.at[0, 1].set(7), jnp.asarray([[0, 0, -1, -1]], jnp.int32)

    def first(tokens, groups=None):
        return np.asarray(text.model.apply(text.variables, tokens, block_groups=groups))[0, 0]

    np.testing.assert_array_equal(first(ids), first(altered))
    assert np.any(first(ids, blocked) != first(altered, blocked))
    assert eos_ids(text.config, text.generation_config) == (3,)


@pytest.mark.parametrize(("fixture", "dropped"), [
    ("qwen35-native-tiny", "model.visual.patch_embed.proj.weight"),
    ("qwen35-native-tiny", "model.visual.merger.linear_fc1.weight"),
    ("qwen38-dense-tiny", "mtp.layers.0.mlp.down_proj.weight"),
], ids=["tower", "projector", "prediction_depth"])
def test_a_wrapper_missing_a_tensor_of_a_component_it_builds_is_refused(tmp_path, fixture, dropped):
    """A wrapper checkpoint lacking one tensor of its tower, its projector, or a prediction depth it
    holds the rest of is refused at load, rather than loading that component without it."""
    import shutil

    from safetensors.numpy import save_file

    from dew.interop import Pretrained

    target = tmp_path / fixture
    shutil.copytree(FIXTURES / fixture, target)
    tensors = load_file(target / "model.safetensors")
    assert dropped in tensors
    kept = {name: array for name, array in tensors.items() if name != dropped}
    save_file(kept, target / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(ValueError):
        Pretrained.load(target, dtype="float32", attention_impl="reference")
