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
import numpy as np
import pytest
from safetensors.numpy import load_file

from dew.interop.hf_decoders import translate_config, translate_wrapper_config, translate_wrapper_weights
from dew.registry import models, projectors, towers, with_precision

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"


def load_wrapper(name):
    directory = FIXTURES / name
    record = translate_wrapper_config(
        json.loads((directory / "config.json").read_text()))
    variables = translate_wrapper_weights(
        load_file(str(directory / "model.safetensors")), record)
    return directory, record, variables


def test_gemma3_wrapper_translates_to_three_records():
    directory = FIXTURES / "gemma3-tiny-mm"
    record = translate_wrapper_config(
        json.loads((directory / "config.json").read_text()))
    assert record["text_model_type"] == "gemma3_text"
    assert record["text"]["emb_features"] == 64
    assert record["tower"]["class"] == "siglip"
    assert record["tower"]["fields"]["hidden_size"] == 32
    assert record["projector"]["class"] == "gemma"
    assert record["projector"]["fields"]["text_width"] == 64
    assert record["image_token_id"] == 202
    assert record["tokens_per_image"] == 1


def test_llama4_wrapper_translates_to_three_records():
    directory = FIXTURES / "llama4-tiny-mm"
    record = translate_wrapper_config(
        json.loads((directory / "config.json").read_text()))
    assert record["text_model_type"] == "llama4_text"
    assert record["tower"]["class"] == "llama4"
    assert record["projector"]["class"] == "llama4"
    assert record["image_token_id"] == 92
    assert record["tokens_per_image"] == 1


def test_gemma4_wrapper_translates_to_three_records():
    directory = FIXTURES / "gemma4-tiny-mm"
    record = translate_wrapper_config(
        json.loads((directory / "config.json").read_text()))
    assert record["text_model_type"] == "gemma4_text"
    assert record["text"]["emb_features"] == 32
    assert record["tower"]["class"] == "gemma4"
    assert record["tower"]["fields"]["hidden_size"] == 32
    assert record["tower"]["fields"]["pooling_kernel_size"] == 2
    assert record["projector"]["class"] == "gemma4"
    assert record["projector"] == {
        "class": "gemma4", "fields": {"text_width": 32,
        "norm_eps": 1e-06}}
    assert record["image_token_id"] == 60
    # The pooled count follows the image resolution, so the record leaves it
    # open; the fixture's 4x4 patch grid pools to four soft tokens.
    assert record["tokens_per_image"] is None


def test_qwen35_wrapper_translates_to_three_records():
    directory = FIXTURES / "qwen35-tiny-mm"
    record = translate_wrapper_config(
        json.loads((directory / "config.json").read_text()))
    assert record["text_model_type"] == "qwen3_5_text"
    assert record["tower"]["class"] == "qwen3_5"
    assert record["tower"]["fields"]["spatial_merge_size"] == 2
    assert record["tower"]["fields"]["temporal_patch_size"] == 2
    assert record["projector"]["class"] == "qwen3_5"
    assert record["projector"] == {
        "class": "qwen3_5", "fields": {"vision_width": 32, "merge_size": 2, "out_width": 64}}
    assert record["image_token_id"] == 200
    assert record["tokens_per_image"] is None


@pytest.mark.parametrize("name", ["qwen35-tiny-mm", "qwen35-moe-native-tiny"])
def test_a_qwen35_wrapper_repeating_its_text_width_translates_the_same(name):
    """Ornith's Qwen3.5 wrappers state hidden_size at the top level too. The
    wrapper config declares no such field and its model reads the text
    config's, so a repeated value translates as if absent; another value
    is refused by name, since it describes no model the reference builds."""
    config = json.loads((FIXTURES / name / "config.json").read_text())
    width = config["text_config"]["hidden_size"]

    assert translate_wrapper_config({**config, "hidden_size": width}) == translate_wrapper_config(config)
    with pytest.raises(ValueError, match=f"^hidden_size={2 * width} is not expressible"):
        translate_wrapper_config({**config, "hidden_size": 2 * width})


@pytest.mark.network
@pytest.mark.parametrize("repo, revision", [
    ("ornith-ai/Ornith-1.0-9B", "83dc1f5e24ef8527af019a6b3bf66ac0f1c2c999"),
    ("ornith-ai/Ornith-1.0-35B", "5df2ed3f675c7beaa490328cc70bb573b65fb660"),
])
def test_ornith_wrappers_translate_from_their_released_configs(repo, revision):
    """The two most downloaded Ornith wrappers (qwen3_5 and qwen3_5_moe),
    config only: each repeats its text width at the top level."""
    from huggingface_hub import hf_hub_download

    config = json.loads(Path(hf_hub_download(repo, "config.json", revision=revision)).read_text())
    assert config["hidden_size"] == config["text_config"]["hidden_size"]
    record = translate_wrapper_config(config)
    assert record["text_model_type"] == config["text_config"]["model_type"]


def test_the_released_gemma4_wrapper_translates():
    """google/gemma-4-26B-A4B's wrapper, config only: the tower reads 1152
    wide with a pooling kernel of 3 and standardization on, the text half
    translates as gemma4_text, and the count stays open."""
    config = json.loads((FIXTURES / "gemma4-26b-a4b" / "config.json").read_text())
    record = translate_wrapper_config(config)
    assert translate_config(config["text_config"])["emb_features"] == 2816
    assert record["tower"]["fields"]["hidden_size"] == 1152
    assert record["tower"]["fields"]["pooling_kernel_size"] == 3
    assert record["tower"]["fields"]["rope_theta"] == 100.0
    assert record["projector"]["fields"]["text_width"] == 2816
    assert record["tokens_per_image"] is None


def test_the_released_qwen35_wrapper_translates():
    """Qwen/Qwen3.5-0.8B's wrapper, config only: the tower keeps the 0.8B
    checkpoint's qwen3_5 model_type spelling, and the merger width meets the
    decoder width."""
    config = json.loads((FIXTURES / "qwen35-0.8b" / "config.json").read_text())
    record = translate_wrapper_config(config)
    assert record["tower"]["fields"]["hidden_size"] == 768
    assert record["projector"] == {
        "class": "qwen3_5", "fields": {"vision_width": 768, "merge_size": 2,
        "out_width": 1024}}
    assert record["image_token_id"] == 248056


def test_the_released_scout_wrapper_translates():
    """meta-llama/Llama-4-Scout-17B-16E's wrapper, config only: the flat rope
    theta its vision config carries reads, and the soft-token count derives
    from the 24x24 patch grid shuffled by a half."""
    config = json.loads((FIXTURES / "llama-4-scout" / "config.json").read_text())
    record = translate_wrapper_config(config)
    assert record["tower"]["fields"]["rope_theta"] == 10000.0
    assert record["tower"]["fields"]["image_size"] == 336
    assert record["tokens_per_image"] == 144
    assert record["projector"] == {
        "class": "llama4", "fields": {"text_width": 5120}}


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
        projector = projectors.from_record(record["projector"]).build()
        assert np.max(np.abs(np.asarray(
            projector.apply(variables["projector"], tower_ref))
            - projector_ref)) < tolerance


def test_only_the_untied_wrappers_get_an_lm_head_leaf():
    """Gemma ties its head to the embedding table and Llama 4 and Qwen 3.5 do
    not, so the map writes an lm_head leaf for exactly the untied families.
    The four parity cases below score these same trees."""
    for name, tied in (("gemma3-tiny-mm", True), ("llama4-tiny-mm", False),
                       ("gemma4-tiny-mm", True), ("qwen35-tiny-mm", False)):
        _, _, variables = load_wrapper(name)
        leaves, _ = jax.tree_util.tree_flatten_with_path(variables["language_model"])
        names = {".".join(str(entry.key) for entry in path) for path, _ in leaves}
        assert ("params.lm_head.kernel" in names) is not tied, name


def test_a_wrapper_tensor_outside_the_three_prefixes_is_refused():
    """Audio and tile metadata ride no prefix this map reads, so they name
    themselves."""
    directory, record, _ = load_wrapper("gemma3-tiny-mm")
    tensors = dict(load_file(str(directory / "model.safetensors")))
    tensors["model.audio_tower.layers.0.weight"] = np.zeros((2, 2), np.float32)
    with pytest.raises(ValueError, match="unknown tensor name"):
        translate_wrapper_weights(tensors, record)


def test_a_wrapper_without_an_image_token_is_refused():
    config = json.loads((FIXTURES / "gemma3-tiny-mm" / "config.json").read_text())
    del config["image_token_index"]
    with pytest.raises(ValueError, match="image_token_id"):
        translate_wrapper_config(config)


def _multimodal_logits(name, image_id, shift=0):
    """Pixels through the translated tower and projector, merged at the image
    marks, into the translated decoder through the input-embeddings hook."""
    directory = FIXTURES / name
    record = translate_wrapper_config(
        json.loads((directory / "config.json").read_text()))
    variables = translate_wrapper_weights(
        load_file(str(directory / "model.safetensors")), record)
    pixels = wrapper_pixels(directory, record)
    ids = np.load(directory / "input_ids.npy")
    tower = towers.from_record(record["tower"]).build()
    features = tower.apply(variables["tower"], pixels)
    projector = projectors.from_record(record["projector"]).build()
    soft = projector.apply(variables["projector"], features)
    length = ids.shape[1]
    positions = (np.stack([np.where(row == image_id)[0] for row in ids]) + shift) % length
    positions = positions.astype(np.int32)
    fields = dict(record["text"])
    if name.startswith("gemma3"):
        # The reference conditional applies no logit cap (only its causal-LM
        # head path does), so the decoder builds without the text record's.
        fields["final_logit_softcap"] = None
    model = models.build("causal_transformer", **with_precision(
        "causal_transformer", fields, dtype="float32", attention_impl="reference"))
    language = variables["language_model"]
    embedded = model.apply(language, ids, method=lambda decoder, ids: decoder.scaled_embeddings(
        decoder.token_embeddings(ids)))
    embedded = embedded.at[np.arange(len(ids))[:, None], positions].set(soft.astype(embedded.dtype))
    return (np.asarray(model.apply(language, ids, input_embeddings=embedded)),
            np.load(directory / "wrapper_ref.npy"), positions)


@pytest.mark.parametrize("name, image_id", [
    ("gemma3-tiny-mm", 202),  # observed 2.4e-06
    ("llama4-tiny-mm", 92),  # observed 4.1e-06
    ("gemma4-tiny-mm", 60),  # observed 1.4e-05
    # Observed 2.4e-05. The wrapper reference runs on pre-merged embeddings,
    # since the image-grid positions it would otherwise use are outside what
    # the decoder models.
    ("qwen35-tiny-mm", 200),
])
def test_a_multimodal_forward_matches_the_wrapper(name, image_id):
    """fp32 end-to-end parity on a tiny wrapper: tolerance 1e-4, the observed
    max |logit difference| beside each family, with identical argmax."""
    logits, reference, _ = _multimodal_logits(name, image_id)
    assert np.max(np.abs(logits - reference)) < 1e-4
    assert (logits.argmax(-1) == reference.argmax(-1)).all()


@pytest.mark.parametrize("name, image_id", [
    ("gemma3-tiny-mm", 202), ("gemma4-tiny-mm", 60), ("qwen35-tiny-mm", 200)])
def test_soft_tokens_at_shifted_positions_miss_the_wrapper(name, image_id):
    """Soft tokens at shifted positions miss by more than 1.0, so the
    placement is live."""
    misplaced, reference, _ = _multimodal_logits(name, image_id, shift=1)
    assert np.max(np.abs(misplaced - reference)) > 1.0
