"""The encoder-free wrapper, its processor layout and its ordinary-forward mask policy."""

import copy
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from reference_error import assert_as_exact
from transformers import Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration

from dew.interop import Pretrained
from tools.diffusers_wan_reference import float64
from tools.multimodal_reference import _row_slices, unified_draw
from tools.torch_precision import default_dtype

FIXTURE = Path(__file__).parent / "fixtures" / "hf" / "gemma4-unified-native-tiny"


def source_pair(directory):
    native = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    reference = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        directory, dtype=torch.float32, attn_implementation="eager").eval()
    with float64(), default_dtype(torch.float64):
        truth = Gemma4UnifiedForConditionalGeneration(copy.deepcopy(reference.config)).double().eval()
    truth.load_state_dict(reference.state_dict(), strict=True)
    return native, reference, truth


@pytest.fixture(scope="module")
def source():
    return source_pair(FIXTURE)


def assert_forward(source, draw, label):
    native, reference, truth = source
    processor = native.processor

    @jax.jit
    def forward(batch):
        return native.model.apply(native.variables, batch.tokens, **batch.kwargs())

    def ours(inputs):
        batch = processor.from_hf(inputs)
        result = {}
        for row, mask in enumerate(inputs["attention_mask"]):
            count = int(np.sum(mask))
            single = batch.take_rows(jnp.asarray([row])).slice_tokens(0, count)
            result[f"row{row}"] = forward(single)
        return result

    def theirs(model, inputs, *, wide=False):
        tensors = {name: torch.from_numpy(np.asarray(value).copy()) for name, value in inputs.items()}
        if wide:
            tensors = {name: value.double() if value.is_floating_point() else value
                       for name, value in tensors.items()}
        with torch.no_grad():
            return {f"row{row}": model(**single, use_cache=False).logits.numpy()
                    for row, single in enumerate(_row_slices(tensors, model))}

    def oracle(inputs):
        with float64():
            return theirs(truth, inputs, wide=True)

    assert_as_exact(ours, lambda inputs: theirs(reference, inputs), lambda seed: (draw(seed),),
                    label, seeds=8, truth_fn=oracle)


def test_unified_processor_and_native_forward_follow_the_published_forward(source):
    assert_forward(source, lambda seed: unified_draw(source[1].config, seed),
                    "Gemma4 Unified source forward")


@pytest.mark.parametrize("media", ["text", "vision", "audio"])
def test_a_unified_source_can_omit_either_media_component(source, tmp_path, media):
    original = source[1]
    config = copy.deepcopy(original.config)
    if media != "vision":
        config.vision_config = None
    if media != "audio":
        config.audio_config = None
    reference = Gemma4UnifiedForConditionalGeneration(config).float().eval()
    names = reference.state_dict().keys()
    reference.load_state_dict({name: value for name, value in original.state_dict().items() if name in names},
                               strict=True)
    reference.save_pretrained(tmp_path, safe_serialization=True)
    reference.config.to_json_file(tmp_path / "config.json", use_diff=False)
    for name in ("processor_config.json", "tokenizer_config.json", "tokenizer.json"):
        shutil.copyfile(FIXTURE / name, tmp_path / name)

    def draw(seed):
        values = unified_draw(original.config, seed)
        if config.vision_config is None:
            vision = (values["input_ids"] == config.image_token_id) | (
                values["input_ids"] == config.video_token_id)
            values["input_ids"][vision] = 7
            values["mm_token_type_ids"][vision] = 0
            for name in ("pixel_values", "image_position_ids", "pixel_values_videos", "video_position_ids"):
                del values[name]
        if config.audio_config is None:
            audio = values["input_ids"] == config.audio_token_id
            values["input_ids"][audio] = 7
            values["mm_token_type_ids"][audio] = 0
            del values["input_features"], values["input_features_mask"]
        return values

    assert_forward(source_pair(tmp_path), draw, f"Gemma4 Unified {media} source")


def test_equal_attention_geometries_still_use_the_full_layer_vision_mask(source, tmp_path):
    from dew.interop.verify import scatter_weights

    config = source[1].config.to_dict()
    config["text_config"].update(global_head_dim=8, num_global_key_value_heads=2, per_layer_config=None)
    config = Gemma4UnifiedConfig.from_dict(config)
    config._attn_implementation = "eager"
    torch.manual_seed(502)
    reference = Gemma4UnifiedForConditionalGeneration(config).float().eval()
    scatter_weights(reference)
    reference.save_pretrained(tmp_path, safe_serialization=True)
    # An explicit null per_layer_config means no override. A diff-only config omits it and
    # reconstructs the reference's default 512-wide global heads on reload.
    reference.config.to_json_file(tmp_path / "config.json", use_diff=False)
    for name in ("processor_config.json", "tokenizer_config.json", "tokenizer.json"):
        shutil.copyfile(FIXTURE / name, tmp_path / name)
    assert_forward(source_pair(tmp_path), lambda seed: unified_draw(config, seed),
                    "Gemma4 Unified equal attention geometries")
