"""The encoder-free wrapper, its processor layout and its ordinary-forward mask policy."""

import copy
import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import torch
from reference_error import assert_as_exact
from transformers import Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration
from transformers.conversion_mapping import get_checkpoint_conversion_mapping

from dew.interop import Pretrained
from tools.diffusers_wan_reference import float64
from tools.multimodal_reference import _row_slices, unified_draw
from tools.torch_precision import default_dtype

FIXTURE = Path(__file__).parent / "fixtures" / "hf" / "gemma4-unified-native-tiny"


def seed_count(site):
    """The count measured by tools/qualify_gemma4_unified.py for this exact reference workload."""
    return json.loads((FIXTURE / "qualification" / f"{site}.json").read_text())["seeds"]


def source_pair(directory, dtype="float32"):
    native = Pretrained.load(directory, dtype=dtype, param_dtype=dtype, attention_impl="reference")
    reference = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        directory, dtype=getattr(torch, dtype), attn_implementation="eager").eval()
    with float64(), default_dtype(torch.float64):
        truth = Gemma4UnifiedForConditionalGeneration(copy.deepcopy(reference.config)).double().eval()
    truth.load_state_dict(reference.state_dict(), strict=True)
    return native, reference, truth


@pytest.fixture(scope="module")
def source():
    return source_pair(FIXTURE)


def forward_functions(source):
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
            return {f"row{row}": model(**single, use_cache=False).logits.to(
                torch.float64 if wide else torch.float32).numpy()
                    for row, single in enumerate(_row_slices(tensors, model))}

    def oracle(inputs):
        with float64():
            return theirs(truth, inputs, wide=True)

    return ours, lambda inputs: theirs(reference, inputs), oracle


def assert_forward(source, draw, label, site):
    ours, reference, truth = forward_functions(source)
    assert_as_exact(ours, reference, lambda seed: (draw(seed),),
                    label, seeds=seed_count(site), truth_fn=truth)


def test_unified_processor_and_native_forward_follow_the_published_forward(source):
    assert_forward(source, lambda seed: unified_draw(source[1].config, seed),
                    "Gemma4 Unified source forward", "forward")


def test_vision_blocks_longer_than_the_window_keep_their_whole_block(source):
    assert_forward(source, lambda seed: unified_draw(source[1].config, seed, side=4),
                    "Gemma4 Unified long vision block", "long")


def test_a_bfloat16_unified_source_uses_the_same_media_and_decoder_policy():
    source = source_pair(FIXTURE, "bfloat16")
    assert_forward(source, lambda seed: unified_draw(source[1].config, seed),
                    "Gemma4 Unified bfloat16 source forward", "bfloat16")


def gradient_functions(source):
    """Every trainable source parameter and every floating media input, in the native input layout."""
    native, reference, truth = source
    parameters = dict(reference.named_parameters())
    conversions = get_checkpoint_conversion_mapping(reference.config.model_type)
    bindings = {}
    for layout in native.weight_layouts:
        source_name = layout.name
        runtime_name = source_name
        for conversion in conversions:
            renamed, matched = conversion.rename_source_key(source_name)
            if matched is not None:
                runtime_name = renamed
                break
        if runtime_name in parameters:
            bindings[runtime_name] = (source_name, layout)
    missing = parameters.keys() - bindings.keys()
    if missing:
        raise ValueError(f"unmapped trainable reference parameters: {sorted(missing)}")

    def collect(rows):
        return {name: (np.stack([row[name] for row in rows]) if name.startswith("parameter.")
                       else np.concatenate([row[name] for row in rows], axis=0)) for name in rows[0]}

    @jax.jit
    def native_observations(batch, probe):
        floating = {name: value for name, value in batch.conditioning.items()
                    if jnp.issubdtype(value.dtype, jnp.floating)}

        def loss(params, media):
            prepared = batch.replace(conditioning={**batch.conditioning, **media})
            logits = native.model.apply({**native.variables, "params": params},
                                         prepared.tokens, **prepared.kwargs())
            return jnp.sum(logits * probe), logits

        (_, logits), (gradient, media) = jax.value_and_grad(loss, (0, 1), has_aux=True)(
            native.variables["params"], floating)
        return logits, gradient, media

    def ours(inputs, probe):
        batch = native.processor.from_hf(inputs)
        observed = []
        for row, mask in enumerate(inputs["attention_mask"]):
            single = batch.take_rows(jnp.asarray([row])).slice_tokens(0, int(mask.sum()))
            logits, gradient, media = native_observations(single, probe[row, mask][None])
            values = {f"parameter.{name}": layout.export({"params": gradient})
                       for name, layout in bindings.values()}
            # Real token rows are the forward contract; parameter Jacobians retain each independent
            # example as an axis instead of judging each tiny example as a separate stochastic site.
            values["logits"] = np.asarray(logits)[0]
            values.update({f"input.{name}": value for name, value in media.items()})
            observed.append(values)
        return collect(observed)

    def theirs(model, inputs, probe, *, wide=False):
        tensors = {name: torch.tensor(value, dtype=torch.float64 if wide and np.issubdtype(
            value.dtype, np.floating) else None) for name, value in inputs.items()}
        media = {name: value.requires_grad_(requires_grad=True) for name, value in tensors.items()
                 if value.is_floating_point()}
        params = dict(model.named_parameters())
        observed = []
        singles = _row_slices(tensors, model)
        for row, single in enumerate(singles):
            logits = model(**single, use_cache=False).logits
            cotangent = torch.tensor(probe[row, inputs["attention_mask"][row]][None], dtype=logits.dtype)
            gradients = torch.autograd.grad(logits, (*params.values(), *media.values()),
                                            grad_outputs=cotangent, allow_unused=True,
                                            retain_graph=row + 1 < len(singles))
            values = {"logits": logits.detach().numpy()[0]}
            for name, value in zip(params, gradients[:len(params)], strict=True):
                if value is None:
                    raise ValueError(f"unreached trainable reference parameter: {name}")
                values[f"parameter.{bindings[name][0]}"] = value.detach().numpy()
            raw = dict(inputs)
            for (name, original), value in zip(media.items(), gradients[len(params):], strict=True):
                raw[name] = torch.zeros_like(original).numpy() if value is None else value.detach().numpy()
            prepared = native.processor.from_hf(raw).take_rows(jnp.asarray([row]))
            values.update({f"input.{name}": value for name, value in prepared.conditioning.items()
                           if np.issubdtype(value.dtype, np.floating)})
            observed.append(values)
        return collect(observed)

    def oracle(inputs, probe):
        with float64(), jax.enable_x64(new_val=True):
            return theirs(truth, inputs, probe, wide=True)

    def draw(seed):
        """Each draw sums more independent rows so one row's conditioning holds less of its error.
        Cotangents use their own tagged stream; every per-example VJP remains in the result."""
        pairs = [unified_draw(reference.config, 16 * seed + index) for index in range(16)]
        inputs = {name: np.concatenate([pair[name] for pair in pairs]) for name in pairs[0]}
        probe = np.random.default_rng([0xBACC, seed]).standard_normal(
            (*inputs["input_ids"].shape, reference.config.text_config.vocab_size), np.float32)
        return inputs, probe

    return ours, lambda *args: theirs(reference, *args), oracle, draw


def test_raw_media_train_and_reload_through_the_public_source(source, tmp_path):
    native, _, _ = source
    rng = np.random.default_rng(52)
    image = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    audio = rng.standard_normal(35).astype(np.float32)
    batch = native.processor("token7 <image_soft_token> token9 <audio> token10",
                             images=[image], audio=[audio])

    @jax.jit
    def step(params):
        def loss(values):
            logits = native.model.apply({**native.variables, "params": values}, batch.tokens,
                                         **batch.kwargs())
            return jnp.mean(jnp.square(logits))

        return jax.value_and_grad(loss)(params)

    value, gradient = step(native.variables["params"])
    optimizer = optax.sgd(1e-3)
    updates, _ = optimizer.update(gradient, optimizer.init(native.variables["params"]),
                                   native.variables["params"])
    params = optax.apply_updates(native.variables["params"], updates)
    assert np.isfinite(value)
    assert any(not np.array_equal(before, after) for before, after in zip(
        jax.tree.leaves(native.variables["params"]), jax.tree.leaves(params), strict=True))
    variables = {**native.variables, "params": params}
    expected = native.model.apply(variables, batch.tokens, **batch.kwargs())
    native.save(tmp_path, variables=variables, max_shard_size=4096)
    restored = Pretrained.load(tmp_path, dtype="float32", attention_impl="reference")
    for got, wanted in zip(jax.tree.leaves(restored.variables), jax.tree.leaves(variables), strict=True):
        np.testing.assert_array_equal(got, wanted)
    np.testing.assert_array_equal(restored.model.apply(restored.variables, batch.tokens, **batch.kwargs()),
                                  expected)


def cached_functions(source):
    native, reference, truth = source
    model = native.model.clone(language_model=native.model.language_model.clone(max_seq_len=32))

    @jax.jit
    def ours_cached(batch, continuation):
        cache = model.apply(native.variables, batch.tokens.shape[0],
                            method="init_cache", mutable=["cache"])[1]["cache"]
        prefill, changed = model.apply({**native.variables, "cache": cache}, batch.tokens,
                                       **batch.kwargs(), decode=True, mutable=["cache"])
        outputs = {"prefill": prefill}
        for step in range(continuation.shape[1]):
            value, changed = model.apply({**native.variables, "cache": changed["cache"]},
                                          continuation[:, step:step + 1], decode=True, mutable=["cache"])
            outputs[f"step{step}"] = value
        return outputs

    def ours(inputs, continuation):
        rows = jnp.arange(1, inputs["input_ids"].shape[0], 2)
        batch = native.processor.from_hf(inputs).take_rows(rows).slice_tokens(0, 10)
        return ours_cached(batch, continuation)

    def theirs(model, inputs, continuation, *, wide=False):
        tensors = {name: torch.tensor(value, dtype=torch.float64 if wide and np.issubdtype(
            value.dtype, np.floating) else None) for name, value in inputs.items()}
        observations = []
        with torch.no_grad():
            for row, single in enumerate(_row_slices(tensors, model)[1::2]):
                result = model(**single, use_cache=True)
                outputs = {"prefill": result.logits.numpy()}
                for step in range(continuation.shape[1]):
                    result = model(input_ids=torch.from_numpy(continuation[row:row + 1, step:step + 1]),
                                   attention_mask=torch.ones((1, single["input_ids"].shape[1] + step + 1),
                                                             dtype=torch.bool),
                                   past_key_values=result.past_key_values, use_cache=True)
                    outputs[f"step{step}"] = result.logits.numpy()
                observations.append(outputs)
        return {name: np.concatenate([row[name] for row in observations]) for name in observations[0]}

    def oracle(inputs, continuation):
        with float64():
            return theirs(truth, inputs, continuation, wide=True)

    def draw(seed):
        """Four independent rows retain every cached output while diluting one row's conditioning.
        Continuations use a separate tagged stream from all media draws."""
        pairs = [unified_draw(reference.config, 4 * seed + index) for index in range(4)]
        inputs = {name: np.concatenate([pair[name] for pair in pairs]) for name in pairs[0]}
        continuation = np.random.default_rng([0xCAC4, seed]).integers(4, 50, (4, 3), dtype=np.int32)
        return inputs, continuation

    return ours, lambda *args: theirs(reference, *args), oracle, draw


@pytest.mark.parametrize(("site", "functions"), [("backward", gradient_functions),
                                                 ("cached", cached_functions)])
def test_unified_runtime_observations_follow_the_reference(source, site, functions):
    ours, reference, truth, draw = functions(source)
    assert_as_exact(ours, reference, draw,
                    f"Gemma4 Unified {site}", seeds=seed_count(site), truth_fn=truth)


def optional_source(source, directory, media):
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
    reference.save_pretrained(directory, safe_serialization=True)
    reference.config.to_json_file(directory / "config.json", use_diff=False)
    for name in ("processor_config.json", "tokenizer_config.json", "tokenizer.json"):
        shutil.copyfile(FIXTURE / name, directory / name)

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

    return source_pair(directory), draw


@pytest.mark.parametrize("media", ["text", "vision", "audio"])
def test_a_unified_source_can_omit_either_media_component(source, tmp_path, media):
    pair, draw = optional_source(source, tmp_path, media)
    assert_forward(pair, draw, f"Gemma4 Unified {media} source", media)


def equal_geometry_source(source, directory):
    from dew.interop.verify import scatter_weights

    config = source[1].config.to_dict()
    config["text_config"].update(global_head_dim=8, num_global_key_value_heads=2, per_layer_config={})
    config = Gemma4UnifiedConfig.from_dict(config)
    config._attn_implementation = "eager"
    torch.manual_seed(502)
    reference = Gemma4UnifiedForConditionalGeneration(config).float().eval()
    scatter_weights(reference)
    reference.save_pretrained(directory, safe_serialization=True)
    # Preserve the explicitly empty per-layer geometry instead of reconstructing default global heads.
    reference.config.to_json_file(directory / "config.json", use_diff=False)
    for name in ("processor_config.json", "tokenizer_config.json", "tokenizer.json"):
        shutil.copyfile(FIXTURE / name, directory / name)
    return source_pair(directory), lambda seed: unified_draw(config, seed)


def test_equal_attention_geometries_still_use_the_full_layer_vision_mask(source, tmp_path):
    pair, draw = equal_geometry_source(source, tmp_path)
    assert_forward(pair, draw,
                    "Gemma4 Unified equal attention geometries", "equal")
