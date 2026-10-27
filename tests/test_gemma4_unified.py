"""The encoder-free wrapper, its processor layout and its ordinary-forward mask policy."""

import copy
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import torch
from flax.traverse_util import flatten_dict, unflatten_dict
from reference_error import (
    as_given,
    assert_as_exact,
    indexed_recording,
    over_windows,
    recording,
    seeds_for,
    windows,
)
from transformers import (
    AutoModelForImageTextToText,
    Gemma4UnifiedConfig,
    Gemma4UnifiedForConditionalGeneration,
)
from transformers.conversion_mapping import get_checkpoint_conversion_mapping

from dew.interop import Pretrained
from dew.nn.inputs import ModelInputs
from tools.multimodal_reference import _row_slices, unified_draw
from tools.torch_precision import float64

FIXTURE = Path(__file__).parent / "fixtures" / "hf" / "gemma4-unified-native-tiny"


def source_pair(directory, dtype="float32"):
    native = Pretrained.load(directory, dtype=dtype, param_dtype=dtype, attention_impl="reference")
    reference = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        directory, dtype=getattr(torch, dtype), attn_implementation="eager").eval()
    with float64():
        truth = AutoModelForImageTextToText.from_config(
            copy.deepcopy(reference.config), dtype=torch.float64, attn_implementation="eager").eval()
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

    # Each arm runs row by row on the host (its row's length read from its mask), as given.
    @as_given
    def ours(inputs):
        batch = processor.from_hf(inputs)
        result = {}
        for row, mask in enumerate(inputs["attention_mask"]):
            count = int(np.sum(mask))
            single = batch.take_rows(jnp.asarray([row])).slice_tokens(0, count)
            result[f"row{row}"] = forward(single)
        return result

    def theirs(model, inputs, *, wide=False):
        device = next(model.parameters()).device
        tensors = {name: torch.from_numpy(np.asarray(value).copy()).to(device)
                   for name, value in inputs.items()}
        if wide:
            tensors = {name: value.double() if value.is_floating_point() else value
                       for name, value in tensors.items()}
        with torch.no_grad():
            return {f"row{row}": model(**single, use_cache=False).logits.to(
                torch.float64 if wide else torch.float32).cpu().numpy()
                    for row, single in enumerate(_row_slices(tensors, model))}

    @as_given
    def oracle(inputs):
        with float64():
            return theirs(truth, inputs, wide=True)

    return ours, as_given(lambda inputs: theirs(reference, inputs)), oracle


def assert_forward(source, draw, label, site):
    ours, _, _ = forward_functions(source)
    stored = recording(FIXTURE / f"{site}.npz")
    seeds = int(stored[f"{site}.seeds"])
    drawn, reference, truth = indexed_recording(stored, site, seeds, inputs_of=draw)
    assert_as_exact(as_given(lambda index, inputs: ours(inputs)), reference, drawn,
                    label, seeds=seeds, truth_fn=truth)


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


AUDIO_FRAMES = 16
"""Audio frames a clip holds in the backward draw. The gradients of the features, of the audio
projection and of the first layer's key path sum over every valid frame: at the shared draw's five
a pair, each draw's error in them is a few terms, too heavy-tailed for the reference compared with
itself to resolve within 1024 seeds. Sixteen frames a clip, most of them valid, average many."""
BACKWARD_WINDOW = 4
"""Draws a backward seed pools (`windows`). The softmax backward of a draw whose attention rows
saturate gives its gradients many times the median draw's relative error, layer 2's attention first,
so a seed of one draw leaves the reference compared with itself unresolved past seeds_for's ceiling;
four draws a seed resolve every leaf at a count under it, the same draws in all."""
AUDIO_VALID = (12, 10)
"""Each pair's valid frames: the first clip's four holes and the second's padded tail of six. The
counts are fixed so every row keeps one length, and one compiled program, across seeds; where the
first clip's holes fall is drawn per seed."""


def audio_rich(config, pair: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    """`pair` (a `unified_draw`) with each row's audio run replaced by a clip of `AUDIO_FRAMES` frames
    and its `AUDIO_VALID` audio tokens in the run's place. The first clip has holes drawn per seed,
    the second a padded tail, so the masking stays under test."""
    rng = np.random.default_rng([0xA0D1, seed])
    masks = np.zeros((2, AUDIO_FRAMES), bool)
    masks[0, 0] = True
    masks[0, 1 + rng.choice(AUDIO_FRAMES - 1, AUDIO_VALID[0] - 1, replace=False)] = True
    masks[1, :AUDIO_VALID[1]] = True
    width = pair["input_ids"].shape[1] + AUDIO_FRAMES
    tokens = np.full((2, width), config.pad_token_id, np.int32)
    for row in range(2):
        real = pair["input_ids"][row][pair["attention_mask"][row]]
        audio = real == config.audio_token_id
        start = int(np.argmax(audio))
        kept = real[~audio]
        ids = np.concatenate([kept[:start], np.full(masks[row].sum(), config.audio_token_id, np.int32),
                              kept[start:]])
        tokens[row, :len(ids)] = ids
    types = np.where(tokens == config.image_token_id, 1, np.where(
        tokens == config.video_token_id, 2, np.where(tokens == config.audio_token_id, 3, 0))).astype(np.int32)
    return {**pair, "input_ids": tokens, "attention_mask": tokens != config.pad_token_id,
            "mm_token_type_ids": types, "input_features_mask": masks,
            "input_features": rng.standard_normal((2, AUDIO_FRAMES, config.audio_config.audio_embed_dim),
                                                  np.float32)}


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

    def collect(rows, xp):
        return {name: (xp.stack([row[name] for row in rows]) if name.startswith("parameter.")
                       else xp.concatenate([row[name] for row in rows], axis=0)) for name in rows[0]}

    @jax.jit
    def native_observations(batch, probe):
        """One row's observations, the program a user's step on that row runs: its logits, every
        parameter's gradient in its source layout and every floating media input's gradient."""
        floating = {name: value for name, value in batch.conditioning.items()
                    if jnp.issubdtype(value.dtype, jnp.floating)}

        def loss(params, media):
            prepared = batch.replace(conditioning={**batch.conditioning, **media})
            logits = native.model.apply({**native.variables, "params": params},
                                         prepared.tokens, **prepared.kwargs())
            return jnp.sum(logits * probe), logits

        (_, logits), (gradient, media) = jax.value_and_grad(loss, (0, 1), has_aux=True)(
            native.variables["params"], floating)
        values = {f"parameter.{name}": layout.source_array({"params": gradient})
                  for name, layout in bindings.values()}
        # Real token rows are the forward contract; parameter Jacobians retain each independent
        # example as an axis instead of judging each tiny example as a separate stochastic site.
        values["logits"] = logits[0]
        values.update({f"input.{name}": value for name, value in media.items()})
        return values

    @as_given
    def ours(inputs):
        """Each row's program over its own length (`native_observations`), compiled once a length
        rather than once a window's lengths, and the rows' observations collected. The rows' programs
        are independent, so they run side by side on the host's cores, one thread a core; the trace in
        which the gate counts a truth's roundings runs them in turn, a trace being its thread's own."""
        rows = []
        for row in range(inputs["input/input_ids"].shape[0]):
            prefix = f"native/{row}/"
            fields = unflatten_dict({name.removeprefix(prefix): value for name, value in inputs.items()
                                     if name.startswith(prefix)}, sep="/")
            rows.append((ModelInputs(**fields), inputs[f"probe/{row}"]))
        if any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(inputs)):
            return collect([native_observations(*row) for row in rows], jnp)
        with ThreadPoolExecutor(len(os.sched_getaffinity(0))) as pool:
            observed = list(pool.map(lambda row: jax.device_get(native_observations(*row)), rows))
        return collect(observed, np)

    def theirs(model, given, *, wide=False):
        inputs = {name.removeprefix("input/"): value for name, value in given.items()
                  if name.startswith("input/")}
        device = next(model.parameters()).device
        tensors = {name: torch.tensor(value, dtype=torch.float64 if wide and np.issubdtype(
            value.dtype, np.floating) else None, device=device) for name, value in inputs.items()}
        media = {name: value.requires_grad_(requires_grad=True) for name, value in tensors.items()
                 if value.is_floating_point()}
        params = dict(model.named_parameters())
        observed = []
        singles = _row_slices(tensors, model)
        for row, single in enumerate(singles):
            logits = model(**single, use_cache=False).logits
            cotangent = torch.tensor(given[f"probe/{row}"], dtype=logits.dtype, device=device)
            gradients = torch.autograd.grad(logits, (*params.values(), *media.values()),
                                            grad_outputs=cotangent, allow_unused=True,
                                            retain_graph=row + 1 < len(singles))
            values = {"logits": logits.detach().cpu().numpy()[0]}
            for name, value in zip(params, gradients[:len(params)], strict=True):
                if value is None:
                    raise ValueError(f"unreached trainable reference parameter: {name}")
                values[f"parameter.{bindings[name][0]}"] = value.detach().cpu().numpy()
            raw = dict(inputs)
            for (name, original), value in zip(media.items(), gradients[len(params):], strict=True):
                raw[name] = (torch.zeros_like(original).cpu().numpy() if value is None
                             else value.detach().cpu().numpy())
            prepared = native.processor.from_hf(raw).take_rows(jnp.asarray([row]))
            values.update({f"input.{name}": value for name, value in prepared.conditioning.items()
                           if np.issubdtype(value.dtype, np.floating)})
            observed.append(values)
        return collect(observed, np)

    def oracle(inputs):
        with float64(), jax.enable_x64(new_val=True):
            return theirs(truth, inputs, wide=True)

    def draw(seed):
        """Each draw sums more independent rows so one row's conditioning holds less of its error.
        Cotangents use their own tagged stream; every per-example VJP remains in the result. Each
        clip carries `AUDIO_FRAMES` frames (`audio_rich`)."""
        pairs = [audio_rich(reference.config, unified_draw(reference.config, 16 * seed + index),
                            16 * seed + index) for index in range(16)]
        inputs = {name: np.concatenate([pair[name] for pair in pairs]) for name in pairs[0]}
        probe = np.random.default_rng([0xBACC, seed]).standard_normal(
            (*inputs["input_ids"].shape, reference.config.text_config.vocab_size), np.float32)
        # Host arrays, which the rows are sliced from on the host.
        batch = jax.tree.map(np.asarray, native.processor.from_hf(inputs))
        given = {f"input/{name}": value for name, value in inputs.items()}
        for row, mask in enumerate(inputs["attention_mask"]):
            single = batch.take_rows(np.asarray([row])).slice_tokens(0, int(mask.sum()))
            fields = {"tokens": single.tokens, "token_fields": dict(single.token_fields),
                      "conditioning": dict(single.conditioning)}
            given.update({f"native/{row}/{name}": np.asarray(value)
                          for name, value in flatten_dict(fields, sep="/").items()})
            given[f"probe/{row}"] = probe[row, mask][None]
        return (given,)

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
        device = next(model.parameters()).device
        tensors = {name: torch.tensor(value, dtype=torch.float64 if wide and np.issubdtype(
            value.dtype, np.floating) else None, device=device) for name, value in inputs.items()}
        observations = []
        with torch.no_grad():
            for row, single in enumerate(_row_slices(tensors, model)[1::2]):
                result = model(**single, use_cache=True)
                outputs = {"prefill": result.logits.cpu().numpy()}
                for step in range(continuation.shape[1]):
                    result = model(input_ids=torch.as_tensor(
                                        continuation[row:row + 1, step:step + 1], device=device),
                                   attention_mask=torch.ones((1, single["input_ids"].shape[1] + step + 1),
                                                             dtype=torch.bool, device=device),
                                   past_key_values=result.past_key_values, use_cache=True)
                    outputs[f"step{step}"] = result.logits.cpu().numpy()
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
    if site == "backward":
        stored = recording(FIXTURE / "backward.npz")
        with jax.enable_x64(new_val=True):
            native = Pretrained.load(FIXTURE, dtype="float64", param_dtype="float64",
                                      attention_impl="reference")
            native_truth = gradient_functions((native, source[1], source[2]))[0]
        seeds = int(stored["backward.seeds"])
        recorded_draw, reference, truth = indexed_recording(
            stored, "backward", seeds, inputs_of=windows(lambda seed: draw(seed)[0], BACKWARD_WINDOW),
            truth_fn=over_windows(native_truth))
        assert_as_exact(as_given(lambda index, inputs: over_windows(ours)(inputs)), reference, recorded_draw,
                        "Gemma4 Unified recorded backward", seeds=seeds, truth_fn=truth)
        return

    def inputs_of(seed):
        inputs, continuation = draw(seed)
        return {**inputs, "continuation": continuation}

    stored = recording(FIXTURE / f"{site}.npz")
    seeds = int(stored[f"{site}.seeds"])
    recorded_draw, reference, truth = indexed_recording(stored, site, seeds, inputs_of=inputs_of)
    assert_as_exact(as_given(lambda index, inputs: ours(
        {name: value for name, value in inputs.items() if name != "continuation"}, inputs["continuation"])),
        reference, recorded_draw, f"Gemma4 Unified {site}", seeds=seeds, truth_fn=truth)


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


def assert_live_forward(pair, draw, label):
    """Check 1 of Dew's forward against transformers' on `draw`'s inputs, the reference compared with
    itself sizing the site's seeds (`seeds_for`)."""
    ours, reference, truth = forward_functions(pair)
    seeds = seeds_for(reference, draw, label, truth_fn=truth)
    assert_as_exact(ours, reference, draw, label, seeds=seeds, truth_fn=truth)


def test_vision_without_its_token_types_runs_causal_as_transformers_does(source):
    """Pixels whose call carries no mm_token_type_ids carry no vision blocks: every vision token reads
    causally, as transformers' own forward computes them without the field (its block ids stay -1,
    modeling_gemma4_unified.py:1097-1101 at 5.17.0). Dew computes what transformers computes on the same
    untyped inputs."""
    def draw(seed):
        values = unified_draw(source[1].config, seed)
        del values["mm_token_type_ids"]
        return (values,)

    assert_live_forward(source, draw, "Gemma4 Unified untyped forward")


def test_a_softcapped_unified_release_caps_its_logits_as_transformers_does(source, tmp_path):
    """The released 12B caps its final logits at 30 (final_logit_softcapping); the fixture with that one
    field set computes transformers' capped logits."""
    original = source[1]
    config = copy.deepcopy(original.config)
    config.text_config.final_logit_softcapping = 30.0
    reference = Gemma4UnifiedForConditionalGeneration(config).float().eval()
    reference.load_state_dict(original.state_dict(), strict=True)
    reference.save_pretrained(tmp_path, safe_serialization=True)
    reference.config.to_json_file(tmp_path / "config.json", use_diff=False)
    for name in ("processor_config.json", "tokenizer_config.json", "tokenizer.json"):
        shutil.copyfile(FIXTURE / name, tmp_path / name)
    pair = source_pair(tmp_path)
    assert_live_forward(pair, lambda seed: (unified_draw(original.config, seed),),
                        "Gemma4 Unified softcapped forward")


def test_a_unified_config_without_its_image_placeholder_is_refused():
    """A wrapper config names the token its images stand in for; one that leaves image_token_id out is
    refused, as Gemma 4's is, rather than read with a guessed id."""
    import json

    from dew.interop.hf_decoders import translate_wrapper_config

    config = json.loads((FIXTURE / "config.json").read_text())
    assert translate_wrapper_config(config)["image_token_id"] == config["image_token_id"]
    del config["image_token_id"]
    with pytest.raises(ValueError):
        translate_wrapper_config(config)


def test_a_unified_config_without_its_audio_placeholder_takes_the_config_class_default():
    """Gemma4UnifiedConfig defaults audio_token_id (258881, configuration_gemma4_unified.py:312 at
    5.17.0), so a config that leaves it out names that placeholder: its audio reads it and the decoder
    keeps it a placeholder, as transformers' config class would."""
    import json

    from dew.interop.hf_decoders import translate_wrapper_config

    config = json.loads((FIXTURE / "config.json").read_text())
    stated = translate_wrapper_config(config)
    assert stated["audio_token_id"] == config["audio_token_id"] in stated["extra_placeholder_ids"]
    del config["audio_token_id"]
    defaulted = translate_wrapper_config(config)
    assert defaulted["audio_token_id"] == 258881 in defaulted["extra_placeholder_ids"]
    assert defaulted["audio_projector"] == stated["audio_projector"]


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
