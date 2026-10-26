"""Native processor, model, objective and checkpoint parity on tiny sources.

The reference is the actual Transformers 5.16.1 processor and conditional
model per family, written by tools/multimodal_reference.py without downloads
one row of the batch encoding at a time. Gemma3 carries the Trainer and
mutation coverage; Gemma4, Qwen3.5, Llama4 and Gemma3n (images and audio)
run the same processor, forward, backward, export and cached generation
path, and the two audio-only fixtures pin waveform conditioning. Unequal
image counts and left padding exercise per-row conditioning and cache
addressing.
"""

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from safetensors.numpy import load_file

from dew.batch import coordinates
from dew.data.dataset import Loader
from dew.interop.pretrained import Pretrained
from dew.nn.inputs import ModelInputs
from dew.nn.mixers.attention import AttentionMixer
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.sampling.text import Sampling
from dew.training import Layout, MeshSpec, Trainer

FIXTURE = Path(__file__).parent / "fixtures" / "hf" / "gemma3-native-tiny"


@pytest.fixture(scope="module")
def source():
    loaded = Pretrained.load(FIXTURE, dtype="float32", attention_impl="reference")
    images = np.load(FIXTURE / "raw_images.npy")
    prompts = json.loads((FIXTURE / "prompts.json").read_text())
    assert loaded.processor is not None
    inputs = loaded.processor(prompts, images=[[images[0]], [images[1], images[2]]])
    return loaded, inputs


def test_public_processor_and_native_forward_match_conditional_model(source):
    """Valid-slot fp32 logit error 5.96e-6; tolerance 1e-4, identical argmax."""
    loaded, inputs = source
    np.testing.assert_array_equal(inputs.tokens, np.load(FIXTURE / "input_ids.npy"))
    model = loaded.model
    output = jax.jit(lambda variables: model.apply(variables, inputs.tokens, **inputs.kwargs()))(
        loaded.variables
    )
    valid = np.asarray(inputs.token_fields["attention_mask"])
    reference = np.load(FIXTURE / "logits.npy")
    np.testing.assert_allclose(np.asarray(output)[valid], reference[valid], atol=1e-4, rtol=0)
    np.testing.assert_array_equal(np.asarray(output)[valid].argmax(-1), reference[valid].argmax(-1))
    from reference_error import assert_moves, widened

    from dew.registry import ModelConfig

    language = model.language_model
    ordinary = language.mixer or AttentionMixer()
    kinds = {name: dataclasses.replace(kind, mixer=dataclasses.replace(
        kind.mixer or ordinary, bidirectional_images=False)) for name, kind in language.kinds.items()}
    causal_images = model.clone(language_model=language.clone(
        mixer=dataclasses.replace(ordinary, bidirectional_images=False), kinds=kinds))
    exact = ModelConfig.from_model(model).with_dtype("float64").build()

    def forward(module, variables, pixels):
        prepared = inputs.replace(conditioning={**inputs.conditioning, "pixel_values": pixels})
        return module.apply(variables, prepared.tokens, **prepared.kwargs())[valid]

    def draw(seed):
        pixels = np.asarray(inputs.conditioning["pixel_values"])
        return (pixels + np.random.default_rng(seed).standard_normal(pixels.shape).astype(np.float32) * 0.01,)

    assert_moves(jax.jit(lambda pixels: forward(model, loaded.variables, pixels)),
                 jax.jit(lambda pixels: forward(causal_images, loaded.variables, pixels)), draw,
                 "Gemma3 image visibility", seeds=8,
                 truth_fn=lambda pixels: forward(exact, widened(loaded.variables), pixels))


def test_objective_pixel_gradient_matches_reference(source):
    """Pixel-gradient max error 2.05e-7 and CE error 4.77e-7 against Torch.

    The gradient crosses image encoder, projection, decoder and shifted loss.
    """
    loaded, inputs = source
    objective = LMObjective(loaded, inputs.tokens.shape[1] - 1,
                            ema_decay=None, pad_id=0)
    step = Step(step=jnp.int32(0), key=jax.random.key(4), ema=None)

    def loss(pixels):
        prepared = dataclasses.replace(inputs, conditioning={**inputs.conditioning, "pixel_values": pixels})
        statistics, _ = objective.loss(loaded.variables, {"text": prepared}, step)
        return objective.reduce_loss(statistics)[0]

    value, gradient = jax.jit(jax.value_and_grad(loss))(inputs.conditioning["pixel_values"])
    reference = json.loads((FIXTURE / "training.json").read_text())
    np.testing.assert_allclose(value, reference["loss"], atol=1e-5, rtol=0)
    real_images = jnp.stack([gradient[0, 0], gradient[1, 0], gradient[1, 1]])
    np.testing.assert_allclose(real_images, np.load(FIXTURE / "pixel_gradient.npy"), atol=1e-5, rtol=1e-4)
    np.testing.assert_array_equal(gradient[0, 1], jnp.zeros_like(gradient[0, 1]))


def test_trainer_update_exports_and_reloads_the_complete_model(source, tmp_path):
    """One real Trainer update agrees with the reference's all-parameter SGD."""
    loaded, inputs = source
    reference = json.loads((FIXTURE / "training.json").read_text())
    objective = LMObjective(loaded, inputs.tokens.shape[1] - 1,
                            ema_decay=None, pad_id=0)
    rows = 2 * jax.device_count()
    training_inputs = inputs.take_rows(jnp.arange(rows) % 2)
    data = Loader(
        train=lambda partition: iter([{"text": training_inputs}]), val=None, records=rows, batch=rows
    )
    trainer = Trainer(objective, optax.sgd(reference["learning_rate"]), key=jax.random.key(3),
                      mesh=MeshSpec(), layout=Layout(min_shard=2**30))
    state = trainer.fit(data, steps=1, log_every=1)
    output = loaded.model.apply(state.variables, inputs.tokens, **inputs.kwargs())
    valid = np.asarray(inputs.token_fields["attention_mask"])
    expected = np.load(FIXTURE / "updated_logits.npy")
    np.testing.assert_allclose(np.asarray(output)[valid], expected[valid], atol=1e-4, rtol=0)
    before = np.load(FIXTURE / "logits.npy")
    assert np.max(np.abs(np.asarray(output)[valid] - before[valid])) > 1e-3
    loaded.save(tmp_path, variables=state.variables)
    reloaded = Pretrained.load(tmp_path, dtype="float32", attention_impl="reference")
    for actual, wanted in zip(jax.tree.leaves(reloaded.variables), jax.tree.leaves(state.variables),
                              strict=True):
        np.testing.assert_array_equal(actual, wanted)
    restored = reloaded.model.apply(reloaded.variables, inputs.tokens, **inputs.kwargs())
    np.testing.assert_allclose(np.asarray(restored)[valid], expected[valid], atol=1e-4, rtol=0)
    assert reloaded.processor is not None
    assert loaded.processor is not None
    assert reloaded.processor.decode(inputs.tokens) == loaded.processor.decode(inputs.tokens)


def test_model_inputs_alignment_preserves_media_identity(source):
    _loaded, inputs = source
    padding = (~inputs.token_fields["attention_mask"]).sum(axis=1)
    aligned = jax.jit(lambda value: value.align_left(padding).take_rows(jnp.array([1, 0])))(inputs)
    np.testing.assert_array_equal(
        aligned.conditioning["pixel_values"], inputs.conditioning["pixel_values"][jnp.array([1, 0])]
    )
    for row, old_row in enumerate((1, 0)):
        start = int(padding[old_row])
        np.testing.assert_array_equal(
            aligned.token_fields["image_indices"][row, : inputs.tokens.shape[1] - start],
            inputs.token_fields["image_indices"][old_row, start:],
        )
    assert isinstance(aligned, ModelInputs)


@pytest.fixture(scope='module')
def gemma4_video_batch():
    from transformers import AutoProcessor
    from transformers.video_utils import VideoMetadata

    loaded = Pretrained.load(
        FIXTURE.parent / "gemma4-native-tiny", dtype="float32", attention_impl="reference"
    )
    native = loaded.processor
    if native is None:
        raise ValueError('the Gemma4 fixture requires its saved Processor')
    processor = AutoProcessor.from_pretrained(
        str(FIXTURE.parent / "gemma4-native-tiny"), local_files_only=True)
    vision = native.config.get('vision_config')
    if not isinstance(vision, dict):
        raise ValueError('the Gemma4 fixture requires its vision configuration')
    patch, pool = vision['patch_size'], vision['pooling_kernel_size']
    side = patch * pool
    frame = (np.arange(side * side * 3).reshape(side, side, 3) % 256).astype(np.uint8)
    videos = [np.stack([frame, np.roll(frame, 3, axis=1)]),
              np.stack([np.flip(frame, axis=0), np.roll(frame, 5, axis=0)])]
    # Nine image tokens exceed the tiny sliding window of four, exercising
    # window AND (causal OR vision-group), not an unbounded group override.
    image = np.tile(frame, (3, 3, 1))
    images = [[image], [np.roll(image, 3, axis=0)]]
    video_token, image_token = processor.video_token, processor.image_token
    text = [f'{video_token} {image_token} token4',
            f'{image_token} {video_token} token5']
    metadata = [VideoMetadata(total_num_frames=2, fps=2.0, width=side, height=side,
                              frames_indices=[0, 1]) for _ in videos]
    # Documented per-call options match the tiny tower; serialized video
    # defaults describe the released patch16/pool3 geometry.
    values = dict(processor(
        text=text, images=images, videos=videos, padding=True, return_tensors='np',
        images_kwargs={'do_resize': False, 'max_soft_tokens': 70},
        videos_kwargs={'patch_size': patch, 'pooling_kernel_size': pool,
                       'do_resize': False, 'do_sample_frames': False,
                       'max_soft_tokens': 70, 'input_data_format': 'channels_last',
                       'video_metadata': metadata}))
    return loaded, values


def test_gemma4_processor_interleaves_independent_image_and_video_streams(gemma4_video_batch):
    import torch
    from transformers.models.gemma4.modeling_gemma4 import get_block_sequence_ids_for_mask

    loaded, values = gemma4_video_batch
    inputs = loaded.processor.from_hf(values)
    frames, images = values['pixel_values_videos'], values['pixel_values']
    expected = np.stack([frames[0, 0], frames[0, 1], images[0],
                         images[1], frames[1, 0], frames[1, 1]])
    actual = np.asarray(inputs.conditioning['pixel_values'])
    np.testing.assert_array_equal(actual.reshape(expected.shape), expected)
    np.testing.assert_array_equal(inputs.conditioning['image_lengths'], [3, 3])
    groups = get_block_sequence_ids_for_mask(
        torch.from_numpy(values['mm_token_type_ids']), torch.device('cpu')).numpy()
    np.testing.assert_array_equal(inputs.token_fields['image_groups'], groups)


@pytest.mark.parametrize('grouped', [False, True], ids=['causal_without_mmtypes', 'vision_groups'])
def test_gemma4_mixed_video_forward_matches_unmodified_reference(gemma4_video_batch, grouped):
    import torch
    from transformers import Gemma4ForConditionalGeneration

    loaded, original = gemma4_video_batch
    values = dict(original)
    if not grouped:
        del values['mm_token_type_ids']
    inputs = loaded.processor.from_hf(values)
    if not grouped:
        assert 'image_groups' not in inputs.token_fields
    reference = Gemma4ForConditionalGeneration.from_pretrained(
        FIXTURE.parent / 'gemma4-native-tiny', dtype=torch.float32).eval()
    reference.set_attn_implementation('eager')
    with torch.no_grad():
        expected = reference(**{name: torch.from_numpy(value) for name, value in values.items()},
                             use_cache=False).logits.numpy()
    actual = jax.jit(lambda variables: loaded.model.apply(
        variables, inputs.tokens, **inputs.kwargs()))(loaded.variables)
    valid = np.asarray(values['attention_mask'], dtype=bool)
    np.testing.assert_allclose(np.asarray(actual)[valid], expected[valid], atol=1e-4, rtol=0)


def test_public_cached_generation_preserves_image_conditioning(source):
    """Greedy continuation matches HF; changing pixels changes raw likelihoods."""
    loaded, inputs = source
    task = loaded.text_generation()
    generated = task(inputs, 3, key=jax.random.key(1), sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(generated.tokens[:, -3:], np.load(FIXTURE / "continuation.npy"))
    np.testing.assert_array_equal(generated.lengths, [3, 3])
    assert task.decode(generated) == tuple(loaded.processor.decode(generated.tokens[:, -3:]))
    blank = dataclasses.replace(inputs, conditioning={
        **inputs.conditioning, "pixel_values": jnp.zeros_like(inputs.conditioning["pixel_values"])})
    altered = task(blank, 3, key=jax.random.key(1), sampling=Sampling(temperature=0))
    # The observed maximum change is 0.0840, even though greedy tokens agree.
    assert np.max(np.abs(generated.raw_log_probs - altered.raw_log_probs)) > 1e-2



def test_gemma4_standardization_buffers_are_frozen_by_real_adamw_training(tmp_path):
    """The exported HF buffers stay bitwise unchanged while AdamW updates weights."""
    directory = FIXTURE.parent / "gemma4-tiny-mm"
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    patches = np.load(directory / "pixels.npy")
    batch, count, _ = patches.shape
    side = int(count ** 0.5)
    patch = loaded.model.vision.patch_size
    images = patches.reshape(batch, side, side, patch, patch, 3).transpose(
        0, 5, 1, 3, 2, 4).reshape(batch, 1, 3, side * patch, side * patch)
    tokens = jnp.asarray(np.load(directory / "input_ids.npy"))
    marks = tokens == loaded.model.image_token_id
    inputs = ModelInputs(tokens, {"image_indices": jnp.where(marks, jnp.cumsum(marks, axis=1) - 1, -1)},
                         {"pixel_values": jnp.asarray(images)})
    rows = 2 * jax.device_count()
    repeated = inputs.take_rows(jnp.arange(rows) % 2)
    objective = LMObjective(loaded, tokens.shape[1] - 1,
                            ema_decay=None, pad_id=0)
    data = Loader(train=lambda partition: iter([{"text": repeated}]), val=None, records=rows, batch=rows)
    trainer = Trainer(objective, optax.adamw(1e-3, weight_decay=0.1), key=jax.random.key(12),
                      mesh=MeshSpec(), layout=Layout(min_shard=2**30))
    state = trainer.fit(data, steps=1, log_every=1)
    loaded.save(tmp_path, variables=state.variables)
    before = load_file(str(directory / "model.safetensors"))
    after = load_file(str(tmp_path / "model.safetensors"))
    for name in ("std_bias", "std_scale"):
        key = "model.vision_tower." + name
        np.testing.assert_array_equal(after[key], before[key])
    for name in before:
        if name.endswith("layer_scalar"):
            np.testing.assert_array_equal(after[name], before[name])
    trained = "model.vision_tower.patch_embedder.input_proj.weight"
    assert np.max(np.abs(after[trained] - before[trained])) > 1e-4



FAMILIES = {
    # family: (references, checkpoint, forward error, pixel-gradient error, post-SGD error)
    "gemma4": ("gemma4-native-tiny", "gemma4-native-tiny", 1.34e-5, 2.04e-6, 1.43e-5),
    "qwen35": ("qwen35-native-tiny", "qwen35-native-tiny", 3.46e-5, 2.69e-6, 5.28e-5),
    "llama4": ("llama4-native-tiny", "llama4-native-tiny", 6.76e-6, 5.4e-8, 1.22e-5),
    # Images and audio through the same prompt; the checkpoint and its
    # processor are the audio fixture's, the references sit beside them.
    "gemma3n": ("gemma3n-native-tiny", "gemma-3n-audio-tiny", 1.95e-6, 2.39e-7, 2.31e-5),
}
ORDER_MARGIN = 3
"""How many of the reference's own CUDA-vs-CPU spreads Dew's post-SGD logits
may move by on another fp32 reduction order. The spread comes from the
gradient, not the updated forward: Gemma 3n's vision tower scales each block
by a per-channel layer-scale gamma, whose gradient sums over every spatial
position with heavy cancellation, so the summation order moves those
gradients by 2-4e-5 relative (stages_1.blocks_0's alone moves the updated
logits 9.5e-5), while the updated forward itself moves 2.9e-6 across
backends. The reference moves 7.5e-5 between its CPU and CUDA runs
(post_sgd_spread.json, written by tools/multimodal_reference.py
gemma3n-spread, TF32 off). The margin is chosen, not derived: Dew's XLA
order moves the logits 1.68e-4 from Dew's CPU run, 2.2 spreads, measured on
an RTX 4080 and an A100 (which agree to 2e-6); a TPU's order is unmeasured."""


def _post_sgd_bound(family: str, directory: Path) -> float:
    """The post-SGD logit bound: 1e-4 where the reference's order spread is
    below the implementation error, else Dew's CPU error against the
    reference plus ORDER_MARGIN of the reference's CUDA-vs-CPU spread."""
    spread = directory / "post_sgd_spread.json"
    if not spread.is_file():
        return 1e-4
    return FAMILIES[family][4] + ORDER_MARGIN * json.loads(spread.read_text())["cuda_vs_cpu"]


def _one_layer_negated(gradient):
    """The gradient with the first decoder layer's subtree negated: a sign
    slip in one layer's backward that leaves every shape alone."""
    def walk(node):
        if not isinstance(node, dict):
            return node, False
        if "layers_0" in node:
            return {**node, "layers_0": jax.tree.map(jnp.negative, node["layers_0"])}, True
        out, found = {}, False
        for key, value in node.items():
            out[key], hit = walk(value) if not found else (value, False)
            found = found or hit
        return out, found

    negated, found = walk(gradient)
    assert found, "no layers_0 subtree in the gradient"
    return negated


def _wrong_inputs(family: str, inputs: ModelInputs) -> ModelInputs:
    """One plausible integration bug per family that the parity assertion must catch."""
    if family == "gemma4":
        # The 2D position tables read (x, y); swapping them is the natural slip.
        positions = inputs.conditioning["image_position_ids"]
        return dataclasses.replace(
            inputs, conditioning={**inputs.conditioning, "image_position_ids": positions[..., ::-1]}
        )
    if family == "qwen35":
        # Text-only rotary coordinates instead of the interleaved spatial ones.
        return dataclasses.replace(inputs, token_fields={
            name: value for name, value in inputs.token_fields.items() if name != "rotary_positions"})
    if family == "gemma3n":
        # The reference hands its audio encoder the inverted mask; getting
        # the polarity wrong keeps every shape.
        mask = inputs.conditioning["input_features_mask"]
        return dataclasses.replace(inputs, conditioning={**inputs.conditioning, "input_features_mask": ~mask})
    # Llama 4 orders local tiles before the global tile; reversing them keeps
    # every shape and count.
    pixels = inputs.conditioning["pixel_values"]
    return dataclasses.replace(inputs, conditioning={**inputs.conditioning, "pixel_values": pixels[:, ::-1]})



def _real_pixels(inputs: ModelInputs, gradient: jax.Array) -> jax.Array:
    """The per-image gradient in the reference's flat image order, without padding."""
    lengths = np.asarray(inputs.conditioning["image_lengths"])
    grid = inputs.conditioning.get("image_grid_thw")
    if grid is None:
        return jnp.concatenate([gradient[row, :int(count)] for row, count in enumerate(lengths)])
    return jnp.concatenate([gradient[row, image, :int(np.prod(np.asarray(grid[row, image])))]
                            for row, count in enumerate(lengths) for image in range(int(count))])


def _family(family: str):
    """A family's loaded checkpoint, its actual processor's inputs and its
    references' directory."""
    pytest.importorskip("torchvision", reason="the vision extra supplies the actual processors")
    references, checkpoint, *_ = FAMILIES[family]
    directory = FIXTURE.parent / references
    source = FIXTURE.parent / checkpoint
    loaded = Pretrained.load(source, dtype="float32", attention_impl="reference")
    images = np.load(directory / "raw_images.npy")
    waveforms = sorted(source.glob("waveform_*.npy"))
    assert loaded.processor is not None
    inputs = loaded.processor(json.loads((directory / "prompts.json").read_text()),
                              images=[[images[0]], [images[1], images[2]]],
                              audio=[np.load(path) for path in waveforms] if waveforms else None)
    return loaded, inputs, directory


@pytest.fixture(scope="module", params=sorted(FAMILIES))
def family_source(request):
    return request.param, *_family(request.param)


def test_training_reaches_the_gemma3n_vision_drop_path():
    """`train` reaches Gemma 3n's MobileNet through the wrapper: with a
    drop-path rate the training forward depends on the dropout key, without
    one nothing else in the model draws, and evaluation still reads the
    reference logits."""
    loaded, inputs, directory = _family("gemma3n")
    dropping = loaded.model.clone(vision=dataclasses.replace(loaded.model.vision, drop_path_rate=0.4))

    def forward(model, train, seed):
        return np.asarray(jax.jit(lambda variables: model.apply(
            variables, inputs.tokens, **inputs.kwargs(), train=train,
            rngs={"dropout": jax.random.key(seed)}))(loaded.variables))

    valid = np.asarray(inputs.token_fields["attention_mask"])
    np.testing.assert_array_equal(forward(loaded.model, train=True, seed=1),
                                  forward(loaded.model, train=True, seed=2))
    first = forward(dropping, train=True, seed=1)
    np.testing.assert_array_equal(first, forward(dropping, train=True, seed=1))
    assert np.max(np.abs(first - forward(dropping, train=True, seed=2))[valid]) > 1e-3
    expected = np.load(directory / "logits.npy")
    np.testing.assert_allclose(
        forward(dropping, train=False, seed=1)[valid], expected[valid], atol=1e-4, rtol=0
    )


@pytest.fixture(scope="module", params=["gemma-3n-audio-tiny", "gemma-4-audio-tiny"])
def audio_source(request):
    pytest.importorskip("torchvision", reason="the vision extra supplies the actual Gemma processors")
    directory = FIXTURE.parent / request.param
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    meta = json.loads((directory / "meta.json").read_text())
    assert loaded.processor is not None
    inputs = loaded.processor(
        meta["prompts"], audio=[np.load(directory / f"waveform_{index}.npy") for index in range(2)]
    )
    return loaded, inputs, np.load(directory / "reference.npz")


def test_audio_only_processor_forward_and_greedy_continuation_match_reference(audio_source):
    """Waveforms through the actual Gemma processors and native audio towers.

    Valid-logit errors 3.2e-7 (Gemma 3n, fixed slots and padding embeddings)
    and 2.3e-7 (Gemma 4, one slot per encoded frame) at tolerance 1e-4;
    greedy continuation matches, including Gemma 3n's hard-range samples.
    """
    loaded, inputs, reference = audio_source
    np.testing.assert_array_equal(inputs.tokens, reference["input_ids"])
    valid = np.asarray(inputs.token_fields["attention_mask"])
    logits = jax.jit(lambda variables: loaded.model.apply(variables, inputs.tokens, **inputs.kwargs()))(
        loaded.variables
    )
    np.testing.assert_allclose(np.asarray(logits)[valid], reference["logits"][valid], atol=1e-4, rtol=0)
    generated = loaded.text_generation()(inputs, 3, key=jax.random.key(1), sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(generated.tokens[:, -3:], reference["generated"][:, -3:])


def test_a_gemma4_audio_models_prediction_depths_read_its_audio_frames():
    """A Gemma 4 audio model with a prediction depth initializes and trains with its audio, and the
    depth reads the embeddings the main forward read, its audio frames in place: its states are
    the language model's depth over the main forward's own prepared embeddings, bitwise."""
    pytest.importorskip("torchvision", reason="the vision extra supplies the actual Gemma processors")
    directory = FIXTURE.parent / "gemma-4-audio-tiny"
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    meta = json.loads((directory / "meta.json").read_text())
    inputs = loaded.processor(meta["prompts"],
                              audio=[np.load(directory / f"waveform_{index}.npy") for index in range(2)])
    model = loaded.model.clone(language_model=loaded.model.language_model.clone(num_nextn_predict_layers=1))
    variables = model.init(jax.random.key(0), inputs.tokens, **inputs.kwargs())
    hidden, sown = model.apply(variables, inputs.tokens, **inputs.kwargs(), method="hidden_states",
                               mutable=["embeddings"])
    depths = model.apply(variables, hidden, inputs.tokens, **inputs.kwargs(), method="mtp_hidden_states")
    language = {collection: tree["language_model"] for collection, tree in variables.items()
                if "language_model" in tree}
    fields = {name: value for name, value in inputs.kwargs().items()
              if name not in ("conditioning", "image_indices", "audio_indices")}
    prepared = sown["embeddings"]["language_model"]["prepared"]
    over_prepared = model.language_model.apply(language, hidden, inputs.tokens, method="mtp_hidden_states",
                                               input_embeddings=prepared, **fields)
    for mine, theirs in zip(depths, over_prepared, strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))
    objective = LMObjective(model, inputs.tokens.shape[1] - 1, ema_decay=None, pad_id=0, mtp_weight=1.0)
    statistics, _ = objective.loss(variables, {"text": inputs},
                                   Step(step=jnp.int32(0), key=jax.random.key(1), ema=None))
    assert np.isfinite(float(objective.reduce_loss(statistics)[0]))


def test_a_cached_continuation_reads_the_coordinates_its_prefill_was_placed_at():
    """A cached prefill whose tokens' places come as coordinates alone leaves the decode cursor
    where raw positions leave it: an implicit continuation step after it is the same, bitwise."""
    loaded = Pretrained.load(FIXTURE, dtype="float32", attention_impl="reference")
    model, variables = loaded.model, loaded.variables
    tokens = jnp.asarray(np.load(FIXTURE / "input_ids.npy"))[:1, -6:]
    positions = jnp.arange(6)[None] + 10
    cache = model.apply(variables, 1, method="init_cache", mutable=["cache"])[1]["cache"]

    def continued(**placed):
        _, mutated = model.apply({**variables, "cache": cache}, tokens[:, :4], decode=True,
                                 mutable=["cache"], **placed)
        return model.apply({**variables, "cache": mutated["cache"]}, tokens[:, 4:5], decode=True,
                           mutable=["cache"])[0]

    np.testing.assert_array_equal(np.asarray(continued(coordinates=coordinates(positions=positions[:, :4]))),
                                  np.asarray(continued(positions=positions[:, :4])))


def test_source_processor_forward_and_cached_generation_match_reference(family_source):
    """Actual processors to native models: Gemma4's padded patches with active
    clipping and its video placeholder decode rule, Qwen3.5's packed
    channel-time patches with spatial M-RoPE, Llama4's local and global tiles,
    Gemma3n's MobileNet features, hard vocabulary ranges, fixed audio slots
    and per-layer input masking beside its conformer audio.

    FP32 valid-logit errors are recorded in FAMILIES at tolerance 1e-4; each
    family's characteristic slip must fail the same assertion.
    """
    family, loaded, inputs, directory = family_source
    np.testing.assert_array_equal(inputs.tokens, np.load(directory / "input_ids.npy"))
    logits = jax.jit(lambda variables: loaded.model.apply(
        variables, inputs.tokens, **inputs.kwargs()))(loaded.variables)
    valid = np.asarray(inputs.token_fields["attention_mask"])
    expected = np.load(directory / "logits.npy")
    np.testing.assert_allclose(np.asarray(logits)[valid], expected[valid], atol=1e-4, rtol=0)
    np.testing.assert_array_equal(np.asarray(logits)[valid].argmax(-1), expected[valid].argmax(-1))
    wrong_inputs = _wrong_inputs(family, inputs)
    wrong = loaded.model.apply(loaded.variables, wrong_inputs.tokens, **wrong_inputs.kwargs())
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(np.asarray(wrong)[valid], expected[valid], atol=1e-4, rtol=0)
    generated = loaded.text_generation()(inputs, 3, key=jax.random.key(1), sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(generated.tokens[:, -3:], np.load(directory / "continuation.npy"))


def test_source_backward_trained_export_and_frozen_buffers_match_reference(family_source, tmp_path):
    """Loss, pixel gradient and all-parameter SGD against the reference backward.

    Recorded errors are in FAMILIES; the post-SGD logits are held to
    `_post_sgd_bound`, which must still fail one layer's gradient with its
    sign flipped. The trained model exports under the
    source's tensor names and reloads to the reference's updated logits; every
    non-parameter collection (Gemma4's standardization and clipping buffers)
    survives training bitwise. Qwen3.5's root tie_word_embeddings=False
    overrides its nested text flag: tying the head keeps the initial logits
    but misses this update by 0.00167.
    """
    family, loaded, inputs, directory = family_source
    objective = LMObjective(loaded, inputs.tokens.shape[1] - 1,
                            ema_decay=None, pad_id=0)
    step = Step(step=jnp.int32(0), key=jax.random.key(4), ema=None)

    def loss(variables, pixels):
        values = {**loaded.variables, "params": variables}
        data = dataclasses.replace(inputs, conditioning={**inputs.conditioning, "pixel_values": pixels})
        statistics, _ = objective.loss(values, {"text": data}, step)
        return objective.reduce_loss(statistics)[0]

    value, (gradient, pixels) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))(
        loaded.variables["params"], inputs.conditioning["pixel_values"])
    expected = json.loads((directory / "training.json").read_text())
    np.testing.assert_allclose(value, expected["loss"], atol=1e-5, rtol=0)
    np.testing.assert_allclose(_real_pixels(inputs, pixels), np.load(directory / "pixel_gradient.npy"),
                               atol=1e-5, rtol=1e-4)
    optimizer = optax.sgd(expected["learning_rate"])
    params = loaded.variables["params"]
    valid = np.asarray(inputs.token_fields["attention_mask"])
    reference = np.load(directory / "updated_logits.npy")[valid]
    bound = _post_sgd_bound(family, directory)

    def trained(gradient, destination):
        updates, _ = optimizer.update(gradient, optimizer.init(params), params)
        loaded.save(
            destination, variables={**loaded.variables, "params": optax.apply_updates(params, updates)}
        )
        restored = Pretrained.load(destination, dtype="float32", attention_impl="reference")
        return restored, np.asarray(
            restored.model.apply(restored.variables, inputs.tokens, **inputs.kwargs())
        )[valid]

    restored, output = trained(gradient, tmp_path / "trained")
    np.testing.assert_allclose(output, reference, atol=bound, rtol=0)
    # The bound still catches one layer's gradient with its sign flipped.
    _, slipped = trained(_one_layer_negated(gradient), tmp_path / "slipped")
    assert np.max(np.abs(slipped - reference)) > bound
    for collection, tree in restored.variables.items():
        if collection != "params":
            for actual, original in zip(jax.tree.leaves(tree),
                                        jax.tree.leaves(loaded.variables[collection]), strict=True):
                np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("fixture", ["gemma3n-vision-tiny", "gemma-3n-audio-tiny"])
def test_a_gemma3n_config_without_its_audio_placeholder_takes_the_config_class_default(fixture):
    """Gemma3nConfig defaults audio_token_id (262273, configuration_gemma3n.py:460 at transformers
    5.17.0), so a wrapper config that leaves it out translates as transformers' config class reads it:
    one with an audio tower names that placeholder, and one without states no audio either way."""
    from dew.interop.hf_decoders import translate_wrapper_config

    config = json.loads((FIXTURE.parent / fixture / "config.json").read_text())
    stated = translate_wrapper_config(config)
    del config["audio_token_id"]
    defaulted = translate_wrapper_config(config)
    audio = None if stated["audio"] is None else 262273
    assert defaulted == {**stated, "audio_token_id": audio}
