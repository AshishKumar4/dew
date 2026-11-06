"""MiMo-V2.6's audio path against its references: the log mel frames against torchaudio, the
tokenizer's encoder and the residual codes and the audio encoder against MiMo's own
`modeling_mimo_v2.py`, each within the float64 gate (tests/reference_error.py), the codes exactly.

tools/mimo_reference.py writes the fixture: two clips of 45 and 13 mel frames, cut into 20-frame
segments as tokenize_audio_batch cuts them, through a four-layer tokenizer (two windowed, two full
causal layers, a skip connection, the pooling of odd segments beside the batch's longest) and a
two-layer encoder over groups of four code frames; once in the reference's one batch, and once with
its batches bounded at 30 frames, where the 13-frame clip runs alone and odd.
"""

import dataclasses
import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from flax import linen as nn
from model_support import recorded_site
from one_draw import assert_as_exact_as_the_reference
from reference_error import as_given, assert_as_exact, assert_rounds_where_the_reference_does, ieee_fixture
from tool_support import run_from

from dew.nn.mimo_audio import (
    AudioTokenizerEncoder,
    MiMoAudioCodeEncoder,
    MiMoAudioTokenizer,
    batch_longest,
    translate_mimo_audio_config,
    translate_mimo_audio_weights,
)
from dew.nn.norms import RMSNorm
from dew.nn.quantizers import residual_quantize
from dew.registry import towers

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hf" / "mimo-audio-tiny"


def load():
    config = json.loads((FIXTURE / "config.json").read_text())
    refs = ieee_fixture(FIXTURE / "model.npz")
    tokenizer = {name.removeprefix("tokenizer/"): value for name, value in refs.items()
                 if name.startswith("tokenizer/")}
    encoder = {name.removeprefix("encoder/").removeprefix("audio_encoder."): value
               for name, value in refs.items() if name.startswith("encoder/")}
    record = translate_mimo_audio_config(config["audio_config"], config["tokenizer"])
    return towers.from_record(record), translate_mimo_audio_weights(tokenizer, encoder), refs


def test_log_mel_frames_are_as_exact_as_torchaudio():
    """MiMo's 24 kHz, 128-mel frames of half a second from transformers' shared STFT and filters."""
    refs = ieee_fixture(FIXTURE / "mel.npz")
    frames = MiMoAudioTokenizer().mel()(refs["waveform"])
    assert frames.shape == refs["mel"].shape
    assert_as_exact_as_the_reference(frames, refs["mel"], refs["mel_f64"], "log mel")


RECORDING = FIXTURE / "recording.npz"
"""The release's tokenizer and audio encoder over drawn mels and codes (tools/mimo_reference.py
--record-audio), summarized over seeds for check 1."""


@functools.cache
def generator():
    """tools/mimo_reference.py, whose seeded draws the recorded sites' inputs are drawn again by."""
    with run_from("mimo_reference", "tools") as module:
        return module


def segmented(config, mels, lengths):
    """The clips cut into segments of `audio_segment_size` frames, as tokenize_audio_batch cuts them, and
    each segment's span; `lengths` on the host, so the cuts are static where the mels are traced."""
    size = config.audio_segment_size
    lengths = [int(length) for length in lengths]
    segments = [(clip, start) for clip, length in enumerate(lengths) for start in range(0, length, size)]
    padded = jnp.pad(mels, ((0, 0), (0, size), (0, 0)))
    return (jnp.stack([padded[clip, start:start + size] for clip, start in segments]),
            jnp.asarray([min(size, lengths[clip] - start) for clip, start in segments], jnp.int32))


def float64_of(tree):
    """`tree` with its floating leaves in float64, for a module's float64 twin."""
    with jax.enable_x64(new_val=True):
        return jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64 if np.issubdtype(
            np.asarray(leaf).dtype, np.floating) else None), tree)


@pytest.mark.parametrize("prefix, bound", [("", 256000), ("bounded_", 30)])
def test_the_tokenizer_s_features_and_codes_are_the_reference_s(prefix, bound):
    """Each segment's pooled features over drawn mels on par with the reference's from its float64
    run's (check 1, the recording's), and over the fixture's mels the codes they quantize to the
    reference's exactly; the reference's own features quantize to its codes under Dew's RVQ. Bounded,
    a batch's own longest ends its convolutions and its pooling (`batch_longest`)."""
    config, variables, refs = load()
    config = dataclasses.replace(config, tokenizer_batch_frames=bound)
    encoder, held = AudioTokenizerEncoder(config.tokenizer), {"params": variables["params"]["tokenizer"]}
    lengths = generator().LENGTHS

    def mels_of(inputs):
        width = max(lengths)
        return jnp.stack([jnp.pad(inputs[f"mel_{clip}"], ((0, width - length), (0, 0)))
                          for clip, length in enumerate(lengths)])

    # The pooled entries each segment holds, which its span alone decides: read on the host once.
    _, spans = segmented(config, np.zeros((len(lengths), max(lengths), config.tokenizer.n_mels)), lengths)
    shape = (len(spans), config.audio_segment_size, config.tokenizer.n_mels)
    valid = encoder.apply(held, np.zeros(shape, np.float32), spans, batch_longest(spans, config))[1]
    # int32 places, which index a float64 twin's entries as they index the float32 run's.
    kept = np.flatnonzero(np.asarray(valid).reshape(-1)).astype(np.int32)

    def entries(features):
        return features.reshape(-1, features.shape[-1])[kept]

    def ours(inputs):
        mels, spans = segmented(config, mels_of(inputs), lengths)
        return {"features": entries(encoder.apply(held, mels, spans, batch_longest(spans, config))[0])}

    wide_held, wide_encoder = float64_of(held), encoder.clone(dtype=jnp.float64)

    def wide(inputs):
        # The segments and each batch's longest, integers the float32 run reads, cut before widening.
        mels, spans = segmented(config, mels_of(inputs), lengths)
        longest = batch_longest(spans, config)
        with jax.enable_x64(new_val=True):
            features, _ = wide_encoder.apply(wide_held, jnp.asarray(mels, jnp.float64), spans, longest)
            return {"features": entries(features)}

    site = f"{'bounded ' if prefix else ''}tokenizer features"
    draw, reference, truth, seeds = recorded_site(RECORDING, site, generator().mels_draw, wide)
    assert_as_exact(as_given(lambda index, inputs: ours(inputs)), reference, draw, site, seeds=seeds,
                    truth_fn=truth)
    mels, spans = segmented(config, refs["mels"], refs["lengths"])
    features, valid = encoder.apply(held, mels, spans, batch_longest(spans, config))
    features = np.asarray(features)[np.asarray(valid)]
    books = [variables["params"]["quantizer"][f"codebook_{stage}"]
             for stage in range(len(config.tokenizer.codebook_size))]
    expected = np.concatenate([refs[f"{prefix}codes_{clip}"] for clip in range(len(refs["lengths"]))])
    np.testing.assert_array_equal(np.asarray(residual_quantize(jnp.asarray(features), books)), expected)
    reference = jnp.asarray(refs[f"{prefix}features"])
    np.testing.assert_array_equal(np.asarray(residual_quantize(reference, books)), expected)


def test_the_audio_encoder_is_as_exact_as_the_reference():
    """Drawn codes of two clips, the second's last group partial, through the speech embeddings, the
    bidirectional groups and the projection: each clip's groups on par with the reference's from its
    float64 run's (check 1, the recording's)."""
    config, variables, _ = load()
    held = {"params": variables["params"]["encoder"]}
    lengths = generator().CODE_LENGTHS
    groups = [-(-length // config.group_size) for length in lengths]

    encoder = MiMoAudioCodeEncoder(config)

    def codes_and_counts(inputs):
        codes = [inputs[f"codes_{clip}"] for clip in range(len(lengths))]
        return (jnp.stack([jnp.pad(c, ((0, max(lengths) - length), (0, 0)))
                           for c, length in zip(codes, lengths, strict=True)]),
                np.asarray(lengths, np.int32))

    def clips(out):
        """Each clip's real groups, in order."""
        return {"encoded": jnp.concatenate([out[clip, :count] for clip, count in enumerate(groups)])}

    wide_held, wide_encoder = float64_of(held), encoder.clone(dtype=jnp.float64)

    def wide(inputs):
        with jax.enable_x64(new_val=True):
            return clips(wide_encoder.apply(wide_held, *codes_and_counts(inputs)))

    draw, reference, truth, seeds = recorded_site(RECORDING, "audio encoder", generator().codes_draw, wide)
    assert_as_exact(as_given(lambda index, inputs: clips(encoder.apply(held, *codes_and_counts(inputs)))),
                    reference, draw, "audio encoder", seeds=seeds, truth_fn=truth)


@pytest.mark.parametrize("prefix, bound", [("", 256000), ("bounded_", 30)])
def test_the_tower_end_to_end_encodes_its_tokenizers_codes(prefix, bound):
    """The tower end to end over the fixture's clips counts and encodes each clip's groups as the code
    encoder does over the reference's codes."""
    config, variables, refs = load()
    config = dataclasses.replace(config, tokenizer_batch_frames=bound)
    count = len(refs["lengths"])
    codes = [refs[f"{prefix}codes_{clip}"] for clip in range(count)]
    width = max(len(c) for c in codes)
    padded = np.stack([np.pad(c, ((0, width - len(c)), (0, 0))) for c in codes])
    counts = np.asarray([len(c) for c in codes], np.int32)
    out = MiMoAudioCodeEncoder(config).apply({"params": variables["params"]["encoder"]}, padded, counts)
    mask = np.arange(refs["mels"].shape[1])[None, :] < refs["lengths"][:, None]
    encoding = config.build().apply(variables, refs["mels"], mask)
    for clip in range(count):
        groups = len(refs[f"{prefix}embedded_{clip}"])
        assert int(np.asarray(encoding.mask)[clip].sum()) == groups
        np.testing.assert_allclose(np.asarray(encoding.features)[clip, :groups],
                                   np.asarray(out)[clip, :groups], rtol=0, atol=1e-5)


def test_a_bf16_checkpoint_binds_in_its_own_dtype_and_runs():
    """Stored in bf16 and read as stored, the tokenizer's key maps get a bias in their own dtype, so
    queries, keys and values meet in one."""
    config = json.loads((FIXTURE / "config.json").read_text())
    refs = ieee_fixture(FIXTURE / "model.npz")
    narrow = {name: np.asarray(value, ml_dtypes.bfloat16) for name, value in refs.items()
              if name.startswith(("tokenizer/", "encoder/")) and np.issubdtype(value.dtype, np.floating)}
    tokenizer = {name.removeprefix("tokenizer/"): value for name, value in narrow.items()
                 if name.startswith("tokenizer/")}
    encoder = {name.removeprefix("encoder/").removeprefix("audio_encoder."): value
               for name, value in narrow.items() if name.startswith("encoder/")}
    variables = translate_mimo_audio_weights(tokenizer, encoder, param_dtype="auto")
    keys = variables["params"]["tokenizer"]["layers_0"]["self_attn"]["k_proj"]
    assert keys["bias"].dtype == keys["kernel"].dtype == ml_dtypes.bfloat16
    tower = towers.from_record(translate_mimo_audio_config(config["audio_config"], config["tokenizer"]))
    mask = np.arange(refs["mels"].shape[1])[None, :] < refs["lengths"][:, None]
    encoding = tower.build().apply(variables, jnp.asarray(refs["mels"], jnp.bfloat16), mask)
    assert np.isfinite(np.asarray(encoding.features, np.float32)).all()


CODE_ENCODER_SEEDS = 16
"""The bfloat16 code encoder's declared check-1 seeds against Qwen2Model, sized by `seeds_for` paired on
armada."""


def test_the_code_encoder_in_bf16_rounds_as_its_qwen2_reference_does():
    """The code encoder's transformer in bfloat16 against the class MiMo builds for it,
    transformers' Qwen2Model (modeling_mimo_v2.py:944-957 at 6a7f202), on the same
    bfloat16 weights and code embeddings: over drawn codes on par with Qwen2Model's bfloat16
    from its float64 run's (check 1). Every norm the encoder runs on the fixture's codes, given
    the input it received, rounds where Qwen2RMSNorm does (modeling_qwen2.py:247-252): the
    normalized activations cast to bfloat16 before the weight multiplies them. Scaling
    before the cast is the more accurate order, so only the per-norm check tells them apart."""
    import contextlib

    import torch
    from transformers import Qwen2Config, Qwen2Model
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    from tools.torch_precision import float64

    config, variables, refs = load()
    audio = json.loads((FIXTURE / "config.json").read_text())["audio_config"]
    encoder = {name.removeprefix("encoder/").removeprefix("audio_encoder."): value
               for name, value in refs.items() if name.startswith("encoder/")}
    params = {"params": jax.tree.map(lambda leaf: np.asarray(leaf, ml_dtypes.bfloat16),
                                     variables["params"]["encoder"])}
    codes = refs["codes_0"]
    norms = []

    def record(call, args, kwargs, context):
        out = call(*args, **kwargs)
        if isinstance(context.module, RMSNorm) and context.method_name == "seam":
            module = context.module
            norms.append((module.path, out, module.variables["params"]["scale"], module.epsilon))
        return out

    with nn.intercept_methods(record):
        MiMoAudioCodeEncoder(config, dtype=jnp.bfloat16).apply(
            params, codes[None], np.asarray([len(codes)], np.int32))
    assert len(norms) == 2 * audio["input_local_layers"] + audio["add_post_norm"]
    for path, (stream, normed), scale, epsilon in norms:
        norm = Qwen2RMSNorm(stream.shape[-1], eps=epsilon).to(torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_(torch.from_numpy(np.asarray(scale, np.float32)))
            theirs = norm(torch.from_numpy(np.asarray(stream, np.float32)).to(torch.bfloat16))
        wide = np.asarray(stream, np.float64)
        truth = (np.asarray(scale, np.float64) * wide
                 / np.sqrt(np.mean(wide * wide, axis=-1, keepdims=True) + epsilon))
        assert_rounds_where_the_reference_does(
            np.asarray(normed, np.float64), theirs.double().numpy(), truth, "/".join(path))

    def reference(dtype):
        # The float64 truth is built and run under `float64`, so the norms' and the
        # rotary's float32 pins compute in float64 too.
        with float64() if dtype == torch.float64 else contextlib.nullcontext():
            model = Qwen2Model(Qwen2Config(
                hidden_size=audio["input_local_dim"], num_hidden_layers=audio["input_local_layers"],
                num_attention_heads=audio["input_local_attn_heads"],
                num_key_value_heads=audio["input_local_attn_heads"],
                intermediate_size=audio["input_local_intermediate_size"],
                rope_theta=audio["rope_theta"])).eval()
            model.load_state_dict({name.removeprefix("input_local_transformer."): torch.from_numpy(value)
                                   for name, value in encoder.items()
                                   if name.startswith("input_local_transformer.")}, strict=False)
            model = model.to(dtype)
        tables = [torch.from_numpy(encoder[f"speech_embeddings.{channel}.weight"]).to(dtype)
                  for channel in range(audio["audio_channels"])]

        @as_given
        def encoded(codes):
            with float64() if dtype == torch.float64 else contextlib.nullcontext():
                group, frames = audio["group_size"], len(codes)
                groups = -(-frames // group)
                padding = codes[-1:].repeat(groups * group - frames, 0)
                held = torch.from_numpy(np.concatenate([codes, padding]))
                embedded = sum(torch.nn.functional.embedding(held[:, channel].long(), table)
                               for channel, table in enumerate(tables))
                with torch.no_grad():
                    hidden = model(inputs_embeds=embedded.reshape(groups, group, -1), use_cache=False,
                                   is_causal=not audio["input_full_attention"]).last_hidden_state
                return hidden.double().numpy().reshape(-1, hidden.shape[-1])

        return encoded

    def draw(seed):
        """A seed's codes of the fixture's clip's length, each channel within its codebook."""
        rng = np.random.default_rng([0x2aa6, seed])
        sizes = config.tokenizer.codebook_size
        return (np.stack([rng.integers(0, size, len(codes)) for size in sizes], axis=-1).astype(codes.dtype),)

    narrow = MiMoAudioCodeEncoder(config, dtype=jnp.bfloat16)

    @jax.jit
    def ours_of(codes):
        _, state = narrow.apply(params, codes[None], jnp.asarray([codes.shape[0]], jnp.int32),
                                capture_intermediates=True, mutable=["intermediates"])
        hidden = state["intermediates"]["norm"]["__call__"][0]
        return jnp.asarray(hidden, jnp.float32).reshape(-1, hidden.shape[-1])

    assert_as_exact(ours_of, reference(torch.bfloat16), draw, "code encoder transformer in bfloat16",
                    seeds=CODE_ENCODER_SEEDS, truth_fn=reference(torch.float64))


@pytest.mark.parametrize("kernel,stride,pooler", [(3, 2, 2), (1, 2, 2), (2, 2, 3), (5, 1, 2)])
def test_the_host_counts_a_clips_tokens_by_the_towers_own_count(kernel, stride, pooler):
    """The host's count of a clip's encoded tokens, which places its placeholders (`MiMoAudio.encoded`),
    is the tower's own, which its masks read (`tower_counts` a segment, `grouped` a clip): over clips of
    no frames, of whole segments, of a short last segment and of fewer frames than one, for any
    convolution the tokenizer's config states, an empty segment holding none (row 4301)."""
    from dew.nn.mimo_audio import MiMoAudio, grouped, tower_counts

    config = json.loads((FIXTURE / "config.json").read_text())
    tower = towers.from_record(translate_mimo_audio_config(config["audio_config"], config["tokenizer"]))
    assert isinstance(tower, MiMoAudio)
    tokenizer = dataclasses.replace(tower.tokenizer, kernel_size=kernel, stride_size=stride,
                                    avg_pooler=pooler)
    tower = dataclasses.replace(tower, tokenizer=tokenizer)
    size = tower.audio_segment_size
    frames = np.asarray([0, size, 2 * size, 2 * size + 3, 3])
    mask = np.arange(2 * size + 3)[None, :] < frames[:, None]
    assert tower_counts(0, tokenizer) == (0, 0)
    whole = int(tower_counts(size, tokenizer)[1])
    codes = [whole * (count // size) + int(tower_counts(count % size, tokenizer)[1]) for count in frames]
    np.testing.assert_array_equal(tower.encoded(mask).sum(axis=1), [grouped(code, tower.group_size)
                                                                     for code in codes])
    assert not tower.encoded(mask)[0].any()
