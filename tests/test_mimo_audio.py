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
import json
from pathlib import Path

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference, ieee_fixture

from dew.nn.mimo_audio import (
    AudioTokenizerEncoder,
    MiMoAudioCodeEncoder,
    MiMoAudioTokenizer,
    batch_longest,
    translate_mimo_audio_config,
    translate_mimo_audio_weights,
)
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


@pytest.mark.parametrize("prefix, bound", [("", 256000), ("bounded_", 30)])
def test_the_tokenizer_s_features_and_codes_are_the_reference_s(prefix, bound):
    """Each segment's pooled features within the float64 gate, and the codes they quantize to, the
    reference's exactly; the reference's own features quantize to its codes under Dew's RVQ. Bounded,
    a batch's own longest ends its convolutions and its pooling (`batch_longest`)."""
    config, variables, refs = load()
    config = dataclasses.replace(config, tokenizer_batch_frames=bound)
    # The clips cut into segments of `audio_segment_size` frames, as tokenize_audio_batch cuts them.
    size = config.audio_segment_size
    segments = [(clip, start) for clip, length in enumerate(refs["lengths"])
                for start in range(0, length, size)]
    padded = np.pad(refs["mels"], ((0, 0), (0, size), (0, 0)))
    mels = np.stack([padded[clip, start:start + size] for clip, start in segments])
    spans = jnp.asarray([min(size, refs["lengths"][clip] - start) for clip, start in segments], jnp.int32)
    features, valid = AudioTokenizerEncoder(config.tokenizer).apply(
        {"params": variables["params"]["tokenizer"]}, mels, spans, batch_longest(spans, config))
    features = np.asarray(features)[np.asarray(valid)]
    assert_as_exact_as_the_reference(features, refs[f"{prefix}features"], refs[f"{prefix}features_f64"],
                                     "tokenizer features")
    books = [variables["params"]["quantizer"][f"codebook_{stage}"]
             for stage in range(len(config.tokenizer.codebook_size))]
    expected = np.concatenate([refs[f"{prefix}codes_{clip}"] for clip in range(len(refs["lengths"]))])
    np.testing.assert_array_equal(np.asarray(residual_quantize(jnp.asarray(features), books)), expected)
    reference = jnp.asarray(refs[f"{prefix}features"])
    np.testing.assert_array_equal(np.asarray(residual_quantize(reference, books)), expected)


@pytest.mark.parametrize("prefix, bound", [("", 256000), ("bounded_", 30)])
def test_the_audio_encoder_is_as_exact_as_the_reference(prefix, bound):
    """The reference's codes through the speech embeddings, the bidirectional groups and the
    projection, each clip's groups within the float64 gate; the tower end to end gives the same."""
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
        assert_as_exact_as_the_reference(np.asarray(out)[clip, :groups], refs[f"{prefix}embedded_{clip}"],
                                         refs[f"{prefix}embedded_f64_{clip}"], f"audio encoder clip {clip}")
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
