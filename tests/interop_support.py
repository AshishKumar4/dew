"""What several interop tests read off a reference: a checkpoint's stored
bytes and the audio features a checkpoint's own extractor makes."""

import numpy as np


def assert_same_stored_tensors(written: dict, original: dict) -> None:
    """`written` holds every tensor of `original` and no other, each with its
    dtype, shape and bytes; a scalar's bytes are read through a flat view."""
    assert set(written) == set(original)
    for name, value in original.items():
        assert (written[name].dtype, written[name].shape) == (value.dtype, value.shape), name
        np.testing.assert_array_equal(written[name].reshape(-1).view(np.uint8),
                                      value.reshape(-1).view(np.uint8), err_msg=name)


def gemma_features(model_type, config, waveforms):
    """The float32 `input_features` and boolean `input_features_mask` the
    checkpoint's own Transformers extractor makes of 16 kHz `waveforms`,
    with the arguments `Processor` passes it: padded to the longest, 30 s
    at most, frames a multiple of the encoder's 128."""
    from transformers import Gemma3nAudioFeatureExtractor, Gemma4AudioFeatureExtractor

    extractor = {"gemma3n_audio": Gemma3nAudioFeatureExtractor,
                 "gemma4_audio": Gemma4AudioFeatureExtractor}[model_type].from_dict(dict(config))
    features = extractor([np.asarray(waveform, np.float32) for waveform in waveforms],
                         padding="longest", max_length=480000, truncation=True,
                         pad_to_multiple_of=128, return_tensors="np", return_attention_mask=True)
    return {"input_features": np.asarray(features["input_features"], np.float32),
            "input_features_mask": np.asarray(features["input_features_mask"], np.bool_)}
