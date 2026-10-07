"""Audio conditioning: a transformers audio model, lowered by torchax, as the
context a diffusion model reads.

The towers are random, tiny and written here with `save_pretrained`, beside
their feature extractors, so nothing downloads: a wav2vec2 that reads
`input_values` and a Whisper that reads `input_features`. The clips the run
trains on are muxed here by the ffmpeg the `av` extra bundles.
"""

import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.config import ModelConfig
from dew.data import DataPartition, Loading, LocalVideos
from dew.inputs import Condition, HFAudio
from dew.objectives.base import Step
from dew.objectives.diffusion import AudioCondition, DiffusionRunConfig
from dew.sampling import CFG
from dew.training import Trainer

torch = pytest.importorskip("torch")
pytest.importorskip("torchax")

RATE = 16000


@pytest.fixture(scope="module")
def towers(tmp_path_factory):
    """Directories of a tiny wav2vec2 and a tiny Whisper, with extractors."""
    from transformers import (
        Wav2Vec2Config,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2Model,
        WhisperConfig,
        WhisperFeatureExtractor,
        WhisperModel,
    )

    root = tmp_path_factory.mktemp("audio_towers")
    torch.manual_seed(0)
    wav2vec2 = Wav2Vec2Config(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                              intermediate_size=32, conv_dim=(8, 8), conv_stride=(5, 4),
                              conv_kernel=(10, 4), num_conv_pos_embeddings=8,
                              num_conv_pos_embedding_groups=2)
    Wav2Vec2Model(wav2vec2).save_pretrained(root / "wav2vec2")
    Wav2Vec2FeatureExtractor(sampling_rate=RATE).save_pretrained(root / "wav2vec2")
    whisper = WhisperConfig(d_model=16, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
                            decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
                            num_mel_bins=8, vocab_size=64, max_target_positions=16,
                            pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1)
    WhisperModel(whisper).save_pretrained(root / "whisper")
    WhisperFeatureExtractor(feature_size=8).save_pretrained(root / "whisper")
    return root


def _tone(hertz, seconds):
    return (0.5 * np.sin(2 * np.pi * hertz * np.arange(int(seconds * RATE)) / RATE)).astype(np.float32)


def test_the_audio_processor_pads_a_batch_to_its_longest_or_to_the_models_window(towers):
    """wav2vec2's extractor pads nothing unless asked, so a batch of two
    lengths comes back padded to the longer; Whisper's encoder reads a fixed
    30-second window, so even a short clip comes back at its 3000 frames."""
    from dew.data import AutoAudioProcessor

    values = AutoAudioProcessor(modelname=str(towers / "wav2vec2"))([_tone(440, 0.025), _tone(440, 0.05)])
    features = AutoAudioProcessor(modelname=str(towers / "whisper"))(_tone(440, 0.05))

    assert values["input_values"].dtype == np.float32 and values["input_values"].shape == (2, 800)
    assert features["input_features"].shape == (1, 8, 3000)


@pytest.mark.parametrize("name, key", [("wav2vec2", "input_values"), ("whisper", "input_features")])
def test_the_audio_tower_encodes_what_transformers_computes(towers, name, key):
    """The states are transformers' own, from the extractor's input the model
    reads. Every waveform is brought to the encoder's length first: a short
    one padded with silence, a record's read under `audio`."""
    from transformers import AutoFeatureExtractor, AutoModel

    encoder = HFAudio.from_pretrained(str(towers / name), seconds=0.25)
    short = _tone(440, 0.1)
    tokens = encoder.tokenize([short, {"audio": _tone(220, 0.25)}])
    context = jax.jit(encoder.encode)(encoder.params, tokens)

    reference = AutoModel.from_pretrained(str(towers / name))
    reference = reference.get_encoder() if reference.config.is_encoder_decoder else reference
    padded = np.concatenate([short, np.zeros(int(0.15 * RATE), np.float32)])
    features = AutoFeatureExtractor.from_pretrained(str(towers / name))(
        [padded, _tone(220, 0.25)], sampling_rate=RATE, return_tensors="pt")[key]
    with torch.no_grad():
        expected = reference(**{key: features}).last_hidden_state.numpy()
    assert list(tokens) == [key]
    np.testing.assert_allclose(np.asarray(context.hidden), expected, atol=1e-4)
    assert np.array_equal(np.asarray(context.mask), np.ones(expected.shape[:2], np.int32))


def test_an_audio_condition_rebuilt_from_its_record_encodes_the_same(towers):
    """A run's record rebuilds the tower around its saved parameters, built
    without weights of its own, and it reads the same audio the same way."""
    condition = Condition(HFAudio.from_pretrained(str(towers / "wav2vec2"), seconds=0.25),
                          field="audio", unconditional=0.0)
    saved = jax.tree.map(lambda leaf: np.asarray(leaf) * 2, condition.encoder.params)
    rebuilt = Condition.from_json(condition.to_json(), params=saved)
    tokens = condition.encoder.tokenize([_tone(330, 0.25)])

    assert rebuilt.unconditional == 0.0 and rebuilt.encoder.seconds == 0.25
    np.testing.assert_array_equal(np.asarray(rebuilt.encoder.encode(rebuilt.encoder.params, tokens).hidden),
                                  np.asarray(condition.encoder.encode(saved, tokens).hidden))


def _clip(path, hertz, frames=20, side=16):
    """A lossless clip of flat frames whose brightness and audio pitch follow
    `hertz`. AVI, where ffmpeg's seek into the PCM track is sample-exact; in
    Matroska a window can start a packet late and come back short."""
    from moviepy.config import FFMPEG_BINARY

    pixels = np.full((frames, side, side, 3), hertz // 4 % 256, np.uint8)
    path.with_suffix(".rgb").write_bytes(pixels.tobytes())
    path.with_suffix(".pcm").write_bytes((_tone(hertz, frames / 25) * 32767).astype(np.int16).tobytes())
    subprocess.run([FFMPEG_BINARY, "-y", "-loglevel", "error",
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{side}x{side}", "-r", "25",
                    "-i", str(path.with_suffix(".rgb")),
                    "-f", "s16le", "-ar", str(RATE), "-ac", "1", "-i", str(path.with_suffix(".pcm")),
                    "-c:v", "ffv1", "-c:a", "pcm_s16le", str(path)], check=True)


def test_an_audio_conditioned_video_run_learns_and_samples_from_audio(towers, tmp_path):
    """A run config selects the audio condition over a video dataset's clips:
    the clips' audio reaches the model through the dataset's own `audio`
    field, training on it lowers the loss, and sampling follows the audio
    it is handed.

    The loss is the denoising loss over drawn noise levels and noise, so it
    is scored before and after training on the same 64 draws, and the drop
    is held to a test rather than a share: training lowered the expected
    loss when the paired drops' t statistic clears Student's t with 63
    degrees of freedom at one in a billion, 7.0
    (scipy.stats.t.isf(1e-9, 63)). A run that learns nothing scores near 0;
    this one scored 15.9, on the pixel regime's sigmas a pixel run draws."""
    pytest.importorskip("moviepy.config", reason="needs the av extra")
    for hertz in range(220, 1100, 110):
        _clip(tmp_path / f"{hertz}.avi", hertz)
    data = LocalVideos(path=str(tmp_path), extensions=(".avi",), frame_size=8, frames=2,
                       audio_padding=1, audio_model=str(towers / "wav2vec2"), val_batches=0,
                       loading=Loading(workers=0))
    config = DiffusionRunConfig(
        model=ModelConfig("video_dit", {"patch_size": 4, "emb_features": 32, "num_layers": 1,
                                        "num_heads": 2, "dtype": "float32", "attention_impl": "reference"}),
        data=data, text=None, audio=AudioCondition(), sampling_steps=4, guidance=CFG(2.0),
        ema_decay=None, val_metrics=())
    objective = config.build()
    assert objective.inputs.conditions["textcontext"].encoder.seconds == data.audio_seconds
    assert DiffusionRunConfig.from_dict(config.to_dict()) == config
    batch = next(data.load(batch=8, tokenize=objective.inputs.tokenize).train(DataPartition()))
    assert set(batch) == {"video", "audio"}

    keys = jax.random.split(jax.random.PRNGKey(7), 64)
    scored = jax.jit(lambda variables, key: objective.loss(
        variables, batch, Step(jnp.asarray(0), key, None))[0].total)

    def losses(variables):
        return np.asarray([float(scored(variables, key)) for key in keys])

    before = losses(objective.init(jax.random.PRNGKey(0)))
    state = Trainer(objective, optax.adam(3e-3), key=jax.random.PRNGKey(0)).fit(
        data.load(batch=8, tokenize=objective.inputs.tokenize), steps=40, log_every=100)
    drop = before - losses(state.variables)
    statistic = drop.mean() / (drop.std(ddof=1) / np.sqrt(drop.size))
    assert statistic > 7.0, (statistic, drop.mean())

    pipe = objective.pipeline(state, ema=False)
    low, high = (pipe([{"audio": _tone(hertz, data.audio_seconds)}], key=0).host().images
                 for hertz in (220, 880))
    assert low.shape == (1, 2, 8, 8, 3)
    assert np.abs(low - high).max() > 1e-3


def test_audio_conditioning_is_refused_beside_text_or_without_clip_audio():
    """The audio takes the models' one context keyword, so a run naming text
    too is refused, and so is a dataset whose samples carry no audio."""
    from dew.data import TFDSImages

    with pytest.raises(ValueError, match="set text to None"):
        DiffusionRunConfig(data=LocalVideos(path="clips"), audio=AudioCondition())
    with pytest.raises(ValueError, match="carries none"):
        DiffusionRunConfig(data=TFDSImages(), text=None, audio=AudioCondition())
