"""The preference, chat-canvas and clip presets prepare what their recipes do, from their settings."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from test_text_presets import same_batches

from dew.config.data import ChatCanvasData, PreferenceData, VideoData, loader
from dew.data import Loading, Memory, presets
from dew.data.compile import compile

CHAT = Path(__file__).resolve().parent / "fixtures" / "tokenizers" / "tiny-chat"
LOADING = Loading(workers=0, threads=1, read_buffer=2)


def test_the_preferences_preset_prepares_what_the_preference_recipe_does():
    """Chosen and rejected token ids with their completion masks, laid out in six tokens padded
    with 7, read by index."""
    pairs = Memory([{"chosen": [1, 2, 3, 4][: 2 + index % 3], "rejected": [1, 2, 5][: 2 + index % 2],
                     "chosen_mask": [0, 0, 1, 1][: 2 + index % 3],
                     "rejected_mask": [0, 0, 1][: 2 + index % 2]}
                    for index in range(12)])
    expected = loader(PreferenceData(pairs, seq_len=6, pad_id=7), batch=4, read="indexed", loading=LOADING)
    chain = pairs.shuffle(0).repeat().pipe(presets.preferences, seq_len=6, batch=4, pad_id=7)
    built = compile(chain, read="indexed", loading=LOADING)
    assert built.plan.identity() == expected.plan.identity()
    same_batches(expected, built, 3)


def test_the_canvas_preset_prepares_what_the_chat_canvas_recipe_does():
    """Conversations rendered by a chat template, each assistant turn one row of a prompt of 24
    tokens and two canvases of 8, ended by the tokenizer's EOS, read by index."""
    conversations = Memory([{"talk": [{"role": "user", "content": "say " + "more " * index},
                                      {"role": "assistant", "content": "done " * (1 + index % 5)}]}
                            for index in range(12)])
    recipe = ChatCanvasData(conversations, tokenizer=str(CHAT), prompt_length=24, canvas_size=8, canvases=2,
                            pad_id=0, column="talk")
    expected = loader(recipe, batch=2, read="indexed", loading=LOADING)
    chain = conversations.shuffle(0).repeat().pipe(presets.canvas, str(CHAT), prompt_length=24, canvas_size=8,
                                                   batch=2, canvases=2, messages="talk")
    built = compile(chain, read="indexed", loading=LOADING)
    assert built.plan.identity() == expected.plan.identity()
    same_batches(expected, built, 3)


def test_a_canvas_needs_a_tokenizer_that_ends_a_response():
    """The chat tokenizer without its EOS token cannot end a response's canvas, as the recipe says;
    with it, the same conversation lays out."""
    from transformers import AutoTokenizer

    conversations = Memory([{"messages": [{"role": "user", "content": "hi"},
                                          {"role": "assistant", "content": "ok"}]}])
    rows = list(conversations.pipe(presets.canvas, str(CHAT), prompt_length=8, canvas_size=4, batch=1))
    assert len(rows) == 1
    endless = AutoTokenizer.from_pretrained(str(CHAT))
    endless.eos_token = None
    with pytest.raises(ValueError):
        conversations.pipe(presets.canvas, endless, prompt_length=8, canvas_size=4, batch=1)


@dataclasses.dataclass(frozen=True)
class Listed:
    """A listed clip's path as the decoder reads it, with the caption the video recipe gives."""

    caption: str

    def __call__(self, row):
        return {"mp4": row["path"], "caption": self.caption}


def test_the_clips_preset_prepares_what_the_video_recipe_does(tmp_path, monkeypatch):
    """Clips whose frames and sound count up, so the drawn window and its margin show, four frames of
    16 pixels and the sound a frame wider each side at 16 kHz, every listed clip required to decode,
    as the video recipe reads them, by index."""
    from test_data_av import _mux
    from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor

    from dew.data import Files
    from dew.data.sources.av_utils import FPS

    (tmp_path / "train").mkdir()
    for index in range(6):
        _mux(tmp_path / "train" / f"{index}.avi")
    monkeypatch.setattr(AutoFeatureExtractor, "from_pretrained",
                        lambda name: Wav2Vec2FeatureExtractor(sampling_rate=16000))
    recipe = VideoData(source=str(tmp_path / "train"), caption="a speaker", frames=4, frame_size=16,
                       audio_padding=1, extensions=(".avi",))
    expected = loader(recipe, batch=2, read="indexed", loading=LOADING)
    clips = Files(str(tmp_path / "train" / "*.avi"), format="files")
    chain = (clips.map(Listed("a speaker"), identity="listed-clip@1").shuffle(0).repeat()
             .pipe(presets.clips, 4 / FPS, frames=4, fps=FPS, size=16, batch=2, sample_rate=16000,
                   margin=1 / FPS, context=1 / FPS, missing="reject", rejects=False))
    built = compile(chain, read="indexed", loading=LOADING, host=recipe.pipelines()[2])
    same_batches(expected, built, 3)


@pytest.mark.parametrize("missing", ["silence", "reject"])
def test_clips_fill_or_drop_media_without_sound_and_drop_or_stop_on_media_too_short(tmp_path, missing):
    """A clip with no sound beside one with sound: filled with silence marked invalid, or dropped and
    counted; a clip shorter than the window is dropped and counted, or stops the run with
    `rejects=False`."""
    from test_data_av import _mux

    from dew.data.sources.av_utils import FPS

    voiced, silent = _mux(tmp_path / "voiced.avi"), _mux(tmp_path / "silent.avi", sound=0)
    short = _mux(tmp_path / "short.avi", frames=2, sound=2)
    media = Memory([{"mp4": voiced}, {"mp4": silent}, {"mp4": short}])
    rows = list(media.pipe(presets.clips, 4 / FPS, frames=4, fps=FPS, size=16, batch=1, sample_rate=16000,
                           missing=missing))
    valid = [bool(np.asarray(row["audio_valid"]).any()) for row in rows]
    assert valid == ([True, False] if missing == "silence" else [True])
    with pytest.raises(ValueError):
        list(media.pipe(presets.clips, 4 / FPS, frames=4, fps=FPS, size=16, batch=1, rejects=False))
