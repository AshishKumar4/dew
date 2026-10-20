"""The preference, chat-canvas and clip presets prepare what their recipes do, from their settings."""

import dataclasses
from pathlib import Path

import pytest
from test_data import _voxceleb_tree
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
    """A tokenizer that declares no EOS cannot end a response's canvas, as the recipe says."""
    conversations = Memory([{"messages": [{"role": "user", "content": "hi"},
                                          {"role": "assistant", "content": "ok"}]}])
    with pytest.raises(ValueError):
        conversations.pipe(presets.canvas, "byte", prompt_length=8, canvas_size=4, batch=1)


@dataclasses.dataclass(frozen=True)
class Listed:
    """A listed clip's path as the decoder reads it, with the caption the video recipe gives."""

    caption: str

    def __call__(self, row):
        return {"mp4": row["path"], "caption": self.caption}


def test_the_clips_preset_prepares_what_the_video_recipe_does(tmp_path, monkeypatch):
    """Clips under a directory, four frames of 16 pixels and the sound a frame wider each side at
    16 kHz, every listed clip required to decode, as the video recipe reads them, by index."""
    from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor

    from dew.data import Files
    from dew.data.sources.av_utils import FPS

    _voxceleb_tree(tmp_path)
    monkeypatch.setattr(AutoFeatureExtractor, "from_pretrained",
                        lambda name: Wav2Vec2FeatureExtractor(sampling_rate=16000))
    recipe = VideoData(source=str(tmp_path / "train"), caption="a speaker", frames=4, frame_size=16,
                       audio_padding=1, extensions=(".avi",))
    expected = loader(recipe, batch=2, read="indexed", loading=LOADING)
    clips = Files(str(tmp_path / "train" / "**" / "*.avi"), format="files")
    chain = (clips.map(Listed("a speaker"), identity="listed-clip@1").shuffle(0).repeat()
             .pipe(presets.clips, 4 / FPS, frames=4, fps=FPS, size=16, batch=2, sample_rate=16000,
                   margin=1 / FPS, context=1 / FPS, missing="reject", rejects=False))
    built = compile(chain, read="indexed", loading=LOADING, host=recipe.pipelines()[2])
    same_batches(expected, built, 2)
