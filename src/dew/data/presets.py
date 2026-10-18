"""Dataset functions that prepare one kind of record for a model, from the public steps, ending in `batch`.

Each is applied with `pipe`, the same preset for training and validation:
`flowers.pipe(presets.images, 64, batch=16, augment=True)`. Augmentation is
off unless asked for. Fields a preset does not read pass through, stacked,
except where it packs several records into one row.

- `images(ds, size, batch=)` reads `image`, bytes or uint8 [H, W, 3], and
  writes `image`, uint8 [B, size, size, 3];
- `lm(ds, tokenizer, seq_len=, batch=)` reads `text`, a string, and writes
  `text`, int32 [B, seq_len + 1], and its windows' documents as
  `text_coordinates` (`dew.nn.inputs.Coordinates`);
- `sft(ds, tokenizer, seq_len=, batch=)` reads `messages`, chat turns, and
  writes the `lm` fields and `text_roles`;
- `clips(ds, seconds, frames=, fps=, size=, batch=)` reads `mp4`, media
  bytes or a path, and writes `video`, uint8 [B, frames, size, size, 3], and
  with `sample_rate=`, `audio`, float32 [B, samples], and `audio_valid`, bool
  [B, samples].
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from dew.data import (
    audio as audio_steps,
    chat,
    images as image_steps,
    text as text_steps,
    video as video_steps,
)
from dew.data.pipeline import Pack

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from dew.data.pipeline import Dataset, Pipeline, RandomStep


def images(dataset: Dataset, size: int, *, batch: int, augment: bool | RandomStep = False,
           crop_scale: tuple[float, float] = (1.0, 1.0)) -> Pipeline:
    """Images decoded and resized to `size` squares, in batches of `batch`.
    With `augment=True`, each batch is cropped to `crop_scale` of the area,
    flipped and colour-jittered on the accelerator, each image by a key its
    record draws (`images.AugmentationKey`), so a resumed run augments as
    the uninterrupted one would. A user random step instead runs on the
    host right after resizing."""

    resized = dataset.filter_map(image_steps.DecodeImage("image")).map(image_steps.Resize(size))
    if augment is False:
        return resized.batch(batch)
    if augment is True:
        return (resized.random_map(image_steps.AugmentationKey()).batch(batch)
                .map(image_steps.DeviceAugment(size, crop_scale=crop_scale, flip=True, jitter=True)))
    return resized.random_map(augment).batch(batch)


def lm(dataset: Dataset, tokenizer: str | PreTrainedTokenizerBase, *, seq_len: int, batch: int,
       field: str = "text") -> Pipeline:
    """Documents in `field` tokenized, ended by the tokenizer's EOS where it
    has one, cut into pieces that keep every next-token target, and packed
    first-fit into rows of `seq_len + 1` tokens (the last a target), in
    batches of `batch` rows."""

    documents = dataset.map(text_steps.tokenize(tokenizer, field=field))
    return _next_token_windows(documents, Pack(seq_len + 1)).batch(batch)


def sft(dataset: Dataset, tokenizer: str | PreTrainedTokenizerBase, *, seq_len: int, batch: int) -> Pipeline:
    """Conversations in `messages` rendered through the tokenizer's chat
    template, each token's speaker kept beside it for an assistant-only loss,
    cut to `seq_len + 1` tokens and packed first-fit into rows of that
    length, in batches of `batch` rows."""

    return _conversation_windows(dataset.map(chat.template(tokenizer)), Pack(seq_len + 1, _CHAT_FIELDS),
                                 long="truncate").batch(batch)


_CHAT_FIELDS = ("text", chat.ROLES_KEY)
"""The per-token fields a rendered conversation packs: its ids and each one's speaker."""


def _next_token_windows(documents: Dataset, pack: Pack) -> Pipeline:
    """Token documents cut into pieces of `pack`'s length overlapping by one token, so each token's
    next-token target stays in one piece, then packed by `pack`. `lm`, a conversation cut into
    pieces and the text and chat recipes (`dew.config.data`) prepare text through it."""
    return documents.flat_map(text_steps.Chunks(pack.seq_len, fields=pack.fields, overlap=1)).then(pack)


def _conversation_windows(rendered: Dataset, pack: Pack, *,
                          long: Literal["truncate", "drop", "chunk"]) -> Pipeline:
    """Rendered conversations (`chat.template`) packed by `pack` with their speakers: one longer
    than a window cut at its end, dropped and counted, or cut into pieces (`_next_token_windows`),
    as `long` says. `sft` and the chat recipe (`dew.config.data.ChatData`) share it."""
    if long == "chunk":
        return _next_token_windows(rendered, pack)
    fitted = chat.fit_length(pack.seq_len, long=long)
    return (rendered.filter_map(fitted) if long == "drop" else rendered.map(fitted)).then(pack)


def clips(dataset: Dataset, seconds: float, *, frames: int, fps: float, size: int, batch: int,
          sample_rate: int | None = None) -> Pipeline:
    """Windows of `seconds` drawn from the media in `mp4`, each record's own
    on every read and resume (`video.PickClip`), as `frames` frames at `fps`
    fitted into `size` squares, in batches of `batch`. With `sample_rate`,
    the window's sound beside them, mono at that rate, filled with silence
    marked invalid where the media has none (`audio.DecodeAudio`). Media shorter
    than the window, or that does not decode, is counted and dropped; the
    media and its window are read here and leave the record."""

    picked = (dataset.random_filter_map(video_steps.PickClip(seconds))
              .filter_map(video_steps.DecodeVideo(frames, fps=fps, size=size)))
    if sample_rate is not None:
        picked = picked.filter_map(audio_steps.DecodeAudio(sample_rate, missing="silence"))
    return picked.remove_columns("mp4", "clip").batch(batch)
