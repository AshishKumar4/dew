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

import dataclasses
from typing import TYPE_CHECKING, Literal

from dew.data import (
    audio as audio_steps,
    chat,
    images as image_steps,
    text as text_steps,
    video as video_steps,
)
from dew.data.pipeline import Configured, DeviceStep, Pack

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from dew.data.pipeline import Dataset, HostStep, Pipeline, RandomStep


def images(dataset: Dataset, size: int, *, batch: int,
           augment: bool | image_steps.DeviceAugment | RandomStep | tuple[RandomStep, ...] = False,
           crop_scale: tuple[float, float] = (1.0, 1.0), field: str = "image",
           staging_size: int | None = None, draft: bool = False, rejects: bool = True,
           caption: image_steps.Caption | None = None,
           records: int | None = None, host: tuple[HostStep, ...] = ()) -> Pipeline:
    """Images in `field` decoded and resized to `size` squares, in batches of `batch`.

    - `augment=True` crops each image of a batch to `crop_scale` of its area, flips and
      colour-jitters it on the accelerator, by a key its record draws (`images.AugmentationKey`),
      so a resumed run augments as the uninterrupted one would; an `images.DeviceAugment(size,
      ...)` value sets those three itself. A random step, or a tuple of them, runs on the host
      right after the resize and the caption instead.
    - `staging_size` resizes to that size instead, for an augmentation that crops to `size`:
      `augment=True` and a `DeviceAugment` do, and host steps return `size` squares, as
      `images.RandomCrop(size)` and `images.RandomResizedCrop(size)` do. With no augmentation it
      is not used. `draft` decodes a JPEG at a reduced scale no smaller than that
      size (`images.DecodeImage(at_least=)`).
    - With `rejects=False`, an image that does not decode stops the run instead of being dropped
      and counted.
    - `caption` (`images.caption(...)`) writes each record's caption after the resize; its class
      names, where it names none, are the source's (`class_names.label` in its metadata), and with
      `templates=` it draws one per record.
    - `records` keeps that many prepared records before the batch; `host` steps then read each
      batch on the host, before any step on the accelerator."""

    device: image_steps.DeviceAugment | None = None
    on_host: tuple[RandomStep, ...] = ()
    if augment is True:
        device = image_steps.DeviceAugment(size, crop_scale=crop_scale, flip=True, jitter=True)
    elif isinstance(augment, image_steps.DeviceAugment):
        device = augment
    elif isinstance(augment, tuple):
        on_host = augment
    elif augment is not False:
        on_host = (augment,)
    staging = size if staging_size is None or (device is None and not on_host) else staging_size
    if size < 1 or staging < size:
        raise ValueError(f"images need a positive size and a staging_size at least that size, not {size} "
                         f"and {staging_size}")
    if device is not None and device.size != size:
        raise ValueError(f"the augmentation crops to {device.size} pixels and the images are {size}; give "
                         "DeviceAugment the preset's size")
    prepared = _decoded_images(dataset, staging, field=field, at_least=staging if draft else None,
                               rejects=rejects)
    if caption is not None:
        prepared = prepared.map(Configured(image_steps.ConfigureCaption(caption), needs=(),
                                           random=isinstance(caption, image_steps.RandomCaption)))
    if device is not None:
        prepared = prepared.random_map(image_steps.AugmentationKey())
    for step in on_host:
        prepared = prepared.random_map(step)
    return _finished(prepared, batch=batch, records=records, host=host,
                     device=() if device is None else (device,))


def _finished(prepared: Dataset, *, batch: int, records: int | None, host: tuple[HostStep, ...],
              device: tuple[DeviceStep, ...] = ()) -> Pipeline:
    """`prepared` cut to `records` where given, in batches of `batch`, read by the `host` steps and
    then the `device` steps, in that order (a device step runs in the compiled step, after every
    host step). Every preset ends through it."""
    kept = prepared if records is None else prepared.take(records)
    return kept.batch(batch).map(*host, *device) if host or device else kept.batch(batch)


def lm(dataset: Dataset, tokenizer: str | PreTrainedTokenizerBase, *, seq_len: int, batch: int,
       field: str = "text", records: int | None = None, host: tuple[HostStep, ...] = ()) -> Pipeline:
    """Documents in `field` tokenized, ended by the tokenizer's EOS where it
    has one, cut into pieces that keep every next-token target, and packed
    first-fit into rows of `seq_len + 1` tokens (the last a target), in
    batches of `batch` rows (`records` and `host` as `images` takes them)."""

    documents = dataset.map(text_steps.tokenize(tokenizer, field=field))
    windows = _next_token_windows(documents, Pack(seq_len + 1))
    return _finished(windows, batch=batch, records=records, host=host)


def sft(dataset: Dataset, tokenizer: str | PreTrainedTokenizerBase, *, seq_len: int, batch: int,
        records: int | None = None, host: tuple[HostStep, ...] = ()) -> Pipeline:
    """Conversations in `messages` rendered through the tokenizer's chat
    template, each token's speaker kept beside it for an assistant-only loss,
    cut to `seq_len + 1` tokens and packed first-fit into rows of that
    length, in batches of `batch` rows (`records` and `host` as `images` takes them)."""

    windows = _conversation_windows(dataset.map(chat.template(tokenizer)), Pack(seq_len + 1, _CHAT_FIELDS),
                                    long="truncate")
    return _finished(windows, batch=batch, records=records, host=host)


_CHAT_FIELDS = ("text", chat.ROLES_KEY)
"""The per-token fields a rendered conversation packs: its ids and each one's speaker."""


def _next_token_windows(documents: Dataset, pack: Pack) -> Pipeline:
    """Token documents packed by `pack` with every cut through one overlapping by one token, so each
    token's next-token target stays in one piece: a fitting pack's documents are cut into pieces of
    its length first, and a concatenating pack overlaps the cuts it makes itself (`Pack.overlap`).
    `lm`, a conversation cut into pieces and the text and chat recipes (`dew.config.data`) prepare
    text through it."""
    if pack.policy == "concat":
        return documents._then(dataclasses.replace(pack, overlap=1))
    return documents.flat_map(text_steps.Chunks(pack.seq_len, fields=pack.fields, overlap=1))._then(pack)


def _conversation_windows(rendered: Dataset, pack: Pack, *,
                          long: Literal["truncate", "drop", "chunk"]) -> Pipeline:
    """Rendered conversations (`chat.template`) packed by `pack` with their speakers: one longer
    than a window cut at its end, dropped and counted, or cut into pieces (`_next_token_windows`),
    as `long` says. `sft` and the chat recipe (`dew.config.data.ChatData`) share it."""
    if long == "chunk":
        return _next_token_windows(rendered, pack)
    fitted = chat.fit_length(pack.seq_len, long=long)
    return (rendered.filter_map(fitted) if long == "drop" else rendered.map(fitted))._then(pack)


def clips(dataset: Dataset, seconds: float, *, frames: int, fps: float, size: int, batch: int,
          sample_rate: int | None = None, records: int | None = None,
          host: tuple[HostStep, ...] = ()) -> Pipeline:
    """Windows of `seconds` drawn from the media in `mp4`, each record's own
    on every read and resume (`video.PickClip`), as `frames` frames at `fps`
    fitted into `size` squares, in batches of `batch`. With `sample_rate`,
    the window's sound beside them, mono at that rate, filled with silence
    marked invalid where the media has none (`audio.DecodeAudio`). Media shorter
    than the window, or that does not decode, is counted and dropped; the
    media and its window are read here and leave the record. `records` and `host` as `images`
    takes them."""

    windows = _clips(dataset, seconds, frames=frames, fps=fps, size=size, sample_rate=sample_rate,
                     missing="silence")
    return _finished(windows, batch=batch, records=records, host=host)


def _decoded_images(rows: Dataset, size: int, *, field: str = "image", at_least: int | None = None,
                    rejects: bool = True) -> Pipeline:
    """The images in `field` decoded, reduced no further than `at_least` pixels where given, and
    resized to `size` squares. With `rejects` an image that does not decode is dropped and counted,
    else it stops the run. `images` and the image recipe (`dew.config.data.ImageData`) share it."""
    decode = image_steps.DecodeImage(field, at_least=at_least, rejects=rejects)
    return (rows.filter_map(decode) if rejects else rows.map(decode)).map(image_steps.Resize(size))


def _clips(media: Dataset, seconds: float, *, frames: int, fps: float, size: int, margin: float = 0.0,
           sample_rate: int | None = None, context: float = 0.0, missing: Literal["reject", "silence"],
           rejects: bool = True) -> Pipeline:
    """Windows of `seconds` drawn `margin` clear of the media's ends in `mp4`, each record's own on
    every read and resume, as `frames` frames at `fps` in `size` squares, and with `sample_rate`
    the window's sound `context` seconds wider (`missing` says what media without sound gives).
    The media and its window leave the record. With `rejects` media too short or that does not
    decode is dropped and counted, else it stops the run. `clips` and the video recipe
    (`dew.config.data.VideoData`) share it."""
    pick = video_steps.PickClip(seconds, margin=margin, rejects=rejects)
    decoders: list[video_steps.DecodeVideo | audio_steps.DecodeAudio] = [
        video_steps.DecodeVideo(frames, fps=fps, size=size, rejects=rejects)]
    if sample_rate is not None:
        decoders.append(audio_steps.DecodeAudio(sample_rate, context=context, missing=missing,
                                                rejects=rejects))
    picked = media.random_filter_map(pick) if rejects else media.random_map(pick)
    decoded = picked.filter_map(*decoders) if rejects else picked.map(*decoders)
    return decoded.remove_columns("mp4", "clip")
