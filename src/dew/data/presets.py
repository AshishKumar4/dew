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
from dew.data.pipeline import Configured, DeviceStep, Limit, Pack, Repeat, Shuffle, Slice, Windows

if TYPE_CHECKING:
    from collections.abc import Iterator

    from transformers import PreTrainedTokenizerBase

    from dew.data.cache import Cache
    from dew.data.generation import Generation
    from dew.data.pipeline import Dataset, HostStep, Mixture, Phases, Pipeline, RandomStep
    from dew.data.sources.base import Source


def images(dataset: Dataset, size: int, *, batch: int,
           augment: bool | image_steps.DeviceAugment | RandomStep | tuple[RandomStep, ...] = False,
           crop_scale: tuple[float, float] = (1.0, 1.0), field: str = "image",
           staging_size: int | None = None, draft: bool = False, rejects: bool = True,
           caption: image_steps.Caption | None = None, cache: str | Cache | Generation | None = None,
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
    - `cache` (a URL, `Cache` or published `Generation`) holds the decoded and resized images and a
      fixed caption, each input's own whatever order reads it: a shuffle or repeat written
      before the preset reads the cache as it reads the inputs. Drawn captions and augmentation
      run after it.
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
    drawn = isinstance(caption, image_steps.RandomCaption)
    captioned = None if caption is None else Configured(image_steps.ConfigureCaption(caption), needs=(),
                                                        random=drawn, one_to_one=True)
    if captioned is not None and not drawn:
        prepared = prepared.map(captioned)
    prepared = _cached(prepared, cache)
    if captioned is not None and drawn:
        prepared = prepared.map(captioned)
    if device is not None:
        prepared = prepared.random_map(image_steps.AugmentationKey())
    for step in on_host:
        prepared = prepared.random_map(step)
    return _finished(prepared, batch=batch, records=records, host=host,
                     device=() if device is None else (device,))


def _cached(prepared: Dataset, cache: str | Cache | Generation | None) -> Dataset:
    """`prepared` through `cache` where one is given (`Dataset.cache`), which holds what its fixed
    steps make of each input."""
    return prepared if cache is None else prepared.cache(cache)


def _finished(prepared: Dataset, *, batch: int, records: int | None, host: tuple[HostStep, ...],
              device: tuple[DeviceStep, ...] = ()) -> Pipeline:
    """`prepared` cut to `records` where given, in batches of `batch`, read by the `host` steps and
    then the `device` steps, in that order (a device step runs in the compiled step, after every
    host step). Every preset ends through it."""
    kept = prepared if records is None else prepared.take(records)
    return kept.batch(batch).map(*host, *device) if host or device else kept.batch(batch)


def lm(dataset: Dataset, tokenizer: str | PreTrainedTokenizerBase | None, *, seq_len: int, batch: int,
       field: str = "text", eos: bool | None = None,
       pack: Literal["first-fit", "best-fit", "concat"] | None = "first-fit", bins: int = 32,
       stride: int | None = None, cache: str | Cache | Generation | None = None,
       records: int | None = None, host: tuple[HostStep, ...] = ()) -> Pipeline:
    """Documents in `field` tokenized, ended by the tokenizer's EOS where it has one (`eos=True`
    always, `False` never), cut into pieces that keep every next-token target, and packed into
    rows of `seq_len + 1` tokens (the last a target) by `pack`, first-fit with `bins` open rows,
    best-fit, or concatenated, in batches of `batch` rows (`records` and `host` as `images` takes
    them). `pack=None` joins the documents into one stream of tokens instead and cuts it into
    windows `stride` apart (`windows`), which carry no document coordinates. With
    `tokenizer=None` the documents are token ids already, under "text", such as a token store's,
    whose windows a store reads by index (`TokenFiles(url).pipe(presets.lm, None, seq_len=512,
    batch=8, pack=None)`); a store packed must state where its documents end. `cache` holds the
    tokenized documents, as `images` holds its images."""

    if pack is not None and stride is not None:
        raise ValueError("stride places fixed windows, pack=None; a pack lays documents out whole")
    documents = _cached(dataset if tokenizer is None else dataset.map(
        text_steps.tokenize(tokenizer, field=field, eos=eos)), cache)
    if pack is None:
        windows = documents.windows(seq_len, stride=stride)
    else:
        if tokenizer is None:
            _documented(dataset)
        windows = _next_token_windows(documents, Pack(seq_len + 1, bins=bins, policy=pack))
    return _finished(windows, batch=batch, records=records, host=host)


def token_windows(dataset: Dataset, *, seq_len: int, batch: int, stride: int | None = None,
                  records: int | None = None, host: tuple[HostStep, ...] = ()) -> Pipeline:
    """A token store's stream of tokens (`TokenFiles`) cut into windows of `seq_len` inputs and their
    targets, `stride` apart (`seq_len` where None), read by index: each window a record of the
    store's, whose rows and valid targets it states, in batches of `batch` (`records` and `host` as
    `images` takes them). A shuffle, repeat or take written before it orders and selects the
    windows; its holdout takes the store's own split (`holdout(split="val")`)."""
    from dew.data.sources.tokens import TokenFiles

    chain = dataset._chain()
    store = chain.source
    if not isinstance(store, TokenFiles):
        raise ValueError(f"token_windows reads a token store's stream (TokenFiles), not "
                         f"{type(store).__name__}")
    views = (Shuffle, Repeat, Slice, Limit)
    stages = [type(stage).__name__ for stage in chain.stages if not isinstance(stage, views)]
    if stages:
        raise ValueError(f"token_windows orders the windows of {store.describe()}, which {stages} would "
                         "change first; pipe the store itself, a shuffle, repeat or take of it at most")
    # Windows written first on the store read its stream by index; the views then order them.
    windows = dataclasses.replace(chain, stages=(Windows(seq_len, stride=stride), *chain.stages))
    return _finished(windows, batch=batch, records=records, host=host)


def sft(dataset: Dataset, tokenizer: str | PreTrainedTokenizerBase, *, seq_len: int, batch: int,
        messages: str = "messages", long: Literal["chunk", "truncate", "drop"] = "chunk",
        pack: Literal["first-fit", "best-fit", "concat"] = "first-fit", bins: int = 32,
        cache: str | Cache | Generation | None = None, records: int | None = None,
        host: tuple[HostStep, ...] = ()) -> Pipeline:
    """Conversations in `messages` rendered through the tokenizer's chat
    template, each token's speaker kept beside it for an assistant-only loss,
    and packed into rows of `seq_len + 1` tokens by `pack`, as `lm` packs,
    in batches of `batch` rows (`records` and `host` as `images` takes
    them). A conversation longer than a row is cut into pieces that keep
    every turn (`long="chunk"`), cut at its end (`"truncate"`), or dropped
    and counted (`"drop"`). `cache` holds the rendered conversations, as `images` holds its
    images."""

    rendered = _cached(dataset.map(chat.template(tokenizer, messages=messages)), cache)
    packing = Pack(seq_len + 1, _CHAT_FIELDS, bins=bins, policy=pack)
    windows = _conversation_windows(rendered, packing, long=long)
    return _finished(windows, batch=batch, records=records, host=host)


def _documented(dataset: Dataset) -> None:
    """Refuse to pack token documents from a store that states no document ends (neither an index
    of them nor an EOS token), wherever it sits in the chain's sources: its tokens would read as
    one document, whose pieces no producer made."""
    from dew.data.sources.tokens import TokenFiles

    for source in _stores(dataset._chain().source):
        if isinstance(source, TokenFiles) and not source.documented:
            raise ValueError(f"packed text needs producer document boundaries or EOS: {source.describe()}")


def _stores(source: Source | Mixture | Phases) -> Iterator[Source]:
    """Every store under `source`: a mixture's or phases' corpora's, a concatenation's parts, and
    the store each holdout selection selects from."""
    from dew.data.holdout import selecting
    from dew.data.pipeline import Mixture, Phases
    from dew.data.sources.concatenate import Concatenate

    if isinstance(source, (Mixture, Phases)):
        for child in source.pipelines:
            yield from _stores(child.source)
        return
    for selected in selecting(source):
        if isinstance(selected, Concatenate):
            for part in selected.sources:
                yield from _stores(part)
        elif not isinstance(selected, (Mixture, Phases)):
            yield selected


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
    # A concatenating pack cuts conversations where its rows end, so its cuts overlap by one token too.
    pack = dataclasses.replace(pack, overlap=1) if pack.policy == "concat" else pack
    return (rendered.filter_map(fitted) if long == "drop" else rendered.map(fitted))._then(pack)


def prompts(dataset: Dataset, tokenizer: str | PreTrainedTokenizerBase, *, max_prompt_len: int, batch: int,
            pad_id: int = 0, long: Literal["drop", "truncate"] = "truncate", thinking: bool | None = None,
            reward_bytes: int | None = None, records: int | None = None,
            host: tuple[HostStep, ...] = ()) -> Pipeline:
    """Prompt rows (verl's columns: "prompt", "data_source", "ground_truth", ...) rendered to
    `max_prompt_len` left-padded ids, keeping a longer prompt's tail, or dropping it (`long="drop"`),
    with their reward columns padded to `reward_bytes`, in batches of `batch` (`records` and `host`
    as `images` takes them). Without `reward_bytes`, a memory read measures it over each store, the
    widest reward string it holds, so every row fits whole; another read needs it given, or stated
    by the store's metadata."""
    rendered = _rendered_prompts(dataset, tokenizer, max_prompt_len, pad_id=pad_id, long=long,
                                 thinking=thinking, reward_bytes=reward_bytes)
    return _finished(rendered, batch=batch, records=records, host=host)


def _rendered_prompts(dataset: Dataset, tokenizer: str | PreTrainedTokenizerBase, max_prompt_len: int, *,
                      pad_id: int, long: Literal["drop", "truncate"], thinking: bool | None,
                      reward_bytes: int | None) -> Pipeline:
    """Prompt rows rendered with their reward columns `reward_bytes` wide, or as wide as measured over
    each store (`prompts`). `prompts` and the prompt recipe (`dew.config.data.PromptData`) share it."""
    from dew.data.prompts import RewardWidth
    from dew.data.steps.chat import ConfigurePrompt
    from dew.data.text import tokenizer_name

    configure = ConfigurePrompt(tokenizer_name(tokenizer), max_prompt_len, pad_id, long, thinking)
    if reward_bytes is not None:
        return dataset.filter_map(configure({"reward_bytes": reward_bytes}))
    return dataset.map(Configured(configure, needs=("reward_bytes",), measures=RewardWidth()))


def preferences(dataset: Dataset, *, seq_len: int, batch: int, pad_id: int = 0, records: int | None = None,
                host: tuple[HostStep, ...] = ()) -> Pipeline:
    """Preference pairs of token ids, chosen and rejected, each laid out in `seq_len` tokens with
    its completion's mask, padded with `pad_id` (`preferences.PreferencePair`), in batches of
    `batch` (`records` and `host` as `images` takes them). No tokenizer: a pair of raw text is
    tokenized by a step before it."""
    from dew.data.preferences import PreferencePair

    return _finished(dataset.map(PreferencePair(seq_len, pad_id=pad_id)), batch=batch, records=records,
                     host=host)


def canvas(dataset: Dataset, tokenizer: str | PreTrainedTokenizerBase, *, prompt_length: int,
           canvas_size: int, batch: int, canvases: int = 1, pad_id: int = 0, messages: str = "messages",
           records: int | None = None, host: tuple[HostStep, ...] = ()) -> Pipeline:
    """Conversations in `messages` rendered through the tokenizer's chat template, then each
    assistant turn one row of a block-diffusion objective (`ResponseCanvases`): what comes before
    it in `prompt_length` tokens, padded with `pad_id`, and its response in `canvases` canvases of
    `canvas_size`, the tokenizer's EOS filling the rest of its last canvas and `pad_id` the canvases
    after it; a longer prompt or response keeps its first tokens. Never packed, in batches of
    `batch` (`records` and `host` as `images` takes them). A tokenizer with no EOS cannot fill a
    canvas, and is refused."""
    from dew.data.steps.chat import ResponseCanvases

    renderer = chat.template(tokenizer, messages=messages)
    eos = renderer.vocabulary.eos_id
    if eos is None:
        raise ValueError(f"tokenizer {renderer.tokenizer!r} declares no eos token to end a response's canvas")
    layout = ResponseCanvases(prompt_length, canvas_size, canvases, eos, pad_id)
    return _finished(dataset.map(renderer).flat_map(layout), batch=batch, records=records, host=host)


def clips(dataset: Dataset, seconds: float, *, frames: int, fps: float, size: int, batch: int,
          sample_rate: int | None = None, margin: float = 0.0, context: float = 0.0,
          missing: Literal["silence", "reject"] = "silence", rejects: bool = True, records: int | None = None,
          host: tuple[HostStep, ...] = ()) -> Pipeline:
    """Windows of `seconds` drawn from the media in `mp4`, each record's own
    on every read and resume (`video.PickClip`), as `frames` frames at `fps`
    fitted into `size` squares, in batches of `batch`. With `sample_rate`,
    the window's sound beside them, mono at that rate, filled with silence
    marked invalid where the media has none (`audio.DecodeAudio`), or with
    `missing="reject"` dropped. Media shorter than the window, or that does not decode, is
    counted and dropped, or with `rejects=False` stops the run. A window is drawn `margin`
    seconds clear of the media's ends, and its sound is `context` seconds wider on each side. The
    media and its window are read here and leave the record. `records` and `host` as `images`
    takes them."""

    windows = _clips(dataset, seconds, frames=frames, fps=fps, size=size, margin=margin,
                     sample_rate=sample_rate, context=context, missing=missing, rejects=rejects)
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
