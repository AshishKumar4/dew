"""Dataset functions that prepare one kind of record for a model, from the public steps, ending in `batch`.

Each is applied with `pipe`, the same preset for training and validation:
`flowers.pipe(presets.images(64, batch=16, augment=True))`. Augmentation is
off unless asked for. Fields a preset does not read pass through, stacked,
except where it packs several records into one row.

- `images(size, batch=)` reads `image`, bytes or uint8 [H, W, 3], and
  writes `image`, uint8 [B, size, size, 3];
- `lm(tokenizer, seq_len=, batch=)` reads `text`, a string, and writes
  `text`, `text_segment_ids` and `text_positions`, int32 [B, seq_len + 1];
- `sft(tokenizer, seq_len=, batch=)` reads `messages`, chat turns, and
  writes the `lm` fields and `text_roles`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from dew.data import chat, images as image_steps, text as text_steps

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from dew.data.pipeline import Dataset, Pipeline

type Preset = Callable[[Dataset], Pipeline]


def images(size: int, *, batch: int, augment: bool = False,
           crop_scale: tuple[float, float] = (1.0, 1.0)) -> Preset:
    """Images decoded and resized to `size` squares, in batches of `batch`.
    With `augment`, each batch is cropped to `crop_scale` of the area,
    flipped and colour-jittered on the accelerator, each image by a key its
    record draws (`images.augmentation_key`), so a resumed run augments as
    the uninterrupted one would."""

    def prepared(dataset: Dataset) -> Pipeline:
        resized = dataset.filter_map(image_steps.decode("image")).map(image_steps.resize(size))
        if not augment:
            return resized.batch(batch)
        return (resized.random_map(image_steps.augmentation_key()).batch(batch)
                .map(image_steps.device_augment(size, crop_scale=crop_scale, flip=True, jitter=True)))

    return prepared


def lm(tokenizer: str | PreTrainedTokenizerBase, *, seq_len: int, batch: int, field: str = "text") -> Preset:
    """Documents in `field` tokenized, ended by the tokenizer's EOS where it
    has one, cut into pieces that keep every next-token target, and packed
    first-fit into rows of `seq_len + 1` tokens (the last a target), in
    batches of `batch` rows."""

    def prepared(dataset: Dataset) -> Pipeline:
        return (dataset.map(text_steps.tokenize(tokenizer, field=field))
                .flat_map(text_steps.chunk(seq_len + 1, overlap=1))
                .pack(seq_len + 1).batch(batch))

    return prepared


def sft(tokenizer: str | PreTrainedTokenizerBase, *, seq_len: int, batch: int) -> Preset:
    """Conversations in `messages` rendered through the tokenizer's chat
    template, each token's speaker kept beside it for an assistant-only loss,
    cut to `seq_len + 1` tokens and packed first-fit into rows of that
    length, in batches of `batch` rows."""

    def prepared(dataset: Dataset) -> Pipeline:
        return (dataset.map(chat.template(tokenizer)).map(chat.fit_length(seq_len + 1))
                .pack(seq_len + 1, fields=("text", "text_roles")).batch(batch))

    return prepared
