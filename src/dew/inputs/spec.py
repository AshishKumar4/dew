"""What a generative objective is fed: the sample field and its conditions.

`InputSpec` names the batch field the model learns to generate, and the
conditions it is given, keyed by the model's own keyword arguments. A
`Condition` holds an encoder, the batch field it reads and the raw datum that
stands for "no condition", which classifier-free guidance and conditioning
dropout substitute. The spec only describes the inputs; `encode_conditions`
runs its encoders over their weights, the one encoding a training step and a
sampling task share.

Image and video batches arrive as uint8 pixels in [0, 255], as the data
workers write them. `unit_range` is the one conversion to [-1, 1], the range
every diffusion loss, sample, artifact and image metric uses, and
`dew.artifacts.uint8_pixels` is the one conversion back.
"""


from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from dew import _records, registry
from dew.batch import Batch, Variables
from dew.inputs.condition import ConditionEncoder
from dew.nn.inputs import ModelInputs, media_slots
from dew.nn.precision import at_least_fp32, divided
from dew.nn.text_checkpoints import DEFAULT_MODEL
from dew.registry import DtypeName

if TYPE_CHECKING:
    from flax import linen as nn

    from dew.nn.blocks import TokenTable
    from dew.nn.wrapping import HeadedModel


def unit_range(pixels: jax.typing.ArrayLike) -> jax.Array:
    """Convert pixels in [0, 255] to [-1, 1].

    uint8 and float32 pixels give float32, and a float64 run's float64 pixels
    stay float64.
    """
    pixels = jnp.asarray(pixels)
    # IEEE division: XLA takes a division by a constant as a product with its rounded reciprocal.
    return divided(pixels.astype(at_least_fp32(pixels.dtype)) - 127.5, 127.5)


class OutsideTable(ValueError):
    """A token id outside the embedding table it indexes (`Field.bounded`, `Field.admits`), which reads no row
    of it: `holder`, what holds the id, the `token`, the table's `rows`, and the token's spelling where a
    processor knows it. Its arguments are its `args`, so it crosses a process as itself (a pickle rebuilds
    an exception from them), its message read off them."""

    def __init__(self, holder: str, token: int, rows: int, spelled: str | None = None):
        super().__init__(holder, token, rows, spelled)
        self.holder, self.token, self.rows, self.spelled = holder, token, rows, spelled

    def __str__(self) -> str:
        named = "" if self.spelled is None else f" ({self.spelled!r})"
        return (f"{self.holder} holds the id {self.token}{named}, outside the {self.rows} rows of the table "
                "it indexes: its tokens come from another vocabulary than the model's")


@dataclass(frozen=True)
class Field:
    """A batch field and its per-example shape, such as `Field("image", (128, 128, 3))`."""

    key: str
    shape: tuple[int, ...]
    variable_axes: tuple[int, ...] = ()
    table: TokenTable | None = dataclasses.field(default=None, metadata={"record": False})
    """The table the field's integers index, a token field's its model's embedding table
    (`dew.nn.blocks.token_table`), read off the model where an objective declares the field: its rows,
    the ids its model replaces wherever they occur and whether it replaces the id at each slot a call's
    media fill, as `bounded` holds them; None for a field that indexes nothing. Not recorded: the model a
    run rebuilds states it."""

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(int(size) for size in self.shape))
        if any(axis < 0 or axis >= len(self.shape) for axis in self.variable_axes):
            raise ValueError("variable field axes must index its per-sample shape")

    @classmethod
    def indexing(cls, key: str, shape: tuple[int, ...], model: nn.Module | HeadedModel, *,
                 variable_axes: tuple[int, ...] = ()) -> Field:
        """The field `key` of `shape` whose integers index `model`'s embedding table
        (`dew.nn.blocks.token_table`), none where it states no table."""
        from dew.nn.blocks import token_table

        return cls(key, shape, variable_axes, token_table(model))

    def bounded(self, value: object, spelling: Callable[[int], str] | None = None) -> None:
        """Refuse `value`, this field's rows (a `ModelInputs`, or bare token rows), where it holds an id
        outside its table's rows its model reads from the table: not one it replaces wherever it occurs,
        nor at a slot a `ModelInputs`'s media fill where its model replaces those (`TokenTable.media`,
        `dew.nn.inputs.media_slots`). Such an id reads no row of the table, whose gather fills it with NaN
        (`jnp.take`'s out-of-bounds fill), a run's loss and a served token NaN with finite logits beside
        it. Names the id, its token where `spelling` knows it, and the table's rows. Rows a caller gave
        already placed (a batch of device arrays) are bounded where they lie, their least and greatest id
        read back alone (two scalars, outside any step), and read back whole only to name a refused id;
        host rows are bounded on the host."""
        table = self.table
        if table is None:
            return
        held = value.tokens if isinstance(value, ModelInputs) else value
        placed = isinstance(held, jax.Array)
        xp = jnp if placed else np
        slots = (media_slots(value.token_fields, value.tokens.shape, xp=xp)
                 if table.media and isinstance(value, ModelInputs) else None)
        if placed:
            if held.size == 0 or not jnp.issubdtype(held.dtype, jnp.integer):
                return
            # The ids its model replaces counted as a row's, so a valid batch reads back two scalars alone.
            kept = (held if not table.replaced
                    else jnp.where(jnp.isin(held, jnp.asarray(sorted(table.replaced), held.dtype)), 0, held))
            kept = kept if slots is None else jnp.where(slots, 0, kept)
            low, high = (int(bound) for bound in jax.device_get((jnp.min(kept), jnp.max(kept))))
            if low >= 0 and high < table.rows:
                return
        ids = np.asarray(held)
        if ids.size == 0 or ids.dtype.kind not in "iu":
            return
        if slots is not None:
            ids = np.where(np.asarray(slots), 0, ids)
        low, high = int(ids.min()), int(ids.max())
        if low >= 0 and high < table.rows:
            return
        outside = np.unique(ids[(ids < 0) | (ids >= table.rows)])
        outside = outside[~np.isin(outside, list(table.replaced))] if table.replaced else outside
        if outside.size == 0:
            return
        token = int(outside[0] if outside[0] < 0 else outside[-1])
        raise OutsideTable(f"field {self.key!r}", token, table.rows,
                           None if spelling is None or token < 0 else spelling(token))

    def admits(self, token: int) -> bool:
        """Whether `token` reads a row of the table this field indexes, or is an id its model replaces
        before the table reads it; any id where the field indexes no table."""
        table = self.table
        return table is None or 0 <= token < table.rows or token in table.replaced

    @classmethod
    def of(cls, key: str, batch: Batch) -> Field:
        """The field `key` at the per-example shape `batch` holds under it, a `ModelInputs`'s its token
        rows', read from shapes alone."""
        value = batch[key]
        return cls(key, tuple(np.shape(value.tokens if isinstance(value, ModelInputs) else value)[1:]))

    def held(self, batch: Batch) -> tuple[int, ...]:
        """The per-example shape `batch` holds under this field (`of`)."""
        return Field.of(self.key, batch).shape


@dataclass(frozen=True)
class Condition:
    """Names one conditioning input: its encoder, the batch field holding its
    tokens, and the raw datum for the unconditional branch."""

    encoder: ConditionEncoder
    field: str = "text"
    unconditional: str | float | Mapping[str, object] = ""

    @staticmethod
    def bound(record: Mapping[str, object], *, params: Variables | None = None) -> Condition:
        """The condition a run recorded (`dew.registry.to_record`), its encoder rebuilt around `params`
        where given (`ConditionEncoder.bound`) and its other fields read as the registry reads them."""
        encoder = ConditionEncoder.bound(_records.record(record["encoder"], "encoder"), params=params)
        # The encoder is built, which the registry leaves as it is (`registry.from_record`).
        return registry.from_record(Condition, {**record, "encoder": encoder})


@dataclass(frozen=True)
class InputSpec:
    """Names the sample field and the conditions, keyed by the model keyword each is passed under.

    For example, `conditions={"textcontext": Condition(...)}`. Two conditions
    cannot share a batch field.

    A captioning dataset passes its text to `tokenize`. Every condition that
    reads captions tokenizes them under its own field, so the encoder a run
    names decides the ids and the context length, and the dataset only
    carries the words. A condition on another modality reads the field its
    dataset writes; audio conditioning, for example, reads a clip's `audio`.
    """

    sample: Field
    conditions: Mapping[str, Condition] = field(default_factory=dict)
    mask: Field | None = None
    """A binary image mask for explicit masked-image latent conditioning."""

    def __post_init__(self):
        fields = [condition.field for condition in self.conditions.values()]
        if len(set(fields)) != len(fields):
            raise ValueError(
                f"two conditions share the batch field {sorted(set(fields))}: "
                "the objective reads batch[condition.field] under each keyword, "
                "so the second tokenization would overwrite the first. "
                "Name a field per condition"
            )

    def tokenize(self, captions: Sequence[str]) -> dict[str, Mapping[str, np.ndarray]]:
        """Return the batch fields this run's caption conditions read, tokenized from `captions`.

        The result is empty for a run with no caption conditions, so the
        captions stop at the loader and no string array reaches a device.
        """
        return {
            condition.field: condition.encoder.tokenize(captions)
            for condition in self.conditions.values()
            if condition.encoder.reads_captions
        }

    @property
    def encoders(self) -> tuple[tuple[str, ConditionEncoder], ...]:
        """Each condition's encoder by the keyword it is passed under, a hashable value a compiled
        encoding takes as a static argument (`encode_conditions`)."""
        return tuple((keyword, condition.encoder) for keyword, condition in self.conditions.items())

    def dependencies(self) -> Mapping[str, object]:
        """What `tokenize` and `encode_fields` make of a batch depends on: each condition's field
        and its encoder's own (`ConditionEncoder.dependencies`)."""
        return {keyword: {"field": condition.field, "encoder": condition.encoder.dependencies()}
                for keyword, condition in self.conditions.items()}

    def encode_fields(self, batch: Batch) -> dict[str, Mapping[str, np.ndarray]]:
        """Return the batch fields this run's other conditions read, each encoded from the rows its
        field holds: a clip's `audio` waveforms become the audio model's features, as `tokenize`
        makes ids of captions, so the dataset carries the sound and the run's encoder featurizes it."""
        return {
            condition.field: condition.encoder.tokenize(list(batch[condition.field]))
            for condition in self.conditions.values()
            if not condition.encoder.reads_captions
        }

    def check(self, batch: Batch, *, whole: bool = True) -> None:
        """Raise `ValueError` if `batch` is missing a field this spec names or has a wrong shape.

        The sample and the mask must have their declared per-example shapes.
        A `ModelInputs` field's shape is taken from its token rows. A batch that
        need not be `whole`, a validation split's that an objective may score
        from fewer fields (a rollout's prompts alone), is held to the shapes of
        the fields it carries, but for an inpainting spec's mask and the sample
        it masks, which every draw and loss reads (ledger 2900)."""
        for condition in self.conditions.values():
            if condition.field not in batch and whole:
                raise ValueError(
                    f"objective.inputs needs condition field {condition.field!r} in the training batch"
                )
        if self.mask is not None and self.mask.key not in batch:
            raise ValueError(f"an inpainting objective reads each image's mask, {self.mask.key!r}, in every "
                             "batch it trains on or scores: draw one with dew.data.images.RandomMask() "
                             "(images(..., augment=RandomMask()), and rejects=False for a validation split, "
                             "whose random steps draw by each record's own index), or decode a dataset's own")
        masked = self.mask is not None
        for declared in (self.sample, self.mask):
            if declared is None or (declared.key not in batch and not whole and not masked):
                continue
            if declared.key not in batch:
                raise ValueError(f"objective.inputs needs field {declared.key!r} in the training batch")
            actual = declared.held(batch)
            matches = len(actual) == len(declared.shape) and all(
                0 < size <= bound if axis in declared.variable_axes else size == bound
                for axis, (size, bound) in enumerate(zip(actual, declared.shape, strict=True))
            )
            if not matches:
                raise ValueError(
                    f"objective.inputs field {declared.key!r} declares shape {declared.shape}, "
                    f"but the first training batch has shape {tuple(actual)}"
                )

    def bounded(self, batch: Batch, spelling: Callable[[int], str] | None = None) -> None:
        """Refuse `batch`, a host batch, where a field this spec names holds an id outside the table it
        indexes (`Field.bounded`): every training batch's, on the prefetch worker before it is placed,
        and every validation batch's before it is placed; `check`, of shapes alone, reads a placed one."""
        for declared in (self.sample, self.mask):
            if declared is not None and declared.key in batch:
                declared.bounded(batch[declared.key], spelling)

    @staticmethod
    def bound(record: Mapping[str, object], *, params: Mapping[str, Variables] | None = None) -> InputSpec:
        """The spec a run recorded (`dew.registry.to_record`), each condition's encoder rebuilt around
        its parameters, `params` by the keyword it is passed under (`Condition.bound`); without
        `params` each encoder loads its own weights. The other fields are read as the registry reads
        them."""
        conditions = {keyword: Condition.bound(_records.record(condition, keyword),
                                               params=None if params is None else params[keyword])
                      for keyword, condition in _records.record(record.get("conditions", {}),
                                                                "conditions").items()}
        return registry.from_record(InputSpec, {**record, "conditions": conditions})

@dataclasses.dataclass(frozen=True)
class TextCondition:
    """Text conditioning: which registered encoder reads the batch's tokens, and from which checkpoint."""

    encoder: str = "clip_text"
    checkpoint: str = DEFAULT_MODEL
    dtype: DtypeName | None = None
    """The encoder's compute dtype; None follows the model's compute dtype."""
    param_dtype: DtypeName = "float32"
    """The storage dtype when loading source weights; supplied params keep theirs."""
    field: str = "text"
    """The batch field holding the tokenized text."""
    unconditional: str = ""
    """The prompt the unconditional branch is encoded from."""
    max_length: int | None = None
    """The token length every prompt is padded to.

    None keeps the encoder's own default, which for CLIP is the checkpoint's
    context length.
    """
    revision: str | None = None
    """The checkpoint's git revision.

    With it, a rerun conditions on the weights the run named, even after the
    branch has moved on.
    """

    def build(self, *, params: Variables | None = None,
              dtype: DtypeName | None = None) -> Condition:
        """Build the encoder over supplied params, without loading source weights or casting storage.

        `dtype` is the run's own compute dtype, which an unset `self.dtype` follows:
        the tower runs next to the model it conditions in every step, so a checkpoint
        stored in float32 is no reason to run it in float32.
        """
        fields = {name: value for name, value in
                  (("max_length", self.max_length), ("revision", self.revision))
                  if value is not None}
        return Condition(
            ConditionEncoder.bound({"class": self.encoder, "fields": {
                "checkpoint": self.checkpoint, "dtype": dtype if self.dtype is None else self.dtype,
                "param_dtype": self.param_dtype, **fields}}, params=params),
            field=self.field, unconditional=self.unconditional)


@dataclasses.dataclass(frozen=True)
class AudioCondition:
    """Audio conditioning on each clip, through an audio tower.

    `checkpoint` names the `hf_audio` tower, whose feature extractor encodes the
    waveform in the batch's `audio` field on the host.
    The dataset's clip length sets the waveform length used to encode every
    clip and the silent unconditional input, so the two always agree.
    """

    encoder: ClassVar[str] = "hf_audio"
    checkpoint: str = "facebook/wav2vec2-base-960h"
    revision: str | None = None
    """The checkpoint's git revision; None reads its default branch, as it stands when the run starts."""
    dtype: DtypeName | None = None
    """The tower's compute dtype; None follows the model's compute dtype."""
    param_dtype: DtypeName = "float32"
    """The storage dtype when loading source weights; supplied params keep theirs."""

    def build(self, seconds: float, *, params: Variables | None = None,
              dtype: DtypeName | None = None) -> Condition:
        """Build the tower over supplied params for waveforms of `seconds` (the clips' `audio_seconds`,
        `Dataset.facts`), without loading source weights or casting storage."""
        return Condition(
            ConditionEncoder.bound({"class": self.encoder, "fields": {
                "checkpoint": self.checkpoint, "seconds": seconds, "revision": self.revision,
                "dtype": dtype if self.dtype is None else self.dtype, "param_dtype": self.param_dtype}},
                params=params),
            field="audio", unconditional=0.0)


def encode_conditions(encoders: Sequence[tuple[str, ConditionEncoder]], weights: Mapping[str, Variables],
                      tokens: Mapping[str, object]) -> dict:
    """Each of `encoders` (`InputSpec.encoders`) over the tokens under its keyword, with the weights
    under its keyword: the conditions a model is passed, by keyword. The one encoding a training
    step (`DiffusionObjective.encode`) and a sampling task (`TextToImage`) run."""
    return {keyword: encoder.encode(weights[keyword], tokens[keyword]) for keyword, encoder in encoders}
