"""One reader per JSON type, for every config a published file hands over.

A decoder config, a scheduler file, a generation config and a processor file
all arrive from `json.loads`, so every field is an unnarrowed value that has
to be tested before the model is built from it. These are the tests. Each one
names the key and shows the value it refused, so a file that disagrees with
the model it claims to describe says which field disagreed.

This module sits under `dew.nn`, `dew.interop` and `dew.diffusion`, which is
why it is here and not beside the `json_value` encoder in
`dew.telemetry.records`: reading a config is not a telemetry concern, and
that encoder runs the other direction, turning a finished run record into
JSON. The `JSON` alias both directions speak is declared here, once.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import datetime
import functools
import json
import math
import re
import typing
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from numbers import Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

type JSON = None | bool | int | float | str | list['JSON'] | dict[str, 'JSON']



def duration(value: str) -> datetime.timedelta:
    """A positive recorded duration, in seconds, minutes or hours."""
    match = re.fullmatch(r'(\d+(?:\.\d+)?)(s|m|h)', value)
    if match is None:
        raise ValueError("duration must be positive, such as 30m (s, m and h are supported)")
    whole, _, fraction = match[1].partition('.')
    denominator = 10 ** len(fraction)
    numerator = (
        (int(whole) * denominator + int(fraction or "0")) * {"s": 1, "m": 60, "h": 3600}[match[2]] * 1_000_000
    )
    micros, remainder = divmod(numerator, denominator)
    # timedelta's nearest-microsecond, ties-to-even rounding, without a
    # floating conversion or ambient decimal context.
    if remainder * 2 > denominator or (remainder * 2 == denominator and micros % 2):
        micros += 1
    if micros <= 0:
        raise ValueError("duration must be positive and representable in microseconds")
    return datetime.timedelta(microseconds=micros)


def is_share(value: int | str | datetime.timedelta | None) -> bool:
    """Whether `value`, a cadence or warmup as written, is a share of a run, such as `5%` (`share`)."""
    return isinstance(value, str) and value.endswith("%")


def share(value: str) -> float:
    """The fraction of a run a share such as `5%` or `0.5%` names: above none of it, at most all."""
    match = re.fullmatch(r'(\d+(?:\.\d+)?)%', value)
    fraction = None if match is None else float(match[1]) / 100
    if fraction is None or not 0 < fraction <= 1:
        raise ValueError(f"a share of a run is a percentage above 0 and at most 100, such as 5%, not "
                         f"{value!r}")
    return fraction


def recorded_duration(value: datetime.timedelta) -> str:
    """One canonical spelling, preserving timedelta's microsecond precision."""
    micros = (value.days * 86400 + value.seconds) * 1_000_000 + value.microseconds
    if micros <= 0:
        raise ValueError("duration must be positive")
    if micros % 3_600_000_000 == 0:
        return f'{micros // 3_600_000_000}h'
    if micros % 60_000_000 == 0:
        return f'{micros // 60_000_000}m'
    seconds, fraction = divmod(micros, 1_000_000)
    tail = f'.{fraction:06d}'.rstrip('0') if fraction else ''
    return f'{seconds}{tail}s'


def integer(value: object, key: str) -> int:
    """One integer field. `True` is an int to Python and a flag to a config."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key}={value!r}: this field is an integer")
    return value


def finite(value: object, what: str) -> float:
    """One finite real as a float, NumPy's scalars among them: a reward, a likelihood, a scale.
    A bool is a flag, not a number; it, every non-number, an infinity and a nan are refused with
    ValueError naming `what` and the value."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{what} is a finite number, got {value!r}")
    return float(value)


def real(value: object, key: str) -> float:
    """One real field as a float, an infinity among them (a bound a field may hold, as a module built in
    Python holds `math.inf`): a bool is a flag, and it, every non-number and a NaN are refused."""
    if isinstance(value, bool) or not isinstance(value, Real) or math.isnan(value):
        raise ValueError(f"{key}={value!r}: this field is a real number")
    return float(value)


def number(value: object, key: str) -> float:
    """One real field: an epsilon, a rate, a frequency. An integer is one.

    transformers writes an infinite bound as `{"__float__": "Infinity"}` or
    `{"__float__": "-Infinity"}` rather than as JSON's out-of-spec literal
    (`PreTrainedConfig._encode_special_floats`, 5.16.1), so those two
    records are the one spelling of an infinity a config carries, and the
    only ones read here. Its NaN record, any other mapping and a bare
    non-finite number are refused: no field here is NaN.
    """
    if isinstance(value, Mapping):
        if dict(value) not in ({"__float__": "Infinity"}, {"__float__": "-Infinity"}):
            raise ValueError(f"{key}={value!r}: this field is a finite number or transformers' "
                             "record of an infinity")
        return float(str(value["__float__"]))
    return finite(value, key)


_SPECIAL_FLOATS = {"Infinity": math.inf, "-Infinity": -math.inf, "NaN": math.nan}


def decoded_floats(config: Mapping[str, object]) -> Mapping[str, JSON]:
    """`config`, a config's JSON, with transformers' records of the floats JSON cannot spell
    (`{"__float__": "Infinity"}`, `"-Infinity"`, `"NaN"`, `number`) read back as those floats, at any
    depth, as its own config load reads them before its class sees them
    (`PreTrainedConfig._dict_from_json_file`, 5.17): what a config class built from a mapping takes."""
    return {key: _decoded(value, key) for key, value in config.items()}


def _decoded(value: object, key: str) -> JSON:
    """One config value as its file holds it, transformers' float records read back (`decoded_floats`):
    a mapping's integer keys (a label map's, as a config object's `to_dict` gives them) as the strings
    its file spells them with, which the config class reads back alike."""
    if isinstance(value, Mapping):
        if set(value) == {"__float__"} and value["__float__"] in _SPECIAL_FLOATS:
            return _SPECIAL_FLOATS[str(value["__float__"])]
        return {(str(name) if isinstance(name, int) else text(name, f"{key} record key")):
                _decoded(entry, key) for name, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return [_decoded(entry, key) for entry in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(f"{key}={value!r}: this field is not a value a JSON config carries")


def boolean(value: object, key: str) -> bool:
    """One flag field, as JSON's true and false."""
    if not isinstance(value, bool):
        raise ValueError(f"{key}={value!r}: this field is a boolean")
    return value


def text(value: object, key: str) -> str:
    """One named field: a dtype, an activation, an architecture."""
    if not isinstance(value, str):
        raise ValueError(f"{key}={value!r}: this field is a string")
    return value


def record(value: object, key: str) -> Mapping[str, object]:
    """One nested section, whose own fields read through the readers above."""
    if not isinstance(value, Mapping) or any(type(name) is not str for name in value):
        raise ValueError(f"{key}={value!r}: this field is a record of named fields")
    return value


def listed[ItemT](value: object, key: str, *, read: Callable[[JSON, str], ItemT],
                  what: str) -> tuple[ItemT, ...]:
    """One list, each entry read by `read(entry, key)`; `what` names what the list holds, as a
    refusal of anything else says."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key}={value!r}: this field is {what}")
    return tuple(read(entry, key) for entry in value)


integers = functools.partial(listed, read=integer, what="a list of integers")
"""One list of integers: layer indices, token ids, per-layer widths (`listed`)."""

numbers = functools.partial(listed, read=number, what="a list of numbers")
"""One list of finite numbers: per-frequency rotary factors (`listed`)."""

strings = functools.partial(listed, read=text, what="a list of strings")
"""One list of names: a layer pattern, an architecture list (`listed`)."""


def json_value(value: object, key: str) -> JSON:
    """One field narrowed to the JSON its file carries, section and all.

    A scalar, a list of them or a record of them is what `json.loads` can
    produce; anything else reached this field from somewhere other than the
    file, and the key names which field it was.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [json_value(entry, key) for entry in value]
    if isinstance(value, Mapping):
        return {text(name, f"{key} record key"): json_value(entry, key)
                for name, entry in value.items()}
    raise ValueError(f"{key}={value!r}: this field is not a value a JSON config carries")


UNREAD = "unread"
"""The field of an inference record naming the parts of the run's tree its model does not read
(`without`), which a loader leaves unrestored (`dew.inference.tasks.SavedRun.variables`): a distillation's
teacher and projections beside the student it saves (`dew.objectives.distillation`)."""

ADDED = "added"
"""The metadata key of a configuration field added after runs were saved with it: declared
`dataclasses.field(default=..., repr=False, metadata={ADDED: True})`, so a run that leaves it at its
default records the configuration it always did (a data step's identity, `dew.data.plan.StepIdentity`)
and resumes under this Dew, and one that sets it records it (`set_added`)."""


RUNS = "runs"
"""The metadata key of a decoder field that says how the model runs or trains, not what it computes:
precision and kernels, dropout, the initialization a loaded model's weights replace. `layer_scalar` is
whether Gemma 4's scalars train, the forward multiplying by them either way; `scale_after_cast` orders a
norm's scale and its cast to the compute dtype, the same product in fp32. An export's config need not
carry such a field (`dew.interop.hf_decoders.refuse_rebuilt`). The decoder declares them
(`dew.nn.backbones.causal_transformer.CausalTransformer`)."""

NUMBERS = "numbers"
"""The metadata key of a field that says in what numbers a module computes, not what (`RUNS`): its
`dtype` and `precision`, which a decoder's and every part of a wrapper's (its towers, projections) spell
alike and no export's config reads back as fields (`dew.interop.hf_decoders.NUMERICS_FIELDS`)."""

TRAINS = "trains"
"""The metadata key of a field that says what a run trains toward though not what its forward computes
(`RUNS`, so an export need not carry it): a masked-diffusion decoder's corruption id. A resume and an average
of runs hold it as they hold what the model computes (`dew.registry.computing_differences`' `kept`): a run
resumed under another, or two runs trained under two, learned another objective."""

ROUNDS = "rounds"
"""The metadata key of a field that orders a model's rounding (`RUNS`): a norm's scale against its cast to the
compute dtype, a softmax's width, which no family's config states and which change what it computes below
float32 alone, so a release states them (`dew.interop.release.ROUNDING`)."""


def set_added(value: DataclassInstance | Sequence[DataclassInstance] | Mapping[str, DataclassInstance] | JSON
              ) -> bool:
    """Whether `value`, a configuration, or one it holds (in a sequence or a mapping), sets a field added
    after runs were saved (`ADDED`) off its default."""
    if isinstance(value, (list, tuple)):
        return any(map(set_added, value))
    if isinstance(value, Mapping):
        return any(map(set_added, value.values()))
    if not dataclasses.is_dataclass(value) or isinstance(value, type):
        return False
    return any((field.metadata.get(ADDED) and getattr(value, field.name) != field.default)
               or set_added(getattr(value, field.name)) for field in dataclasses.fields(value))


ATTENTION_KERNEL_HOLDERS = (
    "dew.decision.head:DecisionHead",
    "dew.decision.head:JointSchemaHead",
    "dew.nn.attention_blocks:BasicTransformerBlock",
    "dew.nn.attention_blocks:NormalAttention",
    "dew.nn.attention_blocks:TransformerBlock",
    "dew.nn.backbones.dit:SimpleDiT",
    "dew.nn.backbones.edm2:Block",
    "dew.nn.backbones.edm2:EDM2UNet",
    "dew.nn.backbones.flux:FluxSingleBlock",
    "dew.nn.backbones.flux:FluxTransformer",
    "dew.nn.backbones.flux2:Flux2SingleBlock",
    "dew.nn.backbones.flux2:Flux2Transformer",
    "dew.nn.backbones.jepa:FactorizedTokenStack",
    "dew.nn.backbones.jepa:JepaEncoder",
    "dew.nn.backbones.jepa:JepaPredictor",
    "dew.nn.backbones.jepa:JepaVideoEncoder",
    "dew.nn.backbones.jepa:TokenStack",
    "dew.nn.backbones.joint:DoubleStreamBlock",
    "dew.nn.backbones.joint:JointAttention",
    "dew.nn.backbones.mmdit:HierarchicalMMDiT",
    "dew.nn.backbones.mmdit:SimpleMMDiT",
    "dew.nn.backbones.qwen_image:QwenImageTransformer",
    "dew.nn.backbones.sd3:SD3Transformer",
    "dew.nn.backbones.ssm_dit:HybridSSMAttentionDiT",
    "dew.nn.backbones.unet:Unet",
    "dew.nn.backbones.unet3d:UNet3D",
    "dew.nn.backbones.unet_condition:UNet2DCondition",
    "dew.nn.backbones.uvit:SimpleUDiT",
    "dew.nn.backbones.uvit:UViT",
    "dew.nn.backbones.video_dit:VideoDiT",
    "dew.nn.backbones.wan:WanBlock",
    "dew.nn.backbones.wan:WanTransformer",
    "dew.nn.backbones.z_image:ZImageBlock",
    "dew.nn.backbones.z_image:ZImageTransformer",
    "dew.nn.cross_attention:GatedCrossAttention",
    "dew.nn.dit:ModulatedBlock",
    "dew.nn.dsa_kpool:KPoolSparseAttention",
    "dew.nn.mla:MultiHeadLatentAttention",
    "dew.nn.mobilenet:MobileAttention",
    "dew.nn.mobilenet:MobileMultiQueryAttention",
    "dew.nn.mobilenet:MobileNetV5Encoder",
    "dew.nn.mobilenet:MobileStage",
)
"""The classes that held an attention kernel (`attention_impl`), the run's since row 11938
(`Diagnostics.kernels`, `generation.chosen(attention=...)`), never a model's."""

RETIRED_KERNELS = frozenset({"generated"})
"""The kernel choices Dew retired since records were saved pinning them: `generated`, which ran the
algebra's checked twins of a template's written math where a call's exact case was declared (rows 3272,
6109: on measurement the twins run in no program), so a saved record that pinned it runs the written math, as
every run does now. A saved record's kernel choices are read without them (`DECODED_REMOVED`, a release's
and a saved run's step's pins `retired_kernels`); a live choice naming one is refused."""


def retired_kernels[V](pins: Mapping[str, V]) -> Mapping[str, V]:
    """A saved record's kernel pins by choice (a release's `dew_release.json`, a saved run's step's) without
    the choices retired since (`RETIRED_KERNELS`); `pins` itself where it holds none."""
    if RETIRED_KERNELS.isdisjoint(pins):
        return pins
    return {choice: kernel for choice, kernel in pins.items() if choice not in RETIRED_KERNELS}


_SAVED: contextvars.ContextVar[bool] = contextvars.ContextVar("reading_saved", default=False)


@contextlib.contextmanager
def reading_saved() -> Iterator[None]:
    """Within the block, records are read as saved ones (`saved_fields`): a run's (`RunConfig.load`,
    `RunConfig.read`), whose classes may have lost fields since it was written. Outside it, a record's every
    field is its class's or refused, as any live configuration's."""
    token = _SAVED.set(True)
    try:
        yield
    finally:
        _SAVED.reset(token)


def saved_fields[V](member: type | Callable[..., object], fields: Mapping[str, V],
                    accepted: Collection[str]) -> Mapping[str, V]:
    """`fields`, a record of `member` (a class, or a function a record names), read as a saved record is
    (`reading_saved`), without the fields `member` or a base lost since (`DECODED_REMOVED`, by each class's
    import path in `member`'s order of resolution) that its reader does not accept (`accepted`, the names the
    decoder takes for it: a subclass that records a retired name keeps it); `fields` itself outside a saved
    record's reading, or where it holds none."""
    if not _SAVED.get():
        return fields
    kinds = member.__mro__ if isinstance(member, type) else (member,)
    removed = frozenset().union(*(DECODED_REMOVED.get(f"{kind.__module__}:{kind.__qualname__}", frozenset())
                                  for kind in kinds)) - frozenset(accepted)
    if removed.isdisjoint(fields):
        return fields
    return {name: value for name, value in fields.items() if name not in removed}


REMOVED_FIELDS: Mapping[str, frozenset[str]] = {
    "dew.decision.layout:OptionLines": frozenset({"stream"}),
    "dew.nn.backbones.causal_transformer:CausalTransformer": frozenset({"scan_layers", "attention_impl"}),
    "dew.interop.source_options:SourceOptions": frozenset({"scan_layers"}),
    **{holder: frozenset({"attention_impl"}) for holder in ATTENTION_KERNEL_HOLDERS},
}
"""The fields a configuration lost after runs were saved with them, by its class's import path: each always
another's (a question's option lines' stream, its instructions', rows 2008 to 2450), or how a model executes
and never what it computes (a decoder's layer scan, which its structure decides since row 4074; an attention
kernel, the run's since row 11938, which a saved run's model's moves to: `dew.config.earlier`), so a saved
record of one means what today's, which has none, does. A saved run's records read without them
(`earlier_fields`)."""

DECODED_REMOVED: Mapping[str, frozenset[str]] = {"dew.kernels:Kernels": RETIRED_KERNELS}
"""The fields a saved record's class lost since that the decoder reads out of a saved record wherever its
field's type reads the class (`saved_fields`): a run's kernel choices retired since (`RETIRED_KERNELS`),
which only a field typed as kernel choices holds, never data that looks like them. `REMOVED_FIELDS` stay
`earlier_fields`' alone."""


MOVED_CLASSES: Mapping[tuple[str, str], tuple[str, ...]] = {
    ("dew.inputs.encoders", "dew.interop.encoders"): ("CLIPText", "T5Text", "HFAudio"),
    ("dew.inputs.diffusion", "dew.interop.conditioners"): (
        "DiffusionConditioner", "QwenImageConditioner", "HiddenStatesConditioner", "WanConditioner"),
    ("dew.config.trainer", "dew.interop.source_options"): (
        "SourceOptions", "SourcePlacement", "SourceBuild", "SourceOptionsKeywords"),
}
"""The classes an earlier Dew kept in another module, by that module and the one each is in now: the
encoders that bind a released checkpoint, dew.interop's since row 5996, which a run's record, its steps'
artifacts and its data plans named under dew.inputs; a run's source options, dew.interop's since row 5999,
which a run's record named under dew.config.trainer. Carried to 1.0 with the other `earlier_*` readers."""

MOVED_PATHS: Mapping[str, str] = {
    f"{earlier}{separator}{name}": f"{now}{separator}{name}"
    for (earlier, now), names in MOVED_CLASSES.items() for name in names for separator in ":."
}
"""Each moved class's path as an earlier Dew wrote it, by the path it has now: an import path a record names
(`module:Class`, `dew.registry.import_path`) and the class a data plan describes a step's part by
(`module.Class`, `dew.data.plan.structured`)."""

_MOVED = re.compile("|".join(rf"\b{re.escape(path)}\b" for path in MOVED_PATHS))
_EARLIER = {now: earlier for earlier, now in MOVED_PATHS.items()}
_SPELLED_EARLIER = re.compile("|".join(rf"\b{re.escape(path)}\b" for path in _EARLIER))


EARLIER_LAYOUTS: Mapping[str, Mapping[str, JSON]] = {
    "dew.decision.layout:MarkerLayout": {"form": "marker"},
    "dew.decision.layout:StateFirstLayout": {"form": "state_first"},
    "dew.decision.layout:QuestionLayout": {"form": "state_first"},
    "dew.decision.layout:JointLayout": {"form": "joint", "max_len": 16384},
}
"""Each decision layout class an earlier Dew recorded (decision-inputs.md §1: three classes, and their base,
which wrote its rows state first), by its import path: the form it is now and the fields its class held by
default that `dew.decision.SegmentLayout` holds otherwise (a joint row's 16384 tokens; its segments and
framing it fills where a record states none, `SegmentLayout.joint_row`)."""


def class_record(value: Mapping[typing.Any, object]) -> tuple[str, Mapping[str, object]] | None:
    """The class a class record, `{"class": ..., "fields": {...}}`, names and its fields (a record with no
    fields may leave them out); None for a mapping that is not one: data that names a class is no record."""
    name = value.get("class")
    if not isinstance(name, str):
        return None
    fields = value.get("fields", {})
    if isinstance(fields, Mapping) and all(key in ("class", "fields") for key in value):
        return name, fields
    return None


def class_records(value: JSON, rewrite: Callable[[str, Mapping[str, JSON], bool], dict[str, JSON] | None]
                  ) -> JSON:
    """`value` (JSON a Dew saved) with each class record in it, at any depth, as `rewrite(name, fields,
    root)` makes it (`class_record`; `root` whether it is `value` itself), or as it is where `rewrite` returns
    None; records under a record rewritten first. A list or mapping is rebuilt only where something under
    it changed, so data no rewrite reads passes as it is. The one walk of a saved record's class records
    (row 11838): `earlier_layout`'s, `dew.config.earlier.earlier_retired`'s."""
    def walked(held: JSON, *, root: bool = False) -> JSON:
        if isinstance(held, list):
            # Copied only from its first element that changed (row 11838's review: no list a node).
            elements = None
            for index, element in enumerate(held):
                new = walked(element)
                if elements is None and new is not element:
                    elements = held[:index]
                if elements is not None:
                    elements.append(new)
            return held if elements is None else elements
        if not isinstance(held, Mapping):
            return held
        changed = {key: new for key, entry in held.items() if (new := walked(entry)) is not entry}
        entries = {**held, **changed} if changed else held
        named = class_record(entries)
        # A JSON record's fields are JSON: `entries`' own, read as the record names them.
        fields = entries.get("fields", {})
        if named is not None and isinstance(fields, dict) and (
                rewritten := rewrite(named[0], fields, root)) is not None:
            return rewritten
        return entries
    return walked(value, root=True)


def earlier_layout(value: JSON) -> JSON:
    """JSON an earlier Dew saved (a step's declaration, a run's record, a release's layout) with every
    decision layout's class record in it, at any depth (an objective's `layout`, a recorded stage's), read
    as the `SegmentLayout` it is now where it names an earlier class (`EARLIER_LAYOUTS`): its own fields
    over the form and defaults its class held; anything else as it is (`class_records`). The one reader of
    those records, which the pinned boundaries call (`dew.config.earlier.earlier_run`,
    `dew.inference.tasks.earlier_shapes`, `dew.decision.release.released_decision`). Carried to 1.0
    (ledger 1552)."""
    def current(named: str, fields: Mapping[str, JSON], root: bool) -> dict[str, JSON] | None:
        if named not in EARLIER_LAYOUTS:
            return None
        return {"class": "dew.decision.layout:SegmentLayout", "fields": {**EARLIER_LAYOUTS[named], **fields}}
    return class_records(value, current)


def moved_record(record: Mapping[str, object]) -> Mapping[str, object]:
    """A saved record's fields, the JSON its file holds, with the paths they name translated
    (`moved_paths`)."""
    return {name: moved_paths(json_value(value, name)) for name, value in record.items()}


def moved_paths(value: JSON) -> JSON:
    """`value`, JSON an earlier Dew saved (a run's record, a step's artifact, a data plan or a position's
    order), with each moved class's path in its text read as the path it has now (`MOVED_PATHS`),
    wherever a string holds it: a record's class names one whole, and a data plan's step configuration,
    itself JSON in a string, among its other parts. Readers of saved records translate by it (`earlier_run`,
    `dew.inference.tasks.earlier_shapes`, `dew.data.plan.Plan.check`); a path given today is read as given
    (`dew.registry.imported`)."""
    return mapped_strings(value, moved_spelling)


def strings_of(value: JSON) -> Iterator[str]:
    """Every string `value` holds at any depth, values and lists walked as `mapped_strings` walks them (a
    mapping's keys left out)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for held in value.values():
            yield from strings_of(held)
    elif isinstance(value, (list, tuple)):
        for held in value:
            yield from strings_of(held)


def mapped_strings(value: JSON, rewrite: Callable[[str], str]) -> JSON:
    """`value` with each string it holds at any depth replaced by `rewrite` of it, its keys as they are."""
    if isinstance(value, str):
        return rewrite(value)
    if isinstance(value, dict):
        return {key: mapped_strings(held, rewrite) for key, held in value.items()}
    if isinstance(value, list):
        return [mapped_strings(held, rewrite) for held in value]
    return value


def moved_spelling(text: str) -> str:
    """`text` with each moved class's path an earlier Dew wrote read as the path it has now
    (`MOVED_PATHS`)."""
    return _MOVED.sub(lambda path: MOVED_PATHS[path.group(0)], text)


def earlier_spelling(text: str) -> str:
    """`text` with each moved class's path spelled as an earlier Dew wrote it (`MOVED_PATHS`): what a data
    plan's digests hash (`dew.data.plan.fingerprint`), so a position saved before a class moved, which
    holds its members' and its parts' digests alone, compares equal to the same plan's now."""
    return _SPELLED_EARLIER.sub(lambda path: _EARLIER[path.group(0)], text)


def earlier_fields(value: JSON) -> JSON:
    """`value`, a saved run's records of components (a decision layout's segments, in its run.json or its
    checkpoint's task record), with each component's fields its class lost since (`REMOVED_FIELDS`) left
    out: a component record names its class and holds its fields (`{"class": ..., "fields": ...}`),
    wherever it nests. Its callers hand it components alone, never a record of data a row holds, which may
    look like one. Walked as every class record is (`class_records`, row 11911)."""
    def without_removed(name: str, fields: Mapping[str, JSON], root: bool) -> dict[str, JSON] | None:
        removed = REMOVED_FIELDS.get(name, frozenset())
        if removed.isdisjoint(fields):
            return None
        kept = {field: held for field, held in fields.items() if field not in removed}
        return {"class": name, "fields": kept}

    return class_records(value, without_removed)


def earlier_described_fields(value: JSON) -> JSON:
    """`value`, a data plan an earlier Dew saved, with each step description's fields its class lost since
    (`REMOVED_FIELDS`) left out: a description names its class by its dotted path beside its fields
    (`dew.data.plan.structured`), wherever it nests, a plan's step's configuration (`StepIdentity.config`,
    an entry's of a plan's `steps`, its JSON in a string) read and written back as the plan writes one
    (`sort_keys`). A position saved before a field was retired resumes exactly (row 11938: a decision head's
    attention kernel). Only a plan's step's configuration is read as JSON, never a string a description
    holds, a caption's template or a step's own `config` field among them, which stays as written, where a
    string under any `config` read as a record and one that was no JSON raised (the API02 review)."""
    removed = {path.replace(":", "."): fields for path, fields in REMOVED_FIELDS.items()}
    named = {field for fields in removed.values() for field in fields}

    def kept(held: Mapping) -> dict:
        named_class = held.get("class")
        dropped = removed.get(named_class, frozenset()) if isinstance(named_class, str) else frozenset()
        return {key: entry for key, entry in held.items() if key not in dropped}

    def described(held: JSON) -> JSON:
        # A description, its strings as written.
        if isinstance(held, list):
            return [described(entry) for entry in held]
        if isinstance(held, Mapping):
            return {key: described(entry) for key, entry in kept(held).items()}
        return held

    def planned(held: JSON) -> JSON:
        # The plan around its steps' identities (`StepIdentity`), each entry of a `steps` list one.
        if isinstance(held, list):
            return [planned(entry) for entry in held]
        if isinstance(held, Mapping):
            return {key: ([identity(step) for step in entry] if key == "steps" and isinstance(entry, list)
                          else planned(entry)) for key, entry in kept(held).items()}
        return held

    def identity(step: JSON) -> JSON:
        # A step's identity: its configuration a generated instance's description, the JSON text the plan
        # wrote (`structured`), walked where it names a retired field; a declared identity or a name, which
        # is no JSON, as written.
        if not isinstance(step, Mapping):
            return planned(step)
        config = step.get("config")
        # A declared identity (`StepIdentity.declared`, which a step's configuration then is too) is opaque,
        # whatever it spells (the API02 review: an identity naming a retired field read as JSON raised).
        if (not (isinstance(config, str) and config.startswith("{")
                 and any(f'"{field}"' in config for field in named)) or config == step.get("declared")):
            return planned(step)
        return {key: json.dumps(described(json.loads(config)), sort_keys=True) if key == "config"
                else planned(entry) for key, entry in kept(step).items()}

    return planned(value)


def annotations_of(owner: type, **options) -> Mapping[str, object]:
    """`owner`'s annotations resolved (`typing.get_type_hints`, `options` its own), its own type
    parameters among the names they are read against: a class with type parameters (PEP 695:
    `dew.objectives.base.TrainingScalar[Statistics, Additions]`) names them in its string annotations,
    which Python 3.12.0 to 3.12.3 resolve without them (CPython gh-114053), and Dew declares 3.12.0. Every
    reader that walks a config's classes reads their hints so (the option witness, the API report)."""
    # Every class states its type parameters from 3.12, none where it declares none.
    held = {parameter.__name__: parameter for parameter in owner.__type_params__}
    return typing.get_type_hints(owner, localns=held, **options)
