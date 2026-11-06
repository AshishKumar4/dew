"""Released decision metadata and the bundle's task declaration.

The registered codec describes components; `Pretrained.load` resolves their
files, translates weights and places them through the ordinary native route.
"""

import dataclasses
import json
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Self

import numpy as np

from dew import _records
from dew.batch import Variables, part, thaw
from dew.data.text import HFTokenizer
from dew.interop.composite import CompositeSource, HeadComponent, NativeComponent, SourceKind
from dew.interop.pretrained import Pretrained
from dew.interop.safetensors_io import MAX_SHARD_SIZE, joined_tensors
from dew.nn.attention import AttentionImpl
from dew.registry import dtype_name, import_path

from . import joint_checkpoint
from .calibration import Calibration, Temperatures
from .head import DecisionHead, Head, JointSchemaHead
from .joint_checkpoint import HEAD_CONFIG, HEAD_WEIGHTS, _path, _shaped
from .laya import CONFIG_FILE, IN_PROJ, _head_path, _token, laya_tensors
from .layout import JointLayout, Layout, MarkerLayout, QuestionLayout, Specials
from .model import DecisionModel
from .release import released_decision
from .task import Decide


@dataclasses.dataclass(frozen=True)
class DecisionHeadSource(HeadComponent):
    """The head constructor reads the native backbone's output width and compute dtype."""

    def build(self, backbone: Pretrained | None, dtype: str, attention_impl: AttentionImpl, *,
              shapes: Mapping[str, tuple[int, ...]] | None) -> Head:
        held = _records.record(self.record, "head")
        fields = _records.record(held["fields"], "head.fields")
        if backbone is None:
            width = _records.integer(fields["hidden_size"], "hidden_size")
            compute = dtype
        else:
            width, compute = DecisionModel.head_size(backbone.model)
        configured = {name: value for name, value in fields.items() if name != "hidden_size"}
        return Head.from_record(_records.json_value({**held, "fields": {
            **configured, "dtype": None if compute is None else dtype_name(compute),
            "attention_impl": attention_impl}}, "head"), width)


@dataclasses.dataclass(frozen=True, kw_only=True)
class PretrainedDecision(Pretrained):
    """A native backbone bundle with its readout and declared decision task.
    Inherited adaptation operates on the backbone; the readout retains its own variables. The release
    is the backbone and its readout together: `save` writes both in the format it was read from, and
    `export` returns every tensor that release holds, the readout's beside the backbone's, by its name
    in the release's files, which a weight push (`NCCLPush`) sends; `backbone` is the backbone alone."""

    head: Head
    head_variables: Variables
    layout: Layout
    specials: Specials
    calibration: Calibration
    decision_tokenizer: HFTokenizer
    writer: Callable[..., None] = dataclasses.field(repr=False, compare=False)
    """The release format's writer, the source codec's own: `save` writes the format it was read from."""
    exporter: Callable[..., Mapping[str, np.ndarray]] = dataclasses.field(repr=False, compare=False)
    """The release format's tensors, `writer`'s own: `export` returns what `save` writes."""
    assumed: frozenset[str] = frozenset()
    """The question budgets its layout holds that its release states none of, Laya's own
    (`LAYA_BUDGETS`), as a run reports where each came from."""

    def map_variables(self, transform: Callable[[Variables], Variables]) -> Self:
        return dataclasses.replace(self, variables=transform(self.variables),
                                   head_variables=transform(self.head_variables))

    def backbone(self) -> Pretrained:
        """The native backbone alone, as it was read and as it is adapted (`adapt`), without the
        readout: its own `export` and `save`, which a release's writer writes beside the readout."""
        held = {field.name: getattr(self, field.name) for field in dataclasses.fields(Pretrained)}
        return Pretrained(**held)

    def export(self, variables: Mapping[str, object] | None = None) -> Mapping[str, np.ndarray]:
        """Every tensor the release `save` writes, the readout's beside the backbone's, by its name in the
        release's files; `variables`, a trainer's split of the backbone's, in place of the bundle's."""
        return self.exporter(self, variables)

    def pipeline(self) -> Decide:
        return Decide.of(self.packed())

    def save(self, directory: str | Path, *, variables: Mapping[str, object] | None = None,
             max_shard_size: int | str = MAX_SHARD_SIZE) -> None:
        bundle = self if variables is None else dataclasses.replace(self, variables=dict(variables))
        self.writer(bundle, directory, max_shard_size=max_shard_size)

    @classmethod
    def of(cls, task: Decide) -> Self:
        """The release bundle of `task`, as it is written (`save`) in the format its head is read from: its
        backbone a decoder bundle (`PretrainedDecoder.from_model`, its LoRA factors merged into their
        kernels), its head and their variables, its layout, calibration, special tokens and tokenizer; the
        inverse of `Decide.released`, which a run's task and a loaded release are both written through.
        A head no format holds (Laya's, the joint head's, the slot readout's), a tokenizer with no files
        to write and a backbone no decoder family states are refused."""
        from .slot_source import SlotSource

        if not isinstance(task.tokenizer, HFTokenizer):
            raise ValueError("a released checkpoint holds the files of a Hugging Face tokenizer, and this "
                             f"task's is a {type(task.tokenizer).__name__}")
        formats = {"joint": (joint_checkpoint.save, joint_checkpoint.export),
                   "slots": (SlotSource.save, SlotSource.export)}
        head = task.model.head
        if head.release_format not in formats:
            raise ValueError(f"a {type(head).__name__} is released in {head.release_format!r}, no format a "
                             f"decoder bundle writes: the joint head's or the slot readout's")
        from dew.interop.pretrained import PretrainedDecoder

        encoder, backbone = task.model.backbone, thaw(part(task.variables, "backbone"))
        decoder = PretrainedDecoder.from_model(encoder, backbone).bound(encoder, backbone)
        # The task's numerics, which a decoder bundle built of its model states none of.
        held = {**{field.name: getattr(decoder, field.name) for field in dataclasses.fields(Pretrained)},
                "numerics": task.precision}
        writer, exporter = formats[head.release_format]
        return cls(**held, head=head, head_variables=thaw(part(task.variables, "head")), layout=task.layout,
                   specials=task.specials, calibration=task.calibration, decision_tokenizer=task.tokenizer,
                   writer=writer, exporter=exporter)

    @classmethod
    def from_components(cls, backbone: Pretrained, head: Head, head_variables: Variables,
                        directory: Path, tokenizer_folder: str, *, layout: Layout,
                        calibration: Calibration, strict_specials: bool, writer: Callable[..., None],
                        exporter: Callable[..., Mapping[str, np.ndarray]],
                        assumed: frozenset[str] = frozenset()) -> Self:
        tokenizer = HFTokenizer(str(directory / tokenizer_folder), local_files_only=True)
        if strict_specials:
            hf = tokenizer.tokenizer
            specials = Specials(begin=_token(hf.cls_token_id, "cls"),
                                separator=_token(hf.sep_token_id, "sep"),
                                marker=_token(hf.mask_token_id, "mask"), marker_text=str(hf.mask_token),
                                pad=_token(hf.pad_token_id, "pad"))
        else:
            specials = Specials.of(tokenizer)
        inherited = {field.name: getattr(backbone, field.name) for field in dataclasses.fields(Pretrained)}
        inherited["source"] = directory
        return cls(**inherited, head=head, head_variables=head_variables, layout=layout, specials=specials,
                   calibration=calibration, decision_tokenizer=tokenizer, writer=writer, exporter=exporter,
                   assumed=assumed)


LAYA_BUDGETS = {"head_max_len": 192, "option_tokens": 48}
"""The question budgets Laya's release declares where its config states none: the tokens its question
and options share, and the most one option keeps (`QuestionLayout.head_max_len`, `option_tokens`)."""


def _save_laya(bundle: PretrainedDecision, directory: str | Path, *,
               max_shard_size: int | str = MAX_SHARD_SIZE) -> None:
    Decide.of(bundle).save_pretrained(directory, max_shard_size=max_shard_size)


def _export_laya(bundle: PretrainedDecision, variables: Mapping[str, object] | None
                 ) -> Mapping[str, np.ndarray]:
    held = bundle if variables is None else dataclasses.replace(bundle, variables=dict(variables))
    task = Decide.of(held)
    return laya_tensors(task.model, task.variables)[0]


def joined_release(backbone: Mapping[str, np.ndarray], readout: Mapping[str, np.ndarray], files: str
                   ) -> Mapping[str, np.ndarray]:
    """A release's backbone tensors and its readout's, which it writes apart (`files`), as one table by
    name, each built from its own as it is read (`joined_tensors`, ledger 4370); a name both hold is
    refused before any is read, since one table cannot hold the two tensors under it (save the release
    instead)."""
    return joined_tensors({"the release's backbone": backbone, f"its readout ({files})": readout})


class DecisionSource(SourceKind):
    """The metadata signatures of Laya and joint-head releases."""

    @staticmethod
    def describe(directory: Path, overrides: Mapping[str, object] | None = None) -> CompositeSource | None:
        # A Laya or joint-head layout reads nothing a backbone's modeling_overrides change.
        if (directory / CONFIG_FILE).is_file():
            config = _records.record(json.loads((directory / CONFIG_FILE).read_text()), CONFIG_FILE)
            native = _records.record(json.loads((directory / "encoder/config.json").read_text()),
                                    "encoder/config.json")
            backbone = NativeComponent({**native, "architectures": ["ModernBertModel"]},
                                       config_folder="encoder", tensor_prefix="encoder.")
            head = DecisionHeadSource(
                {"class": import_path(DecisionHead), "fields": {
                    "layers": _records.integer(config.get("head_layers", 2), "head_layers")}},
                _head_path, IN_PROJ)
            # The release's own budgets, as its config states them, Laya's where it states none; a
            # budget it states as null keeps that part of the question whole.
            head_max_len, option_tokens = (
                None if (value := config.get(name, default)) is None else _records.integer(value, name)
                for name, default in LAYA_BUDGETS.items())
            assumed = frozenset(name for name in LAYA_BUDGETS if name not in config)
            layout = MarkerLayout(max_len=_records.integer(config.get("max_len", 512), "max_len"),
                                  head_max_len=head_max_len, option_tokens=option_tokens,
                                  parallel=config.get("option_layout", "sequential") == "parallel")
            buckets = _records.record(config.get("temperature_by_options", {}), "temperature_by_options")
            types, default = _laya_temperature(config.get("temperature", 1.0))
            calibration = Calibration(Temperatures(
                types, {key: _records.number(value, key) for key, value in buckets.items()}, default=default))
            # The layout and calibration the release states, over those Laya's own file derives.
            derived = layout
            layout, calibration = released_decision(directory, layout, calibration, kind=QuestionLayout)
            assumed = assumed if layout is derived else frozenset()
            tokenizer_folder, strict_specials, writer, exporter = "tokenizer", True, _save_laya, _export_laya
        elif (directory / HEAD_CONFIG).is_file():
            config = _records.record(json.loads((directory / HEAD_CONFIG).read_text()), HEAD_CONFIG)
            if _records.number(config.get("dropout", 0.0), "dropout"):
                raise ValueError("the stored head drops out at a rate Dew's head does not train with")
            config_path = directory / "config.json"
            backbone = (NativeComponent(_records.record(json.loads(config_path.read_text()), "config.json"))
                        if config_path.is_file() else None)
            fields = {key: _records.integer(config[key], key) for key in joint_checkpoint.SIZES}
            head = DecisionHeadSource(_records.json_value({"class": import_path(JointSchemaHead),
                                                          "fields": fields}, "head"),
                                      _path, IN_PROJ, HEAD_WEIGHTS,
                                      partial(_shaped, fields["hidden_size"]))
            layout, calibration = released_decision(directory, JointLayout(), Calibration(), kind=JointLayout)
            assumed = frozenset()
            tokenizer_folder, strict_specials = "", False
            writer, exporter = joint_checkpoint.save, joint_checkpoint.export
        else:
            return None
        return CompositeSource(backbone, head, PretrainedDecision.from_components, tokenizer_folder,
                               {"layout": layout, "calibration": calibration, "assumed": assumed,
                                "strict_specials": strict_specials, "writer": writer, "exporter": exporter})



def _laya_temperature(stated: object) -> tuple[dict[str, float], float]:
    """Laya's `temperature` as `Temperatures`' types and default: a list of choice's, score's and noul's, or
    one number over every type (a release's global temperature)."""
    if isinstance(stated, list):
        if len(stated) != 3:
            raise ValueError("temperature lists choice, score and noul, or is one number over every type")
        return dict(zip(("choice", "score", "noul"), (_records.number(x, "temperature") for x in stated),
                        strict=True)), 1.0
    return {}, _records.number(stated, "temperature")
