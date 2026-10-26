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
from dew.batch import Variables
from dew.data.text import HFTokenizer
from dew.interop.composite import CompositeSource, HeadComponent, NativeComponent, SourceKind
from dew.interop.pretrained import Pretrained
from dew.interop.safetensors_io import MAX_SHARD_SIZE
from dew.registry import dtype_name, import_path

from . import joint_checkpoint
from .calibration import Calibration, Temperatures
from .head import DecisionHead, Head, JointSchemaHead
from .joint_checkpoint import HEAD_CONFIG, HEAD_WEIGHTS, _path, _shaped
from .laya import CONFIG_FILE, IN_PROJ, _head_path, _token
from .layout import JointLayout, Layout, MarkerLayout, Specials
from .model import DecisionModel
from .task import Decide


@dataclasses.dataclass(frozen=True)
class DecisionHeadSource(HeadComponent):
    """The head constructor reads the native backbone's output width and compute dtype."""

    def build(self, backbone: Pretrained | None, dtype: str, attention_impl: str, *,
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
    is the backbone and its readout together, which `save` writes: its tensors are not one flat set, so
    `export` refuses them, and `backbone` is the backbone alone, whose `export` and `save` are its own."""

    head: Head
    head_variables: Variables
    layout: Layout
    specials: Specials
    calibration: Calibration
    decision_tokenizer: HFTokenizer
    writer: Callable[..., None] = dataclasses.field(repr=False, compare=False)
    """The release format's writer, the source codec's own: `save` writes the format it was read from."""

    def map_variables(self, transform: Callable[[Variables], Variables]) -> Self:
        return dataclasses.replace(self, variables=transform(self.variables),
                                   head_variables=transform(self.head_variables))

    def backbone(self) -> Pretrained:
        """The native backbone alone, as it was read and as it is adapted (`adapt`), without the
        readout: its own `export` and `save`, which a release's writer writes beside the readout."""
        held = {field.name: getattr(self, field.name) for field in dataclasses.fields(Pretrained)}
        return Pretrained(**held)

    def export(self, variables: Mapping[str, object] | None = None) -> Mapping[str, np.ndarray]:
        raise ValueError("a decision release writes its backbone and its readout apart, which no one "
                         "flat set of tensors holds: save it, or export its backbone alone "
                         "(`backbone().export()`)")

    def pipeline(self) -> Decide:
        return Decide.of(self.packed())

    def save(self, directory: str | Path, *, variables: Mapping[str, object] | None = None,
             max_shard_size: int | str = MAX_SHARD_SIZE) -> None:
        bundle = self if variables is None else dataclasses.replace(self, variables=dict(variables))
        self.writer(bundle, directory, max_shard_size=max_shard_size)

    @classmethod
    def from_components(cls, backbone: Pretrained, head: Head, head_variables: Variables,
                        directory: Path, tokenizer_folder: str, *, layout: Layout,
                        calibration: Calibration, strict_specials: bool,
                        writer: Callable[..., None]) -> Self:
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
                   calibration=calibration, decision_tokenizer=tokenizer, writer=writer)


def _save_laya(bundle: PretrainedDecision, directory: str | Path, *,
               max_shard_size: int | str = MAX_SHARD_SIZE) -> None:
    Decide.of(bundle).save_pretrained(directory, max_shard_size=max_shard_size)


class DecisionSource(SourceKind):
    """The metadata signatures of Laya and joint-head releases."""

    @staticmethod
    def describe(directory: Path) -> CompositeSource | None:
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
            layout = MarkerLayout(
                max_len=_records.integer(config.get("max_len", 512), "max_len"),
                head_max_len=_records.integer(config.get("head_max_len", 192), "head_max_len"),
                option_tokens=_records.integer(config.get("option_tokens", MarkerLayout.option_tokens),
                                              "option_tokens"),
                parallel=config.get("option_layout", "sequential") == "parallel")
            values = config.get("temperature", [1.0, 1.0, 1.0])
            if not isinstance(values, list) or len(values) != 3:
                raise ValueError("temperature lists choice, score and noul")
            buckets = _records.record(config.get("temperature_by_options", {}), "temperature_by_options")
            calibration = Calibration(Temperatures(
                dict(zip(("choice", "score", "noul"), (_records.number(x, "temperature") for x in values),
                         strict=True)), {key: _records.number(value, key) for key, value in buckets.items()}))
            tokenizer_folder, strict_specials, writer = "tokenizer", True, _save_laya
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
            layout, calibration = JointLayout(), Calibration()
            tokenizer_folder, strict_specials, writer = "", False, joint_checkpoint.save
        else:
            return None
        return CompositeSource(backbone, head, PretrainedDecision.from_components, tokenizer_folder,
                               {"layout": layout, "calibration": calibration,
                                "strict_specials": strict_specials, "writer": writer})
