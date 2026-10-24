"""The slot-readout release's metadata, labels and prompt, over native source components."""

import dataclasses
import json
from pathlib import Path
from typing import Literal

from dew import records
from dew.data.text import HFTokenizer
from dew.interop.composite import CompositeSource, HeadComponent, NativeComponent, SourceKind
from dew.interop.pretrained import Pretrained
from dew.registry import dtype_name, import_path, to_record

from .calibration import Calibration
from .head import Head, LabelReadout, SlotReadout
from .layout import (
    LETTERS,
    AnswerCue,
    Framing,
    Instructions,
    Labels,
    Layout,
    OptionLines,
    QuestionLayout,
    State,
    Text,
)
from .model import DecisionModel
from .sources import PretrainedDecision


SLOT_SYSTEM = ("Apply the question to the state. Choose exactly one of the listed options. Respond with only "
               "its uppercase letter, with no explanation or reasoning.")
_NOT_ONE_TOKEN = frozenset(["BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "EY", "FJ", "FQ", "FV", "FZ",
                            "GJ", "GK", "GQ", "GZ", "HJ", "IY", "JF", "JG", "JH", "JL", "JN", "JQ"])
SLOT_LABELS = (*LETTERS, *[first + second for first in LETTERS for second in LETTERS
                           if first + second not in _NOT_ONE_TOKEN][:256 - 26])
"""OneJev qev/prompt.py's 256 labels at 81ce62f1597c91e46767d4d02d6ac2e18534fe94."""


def _path(name: str) -> tuple[str, ...] | None:
    return {"slot": ("slots",), "q.weight": ("q",), "k.weight": ("k",)}.get(name)


def _labels(tokenizer: HFTokenizer, count: int) -> tuple[Labels, tuple[tuple[int, ...], ...]]:
    """The release requires every bare letter to round-trip as one token, then admits all pairs
    together or none, as its engine's `_verify_slots` does."""
    valid = []
    for label in SLOT_LABELS:
        ids = tokenizer.encode(label, add_special_tokens=False)
        valid.append(len(ids) == 1 and tokenizer.decode(ids) == label)
    if not all(valid[:len(LETTERS)]):
        raise ValueError("the slot tokenizer must spell each bare A-Z label as one round-tripping token")
    names = (SLOT_LABELS if all(valid) else LETTERS)[:count]
    labels = Labels(variants=("{}",), names=names)
    return labels, labels.ids(tokenizer, len(names))


@dataclasses.dataclass(frozen=True)
class SlotHeadSource(HeadComponent):
    def build(self, backbone: Pretrained | None, dtype: str, attention_impl: str) -> Head:
        if backbone is None:
            raise ValueError("a slot readout needs its native backbone")
        width, compute = DecisionModel.head_size(backbone.model)
        held = records.record(self.record, "head")
        fields = records.record(held["fields"], "head.fields")
        return Head.from_record(records.json_value({**held, "fields": {
            **fields, "dtype": None if compute is None else dtype_name(compute)}}, "head"), width)


class SlotSource(SourceKind):
    """Readout metadata beside a native decoder, with the release's prompt as a layout value."""

    @staticmethod
    def layout(*, max_len: int = 32768, keep: Literal["start", "end", "all"] = "all",
               labels: Labels = Labels(variants=("{}",), names=SLOT_LABELS),
               blocks: bool = False) -> QuestionLayout:
        """The release's whole-prompt encoding with thinking disabled and its question last."""
        return QuestionLayout(
            max_len=max_len, labels=labels, unit="whole", blocks=blocks,
            framing=Framing(system=SLOT_SYSTEM, prefill="", thinking=False),
            segments=(Text("<state>\n"), State(render="state", keep=keep), Text("\n</state>\n\n"),
                      Instructions({"choice": "Question: {}", "noul": "Question: {}",
                                    "score": "Rate the state: {}"}, render="entry",
                                   defaults={"choice": "Which option applies to the state?",
                                             "score": "Which level describes the state?",
                                             "noul": ("Is the statement true, or is the answer to the question "
                                                      "yes?")}),
                      Text("\n\nOptions:\n"),
                      OptionLines(criteria={"true": "the statement is true / the answer is yes",
                                            "false": "the statement is false / the answer is no"}),
                      Text("\n\n"), AnswerCue()))

    @staticmethod
    def describe(directory: Path) -> CompositeSource | None:
        path = directory / "config.json"
        if not path.is_file():
            return None
        config = records.record(json.loads(path.read_text()), "config.json")
        overrides = records.record(config.get("modeling_overrides", {}), "modeling_overrides")
        path = directory / "readout.json"
        if not path.is_file() and "readout" not in overrides:
            return None
        fields = ({key: overrides[key] for key in ("readout", "head_dim", "n_slots") if key in overrides}
                  if not path.is_file() else records.record(json.loads(path.read_text()), "readout.json"))
        mode = fields.get("readout", "slot_lm")
        if mode not in ("slot_lm", "head"):
            raise ValueError(f"unknown readout {mode!r}; a slot release reads slot_lm or head")
        count = records.integer(fields.get("n_slots", 256), "n_slots")
        dimension = records.integer(fields.get("head_dim", 256), "head_dim")
        if not 1 <= count <= len(SLOT_LABELS) or dimension < 1:
            raise ValueError("a slot readout has 1-256 slots and a positive head dimension")
        attention = fields.get("attention", "causal")
        if attention not in ("causal", "noncausal"):
            raise ValueError("a slot readout's attention is causal or noncausal")
        if "attention" in overrides:
            masks = records.record(overrides["attention"], "modeling_overrides.attention")
            mask = masks.get("full_attention", "causal")
            if mask not in ("causal", "block_bidirectional"):
                raise ValueError(f"unknown full-attention override {mask!r}")
            declared = "noncausal" if mask == "block_bidirectional" else "causal"
            if "attention" in fields and attention != declared:
                raise ValueError("readout.json and config.json declare different attention policies")
            attention = declared
        native = {key: value for key, value in overrides.items()
                  if key not in ("readout", "head_dim", "n_slots")}
        if attention == "noncausal" and "attention" not in native:
            native["attention"] = {"full_attention": "block_bidirectional"}
        config = {key: value for key, value in config.items() if key != "modeling_overrides"}
        if native:
            config["modeling_overrides"] = native
        tokenizer = HFTokenizer(str(directory), local_files_only=True)
        labels, ids = _labels(tokenizer, count)
        head = (SlotHeadSource({"name": import_path(LabelReadout), "fields": {
                    "label_ids": [list(row) for row in ids]}}, _path)
                if mode == "slot_lm" else SlotHeadSource({"name": import_path(SlotReadout), "fields": {
                    "label_ids": [row[0] for row in ids], "head_dim": dimension, "slot_rows": count}},
                    _path, weight_file="readout_head.pt"))
        layout = SlotSource.layout(labels=labels, blocks=attention == "noncausal")
        return CompositeSource(NativeComponent(config), head, PretrainedDecision.from_components, "",
                               {"layout_record": to_record(layout, Layout),
                                "calibration_record": to_record(Calibration(), Calibration),
                                "strict_specials": False})
