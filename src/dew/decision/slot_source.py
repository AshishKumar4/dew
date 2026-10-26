"""The slot-readout release's metadata, labels and prompt, over native source components."""

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew import _records
from dew.batch import Variables
from dew.data.text import HFTokenizer
from dew.interop.composite import CompositeSource, HeadComponent, NativeComponent, SourceKind
from dew.interop.pretrained import Pretrained, with_overrides
from dew.interop.safetensors_io import MAX_SHARD_SIZE
from dew.registry import dtype_name, from_record, import_path

from .calibration import Calibration
from .head import Head, LabelReadout, SlotReadout
from .layout import (
    LETTERS,
    AnswerCue,
    Framing,
    Instructions,
    Labels,
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
_PATHS = {"slot": ("slots",), "q.weight": ("q",), "k.weight": ("k",)}
_LABELS = Labels(variants=("{}",), names=SLOT_LABELS)


def _path(name: str) -> tuple[str, ...] | None:
    return _PATHS.get(name)


def _attention(overrides: Mapping[str, object]) -> Mapping[str, object]:
    """The attention modes a `modeling_overrides` record states, by layer kind."""
    return _records.record(overrides.get("attention", {}), "modeling_overrides.attention")


def _full_attention(overrides: Mapping[str, object]) -> str:
    """The full-attention layers' mode a `modeling_overrides` record states: causal, or the release's
    block_bidirectional; causal where it states none. The release's `attention` field states it."""
    mask = _attention(overrides).get("full_attention", "causal")
    if mask not in ("causal", "block_bidirectional"):
        raise ValueError(f"unknown full-attention override {mask!r}")
    return mask


def _initialize(head: nn.Module, backbone: Pretrained) -> Variables:
    return DecisionModel(backbone.model, from_record(Head, head)).initialize_head(
        jax.random.key(0), backbone.variables, jnp.zeros((1, 1), jnp.int32))


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
    def build(self, backbone: Pretrained | None, dtype: str, attention_impl: str, *,
              shapes: Mapping[str, tuple[int, ...]] | None) -> Head:
        if backbone is None:
            raise ValueError("a slot readout needs its native backbone")
        width, compute = DecisionModel.head_size(backbone.model)
        held = _records.record(self.record, "head")
        fields = dict(_records.record(held["fields"], "head.fields"))
        if self.initialize is not None:
            labels = _records.integers(fields["label_ids"], "label_ids")
            if shapes is None:
                fields["slot_rows"] = len(labels)
            else:
                if set(shapes) != _PATHS.keys():
                    raise ValueError("a stored slot head holds slot, q.weight and k.weight")
                slot, query, key = (shapes[name] for name in ("slot", "q.weight", "k.weight"))
                if (len(slot) != 2 or slot[1] != width or slot[0] < len(labels) or len(query) != 2
                        or query[1] != width or query[0] < 1 or key != query):
                    raise ValueError("a stored slot head's slots and projections must match its backbone")
                fields.update(slot_rows=slot[0], head_dim=query[0])
        return Head.from_record(_records.json_value({**held, "fields": {
            **fields, "dtype": None if compute is None else dtype_name(compute)}}, "head"), width)


class SlotSource(SourceKind):
    """Readout metadata beside a native decoder, with the release's prompt as a layout value."""

    @staticmethod
    def save(bundle: PretrainedDecision, directory: str | Path, *,
             max_shard_size: int | str = MAX_SHARD_SIZE) -> None:
        """Write the held native backbone and the release's readout, without loading the source again."""
        head = _records.record(bundle.head.record(), "head")
        fields = _records.record(head["fields"], "head.fields")
        names = {import_path(LabelReadout): "slot_lm", import_path(SlotReadout): "head"}
        mode = names.get(_records.text(head["class"], "head.class"))
        if mode is None or fields.get("untied", False):
            raise ValueError("the slot format holds its tied label readout or its slot-and-pointer head")
        layout = bundle.layout
        if not isinstance(layout, QuestionLayout) or layout.labels is None:
            raise ValueError("the slot format needs its whole-question layout")
        labels = layout.labels
        expected = SlotSource.layout(max_len=layout.max_len, labels=labels, blocks=layout.blocks)
        if layout != expected or bundle.calibration != Calibration():
            raise ValueError("the slot format holds its release prompt and uncalibrated logits")
        ids = _records.listed(fields["label_ids"], "label_ids", read=lambda value, _: value,
                             what="a list of label ids")
        canonical, picked = _labels(bundle.decision_tokenizer, len(ids))
        actual = (tuple(_records.integers(group, "label_ids") for group in ids) if mode == "slot_lm"
                  else tuple((_records.integer(value, "label_ids"),) for value in ids))
        if labels != canonical or actual != picked:
            raise ValueError("the slot format records its canonical bare labels in tokenizer order")
        root = Path(directory)
        Pretrained.save(bundle, root, max_shard_size=max_shard_size)
        readout = {"readout": mode, "head_dim": fields.get("head_dim", 256), "n_slots": len(ids)}
        # The release's attention field states its full attention as the saved config's overrides do;
        # the rows' blocks may serve another kind as well.
        written = _records.record(json.loads((root / "config.json").read_text()), "config.json")
        if _full_attention(_records.record(written.get("modeling_overrides", {}),
                                           "modeling_overrides")) == "block_bidirectional":
            readout["attention"] = "noncausal"
        bundle.decision_tokenizer.save_pretrained(root)
        if mode == "head":
            import torch

            params = bundle.head_variables["params"]
            tensors = {}
            for name, path in _PATHS.items():
                host = np.array(jax.device_get(params[path[0]]), copy=True, order="C")
                tensors[name] = torch.frombuffer(host, dtype=getattr(torch, str(host.dtype))).reshape(
                    host.shape)
            torch.save(tensors, root / "readout_head.pt")
        (root / "readout.json").write_text(json.dumps(readout, indent=2) + "\n")

    @staticmethod
    def layout(*, max_len: int = 32768, keep: Literal["start", "end", "all"] = "all",
               labels: Labels = _LABELS,
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
                                             "noul": ("Is the statement true, or is the answer to the "
                                                      "question yes?")}),
                      Text("\n\nOptions:\n"),
                      OptionLines(criteria={"true": "the statement is true / the answer is yes",
                                            "false": "the statement is false / the answer is no"}),
                      Text("\n\n"), AnswerCue()))

    @staticmethod
    def describe(directory: Path, overrides: Mapping[str, object] | None = None) -> CompositeSource | None:
        path = directory / "config.json"
        if not path.is_file():
            return None
        config = _records.record(json.loads(path.read_text()), "config.json")
        released = _records.record(config.get("modeling_overrides", {}), "modeling_overrides")
        path = directory / "readout.json"
        if not path.is_file() and "readout" not in released:
            return None
        fields = ({key: released[key] for key in ("readout", "head_dim", "n_slots") if key in released}
                  if not path.is_file() else _records.record(json.loads(path.read_text()), "readout.json"))
        unknown = set(fields) - {"readout", "head_dim", "n_slots", "attention"}
        if unknown:
            raise ValueError(f"unknown readout fields {sorted(unknown)}")
        mode = fields.get("readout", "slot_lm")
        if mode not in ("slot_lm", "head"):
            raise ValueError(f"unknown readout {mode!r}; a slot release reads slot_lm or head")
        count = _records.integer(fields.get("n_slots", 256), "n_slots")
        dimension = _records.integer(fields.get("head_dim", 256), "head_dim")
        if count < 1 or dimension < 1:
            raise ValueError("a slot readout has a positive slot count and head dimension")
        attention = fields.get("attention", "causal")
        if attention not in ("causal", "noncausal"):
            raise ValueError("a slot readout's attention is causal or noncausal")
        if "attention" in released:
            declared = "noncausal" if _full_attention(released) == "block_bidirectional" else "causal"
            if "attention" in fields and attention != declared:
                raise ValueError("readout.json and config.json declare different attention policies")
            attention = declared
        native = {key: value for key, value in released.items()
                  if key not in ("readout", "head_dim", "n_slots")}
        if attention == "noncausal" and "attention" not in native:
            native["attention"] = {"full_attention": "block_bidirectional"}
        config = {key: value for key, value in config.items() if key != "modeling_overrides"}
        if native:
            config["modeling_overrides"] = native
        # The load's own overrides over the release's: its rows lie in blocks exactly where a layer of
        # the backbone the native loader builds under them reads blocks, of whatever kind.
        built = _records.record(with_overrides(config, overrides).get("modeling_overrides", {}),
                                "modeling_overrides")
        blocks = "block_bidirectional" in _attention(built).values()
        tokenizer = HFTokenizer(str(directory), local_files_only=True)
        labels, ids = _labels(tokenizer, count)
        head = (SlotHeadSource({"class": import_path(LabelReadout), "fields": {
                    "label_ids": [list(row) for row in ids]}}, _path)
                if mode == "slot_lm" else SlotHeadSource({"class": import_path(SlotReadout), "fields": {
                    "label_ids": [row[0] for row in ids], "head_dim": dimension, "slot_rows": count}},
                    _path, weight_file="readout_head.pt", initialize=_initialize))
        layout = SlotSource.layout(labels=labels, blocks=blocks)
        return CompositeSource(NativeComponent(config), head, PretrainedDecision.from_components, "",
                               {"layout": layout, "calibration": Calibration(), "strict_specials": False,
                                "writer": SlotSource.save})
