"""The decision release formats, each once: where its backbone goes, its head's own files and what its record
states beside them (EX02).

A head names its format (`Head.release_format`); a decision bundle's release is its backbone's
(`dew.interop.releases.bundle_plan`) placed as its format places it, its head's files beside it and its
record the backbone's with the decision facts the format's files cannot state
(`dew.decision.release.decision_envelope`). Every writer and every export of a decision release is
`decision_plan`'s: `Decide.save_pretrained`, `PretrainedDecision.save` and `export`, `dew export` of a
decision run.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from dew.decision.calibration import Calibration
from dew.decision.release import decision_envelope
from dew.interop.release import computed_in
from dew.interop.releases import Assets, HeadFiles, Placed, ReleasePlan, bundle_plan

if TYPE_CHECKING:
    from flax import linen as nn

    from dew.decision.layout import Layout
    from dew.decision.sources import PretrainedDecision
    from dew.interop.layouts import Unwritable


@dataclasses.dataclass(frozen=True)
class DecisionFormat:
    """One decision release format: its head's files (`head`), its record's decision facts (`stated`: the
    layout, max length and calibration its files cannot state), and where its backbone goes: its config's
    folder, its tensors' names (`prefix` for `strip`) and what its config `states` beside its family's own
    (`dew.interop.releases.Placed`), its tokenizer's folder and whether it holds the backbone's generation
    config; `model_type` the family it holds where it holds one alone (Laya's ModernBERT encoder)."""

    name: str
    head: Callable[[PretrainedDecision], HeadFiles]
    stated: Callable[[PretrainedDecision], tuple[Layout | None, int | None, Calibration | None]]
    folder: str = ""
    prefix: str = ""
    strip: str = ""
    states: Mapping[str, object] = dataclasses.field(default_factory=dict)
    tokenizer_folder: str = ""
    generation: bool = True
    model_type: str | None = None


def _formats() -> dict[str, DecisionFormat]:
    from dew.decision import joint_checkpoint, laya
    from dew.decision.slot_source import SlotSource

    return {
        "laya": DecisionFormat("laya", laya.head_files, laya.stated, folder="encoder", prefix="encoder.",
                               strip="model.", states={"architectures": ["ModernBertModel"]},
                               tokenizer_folder="tokenizer", generation=False, model_type="modernbert"),
        "joint": DecisionFormat("joint", joint_checkpoint.head_files, joint_checkpoint.stated),
        "slots": DecisionFormat("slots", SlotSource.head_files, SlotSource.stated),
    }


def decision_format(name: str | None) -> DecisionFormat:
    """The format a head names (`Head.release_format`), refused where it names none Dew writes."""
    formats = _formats()
    if name not in formats:
        raise ValueError(f"a head released in {name!r} has no release format; Dew writes "
                         f"{', '.join(sorted(formats))}")
    return formats[name]


def format_unstated(name: str | None, backbone: nn.Module) -> Unwritable | None:
    """Why no release of a head released in `name` (`Head.release_format`) states `backbone`: no format of
    that name, no family's layout that writes the backbone (`dew.interop.layouts.source_layout`), or a
    format that holds another family; None where one does. Read off the model alone, before its weights
    are."""
    from dew.interop.layouts import Unwritable, source_layout

    formats = _formats()
    if name not in formats:
        return Unwritable(f"a head released in {name!r} has no release format; Dew writes "
                          f"{', '.join(sorted(formats))}")
    layout = source_layout(backbone)
    if isinstance(layout, Unwritable):
        return layout
    return _holds(formats[name], layout.config)


def _holds(chosen: DecisionFormat, config: Mapping[str, object]) -> Unwritable | None:
    from dew.interop.layouts import Unwritable

    held = config.get("model_type")
    if chosen.model_type is None or held == chosen.model_type:
        return None
    return Unwritable(f"the {chosen.name} format holds a {chosen.model_type} backbone, not a {held}")


def decision_plan(bundle: PretrainedDecision) -> ReleasePlan | Unwritable:
    """A decision bundle's release in its head's format: its backbone's (`bundle_plan`) placed as the format
    places it, its head's files, its decision tokenizer in the format's folder and its record the
    backbone's with the facts the format's files cannot state; why none writes it where the format holds
    another family than its backbone's."""
    chosen = decision_format(bundle.head.release_format)
    refused = _holds(chosen, bundle.config)
    if refused is not None:
        return refused
    backbone = bundle_plan(bundle.backbone())
    assert isinstance(backbone.model, Placed) and backbone.assets is not None
    placed = dataclasses.replace(backbone.model, folder=chosen.folder, prefix=chosen.prefix,
                                 strip=chosen.strip, stated=chosen.states)
    assets = Assets(tokenizer=bundle.decision_tokenizer, folder=chosen.tokenizer_folder,
                    generation_config=backbone.assets.generation_config if chosen.generation else None)
    layout, max_length, calibration = chosen.stated(bundle)
    # Every bundle states the dtype it computes in, a task's and one read from a release alike, so a load of
    # what it writes computes as it does.
    compute = computed_in(bundle.numerics, bundle.model, bundle.variables)
    envelope = decision_envelope(backbone.envelope, layout, calibration, max_length=max_length,
                                 numerics=bundle.numerics, compute=compute)
    return ReleasePlan(placed, chosen.head(bundle), assets, envelope)


def beyond_temperatures(calibration: Calibration) -> Calibration | None:
    """`calibration` where it holds more than temperatures (a binning, abstention thresholds), which a
    format's files hold alone; else None."""
    return calibration if calibration.binning is not None or calibration.abstention is not None else None


def beyond_default(calibration: Calibration) -> Calibration | None:
    """`calibration` where it is another than the default, which a format holding none states in its
    record; else None."""
    return None if calibration == Calibration() else calibration


__all__ = ["DecisionFormat", "beyond_default", "beyond_temperatures", "decision_format", "decision_plan",
           "format_unstated"]
