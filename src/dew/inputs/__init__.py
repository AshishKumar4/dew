"""What a generative objective is fed: the sample field and its conditions (`dew.inputs.spec`), the
condition encoders (`dew.inputs.encoders`, their contract `dew.inputs.condition`), and the encoders and
a diffusion model's conditioner that bind a released checkpoint (`dew.interop.encoders`,
`dew.interop.conditioners`), each imported from its module when first used, so importing one of them, or a
name from one, reaches only what that module imports."""

from typing import TYPE_CHECKING

from dew._exports import lookup as _lookup

if TYPE_CHECKING:  # written from _EXPORTS (tools/gen_exports.py)
    from dew.inputs.condition import ConditionEncoder
    from dew.inputs.encoders import CharTable
    from dew.interop.conditioners import DiffusionConditioner
    from dew.interop.encoders import CLIPText, HFAudio, T5Text
    from dew.inputs.spec import (
        AudioCondition,
        Condition,
        Field,
        InputSpec,
        OutsideTable,
        TextCondition,
        encode_conditions,
        unit_range,
    )

_EXPORTS = {
    "AudioCondition": "dew.inputs.spec", "CLIPText": "dew.interop.encoders",
    "CharTable": "dew.inputs.encoders", "Condition": "dew.inputs.spec",
    "ConditionEncoder": "dew.inputs.condition", "DiffusionConditioner": "dew.interop.conditioners",
    "Field": "dew.inputs.spec", "HFAudio": "dew.interop.encoders", "InputSpec": "dew.inputs.spec",
    "OutsideTable": "dew.inputs.spec",
    "T5Text": "dew.interop.encoders", "TextCondition": "dew.inputs.spec",
    "encode_conditions": "dew.inputs.spec", "unit_range": "dew.inputs.spec",
}
__getattr__, __dir__ = _lookup(__name__, _EXPORTS)

__all__ = [
    "AudioCondition",
    "CLIPText",
    "CharTable",
    "Condition",
    "ConditionEncoder",
    "DiffusionConditioner",
    "Field",
    "HFAudio",
    "InputSpec",
    "OutsideTable",
    "T5Text",
    "TextCondition",
    "encode_conditions",
    "unit_range",
]
