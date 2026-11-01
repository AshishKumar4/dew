"""Diffusion objectives and their distillations, each imported from its module when first used, so
importing one of the package's modules, or a name from one, reaches only what that module imports."""

from typing import TYPE_CHECKING

from dew._exports import lookup as _lookup

if TYPE_CHECKING:  # written from _EXPORTS (tools/gen_exports.py)
    from dew.objectives.diffusion.adversarial import AdversarialDistillationObjective
    from dew.objectives.diffusion.alignment import Alignment
    from dew.objectives.diffusion.block import BlockDiffusionObjective
    from dew.objectives.diffusion.consistency import ConsistencyDistillationObjective
    from dew.objectives.diffusion.end_to_end import EndToEnd
    from dew.objectives.diffusion.few_step import MeanFlowObjective, ShortcutObjective
    from dew.objectives.diffusion.guidance_distillation import GuidanceDistillationObjective
    from dew.objectives.diffusion.masked import MaskedDiffusionObjective
    from dew.objectives.diffusion.objective import VALIDATION_SAMPLES, DiffusionObjective

_EXPORTS = {
    "VALIDATION_SAMPLES": "dew.objectives.diffusion.objective",
    "AdversarialDistillationObjective": "dew.objectives.diffusion.adversarial",
    "Alignment": "dew.objectives.diffusion.alignment",
    "BlockDiffusionObjective": "dew.objectives.diffusion.block",
    "ConsistencyDistillationObjective": "dew.objectives.diffusion.consistency",
    "DiffusionObjective": "dew.objectives.diffusion.objective",
    "EndToEnd": "dew.objectives.diffusion.end_to_end",
    "GuidanceDistillationObjective": "dew.objectives.diffusion.guidance_distillation",
    "MaskedDiffusionObjective": "dew.objectives.diffusion.masked",
    "MeanFlowObjective": "dew.objectives.diffusion.few_step",
    "ShortcutObjective": "dew.objectives.diffusion.few_step",
}
__getattr__ = _lookup(__name__, _EXPORTS)

__all__ = [
    "VALIDATION_SAMPLES",
    "AdversarialDistillationObjective",
    "Alignment",
    "BlockDiffusionObjective",
    "ConsistencyDistillationObjective",
    "DiffusionObjective",
    "EndToEnd",
    "GuidanceDistillationObjective",
    "MaskedDiffusionObjective",
    "MeanFlowObjective",
    "ShortcutObjective",
]


def __dir__() -> list[str]:
    return list(__all__)
