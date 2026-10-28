"""JEPA: its objective, masks and probes, and the encoder and predictor it trains, each imported
from its module when first used, so importing one of the package's modules, or a name from one,
reaches only what that module imports."""

from typing import TYPE_CHECKING

from dew._exports import lookup as _lookup

if TYPE_CHECKING:  # written from _EXPORTS (tools/gen_exports.py)
    from dew.nn.backbones.jepa import (
        FactorizedTokenStack,
        JepaEncoder,
        JepaPredictor,
        JepaVideoEncoder,
        TokenStack,
    )
    from dew.objectives.jepa.masking import MultiBlockMask
    from dew.objectives.jepa.objective import JepaObjective, normalize_targets, representation_health
    from dew.objectives.jepa.probes import KnnProbe, LinearProbe, knn_probe_accuracy, linear_probe_accuracy

_EXPORTS = {
    "FactorizedTokenStack": "dew.nn.backbones.jepa", "JepaEncoder": "dew.nn.backbones.jepa",
    "JepaObjective": "dew.objectives.jepa.objective", "JepaPredictor": "dew.nn.backbones.jepa",
    "JepaVideoEncoder": "dew.nn.backbones.jepa", "KnnProbe": "dew.objectives.jepa.probes",
    "LinearProbe": "dew.objectives.jepa.probes", "MultiBlockMask": "dew.objectives.jepa.masking",
    "TokenStack": "dew.nn.backbones.jepa", "knn_probe_accuracy": "dew.objectives.jepa.probes",
    "linear_probe_accuracy": "dew.objectives.jepa.probes",
    "normalize_targets": "dew.objectives.jepa.objective",
    "representation_health": "dew.objectives.jepa.objective",
}
__getattr__ = _lookup(__name__, _EXPORTS)

__all__ = [
    "FactorizedTokenStack",
    "JepaEncoder",
    "JepaObjective",
    "JepaPredictor",
    "JepaVideoEncoder",
    "KnnProbe",
    "LinearProbe",
    "MultiBlockMask",
    "TokenStack",
    "knn_probe_accuracy",
    "linear_probe_accuracy",
    "normalize_targets",
    "representation_health",
]


def __dir__() -> list[str]:
    return list(__all__)
