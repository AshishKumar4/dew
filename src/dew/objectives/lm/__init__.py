"""Language-model objectives, each imported from its module when first used, so importing one of
the package's modules, or a name from one, reaches only what that module imports."""

from typing import TYPE_CHECKING

from dew._exports import lookup as _lookup

if TYPE_CHECKING:  # written from _EXPORTS (tools/gen_exports.py)
    from dew.objectives.lm.adaptation import ChunkAdaptation
    from dew.objectives.lm.objective import IndexerTraining, LMObjective, Perplexity, Samples, TokenScorer

_EXPORTS = {
    "ChunkAdaptation": "dew.objectives.lm.adaptation",
    "IndexerTraining": "dew.objectives.lm.objective",
    "LMObjective": "dew.objectives.lm.objective", "Perplexity": "dew.objectives.lm.objective",
    "Samples": "dew.objectives.lm.objective", "TokenScorer": "dew.objectives.lm.objective",
}
__getattr__, __dir__ = _lookup(__name__, _EXPORTS)

__all__ = [
    "ChunkAdaptation",
    "IndexerTraining",
    "LMObjective",
    "Perplexity",
    "Samples",
    "TokenScorer",
]
