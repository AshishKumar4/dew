"""Decision models: one forward pass answers typed questions about a state.

A request is a state (text, or any JSON) and named questions, each a
`Noul`, a `Choice` or a `Score` (TypeSafe Jev's three, docs.typesafe.ai).
The model lays each question out with the state (`MarkerLayout`,
`StateFirstLayout`), reads the row with a backbone, scores every option
with a `DecisionHead`, and answers with a distribution over exactly the
options the question names. `Decide` is the task that answers requests,
and `DecisionObjective` trains one on proper scoring rules.
`LayaCheckpoint` reads convaiinnovations/laya's released checkpoints.
"""

from dew.decision.calibration import Abstention, Binning, Calibration, Scored, Temperatures, bucket
from dew.decision.data import DecisionTable, Example
from dew.decision.head import KINDS, DecisionHead, kind_of
from dew.decision.laya import LayaCheckpoint
from dew.decision.layout import (
    DecisionInputs,
    Encoded,
    Layout,
    MarkerLayout,
    Specials,
    StateFirstLayout,
    render,
)
from dew.decision.metrics import AURC, ECE, Accuracy
from dew.decision.model import DecisionModel
from dew.decision.objective import NONE_OF_THE_ABOVE, DecisionObjective, Encoding
from dew.decision.questions import (
    Answer,
    Choice,
    ChoiceAnswer,
    Confidence,
    EntropyConfidence,
    JevConfidence,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    TopProbability,
)
from dew.decision.scoring import Brier, Combined, LogLoss, RankedProbability, ScoringRule, Spherical
from dew.decision.task import TASK_FILE, Budget, Decide, Usage, Weights

__all__ = [
    "AURC",
    "ECE",
    "KINDS",
    "NONE_OF_THE_ABOVE",
    "TASK_FILE",
    "Abstention",
    "Accuracy",
    "Answer",
    "Binning",
    "Brier",
    "Budget",
    "Calibration",
    "Choice",
    "ChoiceAnswer",
    "Combined",
    "Confidence",
    "Decide",
    "DecisionHead",
    "DecisionInputs",
    "DecisionModel",
    "DecisionObjective",
    "DecisionTable",
    "Encoded",
    "Encoding",
    "EntropyConfidence",
    "Example",
    "JevConfidence",
    "LayaCheckpoint",
    "Layout",
    "LogLoss",
    "MarkerLayout",
    "Noul",
    "NoulAnswer",
    "Question",
    "RankedProbability",
    "Score",
    "ScoreAnswer",
    "Scored",
    "ScoringRule",
    "Specials",
    "Spherical",
    "StateFirstLayout",
    "Temperatures",
    "TopProbability",
    "Usage",
    "Weights",
    "bucket",
    "kind_of",
    "render",
]
