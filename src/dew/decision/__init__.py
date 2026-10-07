"""Decision models: one forward pass answers typed questions about a state.

A request is a state (text, or any JSON) and named questions, each a
`Noul`, a `Choice` or a `Score` (TypeSafe Jev's three, docs.typesafe.ai).
The model lays the questions out with the state (Laya's `MarkerLayout`, the
`StateFirstLayout` for a causal backbone, or Clef's `JointLayout`), reads
the rows with a backbone, scores every option with a head (Laya's
`DecisionHead` or Clef's `JointSchemaHead`), and answers with a
distribution over exactly the options each question names. `Decide` is the
task that answers requests, and `DecisionObjective` trains one on proper
scoring rules. `LayaCheckpoint` reads convaiinnovations/laya's released
checkpoints, and `ClefCheckpoint` Cloudflare/clef's, whose head `ClefHead` reads.
"""

from dew.decision.calibration import Abstention, Binning, Calibration, Scored, Temperatures, bucket
from dew.decision.clef import ClefCheckpoint, ClefHead
from dew.decision.data import DecisionTable, Example, Weighted
from dew.decision.head import HEADS, DecisionHead, Head, JointSchemaHead
from dew.decision.laya import LayaCheckpoint
from dew.decision.layout import (
    DecisionInputs,
    Encoded,
    JointLayout,
    Laid,
    Layout,
    MarkerLayout,
    QuestionLayout,
    Specials,
    StateFirstLayout,
    render,
)
from dew.decision.metrics import AURC, ECE, Accuracy, Answered
from dew.decision.model import DecisionModel
from dew.decision.objective import NONE_OF_THE_ABOVE, DecisionObjective, Encoding
from dew.decision.questions import (
    KINDS,
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
    kind_of,
)
from dew.decision.scoring import Brier, Combined, LogLoss, RankedProbability, ScoringRule, Spherical
from dew.decision.task import TASK_FILE, Budget, Decide, Usage, Weights

__all__ = [
    "AURC",
    "ECE",
    "HEADS",
    "KINDS",
    "NONE_OF_THE_ABOVE",
    "TASK_FILE",
    "Abstention",
    "Accuracy",
    "Answer",
    "Answered",
    "Binning",
    "Brier",
    "Budget",
    "Calibration",
    "Choice",
    "ChoiceAnswer",
    "ClefCheckpoint",
    "ClefHead",
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
    "Head",
    "JevConfidence",
    "JointLayout",
    "JointSchemaHead",
    "Laid",
    "LayaCheckpoint",
    "Layout",
    "LogLoss",
    "MarkerLayout",
    "Noul",
    "NoulAnswer",
    "Question",
    "QuestionLayout",
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
    "Weighted",
    "Weights",
    "bucket",
    "kind_of",
    "render",
]
