"""Decision models: one forward pass answers typed questions about a state.

A request is a state (text, or any JSON) and named questions, each a
`Noul`, a `Choice` or a `Score` (TypeSafe Jev's three, docs.typesafe.ai).
The model lays the questions out with the state (Laya's `MarkerLayout`, the
`StateFirstLayout` for a causal backbone, or the `JointLayout`, every
question of a request in one row), reads the rows with a backbone, scores
every option with a head (Laya's `DecisionHead`, or the `JointSchemaHead`,
which decides a row's questions together, or a readout of options written under
`Labels`: `LabelReadout`, the backbone's own logits at them, and a slot release's
`SlotReadout`), and answers with a distribution
over exactly the options each question names. `Decide` is the task that
answers requests, and `DecisionObjective` trains one on proper scoring
rules. `Pretrained.load` reads released Laya and joint-head sources, and
`Decide.of` turns their declared settings into a decision task.
"""

from dew.decision.calibration import Abstention, Binning, Calibration, Scored, Temperatures, bucket
from dew.decision.data import Example, Group, GroupedExamples, corpus, decisions, mixture
from dew.decision.head import DecisionHead, Head, JointSchemaHead, LabelReadout, SlotReadout
from dew.decision.laya import LayaCheckpoint
from dew.decision.layout import (
    AnswerCue,
    DecisionInputs,
    Encoded,
    Framing,
    Instructions,
    JointLayout,
    Labels,
    Laid,
    Layout,
    MarkerLayout,
    Media,
    OptionLines,
    QuestionLayout,
    QuestionSegment,
    Schema,
    Segment,
    Specials,
    State,
    StateFirstLayout,
    Text,
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
    "KINDS",
    "NONE_OF_THE_ABOVE",
    "TASK_FILE",
    "Abstention",
    "Accuracy",
    "Answer",
    "AnswerCue",
    "Answered",
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
    "Encoded",
    "Encoding",
    "EntropyConfidence",
    "Example",
    "Framing",
    "Group",
    "GroupedExamples",
    "Head",
    "Instructions",
    "JevConfidence",
    "JointLayout",
    "JointSchemaHead",
    "LabelReadout",
    "Labels",
    "Laid",
    "LayaCheckpoint",
    "Layout",
    "LogLoss",
    "MarkerLayout",
    "Media",
    "Noul",
    "NoulAnswer",
    "OptionLines",
    "Question",
    "QuestionLayout",
    "QuestionSegment",
    "RankedProbability",
    "Schema",
    "Score",
    "ScoreAnswer",
    "Scored",
    "ScoringRule",
    "Segment",
    "SlotReadout",
    "Specials",
    "Spherical",
    "State",
    "StateFirstLayout",
    "Temperatures",
    "Text",
    "TopProbability",
    "Usage",
    "Weights",
    "bucket",
    "corpus",
    "decisions",
    "kind_of",
    "mixture",
    "render",
]
