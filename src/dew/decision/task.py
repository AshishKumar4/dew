"""`Decide`: a decision model answering requests, in Dew's types and in Jev's wire form."""

import base64
import binascii
import functools
import io
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Self, overload

import jax
import jax.numpy as jnp
import numpy as np
from etils import epath
from jax.typing import DTypeLike
from PIL import Image

from dew import _records
from dew._records import JSON, json_value, record
from dew.batch import Variables
from dew.checkpoints import Checkpoints
from dew.data.dataset import DataPartition, Reader
from dew.data.text import HFTokenizer, Tokenizer
from dew.decision.calibration import (
    BUCKET_MINIMUM,
    TYPE_MINIMUM,
    Abstention,
    Binning,
    Calibration,
    Scored,
    Temperatures,
)
from dew.decision.data import Example
from dew.decision.head import Head
from dew.decision.images import Images
from dew.decision.laya import LAYA, LayaCheckpoint
from dew.decision.layout import (
    DecisionInputs,
    Encoded,
    JointLayout,
    Laid,
    Layout,
    MarkerLayout,
    Specials,
    render,
    with_context,
)
from dew.decision.model import DecisionModel
from dew.decision.questions import (
    KINDS,
    Answer,
    Choice,
    ChoiceAnswer,
    Confidence,
    JevConfidence,
    Noul,
    NoulAnswer,
    Question,
    ScoreAnswer,
)
from dew.files import write_atomically
from dew.inference import scoring
from dew.inference.tasks import shape_bucket
from dew.interop.processors import Processor
from dew.interop.safetensors_io import MAX_SHARD_SIZE
from dew.interop.sources import SourceReference
from dew.mesh import batch_axes
from dew.nn.cross_attention import ContextEncoder
from dew.nn.inputs import ModelInputs, RowPlan, mesh_of
from dew.nn.kernels.options import program
from dew.objectives.base import Metric, Objective, joined

if TYPE_CHECKING:
    from jax.sharding import Mesh

    from dew.decision.objective import DecisionObjective
    from dew.decision.sources import PretrainedDecision
    from dew.distributed.placement import Layout as Placement, MeshSpec

TASK_FILE = "decide.json"
"""The file `Decide.save` writes into a run: its calibration, budget and name."""


@dataclass(frozen=True)
class Budget:
    """How much one forward pass may hold: at most `tokens` padded tokens and `rows` question rows.

    Laya's `predict_items` uses 16384 and 256. A pass's row count is padded to a
    power of two and its row length to one of at least 64, so passes share
    compiled programs.
    """

    tokens: int = 16384
    rows: int = 256


@dataclass(frozen=True)
class Weights:
    """The checkpoint a task's weights came from: its step, and whether they are the run's average.

    A calibration is fitted to one set of weights, so `save` records them and
    `from_run` loads a saved calibration only for them.
    """

    step: int
    ema: bool
    averaged: tuple[tuple[str, int, bool], ...] = ()
    """The runs' checkpoints these weights are the fp32 mean of (`Decide.averaged`): each run's
    directory, step, and whether its own average was read; none for one run's weights."""


@dataclass(frozen=True)
class Usage:
    """What a request cost: the tokens every question row read, and how much of the state the rows kept.

    The kept share is the smallest any question kept. `trimmed` says whether a
    row shortened a question's instructions or options to fit.
    """

    input_tokens: int
    state_tokens: int
    state_kept: int
    trimmed: bool = False

    @property
    def truncated(self) -> bool:
        return self.state_kept < self.state_tokens


def usage_of(rows: Sequence[Encoded]) -> Usage:
    """What a request's `rows` cost: their tokens, the state's, and the least of it any row kept."""
    return Usage(sum(len(row.tokens) for row in rows), max((row.state_tokens for row in rows), default=0),
                 min((row.state_kept for row in rows), default=0), any(row.trimmed for row in rows))


@dataclass(frozen=True)
class DecideSettings:
    """What a decision run records beside its backbone, and its task is built with
    (`Decide.from_run`): the head's record (`Head.record`), the layout (by import path, its
    segments with it), the special tokens, the text tokenizer's name, and an image processor's
    source, or None."""

    head: JSON
    layout: Layout
    specials: Specials
    tokenizer: str | None
    processor: SourceReference | None
    context_encoder: ContextEncoder = "table"
    """How the model encodes a row's context (`DecisionModel.context_encoder`); a run recorded before
    the choice existed embedded it by its token table."""


@dataclass(frozen=True)
class Decide:
    """Answers typed questions about a state, with one forward pass per batch.

    `decide(state, questions)` returns one answer per question, with its
    probabilities over exactly the question's options. `batch` answers many
    requests together within the `budget`, `tournament` splits a choice too large
    for one pass into rounds, `calibrated` fits the calibration to held-out
    examples, and `systemone` takes and returns TypeSafe Jev's wire format. The
    logits are divided by `calibration`'s temperatures, and each answer reports
    `confidence`'s statistic of its distribution.
    """

    model: DecisionModel
    variables: Variables
    layout: Layout
    tokenizer: Tokenizer
    specials: Specials
    calibration: Calibration = field(default_factory=Calibration)
    confidence: Confidence = field(default_factory=JevConfidence)
    budget: Budget = field(default_factory=Budget)
    name: str = "dew"
    """The model name a Jev response reports."""
    weights: Weights | None = None
    """The run checkpoint the weights came from, when they came from a run."""
    processor: Processor | None = None
    """The backbone's own processor, which prepares a request's images for it."""

    @classmethod
    def from_pretrained(cls, name: str | Path = LAYA, *, subfolder: str | None = None,
                        revision: str | None = None, dtype: str = "float32", param_dtype: str = "float32",
                        attention_impl: str = "auto") -> Self:
        """Load a released Laya or joint-head checkpoint through `Pretrained.load` as a task.

        A Laya task uses the temperatures that Laya's agent reads from
        `rl_agent_config.json`; a joint-head release ships none. Either
        answers with Jev's confidence; `replace(task, confidence=...)` selects
        Laya's (`EntropyConfidence`) or the top probability (`TopProbability`).
        """
        from dew.decision.sources import PretrainedDecision
        from dew.interop.pretrained import Pretrained

        bundle = Pretrained.load(name, subfolder=subfolder, revision=revision, dtype=dtype,
                                  param_dtype=param_dtype, attention_impl=attention_impl)
        if not isinstance(bundle, PretrainedDecision):
            raise TypeError(f"{name} declares no decision task")
        # The name a caller asked for, a Hub folder included, not the cache path it resolved to.
        return cls.of(bundle, name=str(name) if subfolder is None else f"{name}/{subfolder}")

    @classmethod
    def of(cls, bundle: "PretrainedDecision", *, name: str | None = None) -> Self:
        """Build the declared decision task from a loaded composite bundle, without loading again.
        It answers under `name`, by default the Hub id the bundle resolved or else its directory."""
        return cls(DecisionModel(bundle.model, bundle.head),
                   joined({"backbone": bundle.variables, "head": bundle.head_variables}),
                   bundle.layout, bundle.decision_tokenizer, bundle.specials, bundle.calibration,
                   name=name or str(bundle.origin or bundle.source), processor=bundle.processor)

    @classmethod
    def _recorded(cls, directory: str, record: Mapping[str, object], backbone, variables: Variables,
                  weights: Weights, dtype: DTypeLike | None) -> Self:
        """The task a run's `record` describes over `backbone` and `variables`, the `weights` they are:
        its head, layout, tokenizer, special tokens and processor as the run recorded them."""
        from dew.data.text import tokenizer_for
        from dew.inference.tasks import saved_settings
        from dew.interop.pretrained import load_processor

        settings = saved_settings(record, DecideSettings)
        width, head_dtype = DecisionModel.head_size(backbone)
        head = Head.from_record(_records.json_value(settings.head, "head"), width)
        # The head computes in the dtype it recorded, or as a fresh head over
        # the backbone would where the caller overrides the compute dtype.
        model = DecisionModel(backbone, head if dtype is None else head.clone(dtype=head_dtype),
                              settings.context_encoder)
        if settings.tokenizer is None:
            raise ValueError("the run records no tokenizer to lay its rows out with")
        processor = None if settings.processor is None else load_processor(settings.processor, backbone)
        return cls(model, variables, settings.layout, tokenizer_for(settings.tokenizer), settings.specials,
                   name=str(epath.Path(directory).name), weights=weights, processor=processor)

    @classmethod
    def from_run(cls, directory: str, *, ema: bool | None = None, step: int | str | None = None,
                 mesh: "MeshSpec | Mesh | None" = None, layout: "Placement | None" = None,
                 dtype: DTypeLike | None = None, param_dtype: DTypeLike | None = None,
                 trust: Sequence[str] = ()) -> Self:
        """Load the `DecisionObjective` run in `directory` as a task.

        The task is rebuilt from the run's record over the latest checkpoint's
        weights (or `step`'s), and includes any calibration `save` wrote into the
        run. `ema` selects the averaged weights, and None uses them when the run kept
        them. `mesh` and `layout` place the weights as the trainer does, `dtype`
        overrides the compute dtype and `param_dtype` the storage dtype. `trust`
        names the packages outside Dew the run's record may import, as Dew's other
        tasks' `from_run` take it (`dew.inference.tasks.SavedRun.pinned`).
        The image processor reloads from its recorded source independently of
        the tokenizer that lays out the task's text.
        """
        from dew.inference.tasks import SavedRun, saved_model
        from dew.registry import from_record

        run = SavedRun.pinned(directory, step, trust)
        record, step = run.record, run.step
        variables = run.variables(ema=ema, mesh=mesh, layout=layout, param_dtype=param_dtype)
        if ema is None:
            with Checkpoints(directory) as checkpoints:
                averaged = checkpoints.stored(step).get("ema") is not None
        else:
            averaged = ema
        task = cls._recorded(directory, record, saved_model(record, dtype).build(), variables,
                             Weights(step, averaged), dtype)
        saved = epath.Path(directory) / TASK_FILE
        if not saved.exists():
            return task
        settings = _records.record(json.loads(saved.read_text()), TASK_FILE)
        fitted = from_record(Weights, _records.json_value(settings["weights"], "weights"))
        # An average's own directory names the runs it averages beside the step it holds (`save`).
        if (fitted.step, fitted.ema) != (step, averaged):
            raise ValueError(
                f"{TASK_FILE} was fitted on step {fitted.step}'s {'averaged' if fitted.ema else 'live'} "
                f"weights, and this loads step {step}'s {'averaged' if averaged else 'live'} ones; "
                "fit them again with `calibrated` and `save`, or load the step it was fitted on")
        return replace(task, weights=fitted,
                       calibration=from_record(Calibration, _records.json_value(settings["calibration"],
                                                                               "calibration")),
                       budget=from_record(Budget, _records.json_value(settings["budget"], "budget")),
                       name=_records.text(settings["name"], "name"))

    @classmethod
    def averaged(cls, runs: Sequence[str], *, step: int | str | None = None, ema: bool | None = None,
                 calibrate_on: "Iterable[Example | Mapping[str, object]] | Reader | None" = None) -> Self:
        """The task over `runs`' weights as any runs average (`dew.checkpoints.averaged`: their fp32
        mean, each run's at `step`, its latest where None, and `ema` as `from_run` reads them, a LoRA
        backbone's factors merged first), its calibration the identity, or refit on `calibrate_on`'s
        held-out answers (`calibrated`): temperatures are fitted to one set of weights.

        The runs are one task's: their recorded settings (`DecideSettings`: head, layout, tokenizer,
        special tokens, processor and context encoder) the same; a run whose settings differ is
        refused (ValueError), naming them. The task's `weights` record each run's directory, step and
        whether its average was read, so it is no one run's: `save` writes it to a directory of its own
        as a run, which `from_run` and `dew.pipeline` then load, whatever its layout."""
        from dew.checkpoints import averaged
        from dew.inference.tasks import SavedRun, saved_settings

        recorded = [SavedRun.pinned(directory, step).record for directory in runs]
        first = saved_settings(recorded[0], DecideSettings) if recorded else None
        for directory, saved in zip(runs, recorded, strict=True):
            settings = saved_settings(saved, DecideSettings)
            differing = [entry.name for entry in fields(DecideSettings)
                         if getattr(settings, entry.name) != getattr(first, entry.name)]
            if differing:
                raise ValueError(f"the run in {directory} differs from {runs[0]} in its "
                                 f"{', '.join(differing)}, so their weights are not one task's to average")
        average = averaged(runs, step=step, ema=ema)
        task = cls._recorded(runs[0], recorded[0], average.model.build(), average.variables,
                             Weights(average.runs[0][1], ema=False, averaged=average.runs), None)
        return task if calibrate_on is None else task.calibrated(calibrate_on)

    def save_pretrained(self, directory: str | Path, *, max_shard_size: int | str = MAX_SHARD_SIZE) -> None:
        """Write this task as a released checkpoint, which `from_pretrained` reads back.

        A ModernBERT backbone under Laya's head and layout is written in Laya's
        layout (`LayaCheckpoint.save`), the files llama.cpp's converter and
        `/v1/systemone` server read too. The calibration's temperatures go with
        it, held within their bounds as this task applies them, so that a reader
        which does not hold them answers as this task does. A binning map or
        abstention thresholds, which that layout has no place for, are refused
        rather than dropped.
        """
        if self.calibration.binning is not None or self.calibration.abstention is not None:
            raise ValueError("a released checkpoint keeps temperatures alone; save the binning and "
                             "abstention into a run with `save` instead")
        if not isinstance(self.layout, MarkerLayout) or not isinstance(self.tokenizer, HFTokenizer):
            raise ValueError("Laya's layout is a MarkerLayout over a Hugging Face tokenizer")
        applied = self.calibration.temperatures.applied()
        types = applied.types
        LayaCheckpoint(self.model, self.variables, self.layout, self.tokenizer, self.specials,
                       (types.get("choice", 1.0), types.get("score", 1.0), types.get("noul", 1.0)),
                       dict(applied.buckets)).save(directory, max_shard_size=max_shard_size)

    def save(self, directory: str) -> None:
        """Write this task's calibration, budget and name into the run `directory`.

        It also records the checkpoint the weights came from, so `from_run` and
        `dew.pipeline` read the calibration back for that checkpoint. The file is
        written whole or not at all. An average of runs (`averaged`), whose
        weights no one run holds, is written to a `directory` of its own: a run of
        one checkpoint of the average, its optimizer state empty, under the record
        its runs share (`DecisionObjective.inference_record`), its model the base
        an adapter merged into, which `from_run` and `dew.pipeline` then read as
        any run's, whatever its layout; a directory holding another run's
        checkpoints is refused (ValueError).
        """
        from dew.registry import to_record

        if self.weights is None:
            raise ValueError("this task's weights came from no run checkpoint, so a run has no "
                             "calibration of them to hold; save a task from `pipeline` or `from_run`")
        if self.weights.averaged:
            self._claimed(directory)
            # A directory of its own, made before its calibration file is written into it.
            epath.Path(directory).mkdir(parents=True, exist_ok=True)
        write_atomically(epath.Path(directory) / TASK_FILE, json.dumps({
            "weights": to_record(self.weights, Weights),
            "calibration": to_record(self.calibration, Calibration),
            "budget": to_record(self.budget, Budget), "name": self.name}, indent=1) + "\n")
        if self.weights.averaged:
            # The calibration file first: a save stopped before its checkpoint lands leaves a directory
            # that no reader takes for a run, and that a save again completes.
            self._persisted(directory)

    def _claimed(self, directory: str) -> None:
        """Refuse `directory` for this average (`save`) where it holds a run's checkpoints its own
        calibration file does not name as this average's: another run's weights."""
        from dew.registry import from_record

        assert self.weights is not None
        with Checkpoints(directory) as checkpoints:
            if not checkpoints.committed():
                return
        saved = epath.Path(directory) / TASK_FILE
        held = None
        if saved.exists():
            fitted = _records.record(json.loads(saved.read_text()), TASK_FILE)["weights"]
            held = from_record(Weights, _records.json_value(fitted, "weights"))
        if held is None or held.averaged != self.weights.averaged:
            raise ValueError(f"{directory} holds a run's checkpoints other than this average, which its "
                             "calibration does not describe; save the average to a directory of its own")

    def _persisted(self, directory: str) -> None:
        """The average's weights written to `directory`, its calibration file already there, as a run of
        its own (`save`): its state at its weights' step, the optimizer's empty, under its record, where
        the directory holds no checkpoint yet; one it holds is this average's (`_claimed`). Its counters
        and key are placed over the task's mesh (`_mesh`), as a pool's save writes them."""
        import optax

        from dew.decision.objective import DecisionObjective
        from dew.state import TrainState

        assert self.weights is not None
        with Checkpoints(directory) as checkpoints:
            if checkpoints.committed():
                return
            replicated = jax.sharding.NamedSharding(self._mesh, jax.sharding.PartitionSpec())
            counter = jax.device_put(jnp.asarray(self.weights.step, jnp.int32), replicated)
            state = TrainState(step=counter, microstep=counter, updates=counter, variables=self.variables,
                               opt_state=optax.EmptyState(), ema=None,
                               key=jax.device_put(jax.random.key(0), replicated), scale=None,
                               window_size=jax.device_put(jnp.asarray(1, jnp.int32), replicated))
            checkpoints.save(self.weights.step, state, artifact=DecisionObjective(self).inference_record())
            checkpoints.wait()

    def __call__(self, state: JSON, questions: Mapping[str, Question], *,
                 images: Sequence[Image.Image] = (), context: JSON | None = None) -> dict[str, Answer]:
        """Return one answer per question in `questions`, about `state` and any `images`.

        Images need a layout with a place for them (`Layout.images`) and the
        backbone's processor, as a joint-head release brings. `context` is a
        sequence the decision reads beside the state through the backbone's
        cross-attention (`Example.context`).
        """
        if images:
            return self._seen(state, questions, images, context)[0]
        return self.batch([(state, questions)], contexts=[context])[0][0]

    def rows(self, requests: Sequence[tuple[JSON, Mapping[str, Question]]], *,
             contexts: Sequence[JSON | None] | None = None) -> list[list[Encoded]]:
        """Each request's rows as `batch` scores them: its state and questions laid out, with its
        context (`contexts`, in order, or None for none) beside them."""
        contexts = [None] * len(requests) if contexts is None else list(contexts)
        if len(contexts) != len(requests):
            raise ValueError(f"{len(contexts)} contexts for {len(requests)} requests")
        return [self._encoded(state, questions, context=context)
                for (state, questions), context in zip(requests, contexts, strict=True)]

    def batch(self, requests: Sequence[tuple[JSON, Mapping[str, Question]]], *,
              contexts: Sequence[JSON | None] | None = None) -> list[tuple[dict[str, Answer], Usage]]:
        """Return each request's answers and usage.

        The rows of every request are scored together within the budget. `contexts`
        holds each request's context, in order (`__call__`), or None for none.
        """
        return self._answered(self.rows(requests, contexts=contexts),
                              [questions for _, questions in requests])

    def _answered(self, encoded: Sequence[Sequence[Encoded]], asked: Sequence[Mapping[str, Question]]
                ) -> list[tuple[dict[str, Answer], Usage]]:
        """Each request's answers and usage, from the rows it encodes to (`_encoded`) and its
        questions, every request's rows scored together within the budget."""
        requests = list(zip(encoded, asked, strict=True))
        logits = iter(self.logits([row for own, _ in requests for row in own]))
        return [self._replied(questions, own, [next(logits) for _ in own]) for own, questions in requests]

    def _replied(self, questions: Mapping[str, Question], rows: Sequence[Encoded],
                 scores: Sequence[np.ndarray]) -> tuple[dict[str, Answer], Usage]:
        """A request's answers and usage from its rows and each row's raw `[Q, K]` logits (`logits`)."""
        answers = {laid.name: self._answer(questions[laid.name], _ordered(laid, own[slot]))
                   for row, own in zip(rows, scores, strict=True) for slot, laid in enumerate(row.questions)}
        return {name: answers[name] for name in questions}, usage_of(rows)

    def fits(self, rows: Sequence[Encoded]) -> bool:
        """Whether `rows` are one of `logits`' passes: within the budget's rows, and its tokens at the
        pass's bucketed shape (`shape_bucket`)."""
        longest = max(len(row.tokens) for row in rows)
        return (len(rows) <= self.budget.rows
                and shape_bucket(longest, 64) * shape_bucket(len(rows), 1) <= self.budget.tokens)

    def _seen(self, state: JSON, questions: Mapping[str, Question], images: Sequence[Image.Image],
              context: JSON | None = None) -> tuple[dict[str, Answer], Usage]:
        """One request's answers and usage, its images laid out in its one row."""
        return self._looked(questions, *self._shown(state, questions, images, context))

    def _shown(self, state: JSON, questions: Mapping[str, Question], images: Sequence[Image.Image],
               context: JSON | None = None) -> tuple[Encoded, Images]:
        """A request's one row with its images laid out before the state, and the images."""
        if self.processor is None or not self.layout.images:
            raise ValueError("images need a backbone with a vision encoder, which this task does not have")
        media = Images.of(self.processor, images, self.processor.image_text())
        rows = self._encoded(state, questions, media.tokens, context=context)
        if len(rows) != 1:
            raise ValueError("a request's images go in a layout of one row")
        return rows[0], media

    def _looked(self, questions: Mapping[str, Question], row: Encoded, media: Images
                ) -> tuple[dict[str, Answer], Usage]:
        """A request's answers and usage from its one row and images (`_shown`), by a pass of its own."""
        return self._replied(questions, [row], self._seen_pass([(row, media)]))

    def _seen_pass(self, rows: Sequence[tuple[Encoded, Images]],
                   variables: Variables | None = None) -> list[np.ndarray]:
        """Each row's raw `[Q, K]` logits from one pass over rows that hold images, alike as
        `Images.joined` takes them (`seen_together`)."""
        assert self.processor is not None, "a row of images was laid out with the processor"
        encoded = [row for row, _ in rows]
        inputs = DecisionInputs.collate(encoded, self.specials.pad, context_length=self.context_length)
        held = self.variables if variables is None else variables
        scores = np.asarray(self._forward(held, inputs, media=Images.joined(self.processor, rows)))
        return [scores[index, :len(row.questions)] for index, row in enumerate(encoded)]

    @property
    def context_length(self) -> int | None:
        """The most context tokens a request carries (`DecisionModel.context_length`)."""
        return self.model.context_length

    def _encoded(self, state: JSON, questions: Mapping[str, Question],
                 media: Sequence[int] = (), *, context: JSON | None = None) -> list[Encoded]:
        rows = with_context(self.layout.rows(self.tokenizer, self.specials, state, questions, media=media),
                            self.tokenizer, context, self.context_length)
        for row in rows:
            if not self.layout.fits(row):
                raise ValueError(f"the schema needs {len(row.tokens)} tokens besides the state, more than "
                                 f"the maximum context length, max_len={self.layout.max_len}")
        self.model.head.validate_rows(rows)
        for laid in (laid for row in rows for laid in row.questions):
            count = len(questions[laid.name].options)
            if len(laid.options) != count:
                raise ValueError(
                    f"question {laid.name!r}: {len(laid.options)} of its {count} options fit in the "
                    f"maximum context length, max_len={self.layout.max_len}; raise max_len, shorten "
                    "the question, or ask fewer options (`tournament`)")
        return rows

    def _answer(self, question: Question, logits: np.ndarray) -> Answer:
        probabilities = self.calibration.probabilities(question, logits)
        answer = question.answer(probabilities, self.confidence)
        return replace(answer, abstained=self.calibration.abstains(question, probabilities))

    def logits(self, rows: Sequence[Encoded]) -> list[np.ndarray]:
        """Return each row's raw `[Q, K]` logits, one row per question laid out in it, in slot order.

        The rows are scored shortest first, in passes that fit the budget.
        """
        order = sorted(range(len(rows)), key=lambda index: len(rows[index].tokens))
        found: dict[int, np.ndarray] = {}
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and self.fits([rows[index] for index in order[start:stop + 1]]):
                stop += 1
            chosen = order[start:stop]
            found.update(zip(chosen, self.scored([rows[index] for index in chosen]), strict=True))
            start = stop
        return [found[index] for index in range(len(rows))]

    def scored(self, rows: Sequence[Encoded], variables: Variables | None = None) -> list[np.ndarray]:
        """Each of `rows`' raw `[Q, K]` logits from one pass over them all, which `fits` allows, over
        `variables` (the task's own by default)."""
        scores = self._pass(rows, self.variables if variables is None else variables)
        return [scores[index, :len(row.questions)] for index, row in enumerate(rows)]

    def _pass(self, rows: Sequence[Encoded], variables: Variables | None = None) -> np.ndarray:
        """One forward pass at bucketed shapes, its rows placed on the
        weights' mesh (`RowPlan`)."""
        questions = max(len(row.questions) for row in rows)
        options = max(len(laid.options) for row in rows for laid in row.questions)
        inputs = DecisionInputs.collate(rows, self.specials.pad, questions=shape_bucket(questions, 1),
                                        options=shape_bucket(options, 1),
                                        context_length=self.context_length)
        length = shape_bucket(inputs.tokens.shape[1], 64)
        inputs = replace(
            inputs, tokens=_widened(inputs.tokens, length, self.specials.pad),
            segments=_widened(inputs.segments, length),
            positions=None if inputs.positions is None else _widened(inputs.positions, length),
            slots=None if inputs.slots is None else _widened(inputs.slots, length),
            blocks=None if inputs.blocks is None else _widened(inputs.blocks, length, -1))
        count = shape_bucket(len(rows), 1)
        padded = RowPlan(None, len(rows), count, 0, 1).pad(inputs)
        held = self.variables if variables is None else variables
        plan = RowPlan.over(mesh_of(held), count)
        return plan.host(self._forward(held, plan.place(plan.pad(padded))))[:len(rows)]

    def _forward(self, variables: Variables, inputs: DecisionInputs, *,
                 media: ModelInputs | None = None) -> jax.Array:
        """`model.logits`, compiled with the XLA options its backbone and head take where they run
        (`program`)."""
        model = self.model
        logits = program(type(model).logits, model.backbone, model.head, static_argnums=(0,))
        return logits(model, variables, inputs, media=media)

    def tournament(self, state: JSON, questions: Mapping[str, Question], *,
                   group: int = 16) -> dict[str, Answer]:
        """Return answers in which every choice of more than `group` options is decided in rounds.

        The options are cut into near-equal groups of at most `group`, and each
        group's winner goes on, until one group is left to answer the question
        (Laya's `predict_tournament`). A finalist's probabilities cover only the last
        round's options.
        """
        if group < 2:
            raise ValueError("a tournament groups at least two options")
        remaining = {name: list(question.options) for name, question in questions.items()
                     if isinstance(question, Choice) and len(question.options) > group}
        while any(len(options) > group for options in remaining.values()):
            rounds = {}
            for name, options in remaining.items():
                parts = -(-len(options) // group)
                for part in range(parts):
                    chosen = options[part * len(options) // parts:(part + 1) * len(options) // parts]
                    rounds[f"{name}/{part}"] = _restricted(questions[name], chosen)
            answers = self(state, rounds)
            remaining = {name: [_choice(answers[key]) for key in rounds if key.rsplit("/", 1)[0] == name]
                         for name in remaining}
        final = {name: _restricted(question, remaining[name]) if name in remaining else question
                 for name, question in questions.items()}
        return self(state, final)

    def calibrated(self, held_out: Iterable[Example | Mapping[str, object]] | Reader, *,
                   binning: bool = False, target_error: float | None = None,
                   type_minimum: int = TYPE_MINIMUM, bucket_minimum: int = BUCKET_MINIMUM) -> Self:
        """Return this task with temperatures fitted to held-out answers.

        If asked, it also fits a binning map and abstention thresholds that keep the
        error of the accepted answers at `target_error`, as `Calibration`'s parts fit
        them. `held_out` is labelled examples, or a validation reader of decision rows already laid
        out (``dew.data.compile.held(chain, objective=objective.data())``). The
        minimum counts are Laya's: a type's temperature needs `type_minimum` answers
        and a bucket's `bucket_minimum`.
        """
        scored = self._scored(held_out if callable(held_out) else self._held(held_out, self._mesh))
        if not scored:
            raise ValueError("calibration needs examples with answers")
        temperatures = Temperatures.fit(scored, type_minimum=type_minimum, bucket_minimum=bucket_minimum)
        binned = Binning.fit(scored, temperatures) if binning else None
        abstention = (None if target_error is None
                      else Abstention.fit(scored, temperatures, binned, target_error=target_error))
        return replace(self, calibration=Calibration(temperatures, binned, abstention))

    def score(self, examples: Iterable[Example | Mapping[str, object]],
              metrics: Sequence[Metric] = ()) -> dict[str, float]:
        """Return `metrics` over the questions answered for `examples`, through this task's temperatures.

        This is a run's validation pass (`Evaluation.run`) over those questions,
        each probability divided by its temperature; a binning map or abstention
        thresholds are not applied. The default metrics are Accuracy, ECE, AURC
        and the log loss. The pass agrees its result across a process pool, so
        there every process calls it. It scores on the mesh the task's weights are
        placed on, its rows split over the mesh's row axes as a run's validation
        splits them, or on one local device for weights placed on none.
        """
        from dew.decision.scoring import default_metrics
        from dew.training.evaluation import Evaluation

        chosen = list(metrics) or default_metrics()
        evaluation = Evaluation.run(self._scoring, self.variables, self._held(examples, self._mesh),
                                    key=0, metrics=chosen, mesh=self._mesh)
        return {metric.name: evaluation.scores[f"val/{metric.name}"] for metric in chosen}

    @property
    def _mesh(self) -> "Mesh":
        """The mesh the task's weights are placed on, or one local device for weights placed on none."""
        from dew.distributed.placement import MeshSpec

        placed = mesh_of(self.variables)
        return MeshSpec().build(jax.local_devices()[:1]) if placed is None else placed

    def _held(self, examples: Iterable[Example | Mapping[str, object]], mesh: "Mesh") -> Reader:
        """`examples` as a validation reader of this task's rows, laid out as
        its objective evaluates them. A batch is one pass within the budget, as
        `batch` packs its rows, rounded to a multiple of the rows `mesh` splits a
        batch over, so every batch divides; a short last batch is filled as a
        run's validation fills it (`VALID_ROWS`)."""
        from dew.data import Memory
        from dew.data.compile import validation
        from dew.decision.data import decisions

        shards = math.prod(mesh.shape[axis] for axis in batch_axes(mesh))
        rows = max(1, min(self.budget.rows, self.budget.tokens // self.layout.max_len))
        rows = -(-rows // shards) * shards
        rows_of = [{"example": row} if isinstance(row, Example) else row for row in examples]
        held = decisions(Memory(rows_of))
        return validation(held, batch=rows, objective=self._scoring.data(), read="memory")

    @functools.cached_property
    def _scoring(self) -> "DecisionObjective":
        """The objective `score` evaluates through, one for the task, so a
        later call runs the forward the first compiled (`Objective.program`)."""
        from dew.decision.objective import DecisionObjective

        return DecisionObjective(self, temperatures=self.calibration.temperatures)

    def _scored(self, reader: Reader) -> list[Scored]:
        """Each answered question of `reader`'s rows, with its raw logits and its
        gold distribution in the order the row shows its options."""
        scored = []
        segments = self.layout.bounds(self.tokenizer, self.specials, self.context_length).segments
        for batch in reader(DataPartition()):
            inputs = laid_out(batch, segments, spans_in_context=isinstance(self.layout, JointLayout)
                              and self.layout.schema_in_context)
            logits = np.asarray(self._forward(self.variables, inputs))
            answered = (np.asarray(batch["scored"])
                        & (np.asarray(Objective.row_weights(batch, len(logits))) > 0)[:, None])
            for row, slot in zip(*np.nonzero(answered), strict=True):
                count = int(np.sum(batch["options"][row, slot]))
                scored.append(Scored(KINDS[int(batch["kinds"][row, slot])].kind, logits[row, slot, :count],
                                     np.asarray(batch["targets"][row, slot, :count], np.float64)))
        return scored

    def gated(self, min_confidence: float | Mapping[str, float]) -> Self:
        """Return this task abstaining below `min_confidence`.

        `min_confidence` is one threshold for every answer, or one per calibration
        bucket (`bucket`), with the other buckets ungated. An answer abstains when
        its calibrated (and binned) top probability is below its threshold, as with
        Laya's `min_confidence`.
        """
        abstention = (Abstention(dict(min_confidence)) if isinstance(min_confidence, Mapping)
                      else Abstention(default=float(min_confidence)))
        return replace(self, calibration=replace(self.calibration, abstention=abstention))

    @overload
    def systemone(self, request: Mapping[str, object], *, details: bool = False,
                  strict: bool = False, whole: bool = False) -> dict[str, JSON]: ...

    @overload
    def systemone(self, request: Sequence[Mapping[str, object]], *, details: bool = False,
                  strict: bool = False, whole: bool = False) -> list[dict[str, JSON] | ValueError]: ...

    def systemone(self, request, *, details=False, strict=False, whole=False):
        """Answer a Jev request body with a Jev response body, or a list of bodies with a list of
        responses, in order.

        A list's bodies are answered as each would be alone, their rows scored
        together within the budget (`batch`); a body Jev would refuse takes its
        refusal, the ValueError a lone body raises, in its place, and the others
        are answered all the same.

        The request holds `state` and `questions` in Jev's wire format, and
        optionally `model`, which this task ignores. The response holds `model` (this
        task's name), one answer per question, and `usage`. A Noul's answer is
        `noul`; a Choice's is `choice`, `probabilities` and `confidence`; a Score's is
        `score`, `legend`, `probabilities` and `confidence`. Numbers are rounded to
        four places. With `details`, a Noul also reports its confidence, an answer
        reports whether it abstained when the calibration gates, and `usage` reports
        how much of the state it kept.

        Beyond Jev's fields a request may carry `images`: a list of
        images, each a PIL image, encoded image bytes, or a base64 string, bare or as
        a `data:` URL. `strict` answers as Jev's own endpoint does: the request's
        extension fields are dropped and the response holds Jev's fields alone.

        With `whole`, a request that the layout could answer only by cutting its
        state, instructions or options is refused instead, naming the maximum
        context length, as a benchmark that forbids truncation requires
        (Decision Index counts such a refusal as unsupported).
        """
        if isinstance(request, Mapping):
            [answered] = self._systemone([request], details=details, strict=strict, whole=whole)
            if isinstance(answered, ValueError):
                raise answered
            return answered
        return self._systemone(list(request), details=details, strict=strict, whole=whole)

    def _systemone(self, requests: Sequence[Mapping[str, object]], *, details: bool, strict: bool,
                   whole: bool) -> list[dict[str, JSON] | ValueError]:
        """`systemone` over a list of bodies, served as a server serves them (`scoring`,
        `dew.inference.scoring.ScoringServer`): each body admitted on its own, so a refusal is its
        body's alone, and the admitted ones' rows scored together, a pass at a time."""
        server = scoring.ScoringServer.from_task(*self.scoring(details=details, strict=strict, whole=whole))
        answered: dict[int, dict[str, JSON] | ValueError] = {}
        futures = {}
        for index, request in enumerate(requests):
            try:
                futures[index] = server.submit(request)
            except ValueError as refused:
                answered[index] = refused
        server.run()
        answered.update({index: future.result() for index, future in futures.items()})
        return [answered[index] for index in range(len(requests))]

    def scoring(self, *, details: bool = False, strict: bool = False,
                whole: bool = False) -> tuple["WireScoring", Variables]:
        """This task as a scoring task of Jev request bodies (`dew.inference.scoring.Scoring`),
        answering each as `systemone` does with these options, and the weights it scores with. The
        scoring task holds none of them: a server holds them resident and hands them to each pass,
        and swaps both whole on a reload (`ScoringServer.reload`), so an answer always pairs this
        task's calibration with the weights it was fitted to."""
        if strict and details:
            raise ValueError("strict answers hold Jev's fields alone, so they carry no details")
        held = WireScoring(replace(self, variables={}), details=details, strict=strict, whole=whole)
        return held, self.variables

    def _read_wire(self, request: Mapping[str, object], *, strict: bool
                   ) -> tuple[JSON, dict[str, Question], list[Image.Image], JSON | None]:
        """A Jev body's state, questions, images and context (neither where `strict`), refused with a
        ValueError where Jev would refuse it."""
        unknown = set(request) - {"state", "questions", "model", *EXTENSIONS}
        if unknown:
            raise ValueError(f"a request holds state, questions and model, not {sorted(unknown)}")
        if "state" not in request:
            raise ValueError("a request needs a state")
        questions = {name: Question.from_wire(record(wire, f"questions.{name}"))
                     for name, wire in record(request.get("questions"), "questions").items()}
        if not questions:
            raise ValueError("a request needs at least one question")
        state = json_value(request["state"], "state")
        context = None if strict or "context" not in request else json_value(request["context"], "context")
        return state, questions, [] if strict else decoded_images(request.get("images")), context

    def _wire_response(self, questions: Mapping[str, Question], answers: Mapping[str, Answer], usage: Usage,
                       *, details: bool) -> dict[str, JSON]:
        """A Jev response body of a request's answers and usage."""
        gated = self.calibration.abstention is not None
        reported: dict[str, JSON] = {"input_tokens": usage.input_tokens, "output_tokens": 0}
        if details:
            reported.update(state_tokens=usage.state_tokens, state_kept=usage.state_kept,
                            truncated=usage.truncated)
        return {"model": self.name,
                "answers": {name: _wire(questions[name], answer, details=details, gated=gated)
                            for name, answer in answers.items()},
                "usage": reported}


EXTENSIONS = ("images", "context")
"""The request fields `Decide.systemone` reads beyond Jev's, which `strict` drops."""


@dataclass(frozen=True)
class _Admitted:
    """A Jev body as `WireScoring` admitted it: its questions and its rows, laid out by the task."""

    questions: dict[str, Question]
    rows: tuple[Encoded, ...]


@dataclass(frozen=True)
class _Row:
    """One row of an admitted body, a pass's unit: its layout, and its images where it holds any."""

    encoded: Encoded
    media: Images | None = None


@dataclass(frozen=True)
class WireScoring(scoring.Scoring[Mapping[str, object], _Row, _Admitted, np.ndarray, dict[str, JSON]]):
    """A `Decide` task as a scoring task of Jev request bodies (`dew.inference.scoring.Scoring`),
    answering each as `Decide.systemone` does under `details`, `strict` and `whole`.

    `admit` parses and lays out a body, refusing what Jev would refuse, a body the layout could
    answer only by cutting it under `whole` included. A pass is `Decide.logits`' one pass within the
    task's budget (`Decide.fits`) over text rows, or one over rows of images as long as each other and
    of alike images (`Images.joined`); `forward` runs it over the weights it is handed, and `answer`
    reads a body's answers off its rows' logits. `decide` holds no weights (`Decide.scoring`)."""

    decide: Decide
    details: bool = False
    strict: bool = False
    whole: bool = False

    def admit(self, request: Mapping[str, object]) -> scoring.Scored[_Row, _Admitted]:
        decide = self.decide
        state, questions, images, context = decide._read_wire(request, strict=self.strict)
        if images:
            row, media = decide._shown(state, questions, images, context)
            rows = (_Row(row, media),)
        else:
            rows = tuple(_Row(row) for row in decide._encoded(state, questions, context=context))
        usage = usage_of([row.encoded for row in rows])
        if self.whole and (usage.truncated or usage.trimmed):
            cut = "state" if usage.truncated else "instructions or options"
            raise ValueError(f"the request does not fit the maximum context length, max_len="
                             f"{decide.layout.max_len} tokens, without cutting its {cut}")
        return scoring.Scored(rows, _Admitted(questions, tuple(row.encoded for row in rows)))

    def fits(self, rows: Sequence[_Row]) -> bool:
        seen = [row.media for row in rows if row.media is not None]
        if not seen:
            return self.decide.fits([row.encoded for row in rows])
        encoded = [row.encoded for row in rows]
        return (len(seen) == len(rows) and len({len(row.tokens) for row in encoded}) == 1
                and all(seen[0].alike(media) for media in seen) and self.decide.fits(encoded))

    def forward(self, variables: Variables, rows: Sequence[_Row]) -> list[np.ndarray]:
        # The server chose these rows as one pass (`fits`), so they run as one, over the weights it
        # hands the pass; the task's compiled forwards are kept across passes.
        if rows[0].media is None:
            return self.decide.scored([row.encoded for row in rows], variables)
        return self.decide._seen_pass([(row.encoded, row.media) for row in rows if row.media is not None],
                                      variables)

    def answer(self, payload: _Admitted, outputs: Sequence[np.ndarray]) -> dict[str, JSON]:
        return self.decide._wire_response(payload.questions,
                                          *self.decide._replied(payload.questions, payload.rows, outputs),
                                          details=self.details)


def decoded_images(images: object) -> list[Image.Image]:
    """The images of a request's `images` field, None or a list of PIL images,
    encoded image bytes, or base64 strings, bare or as `data:` URLs."""
    if images is None:
        return []
    if not isinstance(images, list):
        raise ValueError("a request's images are a list")
    decoded = []
    for index, image in enumerate(images):
        match image:
            case Image.Image():
                decoded.append(image)
            case bytes():
                decoded.append(_opened(image, index))
            case str():
                payload = image.split(",", 1)[1] if image.startswith("data:") else image
                try:
                    raw = base64.b64decode(payload, validate=True)
                except binascii.Error as error:
                    raise ValueError(f"images[{index}] is not base64: {error}") from error
                decoded.append(_opened(raw, index))
            case _:
                raise ValueError(f"images[{index}] is a PIL image, image bytes or a base64 string, "
                                 f"not {type(image).__name__}")
    return decoded


def _opened(raw: bytes, index: int) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError(f"images[{index}] is not an image Pillow can read: {error}") from error
    return image


def _ordered(laid: Laid, logits: np.ndarray) -> np.ndarray:
    """A question's logits in its own option order, from the slot order its row shows them in, at the
    logits' own precision."""
    ordered = np.empty(len(laid.order), np.asarray(logits).dtype)
    ordered[list(laid.order)] = logits[:len(laid.order)]
    return ordered


def laid_out(batch: Mapping[str, object], max_segments: int, *, spans_in_context: bool = False
             ) -> DecisionInputs:
    """The `DecisionInputs` of a batch of packed windows (`PackedRows`), each
    holding at most `max_segments` rows, the questions' spans in each row's context where
    `spans_in_context` (`JointLayout.schema_in_context`)."""
    def array(name: str) -> jax.Array:
        return jnp.asarray(batch[name])

    def given(name: str) -> jax.Array | None:
        return array(name) if name in batch else None

    return DecisionInputs(tokens=array("tokens"), segments=array("segments"), kinds=array("kinds"),
                          questions=array("questions"), spans=array("spans"),
                          option_spans=array("option_spans"), options=array("options"),
                          positions=array("positions"), slots=given("slots"),
                          option_slots=given("option_slots"), slotted=given("slotted"),
                          context=given("context"), context_lengths=given("context_lengths"),
                          readout=given("readout"), blocks=given("blocks"), max_segments=max_segments,
                          spans_in_context=spans_in_context)


def _widened(leaf: jax.Array, length: int, fill: int = 0) -> jax.Array:
    """A `[B, L]` leaf padded on the right to `length` with `fill`: the
    padding token, or an invalid slot at position and option slot zero."""
    return jnp.pad(leaf, ((0, 0), (0, length - leaf.shape[1])), constant_values=fill)


def _restricted(question: Question, options: Sequence[str]) -> Choice:
    """`question` asked over `options` alone, each with its description."""
    described = dict(zip(question.options, question.descriptions, strict=True))
    return Choice(question.instructions, {option: described[option] for option in options})


def _choice(answer: Answer) -> str:
    assert isinstance(answer, ChoiceAnswer)
    return answer.choice


def _wire(question: Question, answer: Answer, *, details: bool, gated: bool) -> dict[str, JSON]:
    probabilities = [round(float(value), 4) for value in answer.probabilities]
    if isinstance(answer, NoulAnswer):
        wire: dict[str, JSON] = {"type": "noul", "noul": round(answer.noul, 4)}
        if details:
            wire["confidence"] = round(answer.confidence, 4)
    elif isinstance(answer, ChoiceAnswer):
        wire = {"type": "choice", "choice": answer.choice,
                "probabilities": dict(zip(answer.options, probabilities, strict=True)),
                "confidence": round(answer.confidence, 4)}
    else:
        assert isinstance(answer, ScoreAnswer)
        wire = {"type": "score", "score": round(answer.score, 4),
                "legend": {str(level): render(text) for level, text in enumerate(answer.legend)},
                "probabilities": {str(level): value for level, value in enumerate(probabilities)},
                "confidence": round(answer.confidence, 4)}
    if details and gated:
        wire["abstained"] = answer.abstained
    assert isinstance(question, Noul) == isinstance(answer, NoulAnswer)
    return wire



