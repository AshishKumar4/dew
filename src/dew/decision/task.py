"""`Decide`: a decision model answering requests, in Dew's types and in Jev's wire form."""

import base64
import binascii
import functools
import io
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Self

import jax
import jax.numpy as jnp
import numpy as np
from etils import epath
from jax.typing import DTypeLike
from PIL import Image

from dew import records
from dew.artifacts import Decisions
from dew.checkpoints import Checkpoints
from dew.data.dataset import DataPartition, Reader
from dew.data.text import HFTokenizer, Tokenizer
from dew.decision.calibration import Abstention, Binning, Calibration, Scored, Temperatures, softmax
from dew.decision.clef import ClefCheckpoint
from dew.decision.data import Example
from dew.decision.head import Head
from dew.decision.images import Images
from dew.decision.laya import LayaCheckpoint
from dew.decision.layout import (
    DecisionInputs,
    Encoded,
    JointLayout,
    Laid,
    Layout,
    MarkerLayout,
    Specials,
    StateFirstLayout,
    render,
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
    Score,
    ScoreAnswer,
)
from dew.files import write_atomically
from dew.inference.tasks import SHAPE_BUCKETS
from dew.interop.processors import Processor
from dew.nn.inputs import RowPlan, mesh_of
from dew.objectives.base import Metric, Variables
from dew.records import JSON, json_value, record

if TYPE_CHECKING:
    from dew.training.distributed import Layout as Placement, MeshSpec

TASK_FILE = "decide.json"
"""The file `Decide.save` writes into a run: its calibration, budget and name."""
_LAYOUTS: Mapping[str, type[Layout]] = {
    layout.__name__: layout for layout in (MarkerLayout, StateFirstLayout, JointLayout)}


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
    def from_pretrained(cls, name: str | Path = "convaiinnovations/laya", *, subfolder: str | None = None,
                        revision: str | None = None, dtype: str = "float32", param_dtype: str = "float32",
                        attention_impl: str = "auto") -> Self:
        """Load a released Laya (`LayaCheckpoint`) or Clef (`ClefCheckpoint`) checkpoint as a task.

        A Laya task uses the temperatures that Laya's agent reads from
        `rl_agent_config.json`; Clef ships none. Either answers with Jev's
        confidence; `replace(task, confidence=...)` selects Laya's
        (`EntropyConfidence`) or Clef's (`TopProbability`).
        """
        label = str(name) if subfolder is None else f"{name}/{subfolder}"
        if subfolder is None and ClefCheckpoint.exists(name, revision=revision):
            clef = ClefCheckpoint.load(name, revision=revision, dtype=dtype, param_dtype=param_dtype,
                                       attention_impl=attention_impl)
            return cls(clef.model, clef.variables, JointLayout(), clef.tokenizer, clef.specials, name=label,
                       processor=clef.processor)
        checkpoint = LayaCheckpoint.load(name, subfolder=subfolder, revision=revision, dtype=dtype,
                                         param_dtype=param_dtype, attention_impl=attention_impl)
        choice, score, noul = checkpoint.temperatures
        temperatures = Temperatures({"choice": choice, "score": score, "noul": noul},
                                    checkpoint.bucket_temperatures)
        return cls(checkpoint.model, checkpoint.variables, checkpoint.layout, checkpoint.tokenizer,
                   checkpoint.specials, Calibration(temperatures), name=label)

    @classmethod
    def from_run(cls, directory: str, *, ema: bool | None = None, step: int | str | None = None,
                 mesh: "MeshSpec | None" = None, layout: "Placement | None" = None,
                 dtype: DTypeLike | None = None, param_dtype: DTypeLike | None = None) -> Self:
        """Load the `DecisionObjective` run in `directory` as a task.

        The task is rebuilt from the run's record over the latest checkpoint's
        weights (or `step`'s), and includes any calibration `save` wrote into the
        run. `ema` selects the averaged weights, and None uses them when the run kept
        them. `mesh` and `layout` place the weights as the trainer does, `dtype`
        overrides the compute dtype and `param_dtype` the storage dtype.
        """
        from dew.data.text import tokenizer_for
        from dew.inference.tasks import _saved_model, run_record
        from dew.registry import from_record, objectives

        record = run_record(directory, step)
        backbone = _saved_model(record, dtype).build()
        width, head_dtype = DecisionModel.head_size(backbone)
        model = DecisionModel(backbone, Head.from_record(records.json_value(record["head"], "head"),
                                                         width, dtype=head_dtype))
        laid = records.record(record["layout"], "layout")
        rows = from_record(_LAYOUTS[records.text(laid["name"], "layout.name")],
                           records.json_value(laid["fields"], "layout.fields"))
        tokenizer = record.get("tokenizer")
        if tokenizer is None:
            raise ValueError("the run records no tokenizer to lay its rows out with")
        variables = objectives[records.text(record["objective"], "objective")]._saved_variables(
            directory, step=step, ema=ema, mesh=mesh, layout=layout, param_dtype=param_dtype)
        checkpoints = Checkpoints(directory)
        chosen = checkpoints.resolve(step)
        chosen = checkpoints.latest if chosen is None else chosen
        if chosen is None:
            raise FileNotFoundError(f"{directory} holds no checkpoint")
        averaged = checkpoints.stored(chosen).get("ema") is not None if ema is None else ema
        task = cls(model, variables, rows, tokenizer_for(records.text(tokenizer, "tokenizer")),
                   from_record(Specials, records.json_value(record["specials"], "specials")),
                   name=str(epath.Path(directory).name), weights=Weights(chosen, averaged))
        saved = epath.Path(directory) / TASK_FILE
        if not saved.exists():
            return task
        settings = records.record(json.loads(saved.read_text()), TASK_FILE)
        fitted = from_record(Weights, records.json_value(settings["weights"], "weights"))
        if fitted != task.weights:
            raise ValueError(
                f"{TASK_FILE} was fitted on step {fitted.step}'s {'averaged' if fitted.ema else 'live'} "
                f"weights, and this loads step {chosen}'s {'averaged' if averaged else 'live'} ones; "
                "fit them again with `calibrated` and `save`, or load the step it was fitted on")
        return replace(task, calibration=from_record(Calibration, records.json_value(settings["calibration"],
                                                                                    "calibration")),
                       budget=from_record(Budget, records.json_value(settings["budget"], "budget")),
                       name=records.text(settings["name"], "name"))

    def save_pretrained(self, directory: str | Path) -> None:
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
                       dict(applied.buckets)).save(directory)

    def save(self, directory: str) -> None:
        """Write this task's calibration, budget and name into the run `directory`.

        It also records the checkpoint the weights came from, so `from_run` and
        `dew.pipeline` read the calibration back for that checkpoint. The file is
        written whole or not at all.
        """
        from dew.registry import to_record

        if self.weights is None:
            raise ValueError("this task's weights came from no run checkpoint, so a run has no "
                             "calibration of them to hold; save a task from `pipeline` or `from_run`")
        write_atomically(epath.Path(directory) / TASK_FILE, json.dumps({
            "weights": to_record(self.weights, Weights),
            "calibration": to_record(self.calibration, Calibration),
            "budget": to_record(self.budget, Budget), "name": self.name}, indent=1) + "\n")

    def __call__(self, state: JSON, questions: Mapping[str, Question], *,
                 images: Sequence[Image.Image] = ()) -> dict[str, Answer]:
        """Return one answer per question in `questions`, about `state` and any `images`.

        Images need a layout with a place for them (`Layout.image`) and the
        backbone's processor, as a Clef release brings.
        """
        return self._seen(state, questions, images)[0] if images else self.batch([(state, questions)])[0][0]

    def batch(self, requests: Sequence[tuple[JSON, Mapping[str, Question]]]
              ) -> list[tuple[dict[str, Answer], Usage]]:
        """Return each request's answers and usage.

        The rows of every request are scored together within the budget.
        """
        rows = [(index, encoded) for index, (state, questions) in enumerate(requests)
                for encoded in self._encoded(state, questions)]
        logits = self.logits([encoded for _, encoded in rows])
        answered: list[tuple[dict[str, Answer], Usage]] = []
        for index, (_, questions) in enumerate(requests):
            own = [(encoded, scores) for (row, encoded), scores in zip(rows, logits, strict=True)
                   if row == index]
            answers = {laid.name: self._answer(questions[laid.name], _ordered(laid, scores[slot]))
                       for encoded, scores in own for slot, laid in enumerate(encoded.questions)}
            usage = Usage(sum(len(encoded.tokens) for encoded, _ in own),
                          max((encoded.state_tokens for encoded, _ in own), default=0),
                          min((encoded.state_kept for encoded, _ in own), default=0),
                          any(encoded.trimmed for encoded, _ in own))
            answered.append(({name: answers[name] for name in questions}, usage))
        return answered

    def _seen(self, state: JSON, questions: Mapping[str, Question],
              images: Sequence[Image.Image]) -> tuple[dict[str, Answer], Usage]:
        """One request's answers and usage, its images laid out in its one row."""
        if self.processor is None or self.layout.image is None:
            raise ValueError("images need a backbone with a vision encoder, which this task does not have")
        media = Images.of(self.processor, images, self.layout.image)
        rows = self._encoded(state, questions, media.tokens)
        if len(rows) != 1:
            raise ValueError("a request's images go in a layout of one row")
        [row] = rows
        scores = np.asarray(self._forward(self.variables, DecisionInputs.collate(rows, self.specials.pad),
                                          media=media.inputs(self.processor, row)))[0]
        answers = {laid.name: self._answer(questions[laid.name], _ordered(laid, scores[slot]))
                   for slot, laid in enumerate(row.questions)}
        return ({name: answers[name] for name in questions},
                Usage(len(row.tokens), row.state_tokens, row.state_kept, row.trimmed))

    def _encoded(self, state: JSON, questions: Mapping[str, Question],
                 media: Sequence[int] = ()) -> list[Encoded]:
        rows = self.layout.rows(self.tokenizer, self.specials, state, questions, media=media)
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
            while (stop < len(order) and stop - start < self.budget.rows
                   and (_bucket(len(rows[order[stop]].tokens)) * _bucket(stop - start + 1, smallest=1)
                        <= self.budget.tokens)):
                stop += 1
            chosen = order[start:stop]
            scores = self._pass([rows[index] for index in chosen])
            found.update({index: scores[row, :len(rows[index].questions)]
                          for row, index in enumerate(chosen)})
            start = stop
        return [found[index] for index in range(len(rows))]

    def _pass(self, rows: Sequence[Encoded]) -> np.ndarray:
        """One forward pass at bucketed shapes, its rows placed on the
        weights' mesh (`RowPlan`)."""
        questions = max(len(row.questions) for row in rows)
        options = max(len(laid.options) for row in rows for laid in row.questions)
        inputs = DecisionInputs.collate(rows, self.specials.pad, questions=_bucket(questions, smallest=1),
                                        options=_bucket(options, smallest=1))
        length = _bucket(inputs.tokens.shape[1])
        inputs = replace(
            inputs, tokens=_widened(inputs.tokens, length, self.specials.pad),
            valid=_widened(inputs.valid, length),
            positions=None if inputs.positions is None else _widened(inputs.positions, length),
            slots=None if inputs.slots is None else _widened(inputs.slots, length))
        count = _bucket(len(rows), smallest=1)
        padded = RowPlan(None, len(rows), count, 0, 1).pad(inputs)
        plan = RowPlan.over(mesh_of(self.variables), count)
        return plan.host(self._forward(self.variables, plan.place(plan.pad(padded))))[:len(rows)]

    @functools.cached_property
    def _forward(self):
        return jax.jit(self.model.logits)

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
                   binning: bool = False, target_error: float | None = None, type_minimum: int = 10,
                   bucket_minimum: int = 2000) -> Self:
        """Return this task with temperatures fitted to held-out answers.

        If asked, it also fits a binning map and abstention thresholds that keep the
        error of the accepted answers at `target_error`, as `Calibration`'s parts fit
        them. `held_out` is labelled examples, or a dataset's validation reader
        (`DecisionObjective.dataset(...).val`), whose rows are already laid out. The
        minimum counts are Laya's: a type's temperature needs `type_minimum` answers
        and a bucket's `bucket_minimum`.
        """
        scored = self._scored_reader(held_out) if callable(held_out) else self._scored(held_out)
        if not scored:
            raise ValueError("calibration needs examples with answers")
        temperatures = Temperatures.fit(scored, type_minimum=type_minimum, bucket_minimum=bucket_minimum)
        binned = Binning.fit(scored, temperatures) if binning else None
        abstention = (None if target_error is None
                      else Abstention.fit(scored, temperatures, binned, target_error=target_error))
        return replace(self, calibration=Calibration(temperatures, binned, abstention))

    def score(self, examples: Iterable[Example | Mapping[str, object]],
              metrics: Sequence[Metric] = ()) -> dict[str, float]:
        """Return `metrics` over the questions answered for `examples`, through this task's calibration.

        The default metrics are Accuracy, ECE, AURC and the log loss, computed the
        way a validation pass computes them for a run.
        """
        from dew.decision.metrics import AURC, ECE, Accuracy
        from dew.decision.scoring import LogLoss

        defaults: list[Metric] = [Accuracy(), ECE(), AURC(), LogLoss()]
        chosen = list(metrics) or defaults
        scored = self._scored(examples)
        if not scored:
            raise ValueError("scoring needs examples with answers")
        width = max(len(held.logits) for held in scored)
        probabilities = np.zeros((len(scored), width))
        options = np.zeros((len(scored), width), bool)
        for row, held in enumerate(scored):
            probabilities[row, :len(held.logits)] = softmax(
                held.logits, self.calibration.temperatures.of(held.kind, len(held.logits)))
            options[row, :len(held.logits)] = True
        decisions = Decisions(probabilities=jnp.asarray(probabilities)[:, None],
                              options=jnp.asarray(options)[:, None],
                              labels=jnp.asarray([[held.label] for held in scored]),
                              ordinal=jnp.asarray([[held.kind == Score.kind] for held in scored]),
                              scored=jnp.ones((len(scored), 1), bool))
        return {metric.name: metric.finalize(metric(decisions, {})) for metric in chosen}

    def _scored(self, examples: Iterable[Example | Mapping[str, object]]) -> list[Scored]:
        scored = []
        for example in map(Example.of, examples):
            labels = example.labels()
            # A joint row reads every question, answered or not; a row per
            # question needs the answered ones alone.
            asked = (example.questions if self.layout.joint
                     else {name: example.questions[name] for name in labels})
            rows = self._encoded(example.state, asked)
            for encoded, logits in zip(rows, self.logits(rows), strict=True):
                scored.extend(Scored(asked[laid.name].kind, _ordered(laid, logits[slot]), labels[laid.name])
                              for slot, laid in enumerate(encoded.questions) if laid.name in labels)
        return scored

    def _scored_reader(self, reader: Reader) -> list[Scored]:
        scored = []
        for batch in reader(DataPartition()):
            inputs = laid_out(batch)
            logits = np.asarray(self._forward(self.variables, inputs))
            answered = np.asarray(batch["scored"])
            for row, slot in zip(*np.nonzero(answered), strict=True):
                count = int(np.sum(batch["options"][row, slot]))
                scored.append(Scored(KINDS[int(batch["kinds"][row, slot])].kind, logits[row, slot, :count],
                                     int(batch["labels"][row, slot])))
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

    def systemone(self, request: Mapping[str, object], *, details: bool = False,
                  strict: bool = False, whole: bool = False) -> dict[str, JSON]:
        """Answer a Jev request body with a Jev response body.

        The request holds `state` and `questions` in Jev's wire format, and
        optionally `model`, which this task ignores. The response holds `model` (this
        task's name), one answer per question, and `usage`. A Noul's answer is
        `noul`; a Choice's is `choice`, `probabilities` and `confidence`; a Score's is
        `score`, `legend`, `probabilities` and `confidence`. Numbers are rounded to
        four places. With `details`, a Noul also reports its confidence, an answer
        reports whether it abstained when the calibration gates, and `usage` reports
        how much of the state it kept.

        Beyond Jev's fields a request may carry `images`, as Clef's does: a list of
        images, each a PIL image, encoded image bytes, or a base64 string, bare or as
        a `data:` URL. `strict` answers as Jev's own endpoint does: the request's
        extension fields are dropped and the response holds Jev's fields alone.

        With `whole`, a request that the layout could answer only by cutting its
        state, instructions or options is refused instead, naming the maximum
        context length, as a benchmark that forbids truncation requires
        (Decision Index counts such a refusal as unsupported).
        """
        if strict and details:
            raise ValueError("strict answers hold Jev's fields alone, so they carry no details")
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
        images = [] if strict else decoded_images(request.get("images"))
        answers, usage = (self._seen(state, questions, images) if images
                          else self.batch([(state, questions)])[0])
        if whole and (usage.truncated or usage.trimmed):
            cut = "state" if usage.truncated else "instructions or options"
            raise ValueError(f"the request does not fit the maximum context length, max_len="
                             f"{self.layout.max_len} tokens, without cutting its {cut}")
        gated = self.calibration.abstention is not None
        reported: dict[str, JSON] = {"input_tokens": usage.input_tokens, "output_tokens": 0}
        if details:
            reported.update(state_tokens=usage.state_tokens, state_kept=usage.state_kept,
                            truncated=usage.truncated)
        return {"model": self.name,
                "answers": {name: _wire(questions[name], answer, details=details, gated=gated)
                            for name, answer in answers.items()},
                "usage": reported}


EXTENSIONS = ("images",)
"""The request fields `Decide.systemone` reads beyond Jev's, which `strict` drops."""


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
    """A question's logits in its own option order, from the slot order its row shows them in."""
    ordered = np.empty(len(laid.order), np.float32)
    ordered[list(laid.order)] = logits[:len(laid.order)]
    return ordered


def laid_out(batch: Mapping[str, object]) -> DecisionInputs:
    """The `DecisionInputs` of a batch the decision dataset laid out (`Encoding`)."""
    def array(name: str) -> jax.Array:
        return jnp.asarray(batch[name])

    return DecisionInputs(tokens=array("tokens"), valid=array("valid"), kinds=array("kinds"),
                          questions=array("questions"), spans=array("spans"),
                          option_spans=array("option_spans"), options=array("options"),
                          positions=array("positions") if "positions" in batch else None,
                          slots=array("slots") if "slots" in batch else None)


def _bucket(value: int, smallest: int = 64) -> int:
    return next((size for size in SHAPE_BUCKETS if size >= value and size >= smallest), value)


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



