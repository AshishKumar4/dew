"""`DecisionObjective`: training a decision model on labelled questions."""

import dataclasses
import functools
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew._records import integer, record
from dew.artifacts import Decisions
from dew.batch import FROZEN, Batch, Variables, dicts, part
from dew.data.pipeline import Configured, ObjectiveData
from dew.data.text import HFTokenizer, Tokenizer
from dew.decision.calibration import Temperatures
from dew.decision.head import DecisionHead, Head
from dew.decision.layout import (
    OPTION_TILE,
    Bounds,
    DecisionInputs,
    Layout,
    QuestionLayout,
    Specials,
    default_layout,
)
from dew.decision.model import DecisionModel
from dew.decision.questions import KINDS, Score
from dew.decision.rows import ConfigureLayout, Fit, PackedRows, QuestionRows, _rows, option_width, packing
from dew.decision.scoring import LogLoss, ScoringRule
from dew.decision.task import Decide, DecideSettings, Weights, laid_out
from dew.inference.loading import RunProcessor
from dew.inference.tasks import Processor as TaskProcessor
from dew.inputs import Field, InputSpec
from dew.interop.processors import Processor
from dew.lora import Adapter
from dew.nn import categorical
from dew.nn.cross_attention import ContextEncoder, crosses_alone
from dew.nn.flops import MissingFlopsCount, counted_flops, frozen_training_flops, parameter_flops
from dew.nn.inputs import ModelInputs
from dew.nn.protocols import CountedDecisions, CountedTraining, CrossAttending, TokenModel
from dew.objectives.base import (
    OMITTED,
    Aux,
    Objective,
    Omitted,
    ProgramModule,
    Ratio,
    Source,
    Step,
    Weighted,
    joined,
    model_rngs,
)

if TYPE_CHECKING:
    from dew.state import TrainState

_SCORE = KINDS.index(Score)


def tokenizer_of(processor: TaskProcessor | None) -> Tokenizer | None:
    """The tokenizer a loaded model's text processor encodes with: a run's
    own, or a checkpoint's Hugging Face tokenizer, alone or inside a
    multimodal processor, read from the directory it was loaded from, at the
    revision the model was. A processor without a tokenizer gives none, and
    `tokenizer=` names one."""
    from transformers import PreTrainedTokenizerBase, ProcessorMixin

    match processor:
        case RunProcessor():
            return processor.tokenizer
        case (
            Processor(reference=PreTrainedTokenizerBase() as tokenizer)
            | Processor(reference=ProcessorMixin(tokenizer=PreTrainedTokenizerBase() as tokenizer))
        ):
            return HFTokenizer(str(tokenizer.name_or_path), local_files_only=True)
        case _:
            return None


class DecisionObjective(Objective[Ratio | Weighted]):
    """Trains a decision model on questions with known answers.

    The model is any backbone with final states (`dew.nn.protocols.HiddenStates`),
    read by a head, Laya's `DecisionHead` unless `head` says otherwise.
    `backbone` is a model built from scratch, a loaded model
    (`Pretrained.load`, adapted with LoRA or partly frozen through `freeze`, as
    `LMObjective` accepts one), or a whole `Decide` task, such as Laya's released
    checkpoint, which brings its head and layout with it. Unless `layout` says
    otherwise, a head that reads several questions a row reads them in one
    row (`JointLayout`), and one that reads one question a row reads Laya's
    `MarkerLayout` over a bidirectional backbone and the `StateFirstLayout`
    over a causal one (`default_layout`). `tokenizer` and `specials` default
    to the loaded model's.

    The loss is `loss`, a proper scoring rule or a sum of them, computed against
    each row's right option, with `label_smoothing` spread evenly over its
    options. `shuffle_options` and `none_of_the_above` augment the rows as
    `Encoding` describes, and `data` supplies their pipeline steps. The
    training stream and the validation pass pack rows first-fit into windows
    of the layout's `max_len` tokens (`packing`), as many rows, questions and
    options as its `bounds` let a window hold, so a step pays for its tokens
    rather than `max_len` per row, and every batch has one shape. A window
    holds at most `segments` rows, which the backbone's sequence kernel then
    works on a window (`Coordinates.max_segments`) whether or not they come:
    16 by default, a few more than a window of mixed tasks' rows was measured
    to hold, where a window full of the shortest rows would declare tens. A
    mix of rows shorter than `max_len / segments` tokens fills its windows
    less under it and takes a larger one; None lets a window hold as many as
    the layout's shortest row fits. The loss is the mean over the batch's
    scored questions. With `consistency` w > 0 it adds w times the mean over
    the batch's consistency groups (`Example.groups`) of the generalized
    Jensen-Shannon divergence of their members' answers over the group's
    canonical options (`categorical.group_divergence`), reported as the
    `consistency` metric; a group's members stand in one row, so a layout of a
    question a row refuses an example with groups under it. At 0 the groups
    are not read, and the loss is the label loss alone.

    Evaluation returns the model's `Decisions` on the validation rows, which
    `Accuracy`, `ECE`, `AURC` and the scoring rules read, its probabilities
    divided by `temperatures` (none by default: a `Decide`'s calibration fits
    its logits before training moves them, so it is not inherited). The saved
    run loads as a `Decide` task.
    """

    saved_task = Decide
    settings = DecideSettings
    _held_parts = True
    # The record is the backbone's; the head beside it is a task setting.
    _recorded_part = "backbone"

    counts_tokens = True

    def real_lengths(self, batch: Batch) -> np.ndarray:
        """Each packed row's own length, not its window's."""
        return _rows(batch, self.schema_in_context)[0]

    def model_flops(self, model, batch: Batch) -> float:
        """One backbone pass, its cross-attention over each row's context where rows carry one
        (`CrossAttending.cross_flops`), plus the head's real question/option geometry. Forming the
        vocabulary rows a head reads (an adapter's update merged into them) is execution, as
        recompute is."""
        model = self.model if model is None else model
        return (counted_flops(model.backbone, self.real_lengths(batch)) + self._context_flops(model, batch)
                + self._cross_flops(model, batch) + self._head_flops(model, batch))

    def _context_lengths(self, batch: Batch) -> np.ndarray | None:
        """Each window row's context length where the backbone reads the context in a pass of its own,
        as it reads a schema in context (`schema_in_context`) or encodes every context
        (`DecisionModel.context_encoder`); None where it reads none."""
        passes = self.schema_in_context or self.model.context_encoder == "backbone"
        if not passes or "context_lengths" not in batch:
            return None
        return self._row_contexts(batch)

    def _row_contexts(self, batch: Batch) -> np.ndarray:
        """Each packed row's own context length, in the order `real_lengths` counts the rows: its tokens
        of the window's context (`context_segments`), or a window of one row's whole context."""
        if "context_segments" not in batch:
            return np.asarray(batch["context_lengths"])
        segments = np.asarray(batch["segments"])
        contexts = np.asarray(batch["context_segments"])
        return np.concatenate([np.bincount(owned, minlength=int(window.max()) + 1)[1:]
                               for window, owned in zip(segments, contexts, strict=True)])

    def _context_flops(self, model, batch: Batch) -> float:
        """The backbone's forward over each row's context, where it reads one in a pass of its own."""
        lengths = self._context_lengths(batch)
        return 0.0 if lengths is None else counted_flops(model.backbone, lengths)

    def _cross_flops(self, model, batch: Batch) -> float:
        """The backbone's cross-attention work over each packed row's own context (`_row_contexts`):
        none where rows carry none."""
        if "context_lengths" not in batch:
            return 0.0
        backbone = model.backbone
        assert isinstance(backbone, CrossAttending)
        return backbone.cross_flops(self.real_lengths(batch), self._row_contexts(batch))

    def _head_flops(self, model, batch: Batch) -> float:
        """Forward and training work read the same layout geometry and head capability."""
        tokens = batch.get("tokens")
        if isinstance(tokens, ModelInputs) and tokens.conditioning:
            raise MissingFlopsCount("DecisionObjective needs its image tower's multi-sequence geometry")
        if not isinstance(model.head, CountedDecisions):
            raise MissingFlopsCount(f"{type(model.head).__name__} has no declared decision_flops")
        # Spans in the context pool over its tokens, not the row's, where the head pools them.
        pooled = self._context_lengths(batch) if self.schema_in_context and model.head.pools_spans else None
        return model.head.decision_flops(*_rows(batch, self.schema_in_context),
                                         **({} if pooled is None else {"pooled": pooled}))

    def training_flops(self, model, batch: Batch) -> float:
        """The backbone's frozen and trainable work, then three passes of the trained cross-attention
        blocks and decision head: an adapted backbone's own count (`CountedTraining`); a backbone whose
        held parameters train nothing but its cross-attention blocks, as `with_cross_attention` holds a
        source, its forward, input-gradient and no weight-gradient work (`frozen_training_flops`);
        else, trained whole, three passes of everything."""
        model = self.model if model is None else model
        lengths = self.real_lengths(batch)
        context = self._context_lengths(batch)
        trained = 3 * (self._cross_flops(model, batch) + self._head_flops(model, batch))
        if isinstance(model.backbone, CountedTraining):
            # An adapter trains through the context's pass too.
            own = 0.0 if context is None else model.backbone.training_flops(context)
            return model.backbone.training_flops(lengths) + own + trained
        if self._backbone_frozen():
            # A context's pass trains nothing, so it runs forward alone (`encode_context`).
            forward = counted_flops(model.backbone, lengths)
            return (frozen_training_flops(forward, parameter_flops(model.backbone, lengths))
                    + self._context_flops(model, batch) + trained)
        return super().training_flops(model, batch)

    def _backbone_frozen(self) -> bool:
        """Whether the held tree trains none of the backbone's own parameters: every backbone
        parameter it trains is a cross-attention block's (`crossing`), the rest held frozen."""
        held = self.variables
        if held is None or FROZEN not in held or "backbone" not in held[FROZEN]:
            return False
        return crosses_alone(held.get("params", {}).get("backbone", {}))

    def __init__(self, backbone: nn.Module | Source | Adapter | Decide, *,
                 loss: ScoringRule | None = None, tokenizer: Tokenizer | None = None,
                 specials: Specials | None = None, layout: Layout | None = None,
                 head: Head | None = None, variables: Variables | None | Omitted = OMITTED,
                 label_smoothing: float = 0.0, shuffle_options: bool = True,
                 none_of_the_above: float = 0.0, temperatures: Temperatures | None = None,
                 segments: int | None = 16, context_encoder: ContextEncoder | None = None,
                 consistency: float = 0.0, context_window: int | None = None):
        held: Variables | None = None
        widths: tuple[int, int] | None = None
        self.image_processor = None
        match backbone:
            case Decide():
                model, held = backbone.model, backbone.variables
                self.image_processor = backbone.processor
                head = head or model.head
                context_encoder = context_encoder or model.context_encoder
                layout = layout or backbone.layout
                tokenizer = tokenizer or backbone.tokenizer
                specials = specials or backbone.specials
                module = model.backbone
                # A task's passes pad to its budget's widths, kept until the run's data binds its own.
                budget = backbone.budget
                if budget.questions is not None and budget.options is not None:
                    widths = (budget.questions, budget.options)
            case Adapter() | Source():
                module, held = backbone.model, joined({"backbone": backbone.variables})
                if isinstance(backbone, Source):
                    tokenizer = tokenizer or tokenizer_of(backbone.text_processor)
            case _:
                module = backbone
        if variables is not OMITTED:
            held = variables
        if tokenizer is None:
            raise ValueError("a decision model lays its rows out with a tokenizer; pass tokenizer=")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing is a share of the target, in [0, 1)")
        if not 0.0 <= none_of_the_above <= 1.0:
            raise ValueError("none_of_the_above is a probability")
        if segments is not None and segments < 1:
            raise ValueError(f"a window holds at least one row; got segments={segments}")
        if head is None:
            width, dtype = DecisionModel.head_size(module)
            head = DecisionHead(width, dtype=dtype)
        if isinstance(layout, QuestionLayout) and layout.labels is not None:
            # A head that reads labels and was drawn with none reads the layout's, by this tokenizer, the
            # one its rows are laid out by, which names them where the layout names none.
            head, labels = head.labelled(layout.labels, tokenizer)
            layout = dataclasses.replace(layout, labels=labels)
        self.model = DecisionModel(module, head, context_encoder or "table")
        if layout is None:
            if not isinstance(module, TokenModel):
                raise ValueError(
                    f"a {type(module).__name__} does not say whether it reads its tokens in "
                    "order; pass layout= (StateFirstLayout if it does, MarkerLayout if not)"
                )
            layout = default_layout(module.causal, head.one_question)
        self.layout = layout
        self.tokenizer = tokenizer
        self.specials = specials or Specials.of(tokenizer)
        self.loss_rule = LogLoss() if loss is None else loss
        self.variables = dicts(held)
        self.label_smoothing = label_smoothing
        if not consistency >= 0.0:
            raise ValueError(f"consistency weighs a divergence, so it is nonnegative, got {consistency}")
        self.consistency = consistency
        self.shuffle_options = shuffle_options
        self.none_of_the_above = none_of_the_above
        self.temperatures = temperatures
        self.segments = segments
        # The most questions a row and options a question of the run's data hold, which its saved task
        # pads each pass to (`DecideSettings.questions`), as its data binds them (`bind_widths`).
        self.widths = widths
        held_context = self.context_length
        if context_window is not None and (held_context is None or context_window < held_context):
            raise ValueError(f"context_window={context_window} holds a window's contexts, at least one row's "
                             f"context_length ({held_context}), which rows that read a context carry")
        # The most context tokens a packed window holds, its rows' contexts end to end (`packing`): as
        # given, or as many as its own tokens, `max_len`, or one row's `context_length` where that is
        # longer; none where its rows carry no context.
        self.context_window = (None if held_context is None
                               else context_window or max(self.layout.max_len, held_context))
        self.inputs = InputSpec(sample=Field("tokens", (self.layout.max_len,)))

    def program_key(self) -> tuple[ProgramModule, ...]:
        """The backbone, then the head, both trained."""
        return (
            ProgramModule(self.model.backbone, None, trained=True),
            ProgramModule(self.model.head, None, trained=True),
        )

    def substitute(self, modules: Sequence[nn.Module]) -> None:
        backbone, head = modules
        self.model = dataclasses.replace(self.model, backbone=backbone, head=head)

    def fresh_variables(self, key: jax.Array, held: Variables | None) -> Variables:
        """Nothing beyond the held parts: `complete_variables` draws the part the tree lacks."""
        return held or {}

    def complete_variables(self, key: jax.Array, tree: Variables) -> Variables:
        """Draw whichever of the backbone and the head `tree` does not hold."""
        backbone_key, head_key = jax.random.split(key)
        tokens = jnp.zeros((1, self.layout.max_len), jnp.int32)
        parts = {
            name: part(tree, name)
            for name in ("backbone", "head")
            if any(name in collection for collection in tree.values())
        }
        if "backbone" not in parts:
            parts["backbone"] = self.model.backbone.init(backbone_key, tokens)
        if "head" not in parts:
            parts["head"] = self.model.initialize_head(head_key, parts["backbone"], tokens)
        return joined(parts)

    @property
    def context_length(self) -> int | None:
        """The most context tokens a row carries (`DecisionModel.context_length`)."""
        return self.model.context_length

    @property
    def schema_in_context(self) -> bool:
        """Whether a row's questions are its context (`Layout.spans_in_context`): a joint row's schema, or
        a question row's question."""
        return self.layout.spans_in_context

    @functools.cached_property
    def bounds(self) -> Bounds:
        """The most rows, questions and options a window packs (`Layout.bounds`),
        at most `segments` rows; rows that carry a context, an example's or their schema, pack as any
        do, each reading its own (`DecisionInputs.context_segments`)."""
        return self.layout.bounds(self.tokenizer, self.specials, self.context_length, self.segments,
                                  self.context_window)

    def loss(self, variables: Variables, batch: Batch, step: Step):
        inputs = laid_out(batch, self.bounds.segments, spans_in_context=self.schema_in_context)
        logits = self.model.logits(
            variables, inputs, train=step.training, rngs=model_rngs(step.key, training=step.training)
        )
        return self._charged(logits, inputs, batch)

    def _charged(self, logits: jax.Array, inputs: DecisionInputs, batch: Batch
                 ) -> tuple[Ratio | Weighted, Aux]:
        """The loss and its accuracy report over the questions' option logits: where it adds a consistency
        term, the label loss and the groups' divergence, each its own `Ratio`, `consistency` weighing the
        divergence (`Weighted`)."""
        options, scored = inputs.options, jnp.asarray(batch["scored"]).astype(jnp.float32)
        # A padding question has no options; it is scored nothing over a
        # finite, even distribution, so no NaN reaches its gradient.
        empty = ~jnp.any(options, axis=-1, keepdims=True)
        logits, options = jnp.where(empty, 0.0, logits), options | empty
        target = categorical.smoothed(jnp.asarray(batch["targets"]), options, self.label_smoothing)
        charge = self.loss_rule.charge(logits, target, options, inputs.kinds == _SCORE)
        correct = (jnp.argmax(logits, axis=-1) == batch["labels"]).astype(jnp.float32)
        accuracy, _ = self.accuracy(correct, batch, scored).mean()
        loss = Ratio(self.row_mean(charge * scored, batch).total, self.row_mean(scored, batch).total)
        if not self.consistency:
            return loss, Aux(metrics={"accuracy": accuracy})
        # Each question's group in its window, and each option slot's canonical option: the group's
        # members' answers over the canonical options (`categorical.group_divergence`).
        divergence, members = categorical.group_divergence(logits, jnp.asarray(batch["groups"]),
                                                           jnp.asarray(batch["canonical"]), options)
        term = Ratio(self.row_mean(divergence, batch).total, self.row_mean(members, batch).total)
        return (Weighted((loss, term), (1.0, self.consistency)),
                Aux(metrics={"accuracy": accuracy, "consistency": term.mean()[0]}))

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> Decisions:
        inputs = laid_out(batch, self.bounds.segments, spans_in_context=self.schema_in_context)
        return self._decisions(self._logits(self.evaluation_variables(params, step), inputs), inputs, batch)

    def _validation_scores(self, params: Variables, batch: Batch, step: Step
                           ) -> tuple[Ratio | Weighted, Decisions]:
        inputs = laid_out(batch, self.bounds.segments, spans_in_context=self.schema_in_context)
        logits = self.model.logits(params, inputs)
        return self._charged(logits, inputs, batch)[0], self._decisions(logits, inputs, batch)

    def _decisions(self, logits: jax.Array, inputs: DecisionInputs, batch: Batch) -> Decisions:
        """The questions' option probabilities, each at its type's temperature, with what scores them."""
        if self.temperatures is not None:
            # Each question's temperature by its type and its count of real
            # options, as `Temperatures.of` gives it; a padding question has none.
            width = logits.shape[-1]
            table = np.asarray(
                [
                    [
                        1.0 if count == 0 else self.temperatures.of(kind.kind, count)
                        for count in range(width + 1)
                    ]
                    for kind in KINDS
                ],
                np.float32,
            )
            logits = logits / jnp.asarray(table)[inputs.kinds, jnp.sum(inputs.options, axis=-1)][..., None]
        empty = ~jnp.any(inputs.options, axis=-1, keepdims=True)
        probabilities = jax.nn.softmax(jnp.where(empty, 0.0, logits), axis=-1)
        return Decisions(
            probabilities=probabilities,
            options=inputs.options,
            labels=batch["labels"],
            targets=batch["targets"],
            ordinal=inputs.kinds == _SCORE,
            scored=batch["scored"],
            classes=batch.get("classes"),
        )

    @property
    def _logits(self):
        return self.program('logits', lambda variables, inputs: self.model.logits(variables, inputs))

    def data(self, tokenizer: Tokenizer | None = None) -> ObjectiveData:
        """Fit rows before mixing, augment training layouts, and pack training
        and evaluation rows alike, by the objective's own tokenizer, which its
        `bounds` count in."""
        if tokenizer not in (None, self.tokenizer):
            raise ValueError("a decision objective lays its rows out with its own tokenizer; "
                             "pass tokenizer= when building it")
        grouped = bool(self.consistency)
        training = ConfigureLayout(self.layout, self.tokenizer, self.specials, self.shuffle_options,
                                   self.none_of_the_above, train=True, context_length=self.context_length,
                                   grouped=grouped)
        evaluating = dataclasses.replace(training, train=False)
        fit = Fit(self.layout, self.tokenizer, self.specials, self.model.head, self.context_length, grouped,
                  self.bounds, self.none_of_the_above, self.context_window)
        held = dataclasses.replace(fit, none_of_the_above=0.0, scored=True)
        # Rows a record spreads into: a layout of a question a row's, or grouped examples' apart.
        expand = (QuestionRows(),)
        options = self.bounds.options if self.layout.joint else None
        compact = None if options is None else -(-options // OPTION_TILE) * OPTION_TILE
        return ObjectiveData(
            train=(fit, *expand, Configured(training, needs=("slots",), scope="run", random=True)),
            evaluate=(held, *expand, Configured(evaluating, needs=("slots",), scope="run")),
            pack=packing(self.layout, self.bounds, self.context_window, grouped),
            packed=(PackedRows(self.layout.max_len, compact, self.schema_in_context),),
        )

    def bind_widths(self, widths: Mapping[str, object]) -> None:
        """Keep the most questions a row and options a question of the run's data hold, as its layout
        steps bound them (`widths`' `slots`, `ConfigureLayout`), which its saved task pads each pass to
        (`DecideSettings.questions`); data that binds none keeps those of the task it fine-tunes."""
        if "slots" not in widths:
            return
        slots = record(widths["slots"], "decision slots")
        self.widths = (self.layout.row_questions(integer(slots.get("questions"), "slots.questions")),
                       option_width(slots, self.none_of_the_above))

    def task_settings(self) -> DecideSettings:
        """What a saved run rebuilds its task from beside the backbone.

        That is the head, layout, tokenizer, special tokens and the image
        processor's source, independently of the text tokenizer. Interop
        reloads that source's processor files and derives its media geometry
        from the source config; the decision record owns neither format.
        """
        from dew.inference.tasks import recorded_tokenizer

        return DecideSettings(
            self.model.head.record(),
            self.layout,
            self.specials, recorded_tokenizer(RunProcessor(self.tokenizer)),
            # A decision task reads text through its own tokenizer, so only a media processor is kept.
            None if self.image_processor is None or self.image_processor.directory is None
            else self.image_processor.source,
            self.model.context_encoder,
            *(self.widths or (None, None)))

    def recorded_model(self) -> nn.Module:
        """The backbone; the head is a task setting of its own."""
        return self.model.backbone

    def pipeline(
        self,
        state: "TrainState",
        *,
        ema: bool | None = None,
        processor: TaskProcessor | None | Omitted = OMITTED,
    ) -> Decide:
        """Return the trained model as a `Decide` task over the state's weights.

        A `Decide` encodes with the objective's own tokenizer and layout, so it
        takes no processor; it reads images with the backbone's own processor,
        which a task the objective fine-tunes brings.
        """
        if processor is not OMITTED:
            raise TypeError("a Decide task encodes with the objective's tokenizer and takes no processor")
        averaged = not (ema is False or (ema is None and state.ema is None))
        return Decide(
            self.model,
            self._pipeline_weights(state, ema),
            self.layout,
            self.tokenizer,
            self.specials,
            weights=Weights(int(state.step), averaged),
            processor=self.image_processor,
        )
