"""`DecisionObjective`: training a decision model on labelled questions."""

import dataclasses
import functools
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew._records import integer, record
from dew.artifacts import Decisions
from dew.batch import FROZEN, Batch, Variables, dicts, part, thaw, unboxed
from dew.data.pipeline import Configured, ObjectiveData
from dew.data.plan import PLAN_VERSIONS, SHARED_STATE
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
from dew.decision.rows import (
    ConfigureLayout,
    Fit,
    PackedRows,
    QuestionRows,
    _rows,
    forked_documents,
    option_width,
    packing,
)
from dew.decision.scoring import LogLoss, ScoringRule
from dew.decision.shared_state import fork_depth, joint, shared_for
from dew.decision.task import Budget, Decide, DecideSettings, Weights, laid_out, trained_tree, whole_blocks_of
from dew.inference.projections import declared_groups
from dew.inference.tasks import Processor as TaskProcessor, RunProcessor
from dew.inputs import Field, InputSpec
from dew.interop.processors import Processor
from dew.lora import Adapter
from dew.nn import categorical
from dew.nn.cross_attention import ContextEncoder, crosses_alone
from dew.nn.flops import MissingFlopsCount, counted_flops, frozen_training_flops, parameter_flops
from dew.nn.inputs import ModelInputs
from dew.nn.protocols import (
    CountedDecisions,
    CountedLogits,
    CountedLogitsTraining,
    CountedTraining,
    CrossAttending,
    Lacks,
    ProjectionGroup,
    TokenModel,
)
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
    `Encoding` describes, a choice of fewer than `none_of_the_above_options`
    options drawing no "none of the above" (by default three: a yes or a no draws none, where it would
    change what the question asks; a run recorded before keeps its recorded floor,
    `dew.registry.Registered.defaults`), and `data` supplies their pipeline steps. The
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
    the layout's shortest row fits. A backbone that reads one document a row
    (it lacks packing, `dew.nn.protocols.Lacks`) trains and answers at
    `segments=1`, one row a window. The loss is the mean over the batch's
    scored questions. With `consistency` w > 0 it adds w times the mean over
    the batch's consistency groups (`Example.groups`) of the generalized
    Jensen-Shannon divergence of their members' answers over the group's
    canonical options (`categorical.group_divergence`), reported as the
    `consistency` metric; a group's members stand in one row, so a layout of a
    question a row refuses an example with groups under it. At 0 the groups
    are not read, and the loss is the label loss alone. With `input_lm` w > 0 it adds w
    times the mean next-token cross entropy of the rows' own tokens, the backbone reading each row
    again in its tokens' order, causally and alone, whatever its layout reads both ways or in streams
    (a backbone that attends both ways read as its causal twin, the same weights), through its own
    vocabulary head (`input_terms`), reported as the `input_ce` metric over the tokens
    it counts apart from the questions; the head's logits a tile at a time, as a language model's
    tiled head scores them (`chunked_tile`). The pass reads no context and no images, the row's
    tokens alone. At 0 there is no second pass.

    Evaluation returns the model's `Decisions` on the validation rows, which
    `Accuracy`, `ECE`, `AURC` and the scoring rules read, its probabilities
    divided by `temperatures` (none by default: a `Decide`'s calibration fits
    its logits before training moves them, so it is not inherited). The saved
    run loads as a `Decide` task.
    """

    saved_task = Decide
    settings = DecideSettings
    passes_progress = True
    _held_parts = True
    # The record is the backbone's; the head beside it is a task setting.
    _recorded_part = "backbone"

    counts_tokens = True

    def real_lengths(self, batch: Batch) -> np.ndarray:
        """Each packed row's own length, not its window's."""
        return _rows(batch, self.spans_in_context)[0]

    def scored_tokens(self, batch: Batch) -> np.ndarray:
        """Each window's packed rows' lengths summed (`real_lengths`, the default's count): the tokens a
        row of its holds, each a segment's above padding's 0, counted across the batch at once."""
        return np.count_nonzero(np.asarray(batch["segments"]) > 0, axis=-1).astype(np.int64)

    def row_tokens(self, batch: Batch) -> int | None:
        """Each window's width, read from its shape: the most its packed rows' lengths sum to, which a
        window's `scored_tokens` counts. A run with a token budget or cadence bounds its committed tokens
        by it without reading its token clock off the devices after every step, which stopped its steps
        running ahead of the devices (row 4740)."""
        return int(np.shape(batch["segments"])[-1])

    def model_flops(self, model, batch: Batch) -> float:
        """One backbone pass, its cross-attention over each row's context where rows carry one
        (`CrossAttending.cross_flops`), plus the head's real question/option geometry, and with
        `input_lm` the backbone's second pass over the rows (`input_terms`). Forming the vocabulary
        rows a head reads (an adapter's update merged into them) is execution, as recompute is."""
        model = self.model if model is None else model
        passes = 2 if self.input_lm else 1
        return (passes * counted_flops(model.backbone, self.real_lengths(batch))
                + self._context_flops(model, batch) + self._cross_flops(model, batch)
                + self._head_flops(model, batch) + self._input_head_flops(model, batch))

    def _input_head_flops(self, model, batch: Batch, *, training: bool = False) -> float:
        """The vocabulary head's work on the rows' own next tokens (`input_terms`), each row's tokens after
        its first; none without `input_lm`."""
        if not self.input_lm:
            return 0.0
        targets = np.maximum(self.real_lengths(batch) - 1, 0)
        if training:
            if not isinstance(model.backbone, CountedLogitsTraining):
                raise MissingFlopsCount(f"{type(model.backbone).__name__} has no logits_training_flops for "
                                        "input_lm's head")
            return model.backbone.logits_training_flops(targets)
        if not isinstance(model.backbone, CountedLogits):
            raise MissingFlopsCount(f"{type(model.backbone).__name__} has no logits_flops for input_lm's "
                                    "head")
        return model.backbone.logits_flops(targets)

    def _context_lengths(self, batch: Batch) -> np.ndarray | None:
        """Each window row's context length where the backbone reads the context in a pass of its own,
        as it reads a schema in context (`spans_in_context`) or encodes every context
        (`DecisionModel.context_encoder`); None where it reads none."""
        passes = self.spans_in_context or self.model.context_encoder == "backbone"
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

    def _cross_flops(self, model, batch: Batch, held: bool = False) -> float:
        """The backbone's cross-attention work over each packed row's own context (`_row_contexts`), or
        with `held` the part of it done by the backbone's own matrices (`held_cross_flops`): none where
        rows carry none."""
        if "context_lengths" not in batch:
            return 0.0
        backbone = model.backbone
        assert isinstance(backbone, CrossAttending)
        counted = backbone.held_cross_flops if held else backbone.cross_flops
        return counted(self.real_lengths(batch), self._row_contexts(batch))

    def _head_flops(self, model, batch: Batch) -> float:
        """Forward and training work read the same layout geometry and head capability."""
        tokens = batch.get("tokens")
        if isinstance(tokens, ModelInputs) and tokens.conditioning:
            raise MissingFlopsCount("DecisionObjective needs its image tower's multi-sequence geometry")
        # Spans in the context pool over its tokens, not the row's, where the head pools them.
        pools = self.spans_in_context and model.head.pools_spans
        if not isinstance(model.head, CountedDecisions):
            raise MissingFlopsCount(f"{type(model.head).__name__} has no declared decision_flops")
        pooled = self._context_lengths(batch) if pools else None
        return model.head.decision_flops(*_rows(batch, self.spans_in_context),
                                         **({} if pooled is None else {"pooled": pooled}))

    def training_flops(self, model, batch: Batch) -> float:
        """The backbone's frozen and trainable work, then three passes of the trained cross-attention
        blocks and decision head: an adapted backbone's own count (`CountedTraining`); a backbone whose
        held parameters train nothing but its cross-attention blocks, as `with_cross_attention` holds a
        source, its forward, input-gradient and no weight-gradient work (`frozen_training_flops`), so
        too a block's reads through its own matrices (`CrossBlocks.held_flops`); else, trained whole,
        three passes of everything."""
        model = self.model if model is None else model
        lengths = self.real_lengths(batch)
        context = self._context_lengths(batch)
        trained = 3 * (self._cross_flops(model, batch) + self._head_flops(model, batch))
        # The rows' second pass where `input_lm` reads them again (`input_terms`), trained as the first.
        passes = 2 if self.input_lm else 1
        if isinstance(model.backbone, CountedTraining):
            # An adapter trains through the context's pass too.
            own = 0.0 if context is None else model.backbone.training_flops(context)
            return (passes * model.backbone.training_flops(lengths) + own + trained
                    + self._input_head_flops(model, batch, training=True))
        if self._backbone_frozen():
            # A context's pass trains nothing, so it runs forward alone (`encode_context`); a block's
            # reads through the backbone's own matrices take no weight gradient (`held_flops`).
            forward = counted_flops(model.backbone, lengths)
            held = self._cross_flops(model, batch, held=True)
            # The vocabulary head is the backbone's, frozen with it: its input's gradient, no weight's.
            head = self._input_head_flops(model, batch)
            return (passes * frozen_training_flops(forward, parameter_flops(model.backbone, lengths))
                    + frozen_training_flops(head, head) + self._context_flops(model, batch) + trained - held)
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
                 consistency: float = 0.0, context_window: int | None = None,
                 none_of_the_above_options: int = 3, input_lm: float = 0.0):
        held: Variables | None = None
        widths: tuple[int | None, int] | None = None
        self.image_processor = None
        match backbone:
            case Decide():
                # A run trains the model's own tree, a loaded task's projections split back
                # (`trained_tree`, ledger 2299).
                model = backbone.model
                held = trained_tree(model, backbone.variables)
                self.image_processor = backbone.processor
                head = head or model.head
                context_encoder = context_encoder or model.context_encoder
                # The layout as the task was given it, which a plan before rows shared a state reads.
                layout = layout or backbone.declared
                tokenizer = tokenizer or backbone.tokenizer
                specials = specials or backbone.specials
                module = model.backbone
                # A task's passes pad its questions to its budget's options, kept until the run's data
                # binds its own; a pass's question slots are its windows' (`Decide.pass_shapes`).
                budget = backbone.budget
                if budget.options is not None:
                    widths = (None, budget.options)
            case Adapter() | Source():
                module, held = backbone.model, joined({"backbone": backbone.variables})
                if isinstance(backbone, Source):
                    tokenizer = tokenizer or tokenizer_of(backbone.text_processor)
            case _:
                module = backbone
        from dew.interop.pretrained import Pretrained

        # What the backbone's source drew fresh over the weights it holds, under the model's `backbone`:
        # none where other variables replace them.
        self._fresh_backbone = tuple(f"backbone/{module}" for module in backbone.fresh) if (
            variables is OMITTED and isinstance(backbone, Pretrained)) else ()
        if variables is not OMITTED:
            held = variables
        if tokenizer is None:
            raise ValueError("a decision model lays its rows out with a tokenizer; pass tokenizer=")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing is a share of the target, in [0, 1)")
        if not 0.0 <= none_of_the_above <= 1.0:
            raise ValueError("none_of_the_above is a probability")
        if none_of_the_above_options < 0:
            raise ValueError(f"none_of_the_above_options counts options, so it is nonnegative, got "
                             f"{none_of_the_above_options}")
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
        # The layout as stated, and as a request's questions train over one copy of its state where the
        # backbone and head read each question as its own row (`shared_for`), in a training step's form:
        # the two a run reads its rows by, by its data plan's version (`layout`).
        self.stated = layout
        self._shared = (shared_for(layout, module, head, train=True) if isinstance(layout, QuestionLayout)
                        else layout)
        # The data plan version its rows are read by: this Dew's until its data binds its own (`bind_plan`).
        self._plan = PLAN_VERSIONS[-1]
        self.tokenizer = tokenizer
        self.specials = specials or Specials.of(tokenizer)
        self.loss_rule = LogLoss() if loss is None else loss
        # A held tree enters plain: a draw's boxes (`nn.with_logical_partitioning`) are its table's.
        self.variables = dicts(unboxed(held))
        self.label_smoothing = label_smoothing
        if not consistency >= 0.0:
            raise ValueError(f"consistency weighs a divergence, so it is nonnegative, got {consistency}")
        self.consistency = consistency
        if not input_lm >= 0.0:
            raise ValueError(f"input_lm weighs a cross entropy, so it is nonnegative, got {input_lm}")
        self.input_lm = input_lm
        if input_lm:
            self._causal_backbone()
        self.shuffle_options = shuffle_options
        self.none_of_the_above = none_of_the_above
        self.none_of_the_above_options = none_of_the_above_options
        self.temperatures = temperatures
        self.segments = segments
        # The most questions a row and options a question of the run's data hold, which its run records
        # (`DecideSettings.questions`, `DecideSettings.options`), its saved task padding each pass's
        # questions to the options, as its data binds them (`bind_widths`).
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

    def fresh_modules(self) -> tuple[str, ...]:
        """What the backbone's source drew fresh (`Pretrained.fresh`), and the head where it is drawn
        beside a held backbone (`complete_variables`)."""
        held = self.variables or {}
        loaded = any("backbone" in collection for collection in held.values())
        drawn = loaded and not any("head" in collection for collection in held.values())
        return self._fresh_backbone + (("head",) if drawn else ())

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
    def spans_in_context(self) -> bool:
        """Whether a row's questions are its context (`Layout.spans_in_context`): a joint row's schema, or
        a question row's question."""
        return self.layout.spans_in_context

    @property
    def layout(self) -> Layout:
        """The layout its rows are laid out by: a request's questions over one copy of its state where the
        backbone and head read each question as its own row (`shared_for`), and the layout as it was
        stated under a data plan from before rows shared a state (`bind_plan`,
        `dew.data.plan.SHARED_STATE`), so everything the objective reads of its rows, their bounds, the
        widths it records and the task its steps declare, is the rows' its run trains on."""
        return self.stated if self._plan < SHARED_STATE else self._shared

    def bind_plan(self, version: int) -> None:
        """Read its rows as data plan `version` lays them out (`layout`): the version the run's data was
        compiled at (`ObjectiveData.at`), which a run saved under an earlier one resumes by."""
        self._plan = version

    def _bounds(self, layout: Layout) -> Bounds:
        """The most rows, questions and options a window of `layout`'s rows packs (`Layout.bounds`)."""
        return layout.bounds(self.tokenizer, self.specials, self.context_length, self.segments,
                             self.context_window)

    @functools.cached_property
    def _shared_bounds(self) -> Bounds:
        return self._bounds(self._shared)

    @functools.cached_property
    def _stated_bounds(self) -> Bounds:
        return self._shared_bounds if self.stated is self._shared else self._bounds(self.stated)

    @property
    def bounds(self) -> Bounds:
        """The most rows, questions and options a window packs (`Layout.bounds`), of the layout its rows
        are laid out by (`layout`),
        at most `segments` rows; rows that carry a context, an example's or their schema, pack as any
        do, each reading its own (`DecisionInputs.context_segments`)."""
        return self._stated_bounds if self._plan < SHARED_STATE else self._shared_bounds

    def loss(self, variables: Variables, batch: Batch, step: Step):
        inputs = laid_out(batch, self.bounds.segments, spans_in_context=self.spans_in_context,
                          whole_blocks=whole_blocks_of(self.layout),
                          attention=self.layout.attention, fork_depth=self._fork_depth,
                          joint_forks=self._joint_forks)
        logits, sown = self.model.observed_logits(
            variables, inputs, train=step.training, rngs=model_rngs(step.key, training=step.training),
            progress=self.progress(step), opened=step.sown())
        loss, aux = self._charged(logits, inputs, batch)
        loss, term = self._with_input_term(loss, variables, inputs, batch, step)
        if term is not None:
            aux = replace(aux, metrics={**aux.metrics, "input_ce": term.mean()[0]})
        qk, inspected = step.observed(sown)
        return loss, replace(aux, qk_stats=qk, inspected=inspected)

    def _with_input_term(self, loss: Ratio | Weighted, variables: Variables, inputs: DecisionInputs,
                         batch: Batch, step: Step) -> tuple[Ratio | Weighted, Ratio | None]:
        """`loss` with the rows' own next-token term beside it at `input_lm`'s weight, and the term;
        `loss` alone at 0. A validation's loss weighs it as training's does (`_validation_scores`)."""
        if not self.input_lm:
            return loss, None
        term = self.input_terms(variables, inputs, batch, step)
        terms, weights = (loss.terms, loss.weights) if isinstance(loss, Weighted) else ((loss,), (1.0,))
        return Weighted((*terms, term), (*weights, self.input_lm)), term

    def _causal_backbone(self) -> nn.Module:
        """The backbone the rows' own next-token term reads (`input_terms`): itself where it attends
        causally, else its causal twin, the same weights read each token after the ones before it alone;
        a backbone that attends both ways and states no `causal` setting to read it so is refused."""
        backbone = self.model.backbone
        if not isinstance(backbone, TokenModel) or backbone.causal:
            return backbone
        if "causal" not in {field.name for field in dataclasses.fields(backbone)}:
            raise Lacks("causal", f"input_lm reads the rows causally, and a {type(backbone).__name__} that "
                        "attends both ways states no causal setting to read them so")
        return backbone.clone(causal=True)

    def input_terms(self, variables: Variables, batch_inputs: DecisionInputs, batch: Batch,
                    step: Step) -> Ratio:
        """The next-token cross entropy of each row's own tokens (`input_lm`): the backbone reads every
        row of a window again, in its tokens' order, causally and apart from the window's other rows,
        their places counted along it, and each token after a row's first is scored as the next token of
        the one before it, through the backbone's vocabulary head a tile at a time; its `Ratio` over
        those tokens, apart from the questions'. The layout's other reads (blocks read both ways, option
        slots, streams) and the row's context are not this pass's."""
        from dew.batch import coordinates
        from dew.nn.progress import progressed
        from dew.objectives.lm.chunked import chunked_tile, final_read, head_cross_entropy

        backbone, held = self._causal_backbone(), thaw(part(variables, "backbone"))
        tokens, segments = batch_inputs.tokens, batch_inputs.segments
        # The tokens that precede another, as a language model reads its inputs: all but the last.
        laid = coordinates(segments=segments[:, :-1], max_segments=batch_inputs.max_segments,
                           right_padded=True)
        read = final_read(backbone, held, tokens[:, :-1], train=step.training, coordinates=laid,
                          rngs=model_rngs(step.key, training=step.training),
                          **progressed(backbone, self.progress(step)))
        # A token's successor in its own row: none after a row's last token or on the padding.
        scored = ((segments[:, 1:] == segments[:, :-1]) & (segments[:, :-1] > 0)).astype(jnp.float32)
        losses, _, _ = head_cross_entropy(backbone, held, read, tokens[:, 1:], 4, tile=chunked_tile(),
                                          predict=False)
        return self.row_mean(losses, batch, weights=scored)

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
        inputs = laid_out(batch, self.bounds.segments, spans_in_context=self.spans_in_context,
                          whole_blocks=whole_blocks_of(self.layout),
                          attention=self.layout.attention, fork_depth=self._fork_depth,
                          joint_forks=self._joint_forks)
        return self._decisions(self._logits(self.evaluation_variables(params, step), inputs), inputs, batch)

    def _validation_scores(self, params: Variables, batch: Batch, step: Step
                           ) -> tuple[Ratio | Weighted, Decisions]:
        inputs = laid_out(batch, self.bounds.segments, spans_in_context=self.spans_in_context,
                          whole_blocks=whole_blocks_of(self.layout),
                          attention=self.layout.attention, fork_depth=self._fork_depth,
                          joint_forks=self._joint_forks)
        logits = self.model.logits(params, inputs)
        loss, _ = self._with_input_term(self._charged(logits, inputs, batch)[0], params, inputs, batch, step)
        return loss, self._decisions(logits, inputs, batch)

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
        probabilities = categorical.probabilities(jnp.where(empty, 0.0, logits))
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

    @property
    def _fork_depth(self) -> int:
        """The longest chain of documents its batches hold, laid out as it states its layout or as it
        derives it (`data`, by the plan's version): the longer of the two (`fork_depth`), a chain past
        which a document reads nothing more."""
        return max(fork_depth(self._shared), fork_depth(self.stated))

    @property
    def _joint_forks(self) -> bool:
        """Whether its batches' rows read their documents together, as siblings do (`joint`), laid out as
        it states its layout or as it derives it (`_fork_depth`)."""
        return joint(self._shared) or joint(self.stated)

    def data(self, tokenizer: Tokenizer | None = None) -> ObjectiveData:
        """Fit rows before mixing, augment training layouts, and pack training
        and evaluation rows alike, by the objective's own tokenizer, which its
        `bounds` count in: a request's questions over one copy of its state where its layout shares
        (`shared_for`), and as its layout is stated where a run's data plan predates that
        (`ObjectiveData.earlier`, `dew.data.plan.SHARED_STATE`), so a run saved before resumes reading as
        it was written."""
        if tokenizer not in (None, self.tokenizer):
            raise ValueError("a decision objective lays its rows out with its own tokenizer; "
                             "pass tokenizer= when building it")
        shared = self._data(self._shared, self._shared_bounds)
        if self.stated is self._shared or not isinstance(self.stated, QuestionLayout):
            return shared
        return dataclasses.replace(shared, earlier=self._data(self.stated, self._stated_bounds))

    def _data(self, layout: Layout, bounds: Bounds) -> ObjectiveData:
        """`data` as `layout` lays rows out within `bounds`."""
        grouped = bool(self.consistency)
        training = ConfigureLayout(layout, self.tokenizer, self.specials, self.shuffle_options,
                                   self.none_of_the_above, train=True, context_length=self.context_length,
                                   grouped=grouped, none_of_the_above_options=self.none_of_the_above_options)
        evaluating = dataclasses.replace(training, train=False)
        fit = Fit(layout, self.tokenizer, self.specials, self.model.head, self.context_length, grouped,
                  bounds, self.none_of_the_above, self.context_window,
                  none_of_the_above_options=self.none_of_the_above_options)
        held = dataclasses.replace(fit, none_of_the_above=0.0, scored=True)
        # Rows a record spreads into: a layout of a question a row's, or grouped examples' apart.
        expand = (QuestionRows(),)
        options = bounds.options if layout.joint else None
        compact = None if options is None else -(-options // OPTION_TILE) * OPTION_TILE
        return ObjectiveData(
            train=(fit, *expand, Configured(training, needs=("slots",), scope="run", random=True)),
            evaluate=(held, *expand, Configured(evaluating, needs=("slots",), scope="run")),
            pack=packing(layout, bounds, self.context_window, grouped),
            packed=(PackedRows(layout.max_len, compact, self.spans_in_context,
                               documents=forked_documents(layout, bounds)),),
        )

    def bind_widths(self, widths: Mapping[str, object]) -> None:
        """Keep the most questions a row and options a question of the run's data hold, as its layout
        steps bound them (`widths`' `slots`, `ConfigureLayout`), which its saved task pads each pass to
        (`DecideSettings.questions`); data that binds none keeps those of the task it fine-tunes."""
        if "slots" not in widths:
            return
        slots = record(widths["slots"], "decision slots")
        self.widths = (self.layout.row_questions(integer(slots.get("questions"), "slots.questions")),
                       option_width(slots, self.none_of_the_above, self.none_of_the_above_options))

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

    def served_groups(self, variables: Variables) -> tuple[ProjectionGroup, ...]:
        """The backbone's projection groups a served task packs (`Decide._served`), their paths from
        the `params` root: what a checkpoint holds packed where its run trained a loaded task's packed
        tree (mains 947ac9262 up to ledger 2299's fix), which a resume splits back
        (`dew.checkpoints`). Carried to 1.0, when it is deleted (ledger 1552)."""
        backbone = part(thaw(variables), "backbone")
        return tuple(dataclasses.replace(group, path=("backbone", *group.path))
                     for group in declared_groups(self.model.backbone, backbone))

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
        which a task the objective fine-tunes brings. Its passes pad to the
        run's widths (`bind_widths`), as the task loaded from its run does, and
        under a data plan from before rows shared a state (`bind_plan`) it lays
        its rows out as the run stated them, as that task does (`Decide._recorded`).
        """
        if processor is not OMITTED:
            raise TypeError("a Decide task encodes with the objective's tokenizer and takes no processor")
        averaged = not (ema is False or (ema is None and state.ema is None))
        _, options = self.widths or (None, None)
        return Decide(
            self.model,
            self._pipeline_weights(state, ema),
            self.layout,
            self.tokenizer,
            self.specials,
            weights=Weights(int(state.step), averaged),
            processor=self.image_processor,
            budget=Budget(options=options),
            derives_sharing=self._plan >= SHARED_STATE,
        )
