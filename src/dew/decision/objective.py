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

from dew._records import record
from dew.artifacts import Decisions
from dew.batch import FROZEN, Batch, Variables, dicts, part, thaw, unboxed
from dew.data.pipeline import Configured, ObjectiveData
from dew.data.text import HFTokenizer, Tokenizer
from dew.decision.calibration import Temperatures
from dew.decision.head import DecisionHead, Head
from dew.decision.inputs import OPTION_TILE, DecisionInputs
from dew.decision.layout import Bounds, Layout, QuestionLayout, Specials, default_layout
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
from dew.decision.shared_state import asked_over, fork_depth, joint, shared_for
from dew.decision.task import Budget, Decide, DecideSettings, Weights, laid_out, trained_tree, whole_blocks_of
from dew.inference.tasks import Processor as TaskProcessor, RunProcessor
from dew.inputs import Field, InputSpec
from dew.interop.processors import Processor
from dew.lora import Adapter
from dew.nn import categorical
from dew.nn.backbones.looping import exit_objective
from dew.nn.cross_attention import ContextEncoder, crosses_alone
from dew.nn.flops import MissingFlopsCount, counted_flops, frozen_training_flops, parameter_flops
from dew.nn.inputs import ModelInputs
from dew.nn.precision import at_least_fp32
from dew.nn.protocols import (
    CountedDecisions,
    CountedLogits,
    CountedLogitsTraining,
    CountedTraining,
    CrossAttending,
    Lacks,
    Looped,
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
from dew.objectives.passes import PassReports, PassWeighting, PassWeights, exiting, weighed

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

    `pass_weights` trains every pass count of a looped backbone at once
    (`dew.objectives.passes.PassWeights`): "mean", or one weight a pass
    summing to one, each pass's term the decision loss of its exit, its state
    run through the coda and the final norm to the head (no exit gate
    needed), so with no cut the loss is what each `with_passes(k)` model's
    would be, weighed (under a cut each term is the cut loop's exit,
    `PassWeights`), the consistency term each pass's as the label loss is and
    `input_lm` once. A validation that scores metrics reports each pass's and
    its loss under `pass_<k>/` beside the run's, which stay the last pass's. None, or weights
    all on the last pass, trains the last pass alone. A row whose context the
    backbone encodes in its own passes (`context_encoder="backbone"`, a schema
    in context) is refused under it: each pass would read the last pass's
    context, which no `with_passes(k)` model reads.

    `exit_entropy` β trains a looped backbone's exit gate as Ouro does (arXiv 2510.25741,
    `dew.nn.backbones.looping.exit_objective`), as `LMObjective`'s does: each question's label loss at
    each pass, its option logits read off that pass's states by the same head, weighted by the
    probability the question exits there, less β times that distribution's entropy. A question exits
    where the token its head reads its answer from does (`Head.answered`: its answer cue, or its row's
    last token for a head that reads the whole row); its hazards are the gate's there. The expectation
    and the entropy are reported as `exit_ce` and `exit_entropy`, the consistency term is the last
    pass's. It and `pass_weights` each weigh the passes, so a run states one. None, for a backbone
    without a gate, scores the last pass alone.

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
        lengths = self.real_lengths(batch)
        return (sum(counted_flops(backbone, lengths) for backbone in self._reads(model))
                + self._context_flops(model, batch) + self._cross_flops(self._exits(model), batch)
                + self._heads(model) * self._head_flops(model, batch) + self._input_head_flops(model, batch))

    def _exits(self, model: DecisionModel) -> DecisionModel:
        """`model` as the loss runs it: its backbone's every pass exiting where the loss weighs its passes
        (`pass_weights`, `exit_entropy`), else as it is."""
        if self.exit_entropy is None and (self.pass_weights is None
                                          or self.pass_weights.of_model(model.backbone) is None):
            return model
        return replace(model, backbone=exiting(model.backbone))

    def _reads(self, model: DecisionModel) -> tuple[nn.Module, ...]:
        """The backbone passes a step runs over the rows: the decision pass (`_exits`), and with `input_lm`
        the rows' own causal pass (`input_terms`), the backbone as it is."""
        return (self._exits(model).backbone, *((model.backbone,) if self.input_lm else ()))

    def _heads(self, model: DecisionModel) -> int:
        """How many passes' states the head reads: each pass's where the loss weighs them, else the last's."""
        exits = self._exits(model)
        return model.backbone.loop.steps if exits is not model else 1

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
        # The rows' passes (`_reads`): the decision pass, every exit where the loss weighs them, and the
        # second pass where `input_lm` reads them again (`input_terms`), trained as the first.
        reads, exits = self._reads(model), self._exits(model)
        trained = 3 * (self._cross_flops(exits, batch) + self._heads(model) * self._head_flops(model, batch))
        if isinstance(model.backbone, CountedTraining):
            # An adapter trains through the context's pass too.
            own = 0.0 if context is None else model.backbone.training_flops(context)
            return (sum(backbone.training_flops(lengths) for backbone in reads) + own + trained
                    + self._input_head_flops(model, batch, training=True))
        if self._backbone_frozen():
            # A context's pass trains nothing, so it runs forward alone (`encode_context`); a block's
            # reads through the backbone's own matrices take no weight gradient (`held_flops`).
            held = self._cross_flops(exits, batch, held=True)
            # The vocabulary head is the backbone's, frozen with it: its input's gradient, no weight's.
            head = self._input_head_flops(model, batch)
            return (sum(frozen_training_flops(counted_flops(backbone, lengths),
                                              parameter_flops(backbone, lengths)) for backbone in reads)
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
                 none_of_the_above_options: int = 3, input_lm: float = 0.0,
                 pass_weights: PassWeighting | None = None, exit_entropy: float | None = None):
        held: Variables | None = None
        options: int | None = None
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
                options = backbone.budget.options
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
        self.tokenizer = tokenizer
        self.specials = specials or Specials.of(tokenizer)
        self.loss_rule = LogLoss() if loss is None else loss
        # A held tree enters plain: a draw's boxes (`nn.with_logical_partitioning`) are its table's.
        self.variables = dicts(unboxed(held))
        self.label_smoothing = label_smoothing
        self._weigh_terms(consistency, input_lm)
        self.shuffle_options = shuffle_options
        self.none_of_the_above = none_of_the_above
        self.none_of_the_above_options = none_of_the_above_options
        self.temperatures = temperatures
        self.segments = segments
        # The most options a question of the run's data holds, which its run records
        # (`DecideSettings.options`), its saved task padding each pass's questions to them, as its data
        # binds them (`bind_widths`).
        self.options = options
        held_context = self.context_length
        if context_window is not None and (held_context is None or context_window < held_context):
            raise ValueError(f"context_window={context_window} holds a window's contexts, at least one row's "
                             f"context_length ({held_context}), which rows that read a context carry")
        # The most context tokens a packed window holds, its rows' contexts end to end (`packing`): as
        # given, or as many as its own tokens, `max_len`, or one row's `context_length` where that is
        # longer; none where its rows carry no context.
        self.context_window = (None if held_context is None
                               else context_window or max(self.layout.max_len, held_context))
        self.inputs = InputSpec(sample=Field.indexing("tokens", (self.layout.max_len,), self.model))
        self.pass_weights = self._passes_weighed(PassWeights.of(pass_weights))
        self.exit_entropy = self._exit_weighed(exit_entropy)

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
        backbone and head read each question as its own row (`shared_for`), the layout as it was stated
        otherwise."""
        return self._shared

    def _bounds(self, layout: Layout) -> Bounds:
        """The most rows, questions and options a window of `layout`'s rows packs (`Layout.bounds`)."""
        return layout.bounds(self.tokenizer, self.specials, self.context_length, self.segments,
                             self.context_window)

    @functools.cached_property
    def _shared_bounds(self) -> Bounds:
        return self._bounds(self._shared)

    @property
    def bounds(self) -> Bounds:
        """The most rows, questions and options a window packs (`Layout.bounds`), of the layout its rows
        are laid out by (`layout`),
        at most `segments` rows; rows that carry a context, an example's or their schema, pack as any
        do, each reading its own (`DecisionInputs.context_segments`)."""
        return self._shared_bounds

    def loss(self, variables: Variables, batch: Batch, step: Step):
        inputs = laid_out(batch, self.bounds.segments, spans_in_context=self.spans_in_context,
                          whole_blocks=whole_blocks_of(self.layout),
                          attention=self.layout.attention, fork_depth=self._fork_depth,
                          joint_forks=self._joint_forks)
        passes, gates, sown = self._pass_logits(
            variables, inputs, train=step.training, rngs=model_rngs(step.key, training=step.training),
            progress=self.progress(step), opened=step.sown())
        loss, aux = self._charged(passes, inputs, batch, gates)
        term = self.input_terms(variables, inputs, batch, step) if self.input_lm else None
        loss = self._with_input_term(loss, term)
        if term is not None:
            aux = replace(aux, metrics={**aux.metrics, "input_ce": term.mean()[0]})
        qk, inspected = step.observed(sown)
        return loss, replace(aux, qk_stats=qk, inspected=inspected)

    def _with_input_term(self, loss: Ratio | Weighted, term: Ratio | None) -> Ratio | Weighted:
        """`loss` with the rows' own next-token term (`input_terms`) beside it at `input_lm`'s weight;
        `loss` alone at 0, where there is no term. A validation's loss weighs it as training's does
        (`_validation_scores`)."""
        if term is None:
            return loss
        terms, weights = (loss.terms, loss.weights) if isinstance(loss, Weighted) else ((loss,), (1.0,))
        return Weighted((*terms, term), (*weights, self.input_lm))

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
        slots, streams) and the row's context are not this pass's.

        Rows asking a request's questions over one copy of its state (`DecisionInputs.documents`) score
        what each question's own row scores (`_forked_input_terms`)."""
        from dew.batch import coordinates, scored_targets
        from dew.nn.progress import progressed
        from dew.objectives.lm.chunked import chunked_tile, final_read, head_cross_entropy

        if batch_inputs.documents is not None:
            return self._forked_input_terms(variables, batch_inputs, batch, step)
        backbone, held = self._causal_backbone(), thaw(part(variables, "backbone"))
        tokens, segments = batch_inputs.tokens, batch_inputs.segments
        # The tokens that precede another, as a language model reads its inputs: all but the last.
        laid = coordinates(segments=segments[:, :-1], max_segments=batch_inputs.max_segments,
                           right_padded=True)
        read = final_read(backbone, held, tokens[:, :-1], train=step.training, coordinates=laid,
                          rngs=model_rngs(step.key, training=step.training),
                          **progressed(backbone, self.progress(step)))
        # A token's successor in its own row, as a language model scores its targets (`scored_targets`): none
        # after a row's last token or on the padding.
        scored = scored_targets(jnp, tokens, None, None, segments).astype(jnp.float32)
        losses, _, _ = head_cross_entropy(backbone, held, read, tokens[:, 1:], 4, tile=chunked_tile(),
                                          predict=False)
        return self.row_mean(losses, batch, weights=scored)

    def _forked_input_terms(self, variables: Variables, batch_inputs: DecisionInputs, batch: Batch,
                            step: Step) -> Ratio:
        """`input_terms` over rows whose questions share one copy of their state: the term each question's
        own row gives, summed. The backbone reads each window's documents causally, each question's
        continuing its row's shared part (`DecisionInputs.coordinates`), so a question's token reads the
        shared part and its own part before it, and no other question's; each token is scored from the
        state of the token before it in its own row (`Coordinates.predecessors`: a question's first token
        from the shared part's last), and a document's tokens count once for each question whose own row
        holds them (`asked_over`): the shared part's once a question it is asked, where one merged row
        scored them once and each question after its row's first from the question before it."""
        from dew.nn.progress import progressed
        from dew.objectives.lm.chunked import chunked_tile, final_read, head_cross_entropy

        backbone, held = self._causal_backbone(), thaw(part(variables, "backbone"))
        tokens, documents, parents = batch_inputs.tokens, batch_inputs.documents, batch_inputs.parents
        assert documents is not None and parents is not None, "a window's documents and what each continues"
        laid = batch_inputs.coordinates()
        read = final_read(backbone, held, tokens, train=step.training, coordinates=laid,
                          rngs=model_rngs(step.key, training=step.training),
                          **progressed(backbone, self.progress(step)))
        before = laid.predecessors()
        states = jnp.take_along_axis(read, jnp.maximum(before, 0)[:, :, None], axis=1)
        rows = jnp.take_along_axis(asked_over(documents, parents, batch_inputs.fork_depth),
                                   jnp.maximum(documents - 1, 0), axis=1)
        scored = jnp.where((before >= 0) & (documents > 0), rows, 0.0)
        losses, _, _ = head_cross_entropy(backbone, held, states, tokens, 4, tile=chunked_tile(),
                                          predict=False)
        return self.row_mean(losses, batch, weights=scored)

    def _pass_logits(self, variables: Variables, inputs: DecisionInputs, **read
                     ) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...], Variables]:
        """The option logits of each pass the loss weighs (`pass_weights`, `exit_entropy`), in order, its
        last the model's own, else the model's alone (`DecisionModel.observed_logits`); every pass's
        exit-gate logits where they are read (`DecisionModel.pass_logits`); and what the pass sowed."""
        if self._weights is None and self.exit_entropy is None:
            logits, sown = self.model.observed_logits(variables, inputs, **read)
            return (logits,), (), sown
        exits = replace(self.model, backbone=exiting(self.model.backbone))
        return exits.pass_logits(variables, inputs, **read)

    def _weigh_terms(self, consistency: float, input_lm: float) -> None:
        """Keep the loss's terms beside the label loss, each checked: the groups' divergence at `consistency`,
        the rows' own next-token term at `input_lm`, whose causal pass the backbone must read."""
        if not consistency >= 0.0:
            raise ValueError(f"consistency weighs a divergence, so it is nonnegative, got {consistency}")
        self.consistency = consistency
        if not input_lm >= 0.0:
            raise ValueError(f"input_lm weighs a cross entropy, so it is nonnegative, got {input_lm}")
        self.input_lm = input_lm
        if input_lm:
            self._causal_backbone()

    def _passes_weighed(self, weights: PassWeights | None) -> PassWeights | None:
        """`weights` checked against the backbone's loop (`PassWeights.of_model`) and its rows: a row whose
        context the backbone encodes in its own passes is refused, its every pass reading the last pass's
        context, which no `with_passes(k)` backbone reads."""
        if weights is None:
            return None
        weights.of_model(self.model.backbone)
        encoded = self.spans_in_context or self.model.context_encoder == "backbone"
        if self.context_length is not None and encoded:
            raise ValueError("pass_weights scores each pass's exit as the backbone run that many times "
                             "would, and a row whose context the backbone encodes in its own passes reads "
                             "the last pass's context at every pass")
        return weights

    def _exit_weighed(self, exit_entropy: float | None) -> float | None:
        """`exit_entropy` checked: a finite nonnegative weight over a backbone whose loop has an exit gate,
        and no `pass_weights` beside it, which weigh the same passes."""
        if exit_entropy is None:
            return None
        backbone = self.model.backbone
        loop = backbone.loop if isinstance(backbone, Looped) else None
        if loop is None or not loop.exit_gate or not 0 <= exit_entropy < float("inf"):
            raise ValueError(f"exit_entropy weighs the entropy of a looped backbone's exit distribution: "
                             f"a backbone whose loop has an exit gate, and a finite nonnegative weight, "
                             f"got {exit_entropy}")
        if self.pass_weights is not None:
            raise ValueError("pass_weights and exit_entropy each weigh a looped backbone's passes; state one")
        return exit_entropy

    @property
    def _weights(self) -> tuple[float, ...] | None:
        """Each pass's weight in the loss (`PassWeights.of_model`), None where it scores the last alone."""
        return None if self.pass_weights is None else self.pass_weights.of_model(self.model.backbone)

    def _charge(self, logits: jax.Array, inputs: DecisionInputs, batch: Batch) -> tuple[jax.Array, jax.Array]:
        """Each question's label loss `[B, Q]` over its option `logits`, and the logits as they score it: a
        padding question, which has no options, over a finite, even distribution, so no NaN reaches its
        gradient."""
        options = inputs.options
        empty = ~jnp.any(options, axis=-1, keepdims=True)
        logits, options = jnp.where(empty, 0.0, logits), options | empty
        # The gold distribution smoothed in the loss's precision, so a float64 run's sums are float64.
        target = categorical.smoothed(jnp.asarray(batch["targets"]).astype(at_least_fp32(logits.dtype)),
                                      options, self.label_smoothing)
        return self.loss_rule.charge(logits, target, options, inputs.kinds == _SCORE), logits

    def _charged(self, passes: tuple[jax.Array, ...], inputs: DecisionInputs, batch: Batch,
                 gates: tuple[jax.Array, ...] = ()) -> tuple[Ratio | Weighted, Aux]:
        """The loss and its accuracy report over the questions' option logits at each pass the loss weighs
        (`passes`, the last the model's own, at `pass_weights`): where it adds a consistency term, the
        label loss and the groups' divergence, each its own `Ratio`, `consistency` weighing the divergence
        (`Weighted`). Each pass's charge and divergence are weighed before their means, whose masses every
        pass shares; the accuracy is the last pass's. Where it trains a loop's exit gate (`exit_entropy`),
        Ouro's exit objective over every pass's charge and `gates`, each pass's gate logits, is the label
        loss."""
        # The questions' mask in the logits' precision, so a float64 run's masses sum in float64.
        scored = jnp.asarray(batch["scored"]).astype(at_least_fp32(passes[-1].dtype))
        charged = [self._charge(logits, inputs, batch) for logits in passes]
        charges, passes = [charge for charge, _ in charged], tuple(logits for _, logits in charged)
        logits = passes[-1]
        correct = (jnp.argmax(logits, axis=-1) == batch["labels"]).astype(jnp.float32)
        accuracy, _ = self.accuracy(correct, batch, scored).mean()
        metrics = {"accuracy": accuracy}
        if self.exit_entropy is None or len(passes) == 1:
            charge = self._weighed(charges)
        else:
            # Each question's hazards where its head reads its answer (`Head.answered`), in the row's tokens.
            at = self.model.head.answered(inputs)[..., None]
            hazards = [jnp.take_along_axis(gate, at, axis=1) for gate in gates]
            charge, expected, entropy = exit_objective(charges, hazards, self.exit_entropy, charges[-1].dtype)
            counted = self.row_mean(scored, batch).total
            metrics["exit_ce"] = Ratio(self.row_mean(expected * scored, batch).total, counted).mean()[0]
            metrics["exit_entropy"] = Ratio(self.row_mean(entropy * scored, batch).total, counted).mean()[0]
        loss = Ratio(self.row_mean(charge * scored, batch).total, self.row_mean(scored, batch).total)
        if not self.consistency:
            return loss, Aux(metrics=metrics)
        # Each question's group in its window, and each option slot's canonical option: the group's
        # members' answers over the canonical options (`categorical.group_divergence`).
        options = inputs.options | ~jnp.any(inputs.options, axis=-1, keepdims=True)
        groups, canonical = jnp.asarray(batch["groups"]), jnp.asarray(batch["canonical"])
        divided = [categorical.group_divergence(logits, groups, canonical, options) for logits in passes]
        divergence, members = self._weighed([divergence for divergence, _ in divided]), divided[-1][1]
        term = Ratio(self.row_mean(divergence, batch).total, self.row_mean(members, batch).total)
        return (Weighted((loss, term), (1.0, self.consistency)),
                Aux(metrics={**metrics, "consistency": term.mean()[0]}))

    def _weighed(self, terms: list[jax.Array]) -> jax.Array:
        """The passes' terms at their weights (`passes.weighed`), the last pass's where the loss weighs none
        (one pass, or `exit_entropy`, whose exit objective weighs the label loss's passes itself)."""
        return terms[-1] if len(terms) == 1 or self._weights is None else weighed(self._weights, terms)

    def evaluate(self, params: Variables, batch: Batch, step: Step
                 ) -> Decisions | tuple[Decisions, PassReports]:
        """The batch's decisions, and each pass's beside them where the loss weighs a looped backbone's
        passes (`_reported`), its loss not asked: a validation of metrics alone reports each pass's."""
        inputs = laid_out(batch, self.bounds.segments, spans_in_context=self.spans_in_context,
                          whole_blocks=whole_blocks_of(self.layout),
                          attention=self.layout.attention, fork_depth=self._fork_depth,
                          joint_forks=self._joint_forks)
        weights = self.evaluation_variables(params, step)
        if self._weights is None and self.exit_entropy is None:
            return self._decisions(self._logits(weights, inputs), inputs, batch)
        return self._reported(self._passes(weights, inputs), inputs, batch)

    def _reported(self, passes: tuple[jax.Array, ...], inputs: DecisionInputs, batch: Batch,
                  losses: Sequence[Ratio | Weighted | None] | None = None
                  ) -> Decisions | tuple[Decisions, PassReports]:
        """The last pass's decisions, and where there are several passes each pass's beside them
        (`PassReports`): its `losses`' term, none where the caller asked for no loss, and its decisions."""
        decisions = self._decisions(passes[-1], inputs, batch)
        if len(passes) == 1:
            return decisions
        losses = [None] * len(passes) if losses is None else losses
        return decisions, PassReports(tuple((loss, self._decisions(logits, inputs, batch))
                                            for loss, logits in zip(losses, passes, strict=True)))

    def _validation_scores(self, params: Variables, batch: Batch, step: Step
                           ) -> tuple[Ratio | Weighted, Decisions | tuple[Decisions, PassReports]]:
        """The validation loss and decisions, as training weighs its passes, with each pass's beside them
        where it weighs them (`PassReports`): its loss as the run's would be weighing it alone, and its
        decisions."""
        inputs = laid_out(batch, self.bounds.segments, spans_in_context=self.spans_in_context,
                          whole_blocks=whole_blocks_of(self.layout),
                          attention=self.layout.attention, fork_depth=self._fork_depth,
                          joint_forks=self._joint_forks)
        passes, gates, _ = self._pass_logits(params, inputs)
        term = self.input_terms(params, inputs, batch, step) if self.input_lm else None
        loss = self._with_input_term(self._charged(passes, inputs, batch, gates)[0], term)
        losses = [self._with_input_term(self._charged((logits,), inputs, batch)[0], term)
                  for logits in passes]
        return loss, self._reported(passes, inputs, batch, losses if len(passes) > 1 else None)

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
    def _passes(self):
        """Each scored pass's option logits (`_pass_logits`), one program."""
        return self.program('pass logits', lambda variables, inputs: self._pass_logits(variables, inputs)[0])

    @property
    def _fork_depth(self) -> int:
        """The longest chain of documents its batches hold (`fork_depth`), a chain past which a document
        reads nothing more."""
        return fork_depth(self._shared)

    @property
    def _joint_forks(self) -> bool:
        """Whether its batches' rows read their documents together, as siblings do (`joint`)."""
        return joint(self._shared)

    def data(self, tokenizer: Tokenizer | None = None) -> ObjectiveData:
        """Fit rows before mixing, augment training layouts, and pack training
        and evaluation rows alike, by the objective's own tokenizer, which its
        `bounds` count in: a request's questions over one copy of its state where its layout shares
        (`shared_for`)."""
        if tokenizer not in (None, self.tokenizer):
            raise ValueError("a decision objective lays its rows out with its own tokenizer; "
                             "pass tokenizer= when building it")
        return self._data(self._shared, self._shared_bounds)

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
        """Keep the most options a question of the run's data holds, as its layout steps bound them
        (`widths`' `slots`, `ConfigureLayout`), which its saved task pads each pass's questions to
        (`DecideSettings.options`); data that binds none keeps those of the task it fine-tunes."""
        if "slots" not in widths:
            return
        self.options = option_width(record(widths["slots"], "decision slots"), self.none_of_the_above,
                                    self.none_of_the_above_options)

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
            self.options)

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
        run's widths (`bind_widths`), as the task loaded from its run does.
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
            budget=Budget(options=self.options),
        )
