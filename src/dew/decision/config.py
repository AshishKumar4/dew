"""The decision run, as one typed record: a decision model fine-tuned on a
table of labelled text, as `laya-train --data tickets.csv` does.

By default the run starts from Laya's released English checkpoint, encoder
and head. `pretrained` names another: a Laya-style repository (one with an
`rl_agent_config.json`), or any Hugging Face model, whose head is then drawn
fresh, with `lora` to train an adapter in place of the whole model. A fresh
head is Laya's, reading the layout the backbone's order asks for, or with
`head` the joint schema head, reading every question of an example in one
row (`JointLayout`). Its data is a chain through `decisions` over a table, or `mixture`'s over a
mixture manifest whose sets contribute their recorded shares, and its validation the rows a
`holdout` keeps or a store's own split; validation scores them, and after training they fit the
temperatures the
task divides its logits by, saved into the run, so `Decide.from_run` and
`dew.pipeline` answer calibrated. A question type's temperature needs ten
held-out answers and a bucket's 2000, Laya's floors.
"""

from __future__ import annotations

import dataclasses
import os
from typing import TYPE_CHECKING, Annotated, ClassVar

import jax

from dew.config import ModelConfig, ObjectiveConfig, OptimConfig, Prepared, RunConfig
from dew.data.compile import fitted
from dew.data.dataset import record_argument
from dew.decision.head import DecisionHead, JointSchemaHead
from dew.decision.laya import LAYA
from dew.decision.layout import Framing, Layout, Schema, Segment, default_layout
from dew.decision.model import DecisionModel
from dew.decision.objective import DecisionObjective
from dew.decision.scoring import default_metrics
from dew.decision.task import Decide
from dew.nn.backbones.decoder_block import remat_policy
from dew.nn.cross_attention import ContextEncoder, CrossAttention
from dew.nn.protocols import RematChoosing
from dew.records import number, record
from dew.training.state import TrainState

if TYPE_CHECKING:
    from dew.interop import Pretrained


@dataclasses.dataclass(frozen=True)
class JointShape:
    """The sizes of the joint schema head (`JointSchemaHead`), its own defaults.

    Its input width is the backbone's, which the run reads from the model.
    """

    width: int = JointSchemaHead.width
    routing_layers: int = JointSchemaHead.routing_layers
    layers: int = JointSchemaHead.layers
    heads: int = JointSchemaHead.heads
    feedforward: int = JointSchemaHead.feedforward


@dataclasses.dataclass(frozen=True)
class DecisionRunConfig(RunConfig):
    """A run, plus the decision objective's own knobs."""

    RESUMABLE: ClassVar[tuple[str, ...]] = (*RunConfig.RESUMABLE, "calibrate")

    objective: ObjectiveConfig = dataclasses.field(default_factory=lambda: ObjectiveConfig("decision"))
    """The decision objective and its arguments: its scoring rule (`loss`),
    label smoothing, option shuffling and none-of-the-above rate."""
    optim: OptimConfig = dataclasses.field(default_factory=lambda: OptimConfig(learning_rate=2.5e-5))
    model: ModelConfig = dataclasses.field(
        default_factory=lambda: ModelConfig("causal_transformer", {"dtype": "bfloat16"}))
    pretrained: str = LAYA
    """A Laya-style repository, read whole, or a Hugging Face model, read as the backbone."""
    param_dtype: str = "float32"
    """The loaded checkpoint's parameter storage dtype; bfloat16 halves a frozen base's memory,
    leaving fresh LoRA factors and the head in float32."""
    subfolder: str | None = None
    """The checkpoint's folder inside a repository that bundles several, such as `multilingual`."""
    revision: str | None = None
    loop_steps: int | None = None
    """The passes a looped checkpoint runs its block (`CausalTransformer.loop`), such
    as Ouro's at 1, 2 or 4; None keeps the checkpoint's own."""
    loop_layers: tuple[int, int] | None = None
    """The block of layers [first, end) that `loop_steps` repeats: a pretrained
    decoder with no loop gains one over its own layers, as Qwen 3.5's middle block
    repeated twice; None keeps a looped checkpoint's block."""
    loop_backprop_steps: int | None = None
    """Train through the last k passes of the loop alone, with earlier passes
    cut from backpropagation; None keeps the loop's own choice."""
    loop_remat: bool = False
    """Checkpoint every pass of the loop, including the first, and recompute
    it in the backward pass; requires a loop and leaves decoding unchanged.
    False keeps a checkpoint's own choice."""
    remat: str | None = None
    """A `REMAT_POLICIES` name for the loaded backbone's layers, applied before
    looping and LoRA. The trainer's fitting ladder starts here; a policy
    outside `DECODER_REMAT` cannot climb. None keeps the checkpoint's policy."""
    head: JointShape | None = None
    """The joint schema head over a fresh backbone, which then reads the
    joint layout; None draws Laya's head."""
    calibrate: bool = True
    """Fit the task's temperatures on the held-out rows after training."""
    max_len: int | None = None
    """The row's length in tokens; None keeps the checkpoint's (Laya's 512) or the layout's own."""
    head_max_len: int | None = None
    """The tokens the question and its options share; None keeps the
    checkpoint's (Laya's 192). A question of many options wants more, or
    its options are cut to a few tokens each."""
    option_tokens: int | None = None
    """The most tokens one option keeps, in a layout of one row per question."""
    max_state_tokens: int | None = None
    """The most tokens a joint row's state keeps, whatever room `max_len` leaves
    it: a row is its state, cut to this, and its schema, so a large schema
    (a hundred options, or a document's many hypotheses) fits under a `max_len`
    set for it without lengthening every other row's state. None lets the
    state fill the row."""
    segments: Annotated[tuple[Segment, ...], record_argument(tuple[Segment, ...])] | None = None
    """A joint row's user turn, its parts in order (`JointLayout.segments`): the schema first, or
    read as the row's context through `context_layers` (`Schema(stream="context")`), its
    `context_length` holding the whole schema. None keeps the layout's own, the release's turn."""
    framing: Framing | None = None
    """How a joint row frames itself as a conversation (`Framing`): its system message, the
    assistant's prefill, the reasoning switch, a Jinja chat template in place of the tokenizer's
    own, and the template's other variables; recorded with the run's layout, so the run reloads it.
    None keeps the layout's own, the tokenizer's template with thinking off."""
    context_layers: tuple[int, ...] | None = None
    """The backbone layers that read each example's context (`Example.context`)
    through gated cross-attention, the context encoded as `context_encoder` says;
    the blocks train from zero gates beside the head, the loaded backbone frozen
    (`Pretrained.with_cross_attention`), with `lora` its adapter beside them. None
    reads no context."""
    context_length: int = 256
    """The most tokens of an example's context the cross-attention reads; an
    example with a longer context refuses the run where its data reads it,
    before training for data read into memory, as a request with one is
    refused. With `context_layers` alone."""
    context_encoder: ContextEncoder = "table"
    """How a context becomes the states its cross-attention reads
    (`DecisionModel.context_encoder`): the backbone's token table, or the
    backbone's own pass over it. With `context_layers` alone."""
    def prepare(self) -> Prepared:
        """The objective over the backbone `pretrained` names, on the table's
        or the mixture's rows, scored on the held-out ones, which calibrate the
        run after it trains. The checkpoint decides the backbone; the model
        flags choose only its compute dtype and attention kernel;
        `param_dtype` chooses the loaded checkpoint's storage dtype. `remat`
        chooses its layer recomputation before looping or attaching LoRA."""
        from dew.interop import Pretrained

        decided = sorted(set(self.model.fields) - {"dtype", "attention_impl"})
        if decided:
            raise ValueError(f"--model sets {decided}, which {self.pretrained} decides; only dtype and "
                             "attention_impl are choices")
        dtype = str(self.model.fields.get("dtype") or "float32")
        attention_impl = str(self.model.fields.get("attention_impl", "auto"))
        budgets = {"max_len": self.max_len, "head_max_len": self.head_max_len,
                   "option_tokens": self.option_tokens, "max_state_tokens": self.max_state_tokens,
                   "framing": self.framing, "segments": self.segments}
        in_context = any(isinstance(segment, Schema) and segment.stream == "context"
                         for segment in self.segments or ())
        if in_context and self.context_layers is None:
            raise ValueError("a schema read as the row's context is read by context layers; set "
                             "--context-layers")
        if self.context_encoder != "table" and self.context_layers is None:
            raise ValueError("--context-encoder says how context layers read a context; set --context-layers")
        derived = {}
        if self.context_layers is not None:
            derived["context_encoder"] = self.context_encoder
        from dew.decision.sources import PretrainedDecision

        frozen = self.lora is not None or self.context_layers is not None
        start = Pretrained.load(self.pretrained, subfolder=self.subfolder, revision=self.revision,
                                dtype=dtype, param_dtype=self.param_dtype, attention_impl=attention_impl,
                                **self.trainer.source_placement(frozen=frozen))
        if isinstance(start, PretrainedDecision):
            if any(choice is not None for choice in (self.lora, self.head, self.loop_steps,
                                                    self.loop_layers, self.loop_backprop_steps,
                                                    self.remat, self.context_layers)) or self.loop_remat:
                raise ValueError("--lora, --head, --remat, --context-layers and the loop shape a fresh "
                                 "backbone; a decision release trains whole under its own head")
            start = Decide.of(start)
            derived["layout"] = _budgeted(start.layout, budgets)
        else:
            if self.remat is not None:
                if not isinstance(start.model, RematChoosing):
                    raise ValueError(f"--remat chooses a decoder's layer policy (RematChoosing), which "
                                     f"{type(start.model).__name__} does not support")
                start = dataclasses.replace(start, model=start.model.with_remat(remat_policy(self.remat)))
            if (self.loop_steps is not None or self.loop_layers is not None
                    or self.loop_backprop_steps is not None or self.loop_remat):
                start = looping(start, self.loop_steps, self.loop_layers,
                                backprop_steps=self.loop_backprop_steps, remat=self.loop_remat or None)
            if self.context_layers is not None:
                # Drawn from a key of their own, apart from the adapter's and the head's.
                blocks = CrossAttention(self.context_layers, self.context_length)
                start = start.with_cross_attention(blocks, key=jax.random.fold_in(
                    jax.random.key(self.trainer.key), 1))
            start = start if self.lora is None else start.adapt(self.lora, key=self.trainer.key)
            if self.head is not None:
                width, head_dtype = DecisionModel.head_size(start.model)
                derived["head"] = JointSchemaHead(hidden_size=width, dtype=head_dtype,
                                                  **dataclasses.asdict(self.head))
            head = derived.get("head", DecisionHead)
            derived["layout"] = _fresh_layout(start.model.causal, head.one_question, budgets)
        objective = self.objective.build(backbone=start, **derived)
        assert isinstance(objective, DecisionObjective), "a decision run's objective is a DecisionObjective"
        summary: dict[str, object] = {"pretrained": self.pretrained}
        # A mixture's weights and the manifest it was made by, as the chain states them (`Dataset.facts`).
        facts = self.chain().facts()
        if "mixture_weights" in facts:
            weights = record(facts["mixture_weights"], "mixture_weights")
            summary["mixture"] = {name: value for name, value in weights.items()
                                  if number(value, f"mixture_weights.{name}") > 0}
            summary["made"] = record(facts.get("mixture_made", {}), "mixture_made")
        # Compiled once: `train` reads the run's length and cadences off this loader and trains on it.
        dataset, held = fitted(self.chain(), self.validation, objective=objective.data(),
                                seed=self.data_seed)
        summary["records"] = dataset.records
        unfit = {name: counter.rejected.get("unfit", 0) for name, counter in dataset.counts.items()}
        unfit["held out"] = sum(counter.rejected.get("unfit",
            0) for counter in dataset.validation_counts.values())
        if any(unfit.values()):
            summary["unfit"] = unfit

        def calibrate(state: TrainState, directory: str) -> None:
            """Fit the task's temperatures on the held-out rows; process zero saves them into the run."""
            task = objective.pipeline(state)
            assert isinstance(task, Decide), "a decision objective's task is a Decide"
            assert held
            # The first held-out split the run scores is the one it calibrates on.
            calibrated = dataclasses.replace(task.calibrated(next(iter(held.values()))),
                                             name=os.path.basename(directory))
            if jax.process_index() == 0:
                calibrated.save(directory)

        # The task's own default metrics, asked for where its data holds held-out rows to score.
        metrics = tuple(default_metrics()) if held else ()
        after = calibrate if self.calibrate and held else None
        return Prepared(self, lambda name: self.train(objective, dataset, name=name, metrics=metrics,
                                                      summary=summary, after=after, validation=held))


def looping(start: Pretrained, steps: int | None, layers: tuple[int, int] | None, *,
            backprop_steps: int | None = None, remat: bool | None = None) -> Pretrained:
    """A loaded checkpoint whose block runs `steps` passes: a looped one's own block, or `layers`,
    which gives a decoder with no loop one over its own layers. Refused where neither names a block,
    or a new block's passes are not stated. `backprop_steps` and `remat` choose
    the loop's backward memory; None keeps a looped checkpoint's own value."""
    from dew.nn.protocols import Looping

    looped = (start.model.with_passes(steps, layers, backprop_steps=backprop_steps, remat=remat)
              if isinstance(start.model, Looping) else None)
    if looped is None:
        raise ValueError(f"this {type(start.model).__name__} has no loop: --loop-layers names the block of "
                         "its layers to repeat and --loop-steps how many times")
    return dataclasses.replace(start, model=looped)


type Budgets = dict[str, int | Framing | tuple[Segment, ...] | None]
"""A run's budgets of a row's layout, by the layout's field names (`_budgeted`)."""


def _fresh_layout(causal: bool, one_question: bool, budgets: Budgets) -> Layout:
    """The layout a fresh head reads (`default_layout`), with the run's `budgets`."""
    return _budgeted(default_layout(causal, one_question), budgets)


def _budgeted(layout: Layout, budgets: Budgets) -> Layout:
    """`layout` with the `budgets` the run states, each but those left None, which keep the
    layout's own; zero is a budget too (a joint row of no state). A budget the layout has no
    field for is refused: a joint row has no question budget, and a row per question no state
    budget, its state taking what is left."""
    stated = {name: value for name, value in budgets.items() if value is not None}
    fields = {field.name for field in dataclasses.fields(layout)}
    unknown = sorted(f"--{name.replace('_', '-')}" for name in stated if name not in fields)
    if unknown:
        raise ValueError(f"{', '.join(unknown)} budget a row that {type(layout).__name__} does not lay out")
    return dataclasses.replace(layout, **stated)


__all__ = ["DecisionRunConfig", "JointShape"]
