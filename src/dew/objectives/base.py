"""What the trainer is optimizing.

The trainer handles the mechanics of training: the mesh, the compiled step,
the EMA copy, checkpoints and logging. An `Objective` defines what is learned:
it initialises the parameter tree, computes the loss from a batch and decides
what evaluation produces. To study a different question you write a different
objective, and the trainer stays the same.

The trainer gives an objective the step count and a random key in a `Step`.
The objective returns additive loss statistics, optionally with an `Aux` of
reports. These values are JAX pytrees.
"""

from __future__ import annotations

import enum
import functools
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    NamedTuple,
    Protocol,
    Self,
    runtime_checkable,
)

import jax
import jax.numpy as jnp
import optax
from flax import struct
from jax.tree_util import Partial
from typing_extensions import TypeIs, TypeVar

from dew.artifacts import Artifact, Artifacts, TokenScores
from dew.records import JSON

if TYPE_CHECKING:
    from flax import linen as nn
    from jax.typing import DTypeLike

    from dew.inference.tasks import BlockGeneration, MaskedGeneration, Processor, TextGeneration
    from dew.inputs import InputSpec
    from dew.nn.backbones.decoder_stack import DecoderBank
    from dew.sampling.pipelines import TextToImage
    from dew.training.distributed import Layout, MeshSpec
    from dew.training.state import TrainState

    type Task = TextGeneration | BlockGeneration | MaskedGeneration | TextToImage

type Variables = Mapping[str, Any]
"""A Flax variables dict: the `params` collection plus any other collection
the modules keep (`moe`, `batch_stats`, an objective's frozen encoders)."""

type Batch = Mapping[str, Any]
"""One batch of training examples, as the trainer and every objective read it.

It is declared here and imported by `dew.data.dataset`, so a field added on
one side has the same type on the other. The values are not all arrays.
Besides the arrays a step consumes, a batch can hold the prepared text inputs
of a language run, the file paths a video corpus loads lazily, the integer
row counts a packer keeps and the record lists a mixture reads. A narrower
value type would be wrong about what `dew.data` already builds."""
type Path = tuple[str, ...]
type PathFilter = Callable[[Path], bool]
"""A function that selects leaves of a variables tree by the tuple of dict keys above them.

One filter type is used for the EMA selection, for `optax.multi_transform`
labels and for frozen subtrees."""
type Initializer = Partial
"""An objective's `init` with its held variables bound as `Partial` children,
which a JIT takes as arguments (`Objective.initializer` says why)."""

@struct.dataclass
class Ratio:
    """A loss numerator and its denominator, kept apart until the loss is reduced.

    Both parts sum across microbatches and devices, and only then does `mean`
    divide one by the other.

    The denominator (`mass`) is nonnegative and does not depend on the
    parameters. Zero mass means a zero numerator and no contribution.
    """
    total: jax.Array
    mass: jax.Array

    def mean(self) -> tuple[jax.Array, jax.Array]:
        """Return `total` over `mass`, or zero where the mass is zero, and whether there is any mass.

        The mass is treated as a constant, so no gradient flows through it.
        """
        mass = jax.lax.stop_gradient(self.mass)
        active = mass > 0
        dtype = jnp.result_type(self.total.dtype, mass.dtype, jnp.float32)
        value = self.total.astype(dtype) / jnp.where(active, mass.astype(dtype), 1)
        return jnp.where(active, value, 0), active


Loss = TypeVar("Loss", default=Ratio | jax.Array | float)
Effects = TypeVar("Effects", default=None)


@struct.dataclass
class Step:
    """What the trainer tells an objective about the current step.

    `step` is the count of accepted microbatches, which is what an
    objective's own schedules index by. `key` is new for every attempt at a
    microbatch, and a replayed microbatch gets the key of the attempt that
    first read it, so it draws the same randomness.
    """
    step: jax.Array
    key: jax.Array
    ema: Variables | None
    """The variables tree with the averaged leaves in place of the live ones,
    or None when the objective keeps no EMA."""


@struct.dataclass
class Aux(Generic[Effects]):
    """Everything a loss returns besides its statistics.

    `metrics` go to the tracker. `variables` replace whole nonparameter
    collections. `effects` are additive observations that the optimizer
    applies once per committed update, not once per microbatch.
    """
    metrics: dict[str, jax.Array]
    variables: Variables | None = None
    """Whole nonparameter collections from one accepted microbatch, such as BatchNorm statistics.

    Each one replaces the collection of the same name. The optimizer updates
    the `params` collection, so it cannot appear here. Deferred router-bias
    updates go in `effects`."""
    qk_stats: Variables | None = None
    """The `qk` collection the attention layers sowed, which the optimizer's QK-Clip reads.

    It is nested by module path. Each attention layer holds `max_logits`, a
    one-tuple of an fp32 `[rows, heads]` array of per-head logit maxima, and
    an MLA layer also holds `qk_nope`, an int scalar giving its nope width.
    Rows are the batch's rows, with microbatches concatenated under a
    pipeline. It is None when the loss never opened the collection, and then
    the clip does nothing."""
    effects: Effects | None = None
    """Additive observations that `Objective.apply_effects` applies once per optimizer commit."""


def _has_aux(loss: Loss | tuple[Loss, Aux[Effects]]) -> TypeIs[tuple[Loss, Aux[Effects]]]:
    return isinstance(loss, tuple) and len(loss) == 2 and isinstance(loss[1], Aux)


@dataclass(frozen=True)
class Shown:
    """How the training display shows one reported metric.

    `better` says which way the metric improves, "higher" or "lower", and the
    display colours its change as progress or regress by it. `percent` shows a
    fraction as a percentage. `group` puts the metric under a heading other
    than its name's prefix. A metric with no `Shown` is shown under its name,
    without colour.
    """
    better: Literal["higher", "lower"] | None = None
    percent: bool = False
    group: str | None = None


@dataclass(frozen=True)
class TrainingScalar[Statistics, Additions]:
    """A training metric an objective reports, chosen for `Best` to rank by or `Plateau` to stop on."""
    owner: Objective[Statistics, Additions]
    name: str
    shown: Shown


class TrainingScalars[Statistics, Additions]:
    """The training metrics an objective declares, as typed attributes an editor can complete.

    An objective whose metric names are known only at run time can be read by
    item lookup instead. Reading an undeclared name fails at once, before a
    fit starts, and completion lists only the objective's own names.
    """
    loss: TrainingScalar[Statistics, Additions]

    def __init__(self, owner: Objective[Statistics, Additions]):
        self._owner = owner

    def __getattr__(self, name: str) -> TrainingScalar[Statistics, Additions]:
        try:
            return self._owner._scalar(name)
        except ValueError as missing:
            raise AttributeError(str(missing)) from missing

    def __getitem__(self, name: str) -> TrainingScalar[Statistics, Additions]:
        return self._owner._scalar(name)

    def __dir__(self) -> list[str]:
        return sorted(set(object.__dir__(self)) | {'loss'} | set(self._owner.shown))


@struct.dataclass
class Prediction:
    """What a token objective scored a batch with, for comparison with a teacher.

    - `logits`: the `[B, S, vocab]` fp32 scores from the model's forward pass.
    - `losses`: the `[B, S]` per-position loss the objective sums.
    - `weights`: the `[B, S]` weight the objective gives each position.
    - `hidden`: the `[B, S, D]` states of the layers the caller asked for,
      in the order asked.
    """
    logits: jax.Array
    losses: jax.Array
    weights: jax.Array
    hidden: tuple[jax.Array, ...]


def everything(path: Path) -> bool:
    return True


def under(*prefix: str) -> PathFilter:
    """Return a filter that accepts the leaves below `prefix`.

    For example, `under("params", "context_encoder")`.
    """
    return lambda path: path[:len(prefix)] == prefix


def select(tree: Variables, keep: PathFilter) -> Variables:
    """Return the subtree of `tree` whose leaves `keep` accepts, nested the same way.

    A branch with no accepted leaf is dropped, so the result is what the EMA
    stores and what `merge` puts back. Raises `ValueError` when `keep`
    accepts no leaf at all.
    """
    def prune(node, path):
        if isinstance(node, Mapping):
            kept = {name: prune(child, (*path, name)) for name, child in node.items()}
            return {name: child for name, child in kept.items() if child is not None} or None
        return node if keep(path) else None

    selected = prune(tree, ())
    if not selected:
        raise ValueError("the filter selected no leaf of the variables tree")
    return selected


def merge(tree: Variables, overlay: Variables) -> Variables:
    """Return a copy of `tree` with every leaf in `overlay` written over it."""
    merged = dict(tree)
    for name, child in overlay.items():
        held = tree.get(name)
        merged[name] = (merge(held, child)
                        if isinstance(held, Mapping) and isinstance(child, Mapping)
                        else child)
    return merged


FROZEN = "frozen"
"""The name of the collection where a partially trained run keeps its frozen weights.

The optimizer updates the `params` collection and nothing else, so the
leaves that `freeze` keeps in `params` are the ones that train. The frozen
leaves stay in this collection as part of the state, and `thaw` merges the
two before the model reads them."""


VALID_ROWS = "valid_rows"
"""The field every evaluation batch holds: one bool per row, True for a real record and False for a repeat.

A split's last batch is padded to full size with repeats of its own rows, and
a process whose share runs out before the others' scores a copy of its last
batch in which every row is a repeat. That way a pass is whole batches on
every process and counts each record once. A loss weights its rows by this
field, and its batch-wide terms are taken over the real rows only. Training
batches never have it.
"""


def freeze(variables: Variables, trainable: PathFilter) -> Variables:
    """Move the `params` leaves that `trainable` rejects into the `FROZEN` collection.

    `trainable` sees full leaf paths, `("params", ...)`. Raises `ValueError`
    when it keeps every leaf or none, since then there is nothing to split.
    """
    leaves = jax.tree_util.tree_leaves_with_path(variables["params"])
    kept = sum(trainable(("params", *(entry.key for entry in path))) for path, _ in leaves)
    if kept in (0, len(leaves)):
        raise ValueError(
            f"trainable keeps {kept} of {len(leaves)} parameter leaves, which "
            f"{'freezes' if kept else 'trains'} nothing")
    moving = select(variables, lambda path: path[0] == "params" and trainable(path))
    frozen = select(variables, lambda path: path[0] == "params" and not trainable(path))
    return {**variables, "params": moving["params"], FROZEN: frozen["params"]}


def thaw(variables: Variables) -> Variables:
    """Undo `freeze`, merging the frozen leaves back into one `params` collection.

    Variables without a `FROZEN` collection are returned unchanged.
    """
    if FROZEN not in variables:
        return variables
    rest = {name: value for name, value in variables.items() if name != FROZEN}
    return {**rest, "params": merge(variables[FROZEN], variables["params"])}


def part(variables: Variables, name: str) -> Variables:
    """Cut the `name` subtree out of every collection that holds one: one
    module's own tree out of an objective's that nests several under each
    collection, `params/policy` and `frozen/policy`, so an adapted module's
    base comes with its factors."""
    return {collection: subtree[name] for collection, subtree in variables.items() if name in subtree}


def joined(parts: Mapping[str, Variables]) -> Variables:
    """Nest each module's tree under its name in every collection it holds;
    the inverse of `part`."""
    collections = {collection for tree in parts.values() for collection in tree}
    return {collection: {name: tree[collection] for name, tree in parts.items() if collection in tree}
            for collection in sorted(collections)}


@dataclass(frozen=True)
class EMASpec:
    """Which leaves the EMA copy tracks, and how fast it follows them.

    `decay` is a schedule over steps because some objectives ramp the
    momentum; I-JEPA anneals it from 0.996 to 1.0. The step it reads is the
    count of completed optimizer updates. `select` picks the tracked leaves
    and defaults to all of them.
    """
    decay: optax.Schedule
    select: PathFilter = everything

    @classmethod
    def constant(cls, decay: float | None) -> EMASpec | None:
        """Follow what moves at a constant `decay`, never the frozen collection; None keeps no EMA."""
        return None if decay is None else cls(optax.constant_schedule(decay), lambda path: path[0] != FROZEN)


class SavedTask(Protocol):
    """The task class a saved run of an objective loads as (`Objective.saved_task`).

    It is a class whose `from_run` builds the task from a run directory.
    Dew's tasks (`TextToImage`, `TextGeneration`, `BlockGeneration`,
    `MaskedGeneration`) are such classes, and a plugin can supply its own."""

    @classmethod
    def from_run(cls, directory: str, *, ema: bool | None = None, step: int | str | None = None,
                 mesh: MeshSpec | None = None, layout: Layout | None = None,
                 dtype: DTypeLike | None = None, param_dtype: DTypeLike | None = None) -> Self: ...


class Omitted(enum.Enum):
    """A keyword the caller left out, where None is a value of its own.

    An objective built over a loaded bundle takes what the bundle supplies for
    such a keyword (`variables`, `processor`, a pipeline's `autoencoder`) only
    when it is omitted; an explicit None clears it.
    """

    OMITTED = "omitted"


OMITTED = Omitted.OMITTED


class ProgramModule(NamedTuple):
    """One module an objective's compiled programs execute (`Objective.program_key`)."""

    module: nn.Module
    head_tile: tuple[int, int] | None
    """The tile its vocabulary head computes the backward in, or None for a
    head that keeps its whole logits or a module without one."""
    trained: bool
    """Whether the step moves the module's parameters, so its backward runs:
    False for a frozen teacher or reference."""


class Objective(ABC, Generic[Loss, Effects]):
    """Defines what is learned: the parameters, the loss and what evaluation produces."""

    _inputs: InputSpec | None = None
    variables: Variables | None = None
    """The tree training starts from (a loaded checkpoint, an adapter's
    split), or None to draw one (`init`)."""
    processor: Processor | None = None
    """What turns text into ids and back for the objective's task, or None."""

    @property
    def inputs(self) -> InputSpec | None:
        """The declared input shapes, or None for a custom initializer without an `InputSpec`.

        The spec's sample is the batch field the loss reads, at its
        per-example shape. In a run without a rollout, the trainer checks the
        first batch against it (`InputSpec.check`). It is a property so that
        built-in objectives can narrow it; a diffusion objective, for example,
        requires a spec.
        """
        return self._inputs

    @inputs.setter
    def inputs(self, inputs: InputSpec | None) -> None:
        self._inputs = inputs

    ema: EMASpec | None = None
    saved_task: ClassVar[type[SavedTask] | None] = None
    """The task class a saved run of this objective loads as, which `dew.pipeline` builds from the run.

    A subclass inherits its parent's. None means a saved run of this
    objective loads as no task."""
    _ema_is_reference: ClassVar[bool] = False
    # The name an objective that trains other networks beside its model nests
    # the model under in every collection, as PPO nests its policy beside the
    # critic; None when the saved tree is the model's.
    _model_part: ClassVar[str | None] = None
    artifact: type | None = None
    """The artifact type `evaluate` returns, or None when it returns nothing."""
    shown: Mapping[str, Shown] = {}
    """How the display shows the metrics `loss` reports, keyed by the names it reports them under.

    An entry under `loss` covers the loss itself, for an objective where the
    trainer's default (lower is better) does not hold."""
    resolved: ClassVar[tuple[str, ...]] = ()
    """The constructor arguments the objective resolves from what it is built
    around when they are left out, each held as an attribute of that name.
    A run records the value it resolved, which no default in the signature
    can say: a diffusion objective samples with a loaded pipeline's own
    guidance, else its fallback (`RunConfig.recorded`)."""

    def optimizer(self, tx: optax.GradientTransformation, *,
                  accumulation: int) -> optax.GradientTransformation:
        """Return the optimizer the trainer steps this objective's `params` with, built from `tx`.

        The default returns `tx` itself. An objective that trains several
        networks of its own, such as a few-step student and the fake score
        that critiques it, can give each network its own copy of `tx`, with
        its own state and update count, by `optax.multi_transform` over the
        networks. It can then alternate them with `optax.conditionally_mask`,
        whose count is the trainer's number of committed updates.

        `accumulation` is the trainer's setting. An objective whose loss
        alternates networks from one update to the next refuses more than one
        microbatch per update, because an accumulation window would mix the
        phases.
        """
        return tx

    def averages(self, update: jax.Array) -> jax.Array:
        """Return whether the next update moves the EMA, given `update` committed updates so far.

        The default averages every update. An objective whose EMA follows one
        of the networks it alternates averages only on that network's
        updates; the rCM objective does this for its student.
        """
        return jnp.asarray(a=True)

    def bind_model[Model: nn.Module](self, source: Model | Source[Model], *,
                                      variables: Variables | None | Omitted = OMITTED,
                                      processor: Processor | None | Omitted = OMITTED) -> Model:
        """Hold `source`'s starting tree and processor as `variables` and `processor`, and return its model.

        A loaded source (`Source`, such as `Pretrained.load`'s) supplies all
        three, and `variables` and `processor` override its own, an explicit
        None included. A bare model supplies none, so what is omitted is None.
        The constructor keeps the model as `model`, the module the objective
        trains, which `program_key` names by default.
        """
        if _loaded(source):
            variables = source.variables if variables is OMITTED else variables
            processor = source.text_processor if processor is OMITTED else processor
            model = source.model
        else:
            model = source
        self.variables = None if variables is OMITTED else variables
        self.processor = None if processor is OMITTED else processor
        return model

    def held_variables(self) -> Variables | None:
        """Return the arrays this objective starts from, or None when it initializes them from a key.

        A continued-pretraining objective returns its checkpoint here, and one
        that keeps a frozen tower next to the model it trains returns the
        tower. The trainer calls this once and passes the result to `init`, so
        an objective never has to read its own held arrays inside a trace.
        The default is the starting tree, `variables`.
        """
        return self.variables

    def program_key(self) -> tuple[ProgramModule, ...]:
        """Every module this objective's compiled programs execute, in an order its configuration fixes.

        The trainer substitutes them (`substitute`) to rematerialize or
        quantize, and a compiled validation loss is reused only while they
        are the same modules. The default is the `model` the objective
        holds, trained, with no head tile, or nothing for one that holds
        none. One that runs several networks overrides this and names each,
        a frozen teacher's untrained.
        """
        model = vars(self).get("model")
        return () if model is None else (ProgramModule(model, None, trained=True),)

    def substitute(self, modules: Sequence[nn.Module]) -> None:
        """Replace the modules `program_key` names, one per entry in its order."""
        if len(modules) != len(self.program_key()):
            raise ValueError(f"{type(self).__name__} runs {len(self.program_key())} modules, "
                             f"and {len(modules)} were given")
        if modules:
            (self.model,) = modules

    @property
    def bank_sites(self) -> tuple[DecoderBank, ...]:
        """The scanned layer stacks this objective runs, which a host layout streams as banks.

        Each site names a decoder's namespace below every variables
        collection, and its `StackView`, as the model declares them. An
        objective without a layer stack returns an empty tuple, and all its
        variables are entry variables. Only the execution snapshot uses
        these banks; the variables and optimizer trees the trainer keeps
        never take them on.
        """
        return ()

    @property
    def initializer(self) -> Initializer:
        """`init` as one value a JIT can take, with the held arrays as its arguments.

        The trainer builds the initial state inside one JIT. If that JIT took a
        function with no arguments, every concrete array the body reads would
        be captured as a compiled constant. For a loaded checkpoint, the whole
        parameter tree would then be embedded in the executable: 2.2 GiB for
        a 0.6B model, a module too large for the compilation cache to store.

        This property is the one place where held variables enter that JIT as
        data. `Partial` is a pytree whose bound arguments are its children, so
        they arrive as JIT arguments however deeply `init` nests its own
        compilation, and the call always goes through the public `init`.
        """
        held = self.held_variables()
        return Partial(self.init) if held is None else Partial(self.init, variables=held)

    _held_parts: ClassVar[bool] = False
    """Whether the held tree may be parts a fresh draw goes beside (a
    pipeline's frozen towers) rather than a starting tree, which holds `params`."""

    def init(self, key: jax.Array, variables: Variables | None = None) -> Variables:
        """Return the whole variables tree, every collection, from one key.

        It must be pure, because the trainer traces it once for shapes and
        once for values. `variables` is the held tree the caller supplies,
        which is how the trainer passes it as data; None means take it from
        this objective's own `held_variables`.

        A held tree that holds `params` is the starting tree, kept as given,
        split or whole; otherwise `fresh_variables` draws one. A held tree
        without `params` is refused, unless the objective holds parts a draw
        goes beside. `complete_variables` then adds what the objective's own
        modules need. An objective of several networks overrides this whole.
        """
        held = self.held_variables() if variables is None else variables
        if held is not None and "params" in held:
            tree = held
        elif held is None or self._held_parts:
            tree = self.fresh_variables(key, held)
        else:
            raise ValueError("variables is the variables dict ({'params': ...}) that "
                             "Pretrained.load and model.init return")
        return self.complete_variables(key, tree)

    def fresh_variables(self, key: jax.Array, held: Variables | None) -> Variables:
        """Draw the tree training starts from when none is held.

        `held` is the parts a draw goes beside, for an objective that holds
        them, else None. An objective that inherits `init` implements this.
        """
        raise NotImplementedError(f"{type(self).__name__} draws no fresh tree: give it variables, "
                                  "or implement fresh_variables")

    def complete_variables(self, key: jax.Array, tree: Variables) -> Variables:
        """Return `tree`, held or drawn, with what the objective's own modules
        add: a loss head, a projector, a phase's split. The default adds nothing."""
        return tree

    @property
    def _validation_loss(self):
        """Reuse the statistics program only while its model and head stay fixed.

        Fit's ladder replaces the immutable Flax model and may change a
        tiled head. Argument shapes alone cannot identify those programs.
        Keep only the current specialization, not a history of old models.
        """
        programs = tuple((program.module, program.head_tile) for program in self.program_key())
        cached = vars(self).get('_validation_loss_cache')
        if cached is None or len(cached[0]) != len(programs) or any(
                module is not was or tile != had for (module, tile), (was, had) in zip(programs, cached[0],
                                                                                      strict=True)):
            compiled = jax.jit(lambda variables, batch, step: self._loss(variables, batch, step)[0])
            cached = (programs, compiled)
            self._validation_loss_cache = cached
        return cached[1]

    @functools.cached_property
    def _validation_reduction(self):
        """`reduce_loss` compiled once, for the statistics a validation pass sums."""
        return jax.jit(self.reduce_loss)

    @property
    def scalars(self) -> TrainingScalars[Loss, Effects]:
        return TrainingScalars(self)

    def _scalar(self, name: str) -> TrainingScalar[Loss, Effects]:
        """Select a declared training scalar; `objective.loss` selects the loss itself."""
        if name != 'loss' and name not in self.shown:
            raise ValueError(f"the objective does not declare training scalar {name!r}")
        return TrainingScalar(self, name, self.shown.get(name, Shown(better='lower')))

    @abstractmethod
    def loss(self, variables: Variables, batch: Batch, step: Step) -> Loss | tuple[Loss, Aux[Effects]]:
        """Return additive loss statistics for `batch`, optionally paired with an `Aux` of reports.

        A `Ratio` declares a shared normalization mass, and a plain scalar is
        one term of unit mass. Composite statistics are Flax pytrees the
        objective defines; their leaves add across microbatches before
        `reduce_loss` runs on the sum.
        """

    @staticmethod
    def row_mean(values: jax.Array, batch: Batch, axis: int | tuple[int, ...] | None = None, *,
                 rows: int | tuple[int, ...] = 0) -> Ratio:
        """Return the mean of `values` over `axis`, as the `Ratio` of their sum and count.

        `rows` is the axis of `values`, or the consecutive axes, that index `batch`'s
        rows. The sum runs over `axis` (every axis by default), and the count is the
        number of entries summed. When the batch is an evaluation batch with
        `VALID_ROWS`, a repeat row has zero weight in both, so a loss and every
        batch-wide term computed through this count each real record once. Without
        `VALID_ROWS`, as in training, this is `jnp.sum` and the size.
        """
        dtype = jnp.promote_types(values.dtype, jnp.float32)
        axes = (tuple(range(values.ndim)) if axis is None
                else (axis,) if isinstance(axis, int) else tuple(axis))
        valid = batch.get(VALID_ROWS)
        if valid is None:
            return Ratio(jnp.sum(values, axes), jnp.asarray(math.prod(values.shape[a] for a in axes), dtype))
        held = (rows,) if isinstance(rows, int) else rows
        shape = [values.shape[a] if a in held else 1 for a in range(values.ndim)]
        real = jnp.asarray(valid, bool).reshape(shape)
        # Selected rather than multiplied, so a repeat holding a NaN still adds nothing.
        return Ratio(jnp.sum(jnp.where(real, values, jnp.zeros((), values.dtype)), axes),
                     jnp.sum(jnp.broadcast_to(real, values.shape).astype(dtype), axes))

    @staticmethod
    def accuracy(correct: jax.Array, batch: Batch, weights: jax.Array | None = None, *,
                 rows: int | tuple[int, ...] = 0) -> Ratio:
        """Return the right answers among the counted ones, as the `Ratio` of the two.

        `correct` is 1 where an answer is right and 0 where it is wrong, and
        `weights` (1 everywhere by default) is what each answer counts for, a
        target mask for instance. Both sums go through `row_mean` over every
        axis, with `rows` as it reads them, so a repeat row of an evaluation
        batch counts in neither. `.mean()` is the accuracy, and 0 when nothing
        is counted. Every accuracy an objective reports is this one.
        """
        counted = jnp.ones_like(correct) if weights is None else weights
        hits = Objective.row_mean(correct * counted, batch, rows=rows).total
        return Ratio(hits, Objective.row_mean(counted, batch, rows=rows).total)

    def _loss(self, variables: Variables, batch: Batch, step: Step) -> tuple[Loss, Aux[Effects]]:
        loss = self.loss(variables, batch, step)
        if _has_aux(loss):
            return loss
        return loss, Aux(metrics={})

    def reduce_loss(self, stats: Loss) -> tuple[jax.Array, jax.Array]:
        """Return the objective value of summed statistics and whether they have any mass.

        A `Ratio` reduces by `Ratio.mean`, and a scalar is its own value with
        mass one. Composite statistics need an override, since the default
        raises `TypeError` for them.
        """
        if isinstance(stats, Ratio):
            return stats.mean()
        if isinstance(stats, (jax.Array, float, int)):
            value = jnp.asarray(stats)
            value = value.astype(jnp.promote_types(value.dtype, jnp.float32))
            if value.ndim != 0:
                raise ValueError("a unit-mass loss must be scalar")
            return value, jnp.asarray(a=True)
        raise TypeError("custom loss statistics require Objective.reduce_loss")

    @staticmethod
    def with_gradients(stats: Loss, gradients, params: Variables) -> Loss:
        """Return `stats` with `gradients` as their derivative in `params`: a gradient rule not the loss's.

        `gradients` has the structure of `stats`: for each scalar statistic,
        its gradient with respect to `params` (the trained `params`
        collection `loss` was given), or None for one no parameter moves.
        A rule is the derivative of the statistic as returned, a `Ratio`'s
        total say, so the step applies it over the mass after the reduction,
        as it does the total's own derivative. A
        `loss` that computes its own update rule (e-prop's eligibility traces,
        a forward-gradient or evolution-strategies estimate, a synthetic
        gradient) returns this. Each statistic keeps its value, which adds
        zero, `<gradient, params - stop_gradient(params)>`, whose derivative
        is `gradient`; so the trainer's one gradient path, with its
        microbatching, accumulation, sharding and logging, applies the rule
        as it applies `jax.grad`'s.
        """
        moved = jax.tree.leaves(jax.tree.map(lambda leaf: leaf - jax.lax.stop_gradient(leaf), params))

        def held(value, gradient):
            value = jax.lax.stop_gradient(jnp.asarray(value))
            if gradient is None:
                return value
            if value.ndim:
                raise ValueError(f"a statistic with supplied gradients is a scalar, not {value.shape}")
            pairs = zip(jax.tree.leaves(gradient), moved, strict=True)
            return value + sum(jnp.vdot(jax.lax.stop_gradient(rule), change) for rule, change in pairs)

        return jax.tree.map(held, stats, gradients)

    def scalar_loss(self, variables: Variables, batch: Batch, step: Step) -> tuple[jax.Array, Aux[Effects]]:
        """Return the reduced loss and the `Aux` for `batch`, so JAX can differentiate the loss directly."""
        stats, aux = self._loss(variables, batch, step)
        value, _ = self.reduce_loss(stats)
        return value, aux

    def tile_head(self, tile: tuple[int, int] | None = None) -> str | None:
        """Move a head that keeps its whole logits for the backward pass to a bounded tile.

        The tile is `tile`, or the objective's own when that is None. The
        return value describes the new tile, or is None when there was nothing
        to move, as for an objective with no such head. This is the first rung
        of the fit ladder (`dew.training.trainer.recompute_more`), which logs
        the description, and a resumed run moves to the same rung again.
        """
        return None

    def apply_effects(self, variables: Variables, effects: Effects) -> Variables:
        """Return replacement nonparameter collections computed from one accepted window's `effects`.

        The trainer calls this once per committed update with the effects
        summed over the window's microbatches. The default raises `TypeError`.
        """
        raise TypeError("deferred effects require Objective.apply_effects")

    def predict(self, params: Variables, batch: Batch, step: Step, *, train: bool,
                layers: Sequence[int] = ()) -> tuple[Ratio, Aux[Effects], Prediction]:
        """Return the loss over `batch` as `loss` computes it, together with the prediction behind it.

        The result holds the statistics, the `Aux` reports and a `Prediction`
        with the token logits, the weight of every position and the hidden
        states of `layers`. The statistics are one `Ratio` over the positions
        the weights count, so a distillation can mix in terms over the same
        mass. `train` turns dropout on, as `loss` has it; a frozen teacher
        scores with it off. Objectives that score no token logits raise
        `TypeError`.
        """
        raise TypeError(f"{type(self).__name__} scores no token logits")

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> Artifacts | None:
        """Return scoring artifacts for every row of the global batch that all processes evaluate.

        Every process runs this numerical work, outside the optimizer's jit.
        Display sampling and decoding go in `preview`. Each scoring batch has
        its own key, and `step.ema` holds the averaged weights. The default
        returns `_evaluation_scores` over the `evaluation_variables`, compiled
        once, or None for an objective that scores no tokens.
        """
        if self._evaluation_scores is None:
            return None
        return self._compiled_scores(self.evaluation_variables(params, step), batch, step.key)

    _evaluation_scores: Callable[..., TokenScores] | None = None
    """The per-token scores of one evaluation batch, a pure function of the
    weights, the batch and the pass's key, or None when the objective scores
    no tokens."""

    @functools.cached_property
    def _compiled_scores(self) -> Callable[..., TokenScores]:
        """`_evaluation_scores` compiled once per objective. Run op by op, the
        model's forward would dispatch every operation of every validation
        batch from the host, and jax's eager shard_map refuses the chunked
        head's map over the data axis alone."""
        assert self._evaluation_scores is not None
        return jax.jit(self._evaluation_scores)

    def evaluation_variables(self, variables: Variables, step: Step) -> Variables:
        """Return the weights a validation pass and a preview score: the
        step's average when the run keeps one, unless that average is the
        objective's frozen reference (`_ema_is_reference`), else `variables`."""
        return variables if step.ema is None or self._ema_is_reference else step.ema

    def _pipeline_weights(self, state: TrainState, ema: bool | None) -> Variables:
        if self._ema_is_reference or ema is False or (ema is None and state.ema is None):
            return state.variables
        return state.averaged

    @classmethod
    def _saved_variables(cls, directory: str, *, step: int | str | None, ema: bool | None,
                         mesh: MeshSpec | None = None, layout: Layout | None = None,
                         param_dtype: DTypeLike | None = None,
                         parameter_roots: tuple[tuple[str, ...], ...] = (("params",), (FROZEN,))
                         ) -> Variables:
        """Return the model's variables from a run of this objective, as every run loader reads them.

        The arguments are `Checkpoints.variables`'. An objective whose average
        is its frozen reference gives the trained weights whatever `ema` asks,
        and one that nests the model beside other networks gives the model's
        own part, its trained and frozen collections together.
        """
        from dew.checkpoints import Checkpoints

        variables = Checkpoints(directory).variables(
            step=step, ema=False if cls._ema_is_reference else ema, mesh=mesh, layout=layout,
            param_dtype=param_dtype, parameter_roots=parameter_roots)
        return variables if cls._model_part is None else part(variables, cls._model_part)

    def task_record(self) -> Mapping[str, JSON] | None:
        """Return the task settings a saved step records beside its model, or
        None when the objective declares no inference contract.

        They are what the objective's task is built with and a loader cannot
        read off the model: a sampling policy, a token budget, a tokenizer, a
        process. `inference_record` puts them beside the model's record.
        """
        return None

    def inference_record(self) -> JSON:
        """Return the model and task settings that let a saved step be rebuilt.

        It records the objective's import path, its `model`'s record, and
        `task_record`'s settings. An objective that declares no task settings
        returns None; a custom research method can still restore its raw state.
        """
        from dew.config import ModelConfig
        from dew.registry import import_path, to_record

        settings = self.task_record()
        if settings is None:
            return None
        return {'objective': import_path(type(self)),
                'model': to_record(ModelConfig.from_model(self.model), ModelConfig), **settings}

    def build_task(self, variables: Variables, *,
                   processor: Processor | None | Omitted = OMITTED) -> Task | SavedTask:
        """Return the task over `variables`, the trained tree as the run keeps it.

        `processor` encodes and decodes in place of the objective's own, an
        explicit None included. Objectives without a generation task raise
        `TypeError`. A plugin objective returns its own `saved_task` class, so
        the return type also allows a `SavedTask` besides Dew's own tasks.
        """
        raise TypeError(f"{type(self).__name__} has no inference task")

    def pipeline(self, state: TrainState, *, ema: bool | None = None,
                 processor: Processor | None | Omitted = OMITTED) -> Task | SavedTask:
        """Return the trained model as its inference task, over `state`'s weights.

        With `ema` None, the task uses `state.averaged` when the objective keeps an
        average and the live parameters otherwise, which is how `dew.pipeline` reads a
        run. True requires the average, and False selects the live parameters. An
        objective with a reference policy returns the trained policy, never the frozen
        reference its loss compares against. The arrays keep their placement.
        `build_task` builds the task, with `processor` in place of the objective's own.
        """
        return self.build_task(self._pipeline_weights(state, ema), processor=processor)

    def preview(self, params: Variables, batch: Batch, step: Step, *,
                scored: Artifacts | None = None) -> Artifacts | None:
        """Return one event's display artifacts, reusing the first batch's scores when given.

        The trainer calls this on every process, with a preview key in
        `step.key` that is separate from the scoring keys. Before any
        collective inside it, use `agree_process_phase` to agree on local
        setup and generation failures, so every process reaches the same
        point. Finish all gathers before decoding on the root process alone.
        The trainer agrees on the hook's final outcome across processes before
        it runs any later collective. The default returns `scored` when given,
        and otherwise the result of `evaluate`.
        """
        return scored if scored is not None else self.evaluate(params, batch, step)


M = TypeVar("M", bound="nn.Module", covariant=True)


@runtime_checkable
class Source(Protocol[M]):
    """A loaded model an objective can train in place of a bare model:
    `LMObjective(qwen, seq_len=512)` reads its `model`, its `variables` as
    the starting tree and its `text_processor`. `dew.interop.Pretrained` is
    one; the protocol keeps `dew.objectives` from importing `dew.interop`.
    `M` is the kind of model it carries, which an objective that trains one
    kind checks when it reads it."""

    @property
    def model(self) -> M: ...

    @property
    def variables(self) -> Variables: ...

    @property
    def text_processor(self) -> Processor | None: ...


S = TypeVar("S")


def _loaded[Model: nn.Module](source: Model | Source[Model]) -> TypeIs[Source[Model]]:
    """Whether `source` is a loaded source rather than a bare model."""
    return isinstance(source, Source)


@runtime_checkable
class Metric(Protocol[S]):
    """Reduces a validation pass to one scalar, on the host.

    The metric computes statistics for each batch as it arrives, merges them
    into an accumulator and finalizes the accumulator once at the end.

    The first batch's statistics become the pass's accumulator. That state
    belongs to the one pass, so `merge` may update its buffers in place. A
    metric must never run process collectives or keep state between passes.
    It may have a `shown` attribute, a `Shown` for how the training display
    shows it.
    """

    @property
    def name(self) -> str: ...

    @property
    def reads(self) -> type:
        """The scoring artifact type this metric reads."""
        ...

    def __call__(self, artifact: Artifact, batch: Batch, /) -> S:
        """Compute one complete batch's sufficient statistics from the scoring artifact of type `reads`."""
        ...

    def merge(self, accumulated: S, contribution: S, /) -> S:
        """Merge one batch's statistics into the pass's accumulator and return the result."""
        ...

    def finalize(self, accumulated: S, /) -> float:
        """Return the scalar for the completed pass."""
        ...


def merge_totals(accumulated: tuple[float, float],
                 contribution: tuple[float, float]) -> tuple[float, float]:
    """Add one batch's (total, count) pair into a metric's accumulator.

    Every metric whose statistic is a sum over a count merges this way, so
    a pass over batches of different sizes still weighs by the count.
    """
    return accumulated[0] + contribution[0], accumulated[1] + contribution[1]


def mean_of_totals(accumulated: tuple[float, float]) -> float:
    """Divide a metric's summed total by its summed count."""
    return accumulated[0] / accumulated[1]


__all__ = [
    "FROZEN",
    "VALID_ROWS",
    "Aux",
    "Batch",
    "EMASpec",
    "Metric",
    "Objective",
    "Path",
    "PathFilter",
    "Prediction",
    "ProgramModule",
    "Ratio",
    "SavedTask",
    "Shown",
    "Source",
    "Step",
    "TrainingScalar",
    "TrainingScalars",
    "Variables",
    "everything",
    "freeze",
    "joined",
    "merge",
    "part",
    "select",
    "thaw",
    "under",
]
