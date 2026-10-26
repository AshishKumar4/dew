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
import zlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Hashable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    NamedTuple,
    NotRequired,
    Protocol,
    Self,
    TypedDict,
    runtime_checkable,
)

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from jax.tree_util import Partial
from typing_extensions import TypeIs, TypeVar

from dew._distributions import counter_key
from dew.artifacts import Artifact, Artifacts, TokenScores
from dew.nn import categorical
from dew.nn.kernels.options import compiler_options
from dew.nn.precision import accumulated, at_least_fp32, compensated_sum, weighted_total
from dew.nn.protocols import HiddenStates, ProjectsParameters
from dew.nn.sharding import rows_taken
from dew.records import JSON, strings

if TYPE_CHECKING:
    from _typeshed import DataclassInstance
    from flax import linen as nn
    from jax.sharding import Mesh
    from jax.typing import DTypeLike

    from dew.data.pipeline import DeviceStep, ObjectiveData
    from dew.data.text import Tokenizer
    from dew.inference.images import TextToImage
    from dew.inference.tasks import BlockGeneration, MaskedGeneration, Processor, TextGeneration
    from dew.inputs import InputSpec
    from dew.nn.autoencoders import AutoEncoder
    from dew.nn.backbones.decoder_stack import DecoderBank
    from dew.sampling.guidance import Guidance
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


@dataclass(eq=False)
class Recorder:
    """What a loss's `Step.sample` and `Step.boundary` calls produced, by name, each with the examples
    (`Step.example_index`) its rows are: what one run records for others to replay (`Step.tape`). It
    records concrete values, a loss run outside `jax.jit`; a jitted step records through the trainer's
    `record`."""
    values: dict[str, list[tuple[np.ndarray, np.ndarray]]] = field(default_factory=dict)
    kinds: dict[str, str] = field(default_factory=dict)

    def record(self, kind: str, name: str, examples: jax.Array, value: jax.Array) -> None:
        """Keep `value`, a `kind` ("sample" or "boundary") named `name`. A name is one kind and holds
        each example once, as a tape replays it by name and example alone."""
        if self.kinds.setdefault(name, kind) != kind:
            raise ValueError(f"{name!r} is recorded as a {self.kinds[name]} and as a {kind}: a tape "
                             "replays a name as one value")
        places = np.asarray(examples)
        held = [np.asarray(index) for index, _ in self.values.get(name, ())]
        if held and np.intersect1d(np.concatenate(held), places).size:
            raise ValueError(f"{name!r} is recorded twice for an example: a tape holds one value each")
        self.values.setdefault(name, []).append((places, np.asarray(value)))

    def tape(self, examples: int) -> dict[str, np.ndarray]:
        """Every recorded name's values placed at their examples, of `examples` in all: a `Step.tape`.
        Raises `ValueError` for a name not recorded for every example."""
        def placed(name, records):
            first = records[0][1]
            tape = np.zeros((examples, *first.shape[1:]), first.dtype)
            seen = np.zeros(examples, bool)
            for index, value in records:
                tape[index], seen[index] = value, True
            if not seen.all():
                raise ValueError(f"{name!r} is recorded for {int(seen.sum())} of the {examples} examples")
            return tape

        return {name: placed(name, records) for name, records in self.values.items()}


RECORD = "record/"
"""The prefix of the metrics under which a recording step (`Trainer(record=True)`) returns what its
evaluations' `Step.sample` and `Step.boundary` calls produced, one array each (`recorded_metrics`),
which `taken_record` reads back as a `Recorded`."""


class Evaluated(TypedDict):
    """One evaluation's record (`Witness.report`): each kept value by name, `[rows, ...]`, the places
    of its rows (`Step.example_index`), and the trainer column the loss saw (`EXAMPLE_ROWS`), where
    its batch carried one."""
    values: dict[str, jax.Array]
    example_index: jax.Array
    column: NotRequired[jax.Array]


class Recorded(TypedDict):
    """A recording step's record (`transaction.recorded`): its evaluations' values by name and places
    stacked on a first axis, whether each row's place is the one its slot and column give it
    (`column_ok`), and which evaluations ran (`evaluated`)."""
    values: dict[str, jax.Array]
    example_index: jax.Array
    column_ok: jax.Array
    evaluated: jax.Array


def recorded_metrics(record: Recorded) -> dict[str, jax.Array]:
    """`record` as a step's metrics, one array each under `RECORD`, as a compiled step returns them."""
    fields = {f"{RECORD}{name}": record[name] for name in ("example_index", "column_ok", "evaluated")}
    return {**fields, **{f"{RECORD}values/{name}": value for name, value in record["values"].items()}}


def taken_record(metrics: Mapping[str, jax.Array]) -> tuple[Recorded | None, dict[str, jax.Array]]:
    """A step's record read back from its `metrics` (`recorded_metrics`), None for a step that
    records nothing, and the metrics without it."""
    rest = {name: value for name, value in metrics.items() if not name.startswith(RECORD)}
    if f"{RECORD}evaluated" not in metrics:
        return None, rest
    values = {name.removeprefix(f"{RECORD}values/"): value for name, value in metrics.items()
              if name.startswith(f"{RECORD}values/")}
    return Recorded(values=values, example_index=metrics[f"{RECORD}example_index"],
                    column_ok=metrics[f"{RECORD}column_ok"], evaluated=metrics[f"{RECORD}evaluated"]), rest


@dataclass(eq=False)
class Witness:
    """What one traced evaluation of a loss's `Step.sample` and `Step.boundary` calls produced, by name:
    the traced values themselves, which the evaluation returns as outputs (`RECORD`), so a compiled step
    reports what it ran, each microbatch and each replay its own, where a `Recorder` keeps concrete
    values of a loss run eagerly."""
    values: dict[str, jax.Array] = field(default_factory=dict)
    kinds: dict[str, str] = field(default_factory=dict)

    def keep(self, kind: str, name: str, value: jax.Array) -> None:
        """Keep `value`, a `kind` named `name`. Raises `ValueError` for a name an evaluation produces
        twice, which a witness could not tell apart."""
        if name in self.values:
            raise ValueError(f"{name!r} is produced twice in one evaluation ({self.kinds[name]} and {kind}); "
                             "a recording holds one value a name")
        self.values[name], self.kinds[name] = value, kind

    def report(self, batch: Batch, examples: jax.Array) -> Evaluated:
        """The evaluation's record (`Evaluated`) of the batch the loss saw and its rows' places."""
        evaluated = Evaluated(values=dict(self.values), example_index=examples)
        if EXAMPLE_ROWS in batch:
            evaluated["column"] = jnp.asarray(batch[EXAMPLE_ROWS], jnp.int32)
        return evaluated


EXAMPLE_STREAM = 0x6578616D
"""What a run folds its key with for its examples' draws (`Step.example_key`), apart from its attempts'."""

_STREAMS: dict[int, str] = {}


def stream(name: str) -> int:
    """The fold of a named draw (`Step.sample`): the same on every process and every run. Raises
    `ValueError` for a name whose fold another name has, which would draw what it draws."""
    fold = zlib.crc32(name.encode()) & 0x7FFFFFFF
    if _STREAMS.setdefault(fold, name) != name:
        raise ValueError(f"the draws {name!r} and {_STREAMS[fold]!r} fold alike; rename one")
    return fold


@struct.dataclass
class Step:
    """What the trainer tells an objective about the current step.

    `step` is the count of accepted microbatches over the microbatches each
    step's batch is read in (`Trainer.compile`), which is what an objective's
    own schedules index by, however the trainer splits its batches. `key` is
    new for every attempt at a microbatch, and a replayed microbatch gets the
    key of the attempt that first read it, so it draws the same randomness:
    the key a model's own streams (dropout) take.

    An example's draws are its own (`sample`): each row of the batch is an
    example of the update (`example_index`), and its draws fold from the
    update's key (`example_key`) and its place, so a step reads the same
    noise, times and masks for an example however the batch is split, pooled,
    sharded or resumed. A step built outside a run, with neither, reads its
    rows as the examples 0, 1, ... under `key`.
    """
    step: jax.Array
    key: jax.Array
    ema: Variables | None
    """The variables tree with the averaged leaves in place of the live ones,
    or None when the objective keeps no EMA."""
    training: bool = struct.field(pytree_node=False, default=True)
    """Whether model calls train; static so compiled losses can branch on the mode."""
    committed: jax.Array | None = None
    """The updates the run committed before this step's (`TrainState.updates`), or None for a
    step made outside a run (`updates`)."""
    example_key: jax.Array | None = None
    """The key every example of the update folds its draws from, the same for each microbatch the
    update pools and each replay of one; None outside a run, where `key` is."""
    example_index: jax.Array | None = None
    """Each row's place among the update's examples, `[rows]` int32: its step's place in the window
    times a step's rows, plus its row in the step's batch. None outside a run, where rows are 0, 1, ..."""
    example_count: int | None = struct.field(pytree_node=False, default=None)
    """How many examples the update pools, which a draw stratified over them reads (`places`); None
    outside a run, where it is the batch's rows."""
    tape: Mapping[str, jax.Array] | None = None
    """Values recorded by name, each indexed by example on its first axis, which `sample` and `boundary`
    read in place of drawing or computing them (`Recorder.tape`)."""
    recorder: Recorder | None = struct.field(pytree_node=False, default=None)
    """Where `sample` and `boundary` record what they produce, or None."""
    record: bool = struct.field(pytree_node=False, default=False)
    """Whether the loss returns what its `sample` and `boundary` calls produce as outputs (`RECORD`):
    a static flag, so a compiled step's cache holds one program a mode."""
    witness: Witness | None = struct.field(pytree_node=False, default=None)
    """The evaluation's collector of what `sample` and `boundary` produce, which `Objective._loss` opens
    inside the trace of a recording step (`record`); None otherwise."""

    def places(self, rows: int) -> tuple[jax.Array, int]:
        """The places among the update's examples of a batch's `rows` rows (`examples`) and how many
        examples the update pools: what a draw stratified over the update's examples reads, or a role
        the update gives a share of its examples, the same on every layout."""
        return self.examples(rows), rows if self.example_count is None else self.example_count

    def examples(self, rows: int) -> jax.Array:
        """The places among the update's examples of a batch's `rows` rows."""
        if self.example_index is None:
            return jnp.arange(rows, dtype=jnp.int32)
        if self.example_index.shape != (rows,):
            raise ValueError(f"this step's batch holds {self.example_index.shape[0]} examples, and a draw "
                             f"asked for {rows} rows")
        return self.example_index

    def _taped(self, tape: jax.Array, name: str, examples: jax.Array, row: tuple[int, ...]) -> jax.Array:
        """`tape`, the step's tape of `name`, at `examples`, checked against the update's examples and
        `row`."""
        needed = examples.shape[0] if self.example_count is None else self.example_count
        if tape.shape[0] < needed or tuple(tape.shape[1:]) != row:
            raise ValueError(f"the tape's {name!r} holds {tuple(tape.shape)}, and the step reads "
                             f"{needed} examples of {row}")
        return rows_taken(jnp.asarray(tape), examples)

    def sample(self, name: str, draw: Callable[..., jax.Array], shape: tuple[int, ...] | None = None, *,
               dtype=None, rows: int) -> jax.Array:
        """`draw(key, shape, dtype=dtype)` for each of `rows` examples, under the example's own key
        folded with `name` (`stream`): `[rows, *shape]`, the same values an example draws on any layout,
        as `jax.random`'s draws take them. `dtype` is the draw's own, never a model's, so a float64 twin
        draws what the run drew; None leaves it out, and `shape` None calls `draw(key)`, a draw that fixes
        both. The key is the example's place folded into `counter_key` of the update's key, or of the
        attempt's outside a run. With a `tape` holding `name`, its rows for these examples instead,
        at the draw's dtype."""
        examples = self.examples(rows)
        value = self._drawn(name, draw, shape, dtype, examples)
        if self.recorder is not None:
            self.recorder.record("sample", name, examples, value)
        if self.witness is not None:
            self.witness.keep("sample", name, value)
        return value

    def sample_update(self, name: str, draw: Callable[..., jax.Array], shape: tuple[int, ...] | None = None,
                      *, dtype=None, rows: int) -> jax.Array:
        """`sample`'s draws of `name` for every example of the update, in place order, where a batch of
        `rows` rows is a part of it (`places`): what a role the update gives a count of its examples
        by reads, the same on every layout. Not recorded: `sample` records the batch's own rows."""
        _, count = self.places(rows)
        return self._drawn(name, draw, shape, dtype, jnp.arange(count, dtype=jnp.int32))

    def _drawn(self, name, draw, shape, dtype, examples) -> jax.Array:
        given = () if shape is None else (shape,)
        options = {} if dtype is None else {"dtype": dtype}

        def drawn(key):
            return draw(key, *given, **options)

        if self.tape is not None and name in self.tape:
            like = jax.eval_shape(drawn, jax.random.key(0))
            return self._taped(self.tape[name], name, examples, like.shape).astype(like.dtype)
        key = jax.random.fold_in(counter_key(self.key if self.example_key is None else self.example_key),
                                 stream(name))
        return jax.vmap(lambda example: drawn(jax.random.fold_in(key, example)))(examples)

    def boundary(self, name: str, value: jax.Array) -> jax.Array:
        """`value`, one row an example, where the model's computation starts from it (a noised input,
        times, a mask, loss weights, a preprocessed target): a `tape` holding `name` replaces it with its
        rows for these examples, at `value`'s dtype, and a `recorder` keeps it."""
        examples = self.examples(value.shape[0])
        if self.tape is not None and name in self.tape:
            value = self._taped(self.tape[name], name, examples, value.shape[1:]).astype(value.dtype)
        if self.recorder is not None:
            self.recorder.record("boundary", name, examples, value)
        if self.witness is not None:
            self.witness.keep("boundary", name, value)
        return value

    @property
    def updates(self) -> jax.Array:
        """The updates committed before this step's: the clock of an objective that alternates its
        networks update by update, which an update's accumulated microbatches share. A step made
        outside a run, where each step is its own update, counts its `step`."""
        return self.step if self.committed is None else self.committed


ROUNDING_STREAM = 0x726F756E
"""What `training_rngs` folds its key with for the stochastic rounding stream."""


def training_rngs(key: jax.Array) -> dict[str, jax.Array]:
    """The RNG streams a model draws from `key` in a training pass: `dropout`,
    and `stochastic_rounding`, which a `dew.quantization.Quantization` with
    `bwd_stochastic_rounding` rounds its gradients from.

    The rounding stream is a fold of `key`, so its draws are not dropout's.
    A step's key is the same on every process of a pool, and so is every
    rounding draw. An objective passes it as a model apply's `rngs`."""
    return {"dropout": key, "stochastic_rounding": jax.random.fold_in(key, ROUNDING_STREAM)}


def model_rngs(key: jax.Array, *, training: bool) -> dict[str, jax.Array] | None:
    """The model's training streams from `key`, or none during evaluation."""
    return training_rngs(key) if training else None


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
    record: Evaluated | Recorded | None = None
    """What a recording evaluation's `Step.sample` and `Step.boundary` calls produced (`Step.record`),
    which `Objective._loss` fills (`Evaluated`) and a transaction stacks with its replays' (`Recorded`);
    None otherwise."""


def _has_aux(loss: Loss | tuple[Loss, Aux[Effects]]) -> TypeIs[tuple[Loss, Aux[Effects]]]:
    return isinstance(loss, tuple) and len(loss) == 2 and isinstance(loss[1], Aux)


@dataclass(frozen=True)
class Shown:
    """How the training display shows one reported metric.

    `better` says which way the metric improves, "higher" or "lower", and the
    display colours its change as progress or regress by it. `percent` shows a
    fraction as a percentage. `group` puts the metric under a heading other
    than its name's prefix, and `label` shows it under another name than its
    own. A metric with no `Shown` is shown under its name, without colour.
    """
    better: Literal["higher", "lower"] | None = None
    percent: bool = False
    group: str | None = None
    label: str | None = None


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


TEACHER = "teacher"
"""The collection of a distilling objective's frozen teacher variables."""

UNREAD = "unread"
"""The field of an inference record naming the parts of the run's tree its model does not read
(`without`), which a loader leaves unrestored (`Objective._saved_variables`): a distillation's
teacher and projections beside the student it saves."""


def without(tree: Variables, paths: Sequence[Sequence[str]]) -> Variables:
    """`tree` without the subtree at each of `paths`, a path the tree does not hold left as it is."""
    kept = dict(tree)
    for path in paths:
        name, *rest = path
        if name in kept:
            if rest:
                held = kept[name]
                kept[name] = without(held, [rest]) if isinstance(held, Mapping) else held
            else:
                del kept[name]
    return kept

FROZEN = "frozen"
"""The name of the collection where a partially trained run keeps its frozen weights.

The optimizer updates the `params` collection and nothing else, so the
leaves that `freeze` keeps in `params` are the ones that train. The frozen
leaves stay in this collection as part of the state, and `thaw` merges the
two before the model reads them."""

PARAMETER_ROOTS = (("params",), (FROZEN,))
"""The collections of a variables tree that hold a model's parameters, which a run's loaders read
and a storage dtype casts: the trained ones and the frozen."""


VALID_ROWS = "valid_rows"

EXAMPLE_ROWS = "example_rows"
"""A training batch's column of each row's place among its step's records, int32: the order the
records' stream holds them in, whatever process read a row (`dew.training.distributed.marked`). The
trainer's transaction reads it for the examples' draws (`Step.example_index`) and takes it out of the
batch the objective sees."""
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

    `trainable` sees full leaf paths, `("params", ...)`. A tree already split
    keeps training the leaves its `params` hold, beside those `trainable`
    takes from either collection, and keeps the rest frozen: an adapter
    attached over another keeps both training. Raises `ValueError` when the
    split keeps every leaf or none, since then there is nothing to split.
    """
    training = (frozenset() if FROZEN not in variables else frozenset(
        ("params", *(entry.key for entry in path))
        for path, _ in jax.tree_util.tree_leaves_with_path(variables["params"])))
    variables = thaw(variables)

    def trains(path: Path) -> bool:
        return path in training or trainable(path)

    leaves = jax.tree_util.tree_leaves_with_path(variables["params"])
    kept = sum(trains(("params", *(entry.key for entry in path))) for path, _ in leaves)
    if kept in (0, len(leaves)):
        raise ValueError(
            f"trainable keeps {kept} of {len(leaves)} parameter leaves, which "
            f"{'freezes' if kept else 'trains'} nothing")
    moving = select(variables, lambda path: path[0] == "params" and trains(path))
    frozen = select(variables, lambda path: path[0] == "params" and not trains(path))
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


def projected_parts(params: Variables, owners: Mapping[str, ProjectsParameters | nn.Module | HiddenStates], *,
                    updated: bool) -> Variables:
    """`params` with each part `owners` names projected by its owner's constraint
    (`dew.nn.protocols.ProjectsParameters`, which an objective's `projected` also
    answers), where the tree holds that part and the owner declares one; the rest
    as they are. The projection of an objective that nests several networks'
    parameters under their names (`joined`)."""
    projected = {}
    for name, tree in params.items():
        owner = owners.get(name)
        projects = isinstance(owner, ProjectsParameters)
        projected[name] = owner.projected(tree, updated=updated) if projects else tree
    return projected


def with_projected(variables: Variables, project: Callable[[Variables], Variables]) -> Variables:
    """`variables` with its `params` collection as `project` returns it. A tree may hold none: flax
    leaves an empty collection out, so a parameter-free model's tree is `{}` or its state alone."""
    return {name: project(tree) if name == "params" else tree for name, tree in variables.items()}


def projected_draw(owner: nn.Module, variables: Variables) -> Variables:
    """A module's freshly drawn tree as training stores it: its parameters projected
    once by `owner`'s constraint (`updated=True`), as is where it declares none."""
    if not isinstance(owner, ProjectsParameters):
        return variables
    return with_projected(variables, functools.partial(owner.projected, updated=True))


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
                 mesh: MeshSpec | Mesh | None = None, layout: Layout | None = None,
                 dtype: DTypeLike | None = None, param_dtype: DTypeLike | None = None) -> Self: ...


@runtime_checkable
class SettingsTask(SavedTask, Protocol):
    """A saved task that builds from the settings its run records (`Objective.task_settings`), over a
    model, its variables and a processor: a live objective builds it as a saved run loads it
    (`Objective.build_task`). Dew's token tasks (`TextGeneration`, `BlockGeneration`,
    `MaskedGeneration`) are such classes."""

    @classmethod
    def from_settings(cls, model: nn.Module, variables: Variables, settings,
                      processor: Processor | None) -> Self: ...


@runtime_checkable
class ReleasedTask(SavedTask, Protocol):
    """A saved task that publishes its run as a released checkpoint of its own layout
    (`Decide.save_pretrained`), which `dew export` writes in place of its model's source layout."""

    def save_pretrained(self, directory: str) -> None: ...


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
    head_fused: bool = False
    """Whether its tiled head takes its gradient in the forward
    (`chunked.fused_head_loss`), which is faster than the tiled head's
    backward but holds a chunk of the whole vocabulary's logits beside it,
    more than a tile (`Objective.unfuse_head`)."""


@jax.custom_jvp
def _ruled(value: jax.Array, rule: Variables, params: Variables) -> jax.Array:
    """`value`, whose derivative in `params` is `rule` (`Objective.with_gradients`)."""
    return value


@_ruled.defjvp
def _ruled_jvp(primals, tangents):
    value, rule, _ = primals
    moved_value, _, moved = tangents
    return value, moved_value + sum(accumulated(jnp.vdot, leaf, change) for leaf, change in
                                    zip(jax.tree.leaves(rule), jax.tree.leaves(moved), strict=True))



class Objective(ABC, Generic[Loss, Effects]):
    """Defines what is learned: the parameters, the loss and what evaluation produces."""

    def data(self, tokenizer: Tokenizer | None = None) -> ObjectiveData:
        """Record and host-batch steps for the training and evaluation branches."""
        from dew.data.pipeline import ObjectiveData

        return ObjectiveData()

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
    settings: ClassVar[type[DataclassInstance] | None] = None
    """The dataclass of what a saved run records beside its objective and model
    (`task_settings`), the one list of those fields: the objective writes an
    instance, its `saved_task` reads it back, and a record holding any other
    field is refused (`dew.inference.tasks.run_record`). None records nothing
    more."""
    """The task class a saved run of this objective loads as, which `dew.pipeline` builds from the run.

    A subclass inherits its parent's. None means a saved run of this
    objective loads as no task."""
    _ema_is_reference: ClassVar[bool] = False
    # The name an objective that trains other networks beside its model nests
    # the model under in every collection, as PPO nests its policy beside the
    # critic; None when the saved tree is the model's.
    _model_part: ClassVar[str | None] = None
    # The name the model's tree nests its recorded model (`recorded_model`) under, beside parts its record
    # leaves out, as a decision model nests its backbone beside its head; None when the model's tree is
    # the recorded model's. A LoRA record's factors merge there (`dew.checkpoints.averaged`).
    _recorded_part: ClassVar[str | None] = None
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

    def optimizer(self, tx: optax.GradientTransformation) -> optax.GradientTransformation:
        """Return the optimizer the trainer steps this objective's `params` with, built from `tx`.

        The default returns `tx` itself. An objective that trains several
        networks of its own, such as a few-step student and the fake score
        that critiques it, can give each network its own copy of `tx`, with
        its own state and update count, by `optax.multi_transform` over the
        networks. It can then alternate them with `optax.conditionally_mask`,
        whose count is the trainer's number of committed updates, the clock
        its loss reads as `Step.updates`: every microbatch an update
        accumulates is of that update's phase, however many it takes.
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
        trains, which `program_key` names by default. A loaded task's tree, its
        projections packed for serving, is held as the model's own tree
        (`dew.inference.projections.unpacked`).
        """
        from dew.inference.projections import unpacked

        if _loaded(source):
            variables = source.variables if variables is OMITTED else variables
            processor = source.text_processor if processor is OMITTED else processor
            model = source.model
        else:
            model = source
        self.variables = None if variables is OMITTED or variables is None else unpacked(model, variables)
        self.processor = None if processor is OMITTED else processor
        return model

    def held_variables(self) -> Variables | None:
        """Return the arrays this objective starts from, or None when it initializes them from a key.

        A continued-pretraining objective returns its checkpoint here, and one
        that keeps a frozen tower next to the model it trains returns the
        tower. The trainer calls this once and passes the result to `init`, so
        an objective never has to read its own held arrays inside a trace.
        The default is the starting tree, `variables`.

        A leaf of the `FROZEN` collection that `init` returns as it is, by
        reference, as the base `init` returns the whole collection (it projects
        `params` alone), is shared: the trainer's state holds that very array,
        which no step donates, so a run and its objective hold one frozen tree
        (`dew.training.state.apart`). A frozen leaf `init` builds, another
        array's copy or values of its own, is the state's own.
        """
        return self.variables

    counts_tokens: bool = False
    """Whether the objective counts the targets its loss scores (`scored_tokens`), for a run whose
    length is its tokens (`Trainer.fit(tokens=)`) and its tokens a second; an objective over
    another's batches, a distillation's, counts as that one does."""

    def scored_tokens(self, batch: Batch) -> np.ndarray:
        """Each row's tokens the loss scores, read before device placement: what a token budget
        commits. By default every real input, each row's `real_lengths` summed; an objective that
        scores fewer, a language model its shifted targets, counts those. The FLOP count reads the
        inputs (`real_lengths`)."""
        from dew.data.dataset import rows_of

        rows = rows_of(batch)
        return np.asarray([np.sum(self.real_lengths(jax.tree.map(
            lambda leaf, index=index: leaf[index:index + 1] if np.ndim(leaf) and np.shape(leaf)[0] == rows
            else leaf, batch))) for index in range(rows)], np.int64)

    def real_lengths(self, batch: Batch) -> np.ndarray:
        """Real row (or packed-segment) lengths, read before device placement.

        An objective supplies its own geometry; the base cannot infer token
        counts from an image, a padded tensor shape or a loss's weight.
        """
        from dew.nn.flops import MissingFlopsCount
        raise MissingFlopsCount(f"{type(self).__name__} does not declare real_lengths(batch)")

    def model_flops(self, model, batch: Batch) -> float:
        """Compose one forward's analytical work on this batch's real rows.

        Objectives that execute multiple networks, heads or differently
        sized sequences override this single assembly point. No compiler
        estimate replaces a missing count. `model=None` reads the objective's
        current model, as a prefetched batch does before it is placed.
        """
        from dew.nn.flops import MissingFlopsCount, counted_flops
        if model is None:
            model = vars(self).get("model")
        if model is None:
            raise MissingFlopsCount(f"{type(self).__name__} has no model to count for train/mfu")
        return counted_flops(model, self.real_lengths(batch))

    def training_flops(self, model, batch: Batch) -> float:
        """Full-FT forward/backward estimate, distinct from the forward count.

        Parameter-efficient objectives override this assembly point with
        frozen and trainable matrix work (4*N_frozen + 6*N_trainable), plus
        their attention/state terms. They cannot reuse three times forward
        work. Until their geometry is supplied, model_flops refuses the
        adapter by name rather than reporting a full-FT MFU for PEFT.
        """
        return 3 * self.model_flops(model, batch)

    def program_key(self) -> tuple[ProgramModule, ...]:
        """Every module this objective's compiled programs execute, in an order its configuration fixes.

        The trainer substitutes them (`substitute`) to rematerialize or
        quantize, and compiled evaluation programs are reused only while they
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
        modules need, and the whole tree's parameters are projected once
        (`projected`, `updated=True`), which leaves held parts that satisfy
        the model's constraint as they are and brings drawn ones to it. An
        objective of several networks overrides this whole.
        """
        held = self.held_variables() if variables is None else variables
        if held is not None and "params" in held:
            tree = held
        elif held is None or self._held_parts:
            tree = self.fresh_variables(key, held)
        else:
            raise ValueError("variables is the variables dict ({'params': ...}) that "
                             "Pretrained.load and model.init return")
        tree = self.complete_variables(key, tree)
        return with_projected(tree, functools.partial(self.projected, updated=True))

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

    def program(self, name: Hashable, function: Callable, *, static_argnames: tuple[str, ...] = (),
                holds: tuple[DeviceStep | AutoEncoder | Guidance, ...] = (),
                modules: tuple[nn.Module, ...] = (), step: Mapping[str, bool | int] | None = None):
        """`function` jitted as this objective's program `name`, reused only while its modules and head
        tiles stay fixed (`program_key`), and under the context it traced in (`trace_context`).

        Every program that runs this objective's modules is one of these, whoever calls it: an
        evaluation, a rollout, a host step. Fit's ladder replaces the immutable Flax model and may
        change a tiled head, and `substitute` may quantize it at the same shapes. Argument shapes
        alone cannot identify those programs, nor can they the pipeline's microbatch count, the
        layout's rules or the links a model decides from. `name` names everything else the program
        fixes. `holds` are the objects `function` runs beyond the modules, as the device steps a host
        step runs or a sample's guidance and autoencoder, whose numbers it bakes in: the program is
        reused only while they are the same objects, and one over others replaces it. `modules` are
        the networks it runs beside the objective's own, an autoguide's or an autoencoder's: they key
        it as `holds` do, and its options cover them. `step` are the XLA options of the training step
        the program is part of (`dew.training.memory.step_compiler_options`), beside its modules' own,
        so a host step's programs compile as the resident step does. Keep only the current
        specialization, not a history of old models or steps.
        """
        programs = tuple((program.module, (program.head_tile, program.head_fused))
                         for program in self.program_key())
        cached = vars(self).get('_program_cache')
        if cached is None or len(cached[0]) != len(programs) or any(
                module is not was or head != had for (module, head), (was, had) in zip(programs, cached[0],
                                                                                      strict=True)):
            cached = (programs, {})
            self._program_cache = cached
        from dew.nn.sharding import trace_context
        # The options follow where the program runs (`compiler_options`), so they key it too.
        options = {**compiler_options(*(module for module, _ in programs), *modules), **(step or {})}
        key = (name, trace_context(), tuple(sorted(options.items())))
        held = cached[1].get(key)
        kept = (*holds, *modules)
        if held is None or len(held[0]) != len(kept) or any(
                now is not was for now, was in zip(kept, held[0], strict=True)):
            # A fresh callable gives JAX a fresh trace even for the same bound method.
            @functools.wraps(function)
            def run(*args, **kwargs):
                return function(*args, **kwargs)

            held = (kept, jax.jit(run, static_argnames=static_argnames, compiler_options=options))
            cached[1][key] = held
        return held[1]

    @property
    def _validation_loss(self):
        return self.program(
            'loss', lambda variables, batch, step: self._loss(variables, batch, step, validation=True)[0])

    @property
    def _validation_reduction(self):
        """`reduce_loss` compiled, for the statistics a validation pass sums."""
        return self.program('reduction', self.reduce_loss)

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

    def validation_loss(self, variables: Variables, batch: Batch,
                        step: Step) -> Loss | tuple[Loss, Aux[Effects]]:
        """Return `loss`'s statistics in evaluation mode, unless the objective scores held-out rows
        another way. The objective's own sampling still reads the evaluation key."""
        self._check_validation_loss()
        return self.loss(variables, batch, replace(step, training=False))

    def _check_validation_loss(self) -> None:
        """Refuse a validation loss this objective does not define (`validation_loss_refusal`),
        before fit places its state."""
        if (refusal := self.validation_loss_refusal) is not None:
            raise ValueError(refusal)

    @property
    def validation_loss_refusal(self) -> str | None:
        """Why this objective defines no validation loss, None where `validation_loss` scores
        held-out rows, which a validation pass then records whatever else reads it: the one
        policy of whether an objective has a validation loss, which an objective that scores
        held-out rows otherwise overrides."""
        return None

    @property
    def defines_validation_loss(self) -> bool:
        """Whether `validation_loss` scores held-out rows (`validation_loss_refusal`)."""
        return self.validation_loss_refusal is None

    @staticmethod
    def row_mean(values: jax.Array, batch: Batch, axis: int | tuple[int, ...] | None = None, *,
                 rows: int | tuple[int, ...] = 0, weights: jax.Array | None = None) -> Ratio:
        """Return the mean of `values` over `axis`, as the `Ratio` of their sum and count.

        `rows` is the axis of `values`, or the consecutive axes, that index `batch`'s
        rows. The sum runs over `axis` (every axis by default), and the count is the
        number of entries summed. When the batch is an evaluation batch with
        `VALID_ROWS`, a repeat row has zero weight in both, so a loss and every
        batch-wide term computed through this count each real record once. Without
        `VALID_ROWS`, as in training, this is the sum (`compensated_sum`) and the size.

        With `weights`, each value counts for its weight, broadcast to `values`: the `Ratio`
        is of their weighted sum and the weights' (`dew.nn.precision.weighted_total`), a value
        of weight 0, a padded target's loss, selected away before its product, so a NaN or an
        infinity there adds nothing.
        """
        dtype = at_least_fp32(values.dtype)
        axes = (tuple(range(values.ndim)) if axis is None
                else (axis,) if isinstance(axis, int) else tuple(axis))
        real = None
        if VALID_ROWS in batch:
            held = (rows,) if isinstance(rows, int) else rows
            shape = [values.shape[a] if a in held else 1 for a in range(values.ndim)]
            real = Objective.row_weights(batch, math.prod(values.shape[a] for a in held)).reshape(shape) > 0
        if weights is not None:
            weights = jnp.broadcast_to(weights, values.shape)
            counted = weights if real is None else jnp.where(real, weights, 0)
            return Ratio(*weighted_total(values, counted, axes))
        if real is None:
            count = math.prod(values.shape[a] for a in axes)
            return Ratio(compensated_sum(values, axes), jnp.asarray(count, dtype))
        # Selected rather than multiplied, so a repeat holding a NaN still adds nothing.
        return Ratio(compensated_sum(jnp.where(real, values, jnp.zeros((), values.dtype)), axes),
                     jnp.sum(jnp.broadcast_to(real, values.shape).astype(dtype), axes))

    @staticmethod
    def row_moments(values: jax.Array, batch: Batch, axis: int | tuple[int, ...], *,
                    rows: int | tuple[int, ...] = 0) -> tuple[jax.Array, jax.Array]:
        """Return the mean and the biased variance of `values` over `axis`, over `batch`'s rows
        (`row_mean`), in two passes as `jnp.mean` and `jnp.var` take them; both drop `axis`."""
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        mean, _ = Objective.row_mean(values, batch, axes, rows=rows).mean()
        centered = values - jnp.expand_dims(mean, axes)
        var, _ = Objective.row_mean(jnp.square(centered), batch, axes, rows=rows).mean()
        return mean, var

    @staticmethod
    def row_weights(batch: Batch, rows: int) -> jax.Array:
        """Each of the batch's `rows` rows' weight, float32: 1 for a real row
        and 0 for a repeat that fills an evaluation batch (`VALID_ROWS`), and 1
        for every row of a training batch. A rule an objective computes itself
        weights its rows by this, as `row_mean` does."""
        valid = batch.get(VALID_ROWS)
        if valid is None:
            return jnp.ones((rows,), jnp.float32)
        return jnp.asarray(valid, jnp.float32).reshape(rows)

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

    def _loss(self, variables: Variables, batch: Batch, step: Step, *,
              validation: bool = False) -> tuple[Loss, Aux[Effects]]:
        marked = batch
        if EXAMPLE_ROWS in batch:
            # The trainer's column of the examples' places is the step's (`Step.example_index`).
            batch = {name: value for name, value in batch.items() if name != EXAMPLE_ROWS}
        witness = Witness() if step.record and not validation else None
        step = step if witness is None else replace(step, witness=witness)
        loss = (self.validation_loss if validation else self.loss)(variables, batch, step)
        if _has_aux(loss):
            stats, aux = loss
        else:
            stats, aux = loss, Aux[Effects](metrics={})
        if witness is not None:
            # A batch with no rows has no examples (`example_fields`), and records none.
            places = jnp.zeros((0,), jnp.int32) if step.example_index is None else step.example_index
            aux = replace(aux, record=witness.report(marked, places))
        return stats, aux

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
            value = value.astype(at_least_fp32(value.dtype))
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
        gradient) returns this. Each statistic keeps its value and takes the
        rule as its derivative in `params` (a custom JVP), so the trainer's
        one gradient path, with its microbatching, accumulation, sharding and
        logging, applies the rule as it applies `jax.grad`'s, and a pass that
        reads only the values, a validation pass's, never computes the rule.

        A rule binds to the parameter at its own path, so each statistic's
        gradients name exactly the parameters' paths, each leaf of its
        parameter's shape; any mapping container will do (a dict for a
        FrozenDict), and its dtype may differ. Anything else is refused,
        naming the statistic and the paths.
        """
        def paths(tree) -> dict:
            leaves = jax.tree_util.tree_leaves_with_path(tree)
            return {jax.tree_util.keystr(path): leaf for path, leaf in leaves}

        expected = paths(params)

        def held(where, value, gradient):
            value = jax.lax.stop_gradient(jnp.asarray(value))
            if gradient is None:
                return value
            name = jax.tree_util.keystr(where) or "the loss"
            if value.ndim:
                raise ValueError(f"{name} has supplied gradients, so it is a scalar, not {value.shape}")
            given = paths(gradient)
            if given.keys() != expected.keys():
                raise ValueError(f"the gradients of {name} name {sorted(given.keys() - expected.keys())}, "
                                 f"which no trained parameter is, and miss "
                                 f"{sorted(expected.keys() - given.keys())}")
            wrong = [f"{path} {jnp.shape(given[path])} for {jnp.shape(leaf)}"
                     for path, leaf in expected.items() if jnp.shape(given[path]) != jnp.shape(leaf)]
            if wrong:
                raise ValueError(f"the gradients of {name} are not their parameters' shapes: "
                                 f"{', '.join(wrong)}")
            rule = jax.tree.unflatten(jax.tree.structure(params), [given[path] for path in expected])
            return _ruled(value, jax.lax.stop_gradient(rule), params)

        return jax.tree_util.tree_map_with_path(held, stats, gradients)

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
        of the fit ladder (`dew.training.memory.recompute_more`), which logs
        the description, and a resumed run moves to the same rung again.
        """
        return None

    def unfuse_head(self) -> str | None:
        """Move a tiled head that takes its gradient in the forward to the tiled head's backward.

        The fused head is the faster where it fits; its chunk holds the whole
        vocabulary's logits, which the tiled head bounds to a tile, so this is
        the rung of the fit ladder after `tile_head`. The return value
        describes the move, or is None where nothing moved, as for an
        objective whose step fuses no head.
        """
        return None

    def restore_head(self, tile: tuple[int, int] | None, *, fused: bool) -> None:
        """Set the head's backward tile to `tile`, None keeping its whole logits, and whether a tiled
        head takes its gradient in the forward (`unfuse_head`).

        The trainer fits each batch shape it meets on its own, so before it
        compiles a new shape it puts the objective back at the run's own rung,
        the head's included (`tile_head` and `unfuse_head` only ever move up).
        An objective with no such head has only None to restore."""
        if tile is not None:
            raise ValueError(f"{type(self).__name__} has no head to tile, so no tile {tile} to restore")

    def projected(self, params: Variables, *, updated: bool) -> Variables:
        """Return the stored parameters a training step reads and updates: the trainable `params`
        projected by the model's constraint (`dew.nn.protocols.ProjectsParameters`), as is where the
        model declares none.

        A step's forward reads `projected(params, updated=False)`, and its update applies to them,
        which the optimizer sees; the step stores the updated ones `projected(..., updated=True)`.
        The tree a run starts from is projected once with `updated=True` (`init`). The default
        projects the `model` this objective trains; an objective that trains several networks
        projects each one's own tree (`projected_parts`)."""
        model = vars(self).get("model")
        return model.projected(params, updated=updated) if isinstance(model, ProjectsParameters) else params

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
        for the current modules, or None for an objective that scores no tokens.
        """
        if self._evaluation_scores is None:
            return None
        return self._compiled_scores(self.evaluation_variables(params, step), batch, replace(step, ema=None))

    _evaluation_scores: Callable[..., TokenScores] | None = None
    """The per-token scores of one evaluation batch, a pure function of the
    weights, the batch and the pass's step, whose draws are its examples'
    (`Step.sample`), or None when the objective scores no tokens."""

    _validation_scores: Callable[..., tuple[Loss, Artifacts]] | None = None
    """One validation batch's loss statistics and the artifacts `evaluate`
    returns for it, from one scoring of the batch: a pure function of the
    evaluation weights, the batch and the pass's step in evaluation mode. None
    when the objective scores the two apart. An objective that declares it
    returns what `validation_loss` and `evaluate` return, so a subclass that
    changes either declares its own."""

    def _validation_report(self, variables: Variables, batch: Batch, step: Step, *, loss: bool,
                           evaluate: bool) -> tuple[Loss | None, Artifacts | None]:
        """One validation batch's loss statistics where `loss`, and what
        `evaluate` returns where `evaluate`.

        `variables` are the weights `evaluate` takes; the loss reads their
        `evaluation_variables`. Where both are asked for and the objective
        scores them together (`_validation_scores`), the batch is scored once.
        """
        weights = self.evaluation_variables(variables, step)
        if loss and evaluate and self._validation_scores is not None:
            report = self.program('report', self._validation_scores)
            return report(weights, batch, replace(step, training=False))
        statistics = self._validation_loss(weights, batch, step) if loss else None
        return statistics, self.evaluate(variables, batch, step) if evaluate else None

    @property
    def _compiled_scores(self) -> Callable[..., TokenScores]:
        """`_evaluation_scores` compiled for the current modules. Run op by op, the
        model's forward would dispatch every operation of every validation
        batch from the host, and jax's eager shard_map refuses the chunked
        head's map over the data axis alone."""
        assert self._evaluation_scores is not None
        return self.program('scores', self._evaluation_scores)

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
                         mesh: MeshSpec | Mesh | None = None, layout: Layout | None = None,
                         param_dtype: DTypeLike | None = None,
                         parameter_roots: tuple[tuple[str, ...], ...] = PARAMETER_ROOTS
                         ) -> Variables:
        """Return the model's variables from a run of this objective, as every run loader reads them.

        The arguments are `Checkpoints.variables`'. An objective whose average
        is its frozen reference gives the trained weights whatever `ema` asks,
        and one that nests the model beside other networks gives the model's
        own part, its trained and frozen collections together.
        """
        from dew.checkpoints import Checkpoints

        with Checkpoints(directory) as checkpoints:
            step = checkpoints.pinned(step)
            record = checkpoints.artifact(step)
            unread = record.get(UNREAD, []) if isinstance(record, Mapping) else []
            if not isinstance(unread, list):
                raise ValueError(f"{UNREAD} must be a list of paths")
            variables = checkpoints.variables(
                step=step, ema=False if cls._ema_is_reference else ema, mesh=mesh, layout=layout,
                param_dtype=param_dtype, parameter_roots=parameter_roots,
                unread=tuple(strings(path, UNREAD) for path in unread))
        return cls._model_tree(variables)

    @classmethod
    def _model_tree(cls, variables: Variables) -> Variables:
        """The `_model_part` subtree of `variables`, or the whole tree when the objective names none."""
        return variables if cls._model_part is None else part(variables, cls._model_part)

    def model_variables(self, variables: Variables) -> Variables:
        """Return the model's own tree out of `variables`, a tree this objective trains.

        That is the tree the model applies with, and another objective over
        the same model starts from it. It is the whole tree, or the model's
        part, trained and frozen collections together, of an objective that
        nests the model beside other networks or frozen ones.
        """
        return self._model_tree(variables)

    def task_settings(self) -> DataclassInstance | None:
        """Return the task settings a saved step records beside its model, an
        instance of `settings`, or None when the objective declares no
        inference contract.

        They are what the objective's task is built with and a loader cannot
        read off the model: a sampling policy, a token budget, a tokenizer, a
        process. `inference_record` puts them beside the model's record.
        """
        return None

    def recorded_model(self) -> nn.Module:
        """The model a saved step's record names (`inference_record`): the objective's own."""
        return self.model

    def inference_record(self) -> JSON:
        """Return the model and task settings that let a saved step be rebuilt.

        It records the objective's import path, its `recorded_model`'s record,
        and `task_settings`, field by field. An objective that declares no task
        settings returns None; a custom research method can still restore its
        raw state.
        """
        from dew.config import ModelConfig
        from dew.registry import import_path, record_fields, to_record

        settings = self.task_settings()
        if settings is None:
            return None
        return {'objective': import_path(type(self)),
                'model': to_record(ModelConfig.from_model(self.recorded_model()), ModelConfig),
                **record_fields(settings, type(settings))}

    def build_task(self, variables: Variables, *,
                   processor: Processor | None | Omitted = OMITTED) -> Task | SavedTask:
        """Return the task over `variables`, the trained tree as the run keeps it.

        `processor` encodes and decodes in place of the objective's own, an
        explicit None included. An objective whose `saved_task` builds from
        the settings its run records (`SettingsTask`) builds it from its own
        (`task_settings`) over its model, as a saved run of it loads; any other
        overrides this, or has no inference task and raises `TypeError`. A
        plugin objective's task is its own `saved_task` class, so the return
        type also allows a `SavedTask` besides Dew's own tasks.
        """
        task, settings, model = self.saved_task, self.task_settings(), vars(self).get("model")
        if task is None or settings is None or model is None or not isinstance(task, SettingsTask):
            raise TypeError(f"{type(self).__name__} has no inference task")
        return task.from_settings(model, variables, settings,
                                  self.processor if processor is OMITTED else processor)

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


def recorded_name[S, A](key, names: Collection[str], owner: Objective[S, A] | None = None) -> str:
    """The recorded name a run's score object names, one rule for the scores `best=` and `stop=` read
    live and the ones a checkpoint keeps: a `TrainingScalar` the training split's (of the objective
    `owner` where it is given, by identity), a `Metric` the validation split's ('val'), and a
    `(split, metric)` pair that split's. A metric its split did not record raises with the splits that
    did."""
    split = "val"
    if isinstance(key, tuple) and len(key) == 2:
        split, key = key
    if isinstance(key, TrainingScalar):
        if owner is not None and key.owner is not owner:
            raise KeyError("training score belongs to another objective")
        return f"train/{key.name}"
    if isinstance(key, Metric):
        if f"{split}/{key.name}" in names:
            return f"{split}/{key.name}"
        splits = sorted(name.rpartition("/")[0] for name in names if name.rpartition("/")[2] == key.name)
        raise KeyError(f"metric {key.name!r} has no {split} score" + (
            f"; it is recorded under {', '.join(splits)}: ask (split, metric)" if splits else ""))
    raise KeyError(key)


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



def token_log_probs(logits: jax.Array, tokens: jax.Array) -> jax.Array:
    """`log_softmax(logits)[..., tokens]`, reduced in at least fp32 (`dew.nn.categorical`).

    `tokens` is `logits.shape[:-1]` of ids and the result has that shape.
    """
    return -categorical.cross_entropy(logits, tokens)

__all__ = [
    "FROZEN",
    "OMITTED",
    "VALID_ROWS",
    "Aux",
    "Batch",
    "EMASpec",
    "Metric",
    "Objective",
    "Omitted",
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
    "model_rngs",
    "part",
    "select",
    "thaw",
    "training_rngs",
    "under",
]
