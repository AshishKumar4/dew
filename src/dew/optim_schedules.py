"""A solver's learning rate held as operands of its state: what a schedule is (`ScheduleBase`, whose kinds
`dew.optim` states), the stateful hyperparameter that holds a schedule's rate-valued fields and step horizons
(`scheduled_solver`), and what reads or sets them on an optimizer's state (`with_rates`, `with_horizons`,
`learning_rate`, `scaled_rates`): one program at any rate and length of run.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew._records import share
from dew._schedule_pieces import Horizon
from dew.nn.precision import at_least_fp32

type Rates = Mapping[str, jax.Array | float]
"""A schedule's rate-valued fields by path (`ScheduleBase.rates`): the configured values, or a solver's
operands for them (`RateState`)."""


@dataclasses.dataclass(frozen=True)
class ScheduleBase:
    """The base of the schedule records, each of which holds its own fields
    and builds an optax schedule.

    A run's record names the schedule's class and holds only the fields that
    schedule reads; `dew.registry.schedules` holds the short aliases. A
    subclass builds its values in `values`; `schedule` reads them. The fields
    it marks `metadata={'rate': True}` are its rates (`rates`), which a solver
    holds as operands (`scheduled_solver`) and `values` reads from `rates`, so
    a run's program does not depend on them; a subclass that marks none
    implements `values(horizons)` alone. Its step horizons (`horizons`) are
    operands too, so it depends on no run's length.
    """

    every: int = dataclasses.field(default=1, kw_only=True)
    """The updates each of the schedule's own steps lasts.

    torch recipes step a scheduler once an epoch, and with `every` set to the
    updates of an epoch this one holds each value through an epoch as torch's
    does. A run of `steps` updates is then `steps // every` of the
    schedule's own steps, and every field that counts steps (`warmup_steps`,
    `decay_steps`) counts those.
    """

    def __post_init__(self):
        if self.every < 1:
            raise ValueError(f"a schedule advances once every 1 or more updates, not {self.every}")

    def rates(self) -> dict[str, float]:
        """Each rate-valued field's value (`metadata={'rate': True}`) by its dotted path, nested
        records such as PowerTail's too, a None field's derived default resolved where the schedule
        derives one (OneCycle's) and left out otherwise."""
        def walked(record, prefix):
            for field in dataclasses.fields(record):
                value = getattr(record, field.name)
                if field.metadata.get('rate') and value is not None:
                    yield prefix + field.name, value
                elif dataclasses.is_dataclass(value):
                    yield from walked(value, f"{prefix}{field.name}.")
        return dict(walked(self, ""))

    def schedule(self, steps: int | None, rates: Rates | None = None,
                 horizons: Mapping[str, Horizon] | None = None) -> optax.Schedule:
        """Return the value at each update of a run of `steps` updates, as an optax schedule, its
        rate-valued fields `rates`' (`rates()` by default) and its step horizons `horizons`'
        (`horizons(steps)` by default): operands in a solver's state, or the configured values."""
        held = self.rates() if rates is None else rates
        spans = self.horizons(steps) if horizons is None else horizons
        values = self.values(spans, held) if held else self.values(spans)
        if self.every == 1:
            return values
        every = self.every
        return lambda count: values(jnp.asarray(count) // every)

    def horizons(self, steps: int | None) -> dict[str, int | float]:
        """Its step horizons over a run of `steps` updates, in its own steps (`every`), checked: the counts
        its values read (a warmup, a decay's ends, a run's last step), which a solver holds as operands beside
        its rates (`RateState.horizons`), so a run's step is one program at any length."""
        if steps is not None and steps // self.every < 1:
            raise ValueError(f"a run of {steps} updates never advances a schedule that steps once "
                             f"every {self.every}")
        return {}

    def _own(self, steps: int | None) -> int | None:
        ScheduleBase.horizons(self, steps)
        return None if steps is None else steps // self.every

    def _horizon(self, steps: int | None, field: str) -> int:
        """Resolve a decay's step horizon, which a token budget cannot supply."""
        if steps is None:
            raise ValueError(f"{type(self).__name__} needs {field} or trainer.steps as its step horizon")
        return steps

    def warmup(self, steps: int | None) -> int | None:
        """Its warmup in its own steps over a run of `steps` updates, its share of the run resolved
        (`_warmup`), as a run's plan records it; None for a schedule with no warmup of steps."""
        return None

    def _warmup(self, written: int | str, steps: int | None, horizon: int | None = None) -> int:
        """A warmup of `written` steps in its own steps over a run of `steps` of them: as written, or a
        share of the run such as `5%` (`dew._records.share`), the whole steps nearest it, of the run's
        steps or, where the run has none, of the schedule's own `horizon`."""
        if not isinstance(written, str):
            return written
        return round(share(written) * self._horizon(steps if steps is not None else horizon, "warmup_steps"))

    def values(self, horizons: Mapping[str, Horizon], rates: Rates | None = None
               ) -> optax.Schedule:
        """Return the value at each of the schedule's own steps, from its step horizons (`horizons`) and
        its rate-valued fields `rates` (`rates()`), either Python numbers or a solver's operands. A value
        derived from the rates alone, as a ratio of
        two, is the configured fields', which a solver's operands keep: they differ from them by
        one factor, a param group's multiplier."""
        raise NotImplementedError

    def peak(self) -> float:
        """The highest value the schedule takes: its largest rate-valued field, a warmup's or decay's
        ends, where its formula reaches no higher."""
        return max(self.rates().values(), default=1.0)

    def run_relative(self) -> bool:
        """Whether its horizon is the run's, so a run that counts no updates ahead (one bounded in
        examples) reads it over the run's progress (`progressed`); a schedule whose every horizon is its
        own reads its updates there as anywhere."""
        return False

    def written_warmup(self) -> int | str | float | None:
        """Its warmup as written, in its own steps or as a share of the run, which a run that counts no
        updates ahead records as it is (`OptimConfig.warmups`); None where it has none."""
        return None

    def marks(self) -> dict[str, int]:
        """The update counts its law on a run's progress starts a piece at (`progressed`), by name: the
        end of a warmup counted in updates, say. The run records where its progress stood as its updates
        reached each (`ProgressState.marks`)."""
        return {}

    def progressed(self, rates: Rates, progress: jax.Array, updates: jax.Array,
                   marks: Mapping[str, jax.Array]) -> jax.Array:
        """Its value where a run that counts no updates ahead stands: at `progress`, the share of the
        run's bound it has consumed (`dew.state.TrainState.consumed`), after `updates` updates, the
        progress it stood at as it reached each count it `marks`. Each piece written in updates (a
        warmup of 1000) reads the updates, and each that lasts to the run's end reads the progress from
        where it starts, its law kept: a share of the run (`5%`) is that share of the progress, and the
        run's end its whole. A schedule's own steps (`every`) step its updates alone."""
        raise NotImplementedError(f"{type(self).__name__} reads its run's horizon and states no law on the "
                                  "run's progress")

    def _counted(self, updates: jax.Array) -> jax.Array:
        """`updates` in its own steps (`every`)."""
        return jnp.asarray(updates) // self.every

    def _warm(self, written: int | str, progress: jax.Array, updates: jax.Array,
              marks: Mapping[str, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array]:
        """A warmup of `written` on a run's progress: how far through it the run stands, 0 to 1 (`in`),
        whether it is over, and the progress it ended at, its law's next piece's start: written in
        updates, of its own steps and its end marked (`marks['warmup']`); a share of the run, of the
        progress."""
        if isinstance(written, str):
            span = share(written)
            through = jnp.clip(progress / span, 0.0, 1.0) if span else jnp.ones_like(progress)
            return through, progress >= span, jnp.asarray(span, progress.dtype)
        count = self._counted(updates)
        through = (jnp.clip(count.astype(progress.dtype) / written, 0.0, 1.0) if written
                   else jnp.ones_like(progress))
        return through, count >= written, marks.get("warmup", jnp.zeros([], progress.dtype))




class RateState(NamedTuple):
    """A scheduled solver's schedule (`scheduled_solver`): the `updates` it has applied, its schedule's
    rate-valued fields by path (`ScheduleBase.rates`), or for one that marks none the factor its values take
    under `CONSTANT`, and its step horizons by name (`ScheduleBase.horizons`)."""

    updates: jax.Array
    rates: Mapping[str, jax.Array]
    horizons: Mapping[str, jax.Array]


class ProgressState(NamedTuple):
    """A scheduled solver's schedule over a run's progress (`ScheduleBase.progressed`): its `RateState`'s
    `updates` and `rates`, the progress the run stood at as its updates reached each count its law
    marks (`ScheduleBase.marks`), by name, 0 until then, and the progress it stood at as its updates last
    reached a multiple of the schedule's `every` (`held`), which its value holds to the next one, as a
    schedule's own steps hold it over updates."""

    updates: jax.Array
    rates: Mapping[str, jax.Array]
    marks: Mapping[str, jax.Array]
    held: jax.Array


CONSTANT = 'value'
"""The name a constant rate is held under (`rates_of`): a `RateState`'s for a schedule that marks no
rate, and Optax's own `learning_rate` hyperparameter for a constant."""


def rates_of(rate: float | ScheduleBase, multiplier: float = 1.0) -> dict[str, float]:
    """`rate`'s rate-valued fields (`ScheduleBase.rates`), or the constant under `CONSTANT`, each times
    `multiplier` on the host: the values a solver's `RateState` holds, each rounded once into it. A
    schedule that marks no rate holds the multiplier alone, which scales its values."""
    held = (rate.rates() or {CONSTANT: 1.0}) if isinstance(rate, ScheduleBase) else {CONSTANT: rate}
    return {path: value * multiplier for path, value in held.items()}


class _Rate(optax.schedules.StatefulSchedule):
    """Optax's stateful hyperparameter of a scheduled solver: the schedule's value at each update, with
    its rate-valued fields operands in the state rather than constants in the program. Where the run is
    `clocked` (bounded in examples, so it counts no updates ahead) and the schedule's horizon is the
    run's (`ScheduleBase.run_relative`), it reads the run's progress, which each update hands it
    (`progress=`), and keeps the progress its law marks (`ProgressState`)."""

    def __init__(self, rate: ScheduleBase, steps: int | None, dtype, multiplier: float = 1.0,
                 clocked: bool = False, stated: bool = False):
        self.rate, self.steps, self.dtype, self.rates = rate, steps, dtype, rates_of(rate, multiplier)
        self.progressed = clocked and rate.run_relative()
        # A schedule over the run's progress reads no horizon of steps (`ScheduleBase.progressed`).
        self.horizons = {} if self.progressed else rate.horizons(steps)
        self.stated = stated

    def init(self) -> RateState | ProgressState:
        updates = jnp.zeros([], jnp.int32)
        rates = {path: jnp.asarray(value, self.dtype) for path, value in self.rates.items()}
        if not self.progressed:
            # Where the run states them on each state it steps (`stated`, `with_horizons`), a fresh state
            # holds none of them: its start is one program at any length too.
            return RateState(updates, rates, {name: jnp.asarray(0 if self.stated else value, type(value))
                                              for name, value in self.horizons.items()})
        return ProgressState(updates, rates, {name: jnp.zeros([], jnp.float32) for name in self.rate.marks()},
                             jnp.zeros([], jnp.float32))

    def update(self, state: RateState | ProgressState, *, progress: jax.Array | None = None,
               **extra_args) -> RateState | ProgressState:
        del extra_args
        updates = optax.safe_increment(state.updates)
        if not isinstance(state, ProgressState):
            return state._replace(updates=updates)
        assert progress is not None, "a schedule on a run's progress is handed it at each update"
        return state._replace(updates=updates, marks=self._marked(state, progress),
                              held=self._held(state, progress))

    def _held(self, state: ProgressState, progress: jax.typing.ArrayLike) -> jax.Array:
        """The progress the schedule's value reads at `state`'s update: the run's own where the update
        starts one of the schedule's steps (a multiple of `every`), else the one that step started at."""
        at = jnp.asarray(progress)
        return jnp.where(state.updates % self.rate.every == 0, at, state.held.astype(at.dtype))

    def _marked(self, state: ProgressState, progress: jax.typing.ArrayLike) -> dict[str, jax.Array]:
        """`state`'s marks with the progress of the update that stands at a marked count recorded: where
        the run stood as the piece the count starts began."""
        at = jnp.asarray(progress)
        return {name: jnp.where(state.updates == count, at, state.marks[name].astype(at.dtype))
                for name, count in self.rate.marks().items()}

    def __call__(self, state: RateState | ProgressState, *, progress: jax.Array | None = None,
                 **extra_args) -> jax.typing.ArrayLike:
        del extra_args
        if isinstance(state, ProgressState):
            # Optax reads the rate once as it builds its state, where the run has consumed nothing.
            progress = jnp.zeros([], jnp.float32) if progress is None else progress
            at, marks = self._held(state, progress), self._marked(state, progress)
            if not self.rate.rates():
                return state.rates[CONSTANT] * self.rate.progressed({}, at, state.updates, marks)
            return self.rate.progressed(state.rates, at, state.updates, marks)
        return stepped_rate(self.rate, self.steps, state)


def stepped_rate(rate: ScheduleBase, steps: int | None, state: RateState) -> jax.typing.ArrayLike:
    """`rate`'s value at `state`'s update over a run of `steps` updates, its rate fields and step horizons
    `state`'s (`RateState`): the rate the update takes, and the one its solver reports until it takes it."""
    if not rate.rates():
        return state.rates[CONSTANT] * rate.schedule(steps, horizons=state.horizons)(state.updates)
    return rate.schedule(steps, state.rates, state.horizons)(state.updates)


def scheduled_solver(make: Callable[..., optax.GradientTransformation], rate: float | ScheduleBase,
                     steps: int | None, opts: Mapping[str, object], b1: ScheduleBase | None = None,
                     multiplier: float = 1.0, clocked: bool = False, stated: bool = False
                     ) -> optax.GradientTransformation:
    """`make(learning_rate, **opts)` at `rate`, a constant or a schedule over a run of `steps` updates,
    times `multiplier`, its rate-valued fields operands in the optimizer's state (`RateState`): runs
    at different rates of one schedule's timings lower to one executable, and each update's rate is
    the schedule's own value from those fields, Optax's arithmetic where it is Optax's schedule.

    Optax injects the rate and a `b1` schedule (`optax.inject_hyperparams`), and rebuilds the
    factory from their traced values; every other option keeps the Python value the factory branches
    on. The rate reaches the factory as a schedule, so Optax casts it to each update leaf's dtype, and
    the injected values are at least float32 on a bf16 tree. The factory is built once here too, so a
    bad option raises now rather than at the state's init. A `clocked` run counts no updates ahead (it
    is bounded in examples), so a schedule whose horizon is the run's reads the run's progress, which
    each update is handed (`_Rate`). Its step horizons are operands too (`RateState.horizons`), none in a
    fresh state where the run that steps it is `stated` and sets them (`with_horizons`).
    """
    make(1.0 if isinstance(rate, ScheduleBase) else rate, **opts)

    def factory(learning_rate, b1):
        given = opts if b1 is None else {**opts, 'b1': b1}
        return make(lambda count: learning_rate, **given)

    def injected(dtype):
        return optax.inject_hyperparams(factory, hyperparam_dtype=dtype)(
            # A constant is Optax's own numeric hyperparameter, rounded once from its product.
            learning_rate=(_Rate(rate, steps, dtype, multiplier, clocked, stated)
                           if isinstance(rate, ScheduleBase)
                           else rates_of(rate, multiplier)[CONSTANT]),
            b1=(None if b1 is None else _Rate(b1, steps, dtype, clocked=True)
                if clocked and b1.run_relative() else b1.schedule(steps)))

    def init_fn(params):
        # A rate is real: a complex tree's is its components' dtype, at least float32.
        highest = optax.tree.dtype(params, 'highest')
        real = jnp.finfo(highest).dtype if jnp.issubdtype(highest, jnp.complexfloating) else highest
        return injected(at_least_fp32(real)).init(params)

    def update_fn(updates, state, params=None, **extra_args):
        return injected(state.hyperparams['learning_rate'].dtype).update(updates, state, params, **extra_args)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def learning_rate(opt_state: optax.OptState) -> jax.typing.ArrayLike | None:
    """Return the learning rate stored in `opt_state`, if the optimizer stores one.

    An OptimConfig's solver holds the rate it stepped at last (`dew.optim.scheduled_solver`), and an
    Optax optimizer can inject `learning_rate` itself (`optax.inject_hyperparams`); an injected
    wrapper of other hyperparameters is read through to the optimizer it holds. The result is None
    when the optimizer closes over the rate, or when parameter groups hold several rates.
    """
    injected = (optax.InjectHyperparamsState, optax.InjectStatefulHyperparamsState)
    rates = []

    def read(tree):
        for node in jax.tree.leaves(tree, is_leaf=lambda node: isinstance(node, injected)):
            if not isinstance(node, injected):
                continue
            if "learning_rate" in node.hyperparams:
                rates.append(node.hyperparams["learning_rate"])
            else:
                read(node.inner_state)

    read(opt_state)
    return rates[0] if len(rates) == 1 else None


def schedule_states(optimizer: optax.GradientTransformation, params) -> optax.OptState:
    """`optimizer.init(params)` as its shapes, but for its schedules' states (`RateState`,
    `ProgressState`), which are computed: the form `reclocked` gives a supplied state, its step horizons
    the optimizer's own values. One program computes them alone; the moments it would hold it never
    makes."""
    from dew.nn.kernels.options import owned_program

    def scheduled(node) -> bool:
        return isinstance(node, (RateState, ProgressState))

    def states(params) -> list:
        made = jax.tree.leaves(optimizer.init(params), is_leaf=scheduled)
        return [node for node in made if scheduled(node)]

    shapes = jax.eval_shape(optimizer.init, params)
    held = iter(owned_program(states)(params))
    return jax.tree.map(lambda node: next(held) if scheduled(node) else node, shapes, is_leaf=scheduled)


def reclocked(opt_state: optax.OptState, made: optax.OptState) -> optax.OptState:
    """`opt_state`, a state supplied to a run, with each schedule's state in the form the run's own
    optimizer makes (`made`, its init's shapes and its schedules' states, `schedule_states`): a schedule
    over a run's progress (`ProgressState`) that the run counts in updates keeps its updates and rates
    alone (`RateState`), and one the run reads over its progress starts its marks at none, as the run's
    clock starts afresh (`Trainer.fit`'s `state`). A schedule in its own form is kept as it is."""
    def converted(own, held):
        if not isinstance(own, (RateState, ProgressState)) or type(own) is type(held):
            return held
        if isinstance(own, RateState):
            # Its step horizons the destination's own: a stated run's none, which it states (`with_horizons`).
            return RateState(held.updates, held.rates, own.horizons)
        marks = {name: jnp.zeros([], jnp.float32) for name in own.marks}
        return ProgressState(held.updates, held.rates, marks, jnp.zeros([], jnp.float32))

    def scheduled(node) -> bool:
        return isinstance(node, (RateState, ProgressState))

    return jax.tree.map(converted, made, opt_state, is_leaf=scheduled)


def with_rates(opt_state: optax.OptState, rates: Mapping[str | None, Mapping[str, float]]) -> optax.OptState:
    """`opt_state` with each `scheduled_solver`'s rate set to `rates`' for its param group, by name, or
    under None for a solver of no group (`rates_of`): a schedule's rate-valued fields in its
    `RateState`, or a constant in Optax's `learning_rate` hyperparameter. A solver an injected wrapper
    of the run's own holds is reached through the wrapper's inner state (`_rated`). Each keeps the old
    value's dtype and placement."""
    def wanted(path: jax.tree_util.KeyPath, held: Mapping[str, jax.typing.ArrayLike],
               kind: str) -> Mapping[str, float]:
        return rates[group_named(path, rates)]

    return _rated(opt_state, wanted)


def with_horizons(opt_state: optax.OptState, horizons: Mapping[str | None, Mapping[str, int | float]],
                  schedules: Mapping[str | None, ScheduleBase], steps: int | None) -> optax.OptState:
    """`opt_state` with each scheduled solver's step horizons (`RateState.horizons`) set to `horizons`' for
    its param group, by name, or under None for a solver of no group (`OptimConfig.horizons`), each in its
    old value's dtype, placed as the state's update count is, which it is read beside, its shapes' leaves
    made where it holds shapes alone (a restore's template); and the rate the solver reports until its next
    update takes it (Optax's injected `learning_rate`) its schedule's at its update under them
    (`stepped_rate`, `schedules`' for its group, over `steps` updates): a fresh state, built with none,
    reported the schedule as if it ended at once. Solvers are reached as `_rated` reaches them
    (`_injected`)."""
    def stated(path: jax.tree_util.KeyPath, node):
        held = node.hyperparams_states.get("learning_rate")
        if not isinstance(held, RateState):
            return node
        group = group_named(path, horizons)
        # A horizon of a reclocked state comes from a program of the optimizer's alone (`schedule_states`),
        # on one device: it is placed beside the update count instead.
        placed = held.updates.sharding if isinstance(held.updates, jax.Array) else None
        state = held._replace(horizons={name: placed_like(horizons[group][name], like, placed)
                                        for name, like in held.horizons.items()})
        states = {**node.hyperparams_states, "learning_rate": state}
        reported = node.hyperparams["learning_rate"]
        if not isinstance(reported, jax.Array):
            # A restore's template, whose restore reads the rate the checkpoint reported.
            return node._replace(hyperparams_states=states)
        stepping = functools.partial(stepped_rate, schedules[group], steps)
        rate = jax.vmap(stepping)(state) if reported.ndim else stepping(state)
        return node._replace(hyperparams_states=states,
                             hyperparams={**node.hyperparams, "learning_rate": placed_like(rate, reported)})

    return _injected(opt_state, stated)


def held_rates(opt_state: optax.OptState) -> dict[jax.tree_util.KeyPath, dict[str, float]]:
    """Each injected solver's learning rate as `opt_state` holds it, on the host, by the solver's path
    (`_rated`): a schedule's rate-valued fields, or a constant under `CONSTANT`."""
    held: dict[jax.tree_util.KeyPath, dict[str, float]] = {}

    def read(path: jax.tree_util.KeyPath, rates: Mapping[str, jax.typing.ArrayLike], kind: str) -> None:
        held[path] = {name: float(np.asarray(value).reshape(-1)[0]) for name, value in rates.items()}

    _rated(opt_state, read)
    return held


def scaled_rates(opt_state: optax.OptState, held: Mapping[jax.tree_util.KeyPath, Mapping[str, float]],
                 factor: float) -> optax.OptState:
    """`opt_state` training at `factor` times the learning rate each injected solver held where `held`
    was read (`held_rates`), of a state of the same optimizer: a schedule's values are proportional to
    its rate-valued fields (`ScheduleBase.values`), so its every value is the held one's times
    `factor`."""
    return _rated(opt_state, lambda path, *_: {name: value * factor for name, value in held[path].items()})


def settable_rates(opt_state: optax.OptState) -> bool:
    """Whether `opt_state` holds a learning rate and every one it holds is one `with_rates` sets: a
    schedule's fields in its `RateState` or a constant, not a rate Optax computes from a step count
    alone at each update (`optax.inject_hyperparams` of a schedule function), read through `_rated`."""
    kinds: list[str] = []

    def read(path: jax.tree_util.KeyPath, held: Mapping[str, jax.typing.ArrayLike], kind: str) -> None:
        kinds.append(kind)

    _rated(opt_state, read)
    return bool(kinds) and "computed" not in kinds


type Wanted = Callable[[jax.tree_util.KeyPath, Mapping[str, jax.typing.ArrayLike], str],
                       Mapping[str, float] | None]
"""What a solver at a path should hold of the learning rate it holds, of its kind (`_rated`), None for
what it holds."""


def placed_like(value, like, placed=None) -> jax.Array:
    """`value` as `like` holds it, in its dtype and shape, placed where `like` is or at `placed`: a DiLoCo
    run's groups hold their rates, horizons and bounds stacked, each the same
    (`dew.training.diloco.rated`)."""
    return jax.device_put(np.broadcast_to(np.asarray(value, like.dtype), like.shape),
                          like.sharding if placed is None else placed)


type Injected = optax.InjectStatefulHyperparamsState
"""The state Optax's `inject_hyperparams` keeps of an injected solver, its hyperparameters beside it."""


def _injected(opt_state: optax.OptState, edit: Callable[[jax.tree_util.KeyPath, Injected], Injected]
              ) -> optax.OptState:
    """`opt_state` with each injected solver's state that holds a learning rate (Optax's
    `InjectStatefulHyperparamsState`) as `edit` gives it for the solver's path; a solver an injected wrapper
    holds is reached through the wrapper's inner state first."""
    injected = optax.InjectStatefulHyperparamsState

    def walked(tree, prefix: jax.tree_util.KeyPath):
        def replaced(path: jax.tree_util.KeyPath, node):
            if not isinstance(node, injected):
                return node
            full = (*prefix, *path)
            node = node._replace(inner_state=walked(node.inner_state,
                                                    (*full, jax.tree_util.GetAttrKey('inner_state'))))
            return edit(full, node) if 'learning_rate' in node.hyperparams else node

        return jax.tree_util.tree_map_with_path(replaced, tree,
                                                is_leaf=lambda node: isinstance(node, injected))

    return walked(opt_state, ())


def group_named(path: jax.tree_util.KeyPath, named: Mapping) -> str | None:
    """The param group a solver at `path` steps, by its name among `named`'s, or None for one of no group."""
    return next((entry.key for entry in path
                 if isinstance(entry, jax.tree_util.DictKey) and entry.key in named), None)


def _rated(opt_state: optax.OptState, wanted: Wanted) -> optax.OptState:
    """`opt_state` with each injected solver's learning rate set to what `wanted` gives for the solver's
    path, the rate it holds and its kind: `scheduled`, a schedule's rate-valued fields in its
    `RateState` or `ProgressState`; `constant`, a constant's own value under `CONSTANT`; or `computed`,
    the value Optax computed at the last update from a schedule function, which it computes again at
    the next. None leaves it as it is (`_injected`). Each keeps the old value's dtype and placement."""
    def rated(path: jax.tree_util.KeyPath, node):
        schedule = node.hyperparams_states.get('learning_rate')
        scheduled = isinstance(schedule, RateState | ProgressState)
        held = schedule.rates if scheduled else {CONSTANT: node.hyperparams['learning_rate']}
        given = wanted(path, held, "scheduled" if scheduled else "constant" if schedule is None
                       else "computed")
        if given is None:
            return node
        if scheduled:
            fields = {name: placed_like(given[name], held) for name, held in schedule.rates.items()}
            states = {**node.hyperparams_states, 'learning_rate': schedule._replace(rates=fields)}
            return node._replace(hyperparams_states=states)
        rate = placed_like(given[CONSTANT], node.hyperparams['learning_rate'])
        return node._replace(hyperparams={**node.hyperparams, 'learning_rate': rate})

    return _injected(opt_state, rated)


__all__ = ["ScheduleBase"]
