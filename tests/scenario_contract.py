"""The contracts docs/design/testing.md's scenarios check ("each scenario checks every contract its flow
touches", §Scenarios), each one function a scenario or a capability-matrix cell calls on the run it already
made: a claim its own run can show costs no second run, and a claim that needs one (a learning claim, over
independent runs) is paid once a scenario."""

import itertools
from collections.abc import Callable, Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from dew.objectives.base import Step
from dew.training import Trainer

HELD_OUT = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=None)
"""The step a held-out loss is read at: its key fixed, so a run's loss before and after read one draw."""


def assert_steps_finite(scalars: Iterable[tuple[int, Mapping[str, float]]], label: str) -> None:
    """Every committed update's gradient finite: its norm (`train/grad_norm`, the norm of the gradient the
    update applies, an accumulation window's sum, so a microbatch's gradient that is not finite shows in its
    window's commit), read from the scalars a run logged; a run that logged no commit shows nothing and is
    refused. Not `train/accepted`, which a run without dynamic loss scaling sets whatever its gradient
    (`dew.training.transaction`)."""
    norms = [(step, logged["train/grad_norm"]) for step, logged in scalars if "train/grad_norm" in logged]
    assert norms, f"{label}: no committed update logged its gradient's norm"
    lost = [step for step, norm in norms if not np.isfinite(norm)]
    assert not lost, f"{label}: the updates at steps {lost} applied a gradient that was not finite"


def held_out_measures(objective, optimizer, loader_of: Callable[[int], object],
                      held_out_of: Callable[[int], Mapping], *, runs: Iterable[int], steps: int,
                      measure: Callable[[Mapping, Mapping], float] | None = None,
                      **trainer) -> tuple[np.ndarray, np.ndarray]:
    """Each of the independent runs `runs` names, its held-out measure before training and after `steps` steps
    (the runs a learning claim's gate asks for, `learning_gate.assert_learns`, trained as it asks): on the
    run's own held-out batch (`held_out_of(run)`), each run from its own initialization key and data
    (`loader_of(run)`), a Trainer of `trainer`'s options (a rollout's among them). The measure is
    `measure(variables, held_out)`, the objective's loss read at `HELD_OUT` where None. A learning claim's
    contrasts are formed of them, one a run, and judged by the learning gate
    (`learning_gate.assert_learns`)."""
    def loss(variables: Mapping, held_out: Mapping) -> float:
        return float(objective.scalar_loss(variables, held_out, HELD_OUT)[0])

    measured = loss if measure is None else measure
    before, after = [], []
    for run in runs:
        fitted = Trainer(objective, optimizer, key=jax.random.key(run), **trainer)
        start = fitted.initial_state()
        state = fitted.fit(loader_of(run), steps=steps, log_every=steps)
        held_out = held_out_of(run)
        before.append(measured(start.variables, held_out))
        after.append(measured(state.variables, held_out))
    return np.asarray(before), np.asarray(after)


def covering(axes: Mapping[str, Sequence]) -> list[dict]:
    """A small covering set over a scenario's settings axes (docs/design/testing.md, settings and devices),
    chosen pairwise: the fewest of every combination of the axes' values (greedy, in the combinations'
    order, so a scenario's cells are the same each run) that hold every value and every two values of two
    axes, as the GPU tier chooses its cells (`gpu_flow.greedy`)."""
    from gpu_flow import greedy, held

    names = list(axes)
    combinations = [dict(zip(names, values, strict=True)) for values in itertools.product(*axes.values())]
    return [combinations[index]
            for index in greedy(combinations, set().union(*(held(cell) for cell in combinations)))]
