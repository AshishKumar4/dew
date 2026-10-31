"""Optimizers that hold an update back on some calls, as their states tell it (row 1870).

`optax.MultiSteps`, `optax.apply_every` and `optax.apply_if_finite` apply no update on some calls, and
each says by its state which call applied one. The trainer counts its clocks, its update count, its
average and its committed tokens by the updates applied (`update_applied`), keeps the tokens of a held
call waiting (`waiting`), draws each held call apart (`held_places`), and refuses an optimizer whose
state does not say (`refuse_untold`).
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax


class _Hold(NamedTuple):
    """How a wrapper that holds an update back on some calls tells it by its state: whether a call
    applied one, from its state before and after the call (`applied`), and where among the calls it
    holds one stands (`place`), which that call's draws fold in."""

    name: str
    applied: Callable[[Any, Any], jax.Array]
    place: Callable[[Any], jax.Array]


@functools.cache
def _holds() -> Mapping[type, _Hold]:
    """The wrappers that hold an update back on some calls and say by their state which call applied
    one, by their states' types: `optax.MultiSteps`, whose gradient step moves as it applies one, and
    `optax.apply_every`, whose count comes back to 0 as it does. Optax names `apply_every`'s type only
    through its state, made here abstractly, so no backend opens."""
    every = type(jax.eval_shape(optax.apply_every().init, ()))
    return {
        optax.MultiStepsState: _Hold("optax.MultiSteps",
                                     lambda before, after: after.gradient_step != before.gradient_step,
                                     lambda state: state.mini_step),
        every: _Hold("optax.apply_every", lambda before, after: after.count == 0, lambda state: state.count),
    }


def _passed_through() -> Mapping[type, str]:
    """The wrappers whose update is their inner optimizer's, by their states' types, each with the field
    that holds the inner optimizer's state: a held update under one is still the whole update."""
    from dew.optim import PowerProfilesState

    return {optax.InjectHyperparamsState: "inner_state", optax.InjectStatefulHyperparamsState: "inner_state",
            PowerProfilesState: "inner"}


def _outermost(node) -> list:
    """The states of the wrappers that hold the whole update back (`_holds`): the optimizer itself, the
    last of a chain's transforms (`optax.chain`, `optax.named_chain`), or one under a wrapper whose update
    is its inner optimizer's (`_passed_through`)."""
    if isinstance(node, tuple(_holds())):
        return [node]
    passed = _passed_through()
    if type(node) in passed:
        return _outermost(getattr(node, passed[type(node)]))
    if type(node) in (tuple, list) and node:
        return _outermost(node[-1])
    if isinstance(node, dict) and node:
        return _outermost(list(node.values())[-1])
    return []


def holds_back(opt_state: optax.OptState) -> bool:
    """Whether the optimizer whose state is `opt_state` holds its whole update back on some calls."""
    return bool(_outermost(opt_state))


def update_applied(before: optax.OptState, after: optax.OptState) -> jax.Array:
    """Whether the call that took the optimizer's state from `before` to `after` applied an update: the
    wrapper that holds the whole update back applied its own (`_holds`), true where none holds any (row
    1870). The trainer counts its clocks, the update count, the average and the committed tokens, and
    `power_profiles` its averages, by the updates applied."""
    table = _holds()
    done = jnp.asarray(a=True)
    for previous, current in zip(_outermost(before), _outermost(after), strict=True):
        done = done & table[type(current)].applied(previous, current)
    return done


def waiting(opt_state: optax.OptState) -> jax.Array | None:
    """The tokens a fresh run's state holds waiting for an update the optimizer whose state is
    `opt_state` holds back (`TrainState.deferred`): none yet, two uint32 words; None for an optimizer that
    applies every update."""
    return jnp.zeros((2,), jnp.uint32) if holds_back(opt_state) else None


def held_places(opt_state: optax.OptState) -> list[jax.Array]:
    """Where among the calls the wrapper in `opt_state` that holds the whole update back has the next
    call stand (`_Hold.place`), which its draws fold in: the calls of one update draw apart, as the
    slots of the trainer's own window do."""
    table = _holds()
    return [table[type(node)].place(node) for node in _outermost(opt_state)]


_WHOLE = ("hold the whole update back: optax.MultiSteps(optimizer, k), or optax.chain(optimizer, "
          "optax.apply_every(k)); or accumulate gradients with Trainer(accumulation=k)")
"""The compositions whose held updates the trainer counts by, which a refusal names."""


def refuse_untold(opt_state: optax.OptState) -> None:
    """Refuse an optimizer, by its state `opt_state`, that holds an update back where its state does not
    say which call applied one, naming the composition and the one the trainer counts by (`_WHOLE`):
    `optax.apply_if_finite`, which applies a non-finite update past its `max_consecutive_errors`
    unrecorded; `optax.conditionally_mask` over the whole update, whose condition its state does not hold
    (one under a partition routes the update, as a consistency objective's phases do, and is not
    refused); `optax.MultiSteps` with a skip predicate, which discards a skipped input unrecorded; and a
    wrapper that holds updates back anywhere but over the whole update (`_outermost`): within another,
    under a mask, a partition or a condition, before a later transform in a chain, or within a wrapper
    the trainer does not know passes its update through."""
    holds, passed = _holds(), _passed_through()

    def refused(composition: str, instead: str = _WHOLE) -> ValueError:
        return ValueError(f"the optimizer holds updates back where its state does not say which call "
                          f"applied one: {composition}; {instead}")

    def visit(node, where: str | None) -> None:
        if isinstance(node, optax.ApplyIfFiniteState):
            raise refused("optax.apply_if_finite", "drop non-finite steps with Trainer(dynamic_scale=...), "
                          "which leaves a non-finite attempt uncounted and draws its next one afresh")
        if isinstance(node, optax.ConditionallyMaskState) and where is None:
            raise refused("optax.conditionally_mask over the whole update")
        if isinstance(node, tuple(holds)):
            name = holds[type(node)].name
            if where is not None:
                raise refused(f"{name} {where}")
            if isinstance(node, optax.MultiStepsState) and jax.tree.leaves(node.skip_state):
                raise refused("optax.MultiSteps with should_skip_update_fn", "drop non-finite steps with "
                              "Trainer(dynamic_scale=...), and hold updates back with optax.MultiSteps alone")
            for field in node:
                visit(field, f"within {name}")
            return
        if type(node) in passed:
            visit(getattr(node, passed[type(node)]), where)
            return
        if isinstance(node, (optax.MaskedState, optax.PartitionState)):
            inner = "under optax.masked" if isinstance(node, optax.MaskedState) else "under optax.partition"
            for field in node:
                visit(field, inner)
            return
        if isinstance(node, (optax.ConditionallyMaskState, optax.ConditionallyTransformState)):
            for field in node:
                visit(field, "under a condition (optax.conditionally_mask, optax.conditionally_transform)")
            return
        if type(node) in (tuple, list) or isinstance(node, dict):
            members = list(node.values()) if isinstance(node, dict) else list(node)
            for index, member in enumerate(members):
                visit(member, where if index == len(members) - 1 else "before a later transform in its chain")
            return
        if isinstance(node, tuple):
            unknown = (f"within {type(node).__name__}, which the trainer does not know passes its update "
                       "through")
            for field in node:
                visit(field, unknown if where is None else where)

    visit(opt_state, None)
