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

    applied: Callable[[Any, Any], jax.Array]
    place: Callable[[Any], jax.Array]


@functools.cache
def _holds() -> Mapping[type, _Hold]:
    """The wrappers that hold an update back on some calls, by their states' types: `optax.MultiSteps`,
    whose gradient step moves as it applies one; `optax.apply_every`, whose count comes back to 0 as it
    does; `optax.apply_if_finite`, which applies a finite one (one it applies anyway past
    `max_consecutive_errors`, its weights no longer finite, counts as held). Optax names `apply_every`'s
    type only through its state, made here abstractly, so no backend opens."""
    every = type(jax.eval_shape(optax.apply_every().init, ()))
    return {
        optax.MultiStepsState: _Hold(lambda before, after: after.gradient_step != before.gradient_step,
                                     lambda state: state.mini_step),
        every: _Hold(lambda before, after: after.count == 0, lambda state: state.count),
        optax.ApplyIfFiniteState: _Hold(lambda before, after: after.last_finite,
                                        lambda state: state.total_notfinite),
    }


def _held(opt_state: optax.OptState) -> list:
    """The states in `opt_state` of the wrappers that hold an update back (`_holds`), in tree order."""
    kinds = tuple(_holds())
    return [node for node in jax.tree.leaves(opt_state, is_leaf=lambda node: isinstance(node, kinds))
            if isinstance(node, kinds)]


def holds_back(opt_state: optax.OptState) -> bool:
    """Whether the optimizer whose state is `opt_state` holds an update back on some calls."""
    return bool(_held(opt_state))


def update_applied(before: optax.OptState, after: optax.OptState) -> jax.Array:
    """Whether the call that took the optimizer's state from `before` to `after` applied an update: every
    wrapper in it that holds updates back (`_holds`) applied its own, true where none holds any (row
    1870). The trainer counts its clocks, the update count, the average and the committed tokens, and
    `power_profiles` its averages, by the updates applied."""
    table = _holds()
    done = jnp.asarray(a=True)
    for previous, current in zip(_held(before), _held(after), strict=True):
        done = done & table[type(current)].applied(previous, current)
    return done


def waiting(opt_state: optax.OptState) -> jax.Array | None:
    """The tokens a fresh run's state holds waiting for an update the optimizer whose state is
    `opt_state` holds back (`TrainState.deferred`): none yet, two uint32 words; None for an optimizer that
    applies every update."""
    return jnp.zeros((2,), jnp.uint32) if holds_back(opt_state) else None


def held_places(opt_state: optax.OptState) -> list[jax.Array]:
    """Where among the calls each wrapper in `opt_state` holds back the next call stands (`_Hold.place`),
    which its draws fold in: the calls of one update draw apart, as the slots of the trainer's own window
    do."""
    table = _holds()
    return [table[type(node)].place(node) for node in _held(opt_state)]


def refuse_untold(opt_state: optax.OptState) -> None:
    """Refuse an optimizer, by its state `opt_state`, that holds updates back where its state does not
    say which call applied one: `optax.conditionally_mask`, whose condition its state does not hold; a
    wrapper that holds updates back under a mask or a partition (`optax.masked`, `optax.partition`),
    which applies the other parameters' updates every call; and one followed in a chain by a transform
    that keeps state, which moves the weights on the calls it holds."""
    def refused(why: str) -> ValueError:
        return ValueError(f"the optimizer holds updates back where its state does not say which call applied "
                          f"one ({why}); hold them back outermost with optax.MultiSteps, or accumulate "
                          "gradients with Trainer(accumulation=k)")

    def visit(node, partial: bool) -> None:
        if isinstance(node, optax.ConditionallyMaskState):
            raise refused("optax.conditionally_mask")
        if isinstance(node, tuple(_holds())) and partial:
            raise refused("a mask or partition over it")
        partial = partial or isinstance(node, (optax.MaskedState, optax.PartitionState))
        members = list(node.values()) if isinstance(node, Mapping) else (
            list(node) if isinstance(node, (tuple, list)) else [])
        held = False
        for member in members:
            if held and jax.tree.leaves(member):
                raise refused("a transform after it that keeps state")
            visit(member, partial)
            # Only a chain's members follow one another; a wrapper's own fields are its state.
            held = held or (type(node) in (tuple, list, dict) and holds_back(member))

    visit(opt_state, partial=False)
