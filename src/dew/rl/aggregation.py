"""A student's own steps labelled by a privileged teacher and aggregated into what the student learns from:
DAgger (Ross, Gordon and Bagnell, arXiv 1011.0686), for any teacher policy and any student policy over one
environment.

A student cloned from a teacher's flights alone learns the teacher's actions at the teacher's states, and its
own errors take it to states those flights never reached, where its errors grow. Here the student acts, so its
data holds the states its own errors lead to, and at each of them the teacher, reading what the student cannot
(a privileged view of the true state), answers the action it would take there: the step's label (`labelled`).
A rollout given a `Teacher` writes each label beside its step into the data it aggregates
(`dew.objectives.rl.dreamer.DreamerRollout`'s replay, which also holds the teacher's own flights it was
given), and an objective imitates the label in place of the action taken
(`DreamerV3Objective(imitates=LABEL)`), while what the student learns of the environment, a world model,
reads the actions it took.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import jax

from dew.rl.env import Memory, Observation, Params, Policy, Tree

LABEL = "label"
"""The field a labelled step's teacher action is written as (`labelled`), which an objective imitates."""

_LABELLING = 0x4C41424C
"""The fold of a step's key the teacher draws from ('LABL'), apart from the student's."""

_STUDENT, _TEACHER = "student", "teacher"


@dataclass(frozen=True)
class Teacher:
    """A privileged teacher: `policy`, a `dew.rl.env.Policy` over the step a student is given, reading its
    own views of it, and its `variables`. It carries no memory: the view it reads holds the state."""

    policy: Policy
    variables: Params


def labelled(student: Policy, teacher: Policy) -> Policy:
    """A policy over a student's and a teacher's variables (`paired`): `student` acting from the step and
    its memory, its extras, a dict, with `teacher`'s action at the same step beside them under `LABEL`.
    The teacher draws from a key of its own, folded from the step's."""

    def policy(params: Params, observed: Observation, key: jax.Array,
               memory: Memory) -> tuple[jax.Array, Tree, Memory]:
        action, extras, memory = student(params[_STUDENT], observed, key, memory)
        if not isinstance(extras, Mapping):
            raise TypeError("a labelled student's extras are a dict, beside which its label is written")
        label, _, _ = teacher(params[_TEACHER], observed, jax.random.fold_in(key, _LABELLING), None)
        return action, {**extras, LABEL: label}, memory

    return policy


def paired(student: Params, teacher: Params) -> Params:
    """The variables a `labelled` policy reads: the student's and the teacher's."""
    return {_STUDENT: student, _TEACHER: teacher}


__all__ = ["LABEL", "Teacher", "labelled", "paired"]
