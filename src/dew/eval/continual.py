"""Continual learning measured: a model trained on a sequence of tasks, scored on every task after each.

A run of `n` tasks gives an accuracy matrix `A`, `A[i][j]` the accuracy on task `j`'s test set after
training on tasks 0 to `i` (`AccuracyMatrix`). Its summaries are the field's (Chaudhry et al. 2018,
arXiv 1801.10112; Lopez-Paz and Ranzato 2017, arXiv 1706.08840): the average accuracy after the last task,
each earlier task's forgetting, the most it lost from its best, `max_{i < n-1} A[i][j] - A[n-1][j]`, and
the backward transfer `mean_j (A[n-1][j] - A[j][j])`, negative where training on later tasks forgot.

Two settings differ in what a task's test chooses among. Sequential (task-incremental) learning scores
each example among its own task's classes; class-incremental learning among every class seen so far, the
setting Nested Learning's continual-learning runs measure (arXiv 2512.24695, section 9.1). A language
model chooses by the log-likelihood it gives each class's label after the prompt (`label_scores`): its
objective scores the rows, so a model that steps levels in context (`LMObjective(adaptation=...)`)
chooses through its steps, and the same model without them (`adaptation=None`) is its baseline.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class AccuracyMatrix:
    """A continual run's accuracies (module docstring): `values[i][j]`, task `j`'s after training on tasks
    0 to `i`, in [0, 1], one row a stage and one column a task, as many of each."""

    values: np.ndarray

    def __post_init__(self):
        values = np.asarray(self.values, np.float64)
        if values.ndim != 2 or values.shape[0] != values.shape[1] or not values.size:
            raise ValueError(f"an accuracy matrix holds one row a stage and one column a task, as many of "
                             f"each, not {values.shape}")
        if not np.all((values >= 0) & (values <= 1)):
            raise ValueError("an accuracy is in [0, 1]")
        object.__setattr__(self, "values", values)

    @property
    def tasks(self) -> int:
        return self.values.shape[0]

    def average(self) -> float:
        """The average accuracy over every task after the last."""
        return float(np.mean(self.values[-1]))

    def forgetting(self) -> np.ndarray:
        """Each earlier task's forgetting, the most it lost from its best before the last stage:
        `max_{i < n-1} A[i][j] - A[n-1][j]`, `[n - 1]`; none after one task."""
        if self.tasks == 1:
            return np.zeros((0,))
        return np.max(self.values[:-1, :-1], axis=0) - self.values[-1, :-1]

    def average_forgetting(self) -> float:
        """The mean of every earlier task's forgetting, 0 after one task."""
        forgetting = self.forgetting()
        return float(np.mean(forgetting)) if forgetting.size else 0.0

    def backward_transfer(self) -> float:
        """`mean_j (A[n-1][j] - A[j][j])` over every earlier task: negative where later tasks forgot it."""
        if self.tasks == 1:
            return 0.0
        return float(np.mean(self.values[-1, :-1] - np.diag(self.values)[:-1]))

    def record(self) -> dict[str, float]:
        """The run's summaries as a log reads them."""
        return {"continual/average": self.average(), "continual/forgetting": self.average_forgetting(),
                "continual/backward_transfer": self.backward_transfer()}


def continual_run[Task](tasks: Sequence[Task], fit: Callable[[int, Task], None],
                        accuracy: Callable[[int, int], float]) -> AccuracyMatrix:
    """Train on `tasks` in order and score every task after each: `fit(i, task)` trains on task `i`, and
    `accuracy(i, j)` scores task `j` after it, in [0, 1]. A class-incremental `accuracy` chooses among
    every class of tasks 0 to `i`, a task-incremental one among task `j`'s own (module docstring)."""
    values = np.zeros((len(tasks), len(tasks)))
    for stage, task in enumerate(tasks):
        fit(stage, task)
        for scored in range(len(tasks)):
            values[stage, scored] = accuracy(stage, scored)
    return AccuracyMatrix(values)


def label_scores(token_scores: Callable, prompts: Sequence[Sequence[int]], labels: Sequence[Sequence[int]],
                 pad_id: int = 0) -> jax.Array:
    """The log-likelihood a language model gives each label after each prompt, `[prompts, labels]`: each
    pair one row, the prompt then the label, padded after to the longest row, whose next-token losses
    `token_scores(rows)` gives (`LMObjective.token_scores(variables, rows).losses`, `[B, S]`, the row's
    target `t + 1` at `t`), summed over the label's tokens alone. Padding after a row reads nothing a
    label's token reads. The rows are laid out on the host, so the scores trace and compile as the
    objective's own program does. A prompt holds at least one token, from which its labels' first is
    scored (row 3941: an empty one's was left unscored)."""
    pairs = [(prompt, label) for prompt in prompts for label in labels]
    if not pairs or any(not label for label in labels):
        raise ValueError("label scores read at least one prompt and one label of at least one token")
    if any(not prompt for prompt in prompts):
        # A label's first token is scored from the token before it, which an empty prompt has none of.
        raise ValueError("a label's first token is scored from the prompt's last, so every prompt holds at "
                         "least one token: give an empty prompt the tokenizer's beginning-of-sequence token")
    width = max(len(prompt) + len(label) for prompt, label in pairs)
    rows = np.full((len(pairs), width), pad_id, np.int32)
    scored = np.zeros((len(pairs), width - 1), bool)
    for index, (prompt, label) in enumerate(pairs):
        rows[index, :len(prompt) + len(label)] = [*prompt, *label]
        # The loss at t scores token t + 1: a label's tokens are the targets at its prompt's last onward.
        scored[index, len(prompt) - 1:len(prompt) + len(label) - 1] = True
    losses = jnp.asarray(token_scores(jnp.asarray(rows)))
    return -jnp.sum(jnp.where(scored, losses, 0), axis=1).reshape(len(prompts), len(labels))


def label_accuracy(scores, answers: Sequence[int]) -> float:
    """The share of prompts whose highest-scoring label (`label_scores`) is their answer, by index."""
    held, chosen = np.asarray(scores), np.asarray(answers)
    if held.shape[0] != chosen.shape[0]:
        raise ValueError(f"one answer a prompt: {held.shape[0]} prompts, {chosen.shape[0]} answers")
    return float(np.mean(np.argmax(held, axis=1) == chosen))
