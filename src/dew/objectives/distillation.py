"""Knowledge distillation from a frozen teacher, as MaxText 0.2.4 trains it.

A student objective scores a batch the way it always does, a frozen teacher
scores the same batch, and the loss mixes the student's own with the KL
between the two temperature-softened token distributions and, over named
layer pairs, a distance between their hidden states:

    (1 - alpha) * student + alpha * T^2 * KL(teacher_T || student_T) + beta * feature

every term summed over the positions the student counts, with the student's
weights, and divided by the student's mass (`distillation_utils.py:492-547`,
`:562-566`). The feature term averages the pairs' per-position cosine
distance or mean squared error (`:418-436`). alpha, T and beta are
`optax` schedules of the step the trainer hands the objective, or constants;
MaxText's linear and cosine anneals (`:165-194`) are `optax.linear_schedule`
and `optax.cosine_decay_schedule(start, steps, alpha=end / start)`.

The teacher's variables live in the tree's `teacher` collection beside the
student's, so they reach the compiled step as arguments and the optimizer,
the EMA and the student's published weights never see them. A layer pair
whose widths differ gets a trainable `[student, teacher]` projection under
`params/distillation`; the student is projected onto the teacher.

Any objective that scores token logits serves as student or teacher through
`Objective.predict`; `LMObjective` does. Both score the same batch, so they
share a tokenizer, and a vocabulary mismatch is refused.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Generic, Literal

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from dew.artifacts import Artifacts
from dew.objectives.base import (
    Aux,
    Batch,
    Effects,
    EMASpec,
    Loss,
    Mean,
    Objective,
    Prediction,
    Step,
    Variables,
)
from dew.registry import objectives

if TYPE_CHECKING:
    from dew.training.state import TrainState

TEACHER = "teacher"
"""The collection holding the teacher's whole variables tree."""
PROJECTIONS = "distillation"
"""The params subtree holding the feature pairs' width projections."""

Weight = float | optax.Schedule
FeatureLoss = Literal["cosine", "l2"]


def _schedule(value: Weight) -> optax.Schedule:
    """A constant as the step-indexed schedule every weight is read as."""
    return value if callable(value) else optax.constant_schedule(value)


@objectives("distillation")
class DistillationObjective(Objective[Mean, Effects], Generic[Loss, Effects]):
    """The student's loss mixed with a frozen teacher's soft targets."""

    def __init__(
        self,
        student: Objective[Loss, Effects],
        teacher: Objective[Loss, Effects],
        *,
        alpha: Weight = 0.5,
        temperature: Weight = 1.0,
        features: Sequence[tuple[int, int]] = (),
        beta: Weight = 0.0,
        feature_loss: FeatureLoss = "cosine",
    ):
        """`alpha` weights the KL against the student's own loss (MaxText's
        `distill_alpha`), `temperature` softens both distributions before
        the KL and scales it by its square (`distill_temperature`).

        `features` names `(teacher layer, student layer)` pairs whose output
        states the feature term compares, `beta` weights it
        (`distill_beta`) and `feature_loss` picks the distance
        (`distill_feature_loss_type`): the cosine distance with MaxText's
        1e-6 norm floor, or the squared error averaged over the width. No
        pairs leaves the term out; a `beta` with no pairs is refused.

        Each of the three is a constant or an `optax.Schedule` of the step.
        """
        for name, value, ok in (("alpha", alpha, lambda v: 0 <= v <= 1),
                                ("temperature", temperature, lambda v: v > 0),
                                ("beta", beta, lambda v: v >= 0)):
            if not callable(value) and not ok(value):
                raise ValueError(f"{name}={value} is outside its range: alpha in [0, 1], "
                                 "temperature above 0, beta at least 0")
        self.student = student
        self.teacher = teacher
        self.alpha, self.temperature, self.beta = _schedule(alpha), _schedule(temperature), _schedule(beta)
        self.features = tuple((teacher_layer, student_layer) for teacher_layer, student_layer in features)
        if not self.features and beta:
            raise ValueError(
                "beta weights the feature term over the (teacher, student) layer "
                "pairs in features, and none were named")
        self.feature_loss: FeatureLoss = feature_loss
        self.inputs = student.inputs
        self.artifact = student.artifact
        averaged = student.ema
        # The EMA follows what moves; the teacher never does.
        self.ema = None if averaged is None else EMASpec(
            decay=averaged.decay,
            select=lambda path: path[0] != TEACHER and averaged.select(path))

    def held_variables(self) -> Variables | None:
        """The student's held tree, if any, with the teacher's under `teacher`."""
        held = dict(self.student.held_variables() or {})
        teacher = self.teacher.held_variables()
        if teacher is not None:
            held[TEACHER] = teacher
        return held or None

    def init(self, key: jax.Array, variables: Variables | None = None) -> Variables:
        held = dict((self.held_variables() if variables is None else variables) or {})
        teacher_held = held.pop(TEACHER, None)
        student_key, teacher_key, projection_key = jax.random.split(key, 3)
        tree = {**self.student.init(student_key, held or None),
                TEACHER: self.teacher.init(teacher_key, teacher_held)}
        if not self.features:
            return tree
        # A pair's widths decide whether it needs a projection, which only
        # a forward pass can say; the shapes come from an abstract one.
        probe = {self.inputs.sample.key: jnp.zeros((1, *self.inputs.sample.shape), jnp.int32)}
        _, _, student, teacher = jax.eval_shape(
            self._predictions, tree, probe, Step(jnp.zeros((), jnp.int32), key, None))
        projections = {}
        for index, (own, other) in enumerate(zip(student.hidden, teacher.hidden, strict=True)):
            if own.shape[-1] != other.shape[-1]:
                projections[f"projection_{index}"] = nn.initializers.lecun_normal()(
                    jax.random.fold_in(projection_key, index),
                    (own.shape[-1], other.shape[-1]), jnp.float32)
        if projections:
            tree["params"] = {**tree["params"], PROJECTIONS: projections}
        return tree

    def student_variables(self, params: Variables) -> Variables:
        """The student's own tree out of the whole: what its methods read,
        and what a distilled checkpoint hands on to a plain student run."""
        own = {name: value for name, value in params.items() if name != TEACHER}
        if PROJECTIONS in own["params"]:
            own["params"] = {name: value for name, value in own["params"].items() if name != PROJECTIONS}
        return own

    def _student_step(self, step: Step) -> Step:
        return replace(step, ema=None if step.ema is None else self.student_variables(step.ema))

    def _predictions(self, params: Variables, batch: Batch, step: Step
                     ) -> tuple[Mean, Aux[Effects], Prediction, Prediction]:
        """Both sides over the batch: the student training, the teacher not."""
        statistics, aux, student = self.student.predict(
            self.student_variables(params), batch, self._student_step(step), train=True,
            layers=tuple(student_layer for _, student_layer in self.features))
        _, _, teacher = self.teacher.predict(
            params[TEACHER], batch, replace(step, ema=None), train=False,
            layers=tuple(teacher_layer for teacher_layer, _ in self.features))
        if student.logits.shape[-1] != teacher.logits.shape[-1]:
            raise ValueError(
                f"the KL compares the two distributions column by column, so the "
                f"teacher and the student share a tokenizer and a vocabulary; the "
                f"student scores {student.logits.shape[-1]} ids and the teacher "
                f"{teacher.logits.shape[-1]}")
        return statistics, aux, student, teacher

    def _distance(self, student: jax.Array, teacher: jax.Array) -> jax.Array:
        """The pair's per-position feature distance, `[B, S]`, in fp32."""
        student, teacher = student.astype(jnp.float32), teacher.astype(jnp.float32)
        if self.feature_loss == "cosine":
            return optax.cosine_distance(student, teacher, epsilon=1e-6)
        return jnp.mean(jnp.square(student - teacher), axis=-1)

    def loss(self, params: Variables, batch: Batch, step: Step) -> tuple[Mean, Aux[Effects]]:
        statistics, aux, student, teacher = self._predictions(params, batch, step)
        alpha, temperature, beta = (jnp.asarray(schedule(step.step), jnp.float32) for schedule in
                                    (self.alpha, self.temperature, self.beta))
        # Every term is summed with the student's weights and divided by its
        # mass, the teacher's reported loss included.
        weights = student.weights.astype(jnp.float32)
        mass = statistics.mass
        counted = jnp.where(mass > 0, mass, 1)
        divergence = optax.kl_divergence(jax.nn.log_softmax(student.logits / temperature),
                                         jax.nn.softmax(teacher.logits / temperature))
        soft = jnp.sum(divergence * weights)
        total = (1 - alpha) * statistics.total + alpha * temperature ** 2 * soft
        reported = {**aux.metrics, "distill/alpha": alpha, "distill/temperature": temperature,
                    "distill/kl": soft / counted, "distill/soft_loss": temperature ** 2 * soft / counted,
                    "distill/teacher_loss": jnp.sum(teacher.losses * weights) / counted}
        if self.features:
            projections = params["params"].get(PROJECTIONS, {})
            distances = []
            for index, (own, other) in enumerate(zip(student.hidden, teacher.hidden, strict=True)):
                if own.shape[-1] != other.shape[-1]:
                    own = own.astype(jnp.float32) @ projections[f"projection_{index}"]
                distances.append(self._distance(own, other))
            feature = jnp.sum(jnp.stack(distances) * weights) / len(self.features)
            total = total + beta * feature
            reported.update({"distill/beta": beta, "distill/feature": feature / counted})
        return Mean(total, mass), replace(aux, metrics=reported)

    def apply_effects(self, variables: Variables, effects: Effects) -> Variables:
        return self.student.apply_effects(self.student_variables(variables), effects)

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> Artifacts | None:
        return self.student.evaluate(self.student_variables(params), batch, self._student_step(step))

    def preview(self, params: Variables, batch: Batch, step: Step, *,
                scored: Artifacts | None = None) -> Artifacts | None:
        return self.student.preview(self.student_variables(params), batch, self._student_step(step),
                                    scored=scored)

    def pipeline(self, state: TrainState, *, ema: bool = True):
        """The student as its inference task; the teacher stays behind."""
        return self.student.pipeline(
            replace(state, params=self.student_variables(state.params),
                    ema=None if state.ema is None else self.student_variables(state.ema)),
            ema=ema)
