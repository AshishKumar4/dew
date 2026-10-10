"""Knowledge distillation from a frozen teacher, as MaxText 0.2.4 trains it.

A student objective scores a batch the way it always does, and a frozen
teacher scores the same batch. The loss mixes the student's own loss with
the KL between the two temperature-softened token distributions and, over
named layer pairs, with a distance between their hidden states:

    (1 - alpha) * student + alpha * T^2 * KL(teacher_T || student_T) + beta * feature

Every term is summed over the positions the student counts, with the
student's weights, and divided by the student's mass. The feature term
averages the pairs' per-position cosine distance or mean squared error.
alpha, T and beta are each a constant or an `optax` schedule of the step the
trainer passes the objective. MaxText's linear and cosine anneals are
`optax.linear_schedule` and
`optax.cosine_decay_schedule(start, steps, alpha=end / start)`.

The teacher's variables are in the tree's `teacher` collection beside the
student's, so they reach the compiled step as arguments, and the optimizer,
the EMA and the student's published weights never see them. A layer pair
whose widths differ gets a trainable `[student, teacher]` projection under
`params/distillation`, which projects the student onto the teacher.

Any objective that scores token logits can be the student or the teacher
through `Objective.predict`; `LMObjective` does. Both score the same batch,
so they must share a tokenizer, and a vocabulary mismatch is refused.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Generic, Literal

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from typing_extensions import TypeVar

from dew.artifacts import Artifacts
from dew.objectives.base import (
    OMITTED,
    TEACHER,
    Aux,
    Batch,
    EMASpec,
    Objective,
    Omitted,
    Prediction,
    ProgramModule,
    Ratio,
    Step,
    Variables,
)

Loss = TypeVar("Loss", default=Ratio | jax.Array | float)
Effects = TypeVar("Effects", default=None)

if TYPE_CHECKING:
    from dew.inference.tasks import Processor

PROJECTIONS = "distillation"
"""The params subtree holding the feature pairs' width projections."""

Weight = float | optax.Schedule
FeatureLoss = Literal["cosine", "l2"]


def _schedule(value: Weight) -> optax.Schedule:
    """Wrap a constant as the step-indexed schedule every weight is read as."""
    # MaxText's anneals: distillation_utils.py:165-194.
    return value if callable(value) else optax.constant_schedule(value)


class DistillationObjective(Objective[Ratio, Effects], Generic[Loss, Effects]):
    """Mixes the student's own loss with a frozen teacher's soft targets."""

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
        """Build a distillation of `teacher` into `student` over one batch.

        `alpha` weights the KL against the student's own loss (MaxText's
        `distill_alpha`). `temperature` softens both distributions before the
        KL and scales the KL by its square (`distill_temperature`).

        `features` names `(teacher layer, student layer)` pairs whose output
        states the feature term compares. `beta` weights that term
        (`distill_beta`), and `feature_loss` picks the distance
        (`distill_feature_loss_type`): the cosine distance with MaxText's
        1e-6 norm floor, or the squared error averaged over the width. With
        no pairs the term is left out, and a nonzero `beta` with no pairs
        raises `ValueError`.

        Each of `alpha`, `temperature` and `beta` is a constant or an
        `optax.Schedule` of the step. A constant outside its range (alpha in
        [0, 1], temperature above 0, beta at least 0) raises `ValueError`.
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

    def program_key(self) -> tuple[ProgramModule, ...]:
        """The student's modules, then the teacher's, which the step does not train."""
        return (*self.student.program_key(),
                *(program._replace(trained=False) for program in self.teacher.program_key()))

    def substitute(self, modules: Sequence[nn.Module]) -> None:
        count = len(self.student.program_key())
        self.student.substitute(modules[:count])
        self.teacher.substitute(modules[count:])

    def held_variables(self) -> Variables | None:
        """Return the student's held tree, if any, with the teacher's under `teacher`."""
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
        inputs = self.inputs
        if inputs is None:
            raise ValueError("feature distillation requires the student's InputSpec")
        probe = {inputs.sample.key: jnp.zeros((1, *inputs.sample.shape), jnp.int32)}
        _, _, student, teacher = jax.eval_shape(
            self._predictions, tree, probe, Step(jnp.zeros((), jnp.int32), key, None))
        projections = dict(tree["params"].get(PROJECTIONS, {}))
        for index, (own, other) in enumerate(zip(student.hidden, teacher.hidden, strict=True)):
            name = f"projection_{index}"
            if own.shape[-1] != other.shape[-1] and name not in projections:
                projections[name] = nn.initializers.lecun_normal()(
                    jax.random.fold_in(projection_key, index),
                    (own.shape[-1], other.shape[-1]), jnp.float32)
        if projections:
            tree["params"] = {**tree["params"], PROJECTIONS: projections}
        return tree

    def student_variables(self, params: Variables) -> Variables:
        """Return the student's own tree, without the teacher or the feature projections.

        The student's methods read this tree, and a distilled checkpoint
        passes it on to a plain student run.
        """
        own = {name: value for name, value in params.items() if name != TEACHER}
        if PROJECTIONS in own["params"]:
            own["params"] = {name: value for name, value in own["params"].items() if name != PROJECTIONS}
        return own

    def _student_step(self, step: Step) -> Step:
        return replace(step, ema=None if step.ema is None else self.student_variables(step.ema))

    def _predictions(self, params: Variables, batch: Batch, step: Step
                     ) -> tuple[Ratio, Aux[Effects], Prediction, Prediction]:
        """Score the student in the step's mode and the frozen teacher in evaluation mode."""
        statistics, aux, student = self.student.predict(
            self.student_variables(params), batch, self._student_step(step), train=step.training,
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
        # MaxText's distillation_utils.py:418-436.
        student, teacher = student.astype(jnp.float32), teacher.astype(jnp.float32)
        if self.feature_loss == "cosine":
            return optax.cosine_distance(student, teacher, epsilon=1e-6)
        return jnp.mean(jnp.square(student - teacher), axis=-1)

    def loss(self, variables: Variables, batch: Batch, step: Step) -> tuple[Ratio, Aux[Effects]]:
        statistics, aux, student, teacher = self._predictions(variables, batch, step)
        alpha, temperature, beta = (jnp.asarray(schedule(step.step), jnp.float32) for schedule in
                                    (self.alpha, self.temperature, self.beta))
        # Every term is summed with the student's weights and divided by its
        # mass, the teacher's reported loss included (MaxText's
        # distillation_utils.py:492-547 and :562-566).
        weights = student.weights.astype(jnp.float32)
        mass = statistics.mass
        counted = jnp.where(mass > 0, mass, 1)
        divergence = optax.kl_divergence(jax.nn.log_softmax(student.logits / temperature),
                                         jax.nn.softmax(teacher.logits / temperature))
        soft = self.row_mean(divergence * weights, batch).total
        total = (1 - alpha) * statistics.total + alpha * temperature ** 2 * soft
        reported = {**aux.metrics, "distill/alpha": alpha, "distill/temperature": temperature,
                    "distill/kl": soft / counted, "distill/soft_loss": temperature ** 2 * soft / counted,
                    "distill/teacher_loss": self.row_mean(teacher.losses * weights, batch).total / counted}
        if self.features:
            projections = variables["params"].get(PROJECTIONS, {})
            distances = []
            for index, (own, other) in enumerate(zip(student.hidden, teacher.hidden, strict=True)):
                if own.shape[-1] != other.shape[-1]:
                    own = own.astype(jnp.float32) @ projections[f"projection_{index}"]
                distances.append(self._distance(own, other))
            feature = self.row_mean(jnp.stack(distances) * weights, batch, rows=1).total / len(self.features)
            total = total + beta * feature
            reported.update({"distill/beta": beta, "distill/feature": feature / counted})
        return Ratio(total, mass), replace(aux, metrics=reported)

    def apply_effects(self, variables: Variables, effects: Effects) -> Variables:
        return self.student.apply_effects(self.student_variables(variables), effects)

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> Artifacts | None:
        return self.student.evaluate(self.student_variables(params), batch, self._student_step(step))

    def preview(self, params: Variables, batch: Batch, step: Step, *,
                scored: Artifacts | None = None) -> Artifacts | None:
        return self.student.preview(self.student_variables(params), batch, self._student_step(step),
                                    scored=scored)

    def build_task(self, variables: Variables, *, processor: Processor | None | Omitted = OMITTED):
        """Return the student's task, without the teacher."""
        return self.student.build_task(self.student_variables(variables), processor=processor)
