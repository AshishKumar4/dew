"""Gauss-Newton's update, damped as Levenberg-Marquardt's, as an optimizer: an update's step solves the
system its objective states for the update's residuals r and their Jacobian J in the trained parameters,

    delta = (J^T J + damping I)^-1 J^T r,

one solve an update, whatever microbatches the trainer reads it in. The objective (`dew.objectives.
residuals.Residuals`) states its part of the system as its loss's statistics, which pool by adding as the
trainer pools any statistics: J_i^T J_i and J_i^T r_i where the update's residuals are at least its
parameters (`NormalEquations`), its rows of J and r placed at its examples' places among the update's where
they are fewer, so the pooled ones are the update's whole and the step is the dual's,
J^T (J J^T + damping I)^-1 r (`RowEquations`). A run whose optimizer is `gauss_newton` asks its objective
for them (`dew.objectives.Step.equations`) and hands the update's pooled statistics to the optimizer's
update (`statistics`), as a QK-Clip is handed its logit maxima.

The system is formed in the parameters' dtype on each device, so the step fits a model whose smaller side
of the update, its parameters or its residuals, squared fits there: a PINN's, not a language model's. A
float64 run solves in float64.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import struct
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import cho_factor, cho_solve

from dew.held_updates import holds_state
from dew.nn.precision import float32_einsum


@struct.dataclass
class NormalEquations:
    """A microbatch's part of an update's Gauss-Newton system, each statistic a sum over its rows that
    pools by adding: the squared residuals' `total` and the rows' `mass`, which the loss reports, the
    `rows` it holds of the update's `update` (static), J_i^T r_i (`pulled`) and J_i^T J_i (`normal`)."""

    total: jax.Array
    mass: jax.Array
    rows: jax.Array
    pulled: jax.Array
    normal: jax.Array
    update: int = struct.field(pytree_node=False)

    def solved(self, damping: float) -> jax.Array:
        """The normal equations' damped solve against J^T r, flat."""
        eye = jnp.eye(self.normal.shape[0], dtype=self.normal.dtype)
        return cho_solve(cho_factor(self.normal + damping * eye), self.pulled)


@struct.dataclass
class RowEquations:
    """A microbatch's part of an update's dual Gauss-Newton system: `total`, `mass`, `rows` and `update` as
    `NormalEquations` holds them, and its rows of the update's Jacobian and residuals at its examples'
    places, zero at the rest (`jacobian`, `residual`), so the pooled ones are the update's."""

    total: jax.Array
    mass: jax.Array
    rows: jax.Array
    jacobian: jax.Array
    residual: jax.Array
    update: int = struct.field(pytree_node=False)

    def solved(self, damping: float) -> jax.Array:
        """The dual's damped step, J^T (J J^T + damping I)^-1 r, flat. A row of no Jacobian (a row its
        batch marks unreal, `dew.objectives.Residuals`) is its own equation of the Gram's, of one, which
        the step reads none of (J^T's column of it is zero), so its zero row and column leave the real
        rows' system solvable undamped and its solve theirs alone (row 3929)."""
        gram = float32_einsum("rp,sp->rs", self.jacobian, self.jacobian)
        eye = jnp.eye(gram.shape[0], dtype=gram.dtype)
        unread = jnp.all(self.jacobian == 0, axis=1).astype(gram.dtype)
        return float32_einsum("rp,r->p", self.jacobian, cho_solve(
            cho_factor(gram + damping * eye + jnp.diag(unread)), self.residual))


class GaussNewtonState(NamedTuple):
    """A Gauss-Newton solve's state, which holds nothing: it marks a run whose optimizer solves its
    objective's system (`solves_equations`)."""


def solves_equations(opt_state) -> bool:
    """Whether an optimizer state holds a Gauss-Newton update's (`GaussNewtonState`), however the optimizer
    composes it: a run then asks its objective for its system (`dew.objectives.Step.equations`) and hands
    the update's pooled statistics to the update."""
    return holds_state(opt_state, GaussNewtonState)


def solved_directions(damping: float = 0.0,
                      clip: float | None = None) -> optax.GradientTransformationExtraArgs:
    """The damped Gauss-Newton solve of an update's system (the module's docstring) as a descent direction,
    in the parameters' tree and each in its own dtype, the gradient it is handed aside: the direction a
    rate's transform then scales and negates (`gauss_newton`, or a param group's own rate). An update whose
    statistics hold fewer rows than it has (a run's last, whose stream ended inside its window) moves
    nothing, as a solve of part of the system is no update's. Where `clip` is set, a direction longer
    than it is scaled to that norm (`OptimConfig.clip_grads`, as a clip caps a gradient's). Its objective
    states the system, which `Residuals` does; any other's statistics are refused."""
    if damping < 0:
        raise ValueError(f"a Levenberg-Marquardt damping is at least zero, got {damping}")

    def init_fn(params) -> GaussNewtonState:
        del params
        return GaussNewtonState()

    def update_fn(updates, state, params=None, *, statistics=None, **extra_args):
        del params, extra_args
        if statistics is None:
            raise ValueError("a Gauss-Newton update was handed no statistics: a wrapper that hands its inner "
                             "update no extra arguments (optax.MultiSteps) holds it, which pools no system "
                             "across the steps it holds; accumulate with Trainer(accumulation=k), which "
                             "pools the update's system whole")
        if not isinstance(statistics, NormalEquations | RowEquations):
            raise ValueError("a Gauss-Newton update solves its objective's system (NormalEquations or "
                             "RowEquations, dew.objectives.Step.equations), which this run's statistics are "
                             f"not ({type(statistics).__name__}): train a Residuals objective")
        flat, unravel = ravel_pytree(updates)
        solved = jax.eval_shape(statistics.solved, damping)
        step = jax.lax.cond(statistics.rows >= statistics.update, statistics.solved,
                            lambda _: jnp.zeros(solved.shape, solved.dtype), damping)
        if clip is not None:
            # Optax's own cap of a global norm, which divides by the norm before it scales.
            capped, _ = optax.clip_by_global_norm(clip).update(step, optax.EmptyState())
            step = jnp.asarray(capped)
        return unravel(step.astype(flat.dtype)), state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def gauss_newton(learning_rate: optax.ScalarOrSchedule = 1.0, damping: float = 0.0,
                 clip: float | None = None) -> optax.GradientTransformationExtraArgs:
    """Gauss-Newton's update, damped as Levenberg-Marquardt's: the solve's direction (`solved_directions`)
    scaled by `learning_rate` (a constant, or a schedule over the updates; 1.0 the whole step) as Optax
    scales any optimizer's (`optax.scale_by_learning_rate`), its direction capped at `clip` where set, so
    `OptimConfig` names it as it names any (`optimizer: gauss_newton`, `damping` among its
    `optimizer_opts`, `clip_grads` its `clip`)."""
    return optax.chain(solved_directions(damping, clip), optax.scale_by_learning_rate(learning_rate))


__all__ = ["GaussNewtonState", "NormalEquations", "RowEquations", "gauss_newton", "solved_directions",
           "solves_equations"]
