"""Objectives whose loss is a mean square of residuals, stepped by the gradient or by Gauss-Newton.

A physics-informed network's loss is its equation's residuals at its collocation points, and a
least-squares fit's is its errors: `Residuals` takes each batch row's residuals from its subclass and
trains on their mean square. A run whose optimizer is Gauss-Newton's (`dew.gauss_newton.gauss_newton`,
damped as Levenberg-Marquardt's) asks it for its residuals' system (`Step.equations`): each microbatch
states its part of the update's as statistics that pool by adding, which the trainer sums as it sums any
statistic and hands the optimizer whole, J_i^T J_i and J_i^T r_i where the update's residuals are at least
its parameters (`NormalEquations`), its rows of J and r placed at its examples' places among the update's
where they are fewer (`RowEquations`). Which form an update takes is its geometry, the update's examples
(`Step.example_count`) against the parameters, which the trainer gives the loss as it plans the step's
shapes. The loss is the mean square still, its gradient its own; the optimizer steps by the solve.
"""

from __future__ import annotations

from abc import abstractmethod

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from dew.batch import Batch, Variables
from dew.gauss_newton import NormalEquations, RowEquations
from dew.nn.precision import float32_einsum
from dew.objectives.base import Objective, Ratio, Step


class Residuals(Objective[Ratio | NormalEquations | RowEquations]):
    """An objective whose loss is the mean over its batch's rows of the sum of each row's squared
    residuals (`residuals`), which states its residuals' Gauss-Newton system where its step asks for it
    (`Step.equations`, the module's docstring)."""

    @abstractmethod
    def residuals(self, variables: Variables, batch: Batch, step: Step) -> jax.Array:
        """Each row's residuals, [B, ...], every entry of a row a residual its loss squares."""

    def loss(self, variables: Variables, batch: Batch,
             step: Step) -> Ratio | NormalEquations | RowEquations:
        residuals = self.residuals(variables, batch, step)
        stats = self.row_mean(jnp.sum(jnp.square(residuals.reshape(residuals.shape[0], -1)), -1), batch)
        if not step.equations or not step.training:
            return stats
        jacobian, residual = self._linearized(variables, batch, step)
        rows, size = residuals.shape[0], jacobian.shape[1]
        per_row = jacobian.shape[0] // rows
        # The system is the real rows' alone, as the loss counts them (`row_weights`): a row the batch marks
        # unreal (an exhausted share's, row 3929) is selected to zero, its residuals and their Jacobian.
        real = jnp.repeat(self.row_weights(batch, rows) > 0, per_row)
        jacobian = jnp.where(real[:, None], jacobian, jnp.zeros((), jacobian.dtype))
        residual = jnp.where(real, residual, jnp.zeros((), residual.dtype))
        # The update's examples, of which this microbatch's rows are a share (the trainer's geometry).
        update = rows if step.example_count is None else step.example_count
        held = jnp.asarray(rows, jnp.float32)
        if size <= update * per_row:
            pulled = float32_einsum("rp,r->p", jacobian, residual)
            normal = float32_einsum("rp,rq->pq", jacobian, jacobian)
            return NormalEquations(stats.total, stats.mass, held, pulled, normal, update)
        # Each row's residuals at its example's place among the update's.
        places = (step.examples(rows)[:, None] * per_row + jnp.arange(per_row)).reshape(-1)
        placed_jacobian = jnp.zeros((update * per_row, size), jacobian.dtype).at[places].set(jacobian)
        placed_residual = jnp.zeros((update * per_row,), residual.dtype).at[places].set(residual)
        return RowEquations(stats.total, stats.mass, held, placed_jacobian, placed_residual, update)

    def reduce_loss(self, stats: Ratio | NormalEquations | RowEquations) -> tuple[jax.Array, jax.Array]:
        """The mean squared residual, whatever system the statistics state beside it."""
        return super().reduce_loss(stats if isinstance(stats, Ratio) else Ratio(stats.total, stats.mass))

    def _linearized(self, variables: Variables, batch: Batch, step: Step) -> tuple[jax.Array, jax.Array]:
        """The microbatch's residuals' Jacobian in the trained parameters, flat, and the residuals,
        neither differentiated."""
        flat, unravel = ravel_pytree(jax.lax.stop_gradient(variables["params"]))
        held = jax.lax.stop_gradient(variables)

        def residual_vector(point):
            return self.residuals({**held, "params": unravel(point)}, batch, step).reshape(-1)

        rows = jax.eval_shape(residual_vector, flat).shape[0]
        # Forward mode where the parameters are fewer than the residuals, reverse mode where they are more.
        jacobian = (jax.jacfwd if flat.size <= rows else jax.jacrev)(residual_vector)(flat)
        return jacobian, residual_vector(flat)
