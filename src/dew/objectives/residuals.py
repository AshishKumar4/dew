"""Objectives whose loss is a mean square of residuals, stepped by the gradient or by Gauss-Newton.

A physics-informed network's loss is its equation's residuals at its collocation points, and a
least-squares fit's is its errors: `Residuals` takes each batch row's residuals from its subclass and
trains on their mean square. Where `damping` is set, its update is Gauss-Newton's, damped as
Levenberg-Marquardt's: with J the Jacobian of the batch's residuals r in the trained parameters,

    delta = (J^T J + damping I)^-1 J^T r,  or  J^T (J J^T + damping I)^-1 r  where J has fewer rows,

the same step either way, solved by Cholesky in the parameters' dtype. The step is what the loss states as
its gradient (`Objective.with_gradients`), so `optax.sgd(1.0)` takes it whole and any optimizer composes
with it. The Jacobian and its solve are formed whole for each microbatch on each device, so the step fits
a model whose residuals times parameters fit there: a PINN's, not a language model's. A window of
microbatches steps by their directions' mean, each weighted by its rows, as the trainer pools any
gradient. A float64 run (x64 set for the process, `jax.config.update("jax_enable_x64", True)`, which
its data's worker threads read too, and float64 parameters) solves in float64, where Gauss-Newton
reaches residuals near float64's roundoff that first-order steps do not.
"""

from __future__ import annotations

from abc import abstractmethod

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import cho_factor, cho_solve

from dew.batch import Batch, Variables
from dew.nn.precision import float32_einsum
from dew.objectives.base import Objective, Ratio, Step


class Residuals(Objective[Ratio]):
    """An objective whose loss is the mean over its batch's rows of the sum of each row's squared
    residuals (`residuals`). `damping` None steps by the loss's gradient; a float steps by Gauss-Newton
    with that Levenberg-Marquardt damping (the module's docstring)."""

    damping: float | None = None

    @abstractmethod
    def residuals(self, variables: Variables, batch: Batch, step: Step) -> jax.Array:
        """Each row's residuals, [B, ...], every entry of a row a residual its loss squares."""

    def loss(self, variables: Variables, batch: Batch, step: Step) -> Ratio:
        residuals = self.residuals(variables, batch, step)
        stats = self.row_mean(jnp.sum(jnp.square(residuals.reshape(residuals.shape[0], -1)), -1), batch)
        if self.damping is None or not step.training:
            return stats
        # The rule is the total's: the trainer divides it by the mass, as it divides the total's gradient.
        rule = jax.tree.map(lambda leaf: leaf * stats.mass, self._direction(variables, batch, step))
        # The statistics' own structure, the rule at the total and none at the mass, which no parameter moves.
        return self.with_gradients(stats, jax.tree.unflatten(jax.tree.structure(stats), [rule, None]),
                                   variables["params"])

    def _direction(self, variables: Variables, batch: Batch, step: Step):
        """The batch's damped Gauss-Newton step in the trained parameters, in their tree."""
        assert self.damping is not None, "the step is taken where a damping is set"
        flat, unravel = ravel_pytree(variables["params"])

        def residual_vector(point):
            return self.residuals({**variables, "params": unravel(point)}, batch, step).reshape(-1)

        rows = jax.eval_shape(residual_vector, flat).shape[0]
        # Forward mode where the parameters are fewer than the residuals, reverse mode where they are more.
        jacobian = (jax.jacfwd if flat.size <= rows else jax.jacrev)(residual_vector)(flat)
        residual = residual_vector(flat)
        pulled = float32_einsum("rp,r->p", jacobian, residual)
        if flat.size <= rows:
            normal = float32_einsum("rp,rq->pq", jacobian, jacobian)
            direction = cho_solve(cho_factor(normal + self.damping * jnp.eye(flat.size, dtype=normal.dtype)),
                                  pulled)
        else:
            gram = float32_einsum("rp,sp->rs", jacobian, jacobian)
            solved = cho_solve(cho_factor(gram + self.damping * jnp.eye(rows, dtype=gram.dtype)), residual)
            direction = float32_einsum("rp,r->p", jacobian, solved)
        return unravel(direction)
