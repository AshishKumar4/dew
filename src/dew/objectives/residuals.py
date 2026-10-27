"""Objectives whose loss is a mean square of residuals, stepped by the gradient or by Gauss-Newton.

A physics-informed network's loss is its equation's residuals at its collocation points, and a
least-squares fit's is its errors: `Residuals` takes each batch row's residuals from its subclass and
trains on their mean square. Where `damping` is set, its update is Gauss-Newton's, damped as
Levenberg-Marquardt's: with J the Jacobian of an update's residuals r in the trained parameters,

    delta = (J^T J + damping I)^-1 J^T r,

one solve an optimizer update, whatever microbatches the trainer reads the update's batch in. Each
microbatch states its part of the update's system as additive statistics, which the trainer pools by
summing as it pools any statistic, and the pooled reduction's derivative in them carries the damped solve,
Cholesky's, back to each microbatch, whose gradient is its own part of the one solve: the update's
gradient is the damped step of its whole batch, which `optax.sgd(1.0)` takes whole and any optimizer
composes with. A window of microbatches recomputes each microbatch's Jacobian under the pooled reduction,
as the trainer replays any statistics it pools by summing.

Where a microbatch's residuals are at least the parameters, the statistics are its normal equations,
J^T J of its rows (`NormalEquations`), the parameters' square. Where they are fewer, the step is the
dual's, J^T (J J^T + damping I)^-1 r, the same step from the rows' Gram matrix (`RowEquations`), which
an update read in one microbatch takes; an update pooled from several is refused there, its Gram matrix
no sum of its microbatches'. Either is formed in the parameters' dtype on each device, so the step fits
a model whose smaller side squared fits there: a PINN's, not a language model's. A float64 run (x64 set
for the process, `jax.config.update("jax_enable_x64", True)`, which its data's worker threads read too,
and float64 parameters) solves in float64, where Gauss-Newton reaches residuals near float64's roundoff
that first-order steps do not.
"""

from __future__ import annotations

import functools
from abc import abstractmethod

import jax
import jax.numpy as jnp
from flax import struct
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import cho_factor, cho_solve

from dew.batch import Batch, Variables
from dew.nn.precision import float32_einsum
from dew.objectives.base import Objective, Ratio, Step


@struct.dataclass
class NormalEquations:
    """A microbatch's Gauss-Newton statistics, each a sum over its rows that pools by adding: the squared
    residuals' `total` and the rows' `mass`, which the loss reports, and `normal`, J^T J, whose
    derivative in the parameters carries the rows' J^T r (`_normal`)."""

    total: jax.Array
    mass: jax.Array
    normal: jax.Array


@jax.custom_vjp
def _normal(params: Variables, normal: jax.Array, pulled: jax.Array) -> jax.Array:
    """`normal`, whose pullback takes a cotangent C to the parameters' C J^T r, `pulled`, in their tree."""
    return normal


def _normal_fwd(params, normal, pulled):
    return normal, (params, pulled)


def _normal_bwd(residuals, cotangent):
    params, pulled = residuals
    _, unravel = ravel_pytree(params)
    return (unravel(float32_einsum("pq,q->p", cotangent, pulled)), jnp.zeros_like(cotangent),
            jnp.zeros_like(pulled))


_normal.defvjp(_normal_fwd, _normal_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def _solved(normal: jax.Array, damping: float) -> jax.Array:
    """Zero, whose derivative in `normal` is (normal + damping I)^-1: the pooled system's damped solve."""
    return jnp.zeros((), normal.dtype)


def _solved_fwd(normal, damping):
    return jnp.zeros((), normal.dtype), normal


def _solved_bwd(damping, normal, cotangent):
    eye = jnp.eye(normal.shape[0], dtype=normal.dtype)
    inverse = cho_solve(cho_factor(normal + damping * eye), eye)
    return (cotangent * inverse,)


_solved.defvjp(_solved_fwd, _solved_bwd)


@struct.dataclass
class RowEquations:
    """An update's dual Gauss-Newton statistics, read in one microbatch: the squared residuals' `total`
    and the rows' `mass`, and the residuals' `jacobian` and `residual` (`_rows`)."""

    total: jax.Array
    mass: jax.Array
    jacobian: jax.Array
    residual: jax.Array


@jax.custom_vjp
def _rows(params: Variables, jacobian: jax.Array, residual: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`jacobian` and `residual`, whose pullback takes a residual's cotangent c to the parameters' J^T c,
    in their tree."""
    return jacobian, residual


def _rows_fwd(params, jacobian, residual):
    return (jacobian, residual), (params, jacobian)


def _rows_bwd(saved, cotangents):
    params, jacobian = saved
    _, unravel = ravel_pytree(params)
    _, toward = cotangents
    return (unravel(float32_einsum("rp,r->p", jacobian, toward)), jnp.zeros_like(jacobian),
            jnp.zeros_like(toward))


_rows.defvjp(_rows_fwd, _rows_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _dual_solved(jacobian: jax.Array, residual: jax.Array, damping: float) -> jax.Array:
    """Zero, whose derivative in `residual` is (J J^T + damping I)^-1 r: the pooled dual's damped solve."""
    return jnp.zeros((), residual.dtype)


def _dual_fwd(jacobian, residual, damping):
    return jnp.zeros((), residual.dtype), (jacobian, residual)


def _dual_bwd(damping, saved, cotangent):
    jacobian, residual = saved
    gram = float32_einsum("rp,sp->rs", jacobian, jacobian)
    solved = cho_solve(cho_factor(gram + damping * jnp.eye(gram.shape[0], dtype=gram.dtype)), residual)
    return jnp.zeros_like(jacobian), cotangent * solved


_dual_solved.defvjp(_dual_fwd, _dual_bwd)


class Residuals(Objective[Ratio | NormalEquations | RowEquations]):
    """An objective whose loss is the mean over its batch's rows of the sum of each row's squared
    residuals (`residuals`). `damping` None steps by the loss's gradient; a float steps by Gauss-Newton
    with that Levenberg-Marquardt damping, one solve an update (the module's docstring)."""

    damping: float | None = None

    @abstractmethod
    def residuals(self, variables: Variables, batch: Batch, step: Step) -> jax.Array:
        """Each row's residuals, [B, ...], every entry of a row a residual its loss squares."""

    def loss(self, variables: Variables, batch: Batch,
             step: Step) -> Ratio | NormalEquations | RowEquations:
        residuals = self.residuals(variables, batch, step)
        stats = self.row_mean(jnp.sum(jnp.square(residuals.reshape(residuals.shape[0], -1)), -1), batch)
        if self.damping is None or not step.training:
            return stats
        total, mass = jax.lax.stop_gradient(stats.total), jax.lax.stop_gradient(stats.mass)
        jacobian, residual = self._linearized(variables, batch, step)
        if jacobian.shape[1] <= jacobian.shape[0]:
            normal = float32_einsum("rp,rq->pq", jacobian, jacobian)
            pulled = float32_einsum("rp,r->p", jacobian, residual)
            return NormalEquations(total, mass, _normal(variables["params"], normal, pulled))
        rows = residuals.shape[0]
        if step.example_count is not None and step.example_count != rows:
            raise ValueError(
                f"the dual Gauss-Newton step of {jacobian.shape[1]} parameters over {jacobian.shape[0]} "
                f"residuals solves its rows' Gram matrix, which an update of {step.example_count} examples "
                f"read in microbatches of {rows} does not sum to; read each update in one microbatch, or "
                "microbatches of at least as many residuals as parameters, whose normal equations pool")
        return RowEquations(total, mass, *_rows(variables["params"], jacobian, residual))

    def reduce_loss(self, stats: Ratio | NormalEquations | RowEquations) -> tuple[jax.Array, jax.Array]:
        """The mean squared residual; under Gauss-Newton its derivative in the pooled system is the damped
        solve (`_solved`, `_dual_solved`), which leaves its value as it is."""
        if isinstance(stats, Ratio):
            return super().reduce_loss(stats)
        assert self.damping is not None, "Gauss-Newton statistics are stated where a damping is set"
        value, active = Ratio(stats.total, stats.mass).mean()
        solved: jax.Array = (_solved(stats.normal, self.damping) if isinstance(stats, NormalEquations)
                             else _dual_solved(stats.jacobian, stats.residual, self.damping))
        return value + solved.astype(value.dtype), active

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
