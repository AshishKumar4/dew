"""Objectives whose loss is a mean square of residuals, stepped by the gradient or by Gauss-Newton.

A physics-informed network's loss is its equation's residuals at its collocation points, and a
least-squares fit's is its errors: `Residuals` takes each batch row's residuals from its subclass and
trains on their mean square. Where `damping` is set, its update is Gauss-Newton's, damped as
Levenberg-Marquardt's: with J the Jacobian of an update's residuals r in the trained parameters,

    delta = (J^T J + damping I)^-1 J^T r,

one solve an optimizer update, whatever microbatches the trainer reads the update's batch in. Each
microbatch states its part of the update's system as statistics that pool by adding, which the trainer
sums as it sums any statistic: J_i^T J_i and J_i^T r_i, whatever the microbatch's shape, where the
update's residuals are at least its parameters (`NormalEquations`); its rows of J and r, placed at its
examples' places among the update's, where they are fewer, so the pooled J and r are the update's whole
and the step is the dual's, J^T (J J^T + damping I)^-1 r (`RowEquations`). Which form an update takes
is its geometry, the update's examples (`Step.example_count`) against the parameters, which the trainer
gives the loss as it plans the step's shapes.

The reduction factors the pooled system once and solves it against its right-hand side alone, once the
statistics hold every row of the update (`rows`), so a microbatch's provisional reduction solves
nothing. The step reaches the parameters through the statistic J_i^T r_i, whose pullback takes the
step's share of each microbatch to their tree (`_pulled`): the update's gradient is the damped step of
its whole batch, which `optax.sgd(1.0)` takes whole and any optimizer composes with.

The system is formed in the parameters' dtype on each device, so the step fits a model whose smaller
side of the update, its parameters or its residuals, squared fits there: a PINN's, not a language
model's. A float64 run (x64 set for the process, `jax.config.update("jax_enable_x64", True)`, which its
data's worker threads read too, and float64 parameters) solves in float64, where Gauss-Newton reaches
residuals near float64's roundoff that first-order steps do not.
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
    """A microbatch's part of an update's Gauss-Newton system, each statistic a sum over its rows that
    pools by adding: the squared residuals' `total` and the rows' `mass`, which the loss reports, the
    `rows` it holds of the update's `update` (static), J_i^T r_i (`pulled`) and J_i^T J_i (`normal`)."""

    total: jax.Array
    mass: jax.Array
    rows: jax.Array
    pulled: jax.Array
    normal: jax.Array
    update: int = struct.field(pytree_node=False)


@struct.dataclass
class RowEquations:
    """A microbatch's part of an update's dual Gauss-Newton system: `total`, `mass`, `rows`, `update` and
    `pulled` as `NormalEquations` holds them, and its rows of the update's Jacobian and residuals at its
    examples' places, zero at the rest (`jacobian`, `residual`), so the pooled ones are the update's."""

    total: jax.Array
    mass: jax.Array
    rows: jax.Array
    pulled: jax.Array
    jacobian: jax.Array
    residual: jax.Array
    update: int = struct.field(pytree_node=False)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _pulled(params: Variables, pulled: jax.Array, share: float) -> jax.Array:
    """`pulled`, a microbatch's J_i^T r_i, whose pullback takes the step's cotangent c to the parameters'
    `share` c, in their tree: the microbatch's rows' share of the update's, so the shares of a whole
    update sum to its step."""
    return pulled


def _pulled_fwd(params, pulled, share):
    return pulled, params


def _pulled_bwd(share, params, cotangent):
    _, unravel = ravel_pytree(params)
    return unravel(cotangent * share), jnp.zeros_like(cotangent)


_pulled.defvjp(_pulled_fwd, _pulled_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def _stepped(damping: float, update: int, rows: jax.Array, pulled: jax.Array,
             system: tuple[jax.Array, ...]) -> jax.Array:
    """Zero, whose derivative in the pooled `pulled` is the update's damped Gauss-Newton step: the normal
    equations' solve against J^T r where `system` is (J^T J,), the dual's J^T (J J^T + damping I)^-1 r
    where it is (J, r), once the statistics hold the update's every row, and zero before."""
    return jnp.zeros((), pulled.dtype)


def _stepped_fwd(damping, update, rows, pulled, system):
    return jnp.zeros((), pulled.dtype), (rows, pulled, system)


def _stepped_bwd(damping, update, saved, cotangent):
    rows, pulled, system = saved

    def solved(_) -> jax.Array:
        if len(system) == 1:
            (normal,) = system
            return cho_solve(cho_factor(normal + damping * jnp.eye(normal.shape[0], dtype=normal.dtype)),
                             pulled)
        jacobian, residual = system
        gram = float32_einsum("rp,sp->rs", jacobian, jacobian)
        dual = cho_solve(cho_factor(gram + damping * jnp.eye(gram.shape[0], dtype=gram.dtype)), residual)
        return float32_einsum("rp,r->p", jacobian, dual)

    # A provisional reduction, of a microbatch the update has more of, steps nothing and solves nothing.
    step = jax.lax.cond(rows >= update, solved, lambda _: jnp.zeros_like(pulled), None)
    return jnp.zeros_like(rows), cotangent * step, jax.tree.map(jnp.zeros_like, system)


_stepped.defvjp(_stepped_fwd, _stepped_bwd)


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
        rows, size = residuals.shape[0], jacobian.shape[1]
        per_row = jacobian.shape[0] // rows
        # The update's examples, of which this microbatch's rows are a share (the trainer's geometry).
        update = rows if step.example_count is None else step.example_count
        held = jnp.asarray(rows, jnp.float32)
        pulled: jax.Array = _pulled(variables["params"], float32_einsum("rp,r->p", jacobian, residual),
                                    rows / update)
        if size <= update * per_row:
            normal = float32_einsum("rp,rq->pq", jacobian, jacobian)
            return NormalEquations(total, mass, held, pulled, normal, update)
        # Each row's residuals at its example's place among the update's.
        places = (step.examples(rows)[:, None] * per_row + jnp.arange(per_row)).reshape(-1)
        placed_jacobian = jnp.zeros((update * per_row, size), jacobian.dtype).at[places].set(jacobian)
        placed_residual = jnp.zeros((update * per_row,), residual.dtype).at[places].set(residual)
        return RowEquations(total, mass, held, pulled, placed_jacobian, placed_residual, update)

    def reduce_loss(self, stats: Ratio | NormalEquations | RowEquations) -> tuple[jax.Array, jax.Array]:
        """The mean squared residual; under Gauss-Newton its derivative in the pooled J^T r is the
        update's damped step (`_stepped`), which leaves its value as it is."""
        if isinstance(stats, Ratio):
            return super().reduce_loss(stats)
        assert self.damping is not None, "Gauss-Newton statistics are stated where a damping is set"
        value, active = Ratio(stats.total, stats.mass).mean()
        system = (stats.normal,) if isinstance(stats, NormalEquations) else (stats.jacobian, stats.residual)
        stepped: jax.Array = _stepped(self.damping, stats.update, stats.rows, stats.pulled, system)
        return value + stepped.astype(value.dtype), active

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
