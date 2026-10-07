"""UniPC's predictor-corrector updates and their host-prepared coefficient tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .common import _check_endpoint_domain, _push


class UniPCState(NamedTuple):
    """`UniPC`'s state: the last `order` model outputs, most recent last; the
    sample its last predictor left from; that predictor's order, which its
    corrector takes (0 before the first predictor, when there is nothing to
    correct); the steps taken and the walk's count; and the walk's grid with
    every interval's coefficients (`_unipc_tables`)."""

    outputs: jax.Array
    last_x: jax.Array
    last_order: jax.Array
    taken: jax.Array
    steps: jax.Array
    grid: jax.Array
    corrector: jax.Array
    predictor: jax.Array


def _unipc_weights(rks: list[np.float64], hh: np.float64, B_h: np.float64, order: int,
                   predictor: bool) -> list[np.float64]:
    """UniPC's B(h) weights at `order`: the solution of R rho = b built from
    the relative positions `rks` of the history points (the current point's
    own 1 last). The predictor has no difference at the point it steps to,
    so it drops the last row and column; Diffusers takes 0.5 outright for
    the second-order predictor and the first-order corrector."""
    if order == (2 if predictor else 1):
        return [np.float64(0.5)]
    columns = order - 1 if predictor else order
    nodes = [*rks[:-1], np.float64(1.0)][:columns]
    # Scale the infinite node's Vandermonde column by r**(columns-1).
    # Its lower entries and recovered weight vanish; the leading entry
    # stays finite. This is the limit of the same system, not a lower-order solver.
    inverse = [np.float64(0.0 if np.isinf(rk) else 1.0) for rk in nodes]
    normalized = [np.sign(rk) if np.isinf(rk) else rk for rk in nodes]
    rows, b = [], []
    h_phi_k = np.expm1(hh) / hh - 1
    factorial = 1
    for i in range(1, columns + 1):
        rows.append([rk ** (i - 1) * inv ** (columns - i)
                     for rk, inv in zip(normalized, inverse, strict=True)])
        b.append(h_phi_k * factorial / B_h)
        factorial *= i + 1
        h_phi_k = h_phi_k / hh - 1 / factorial
    solution = np.linalg.solve(np.asarray(rows, np.float64), np.asarray(b, np.float64))
    return [solution[k] * inverse[k] ** (columns - 1) for k in range(columns)]


def _unipc_tables(alpha: np.ndarray, sigma: np.ndarray, solver: UniPC) -> tuple[np.ndarray, np.ndarray]:
    """Every interval's UniPC update as coefficients, in float64 on the host.

    Each depends only on the grid's rates and the order it is taken at, so
    lambda, h, e^h - 1 and the weight systems are solved once here rather
    than in float32 inside the walk, where Diffusers' scheduler also solves
    them. `corrector[j, p]` corrects the sample at grid point j with order
    p: weights on the sample, the last predictor's sample, m_{j-1}, the
    difference m_j - m_{j-1} and the history's differences
    m_{j-1-k} - m_{j-1}; p = 0 keeps the sample. `predictor[j, q-1]` steps
    from point j to j+1 with order q: weights on the corrected sample, the
    clean prediction, eps and the differences m_{j-k} - m_j. The
    differences stay in the walk, taken before any weight, as the source
    takes them.

    Only the pairs a walk can take are solved: the order a step reaches
    (`reach`, the history and `lower_order_final`'s cap) and below, as a
    walk restarted on the grid takes them, and no correction where
    `disable_corrector` names the step before. Every other pair holds
    zeros, or keeps the sample for a correction.
    """
    order, predict_x0 = solver.order, solver.predict_x0
    # An endpoint's lambda is infinite: log(0) at sigma 0 or alpha 0.
    with np.errstate(divide="ignore"):
        lambdas = np.log(alpha) - np.log(sigma)
    intervals = alpha.shape[0] - 1
    corrector = np.zeros((max(intervals, 0), order + 1, order + 3))
    predictor = np.zeros((max(intervals, 0), order, order + 2))
    corrector[:, :, 0] = 1.0
    here = 1 if predict_x0 else 2

    def reach(j: int) -> int:
        return min(order, j + 1, intervals - j if solver.lower_order_final else order)

    def b_h(hh):
        return hh if solver.solver_type == "bh1" else np.expm1(hh)

    for j in range(intervals):
        for p in range(1, reach(j - 1) + 1 if j > 0 and j - 1 not in solver.disable_corrector else 1):
            row = corrector[j, p]
            row[0] = 0.0
            h = lambdas[j] - lambdas[j - 1]
            hh = -h if predict_x0 else h
            rks = [(lambdas[j - 1 - k] - lambdas[j - 1]) / h for k in range(1, p)] + [np.float64(1.0)]
            B_h = b_h(hh)
            rhos = _unipc_weights(rks, hh, B_h, p, predictor=False)
            head = alpha[j] if predict_x0 else sigma[j]
            row[1] = sigma[j] / sigma[j - 1] if predict_x0 else alpha[j] / alpha[j - 1]
            row[2] = -head * np.expm1(hh)
            row[3] = -head * B_h * rhos[-1]
            for k in range(1, p):
                row[3 + k] = -head * B_h * rhos[k - 1] / rks[k - 1]
        for q in range(1, reach(j) + 1):
            row = predictor[j, q - 1]
            terminal = sigma[j + 1] <= 0
            if terminal:
                # The h -> infinity limit, which each prediction proves apart.
                if predict_x0 or alpha[j] == 0:
                    row[1] = alpha[j + 1]
                else:
                    row[0], row[2] = alpha[j + 1] / alpha[j], -alpha[j + 1] * sigma[j] / alpha[j]
                continue
            h = np.float64(1.0) if alpha[j] == 0 else lambdas[j + 1] - lambdas[j]
            hh = -h if predict_x0 else h
            B_h = b_h(hh)
            head = alpha[j + 1] if predict_x0 else sigma[j + 1]
            if alpha[j] == 0:
                row[1], row[2] = alpha[j + 1], sigma[j + 1]
            else:
                row[0] = sigma[j + 1] / sigma[j] if predict_x0 else alpha[j + 1] / alpha[j]
                row[here] = -head * np.expm1(hh)
            if q > 1:
                rks = [(lambdas[j - k] - lambdas[j]) / h for k in range(1, q)] + [np.float64(1.0)]
                rhos = _unipc_weights(rks, hh, B_h, q, predictor=True)
                for k in range(1, q):
                    row[2 + k] = -head * B_h * rhos[k - 1] / rks[k - 1]
    return corrector, predictor


@dataclass(frozen=True)
class UniPC:
    """UniPC, a unified predictor and corrector in lambda, as Diffusers 0.34.0's `UniPCMultistepScheduler`.

    UniPC is from Zhao et al. (2023, arXiv 2302.04867). Its weights solve a
    small linear system over the history's positions. Each step first
    corrects the sample the last predictor produced, with
    the model output just read there and that predictor's order, then
    predicts the next sample from the corrected one. `solver_type` picks
    B(h) as h (`bh1`) or e^h - 1 (`bh2`); `predict_x0` integrates the clean
    prediction, otherwise eps. `lower_order_final` caps the order by the
    steps remaining, Diffusers' rule, so the last step is first order;
    `disable_corrector` names the step indices whose predictor's output is
    not corrected. The lowered clean-prediction terminal is alpha_t*x_0;
    epsilon prediction uses the corrected sample with the pre-correction
    epsilon. Initialization rejects an unlowered higher-order zero target.
    At an alpha=0 source, bh1 and epsilon correctors diverge unless the
    first correction is disabled. The finite infinite-node weights use the
    same Vandermonde system with column scaling.

    Every coefficient depends on the grid alone, so `init` solves them in
    float64 for each interval and order, and a step reads
    its interval's row by where `t` sits on that grid and its order from
    the state. A step's `t` is a point of the grid `init` was given.
    """
    # `_unipc_tables` solves the coefficient tables in `init`.

    order: int = 2
    solver_type: Literal["bh1", "bh2"] = "bh2"
    predict_x0: bool = True
    lower_order_final: bool = True
    disable_corrector: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError(f"UniPC's order is at least 1, not {self.order}")

    def init(self, x, times, process, *, key):
        corrects_source = times.shape[0] > 2 and 0 not in self.disable_corrector
        _check_endpoint_domain(
            process, times,
            source=corrects_source and (not self.predict_x0 or self.solver_type == "bh1"),
            target=not self.lower_order_final and self.order > 1 and times.shape[0] > 2,
            reason="UniPC corrector/predictor requires finite log-SNR at this endpoint")
        with jax.ensure_compile_time_eval():
            alpha, sigma = process.sampler_schedule.rates(jnp.asarray(times))
            corrector, predictor = _unipc_tables(np.asarray(alpha, np.float64), np.asarray(sigma, np.float64),
                                                 self)
        return UniPCState(jnp.zeros((self.order, *x.shape), x.dtype), x, jnp.zeros((), jnp.int32),
                          jnp.zeros((), jnp.int32), jnp.asarray(times.shape[0] - 1, jnp.int32),
                          jnp.asarray(times, jnp.float32), jnp.asarray(corrector, jnp.float32),
                          jnp.asarray(predictor, jnp.float32))

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        def weighted(row: jax.Array, terms: list[jax.Array]) -> jax.Array:
            total = row[0] * terms[0]
            for index in range(1, len(terms)):
                total = total + row[index] * terms[index]
            return total

        interval = jnp.argmin(jnp.abs(state.grid - t.reshape(-1)[0]))
        here = denoised if self.predict_x0 else eps
        history = [state.outputs[-k] for k in range(1, self.order + 1)]
        row = state.corrector[interval, state.last_order]
        previous = history[0]
        x = weighted(row, [x, state.last_x, previous, here - previous,
                           *(older - previous for older in history[1:])])
        this_order = (jnp.minimum(self.order, state.steps - state.taken) if self.lower_order_final
                      else self.order)
        this_order = jnp.minimum(this_order, state.taken + 1)
        row = state.predictor[interval, this_order - 1]
        next_x = weighted(row, [x, denoised, eps, *(older - here for older in history[:self.order - 1])])
        return next_x, state._replace(outputs=_push(state.outputs, here), last_x=x, last_order=this_order,
                                      taken=state.taken + 1)
