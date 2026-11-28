"""DPM-Solver and DEIS exponential integrators in log-SNR.

Every weight of an update depends on the grid alone, so `init` solves each
step's row of weights once in float64 on the host, from the schedule's own
rates (`_solved_levels`), and a step is its row's weighted sum of the sample,
the model's outputs and their differences (`_weighted`), the differences
taken in the walk before any weight, as the source takes them. The source
computes the same coefficients in float32 inside the walk; their logs of
rates near zero rounded there put the noise-prediction forms several times
the source's error from the float64 truth (rows 13650 to 13653).
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import lax

from dew.nn.programs import on_host

from .common import (
    Tabled,
    Term,
    _apart,
    _check_endpoint_domain,
    _check_gaps,
    _column,
    _history_gaps,
    _moved,
    _shifted,
    _solved_levels,
    _tabled,
    _tapered_orders,
    _unit_noise,
    _weighted,
)

Algorithm = Literal["dpmsolver++", "dpmsolver", "sde-dpmsolver++", "sde-dpmsolver"]


def _dpm_terms(algorithm: str, alpha_s, sigma_s, alpha_t, sigma_t, h):
    """The coefficients of one DPM-Solver algorithm over h, as Diffusers
    writes them: `(a, b, c_midpoint, c_heun, c_2, n)` for
    x_t = a x + b D0 + c D1 + c_2 D2 + n z, with c the midpoint or Heun
    second-order form (the third order takes the Heun one) and z standard
    normal noise the SDE forms add. `dpmsolver++` and `sde-dpmsolver++`
    integrate the clean prediction; the other two integrate eps. Host
    float64 numbers in, the same out (`sde-dpmsolver`'s c_2 NaN: it has no third order,
    which `DPMSolverMultistep` refuses). Over an
    interval of no length (h 0) the forms a step does not take there are
    0 / 0; `_check_gaps` refuses a grid on which a step takes one."""
    alpha_s = 1.0 if alpha_s == 0 else alpha_s
    with np.errstate(divide="ignore", invalid="ignore"):
        if algorithm == "dpmsolver++":
            phi = np.expm1(-h)
            # The source's -alpha_t (e^-h - 1). The same scale written from the rates,
            # alpha_t - (sigma_t / sigma_s) alpha_s, cancels two terms of alpha_t's size.
            clean_scale = -(alpha_t * phi)
            return (sigma_t / sigma_s, clean_scale, 0.5 * clean_scale,
                    alpha_t * (phi / h + 1.0), -alpha_t * ((phi + h) / h ** 2 - 0.5), 0.0)
        if algorithm == "dpmsolver":
            psi = np.expm1(h)
            return (alpha_t / alpha_s, -sigma_t * psi, -0.5 * sigma_t * psi,
                    -sigma_t * (psi / h - 1.0), -sigma_t * ((psi - h) / h ** 2 - 0.5), 0.0)
        if algorithm == "sde-dpmsolver++":
            decay = -np.expm1(-2.0 * h)
            return (sigma_t / sigma_s * np.exp(-h), alpha_t * decay, 0.5 * alpha_t * decay,
                    alpha_t * (decay / (-2.0 * h) + 1.0),
                    alpha_t * ((decay - 2.0 * h) / (2.0 * h) ** 2 - 0.5),
                    sigma_t * np.sqrt(decay))
        psi = np.expm1(h)
        return (alpha_t / alpha_s, -2.0 * sigma_t * psi, -sigma_t * psi,
                -2.0 * sigma_t * (psi / h - 1.0), np.nan, sigma_t * np.sqrt(np.expm1(2 * h)))


def _step_h(alpha, sigma, lambdas, source: int, target: int) -> float:
    """h = lambda_target - lambda_source, or 1 where either end is an endpoint (a zero target sigma or a
    zero source alpha), whose step each update writes as its own limit instead."""
    if sigma[target] <= 0 or alpha[source] == 0:
        return 1.0
    return float(lambdas[target] - lambdas[source])


_MULTISTEP = ("x", "denoised", "eps", "first", "second", "noise", "fresh", "shift", "kept")
"""A multistep row's columns: the sample moved by the newest output (`x`, the sample less `shift` times
that output), the clean prediction and the noise read at the sample, the newest output's difference from
the one before (`first`), that one's from the one before it (`second`), the step's unit draw, the draw
less the noise read (`fresh`: the non-++ SDE's terminal limit scales the difference, not each apart), the
shift, and the newest output's weight unshifted (`Moved.kept`)."""


def _multistep_terms(x, denoised, eps, output: str, outputs: jax.Array, noise, row, *, fresh: bool = False
                     ) -> list[Term]:
    """The terms a `_MULTISTEP` row weighs, from the step's output (`output`: its column, `denoised` or
    `eps`) and the history `outputs` (most recent last)."""
    zero = jnp.zeros_like(x)
    first = _apart(outputs[-1], outputs[-2]) if outputs.shape[0] > 1 else zero
    second = _apart(outputs[-2], outputs[-3]) if outputs.shape[0] > 2 else zero
    drawn = zero if noise is None else noise
    moved = _moved(x, denoised if output == "denoised" else eps, row, _column(_MULTISTEP, "shift"),
                   _column(_MULTISTEP, output), _column(_MULTISTEP, "kept"))
    terms: list[Term] = [moved, denoised, eps, first, second, drawn]
    # The draw less the noise read, which the non-++ SDE's terminal limit alone weighs.
    return [*terms, _apart(drawn, eps)] if fresh else terms


@struct.dataclass
class DPMSolverMultistep:
    """DPM-Solver and DPM-Solver++ as multistep integrators in lambda = log(alpha) - log(sigma).

    The papers are Lu et al. (2022, arXiv 2206.00927) and arXiv 2211.01095.
    This covers the four algorithms, three orders and two second-order forms
    of Diffusers 0.34.0's `DPMSolverMultistepScheduler`, with its defaults.

    `dpmsolver++` and `sde-dpmsolver++` integrate the clean prediction, the
    other two eps; the `sde-` forms add the noise term of the SDE solver.
    With h = lambda_t - lambda_s0 over the step and D0, D1, D2 the finite
    differences of the last outputs in lambda, the deterministic `dpmsolver++`
    step is

        x_t = (sigma_t / sigma_s0) x - alpha_t (e^-h - 1) D0 + c D1 + c_2 D2

    with each algorithm's coefficients written as Diffusers writes them. The
    first step has no history and is first order, the second at
    most second. `lower_order_final` is Diffusers' rule verbatim, which acts
    only in a walk under 15 steps: first order on the last step and at most
    second on the one before. `euler_at_final` makes the last step first
    order whatever the length. A zero target forces the clean limit for
    deterministic and ++ updates. The non-++ SDE first-order limit also
    retains alpha_t*sigma_s/alpha_s*(noise-eps); it requires a finite source
    alpha and a final order reduction. That endpoint is an equation-limit
    extension: Diffusers refuses a literal zero-terminal non-++ config.
    EDM uses the ++ algorithms over its existing process.

    DPM-Solver++ 2M with no order taper is order=2,
    algorithm="dpmsolver++", solver_type="midpoint", lower_order_final=False,
    euler_at_final=False.
    """

    order: int = struct.field(pytree_node=False, default=2)
    algorithm: Algorithm = struct.field(pytree_node=False, default="dpmsolver++")
    solver_type: Literal["midpoint", "heun"] = struct.field(pytree_node=False, default="midpoint")
    lower_order_final: bool = struct.field(pytree_node=False, default=True)
    euler_at_final: bool = struct.field(pytree_node=False, default=False)

    def __post_init__(self) -> None:
        if self.order not in (1, 2, 3):
            raise ValueError(f"DPM-Solver has orders 1, 2 and 3, not {self.order}")
        if self.algorithm == "sde-dpmsolver" and self.order == 3:
            raise ValueError("sde-dpmsolver has no third-order update; use order 2 or sde-dpmsolver++")

    def init(self, x, times, process, *, key):
        if self.algorithm == "sde-dpmsolver":
            first_at_end = (self.order == 1 or self.euler_at_final
                            or (self.lower_order_final and times.shape[0] - 1 < 15))
            _check_endpoint_domain(
                process, times, source=True, target=not first_at_end,
                reason="sde-dpmsolver requires finite alpha and a lowered order at sigma zero")
        orders = _tapered_orders(self.order, times.shape[0] - 1, self.lower_order_final, self.euler_at_final)
        alpha, sigma, levels = _solved_levels(process, times)
        # A Heun or third-order coefficient divides by its own step's h too, but at an endpoint, where h
        # is a placeholder (`_step_h`); the midpoint one is 0 with h, and its step stays put.
        divides = [k == 3 or (k == 2 and self.solver_type == "heun") for k in orders]
        spans = [(i, i + 1) for i, divided in enumerate(divides)
                 if divided and sigma[i + 1] > 0 and alpha[i] != 0]
        _check_gaps(levels, _history_gaps(orders) + spans, reason="DPM-Solver's higher orders divide by")
        rows = np.zeros((len(orders), len(_MULTISTEP)))
        for i, order in enumerate(orders):
            rows[i] = self._row(alpha, sigma, levels, i, order)
        return _tabled(x, self.order, rows)

    def _row(self, alpha, sigma, levels, i: int, order: int) -> np.ndarray:
        """Step `i`'s weights at `order`, from grid point i to i + 1 (`_MULTISTEP`)."""
        row = np.zeros(len(_MULTISTEP))
        alpha_s0, sigma_s0, alpha_t, sigma_t = alpha[i], sigma[i], alpha[i + 1], sigma[i + 1]
        output = _column(_MULTISTEP, "denoised" if self.algorithm.endswith("++") else "eps")
        if sigma_t <= 0:
            # The clean limit; the non-++ SDE keeps its noise's limit too.
            row[_column(_MULTISTEP, "denoised")] = alpha_t
            if self.algorithm == "sde-dpmsolver":
                row[_column(_MULTISTEP, "fresh")] = alpha_t * sigma_s0 / alpha_s0
            return row
        h = _step_h(alpha, sigma, levels, i, i + 1)
        a, b, c_midpoint, c_heun, c_2, n = _dpm_terms(self.algorithm, alpha_s0, sigma_s0, alpha_t, sigma_t, h)
        stochastic = self.algorithm.startswith("sde")
        if alpha_s0 == 0:
            row[_column(_MULTISTEP, "denoised")] = alpha_t
            row[_column(_MULTISTEP, "noise" if stochastic else "eps")] = sigma_t
        else:
            shift = alpha_s0 if self.algorithm.endswith("++") else sigma_s0
            (row[_column(_MULTISTEP, "x")], row[output], row[_column(_MULTISTEP, "shift")],
             row[_column(_MULTISTEP, "kept")]) = _shifted(a, b, shift)
            row[_column(_MULTISTEP, "noise")] = n
        # D1 = (m0 - m1) / r0 and D2 from the gaps h_0 = r0 h and h_1 = r1 h behind the step.
        if order == 2:
            c = c_midpoint if self.solver_type == "midpoint" else c_heun
            row[_column(_MULTISTEP, "first")] = c * h / (levels[i] - levels[i - 1])
        elif order == 3:
            r0 = (levels[i] - levels[i - 1]) / h
            r1 = (levels[i - 1] - levels[i - 2]) / h
            # d1 = d1_0 + r0 / (r0 + r1) (d1_0 - d1_1), d2 = (d1_0 - d1_1) / (r0 + r1),
            # d1_0 = first / r0, d1_1 = second / r1.
            row[_column(_MULTISTEP, "first")] = (c_heun * (1 + r0 / (r0 + r1)) + c_2 / (r0 + r1)) / r0
            row[_column(_MULTISTEP, "second")] = -(c_heun * r0 + c_2) / ((r0 + r1) * r1)
        return row

    def step(self, x, t, t_next, denoised, eps, state: Tabled, key, process, denoise):
        output = "denoised" if self.algorithm.endswith("++") else "eps"
        state, row = state.push(denoised if output == "denoised" else eps), state.row()
        noise = _unit_noise(key, x) if self.algorithm.startswith("sde") else None
        stepped = _weighted(row, _multistep_terms(x, denoised, eps, output, state.outputs, noise, row,
                                                  fresh=self.algorithm == "sde-dpmsolver"))
        return stepped.astype(x.dtype), state.advance()


class Singlestep(NamedTuple):
    """`DPMSolverSinglestep`'s state: the output history with every step's
    row of weights, the sample its current group of steps started from, and
    the order of every step."""

    history: Tabled
    anchor: jax.Array
    orders: jax.Array


_SINGLESTEP = ("anchor", "denoised", "base", "first", "second", "noise", "shift", "kept")
"""A singlestep row's columns: the sample its group started from moved by the model output read there
(`anchor`, less `shift` times `base`; `_shifted`), the clean prediction read now, that output (`base`),
the next two outputs' differences from it, the step's unit draw, the shift, and `base`'s weight unshifted
(`Moved.kept`)."""


@struct.dataclass
class DPMSolverSinglestep:
    """Diffusers 0.34.0's grouped DPM-Solver updates from each group's anchor.

    A group of k model evaluations completes one k-th order update. Orders
    repeat [1, 2] or [1, 2, 3]; an incomplete final group uses lower order,
    as set_timesteps does in the reference. lower_order_final also lowers
    the final complete group, and a zero-sigma target forces final order 1.
    Source-domain checks use that effective order list.

    At alpha=0, clean-prediction midpoint and deterministic Heun groups have
    finite limits. Noise-prediction groups above order 1 and third-order
    SDE Heun groups diverge; initialization rejects those source/grid pairs.
    """

    order: int = struct.field(pytree_node=False, default=2)
    algorithm: Literal["dpmsolver++", "dpmsolver", "sde-dpmsolver++"] = struct.field(
        pytree_node=False, default="dpmsolver++")
    solver_type: Literal["midpoint", "heun"] = struct.field(pytree_node=False, default="midpoint")
    lower_order_final: bool = struct.field(pytree_node=False, default=False)

    def __post_init__(self) -> None:
        if self.order not in (1, 2, 3):
            raise ValueError(f"DPM-Solver has orders 1, 2 and 3, not {self.order}")

    def order_list(self, steps: int) -> list[int]:
        """Reference groups, completing an uneven final group at lower order."""
        if steps == 0:
            return []
        order = self.order
        if self.lower_order_final or steps % order:
            if order == 3:
                if steps % 3 == 0:
                    return [1, 2, 3] * (steps // 3 - 1) + [1, 2] + [1]
                if steps % 3 == 1:
                    return [1, 2, 3] * (steps // 3) + [1]
                return [1, 2, 3] * (steps // 3) + [1, 2]
            if order == 2:
                if steps % 2 == 0:
                    return [1, 2] * (steps // 2 - 1) + [1, 1]
                return [1, 2] * (steps // 2) + [1]
            return [1] * steps
        return list(range(1, order + 1)) * (steps // order)

    def init(self, x, times, process, *, key):
        steps = times.shape[0] - 1
        orders = self.order_list(steps)
        alpha, sigma, levels = _solved_levels(process, times)
        if orders and sigma[-1] == 0:
            orders[-1] = 1
        if self.algorithm == "dpmsolver" and len(orders) > 1 and orders[1] > 1:
            _check_endpoint_domain(
                process,
                times,
                source=True,
                reason="noise-prediction singlestep differences diverge at an alpha=0 anchor",
            )
        if (self.algorithm == "sde-dpmsolver++" and self.solver_type == "heun"
                and 3 in orders[:3]):
            _check_endpoint_domain(process, times, source=True,
                                   reason="third-order SDE Heun differences diverge at an alpha=0 anchor")
        _check_gaps(levels, self._gaps(orders, alpha), reason="DPM-Solver's higher orders divide by")
        rows = np.zeros((len(orders), len(_SINGLESTEP)))
        for i, order in enumerate(orders):
            rows[i] = self._row(alpha, sigma, levels, i, order)
        with on_host():
            order_of = jnp.asarray(orders, jnp.int32)
        return Singlestep(_tabled(x, self.order, rows), x, order_of)

    def _gaps(self, orders: list[int], alpha: np.ndarray) -> list[tuple[int, int]]:
        """The pairs of points whose gap each step's update divides by (`_check_gaps`).

        An update of order k reads the k points from its group's anchor, k - 1
        back. Its h runs from the anchor to its target, and its ratios from the
        anchor to each later point, except from an anchor at alpha 0, whose
        limit reads neither. A third-order Heun form also divides by the gap
        between its two latest points, and so does its `dpmsolver++` limit from
        an alpha=0 anchor. A terminal step is first order (`init`)."""
        gaps = []
        for i, k in enumerate(orders):
            anchor = i - k + 1
            finite = alpha[anchor] != 0
            if k > 1 and finite:
                gaps += [(anchor, i + 1), *((anchor, anchor + m) for m in range(1, k))]
            if k == 3 and self.solver_type == "heun" and (finite or self.algorithm == "dpmsolver++"):
                gaps.append((i - 1, i))
        return gaps

    def _row(self, alpha, sigma, levels, i: int, order: int) -> np.ndarray:
        """Step `i`'s weights at `order`, from its group's anchor `order - 1` points back to point i + 1
        (`_SINGLESTEP`). The outputs from the anchor on are m_a = base, then base + first and base +
        second."""
        row = np.zeros(len(_SINGLESTEP))
        column = {name: _column(_SINGLESTEP, name) for name in _SINGLESTEP}
        alpha_t, sigma_t = alpha[i + 1], sigma[i + 1]
        if sigma_t <= 0:
            row[column["denoised"]] = alpha_t
            return row
        anchor = i - order + 1
        alpha_a, sigma_a = alpha[anchor], sigma[anchor]
        h = _step_h(alpha, sigma, levels, anchor, i + 1)
        a, b, c_midpoint, c_heun, c_2, n = _dpm_terms(self.algorithm, alpha_a, sigma_a, alpha_t, sigma_t, h)
        stochastic = self.algorithm.startswith("sde")
        if alpha_a == 0:
            # The alpha=0 anchor's limit: alpha_t times a clean estimate, beside the noise the SDE draws or
            # the anchor's own share.
            midpoint = 0.5 if self.solver_type == "midpoint" else 1.0
            clean = {1: {"denoised": 1.0}, 2: {"base": 1.0, "first": midpoint},
                     3: {"base": 1.0, "second": 1.0}}[order]
            if order == 3 and self.solver_type == "heun" and self.algorithm == "dpmsolver++":
                # D1 and D2 combined before the infinite-anchor limit: their divergent leading terms cancel.
                with np.errstate(divide="ignore"):
                    gap = np.log(alpha_t) - np.log(sigma_t) - levels[i]
                share = (gap - 1.0) / (levels[i] - levels[i - 1])
                clean = {"base": 1.0, "second": 1.0 + share, "first": -share}
            for name, weight in clean.items():
                row[column[name]] = alpha_t * weight
            if stochastic:
                row[column["noise"]] = sigma_t
            else:
                row[column["anchor"]] = sigma_t / sigma_a
            return row
        shift = alpha_a if self.algorithm.endswith("++") else sigma_a
        row[column["anchor"]], row[column["base"]], row[column["shift"]], row[column["kept"]] = _shifted(
            a, b, shift)
        row[column["noise"]] = n
        if order == 2:
            c = c_midpoint if self.solver_type == "midpoint" else c_heun
            row[column["first"]] = c * h / (levels[i] - levels[anchor])
        elif order == 3:
            r0 = (levels[i] - levels[anchor]) / h
            r1 = (levels[i - 1] - levels[anchor]) / h
            if self.solver_type == "midpoint":
                row[column["second"]] = c_heun / r0
            else:
                # d1 = (r0 d1_0 - r1 d1_1) / (r0 - r1), d2 = 2 (d1_1 - d1_0) / (r0 - r1),
                # d1_0 = first / r1, d1_1 = second / r0.
                row[column["first"]] = (c_heun * r0 - 2 * c_2) / (r1 * (r0 - r1))
                row[column["second"]] = (2 * c_2 - c_heun * r1) / (r0 * (r0 - r1))
        return row

    def step(self, x, t, t_next, denoised, eps, state: Singlestep, key, process, denoise):
        history = state.history.push(denoised if self.algorithm.endswith("++") else eps)
        order = state.orders[history.taken]
        anchor = jnp.where(order == 1, x, state.anchor)
        depth = history.outputs.shape[0]
        start = depth - order

        def output(offset: int) -> jax.Array:
            return lax.dynamic_index_in_dim(history.outputs, jnp.minimum(start + offset, depth - 1),
                                            keepdims=False)

        base, row = output(0), history.row()
        noise = _unit_noise(key, x) if self.algorithm.startswith("sde") else jnp.zeros_like(x)
        moved = _moved(anchor, base, row, _column(_SINGLESTEP, "shift"), _column(_SINGLESTEP, "base"),
                       _column(_SINGLESTEP, "kept"))
        terms = [moved, denoised, base, _apart(output(1), base), _apart(output(2), base), noise]
        stepped = _weighted(row, terms)
        return stepped.astype(x.dtype), Singlestep(history.advance(), anchor, state.orders)


def _deis_second(t, b, c):
    """Integral of the log-rho Lagrange basis, including t=0 and a node at infinity (host float64)."""
    if np.isinf(b):
        return 0.0
    if np.isinf(c):
        return t
    log_t = np.log(t) if t != 0 else 0.0
    return t * (-np.log(c) + log_t - 1) / (np.log(b) - np.log(c))


def _deis_third(t, b, c, d):
    """Integral of the quadratic log-rho basis; an infinite node has zero
    weight, and each other basis loses its corresponding factor (host float64)."""
    if np.isinf(b):
        return 0.0
    if np.isinf(c):
        return _deis_second(t, b, d)
    if np.isinf(d):
        return _deis_second(t, b, c)
    lb, lc, ld = np.log(b), np.log(c), np.log(d)
    lt = np.log(t) if t != 0 else 0.0
    numerator = t * (lc * (ld - lt + 1) - ld * lt + ld + lt ** 2 - 2 * lt + 2)
    return numerator / ((lb - lc) * (lb - ld))


@struct.dataclass
class DEIS:
    """DEIS (Zhang and Chen 2023, arXiv 2204.13902) in its log-rho multistep
    form, Diffusers 0.34.0's `DEISMultistepScheduler`: the exponential
    integrator of eps with the polynomial-in-log(rho) interpolation of the
    last outputs, rho = sigma / alpha, integrated in closed form over the
    step. The first order is DPM-Solver's. Orders grow with the history;
    `lower_order_final` is the same short-walk taper as
    `DPMSolverMultistep`'s. At sigma=0 the integrated log-rho basis retains
    its history terms. An alpha=0 source contributes a node at infinite rho;
    that node's weight vanishes in subsequent finite-interval integrals.
    The outputs' weights sum to the first order's, so a row weighs the
    newest output by that sum and the history by its differences.
    """

    order: int = struct.field(pytree_node=False, default=2)
    lower_order_final: bool = struct.field(pytree_node=False, default=True)

    def __post_init__(self) -> None:
        if self.order not in (1, 2, 3):
            raise ValueError(f"DEIS has orders 1, 2 and 3, not {self.order}")

    def init(self, x, times, process, *, key):
        orders = _tapered_orders(self.order, times.shape[0] - 1, self.lower_order_final, euler_at_final=False)
        alpha, sigma, levels = _solved_levels(process, times)
        _check_gaps(levels, _history_gaps(orders), reason="DEIS's higher orders divide by")
        rows = np.zeros((len(orders), len(_MULTISTEP)))
        with np.errstate(divide="ignore"):
            rhos = np.where(alpha == 0, np.inf, sigma / np.where(alpha == 0, 1.0, alpha))
        for i, order in enumerate(orders):
            rows[i] = self._row(alpha, sigma, levels, rhos, i, order)
        return _tabled(x, self.order, rows)

    @staticmethod
    def _row(alpha, sigma, levels, rhos, i: int, order: int) -> np.ndarray:
        """Step `i`'s weights at `order` (`_MULTISTEP`)."""
        row = np.zeros(len(_MULTISTEP))
        alpha_s0, alpha_t, sigma_t = alpha[i], alpha[i + 1], sigma[i + 1]
        if order == 1:
            if alpha_s0 == 0 or sigma_t <= 0:
                row[_column(_MULTISTEP, "denoised")] = alpha_t
                row[_column(_MULTISTEP, "eps")] = sigma_t
            else:
                # DPM-Solver's first-order step in eps (`_dpm_terms`).
                h = _step_h(alpha, sigma, levels, i, i + 1)
                a, b, *_ = _dpm_terms("dpmsolver", alpha_s0, sigma[i], alpha_t, sigma_t, h)
                (row[_column(_MULTISTEP, "x")], row[_column(_MULTISTEP, "eps")],
                 row[_column(_MULTISTEP, "shift")], row[_column(_MULTISTEP, "kept")]) = _shifted(
                    a, b, sigma[i])
            return row
        rho_t = sigma_t / alpha_t
        nodes = [rhos[i - k] for k in range(order)]
        if order == 2:
            coefficients = [_deis_second(rho_t, nodes[k], nodes[1 - k])
                            - _deis_second(nodes[0], nodes[k], nodes[1 - k]) for k in range(2)]
        else:
            others = [(1, 2), (2, 0), (0, 1)]
            coefficients = [_deis_third(rho_t, nodes[k], nodes[p], nodes[q])
                            - _deis_third(nodes[0], nodes[k], nodes[p], nodes[q])
                            for k, (p, q) in enumerate(others)]
        # x_t = alpha_t (x / alpha_s0 + sum_k c_k m_k): the newest output by the coefficients' sum, then
        # m_0 - m_1 by minus the sum of the older ones, m_1 - m_2 by minus the oldest.
        (row[_column(_MULTISTEP, "x")], row[_column(_MULTISTEP, "eps")], row[_column(_MULTISTEP, "shift")],
         row[_column(_MULTISTEP, "kept")]) = _shifted(alpha_t / alpha_s0, alpha_t * sum(coefficients),
                                                      sigma[i])
        row[_column(_MULTISTEP, "first")] = -alpha_t * sum(coefficients[1:])
        if order == 3:
            row[_column(_MULTISTEP, "second")] = -alpha_t * coefficients[2]
        return row

    def step(self, x, t, t_next, denoised, eps, state: Tabled, key, process, denoise):
        state, row = state.push(eps), state.row()
        stepped = _weighted(row, _multistep_terms(x, denoised, eps, "eps", state.outputs, None, row))
        return stepped.astype(x.dtype), state.advance()
