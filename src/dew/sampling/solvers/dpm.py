"""DPM-Solver and DEIS exponential integrators in log-SNR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .common import Multistep, _check_endpoint_domain, _half_log_snr, _lambda_step, _multistep, _tapered_order

Algorithm = Literal["dpmsolver++", "dpmsolver", "sde-dpmsolver++", "sde-dpmsolver"]


def _dpm_terms(algorithm: str, alpha_s, sigma_s, alpha_t, sigma_t, h):
    """The coefficients of one DPM-Solver algorithm over h, as Diffusers
    writes them: `(a, b, c_midpoint, c_heun, c_2, n)` for
    x_t = a x + b D0 + c D1 + c_2 D2 + n z, with c the midpoint or Heun
    second-order form (the third order takes the Heun one) and z standard
    normal noise the SDE forms add. `dpmsolver++` and `sde-dpmsolver++`
    integrate the clean prediction; the other two integrate eps."""
    alpha_s = jnp.where(alpha_s == 0, 1.0, alpha_s)
    if algorithm == "dpmsolver++":
        phi = jnp.exp(-h) - 1.0
        sample_scale = sigma_t / sigma_s
        # exp(-h) = alpha_s / alpha_t * sigma_t / sigma_s. Use the
        # rates directly rather than round them through logarithms and exp.
        clean_scale = alpha_t - sample_scale * alpha_s
        return (sample_scale, clean_scale, 0.5 * clean_scale,
                alpha_t * (phi / h + 1.0), -alpha_t * ((phi + h) / h ** 2 - 0.5), 0.0)
    if algorithm == "dpmsolver":
        psi = jnp.exp(h) - 1.0
        return (alpha_t / alpha_s, -sigma_t * psi, -0.5 * sigma_t * psi,
                -sigma_t * (psi / h - 1.0), -sigma_t * ((psi - h) / h ** 2 - 0.5), 0.0)
    if algorithm == "sde-dpmsolver++":
        decay = 1.0 - jnp.exp(-2.0 * h)
        return (sigma_t / sigma_s * jnp.exp(-h), alpha_t * decay, 0.5 * alpha_t * decay,
                alpha_t * (decay / (-2.0 * h) + 1.0),
                alpha_t * ((decay - 2.0 * h) / (2.0 * h) ** 2 - 0.5),
                sigma_t * jnp.sqrt(decay))
    psi = jnp.exp(h) - 1.0
    return (alpha_t / alpha_s, -2.0 * sigma_t * psi, -sigma_t * psi,
            -2.0 * sigma_t * (psi / h - 1.0), None, sigma_t * jnp.sqrt(jnp.exp(2 * h) - 1.0))


@dataclass(frozen=True)
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
    # `_dpm_terms` holds each algorithm's coefficients.

    order: int = 2
    algorithm: Algorithm = "dpmsolver++"
    solver_type: Literal["midpoint", "heun"] = "midpoint"
    lower_order_final: bool = True
    euler_at_final: bool = False

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
        return _multistep(x, times, self.order)

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        alpha_s0, sigma_s0 = process.rates(t, like=x)
        alpha_t, sigma_t = process.rates(t_next, like=x)
        history = state.push(denoised if self.algorithm.endswith("++") else eps, alpha_s0, sigma_s0)
        terminal, h = _lambda_step(alpha_s0, sigma_s0, alpha_t, sigma_t)
        a, b, c_midpoint, c_heun, c_2, n = _dpm_terms(
            self.algorithm, alpha_s0, sigma_s0, alpha_t, sigma_t, h)
        lambdas, outputs = history.lambdas, history.outputs
        m0 = outputs[-1]
        base = a * x + b * m0
        target_limit = alpha_t * denoised
        if self.algorithm.startswith("sde"):
            noise = jax.random.normal(key, x.shape, dtype=jnp.float32)
            base = base + n * noise
            source_limit = alpha_t * denoised + sigma_t * noise
            if self.algorithm == "sde-dpmsolver":
                target_limit = target_limit + alpha_t * sigma_s0 / alpha_s0 * (noise - eps)
        else:
            source_limit = alpha_t * denoised + sigma_t * eps
        base = jnp.where(alpha_s0 == 0, source_limit, base)

        def first(_):
            return base

        def second(_):
            r0 = (lambdas[-1] - lambdas[-2]) / h
            d1 = (m0 - outputs[-2]) / r0
            return base + (c_midpoint if self.solver_type == "midpoint" else c_heun) * d1

        def third(_):
            r0 = (lambdas[-1] - lambdas[-2]) / h
            r1 = (lambdas[-2] - lambdas[-3]) / h
            d1_0 = (m0 - outputs[-2]) / r0
            d1_1 = (outputs[-2] - outputs[-3]) / r1
            d1 = d1_0 + (r0 / (r0 + r1)) * (d1_0 - d1_1)
            d2 = (d1_0 - d1_1) / (r0 + r1)
            return base + c_heun * d1 + c_2 * d2

        order = _tapered_order(self.order, history, self.lower_order_final, self.euler_at_final)
        stepped = lax.switch(order - 1, [first, second, third][:self.order], None)
        return jnp.where(terminal, target_limit, stepped), history.advance()


class Singlestep(NamedTuple):
    """`DPMSolverSinglestep`'s state: the output history, the sample its
    current group of steps started from, and the order of every step."""

    history: Multistep
    anchor: jax.Array
    orders: jax.Array


@dataclass(frozen=True)
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

    order: int = 2
    algorithm: Literal["dpmsolver++", "dpmsolver", "sde-dpmsolver++"] = "dpmsolver++"
    solver_type: Literal["midpoint", "heun"] = "midpoint"
    lower_order_final: bool = False

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
        if orders:
            with jax.ensure_compile_time_eval():
                _, last_sigma = process.sampler_schedule.rates(times[-1:])
                if bool(jnp.all(last_sigma == 0)):
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
        return Singlestep(_multistep(x, times, self.order), x, jnp.asarray(orders, jnp.int32))

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        alpha_s0, sigma_s0 = process.rates(t, like=x)
        alpha_t, sigma_t = process.rates(t_next, like=x)
        history = state.history.push(
            denoised if self.algorithm.endswith("++") else eps, alpha_s0, sigma_s0)
        order = jnp.where(jnp.all(sigma_t == 0), 1, state.orders[history.taken])
        anchor = jnp.where(order == 1, x, state.anchor)
        lambdas, outputs = history.lambdas, history.outputs
        m0 = outputs[-1]
        noise = jax.random.normal(key, x.shape, dtype=jnp.float32)

        def update(back: int):
            """The update's coefficients from the group's anchor, `back`
            points behind, and the noise term at its h."""
            alpha_a, sigma_a = history.alphas[-back], history.sigmas[-back]
            _, h = _lambda_step(alpha_a, sigma_a, alpha_t, sigma_t)
            a, b, c_midpoint, c_heun, c_2, n = _dpm_terms(
                self.algorithm, alpha_a, sigma_a, alpha_t, sigma_t, h)
            return h, a * anchor, b, c_midpoint, c_heun, c_2, n * noise

        def source_limit(back: int, clean):
            if self.algorithm.startswith("sde"):
                return alpha_t * clean + sigma_t * noise
            return sigma_t / history.sigmas[-back] * anchor + alpha_t * clean

        def first(_):
            _, ax, b, _, _, _, dz = update(1)
            stepped = ax + b * m0
            if self.algorithm.startswith("sde"):
                stepped = stepped + dz
            return jnp.where(alpha_s0 == 0, source_limit(1, denoised), stepped)

        def second(_):
            h, ax, b, c_midpoint, c_heun, _, dz = update(2)
            m1 = outputs[-2]
            from_zero = history.alphas[-2] == 0
            r0 = jnp.where(from_zero, 1.0, (lambdas[-1] - lambdas[-2]) / h)
            d1 = (m0 - m1) / r0
            stepped = ax + b * m1 + (c_midpoint if self.solver_type == "midpoint" else c_heun) * d1
            if self.algorithm.startswith("sde"):
                stepped = stepped + dz
            weight = 0.5 if self.solver_type == "midpoint" else 1.0
            return jnp.where(from_zero, source_limit(2, m1 + weight * (m0 - m1)), stepped)

        def third(_):
            return _singlestep_third(self, update, source_limit, history, alpha_t, sigma_t)

        stepped = lax.switch(order - 1, [first, second, third][:self.order], None)
        terminal = sigma_t <= 0
        next_x = jnp.where(terminal, alpha_t * denoised, stepped)
        return next_x, Singlestep(history.advance(), anchor, state.orders)


def _singlestep_third(solver: DPMSolverSinglestep, update, source_limit, history: Multistep,
                      alpha_t, sigma_t):
    """The third-order singlestep update from the anchor three points back.

    `update` gives that anchor's coefficients and its noise term, and
    `source_limit` the alpha=0 limit. The Heun `dpmsolver++` form combines
    the two differences before the limit is taken, because their leading
    terms diverge individually and cancel together.
    """
    lambdas, outputs = history.lambdas, history.outputs
    m0, m1, m2 = outputs[-1], outputs[-2], outputs[-3]
    h, ax, b, _, c_heun, c_2, dz = update(3)
    from_zero = history.alphas[-3] == 0
    r0 = jnp.where(from_zero, 1.0, (lambdas[-1] - lambdas[-3]) / h)
    r1 = jnp.where(from_zero, 0.5, (lambdas[-2] - lambdas[-3]) / h)
    d1_0 = (m1 - m2) / r1
    d1_1 = (m0 - m2) / r0
    if solver.solver_type == "midpoint":
        stepped = ax + b * m2 + c_heun * d1_1
    else:
        d1 = (r0 * d1_0 - r1 * d1_1) / (r0 - r1)
        d2 = 2.0 * (d1_1 - d1_0) / (r0 - r1)
        stepped = ax + b * m2 + c_heun * d1 + c_2 * d2
    if solver.algorithm.startswith("sde"):
        stepped = stepped + dz
    endpoint_clean = m0
    if solver.solver_type == "heun" and solver.algorithm == "dpmsolver++":
        # Combine D1 and D2 before taking the infinite-anchor limit.
        # Their individually divergent leading terms cancel.
        gap = _half_log_snr(alpha_t, jnp.where(sigma_t == 0, 1.0, sigma_t)) - lambdas[-1]
        endpoint_clean = m0 + (gap - 1.0) * (m0 - m1) / (lambdas[-1] - lambdas[-2])
    return jnp.where(from_zero, source_limit(3, endpoint_clean), stepped)


def _deis_second(t, b, c):
    """Integral of the log-rho Lagrange basis, including t=0 and a node at infinity."""
    infinite_b, infinite_c = jnp.isinf(b), jnp.isinf(c)
    log_b = jnp.log(jnp.where(infinite_b, 1.0, b))
    log_c = jnp.log(jnp.where(infinite_c, 1.0, c))
    log_t = jnp.log(jnp.where(t == 0, 1.0, t))
    denominator = jnp.where(infinite_b | infinite_c, 1.0, log_b - log_c)
    integral = t * (-log_c + log_t - 1) / denominator
    return jnp.where(infinite_b, 0.0, jnp.where(infinite_c, t, integral))


def _deis_third(t, b, c, d):
    """Integral of the quadratic log-rho basis; an infinite node has zero
    weight, and each other basis loses its corresponding factor."""
    infinite_b, infinite_c, infinite_d = jnp.isinf(b), jnp.isinf(c), jnp.isinf(d)
    lb = jnp.log(jnp.where(infinite_b, 1.0, b))
    lc = jnp.log(jnp.where(infinite_c, 1.0, c))
    ld = jnp.log(jnp.where(infinite_d, 1.0, d))
    lt = jnp.log(jnp.where(t == 0, 1.0, t))
    numerator = t * (lc * (ld - lt + 1) - ld * lt + ld + lt ** 2 - 2 * lt + 2)
    denominator = jnp.where(infinite_b | infinite_c | infinite_d, 1.0, (lb - lc) * (lb - ld))
    integral = numerator / denominator
    integral = jnp.where(infinite_c, _deis_second(t, b, d), integral)
    integral = jnp.where(infinite_d, _deis_second(t, b, c), integral)
    return jnp.where(infinite_b, 0.0, integral)


@dataclass(frozen=True)
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
    """

    order: int = 2
    lower_order_final: bool = True

    def __post_init__(self) -> None:
        if self.order not in (1, 2, 3):
            raise ValueError(f"DEIS has orders 1, 2 and 3, not {self.order}")

    def init(self, x, times, process, *, key):
        return _multistep(x, times, self.order)

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        alpha_s0, sigma_s0 = process.rates(t, like=x)
        alpha_t, sigma_t = process.rates(t_next, like=x)
        history = state.push(eps, alpha_s0, sigma_s0)
        terminal, h = _lambda_step(alpha_s0, sigma_s0, alpha_t, sigma_t)
        rho_t = sigma_t / alpha_t
        rhos = history.sigmas / history.alphas
        rho_s0, rho_s1 = rhos[-1], rhos[-2]
        m0, m1 = history.outputs[-1], history.outputs[-2]

        def first(_):
            safe_alpha_s0 = jnp.where(alpha_s0 == 0, 1.0, alpha_s0)
            regular = (alpha_t / safe_alpha_s0) * x - sigma_t * (jnp.exp(h) - 1.0) * m0
            return jnp.where((alpha_s0 == 0) | terminal, alpha_t * denoised + sigma_t * eps, regular)

        def second(_):
            coef1 = _deis_second(rho_t, rho_s0, rho_s1) - _deis_second(rho_s0, rho_s0, rho_s1)
            coef2 = _deis_second(rho_t, rho_s1, rho_s0) - _deis_second(rho_s0, rho_s1, rho_s0)
            return alpha_t * (x / alpha_s0 + coef1 * m0 + coef2 * m1)

        def third(_):
            rho_s2, m2 = rhos[-3], history.outputs[-3]
            coef1 = (_deis_third(rho_t, rho_s0, rho_s1, rho_s2)
                     - _deis_third(rho_s0, rho_s0, rho_s1, rho_s2))
            coef2 = (_deis_third(rho_t, rho_s1, rho_s2, rho_s0)
                     - _deis_third(rho_s0, rho_s1, rho_s2, rho_s0))
            coef3 = (_deis_third(rho_t, rho_s2, rho_s0, rho_s1)
                     - _deis_third(rho_s0, rho_s2, rho_s0, rho_s1))
            return alpha_t * (x / alpha_s0 + coef1 * m0 + coef2 * m1 + coef3 * m2)

        order = _tapered_order(self.order, history, self.lower_order_final, euler_at_final=False)
        stepped = lax.switch(order - 1, [first, second, third][:self.order], None)
        return stepped, history.advance()
