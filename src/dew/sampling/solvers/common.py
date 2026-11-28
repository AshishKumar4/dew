"""The solver step contract, endpoint checks and history arithmetic shared by concrete solvers."""

from __future__ import annotations

import functools
from collections.abc import Iterable, Sequence
from typing import NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from typing_extensions import TypeVar

from dew.diffusion.process import Process
from dew.diffusion.schedules import GeneralizedNoiseScheduler
from dew.nn.derivatives import straight_through
from dew.nn.kernels.generation import linear_map
from dew.nn.precision import at_least_fp32, ordered_two_sum, power2, summed_once, two_product
from dew.nn.programs import on_host

# tests/test_samplers.py checks the Diffusers-named solvers' trajectories and
# trajectory gradients against fixtures recorded by tools/diffusers_reference.py;
# tools/diffusers_limits_reference.py verifies the finite endpoint limits.

# A solver whose state nobody names is a solver over any state: `StateT` is
# covariant, so `Solver` written bare is the type every concrete solver
# satisfies, and a call site that carries the state names it.
StateT = TypeVar("StateT", covariant=True, default=object)


def given(value: float | int | jax.Array | jax.stages.ArgInfo) -> bool:
    """Whether a struct's data leaf `value` was given as a value its `__post_init__` checks, not handed it by
    a transformation: jit's tracer, an array, or the description a lowered program's arguments hold
    (`jax.stages.ArgInfo`), which `jax.jit(...).lower` builds the struct over and Dew's programs lower
    (`dew.nn.kernels.options.program`)."""
    return not isinstance(value, jax.Array | jax.stages.ArgInfo)


class Solver(Protocol[StateT]):
    """The interface for one diffusion solver step and the state it keeps between steps.

    `StateT` is that state: nothing for a one-step solver, the previous model
    outputs for a multistep one. It is a type parameter, so each solver's own
    state type is checked at its call sites.

    A solver is a pytree, as a pipeline call takes it as data: a
    `flax.struct.dataclass` whose numbers are fields and whose settings are
    `struct.field(pytree_node=False)`, and a solver wrapping another holds it
    as a field.
    """

    def init(self, x, times, process, *, key) -> StateT:
        """Return the initial state for sampling from `x` over the concrete grid `times`.

        It checks the grid's endpoint domains at compile time. `key` is the
        root key of the whole sampling run, or a `RowSolver`'s `RowKeys`; only
        `DPMSolverSDE` reads it, for its whole-trajectory Brownian tree. Each
        step draws from the folded key passed to it.
        """
        ...

    def step(self, x, t, t_next, denoised, eps, state, key, process,
             denoise, /) -> tuple[jax.Array, StateT]:
        """Return the sample at `t_next` and the new state, from `x` and the model's `(denoised, eps)` at `t`.

        `sample` passes every argument by position, so a solver over another
        algebra can name the pair for what it reads: the discrete solver takes
        log-probabilities where a Gaussian one takes eps.
        """
        ...


@struct.dataclass
class RowKeys:
    """One typed PRNG key per row of a batch, `keys` `[B]`, for a solver whose draws are row by row
    (`RowSolver`): a row's draws are its own key's, whatever rows sample beside it, as they are
    sampled alone, while the model still runs once over the batch."""

    keys: jax.Array


@runtime_checkable
class RowSolver(Solver[StateT], Protocol[StateT]):
    """A solver that draws each row by its own key (`RowKeys`): `sample` hands `step_rows` the
    step's row keys where it hands `step` one key."""

    def step_rows(self, x, t, t_next, denoised, eps, state, keys: jax.Array, process,
                  denoise, /) -> tuple[jax.Array, StateT]:
        """`step`, row `i` drawing from `keys[i]` alone."""
        ...


def _check_endpoint_domain(process: Process, times: jax.Array, *,
                           source: bool = False, target: bool = False, reason: str) -> None:
    """Reject undefined endpoint integrals while the sampling grid is concrete.

    This runs at initialization, including compile-time initialization in
    sample(), rather than introducing host callbacks into a solver step.
    """
    if times.shape[0] < 2:
        return
    with on_host():
        alpha, sigma = process.sampler_schedule.rates(times)
        if source and bool(jnp.any(alpha[:-1] == 0)):
            raise ValueError("alpha=0 source has no finite update: " + reason)
        if target and bool(jnp.any(sigma[1:] == 0)):
            raise ValueError("sigma=0 target has no finite update: " + reason)


def _grid_levels(process: Process, times: jax.Array) -> tuple[np.ndarray, np.ndarray]:
    """The concrete grid's alpha and sigma at each point, on the host, as a walk that steps in them
    reads them in float32."""
    with on_host():
        alpha, sigma = process.sampler_schedule.rates(jnp.asarray(times))
        return np.asarray(alpha), np.asarray(sigma)


def _solved_levels(process: Process, times: jax.Array) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The concrete grid's alpha, sigma and lambda = log(alpha) - log(sigma) at each point, in float64
    on the host: what a solver whose weights depend on the grid alone solves them from, once, before
    the walk, so the walk's own arithmetic is its weighted sums alone (`_weighted`). The rates are the
    ones the walk's model reads (the schedule's, at the walk's precision), widened: weights solved from
    them cancel as the model's outputs at those rates do, where exact rates the model never reads
    would not. In float32 the walk's own lambda from a log-SNR of -5 rounded to 2.4e-7, which the
    noise-prediction updates amplified (rows 13650 to 13653)."""
    alpha, sigma = (rate.astype(np.float64) for rate in _grid_levels(process, times))
    with np.errstate(divide="ignore"):
        return alpha, sigma, np.log(alpha) - np.log(sigma)


def _split(weights: np.ndarray, dtype) -> np.ndarray:
    """Float64 `weights` as `dtype` triples, `[..., 3]`: each weight over a power of two, `2^e`, rounded,
    then what the rounding left, rounded again, and `e`. Weighed by the pair and its power (`_sum`), a term
    carries its weight to about twice `dtype`'s precision: a weight rounded once to float32 errs by up to
    half an ulp, the same for every entry of the tensor it scales, an error no elementwise rounding averages
    away, and the walk's float32 reference's own weights, rounded inside its walk, err by as much on either
    side, cancelling at some steps by chance (rows 13650 to 13653). `e` is zero for a weight a normal pair
    holds, else its binade, the pair then its significand, so a weight below the format's normals keeps its
    digits where a device flushes subnormals (DPM-Solver's history weight of a variance-exploding grid at
    64, 16 and 12 times 2^-126, about -0.4 2^-126 of the history's difference; train 1461's P0)."""
    format_ = np.finfo(dtype)
    tiny = (weights != 0) & (np.abs(weights) < 2.0 ** (float(format_.minexp) + float(format_.nmant) + 2))
    _, exponent = np.frexp(np.where(tiny, weights, 1.0))
    powers = np.where(tiny, exponent - 1, 0)
    scaled = np.ldexp(weights, -powers)
    high = scaled.astype(dtype)
    return np.stack([high, (scaled - high.astype(np.float64)).astype(dtype), powers.astype(dtype)], axis=-1)


def _scaled(weight: jax.Array, term: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """A row's `weight` (`_split`'s triple) and `term` as `_sum` multiplies them: the weight's power of two
    `2^e` split between them, `term` taking as much of it as keeps it a normal value
    whose product with the pair stays finite, the pair the rest, so neither factor nor their product leaves
    the range where the weighed term does not: a weight below the normals times a term of the normals' size
    (1.5 2^-126 times 0.75 is 1.125 2^-126, where 0.75 2^-126 alone flushes; train 1461's review), a term
    near the float's largest under a weight below one. Returns the pair's two parts and the term, each scaled
    by an exact power of two."""
    format_ = jnp.finfo(term.dtype)
    power = weight[2].astype(jnp.int32)
    _, binade = jnp.frexp(term)
    # term 2^k is normal where its binade plus k reaches the least normal's, and times the pair (under 2)
    # finite where it stays a binade under the largest's.
    least = int(format_.minexp) + 1 - binade
    most = int(format_.maxexp) - 1 - binade
    # A weight the pair holds whole (`e` zero) weighs as it is.
    taken = jnp.where((term == 0) | (power == 0), power, jnp.clip(power, least, most))
    rest = power - taken
    return power2(weight[0], rest), power2(weight[1], rest), power2(term, taken)


class Moved(NamedTuple):
    """A shifted difference x - shift m (`_moved`) as `_weighted` weighs it beside its partner m: `whole` as
    the walk takes it, rounded, and `low`, what its product's and its difference's rounding dropped
    (`two_product`, `two_sum`), weighed beside it, so a
    shift that rounds x away keeps it (2^-26 less 1 is -1 and 2^-26: train 1389's P0); both zero where the
    difference is past the float's range though x and m are not (`over`, `finite`), there taken unshifted, a
    x + b m, x weighed by the difference's weight and m by its own unshifted weight (`_shifted`'s b) where its
    row shifted them (`shift`, its column): `x`, the column of m among the step's terms (`partner`) and of b
    in its row (`kept`)."""

    whole: jax.Array
    low: jax.Array
    x: jax.Array
    over: jax.Array
    finite: jax.Array
    shift: int
    partner: int
    kept: int


class Apart(NamedTuple):
    """A history's difference a - b (`_apart`) as `_weighted` weighs it: `whole` as the walk takes it, zero
    where it is past the float's range though a and b are not (`over`, `finite`); apart, w a - w b, where the
    step taken whole is (a step's two outputs at the float's extremes: train 1379's P-F1)."""

    whole: jax.Array
    a: jax.Array
    b: jax.Array
    over: jax.Array
    finite: jax.Array


Term = jax.Array | Moved | Apart
"""A term a solved step weighs (`_weighted`)."""


def _apart(a, b) -> Apart:
    """a - b as `_weighted` weighs it (`Apart`)."""
    whole = a - b
    finite = jnp.isfinite(a) & jnp.isfinite(b)
    over = ~jnp.isfinite(whole) & finite
    return Apart(jnp.where(over, jnp.zeros_like(whole), whole), a, b, over, finite)


def _sum(row: jax.Array, terms: Sequence[jax.Array],
         residuals: Sequence[tuple[int, jax.Array]] = ()) -> jax.Array:
    """The sum of `terms` each weighed by its entry of `row` (`[columns, 3]`, each weight as the triple
    `_split` makes), compensated (`summed_once`, Dot2): each rounded weight's product and each sum's rounding
    carried and added once, with the weights' residuals' products and `residuals`, what some terms' own
    rounding dropped, each beside its term's column. A term the row weighs by zero is not read, so an operand
    the step does not take (an unused conversion's overflow, a history slot not yet filled) adds nothing, not
    its zero product's NaN. A plain float32 sum rounds each term onto the grid of the largest so far, where a
    shifted difference's rounding is lost (`unipc.default`, m's weight several times x's: its third step erred
    1.498 times the reference's from float64 summed plainly, 1.434 compensated, pooled over its 40 seeds)."""
    scaled = [_scaled(row[index], jnp.where(row[index, 0] == 0, jnp.zeros_like(term), term))
              for index, term in enumerate(terms)]
    small = scaled[0][1] * scaled[0][2]
    for _, low, term in scaled[1:]:
        small = small + low * term
    for index, residual in residuals:
        high, _, held = _scaled(row[index], jnp.where(row[index, 0] == 0, jnp.zeros_like(residual), residual))
        small = small + high * held
    return summed_once([small], [(high, term) for high, _, term in scaled])


def _weighted(row: jax.Array, terms: Sequence[Term]) -> jax.Array:
    """The sum of `terms` each weighed by its entry of `row` (`_sum`): one step of a solver whose weights
    are solved before the walk (`_solved_levels`). A shifted difference (`Moved`, its rounding beside it)
    and a history's (`Apart`) are weighed whole, and where one is past the float's range, or the step taken
    so is, though their operands are finite, the step is taken plain: each shifted one
    unshifted, a x + b m (`Moved`), each history's difference apart, w a - w b, where it alone is past the
    float's range, and whole where it is not; the weights a walk without them takes, so no step that walk
    takes finite overflows or loses x for them (train 1362's P-F2, 1373's F1, 1379's P-F1, 1389's P0s)."""
    # Each shifted difference's rounding weighed beside it, among the weights' residuals (`Moved.low`).
    total = _sum(row, [term if isinstance(term, jax.Array) else term.whole for term in terms],
                 [(index, term.low) for index, term in enumerate(terms) if isinstance(term, Moved)])
    # The terms the row weighs: a history slot it weighs by zero is none of the step's (`_sum`).
    special = [(term, row[index, 0] != 0) for index, term in enumerate(terms)
               if not isinstance(term, jax.Array)]
    if not special:
        return total
    # The row's weights of the terms alone, its further columns (a shift, a kept weight) none of them.
    # Taken so too where a weighed product of the step taken whole reads as zero, a subnormal a device
    # flushes, that the plain step does not form (`_flushed`): a shifted difference's, or its partner's under
    # its shifted weight (0.5 times 1.5 2^-126, where the plain step weighs 2^-124 by 2 and 1.5 2^-126 by -1;
    # train 1452's P0), or a history difference's, then weighed apart (-0.11 times -4 2^-126, a DEIS step's,
    # whose operands its weight keeps normal; its review).
    plain_row, plain, apart, weights = row[:len(terms)], [], [], []
    flushed = jnp.zeros(jnp.shape(total), bool)
    for index, term in enumerate(terms):
        if isinstance(term, Moved):
            # Unshifted where its row shifted; an endpoint's row (a pure-noise source, a clean target)
            # weighs its terms as they are.
            shifted = row[term.shift, 0] != 0
            plain.append(jnp.where(shifted, term.x, term.whole))
            plain_row = plain_row.at[term.partner].set(jnp.where(shifted, row[term.kept], row[term.partner]))
            flushed = flushed | _flushed(row, index, term.whole)
            if isinstance(partner := terms[term.partner], jax.Array):
                flushed = flushed | _flushed(row, term.partner, partner)
        elif isinstance(term, Apart):
            # Apart where it alone is past the float's range, or its weighed difference flushes, or the
            # difference itself does; a finite difference whole elsewhere (train 1389's P0: a shifted term's
            # overflow expands no difference that does not).
            vanished = (term.whole == 0) & (term.a != term.b) & term.finite & (row[index, 0] != 0)
            lost = _flushed(row, index, term.whole) | vanished
            flushed = flushed | lost
            expanded = term.over | lost
            plain.append(jnp.where(expanded, term.a, term.whole))
            apart.append(jnp.where(expanded, term.b, jnp.zeros_like(term.b)))
            weights.append(row[index].at[:2].multiply(-1))
        else:
            plain.append(term)
    if apart:
        plain_row = jnp.concatenate([plain_row, jnp.stack(weights)])
    over = functools.reduce(jnp.logical_or, [term.over & active for term, active in special])
    finite = functools.reduce(jnp.logical_and, [term.finite | ~active for term, active in special])
    stepped = jnp.where((over | flushed | ~jnp.isfinite(total)) & finite, _sum(plain_row, [*plain, *apart]),
                        total)
    # Its tangent the step's on its operands as the row weighs them, each shifted difference unshifted (a
    # difference's tangent formed before it is weighed passes the float's range where the step's does not:
    # x = m = 1, their tangents 2^127 and -2^127; train 1430's P0) and each history's difference weighed by
    # a map of its own (`_weighed_apart`).
    return straight_through(_sum(*_unshifted(row, terms)), stepped)


def _flushed(row: jax.Array, index: int, term: jax.Array) -> jax.Array:
    """Whether the row's weight of its column `index` times `term`, neither zero, reads as zero as `_sum`
    forms it (`_scaled`): a subnormal product a device flushes."""
    high, _, scaled = _scaled(row[index], term)
    return (row[index, 0] != 0) & (term != 0) & (high * scaled == 0)


def _unshifted(row: jax.Array, terms: Sequence[Term]) -> tuple[jax.Array, list[jax.Array]]:
    """The row and terms of a step with every shifted difference (`Moved`) unshifted and every history's
    (`Apart`) weighed by its own map (`_weighed_apart`), its column's weight then one, whatever their
    values: the step's linear form in its operands (`_weighted`), every derivative of it the step's own."""
    weighed, read = row[:len(terms)], []
    for index, term in enumerate(terms):
        if isinstance(term, Moved):
            read.append(term.x)
            shifted = row[term.shift, 0] != 0
            weighed = weighed.at[term.partner].set(jnp.where(shifted, row[term.kept], row[term.partner]))
        elif isinstance(term, Apart):
            read.append(_weighed_apart(row[index], term.a, term.b))
            weighed = weighed.at[index].set(jnp.stack([jnp.ones_like(row[index, 0]),
                                                       jnp.zeros_like(row[index, 1]),
                                                       jnp.zeros_like(row[index, 2])]))
        else:
            read.append(term)
    return weighed, read


def _weighed_apart(weight: jax.Array, a: jax.Array, b: jax.Array) -> jax.Array:
    """A history's difference a - b weighed by `weight` (its row's pair, `_split`'s), a map linear in a and b
    (`linear_map`) whose tangent and transpose are each in range wherever the weighed difference is: the
    difference formed whole, then weighed, where it is in range (tangents of -1.5 2^-126 and 1.5 2^-126 under
    a weight of 11, where halving each first flushes both; train 1452's P0), and taken at half scale where it
    is not (two of 2^127 and -2^127 under a weight below a half); pulled back as each operand's weight, which
    forms no difference. Zero where the row weighs it by zero, as `_sum` reads no term it weighs so."""
    return linear_map(_apart_applied, _apart_pulled, weight, (a, b))


def _apart_applied(weight: jax.Array, moving: tuple[jax.Array, jax.Array]) -> jax.Array:
    a, b = moving
    whole = a - b
    formed = jnp.isfinite(whole)
    held = jnp.where(formed, whole, 0)
    half = 0.5 * jnp.where(formed, 0, a) - 0.5 * jnp.where(formed, 0, b)
    whole_hi, whole_lo, held = _scaled(weight, held)
    half_hi, half_lo, half = _scaled(weight, half)
    weighed = jnp.where(formed, whole_hi * held + whole_lo * held, (half_hi * half + half_lo * half) * 2)
    return jnp.where(weight[0] == 0, jnp.zeros_like(weighed), weighed)


def _apart_pulled(weight: jax.Array, cotangent: jax.Array) -> tuple[jax.Array, jax.Array]:
    hi, lo, held = _scaled(weight, cotangent)
    pulled = jnp.where(weight[0] == 0, jnp.zeros_like(cotangent), hi * held + lo * held)
    return pulled, -pulled


def _shifted(a: float, b: float, shift: float) -> tuple[float, float, float, float]:
    """The weights of x - shift m and of m that weigh x by `a` and m by `b`, so a x + b m is
    a (x - shift m) + (b + a shift) m. With `shift` the rate the model read m at (sigma for eps, alpha
    for the clean prediction) x - shift m is the walk's cancellation of the sample's noise or signal,
    taken in the model's own rate, and the weight left on m is the target's rate there: a first-order
    exponential step is a (x - sigma_s eps) + sigma_t eps, or a (x - alpha_s x_0) + alpha_t x_0. A rounded
    a then scales what is left of x, not x itself. Returns the weights of x - shift m (`_moved`) and of m,
    the shift, and b, m's weight in the step unshifted (`Moved.kept`); a step that moves x by nothing (b 0,
    an interval of no length) keeps x as it is, unshifted, where x - shift m and back would round x away."""
    if b == 0:
        return a, b, 0.0, b
    return a, b + a * shift, shift, b


def _moved(x, output, row: jax.Array, shift: int, partner: int, kept: int) -> Moved:
    """x - shift output (`_shifted`; `shift` its row's column) as `_weighted` weighs it, beside `output` as
    the step's term `partner`, whose unshifted weight is the row's column `kept`: taken whole, as the walk
    takes it, and unshifted where the whole is past the float's range though x and output are not (2^127
    less -2^127 is past float32's, while the step itself need not be; train 1362's P-F2)."""
    # The shift's product and the difference each rounded once, what both dropped carried beside it
    # (`two_product`, `two_sum`): exact where no product or sum passes the float's range.
    high, _, scaled = _scaled(row[shift], output)
    moving, dropped = two_product(high, scaled, bit_split=True)
    # The larger operand first (`ordered_two_sum`): a difference landing near the float's largest from
    # operands of other signs keeps its rounding, where TwoSum's own differences overflow (train 1430's
    # P0: -1.5 2^104 less -FLT_MAX, its -2^103 dropped).
    whole, low = ordered_two_sum(x, -moving)
    low = low - dropped
    finite = jnp.isfinite(x) & jnp.isfinite(output)
    # Unshifted too where the difference of two unequal values reads as zero, or the shift's own product
    # does of two nonzero ones: a subnormal a device flushes (2 2^-126 less 2.5 2^-126 halved, -2^-127;
    # train 1423's P0; 0.5 times 1.25 2^-126, train 1430's), whose shifted step would lose it.
    flushed = (moving == 0) & (row[shift, 0] != 0) & (output != 0)
    over = (~jnp.isfinite(whole) | ~jnp.isfinite(low) | ((whole == 0) & (x != moving)) | flushed) & finite
    held = ~over & jnp.isfinite(low)
    return Moved(jnp.where(over, jnp.zeros_like(whole), whole), jnp.where(held, low, jnp.zeros_like(low)), x,
                 over, finite, shift, partner, kept)


def _column(names: tuple[str, ...], name: str) -> int:
    """`name`'s column in a row of weights that leads with `names`."""
    return names.index(name)


def _history_gaps(orders: Sequence[int]) -> list[tuple[int, int]]:
    """The pairs of grid points a multistep update divides by the gap between: at step i of order k,
    each two of the k points it interpolates, i - k + 1 through i."""
    return [(i - m, i - n) for i, k in enumerate(orders) for m in range(k) for n in range(m + 1, k)]


def _check_gaps(levels: np.ndarray, gaps: Iterable[tuple[int, int]], reason: str) -> None:
    """Reject a grid on which the walk divides by a gap of no length, before the walk runs.

    `levels` is the variable the updates take differences in, one per grid
    point at the precision they divide in: the walk's for a solver that
    divides in the walk (`_grid_levels`), so values distinct only where the
    grid was built are caught too, float64 for one that solves its weights
    before it (`_solved_levels`). `gaps` are the pairs of points
    whose difference the updates the walk takes divide by, as each step's
    order selects them. An interval no update divides by is not one of them:
    a first-order step over an empty interval stays where it is, as the
    source's does, while a higher order there is undefined, as the source's
    is.
    """
    if any(levels[first] == levels[second] for first, second in gaps):
        raise ValueError("two points of the grid have one noise level, an interval of no length that "
                         + reason + "; take fewer steps, another spacing or a lower order")


def _sigma_integrator(name: str, process: Process) -> GeneralizedNoiseScheduler:
    schedule = process.sampler_schedule
    if not isinstance(schedule, GeneralizedNoiseScheduler):
        raise ValueError(
            f"{name} integrates dx/dsigma = eps, which holds only when alpha is 1, "
            f"so it needs a GeneralizedNoiseScheduler and not {type(schedule).__name__}")
    return schedule


def _unit_noise(key, x):
    """The unit normal a stochastic step draws from its key at `x`'s shape: in float32 whatever the
    run's precision, so a float64 walk draws a float32 walk's innovations widened, and is its truth."""
    return jax.random.normal(key, x.shape, dtype=jnp.float32)


def _push(buffer, value):
    """`buffer` with its oldest entry dropped and `value` appended, so the
    most recent entry is last."""
    return jnp.concatenate([buffer[1:], value[None]], axis=0)


class Tabled(NamedTuple):
    """What a multistep solver whose weights are solved before the walk carries: the last `depth` model
    outputs, most recent last (slots not yet filled hold zeros, which no row weighs), each step's row of
    weights, and the number of steps taken, which picks the row."""

    outputs: jax.Array
    rows: jax.Array
    taken: jax.Array

    def push(self, output) -> Tabled:
        return self._replace(outputs=_push(self.outputs, output))

    def row(self) -> jax.Array:
        return self.rows[self.taken]

    def advance(self) -> Tabled:
        return self._replace(taken=self.taken + 1)


def _tabled(x, depth: int, rows: np.ndarray) -> Tabled:
    """A walk's first `Tabled` state from `x`: no outputs yet and `rows` split in `x`'s dtype (`_split`),
    so a float64 walk weighs in float64 (ledger 2063)."""
    dtype = at_least_fp32(x.dtype)
    return Tabled(jnp.zeros((depth, *x.shape), x.dtype), jnp.asarray(_split(rows, dtype)),
                  jnp.zeros((), jnp.int32))


def _tapered_orders(order: int, steps: int, lower_order_final: bool, euler_at_final: bool) -> list[int]:
    """The order a Diffusers multistep DPM-Solver takes at each of a walk's `steps`: it grows with the
    history up to `order`; the last step is first order under `euler_at_final`, and in a walk under 15
    steps `lower_order_final` makes the last step first order and caps the one before it at second."""
    short = lower_order_final and steps < 15
    orders = [min(order, taken + 1) for taken in range(steps)]
    if steps and (euler_at_final or short):
        orders[-1] = 1
    if short and steps > 1:
        orders[-2] = min(orders[-2], 2)
    return orders
