"""What `reference_error.float64_terms` takes for a leaf's terms: the root sum
of squares of the terms its final sums add, carried into it, and nothing behind them;
and the attention oracle's float64 truth held to an exact evaluation."""

import jax
import jax.numpy as jnp
import mpmath
import numpy as np
import pytest
from attention_oracle import anchored_softmax
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh, PartitionSpec as P
from reference_error import float64_terms

from dew.accuracy import reduction_bound, within


def loss_of(keys, query, values):
    scores = jnp.einsum("bd,bkd->bk", query, keys)
    return jnp.sum(jax.nn.softmax(scores, axis=-1)[..., None] * values)


def test_a_key_bias_gradient_sums_terms_that_cancel():
    """A key bias shifts every logit of a query's softmax alike, so its
    gradient is zero in exact arithmetic, while the cotangents it sums are
    not: its terms are their root sum of squares over the rows and keys, as
    the bias's gradient sums them."""
    rng = np.random.default_rng(0)
    query, keys, values = (rng.normal(size=shape) for shape in ((2, 4), (2, 6, 4), (2, 6, 3)))

    with jax.enable_x64(new_val=True):
        gradient, terms = float64_terms(jax.grad(lambda bias: loss_of(keys + bias, query, values)),
                                        jnp.zeros(4))
        cotangent = np.asarray(jax.grad(loss_of)(jnp.asarray(keys), query, values))
    np.testing.assert_allclose(gradient, 0, atol=1e-15)
    np.testing.assert_allclose(terms, np.sqrt(np.sum(cotangent**2, axis=(0, 1))), rtol=1e-12)
    assert np.all(terms > 1e-3)


def test_a_sum_of_one_sign_has_its_own_size_and_an_update_its_steps_terms():
    """A weight's gradient of positive inputs sums positive terms, whose root
    sum of squares is below the sum itself. Through a scan of momentum
    steps, the momentum's terms are its gradients' and a parameter's its
    updates', not the value it started at."""
    x = np.abs(np.random.default_rng(1).normal(size=(8, 3)))

    def gradient(weight):
        return jax.grad(lambda w: jnp.sum(x @ w))(weight)

    def steps(weight):
        def step(carry, _):
            weight, trace = carry
            trace = 0.9 * trace + gradient(weight)
            return (weight - 0.1 * trace, trace), None
        return jax.lax.scan(step, (weight, jnp.zeros_like(weight)), None, length=2)[0]

    with jax.enable_x64(new_val=True):
        weight = jnp.ones((3, 2))
        value, terms = float64_terms(gradient, weight)
        _, (moved_terms, trace_terms) = float64_terms(steps, weight)
    energy = np.sum(x**2, axis=0)[:, None] * np.ones((1, 2))
    np.testing.assert_allclose(terms, np.sqrt(energy), rtol=1e-12)
    assert np.all(terms < np.asarray(value))
    # The momentum's second value is 0.9 of the first gradient and all of the second.
    np.testing.assert_allclose(trace_terms, np.sqrt((0.81 + 1) * energy), rtol=1e-12)
    # The weight moved by 0.1 of each step's momentum; the value it started at carries none.
    np.testing.assert_allclose(moved_terms, np.sqrt(0.01 * (1 + 0.81 + 1) * energy), rtol=1e-12)


def test_a_recomputed_body_sums_its_own_terms():
    """A body under jax.checkpoint, as a recomputed norm's, is the computation's own: a key bias's
    gradient computed inside one sums the same cancelling terms as outside it, not its own near-zero
    size."""
    rng = np.random.default_rng(0)
    query, keys, values = (rng.normal(size=shape) for shape in ((2, 4), (2, 6, 4), (2, 6, 3)))
    gradient = jax.grad(lambda bias: loss_of(keys + bias, query, values))

    with jax.enable_x64(new_val=True):
        _, plain = float64_terms(gradient, jnp.zeros(4))
        _, recomputed = float64_terms(jax.checkpoint(gradient), jnp.zeros(4))
    np.testing.assert_allclose(recomputed, plain, rtol=1e-12)
    assert np.all(recomputed > 1e-3)


def test_a_named_sum_carries_its_terms_through_a_residual_add():
    """A block names its projection's output for a remat policy
    (`checkpoint_name`) before the residual add. The name moves the sum, so
    the stream's terms are the projection's whatever the stream holds: an
    entry where the add cancels keeps the projection's terms, not its own
    near-zero size (FrontDamselfly, MiMo's vision tower)."""
    rng = np.random.default_rng(2)
    x, weight = rng.normal(size=(4, 8)), rng.normal(size=(8, 3))

    def block(stream, x, weight):
        return stream + checkpoint_name(x @ weight, "down_proj")

    with jax.enable_x64(new_val=True):
        stream = -(x @ weight)
        stream[0, 0] += 1e-9
        value, terms = float64_terms(block, stream, x, weight)
    np.testing.assert_allclose(terms, np.sqrt(np.square(x) @ np.square(weight)), rtol=1e-12)
    assert abs(float(value[0, 0])) < 1e-8 < float(terms[0, 0])


def test_a_grouped_matmul_sums_each_groups_terms_as_a_dot_does():
    """An expert layer's grouped matmul (`jax.lax.ragged_dot`) and its kernel's gradient sum their
    rows' products within each group: their terms are each group's dense dot's, not their own
    near-zero sizes where the rows cancel (the MoE layout case's expert kernels)."""
    rng = np.random.default_rng(3)
    rows, groups = rng.normal(size=(6, 4)), np.array([2, 4], np.int32)
    kernel = rng.normal(size=(2, 4, 3))
    with jax.enable_x64(new_val=True):
        _, terms = float64_terms(lambda x, w: jax.lax.ragged_dot(x, w, jnp.asarray(groups)), rows, kernel)
        _, gradient_terms = float64_terms(jax.grad(lambda w, x, probe: jnp.sum(
            jax.lax.ragged_dot(x, w, jnp.asarray(groups)) * probe)), kernel, rows, -np.ones((6, 3)))
    dense = np.split(rows, np.cumsum(groups)[:-1])
    np.testing.assert_allclose(terms, np.concatenate([
        np.sqrt(np.square(part) @ np.square(kernel[group])) for group, part in enumerate(dense)]), rtol=1e-12)
    np.testing.assert_allclose(gradient_terms, np.stack([
        np.sqrt(np.square(part).T @ np.ones((len(part), 3))) for part in dense]), rtol=1e-12)


@pytest.mark.mesh(devices=2)
def test_a_per_device_program_sums_its_terms_and_its_devices_add_them():
    """A shard_map's program is evaluated op by op on each device's rows, as an
    expert layer's dispatch runs (`dew.nn.moe`): its dot has the dense dot's
    terms, and its kernel's gradient, each device's rows summed and the map's
    transpose adding them across devices (`psum`), has every row's, as one
    device summing them all would, not its own size."""
    rng = np.random.default_rng(5)
    rows, kernel, probe = rng.normal(size=(8, 4)), rng.normal(size=(4, 3)), rng.normal(size=(8, 3))
    mapped = jax.shard_map(jnp.matmul, mesh=Mesh(np.array(jax.devices()[:2]), ("data",)),
                           in_specs=(P("data"), P()), out_specs=P("data"), check_vma=False)

    with jax.enable_x64(new_val=True):
        _, terms = float64_terms(mapped, rows, kernel)
        _, gradient_terms = float64_terms(jax.grad(lambda w, x: jnp.sum(mapped(x, w) * probe)), kernel, rows)
    np.testing.assert_allclose(terms, np.sqrt(np.square(rows) @ np.square(kernel)), rtol=1e-12)
    np.testing.assert_allclose(gradient_terms, np.sqrt(np.square(rows).T @ np.square(probe)), rtol=1e-12)


# The attention oracle's float64 truth (`attention_oracle.anchored_softmax`) against an exact evaluation
# (mpmath at 60 digits): its softmax derivative about each row's largest weight is within its float64
# rounding bound of exact, where the plain derivative cancels below float64's ulp (logits near 1e4).

DIGITS = 60
"""The exact evaluation's working precision."""


def exact_gradient(logits: np.ndarray, cotangent: np.ndarray) -> np.ndarray:
    """The softmax's VJP of one row in 60-digit arithmetic: `p (g - sum p g)`, rounded to float64."""
    mpmath.mp.dps = DIGITS
    terms = [mpmath.mpf(float(x)) for x in logits]
    top = max(terms)
    weights = [mpmath.exp(x - top) for x in terms]
    total = mpmath.fsum(weights)
    p = [w / total for w in weights]
    g = [mpmath.mpf(float(x)) for x in cotangent]
    mean = mpmath.fsum(pi * gi for pi, gi in zip(p, g, strict=True))
    return np.asarray([float(pi * (gi - mean)) for pi, gi in zip(p, g, strict=True)])


def rows(case: str, count: int):
    """`count` rows of 16 logits and cotangents, float64: logits near 1e4 (weights spanning 1 to 1e-22
    and below), standard-normal ones, or rows whose largest logit is a tiny one beside -700 (`argument`:
    the subtraction `-700 - 2^-45` rounds to -700, so the exponential's argument, not its own rounding,
    carries the error)."""
    streams = jax.random.split(jax.random.key({"extreme": 0, "ordinary": 1, "argument": 2}[case]), 2)
    with jax.enable_x64(new_val=True):
        logits = np.asarray(jax.random.normal(streams[0], (count, 16), jnp.float64))
        cotangents = np.asarray(jax.random.normal(streams[1], (count, 16), jnp.float64))
    if case == "extreme":
        logits = logits * 1e4
        # Each row's runner-up 50 below its largest, a weight near 1e-22 beside one near 1.
        logits[:, 1] = logits.max(axis=1) - 50.0 - np.arange(count) / count
    if case == "argument":
        logits = np.full((count, 16), -1000.0) - np.arange(count)[:, None]
        logits[:, 0], logits[:, 1] = 2.0 ** -45 * (1 + np.arange(count)), -700.0
        cotangents = np.zeros((count, 16))
        cotangents[:, 1] = 1.0
    return logits, cotangents


def anchored_bound(logits: np.ndarray, cotangent: np.ndarray) -> np.ndarray:
    """The anchored form's float64 rounding bound, entrywise, `u` float64's unit roundoff and
    `r = g - g_a`. Each exponent `x_i = l_i - l_a` rounds by at most `u |x_i|`, which moves `exp(x_i)` by
    that relative amount, and the exponential rounds by at most `2u` (`eta_i`, enclosed by `expm1`); the
    denominator's sum adds `gamma_n` and carries the weights' errors (`tau`), and the division `u`, so each
    weight is within `delta_i = eta_i + 2 tau + u` relatively. The gradient `p_i (r_i - sum_j p_j r_j)`
    then errs by at most `p_i (delta_i (|r_i| + S) + sum_j p_j |r_j| delta_j)`, `S = sum_j p_j |r_j|`, plus
    its own operations' rounding, `n + 4` of them (the subtraction, the products and the sum): the gate's
    reduction bound in float64 (`reduction_bound`; Higham, Accuracy and Stability of Numerical
    Algorithms, 3.1 and 4.2)."""
    unit = np.finfo(np.float64).eps / 2
    exponents = logits - logits.max()
    weights = np.exp(exponents)
    weights /= weights.sum()
    eta = np.expm1(unit * np.abs(exponents)) * (1 + 2 * unit) + 2 * unit
    steps = logits.size
    tau = steps * unit / (1 - steps * unit) + np.sum(weights * eta)
    delta = eta + 2 * tau + unit
    relative = np.abs(cotangent - cotangent[np.argmax(logits)])
    spread = np.sum(weights * relative)
    terms = weights * (relative + spread)
    carried = weights * (delta * (relative + spread) + np.sum(weights * relative * delta))
    return carried + reduction_bound(terms, logits.size + 4, np.float64)


@pytest.mark.parametrize("case", ["extreme", "ordinary", "argument"])
def test_the_anchored_truth_is_within_its_bound_and_never_further_than_the_plain_one(case):
    """Each anchored entry is within its float64 rounding bound of exact (`anchored_bound`), and never
    further from exact than the plain form's entry except within that bound; on the extreme rows the
    plain form loses the anchor's gradient, the runner-up's share, past it."""
    logits, cotangents = rows(case, 32)
    with jax.enable_x64(new_val=True):
        def gradient(softmax, logit, cotangent):
            return np.asarray(jax.vjp(softmax, jnp.asarray(logit))[1](jnp.asarray(cotangent))[0])

        plain = np.asarray([gradient(jax.nn.softmax, x, g) for x, g in zip(logits, cotangents, strict=True)])
        anchored = np.asarray([gradient(anchored_softmax, x, g)
                               for x, g in zip(logits, cotangents, strict=True)])
    exact = np.asarray([exact_gradient(x, g) for x, g in zip(logits, cotangents, strict=True)])
    bound = np.asarray([anchored_bound(x, g) for x, g in zip(logits, cotangents, strict=True)])
    anchored_errors, plain_errors = np.abs(anchored - exact), np.abs(plain - exact)
    assert within(anchored_errors, bound)
    assert within(anchored_errors, np.maximum(plain_errors, bound))
    if case == "extreme":
        # Where an anchor's exact gradient is past the bound, a share float64 resolves, the plain form
        # cancels it away and the anchored keeps it.
        anchors = list(zip(range(32), np.argmax(logits, axis=1), strict=True))
        resolved = [at for at in anchors if not within(exact[at], 2 * bound[at])]
        assert resolved
        assert not any(within(plain_errors[at], bound[at]) for at in resolved)
