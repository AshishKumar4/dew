"""Nested Learning's optimizers (Behrouz et al., arXiv 2512.24695) as Optax transforms: M3, its
Algorithm 1, and delta gradient descent, its eqs. 56-57, each against its paper's equation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from reference_error import assert_as_exact

from dew.optim import delta_gradient_descent, newton_schulz, scale_by_m3

M3_FACTORS = {"beta1": 0.9, "beta2": 0.99, "beta3": 0.5, "alpha": 0.3, "eps": 1e-8, "frequency": 3}
"""M3's factors in the proofs below: three updates a chunk, so seven updates close two chunks."""
NESTED_SEEDS = 8
"""Check 1's declared seeds for M3's directions and delta gradient descent's step."""


def newton_schulz_reference(matrix: np.ndarray, steps: int = 5) -> np.ndarray:
    """Muon's quintic Newton-Schulz in NumPy, in the matrix's own dtype: the matrix over its Frobenius
    norm (plus 1e-8), the wide way round, then `a X + (b A + c A A) X` with `A = X X^T` `steps` times, at
    Jordan's coefficients (modded-nanogpt)."""
    a, b, c = (np.asarray(value, matrix.dtype) for value in (3.4445, -4.7750, 2.0315))
    tall = matrix.shape[0] > matrix.shape[1]
    wide = matrix.T if tall else matrix
    wide = wide / (np.linalg.norm(wide) + np.asarray(1e-8, matrix.dtype))
    for _ in range(steps):
        gram = wide @ wide.T
        wide = a * wide + (b * gram + c * gram @ gram) @ wide
    return wide.T if tall else wide


def test_newton_schulz_orthogonalizes_a_matrix_and_a_stack_as_muons_iteration_does():
    """`newton_schulz` on a tall and a wide matrix, and on a stack of four 6 x 4 matrices whose stack axis
    sits between their reduction and output axes (Muon's dimension numbers), each matrix on par with
    Muon's quintic iteration in NumPy from float64 (check 1), the stack's matrices each as alone."""
    numbers = optax.contrib.MuonDimensionNumbers(reduction_axis=0, output_axis=2)

    @jax.jit
    def orthogonalized(tall, wide, stacked):
        return newton_schulz(tall), newton_schulz(wide), newton_schulz(stacked, dimension_numbers=numbers)

    def reference(tall, wide, stacked):
        return (newton_schulz_reference(tall), newton_schulz_reference(wide),
                np.stack([newton_schulz_reference(stacked[:, index]) for index in range(4)], axis=1))

    def draw(seed):
        rng = np.random.default_rng(seed)
        return tuple(rng.standard_normal(shape).astype(np.float32) for shape in ((6, 4), (4, 6), (6, 4, 4)))

    assert_as_exact(orthogonalized, reference, draw, "Newton-Schulz", seeds=NESTED_SEEDS)


def m3_directions(gradients: np.ndarray) -> np.ndarray:
    """Algorithm 1 of arXiv 2512.24695 in NumPy, in the gradients' own dtype: each update's
    `(NS(M1) + alpha O2) / sqrt(V + eps)`, `M1` and `V` summing `beta1 g` and `beta2 g**2`, and `O2`
    the slow momentum orthogonalized when the chunk before closed, `M2` summing `beta3` times each closed
    chunk's gradients."""
    beta1, beta2, beta3, alpha, eps, frequency = (M3_FACTORS[name] for name in (
        "beta1", "beta2", "beta3", "alpha", "eps", "frequency"))
    fast, second, slow, chunk, slow_direction = (np.zeros_like(gradients[0]) for _ in range(5))
    directions = []
    for update, gradient in enumerate(gradients, 1):
        fast, second = fast + beta1 * gradient, second + beta2 * gradient * gradient
        directions.append((newton_schulz_reference(fast) + alpha * slow_direction) / np.sqrt(second + eps))
        chunk = chunk + gradient
        if update % frequency == 0:
            slow, chunk = slow + beta3 * chunk, np.zeros_like(chunk)
            slow_direction = newton_schulz_reference(slow)
    return np.stack(directions)


def test_m3s_directions_are_algorithm_1s_across_its_chunks():
    """`scale_by_m3` over seven updates of a tree of matrices, one tall and one wide, two chunks closing:
    each direction on par with Algorithm 1's from float64 (check 1), the slow momentum's reaching the
    updates from the second chunk on. Within a chunk the slow momentum and its orthogonalization hold
    still, and each changes once, as the chunk closes; the first chunk's slow direction is zero."""
    transform = scale_by_m3(**M3_FACTORS)

    @jax.jit
    def directions(gradients):
        def step(state, gradient):
            direction, state = transform.update(gradient, state)
            return state, direction
        return jax.lax.scan(step, transform.init(gradients[0]), gradients)[1]

    def draw(seed):
        rng = np.random.default_rng(seed)
        return ({"attention": {"kernel": rng.standard_normal((7, 6, 4)).astype(np.float32)},
                 "head": rng.standard_normal((7, 4, 6)).astype(np.float32)},)

    def reference(gradients):
        return jax.tree.map(m3_directions, gradients)

    assert_as_exact(directions, reference, draw, "M3 directions", seeds=NESTED_SEEDS)
    (gradients,) = draw(0)
    gradients = gradients["head"]
    state, held = transform.init(gradients[0]), []
    for gradient in gradients:
        _, state = transform.update(gradient, state)
        held.append((np.asarray(state.slow), np.asarray(state.slow_direction)))
    for update, (slow, direction) in enumerate(held, 1):
        if update < M3_FACTORS["frequency"]:
            assert not slow.any() and not direction.any()
        changed = update % M3_FACTORS["frequency"] == 0
        if update > 1:
            assert changed == (not np.array_equal(slow, held[update - 2][0]))
            assert changed == (not np.array_equal(direction, held[update - 2][1]))
    with pytest.raises(ValueError):
        scale_by_m3(**{**M3_FACTORS, "frequency": 0})


def test_m3_keeps_a_bfloat16_matrixs_state_in_float32():
    """M3 over a bfloat16 matrix given float32 gradients, across a chunk's close under jit: its momenta,
    second moment, chunk sum and slow direction stay float32 on every update, so neither branch of the
    once-a-chunk close changes the state's dtype, and the directions are float32 and finite."""
    transform = scale_by_m3(**M3_FACTORS)
    params = {"kernel": jnp.ones((6, 4), jnp.bfloat16)}
    gradients = np.random.default_rng(0).standard_normal((4, 6, 4)).astype(np.float32)
    step = jax.jit(transform.update)
    state = transform.init(params)
    for gradient in gradients:
        direction, state = step({"kernel": jnp.asarray(gradient)}, state)
        assert direction["kernel"].dtype == jnp.float32 and bool(jnp.isfinite(direction["kernel"]).all())
        moments = (state.fast, state.second, state.slow, state.chunk, state.slow_direction)
        assert {leaf.dtype for leaf in jax.tree.leaves(moments)} == {jnp.dtype(jnp.float32)}


ETA = 0.7
"""Delta gradient descent's rate in the proofs below."""


def delta_minimizer(kernel: np.ndarray, x: np.ndarray, output_gradient: np.ndarray) -> np.ndarray:
    """Eq. 56 of arXiv 2512.24695 solved directly in NumPy, in the inputs' own dtype: the kernel `K`
    (`y = x K`) minimizing `|K^T x - u|**2 / 2 + |K - K_t|**2 / (2 eta)`, `u` the output's negative
    gradient, from its normal equations `(x x^T + I / eta) K = x u^T + K_t / eta`."""
    rate = np.asarray(ETA, kernel.dtype)
    normal = np.outer(x, x) + np.eye(len(x), dtype=kernel.dtype) / rate
    return np.linalg.solve(normal, np.outer(x, -output_gradient) + kernel / rate)


def test_delta_gradient_descent_steps_to_eq_56s_minimizer():
    """`delta_gradient_descent` steps a kernel to eq. 56's minimizer, on par with a direct solve of its
    normal equations from float64 (check 1): the closed form `K_t - eta' (x (x^T K_t) + dL/dK)`,
    `eta' = eta / (1 + eta |x|**2)`. A leaf given no input steps by gradient descent, bit for bit; a chain
    hands it its inputs beside other transforms' arguments and steps the same; an update given no inputs,
    or an input that is not its kernel's input axis, is refused."""
    transform = delta_gradient_descent(ETA)

    @jax.jit
    def stepped(kernel, x, output_gradient):
        gradient = {"kernel": jnp.outer(x, output_gradient)}
        params = {"kernel": kernel}
        update, _ = transform.update(gradient, transform.init(params), params, inputs={"kernel": x})
        return kernel + update["kernel"]

    def draw(seed):
        rng = np.random.default_rng(seed)
        return tuple(rng.standard_normal(shape).astype(np.float32) for shape in ((5, 3), (5,), (3,)))

    assert_as_exact(stepped, delta_minimizer, draw, "delta gradient descent", seeds=NESTED_SEEDS)
    kernel, x, output_gradient = (jnp.asarray(value) for value in draw(0))
    params = {"kernel": kernel, "bias": jnp.ones(3)}
    gradients = {"kernel": jnp.outer(x, output_gradient), "bias": output_gradient}
    inputs = {"kernel": x, "bias": None}
    update, state = transform.update(gradients, transform.init(params), params, inputs=inputs)
    np.testing.assert_array_equal(update["bias"], -jnp.asarray(ETA) * output_gradient)
    chained = optax.chain(optax.clip_by_global_norm(1e30), transform)
    through, _ = chained.update(gradients, chained.init(params), params, inputs=inputs, qk_stats=None)
    jax.tree.map(np.testing.assert_array_equal, through, update)
    assert int(state.updates) == 1
    with pytest.raises(ValueError):
        transform.update(gradients, transform.init(params), params)
    with pytest.raises(ValueError):
        transform.update(gradients, transform.init(params), params, inputs={"kernel": output_gradient,
                                                                            "bias": None})


def test_delta_gradient_descent_steps_its_share_of_an_optax_partition():
    """Partitioned with SGD over a kernel and its bias, delta gradient descent steps the kernel by its
    input exactly as it does alone, and leaves the bias, which reaches it masked, to SGD: the bias moves
    by SGD's own step, bit for bit."""
    rng = np.random.default_rng(0)
    kernel, x, output_gradient = (jnp.asarray(rng.standard_normal(shape), jnp.float32)
                                  for shape in ((5, 3), (5,), (3,)))
    params = {"kernel": kernel, "bias": jnp.ones(3)}
    gradients = {"kernel": jnp.outer(x, output_gradient), "bias": output_gradient}
    inputs = {"kernel": x, "bias": None}
    split = optax.partition({"delta": delta_gradient_descent(ETA), "sgd": optax.sgd(ETA)},
                            {"kernel": "delta", "bias": "sgd"})
    update, _ = split.update(gradients, split.init(params), params, inputs=inputs)
    alone = delta_gradient_descent(ETA)
    kernel_step, _ = alone.update({"kernel": gradients["kernel"]}, alone.init(params), {"kernel": kernel},
                                  inputs={"kernel": x})
    sgd = optax.sgd(ETA)
    bias_step, _ = sgd.update(gradients["bias"], sgd.init(params["bias"]))
    np.testing.assert_array_equal(update["kernel"], kernel_step["kernel"])
    np.testing.assert_array_equal(update["bias"], bias_step)
