"""What `reference_error.float64_terms` takes for a leaf's terms: the root sum
of squares of the terms its final sums add, carried into it, each at its
operands' sizes, their own terms where sums behind them made them."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh, PartitionSpec as P
from reference_error import float64_terms


def loss_of(keys, query, values):
    scores = jnp.einsum("bd,bkd->bk", query, keys)
    return jnp.sum(jax.nn.softmax(scores, axis=-1)[..., None] * values)


def test_a_key_bias_gradient_sums_terms_that_cancel():
    """A key bias shifts every logit of a query's softmax alike, so its
    gradient is zero in exact arithmetic, while the cotangents it sums are
    not: its terms are the root sum of squares over the rows and keys of the
    keys' cotangent's sizes, as the bias's gradient sums them, each the
    cotangent's own terms where the softmax's backward sums made them larger
    than it, and so at least the cotangents' own root sum of squares."""
    rng = np.random.default_rng(0)
    query, keys, values = (rng.normal(size=shape) for shape in ((2, 4), (2, 6, 4), (2, 6, 3)))

    with jax.enable_x64(new_val=True):
        gradient, terms = float64_terms(jax.grad(lambda bias: loss_of(keys + bias, query, values)),
                                        jnp.zeros(4))
        cotangent, cotangent_terms = float64_terms(jax.grad(loss_of), jnp.asarray(keys), query, values)
    sizes = np.maximum(np.asarray(cotangent_terms), np.abs(np.asarray(cotangent)))
    np.testing.assert_allclose(gradient, 0, atol=1e-15)
    np.testing.assert_allclose(terms, np.sqrt(np.sum(sizes**2, axis=(0, 1))), rtol=1e-12)
    assert np.all(terms >= np.sqrt(np.sum(np.asarray(cotangent)**2, axis=(0, 1))))
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


def test_a_sum_of_sums_that_cancel_is_sized_by_the_terms_behind_it():
    """A sum whose operand is a sum that cancels, scaled element by element
    between them, as a deep gradient's cotangent is, takes that operand's
    terms for its size, not its zero value: its terms are the root sum of
    squares of the terms behind it. Without them a wrong entry of the outer
    sum would be held to its own rounding of zero."""
    rng = np.random.default_rng(4)
    x, a, b = rng.normal(size=(4, 6)), rng.normal(size=(6, 5)), rng.normal(size=(5, 3))
    scale = rng.normal(size=5)

    def chained(x, a, b):
        inner = x @ a + x @ -a
        return (inner * scale) @ b

    with jax.enable_x64(new_val=True):
        value, terms = float64_terms(chained, x, a, b)
    np.testing.assert_array_equal(value, 0)
    inner = 2 * np.square(x) @ np.square(a)
    np.testing.assert_allclose(terms, np.sqrt((inner * scale**2) @ np.square(b)), rtol=1e-12)


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
