"""The categorical distribution's one owner (`dew.nn.categorical`): every log-likelihood Dew takes
computes in at least float32 and keeps float64 logits in float64, masks the classes a row does not
hold, and each form agrees with the others; a label's log probability through the token owner
(`dew.objectives.base.token_log_probs`) is float64's own for a float64 model."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.nn import categorical
from dew.objectives.base import token_log_probs


def logits(dtype, shape=(3, 7), seed=0):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape) * 4, dtype)


@pytest.mark.parametrize(("given", "computed"), [(jnp.bfloat16, jnp.float32), (jnp.float16, jnp.float32),
                                                 (jnp.float32, jnp.float32), (jnp.float64, jnp.float64)])
def test_each_form_computes_in_at_least_float32_and_keeps_float64(given, computed):
    with jax.enable_x64(new_val=True):
        rows = logits(given)
        labels = jnp.asarray([0, 3, 6])
        forms = [categorical.log_probs(rows), categorical.probabilities(rows),
                 categorical.cross_entropy(rows, labels), categorical.entropy(rows),
                 categorical.soft_cross_entropy(rows, jax.nn.one_hot(labels, 7, dtype=given)),
                 categorical.kl_divergence(rows, logits(given, seed=1)), token_log_probs(rows, labels)]
        assert {form.dtype for form in forms} == {jnp.dtype(computed)}


def test_a_float64_models_label_log_probability_is_float64s_own():
    """A float64 model's token log probabilities are computed in float64, not rounded to float32 first:
    the label's log partition less its logit in float64 (Optax's form), which float32 would not give."""
    import optax

    with jax.enable_x64(new_val=True):
        rows = logits(jnp.float64) * 1e3 + 0.1
        labels = jnp.asarray([1, 2, 5])
        got = np.asarray(token_log_probs(rows, labels))
        wide = -np.asarray(optax.softmax_cross_entropy_with_integer_labels(rows, labels))
        np.testing.assert_array_equal(got, wide)
        narrow = -np.asarray(optax.softmax_cross_entropy_with_integer_labels(rows.astype(jnp.float32),
                                                                             labels))
        assert not np.array_equal(got, narrow.astype(np.float64))


def test_the_forms_agree_and_a_masked_class_takes_no_probability():
    """A one-hot soft target's cross entropy is minus the label's log probability, the entropy minus
    the expected log probability, the divergence Optax's of the two normalizations; a class `where`
    excludes takes no
    probability while the others hold it, and target mass on an excluded class is an infinite
    charge."""
    rows, labels = logits(jnp.float32), jnp.asarray([0, 3, 6])
    log_p = np.asarray(categorical.log_probs(rows))
    picked = np.take_along_axis(log_p, np.asarray(labels)[:, None], 1)[:, 0]
    np.testing.assert_array_equal(categorical.soft_cross_entropy(rows, jax.nn.one_hot(labels, 7)), -picked)
    np.testing.assert_array_equal(categorical.entropy(rows), -(jnp.exp(log_p) * log_p).sum(-1))
    # The divergence is Optax's of the two normalizations; from oneself it is a rounding's worth off
    # zero in fp32 (the softmax's cancellation), which the owner's divergence form takes up apart.
    np.testing.assert_array_equal(categorical.kl_divergence(rows, rows), optax.kl_divergence(
        categorical.log_probs(rows), categorical.probabilities(rows)))
    held = jnp.arange(7) < 4
    masked = np.asarray(categorical.probabilities(rows, held))
    np.testing.assert_array_equal(masked[:, 4:], 0)
    assert np.all(masked[:, :4] > 0)
    stray = jnp.zeros((3, 7)).at[:, 5].set(1.0)
    assert np.all(np.isinf(np.asarray(categorical.soft_cross_entropy(rows, stray, held))))
