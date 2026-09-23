"""Token log probabilities from bf16 logits, against a float64 reference.

The reference normalizes the bf16 logits' exact values in float64, so the
only error left is the reduction's. In fp32 that is a few roundings at the
magnitude of the result, bounded here by 4 * eps32 * max|log p|, which is
1.7e-5 for the 32,768-column rows and 1.4e-5 for the denoiser's 1,024, against a
measured worst of 9.4e-7 and 9.5e-7. A reduction left in bf16 misses by up
to half a bf16 step at that magnitude, 0.04 to 0.11 measured, so the bound
tells the two apart by three orders.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.diffusion.discrete import DiscreteProcess, LogLinear
from dew.objectives.likelihood import token_log_probs

VOCAB = 32768


def _reference(logits: jax.Array) -> np.ndarray:
    exact = np.asarray(logits.astype(jnp.float32), np.float64)
    shifted = exact - exact.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def _bound(reference: np.ndarray) -> float:
    finite = reference[np.isfinite(reference)]
    return 4 * float(np.finfo(np.float32).eps) * float(np.abs(finite).max())


def _logits(shape) -> jax.Array:
    return (4 * jax.random.normal(jax.random.key(0), shape)).astype(jnp.bfloat16)


def test_token_log_probs_reduces_bf16_logits_in_fp32():
    logits = _logits((8, VOCAB))
    tokens = jax.random.randint(jax.random.key(1), (8,), 0, VOCAB)
    reference = np.take_along_axis(_reference(logits), np.asarray(tokens)[:, None], -1)[:, 0]

    picked = np.asarray(jax.jit(token_log_probs)(logits, tokens), np.float64)
    in_bf16 = np.asarray(jnp.take_along_axis(jax.nn.log_softmax(logits), tokens[:, None], -1)[:, 0]
                         .astype(jnp.float32), np.float64)

    assert np.abs(picked - reference).max() <= _bound(reference)
    assert np.abs(in_bf16 - reference).max() > 100 * _bound(reference)


class Fixed(nn.Module):
    """A model whose bf16 logits are one fixed draw, whatever it reads."""

    @nn.compact
    def __call__(self, tokens):
        return _logits((*tokens.shape, 1024))


def test_the_masked_denoiser_normalizes_bf16_logits_in_fp32():
    """The reveal draws its categorical from these log probabilities."""
    mask = 1023
    process = DiscreteProcess(LogLinear(), mask_id=mask)
    tokens = jnp.full((2, 4), mask, jnp.int32)

    _, log_probs = process.denoiser(Fixed(), {})(tokens, jnp.full((2,), 0.5))

    expected = _reference(_logits((2, 4, 1024)).at[..., mask].set(-jnp.inf))
    assert log_probs.dtype == jnp.float32
    assert np.all(np.isneginf(np.asarray(log_probs)[..., mask]))
    live = np.isfinite(expected)
    assert np.abs(np.asarray(log_probs, np.float64)[live] - expected[live]).max() <= _bound(expected)
