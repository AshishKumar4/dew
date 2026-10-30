"""DreamerV3's output transforms and distributions on plain arrays (arXiv 2301.04104 and its code,
danijar/dreamerv3 at e01491fad6434b2245a3b8ca201dd7faedcc458c, embodied/jax/nets.py and outs.py).

`symlog` compresses a target's magnitude and `symexp` inverts it. A two-hot distribution over the
bins symexp(linspace(-20, 20, 255)) predicts a scalar of any scale as its bins' expectation, summed
from both ends toward the middle so uniform probabilities predict zero exactly, and is trained by
the cross-entropy of a target's two-hot encoding, its weight split between its two neighbouring bins.
A categorical latent mixes its softmax with 1% uniform (`unimix`) and is drawn one-hot, its gradient
passed straight through the probabilities.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def symlog(x: jax.Array) -> jax.Array:
    """sign(x) log(1 + |x|)."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: jax.Array) -> jax.Array:
    """sign(x) (exp(|x|) - 1), `symlog`'s inverse."""
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def twohot_bins(count: int = 255, limit: float = 20.0) -> jax.Array:
    """The two-hot bins: symexp of `count` points evenly spaced over [-limit, limit], the negative half
    mirrored so the bins are exactly symmetric (embodied/jax/heads.py:132-144)."""
    if count % 2 == 1:
        half = symexp(jnp.linspace(-limit, 0, (count - 1) // 2 + 1, dtype=jnp.float32))
        return jnp.concatenate([half, -half[:-1][::-1]])
    half = symexp(jnp.linspace(-limit, 0, count // 2, dtype=jnp.float32))
    return jnp.concatenate([half, -half[::-1]])


def twohot_mean(logits: jax.Array, bins: jax.Array) -> jax.Array:
    """The expectation of `bins` under softmax(`logits`), summed outward-in from both ends to the
    middle bin, so symmetric bins under uniform probabilities give exactly zero (outs.py:285-309)."""
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    count = bins.shape[-1]
    middle = count // 2
    if count % 2 == 1:
        low, centre, high = probs[..., :middle], probs[..., middle], probs[..., middle + 1:]
        return centre * bins[middle] + ((low * bins[:middle])[..., ::-1] + high * bins[middle + 1:]).sum(-1)
    return ((probs[..., :middle] * bins[:middle])[..., ::-1] + probs[..., middle:] * bins[middle:]).sum(-1)


def twohot_loss(logits: jax.Array, target: jax.Array, bins: jax.Array) -> jax.Array:
    """The cross-entropy of softmax(`logits`) against `target`'s two-hot encoding over `bins`: its
    weight on the bins either side of it, each the other's share of the distance between them, all
    of it on the first or last bin past the range (outs.py:311-323)."""
    target = jax.lax.stop_gradient(target.astype(jnp.float32))
    below = jnp.sum(bins <= target[..., None], axis=-1).astype(jnp.int32) - 1
    above = bins.shape[-1] - jnp.sum(bins > target[..., None], axis=-1).astype(jnp.int32)
    below = jnp.clip(below, 0, bins.shape[-1] - 1)
    above = jnp.clip(above, 0, bins.shape[-1] - 1)
    equal = below == above
    to_below = jnp.where(equal, 1, jnp.abs(bins[below] - target))
    to_above = jnp.where(equal, 1, jnp.abs(bins[above] - target))
    total = to_below + to_above
    encoded = (jax.nn.one_hot(below, bins.shape[-1]) * (to_above / total)[..., None]
               + jax.nn.one_hot(above, bins.shape[-1]) * (to_below / total)[..., None])
    return -jnp.sum(encoded * jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1), axis=-1)


def unimix_log_probs(logits: jax.Array, unimix: float = 0.01) -> jax.Array:
    """A categorical's log-probabilities, its softmax mixed with `unimix` of the uniform."""
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    if unimix:
        probs = (1 - unimix) * probs + unimix / logits.shape[-1]
    return jnp.log(probs)


def onehot_sample(key: jax.Array, log_probs: jax.Array) -> jax.Array:
    """A one-hot draw from the categorical `log_probs`, its gradient the probabilities' (straight
    through)."""
    probs = jnp.exp(log_probs)
    drawn = jax.nn.one_hot(jax.random.categorical(key, log_probs), log_probs.shape[-1], dtype=probs.dtype)
    return drawn + probs - jax.lax.stop_gradient(probs)


def categorical_kl(log_p: jax.Array, log_q: jax.Array) -> jax.Array:
    """KL(p || q) of categoricals over the last axis."""
    return jnp.sum(jnp.exp(log_p) * (log_p - log_q), axis=-1)


def categorical_entropy(log_p: jax.Array) -> jax.Array:
    return -jnp.sum(jnp.exp(log_p) * log_p, axis=-1)


__all__ = ["categorical_entropy", "categorical_kl", "onehot_sample", "symexp", "symlog", "twohot_bins",
           "twohot_loss", "twohot_mean", "unimix_log_probs"]
