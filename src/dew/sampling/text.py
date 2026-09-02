"""Autoregressive text sampling: prefill the prompt, then decode one token a step.

The prompt goes through the model once with the KV cache mutable, which fills
the cache and gives the logits for the first new token. Every step after that
is a single-token forward pass inside a lax.scan, so the whole generation is
one compiled loop rather than max_new_tokens traces.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax


def _sample_token(logits, rng, temperature: float, top_k):
    """One token per row from [B, vocab] logits. temperature=0 is greedy."""
    if temperature == 0:
        return jnp.argmax(logits, axis=-1).astype(jnp.int32)
    logits = logits / temperature
    if top_k is not None:
        keep = min(int(top_k), logits.shape[-1])
        cutoff = lax.top_k(logits, keep)[0][..., -1:]
        logits = jnp.where(logits < cutoff, -jnp.inf, logits)
    return jax.random.categorical(rng, logits).astype(jnp.int32)


def _generate(model, params, prompt, rng, max_new_tokens: int,
              temperature: float, top_k):
    cache = model.apply(params, prompt.shape[0], method=type(model).init_cache,
                        mutable=['cache'])[1]['cache']
    variables = {**params, 'cache': cache}
    logits, mutated = model.apply(variables, prompt, decode=True, mutable=['cache'])
    rng, prefill_rng = jax.random.split(rng)
    first = _sample_token(logits[:, -1], prefill_rng, temperature, top_k)

    def step(carry, _):
        rng, token, cache = carry
        rng, step_rng = jax.random.split(rng)
        logits, mutated = model.apply({**params, 'cache': cache}, token[:, None],
                                      decode=True, mutable=['cache'])
        following = _sample_token(logits[:, -1], step_rng, temperature, top_k)
        return (rng, following, mutated['cache']), following

    _, rest = lax.scan(step, (rng, first, mutated['cache']), None,
                       length=max_new_tokens - 1)
    generated = jnp.concatenate([first[:, None], jnp.swapaxes(rest, 0, 1)], axis=1)
    return jnp.concatenate([prompt, generated], axis=1)


# The model, the length and the sampling knobs are compile-time constants, so
# repeated calls with the same settings hit the same compiled loop.
_jit_generate = jax.jit(
    _generate, static_argnames=('model', 'max_new_tokens', 'temperature', 'top_k'))


def generate(model, params, prompt, max_new_tokens: int, *, rng,
             temperature: float = 1.0, top_k=None):
    """Sample `max_new_tokens` tokens after `prompt`: [B, P] -> [B, P + max_new_tokens].

    params is the full variables dict the trainer holds ({'params': ...}), the
    same thing the diffusion samplers take. temperature=0 is greedy decoding
    (deterministic, ignores rng); top_k restricts each step to the k most
    likely tokens. The prompt must fit the model's KV cache together with the
    tokens being generated.
    """
    prompt = jnp.asarray(prompt, jnp.int32)
    if prompt.ndim != 2:
        raise ValueError(f"prompt must be [B, P] token ids, got shape {prompt.shape}")
    if 'params' not in params:
        raise ValueError(
            "generate takes the full variables dict ({'params': ...}), the same "
            "thing model.init returns and the trainer state carries.")
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    if max_new_tokens == 0:
        return prompt
    cache_len = getattr(model, 'max_seq_len', None)
    if cache_len is not None and prompt.shape[1] + max_new_tokens > cache_len:
        raise ValueError(
            f"{prompt.shape[1]} prompt tokens plus {max_new_tokens} new ones "
            f"exceed the model's KV cache of {cache_len}; raise max_seq_len.")
    return _jit_generate(model=model, params=params, prompt=prompt, rng=rng,
                         max_new_tokens=int(max_new_tokens),
                         temperature=float(temperature),
                         top_k=None if top_k is None else int(top_k))
