"""Text generation: the cached decode loop has to agree with the plain model.

Greedy generation is the strict check, because it must reproduce, token for
token, what walking the argmax of a full forward pass produces. The copy task is
the end-to-end one: train a tiny decoder with plain optax, then read the
sequence back out of generate().
"""

import jax
import jax.numpy as jnp
import optax
import pytest

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.sampling import generate

VOCAB = 29
PAYLOAD = 6
SEPARATOR = 0


def tiny(**overrides):
    config = dict(vocab_size=VOCAB, emb_features=32, num_layers=2, num_heads=4,
                  mlp_ratio=2, max_seq_len=16)
    return CausalTransformer(**{**config, **overrides})


def argmax_walk(model, params, prompt, steps):
    """Greedy continuation without a cache: one full forward pass per token."""
    sequence = prompt
    for _ in range(steps):
        logits = model.apply(params, sequence)
        sequence = jnp.concatenate(
            [sequence, jnp.argmax(logits[:, -1:], axis=-1).astype(jnp.int32)], axis=1)
    return sequence


def test_greedy_generation_follows_the_full_sequence_argmax(rng):
    model = tiny()
    prompt = jax.random.randint(rng, (3, 5), 0, VOCAB)
    params = model.init(rng, prompt)

    generated = generate(model, params, prompt, 6, rng=jax.random.PRNGKey(1),
                         temperature=0)
    assert generated.shape == (3, 11)
    assert generated.dtype == jnp.int32
    assert jnp.array_equal(generated[:, :5], prompt)
    assert jnp.array_equal(generated, argmax_walk(model, params, prompt, 6))
    # greedy ignores the rng, so two calls cannot disagree
    assert jnp.array_equal(
        generated, generate(model, params, prompt, 6, rng=jax.random.PRNGKey(7),
                            temperature=0))


def test_layer_types_from_json_can_generate(rng):
    model = tiny(
        layer_types=["full_attention", "sliding_attention"],
        sliding_window=4,
    )
    prompt = jax.random.randint(rng, (2, 4), 0, VOCAB)
    params = model.init(rng, prompt)

    generated = generate(
        model, params, prompt, 2, rng=jax.random.PRNGKey(1), temperature=0)

    assert jnp.array_equal(generated, argmax_walk(model, params, prompt, 2))


def test_sampling_stays_in_the_vocab_and_reacts_to_the_rng(rng):
    model = tiny()
    prompt = jax.random.randint(rng, (4, 4), 0, VOCAB)
    params = model.init(rng, prompt)

    sampled = generate(model, params, prompt, 8, rng=jax.random.PRNGKey(0),
                       temperature=1.0)
    assert sampled.shape == (4, 12)
    assert jnp.all((sampled >= 0) & (sampled < VOCAB))

    other = generate(model, params, prompt, 8, rng=jax.random.PRNGKey(1),
                     temperature=1.0)
    assert not jnp.array_equal(sampled, other)

    warm = generate(model, params, prompt, 8, rng=jax.random.PRNGKey(0),
                    temperature=1.0)
    assert jnp.array_equal(sampled, warm)


def test_top_k_restricts_the_choice_and_top_one_is_greedy(rng):
    model = tiny()
    prompt = jax.random.randint(rng, (2, 4), 0, VOCAB)
    params = model.init(rng, prompt)

    greedy = generate(model, params, prompt, 5, rng=jax.random.PRNGKey(2), temperature=0)
    assert jnp.array_equal(
        greedy, generate(model, params, prompt, 5, rng=jax.random.PRNGKey(3),
                         temperature=1.0, top_k=1))

    sampled = generate(model, params, prompt, 5, rng=jax.random.PRNGKey(4),
                       temperature=0.8, top_k=5)
    assert jnp.all((sampled >= 0) & (sampled < VOCAB))
    # every sampled token has to be inside the top 5 of its own step
    for step in range(5):
        position = prompt.shape[1] + step
        logits = model.apply(params, sampled[:, :position])[:, -1]
        allowed = jnp.argsort(logits, axis=-1)[:, -5:]
        chosen = sampled[:, position]
        assert jnp.all(jnp.any(allowed == chosen[:, None], axis=-1))


def test_a_single_new_token_and_none_at_all(rng):
    model = tiny()
    prompt = jax.random.randint(rng, (2, 4), 0, VOCAB)
    params = model.init(rng, prompt)

    assert jnp.array_equal(prompt, generate(model, params, prompt, 0,
                                            rng=jax.random.PRNGKey(0)))
    one = generate(model, params, prompt, 1, rng=jax.random.PRNGKey(0), temperature=0)
    assert one.shape == (2, 5)
    assert jnp.array_equal(one, argmax_walk(model, params, prompt, 1))


def test_generation_longer_than_the_cache_is_refused(rng):
    model = tiny(max_seq_len=8)
    prompt = jax.random.randint(rng, (1, 6), 0, VOCAB)
    params = model.init(rng, prompt)
    with pytest.raises(ValueError, match="max_seq_len"):
        generate(model, params, prompt, 4, rng=jax.random.PRNGKey(0))


def copy_batch(rng, size):
    """[payload, separator, payload]: the second half is only predictable from
    the first, so a model that scores it learned to look back."""
    payload = jax.random.randint(rng, (size, PAYLOAD), 1, VOCAB)
    separator = jnp.full((size, 1), SEPARATOR, jnp.int32)
    return jnp.concatenate([payload, separator, payload], axis=1)


def test_copy_task_trains_and_generate_reads_the_sequence_back():
    """End to end on plain optax, no trainer: 300 steps must get the copy region
    above 90% next-token accuracy, and generate() must reproduce the payload."""
    model = tiny(emb_features=64)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.zeros((1, 2 * PAYLOAD), jnp.int32))
    optimizer = optax.adam(3e-3)
    state = optimizer.init(params)

    def loss_fn(params, sequence):
        logits = model.apply(params, sequence[:, :-1])
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, sequence[:, 1:]).mean()

    @jax.jit
    def train_step(params, state, sequence):
        loss, grads = jax.value_and_grad(loss_fn)(params, sequence)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    for _ in range(300):
        rng, batch_rng = jax.random.split(rng)
        params, state, loss = train_step(params, state, copy_batch(batch_rng, 64))
    assert jnp.isfinite(loss)

    held_out = copy_batch(jax.random.PRNGKey(99), 16)
    predicted = jnp.argmax(model.apply(params, held_out[:, :-1]), axis=-1)
    copy_region = (predicted[:, PAYLOAD:] == held_out[:, PAYLOAD + 1:]).mean()
    assert copy_region > 0.9

    prompt = held_out[:, :PAYLOAD + 1]
    generated = generate(model, params, prompt, PAYLOAD, rng=jax.random.PRNGKey(1),
                         temperature=0)
    assert jnp.array_equal(generated[:, PAYLOAD + 1:], held_out[:, :PAYLOAD])
