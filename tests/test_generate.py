"""Text generation: the cached decode loop has to agree with the plain model.

Greedy generation is the strict check, because it must reproduce, token for
token, what walking the argmax of a full forward pass produces. The copy task is
the end-to-end one: train a tiny decoder with plain optax, then read the
sequence back out of generate().
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.nn.backbones.causal_transformer import CausalTransformer, Mixture
from dew.sampling import Sampling, generate

VOCAB = 29
PAYLOAD = 6
SEPARATOR = 0


def tiny(**overrides):
    config = dict(vocab_size=VOCAB, emb_features=32, num_layers=2, num_heads=4,
                  mlp_features=64, max_seq_len=16)
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

    generated = generate(model, params, prompt, 6, key=jax.random.PRNGKey(1), sampling=Sampling(temperature=0)).tokens
    assert generated.shape == (3, 11)
    assert generated.dtype == jnp.int32
    assert jnp.array_equal(generated[:, :5], prompt)
    assert jnp.array_equal(generated, argmax_walk(model, params, prompt, 6))
    # greedy ignores the rng, so two calls cannot disagree
    assert jnp.array_equal(
        generated, generate(model, params, prompt, 6, key=jax.random.PRNGKey(7), sampling=Sampling(temperature=0)).tokens)


def test_a_pattern_and_its_kinds_from_json_can_generate(rng):
    """A run record hands the pattern as a list and each kind as a record,
    the form a config parses to."""
    model = tiny(
        layer_types=["full_attention", "sliding_attention"],
        kinds={"sliding_attention": {"window": 4}},
    )
    prompt = jax.random.randint(rng, (2, 4), 0, VOCAB)
    params = model.init(rng, prompt)

    generated = generate(model, params, prompt, 2, key=jax.random.PRNGKey(1), sampling=Sampling(temperature=0)).tokens

    assert jnp.array_equal(generated, argmax_walk(model, params, prompt, 2))


def test_sampling_stays_in_the_vocab_and_reacts_to_the_rng(rng):
    model = tiny()
    prompt = jax.random.randint(rng, (4, 4), 0, VOCAB)
    params = model.init(rng, prompt)

    sampled = generate(model, params, prompt, 8, key=jax.random.PRNGKey(0), sampling=Sampling(temperature=1.0)).tokens
    assert sampled.shape == (4, 12)
    assert jnp.all((sampled >= 0) & (sampled < VOCAB))

    other = generate(model, params, prompt, 8, key=jax.random.PRNGKey(1), sampling=Sampling(temperature=1.0)).tokens
    assert not jnp.array_equal(sampled, other)

    warm = generate(model, params, prompt, 8, key=jax.random.PRNGKey(0), sampling=Sampling(temperature=1.0)).tokens
    assert jnp.array_equal(sampled, warm)


def test_top_k_restricts_the_choice_and_top_one_is_greedy(rng):
    model = tiny()
    prompt = jax.random.randint(rng, (2, 4), 0, VOCAB)
    params = model.init(rng, prompt)

    greedy = generate(model, params, prompt, 5, key=jax.random.PRNGKey(2), sampling=Sampling(temperature=0)).tokens
    assert jnp.array_equal(
        greedy, generate(model, params, prompt, 5, key=jax.random.PRNGKey(3), sampling=Sampling(temperature=1.0, top_k=1)).tokens)

    sampled = generate(model, params, prompt, 5, key=jax.random.PRNGKey(4), sampling=Sampling(temperature=0.8, top_k=5)).tokens
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
                                            key=jax.random.PRNGKey(0)).tokens)
    one = generate(model, params, prompt, 1, key=jax.random.PRNGKey(0), sampling=Sampling(temperature=0)).tokens
    assert one.shape == (2, 5)
    assert jnp.array_equal(one, argmax_walk(model, params, prompt, 1))


def test_every_continuation_row_belongs_to_the_prompt_it_sits_under(rng):
    """The rows are prompt major: prompt zero's `n` rows first, then prompt
    one's. Greedy draws are the check, because then a prompt's continuations
    are all its own argmax walk and a row paired with another prompt shows
    up immediately."""
    model = tiny()
    prompt = jax.random.randint(rng, (3, 4), 0, VOCAB)
    params = model.init(rng, prompt)

    result = generate(model, params, prompt, 5, key=jax.random.PRNGKey(0),
                      sampling=Sampling(temperature=0), n=4)

    walked = np.asarray(argmax_walk(model, params, prompt, 5))
    np.testing.assert_array_equal(np.asarray(result.tokens), np.repeat(walked, 4, axis=0))


def test_continuations_of_one_prompt_are_separate_draws(rng):
    """Every row carries its own draw and its own metadata, so `n` rows of one
    prompt are `n` samples rather than one sample repeated."""
    model = tiny()
    prompt = jax.random.randint(rng, (3, 4), 0, VOCAB)
    params = model.init(rng, prompt)

    result = generate(model, params, prompt, 5, key=jax.random.PRNGKey(0),
                      sampling=Sampling(temperature=1.0), n=4)

    assert result.tokens.shape == (12, 9) and result.rows == 12
    assert result.lengths.shape == result.terminated.shape == (12,)
    assert result.behavior_log_probs.shape == result.raw_log_probs.shape == (12, 5)
    np.testing.assert_array_equal(result.tokens[:, :4], np.repeat(np.asarray(prompt), 4, axis=0))
    drawn = np.asarray(result.tokens)[:, 4:].reshape(3, 4, 5)
    for prompt_rows in drawn:
        assert len({tuple(row) for row in prompt_rows}) == 4
    assert not np.array_equal(drawn[0], drawn[1])


def test_continuation_zero_is_the_single_draw_and_earlier_continuations_do_not_move(rng):
    """The keys do not depend on how many continuations were asked for: row
    zero of every prompt repeats the single-continuation draw bit for bit, and
    raising `n` leaves the continuations already drawn where they were."""
    model = tiny()
    prompt = jax.random.randint(rng, (2, 4), 0, VOCAB)
    params = model.init(rng, prompt)
    sampling = Sampling(temperature=0.9, top_k=7, eos_id=(3, 9), pad_id=1)
    draw = functools.partial(generate, model, params, prompt, 5,
                             key=jax.random.PRNGKey(4), sampling=sampling)

    one, two, three = draw(n=1), draw(n=2), draw(n=3)

    for field in ("tokens", "lengths", "terminated", "behavior_log_probs", "raw_log_probs"):
        single = np.asarray(getattr(one, field))
        np.testing.assert_array_equal(np.asarray(getattr(two, field))[::2], single)
        np.testing.assert_array_equal(np.asarray(getattr(three, field))[::3], single)
        grown = np.asarray(getattr(three, field)).reshape((2, 3, *single.shape[1:]))
        np.testing.assert_array_equal(grown[:, 1],
                                      np.asarray(getattr(two, field)).reshape((2, 2, *single.shape[1:]))[:, 1])
    assert not np.array_equal(np.asarray(two.tokens)[0], np.asarray(two.tokens)[1])
    np.testing.assert_array_equal(np.asarray(draw(n=3).tokens), np.asarray(three.tokens))


def test_each_continuation_stops_at_its_own_eos(rng):
    """A drawn EOS ends the row that drew it: its length, its termination flag
    and its padding are its own, and one prompt's continuations do not stop
    together."""
    model = tiny()
    prompt = jax.random.randint(rng, (2, 4), 0, VOCAB)
    params = model.init(rng, prompt)
    # A token the greedy policy walks into, so sampled rows reach it as well.
    eos = int(generate(model, params, prompt, 1, key=jax.random.PRNGKey(0),
                       sampling=Sampling(temperature=0)).tokens[0, -1])
    sampling = Sampling(temperature=1.0, eos_id=eos, pad_id=VOCAB - 1)

    result = generate(model, params, prompt, 6, key=jax.random.PRNGKey(1), sampling=sampling, n=5)

    drawn = np.asarray(result.tokens)[:, 4:]
    lengths, terminated = np.asarray(result.lengths), np.asarray(result.terminated)
    for row, (length, stopped) in enumerate(zip(lengths, terminated)):
        assert not np.any(drawn[row, :length - 1] == eos)
        assert bool(drawn[row, length - 1] == eos) == bool(stopped)
        np.testing.assert_array_equal(drawn[row, length:], sampling.pad_id)
        np.testing.assert_array_equal(np.asarray(result.behavior_log_probs)[row, length:], 0)
        np.testing.assert_array_equal(np.asarray(result.raw_log_probs)[row, length:], 0)
    assert len(set(lengths[:5].tolist())) > 1


def test_continuations_run_on_a_routed_expert_decoder(rng):
    """Routed experts sort their tokens by expert inside the forward, so the
    continuation map has to leave the batch axis alone: a mixture decoder draws
    several continuations and its first one is the single-continuation draw."""
    model = tiny(mixture=Mixture(experts=4, top_k=2))
    prompt = jax.random.randint(rng, (2, 4), 0, VOCAB)
    params = model.init(rng, prompt)
    sampling = Sampling(temperature=0.9, top_k=5)

    two = generate(model, params, prompt, 4, key=jax.random.PRNGKey(2), sampling=sampling, n=2)

    one = generate(model, params, prompt, 4, key=jax.random.PRNGKey(2), sampling=sampling)
    assert two.tokens.shape == (4, 8)
    np.testing.assert_array_equal(np.asarray(two.tokens)[::2], np.asarray(one.tokens))
    np.testing.assert_array_equal(np.asarray(two.raw_log_probs)[::2], np.asarray(one.raw_log_probs))
    assert not np.array_equal(np.asarray(two.tokens)[0, 4:], np.asarray(two.tokens)[1, 4:])


def test_generation_longer_than_the_cache_is_refused(rng):
    model = tiny(max_seq_len=8)
    prompt = jax.random.randint(rng, (1, 6), 0, VOCAB)
    params = model.init(rng, prompt)
    with pytest.raises(ValueError, match="max_seq_len"):
        generate(model, params, prompt, 4, key=jax.random.PRNGKey(0))


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
        grads = jax.grad(loss_fn)(params, sequence)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state

    for _ in range(300):
        rng, batch_rng = jax.random.split(rng)
        params, state = train_step(params, state, copy_batch(batch_rng, 64))

    held_out = copy_batch(jax.random.PRNGKey(99), 16)
    predicted = jnp.argmax(model.apply(params, held_out[:, :-1]), axis=-1)
    copy_region = (predicted[:, PAYLOAD:] == held_out[:, PAYLOAD + 1:]).mean()
    assert copy_region > 0.9

    prompt = held_out[:, :PAYLOAD + 1]
    generated = generate(model, params, prompt, PAYLOAD, key=jax.random.PRNGKey(1), sampling=Sampling(temperature=0)).tokens
    assert jnp.array_equal(generated[:, PAYLOAD + 1:], held_out[:, :PAYLOAD])


@pytest.mark.mesh
def test_the_sampled_rows_keep_their_sharding_and_host_reads_them_back(rng):
    """Where a decode lands is part of the contract: rows split over the
    mesh's batch axes the way the prompt does, so no device holds a batch it
    did not compute, and `host()` hands back the rows a process owns in
    order. Without the pinned output layout the placement would be the
    compiler's choice."""
    from dew.training import Layout, MeshSpec
    from dew.training.distributed import batch_shardings, build_mesh
    from dew.nn.inputs import BATCH_AXES

    model = tiny(max_seq_len=8)
    mesh = build_mesh(MeshSpec(fsdp=2))
    params = model.init(rng, jnp.ones((2, 4), jnp.int32))
    placed = jax.device_put(params, Layout(min_shard=2 ** 8).shardings(mesh, params))
    prompt = jax.random.randint(rng, (8, 3), 0, VOCAB)
    sharded = jax.device_put(prompt, batch_shardings(mesh, prompt))

    generated = generate(model, placed, sharded, 3, key=jax.random.PRNGKey(0), sampling=Sampling(temperature=0))
    plain = generate(model, params, prompt, 3, key=jax.random.PRNGKey(0), sampling=Sampling(temperature=0))

    assert generated.tokens.sharding.mesh == mesh
    assert generated.tokens.sharding.spec == jax.sharding.PartitionSpec(BATCH_AXES)
    assert generated.rows == 8
    rows = generated.host()
    assert isinstance(rows.tokens, np.ndarray) and rows.tokens.shape == (8, 6)
    np.testing.assert_array_equal(rows.tokens, plain.host().tokens)


@pytest.mark.mesh
def test_row_padding_pads_prompts_and_hands_back_only_their_continuations(rng):
    """A batch the mesh cannot split evenly is padded in prompts, before the
    continuations exist, so a process's real rows are its own prompts' `n`
    continuations and nothing else. The padded prompt groups stay on the
    global array, which is why `rows` counts real prompts times `n`."""
    from dew.training import Layout, MeshSpec
    from dew.training.distributed import build_mesh
    from dew.nn.inputs import BATCH_AXES, local_rows

    model = tiny(max_seq_len=8)
    mesh = build_mesh(MeshSpec(fsdp=2))
    params = model.init(rng, jnp.ones((2, 4), jnp.int32))
    placed = jax.device_put(params, Layout(min_shard=2 ** 8).shardings(mesh, params))
    prompt = jax.random.randint(rng, (5, 3), 0, VOCAB)
    sampling = Sampling(temperature=0.8, top_k=4)

    generated = generate(model, placed, prompt, 3, key=jax.random.PRNGKey(0), sampling=sampling, n=2)
    plain = generate(model, params, prompt, 3, key=jax.random.PRNGKey(0), sampling=sampling, n=2)

    assert generated.tokens.sharding.spec == jax.sharding.PartitionSpec(BATCH_AXES)
    assert generated.tokens.shape == (16, 6) and generated.rows == 10
    rows = generated.host()
    assert rows.tokens.shape == (10, 6)
    np.testing.assert_array_equal(rows.tokens, plain.host().tokens)
    np.testing.assert_array_equal(rows.lengths, plain.host().lengths)
    np.testing.assert_allclose(rows.behavior_log_probs, plain.host().behavior_log_probs,
                               atol=2e-6, rtol=2e-6)
    # The rows past the real ones repeat padded prompts, not another prompt's
    # continuations, so dropping them cannot drop a real answer.
    padded = local_rows(generated.tokens)[10:]
    np.testing.assert_array_equal(padded[:, :3], np.repeat(np.asarray(prompt)[:3], 2, axis=0))
