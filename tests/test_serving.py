"""A served request draws what the same request draws alone.

The server keeps a resident cache and moves rows in and out of it, so the
checks are that no row sees another row's state: a batch of mixed lengths
and budgets, a row admitted while others run, a slot reused after its row
left, a queue longer than the slots, and a request too large for the cache.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.inference import RunProcessor, TextGeneration
from dew.inference.serving import Server
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.sampling import Sampling

VOCAB = 13
EOS = 12
PROMPTS = ["12", "5", "1234567", "98", "321"]
BUDGETS = [5, 9, 3, 12, 7]


class Digits:
    def encode(self, text):
        return [int(character) for character in text]

    def decode(self, ids):
        return "".join(str(int(token)) for token in ids)


def task(sampling=Sampling(temperature=0, eos_id=EOS), capacity=128):
    model = CausalTransformer(vocab_size=VOCAB, emb_features=16, num_layers=1, num_heads=2,
                              head_dim=8, mlp_features=32, max_seq_len=capacity, dtype="float32")
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    return TextGeneration(model, params, RunProcessor(Digits()), sampling=sampling)


def assert_same_generation(served, alone):
    """The served row is the lone row: tokens, length, termination and both likelihoods."""
    served, alone = served.host(), alone.host()
    np.testing.assert_array_equal(served.tokens, alone.tokens)
    np.testing.assert_array_equal(served.lengths, alone.lengths)
    np.testing.assert_array_equal(served.terminated, alone.terminated)
    np.testing.assert_allclose(served.behavior_log_probs, alone.behavior_log_probs, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(served.raw_log_probs, alone.raw_log_probs, atol=2e-6, rtol=2e-6)


def test_mixed_lengths_and_budgets_submitted_together_draw_what_each_draws_alone():
    """Five prompts of different widths and budgets, greedy, one seed per
    row: the texts are byte-identical to the one-row task calls, and so is
    everything else the generation carries."""
    bound = task()
    alone = [bound(prompt, budget, seed=index)
             for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
    assert len({generation.text for generation in alone}) == len(alone)
    server = Server.from_task(bound, slots=4, capacity=128, admission=2)
    tickets = [server.submit(prompt, budget, seed=index)
               for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
    server.run()
    served = [ticket.result() for ticket in tickets]
    assert [generation.text for generation in served] == [generation.text for generation in alone]
    for mine, theirs in zip(served, alone, strict=True):
        assert_same_generation(mine, theirs)
    assert all(ticket.admitted is not None and ticket.finished is not None for ticket in tickets)


def test_a_sampled_request_keeps_its_own_draws():
    """A served row's key is the request key folded by row zero, which is
    what a one-row task call folds, so a sampled request draws the same
    tokens alone and served, and a batch call folds by row as a batched
    task call does."""
    bound = task(Sampling(temperature=1.0, top_k=5, eos_id=EOS))
    server = Server.from_task(bound, slots=3, capacity=128)
    tickets = [server.submit(prompt, budget, seed=index)
               for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
    server.run()
    for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True)):
        assert_same_generation(tickets[index].result(), bound(prompt, budget, seed=index))
    batched = bound(PROMPTS, 6, seed=3)
    served = server(PROMPTS, 6, seed=3)
    assert tuple(generation.text[0] for generation in served) == batched.text
    rows = batched.host()
    for index, generation in enumerate(served):
        mine = generation.host()
        np.testing.assert_array_equal(mine.tokens[0, generation.prompt_width:],
                                      rows.tokens[index, batched.prompt_width:])
        np.testing.assert_array_equal(mine.lengths[0], rows.lengths[index])


def test_a_request_admitted_mid_flight_draws_what_it_draws_alone():
    bound = task()
    server = Server.from_task(bound, slots=4, capacity=128, admission=2)
    first = server.submit(PROMPTS[0], 12, seed=0)
    for _ in range(3):
        server.step()
    assert server.occupancy == 1
    later = server.submit(PROMPTS[3], 12, seed=3)
    server.run()
    assert later.result().text == bound(PROMPTS[3], 12, seed=3).text
    assert first.result().text == bound(PROMPTS[0], 12, seed=0).text


def stored(cache):
    """The key and value buffers' device addresses: the leaves a step
    updates in place. The cursor and validity vectors are recomputed whole
    each step, and XLA is free to give those a fresh buffer."""
    return [leaf.unsafe_buffer_pointer() for leaf in jax.tree.leaves(cache) if leaf.ndim >= 3]


def test_a_slot_is_reused_over_the_same_cache():
    """The cache is one allocation for the server's life: the step donates
    it, so its key and value buffers are updated in place, and a row that
    finishes gives its slot to the next request."""
    bound = task()
    server = Server.from_task(bound, slots=2, capacity=64, admission=1)
    before = stored(server.cache)
    assert len(before) == 2
    shapes = [leaf.shape for leaf in jax.tree.leaves(server.cache)]
    assert all(shape[0] == 2 for shape in shapes)
    short = server.submit("12", 2, seed=0)
    long = server.submit("5", 9, seed=1)
    peak = 0
    seen_drop = False
    while not long.done():
        server.step()
        peak = max(peak, server.occupancy)
        if short.done() and not long.done():
            seen_drop = seen_drop or server.occupancy == 1
    server.run()
    assert peak == 2 and seen_drop
    assert server.occupancy == 0
    third = server.submit("98", 3, seed=2)
    server.run()
    assert third.result().text == bound("98", 3, seed=2).text
    assert stored(server.cache) == before
    assert [leaf.shape for leaf in jax.tree.leaves(server.cache)] == shapes


def test_more_requests_than_slots_queue_and_all_complete():
    bound = task()
    server = Server.from_task(bound, slots=2, capacity=64, admission=2)
    prompts = PROMPTS * 2
    budgets = BUDGETS * 2
    tickets = [server.submit(prompt, budget, seed=index)
               for index, (prompt, budget) in enumerate(zip(prompts, budgets, strict=True))]
    assert server.queued == len(tickets)
    server.run()
    assert server.queued == 0 and server.occupancy == 0
    for index, (prompt, budget) in enumerate(zip(prompts, budgets, strict=True)):
        assert tickets[index].result().text == bound(prompt, budget, seed=index).text
    assert all(ticket.admitted is not None and ticket.admitted >= ticket.submitted for ticket in tickets)


def test_a_request_over_the_capacity_is_refused_as_the_task_refuses_it():
    bound = task(capacity=64)
    server = Server.from_task(bound, slots=2, capacity=64)
    with pytest.raises(ValueError, match="exceeds max_seq_len") as refused:
        bound("1234567", 60, seed=0)
    with pytest.raises(ValueError, match="exceeds max_seq_len") as served:
        server.submit("1234567", 60, seed=0)
    assert str(served.value) == str(refused.value)
    assert server.queued == 0
    with pytest.raises(ValueError, match="max_seq_len"):
        Server.from_task(bound, slots=2, capacity=65)
    with pytest.raises(ValueError, match="one continuation"):
        Server.from_task(bound.__class__(bound.model, bound.variables, bound.processor,
                                         sampling=bound.sampling, n=2), slots=2, capacity=64)
