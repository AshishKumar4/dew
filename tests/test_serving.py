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
from dew.inference.pages import Pages
from dew.inference.pipeline import place
from dew.inference.serving import PagedRows, Server
from dew.nn.backbones.causal_transformer import CausalTransformer, Mixture
from dew.nn.kv_cache import KVCache
from dew.nn.sharding import BATCH_AXES
from dew.sampling import Sampling
from dew.training import Layout, MeshSpec

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
    finishes gives its slot to the next request. The capacity is wider
    than the prompt bucket, so a reused slot keeps its former occupant's
    keys past the new prompt, hidden by the validity the admission wrote."""
    bound = task()
    server = Server.from_task(bound, slots=2, capacity=128, admission=1)
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
    fourth = server.submit("1234567", 40, seed=3)
    server.run()
    assert fourth.result().text == bound("1234567", 40, seed=3).text
    fifth = server.submit("5", 40, seed=4)
    server.run()
    assert fifth.result().text == bound("5", 40, seed=4).text
    assert stored(server.cache) == before
    assert [leaf.shape for leaf in jax.tree.leaves(server.cache)] == shapes


def test_more_requests_than_slots_queue_and_all_complete():
    bound = task()
    server = Server.from_task(bound, slots=2, capacity=128, admission=2)
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


def served_alongside(bound, slots=4, admission=2, **options):
    """Every prompt submitted at once, plus the longest again once it is done,
    through a server with `options`; returns the server and the tickets."""
    server = Server.from_task(bound, slots=slots, capacity=128, admission=admission, **options)
    tickets = [server.submit(prompt, budget, seed=index)
               for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
    server.run()
    tickets.append(server.submit(PROMPTS[2], BUDGETS[2], seed=2))
    server.run()
    return server, tickets


@pytest.mark.parametrize("options", [
    {"kv_cache": KVCache(page_size=16, pages=12)},
    {"kv_cache": KVCache(page_size=16, pages=12), "chunk": 2},
    {"kv_cache": KVCache(page_size=4, pages=40), "chunk": 3, "prefix_cache": True},
], ids=["paged", "chunked", "prefix"])
def test_a_paged_server_draws_what_each_request_draws_alone(options):
    """A pool of 12 pages holds fewer tokens than the 4 x 128 slots the dense
    server reserves, a prompt prefilled two or three tokens a step
    attends to its earlier pieces through the page table, and a repeated
    prompt starts from the page its first run published. None of it
    changes a greedy draw: every row is the lone task call's."""
    bound = task()
    alone = [bound(prompt, budget, seed=index)
             for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
    server, tickets = served_alongside(bound, **options)
    for ticket, lone in zip(tickets, [*alone, alone[2]], strict=True):
        assert_same_generation(ticket.result(), lone)
    # "1234567" keeps its last token to prefill: one full page of four is shared.
    assert server.prefix_hits == (4 if options.get("prefix_cache") else 0)


@pytest.mark.mesh(devices=4)
@pytest.mark.parametrize("mesh, options", [
    (MeshSpec(), {}),
    (MeshSpec(tensor=2), {"kv_cache": KVCache(page_size=4, pages=64), "chunk": 3, "prefix_cache": True}),
    (MeshSpec(fsdp=2, tensor=2), {"kv_cache": KVCache(page_size=16, pages=16)}),
], ids=["data", "data-tensor-chunked-prefix", "data-fsdp-tensor-paged"])
def test_a_server_on_a_mesh_draws_what_each_request_draws_alone(mesh, options):
    """Weights placed on a mesh serve on it: the slots and the pool's pages
    split over the row axes, one group of rows per device group, the heads
    over the tensor axis. Every row is still the lone single-device call, a
    chunked prompt included, and the repeated prompt starts from the page the
    first one published in its group."""
    bound = task()
    alone = [bound(prompt, budget, seed=index)
             for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
    placed = bound.bind(place(bound.variables, mesh, Layout(min_shard=1, tolerance=1.0)))
    server, tickets = served_alongside(placed, slots=8, admission=8, **options)
    assert server.groups == jax.device_count() // mesh.tensor
    for ticket, lone in zip(tickets, [*alone, alone[2]], strict=True):
        assert_same_generation(ticket.result(), lone)
    assert server.prefix_hits == (4 if options.get("prefix_cache") else 0)
    if "kv_cache" in options:
        pool = next(leaf for leaf in jax.tree.leaves(server.cache) if leaf.ndim == 4)
        pages = pool.sharding.spec[1]
        assert ((pages,) if isinstance(pages, str) else tuple(pages)) == tuple(
            axis for axis in BATCH_AXES if server.mesh.shape[axis] > 1)


@pytest.mark.mesh(devices=4)
@pytest.mark.parametrize("mesh", [MeshSpec(sequence=2), MeshSpec(stage=2)], ids=["sequence", "stage"])
def test_a_server_refuses_a_mesh_that_splits_a_row_or_the_layer_stack(mesh):
    bound = task()
    with pytest.raises(ValueError, match=r"sequence=1|stage=1"):
        Server.from_task(bound.bind(place(bound.variables, mesh, Layout(min_shard=1, tolerance=1.0))),
                         slots=8, capacity=128)


def test_a_full_pool_queues_a_request_until_a_row_gives_pages_back():
    """Three pages of sixteen hold one request of prompt and budget at a time:
    the second waits in the queue, not in a slot, and draws what it draws
    alone once the first returns its pages."""
    bound = task()
    server = Server.from_task(bound, slots=2, capacity=128, kv_cache=KVCache(page_size=16, pages=3))
    first = server.submit("1234567", 30, seed=0)
    second = server.submit("98", 30, seed=1)
    server.step()
    assert server.occupancy == 1 and server.queued == 1
    server.run()
    assert first.result().text == bound("1234567", 30, seed=0).text
    assert second.result().text == bound("98", 30, seed=1).text
    with pytest.raises(ValueError, match="pool holds 3"):
        server.submit("1234567", 60, seed=0)


@pytest.mark.mesh(devices=4)
@pytest.mark.parametrize("dispatch", ["exchange", "global"])
def test_a_server_on_an_expert_mesh_draws_what_one_device_draws(dispatch):
    """A mixture layer on an expert mesh can run its dispatch in a shard_map,
    and jax's checkify cannot carry a device check into one (jax-ml/jax#40907).
    The server's step runs the model before its draws check anything, and a
    pool smaller than every row's capacity checks nothing on the device, so
    either dispatch serves what the global one draws alone on one device.
    TextGeneration, whose loop carries each draw's checks into the next
    step's model call, refuses the exchange."""
    def generation(dispatch, params):
        model = CausalTransformer(vocab_size=VOCAB, emb_features=16, num_layers=1, num_heads=2, head_dim=8,
                                  mlp_features=32, max_seq_len=128, dtype="float32",
                                  mixture=Mixture(experts=4, top_k=2, dispatch=dispatch))
        params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32)) if params is None else params
        return TextGeneration(model, params, RunProcessor(Digits()), sampling=Sampling(temperature=0, eos_id=EOS))

    lone = generation("global", None)
    alone = [lone(prompt, budget, seed=index)
             for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
    served = generation(dispatch, place(lone.variables, MeshSpec(expert=4), Layout(min_shard=1, tolerance=1.0)))
    server, tickets = served_alongside(served, slots=8, admission=8, kv_cache=KVCache(page_size=16, pages=32))
    assert server.groups == jax.device_count()
    for ticket, row in zip(tickets, [*alone, alone[2]], strict=True):
        assert_same_generation(ticket.result(), row)
    if dispatch == "exchange":
        with pytest.raises(ValueError, match="dispatch='global'"):
            served("12", 3, seed=0)


def test_chunks_and_prefix_sharing_need_a_paged_cache():
    with pytest.raises(ValueError, match="paged cache"):
        Server.from_task(task(), slots=2, capacity=128, chunk=4)


def test_a_paged_server_refuses_a_model_whose_layers_kept_a_dense_cache():
    """A paged server moves rows by their page tables; a cache without any
    means the layout did not reach the layers, and serving it would hold a
    dense slots x capacity cache instead of the bounded pool."""
    bound = task()
    dense = Server.from_task(bound, slots=2, capacity=128)
    with pytest.raises(ValueError, match="did not take the paged layout"):
        PagedRows([Pages(8, 16, prefix_cache=False)], None, 8).check(dense.cache)


def test_a_released_prefix_page_is_shared_until_the_free_pages_run_out():
    """The ledger shares a full prompt page by the hash of everything up to
    its end, keeps it after its row leaves, and reclaims it only when a
    request needs more pages than are free, oldest release first."""
    pages = Pages(4, 2, prefix_cache=True)
    prompt = np.array([1, 2, 3, 4, 5])
    first, hit = pages.reserve(prompt, 6)
    assert hit == 0 and len(first) == 3
    pages.publish(prompt, first)
    pages.release(first)
    assert pages.available == 4
    # Same first four tokens: two pages shared, the last token still prefilled.
    again, hit = pages.reserve(np.array([1, 2, 3, 4, 9]), 6)
    assert hit == 4 and again[:2] == first[:2]
    pages.release(again)
    # A different first page shares nothing, even where later tokens agree.
    other, hit = pages.reserve(np.array([7, 2, 3, 4, 5]), 8)
    assert hit == 0 and len(other) == 4
    pages.release(other)
    assert pages.reserve(np.array([1, 2, 3, 4, 5]), 6)[1] == 0


def test_reclaiming_a_released_chain_takes_its_tail_before_its_head():
    """A shared prefix loses its last page first under memory pressure, so
    the head that every later prompt starts with stays reachable."""
    pages = Pages(3, 2, prefix_cache=True)
    prompt = np.array([1, 2, 3, 4, 5])
    chain, _ = pages.reserve(prompt, 6)
    pages.publish(prompt, chain)
    pages.release(chain)
    # Two pages are needed and one is free: the one reclaimed is the tail.
    taken, _ = pages.reserve(np.array([9, 9, 9]), 4)
    assert taken == [chain[2], chain[1]]
    pages.release(taken)
    again, hit = pages.reserve(np.array([1, 2, 7, 7, 7]), 6)
    assert hit == 2 and again[0] == chain[0]


def test_reloaded_weights_share_no_prefix_page_the_old_weights_wrote():
    """A cached prompt page holds keys the old weights computed; after a
    reload the same prompt prefills again and draws what the new weights
    draw alone."""
    bound = task()
    server = Server.from_task(bound, slots=2, capacity=128, kv_cache=KVCache(page_size=4, pages=40),
                              prefix_cache=True)
    server.submit("1234567", 8, seed=0)
    server.run()
    other = bound.variables.copy({"params": jax.tree.map(lambda leaf: leaf * 1.5, bound.variables["params"])})
    server.reload(other)
    hits = server.prefix_hits
    ticket = server.submit("1234567", 8, seed=0)
    server.run()
    assert server.prefix_hits == hits
    assert ticket.result().text == bound.bind(other)("1234567", 8, seed=0).text
