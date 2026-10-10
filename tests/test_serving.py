"""A served request draws what the same request draws alone.

The server keeps a resident cache and moves rows in and out of it, so the
checks are that no row sees another row's state: a batch of mixed lengths
and budgets, a row admitted while others run, a slot reused after its row
left, a queue longer than the slots, and a request too large for the cache.
"""

import dataclasses
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from model_support import Digits, decoder, serving_task as task
from sharded import assert_sharded
from steady_state import guarded, steady_state

from dew.inference import RunProcessor, TextGeneration
from dew.inference.pages import Pages
from dew.inference.pipeline import place
from dew.inference.serving import PagedRows, Server
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import GatedMLP, Mixture
from dew.nn.kv_cache import KVCache
from dew.nn.mixers.attention import CausalSelfAttention
from dew.nn.sharding import BATCH_AXES
from dew.sampling import Sample, Sampling
from dew.training import Layout, MeshSpec

VOCAB = 13
EOS = 12
PROMPTS = ["12", "5", "1234567", "98", "321"]
BUDGETS = [5, 9, 3, 12, 7]





def assert_same_generation(served, alone):
    """The served row is the lone row: tokens, length, termination and both likelihoods."""
    served, alone = served.host(), alone.host()
    np.testing.assert_array_equal(served.tokens, alone.tokens)
    np.testing.assert_array_equal(served.lengths, alone.lengths)
    np.testing.assert_array_equal(served.terminated, alone.terminated)
    np.testing.assert_allclose(served.behavior_log_probs, alone.behavior_log_probs, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(served.raw_log_probs, alone.raw_log_probs, atol=2e-6, rtol=2e-6)


def test_prepacked_gate_and_up_draw_the_same_gated_projection():
    """The serving layout is one dot, split before the trained activation."""
    with jax.enable_x64():
        model = GatedMLP(hidden_features=6, out_features=4, use_bias=True, dtype=jnp.float64)
        x = jnp.arange(32, dtype=jnp.float64).reshape(2, 2, 8) / 16
        kernel = jnp.arange(96, dtype=jnp.float64).reshape(8, 12) / 128
        bias = jnp.arange(12, dtype=jnp.float64) / 32
        down = jnp.arange(24, dtype=jnp.float64).reshape(6, 4) / 64
        down_bias = jnp.arange(4, dtype=jnp.float64) / 16
        variables = {"params": {"gate_up_proj": {"kernel": kernel, "bias": bias},
                                "down_proj": {"kernel": down, "bias": down_bias}}}
        gate, up = jnp.split(x @ kernel + bias, 2, axis=-1)
        expected = (jax.nn.silu(gate) * up) @ down + down_bias
        np.testing.assert_array_equal(jax.jit(model.apply)(variables, x), expected)


def test_prepacked_queries_keys_and_values_preserve_attention_and_cache():
    """Biases, head norms, RoPE and the cached append still read the same projections."""
    with jax.enable_x64():
        model = CausalSelfAttention(emb_features=8, num_heads=2, num_kv_heads=1,
                                   head_dim=4, max_seq_len=64, attention_bias=True,
                                   attention_impl="xla", dtype=jnp.float64)
        x = jnp.arange(32, dtype=jnp.float64).reshape(1, 4, 8) / 32
        variables = model.init(jax.random.key(0), x)
        # Dyadic weights make the projection sums exact, so a different dot
        # width cannot hide a wrong slice behind a floating-point bound.
        for index, name in enumerate(("q_proj", "k_proj", "v_proj")):
            projection = variables["params"][name]
            projection["kernel"] = jnp.arange(projection["kernel"].size, dtype=jnp.float64).reshape(
                projection["kernel"].shape) / (128 * 2 ** index)
            projection["bias"] = jnp.arange(projection["bias"].size, dtype=jnp.float64) / 32
        packed = {**variables, "params": dict(variables["params"])}
        packed["params"]["qkv_proj"] = {
            name: jnp.concatenate([variables["params"][projection][name]
                                   for projection in ("q_proj", "k_proj", "v_proj")], axis=-1)
            for name in ("kernel", "bias")}
        for name in ("q_proj", "k_proj", "v_proj"):
            del packed["params"][name]
        applied = jax.jit(lambda weights: model.apply(weights, x, decode=True, mutable=["cache"]))
        expected, cache = applied(variables)
        actual, packed_cache = applied(packed)
        np.testing.assert_array_equal(actual, expected)
        for before, after in zip(jax.tree.leaves(cache), jax.tree.leaves(packed_cache), strict=True):
            np.testing.assert_array_equal(after, before)


def test_prepacked_serving_keeps_the_source_and_reloads_its_original_tree():
    """Serving holds the same weight bytes; reloading casts and repacks the trained tree."""
    from flax.core import freeze

    from dew.inference.projections import inference_projections

    bound = task(Sampling(temperature=0, eos_token_ids=None))
    source = jax.tree.map(np.asarray, bound.variables)
    packed_task = TextGeneration(bound.model, jax.device_put(inference_projections(bound.model, source)),
                                 bound.processor, sampling=bound.sampling)
    server = Server.from_task(packed_task, slots=2, capacity=128)
    before = server(["12", "34"], 5, key=3)
    assert sum(leaf.nbytes for leaf in jax.tree.leaves(server.variables)) == sum(
        leaf.nbytes for leaf in jax.tree.leaves(bound.variables))
    for original, saved in zip(jax.tree.leaves(bound.variables), jax.tree.leaves(source), strict=True):
        np.testing.assert_array_equal(original, saved)
    trained = jax.tree.map(lambda leaf: leaf * 1.5, bound.variables)
    server.reload(freeze(trained))
    served = server(["12", "34"], 5, key=3)
    assert any(not np.array_equal(old.host().raw_log_probs, new.host().raw_log_probs)
               for old, new in zip(before, served, strict=True))
    other = TextGeneration(bound.model, trained, bound.processor, sampling=bound.sampling)
    for prompt, actual in zip(("12", "34"), served, strict=True):
        assert_same_generation(actual, other(prompt, 5, key=3))
    fresh = Server.from_task(other, slots=2, capacity=128)
    for actual, expected in zip(served, fresh(["12", "34"], 5, key=3), strict=True):
        assert_same_generation(actual, expected)
    jax.jit(lambda weights: jax.tree.map(lambda leaf: leaf * 2, weights), donate_argnums=(0,))(trained)
    for actual, expected in zip(server(["12", "34"], 5, key=3), served, strict=True):
        assert_same_generation(actual, expected)
    for original, saved in zip(jax.tree.leaves(bound.variables), jax.tree.leaves(source), strict=True):
        np.testing.assert_array_equal(original, saved)


@pytest.mark.parametrize("case", ["kv_shared", "k_eq_v", "output_gate"])
def test_inference_projection_layout_preserves_special_attention_logits(case):
    from dew.inference.projections import inference_projections

    with jax.enable_x64():
        model = CausalTransformer(
            vocab_size=VOCAB, emb_features=16, num_layers=2, num_heads=2, num_kv_heads=1,
            head_dim=8, mlp_features=32, max_seq_len=64, dtype=jnp.float64,
            precision=jax.lax.Precision.HIGHEST, attention_impl="reference",
            kv_shared_layers=(1,) if case == "kv_shared" else None,
            attention_k_eq_v=case == "k_eq_v", v_norm=case == "k_eq_v",
            output_gate=case == "output_gate")
        tokens = jnp.asarray([[1, 2, 3, 4]], jnp.int32)
        variables = jax.tree.map(lambda leaf: np.asarray(leaf, np.float64),
                                 model.init(jax.random.key(13), tokens))
        packed = inference_projections(model, variables)
        expected = jax.jit(model.apply)(variables, tokens)
        actual = jax.jit(model.apply)(packed, tokens)
        # Both run in float64; 64 epsilon covers the tiny decoder's few
        # reassociated dot sums, well inside the existing FP32 parity bound.
        eps = 64 * np.finfo(np.float64).eps
        np.testing.assert_allclose(actual, expected, atol=eps, rtol=eps)
        np.testing.assert_array_equal(jnp.argmax(actual, axis=-1), jnp.argmax(expected, axis=-1))


def test_an_admitting_step_pads_to_the_fewest_rows_that_hold_its_prompts():
    """The admitting program prefills every row it is given: a request
    arriving alone is one row, three together are four, never the whole
    admission of eight (`admission_share`), and each draws what it draws alone."""
    from dew.inference.serving import admission_share

    assert [admission_share(waiting, 8) for waiting in range(1, 9)] == [1, 2, 4, 4, 8, 8, 8, 8]
    assert admission_share(5, 6) == 6
    bound = task()
    server = Server.from_task(bound, slots=8, capacity=128, admission=8)
    widths, admit = [], server._admit

    def recorded():
        admission = admit()
        widths.append(None if admission is None else admission.prompts.tokens.shape[0])
        return admission

    server._admit = recorded
    first = server.submit(PROMPTS[0], 3, key=0)
    server.step()
    rest = [server.submit(prompt, 3, key=index) for index, prompt in enumerate(PROMPTS[1:4], start=1)]
    server.run()
    assert [width for width in widths if width] == [1, 4]
    for index, ticket in enumerate([first, *rest]):
        assert ticket.result().text == bound(PROMPTS[index], 3, key=index).text


@pytest.mark.parametrize("decode_steps", [1, 4])
def test_mixed_lengths_and_budgets_submitted_together_draw_what_each_draws_alone(decode_steps):
    """Five prompts of different widths and budgets, greedy, one seed per
    row: the texts are byte-identical to the one-row task calls, and so is
    everything else the generation carries. With four iterations a call,
    rows end inside a call and wait for the next one's admission."""
    bound = task()
    alone = [bound(prompt, budget, key=index)
             for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
    assert len({generation.text for generation in alone}) == len(alone)
    server = Server.from_task(bound, slots=4, capacity=128, admission=2, decode_steps=decode_steps)
    tickets = [server.submit(prompt, budget, key=index)
               for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
    server.run()
    served = [ticket.result() for ticket in tickets]
    assert [generation.text for generation in served] == [generation.text for generation in alone]
    for mine, theirs in zip(served, alone, strict=True):
        assert_same_generation(mine, theirs)
    assert all(ticket.admitted is not None and ticket.finished is not None for ticket in tickets)


def test_host_task_and_server_preserve_nonzero_lora_branches():
    from dew.lora import LoRA

    bound = task(Sampling(temperature=0, eos_token_ids=None))
    adapter = LoRA(rank=2, modules=("q_proj", "gate_proj")).apply(bound.model, bound.variables, key=17)
    trained = jax.tree_util.tree_map_with_path(
        lambda path, leaf: leaf + jax.random.normal(jax.random.key(29), leaf.shape) * .25
        if getattr(path[-1], "key", None) == "lora_B" else leaf, adapter.variables)
    model = adapter.model
    unplaced = TextGeneration(model, jax.tree.map(np.asarray, trained), bound.processor,
                             sampling=bound.sampling)
    placed = TextGeneration(model, trained, bound.processor, sampling=bound.sampling)
    expected = placed("12", 6, key=3)
    assert not np.array_equal(expected.host().raw_log_probs, bound("12", 6, key=3).host().raw_log_probs)
    assert_same_generation(unplaced("12", 6, key=3), expected)
    served = Server.from_task(unplaced, slots=2, capacity=128)
    assert_same_generation(served(["12"], 6, key=3)[0], expected)


def test_reload_normalizes_source_precision_before_concatenating_projections():
    """A float64 value just above an FP16 midpoint must not double-round through FP32."""
    from dew.inference.projections import inference_projections

    bound = task(Sampling(temperature=0, eos_token_ids=None))
    source = jax.tree.map(lambda leaf: np.asarray(leaf, dtype=np.float16), bound.variables)
    host = TextGeneration(bound.model, jax.device_put(inference_projections(bound.model, source)),
                          bound.processor, sampling=bound.sampling)
    server = Server.from_task(host, slots=2, capacity=128)
    incoming = jax.tree.map(lambda leaf: np.asarray(leaf, np.float64), source)
    kernel = incoming["params"]["layers_0"]["self_attn"]["q_proj"]["kernel"]
    kernel[0, 0] = np.nextafter(1.00048828125, np.inf)
    server.reload(incoming)
    packed = server.variables["params"]["layers_0"]["self_attn"]["qkv_proj"]["kernel"]
    assert float(np.asarray(packed)[0, 0]) == 1.0009765625
    normalized = jax.tree.map(lambda leaf: np.asarray(leaf, dtype=np.float16), incoming)
    other = TextGeneration(bound.model, jax.tree.map(jnp.asarray, normalized), bound.processor,
                           sampling=bound.sampling)
    assert_same_generation(server(["12"], 5, key=3)[0], other("12", 5, key=3))


@pytest.mark.parametrize("dtype", ["int8", "fp8"])
def test_quantized_weights_serve_the_same_greedy_text_as_the_task(dtype):
    from dew.training.quantization import Quantization

    pytest.importorskip("qwix")
    bound = task().quantized(Quantization(dtype=dtype, weight_only=True))
    server = Server.from_task(bound, slots=2, capacity=128)
    prompts = ["12", "567"]
    served = server(prompts, 4, key=3)
    alone = [bound(prompt, 4, key=3) for prompt in prompts]
    assert [result.text for result in served] == [result.text for result in alone]
    for result, expected in zip(served, alone, strict=True):
        assert_same_generation(result, expected)


def test_a_sampled_request_keeps_its_own_draws():
    """A served row's key is the request key folded by row zero, which is
    what a one-row task call folds, so a sampled request draws the same
    tokens alone and served, and a batch call folds by row as a batched
    task call does."""
    bound = task(Sampling(temperature=1.0, top_k=5, eos_token_ids=EOS))
    server = Server.from_task(bound, slots=3, capacity=128)
    tickets = [server.submit(prompt, budget, key=index)
               for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
    server.run()
    for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True)):
        assert_same_generation(tickets[index].result(), bound(prompt, budget, key=index))
    batched = bound(PROMPTS, 6, key=3)
    served = server(PROMPTS, 6, key=3)
    assert tuple(generation.text[0] for generation in served) == batched.text
    rows = batched.host()
    for index, generation in enumerate(served):
        mine = generation.host()
        np.testing.assert_array_equal(mine.tokens[0, generation.prompt_width:],
                                      rows.tokens[index, batched.prompt_width:])
        np.testing.assert_array_equal(mine.lengths[0], rows.lengths[index])


def test_submission_keeps_device_keys_on_device_until_admission():
    """A request can queue its key without waiting for a copy to the host.

    The request still draws the same sampled tokens and likelihoods once it
    is admitted; only the host-side preparation's synchronization changes.
    """
    bound = task(Sampling(temperature=1.0, top_k=5, eos_token_ids=None))
    key = jax.random.key(7)
    prompt = np.asarray([1, 2, 3], np.int32)
    server = Server.from_task(bound, slots=2, capacity=128, admission=2)
    with guarded(allow=("host_to_device",)):
        ticket = server.submit(prompt, 5, key=key)
    server.run()
    assert_same_generation(ticket.result(), bound(prompt[None], 5, key=key))


@pytest.mark.parametrize("seed", [7, 2**31, -1, 2**40 + 3])
def test_a_request_with_an_integer_seed_launches_nothing_until_admission(seed):
    """Submitting with an integer seed moves nothing to the device, not even
    the seed: the admission's program makes the key (`_row_keys`), whose bits
    are an eager `jax.random.key(seed)`'s, wrapped where the seed overflows
    32 bits as eager keys wrap it. The request draws the sampled tokens and
    likelihoods the one-row task draws with that seed."""
    bound = task(Sampling(temperature=1.0, top_k=5, eos_token_ids=None))
    prompt = np.asarray([1, 2, 3], np.int32)
    server = Server.from_task(bound, slots=2, capacity=128, admission=2)
    with guarded():
        ticket = server.submit(prompt, 5, key=seed)
    server.run()
    assert_same_generation(ticket.result(), bound(prompt[None], 5, key=seed))


def test_a_step_donates_the_slot_matrices_and_rewrites_the_vectors():
    """XLA copies a donated buffer that is read after the output is written,
    as every cursor, step count and active flag is, so a step donates only
    the matrices: the resident half holds the cache, tokens and validity,
    the carried half each row's vectors."""
    server = Server.from_task(task(Sampling(temperature=0, eos_token_ids=None)), slots=2, capacity=64)
    resident, carried = server._resident, server._carried
    assert jax.tree.leaves(resident) and all(leaf.ndim >= 2 for leaf in jax.tree.leaves(resident))
    assert all(leaf.ndim < 2 for leaf in jax.tree.leaves(carried))
    for name in ("step", "budget", "active"):
        assert getattr(resident, name) is None and getattr(carried, name) is not None


def test_a_step_without_admission_draws_from_its_own_logits_without_merging_every_slot():
    """With no admission every drawing row fed this step, so the logits it
    draws from are the model's whole: no select over every slot's vocabulary
    keeps a held row's (on an RTX 4080 at 64 slots, 136 us of an 8.4 ms
    step). A step that seats rows still merges, since those draw from their
    prompt's logits."""
    from dew.inference.serving_kernel import _advanced, joined

    server = Server.from_task(task(Sampling(temperature=0, eos_token_ids=None)), slots=2, capacity=64,
                              admission=1)
    server.submit(np.asarray([1, 2, 3], np.int32), 4, key=0)
    admission = server._admit()
    state = joined(server._resident, server._carried)

    def merges(admission):
        jaxpr = jax.make_jaxpr(lambda state: _advanced(
            server.model, server.variables, server.pad_id, server.rows.placement, state, admission,
            server.transforms, server.stopping, server.grammar))(state)
        return _selects(jaxpr.jaxpr, state.decoder.logits.shape)

    assert admission is not None and merges(admission)
    assert not merges(None)


def _selects(jaxpr, shape) -> bool:
    """Whether `jaxpr` or a jaxpr it calls selects an array of `shape`."""
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "select_n" and eqn.outvars[0].aval.shape == shape:
            return True
        for value in eqn.params.values():
            inner = getattr(value, "jaxpr", value)
            if hasattr(inner, "eqns") and _selects(inner, shape):
                return True
    return False


def test_repeated_requests_reuse_their_programs_and_read_back_only_results():
    """Text requests of one bucket after the first run the program it
    compiled, and the host waits on nothing but what a request asks for:
    the device check it raises from and the result. A request's own inputs
    reach the device as it arrives, so only reads are held (`steady_state`).
    A prompt that went to the device and came back for validation, or a
    recompile per request, fails it."""
    bound = task()
    keys = [jax.random.key(index) for index in range(4)]
    bound("12", 6, key=keys[0])
    with steady_state(allow=("host_to_device",)):
        drawn = [bound(prompt, 6, key=key) for prompt, key in zip(["34", "56", "78"], keys[1:], strict=True)]
    assert [generation.tokens.shape for generation in drawn] == [(1, 8)] * 3


def test_a_server_round_after_the_first_reuses_its_programs_and_reads_once_a_step():
    """A second round of the same requests runs the programs the first
    compiled, and each step's one read is the explicit copy of the drawn
    tokens the admission works from (`steady_state`). Token rows, which the
    server keeps on the host until it admits them."""
    server = Server.from_task(task(), slots=4, capacity=128, admission=2, decode_steps=4)
    rounds = [[jax.random.key(10 * round + index) for index in range(len(PROMPTS))] for round in range(2)]
    rows = [np.asarray(Digits().encode(prompt), np.int32) for prompt in PROMPTS]

    def served(keys):
        tickets = [server.submit(prompt, budget, key=key)
                   for prompt, budget, key in zip(rows, BUDGETS, keys, strict=True)]
        server.run()
        return [ticket.result() for ticket in tickets]

    first = served(rounds[0])
    with steady_state(allow=("host_to_device",)):
        second = served(rounds[1])
    for mine, theirs in zip(second, first, strict=True):
        np.testing.assert_array_equal(mine.host().tokens, theirs.host().tokens)


@pytest.mark.parametrize("ids", [np.asarray([1.5, 2.0]), np.asarray([True, False]),
                                  np.asarray([2**32], np.int64), np.asarray([-1, 2], np.int32),
                                  np.asarray([VOCAB], np.int32), np.asarray([], np.int32)])
def test_host_numeric_submission_still_refuses_invalid_token_rows(ids):
    server = Server.from_task(task(), slots=1, capacity=128)
    with pytest.raises(ValueError):
        server.submit(ids, 2, key=jax.random.key(1))
    assert server.queued == 0 and server.occupancy == 0


def test_a_request_admitted_mid_flight_draws_what_it_draws_alone():
    bound = task()
    server = Server.from_task(bound, slots=4, capacity=128, admission=2)
    first = server.submit(PROMPTS[0], 12, key=0)
    for _ in range(3):
        server.step()
    assert server.occupancy == 1
    later = server.submit(PROMPTS[3], 12, key=3)
    server.run()
    assert later.result().text == bound(PROMPTS[3], 12, key=3).text
    assert first.result().text == bound(PROMPTS[0], 12, key=0).text


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
    short = server.submit("12", 2, key=0)
    long = server.submit("5", 9, key=1)
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
    third = server.submit("98", 3, key=2)
    server.run()
    assert third.result().text == bound("98", 3, key=2).text
    fourth = server.submit("1234567", 40, key=3)
    server.run()
    assert fourth.result().text == bound("1234567", 40, key=3).text
    fifth = server.submit("5", 40, key=4)
    server.run()
    assert fifth.result().text == bound("5", 40, key=4).text
    assert stored(server.cache) == before
    assert [leaf.shape for leaf in jax.tree.leaves(server.cache)] == shapes


def test_more_requests_than_slots_queue_and_all_complete():
    bound = task()
    server = Server.from_task(bound, slots=2, capacity=128, admission=2)
    prompts = PROMPTS * 2
    budgets = BUDGETS * 2
    tickets = [server.submit(prompt, budget, key=index)
               for index, (prompt, budget) in enumerate(zip(prompts, budgets, strict=True))]
    assert server.queued == len(tickets)
    server.run()
    assert server.queued == 0 and server.occupancy == 0
    for index, (prompt, budget) in enumerate(zip(prompts, budgets, strict=True)):
        assert tickets[index].result().text == bound(prompt, budget, key=index).text
    assert all(ticket.admitted is not None and ticket.admitted >= ticket.submitted for ticket in tickets)


def test_a_ticket_shows_its_tokens_as_each_step_reads_them_back():
    """A caller streams a request from its ticket: each read of the draws
    replaces `tokens` with a longer prefix and calls back, the first before
    the request finishes, and the last is the generation's own tokens."""
    bound = task()
    server = Server.from_task(bound, slots=2, capacity=128, admission=2, decode_steps=2)
    ticket = server.submit(PROMPTS[2], 7, key=0)
    other = server.submit(PROMPTS[0], 3, key=1)
    seen = []
    ticket.add_tokens_callback(lambda done: seen.append((done.tokens, done.done())))
    ticket.add_tokens_callback(lambda done: 1 / 0)
    server.run()
    generation = ticket.result()
    drawn = tuple(int(token) for token in generation.tokens[0, -7:][:int(generation.lengths[0])])
    snapshots = [tokens for tokens, _ in seen]
    assert len(snapshots) > 1 and snapshots[-1] == drawn == ticket.tokens
    assert all(len(shorter) < len(longer) and longer[:len(shorter)] == shorter
               for shorter, longer in itertools.pairwise(snapshots))
    assert not seen[0][1] and not seen[-1][1]
    assert generation.text == bound(PROMPTS[2], 7, key=0).text
    assert other.tokens and other.result().text == bound(PROMPTS[0], 3, key=1).text


def test_a_request_over_the_capacity_is_refused_as_the_task_refuses_it():
    bound = task(capacity=64)
    server = Server.from_task(bound, slots=2, capacity=64)
    with pytest.raises(ValueError, match="exceeds max_seq_len") as refused:
        bound("1234567", 60, key=0)
    with pytest.raises(ValueError, match="exceeds max_seq_len") as served:
        server.submit("1234567", 60, key=0)
    assert str(served.value) == str(refused.value)
    assert server.queued == 0
    with pytest.raises(ValueError, match="max_seq_len"):
        Server.from_task(bound, slots=2, capacity=65)
    with pytest.raises(ValueError, match="one continuation"):
        Server.from_task(bound.__class__(bound.model, bound.variables, bound.processor,
                                         sampling=bound.sampling, n=2), slots=2, capacity=64)


def test_a_server_holds_its_capacity_in_whole_tiles_not_a_power_of_two():
    """Every decode step's attention reads each slot of a row's cache, so the
    cache holds the capacity asked for, rounded up to whole 64-slot tiles and
    whole pages. A capacity of 384 was bucketed to 512 slots, a third more
    cache memory and attention traffic than any row could use."""
    bound = task(capacity=512)

    def slots(**options):
        server = Server.from_task(bound, slots=2, **options)
        page = options.get("kv_cache", KVCache()).page_size
        # A row's slots: the dense keys' second axis, or its pages' tokens.
        leaves = jax.tree_util.tree_flatten_with_path(server.cache)[0]
        held = {leaf.shape[1] * (page or 1) for path, leaf in leaves
                if jax.tree_util.keystr(path).endswith("['page_table']" if page else "['cached_key']")}
        return server.capacity, held

    assert slots(capacity=384) == (384, {384})
    assert slots(capacity=300) == (320, {320})
    assert slots(capacity=300, kv_cache=KVCache(page_size=128)) == (384, {384})


def test_a_prompt_past_the_bucket_under_the_capacity_is_served_as_the_task_serves_it():
    """A capacity of 384 is no power of two, and a 300-token prompt's
    prefill bucket is 512: the prefill is bounded by the capacity, so the
    server draws what the task draws alone, where a 512-wide prefill did
    not fit the 384-slot rows."""
    bound = task(capacity=512)
    prompt = "".join(str(1 + index % 9) for index in range(300))
    server = Server.from_task(bound, slots=2, capacity=384)
    ticket = server.submit(prompt, 50, key=0)
    server.run()
    assert_same_generation(ticket.result(), bound(prompt, 50, key=0))


def served_alongside(bound, slots=4, admission=2, **options):
    """Every prompt submitted at once, plus the longest again once it is done,
    through a server with `options`; returns the server and the tickets."""
    server = Server.from_task(bound, slots=slots, capacity=128, admission=admission, **options)
    tickets = [server.submit(prompt, budget, key=index)
               for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
    server.run()
    tickets.append(server.submit(PROMPTS[2], BUDGETS[2], key=2))
    server.run()
    return server, tickets


@pytest.mark.parametrize("options", [
    {"kv_cache": KVCache(page_size=16, pages=12)},
    {"kv_cache": KVCache(page_size=16, pages=12), "chunk": 2},
    {"kv_cache": KVCache(page_size=4, pages=40), "chunk": 3, "prefix_cache": True},
    {"kv_cache": KVCache(page_size=4, pages=40), "chunk": 3, "prefix_cache": True, "decode_steps": 3},
    {"chunk": 2},
], ids=["paged", "chunked", "prefix", "prefix-three-steps", "dense-chunked"])
def test_a_paged_server_draws_what_each_request_draws_alone(options):
    """A pool of 12 pages holds fewer tokens than the 4 x 128 slots the dense
    server reserves, a prompt prefilled two or three tokens a step
    attends to its earlier pieces through the page table (or its dense
    row), and a repeated prompt starts from the page its first run
    published. None of it changes a greedy draw: every row is the lone task
    call's. Each runs the mixed admitting step."""
    bound = task()
    alone = [bound(prompt, budget, key=index)
             for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
    server, tickets = served_alongside(bound, **options)
    assert server.mixed_refusal is None
    for ticket, lone in zip(tickets, [*alone, alone[2]], strict=True):
        assert_same_generation(ticket.result(), lone)
    # "1234567" keeps its last token to prefill: one full page of four is shared.
    assert server.prefix_hits == (4 if options.get("prefix_cache") else 0)


@pytest.mark.parametrize("decode_steps", [1, 3])
def test_a_paged_kernel_reads_one_key_of_each_idle_row(monkeypatch, request, decode_steps):
    """A freed row keeps its last request's slot count on the device until
    the next request is admitted, and the paged kernel reads every key under
    the count it is given: the four-slot server's second run decodes one row
    beside three idle ones, each of which reads one key, as the dense path's
    per-row count reads. The kernel here is softmax attention over the
    gathered rows, which records what it was told to read, so every draw is
    still the lone call's. JAX's caches are cleared before and after: the
    patched kernel is only read where the server's step is traced, and a
    trace of it must not serve a later test."""
    from dew.inference import serving_kernel
    from dew.nn.kv_cache import KVStore

    jax.clear_caches()
    request.addfinalizer(jax.clear_caches)

    reads = []

    def recorded(store, query, lengths, softcap):
        jax.debug.callback(lambda counts: reads.append(np.asarray(counts)), lengths)
        keys, values = (jnp.repeat(part, query.shape[1] // part.shape[2], axis=2) for part in store.read())
        scores = jnp.einsum("rhd,rkhd->rhk", query, keys) / np.sqrt(query.shape[-1])
        scores = jnp.where(jnp.arange(keys.shape[1]) < lengths[:, None, None], scores, -jnp.inf)
        return jnp.einsum("rhk,rkhd->rhd", jax.nn.softmax(scores, axis=-1), values).astype(query.dtype)

    monkeypatch.setattr(KVStore, "kernel", lambda store, query: True)
    monkeypatch.setattr(KVStore, "decode", recorded)
    monkeypatch.setattr(CausalSelfAttention, "_page_kernel_runs", lambda layer, query: True)
    monkeypatch.setattr(serving_kernel, "_PROGRAMS", {})
    bound = task()
    alone = bound(PROMPTS[2], BUDGETS[2], key=2)
    server = Server.from_task(bound, slots=4, capacity=128, admission=2, decode_steps=decode_steps,
                              kv_cache=KVCache(page_size=16, pages=12))
    for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True)):
        server.submit(prompt, budget, key=index)
    server.run()
    reads.clear()
    ticket = server.submit(PROMPTS[2], BUDGETS[2], key=2)
    server.run()
    assert_same_generation(ticket.result(), alone)
    assert reads and all(np.sum(counts > 1) <= 1 for counts in reads), reads


@pytest.mark.parametrize("options", [{}, {"kv_cache": KVCache(page_size=16, pages=12)}],
                         ids=["dense", "paged"])
def test_a_server_runs_a_model_with_prediction_depths_without_them(options):
    """The server only samples, so it allocates and seeds no prediction
    depth: a model that declares them admits in the one mixed forward, over
    a dense or a paged cache, its slots hold no prediction cache, and every
    row is the lone task call's."""
    from dew.sampling.strategies import Beam, Speculative

    assert not Sample.drafts and not Beam.drafts and Speculative.drafts
    model = decoder().clone(max_seq_len=128, num_nextn_predict_layers=1)
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    bound = TextGeneration(model, params, RunProcessor(Digits()), sampling=Sampling(temperature=0))
    alone = [bound(prompt, budget, key=index)
             for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
    server, tickets = served_alongside(bound, **options)
    assert server.mixed_refusal is None
    for ticket, lone in zip(tickets, [*alone, alone[2]], strict=True):
        assert_same_generation(ticket.result(), lone)
    held = jax.tree_util.tree_flatten_with_path(server.cache)[0]
    assert not [path for path, _ in held if "mtp" in jax.tree_util.keystr(path).lower()]


@pytest.mark.mesh(devices=4)
@pytest.mark.parametrize("mesh, options", [
    (MeshSpec(), {}),
    (MeshSpec(tensor=2), {"kv_cache": KVCache(page_size=4, pages=64), "chunk": 3, "prefix_cache": True}),
    (MeshSpec(fsdp=2, tensor=2), {"kv_cache": KVCache(page_size=16, pages=16), "decode_steps": 4}),
], ids=["data", "data-tensor-chunked-prefix", "data-fsdp-tensor-paged-four-steps"])
def test_a_server_on_a_mesh_draws_what_each_request_draws_alone(mesh, options):
    """Weights placed on a mesh serve on it: the slots and the pool's pages
    split over the row axes, one group of rows per device group, the heads
    over the tensor axis. Every row is still the lone single-device call, a
    chunked prompt included, and the repeated prompt starts from the page the
    first one published in its group."""
    bound = task()
    alone = [bound(prompt, budget, key=index)
             for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
    placed = bound.bind(place(bound.variables, mesh, Layout(min_shard=1, tolerance=1.0)))
    server, tickets = served_alongside(placed, slots=8, admission=8, **options)
    assert_sharded(placed.variables["params"], server.mesh)
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
    first = server.submit("1234567", 30, key=0)
    second = server.submit("98", 30, key=1)
    server.step()
    assert server.occupancy == 1 and server.queued == 1
    server.run()
    assert first.result().text == bound("1234567", 30, key=0).text
    assert second.result().text == bound("98", 30, key=1).text
    with pytest.raises(ValueError, match="pool holds 3"):
        server.submit("1234567", 60, key=0)


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
        return TextGeneration(
            model, params, RunProcessor(Digits()), sampling=Sampling(temperature=0, eos_token_ids=EOS)
        )

    lone = generation("global", None)
    alone = [lone(prompt, budget, key=index)
             for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
    served = generation(
        dispatch, place(lone.variables, MeshSpec(expert=4), Layout(min_shard=1, tolerance=1.0))
    )
    server, tickets = served_alongside(served, slots=8, admission=8, kv_cache=KVCache(page_size=16, pages=32))
    assert_sharded(served.variables["params"], server.mesh)
    assert server.groups == jax.device_count()
    for ticket, row in zip(tickets, [*alone, alone[2]], strict=True):
        assert_same_generation(ticket.result(), row)
    if dispatch == "exchange":
        with pytest.raises(ValueError, match="dispatch='global'"):
            served("12", 3, key=0)


def test_a_wrapped_decoder_holds_the_capacity_it_is_served_at():
    """A multimodal wrapper declares its language model's max_seq_len, which
    `_sized` read only as the wrapper's own field: Qwen3.5-0.8B served at 384
    slots held a cache of 8192 (4.1 GiB at 32 rows, out of memory at 128,
    and its decode step transposed every full-attention layer's 8192 slots).
    The language model inside is sized, whatever the wrapper declares."""
    from dew.nn.multimodal import MultimodalTransformer
    from dew.nn.vision import GemmaProjector, SiglipVision

    text = CausalTransformer(vocab_size=VOCAB, emb_features=16, num_layers=1, num_heads=2, head_dim=8,
                             mlp_features=32, max_seq_len=1024, dtype="float32")
    model = MultimodalTransformer(
        text, SiglipVision(hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2,
                           image_size=8, patch_size=4),
        GemmaProjector(text_width=16, patches_per_side=2, tokens_per_side=1),
        family="gemma3", image_token_id=1)
    bound = TextGeneration(model, model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32)),
                           RunProcessor(Digits()))
    server = Server.from_task(bound, slots=2, capacity=128)
    held = {leaf.shape[1] for path, leaf in jax.tree_util.tree_leaves_with_path(server.cache)
            if jax.tree_util.keystr(path).endswith("['cached_key']")}
    assert held == {128}


class UserDecoder(nn.Module):
    """A user's own decoder module, no Dew class: a plain module around a
    decoder that forwards the calls a task and a server make, and answers
    nothing a server asks of a model."""

    decoder: nn.Module

    @property
    def vocab_size(self):
        return self.decoder.vocab_size

    @property
    def max_seq_len(self):
        return self.decoder.max_seq_len

    def init_cache(self, batch_size):
        self.decoder.init_cache(batch_size)

    def __call__(self, tokens, **fields):
        return self.decoder(tokens, **fields)

    def states_and_logits(self, tokens, **fields):
        return self.decoder.states_and_logits(tokens, **fields)

    def states_and_logits_at(self, tokens, slots, **fields):
        return self.decoder.states_and_logits_at(tokens, slots, **fields)


class AnsweringDecoder(UserDecoder):
    """`UserDecoder`, answering what a server asks of a model from the decoder it holds."""

    @nn.nowrap
    def with_cache_capacity(self, capacity):
        return self.clone(decoder=self.decoder.with_cache_capacity(capacity))

    @nn.nowrap
    def mixed_admission_refusal(self):
        return self.decoder.mixed_admission_refusal()

    @property
    def cache_rebuild_position(self):
        return self.decoder.cache_rebuild_position

    @nn.nowrap
    def inference_projection_groups(self, variables):
        held = {collection: tree["decoder"] for collection, tree in variables.items() if "decoder" in tree}
        return tuple(dataclasses.replace(group, path=("decoder", *group.path))
                     for group in self.decoder.inference_projection_groups(held))


def wrapped(kind, model, variables, sampling, processor=None):
    """A task over `kind` around `model`, its `variables` under `decoder`."""
    held = {collection: {"decoder": tree} for collection, tree in variables.items()}
    return TextGeneration(kind(model), held, processor, sampling=sampling)


def test_a_user_decoder_module_is_served_as_far_as_it_answers():
    """A server reads a model through what it answers, not its class. A
    user's module around a decoder that answers from it is served at the
    capacity asked for (`CacheCapacity`) and admits in one mixed forward
    (`Serving`); one that answers neither keeps its whole context and
    prefills in a forward of its own, and says why. Both draw what the bare
    decoder draws."""
    bound = task(capacity=1024)
    alone = [bound(prompt, budget, key=index)
             for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
    unanswered = "UserDecoder does not say its layers run a mixed step (Serving)"
    for kind, held, refusal in ((AnsweringDecoder, 128, None), (UserDecoder, 1024, unanswered)):
        user = wrapped(kind, bound.model, bound.variables, bound.sampling, bound.processor)
        server = Server.from_task(user, slots=4, capacity=128, admission=2)
        assert server.mixed_refusal == refusal
        assert {leaf.shape[1] for path, leaf in jax.tree_util.tree_leaves_with_path(server.cache)
                if jax.tree_util.keystr(path).endswith("['cached_key']")} == {held}
        tickets = [server.submit(prompt, budget, key=index)
                   for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
        server.run()
        for ticket, row in zip(tickets, alone, strict=True):
            assert_same_generation(ticket.result(), row)


def test_a_user_decoder_module_is_packed_where_it_names_its_groups():
    """Placement packs the projections a model names (`Serving`)
    at their paths in its variables: a user's module around a decoder that
    names the decoder's is packed, draws what the unpacked weights draw, and
    is served and reloaded from them; one that names none keeps its layout."""
    from dew.inference.projections import inference_projections

    bound = task(Sampling(temperature=0, eos_token_ids=None))
    answering = wrapped(AnsweringDecoder, bound.model, bound.variables, bound.sampling, bound.processor)
    plain = wrapped(UserDecoder, bound.model, bound.variables, bound.sampling, bound.processor)
    members = {"q_proj", "k_proj", "v_proj"}
    packed = inference_projections(answering.model, answering.variables)
    attention = set(packed["params"]["decoder"]["layers_0"]["self_attn"])
    assert "qkv_proj" in attention and not members & attention
    assert members <= set(
        inference_projections(plain.model, plain.variables)["params"]["decoder"]["layers_0"]["self_attn"])
    served = TextGeneration(answering.model, packed, bound.processor, sampling=bound.sampling)
    for index, prompt in enumerate(PROMPTS[:3]):
        assert_same_generation(served(prompt, 5, key=index), bound(prompt, 5, key=index))
    server = Server.from_task(served, slots=2, capacity=128)
    trained = jax.tree.map(lambda leaf: leaf * 1.5, answering.variables)
    server.reload(trained)
    retrained = wrapped(UserDecoder, bound.model, jax.tree.map(lambda leaf: leaf * 1.5, bound.variables),
                        bound.sampling, bound.processor)
    for actual, prompt in zip(server(PROMPTS[:2], 5, key=3), PROMPTS[:2], strict=True):
        assert_same_generation(actual, retrained(prompt, 5, key=3))


def test_a_user_decoder_module_rebuilds_its_cache_where_its_decoder_says_it_goes_stale():
    """Phi-3's LongRoPE keys go stale past position 8, and a model that says
    so (`Serving`) has a crossing row's prefix recomputed: a user's
    module around the decoder draws what the decoder draws across the
    crossing, and a server refuses for it the modes that cannot rebuild."""
    from pathlib import Path

    from dew.interop import Pretrained

    directory = Path(__file__).parent / "fixtures" / "hf" / "phi3-tiny"
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference", max_seq_len=64)
    greedy = Sampling(temperature=0, eos_token_ids=None)
    bare = TextGeneration(loaded.model, loaded.variables, sampling=greedy)
    user = wrapped(AnsweringDecoder, loaded.model, loaded.variables, greedy)
    prompts = np.load(directory / "input_ids.npy").astype(np.int32)[:, :4]
    assert_same_generation(user(prompts, 8, key=0), bare(prompts, 8, key=0))
    with pytest.raises(ValueError, match=r"LongRoPE crossing position 8.*chunked"):
        Server.from_task(user, slots=2, capacity=64, chunk=2)


def test_prefix_sharing_needs_a_paged_cache():
    with pytest.raises(ValueError, match="paged cache"):
        Server.from_task(task(), slots=2, capacity=128, prefix_cache=True)


@pytest.mark.parametrize("scan_layers", [False, True])
def test_a_plain_decoder_takes_the_mixed_step_scanned_or_not(scan_layers):
    """A decoder answers for its plain attention whether the stack is a loop
    or scanned (a decode runs the plain loop, so a scanned stack's caches sit
    under its layers too): neither loses the mixed admitting step."""
    model = CausalTransformer(vocab_size=VOCAB, emb_features=16, num_layers=2, num_heads=2, head_dim=8,
                              mlp_features=32, max_seq_len=128, dtype="float32", scan_layers=scan_layers)
    bound = TextGeneration(model, model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32)),
                           RunProcessor(Digits()))
    assert Server.from_task(bound, slots=4, capacity=128).mixed_refusal is None


@pytest.mark.parametrize("kind", ["llama4", "mla", "gated_delta_net"])
def test_a_layer_that_cannot_run_the_mixed_step_keeps_two_forwards_by_name(kind):
    """A cache any layer but plain attention holds keeps the admitting step's
    two forwards, named: such a layer does not read the mixed call's layout
    and would take its one row of tokens as one sequence (a Llama 4 server
    failed to build). Served rows are still each request's alone."""
    from dew.nn.llama4 import Llama4Mixer
    from dew.nn.mixers.gated_delta_net import GatedDeltaNetMixer
    from dew.nn.mla import MLAMixer

    linear = GatedDeltaNetMixer(linear_num_key_heads=2, linear_num_value_heads=2,
                                linear_key_head_dim=8, linear_value_head_dim=8)
    extra = {
        "llama4": {"mixer": Llama4Mixer()},
        "mla": {"mixer": MLAMixer(kv_lora_rank=8, qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=8)},
        "gated_delta_net": {"layer_types": ("linear", "full"), "kinds": {"linear": {"mixer": linear}}},
    }[kind]
    model = CausalTransformer(vocab_size=VOCAB, emb_features=16, num_layers=2, num_heads=2, head_dim=8,
                              mlp_features=32, max_seq_len=128, dtype="float32", qk_norm=False, **extra)
    bound = TextGeneration(model, model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32)),
                           RunProcessor(Digits()))
    server = Server.from_task(bound, slots=4, capacity=128)
    assert "which a mixed step does not run" in (server.mixed_refusal or "")
    tickets = [server.submit(prompt, 3, key=index) for index, prompt in enumerate(PROMPTS[:2])]
    server.run()
    for index, ticket in enumerate(tickets):
        assert_same_generation(ticket.result(), bound(PROMPTS[index], 3, key=index))


def test_a_model_the_mixed_step_cannot_take_keeps_two_forwards_and_says_why(monkeypatch):
    """A layer that refuses the mixed admitting step names why; the server
    keeps the prompts' prefill in a forward of its own and reports the
    reason, and refuses a dense chunked prefill, which only the mixed step
    serves."""
    from dew.nn.mixers.attention import CausalSelfAttention

    monkeypatch.setattr(CausalSelfAttention, "mixed_refusal", lambda self: f"{self.name}: a test refusal")
    server = Server.from_task(task(), slots=2, capacity=128)
    assert "a test refusal" in (server.mixed_refusal or "") and not server.rows.placement.mixed
    with pytest.raises(ValueError, match="a test refusal"):
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
    server.submit("1234567", 8, key=0)
    server.run()
    other = bound.variables.copy({"params": jax.tree.map(lambda leaf: leaf * 1.5, bound.variables["params"])})
    server.reload(other)
    hits = server.prefix_hits
    ticket = server.submit("1234567", 8, key=0)
    server.run()
    assert server.prefix_hits == hits
    assert ticket.result().text == bound.bind(other)("1234567", 8, key=0).text


def test_a_text_prompt_through_the_hf_processor_serves_what_the_task_draws():
    """The server derives the same positions from its row cursor as the HF processor."""
    from pathlib import Path

    from dew.interop import PretrainedDecoder

    fixture = Path(__file__).parent / "fixtures/hf/qwen38-dense-tiny"
    bundle = PretrainedDecoder.load(str(fixture), dtype=jnp.float32, max_seq_len=64)
    bound = bundle.text_generation(sampling=Sampling(temperature=0))
    server = Server.from_task(bound, slots=2, capacity=64)
    ticket = server.submit("hello there", 4, key=0)
    server.run()
    np.testing.assert_array_equal(ticket.result().tokens, bound("hello there", 4, key=0).tokens)


def test_a_served_request_is_forced_to_end_at_its_own_budget():
    """ForcedEOS forces EOS at a request's own last token, not at the slots'
    capacity, which a server's rows hold whatever their budget."""
    from dew.sampling import decoding

    forced = (decoding.Greedy(), decoding.ForcedEOS(jnp.array([5], jnp.int32)))
    bound = dataclasses.replace(task(Sampling(temperature=0)), logits=forced)
    server = Server.from_task(bound, slots=2, capacity=64)
    ticket = server.submit("12", 3, key=0)
    server.run()
    expected = bound("12", 3, key=0)
    np.testing.assert_array_equal(ticket.result().tokens, expected.tokens)
    assert int(np.asarray(ticket.result().tokens)[0, -1]) == 5


@pytest.mark.parametrize("through", ["submit", "batch"])
@pytest.mark.parametrize("fields, conditioning, refusal", [
    ({"positions": jnp.array([[5, 6, 7]], jnp.int32)}, {}, "positions"),
    ({"attention_mask": jnp.array([[1, 2, 1]], jnp.int32)}, {}, "binary"),
    ({}, {"pixel_values": jnp.zeros((1, 3, 4, 4))}, "media"),
], ids=["noncanonical positions", "a validity of 2", "media"])
def test_a_prompt_the_row_cursor_cannot_reproduce_is_refused(through, fields, conditioning, refusal):
    """Only the positions the row cursor reproduces are safe to drop, and a
    served row carries tokens and binary validity alone: a batch through the
    server's call is held to what `submit` holds one row to."""
    from dew.nn.inputs import ModelInputs

    bound = task(Sampling(temperature=0))
    server = Server.from_task(bound, slots=2, capacity=64)
    inputs = ModelInputs(jnp.array([[1, 2, 3]], jnp.int32), fields, conditioning)
    with pytest.raises(ValueError, match=refusal):
        server.submit(inputs, 4, key=0) if through == "submit" else server(inputs, 4, key=0)


def test_a_padding_positions_value_is_not_part_of_a_served_row():
    """The server strips padding and derives valid-token positions from its cursor."""
    from dew.nn.inputs import ModelInputs

    bound = task(Sampling(temperature=0))
    server = Server.from_task(bound, slots=2, capacity=64)
    inputs = ModelInputs(jnp.array([[0, 1, 2]], jnp.int32),
                         {"attention_mask": jnp.array([[False, True, True]]),
                          "positions": jnp.array([[1, 0, 1]], jnp.int32)})
    ticket = server.submit(inputs, 4, key=0)
    server.run()
    expected = bound(jnp.array([[1, 2]], jnp.int32), 4, key=0)
    np.testing.assert_array_equal(ticket.result().tokens, expected.tokens)
