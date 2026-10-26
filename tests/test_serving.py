"""A served request draws what the same request draws alone.

The server keeps a resident cache and moves rows in and out of it, so the
checks are that no row sees another row's state: a batch of mixed lengths
and budgets, a row admitted while others run, a slot reused after its row
left, a queue longer than the slots, and a request too large for the cache.
"""

import dataclasses
import functools
import itertools
import threading

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from episode_support import ToolPolicy
from flax import linen as nn
from jax.experimental import checkify
from model_support import Digits, decoder, serving_task as task
from reference_error import as_given, assert_as_exact
from sharded import assert_sharded
from steady_state import guarded, steady_state

from dew.inference import RunProcessor, TextGeneration
from dew.inference.loading import place
from dew.inference.serving import Server
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.inputs import MixerCall
from dew.nn.kv_cache import KVCache
from dew.nn.mixer_base import MixerContext
from dew.nn.mixers.attention import CausalSelfAttention
from dew.nn.protocols import held_groups
from dew.nn.sharding import BATCH_AXES
from dew.sampling import Sampling
from dew.training import Layout, MeshSpec

VOCAB = 13
EOS = 12
PROMPTS = ["12", "5", "1234567", "98", "321"]
BUDGETS = [5, 9, 3, 12, 7]
MESH_SEEDS = 16
"""The mesh server's declared seeds for check 1 (`assert_as_exact`), sized by `seeds_for` (2026-10-21,
on armada: 16 for each of the three meshes)."""





def assert_same_generation(served, alone):
    """The served row is the lone row: tokens, length, termination and both likelihoods."""
    served, alone = served.host(), alone.host()
    np.testing.assert_array_equal(served.tokens, alone.tokens)
    np.testing.assert_array_equal(served.lengths, alone.lengths)
    np.testing.assert_array_equal(served.terminated, alone.terminated)
    np.testing.assert_allclose(served.behavior_log_probs, alone.behavior_log_probs, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(served.raw_log_probs, alone.raw_log_probs, atol=2e-6, rtol=2e-6)


@pytest.mark.mesh(devices=2)
def test_a_meshless_server_draws_on_the_selected_default_device():
    """Weights and resident slots both follow a nonzero default device, so the server can compile
    its step, finish a request and release its slot there."""
    device = jax.local_devices()[1]
    with jax.default_device(device):
        served = task()
        server = Server.from_task(served, slots=1, capacity=128)
        result = server.submit("12", max_new_tokens=3, key=0)
        for _ in range(4):
            if result.done():
                break
            server.step()
        assert result.done()
        draw = result.result()
        assert int(draw.lengths[0]) > 0
        assert all(leaf.devices() == {device} for leaf in jax.tree.leaves(server.variables))
        assert all(leaf.devices() == {device} for leaf in jax.tree.leaves(server.cache))


def test_prepacked_queries_keys_and_values_preserve_attention_and_cache():
    """Biases, head norms, RoPE and the cached append still read the same projections."""
    with jax.enable_x64():
        model = CausalSelfAttention(MixerContext(emb_features=8, num_heads=2, num_kv_heads=1, head_dim=4,
                                                 max_seq_len=64, attention_bias=True, attention_impl="xla",
                                                 dtype=jnp.float64))
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
        applied = jax.jit(lambda weights: model.apply(weights, x, MixerCall(decode=True), mutable=["cache"]))
        expected, cache = applied(variables)
        actual, packed_cache = applied(packed)
        np.testing.assert_array_equal(actual, expected)
        for before, after in zip(jax.tree.leaves(cache), jax.tree.leaves(packed_cache), strict=True):
            np.testing.assert_array_equal(after, before)


def test_a_server_holds_its_weights_at_decoding_storage_and_reloads_into_it():
    """A server over a bf16 decoder's fp32 masters holds what a load at bf16
    storage holds, its head bias fp32, and a reload of trained fp32 weights
    casts them as a server over that load casts them: the two hold the same
    leaves and draw the same tokens."""
    from pathlib import Path

    from dew.interop import Pretrained

    masters, native = (Pretrained.load(Path(__file__).parent / "fixtures/hf/phi-tiny", dtype="bfloat16",
                                       attention_impl="xla", max_seq_len=64, **storage)
                       for storage in ({}, {"param_dtype": "bfloat16"}))
    assert native.variables["params"]["head_bias"].dtype == jnp.float32
    sampling = Sampling(temperature=0, eos_token_ids=None)
    wide, narrow = (Server.from_task(source.text_generation(sampling=sampling), slots=2, capacity=32)
                    for source in (masters, native))

    def same(left, right):
        for (path, actual), expected in zip(jax.tree_util.tree_leaves_with_path(left), jax.tree.leaves(right),
                                            strict=True):
            assert actual.dtype == expected.dtype, path
            np.testing.assert_array_equal(actual, expected, err_msg=str(path))

    same(wide.variables, narrow.variables)
    trained = jax.tree.map(lambda leaf: leaf * 1.5, masters.variables)
    for server in (wide, narrow):
        server.reload(trained)
    same(wide.variables, narrow.variables)
    for actual, expected in zip(wide([[3, 4, 5]], 5, key=3), narrow([[3, 4, 5]], 5, key=3), strict=True):
        assert_same_generation(actual, expected)


def test_a_dropped_server_frees_its_weights_without_a_collection():
    """A server and its task, once their last references go, free their weights, the packed copy
    the server serves included, with the cyclic collector off: reading which projections the
    model packs binds it over the weights' shapes, since flax's bound tree is cyclic and, bound
    over the weights themselves, would keep them alive until a full collection."""
    import gc
    from collections import Counter

    def live() -> Counter:
        return Counter(tuple(array.shape) for array in jax.live_arrays())

    def serve() -> Counter:
        # Drawn under jit, as a run draws them: an eager `init` leaves its own cyclic scope
        # holding what it drew, which is flax's to drop, not the server's.
        model = decoder().clone(max_seq_len=128)
        params = jax.jit(model.init)(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
        bound = TextGeneration(model, params, RunProcessor(Digits()),
                               sampling=Sampling(temperature=0, eos_token_ids=None))
        server = Server.from_task(bound, slots=2, capacity=128)
        server(["12", "34"], 3, key=0)
        return Counter(tuple(leaf.shape) for leaf in jax.tree.leaves(bound.variables))

    gc.collect()
    gc.disable()
    try:
        before = live()
        weights = serve()
        after = live()
    finally:
        gc.enable()
    kept = {shape: after[shape] - before[shape] for shape in weights if after[shape] > before[shape]}
    assert not kept, kept


def _host_loaded(dtype="float32"):
    from pathlib import Path

    from dew.interop import Pretrained

    loaded = Pretrained.load(Path(__file__).parent / "fixtures/hf/phi-tiny", dtype=dtype,
                             attention_impl="xla", max_seq_len=64, on_host=True)
    assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(loaded.variables["params"]))
    return loaded.text_generation(sampling=Sampling(temperature=0, eos_token_ids=None))


def test_a_served_length_limit_truncates_a_request_as_a_direct_call_does():
    """A task whose criteria cut a request short at a length (`MaxNewTokens`) before its budget
    finishes it unterminated through a server, as the task's own call does, and a task that ends
    it on a criterion of its own finishes it terminated through both."""
    from dew.sampling import Sampling, decoding

    def at_two(state, tokens):
        return state.step >= 2

    prompt = np.asarray([1, 2, 3], np.int32)
    for stopping, terminated in (((decoding.MaxNewTokens(2),), False), ((at_two,), True)):
        bound = dataclasses.replace(task(Sampling(temperature=0, eos_token_ids=None)), stopping=stopping)
        direct = bound(prompt[None], 8, key=0)
        server = Server.from_task(bound, slots=2, capacity=32)
        ticket = server.submit(prompt, 8, key=0)
        server.run()
        served = ticket.result()
        assert direct.lengths.tolist() == served.lengths.tolist() == [2]
        assert direct.terminated.tolist() == served.terminated.tolist() == [terminated]


def test_a_server_over_loaded_weights_places_them_once():
    """`Pretrained.load(on_host=True)` returns its weights in host memory, which a task keeps as given
    and places on the device once, at its first generation, as decoding reads them (`resident`); a server over
    the task serves that same device copy. Decode steps after a request's admission move nothing
    from the host, and the server draws the task's tokens."""
    bound = _host_loaded()
    server = Server.from_task(bound, slots=2, capacity=32)
    assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(bound.variables["params"]))
    for placed, served in zip(jax.tree.leaves(bound.placed), jax.tree.leaves(server.variables), strict=True):
        assert isinstance(placed, jax.Array) and placed is served
    prompt = np.asarray([1, 2, 3], np.int32)
    ticket = server.submit(prompt, 8, key=0)
    server.step()  # admits the request, whose rows move to the device
    with guarded():
        server.step()
        server.step()
    server.run()
    assert_same_generation(ticket.result(), bound(prompt[None], 8, key=0))


@pytest.mark.mesh(devices=2)
def test_a_task_places_loaded_weights_on_the_default_device_as_its_compute_reads_them():
    """A task over host weights places them where JAX's default device is, committed there, and
    in the storage decoding reads (`decoding_storage`), cast on the host: a bfloat16 load's float32
    masters reach the device as the host's own bfloat16 rounding of them, no float32 copy."""
    from dew.interop.weights import decoding_storage

    with jax.default_device(jax.devices()[1]):
        bound = _host_loaded("bfloat16")
        placed = bound.placed
    assert {leaf.devices().pop() for leaf in jax.tree.leaves(placed)} == {jax.devices()[1]}
    # Committed there, so a call made outside the context runs on them rather than moving them.
    assert all(leaf.committed for leaf in jax.tree.leaves(placed))
    cast = decoding_storage(bound.model, bound.variables)
    for leaf, host in zip(jax.tree.leaves(placed), jax.tree.leaves(cast), strict=True):
        assert leaf.dtype == host.dtype
        np.testing.assert_array_equal(np.asarray(leaf), host)
    assert jnp.dtype(jnp.bfloat16) in {leaf.dtype for leaf in jax.tree.leaves(placed)}


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


def test_a_server_compiles_its_decoding_step_once_and_each_admission_width_once():
    """A server's warm-up is its step programs' compiles: building it compiles the decoding step,
    which every decode-only call then runs without compiling again, and each admission width it
    meets compiles its admitting step at most once, so the widths 1, 2, 4 and 8 cost at most four
    admitting compiles beside the one decoding step, and meeting them again compiles nothing."""
    from dew.telemetry.instrumentation import Compiles

    with Compiles("_stepped") as compiled:
        server = Server.from_task(task(), slots=8, capacity=128, admission=8)
        assert compiled.count == 1, "building a server compiles its decoding step"
        server([PROMPTS[0]], 12, key=0)
        assert compiled.count == 2, "one admission and the decode-only calls after it compile one program"
        widths = (1, 2, 4, 8)
        for width in widths:
            server([PROMPTS[0]] * width, 2, key=0)
        first = compiled.count
        assert first <= 1 + len(widths)
        for width in widths:
            server([PROMPTS[0]] * width, 2, key=0)
        assert compiled.count == first, "a width the server has met compiles nothing again"


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
    from dew.quantization import Quantization

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


def test_a_request_keyed_by_another_prng_implementation_is_refused_before_it_queues():
    """A server makes each admission's keys with the default PRNG implementation, so a key of
    another one (rbg) is refused at submission, before anything queues, and a request beside it
    draws the sampled tokens and likelihoods the one-row task draws."""
    bound = task(Sampling(temperature=1.0, top_k=5, eos_token_ids=None))
    prompt = np.asarray([1, 2, 3], np.int32)
    server = Server.from_task(bound, slots=2, capacity=128, admission=2)
    with pytest.raises(ValueError):
        server.submit(prompt, 5, key=jax.random.key(7, impl="rbg"))
    ticket = server.submit(prompt, 5, key=3)
    server.run()
    assert_same_generation(ticket.result(), bound(prompt[None], 5, key=3))


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


@pytest.mark.parametrize("decode_steps", [1, 4])
def test_a_step_whose_draw_has_no_distribution_stops_the_server(decode_steps):
    """A bigram policy that draws 5 after every id but 5, after which every score is 3e38, which
    a temperature of 0.5 doubles past float32's range: the step after the first draw has no
    distribution to draw from, which the device checks. The server raises that check and stops,
    whether the failing step is a call of its own or a later step of the call that admitted the
    row."""
    vocab = 16

    def logits():
        table = np.full((vocab, vocab), -3e38, np.float32)
        table[:, 5] = 0.0
        table[5] = 3e38
        return table

    model = ToolPolicy(vocab_size=vocab, max_seq_len=64, logits=logits)
    params = model.init(jax.random.key(0), jnp.ones((1, 1), jnp.int32))
    bound = TextGeneration(model, params, None, sampling=Sampling(temperature=0.5, eos_token_ids=None))
    server = Server.from_task(bound, slots=2, capacity=64, decode_steps=decode_steps)
    server.submit(np.asarray([1, 2, 3], np.int32), 4, key=0)
    with pytest.raises(checkify.JaxRuntimeError):
        server.run()
    with pytest.raises(RuntimeError):
        server.step()


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


@pytest.mark.parametrize("options", [{}, {"kv_cache": KVCache(page_size=16, pages=12)}],
                         ids=["dense", "paged"])
def test_requests_submitted_while_another_thread_steps_are_served_in_order(options):
    """A thread submits while another steps the server: each request is
    served as the task serves it alone, admitted no earlier than it was
    submitted and no earlier than any request submitted before it."""
    bound = task()
    server = Server.from_task(bound, slots=2, capacity=128, admission=2, **options)
    prompts, budgets = PROMPTS * 3, BUDGETS * 3
    tickets, submitted = [], threading.Event()

    def submit():
        for index, (prompt, budget) in enumerate(zip(prompts, budgets, strict=True)):
            tickets.append(server.submit(prompt, budget, key=index))
        submitted.set()

    submitter = threading.Thread(target=submit)
    submitter.start()
    while not submitted.is_set() or server.queued or server.occupancy:
        server.step()
    submitter.join()
    server.run()
    for index, (prompt, budget) in enumerate(zip(prompts, budgets, strict=True)):
        assert tickets[index].result().text == bound(prompt, budget, key=index).text
    assert all(ticket.admitted >= ticket.submitted for ticket in tickets)
    assert all(before.admitted <= after.admitted for before, after in itertools.pairwise(tickets))


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


@pytest.mark.parametrize("options", [{}, {"kv_cache": KVCache(page_size=16, pages=12)}],
                         ids=["dense", "paged"])
def test_a_multimodal_model_serves_over_its_language_models_cache(options):
    """A wrapper that conditions its language model on images, as Qwen3.5 and Gemma 3 load,
    holds its cache in that language model, whose layout a server replaces through the wrapper
    (`Layered.with_kv_cache`), dense or paged: every served row of a text prompt is the greedy
    continuation a fresh full forward gives at each token, and the lone task call's."""
    from model_support import greedy_walk
    from tiny_models import build_model, tiny_inputs

    model = build_model("multimodal_transformer", dtype="float32").with_cache_capacity(128)
    args, fields = tiny_inputs("multimodal_transformer", jax.random.key(0))
    params = model.init(jax.random.key(1), *args, **fields)
    bound = TextGeneration(model, params, RunProcessor(Digits()), sampling=Sampling(temperature=0))
    alone = [bound(prompt, budget, key=index)
             for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
    _, tickets = served_alongside(bound, **options)
    for ticket, lone, prompt, budget in zip(tickets, [*alone, alone[2]], [*PROMPTS, PROMPTS[2]],
                                            [*BUDGETS, BUDGETS[2]], strict=True):
        served = ticket.result()
        assert_same_generation(served, lone)
        ids = np.asarray([Digits().encode(prompt)], np.int32)
        # A generation's tokens are the prompt and its continuation.
        walked = greedy_walk(model, params, ids, budget)
        np.testing.assert_array_equal(served.host().tokens[:, :walked.shape[1]], walked)


def hybrid_task():
    """Qwen3.5's stack at toy size: two gated-delta-net layers, whose recurrent memory and conv
    tail are each row's own state, and one full-attention layer, whose keys a paged server pages."""
    model = CausalTransformer(
        vocab_size=VOCAB, emb_features=16, num_layers=3, num_heads=2, num_kv_heads=2, head_dim=8,
        mlp_features=32, max_seq_len=128, dtype="float32",
        layer_types=("linear_attention",) * 2 + ("full_attention",),
        kinds={"linear_attention": {"mixer": {"class": "gated_delta_net", "fields": {
            "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 6,
            "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4}}}})
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    return TextGeneration(model, params, RunProcessor(Digits()),
                          sampling=Sampling(temperature=0, eos_token_ids=EOS))


@pytest.mark.parametrize("options", [
    {}, {"kv_cache": KVCache(page_size=4, pages=64)},
    {"kv_cache": KVCache(page_size=4, pages=64), "chunk": 3},
], ids=["dense", "paged", "paged-chunked"])
def test_a_hybrid_stack_serves_each_rows_own_state_beside_its_pages(options):
    """A paged server keeps the attention layer's keys in its pool and each row's recurrent state
    in the row's slot: a row starts from a fresh state, wherever a former occupant left its own,
    and a prompt prefilled three tokens a step continues the state its earlier pieces left. Every
    row is the lone task call's."""
    bound = hybrid_task()
    alone = [bound(prompt, budget, key=index)
             for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
    _, tickets = served_alongside(bound, **options)
    for ticket, lone in zip(tickets, [*alone, alone[2]], strict=True):
        assert_same_generation(ticket.result(), lone)


def test_a_shared_prefix_is_refused_where_rows_hold_state_beside_their_pages():
    """A shared prefix's pages hold its keys and nothing of the recurrent state its prompt left,
    so a row reusing them would start that state from nothing: the server refuses the prefix
    cache for such a stack rather than serve it."""
    with pytest.raises(ValueError):
        Server.from_task(hybrid_task(), slots=4, capacity=128, kv_cache=KVCache(page_size=4, pages=64),
                         prefix_cache=True)


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


@pytest.mark.parametrize("prefix_cache", [False, True], ids=["unshared", "prefix"])
@pytest.mark.parametrize("fields", [{"position_embedding": "alibi"},
                                    {"layer_types": ("chunked_attention",),
                                     "kinds": {"chunked_attention": {"chunk": 4}}}], ids=["alibi", "chunked"])
def test_a_paged_server_serves_layers_that_read_their_keys_places(fields, prefix_cache):
    """ALiBi and a chunk read their keys' places, which the cache keeps
    beside its keys, in a paged pool's pages too, so a shared prefix's pages
    carry theirs. Over a pool of pages every row is the lone task call's."""
    model = decoder().clone(max_seq_len=128, **fields)
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    bound = TextGeneration(model, params, RunProcessor(Digits()), sampling=Sampling(temperature=0))
    alone = [bound(prompt, budget, key=index)
             for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
    server, tickets = served_alongside(bound, kv_cache=KVCache(page_size=4, pages=40),
                                       prefix_cache=prefix_cache)
    for ticket, lone in zip(tickets, [*alone, alone[2]], strict=True):
        assert_same_generation(ticket.result(), lone)
    assert server.prefix_hits == (4 if prefix_cache else 0)


@pytest.mark.parametrize("options", [{}, {"kv_cache": KVCache(page_size=16, pages=12)}],
                         ids=["dense", "paged"])
def test_a_server_runs_a_model_with_prediction_depths_without_them(options):
    """The server only samples, so it allocates and seeds no prediction
    depth: a model that declares them admits in the one mixed forward, over
    a dense or a paged cache, its slots hold no prediction cache, and every
    row is the lone task call's."""
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
    first one published in its group: over weights drawn from each declared
    seed, its tokens, lengths and terminations are the lone call's, its
    behaviour log-likelihoods (the greedy policy's, exactly zero) and the
    likelihood arrays' fill past each row's length are the lone call's bit for
    bit, and its log-likelihoods are as exact as the lone call's (check 1),
    truth the float64 model's on the lone call's tokens."""
    layout = Layout(min_shard=1, tolerance=1.0)
    rows = [*zip(PROMPTS, BUDGETS, strict=True), (PROMPTS[2], BUDGETS[2])]

    @functools.cache
    def drawn(seed):
        """The lone calls' and the mesh server's rows over the weights `seed` draws."""
        bound = task()
        bound = bound.bind(bound.model.init(jax.random.key(seed), jnp.ones((1, 2), jnp.int32)))
        alone = [bound(prompt, budget, key=index).host()
                 for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
        placed = bound.bind(place(bound.variables, mesh, layout))
        server, tickets = served_alongside(placed, slots=8, admission=8, **options)
        assert_sharded(placed.variables["params"], server.mesh)
        assert server.groups == jax.device_count() // mesh.tensor
        assert server.prefix_hits == (4 if options.get("prefix_cache") else 0)
        if "kv_cache" in options:
            pool = next(leaf for leaf in jax.tree.leaves(server.cache) if leaf.ndim == 4)
            pages = pool.sharding.spec[1]
            assert ((pages,) if isinstance(pages, str) else tuple(pages)) == tuple(
                axis for axis in BATCH_AXES if server.mesh.shape[axis] > 1)
        return bound, [*alone, alone[2]], [ticket.result().host() for ticket in tickets]

    def likelihoods(generations):
        return {"raw": np.concatenate([generation.raw_log_probs[0, :generation.lengths[0]]
                                       for generation in generations])}

    @as_given
    def served(seed):
        _, alone, rows_served = drawn(int(seed))
        for row, lone in zip(rows_served, alone, strict=True):
            for field in ("tokens", "lengths", "terminated", "behavior_log_probs"):
                np.testing.assert_array_equal(getattr(row, field), getattr(lone, field))
            np.testing.assert_array_equal(row.raw_log_probs[0, row.lengths[0]:],
                                          lone.raw_log_probs[0, lone.lengths[0]:])
        return likelihoods(rows_served)

    def truth(seed):
        """The float64 model's raw log-likelihood of each lone row's tokens, teacher-forced."""
        bound, alone, _ = drawn(int(seed))
        with jax.enable_x64(new_val=True):
            model = bound.model.clone(dtype=jnp.float64)
            params = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), bound.variables)
            picked = []
            for (prompt, _), lone in zip(rows, alone, strict=True):
                ids = Digits().encode(prompt)
                drawn_ids = lone.tokens[0, lone.prompt_width:lone.prompt_width + lone.lengths[0]]
                sequence = jnp.asarray([*ids, *drawn_ids.tolist()], jnp.int32)[None]
                log_probs = jax.nn.log_softmax(model.apply(params, sequence)[0], axis=-1)
                picked.append(np.asarray(log_probs[np.arange(len(ids) - 1, len(ids) - 1 + len(drawn_ids)),
                                                   drawn_ids]))
        return {"raw": np.concatenate(picked)}

    assert_as_exact(served, as_given(lambda seed: likelihoods(drawn(int(seed))[1])),
                    lambda seed: (np.int32(seed),), f"a server on {mesh}", seeds=MESH_SEEDS, truth_fn=truth)


@pytest.mark.mesh(devices=4)
@pytest.mark.parametrize("decode_steps", [1, 4])
@pytest.mark.parametrize("dispatch", ["global"])
def test_a_server_on_an_expert_mesh_draws_what_one_device_draws(dispatch, decode_steps):
    """A mixture layer on an expert mesh can run its dispatch in a shard_map,
    and jax's checkify cannot carry a device check into one (jax-ml/jax#40907).
    The server's step runs the model before its draws check anything, each of
    a call's steps checks its own draws, and a pool smaller than every row's
    capacity checks nothing on the device, so either dispatch serves what the
    global one draws alone on one device, one step a call or several.
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
    server, tickets = served_alongside(served, slots=8, admission=8, kv_cache=KVCache(page_size=16, pages=32),
                                       decode_steps=decode_steps)
    assert_sharded(served.variables["params"], server.mesh)
    assert server.groups == jax.device_count()
    for ticket, row in zip(tickets, [*alone, alone[2]], strict=True):
        assert_same_generation(ticket.result(), row)



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

    @nn.nowrap
    def split_prefill_refusal(self):
        return self.decoder.split_prefill_refusal()

    @property
    def cache_rebuild_position(self):
        return self.decoder.cache_rebuild_position

    @nn.nowrap
    def inference_projection_groups(self, variables):
        return held_groups(self.decoder, "decoder", variables)


def wrapped(kind, model, variables, sampling, processor=None):
    """A task over `kind` around `model`, its `variables` under `decoder`."""
    held = {collection: {"decoder": tree} for collection, tree in variables.items()}
    return TextGeneration(kind(model), held, processor, sampling=sampling)


def test_a_user_decoder_module_is_packed_where_it_names_its_groups():
    """Packing reaches the projections a model names (`Serving`)
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


def test_a_projection_quantized_apart_from_its_group_is_served_unpacked():
    """A model quantized by its modules' paths (`Quantization.patterns`) selecting its query
    projection alone keeps its query, key and value apart, each quantized as the pattern selects it,
    and packs its gate and up projections, which no pattern selects: a server draws what the task
    draws on its own. Packed, the query was served as `qkv_proj`, which the pattern does not
    select, so unquantized."""
    from dew.inference.projections import projection_groups
    from dew.quantization import Quantization

    bound = task(Sampling(temperature=0, eos_token_ids=None))
    quantized = Quantization(patterns=(".*/q_proj",)).apply(bound.model)
    groups = {group.packed for group in projection_groups(quantized, bound.variables)}
    assert "qkv_proj" not in groups and "gate_up_proj" in groups
    alone = TextGeneration(quantized, bound.variables, bound.processor, sampling=bound.sampling)
    server = Server.from_task(alone, slots=2, capacity=128)
    for actual, prompt in zip(server(PROMPTS[:2], 5, key=3), PROMPTS[:2], strict=True):
        assert_same_generation(actual, alone(prompt, 5, key=3))


@pytest.mark.parametrize("finished", [True, False], ids=["prefilled", "prefilling"])
def test_reloaded_weights_share_no_prefix_page_the_old_weights_wrote(finished):
    """A cached prompt page holds keys the old weights computed; after a
    reload the same prompt prefills again and draws what the new weights
    draw alone. That holds too for a prompt still prefilling in pieces of
    three when the reload lands: its first piece kept the old weights'
    keys, so its pages are never shared."""
    bound = task()
    server = Server.from_task(bound, slots=2, capacity=128, kv_cache=KVCache(page_size=4, pages=40),
                              chunk=3, prefix_cache=True)
    server.submit("1234567", 8, key=0)
    if finished:
        server.run()
    else:
        server.step()
    other = bound.variables.copy({"params": jax.tree.map(lambda leaf: leaf * 1.5, bound.variables["params"])})
    server.reload(other)
    server.run()
    hits = server.prefix_hits
    ticket = server.submit("1234567", 8, key=0)
    server.run()
    assert server.prefix_hits == hits
    assert_same_generation(ticket.result(), bound.bind(other)("1234567", 8, key=0))


@pytest.mark.parametrize("options", [{}, {"kv_cache": KVCache(page_size=4, pages=3), "prefix_cache": True}],
                         ids=["dense", "paged"])
def test_a_request_cancelled_while_it_waits_leaves_its_peers_running(options):
    """One slot, three requests: the second is cancelled while the first
    runs. The server goes on, the first and third draw what they draw
    alone, and the cancelled one never takes the slot. A seated request
    can no longer be cancelled. The paged pool holds three pages, which the
    third request needs whole, so it seats only once the pages the
    cancelled request was given back are free again."""
    bound = task()
    server = Server.from_task(bound, slots=1, capacity=128, **options)
    tickets = [server.submit(prompt, 5, key=index) for index, prompt in enumerate(PROMPTS[:3])]
    server.step()
    assert not tickets[0].cancel()
    assert tickets[1].cancel()
    for _ in range(64):
        if not (server.queued or server.occupancy):
            break
        server.step()
    assert server.occupancy == 0 and not server.queued, "a request is still waiting for its pages"
    server.run()
    assert tickets[1].cancelled()
    for index in (0, 2):
        assert_same_generation(tickets[index].result(), bound(PROMPTS[index], 5, key=index))


def test_a_request_past_the_servers_own_capacity_is_refused_at_submit():
    """A user's decoder declares a context of 128 and is served at a
    capacity of 64: a request that fits 64 draws what the task draws alone,
    and a budget or a prompt past it is refused at submission, leaving the
    server serving."""
    bound = task(Sampling(temperature=0, eos_token_ids=None))
    plain = wrapped(UserDecoder, bound.model, bound.variables, bound.sampling, bound.processor)
    server = Server.from_task(plain, slots=2, capacity=64)
    assert plain.model.max_seq_len == 128
    with pytest.raises(ValueError):
        server.submit("12", 63, key=0)
    with pytest.raises(ValueError):
        server.submit("1" * 64, 1, key=0)
    ticket = server.submit("12", 62, key=0)
    server.run()
    assert_same_generation(ticket.result(), plain("12", 62, key=0))


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


def test_a_served_request_without_a_budget_takes_what_the_tasks_max_length_leaves():
    """A task that declares `max_length` and no `max_new_tokens` sizes a request by what the length
    leaves past its prompt's width; a server built from it sizes the same request alike, alone or
    in a batch whose rows are as wide as the widest."""
    bound = dataclasses.replace(task(Sampling(temperature=0)), max_length=8)
    server = Server.from_task(bound, slots=2, capacity=64)
    ticket = server.submit("123", key=0)
    server.run()
    alone = bound("123", key=0)
    assert int(np.asarray(alone.lengths)[0]) == 5
    assert_same_generation(ticket.result(), alone)
    batch = bound(["12", "1234"], key=0).host()
    served = server(["12", "1234"], key=0)
    assert np.asarray(batch.lengths).tolist() == [int(np.asarray(row.lengths)[0]) for row in served] == [4, 4]
    assert [row.text[0] for row in served] == list(batch.text)


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
