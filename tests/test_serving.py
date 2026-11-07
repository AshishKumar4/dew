"""A served request draws what the same request draws alone.

The server keeps a resident cache and moves rows in and out of it, so the
checks are that no row sees another row's state: a batch of mixed lengths
and budgets, a row admitted while others run, a slot reused after its row
left, a queue longer than the slots, and a request too large for the cache.
"""

import contextlib
import dataclasses
import functools
import itertools
import threading

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from arm_arithmetic import float64_held
from flax import linen as nn
from jax.experimental import checkify
from model_support import Digits, decoder, serving_task as task
from reference_error import as_given, assert_as_exact
from serving_support import Served, assert_served_as_exact
from sharded import assert_sharded
from steady_state import guarded, steady_state
from tool_policy import ToolPolicy

from dew.inference import RunProcessor, TextGeneration
from dew.inference.banks import HeldBanks
from dew.inference.loading import place
from dew.inference.serving import Server
from dew.interop.decoder_parts import decoder_axes
from dew.mesh import BATCH_AXES
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.inputs import MixerCall
from dew.nn.kv_cache import POOLED, KVCache, leaf_name, row_axis
from dew.nn.mixer_base import MixerContext
from dew.nn.mixers.attention import CausalSelfAttention
from dew.nn.protocols import held_groups
from dew.nn.sharding import boxed
from dew.sampling import Sampling, Speculative
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


def test_a_server_draws_at_its_tasks_progress_and_at_each_version_it_loads():
    """A server built from a task that draws at a run's progress (a DEX decoder half way through its
    anneal, `TextGeneration.progress`) draws as the task does, and reloaded at another version's
    (`Server.reload`, as `NativeRolloutServer.load` reloads it) as the task at that progress does; one
    built at the run's end serves that end, refusing a run's progress, and a rollout server, which loads
    each run version at its progress, refuses it (ledger 2695)."""
    from dew.inference.rollouts import NativeRolloutServer
    from dew.nn.dex import DifferentialExtension
    from dew.nn.progress import END, Progress

    model = decoder().clone(max_seq_len=128).with_dex(DifferentialExtension(((0, (0, 1)),), anneal_steps=10))
    params = nn.unbox(model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32)))

    def at(updates: int | None) -> TextGeneration:
        progress = END if updates is None else Progress(jnp.asarray(updates, jnp.int32))
        return TextGeneration(model, params, RunProcessor(Digits()), sampling=task().sampling,
                              progress=progress)

    server = Server.from_task(at(5), slots=2, capacity=128)
    for actual, prompt in zip(server(PROMPTS[:2], 5, key=3), PROMPTS[:2], strict=True):
        assert_same_generation(actual, at(5)(prompt, 5, key=3))
    server.reload(params, Progress(jnp.asarray(1, jnp.int32)))
    for actual, prompt in zip(server(PROMPTS[:2], 5, key=3), PROMPTS[:2], strict=True):
        assert_same_generation(actual, at(1)(prompt, 5, key=3))
    assert not np.array_equal(at(5)(PROMPTS[0], 5, key=3).host().raw_log_probs,
                              at(1)(PROMPTS[0], 5, key=3).host().raw_log_probs)
    with pytest.raises(ValueError):
        Server.from_task(at(None), slots=2, capacity=128).reload(params, Progress(jnp.asarray(1, jnp.int32)))
    with pytest.raises(ValueError):
        NativeRolloutServer(Server.from_task(at(None), slots=2, capacity=128))


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
        variables = nn.unbox(model.init(jax.random.key(0), x))
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
        init = jax.jit(lambda *args: nn.unbox(model.init(*args)))
        params = init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
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
                                 nn.unbox(model.init(jax.random.key(13), tokens)))
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
    from dew.telemetry.compiles import Compiles

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


def test_a_servers_step_runs_one_executable_for_each_admission_size():
    """A server's step (`dew.inference.serving_kernel.dispatched`) asks its program for the executable a
    call takes once for each size of admission, and runs it with the dynamic arguments alone after: no
    later step builds the program's dispatch key over the weights and cache (train 329's review: about 1
    ms a step)."""
    from dew.inference.serving_kernel import dispatched

    asked: list[int | None] = []

    class Program:
        def executable(self, model, params, pad_id, placement, steps, resident, carried, admission, *rest):
            asked.append(None if admission is None else len(admission["slots"]))
            return lambda *dynamic: dynamic[3]

    step = dispatched(Program())
    sizes = [None, {"slots": np.zeros(2)}, None, {"slots": np.zeros(2)}, {"slots": np.zeros(4)}, None]
    ran = [step("model", {}, 0, "placement", 1, "resident", "carried", admission, (), (), None, "progress")
           for admission in sizes]
    assert asked == [None, 2, 4] and ran == sizes


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
    params = nn.unbox(model.init(jax.random.key(0), jnp.ones((1, 1), jnp.int32)))
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
    params = nn.unbox(model.init(jax.random.key(1), *args, **fields))
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


DELTA_NET = {"linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 6,
             "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4}
"""`hybrid_model`'s gated delta net's fields."""

THIN_DELTA_NET = {"linear_num_key_heads": 1, "linear_num_value_heads": 1, "linear_key_head_dim": 4,
                  "linear_value_head_dim": 4, "linear_conv_kernel_dim": 2}
"""A delta net of one 4-by-4 value head and a two-tap convolution: a row state of 224 bytes, under half
a page of four tokens' keys (512 bytes), so a host tier holds more states than pages."""


def hybrid_model(vocab_size: int = VOCAB, width: int = 16, delta_net: dict = DELTA_NET) -> CausalTransformer:
    """Qwen3.5's stack at toy size: two gated-delta-net layers (`delta_net`'s fields), whose recurrent
    memory and conv tail are each row's own state, and one full-attention layer, whose keys a paged server
    pages."""
    return CausalTransformer(
        vocab_size=vocab_size, emb_features=width, num_layers=3, num_heads=2, num_kv_heads=2, head_dim=8,
        mlp_features=2 * width, max_seq_len=128, dtype="float32",
        layer_types=("linear_attention",) * 2 + ("full_attention",),
        kinds={"linear_attention": {"mixer": {"class": "gated_delta_net", "fields": delta_net}}})


def parallel_hybrid_model(vocab_size: int = VOCAB, width: int = 16) -> CausalTransformer:
    """Falcon-H1's stack at toy size: every layer Mamba-2 beside attention on one norm
    (`dew.nn.mixers.parallel`), each branch scaled in and out, the Mamba-2 branch's state and conv tail
    each row's own and the attention branch's keys a paged server's pages."""
    return CausalTransformer(
        vocab_size=vocab_size, emb_features=width, num_layers=2, num_heads=2, num_kv_heads=2, head_dim=8,
        mlp_features=2 * width, max_seq_len=128, dtype="float32",
        mixer={"class": "parallel", "fields": {"branches": [
            {"name": "mamba", "input_multiplier": 1.3, "output_multiplier": 0.7,
             "mixer": {"class": "mamba2", "fields": {"num_heads": 4, "head_dim": 8, "state_size": 8,
                                                     "n_groups": 1, "chunk_size": 4}}},
            {"name": "attention", "input_multiplier": 0.8, "output_multiplier": 0.6,
             "mixer": {"class": "attention", "fields": {}}}]}})


def hybrid_task(stack=hybrid_model):
    """A hybrid `stack`'s greedy task (`hybrid_model`'s by default), over drawn weights."""
    model = stack()
    params = nn.unbox(model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32)))
    return TextGeneration(model, params, RunProcessor(Digits()),
                          sampling=Sampling(temperature=0, eos_token_ids=EOS))


@pytest.mark.parametrize("stack", [hybrid_model, parallel_hybrid_model], ids=["layered", "parallel"])
@pytest.mark.parametrize("options", [
    {}, {"kv_cache": KVCache(page_size=4, pages=64)},
    {"kv_cache": KVCache(page_size=4, pages=64), "chunk": 3},
], ids=["dense", "paged", "paged-chunked"])
def test_a_hybrid_stack_serves_each_rows_own_state_beside_its_pages(options, stack):
    """A paged server keeps the attention layer's keys in its pool and each row's recurrent state
    in the row's slot: a row starts from a fresh state, wherever a former occupant left its own,
    and a prompt prefilled three tokens a step continues the state its earlier pieces left. Every
    row is the lone task call's. A parallel hybrid's layers (Falcon-H1's, `parallel_hybrid_model`)
    hold both in each layer, beside each other."""
    bound = hybrid_task(stack)
    alone = [bound(prompt, budget, key=index)
             for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
    _, tickets = served_alongside(bound, **options)
    for ticket, lone in zip(tickets, [*alone, alone[2]], strict=True):
        assert_same_generation(ticket.result(), lone)


HYBRID_SHARED = {"slots": 2, "capacity": 64, "kv_cache": KVCache(page_size=4, pages=40), "prefix_cache": True}
"""A paged server sharing full pages of 4 prompt tokens, its rows' recurrent state beside them."""


@dataclasses.dataclass(frozen=True)
class HybridSite:
    """A shared prefix site's model and draws: its declared seeds, sized paired by `seeds_for` on armada,
    its vocabulary and width, and the tokens each request draws. A site sized each run drew twice its
    count again as pilots, two thirds of its draws (evicted's 141 s, armada)."""

    seeds: int
    vocab: int = 256
    width: int = 16
    budget: int = 8


WIDE_HYBRID = HybridSite(vocab=512, width=64, budget=16, seeds=256)
"""The site of the requests whose repeats share the most: at 256 words, 16 wide and 8 draws, fp32 sums as
users' are (ledger 1698) leave their scores' paired ratio unresolved within 512 seeds, which 512 words, 64
wide and 16 draws a request resolve at 256 (armada, rows/1698-serving-hybrid)."""
NARROW_HYBRID, NARROW_HYBRID_TAILED = HybridSite(seeds=256), HybridSite(seeds=512)
SHORT_HYBRID = HybridSite(seeds=256, budget=4)
"""NARROW_HYBRID's model drawing 4 tokens a request, so a request of 7 to 9 prompt tokens holds 3 or 4
pages and a pool of 6 or 8 runs the segment's host cases."""
"""The sites at 256 words, 16 wide and 8 draws, sized paired by `seeds_for` on armada (11-03, at
0fccb01d79): every-page's and restores-past-admission's resolve at 256, evicted's and
one-entry-booked-twice's at 512."""


SEGMENT_HOST = {"slots": 2, "admission": 2, "kv_cache": KVCache(page_size=4, pages=6), "host_pages": 2,
                "prefix_states": 4}
"""A pool whose host tier keeps a segment's pages: two rows at once over 6 pages of 4 tokens, 2 host slots
and 4 states."""

SEGMENT = (Served(12, shared=6), Served(12, flipped=(7,), shared=6), Served(12, flipped=(6,), shared=6))
"""A segment of 6 tokens three prompts declare shared (a decision's state, each question past it its own
request), ending inside their second page: past it they differ before any kept page boundary, so each
later one reads the first's first page and its segment's 2 tokens of the second, copied into its own,
starting from the state kept at the segment's end (`dew.inference.pages.segment_hash`); asked at once,
the later ones wait for the first to write it."""


@pytest.mark.parametrize("requests, options, hits, rounds, site", [
    ((Served(12), Served(12), Served(12, flipped=(9,)), Served(12, flipped=(1,)), Served(12),
      Served(12, flipped=(6,))), {}, (0, 8, 16, 16, 24, 24), None, WIDE_HYBRID),
    ((Served(12), Served(12), Served(12, flipped=(9,))), {"chunk": 3}, (0, 8, 16), None, WIDE_HYBRID),
    ((Served(12), Served(12, flipped=(6,))), {"prefix_state_every": 1}, (0, 4), None, NARROW_HYBRID),
    ((Served(12), Served(12), Served(12)), {"admission": 1}, (0, 16), (1, 2), NARROW_HYBRID),
    ((Served(12), Served(12, flipped=(0,)), Served(12)),
     {"slots": 1, "kv_cache": KVCache(page_size=4, pages=6), "host_pages": 8, "prefix_states": 4},
     (0, 0, 8), None, NARROW_HYBRID),
    ((Served(9), Served(12, flipped=(0,)), Served(12, flipped=(0, 1)), Served(9)),
     {"slots": 1, "admission": 1, "kv_cache": KVCache(page_size=4, pages=8), "host_pages": 1,
      "prefix_states": 4}, (0, 0, 0, 0), None, NARROW_HYBRID),
    ((Served(9), Served(9, flipped=(0,)), Served(9)),
     {"slots": 1, "admission": 1, "kv_cache": KVCache(page_size=4, pages=8), "host_pages": 6,
      "prefix_states": 1}, (0, 0, 8), None, NARROW_HYBRID),
    (SEGMENT, {}, (0, 6, 12), None, NARROW_HYBRID),
    (SEGMENT, {}, (12,), (3,), NARROW_HYBRID),
    ((Served(7, shared=6), Served(7, flipped=(0,)), Served(7, flipped=(0, 1)),
      Served(7, flipped=(6,), shared=6), Served(7, shared=6)), SEGMENT_HOST, (0, 0, 12), (1, 2, 2),
     SHORT_HYBRID),
    ((Served(9, flipped=(0,)), Served(7, shared=6), Served(9, flipped=(0, 1)),
      Served(7, flipped=(6,), shared=6), Served(9, flipped=(0,))),
     {**SEGMENT_HOST, "kv_cache": KVCache(page_size=4, pages=8)}, (0, 0, 6, 14), (1, 1, 2, 1), SHORT_HYBRID)],
    ids=["by-default", "in-pieces", "every-page", "restores-past-admission", "from-the-host",
         "from-one-host-slot", "a-state-from-the-host", "a-shared-segment", "a-shared-segment-asked-at-once",
         "a-segment-uploaded-and-copied-at-once", "a-segment-copied-into-a-spilled-page"])
@pytest.mark.check1_shards(seeds=256, per_task=64)
def test_a_hybrid_stack_reads_a_shared_prefix_with_the_state_its_prompt_left(requests, options, hits, rounds,
                                                                             site):
    """`assert_reads_a_shared_prefix`'s sites of 256 seeds, which CI draws in map tasks of 64: the default
    store's took 92 to 108 s in one task with its bundle on armada, its first task of 128 cold 115 s
    (ledger 3759)."""
    assert_reads_a_shared_prefix(requests, options, hits, rounds, site)


@pytest.mark.parametrize("requests, options, hits, rounds, site", [
    ((Served(12), Served(12, flipped=(1,)), Served(12)), {"prefix_states": 1}, (0, 0, 0), None,
     NARROW_HYBRID_TAILED),
    ((Served(12), Served(12, flipped=(1,)), Served(12), Served(12, flipped=(1,))), {"prefix_states": 1},
     (0, 8), (2, 2), NARROW_HYBRID_TAILED)],
    ids=["evicted", "one-entry-booked-twice"])
@pytest.mark.check1_shards(seeds=512, per_task=128)
def test_a_hybrid_stack_reads_a_shared_prefix_from_a_store_of_one_entry(requests, options, hits, rounds,
                                                                        site):
    """`assert_reads_a_shared_prefix`'s sites of 512 seeds, a store of one entry's, which CI draws in map
    tasks of 128."""
    assert_reads_a_shared_prefix(requests, options, hits, rounds, site)


def assert_reads_a_shared_prefix(requests, options, hits, rounds, site):
    """A shared prefix's pages hold its keys, and a hybrid stack's rows keep their recurrent memory and
    conv tail beside them: the server keeps the state a prompt leaves at its deepest shareable page
    boundary (and every `prefix_state_every` pages below it), a piece cut to end there, and a later prompt
    reuses the pages as deep as a state is kept, starting its row from that state. By default (a store of
    two entries): a repeat reads both full pages, as does a prompt agreeing with it past its deepest
    boundary; one agreeing nowhere reads none, its state taking the other entry, so the first prompt's
    is still kept and its repeat reads its pages again; one agreeing only through its first page reads
    none, as no state is kept there. Where a state is kept every page, that one reads its first page; a
    store of one entry evicts the first prompt's state for the next one's and its reuse with it. Two
    prompts admitted together into a store of one entry keep one state, the later booking's, which the
    next round's repeat of it reads; two repeats seated together each restore their state, past the
    one prompt piece a call admits. A pool of six pages with a host tier spills the first prompt's pages
    while another prompt runs, its state kept in a store of four entries, and its repeat reads its pages
    back (a pool without the tier reads none: its other prompt takes the pages); in a pool of eight pages
    two other prompts reclaim a prompt's two shared pages, and a tier of one host slot keeps one of them
    alone, where the store holds the state after the deepest page only, so the repeat reads nothing back;
    a store of one entry sends a prompt's state to the host tier when another prompt's takes the entry,
    and its repeat reads it back with its pages (`dew.inference.pages.States`; a tier holds as many states
    as its host pages' bytes, one in six pages' here). A segment three prompts
    declare shared, ending inside a page (`SEGMENT`), is read by each later one from the state kept at
    its end, its last page copied into the prompt's own, one a round or all asked at once. With a host
    tier (`SEGMENT_HOST`): a segment two other prompts spill to the host, its page named by the segment
    alone, comes back for two prompts seated together, the first uploading it and the second copying it
    from there after the upload lands; and a prompt copying a segment into a page the pool reclaims from
    a cached prefix spills that prefix first, which its repeat reads back. Each request draws what it
    draws alone and scores it as a full forward does (check 1)."""
    # A vocabulary of 256 at least and eight draws a request, so each request's scores spread enough to
    # measure: at 64, fp32 sums as users' are (ledger 2468) leave the ratio unresolved within 512 seeds.
    model = hybrid_model(vocab_size=site.vocab, width=site.width)
    task = TextGeneration(model, nn.unbox(model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))),
                          sampling=Sampling(temperature=0, pad_token_id=0))
    assert_served_as_exact(task, requests, f"hybrid shared prefix {hits}",
                           serving={**HYBRID_SHARED, **options}, hits=hits, rounds=rounds, budget=site.budget,
                           seeds=(site.seeds, site.seeds))


@pytest.mark.parametrize("host_pages", [1, 5, 6, 40])
def test_a_hybrid_servers_host_states_fit_the_bytes_of_its_host_pages(host_pages):
    """A hybrid server keeps as many row states on the host as the bytes of its `host_pages` pages hold
    (ledger 4602): a state is a whole recurrent memory and conv tail, here several pages' bytes, so
    `host_pages` states would take several times the host memory it names. Each group's state slots,
    times one entry's bytes in the server's own store, are within its host pages' bytes in the server's
    own pool, and one more would not be."""
    server = Server.from_task(hybrid_task(), **{**HYBRID_SHARED, "host_pages": host_pages,
                                                 "prefix_states": 2})
    part = server.rows.pages[0]
    assert part.states is not None
    leaves = jax.tree_util.tree_leaves_with_path(server._states)
    entry = sum(leaf.nbytes // leaf.shape[row_axis(path)] for path, leaf in leaves)
    # The pool's leaves hold their pages on their second axis, heads first (`_pages_axis`).
    pool = [(path, leaf) for path, leaf in jax.tree_util.tree_leaves_with_path(server.cache)
            if leaf_name(path) in POOLED]
    page = sum(leaf.nbytes // leaf.shape[row_axis(path) + 1] for path, leaf in pool)
    budget = part.host * page
    assert part.states.host.slots * entry <= budget < (part.states.host.slots + 1) * entry


@pytest.mark.check1_shards(seeds=256, per_task=64)
def test_a_host_tier_of_more_states_than_pages_reads_each_state_back():
    """A hybrid stack whose row state is smaller than a page keeps more states on the host than pages
    (`THIN_DELTA_NET`: four states in two pages' bytes), each in its own slot: three prompts send
    their states there as a store of one entry gives way, the third to the slot past the pages' count,
    and their repeats read each back with its own prefix (ledger 4602's review), each request drawing
    what it draws alone and scoring it as a full forward does (check 1)."""
    model = hybrid_model(256, delta_net=THIN_DELTA_NET)
    # It draws greedily to its budget alone, no end token, as the check scores every draw.
    task = TextGeneration(model, nn.unbox(model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))),
                          sampling=Sampling(temperature=0, pad_token_id=0))
    requests = (Served(9), Served(9, flipped=(0,)), Served(9, flipped=(0, 1)), Served(9, flipped=(0, 2)),
                Served(9, flipped=(0, 1)), Served(9, flipped=(0,)))
    serving = {**HYBRID_SHARED, "slots": 1, "admission": 1, "kv_cache": KVCache(page_size=4, pages=8),
               "host_pages": 2, "prefix_states": 1}
    server = Server.from_task(task, **serving)
    assert server.rows.pages[0].states.host.slots == 4
    assert_served_as_exact(task, requests, "thin hybrid host states", serving=serving,
                           rounds=(1,) * len(requests), budget=8, seeds=(256, 256))


def test_a_one_tap_convolutions_empty_state_fits_any_host_tier():
    """A one-tap short convolution keeps an empty conv tail beside paged attention, a row state of no
    bytes: a server of it builds, its state host tier as many slots as the pages' (ledger 4602's review),
    with no host tier and with one."""
    model = CausalTransformer(
        vocab_size=VOCAB, emb_features=16, num_layers=2, num_heads=2, num_kv_heads=2, head_dim=8,
        mlp_features=32, max_seq_len=128, dtype="float32", layer_types=("conv", "full_attention"),
        kinds={"conv": {"mixer": {"class": "short_conv", "fields": {"conv_L_cache": 1}}}})
    params = nn.unbox(model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32)))
    task = TextGeneration(model, params, RunProcessor(Digits()),
                          sampling=Sampling(temperature=0, eos_token_ids=EOS))
    for host_pages in (0, 3):
        server = Server.from_task(task, **{**HYBRID_SHARED, "host_pages": host_pages})
        part = server.rows.pages[0]
        assert part.states is not None and part.states.host.slots == part.host == host_pages


def test_a_shared_segment_is_refused_where_no_server_keeps_it_or_it_splits_its_prompt():
    """A declared shared segment (`Server.submit`'s `shared`) is kept by a paged server with a prefix cache
    alone and holds some of the prompt's tokens but not its last: a dense server's, an empty one and a
    whole prompt are refused."""
    bound = hybrid_task()
    prompt = np.arange(1, 9, dtype=np.int32)
    with pytest.raises(ValueError):
        Server.from_task(bound, slots=2, capacity=64).submit(prompt, 2, key=0, shared=4)
    server = Server.from_task(bound, **HYBRID_SHARED)
    for shared in (0, 8):
        with pytest.raises(ValueError):
            server.submit(prompt, 2, key=0, shared=shared)


def served_alongside(bound, slots=4, admission=2, **options):
    """Every prompt submitted at once, plus the longest again once it is done,
    through a server with `options`; returns the server and the tickets."""
    return served_by(Server.from_task(bound, slots=slots, capacity=128, admission=admission, **options))


def served_by(server):
    """`served_alongside`'s requests through `server`; returns the server and the tickets."""
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


def test_a_host_tier_reads_back_a_prefix_whose_upload_reclaims_the_last_device_pages():
    """A pool of two pages and one host slot: a prompt's one shared page goes to the slot when another
    prompt's two named pages take the pool, and the first prompt reads it back while its fresh pages reclaim
    the other prompt's, which the full slot cannot keep: its upload reads the slot as the spill left it. A
    reserve released before its copies are made leaves its upload's page cached, and a prompt that takes
    that page back finds the slot still holding the first prompt's page, not the device page the upload
    never wrote (`dew.inference.pages.Pages`, ledger 874)."""
    from dew.inference.pages import Pages

    pages = Pages(2, 4, prefix_cache=True, host=1)
    first, other = np.arange(1, 6), np.arange(20, 28)

    def ran(prompt):
        held, prefilled, _, _ = pages.reserve(prompt, 8)
        pages.publish(prompt, held)
        pages.release(held)
        return held, prefilled

    def spilled():
        held, _ = ran(first)
        ran(other)
        assert pages.drained().spills == [(held[0], 0)]

    spilled()
    held, prefilled, _, _ = pages.reserve(first, 8)
    owed = pages.drained()
    assert (prefilled, owed.spills, owed.uploads) == (4, [], [(0, held[0])])
    pages.release(held)
    spilled()
    held, prefilled, _, _ = pages.reserve(first, 8)
    assert prefilled == 4
    pages.release(held)
    pages.reserve(other, 8)
    owed = pages.drained()
    assert (owed.spills, owed.uploads, pages.shared(first)) == ([], [], [-1])


@pytest.mark.distributed
def test_a_pooled_servers_host_tier_reads_each_processs_prefix_back_from_its_own_slots(tmp_path):
    """One server over a mesh that spans two processes, a group of rows a process, with a host tier: each
    process's prompt is cached, its group's pages taken by two other prompts, which spill the prompt's pages
    to the process's host slots, and served again, reading them back into fresh pages; every request draws
    what one process serving them all draws, and each process reads its prompt's 8 tokens from its slots
    (ledger 874)."""
    from pathlib import Path

    from process_support import run_pool, run_worker

    worker = Path(__file__).with_name("serving_worker.py")
    reports = run_pool("pooled_host_tier", tmp_path / "pool", 2, script=worker, devices=1, fsdp_size=2,
                       timeout=300)
    single = run_worker("pooled_host_tier", tmp_path / "single.json", script=worker, fsdp_size=1, devices=1)
    assert {**reports[0]["drawn"], **reports[1]["drawn"]} == single["drawn"]
    assert [report["shared"] for report in reports] == [8, 8]


@pytest.mark.parametrize("host_pages, hits", [(0, 0), (1, 4), (8, 8)], ids=["device", "one-slot", "host"])
def test_a_host_tier_keeps_prefix_pages_the_pool_reclaims(host_pages, hits):
    """A prefix cache of eight pages of four tokens, one slot: a prompt's two shared pages are cached as it
    ends, then two other prompts take the pool's pages and it reclaims them. With a host tier the reclaimed
    pages go to host memory and the prompt again reads its 8 tokens back from there, or its first 4 where
    one host slot keeps the page reclaimed last; without one it prefills them again. Every request draws
    what its lone call draws (ledger 874); a hybrid stack's host tier is `assert_reads_a_shared_prefix`'s
    `from-the-host` and `from-one-host-slot` sites, judged by check 1."""
    bound = task(Sampling(temperature=1.0))
    shared, others, budget = "123456781", ["987654329876", "555544443333"], 6
    requests = [shared, *others, shared]
    alone = [bound(held, budget, key=index) for index, held in enumerate(requests)]
    server = Server.from_task(bound, slots=1, capacity=64, admission=1, prefix_cache=True,
                              kv_cache=KVCache(page_size=4, pages=8), host_pages=host_pages)
    drawn = []
    for index, held in enumerate(requests):
        ticket = server.submit(held, budget, key=index)
        server.run()
        drawn.append(ticket.result())
    assert server.prefix_hits == hits
    for generation, lone in zip(drawn, alone, strict=True):
        assert_same_generation(generation, lone)


def test_a_banked_servers_host_tier_reads_back_the_prefix_pages_its_pool_reclaimed():
    """`test_a_host_tier_keeps_prefix_pages_the_pool_reclaims`'s host case over a banked store, whose paged
    pool holds its run's layers first (`dew.nn.kv_cache.row_axis`): the reclaimed pages go to host memory
    a page at a time, every layer of each, and come back into fresh pages where the prompt reads its 8
    tokens, every request drawing what its lone call draws."""
    bound = banked_task(Sampling(temperature=1.0))
    shared, others, budget = "123456781", ["987654329876", "555544443333"], 6
    requests = [shared, *others, shared]
    alone = [bound(held, budget, key=index) for index, held in enumerate(requests)]
    server = Server.from_task(bound, slots=1, capacity=64, admission=1, prefix_cache=True,
                              kv_cache=KVCache(page_size=4, pages=8), host_pages=8)
    drawn = []
    for index, held in enumerate(requests):
        ticket = server.submit(held, budget, key=index)
        server.run()
        drawn.append(ticket.result())
    assert "layers_0_3" in server.cache and server.prefix_hits == 8
    for generation, lone in zip(drawn, alone, strict=True):
        assert_same_generation(generation, lone)


def test_samples_of_one_prompt_prefill_it_once_and_each_draws_what_it_draws_alone():
    """Four samples of one prompt and another prompt behind them, submitted together to a paged server with
    a prefix cache: the first sample writes the prompt's full pages, the other three wait and read them
    once written, as vLLM shares a running prompt's blocks, while the other prompt seats past them at
    once, so 3 x 16 of the 19-token prompt's tokens come from shared pages; each request draws under its
    own key what its lone call draws (ledger 1423)."""
    bound = task(Sampling(temperature=1.0))
    prompt, other, budget = "1234567812345678912", "9876543", 6
    requests = [prompt] * 4 + [other]
    alone = [bound(held, budget, key=index) for index, held in enumerate(requests)]
    server = Server.from_task(bound, slots=4, capacity=128, admission=4,
                              kv_cache=KVCache(page_size=4, pages=64), prefix_cache=True)
    tickets = [server.submit(held, budget, key=index) for index, held in enumerate(requests)]
    server.run()
    assert server.prefix_hits == 3 * 16
    assert tickets[4].admitted <= tickets[1].admitted
    for ticket, lone in zip(tickets, alone, strict=True):
        assert_same_generation(ticket.result(), lone)


@pytest.mark.parametrize("prefix_cache", [False, True], ids=["unshared", "prefix"])
@pytest.mark.parametrize("fields", [{"position_embedding": "alibi"},
                                    {"layer_types": ("chunked_attention",),
                                     "kinds": {"chunked_attention": {"chunk": 4}}}], ids=["alibi", "chunked"])
def test_a_paged_server_serves_layers_that_read_their_keys_places(fields, prefix_cache):
    """ALiBi and a chunk read their keys' places, which the cache keeps
    beside its keys, in a paged pool's pages too, so a shared prefix's pages
    carry theirs. Over a pool of pages every row is the lone task call's."""
    model = decoder().clone(max_seq_len=128, **fields)
    params = nn.unbox(model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32)))
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
    params = nn.unbox(model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32)))
    bound = TextGeneration(model, params, RunProcessor(Digits()), sampling=Sampling(temperature=0))
    alone = [bound(prompt, budget, key=index)
             for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
    server, tickets = served_alongside(bound, **options)
    assert server.mixed_refusal is None
    for ticket, lone in zip(tickets, [*alone, alone[2]], strict=True):
        assert_same_generation(ticket.result(), lone)
    held = jax.tree_util.tree_flatten_with_path(server.cache)[0]
    assert not [path for path, _ in held if "mtp" in jax.tree_util.keystr(path).lower()]


@pytest.mark.parametrize("temperature, decode_steps, keys", [(0.0, 1, None), (1.0, 4, None), (1.0, 4, "rbg")],
                         ids=["greedy", "sampled-four-steps", "sampled-four-steps-rbg"])
def test_a_server_drafts_speculative_blocks_as_a_lone_call_does(temperature, decode_steps, keys):
    """A task whose strategy is speculative blocks (`Speculative`) serves them: each iteration every row
    drafts with the model's prediction depth, verifies and emits a block, rows of different widths and
    budgets beside one another, seated mid-flight, ending inside a call, and every request draws what its
    lone task call draws, greedy or sampled, the sampled ones under counter-based row keys too, as the
    RBG implementation keys them (3617: every row's own stream, whatever slot it sits in, where vmap would
    draw them all from the first), served outside the context that made RBG the default, as an engine's
    thread serves it (3943). Speculation over paged or chunked rows is refused by name, and so is a
    reload with a row running, whose next block would open on the old weights' logits."""
    model = decoder().clone(max_seq_len=128, num_nextn_predict_layers=1)
    params = nn.unbox(model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32)))
    bound = TextGeneration(model, params, RunProcessor(Digits()), sampling=Sampling(temperature=temperature),
                           strategy=Speculative(block=3))
    with contextlib.nullcontext() if keys is None else jax.default_prng_impl(keys):
        alone = [bound(prompt, budget, key=index)
                 for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
        server = Server.from_task(bound, slots=4, capacity=128, admission=2, decode_steps=decode_steps)
    server, tickets = served_by(server)
    for ticket, lone in zip(tickets, [*alone, alone[2]], strict=True):
        assert_same_generation(ticket.result(), lone)
    for options in ({"kv_cache": KVCache(page_size=16, pages=12)}, {"chunk": 3}):
        with pytest.raises(ValueError):
            Server.from_task(bound, slots=2, capacity=128, **options)
    server.submit(PROMPTS[0], 50, key=0)
    server.step()
    with pytest.raises(ValueError):
        server.reload(params)


@pytest.mark.distributed
@pytest.mark.parametrize("paged", [False, True], ids=["dense", "paged"])
def test_a_server_over_a_pool_draws_what_each_request_draws_alone(tmp_path, paged):
    """One server over a mesh that spans two processes, each seating and reading its own groups of rows and
    the pool stepping it together, each step's admission, captures and restores agreed: a round of
    requests on the second process alone, then one on both with the widest on the second, serves every
    request what one process serving them all draws, over a dense cache and a hybrid stack's paged one
    sharing prompt pages and the recurrent states kept at them; and an engine, which steps one process's
    kinds, refuses the pooled server (ledger 1423)."""
    from pathlib import Path

    from process_support import run_pool, run_worker

    worker = Path(__file__).with_name("serving_worker.py")
    reports = run_pool("pooled_server", tmp_path / "pool", 2, script=worker, devices=2, fsdp_size=4,
                       paged=paged or None, timeout=300)
    single = run_worker("pooled_server", tmp_path / "single.json", script=worker, fsdp_size=1, devices=1,
                        paged=paged or None)
    assert {**reports[0]["drawn"], **reports[1]["drawn"]} == single["drawn"]
    assert [report["engine"] for report in (*reports, single)] == ["refuses it", "refuses it", "takes it"]
    # The last process's second round read its first round's prefix pages and the states kept at them.
    assert (reports[1]["shared"] > 0) == paged


def drawn_adapter(model, variables, module: str, key: int, layouts=None):
    """A rank-2 adapter of `module` over `model`, its B drawn from `key` (its init holds B at zero, which
    adapts nothing), as the adapter-set sites draw each of theirs."""
    from dew.lora import LoRA

    adapter = LoRA(rank=2, modules=(module,)).apply(model, variables, key=key, layouts=layouts)
    factors = jax.tree_util.tree_map_with_path(
        lambda path, leaf: (jax.random.normal(jax.random.key(key), leaf.shape, leaf.dtype)
                            if jax.tree_util.keystr(path).endswith("'lora_B']") else leaf),
        adapter.variables["params"])
    return dataclasses.replace(adapter, variables={**adapter.variables, "params": factors})


def test_a_speculating_server_over_an_adapter_set_reseeds_its_depths_under_each_rows_adapter():
    """A speculating server (`Speculative`) over an adapter set whose adapters adapt the stream's input
    map (`project_in`, which reseeding a block's prediction depths reads, `DecodeOps.embed`) draws each
    request at zero temperature as the greedy walk under its own adapter, or the base, each block's
    reseeding under the row's adapter (row 4222's review)."""
    from dew.lora import AdapterSet
    from dew.sampling import Sampling, Speculative

    model = decoder().clone(max_seq_len=32, num_nextn_predict_layers=1, embedding_features=8)
    params = nn.unbox(model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32)))

    adapters = AdapterSet.of({"first": drawn_adapter(model, params, "project_in", 1),
                              "second": drawn_adapter(model, params, "project_in", 2)})
    served = TextGeneration(adapters.model, adapters.variables, RunProcessor(Digits()),
                            sampling=Sampling(temperature=0), strategy=Speculative(block=3))
    server = Server.from_task(served, slots=3, capacity=32)
    prompts, names = ([1, 2, 3], [7, 8, 9], [4, 5, 6]), ("first", None, "second")
    tickets = [server.submit(np.asarray([prompt]), 6, key=0, adapter=name)
               for prompt, name in zip(prompts, names, strict=True)]
    server.run()
    walks = []
    for ticket, prompt, index in zip(tickets, prompts, (0, -1, 1), strict=True):
        walked = np.asarray([prompt], np.int32)
        for _ in range(6):
            scores = adapters.model.apply(adapters.variables, jnp.asarray(walked),
                                          adapters=jnp.asarray([index], jnp.int32))[:, -1]
            walked = np.concatenate([walked, np.asarray(scores).argmax(-1)[:, None].astype(np.int32)], axis=1)
        held = ticket.result().host()
        # The prompt and the response it drew, which an end of sequence may stop short of six.
        drawn_to = 3 + int(np.asarray(held.lengths)[0])
        np.testing.assert_array_equal(np.asarray(held.tokens)[0, :drawn_to], walked[0, :drawn_to])
        walks.append(walked[0, 3:])
    # The adapters move the walk: the rows' adapters are what the draws read.
    assert len({tuple(walk) for walk in walks}) > 1


@pytest.mark.parametrize("cache", [None, KVCache(page_size=2, pages=64)], ids=["dense", "paged"])
def test_a_server_over_an_adapter_set_draws_each_request_under_its_own_adapter(tmp_path, cache):
    """One server over an adapter set (`dew.lora.AdapterSet`) of a published decoder: the fixture's PEFT
    adapter (rank 8 over q and v, with an alpha pattern) and one of rank 2 over q alone, saved side by side
    and loaded by directory. Requests naming either adapter, or none, share the server's steps and its
    one copy of the base, and each draws what a task over its adapter alone (or the base) draws, on par
    with that task against its float64 twin (check 1), over a dense cache and over a paged one sharing
    prompt pages, where one prompt under two adapters shares none (ledger 870). An adapter bound over
    other weights than the set's base is refused, which the set would run over the first's (ledger
    4494)."""
    from pathlib import Path

    from arm_arithmetic import widened
    from reference_error import as_given, assert_as_exact, seeds_for

    from dew.interop.pretrained import Pretrained
    from dew.lora import AdapterSet, LoRA

    fixtures = Path(__file__).resolve().parent / "fixtures"
    decoder = Pretrained.load(fixtures / "hf" / "llama-tiny", dtype="float32", attention_impl="reference")
    fixture = LoRA.load(decoder.model, decoder.variables, fixtures / "lora" / "llama-tiny" / "adapter",
                        layouts=decoder.layouts)
    drawn = LoRA(rank=2, modules=("q_proj",)).apply(decoder.model, decoder.variables, key=0,
                                                       layouts=decoder.layouts)
    # Its B starts at zero; a draw of its own moves the model.
    factors = jax.tree_util.tree_map_with_path(
        lambda path, leaf: (jax.random.normal(jax.random.key(len(path)), leaf.shape, leaf.dtype) * 0.5
                            if jax.tree_util.keystr(path).endswith("'lora_B']") else leaf),
        drawn.variables["params"])
    second = dataclasses.replace(drawn, variables={**drawn.variables, "params": factors})
    fixture.save(fixture.variables, tmp_path / "set" / "fixture")
    second.save(second.variables, tmp_path / "set" / "second")
    adapters = AdapterSet.load(decoder.model, decoder.variables, tmp_path / "set", layouts=decoder.layouts)
    assert adapters.names == ("fixture", "second")
    # An adapter over other weights than the set's base would run over the first adapter's: refused.
    moved = jax.tree.map(lambda leaf: leaf * 0.5, decoder.variables)
    elsewhere = LoRA(rank=2, modules=("q_proj",)).apply(decoder.model, moved, key=0, layouts=decoder.layouts)
    with pytest.raises(ValueError):
        AdapterSet.of({"fixture": fixture, "elsewhere": elsewhere})
    sampling = Sampling(temperature=1.0)
    served = dataclasses.replace(decoder, model=adapters.model, variables=adapters.variables
                                 ).text_generation(sampling=sampling)
    alone = {name: dataclasses.replace(decoder, model=adapter.model, variables=adapter.variables
                                       ).text_generation(sampling=sampling)
             for name, adapter in (("fixture", fixture), ("second", second))} | {
                 None: decoder.text_generation(sampling=sampling)}
    prompts = ([1, 5, 9, 13, 2], [2, 4], [7, 7, 7, 3, 1], [1, 5, 9, 13, 2], [1, 5, 9, 13, 2])
    names = ("fixture", "second", None, "second", "fixture")
    server = Server.from_task(served, slots=4, capacity=64, admission=2, kv_cache=cache,
                              prefix_cache=cache is not None)

    def draw(seed):
        return (np.arange(len(prompts)) + len(prompts) * seed,)

    @as_given
    def through_the_server(keys):
        tickets = [server.submit(np.asarray([prompt]), 6, key=int(key), adapter=name)
                   for prompt, name, key in zip(prompts, names, keys, strict=True)]
        server.run()
        return tuple(drawn_of(ticket.result()) for ticket in tickets)

    def drawn_of(generation):
        """A draw's tokens and the likelihoods its adapter gave them, which a wrong adapter moves."""
        held = generation.host()
        return held.tokens, held.raw_log_probs

    def by_itself(dtype):
        @as_given
        def drawn_alone(keys):
            out = []
            for prompt, name, key in zip(prompts, names, keys, strict=True):
                task = alone[name]
                if dtype == "float64":
                    task = dataclasses.replace(task, model=task.model.clone(dtype=jnp.float64),
                                               variables=widened(task.variables))
                with jax.enable_x64(new_val=dtype == "float64"):
                    out.append(drawn_of(task(np.asarray([prompt]), 6, key=int(key))))
            return tuple(out)

        return drawn_alone

    label = "served adapter set"
    count = seeds_for(by_itself("float32"), draw, label, truth_fn=by_itself("float64"),
                      dew_fn=through_the_server)
    assert_as_exact(through_the_server, by_itself("float32"), draw, label, seeds=count,
                    truth_fn=by_itself("float64"))


def test_an_adapter_set_whose_adapters_adapt_different_layers_runs_each_rows_own():
    """An adapter set of two adapters over different layers of one decoder, one over layer 0's q and one
    over layer 1's: each layer's update is its own adapter's or none, and each row's logits are those of
    its adapter alone, on par with that adapter against its float64 twin (check 1). A scanned stack, which
    runs its like layers under one factor pair, refuses layers whose adapters differ, as it refuses one
    adapter's uneven targets (test_lora's test_the_branch_reaches_every_layer_of_a_scanned_run)."""
    from pathlib import Path

    from arm_arithmetic import widened
    from reference_error import as_given, assert_as_exact, seeds_for

    from dew.interop.pretrained import Pretrained
    from dew.lora import AdapterSet

    decoder = Pretrained.load(Path(__file__).resolve().parent / "fixtures" / "hf" / "llama-tiny",
                              dtype="float32", attention_impl="reference")

    def drawn(layer: int, key: int):
        return drawn_adapter(decoder.model, decoder.variables, f"layers.{layer}.self_attn.q_proj", key,
                             layouts=decoder.layouts)

    first, second = drawn(0, 1), drawn(1, 2)
    adapters = AdapterSet.of({"first": first, "second": second})

    def draw(seed):
        return (jax.random.randint(jax.random.key(seed), (2, 5), 1, 30),)

    @as_given
    def served(tokens):
        return np.asarray(adapters.model.apply(adapters.variables, jnp.asarray(tokens),
                                               adapters=jnp.asarray([0, 1], jnp.int32)))

    def alone(dtype):
        @as_given
        def logits(tokens):
            rows = []
            for row, adapter in enumerate((first, second)):
                model, variables = adapter.model, adapter.variables
                if dtype == "float64":
                    model, variables = model.clone(dtype=jnp.float64), widened(variables)
                with jax.enable_x64(new_val=dtype == "float64"):
                    rows.append(np.asarray(model.apply(variables, jnp.asarray(tokens)))[row])
            return np.stack(rows)

        return logits

    label = "adapter set over different layers"
    count = seeds_for(alone("float32"), draw, label, truth_fn=alone("float64"), dew_fn=served)
    assert_as_exact(served, alone("float32"), draw, label, seeds=count, truth_fn=alone("float64"))


def test_a_row_of_an_adapter_sets_base_computes_the_base_bit_for_bit():
    """A row naming no adapter of a set (-1 in the call's `adapters`) takes no branch: beside a row whose
    adapter's factors overflow its update to inf, its logits are the base model's, bit for bit."""
    from pathlib import Path

    from dew.interop.pretrained import Pretrained
    from dew.lora import AdapterSet, LoRA

    fixtures = Path(__file__).resolve().parent / "fixtures"
    decoder = Pretrained.load(fixtures / "hf" / "llama-tiny", dtype="float32", attention_impl="reference")
    fixture = LoRA.load(decoder.model, decoder.variables, fixtures / "lora" / "llama-tiny" / "adapter",
                        layouts=decoder.layouts)
    huge = dataclasses.replace(fixture, variables={**fixture.variables, "params": jax.tree.map(
        lambda leaf: leaf * 1e30, fixture.variables["params"])})
    adapters = AdapterSet.of({"huge": huge})
    tokens = jnp.asarray([[1, 5, 9, 13], [2, 4, 6, 8]], jnp.int32)
    served = adapters.model.apply(adapters.variables, tokens, adapters=jnp.asarray([0, -1], jnp.int32))
    base = decoder.model.apply(decoder.variables, tokens)
    assert not np.isfinite(np.asarray(served[0])).all()
    np.testing.assert_array_equal(np.asarray(served[1]), np.asarray(base[1]))


def test_an_adapter_sets_decode_carries_each_rows_adapter_through_a_reindex():
    """An adapter set's decode carries each row's adapter in its state (`DecoderState.adapters`, int32):
    four prompts prefilled under the set's two adapters and the base (-1), their rows reordered,
    duplicated and one dropped (`DecodeOps.reindex`, as a fork of a row's draws moves them) and advanced a
    token, each row's logits are those of its prompt prefilled in that place under its own adapter and
    advanced alike, bit for bit (row 4222)."""
    from pathlib import Path

    from dew.interop.pretrained import Pretrained
    from dew.lora import AdapterSet
    from dew.nn.inputs import ModelInputs
    from dew.sampling.text import decode_ops, prefill_state

    decoder = Pretrained.load(Path(__file__).resolve().parent / "fixtures" / "hf" / "llama-tiny",
                              dtype="float32", attention_impl="reference")

    def drawn(layer: int, key: int):
        return drawn_adapter(decoder.model, decoder.variables, f"layers.{layer}.self_attn.q_proj", key,
                             layouts=decoder.layouts)

    adapters = AdapterSet.of({"first": drawn(0, 1), "second": drawn(1, 2)})
    model, variables = adapters.model, adapters.variables
    prompts = jnp.asarray([[1, 5, 9, 13], [2, 4, 6, 8], [3, 7, 11, 15], [6, 2, 9, 4]], jnp.int32)
    rows = np.asarray([0, 1, -1, 1], np.int64)
    order = jnp.asarray([2, 0, 0, 3], jnp.int32)
    following, active = jnp.asarray([17, 19, 21, 23], jnp.int32), jnp.ones((4,), bool)
    ops = decode_ops(model, variables, 0, None)

    def advanced(tokens, held):
        return ops.advance(prefill_state(model, variables, ModelInputs(tokens), ops, adapters=held)[0],
                           following, active)

    state = prefill_state(model, variables, ModelInputs(prompts), ops, adapters=rows)[0]
    assert state.adapters is not None and state.adapters.dtype == jnp.int32
    moved = ops.reindex(state, order)
    assert moved.adapters is not None and moved.adapters.tolist() == [-1, 0, 0, 1]
    stepped = ops.advance(moved, following, active)
    expected = advanced(prompts[order], rows[np.asarray(order)])
    np.testing.assert_array_equal(np.asarray(stepped.logits), np.asarray(expected.logits))
    # The rows' adapters change what they compute: the base row's prompt under the first adapter differs.
    under = advanced(prompts[order], np.asarray([0, 0, 0, 1]))
    assert not np.array_equal(np.asarray(under.logits[0]), np.asarray(stepped.logits[0]))


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
        bound = bound.bind(nn.unbox(bound.model.init(jax.random.key(seed), jnp.ones((1, 2), jnp.int32))))
        alone = [bound(prompt, budget, key=index).host()
                 for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
        with boxed(decoder_axes(bound.model)):
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

    @as_given
    def truth(seed):
        """The float64 model's raw log-likelihood of each lone row's tokens, teacher-forced."""
        bound, alone, _ = drawn(int(seed))
        with jax.enable_x64(new_val=True):
            model = bound.model.clone(dtype=jnp.float64)
            params = float64_held(bound.variables)
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


def expert_generation(dispatch: str, params, *, layers: int) -> TextGeneration:
    """A greedy task over a decoder of `layers` layers whose mixture of four experts dispatches as
    `dispatch`, over `params`, or drawn weights where None."""
    model = CausalTransformer(vocab_size=VOCAB, emb_features=16, num_layers=layers, num_heads=2, head_dim=8,
                              mlp_features=32, max_seq_len=128, dtype="float32",
                              mixture=Mixture(experts=4, top_k=2, dispatch=dispatch))
    if params is None:
        params = nn.unbox(model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32)))
    return TextGeneration(model, params, RunProcessor(Digits()),
                          sampling=Sampling(temperature=0, eos_token_ids=EOS))


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
        return expert_generation(dispatch, params, layers=1)

    lone = generation("global", None)
    alone = [lone(prompt, budget, key=index)
             for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
    with boxed(decoder_axes(lone.model)):
        placed = place(lone.variables, MeshSpec(expert=4), Layout(min_shard=1, tolerance=1.0))
    served = generation(dispatch, placed)
    server, tickets = served_alongside(served, slots=8, admission=8, kv_cache=KVCache(page_size=16, pages=32),
                                       decode_steps=decode_steps)
    assert_sharded(served.variables["params"], server.mesh)
    assert server.groups == jax.device_count()
    for ticket, row in zip(tickets, [*alone, alone[2]], strict=True):
        assert_same_generation(ticket.result(), row)


def banked_task(sampling: Sampling, *, layers: int = 4, devices: int = 1) -> TextGeneration:
    """A scanned MoE decoder's task over a banked store (`dew.inference.banks.HeldBanks.place`, the run's
    `layers` weights one array a leaf, `layers_0_3`), on one device, as the task's lone calls run (a mesh
    over the suite's devices splits the slots), or on a mesh of the last `devices` rows' devices."""
    def routed(*, scanned: bool) -> CausalTransformer:
        return CausalTransformer(vocab_size=VOCAB, emb_features=16, num_layers=layers, num_heads=2,
                                 head_dim=8, mlp_features=32, max_seq_len=128, dtype="float32",
                                 scan_layers=scanned, mixture=Mixture(experts=4, top_k=2))

    scanned = routed(scanned=True)
    weights = routed(scanned=False).init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    mesh = MeshSpec(fsdp=devices).build(jax.devices()[-devices:] if devices > 1 else jax.devices()[:1])
    banked = HeldBanks(weights).place(scanned, mesh=mesh)
    return TextGeneration(scanned, banked, RunProcessor(Digits()), sampling=sampling)


@pytest.mark.parametrize("options", [{}, {"kv_cache": KVCache(page_size=16, pages=12)}],
                         ids=["dense", "paged"])
def test_a_server_over_a_banked_store_keeps_its_cache_banked_and_draws_what_the_task_draws(options):
    """A scanned MoE decoder over a banked store keeps its resident cache banked as its weights are, the
    run's layers one array under the run's name, which every step reads and writes where it lies: an
    admitted prompt lands in its row of every layer of it, dense or paged, and each row draws what the task
    over the same store draws alone."""
    bound = banked_task(Sampling(temperature=0, eos_token_ids=EOS))
    alone = [bound(prompt, budget, key=index)
             for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
    server, tickets = served_alongside(bound, **options)
    assert "layers_0_3" in server.cache and not any(f"layers_{index}" in server.cache for index in range(4))
    for ticket, row in zip(tickets, [*alone, alone[2]], strict=True):
        assert_same_generation(ticket.result(), row)



@pytest.mark.mesh(devices=4)
def test_a_banked_server_over_row_groups_seats_its_paged_prompts_in_every_layer_of_their_rows():
    """A banked server whose slots split into two row groups, paged, prefills its prompts in a forward of
    their own (`mixed_refusal`): each piece's page table and cursor land in its row of every layer of the
    run's bank, its rows past the layers (`dew.nn.kv_cache.row_axis`), and each row draws what the task
    over the one-device store draws alone."""
    sampling = Sampling(temperature=0, eos_token_ids=EOS)
    lone = banked_task(sampling)
    alone = [lone(prompt, budget, key=index)
             for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
    server, tickets = served_alongside(banked_task(sampling, devices=2),
                                       kv_cache=KVCache(page_size=16, pages=12))
    assert server.groups == 2 and server.mixed_refusal is not None and "layers_0_3" in server.cache
    for ticket, row in zip(tickets, [*alone, alone[2]], strict=True):
        assert_same_generation(ticket.result(), row)


@pytest.mark.mesh(devices=4)
def test_a_banked_server_that_prefills_on_a_mesh_of_its_own_moves_its_rows_past_the_banks_layers():
    """A banked server that prefills on a mesh of its own, its bank three layers deep, which the decoding
    mesh's two row groups do not divide: the prefilled rows move onto the decoding rows rows first, a bank's
    past its layers moved before them (`dew.inference.serving_kernel._rows_first`), not split on the
    layers, and every request draws what the task over the one-device store draws alone."""
    sampling = Sampling(temperature=0, eos_token_ids=EOS)
    lone = banked_task(sampling, layers=3)
    alone = [lone(prompt, budget, key=index)
             for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
    server = Server.from_task(banked_task(sampling, layers=3, devices=2), slots=4, capacity=128, admission=2,
                              prefill_mesh=MeshSpec(fsdp=2).build(jax.devices()[:2]))
    server, tickets = served_by(server)
    assert "layers_0_2" in server.cache
    for ticket, row in zip(tickets, [*alone, alone[2]], strict=True):
        assert_same_generation(ticket.result(), row)


@pytest.mark.distributed
def test_a_pooled_server_that_prefills_on_a_mesh_of_its_own_draws_what_one_process_draws(tmp_path):
    """A server over two processes that decodes on a mesh of each process's first device and prefills on a
    mesh of each process's second: the weights, each admission and its prefilled rows move between the two
    meshes over the pool's hosts, and every request draws what one process serving them all draws, decoding
    on one device and prefilling on another (ledger 875)."""
    from pathlib import Path

    from process_support import run_pool, run_worker

    worker = Path(__file__).with_name("serving_worker.py")
    reports = run_pool("pooled_prefill_apart", tmp_path / "pool", 2, script=worker, devices=2, timeout=300)
    single = run_worker("pooled_prefill_apart", tmp_path / "single.json", script=worker, devices=2)
    assert {**reports[0]["drawn"], **reports[1]["drawn"]} == single["drawn"]


@pytest.mark.mesh(devices=4)
@pytest.mark.parametrize("decoding", [2, 1, 0], ids=["two", "one", "unmeshed"])
def test_a_server_that_prefills_on_a_mesh_of_its_own_draws_what_one_mesh_draws(decoding):
    """Disaggregated serving: the prompts prefill on two devices of their own and their rows' decoder state
    moves to the decoding devices, a mesh of two or of one or one device with no mesh, where they decode;
    every request, a prompt arriving while others decode among them, draws what its lone call draws, and
    a reload reaches the prefill mesh's copy as the decoding one's (ledger 875)."""
    from dew.training import Layout, MeshSpec

    devices = jax.devices()[:4]
    bound = task(Sampling(temperature=1.0))
    layout = Layout(min_shard=1, tolerance=1.0)
    with boxed(decoder_axes(bound.model)):
        decoded = (place(bound.variables, MeshSpec(fsdp=decoding).build(devices[4 - decoding:]), layout)
                   if decoding else jax.device_put(bound.variables, jax.devices()[0]))
    served = dataclasses.replace(bound, variables=decoded)
    alone = [bound(prompt, budget, key=index)
             for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
    # Two devices apart from the decoding ones.
    prefilling = MeshSpec(fsdp=2).build(devices[:2] if decoding else devices[1:3])
    server = Server.from_task(served, slots=4, capacity=128, admission=2, prefill_mesh=prefilling)
    server, tickets = served_by(server)
    for ticket, row in zip(tickets, [*alone, alone[2]], strict=True):
        assert_same_generation(ticket.result(), row)
    moved = jax.tree.map(lambda leaf: leaf * 0.5, bound.variables)
    server.reload(moved)
    ticket = server.submit(PROMPTS[0], BUDGETS[0], key=0)
    server.run()
    assert_same_generation(ticket.result(), dataclasses.replace(bound, variables=moved)(
        PROMPTS[0], BUDGETS[0], key=0))


@pytest.mark.mesh(devices=4)
def test_a_server_on_an_expert_and_stage_mesh_exchanges_its_experts_and_runs_its_stack_whole():
    """A mixture whose dispatch exchanges tokens over the expert axis, placed on a mesh with a pipeline's
    stage axis beside it, and built and stepped inside that training mesh's scope: the server decodes its
    stack whole while the expert axis stays in context for the exchange (`dew.nn.sharding.whole`), and
    serves what the global dispatch draws alone on one device (ledger 1817)."""
    lone = expert_generation("global", None, layers=2)
    alone = [lone(prompt, budget, key=index)
             for index, (prompt, budget) in enumerate(zip(PROMPTS, BUDGETS, strict=True))]
    spec = MeshSpec(expert=2, stage=2)
    with boxed(decoder_axes(lone.model)):
        placed = place(lone.variables, spec, Layout(min_shard=1, tolerance=1.0))
    served = expert_generation("exchange", placed, layers=2)
    with jax.set_mesh(spec.build()):
        _, tickets = served_alongside(served, slots=8, admission=8)
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


def test_a_tree_split_for_training_packs_as_its_whole_tree_and_splits_back_as_it_was():
    """A tree split for training (`dew.batch.freeze`), its first layer's keys and values frozen and its
    queries trained and its second layer trained whole, packs as its whole tree does (ledger 2439): thawed,
    it is the unsplit weights packed, the flags of its frozen members (`PACKED_FROZEN`) gone with the
    split, so a model reads the same tree whatever its layers' membership; split, it is the tree it was,
    each member in its own collection; and a server of it reloads its unsplit source weights."""
    from dew.batch import PACKED_FROZEN, freeze, thaw
    from dew.inference.projections import inference_projections, packed_owned, unpacked

    model = decoder().clone(num_layers=2, max_seq_len=128)
    whole = nn.unbox(model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32)))
    split = freeze(whole, lambda path: not ("layers_0" in path and {"k_proj", "v_proj"} & set(path)))
    packed = packed_owned(model, jax.tree.map(jnp.array, split))
    assert PACKED_FROZEN in packed and PACKED_FROZEN not in thaw(packed)

    def assert_trees_equal(tree, expected):
        assert jax.tree.structure(tree) == jax.tree.structure(expected)
        for leaf, alike in zip(jax.tree.leaves(tree), jax.tree.leaves(expected), strict=True):
            np.testing.assert_array_equal(np.asarray(leaf), np.asarray(alike), strict=True)

    assert_trees_equal(thaw(packed), inference_projections(model, whole))
    assert_trees_equal(unpacked(model, packed), split)
    sampling = Sampling(temperature=0, eos_token_ids=None)
    server = Server.from_task(TextGeneration(model, packed, RunProcessor(Digits()), sampling=sampling),
                              slots=2, capacity=128)
    server.reload(whole)
    alone = TextGeneration(model, whole, RunProcessor(Digits()), sampling=sampling)
    for actual, prompt in zip(server(PROMPTS[:2], 5, key=3), PROMPTS[:2], strict=True):
        assert_same_generation(actual, alone(prompt, 5, key=3))


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
