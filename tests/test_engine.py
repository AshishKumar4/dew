"""The persistent engine against batch generation: parity, reuse, versions, bounds."""

import dataclasses
import json
import threading
import time
from concurrent.futures import CancelledError
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.diffusion.block import CanvasGeneration, SpanEvent
from dew.interop.pretrained import load_pretrained
from dew.nn.inputs import ModelInputs
from dew.sampling import Sampling, generate
from dew.sampling.engine import CapacityError, Engine, WeightVersion
from dew.sampling.text import TokenEvent
from test_text_rollout_contract import decoder

FIXTURES = Path(__file__).parent / "fixtures" / "hf"
SAMPLING = Sampling(temperature=0.9, top_k=6, eos_id=(3, 9))


def text(tokens, valid=None):
    tokens = jnp.asarray(tokens, jnp.int32)
    mask = jnp.ones(tokens.shape, bool) if valid is None else jnp.asarray(valid, bool)
    return ModelInputs(tokens, {"attention_mask": mask})


REQUESTS = [
    (text([[1, 2, 3]]), 6, jax.random.key(1)),
    (text([[0, 5, 6, 7, 8]], [[False, True, True, True, True]]), 4, jax.random.key(2)),
    (text([[4, 4], [7, 1]]), 6, jax.random.key(3)),
    (text([[2, 2, 2, 2]]), 5, jax.random.key(4)),
    (text([[6]]), 6, jax.random.key(5)),
    (text([[6]]), 0, jax.random.key(6)),
]


def initialized(kind="attention"):
    model = decoder(kind).clone(max_seq_len=64)
    return model, model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))


def assert_same_generation(actual, expected):
    np.testing.assert_array_equal(actual.tokens, expected.tokens)
    np.testing.assert_array_equal(actual.lengths, expected.lengths)
    np.testing.assert_array_equal(actual.terminated, expected.terminated)
    np.testing.assert_allclose(actual.behavior_log_probs, expected.behavior_log_probs, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(actual.raw_log_probs, expected.raw_log_probs, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("kind, steps", [("attention", 1), ("mla", 2), ("recurrent", 3)])
def test_concurrent_requests_share_decode_rows_and_match_batch_generation(kind, steps):
    """Mixed widths, a two-row request, EOS at different steps and a zero
    budget, all inside a four-row bound that forces slot growth and shrink."""
    model, params = initialized(kind)
    expected = [generate(model, params, inputs, budget, key=key, sampling=SAMPLING)
                for inputs, budget, key in REQUESTS]
    with Engine(model, params, max_batch_size=4, steps_per_dispatch=steps) as engine:
        jobs = [engine.submit(inputs, budget, key=key, generation=SAMPLING) for inputs, budget, key in REQUESTS]
        events = list(jobs[2].stream())
        results = [job.result(timeout=120) for job in jobs]
        stats = engine.stats()
    for actual, reference in zip(results, expected):
        assert_same_generation(actual, reference)
    assert all(job.status == "done" for job in jobs)
    two_rows = results[2]
    for row in range(2):
        mine = [event for event in events if event.row == row]
        assert [event.position for event in mine] == list(range(int(two_rows.lengths[row])))
        assert [event.token for event in mine] == np.asarray(two_rows.tokens[row, 2:2 + len(mine)]).tolist()
        np.testing.assert_allclose([event.behavior_log_prob for event in mine],
                                   np.asarray(two_rows.behavior_log_probs[row, :len(mine)]), atol=1e-6)
        assert [event.terminated for event in mine][-1] == bool(two_rows.terminated[row])
        assert not any(event.terminated for event in mine[:-1])
    assert stats.active == 0 and stats.pending == 0 and stats.allocated_rows == 0
    assert stats.prefix_misses == 5 and stats.prefix_hits == 0


def test_prefix_reuse_is_deterministic_bounded_and_invisible_in_results():
    model, params = initialized()
    prompt, other = text([[1, 2, 3, 4]]), text([[1, 2, 3, 5]])
    reference = generate(model, params, prompt, 5, key=jax.random.key(7), sampling=SAMPLING)
    from dew.sampling.text import _prefill
    one_prefix = sum(leaf.nbytes for leaf in jax.tree.leaves(_prefill(model, params, prompt)))
    with Engine(model, params, max_batch_size=2, prefix_cache_bytes=one_prefix) as engine:
        first = engine.submit(prompt, 5, key=jax.random.key(7), generation=SAMPLING).result(timeout=60)
        repeated = engine.submit(prompt, 5, key=jax.random.key(7), generation=SAMPLING).result(timeout=60)
        greedy = engine.submit(prompt, 5, key=jax.random.key(8), generation=Sampling(temperature=0)).result(timeout=60)
        after_hits = engine.stats()
        engine.submit(other, 5, key=jax.random.key(7), generation=SAMPLING).result(timeout=60)
        evicted = engine.stats()
        again = engine.submit(prompt, 5, key=jax.random.key(7), generation=SAMPLING).result(timeout=60)
        final = engine.stats()
    for actual in (first, repeated, again):
        assert_same_generation(actual, reference)
    assert_same_generation(greedy, generate(model, params, prompt, 5, key=jax.random.key(8),
                                            sampling=Sampling(temperature=0)))
    assert after_hits.prefix_hits == 2 and after_hits.prefix_misses == 1 and after_hits.prefix_entries == 1
    assert evicted.prefix_misses == 2 and evicted.prefix_entries == 1 and evicted.prefix_bytes <= one_prefix
    assert final.prefix_misses == 3 and final.prefix_entries == 1


def test_conditioning_joins_prefix_identity_for_a_loaded_source():
    """Equal token ids with different pixels miss the cache and change raw likelihoods."""
    loaded = load_pretrained(FIXTURES / "gemma3-native-tiny", dtype="float32", attention_impl="reference")
    images = np.load(FIXTURES / "gemma3-native-tiny" / "raw_images.npy")
    prompts = json.loads((FIXTURES / "gemma3-native-tiny" / "prompts.json").read_text())
    assert loaded.processor is not None
    inputs = loaded.processor(prompts, images=[[images[0]], [images[1], images[2]]])
    changed = dataclasses.replace(inputs, conditioning={
        **inputs.conditioning, "pixel_values": -inputs.conditioning["pixel_values"]})
    expected = loaded.generate(inputs, 3, key=jax.random.key(1))
    with Engine(loaded, max_batch_size=4, prefix_cache_bytes=1 << 24) as engine:
        original = engine.submit(inputs, 3, key=jax.random.key(1)).result(timeout=120)
        other = engine.submit(changed, 3, key=jax.random.key(1)).result(timeout=120)
        repeated = engine.submit(inputs, 3, key=jax.random.key(1)).result(timeout=120)
        stats = engine.stats()
    assert_same_generation(original, expected)
    assert_same_generation(repeated, expected)
    assert stats.prefix_misses == 2 and stats.prefix_hits == 1
    assert float(jnp.max(jnp.abs(original.raw_log_probs - other.raw_log_probs))) > 1e-3


def test_publication_pins_copied_versions_and_retires_unpinned_ones():
    model, params = initialized()
    prompt = text([[1, 2, 3]])
    sampling = Sampling(temperature=0.7, top_k=4)
    newer = jax.tree.map(lambda leaf: leaf + 0.05, params)
    old_reference = generate(model, params, prompt, 12, key=jax.random.key(1), sampling=sampling)
    new_reference = generate(model, newer, prompt, 12, key=jax.random.key(2), sampling=sampling)
    with Engine(model, params, max_batch_size=2, max_weight_versions=2) as engine:
        first_version = engine.version
        old = engine.submit(prompt, 12, key=jax.random.key(1), generation=sampling)
        stream = old.stream()
        next(stream)
        # The condition is re-entrant; holding it parks the controller between
        # dispatches so the publication happens while the first request runs.
        with engine._condition:
            version = engine.publish(newer)
            # Trainer-style donation of the publisher's arrays after publication.
            donated = jax.jit(lambda tree: jax.tree.map(lambda leaf: leaf * 2.0, tree), donate_argnums=(0,))(newer)
            jax.block_until_ready(donated)
            assert all(leaf.is_deleted() for leaf in jax.tree.leaves(newer))
            new = engine.submit(prompt, 12, key=jax.random.key(2), generation=sampling)
            pinned = engine.submit(prompt, 12, key=jax.random.key(1), generation=sampling, version=first_version)
            with pytest.raises(ValueError):
                engine.publish({"params": {"wrong": jnp.zeros(3)}})
            with pytest.raises(CapacityError):
                engine.publish(params)
            with pytest.raises(ValueError):
                engine.submit(prompt, 2, key=jax.random.key(3), version=WeightVersion("other", 0))
        list(stream)
        results = [job.result(timeout=120) for job in (old, new, pinned)]
        assert old.version == pinned.version == first_version and new.version == version
        assert engine.stats().versions == 1 and engine.version == version
        with pytest.raises(ValueError):
            engine.submit(prompt, 2, key=jax.random.key(3), version=first_version)
    assert_same_generation(results[0], old_reference)
    assert_same_generation(results[1], new_reference)
    assert_same_generation(results[2], old_reference)


def test_cancellation_timeouts_and_close_boundaries():
    model, params = initialized()
    prompt = text([[1, 2, 3]])
    sampling = Sampling(temperature=0.8)
    with Engine(model, params, max_batch_size=1) as engine:
        running = engine.submit(prompt, 40, key=jax.random.key(1), generation=sampling)
        queued = engine.submit(prompt, 40, key=jax.random.key(2), generation=sampling)
        assert queued.cancel() and queued.status == "cancelled" and not queued.cancel()
        with pytest.raises(CancelledError):
            queued.result()
        with pytest.raises(TimeoutError):
            running.result(timeout=0.01)
        stream = running.stream()
        first = next(stream)
        assert isinstance(first, TokenEvent) and running.status == "active"
        assert running.cancel()
        with pytest.raises(CancelledError):
            list(stream)
        with pytest.raises(CancelledError):
            running.result(timeout=30)
        with pytest.raises(RuntimeError):
            next(running.stream())
        drained = engine.submit(prompt, 3, key=jax.random.key(3), generation=sampling)
    assert drained.status == "done" and int(drained.result().lengths[0]) == 3
    with pytest.raises(RuntimeError):
        engine.submit(prompt, 3, key=jax.random.key(4))

    engine = Engine(model, params, max_batch_size=1).start()
    active = engine.submit(prompt, 40, key=jax.random.key(1), generation=sampling)
    waiting = engine.submit(prompt, 40, key=jax.random.key(2), generation=sampling)
    engine.close(cancel=True, timeout=60)
    assert active.status == "cancelled" and waiting.status == "cancelled"
    assert engine.stats().versions == 0 and engine.stats().allocated_rows == 0


def test_admission_bounds_reject_without_accepting():
    model, params = initialized()
    prompt = text([[1, 2, 3]])
    with Engine(model, params, max_batch_size=2, max_pending_requests=1, max_request_bytes=400) as engine:
        with pytest.raises(CapacityError):
            engine.submit(text([[1, 2], [3, 4], [5, 6]]), 2, key=jax.random.key(1))
        with pytest.raises(CapacityError):
            engine.submit(prompt, 30, key=jax.random.key(1))
        blocker = engine.submit(prompt, 6, key=jax.random.key(1), generation=Sampling(temperature=1.0))
        deadline = time.monotonic() + 60
        while blocker.status == "waiting" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert blocker.status == "active"
        with engine._condition:
            # A second cohort needs rows the first one holds; it waits.
            second = engine.submit(prompt, 6, key=jax.random.key(2), generation=Sampling(temperature=0.5))
            with pytest.raises(CapacityError):
                engine.submit(prompt, 6, key=jax.random.key(3), generation=Sampling(temperature=0.25))
        blocker.result(timeout=60)
        second.result(timeout=60)
        assert engine.stats().pending == 0


def test_a_failing_request_does_not_take_its_peers_down():
    model, params = initialized()
    sampling = Sampling(temperature=0.5, eos_id=3)
    good_inputs = text([[1, 2, 3]])
    bad_inputs = dataclasses.replace(good_inputs, token_fields={
        **good_inputs.token_fields, "unsupported_field": good_inputs.tokens})
    expected = generate(model, params, good_inputs, 5, key=jax.random.key(1), sampling=sampling)
    with Engine(model, params, max_batch_size=2) as engine:
        bad = engine.submit(bad_inputs, 5, key=jax.random.key(2), generation=sampling)
        good = engine.submit(good_inputs, 5, key=jax.random.key(1), generation=sampling)
        with pytest.raises(TypeError, match="unsupported_field"):
            bad.result(timeout=60)
        assert_same_generation(good.result(timeout=60), expected)
        assert engine.stats().active == 0 and engine.stats().allocated_rows == 0


@pytest.mark.parametrize("steps", [1, 2])
def test_canvas_requests_stay_atomic_and_match_block_generation(steps):
    bundle = load_pretrained(FIXTURES / "diffusion-gemma-workflow", dtype="float32", attention_impl="xla",
                             max_seq_len=32)
    assert bundle.processor is not None and bundle.generation_adapter is not None
    inputs = bundle.processor(["<bos> t5 t7 t9 t11", "<bos> t6 t8 t10 t12"])
    with np.load(FIXTURES / "diffusion-gemma-workflow" / "reference.npz") as reference:
        eos = int(reference["eos_id"])
    stopping = dataclasses.replace(bundle.generation_adapter, eos_token_ids=(eos,))
    expected = bundle.generate(inputs, 12, key=jax.random.key(11))
    expected_stop = stopping.generate(bundle.model, bundle.variables, inputs, 7, key=jax.random.key(11))
    with Engine(bundle, max_batch_size=4, prefix_cache_bytes=1 << 24, steps_per_dispatch=steps) as engine:
        plain = engine.submit(inputs, 12, key=jax.random.key(11))
        events = list(plain.stream())
        result = plain.result(timeout=120)
        cancelled = engine.submit(inputs, 12, key=jax.random.key(11))
        spans = cancelled.stream()
        first = next(spans)
        # Two canvases remain; the re-entrant condition keeps the controller
        # from a terminal transition between the event and the cancellation.
        with engine._condition:
            assert cancelled.cancel()
        with pytest.raises(CancelledError):
            cancelled.result(timeout=120)
        assert first.tokens and cancelled.status == "cancelled"
        stats = engine.stats()
    assert isinstance(result, CanvasGeneration)
    np.testing.assert_array_equal(result.tokens, expected.tokens)
    np.testing.assert_array_equal(result.lengths, expected.lengths)
    np.testing.assert_array_equal(result.decoder_steps, expected.decoder_steps)
    assert all(isinstance(event, SpanEvent) for event in events)
    for row in range(2):
        spans = [event for event in events if event.row == row]
        committed = [token for event in spans for token in event.tokens]
        assert committed == np.asarray(result.tokens[row, 5:5 + int(result.lengths[row])]).tolist()
        assert sum(event.decoder_steps for event in spans) == int(result.decoder_steps[row])
    assert stats.prefix_hits == 1 and stats.prefix_misses == 1 and stats.allocated_rows == 0
    with Engine(bundle.model, bundle.variables, family=stopping, max_batch_size=2) as engine:
        stopped = engine.submit(inputs, 7, key=jax.random.key(11))
        spans = list(stopped.stream())
        result = stopped.result(timeout=120)
    np.testing.assert_array_equal(result.tokens, expected_stop.tokens)
    np.testing.assert_array_equal(result.lengths, expected_stop.lengths)
    np.testing.assert_array_equal(result.terminated, expected_stop.terminated)
    assert [event.terminated for event in spans if event.row == 0] == [True]


def test_slow_stream_consumers_and_parallel_submitters_see_complete_results():
    model, params = initialized()
    sampling = Sampling(temperature=0.6, top_k=5, eos_id=9)
    prompts = [text([[1 + index, 2, 3]]) for index in range(6)]
    expected = [generate(model, params, prompt, 8, key=jax.random.key(index), sampling=sampling)
                for index, prompt in enumerate(prompts)]
    results = [None] * len(prompts)

    def worker(engine, index):
        job = engine.submit(prompts[index], 8, key=jax.random.key(index), generation=sampling)
        seen = []
        for event in job.stream():
            seen.append(event.token)
            time.sleep(0.002)
        results[index] = job.result(timeout=120), seen

    with Engine(model, params, max_batch_size=4, max_pending_requests=8) as engine:
        threads = [threading.Thread(target=worker, args=(engine, index)) for index in range(len(prompts))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=180)
    for (actual, seen), reference in zip(results, expected):
        assert_same_generation(actual, reference)
        assert seen == np.asarray(reference.tokens[0, 3:3 + int(reference.lengths[0])]).tolist()


@pytest.mark.mesh
def test_mesh_placed_parameters_generate_the_same_rows():
    from dew.training import Layout, MeshSpec
    from dew.training.distributed import build_mesh

    model, params = initialized()
    mesh = build_mesh(MeshSpec(fsdp=2))
    placed = jax.device_put(params, Layout(min_shard=2 ** 8).shardings(mesh, params))
    prompt = text([[1, 2, 3], [4, 5, 6]])
    expected = generate(model, params, prompt, 5, key=jax.random.key(1), sampling=SAMPLING)
    with Engine(model, placed, max_batch_size=4) as engine:
        result = engine.submit(prompt, 5, key=jax.random.key(1), generation=SAMPLING).result(timeout=120)
    assert_same_generation(result, expected)
