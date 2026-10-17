"""Dew's own continuous-batching server as a rollout server.

A push lands between steps and later submissions report its version; the
served tree keeps the server's precision; a refused request raises at
submission and leaves the rest running; a draw that fails its checks fails
its future instead of leaving it pending; a submission does not wait for a
step.
"""

import threading

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from episode_support import ToolPolicy
from process_support import assert_exits_cleanly

from dew.inference import NativeRolloutServer, TextGeneration
from dew.inference.serving import Server
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.rl import GRPOObjective
from dew.sampling import Sampling

VOCAB = 13
EOS = 12
BUDGET = 4
SAMPLING = Sampling(temperature=1.0, eos_token_ids=EOS)


def objective():
    model = CausalTransformer(vocab_size=VOCAB, emb_features=16, num_layers=1, num_heads=2,
                              head_dim=8, mlp_features=32, max_seq_len=64, dtype="float32")
    return GRPOObjective(model, seq_len=9)


def test_the_native_server_serves_loaded_weights_at_its_own_precision():
    target = objective()
    params = target.init(jax.random.key(0))
    served = jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16), params)
    backend = Server.from_task(
        TextGeneration(target.model, served, None, sampling=SAMPLING), slots=2, capacity=64
    )
    server = NativeRolloutServer(backend)
    try:
        before = server.submit([1, 2, 3], BUDGET, key=5).result()
        trained = jax.tree.map(lambda leaf: leaf * 1.5, params)
        server.load(trained, 7)
        after = server.submit([1, 2, 3], BUDGET, key=5).result()
    finally:
        server.close()
    assert (before.version, after.version) == (0, 7)
    assert before.behavior_log_probs != after.behavior_log_probs
    assert all(leaf.dtype == jnp.bfloat16 for leaf in jax.tree.leaves(backend.variables))
    expected = TextGeneration(target.model, jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16), trained),
                              None, sampling=SAMPLING)([[1, 2, 3]], BUDGET, key=5).host()
    count = len(after.tokens)
    assert after.tokens == tuple(int(token) for token in expected.tokens[0, 3:3 + count])
    np.testing.assert_allclose(after.behavior_log_probs, expected.behavior_log_probs[0, :count],
                               atol=2e-6, rtol=2e-6)


def test_native_rollouts_preserve_a_split_jax_key():
    target = objective()
    params = target.init(jax.random.key(0))
    task = TextGeneration(target.model, params, None, sampling=SAMPLING)
    key = jax.random.split(jax.random.key(19))[1]
    expected = task([[1, 2, 3]], BUDGET, key=key).host()
    server = NativeRolloutServer(Server.from_task(task, slots=2, capacity=64))
    try:
        actual = server.submit([1, 2, 3], BUDGET, key=key).result(timeout=120)
    finally:
        server.close()
    count = int(expected.lengths[0])
    assert actual.tokens == tuple(int(token) for token in expected.tokens[0, 3:3 + count])


@pytest.fixture
def native_server():
    target = objective()
    params = target.init(jax.random.key(0))
    server = NativeRolloutServer(
        Server.from_task(TextGeneration(target.model, params, None, sampling=SAMPLING), slots=2, capacity=64)
    )
    yield server
    server.close()


def test_a_task_whose_criteria_end_draws_before_eos_is_refused():
    """A draw tells EOS from its budget alone; a task whose stop strings or
    criteria would end it on neither cannot back a rollout server."""
    from dew.sampling.decoding import MaxNewTokens

    target = objective()
    task = TextGeneration(target.model, target.init(jax.random.key(0)), None,
                          sampling=Sampling(temperature=1.0, eos_token_ids=None), stopping=(MaxNewTokens(1),))
    with pytest.raises(ValueError):
        NativeRolloutServer(Server.from_task(task, slots=2, capacity=64))


def test_a_refused_request_raises_at_submit_and_the_native_server_keeps_serving(native_server):
    running = native_server.submit([1, 2, 3], 40, key=1)
    with pytest.raises(ValueError):
        native_server.submit(list(range(1, 11)) * 6, 10, key=2)
    assert len(running.result(timeout=120).tokens) >= 1
    assert native_server.submit([4, 5], 3, key=3).result(timeout=120).prompt == (4, 5)


def test_a_draw_that_fails_its_checks_fails_its_future_instead_of_hanging():
    """A bigram policy so peaked that the model's own likelihood of every id but one underflows
    to -inf, sampled hot enough to draw those ids. No `Draw` holds a non-finite likelihood, so
    the draw's future fails with that error, where a check raised in the done-callback reached
    no one and left the future pending."""
    vocab = 64

    def peaked():
        table = np.full((vocab, vocab), -2e38, np.float32)
        table[:, 0] = 2e38
        return table

    model = ToolPolicy(vocab_size=vocab, max_seq_len=64, logits=peaked)
    params = model.init(jax.random.key(0), jnp.ones((1, 1), jnp.int32))
    # At this temperature the scores are 1 and -1, so the ids the model gives -inf are drawn.
    sampling = Sampling(temperature=2e38, eos_token_ids=vocab - 1)
    server = NativeRolloutServer(
        Server.from_task(TextGeneration(model, params, None, sampling=sampling), slots=2, capacity=64)
    )
    try:
        with pytest.raises(ValueError):
            server.submit([1, 2, 3], BUDGET, key=5).result(timeout=60)
    finally:
        server.close()


def test_a_submission_does_not_wait_for_the_step_under_way(native_server):
    """Draws submitted while another thread holds the step lock return at
    once, and are served once the step ends."""
    # A rollout source submits a batch's draws one after another; each waiting
    # for a step made one-turn RLVR eight times slower (ledger 518).
    stepping, done = threading.Event(), threading.Event()

    def step_under_way():
        with native_server._lock:
            stepping.set()
            done.wait()

    holder = threading.Thread(target=step_under_way)
    futures = []
    submitter = threading.Thread(target=lambda: futures.extend(
        native_server.submit([1, 2, 3], BUDGET, key=seed) for seed in range(4)))
    try:
        holder.start()
        stepping.wait()
        submitter.start()
        submitter.join(timeout=60)
        returned = not submitter.is_alive() and len(futures) == 4
    finally:
        done.set()
        holder.join()
        submitter.join()
    assert returned
    assert all(future.result(timeout=120).prompt == (1, 2, 3) for future in futures)


def test_a_program_that_ends_with_draws_in_flight_exits_cleanly():
    """The stepping thread spends its steps inside jaxlib. A program that
    ended without `close` while draws were in flight left it there when
    Python finalized, and the process aborted after its last line (SIGABRT,
    "terminate called ..."). The stepping now stops at exit."""
    program = (
        "import jax, jax.numpy as jnp\n"
        "from dew.inference import NativeRolloutServer, TextGeneration\n"
        "from dew.inference.serving import Server\n"
        "from dew.nn.backbones.causal_transformer import CausalTransformer\n"
        "from dew.sampling import Sampling\n"
        "model = CausalTransformer(vocab_size=13, emb_features=16, num_layers=1, num_heads=2, head_dim=8,\n"
        "                          mlp_features=32, max_seq_len=512, dtype='float32')\n"
        "params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))\n"
        "task = TextGeneration(model, params, None, sampling=Sampling(temperature=1.0))\n"
        "servers = [NativeRolloutServer(Server.from_task(task, slots=4, capacity=512)) for _ in range(4)]\n"
        "draws = [server.submit([1, 2, 3], 400, key=seed) for server in servers for seed in range(64)]\n"
        "draws[0].result()\n")
    assert_exits_cleanly(program)
