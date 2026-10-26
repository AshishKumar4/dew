"""Dew's own continuous-batching server as a rollout server.

A push lands between steps and later submissions report its version; the
served tree keeps the server's precision; a refused request raises at
submission and leaves the rest running; a draw that fails its checks fails
its future instead of leaving it pending; a submission does not wait for a
step.
"""

import itertools
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
KEY = jax.random.key(7)
VOCAB_PIECES = 11


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


def test_a_native_server_starts_stepping_at_its_first_submission():
    """Constructing a rollout server starts no thread (api spec A15); the first submission starts
    its stepping, and a server closed before any submission closes at once."""
    target = objective()
    params = target.init(jax.random.key(0))
    task = TextGeneration(target.model, params, None, sampling=SAMPLING)

    def stepping():
        return [thread for thread in threading.enumerate() if thread.name == "dew-rollout-server"]

    before = len(stepping())
    unused = NativeRolloutServer(Server.from_task(task, slots=2, capacity=64))
    assert len(stepping()) == before
    unused.close()
    server = NativeRolloutServer(Server.from_task(task, slots=2, capacity=64))
    try:
        assert len(stepping()) == before
        server.submit([1, 2, 3], 2, key=0).result()
        assert len(stepping()) == before + 1
    finally:
        server.close()
    assert len(stepping()) == before


def test_a_draw_a_criterion_cuts_short_is_truncated_and_one_the_task_alone_ends_is_refused():
    """A criterion that truncates (`MaxNewTokens`) cuts a draw short, unterminated, as a budget does.
    One of the task's own that ends rows on neither an EOS id nor a stop string ends draws no
    policy names, so it cannot back a rollout server."""
    from dew.sampling.decoding import MaxNewTokens

    target = objective()
    params = target.init(jax.random.key(0))
    unstopped = Sampling(temperature=1.0, eos_token_ids=None)
    server = NativeRolloutServer(Server.from_task(
        TextGeneration(target.model, params, None, sampling=unstopped, stopping=(MaxNewTokens(1),)),
        slots=2, capacity=64))
    try:
        draw = server.submit([1, 2, 3], BUDGET, key=5).result(timeout=120)
    finally:
        server.close()
    assert len(draw.tokens) == 1 and not draw.terminated

    def three(state, tokens):
        return tokens == 3

    task = TextGeneration(target.model, params, None, sampling=unstopped, stopping=(three,))
    with pytest.raises(ValueError):
        NativeRolloutServer(Server.from_task(task, slots=2, capacity=64))
    # An EOS criterion beside the policy's own ends draws on an id the policy does not name.
    from dew.sampling.decoding import EndOfSequence

    task = TextGeneration(target.model, params, None, sampling=SAMPLING,
                          stopping=(EndOfSequence(jnp.asarray([3], jnp.int32)),))
    with pytest.raises(ValueError):
        NativeRolloutServer(Server.from_task(task, slots=2, capacity=64))


@pytest.fixture(scope="module")
def pieces(tmp_path_factory):
    """A tokenizer of whole pieces, `<eos>` among them, whose ids spell text by joining; the letters
    spell the probe `stop_strings` reads a piece's own text with."""
    from tokenizers import Tokenizer, decoders, models
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    vocabulary = {piece: index for index, piece in enumerate(["<unk>", "<eos>", "st", "op", "yy", *"abcdef"])}
    backend = Tokenizer(models.BPE(vocabulary, [], unk_token="<unk>"))
    backend.decoder = decoders.Fuse()
    path = tmp_path_factory.mktemp("pieces")
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>").save_pretrained(path)
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def test_a_native_draw_ends_on_the_stop_string_it_completes(pieces):
    """A bigram policy that spells "st", "op", "yy", "<eos>" after "a". A stop string ends the draw on
    the id that completes it, terminated, and the draw keeps that id, as `TextGeneration` and the
    remote engines do; a policy whose string the draw never spells ends on its EOS id, and one with
    neither runs to its budget."""
    from dew.inference.loading import RunProcessor
    from dew.objectives.rl.rollout import prompt_tasks
    from dew.objectives.rl.sources import PromptSource

    ids = {piece: pieces.convert_tokens_to_ids(piece) for piece in ("<eos>", "st", "op", "a", "yy")}
    order = [ids["a"], ids["st"], ids["op"], ids["yy"], ids["<eos>"]]

    def scripted():
        table = np.full((VOCAB_PIECES, VOCAB_PIECES), -1e4, np.float32)
        for before, after in itertools.pairwise(order):
            table[before, after] = 1e4
        table[ids["<eos>"], ids["<eos>"]] = 1e4
        return table

    model = ToolPolicy(vocab_size=VOCAB_PIECES, max_seq_len=64, logits=scripted)
    params = model.init(jax.random.key(0), jnp.ones((1, 1), jnp.int32))

    def drawn(sampling, budget=6):
        task = TextGeneration(model, params, RunProcessor(pieces), sampling=sampling)
        server = NativeRolloutServer(Server.from_task(task, slots=2, capacity=64))
        try:
            draw = server.submit([ids["a"]], budget, key=5).result(timeout=120)
        finally:
            server.close()
        expected = task([[ids["a"]]], budget, key=5).host()
        count = int(expected.lengths[0])
        assert draw.tokens == tuple(int(token) for token in expected.tokens[0, 1:1 + count])
        assert draw.terminated == bool(expected.terminated[0])
        return draw

    stopped = drawn(Sampling(temperature=1.0, eos_token_ids=ids["<eos>"], stop=("stop",)))
    assert (stopped.tokens, stopped.terminated) == ((ids["st"], ids["op"]), True)
    assert stopped.response(Sampling(eos_token_ids=ids["<eos>"])) == stopped.tokens
    ended = drawn(Sampling(temperature=1.0, eos_token_ids=ids["<eos>"], stop=("yya",)))
    assert (ended.tokens, ended.terminated) == ((ids["st"], ids["op"], ids["yy"], ids["<eos>"]), True)
    cut = drawn(Sampling(temperature=1.0, stop=("yya",)), budget=2)
    assert (cut.tokens, cut.terminated) == ((ids["st"], ids["op"]), False)

    # The prompt source scores the stopped draw's text with its string in it.
    seen = []
    task = TextGeneration(model, params, RunProcessor(pieces),
                          sampling=Sampling(temperature=1.0, eos_token_ids=ids["<eos>"], stop=("stop",)))
    server = NativeRolloutServer(Server.from_task(task, slots=2, capacity=64))
    batch = {"prompt": np.asarray([[ids["a"]]], np.int32), "prompt_length": np.asarray([1], np.int32),
             **{name: np.zeros((1, 8), np.int32) for name in ("data_source", "ground_truth", "extra_info")}}
    [prompt] = prompt_tasks(batch)
    source = PromptSource(server, lambda text, record: seen.append(text) or 1.0, decode=pieces.decode,
                          max_new_tokens=6)
    try:
        [session] = [future.result(timeout=120) for future in source.submit(prompt, 1, version=0, key=KEY)]
    finally:
        source.close()
        server.close()
    assert seen == ["stop"] and session.calls[0].finish_reason == "stop"

    # A tool episode under a policy with a stop string and no EOS id: each turn ends on the string,
    # the environment steps on it, and the turns chain into one packed row of trained actions.
    from contextlib import contextmanager

    from dew.objectives.rl import EnvironmentSource
    from dew.objectives.rl.episodes import EpisodeStatus, Observation
    from dew.objectives.rl.sessions import Status, Task, pack

    class Turns:
        def __init__(self):
            self.context, self.steps = (ids["a"],), 0

        def reset(self):
            return Observation(self.context)

        def step(self, action):
            self.steps += 1
            self.context = (*action.prompt, *action.tokens, ids["a"])
            if self.steps == 2:
                return Observation((), EpisodeStatus.COMPLETED, "two turns")
            return Observation(self.context)

    @contextmanager
    def environment(task, identity):
        yield Turns()

    task = TextGeneration(model, params, RunProcessor(pieces),
                          sampling=Sampling(temperature=1.0, stop=("stop",)))
    server = NativeRolloutServer(Server.from_task(task, slots=2, capacity=64))
    episodes = EnvironmentSource(server, environment, lambda task, episode: 1.0, max_prompt_tokens=16,
                                 max_new_tokens=6, max_turns=4)
    try:
        [session] = [future.result(timeout=120)
                     for future in episodes.submit(Task("0"), 1, version=0, key=KEY)]
    finally:
        episodes.close()
        server.close()
    assert session.status == Status.COMPLETED
    assert [(call.tokens, call.finish_reason) for call in session.calls] == [
        ((ids["st"], ids["op"]), "stop")] * 2
    packed = pack([session], 16)
    np.testing.assert_array_equal(packed["input_ids"][0][packed["response_mask"][0] > 0],
                                  [ids["st"], ids["op"]] * 2)


def test_a_draw_that_terminated_ends_as_its_policy_ends_one():
    """A terminated draw ends on an EOS id of its policy or on the id completing one of its stop
    strings, which any id may be; one that ended on neither, or ran past an EOS id, breaks its
    contract."""
    from dew.inference import Draw

    def draw(tokens, terminated):
        return Draw((1,), tokens, (-0.5,) * len(tokens), "stop" if terminated else "length", 0)

    stopping, plain = Sampling(eos_token_ids=EOS, stop=("x",)), Sampling(eos_token_ids=EOS)
    for tokens, terminated in (((4, EOS), True), ((4, 5), True), ((4, 5), False)):
        assert draw(tokens, terminated).checked(stopping).tokens == tokens
    for tokens, terminated, policy in (((4, 5), True, plain), ((4, EOS), False, stopping),
                                       ((EOS, 4), True, stopping)):
        with pytest.raises(ValueError):
            draw(tokens, terminated).checked(policy)


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
    """Draws submitted while another thread holds the engine's step return at
    once, and are served once the step ends."""
    # A rollout source submits a batch's draws one after another; each waiting
    # for a step made one-turn RLVR eight times slower (ledger 518).
    stepping, done = threading.Event(), threading.Event()

    def held():
        stepping.set()
        done.wait()

    def step_under_way():
        # What the engine runs between two steps holds its step as a step does.
        native_server.engine.reload(held)

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
