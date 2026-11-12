"""An episode cohort served in process (`EpisodeCollector.kv_cache`): its turns run through the serving
engine, each turn prefilling only what follows its row's last, and draw what the policy draws whole."""

import hashlib
import logging
from contextlib import contextmanager
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from model_support import float64_twin
from reference_error import as_given, assert_as_exact
from test_serving import hybrid_model

from dew.inference.tasks import TextGeneration
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.inputs import ModelInputs, pad_token_rows
from dew.nn.kv_cache import KVCache
from dew.objectives.rl import EpisodeCollector
from dew.objectives.rl.episodes import EpisodeStatus, Observation
from dew.objectives.rl.sessions import Snapshot, Task
from dew.sampling import Sampling

VOCAB, START, REPLY, EOS, PAD = 16, 1, 2, 3, 0
PROMPT, RESPONSE, TURNS, GROUPS = 32, 4, 3, 4
SAMPLED = Sampling(temperature=.7, top_k=2, eos_token_ids=EOS, pad_token_id=PAD)
GREEDY = replace(SAMPLED, temperature=0.0, top_k=None)
SEEDS = 64
"""The forked calls' draws (test_forked_calls' FORK_SEEDS): a cohort's likelihoods over drawn weights."""


def attention_model() -> CausalTransformer:
    return CausalTransformer(vocab_size=VOCAB, emb_features=16, num_layers=2, num_heads=2, num_kv_heads=2,
                             head_dim=8, mlp_features=32, max_seq_len=64, dtype="float32")


MODELS = {"attention": attention_model, "hybrid": lambda: hybrid_model(VOCAB)}


class Conversation:
    """A session that answers any action: its context the action's and a reply, `TURNS` times, then done,
    so every turn's context continues the last (what a served row reads back)."""

    def __init__(self):
        self.turns = 0

    def reset(self):
        return Observation((START,))

    def step(self, action):
        self.turns += 1
        if self.turns == TURNS:
            return Observation((), EpisodeStatus.COMPLETED, "done", reward=1.0)
        return Observation((*action.prompt, *action.tokens, REPLY))


@contextmanager
def conversations(identity):
    del identity
    yield Conversation()


def collector(model, variables, sampling, **changes) -> EpisodeCollector:
    return EpisodeCollector(TextGeneration(model, variables), conversations, lambda episode: 1.0, PROMPT,
                            RESPONSE, TURNS, sampling=sampling, **changes)


def collected(collector: EpisodeCollector, variables, key: int = 23) -> dict[str, np.ndarray]:
    """One cohort of `GROUPS` episodes of one task: every turn's context (right-padded with -1), action
    (padded with -1) and both likelihoods, by episode and turn."""
    snapshot = Snapshot(variables, 0, 0, jax.random.key(key))
    episodes = collector.episodes(snapshot, [Task("31")], GROUPS)
    shape = (len(episodes) * TURNS, RESPONSE)
    held = {"tokens": np.full(shape, -1, np.int32), "raw": np.zeros(shape, np.float64),
            "behavior": np.zeros(shape, np.float64),
            "contexts": np.full((len(episodes) * TURNS, PROMPT), -1, np.int32)}
    for index, episode in enumerate(episodes):
        for turn, transition in enumerate(episode.transitions):
            action, row = transition.action, index * TURNS + turn
            count = len(action.tokens)
            held["tokens"][row, :count] = action.tokens
            held["raw"][row, :count] = action.raw_log_probs
            held["behavior"][row, :count] = action.behavior_log_probs
            held["contexts"][row, :len(action.prompt)] = action.prompt
    return held


def scored(model, sampling: Sampling):
    """The likelihoods a policy draws a turn's action with, teacher-forced in `model` (the float64 twin, the
    truth): each action token's raw log-probability, and its log-probability under the sampling's
    temperature and top-k where the sampling draws (none where it is greedy)."""
    def score(variables, contexts, tokens) -> dict[str, np.ndarray]:
        raw = np.zeros(tokens.shape, np.float64)
        behavior = np.zeros(tokens.shape, np.float64)
        for row in range(tokens.shape[0]):
            context = contexts[row][contexts[row] >= 0]
            action = tokens[row][tokens[row] >= 0]
            if not len(action):
                continue
            sequence = jnp.asarray(np.concatenate([context, action])[None], jnp.int32)
            logits = model.apply(variables, sequence, method="logits")[0, len(context) - 1:-1]
            picked = jnp.asarray(action)[:, None]

            def chosen(scores, picked=picked):
                return np.asarray(jnp.take_along_axis(jax.nn.log_softmax(scores), picked, 1)[:, 0])

            raw[row, :len(action)] = chosen(logits)
            if sampling.temperature:
                scaled = logits / sampling.temperature
                floor = jnp.sort(scaled, axis=-1)[:, -sampling.top_k][:, None]
                behavior[row, :len(action)] = chosen(jnp.where(scaled >= floor, scaled, -jnp.inf))
        return {"raw": raw, "behavior": behavior} if sampling.temperature else {"raw": raw}
    return score


@pytest.mark.parametrize("kind", sorted(MODELS))
@pytest.mark.parametrize("sampling", [GREEDY, SAMPLED], ids=["greedy", "sampled"])
def test_a_served_cohort_draws_what_its_policy_draws_whole(kind, sampling):
    """A cohort of conversations served over a paged cache, each turn's context continuing its row's last
    (the server reading that history's pages, and an attention-and-delta-net hybrid its recurrent state,
    and prefilling the rest), draws what the policy draws prefilling every turn whole: every context and
    action token for token on every draw check 1 takes, greedy and sampled alike (the key stream folded
    by the global row, however a prompt was split), and the likelihoods it draws them with on par with
    the whole policy's, by check 1 against the actions teacher-forced in the float64 twin, over drawn
    weights."""
    model = MODELS[kind]()
    initialized = jax.jit(model.init)

    def draw(seed: int):
        return (nn.unbox(initialized(jax.random.key(seed), jnp.ones((1, 2), jnp.int32))),)

    served = collector(model, draw(0)[0], sampling, kv_cache=KVCache(page_size=2))
    whole = collector(model, draw(0)[0], sampling)
    names = ("raw", "behavior") if sampling.temperature else ("raw",)
    truth_score = scored(float64_twin(model), sampling)
    # Each arm's contexts and actions by the bytes of the weights it drew them over: the draws check 1 takes.
    discrete: dict[str, dict[bytes, tuple[np.ndarray, np.ndarray]]] = {"served": {}, "whole": {}}

    def likelihoods(arm, name):
        @as_given
        def run(variables):
            drawn = collected(arm, variables)
            key = hashlib.sha256(b"".join(np.asarray(leaf).tobytes() for leaf in jax.tree.leaves(variables)))
            discrete[name][key.digest()] = (drawn["contexts"], drawn["tokens"])
            return {held: drawn[held] for held in names}
        return run

    @as_given
    def truth(variables):
        # The actions the whole policy drew, teacher-forced in the float64 twin.
        drawn = collected(whole, variables)
        with jax.enable_x64(new_val=True):
            wide = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), variables)
            return truth_score(wide, drawn["contexts"], drawn["tokens"])

    assert_as_exact(likelihoods(served, "served"), likelihoods(whole, "whole"), draw,
                    f"served {kind} episodes", seeds=SEEDS, truth_fn=truth)
    # Every draw the served arm took, the whole one took too.
    assert discrete["served"].keys() <= discrete["whole"].keys() and len(discrete["served"]) >= SEEDS
    for key, (contexts, tokens) in discrete["served"].items():
        np.testing.assert_array_equal(contexts, discrete["whole"][key][0])
        np.testing.assert_array_equal(tokens, discrete["whole"][key][1])


def test_a_served_row_whose_context_does_not_continue_its_last_turn_is_prefilled_whole(caplog):
    """A served cohort's row whose next context does not begin with the history its last turn left (a
    context the environment rewrote) is prefilled whole, its draws the policy's whole draws, and says so: a
    warning from the collector naming the row."""
    from dew.objectives.rl.episodes import _ServedTurns

    model = attention_model()
    variables = nn.unbox(model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32)))
    turns = _ServedTurns.over(TextGeneration(model, variables), collector(
        model, variables, GREEDY, kv_cache=KVCache(page_size=2)), rows=2)
    policy = TextGeneration(model, variables)

    def laid(contexts):
        tokens, fields = pad_token_rows(contexts, pad_id=GREEDY.pad, width=PROMPT)
        return ModelInputs(jnp.asarray(tokens), {name: jnp.asarray(value) for name, value in fields.items()})

    key = jax.random.key(3)
    drawn = turns(laid([[1, 2, 3], [4, 5, 6]]), RESPONSE, key=key, sampling=GREEDY)
    tokens, lengths = np.asarray(drawn.tokens), np.asarray(drawn.lengths)
    actions = [list(tokens[row, PROMPT:PROMPT + int(lengths[row])]) for row in range(2)]
    # Row 0 continues its last turn; row 1's context is rewritten.
    second = laid([[1, 2, 3, *actions[0], 7], [8, 9, 4, 5, 6]])
    with caplog.at_level(logging.WARNING, logger="dew.objectives.rl.episodes"):
        ours = turns(second, RESPONSE, key=key, sampling=GREEDY)
    theirs = policy(second, RESPONSE, key=key, sampling=GREEDY)
    np.testing.assert_array_equal(np.asarray(ours.tokens), np.asarray(theirs.tokens))
    warned = [record.args for record in caplog.records
              if record.name == "dew.objectives.rl.episodes" and record.levelno == logging.WARNING]
    assert warned == [(1,)]


def test_a_served_cohort_after_an_update_draws_what_the_updated_policy_draws():
    """A cohort served, the weights updated (a step's, drawn), and a cohort served again: the second draws
    what the updated policy draws whole, no page or state the first cohort's weights computed read again
    (each cohort's server is its own, released with its collection)."""
    model = attention_model()
    variables = nn.unbox(model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32)))
    stepped = nn.unbox(model.init(jax.random.key(1), jnp.ones((1, 2), jnp.int32)))
    served = collector(model, variables, SAMPLED, kv_cache=KVCache(page_size=2))
    collected(served, variables)
    # The collector binds each cohort's snapshot (`EpisodeInference.bind`): the second reads the step's.
    ours = collected(served, stepped)
    theirs = collected(collector(model, stepped, SAMPLED), stepped)
    np.testing.assert_array_equal(ours["contexts"], theirs["contexts"])
    np.testing.assert_array_equal(ours["tokens"], theirs["tokens"])
