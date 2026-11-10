"""An episode cohort served in process (`EpisodeCollector.kv_cache`): its turns run through the serving
engine, each turn prefilling only what follows its row's last, and draw what the policy draws whole."""

import logging
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from episode_support import SAMPLING, Harness, verify
from flax import linen as nn
from model_support import float64_twin
from reference_error import assert_as_exact
from test_serving import hybrid_model
from tool_policy import GROUPS, PROMPT, RESPONSE, TURNS, VOCAB

from dew.inference.tasks import TextGeneration
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.inputs import ModelInputs, pad_token_rows
from dew.nn.kv_cache import KVCache
from dew.objectives.rl import EpisodeCollector
from dew.objectives.rl.sessions import Snapshot, Task

SEEDS = 64
"""The forked calls' draws (test_forked_calls' FORK_SEEDS): a cohort's likelihoods over drawn weights."""


def attention_model() -> CausalTransformer:
    return CausalTransformer(vocab_size=VOCAB, emb_features=16, num_layers=2, num_heads=2, num_kv_heads=2,
                             head_dim=8, mlp_features=32, max_seq_len=64, dtype="float32")


MODELS = {"attention": attention_model, "hybrid": lambda: hybrid_model(VOCAB)}


def collector(model, variables, sampling, **changes) -> EpisodeCollector:
    return EpisodeCollector(TextGeneration(model, variables), Harness(), verify, PROMPT, RESPONSE, TURNS,
                            sampling=sampling, **changes)


def collected(collector: EpisodeCollector, variables, key: int) -> dict[str, np.ndarray]:
    """One cohort of `GROUPS` episodes of one task: every turn's context, action and both likelihoods, each
    episode's turns padded to `TURNS`."""
    snapshot = Snapshot(variables, 0, 0, jax.random.key(key))
    episodes = collector.episodes(snapshot, [Task("31")], GROUPS)
    shape = (len(episodes) * TURNS, RESPONSE)
    held = {"tokens": np.full(shape, -1, np.int32), "raw": np.zeros(shape, np.float32),
            "behavior": np.zeros(shape, np.float32), "contexts": np.full((len(episodes) * TURNS, 64), -1)}
    for index, episode in enumerate(episodes):
        for turn, transition in enumerate(episode.transitions):
            action, row = transition.action, index * TURNS + turn
            count = len(action.tokens)
            held["tokens"][row, :count] = action.tokens
            held["raw"][row, :count] = action.raw_log_probs
            held["behavior"][row, :count] = action.behavior_log_probs
            held["contexts"][row, :len(action.prompt)] = action.prompt
    return held


@pytest.mark.parametrize("kind", sorted(MODELS))
@pytest.mark.parametrize("temperature", [0.0, SAMPLING.temperature], ids=["greedy", "sampled"])
def test_a_served_cohort_draws_what_its_policy_draws_whole(kind, temperature):
    """A cohort of tool episodes served over a paged cache, each turn's context continuing its row's last
    (the server reading that history's pages, and an attention-and-delta-net hybrid its recurrent state,
    and prefilling the rest), draws what the policy draws prefilling every turn whole: every context and
    action token for token, greedy and sampled alike (the key stream folded by the global row, however a
    prompt was split), and both likelihoods within check 1's bound of the whole policy's (float64 truth),
    over drawn weights."""
    model = MODELS[kind]()
    sampling = replace(SAMPLING, temperature=temperature)
    initialized = jax.jit(model.init)

    def draw(seed: int):
        return (nn.unbox(initialized(jax.random.key(seed), jnp.ones((1, 2), jnp.int32))),)

    def arm(model, **changes):
        def run(variables):
            return collected(collector(model, variables, sampling, **changes), variables, key=23)
        return run

    served, whole = arm(model, kv_cache=KVCache(page_size=2)), arm(model)
    for seed in range(4):
        ours, theirs = served(*draw(seed)), whole(*draw(seed))
        np.testing.assert_array_equal(ours["contexts"], theirs["contexts"])
        np.testing.assert_array_equal(ours["tokens"], theirs["tokens"])

    def likelihoods(run):
        return lambda variables: {name: run(variables)[name] for name in ("raw", "behavior")}

    assert_as_exact(likelihoods(served), likelihoods(whole), draw, f"served {kind} episodes, {temperature}",
                    seeds=SEEDS, truth_fn=likelihoods(arm(float64_twin(model))))


def test_a_served_row_whose_context_does_not_continue_its_last_turn_is_prefilled_whole(caplog):
    """A served cohort's row whose next context does not begin with the history its last turn left (a
    context the environment rewrote) is prefilled whole, its draws the policy's whole draws, and says so: a
    warning from the collector naming the row."""
    from dew.objectives.rl.episodes import _ServedTurns

    model = attention_model()
    variables = nn.unbox(model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32)))
    sampling = replace(SAMPLING, temperature=0.0)
    turns = _ServedTurns.over(TextGeneration(model, variables), collector(
        model, variables, sampling, kv_cache=KVCache(page_size=2)), rows=2)
    policy = TextGeneration(model, variables)

    def laid(contexts):
        tokens, fields = pad_token_rows(contexts, pad_id=sampling.pad, width=PROMPT)
        return ModelInputs(jnp.asarray(tokens), {name: jnp.asarray(value) for name, value in fields.items()})

    key = jax.random.key(3)
    drawn = turns(laid([[1, 2, 3], [4, 5, 6]]), RESPONSE, key=key, sampling=sampling)
    tokens, lengths = np.asarray(drawn.tokens), np.asarray(drawn.lengths)
    actions = [list(tokens[row, PROMPT:PROMPT + int(lengths[row])]) for row in range(2)]
    # Row 0 continues its last turn; row 1's context is rewritten.
    second = laid([[1, 2, 3, *actions[0], 7], [8, 9, 4, 5, 6]])
    with caplog.at_level(logging.WARNING, logger="dew.objectives.rl.episodes"):
        ours = turns(second, RESPONSE, key=key, sampling=sampling)
    theirs = policy(second, RESPONSE, key=key, sampling=sampling)
    np.testing.assert_array_equal(np.asarray(ours.tokens), np.asarray(theirs.tokens))
    warned = [record.args for record in caplog.records
              if record.name == "dew.objectives.rl.episodes" and record.levelno == logging.WARNING]
    assert warned == [(1,)]


def test_a_served_cohort_after_an_update_draws_what_the_updated_policy_draws():
    """A cohort served, the weights updated (a step's, drawn), and a cohort served again: the second draws
    what the updated policy draws whole, no page or state the first cohort's weights computed read again
    (each cohort's server is its own, released with its collection)."""
    from model_support import moved

    model = attention_model()
    variables = nn.unbox(model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32)))
    stepped = {**variables, "params": moved(variables["params"], jax.random.key(1), 0.05)}
    served = collector(model, variables, SAMPLING, kv_cache=KVCache(page_size=2))
    collected(served, variables, key=23)
    # The collector binds each cohort's snapshot (`EpisodeInference.bind`): the second reads the step's.
    ours = collected(served, stepped, key=23)
    theirs = collected(collector(model, stepped, SAMPLING), stepped, key=23)
    np.testing.assert_array_equal(ours["contexts"], theirs["contexts"])
    np.testing.assert_array_equal(ours["tokens"], theirs["tokens"])
