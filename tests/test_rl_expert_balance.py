"""A policy run balances its routers as an LM does.

`GRPOObjective`, `DPOObjective` and `PPOObjective` (through its policy) take `balance=ExpertBalance(...)`,
as `LMObjective` does: the policy's forward opens the routers, the step counts the slots each router
selected over the response tokens its loss trains, and the commit applies the counts once per committed
update (`dew.objectives.routing`).
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from moe_support import favour_expert_zero, uniform_gates

from dew.data.preferences import IDS_KEY as PAIRS_KEY, MASK_KEY as PAIR_MASK_KEY
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.base import Step
from dew.objectives.rl import DPOObjective, GRPOObjective, PPOObjective
from dew.objectives.rl.ppo import OLD_VALUES_KEY, RETURNS_KEY
from dew.objectives.rl.sessions import (
    ADVANTAGES_KEY,
    BEHAVIOR_LOG_PROBS_KEY,
    IDS_KEY,
    POSITIONS_KEY,
    RESPONSE_MASK_KEY,
    SEGMENT_IDS_KEY,
)
from dew.objectives.routing import ExpertBalance
from dew.training import Trainer

VOCAB = 16
EXPERTS = 4
PROMPT = 3
WIDTH = 9
PAD = 4
RATE = 0.05
KINDS = ("grpo", "dpo", "ppo")
CHAIN = np.arange(WIDTH) < WIDTH - 1
"""The columns a row's chain fills: its last column is padding, so the chain's last token is an input the
forward reads, which holds whether or not the row is padded after it (as an LM's shifted rows do)."""


class Critic(nn.Module):
    """A per-token value over an embedding, which PPO reads as `[B, T]` values."""

    @nn.compact
    def __call__(self, tokens, *, segment_ids=None, positions=None):
        return nn.Embed(VOCAB, 1)(tokens)[..., 0]


def routed(max_seq_len):
    return CausalTransformer(vocab_size=VOCAB, emb_features=8, num_layers=2, num_heads=2, mlp_features=16,
                             max_seq_len=max_seq_len, qk_norm=False,
                             mixture={"experts": EXPERTS, "top_k": 2, "bias": True, "bias_in_weights": True,
                                      "score_function": "softmax"})


def objective(kind, extra=0):
    """The `kind` objective over the routed decoder, balancing its routers, with rows `extra` columns
    wider than `WIDTH`. It starts from routers that favour expert 0 through their biases alone."""
    model = routed(WIDTH + extra)
    initial = routed(WIDTH).init(jax.random.key(0), jnp.zeros((1, WIDTH - 1), jnp.int32))
    variables = uniform_gates(favour_expert_zero(initial, 1.0))
    balance = ExpertBalance(balance_rate=RATE, aux_loss_alpha=0.2)
    length = WIDTH - 1 + extra
    if kind == "grpo":
        return GRPOObjective(model, length, balance=balance, variables=variables)
    if kind == "dpo":
        return DPOObjective(model, length, balance=balance, variables=variables)
    return PPOObjective(model, length, critic=Critic(), balance=balance, variables=variables)


def step_for(variables):
    return Step(step=jnp.asarray(0), key=jax.random.key(3), ema=variables)


def chains(seed, rows, kind):
    """Packed rows of one chain each, for GRPO and PPO: a prompt of `PROMPT` ids, then a completion, then
    the row's padding column."""
    rng = np.random.default_rng(seed)
    trained = (np.arange(WIDTH) >= PROMPT) & CHAIN
    response = np.broadcast_to(trained, (rows, WIDTH)).astype(np.float32)
    batch = {
        IDS_KEY: rng.integers(0, VOCAB, size=(rows, WIDTH)).astype(np.int32),
        SEGMENT_IDS_KEY: np.broadcast_to(CHAIN.astype(np.int32), (rows, WIDTH)),
        POSITIONS_KEY: np.broadcast_to(np.arange(WIDTH, dtype=np.int32), (rows, WIDTH)),
        RESPONSE_MASK_KEY: response,
        ADVANTAGES_KEY: rng.normal(size=(rows, WIDTH)).astype(np.float32),
        BEHAVIOR_LOG_PROBS_KEY: np.full((rows, WIDTH), -2.0, np.float32),
    }
    if kind == "ppo":
        batch[OLD_VALUES_KEY] = np.zeros((rows, WIDTH), np.float32)
        batch[RETURNS_KEY] = rng.normal(size=(rows, WIDTH)).astype(np.float32) * response
    return batch


def pairs(seed, rows):
    """Preference pairs: `[rows, 2, WIDTH]` ids, the chosen half first, and each half's completion mask,
    which leaves the row's padding column out as `chains` does."""
    rng = np.random.default_rng(seed)
    completion = np.broadcast_to((np.arange(WIDTH) >= PROMPT) & CHAIN, (rows, 2, WIDTH)).astype(np.float32)
    return {PAIRS_KEY: rng.integers(0, VOCAB, size=(rows, 2, WIDTH)).astype(np.int32),
            PAIR_MASK_KEY: completion}


def drawn(kind, seed, rows):
    return pairs(seed, rows) if kind == "dpo" else chains(seed, rows, kind)


def mask_key(kind):
    return PAIR_MASK_KEY if kind == "dpo" else RESPONSE_MASK_KEY


def widened(batch, pad):
    """`batch` with `pad` padding columns after its last: no segment, no response mark, no target."""
    return {name: np.concatenate([value, np.zeros((*value.shape[:-1], pad), value.dtype)], axis=-1)
            for name, value in batch.items()}


def untrained(batch, kind):
    """`batch` with every response mark cleared, so its loss trains no token."""
    return {**batch, mask_key(kind): np.zeros_like(batch[mask_key(kind)])}


def leaves(tree):
    return jax.tree.leaves(tree)


@pytest.mark.parametrize("kind", KINDS)
def test_a_policy_run_moves_each_balancing_bias_once_per_committed_update(kind):
    """Two microbatches make one committed update, which moves each router's bias by one step of the load
    its routers selected over both. The commit must equal that one step exactly, so applying the counts
    per microbatch, which moves the bias twice, fails it. Two commits run, the second from the bias the
    first left."""
    obj = objective(kind)
    rows = 2 * jax.device_count()
    train = Trainer(obj, optax.sgd(0.0), key=jax.random.PRNGKey(3), accumulation=2)
    state = train.initial_state()
    run = train.compile(state, drawn(kind, 0, rows))
    for commit in range(2):
        reference = jax.tree.map(np.asarray, state.variables)
        first, second = drawn(kind, 2 * commit + 1, rows), drawn(kind, 2 * commit + 2, rows)
        partial, *_ = run(state, first)
        assert int(partial.updates) == commit
        for before, after in zip(leaves(reference["moe"]), leaves(partial.variables["moe"]), strict=True):
            np.testing.assert_array_equal(before, after)
        state, *_ = run(partial, second)
        assert int(state.updates) == commit + 1
        pooled = {name: np.concatenate([first[name], second[name]]) for name in first}
        _, aux = obj.scalar_loss(reference, pooled, step_for(reference))
        expected = obj.apply_effects(reference, aux.effects)["moe"]
        for want, got in zip(leaves(expected), leaves(state.variables["moe"]), strict=True):
            np.testing.assert_array_equal(got, want)
        assert any(not np.array_equal(before, after) for before, after in
                   zip(leaves(reference["moe"]), leaves(state.variables["moe"]), strict=True))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("change", ["padding", "untrained"])
def test_tokens_that_train_nothing_add_nothing_to_a_policy_runs_router_counts(kind, change):
    """Padding after every row counts exactly what the trained rows count, and a batch with no trained
    token counts nothing, so the biases move exactly as they move for the trained rows alone (the
    invariant rows 17 and 18 hold for LM)."""
    obj = objective(kind)
    variables = obj.init(jax.random.key(1))
    rows = 2 * jax.device_count()
    plain = drawn(kind, 3, rows)
    _, plain_aux = obj.scalar_loss(variables, plain, step_for(variables))
    if change == "padding":
        other, other_objective = widened(plain, PAD), objective(kind, extra=PAD)
    else:
        other, other_objective = untrained(plain, kind), obj
    _, other_aux = other_objective.scalar_loss(variables, other, step_for(variables))
    if change == "padding":
        for before, after in zip(leaves(plain_aux.effects), leaves(other_aux.effects), strict=True):
            assert jnp.issubdtype(before.dtype, jnp.integer)
            np.testing.assert_array_equal(before, after)
        expected = obj.apply_effects(variables, plain_aux.effects)["moe"]
    else:
        for count in leaves(other_aux.effects):
            np.testing.assert_array_equal(count, 0)
        expected = variables["moe"]
    actual = other_objective.apply_effects(variables, other_aux.effects)["moe"]
    for want, got in zip(leaves(expected), leaves(actual), strict=True):
        np.testing.assert_array_equal(got, want)
