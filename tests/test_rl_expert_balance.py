"""A policy run balances its routers as an LM does.

`GRPOObjective`, `DPOObjective` and `PPOObjective` (through its policy) take
`balance=ExpertBalance(...)`, as `LMObjective` does: the policy's forward opens the
routers, the step counts the slots each router selected over the response tokens its
loss trains, and the commit applies the counts once per committed update
(`dew.objectives.routing`). A record holds the balance as the `ExpertBalance` it is,
and the flag reads it so.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
from moe_support import favour_expert_zero, uniform_gates

from dew.config import ObjectiveConfig
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
from dew.registry import parameter_type, parameters
from dew.training import Trainer

VOCAB = 16
EXPERTS = 4
PROMPT = 3
WIDTH = 9
PAD = 4
RATE = 0.05
KINDS = ("grpo", "dpo", "ppo")
CLASSES = {"grpo": GRPOObjective, "dpo": DPOObjective, "ppo": PPOObjective}
CHAIN = np.arange(WIDTH) < WIDTH - 1
"""The columns a chain fills. Its last column is padding, so a chain's last token is
an input that a padded row reads too, as an LM's shifted rows do."""
TRAINED = (np.arange(WIDTH) >= PROMPT) & CHAIN
"""A chain's response tokens."""
COUNTED = (np.arange(WIDTH) >= PROMPT) & (np.arange(WIDTH) < PROMPT + 4)
"""The four response tokens the pooling test designs. Their inputs are the router
positions it counts."""
TOKEN_EXPERTS = {1: (0, 1), 2: (2, 3), 3: (0, 2), 4: (1, 3)}
"""The designed token ids, and the two experts `routed_by_id` sends each one to."""
FIRST = (1, 1, 3, 3)
"""The first microbatch's counted ids in each row: a load of [4, 2, 2, 0] per row."""
SECOND = (4, 2, 2, 2)
"""The second microbatch's counted ids in each row: a load of [0, 1, 3, 4] per row."""


class Critic(nn.Module):
    """A per-token value over an embedding, which PPO reads as `[B, T]` values."""

    @nn.compact
    def __call__(self, tokens, *, segment_ids=None, positions=None):
        return nn.Embed(VOCAB, 1)(tokens)[..., 0]


def routed(max_seq_len, layers=2):
    return CausalTransformer(vocab_size=VOCAB, emb_features=8, num_layers=layers,
                             num_heads=2, mlp_features=16, max_seq_len=max_seq_len, qk_norm=False,
                             mixture={"experts": EXPERTS, "top_k": 2, "bias": True,
                                      "bias_in_weights": True, "score_function": "softmax"})


def favoured():
    """Two-layer routers that favour expert 0 through their biases, with uniform gates.

    Every token then chooses the same experts, so a padding test's counts depend on which
    slots are real and on nothing else.
    """
    initial = routed(WIDTH).init(jax.random.key(0), jnp.zeros((1, WIDTH - 1), jnp.int32))
    return uniform_gates(favour_expert_zero(initial, 1.0))


def routed_by_id():
    """One router that sends each token by its id alone (`TOKEN_EXPERTS`), biases at zero.

    Attention is zeroed, so no token reads another. Each designed id embeds on an axis of
    its own, and the gate scores that id's two experts above the rest, so the router's
    choice for a token is its id's pair. The RMS norm before the router keeps each axis's
    direction.
    """
    variables = routed(WIDTH, layers=1).init(jax.random.key(0), jnp.zeros((1, WIDTH - 1), jnp.int32))
    width = variables["params"]["embed_tokens"]["embedding"].shape[1]
    embedding = np.zeros((VOCAB, width), np.float32)
    gate = np.zeros((width, EXPERTS), np.float32)
    for token, experts in TOKEN_EXPERTS.items():
        embedding[token, token - 1] = 1.0
        gate[token - 1, list(experts)] = 1.0
    params = flatten_dict(variables["params"])
    (router,) = [path for path in params if path[-2:] == ("gate", "kernel")]
    params = {path: jnp.zeros_like(leaf) if "self_attn" in path else leaf
              for path, leaf in params.items()}
    params[("embed_tokens", "embedding")] = jnp.asarray(embedding)
    params[router] = jnp.asarray(gate)
    return {**variables, "params": unflatten_dict(params)}


def objective(kind, variables, layers, extra=0):
    """The `kind` objective over `variables`' decoder of `layers` layers, balancing its
    routers, with rows `extra` columns wider than `WIDTH`."""
    balance = ExpertBalance(balance_rate=RATE, aux_loss_alpha=0.2)
    model = routed(WIDTH + extra, layers)
    length = WIDTH - 1 + extra
    if kind == "grpo":
        return GRPOObjective(model, length, balance=balance, variables=variables)
    if kind == "dpo":
        return DPOObjective(model, length, balance=balance, variables=variables)
    return PPOObjective(model, length, critic=Critic(), balance=balance, variables=variables)


def step_for(variables):
    return Step(step=jnp.asarray(0), key=jax.random.key(3), ema=variables)


def policy_batch(kind, tokens, marks):
    """A batch of `kind` over `tokens` `[rows, WIDTH]`, each row one chain whose response
    tokens `marks` set: packed chains for GRPO and PPO, and for DPO a pair of the same
    chain, chosen and rejected alike."""
    rows = tokens.shape[0]
    if kind == "dpo":
        return {PAIRS_KEY: np.broadcast_to(tokens[:, None, :], (rows, 2, WIDTH)).astype(np.int32),
                PAIR_MASK_KEY: np.broadcast_to(marks, (rows, 2, WIDTH)).astype(np.float32)}
    batch = {
        IDS_KEY: tokens.astype(np.int32),
        SEGMENT_IDS_KEY: np.broadcast_to(CHAIN.astype(np.int32), (rows, WIDTH)),
        POSITIONS_KEY: np.broadcast_to(np.arange(WIDTH, dtype=np.int32), (rows, WIDTH)),
        RESPONSE_MASK_KEY: np.broadcast_to(marks, (rows, WIDTH)).astype(np.float32),
        ADVANTAGES_KEY: np.ones((rows, WIDTH), np.float32),
        BEHAVIOR_LOG_PROBS_KEY: np.full((rows, WIDTH), -2.0, np.float32),
    }
    if kind == "ppo":
        batch[OLD_VALUES_KEY] = np.zeros((rows, WIDTH), np.float32)
        batch[RETURNS_KEY] = np.ones((rows, WIDTH), np.float32)
    return batch


def drawn(kind, seed, rows):
    """Random chains of `rows` rows, each trained where `TRAINED` marks it."""
    tokens = np.random.default_rng(seed).integers(0, VOCAB, size=(rows, WIDTH))
    return policy_batch(kind, tokens, TRAINED)


def designed(kind, pattern, rows):
    """`rows` rows whose counted response tokens (`COUNTED`) are `pattern`'s ids.

    The other columns are id 1, which no count sees, since no other column is counted.
    """
    tokens = np.ones((rows, WIDTH), np.int32)
    tokens[:, COUNTED] = pattern
    return policy_batch(kind, tokens, COUNTED)


def load(pattern):
    """The slots one row of counted ids sends to each expert under `routed_by_id`."""
    counts = np.zeros(EXPERTS, np.int32)
    for token in pattern:
        counts[list(TOKEN_EXPERTS[token])] += 1
    return counts


def widened(batch, pad):
    """`batch` with `pad` padding columns after its last: no segment, response mark or target."""
    return {name: np.concatenate([value, np.zeros((*value.shape[:-1], pad), value.dtype)], axis=-1)
            for name, value in batch.items()}


def untrained(batch, kind):
    """`batch` with every response mark cleared, so its loss trains no token."""
    mark = PAIR_MASK_KEY if kind == "dpo" else RESPONSE_MASK_KEY
    return {**batch, mark: np.zeros_like(batch[mark])}


@pytest.mark.parametrize("kind", KINDS)
def test_a_policy_run_moves_each_balancing_bias_by_its_pooled_load_once_per_committed_update(kind):
    """Two microbatches make one committed update, which moves each router's bias by one
    step of the load both microbatches selected together, not of either alone.

    The designed loads differ between the microbatches. At two rows the first's loads are
    [8, 4, 4, 0] per router, the second's [0, 2, 6, 8], and pooled [8, 6, 10, 8], whose
    overloaded and idle experts give the step RATE * [0, 1, -1, 0]. DPO counts both halves
    of each pair, doubling every load, which moves nothing. The first microbatch's counts
    must sit in the window's accumulator, with the bias unmoved, until the second arrives.
    Applying one microbatch's counts alone, or a step per microbatch, fails it. Two commits
    run, the second from the bias the first left.
    """
    rows = 2 * jax.device_count()
    copies = 2 if kind == "dpo" else 1
    booked = rows * copies * load(FIRST)
    step_direction = np.float32(RATE) * np.array([0, 1, -1, 0], np.float32)
    obj = objective(kind, routed_by_id(), 1)
    train = Trainer(obj, optax.sgd(0.0), key=jax.random.PRNGKey(3), accumulation=2)
    state = train.initial_state()
    run = train.compile(state, designed(kind, FIRST, rows))
    for commit in range(2):
        reference = jax.tree.map(np.asarray, state.variables)
        partial, *_ = run(state, designed(kind, FIRST, rows))
        assert int(partial.updates) == commit
        (before,) = jax.tree.leaves(reference["moe"])
        (unmoved,) = jax.tree.leaves(partial.variables["moe"])
        np.testing.assert_array_equal(unmoved, before)
        (accumulated,) = jax.tree.leaves(partial.accumulation.effects)
        np.testing.assert_array_equal(accumulated, booked)
        state, *_ = run(partial, designed(kind, SECOND, rows))
        assert int(state.updates) == commit + 1
        (after,) = jax.tree.leaves(state.variables["moe"])
        np.testing.assert_array_equal(after, before + step_direction)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("change", ["padding", "untrained"])
def test_tokens_that_train_nothing_add_nothing_to_a_policy_runs_router_counts(kind, change):
    """Padding after every row counts exactly what the trained rows count, and a batch with
    no trained token counts nothing, so the biases move exactly as they move for the trained
    rows alone (the invariant rows 17 and 18 hold for LM)."""
    obj = objective(kind, favoured(), 2)
    variables = obj.init(jax.random.key(1))
    rows = 2 * jax.device_count()
    plain = drawn(kind, 3, rows)
    _, plain_aux = obj.scalar_loss(variables, plain, step_for(variables))
    if change == "padding":
        other, other_objective = widened(plain, PAD), objective(kind, favoured(), 2, extra=PAD)
    else:
        other, other_objective = untrained(plain, kind), obj
    _, other_aux = other_objective.scalar_loss(variables, other, step_for(variables))
    if change == "padding":
        for before, after in zip(jax.tree.leaves(plain_aux.effects), jax.tree.leaves(other_aux.effects),
                                 strict=True):
            assert jnp.issubdtype(before.dtype, jnp.integer)
            np.testing.assert_array_equal(before, after)
        expected = obj.apply_effects(variables, plain_aux.effects)["moe"]
    else:
        for count in jax.tree.leaves(other_aux.effects):
            np.testing.assert_array_equal(count, 0)
        expected = variables["moe"]
    actual = other_objective.apply_effects(variables, other_aux.effects)["moe"]
    for want, got in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("kind", KINDS)
def test_a_json_record_rebuilds_a_policy_run_that_balances_its_routers(kind):
    """A policy run's record holds its `balance` as the `ExpertBalance` it is, so the
    record's JSON rebuilds the same run: its policy balances its routers, and the rebuilt
    objective evaluates a loss. PPO is built around its critic, which a record does not
    hold."""
    balance = ExpertBalance(balance_rate=RATE, aux_loss_alpha=0.2)
    config = ObjectiveConfig(kind, {"balance": balance})
    saved = json.loads(json.dumps({"fields": config.fields, "defaults": config.defaults}))
    rebuilt = ObjectiveConfig(kind, **saved)
    assert rebuilt == config
    derived = {"critic": Critic()} if kind == "ppo" else {}
    built = rebuilt.build(model=routed(WIDTH), seq_len=WIDTH - 1, **derived)
    policy = built.actor if kind == "ppo" else built
    assert policy.balance == balance
    variables = built.init(jax.random.key(1))
    loss, aux = built.scalar_loss(variables, drawn(kind, 3, 2), step_for(variables))
    assert bool(jnp.isfinite(loss))
    assert aux.effects is not None


@pytest.mark.parametrize("kind", KINDS)
def test_a_policy_runs_balance_flag_reads_its_argument_as_an_expert_balance(kind):
    """`--objective.balance` is a flag of each policy run, typed by the annotation of the
    argument its class names (`dew.registry.parameters`): `ExpertBalance | None`, which a
    record reads back as the `ExpertBalance` it is."""
    named, _ = parameters(CLASSES[kind])
    _, owner = named["balance"]
    assert parameter_type(owner, "balance") == ExpertBalance | None
