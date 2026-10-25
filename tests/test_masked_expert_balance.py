"""A masked-diffusion run balances its routers as an LM does.

A LLaDA-MoE mixture keeps a balancing bias that its router gates on (`bias=True`,
`bias_in_weights=True`). `MaskedDiffusionObjective(balance=ExpertBalance(...))` trains that
bias as `LMObjective` does: the model opens the routers, the step counts the slots each
router selected, and the commit applies the counts once per committed update
(`dew.objectives.routing`).
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from moe_support import favour_expert_zero, uniform_gates

from dew.diffusion.discrete import MDLM
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.inputs import ModelInputs, coordinates
from dew.objectives.base import Step
from dew.objectives.diffusion.masked import MaskedDiffusionObjective
from dew.objectives.routing import ExpertBalance
from dew.training import Trainer

VOCAB = 16
MASK = VOCAB - 1
EXPERTS = 4
SEQ = 8
RATE = 0.05
STEP = Step(step=jnp.asarray(0), key=jax.random.key(3), ema=None)


def routed(max_seq_len=SEQ):
    return CausalTransformer(vocab_size=VOCAB, emb_features=8, num_layers=2, num_heads=2, mlp_features=16,
                             max_seq_len=max_seq_len, causal=False, qk_norm=False,
                             mixture={"experts": EXPERTS, "top_k": 2, "bias": True, "bias_in_weights": True,
                                      "score_function": "softmax"})


def initial(length=SEQ):
    return routed().init(jax.random.key(0), jnp.zeros((1, length), jnp.int32))


def objective(variables, length=SEQ):
    return MaskedDiffusionObjective(routed(max_seq_len=length), MDLM(mask_id=MASK)(), seq_len=length,
                                    head_chunks=1, variables=variables, ema_decay=None,
                                    balance=ExpertBalance(balance_rate=RATE))


def text(seed, rows, length=SEQ):
    ids = np.random.default_rng(seed).integers(0, MASK, size=(rows, length))
    return {"text": jnp.asarray(ids, jnp.int32)}


def leaves(tree):
    return jax.tree.leaves(tree)


def test_a_masked_run_moves_each_balancing_bias_once_per_committed_update():
    """Two microbatches make one committed update, which moves each router's bias by one step of the load
    its routers selected over both. The commit must equal that one step exactly, so applying the counts per
    microbatch, which moves the bias twice, fails it. Two commits run, the second from the bias the first
    left."""
    obj = objective(uniform_gates(favour_expert_zero(initial(), 1.0)))
    rows = 2 * jax.device_count()
    train = Trainer(obj, optax.sgd(0.0), key=jax.random.PRNGKey(3), accumulation=2)
    state = train.initial_state()
    run = train.compile(state, text(0, rows))
    for commit in range(2):
        reference = jax.tree.map(np.asarray, state.variables)
        first, second = text(2 * commit + 1, rows), text(2 * commit + 2, rows)
        partial, *_ = run(state, first)
        assert int(partial.updates) == commit
        for before, after in zip(leaves(reference["moe"]), leaves(partial.variables["moe"]), strict=True):
            np.testing.assert_array_equal(before, after)
        state, *_ = run(partial, second)
        assert int(state.updates) == commit + 1
        pooled = {"text": jnp.concatenate([first["text"], second["text"]])}
        _, aux = obj.scalar_loss(reference, pooled, STEP)
        expected = obj.apply_effects(reference, aux.effects)["moe"]
        for want, got in zip(leaves(expected), leaves(state.variables["moe"]), strict=True):
            np.testing.assert_array_equal(got, want)
        moved = [not np.array_equal(before, after) for before, after in
                 zip(leaves(reference["moe"]), leaves(state.variables["moe"]), strict=True)]
        assert any(moved)


def test_a_masked_run_narrows_the_load_its_routers_carry():
    """A router that favours one expert from the start sends most tokens to it. Each committed update moves
    the biases against the load, so the spread between the busiest and the idlest expert narrows. The
    spread is the step's own report, read on one fixed batch and key."""
    probe = text(9, 8)
    probe_step = Step(step=jnp.asarray(0), key=jax.random.key(9), ema=None, training=False)
    obj = objective(favour_expert_zero(initial(), 0.5))

    def spread(variables):
        _, aux = obj.scalar_loss(variables, probe, probe_step)
        return float(aux.metrics["moe/max_load"] - aux.metrics["moe/min_load"])

    rows = 2 * jax.device_count()
    train = Trainer(obj, optax.sgd(0.0), key=jax.random.PRNGKey(3), accumulation=2)
    state = train.initial_state()
    run = train.compile(state, text(0, rows))
    spreads = [spread(state.variables)]
    for commit in range(6):
        partial, *_ = run(state, text(2 * commit + 1, rows))
        state, *_ = run(partial, text(2 * commit + 2, rows))
        spreads.append(spread(state.variables))
    assert spreads[-1] < spreads[0], spreads


@pytest.mark.parametrize("kind", ["attention_mask", "coordinates"])
def test_padding_adds_nothing_to_a_masked_runs_router_counts_or_bias(kind):
    """Invalid slots appended to each row add nothing: the routers count only the real queries, and the
    bias moves exactly as it moves for the unpadded rows (the invariant rows 17 and 18 hold for LM)."""
    pad = 4
    rows = 2 * jax.device_count()
    ids = text(3, rows)["text"]
    extra = text(4, rows, pad)["text"]
    valid = jnp.broadcast_to(jnp.arange(SEQ + pad)[None, :] < SEQ, (rows, SEQ + pad))
    if kind == "attention_mask":
        padded = {"text": ModelInputs(jnp.concatenate([ids, extra], axis=1), {"attention_mask": valid})}
    else:
        padded = {"text": jnp.concatenate([ids, extra], axis=1),
                  "text_coordinates": coordinates(valid=valid, right_padded=True)}
    variables = uniform_gates(favour_expert_zero(initial(), 1.0))
    _, plain_aux = objective(variables).scalar_loss(variables, {"text": ids}, STEP)
    _, padded_aux = objective(variables, SEQ + pad).scalar_loss(variables, padded, STEP)
    for before, after in zip(leaves(plain_aux.effects), leaves(padded_aux.effects), strict=True):
        assert jnp.issubdtype(before.dtype, jnp.integer)
        np.testing.assert_array_equal(before, after)
    plain_moe = objective(variables).apply_effects(variables, plain_aux.effects)["moe"]
    padded_moe = objective(variables).apply_effects(variables, padded_aux.effects)["moe"]
    for before, after in zip(leaves(plain_moe), leaves(padded_moe), strict=True):
        np.testing.assert_array_equal(before, after)
