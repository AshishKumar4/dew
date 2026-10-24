"""Support replay: filtered sampling becomes trainable.

A top-k/top-p sampler draws from the renormalized kept set, and the engine
reports that filtered log-probability as the behavior likelihood. The trainer
reproduces it by renormalizing over the recorded support (DeepSeek-V3.2
section 3.1, "keep sampling mask"; slime `_build_topp_keep_mask`). The
fixture is Dew's own sampler: its decode loop reports the filtered
likelihoods, and the packed GRPO scoring of the same draws with the recorded
supports must match them, softcapped models included.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import dataset

from dew.batch import FROZEN
from dew.lora import LoRA
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.protocols import OutputTable
from dew.objectives.lm import LMObjective
from dew.objectives.lm.chunked import chunked_cross_entropy, logits_support_log_probs, model_logits
from dew.objectives.rl.grpo import GRPOObjective
from dew.objectives.rl.sessions import (
    SUPPORT_COLUMNS_KEY,
    SUPPORT_KEY,
    Call,
    Session,
    Status,
    chain_lengths,
    pack,
    rows_needed,
)
from dew.sampling.text import Sampling
from dew.training import Layout, MeshSpec, Trainer

VOCAB, PROMPT, NEW = 48, 3, 6


def sampled(softcap=None, temperature=0.7, adapted=False):
    """Draws at `temperature` with top-k 12 and top-p 0.8, and each drawn
    id's support, read off the same logits the sampler filtered. `adapted`
    puts LoRA factors on the head, moved off zero, so no matrix alone is it."""
    sampling = Sampling(temperature=temperature, top_k=12, top_p=0.8)
    model = CausalTransformer(vocab_size=VOCAB, emb_features=32, num_layers=2, num_heads=2,
                              num_kv_heads=1, mlp_features=64, max_seq_len=16,
                              final_logit_softcap=softcap, tie_embeddings=False)
    variables = None
    if adapted:
        adapter = LoRA(rank=2, modules=("lm_head",)).apply(model, model.init(jax.random.key(0), jnp.zeros(
            (1, 4), jnp.int32)), key=1)
        model, variables = adapter.model, adapter.variables
    obj = LMObjective(model, PROMPT + NEW - 1, variables=variables)
    params = obj.init(jax.random.key(0))
    if softcap is not None:
        # Large enough logits that the cap bends them.
        params = jax.tree.map(lambda leaf: leaf * 3.0, params)
    if adapted:
        params = {**params, "params": jax.tree.map(
            lambda leaf: jax.random.normal(jax.random.key(2), leaf.shape, leaf.dtype), params["params"])}
    drawn = obj.policy(params, sampling)([[1, 2, 3], [4, 5, 6]], NEW, key=7).host()
    assert (drawn.lengths == NEW).all(), "the fixture wants full-length draws"
    tokens = jnp.asarray(drawn.tokens)
    hidden = obj.token_scores(params, tokens).hidden
    logits = model_logits(model, params, hidden)
    flat = logits.reshape(-1, VOCAB)
    for transform in sampling.transforms():
        flat = transform(None, flat)
    kept = np.isfinite(np.asarray(flat)).reshape(logits.shape)
    rollouts = []
    for row in range(tokens.shape[0]):
        ids = [int(token) for token in np.asarray(tokens[row])]
        support = tuple(tuple(int(token) for token in np.flatnonzero(kept[row, position - 1]))
                        for position in range(PROMPT, PROMPT + NEW))
        call = Call(
            tuple(ids[:PROMPT]),
            tuple(ids[PROMPT:]),
            tuple(float(value) for value in drawn.behavior_log_probs[row]),
            "length",
            0,
            support=support,
        )
        rollouts.append(Session("t", "g", row, 0, (call,), Status.COMPLETED, float(row)))
    return obj, params, drawn, sampling, rollouts


@pytest.mark.parametrize("softcap", [None, 30.0], ids=["plain", "softcapped"])
def test_packed_scoring_equals_the_samplers_filtered_likelihoods(softcap):
    """Temperature 0.7 applied after the cap, as the engine tempers what the
    head returns: the packed GRPO scoring gives the filtered likelihoods, and
    without the supports the tempered full-vocabulary ones, which are far off."""
    obj, params, _, sampling, rollouts = sampled(softcap)
    grpo = GRPOObjective(obj.model, obj.seq_len, sampling_temperature=sampling.temperature)
    batch = pack(rollouts, PROMPT + NEW, rows=2, support_capacity=12 * NEW)
    drawn_ids = batch["response_mask"] != 0
    scored = np.asarray(grpo.packed_log_probs(params, batch))[drawn_ids]
    np.testing.assert_allclose(scored, batch["behavior_log_probs"][drawn_ids], atol=2e-5)
    unfiltered = {key: value for key, value in batch.items() if key not in (SUPPORT_KEY, SUPPORT_COLUMNS_KEY)}
    assert np.abs(np.asarray(grpo.packed_log_probs(params, unfiltered))[drawn_ids] - scored).max() > 0.1
    assert len(batch[SUPPORT_KEY]) < len(scored) * VOCAB


def test_an_adapted_head_rescores_its_support_and_trains_its_factors_alone():
    """No matrix alone is a head with LoRA factors on it, so the packed GRPO
    scoring renormalizes the whole exact logits over each recorded support:
    the sampler's own filtered likelihoods. A Trainer step on GRPO's loss
    moves every factor and leaves the frozen base as it was."""
    obj, params, _, sampling, rollouts = sampled(adapted=True)
    grpo = GRPOObjective(obj.model, obj.seq_len, sampling_temperature=sampling.temperature,
                         variables=params)
    # As many rows as the chains need, in whole rows for every device.
    needed = rows_needed(chain_lengths(rollouts, PROMPT + NEW, truncation="mask"), PROMPT + NEW)
    rows = -(-needed // jax.device_count()) * jax.device_count()
    batch = pack(rollouts, PROMPT + NEW, rows=rows, support_capacity=12 * NEW)
    drawn_ids = batch["response_mask"] != 0
    scored = np.asarray(grpo.packed_log_probs(params, batch))[drawn_ids]
    np.testing.assert_allclose(scored, batch["behavior_log_probs"][drawn_ids], atol=2e-5)

    data = dataset(lambda: iter([batch]), batch=rows)
    trainer = Trainer(grpo, optax.sgd(0.1), key=jax.random.key(3), mesh=MeshSpec(),
                      layout=Layout(min_shard=2**30))
    initial = trainer.initial_state()
    state = trainer.fit(data, steps=1)
    leaves = {name: [jax.tree.leaves(tree.variables[name]) for tree in (initial, state)]
              for name in (FROZEN, "params")}
    for before, after in zip(*leaves[FROZEN], strict=True):
        np.testing.assert_array_equal(before, after)
    assert all(np.any(before != after) for before, after in zip(*leaves["params"], strict=True))


def test_a_heads_supported_likelihoods_are_at_least_float32_and_keep_float64():
    """Over a head no matrix gives (`logits_support_log_probs`), each support renormalizes a bfloat16
    head's logits in float32 and a float64 head's in float64 (`at_least_fp32`), as its whole-vocabulary
    likelihoods are."""
    targets = jnp.asarray([[1, 2, 3], [4, 5, 6]])
    ids = jnp.asarray([[1, 9, 2, 3, -1], [4, 5, 7, 6, -1]])
    columns = jnp.asarray([[0, 0, 1, 2, -1], [0, 1, 2, 2, -1]])
    with jax.enable_x64(new_val=True):
        logits = jax.random.normal(jax.random.key(0), (2, 3, VOCAB), jnp.float64)
        for dtype, wide in ((jnp.bfloat16, jnp.float32), (jnp.float64, jnp.float64)):
            log_probs, present = logits_support_log_probs(logits.astype(dtype), targets, ids, columns)
            assert log_probs.dtype == wide and bool(present.all()), dtype


def test_a_tempered_softcapped_cross_entropy_divides_after_the_cap():
    """`chunked_cross_entropy(temperature=)` is the log-softmax of the capped
    logits over the temperature, forward and backward, against the dense
    computation."""
    key = jax.random.split(jax.random.key(0), 3)
    hidden = 4 * jax.random.normal(key[0], (5, 8))
    head = jax.random.normal(key[1], (8, 40))
    targets = jax.random.randint(key[2], (5,), 0, 40)

    def dense(hidden, head):
        logits = 30.0 * jnp.tanh(hidden @ head / 30.0) / 0.7
        return -jnp.take_along_axis(jax.nn.log_softmax(logits), targets[:, None], 1)[:, 0]

    def chunked(hidden, head):
        return chunked_cross_entropy(
            hidden,
            OutputTable(head, vocab_major=False, softcap=30.0),
            targets,
            3,
            temperature=0.7,
            tile=(2, 16),
        )[0]

    np.testing.assert_allclose(chunked(hidden, head), dense(hidden, head), rtol=1e-5)
    for mine, theirs in zip(jax.grad(lambda *a: chunked(*a).sum(), (0, 1))(hidden, head),
                            jax.grad(lambda *a: dense(*a).sum(), (0, 1))(hidden, head), strict=True):
        np.testing.assert_allclose(mine, theirs, rtol=1e-4, atol=1e-5)


def test_the_gradient_flows_only_through_the_kept_columns():
    """Differentiating the packed scoring touches only the head columns some
    support kept (the head is untied, so no input lookup reaches it)."""
    obj, params, _, sampling, sessions = sampled()
    grpo = GRPOObjective(obj.model, obj.seq_len, sampling_temperature=sampling.temperature)
    batch = pack(sessions, PROMPT + NEW, rows=2, support_capacity=12 * NEW)
    drawn = jnp.asarray(batch["response_mask"] != 0)

    def total(params):
        return jnp.sum(jnp.where(drawn, grpo.packed_log_probs(params, batch), 0.0))

    table = np.asarray(jax.grad(total)(params)["params"]["lm_head"]["kernel"])
    touched = np.zeros(VOCAB, bool)
    touched[np.unique(batch[SUPPORT_KEY][batch[SUPPORT_KEY] >= 0])] = True
    assert np.all(np.isfinite(table))
    assert np.abs(table[:, ~touched]).max() == 0
    assert np.abs(table[:, touched]).max() > 0
