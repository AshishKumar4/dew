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
import pytest

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.lm import LMObjective
from dew.objectives.lm.chunked import chunked_cross_entropy, head_logits, support_log_probs
from dew.objectives.rl.grpo import GRPOObjective
from dew.objectives.rl.sessions import SUPPORT_COLUMNS_KEY, SUPPORT_KEY, Call, Session, Status, pack
from dew.sampling.text import Sampling

VOCAB, PROMPT, NEW = 48, 3, 6


def sampled(softcap=None, temperature=0.7):
    """Draws at `temperature` with top-k 12 and top-p 0.8, and each drawn
    id's support, read off the same logits the sampler filtered."""
    sampling = Sampling(temperature=temperature, top_k=12, top_p=0.8)
    model = CausalTransformer(vocab_size=VOCAB, emb_features=32, num_layers=2, num_heads=2,
                              num_kv_heads=1, mlp_features=64, max_seq_len=16,
                              final_logit_softcap=softcap, tie_embeddings=False)
    obj = LMObjective(model, PROMPT + NEW - 1)
    params = obj.init(jax.random.key(0))
    if softcap is not None:
        # Large enough logits that the cap bends them.
        params = jax.tree.map(lambda leaf: leaf * 3.0, params)
    drawn = obj.policy(params, sampling)([[1, 2, 3], [4, 5, 6]], NEW, seed=7).host()
    assert (drawn.lengths == NEW).all(), "the fixture wants full-length draws"
    tokens = jnp.asarray(drawn.tokens)
    hidden = obj.token_scores(params, tokens).hidden
    head = model.apply(params, params["params"], method=type(model).head_weight)
    logits = head_logits(hidden, head, softcap=softcap, precision=None)
    flat = logits.reshape(-1, VOCAB)
    for transform in sampling.transforms():
        flat = transform(None, flat)
    kept = np.isfinite(np.asarray(flat)).reshape(logits.shape)
    rollouts = []
    for row in range(tokens.shape[0]):
        ids = [int(token) for token in np.asarray(tokens[row])]
        support = tuple(tuple(int(token) for token in np.flatnonzero(kept[row, position - 1]))
                        for position in range(PROMPT, PROMPT + NEW))
        call = Call(tuple(ids[:PROMPT]), tuple(ids[PROMPT:]),
                    tuple(float(value) for value in drawn.behavior_log_probs[row]), "length", 0, support=support)
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
        return chunked_cross_entropy(hidden, head, targets, 3, softcap=30.0, temperature=0.7,
                                     tile=(2, 16))[0]

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


def test_a_target_outside_its_support_scores_minus_infinity():
    hidden = jnp.ones((1, 2, 4))
    head = jnp.eye(4, 6)
    scores, present = support_log_probs(hidden, head, jnp.asarray([[5, 1]]),
                                        jnp.asarray([[0, 1]]), jnp.asarray([[0, 0]]))
    assert present.tolist() == [[True, False]]
    assert float(scores[0, 0]) == -np.inf and float(scores[0, 1]) == 0.0


def test_pack_keeps_supports_ragged_within_their_rows():
    """One wide nucleus widens only its own row's capacity; each kept id is
    tagged with its sampled id's column in that row."""
    call = Call((1, 2), (3, 4), (-0.5, -0.1), "stop", 0, support=(tuple(range(3, 40)), (4,)))
    session = Session("t", "g", 0, 0, (call,), Status.COMPLETED, 1.0)
    batch = pack([session], 6, support_capacity=40)
    assert batch[SUPPORT_KEY].shape == batch[SUPPORT_COLUMNS_KEY].shape == (1, 40)
    assert batch[SUPPORT_KEY][0, :38].tolist() == [*range(3, 40), 4]
    assert batch[SUPPORT_COLUMNS_KEY][0, :38].tolist() == [2] * 37 + [3]
    assert (batch[SUPPORT_KEY][0, 38:] == -1).all() and (batch[SUPPORT_COLUMNS_KEY][0, 38:] == -1).all()
    with pytest.raises(ValueError, match="more than support_capacity"):
        pack([session], 6, support_capacity=10)
    with pytest.raises(ValueError, match="outside its recorded support"):
        Call((1,), (3,), (-0.5,), "stop", 0, support=((4,),))


def test_supports_need_a_fixed_capacity_so_every_batch_has_one_shape():
    """Two batches whose widest rows keep different id counts pack to one
    shape, so the jitted step is traced once; without a capacity, pack refuses."""
    def batch(kept, **options):
        call = Call((1, 2), (3,), (-0.5,), "stop", 0, support=(kept,))
        return pack([Session("t", "g", 0, 0, (call,), Status.COMPLETED, 1.0)], 4, **options)

    with pytest.raises(ValueError, match="support_capacity"):
        batch((3, 5))
    assert batch((3, 5), support_capacity=8)[SUPPORT_KEY].shape == batch((3,), support_capacity=8)[SUPPORT_KEY].shape


@pytest.mark.mesh(devices=4)
def test_supports_shard_with_their_rows():
    """The trainer splits axis 0 of every batch leaf over the data axes, so the
    supports have to lead with rows, as every other packed array does."""
    from dew.training import MeshSpec, build_mesh
    from dew.training.distributed import shard_batch

    sessions = [Session("t", "g", index, 0, (Call((1, 2), (3,), (-0.5,), "stop", 0, support=((3, 5, 7),)),),
                        Status.COMPLETED, float(index)) for index in range(7)]
    batch = pack(sessions, 3, rows=8, support_capacity=3)
    placed = shard_batch(build_mesh(MeshSpec(fsdp=4)), batch)
    assert placed[SUPPORT_KEY].shape == (8, 3)



def test_padded_support_entries_keep_the_gradient_finite():
    """Padding reads head row 0; with a large row-0 logit at a low temperature
    its exp overflows, and the masked entry must not turn the backward NaN."""
    hidden = jnp.zeros((1, 2, 4)).at[0, 0, 0].set(100.0)
    head = jnp.eye(4, 6)

    def total(hidden, head):
        scores, present = support_log_probs(hidden, head, jnp.asarray([[1, 3]]), jnp.asarray([[1, 2, -1, -1]]),
                                            jnp.asarray([[0, 0, -1, -1]]), temperature=0.5)
        return jnp.sum(jnp.where(present, scores, 0.0))

    assert np.isfinite(float(total(hidden, head)))
    for gradient in jax.grad(total, (0, 1))(hidden, head):
        assert np.all(np.isfinite(np.asarray(gradient)))
