"""Sequence-parallel attention on the simulated 8-device mesh.

Under a mesh whose sequence axis is above one, the attention seam runs one
of two exchanges. The all-to-all one (Ulysses) trades every shard's rows of
the sequence for a slice of the heads, attends whole sequences, and trades
back. The all-gather one splits the queries over that axis, gathers keys and
values whole once a layer, and reorders a causal call's queries into the
striped layout that gives every shard the same causal work, with the
positions carried into the mask and the output put back in sequence order.
The proof is equality with whole sequences: the same loss, the same
gradients, the same attention output, for both.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

# Needs the eight simulated CPU devices conftest configures; the GPU lane skips it.
pytestmark = pytest.mark.mesh

import functools

import dew.training.trainer as trainer_module
from dew.nn.attention import (
    NormalAttention,
    all_to_all_moves_less,
    attention_kernel,
    causal_attention_mask,
    exchanged_heads_attention,
    gathered_keys_attention,
    rotary_freqs,
    scaled_dot_product_attention,
    stripe,
    unstripe,
)
from dew.telemetry.instrumentation import compiled_flops
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.training import Layout, MeshSpec, Trainer, build_mesh
from dew.training.distributed import shard_batch

EXCHANGES = {"all_to_all": exchanged_heads_attention, "all_gather": gathered_keys_attention}

VOCAB = 64
SEQ_LEN = 16
BATCH = 8
TINY_SHARD = 256
WHOLE = MeshSpec(fsdp=8)
SPLIT = MeshSpec(fsdp=4, sequence=2)
# Observed 2.4e-7 on the loss and 4.8e-7 on the gradients between the two
# meshes over the dense and the packed batch on CPU at highest matmul
# precision; the collectives sum in another order, so fp32 tolerance is 1e-6.
TOLERANCE = 1e-6


def tiny():
    return models.build(
        "causal_transformer", vocab_size=VOCAB, emb_features=32, num_layers=2,
        num_heads=4, num_kv_heads=2, mlp_features=64, max_seq_len=SEQ_LEN)


def dense_batch():
    rng = np.random.default_rng(0)
    return {"text": rng.integers(1, VOCAB, size=(BATCH, SEQ_LEN + 1)).astype(np.int32)}


def packed_batch():
    """Two documents a row, the second starting at a different column on
    every row, and the tail of the last row padding (segment 0)."""
    batch = dense_batch()
    rng = np.random.default_rng(1)
    segments = np.ones((BATCH, SEQ_LEN + 1), np.int32)
    positions = np.zeros((BATCH, SEQ_LEN + 1), np.int32)
    for row in range(BATCH):
        boundary = int(rng.integers(3, SEQ_LEN - 2))
        segments[row, boundary:] = 2
        positions[row, :boundary] = np.arange(boundary)
        positions[row, boundary:] = np.arange(SEQ_LEN + 1 - boundary)
    segments[-1, -3:] = 0
    batch["text_segment_ids"] = segments
    batch["text_positions"] = positions
    return batch


# --------------------------------------------------------------------------
# The striped order
# --------------------------------------------------------------------------


def test_stripe_writes_maxtexts_order_and_unstripe_undoes_it():
    """Two shards turn [0..7] into [0, 1, 6, 7, 2, 3, 4, 5]
    (maxtext_utils.reorder_sequence's own example), and the inverse holds
    for every shard count the eight-device mesh can hold."""
    np.testing.assert_array_equal(
        stripe(jnp.arange(8), 2, axis=0), [0, 1, 6, 7, 2, 3, 4, 5])
    rows = jnp.arange(2 * 32 * 3).reshape(2, 32, 3)
    for shards in (1, 2, 4, 8):
        np.testing.assert_array_equal(unstripe(stripe(rows, shards), shards), rows)


@pytest.mark.parametrize("shards", [2, 4, 8])
def test_every_shard_holds_the_same_causal_work(shards):
    """Row p of a causal sequence attends p + 1 keys; the striped order gives
    each shard's rows the same total."""
    length = 64
    positions = np.asarray(stripe(jnp.arange(length), shards, axis=0)).reshape(shards, -1)
    work = (positions + 1).sum(axis=1)
    assert np.all(work == work[0]), work
    contiguous = (np.arange(length).reshape(shards, -1) + 1).sum(axis=1)
    assert contiguous.max() > contiguous.min()


def test_a_length_the_shards_cannot_pair_is_refused():
    with pytest.raises(ValueError, match="multiple of 4, got 15"):
        stripe(jnp.zeros((2, 15, 4, 8)), 2)


# --------------------------------------------------------------------------
# The attention seam
# --------------------------------------------------------------------------


def heads(key, kv_heads):
    q_key, k_key, v_key = jax.random.split(key, 3)
    query = jax.random.normal(q_key, (BATCH, SEQ_LEN, 4, 8), jnp.float32)
    key_ = jax.random.normal(k_key, (BATCH, SEQ_LEN, kv_heads, 8), jnp.float32)
    value = jax.random.normal(v_key, (BATCH, SEQ_LEN, kv_heads, 8), jnp.float32)
    return query, key_, value


def packed_mask(segments):
    inside = ((segments[:, :, None] == segments[:, None, :]) & (segments[:, :, None] != 0))
    return jnp.logical_and(inside[:, None], causal_attention_mask(jnp.arange(SEQ_LEN), SEQ_LEN))


CALLS = {
    "causal": dict(causal=True),
    "window": dict(causal=True, sliding_window=5),
    "packed": dict(mask=packed_mask(jnp.asarray(packed_batch()["text_segment_ids"][:, :-1]))),
    # Query-broadcast rows must survive the reorder without being expanded.
    "broadcast_mask": dict(
        causal=True, mask=(jnp.arange(SEQ_LEN) < SEQ_LEN // 2)[None, None, None, :]),
    "broadcast_bias": dict(
        causal=True, bias=jnp.linspace(-4.0, 4.0, SEQ_LEN)[None, None, None, :]),
    "bias": dict(causal=True,
                 bias=jax.random.normal(jax.random.key(3), (1, 4, SEQ_LEN, SEQ_LEN))),
    "full": dict(),
    # One learned logit per query head, split with the heads by the exchange.
    "sinks": dict(causal=True, sinks=jax.random.normal(jax.random.key(4), (4,))),
}


def through(exchange, query, key, value, *, implementation="reference", **call):
    """One attention call through the named exchange, over the mesh's
    sequence axis, whichever the per-call choice would have taken."""
    kernel = functools.partial(attention_kernel, implementation=implementation)
    return EXCHANGES[exchange](
        kernel, query, key, value, jax.sharding.get_abstract_mesh().shape["sequence"],
        causal=call.get("causal", False), sliding_window=call.get("sliding_window"),
        mask=call.get("mask"), bias=call.get("bias"), sinks=call.get("sinks"))


@pytest.mark.parametrize("exchange", sorted(EXCHANGES))
@pytest.mark.parametrize("implementation", ["reference", "xla"])
@pytest.mark.parametrize("name", sorted(CALLS))
def test_the_seam_agrees_with_whole_sequences(name, implementation, exchange):
    """The same output, row for row, whether the sequence axis is one or two:
    the mask follows the positions through either exchange, and the output
    comes back in sequence order. Observed 4.8e-7 at most on CPU."""
    query, key, value = heads(jax.random.key(0), kv_heads=2)
    call = dict(CALLS[name], implementation=implementation)
    whole = scaled_dot_product_attention(query, key, value, **call)
    with jax.set_mesh(build_mesh(SPLIT)):
        split = jax.jit(lambda q, k, v: through(exchange, q, k, v, **call))(query, key, value)
    np.testing.assert_allclose(np.asarray(split), np.asarray(whole), atol=TOLERANCE, rtol=0)


# Heads split over tensor then sequence (2x2), and four sequence shards over
# two key heads, which the exchange repeats to four and no further.
HEAD_SPLITS = [MeshSpec(fsdp=2, tensor=2, sequence=2), MeshSpec(fsdp=2, sequence=4)]


@pytest.mark.parametrize("spec", HEAD_SPLITS, ids=["tensor2_sequence2", "sequence4"])
@pytest.mark.parametrize("name", ["causal", "packed", "bias", "sinks"])
def test_the_exchange_agrees_forward_and_backward_where_heads_split_further(spec, name):
    """The all-to-all's gradients are its transposes: the query, key and
    value cotangents of a whole-sequence call come back in the split one,
    with the heads cut over tensor and sequence at once and grouped keys
    repeated to the split. Observed 5.7e-6 on gradients of order 10."""
    query, key, value = heads(jax.random.key(0), kv_heads=2)
    call = CALLS[name]

    def loss(attend):
        return lambda q, k, v: jnp.sum(attend(q, k, v) ** 2)

    whole = jax.grad(loss(lambda q, k, v: scaled_dot_product_attention(q, k, v, **call)),
                     argnums=(0, 1, 2))(query, key, value)
    with jax.set_mesh(build_mesh(spec)):
        split = jax.jit(jax.grad(loss(lambda q, k, v: through("all_to_all", q, k, v, **call)),
                                 argnums=(0, 1, 2)))(query, key, value)
    for got, want in zip(split, whole, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=2e-5, rtol=1e-6)


def placed_on(spec, query, key, value, sequence=True):
    mesh = build_mesh(spec)
    rows = NamedSharding(mesh, P(("data", "expert", "fsdp"), "sequence" if sequence else None))
    return mesh, jax.device_put((query, key, value), rows)


def exchanges_heads(spec, query, key, value, **call) -> bool:
    """Whether the program one attention call lowers to holds the
    all-to-all's shard_map, read before GSPMD adds collectives of its own."""
    mesh, operands = placed_on(spec, query, key, value, sequence=False)
    with jax.set_mesh(mesh):
        attend = jax.jit(lambda q, k, v: scaled_dot_product_attention(q, k, v, **call))
        return "all_to_all" in attend.lower(*operands).as_text()


def test_the_exchange_moves_heads_and_never_gathers_a_key():
    """Ulysses's point: a causal call trades rows for heads with all-to-alls,
    and no device ever assembles the whole key or value."""
    mesh, operands = placed_on(SPLIT, *heads(jax.random.key(0), kv_heads=4))
    with jax.set_mesh(mesh):
        attend = jax.jit(lambda q, k, v: scaled_dot_product_attention(q, k, v, causal=True))
        text = attend.lower(*operands).compile().as_text()
    assert "all-to-all" in text and "all-gather" not in text


def test_the_byte_count_picks_the_exchange_the_arithmetic_favours():
    """H=32, K=8, n=2: the all-to-all sends 20 units either way, the gather
    8 unmasked and 24 causal. One tensor shard of two halves every side."""
    assert not all_to_all_moves_less(32, 8, 1, 2, reordered=False)
    assert all_to_all_moves_less(32, 8, 1, 2, reordered=True)
    # Unmasked multi-head attention sends 4H / n against the gather's 2H:
    # a tie at two shards, which the gather takes, and the exchange past it.
    assert not all_to_all_moves_less(8, 8, 1, 2, reordered=False)
    assert all_to_all_moves_less(8, 8, 1, 4, reordered=False)
    # A tie goes to the gather: H=4, K=1, n=4 sends 2(4 + 4)/4 = 4 against
    # 2 + 2 * 4 / 4 = 4.
    assert not all_to_all_moves_less(4, 1, 1, 4, reordered=True)
    assert all_to_all_moves_less(32, 8, 2, 2, reordered=True) == all_to_all_moves_less(
        16, 4, 1, 2, reordered=True)


@pytest.mark.parametrize("case, spec, shape, call, exchanged", [
    ("causal", SPLIT, (BATCH, SEQ_LEN, 4, 2), dict(causal=True), True),
    ("unmasked_grouped", SPLIT, (BATCH, SEQ_LEN, 8, 1), dict(), False),
    ("heads_the_split_cannot_divide", MeshSpec(tensor=2, sequence=4),
     (BATCH, SEQ_LEN, 4, 2), dict(causal=True), False),
    ("odd_joint_length", SPLIT, (BATCH, 15, 4, 4), dict(), False),
], ids=lambda value: value if isinstance(value, str) else "")
def test_every_call_takes_the_exchange_its_shape_admits(case, spec, shape, call, exchanged):
    """The choice is per call and falls to the gather for any shape the
    all-to-all cannot split, so every shape the gather took still runs, and
    runs exactly."""
    batch, length, heads_, kv_heads = shape
    keys = jax.random.split(jax.random.key(0), 3)
    query = jax.random.normal(keys[0], (batch, length, heads_, 8))
    key, value = (jax.random.normal(k, (batch, length, kv_heads, 8)) for k in keys[1:])
    assert exchanges_heads(spec, query, key, value, **call) is exchanged
    whole = scaled_dot_product_attention(query, key, value, **call)
    with jax.set_mesh(build_mesh(spec)):
        split = jax.jit(lambda q, k, v: scaled_dot_product_attention(q, k, v, **call))(
            query, key, value)
    np.testing.assert_allclose(np.asarray(split), np.asarray(whole), atol=TOLERANCE, rtol=0)


def test_splash_runs_inside_the_exchange():
    """The all-to-all hands the kernel whole sequences, so splash takes a
    causal call under a split sequence with its causal descriptor. The
    all-gather exchange can only hand it a striped traced mask, which splash
    has no form for. Run under pallas's interpreter off a TPU."""
    q_key, k_key, v_key = jax.random.split(jax.random.key(0), 3)
    shape = (2, 256, 4, 8)
    query, key, value = (jax.random.normal(k, shape, jnp.float32) for k in (q_key, k_key, v_key))
    whole = scaled_dot_product_attention(query, key, value, causal=True)
    with jax.set_mesh(build_mesh(MeshSpec(fsdp=2, sequence=2))):
        split = jax.jit(lambda q, k, v: through(
            "all_to_all", q, k, v, causal=True, implementation="tpu"))(query, key, value)
    np.testing.assert_allclose(np.asarray(split), np.asarray(whole), atol=TOLERANCE, rtol=0)


def test_a_shape_the_exchange_cannot_split_is_refused_by_name():
    query, key, value = heads(jax.random.key(0), kv_heads=2)
    with jax.set_mesh(build_mesh(MeshSpec(tensor=2, sequence=4))):
        with pytest.raises(ValueError, match="gathered_keys_attention"):
            jax.jit(lambda q, k, v: through("all_to_all", q, k, v, causal=True))(
                query, key, value)


def test_rotary_positions_and_the_causal_mask_survive_the_exchange():
    """A full attention module: rotary angles from the row's position, then
    the causal mask. Both were applied in sequence order, and the exchanged
    call has to match them exactly (observed 4.8e-7)."""
    module = NormalAttention(query_dim=32, heads=4, dim_head=8, causal=True)
    x = jax.random.normal(jax.random.key(1), (BATCH, SEQ_LEN, 32), jnp.float32)
    freqs = rotary_freqs(jnp.arange(SEQ_LEN), 8, 10000.0)
    variables = module.init(jax.random.key(2), x, freqs_cis=freqs)
    whole = module.apply(variables, x, freqs_cis=freqs)
    with jax.set_mesh(build_mesh(SPLIT)):
        split = jax.jit(lambda v, x: module.apply(v, x, freqs_cis=freqs))(variables, x)
    np.testing.assert_allclose(np.asarray(split), np.asarray(whole), atol=TOLERANCE, rtol=0)


def test_decoding_is_refused_under_a_sequence_axis():
    model = tiny()
    tokens = jnp.ones((1, SEQ_LEN), jnp.int32)
    variables = model.init(jax.random.key(0), tokens)
    with jax.set_mesh(build_mesh(SPLIT)):
        with pytest.raises(ValueError, match="sequence axis of 2"):
            jax.jit(lambda v, t: model.apply(v, t, decode=True, mutable=["cache"]))(
                variables, tokens)


# --------------------------------------------------------------------------
# The model, through the trainer
# --------------------------------------------------------------------------


def one_step(spec, batch, monkeypatch=None, tolerance=0.02):
    """The loss of one step and the parameters after it, on `spec`; with
    sgd(1.0) the parameters move by exactly the gradient. With
    `monkeypatch`, also the program the trainer compiles for its own step,
    as lowered, before GSPMD adds collectives of its own."""
    lowered = []
    if monkeypatch is not None:
        def recording(jitted, *arguments):
            program = jitted.lower(*arguments)
            lowered.append(program.as_text())
            return compiled_flops(program.compile())
        monkeypatch.setattr(trainer_module, "step_flops", recording)
    trainer = Trainer(LMObjective(tiny(), SEQ_LEN), optax.sgd(1.0), key=jax.random.key(0),
                      mesh=spec, layout=Layout(min_shard=TINY_SHARD, tolerance=tolerance))
    state, _, _ = trainer.place()
    placed = shard_batch(trainer.device_mesh, batch)
    state, loss, _, _, _ = trainer.compile(state, placed)(state, placed)
    return float(loss), jax.tree.map(np.asarray, state.params["params"]), "".join(lowered)


def assert_same_step(whole, split):
    whole_loss, whole_params, _ = whole
    split_loss, split_params, _ = split
    assert abs(whole_loss - split_loss) < TOLERANCE, (whole_loss, split_loss)
    differences = jax.tree.map(
        lambda a, b: float(np.max(np.abs(a - b))), whole_params, split_params)
    assert max(jax.tree.leaves(differences)) < TOLERANCE, differences


# The tiny decoder's causal layers, 4 query heads over 2 key heads: over two
# sequence shards the all-to-all sends 6 units to the gather's 8, and over
# tensor=2 by sequence=4 the heads do not divide, so the gather runs. That
# mesh leaves the 32-wide output projections whole (fsdp is 1), which the
# layout tolerance has to allow.
TRAINED_EXCHANGES = {
    "all_to_all": (MeshSpec(fsdp=4, sequence=2), 0.02),
    "all_gather": (MeshSpec(tensor=2, sequence=4), 0.2),
}


@pytest.mark.parametrize("exchange", sorted(TRAINED_EXCHANGES))
@pytest.mark.parametrize("make_batch", [dense_batch, packed_batch])
def test_loss_and_gradients_agree_with_whole_sequences(make_batch, exchange, monkeypatch):
    """Loss and every gradient leaf equal between fsdp=8 and a split
    sequence, on a dense batch and on a packed one with segment ids and
    positions. The trainer's compiled step shows which exchange ran."""
    batch = make_batch()
    spec, tolerance = TRAINED_EXCHANGES[exchange]
    split = one_step(spec, batch, monkeypatch, tolerance)
    assert ("all_to_all" in split[2]) is (exchange == "all_to_all")
    assert_same_step(one_step(WHOLE, batch), split)


def test_the_exchange_runs_inside_the_pipeline_stages(monkeypatch):
    """The pipeline vmaps its stages over the stage axis, and the exchange's
    shard_map runs inside that vmap: fsdp=2, stage=2, sequence=2 trains the
    step fsdp=8 does, loss and gradients."""
    batch = packed_batch()
    split = one_step(MeshSpec(fsdp=2, stage=2, sequence=2), batch, monkeypatch)
    assert "all_to_all" in split[2]
    assert_same_step(one_step(WHOLE, batch), split)
