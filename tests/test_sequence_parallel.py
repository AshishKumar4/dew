"""Sequence-parallel attention and Mamba-2 on the simulated 8-device mesh,
and cuDNN's attention inside both exchanges on two GPUs.

Under a mesh whose sequence axis is above one, the attention seam runs one
of two exchanges. The all-to-all one (Ulysses) trades every shard's rows of
the sequence for a slice of the heads, attends whole sequences, and trades
back. The all-gather one splits the queries over that axis, gathers keys and
values whole once a layer, and reorders a causal call's queries into the
striped layout that gives every shard the same causal work, with the
positions carried into the mask and the output put back in sequence order.
The proof is equality with whole sequences: the same loss, the same
gradients, the same attention output, for both.

Mamba-2 splits its conv and its scan over the same axis: each shard reads
the previous shard's conv tail and starts its scan from the state the
earlier shards leave, and the proof is the same, the layer's output and
gradients and a hybrid model's training step against whole sequences.
"""

import functools
import re

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from dew.nn.attention import (
    NormalAttention,
    all_to_all_moves_less,
    attention_kernel,
    causal_attention_mask,
    exchanged_heads_attention,
    gathered_keys_attention,
    local_attention,
    scaled_dot_product_attention,
    stripe,
    unstripe,
)
from dew.nn.mixers.mamba2 import Mamba2
from dew.nn.rope import rotary_freqs
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
    # Each row's keys end at its own length; the rows split with the batch.
    "key_lengths": dict(key_value_seq_lengths=jnp.asarray([16, 9, 3, 1, 12, 16, 5, 7], jnp.int32)),
    "causal_key_lengths": dict(causal=True, key_value_seq_lengths=jnp.asarray(
        [16, 9, 3, 1, 12, 16, 5, 7], jnp.int32)),
}


def through(exchange, query, key, value, *, implementation="reference", **call):
    """One attention call through the named exchange, over the mesh's
    sequence axis, whichever the per-call choice would have taken."""
    kernel = functools.partial(attention_kernel, implementation=implementation)
    return EXCHANGES[exchange](
        kernel, query, key, value, jax.sharding.get_abstract_mesh().shape["sequence"],
        causal=call.get("causal", False), sliding_window=call.get("sliding_window"),
        mask=call.get("mask"), bias=call.get("bias"), sinks=call.get("sinks"),
        key_value_seq_lengths=call.get("key_value_seq_lengths"))


@pytest.mark.mesh
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


@pytest.mark.mesh
@pytest.mark.parametrize("spec", HEAD_SPLITS, ids=["tensor2_sequence2", "sequence4"])
@pytest.mark.parametrize("name", ["causal", "packed", "bias", "sinks", "causal_key_lengths"])
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


@pytest.mark.mesh
def test_the_exchange_moves_heads_and_never_gathers_a_key():
    """Ulysses's point: a causal call trades rows for heads with all-to-alls,
    and no device ever assembles the whole key or value."""
    mesh, operands = placed_on(SPLIT, *heads(jax.random.key(0), kv_heads=4))
    with jax.set_mesh(mesh):
        attend = jax.jit(lambda q, k, v: scaled_dot_product_attention(q, k, v, causal=True))
        text = attend.lower(*operands).compile().as_text()
    assert "all-to-all" in text and "all-gather" not in text


def test_the_byte_count_picks_the_exchange_for_a_call_with_no_mask():
    """H=32, K=8, n=2: the all-to-all sends 40 units, the gather 16. One
    tensor shard of two halves every side."""
    assert not all_to_all_moves_less(32, 8, 1, 2)
    # Multi-head attention sends 4H / n against the gather's 2H: a tie at
    # two shards, which the gather takes, and the exchange past it.
    assert not all_to_all_moves_less(8, 8, 1, 2)
    assert all_to_all_moves_less(8, 8, 1, 4)
    assert all_to_all_moves_less(32, 8, 2, 2) == all_to_all_moves_less(16, 4, 1, 2)


@pytest.mark.mesh
@pytest.mark.parametrize("case, spec, shape, call, exchanged", [
    ("causal", SPLIT, (BATCH, SEQ_LEN, 4, 2), dict(causal=True), True),
    # H=4, K=1, n=4: 2(4 + 4)/4 = 4 units against the gather's 2 + 2 * 4/4 = 4.
    # A masked call takes the all-to-all even at a tie in bytes, whose kernel
    # skips the masked blocks.
    ("causal_one_key_head", MeshSpec(fsdp=2, sequence=4), (BATCH, SEQ_LEN, 4, 1),
     dict(causal=True), True),
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


@pytest.mark.mesh
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


@pytest.mark.skipif(jax.default_backend() != "gpu" or jax.device_count() < 2,
                    reason="cuDNN's fused attention, split over two GPUs")
@pytest.mark.parametrize("exchange", sorted(EXCHANGES))
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_cudnn_runs_inside_either_exchange(exchange, causal, without_deterministic_ops):
    """cuDNN's partitioning rule refuses queries split unlike their keys
    (`_check_qkv_bias_mask_spec` in jax/_src/cudnn/fused_attention_stablehlo.py),
    which is what the gather hands a kernel it leaves to GSPMD: every call
    that took the gather failed to compile on a GPU. Both exchanges run the
    kernel on local arrays, forward and backward. bf16 has no fixed bound,
    so each is measured against fp32 attention on one device and may land
    at most twice as far from it as the same cuDNN call on one GPU."""
    keys = jax.random.split(jax.random.key(0), 3)
    query = jax.random.normal(keys[0], (2, 256, 4, 64), jnp.bfloat16)
    key, value = (jax.random.normal(k, (2, 256, 2, 64), jnp.bfloat16) for k in keys[1:])

    def outputs(attend):
        def loss(q, k, v):
            return jnp.sum(attend(q, k, v).astype(jnp.float32) ** 2)
        return jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))

    exact = outputs(lambda q, k, v: scaled_dot_product_attention(
        q, k, v, causal=causal, implementation="xla"))(
            *(x.astype(jnp.float32) for x in (query, key, value)))
    whole = outputs(lambda q, k, v: scaled_dot_product_attention(
        q, k, v, causal=causal, implementation="cudnn"))(query, key, value)
    with jax.set_mesh(build_mesh(MeshSpec(sequence=2), jax.devices()[:2])):
        split = outputs(lambda q, k, v: through(
            exchange, q, k, v, causal=causal, implementation="cudnn"))(query, key, value)

    def distance(got):
        return [float(np.max(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)))
                      / np.max(np.abs(np.asarray(b, np.float32))))
                for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(exact), strict=True)]

    for split_distance, whole_distance in zip(distance(split), distance(whole), strict=True):
        assert split_distance <= 2 * whole_distance, (split_distance, whole_distance)


@pytest.mark.mesh
def test_a_shape_the_exchange_cannot_split_is_refused_by_name():
    query, key, value = heads(jax.random.key(0), kv_heads=2)
    with jax.set_mesh(build_mesh(MeshSpec(tensor=2, sequence=4))):
        with pytest.raises(ValueError, match="gathered_keys_attention"):
            jax.jit(lambda q, k, v: through("all_to_all", q, k, v, causal=True))(
                query, key, value)


LOCAL_LENGTH = 32


def local_inputs():
    """Two packed rows of 32 tokens, documents starting at a different column
    of each, the last three tokens of the second row padding, and positions
    restarting at every document."""
    segments = np.ones((2, LOCAL_LENGTH), np.int32)
    positions = np.zeros((2, LOCAL_LENGTH), np.int32)
    for row, cut in enumerate((7, 18)):
        segments[row, cut:] = 2
        positions[row, :cut] = np.arange(cut)
        positions[row, cut:] = np.arange(LOCAL_LENGTH - cut)
    segments[1, -3:] = 0
    return jnp.asarray(segments), jnp.asarray(positions)


LOCAL_CALLS = {
    "window": dict(window=5),
    "window_packed": dict(window=5, packed=True),
    "window_valid_sinks": dict(window=5, valid=True, sinks=True),
    # Chunks of 4 start at every shard's first row; chunks of 6 straddle them.
    "chunk_aligned": dict(chunk=4),
    "chunk_straddling": dict(chunk=6),
    "chunk_positions": dict(chunk=6, positions=True, packed=True),
}


def local_call(q, k, v, window=None, chunk=None, packed=False, positions=False, valid=False,
               sinks=False):
    segments, places = local_inputs()
    return local_attention(
        q, k, v, window=window, chunk=chunk, implementation="xla",
        segment_ids=segments if packed else None, positions=places if positions else None,
        valid=(segments != 0) if valid else None,
        sinks=jnp.linspace(-1.0, 1.0, q.shape[2]) if sinks else None)


def local_outputs(call, heads: int, kv_heads: int, mesh=None):
    """`local_call`'s output and its queries', keys' and values' gradients
    under a random cotangent, on one device or split over `mesh`'s sequence
    axis, with the compiled program's text. A packed call's padding rows
    attend nothing, so what a kernel puts there is its own: the cotangent
    leaves them out, and so does the comparison."""
    keys = jax.random.split(jax.random.key(0), 4)
    query = jax.random.normal(keys[0], (2, LOCAL_LENGTH, heads, 8))
    key, value = (jax.random.normal(k, (2, LOCAL_LENGTH, kv_heads, 8)) for k in keys[1:3])
    live = np.ones((2, LOCAL_LENGTH, 1, 1), bool)
    if call.get("packed"):
        live = np.asarray(local_inputs()[0] != 0)[:, :, None, None]
    cotangent = jax.random.normal(keys[3], query.shape) * live

    def attend(q, k, v, cotangent):
        out, pullback = jax.vjp(lambda q, k, v: local_call(q, k, v, **call), q, k, v)
        return out * live, pullback(cotangent)

    operands = (query, key, value, cotangent)
    if mesh is None:
        return jax.jit(attend)(*operands), ""
    operands = jax.device_put(operands, NamedSharding(mesh, P(None, "sequence")))
    with jax.set_mesh(mesh):
        program = jax.jit(attend).lower(*operands).compile()
        return program(*operands), program.as_text()


def assert_close_by_leaf(got, want):
    """Each leaf within 2e-6 of its largest value: the two runs add the same
    at most 2 * 6 terms (a window of 5 over two blocks) in another order, a
    dozen fp32 ulps of the leaf's scale."""
    for have, expected in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        expected = np.asarray(expected)
        np.testing.assert_allclose(np.asarray(have), expected, rtol=0,
                                   atol=2e-6 * np.max(np.abs(expected)))


@pytest.mark.mesh
@pytest.mark.parametrize("spec", [MeshSpec(sequence=2), MeshSpec(fsdp=2, sequence=4),
                                  MeshSpec(tensor=2, sequence=4)],
                         ids=["sequence2", "fsdp2_sequence4", "tensor2_sequence4"])
@pytest.mark.parametrize("name", sorted(LOCAL_CALLS))
def test_local_attention_splits_over_the_sequence_with_one_halo(name, spec):
    """A window or a chunk no wider than a shard's slice attends each shard's
    own rows, the first of them reading the previous shard's last rows
    through one collective-permute; no device gathers the sequence (GSPMD
    gathered every query and banded key before). The output and the
    gradients of the queries, keys and values match whole sequences."""
    call = LOCAL_CALLS[name]
    whole, _ = local_outputs(call, 4, 2)
    split, text = local_outputs(call, 4, 2, build_mesh(spec))
    # An HLO collective names the axes it runs over after the mesh: `{'sequence'}`.
    over_sequence = [line for line in text.splitlines()
                     if " all-gather(" in line and re.search(r"\] \{[^}]*'sequence'", line)]
    assert not over_sequence, over_sequence[:3]
    assert ("collective-permute" in text) is (name != "chunk_aligned")
    assert_close_by_leaf(split, whole)


@pytest.mark.mesh
def test_a_window_wider_than_a_shard_takes_the_exchange():
    """Eight shards of 4 rows cannot hold a window of 5 in one neighbour, so
    the call runs whole through the sequence exchange, and still matches."""
    call = dict(window=5, packed=True)
    whole, _ = local_outputs(call, 8, 8)
    split, text = local_outputs(call, 8, 8, build_mesh(MeshSpec(sequence=8)))
    assert "all-to-all" in text
    assert_close_by_leaf(split, whole)


@pytest.mark.mesh
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


@pytest.mark.mesh
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


def one_step(spec, batch, tolerance=0.02, model=tiny):
    """The loss of one step and the parameters after it, on `spec`; with
    sgd(1.0) the parameters move by exactly the gradient. Also the program
    the trainer compiles for its own step, as lowered, before GSPMD adds
    collectives of its own."""
    trainer = Trainer(LMObjective(model(), SEQ_LEN), optax.sgd(1.0), key=jax.random.key(0),
                      mesh=spec, layout=Layout(min_shard=TINY_SHARD, tolerance=tolerance))
    state, _, _ = trainer.place()
    placed = shard_batch(trainer.device_mesh, batch)
    state, loss, _, _, _ = trainer.compile(state, placed)(state, placed)
    assert trainer.program is not None
    return float(loss), jax.tree.map(np.asarray, state.params["params"]), trainer.program.as_text()


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


@pytest.mark.mesh
@pytest.mark.parametrize("exchange", sorted(TRAINED_EXCHANGES))
@pytest.mark.parametrize("make_batch", [dense_batch, packed_batch])
def test_loss_and_gradients_agree_with_whole_sequences(make_batch, exchange):
    """Loss and every gradient leaf equal between fsdp=8 and a split
    sequence, on a dense batch and on a packed one with segment ids and
    positions. The trainer's compiled step shows which exchange ran."""
    batch = make_batch()
    spec, tolerance = TRAINED_EXCHANGES[exchange]
    split = one_step(spec, batch, tolerance)
    assert ("all_to_all" in split[2]) is (exchange == "all_to_all")
    assert_same_step(one_step(WHOLE, batch), split)


@pytest.mark.mesh
def test_the_exchange_runs_inside_the_pipeline_stages():
    """The pipeline vmaps its stages over the stage axis, and the exchange's
    shard_map runs inside that vmap: fsdp=2, stage=2, sequence=2 trains the
    step fsdp=8 does, loss and gradients."""
    batch = packed_batch()
    split = one_step(MeshSpec(fsdp=2, stage=2, sequence=2), batch)
    assert "all_to_all" in split[2]
    assert_same_step(one_step(WHOLE, batch), split)


# --------------------------------------------------------------------------
# Mamba-2: the conv and the scan split over the sequence axis
# --------------------------------------------------------------------------

MAMBA_LENGTH = 64
# One document ends on a shard boundary at eight shards (8), one inside a
# shard (30), a two-token document straddles the boundary at 16, shorter
# than the conv's three-token history, and the last row ends in padding.
MAMBA_CUTS = ((8, 30), (15, 17, 60))


def mamba_segments():
    segments = np.ones((2, MAMBA_LENGTH), np.int32)
    for row, cuts in enumerate(MAMBA_CUTS):
        for index, cut in enumerate(cuts):
            segments[row, cut:] = index + 2
    segments[1, 60:] = 0
    return jnp.asarray(segments)


MAMBA_SPLITS = {
    "sequence8": MeshSpec(sequence=8),
    "fsdp2_sequence4": MeshSpec(fsdp=2, sequence=4),
    "tensor2_sequence4": MeshSpec(tensor=2, sequence=4),
}


def mamba_layer(dtype):
    return Mamba2(emb_features=16, num_heads=4, head_dim=8, state_size=6, n_groups=2,
                  chunk_size=4, dtype=dtype)


def leafwise_largest(want, got) -> dict[str, float]:
    """Per leaf, by path, the largest difference as a fraction of the leaf's
    largest value."""
    def relative(a, b):
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        return float(np.max(np.abs(a - b)) / np.max(np.abs(a)))
    differences = jax.tree.map(relative, want, got)
    return {jax.tree_util.keystr(path): value
            for path, value in jax.tree_util.tree_flatten_with_path(differences)[0]}


# fp32: the shards sum the same terms in another order (the entering state
# composed over shards rather than carried over chunks); observed at most
# 9.0e-7 of a leaf's largest value, on the output, every parameter gradient
# and the input gradient. bf16: no fixed bound between the whole and the
# split run holds. The projection and norm weight gradients are bf16
# reductions over the tokens, rounded on each shard's partial sum, so any
# change of summation order moves them by bf16 rounding, sequence split or
# not (the same layer split 8 ways over fsdp alone moves norm.weight by
# 2.9e-2). What the split must not do is add error of its own, so each bf16
# run is measured against an fp32 run of the same variables, and the split
# one may land at most twice as far from it as the whole one; over hidden
# seeds 0 to 7, the three meshes, dense and packed, the worst ratio seen is
# 1.68, on out_proj's kernel. A shard that
# started from a zero state instead of the earlier shards' moves the fp32
# output by 0.66, and a conv that did not read the previous shard's tail
# by 3.3.
FP32_BOUND = 2e-6


def mamba_parity(split: str, packed: bool, dtype, seed: int = 0) -> list[tuple[str, float, float]]:
    """The leaves where the split run misses its bound, each with the split
    run's and the whole run's distance (fp32: from each other; bf16: from
    the fp32 run); asserts the lowered state exchange on the way."""
    hidden = jax.random.normal(jax.random.key(seed), (2, MAMBA_LENGTH, 16), dtype)
    variables = mamba_layer(dtype).init(jax.random.key(1), hidden)
    # A step near 1 and decays of 0.05 to 0.5 per unit step, so what one
    # shard writes into the state still weighs on the shards after it.
    variables["params"]["dt_bias"] = jnp.full((4,), 0.5)
    variables["params"]["A_log"] = jnp.log(jnp.linspace(0.05, 0.5, 4))
    segments = mamba_segments() if packed else None

    def both(layer):
        def run(variables, hidden):
            return layer.apply(variables, hidden, segment_ids=segments)

        def loss(variables, hidden):
            return jnp.sum(jnp.sin(run(variables, hidden).astype(jnp.float32)))

        def outputs(variables, hidden):
            parameters, inputs = jax.grad(loss, argnums=(0, 1))(variables, hidden)
            return {"output": run(variables, hidden), "parameters": parameters, "input": inputs}
        return outputs

    whole = both(mamba_layer(dtype))(variables, hidden)
    with jax.set_mesh(build_mesh(MAMBA_SPLITS[split])):
        program = jax.jit(both(mamba_layer(dtype)))
        text = program.lower(variables, hidden).as_text()
        sharded = program(variables, hidden)
    assert "collective_permute" in text and "all_gather" not in text
    if dtype == jnp.float32:
        return [(leaf, difference, 0.0) for leaf, difference in leafwise_largest(whole, sharded).items()
                if difference >= FP32_BOUND]
    exact = both(mamba_layer(jnp.float32))(variables, hidden.astype(jnp.float32))
    whole_error, split_error = leafwise_largest(exact, whole), leafwise_largest(exact, sharded)
    return [(leaf, split_error[leaf], whole_error[leaf]) for leaf in whole_error
            if split_error[leaf] > 2 * whole_error[leaf]]


@pytest.mark.mesh
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
@pytest.mark.parametrize("split", sorted(MAMBA_SPLITS))
def test_mamba2_agrees_with_whole_sequences_forward_and_backward(split, packed, dtype):
    """The layer's output and the gradients of every parameter and of its
    input, whole sequences against the sequence axis split two to eight
    ways, the rows over fsdp or the tensor axis replicated beside it. The
    packed rows put documents on, inside and straddling the shard
    boundaries, so the state and the conv history reset across them. The
    lowered program holds the state exchange: the conv tail's
    collective-permutes and the state's, and no all-gather: the shards'
    states pass in log2(n) + 1 shifts rather than all to every shard."""
    assert mamba_parity(split, packed, dtype) == []


def hybrid():
    """Three Mamba-2 layers then one attention layer, the pattern of the
    Mamba-2 hybrids at toy size."""
    return models.build(
        "causal_transformer", vocab_size=VOCAB, emb_features=32, num_layers=4,
        num_heads=4, num_kv_heads=2, mlp_features=64, max_seq_len=SEQ_LEN,
        layer_types=("mamba",) * 3 + ("attention",),
        kinds={"mamba": {"mixer": {"kind": "mamba2", "num_heads": 4, "head_dim": 8,
                                   "state_size": 8, "n_groups": 1, "chunk_size": 4}}})


# tensor=2 by sequence=4 leaves fsdp at 1, so the layout's tolerance has to
# allow the widths it cannot split (29.8% of this model's elements).
HYBRID_SPLITS = {
    "fsdp2_sequence4": (MeshSpec(fsdp=2, sequence=4), 0.02),
    "tensor2_sequence4": (MeshSpec(tensor=2, sequence=4), 0.35),
}


@pytest.mark.mesh
@pytest.mark.parametrize("split", sorted(HYBRID_SPLITS))
@pytest.mark.parametrize("make_batch", [dense_batch, packed_batch])
def test_a_mamba2_hybrid_trains_the_same_step_under_a_split_sequence(make_batch, split):
    """A hybrid decoder, three Mamba-2 layers and one attention layer, takes
    the step fsdp=8 takes with the sequence split four ways: the same loss
    and every parameter within fp32 rounding after sgd(1.0), on a dense
    batch and on a packed one whose documents start at a different column
    of every row. The trainer's own compiled step holds the Mamba-2 state
    exchange. Observed: equal losses, parameters at most 3.6e-7 apart."""
    batch = make_batch()
    spec, tolerance = HYBRID_SPLITS[split]
    sharded = one_step(spec, batch, tolerance, model=hybrid)
    assert "collective_permute" in sharded[2]
    whole = one_step(WHOLE, batch, model=hybrid)
    assert_same_step(whole, sharded)
