"""A recurrent mixer split over the sequence axis, and its heads over the axes a split leaves them: the gated
delta net and Kimi delta attention, each shard running its slice from the memory it enters with, its
summary's composition over the earlier shards (`recurrence_chunk.entering_memory`), and its conv after the
real tokens before it (`linear.sequence_conv`), and Mamba-1, Mamba-2 and lightning, checked against the same
layer on one device (row 1437)."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import as_given, assert_as_exact, widened
from sequence_split_support import (
    SPLIT_BATCH,
    SPLIT_ROWS,
    assert_as_on_one_device,
    assert_decodes_as_whole,
    assert_split_as_whole,
)

from dew.nn.inputs import MixerCall
from dew.nn.kda import KimiDeltaAttention
from dew.nn.linear import GatedDeltaNet, Heads, held_conv1d, over_sequence, sequence_conv, sequence_halo
from dew.nn.mixers.lightning import LightningAttention
from dew.nn.mixers.mamba import Mamba
from dew.nn.mixers.mamba2 import Mamba2
from dew.nn.mixers.mamba3 import Mamba3
from dew.training import MeshSpec

WIDTH = 32
"""The hidden width every mixer here reads and writes."""

MIXERS = {
    "gated delta net": GatedDeltaNet(emb_features=WIDTH, num_k_heads=2, num_v_heads=4, head_k_dim=8,
                                     head_v_dim=8, chunk_size=4, dtype=jnp.float32),
    "kimi delta attention": KimiDeltaAttention(emb_features=WIDTH, num_heads=2, head_dim=8, conv_kernel=4,
                                               lower_bound=-5.0, chunk_size=4),
}
"""Each mixer at a small geometry: the gated delta net's queries and keys on half its value heads, chunks of
four tokens so a shard's slice of eight holds two."""

SEEDS = {"all_valid": 64, "packed": 64, "right_padded": 64, "empty_shard_packed": 64}
"""Each row's declared draws."""


@pytest.mark.mesh
@pytest.mark.parametrize("row", sorted(SEEDS))
@pytest.mark.parametrize("name", sorted(MIXERS))
def test_a_delta_rule_split_over_the_sequence_runs_its_rows_as_whole_ones(name, row):
    """Under a sequence axis of 4, the mixer runs each row as it runs it on one device: real tokens, packed
    documents (one continuing across a cut, one starting on a shard's first real token), right padding and
    an interior shard of padding, which the memory crosses unchanged (`assert_split_as_whole`)."""
    module = MIXERS[name]
    hidden = jnp.zeros((1, 32, WIDTH), jnp.float32)
    variables = jax.tree.map(lambda leaf: np.asarray(leaf), module.init(jax.random.key(7), hidden))
    assert_split_as_whole(module, variables, row, SEEDS[row], f"{name} split over the sequence, {row}",
                          width=WIDTH)


WIDE_HEADS = {
    "gated delta net": GatedDeltaNet(emb_features=WIDTH, num_k_heads=4, num_v_heads=8, head_k_dim=32,
                                     head_v_dim=32, chunk_size=4, dtype=jnp.float32),
    "kimi delta attention": KimiDeltaAttention(emb_features=WIDTH, num_heads=4, head_dim=32, conv_kernel=4,
                                               lower_bound=-5.0, chunk_size=4),
}
"""Each delta-rule mixer with heads its scan's work dwarfs its projections' on: four key heads 32 wide."""


PAIRED_HEADS = {
    "gated delta net": GatedDeltaNet(emb_features=WIDTH, num_k_heads=2, num_v_heads=4, head_k_dim=64,
                                     head_v_dim=64, chunk_size=4, dtype=jnp.float32),
    "kimi delta attention": KimiDeltaAttention(emb_features=WIDTH, num_heads=2, head_dim=64, conv_kernel=4,
                                               lower_bound=-5.0, chunk_size=4),
}
"""Each delta-rule mixer with two key heads its scan's work dwarfs its projections' on: a tensor axis of 2
divides them, the tensor and sequence axes together do not."""

HEADS_BEFORE_TIME = (("activation_heads", ("tensor", "sequence")),)
"""A rule under which the heads would take the sequence axis too: the split keeps it the sequence's."""


def heads_rules(rules: tuple) -> tuple:
    """`rules` before the default table, the heads' own alone where `rules` give them any: no fallback of the
    defaults' to a tensor split of their own."""
    from dew.mesh import DEFAULT_RULES

    given = {name for name, _ in rules}
    return (*rules, *(rule for rule in DEFAULT_RULES if rule[0] not in given))


@pytest.mark.mesh
@pytest.mark.parametrize(("mixers", "rules"), [(MIXERS, ()), (WIDE_HEADS, HEADS_BEFORE_TIME)],
                         ids=["default rules", "heads before time, four key heads"])
@pytest.mark.parametrize("row", ["packed", "empty_shard_packed"])
@pytest.mark.parametrize("name", sorted(MIXERS))
def test_a_delta_rule_split_over_the_sequence_and_its_heads_runs_its_rows_as_whole_ones(name, row, mixers,
                                                                                         rules):
    """Under a sequence axis of 2 beside a tensor axis of 2, which splits the mixer's heads (each device its
    share of the queries', keys' and values' heads and of their conv's taps, `linear.sequence_delta_rule`),
    the mixer runs packed rows as on one device (`assert_split_as_whole`): one starting a document at the
    cut, one carrying a document's conv and memory across it past padding, under the default rules and
    under rules giving the heads the sequence axis too (four key heads, which both axes would divide), which
    the split keeps the sequence's."""
    module = mixers[name]
    hidden = jnp.zeros((1, 32, WIDTH), jnp.float32)
    variables = jax.tree.map(lambda leaf: np.asarray(leaf), module.init(jax.random.key(7), hidden))
    with nn.logical_axis_rules(heads_rules(rules)):
        assert_split_as_whole(module, variables, row, SEEDS[row], f"{name} split over the sequence and its "
                              f"heads, {row}, {rules}", width=WIDTH, shards=2, tensor=2)


@pytest.mark.mesh
@pytest.mark.parametrize(("mixers", "rules"), [(WIDE_HEADS, ()), (PAIRED_HEADS, HEADS_BEFORE_TIME)],
                         ids=["default rules", "heads before time, two key heads"])
@pytest.mark.parametrize("name", sorted(WIDE_HEADS))
def test_a_delta_rule_split_over_the_sequence_divides_its_heads_work_over_a_tensor_axis(name, mixers, rules):
    """Under a sequence axis of 2, a tensor axis of 2 beside it divides the mixer's heads (`linear.
    sequence_delta_rule`): each device's compiled forward and backward on the host's devices (XLA's count of
    the portable math's every dot) does under three quarters of the work it does with no tensor axis, its
    scan's and conv's half (row 1437: each device ran every head, 0.99 of it), and so under rules giving the
    heads the sequence axis too, and no other, where the tensor axis alone divides them (two key heads)."""
    module = mixers[name]
    assert split_work(module, MixerCall(), rules, tensor=2) < 0.75 * split_work(module, MixerCall(), rules,
                                                                                tensor=1)


def split_work(module, call: MixerCall, rules: tuple, *, tensor: int) -> float:
    """The work each device's compiled forward and backward of `module` called with `call` does under a
    sequence axis of 2 beside a tensor axis of `tensor`, on the host's devices, which take the portable
    math, whose every dot XLA's count holds."""
    hidden = jnp.asarray(np.random.default_rng(0).standard_normal((2, 32, WIDTH)), jnp.float32)
    variables = module.init(jax.random.key(7), hidden, call)

    def step(params, hidden):
        out, pull = jax.vjp(lambda params, hidden: module.apply(params, hidden, call), params, hidden)
        return out, pull(jnp.ones_like(out))

    mesh = MeshSpec(sequence=2, tensor=tensor).build(jax.devices("cpu")[:2 * tensor])
    with jax.set_mesh(mesh), nn.logical_axis_rules(heads_rules(rules)):
        return jax.jit(step).lower(variables, hidden).compile().cost_analysis()["flops"]


SPLIT_HEADS = {
    "mamba": Mamba(emb_features=WIDTH, intermediate_size=64, state_size=64, time_step_rank=2, chunk_size=4),
    "mamba2": Mamba2(emb_features=WIDTH, num_heads=4, head_dim=32, state_size=64, n_groups=2, chunk_size=4,
                     dtype=jnp.float32),
    "lightning": LightningAttention(emb_features=WIDTH, num_heads=4, head_dim=32,
                                    slopes=(0.5, 0.25, 0.125, 0.0625), layers=4, chunk_size=4),
}
"""The other recurrent mixers that split a training call over the sequence, each with heads (Mamba-1's
channels) its scan's work dwarfs its projections' on, Mamba-2's B and C in two groups a tensor axis of 2
divides."""


def _called(name: str) -> MixerCall:
    """A training call of `SPLIT_HEADS[name]` over two rows of 32 tokens: a lightning layer's the second of
    its stack's (`MixerCall.depth`)."""
    return MixerCall(depth=jnp.ones((2, 32), jnp.int32)) if name == "lightning" else MixerCall()


@pytest.mark.mesh
@pytest.mark.parametrize("name", sorted(SPLIT_HEADS))
def test_a_recurrent_mixer_split_over_the_sequence_divides_its_heads_work_over_a_tensor_axis(name):
    """Under a sequence axis of 2, a tensor axis of 2 beside it divides Mamba-1's channels and Mamba-2's and
    a lightning layer's heads (`linear.over_sequence`'s `Heads`), as it divides a delta rule's: each
    device's compiled forward and backward does under three quarters of the work it does with no tensor
    axis, where each ran every head (train 1108's F1)."""
    module, call = SPLIT_HEADS[name], _called(name)
    assert split_work(module, call, (), tensor=2) < 0.75 * split_work(module, call, (), tensor=1)


HEADED_MIXERS = {
    "mamba": Mamba(emb_features=WIDTH, intermediate_size=16, state_size=4, time_step_rank=2, chunk_size=4),
    "mamba2": Mamba2(emb_features=WIDTH, num_heads=4, head_dim=8, state_size=8, n_groups=2, chunk_size=4,
                     dtype=jnp.float32),
}
"""Mamba-1 and -2 at a small geometry whose heads (channels) and groups a tensor axis of 2 divides."""


@pytest.mark.mesh
@pytest.mark.parametrize("row", ["packed", "empty_shard_packed"])
@pytest.mark.parametrize("name", sorted(HEADED_MIXERS))
def test_a_mamba_layer_split_over_the_sequence_and_its_heads_runs_its_rows_as_whole_ones(name, row):
    """Under a sequence axis of 2 beside a tensor axis of 2, which splits Mamba-1's channels and Mamba-2's
    heads and groups (each device its share of the conv's channels and taps and of the scan's, `Heads`),
    the layer runs packed rows as on one device (`assert_split_as_whole`): one starting a document at the
    cut, one carrying a document's conv and state across it past padding."""
    module = HEADED_MIXERS[name]
    hidden = jnp.zeros((1, 32, WIDTH), jnp.float32)
    variables = jax.tree.map(lambda leaf: np.asarray(leaf), module.init(jax.random.key(7), hidden))
    assert_split_as_whole(module, variables, row, SEEDS[row],
                          f"{name} split over the sequence and its heads, {row}", width=WIDTH, shards=2,
                          tensor=2)


DECODE_MIXERS = {
    **MIXERS,
    "mamba": Mamba(emb_features=WIDTH, intermediate_size=16, state_size=4, time_step_rank=2, chunk_size=4),
    "mamba2": Mamba2(emb_features=WIDTH, num_heads=4, head_dim=8, state_size=8, n_groups=1, chunk_size=4,
                     dtype=jnp.float32),
}
"""The recurrent mixers that split a training call over the sequence (`linear.sequence_split`)."""

WHOLE_MIXERS = {
    **DECODE_MIXERS,
    "mamba3": Mamba3(emb_features=WIDTH, num_heads=4, head_dim=8, state_size=8, mimo_rank=2, chunk_size=4,
                     dtype=jnp.float32),
}
"""The recurrent mixers whose decode call and call over forks run whole under a sequence axis
(`linear.sequence_split`): those that split a training call over it, and Mamba-3, which refuses one."""

DECODE_SEEDS = 32
"""Each mixer's declared draws for its decode under the axis."""


@pytest.mark.mesh
@pytest.mark.parametrize("name", sorted(WHOLE_MIXERS))
def test_a_recurrent_mixer_decodes_under_a_sequence_axis_as_on_one_device(name):
    """Under a sequence axis of 4, a decode call runs whole, its state replicated over the axis: one token a
    row holds no sequence to split (`linear.sequence_split`, rows 7400, 7424). A prefill and the steps after
    it decode as on one device (`assert_decodes_as_whole`), where the mixer refused the call."""
    module = WHOLE_MIXERS[name]
    hidden = jnp.zeros((1, 8, WIDTH), jnp.float32)
    variables = jax.tree.map(np.asarray, module.init(jax.random.key(7), hidden))
    assert_decodes_as_whole(module, variables, DECODE_SEEDS, f"{name} decoded under a sequence axis",
                            width=WIDTH)


LIGHTNING = LightningAttention(emb_features=WIDTH, num_heads=2, head_dim=8, slopes=(0.5, 0.25), layers=4,
                               chunk_size=4)
"""MiniMax's lightning layer at a small geometry, the second of a stack of four."""


def on_one_device(module, call: MixerCall, shape: tuple[int, int], seeds: int, site: str, *, shards: int = 4,
                  tensor: int = 1):
    """`module`, initialized here, called with `call` over hidden states of `shape` `[B, S]` and `WIDTH`
    wide on a sequence axis of `shards` beside a tensor axis of `tensor` as on one device
    (`assert_as_on_one_device`)."""
    depth = None if call.depth is None else jnp.ones((1, shape[1]), jnp.int32)
    variables = jax.tree.map(np.asarray, module.init(jax.random.key(7), jnp.zeros((1, shape[1], WIDTH)),
                                                     MixerCall(depth=depth)))
    assert_as_on_one_device(module, variables, call, (*shape, WIDTH), seeds, site, shards=shards,
                            tensor=tensor)


@pytest.mark.mesh
@pytest.mark.parametrize("row", sorted(SEEDS))
def test_a_lightning_layer_split_over_the_sequence_runs_its_rows_as_whole_ones(row):
    """Under a sequence axis of 4, a lightning layer runs each row of `SEEDS`' as on one device, each shard
    from the memory the earlier ones leave it (`linear.sequence_entered`, rows 7400, 7423), where it refused
    the axis: real tokens, packed documents, right padding and an interior shard of padding."""
    from dew.batch import coordinates
    from dew.nn.inputs import AttentionMetadata

    valid, segments = SPLIT_ROWS[row]
    tokens = coordinates(valid=jnp.asarray(np.broadcast_to(valid, (SPLIT_BATCH, 32))),
                         segments=None if segments is None else jnp.asarray(
                             np.broadcast_to(segments, (SPLIT_BATCH, 32)), jnp.int32))
    call = MixerCall(AttentionMetadata(coordinates=tokens), depth=jnp.ones((SPLIT_BATCH, 32), jnp.int32))
    on_one_device(LIGHTNING, call, (SPLIT_BATCH, 32), SEEDS[row],
                            f"lightning split over the sequence, {row}")


@pytest.mark.mesh
@pytest.mark.parametrize("row", ["packed", "empty_shard_packed"])
def test_a_lightning_layer_split_over_the_sequence_and_its_heads_runs_its_rows_as_whole_ones(row):
    """Under a sequence axis of 2 beside a tensor axis of 2, which splits a lightning layer's heads (each
    device its share of the queries', keys', values' and decays' heads, `Heads`), the layer runs packed rows
    as on one device, each shard and head from the memory the earlier shards leave it."""
    from dew.batch import coordinates
    from dew.nn.inputs import AttentionMetadata

    valid, segments = SPLIT_ROWS[row]
    tokens = coordinates(valid=jnp.asarray(np.broadcast_to(valid, (SPLIT_BATCH, 32))),
                         segments=jnp.asarray(np.broadcast_to(segments, (SPLIT_BATCH, 32)), jnp.int32))
    call = MixerCall(AttentionMetadata(coordinates=tokens), depth=jnp.ones((SPLIT_BATCH, 32), jnp.int32))
    on_one_device(LIGHTNING, call, (SPLIT_BATCH, 32), SEEDS[row],
                  f"lightning split over the sequence and its heads, {row}", shards=2, tensor=2)


FORK_ROWS = [[3, 4, 5, 6, 7, 8, 9, 10, 11], [3, 4, 5, 6, 7, 8, 9, 12], [20, 21, 22, 23, 24]]
"""Two rows over one prompt of seven tokens, and a row of its own."""

FORK_SEEDS = 32
"""Each mixer's declared draws over forks under the axis."""


@pytest.mark.mesh
@pytest.mark.parametrize("name", [*sorted(WHOLE_MIXERS), "lightning"])
def test_a_recurrent_mixer_reads_forks_under_a_sequence_axis_as_on_one_device(name):
    """Under a sequence axis of 4, a call over forks runs whole, each document continuing its parent's state
    across the axis (`linear.sequence_split`, rows 7400, 7424), where `held_scan`'s forked path and Mamba-3
    refused the axis (`on_one_device`)."""
    from dew.data.forked_rows import ForkedRows
    from dew.nn.inputs import AttentionMetadata

    module = LIGHTNING if name == "lightning" else WHOLE_MIXERS[name]
    forks = ForkedRows.pack(FORK_ROWS, 24)
    batch, length = forks.tokens.shape
    depth = jnp.ones((batch, length), jnp.int32) if name == "lightning" else None
    call = MixerCall(AttentionMetadata(coordinates=forks.coordinates), depth=depth)
    on_one_device(module, call, (batch, length), FORK_SEEDS,
                            f"{name} over forks under a sequence axis")


CONV_SEEDS = 96
"""The split conv's declared draws, the count its bias gradient's excess was measured over (row 2169)."""


@pytest.mark.mesh
@pytest.mark.parametrize("row", ["packed", "empty_shard_packed"])
def test_a_split_convs_bias_gradient_sums_its_shards_as_the_whole_conv_does(row):
    """A causal conv over a sequence split four ways (`linear.sequence_conv`), on packed rows: its output
    and every gradient on par with the unsplit conv's from float64 (check 1). Summed as the boundary sums a
    replicated weight's cotangent, four rounded partials added in float32, the bias gradient's error
    doubled the unsplit conv's (1.82x and 2.03x, row 2169); the shards' partials are added in one rounding
    (`linear._shards_summed`). What remains is each shard's own sum, XLA:CPU's sequential reduction over
    its rows where the unsplit conv's is one blocked reduction over all of them: the bias gradient is held
    within the larger of the unsplit conv's error and a library arm's (the gate's ruling on row 2169),
    which sums each shard's partial through XLA as the split does and adds the four in float64."""
    valid, segments = (jnp.asarray(np.broadcast_to(field, (SPLIT_BATCH, 32))) for field in SPLIT_ROWS[row])
    rng = np.random.default_rng(0x2169)
    taps, bias = (rng.standard_normal(shape).astype(np.float32) for shape in ((16, 4), (16,)))
    mesh = MeshSpec(sequence=4).build(jax.devices()[:4])

    def pulled(conv):
        def run(taps, bias, mixed, cotangent):
            out, pull = jax.vjp(conv, taps, bias, mixed)
            return out, *pull(cotangent)
        return jax.jit(run)

    def split_conv(taps, bias, mixed):
        # Each channel its own, as Mamba-1's conv splits them (`Heads`).
        return over_sequence(lambda mixed, valid, segments, taps, bias: sequence_conv(
            mixed, taps, bias, segments, valid), 4, (mixed, valid, segments), (taps, bias),
                             heads=Heads(16, 16, {0: 2, 3: 0, 4: 0}))

    def whole_conv(taps, bias, mixed):
        return held_conv1d(mixed, taps, valid, bias=bias, segments=segments)[0]

    split_run, whole, wide = pulled(split_conv), pulled(whole_conv), pulled(whole_conv)

    def shard_partial(mixed, valid, segments, cotangent, taps, bias):
        # The shard's own bias gradient as the split computes it (`sequence_conv`'s halo and conv), before
        # any sum across the shards: `[rows, 1, D]`, its rows alike, one slot a shard.
        halo = sequence_halo(mixed, valid, segments, taps.shape[1] - 1)
        _, pull = jax.vjp(lambda bias: held_conv1d(mixed, taps, valid, halo, bias=bias, segments=segments)[0],
                          bias)
        (partial,) = pull(cotangent)
        return jnp.broadcast_to(partial, (mixed.shape[0], 1, partial.shape[-1]))

    partials = jax.jit(lambda taps, bias, mixed, cotangent: over_sequence(
        shard_partial, 4, (mixed, valid, segments, cotangent), (taps, bias),
        heads=Heads(16, 16, {0: 2, 3: 2, 4: 0, 5: 0})))

    def draw(seed):
        generator = np.random.default_rng([0x2169, seed])
        return tuple(generator.standard_normal((SPLIT_BATCH, 32, 16)).astype(np.float32) for _ in range(2))

    @as_given
    def split(mixed, cotangent):
        with jax.set_mesh(mesh):
            return split_run(taps, bias, mixed, cotangent)

    @as_given
    def truth(mixed, cotangent):
        with jax.enable_x64(new_val=True):
            return wide(*widened((taps, bias, mixed, cotangent)))

    @as_given
    def library(mixed, cotangent):
        out, grad_taps, _, grad_mixed = whole(taps, bias, mixed, cotangent)
        with jax.set_mesh(mesh):
            shards = np.asarray(partials(taps, bias, mixed, cotangent), np.float64)[0]
        return out, grad_taps, np.sum(shards, axis=0).astype(np.float32), grad_mixed

    assert_as_exact(split, lambda mixed, cotangent: whole(taps, bias, mixed, cotangent), draw,
                    f"a conv split over the sequence, {row}", seeds=CONV_SEEDS, truth_fn=truth,
                    library_fn=library, library_row=2169, library_leaves=("[2]",))

TRANSFER_SEEDS = 16
"""The draws a slice's transfer is held on par over (its test below)."""


@pytest.mark.parametrize("per_key", [False, True], ids=["a decay a head", "a decay a key dimension"])
def test_a_slices_transfer_is_on_par_with_the_rule_from_the_identity(per_key):
    """A sequence shard's transfer (`recurrence_chunk.transferred`: the derived forms' walk chaining each
    chunk's transition in fp32, row 1437) on par from float64 with the delta rule's fp32 final memory over
    zero values from the identity, which `entering_memory` took before (check 1), in 16-token chunks (not
    the rule's 64) over 150 tokens, off their grid: a row whose document starts inside the slice (the
    transfer zero across it) and a row whose one document spans it, its transfer ten chunks' transitions
    chained, about 0.3% of the identity's norm under these gates (a reset on every row made every
    transfer zero, whatever its chunks: #3757's review). A slice of no tokens transfers the identity."""
    from dew.nn.kernels import generation
    from dew.nn.kernels.recurrence_chunk import chunk_gated_delta_rule, recurrent_delta_rule, transferred

    rows, length, heads, width, chunk = 2, 150, 2, 16, 16
    starts = np.zeros((rows, length), bool)
    starts[0, 70] = True

    def draw(seed):
        generator = np.random.default_rng([1437, seed])
        query, key = (generator.standard_normal((rows, length, heads, width)) for _ in range(2))
        key = key / np.linalg.norm(key, axis=-1, keepdims=True)
        decay = (rows, length, heads) + ((width,) if per_key else ())
        # Faint decays and write gates (softplus(n - 4), sigmoid(n - 2)): the spanning row's transfer kept.
        g = -np.log1p(np.exp(generator.standard_normal(decay) - 4.0))
        beta = 1 / (1 + np.exp(2.0 - generator.standard_normal((rows, length, heads))))
        return tuple(np.asarray(x, np.float32) for x in (query, key, g, beta))

    def unread(key):
        return jnp.zeros((rows, length, heads, width), key.dtype)

    def identity(key):
        return jnp.broadcast_to(jnp.eye(width, dtype=key.dtype), (rows, heads, width, width))

    def chained(query, key, g, beta):
        return transferred(query, key, unread(key), g, beta, starts, chunk_size=chunk)

    def ruled(query, key, g, beta):
        with generation.chosen(gated_delta_rule="xla"):
            return chunk_gated_delta_rule(query, key, unread(key), g, beta, identity(key), starts=starts)[1]

    @as_given
    def truth(query, key, g, beta):
        with jax.enable_x64(new_val=True):
            query, key, g, beta = widened((query, key, g, beta))
            # The token recurrence's decay a key dimension: a head's, each key's alike.
            g = g if per_key else jnp.broadcast_to(g[..., None], (*g.shape, width))
            return recurrent_delta_rule(query, key, unread(key), g, beta, identity(key), starts=starts)[1]

    assert_as_exact(jax.jit(chained), jax.jit(ruled), draw, f"a slice's transfer, per_key {per_key}",
                    seeds=TRANSFER_SEEDS, truth_fn=truth)
    empty = tuple(x[:, :0] for x in draw(0))
    found = transferred(*empty[:2], jnp.zeros((rows, 0, heads, width)), *empty[2:], chunk_size=chunk)
    assert np.array_equal(np.asarray(found), np.asarray(identity(jnp.zeros((), jnp.float32))))
