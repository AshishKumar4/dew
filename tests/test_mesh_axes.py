"""Sequence, tensor and stage mesh axes: placement, batches, and loss equality.

The mesh carries six axes; parameters distinguish fsdp, expert and tensor,
the batch's sequence dimension rides sequence, which holds the state with
fsdp, and the layer stack's pipeline stages ride stage. Widths stay on fsdp unless a run's rules
redirect them onto tensor, so the default mesh places as the three-axis one
did. A fit on each of the sim-mesh topologies trains the same losses:
sharding moves values, never changes them.
"""


import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import sharded
from flax import linen as nn
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
from model_support import TINY_DECODER
from recording import RecordingTracker
from sharded import assert_sharded
from token_support import repeated_token_batch, write_token_corpus

from dew.checkpoints import Checkpoints
from dew.data.dataset import Loader
from dew.mesh import mesh_axes
from dew.nn.backbones.dit import SimpleDiT
from dew.nn.blocks import Upsample
from dew.nn.conv import Conv
from dew.nn.sharding import declarations, declared_axes
from dew.nn.ssm import SpatialFusionConv
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.telemetry.instrumentation import COLLECTIVES
from dew.training import Layout, MeshSpec, Trainer
from dew.training.distributed import shard_batch

# Needs the eight simulated CPU devices conftest configures; the GPU lane skips it.
pytestmark = pytest.mark.mesh

VOCAB = 64
# Training batches carry seq_len + 1 columns for the one-token shift, so the
# 17 columns of a 16-token model never split over a sequence axis of two: the
# batch stays whole over it, and the attention claims the axis for the
# model's 16 tokens itself (tests/test_sequence_parallel.py). The striped
# order needs a length that is a multiple of twice the shard count.
SEQ_LEN = 16
BATCH = 8
TINY_SHARD = 256


def tiny(**overrides):
    return models.build("causal_transformer", **{"vocab_size": VOCAB, **TINY_DECODER, "num_kv_heads": 2,
                                                 **overrides})


DELTA = {"class": "gated_delta_net", "fields": {
    "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 8,
    "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4}}


def test_fsdp_beside_tensor_computes_each_matmul_of_the_step_once():
    """Two fsdp shards beside two tensor shards split every matmul of the
    step, so their devices together compute one device's FLOPs. With the
    mlp's and o_proj's widths split over fsdp and tensor at once, the
    residual width left whole, GSPMD gathered a block's activations over
    fsdp and repeated those products on both devices of each pair: 1.21
    times one device's matmul FLOPs on layout_parity's dense decoder."""
    from dew.telemetry.instrumentation import compiled_flops

    def flops(mesh: MeshSpec, devices) -> float:
        trainer = Trainer(LMObjective(tiny(), SEQ_LEN), optax.adam(1e-3), key=jax.random.key(0),
                          mesh=mesh, layout=Layout(min_shard=TINY_SHARD), checkpoints=None,
                          tracker=None)
        trainer.device_mesh = mesh.build(devices)
        state, _, _ = trainer.place()
        trainer.compile(state, shard_batch(trainer.device_mesh, next(token_batches())))
        assert trainer.executable is not None
        counted = compiled_flops(trainer.executable)
        assert counted is not None
        return counted * len(devices)

    one = flops(MeshSpec(), jax.devices()[:1])
    assert flops(MeshSpec(fsdp=2, tensor=2), jax.devices()[:4]) == pytest.approx(one, rel=1e-6)


def test_tensor_parallelism_computes_latent_attentions_down_projections_once():
    """Multi-head latent attention compresses the residual into a query
    latent and a key-value latent before the head-split up-projections. The
    tensor axis splits the heads, not those latents, so every tensor shard
    computed both down-projections over every token: at DeepSeek-V3's shape
    (hidden 7168, latents 1536 and 512 + 64, 4096 tokens) 13.3% more work a
    device than an even split under tensor=8, 5.7% under tensor=4. Four
    tensor shards compute one device's matmul FLOPs. (The latents' kernels
    ride fsdp, so with no fsdp axis they stay whole on every device: a sixth
    of this toy's weights, a thousandth of V3's.)"""
    from dew.telemetry.instrumentation import compiled_flops

    model = models.build(
        "causal_transformer", vocab_size=VOCAB, emb_features=64, num_layers=1, num_heads=4,
        mlp_features=64, max_seq_len=SEQ_LEN,
        mixer={"class": "mla", "fields": {"q_lora_rank": 48, "kv_lora_rank": 32, "qk_nope_head_dim": 16,
               "qk_rope_head_dim": 8, "v_head_dim": 16}})

    def flops(mesh: MeshSpec, devices) -> float:
        trainer = Trainer(LMObjective(model, SEQ_LEN), optax.adam(1e-3), key=jax.random.key(0),
                          mesh=mesh, layout=Layout(min_shard=TINY_SHARD, tolerance=1.0),
                          checkpoints=None, tracker=None)
        trainer.device_mesh = mesh.build(devices)
        state, _, _ = trainer.place()
        trainer.compile(state, shard_batch(trainer.device_mesh, next(token_batches())))
        assert trainer.executable is not None
        counted = compiled_flops(trainer.executable)
        assert counted is not None
        return counted * len(devices)

    one = flops(MeshSpec(), jax.devices()[:1])
    assert flops(MeshSpec(tensor=4), jax.devices()[:4]) == pytest.approx(one, rel=1e-6)


LENGTH = 20
"""One row a device of 20 positions, as a long run puts one."""


def stepped(mesh: MeshSpec, devices: int, **hybrid):
    """One SGD step of an untied tiny decoder on `mesh` over `devices`
    devices: its parameters' shardings, the compiled step's HLO, and the
    loss and trained parameters. SGD moves each parameter by its gradient
    alone, where adam's normalized step turns a near-zero gradient's last
    bits into a visible move."""
    batch = {"text": np.random.default_rng(0).integers(0, VOCAB, (2, LENGTH + 1)).astype(np.int32)}
    model = tiny(max_seq_len=LENGTH, tie_embeddings=False, **hybrid)
    trainer = Trainer(LMObjective(model, LENGTH), optax.sgd(1e-2), key=jax.random.key(0), mesh=mesh,
                      layout=Layout(min_shard=TINY_SHARD), checkpoints=None, tracker=None)
    trainer.device_mesh = mesh.build(jax.devices()[:devices])
    state, shardings, _ = trainer.place()
    placed = shard_batch(trainer.device_mesh, batch)
    trained, loss, *_ = trainer.compile(state, placed)(state, placed)
    return (shardings.variables["params"], trainer.executable.as_text(),
            jax.device_get((loss, trained.variables["params"])))


def delta_layers(*, fused: bool) -> dict:
    return {"layer_types": ("linear_attention", "full_attention"), "kinds": {"linear_attention": {
        "mixer": {**DELTA, "fields": {**DELTA["fields"], "fused_in_proj": fused}}}}}


def assert_trains_alike(trained, expected):
    for got, want in zip(jax.tree.leaves(trained), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("fused", [False, True], ids=["delta-net", "fused-projections"])
def test_an_fsdp_step_splits_leading_dimensions_and_moves_no_activation(fused):
    """A hybrid decoder's step over fsdp alone trains what one device
    trains. fsdp stores a split, so each declared kernel's shard is a block
    of its rows: the gate and up projections and the delta net's input
    projections split the residual width they read, not the width a rule
    names first; the head keeps its vocabulary split, which its loss
    computes on. Each device computes its own rows, so the step's
    activation moves are the vocabulary's (the token ids and hidden states
    its split head and embedding exchange), what an all-attention decoder
    moves: unplaced, the delta net ran head-split with the rows gathered
    (`GatedDeltaNet._project`), Qwen3-Next's fused projections before their
    reshape."""
    def moved(text: str) -> list:
        # Every exchange of entries, and each gathered array of the step's
        # positions, read apart from those a combined gather carries with it.
        return sorted((op, shape) for op, shapes in sharded.collectives(text) if op in sharded.MOVES
                      for shape in shapes if op != "all-gather" or LENGTH in shape)

    shardings, text, trained = stepped(MeshSpec(fsdp=2), 2, **delta_layers(fused=fused))
    declared = declarations(shardings)
    specs = {jax.tree_util.keystr(path): sharding.spec
             for path, sharding in jax.tree_util.tree_leaves_with_path(shardings)
             if sharding.spec and "kernel" in jax.tree_util.keystr(path) and declared_axes(path, 2, declared)}
    assert specs.pop("['lm_head']['kernel']") == P(None, "fsdp")
    assert set(specs.values()) == {P("fsdp")}, specs
    if fused:
        # Qwen3-Next's fused projections declare the residual axis their unfused ones do.
        assert all(any(f"['{name}']" in path for path in specs) for name in ("in_proj_qkvz", "in_proj_ba")), (
            sorted(specs))
    assert moved(text) == moved(stepped(MeshSpec(fsdp=2), 2)[1])
    assert_trains_alike(trained, stepped(MeshSpec(), 1, **delta_layers(fused=fused))[2])


@pytest.mark.mesh(devices=4)
def test_a_delta_net_step_over_fsdp_and_tensor_trains_what_one_device_trains():
    """fsdp beside tensor: the delta net's gates keep the heads the tensor
    axis splits their kernels by, its other projections their width whole."""
    assert_trains_alike(stepped(MeshSpec(fsdp=2, tensor=2), 4, **delta_layers(fused=False))[2],
                        stepped(MeshSpec(), 1, **delta_layers(fused=False))[2])


@pytest.mark.parametrize(("mesh", "devices"), [
    pytest.param(MeshSpec(sequence=2), 2, marks=pytest.mark.mesh(devices=2), id="sequence2"),
    pytest.param(MeshSpec(fsdp=2, sequence=2), 4, marks=pytest.mark.mesh(devices=4), id="fsdp2_sequence2"),
])
def test_a_sequence_split_holds_the_state_in_shares_and_trains_what_one_device_trains(mesh, devices):
    """The sequence axis's devices compute their own positions with the same weights, so they hold
    the persistent state in shares beside fsdp's (`STATE_AXES`): each parameter lies on them as an
    fsdp of as many devices lays it, every device holding the same share, and one step trains what
    one device trains."""
    shardings, _, (loss, trained) = stepped(mesh, devices)
    model = tiny(max_seq_len=LENGTH, tie_embeddings=False)
    params = LMObjective(model, LENGTH).init(jax.random.key(0))["params"]
    fsdp = MeshSpec(fsdp=devices).build(jax.devices()[:devices])
    grouped = Layout(min_shard=TINY_SHARD).shardings(fsdp, params)

    def shares(placement):
        return [sharding.shard_shape(np.shape(leaf))
                for sharding, leaf in zip(jax.tree.leaves(placement), jax.tree.leaves(trained), strict=True)]

    assert shares(shardings) == shares(grouped)
    assert any("sequence" in mesh_axes(entry)
               for sharding in jax.tree.leaves(shardings) for entry in sharding.spec)
    _, _, expected = stepped(MeshSpec(), 1)
    assert_trains_alike((loss, trained), expected)


@pytest.mark.mesh(devices=8)
def test_the_sequence_axis_shares_the_state_beside_every_split_the_rules_give_it():
    """The state axes split what the rules put on fsdp and leave every other axis's split as it is:
    a vocabulary of 516 that tensor 4 divides and tensor 4 times sequence 2 does not keeps its
    table split over tensor, and no leaf's tensor split moves when sequence joins the mesh."""
    model = tiny(vocab_size=516, emb_features=256, mlp_features=512)
    params = LMObjective(model, SEQ_LEN).init(jax.random.key(0))["params"]

    def others(placement):
        """Each leaf's axes past the state axes, by dimension."""
        specs = [tuple(sharding.spec) + (None,) * 4 for sharding in jax.tree.leaves(placement)]
        return [[tuple(axis for axis in mesh_axes(entry) if axis not in ("fsdp", "sequence"))
                 for entry in spec[:4]] for spec in specs]

    tensor = Layout().shardings(MeshSpec(tensor=4).build(jax.devices()[:8]), params)
    shared = Layout().shardings(MeshSpec(tensor=4, sequence=2).build(jax.devices()[:8]), params)
    assert others(shared) == others(tensor)
    assert any("sequence" in mesh_axes(entry)
               for sharding in jax.tree.leaves(shared) for entry in sharding.spec)


NATIVE_RULES = {"embed": ("zero", "zero2"), "mlp": "zero", "heads": "zero", "kv": "zero", "vocab": "zero",
                "linear": "zero"}
"""A layout of a native mesh's own axes: two fsdp levels, `zero` and
`zero2`, beside a `replica` axis no parameter takes."""


def native_trainer(mesh, layout, directory, **hybrid) -> Trainer:
    model = tiny(max_seq_len=LENGTH, tie_embeddings=False, **hybrid)
    return Trainer(LMObjective(model, LENGTH), optax.sgd(1e-2), key=jax.random.key(0), mesh=mesh,
                   layout=layout, checkpoints=None if directory is None else Checkpoints(str(directory)),
                   tracker=None)


def token_windows(directory):
    """Eight-row batches of windows over a token file, a stream fit
    checkpoints and resumes."""
    from dew.data import Loading, presets
    from dew.data.compile import compile
    from dew.data.sources.tokens import TokenFiles

    tokens = np.arange(400) * 7 % VOCAB
    corpus = write_token_corpus(directory, tokens, tokens, vocab_size=VOCAB)
    train = TokenFiles(corpus.train).shuffle(7).repeat().pipe(presets.token_windows, seq_len=LENGTH,
                                                               stride=1, batch=8)
    held = TokenFiles(corpus.val).pipe(presets.token_windows, seq_len=LENGTH, stride=1, batch=8, records=32)
    return compile(train, validation=held, read="memory", seed=7,
                   loading=Loading(workers=0, threads=1, read_buffer=1))


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("hybrid", [False, True], ids=["attention", "delta-net"])
def test_a_native_mesh_of_its_own_axes_trains_saves_and_resumes_as_one_device(tmp_path, hybrid):
    """A `jax.sharding.Mesh` of axes Dew does not name, two fsdp levels and
    a replica axis, under a layout whose rules name them and whose
    `data_axes` split the rows over all three: fit splits the parameters
    over both fsdp levels and the rows over all three axes, and two steps,
    a checkpoint and a resumed third step train what one device trains in
    three, as a `MeshSpec` mesh does. A rule naming an axis the mesh lacks,
    one of Dew's role axes among them, is refused before anything is placed;
    Dew's own rules, which name every role axis, place on whatever the mesh has."""
    layers = delta_layers(fused=False) if hybrid else {}
    native = Mesh(np.asarray(jax.devices()[:8]).reshape(2, 2, 2), ("replica", "zero", "zero2"))
    layout = Layout(rules=NATIVE_RULES, data_axes=("replica", "zero", "zero2"), min_shard=TINY_SHARD)
    data = token_windows(tmp_path / "tokens")

    def run(steps):
        return native_trainer(native, layout, tmp_path / "run", **layers).fit(data, steps=steps,
                                                                               checkpoint_every=2)

    assert {"zero", "zero2"} <= sharded.split_axes(run(2).variables["params"])
    resumed = run(3)
    assert int(resumed.step) == 3

    plain = Layout(min_shard=TINY_SHARD)
    whole = native_trainer(Mesh(np.asarray(jax.devices()[:1]), ("data",)), plain, None, **layers).fit(
        data, steps=3)
    spec = native_trainer(MeshSpec(fsdp=2), plain, None, **layers).fit(data, steps=3)
    for state in (resumed, spec):
        assert_trains_alike(*jax.device_get((state.variables["params"], whole.variables["params"])))

    misspelt = Layout(rules={**NATIVE_RULES, "embed": "zer0"}, data_axes=layout.data_axes)
    with pytest.raises(ValueError):
        native_trainer(native, misspelt, None, **layers).fit(data, steps=1)
    # A rule of one's own (none of Dew's: those put the width on fsdp) that names Dew's tensor axis,
    # which this mesh lacks, would split nothing.
    absent = Layout(rules={**NATIVE_RULES, "embed": "tensor"}, data_axes=layout.data_axes)
    with pytest.raises(ValueError):
        native_trainer(native, absent, None, **layers).fit(data, steps=1)


@pytest.mark.mesh(devices=8)
def test_a_meshs_explicit_axes_split_rows_and_hold_no_parameter(tmp_path):
    """`jax.make_mesh`'s axes are Explicit unless named otherwise. The rows
    split over one as their type places them and train what one device
    trains; a parameter the rules put on one is refused by name, since the
    partitioner places parameters."""
    data = token_windows(tmp_path / "tokens")
    plain = Layout(min_shard=TINY_SHARD)
    whole = native_trainer(Mesh(np.asarray(jax.devices()[:1]), ("data",)), plain, None).fit(data, steps=3)
    typed = native_trainer(jax.make_mesh((8,), ("data",), axis_types=(AxisType.Explicit,)), plain,
                           None).fit(data, steps=3)
    assert_trains_alike(*jax.device_get((typed.variables["params"], whole.variables["params"])))

    explicit = jax.make_mesh((2, 2, 2), ("replica", "zero", "zero2"), axis_types=(AxisType.Explicit,) * 3)
    layout = Layout(rules=NATIVE_RULES, data_axes=explicit.axis_names, min_shard=TINY_SHARD)
    with pytest.raises(ValueError):
        native_trainer(explicit, layout, None).fit(data, steps=1)


@pytest.mark.parametrize(("length", "tensor_shards", "heads", "slow", "fast"), [
    (1, 1, 4, 60e9, None),
    (13, 1, 4, 60e9, 80e9),
    (13, 2, 4, 75e9, 90e9),
    (13, 2, 1, 90e9, 100e9),
], ids=["one-position", "sequence", "tensor-sequence", "indivisible-heads"])
def test_cross_attention_spends_communication_only_where_the_sequence_link_pays(
        length, tensor_shards, heads, slow, fast):
    """An indivisible context's projection trades redundant FLOPs for bytes.

    Two rows of 13 positions, width 96, projected to 128 key/value features
    need 71.8 GB/s at a 1 TFLOP/s peak: each shard saves nine positions,
    not twelve, because three padded positions add work. The links straddle
    that crossover. Splitting four heads over tensor2 raises it to 78.7 GB/s;
    one head cannot split, and its smaller projection needs 92.6 GB/s.
    A one-position context saves no work even on a CPU link.
    """
    from dew.mesh import SEQUENCE_AXIS
    from dew.nn.attention import NormalAttention
    from dew.nn.sharding import Link, measured_links
    from dew.telemetry.instrumentation import compiled_flops

    model = NormalAttention(64, heads=heads, dim_head=16, attention_impl="reference")
    query = jax.random.normal(jax.random.key(21), (2, 8, 64))
    context = jax.random.normal(jax.random.key(22), (2, length, 96))
    params = model.init(jax.random.key(23), query, context)["params"]

    def compiled(sequence, tensor, link):
        mesh = MeshSpec(sequence=sequence, tensor=tensor).build(jax.devices()[:sequence * tensor])
        whole = NamedSharding(mesh, P())
        inputs = jax.tree.map(lambda value: jax.device_put(value, whole), (params, query, context))

        def loss(weights, x, tokens):
            output = model.apply({"params": weights}, x, tokens)
            return jnp.mean(jnp.square(output)), output

        with jax.set_mesh(mesh), measured_links({SEQUENCE_AXIS: link}):
            step = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True))
            executable = step.lower(*inputs).compile()
        result = jax.device_get(executable(*inputs))
        flops = compiled_flops(executable)
        assert flops is not None
        moved = sharded.collective_bytes(executable.as_text(), COLLECTIVES)
        return result, flops * sequence * tensor, moved

    expected, _, _ = compiled(1, 1, Link(None, None, "cpu"))
    kept, kept_flops, kept_bytes = compiled(4, tensor_shards, Link(slow, 1e12, "gpu"))
    link = Link(fast, 1e12, "gpu") if fast is not None else Link(None, None, "cpu")
    split, split_flops, split_bytes = compiled(4, tensor_shards, link)
    for actual in (kept, split):
        for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    if length > 1:
        assert split_flops < kept_flops
        assert split_bytes > kept_bytes
    else:
        assert split_flops == kept_flops
        assert split_bytes == kept_bytes


@pytest.mark.parametrize(("mesh", "devices"), [
    pytest.param(MeshSpec(fsdp=2), 2, marks=pytest.mark.mesh(devices=2), id="fsdp2"),
    pytest.param(MeshSpec(fsdp=2, tensor=2), 4, marks=pytest.mark.mesh(devices=4), id="fsdp2_tensor2"),
    pytest.param(MeshSpec(fsdp=2, sequence=2), 4, marks=pytest.mark.mesh(devices=4), id="fsdp2_sequence2"),
])
def test_a_frozen_collection_kept_whole_is_never_gathered_over_fsdp(mesh, devices):
    """`Layout.frozen_whole` holds a LoRA base whole over the state axes,
    fsdp and sequence, and split as the rules split it over the rest
    (tensor): the step gathers none of what they split, the optimizer state
    and the trained parameters lie as the split layout lays them, and one
    step trains what it trains."""
    from dew.batch import FROZEN
    from dew.lora import LoRA
    from dew.training.distributed import placed_bytes

    model = tiny()
    adapter = LoRA(rank=2, modules=("q_proj", "v_proj", "gate_proj", "down_proj")).apply(
        model, model.init(jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32)), key=1)
    batch = next(token_batches())

    def stepped(frozen_whole: bool):
        objective = LMObjective(adapter.model, SEQ_LEN, variables=jax.tree.map(np.asarray, adapter.variables))
        trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.key(0), mesh=mesh, checkpoints=None,
                          tracker=None, layout=Layout(min_shard=TINY_SHARD, frozen_whole=frozen_whole))
        trainer.device_mesh = mesh.build(jax.devices()[:devices])
        state, shardings, _ = trainer.place()
        placed = shard_batch(trainer.device_mesh, batch)
        run = trainer.compile(state, placed)
        gathered = sharded.collective_bytes(trainer.executable.as_text(), ("all-gather",))
        trained, *_ = run(state, placed)
        return state, shardings, gathered, jax.device_get(trained.variables["params"])

    split_state, split, split_gathered, split_trained = stepped(frozen_whole=False)
    whole_state, whole, whole_gathered, whole_trained = stepped(frozen_whole=True)
    split_specs = [sharding.spec for sharding in jax.tree.leaves(split.variables[FROZEN])]
    whole_specs = [sharding.spec for sharding in jax.tree.leaves(whole.variables[FROZEN])]
    assert any("fsdp" in mesh_axes(entry) for spec in split_specs for entry in spec)
    assert mesh.sequence == 1 or any("sequence" in mesh_axes(entry) for spec in split_specs for entry in spec)
    def unsplit(entry):
        kept = [axis for axis in mesh_axes(entry) if axis not in ("fsdp", "sequence")]
        return tuple(kept) if kept else None

    assert whole_specs == [P(*map(unsplit, spec)) for spec in split_specs]
    assert mesh.tensor == 1 or any("tensor" in mesh_axes(entry) for spec in whole_specs for entry in spec)
    assert jax.tree.leaves(whole.variables["params"]) == jax.tree.leaves(split.variables["params"])
    assert placed_bytes(whole_state.opt_state) == placed_bytes(split_state.opt_state)
    kept = placed_bytes(whole_state.variables[FROZEN]) - placed_bytes(split_state.variables[FROZEN])
    assert kept > 0 and split_gathered - whole_gathered >= kept
    for got, want in zip(jax.tree.leaves(whole_trained), jax.tree.leaves(split_trained), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


class Named(nn.Module):
    """A two-layer map whose trainable second layer is named as the frozen collection is."""

    @nn.compact
    def __call__(self, x):
        from dew.batch import FROZEN
        from dew.nn.dense import Dense

        return Dense(16, name=FROZEN)(jax.nn.relu(Dense(64, name="hidden")(x)))


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("grouped", [False, True], ids=["adam", "optax-groups"])
def test_a_module_or_optimizer_group_named_frozen_keeps_its_split_under_frozen_whole(grouped):
    """`Layout.frozen_whole` reads a leaf's collection where the variables are, so a trainable
    module or an optimizer group that is only named `frozen` keeps its fsdp split, in its
    parameters and its Adam moments: with no frozen collection the layout places every train-state
    leaf and trains one step exactly as the split layout does."""
    from dew.batch import FROZEN
    from dew.inputs import Field, InputSpec
    from dew.objectives.supervised import Supervised

    rng = np.random.default_rng(0)
    batch = {"x": rng.normal(size=(8, 32)).astype(np.float32),
             "y": rng.normal(size=(8, 16)).astype(np.float32)}
    mesh = MeshSpec(fsdp=2)

    def stepped(frozen_whole: bool):
        optimizer = optax.adam(1e-2)
        if grouped:
            optimizer = optax.multi_transform({FROZEN: optax.adam(1e-2), "hidden": optax.adam(1e-3)},
                                              lambda params: {name: name for name in params})
        objective = Supervised(Named(), lambda outputs, rows: (outputs - rows["y"]) ** 2,
                               inputs=InputSpec(Field("x", (32,))))
        trainer = Trainer(objective, optimizer, key=jax.random.key(0), mesh=mesh, checkpoints=None,
                          tracker=None, layout=Layout(min_shard=TINY_SHARD, frozen_whole=frozen_whole))
        trainer.device_mesh = mesh.build(jax.devices()[:2])
        state, shardings, _ = trainer.place()
        placed = shard_batch(trainer.device_mesh, batch)
        trained, *_ = trainer.compile(state, placed)(state, placed)
        return shardings, jax.device_get(trained.variables["params"])

    split, split_trained = stepped(frozen_whole=False)
    whole, whole_trained = stepped(frozen_whole=True)

    split_specs = [sharding.spec for sharding in jax.tree.leaves(split)]
    assert any("fsdp" in mesh_axes(entry) for sharding in jax.tree.leaves(split.opt_state)
               for entry in sharding.spec)
    assert [sharding.spec for sharding in jax.tree.leaves(whole)] == split_specs
    for got, want in zip(jax.tree.leaves(whole_trained), jax.tree.leaves(split_trained), strict=True):
        np.testing.assert_array_equal(got, want)


def token_batches():
    return repeated_token_batch(VOCAB, BATCH, SEQ_LEN + 1)


def run_losses(mesh, layout, steps, microbatches=None):
    tracker = RecordingTracker()
    trainer = Trainer(
        LMObjective(tiny(), SEQ_LEN), optax.adam(1e-3), key=jax.random.key(0),
        mesh=mesh, layout=layout, microbatches=microbatches, tracker=tracker)
    state = trainer.fit(Loader(train=lambda partition: token_batches(), val=None, records=None, batch=BATCH),
                        steps=steps, log_every=1)
    assert_sharded(state.variables["params"], trainer.device_mesh)
    return [entry["train/loss"] for _, entry in tracker.scalars if "train/loss" in entry]


def dense_layout():
    return Layout(min_shard=TINY_SHARD)


TOPOLOGIES = {
    # Four data/FSDP/tensor mesh configurations on eight simulated devices,
    # plus the sequence axis beside a data axis, where the batch rows split
    # over data and fsdp and the sequence over its own axis, and the stage
    # axis beside them: the two-layer model splits into two stages of one
    # layer, fed two microbatches of four rows, one of each device's two.
    # The default rules split the widths over tensor and the residual width
    # over fsdp, so a tensor mesh splits its parameters both ways
    # (`sharded.assert_sharded`).
    "fsdp": (MeshSpec(fsdp=8), dense_layout()),
    "tensor": (MeshSpec(fsdp=4, tensor=2), dense_layout()),
    "sequence": (MeshSpec(fsdp=4, sequence=2), dense_layout()),
    "data_sequence": (MeshSpec(fsdp=2, sequence=2), dense_layout()),
    "both": (MeshSpec(fsdp=2, tensor=2, sequence=2), dense_layout()),
    "stage": (MeshSpec(fsdp=2, stage=2), dense_layout(), 2),
    "stage_tensor": (MeshSpec(fsdp=2, tensor=2, stage=2), dense_layout(), 2),
}


def test_topologies_agree_with_data_parallel():
    """Every topology converges along the same finite loss trajectory."""
    steps = 30
    runs = {name: np.array(run_losses(*spec[:2], steps, *spec[2:])) for name, spec in TOPOLOGIES.items()}

    for losses in runs.values():
        assert len(losses) == steps and np.all(np.isfinite(losses))
        assert losses[-1] < losses[0] / 2, losses

    difference = max(
        np.max(np.abs(first - second))
        for first in runs.values() for second in runs.values())
    # Observed 9.5e-07 at most between any two of the seven topologies over
    # 30 steps on CPU. Reduction order differs per topology, and fp32
    # rounding over 30 steps on losses of order 4 is about 1e-6; a placement
    # that changed a value would show at 1e-2 or worse.
    assert difference < 4e-6, difference


def one_step(spec):
    """One step's loss and gradients under `spec`: the parameters one seed
    initialises, placed by the default rules, and one batch of the fixture.

    The step is the trainer's inner call without the optimizer: what the
    default rules place is the only thing `spec` changes between two runs of
    it, so a gradient that moved means the collectives GSPMD derived from the
    tensor axis are not the ones the rules meant.
    """
    mesh = spec.build()
    objective = LMObjective(tiny(), SEQ_LEN)
    initial = objective.init(jax.random.key(0))
    placed = jax.device_put(initial, Layout(min_shard=TINY_SHARD).shardings(mesh, initial))
    assert_sharded(placed["params"], mesh)
    batch = shard_batch(mesh, next(token_batches()))

    @jax.jit
    def step(trainable):
        def loss_fn(inner):
            return objective.scalar_loss({**placed, "params": inner}, batch,
                               Step(step=jnp.zeros((), jnp.int32),
                                    key=jax.random.key(1), ema=None))

        return jax.value_and_grad(loss_fn, has_aux=True)(trainable)

    with jax.set_mesh(mesh):
        (loss, _), grads = step(placed["params"])
    return float(loss), grads


def test_the_tensor_axis_changes_no_value():
    """Four tensor shards against none, from one seed and one batch: the same
    loss and the same gradient in every leaf."""
    whole_loss, whole_grads = one_step(MeshSpec(fsdp=2))
    split_loss, split_grads = one_step(MeshSpec(fsdp=2, tensor=4))

    assert abs(split_loss - whole_loss) < 1e-5, (whole_loss, split_loss)
    differences = jax.tree.map(
        lambda whole, split: float(np.max(np.abs(np.asarray(whole) - np.asarray(split)))),
        whole_grads, split_grads)
    # The two losses agree to the bit on CPU, and the gradients to 1.0e-07 at
    # most against magnitudes of 1.2e-01: the tensor axis reassociates one
    # reduction, nothing more. A spec that split a dimension the model reads
    # whole would show at 1e-2 or worse.
    assert max(jax.tree.leaves(differences)) < 1e-5, differences


def test_a_dit_steps_under_a_tensor_axis():
    """The DiT names 'mlp' and 'heads' on modulated blocks rather than a
    decoder's gated ones, and carries an adaLN projection and a patch
    embedding besides: two tensor shards initialise it in place and
    differentiate one forward pass over it."""
    mesh = MeshSpec(fsdp=2, tensor=2).build()
    model = SimpleDiT(patch_size=4, emb_features=32, num_layers=1, num_heads=4, mlp_ratio=2)
    images = jax.random.normal(jax.random.key(1), (BATCH, 8, 8, 3), jnp.float32)
    times = jnp.full((BATCH,), 0.5, jnp.float32)
    layout = Layout(min_shard=TINY_SHARD)
    shardings = layout.shardings(
        mesh, jax.eval_shape(model.init, jax.random.key(0), images, times))

    with jax.set_mesh(mesh):
        placed = jax.jit(model.init, out_shardings=shardings)(
            jax.random.key(0), images, times)
        loss, grads = jax.jit(jax.value_and_grad(
            lambda trainable: jnp.mean(
                (model.apply({**placed, "params": trainable}, images, times) - images) ** 2)))(
            placed["params"])

    assert np.isfinite(float(loss)) and float(loss) > 0
    assert all(np.all(np.isfinite(leaf)) for leaf in jax.tree.leaves(grads))
    assert jax.tree.all(jax.tree.map(
        lambda leaf, sharding: leaf.sharding == sharding, placed, shardings))
    layout.check(placed["params"], shardings["params"], mesh)


@pytest.mark.mesh(devices=4)
def test_a_vocabulary_split_head_leaves_the_excluded_column_out_on_every_shard():
    """The masked objective leaves its mask column out of the partition. On
    a vocabulary split the column sits in one device's rows (the last
    shard's here), and each device shifts it into its own numbering as it
    shifts the targets, so the loss and every gradient are the unsplit
    head's."""
    from dew.diffusion.discrete import MDLM
    from dew.objectives.diffusion.masked import MaskedDiffusionObjective

    model = models.build(
        "causal_transformer", vocab_size=4096, emb_features=32, num_layers=1, num_heads=4,
        num_kv_heads=2, mlp_features=64, max_seq_len=SEQ_LEN, causal=False)
    objective = MaskedDiffusionObjective(model, MDLM(mask_id=4095)(), SEQ_LEN)
    initial = objective.init(jax.random.key(0))
    text = jax.random.randint(jax.random.key(2), (BATCH, SEQ_LEN), 0, 4095)
    step = Step(step=jnp.zeros((), jnp.int32), key=jax.random.key(1), ema=None)

    def loss(params, text):
        return objective.scalar_loss({**initial, "params": params}, {"text": text}, step)[0]

    want, want_grads = jax.value_and_grad(loss)(initial["params"], text)
    mesh = MeshSpec(fsdp=4).build(jax.devices()[:4])
    shardings = Layout(min_shard=TINY_SHARD).shardings(mesh, initial)
    with jax.set_mesh(mesh):
        got, got_grads = jax.jit(jax.value_and_grad(loss))(
            jax.device_put(initial["params"], shardings["params"]),
            shard_batch(mesh, {"text": np.asarray(text)})["text"])
    np.testing.assert_allclose(got, want, rtol=1e-5)
    for expected, actual in zip(jax.tree.leaves(want_grads), jax.tree.leaves(got_grads), strict=True):
        assert jnp.abs(actual - expected).max() <= 1e-4 * jnp.abs(expected).max()


def test_the_causal_convs_taps_gradient_under_a_partly_replicated_batch():
    """Rows split over fsdp beside a tensor axis the conv's input does not
    use: the taps' gradient is one device's. XLA partitioned a grouped
    conv's filter gradient by summing it over all four devices, the two
    that hold the same rows included, which doubled it and trained the
    Mamba-2 and gated delta net convs of a hybrid on fsdp x tensor wrong."""
    from dew.nn.linear import causal_conv1d

    rng = np.random.default_rng(0)
    x, cotangent = (rng.normal(size=(4, 8, 16)).astype(np.float32) for _ in range(2))
    taps = rng.normal(size=(16, 4)).astype(np.float32)

    # The conv alone: its taps' gradient sums products of x and the
    # cotangent, whose magnitudes bound the sum's rounding. After the
    # activation the cotangent is scaled by silu', which is negative below
    # about -1.28, so that bound would not hold for |x| and |cotangent|.
    def loss(x, taps, cotangent):
        return jnp.sum(causal_conv1d(x, taps, activation=False) * cotangent)

    alone = jax.grad(loss, argnums=1)(x, taps, cotangent)
    mesh = MeshSpec(fsdp=2, tensor=2).build(jax.devices()[:4])
    rows = NamedSharding(mesh, P("fsdp"))
    with jax.set_mesh(mesh):
        split = jax.jit(jax.grad(loss, argnums=1))(
            jax.device_put(x, rows), taps, jax.device_put(cotangent, rows))

    # A tap's gradient sums one product per row and position; any order of
    # that sum lands within their count times fp32 epsilon times the sum of
    # the products' magnitudes, which the same gradient of |x| and
    # |cotangent| is. The doubled gradient missed by the whole sum.
    terms = x.shape[0] * x.shape[2]
    magnitude = jax.grad(loss, argnums=1)(np.abs(x), taps, np.abs(cotangent))
    np.testing.assert_array_less(np.abs(split - alone), terms * np.finfo(np.float32).eps * magnitude)


CONV_LAYOUTS = {
    # The UNet's upsampling 3x3 convolution with its input's batch split over
    # data and its output's image rows over sequence, the batch whole: the
    # placement a sequence-parallel layer after it gives a batch too small
    # for the data axis.
    "upsample": (Upsample(features=16, scale=2), MeshSpec(sequence=2), P("data"),
                 P(None, "sequence")),
    # The SSM DiT's depthwise fusion convolutions with only their batch split,
    # over fsdp beside a tensor axis they do not use: a hybrid mesh's plain
    # data-parallel layout.
    "depthwise": (SpatialFusionConv(features=8), MeshSpec(fsdp=2, tensor=2),
                  P(("data", "fsdp")), None),
}


@pytest.mark.parametrize("name", sorted(CONV_LAYOUTS))
def test_a_convolutions_kernel_gradient_under_a_partly_replicated_layout(name):
    """The kernel's gradient is one device's. XLA's partitioner doubled it in
    both layouts: it scales a convolution's kernel gradient by a power of two
    in several layouts where a mesh axis holds the convolution's input or
    output replicated (openxla/xla#49382). The UNets, the VAEs, the patch
    embeddings and the depthwise towers build on `dew.nn.conv.Conv`, which
    places the input and output over every mesh axis."""
    block, spec, rows, outputs = CONV_LAYOUTS[name]
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 8, 8, 8)).astype(np.float32)
    params = block.init(jax.random.key(0), x)["params"]
    shape = jax.eval_shape(block.apply, {"params": params}, x).shape
    cotangent = rng.normal(size=shape).astype(np.float32)

    def loss(variables, x, cotangent, constrained):
        out = block.apply({"params": variables}, x)
        if constrained and outputs is not None:
            out = jax.lax.with_sharding_constraint(out, outputs)
        return jnp.sum(out * cotangent)

    alone = jax.grad(loss)(params, x, cotangent, constrained=False)
    mesh = spec.build(jax.devices()[:4])
    with jax.set_mesh(mesh):
        constrained = True
        split = jax.jit(jax.grad(loss), static_argnums=3)(
            params, jax.device_put(x, NamedSharding(mesh, rows)), cotangent, constrained)

    # Each kernel entry's and bias entry's gradient sums one product per row
    # and output position, bounded as the taps' above.
    terms = shape[0] * shape[1] * shape[2]
    magnitude = jax.grad(loss)(params, np.abs(x), np.abs(cotangent), constrained=False)
    for got, want, size in zip(jax.tree.leaves(split), jax.tree.leaves(alone),
                               jax.tree.leaves(magnitude), strict=True):
        np.testing.assert_array_less(np.abs(got - want), terms * np.finfo(np.float32).eps * size)


def test_a_convolution_whose_kernel_splits_its_input_channels_computes_one_devices_output():
    """A 3x3 convolution of an 8x8 image under fsdp x tensor, its kernel's
    input channels split over fsdp as the layout stores the UNet's (fsdp
    takes the first of two equal widths), and its image rows split over
    tensor by `Conv`'s placement. XLA's partitioner computed another result
    (openxla/xla, a 3x3 convolution with a halo and a kernel split on its
    input features): the UNet's second level's conv2 came out 13.4 off an
    output of 4.5, so a UNet on fsdp2 x tensor2 trained on a loss 6.5e-3 off
    one device's and gradients up to 2.6e5 times their bound."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 8, 8, 64)).astype(np.float32)
    conv = Conv(64, (3, 3))
    params = conv.init(jax.random.key(0), x[:1])["params"]
    alone = conv.apply({"params": params}, x)
    mesh = MeshSpec(fsdp=2, tensor=2).build(jax.devices()[:4])
    with jax.set_mesh(mesh):
        stored = {"kernel": jax.device_put(params["kernel"], NamedSharding(mesh, P(None, None, "fsdp"))),
                  "bias": params["bias"]}
        split = jax.jit(conv.apply)({"params": stored}, jax.device_put(x, NamedSharding(mesh, P("fsdp"))))

    # Each output sums one product per tap and input channel; any order of
    # that sum lands within their count times fp32 epsilon times the sum of
    # the products' magnitudes, which the same convolution of |x| and
    # |kernel| is (the bias adds one term).
    terms = 3 * 3 * x.shape[-1] + 1
    magnitude = conv.apply({"params": jax.tree.map(np.abs, params)}, np.abs(x))
    np.testing.assert_array_less(np.abs(np.asarray(split) - np.asarray(alone)),
                                 terms * np.finfo(np.float32).eps * np.asarray(magnitude))


def test_a_layout_pass_at_a_kernel_calls_edge_is_found_through_views_and_fusion_bodies():
    """`sharded.boundary_copies` finds a copy feeding a Triton call through a bitcast, and a fusion
    whose body only transposes reading its result through a tuple element, whatever its name; a
    fusion that computes beside its transpose writes its result once and is not a layout pass."""
    text = """HloModule module

%only_moves (p.0: bf16[8,64]) -> bf16[64,8] {
  %p.0 = bf16[8,64]{1,0} parameter(0)
  ROOT %t.0 = bf16[64,8]{1,0} transpose(bf16[8,64]{1,0} %p.0), dimensions={1,0}
}

%computes (p.1: bf16[8,64]) -> bf16[64,8] {
  %p.1 = bf16[8,64]{1,0} parameter(0)
  %e.1 = bf16[8,64]{1,0} exponential(bf16[8,64]{1,0} %p.1)
  ROOT %t.1 = bf16[64,8]{1,0} transpose(bf16[8,64]{1,0} %e.1), dimensions={1,0}
}

ENTRY %main (x: bf16[8,64]) -> (bf16[64,8], bf16[64,8]) {
  %x = bf16[8,64]{1,0} parameter(0)
  %copy.1 = bf16[8,64]{0,1} copy(bf16[8,64]{1,0} %x)
  %view = bf16[8,64]{1,0} bitcast(bf16[8,64]{0,1} %copy.1)
  %kernel = (bf16[8,64]{1,0}, f32[8]{0}) custom-call(bf16[8,64]{1,0} %view), \
custom_call_target="__gpu$xla.gpu.triton"
  %result = bf16[8,64]{1,0} get-tuple-element((bf16[8,64]{1,0}, f32[8]{0}) %kernel), index=0
  %fusion.7 = bf16[64,8]{1,0} fusion(bf16[8,64]{1,0} %result), kind=kLoop, calls=%only_moves
  %fusion.8 = bf16[64,8]{1,0} fusion(bf16[8,64]{1,0} %result), kind=kLoop, calls=%computes
  ROOT %out = (bf16[64,8]{1,0}, bf16[64,8]{1,0}) tuple(bf16[64,8]{1,0} %fusion.7, bf16[64,8]{1,0} %fusion.8)
}
"""
    assert sharded.boundary_copies(text) == ["copy.1 -> kernel", "kernel -> fusion.7"]


def test_an_async_collective_counts_its_results_not_its_operands():
    """An async pair counts once, by its `-done`, which holds what moved: its `-start` names
    `(operands, results)`. Each result's bytes are as XLA sizes them, an fp8 element's one byte."""
    text = """ENTRY %main (x: bf16[4,8], y: f32[16], z: f32[8], w: f8e4m3fn[2,8]) -> f32[8] {
  %ag = ((bf16[4,8]{1,0}), bf16[8,8]{1,0}) all-gather-start(bf16[4,8]{1,0} %x), dimensions={0}
  %agd = bf16[8,8]{1,0} all-gather-done(((bf16[4,8]{1,0}), bf16[8,8]{1,0}) %ag)
  %rs = (f32[16]{0}, f32[8]{0}) reduce-scatter-start(f32[16]{0} %y), dimensions={0}
  %rsd = f32[8]{0} reduce-scatter-done((f32[16]{0}, f32[8]{0}) %rs)
  %ar = f32[8]{0} all-reduce(f32[8]{0} %z), to_apply=%add
  ROOT %wg = f8e4m3fn[8,8]{1,0} all-gather(f8e4m3fn[2,8]{1,0} %w), dimensions={0}
}"""
    assert sharded.collectives(text) == [("all-gather", [(8, 8)]), ("reduce-scatter", [(8,)]),
                                         ("all-reduce", [(8,)]), ("all-gather", [(8, 8)])]
    assert sharded.collective_bytes(text, COLLECTIVES) == 128 + 32 + 32 + 64
