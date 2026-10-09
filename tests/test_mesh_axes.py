"""Sequence, tensor and stage mesh axes: placement, batches, and loss equality.

The mesh carries six axes; parameters distinguish fsdp, expert and tensor,
the batch's sequence dimension rides sequence, and the layer stack's
pipeline stages ride stage. Widths stay on fsdp unless a run's rules
redirect them onto tensor, so the default mesh places as the three-axis one
did. A fit on each of the sim-mesh topologies trains the same losses:
sharding moves values, never changes them.
"""

import math
import re

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P
from recording import RecordingTracker
from sharded import assert_sharded

from dew.data import Dataset
from dew.nn.backbones.dit import SimpleDiT
from dew.nn.blocks import Upsample
from dew.nn.conv import Conv
from dew.nn.ssm import SpatialFusionConv
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.training import Layout, MeshSpec, Trainer
from dew.training.distributed import batch_shardings, shard_batch

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
# The widths a tensor run redirects off fsdp: the plan's heads, mlp and
# vocab, plus embed, the other side of those matmuls. Everything else keeps
# fsdp, so this is a partial tensor split, not a migration of the defaults.
TENSOR_RULES = {"heads": "tensor", "mlp": "tensor", "vocab": "tensor", "embed": "tensor"}


def tiny():
    return models.build(
        "causal_transformer", vocab_size=VOCAB, emb_features=32, num_layers=2,
        num_heads=4, num_kv_heads=2, mlp_features=64, max_seq_len=SEQ_LEN)


def variables():
    return jax.eval_shape(
        tiny().init, jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32))


def tensor_layout():
    rules = tuple((name, TENSOR_RULES.get(name, axes)) for name, axes in Layout().rules)
    return Layout(rules=rules, min_shard=TINY_SHARD)


def test_the_default_mesh_places_like_the_three_axis_one():
    """New axes at size 1 change no spec: widths keep fsdp, and the layout fits."""
    mesh = MeshSpec(fsdp=8).build()
    assert mesh.axis_names == ("data", "expert", "fsdp", "tensor", "sequence", "stage")
    specs = jax.tree.map(
        lambda sharding: sharding.spec,
        Layout(min_shard=TINY_SHARD).shardings(mesh, variables()))["params"]

    assert specs["embed_tokens"]["embedding"] == P("fsdp")
    assert specs["layers_0"]["self_attn"]["q_proj"]["kernel"] == P("fsdp")
    assert specs["layers_0"]["mlp"]["gate_proj"]["kernel"] == P(None, "fsdp")
    Layout(min_shard=TINY_SHARD).check(
        variables()["params"],
        Layout(min_shard=TINY_SHARD).shardings(mesh, variables())["params"], mesh)


def test_the_default_rules_split_the_megatron_widths_on_the_tensor_axis():
    """Four tensor shards beside two fsdp: the mlp's hidden width, the query
    and grouped key-value heads, the attention width o_proj reads and the
    vocabulary take the tensor axis, and the residual width the blocks pass
    between themselves does not: in each projection it takes fsdp, a split
    in two dimensions."""
    mesh = MeshSpec(fsdp=2, tensor=4).build()
    layout = Layout(min_shard=TINY_SHARD)
    specs = jax.tree.map(
        lambda sharding: sharding.spec, layout.shardings(mesh, variables()))["params"]
    attention = specs["layers_0"]["self_attn"]

    assert attention["q_proj"]["kernel"] == P("fsdp", "tensor")
    assert attention["k_proj"]["kernel"] == P("fsdp", "tensor")
    assert attention["v_proj"]["kernel"] == P("fsdp", "tensor")
    assert specs["layers_0"]["mlp"]["gate_proj"]["kernel"] == P("fsdp", "tensor")
    assert specs["layers_0"]["mlp"]["down_proj"]["kernel"] == P("tensor", "fsdp")
    assert specs["embed_tokens"]["embedding"] == P(("fsdp", "tensor"))
    # o_proj reads the heads the q/k/v projections split and writes the
    # residual stream: Megatron's row-parallel side, the attention's twin of
    # down_proj. The final norm is embed alone.
    assert attention["o_proj"]["kernel"] == P("tensor", "fsdp")
    assert specs["norm"]["scale"] == P()
    layout.check(variables()["params"],
                 layout.shardings(mesh, variables())["params"], mesh)


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


@pytest.mark.parametrize(("bytes_per_second", "peak", "platform", "spreads"), [
    (None, None, None, False),
    (5.8e9, 71e12, "gpu", False),
    (31.0e9, 71e12, "gpu", True),
    (64e9, 756e12, "gpu", False),
    (450e9, 989e12, "gpu", True),
    (None, None, "gpu", False),
    (None, None, "cpu", True),
], ids=["unmeasured", "3090-pcie", "3090-nvlink", "h100-pcie", "h100-nvlink", "unknown-gpu", "cpu"])
def test_a_down_projection_spreads_only_where_the_link_pays_for_it(bytes_per_second, peak, platform, spreads):
    """Each tensor shard computing its own tokens' down-projection saves FLOPs
    and gathers the residual's gradient and the latent back, and sums the
    projection's weight gradient over the tensor axis. At DeepSeek-V3's
    widths in bf16 (7168 into 1536 + 512 + 64) and 16384 tokens a microbatch
    that pays, even at the device's peak, where the link moves at least
    20.3 GB/s for an RTX 3090 (71 TFLOPS) and 283 GB/s for an H100 SXM
    (989 TFLOPS), 216 GB/s for the PCIe card (756 TFLOPS). The 3090
    rows are the box's measured all-gathers at 128 MiB (4x RTX 3090, PCIe
    3.0, one host): the NVLink pair pays and the PCIe pair does not. The
    H100 rows are the links' nominal rates a direction, PCIe 5.0 x16 for the
    PCIe card and NVLink 4 for the SXM one. A device the peak table does not
    name, and a step with no measured link, keep the projection on every
    token; a CPU mesh spreads."""
    from dew.nn.sharding import RESIDUAL, SPREAD, TENSOR_AXIS, Link, down_projection, measured_links

    mesh = MeshSpec(tensor=4).build(jax.devices()[:4])
    residual = jax.ShapeDtypeStruct((4, 4096, 7168), jnp.bfloat16)
    link = None if platform is None else Link(bytes_per_second, peak, platform)
    with jax.set_mesh(mesh), measured_links({} if link is None else {TENSOR_AXIS: link}):
        assert down_projection(residual, 1536 + 512 + 64) == (SPREAD if spreads else RESIDUAL)
    assert link is None or link.spread == spreads


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
    from dew.nn.attention import NormalAttention
    from dew.nn.sharding import SEQUENCE_AXIS, Link, measured_links
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
        moved = 0
        for line in executable.as_text().splitlines():
            match = COLLECTIVE.match(line)
            if match:
                moved += sum(ITEMSIZE[dtype] * math.prod(int(size) for size in dims.split(",") if size)
                             for dtype, dims in TYPED.findall(match["shape"]))
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


def test_a_tensor_only_mesh_splits_every_projection_of_the_block():
    """Tensor parallelism alone, no fsdp: every attention and mlp projection
    names the tensor axis, so the default tolerance holds with nothing but
    the norms left whole."""
    mesh = MeshSpec(tensor=4).build()
    layout = Layout(min_shard=TINY_SHARD)
    shardings = layout.shardings(mesh, variables())
    attention = jax.tree.map(lambda sharding: sharding.spec, shardings)["params"]["layers_0"]["self_attn"]

    assert attention["q_proj"]["kernel"] == P(None, "tensor")
    assert attention["o_proj"]["kernel"] == P("tensor")
    layout.check(variables()["params"], shardings["params"], mesh)


def test_redirected_widths_take_the_tensor_axis():
    """A run's rules move the big matmul dims onto tensor; the layout fits."""
    mesh = MeshSpec(fsdp=4, tensor=2).build()
    layout = tensor_layout()
    specs = jax.tree.map(
        lambda sharding: sharding.spec,
        layout.shardings(mesh, variables()))["params"]

    assert specs["embed_tokens"]["embedding"] == P("tensor")
    assert specs["layers_0"]["self_attn"]["q_proj"]["kernel"] == P("tensor")
    assert specs["layers_0"]["mlp"]["gate_proj"]["kernel"] == P(None, "tensor")
    layout.check(variables()["params"],
                 layout.shardings(mesh, variables())["params"], mesh)


def test_the_batch_sequence_dimension_takes_the_sequence_axis():
    """Sequence parallelism splits activations: rows over every other axis,
    positions over sequence."""
    mesh = MeshSpec(fsdp=4, sequence=2).build()
    batch = shard_batch(mesh, np.zeros((BATCH, SEQ_LEN), np.float32))

    assert len(batch.addressable_shards) == jax.device_count()
    assert batch.addressable_shards[0].data.shape == (BATCH // 4, SEQ_LEN // 2)


def test_a_width_the_sequence_axis_cannot_split_stays_replicated():
    """Seventeen columns over two sequence shards divide nothing, so the
    rows still split and the width replicates."""
    mesh = MeshSpec(fsdp=4, sequence=2).build()
    batch = shard_batch(mesh, np.zeros((BATCH, SEQ_LEN + 1), np.float32))

    assert batch.sharding.spec == P(("data", "expert", "fsdp"))
    assert batch.addressable_shards[0].data.shape == (BATCH // 4, SEQ_LEN + 1)


def test_an_image_batch_never_takes_the_sequence_axis():
    """Only a sequence per row splits over the sequence axis. An image's
    second dimension is its height, so a rank-4 leaf keeps every dimension
    but its rows whole, and a global array is placed from its shape alone."""
    mesh = MeshSpec(fsdp=4, sequence=2).build()
    images = np.zeros((BATCH, 8, 8, 3), np.float32)
    batch = shard_batch(mesh, {"image": images, "label": np.zeros((BATCH,), np.int32)})

    assert batch["image"].sharding.spec == P(("data", "expert", "fsdp"))
    assert batch["label"].sharding.spec == P(("data", "expert", "fsdp"))
    assert batch_shardings(mesh, batch)["image"].spec == batch["image"].sharding.spec


def test_mesh_build_rejects_sizes_the_devices_cannot_hold():
    """The refusal names the axes the spec splits and their product against
    the devices, not the axes it leaves at one."""
    with pytest.raises(
        ValueError,
        match=rf"^fsdp 4 x tensor 2 x sequence 3 is 24 devices a data "
        rf"replica, which does not divide the {jax.device_count()} devices$",
    ):
        MeshSpec(fsdp=4, tensor=2, sequence=3).build()


def token_batches():
    rng = np.random.default_rng(0)
    batch = {"text": rng.integers(0, VOCAB, size=(BATCH, SEQ_LEN + 1)).astype(np.int32)}
    while True:
        yield batch


def run_losses(mesh, layout, steps):
    tracker = RecordingTracker()
    trainer = Trainer(
        LMObjective(tiny(), SEQ_LEN), optax.adam(1e-3), key=jax.random.key(0),
        mesh=mesh, layout=layout, tracker=tracker)
    state = trainer.fit(Dataset(train=lambda partition: token_batches(), val=None, records=None, batch=BATCH),
                        steps=steps, log_every=1)
    assert_sharded(state.variables["params"], trainer.device_mesh)
    return [entry["train/loss"] for _, entry in tracker.scalars if "train/loss" in entry]


def dense_layout():
    return Layout(min_shard=TINY_SHARD)


TOPOLOGIES = {
    # plan.md 4.5's four mesh configs, on the eight-device simulated mesh,
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
    "stage": (MeshSpec(fsdp=2, stage=2, microbatches=2), dense_layout()),
    "stage_tensor": (MeshSpec(fsdp=2, tensor=2, stage=2, microbatches=2), dense_layout()),
}


@pytest.mark.parametrize("name", sorted(TOPOLOGIES))
def test_every_topology_trains_the_same_losses(name):
    """Thirty steps on each topology: finite, descending, and equal to 1e-6."""
    steps = 30
    losses = run_losses(*TOPOLOGIES[name], steps)

    assert len(losses) == steps and np.all(np.isfinite(losses))
    assert losses[-1] < losses[0] / 2, losses


def test_topologies_agree_with_data_parallel():
    """The largest difference across the topologies, with its number."""
    steps = 30
    runs = {name: np.array(run_losses(*spec, steps)) for name, spec in TOPOLOGIES.items()}

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


COLLECTIVE = re.compile(
    r"^\s*(?:ROOT\s+)?%\S+\s+=\s+(?P<shape>.+?)\s+"
    r"(?P<op>all-reduce|all-gather|reduce-scatter|all-to-all|collective-permute)(?:-start)?\(")
ARRAY = re.compile(r"[a-z]+\d*\[([\d,]*)\]")
TYPED = re.compile(r"([a-z]+\d*)\[([\d,]*)\]")
ITEMSIZE = {
    "pred": 1,
    "s8": 1,
    "u8": 1,
    "bf16": 2,
    "f16": 2,
    "f32": 4,
    "s32": 4,
    "u32": 4,
    "f64": 8,
    "s64": 8,
}


def collectives(spec, model):
    """`(op, result shapes)` of every collective one loss-and-gradient step of
    an LM objective over `model` compiles to under `spec` on four devices."""
    found = []
    for line in step_text(spec, model).splitlines():
        match = COLLECTIVE.match(line)
        if match:
            shapes = [tuple(int(size) for size in dims.split(",") if size)
                      for dims in ARRAY.findall(match["shape"])]
            found.append((match["op"], shapes))
    return found


def collective_bytes(spec, model, ops):
    """Bytes of the results of every collective among `ops` the step
    compiles to: a measure no combining of collectives into one, or
    flattening of their operands, changes."""
    total = 0
    for line in step_text(spec, model).splitlines():
        match = COLLECTIVE.match(line)
        if match and match["op"] in ops:
            total += sum(ITEMSIZE[dtype] * math.prod(int(size) for size in dims.split(",") if size)
                         for dtype, dims in TYPED.findall(match["shape"]))
    return total


def step_text(spec, model):
    """The compiled text of one loss-and-gradient step of an LM objective
    over `model` under `spec` on four devices."""
    mesh = spec.build(jax.devices()[:4])
    objective = LMObjective(model, SEQ_LEN)
    initial = jax.eval_shape(objective.init, jax.random.key(0))
    shardings = Layout(min_shard=TINY_SHARD).shardings(mesh, initial)
    tokens = jax.ShapeDtypeStruct(
        (BATCH, SEQ_LEN + 1),
        jnp.int32,
        sharding=batch_shardings(mesh, {"text": np.zeros((BATCH, SEQ_LEN + 1))})["text"],
    )

    def loss(variables, rest, text):
        return objective.scalar_loss({**rest, "params": variables}, {"text": text},
                           Step(step=jnp.zeros((), jnp.int32), key=jax.random.key(1), ema=None))[0]

    placed = jax.tree.map(
        lambda leaf, sharding: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding),
        initial,
        shardings,
    )
    rest = {name: value for name, value in placed.items() if name != "params"}
    with jax.set_mesh(mesh):
        # The gradient lands where its parameter is, as the optimizer reads it.
        return jax.jit(jax.grad(loss), out_shardings=shardings["params"]).lower(
            placed["params"], rest, tokens).compile().as_text()


def wide():
    """A decoder wide enough that GSPMD, placing activations from the weights,
    splits their width over four fsdp shards; no weight is as long as the
    batch's 128 tokens."""
    return models.build(
        "causal_transformer", vocab_size=VOCAB, emb_features=96, num_layers=2,
        num_heads=8, num_kv_heads=4, mlp_features=192, max_seq_len=SEQ_LEN)


def test_fsdp_sums_no_activation_across_devices():
    """Fully sharded data parallelism gathers each weight and splits the rows.
    GSPMD, left to place the activations from the weights around them, split
    the residual width where fsdp splits the matrices and all-reduced every
    projection's partial products instead: arrays of a batch's rows and
    positions, where the only sums fsdp needs are gradients, of matrices at
    most. (The CPU backend all-reduces the gradients whole and slices them
    rather than reduce-scattering them, so their shapes say nothing here.)"""
    summed = [shape for op, shapes in collectives(MeshSpec(fsdp=4), wide())
              if op == "all-reduce" for shape in shapes]

    assert summed and all(len(shape) <= 2 for shape in summed), summed


def test_the_loss_scores_each_devices_own_tokens():
    """The cross entropy walks its tokens in tiles, and GSPMD computed those
    on the whole of every token's hidden state, gathered onto every device:
    the head's work times the device count. Scored on each device's own
    rows, only the head and its gradient cross devices."""
    tokens = BATCH * SEQ_LEN
    gathered = [shape for op, shapes in collectives(MeshSpec(fsdp=4), wide())
                if op == "all-gather" for shape in shapes]

    assert not [shape for shape in gathered
                if shape[:2] == (BATCH, SEQ_LEN) or shape[:1] == (tokens,)], gathered


@pytest.mark.mesh(devices=4)
def test_the_token_lookups_gradient_sums_no_table_under_data_parallelism():
    """Data parallelism has to sum a gradient across devices only where the
    devices' shares of it differ, and a token lookup's gradient is decided
    by the batch's rows: 128 here against a table of 4096. GSPMD scattered
    each device's rows into a table-sized buffer and all-reduced the buffers
    every step, at the dense bench's shape (16 x 1024 tokens, Qwen3's 151936
    rows) nineteen times the rows' bytes; the rows travel instead. The head
    is untied, so every other gradient, its own included, is summed once:
    the sums hold every gradient's bytes but the table's. Measured in bytes,
    since the GPU compiler combines the sums into buffers of its own."""
    model = models.build(
        "causal_transformer", vocab_size=4096, emb_features=32, num_layers=1,
        num_heads=4, num_kv_heads=2, mlp_features=64, max_seq_len=SEQ_LEN,
        tie_embeddings=False)
    shapes = jax.eval_shape(LMObjective(model, SEQ_LEN).init, jax.random.key(0))["params"]
    gradients = sum(leaf.size * leaf.dtype.itemsize for leaf in jax.tree.leaves(shapes))
    table = 4096 * 32 * 4
    summed = collective_bytes(MeshSpec(), model, {"all-reduce"})

    assert summed < gradients - table // 2, (summed, gradients, table)


@pytest.mark.mesh(devices=4)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_a_vocabulary_split_head_keeps_its_table_where_it_is(dtype):
    """fsdp splits the tied table's vocabulary over the devices that split
    the tokens. The loss gathered the table whole onto every device and
    reduce-scattered its gradient, twice the table's bytes a step, where the
    batch's 128 states are a thirty-second of its 4096 rows: each device
    now scores every token against its own rows, and the table and its
    gradient stay where they are: what moves is the other weights' shards and
    the tokens', well under the table's bytes. Measured in bytes, since the
    GPU compiler combines collectives into buffers of its own.

    Under a bf16 policy the states' gradient comes back reduce-scattered in
    bf16, which XLA's CPU compiler widens to fp32: it aborted there ("Invalid
    binary instruction opcode copy") on the reducer JAX lowers inside a map
    that leaves axes automatic."""
    model = models.build("causal_transformer", {
        "vocab_size": 4096, "emb_features": 32, "num_layers": 1, "num_heads": 4, "num_kv_heads": 2,
        "mlp_features": 64, "max_seq_len": SEQ_LEN}, dtype=dtype, attention_impl="xla")
    table = 4096 * 32 * 4
    moved = collective_bytes(MeshSpec(fsdp=4), model, {"all-gather", "reduce-scatter", "all-reduce"})

    assert moved < table // 2, (moved, table)


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


def test_tensor_parallelism_keeps_every_projection_weight_in_place():
    """Megatron's split computes each projection on the shard of the weight a
    device holds and sums the row-parallel outputs; it never gathers a
    projection's weight. Without the activations placed, GSPMD compiled the
    tensor axis to the collectives of fsdp: every weight gathered whole. Every
    head a shard of its own, and a vocabulary no projection's shape shares."""
    model = models.build(
        "causal_transformer", vocab_size=96, emb_features=32, num_layers=2,
        num_heads=4, num_kv_heads=4, mlp_features=64, max_seq_len=SEQ_LEN)
    projections = {(32, 32), (32, 64), (64, 32)}
    gathered = [shape for op, shapes in collectives(MeshSpec(tensor=4), model)
                if op == "all-gather" for shape in shapes]

    assert not projections & set(gathered), gathered


def test_a_pipeline_moves_no_microbatch_between_the_batch_shards():
    """The pipeline cuts the batch into microbatches. Cut into runs of
    consecutive rows, each microbatch sat in one fsdp shard of the rows, so
    GSPMD all-gathered the whole batch's hidden states to split every
    microbatch over the shards again. Strided, each keeps its rows where
    they are."""
    hidden = BATCH * SEQ_LEN * 32
    moved = [shape for op, shapes in collectives(MeshSpec(fsdp=2, stage=2, microbatches=2), tiny())
             if op in ("all-gather", "all-to-all") for shape in shapes]

    assert not [shape for shape in moved if math.prod(shape) >= hidden], moved


def test_the_causal_convs_taps_gradient_under_a_partly_replicated_batch():
    """Rows split over fsdp beside a tensor axis the conv's input does not
    use: the taps' gradient is one device's. XLA partitioned a grouped
    conv's filter gradient by summing it over all four devices, the two
    that hold the same rows included, which doubled it and trained the
    Mamba-2 and gated delta net convs of a hybrid on fsdp x tensor wrong."""
    from dew.nn.linear import causal_conv1d

    rng = np.random.default_rng(0)
    x, cotangent = (rng.normal(size=(4, 16, 8)).astype(np.float32) for _ in range(2))
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


@pytest.mark.parametrize("layout", [Layout(min_shard=TINY_SHARD),
                                    Layout(host=("variables",), min_shard=1, tolerance=1.)],
                         ids=["resident", "host"])
def test_a_stage_axis_under_a_model_with_no_pipeline_is_refused(layout):
    """A DiT has no layer stack to pipeline: on a stage axis of two every
    stage computed the whole step, twice the work for the same result, and
    the run said nothing. The step refuses it when it traces, whether the
    devices or the host own the state."""
    from dew.diffusion import presets
    from dew.inputs import Field, InputSpec
    from dew.objectives.diffusion import DiffusionObjective

    objective = DiffusionObjective(
        SimpleDiT(patch_size=4, emb_features=32, num_layers=1, num_heads=4, mlp_ratio=2),
        presets.Flow(), InputSpec(Field("image", (8, 8, 3))), guidance=None, steps=2)
    trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.key(0),
                      mesh=MeshSpec(fsdp=2, stage=2), layout=layout,
                      checkpoints=None, tracker=None)
    state, _, _ = trainer.place()
    batch = shard_batch(trainer.device_mesh, {"image": np.zeros((BATCH, 8, 8, 3), np.float32)})

    with pytest.raises(ValueError, match="runs no pipeline, so every stage would compute"):
        trainer.compile(state, batch)
