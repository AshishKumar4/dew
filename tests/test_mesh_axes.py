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


def tiny():
    return models.build(
        "causal_transformer", vocab_size=VOCAB, emb_features=32, num_layers=2,
        num_heads=4, num_kv_heads=2, mlp_features=64, max_seq_len=SEQ_LEN)


def variables():
    return jax.eval_shape(
        tiny().init, jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32))


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


def test_topologies_agree_with_data_parallel():
    """Every topology converges along the same finite loss trajectory."""
    steps = 30
    runs = {name: np.array(run_losses(*spec, steps)) for name, spec in TOPOLOGIES.items()}

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


COLLECTIVE = re.compile(
    r"^\s*(?:ROOT\s+)?%\S+\s+=\s+(?P<shape>.+?)\s+"
    r"(?P<op>all-reduce|all-gather|reduce-scatter|all-to-all|collective-permute)(?:-start)?\(")
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
