"""Sequence, tensor and stage mesh axes: placement, batches, and loss equality.

The mesh carries six axes; parameters distinguish fsdp, expert and tensor,
the batch's sequence dimension rides sequence, and the layer stack's
pipeline stages ride stage. Widths stay on fsdp unless a run's rules
redirect them onto tensor, so the default mesh places as the three-axis one
did. A fit on each of the sim-mesh topologies trains the same losses:
sharding moves values, never changes them.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

# Needs the eight simulated CPU devices conftest configures; the GPU lane skips it.
pytestmark = pytest.mark.mesh
from jax.sharding import PartitionSpec as P

from dew.data import Dataset
from dew.nn.backbones.dit import SimpleDiT
from dew.objectives.base import Step, scalar_loss
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.training import Layout, MeshSpec, Trainer, build_mesh
from dew.training.distributed import batch_shardings, shard_batch

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
    rules = dict(Layout().rules)
    rules.update(TENSOR_RULES)
    return Layout(rules=rules, min_shard=TINY_SHARD)


def test_the_default_mesh_places_like_the_three_axis_one():
    """New axes at size 1 change no spec: widths keep fsdp, and the layout fits."""
    mesh = build_mesh(MeshSpec(fsdp=8))
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
    and grouped key-value heads and the vocabulary take the tensor axis, and
    the residual width the blocks pass between themselves does not."""
    mesh = build_mesh(MeshSpec(fsdp=2, tensor=4))
    layout = Layout(min_shard=TINY_SHARD)
    specs = jax.tree.map(
        lambda sharding: sharding.spec, layout.shardings(mesh, variables()))["params"]
    attention = specs["layers_0"]["self_attn"]

    assert attention["q_proj"]["kernel"] == P("fsdp", "tensor")
    assert attention["k_proj"]["kernel"] == P("fsdp", "tensor")
    assert attention["v_proj"]["kernel"] == P("fsdp", "tensor")
    assert specs["layers_0"]["mlp"]["gate_proj"]["kernel"] == P(None, ("fsdp", "tensor"))
    assert specs["layers_0"]["mlp"]["down_proj"]["kernel"] == P(("fsdp", "tensor"))
    assert specs["embed_tokens"]["embedding"] == P(("fsdp", "tensor"))
    # o_proj reads the attention width and writes the residual stream, and the
    # width is the dimension 'embed' already took fsdp for, so neither side of
    # this kernel names the tensor axis. The final norm is embed alone.
    assert attention["o_proj"]["kernel"] == P("fsdp")
    assert specs["norm"]["scale"] == P()
    layout.check(variables()["params"],
                 layout.shardings(mesh, variables())["params"], mesh)


def test_redirected_widths_take_the_tensor_axis():
    """A run's rules move the big matmul dims onto tensor; the layout fits."""
    mesh = build_mesh(MeshSpec(fsdp=4, tensor=2))
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
    mesh = build_mesh(MeshSpec(fsdp=4, sequence=2))
    batch = shard_batch(mesh, np.zeros((BATCH, SEQ_LEN), np.float32))

    assert len(batch.addressable_shards) == jax.device_count()
    assert batch.addressable_shards[0].data.shape == (BATCH // 4, SEQ_LEN // 2)


def test_a_width_the_sequence_axis_cannot_split_stays_replicated():
    """Seventeen columns over two sequence shards divide nothing, so the
    rows still split and the width replicates."""
    mesh = build_mesh(MeshSpec(fsdp=4, sequence=2))
    batch = shard_batch(mesh, np.zeros((BATCH, SEQ_LEN + 1), np.float32))

    assert batch.sharding.spec == P(("data", "expert", "fsdp", "tensor"))
    assert batch.addressable_shards[0].data.shape == (BATCH // 4, SEQ_LEN + 1)


def test_an_image_batch_never_takes_the_sequence_axis():
    """Only a sequence per row splits over the sequence axis. An image's
    second dimension is its height, so a rank-4 leaf keeps every dimension
    but its rows whole, and a global array is placed from its shape alone."""
    mesh = build_mesh(MeshSpec(fsdp=4, sequence=2))
    images = np.zeros((BATCH, 8, 8, 3), np.float32)
    batch = shard_batch(mesh, {"image": images, "label": np.zeros((BATCH,), np.int32)})

    assert batch["image"].sharding.spec == P(("data", "expert", "fsdp", "tensor"))
    assert batch["label"].sharding.spec == P(("data", "expert", "fsdp", "tensor"))
    assert batch_shardings(mesh, batch)["image"].spec == batch["image"].sharding.spec


def test_build_mesh_rejects_sizes_the_devices_cannot_hold():
    with pytest.raises(ValueError, match="sequence 3"):
        build_mesh(MeshSpec(fsdp=4, tensor=2, sequence=3))


def token_batches():
    rng = np.random.default_rng(0)
    batch = {"text": rng.integers(0, VOCAB, size=(BATCH, SEQ_LEN + 1)).astype(np.int32)}
    while True:
        yield batch


class RecordingTracker:
    def __init__(self):
        self.scalars = []

    def log(self, scalars, step):
        self.scalars.append(dict(scalars))

    def artifact(self, value, step):
        pass


def run_losses(mesh, layout, steps):
    tracker = RecordingTracker()
    trainer = Trainer(
        LMObjective(tiny(), SEQ_LEN), optax.adam(1e-3), key=jax.random.key(0),
        mesh=mesh, layout=layout, tracker=tracker)
    trainer.fit(Dataset(train=token_batches, val=None, records=None, batch=BATCH),
                steps=steps, log_every=1)
    return [entry["train/loss"] for entry in tracker.scalars if "train/loss" in entry]


def dense_layout():
    return Layout(min_shard=TINY_SHARD)


TOPOLOGIES = {
    # plan.md 4.5's four mesh configs, on the eight-device simulated mesh,
    # plus the sequence axis beside a data axis, where the batch rows split
    # over data and fsdp and the sequence over its own axis, and the stage
    # axis beside them: the two-layer model splits into two stages of one
    # layer, fed four microbatches of two rows.
    "fsdp": (MeshSpec(fsdp=8), dense_layout()),
    "tensor": (MeshSpec(fsdp=4, tensor=2), tensor_layout()),
    "sequence": (MeshSpec(fsdp=4, sequence=2), dense_layout()),
    "data_sequence": (MeshSpec(fsdp=2, sequence=2), dense_layout()),
    "both": (MeshSpec(fsdp=2, tensor=2, sequence=2), tensor_layout()),
    "stage": (MeshSpec(fsdp=2, stage=2, microbatches=4), dense_layout()),
    "stage_tensor": (MeshSpec(fsdp=2, tensor=2, stage=2, microbatches=2), tensor_layout()),
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
    mesh = build_mesh(spec)
    objective = LMObjective(tiny(), SEQ_LEN)
    initial = objective.init(jax.random.key(0))
    placed = jax.device_put(initial, Layout(min_shard=TINY_SHARD).shardings(mesh, initial))
    batch = shard_batch(mesh, next(token_batches()))

    @jax.jit
    def step(trainable):
        def loss_fn(inner):
            return scalar_loss(objective, {**placed, "params": inner}, batch,
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
    mesh = build_mesh(MeshSpec(fsdp=2, tensor=2))
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
