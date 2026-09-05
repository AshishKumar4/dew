"""Sequence and tensor mesh axes: placement, batches, and loss equality.

The mesh carries five axes; parameters distinguish fsdp, expert and tensor,
and the batch's sequence dimension rides sequence. Widths stay on fsdp
unless a run's rules redirect them onto tensor, so the default mesh places
exactly as the three-axis one did. A fit on each of the four sim-mesh
topologies trains the same losses: sharding moves values, never changes
them.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

# Needs the eight simulated CPU devices conftest configures; the GPU lane skips it.
pytestmark = pytest.mark.mesh
from jax.sharding import PartitionSpec as P

import dew.nn.backbones.causal_transformer  # registers the model built below
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.training import Layout, MeshSpec, Trainer, build_mesh
from dew.training.distributed import batch_shardings, shard_batch
VOCAB = 64
# Training batches carry seq_len + 1 columns for the one-token shift, and a
# sequence-sharded batch needs that width to divide: 15 + 1 splits over two
# sequence shards. A width that does not divide stays replicated instead of
# failing, the batch analogue of the layout dropping an indivisible name.
SEQ_LEN = 15
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
    assert mesh.axis_names == ("data", "expert", "fsdp", "tensor", "sequence")
    specs = jax.tree.map(
        lambda sharding: sharding.spec,
        Layout(min_shard=TINY_SHARD).shardings(mesh, variables()))["params"]

    assert specs["embed_tokens"]["embedding"] == P("fsdp")
    assert specs["layers_0"]["self_attn"]["q_proj"]["kernel"] == P("fsdp")
    assert specs["layers_0"]["mlp"]["gate_proj"]["kernel"] == P(None, "fsdp")
    Layout(min_shard=TINY_SHARD).check(
        variables()["params"],
        Layout(min_shard=TINY_SHARD).shardings(mesh, variables())["params"], mesh)


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
    batch = shard_batch(mesh, np.zeros((BATCH, SEQ_LEN + 1), np.float32))

    assert len(batch.addressable_shards) == jax.device_count()
    assert batch.addressable_shards[0].data.shape == (BATCH // 4, (SEQ_LEN + 1) // 2)


def test_a_width_the_sequence_axis_cannot_split_stays_replicated():
    """Seventeen columns over two sequence shards divide nothing, so the
    rows still split and the width replicates instead of failing."""
    mesh = build_mesh(MeshSpec(fsdp=4, sequence=2))
    batch = shard_batch(mesh, np.zeros((BATCH, SEQ_LEN + 2), np.float32))

    assert batch.sharding.spec == P(("data", "expert", "fsdp", "tensor"))
    assert batch.addressable_shards[0].data.shape == (BATCH // 4, SEQ_LEN + 2)


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


class Data:
    def __init__(self, train):
        self._train, self.val, self.batch, self.records = train, None, BATCH, None

    def train(self):
        return self._train()

    steps_per_epoch = None


class RecordingTracker:
    def __init__(self):
        self.scalars = []

    def log(self, scalars, step):
        self.scalars.append(dict(scalars))

    def artifact(self, value, step):
        pass


def run_losses(mesh, layout, steps):
    trainer = Trainer(
        LMObjective(tiny(), SEQ_LEN), optax.adam(1e-3), key=jax.random.key(0),
        mesh=mesh, layout=layout, tracker=RecordingTracker())
    trainer.fit(Data(token_batches), steps=steps, log_every=1)
    return [entry["train/loss"] for entry in trainer.tracker.scalars]


def dense_layout():
    return Layout(min_shard=TINY_SHARD)


TOPOLOGIES = {
    # plan.md 4.5's four mesh configs, on the eight-device simulated mesh.
    "fsdp": (MeshSpec(fsdp=8), dense_layout()),
    "tensor": (MeshSpec(fsdp=4, tensor=2), tensor_layout()),
    "sequence": (MeshSpec(fsdp=4, sequence=2), dense_layout()),
    "both": (MeshSpec(fsdp=2, tensor=2, sequence=2), tensor_layout()),
}


@pytest.mark.parametrize("name", sorted(TOPOLOGIES))
def test_every_topology_trains_the_same_losses(name):
    """Thirty steps on each topology: finite, descending, and equal to 1e-6."""
    steps = 30
    losses = run_losses(*TOPOLOGIES[name], steps)

    assert len(losses) == steps and np.all(np.isfinite(losses))
    assert losses[-1] < losses[0] / 2, losses


def test_topologies_agree_with_data_parallel():
    """The largest difference across the four topologies, with its number."""
    steps = 30
    runs = {name: np.array(run_losses(*spec, steps)) for name, spec in TOPOLOGIES.items()}

    difference = max(
        np.max(np.abs(first - second))
        for first in runs.values() for second in runs.values())
    # Observed 0.0 across all four topologies and all 30 steps on CPU; the
    # tolerance is 1e-6 because a different collective order on another
    # backend is allowed to round differently.
    assert difference < 1e-6, difference
