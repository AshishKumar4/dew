"""A load onto a mesh builds each leaf from the mapped checkpoint, one device shard at a time.

`load_pretrained(..., mesh=...)` translates the checkpoint into `SourceLeaf`
recipes and lands each straight on its sharding through
`jax.make_array_from_callback`. What a consumer can observe is that the
placed values are the host load's values under the layout's sharding, and
that no read the placement makes spans more than one device's shard: a
read of a whole leaf is exactly the gather into one host tree this path
exists to avoid.
"""
from pathlib import Path

import jax
import numpy as np
import pytest

from dew.interop import load_pretrained
from dew.interop.streaming import SourceLeaf
from dew.training import Layout, MeshSpec

FIXTURES = Path(__file__).parent / "fixtures" / "hf"

# Tiny leaves stay replicated under the default threshold, so a threshold of
# one element shards every leaf the mesh divides, and the tolerance lets the
# ones it does not divide stay whole.
LAYOUT = Layout(min_shard=1, tolerance=1.0)


@pytest.mark.mesh
@pytest.mark.parametrize("fixture", ["qwen3-moe-tiny", "qwen35-tiny-mm"])
@pytest.mark.parametrize("param_dtype", ["float32", "bfloat16"])
def test_a_mesh_load_lands_the_host_values_and_reads_at_most_one_shard(fixture, param_dtype, monkeypatch):
    """Qwen3-MoE stacks per-expert tensors and transposes every kernel; the
    Qwen 3.5 wrapper streams its language model beside its host-built tower.
    Every placed leaf equals the host load's, carries the layout's sharding,
    and every read is one device's shard of it."""
    host = load_pretrained(FIXTURES / fixture, dtype="float32", param_dtype=param_dtype, attention_impl="xla")
    reads: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    read = SourceLeaf.read

    def recorded(leaf: SourceLeaf, index: tuple[slice, ...] | None = None) -> np.ndarray:
        value = read(leaf, index)
        reads.append((leaf.shape, value.shape))
        return value

    monkeypatch.setattr(SourceLeaf, "read", recorded)
    placed = load_pretrained(FIXTURES / fixture, dtype="float32", param_dtype=param_dtype,
                             attention_impl="xla", mesh=MeshSpec(fsdp=jax.device_count()), layout=LAYOUT)

    placed_leaves = jax.tree_util.tree_leaves_with_path(placed.variables)
    host_leaves = jax.tree_util.tree_leaves_with_path(host.variables)
    assert [path for path, _ in placed_leaves] == [path for path, _ in host_leaves]
    shards = set()
    for (path, value), (_, expected) in zip(placed_leaves, host_leaves, strict=True):
        assert isinstance(value, jax.Array), jax.tree_util.keystr(path)
        assert value.dtype == expected.dtype, jax.tree_util.keystr(path)
        np.testing.assert_array_equal(np.asarray(value), expected, err_msg=jax.tree_util.keystr(path))
        shards.add((value.shape, value.sharding.shard_shape(value.shape)))
    assert any(shard != shape for shape, shard in shards)
    assert reads and set(reads) <= shards, sorted(set(reads) - shards)[:3]


def test_a_stacked_transposed_leaf_reads_any_block_as_the_whole_leaf_holds_it():
    """The per-shard read of a stacked, transposed, cast leaf is the same
    block of the leaf read whole, for blocks on every axis."""
    rng = np.random.default_rng(0)
    members = tuple(rng.standard_normal((6, 4)).astype(np.float32) for _ in range(3))
    leaf = SourceLeaf.stack([SourceLeaf((member,), np.dtype(np.float16), transposed=True)
                             for member in members], "experts")
    whole = np.stack([member.T for member in members]).astype(np.float16)
    assert leaf.shape == whole.shape == (3, 4, 6)
    np.testing.assert_array_equal(leaf.read(), whole)
    for index in [(slice(1, 3), slice(None), slice(2, 4)), (slice(0, 1), slice(1, 3), slice(None)),
                  (slice(None), slice(3, 4), slice(0, 6))]:
        np.testing.assert_array_equal(leaf.read(index), whole[index])


@pytest.mark.mesh
def test_a_host_source_is_placed_holding_one_device_shard_at_a_time():
    """Each shard a `HostSource` reads is on its device and released before
    the next is read, so the host never holds two of a leaf's shards; a
    placement that read every shard first would hold all eight."""
    import gc
    import weakref

    from jax.sharding import NamedSharding, PartitionSpec

    from dew.training.host import place_leaf

    class Recorded:
        def __init__(self, value: np.ndarray):
            self.value, self.shape, self.ndim = value, value.shape, value.ndim
            self.held: list[weakref.ref] = []
            self.most = 0

        def read(self, index):
            gc.collect()
            self.held = [ref for ref in self.held if ref() is not None]
            self.most = max(self.most, len(self.held) + 1)
            part = np.array(self.value[index])
            self.held.append(weakref.ref(part))
            return part

        def release(self):
            pass

    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("fsdp",))
    source = Recorded(np.arange(jax.device_count() * 256, dtype=np.float32).reshape(jax.device_count(), 256))
    placed = place_leaf(source, NamedSharding(mesh, PartitionSpec("fsdp")))
    np.testing.assert_array_equal(np.asarray(placed), source.value)
    assert source.most == 1
