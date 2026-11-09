"""What a checkpoint's save writes of the arrays a run's devices hold: an array its replicas each hold
whole, written by one of them below `REPLICA_SLICE_BYTES` a replica; and a key of any implementation,
whose stored data a restore types as the key it was (row 3512)."""

import jax
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec

from dew.checkpoints import REPLICA_SLICE_BYTES, Checkpoints
from dew.telemetry.compiles import Compiles


@pytest.mark.mesh
def test_a_small_replicated_array_is_saved_whole_by_one_replica(tmp_path):
    """An array every device holds whole, smaller than `REPLICA_SLICE_BYTES` a replica, is written by
    one replica, so its save compiles no program: Orbax's standard handler cuts every replicated array
    across its replicas however small, a slice program each (on eight CPU devices an rCM UNet's first
    save took 11.7 s against 3.6 s with slices of at least 1 MiB). The same save of the array cut over
    the devices, which holds no replica to slice, compiles what any save does first."""
    mesh = jax.make_mesh((jax.device_count(),), ("data",))
    values = np.arange(jax.device_count() * 37, dtype=np.float32).reshape(jax.device_count(), 37)
    assert values.nbytes < REPLICA_SLICE_BYTES
    cut = jax.device_put(values, NamedSharding(mesh, PartitionSpec("data")))
    replicated = jax.device_put(values, NamedSharding(mesh, PartitionSpec()))
    with Checkpoints(str(tmp_path)) as checkpoints:
        checkpoints.save(1, {"weights": cut})
        checkpoints.wait()
        with Compiles(None) as heard:
            checkpoints.save(2, {"weights": replicated})
            checkpoints.wait()
        restored, _ = checkpoints.restore({"weights": replicated}, step=2)
    assert [program.name for program in heard.programs if program.kind == "compiled"] == []
    np.testing.assert_array_equal(np.asarray(restored["weights"]), values)


@pytest.mark.parametrize("impl", ["threefry2x32", "rbg"])
def test_a_key_restores_as_the_key_of_its_own_implementation(tmp_path, impl):
    """A key's stored data restores as a key of the implementation it was saved in (rbg's four words,
    threefry's two), from a template that states it only by its dtype, as a resumed run's is: the
    step records each key's implementation, which the handler Dew saves with keeps no store of."""
    key = jax.random.key(3, impl=impl)
    with Checkpoints(str(tmp_path)) as checkpoints:
        checkpoints.save(1, {"key": key})
        checkpoints.wait()
        restored, _ = checkpoints.restore({"key": jax.ShapeDtypeStruct(key.shape, key.dtype)}, step=1)
    assert restored["key"].dtype == key.dtype
    np.testing.assert_array_equal(np.asarray(jax.random.key_data(restored["key"])),
                                  np.asarray(jax.random.key_data(key)))
