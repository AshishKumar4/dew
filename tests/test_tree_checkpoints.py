"""A mapping saved in place of a train state checkpoints, resumes and is retained as a train state is."""
import contextlib
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from dew.checkpoints import Checkpoints, Keep, Ranking

TOUCHES = frozenset({"open", "os.listdir", "os.scandir", "os.mkdir", "os.rename", "os.remove", "os.rmdir",
                     "os.truncate", "shutil.rmtree", "shutil.copyfile", "shutil.move"})
"""The interpreter's audit events a store on the local filesystem is read and written through."""

_watched: list | None = None


def _audited(event: str, arguments: tuple) -> None:
    if _watched is not None and event in TOUCHES:
        _watched.append((event, arguments))


sys.addaudithook(_audited)


@contextlib.contextmanager
def store_touches(root):
    """Every read and write of the store under `root` while held, as the interpreter's audit events
    record them at the filesystem's boundary (`TOUCHES`): `(event, path)` for each, nothing patched."""
    global _watched
    root, seen = os.fspath(root), []
    touched: list[tuple[str, str]] = []
    _watched = seen
    try:
        yield touched
    finally:
        _watched = None
        for event, arguments in seen:
            path = arguments[0] if arguments else None
            if isinstance(path, bytes):
                path = os.fsdecode(path)
            if isinstance(path, (str, os.PathLike)) and os.path.abspath(os.fspath(path)).startswith(root):
                touched.append((event, os.path.relpath(os.fspath(path), root)))


def tree(seed: int = 0):
    """A simulation-shaped state: nested arrays of several dtypes and a typed key."""
    values = np.random.default_rng(seed)
    return {"state": {"v": jnp.asarray(values.normal(size=(4, 6)), jnp.float32),
                      "count": jnp.asarray(values.integers(0, 9, size=(6,)), jnp.int32),
                      "spikes": (jnp.asarray(values.random((6,)) > .5), jnp.zeros((2,), jnp.bfloat16))},
            "key": jax.random.key(seed)}


def template(of, sharding=None):
    """`of`'s shapes and dtypes, placed on `sharding` or on the first device."""
    where = sharding or jax.sharding.SingleDeviceSharding(jax.devices()[0])
    return jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=where), of)


def same(left, right):
    left, right = (jax.tree.map(lambda leaf: np.asarray(jax.random.key_data(leaf) if jnp.issubdtype(
        leaf.dtype, jax.dtypes.prng_key) else leaf), side) for side in (left, right))
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for one, other in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        assert one.dtype == other.dtype
        np.testing.assert_array_equal(one, other)


@jax.jit
def advance(state):
    key, noise = jax.random.split(state["key"])
    inner = state["state"]
    v = .9 * inner["v"] + jax.random.normal(noise, inner["v"].shape)
    spikes = v[-1] > 0
    return {"state": {"v": v, "count": inner["count"] + spikes, "spikes": (spikes, inner["spikes"][1] + 1)},
            "key": key}


def test_a_tree_restores_bit_for_bit_with_its_control(tmp_path):
    checkpoints = Checkpoints(str(tmp_path))
    written = tree()
    checkpoints.save(3, written, control={"dt": .1, "trials": None})
    checkpoints.wait()
    reread = Checkpoints(str(tmp_path))
    assert reread.latest == 3
    assert reread.control(3) == {"dt": .1, "trials": None}
    assert [checkpoint.kind for checkpoint in reread.kept()] == ["tree"]
    restored = reread.restore(template(written))[0]
    same(restored, written)
    assert isinstance(restored["state"]["spikes"], tuple)


def test_a_tree_reads_as_host_arrays_without_a_template(tmp_path):
    checkpoints = Checkpoints(str(tmp_path))
    written = {"v": jnp.arange(6.).reshape(2, 3), "n": jnp.int32(4)}
    checkpoints.save(1, written)
    checkpoints.wait()
    restored, _ = checkpoints.restore()
    assert isinstance(restored["v"], np.ndarray)
    same(restored, written)


def test_an_interrupted_run_resumes_to_the_uninterrupted_state(tmp_path):
    whole = tree()
    for _ in range(10):
        whole = advance(whole)

    checkpoints = Checkpoints(str(tmp_path))
    state = tree()
    for step in range(1, 7):
        state = advance(state)
        if step % 3 == 0:
            checkpoints.save(step, state)
    state = advance(state)  # the run is interrupted after this unsaved step
    checkpoints.wait()

    resumed = Checkpoints(str(tmp_path))
    start = resumed.latest
    assert start == 6
    state, _ = resumed.restore(template(tree()))
    for _ in range(start, 10):
        state = advance(state)
    same(state, whole)


def test_keep_retains_latest_periodic_and_best_tree_steps(tmp_path):
    checkpoints = Checkpoints(str(tmp_path), keep=Keep(latest=2, every=4))
    for step in range(1, 10):
        loss = abs(step - 3) + 1.
        checkpoints.save(step, {"x": jnp.full((3,), step)}, metrics={"loss": loss},
                         ranking=Ranking("loss", loss))
        checkpoints.wait()
    assert [checkpoint.step for checkpoint in checkpoints.kept()] == [3, 4, 8, 9]
    assert checkpoints.best == 3
    np.testing.assert_array_equal(checkpoints.restore(step="best")[0]["x"], np.full((3,), 3))


@pytest.mark.mesh
def test_a_sharded_tree_restores_onto_another_mesh(tmp_path):
    devices = np.asarray(jax.devices()[:8])
    written_on = jax.make_mesh((8,), ("x",), devices=devices.tolist())
    sharded = NamedSharding(written_on, P(None, "x"))
    written = {"v": jax.device_put(jnp.arange(32., dtype=jnp.float32).reshape(4, 8), sharded),
               "w": jax.device_put(jnp.arange(32, dtype=jnp.int32).reshape(4, 8), sharded)}
    checkpoints = Checkpoints(str(tmp_path))
    checkpoints.save(2, written)
    checkpoints.wait()

    read_on = jax.make_mesh((4, 2), ("a", "b"), devices=devices[::-1].tolist())
    target = NamedSharding(read_on, P("a", "b"))
    restored, _ = Checkpoints(str(tmp_path)).restore(template(written, target))
    for name in written:
        assert restored[name].sharding == target
        np.testing.assert_array_equal(np.asarray(restored[name]), np.asarray(written[name]))


@pytest.mark.parametrize("dtype", [np.bool_, np.uint8, jnp.bfloat16, np.int32, np.float64])
def test_a_leaf_marks_the_same_from_the_host_as_on_the_devices(dtype):
    """A frozen digest marks a host leaf on the host and a device leaf on the devices: the two give
    one leaf's stored bytes the same marks, at any chunk the host reads them in, where JAX holds the
    leaf as stored (a float64 leaf with x64 on)."""
    from dew.checkpoints import _host_marks, _marks

    values = np.abs(np.random.default_rng(0).normal(size=(3, 5, 7))) * 40
    leaf = (values > 30) if dtype is np.bool_ else values.astype(dtype)
    with jax.enable_x64(dtype is np.float64):
        device = np.asarray(jax.device_get(_marks([jnp.asarray(leaf)])))[0]
    for chunk in (1 << 24, 16, 7):
        np.testing.assert_array_equal(_host_marks(leaf, chunk), device)


def test_a_reader_beside_the_run_lists_each_step_as_it_commits(tmp_path):
    """A process that reads a run another saves into, as an evaluation worker does, lists the
    committed steps at each call: a step saved after the reader first looked is listed, a save in
    flight (Orbax's temporary directory) is not, and listing writes nothing to the store."""
    writer = Checkpoints(str(tmp_path))
    writer.save(1, {"x": jnp.full((3,), 1)})
    writer.wait()
    reader = Checkpoints(str(tmp_path))
    assert reader.committed() == [1]
    writer.save(2, {"x": jnp.full((3,), 2)})
    writer.wait()
    (tmp_path / "3.orbax-checkpoint-tmp-1730000000").mkdir()
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert reader.committed() == [1, 2]
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before
    np.testing.assert_array_equal(reader.restore(step=2)[0]["x"], np.full((3,), 2))


def test_a_listing_after_a_restore_and_close_opens_nothing(tmp_path):
    """A reader that restored a step and closed lists the committed steps without reopening the store:
    observed at the filesystem's boundary (`store_touches`), the listing lists directories and opens no
    file of the store, and writes, moves or removes nothing there, as the restore before it did read
    the step's files."""
    writer = Checkpoints(str(tmp_path))
    writer.save(1, {"x": jnp.full((3,), 1)})
    writer.wait()
    reader = Checkpoints(str(tmp_path))
    with store_touches(tmp_path) as restoring:
        reader.restore(step=1)
    assert any(event == "open" for event, _ in restoring)
    reader.close()
    with store_touches(tmp_path) as listing:
        assert reader.committed() == [1]
    assert listing and {event for event, _ in listing} <= {"os.listdir", "os.scandir"}, listing
