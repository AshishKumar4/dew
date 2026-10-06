"""A mapping saved in place of a train state checkpoints, resumes and is retained as a train state is."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from dew.checkpoints import Checkpoints, Keep, Ranking
from dew.training.state import TrainState


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


def test_a_template_of_another_shape_is_refused(tmp_path):
    checkpoints = Checkpoints(str(tmp_path))
    checkpoints.save(1, {"v": jnp.zeros((4,))})
    checkpoints.wait()
    with pytest.raises(ValueError, match="holds a mapping that does not fit"):
        checkpoints.restore(template({"v": jnp.zeros((5,))}))


def test_a_mapping_is_not_resumed_as_a_train_state(tmp_path):
    """A step that holds a mapping refuses a train-state template."""
    checkpoints = Checkpoints(str(tmp_path))
    checkpoints.save(1, {"w": jnp.ones(2)})
    checkpoints.wait()
    zero = jnp.zeros((), jnp.int32)
    state = TrainState(step=jnp.int32(1), microstep=zero, updates=zero,
                       variables={"params": {"w": jnp.ones(2)}}, opt_state=(), ema=None,
                       key=jax.random.key(0), scale=None, window_size=jnp.ones((), jnp.int32))
    with pytest.raises(ValueError, match="holds a mapping saved in place of a train state"):
        checkpoints.restore(state)


def test_a_mapping_takes_no_data_position(tmp_path):
    """What only a train state's save records is refused with a mapping."""
    with pytest.raises(ValueError, match="it takes no data position"):
        Checkpoints(str(tmp_path)).save(1, {"w": jnp.ones(2)}, b"{}")


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
