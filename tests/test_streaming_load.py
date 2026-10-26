"""A load onto a mesh builds each leaf from the mapped checkpoint, one device shard at a time.

`Pretrained.load(..., mesh=...)` translates the checkpoint into `SourceLeaf`
recipes and lands each straight on its sharding through
`jax.make_array_from_callback`. What a consumer can observe is that the
placed values are the host load's values under the layout's sharding, and
that no read the placement makes spans more than one device's shard: a
read of a whole leaf is exactly the gather into one host tree this path
exists to avoid.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.interop.streaming import SourceLeaf


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_a_transposed_read_across_many_tiles_is_the_strided_copy_bit_for_bit(dtype):
    """A transposed leaf is read and cast in its stored layout, then swapped
    in 64x64 tiles; on shapes that leave ragged tiles, at blocks, strides and
    reversals, alone and stacked, that is numpy's own strided copy, bit for
    bit and C-ordered."""
    import ml_dtypes

    storage = np.dtype(ml_dtypes.bfloat16) if dtype == "bfloat16" else np.dtype(dtype)
    rng = np.random.default_rng(3)
    member = rng.standard_normal((200, 131)).astype(np.float32)
    leaf = SourceLeaf((member,), storage, transposed=True)
    whole = member.T.astype(storage)
    for index in [None, (slice(5, 190), slice(3, 130)), (slice(None, None, 3), slice(100, 0, -7)),
                  (slice(64, 129), slice(None)), (slice(130, 131), slice(0, 1))]:
        read = leaf.read(index)
        expected = np.ascontiguousarray(whole if index is None else whole[index])
        assert read.dtype == storage and read.flags.c_contiguous
        assert read.shape == expected.shape and read.tobytes() == expected.tobytes(), index
    members = tuple(rng.standard_normal((70, 90)).astype(np.float32) for _ in range(3))
    stacked = SourceLeaf.stack([SourceLeaf((one,), storage, transposed=True) for one in members], "experts")
    every = np.stack([one.T for one in members]).astype(storage)
    for index in [None, (slice(1, 3), slice(10, 80), slice(None, None, -1))]:
        expected = np.ascontiguousarray(every if index is None else every[index])
        assert stacked.read(index).tobytes() == expected.tobytes(), index

    small_members = tuple(rng.standard_normal((6, 4)).astype(np.float32) for _ in range(3))
    small = SourceLeaf.stack(
        [SourceLeaf((one,), storage, transposed=True) for one in small_members], "experts")
    whole_small = np.stack([one.T for one in small_members]).astype(storage)
    np.testing.assert_array_equal(small.read(), whole_small)
    for index in [(slice(1, 3), slice(None), slice(2, 4)), (slice(0, 1), slice(1, 3), slice(None)),
                  (slice(None), slice(3, 4), slice(0, 6))]:
        np.testing.assert_array_equal(small.read(index), whole_small[index])

    for transposed, shape in ((False, (10, 20, 131)), (True, (131, 10, 20))):
        headed = SourceLeaf((member,), storage, transposed=transposed).reshape(shape)
        member[3, 5] += 7
        native = member.T if transposed else member
        expected = native.reshape(shape).astype(storage)
        index = (slice(1, 9), slice(None, None, 2), slice(1, 19, 3))
        np.testing.assert_array_equal(headed.read(index), expected[index])
        np.testing.assert_array_equal(headed.read(), expected)

    headed = small.reshape((3, 4, 2, 3))
    small_members[1][2, 1] += 4
    expected = np.stack([one.T.reshape(4, 2, 3) for one in small_members]).astype(storage)
    index = (slice(1, 3), slice(None), slice(0, 1), slice(None))
    np.testing.assert_array_equal(headed.read(index), expected[index])
    with pytest.raises(ValueError):
        small.reshape((6, 4, 3))


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_an_eager_translation_swaps_kernels_in_tiles_as_the_strided_copy_does(dtype):
    """`translate_parameters` casts each kernel in its stored layout and
    swaps it in the same 64x64 tiles as a transposed read; on ragged shapes
    that is numpy's own strided copy, bit for bit. A 4-D convolution kernel,
    whose layout is no swap of the last two axes, keeps numpy's copy."""
    import ml_dtypes

    from dew.interop.weights import translate_parameters

    storage = np.dtype(ml_dtypes.bfloat16) if dtype == "bfloat16" else np.dtype(dtype)
    rng = np.random.default_rng(5)
    tensors = {"dense": rng.standard_normal((200, 131)).astype(np.float32),
               "conv": rng.standard_normal((8, 3, 3, 5)).astype(np.float32)}
    params = translate_parameters(tensors, lambda name: ("params", name, "kernel"), dtype)
    for name, order in (("dense", (1, 0)), ("conv", (2, 3, 1, 0))):
        leaf = params["params"][name]["kernel"]
        expected = np.ascontiguousarray(tensors[name].astype(storage).transpose(order))
        assert leaf.dtype == storage and leaf.flags.c_contiguous
        assert leaf.shape == expected.shape and leaf.tobytes() == expected.tobytes(), name


@pytest.mark.parametrize("copy", ["equal", "rounds alike", "differs"])
def test_an_eager_and_a_streamed_binding_bind_aliases_alike(copy):
    """Two source names that fill one leaf (a tied embedding stored twice) bind alike eagerly
    (`translate_parameters`) and through the streamed binding (`record_layouts`): equal copies
    bind once, each name keeping its layout, and copies that differ are refused before the
    storage cast, even where they round to the same bf16."""
    import ml_dtypes

    from dew.interop.weights import placed, record_layouts

    rng = np.random.default_rng(3)
    # bf16 values, which a relative change of 2^-12, under half their spacing, leaves where they are.
    first = rng.standard_normal((16, 8)).astype(ml_dtypes.bfloat16).astype(np.float32)
    second = {"equal": first, "rounds alike": first * np.float32(1 + 2 ** -12), "differs": first + 1}[copy]
    assert copy != "rounds alike" or np.array_equal(first.astype(ml_dtypes.bfloat16),
                                                    second.astype(ml_dtypes.bfloat16))
    tensors = {"embed": first, "lm_head": second, "norm": rng.standard_normal(8).astype(np.float32)}

    def path_of(name):
        return ("params", "norm", "scale") if name == "norm" else ("params", "embed", "embedding")

    bindings = (lambda: placed(tensors, path_of, "bfloat16"),
                lambda: record_layouts("", tensors, path_of, (), param_dtype="bfloat16"))
    if copy != "equal":
        for bind in bindings:
            with pytest.raises(ValueError):
                bind()
        return
    (eager, eager_layouts), (streamed, streamed_layouts) = (bind() for bind in bindings)
    assert eager_layouts == streamed_layouts
    assert sorted(layout.name for layout in eager_layouts) == ["embed", "lm_head", "norm"]
    for ours, theirs in zip(jax.tree.leaves(eager), jax.tree.leaves(streamed), strict=True):
        assert ours.dtype == theirs.dtype and ours.tobytes() == theirs.tobytes()


@pytest.mark.mesh
def test_a_mesh_resident_leaf_can_change_its_sharding_before_host_transport():
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    from dew.training.host import place_leaf

    mesh = Mesh(np.asarray(jax.devices()[:2]), ("fsdp",))
    values = np.arange(8 * 12, dtype=np.float32).reshape(8, 12)
    current = place_leaf(values, NamedSharding(mesh, PartitionSpec("fsdp")))
    for spec in (PartitionSpec(), PartitionSpec(None, "fsdp"), PartitionSpec("fsdp")):
        target = NamedSharding(mesh, spec)
        current = place_leaf(current, target)
        assert current.sharding == target
        np.testing.assert_array_equal(current, values)
    host = NamedSharding(mesh, PartitionSpec("fsdp"), memory_kind="pinned_host")
    held = place_leaf(current, host)
    assert held.sharding == host
    np.testing.assert_array_equal(held, values)
    returned = place_leaf(held, current.sharding)
    np.testing.assert_array_equal(returned, values)


@pytest.mark.mesh
def test_a_host_source_is_placed_holding_one_device_shard_at_a_time():
    """Each shard a `HostSource` reads is on its device before the next is
    read, so the host never holds two of a leaf's shards; a placement that
    read every shard first would hold all eight. On CPU a placed shard may
    keep the read buffer as its device storage, so a buffer that is some
    live device array's storage counts as placed, not held."""
    import gc
    import weakref

    from jax.sharding import NamedSharding, PartitionSpec

    from dew.training.host import place_leaf

    class Recorded:
        def __init__(self, value: np.ndarray):
            self.value, self.shape, self.ndim = value, value.shape, value.ndim
            self.reads: list[tuple[weakref.ref, int]] = []
            self.most = 0

        def read(self, index):
            gc.collect()
            # A donated shard holds nothing (see the NCCL push test in test_rollout_servers.py).
            placed = {shard.data.unsafe_buffer_pointer() for array in jax.live_arrays()
                      for shard in array.addressable_shards if not shard.data.is_deleted()}
            held = [ref for ref, address in self.reads if ref() is not None and address not in placed]
            self.most = max(self.most, len(held) + 1)
            part = np.array(self.value[index])
            self.reads.append((weakref.ref(part), part.ctypes.data))
            return part

        def release(self):
            pass

    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("fsdp",))
    source = Recorded(np.arange(jax.device_count() * 256, dtype=np.float32).reshape(jax.device_count(), 256))
    placed = place_leaf(source, NamedSharding(mesh, PartitionSpec("fsdp")))
    np.testing.assert_array_equal(np.asarray(placed), source.value)
    assert source.most == 1


class Whole:
    """A host source read whole, as a bank's row is, recording whether it was released."""

    def __init__(self, value: np.ndarray, *, fails: bool = False):
        self.value, self.fails, self.released = value, fails, False

    @property
    def shape(self):
        return self.value.shape

    @property
    def ndim(self):
        return self.value.ndim

    def read(self, index):
        if self.fails:
            raise OSError("the read failed")
        return self.value if index is None else self.value[index]

    def release(self):
        self.released = True


def test_a_format_placement_lands_a_source_whole_and_releases_it_whether_or_not_it_lands():
    """`place_leaf` puts a host source on a physical layout (`Format`, a bank's target) whole,
    in that layout, and releases the source; a placement that fails releases it too."""
    from jax.experimental.layout import Format, Layout as DeviceLayout
    from jax.sharding import SingleDeviceSharding

    from dew.training.host import place_leaf

    target = Format(DeviceLayout(major_to_minor=(0, 1)), SingleDeviceSharding(jax.devices()[0]))
    source = Whole(np.arange(12, dtype=np.float32).reshape(3, 4))
    placed = place_leaf(source, target)
    np.testing.assert_array_equal(np.asarray(placed), source.value)
    assert placed.format.layout.major_to_minor == (0, 1) and source.released
    failing = Whole(source.value, fails=True)
    with pytest.raises(OSError):
        place_leaf(failing, target)
    assert failing.released


def test_a_loaded_source_is_ready_to_run_on_the_default_device():
    """A source loaded without a mesh or layout holds its variables on JAX's default device,
    uncommitted, as a `jnp` array is held, so a jitted call reads them where they are rather than
    copying the whole tree from the host at each call, and a call or placement over every device
    still takes them; `on_host=True` keeps host arrays, which a mesh or layout
    beside it contradicts. A load returned host arrays, and a 4.57B model's step moved 9.1 GB
    each call."""
    from pathlib import Path

    from dew.interop import Pretrained

    fixture = Path(__file__).resolve().parent / "fixtures" / "hf" / "llama-tiny"
    loaded = Pretrained.load(fixture, dtype="float32", attention_impl="reference")
    leaves = jax.tree.leaves(loaded.variables)
    # Where a jnp array goes: the default device, uncommitted.
    native = jnp.zeros(())
    assert leaves and all(isinstance(leaf, jax.Array) and leaf.devices() == native.devices()
                          and leaf.committed == native.committed for leaf in leaves)
    # Uncommitted, as a jnp array is: a call placed on every device takes it where it is.
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("data",))
    spread = jax.jit(lambda leaf: leaf * 2, out_shardings=jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()))(leaves[0])
    np.testing.assert_array_equal(np.asarray(spread), 2 * np.asarray(leaves[0]))
    host = Pretrained.load(fixture, dtype="float32", attention_impl="reference", on_host=True)
    assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(host.variables))
    for placed, held in zip(leaves, jax.tree.leaves(host.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(placed), held)
    with pytest.raises(ValueError):
        Pretrained.load(fixture, on_host=True, mesh=jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",)))


@pytest.mark.parametrize("where", ["device", "host"])
def test_a_source_trained_from_keeps_its_arrays_after_the_step_donates_the_state(where):
    """A step donates the state it is given, and a run's state holds buffers of its own: a source
    loaded on the device and trained from, its state resident or held on the host, has every array
    readable and unchanged after the step, placed as `fit` places it or built eagerly
    (tests/test_flux_source.py holds a pipeline's objective, whose eager state was its source's
    own arrays, to the same)."""
    from pathlib import Path

    import jax.numpy as jnp
    import optax

    from dew.interop import Pretrained
    from dew.objectives.lm import LMObjective
    from dew.training import Layout, Trainer

    fixture = Path(__file__).resolve().parent / "fixtures" / "hf" / "llama-tiny"
    loaded = Pretrained.load(fixture, dtype="float32", attention_impl="reference")
    before = jax.device_get(loaded.variables)
    # The default layout replicates a small source's leaves, so its placed state can share their buffers.
    layout = Layout(host=("variables",)) if where == "host" else Layout()
    trainer = Trainer(LMObjective(loaded, 8, ema_decay=None), optax.sgd(0.1), key=jax.random.key(5),
                      layout=layout)
    batch = {"text": jnp.tile(jnp.arange(1, 10, dtype=jnp.int32)[None], (jax.device_count(), 1))}
    states = [trainer.place()[0]] + ([trainer.initial_state()] if where == "device" else [])
    for state in states:
        stepped, *_ = trainer.compile(state, batch)(state, batch)
        jax.block_until_ready(stepped)
        for leaf, held in zip(jax.tree.leaves(loaded.variables), jax.tree.leaves(before), strict=True):
            np.testing.assert_array_equal(np.asarray(leaf), held)


def test_a_host_owned_state_takes_its_held_tree_over_leaf_by_leaf():
    """A state held on the host is placed from the objective's own arrays one leaf at a time, each
    handed over as it lands (`host.stream`): its step is not donated, so the state takes no copy of
    the held tree first, and the objective's held tree then holds the state's placed arrays, one copy
    between the two."""
    from pathlib import Path

    import optax

    from dew.interop import Pretrained
    from dew.objectives.lm import LMObjective
    from dew.training import Layout, Trainer

    fixture = Path(__file__).resolve().parent / "fixtures" / "hf" / "llama-tiny"
    objective = LMObjective(Pretrained.load(fixture, dtype="float32", attention_impl="reference"), 8,
                            ema_decay=None)
    trainer = Trainer(objective, optax.sgd(0.1), key=jax.random.key(5), layout=Layout(host=("variables",)))
    state, _, _ = trainer.place()
    placed = {id(leaf) for leaf in jax.tree.leaves(state.variables)}
    held = jax.tree.leaves(objective.held_variables())
    assert held and all(id(leaf) in placed for leaf in held)
