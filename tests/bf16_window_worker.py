"""The worker of the pool that packs a kernel no process holds whole as its lossless bfloat16 window
(`dew.quantization.bf16_windows`, row 13702): each process a rank holding its columns.

dew is imported inside the mode, as `multiprocess_worker`'s are, so the pool starts before JAX does.
"""

from __future__ import annotations

import numpy as np
from process_support import serve_mode, worker_parser


def width_split(args) -> dict:
    """A `[256, 256]` bfloat16 kernel split by its columns across every device of the pool's processes:
    its window encoded where each column lives, each process's planes the window of its own columns, and
    decoded on each process its own columns' bits again."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec

    from dew import quantization
    from dew.nn.kernels.formats import Bf16Window

    rng = np.random.default_rng(13702)
    kernel = (rng.normal(size=(256, 256)) * 0.02).astype(jnp.bfloat16)
    # Its depth named by an Explicit axis of one device, which leaves each column whole.
    mesh = Mesh(np.asarray(jax.devices()).reshape(1, -1), ("depths", "columns"),
                axis_types=(AxisType.Explicit, AxisType.Explicit))
    spec = PartitionSpec("depths", "columns")
    placed = jax.make_array_from_callback(kernel.shape, NamedSharding(mesh, spec),
                                          lambda index: kernel[index])
    assert not placed.is_fully_addressable, "the pool's processes each hold their own columns"
    window = quantization.bf16_windows({"params": {"kernel": placed}}, sites=[""])[quantization.QUANTIZATION][
        "kernel"]
    # Each device's plane shard the window of the columns the kernel placed on that device (its own map of
    # devices to columns, not the planes'), and the window decoded there those columns' bits again.
    columns = {shard.device: shard.index[1] for shard in placed.addressable_shards}
    planes = [bool(np.array_equal(np.asarray(shard.data),
                                  Bf16Window.of(kernel[:, columns[shard.device]]).words))
              for shard in window.words.addressable_shards]
    decoded = [bool(np.array_equal(np.asarray(shard.data).view(np.uint16),
                                   kernel[:, columns[shard.device]].view(np.uint16)))
               for shard in jax.jit(lambda held: held.decoded())(window).addressable_shards]
    return {"planes": planes, "decoded": decoded}


def depth_split(args) -> dict:
    """A `[256, 256]` bfloat16 kernel split by its depth across the pool's processes, two whole 128-deep
    groups apart: its window refused by its site's name, which no device holding a part of a column's
    groups decodes where it lives."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    from dew import quantization

    kernel = (np.random.default_rng(13702).normal(size=(256, 256)) * 0.02).astype(jnp.bfloat16)
    mesh = Mesh(np.asarray(jax.devices()), ("depths",))
    placed = jax.make_array_from_callback(kernel.shape, NamedSharding(mesh, PartitionSpec("depths", None)),
                                          lambda index: kernel[index])
    try:
        quantization.bf16_windows({"params": {"kernel": placed}}, sites=[""])
    except ValueError as error:
        return {"refused": type(error).__name__}
    return {"refused": None}


def one_rank(args) -> dict:
    """A `[256, 256]` bfloat16 kernel placed on process 0's devices alone, its columns split across them:
    no process waits on another's agreement; process 0, holding it whole, windows it whole, and process 1
    holds none of its planes."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    from dew import quantization
    from dew.nn.kernels.formats import Bf16Window

    kernel = (np.random.default_rng(13702).normal(size=(256, 256)) * 0.02).astype(jnp.bfloat16)
    owned = [device for device in jax.devices() if device.process_index == 0]
    mesh = Mesh(np.asarray(owned), ("columns",))
    # Its dtype stated: process 1 holds no shard to read it from.
    placed = jax.make_array_from_callback(kernel.shape, NamedSharding(mesh, PartitionSpec(None, "columns")),
                                          lambda index: kernel[index], dtype=kernel.dtype)
    window = quantization.bf16_windows({"params": {"kernel": placed}}, sites=[""])[quantization.QUANTIZATION][
        "kernel"]
    if args.process_id == 0:
        return {"planes": [bool(np.array_equal(np.asarray(window.words), Bf16Window.of(kernel).words))],
                "shape": list(window.shape)}
    return {"planes": [shard.data.size for shard in window.words.addressable_shards],
            "shape": list(window.shape)}


def own_device(args) -> dict:
    """Each process's own `[256, 256]` bfloat16 kernel on its default device, an ordinary JAX array of
    none of the pool's meshes: each windowed whole where it is, decoded its bits again."""
    import jax.numpy as jnp

    from dew import quantization

    kernel = (np.random.default_rng([13702, args.process_id]).normal(size=(256, 256)) * 0.02).astype(
        jnp.bfloat16)
    window = quantization.bf16_windows({"params": {"kernel": jnp.asarray(kernel)}}, sites=[""])[
        quantization.QUANTIZATION]["kernel"]
    return {"decoded": bool(np.array_equal(np.asarray(window.decoded()).view(np.uint16),
                                           kernel.view(np.uint16)))}


def own_mesh(args) -> dict:
    """Process 0 alone windows a `[256, 256]` kernel on a mesh of its own devices, its peer windowing none:
    encoded where it lives without waiting on a peer that never meets it, decoded its bits again."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    from dew import quantization

    if args.process_id != 0:
        return {"decoded": None}
    kernel = (np.random.default_rng(13702).normal(size=(256, 256)) * 0.02).astype(jnp.bfloat16)
    mesh = Mesh(np.asarray(jax.local_devices()), ("columns",))
    placed = jax.device_put(kernel, NamedSharding(mesh, PartitionSpec(None, "columns")))
    window = quantization.bf16_windows({"params": {"kernel": placed}}, sites=[""])[quantization.QUANTIZATION][
        "kernel"]
    decoded = np.asarray(window.decoded()).view(np.uint16)
    return {"decoded": bool(np.array_equal(decoded, kernel.view(np.uint16)))}


MODES = {"width_split": width_split, "depth_split": depth_split, "one_rank": one_rank,
         "own_device": own_device, "own_mesh": own_mesh}


def main(argv=None) -> None:
    parser = worker_parser(MODES, __doc__)
    serve_mode(MODES, parser.parse_args(argv))


if __name__ == "__main__":
    main()
