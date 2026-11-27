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
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    from dew import quantization
    from dew.nn.kernels.formats import Bf16Window

    rng = np.random.default_rng(13702)
    kernel = (rng.normal(size=(256, 256)) * 0.02).astype(jnp.bfloat16)
    # Its depth named by an axis of one device, which leaves each column whole.
    mesh = Mesh(np.asarray(jax.devices()).reshape(1, -1), ("depths", "columns"))
    spec = PartitionSpec("depths", "columns")
    placed = jax.make_array_from_callback(kernel.shape, NamedSharding(mesh, spec),
                                          lambda index: kernel[index])
    assert not placed.is_fully_addressable, "the pool's processes each hold their own columns"
    window = quantization.bf16_windows({"params": {"kernel": placed}}, sites=[""])[quantization.QUANTIZATION][
        "kernel"]
    # Each plane shard the window of its own columns, and the window decoded each column's bits again.
    planes = [bool(np.array_equal(np.asarray(shard.data),
                                  Bf16Window.of(kernel[:, shard.index[0]]).words[:, shard.index[1]]))
              for shard in window.words.addressable_shards]
    decoded = [bool(np.array_equal(np.asarray(shard.data).view(np.uint16),
                                   kernel[shard.index].view(np.uint16)))
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


MODES = {"width_split": width_split, "depth_split": depth_split}


def main(argv=None) -> None:
    parser = worker_parser(MODES, __doc__)
    serve_mode(MODES, parser.parse_args(argv))


if __name__ == "__main__":
    main()
