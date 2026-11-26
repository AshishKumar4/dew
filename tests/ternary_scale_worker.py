"""The workers of a pool whose packed ternary matrix's scale lies across, or on one of, its two processes
(tests/test_multiprocess.py, row 13340).

dew is imported inside the mode, as `multiprocess_worker`'s are, so the pool starts before JAX does.
"""

from __future__ import annotations

import numpy as np
from process_support import serve_mode, worker_parser


def ternary_scale(args) -> dict:
    """A packed ternary matrix whose per-column scale spans both processes (each its half of the columns)
    and holds a subnormal on process 1's half alone: each rank's `checked` outcome (the error's type), and
    every process then passes one more agreement, which a rank left refusing alone would block (row 13340)."""
    del args
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    from dew import _coordination
    from dew.checkpoints import _packed_checked
    from dew.nn.kernels.formats import TernaryWeights

    width = jax.device_count()  # a column a device, the last process's devices last
    weights = np.zeros((256, width), np.float32)
    weights[0] = 1.0
    packed = TernaryWeights.of(weights, scale=np.ones(width, np.float32))
    mesh = Mesh(np.asarray(jax.devices()), ("columns",))
    host = np.ones(width, np.float32)
    host[-1] = 2.0 ** -130
    scale = jax.make_array_from_callback((width,), NamedSharding(mesh, PartitionSpec("columns")),
                                         lambda index: host[index])
    def outcome(held) -> str:
        try:
            TernaryWeights(packed.words, held, packed.form).checked()
        except (ValueError, _coordination.PeerFailure) as error:
            return type(error).__name__
        return "admitted"

    spanning = outcome(scale)
    # A normal scale on process 0's devices alone: its owner scans it, the other holds none, neither waits.
    first = [device for device in jax.devices() if device.process_index == 0]
    alone_on = NamedSharding(Mesh(np.asarray(first), ("columns",)), PartitionSpec())
    def on_first(values: np.ndarray) -> jax.Array:
        return jax.make_array_from_callback((width,), alone_on, lambda index: values[index], dtype=np.float32)

    alone = outcome(on_first(np.ones(width, np.float32)))
    # A subnormal on process 0's devices alone, at a pool-wide restore's check: refused on both ranks.
    try:
        # The bad one-owner leaf first, a valid pool-wide one after: one agreement for both.
        _packed_checked({"a": TernaryWeights(packed.words, on_first(host), packed.form),
                         "b": TernaryWeights(packed.words, jax.make_array_from_callback(
                             (width,), NamedSharding(mesh, PartitionSpec("columns")),
                             lambda index: np.ones(width, np.float32)[index]), packed.form)})
        restored = "admitted"
    except (ValueError, _coordination.PeerFailure) as error:
        restored = type(error).__name__
    after = _coordination.agreed("after the scale checks", lambda: True)
    return {"outcome": spanning, "alone": alone, "restored": restored, "after": after}


MODES = {"ternary_scale": ternary_scale}


def main(argv=None) -> None:
    serve_mode(MODES, worker_parser(MODES, __doc__).parse_args(argv))


if __name__ == "__main__":
    main()
