#!/usr/bin/env python3
"""Draw K residual orders for a decoder fixture and write its weights under each.

    PYTHONPATH=src python tools/rounding_orders.py <fixture dir> <out dir> <K> <seed> [group]

tests/reference_error.py's K-order rule measures Dew and its reference over
the same K rounding draws, each the fixture's model with its residual stream's
units reordered (tests/residual_orders.py). This writes `<out>/orders.npy`
and, for each order k, `<out>/<k>/` holding the fixture's files with its
weights saved in that order through `Pretrained.save`, the source format a
reference tool loads. Order 0 is the identity, and its weights are checked
to load as the fixture's own, bit for bit. `group` keeps quantization groups along
the residual whole (tests/residual_orders.py).
"""

import shutil
import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from residual_orders import orders, permuted, residual_width

from dew.interop import Pretrained


def main(fixture: Path, out: Path, count: int, seed: int, group: int) -> None:
    loaded = Pretrained.load(fixture, dtype="float32", attention_impl="reference")
    drawn = orders(residual_width(loaded.variables), count, seed, group)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "orders.npy", drawn)
    for k, order in enumerate(drawn):
        loaded.save(out / str(k), variables=permuted(loaded.variables, order))
        for name in ("reference.npz", "numerics.npz", "generation_config.json"):
            if (fixture / name).exists():
                shutil.copy(fixture / name, out / str(k) / name)
    # A quantized format may pack an equal value in other bits (FP4's two
    # zeros), so order 0 is held to the fixture as loaded, value for value.
    again = Pretrained.load(out / "0", dtype="float32", attention_impl="reference")
    same = jax.tree.map(lambda a, b: bool(np.array_equal(a, b)), again.variables, loaded.variables)
    assert all(jax.tree.leaves(same)), "order 0 does not load as the fixture"
    print("wrote", count, "orders to", out)


if __name__ == "__main__":
    if len(sys.argv) not in (5, 6):
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
         int(sys.argv[5]) if len(sys.argv) == 6 else 1)
