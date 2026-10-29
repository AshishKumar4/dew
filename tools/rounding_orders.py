#!/usr/bin/env python3
"""Draw K residual orders for a decoder fixture and write its weights under each.

    PYTHONPATH=src python tools/rounding_orders.py <fixture dir> <out dir> <K> <seed> [group]
    PYTHONPATH=src python tools/rounding_orders.py symmetric <fixture dir> <out dir>

tests/reference_error.py's K-order rule measures Dew and its reference over
the same K rounding draws, each the fixture's model with its residual stream's
units reordered (tests/residual_orders.py). This writes `<out>/orders.npy`
and, for each order k, `<out>/<k>/` holding the fixture's files with its
weights saved in that order through `Pretrained.save`, the source format a
reference tool loads. Order 0 is the identity, and its weights are checked
to load as the fixture's own, bit for bit. `group` keeps quantization groups along
the residual whole (tests/residual_orders.py).
"""

import json
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
        # Not numerics.npz: each order's float64 truth is its own, which its reference's recorder writes
        # there (`symmetric`).
        for name in ("reference.npz", "generation_config.json"):
            if (fixture / name).exists():
                shutil.copy(fixture / name, out / str(k) / name)
    # A quantized format may pack an equal value in other bits (FP4's two
    # zeros), so order 0 is held to the fixture as loaded, value for value.
    again = Pretrained.load(out / "0", dtype="float32", attention_impl="reference")
    same = jax.tree.map(lambda a, b: bool(np.array_equal(a, b)), again.variables, loaded.variables)
    assert all(jax.tree.leaves(same)), "order 0 does not load as the fixture"
    print("wrote", count, "orders to", out)


def symmetric(fixture: Path, drawn: Path) -> None:
    """Hold the float64 truth of the second and the last residual order drawn under `drawn` (each order's
    numerics.npz, which its reference's recorder writes there: `kimi_k3_reference.py truth DIR`) to the
    fixture's own, within float64 rounding over the computation's longest chain of reductions (each layer's
    hidden and token sums, then the head's): the orders are a symmetry of the reference's computation, not
    only of Dew's."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from reference_error import assert_computes_the_oracle

    text = json.loads((fixture / "config.json").read_text())
    text = text.get("text_config", text)
    with np.load(fixture / "numerics.npz") as exact, np.load(fixture / "reference.npz") as stored:
        valid = stored["attention_mask"].astype(bool)
        roundings = text["num_hidden_layers"] * (text["hidden_size"] + valid.shape[1]) + text["hidden_size"]
        count = len(np.load(drawn / "orders.npy"))
        for k in (1, count - 1):
            with np.load(drawn / str(k) / "numerics.npz") as arrays:
                for key in ("logits_f64", "updated_logits_f64"):
                    assert_computes_the_oracle(arrays[key][valid], exact[key][valid],
                                               f"{fixture.name} order {k} {key}", roundings=roundings)
    print(fixture.name, "orders 1 and", count - 1, "hold the float64 truth")


if __name__ == "__main__":
    if sys.argv[1:2] == ["symmetric"] and len(sys.argv) == 4:
        symmetric(Path(sys.argv[2]), Path(sys.argv[3]))
        raise SystemExit(0)
    if len(sys.argv) not in (5, 6):
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
         int(sys.argv[5]) if len(sys.argv) == 6 else 1)
