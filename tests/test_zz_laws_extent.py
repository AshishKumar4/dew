"""Scratch probe (not for landing): a law certificate's seconds at a walked extent of 2 and of 3 per tile."""
import time

import jax.numpy as jnp
import pytest

import algebra_laws as laws
import test_algebra_laws as t


@pytest.mark.parametrize("extent", [2, 3])
def test_probe(monkeypatch, extent):
    original = laws.reduced

    def reduced(shapes, walked):
        made = original(shapes, walked)
        return {name: (tuple(extent if size == 2 and axis == walked.get(name) else size
                             for axis, size in enumerate(shape)), dtype)
                for name, (shape, dtype) in made.items()}

    monkeypatch.setattr(laws, "reduced", reduced)
    lines = []
    for label, (declaration, shapes) in {"forward": t._forward(False, False, jnp.float32),
                                         "backward tile": t._cotangents(False, False, False)}.items():
        tile = (2, 2, 2) if label == "forward" else (2, 2, 0)
        began = time.perf_counter()
        laws.certified(declaration, shapes(*tile))
        lines.append(f"{label} extent {extent}: {time.perf_counter() - began:.1f} s")
    raise AssertionError("\n".join(lines))
