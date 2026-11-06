"""Scratch probe (not for landing): a law certificate's seconds at 2 and at 3 walked elements a tile."""
import time

import jax.numpy as jnp
import pytest

import algebra_laws as laws
import test_algebra_laws as t


@pytest.mark.parametrize("keys", [2, 3])
def test_probe(keys):
    lines = []
    for label, (declaration, shapes), tile in (
            ("forward", t._forward(False, False, jnp.float32), (2, keys, 4)),
            ("forward raw", t._forward(False, False, jnp.float32, 0.125), (2, keys, 4)),
            ("backward tile", t._cotangents(False, False, False), (2, keys, 0)),
            ("backward keys", t._backward("keys", False, False, jnp.float32, 64)[:2], (2, keys, 4))):
        began = time.perf_counter()
        laws.certified(declaration, shapes(*tile))
        lines.append(f"{label} keys {keys}: {time.perf_counter() - began:.1f} s")
    raise AssertionError("\n".join(lines))
