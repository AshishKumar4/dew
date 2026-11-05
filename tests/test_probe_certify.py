"""Scratch probe: the certifier on main against the walker's, on the chunk ops."""
import jax
import jax.numpy as jnp
import pytest

import certified
import certified_main
from test_recurrence_chunk import CHUNK, portable_draw

from dew.nn.kernels import recurrence_chunk


def test_probe(monkeypatch):
    from dew.nn import precision

    monkeypatch.setenv(precision.WIDE_CPU_ARITHMETIC, "0")
    operands, starts, _ = portable_draw(0, 1, False)
    query, key, value, g, beta = operands

    def rule(query, key, value, g, beta, starts):
        return recurrence_chunk.chunk_delta_rule(query, key, value, g, beta, chunk_size=CHUNK, starts=starts)

    args = (query, key, value, g, beta, starts)
    _, old = certified_main.certified(rule, *args)
    _, new = certified.certified(rule, *args)
    print("\nPROBE old", len(old), old[:2])
    print("PROBE new", len(new), [f[:200] for f in new[:3]])
    closed = jax.make_jaxpr(rule)(*args)
    text = str(closed)
    i = text.find("marked_call")
    print("PROBE jaxpr", text[i - 300:i + 1500].replace("\n", " | ")[:1500])
