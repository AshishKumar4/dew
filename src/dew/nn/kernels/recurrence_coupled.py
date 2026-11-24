"""A coupled momentum's chunked op (`recurrence.Hooks.coupled`, row 1454; fla's momentum_delta_rule, which
has no kernel): Momentum DeltaNet's memory and momentum, a pair of states a head under a 2x2 triangular
transition of scalars, on the portable math's walk (`recurrence_chunk.xla_mainloop`) of the forms its token
update derives (`dew.algebra.coupled_scan`), with the chunked op's own padding, packed documents' resets and
token entry."""

from __future__ import annotations

import math

import jax.numpy as jnp

from dew.algebra import scan

from . import recurrence
from .recurrence_chunk import RESET, Chunk, Tiles, xla_mainloop


def coupled_chunk(hooks: recurrence.Hooks, given: dict, query, key, value, g, beta, operands: Chunk, *,
                  chunk_size: int, scale: float, work):
    """A coupled momentum's chunked op (`recurrence.Hooks.coupled`) on the portable math alone: `xla_mainloop`
    walking the pair's forms its token update derives (`declaration`), its erase key the keys' or
    its own (each key head's over its value heads), its packed documents' resets the walk's. `(output in
    work, final pair [B, Hv, 2, Dk, Dv])`."""
    if operands.loads is not None:
        raise ValueError("a coupled momentum's call walks no forked documents (`Chunk.loads`): its memory "
                         "and momentum enter a sequence's first token alone")
    erase = key if hooks.erase == 'tied' else given['erase_key']
    erase = jnp.repeat(erase, value.shape[2] // erase.shape[2], axis=2)
    return xla_mainloop(Tiles(chunk_size), query, key, value, g, beta, operands.state, operands.starts,
                        scale=scale, delta=True, dtype=work, bound=hooks.write_bound,
                        coupled=(erase, given['momentum_decay'].astype(work)))


def declaration(scale: float, width: int, columns: int, work, reset: bool, bound: float | None):
    """A coupled momentum's pair `(h, m)` (`recurrence.Hooks.coupled`, `dew.algebra.coupled_scan`), each
    `[width, columns]` in `work`, as a token updates it: the erase key's read of the memory less the values
    gathered into the momentum, `m' = exp(mu) m + k (p^T h - v)^T`, the memory stepping along it, `h' = exp(g)
    h - beta m'`, read `scale q^T h'`; its log decays at most 0 and its write gate at most `bound` (the
    layer's `write_bound`), which bound its chunks' sums; packed documents' resets where `reset`."""
    def update(pair, q, k, v, g, beta, p, mu):
        del q
        h, m = pair
        m = jnp.exp(mu) * m + k[:, None] * (p @ h - v)[None, :]
        return jnp.exp(g) * h - beta * m, m

    def readout(pair, q, k, v, g, beta, p, mu):
        del k, v, g, beta, p, mu
        return scale * (q @ pair[0])

    axes = {"q": ("time", "key"), "k": ("time", "key"), "v": ("time", "value"), "g": ("time",),
            "beta": ("time",), "p": ("time", "key"), "mu": ("time",)}
    declared = {**axes, RESET: ("time",)} if reset else axes
    ranges = {"g": (-math.inf, 0.0), "mu": (-math.inf, 0.0), **({} if bound is None else {"beta": bound})}
    return scan.Scan(update, readout, declared, walked="time", state=(((width, columns), work),) * 2,
                     reset=RESET if reset else None, ranges=ranges)
