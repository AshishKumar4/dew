"""The device side of a server's host tier (`dew.inference.pages.HostTier`): copying the pages its
prefix cache reclaims out to pinned host memory, and a hit's pages back into the pool.

A page's payload is every pooled leaf (`dew.nn.kv_cache.POOLED`: the keys, the values, their scales
and stored places) of every layer, sliced at the page. A copy out gathers an admission's reclaimed
pages from the pool the step about to run will donate, so it reads the keys before they are written
again, into fresh device arrays that a `device_put` then sends to `pinned_host`, each leaf sharded as
the pool's is. A load sends a host batch back to the device and, once it has landed, scatters the
rows a request takes into its pages, donating the pool's leaves in their own formats, so the
serving programs read them as they left them. Both programs compile at the server's start, once per
power-of-two count of pages, never on an admission.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.layout import Format

from dew.inference.serving_kernel import Formats, Slots
from dew.nn.kv_cache import POOLED, leaf_name

HOST = "pinned_host"


def buckets(most: int) -> tuple[int, ...]:
    """The page counts the programs compile at: powers of two up to the first that covers `most`."""
    counts = [1]
    while counts[-1] < most:
        counts.append(2 * counts[-1])
    return tuple(counts)


def holds_host_memory(leaves: list[jax.Array]) -> bool:
    """Whether every device the pool lives on also addresses pinned host memory."""
    return all(any(memory.kind == HOST for memory in device.addressable_memories())
               for leaf in leaves for device in leaf.sharding.device_set)


@runtime_checkable
class Settled(Protocol):
    """An array that says, without waiting, whether its value is computed. JAX's arrays do
    (`is_ready`), though `jax.Array` does not declare it."""

    def is_ready(self) -> bool: ...


def landed(arrays: Sequence[jax.Array]) -> bool:
    """Whether every one of `arrays` holds its value, asked without waiting."""
    return all(array.is_ready() for array in arrays if isinstance(array, Settled))


class PagesOnHost:
    """The copy programs of one server's pool: `pool` the resident state's pooled leaves by path,
    `formats` their formats in the serving programs (`dew.inference.serving_kernel.resident_formats`),
    `most` the most pages one copy moves."""

    def __init__(self, pool: dict[tuple, jax.Array], formats: dict[tuple, Format], most: int) -> None:
        self.paths = tuple(pool)
        leaves = [pool[path] for path in self.paths]
        self.formats = [formats[path] for path in self.paths]
        self.pages = leaves[0].shape[1]
        rows = [jax.ShapeDtypeStruct((leaf.shape[0], 1, *leaf.shape[2:]), leaf.dtype) for leaf in leaves]
        self.page_bytes = sum(row.size * row.dtype.itemsize for row in rows)
        """Bytes one page holds across every pooled leaf of every layer."""
        self.shardings = [leaf.sharding for leaf in leaves]
        self.host = [sharding.with_memory_kind(HOST) for sharding in self.shardings]
        self.counts = buckets(most)
        self._gather, self._scatter = {}, {}
        for count in self.counts:
            ids = jax.ShapeDtypeStruct((count,), jnp.int32)
            taken = [jax.ShapeDtypeStruct((leaf.shape[0], count, *leaf.shape[2:]), leaf.dtype,
                                          sharding=sharding) for leaf, sharding in zip(leaves, self.shardings,
                                                                                         strict=True)]
            self._gather[count] = jax.jit(
                _gathered, in_shardings=(self.formats, None), out_shardings=self.shardings).lower(
                    leaves, ids).compile()
            self._scatter[count] = jax.jit(
                _scattered, donate_argnums=0, in_shardings=(self.formats, self.shardings, None, None),
                out_shardings=self.formats).lower(leaves, taken, ids, ids).compile()

    def copy_out(self, pool: dict[tuple, jax.Array], pages: list[int]) -> tuple[list[jax.Array], int]:
        """`pages` of the pool, every pooled leaf's rows, on their way to pinned host memory, and the
        bytes they hold. The gather reads the pool as it is now, so it runs before whatever donates it
        next."""
        count = self._count(len(pages))
        ids = np.zeros((count,), np.int32)
        ids[:len(pages)] = pages
        gathered = self._gather[count]([pool[path] for path in self.paths], ids)
        return jax.device_put(gathered, self.host), count * self.page_bytes

    def stage(self, payload: list[jax.Array]) -> list[jax.Array]:
        """A host batch on its way back to the devices, each leaf where the pool's lives."""
        return jax.device_put(payload, self.shardings)

    def load(self, pool: dict[tuple, jax.Array], staged: list[jax.Array], rows: list[int],
             pages: list[int]) -> dict[tuple, jax.Array]:
        """The pool with `staged`'s `rows` written into `pages`, its leaves donated. A staged batch is
        a bucket's width (`copy_out`), and the rows a request takes are among its rows, so the scatter
        runs at that width."""
        count = staged[0].shape[1]
        sources = np.zeros((count,), np.int32)
        sources[:len(rows)] = rows
        targets = np.full((count,), self.pages, np.int32)
        targets[:len(pages)] = pages
        written = self._scatter[count]([pool[path] for path in self.paths], staged, sources, targets)
        return dict(zip(self.paths, written, strict=True))

    def round_trip(self, pool: dict[tuple, jax.Array], pages: list[int]) -> tuple[dict, float]:
        """Seconds a page takes out to the host and back into the pool, measured on `pages` (whose
        keys it writes back as they were), and the pool after."""
        payload, _ = self.copy_out(pool, pages)
        jax.block_until_ready(payload)
        began = time.perf_counter()
        payload, _ = self.copy_out(pool, pages)
        pool = self.load(pool, self.stage(payload), list(range(len(pages))), pages)
        jax.block_until_ready(pool)
        return pool, (time.perf_counter() - began) / len(pages)

    def _count(self, needed: int) -> int:
        return next(count for count in self.counts if count >= needed)


def _gathered(leaves: list[jax.Array], ids: jax.Array) -> list[jax.Array]:
    return [jnp.take(leaf, ids, axis=1) for leaf in leaves]


def _scattered(leaves: list[jax.Array], staged: list[jax.Array], rows: jax.Array, pages: jax.Array
               ) -> list[jax.Array]:
    return [leaf.at[:, pages].set(jnp.take(part, rows, axis=1), mode="drop")
            for leaf, part in zip(leaves, staged, strict=True)]


def pool_of(state: Slots) -> dict[tuple, jax.Array]:
    """The pooled leaves of a server's resident state, by path (`dew.nn.kv_cache.POOLED`)."""
    return _pooled(state)


def pool_formats(formats: Formats) -> dict[tuple, Format]:
    """The formats of the pooled leaves, by path, in the serving programs' `formats`."""
    return _pooled(formats)


def _pooled(tree: Slots) -> dict:
    flat, _ = jax.tree_util.tree_flatten_with_path(tree, is_leaf=lambda leaf: leaf is None)
    return {tuple(path): leaf for path, leaf in flat if leaf is not None and leaf_name(path) in POOLED}


def with_pool(state: Slots, pool: dict[tuple, jax.Array]) -> Slots:
    """`state` with its pooled leaves replaced by `pool`'s."""
    flat, structure = jax.tree_util.tree_flatten_with_path(state, is_leaf=lambda leaf: leaf is None)
    return jax.tree_util.tree_unflatten(structure, [pool.get(tuple(path), leaf) for path, leaf in flat])
