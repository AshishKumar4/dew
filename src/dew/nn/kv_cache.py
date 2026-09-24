"""How a decode cache stores its keys and values: dense or paged, full or quantized.

`dew.nn.attention.open_kv_cache` owns the cursor and the compact slot
positions; this module owns the storage behind them. A `KVCache` value names
the layout, and `KVStore` reads and writes it inside one attention module.

Dense storage is one `[rows, capacity, kv_heads, head_dim]` block per row.
Paged storage is one pool of fixed-size pages shared by every row,
`[kv_heads, pages, page_size, head_dim]`, the layout the Pallas TPU paged
attention kernel reads (`jax.experimental.pallas.ops.tpu.paged_attention`),
and a
`[rows, capacity // page_size]` page table maps a row's slot `s` to page
`table[row, s // page_size]`, offset `s % page_size`. A row owns its pages
only through the table, so a server can hand pages out on demand, share the
pages of a common prompt prefix between rows, and hold more rows than
`pages * page_size / capacity` whenever they are shorter than the capacity.

A pool may be split into `groups`: the rows fall into that many equal
groups in row order, the pool into as many equal parts, and a row's table
names pages of its own group's part, counted from the start of that part.
Reads and writes run mapped over the groups, so the group is a batch
dimension of every gather and scatter into the pool. A server whose mesh
splits its rows `n` ways splits the pool into `n` groups, and each device
then reads and writes only the pages it holds. A pool indexed as one table
into the whole of a split pool would be gathered onto every device first.

A quantized cache stores int8 or float8 (e4m3) values with one float32
scale per token and head, the absmax over the head's features divided by
the format's largest value. Reads dequantize to the compute dtype.

An int8 cache stores its keys rotated by the orthonormal Hadamard matrix
over head_dim, as QuaRot (Ashkboos et al. 2024, arXiv:2404.00456) rotates
its KV cache. A checkpoint whose keys carry a few outlier channels (Qwen3's
do) otherwise spends the one per-token scale on those channels, and the rest
round to a handful of levels; rotated, every channel carries an even share
of the outlier. The keys stay rotated when read, and the query takes the
same rotation (`Append.query`), which leaves every logit as it was, since
`H Hᵀ = I`: one `[rows, heads, head_dim]` product per step, not one over the
whole cache. Values are not rotated. Float8 keys are not rotated either:
e4m3 rounds each element relative to itself, so an outlier costs the other
channels nothing, and spreading it made things worse. On Qwen3-0.6B over
wikitext-2, against the bf16 cache: int8 unrotated +0.66 perplexity,
rotated +0.008; float8 unrotated -0.03, rotated +1.03. The full table is in
docs/concepts/inference.md.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Literal, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.nn.scatter import DROPPED

KVDtype = Literal["int8", "float8_e4m3fn"]
"""The storage formats a quantized cache takes."""

TABLE = "page_table"
"""A paged cache's per-row page table, `[rows, capacity // page_size]`."""
CURSOR = "cache_index"
"""The per-row count of tokens a cache holds (`attention._cache_positions`)."""
VALIDITY = "cache_valid"
"""The per-row mask of filled slots, `[rows, capacity]`."""
POOLED = frozenset({"cached_key", "cached_value", "key_scale", "value_scale"})
"""The leaves a paged cache keeps in its shared pool rather than per row."""

_LIMITS = {"int8": 127.0, "float8_e4m3fn": float(jnp.finfo(jnp.float8_e4m3fn).max)}


def leaf_name(path: tuple[jax.tree_util.KeyEntry, ...]) -> str | None:
    """The variable name a cache leaf's path ends in."""
    if not path:
        return None
    last = path[-1]
    if not isinstance(last, jax.tree_util.DictKey):
        return None
    name = last.key
    return name if isinstance(name, str) else None


def is_paged(cache: Mapping[str, object]) -> bool:
    """Whether any layer of `cache` keeps a paged pool."""
    return any(leaf_name(path) == TABLE for path, _ in jax.tree_util.tree_leaves_with_path(cache))


@dataclasses.dataclass(frozen=True)
class KVCache:
    """The storage layout of a decode cache.

    `quantized` stores keys and values in that format with per-token,
    per-head scales; int8 keys are Hadamard-rotated, which needs a
    power-of-two head_dim. None keeps the compute dtype. `page_size` pages the cache:
    `capacity` has to be a multiple of it, and `pages` sizes the shared
    pool, None allocating one page per slot of every row the cache is
    opened for (as much memory as the dense layout). A smaller pool is a
    server's to hand out (`dew.inference.serving.Server`). `groups` splits
    a paged pool into that many parts, one per equal group of rows (see
    the module docstring); the row count and `pages` have to divide by it.
    """

    quantized: KVDtype | None = None
    page_size: int | None = None
    pages: int | None = None
    groups: int = 1

    def __post_init__(self) -> None:
        if self.quantized is not None and self.quantized not in _LIMITS:
            raise ValueError(f"quantized is one of {sorted(_LIMITS)} or None, got {self.quantized!r}")
        if self.page_size is not None and (type(self.page_size) is not int or self.page_size < 1):
            raise ValueError(f"page_size must be a positive integer, got {self.page_size!r}")
        if self.pages is not None:
            if self.page_size is None:
                raise ValueError("a page count needs a page_size")
            if type(self.pages) is not int or self.pages < 1:
                raise ValueError(f"pages must be a positive integer, got {self.pages!r}")
        if type(self.groups) is not int or self.groups < 1:
            raise ValueError(f"groups must be a positive integer, got {self.groups!r}")
        if self.groups > 1 and self.page_size is None:
            raise ValueError("groups split a paged pool; a dense cache is already split by its rows")
        if self.pages is not None and self.pages % self.groups:
            raise ValueError(f"a pool of {self.pages} pages does not split into {self.groups} equal groups")

    def storage(self, dtype: jnp.dtype) -> jnp.dtype:
        """The dtype the cache holds values of `dtype` in."""
        return jnp.dtype(dtype) if self.quantized is None else jnp.dtype(self.quantized)

    def key_rotation(self, head_dim: int) -> jax.Array | None:
        """The rotation stored keys carry, and a query has to take; None when unrotated."""
        return hadamard(head_dim) if self.quantized == "int8" else None


def quantize(values: jax.Array, dtype: KVDtype) -> tuple[jax.Array, jax.Array]:
    """`values` `[..., head_dim]` in `dtype`, and the float32 scale per vector.

    The scale maps the vector's absmax to the format's largest value; an
    all-zero vector keeps scale one so it dequantizes to zeros.
    """
    wide = values.astype(jnp.float32)
    amax = jnp.max(jnp.abs(wide), axis=-1)
    scale = jnp.where(amax > 0, amax / _LIMITS[dtype], 1.0)
    scaled = wide / scale[..., None]
    if dtype == "int8":
        return jnp.clip(jnp.rint(scaled), -127, 127).astype(jnp.int8), scale
    return scaled.astype(jnp.dtype(dtype)), scale


def dequantize(stored: jax.Array, scale: jax.Array, dtype: jnp.dtype) -> jax.Array:
    return (stored.astype(jnp.float32) * scale[..., None]).astype(dtype)


def hadamard(size: int) -> jax.Array:
    """Sylvester's Hadamard matrix of order `size`, scaled to be orthonormal and symmetric."""
    if size < 1 or size & (size - 1):
        raise ValueError(f"an int8 KV cache rotates keys by a Hadamard matrix, which needs a "
                         f"power-of-two head_dim; got {size}. quantized='float8_e4m3fn' takes any width")
    matrix = jnp.ones((1, 1), jnp.float32)
    while matrix.shape[0] < size:
        matrix = jnp.block([[matrix, matrix], [matrix, -matrix]])
    return matrix / math.sqrt(size)


def rotated(values: jax.Array, rotation: jax.Array) -> jax.Array:
    """`values` `[..., head_dim]` times the rotation, computed in float32, in `values`' dtype."""
    product = jnp.einsum("...d,de->...e", values.astype(jnp.float32), rotation,
                         precision=jax.lax.Precision.HIGHEST)
    return product.astype(values.dtype)


def write_cache(buffer: jax.Array, values: jax.Array, positions: jax.Array) -> jax.Array:
    """`values` `[rows, tokens, ...]` written at per-row slots `positions`; a slot of -1 drops.

    The write is mapped over rows, so the row is a batch dimension of the
    scatter. GSPMD splits such a scatter wherever the rows split, with no
    collective; written as one scatter indexed by an iota over the rows, it
    gathers the values and indices of every row onto every device first.
    """
    slots = jnp.where(positions >= 0, positions, DROPPED)
    return jax.vmap(lambda row, incoming, at: row.at[at].set(incoming.astype(row.dtype), mode="drop"))(
        buffer, values, slots)


def filled_slots(cursor: jax.Array, capacity: int) -> jax.Array:
    """Which of `capacity` slots hold a token, `[rows, capacity]`, for per-row cursors `[rows]`."""
    return jnp.arange(capacity)[None, :] < cursor[:, None]


@runtime_checkable
class Layered(Protocol):
    """A model that declares its decode cache's layout, as `CausalTransformer` does."""

    @property
    def kv_cache(self) -> KVCache: ...


def refuse_unassigned(layout: KVCache, rows: int, capacity: int) -> None:
    """Refuse a paged pool that cannot give each of `rows` rows its whole `capacity`.

    A caller that writes through the default page tables, as generation
    does, needs a private block of pages for every row. Jax's checkify
    cannot carry a device check into a model layer's shard_map
    (jax-ml/jax#40907), so the store checks no write, and the shapes decide
    it here instead, before any write. A server hands a smaller pool out
    itself (`dew.inference.serving.Server`).
    """
    if layout.page_size is None or layout.pages is None:
        return
    needed = rows * (capacity // layout.page_size)
    if layout.pages < needed:
        raise ValueError(f"{rows} rows of {capacity} slots need {needed} pages; the pool holds "
                         f"{layout.pages}. A pool smaller than every row's capacity needs a server "
                         "to assign its pages")


def default_page_table(rows: int, per_row: int, pages: int, groups: int) -> jax.Array:
    """Each row owns a private contiguous block of `per_row` pages in its group's part.

    Row `r` of a group holds pages `r * per_row` onward, counted from the
    start of the part. That is what a cache opened outside a server reads,
    over the default pool of one page per slot of every row. A smaller pool
    cannot give every row a block: the pages past the end of a part read as
    the part's size, an index no part holds, and a write through one is
    dropped, so `refuse_unassigned` refuses such a pool before any write. A
    server writes its rows' tables itself, so it never writes through them.
    """
    part = pages // groups
    table = jnp.tile(jnp.arange(rows // groups * per_row, dtype=jnp.int32).reshape(rows // groups, per_row),
                     (groups, 1))
    return jnp.where(table < part, table, part)


def grouped(array: jax.Array, axis: int, groups: int) -> jax.Array:
    """`array` with `axis` split in two, `[groups, size // groups]`, the group outer."""
    return array.reshape(*array.shape[:axis], groups, array.shape[axis] // groups, *array.shape[axis + 1:])


@dataclasses.dataclass(frozen=True)
class KVStore:
    """One attention module's key and value storage, opened for `rows` rows.

    `write` puts keys and values `[rows, tokens, kv_heads, head_dim]` at
    compact slot positions `[rows, tokens]`, dropping the ones at -1.
    `read` returns every slot of every row, `[rows, capacity, kv_heads,
    head_dim]`, dequantized and with the keys in the stored rotation: a
    slot no token has reached holds whatever its page last held, which the
    caller's validity mask excludes.
    """

    module: nn.Module
    layout: KVCache
    rows: int
    capacity: int
    kv_heads: int
    head_dim: int
    dtype: jnp.dtype

    @classmethod
    def open(cls, module: nn.Module, layout: KVCache, rows: int, capacity: int, kv_heads: int,
             head_dim: int, dtype: jnp.dtype) -> KVStore:
        """Declare the module's cache variables, allocating them on first use."""
        store = cls(module, layout, rows, capacity, kv_heads, head_dim, jnp.dtype(dtype))
        layout.key_rotation(head_dim)
        storage = layout.storage(dtype)
        if layout.page_size is None:
            shape = (rows, capacity, kv_heads, head_dim)
        else:
            if capacity % layout.page_size:
                raise ValueError(f"a paged cache of {capacity} slots needs a multiple of "
                                 f"page_size {layout.page_size}")
            per_row = capacity // layout.page_size
            if rows % layout.groups:
                raise ValueError(f"{rows} rows do not split into the pool's {layout.groups} groups")
            pages = rows * per_row if layout.pages is None else layout.pages
            shape = (kv_heads, pages, layout.page_size, head_dim)
            module.variable("cache", TABLE, default_page_table, rows, per_row, pages, layout.groups)
        for name in ("cached_key", "cached_value"):
            module.variable("cache", name, jnp.zeros, shape, storage)
        if layout.quantized is not None:
            for name in ("key_scale", "value_scale"):
                module.variable("cache", name, jnp.ones, shape[:-1], jnp.float32)
        return store

    @property
    def rotation(self) -> jax.Array | None:
        return self.layout.key_rotation(self.head_dim)

    def _get(self, name: str) -> jax.Array:
        return self.module.get_variable("cache", name)

    def _put(self, name: str, value: jax.Array) -> None:
        self.module.put_variable("cache", name, value)

    def write(self, key: jax.Array, value: jax.Array, positions: jax.Array) -> None:
        rotation = self.rotation
        if rotation is not None:
            key = rotated(key, rotation)
        pairs = [("cached_key", "key_scale", key), ("cached_value", "value_scale", value)]
        for name, scale_name, incoming in pairs:
            if self.layout.quantized is None:
                stored, scale = incoming.astype(self.layout.storage(self.dtype)), None
            else:
                stored, scale = quantize(incoming, self.layout.quantized)
            self._put(name, self._written(self._get(name), stored, positions))
            if scale is not None:
                self._put(scale_name, self._written(self._get(scale_name), scale, positions))

    def _written(self, buffer: jax.Array, values: jax.Array, positions: jax.Array) -> jax.Array:
        """`values` `[rows, tokens, heads, ...]` stored at `positions`; -1 drops."""
        if self.layout.page_size is None:
            return write_cache(buffer, values, positions)
        page_size, groups = self.layout.page_size, self.layout.groups
        safe = jnp.maximum(positions, 0)
        page = jnp.take_along_axis(self._get(TABLE), safe // page_size, axis=1)
        page = jnp.where(positions >= 0, page, DROPPED)

        def stored(pool: jax.Array, page: jax.Array, offset: jax.Array, incoming: jax.Array) -> jax.Array:
            # [rows, tokens, heads, ...] -> [heads, rows, tokens, ...] for the pool's head-major index.
            return pool.at[:, page, offset].set(jnp.moveaxis(incoming, 2, 0), mode="drop")

        return jax.vmap(stored, in_axes=(1, 0, 0, 0), out_axes=1)(
            grouped(buffer, 1, groups), grouped(page, 0, groups), grouped(safe % page_size, 0, groups),
            grouped(values, 0, groups)).reshape(buffer.shape)

    def read(self) -> tuple[jax.Array, jax.Array]:
        return self._read("cached_key", "key_scale"), self._read("cached_value", "value_scale")

    def _read(self, name: str, scale_name: str) -> jax.Array:
        stored = self._get(name)
        scale = None if self.layout.quantized is None else self._get(scale_name)
        if self.layout.page_size is not None:
            stored = self._gathered(stored)
            scale = None if scale is None else self._gathered(scale)
        return stored.astype(self.dtype) if scale is None else dequantize(stored, scale, self.dtype)

    def _gathered(self, pool: jax.Array) -> jax.Array:
        """Every row's slots of `pool` `[heads, pages, page_size, ...]`, as `[rows, capacity, heads, ...]`."""
        groups = self.layout.groups
        rows = jax.vmap(lambda part, table: part[:, table], in_axes=(1, 0), out_axes=1)(
            grouped(pool, 1, groups), grouped(self._get(TABLE), 0, groups))
        # [heads, groups, rows / groups, per_row, page_size, ...] -> [rows, capacity, heads, ...]
        return jnp.moveaxis(rows.reshape(self.kv_heads, self.rows, self.capacity, *pool.shape[3:]), 0, 2)

    def kernel(self) -> bool:
        """Whether decode runs the Pallas TPU paged kernel rather than the XLA gather.

        The kernel reads a bfloat16 pool: it casts any other page dtype to
        bfloat16 on the way in, and its int8 path broadcasts the scales to
        the pool's full width first, so a float32 or quantized pool takes
        the gather. A GPU takes the gather too: jax deprecated its Triton
        paged kernel (`jax.experimental.pallas.ops.gpu.paged_attention`),
        which ran 2% faster than the gather on an A100. A pool split into
        groups takes the gather as well: the kernel indexes one pool with
        one table, and the grouped gather is what keeps each group's pages
        on the device that holds them.
        """
        return (self.layout.page_size is not None and self.layout.quantized is None
                and self.layout.groups == 1 and self.dtype == jnp.bfloat16 and jax.default_backend() == "tpu")

    def decode(self, query: jax.Array, lengths: jax.Array, softcap: float | None) -> jax.Array:
        """One query per row `[rows, heads, head_dim]` against the first
        `lengths` slots of each row, through the Pallas TPU paged kernel.

        The kernel does not scale the logits, so the query carries
        1/sqrt(head_dim) as every other attention path applies it.
        """
        from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention

        scaled = query * jnp.asarray(1.0 / math.sqrt(self.head_dim), query.dtype)
        table = self._get(TABLE)
        return paged_attention(scaled, self._get("cached_key"), self._get("cached_value"), lengths, table,
                               attn_logits_soft_cap=softcap, pages_per_compute_block=_pages_per_block(table.shape[1]))


def _pages_per_block(pages: int) -> int:
    """How many of a row's `pages` the TPU paged kernel attends in one block:
    the largest divisor of `pages` up to 8, since the kernel needs the block
    to divide a row's pages. The gcd with 8 gave a row of three pages, a
    capacity of 384 at 128 a page, one-page blocks."""
    return max(block for block in range(1, 9) if pages % block == 0)


@dataclasses.dataclass(frozen=True)
class Append:
    """`open_kv_cache`'s writer: store a call's keys and values, then read the cache.

    The allocation-only call (the cache did not exist yet) stores nothing.
    The keys it reads back carry the store's rotation, so the query that
    attends to them goes through `query` first.
    """

    store: KVStore
    positions: jax.Array
    allocated: bool

    def __call__(self, key: jax.Array, value: jax.Array) -> tuple[jax.Array, jax.Array]:
        if self.allocated:
            self.store.write(key, value, self.positions)
        return self.store.read()

    def query(self, query: jax.Array) -> jax.Array:
        """`query` `[..., head_dim]` in the basis the stored keys are in."""
        rotation = self.store.rotation
        return query if rotation is None else rotated(query, rotation)
