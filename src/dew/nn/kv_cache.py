"""How a decode cache stores its keys and values: dense or paged, full precision or quantized.

`open_kv_cache` keeps the cursor and the compact slot positions. A `KVCache`
value names the storage layout behind them, and `KVStore` reads and writes
that storage inside one attention module.

Dense storage is one `[rows, capacity, kv_heads, head_dim]` block per row.
Paged storage is one pool `[kv_heads, pages, page_size, head_dim]` shared by
every row, which is the layout the Pallas TPU paged attention kernel reads.
A `[rows, capacity // page_size]` page table maps slot `s` to page
`table[row, s // page_size]` at offset `s % page_size`. With pages, a server
can assign pages on demand, share the pages of a common prefix, and hold more
than `pages * page_size / capacity` rows while they are short. A pool split
into `groups` gives each equal group of rows its own part of the pool,
indexed from the part's start. A server whose mesh splits rows `n` ways then
reads and writes only the pages each device holds, and does not gather the
whole pool.

A quantized cache stores int8 or float8 (e4m3) values with one float32 scale
per token and head, and dequantizes them on read. The scale is the absmax
over the head's features divided by the format's largest value. int8 keys
are stored rotated by the orthonormal Hadamard matrix over head_dim, the way
QuaRot (arXiv:2404.00456) rotates its cache, so a few outlier channels
(Qwen3's keys) do not set the whole per-token scale. The query gets the same
rotation (`Append.query`), and every logit stays the same because
`H Hᵀ = I`. Values and float8 keys are not rotated, because e4m3 rounds each
element relative to itself. On Qwen3-0.6B over wikitext-2, compared with a
bf16 cache, perplexity changed by +0.66 for unrotated int8 and +0.008 for
rotated int8, and by -0.03 for unrotated float8 and +1.03 for rotated float8
(docs/concepts/inference.md).
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Mapping
from typing import Literal

import jax
import jax.numpy as jnp
import scipy.linalg
from flax import linen as nn
from jax.sharding import PartitionSpec as P

from dew.mesh import LayoutRefused, logical_spec, mesh_axes
from dew.nn.kernels import decode_attention, generation
from dew.nn.kernels.generation import INHERENT, backend
from dew.nn.kernels.selection import Kernel, Op, Requirement, selected
from dew.nn.precision import float32_einsum
from dew.nn.scatter import DROPPED
from dew.nn.sharding import manual_map, sequence_shards

KVDtype = Literal["int8", "float8_e4m3fn"]
"""The storage formats a quantized cache takes."""

TABLE = "page_table"
"""A paged cache's per-row page table, `[rows, capacity // page_size]`."""
CURSOR = "cache_index"
"""The per-row count of tokens a cache holds (`cache_positions`)."""
POSITION = "next_position"
"""A wrapper's per-row next position, past the last its row holds (`dew.nn.inputs.next_position`),
which for a text row is its cursor's count of tokens."""
ROW_CURSORS = frozenset({CURSOR, POSITION})
"""The per-row cursors a paged server seats a row at, from the row's own place in its pages."""
PLACES = "cached_places"
"""Each cached key's place, kept beside the key where a chunk or ALiBi reads it (`Append.places`)."""
GROUPS = {"image_groups": "cached_image_groups", "block_groups": "cached_block_groups"}
"""Each cached key's image or block group (`dew.nn.inputs.AttentionMetadata`), kept beside the key where a
layer reads it (`Append.groups`), by the field's name."""
POOLED = frozenset({"cached_key", "cached_value", "key_scale", "value_scale", PLACES, *GROUPS.values()})
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


def overflowed(cache: Mapping[str, object], capacity: int) -> jax.Array:
    """`[rows]`: whether a row's decode calls appended more real tokens than its cache's `capacity`.

    Every cursor counts each real token a call appends, writes past `capacity` drop, and the cursor
    never falls back, so a cursor past capacity is the row's lasting record that later calls read an
    incomplete prefix. `generate` and the server refuse such a prompt on the host first; a caller
    appending with `apply(decode=True)` itself checks this after its calls."""
    beyond = [jnp.any(leaf.reshape(-1, leaf.shape[-1]) > capacity, axis=0)
              for path, leaf in jax.tree_util.tree_leaves_with_path(cache) if leaf_name(path) == CURSOR]
    if not beyond:
        raise ValueError("this cache keeps no cursor, so it has no capacity to overflow")
    return functools.reduce(jnp.logical_or, beyond)


def is_paged(cache: Mapping[str, object]) -> bool:
    """Whether any layer of `cache` keeps a paged pool."""
    return any(leaf_name(path) == TABLE for path, _ in jax.tree_util.tree_leaves_with_path(cache))


@dataclasses.dataclass(frozen=True)
class KVCache:
    """The storage layout of a decode cache.

    `quantized` stores keys and values in that format with per-token, per-head
    scales. int8 keys are Hadamard-rotated, which needs a power-of-two
    head_dim. `page_size` pages the cache, and `capacity` must be a multiple of
    it. `pages` is the size of the pool. None gives every row enough pages for
    its whole capacity, and a smaller pool is for a server to assign
    (`dew.inference.serving.Server`). `groups` splits a paged pool among equal
    groups of rows, so the row count and `pages` must both divide by it.
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
        """Return the dtype that the cache stores values of `dtype` in."""
        return jnp.dtype(dtype) if self.quantized is None else jnp.dtype(self.quantized)

    def key_rotation(self, head_dim: int) -> jax.Array | None:
        """Return the rotation that stored keys are in and a query must take, or None for unrotated keys."""
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
    return jnp.asarray(scipy.linalg.hadamard(size), jnp.float32) / math.sqrt(size)


def rotated(values: jax.Array, rotation: jax.Array) -> jax.Array:
    """`values` `[..., head_dim]` times the rotation, computed in float32, in `values`' dtype."""
    product = float32_einsum("...d,de->...e", values.astype(jnp.float32), rotation,
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
    if values.shape[1] >= buffer.shape[1]:
        # A write as wide as the buffer (a prefill into a cache at the
        # prompt's width) gathers each slot's token instead: XLA lays such a
        # buffer out with its slots minor for the attention that reads it,
        # and a scatter of whole tokens into that layout ran at 28 GB/s, 8 ms
        # of a 40 ms admission step on an RTX 4080 (docs/performance.md).
        # Only the [rows, slots] map of which token lands where is scattered.
        tokens = jnp.broadcast_to(jnp.arange(values.shape[1], dtype=jnp.int32), slots.shape)
        source = jax.vmap(lambda at, index: jnp.full(buffer.shape[1], -1, jnp.int32).at[at].set(
            index, mode="drop"))(slots, tokens)
        picked = jnp.take_along_axis(values, jnp.maximum(source, 0).reshape(
            *source.shape, *(1,) * (values.ndim - 2)), axis=1)
        held = (source >= 0).reshape(*source.shape, *(1,) * (values.ndim - 2))
        return jnp.where(held, picked.astype(buffer.dtype), buffer)
    words = jax.vmap(lambda row, incoming, at: row.at[at].set(incoming, mode="drop"))(
        as_words(buffer), as_words(values.astype(buffer.dtype)), slots)
    return from_words(words, buffer.dtype)


def write_tokens(buffer: jax.Array, values: jax.Array, rows: jax.Array, positions: jax.Array) -> jax.Array:
    """`values` `[tokens, ...]` written to `buffer` `[rows, slots, ...]`, each
    at its own row and slot; a row past the buffer's or a slot of -1 drops.

    A serving step's mixed call (`dew.nn.inputs.Admitted`) writes its
    decoding rows' tokens and its prompts' in this one scatter, as words.
    Rows and slots index the leading axes, which the CUDA test lane checks
    under --xla_gpu_deterministic_ops (`KVStore.write_tokens` says why a
    pool's write is mapped)."""
    slots = jnp.where(positions >= 0, positions, DROPPED)
    written = as_words(buffer).at[rows, slots].set(as_words(values.astype(buffer.dtype)), mode="drop")
    return from_words(written, buffer.dtype)


def as_words(x: jax.Array) -> jax.Array:
    """`x`'s bits with its last axis packed into uint32 words, for a scatter
    to move on a GPU.

    XLA's scatter stores one element a thread, so bf16 caches moved two bytes
    a store: Qwen3-0.6B's 8-prompt admission write took 0.63 against 0.28 ms
    as words over 16 caches on an RTX 4080, the same bits (docs/performance.md).
    `x` itself where its elements are bool or a word wide already, its last
    axis does not fill whole words, or the backend is not a GPU, where a TPU
    tiles two-byte arrays on another axis.
    """
    per_word = 4 // x.dtype.itemsize
    if backend() != 'gpu' or x.dtype == jnp.bool_ or per_word < 2 or x.shape[-1] % per_word:
        return x
    pairs = x.reshape(*x.shape[:-1], x.shape[-1] // per_word, per_word)
    return jax.lax.bitcast_convert_type(pairs, jnp.uint32)


def from_words(words: jax.Array, dtype: jnp.dtype) -> jax.Array:
    """The `dtype` elements `as_words` packed into `words`."""
    if words.dtype == jnp.dtype(dtype):
        return words
    unpacked = jax.lax.bitcast_convert_type(words, dtype)
    # The last axis named, not inferred, so a leaf with no elements (a one-tap conv's history) keeps it.
    return unpacked.reshape(*words.shape[:-1], words.shape[-1] * unpacked.shape[-1])


def filled_slots(cursor: jax.Array, capacity: int) -> jax.Array:
    """Which of `capacity` slots hold a token, `[rows, capacity]`, for per-row cursors `[rows]`."""
    return jnp.arange(capacity)[None, :] < cursor[:, None]


def refuse_unassigned(layout: KVCache, rows: int, capacity: int) -> None:
    """Refuse a paged pool that cannot give each of `rows` rows its whole `capacity`.

    Writing through the default page tables, as generation does, needs a private
    block of pages per row. checkify cannot carry a device check into a layer's
    shard_map (jax-ml/jax#40907), so the shapes decide it here, before any
    write.
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

    Row `r` of a group holds pages `r * per_row` onward, as a cache opened
    outside a server reads. In a smaller pool the pages past a part's end read
    as the part's size, an index no part holds, so a write through one drops;
    `refuse_unassigned` refuses such a pool first. A server writes its own
    tables.
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
    """Reads and writes one attention module's key and value storage, opened for `rows` rows.

    `write` puts keys `[rows, tokens, kv_heads, head_dim]` and values as wide
    as `value_dim` at
    compact slot positions `[rows, tokens]` and drops the ones at -1. `read`
    returns every slot of every row, `[rows, capacity, kv_heads, head_dim]`,
    dequantized and with the keys in the stored rotation. A slot that no token
    has reached holds whatever its page last held, and the caller's validity
    mask excludes it.
    """

    module: nn.Module
    layout: KVCache
    rows: int
    kv_heads: int
    head_dim: int
    dtype: jnp.dtype

    @classmethod
    def open(cls, module: nn.Module, layout: KVCache, rows: int, capacity: int, kv_heads: int,
             head_dim: int, dtype: jnp.dtype, *, value_dim: int | None = None,
             places: bool = False, groups: tuple[str, ...] = ()) -> KVStore:
        """Declare the module's cache variables, allocating them on first use, and return the store.

        Values are `value_dim` wide, None for the keys' `head_dim`. `places`
        also keeps each token's place beside its key (`Append.places`), and
        `groups` each token's group of the fields it names (`GROUPS`,
        `Append.groups`), -1 where none has been written. A paged
        layout raises `ValueError` when `capacity` is not a multiple of
        `page_size`, or when `rows` does not split into the pool's groups.
        """
        store = cls(module, layout, rows, kv_heads, head_dim, jnp.dtype(dtype))
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
        values = (*shape[:-1], head_dim if value_dim is None else value_dim)
        for name, stored in (("cached_key", shape), ("cached_value", values)):
            module.variable("cache", name, jnp.zeros, stored, storage)
        if layout.quantized is not None:
            for name in ("key_scale", "value_scale"):
                module.variable("cache", name, jnp.ones, shape[:-1], jnp.float32)
        # A scale's storage with one head: every key head of a token sits at its place and in its group.
        held = (*shape[:2], 1) if layout.page_size is None else (1, *shape[1:3])
        if places:
            module.variable("cache", PLACES, jnp.zeros, held, jnp.int32)
        for name in groups:
            module.variable("cache", GROUPS[name], jnp.full, held, -1, jnp.int32)
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
        rows = jnp.broadcast_to(jnp.arange(positions.shape[0])[:, None], positions.shape)
        return self._paged(buffer, values, rows, positions, self.layout.page_size, self.layout.groups)

    def _paged(self, pool: jax.Array, values: jax.Array, rows: jax.Array, positions: jax.Array,
               size: int, groups: int) -> jax.Array:
        """The paged `pool` `[heads, pages, page_size, ...]` with each token's `values` `[..., heads,
        ...]` at its row's slot: `rows` and `positions` index the tokens alike, `[rows, tokens]` for a
        rectangular append or `[tokens]` for a mixed one. A slot at -1, past the row's pages
        (`overflowed`), or a row past the table's, drops. The tokens split into `groups` along their
        first axis, each group scattered into its part of the pool; `size` is the layout's page size."""
        table = self._get(TABLE)
        safe = jnp.maximum(positions, 0)
        inside = (positions >= 0) & (rows < table.shape[0]) & (safe // size < table.shape[1])
        column = jnp.where(inside, safe // size, 0)
        page = jnp.where(inside, table[jnp.where(inside, rows, 0), column], DROPPED)
        heads = positions.ndim

        def stored(pool: jax.Array, page: jax.Array, offset: jax.Array, incoming: jax.Array) -> jax.Array:
            # The tokens' heads lead, for the pool's head-major index.
            return pool.at[:, page, offset].set(jnp.moveaxis(incoming, heads, 0), mode="drop")

        # Mapped over the groups even for one: unmapped, under --xla_gpu_deterministic_ops (jax
        # 0.11.2) a dropped token's keys landed at another head's kept slot, the implicit-dimension
        # case openxla/xla#49498 fixes (issue #49380), after the jax pin. A jax bump past that fix
        # can write a group of one unmapped.
        return jax.vmap(stored, in_axes=(1, 0, 0, 0), out_axes=1)(
            grouped(pool, 1, groups), grouped(page, 0, groups), grouped(safe % size, 0, groups),
            grouped(values.astype(pool.dtype), 0, groups)).reshape(pool.shape)

    def plain(self, keys: jax.Array) -> jax.Array:
        """Return keys read back in the store's rotation `[..., head_dim]` in their own basis: for keys a
        projection expands before any query meets them (a latent attention's latents), or that serve as
        values too (DeepSeek V4's shared head), where no query is rotated into their basis."""
        return keys if self.rotation is None else rotated(keys, self.rotation.T)

    def read(self) -> tuple[jax.Array, jax.Array]:
        return self._read("cached_key", "key_scale"), self._read("cached_value", "value_scale")

    def write_places(self, places: jax.Array, positions: jax.Array) -> None:
        """Keep the tokens' `places` `[rows, tokens]` at their slots `positions`, where their keys go."""
        self._put(PLACES, self._written(self._get(PLACES), places[..., None].astype(jnp.int32), positions))

    def read_places(self) -> jax.Array:
        """Every slot's kept place, `[rows, capacity]`."""
        return self._read_slots(PLACES)

    def write_groups(self, name: str, groups: jax.Array, positions: jax.Array) -> None:
        """Keep the tokens' groups of field `name` `[rows, tokens]` at their slots `positions`, where their
        keys go (`GROUPS`)."""
        self._put(GROUPS[name], self._written(self._get(GROUPS[name]), groups[..., None].astype(jnp.int32),
                                              positions))

    def read_groups(self, name: str) -> jax.Array:
        """Every slot's kept group of field `name`, `[rows, capacity]`."""
        return self._read_slots(GROUPS[name])

    def _read_slots(self, name: str) -> jax.Array:
        """Every slot's entry of the one-head leaf `name`, `[rows, capacity]`."""
        held = self._get(name)
        return (held if self.layout.page_size is None else self._gathered(held))[..., 0]

    def write_tokens(self, key: jax.Array, value: jax.Array, rows: jax.Array, positions: jax.Array) -> None:
        """Write keys and values `[tokens, kv_heads, head_dim]`, each at its own row and slot.

        This is the write of a serving step's mixed call
        (`dew.nn.inputs.Admitted`). A token whose row is past the cache's rows,
        or whose slot is -1, is dropped. The cache must be full precision and
        unrotated, and a paged pool must have one group, as the mixed call
        requires.
        """
        for name, incoming in (("cached_key", key), ("cached_value", value)):
            pool = self._get(name)
            size = self.layout.page_size
            self._put(name, write_tokens(pool, incoming, rows, positions) if size is None
                      else self._paged(pool, incoming, rows, positions, size, 1))

    def read_rows(self, rows: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return what `read` returns for only `rows`, `[len(rows), capacity, kv_heads, head_dim]`.

        It does not dequantize, so it matches `read` only on a full-precision
        cache, which the mixed call that uses it requires.
        """
        def rows_of(name: str) -> jax.Array:
            stored = self._get(name)
            if self.layout.page_size is not None:
                return _gather_pages(stored, self._get(TABLE)[rows], 1).astype(self.dtype)
            return stored[rows].astype(self.dtype)

        return rows_of("cached_key"), rows_of("cached_value")

    def _read(self, name: str, scale_name: str) -> jax.Array:
        stored = self._get(name)
        scale = None if self.layout.quantized is None else self._get(scale_name)
        if self.layout.page_size is not None:
            stored = self._gathered(stored)
            scale = None if scale is None else self._gathered(scale)
        return stored.astype(self.dtype) if scale is None else dequantize(stored, scale, self.dtype)

    def _gathered(self, pool: jax.Array) -> jax.Array:
        """Every row's slots of `pool` `[heads, pages, page_size, ...]`, as `[rows, capacity, heads, ...]`."""
        return _gather_pages(pool, self._get(TABLE), self.layout.groups)

    def case(self, query: jax.Array, *, value_dim: int, softcap: float | None, arithmetic: bool,
             named: str, windowed: bool = False, sinks: bool = False) -> PagedCase:
        """The `PagedCase` of a decode step's query `[rows, heads, head_dim]` over this pool, its values
        `value_dim` wide, under the layer's `softcap`, its reference-only `arithmetic`
        (`dew.nn.attention.reference_only`) and its `AttentionImpl`, `named`; a `windowed` step reads
        each row from its window's first key; a layer with `sinks` has attention sinks."""
        return PagedCase(heads=query.shape[-2], kv_heads=self.kv_heads, head_dim=self.head_dim,
                         value_dim=value_dim, dtype=jnp.dtype(query.dtype), pool=jnp.dtype(self.dtype),
                         page_size=self.layout.page_size, quantized=self.layout.quantized is not None,
                         groups=self.layout.groups, head_shards=_head_shards(query), softcap=softcap,
                         arithmetic=arithmetic, named=named, windowed=windowed, sinks=sinks)

    def decode(self, query: jax.Array, lengths: jax.Array, softcap: float | None,
               starts: jax.Array | None = None, sinks: jax.Array | None = None) -> jax.Array:
        """Attend one query per row `[rows, heads, head_dim]` to the first `lengths` slots of that row,
        from `starts` `[rows]` where given (a window's first slot, which the GPU's kernel reads), with
        `sinks` `[heads]` where the layer has them (the GPU's kernel applies them).

        Decode runs the device's paged kernel, which raises `KernelRefused` where its requirements
        (`PAGED`) turn the step down. The TPU kernel does not scale
        the logits, so the query is multiplied by 1/sqrt(head_dim) first, the
        same scale every other attention path applies. A Mosaic kernel cannot
        be partitioned automatically, and neither can the GPU's Pallas kernel,
        so on a mesh each runs per shard (`manual_map`): each holds its rows'
        queries, lengths and page tables, and the key heads of the pool its
        query heads read, with every page, as the tables index the whole pool.
        """
        case = self.case(query, value_dim=self.head_dim, softcap=softcap, arithmetic=False, named='auto',
                         windowed=starts is not None, sinks=sinks is not None)
        # The device's paged kernel, named: one its requirements turn down raises.
        selected(PAGED, case, 'triton' if backend() == "gpu" else 'tpu')

        def local(query, key, value, lengths, table, *extras):
            # The optional operands as `decode` gave them: the window's starts, then the sinks.
            starts, sinks = (extras[0] if given else None), (extras[-1] if sunk else None)
            if backend() == "gpu":
                return gpu_paged(softcap)(query, key, value, table, lengths, starts, sinks)
            from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention

            scaled = query * jnp.asarray(1.0 / math.sqrt(self.head_dim), query.dtype)
            return paged_attention(scaled, key, value, lengths, table, attn_logits_soft_cap=softcap,
                                   pages_per_compute_block=_pages_per_block(table.shape[1]))

        given = () if starts is None else (starts,)
        sunk = () if sinks is None else (sinks,)
        arrays = (query, self._get("cached_key"), self._get("cached_value"), lengths, self._get(TABLE),
                  *given, *sunk)
        queries = logical_spec(DECODE_QUERY, query.shape)
        # A spec leaves off trailing whole dimensions.
        rows, heads = (*queries, None, None)[:2]
        specs = (queries, P(heads), P(heads), P(rows), P(rows), *(P(rows) for _ in given),
                 *(P(heads) for _ in sunk))
        return manual_map(local, specs, queries)(*arrays)


DECODE_QUERY = ("activation_batch", "activation_heads", None)
"""The logical axes of a decode query `[rows, heads, head_dim]`."""


@dataclasses.dataclass(frozen=True)
class PagedCase:
    """A decode step over a paged pool as the paged selector reads it (`PAGED`, `KVStore.case`): the
    query's `heads` over the pool's `kv_heads`, `head_dim` wide with values `value_dim` wide; the
    query's `dtype` and the `pool`'s; its `page_size` (None for a compact cache), whether it is
    `quantized`, its `groups`; how many ways a mesh splits the query heads (`head_shards`); the layer's
    `softcap`; whether its arithmetic is the reference path's alone (`arithmetic`); the
    `AttentionImpl` it `named`; whether it reads each row from its window's first key
    (`windowed`); and whether the layer has attention sinks (`sinks`)."""

    heads: int
    kv_heads: int
    head_dim: int
    value_dim: int
    dtype: jnp.dtype
    pool: jnp.dtype
    page_size: int | None
    quantized: bool
    groups: int
    head_shards: int
    softcap: float | None
    arithmetic: bool
    named: str
    windowed: bool = False
    sinks: bool = False


def _paged_layout(device: str) -> tuple[Requirement[PagedCase], ...]:
    """What either device's paged kernel needs of the pool and the step: pages of bf16, unquantized and
    ungrouped, keys and values of one width, a mesh's query-head split within the pool's key heads,
    and arithmetic other than the reference path's."""
    return (
        Requirement('pages', lambda case: case.page_size is not None, "the cache keeps no pages",
                    row=INHERENT),
        Requirement('quantized', lambda case: not case.quantized,
                    f"the {device} kernel reads bf16 pages; a quantized pool is read through the gather, "
                    "which dequantizes it", row='roadmap: paged decode over quantized pools'),
        Requirement('groups', lambda case: case.groups == 1,
                    "a grouped pool keeps each group's pages where they are, which the gather reads",
                    row='roadmap: paged decode over grouped pools'),
        Requirement('dtype', lambda case: case.pool == jnp.bfloat16 and case.dtype == case.pool,
                    lambda case: f"the {device} kernel reads a bf16 query over bf16 pages, not "
                                 f"{case.dtype} over {case.pool}", row=INHERENT),
        Requirement('widths', lambda case: case.value_dim == case.head_dim,
                    "the paged kernels read keys and values of one width; narrower values take the "
                    "gathered rows, which the attention core widens", row=INHERENT),
        Requirement('head_shards', lambda case: case.kv_heads % case.head_shards == 0,
                    "a mesh splits the query heads past the pool's key heads, whose pages the kernel reads "
                    "per shard", row=INHERENT),
        Requirement('arithmetic', lambda case: not case.arithmetic,
                    "the step asks for arithmetic only the reference path computes (`reference_only`)",
                    row=INHERENT),
    )


TRITON_PAGED_SUPPORTS: tuple[Requirement[PagedCase], ...] = (
    Requirement('capability', lambda case: generation.triton_runs(),
                "the paged decode kernel needs a GPU of sm80 or later", row=INHERENT),
    *_paged_layout('GPU'),
    decode_attention.PAGED_GEOMETRY,
)
"""Dew's Pallas paged decode kernel (`decode_attention.attend_paged`). Its derivatives at every order
are the gathered attention's, through the attention dispatch (`gpu_paged`), so neither a backward's
kernel nor forward mode limits it."""

TPU_PAGED_SUPPORTS: tuple[Requirement[PagedCase], ...] = (
    Requirement('capability', lambda case: generation.backend() == 'tpu',
                "the paged attention kernel runs on a TPU", row=INHERENT),
    *_paged_layout('TPU'),
    Requirement('windowed', lambda case: not case.windowed,
                "the TPU paged kernel reads every slot to a row's length, not a window's span",
                row='roadmap: TPU paged decode over windows'),
    Requirement('sinks', lambda case: not case.sinks,
                "jax's TPU paged attention has no sinks; the gathered rows' attention applies them",
                row=INHERENT),
)
"""jax's Pallas TPU paged attention, which applies a logit softcap itself."""


def _paged_preferred(case: PagedCase) -> str:
    """The device's paged kernel where the layer's `AttentionImpl` leaves the choice to it ('auto', or the
    fused kernel the device's own decode would take, 'cudnn' or 'tpu'), else the gathered rows through
    the layer's own kernel."""
    device = generation.backend()
    if device == 'gpu' and case.named in ('auto', 'cudnn'):
        return 'triton'
    if device == 'tpu' and case.named in ('auto', 'tpu'):
        return 'tpu'
    return 'gathered'


PAGED: Op[PagedCase] = Op('paged_decode', (
    Kernel('triton', TRITON_PAGED_SUPPORTS),
    Kernel('tpu', TPU_PAGED_SUPPORTS),
    Kernel('gathered', (), generic=True),
), _paged_preferred)
"""The paged selector: a decode step over a paged pool reads it through its device's paged kernel where
that kernel's requirements hold (`TRITON_PAGED_SUPPORTS`, `TPU_PAGED_SUPPORTS`), else the gathered rows
through the layer's attention (`CausalSelfAttention._decode_attention`)."""


def _head_shards(query: jax.Array) -> int:
    """How many ways the mesh splits a decode query's heads."""
    mesh = jax.sharding.get_abstract_mesh()
    heads = (*logical_spec(DECODE_QUERY, query.shape), None, None)[1]
    return math.prod(mesh.shape[axis] for axis in mesh_axes(heads))


def _gather_pages(pool: jax.Array, table: jax.Array, groups: int) -> jax.Array:
    """The existing grouped page read as `[rows, capacity, heads, ...]`."""
    rows = jax.vmap(lambda part, index: part[:, index], in_axes=(1, 0), out_axes=1)(
        grouped(pool, 1, groups), grouped(table, 0, groups))
    return jnp.moveaxis(rows.reshape(pool.shape[0], table.shape[0],
                                     table.shape[1] * pool.shape[2], *pool.shape[3:]), 0, 2)


def _gathered_attention(query, key, value, table, lengths, starts=None, sinks=None, *, softcap=None):
    """The paged kernel's arithmetic over the gathered rows, through the attention dispatch: one query a
    row over the first `lengths` keys of its pages, from its `starts` where a window gives them, its
    scaled logits bounded by `softcap` where given, with the layer's `sinks` `[heads]` where it has them.
    An inactive row reads one slot. Its backward is the paged kernel's (`_paged_backward`)."""
    from dew.nn.attention import scaled_dot_product_attention

    keys, values = _gather_pages(key, table, 1), _gather_pages(value, table, 1)
    return scaled_dot_product_attention(query[:, None], keys, values, softcap=softcap, sinks=sinks,
                                        key_value_seq_lengths=jnp.maximum(lengths, 1),
                                        key_value_seq_starts=starts)[:, 0]


def _gathered_tangent(*operands, softcap=None):
    """`_gathered_attention` traced for forward mode (`forward_mode_attention`): a fused kernel the
    dispatch takes keeps its value and takes its tangent from the reference path's, so `jax.jvp`
    differentiates it at every order, which a reverse-only fused kernel would not."""
    from dew.nn.attention import forward_mode_attention

    with forward_mode_attention():
        return _gathered_attention(*operands, softcap=softcap)


def _paged_forward(query, key, value, table, lengths, starts=None, sinks=None, *, softcap=None):
    return decode_attention.attend_paged(query, key, value, table, jnp.maximum(lengths, 1), starts,
                                         softcap=softcap, sinks=sinks), ()


def _paged_backward(saved, operands, cotangent, *, softcap=None):
    return jax.vjp(functools.partial(_gathered_attention, softcap=softcap), *operands)[1](cotangent)


@functools.cache
def gpu_paged(softcap: float | None):
    """Dew's Pallas decode kernel over the pages (`decode_attention.attend_paged`) under the layer's
    `softcap`, over `(query, key, value, table, lengths, starts, sinks)`, the last two None where the
    step has none; its derivatives at every order are the gathered attention's
    (`generation.linear_tangent`), the sinks' among them: reverse mode runs the backward of whichever
    kernel the dispatch takes for it (`_gathered_attention`), a call cuDNN's width and determinism limits
    turn down included, and forward mode takes the forward-mode path's tangent (`_gathered_tangent`),
    with no context of the caller's. An inactive row reads one slot, so its unused output stays
    finite."""
    return generation.linear_tangent(functools.partial(_paged_forward, softcap=softcap),
                                     functools.partial(_paged_backward, softcap=softcap),
                                     functools.partial(_gathered_tangent, softcap=softcap))


def _pages_per_block(pages: int) -> int:
    """How many of a row's `pages` the TPU paged kernel attends in one block:
    the largest divisor of `pages` up to 8, since the kernel needs the block
    to divide a row's pages. The gcd with 8 gave a row of three pages, a
    capacity of 384 at 128 a page, one-page blocks."""
    return max(block for block in range(1, 9) if pages % block == 0)


@dataclasses.dataclass(frozen=True)
class Append:
    """Stores a call's keys and values, then reads the cache; `open_kv_cache` returns one.

    The allocation-only call (when the cache did not exist yet) stores
    nothing. The keys it reads back are in the store's rotation, so the query
    that attends to them must go through `query` first.
    """

    store: KVStore
    positions: jax.Array
    allocated: bool

    def __call__(self, key: jax.Array, value: jax.Array) -> tuple[jax.Array, jax.Array]:
        if self.allocated:
            self.store.write(key, value, self.positions)
        return self.store.read()

    def query(self, query: jax.Array) -> jax.Array:
        """Return `query` `[..., head_dim]` in the basis that the stored keys are in."""
        rotation = self.store.rotation
        return query if rotation is None else rotated(query, rotation)

    def places(self, places: jax.Array) -> jax.Array:
        """Store the call's token places beside its keys and return every slot's, `[rows, capacity]`.

        `places` are `[rows, tokens]` or one vector the rows share. A chunk
        and ALiBi read a cached key at the place the call that wrote it
        counted its token at (`dew.nn.inputs.counted_places`). A caller may
        place a row's tokens anywhere, so a slot is no place of its own, and a
        paged pool keeps the places in its pages, which a shared prefix's
        pages then carry.
        """
        if self.allocated:
            self.store.write_places(jnp.broadcast_to(places, self.positions.shape), self.positions)
        return self.store.read_places()

    def groups(self, name: str, groups: jax.Array | None) -> jax.Array:
        """Store the call's tokens' groups of field `name` `[rows, tokens]` beside their keys, -1 where it
        carries none, and return every slot's, `[rows, capacity]`: a later call's token reads its block's
        earlier keys by them, and a layer sharing these keys reads them too."""
        if self.allocated:
            written = jnp.full(self.positions.shape, -1, jnp.int32) if groups is None else groups
            self.store.write_groups(name, written, self.positions)
        return self.store.read_groups(name)


def gather_cache_rows(cache, rows):
    """A decode cache reindexed on its batch axis, one gather per leaf.

    Every leaf a decode step writes carries its batch on axis zero outside the
    stack (`StackView` removes a scanned layer axis before the cache crosses
    `apply`): keys and values with validity and cursor, a delta net's states,
    latent attention's cache, image groups, a multimodal model's next position.
    `rows` is any index array: repeats duplicate a row's state, a permutation
    reparents rows, a different length changes the row count; beam branching
    and speculative rollback are both this. A paged cache is refused, since
    gathered rows would write into each other's pages.
    """
    if is_paged(cache):
        raise ValueError("beam search and speculative decoding regroup cache rows, which a "
                         "paged cache's shared pool cannot do; decode them with a dense cache")
    return jax.tree.map(lambda leaf: jnp.take(leaf, rows, axis=0), cache)


def cache_positions(module: nn.Module, batch: int, length: int, valid):
    """Allocate compact cache slots for real tokens, independently per row.

    A row's tokens fill its slots in order, so the cursor alone says which
    hold one (`cached_validity`)."""
    valid = jnp.ones((batch, length), bool) if valid is None else jnp.asarray(valid, bool)
    if valid.shape != (batch, length):
        raise ValueError(f"cache validity must be {(batch, length)}, got {valid.shape}")
    allocated = module.has_variable("cache", "cache_index")
    index = module.variable("cache", "cache_index", jnp.zeros, (batch,), jnp.int32)
    positions = index.value[:, None] + jnp.cumsum(valid, axis=1, dtype=jnp.int32) - 1
    positions = jnp.where(valid, positions, -1)
    if allocated:
        index.value = index.value + jnp.sum(valid, axis=1, dtype=jnp.int32)
    return positions, allocated


def cached_validity(module: nn.Module, length: int) -> jax.Array:
    """Which of `module`'s `length` cache slots hold a token, `[rows, length]`:
    those before the row's cursor, which `cache_positions` fills in order.

    Derived where it is read rather than stored beside the cursor, so a
    decode step writes no `[rows, capacity]` copy of it a layer: 28 kernels
    a step on Qwen3-0.6B, 0.6% of a 32-slot serving step on an RTX 4080
    (docs/performance.md)."""
    return filled_slots(module.get_variable("cache", "cache_index"), length)


_DEFAULT_LAYOUT = KVCache()


def open_kv_cache(module: nn.Module, key, max_seq_len, *, valid=None, layout: KVCache = _DEFAULT_LAYOUT,
                  value_dim: int | None = None, places: bool = False, groups: tuple[str, ...] = ()):
    """Fixed-size K/V with a cursor and cached validity for each batch row.

    Returns [B, S] compact slot positions and a writer. Invalid tokens have
    position -1 and do not advance the cursor. The allocation-only call leaves
    every row empty. The writer returns full cache arrays, including unused
    slots which the caller excludes with `cached_validity`. `layout` chooses the
    storage behind the slots, dense or paged, full or quantized
    (`dew.nn.kv_cache`); the slots and the cursor are the same for all. Values
    are `value_dim` wide, None for the keys' width. `places` also keeps each
    token's place beside its key (`dew.nn.kv_cache.Append.places`), and
    `groups` its groups of the fields named (`Append.groups`).
    """
    shards = sequence_shards()
    if shards > 1:
        raise LayoutRefused(
            f"decoding is not supported under a sequence axis of {shards}: the KV "
            "cache holds whole sequences. Generate on a mesh with sequence=1.")
    if max_seq_len is None:
        raise ValueError("decoding needs max_seq_len for its fixed-capacity cache")
    batch, length, heads, head_dim = key.shape
    if valid is None and length > max_seq_len:
        raise ValueError(f"{length} tokens do not fit a KV cache of {max_seq_len}.")
    store = KVStore.open(module, layout, batch, max_seq_len, heads, head_dim, key.dtype, value_dim=value_dim,
                         places=places, groups=groups)
    positions, allocated = cache_positions(module, batch, length, valid)
    return positions, Append(store, positions, allocated)


__all__ = ["Append", "KVCache", "KVStore"]
