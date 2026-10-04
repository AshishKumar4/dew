"""How a decode cache stores its keys and values: dense or paged, full or quantized.

`dew.nn.attention.open_kv_cache` owns the cursor and the compact slot
positions; a `KVCache` value names the storage layout behind them and
`KVStore` reads and writes it inside one attention module.

Dense storage is one `[rows, capacity, kv_heads, head_dim]` block per row.
Paged storage is one pool `[kv_heads, pages, page_size, head_dim]` shared by
every row, the layout the Pallas TPU paged attention kernel reads, with a
`[rows, capacity // page_size]` page table mapping slot `s` to page
`table[row, s // page_size]`, offset `s % page_size`. A server can then hand
pages out on demand, share a common prefix's pages, and hold more rows than
`pages * page_size / capacity` while they are short. A pool split into
`groups` gives each equal group of rows its own part, indexed from the
part's start, so a server whose mesh splits rows `n` ways reads and writes
only the pages each device holds instead of gathering the whole pool.

A quantized cache stores int8 or float8 (e4m3) values with one float32
scale per token and head (absmax over the head's features over the
format's largest value), dequantized on read. int8 keys are stored rotated
by the orthonormal Hadamard matrix over head_dim, as QuaRot (arXiv:2404.00456)
rotates its cache, so a few outlier channels (Qwen3's keys) do not take the
whole per-token scale; the query takes the same rotation (`Append.query`),
leaving every logit unchanged since `H Hᵀ = I`. Values and float8 keys are
not rotated (e4m3 rounds each element relative to itself). On Qwen3-0.6B
over wikitext-2 against bf16: int8 +0.66 perplexity unrotated, +0.008
rotated; float8 -0.03 unrotated, +1.03 rotated (docs/concepts/inference.md).
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

    `quantized` stores keys and values in that format with per-token, per-head
    scales; int8 keys are Hadamard-rotated, which needs a power-of-two head_dim.
    `page_size` pages the cache (`capacity` a multiple of it) and `pages` sizes
    the pool, None giving one page per slot of every row; a smaller pool is a
    server's to hand out (`dew.inference.serving.Server`). `groups` splits a
    paged pool by equal groups of rows; the row count and `pages` divide by it.
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
    decoding rows' tokens and its prompts' in this one scatter, as words."""
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
    if jax.default_backend() != 'gpu' or x.dtype == jnp.bool_ or per_word < 2 or x.shape[-1] % per_word:
        return x
    pairs = x.reshape(*x.shape[:-1], x.shape[-1] // per_word, per_word)
    return jax.lax.bitcast_convert_type(pairs, jnp.uint32)


def from_words(words: jax.Array, dtype: jnp.dtype) -> jax.Array:
    """The `dtype` elements `as_words` packed into `words`."""
    if words.dtype == jnp.dtype(dtype):
        return words
    return jax.lax.bitcast_convert_type(words, dtype).reshape(*words.shape[:-1], -1)


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
        return _gather_pages(pool, self._get(TABLE), self.layout.groups)

    def kernel(self) -> bool:
        """Whether decode can read a full-precision BF16 page pool natively.

        The TPU kernel reads bfloat16 and casts other page dtypes (its int8 path
        broadcasts the scales to full width first), so a float32 or quantized pool
        takes the gather; a GPU uses cuDNN's paged forward on supported heads and
        whole 16-token pages. A grouped pool takes the gather, which keeps each
        group's pages where they are.
        """
        page_size = self.layout.page_size
        if (page_size is None or self.layout.quantized is not None
                or self.layout.groups != 1 or self.dtype != jnp.bfloat16):
            return False
        if jax.default_backend() == "tpu":
            return True
        if jax.default_backend() != "gpu":
            return False
        from dew.nn.attention import _FORWARD_MODE, cudnn_runs

        query = jax.ShapeDtypeStruct((self.rows, 1, self.kv_heads, self.head_dim), self.dtype)
        return (page_size % 16 == 0 and cudnn_runs(query)
                and not _FORWARD_MODE.get())

    def decode(self, query: jax.Array, lengths: jax.Array, softcap: float | None) -> jax.Array:
        """One query per row `[rows, heads, head_dim]` against the first
        `lengths` slots of each row, through its device's paged kernel.

        The TPU kernel does not scale the logits, so the query carries
        1/sqrt(head_dim) as every other attention path applies it.
        """
        if jax.default_backend() == "gpu":
            if softcap is not None:
                raise ValueError("the cuDNN paged kernel does not apply a logit softcap")
            # JAX 0.11.2's paged cuDNN backward returns [rows, page, heads, dim]
            # for a [pool_pages, page, heads, dim] operand; the verifier rejects
            # it. Keep the old gathered attention as the wrapper's VJP.
            return _gpu_paged(query, self._get("cached_key"), self._get("cached_value"),
                              self._get(TABLE), lengths)
        from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention

        scaled = query * jnp.asarray(1.0 / math.sqrt(self.head_dim), query.dtype)
        table = self._get(TABLE)
        return paged_attention(
            scaled,
            self._get("cached_key"),
            self._get("cached_value"),
            lengths,
            table,
            attn_logits_soft_cap=softcap,
            pages_per_compute_block=_pages_per_block(table.shape[1]),
        )


def _gather_pages(pool: jax.Array, table: jax.Array, groups: int) -> jax.Array:
    """The existing grouped page read as `[rows, capacity, heads, ...]`."""
    rows = jax.vmap(lambda part, index: part[:, index], in_axes=(1, 0), out_axes=1)(
        grouped(pool, 1, groups), grouped(table, 0, groups))
    return jnp.moveaxis(rows.reshape(pool.shape[0], table.shape[0],
                                     table.shape[1] * pool.shape[2], *pool.shape[3:]), 0, 2)


@jax.custom_vjp
def _gpu_paged(query: jax.Array, key: jax.Array, value: jax.Array,
               table: jax.Array, lengths: jax.Array) -> jax.Array:
    # Private JAX API, pinned by jax<0.11.3. Moving it breaks
    # test_native_gpu_paged_value_and_vjp_keep_the_gathered_attention.
    from jax._src.cudnn.fused_attention_stablehlo import MaskType, paged_attention

    rows = query.shape[0]
    # The old gather pads q=1 to q=2 for cuDNN's odd-query backward restriction
    # and passes query lengths of 2. Keep that forward tiling/rounding; discard
    # the added query's output, rather than changing the accepted greedy tokens.
    padded = jnp.pad(query[:, None], ((0, 0), (0, 1), (0, 0), (0, 0)))
    pages = table[:, None, :, None]
    # Inactive rows read one slot to keep the unused output finite; nobody reads it.
    out = paged_attention(
        padded, jnp.moveaxis(key, 0, 2), jnp.moveaxis(value, 0, 2),
        jnp.full(rows, 2, jnp.int32), jnp.maximum(lengths, 1), pages, pages,
        scale=query.shape[-1] ** -0.5, mask_type=MaskType.PADDING)
    if not isinstance(out, jax.Array):
        raise TypeError("cuDNN paged attention must return an array")
    return out[:, 0]


def _gpu_paged_fwd(query, key, value, table, lengths):
    return _gpu_paged(query, key, value, table, lengths), (query, key, value, table, lengths)


def _gpu_paged_bwd(saved, cotangent):
    from dew.nn.attention import cudnn_attention

    query, key, value, table, lengths = saved

    def gathered(q, k, v):
        return cudnn_attention(q[:, None], _gather_pages(k, table, 1), _gather_pages(v, table, 1),
                                bias=None, mask=None, causal=False, sliding_window=None,
                                key_value_seq_lengths=jnp.maximum(lengths, 1))[:, 0]

    _, backward = jax.vjp(gathered, query, key, value)
    return (*backward(cotangent), None, None)


_gpu_paged.defvjp(_gpu_paged_fwd, _gpu_paged_bwd)


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


__all__ = ["Append", "KVCache", "KVStore", "Layered"]
