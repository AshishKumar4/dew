"""A band of keys about each query (kernel-templates.md row 58): its geometry, the columns a query's band
holds and the key each one's column 0 is, and its tiles as a launch reads them, a `Kind.BAND` side
operand's (`attention.Kind`) and an input-conditioned relative bias's (`dew.nn.attention.BandBias`)."""

import jax
import jax.numpy as jnp

BAND_GROUP = 768
"""The queries one band origin serves (`band_origins`): a multiple of every program's query rows a launch
of the template holds (64 to 256: one to three 64-row consumers, two 128-row ones), so a tile of a band
reads one rectangle whatever launch reads it."""

BAND_ALIGN = 256
"""What a band's origins are whole multiples of (`band_origins`): every key tile the template's launches
walk (16 to 256 keys), so a band's column of a key tile is a whole tile of it."""


def band_width(extent: int) -> int:
    """The columns a band of `extent` holds a query (`band_origins`): every key within `extent` of each query
    of a `BAND_GROUP`, from the group's origin, in whole `BAND_ALIGN`s."""
    return -(-(BAND_GROUP + extent + BAND_ALIGN - 2) // BAND_ALIGN) * BAND_ALIGN


def band_origins(offset: jax.Array, queries: int, extent: int) -> jax.Array:
    """The key each query's band column 0 holds, `[B or 1, Q]` int32: of query `q` in group `g` (`q //
    BAND_GROUP`), the last whole `BAND_ALIGN` at or before `offset + g BAND_GROUP - (extent - 1)`, where
    `offset` (`[]` or `[B]`) is a query's place less its index, the key it is at (0 for a prefill from the
    first key, a chunk's first place for a chunked one). Column `c` of query `q`'s band is key `origin + c`;
    every key from `extent - 1` before the query's place through its place is within its `band_width`."""
    groups = (jnp.arange(queries, dtype=jnp.int32) // BAND_GROUP) * BAND_GROUP
    start = jnp.reshape(jnp.asarray(offset, jnp.int32), (-1, 1)) + groups[None, :] - (extent - 1)
    return jnp.floor_divide(start, BAND_ALIGN) * BAND_ALIGN


def band_dense(values: jax.Array, offset: jax.Array, extent: int, keys: int) -> jax.Array:
    """A band's `values` `[B or 1, H or 1, Q, band_width(extent)]` (`band_origins`) written out over `keys`,
    `[B or 1, H or 1, Q, K]`: each key in a query's band its column's value, every other key's zero."""
    rows, heads, queries, width = values.shape
    origins = band_origins(offset, queries, extent)
    column = jnp.arange(keys, dtype=jnp.int32)[None, None, :] - origins[:, :, None]
    inside = (column >= 0) & (column < width)
    # The rows the values and the origins broadcast to: a row of either broadcasts, none of either is none.
    batch = jnp.broadcast_shapes((rows,), (origins.shape[0],))[0]
    index = jnp.broadcast_to(jnp.clip(column, 0, width - 1)[:, None], (batch, heads, queries, keys))
    read = jnp.take_along_axis(jnp.broadcast_to(values, (batch, heads, queries, width)), index, axis=-1)
    return jnp.where(inside[:, None], read, jnp.zeros((), values.dtype))


def band_column(origin: jax.Array, k_first, keys: int, width: int) -> tuple[jax.Array, jax.Array]:
    """A key tile of `keys` keys from `k_first`'s column in a band of `width` columns whose column 0 is key
    `origin` (`band_origins`), clamped within the band, and whether the tile lies within it: origins and
    widths whole `BAND_ALIGN`s, and every key tile's keys dividing one, a tile lies wholly within the band or
    wholly past it."""
    column = jnp.asarray(k_first, jnp.int32) - origin
    inside = (column >= 0) & (column <= width - keys)
    return jnp.clip(column, 0, width - keys), inside


def band_tile(rows: jax.Array, origin: jax.Array, k_first, keys: int) -> jax.Array:
    """A band's tile `[bq, keys]` of a query tile's band rows `rows` `[bq, Kb]` (its queries' origin
    `origin`, one a `BAND_GROUP`) at the key tile of `keys` keys from `k_first`: its columns there, zeros
    where the tile lies past the band (`band_column`)."""
    column, inside = band_column(origin, k_first, keys, rows.shape[-1])
    tile = jax.lax.dynamic_slice_in_dim(rows, column, keys, axis=-1)
    return jnp.where(inside, tile, jnp.zeros((), tile.dtype))
