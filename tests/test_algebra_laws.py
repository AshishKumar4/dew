"""The laws of every form a shipped declaration's derivation emits at a GPU's kernel tiles, certified
(`dew.algebra.laws`, docs/design/algebra.md §6.9). The mainloops' tile search spaces for an A100, an H100 and
an RTX PRO 6000 (sm80, sm90, sm120) lie in `ROWS`, `KEYS` and `WIDTHS`. Over them each declaration's forms
take a few distinct structures (`laws.skeletons`: extraction is priced, so a tile's extents can choose
another form), and each is certified at a small tile of the same structure (`witness`), proven over the
reals. A witness keeps every axis past one where it can and the kernel tile's equal extents equal, so a form
that confuses two axes of one extent shows it in its values (the keys-first forward's broadcast did, at
square tiles alone)."""

import functools
import itertools

import jax.numpy as jnp
import pytest

from dew.algebra import SATURATION, derive, laws
from dew.nn.kernels import recurrence_algebra
from dew.nn.kernels.templates import (
    attention as template,
    attention_algebra,
    attention_mosaic_program,
    attention_triton,
    bundles,
)

ROWS, KEYS, WIDTHS = (32, 64, 128), (32, 64, 128, 192, 256), (16, 32, 64, 128, 256)
"""A kernel tile's extents: its queries, its keys (the walked axis) and its head widths, each mainloop's
(`test_every_mainloop_s_tiles_lie_in_the_kernel_extents`)."""

GPUS = ("sm80", "sm90", "sm120")
DTYPES = (jnp.bfloat16, jnp.float16, jnp.float32)


def test_every_mainloop_s_tiles_lie_in_the_kernel_extents():
    """No derivation emits a tile outside `ROWS`, `KEYS` and `WIDTHS`: the Pallas-Triton mainloop's steps,
    and on every family of each GPU (`bundles.layouts`) the Mosaic mainloop's rows (its unit, `_derived`)
    and walked extents (`attention_mosaic_program.extents`), at every head width's staging (a power of two,
    `template.power_of_two`, in the Mosaic mainloop's whole swizzled rows)."""
    for tiles in (*attention_triton.FORWARD_STEPS, *attention_triton.BACKWARD_STEPS):
        assert tiles.block_q in ROWS and tiles.block_k in KEYS, tiles
    for name, dtype in itertools.product(GPUS, DTYPES):
        for family in bundles.layouts(name, dtype):
            extents = attention_mosaic_program.extents(family, dtype, max(KEYS))
            assert extents[-1] in ROWS and set(extents) <= set(KEYS), (name, family.name, extents)
            widths = {attention_mosaic_program._width(width, family, dtype) for width in range(1, 257)}
            assert widths <= set(WIDTHS), (name, family.name, widths)
    assert {template.power_of_two(width) for width in range(1, 257)} <= set(WIDTHS)


def _forward(base2: bool, keys_first: bool, dtype):
    declared = attention_algebra.declaration(base2=base2, keys_first=keys_first)

    def shapes(rows, keys, width):
        tile = (keys, rows) if keys_first else (rows, keys)
        return {"logits": (tile, jnp.float32), "values": ((keys, width), dtype)}

    return declared, shapes


def _backward(walked: str, hooked: bool, dropped: bool, dtype, width: int):
    declared = attention_algebra.backward_declaration(scale=width ** -0.5, walked=walked, hooked=hooked,
                                                      dropped=dropped)

    def shapes(rows, keys, columns):
        tile = ((rows, keys), jnp.float32)
        given = {"logits": tile, "q": ((rows, columns), dtype), "k": ((keys, columns), dtype),
                 "values": ((keys, columns), dtype), "dout": ((rows, columns), dtype), "seen": ((rows, keys),
                                                                                            jnp.bool_)}
        given |= dict.fromkeys(("peak", "total", "anchor", "delta"), ((rows,), jnp.float32))
        return given | ({"slope": tile} if hooked else {}) | ({"keep": tile} if dropped else {})

    return declared, shapes


def _cotangents(keys_first: bool, dropped: bool, masked: bool):
    tile = ("keys", "rows") if keys_first else ("rows", "keys")
    axes = {"logits": tile, "products": tile, "peak": ("rows",), "total": ("rows",), "anchor": ("rows",),
            "delta": ("rows",), **({"keep": tile} if dropped else {}), **({"seen": tile} if masked else {})}
    declared = functools.partial(attention_algebra.softmax_cotangents, keys_first=keys_first)
    declaration = derive.Reduction(declared, axes, walked="keys", exponent="e")

    def shapes(rows, keys, width):
        del width
        extents = {"rows": rows, "keys": keys}
        return {name: (tuple(extents[axis] for axis in named), jnp.bool_ if name == "seen" else jnp.float32)
                for name, named in axes.items()}

    return declaration, shapes


def _recurrence(name: str):
    axes = {"products": ("rows", "columns"), "decays": ("rows", "columns"),
            "scales": ("rows",) if name == "pairs" else ("columns",)}
    declaration = derive.Reduction(recurrence_algebra.TILES[name], axes, walked="columns", exponent="e")

    def shapes(rows, columns, width):
        del width
        extents = {"rows": rows, "columns": columns}
        return {name: (tuple(extents[axis] for axis in named), jnp.float32) for name, named in axes.items()}

    return declaration, shapes


def _pattern(tile: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Which of a tile's axes have equal extents."""
    return tuple(tuple(index for index, other in enumerate(tile) if other == extent) for extent in tile)


SMALL = tuple(sorted(itertools.product((2, 3, 4, 5), (1, 2), (2, 3, 4, 5)),
                     key=lambda tile: (sum(tile), tile)))
"""A witness's candidate tiles, (rows, keys, width): the walked axis at most 2 (a law's maxima compare every
key of two tiles), every other axis past one, smallest first."""

SMALLEST = tuple(sorted(itertools.product((1, 2, 3, 4, 5), (1, 2), (1, 2, 3, 4, 5)),
                        key=lambda tile: (sum(tile), tile)))
"""Where no tile of `SMALL` takes a structure: any small tile."""


def witness(declaration, shapes, kernel_tile: tuple[int, int, int], found: tuple) -> tuple[int, int, int]:
    """A small tile whose forms' structure is `found` (the kernel tile's): first one whose axes' equal
    extents are the kernel tile's (the walked axis apart), else any of `SMALL`, else of `SMALLEST`."""
    rows, _, width = kernel_tile
    alike = [tile for tile in SMALL if (tile[0] == tile[2]) == (rows == width)]
    for tile in (*alike, *SMALL, *SMALLEST):
        if laws.skeletons(declaration, shapes(*tile)) == found:
            return tile
    raise AssertionError(f"no small tile takes the structure the kernel tile {kernel_tile} takes")


def _certify_every_structure(declaration, shapes, tiles) -> None:
    """Each distinct structure `declaration`'s forms take over the kernel `tiles` certified at its
    witness."""
    structures: dict = {}
    for tile in tiles:
        structures.setdefault(laws.skeletons(declaration, shapes(*tile)), tile)
    for found, tile in structures.items():
        laws.certified(declaration, shapes(*witness(declaration, shapes, tile, found)))


KERNEL_TILES = tuple(itertools.product(ROWS, KEYS, WIDTHS))


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: jnp.dtype(dtype).name)
@pytest.mark.parametrize(("base2", "keys_first"), list(itertools.product((False, True), repeat=2)),
                         ids=lambda value: str(value))
def test_every_structure_of_the_forward_is_certified(base2, keys_first, dtype):
    """The forward's softmax walk (`attention_algebra.declaration`), in either base, queries or keys first."""
    _certify_every_structure(*_forward(base2, keys_first, dtype), KERNEL_TILES)


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: jnp.dtype(dtype).name)
@pytest.mark.parametrize(("walked", "hooked", "dropped"),
                         list(itertools.product(("keys", "rows"), (False, True), (False, True))),
                         ids=lambda value: str(value))
def test_every_structure_of_the_backward_is_certified(walked, hooked, dropped, dtype, width):
    """The backward (`attention_algebra.backward_declaration`) at the scale of each head width, walked over
    keys or queries, hooked and dropping or not."""
    declaration, shapes = _backward(walked, hooked, dropped, dtype, width)
    tiles = tuple((rows, keys, width) for rows, keys in itertools.product(ROWS, KEYS))
    _certify_every_structure(declaration, shapes, tiles)


@pytest.mark.parametrize(("keys_first", "dropped", "masked"),
                         list(itertools.product((False, True), repeat=3)), ids=lambda value: str(value))
def test_every_structure_of_the_backward_tile_is_certified(keys_first, dropped, masked):
    """The backward tile's softmax algebra (`attention_algebra.softmax_cotangents`)."""
    declaration, shapes = _cotangents(keys_first, dropped, masked)
    _certify_every_structure(declaration, shapes, tuple((rows, keys, 0) for rows, keys in
                                                        itertools.product(ROWS, KEYS)))


@pytest.mark.parametrize("name", list(recurrence_algebra.TILES))
def test_every_structure_of_the_recurrence_s_tiles_is_certified(name):
    """The recurrence's tile declarations (`recurrence_algebra.TILES`), at its chunks' extents."""
    declaration, shapes = _recurrence(name)
    chunks = (16, 32, 64)
    _certify_every_structure(declaration, shapes, tuple((rows, columns, 0) for rows, columns in
                                                        itertools.product(chunks, chunks)))


def test_a_form_that_breaks_a_law_is_refused():
    """The certifier refuses forms that break a law: a forward whose merge keeps its first state whatever
    the second holds breaks the identity (a merge of the empty state and a state is the state)."""
    declaration, shapes = _forward(base2=False, keys_first=False, dtype=jnp.float32)
    forms = derive.derive(declaration, accurate=False, saturation=SATURATION)

    def kept(a, b, operands=None, *, shapes):
        del b, operands, shapes
        return a

    with pytest.raises(laws.Unlawful, match="identity"):
        laws.certify(init=forms.init, step=forms.step, merge=kept, finish=forms.finish,
                     declared=declaration.function, shapes=shapes(3, 2, 4), walked={"logits": 1, "values": 0})
