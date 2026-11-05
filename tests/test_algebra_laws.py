"""The laws of every form a shipped declaration's derivation emits at a GPU's kernel tiles, certified
(`algebra_laws`, docs/design/algebra.md §6.9). The mainloops' tile search spaces for an A100, an H100 and
an RTX PRO 6000 (sm80, sm90, sm120) lie in `ROWS`, `KEYS` and `WIDTHS`. Over them each declaration's forms
take a few distinct structures (`laws.skeletons`: extraction is priced, so a tile's extents can choose
another form), and each is certified at a small tile of the same structure (`witness`), proven over the
reals. A witness keeps every axis past one and its extents equal and unequal as the kernel tile's are, so a
form that confuses two axes of one extent shows it in its values (the keys-first forward's broadcast did, at
square tiles alone). The proofs are reduced-tile proofs, shared by the kernel tiles of their structure
(`algebra_laws`)."""

import functools
import itertools

import algebra_laws as laws
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from map_judge import sharded

from dew.algebra import SATURATION, derive
from dew.nn.kernels import recurrence_algebra
from dew.nn.kernels.templates import (
    attention as template,
    attention_algebra,
    attention_backward_algebra,
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


def _forward(base2: bool, keys_first: bool, dtype, factor: float | None = None):
    declared = attention_algebra.declaration(base2=base2, keys_first=keys_first, factor=factor)

    def shapes(rows, keys, width):
        tile = (keys, rows) if keys_first else (rows, keys)
        return {"logits": (tile, jnp.float32), "values": ((keys, width), dtype)}

    return declared, shapes


FACTORS = (None, 1.0, 0.125, float(np.float32(128 ** -0.5 * np.log2(np.e))), -0.125)
"""The forward's logits as the mainloop gives them (`Keys.products`, `logits_of`): its formed logits (None,
under a hook), or its raw products and their factor (`Forward.factor`): one where the accurate form keeps them
raw, a power of two's scale, a scale times log2(e) folded (an fp32's), and a negative scale, whose order no
maximum keeps."""


def _backward(walked: str, hooked: bool, dropped: bool, dtype, width: int):
    declared = attention_backward_algebra.backward_declaration(scale=width ** -0.5, walked=walked,
                                                               hooked=hooked, dropped=dropped)

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
    declared = functools.partial(attention_backward_algebra.softmax_cotangents, keys_first=keys_first)
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


SMALL = tuple(sorted(itertools.product((2, 3, 4, 5), (2,), (2, 3, 4, 5)), key=lambda tile: (sum(tile), tile)))
"""A witness's candidate tiles, (rows, keys, width): the walked axis 2 (a law's maxima compare every key of
two tiles), every axis past one (an axis of one broadcasts as a kept axis does), smallest first."""


def witness(declaration, shapes, kernel_tile: tuple[int, int, int], found: tuple) -> tuple[int, int, int]:
    """The smallest tile whose extents are equal and unequal as the kernel tile's are (an unused axis, 0,
    apart) and whose forms' structure is `found`, the kernel tile's."""
    for rows, keys, width in SMALL:
        tile = (rows, keys, width if kernel_tile[2] else 0)
        if _pattern(tile) == _pattern(kernel_tile) and laws.skeletons(declaration, shapes(*tile)) == found:
            return tile
    raise AssertionError(f"no small tile takes the structure and the equal extents of {kernel_tile}")


def _certify_every_structure(declaration, shapes, tiles) -> None:
    """Each distinct structure `declaration`'s forms take over the kernel `tiles`, with each pattern of equal
    extents it takes it at, certified at its witness."""
    structures: dict = {}
    for tile in tiles:
        structures.setdefault((laws.skeletons(declaration, shapes(*tile)), _pattern(tile)), tile)
    for (found, _), tile in structures.items():
        laws.certified(declaration, shapes(*witness(declaration, shapes, tile, found)))


KERNEL_TILES = tuple(itertools.product(ROWS, KEYS, WIDTHS))


FORWARD_CASES = tuple(itertools.product((False, True), (False, True), DTYPES, FACTORS))
"""The forward's cases (`test_every_structure_of_the_forward_is_certified`): its base, its order, its
operands' dtype and its logits' factor."""

BACKWARD_CASES = tuple(itertools.product(("keys", "rows"), (False, True), (False, True), DTYPES, WIDTHS))
"""The backward's cases (`test_every_structure_of_the_backward_is_certified`): its walked axis, whether it
is hooked and dropping, its operands' dtype and its head width."""


def _certified_in_shards(label: str, cases: tuple, units: int, certify) -> None:
    """Each of `cases` certified (`certify`), the cases the units of armada's map tasks (the test's `shards`
    mark, `units` of them): each map task certifies its cases and records how many (`map_judge`), failing on
    the first that is not lawful, and the judge holds that every case was; run whole, every case in turn. A
    case certifies its structures at their witnesses whatever task it runs in."""
    assert len(cases) == units, "the shards mark counts each case"

    def certified(start: int, stop: int) -> int:
        for case in cases[start:stop]:
            certify(*case)
        return stop - start

    sharded(label, certified, units)


@pytest.mark.shards(units=60, per_task=1)
def test_every_structure_of_the_forward_is_certified():
    """The forward's softmax walk (`attention_algebra.declaration`), in either base, queries or keys first,
    over its formed logits or its raw products and each regime of their factor, at every dtype: each case
    40 s to 55 s, one a map task: two took 100 s to 127 s a task on armada (b376a92ee3)."""
    def certify(base2, keys_first, dtype, factor):
        _certify_every_structure(*_forward(base2, keys_first, dtype, factor), KERNEL_TILES)

    _certified_in_shards("the forward's structures", FORWARD_CASES, 60, certify)


@pytest.mark.shards(units=120, per_task=4)
def test_every_structure_of_the_backward_is_certified():
    """The backward (`attention_backward_algebra.backward_declaration`) at the scale of each head width,
    walked over keys or queries, hooked and dropping or not, at every dtype: each case 14 s or so, 25 at the
    most, four a map task (armada, 3bfbdfc842)."""
    def certify(walked, hooked, dropped, dtype, width):
        declaration, shapes = _backward(walked, hooked, dropped, dtype, width)
        tiles = tuple((rows, keys, width) for rows, keys in itertools.product(ROWS, KEYS))
        _certify_every_structure(declaration, shapes, tiles)

    _certified_in_shards("the backward's structures", BACKWARD_CASES, 120, certify)


@pytest.mark.parametrize(("keys_first", "dropped", "masked"),
                         list(itertools.product((False, True), repeat=3)), ids=lambda value: str(value))
def test_every_structure_of_the_backward_tile_is_certified(keys_first, dropped, masked):
    """The backward tile's softmax algebra (`attention_backward_algebra.softmax_cotangents`)."""
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

    with pytest.raises(laws.Unlawful) as raised:
        laws.certify(init=forms.init, step=forms.step, merge=kept, finish=forms.finish,
                     declared=declaration.function, shapes=shapes(3, 2, 4), walked={"logits": 1, "values": 0})
    assert raised.value.law == "identity"


def _biased(start: int):
    """Attention plus a bias wider than the values by one, sliced from `start`: a form over an operand the
    walk does not read, its window's place a parameter."""
    def attention(logits, values, bias):
        p = jnp.exp(logits - jnp.max(logits, axis=-1)[:, None])
        return (p @ values) / jnp.sum(p, axis=-1)[:, None] + bias[:, start:start + values.shape[1]]

    declaration = derive.Reduction(attention, {"logits": ("rows", "keys"), "values": ("keys", "width"),
                                               "bias": ("rows", "wide")}, walked="keys", exponent="e")

    def shapes(rows, keys, width):
        return {"logits": ((rows, keys), jnp.float32), "values": ((keys, width), jnp.float32),
                "bias": ((rows, width + 1), jnp.float32)}

    return declaration, shapes


def test_forms_sliced_at_two_places_share_no_certificate():
    """Two declarations that slice an operand one place apart have forms of two structures at a kernel's
    tile, so neither's certificate is the other's: a window's place is no shape's (`laws.SHAPES`)."""
    (first, shapes), (second, _) = _biased(0), _biased(1)
    assert laws.skeletons(first, shapes(64, 64, 128)) != laws.skeletons(second, shapes(64, 64, 128))


def test_forms_sliced_off_by_one_are_refused():
    """The forms of a declaration that slices one place on, certified against the declaration that slices
    where they should: the declared function's law breaks."""
    (declared, shapes), (shifted, _) = _biased(0), _biased(1)
    forms = derive.derive(shifted, accurate=False, saturation=SATURATION)
    with pytest.raises(laws.Unlawful) as raised:
        laws.certify(init=forms.init, step=forms.step, merge=forms.merge, finish=forms.finish,
                     declared=declared.function, shapes=shapes(3, 2, 4), walked={"logits": 1, "values": 0},
                     mapped=forms.mapped)
    assert raised.value.law == "declared"


def test_forms_whose_output_is_laid_out_otherwise_are_refused():
    """The forward's forms with their output reshaped from rows by width to width by rows, its elements the
    same in the same order: the declared function's law breaks on the output's shape."""
    declaration, shapes = _forward(base2=False, keys_first=False, dtype=jnp.float32)
    forms = derive.derive(declaration, accurate=False, saturation=SATURATION)

    def reshaped(*args, **kwargs):
        output = forms.finish(*args, **kwargs)
        return jnp.reshape(output, output.shape[::-1])

    with pytest.raises(laws.Unlawful) as raised:
        laws.certify(init=forms.init, step=forms.step, merge=forms.merge, finish=reshaped,
                     declared=declaration.function, shapes=shapes(3, 2, 4), walked={"logits": 1, "values": 0},
                     mapped=forms.mapped)

    assert raised.value.law == "declared"


@jax.custom_jvp
def _scaled(bias, scale):
    return bias * scale


@_scaled.defjvp
def _scaled_jvp(primals, tangents):
    (bias, scale), (dbias, dscale) = primals, tangents
    return _scaled(bias, scale), dbias * scale + bias * dscale


def test_forms_whose_opaque_literal_differs_share_no_structure():
    """A declaration adding a bias through an equation outside the language (a custom JVP's) whose literal
    scale changes with the tile's width (1 at a small one, 2 at a kernel's) has two structures: an opaque
    equation's literal is its own slot's, so the small tile's certificate is not the kernel tile's."""
    def function(logits, values, bias):
        p = jnp.exp(logits - jnp.max(logits, axis=-1)[:, None])
        scale = jnp.float32(2 if values.shape[-1] >= 8 else 1)
        return (p @ values) / jnp.sum(p, axis=-1)[:, None] + _scaled(bias, scale)

    declaration = derive.Reduction(function, {"logits": ("rows", "keys"), "values": ("keys", "width"),
                                              "bias": ()}, walked="keys", exponent="e")

    def shapes(rows, keys, width):
        return {"logits": ((rows, keys), jnp.float32), "values": ((keys, width), jnp.float32),
                "bias": ((), jnp.float32)}

    assert laws.skeletons(declaration, shapes(3, 2, 4)) != laws.skeletons(declaration, shapes(64, 32, 128))
