"""A declaration's tiles of one class of sizes (`dew.algebra.derive._classed`, docs/design/algebra.md §6.8):
one derivation a class, its canonical sizes', bound at each size of it whichever a process derives first, and
the tiles a class cannot take, derived at their own sizes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact
from test_algebra_derive import AXES, WIDTH, attention, declared, drawn, shaped, walked, whole

from dew.algebra import SATURATION, derive
from dew.nn import precision


def walked_in(forms: derive.Derived, rows: int, tile: int, tiles: int):
    """`forms` stepped over `tiles` key tiles of `tile` keys each, `rows` queries, finished: a function of
    (q, k, v)."""
    def run(q, k, v):
        values = {"q": q, "k": k, "v": v, "rows": jnp.arange(rows), "keys": jnp.arange(tile * tiles)}
        cut = [{name: value[at * tile:(at + 1) * tile] if AXES[name][0] == "keys" else value
                for name, value in values.items()} for at in range(tiles)]
        return forms.finish(walked(forms, cut), values, shapes=shaped(cut[0]))

    return run


CLASS = ((12, 24), (13, 20))
"""Two tiles of one class of sizes (`derive._shape_of`): 12 and 13 queries, of one power of two (16), and 24
and 20 keys beside the 32 a query is wide, of another (32), the keys the smaller: canonical sizes 16 queries
and 31 keys (`derive._canonical`)."""


def primed(order) -> tuple[derive.Derived, list]:
    """`attention`'s walk derived at `CLASS`'s tiles in `order`: its forms, and the reports each tile made."""
    from dew.nn.kernels.templates import report

    forms = derive.derive(declared(attention), accurate=False, saturation=SATURATION)
    with report.refusals():
        for rows, keys in order:
            forms.init(shaped({"q": jnp.zeros((rows, WIDTH)), "k": jnp.zeros((keys, WIDTH)),
                               "v": jnp.zeros((keys, WIDTH)), "rows": jnp.zeros((rows,), jnp.int32),
                               "keys": jnp.zeros((keys,), jnp.int32)}))
        return forms, report.rewrites()


def test_a_classs_sizes_take_its_canonical_forms_whichever_the_process_derives_first():
    """`attention`'s walk at two tiles of one class (`CLASS`), each declaration primed in an order of its
    own: one derivation for both in either (its class's canonical sizes'), each tile's forms bound from it
    (its report reused), each tile's walk the same bit for bit whichever tile was derived first, as no
    history of the process reaches its forms, and on par with the whole-operand function against the
    float64 truth (check 1)."""
    first, made = primed(CLASS)
    second, again = primed(CLASS[::-1])
    assert [each.reused for each in made] == [each.reused for each in again] == [False, True, True], made
    for rows, keys in CLASS:
        run = walked_in(first, rows, keys, 3)
        q, k, v = drawn(rows, 3 * keys)(0)
        np.testing.assert_array_equal(np.asarray(run(q, k, v)).view(np.int32),
                                      np.asarray(walked_in(second, rows, keys, 3)(q, k, v)).view(np.int32))
        assert_as_exact(jax.jit(run), jax.jit(whole(attention)), drawn(rows, 3 * keys),
                        f"a classed tile of {rows} by {keys}", seeds=8)


def rowed(x, y):
    """A sum along the walk beside an unwalked operand's sum in three rows of it (a reshape of its own
    size)."""
    return jnp.sum(x) + jnp.sum(y.reshape(3, -1))


def flattened(x, y):
    """A sum along the walk beside an unwalked matrix's sum, flat (a reshape of its own sizes)."""
    return jnp.sum(x) + jnp.sum(y.reshape(-1))


def thirds(x, y):
    """A sum along the walk beside an unwalked operand's sum, which the declaration asserts comes in
    thirds."""
    assert y.shape[-1] % 3 == 0, y.shape
    return jnp.sum(x) + jnp.sum(y)


def integral(shape, seed: int = 0) -> jax.Array:
    """Small integers as float32 at `shape`: every partial sum of them exact, so a walk is its function bit
    for bit whatever its order."""
    return jnp.asarray(np.random.default_rng(seed).integers(-8, 9, shape), jnp.float32)


@pytest.mark.parametrize(("function", "unwalked"), [(rowed, (12,)), (flattened, (11, 12)), (thirds, (12,))],
                         ids=["its canonical sizes refuse its reshape", "its reshape flattens past them",
                              "it asserts its own size"])
def test_a_tile_its_class_cannot_take_derives_at_its_own_sizes(function, unwalked):
    """A declaration at a tile its class's canonical sizes do not take: twelve unwalked values in three rows,
    whose canonical sixteen no three rows divide (the trace refusing); a matrix of eleven by twelve
    flattened to 132, where its canonical fifteen by sixteen flatten to 240, not its canonical 256 (a lookup
    of the class at those sizes would ask for them again); or twelve values the declaration asserts come in
    thirds, which sixteen do not: derived at its own sizes, and walked as its function."""
    forms = derive.derive(derive.Reduction(function, {"x": ("items",), "y": ("other",) * len(unwalked)},
                                           walked="items"), accurate=False, saturation=SATURATION)
    x, y = integral((8,)), integral(unwalked, 1)
    halves = [{"x": x[:4], "y": y}, {"x": x[4:], "y": y}]
    state = forms.init(shaped(halves[0]))
    for half in halves:
        state = forms.step(state, half)
    np.testing.assert_array_equal(forms.finish(state, {"x": x, "y": y}, shapes=shaped(halves[0])),
                                  function(x, y))


def test_a_tile_closing_over_an_outer_traces_value_derives_at_its_own_sizes():
    """A declaration made within a trace, closing over its value (a bias), walked within it at a tile of no
    canonical sizes of its own (twelve values): the value no host holds keys no class, and the walk is its
    function's."""
    x = integral((24,))

    @jax.jit
    def run(x, bias):
        declaration = derive.Reduction(lambda x: jnp.sum(x) + bias, {"x": ("items",)}, walked="items")
        forms = derive.derive(declaration, accurate=False, saturation=SATURATION)
        tiles = [{"x": x[:12]}, {"x": x[12:]}]
        state = forms.init(shaped(tiles[0]))
        for tile in tiles:
            state = forms.step(state, tile)
        return forms.finish(state, {"x": x}, shapes=shaped(tiles[0]))

    np.testing.assert_array_equal(run(x, jnp.float32(0.5)), jnp.sum(x) + 0.5)


def by_width(x):
    """The maximum of its values where 64-bit values are on, their sum where they are off: a declaration that
    reads the mode it traces under."""
    return jnp.max(x) if jax.config.jax_enable_x64 else jnp.sum(x)


def walk_of(forms: derive.Derived, x: jax.Array, tile: int) -> jax.Array:
    """`forms` stepped over `x` in tiles of `tile` values, finished."""
    tiles = [{"x": x[at:at + tile]} for at in range(0, x.shape[0], tile)]
    state = forms.init(shaped(tiles[0]))
    for each in tiles:
        state = forms.step(state, each)
    return forms.finish(state, {"x": x}, shapes=shaped(tiles[0]))


def test_a_class_asked_under_another_mode_takes_that_modes_forms():
    """`by_width` walked at sixteen values a tile with 64-bit values off (its canonical sizes, a sum), then at
    twelve with them on (a maximum): the second's class is its own mode's, traced and derived under it, and
    its walk is the maximum, where a class looked up by its canonical sizes alone took the sum's forms."""
    forms = derive.derive(derive.Reduction(by_width, {"x": ("items",)}, walked="items"), accurate=False,
                          saturation=SATURATION)
    with jax.enable_x64(new_val=False):
        np.testing.assert_array_equal(walk_of(forms, integral((32,)), 16), jnp.sum(integral((32,))))
    with jax.enable_x64(new_val=True):
        x = integral((24,), 1)
        np.testing.assert_array_equal(walk_of(forms, x, 12), jnp.max(x))


def by_cpu_width(x):
    """The sum of its values, or at sixteen their maximum where the CPU's wide arithmetic is asked
    (`precision.wide_cpu_arithmetic`): a declaration whose class's canonical trace reads a fact its other
    sizes' do not."""
    return jnp.max(x) if x.shape[0] == 16 and precision.wide_cpu_arithmetic() else jnp.sum(x)


@pytest.mark.parametrize(("first", "reused"), [(12, [[False, True], [False]]), (16, [[False], [False]])],
                         ids=["a sibling first", "the canonical sizes first"])
def test_a_class_holds_the_facts_its_canonical_trace_read(monkeypatch, first, reused):
    """`by_cpu_width`, its class's canonical sizes sixteen: walked with the CPU's wide arithmetic off at
    `first` values a tile (a sibling, the class derived at sixteen and twelve bound from it; or sixteen
    itself, its own trace seeding the class), then with it on at twelve: the canonical trace now a maximum of
    another class, twelve derives at its own sizes (`reused`), where a class held without the facts its
    canonical trace read bound the record that trace no longer makes."""
    from dew.nn.kernels.templates import report

    forms = derive.derive(derive.Reduction(by_cpu_width, {"x": ("items",)}, walked="items"), accurate=False,
                          saturation=SATURATION)
    made = []
    for wide, size in (("0", first), ("1", 12)):
        monkeypatch.setenv(precision.WIDE_CPU_ARITHMETIC, wide)
        with report.refusals():
            forms.init(shaped({"x": jnp.zeros((size,))}))
            made.append([each.reused for each in report.rewrites()])
    assert made == reused, made


def by_rank_promotion(x):
    """The sum of its values, or at sixteen their maximum where JAX raises on rank promotion: a declaration
    whose class's canonical trace reads a mode of JAX's its other sizes' do not."""
    return jnp.max(x) if x.shape[0] == 16 and jax.config.jax_numpy_rank_promotion == "raise" else jnp.sum(x)


def test_a_class_is_held_under_the_jax_modes_it_was_traced_in():
    """`by_rank_promotion` at twelve values a tile with rank promotion allowed (the class derived at sixteen,
    a sum, and twelve bound from it), then where JAX raises on it at twelve again and at thirteen, of the same
    class: the canonical trace now a maximum of another class, each derives at its own sizes, where forms or a
    class held without JAX's promotion modes (`derive.context`) bound the sum's record."""
    from dew.nn.kernels.templates import report

    forms = derive.derive(derive.Reduction(by_rank_promotion, {"x": ("items",)}, walked="items"),
                          accurate=False, saturation=SATURATION)
    made = []
    for promotion, size in (("allow", 12), ("raise", 12), ("raise", 13)):
        with jax.numpy_rank_promotion(promotion), report.refusals():
            forms.init(shaped({"x": jnp.zeros((size,))}))
            made.append([each.reused for each in report.rewrites()])
    assert made == [[False, True], [False], [False]], made


def summed_beside_maxima(a, b, c):
    """A sum along the walk of a matrix of rows by columns beside the maxima of a rows' and a columns'
    operand: two anchors of two extents a rescaled sum lifts along each axis (`derive._lift`)."""
    return jnp.sum(a, axis=-1), jnp.max(b, axis=-1), jnp.max(c, axis=-1)


def test_a_classs_anchor_lifts_are_appended_in_one_order_at_every_size():
    """`summed_beside_maxima` at 14 rows and 15 columns, of the class whose canonical sizes are 15 and 16:
    the lifts its anchors' two extents take are appended in the region's order at each size, so the tables
    the class binds at its sizes are a derivation's there (the oracle's hold, `derive._checked`, which a set's
    order of their extents parted), and its walk is its function's."""
    axes = {"a": ("rows", "cols", "keys"), "b": ("rows", "keys"), "c": ("cols", "keys")}
    forms = derive.derive(derive.Reduction(summed_beside_maxima, axes, walked="keys"), accurate=False,
                          saturation=SATURATION)
    values = {"a": integral((14, 15, 8)), "b": integral((14, 8), 1), "c": integral((15, 8), 2)}
    halves = [{name: value[..., :4] for name, value in values.items()},
              {name: value[..., 4:] for name, value in values.items()}]
    state = forms.init(shaped(halves[0]))
    for half in halves:
        state = forms.step(state, half)
    for walked_out, whole_out in zip(forms.finish(state, values, shapes=shaped(halves[0])),
                                     summed_beside_maxima(**values), strict=True):
        np.testing.assert_array_equal(walked_out, whole_out)


@pytest.mark.parametrize(("first", "second"),
                         [((12, 24), (12, 40)), ((12, 24), (20, 24)), ((1, 24), (2, 24))],
                         ids=["keys past a power of two", "queries into the keys' power", "a unit axis"])
def test_tiles_across_a_power_of_two_or_a_unit_axis_derive_apart(first, second):
    """Two tiles whose sizes part at a power of two (a fact of each extent's a derivation reads,
    `rules.log2_of`, and the thresholds its guards test: a counting sum's, an exact int8 product's), share
    one (the queries the keys' and the width's), or part at a unit axis (a reshape's): two classes, each
    derived at its own canonical sizes, two derivations."""
    from dew.nn.kernels.templates import report

    forms = derive.derive(declared(attention), accurate=False, saturation=SATURATION)
    with report.refusals():
        for rows, keys in (first, second):
            forms.init(shaped({"q": jnp.zeros((rows, WIDTH)), "k": jnp.zeros((keys, WIDTH)),
                               "v": jnp.zeros((keys, WIDTH)), "rows": jnp.zeros((rows,), jnp.int32),
                               "keys": jnp.zeros((keys,), jnp.int32)}))
        made = report.rewrites()
    assert sum(not each.reused for each in made) == 2, made
