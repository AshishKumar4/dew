"""The algebra engine's custom derivatives each hold their contract (docs/design/testing.md, derivative
contracts): every order finite wherever its float64 truth is a normal float32 and its plain float32 formula is
finite."""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from contract_support import EDGES, ORDER_SETS, assert_contract, normal_draw

from dew.algebra import _generated, format_cases, means, scan, windows


class _Recorded:
    """A generated module whose forms record each call by their body's name (`taken`): a test holds a composed
    surrogate to have run its generated bodies, not the written ones its declaration falls back to."""

    def __init__(self, module, tables: dict | None = None):
        self.taken: list[str] = []
        self._module, self._tables = module, tables

    def __getattr__(self, attribute):
        held = self._tables[attribute] if self._tables is not None else getattr(self._module, attribute)
        name = attribute.removeprefix("_").removesuffix("_forms")

        def recorded(form):
            def run(*operands):
                self.taken.append(name)
                return form(*operands)
            return run
        return _generated.Forms(held.single, {key: recorded(form) for key, form in held.cases.items()})


def _recorded(module):
    emitted = _Recorded(module)
    return emitted, emitted.taken


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_generated_rounded_bodies_keep_the_written_surrogate_law(dtype):
    from dew.algebra import format_cases_derived
    from dew.nn import precision

    emitted, taken = _recorded(format_cases_derived)
    composed = format_cases.ROUNDED.assembled(emitted)

    def rounded(value):
        return composed(value, dtype=dtype)

    gradient = jax.grad(lambda x: jnp.sum(rounded(x) ** 2))
    name = "rounded" if dtype == jnp.bfloat16 else "rounded_fp16"

    for shape, generated in (((16, 32), True), ((3, 5), False)):  # generated geometry, the written fallback
        taken.clear()
        x = np.full(shape, 1.003, np.float32)
        dx = np.full_like(x, 1.007)
        primal, direction = jax.jit(lambda x, dx: jax.jvp(rounded, (x,), (dx,)))(x, dx)
        np.testing.assert_array_equal(primal, precision.rounded_operand(x, dtype))
        np.testing.assert_array_equal(direction, dx)
        _, pullback = jax.vjp(rounded, jnp.asarray(x))
        np.testing.assert_array_equal(pullback(jnp.asarray(dx))[0], dx)
        _, higher = jax.jit(lambda x, dx: jax.jvp(gradient, (x,), (dx,)))(x, dx)
        np.testing.assert_array_equal(higher, 2 * dx)
        assert ({name, f"{name}_tangent"} <= set(taken)) if generated else not taken, taken
    taken.clear()
    x, dx, cotangent = normal_draw(6110, (16, 32), (16, 32), (16, 32))
    assert_contract(rounded, (x,), (dx,), cotangent, f"the generated {name} surrogate",
                    plain_fn=lambda value: precision.rounded_operand(value, dtype))
    assert name in taken


def test_generated_quantized_straight_through_holds_signed_zero_and_higher_orders():
    from dew.algebra import format_cases_derived
    from dew.nn import fake_quant

    emitted, taken = _recorded(format_cases_derived)
    composed = format_cases.QUANTIZED_STE.assembled(emitted)
    x = jnp.full((16, 32), 7.0, jnp.float32)
    stored = jnp.full(x.shape, -0.0, jnp.bfloat16)
    tangent = jnp.full_like(x, 3)
    rounded_tangent = jnp.ones_like(stored)
    primal, moved = jax.jit(lambda a, b, da, db: jax.jvp(composed, (a, b), (da, db)))(
        x, stored, tangent, rounded_tangent)
    np.testing.assert_array_equal(np.asarray(primal).view(np.uint32),
                                  np.asarray(fake_quant.straight_through(x, stored)).view(np.uint32))
    np.testing.assert_array_equal(moved, tangent)
    gradient = jax.grad(lambda arg: jnp.square(composed(arg, stored)).sum())
    _, higher = jax.jvp(gradient, (x,), (tangent,))
    np.testing.assert_array_equal(higher, 2 * tangent)
    assert {"quantized_ste", "quantized_ste_tangent"} <= set(taken), taken


def test_generated_guidance_preserves_the_scale_direction_at_both_endpoints():
    from dew.sampling import guidance_cases, guidance_cases_derived

    emitted, taken = _recorded(guidance_cases_derived)
    composed = guidance_cases.GUIDED.assembled(emitted)
    c = jnp.full((16, 64), 3.0, jnp.float32)
    u = jnp.full_like(c, 1)
    weight = jnp.broadcast_to(jnp.arange(16, dtype=jnp.float32)[:, None] % 2, (16, 1))

    def moved_at(scale):
        return jax.jvp(lambda w: composed(c, u, w), (scale,), (jnp.ones_like(scale),))

    primal, moved = jax.jit(moved_at)(weight)
    np.testing.assert_array_equal(primal, jnp.where(weight == 0, u, c))
    np.testing.assert_array_equal(moved, c - u)
    def higher(fn):
        gradient = jax.grad(lambda arg, w: jnp.sum(fn(arg, u, w)), argnums=0)
        return jax.jvp(lambda w: gradient(c, w), (weight,), (jnp.ones_like(weight),))[1]

    np.testing.assert_array_equal(higher(composed), jnp.ones_like(c))
    assert {"guided", "guided_tangent"} <= set(taken), taken


def _plain(outputs, totals, anchors):
    """The convex mean as written: each state's mass at the joined anchor, its outputs weighed by it."""
    masses = totals * jnp.exp(anchors - jnp.max(anchors, axis=0, keepdims=True))
    return jnp.sum(masses[..., None] * outputs, axis=0) / jnp.sum(masses, axis=0)[..., None]


def _merged(outputs, totals, anchors):
    """The merge of finished states the contract holds, one function at every edge, so the cell's programs
    compiled at one edge serve the others (`contract_support._mesh_program`)."""
    return means.merged_output(outputs, totals, anchors, windows.Units(), 0)


@pytest.mark.parametrize("taken", ORDER_SETS)
def test_the_convex_mean_holds_every_order(taken):
    """The merge of finished states (`means.merged_output`, its derivative the law's own over scaled values,
    `dew.algebra.scaled`'s `applied` and `_split` within it): every order, forward and reverse, under a nested
    jit and a scan, in the outputs, the totals and the anchors, finite wherever its own float64 run is a
    normal float32 and the plain mean is, its outputs and totals at each edge (`EDGES`): the law runs in
    float64 too, its construct of its operands' format. The order set `taken` (`ORDER_SETS`), at every edge,
    each order's programs compiled once a cell (`contract_support._mesh_program`): a cell an order set, where
    sets sharing a cell took 246 to 264 s (row 11720)."""
    outputs, totals, anchors = normal_draw(6096, (3, 4, 8), (3, 4), (3, 4))
    along = normal_draw(6097, (3, 4, 8), (3, 4), (3, 4))
    (cotangent,) = normal_draw(6098, (4, 8))
    for edge, scale in EDGES.items():
        primals = (outputs * scale, (jnp.abs(totals) + 0.25) * scale, anchors)
        direction = (along[0] * scale, along[1] * scale, along[2])
        assert_contract(_merged, primals, direction, cotangent, f"the convex mean {edge}", plain_fn=_plain,
                        taking=ORDER_SETS[taken])


def test_a_decayed_pair_holds_its_derivatives():
    keys = jnp.asarray([[2.0, 0.0], [0.0, 0.0]], jnp.float32)
    decay = jnp.zeros((2, 2, 2), jnp.float32).at[1, 0, 0].set(0.5)

    def paired(x):
        rows = jnp.zeros_like(keys).at[1, 0].set(x)
        return scan._paired(rows, keys, decay)[1, 0]

    assert jax.jit(jax.grad(jax.grad(lambda x: jnp.square(paired(x)))))(jnp.float32(3)) == 2

    # The user input is finite; the read it produces overflows, but d(read)/dx and the held factors do not.
    value, gradient = jax.jit(jax.value_and_grad(lambda x: paired(2 * x)))(jnp.float32(2.0 ** 127))
    assert jnp.isposinf(value)
    assert gradient == 2


def test_an_nvfp4_tensor_scale_keeps_its_ordinary_maximum_derivative():
    import numpy as np

    from dew.algebra import bits

    x = np.zeros((16, 1), np.float32)
    x[0, 0] = 2688
    direction = np.zeros_like(x)
    direction[0, 0] = 1

    def scale(tile):
        return bits.nvfp4_coded(tile)[2]

    value, tangent = jax.jit(lambda x, dx: jax.jvp(scale, (x,), (dx,)))(x, direction)
    np.testing.assert_array_equal(value, np.float32(1))
    np.testing.assert_array_equal(tangent, np.float32(1 / 2688))
    np.testing.assert_array_equal(jax.jit(jax.grad(scale))(x), direction * np.float32(1 / 2688))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_a_normal_draw_is_read_where_a_gpu_parts_prepare_places_its_calls(dtype):
    """`contract_support.normal_draw` made where a GPU tier part's prepare places its calls (a compile-only
    GPU, on which nothing computes or is read) is read there, each value bit for bit what JAX's conversion
    of the drawn value to `dtype` is, at float32 and at float64 under x64 (row 10552: a draw placed on the
    prepare's GPU could not be read, and its cell went unprepared)."""
    import jax
    import numpy as np
    from contract_support import normal_draw

    from dew.nn.kernels.generation import compile_only_gpus

    def converted():
        rng = np.random.default_rng(10552)
        return [np.asarray(jnp.asarray(rng.standard_normal(shape) * 0.3, dtype)) for shape in ((3, 5), (7,))]

    (gpu,) = compile_only_gpus("sm90", 1, 80 * 2**30)
    for x64 in (False, True):
        with jax.enable_x64(x64):
            expected = converted()
            with jax.default_device(gpu):
                drawn = [np.asarray(each)
                         for each in normal_draw(10552, (3, 5), (7,), scale=0.3, dtype=dtype)]
            for mine, theirs in zip(drawn, expected, strict=True):
                bits = np.uint16 if mine.itemsize == 2 else np.uint32
                np.testing.assert_array_equal(mine.view(bits), theirs.view(bits))


def test_a_surrogate_case_keys_its_static_arguments_exactly():
    """A generated surrogate case is keyed by its static arguments exactly (`generate._exact_key`): `-0.0`,
    equal to `0.0` as a Python value, is no `0.0` case; a set, unhashable, is no `frozenset` case; a tuple
    subclass equal only to itself is no case of another with its contents; and a list holding itself keys
    none: each runs the written primal and tangent, which differ from the declared bodies'."""
    from dew.algebra.generate import Cases, ComponentForm, SurrogateBodies, SurrogateCase

    def written(x, shift):
        return x + jnp.copysign(1.0, shift if isinstance(shift, float) else -1.0)

    def written_tangent(shift, primals, tangents, value):
        del primals, value
        return tangents[0] * jnp.copysign(2.0, shift if isinstance(shift, float) else -1.0)

    def cases(count):
        operand = jax.ShapeDtypeStruct((4,), jnp.float32)
        return Cases(((operand,) * count,), lambda seed, operands: operands, seeds=1, policies=("default",))

    def declared_case(name):
        return SurrogateCase(name, ComponentForm(lambda x: x + 1, lambda x: x + 1, cases(1)),
                             {(): ComponentForm(lambda x, dx, value: dx, lambda x, dx, value: dx, cases(3))},
                             output_tree=(jax.tree.structure(0),))

    class Held(list):
        """A list no pytree registry knows, which reaches the surrogate whole, holding itself."""

    class Named(tuple):
        """A tuple equal only to itself."""

        __eq__ = object.__eq__
        __hash__ = object.__hash__

    cyclic = Held()
    cyclic.append(cyclic)
    named = Named((1,))
    surrogate = SurrogateBodies(written, written_tangent, (1,), symbolic_zeros=False, cases=(
        ((0.0,), declared_case("plus")), ((frozenset({1}),), declared_case("member")),
        ((named,), declared_case("named"))))

    key = ("default", "sm90", _generated.case_of((jnp.zeros((4,), jnp.float32),)))
    tangent_key = ("default", "sm90", _generated.case_of((jnp.zeros((4,), jnp.float32),) * 3))
    emitted = _Recorded(None, {
        _generated.table_name(name): _generated.Forms(True, {key: lambda x: (x + 1,)})
        for name in ("plus", "member", "named")} | {
        _generated.table_name(f"{name}_tangent"):
            _generated.Forms(True, {tangent_key: lambda x, dx, value: (dx,)})
        for name in ("plus", "member", "named")})
    composed = surrogate.assembled(emitted)
    x = jnp.zeros((4,), jnp.float32)
    for shift, value, slope in ((0.0, 1.0, 1.0), (frozenset({1}), 1.0, 1.0), (named, 1.0, 1.0),
                                (-0.0, -1.0, -2.0), ({1}, -1.0, -2.0), (Named((1,)), -1.0, -2.0),
                                (cyclic, -1.0, -2.0)):
        primal, tangent = jax.jvp(lambda arg, shift=shift: composed(arg, shift), (x,), (jnp.ones_like(x),))
        np.testing.assert_array_equal(primal, np.full(4, value, np.float32))
        np.testing.assert_array_equal(tangent, np.full(4, slope, np.float32))


def test_a_surrogate_static_key_tells_apart_exactly_what_differs():
    """`generate._exact_key` keys two static arguments alike only where they are one: each sign of zero, each
    NaN's sign and payload, each number type (`1`, `1.0`, `True`), NumPy and bfloat16 scalars by type and
    bits, a long double's NaNs by their payloads, nested plain tuples by their parts; an unhashable value has
    none, and a value of a class whose own type is unhashable is keyed by its own hash, its type by identity.
    Two cases of one key, or of one body's name, are refused as they are declared."""
    from dew.algebra.generate import Cases, ComponentForm, SurrogateBodies, SurrogateCase, _exact_key

    def unhashable(*_):
        raise TypeError("an unhashable type")

    class Unhashable(type):
        __hash__ = unhashable

    class Odd(metaclass=Unhashable):
        pass

    nan = np.float64("nan")
    payload = np.frombuffer(np.uint64(0x7FF8000000000001).tobytes(), np.float64)[0]
    bfloat16 = np.dtype(jnp.bfloat16).type
    # Two long-double NaNs of one sign, their payloads apart (x87's significand's low bit, or a quad's).
    quiet = np.longdouble("nan")
    raw = bytearray(np.asarray(quiet).tobytes())
    raw[0] ^= 1
    other = np.frombuffer(bytes(raw), np.longdouble)[0]
    odd = Odd()
    distinct = [0.0, -0.0, nan, -nan, payload, 1, 1.0, True, np.float32(1), np.float16(1), np.longdouble(1),
                np.longdouble(-0.0), np.longdouble(0.0), quiet, other, np.complex64(1), bfloat16(1),
                bfloat16(-0.0), (1, (2.0,)), (1, (-2.0,)), frozenset({1}), odd, Odd()]
    keys = [_exact_key(value) for value in distinct]
    assert None not in keys and len(set(keys)) == len(keys)  # each hashed, as a dictionary's lookup hashes it
    assert _exact_key(float("nan")) == _exact_key(float("nan")) and _exact_key((0.0,)) == _exact_key((0.0,))
    assert _exact_key(np.longdouble(1) / 3) == _exact_key(np.longdouble(1) / 3)
    assert _exact_key([1]) is None and _exact_key(({1},)) is None
    assert {_exact_key(odd): "declared"}.get(_exact_key(odd)) == "declared"

    operand = jax.ShapeDtypeStruct((4,), jnp.float32)
    form = ComponentForm(lambda x: x, lambda x: x, Cases(((operand,),), lambda seed, operands: operands, 1))
    case = SurrogateCase("twin", form, {}, output_tree=(jax.tree.structure(0),))
    with pytest.raises(ValueError):
        SurrogateBodies(lambda x, shift: x, lambda *values: values[-2][0], (1,), symbolic_zeros=False,
                        cases=(((nan,), case), ((np.float64("nan"),), case)))
    twin = SurrogateCase("twin", form, {}, output_tree=(jax.tree.structure(0),))
    with pytest.raises(ValueError):
        SurrogateBodies(lambda x, shift: x, lambda *values: values[-2][0], (1,), symbolic_zeros=False,
                        cases=(((2.0,), case), ((3.0,), twin)))


def test_a_generated_module_whose_tables_are_not_its_declarations_is_refused():
    """A surrogate assembled over a generated module whose table lacks a declared case, or holds one no
    longer declared, is refused (a stale module), never run by one lookup as if current."""
    from dew.algebra import format_cases_derived

    for change in ("missing", "extra"):
        tables = {}
        for _, case in format_cases.ROUNDED.cases:
            for name in (case.name, *map(case.tangent_name, case.tangents)):
                held = _generated.table(format_cases_derived, name)
                tables[_generated.table_name(name)] = _generated.Forms(held.single, dict(held.cases))
        rows = tables[_generated.table_name("rounded")].cases
        if change == "missing":
            rows.pop(next(iter(rows)))
        else:
            rows["default", "sm90", ((((3, 5), "float32"),))] = next(iter(rows.values()))
        with pytest.raises(ValueError):
            format_cases.ROUNDED.assembled(types.SimpleNamespace(**tables))
