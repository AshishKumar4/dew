"""A program's constants as the host computes them (`dew.nn.programs.host_evaluated`): integer and boolean
scalars folded in the host's own integers, no program compiled to fold one."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax._src.literals import TypedNdArray

from dew.nn import programs


def _convert(dtype) -> dict:
    return {"new_dtype": np.dtype(dtype), "weak_type": False, "sharding": None}


def _weak(value: np.generic) -> TypedNdArray:
    """`value` as a weakly typed literal of its own dtype, as a program holds one."""
    return TypedNdArray(np.asarray(value), jax.core.ShapedArray((), np.result_type(value), weak_type=True))


FOLDS = [
    (lax.div_p, (np.int32(-7), np.int32(2)), {}), (lax.div_p, (np.int32(7), np.int32(-2)), {}),
    (lax.div_p, (np.int32(-7), np.int32(-2)), {}), (lax.rem_p, (np.int32(-7), np.int32(2)), {}),
    (lax.rem_p, (np.int32(7), np.int32(-2)), {}), (lax.add_p, (np.int32(2 ** 31 - 1), np.int32(1)), {}),
    (lax.sub_p, (np.uint32(0), np.uint32(1)), {}), (lax.neg_p, (np.int32(-2 ** 31),), {}),
    (lax.mul_p, (np.int8(16), np.int8(16)), {"out_dtype": np.dtype(np.int32)}),
    (lax.mul_p, (np.int32(2), np.int32(1)), {"out_dtype": np.dtype(np.bool_)}),
    (lax.and_p, (np.int32(6), np.int32(3)), {}), (lax.xor_p, (np.int32(6), np.int32(3)), {}),
    (lax.not_p, (np.int32(5),), {}), (lax.not_p, (np.bool_(1),), {}),
    (lax.or_p, (np.bool_(0), np.bool_(1)), {}), (lax.lt_p, (np.int32(-1), np.int32(0)), {}),
    (lax.max_p, (np.int32(-3), np.int32(2)), {}), (lax.clamp_p, (np.int32(0), np.int32(9), np.int32(4)), {}),
    (lax.shift_left_p, (np.int32(3), np.int32(30)), {}), (lax.shift_left_p, (np.int32(3), np.int32(32)), {}),
    (lax.shift_right_arithmetic_p, (np.int32(-8), np.int32(1)), {}),
    (lax.shift_right_arithmetic_p, (np.int32(-8), np.int32(40)), {}),
    (lax.shift_right_arithmetic_p, (np.uint32(2 ** 31), np.uint32(1)), {}),
    (lax.shift_right_arithmetic_p, (np.uint32(2 ** 31), np.uint32(40)), {}),
    (lax.shift_right_logical_p, (np.int32(-1), np.int32(28)), {}),
    (lax.shift_right_logical_p, (np.int32(-1), np.int32(32)), {}),
    (lax.convert_element_type_p, (np.int32(10),), _convert(jnp.int4)),
    (lax.convert_element_type_p, (np.int32(200),), _convert(np.int8)),
    (lax.convert_element_type_p, (np.int32(-2),), _convert(np.bool_)),
    (lax.convert_element_type_p, (np.bool_(1),), _convert(np.int32)),
    (lax.mul_p, (np.asarray(3, jnp.int4)[()], np.int32(5)), {"out_dtype": np.dtype(np.int32)}),
    (lax.eq_p, (np.int64(2 ** 32), np.int64(0)), {}),
    (lax.shift_right_logical_p, (_weak(np.int8(-1)), _weak(np.int8(4))), {}),
    (lax.add_p, (5, 6), {}),
    (lax.max_p, (np.int32(-3), np.int32(1)), {"out_dtype": np.dtype(np.int32)}),
    (lax.div_p, (np.int32(7), np.int32(0)), {}), (lax.rem_p, (np.int32(7), np.int32(0)), {}),
    (lax.div_p, (np.uint32(7), np.uint32(0)), {}), (lax.rem_p, (np.uint8(7), np.uint8(0)), {}),
    (lax.div_p, (np.int32(-2 ** 31), np.int32(-1)), {}), (lax.rem_p, (np.int32(-2 ** 31), np.int32(-1)), {}),
    (lax.add_p, (np.ulonglong(1), np.ulonglong(2)), {}),
    (lax.max_p, (np.int16(255), np.int16(1)), {"out_dtype": np.dtype(np.int8)}),
    (lax.min_p, (np.int32(0), np.int32(-1)), {"out_dtype": np.dtype(np.uint32)}),
]
"""Folds with a quotient and a remainder by either sign, a width's wrap (int32's, uint32's, int8's, int4's, a
product at the output dtype it names, a boolean's), bitwise and boolean logic, a comparison, an extremum and a
clamp, shifts within and past the width (an unsigned value's arithmetic one reading it as signed),
conversions among integer and boolean dtypes, and operands as JAX binds them: a product of an int4 and an
int32 at its stated output dtype, a NumPy int64 past int32 narrowed as JAX narrows it, a weak int8's shift at
its own width, two Python integers, weakly typed, and an extremum at a stated output dtype its operands
already hold, a quotient and a remainder by zero and of the least integer by -1, as XLA defines them, NumPy
scalars of a class JAX canonicalizes at bind (`np.ulonglong`), and extrema at a stated output dtype of
another width or signedness, each operand cast to it first as XLA casts it."""


@pytest.mark.parametrize("primitive, operands, params", FOLDS, ids=lambda value: getattr(value, "name", None))
def test_an_integer_scalar_constant_folds_as_its_primitive_computes_it_and_compiles_nothing(
        primitive, operands, params):
    """`host_evaluated` of an integer or boolean scalar operation (`FOLDS`) is the primitive bound at its
    operands, bit for bit and of its type (its dtype and weak type, which the next equation binds), and
    traces no program to fold (`executable_only`)."""
    from dew.telemetry.compiles import executable_only

    bound = np.asarray(primitive.bind(*operands, **params))
    # JAX's in-memory caches cleared, so a fold that binds the primitive traces its program again.
    jax.clear_caches()
    with executable_only():
        made = programs.host_evaluated(primitive, operands, params)
    assert made is not None and len(made) == 1
    (folded,) = made
    assert np.asarray(folded).dtype == bound.dtype and np.asarray(folded) == bound, (folded, bound)
    assert jax.typeof(folded) == jax.typeof(primitive.bind(*operands, **params))


@pytest.mark.parametrize("primitive, operands, params", [
    (lax.add_p, (np.float32(1.5), np.float32(2.0)), {}),
    (lax.mul_p, (np.int32(65536), np.int32(1)), {"out_dtype": np.dtype(np.float16)})])
def test_a_constant_the_fold_leaves_is_the_hosts_bind(primitive, operands, params):
    """A floating operation, whose IEEE semantics the host's CPU computes as it does, and an integer product
    stated in a float (its operands never cast before the fold declines it, under NumPy's raising overflow)
    are not folded (`scalar_folded` gives none) and `host_evaluated` gives the bound primitive's value."""
    stated = params.get("out_dtype", np.result_type(*operands))
    assert programs.scalar_folded(primitive, operands, params, stated) is None
    with np.errstate(over="raise"):
        (made,) = programs.host_evaluated(primitive, operands, params)
    np.testing.assert_array_equal(np.asarray(made), np.asarray(primitive.bind(*operands, **params)))


def test_a_conversion_placed_on_one_mesh_folds_under_another():
    """A scalar conversion placed on one explicit mesh (its `sharding`) folds under another in context, no
    program traced (`executable_only`), as `host_evaluated` binds a constant: its placement none, under no
    mesh."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec

    from dew.telemetry.compiles import executable_only

    placed, other = (jax.make_mesh((1,), (name,), axis_types=(AxisType.Explicit,)) for name in ("a", "b"))
    params = {**_convert(np.int8), "sharding": NamedSharding(placed, PartitionSpec())}
    jax.clear_caches()
    with jax.set_mesh(other), executable_only():
        (made,) = programs.host_evaluated(lax.convert_element_type_p, (np.int32(300),), params)
    assert np.asarray(made).dtype == np.int8 and np.asarray(made) == np.int8(44)


def test_a_device_array_is_bound_where_a_host_literal_folds():
    """A scalar held on a device (a fresh CPU array) is bound, never read to the host to fold, so a transfer
    guard that disallows reading it holds; its value the bind's."""
    x = jax.jit(lax.add)(np.int32(3), np.int32(4))
    with jax.transfer_guard_device_to_host("disallow"):
        (made,) = programs.host_evaluated(lax.add_p, (x, x), {})
    assert int(jax.device_get(made)) == 14


class _Dispatchless(np.ndarray):
    """A NumPy array subclass refusing NumPy's function dispatch, which JAX canonicalizes all the same."""

    def __array_function__(self, *args):
        return NotImplemented


def test_a_scalar_of_a_numpy_subclass_folds_as_jax_canonicalizes_it():
    """A zero-dimensional NumPy subclass refusing NumPy's dispatch folds from JAX's canonical aval (its rank
    and dtype), as the bind takes it, where a rank read by NumPy refused it (base review 68aa03836 F1)."""
    x = np.array(3, dtype=np.int32).view(_Dispatchless)
    (made,) = programs.host_evaluated(lax.add_p, (x, x), {})
    assert np.asarray(made).dtype == np.int32 and int(np.asarray(made)) == 6


def test_a_trees_holes_take_a_mask_as_they_take_a_predicate():
    """`programs.holes` keeps the leaves a mask marks (one bool per leaf in JAX's flatten order, a dict's by
    its sorted keys) as it keeps those a predicate selects, equal leaves apart by their places, its `fill`
    putting values back in those places and every other leaf as it was; a mask shorter or longer than the
    leaves refused (row 8839)."""
    tree = ({"b": 7, "a": 7}, [7, 4.0])
    # Flatten order: a's 7, b's 7, the list's 7, 4.0.
    kept, fill = programs.holes(tree, [False, True, False, True])
    assert kept == [7, 4.0]
    assert fill([20, 30]) == ({"b": 20, "a": 7}, [7, 30])
    by_predicate, _ = programs.holes(tree, lambda leaf: leaf == 4.0)
    assert by_predicate == [4.0]
    for mask in ([True, False], [True, False, False, False, True]):
        with pytest.raises(ValueError):
            programs.holes(tree, mask)
