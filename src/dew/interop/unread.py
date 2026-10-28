"""Stored tensors a load without its weights holds: their shapes and dtypes, and no values.

`Pretrained.load(weights=False)` reads a source's safetensors headers and fetches no tensor's bytes. Its
readers run as a load with weights runs them, on `Unread` arrays in place of the stored tensors: every
NumPy operation on one is evaluated abstractly (`jax.eval_shape` on the operation's `jax.numpy`
counterpart, in NumPy's own dtypes), so a split, a transpose, a cast or a decode computes the shape and
dtype its result would have, and nothing else. What would need a value, a branch on one, a conversion
to a Python number or a NumPy array, raises `PayloadUnread`: a reader that checks or decides on values
reads none here, and says so, rather than computing on values it was not given. A reader takes and
gives `Stored`, the stored tensor it holds either way.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from types import EllipsisType
from typing import TYPE_CHECKING, NoReturn

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from typing_extensions import TypeIs

if TYPE_CHECKING:
    from dew.interop.safetensors_io import LazyTensors


class PayloadUnread(LookupError):
    """A value was asked of a tensor a load without its weights holds (`Unread`), which has its shape
    and dtype alone: whatever asked checks or decides on values, which this load does not read."""


type Operand = (ArrayLike | DTypeLike | Unread | slice | EllipsisType | None
                | Sequence[Operand] | Mapping[str, Operand])
"""What a NumPy function takes and gives: arrays (an `Unread` among them), scalars, dtypes, shapes, axes
and indices, and sequences and keywords of them."""

type Traced = jax.Array | tuple[Traced, ...] | list[Traced]
"""What a `jax.numpy` function gives: an array, or a sequence of them."""

type Abstract = Unread | tuple[Abstract, ...] | list[Abstract]
"""What NumPy gives of `Unread` operands: arrays of the shapes and dtypes its values would have."""


def evaluated(function: Callable[..., Traced], *args: Operand, **kwargs: Operand) -> Abstract:
    """`function`'s result on `args` and `kwargs` with every `Unread` among them abstract: `Unread`
    arrays of the shapes and dtypes it would return."""
    leaves, tree = jax.tree.flatten((args, kwargs), is_leaf=lambda leaf: isinstance(leaf, Unread))
    unread_at = [index for index, leaf in enumerate(leaves) if isinstance(leaf, Unread)]

    def run(*abstract: jax.Array) -> Traced:
        filled = list(leaves)
        for index, value in zip(unread_at, abstract, strict=True):
            filled[index] = value
        given, named = jax.tree.unflatten(tree, filled)
        return function(*given, **named)

    # The stored dtypes as NumPy holds them: float64 and int64 stay so.
    with jax.enable_x64(new_val=True):
        shapes = jax.eval_shape(run, *(jax.ShapeDtypeStruct(leaves[index].shape, leaves[index].dtype)
                                       for index in unread_at))
    return jax.tree.map(lambda leaf: Unread(leaf.shape, leaf.dtype), shapes)


def _array(function: Callable[..., Traced], *args: Operand, **kwargs: Operand) -> Unread:
    """`evaluated` of a `function` that gives one array."""
    shapes = evaluated(function, *args, **kwargs)
    if not isinstance(shapes, Unread):
        raise TypeError(f"{function} gives {type(shapes).__name__}, not one array")
    return shapes


_WEAK = (int, float, complex)
"""The Python scalar types NumPy keeps weak in a ufunc's loop (NEP 50), by exact type: a bool, or a NumPy
scalar that subclasses one of them (`np.float64`), takes its own dtype."""


def _dtype_of(value: ArrayLike) -> np.dtype | type:
    """A ufunc operand's dtype as NumPy resolves loops by, a weak Python scalar's type (`_WEAK`)."""
    if type(value) in _WEAK:
        return type(value)
    return value.dtype if isinstance(value, Unread) else np.asarray(value).dtype


def _cast(value: ArrayLike, dtype: np.dtype) -> ArrayLike:
    """`value` in `dtype`, an `Unread` as one, a weak Python scalar as it is (`_WEAK`)."""
    if type(value) in _WEAK:
        return value
    return value.astype(dtype) if isinstance(value, Unread) else np.asarray(value).astype(dtype)


def _stand_in(value: Operand) -> Operand:
    """An array operand, an `Unread` or one holding values, as ones of its rank and dtype; any other
    operand as it is (`_NUMPY_DTYPED`)."""
    if isinstance(value, Unread | np.ndarray):
        return np.ones((1,) * value.ndim, value.dtype)
    return value


def _retyped(shapes: Abstract, dtypes: tuple[np.dtype, ...]) -> Abstract:
    """`shapes`' arrays in `dtypes`, the dtypes NumPy computes them in."""
    if isinstance(shapes, Unread):
        return Unread(shapes.shape, dtypes[0])
    return tuple(_retyped(shape, (dtype,)) for shape, dtype in zip(shapes, dtypes, strict=True))


def _counterpart(name: str) -> Callable[..., Traced]:
    """`jax.numpy`'s function of NumPy's `name`, which runs abstractly."""
    function = getattr(jnp, name, None)
    if function is None:
        raise PayloadUnread(f"numpy.{name} has no abstract counterpart, so it cannot run on a tensor "
                            "a load without its weights holds")
    return function


class _Operator:
    """An operator of `Unread`'s as NumPy's `ufunc` runs it, with the operands swapped where
    `reflected`: `Unread.__array_ufunc__` runs it abstractly."""

    def __init__(self, ufunc: np.ufunc, *, reflected: bool = False):
        self.ufunc = ufunc
        self.reflected = reflected

    @staticmethod
    def paired(*ufuncs: np.ufunc) -> tuple[_Operator, ...]:
        """Each of `ufuncs`' operator, then its reflection."""
        return tuple(_Operator(ufunc, reflected=reflected) for ufunc in ufuncs for reflected in (False, True))

    def __get__(self, array: Unread, owner: type[Unread]) -> Callable[[ArrayLike], Unread]:
        if self.reflected:
            return lambda other: self.ufunc(other, array)
        return lambda other: self.ufunc(array, other)


class Unread:
    """A stored tensor's shape and dtype with no values (module docstring)."""

    __slots__ = ("dtype", "shape")
    __array_priority__ = 1000

    def __init__(self, shape: Collection[int], dtype: DTypeLike):
        self.shape: tuple[int, ...] = tuple(int(size) for size in shape)
        self.dtype: np.dtype = np.dtype(dtype)

    def __repr__(self) -> str:
        return f"Unread({self.shape}, {self.dtype})"

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    @property
    def itemsize(self) -> int:
        return self.dtype.itemsize

    @property
    def nbytes(self) -> int:
        return self.size * self.itemsize

    def __len__(self) -> int:
        return self.shape[0]

    def __array__(self, dtype: DTypeLike | None = None, copy: bool | None = None) -> NoReturn:
        raise PayloadUnread(f"{self!r}: a NumPy array of its values was asked, which a load without its "
                            "weights does not read")

    def _refused(self) -> NoReturn:
        raise PayloadUnread(f"{self!r}: a Python value of it was asked, which a load without its weights "
                            "does not read")

    __bool__ = __float__ = __int__ = __index__ = __complex__ = tolist = _refused

    def item(self, *index: int) -> NoReturn:
        self._refused()

    # Python's `==` answers one bool of the values; NumPy's elementwise `==`, `np.equal`, runs on an
    # Unread abstractly.
    def __eq__(self, other) -> NoReturn:
        self._refused()

    def __ne__(self, other) -> NoReturn:
        self._refused()

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: ArrayLike,
                        **kwargs: Operand) -> Abstract:
        if kwargs.get("out") is not None:
            raise PayloadUnread(f"numpy.{ufunc.__name__} writes into an array, which a tensor a load "
                                "without its weights holds cannot fill")
        counterpart = _counterpart(ufunc.__name__)
        if method != "__call__":
            return evaluated(getattr(counterpart, method), *inputs, **kwargs)
        # NumPy's own loop for these operands (its promotion, not JAX's lattice), run abstractly.
        resolved = ufunc.resolve_dtypes((*(_dtype_of(value) for value in inputs), *(None,) * ufunc.nout))
        cast = [_cast(value, dtype) for value, dtype in zip(inputs, resolved[:ufunc.nin], strict=True)]
        return _retyped(evaluated(counterpart, *cast, **kwargs), resolved[ufunc.nin:])

    def __array_function__(self, func: Callable[..., Operand], types: Collection[type],
                           args: tuple[Operand, ...], kwargs: dict[str, Operand]) -> Operand:
        if func in _METADATA:
            return _METADATA[func](*args, **kwargs)
        shapes = evaluated(_counterpart(func.__name__), *args, **kwargs)
        if func not in _NUMPY_DTYPED:
            return shapes
        # NumPy's own dtypes, its promotion and accumulation rather than JAX's, on stand-ins of ones.
        given, named = jax.tree.map(_stand_in, (args, kwargs), is_leaf=lambda leaf: isinstance(leaf, Unread))
        dtypes = [np.asarray(standing).dtype for standing in jax.tree.leaves(func(*given, **named))]
        arrays, tree = jax.tree.flatten(shapes, is_leaf=lambda leaf: isinstance(leaf, Unread))
        return jax.tree.unflatten(tree, [Unread(array.shape, dtype)
                                         for array, dtype in zip(arrays, dtypes, strict=True)])

    def __getitem__(self, index: Operand) -> Unread:
        return _array(lambda array, index: array[index], self, index)

    def __setitem__(self, index: Operand, value: ArrayLike) -> None:
        # A write keeps the array's shape and dtype; it is checked to fit where it writes.
        _array(lambda array, index, value: array.at[index].set(value), self, index, value)

    def __getattr__(self, name: str) -> Callable[..., Unread]:
        # The array methods NumPy readers call (`all`, `dot`, `mean`, `clip`, ...): NumPy's function of
        # the name where it has one, as the function form runs on an Unread, else `jax.Array`'s.
        if name.startswith("_") or not callable(getattr(np.ndarray, name, None)):
            raise AttributeError(name)
        function = getattr(np, name, None)
        if function is not None:
            return lambda *args, **kwargs: function(self, *args, **kwargs)
        return lambda *args, **kwargs: _array(
            lambda array, *given, **named: getattr(array, name)(*given, **named), self, *args, **kwargs)

    def __iter__(self) -> Iterator[Unread]:
        return (self[position] for position in range(len(self)))

    def reshape(self, *shape: int | tuple[int, ...], order: str = "C") -> Unread:
        return _array(lambda array: array.reshape(*shape, order=order), self)

    def transpose(self, *axes: int | tuple[int, ...]) -> Unread:
        return _array(lambda array: array.transpose(*axes), self)

    def swapaxes(self, first: int, second: int) -> Unread:
        return _array(lambda array: array.swapaxes(first, second), self)

    def astype(self, dtype: DTypeLike, *, copy: bool = True) -> Unread:
        return Unread(self.shape, dtype)

    def view(self, dtype: DTypeLike | None = None) -> Unread:
        # NumPy's geometry: the last axis holds as many of the new items as its bytes do.
        target = self.dtype if dtype is None else np.dtype(dtype)
        if target.itemsize == self.itemsize:
            return Unread(self.shape, target)
        bytes_held = self.shape[-1] * self.itemsize if self.shape else 0
        if not self.shape or bytes_held % target.itemsize:
            raise ValueError(f"{self!r} cannot be viewed as {target}: its last axis holds {bytes_held} bytes")
        return Unread((*self.shape[:-1], bytes_held // target.itemsize), target)

    def copy(self, order: str = "C") -> Unread:
        return self

    @property
    def T(self) -> Unread:
        return self.transpose()

    # NumPy and JAX keep the dtype of each of these, but the absolute of a complex array (a float).
    def __neg__(self) -> Unread:
        return _array(jnp.negative, self)

    def __invert__(self) -> Unread:
        return _array(jnp.invert, self)

    def __abs__(self) -> Unread:
        return _array(jnp.absolute, self)

    # Each binary ufunc's operator and its reflection, in this order.
    (__add__, __radd__, __sub__, __rsub__, __mul__, __rmul__, __truediv__, __rtruediv__, __floordiv__,
     __rfloordiv__, __mod__, __rmod__, __pow__, __rpow__, __matmul__, __rmatmul__, __and__, __rand__, __or__,
     __ror__, __xor__, __rxor__, __lshift__, __rlshift__, __rshift__, __rrshift__) = _Operator.paired(
        np.add, np.subtract, np.multiply, np.true_divide, np.floor_divide, np.remainder, np.power, np.matmul,
        np.bitwise_and, np.bitwise_or, np.bitwise_xor, np.left_shift, np.right_shift)
    __lt__, __le__, __gt__, __ge__ = (_Operator(np.less), _Operator(np.less_equal), _Operator(np.greater),
                                      _Operator(np.greater_equal))


type Stored = np.ndarray | Unread
"""A stored tensor as a reader holds it: its values, or, of a load without its weights, its shape and
dtype alone (`Unread`), each going through the reader's NumPy."""


def held(value: ArrayLike | Unread, dtype: DTypeLike | None = None, *, contiguous: bool = False) -> Stored:
    """`np.asarray(value, dtype)` (`np.ascontiguousarray` with `contiguous`), an `Unread` kept one, cast
    to `dtype`: the stored tensor a reader goes on with, values or not."""
    if isinstance(value, Unread):
        return value if dtype is None else value.astype(dtype)
    return np.ascontiguousarray(value, dtype) if contiguous else np.asarray(value, dtype)


def valued(value: Stored) -> TypeIs[np.ndarray]:
    """Whether `value` holds its values, not an `Unread`: a check on values runs where they are held,
    and a load without its weights checks the shapes alone."""
    return not isinstance(value, Unread)


def equal(first: Stored, second: Stored) -> bool:
    """Whether `first` and `second` hold equal values, as `np.array_equal`; where either is `Unread`,
    whether their shapes are, all a load without its weights can check."""
    if isinstance(first, Unread) or isinstance(second, Unread):
        return tuple(first.shape) == tuple(second.shape)
    return bool(np.array_equal(first, second))


def lookup(table: np.ndarray, index: Stored) -> Stored:
    """`table[index]` along its first axis, `index` an integer array of codes; an `Unread` index gives
    an `Unread` of the result's shape and dtype (NumPy's own `take` reads its index)."""
    if isinstance(index, Unread):
        return _array(lambda codes: jnp.take(jnp.asarray(table), codes, axis=0), index)
    return np.take(table, index, axis=0)


_NUMPY_DTYPED: frozenset[Callable[..., Operand]] = frozenset({
    np.concatenate, np.stack, np.hstack, np.vstack, np.append, np.where, np.dot, np.vdot, np.inner, np.outer,
    np.tensordot, np.einsum, np.kron, np.clip, np.sum, np.prod, np.cumsum, np.cumprod, np.mean})
"""The NumPy functions whose result dtype NumPy's promotion or accumulation decides, which JAX's lattice
does not follow (an int32 beside a float32 is float64 in NumPy's, float32 in JAX's; an int32 sum is int64),
and which no operand's shape decides: NumPy answers it on stand-ins of ones of each array operand's rank
and dtype (`_stand_in`), and `jax.eval_shape` the shapes."""


def _shape_of(array: Stored) -> tuple[int, ...]:
    return tuple(array.shape)


def _rank_of(array: Stored) -> int:
    return len(array.shape)


def _size_of(array: Stored, axis: int | None = None) -> int:
    return math.prod(array.shape) if axis is None else array.shape[axis]


def _result_type_of(*arrays: ArrayLike | DTypeLike) -> np.dtype:
    return np.result_type(*(array.dtype if isinstance(array, Unread) else array for array in arrays))


_METADATA: dict[Callable[..., Operand], Callable[..., Operand]] = {
    np.shape: _shape_of, np.ndim: _rank_of, np.size: _size_of, np.result_type: _result_type_of}
"""The NumPy functions an `Unread` answers from its shape and dtype, which hold what they return."""


def unread(specs: Mapping[str, jax.ShapeDtypeStruct]) -> LazyTensors[Unread]:
    """`specs`' tensors as a load without its weights holds them: each an `Unread` of its shape and
    dtype, its readers' NumPy run on it abstractly."""
    # The safetensors reader's module reads weights through this one.
    from dew.interop.safetensors_io import LazyTensors

    return LazyTensors(specs, lambda name: Unread(specs[name].shape, specs[name].dtype))
