"""Stored tensors a load without its weights holds: their shapes and dtypes, and no values.

`Pretrained.load(weights=False)` reads a source's safetensors headers and fetches no tensor's bytes. Its
readers run as a load with weights runs them, on `Unread` arrays in place of the stored tensors: every
NumPy operation on one is evaluated abstractly (`jax.eval_shape` on the operation's `jax.numpy`
counterpart), so a split, a transpose, a cast or a decode computes the shape and dtype its result would
have, and nothing else. What would need a value, a branch on one, a conversion to a Python number or a
NumPy array, raises `PayloadUnread`: a reader that checks or decides on values reads none here, and says
so, rather than computing on values it was not given.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np


class PayloadUnread(LookupError):
    """A value was asked of a tensor a load without its weights holds (`Unread`), which has its shape
    and dtype alone: whatever asked checks or decides on values, which this load does not read."""


def _abstract(value):
    return jax.ShapeDtypeStruct(value.shape, value.dtype) if isinstance(value, Unread) else value


def evaluated(function: Callable, *args, **kwargs):
    """`function`'s result on `args` and `kwargs` with every `Unread` among them abstract: `Unread`
    arrays of the shapes and dtypes it would return, its other values as they are."""
    leaves, tree = jax.tree.flatten((args, kwargs), is_leaf=lambda leaf: isinstance(leaf, Unread))
    held = [index for index, leaf in enumerate(leaves) if isinstance(leaf, Unread)]

    def run(*abstract):
        filled = list(leaves)
        for index, value in zip(held, abstract, strict=True):
            filled[index] = value
        given, named = jax.tree.unflatten(tree, filled)
        return function(*given, **named)

    # The stored dtypes as NumPy holds them: float64 and int64 stay so.
    with jax.enable_x64(new_val=True):
        result = jax.eval_shape(run, *(_abstract(leaves[index]) for index in held))
    return jax.tree.map(lambda leaf: Unread(leaf.shape, leaf.dtype)
                        if isinstance(leaf, jax.ShapeDtypeStruct) else leaf, result)


def _counterpart(function, name: str):
    held = getattr(jnp, name, None)
    if held is None:
        raise PayloadUnread(f"numpy.{name} has no abstract counterpart, so it cannot run on a tensor "
                            "a load without its weights holds")
    return held


class Unread:
    """A stored tensor's shape and dtype with no values (module docstring)."""

    __slots__ = ("dtype", "shape")
    __array_priority__ = 1000

    def __init__(self, shape, dtype):
        self.shape = tuple(int(size) for size in shape)
        self.dtype = np.dtype(dtype)

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

    def __array__(self, dtype=None, copy=None):
        raise PayloadUnread(f"{self!r}: a NumPy array of its values was asked, which a load without its "
                            "weights does not read")

    def _refused(self, *_):
        raise PayloadUnread(f"{self!r}: a Python value of it was asked, which a load without its weights "
                            "does not read")

    __bool__ = __float__ = __int__ = __index__ = __complex__ = _refused

    def item(self, *_):
        self._refused()

    def tolist(self):
        self._refused()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if kwargs.get("out") is not None:
            raise PayloadUnread(f"numpy.{ufunc.__name__} writes into an array, which a tensor a load "
                                "without its weights holds cannot fill")
        held = _counterpart(ufunc, ufunc.__name__)
        return evaluated(held if method == "__call__" else getattr(held, method), *inputs, **kwargs)

    def __array_function__(self, function, types, args, kwargs):
        return evaluated(_counterpart(function, function.__name__), *args, **kwargs)

    def __getitem__(self, index):
        return evaluated(lambda array, index: array[index], self, index)

    def __setitem__(self, index, value):
        # A write keeps the array's shape and dtype; it is checked to fit where it writes.
        evaluated(lambda array, index, value: array.at[index].set(value), self, index, value)

    def __getattr__(self, name: str):
        # The array methods NumPy readers call (`all`, `any`, `mean`, `clip`, ...), as `jax.Array`'s.
        if name.startswith("_") or not callable(getattr(jax.Array, name, None)):
            raise AttributeError(name)
        return lambda *args, **kwargs: evaluated(lambda array: getattr(array, name)(*args, **kwargs), self)

    def __iter__(self):
        return (self[position] for position in range(len(self)))

    def reshape(self, *shape, **kwargs):
        return evaluated(lambda array: array.reshape(*shape, **kwargs), self)

    def transpose(self, *axes):
        return evaluated(lambda array: array.transpose(*axes), self)

    def swapaxes(self, first, second):
        return evaluated(lambda array: array.swapaxes(first, second), self)

    def astype(self, dtype, **_):
        return Unread(self.shape, dtype)

    def view(self, dtype=None):
        if dtype is None or np.dtype(dtype).itemsize == self.itemsize:
            return Unread(self.shape, self.dtype if dtype is None else dtype)
        return evaluated(lambda array: jax.lax.bitcast_convert_type(array, dtype), self)

    def copy(self, *_):
        return self

    @property
    def T(self):
        return self.transpose()

    def _binary(name):
        def operate(self, other):
            return getattr(np, name)(self, other)

        def reflected(self, other):
            return getattr(np, name)(other, self)
        return operate, reflected

    __add__, __radd__ = _binary("add")
    __sub__, __rsub__ = _binary("subtract")
    __mul__, __rmul__ = _binary("multiply")
    __truediv__, __rtruediv__ = _binary("true_divide")
    __floordiv__, __rfloordiv__ = _binary("floor_divide")
    __mod__, __rmod__ = _binary("remainder")
    __pow__, __rpow__ = _binary("power")
    __matmul__, __rmatmul__ = _binary("matmul")
    __and__, __rand__ = _binary("bitwise_and")
    __or__, __ror__ = _binary("bitwise_or")
    __xor__, __rxor__ = _binary("bitwise_xor")
    __lshift__, __rlshift__ = _binary("left_shift")
    __rshift__, __rrshift__ = _binary("right_shift")
    __lt__, _ = _binary("less")
    __le__, _ = _binary("less_equal")
    __gt__, _ = _binary("greater")
    __ge__, _ = _binary("greater_equal")
    __eq__, _ = _binary("equal")
    __ne__, _ = _binary("not_equal")
    del _binary, _
    __hash__ = None

    def __neg__(self):
        return np.negative(self)

    def __invert__(self):
        return np.invert(self)

    def __abs__(self):
        return np.absolute(self)


def held(value, dtype=None, *, contiguous: bool = False):
    """`np.asarray(value, dtype)` (`np.ascontiguousarray` with `contiguous`), an `Unread` kept one: the
    array a reader goes on with, values or not."""
    if isinstance(value, Unread):
        return value if dtype is None else value.astype(dtype)
    return np.ascontiguousarray(value, dtype) if contiguous else np.asarray(value, dtype)


def equal(first, second) -> bool:
    """Whether `first` and `second` hold equal values, as `np.array_equal`; where either is `Unread`,
    whether their shapes and dtypes are, all a load without its weights can check."""
    if valued(first, second):
        return bool(np.array_equal(first, second))
    return (tuple(first.shape), np.dtype(first.dtype)) == (tuple(second.shape), np.dtype(second.dtype))


def lookup(table: np.ndarray, index):
    """`table[index]` along its first axis, `index` an integer array of codes; an `Unread` index gives
    an `Unread` of the result's shape and dtype (NumPy's own `take` reads its index)."""
    if isinstance(index, Unread):
        return evaluated(lambda codes: jnp.take(jnp.asarray(table), codes, axis=0), index)
    return np.take(table, index, axis=0)


def valued(*values) -> bool:
    """Whether each of `values` holds its values, none `Unread`: a check on values runs where they are
    held, and a load without its weights checks the shapes alone."""
    return not any(isinstance(value, Unread) for value in values)


def unread(specs: Mapping[str, jax.ShapeDtypeStruct]):
    """`specs`' tensors as a load without its weights holds them: each an `Unread` of its shape and
    dtype."""
    from dew.interop.safetensors_io import LazyTensors

    return LazyTensors(specs, lambda name: Unread(specs[name].shape, specs[name].dtype))
