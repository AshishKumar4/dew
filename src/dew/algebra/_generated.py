"""Case keys and dispatch for checked-in generated modules, without loading the dev-time registry."""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType

import jax
import jax.numpy as jnp
from jax.core import ShapedArray


def case_of(operands: Sequence) -> tuple:
    """The flat, unplaced array ABI admitted by the generator."""
    return tuple((tuple(value.shape), jnp.dtype(value.dtype).name) for value in operands)


def refused(operands: Sequence) -> str | None:
    """Why `operands` are not the generated ABI, strongly typed unplaced arrays alone: a Python number, a
    weakly typed value or a placed one keys no case as the written math reads it. None where they are."""
    for operand in operands:
        if not (hasattr(operand, "shape") and hasattr(operand, "dtype")):
            return f"an operand of type {type(operand).__name__}, not an array"
        aval = jax.typeof(operand)
        if not isinstance(aval, ShapedArray) or aval.weak_type or not aval.sharding.mesh.empty:
            return f"an operand {aval}, not a strongly typed unplaced array"
    return None


@dataclasses.dataclass(frozen=True)
class Forms:
    """A generated function's checked forms by policy, target and case, and its return convention: one value
    (`single`, its forms' one-element tuples unwrapped) or its forms' tuple."""

    single: bool
    cases: Mapping[tuple, Callable[..., tuple]]


def table_name(name: str) -> str:
    """The name of a generated module's table of `name`'s forms (`Forms`)."""
    return f"_{name}_forms"


def table(module: ModuleType, name: str) -> Forms:
    """The generated `module`'s table of `name`'s forms; one it lacks raises (the module is stale: generate
    it again)."""
    return getattr(module, table_name(name))


def found(forms: Forms, operands: Sequence, policy: str, target: str) -> Callable[..., tuple] | None:
    """The checked form for `operands`' exact policy, target, shape and dtype key, one lookup; None for
    operands of no generated ABI (`refused`) or a case the table does not hold."""
    if refused(operands) is not None:
        return None
    return forms.cases.get((policy, target, case_of(operands)))


def entry(form: Callable[..., tuple], *, single: bool) -> Callable:
    """A checked form as its declaration's function returns, for a caller that runs it later: its one value
    where `single`."""
    return (lambda *operands: form(*operands)[0]) if single else form


def call(forms: Forms, operands: tuple, policy: str, target: str):
    """Run only the checked form for the exact policy, target, shape and dtype key, as its function
    returns."""
    outputs = forms.cases[policy, target, case_of(operands)](*operands)
    return outputs[0] if forms.single else outputs
