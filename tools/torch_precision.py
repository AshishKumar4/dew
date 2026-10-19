"""The one precision context the reference generators build and run their float64 oracles under, with
torch alone, so a generator in a torch-only environment imports it as freely as one beside Diffusers
or transformers (tools/diffusers_wan_reference.py widens on top of it).

A reference that runs published code at another precision than the code pins
replaces torch's own casts for a block (`replaced`). Each generator names
the casts it replaces and what they become, its precision policy; the
replacements here are the ones several generators make.
"""

import contextlib
import inspect
from collections.abc import Callable

import torch


def wide(dtype):
    """`dtype`, float32 read as float64: what a widened reference casts to where it pins float32."""
    return torch.float64 if dtype == torch.float32 else dtype


@contextlib.contextmanager
def default_dtype(dtype):
    """torch's default floating dtype `dtype` within the block, the previous one restored after it."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


@contextlib.contextmanager
def replaced(*patches: tuple[object, str, object]):
    """Each `(owner, name, value)` set as `owner.name` within the block, every previous value restored
    after it, whether or not the block raised. A class keeps its own descriptor, a staticmethod
    included, rather than the function the attribute reads as."""
    kept = [(owner, name, inspect.getattr_static(owner, name)) for owner, name, _ in patches]
    try:
        for owner, name, value in patches:
            setattr(owner, name, value)
        yield
    finally:
        for owner, name, value in reversed(kept):
            setattr(owner, name, value)


def float_as(dtype) -> tuple[object, str, object]:
    """The `replaced` patch under which `.float()` casts to `dtype`, the run's own precision, as
    published code that calls it means float32 only where float32 is the run's."""
    return torch.Tensor, "float", lambda tensor, *args, **kwargs: tensor.to(dtype)


def dtype_widened(function: Callable) -> Callable:
    """`function` with its `dtype` keyword widened (`wide`): a constructor, `zeros_like`, `new_zeros`
    or `softmax` that pins float32 makes float64 instead."""
    return lambda *args, dtype=None, **kwargs: function(*args, dtype=wide(dtype), **kwargs)


def widened_to(to: Callable) -> Callable:
    """`Tensor.to` with a float32 target widened, given positionally or as `dtype`; a device, a tensor
    or any other dtype passes through."""
    def converted(tensor, *args, **kwargs):
        if "dtype" in kwargs:
            kwargs["dtype"] = wide(kwargs["dtype"])
        return to(tensor, *(wide(arg) if isinstance(arg, torch.dtype) else arg for arg in args), **kwargs)

    return converted
