"""The one precision context the reference generators build and run their float64 oracles under, with
torch alone, so a generator in a torch-only environment imports it as freely as one beside Diffusers
or transformers.

A reference that runs published code at another precision than the code pins
replaces torch's own casts for a block (`replaced`). Each generator names
the casts it replaces and what they become, its precision policy; the
replacements here are the ones several generators make, and `float64` is the
float64 oracle's: a model built and run inside it computes in float64 every
value its code computes, its constructor's tables included. A model built in
float32 and cast afterwards keeps what its constructor computed rounded to
float32 (a rotary `inv_freq`, Gemma's `embed_scale`), so a truth builds its
model inside `float64`, through its library's public factory at float64
(transformers' `from_config(config, dtype=torch.float64)`), loads the stored
weights into it, and runs its whole evaluation there, generation, loss and
backward included. A scheduler's tables are its own module's, widened by the
diffusers tools' `float64_scheduler` around its construction and every call.
"""

import contextlib
import inspect
import os
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
    included, rather than the function the attribute reads as.

    torch's device contexts (`with torch.device("meta")`, which transformers builds its models under)
    learn which functions are constructors once, at their first use in the process, and keep those
    function objects; first used inside a block that replaced `torch.arange`, they would hold the
    replacement for the rest of the process, and an `arange` under a later device context would make
    its tensor on the CPU (ledger 1553). So a device context is used once before anything is
    replaced."""
    with torch.device("meta"):
        torch.empty(0)
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


@contextlib.contextmanager
def widened():
    """The reference in float64: every float32 pin a model's code carries widened, which are its
    `.float()`, `.to(torch.float32)` and `.type(torch.float32)` casts (Nemotron-H's router pins the
    last), a float32 `arange` or `linspace` (the sinusoids', a rotary table's, a schedule's), a float32
    `torch.tensor`, `torch.full`, `torch.zeros`, `torch.ones` or `torch.empty` (the tables a model builds
    from Python numbers, Phi-3's LongRoPE factors among them, and the buffers generation adds a bias or a
    penalty into, which must match the widened scores they meet), and float64 the default dtype."""
    type_ = torch.Tensor.type

    def widened_type(self, dtype=None, *args, **kwargs):
        return type_(self, *(() if dtype is None else (wide(dtype),)), *args, **kwargs)

    with (replaced(float_as(torch.float64), (torch.Tensor, "to", widened_to(torch.Tensor.to)),
                   (torch.Tensor, "type", widened_type), (torch, "arange", dtype_widened(torch.arange)),
                   (torch, "linspace", dtype_widened(torch.linspace)),
                   *((torch, name, dtype_widened(getattr(torch, name)))
                     for name in ("tensor", "full", "zeros", "ones", "empty"))),
          default_dtype(torch.float64)):
        yield


@contextlib.contextmanager
def widened_softmax():
    """`softmax(..., dtype=torch.float32)`, the pin transformers' eager attention and routers carry, in
    float64."""
    with replaced((torch.nn.functional, "softmax", dtype_widened(torch.nn.functional.softmax))):
        yield


@contextlib.contextmanager
def float64():
    """A model built and run in float64: `widened` and `widened_softmax` together."""
    with widened(), widened_softmax():
        yield


def float64_twin(model: torch.nn.Module,
                 build: Callable[[], torch.nn.Module] | None = None) -> torch.nn.Module:
    """`model`'s float64 truth, the one way a truth's twin is made: built inside `float64`, so every value
    its constructor computes (a rotary `inv_freq`, Gemma's `embed_scale`) is float64's where a float32
    build cast afterwards keeps float32's, by `build` (a library's reader at float64, a head from its
    config) or else its class from a copy of its config at float64 (Diffusers' `from_config` for a
    config that is no transformers one), and its floating tensors promoted to float64 where a reader
    decodes another dtype, which computes nothing a construction did; `model`'s weights and persistent
    buffers loaded into it strictly, on `model`'s device, in eval mode. Run it inside `float64` too.

    A complex table keeps its imaginary half: built inside `float64` it is complex128 already, and
    `Module.to(torch.float64)` would cast it to a real dtype, dropping the sines of a rotation (Llama 4's
    vision `freqs_ci`), where `double` promotes floating tensors alone."""
    import copy

    with float64():
        if build is not None:
            twin = build()
        else:
            config = copy.deepcopy(model.config)
            if hasattr(type(model), "from_config") and not hasattr(config, "to_dict"):
                twin = type(model).from_config(config)
            else:
                config.dtype = torch.float64
                twin = type(model)(config)
        twin.double()
    twin.load_state_dict(model.state_dict(), strict=True)
    return twin.to(next(model.parameters()).device).eval()


def recorded_matmuls(**solves) -> str:
    """What a reference's array file records of its run's matmuls (tests/reference_error.py's
    `PRECISION`): 'ieee' where torch, Triton's default and each of `solves` (its name to the mode it
    ran, as fla's triangular solves name theirs) keep fp32, else those that do not, which the
    accuracy rule refuses."""
    reduced = torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
    modes = {"torch": "tf32" if reduced else "ieee",
             "TRITON_F32_DEFAULT": os.environ.get("TRITON_F32_DEFAULT", "tf32"), **solves}
    return ", ".join(f"{name} {mode}" for name, mode in modes.items() if mode != "ieee") or "ieee"
