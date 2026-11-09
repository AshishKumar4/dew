"""The rounding model of check 1's weighed check (dew.parity's `Weighed`), and its tap of a torch
program: each rounding of a program moves its value by its bound w(v) u times a random sign, every
operation its own draw, and an output element's rounding scale c is the root mean square of its
displacement over `DRAWS` draws (tests/rounding_bound.py reads a JAX program's alike).

`rounding_scale` reads a torch program's, its float64 twin's computation tapped op by op, so a reference's
scale is its own computation's and an arm judged against it weighs none of its own errors down
(docs/design/testing.md). It imports torch only when its tap is called, so a JAX program's scale
(tests/rounding_bound.py) reads its constants without it."""

import functools
import math
from collections.abc import Callable

import numpy as np

ULPS = {"exp": 1.42, "log": 2.05, "log1p": 3.64, "expm1": 1.21, "logistic": 2.49, "tanh": 4.9,
        "erf": 6.71, "rsqrt": 2.8, "sin": 1.07, "cos": 1.07}
"""Each elementary function's error bound, in units of u times its true result: the largest of XLA
CPU's float32 implementations over every normal float32 input whose result is a normal float32, read
exhaustively against float64 references (2^32 inputs each, 11-02; sin and cos 11-08, 1.0606 each, a
rotary position's), rounded up at the second decimal.
XLA CPU flushes subnormal inputs to zero, so the bound holds where no elementary function reads one."""

DRAWS = 8
"""The tap draws a rounding scale averages (`rounding_scale`): over 8 draws its root mean square is at least
0.41 of its expectation in 995 of 1000 elements (a chi-square of 8 degrees of freedom's 0.005 quantile is
1.344, and sqrt(1.344 / 8) = 0.41)."""


def rounding_scale[Tree](fn: Callable[[], Tree], unit: float, seed: int = 0) -> tuple[Tree, Tree]:
    """`fn()`'s float64 outputs and each floating output element's rounding scale c (module docstring),
    a non-floating one's None: `fn` a torch program's float64 twin's computation, its forward, its
    `torch.autograd.grad` or its `torch.func.jvp`, run eagerly `DRAWS + 1` times with every aten operation
    it dispatches tapped where it rounds (`_tapped`), composite operations decomposed into the primitive
    ones they compute (as a softmax or a norm rounds each step), and `unit` the roundoff of the format the
    program computes in (2^-24 for float32), every operation's alike. `seed` keys the draws, so the same
    draw of the same program weighs alike for every arm. An operation of no rule is refused, naming it
    (a fused attention's backward of several results among them: the program is run with its attention's
    eager form)."""
    def run(scale: float, draw: int):
        with _tapped()(scale * unit, seed * (DRAWS + 1) + draw):
            return fn()

    plain = run(0.0, 0)
    leaves, rebuild = _flattened(plain)
    moved = [_flattened(run(1.0, draw + 1))[0] for draw in range(DRAWS)]
    scales = []
    for at, value in enumerate(leaves):
        if not np.issubdtype(value.dtype, np.floating):
            scales.append(None)
            continue
        shifts = np.stack([np.abs(draw[at] - value) for draw in moved])
        largest = shifts.max(axis=0)
        held = np.where(largest > 0, largest, 1)
        scales.append(np.where(largest > 0, largest * np.sqrt(np.mean(np.square(shifts / held), axis=0)),
                               largest))
    return rebuild(leaves), rebuild(scales)


def _flattened(tree):
    """`tree`'s tensor and array leaves as float64 NumPy arrays where floating, and the function rebuilding
    a tree of its shape from such leaves."""
    import torch
    import torch.utils._pytree as pytree

    leaves, spec = pytree.tree_flatten(tree)
    held = [np.asarray(leaf.detach().cpu().numpy() if isinstance(leaf, torch.Tensor) else leaf)
            for leaf in leaves]
    return held, lambda values: pytree.tree_unflatten(list(values), spec)


EXACT_ATEN = frozenset({
    "neg", "abs", "maximum", "minimum", "max", "min", "amax", "amin", "where", "clone", "copy", "copy_",
    "_to_copy", "view", "_unsafe_view", "reshape", "expand", "permute", "transpose", "t", "squeeze",
    "unsqueeze", "slice", "select", "index", "index_select", "gather", "cat", "stack", "split",
    "split_with_sizes", "unbind", "narrow", "flip", "roll", "repeat", "constant_pad_nd", "full",
    "full_like", "zeros", "zeros_like", "ones", "ones_like", "empty", "empty_like", "empty_strided",
    "new_zeros", "new_ones", "new_empty", "new_full", "fill", "fill_", "zero_", "arange", "scalar_tensor",
    "lift_fresh", "lift_fresh_copy", "detach", "alias", "eq", "ne", "lt", "le", "gt", "ge", "logical_and",
    "logical_or", "logical_not", "bitwise_and", "bitwise_or", "bitwise_not", "isnan", "isinf", "sign",
    "floor", "ceil", "round", "trunc", "argmax", "argmin", "sort", "topk", "clamp", "clamp_min",
    "clamp_max", "masked_fill", "masked_fill_", "tril", "triu", "_local_scalar_dense", "as_strided",
    "embedding", "index_put", "index_put_", "scatter", "scatter_",
    "slice_scatter", "select_scatter", "diagonal", "diag_embed", "expand_copy", "view_copy",
    "permute_copy", "squeeze_copy", "unsqueeze_copy", "t_copy", "transpose_copy", "set_", "resize_",
    "_unsafe_index", "nonzero", "unfold", "any", "all", "bernoulli", "rand", "randn", "randint", "to",
    "type_as", "contiguous", "_conj", "resolve_conj", "resolve_neg", "_new_zeros_with_same_feature_meta",
    "_has_same_storage_numel", "_make_dual", "_unpack_dual", "_fw_primal", "_efficientzerotensor"})
"""Aten operations whose float results are their operands' values moved, selected or made, or exact: no
rounding (tests/rounding_bound.py's `EXACT`); a conversion in the float64 twin is to float64 itself, and
forward-mode AD's own bookkeeping (a tangent's zeros made to its primal's layout, a dual's halves) moves
nothing."""
ONCE_ATEN = frozenset({"add", "add_", "sub", "sub_", "mul", "mul_", "div", "div_", "sqrt", "reciprocal",
                       "rsub", "true_divide", "addcmul", "addcdiv"})
"""IEEE 754's correctly rounded operations: one rounding, at most u times the result (`ONCE`); a fused
`addcmul` and `addcdiv` round their product and their sum, weighed by the result alike."""
ULPS_ATEN = {"exp": "exp", "log": "log", "log1p": "log1p", "expm1": "expm1", "sigmoid": "logistic",
             "tanh": "tanh", "erf": "erf", "rsqrt": "rsqrt"}
"""Each elementary aten operation by the name of its bound in `ULPS`."""
CIRCULAR = {"sin", "cos"}
"""The circular functions, whose bound in `ULPS` is XLA's own implementation's: torch computes them on the CPU
through SLEEF's 1.0-ULP functions (`Sleef_sinf8_u10`, `Sleef_cosf8_u10`: within one unit in the last place),
at most 2u times the result."""
SUMS_ATEN = frozenset({"mm", "bmm", "addmm", "baddbmm", "sum", "cumsum", "convolution", "mean", "dot", "mv",
                       "addmv", "linalg_vecdot", "index_add", "scatter_add", "embedding_dense_backward"})
"""The operations that sum terms (`float64_terms.SUMS`), weighed by their terms' count times the sum of the
terms' magnitudes (`_terms`)."""


@functools.cache
def _tapped():
    """The tap's dispatch mode (`Tapped`), made where it is first asked, so torch loads only then."""
    import torch
    from torch._decomp import decomposition_table
    from torch.utils._python_dispatch import TorchDispatchMode

    class Tapped(TorchDispatchMode):
        """Every aten operation dispatched in the block decomposed where torch decomposes it, and each
        floating result it rounds moved by its weight (`_weight`) times `unit` times a random sign, drawn
        from a stream keyed by `seed` and the operation's place in the run; `unit` 0 moves nothing."""

        def __init__(self, unit: float, seed: int):
            super().__init__()
            self.unit, self.generator = unit, torch.Generator().manual_seed(int(seed))

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = func.overloadpacket.__name__
            ruled = name == "pow" or any(name in rules for rules in (SUMS_ATEN, ULPS_ATEN, ONCE_ATEN,
                                                                     EXACT_ATEN, CIRCULAR))
            if not ruled:
                # A composite operation is the primitive ones it computes: its own composite form where it
                # has one (a math attention's), else torch's decomposition of it.
                with self:
                    composed = func.decompose(*args, **kwargs)
                    if composed is not NotImplemented:
                        return composed
                    if func in decomposition_table:
                        return decomposition_table[func](*args, **kwargs)
            computed = func(*args, **kwargs)
            if name in EXACT_ATEN:
                return computed
            if isinstance(computed, (tuple, list)):
                if name == "convolution_backward":
                    weights = _convolution_backward_weights(func, args, kwargs)
                    return type(computed)(part if part is None else self.tapped(part, weight)
                                        for part, weight in zip(computed, weights, strict=True))
                # Several results of one operation: no other rule weighs them apart.
                if any(isinstance(part, torch.Tensor) and part.is_floating_point() for part in computed):
                    raise ValueError(f"the rounding scale has no rule for aten.{name}'s several results: "
                                     "decompose it or add it to EXACT_ATEN")
                return computed
            if not isinstance(computed, torch.Tensor) or not computed.is_floating_point():
                return computed
            moved = self.tapped(computed, _weight(name, func, args, kwargs, computed))
            if name.endswith("_") and moved is not computed:
                # An in-place operation's value is its first operand: the move written where it lives.
                args[0].copy_(moved)
                return args[0]
            return moved

        def tapped(self, computed, weight):
            """`computed`, an operation's value, moved by `weight` times the unit times a random sign an
            element, drawn on every run, the untapped one's too, so each operation's stream is its place's
            alone; an infinity held as it is (a mask's -inf, an overflow), whose tap would be inf - inf."""
            signs = torch.randint(0, 2, computed.shape, generator=self.generator).to(computed.dtype) * 2 - 1
            if self.unit == 0:
                return computed
            tap = self.unit * weight * signs.to(computed.device)
            return torch.where(torch.isfinite(computed), computed + tap, computed)

    return Tapped


def _weight(name: str, func, args, kwargs, computed):
    """The most one rounding of aten `func` moves its float64 `computed`, in units of u
    (tests/rounding_bound.py's `_weight`); an operation of no rule refused, naming it."""
    import torch

    magnitude = computed.abs()
    if name in ONCE_ATEN:
        return magnitude
    if name in ULPS_ATEN:
        return ULPS[ULPS_ATEN[name]] * magnitude
    if name in CIRCULAR:
        return 2 * magnitude
    if name == "pow":
        exponent = args[1]
        if isinstance(exponent, (int, float)) and float(exponent).is_integer():
            return max(abs(int(exponent)) - 1, 1) * magnitude
        return ULPS["exp"] * magnitude + ULPS["log"] * magnitude
    if name in SUMS_ATEN:
        magnitudes = [arg.abs() if isinstance(arg, torch.Tensor) and arg.is_floating_point() else arg
                      for arg in args]
        summed = _terms(name, func, args, kwargs) * func(*magnitudes, **kwargs).abs()
        # A mean's division by its count rounds once more, by its computed.
        return summed + magnitude if name == "mean" else summed
    raise ValueError(f"the rounding scale has no rule for aten.{name}: add it to EXACT_ATEN, ONCE_ATEN, "
                     "ULPS_ATEN, CIRCULAR or SUMS_ATEN")


def _convolution_backward_weights(func, args, kwargs):
    """A convolution's gradients' weights, each a sum (`_terms`): its input's, the cotangent's products with
    the kernel, `out_channels / groups` times the window's of them; its kernel's and its bias's, the
    cotangent's with the input over the batch and every output position; each the same gradient of the
    operands' magnitudes times its count."""
    grad, given, kernel = args[:3]
    magnitudes = func(grad.abs(), given.abs(), kernel.abs(), *args[3:], **kwargs)
    # aten::convolution_backward(grad_output, input, weight, bias_sizes, stride, padding, dilation,
    # transposed, output_padding, groups, output_mask).
    groups = args[9]
    window = math.prod(kernel.shape[2:])
    positions = grad.shape[0] * math.prod(grad.shape[2:])
    counts = (kernel.shape[0] // groups * window, positions, positions)
    return tuple(None if magnitude is None else count * magnitude.abs()
                 for magnitude, count in zip(magnitudes, counts, strict=True))


def _terms(name: str, func, args, kwargs) -> int:
    """How many terms each element of a sum adds (tests/rounding_bound.py's `_terms_count`): a product's
    contraction length, a reduction's `m - 1`, a convolution's window times its input features a group, a
    scatter-add's updates."""
    if name in ("mm", "bmm", "dot", "mv", "linalg_vecdot"):
        return int(args[0].shape[-1])
    if name in ("addmm", "baddbmm", "addmv"):
        return int(args[1].shape[-1]) + 1
    if name == "convolution":
        weight = args[1]
        return int(weight.shape[1] * math.prod(weight.shape[2:]))
    if name in ("sum", "mean"):
        dims = args[1] if len(args) > 1 and args[1] is not None else kwargs.get("dim")
        shape = args[0].shape
        axes = [dims] if isinstance(dims, int) else dims
        count = math.prod(shape) if not dims else math.prod(shape[axis] for axis in axes)
        return max(count - 1, 1)
    if name == "cumsum":
        return max(int(args[0].shape[args[1]]) - 1, 1)
    if name == "embedding_dense_backward":
        # A scatter-add: each row's gradient sums the updates of the places that read it, at most every
        # index's (`_terms_count`'s scatter-add).
        return int(args[1].numel())
    if name in ("index_add", "scatter_add"):
        return int(args[2].numel())
    return 1
