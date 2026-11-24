"""A signed sum of contractions of products and quotients of float arrays, each formed apart from its
binades (`scaled_contraction_p`, rows 2654 and 12388).

`contract(form, *arrays)` is `sum_t sign_t einsum(groups_t -> out)` (`Form.terms`): a term's group the
product of its operands, each to the power `+1` or `-1`, each operand over its own subscript
(`Form.subscripts`). Each operand is split into a significand in `[1, 2)` and an integer exponent from its
bits (`_split`, so a subnormal keeps its value where a process flushes subnormal arithmetic); each group is
held at its exponent's largest over the letters it sums (`Form.out` lacks them), so every operand a
contraction reads lies within `[2^-n, 2^n]` of one, `n` its factors' count; and the terms are joined at
their exponents' largest before the one power of two that lands the result (`power2`). So a sum of values
near the format's largest, a quotient of `2^126` by a total of `2^62` moved by `2^62`, or a product rule's
`+2^200` and `-2^200` parts, form no float past the format beside the factor that brings it back.

Its derivative at every order, forward and reverse, is the primitive again: along an operand to `+1` the
same term with its tangent in its place; along one to `-1`, `-dx / x^2`, the term with the tangent to `+1`
and the operand to `-1` twice (`_jvp`, every term of the product rule in one sum, landed once); and the
transpose along a linear operand the sum of its terms' other groups contracted with the cotangent onto its
letters (`_transpose`). No order takes a quotient, a power or a sum outside it, so none forms `total^-2`
(2^-200 at a mass of 2^100, zero) or a cotangent's sum past the format before the factor that brings it
back (the reference's mean, `attention.held_mean`).

A term's group whose values span more binades than a band (`_band`) is contracted a band of each group at a
time, each band pair at its own power of two, and every term's contractions joined as double floats at their
own powers from the largest (`windows._joined`, `_banded`): so terms that cancel keep a part any number of
binades beneath them (`2^127 - 2^127 + 2^-30` is `2^-30`, where one exponent for the group flushed it, row
12452). Within a band a contraction rounds as an fp32 sum does (`2^127 + 2^100 - 2^127`, one band, is 0).
"""

from __future__ import annotations

import dataclasses
import functools
import string
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax._src import api_util
from jax._src.interpreters import partial_eval as pe
from jax.extend import core as jcore
from jax.interpreters import ad, batching, mlir

from dew.algebra import windows
from dew.nn.float_bits import binade_of, float_bits
from dew.nn.precision import at_least_fp32, exact_float32, power2

Factor = tuple[int, int]
"""An operand's index and its power, `+1` or `-1`; or `0`, an int32 operand of exponents its group's
product is raised by (`2^e`), no value of its own: a divisor carried apart from its binades (`_transpose`)."""

Term = tuple[int, tuple[tuple[Factor, ...], ...]]
"""A term's sign and its groups, each a product of its factors."""

PRECISION = exact_float32(None)
"""The einsums' in float32: fp32 products at fp32 (`exact_float32`), as the reference's own contractions."""

LEAST = jnp.iinfo(jnp.int32).min // 4
"""An exponent beneath every float's, a zero's in a largest (`_held`, `_value`)."""


@dataclasses.dataclass(frozen=True)
class Form:
    """A sum's shape: each operand's subscript, each term (`Term`, its factors naming operands) and the
    result's subscript `out`; with `divisor` terms, the sum over them (over the letters `divided`, a subset
    of `out`'s) divides it, both formed apart from their binades and divided before they land, so a quotient
    whose numerator and divisor pass the format lands within it (a zero divisor read as one). Each of
    `narrowed`'s `(term, operand, dtype)` lands that term's pull onto that operand apart, rounded to `dtype`
    before it joins the rest (a product of weights narrowed to their values' dtype, whose cotangent JAX's own
    transpose rounds there). Every letter has one extent wherever it appears, and none twice in one
    subscript."""

    subscripts: tuple[str, ...]
    terms: tuple[Term, ...]
    out: str
    divisor: tuple[Term, ...] = ()
    divided: str = ""
    narrowed: tuple[tuple[int, int, str], ...] = ()

    def letters(self, group: tuple[Factor, ...]) -> str:
        """`group`'s letters, each once, in the order its factors' subscripts name them."""
        return "".join(dict.fromkeys("".join(self.subscripts[index] for index, _ in group)))


def _laid(array: jax.Array, subscript: str, letters: str) -> jax.Array:
    """`array` over `subscript` laid over `letters`, a letter it lacks of extent one."""
    ordered = [letter for letter in letters if letter in subscript]
    moved = jnp.transpose(array, [subscript.index(letter) for letter in ordered])
    return moved.reshape([array.shape[subscript.index(letter)] if letter in subscript else 1
                          for letter in letters])


def _split(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`x` as its significand in `[1, 2)` (signed; zero a zero's, a non-finite `x` itself) and its exponent
    `floor(log2 |x|)` (int32; zero's and a non-finite's 0), read from its bits (`float_bits`): a subnormal's
    by its leading bit, where a process flushing subnormal arithmetic reads none of it."""
    limits = jnp.finfo(x.dtype)
    sign, biased, fraction, normalized, exponent = float_bits(x)
    unit = (limits.maxexp - 1) << limits.nmant
    significand = jax.lax.bitcast_convert_type(sign | unit | normalized, x.dtype)
    zero = (biased == 0) & (fraction == 0)
    finite = biased < (2 * limits.maxexp - 1)
    significand = jnp.where(zero, jnp.zeros((), x.dtype), jnp.where(finite, significand, x))
    return significand, jnp.where(zero | ~finite, 0, exponent).astype(jnp.int32)


def _held(form: Form, group: tuple[Factor, ...], arrays: Sequence[jax.Array], dtype):
    """A group's product as its significand within `[2^-n, 2^n]` of one (`n` its factors' count), its
    exponent's offset beneath its largest over the letters it sums (at most 0; 0 at a zero or a non-finite),
    and that largest over the letters it keeps (`out`'s, the group's order), with its letters."""
    letters = form.letters(group)
    significand, exponent = jnp.ones((), dtype), jnp.int32(0)
    for index, power in group:
        if power == 0:
            exponent = exponent + _laid(arrays[index].astype(jnp.int32), form.subscripts[index], letters)
            continue
        part, binade = _split(_laid(arrays[index].astype(dtype), form.subscripts[index], letters))
        significand = significand * part if power > 0 else significand / part
        exponent = exponent + binade if power > 0 else exponent - binade
    significand = jnp.broadcast_to(significand, jnp.broadcast_shapes(significand.shape, jnp.shape(exponent)))
    summed = tuple(axis for axis, letter in enumerate(letters) if letter not in form.out)
    live = (significand != 0) & jnp.isfinite(significand)
    largest = jnp.max(jnp.where(live, exponent, LEAST), axis=summed, keepdims=True, initial=LEAST)
    largest = jnp.where(largest == LEAST, 0, largest)
    return letters, significand, jnp.where(live, exponent - largest, 0), jnp.squeeze(largest, summed)


def _band(groups: int) -> int:
    """The binades a group's band spans in a term of `groups` groups: each group's banded significand at
    least `2^-band` its band's largest times `2^-n`, so the term's product of one band of each stays a
    normal fp32 (`groups` bands and their factors' `2^-n` beneath `2^-120`)."""
    return 120 // groups - 4


def _dtype(arrays) -> jnp.dtype:
    """The sum's format: its float operands', at least float32 (an exponent operand's none)."""
    return at_least_fp32(jnp.result_type(*(array.dtype for array in arrays
                                            if jnp.issubdtype(array.dtype, jnp.floating))))


def _sum(arrays, form: Form, terms: tuple[Term, ...], out: str, dtype):
    """`terms`' signed sum over `out` unlanded: its significands' sum and the exponent it lies at. Each
    term's groups held apart from their binades (`_held`) and contracted at their exponents' largest, the
    terms joined as double floats at their own powers; where a group's values span more than a band
    (`_band`) every term contracted a band pair at a time and joined so (`_banded`), so a part beneath terms
    that cancel survives them. In float32 (a float64 sum's one exponent spans the float32 operands it takes;
    its terms joined at their largest)."""
    precision = PRECISION if dtype == jnp.float32 else jax.lax.Precision.HIGHEST
    shaped = Form(form.subscripts, terms, out)
    held = [[_held(shaped, group, arrays, dtype) for group in groups] for _, groups in terms]
    scales = [sum((_laid(largest, "".join(letter for letter in letters if letter in out), out)
                   for letters, _, _, largest in groups), jnp.zeros((1,) * len(out), jnp.int32))
              for groups in held]

    def contracted(groups, significands):
        return jnp.einsum(",".join(letters for letters, *_ in groups) + "->" + out, *significands,
                          precision=precision, preferred_element_type=dtype)

    parts = []
    for (sign, _), groups, scale in zip(terms, held, scales, strict=True):
        summed = contracted(groups, [power2(significand, offset) for _, significand, offset, _ in groups])
        parts.append((sign * summed, jnp.where(summed != 0, scale, LEAST)))
    joined = functools.reduce(jnp.maximum, [scale for _, scale in parts])
    joined = jnp.where(joined == LEAST, 0, joined)
    landed = sum((power2(summed, scale - joined) for summed, scale in parts), jnp.zeros((), dtype))
    if dtype != jnp.float32:
        return landed, joined
    # The terms joined as double floats at their own powers (`windows._joined`), so terms that cancel keep a
    # part beneath them; a non-finite sum IEEE's, as landed.
    high, low, exponent = functools.reduce(windows._joined, [
        (summed, jnp.zeros_like(summed), jnp.where(summed != 0, scale, 0)) for summed, scale in parts])
    special = ~jnp.isfinite(landed)
    landed, joined = jnp.where(special, landed, high + low), jnp.where(special, joined, exponent)
    # A band past the first in any group, and every value finite: the sum again a band pair at a time.
    spans = [functools.reduce(jnp.maximum, [jnp.max(-offset, initial=0) for _, _, offset, _ in groups])
             // _band(len(groups)) for groups in held]
    finite = jnp.all(jnp.asarray([jnp.all(jnp.isfinite(significand))
                                  for groups in held for _, significand, _, _ in groups]))
    return jax.lax.cond(finite & (functools.reduce(jnp.maximum, spans) > 0),
                        lambda: _banded(terms, held, scales, contracted),
                        lambda: (landed, joined))


def _banded(terms: tuple[Term, ...], held: list, scales: list, contracted) -> tuple[jax.Array, jax.Array]:
    """`_sum`'s terms contracted a band of each group at a time (`_band`: a value whose offset beneath its
    group's largest lies in `[-(i + 1) band, -i band)` in band `i`, raised `2^(i band)`), each band pair's sum
    at its own power of two joined into the total as double floats (`windows._joined`), the terms' pairs in
    one `lax.while_loop` from each term's first: its program one contraction of each term whatever the
    bands, and a part beneath a pair that cancels kept to the double floats' reach."""
    bands = [[(-offset) // _band(len(groups)) for _, _, offset, _ in groups] for groups in held]
    counts = [[jnp.max(band, initial=0) + 1 for band in term] for term in bands]
    pairs = jnp.stack([functools.reduce(jnp.multiply, term) for term in counts])
    zero = scales[0] - scales[0]

    def pair(t: int, at):
        """Term `t`'s band pair `at` (its groups' bands in mixed radix of their counts), its double float."""
        sign, _ = terms[t]
        significands, raised, rest = [], 0, at
        for (_, significand, offset, _), band, count in zip(held[t], bands[t], counts[t], strict=True):
            index = rest % count
            rest = rest // count
            width = _band(len(held[t]))
            significands.append(jnp.where(band == index, power2(significand, offset + index * width), 0))
            raised = raised + index * width
        summed = sign * contracted(held[t], significands)
        return summed, jnp.zeros_like(summed), jnp.where(summed != 0, scales[t] - raised, 0)

    def step(state):
        t, at, total = state
        total = jax.lax.switch(t, [lambda at, total, t=t: windows._joined(total, pair(t, at))
                                   for t in range(len(terms))], at, total)
        moved = at + 1 >= pairs[t]
        return jnp.where(moved, t + 1, t), jnp.where(moved, 0, at + 1), total

    start = (jnp.int32(0), jnp.int32(0), (zero.astype(jnp.float32), zero.astype(jnp.float32), zero))
    _, _, (high, low, exponent) = jax.lax.while_loop(lambda state: state[0] < len(terms), step, start)
    return high + low, exponent


def _value(*arrays: jax.Array, form: Form) -> jax.Array:
    """The sum (the module's law) in its operands' format, at least float32 (`_dtype`), its numerator over
    its divisor (`Form.divisor`) divided before one power of two lands it (`power2`)."""
    extents = _extents(arrays, form)
    dtype = _dtype(arrays)
    if not form.terms:
        return jnp.zeros([extents[letter] for letter in form.out], dtype)
    total, joined = _sum(arrays, form, form.terms, form.out, dtype)
    if form.divisor:
        below, exponent = _sum(arrays, form, form.divisor, form.divided, dtype)
        none = below == 0
        total = total / _laid(jnp.where(none, 1, below), form.divided, form.out)
        joined = joined - _laid(jnp.where(none, 0, exponent), form.divided, form.out)
    return power2(total, joined)


def _extents(arrays, form: Form) -> dict[str, int]:
    """Each letter's extent, one wherever it appears; a subscript naming a letter twice, or of another rank
    than its operand's, refused."""
    extents: dict[str, int] = {}
    for subscript, array in zip(form.subscripts, arrays, strict=True):
        if len(set(subscript)) != len(subscript) or len(subscript) != len(array.shape):
            raise ValueError(f"a scaled contraction's subscript {subscript!r} of an operand {array.shape}")
        for letter, extent in zip(subscript, array.shape, strict=True):
            if extents.setdefault(letter, extent) != extent:
                raise ValueError(f"letter {letter!r} of extents {extents[letter]} and {extent}")
    if not set(form.divided) <= set(form.out):
        raise ValueError(f"a divisor over {form.divided!r} beyond the result's {form.out!r}")
    return extents


scaled_contraction_p = jcore.Primitive("scaled_contraction")


@scaled_contraction_p.def_abstract_eval
def _abstract(*arrays, form: Form):
    """The sum's type as JAX types its value (`_value`): its shape and format, and its operands' shardings,
    weak types and manual axes as JAX's own rules carry them."""
    def value(*operands):
        return _value(*operands, form=form)

    (typed,) = pe.abstract_eval_fun(value, *arrays, debug_info=api_util.debug_info(
        "scaled_contraction", value, arrays, {}))
    return typed


scaled_contraction_p.def_impl(lambda *arrays, form: _value(*arrays, form=form))
mlir.register_lowering(scaled_contraction_p, mlir.lower_fun(_value, multiple_results=False))


def contract(form: Form, *arrays: jax.Array) -> jax.Array:
    """`form`'s sum of `arrays` (the module's law), at every order the primitive again."""
    return scaled_contraction_p.bind(*arrays, form=form)


def _moved(terms: tuple[Term, ...], moving: dict, at: dict[int, int]) -> list[tuple[Term, int, int]]:
    """`terms`' product rule along the moving operands, each moved term with the term and operand it moved:
    along one to `+1` its tangent in its place; along one to `-1`, `-dx / x^2`, its tangent to `+1` and the
    operand to `-1` twice."""
    moved = []
    for t, (sign, groups) in enumerate(terms):
        for g, group in enumerate(groups):
            for f, (index, power) in enumerate(group):
                if index not in moving:
                    continue
                factors = (((at[index], 1),) if power > 0 else ((at[index], 1), (index, -1), (index, -1)))
                changed = (*group[:f], *factors, *group[f + 1:])
                term = (sign if power > 0 else -sign, (*groups[:g], changed, *groups[g + 1:]))
                moved.append((term, t, index))
    return moved


def _held_out(arrays) -> list[jax.Array]:
    """`arrays` held out of the graph: a rule's sum read for its value alone (a zero divisor, its exponent),
    whose double floats' splits (`windows._joined`) take no derivative of their own."""
    return [jax.lax.stop_gradient(array) for array in arrays]


def _jvp(primals, tangents, *, form: Form):
    """The sum's tangent, one sum of the product rule's terms (`_moved`, `ad.Zero` a still operand's) landed
    once; a quotient's `(dN - out dD) / D` over its own divisor, the result `out` an operand of the
    divisor's moved terms, so its numerator's and divisor's parts cancel within the one sum."""
    out = contract(form, *primals)
    moving = {index: tangent for index, tangent in enumerate(tangents)
              if type(tangent) is not ad.Zero and jnp.issubdtype(primals[index].dtype, jnp.floating)}
    if not moving:
        return out, ad.Zero(jax.typeof(out).to_tangent_aval())
    at = {index: len(primals) + place for place, index in enumerate(moving)}
    numerator = _moved(form.terms, moving, at)
    terms = [term for term, _, _ in numerator]
    marks = {(t, index): dtype for t, index, dtype in form.narrowed}
    narrowed = tuple((j, at[index], marks[t, index]) for j, (_, t, index) in enumerate(numerator)
                     if (t, index) in marks)
    operands = [*primals, *moving.values()]
    subscripts = [*form.subscripts, *(form.subscripts[index] for index in moving)]
    if form.divisor:
        below = _moved(form.divisor, moving, at)
        if below:
            # A zero divisor read as one is a constant: its moved terms none there.
            total, _ = _sum(_held_out(primals), form, form.divisor, form.divided, _dtype(primals))
            none = _laid(total == 0, form.divided, form.out)
            operands.append(jnp.where(none, jnp.zeros((), out.dtype), out))
            subscripts.append(form.out)
            terms += [(-sign, (*groups, ((len(operands) - 1, 1),))) for (sign, groups), _, _ in below]
    if not terms:
        return out, ad.Zero(jax.typeof(out).to_tangent_aval())
    tangent = Form(tuple(subscripts), tuple(terms), form.out, form.divisor, form.divided, narrowed)
    return out, contract(tangent, *operands)


def _transpose(cotangent, *arrays, form: Form):
    """Each linear operand's cotangent (an `ad.UndefinedPrimal`, to `+1` in each term that reads it, as
    `_jvp` places it): the sum over those terms of their other factors contracted with the cotangent over
    `out`, onto its letters; a letter none of them reads taken by a unit operand along it."""
    out: list[jax.Array | None] = [None] * len(arrays)
    if type(cotangent) is ad.Zero:
        return out
    known = [index for index, array in enumerate(arrays) if not ad.is_undefined_primal(array)]
    place = {index: at for at, index in enumerate(known)}
    below: tuple[Factor, ...] = ()
    shared = [arrays[index] for index in known]
    shared_subscripts = [form.subscripts[index] for index in known]
    if form.divisor:
        if any(index not in place for _, groups in form.divisor for group in groups for index, _ in group):
            raise NotImplementedError("a scaled contraction is linear in no divisor's operand")
        divisor = tuple((sign, tuple(tuple((place[index], power) for index, power in group)
                                     for group in groups)) for sign, groups in form.divisor)
        # The divisor carried apart from its binades, as the forward keeps it unlanded: its significand at
        # its exponent `e` held out of the graph (a contraction of the primals again, its derivatives with it)
        # and `2^-e` an exponent operand, so a divisor past the format divides finite (`y + z` at 2^127).
        unlanded, joined = _sum(_held_out(shared), Form(tuple(shared_subscripts), divisor, form.divided),
                                divisor, form.divided, _dtype(shared))
        exponent = jax.lax.stop_gradient(jnp.where(unlanded == 0, 0, joined + binade_of(unlanded)))
        shared.append(-exponent)
        shared_subscripts.append(form.divided)
        lowering = len(shared) - 1
        lowered = tuple((sign, (*groups, ((lowering, 0),))) for sign, groups in divisor)
        significand = contract(Form(tuple(shared_subscripts), lowered, form.divided), *shared)
        shared.append(jnp.where(significand == 0, jnp.ones((), significand.dtype), significand))
        shared_subscripts.append(form.divided)
        below = ((len(shared) - 1, -1), (lowering, 0))
    for linear, array in enumerate(arrays):
        if not ad.is_undefined_primal(array):
            continue
        subscript = form.subscripts[linear]
        operands, subscripts = list(shared), list(shared_subscripts)
        operands.append(cotangent)
        subscripts.append(form.out)
        pulled = len(operands) - 1
        terms = []
        # A mark of a dtype no narrower than the sum's (an fp32 weight's identity cast) rounds nothing: its
        # term stays in the one sum, its cancellation with the rest kept.
        width = jnp.dtype(_dtype(operands)).itemsize
        marks = {t: dtype for t, index, dtype in form.narrowed
                 if index == linear and jnp.dtype(dtype).itemsize < width}
        rounded: dict[str, list[Term]] = {}
        for t, (sign, groups) in enumerate(form.terms):
            found = [(g, f) for g, group in enumerate(groups) for f, (index, _) in enumerate(group)
                     if index == linear]
            if not found:
                continue
            ((g, f),) = found
            if groups[g][f][1] < 0:
                raise NotImplementedError("a scaled contraction is linear in no divisor")
            rest = tuple(tuple((place[index], power) for index, power in group if index != linear)
                         for group in (*groups[:g], tuple(x for i, x in enumerate(groups[g]) if i != f),
                                       *groups[g + 1:]))
            read = set(form.out).union(*(form.subscripts[index] for group in groups for index, _ in group
                                         if index != linear))
            unread = "".join(letter for letter in subscript if letter not in read)
            extra: tuple[Factor, ...] = ()
            if unread:
                operands.append(jnp.ones([array.aval.shape[subscript.index(letter)] for letter in unread],
                                         array.aval.dtype))
                subscripts.append(unread)
                extra = ((len(operands) - 1, 1),)
            pulled_term = (sign, (*(group for group in rest if group), ((pulled, 1), *below, *extra)))
            (rounded.setdefault(marks[t], []) if t in marks else terms).append(pulled_term)
        if not terms and not rounded:
            # An operand no term reads (a sum's operand its own transpose carried along): no cotangent.
            continue
        landed = (contract(Form(tuple(subscripts), tuple(terms), subscript), *operands) if terms
                  else jnp.zeros(array.aval.shape, _dtype(operands)))
        for dtype, apart in rounded.items():
            alone = contract(Form(tuple(subscripts), tuple(apart), subscript), *operands)
            landed = landed + alone.astype(dtype).astype(alone.dtype)
        out[linear] = landed.astype(array.aval.dtype)
    return out


ad.primitive_jvps[scaled_contraction_p] = _jvp
ad.primitive_transposes[scaled_contraction_p] = _transpose


def _batched(arrays, dims, *, form: Form):
    """The sum of a batch: a fresh letter on each batched operand (its batch axis moved first) and on the
    result; an unbatched operand shared along it, and a term that reads no batched operand taking a unit
    operand along it, so each term spans the result's letters."""
    used = set(form.out).union(*form.subscripts)
    letter = next((each for each in string.ascii_letters if each not in used), None)
    if letter is None:
        raise ValueError("a scaled contraction batched past its letters")
    size = next(array.shape[dim] for array, dim in zip(arrays, dims, strict=True) if dim is not None)
    operands, subscripts = [], []
    for array, dim, subscript in zip(arrays, dims, form.subscripts, strict=True):
        operands.append(array if dim is None else jnp.moveaxis(array, dim, 0))
        subscripts.append(subscript if dim is None else letter + subscript)
    batched = {index for index, dim in enumerate(dims) if dim is not None}

    def spanned(terms: tuple[Term, ...]) -> tuple[Term, ...]:
        laid = []
        for sign, groups in terms:
            if not any(index in batched for group in groups for index, _ in group):
                if len(operands) == len(arrays):
                    operands.append(jnp.ones((size,), _dtype(arrays)))
                    subscripts.append(letter)
                groups = (*groups, ((len(arrays), 1),))
            laid.append((sign, groups))
        return tuple(laid)

    terms = spanned(form.terms)
    divisor = spanned(form.divisor) if form.divisor else ()
    divided = letter + form.divided if form.divisor else ""
    return contract(Form(tuple(subscripts), terms, letter + form.out, divisor, divided, form.narrowed),
                    *operands), 0


batching.primitive_batchers[scaled_contraction_p] = _batched
