"""What a GEMM epilogue's derivative reads, from its programs (`gemm`'s hooked backward): an accumulator read
through one rounding alone and whether a stored output holds that rounding exactly (a kept output standing in
for its accumulator, row 1376), and the widths its cotangents arrive in (a joined launch's operand dtype)."""

import jax
import jax.numpy as jnp

from dew.nn.programs import called


def lossless_cast(source, target) -> bool:
    """Whether casting `source` values to `target` keeps every one exactly: the same dtype, or two floating
    dtypes whose `target` holds every normal and subnormal of `source` (as many mantissa bits, as wide an
    exponent range, as small a least subnormal). Never an integer's (int32 to fp32 rounds 2**24 + 1), and
    no promotion is asked (float8 against fp32 has none, `jnp.promote_types` raising; #4050's review)."""
    source, target = jnp.dtype(source), jnp.dtype(target)
    if source == target:
        return True
    if not (jnp.issubdtype(source, jnp.floating) and jnp.issubdtype(target, jnp.floating)):
        return False
    have, want = jnp.finfo(target), jnp.finfo(source)
    return (have.nmant >= want.nmant and have.maxexp >= want.maxexp and have.minexp <= want.minexp
            and have.minexp - have.nmant <= want.minexp - want.nmant)


def read_rounded_alone(jaxpr, var, width) -> bool:
    """Whether `jaxpr` reads `var` only through conversions to `width` (and calls whose programs do,
    `called`): every value it computes of `var` then a function of that rounding."""
    if var in jaxpr.outvars:
        return False
    for eqn in jaxpr.eqns:
        if var not in eqn.invars:
            continue
        if eqn.primitive is jax.lax.convert_element_type_p and eqn.outvars[0].aval.dtype == width:
            continue
        body = called(eqn)
        if body is None or not all(read_rounded_alone(body, inner, width)
                                   for inner, outer in zip(body.invars, eqn.invars, strict=True)
                                   if outer is var):
            return False
    return True


def rounding(jaxpr, var):
    """The value of `jaxpr` that is `var` rounded, where `var`'s one reader is that conversion, or a call
    whose program's is (`dew.nn.programs.called`: an epilogue its fused consumer's derivative declares,
    `fused_gemm._declared`), its output then the call's; None otherwise."""
    reading = [eqn for eqn in jaxpr.eqns if var in eqn.invars]
    if len(reading) != 1:
        return None
    (eqn,) = reading
    if eqn.primitive is jax.lax.convert_element_type_p:
        return eqn.outvars[0]
    body = called(eqn)
    if body is None or eqn.invars.count(var) != 1:
        return None
    inner = rounding(body, body.invars[eqn.invars.index(var)])
    return eqn.outvars[body.outvars.index(inner)] if inner in body.outvars else None


def passes_cotangent(vjp: tuple, at: int, dtypes: tuple, narrow) -> bool:
    """Whether the epilogue's VJP (`_epilogue_vjp`'s program) hands its accumulator `at` a cotangent whose
    values `narrow` holds exactly: one output's passed through unchanged (a bias's, a cast's) in a dtype
    `narrow` holds, or a value rounded to such a dtype and only widened since (a nonlinear epilogue's
    cotangent of a bf16 rounding of its accumulator: the rounding's transpose rounds it to bf16, its
    widening back holds it), read from the VJP's own program, its result traced back through conversions
    alone."""
    closed, cotangents = vjp
    var = closed.jaxpr.outvars[at]
    producers = {out: equation for equation in closed.jaxpr.eqns for out in equation.outvars}
    narrow = jnp.dtype(narrow)

    def held(dtype) -> bool:
        return lossless_cast(dtype, narrow)

    widened = True  # every conversion between `var` and the result widens, exactly
    while var in producers and producers[var].primitive is jax.lax.convert_element_type_p:
        if widened and held(var.aval.dtype):
            return True
        source = producers[var].invars[0]
        widened = widened and lossless_cast(source.aval.dtype, var.aval.dtype)
        var = source
    if widened and held(var.aval.dtype):
        return True
    if var not in cotangents:
        return False
    return held(dtypes[cotangents.index(var)])
