"""The FP8 products' contract where no FP8 kernel reaches check 1's parity (docs/design/testing.md, "FP8
products"): Dew's error from the float64 truth below the arm the spec names, held by check 1 with its pooled
ratio's rule changed (`assert_fp8_below`), and what each site measured (`BELOW`)."""

import functools

import jax.numpy as jnp
from check1_judge import _logged, _narrow, _with_mode
from reference_error import (
    _arms_with,
    _drawn,
    _enough_seeds,
    _mode_of,
    _refuse_numpy_arms,
    _site_terms,
    _truth_program,
)

from dew.parity import GateFailure, Inconclusive, _below_one, _Draws, _judged, _pooled, _staged, _verdict

BELOW: list[dict] = []
"""What each FP8 site this test judged measured (`assert_fp8_below`), the JUnit property `fp8_below`
(tests/conftest.py) of a test that passes as of one that fails: a leaf, the arm it is held below, Dew's
pooled ratio to that arm and its interval, Dew's ratio to the dequantized reference (the measured miss of
check 1's parity, which goes to the owner) and the seeds they were measured over."""


def _fp8_codes(operand) -> bool:
    """Whether `operand` holds an FP8 format's codes: floats of eight bits with a significand (E4M3, E5M2
    and their kin), which a block scale's E8M0 has none of, as JAX's dtypes read them."""
    dtype = jnp.result_type(operand)
    return jnp.issubdtype(dtype, jnp.floating) and jnp.finfo(dtype).bits == 8 and jnp.finfo(dtype).nmant > 0


def assert_fp8_below(dew_fn, named_fn, draw, label: str, *, seeds: int, named: str, codes, reference_fn,
                     truth_fn) -> None:
    """The contract of an FP8 form whose scales are a whole operand's, where no FP8 kernel reaches check
    1's parity (docs/design/testing.md, "FP8 products": a tensor core's running sum within a step is
    narrower than fp32, and the dequantized reference's fp32 sums of FP8 products are exact): `dew_fn`'s
    error from the float64 truth is below `named_fn`'s, the arm the spec names (`named`, cuBLASLt's FP8
    GEMM, which sums the whole depth narrow), on the same draws. A block-scaled form sums a block as
    cuBLASLt's GEMM of that block does and is held to parity with it, check 1's own (`assert_as_exact`
    with that arm as its reference), not this.

    It is check 1 against the named arm (`_staged`: its draws, its two pre-declared stages, its rows and
    values) with one rule changed, the pooled ratio's: a leaf holds where its ratio's interval lies wholly
    under one (`dew.parity._below_one`), where check 1 allows PARITY. A leaf the named arm holds exactly
    has nothing to be below: Dew off float64 there fails as under check 1, and Dew exact there fails too,
    as a form at parity holds check 1 itself. `truth_fn` is the float64 product of the operands as
    quantized, a row that cancels sized by its terms as check 1 sizes it (`_site_terms`), and
    `reference_fn` the dequantized product with fp32 sums, check 1's reference for an FP8 product: it runs
    beside the two on the same draws and Dew's ratio to it is recorded with the verdict's (`BELOW`), never
    judged.

    Only an FP8 product's site takes this contract: `codes(*inputs)` gives the product's multiplicands of a
    draw's inputs, and each must hold an FP8 format's codes (`_fp8_codes`: not a narrower format's, nor an
    FP8 scale beside them), else the site is refused and holds check 1."""
    _enough_seeds(seeds, label)
    multiplied = tuple(codes(*draw(0)))
    if not multiplied or not all(_fp8_codes(operand) for operand in multiplied):
        raise TypeError(f"{label}: the product's multiplicands are "
                        f"{[str(jnp.result_type(operand)) for operand in multiplied]}, not FP8 codes; an "
                        f"error below a named arm's is the FP8 products' contract alone "
                        f"(docs/design/testing.md), and this site holds check 1 (`assert_as_exact`)")
    _refuse_numpy_arms(draw, dew_fn, named_fn, reference_fn)
    dew_fn, named_fn, (reference_fn,) = _arms_with(dew_fn, named_fn, [reference_fn], label)
    truth_program = _truth_program(truth_fn)
    terms_fn = _site_terms(None, None, draw, truth_fn, truth_program)
    arms = [dew_fn, named_fn, reference_fn]
    # Each range's draws against the named arm and against the reference, Dew and the truth run once.
    ranges: dict[tuple[int, int], list[_Draws]] = {}

    def drawn(start, stop):
        ranges[start, stop] = _drawn(dew_fn, named_fn, draw, range(start, stop), truth_program,
                                     also=[reference_fn], terms_fn=terms_fn, label=label)
        return ranges[start, stop][0]

    held = f"{label}, against {named}"
    over, unresolved, stage = _staged(drawn, seeds, held, ratio_rule=_below_one)
    against, parity = (functools.reduce(_Draws.joined, each) for each in zip(*ranges.values(), strict=True))
    for index, name in enumerate(against.names):
        ratio, low, high = _judged(against, index)
        BELOW.append({"leaf": held + name, "named": named, "seeds": len(against.mine), "ratio": ratio,
                      "interval": [low, high],
                      "reference_ratio": _pooled(parity.mine[:, index], parity.theirs[:, index])})
    _logged(held, stage, _verdict(over, unresolved), _narrow(against), _mode_of(arms))
    if over:
        raise GateFailure(_with_mode(over + unresolved, _mode_of(arms)))
    if unresolved:
        raise Inconclusive(_with_mode(unresolved, _mode_of(arms)))
