"""The FP8 products' contract where no kernel reaches parity (`fp8_below.assert_fp8_below`,
docs/design/testing.md): an error below a named arm's, judged as check 1 judges, its ratios recorded."""

import json
import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import fp8_below
import jax.numpy as jnp
import numpy as np
import pytest
from fp8_below import assert_fp8_below

from dew.parity import PARITY, GateFailure, Inconclusive

ROWS, DEPTH, WIDTH = 32, 128, 16


def _codes(seed: int, dtype=jnp.float8_e4m3fn) -> tuple:
    """Rows and a matrix as `dtype`'s values, and the product's column scales."""
    rng = np.random.default_rng(seed)
    rows, matrix = (jnp.asarray(rng.normal(size=(count, DEPTH)), dtype) for count in (ROWS, WIDTH))
    return rows, matrix, jnp.asarray(rng.uniform(0.5, 2.0, size=WIDTH), jnp.float32)


def _whole(seed: int, most: int = 15) -> tuple:
    """FP8 codes of whole numbers up to `most` in size under power-of-two scales: a product fp32 sums
    exactly, and bfloat16's eight bits do not hold at 15 and do at 1."""
    rng = np.random.default_rng(seed)
    rows, matrix = (jnp.asarray(rng.integers(-most, most + 1, size=(count, DEPTH)), jnp.float8_e4m3fn)
                    for count in (ROWS, WIDTH))
    return rows, matrix, jnp.asarray(2.0 ** rng.integers(-1, 2, size=WIDTH), jnp.float32)


def _cancelling(seed: int) -> tuple:
    """`_codes`, its first row and its matrix laid out so that row's products cancel in pairs exactly (3 by
    5m beside -15 by m, m a power of two), under the scales `_codes` draws."""
    rng = np.random.default_rng([seed, 1])
    rows, _, scales = _codes(seed)
    powers = 2.0 ** rng.integers(-2, 3, size=(WIDTH, DEPTH // 2))
    matrix = np.stack([5 * powers, powers], axis=-1).reshape(WIDTH, DEPTH)
    rows = np.asarray(rows, np.float32)
    rows[0] = np.tile([3.0, -15.0], DEPTH // 2)
    return jnp.asarray(rows, jnp.float8_e4m3fn), jnp.asarray(matrix, jnp.float8_e4m3fn), scales


def _dequantized(rows, matrix, scales):
    """The scaled product of the codes' values, summed in the scales' type: fp32 as an arm, float64 as the
    truth."""
    wide = jnp.promote_types(scales.dtype, jnp.float32)
    return (rows.astype(wide) @ matrix.astype(wide).T) * scales


def _narrow(rows, matrix, scales):
    """The same product kept to bfloat16's eight bits, a kernel further from float64 than fp32 sums."""
    return _dequantized(rows, matrix, scales).astype(jnp.bfloat16).astype(jnp.float32)


def _off(by: float):
    """The dequantized product off by the relative `by`: past fp32's rounding, so it misses parity."""
    return lambda rows, matrix, scales: _dequantized(rows, matrix, scales) * jnp.float32(1 + by)


def _judged(dew_fn, named_fn, draw=_codes, codes=lambda rows, matrix, scales: (rows, matrix)):
    assert_fp8_below(dew_fn, named_fn, draw, "product", seeds=8, named="the narrow kernel", codes=codes,
                     reference_fn=_dequantized, truth_fn=_dequantized)


def holding() -> dict:
    """A form that misses parity and is below the named arm, judged: what the gate recorded of it."""
    _judged(_off(2.0 ** -14), _narrow)
    return fp8_below.BELOW[-1]


def test_a_form_that_misses_parity_but_is_below_the_named_arm_holds_and_one_further_fails():
    """A form past check 1's parity against the dequantized reference whose error is below the named arm's on
    the same draws holds the contract, both measured ratios recorded; a form further from float64 than the
    named arm fails it outright, decided rather than left inconclusive."""
    recorded = holding()
    assert recorded["ratio"] <= recorded["interval"][1] < 1 and recorded["reference_ratio"] > PARITY
    with pytest.raises(GateFailure) as failed:
        _judged(lambda *operands: _narrow(*operands) * jnp.float32(1 + 2.0 ** -6), _narrow)
    assert not isinstance(failed.value, Inconclusive)
    assert "pooled" in [finding.rule for finding in failed.value.findings]


def test_a_named_arm_that_is_exact_leaves_nothing_to_be_below():
    """Against a named arm that equals float64 on every seed nothing is below: a form off float64 fails as
    under check 1, and an exact form fails too, a form at parity holding check 1 itself. An exact form is
    below a named arm that is not."""
    _judged(_dequantized, _narrow, _whole)
    for form, rule in ((_narrow, "exact reference"), (_dequantized, "no measure")):
        with pytest.raises(GateFailure) as failed:
            _judged(form, _dequantized, _whole)
        assert rule in [finding.rule for finding in failed.value.findings]


def test_a_first_stage_of_exact_seeds_draws_the_second_before_it_is_the_verdict():
    """A site whose first seeds every arm holds exactly has nothing to be below yet, a finding as any other:
    the second stage is drawn, and a form below the named arm over its seeds holds."""
    _judged(_dequantized, _narrow, lambda seed: _whole(seed, most=1) if seed < 8 else _codes(seed))


def test_a_row_that_cancels_is_sized_by_its_terms():
    """A row whose products cancel exactly in float64 is judged at the size of what it sums, as check 1
    judges it: a form that folds the scales into the matrix leaves a rounding's residue on it, the named arm
    by its exact pairs none, and the form, far below that arm everywhere, holds; held to the named arm's
    residue on that row, it could not."""
    def folded(rows, matrix, scales):
        return rows.astype(jnp.float32) @ (matrix.astype(jnp.float32) * scales[:, None]).T

    residue, exact = (np.asarray(arm(*_cancelling(0)))[0] for arm in (folded, _narrow))
    assert np.any(residue != 0) and not np.any(exact)
    _judged(folded, _narrow, _cancelling)


@pytest.mark.parametrize("draw", [
    lambda seed: _codes(seed, jnp.float32),
    lambda seed: (*(np.zeros((count, DEPTH), jnp.float4_e2m1fn) for count in (ROWS, WIDTH)),
                  np.ones(WIDTH, jnp.float8_e4m3fn)),
    lambda seed: (*(np.ones((count, DEPTH), jnp.float8_e8m0fnu) for count in (ROWS, WIDTH)),
                  np.ones(WIDTH, np.float32)),
], ids=["float32", "fp4 codes under fp8 scales", "e8m0 scales as codes"])
def test_a_site_whose_multiplicands_are_not_fp8_codes_is_refused_the_contract(draw):
    """Multiplicands of float32, of FP4 codes beside FP8 block scales (NVFP4's layout), or of E8M0 scales
    are no FP8 product: the helper refuses them before any arm runs, so the site holds check 1's parity."""
    ran = []

    def counted(*operands):
        ran.append(operands)
        return _dequantized(*operands)

    with pytest.raises(TypeError):
        _judged(counted, _narrow, draw)
    assert not ran


def test_a_judged_fp8_site_reports_its_ratios_as_a_junit_property(tmp_path):
    """A passing test that judged an FP8 site reports what the gate recorded of it (`holding`) as the JUnit
    property `fp8_below`: the leaf, the named arm, the seeds and both ratios."""
    site = tmp_path / "test_site.py"
    site.write_text("from test_fp8_below import holding\n\n\ndef test_held():\n    holding()\n")
    report = tmp_path / "report.xml"
    tests = Path(__file__).resolve().parent
    path = os.pathsep.join([str(tests), os.environ.get("PYTHONPATH", "")])
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "conftest", "-p", "no:cacheprovider",
                    f"--junitxml={report}", str(site)], cwd=tmp_path, env={**os.environ, "PYTHONPATH": path})
    [case] = ElementTree.parse(report).iter("testcase")
    [reported] = [json.loads(node.get("value")) for node in case.iter("property")
                  if node.get("name") == "fp8_below"]
    assert [child.tag for child in case if child.tag != "properties"] == [] and reported == [holding()]
