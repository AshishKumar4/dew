"""The inputs a recorded site draws (docs/design/testing.md, "A recorded site draws the same inputs on every
host"): a recording keeps each seed's input digest (`reference_error.input_digest`), and inputs drawn again
that differ are refused (`reference_error.indexed_recording`), so its float inputs are the same bits on every
host by construction (`normals`)."""

import numpy as np

_CENTRAL = ((3.3871328727963666080e0, 1.3314166789178437745e+2, 1.9715909503065514427e+3,
             1.3731693765509461125e+4, 4.5921953931549871457e+4, 6.7265770927008700853e+4,
             3.3430575583588128105e+4, 2.5090809287301226727e+3),
            (1.0, 4.2313330701600911252e+1, 6.8718700749205790830e+2, 5.3941960214247511077e+3,
             2.1213794301586595867e+4, 3.9307895800092710610e+4, 2.8729085735721942674e+4,
             5.2264952788528545610e+3))
_NEAR = ((1.42343711074968357734e0, 4.63033784615654529590e0, 5.76949722146069140550e0,
          3.64784832476320460504e0, 1.27045825245236838258e0, 2.41780725177450611770e-1,
          2.27238449892691845833e-2, 7.74545014278341407640e-4),
         (1.0, 2.05319162663775882187e0, 1.67638483018380384940e0, 6.89767334985100004550e-1,
          1.48103976427480074590e-1, 1.51986665636164571966e-2, 5.47593808499534494600e-4,
          1.05075007164441684324e-9))
_FAR = ((6.65790464350110377720e0, 5.46378491116411436990e0, 1.78482653991729133580e0,
         2.96560571828504891230e-1, 2.65321895265761230930e-2, 1.24266094738807843860e-3,
         2.71155556874348757815e-5, 2.01033439929228813265e-7),
        (1.0, 5.99832206555887937690e-1, 1.36929880922735805310e-1, 1.48753612908506148525e-2,
         7.86869131145613259100e-4, 1.84631831751005468180e-5, 1.42151175831644588870e-7,
         2.04426310338993978564e-15))
"""Wichura's AS 241 (PPND16, Applied Statistics 37, 1988): the normal quantile's rational forms, numerator and
denominator coefficients from the constant term up, in its central region and its near and far tails."""


_FAR_END = 1.3887943864964021e-11
"""exp(-25), correctly rounded: a uniform nearer an end than it is in AS 241's far tail."""


def _ratio_of(coefficients: tuple, r: np.ndarray) -> np.ndarray:
    """`coefficients`' numerator over its denominator at `r`, each by Horner's rule in separate IEEE products
    and sums."""
    def horner(terms):
        value = np.full_like(r, terms[-1])
        for term in reversed(terms[:-1]):
            value = value * r + term
        return value
    numerator, denominator = coefficients
    return horner(numerator) / horner(denominator)


def _log(x: np.ndarray) -> np.ndarray:
    """The natural log of positive float64s by IEEE operations alone: `x` split exactly into `m 2^e` with `m`
    in [sqrt(1/2), sqrt(2)) (`np.frexp`), and `log m` as `2 atanh((m - 1) / (m + 1))`'s series to its
    twenty-third power, past float64's resolution there; no libm's log, whose last bits differ by host."""
    m, e = np.frexp(x)
    low = m < np.sqrt(0.5)
    m, e = np.where(low, 2 * m, m), np.where(low, e - 1, e).astype(np.float64)
    s = (m - 1) / (m + 1)
    s2 = s * s
    series = np.zeros_like(s)
    for power in range(23, 0, -2):
        series = series * s2 + 1.0 / power
    # log 2 in two parts, the first exact times any exponent here.
    high, low_part = 6.93147180369123816490e-01, 1.90821492927058770002e-10
    return e * high + (e * low_part + 2 * s * series)


def normals(seed, shape, dtype=np.float32) -> np.ndarray:
    """Standard normal values of `shape` in `dtype` from `seed` (anything `np.random.default_rng` takes), the
    same bits on every host: the generator's 52-bit uniforms, each odd multiple of 2^-53 (integer arithmetic,
    then one exact scaling: none 0 or 1), each
    taken to the normal quantile by AS 241's rational forms (`_CENTRAL`, `_NEAR`, `_FAR`) in IEEE sums,
    products, quotients and square roots alone (`_log` among them), rounded to `dtype` once. What a recorded
    site draws (testing.md): its inputs are checked by digest where it runs again (`input_digest`), and
    NumPy's own normals are not the same on every host (its float32 ziggurat's tail calls libm's `expf` and
    `logf`, which differ in about one value in 41,000 between the GPU hosts and armada's)."""
    raw = np.random.default_rng(seed).integers(0, 2 ** 52, size=shape, dtype=np.uint64)
    p = (raw.astype(np.float64) + 0.5) * 2.0 ** -52
    # AS 241's regions by the uniform itself: central where |p - 1/2| <= 0.425, far in a tail where
    # sqrt(-log) of its nearer end passes 5 (that end under exp(-25), `_FAR_END`).
    central = (p >= 0.075) & (p <= 0.925)
    far = (p < _FAR_END) | (p > 1 - _FAR_END)
    q = p - 0.5
    near_central = q * _ratio_of(_CENTRAL, np.where(central, 0.180625 - q * q, 0.0))
    tail = np.sqrt(-_log(np.where(central, 0.5, np.minimum(p, 1 - p))))
    tails = np.where(far, _ratio_of(_FAR, np.where(far, tail - 5, 0.0)),
                     _ratio_of(_NEAR, np.where(far, 0.0, tail - 1.6)))
    value = np.where(central, near_central, np.where(p < 0.5, -tails, tails))
    return value.astype(dtype)
