"""Residual vector quantization: each stage's nearest codeword to what the stages before left.

The codebooks are frozen tables, `[size, width]` each, a size per stage. Encoding takes, stage by
stage, the codeword nearest the residual in Euclidean distance (the first on a tie) and subtracts
it; decoding sums the chosen codewords. EnCodec's and Moshi's RVQ, as MiMo's audio tokenizer runs
it (MiMo-V2.6's `modeling_mimo_v2.py`, ResidualVectorQuantization), in fp32.
"""

import functools
from collections.abc import Sequence

import jax
import jax.numpy as jnp

from dew.nn.precision import float32_einsum


def nearest_codewords(x, codebook) -> jax.Array:
    """Each row of `x` `[..., width]`'s nearest codeword index in `codebook` `[size, width]`: the
    largest `-(|x|^2 - 2 x.e + |e|^2)`, as the reference's EuclideanCodebook.quantize forms it."""
    x = x.astype(jnp.float32)
    codebook = codebook.astype(jnp.float32)
    score = -(jnp.sum(x * x, axis=-1, keepdims=True)
              - 2 * float32_einsum('...d,vd->...v', x, codebook, precision=jax.lax.Precision.HIGHEST)
              + jnp.sum(codebook * codebook, axis=-1))
    return jnp.argmax(score, axis=-1)


def residual_quantize(x, codebooks: Sequence[jax.Array]) -> jax.Array:
    """The codes `[..., stages]` of `x` `[..., width]` under the stages' `codebooks`."""
    residual = x.astype(jnp.float32)
    codes = []
    for codebook in codebooks:
        index = nearest_codewords(residual, codebook)
        residual = residual - jnp.take(codebook.astype(jnp.float32), index, axis=0)
        codes.append(index)
    return jnp.stack(codes, axis=-1).astype(jnp.int32)


def residual_dequantize(codes, codebooks: Sequence[jax.Array]) -> jax.Array:
    """The sum of each stage's codeword for `codes` `[..., stages]`."""
    return functools.reduce(jnp.add, [jnp.take(codebook.astype(jnp.float32), codes[..., stage], axis=0)
                                      for stage, codebook in enumerate(codebooks)])
