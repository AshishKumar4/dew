"""A `CausalTransformer`'s head: the modules it builds past the final norm (`build`: `lm_head`, the
head transform, the head bias and the continuum levels), the fp32 logits over final states, the matrix
they contract (`head_table`, `output_table`, `vocabulary_rows`) and what bounds a slot loss's curvature
in them (`bounds`). The decoder's methods of these names read them here, within its scope."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.nn.continuum import HeadTerms, HeadTransform, _head_growth, _squared_norm, continuum_modules
from dew.nn.dense import Dense

from ..activations import _SLOPES, UNGATED, ungated_activation
from ..blocks import normal_kernel
from ..kernels import projections
from ..ngpt import gain
from ..norms import NormSite
from ..precision import (
    at_least_fp32,
    capped_logits,
    float32_dot_general,
    head_dot_general,
    head_product,
    output_dtype,
    scaled_logits,
)
from ..protocols import BoundedNorm, OutputTable, Projection, QuantizesInputs
from ..sharding import LogicalAxes, declared
from .decoder_block import decoder_norm

if TYPE_CHECKING:
    from .causal_transformer import CausalTransformer

HEAD_AXES: dict[str, LogicalAxes] = {"lm_head": ("embed", "vocab"), "head_bias": ("vocab",)}
"""The logical axes of the head's own parameters, by name, as `build` declares them where it makes them."""


def build(decoder: CausalTransformer, projections) -> None:
    """The head's modules past the final norm, which the decoder's setup builds: `lm_head` where the
    embeddings are not tied, the continuum levels, the head transform's map and norm, and the head bias."""
    if not decoder.tie_embeddings:
        decoder.lm_head = Dense(
            features=projections['lm_head'][1], use_bias=False, dtype=at_least_fp32(decoder.dtype),
            precision=decoder.precision,
            dot_general=head_dot_general(decoder.dtype, decoder.precision),
            kernel_axes=HEAD_AXES['lm_head'], name='lm_head', **normal_kernel(decoder.initializer_range))
    if decoder.continuum is not None:
        decoder.continuum_level = continuum_modules(decoder.continuum, decoder.emb_features, decoder.dtype,
                                                 decoder.precision)
    if decoder.head_transform is not None:
        decoder.head_dense = Dense(
            projections['head_dense'][1], use_bias=False, dtype=decoder.dtype, precision=decoder.precision,
            name='head_dense', **normal_kernel(decoder.initializer_range), dot_general=float32_dot_general)
        decoder.head_norm = decoder_norm(
            decoder.norm_type, epsilon=decoder.norm_eps, bias=decoder.norm_bias,
            scale_offset=decoder.scale_offset, scale_after_cast=decoder.scale_after_cast,
            dtype=decoder.dtype, precision=decoder.precision, specs=decoder.norms)(name=NormSite.HEAD)
    if decoder.head_bias:
        decoder.head_bias_value = decoder.param(
            'head_bias', declared(nn.initializers.zeros_init(), HEAD_AXES['head_bias']),
            (decoder.vocab_size,), jnp.float32)


def token_gains(decoder: CausalTransformer) -> jax.Array | None:
    """nGPT's fp32 per-token gain on the logits, past the head's product; None without one."""
    return None if decoder.ngpt is None else gain(decoder.logit_scale, decoder.ngpt.logits, decoder.dtype)


def bounds(decoder: CausalTransformer, states) -> HeadTerms:
    """What bounds a slot loss's curvature and slope in the `states` `head_logits` reads (`HeadTerms`):
    the bound rows (`vocabulary_rows`), softcap and transform, its norm read at the states
    (`BoundedNorm`), each kernel as Qwix's rounding can grow it (`_head_growth`)."""
    table = head_table(decoder)
    projection = table.projection
    maps = {None if decoder.tie_embeddings else "lm_head": jnp.shape(table.rows())[-1],
            **({} if projection is None else {"project_out": jnp.shape(projection.kernel)[0]}),
            **({} if decoder.head_transform is None else {"head_dense": decoder.emb_features})}
    growth = _head_growth(decoder, maps, decoder.dtype or states.dtype)
    logits = growth[None if decoder.tie_embeddings else "lm_head"]
    squared = jnp.square(logits) * _squared_norm(jnp.asarray(vocabulary_rows(decoder)))
    if projection is not None and logits * growth["project_out"] > 1:
        # Two rounded matrices' product, bounded by their norms'.
        squared = (jnp.square(logits * growth["project_out"]) * _squared_norm(table.rows())
                   * _squared_norm(projection.kernel))
    if decoder.head_transform is None:
        return HeadTerms.of(squared, decoder.final_logit_softcap, jnp.zeros_like(states[..., 0]))
    if not isinstance(decoder.head_norm, BoundedNorm):
        raise ValueError(f"a damped step bounds a head transform's curvature through its norm's "
                         f"derivatives, which {type(decoder.head_norm).__name__} bounds none of")
    dense = decoder.applied_kernel("head_dense")
    spread = decoder.head_norm.spread(ungated_activation(decoder.head_transform, decoder.head_dense(states)))
    return HeadTerms.of(squared, decoder.final_logit_softcap, spread, HeadTransform.of(
        _SLOPES[UNGATED[decoder.head_transform]], growth["head_dense"] * jnp.sqrt(_squared_norm(dense)),
        decoder.head_norm.derivative_bounds(dense.shape[-1])))


def head_logits(decoder: CausalTransformer, x):
    """The fp32 head over states past the continuum levels (`continuum_states`), or final states."""
    # A stream-width transform reads the final states before `project_out` takes them to the table's.
    if decoder.head_transform is not None:
        x = decoder.head_norm(ungated_activation(decoder.head_transform, decoder.head_dense(x)))
    if decoder.embedding_features is not None:
        x = decoder.project_out(x)
    # fp32 logits whose product follows the compute dtype
    # (`dew.nn.precision.head_product`, the chunked loss's arithmetic).
    width = decoder.logits_width
    if decoder.tie_embeddings:
        logits = head_product('...d,vd->...v', x, decoder.embed_tokens.embedding[:width], decoder.precision)
    else:
        # Every row the head stores, its first `width` scored (`output_vocab`).
        logits = decoder.lm_head(x)[..., :width]
    # The head's own dtype stands for the compute's where the model states none (a float64 truth's).
    computed, logits = logits.dtype, logits.astype(at_least_fp32(logits.dtype))
    # Returned in the run's output dtype (`output_dtype`), past the fp32 gains, bias and cap.
    return capped_logits(scaled_logits(logits, _scored(token_gains(decoder), width),
                                       _scored(decoder.head_bias_value if decoder.head_bias else None,
                                               width)),
                         decoder.final_logit_softcap).astype(
        output_dtype(computed if decoder.dtype is None else decoder.dtype))


def output_table(decoder: CausalTransformer) -> OutputTable | None:
    """Return the head as the matrix `logits_from_hidden` contracts, read from the bound variables
    (`head_table`). None where no matrix alone is the head: past a `head_transform` or a continuum
    level, or where `lm_head`'s or `project_out`'s parameters are not its kernel alone: `dew.lora`'s
    factors beside it, which its interceptor adds, or no kernel, a frozen FP8 layer's."""
    layers = [*([] if decoder.tie_embeddings else [decoder.lm_head]),
              *([] if decoder.embedding_features is None else [decoder.project_out])]
    if (decoder.head_transform is not None or decoder.continuum is not None
            or any(set(layer.variables.get("params", {})) != {"kernel"} for layer in layers)):
        return None
    return head_table(decoder)


def vocabulary_rows(decoder: CausalTransformer) -> jax.Array:
    """Each vocabulary token's `[features]` row of the head (`OutputTable.rows`) where no exact
    table is: past a `head_transform` the matrix's, with an adapter on the head as it updates them
    (`applied_kernel`), and past an embedding's own width in the stream's, through `project_out`."""
    table = head_table(decoder)
    rows = table.rows()
    return rows if table.projection is None else rows @ table.projection.kernel.T.astype(rows.dtype)


def quantizer(decoder: CausalTransformer, layer: nn.Module):
    """A checkpoint's quantizer of `layer`'s input, None where the model computes under none."""
    return decoder.input_quantizer(layer) if isinstance(decoder, QuantizesInputs) else None


def head_table(decoder: CausalTransformer) -> OutputTable:
    """The head's matrix, as `lm_head`'s call multiplies by it (`applied_kernel`) or the tied
    embeddings, with `logits_from_hidden`'s scale, bias, softcap, precision and compute dtype, a checkpoint's
    quantizer of its input, and `project_out` before it past an embedding's own width (`Projection`)."""
    width = decoder.logits_width
    matrix = (decoder.embed_tokens.embedding[:width] if decoder.tie_embeddings
              else decoder.applied_kernel("lm_head")[:, :width])
    projection = None if decoder.embedding_features is None else Projection(
        decoder.applied_kernel("project_out"), decoder.precision, compute=decoder.dtype,
        input_quantizer=quantizer(decoder, decoder.project_out),
        path=projections.kernel_path(decoder.project_out))
    return OutputTable(matrix, decoder.tie_embeddings,
                       _scored(decoder.head_bias_value if decoder.head_bias else None, width),
                       decoder.final_logit_softcap, decoder.precision, compute=decoder.dtype,
                       input_quantizer=(None if decoder.tie_embeddings
                                        else quantizer(decoder, decoder.lm_head)),
                       scale=_scored(token_gains(decoder), width), projection=projection)


def _scored(per_token: jax.Array | None, width: int) -> jax.Array | None:
    """A per-token term of the head (its bias, its gains) over the tokens it scores, its first `width`."""
    return None if per_token is None else per_token[..., :width]
