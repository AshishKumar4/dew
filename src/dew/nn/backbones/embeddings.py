"""How a `CausalTransformer`'s token becomes its stream's first state: the table's lookup with its zeroed
ids, the stream-width projection, the first layer's scaling and its factor, each layer's own input signal
(Gemma's per-layer inputs), and a caller's prepared embeddings in their place. The decoder builds the
tables and their projections in its setup (`build`); its methods of these names read them here, within
its scope."""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.nn.dense import Dense
from dew.nn.dropout import Dropout

from ..blocks import TokenEmbedding, normal_kernel
from ..kv_cache import DecodeCursor
from ..norms import NormSite, RMSNorm
from ..precision import float32_dot_general, scaled
from ..sharding import LogicalAxes, declared
from .decoder_block import decoder_norm

if TYPE_CHECKING:
    from .causal_transformer import CausalTransformer


EMBEDDING_AXES: dict[str, LogicalAxes] = {
    "embed_tokens": ("vocab", "embed"),
    "embed_positions": (None, "embed"),
    "embed_tokens_per_layer": ("vocab", None),
    "per_layer_model_projection": ("embed", "mlp"),
}
"""The logical axes of the embeddings' own parameters, by name, as `build` declares them where it makes
them."""


def build(decoder: CausalTransformer, projections) -> None:
    """The decoder's embeddings, which its setup builds: the token and position tables, their dropout and
    norm, their projections in and out, and Gemma's per-layer input table with its projection and norm."""
    decoder.embed_tokens = TokenEmbedding(
        num_embeddings=decoder.vocab_size, features=decoder.embedding_features or decoder.emb_features,
        dtype=decoder.dtype, name='embed_tokens', padding_idx=decoder.padding_idx,
        embedding_init=declared(TokenEmbedding.embedding_init if decoder.initializer_range is None
                                else nn.initializers.normal(decoder.initializer_range),
                                EMBEDDING_AXES['embed_tokens']))
    if decoder.embedding_features is not None:
        projection = functools.partial(Dense, use_bias=False, dtype=decoder.dtype,
                                       precision=decoder.precision,
                                       **normal_kernel(decoder.initializer_range))
        decoder.project_in = projection(projections['project_in'][1], name='project_in')
        decoder.project_out = projection(projections['project_out'][1], name='project_out')
    if decoder.embedding_dropout_rate:
        decoder.embedding_dropout = Dropout(decoder.embedding_dropout_rate, name="embedding_dropout")
    if decoder.embedding_norm:
        decoder.embedding_layernorm = decoder_norm(
            decoder.norm_type, epsilon=decoder.norm_eps, bias=decoder.norm_bias,
            scale_offset=decoder.scale_offset, scale_after_cast=decoder.scale_after_cast,
            dtype=decoder.dtype, precision=decoder.precision, specs=decoder.norms)(name=NormSite.EMBEDDING)
    if decoder.position_embedding == 'learned':
        decoder.embed_positions = TokenEmbedding(
            num_embeddings=decoder.position_rows,
            features=decoder.emb_features, dtype=decoder.dtype, name='embed_positions',
            embedding_init=declared(TokenEmbedding.embedding_init if decoder.initializer_range is None
                                    else nn.initializers.normal(decoder.initializer_range),
                                    EMBEDDING_AXES['embed_positions']))
        decoder.decode_cursor = DecodeCursor()
    if ple := decoder.per_layer_input_dim:
        # The packed table every layer reads its own slice of
        # (modeling_gemma4.py, Gemma4TextModel): one row per token, a
        # hidden_size_per_layer_input slice per layer.
        decoder.embed_tokens_per_layer = TokenEmbedding(
            num_embeddings=decoder.per_layer_vocab, features=decoder.num_layers * ple,
            dtype=decoder.dtype, name='embed_tokens_per_layer', padding_idx=decoder.padding_idx,
            embedding_init=declared(TokenEmbedding.embedding_init,
                                    EMBEDDING_AXES['embed_tokens_per_layer']))
        decoder.per_layer_model_projection = Dense(
            projections['per_layer_model_projection'][1], use_bias=False,
            dtype=decoder.dtype, precision=decoder.precision, name='per_layer_model_projection',
            dot_general=float32_dot_general, kernel_axes=EMBEDDING_AXES['per_layer_model_projection'])
        decoder.per_layer_projection_norm = RMSNorm(
            epsilon=decoder.norm_eps, scale_offset=decoder.scale_offset,
            scale_after_cast=decoder.scale_after_cast, dtype=decoder.dtype,
            name='per_layer_projection_norm')


def token_embeddings(decoder: CausalTransformer, tokens) -> jax.Array:
    """The table's rows for `tokens`, each of `embedding_zero_ids` read as token zero, unscaled: what a
    prediction depth pairs with them. A drawn token is always text, so no media embedding replaces it."""
    lookup = tokens
    for token_id in decoder.embedding_zero_ids:
        lookup = jnp.where(tokens == token_id, 0, lookup)
    return decoder.embed_tokens(lookup)


def stream_embeddings(decoder: CausalTransformer, tokens) -> jax.Array:
    """`token_embeddings` at the stream's width, through `project_in` past an embedding's own: what a
    prediction depth pairs with `tokens`, and a context's tokens a cross-attention reads by the table."""
    looked = token_embeddings(decoder, tokens)
    return looked if decoder.embedding_features is None else decoder.project_in(looked)


def scaled_embeddings(decoder: CausalTransformer, x) -> jax.Array:
    """The token embeddings `x` scaled the way the first layer reads them.

    `x` is normalized first where the norm reads the lookup alone (`embedding_norm_reads`, Muse Glimmer),
    multiplied by sqrt(emb_features) when `embedding_scale` is set (Gemma), then by `embedding_multiplier`
    (lm-engine, GraniteMoeHybrid). The decoder, the multimodal wrapper and the Qwen-Image conditioner all
    scale here.

    Gemma casts embed_scale to the embedding weight dtype. The lookup holds that table in fp32, so the
    factor stays fp32 and only the product is rounded (a bf16 factor would be 34.0 at hidden size 1152,
    not 33.941...). lm-engine multiplies in fp32, and `scaled` does the same.
    """
    # Gemma's cast: modeling_gemma3.py:117. lm-engine's fp32 opmath: mixins/dense/base.py at 45b6b57b.
    if decoder.embedding_norm and decoder.embedding_norm_reads == 'lookup':
        x = decoder.embedding_layernorm(x)
    if decoder.embedding_scale:
        x = (x * jnp.asarray(math.sqrt(decoder.emb_features),
                             decoder.embed_tokens.embedding.dtype)).astype(x.dtype)
    x = scaled(x, decoder.embedding_multiplier)
    return x if decoder.embedding_features is None else decoder.project_in(x)


def embedding_factor(decoder: CausalTransformer) -> float:
    """The factor `scaled_embeddings` scales an embedding by, unrounded, for a mixture of the table's rows
    to take in its own arithmetic (`dew.nn.diffusion_gemma.soft_embeddings`)."""
    return ((math.sqrt(decoder.emb_features) if decoder.embedding_scale else 1.0)
            * decoder.embedding_multiplier)


def per_layer_inputs(decoder: CausalTransformer, tokens, inputs_embeds) -> jax.Array:
    """Every layer's input signal `[B, S, L, P]` (Gemma 3n/4 PLE).

    The token-identity part is the packed table's row for each token, scaled like the main embedding. The
    context part is the input embeddings projected down, scaled and normed. Each layer's gate multiplies
    in their sum divided by sqrt(2). A model without `per_layer_input_dim` raises `ValueError`.
    """
    # modeling_gemma4.py, get_per_layer_inputs and project_per_layer_inputs.
    ple = decoder.per_layer_input_dim
    if ple is None:
        raise ValueError(
            "per_layer_inputs is the signal of a model with per-layer input "
            "embeddings, and this one has none: per_layer_input_dim is unset")
    table = decoder.embed_tokens_per_layer(tokens).reshape(*tokens.shape, decoder.num_layers, ple)
    # The reference scales by sqrt(P) cast to the table's weight dtype (modeling_gemma4.py,
    # Gemma4TextScaledWordEmbedding), not to the activation dtype.
    table = table * jnp.asarray(math.sqrt(ple), decoder.embed_tokens_per_layer.embedding.dtype)
    context = decoder.per_layer_model_projection(inputs_embeds)
    context = context * jnp.asarray(decoder.emb_features ** -0.5, context.dtype)
    context = decoder.per_layer_projection_norm(
        context.reshape(*inputs_embeds.shape[:-1], decoder.num_layers, ple))
    return (context + table) * jnp.asarray(2.0 ** -0.5, context.dtype)


def prepared_embeddings(x: jax.Array, input_embeddings) -> jax.Array:
    """`x`, or in its place a caller's prepared `[B, S, D]` embeddings (another encoder's outputs fused in),
    cast to the stream dtype."""
    if input_embeddings is None:
        return x
    prepared = jnp.asarray(input_embeddings)
    if prepared.shape != x.shape:
        raise ValueError(f"input_embeddings are the whole {x.shape} sequence, got {prepared.shape}")
    return prepared.astype(x.dtype)
