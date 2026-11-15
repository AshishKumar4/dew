"""How a `CausalTransformer`'s token becomes its stream's first state: the table's lookup with its zeroed
ids, the stream-width projection, the first layer's scaling and its factor, each layer's own input signal
(Gemma's per-layer inputs), and a caller's prepared embeddings in their place. The decoder builds the
tables and their projections (`_embeddings`); its methods of these names read them here, within its
scope."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from ..precision import scaled

if TYPE_CHECKING:
    from .causal_transformer import CausalTransformer


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
