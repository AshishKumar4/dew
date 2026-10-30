"""A causal transformer's work counts (`CausalTransformer.model_flops` and its kin): its layers' and its
root's FLOPs over real rows, its head's, its prediction depths', and how many rows each of its matrices
reads, from the same block factories its setup builds."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from ..attention_residuals import DepthAttention
from ..flops import MissingFlopsCount, counted_flops, parameter_flops, projection_flops
from ..gemma3n import AltUpLayer
from ..hyper_connections import stream_head

if TYPE_CHECKING:
    import flax.linen as nn

    from .causal_transformer import CausalTransformer

    Counter = Callable[[nn.Module, np.ndarray], float]


def model_flops(decoder: CausalTransformer, lengths) -> float:
    """Real-row hidden-state work, summing the same block factories as setup."""
    return (_layer_flops(decoder, lengths, counted_flops)
            + _root_flops(decoder, lengths, DepthAttention.depth_flops))


def trained_flops(decoder: CausalTransformer, lengths) -> float:
    """Learned matrices' work, apart from attention's and a state's that no parameter does."""
    return (_layer_flops(decoder, lengths, parameter_flops)
            + _root_flops(decoder, lengths, DepthAttention.depth_parameter_flops))


def _root_flops(decoder: CausalTransformer, lengths, depth_counter) -> float:
    """The work of the maps around the layers: AltUp's and the input projection, mHC's weighted head,
    and the depth mixture's."""
    maps = decoder.projections()
    count = sum(projection_flops(lengths, *shape) for name, shape in maps.items()
                if name.startswith(('altup_projections_', 'altup_unembed_projections_', 'project_in')))
    if decoder.hyper_connections is not None and decoder.hyper_connections.head in ('weighted', 'gated'):
        count += stream_head(decoder.hyper_connections, decoder.emb_features, decoder.norm_eps,
                             parent=None).model_flops(lengths)
    if decoder.attention_residuals is not None:
        sources = decoder.attention_residuals.blocks(decoder.num_layers) + 1
        count += depth_counter(DepthAttention(decoder.emb_features, parent=None), lengths, sources)
    return count


def layer_counts(decoder: CausalTransformer, lengths, counter: Counter) -> list[float]:
    """Each layer's work by `counter`, over the passes a call runs it."""
    return [layer_passes(decoder, index) * counter(block, lengths)
            for index, block in enumerate(decoder.counted_blocks())]


def replicated_flops(decoder: CausalTransformer, stages: int, lengths) -> float:
    """The work of the layers a pipeline of `stages` runs on every stage rather than once
    (`CausalTransformer.stage_split`): those before and after the run its stages repeat."""
    split = decoder.stage_split(stages)
    counts = layer_counts(decoder, lengths, counted_flops)
    return sum(counts[:split.prefix]) + sum(counts[decoder.num_layers - split.suffix:])


def _layer_flops(decoder: CausalTransformer, lengths, counter: Counter) -> float:
    total = sum(layer_counts(decoder, lengths, counter))
    if decoder.per_layer_input_dim:
        total += projection_flops(lengths, *decoder.projections()['per_layer_model_projection'])
    if decoder.loop is not None and decoder.loop.exit_gate:
        total += decoder.loop.steps * projection_flops(lengths, *decoder.projections()['early_exit_gate'])
    return total


def layer_passes(decoder: CausalTransformer, index: int) -> int:
    """How many times a call runs layer `index`: a loop's steps for a layer it repeats, else once."""
    if decoder.loop is None:
        return 1
    start, end = (0, decoder.num_layers) if decoder.loop.layers is None else decoder.loop.layers
    return decoder.loop.steps if start <= index < end else 1


def logits_flops(decoder: CausalTransformer, lengths) -> float:
    """The vocabulary matrix and optional prediction transform, `project_out`'s included. A tied head
    reads the embedding table, `vocab_size` rows of the embedding's width."""
    projections = decoder.projections()
    tied = (decoder.embedding_features or decoder.emb_features, decoder.vocab_size)
    count = projection_flops(lengths, *projections.get('lm_head', tied))
    for site in ('head_dense', 'continuum_down', 'continuum_up', 'project_out'):
        if site in projections:
            count += projection_flops(lengths, *projections[site])
    return count


def _prediction_rows(decoder: CausalTransformer, lengths, prediction_lengths):
    if prediction_lengths is not None:
        return prediction_lengths
    return [(shorter, shorter) for depth in range(1, decoder.num_nextn_predict_layers + 1)
            for shorter in (np.maximum(np.asarray(lengths) - depth, 0),)]


def prediction_flops(decoder: CausalTransformer, lengths, counter: Counter, prediction_lengths) -> float:
    """Each trained prediction depth's forward by `counter`, its head included: each shortens the real
    sequence by one token."""
    count = 0.0
    rows = _prediction_rows(decoder, lengths, prediction_lengths)
    for block, (inputs, outputs) in zip(decoder.prediction_blocks(), rows, strict=True):
        count += counter(block, inputs) + logits_flops(decoder, outputs)
    return count


def prediction_matrix_usage(decoder: CausalTransformer, path: tuple[str, ...], lengths, prediction_lengths):
    """Each depth's maps run once; the shared head runs on every shorter state."""
    relative = path[1:] if path and path[0] == 'params' else path
    if not relative:
        raise MissingFlopsCount("CausalTransformer prediction matrix has no native scope")
    lengths = np.asarray(lengths)
    rows = _prediction_rows(decoder, lengths, prediction_lengths)
    if relative[0].startswith('mtp_'):
        depth = int(relative[0].removeprefix('mtp_'))
        copies = (decoder.mtp_hyper_connections.hc_mult
                  if relative[1] == 'h_proj' and decoder.mtp_hyper_connections is not None else 1)
        return rows[depth][0] * copies
    if relative[0] in ('lm_head', 'head_dense', 'continuum', 'project_out'):
        return np.concatenate([np.empty((0,), np.int64), *(outputs for _, outputs in rows)])
    return np.zeros_like(lengths)


def matrix_usage(decoder: CausalTransformer, path: tuple[str, ...], lengths, *, logits: bool):
    """Native matrix scopes declare head versus trunk work and repeated loop passes."""
    relative = path[1:] if path and path[0] == 'params' else path
    if not relative:
        raise MissingFlopsCount("CausalTransformer adapter matrix has no native scope")
    # The continuum level stands before the head, which every head read takes it through.
    head = relative[0] in ('lm_head', 'head_dense', 'continuum', 'project_out')
    if head != logits or relative[0].startswith(('mtp_', 'dspark_')):
        return np.zeros(np.shape(lengths), np.int64)
    passes = layer_passes(decoder, int(relative[0].removeprefix('layers_'))) if relative[0].startswith(
        'layers_') else 1
    if decoder.altup is not None and relative[-1] == 'modality_router':
        site = AltUpLayer(decoder.altup, decoder.emb_features, parent=None)
        passes *= site.projection_uses()['modality_router']
    return np.asarray(lengths) * passes
