"""A pretrained decoder whose own feed-forward blocks step in context: Nested Learning's ad-hoc level
stacking (Behrouz et al., arXiv 2512.24695, section 7.3).

Section 7.3 initializes a continuum memory system's levels with a pretrained model's MLP blocks,
MLP_0^(f_i) = MLP_i^pre-trained, and steps each by equation 71 at its own frequency; section 9 adapts
Llama 3 so, its MLP blocks "placed in different levels with different frequency updates", followed by
continued pre-training. Here the levels are the decoder's own feed-forward blocks of the layers an
adaptation names (`ChunkAdaptation.layers`), each at its window and rate: the model's tree is unchanged,
its blocks' trained values the levels' starts, and an objective that steps them in context trains those
starts through every step, the outer loss's meta-gradient.

A block mid-stack reads every earlier token through attention, so the row runs chunk by chunk, a chunk
the least window, through the decoder's decode cache (`CausalTransformer.init_cache`): each chunk reads
the chunks before it from the cache as they were, at the levels' versions they read, and its tokens
read each level at the version its window opened with. A window's gradient in its level is the sum over
its chunks of each chunk's loss's gradient there, the cache holding the chunks before it fixed: dynamic
evaluation's segment gradient, a chunk its segment. A level's window counts the row's slots, its padding
among them, as its cache reads them. The decode cache keeps no document ids, so a packed row of several
documents is refused there (`CausalTransformer`'s own refusal): one document a row.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.batch import token_coordinates
from dew.nn.inputs import sequence_blocks
from dew.nn.protocols import SteppedFeedForward
from dew.objectives.lm.adaptation import CARRIED, Adapted, ChunkAdaptation, _moved, _rows, _stepped_mask


def stacked_scores(model: nn.Module, variables, inputs: jax.Array, targets: jax.Array, weights: jax.Array,
                   fields: Mapping[str, jax.Array], adaptation: ChunkAdaptation, *, predict: bool,
                   logits: bool, train: bool = False, rngs=None) -> tuple[jax.Array, Adapted]:
    """Each target's next-token loss with the decoder's feed-forward blocks of `adaptation.layers`
    stepped in context (module docstring), over `inputs` `[B, S]` and their `targets` `[B, S]`, each
    weighted as the loss weighs it (`weights`), the rows' token `fields` (a validity mask, document ids)
    read by the decoder's calls: the final states the head read and the scores (`Adapted`)."""
    from dew.objectives.lm.chunked import logits_cross_entropy

    if not isinstance(model, SteppedFeedForward):
        raise TypeError(f"{type(model).__name__} steps no feed-forward block in context; a decoder that does "
                        "names its blocks' parameters (`SteppedFeedForward`)")
    if adaptation.damped:
        raise ValueError("a damped step bounds a continuum level's curvature through the head, which a "
                         "feed-forward block mid-stack reads through every later layer: step it undamped")
    layers = adaptation.layers
    windows, rates = adaptation.windows(len(layers)), adaptation.rates(len(layers))
    chunk = min(windows)
    starts = model.feedforward_levels(variables, layers)
    if adaptation.carry == "run":
        if CARRIED not in variables:
            raise ValueError(f"a run that carries its levels holds what it carried in `{CARRIED}`, and these "
                             "variables hold none: a state built for this objective (`LMObjective.init`)")
        starts = jax.tree.map(lambda leaf, held: leaf + jax.lax.stop_gradient(held).astype(leaf.dtype),
                              starts, variables[CARRIED])
    names = tuple(starts)
    steps = {name: _stepped_mask(starts[name], adaptation) for name in names}
    rows, length = targets.shape
    chunks = -(-length // chunk)
    cache = model.apply(variables, 1, method="init_cache", mutable=["cache"])[1]["cache"]

    def padded(value: jax.Array, fill: float = 0) -> jax.Array:
        """`value` `[B, S, ...]` padded to whole chunks and cut into them, `[chunks, B, chunk, ...]`."""
        edge = ((0, 0), (0, chunks * chunk - length), *((0, 0),) * (value.ndim - 2))
        return sequence_blocks(jnp.pad(value, edge, constant_values=fill), chunk, leading=True)

    # A slot past the row is padding, which weighs nothing and the cache does not hold.
    mask = fields.get("attention_mask", jnp.ones((rows, length), bool))
    blocked = {"tokens": padded(inputs), "targets": padded(targets), "weights": padded(weights),
               "attention_mask": padded(jnp.asarray(mask, bool)),
               **({"segment_ids": padded(fields["segment_ids"])} if "segment_ids" in fields else {}),
               "index": jnp.arange(chunks)}

    def row_forward(levels, row_cache, row_chunk):
        """One row's chunk through the decoder at `levels`, from its cache: its final states, logits and
        cache past the chunk."""
        held = {**model.with_feedforward_levels(variables, levels), "cache": row_cache}
        # The chunk's layout as the decoder's own call converts it (`token_coordinates`).
        segments = row_chunk.get("segment_ids")
        coordinates = token_coordinates((1, chunk), segments=None if segments is None else segments[None],
                                        valid=row_chunk["attention_mask"][None])
        (states, chunk_logits), mutated = model.apply(
            held, row_chunk["tokens"][None], decode=True, train=train, rngs=rngs, mutable=["cache"],
            method="states_and_logits", coordinates=coordinates)
        return states[0], chunk_logits[0], mutated["cache"]

    def chunk_loss(levels, caches, block):
        states, chunk_logits, moved = jax.vmap(row_forward)(
            levels, caches, {name: value for name, value in block.items() if name != "index"})
        losses, top, log_z = logits_cross_entropy(chunk_logits, block["targets"], predict=predict)
        total = jnp.sum(jnp.asarray(block["weights"], losses.dtype) * losses)
        return total, (losses, top, log_z, chunk_logits if logits else None, states, moved)

    def stepped(name: str, level, gradient, rate: float):
        return jax.tree.map(lambda leaf, grad, moves: leaf - rate * grad if moves else leaf, level, gradient,
                            steps[name])

    def chunk_step(carry, block):
        opens, partials, caches = carry
        index = block["index"]
        levels = {}
        opened = {}
        for at, name in enumerate(names):
            # A level's next window opens at a chunk a multiple of its window's chunks from the row's start.
            opened[name] = (index > 0) & (index % (windows[at] // chunk) == 0)
            moved = stepped(name, opens[name], partials[name], rates[at])
            levels[name] = jax.tree.map(lambda one, other, now=opened[name]: jnp.where(now, one, other),
                                        moved, opens[name])
        # Each chunk's gradient in every level is its loss's, the chunks before it fixed in the cache.
        (_, (losses, top, log_z, chunk_logits, states, caches)), gradients = jax.value_and_grad(
            functools.partial(chunk_loss, caches=caches, block=block), has_aux=True)(levels)
        partials = {name: jax.tree.map(lambda grad, held, now=opened[name]: jnp.where(now, grad, held + grad),
                                       gradients[name], partials[name]) for name in names}
        return (levels, partials, caches), (losses, top, log_z, chunk_logits, states)

    started = {name: _rows(level, rows) for name, level in starts.items()}
    zeros = jax.tree.map(jnp.zeros_like, started)
    caches = _rows(cache, rows)
    (levels, partials, _), (losses, top, log_z, whole, states) = jax.lax.scan(
        jax.checkpoint(chunk_step), (started, zeros, caches), blocked)

    def joined(value: jax.Array) -> jax.Array:
        """A scan's `[chunks, B, chunk, ...]` output as the row's `[B, S, ...]`."""
        return jnp.moveaxis(value, 0, 1).reshape(rows, chunks * chunk, *value.shape[3:])[:, :length]

    moved = None
    if adaptation.carry == "run":
        # Each row's levels with their open windows' steps applied.
        reached = {name: stepped(name, levels[name], partials[name], rates[at])
                   for at, name in enumerate(names)}
        moved = _moved(reached, started, weights)
    return joined(states), Adapted(joined(losses), None if top is None else joined(top), joined(log_z),
                                   None if whole is None else joined(whole), moved)
