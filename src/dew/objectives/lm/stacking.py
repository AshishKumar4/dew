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
among them, as its cache reads them. Each chunk reads the row's layout (`Coordinates`) at its own slots,
its places, validity and documents as the caller gave them; the decode cache keeps no document ids, so a
packed row of several documents is refused there (`CausalTransformer`'s own refusal): one document a row.
A row whose attention reads its blocks whole (a prefix run's, `block_groups`) reads each block in one call,
as a server's prefill reads a prompt's: a chunk's end that falls inside a block moves to the block's end
(`_chunk_bounds`), so the chunk the block starts in reads it whole before the cache holds it, at that
chunk's levels, and a block within one chunk stays there. Each chunk's call is the least of three widths
that holds its longest row's chunk (`_widths`): the least window, twice it, or the whole row; a chunk a
longer block covers whole is not called, and its levels' windows open and close as any other's.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import PRNGKey, RNGSequences

from dew.batch import BLOCKS_FIELD, COORDINATES_FIELD, TOKEN_LAYOUT_FIELDS, Coordinates, field_coordinates
from dew.nn.backbones.decoder_stack import StackView
from dew.nn.continuum import feedforward_levels, with_feedforward_levels
from dew.nn.protocols import SteppedFeedForward
from dew.objectives.lm.adaptation import CARRIED, Adapted, ChunkAdaptation, _moved, _rows, _stepped_mask


def stacked_scores(model: nn.Module, variables, inputs: jax.Array, targets: jax.Array, weights: jax.Array,
                   fields: Mapping, adaptation: ChunkAdaptation, *, predict: bool,
                   logits: bool, train: bool = False, rngs=None) -> tuple[jax.Array, Adapted]:
    """Each target's next-token loss with the decoder's feed-forward blocks of `adaptation.layers`
    stepped in context (module docstring), over `inputs` `[B, S]` and their `targets` `[B, S]`, each
    weighted as the loss weighs it (`weights`), the rows' token `fields` (their layout, flat or as
    `Coordinates`) read by the decoder's calls: the final states the head read and the scores
    (`Adapted`)."""
    if not isinstance(model, SteppedFeedForward):
        raise TypeError(f"{type(model).__name__} steps no feed-forward block in context; a decoder that does "
                        "names its blocks' parameters (`SteppedFeedForward`)")
    if adaptation.damped:
        raise ValueError("a damped step bounds a continuum level's curvature through the head, which a "
                         "feed-forward block mid-stack reads through every later layer: step it undamped")
    unread = sorted(set(fields) - {COORDINATES_FIELD, BLOCKS_FIELD, *TOKEN_LAYOUT_FIELDS})
    if unread:
        raise ValueError(f"a decoder stepping its own blocks reads a row's layout alone through its cache, "
                         f"not its token fields {unread}")
    layers = adaptation.layers
    windows, rates = adaptation.windows(len(layers)), adaptation.rates(len(layers))
    chunk = min(windows)
    view = model.stored_view(variables)
    starts = feedforward_levels(view, variables, layers)
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
    padded = chunks * chunk
    cache = model.apply(variables, 1, method="init_cache", mutable=["cache"])[1]["cache"]
    layout = _padded_layout(field_coordinates(fields, (rows, length)), rows, length, padded)
    blocks = fields.get(BLOCKS_FIELD)
    if blocks is not None:
        blocks = jnp.pad(jnp.asarray(blocks, jnp.int32), ((0, 0), (0, padded - length)), constant_values=-1)
    bounds = _chunk_bounds(blocks, rows, chunk, chunks)
    rowed = {name: jnp.pad(value, ((0, 0), (0, padded - length)))
             for name, value in (("tokens", inputs), ("targets", targets), ("weights", weights))}
    started = {name: _rows(level, rows) for name, level in starts.items()}

    def stepped(name: str, level, gradient, rate: float):
        return jax.tree.map(lambda leaf, grad, moves: leaf - rate * grad if moves else leaf, level, gradient,
                            steps[name])

    widths = (chunk,) if blocks is None else _widths(chunk, padded)
    wide = widths[-1]
    spans = jnp.max(bounds[:, 1:] - bounds[:, :-1], axis=0)
    # Each chunk's call: none where no row's chunk holds a slot, else the least width its longest holds.
    calls = jnp.where(spans > 0, jnp.searchsorted(jnp.asarray(widths), spans) + 1, 0)

    reads = _Reads(model, variables, view, layout, rowed, blocks, train, rngs, predict, logits)
    branches = [reads.called(chunk)] if blocks is None else [reads.skipped, *map(reads.called, widths)]

    def chunk_call(levels, caches, block, written):
        # A row of blocks reads each chunk at its own width (`calls`).
        return branches[0](levels, caches, block, written) if blocks is None else jax.lax.switch(
            block["call"], branches, levels, caches, block, written)

    def chunk_step(carry, block):
        opens, partials, caches, written = carry
        index = block["index"]
        levels = {}
        opened = {}
        for at, name in enumerate(names):
            # A level's next window opens at a chunk a multiple of its window's chunks from the row's start.
            opened[name] = (index > 0) & (index % (windows[at] // chunk) == 0)
            moved = stepped(name, opens[name], partials[name], rates[at])
            levels[name] = jax.tree.map(lambda one, other, now=opened[name]: jnp.where(now, one, other),
                                        moved, opens[name])
        written, caches, gradients = chunk_call(levels, caches, block, written)
        partials = {name: jax.tree.map(
            lambda grad, held, now=opened[name]: jnp.where(now, grad, held + grad), gradients[name],
            partials[name]) for name in names}
        return (levels, partials, caches, written), None

    chunked = {"index": jnp.arange(chunks), "first": bounds[:, :-1].T, "last": bounds[:, 1:].T, "call": calls}
    caches = _rows(cache, rows)
    first = jax.tree.map(lambda leaf: leaf[0], chunked)
    shapes, _, _ = jax.eval_shape(reads.scored(chunk), started, caches, first)
    # Each row's outputs at their slots: a call writes its own width from its chunk's first, the slots
    # past its chunk's last overwritten by the chunks after it.
    written = jax.tree.map(lambda shape: jnp.zeros((rows, padded + wide, *shape.shape[2:]), shape.dtype),
                           shapes)
    zeros = jax.tree.map(jnp.zeros_like, started)
    (levels, partials, _, written), _ = jax.lax.scan(chunk_step, (started, zeros, caches, written), chunked)
    losses, top, log_z, whole, states = jax.tree.map(lambda buffer: buffer[:, :length], written)

    moved = None
    if adaptation.carry == "run":
        # Each row's levels with their open windows' steps applied.
        reached = {name: stepped(name, levels[name], partials[name], rates[at])
                   for at, name in enumerate(names)}
        moved = _moved(reached, started, weights)
    return states, Adapted(losses, top, log_z, whole, moved)


@dataclasses.dataclass(frozen=True)
class _Reads:
    """What a chunk's call through the decoder reads (`stacked_scores`): the decoder and its `variables`,
    their store (`view`), the rows' padded `layout`, token fields (`rowed`) and `blocks` (None for none),
    and the call's own options."""

    model: nn.Module
    variables: Mapping
    view: StackView
    layout: Coordinates
    rowed: Mapping
    blocks: jax.Array | None
    train: bool
    rngs: PRNGKey | RNGSequences | None
    predict: bool
    logits: bool

    def row_forward(self, levels, row_cache, tokens, row_layout, row_blocks):
        """One row's chunk through the decoder at `levels`, from its cache: its final states, logits and
        cache past the chunk."""
        held = {**with_feedforward_levels(self.view, self.variables, levels), "cache": row_cache}
        (states, chunk_logits), mutated = self.model.apply(
            held, tokens[None], decode=True, train=self.train, rngs=self.rngs, mutable=["cache"],
            method="states_and_logits", coordinates=row_layout,
            block_groups=None if row_blocks is None else row_blocks[None])
        return states[0], chunk_logits[0], mutated["cache"]

    def scored(self, width: int):
        """A chunk's call, each row's `width` slots from its first (`bounds`), those past its last invalid,
        rematerialized for the outer gradient: its final states, scores and logits, the caches past it and
        its loss's gradient in every level, the chunks before it fixed in the cache."""
        from dew.objectives.lm.chunked import logits_cross_entropy

        def chunk_loss(levels, caches, block):
            first, last = block["first"], block["last"]
            inside = first[:, None] + jnp.arange(width) < last[:, None]
            here = self.layout.along_tokens(lambda leaf: _cut(leaf, first, width))
            here = dataclasses.replace(here, valid=here.valid & inside)
            read = {name: _cut(value, first, width) for name, value in self.rowed.items()}
            blocks = (None if self.blocks is None
                      else jnp.where(inside, _cut(self.blocks, first, width, -1), -1))
            states, chunk_logits, moved = jax.vmap(self.row_forward)(
                levels, caches, read["tokens"], jax.tree.map(lambda leaf: leaf[:, None], here), blocks)
            losses, top, log_z = logits_cross_entropy(chunk_logits, read["targets"], predict=self.predict)
            counted = jnp.where(inside, jnp.asarray(read["weights"], losses.dtype), 0)
            outputs = (losses, top, log_z, chunk_logits if self.logits else None, states)
            return jnp.sum(counted * losses), (outputs, moved)

        @jax.checkpoint
        def call(levels, caches, block):
            (_, (outputs, caches)), gradients = jax.value_and_grad(
                functools.partial(chunk_loss, caches=caches, block=block), has_aux=True)(levels)
            return outputs, caches, gradients
        return call

    def called(self, width: int):
        """`scored(width)` with its outputs written into the row's buffers `written` at each row's chunk's
        slots, `width` of them from its first."""
        scored = self.scored(width)

        def call(levels, caches, block, written):
            outputs, caches, gradients = scored(levels, caches, block)
            return jax.tree.map(lambda buffer, value: _placed(buffer, value, block["first"]), written,
                                outputs), caches, gradients
        return call

    @staticmethod
    def skipped(levels, caches, block, written):
        """A chunk no row's holds a slot of, a longer block's chunk having read it whole: no call, nothing
        written or cached, and no gradient in any level."""
        del block
        return written, caches, jax.tree.map(jnp.zeros_like, levels)


def _chunk_bounds(blocks: jax.Array | None, rows: int, chunk: int, chunks: int) -> jax.Array:
    """Each row's chunk bounds `[B, chunks + 1]` over its blocks `[B, chunks * chunk]` (-1 for none; None
    for a row of none): the multiples of `chunk`, each that falls inside a block (a slot of the block of
    the slot before it) moved to the block's end, so a block is read whole by the chunk it starts in and a
    block within one chunk stays there; a chunk a longer block covers whole holds no slot."""
    nominal = jnp.arange(chunks + 1) * chunk
    if blocks is None:
        return jnp.broadcast_to(nominal, (rows, chunks + 1))
    padded = chunks * chunk
    within = jnp.concatenate([jnp.zeros((rows, 1), bool),
                              (blocks[:, 1:] >= 0) & (blocks[:, 1:] == blocks[:, :-1])], axis=1)
    # Each slot's first slot from it on that continues no block: a bound there moves to it.
    breaks = jax.lax.cummin(jnp.where(within, padded, jnp.arange(padded)), axis=1, reverse=True)
    return jnp.concatenate([breaks, jnp.full((rows, 1), padded)], axis=1)[:, nominal]


def _widths(chunk: int, padded: int) -> tuple[int, ...]:
    """The call widths a row of blocks reads its chunks in, three programs whatever the row's length: a chunk
    at `chunk`, one a block no longer than a chunk lengthens at twice it (a block crossing a chunk's end runs
    on at most a chunk past it), and one a longer block lengthens at the whole `padded` row."""
    return tuple(sorted({chunk, min(2 * chunk, padded), padded}))


def _cut(leaf: jax.Array, first: jax.Array, width: int, fill=0) -> jax.Array:
    """Each row's `width` slots of `leaf` `[B, S, ...]` from its `first` `[B]`, past the row's end `fill`."""
    leaf = jnp.pad(leaf, ((0, 0), (0, width), *((0, 0),) * (leaf.ndim - 2)), constant_values=fill)
    return jax.vmap(lambda row, at: jax.lax.dynamic_slice_in_dim(row, at, width))(leaf, first)


def _placed(buffer: jax.Array, value: jax.Array, first: jax.Array) -> jax.Array:
    """`buffer` `[B, S, ...]` with each row's `value` `[B, W, ...]` written from its `first` `[B]` on."""
    return jax.vmap(lambda row, part, at: jax.lax.dynamic_update_slice_in_dim(row, part, at, axis=0))(
        buffer, value, first)


def _padded_layout(layout: Coordinates | None, rows: int, length: int, padded: int) -> Coordinates:
    """A row's `layout` (None for every slot real, one document) over `padded` slots: the slots past the
    row padding, invalid, of no document and at no place, which no real token reads."""
    if layout is None or layout.valid is None:
        base = Coordinates() if layout is None else layout
        layout = dataclasses.replace(base, valid=jnp.ones((rows, length), bool))
    extra = padded - length
    return layout.along_tokens(lambda leaf: jnp.pad(leaf, ((0, 0), (0, extra))))
