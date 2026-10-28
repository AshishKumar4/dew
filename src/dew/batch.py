"""The batch contract every layer shares, the one module below them all that names it: a batch
(`Batch`) and a variables tree (`Variables`), the columns a batch's rows carry beside their fields
(`VALID_ROWS`, `EXAMPLE_ROWS`) and the collection frozen weights stay in (`FROZEN`), how a batch is
filled out with repeats of its own rows (`padded_rows`), how a batch's token fields say which slots
are real and where each token sits (`Coordinates`, `real_slots`), the token ids a field holds
(`token_ids`) and which targets a language-model loss scores (`scored_targets`). Data builds batches
by it, models read them by it, and objectives and the trainer score and place them by it, so none of
them imports another for these."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.core import FrozenDict

type Variables = Mapping[str, Any]
"""A Flax variables dict: the `params` collection plus any other collection
the modules keep (`moe`, `batch_stats`, an objective's frozen encoders)."""


type Batch = Mapping[str, Any]
"""One batch of training examples, as the trainer and every objective read it.

It is declared here and imported by `dew.data.dataset`, so a field added on
one side has the same type on the other. The values are not all arrays.
Besides the arrays a step consumes, a batch can hold the prepared text inputs
of a language run, the file paths a video corpus loads lazily, the integer
row counts a packer keeps and the record lists a mixture reads. A narrower
value type would be wrong about what `dew.data` already builds."""


def merge(tree: Variables, overlay: Variables) -> Variables:
    """Return a copy of `tree` with every leaf in `overlay` written over it."""
    merged = dict(tree)
    for name, child in overlay.items():
        held = tree.get(name)
        merged[name] = (merge(held, child)
                        if isinstance(held, Mapping) and isinstance(child, Mapping)
                        else child)
    return merged


FROZEN = "frozen"
"""The name of the collection where a partially trained run keeps its frozen weights.

The optimizer updates the `params` collection and nothing else, so the
leaves that `freeze` keeps in `params` are the ones that train. The frozen
leaves stay in this collection as part of the state, and `thaw` merges the
two before the model reads them."""


VALID_ROWS = "valid_rows"
"""The field every evaluation batch holds: one bool per row, True for a real record and False for a repeat.

A split's last batch is padded to full size with repeats of its own rows, and
a process whose share runs out before the others' scores a copy of its last
batch in which every row is a repeat. That way a pass is whole batches on
every process and counts each record once. A loss weights its rows by this
field, and its batch-wide terms are taken over the real rows only. Training
batches never have it.
"""


EXAMPLE_ROWS = "example_rows"
"""A training batch's column of each row's place among its step's records, int32: the order the
records' stream holds them in, whatever process read a row (`dew.training.distributed.marked`). The
trainer's transaction reads it for the examples' draws (`Step.example_index`) and takes it out of the
batch the objective sees."""


def padded_rows(rows: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """The rows `count` places hold of `rows` real ones, cycling through them, and which places hold
    a repeat: the rule a batch filled out with its own rows (`dew.data.dataset.FilledBatch`) and a
    request's rows padded to its devices (`dew.nn.inputs.RowPlan`) both follow. Places to fill from
    no rows are refused: there is nothing to repeat."""
    if count and not rows:
        raise ValueError(f"{count} places can't repeat rows of none")
    places = np.arange(count)
    return places % rows, places >= rows


VALIDITY_FIELD = "attention_mask"
"""The token field that marks real slots. Absent means every slot is real."""


COORDINATES_FIELD = "coordinates"
"""The token field that holds the tokens' `Coordinates`, which a decoder reads in place of the
validity field, `segment_ids` and `positions`."""

TEXT_KEY = "text"
"""Batch key the token pipeline packs `[B, seq_len + 1]` int32 ids under."""

BLOCKS_FIELD = "block_groups"
"""The token field that numbers the tokens' blocks, which the attention layers that read blocks see whole
(`AttentionMetadata.block_groups`, `dew.nn.visibility.prefix_blocks`)."""


def token_id(value: object, *, where: str) -> int:
    """`value` read as one token id: a Python or NumPy integer, not a bool,
    from 0 through int32's largest. Every parser of ids, a dataset's or a
    request's, reads them by this law; `where` names what held the value."""
    if (not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
            or not 0 <= int(value) <= np.iinfo(np.int32).max):
        raise ValueError(f"{where} holds token ids, nonnegative int32 integers, got {value!r}")
    return int(value)


def token_ids(values: object, *, where: str) -> list[int]:
    """`values` read as a nonempty row of token ids by `token_id`'s law: a list, a tuple or a
    one-dimensional NumPy array, as a tokenizer, a dataset's column or a step of Dew's own
    (`dew.data.text.Tokenize`) holds them."""
    if isinstance(values, np.ndarray) and values.ndim == 1:
        values = values.tolist()
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{where} holds a nonempty row of token ids, got {values!r}")
    return [token_id(token, where=where) for token in values]


@struct.dataclass
class Coordinates:
    """Where each token of a batch sits, built once (`coordinates`) and read by every token mixer.

    `positions` `[B, S]` is each real token's logical place in its document,
    counted over real tokens: a packed batch's or the caller's own, None
    where none is given, each token's place then its rank among its row's
    real tokens (`places`) or, in a cached call, the slot its cache writes it
    at. Rotation and NoPE's temperature read the logical places; windows,
    chunks and ALiBi count each document's real tokens from its first place
    (`counted_places`), the logical places wherever
    those count real tokens; a cache addresses keys by slot. `segments` numbers each
    row's packed documents from 1, 0 on the padding slots after them, which
    run as a document of their own, so no layer carries a real document's
    state or history into them or back; `valid` is the caller's mask of real
    tokens; `starts` marks each document's first real token but a row's
    first, which continues whatever state the row holds. Each is None where
    it holds nothing (one document a row, every token real, no start), which
    a layer's fast path reads as a static fact. `firsts` `[B, D]` are each
    row's documents' first slots in order, the row's first slot among them,
    `S` past its last (`document_firsts`): the packed documents' coordinates
    every varlen kernel reads (`dew.nn.inputs.cu_seqlens`), built once a
    step, never by a layer: by `coordinates` in a traced step, and for a batch
    a host built (whose fields stay its loader's) by the model's entry
    (`with_firsts`); each layer's own scan over the batch's flattened tokens,
    which a mesh splits, ran one token a loop iteration on a 2xH100 run (row
    1897). `max_segments` bounds the documents a row holds
    (the packer's static plan, else the row length) and `right_padded` says
    every row's real tokens come first.
    """

    positions: jax.Array | None = None
    segments: jax.Array | None = None
    valid: jax.Array | None = None
    starts: jax.Array | None = None
    firsts: jax.Array | None = None
    max_segments: int = struct.field(pytree_node=False, default=1)
    right_padded: bool = struct.field(pytree_node=False, default=False)

    TOKEN_ALIGNED: ClassVar[tuple[str, ...]] = ("positions", "segments", "valid", "starts")
    """The fields one entry a token, `[B, S]`; `firsts` is one a document, `[B, D]`, its entries slots."""

    def tokens(self) -> dict[str, jax.Array]:
        """The token-aligned fields it holds (`TOKEN_ALIGNED`), by name: what a check of a batch's token
        layout reads."""
        return {name: getattr(self, name) for name in self.TOKEN_ALIGNED if getattr(self, name) is not None}

    def along_tokens(self, transform: Callable[[jax.Array], jax.Array]) -> Coordinates:
        """`transform` applied along the tokens to each token-aligned field (`TOKEN_ALIGNED`): a slice or a
        rotation of the slots. The documents' firsts, which index the slots, go, their reader rebuilding
        them from the moved starts (`with_firsts`, `dew.nn.inputs.cu_seqlens`)."""
        return replace(self, firsts=None, **{name: transform(value) for name, value in self.tokens().items()})

    def places(self) -> jax.Array | None:
        """Each token's logical place in an uncached call: the given `positions`, else its rank among
        its row's real tokens `[B, S]` (a padding slot's the rank before it, 0 before any); None
        where every token is real and none is given, each token's place its index. A cached call's
        places are the slots its cache writes them at where none are given."""
        if self.positions is not None or self.valid is None:
            return self.positions
        return real_places(self.valid)


def real_places(valid: jax.Array) -> jax.Array:
    """Each slot's rank among its row's real tokens under `valid` `[B, S]`: a real token's place, a
    padding slot's the rank before it, 0 before any (`Coordinates.places` where no positions are given)."""
    return jnp.maximum(jnp.cumsum(valid, axis=1, dtype=jnp.int32) - 1, 0)


def coordinates(shape: tuple[int, int] | None = None, *, segments=None, valid=None, positions=None,
                max_segments: int | None = None, right_padded: bool = False, xp=jnp) -> Coordinates:
    """The `Coordinates` of `[batch, length]` tokens from what the batch carries: packed `segments`
    (one document a row where None), a `valid` mask (every token real where None), explicit
    `positions` (with `segments`, each document's real tokens counted from 0 where None; without,
    None, each token's place its row index or cache slot) and the packer's `max_segments` (the row
    length where None). `shape` is the first given field's where None. `xp` is the array module the
    fields are built with: jax.numpy in a traced call, numpy where a data reader builds a batch's on
    the host."""
    if shape is None:
        given = [x for x in (segments, valid, positions) if x is not None]
        if not given:
            raise ValueError("token coordinates take their shape from a field, or `shape`")
        batch, length = jnp.shape(given[0])[:2]
        shape = (batch, length)
    length = shape[1]
    segments = None if segments is None else xp.asarray(segments, np.int32)
    valid = None if valid is None else xp.asarray(valid, bool)
    for name, field in (("segments", segments), ("valid", valid)):
        if field is not None and field.shape != shape:
            raise ValueError(f"token coordinates are {shape}, got {name} {field.shape}")
    starts = None if segments is None else document_starts(
        segments, xp.ones(shape, bool) if valid is None else valid, xp=xp)
    if positions is None and segments is not None:
        real = xp.ones(shape, bool) if valid is None else valid
        real = real & (segments != 0)
        positions = xp.where(real, document_ranks(real, starts, xp=xp), 0)
    positions = None if positions is None else xp.asarray(positions, np.int32)
    max_segments = (1 if segments is None else length) if max_segments is None else max_segments
    # The documents' firsts in a traced step; a host's batch keeps its loader's fields (`with_firsts`).
    firsts = None if starts is None or xp is np else document_firsts(starts, max_segments + 1)
    return Coordinates(positions, segments, valid, starts, firsts, max_segments=max_segments,
                       right_padded=right_padded)


def with_firsts(tokens: Coordinates | None) -> Coordinates | None:
    """`tokens` with their documents' firsts (`Coordinates.firsts`), built from their starts where a host
    built them without: a model's entry calls it once a step, its layers reading them."""
    if tokens is None or tokens.starts is None or tokens.firsts is not None:
        return tokens
    return replace(tokens, firsts=document_firsts(jnp.asarray(tokens.starts), tokens.max_segments + 1))


def document_firsts(starts, documents: int, *, xp=jnp):
    """Each row's first `documents` documents' first slots in order `[B, min(documents, S)]`, the row's
    first slot and each of `starts` `[B, S]`, `S` past its last: a sort of each row's slots, which a mesh
    splitting the rows sorts on each device (`Coordinates.firsts`)."""
    rows, length = starts.shape
    if xp is np:
        begun = np.asarray(starts, bool).copy()
        begun[:, 0] = True
        slots = np.broadcast_to(np.arange(length, dtype=np.int32), (rows, length))
    else:
        # Placed as the starts are (an Explicit mesh's rows split), each row sorted on its own device.
        begun = jnp.asarray(starts, bool).at[:, 0].set(True)
        placed = jax.typeof(begun).sharding
        slots = jax.lax.broadcasted_iota(jnp.int32, (rows, length), 1, out_sharding=placed)
    return xp.sort(xp.where(begun, slots, length), axis=1)[:, :min(documents, length)].astype(np.int32)


def document_ranks(real, starts, *, xp=jnp):
    """Each slot's count of the real tokens before it in its document `[B, S]`, `starts` marking each
    document's first real token but a row's first, a padding slot taking the count before it (0
    before any)."""
    rank = xp.cumsum(real, axis=1, dtype=np.int32) - 1
    begun = xp.where(starts, rank, 0)
    first = np.maximum.accumulate(begun, axis=1) if xp is np else jax.lax.cummax(begun, axis=1)
    return xp.maximum(rank - first, 0)


def valid_order(valid: jax.Array, *, xp=jnp) -> tuple[jax.Array, jax.Array]:
    """Each row's real slots first, in order, then its padding slots: the
    physical slot behind each compact column `[rows, slots]`, and how many
    slots of each row are real `[rows]`. A stable sort of the validity, so a
    row's tokens keep their order; a caller says what its columns past the
    count hold (decoding's -1 history tokens, a convolution's zero column). `xp` is the array module."""
    valid = xp.asarray(valid, bool)
    return xp.argsort(~valid, axis=-1, stable=True), xp.sum(valid, axis=-1, dtype=np.int32)


def document_starts(segments, valid, *, xp=jnp):
    """`[B, S]`: each real token whose segment differs from the previous real token's in its row, a
    row's first real token continuing what precedes the row and a padding slot starting nothing."""
    order, _ = valid_order(valid, xp=xp)
    rank = xp.cumsum(valid, axis=1, dtype=np.int32) - 1
    compact = xp.take_along_axis(segments, order, axis=1)
    changed = xp.concatenate([xp.zeros_like(compact[:, :1], bool), compact[:, 1:] != compact[:, :-1]],
                             axis=1)
    return valid & xp.take_along_axis(changed, xp.maximum(rank, 0), axis=1)


def token_validity(fields: Mapping[str, jax.Array | np.ndarray | Coordinates]):
    """The validity `[B, S]` a row's token fields hold, as given: their `Coordinates`' where the fields
    carry them, else the validity field; None where neither holds one."""
    laid = fields.get(COORDINATES_FIELD)
    return laid.valid if isinstance(laid, Coordinates) else fields.get(VALIDITY_FIELD)


def token_layout(fields: Mapping[str, jax.Array | np.ndarray | Coordinates], *, xp=jnp):
    """The validity `[B, S]` and documents `[B, S]` of a row's token fields as the decoder reads them:
    their `Coordinates`' where the fields carry them, else the validity field (`token_validity`) and
    `segment_ids`. Each is None where nothing gives it (every slot real, one document a row). `xp` is
    the array module."""
    laid = fields.get(COORDINATES_FIELD)
    valid = token_validity(fields)
    segments = laid.segments if isinstance(laid, Coordinates) else fields.get("segment_ids")
    return (None if valid is None else xp.asarray(valid, bool),
            None if segments is None else xp.asarray(segments))


def with_validity(fields: Mapping[str, jax.Array | np.ndarray | Coordinates],
                  valid: jax.Array | np.ndarray | None) -> dict:
    """`fields` holding `valid` as their validity where `token_validity` reads it, None removing it:
    in their `Coordinates` where the fields carry them, which the decoder reads in place of the
    validity field, else in the validity field."""
    laid = fields.get(COORDINATES_FIELD)
    if isinstance(laid, Coordinates):
        return {**fields, COORDINATES_FIELD: replace(laid, valid=valid)}
    rest = {name: value for name, value in fields.items() if name != VALIDITY_FIELD}
    return rest if valid is None else {**rest, VALIDITY_FIELD: valid}


def token_positions(fields: Mapping[str, jax.Array | np.ndarray | Coordinates], *, xp=jnp):
    """The logical positions `[B, S]` of a row's token fields as the decoder reads them: their
    `Coordinates`' where the fields carry them, else the `positions` field; None where neither gives
    any. `xp` is the array module."""
    laid = fields.get(COORDINATES_FIELD)
    positions = laid.positions if isinstance(laid, Coordinates) else fields.get("positions")
    return None if positions is None else xp.asarray(positions)


def real_slots(fields: Mapping[str, jax.Array | np.ndarray | Coordinates], shape: tuple[int, ...], *, xp=jnp):
    """Which of a request's `shape` `[B, S]` slots hold a real token: those its
    validity marks, every one where it has none, less the padding a packed
    row numbers 0 in its documents (`token_layout`). Samplers select, place
    and continue a row by these, and losses score and count them, so a row
    padded either way reads as the row alone; a caller's mask is narrowed,
    never replaced. `xp` is the array module."""
    valid, segments = token_layout(fields, xp=xp)
    real = xp.ones(shape, bool) if valid is None else valid
    return real if segments is None else real & (segments != 0)


def shifted_span(shape: tuple[int, ...], shift: int, *, valid=None, segments=None, xp=jnp):
    """Which `[B, S - shift]` source positions can reach the token `shift` places later: every
    intervening token is valid, including both endpoints, and their document IDs match. Padding
    IDs and target roles belong to the caller; `segments` describes identity, not validity.
    `xp` is NumPy for host geometry and JAX NumPy for model/loss execution."""
    batch, length = shape
    width = length - shift
    admitted = xp.ones((batch, width), bool)
    if valid is not None:
        for offset in range(shift + 1):
            admitted = admitted & xp.asarray(valid[:, offset:offset + width], bool)
    if segments is not None:
        admitted = admitted & (segments[:, :width] == segments[:, shift:shift + width])
    return admitted


def scored_targets(xp, tokens, pad_id: int | None, attention_mask, segment_ids, depth: int = 0,
                   roles=None, role: int | None = None, blocks=None):
    """Whether each target the loss scores at prediction `depth` counts, as
    `[B, S - 1 - depth]` booleans, in `xp` (numpy for the host's FLOP
    geometry and the data's target counts, jax.numpy for the loss): the
    state at p scores the token `depth + 1` later, which counts when it is
    not padding, when the attention mask keeps every position from the state
    to it, in a packed batch when the state and it share a document (segment
    0 is padding), and with a `role` (`loss_role`) when `roles`, one per
    token, give the target that role. With `blocks`, each token's block (-1 none), a target in its state's
    block does not count. Blocks are contiguous runs (`AttentionMetadata.block_groups`), so this is exact
    for the trunk's targets (`depth` 0): a prediction depth reads blocks beyond its own source, so
    `token_scores` refuses depths with blocks."""
    span = depth + 1
    counted = shifted_span(tokens.shape, span, valid=attention_mask, segments=segment_ids, xp=xp)
    if pad_id is not None:
        counted = counted & (tokens[:, span:] != pad_id)
    if segment_ids is not None:
        counted = counted & (segment_ids[:, span:] != 0)
    if role is not None and roles is not None:
        counted = counted & (roles[:, span:] == role)
    if blocks is not None:
        # The state's block holds the target (`dew.nn.visibility.same_group`, which this module sits below).
        source = blocks[:, :-span]
        counted = counted & ~((source == blocks[:, span:]) & (source >= 0))
    return counted


type Path = tuple[str, ...]


type PathFilter = Callable[[Path], bool]
"""A function that selects leaves of a variables tree by the tuple of dict keys above them.

One filter type is used for the EMA selection, for `optax.multi_transform`
labels and for frozen subtrees."""


def everything(path: Path) -> bool:
    return True


def under(*prefix: str) -> PathFilter:
    """Return a filter that accepts the leaves below `prefix`.

    For example, `under("params", "context_encoder")`.
    """
    return lambda path: path[:len(prefix)] == prefix


def select(tree: Variables, keep: PathFilter) -> Variables:
    """Return the subtree of `tree` whose leaves `keep` accepts, nested the same way.

    A branch with no accepted leaf is dropped, so the result is what the EMA
    stores and what `merge` puts back. Raises `ValueError` when `keep`
    accepts no leaf at all.
    """
    def prune(node, path):
        if isinstance(node, Mapping):
            kept = {name: prune(child, (*path, name)) for name, child in node.items()}
            return {name: child for name, child in kept.items() if child is not None} or None
        return node if keep(path) else None

    selected = prune(tree, ())
    if not selected:
        raise ValueError("the filter selected no leaf of the variables tree")
    return selected


def without(tree: Variables, paths: Sequence[Sequence[str]]) -> Variables:
    """`tree` without the subtree at each of `paths`, a path the tree does not hold left as it is."""
    kept = dict(tree)
    for path in paths:
        name, *rest = path
        if name in kept:
            if rest:
                held = kept[name]
                kept[name] = without(held, [rest]) if isinstance(held, Mapping) else held
            else:
                del kept[name]
    return kept


PARAMETER_ROOTS = (("params",), (FROZEN,))
"""The collections of a variables tree that hold a model's parameters, which a run's loaders read
and a storage dtype casts: the trained ones and the frozen."""


def freeze(variables: Variables, trainable: PathFilter) -> Variables:
    """Move the `params` leaves that `trainable` rejects into the `FROZEN` collection.

    `trainable` sees full leaf paths, `("params", ...)`. A tree already split
    keeps training the leaves its `params` hold, beside those `trainable`
    takes from either collection, and keeps the rest frozen: an adapter
    attached over another keeps both training. Raises `ValueError` when the
    split keeps every leaf or none, since then there is nothing to split.
    """
    training = (frozenset() if FROZEN not in variables else frozenset(
        ("params", *(entry.key for entry in path))
        for path, _ in jax.tree_util.tree_leaves_with_path(variables["params"])))
    variables = thaw(variables)

    def trains(path: Path) -> bool:
        return path in training or trainable(path)

    leaves = jax.tree_util.tree_leaves_with_path(variables["params"])
    kept = sum(trains(("params", *(entry.key for entry in path))) for path, _ in leaves)
    if kept in (0, len(leaves)):
        raise ValueError(
            f"trainable keeps {kept} of {len(leaves)} parameter leaves, which "
            f"{'freezes' if kept else 'trains'} nothing")
    moving = select(variables, lambda path: path[0] == "params" and trains(path))
    frozen = select(variables, lambda path: path[0] == "params" and not trains(path))
    return {**variables, "params": moving["params"], FROZEN: frozen["params"]}


def thaw(variables: Variables) -> Variables:
    """Undo `freeze`, merging the frozen leaves back into one `params` collection.

    Variables without a `FROZEN` collection are returned unchanged.
    """
    if FROZEN not in variables:
        return variables
    rest = {name: value for name, value in variables.items() if name != FROZEN}
    return {**rest, "params": merge(variables[FROZEN], variables["params"])}


def dicts[Tree](tree: Tree) -> Tree:
    """`tree` (a variables tree, or a state holding several) with every container mapping a dict: a tree
    that holds a mapping of the kinds Flax's `unfreeze` makes dicts of (a `FrozenDict`, a dict of
    another type, an `OrderedDict`) anywhere, under a mapping, a sequence or any other node, rebuilt
    over the same leaves with each one a dict, any other the caller's own tree. `unfreeze` itself stops
    at a list. A record that is a mapping of its own kind (an optimizer's `nnx.State`) stays as it is,
    its contents read the same way. The trees an objective holds, the state's and a step's
    written-back collections are dicts, so that a placement writing each placed leaf into the dict
    that held it (`dew.distributed.host.stream`) writes into the objective's tree, and every tree of a
    step has the state's structure."""
    nodes = jax.tree.leaves(
        tree, is_leaf=lambda node: isinstance(node, (dict, FrozenDict)) and type(node) is not dict)
    if not any(isinstance(node, (dict, FrozenDict)) and type(node) is not dict for node in nodes):
        return tree
    return jax.tree.map(
        lambda node: ({key: dicts(value) for key, value in node.items()}
                      if isinstance(node, (dict, FrozenDict)) and type(node) is not dict else node),
        tree, is_leaf=lambda node: isinstance(node, (dict, FrozenDict)) and type(node) is not dict)


def owned_dicts[Tree](tree: Tree) -> Tree:
    """`tree` as a state holds its trees: every mapping a dict (`dicts`), the containers the
    state's own, so nothing the state does to them reaches the caller's tree."""
    return jax.tree.map(lambda leaf: leaf, dicts(tree))


def part(variables: Variables, name: str) -> Variables:
    """Cut the `name` subtree out of every collection that holds one: one
    module's own tree out of an objective's that nests several under each
    collection, `params/policy` and `frozen/policy`, so an adapted module's
    base comes with its factors."""
    return {collection: subtree[name] for collection, subtree in variables.items() if name in subtree}


def leaves_by_path(tree: Variables) -> dict[tuple[str, ...], jax.Array]:
    """Every leaf of a variables subtree, by its dict-key path."""
    return {tuple(entry.key for entry in path): leaf
            for path, leaf in jax.tree_util.tree_leaves_with_path(tree)}


def sown_nodes(tree, *markers: str) -> list[tuple[tuple[str, ...], Mapping]]:
    """Find every node of a sown collection that holds all of `markers`, with
    its path: the module path that sowed it.

    A sown collection nests by module path, so what a layer sowed is the
    branch that names the keys it sowed under. The search stops at such a
    branch rather than descending into the arrays below it, and walks the
    children it does descend into in sorted key order.
    """
    found: list[tuple[tuple[str, ...], Mapping]] = []

    def visit(node, path: tuple[str, ...]) -> None:
        if not isinstance(node, Mapping):
            return
        if all(marker in node for marker in markers):
            found.append((path, node))
            return
        for key in sorted(node):
            visit(node[key], (*path, key))

    visit(tree, ())
    return found


__all__ = [
    "BLOCKS_FIELD",
    "COORDINATES_FIELD",
    "EXAMPLE_ROWS",
    "FROZEN",
    "PARAMETER_ROOTS",
    "VALIDITY_FIELD",
    "VALID_ROWS",
    "Batch",
    "Coordinates",
    "Path",
    "PathFilter",
    "Variables",
    "coordinates",
    "dicts",
    "document_ranks",
    "document_starts",
    "everything",
    "freeze",
    "leaves_by_path",
    "merge",
    "owned_dicts",
    "padded_rows",
    "part",
    "real_places",
    "real_slots",
    "scored_targets",
    "select",
    "shifted_span",
    "sown_nodes",
    "thaw",
    "token_id",
    "token_ids",
    "token_layout",
    "token_positions",
    "token_validity",
    "under",
    "valid_order",
    "with_validity",
    "without",
]
