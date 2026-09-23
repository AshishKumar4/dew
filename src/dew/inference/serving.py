"""Continuous batching for text: a slot scheduler over one resident KV cache.

`TextGeneration` runs a request as one program, prefill then a fixed scan
of decode trips, so a batch shares a prompt width and a budget, a finished
row keeps scanning, and nothing joins until the batch returns. `Server`
keeps the cache instead. It holds `slots` rows of `capacity` cache slots
each, admits queued requests into free rows, and runs one jitted step per
iteration over every row: the prompts admitted this iteration prefill, their
rows are written into the resident cache, and every occupied row draws one
token, all in the same executable. A row leaves when it draws EOS or spends
its budget, and its slot takes the next request. Each row carries its own
cache cursor, length and budget; the attention mask hides the free rows and
the filler the prompt bucket needed.

The model's attention indexes its cache by batch row (`open_kv_cache` writes
`cache[row, cursor]`), so a token stream where one row's prompt and another
row's decode token share a forward is not expressible: the step runs the
prompts as one forward at their own width and the decode trip as another,
and both live in one program. An iteration that admits nothing compiles to
the decode trip alone. The admission forward is sized `admission` rows at a
`SHAPE_BUCKETS` prompt width, so a server compiles one program per prompt
bucket plus the decode-only one.

The server is built from a task, `Server.from_task(task, slots=, capacity=)`,
so `dew.pipeline` stays the one way to load weights and a processor. Every
request runs the task's bound policy: its transform chain, its stopping
criteria beside EOS, greedy or sampled as the task says. A request draws
with the key it was submitted with, folded the way a one-row `TextGeneration`
call folds it, so a served request and the same request run alone draw the
same tokens. `submit` hands back a future for the request's `Generation`;
`step` runs one iteration, `run` steps until the queue and the rows are
empty, and calling the server with a batch of prompts does both.

Per step the host does the admission bookkeeping and one copy of the drawn
ids; sampling, the cache writes and the stopping test stay on the device.
The step reads the previous step's draws after dispatching the next one, so
the device does not wait for the host between steps; a row's exit reaches
the host one step after it happens, and its slot is refilled the step after.

Over a paged cache (`kv_cache=KVCache(page_size=...)`, `dew.nn.kv_cache`) the
rows share one pool of pages and `dew.inference.pages.Pages` keeps the
ledger. A request is seated once the pool holds its prompt and budget, and
its prompt runs as a forward over the pool through the row's page table,
continuing from the row's cursor. That makes two things possible the dense
cache cannot do: a prompt can prefill in pieces over several steps (`chunk`)
while the other rows keep drawing, and a request can start from the pages of
a prompt prefix an earlier request wrote (`prefix_cache`). A policy with a
grammar (`strategies.Sample(grammar)`) carries each row's automaton state
beside its cache. `DenseRows` and `PagedRows` are the two host sides of a
cache, `Dense` and `Paged` what the step program does with an admission.

Over weights placed on a mesh the server runs one program over every
device, its state placed through the rule table that places the weights
(`dew.nn.sharding.DEFAULT_RULES`). The slots split as a batch's rows do
(`activation_batch`: the data, expert and fsdp axes), so each group of
devices holds its own rows, its rows' dense cache or its part of the page
pool (`pages`), and draws for them; a request is seated in the group with
room for it. Tensor parallelism splits the heads, the mlp and the
vocabulary of every row a group holds; the cache keeps its heads where the
key and value projections leave them and the logits their vocabulary. Every
write of a row's state is mapped over the groups, so none of it crosses a
group.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import math
import time
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct
from jax.experimental import checkify
from jax.experimental.layout import Format, Layout
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.typing import ArrayLike

from dew.inference.pages import Pages
from dew.inference.tasks import (
    Processor,
    Request,
    TextGeneration,
    _bucket,
    _ceiling,
    _decoded,
    _prepared,
    _sized,
)
from dew.nn.inputs import ModelInputs, mesh_of, request_key
from dew.nn.kv_cache import (
    CURSOR,
    POOLED,
    TABLE,
    VALIDITY,
    KVCache,
    Layered,
    filled_slots,
    grouped,
    is_paged,
    leaf_name,
)
from dew.nn.sharding import SEQUENCE_AXIS, STAGE_AXIS, batch_axes, logical_spec
from dew.objectives.base import Variables
from dew.sampling import decoding
from dew.sampling.decoding import LogitsTransform, StepState, Stopping
from dew.sampling.guided import Grammar
from dew.sampling.strategies import DecoderState, Sample, draw
from dew.sampling.text import (
    Generation,
    Sampling,
    _operations,
    _prefill,
    _validated,
    prediction_depths,
    resolve,
)

Prompt = str | Sequence[int] | ArrayLike | ModelInputs
"""One request: text for the processor, one row of token ids, or one prepared row."""


def _row_groups(mesh: Mesh | None) -> int:
    """How many groups of rows a server over `mesh` keeps, one per share of the slots."""
    return 1 if mesh is None else math.prod(mesh.shape[axis] for axis in batch_axes(mesh))


@struct.dataclass
class Slots:
    """The device state of every slot.

    `decoder` holds the model's cache, `[slots, capacity, ...]` per leaf, and
    logits per row. A drawing row's cache holds every token but its last
    draw, which the next step feeds before it draws (`_advanced`), and its
    logits are the ones that draw came from; a row seated this step holds
    its prompt, whose logits score its first draw. `tokens` and `valid` are the
    `StepState` buffer the transforms and criteria read, `[slots, 2 *
    capacity]`: a row's prompt sits right-aligned in the first half and its
    draws start at `capacity`, so `prompt_width` is the capacity for every
    row whatever its prompt length. `step` counts a row's draws, `budget`
    caps them, `active` marks the rows still drawing, and `keys` holds each
    row's PRNG key data. `automaton` is each row's grammar state under a
    guided policy, zero (the start) on admission, and unread otherwise.
    """

    decoder: DecoderState
    tokens: jax.Array
    valid: jax.Array
    step: jax.Array
    budget: jax.Array
    active: jax.Array
    keys: jax.Array
    automaton: jax.Array


@struct.dataclass
class Admission:
    """The prompt pieces one step prefills, `[rows, width]`, and per row:
    the slot it fills (`slots` for an unused row, which the scatter drops),
    its budget and key data, the page table (width zero over a dense cache)
    and how many of its prompt tokens the cache already holds. `final`
    marks the rows whose piece ends their prompt: those start drawing this
    step, and `history` with `history_valid` is their whole prompt
    right-aligned in `capacity` slots, for the transforms to read."""

    prompts: ModelInputs
    slots: jax.Array
    budgets: jax.Array
    keys: jax.Array
    tables: jax.Array
    cursors: jax.Array
    final: jax.Array
    history: jax.Array
    history_valid: jax.Array


@struct.dataclass
class Draws:
    """What one step drew, per slot: the token, whether the row drew at
    all, whether a criterion stopped it, and both likelihoods."""

    token: jax.Array
    drawn: jax.Array
    stopped: jax.Array
    behavior: jax.Array
    raw: jax.Array


def _opened(model: nn.Module, params: Variables, pad_id: int, slots: int, capacity: int) -> Slots:
    """Every slot free: an allocated, empty cache and zeroed carries.

    A prefill over an all-invalid prompt has the carry's real structure,
    cache leaves and logits and hidden states alike; only its shapes are
    read, and zeros in them leave the cursors at zero and the validity
    false, which is a free row. The prefill's device checks come along in
    the shapes, since a paged store checks its writes.
    """
    ops = _operations(model, params, pad_id, prediction_depths(model))
    blank = ModelInputs(jnp.zeros((slots, 1), jnp.int32),
                        {"attention_mask": jnp.zeros((slots, 1), bool)})
    _, decoder = jax.eval_shape(checkify.checkify(lambda: _prefill(model, params, blank, ops)[0],
                                                  errors=checkify.user_checks))
    keys = jax.random.key_data(jax.random.key(0))
    return Slots(jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), decoder),
                 jnp.zeros((slots, 2 * capacity), jnp.int32),
                 jnp.zeros((slots, 2 * capacity), bool),
                 jnp.zeros((slots,), jnp.int32), jnp.zeros((slots,), jnp.int32),
                 jnp.zeros((slots,), bool),
                 jnp.zeros((slots, *keys.shape), keys.dtype),
                 jnp.zeros((slots,), jnp.int32))


def _placed(resident: jax.Array, incoming: jax.Array, rows: jax.Array, groups: int) -> jax.Array:
    """`incoming`'s rows written into `resident` at `rows`, group by group.

    The slots fall into `groups` equal groups in order, and so do the
    incoming rows; `rows` counts from the start of each row's own group,
    and a row at the group's size is dropped. Mapped over the groups, the
    group is a batch dimension of the scatter, which GSPMD splits wherever
    the slots split, with no collective (`dew.nn.kv_cache.write_cache`).

    A cache leaf from a prefill over a cache of the prompt's width is
    narrower than the resident leaf on its slot axis, the one axis whose
    size the model's `max_seq_len` sets; it lands in the first slots of
    its rows. Every other leaf is the same shape row for row.
    """
    window = tuple(slice(0, incoming.shape[axis]) if incoming.shape[axis] != resident.shape[axis]
                   else slice(None) for axis in range(1, incoming.ndim))

    def place(resident: jax.Array, incoming: jax.Array, rows: jax.Array) -> jax.Array:
        return resident.at[(rows, *window)].set(incoming, mode="drop")

    return jax.vmap(place)(*(grouped(leaf, 0, groups) for leaf in (resident, incoming, rows))
                           ).reshape(resident.shape)


def _seated(state: Slots, decoder: DecoderState, real: jax.Array, admission: Admission, groups: int) -> Slots:
    """`state` with the prefilled cache, and the rows whose prompt ended seated to draw."""
    capacity = state.tokens.shape[1] // 2
    finished = jnp.where(admission.final, admission.slots, state.step.shape[0] // groups)
    padding = ((0, 0), (0, capacity))

    def place(resident: jax.Array, incoming: jax.Array) -> jax.Array:
        return _placed(resident, incoming, finished, groups)

    fresh = jnp.zeros_like(admission.slots)
    return Slots(decoder,
                 place(state.tokens, jnp.pad(admission.history, padding)),
                 place(state.valid, jnp.pad(admission.history_valid, padding)),
                 place(state.step, fresh), place(state.budget, admission.budgets),
                 place(state.active, real & admission.final), place(state.keys, admission.keys),
                 place(state.automaton, fresh))


@dataclasses.dataclass(frozen=True)
class Dense:
    """Admission over a dense cache: whole prompts, each into its own rows, in `groups` groups."""

    groups: int = 1

    def prefilled(self, model: nn.Module, params: Variables, pad_id: int, state: Slots,
                  admission: Admission) -> tuple[DecoderState, jax.Array]:
        """The resident carry with the admitted prompts prefilled into their rows.

        The prompts run as their own forward over a fresh cache sized to
        their row count and bucket width, so the prefill's attention reads
        only the prompt's keys, not the resident capacity, which would cost
        a large share of the batch's wall time in f32 attention scores. A
        resident row's slots past the prompt keep a former occupant's keys,
        which the cache's validity hides, since the validity is written
        whole.
        """
        narrow = _sized(model, admission.prompts.tokens.shape[1])
        fresh, real = _prefill(narrow, params, admission.prompts,
                               _operations(narrow, params, pad_id, prediction_depths(narrow)))
        rows = admission.slots
        cache = jax.tree.map(
            lambda leaf: _placed(leaf, jnp.zeros((rows.shape[0], *leaf.shape[1:]), bool), rows, self.groups)
            if leaf.dtype == bool else leaf, state.decoder.cache)
        decoder = dataclasses.replace(state.decoder, cache=cache)
        return jax.tree.map(lambda resident, incoming: _placed(resident, incoming, rows, self.groups),
                            decoder, fresh), real


@dataclasses.dataclass(frozen=True)
class Paged:
    """Admission over a paged cache: prompt pieces, each continuing its row, in `groups` groups."""

    groups: int = 1

    def prefilled(self, model: nn.Module, params: Variables, pad_id: int, state: Slots,
                  admission: Admission) -> tuple[DecoderState, jax.Array]:
        """The resident carry with each admitted row's next prompt piece in the pool.

        The piece runs as a forward over a view of the resident cache: the
        shared pool itself, and per row the page table and cursor the host
        supplies. Its attention reads the row's earlier pieces and shared
        prefix pages through the table, and its keys land in the row's own
        pages. The per-row leaves and the logits are scattered back to the
        rows' slots.
        """
        def view(path: tuple[jax.tree_util.KeyEntry, ...], leaf: jax.Array) -> jax.Array:
            name = leaf_name(path)
            if name == TABLE:
                return admission.tables
            if name == CURSOR:
                return admission.cursors
            if name == VALIDITY:
                return filled_slots(admission.cursors, leaf.shape[1])
            return leaf

        fresh, real = _prefill(model, params, admission.prompts, _operations(model, params, pad_id, 0),
                               cache=jax.tree_util.tree_map_with_path(view, state.decoder.cache))

        def merged(path: tuple[jax.tree_util.KeyEntry, ...], resident: jax.Array,
                   updated: jax.Array) -> jax.Array:
            if leaf_name(path) in POOLED:
                return updated
            return _placed(resident, updated, admission.slots, self.groups)

        return jax.tree_util.tree_map_with_path(merged, state.decoder, fresh), real


Placement = Dense | Paged
"""How the step program writes an admission into the resident cache."""


def _advanced(model: nn.Module, params: Variables, pad_id: int, placement: Placement, state: Slots,
              admission: Admission | None, transforms: tuple[LogitsTransform, ...],
              stopping: tuple[Stopping, ...], grammar: Grammar | None) -> tuple[Slots, Draws]:
    """One iteration: admit, feed every drawing row its last draw, then every active row draws.

    This is `strategies._sample_rows`'s step over rows that carry their own
    budget, so a row that stops or spends its budget goes inactive while the
    rest keep drawing; an inactive row's cache is not written. A grammar
    guides the draws as `strategies.Sample` does. The step runs the model
    before it draws, so no device check comes before the model call: jax's
    checkify cannot carry a check into a shard_map (jax-ml/jax#40907), and a
    model on a mesh runs its expert dispatch in one. A token is therefore fed
    in the step after the one that drew it, and a row that feeds nothing,
    one seated this step, keeps the logits its prompt left.
    """
    if admission is not None:
        state = _seated(state, *placement.prefilled(model, params, pad_id, state, admission), admission,
                        placement.groups)
    ops = _operations(model, params, pad_id, prediction_depths(model))
    capacity = state.tokens.shape[1] // 2
    fed = state.active & (state.step > 0)
    last = jnp.take_along_axis(state.tokens, (capacity + state.step - 1)[:, None], axis=1)[:, 0]
    advanced = ops.advance(state.decoder, last, fed)
    decoder = dataclasses.replace(advanced, logits=jnp.where(fed[:, None], advanced.logits,
                                                             state.decoder.logits))
    view = StepState(state.tokens, state.valid, state.step, state.active,
                     jax.random.wrap_key_data(state.keys), prompt_width=capacity)
    chain = decoding.chain(transforms)
    token, behavior, raw = draw(view, decoder.logits,
                                chain if grammar is None else grammar.guiding(chain, state.automaton))
    committed = view.commit(token, state.active)
    stopped = state.active & decoding.criterion(stopping)(committed, token)
    drawn = state.active
    return (Slots(decoder, committed.tokens, committed.valid, committed.step, state.budget,
                  drawn & ~stopped & (committed.step < state.budget), state.keys,
                  state.automaton if grammar is None else grammar.advanced(state.automaton, token, drawn)),
            Draws(token, drawn, stopped, jnp.where(drawn, behavior, 0.0), jnp.where(drawn, raw, 0.0)))


def _split(state: Slots) -> tuple[Slots, Slots]:
    """Split the state into the matrices a step donates and the vectors it rewrites.

    XLA copies a donated buffer whose old value is still read after the output
    is written, which is true of every cursor, step count and active flag. So
    only the matrices are donated; the vectors get fresh buffers. Each half
    holds None where the other holds the leaf, and `_joined` recombines them.
    A `Formats` tree splits the same way, by the rank each layout describes.
    """
    def matrix(leaf: jax.Array | Format) -> bool:
        if isinstance(leaf, Format):
            return isinstance(leaf.layout, Layout) and len(leaf.layout.major_to_minor) >= 2
        return leaf.ndim >= 2

    return (jax.tree.map(lambda leaf: leaf if matrix(leaf) else None, state),
            jax.tree.map(lambda leaf: None if matrix(leaf) else leaf, state))


def _joined(resident: Slots, carried: Slots) -> Slots:
    return jax.tree.map(lambda held, other: other if held is None else held, resident, carried,
                        is_leaf=lambda leaf: leaf is None)


def _stepped(model: nn.Module, params: Variables, pad_id: int, placement: Placement, steps: int,
             resident: Slots, carried: Slots,
             admission: Admission | None, transforms: tuple[LogitsTransform, ...],
             stopping: tuple[Stopping, ...], grammar: Grammar | None
             ) -> tuple[checkify.Error, tuple[Slots, Slots, Draws]]:
    """Run `steps` iterations of `_advanced` over a split state, carrying their device checks as a value.

    The first iteration takes the admission; the draws come back stacked,
    `[steps, slots]` per leaf. `text._checked` carries the checks the same
    way. The host throws the error when it reads the draws, one call later.
    The state comes back split as it went in, so the host passes the halves
    straight to the next call.
    """

    def run(params, resident, carried, admission, transforms, stopping, grammar):
        state, draws = _advanced(model, params, pad_id, placement, _joined(resident, carried), admission,
                                 transforms, stopping, grammar)
        draws = jax.tree.map(lambda leaf: leaf[None], draws)
        if steps > 1:
            def following(state: Slots, _: None) -> tuple[Slots, Draws]:
                return _advanced(model, params, pad_id, placement, state, None, transforms, stopping, grammar)

            state, more = jax.lax.scan(following, state, length=steps - 1)
            draws = jax.tree.map(lambda first, rest: jnp.concatenate([first, rest]), draws, more)
        return *_split(state), draws

    return checkify.checkify(run, errors=checkify.user_checks)(
        params, resident, carried, admission, transforms, stopping, grammar)


Formats = Slots
"""A `Slots` whose leaves are `Format`s: the resident layout of each leaf."""


def _state_shardings(mesh: Mesh | None, state: Slots) -> Slots:
    """Where each leaf of the resident state lives: `state` with a sharding at every leaf.

    Every leaf holds one row per slot on axis zero (`activation_batch`),
    except a paged cache's pool, `[kv_heads, pages, page_size, ...]`, whose
    `pages` split over the same axes, one part per group of rows. Keys,
    values and their scales keep their heads where the key and value
    projections leave them (`activation_kv`), and the logits their
    vocabulary (`activation_vocab`). `state` is abstract: only its shapes
    are read.
    """
    if mesh is None:
        device = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        return jax.tree.map(lambda _: device, state)
    paged = is_paged(state.decoder.cache)

    def placed(leaf: jax.ShapeDtypeStruct | jax.Array, *names: str | None) -> NamedSharding:
        axes = (*names, *(None,) * (leaf.ndim - len(names)))
        return NamedSharding(mesh, logical_spec(axes, leaf.shape, mesh=mesh))

    def cached(path: tuple[jax.tree_util.KeyEntry, ...], leaf: jax.ShapeDtypeStruct | jax.Array) -> NamedSharding:
        if leaf_name(path) not in POOLED:
            return placed(leaf, "activation_batch")
        if paged:
            return placed(leaf, "activation_kv", "pages")
        return placed(leaf, "activation_batch", None, "activation_kv")

    rows = jax.tree.map(lambda leaf: placed(leaf, "activation_batch"), state)
    return rows.replace(decoder=rows.decoder.replace(
        cache=jax.tree_util.tree_map_with_path(cached, state.decoder.cache),
        logits=placed(state.decoder.logits, "activation_batch", "activation_vocab")))


def _compiled(resident: Formats, carried: Formats, rows: NamedSharding | None) -> jax.stages.Wrapped:
    """Compile the step program over a state split into resident and carried halves.

    The matrix half is donated so XLA updates the cache in place instead of
    copying it. `rows` places every leaf of an admission, whose rows come
    group by group, the way the slots are placed; None leaves them where they
    are. The step sets no XLA flag of its own; pass XLA_FLAGS to change the
    backend's defaults. See docs/performance.md for the measurements.
    """
    return jax.jit(_stepped, static_argnums=(0, 2, 3, 4), donate_argnums=(5,),
                   in_shardings=(None, resident, carried, rows, None, None, None),
                   out_shardings=(None, (resident, carried, None)))


def _resident_formats(model: nn.Module, params: Variables, pad_id: int, placement: Placement, steps: int,
                      state: Slots, shardings: Slots, rows: NamedSharding | None,
                      transforms: tuple[LogitsTransform, ...], stopping: tuple[Stopping, ...],
                      grammar: Grammar | None) -> Formats:
    """Choose the memory layout the resident state keeps, by asking XLA for it.

    The attention dot reads the cached keys and values with the slot axis
    minor. Across a jit boundary an array is row-major unless a layout is
    given, which would transpose the whole cache twice per layer. So the
    decode-only step is compiled once with layouts left to XLA, and the
    output formats it chose become the format every step takes and returns.
    `state` is abstract: choosing the layout allocates nothing. `shardings`
    places each leaf (`_state_shardings`).
    """
    resident, carried = _split(state)

    def automatic(half: Slots) -> Formats:
        return jax.tree.map(lambda leaf, sharding: None if leaf is None else Format(Layout.AUTO, sharding),
                            half, shardings, is_leaf=lambda leaf: leaf is None)

    program = _compiled(automatic(resident), automatic(carried), rows)
    compiled = program.lower(model, params, pad_id, placement, steps, resident, carried, None, transforms,
                             stopping, grammar).compile()
    return _joined(*compiled.output_formats[1][:2])


def _opened_in(formats: Formats) -> jax.stages.Wrapped:
    """`_opened`, allocating its state in `formats`."""
    return jax.jit(_opened, static_argnums=(0, 2, 3, 4), out_shardings=formats)


_PROGRAMS: dict[tuple[tuple[Format, ...], str, NamedSharding | None], jax.stages.Wrapped] = {}
"""One step program per resident layout; see `_program`."""


def _program(formats: Formats, rows: NamedSharding | None) -> jax.stages.Wrapped:
    """`_compiled` over `formats`, once per layout: one compile per (model,
    admission shape) serves every server that keeps its state in the same
    layout. The formats tree holds the cache mapping and is not hashable
    itself, so its leaves and its structure's text key the programs."""
    leaves, structure = jax.tree_util.tree_flatten(formats)
    key = (tuple(leaves), str(structure), rows)
    program = _PROGRAMS.get(key)
    if program is None:
        program = _PROGRAMS[key] = _compiled(*_split(formats), rows)
    return program


class Ticket(Future):
    """A submitted request's future `Generation`, with when it was
    submitted, admitted to a slot, when its first token reached the host
    and when it finished, in `time.perf_counter` seconds; `admitted`,
    `first` and `finished` are None until they happen. `first` less
    `submitted` is the time to first token."""

    def __init__(self) -> None:
        super().__init__()
        self.submitted = time.perf_counter()
        self.admitted: float | None = None
        self.first: float | None = None
        self.finished: float | None = None


@dataclass
class _Row:
    """One request on the host: its prompt, budget and key, and what it drew."""

    prompt: np.ndarray
    budget: int
    keys: np.ndarray
    ticket: Ticket
    tokens: list[int] = field(default_factory=list)
    behavior: list[float] = field(default_factory=list)
    raw: list[float] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    prefilled: int = 0
    """Prompt tokens the cache holds for the row: shared prefix pages and pieces run."""
    group: int = 0
    """The group of slots the row is seated in, whose part of a paged pool its pages are."""


def _seated_in(group: int, free: list[list[int]], taken: list[int], row: _Row) -> tuple[int, _Row]:
    """Seat `row` in `group`'s next free slot."""
    row.group = group
    slot = free[group][taken[group]]
    taken[group] += 1
    return slot, row


class DenseRows:
    """The host side of a dense cache: every slot owns `capacity` cache slots.

    A queued request takes a free slot in the group with the most of them,
    up to `admission` a step in each group, and prefills its whole prompt
    in one piece.
    """

    chunk = None
    width = 0
    """Pages in a row's table: a dense row has none."""

    def __init__(self, groups: int = 1) -> None:
        self.placement = Dense(groups)

    def check(self, cache: Variables) -> None:
        if is_paged(cache):
            raise ValueError("a dense server's model keeps a paged cache; serve it with its KVCache")

    def refuse(self, length: int, budget: int) -> None:
        """The capacity check `_validated` makes covers a dense row."""

    def seat(self, queue: deque[_Row], free: list[list[int]], admission: int) -> list[tuple[int, _Row]]:
        """Seat queued requests in order; `free` is each group's free slots."""
        seated: list[tuple[int, _Row]] = []
        taken = [0] * len(free)
        while queue:
            open_groups = [group for group, slots in enumerate(free)
                           if taken[group] < min(admission, len(slots))]
            if not open_groups:
                break
            group = max(open_groups, key=lambda group: len(free[group]) - taken[group])
            seated.append(_seated_in(group, free, taken, queue.popleft()))
        return seated

    def table(self, row: _Row) -> list[int]:
        return []

    def prefilled(self, row: _Row) -> None:
        """Nothing outlives a dense row's prompt."""

    def release(self, row: _Row) -> None:
        """A dense row's slot is its storage; freeing the slot frees it."""

    def reloaded(self) -> None:
        """A dense cache shares nothing across rows to invalidate."""


class PagedRows:
    """The host side of a paged cache: one `Pages` ledger per group hands out its part of the pool.

    A request is seated once a group's part holds its prompt and its whole
    budget, counting the prefix pages it shares, so a running row never
    waits for a page. Of the groups with a free slot it takes the one
    holding the longest cached prefix of its prompt, then the one with the
    most pages free. A prompt prefills in pieces of at most `chunk` tokens,
    one piece a step.
    """

    def __init__(self, pages: list[Pages], chunk: int | None, width: int) -> None:
        self.pages = pages
        self.chunk = chunk
        self.width = width
        """Pages in a row's table, capacity over the page size."""
        self.placement = Paged(len(pages))

    def check(self, cache: Variables) -> None:
        """Every cached leaf has to be one of paged attention's.

        A paged server moves rows by their page tables and cursors alone; a
        layer that keeps other per-row state (a recurrent mixer, latent
        attention) would need its rows moved as well, and has the dense
        server for that; a layer that kept a dense cache did not take the
        layout at all.
        """
        leaves = jax.tree_util.tree_leaves_with_path(cache)
        if not is_paged(cache):
            raise ValueError("the model did not take the paged layout: none of its layers keeps a page table")
        for path, _ in leaves:
            if leaf_name(path) not in POOLED | {TABLE, CURSOR, VALIDITY}:
                raise ValueError(f"a paged server needs every cached layer to be paged attention; "
                                 f"{jax.tree_util.keystr(path)} is not")

    def refuse(self, length: int, budget: int) -> None:
        part = self.pages[0]
        needed = -(-(length + budget) // part.size)
        if needed > part.count:
            where = "" if len(self.pages) == 1 else f" in each of its {len(self.pages)} groups"
            raise ValueError(f"the request needs {needed} pages; the pool holds {part.count}{where}")

    def seat(self, queue: deque[_Row], free: list[list[int]], admission: int) -> list[tuple[int, _Row]]:
        """Seat queued requests in order while some group's part of the pool covers the next one."""
        seated: list[tuple[int, _Row]] = []
        taken = [0] * len(free)
        while queue:
            row = queue[0]
            open_groups = sorted((group for group, slots in enumerate(free) if taken[group] < len(slots)),
                                 key=lambda group: (len(self.pages[group].shared(row.prompt)),
                                                    self.pages[group].available), reverse=True)
            for group in open_groups:
                reserved = self.pages[group].reserve(row.prompt, len(row.prompt) + row.budget)
                if reserved is not None:
                    row.pages, row.prefilled = reserved
                    break
            else:
                break
            queue.popleft()
            seated.append(_seated_in(group, free, taken, row))
        return seated

    def table(self, row: _Row) -> list[int]:
        return row.pages

    def prefilled(self, row: _Row) -> None:
        """The row's prompt keys are written by the step this admission joins."""
        self.pages[row.group].publish(row.prompt, row.pages)

    def release(self, row: _Row) -> None:
        self.pages[row.group].release(row.pages)

    def reloaded(self) -> None:
        for part in self.pages:
            part.forget()


Rows = DenseRows | PagedRows


class Server:
    """A slot scheduler over a resident KV cache; see the module docstring.

    Build one with `from_task`. `slots` rows run at once over `capacity`
    cache slots each; `admission` bounds the prompts one step prefills. A
    request whose prompt and budget do not fit the capacity is refused at
    `submit`, with the error `TextGeneration` raises for a request over the
    model's context. Over a mesh both counts split evenly over the groups
    of rows, and so do the pages of a paged pool.
    """

    def __init__(self, model: nn.Module, variables: Variables, processor: Processor | None, *,
                 sampling: Sampling, transforms: tuple[LogitsTransform, ...], stopping: tuple[Stopping, ...],
                 grammar: Grammar | None, rows: Rows, slots: int, capacity: int, admission: int,
                 default_budget: int | None, decode_steps: int) -> None:
        if type(slots) is not int or slots < 1:
            raise ValueError("slots must be a positive number of resident rows")
        if type(admission) is not int or not 1 <= admission <= slots:
            raise ValueError("admission must be between one and the slot count")
        if type(decode_steps) is not int or decode_steps < 1:
            raise ValueError("decode_steps must be a positive number of iterations per device call")
        self.mesh = mesh_of(variables)
        self.groups = _row_groups(self.mesh)
        if slots % self.groups or admission % self.groups:
            raise ValueError(f"slots ({slots}) and admission ({admission}) must divide by the "
                             f"{self.groups} groups the mesh splits rows into")
        self.model = model
        self.variables = variables
        self.processor = processor
        self.sampling = sampling
        self.pad_id = sampling.pad_id
        self.transforms = transforms
        self.stopping = stopping
        self.slots = slots
        self.capacity = capacity
        self.admission = admission
        self.default_budget = default_budget
        self.decode_steps = decode_steps
        self.steps = 0
        self._queue: deque[_Row] = deque()
        self._rows: dict[int, _Row] = {}
        self._pending: tuple[checkify.Error, Draws] | None = None
        self._failed: BaseException | None = None
        self.rows = rows
        self.grammar = grammar
        self.prefix_hits = 0
        """Prompt tokens served from shared prefix pages instead of prefilled."""
        self._admitted = None if self.mesh is None else NamedSharding(self.mesh, P(batch_axes(self.mesh) or None))
        with self._context():
            shapes = jax.eval_shape(functools.partial(_opened, model, pad_id=self.pad_id, slots=slots,
                                                      capacity=capacity), variables)
            rows.check(shapes.decoder.cache)
            if isinstance(rows, PagedRows) and prediction_depths(model):
                raise ValueError("a paged server runs no prediction depths; their cache is seeded "
                                 "over the whole prompt at once")
            formats = _resident_formats(model, variables, self.pad_id, rows.placement, decode_steps, shapes,
                                        _state_shardings(self.mesh, shapes), self._admitted, transforms,
                                        stopping, grammar)
            self._step = _program(formats, self._admitted)
            self._resident, self._carried = _split(
                _opened_in(formats)(model, variables, self.pad_id, slots, capacity))

    def _context(self) -> contextlib.AbstractContextManager[None]:
        """The mesh the model traces under, so a layer that reads it, such as an expert exchange, finds it."""
        return contextlib.nullcontext() if self.mesh is None else jax.set_mesh(self.mesh)

    @classmethod
    def from_task(cls, task: TextGeneration, *, slots: int, capacity: int,
                  admission: int | None = None, kv_cache: KVCache | None = None,
                  chunk: int | None = None, prefix_cache: bool = False, decode_steps: int = 1) -> Server:
        """A server over the task's model, weights, processor and policy.

        `capacity` rounds up to a shape bucket and may not exceed the
        model's context. The task's `n` has to be one and its strategy the
        row-wise sampler (with or without a grammar), since the server's loop
        is that sampler over rows that come and go.

        `kv_cache` replaces the model's cache layout (`dew.nn.kv_cache`). A
        paged layout pools every row's pages: `pages` bounds the memory,
        and a request is seated once the pool holds its prompt and budget,
        so short requests fit more rows than `pages * page_size /
        capacity`. Over a paged cache, `chunk` splits a prompt into pieces
        of at most that many tokens, one piece a step, so a long prompt
        does not stall the rows decoding beside it, and `prefix_cache`
        shares the pages of a prompt prefix an earlier request computed.

        Weights placed on a mesh are served on it: the slots, the admission
        and the pages split over the mesh's row axes, which have to divide
        them. Admission defaults to the largest multiple of the group count
        up to eight rows an iteration. A mesh with a stage or a sequence
        axis above one, or one over several processes, is refused.

        `decode_steps` runs that many iterations in each device call. The
        host launches a call's kernels one after another, which on a tensor
        axis costs about as long as the step computes, so the devices wait
        on the host; a call of several iterations pays that once. Requests
        are seated and draws reach the host at call boundaries: a request
        waits up to `decode_steps` iterations for its slot, and a slot a row
        leaves mid-call stays empty until the next call. The default
        admission seats `decode_steps` iterations' worth of rows a call, so
        the slots fill as fast. The draws are the same for any value.
        """
        mesh = mesh_of(task.variables)
        if mesh is not None:
            if mesh.shape[STAGE_AXIS] > 1:
                raise ValueError("a server decodes through the whole layer stack every step; the stage axis "
                                 "runs the stack as the training pipeline, which the decoder refuses to decode "
                                 "through (one token a step leaves no microbatches to pipeline). Serve on a "
                                 "mesh with stage=1")
            if mesh.shape[SEQUENCE_AXIS] > 1:
                raise ValueError("a server keeps each row's cached keys whole on the devices of its group, "
                                 "and a decoder refuses to decode under a sequence axis, which splits a "
                                 "sequence's positions. Serve on a mesh with sequence=1")
            if not NamedSharding(mesh, P()).is_fully_addressable:
                raise ValueError("a server schedules its rows from one process; a mesh over several "
                                 "processes would need them to agree on every step's admission")
        if task.n != 1:
            raise ValueError("a served request draws one continuation; submit a prompt once per draw")
        transforms, stopping, strategy = resolve(task.sampling, task.logits, task.stopping, task.strategy)
        if not isinstance(strategy, Sample):
            raise ValueError("a server runs the row-wise sampler; beam and speculative loops are batch-wide")
        if chunk is not None and (type(chunk) is not int or chunk < 1):
            raise ValueError("chunk must be a positive number of prompt tokens per piece")
        ceiling = _ceiling(task.model)
        if ceiling is None:
            raise ValueError("a server needs a model that declares max_seq_len for its cache")
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive number of cache slots per row")
        rounded = _bucket(capacity, 64)
        if rounded > ceiling:
            raise ValueError(f"a capacity of {capacity} rounds to {rounded}, over the model's max_seq_len of {ceiling}")
        model = _sized(task.model, rounded)
        if kv_cache is not None:
            if not any(entry.name == "kv_cache" for entry in dataclasses.fields(model)):
                raise ValueError(f"{type(model).__name__} declares no kv_cache layout to replace")
            model = model.clone(kv_cache=kv_cache)
        layout = model.kv_cache if isinstance(model, Layered) else KVCache()
        groups = _row_groups(mesh)
        rows: Rows
        if layout.page_size is None:
            if chunk is not None or prefix_cache:
                raise ValueError("chunked prefill and prefix caching run over a paged cache; "
                                 "serve with kv_cache=KVCache(page_size=...)")
            rows = DenseRows(groups)
        else:
            count = slots * (rounded // layout.page_size) if layout.pages is None else layout.pages
            model = model.clone(kv_cache=dataclasses.replace(layout, pages=count, groups=groups))
            rows = PagedRows([Pages(count // groups, layout.page_size, prefix_cache=prefix_cache)
                              for _ in range(groups)], chunk, rounded // layout.page_size)
        return cls(model, task.variables, task.processor, sampling=task.sampling,
                   transforms=transforms, stopping=stopping, grammar=strategy.grammar, rows=rows,
                   slots=slots, capacity=rounded,
                   admission=(max(groups, min(slots, 8 * decode_steps) // groups * groups) if admission is None
                              else admission),
                   default_budget=task.max_new_tokens, decode_steps=decode_steps)

    @property
    def cache(self) -> Variables:
        """The resident cache, updated in place by every step."""
        return _joined(self._resident, self._carried).decoder.cache

    @property
    def occupancy(self) -> int:
        """Rows the host knows to be running, one step behind the device."""
        return len(self._rows)

    @property
    def queued(self) -> int:
        return len(self._queue)

    def reload(self, variables: Variables) -> None:
        """Serve `variables` from the next step on, in place of the current weights.

        The tree must match the served one leaf for leaf in shape, and a
        floating leaf is cast to the served precision (train in float32,
        serve in bfloat16), so the compiled step runs on unchanged. The leaves
        are copied onto the served placement: the caller may donate or
        overwrite its own buffers right after this returns. Rows already running keep their cache and
        draw their next token from the new weights; a caller that stamps a
        policy version on a request takes the version it was submitted under.
        Prompt prefix pages the old weights wrote are no longer shared.
        Not thread-safe against `step`: the caller serializes the two.
        """
        # A frozen and a plain mapping flatten to different tree structures but
        # the same leaf paths, so the paths are what must match.
        incoming, _ = jax.tree_util.tree_flatten_with_path(variables)
        served, structure = jax.tree_util.tree_flatten_with_path(self.variables)
        if [path for path, _ in incoming] != [path for path, _ in served]:
            raise ValueError("reloaded variables must have the served tree structure")
        leaves = []
        for (_, new), (_, old) in zip(incoming, served, strict=True):
            kind, served_kind = jnp.result_type(new), jnp.result_type(old)
            if np.shape(new) != np.shape(old) or (kind != served_kind and not (
                    jnp.issubdtype(kind, jnp.floating) and jnp.issubdtype(served_kind, jnp.floating))):
                raise ValueError(
                    f"a reloaded leaf is {kind}{list(np.shape(new))}, "
                    f"the served leaf {served_kind}{list(np.shape(old))}")
            # A placement can alias a shard of the caller's array; the copy
            # owns its buffer whatever the caller donates next.
            placement = old.sharding if isinstance(old, jax.Array) else None
            leaves.append(jnp.array(jax.device_put(new, placement), dtype=served_kind, copy=True))
        self.variables = jax.tree.unflatten(structure, leaves)
        self.rows.reloaded()

    def submit(self, prompt: Prompt, max_new_tokens: int | None = None, *,
               key: jax.Array | None = None, seed: int | None = None) -> Ticket:
        """Queue one request; the ticket resolves to its `Generation`.

        The prompt is validated as `TextGeneration` validates it, against
        the server's capacity in place of the model's context. A zero budget
        resolves at once with the prompt alone.
        """
        # One row's key, folded the way `RowPlan.keys` folds row zero.
        return self._enqueued(prompt, max_new_tokens, jax.random.fold_in(request_key(key, seed), 0))

    def _enqueued(self, prompt: Prompt, max_new_tokens: int | None, key: jax.Array) -> Ticket:
        if self._failed is not None:
            raise RuntimeError("the server stopped after a device check failed") from self._failed
        row = self._prepared(prompt, max_new_tokens, key)
        if row.budget == 0:
            self._finish(row, terminated=False)
        else:
            self._queue.append(row)
        return row.ticket

    def _prepared(self, prompt: Prompt, max_new_tokens: int | None, key: jax.Array) -> _Row:
        request: Request = prompt if isinstance(prompt, (str, ModelInputs)) else np.atleast_2d(np.asarray(prompt))
        inputs = _prepared(self.processor, request, images=None)
        if inputs.tokens.shape[0] != 1:
            raise ValueError("submit takes one prompt; call the server with a batch")
        if set(inputs.token_fields) - {"attention_mask"} or inputs.conditioning:
            raise ValueError("a served prompt carries tokens and validity only; media and positions do not slot")
        budget = self.default_budget if max_new_tokens is None else max_new_tokens
        if budget is None:
            raise ValueError("max_new_tokens is required; the source declares no default budget")
        ids = np.asarray(inputs.tokens)
        fields = {name: np.asarray(value) for name, value in inputs.token_fields.items()}
        _validated(self.model, ids, fields, {}, budget, self.sampling, 1)
        valid = fields.get("attention_mask", np.ones(ids.shape, bool)).astype(bool)
        self.rows.refuse(int(valid[0].sum()), budget)
        return _Row(ids[0][valid[0]].astype(np.int32), budget, np.asarray(jax.random.key_data(key)), Ticket())

    def step(self) -> None:
        """One device call: admit what fits, run `decode_steps` iterations, read the last call's."""
        if self._failed is not None:
            raise RuntimeError("the server stopped after a device check failed") from self._failed
        admission = self._admit()
        with self._context():
            error, (self._resident, self._carried, draws) = self._step(
                self.model, self.variables, self.pad_id, self.rows.placement, self.decode_steps,
                self._resident, self._carried, admission, self.transforms, self.stopping, self.grammar)
        self.steps += self.decode_steps
        self._settle()
        self._pending = (error, draws)

    def run(self) -> None:
        """Step until every queued and running request has resolved."""
        while self._queue or self._rows:
            self.step()
        self._settle()

    def __call__(self, prompts: str | Sequence[str] | Sequence[Sequence[int]] | ModelInputs,
                 max_new_tokens: int | None = None, *, key: jax.Array | None = None,
                 seed: int | None = None) -> list[Generation]:
        """Submit a batch, run it through, and return its generations in order.

        Row `i` draws with the request key folded by `i`, as the same batch
        through `TextGeneration` would.
        """
        base = request_key(key, seed)
        inputs = _prepared(self.processor, prompts, images=None)
        valid = inputs.token_fields.get("attention_mask")
        rows = np.asarray(inputs.tokens)
        mask = np.ones(rows.shape, bool) if valid is None else np.asarray(valid).astype(bool)
        tickets = [self._enqueued(rows[index][mask[index]], max_new_tokens, jax.random.fold_in(base, index))
                   for index in range(rows.shape[0])]
        self.run()
        return [ticket.result() for ticket in tickets]

    def _admit(self) -> Admission | None:
        """Seat queued requests in free slots, then pick this step's prompt pieces.

        Rows still prefilling send their next piece, oldest first, each
        group's share of `admission` a step; a piece is the rest of the
        prompt, or at most `chunk` tokens when the rows prefill in pieces.
        The admission's rows come group by group, as the slots do, and each
        names its slot within its group.
        """
        now = time.perf_counter()
        size, share = self.slots // self.groups, self.admission // self.groups
        free = [[slot for slot in range(group * size, (group + 1) * size) if slot not in self._rows]
                for group in range(self.groups)]
        for slot, row in self.rows.seat(self._queue, free, share):
            self.prefix_hits += row.prefilled
            row.ticket.admitted = now
            self._rows[slot] = row
        pending: list[list[tuple[int, _Row]]] = [[] for _ in range(self.groups)]
        for slot, row in self._rows.items():
            waiting = pending[slot // size]
            if row.prefilled < len(row.prompt) and len(waiting) < share:
                waiting.append((slot, row))
        chosen = [(group * share + index, slot, row) for group, waiting in enumerate(pending)
                  for index, (slot, row) in enumerate(waiting)]
        if not chosen:
            return None
        width = _bucket(max(len(row.prompt) - row.prefilled for _, _, row in chosen), 64)
        width = width if self.rows.chunk is None else min(width, self.rows.chunk)
        count, capacity = self.admission, self.capacity
        tokens = np.zeros((count, width), np.int32)
        valid = np.zeros((count, width), bool)
        slots = np.full((count,), size, np.int32)
        budgets = np.zeros((count,), np.int32)
        keys = np.zeros((count, *chosen[0][2].keys.shape), chosen[0][2].keys.dtype)
        tables = np.zeros((count, self.rows.width), np.int32)
        cursors = np.zeros((count,), np.int32)
        final = np.zeros((count,), bool)
        history = np.zeros((count, capacity), np.int32)
        history_valid = np.zeros((count, capacity), bool)
        for index, slot, row in chosen:
            piece = row.prompt[row.prefilled:row.prefilled + width]
            tokens[index, width - len(piece):] = piece
            valid[index, width - len(piece):] = True
            slots[index], budgets[index], keys[index] = slot % size, row.budget, row.keys
            table = self.rows.table(row)
            tables[index, :len(table)] = table
            cursors[index] = row.prefilled
            row.prefilled += len(piece)
            final[index] = row.prefilled == len(row.prompt)
            history[index, capacity - len(row.prompt):] = row.prompt
            history_valid[index, capacity - len(row.prompt):] = True
            if final[index]:
                self.rows.prefilled(row)
        placed = jax.device_put([tokens, valid, slots, budgets, keys, tables, cursors, final, history,
                                 history_valid], self._admitted)
        return Admission(ModelInputs(placed[0], {"attention_mask": placed[1]}), *placed[2:])

    def _settle(self) -> None:
        """Read the previous call's draws into their rows, iteration by iteration; resolve the rows that ended."""
        if self._pending is None:
            return
        error, draws = jax.device_get(self._pending)
        self._pending = None
        try:
            error.throw()
        except BaseException as failure:
            self._fail(failure)
            raise
        now = time.perf_counter()
        for step in range(draws.drawn.shape[0]):
            for slot, row in list(self._rows.items()):
                if not draws.drawn[step, slot]:
                    continue
                if not row.tokens:
                    row.ticket.first = now
                row.tokens.append(int(draws.token[step, slot]))
                row.behavior.append(float(draws.behavior[step, slot]))
                row.raw.append(float(draws.raw[step, slot]))
                if draws.stopped[step, slot] or len(row.tokens) == row.budget:
                    del self._rows[slot]
                    self.rows.release(row)
                    self._finish(row, terminated=bool(draws.stopped[step, slot]))

    def _fail(self, failure: BaseException) -> None:
        self._failed = failure
        for row in [*self._rows.values(), *self._queue]:
            row.ticket.set_exception(failure)
        self._rows.clear()
        self._queue.clear()

    def _finish(self, row: _Row, *, terminated: bool) -> None:
        budget = row.budget
        drawn = np.full((1, budget), self.pad_id, np.int32)
        behavior = np.zeros((1, budget), np.float32)
        raw = np.zeros((1, budget), np.float32)
        count = len(row.tokens)
        drawn[0, :count] = row.tokens
        behavior[0, :count] = row.behavior
        raw[0, :count] = row.raw
        decoder = None if self.processor is None else functools.partial(_decoded, self.processor)
        row.ticket.finished = time.perf_counter()
        row.ticket.set_result(Generation(
            np.concatenate([row.prompt[None], drawn], axis=1), np.array([count], np.int32),
            np.array([terminated]), behavior, raw, rows=1, decoder=decoder))
