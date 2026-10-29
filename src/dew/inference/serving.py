"""Continuous batching for text: a slot scheduler over one resident KV cache.

`TextGeneration` runs a batch as one program, so its rows share a prompt
width and a budget and nothing joins until it returns. `Server` keeps the
cache instead: `slots` rows of `capacity` cache slots, refilled from a queue.
One jitted step per iteration prefills the prompts admitted this iteration
into their rows and draws one token for every occupied row; a row leaves on
EOS or at its budget, and its slot takes the next request. Each row carries
its own cursor, length and budget, and the attention mask hides free rows
and bucket filler.

The model's attention indexes its cache by batch row (`open_kv_cache` writes
`cache[row, cursor]`). Where the model says its layers run it
(`dew.nn.protocols.Serving`), over a dense cache or one page pool, an
admitting step is one forward over every token it runs: each row's last draw
and the admitted prompts, laid out in one row (`dew.nn.inputs.Admitted`), so
the projections read their weights once.
Otherwise (`Server.mixed_refusal` says why) the prompts run as one forward at
their own width and the decode trip as another, in one program. The prompts
are the fewest power-of-two rows that hold them (`admission_share`) at a
`SHAPE_BUCKETS` width capped at the capacity, so a server compiles one
program per prompt bucket plus the decode-only one.

`Server.from_task(task, slots=, capacity=)` builds it, so `dew.pipeline`
stays the one way to load weights and a processor, and every request runs
the task's bound policy. A request draws with its submitted key folded as a
one-row `TextGeneration` call folds it, so it draws the same tokens served
or alone, except where the model's products round at boundaries that its
programs' own roundings cross (a quantized decoder's moving weights, ledger
2032): there each draw is on par with the model's full forward, by check 1,
not equal to the lone task's. `submit` returns a future for its `Generation`, `step` runs one
iteration, `run` steps until queue and rows are empty, and calling the
server with a batch does both.

Per step the host does the admission bookkeeping and one copy of the drawn
ids, and reads a step's draws only after dispatching the next, so the device
never waits for it; a row's exit reaches the host a step late, and its slot
is refilled the step after. Submitting a request with an integer seed
launches nothing on the device: the admission's program makes its key.

Over a paged cache (`kv_cache=KVCache(page_size=...)`) the rows share one
page pool whose ledger is `dew.inference.pages.Pages`, and a prompt runs over
the pool through its row's page table from the row's cursor. A prompt can
then prefill in pieces while the other rows draw (`chunk`, which a dense
cache takes too where the step is mixed), and start from the pages of a
prefix an earlier request wrote (`prefix_cache`). A grammar
policy carries each row's automaton beside its cache. `DenseRows` and
`PagedRows` are the host sides of a cache, `Dense` and `Paged` the step's
admission.

This module is the host's scheduler. The device records and the programs it
feeds live in `dew.inference.serving_kernel`, and the packing of the weights
a decode step reads in `dew.inference.projections`.

On a mesh the server runs one program over every device, its state placed
by the weights' rule table (`dew.mesh.DEFAULT_RULES`). The slots split
as a batch's rows do (`activation_batch`), so each group of devices holds its
own rows and their cache or page share, draws for them, and seats requests
it has room for. Tensor parallelism splits the heads, mlp and vocabulary
within a group, and every write of a row's state is mapped over the groups,
so none of it crosses a group.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax.experimental import checkify
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.typing import ArrayLike

from dew._hashing import array_fingerprint
from dew.batch import Variables
from dew.inference.engine import Timed, normalized, reloaded, settled
from dew.inference.pages import Pages, States, prefix_hashes
from dew.inference.projections import pack_projections, packed_layout
from dew.inference.serving_kernel import (
    Admission,
    Dense,
    Draws,
    Paged,
    Placement,
    decode_programs,
    joined,
    mixed_refusal,
    opened,
    opened_in,
    resident_formats,
    row_state,
    split_donated,
    state_programs,
    state_shardings,
)
from dew.inference.tasks import (
    Processor,
    TextGeneration,
    cache_ceiling,
    cache_sized,
    generation_budget,
    prepared_inputs,
    resident,
    shape_bucket,
)
from dew.mesh import SEQUENCE_AXIS, STAGE_AXIS, batch_axes
from dew.nn.cross_attention import CrossBlocks
from dew.nn.inputs import (
    CONTEXT,
    CONTEXT_LENGTHS,
    CrossContext,
    ModelInputs,
    host_token_rows,
    mesh_of,
    request_key,
)
from dew.nn.kernels.options import compiler_options
from dew.nn.kv_cache import GROUPS, ROUNDING_KEYS, KVCache, is_paged, leaf_name
from dew.nn.protocols import CrossAttending, Layered, ProjectionGroup, Serving
from dew.nn.scatter import DROPPED
from dew.nn.visibility import with_served_blocks
from dew.sampling.decoding import Finishing, LogitsTransform, StopStrings
from dew.sampling.guided import Grammar
from dew.sampling.strategies import Sample
from dew.sampling.text import Generation, Sampling, check_inputs, decoded_rows, rebuild_position, resolve

Prompt = str | Sequence[int] | ArrayLike | ModelInputs
"""One request: text for the processor, one row of token ids, or one prepared row."""


def _row_groups(mesh: Mesh | None) -> int:
    """How many groups of rows a server over `mesh` keeps, one per share of the slots."""
    return 1 if mesh is None else math.prod(mesh.shape[axis] for axis in batch_axes(mesh))


_log = logging.getLogger(__name__)


def admission_share(waiting: int, share: int) -> int:
    """The rows a group's share of an admitting step is padded to: the
    smallest power of two that holds `waiting` pieces, at most `share`.

    The admitting program prefills every row it is given, so padding one
    arriving prompt to the whole admission prefilled eight. Each width is its
    own compiled program, one per power of two. Serving Qwen3-0.6B on an RTX
    4080 to requests arriving as a Poisson process, 32 slots at 32 a second
    went from 3216 to 3832 tokens a second, TTFT p50 536 to 15.5 ms and the
    p99 token gap 30.2 to 11.2 ms; 128 slots at 56 a second from 5320 to 7064,
    2217 to 34.6 ms and 83.5 to 23.4 ms. A narrower prefill's GEMMs run at
    other shapes, so its bits can differ from the padded one's, within
    tests/reference_error.py's bound (docs/performance.md).
    """
    return min(share, 1 << max(waiting - 1, 0).bit_length())


class Ticket(Timed):
    """A submitted request's future `Generation`, with when it was
    submitted, admitted to a slot, when its first token reached the host
    and when it finished, in `time.perf_counter` seconds; `admitted`,
    `first` and `finished` are None until they happen. `first` less
    `submitted` is the time to first token.

    `tokens` are the ids the request has drawn so far, replaced whole each
    time the server reads a step's draws back, so a caller streams the
    request from them; the last value is the generation's own, and it is
    set before the result.

    A request can be cancelled (`cancel`) while it waits in the queue: the
    server drops it when it would seat it. A seated request runs to its end
    and can no longer be cancelled."""

    def __init__(self) -> None:
        super().__init__()
        self.first: float | None = None
        self.tokens: tuple[int, ...] = ()
        self._token_callbacks: list[Callable[[Ticket], None]] = []

    def add_tokens_callback(self, callback: Callable[[Ticket], None]) -> None:
        """Call `callback(ticket)` on the serving thread after each read that
        adds to `tokens`. As with `add_done_callback`, an exception it raises
        is logged and the server goes on."""
        self._token_callbacks.append(callback)

    def _drew(self, tokens: Sequence[int]) -> None:
        self.tokens = tuple(tokens)
        for callback in self._token_callbacks:
            try:
                callback(self)
            except Exception as error:
                _log.error("exception calling a tokens callback for %r", self, exc_info=error)


_KEY_DATA = jax.eval_shape(lambda: jax.random.key_data(jax.random.key(0)))
"""The shape and dtype of a default key's data, read without a device."""


def _seed(key: int | jax.Array) -> int | jax.Array:
    """A request's key as a row holds it: an integer seed as given, for
    `_row_keys` to make into a key, or a key's data. `_row_keys` makes and
    reads every row's key with the default PRNG implementation, so a key of
    another is refused here, before the request queues."""
    if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
        return int(key)
    typed = request_key(key)
    if typed.dtype != jax.random.key_dtype():
        raise ValueError(f"a served request's key is {jax.random.key_impl(typed)}, and the server batches "
                         f"keys of the default implementation, {jax.config.jax_default_prng_impl}; pass an "
                         "integer seed or a key of that implementation")
    return jax.random.key_data(typed)


@jax.jit
def _row_keys(keys: tuple[jax.Array | np.ndarray, ...], seeds: jax.Array, seeded: jax.Array,
              folds: jax.Array) -> jax.Array:
    """One fixed-width admission's key data, without a device-to-host read:
    each row's key, `key(seed)` where it came as an integer seed, folded by
    the row's index in the batch it came in. A key made here has the bits
    an eager `jax.random.key(seed)` has, so a request draws as it would
    alone, and submitting it launched nothing (each submit's own small
    programs left the RTX 4080 idle between serving steps)."""
    def folded(row: jax.Array, fold: jax.Array) -> jax.Array:
        return jax.random.key_data(jax.random.fold_in(jax.random.wrap_key_data(row), fold))

    made = jax.vmap(lambda seed: jax.random.key_data(jax.random.key(seed)))(seeds)
    return jax.vmap(folded)(jnp.where(seeded[:, None], made, jnp.stack(keys)), folds)


@dataclass
class _Row:
    """One request on the host: its prompt, budget and key, and what it drew."""

    prompt: np.ndarray
    budget: int
    seed: int | jax.Array
    """The request's integer seed, or its key's data where it came as a key."""
    fold: int
    """What the row's key is folded by: its index in the batch it came in."""
    ticket: Ticket
    tokens: list[int] = field(default_factory=list)
    behavior: list[float] = field(default_factory=list)
    raw: list[float] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    prefilled: int = 0
    """Prompt tokens the cache holds for the row: shared prefix pages and pieces run."""
    weights: int = 0
    """The weights the row's prompt keys are written under: how many reloads came before its seat."""
    group: int = 0
    """The group of slots the row is seated in, whose part of a paged pool its pages are."""
    context: np.ndarray | None = None
    """The real tokens of the context a cross-attention decoder reads beside the prompt,
    `[length, features]`; None for none. Every piece of the prompt carries it."""
    salt: bytes = b""
    """What the prompt's keys depend on beside its tokens, its context's digest and its groups with the
    tokens up to its last grouped one, under which its prefix pages are named (`dew.inference.pages`);
    empty for none."""
    restore: int | None = None
    """The `States` entry the row's slot starts from after its shared prefix pages, before its first piece;
    None where it starts afresh."""
    snapshots: tuple[int, ...] = ()
    """The prompt lengths, page boundaries past its shared prefix, after which the row's state is kept for a
    later prompt that shares its pages (`PagedRows.boundaries`)."""
    groups: dict[str, np.ndarray] = field(default_factory=dict)
    """The groups of the prompt's tokens each group field the model reads holds (`dew.nn.kv_cache.GROUPS`),
    -1 for a token in none: the groups the prompt carries, and the blocks the server's prefix run makes
    where it carries none (`with_served_blocks`). Empty where the row reads its prompt causally."""


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
    in one piece, or in pieces of at most `chunk` tokens, one a step, where
    the server runs the mixed admitting step.
    """

    width = 0
    """Pages in a row's table: a dense row has none."""

    def __init__(self, groups: int = 1, chunk: int | None = None) -> None:
        self.chunk = chunk
        self.placement: Placement = Dense(groups, continuing=chunk is not None)

    def check(self, cache: Variables) -> None:
        if is_paged(cache):
            raise ValueError("a dense server's model keeps a paged cache; serve it with its KVCache")

    def refuse(self, length: int, budget: int) -> None:
        """The capacity check `check_inputs` makes covers a dense row."""

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

    def __init__(self, pages: list[Pages], chunk: int | None, width: int, *, states: int = 0,
                 every: int | None = None) -> None:
        self.pages = pages
        self.chunk = chunk
        self.width = width
        """Pages in a row's table, capacity over the page size."""
        self.states = states
        """The row states each group keeps for a shared prefix where its rows hold state beside pages."""
        self.every = every
        """The pages between the states a prompt keeps; None keeps its deepest shared boundary's alone."""
        self.placement: Placement = Paged(len(pages), continuing=chunk is not None
                                          or any(part.prefix_cache for part in pages))
        self.weights = 0
        """How many reloads the rows have seen; a row seated before the last one keeps older keys."""

    def check(self, cache: Variables) -> None:
        """The pool holds paged attention's keys and values, and every other
        cached leaf is a row's own state (a recurrent mixer's memory, latent
        attention's index keys), kept in the row's slot (`row_state`). A shared
        prefix's pages hold its keys alone, so where rows hold state and share
        prefixes, each group keeps the states prompts left at their page
        boundaries (`dew.inference.pages.States`), and a row reuses a prefix as
        deep as one is kept, starting from it. A cross-attention layer's
        context is not left by the prompt: every piece carries it and writes
        it again, and the pages are named under its digest."""
        if not is_paged(cache):
            raise ValueError("the model did not take the paged layout: none of its layers keeps a page table")
        if any(part.prefix_cache for part in self.pages) and any(
                row_state(path) for path, _ in jax.tree_util.tree_leaves_with_path(cache)):
            for part in self.pages:
                part.states = States(self.states)

    @property
    def stateful(self) -> bool:
        """Whether a shared prefix is read with the state its prompt left (`check`)."""
        return self.pages[0].states is not None

    def boundaries(self, row: _Row, size: int) -> tuple[int, ...]:
        """The page boundaries past the row's shared prefix after which its state is kept: its deepest
        shareable one, the pages before its last token, and every `every` pages below it."""
        deepest = (len(row.prompt) - 1) // size
        kept = {deepest, *(range(self.every, deepest, self.every) if self.every else ())}
        return tuple(sorted(page * size for page in kept if page * size > row.prefilled))

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
                                 key=lambda group: (len(self.pages[group].shared(row.prompt, row.salt)),
                                                    self.pages[group].available), reverse=True)
            for group in open_groups:
                part = self.pages[group]
                reserved = part.reserve(row.prompt, len(row.prompt) + row.budget, row.salt)
                if reserved is not None:
                    row.pages, row.prefilled, entry = reserved
                    row.restore = None if entry is None else group * self.states + entry
                    row.weights = self.weights
                    if part.states is not None:
                        row.snapshots = self.boundaries(row, part.size)
                    break
            else:
                break
            queue.popleft()
            seated.append(_seated_in(group, free, taken, row))
        return seated

    def table(self, row: _Row) -> list[int]:
        return row.pages

    def prefilled(self, row: _Row) -> None:
        """The row's prompt keys are written by the step this admission joins: its pages are shared
        where every key came from the weights served now, and not where it was seated before a
        reload and its earlier pieces kept the old weights' keys."""
        if row.weights == self.weights:
            self.pages[row.group].publish(row.prompt, row.pages, row.salt)

    def release(self, row: _Row) -> None:
        self.pages[row.group].release(row.pages)

    def kept(self, row: _Row) -> int | None:
        """The `States` entry, global over the groups, that keeps the row's state after its prompt's first
        `row.prefilled` tokens, a boundary its piece ended at; None where one is held already, or the
        row's keys came from weights a reload replaced."""
        part = self.pages[row.group]
        if part.states is None or row.weights != self.weights:
            return None
        pages = row.prefilled // part.size
        entry = part.states.take(prefix_hashes(row.prompt, part.size, pages, row.salt)[-1])
        return None if entry is None else row.group * self.states + entry

    def reloaded(self) -> None:
        self.weights += 1
        for part in self.pages:
            part.forget()


Rows = DenseRows | PagedRows


def _refuse_unserved_modes(model: nn.Module, *, paged: bool = False, chunked: bool = False,
                           prefix: bool = False, mixed: bool = False) -> None:
    """Refuse the serving modes that would change what `model` computes: a prompt prefilled in
    pieces or after a reused prefix where the model takes each prompt whole
    (`Serving.split_prefill_refusal`), and a paged or mixed cache where LongRoPE rebuilds a crossing
    request's (`rebuild_position`)."""
    whole = model.split_prefill_refusal() if isinstance(model, Serving) else None
    split = [name for name, active in (('chunked prefill', chunked), ('prefix reuse', prefix)) if active]
    if whole is not None and split:
        raise ValueError(f'{" and ".join(split)} would change what this model computes: {whole}')
    position = rebuild_position(model)
    rebuilt = [name for name, active in (('paged', paged), ('mixed admission', mixed)) if active]
    if position is not None and rebuilt:
        raise ValueError(f'LongRoPE crossing position {position} cannot rebuild {", ".join(rebuilt)} '
                         'serving; use whole-prompt dense admission')


class Server:
    """Serves text requests with continuous batching over one resident KV cache.

    A `TextGeneration` call runs its batch as one program, so no request can
    join until it returns. A server keeps `slots` rows of `capacity` cache
    slots each and refills them from a queue. Each step prefills the prompts
    admitted that step into free rows and draws one token for every occupied
    row. A row leaves where the task's criteria end it (EOS, a stop string) or
    at its budget, and the next request takes its slot. Every request runs
    the task's policy and draws the same tokens it would draw through the
    task alone with the same key, except where the model's products round at
    boundaries that the two programs' roundings cross (the module docstring).

    Build one with `from_task`. `admission` bounds the prompts one step
    prefills. A request whose prompt and budget do not fit the capacity is
    refused at `submit`, with the error `TextGeneration` raises for a request
    over the model's context. Over a mesh, the slots and the admission split
    evenly over the groups of rows, and so do the pages of a paged pool. The
    server holds the weights at their decoding storage
    (`dew.interop.weights.decoding_storage`), to which `reload` casts the
    weights it is handed.
    """

    def __init__(self, model: nn.Module, variables: Variables, processor: Processor | None, *,
                 sampling: Sampling, transforms: tuple[LogitsTransform, ...], stopping: Finishing,
                 grammar: Grammar | None, rows: Rows, slots: int, capacity: int, admission: int,
                 default_budget: int | None, max_length: int | None, decode_steps: int,
                 prefix: int | Literal["prompt"] | None = None) -> None:
        if type(slots) is not int or slots < 1:
            raise ValueError("slots must be a positive number of resident rows")
        if type(admission) is not int or not 1 <= admission <= slots:
            raise ValueError("admission must be between one and the slot count")
        _refuse_unserved_modes(
            model, paged=isinstance(rows, PagedRows), chunked=rows.chunk is not None,
            prefix=isinstance(rows, PagedRows) and any(part.prefix_cache for part in rows.pages),
            mixed=rows.placement.mixed)
        if type(decode_steps) is not int or decode_steps < 1:
            raise ValueError("decode_steps must be a positive number of iterations per device call")
        variables = resident(model, variables)
        self.mesh = mesh_of(variables)
        self.groups = _row_groups(self.mesh)
        if slots % self.groups or admission % self.groups:
            raise ValueError(f"slots ({slots}) and admission ({admission}) must divide by the "
                             f"{self.groups} groups the mesh splits rows into")
        self.model = model
        self.cross: tuple[CrossBlocks, int, np.dtype] | None = None
        """The model's cross-attention layers and its context's width and dtype, where a request may
        carry a context beside its prompt."""
        if isinstance(model, CrossAttending) and model.cross_attention is not None:
            self.cross = (model.cross_attention, model.context_width, np.dtype(model.context_dtype))
        self.variables = variables
        self._weight_groups: list[ProjectionGroup] = []
        self.processor = processor
        if sampling.stop and not any(isinstance(criterion, StopStrings) for criterion in stopping.ending):
            raise ValueError("a server takes stop strings compiled into stopping; build it with "
                             "Server.from_task, which compiles the task's")
        self.pad_id = sampling.pad
        self.sampling = dataclasses.replace(sampling, pad_token_id=self.pad_id)
        self.transforms = transforms
        self.stopping = stopping
        self.slots = slots
        self.capacity = capacity
        self.admission = admission
        self.default_budget = default_budget
        self.max_length = max_length
        self.prefix: int | Literal["prompt"] | None = prefix
        self.decode_steps = decode_steps
        self.steps = 0
        self._queue: deque[_Row] = deque()
        """The requests waiting for a slot, oldest first. `submit` appends on
        the right while a step seats from the left, so a submitting thread
        never waits for a step (a deque's append and popleft are atomic)."""
        self._ingress = threading.Lock()
        """Orders a submission's append with a failure's sweep."""
        self.arrived: Callable[[], None] = _nothing
        """Called after each submission: an `Engine` serving this kind wakes to it
        (`dew.inference.engine.Kind`)."""
        self._rows: dict[int, _Row] = {}
        self._pending: tuple[checkify.Error, Draws] | None = None
        self._failed: BaseException | None = None
        self.rows = rows
        self.grammar = grammar
        self.prefix_hits = 0
        """The number of prompt tokens served from shared prefix pages instead of prefilled."""
        self._admitted = (
            None if self.mesh is None else NamedSharding(self.mesh, P(batch_axes(self.mesh) or None))
        )

        with self._context():
            groups, source_shapes = packed_layout(model, variables)
            self._weight_groups.extend(groups)
            self._source_shapes = jax.tree_util.tree_flatten_with_path(source_shapes)[0]
            shapes = jax.eval_shape(functools.partial(opened, model, pad_id=self.pad_id, slots=slots,
                                                      capacity=capacity), variables)
            rows.check(shapes.decoder.cache)
            held = {leaf_name(path) for path, _ in jax.tree_util.tree_leaves_with_path(shapes.decoder.cache)}
            self.group_fields = frozenset(name for name, leaf in GROUPS.items() if leaf in held)
            """The group fields some layer reads whole (`dew.nn.kv_cache.GROUPS`), whose groups its cache
            keeps beside the keys: the ones a served prompt's slot beside its tokens."""
            self.keyed = ROUNDING_KEYS in held
            """Whether a layer's decode cache holds its rows' rounding keys (`dew.nn.kv_cache.ROUNDING_KEYS`),
            so a prompt's state follows its request's key, which then names its prefix too (`_prepared`)."""
            # None selects the one mixed forward of `_mixed_step`.
            self.mixed_refusal = mixed_refusal(model, shapes, rows.placement)
            """Why the admitting step prefills in a separate forward, or None when it runs one mixed forward.

            The mixed forward covers each row's last draw and the admitted
            prompts in one pass, so the projections read their weights once.
            """
            if self.mixed_refusal is None:
                rows.placement = dataclasses.replace(rows.placement, mixed=True)
            elif isinstance(rows, DenseRows) and rows.chunk is not None:
                raise ValueError(f"chunked prefill over a dense cache runs in the mixed admitting step, "
                                 f"which this model cannot take: {self.mixed_refusal}")
            else:
                _log.info("the admitting step prefills in a forward of its own: %s", self.mixed_refusal)
            formats, chosen = resident_formats(
                model, self.variables, self.pad_id, rows.placement, decode_steps, shapes,
                state_shardings(self.mesh, shapes), self._admitted, transforms, stopping, grammar)
            self._step, self._admitting = decode_programs(formats, self._admitted, compiler_options(model))
            if chosen is not None:
                # The compile that chose the layout is the decoding step: compiling it again at the
                # formats would repeat the server's largest compile (`resident_formats`).
                self._step = chosen
            self._resident, self._carried = split_donated(
                opened_in(formats)(model, self.variables, self.pad_id, slots, capacity))
            self._keep_states(formats)

    def _keep_states(self, formats) -> None:
        """Open the store of row states a shared prefix is read with, where the rows keep state beside
        their pages (`PagedRows.stateful`), and the programs that move them (`state_programs`)."""
        self._states: Variables | None = None
        """Each row state leaf (`row_state`) with its slot axis holding the groups' `States` entries,
        placed as the slots are; None where rows keep none."""
        self._restores: list[tuple[int, int]] = []
        """The (slot, entry) pairs the next call restores before it runs: rows seated after a prefix, up
        to every slot, whether or not their first piece goes in that call."""
        self._captures: list[tuple[int, int]] = []
        """The (slot, entry) pairs whose state the last call left at a kept boundary."""
        rows = self.rows
        if isinstance(rows, PagedRows) and rows.stateful:
            entries = rows.states * self.groups
            states = jax.tree_util.tree_map_with_path(
                lambda path, leaf: jax.device_put(jnp.zeros((entries, *leaf.shape[1:]), leaf.dtype),
                                                  leaf.sharding) if row_state(path) else None,
                self.cache)
            self._states = states
            self._restored, self._captured = state_programs(formats, states)

    def _context(self) -> contextlib.AbstractContextManager[None]:
        """The mesh the model traces under, so a layer that reads it, such as an expert exchange, finds it."""
        return contextlib.nullcontext() if self.mesh is None else jax.set_mesh(self.mesh)

    @classmethod
    def from_task(cls, task: TextGeneration, *, slots: int, capacity: int,
                  admission: int | None = None, kv_cache: KVCache | None = None,
                  chunk: int | None = None, prefix_cache: bool = False, prefix_states: int | None = None,
                  prefix_state_every: int | None = None, decode_steps: int = 1) -> Server:
        """Build a server over the task's model, weights, processor and policy.

        `capacity` is rounded up to whole 64-slot tiles and whole pages, and
        must stay within the model's context. The task's `n` must be one, and
        its strategy must be the row-wise sampler, with or without a grammar.

        `kv_cache` replaces the model's cache layout (`dew.nn.kv_cache`). A
        paged layout pools the pages of every row, so short requests fit more
        rows than `pages * page_size / capacity`. `chunk` prefills a prompt in
        pieces of at most that many tokens, one piece a step, so a long prompt
        does not stall the rows beside it. It also works over a dense cache
        where the admitting step is mixed. `prefix_cache` lets a request reuse
        the pages of a prompt prefix an earlier request computed, and needs a
        paged cache. Where the model keeps a row's own state beside its pages
        (a recurrent mixer's memory, latent attention's index keys), a reused
        prefix is read with the state its prompt left at its end: each group
        keeps `prefix_states` of them (one a slot by default), taken at a
        prompt's deepest shareable page boundary and every
        `prefix_state_every` pages below it, a piece cut to end there, and a
        request reuses pages as deep as a state is kept
        (`dew.inference.pages.States`).

        Weights on a mesh are served on that mesh. The slots, the admission
        and the pages are split over its row axes, so the number of row
        groups must divide the slots and the admission. The admission
        defaults to the largest multiple of the group count up to eight rows
        an iteration. A stage or sequence axis larger than one, or a mesh
        over several processes, is refused.

        `decode_steps` runs that many iterations per device call, so the host
        launches the step's kernels once per call; on a tensor axis those
        launches take about as long as the step. Requests are seated, and
        draws read, only at call boundaries, so a request waits up to
        `decode_steps` iterations for its slot. The default admission seats
        that many iterations' rows per call. The draws are the same for any
        value. More iterations pay off where the host's launches leave the
        devices idle, as across an NVLink pair at 128 slots
        (docs/concepts/inference.md). On one RTX 4080, which was already
        busy, more iterations were slower at 32 and 128 slots, and at 64
        slots only two iterations gained, by under 1%. A call seats requests
        only before its first iteration, so the slots take more iterations
        to fill (docs/performance.md).
        """
        mesh = mesh_of(task.variables)
        if mesh is not None:
            if mesh.shape.get(STAGE_AXIS, 1) > 1:
                raise ValueError(
                    "a server decodes through the whole layer stack every step; the stage axis "
                    "runs the stack as the training pipeline, which the decoder refuses to decode "
                    "through (one token a step leaves no microbatches to pipeline). Serve on a "
                    "mesh with stage=1"
                )
            if mesh.shape.get(SEQUENCE_AXIS, 1) > 1:
                raise ValueError("a server keeps each row's cached keys whole on the devices of its group, "
                                 "and a decoder refuses to decode under a sequence axis, which splits a "
                                 "sequence's positions. Serve on a mesh with sequence=1")
            if not NamedSharding(mesh, P()).is_fully_addressable:
                raise ValueError("a server schedules its rows from one process; a mesh over several "
                                 "processes would need them to agree on every step's admission")
        if task.n != 1:
            raise ValueError("a served request draws one continuation; submit a prompt once per draw")
        policy, chain, criteria = task._controls(None, None, None)
        transforms, stopping, strategy = resolve(policy, chain, criteria, task.strategy)
        if not isinstance(strategy, Sample):
            raise ValueError("a server runs the row-wise sampler; beam and speculative loops are batch-wide")
        if chunk is not None and (type(chunk) is not int or chunk < 1):
            raise ValueError("chunk must be a positive number of prompt tokens per piece")
        for name, value in (("prefix_states", prefix_states), ("prefix_state_every", prefix_state_every)):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        ceiling = cache_ceiling(task.model)
        if ceiling is None:
            raise ValueError("a server needs a model that declares max_seq_len for its cache")
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive number of cache slots per row")
        model = task.model
        if kv_cache is not None:
            if not isinstance(model, Layered):
                raise ValueError(f"{type(model).__name__} declares no kv_cache layout to replace")
            model = model.with_kv_cache(kv_cache)
        layout = model.kv_cache if isinstance(model, Layered) else KVCache()
        # One program serves one capacity, so the cache holds whole tiles of it
        # rather than a power-of-two bucket: every decode step's attention reads
        # each slot, and a capacity of 384 bucketed to 512 read a third more.
        unit = math.lcm(64, layout.page_size or 64)
        rounded = -(-capacity // unit) * unit
        if rounded > ceiling:
            raise ValueError(
                f"a capacity of {capacity} rounds to {rounded}, over the model's max_seq_len of {ceiling}"
            )
        groups = _row_groups(mesh)
        model, rows = _cache_rows(cache_sized(model, rounded), layout, slots, rounded, groups, chunk,
                                  prefix_cache, prefix_states, prefix_state_every)
        return cls(
            model,
            task.placed,
            task.processor,
            # The policy keeps the stop strings `stopping` compiled, so a server's draws are read
            # against what ends them (`dew.inference.rollouts.check_ending`).
            sampling=dataclasses.replace(policy, stop=task.sampling.stop),
            transforms=transforms,
            stopping=stopping,
            grammar=strategy.grammar,
            rows=rows,
            slots=slots,
            capacity=rounded,
            admission=(
                max(groups, min(slots, 8 * decode_steps) // groups * groups)
                if admission is None
                else admission
            ),
            default_budget=task.max_new_tokens,
            max_length=task.max_length,
            decode_steps=decode_steps, prefix=task.prefix,
        )

    @property
    def cache(self) -> Variables:
        """The resident cache, updated in place by every step."""
        return joined(self._resident, self._carried).decoder.cache

    @property
    def occupancy(self) -> int:
        """The number of rows the host knows to be running, one step behind the device."""
        return len(self._rows)

    @property
    def queued(self) -> int:
        return len(self._queue)

    @property
    def busy(self) -> bool:
        """Whether a request waits or runs, or a call's draws wait to be read."""
        return bool(self._queue or self._rows or self._pending is not None)

    def reload(self, variables: Variables) -> None:
        """Serve `variables` from the next step on, in place of the current weights.

        The tree must match the served one leaf for leaf in shape, and a leaf
        that is not floating point must match its dtype too. A floating leaf
        is cast to the served precision, so the compiled step keeps running
        without a recompile. The leaves are copied onto the served placement,
        so the caller may donate its buffers right after. Running rows keep
        their cache and draw their next token from the new weights, and
        prefix pages the old weights wrote are no longer shared. `reload` is
        not thread-safe against `step`, so the caller must serialize the two.
        """
        packed = pack_projections(normalized(variables, self._source_shapes), self._weight_groups)
        self.variables = reloaded(packed, self.variables)
        self.rows.reloaded()

    def submit(self, prompt: Prompt, max_new_tokens: int | None = None, *,
               key: int | jax.Array) -> Ticket:
        """Queue one request and return a ticket that resolves to its `Generation`.

        `prompt` is one text prompt, `ModelInputs` row or token row, without
        media or noncanonical positions; a `ModelInputs` row may carry the
        context a cross-attention decoder reads (`dew.nn.inputs.CONTEXT`) and
        the image and block groups its layers read whole
        (`dew.nn.kv_cache.GROUPS`), which slot beside its tokens. It is
        validated as `TextGeneration` validates it, against the server's
        capacity in place of the model's context.
        Without `max_new_tokens`, the task's default budget applies, its
        `max_new_tokens` or what its `max_length` leaves past the prompt. A
        request with a zero budget resolves at once, with the prompt alone.
        Another thread may submit while one steps. The request is validated
        here and queued at once; a step seats it when a slot is free.
        """
        # One row's key, folded the way `RowPlan.keys` folds row zero.
        return self._enqueued(prompt, max_new_tokens, _seed(key), 0)

    def _enqueued(self, prompt: Prompt, max_new_tokens: int | None, seed: int | jax.Array,
                  fold: int) -> Ticket:
        row = self._prepared(prompt, max_new_tokens, seed, fold)
        with self._ingress:
            if self._failed is not None:
                raise RuntimeError("the server stopped after a device check failed") from self._failed
            if row.budget:
                self._queue.append(row)
        self.arrived()
        if not row.budget:
            self._finish(row, terminated=False)
        return row.ticket

    def _prepared(self, prompt: Prompt, max_new_tokens: int | None, seed: int | jax.Array, fold: int) -> _Row:
        positions = None
        context = None
        if isinstance(prompt, (str, ModelInputs)):
            inputs = prepared_inputs(self.processor, prompt, images=None)
            conditioning = dict(inputs.conditioning)
            if self.cross is not None and CONTEXT in conditioning:
                context = self._request_context(conditioning.pop(CONTEXT),
                                                  conditioning.pop(CONTEXT_LENGTHS, None))
            if set(inputs.token_fields) - {"attention_mask", "positions", *GROUPS} or conditioning:
                raise ValueError(
                    "a served prompt carries tokens, validity, positions, image and block groups and a "
                    "cross-attention decoder's context only; media and custom token fields do not slot"
                )
            ids = np.asarray(inputs.tokens)
            fields = {name: np.asarray(value) for name, value in inputs.token_fields.items()}
            positions = fields.pop("positions", None)
        else:
            ids = host_token_rows(np.atleast_2d(np.asarray(prompt)))
            fields = {}
        if ids.shape[0] != 1:
            raise ValueError("submit takes one prompt; call the server with a batch")
        # Sized before the padding goes, by the rows' width, as the task sizes the same rows.
        budget = generation_budget(max_new_tokens, self.default_budget, self.max_length, ids.shape[1])
        valid = check_inputs(self.model, ids, fields, budget, self.sampling, 1,
                             capacity=self.capacity).astype(bool)
        if positions is not None and (positions.shape != valid.shape or not np.array_equal(
                positions[valid], (np.cumsum(valid, axis=1) - 1)[valid])):
            raise ValueError("a served prompt cannot carry noncanonical positions")
        self.rows.refuse(int(valid[0].sum()), budget)
        tokens = ids[0][valid[0]].astype(np.int32)
        # The prompt's own blocks, or the prefix run's where it carries none (`with_served_blocks`, which
        # refuses a prompt shorter than an int prefix); a group field no layer reads is not slotted, as the
        # model's call reads none of it.
        fields = with_served_blocks(fields, valid, self.prefix)
        groups = {name: np.asarray(fields[name])[0][valid[0]].astype(np.int32)
                  for name in GROUPS if name in fields and name in self.group_fields}
        salt = b""
        if context is not None or groups or self.keyed:
            hasher = hashlib.blake2b(digest_size=16)
            if self.keyed:
                # A cached memory rounds by the request's own key (`dew.nn.kv_cache.ROUNDING_KEYS`), so the
                # state after a prefix is that key's: the prefix is shared only under one key.
                hasher.update(b"seed" if isinstance(seed, int) else b"key")
                hasher.update(np.asarray(seed).tobytes() + np.int64(fold).tobytes())
            if context is not None:
                # The context's dtype too, so equal bytes of another dtype salt apart.
                array_fingerprint(hasher, np.asarray(context))
            if groups:
                # A token's keys read its whole group, wherever its members lie, and what each member reads,
                # so a prompt's keys depend on its tokens and groups up to its last grouped token: its prefix
                # pages are shared only with a prompt that agrees with it there.
                grouped = np.any([held >= 0 for held in groups.values()], axis=0)
                reach = int(np.flatnonzero(grouped)[-1]) + 1 if grouped.any() else 0
                for name, held in groups.items():
                    hasher.update(name.encode())
                    hasher.update(held[:reach].tobytes())
                hasher.update(tokens[:reach].tobytes())
            salt = hasher.digest()
        return _Row(tokens, budget, seed, fold, Ticket(), context=context, salt=salt, groups=groups)

    def _book_capture(self, slot: int, row: _Row) -> None:
        """Keep the state the call leaves in `slot` where the row's piece ends at a boundary it keeps
        (`_Row.snapshots`, `PagedRows.kept`). An entry two rows book in one call keeps the later
        booking's state, as `States` names it."""
        if row.prefilled not in row.snapshots or not isinstance(self.rows, PagedRows):
            return
        entry = self.rows.kept(row)
        if entry is not None:
            others = [(held, kept) for held, kept in self._captures if kept != entry]
            self._captures = [*others, (slot, entry)]

    def _piece_end(self, row: _Row) -> int:
        """Where the row's next prefill piece ends: the prompt's end, or `chunk` tokens on when the rows
        prefill in pieces, and on past `chunk` to the last member of a group the cut would split, a block
        or an image. A layer reads a group whole, so its tokens' keys and values are computed in one piece:
        a group longer than `chunk` is one longer piece. A page boundary whose state the row keeps
        (`_Row.snapshots`) ends the piece before it, so the state the call leaves is that boundary's."""
        length, chunk = len(row.prompt), self.rows.chunk
        end = length if chunk is None else min(length, row.prefilled + chunk)
        # A group's members need not be adjacent: the cut moves past the last member of every group with
        # members on both sides of it, until none straddles it.
        spans = [(int(members[0]), int(members[-1])) for held in row.groups.values()
                 for group in np.unique(held[held >= 0]) for members in [np.flatnonzero(held == group)]]
        while (last := max((last for first, last in spans if first < end <= last), default=None)) is not None:
            end = last + 1
        # A boundary whose state is kept for a later prompt ends a piece, where no group straddles it.
        return next((boundary for boundary in row.snapshots if row.prefilled < boundary < end
                     and not any(first < boundary <= last for first, last in spans)), end)

    def _request_context(self, embeddings: ArrayLike, lengths: ArrayLike | None) -> np.ndarray:
        """A request's context as its real tokens on the host, `[length, features]`, in the dtype
        the model reads a context in (`CrossAttending.context_dtype`), refused where it is not one
        row of the model's width, counts tokens it does not hold, or holds more than the
        cross-attention layers cache (`CrossBlocks.context_length`)."""
        assert self.cross is not None
        cross, features, dtype = self.cross
        # Checked where it is, on the host: admission places the batch (`_admit`), never the request.
        given, counts, _ = CrossContext.checked({CONTEXT: embeddings, **(
            {} if lengths is None else {CONTEXT_LENGTHS: lengths})}, np)
        tokens = np.asarray(given, dtype)
        # The count as given, before the cache's int32 holds it, where a wider one would wrap.
        counted = np.asarray(counts)
        if tokens.shape[0] != 1 or tokens.shape[-1] != features:
            raise ValueError(f"a request's context is one row of {features}-wide tokens, got {tokens.shape}")
        length = int(counted.reshape(-1)[0])
        if not 0 <= length <= tokens.shape[1]:
            raise ValueError(f"a request's context counts {length} real tokens of its {tokens.shape[1]}")
        if length > cross.context_length:
            raise ValueError(f"the cross-attention layers keep at most {cross.context_length} context "
                             f"tokens a row, got {length}")
        return tokens[0, :length]

    def step(self) -> None:
        """Run one device call and read the previous call's draws.

        The call admits the queued requests that fit and runs `decode_steps`
        iterations. Its own draws are read by the next `step`, or by `run`.
        """
        if self._failed is not None:
            raise RuntimeError("the server stopped after a device check failed") from self._failed
        if not (self._queue or self._rows):
            # Nothing to run: the host reads the last call's draws.
            self._settle()
            return
        admission = self._admit()
        with self._context():
            if self._restores:
                self._resident, self._carried = self._restored(
                    self._resident, self._carried, self._states, *_padded(self._restores, self.slots, 0))
                self._restores = []
            program = self._step if admission is None else self._admitting
            error, (self._resident, self._carried, draws) = program(
                self.model, self.variables, self.pad_id, self.rows.placement, self.decode_steps,
                self._resident, self._carried, admission, self.transforms, self.stopping, self.grammar)
            if self._captures:
                self._states = self._captured(self._states, self._resident, self._carried,
                                              *_padded(self._captures, self.admission, 1))
                self._captures = []
        self.steps += self.decode_steps
        self._settle()
        self._pending = (error, draws)

    def run(self) -> None:
        """Step until every queued and running request has resolved."""
        while self._queue or self._rows:
            self.step()
        self._settle()

    def __call__(
        self,
        prompts: str | Sequence[str] | Sequence[Sequence[int]] | ModelInputs,
        max_new_tokens: int | None = None,
        *,
        key: int | jax.Array,
    ) -> list[Generation]:
        """Submit a batch, run it through, and return its generations in order.

        Row `i` draws with the request key folded by `i`, as the same batch
        through `TextGeneration` would. Each row is validated as `submit`
        validates one.
        """
        base = _seed(key)
        inputs = prepared_inputs(self.processor, prompts, images=None)
        tickets = [self._enqueued(inputs.take_rows(jnp.arange(index, index + 1)), max_new_tokens, base, index)
                   for index in range(inputs.tokens.shape[0])]
        self.run()
        return [ticket.result() for ticket in tickets]

    def _admit(self) -> Admission | None:
        """Seat queued requests in free slots, then pick this step's prompt pieces.

        Rows still prefilling send their next piece, oldest first, each
        group's share of `admission` a step; a piece is the rest of the
        prompt, or at most `chunk` tokens when the rows prefill in pieces,
        but for a prefix block the cut would split, which goes whole
        (`_piece_end`).
        The admission's rows come group by group, as the slots do, and each
        names its slot within its group.
        """
        size, share = self.slots // self.groups, self.admission // self.groups
        free = [[slot for slot in range(group * size, (group + 1) * size) if slot not in self._rows]
                for group in range(self.groups)]
        seated = self.rows.seat(self._queue, free, share)
        # After seating: a request another thread submitted while the queue
        # was read is seated too, and is admitted after it was submitted.
        now = time.perf_counter()
        for slot, row in seated:
            if not row.ticket.set_running_or_notify_cancel():
                # Cancelled while it waited: its slot stays free, and its reservation goes back.
                self.rows.release(row)
                continue
            self.prefix_hits += row.prefilled
            row.ticket.admitted = now
            self._rows[slot] = row
            if row.restore is not None:
                self._restores.append((slot, row.restore))
        pending: list[list[tuple[int, _Row]]] = [[] for _ in range(self.groups)]
        for slot, row in self._rows.items():
            waiting = pending[slot // size]
            if row.prefilled < len(row.prompt) and len(waiting) < share:
                waiting.append((slot, row))
        if not any(pending):
            return None
        # Padded to the fewest rows that hold every group's pieces, not to the
        # whole admission: a request arriving alone prefills one row.
        share = admission_share(max(map(len, pending)), share)
        chosen = [(group * share + index, slot, row) for group, waiting in enumerate(pending)
                  for index, (slot, row) in enumerate(waiting)]
        # The capacity is whole tiles, not a bucket, so a piece's width bucket can pass it.
        ends = {slot: self._piece_end(row) for _, slot, row in chosen}
        longest = max(ends[slot] - row.prefilled for _, slot, row in chosen)
        width = min(shape_bucket(longest, 64), self.capacity)
        if self.rows.chunk is not None and longest <= self.rows.chunk:
            width = min(width, self.rows.chunk)
        count, capacity = share * self.groups, self.capacity
        tokens = np.zeros((count, width), np.int32)
        valid = np.zeros((count, width), bool)
        slots = np.full((count,), size, np.int32)
        budgets = np.zeros((count,), np.int32)
        empty_key = np.zeros(_KEY_DATA.shape, _KEY_DATA.dtype)
        keys: list[jax.Array | np.ndarray] = [empty_key] * count
        seeds = np.zeros((count,), np.int64)
        seeded = np.zeros((count,), bool)
        folds = np.zeros((count,), np.int32)
        tables = np.zeros((count, self.rows.width), np.int32)
        cursors = np.zeros((count,), np.int32)
        final = np.zeros((count,), bool)
        history = np.zeros((count, capacity), np.int32)
        history_valid = np.zeros((count, capacity), bool)
        cross = self.cross
        # Every piece carries its row's context, so a row continuing over a former occupant's slot,
        # or after shared prefix pages, reads its own; a row without one reads none.
        context_length, features, dtype = (0, 0, np.float32) if cross is None else (
            cross[0].context_length, cross[1], cross[2])
        contexts = np.zeros((count, context_length, features), dtype)
        context_lengths = np.zeros((count,), np.int32)
        # A group field goes in where an admitted row carries it: prompts without groups keep the causal call.
        pieces = {name: np.full((count, width), -1, np.int32)
                  for name in sorted({name for _, _, row in chosen for name in row.groups})}
        for index, slot, row in chosen:
            piece = row.prompt[row.prefilled:ends[slot]]
            tokens[index, width - len(piece):] = piece
            valid[index, width - len(piece):] = True
            for name, held in row.groups.items():
                pieces[name][index, width - len(piece):] = held[row.prefilled:row.prefilled + len(piece)]
            slots[index], budgets[index], folds[index] = slot % size, row.budget, row.fold
            if isinstance(row.seed, int):
                seeds[index], seeded[index] = row.seed, True
            else:
                keys[index] = row.seed
            table = self.rows.table(row)
            tables[index, :len(table)] = table
            cursors[index] = row.prefilled
            row.prefilled += len(piece)
            self._book_capture(slot, row)
            final[index] = row.prefilled == len(row.prompt)
            history[index, capacity - len(row.prompt):] = row.prompt
            history_valid[index, capacity - len(row.prompt):] = True
            if cross is not None and row.context is not None:
                contexts[index, :len(row.context)] = row.context
                context_lengths[index] = len(row.context)
            if final[index]:
                self.rows.prefilled(row)
        placed = jax.device_put(
            [
                tokens,
                valid,
                slots,
                budgets,
                _row_keys(tuple(keys), seeds, seeded, folds),
                tables,
                cursors,
                final,
                history,
                history_valid,
                *([] if cross is None else [contexts, context_lengths]),
                *pieces.values(),
            ],
            self._admitted,
        )
        conditioning = {} if cross is None else {CONTEXT: placed[10], CONTEXT_LENGTHS: placed[11]}
        fields = {"attention_mask": placed[1], **dict(zip(pieces, placed[len(placed) - len(pieces):],
                                                          strict=True))}
        return Admission(ModelInputs(placed[0], fields, conditioning), *placed[2:10])

    def _settle(self) -> None:
        """Read the previous call's draws into their rows, iteration by
        iteration; resolve the rows that ended."""
        if self._pending is None:
            return
        error, draws = jax.device_get(self._pending)
        self._pending = None
        try:
            error.throw()
        except BaseException as failure:
            self.fail(failure)
            raise
        now = time.perf_counter()
        drew: dict[int, _Row] = {}
        for step in range(draws.drawn.shape[0]):
            for slot, row in list(self._rows.items()):
                if not draws.drawn[step, slot]:
                    continue
                if not row.tokens:
                    row.ticket.first = now
                row.tokens.append(int(draws.token[step, slot]))
                row.behavior.append(float(draws.behavior[step, slot]))
                row.raw.append(float(draws.raw[step, slot]))
                drew[slot] = row
                if draws.stopped[step, slot] or len(row.tokens) == row.budget:
                    del self._rows[slot]
                    self.rows.release(row)
                    del drew[slot]
                    row.ticket._drew(row.tokens)
                    self._finish(row, terminated=bool(draws.ended[step, slot]))
        for row in drew.values():
            row.ticket._drew(row.tokens)

    def fail(self, failure: BaseException) -> None:
        """Stop after `failure`: every waiting and running request settles with it, and a later
        submission or step is refused."""
        with self._ingress:
            self._failed = failure
            waiting = [*self._rows.values(), *self._queue]
            self._rows.clear()
            self._queue.clear()
        for row in waiting:
            settled(row.ticket, failure)

    def _finish(self, row: _Row, *, terminated: bool) -> None:
        budget = row.budget
        drawn = np.full((1, budget), self.pad_id, np.int32)
        behavior = np.zeros((1, budget), np.float32)
        raw = np.zeros((1, budget), np.float32)
        count = len(row.tokens)
        drawn[0, :count] = row.tokens
        behavior[0, :count] = row.behavior
        raw[0, :count] = row.raw
        decoder = None if self.processor is None else functools.partial(decoded_rows, self.processor)
        row.ticket.finished = time.perf_counter()
        settled(row.ticket, Generation(
            np.concatenate([row.prompt[None], drawn], axis=1), np.array([count], np.int32),
            np.array([terminated]), behavior, raw, rows=1, decoder=decoder))


def _cache_rows(model: nn.Module, layout: KVCache, slots: int, capacity: int, groups: int, chunk: int | None,
                prefix_cache: bool, states: int | None, every: int | None) -> tuple[nn.Module, Rows]:
    """The model over its cache's layout, its pages pooled for `slots` rows of `capacity` slots in `groups`
    where it pages, and the rows' host side (`Server.from_task`'s arguments)."""
    if layout.page_size is None:
        if prefix_cache:
            raise ValueError("prefix caching runs over a paged cache; "
                             "serve with kv_cache=KVCache(page_size=...)")
        return model, DenseRows(groups, chunk)
    count = slots * (capacity // layout.page_size) if layout.pages is None else layout.pages
    # Only a model that declares its layout has a paged one; the default pages nothing.
    assert isinstance(model, Layered)
    model = model.with_kv_cache(dataclasses.replace(layout, pages=count, groups=groups))
    return model, PagedRows([Pages(count // groups, layout.page_size, prefix_cache=prefix_cache)
                             for _ in range(groups)], chunk, capacity // layout.page_size,
                            states=slots // groups if states is None else states, every=every)


def _padded(pairs: Sequence[tuple[int, int]], count: int, dropped: int) -> tuple[np.ndarray, np.ndarray]:
    """`pairs`' two columns, (slot, entry) each, padded to `count` rows, so one program serves every step:
    column `dropped`'s padding lies past every row (`DROPPED`), where the write drops it, the other's is 0."""
    columns = np.zeros((2, count), np.int32)
    columns[dropped] = DROPPED
    columns[:, :len(pairs)] = np.asarray(pairs, np.int32).T
    return columns[0], columns[1]


def _nothing() -> None:
    """No engine to wake."""
