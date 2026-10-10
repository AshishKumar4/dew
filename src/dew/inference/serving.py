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
or alone. `submit` returns a future for its `Generation`, `step` runs one
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
by the weights' rule table (`dew.nn.sharding.DEFAULT_RULES`). The slots split
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
import logging
import math
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.core import unfreeze
from jax.experimental import checkify
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.typing import ArrayLike

from dew.inference.pages import Pages
from dew.inference.projections import pack_projections, projection_groups
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
    split_donated,
    state_shardings,
)
from dew.inference.tasks import (
    Processor,
    TextGeneration,
    cache_ceiling,
    cache_sized,
    decoded_rows,
    prepared_inputs,
    shape_bucket,
)
from dew.nn.inputs import ModelInputs, host_token_rows, mesh_of, request_key
from dew.nn.kv_cache import CURSOR, POOLED, TABLE, KVCache, Layered, is_paged, leaf_name
from dew.nn.protocols import ProjectionGroup
from dew.nn.sharding import SEQUENCE_AXIS, STAGE_AXIS, batch_axes
from dew.objectives.base import Variables
from dew.sampling.decoding import LogitsTransform, Stopping
from dew.sampling.guided import Grammar
from dew.sampling.strategies import Sample
from dew.sampling.text import Generation, Sampling, check_inputs, prediction_depths, rebuild_position, resolve

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


class Ticket(Future):
    """A submitted request's future `Generation`, with when it was
    submitted, admitted to a slot, when its first token reached the host
    and when it finished, in `time.perf_counter` seconds; `admitted`,
    `first` and `finished` are None until they happen. `first` less
    `submitted` is the time to first token.

    `tokens` are the ids the request has drawn so far, replaced whole each
    time the server reads a step's draws back, so a caller streams the
    request from them; the last value is the generation's own, and it is
    set before the result."""

    def __init__(self) -> None:
        super().__init__()
        self.submitted = time.perf_counter()
        self.admitted: float | None = None
        self.first: float | None = None
        self.finished: float | None = None
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


def _seed(key: int | jax.Array | None) -> int | jax.Array:
    """A request's key as a row holds it: an integer seed as given, for
    `_row_keys` to make into a key, or a key's data."""
    if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
        return int(key)
    return jax.random.key_data(request_key(key))


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
        self.placement: Placement = Paged(len(pages), continuing=chunk is not None
                                          or any(part.prefix_cache for part in pages))

    def check(self, cache: Variables) -> None:
        """Every cached leaf has to be one of paged attention's: a paged server
        moves rows by page table and cursor alone, and a layer with other
        per-row state (a recurrent mixer, latent attention) takes the dense one."""
        leaves = jax.tree_util.tree_leaves_with_path(cache)
        if not is_paged(cache):
            raise ValueError("the model did not take the paged layout: none of its layers keeps a page table")
        for path, _ in leaves:
            if leaf_name(path) not in POOLED | {TABLE, CURSOR}:
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


def _refuse_longrope_modes(model: nn.Module, *, paged: bool = False, chunked: bool = False,
                           prefix: bool = False, mixed: bool = False) -> None:
    """Refuse serving modes whose cache cannot rebuild a crossing request (`rebuild_position`)."""
    position = rebuild_position(model)
    if position is not None:
        modes = [name for name, active in (('paged', paged), ('chunked', chunked),
                                          ('prefix', prefix), ('mixed admission', mixed)) if active]
        if modes:
            raise ValueError(
                f'LongRoPE crossing position {position} '
                f'cannot rebuild {", ".join(modes)} serving; use whole-prompt dense admission')


class Server:
    """Serves text requests with continuous batching over one resident KV cache.

    A `TextGeneration` call runs its batch as one program, so no request can
    join until it returns. A server keeps `slots` rows of `capacity` cache
    slots each and refills them from a queue. Each step prefills the prompts
    admitted that step into free rows and draws one token for every occupied
    row. A row leaves on EOS or at its budget, and the next request takes its
    slot. Every request runs the task's policy and draws the same tokens it
    would draw through the task alone with the same key.

    Build one with `from_task`. `admission` bounds the prompts one step
    prefills. A request whose prompt and budget do not fit the capacity is
    refused at `submit`, with the error `TextGeneration` raises for a request
    over the model's context. Over a mesh, the slots and the admission split
    evenly over the groups of rows, and so do the pages of a paged pool.
    """

    def __init__(self, model: nn.Module, variables: Variables, processor: Processor | None, *,
                 sampling: Sampling, transforms: tuple[LogitsTransform, ...], stopping: tuple[Stopping, ...],
                 grammar: Grammar | None, rows: Rows, slots: int, capacity: int, admission: int,
                 default_budget: int | None, decode_steps: int) -> None:
        if type(slots) is not int or slots < 1:
            raise ValueError("slots must be a positive number of resident rows")
        if type(admission) is not int or not 1 <= admission <= slots:
            raise ValueError("admission must be between one and the slot count")
        _refuse_longrope_modes(model, paged=isinstance(rows, PagedRows),
                               chunked=rows.chunk is not None, mixed=rows.placement.mixed)
        if type(decode_steps) is not int or decode_steps < 1:
            raise ValueError("decode_steps must be a positive number of iterations per device call")
        self.mesh = mesh_of(variables)
        self.groups = _row_groups(self.mesh)
        if slots % self.groups or admission % self.groups:
            raise ValueError(f"slots ({slots}) and admission ({admission}) must divide by the "
                             f"{self.groups} groups the mesh splits rows into")
        self.model = model
        self.variables = variables
        self._weight_groups: list[ProjectionGroup] = []
        self.processor = processor
        if sampling.stop:
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
        self.decode_steps = decode_steps
        self.steps = 0
        self._queue: deque[_Row] = deque()
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
            source_shapes = unfreeze(dict(jax.tree.map(
                lambda leaf: jax.ShapeDtypeStruct(np.shape(leaf), jnp.result_type(leaf)), variables)))
            for group in projection_groups(model, variables):
                node = source_shapes["params"]
                for part in group.path:
                    node = node[part]
                if group.packed in node:
                    packed = node.pop(group.packed)
                    for projection, width in zip(group.members, group.widths, strict=True):
                        node[projection] = {field: jax.ShapeDtypeStruct((*leaf.shape[:-1], width), leaf.dtype)
                                            for field, leaf in packed.items()}
                    self._weight_groups.append(group)
            self._source_shapes = jax.tree_util.tree_flatten_with_path(source_shapes)[0]
            shapes = jax.eval_shape(functools.partial(opened, model, pad_id=self.pad_id, slots=slots,
                                                      capacity=capacity), variables)
            rows.check(shapes.decoder.cache)
            # None selects the one mixed forward of `_mixed_step`.
            self.mixed_refusal = mixed_refusal(model, self.variables, shapes, rows.placement, rows.width)
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
            if isinstance(rows, PagedRows) and prediction_depths(model):
                raise ValueError("a paged server runs no prediction depths; their cache is seeded "
                                 "over the whole prompt at once")
            formats = resident_formats(
                model, self.variables, self.pad_id, rows.placement, decode_steps, shapes,
                state_shardings(self.mesh, shapes), self._admitted, transforms, stopping, grammar)
            self._step, self._admitting = decode_programs(formats, self._admitted)
            self._resident, self._carried = split_donated(
                opened_in(formats)(model, self.variables, self.pad_id, slots, capacity))

    def _context(self) -> contextlib.AbstractContextManager[None]:
        """The mesh the model traces under, so a layer that reads it, such as an expert exchange, finds it."""
        return contextlib.nullcontext() if self.mesh is None else jax.set_mesh(self.mesh)

    @classmethod
    def from_task(cls, task: TextGeneration, *, slots: int, capacity: int,
                  admission: int | None = None, kv_cache: KVCache | None = None,
                  chunk: int | None = None, prefix_cache: bool = False, decode_steps: int = 1) -> Server:
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
        paged cache.

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
            if mesh.shape[STAGE_AXIS] > 1:
                raise ValueError(
                    "a server decodes through the whole layer stack every step; the stage axis "
                    "runs the stack as the training pipeline, which the decoder refuses to decode "
                    "through (one token a step leaves no microbatches to pipeline). Serve on a "
                    "mesh with stage=1"
                )
            if mesh.shape[SEQUENCE_AXIS] > 1:
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
        ceiling = cache_ceiling(task.model)
        if ceiling is None:
            raise ValueError("a server needs a model that declares max_seq_len for its cache")
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive number of cache slots per row")
        model = task.model
        if kv_cache is not None:
            if not any(entry.name == "kv_cache" for entry in dataclasses.fields(model)):
                raise ValueError(f"{type(model).__name__} declares no kv_cache layout to replace")
            model = model.clone(kv_cache=kv_cache)
        layout = model.kv_cache if isinstance(model, Layered) else KVCache()
        _refuse_longrope_modes(model, paged=layout.page_size is not None,
                               chunked=chunk is not None, prefix=prefix_cache)
        # One program serves one capacity, so the cache holds whole tiles of it
        # rather than a power-of-two bucket: every decode step's attention reads
        # each slot, and a capacity of 384 bucketed to 512 read a third more.
        unit = math.lcm(64, layout.page_size or 64)
        rounded = -(-capacity // unit) * unit
        if rounded > ceiling:
            raise ValueError(
                f"a capacity of {capacity} rounds to {rounded}, over the model's max_seq_len of {ceiling}"
            )
        model = cache_sized(model, rounded)
        groups = _row_groups(mesh)
        rows: Rows
        if layout.page_size is None:
            if prefix_cache:
                raise ValueError("prefix caching runs over a paged cache; "
                                 "serve with kv_cache=KVCache(page_size=...)")
            rows = DenseRows(groups, chunk)
        else:
            count = slots * (rounded // layout.page_size) if layout.pages is None else layout.pages
            model = model.clone(kv_cache=dataclasses.replace(layout, pages=count, groups=groups))
            rows = PagedRows([Pages(count // groups, layout.page_size, prefix_cache=prefix_cache)
                              for _ in range(groups)], chunk, rounded // layout.page_size)
        return cls(
            model,
            task.variables,
            task.processor,
            sampling=policy,
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
            decode_steps=decode_steps,
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
        # A frozen and a plain mapping flatten to different tree structures but
        # the same leaf paths, so the paths are what must match.
        incoming, incoming_structure = jax.tree_util.tree_flatten_with_path(variables)
        source = self._source_shapes
        if [path for path, _ in incoming] != [path for path, _ in source]:
            raise ValueError("reloaded variables must have the served tree structure")
        for (_, new), (_, old) in zip(incoming, source, strict=True):
            kind, served_kind = jnp.result_type(new), old.dtype
            if np.shape(new) != np.shape(old) or (kind != served_kind and not (
                    jnp.issubdtype(kind, jnp.floating) and jnp.issubdtype(served_kind, jnp.floating))):
                raise ValueError(
                    f"a reloaded leaf is {kind}{list(np.shape(new))}, "
                    f"the served leaf {served_kind}{list(np.shape(old))}")
        normalized = jax.tree.unflatten(
            incoming_structure, [jnp.asarray(np.asarray(new, dtype=old.dtype)
                                             if isinstance(new, np.ndarray) else new, dtype=old.dtype)
                                 for (_, new), (_, old) in zip(incoming, source, strict=True)])
        packed = pack_projections(normalized, self._weight_groups)
        incoming, _ = jax.tree_util.tree_flatten_with_path(packed)
        served, structure = jax.tree_util.tree_flatten_with_path(self.variables)
        leaves = []
        for (_, new), (_, old) in zip(incoming, served, strict=True):
            served_kind = jnp.result_type(old)
            # A placement can alias a shard of the caller's array; the copy
            # owns its buffer whatever the caller donates next.
            placement = old.sharding if isinstance(old, jax.Array) else None
            leaves.append(jnp.array(jax.device_put(new, placement), dtype=served_kind, copy=True))
        self.variables = jax.tree.unflatten(structure, leaves)
        self.rows.reloaded()

    def submit(self, prompt: Prompt, max_new_tokens: int | None = None, *,
               key: int | jax.Array | None = None) -> Ticket:
        """Queue one request and return a ticket that resolves to its `Generation`.

        `prompt` is one text prompt, `ModelInputs` row or token row, without
        media or positions. It is validated as `TextGeneration` validates it,
        against the server's capacity in place of the model's context.
        Without `max_new_tokens`, the task's default budget applies. A
        request with a zero budget resolves at once, with the prompt alone.
        """
        # One row's key, folded the way `RowPlan.keys` folds row zero.
        return self._enqueued(prompt, max_new_tokens, _seed(key), 0)

    def _enqueued(self, prompt: Prompt, max_new_tokens: int | None, seed: int | jax.Array,
                  fold: int) -> Ticket:
        if self._failed is not None:
            raise RuntimeError("the server stopped after a device check failed") from self._failed
        row = self._prepared(prompt, max_new_tokens, seed, fold)
        if row.budget == 0:
            self._finish(row, terminated=False)
        else:
            self._queue.append(row)
        return row.ticket

    def _prepared(self, prompt: Prompt, max_new_tokens: int | None, seed: int | jax.Array, fold: int) -> _Row:
        positions = None
        if isinstance(prompt, (str, ModelInputs)):
            inputs = prepared_inputs(self.processor, prompt, images=None)
            if set(inputs.token_fields) - {"attention_mask", "positions"} or inputs.conditioning:
                raise ValueError(
                    "a served prompt carries tokens and validity only; "
                    "media and custom token fields do not slot"
                )
            ids = np.asarray(inputs.tokens)
            fields = {name: np.asarray(value) for name, value in inputs.token_fields.items()}
            positions = fields.pop("positions", None)
        else:
            ids = host_token_rows(np.atleast_2d(np.asarray(prompt)))
            fields = {}
        if ids.shape[0] != 1:
            raise ValueError("submit takes one prompt; call the server with a batch")
        budget = self.default_budget if max_new_tokens is None else max_new_tokens
        if budget is None:
            raise ValueError("max_new_tokens is required; the source declares no default budget")
        valid = check_inputs(self.model, ids, fields, budget, self.sampling, 1).astype(bool)
        if positions is not None and (positions.shape != valid.shape or not np.array_equal(
                positions[valid], (np.cumsum(valid, axis=1) - 1)[valid])):
            raise ValueError("a served prompt cannot carry noncanonical positions")
        self.rows.refuse(int(valid[0].sum()), budget)
        return _Row(ids[0][valid[0]].astype(np.int32), budget, seed, fold, Ticket())

    def step(self) -> None:
        """Run one device call and read the previous call's draws.

        The call admits the queued requests that fit and runs `decode_steps`
        iterations. Its own draws are read by the next `step`, or by `run`.
        """
        if self._failed is not None:
            raise RuntimeError("the server stopped after a device check failed") from self._failed
        admission = self._admit()
        with self._context():
            program = self._step if admission is None else self._admitting
            error, (self._resident, self._carried, draws) = program(
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

    def __call__(
        self,
        prompts: str | Sequence[str] | Sequence[Sequence[int]] | ModelInputs,
        max_new_tokens: int | None = None,
        *,
        key: int | jax.Array | None = None,
    ) -> list[Generation]:
        """Submit a batch, run it through, and return its generations in order.

        Row `i` draws with the request key folded by `i`, as the same batch
        through `TextGeneration` would. Each row is validated as `submit`
        validates one.
        """
        base = _seed(key)
        inputs = prepared_inputs(self.processor, prompts, images=None)
        tickets = [self._enqueued(inputs.take_rows(np.arange(index, index + 1)), max_new_tokens, base, index)
                   for index in range(inputs.tokens.shape[0])]
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
        if not any(pending):
            return None
        # Padded to the fewest rows that hold every group's pieces, not to the
        # whole admission: a request arriving alone prefills one row.
        share = admission_share(max(map(len, pending)), share)
        chosen = [(group * share + index, slot, row) for group, waiting in enumerate(pending)
                  for index, (slot, row) in enumerate(waiting)]
        # The capacity is whole tiles, not a bucket, so a piece's width bucket can pass it.
        longest = max(len(row.prompt) - row.prefilled for _, _, row in chosen)
        width = min(shape_bucket(longest, 64), self.capacity)
        width = width if self.rows.chunk is None else min(width, self.rows.chunk)
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
        for index, slot, row in chosen:
            piece = row.prompt[row.prefilled:row.prefilled + width]
            tokens[index, width - len(piece):] = piece
            valid[index, width - len(piece):] = True
            slots[index], budgets[index], folds[index] = slot % size, row.budget, row.fold
            if isinstance(row.seed, int):
                seeds[index], seeded[index] = row.seed, True
            else:
                keys[index] = row.seed
            table = self.rows.table(row)
            tables[index, :len(table)] = table
            cursors[index] = row.prefilled
            row.prefilled += len(piece)
            final[index] = row.prefilled == len(row.prompt)
            history[index, capacity - len(row.prompt):] = row.prompt
            history_valid[index, capacity - len(row.prompt):] = True
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
            ],
            self._admitted,
        )
        return Admission(ModelInputs(placed[0], {"attention_mask": placed[1]}), *placed[2:])

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
            self._fail(failure)
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
                    self._finish(row, terminated=bool(draws.stopped[step, slot]))
        for row in drew.values():
            row.ticket._drew(row.tokens)

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
        decoder = None if self.processor is None else functools.partial(decoded_rows, self.processor)
        row.ticket.finished = time.perf_counter()
        row.ticket.set_result(Generation(
            np.concatenate([row.prompt[None], drawn], axis=1), np.array([count], np.int32),
            np.array([terminated]), behavior, raw, rows=1, decoder=decoder))
