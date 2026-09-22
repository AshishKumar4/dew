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
"""

from __future__ import annotations

import dataclasses
import functools
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
from jax.typing import ArrayLike

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
from dew.objectives.base import Variables
from dew.sampling import decoding
from dew.sampling.decoding import LogitsTransform, StepState, Stopping
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


@struct.dataclass
class Slots:
    """The device state of every slot.

    `decoder` holds the model's cache, `[slots, capacity, ...]` per leaf, and
    the logits scoring each row's next draw. `tokens` and `valid` are the
    `StepState` buffer the transforms and criteria read, `[slots, 2 *
    capacity]`: a row's prompt sits right-aligned in the first half and its
    draws start at `capacity`, so `prompt_width` is the capacity for every
    row whatever its prompt length. `step` counts a row's draws, `budget`
    caps them, `active` marks the rows still drawing, and `keys` holds each
    row's PRNG key data.
    """

    decoder: DecoderState
    tokens: jax.Array
    valid: jax.Array
    step: jax.Array
    budget: jax.Array
    active: jax.Array
    keys: jax.Array


@struct.dataclass
class Admission:
    """The prompts one step admits: `[rows, width]` inputs, and per row the
    slot they take (`slots` for an unused row, which the scatter drops),
    the budget and the key data."""

    prompts: ModelInputs
    slots: jax.Array
    budgets: jax.Array
    keys: jax.Array


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

    A prefill over an all-invalid prompt hands back the carry at its real
    structure, cache leaves and logits and hidden states alike; zeroing it
    leaves the cursors at zero and the validity false, which is a free row.
    """
    ops = _operations(model, params, pad_id, prediction_depths(model))
    blank = ModelInputs(jnp.zeros((slots, 1), jnp.int32),
                        {"attention_mask": jnp.zeros((slots, 1), bool)})
    decoder, _ = _prefill(model, params, blank, ops)
    keys = jax.random.key_data(jax.random.key(0))
    return Slots(jax.tree.map(jnp.zeros_like, decoder),
                 jnp.zeros((slots, 2 * capacity), jnp.int32),
                 jnp.zeros((slots, 2 * capacity), bool),
                 jnp.zeros((slots,), jnp.int32), jnp.zeros((slots,), jnp.int32),
                 jnp.zeros((slots,), bool),
                 jnp.zeros((slots, *keys.shape), keys.dtype))


def _placed(resident: jax.Array, incoming: jax.Array, rows: jax.Array) -> jax.Array:
    """`incoming`'s rows written into `resident` at `rows`.

    A cache leaf from a prefill over a cache of the prompt's width is
    narrower than the resident leaf on its slot axis, the one axis whose
    size the model's `max_seq_len` sets; it lands in the first slots of
    its rows. Every other leaf is the same shape row for row. A row past
    the last slot is dropped.
    """
    narrow = [axis for axis in range(1, incoming.ndim) if incoming.shape[axis] != resident.shape[axis]]
    if not narrow:
        return resident.at[rows].set(incoming, mode="drop")
    axis, = narrow
    window = tuple(slice(None) if index != axis else slice(0, incoming.shape[axis])
                   for index in range(1, incoming.ndim))
    return resident.at[(rows, *window)].set(incoming, mode="drop")


def _admitted(model: nn.Module, params: Variables, pad_id: int, state: Slots, admission: Admission) -> Slots:
    """Return `state` with the admitted prompts prefilled into their slots.

    The prompts run as their own forward over a fresh cache sized to their row
    count and bucket width. The prefill's attention then reads only the prompt's
    keys, not the resident capacity, which would cost a large share of the
    batch's wall time in f32 attention scores.

    Every leaf of the result is scattered into the resident state. A resident
    row's slots past the prompt keep a former occupant's keys, which the
    cache's validity hides, since the validity is written whole. An unused
    admission row points past the last slot and the scatter drops it.
    """
    width = admission.prompts.tokens.shape[1]
    narrow = _sized(model, width)
    ops = _operations(narrow, params, pad_id, prediction_depths(narrow))
    fresh, real = _prefill(narrow, params, admission.prompts, ops)
    rows = admission.slots
    capacity = state.tokens.shape[1] // 2
    valid = admission.prompts.token_fields.get("attention_mask")
    valid = jnp.ones(admission.prompts.tokens.shape, bool) if valid is None else valid.astype(bool)
    padding = ((0, 0), (capacity - width, capacity))

    def place(resident, incoming):
        return _placed(resident, incoming, rows)

    cache = jax.tree.map(lambda leaf: leaf.at[rows].set(jnp.zeros_like(leaf[:1]), mode="drop")
                         if leaf.dtype == bool else leaf, state.decoder.cache)
    decoder = dataclasses.replace(state.decoder, cache=cache)
    return Slots(jax.tree.map(place, decoder, fresh),
                 place(state.tokens, jnp.pad(admission.prompts.tokens, padding)),
                 place(state.valid, jnp.pad(valid, padding)),
                 place(state.step, jnp.zeros_like(admission.slots)),
                 place(state.budget, admission.budgets),
                 place(state.active, real),
                 place(state.keys, admission.keys))


def _advanced(model: nn.Module, params: Variables, pad_id: int, state: Slots,
              admission: Admission | None, transforms: tuple[LogitsTransform, ...],
              stopping: tuple[Stopping, ...]) -> tuple[Slots, Draws]:
    """One iteration: admit, then every active row draws and feeds its token.

    This is `strategies._sample_rows`'s step over rows that carry their own
    budget, so a row that stops or spends its budget goes inactive while the
    rest keep drawing; an inactive row's cache is not written.
    """
    if admission is not None:
        state = _admitted(model, params, pad_id, state, admission)
    ops = _operations(model, params, pad_id, prediction_depths(model))
    capacity = state.tokens.shape[1] // 2
    view = StepState(state.tokens, state.valid, state.step, state.active,
                     jax.random.wrap_key_data(state.keys), prompt_width=capacity)
    token, behavior, raw = draw(view, state.decoder.logits, decoding.chain(transforms))
    committed = view.commit(token, state.active)
    stopped = state.active & decoding.criterion(stopping)(committed, token)
    following = ops.advance(state.decoder, token, state.active)
    drawn = state.active
    return (Slots(following, committed.tokens, committed.valid, committed.step, state.budget,
                  drawn & ~stopped & (committed.step < state.budget), state.keys),
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


def _stepped(model: nn.Module, params: Variables, pad_id: int, resident: Slots, carried: Slots,
             admission: Admission | None, transforms: tuple[LogitsTransform, ...],
             stopping: tuple[Stopping, ...]) -> tuple[checkify.Error, tuple[Slots, Draws]]:
    """Run `_advanced` over a split state, carrying its device checks as a value.

    `text._checked` carries them the same way. The host throws the error when
    it reads the draws, one step later.
    """

    def run(params, resident, carried, admission, transforms, stopping):
        return _advanced(model, params, pad_id, _joined(resident, carried), admission, transforms, stopping)

    return checkify.checkify(run, errors=checkify.user_checks)(
        params, resident, carried, admission, transforms, stopping)


Formats = Slots
"""A `Slots` whose leaves are `Format`s: the resident layout of each leaf."""

def _compiled(resident: Formats, carried: Formats) -> jax.stages.Wrapped:
    """Compile the step program over a state split into resident and carried halves.

    The matrix half is donated so XLA updates the cache in place instead of
    copying it. The step sets no XLA flag of its own; pass XLA_FLAGS to change
    the backend's defaults. See docs/performance.md for the measurements.
    """
    return jax.jit(_stepped, static_argnums=(0, 2), donate_argnums=(3,),
                   in_shardings=(None, resident, carried, None, None, None),
                   out_shardings=(None, (_joined(resident, carried), None)))


def _resident_formats(model: nn.Module, params: Variables, pad_id: int, state: Slots,
                      transforms: tuple[LogitsTransform, ...], stopping: tuple[Stopping, ...]) -> Formats:
    """Choose the memory layout the resident state keeps, by asking XLA for it.

    The attention dot reads the cached keys and values with the slot axis
    minor. Across a jit boundary an array is row-major unless a layout is
    given, which would transpose the whole cache twice per layer. So the
    decode-only step is compiled once with layouts left to XLA, and the
    output formats it chose become the format every step takes and returns.
    `state` is abstract: choosing the layout allocates nothing.
    """
    device = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    resident, carried = _split(state)
    program = _compiled(*(jax.tree.map(lambda leaf: Format(Layout.AUTO, device), half)
                          for half in (resident, carried)))
    compiled = program.lower(model, params, pad_id, resident, carried, None, transforms, stopping).compile()
    return compiled.output_formats[1][0]


def _opened_in(formats: Formats) -> jax.stages.Wrapped:
    """`_opened`, allocating its state in `formats`."""
    return jax.jit(_opened, static_argnums=(0, 2, 3, 4), out_shardings=formats)


_PROGRAMS: dict[tuple[tuple[Format, ...], str], jax.stages.Wrapped] = {}
"""One step program per resident layout; see `_program`."""


def _program(formats: Formats) -> jax.stages.Wrapped:
    """`_compiled` over `formats`, once per layout: one compile per (model,
    admission shape) serves every server that keeps its state in the same
    layout. The formats tree holds the cache mapping and is not hashable
    itself, so its leaves and its structure's text key the programs."""
    leaves, structure = jax.tree_util.tree_flatten(formats)
    key = (tuple(leaves), str(structure))
    program = _PROGRAMS.get(key)
    if program is None:
        program = _PROGRAMS[key] = _compiled(*_split(formats))
    return program


class Ticket(Future):
    """A submitted request's future `Generation`, with when it was
    submitted, admitted to a slot and finished, in `time.perf_counter`
    seconds; `admitted` and `finished` are None until they happen."""

    def __init__(self) -> None:
        super().__init__()
        self.submitted = time.perf_counter()
        self.admitted: float | None = None
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


class Server:
    """A slot scheduler over a resident KV cache; see the module docstring.

    Build one with `from_task`. `slots` rows run at once over `capacity`
    cache slots each; `admission` bounds the prompts one step prefills. A
    request whose prompt and budget do not fit the capacity is refused at
    `submit`, with the error `TextGeneration` raises for a request over the
    model's context.
    """

    def __init__(self, model: nn.Module, variables: Variables, processor: Processor | None, *,
                 sampling: Sampling, transforms: tuple[LogitsTransform, ...], stopping: tuple[Stopping, ...],
                 slots: int, capacity: int, admission: int, default_budget: int | None) -> None:
        if type(slots) is not int or slots < 1:
            raise ValueError("slots must be a positive number of resident rows")
        if type(admission) is not int or not 1 <= admission <= slots:
            raise ValueError("admission must be between one and the slot count")
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
        self.steps = 0
        self._queue: deque[_Row] = deque()
        self._rows: dict[int, _Row] = {}
        self._pending: tuple[checkify.Error, Draws] | None = None
        self._failed: BaseException | None = None
        shapes = jax.eval_shape(functools.partial(_opened, model, pad_id=self.pad_id, slots=slots,
                                                  capacity=capacity), variables)
        formats = _resident_formats(model, variables, self.pad_id, shapes, transforms, stopping)
        self._step = _program(formats)
        self._state: Slots = _opened_in(formats)(model, variables, self.pad_id, slots, capacity)

    @classmethod
    def from_task(cls, task: TextGeneration, *, slots: int, capacity: int,
                  admission: int | None = None) -> Server:
        """A server over the task's model, weights, processor and policy.

        `capacity` rounds up to a shape bucket and may not exceed the
        model's context. The task's `n` has to be one and its strategy the
        row-wise sampler, since the server's loop is that sampler over rows
        that come and go.
        """
        mesh = mesh_of(task.variables)
        if mesh is not None and mesh.size > 1:
            raise ValueError("a server runs over one device's weights; it does not split rows over a mesh")
        if task.n != 1:
            raise ValueError("a served request draws one continuation; submit a prompt once per draw")
        transforms, stopping, strategy = resolve(task.sampling, task.logits, task.stopping, task.strategy)
        if not isinstance(strategy, Sample):
            raise ValueError("a server runs the row-wise sampler; beam and speculative loops are batch-wide")
        ceiling = _ceiling(task.model)
        if ceiling is None:
            raise ValueError("a server needs a model that declares max_seq_len for its cache")
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive number of cache slots per row")
        rounded = _bucket(capacity, 64)
        if rounded > ceiling:
            raise ValueError(f"a capacity of {capacity} rounds to {rounded}, over the model's max_seq_len of {ceiling}")
        return cls(_sized(task.model, rounded), task.variables, task.processor, sampling=task.sampling,
                   transforms=transforms, stopping=stopping, slots=slots, capacity=rounded,
                   admission=min(slots, 8) if admission is None else admission,
                   default_budget=task.max_new_tokens)

    @property
    def cache(self) -> Variables:
        """The resident cache, updated in place by every step."""
        return self._state.decoder.cache

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
        return _Row(ids[0][valid[0]].astype(np.int32), budget, np.asarray(jax.random.key_data(key)), Ticket())

    def step(self) -> None:
        """One iteration: admit what fits, run the step, read the last one."""
        if self._failed is not None:
            raise RuntimeError("the server stopped after a device check failed") from self._failed
        admission = self._admit()
        error, (self._state, draws) = self._step(self.model, self.variables, self.pad_id, *_split(self._state),
                                                 admission, self.transforms, self.stopping)
        self.steps += 1
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
        free = [slot for slot in range(self.slots) if slot not in self._rows]
        if not free or not self._queue:
            return None
        taken: list[tuple[int, _Row]] = []
        while self._queue and len(taken) < min(self.admission, len(free)):
            taken.append((free[len(taken)], self._queue.popleft()))
        width = _bucket(max(len(row.prompt) for _, row in taken), 64)
        count = self.admission
        tokens = np.zeros((count, width), np.int32)
        valid = np.zeros((count, width), bool)
        slots = np.full((count,), self.slots, np.int32)
        budgets = np.zeros((count,), np.int32)
        keys = np.zeros((count, *taken[0][1].keys.shape), taken[0][1].keys.dtype)
        now = time.perf_counter()
        for index, (slot, row) in enumerate(taken):
            tokens[index, width - len(row.prompt):] = row.prompt
            valid[index, width - len(row.prompt):] = True
            slots[index], budgets[index], keys[index] = slot, row.budget, row.keys
            row.ticket.admitted = now
            self._rows[slot] = row
        return Admission(ModelInputs(jnp.asarray(tokens), {"attention_mask": jnp.asarray(valid)}),
                         jnp.asarray(slots), jnp.asarray(budgets), jnp.asarray(keys))

    def _settle(self) -> None:
        """Read the previous step's draws into their rows; resolve the rows that ended."""
        if self._pending is None:
            return
        error, draws = jax.device_get(self._pending)
        self._pending = None
        try:
            error.throw()
        except BaseException as failure:
            self._fail(failure)
            raise
        for slot, row in list(self._rows.items()):
            if not draws.drawn[slot]:
                continue
            row.tokens.append(int(draws.token[slot]))
            row.behavior.append(float(draws.behavior[slot]))
            row.raw.append(float(draws.raw[slot]))
            if draws.stopped[slot] or len(row.tokens) == row.budget:
                del self._rows[slot]
                self._finish(row, terminated=bool(draws.stopped[slot]))

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
