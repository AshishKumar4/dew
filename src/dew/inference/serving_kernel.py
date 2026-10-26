"""The device side of a `dew.inference.serving.Server`: its slot records and its programs.

`Slots` is the state a server keeps on its devices, `Admission` and `Draws`
what one step takes in and gives back, and `Dense` and `Paged` how a cache
lays out its rows. `decode_programs` compiles the step (`_stepped`) and the
admitting step (`_mixed_step`) once per resident format, so the host-side
scheduler in `serving` only feeds and reads them.
"""


from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
from flax import linen as nn, struct
from jax.experimental import checkify
from jax.experimental.layout import Format, Layout
from jax.sharding import Mesh, NamedSharding

from dew.inference.tasks import cache_sized
from dew.nn.cross_attention import CROSS_KEY, CROSS_VALUE
from dew.nn.inputs import Admitted, ModelInputs, default_placement
from dew.nn.kernels.options import compiler_options
from dew.nn.kernels.tuning import tune
from dew.nn.kv_cache import POOLED, ROW_CURSORS, TABLE, as_words, from_words, grouped, is_paged, leaf_name
from dew.nn.protocols import Serving
from dew.nn.scatter import DROPPED
from dew.nn.sharding import logical_spec
from dew.objectives.base import Variables
from dew.sampling import decoding
from dew.sampling.decoding import LogitsTransform, StepState
from dew.sampling.guided import Grammar
from dew.sampling.strategies import DecoderState, row_step
from dew.sampling.text import decode_ops, prefill_state


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
    all, whether a criterion finished it and whether one that ends a row
    did (`decoding.Finish`), and both likelihoods."""

    token: jax.Array
    drawn: jax.Array
    stopped: jax.Array
    ended: jax.Array
    behavior: jax.Array
    raw: jax.Array


def opened(model: nn.Module, params: Variables, pad_id: int, slots: int, capacity: int) -> Slots:
    """Every slot free: zeros in the shapes of a prefill over an all-invalid
    prompt, which leave the cursors at zero and the validity false."""
    ops = decode_ops(model, params, pad_id, None)
    blank = ModelInputs(jnp.zeros((slots, 1), jnp.int32),
                        {"attention_mask": jnp.zeros((slots, 1), bool)})
    _, decoder = jax.eval_shape(checkify.checkify(lambda: prefill_state(model, params, blank, ops)[0],
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

    Slots and incoming rows fall into `groups` equal groups in order, and
    `rows` counts within each group; a row at the group's size is dropped.
    Mapped over the groups, the group is a batch dimension of the scatter,
    which GSPMD splits wherever the slots split, with no collective. A leaf
    from a prefill at the prompt's width, narrower on the cache-slot axis,
    lands in the first slots of its rows.
    """
    words, fresh = as_words(resident), as_words(incoming.astype(resident.dtype))
    window = tuple(slice(0, fresh.shape[axis]) if fresh.shape[axis] != words.shape[axis]
                   else slice(None) for axis in range(1, fresh.ndim))

    def place(resident: jax.Array, incoming: jax.Array, rows: jax.Array) -> jax.Array:
        rows = jnp.where(rows < resident.shape[0], rows, DROPPED)
        return resident.at[(rows, *window)].set(incoming, mode="drop")

    placed = jax.vmap(place)(*(grouped(leaf, 0, groups) for leaf in (words, fresh, rows)))
    return from_words(placed.reshape(words.shape), resident.dtype)


def _taken(resident: jax.Array, rows: jax.Array, groups: int) -> jax.Array:
    """`resident`'s rows at `rows`, group by group, as `_placed` writes them; a row at the
    group's size reads zeros."""
    def take(resident: jax.Array, rows: jax.Array) -> jax.Array:
        return jnp.take(resident, rows, axis=0, mode="fill", fill_value=0)

    taken = jax.vmap(take)(grouped(resident, 0, groups), grouped(rows, 0, groups))
    return taken.reshape(rows.shape[0], *resident.shape[1:])


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
    """Admission over a dense cache: whole prompts, each into its own rows, in `groups` groups.

    `mixed` runs the admitting step as one forward over the decoding rows'
    tokens and the prompts (`_mixed_step`) where the model allows it
    (`mixed_refusal`); otherwise the prompts prefill in a forward of their own.
    With `continuing` a prompt can prefill in pieces, which the mixed step
    alone serves, each reading its row's earlier keys.
    """

    groups: int = 1
    mixed: bool = False
    continuing: bool = False

    def prefilled(self, model: nn.Module, params: Variables, pad_id: int, state: Slots,
                  admission: Admission) -> tuple[DecoderState, jax.Array]:
        """The resident carry with the admitted prompts prefilled into their rows.

        The prompts run over a fresh cache at their own bucket width, so the
        prefill's attention reads only the prompt's keys, not the resident
        capacity's f32 scores. A row's slots past the prompt keep a former
        occupant's keys, past the cursor the prefill sets.
        """
        narrow = cache_sized(model, admission.prompts.tokens.shape[1])
        fresh, real = prefill_state(narrow, params, admission.prompts,
                               decode_ops(narrow, params, pad_id, None))
        def place(resident: jax.Array, incoming: jax.Array) -> jax.Array:
            return _placed(resident, incoming, admission.slots, self.groups)

        return jax.tree.map(place, state.decoder, fresh), real


@dataclasses.dataclass(frozen=True)
class Paged:
    """Admission over a paged cache: prompt pieces, each continuing its row, in `groups` groups.

    `mixed` runs the admitting step as one forward, as `Dense.mixed` does;
    `continuing` pieces (chunks, or a shared prefix's pages) read their
    row's earlier keys.
    """

    groups: int = 1
    mixed: bool = False
    continuing: bool = False

    def prefilled(self, model: nn.Module, params: Variables, pad_id: int, state: Slots,
                  admission: Admission) -> tuple[DecoderState, jax.Array]:
        """The resident carry with each admitted row's next prompt piece in the pool.

        The piece runs over the shared pool with the page table and cursor the
        host supplies per row, so it reads earlier pieces and shared prefix
        pages and writes its keys to the row's own pages. Any other per-row
        state (a recurrent mixer's memory, image groups) comes from the row's
        slot for a piece that continues the row, and fresh, as the model
        allocates it, for one that starts it. Per-row leaves and logits are
        scattered back to the rows' slots.
        """
        rows = admission.cursors.shape[0]
        allocated = model.apply(params, rows, method="init_cache", mutable=["cache"])[1]["cache"]
        continues = admission.cursors > 0

        def view(path: tuple[jax.tree_util.KeyEntry, ...], leaf: jax.Array, fresh: jax.Array) -> jax.Array:
            name = leaf_name(path)
            if name == TABLE:
                return admission.tables
            if name in ROW_CURSORS:
                # A text row's next position is its count of tokens (`dew.nn.kv_cache.POSITION`).
                return admission.cursors
            if name in POOLED:
                return leaf
            held = _taken(leaf, admission.slots, self.groups)
            return jnp.where(continues.reshape(rows, *(1,) * (held.ndim - 1)), held, fresh)

        cache = jax.tree_util.tree_map_with_path(view, state.decoder.cache, allocated)
        fresh, real = prefill_state(model, params, admission.prompts, decode_ops(model, params, pad_id, None),
                                    cache=cache)

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
              stopping: decoding.Finishing, grammar: Grammar | None) -> tuple[Slots, Draws]:
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
    mixed = admission is not None and placement.mixed
    if mixed:
        assert admission is not None
        real = jnp.any(admission.prompts.token_fields["attention_mask"], axis=1)
        state = _seated(state, state.decoder, real, admission, placement.groups)
    elif admission is not None:
        state = _seated(state, *placement.prefilled(model, params, pad_id, state, admission), admission,
                        placement.groups)
    capacity = state.tokens.shape[1] // 2
    fed = state.active & (state.step > 0)
    last = jnp.take_along_axis(state.tokens, (capacity + state.step - 1)[:, None], axis=1)[:, 0]
    if mixed:
        assert admission is not None
        decoder = _mixed_step(model, params, pad_id, placement, state.decoder, last, fed, admission)
    else:
        ops = decode_ops(model, params, pad_id, None)
        decoder = ops.advance(state.decoder, last, fed)
    if admission is not None and not mixed:
        # Only a row seated this step draws without feeding; with no admission
        # every drawing row fed, and the step's logits replace the held ones
        # whole, without a pass over every slot's vocabulary to merge them.
        decoder = dataclasses.replace(decoder, logits=jnp.where(fed[:, None], decoder.logits,
                                                                state.decoder.logits))
    view = StepState(state.tokens, state.valid, state.step, state.active,
                     jax.random.wrap_key_data(state.keys), prompt_width=capacity, budget=state.budget)
    drew = row_step(view, decoder.logits, decoding.chain(transforms), decoding.finish(stopping),
                    None if grammar is None else (grammar, state.automaton))
    drawn = state.active
    stopped = drawn & drew.finished.done
    committed = drew.committed
    # A row stops at its own budget here, where `Sample` scans one budget for every row.
    return (Slots(decoder, committed.tokens, committed.valid, committed.step, state.budget,
                  drawn & ~stopped & (committed.step < state.budget), state.keys,
                  state.automaton if drew.automaton is None else drew.automaton),
            Draws(drew.token, drawn, stopped, drawn & drew.finished.ended, drew.behavior, drew.raw))


def _mixed_step(model: nn.Module, params: Variables, pad_id: int, placement: Placement, decoder: DecoderState,
                last: jax.Array, fed: jax.Array, admission: Admission) -> DecoderState:
    """The admitting step's forward over every token it runs: each slot's
    last draw where it feeds, then the admitted prompt pieces, left-padded,
    in one row (`dew.nn.inputs.Admitted`). The projections read their
    weights once for both, where a prefill of its own read them a second
    time. A fed row's logits are its token's; a row whose prompt ends this
    step takes its last piece's. A piece continuing its row (a paged cache's
    shared prefix, a chunked prompt) reads the row's earlier keys."""
    rows = last.shape[0]
    tokens, valid = admission.prompts.tokens, admission.prompts.token_fields["attention_mask"]
    pieces, width = tokens.shape
    flat = jnp.concatenate([jnp.where(fed, last, pad_id), jnp.where(valid, tokens, pad_id).reshape(-1)])
    real = jnp.concatenate([fed, valid.reshape(-1)])
    scored = jnp.concatenate([jnp.arange(rows), rows + jnp.arange(pieces) * width + width - 1])
    (_, logits), updated = model.apply(
        {**params, "cache": decoder.cache}, flat[None], scored[None], decode=True, attention_mask=real[None],
        admitted=Admitted(slots=admission.slots, cursors=admission.cursors,
                          tables=admission.tables if isinstance(placement, Paged) else None,
                          continuing=placement.continuing),
        mutable=["cache"],
        method="states_and_logits_at")
    held = jnp.where(fed[:, None], logits[0, :rows], decoder.logits)
    seated = jnp.where(admission.final, admission.slots, rows)
    return dataclasses.replace(decoder, cache=updated["cache"],
                               logits=held.at[seated].set(logits[0, rows:], mode="drop"))


def mixed_refusal(model: nn.Module, shapes: Slots, placement: Placement) -> str | None:
    """Why a server's admitting step keeps the prompts' prefill in a forward
    of its own, or None when one mixed forward (`_mixed_step`) runs the model
    as the two would. The model answers for its layers (`Serving`), and one
    that does not keeps the two forwards; the placement answers for the rest."""
    if not isinstance(model, Serving):
        return f"{type(model).__name__} does not say its layers run a mixed step (Serving)"
    refusal = model.mixed_admission_refusal()
    if refusal is not None:
        return refusal
    if placement.groups > 1:
        return "its slots split into the mesh's row groups"
    if shapes.decoder.positions is not None:
        return "its rows carry their own rotary positions"
    return None


def split_donated(state: Slots) -> tuple[Slots, Slots]:
    """Split the state into the matrices a step donates and the vectors it rewrites.

    XLA copies a donated buffer whose old value is still read after the output
    is written, which is true of every cursor, step count and active flag. So
    only the matrices are donated; the vectors get fresh buffers. Each half
    holds None where the other holds the leaf, and `joined` recombines them.
    A `Formats` tree splits the same way, by the rank each layout describes.
    """
    def matrix(leaf: jax.Array | Format) -> bool:
        if isinstance(leaf, Format):
            return isinstance(leaf.layout, Layout) and len(leaf.layout.major_to_minor) >= 2
        return leaf.ndim >= 2

    return (jax.tree.map(lambda leaf: leaf if matrix(leaf) else None, state),
            jax.tree.map(lambda leaf: None if matrix(leaf) else leaf, state))


def joined(resident: Slots, carried: Slots) -> Slots:
    return jax.tree.map(lambda held, other: other if held is None else held, resident, carried,
                        is_leaf=lambda leaf: leaf is None)


def _stepped(model: nn.Module, params: Variables, pad_id: int, placement: Placement, steps: int,
             resident: Slots, carried: Slots,
             admission: Admission | None, transforms: tuple[LogitsTransform, ...],
             stopping: decoding.Finishing, grammar: Grammar | None
             ) -> tuple[checkify.Error, tuple[Slots, Slots, Draws]]:
    """Run `steps` iterations of `_advanced` over a split state, carrying their device checks as a value.

    The first iteration takes the admission, the draws come back stacked
    `[steps, slots]`, and the host throws the error when it reads them, one
    call later. The state comes back split as it went in.

    Each iteration checks its own draws. jax's checkify cannot carry a check
    into a shard_map (jax-ml/jax#40907), and every iteration after the first
    calls the model, which may run one, after the checks of the iteration
    before it; so the iterations' errors come out of the loop as values and
    meet in one after it, the first iteration's ahead of the rest.
    """

    def advanced(state: Slots, admission: Admission | None) -> tuple[checkify.Error, tuple[Slots, Draws]]:
        return checkify.checkify(
            lambda state, admission: _advanced(model, params, pad_id, placement, state, admission, transforms,
                                               stopping, grammar),
            errors=checkify.user_checks)(state, admission)

    error, (state, draws) = advanced(joined(resident, carried), admission)
    draws = jax.tree.map(lambda leaf: leaf[None], draws)
    if steps > 1:
        def following(state: Slots, _: None) -> tuple[Slots, tuple[checkify.Error, Draws]]:
            error, (state, draws) = advanced(state, None)
            return state, (error, draws)

        state, (later, more) = jax.lax.scan(following, state, length=steps - 1)
        draws = jax.tree.map(lambda first, rest: jnp.concatenate([first, rest]), draws, more)

        def raised(first: checkify.Error, later: checkify.Error) -> None:
            checkify.check_error(first)
            checkify.check_error(later)

        error, _ = checkify.checkify(raised, errors=checkify.user_checks)(error, later)
    return error, (*split_donated(state), draws)


Formats = Slots
"""A `Slots` whose leaves are `Format`s: the resident layout of each leaf."""


def state_shardings(mesh: Mesh | None, state: Slots) -> Slots:
    """Where each leaf of the resident state lives: `state` with a sharding at every leaf.

    Every leaf holds one row per slot on axis zero (`activation_batch`),
    except a paged cache's pool, `[kv_heads, pages, page_size, ...]`, whose
    `pages` split over the same axes, one part per group of rows. Keys,
    values and their scales keep their heads where the key and value
    projections leave them (`activation_kv`), and the logits their
    vocabulary (`activation_vocab`), as a row's cross-attention keys and
    values keep theirs. `state` is abstract: only its shapes are read.
    """
    if mesh is None:
        device = default_placement()
        return jax.tree.map(lambda _: device, state)
    paged = is_paged(state.decoder.cache)

    def placed(leaf: jax.ShapeDtypeStruct | jax.Array, *names: str | None) -> NamedSharding:
        axes = (*names, *(None,) * (leaf.ndim - len(names)))
        return NamedSharding(mesh, logical_spec(axes, leaf.shape, mesh=mesh))

    def cached(
        path: tuple[jax.tree_util.KeyEntry, ...], leaf: jax.ShapeDtypeStruct | jax.Array
    ) -> NamedSharding:
        if leaf_name(path) in (CROSS_KEY, CROSS_VALUE):
            # A row's context keys and values, its heads where the key and value projections leave them.
            return placed(leaf, "activation_batch", None, "activation_kv")
        if leaf_name(path) not in POOLED:
            return placed(leaf, "activation_batch")
        if paged:
            return placed(leaf, "activation_kv", "pages")
        return placed(leaf, "activation_batch", None, "activation_kv")

    rows = jax.tree.map(lambda leaf: placed(leaf, "activation_batch"), state)
    return rows.replace(decoder=rows.decoder.replace(
        cache=jax.tree_util.tree_map_with_path(cached, state.decoder.cache),
        logits=placed(state.decoder.logits, "activation_batch", "activation_vocab")))


def _compiled(resident: Formats, carried: Formats, rows: NamedSharding | None,
              options: Mapping[str, bool | int | str] | None = None) -> jax.stages.Wrapped:
    """Compile the step program over a state split into resident and carried halves.

    The matrix half is donated so XLA updates the cache in place instead of
    copying it. `rows` places every leaf of an admission, whose rows come
    group by group, the way the slots are placed; None leaves them where they
    are. `options` are the program's own XLA options (`decode_programs`, the
    model's `dew.nn.kernels.options.compiler_options`); pass XLA_FLAGS to
    change the backend's defaults. See docs/performance.md for
    the measurements.
    """
    return jax.jit(_stepped, static_argnums=(0, 2, 3, 4), donate_argnums=(5,),
                   in_shardings=(None, resident, carried, rows, None, None, None),
                   out_shardings=(None, (resident, carried, None)),
                   compiler_options=None if options is None else dict(options))


def resident_formats(model: nn.Module, params: Variables, pad_id: int, placement: Placement, steps: int,
                      state: Slots, shardings: Slots, rows: NamedSharding | None,
                      transforms: tuple[LogitsTransform, ...], stopping: decoding.Finishing,
                      grammar: Grammar | None) -> tuple[Formats, Step | None]:
    """Choose the memory layout the resident state keeps, by asking XLA for it, and the decode-only
    step that chose it where it takes the state in that layout.

    The attention dot reads the cached keys and values with the slot axis
    minor. Across a jit boundary an array is row-major unless a layout is
    given, which would transpose the whole cache twice per layer. So the
    decode-only step is compiled once with layouts left to XLA, and the
    output formats it chose become the format every step takes and returns.
    Where that compile also reads its state in those formats, it is the
    decode step itself (`_reusable`), so a server compiles its largest program
    once, not again at the formats (`decode_programs`); else None.
    `state` is abstract: choosing the layout allocates nothing. `shardings`
    places each leaf (`state_shardings`).
    """
    resident, carried = split_donated(state)

    def automatic(half: Slots) -> Formats:
        return jax.tree.map(lambda leaf, sharding: None if leaf is None else Format(Layout.AUTO, sharding),
                            half, shardings, is_leaf=lambda leaf: leaf is None)

    program = _compiled(automatic(resident), automatic(carried), rows, compiler_options(model) or None)
    arguments = (model, params, pad_id, placement, steps, resident, carried, None, transforms, stopping,
                 grammar)
    tune(lambda: program.lower(*arguments))
    compiled = program.lower(*arguments).compile()
    formats = joined(*compiled.output_formats[1][:2])
    return formats, _reusable(compiled, formats)


type Step = Callable[..., tuple[checkify.Error, tuple[Slots, Slots, Draws]]]
"""A step program as a server calls it: `_stepped`'s arguments, the static ones among them."""


def _reusable(compiled: jax.stages.Compiled, formats: Formats) -> Step | None:
    """`compiled`, the layout-choosing decode step, as a `Step`, where it reads every leaf of the state
    in `formats` (an input left at the default layout reads a row-major leaf as one), so it takes the
    state its own outputs leave; else None. An executable takes its jit's dynamic arguments alone, so
    the static ones, the same every call of one server, are dropped."""
    resident, carried = compiled.input_formats[0][1:3]

    def read_as_left(taken: Format | None, left: Format | None) -> bool:
        if taken is None or left is None:
            return taken is left
        layout = taken.layout
        if layout is None:
            # An input at the default layout: row-major, which a left layout matches if it is.
            return isinstance(left.layout, Layout) and left.layout.major_to_minor == tuple(
                range(len(left.layout.major_to_minor))) and not left.layout.tiling
        return layout == left.layout

    taken = jax.tree.leaves(joined(resident, carried), is_leaf=lambda leaf: leaf is None)
    left = jax.tree.leaves(formats, is_leaf=lambda leaf: leaf is None)
    if len(taken) != len(left) or not all(map(read_as_left, taken, left)):
        return None

    def step(model, params, pad_id, placement, steps, resident, carried, admission, transforms, stopping,
             grammar):
        return compiled(params, resident, carried, admission, transforms, stopping, grammar)
    return step


def opened_in(formats: Formats) -> jax.stages.Wrapped:
    """`opened`, allocating its state in `formats`."""
    return jax.jit(opened, static_argnums=(0, 2, 3, 4), out_shardings=formats)


type _Options = tuple[tuple[str, bool | int | str], ...]
_PROGRAMS: dict[tuple[tuple[Format, ...], str, NamedSharding | None, _Options],
                tuple[jax.stages.Wrapped, jax.stages.Wrapped]] = {}
"""One pair of step programs per resident layout; see `decode_programs`."""

ADMISSION_OPTIONS = {"xla_gpu_enable_command_buffer": ""}
"""The admitting step's XLA options: no CUDA command buffers. Its inputs are
fresh buffers every call, so a command buffer is updated before it can
replay, and the device waited out the update: on an RTX 4080 serving
Qwen3-0.6B at 64 slots, about 5 ms before 9 of a run's 16 admitting steps
(docs/performance.md). Only a GPU backend is handed them."""


def decode_programs(formats: Formats, rows: NamedSharding | None, options: dict[str, bool | int | str]
                    ) -> tuple[jax.stages.Wrapped, jax.stages.Wrapped]:
    """`_compiled` over `formats`, once per layout and model `options`
    (`dew.nn.kernels.options.compiler_options`), for the decoding step and
    the admitting one (`ADMISSION_OPTIONS`): one compile per (model,
    admission shape) serves every server that keeps its state in the same
    layout, and a server whose layout-choosing compile reads the state as
    it leaves it runs that compile as its decoding step instead
    (`resident_formats`). The formats tree holds the cache mapping and is
    not hashable itself, so its leaves and its structure's text key the
    programs."""
    leaves, structure = jax.tree_util.tree_flatten(formats)
    key = (tuple(leaves), str(structure), rows, tuple(options.items()))
    programs = _PROGRAMS.get(key)
    if programs is None:
        halves = split_donated(formats)
        admission = ADMISSION_OPTIONS if jax.default_backend() == "gpu" else {}
        programs = _PROGRAMS[key] = (_compiled(*halves, rows, options or None),
                                     _compiled(*halves, rows, {**admission, **options} or None))
    return programs
