"""The device side of a `dew.inference.serving.Server`: its slot records and its programs.

`Slots` is the state a server keeps on its devices, `Admission` and `Draws`
what one step takes in and gives back, and `Dense` and `Paged` how a cache
lays out its rows, and whether its rows draw a token a step or a speculative
block (`plan`). `decode_programs` compiles the step (`_stepped`) and the
admitting step (`_mixed_step`) once per resident format, so the host-side
scheduler in `serving` only feeds and reads them.
"""


from __future__ import annotations

import dataclasses
from collections.abc import Callable, Hashable, Mapping

import jax
import jax.numpy as jnp
from flax import linen as nn, struct
from jax._src import config as jax_config
from jax.experimental import checkify
from jax.experimental.layout import Format, Layout
from jax.sharding import Mesh, NamedSharding

from dew.batch import VALIDITY_FIELD, Variables, coordinates, real_slots
from dew.inference.tasks import cache_sized
from dew.lora import adapter_names
from dew.mesh import logical_spec
from dew.nn.attention_blocks import CROSS_KEY, CROSS_VALUE
from dew.nn.cross_attention import CROSS_CACHE
from dew.nn.inputs import Admitted, ModelInputs, default_placement
from dew.nn.kernels.options import Option, OuterProgram, compiler_options, program as outer_program
from dew.nn.kernels.tuning import tune
from dew.nn.kv_cache import (
    GROUPS,
    POOLED,
    ROW_CURSORS,
    TABLE,
    as_words,
    from_words,
    grouped,
    is_paged,
    leaf_name,
    with_rounding_keys,
)
from dew.nn.progress import END, Progress, progressed
from dew.nn.protocols import Serving
from dew.nn.scatter import DROPPED
from dew.sampling import decoding
from dew.sampling.decoding import LogitsTransform, StepState
from dew.sampling.guided import Grammar
from dew.sampling.strategies import DecoderState, Speculative, block_step, row_step
from dew.sampling.text import adapter_field, decode_ops, prediction_depths, prefill_state, rounding_keys


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
    row whatever its prompt length. A server drawing speculative blocks
    (`Dense.plan`) keeps every emitted token in a row's cache, its logits
    scoring the next, as `Speculative` does, and its drafting state beside
    them (`DecoderState.hidden`, `drafts`). `step` counts a row's draws, `budget`
    caps them, `active` marks the rows still drawing, and `keys` holds each
    row's PRNG key data. `automaton` is each row's grammar state under a
    guided policy, zero (the start) on admission, and unread otherwise.
    A row's adapter of an adapter set's model rides its decoder state
    (`DecoderState.adapters`), -1 in a free slot.
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
    right-aligned in `capacity` slots, for the transforms to read. `adapters`
    is each row's adapter, its place in an adapter set's names, -1 for the
    base model alone (`DecoderState.adapters`). `prefilled`, where the server
    prefills on a mesh of its own (`Server.from_task`'s `prefill_mesh`), is
    the rows' prefilled decoder state and their real slots that program
    left (`Dense.rows_prefilled`), moved to the decoding mesh; None, the
    step prefills them itself."""

    prompts: ModelInputs
    slots: jax.Array
    budgets: jax.Array
    keys: jax.Array
    tables: jax.Array
    cursors: jax.Array
    final: jax.Array
    history: jax.Array
    history_valid: jax.Array
    adapters: jax.Array
    prefilled: tuple[DecoderState, jax.Array] | None = None


@struct.dataclass
class Draws:
    """What one step drew, per slot: its tokens, `[slots, emitted]`, one a
    step or a speculative block's slots, which of them the row kept, whether
    a criterion finished the row and whether one that ends a row did
    (`decoding.Finish`), and both likelihoods of each token."""

    token: jax.Array
    drawn: jax.Array
    stopped: jax.Array
    ended: jax.Array
    behavior: jax.Array
    raw: jax.Array


def _base_rows(model: nn.Module, rows: int) -> jax.Array | None:
    """`rows` rows of the base model alone, -1 each, for an adapter set's model (`DecoderState.adapters`);
    None for a model of no set, which reads no rows' adapters."""
    return jnp.full((rows,), -1, jnp.int32) if adapter_names(model) else None


def _rows_of(model: nn.Module, admission: Admission) -> jax.Array | None:
    """The admitted rows' adapters for an adapter set's model, None for a model of no set."""
    return admission.adapters if adapter_names(model) else None


def drafting_depths(model: nn.Module, plan: Speculative | None) -> int | None:
    """The prediction depths a server's decode keeps state for: the model's (`prediction_depths`) where
    its rows draw speculative blocks (`plan`), else none."""
    return None if plan is None else prediction_depths(model)


def opened(model: nn.Module, params: Variables, pad_id: int, slots: int, capacity: int,
           plan: Speculative | None = None, impl: str = "threefry2x32") -> Slots:
    """Every slot free: zeros in the shapes of a prefill over an all-invalid
    prompt, which leave the cursors at zero and the validity false, its
    drafting state among them where the rows draw speculative blocks, and
    its keys the data of `impl`'s (the server's PRNG implementation, a static
    argument, as no trace context holds the default one)."""
    ops = decode_ops(model, params, pad_id, drafting_depths(model, plan))
    blank = ModelInputs(jnp.zeros((slots, 1), jnp.int32),
                        {VALIDITY_FIELD: jnp.zeros((slots, 1), bool)})
    adapters = _base_rows(model, slots)

    def blank_prefill():
        return prefill_state(model, params, blank, ops, adapters=adapters)[0]

    _, decoder = jax.eval_shape(checkify.checkify(blank_prefill, errors=checkify.user_checks))
    keys = jax.random.key_data(jax.random.key(0, impl=impl))
    # A free slot's adapter is the base model alone.
    return Slots(dataclasses.replace(jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), decoder),
                                     adapters=adapters),
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
    alone serves, each reading its row's earlier keys. `plan` draws each row's
    tokens a speculative block a step (`Speculative`), its prefill seeding the
    row's drafting state; None, a token a step. `prng` is the PRNG
    implementation of the rows' keys, the default as the rows were made: a
    step reads their key data as keys of it, and its program is keyed by it,
    as JAX's trace cache does not key on the default.
    """

    groups: int = 1
    mixed: bool = False
    continuing: bool = False
    plan: Speculative | None = None
    prng: str = dataclasses.field(default_factory=lambda: str(jax.config.jax_default_prng_impl))

    def prefilled(self, model: nn.Module, params: Variables, pad_id: int, state: Slots,
                  admission: Admission, progress: Progress = END) -> tuple[DecoderState, jax.Array]:
        """The resident carry with the admitted prompts prefilled into their rows, at the run's
        `progress` (`Server.progress`).

        The prompts run over a fresh cache at their own bucket width, so the
        prefill's attention reads only the prompt's keys, not the resident
        capacity's f32 scores. A row's slots past the prompt keep a former
        occupant's keys, past the cursor the prefill sets.
        """
        fresh, real = (self.rows_prefilled(model, params, pad_id, admission, progress)
                       if admission.prefilled is None else admission.prefilled)

        def place(resident: jax.Array, incoming: jax.Array) -> jax.Array:
            return _placed(resident, incoming, admission.slots, self.groups)

        return jax.tree.map(place, state.decoder, fresh), real


    def rows_prefilled(self, model: nn.Module, params: Variables, pad_id: int, admission: Admission,
                       progress: Progress = END) -> tuple[DecoderState, jax.Array]:
        """The admitted prompts' own decoder state, their rows prefilled over a fresh cache at their bucket
        width, and their real slots: what `prefilled` seats, which a server that prefills on a mesh of its
        own runs there and moves (`Admission.prefilled`)."""
        narrow = cache_sized(model, admission.prompts.tokens.shape[1])
        ops = decode_ops(narrow, params, pad_id, drafting_depths(model, self.plan), progress)
        rounding = rounding_keys(jax.random.wrap_key_data(admission.keys, impl=self.prng))
        return prefill_state(narrow, params, admission.prompts, ops, rounding=rounding, progress=progress,
                             adapters=_rows_of(model, admission))


def _prefilled_rows(model: nn.Module, params: Variables, pad_id: int, placement: Dense, admission: Admission,
                    progress: Progress) -> tuple[DecoderState, jax.Array]:
    return placement.rows_prefilled(model, params, pad_id, admission, progress)


def prefill_program() -> OuterProgram:
    """The program a server that prefills on a mesh of its own runs there (`Dense.rows_prefilled`), over
    the admission where its caller places it on that mesh (`Server._prefilled_elsewhere`)."""
    return outer_program(_prefilled_rows, static_argnums=(0, 2, 3))


@dataclasses.dataclass(frozen=True)
class Paged:
    """Admission over a paged cache: prompt pieces, each continuing its row, in `groups` groups.

    `mixed` runs the admitting step as one forward, as `Dense.mixed` does;
    `continuing` pieces (chunks, or a shared prefix's pages) read their
    row's earlier keys; `plan` and `prng` are `Dense`'s.
    """

    groups: int = 1
    mixed: bool = False
    continuing: bool = False
    plan: Speculative | None = None
    prng: str = dataclasses.field(default_factory=lambda: str(jax.config.jax_default_prng_impl))

    def prefilled(self, model: nn.Module, params: Variables, pad_id: int, state: Slots,
                  admission: Admission, progress: Progress = END) -> tuple[DecoderState, jax.Array]:
        """The resident carry with each admitted row's next prompt piece in the pool, at the run's
        `progress`.

        The piece runs over the shared pool with the page table and cursor the
        host supplies per row, so it reads earlier pieces and shared prefix
        pages and writes its keys to the row's own pages. Any other per-row
        state (a recurrent mixer's memory, image groups) comes from the row's
        slot for a piece that continues the row, and fresh, as the model
        allocates it, for one that starts it. Per-row leaves and logits are
        scattered back to the rows' slots.
        """
        rows = admission.cursors.shape[0]
        adapters = _rows_of(model, admission)
        # The cache's allocation runs the model too (a forward over a dummy token): under the rows' adapters.
        allocated = with_rounding_keys(
            model.apply(params, rows, method="init_cache", mutable=["cache"],
                        **adapter_field(adapters))[1]["cache"],
            rounding_keys(jax.random.wrap_key_data(admission.keys, impl=self.prng)))
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
        ops = decode_ops(model, params, pad_id, drafting_depths(model, self.plan), progress)
        fresh, real = prefill_state(model, params, admission.prompts, ops, cache=cache, progress=progress,
                                    adapters=adapters)

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
              stopping: decoding.Finishing, grammar: Grammar | None,
              progress: Progress = END) -> tuple[Slots, Draws]:
    """One iteration: admit, feed every drawing row its last draw, then every active row draws, every
    forward at the run's `progress` (`Server.progress`).

    This is `strategies._sample_rows`'s step over rows that carry their own
    budget, so a row that stops or spends its budget goes inactive while the
    rest keep drawing; an inactive row's cache is not written. A grammar
    guides the draws as `strategies.Sample` does. The step runs the model
    before it draws, so no device check comes before the model call: jax's
    checkify cannot carry a check into a shard_map (jax-ml/jax#40907), and a
    model on a mesh runs its expert dispatch in one. A token is therefore fed
    in the step after the one that drew it, and a row that feeds nothing,
    one seated this step, keeps the logits its prompt left. A server drawing
    speculative blocks runs `_speculated` instead.
    """
    if placement.plan is not None:
        return _speculated(model, params, pad_id, placement, state, admission, transforms, stopping, progress)
    mixed = admission is not None and placement.mixed
    if mixed:
        assert admission is not None
        real = jnp.any(real_slots(admission.prompts.token_fields, admission.prompts.tokens.shape), axis=1)
        state = _seated(state, state.decoder, real, admission, placement.groups)
    elif admission is not None:
        state = _seated(state, *placement.prefilled(model, params, pad_id, state, admission, progress),
                        admission, placement.groups)
    capacity = state.tokens.shape[1] // 2
    fed = state.active & (state.step > 0)
    last = jnp.take_along_axis(state.tokens, (capacity + state.step - 1)[:, None], axis=1)[:, 0]
    if mixed:
        assert admission is not None
        decoder = _mixed_step(model, params, pad_id, placement, state.decoder, last, fed, admission, progress)
    else:
        ops = decode_ops(model, params, pad_id, None, progress)
        decoder = ops.advance(state.decoder, last, fed)
    if admission is not None and not mixed:
        # Only a row seated this step draws without feeding; with no admission
        # every drawing row fed, and the step's logits replace the held ones
        # whole, without a pass over every slot's vocabulary to merge them.
        decoder = dataclasses.replace(decoder, logits=jnp.where(fed[:, None], decoder.logits,
                                                                state.decoder.logits))
    view = StepState(state.tokens, state.valid, state.step, state.active,
                     jax.random.wrap_key_data(state.keys, impl=placement.prng), prompt_width=capacity,
                     budget=state.budget)
    drew = row_step(view, decoder.logits, decoding.chain(transforms), decoding.finish(stopping),
                    None if grammar is None else (grammar, state.automaton))
    drawn = state.active
    stopped = drawn & drew.finished.done
    committed = drew.committed
    # A row stops at its own budget here, where `Sample` scans one budget for every row.
    return (Slots(decoder, committed.tokens, committed.valid, committed.step, state.budget,
                  drawn & ~stopped & (committed.step < state.budget), state.keys,
                  state.automaton if drew.automaton is None else drew.automaton),
            Draws(drew.token[:, None], drawn[:, None], stopped, drawn & drew.finished.ended,
                  drew.behavior[:, None], drew.raw[:, None]))


def _speculated(model: nn.Module, params: Variables, pad_id: int, placement: Placement, state: Slots,
                admission: Admission | None, transforms: tuple[LogitsTransform, ...],
                stopping: decoding.Finishing, progress: Progress = END) -> tuple[Slots, Draws]:
    """One iteration of a server whose rows draw speculative blocks (`Dense.plan`): admit, then every
    active row drafts, verifies and emits a block (`block_step`) within its own budget, every forward at
    the run's `progress`. A row's cache holds every token it emitted and its logits score its next, so a
    row seated this step drafts from the logits its prompt left, as one that drew last step does from its
    block's."""
    if admission is not None:
        state = _seated(state, *placement.prefilled(model, params, pad_id, state, admission, progress),
                        admission, placement.groups)
    plan = placement.plan
    assert plan is not None
    capacity = state.tokens.shape[1] // 2
    ops = decode_ops(model, params, pad_id, drafting_depths(model, plan), progress)
    view = StepState(state.tokens, state.valid, state.step, state.active,
                     jax.random.wrap_key_data(state.keys, impl=placement.prng), prompt_width=capacity,
                     budget=state.budget)
    block = block_step(state.decoder, view, jnp.zeros_like(view.active), plan, ops,
                       decoding.chain(transforms), decoding.finish(stopping), state.budget)
    drawn, kept = state.active, block.kept
    committed = block.step
    return (Slots(block.state, committed.tokens, committed.valid, committed.step, state.budget,
                  committed.active, state.keys, state.automaton),
            Draws(block.tokens, kept & drawn[:, None], drawn & block.stopped, drawn & block.terminated,
                  jnp.where(kept, block.behavior, 0.0), jnp.where(kept, block.raw, 0.0)))


def _mixed_step(model: nn.Module, params: Variables, pad_id: int, placement: Placement, decoder: DecoderState,
                last: jax.Array, fed: jax.Array, admission: Admission, progress: Progress = END
                ) -> DecoderState:
    """The admitting step's forward over every token it runs: each slot's
    last draw where it feeds, then the admitted prompt pieces, left-padded,
    in one row (`dew.nn.inputs.Admitted`). The projections read their
    weights once for both, where a prefill of its own read them a second
    time. A fed row's logits are its token's; a row whose prompt ends this
    step takes its last piece's. A piece continuing its row (a paged cache's
    shared prefix, a chunked prompt) reads the row's earlier keys."""
    rows = last.shape[0]
    tokens = admission.prompts.tokens
    valid = real_slots(admission.prompts.token_fields, tokens.shape)
    pieces, width = tokens.shape
    flat = jnp.concatenate([jnp.where(fed, last, pad_id), jnp.where(valid, tokens, pad_id).reshape(-1)])
    real = jnp.concatenate([fed, valid.reshape(-1)])
    scored = jnp.concatenate([jnp.arange(rows), rows + jnp.arange(pieces) * width + width - 1])
    # The pieces' tokens' image and block groups where they carry them (`GROUPS`), -1 at the fed rows' tokens
    # and at padding, as a prefill of their own reads them.
    fields = admission.prompts.token_fields
    groups = {name: jnp.concatenate([jnp.full((rows,), -1, jnp.int32),
                                     jnp.where(valid, jnp.asarray(fields[name], jnp.int32), -1).reshape(-1)])[None]
              for name in GROUPS if name in fields}
    (_, logits), updated = model.apply(
        {**params, "cache": decoder.cache}, flat[None], scored[None], decode=True,
        coordinates=coordinates(flat[None].shape, valid=real[None]), **groups,
        admitted=Admitted(slots=admission.slots, cursors=admission.cursors,
                          tables=admission.tables if isinstance(placement, Paged) else None,
                          continuing=placement.continuing),
        mutable=["cache"],
        method="states_and_logits_at", **progressed(model, progress))
    held = jnp.where(fed[:, None], logits[0, :rows], decoder.logits)
    seated = jnp.where(admission.final, admission.slots, rows)
    return dataclasses.replace(decoder, cache=updated["cache"],
                               logits=held.at[seated].set(logits[0, rows:], mode="drop"))


def mixed_refusal(model: nn.Module, shapes: Slots, placement: Placement) -> str | None:
    """Why a server's admitting step keeps the prompts' prefill in a forward
    of its own, or None when one mixed forward (`_mixed_step`) runs the model
    as the two would. The model answers for its layers (`Serving`), and one
    that does not keeps the two forwards; the placement answers for the rest."""
    if placement.plan is not None:
        return "its rows draw speculative blocks, whose drafting state a prompt's own prefill seeds"
    if adapter_names(model):
        return "its rows run their own adapters (an adapter set), and the mixed forward packs them in one row"
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
             stopping: decoding.Finishing, grammar: Grammar | None, progress: Progress
             ) -> tuple[checkify.Error, tuple[Slots, Slots, Draws]]:
    """Run `steps` iterations of `_advanced` over a split state, carrying their device checks as a value,
    at the run's `progress`, an input like the weights, so a server serves each version it loads at its
    own without a compile of its own.

    The first iteration takes the admission, the draws come back stacked
    `[steps, slots, emitted]` (a token, or a speculative block's slots, an
    iteration), and the host throws the error when it reads them, one call
    later. The state comes back split as it went in.

    Each iteration checks its own draws. jax's checkify cannot carry a check
    into a shard_map (jax-ml/jax#40907), and every iteration after the first
    calls the model, which may run one, after the checks of the iteration
    before it; so the iterations' errors come out of the loop as values and
    meet in one after it, the first iteration's ahead of the rest.
    """

    def advanced(state: Slots, admission: Admission | None) -> tuple[checkify.Error, tuple[Slots, Draws]]:
        return checkify.checkify(
            lambda state, admission: _advanced(model, params, pad_id, placement, state, admission, transforms,
                                               stopping, grammar, progress),
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
              options: Mapping[str, Option] | None = None) -> OuterProgram:
    """The step program over a state split into resident and carried halves, an outer program
    (`dew.nn.kernels.options.program`, one for every layout, placement and options).

    The matrix half is donated so XLA updates the cache in place instead of
    copying it. `rows` places every leaf of an admission, whose rows come
    group by group, the way the slots are placed; None leaves them where they
    are. `options` are the program's own XLA options beside the run's
    (`decode_programs`: the model's `dew.nn.kernels.options.compiler_options`,
    and the admitting step's); pass XLA_FLAGS to change the backend's
    defaults. See docs/performance.md for the measurements.
    """
    return outer_program(_stepped, step=options, static_argnums=(0, 2, 3, 4), donate_argnums=(5,),
                         in_shardings=(None, resident, carried, rows, None, None, None, None),
                         out_shardings=(None, (resident, carried, None)))


def resident_formats(model: nn.Module, params: Variables, pad_id: int, placement: Placement, steps: int,
                      state: Slots, shardings: Slots, rows: NamedSharding | None,
                      transforms: tuple[LogitsTransform, ...], stopping: decoding.Finishing,
                      grammar: Grammar | None, progress: Progress) -> tuple[Formats, Step | None]:
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
                 grammar, progress)
    tune(lambda: program.lower(*arguments))
    compiled = program.compiled(*arguments)
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
             grammar, progress):
        return compiled(params, resident, carried, admission, transforms, stopping, grammar, progress)
    return step


def row_state(path) -> bool:
    """Whether the decoder cache leaf at `path` is a row's own state, which its prompt leaves in the row's
    slot (a recurrent mixer's memory and conv tail, latent attention's index keys): any leaf but the paged
    pool's, its table, a row's cursors and a cross-attention layer's context."""
    return leaf_name(path) not in POOLED | {TABLE} | ROW_CURSORS | CROSS_CACHE


def _restored(resident: Slots, carried: Slots, states: Variables, slots: jax.Array,
              entries: jax.Array) -> tuple[Slots, Slots]:
    """The state with each row state leaf's `slots` rows set to `states`' `entries` (`row_state`), the rest
    as it is; a slot past the rows is dropped."""
    state = joined(resident, carried)

    def restored(held, leaf):
        return leaf if held is None else leaf.at[slots].set(held[entries].astype(leaf.dtype), mode="drop")

    cache = jax.tree.map(restored, states, state.decoder.cache, is_leaf=lambda x: x is None)
    return split_donated(dataclasses.replace(state, decoder=dataclasses.replace(state.decoder, cache=cache)))


def _captured(states: Variables, resident: Slots, carried: Slots, slots: jax.Array,
              entries: jax.Array) -> Variables:
    """`states` with `entries` set to the row state leaves' `slots` rows (`row_state`); an entry past the
    store is dropped."""
    def captured(held, leaf):
        return None if held is None else held.at[entries].set(leaf[slots].astype(held.dtype), mode="drop")

    cache = joined(resident, carried).decoder.cache
    return jax.tree.map(captured, states, cache, is_leaf=lambda x: x is None)


def _pooled(cache: Variables) -> Variables:
    """The paged pool's leaves of `cache` (`POOLED`), `[heads, pages, page_size, ...]`, None elsewhere."""
    return jax.tree_util.tree_map_with_path(lambda path, leaf: leaf if leaf_name(path) in POOLED else None,
                                            cache)


def _read_pages(resident: Slots, carried: Slots, pages: jax.Array) -> Variables:
    """The pool's `pages`, each leaf's `[heads, len(pages), page_size, ...]`, None off the pool."""
    return jax.tree.map(lambda leaf: None if leaf is None else jnp.take(leaf, pages, axis=1),
                        _pooled(joined(resident, carried).decoder.cache), is_leaf=lambda x: x is None)


def page_shapes(resident: Slots, carried: Slots, count: int) -> Variables:
    """The shapes `count` of the pool's pages take as `_read_pages` reads them."""
    return jax.eval_shape(_read_pages, resident, carried, jax.ShapeDtypeStruct((count,), jnp.int32))


def _written_pages(resident: Slots, carried: Slots, pages: jax.Array, held: Variables) -> tuple[Slots, Slots]:
    """The state with the pool's `pages`, `[groups, count]`, each group's own, set to `held`'s, each leaf
    `[groups, heads, count, ...]`, a group's pages in its row; a page past the pool is dropped."""
    state = joined(resident, carried)

    def written(leaf, values):
        if values is None:
            return leaf
        flat = jnp.moveaxis(values, 0, 1).reshape(values.shape[1], -1, *values.shape[3:])
        return leaf.at[:, pages.reshape(-1)].set(flat.astype(leaf.dtype), mode="drop")

    cache = jax.tree.map(written, state.decoder.cache, held, is_leaf=lambda x: x is None)
    return split_donated(dataclasses.replace(state, decoder=dataclasses.replace(state.decoder, cache=cache)))


def page_programs(formats: Formats, rows: NamedSharding | None) -> tuple[OuterProgram, OuterProgram]:
    """The programs that move a paged pool's pages between the resident state, kept in `formats`, and the
    host (`dew.inference.pages.Transfers`): reading pages out, and writing pages in, the pages handed in a
    row per group placed as the server's admitted rows are (`rows`), the resident matrices donated. JAX
    writes no slice into an array in host memory, so the host keeps the pages it reads."""
    resident, carried = split_donated(formats)
    return (outer_program(_read_pages, in_shardings=(resident, carried, None)),
            outer_program(_written_pages, donate_argnums=(0,), in_shardings=(resident, carried, None, rows),
                          out_shardings=(resident, carried)))


def state_programs(formats: Formats, states: Variables) -> tuple[OuterProgram, OuterProgram]:
    """The programs that move row states between the resident state, kept in `formats`, and a store of
    them placed as `states` is: restoring a store's entries into slots, the resident matrices donated, and
    capturing slots into a store's entries, the store donated (`dew.inference.pages.States`). Each is an
    outer program (`dew.nn.kernels.options.program`), one for every layout and placement."""
    resident, carried = split_donated(formats)
    placed = jax.tree.map(lambda leaf: leaf.sharding, states)
    restore, capture = (resident, carried, placed, None, None), (placed, resident, carried, None, None)
    return (outer_program(_restored, donate_argnums=(0,), in_shardings=restore,
                          out_shardings=(resident, carried)),
            outer_program(_captured, donate_argnums=(0,), in_shardings=capture, out_shardings=placed))


def opened_in(formats: Formats) -> OuterProgram:
    """`opened`, allocating its state in `formats`: an outer program (`dew.nn.kernels.options.program`),
    one for every layout."""
    return outer_program(opened, static_argnums=(0, 2, 3, 4, 5, 6), out_shardings=formats)


ADMISSION_OPTIONS = {"xla_gpu_enable_command_buffer": ""}
"""The admitting step's XLA options: no CUDA command buffers. Its inputs are
fresh buffers every call, so a command buffer is updated before it can
replay, and the device waited out the update: on an RTX 4080 serving
Qwen3-0.6B at 64 slots, about 5 ms before 9 of a run's 16 admitting steps
(docs/performance.md). Only a GPU backend is handed them."""


def dispatched(program: OuterProgram) -> Step:
    """`program`, a server's step, as a `Step` that runs its executables (`OuterProgram.executable`), one
    for each size of admission its bucketing pads to and each trace context it runs in (`Fetched`'s), called
    with the dynamic arguments alone: a server's weights, cache and policies keep their shapes and placements
    from call to call, so no step's Python walks them to build jit's dispatch key."""
    executables: dict[Hashable, jax.stages.Compiled] = {}

    def step(model, params, pad_id, placement, steps, resident, carried, admission, transforms, stopping,
             grammar, progress):
        size = None if admission is None else tuple(jnp.shape(leaf) for leaf in jax.tree.leaves(admission))
        key = (size, jax_config.trace_context())
        executable = executables.get(key)
        if executable is None:
            executable = executables[key] = program.executable(
                model, params, pad_id, placement, steps, resident, carried, admission, transforms, stopping,
                grammar, progress)
        return executable(params, resident, carried, admission, transforms, stopping, grammar, progress)
    return step


def decode_programs(formats: Formats, rows: NamedSharding | None, options: dict[str, Option]
                    ) -> tuple[OuterProgram, OuterProgram]:
    """`_compiled` over `formats` and model `options`
    (`dew.nn.kernels.options.compiler_options`), for the decoding step and
    the admitting one (`ADMISSION_OPTIONS`): one compile per (model,
    admission shape) serves every server that keeps its state in the same
    layout (`dew.nn.kernels.options.program` keeps one program for every
    layout and options), and a server whose layout-choosing compile reads the
    state as it leaves it runs that compile as its decoding step instead
    (`resident_formats`)."""
    halves = split_donated(formats)
    admission = ADMISSION_OPTIONS if jax.default_backend() == "gpu" else {}
    return (_compiled(*halves, rows, options or None),
            _compiled(*halves, rows, {**admission, **options} or None))
