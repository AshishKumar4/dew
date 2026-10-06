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
`cache[row, cursor]`). Where every cached layer is plain attention over a
dense cache or one page pool, an admitting step is one forward over every
token it runs: each row's last draw and the admitted prompts, laid out in one
row (`dew.nn.inputs.Admitted`), so the projections read their weights once.
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
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct
from flax.core import unfreeze
from jax.experimental import checkify
from jax.experimental.layout import Format, Layout
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.typing import ArrayLike

from dew.inference.pages import Pages
from dew.inference.tasks import Processor, TextGeneration, _bucket, _ceiling, _decoded, _prepared, _sized
from dew.interop.streaming import SourceLeaf
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import GatedMLP
from dew.nn.backbones.layer_plan import group_layers
from dew.nn.inputs import Admitted, ModelInputs, host_token_rows, mesh_of, request_key
from dew.nn.kv_cache import (
    CURSOR,
    POOLED,
    TABLE,
    KVCache,
    Layered,
    as_words,
    from_words,
    grouped,
    is_paged,
    leaf_name,
)
from dew.nn.mixers.attention import CausalSelfAttention
from dew.nn.scatter import DROPPED
from dew.nn.sharding import SEQUENCE_AXIS, STAGE_AXIS, batch_axes, logical_spec
from dew.objectives.base import Variables
from dew.sampling import decoding
from dew.sampling.decoding import LogitsTransform, StepState, Stopping
from dew.sampling.guided import Grammar
from dew.sampling.strategies import DecoderState, Sample, draw
from dew.sampling.text import (
    Generation,
    Sampling,
    _check_inputs,
    _operations,
    _prefill,
    prediction_depths,
    resolve,
)

Prompt = str | Sequence[int] | ArrayLike | ModelInputs
"""One request: text for the processor, one row of token ids, or one prepared row."""


def _row_groups(mesh: Mesh | None) -> int:
    """How many groups of rows a server over `mesh` keeps, one per share of the slots."""
    return 1 if mesh is None else math.prod(mesh.shape[axis] for axis in batch_axes(mesh))


def _projection_groups(model: nn.Module, variables: Variables
                       ) -> dict[tuple[str, ...], tuple[str, tuple[str, ...], tuple[int, ...]]]:
    from dew.lora import _Adapted

    if isinstance(type(model), _Adapted):
        return {}
    groups = {}

    def projections(next_fun, args, kwargs, context):
        module = context.module
        group = None
        if context.method_name == "setup":
            if isinstance(module, CausalSelfAttention) and not (module.kv_shared or module.k_eq_v):
                width = module.num_heads * module.head_dim * (2 if module.output_gate else 1)
                kv_width = module.num_kv_heads * module.head_dim
                group = ("qkv_proj", ("q_proj", "k_proj", "v_proj"), (width, kv_width, kv_width))
            elif isinstance(module, GatedMLP) and module.activation not in ('gelu', 'gelu_exact', 'relu'):
                group = ("gate_up_proj", ("gate_proj", "up_proj"), (module.hidden_features,) * 2)
        if group is not None:
            paths = [module.path]
            for depth, part in enumerate(module.path):
                layers = group_layers(part)
                if layers is not None and len(layers) > 1:
                    paths = [(*path[:depth], f"layers_{index}", *path[depth + 1:])
                             for path in paths for index in layers]
            for path in paths:
                node = variables.get("params", {})
                for part in path:
                    node = node.get(part, {})
                held = [node.get(name, {}) for name in group[1]]
                fields = set(held[0])
                if group[0] in node or ("kernel" in fields and fields <= {"kernel", "bias"} and all(
                        set(projection) == fields for projection in held) and all(
                        isinstance(projection[field], (jax.Array, np.ndarray, SourceLeaf))
                        and projection[field].dtype == held[0][field].dtype
                        and projection[field].shape[:-1] == held[0][field].shape[:-1]
                        for projection in held for field in fields)):
                    groups[path] = group
        return next_fun(*args, **kwargs)

    def visit(module: nn.Module) -> None:
        # The projections are setup children. Binding and walking that
        # hierarchy avoids tracing a decoder forward just to name weights.
        module._try_setup()
        for child in module._state.children.values():
            if isinstance(child, nn.Module):
                visit(child)

    with nn.intercept_methods(projections):
        visit(model.bind(variables))
    return groups


def _pack_projections(
    variables: Variables,
    groups: Mapping[tuple[str, ...], tuple[str, tuple[str, ...], tuple[int, ...]]],
) -> Variables:
    """Move the concatenation of constant serving weights out of the decode step."""
    if not groups:
        return variables
    packed = unfreeze(dict(variables))
    for path, (name, projections, _) in groups.items():
        node = packed["params"]
        for part in path:
            node = node[part]
        if name in node:
            continue
        def joined(field, node=node, projections=projections):
            leaves = [node[projection][field] for projection in projections]
            if isinstance(leaves[0], SourceLeaf):
                return SourceLeaf.concatenate(leaves)
            concatenate = np.concatenate if isinstance(leaves[0], np.ndarray) else jnp.concatenate
            return concatenate(leaves, axis=-1)
        node[name] = {field: joined(field) for field in node[projections[0]]}
        for projection in projections:
            del node[projection]
    return packed


def _inference_projections(model: nn.Module, variables: Variables) -> Variables:
    """Pack an autoregressive decoder's constant projections during placement.

    Other task kinds retain their own parameter layouts; adapted decoders
    retain the original projection paths their LoRA branches bind.
    """
    if not isinstance(model, CausalTransformer) or not model.causal:
        return variables
    return _pack_projections(variables, _projection_groups(model, variables))


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


_log = logging.getLogger(__name__)

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
    """Every slot free: zeros in the shapes of a prefill over an all-invalid
    prompt, which leave the cursors at zero and the validity false."""
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


@dataclasses.dataclass(frozen=True)
class Dense:
    """Admission over a dense cache: whole prompts, each into its own rows, in `groups` groups.

    `mixed` runs the admitting step as one forward over the decoding rows'
    tokens and the prompts (`_mixed_step`) where the model allows it
    (`_mixed_refusal`); otherwise the prompts prefill in a forward of their own.
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
        narrow = _sized(model, admission.prompts.tokens.shape[1])
        fresh, real = _prefill(narrow, params, admission.prompts,
                               _operations(narrow, params, pad_id, prediction_depths(narrow)))
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
        pages and writes its keys to the row's own pages; per-row leaves and
        logits are scattered back to the rows' slots.
        """
        def view(path: tuple[jax.tree_util.KeyEntry, ...], leaf: jax.Array) -> jax.Array:
            name = leaf_name(path)
            if name == TABLE:
                return admission.tables
            if name == CURSOR:
                return admission.cursors
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
        ops = _operations(model, params, pad_id, prediction_depths(model))
        decoder = ops.advance(state.decoder, last, fed)
    if admission is not None and not mixed:
        # Only a row seated this step draws without feeding; with no admission
        # every drawing row fed, and the step's logits replace the held ones
        # whole, without a pass over every slot's vocabulary to merge them.
        decoder = dataclasses.replace(decoder, logits=jnp.where(fed[:, None], decoder.logits,
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


def _mixed_refusal(model: nn.Module, params: Variables, shapes: Slots, placement: Placement,
                   width: int) -> str | None:
    """Why a server's admitting step keeps the prompts' prefill in a forward
    of its own, or None when one mixed forward (`_mixed_step`) runs the model
    as the two would. `width` is a row's pages in its table."""
    if not isinstance(model, CausalTransformer):
        return f"{type(model).__name__} is not a CausalTransformer"
    if placement.groups > 1:
        return "its slots split into the mesh's row groups"
    if prediction_depths(model):
        return "it runs prediction depths"
    if shapes.decoder.positions is not None:
        return "its rows carry their own rotary positions"
    if model.position_embedding == "learned" or model.engram is not None or model.hash_layers:
        return "a learned position embedding, n-gram or hash routing reads beyond the token"
    owners = _cache_owners(model, params)
    for path, _ in jax.tree_util.tree_leaves_with_path(shapes.decoder.cache):
        owner = owners.get(tuple(str(key.key) for key in path[:-1] if isinstance(key, jax.tree_util.DictKey)))
        if owner not in MIXED_LAYERS:
            held_by = "a layer the walk does not reach" if owner is None else owner.__name__
            return f"{jax.tree_util.keystr(path)} is {held_by}'s, which a mixed step does not run"
    rows = shapes.decoder.logits.shape[0]
    try:
        jax.eval_shape(lambda params, decoder: _mixed_step(
            model, params, 0, placement, decoder, jnp.zeros(rows, jnp.int32), jnp.zeros(rows, bool),
            _probe_admission(width)), params, shapes.decoder)
    except ValueError as refused:
        if "mixed serving step" not in str(refused):
            raise
        return str(refused)
    return None


MIXED_LAYERS: frozenset[type[nn.Module]] = frozenset({CausalSelfAttention})
"""The layers that run a mixed call (`dew.nn.inputs.Admitted`) as the
separate decode and prefill calls would; a cache any other layer holds keeps
the two forwards, since a layer that does not read the metadata would treat
the step's one row of tokens as one sequence."""


def _cache_owners(model: nn.Module, params: Variables) -> dict[tuple[str, ...], type[nn.Module]]:
    """Each submodule's type by its path, so a cache leaf's holder can be named."""
    owners: dict[tuple[str, ...], type[nn.Module]] = {}

    def visit(module: nn.Module) -> None:
        module._try_setup()
        owners[tuple(module.path)] = type(module)
        for child in module._state.children.values():
            if isinstance(child, nn.Module):
                visit(child)

    visit(model.bind(params))
    return owners


def _probe_admission(width: int) -> Admission:
    """One admitted row of 64 tokens for `_mixed_refusal`'s trace, which
    reads its prompt, slot, cursor, page table and finality."""
    one = jnp.zeros(1, jnp.int32)
    return Admission(ModelInputs(jnp.zeros((1, 64), jnp.int32), {"attention_mask": jnp.zeros((1, 64), bool)}),
                     slots=one, budgets=one, keys=one, tables=jnp.zeros((1, width), jnp.int32), cursors=one,
                     final=jnp.zeros(1, bool), history=one, history_valid=one)


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

    The first iteration takes the admission, the draws come back stacked
    `[steps, slots]`, and the host throws the error when it reads them, one
    call later. The state comes back split as it went in.
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

    def cached(
        path: tuple[jax.tree_util.KeyEntry, ...], leaf: jax.ShapeDtypeStruct | jax.Array
    ) -> NamedSharding:
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
              options: dict[str, str] | None = None) -> jax.stages.Wrapped:
    """Compile the step program over a state split into resident and carried halves.

    The matrix half is donated so XLA updates the cache in place instead of
    copying it. `rows` places every leaf of an admission, whose rows come
    group by group, the way the slots are placed; None leaves them where they
    are. `options` are the program's own XLA options (`_program`); pass
    XLA_FLAGS to change the backend's defaults. See docs/performance.md for
    the measurements.
    """
    return jax.jit(_stepped, static_argnums=(0, 2, 3, 4), donate_argnums=(5,),
                   in_shardings=(None, resident, carried, rows, None, None, None),
                   out_shardings=(None, (resident, carried, None)), compiler_options=options)


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


_PROGRAMS: dict[tuple[tuple[Format, ...], str, NamedSharding | None],
                tuple[jax.stages.Wrapped, jax.stages.Wrapped]] = {}
"""One pair of step programs per resident layout; see `_program`."""

ADMISSION_OPTIONS = {"xla_gpu_enable_command_buffer": ""}
"""The admitting step's XLA options: no CUDA command buffers. Its inputs are
fresh buffers every call, so a command buffer is updated before it can
replay, and the device waited out the update: on an RTX 4080 serving
Qwen3-0.6B at 64 slots, about 5 ms before 9 of a run's 16 admitting steps
(docs/performance.md). Only a GPU backend is handed them."""


def _program(formats: Formats, rows: NamedSharding | None) -> tuple[jax.stages.Wrapped, jax.stages.Wrapped]:
    """`_compiled` over `formats`, once per layout, for the decoding step and
    the admitting one (`ADMISSION_OPTIONS`): one compile per (model,
    admission shape) serves every server that keeps its state in the same
    layout. The formats tree holds the cache mapping and is not hashable
    itself, so its leaves and its structure's text key the programs."""
    leaves, structure = jax.tree_util.tree_flatten(formats)
    key = (tuple(leaves), str(structure), rows)
    programs = _PROGRAMS.get(key)
    if programs is None:
        halves = _split(formats)
        options = ADMISSION_OPTIONS if jax.default_backend() == "gpu" else None
        programs = _PROGRAMS[key] = (_compiled(*halves, rows), _compiled(*halves, rows, options))
    return programs


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
        if type(decode_steps) is not int or decode_steps < 1:
            raise ValueError("decode_steps must be a positive number of iterations per device call")
        self.mesh = mesh_of(variables)
        self.groups = _row_groups(self.mesh)
        if slots % self.groups or admission % self.groups:
            raise ValueError(f"slots ({slots}) and admission ({admission}) must divide by the "
                             f"{self.groups} groups the mesh splits rows into")
        self.model = model
        self.variables = variables
        self._weight_groups = {}
        self.processor = processor
        if sampling.stop:
            raise ValueError("a server takes stop strings compiled into stopping; build it with "
                             "Server.from_task, which compiles the task's")
        self.pad_id = sampling.pad
        self.sampling = dataclasses.replace(sampling, pad_id=self.pad_id)
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
            for path, group in _projection_groups(model, variables).items():
                node = source_shapes["params"]
                for part in path:
                    node = node[part]
                name, projections, widths = group
                if name in node:
                    packed = node.pop(name)
                    for projection, width in zip(projections, widths, strict=True):
                        node[projection] = {field: jax.ShapeDtypeStruct((*leaf.shape[:-1], width), leaf.dtype)
                                            for field, leaf in packed.items()}
                    self._weight_groups[path] = group
            self._source_shapes = jax.tree_util.tree_flatten_with_path(source_shapes)[0]
            shapes = jax.eval_shape(functools.partial(_opened, model, pad_id=self.pad_id, slots=slots,
                                                      capacity=capacity), variables)
            rows.check(shapes.decoder.cache)
            # None selects the one mixed forward of `_mixed_step`.
            self.mixed_refusal = _mixed_refusal(model, self.variables, shapes, rows.placement, rows.width)
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
            formats = _resident_formats(
                model, self.variables, self.pad_id, rows.placement, decode_steps, shapes,
                _state_shardings(self.mesh, shapes), self._admitted, transforms, stopping, grammar)
            self._step, self._admitting = _program(formats, self._admitted)
            self._resident, self._carried = _split(
                _opened_in(formats)(model, self.variables, self.pad_id, slots, capacity))

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
        ceiling = _ceiling(task.model)
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
        # One program serves one capacity, so the cache holds whole tiles of it
        # rather than a power-of-two bucket: every decode step's attention reads
        # each slot, and a capacity of 384 bucketed to 512 read a third more.
        unit = math.lcm(64, layout.page_size or 64)
        rounded = -(-capacity // unit) * unit
        if rounded > ceiling:
            raise ValueError(
                f"a capacity of {capacity} rounds to {rounded}, over the model's max_seq_len of {ceiling}"
            )
        model = _sized(model, rounded)
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
        return _joined(self._resident, self._carried).decoder.cache

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
        packed = _pack_projections(normalized, self._weight_groups)
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
            inputs = _prepared(self.processor, prompt, images=None)
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
        valid = _check_inputs(self.model, ids, fields, budget, self.sampling, 1).astype(bool)
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
        through `TextGeneration` would.
        """
        base = _seed(key)
        inputs = _prepared(self.processor, prompts, images=None)
        valid = inputs.token_fields.get("attention_mask")
        rows = np.asarray(inputs.tokens)
        mask = np.ones(rows.shape, bool) if valid is None else np.asarray(valid).astype(bool)
        tickets = [self._enqueued(rows[index][mask[index]], max_new_tokens, base, index)
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
        if not any(pending):
            return None
        # Padded to the fewest rows that hold every group's pieces, not to the
        # whole admission: a request arriving alone prefills one row.
        share = admission_share(max(map(len, pending)), share)
        chosen = [(group * share + index, slot, row) for group, waiting in enumerate(pending)
                  for index, (slot, row) in enumerate(waiting)]
        # The capacity is whole tiles, not a bucket, so a piece's width bucket can pass it.
        width = min(_bucket(max(len(row.prompt) - row.prefilled for _, _, row in chosen), 64), self.capacity)
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
