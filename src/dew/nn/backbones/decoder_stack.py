"""How a decoder's layer stack runs, and how its variables are laid out while it does.

`run_stack` runs the layers one run at a time, each run of like layers under a
scan, and fetches the runs a store keeps banked or in host memory one layer at
a time. `run_pipeline` runs them as a GPipe pipeline over the mesh's stage
axis, one `PipelineStage` per stage. `StackView` converts the variables between
the per-layer tree a checkpoint stores and the stacked tree those loops read,
and `DecoderBank` declares where a decoder keeps its stack. Which of these a
call takes is the model's choice (`CausalTransformer.stack`).
"""

import dataclasses
import functools
from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.sharding import NamedSharding, PartitionSpec as P

from dew.objectives.base import merge

from ..hyper_connections import Carried
from ..kv_cache import CURSOR
from ..mla import INDEXER_COLLECTION
from ..moe import ROUTER_COLLECTION
from ..sharding import STAGE_AXIS, microbatched
from .decoder_block import DecoderBlock
from .layer_plan import LayerSpec, group_layers, group_name

Block = Callable[[int, str], DecoderBlock]
"""Layer `index`'s block under a module name: what the stack and its stages
build their layers from, so one factory describes every view of them."""

def _fetched(tree):
    """`tree` in device memory, the copy issued and not waited for.

    `jax.device_put` to a memory space alone keeps each leaf's sharding and
    dtype, so a fetched shard is the shard the layout placed on the host and
    the collectives its layer issues are the ones a resident run issues. A
    leaf already in device memory is not moved, so a bank the layout left
    resident reads the same way and gives the same values.
    """
    return jax.tree.map(lambda leaf: jax.device_put(leaf, jax.memory.Space.Device), tree)


def _fetched_layer(tree, index):
    """Copy the layer with its bank axis intact; squeeze only in device memory.

    TPU host tiles cannot in general be bitcast to the lower-rank shape.
    The bank's layer-major physical layout also keeps the copied tile valid.
    The index reaches the slice through an optimization barrier, so the
    slice stays the dynamic one a scanned layer takes: at a constant index
    (a run's first layer) over host rows a call stacks itself, XLA:GPU
    folds the slice to the row and the squeeze into the host parameter's
    own bitcast, which its layout assignment refuses ("Bitcast cannot have
    different memory spaces").
    """
    index = jax.lax.optimization_barrier(jnp.asarray(index, jnp.int32))
    return jax.tree.map(
        lambda leaf: jax.lax.squeeze(jax.device_put(jax.lax.dynamic_index_in_dim(leaf, index, 0),
                                                    jax.memory.Space.Device), (0,)), tree)


def _on_host(tree) -> bool:
    """Whether any leaf of `tree` sits in host memory."""
    return any(jax.typeof(leaf).memory_space is jax.memory.Space.Host
               for leaf in jax.tree.leaves(tree))


def _layer_slice(tree, index):
    return jax.tree.map(
        lambda leaf: jax.lax.dynamic_index_in_dim(leaf, index, 0, keepdims=False), tree)


def _layer_written(tree, values, index):
    return jax.tree.map(
        lambda bank, leaf: jax.lax.dynamic_update_index_in_dim(bank, leaf, index, 0),
        tree, values)


WRITTEN = ('cache', ROUTER_COLLECTION, 'qk', INDEXER_COLLECTION)
"""The collections a decoder block writes: its decode cache, and what its
router, its attention and its sparse indexer sow, which a looped pass keeps
apart per pass (`PassView`)."""


@dataclasses.dataclass(frozen=True)
class Loop:
    """A block of a layer stack that runs `steps` times over its own output,
    the passes sharing its parameters: a looped (universal) transformer, as
    Ouro (arXiv 2510.25741) runs its whole stack four times.

    `layers` is the block, layers [first, end): the layers before it (the
    prelude) and after it (the coda) run once, as Huginn's do, and a
    pretrained decoder gains a loop by repeating a block of its own. An
    empty block's passes run no layer, so the loop is its norms alone. None
    loops the whole stack. Every pass reads the same positions and masks.
    With `step_norm`, the decoder's final norm is applied between passes, as
    Ouro applies its shared `model.norm` after each one; the last pass's
    output takes the norm as any decoder's does. `backprop_steps` k trains
    through the last k passes alone: the passes before them run under
    `stop_gradient` (truncated backpropagation, as Huginn trains), so neither
    they nor the layers before them get a gradient. None trains through
    every pass. `exit_gate` adds Ouro's hazard head over the whole stack, a
    biased map of each pass's normed states to one logit, which the decoder
    sows into its `exits` collection; the decoder's output is always the
    last pass's.

    `remat` checkpoints each pass of the block, the first included, with one
    Flax lifted checkpoint. Backpropagation keeps each pass's input and
    recomputes the pass instead of retaining every layer's residuals for
    every pass. The prelude and coda keep their own policy, and decoding
    never checkpoints. A layer's own remat policy still applies inside the
    pass: with both enabled, the pass recomputes twice, once to recover the
    layers' inputs and again inside the layers' backward passes.

    With `reinject`, every pass reads the block's input added to the last
    pass's state under the step norm, the first pass's state that input
    itself, and the state a pass leaves is its output under the step norm,
    which the layers after the block read: Dory's looped nGPT
    (vllm-project/vllm#60149, dory.py:254-280). It needs `step_norm`. Under
    `backprop_steps` the cut stops the state alone, so the layers before the
    block still train through the input every pass after the cut reads, as
    Huginn's prelude does (arXiv 2502.05171, section 3.3).

    A decode cache holds one entry per pass and layer. The first pass keeps
    its layers' own names; pass t > 0's layer i is `layers_{L + (t - 1) n + i - first}`,
    L the decoder's layers and n the block's (`PassView`). Over the whole
    stack that is `layers_{t L + i}`, as Ouro's cache indexes them. With
    `shared_cache` every pass reads and writes its layers' one entry, each
    pass overwriting the step's keys and values the pass before wrote, so
    a step's earlier tokens read the last pass's (Dory's last-loop cache):
    a pass's cursor starts where the first pass's started (`SharedPassView`).
    """
    steps: int
    step_norm: bool = True
    backprop_steps: int | None = None
    exit_gate: bool = False
    layers: tuple[int, int] | None = None
    remat: bool = False
    reinject: bool = False
    shared_cache: bool = False

    def __post_init__(self):
        if self.steps < 1:
            raise ValueError(f"a loop runs its stack at least once, got steps={self.steps}")
        if self.reinject and (not self.step_norm or self.exit_gate):
            raise ValueError("a reinjecting loop norms each pass's state (step_norm) and exits at "
                             "the last pass")
        if self.backprop_steps is not None and not 1 <= self.backprop_steps <= self.steps:
            raise ValueError(f"backprop_steps counts passes trained through, 1 to steps={self.steps}, "
                             f"got {self.backprop_steps}")
        if self.layers is not None:
            first, end = (int(index) for index in self.layers)
            object.__setattr__(self, "layers", (first, end))
            if not 0 <= first <= end:
                raise ValueError(f"a loop's block is layers [first, end), first <= end, not {self.layers}")
            if self.exit_gate:
                raise ValueError("the exit gate reads the whole stack's output; its loop is the whole stack")


type Cursors = Mapping[tuple[str, ...], object]
"""A cache tree's cursors (`dew.nn.kv_cache.CURSOR`), by their paths."""


@dataclasses.dataclass
class SharedPassView:
    """The passes of a `Loop` with `shared_cache` over its block's own cache entries, which every
    pass reads and writes. The first pass's cursors are recorded as it finds them (`first`) and as
    it leaves them (`first_out`). A later pass sees every entry as the pass before left it but the
    cursors, which start where the first pass's started, so it writes the step's keys and values
    over the last pass's (`inside`), and it leaves the cursors as the first pass left them
    (`outside`): a pass that allocates the cache advances none (`cache_positions`), and the later
    ones, finding it allocated, must not either."""
    before: Cursors = dataclasses.field(default_factory=dict)
    after: Cursors = dataclasses.field(default_factory=dict)

    def first(self, variables: Mapping[str, Mapping]) -> dict:
        self.before = _cursors(variables)
        return dict(variables)

    def first_out(self, variables: Mapping[str, Mapping]) -> dict:
        self.after = _cursors(variables)
        return dict(variables)

    def inside(self, variables: Mapping[str, Mapping]) -> dict:
        return _with_cursors(variables, {**self.after, **self.before})

    def outside(self, variables: Mapping[str, Mapping]) -> dict:
        return _with_cursors(variables, self.after)


def _cursors(variables: Mapping[str, Mapping]) -> Cursors:
    flat = flatten_dict(variables.get('cache', {}))
    return {path: leaf for path, leaf in flat.items() if path[-1] == CURSOR}


def _with_cursors(variables: Mapping[str, Mapping], cursors: Cursors) -> dict:
    """`variables` with each cache cursor `cursors` names replaced."""
    cache = variables.get('cache')
    if cache is None:
        return dict(variables)
    placed = {path: cursors.get(path, leaf) for path, leaf in flatten_dict(cache).items()}
    return {**variables, 'cache': unflatten_dict(placed)}


@dataclasses.dataclass
class PassView:
    """The collections a stack writes (`WRITTEN`), as one pass of a `Loop`
    sees them: its `count` layers' entries stored from `layers_{offset}`
    shown under its layers' own names, from `layers_{first}`.

    `inside` shows the pass those entries (a run's name moves with its first
    layer) and hides every other layer's, and keeps the tree it was given;
    `outside` puts the pass's entries back beside the others'. A name that
    is not a layer's is every pass's. Flax calls `inside` before `outside`
    within one trace (`nn.map_variables`), which is what holding the tree
    relies on.
    """
    offset: int
    first: int
    count: int
    held: dict = dataclasses.field(default_factory=dict)

    @staticmethod
    def _moved(name: str, by: int) -> str:
        run = group_layers(name)
        if run is None:
            return name
        return f'layers_{run.start + by}' if len(run) == 1 else group_name(run.start + by, len(run))

    def _ours(self, name: str) -> bool:
        """Whether `name` is a layer entry this pass stores, or a name of no layer."""
        run = group_layers(name)
        return run is None or self.offset <= run.start < self.offset + self.count

    def inside(self, variables: Mapping[str, Mapping]) -> dict:
        self.held = {collection: dict(tree) for collection, tree in variables.items()}
        back = self.first - self.offset
        return {collection: {self._moved(name, back): value for name, value in tree.items()
                             if self._ours(name)}
                for collection, tree in self.held.items()}

    def outside(self, variables: Mapping[str, Mapping]) -> dict:
        out = {collection: {name: value for name, value in tree.items() if not self._ours(name)}
               for collection, tree in self.held.items()}
        for collection, tree in variables.items():
            out.setdefault(collection, {}).update(
                {self._moved(name, self.offset - self.first): value for name, value in tree.items()})
        return out


def _scanned_runs(runs, groups: Sequence[tuple[int, int]], specs: Sequence[LayerSpec], x, *, fetching: bool,
                  train: bool, decode: bool, kv_store, per_layer_input, attention_metadata):
    """`run_stack` without the inference prefetch: each run of one layer
    called as the plain loop calls it, and each longer run under flax's
    scan. Training, or a forward that keeps intermediates, through
    host-resident banks scans with the fetch mapped onto each row under remat
    (`run_stack`)."""
    for run, (first, count) in zip(runs, groups, strict=True):
        inputs = None if per_layer_input is None else per_layer_input.span(first, count)
        store = kv_store if count == 1 or specs[first].kv_shared else None

        def step(layer, carry, per_layer_input, store):
            # A provider writes its keys and values into the store it is given; inside a lifted
            # transform the writes come back explicitly, as `DecoderBlock.__call__`'s remat returns
            # them, so no tracer of the transform leaks through the dict.
            written = None if store is None else dict(store)
            out = layer(carry, train=train, decode=decode, kv_store=written,
                        per_layer_input=per_layer_input, attention_metadata=attention_metadata)
            changes = {} if written is None or store is None else {
                name: value for name, value in written.items() if value is not store.get(name)}
            return out, changes

        if fetching:
            # Scan variables are xs, not closed-over bank slices. Native
            # transposition stacks host cotangent rows instead of adding
            # a whole-bank device accumulator. Remat retains the original
            # host operand and refetches only this layer in backward
            # (MaxText layers/decoders.py:544-565).
            step = nn.remat(nn.map_variables(
                step, mapped_collections=True, trans_in_fn=_fetched, init=False, mutable=True))
        if count == 1:
            x, written = step(run, x, None if inputs is None else inputs.layer(0), store)
            if store is not None:
                store.update(written)
        else:
            # A longer run's layers share keys or own them, so they only read the store: a provider
            # is always a run of one (`run_stack`).
            def scanned(layer, carry, per_layer_input, *, store=store, step=step):
                return step(layer, carry, per_layer_input, store)[0], None

            x, _ = nn.scan(scanned, variable_axes={True: 0}, split_rngs={True: True},
                           in_axes=2, length=count)(run, x, inputs)
    return x


def run_stack(layers: Sequence[DecoderBlock], block: Block, specs: Sequence[LayerSpec],
              groups: Sequence[tuple[int, int]], x, *, train: bool, decode: bool,
              kv_store, per_layer_input, attention_metadata=None, banked: bool = False):
    """Run the layers over `x`, one run at a time as `groups` says.

    A run of one is `layers[first]`; a longer run is one block named for its
    range under flax's scan, its variables on a leading layer axis the view
    stacks and unstacks, each iteration reading its slice of the per-layer
    inputs. A run's layers all share or all own their keys and values: sharing
    layers read the store their providers filled (a closed-over constant), and
    owning layers get no store, since a Python dict cannot follow writes from
    inside the loop and a provider is always a run of one.

    `banked` says the store holds each run's parameters as one array
    (`dew.inference.banks`). Those runs, and runs the layout left in host
    memory, are read one layer at a time in `_prefetched_run`: at most two of a
    run's layers are on the device, staging crosses run boundaries, and no fetch
    can be hoisted above the layer before it (it is issued inside that layer's
    iteration or ordered by the carry). Training uses the native Linen scan
    instead, with map_variables staging one row under remat, so the backward
    refetches the pinned bank rather than keeping a device copy. Each fetched
    layer binds its variables through Linen's `Module.bind`, which keeps the
    enclosing call's intermediate-capture policy active.
    """
    runs = [layers[first] if count == 1 else block(first, group_name(first, count))
            for first, count in groups]
    streaming = any('streaming' in run.variables for run in runs)
    if streaming and train:
        raise ValueError("disk streaming is inference-only")
    fetching = streaming or banked or any(_on_host(run.variables.get('params', {})) for run in runs)
    if not fetching or train:
        return _scanned_runs(runs, groups, specs, x, fetching=fetching, train=train, decode=decode,
                             kv_store=kv_store, per_layer_input=per_layer_input,
                             attention_metadata=attention_metadata)

    def read_only(run: DecoderBlock) -> list[str]:
        """List the collections a run reads and does not write.

        These are its parameters, which a forward never writes, and whatever
        else the caller gave it and left immutable, all of which a bank holds
        per layer.
        """
        return [name for name in run.variables
                if name != 'streaming' and (name == 'params' or not run.is_mutable_collection(name))]

    def held(run: DecoderBlock) -> dict:
        """The values the caller holds in the collections it opened, but the
        cache, which the loop carries apart: the run reads them as it writes
        them, as a resident run does, one layer's slice at a time."""
        return {name: tree for name, tree in run.variables.items()
                if name not in ('params', 'streaming', 'cache') and run.is_mutable_collection(name)}

    def first_of(index: int, dependency):
        """Fetch run `index`'s first layer's read-only variables.

        Returns None past the last run: the copy nothing computes with is
        the one not issued.
        """
        if index >= len(runs):
            return None
        reader = runs[index].variables.get('streaming', {}).get('bank')
        if reader is not None:
            return reader.fetch(0, dependency)
        held = {name: runs[index].variables[name] for name in read_only(runs[index])}
        return _fetched(held) if groups[index][1] == 1 else _fetched_layer(held, 0)

    probed: list[tuple[LayerSpec, list[str]]] = []
    staged = first_of(0, x)
    for index, (run, (first, count)) in enumerate(zip(runs, groups, strict=True)):
        inputs = None if per_layer_input is None else per_layer_input.span(first, count)
        store = kv_store if count == 1 or specs[first].kv_shared else None
        state = held(run)
        cached = (run.variables.get('cache') or None) if run.is_mutable_collection('cache') else None
        first_slice = None if inputs is None else inputs.layer(0)
        opened = _opened(run, specs[first], probed, staged, state if count == 1 else _layer_slice(state, 0),
                         cached if count == 1 or cached is None else _layer_slice(cached, 0), x,
                         first_slice, store=store, train=train, decode=decode,
                         attention_metadata=attention_metadata)

        layer = functools.partial(_applied, run, store, opened=opened, train=train, decode=decode,
                                  attention_metadata=attention_metadata)

        if count == 1:
            following = first_of(index + 1, x)
            x, changed = layer(staged, state, cached, x, first_slice)
        else:
            banks = {name: run.variables[name] for name in read_only(run)}
            x, changed, following = _prefetched_run(
                banks, staged, state, cached, x, inputs, layer, count,
                following=functools.partial(first_of, index + 1),
                reader=run.variables.get('streaming', {}).get('bank'))
        staged = following
        for collection, tree in changed.items():
            for name, value in tree.items():
                run.put_variable(collection, name, value)
    return x


def _applied[Hidden: (jax.Array, Carried)](run: DecoderBlock, store, read, state, cache, hidden: Hidden,
                     per_layer_slice, *, opened: bool | list[str], train: bool, decode: bool,
                     attention_metadata) -> tuple[Hidden, dict]:
    """One layer of `run` applied in a scope of its own: `read` its read-only variables, `state`
    the caller's held collections and `cache` its cache slice, writing the `opened` collections
    (`run_stack`). Binding instead of nesting `apply` leaves Linen's enclosing
    capture policy active, including a filter over individual sublayers.
    The hidden state comes out as it went in, a residual stream or `Carried` streams."""
    variables = {**read, **state} if cache is None else {**read, **state, 'cache': cache}
    bound = run.bind(variables, mutable=opened)
    hidden = bound(hidden, train=train, decode=decode, kv_store=store,
                   per_layer_input=per_layer_slice, attention_metadata=attention_metadata)
    return hidden, {name: tree for name, tree in bound.variables.items()
                    if bound.is_mutable_collection(name)}


def _opened(run: DecoderBlock, spec: LayerSpec, probed: list[tuple[LayerSpec, list[str]]], read, state,
            cache, hidden, per_layer_slice, *, store, train: bool, decode: bool,
            attention_metadata) -> list[str]:
    """The collections a run applied in a scope of its own opens: those its caller
    opened that a layer of it writes into, DSpark's `prediction_inputs` among them,
    which a layer reads to choose what it records, and the ones the caller holds
    (`state`), which it reads as it writes them; the loop carries them back out.

    What a layer writes with every collection open is read off an abstract run of
    the run's first layer. Layers of equal specs run the same program, so each
    spec runs once, recorded in `probed`.
    """
    written = next((names for seen, names in probed if seen == spec), None)
    if written is None:
        variables = {**read, **state} if cache is None else {**read, **state, 'cache': cache}
        # A provider writes its keys and values into the store it is given: the
        # abstract run writes into a copy, so nothing it traces leaves it.
        _, changed = jax.eval_shape(lambda variables: _applied(
            run, None if store is None else dict(store), variables, {}, None, hidden,
            per_layer_slice, opened=True, train=train, decode=decode,
            attention_metadata=attention_metadata), variables)
        written = list(changed)
        probed.append((spec, written))
    # Capture filters may select one module name but not another with the same layer spec. Keep
    # that collection open wherever the caller does, independently of the spec's write probe.
    return sorted(name for name in {*written, *state, 'intermediates'} if name not in ('params', 'streaming')
                  and run.is_mutable_collection(name))


def _prefetched_run(banks, primed, state, cache, x, inputs, layer, count: int, *, following, reader=None):
    """Run `count` layers under `jax.lax.scan`, read one layer at a time.

    `primed` is layer 0's read-only variables, already on the device. Iteration
    `i` issues layer `i + 1`'s copy, then computes layer `i`; the last layer is
    computed after the loop and `following` stages the next run's first layer in
    place of its copy. Each layer reads its slice of `state`, the caller's
    opened collections, and their values come out stacked per iteration, as
    flax's scan hands them out. The carry is the hidden state, the staged
    variables and the cache, written in place one layer's slice per iteration
    so a decode step holds one banked cache; a cache the layers create comes
    out stacked like `state`.
    """
    def body(carry, index):
        hidden, current, held = carry
        staged = _fetched_layer(banks, index + 1) if reader is None else reader.fetch(index + 1, hidden)
        per_layer_slice = None if inputs is None else inputs.layer(index)
        hidden, changed = layer(current, _layer_slice(state, index),
                                None if held is None else _layer_slice(held, index),
                                hidden, per_layer_slice)
        if held is not None:
            held = _layer_written(held, changed.pop('cache'), index)
        return (hidden, staged, held), changed

    (x, current, cache), sown = jax.lax.scan(body, (x, primed, cache), jnp.arange(count - 1))
    last = count - 1
    staged = following(x)
    x, changed = layer(current, _layer_slice(state, last),
                       None if cache is None else _layer_slice(cache, last), x,
                       None if inputs is None else inputs.layer(last))
    if cache is not None:
        cache = _layer_written(cache, changed.pop('cache'), last)
    changed = jax.tree.map(
        lambda rows, final: jnp.concatenate([rows, final[None]]), sown, changed)
    if cache is not None:
        changed['cache'] = cache
    return x, changed, staged


class PipelineStage(nn.Module):
    """Runs the layers of one pipeline stage over one microbatch.

    The pipeline vmaps this module over the stage axis, so each stage runs it
    on its own slice of the stacked layer weights. Its variables therefore have
    a leading stage axis, which the `StackView` outside stacks and unstacks.
    Layer `j` of a stage is `layers_j` here and `layers_{stage * count + j}` in
    the stored tree, where `count` is the number of layers per stage. `specs`
    and `groups` describe stage 0, and every stage repeats them.
    """
    block: Block
    specs: tuple[LayerSpec, ...]
    groups: tuple[tuple[int, int], ...]

    def setup(self):
        self.layers = [self.block(index, f'layers_{index}') for index in range(len(self.specs))]

    @nn.compact
    def __call__(self, x, train: bool = False, per_layer_input=None, attention_metadata=None):
        return run_stack(self.layers, self.block, self.specs, self.groups, x,
                         train=train, decode=False, kv_store=None,
                         per_layer_input=per_layer_input, attention_metadata=attention_metadata)


def stack_layers(*trees):
    """Layers' trees stacked on a new leading layer axis, a bank."""
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *trees)


def _refuse_unlike(collection: str, trees: Sequence, names: Sequence[str]) -> None:
    """Refuse layers one bank or the pipeline stages' like positions read
    whose `collection` differs in structure, as layers a checkpoint quantizes
    differently do: a bank is one array per leaf, and every stage runs stage
    0's body. Layers held apart run as scanned runs of their own instead
    (`StackView.layer_structures`)."""
    structures = [jax.tree.structure(tree) for tree in trees]
    unlike = [name for name, structure in zip(names, structures, strict=True) if structure != structures[0]]
    if unlike:
        raise ValueError(f"collection {collection!r} differs between {names[0]} and {', '.join(unlike)}, "
                         "which one bank or pipeline stage reads as like layers; hold these layers "
                         "unbanked, on one stage")


@dataclasses.dataclass(frozen=True)
class StackView:
    """Converts the layer stack's variables between their stored layout and the one its loops read.

    Outside a loop, every collection holds one subtree per layer, `layers_N`.
    That is the tree a checkpoint stores and a Hugging Face loader writes.
    Inside a scanned run, the leaves are stacked on a leading layer axis under
    the run's name (`layers_3_7`). Inside a pipeline, they are stacked on a
    leading stage axis under `stages`. `stack` converts to the inside layout and
    `unstack` converts back, so whatever a run reads, sows and caches ends up
    where the plain loop would put it.

    The `banked` collections are already stored the inside way, one array per
    run, and both methods leave them alone. `bank_names` returns the name of the
    bank that holds each run's layers. Outside a scope, `unstack` reads one
    layer's slice at a time, for a save or an export. Inside a pipeline, a
    collection that entered it has `[stage, ...]` leaves, and one the loop
    created has `[iteration, stage, microbatch, ...]` leaves; a stage's real
    iterations are its microbatches, in order, and a leaf that holds rows
    takes them back in the batch's order (`_whole`). `batch_axis` is where
    the stack's residual holds its rows, after AltUp's copies: the pipeline
    cuts the residual into microbatches there (`run_pipeline`), and a
    layer's own output (what `capture_intermediates` keeps of its
    `__call__`) holds them there too; what its sublayers sow holds them first.
    """
    groups: tuple[tuple[int, int], ...]
    stages: int = 1
    microbatches: int = 1
    broadcast: tuple[str, ...] = ()
    banked: tuple[str, ...] = ()
    batch_axis: int = 0

    @property
    def per_stage(self) -> int:
        return sum(count for _, count in self.groups)

    def _inside_name(self, first: int, count: int) -> str | None:
        """A run's name inside, None for a layer the view leaves as it is."""
        if count > 1:
            return group_name(first, count)
        return f'layers_{first}' if self.stages > 1 else None

    def _outside_names(self, first: int, count: int) -> list[list[str]]:
        """The stored names of a run's layers, one list per stage."""
        return [[f'layers_{stage * self.per_stage + first + offset}' for offset in range(count)]
                for stage in range(self.stages)]

    @staticmethod
    def layer_structures(variables: Mapping[str, Mapping], layers: int) -> tuple[tuple, ...]:
        """What one scanned body has to share beyond its spec, for each of `layers`
        stored layers: the runs a collection holds as one bank (`layers_3_7`) that
        hold it, and the structure of its subtree in every collection that keeps
        one per layer (`layers_N`), as layers a checkpoint quantizes differently,
        or adapts in part, differ. A banked run stays the run its store holds
        (`bank_sites`), each of its layers read as its first, so rows it holds
        unlike are refused by name where they stack (`StackView.stack`). A
        collection no layer is held in adds nothing."""
        banks: list[tuple[str, ...]] = [()] * layers
        read = list(range(layers))
        for name in sorted({name for tree in variables.values() for name in tree}):
            run = group_layers(name)
            if run is not None and len(run) > 1:
                for index in range(run.start, min(run.stop, layers)):
                    banks[index] += (name,)
                    read[index] = min(read[index], run.start)
        own = [tuple((collection, jax.tree.structure(tree[f'layers_{index}']))
                     for collection, tree in sorted(variables.items()) if f'layers_{index}' in tree)
               for index in range(layers)]
        return tuple((held, own[first]) for held, first in zip(banks, read, strict=True))

    def bank_names(self) -> list[str]:
        """Return every run's stored name, in order. A banked store uses these names as keys."""
        return [self._inside_name(first, count) or f'layers_{first}'
                for first, count in self.groups]

    def stack(self, variables: Mapping[str, Mapping]) -> dict:
        inside = {}
        for collection, tree in variables.items():
            tree = dict(tree)
            if collection in self.banked:
                inside[collection] = tree
                continue
            stages = {}
            for first, count in self.groups:
                name = self._inside_name(first, count)
                if name is None:
                    continue
                names = self._outside_names(first, count)
                held = [layer in tree for stage in names for layer in stage]
                if not any(held):
                    continue
                if not all(held):
                    raise ValueError(
                        f"collection {collection!r} holds some of the layers "
                        f"{sorted(layer for stage in names for layer in stage)} and not "
                        "the others; a run reads every one of its layers or none")
                _refuse_unlike(collection, [tree[layer] for stage in names for layer in stage],
                               [layer for stage in names for layer in stage])
                runs = [stack_layers(*[tree.pop(layer) for layer in stage]) if count > 1
                        else tree.pop(stage[0]) for stage in names]
                if self.stages == 1:
                    # A bank already stored under the run's name holds the
                    # leaves its rows lack (`banked_collections`); the rows
                    # stack into it.
                    tree[name] = merge(tree[name], runs[0]) if name in tree else runs[0]
                else:
                    stages[name] = jax.tree.map(_on_stage_axis, stack_layers(*runs))
            if stages:
                tree['stages'] = stages
            inside[collection] = tree
        return inside

    def unstack(self, variables: Mapping[str, Mapping]) -> dict:
        outside = {}
        for collection, tree in variables.items():
            tree = dict(tree)
            if collection in self.banked:
                outside[collection] = tree
                continue
            stages = dict(tree.pop('stages', {})) if self.stages > 1 else tree
            for first, count in self.groups:
                name = self._inside_name(first, count)
                if name is None or name not in stages:
                    continue
                view = stages.pop(name)
                for stage, names in enumerate(self._outside_names(first, count)):
                    for offset, layer in enumerate(names):
                        tree[layer] = jax.tree_util.tree_map_with_path(
                            functools.partial(self._leaf, stage, offset if count > 1 else None,
                                              collection in self.broadcast), view)
            if self.stages > 1 and stages:
                tree['stages'] = stages
            outside[collection] = tree
        return outside

    def _leaf(self, stage: int, offset: int | None, broadcast: bool, path, leaf):
        """One layer's leaf, at `path` below its run, out of a run's stacked one, taken with
        `index_in_dim`: an index array would be a gather, which over a host-resident bank needs
        host-memory indices a save or export does not hold.
        """
        if self.stages == 1:
            assert offset is not None, "a run of one is not stacked, so the view keeps it"
            return jax.lax.index_in_dim(leaf, offset, axis=0, keepdims=False)
        if broadcast:
            leaf = leaf[stage]
            return leaf if offset is None else leaf[offset]
        real = leaf[stage:stage + self.microbatches, stage]
        if offset is not None:
            real = real[:, offset]
        # A count a layer sows once a microbatch stays one per microbatch.
        if real.ndim == 1:
            return real
        output = path[:1] == (jax.tree_util.DictKey("__call__"),)
        return _whole(real, self.batch_axis if output else 0)


@dataclasses.dataclass(frozen=True)
class DecoderBank:
    """Where one decoder's layer stack is stored, below every variables collection.

    `namespace` is the module path to the decoder. A container module adds its
    own name in front of the namespace and leaves `view` unchanged, so the layer
    groups, module names and RNG streams stay the decoder's. A shared scope is
    one site, even when several methods read the same parameters.
    """
    namespace: tuple[str, ...]
    view: StackView
    scanned: bool = True
    """Whether the stack runs its banks under `scan`. The scan is what makes a
    host-resident bank's fetches happen one row at a time. A plain loop declares
    the same banks, but nothing orders their fetches, so the compiler moves
    every layer's copy to the front and the whole stack is on the device at
    once. A host layout therefore refuses a plain loop
    (`dew.training.execution.resident`)."""


def _on_stage_axis(leaf):
    """`leaf`, its leading dimension placed on the stage axis of the mesh in context."""
    return jax.lax.with_sharding_constraint(leaf, _stage_sharding(leaf.ndim))


def _stage_sharding(ndim: int) -> NamedSharding:
    """The leading dimension on the stage axis; the rest as the layout and the
    batch's placement propagate them."""
    return NamedSharding(jax.sharding.get_abstract_mesh(),
                         P(STAGE_AXIS, *([P.UNCONSTRAINED] * (ndim - 1))))


def _whole(value, axis: int):
    """The batch `microbatched` cut, back in one piece and in its order."""
    moved = jnp.moveaxis(value, 0, axis + 1)
    shape = moved.shape
    return moved.reshape((*shape[:axis], shape[axis] * shape[axis + 1], *shape[axis + 2:]))


def run_pipeline(model: nn.Module, view: StackView, x, *, train: bool, per_layer_input,
                 attention_metadata=None):
    """GPipe over the stage axis, as MaxText's `layers/pipeline.py` runs it.

    Iteration t runs stage s on microbatch t - s, the layers stacked over the
    stages under `jax.vmap` so GSPMD keeps each stage on its devices, and
    outputs shift a stage down by `ppermute`. The microbatches sit in
    `state_io`, `[stages, microbatches / stages, ...]`, stage-sharded: stage 0
    reads slot t % (microbatches / stages) and the slots rotate up a stage per
    iteration, so every microbatch reaches stage 0 and finishes in the last
    stage's slot without a gather. The first and last stages - 1 iterations are
    the bubble. Only the layers pipeline; embeddings, norm, head and loss run on
    the whole batch, so the loss is the plain loop's mean.

    `model` is the decoder whose `_stacked` calls this; its `block` and
    `specs` build the stages.
    """
    stages, count, batch_axis = view.stages, view.microbatches, view.batch_axis
    per_stage = view.per_stage
    micro = functools.partial(microbatched, count=count)
    x = micro(x, batch_axis)
    inputs = (
        None if per_layer_input is None else jax.tree.map(lambda value: micro(value, 0), per_layer_input)
    )
    metadata = jax.tree.map(lambda value: micro(value, 0), attention_metadata)
    slots = count // stages
    state_io = _on_stage_axis(x.reshape((stages, slots, *x.shape[1:])))
    shift = _on_stage_axis(jnp.zeros((stages, *x.shape[1:]), x.dtype))
    mesh = jax.sharding.get_abstract_mesh()
    stage_ids = jnp.arange(stages)

    @functools.partial(jax.shard_map, mesh=mesh, in_specs=P(STAGE_AXIS),
                       out_specs=P(STAGE_AXIS), axis_names={STAGE_AXIS})
    def shift_down(out):
        """Each stage's output as the next stage's input; the first gets nothing."""
        out = jax.lax.ppermute(out, STAGE_AXIS, [(s, (s + 1) % stages) for s in range(stages)])
        return jnp.where(jax.lax.axis_index(STAGE_AXIS) == 0, jnp.zeros_like(out), out)

    @functools.partial(jax.shard_map, mesh=mesh, in_specs=(P(STAGE_AXIS), P(STAGE_AXIS)),
                       out_specs=P(STAGE_AXIS), axis_names={STAGE_AXIS})
    def rotate_up(slot, out):
        """The slot one stage up, the last stage's taking the finished microbatch."""
        slot = jax.lax.ppermute(slot, STAGE_AXIS, [(s, (s - 1) % stages) for s in range(stages)])
        return jnp.where(jax.lax.axis_index(STAGE_AXIS) == stages - 1, out, slot)

    def gather(values, ids):
        """Each stage's microbatch out of `[microbatches, ...]` values."""
        if values is None:
            return values
        return _on_stage_axis(jax.vmap(
            lambda index: jax.lax.dynamic_index_in_dim(values, index, 0, keepdims=False))(ids))

    def call_stage(stage, x, per_layer_input, attention_metadata):
        return stage(x, train=train, per_layer_input=per_layer_input, attention_metadata=attention_metadata)

    def iteration(module, carry, step):
        state_io, shift = carry
        slot = step % slots
        stream = jax.lax.dynamic_index_in_dim(state_io, slot, 1, keepdims=False)
        stages_in = _on_stage_axis(jnp.where(
            jax.lax.broadcasted_iota(jnp.int32, shift.shape, 0) == 0, stream, shift))
        ids = jnp.clip(step - stage_ids, 0, count - 1)
        stage_inputs = (None if inputs is None else jax.tree.map(_on_stage_axis, jax.vmap(
            lambda index, stage: jax.tree.map(lambda value: jax.lax.dynamic_slice_in_dim(
                jax.lax.dynamic_index_in_dim(value, index, 0, keepdims=False),
                stage * per_stage, per_stage, axis=2), inputs))(ids, stage_ids)))
        stage = PipelineStage(block=module.block, specs=module.specs[:per_stage],
                              groups=view.groups, name='stages')
        run = nn.vmap(call_stage, variable_axes={True: 0}, split_rngs={True: True},
                      in_axes=0, out_axes=0, spmd_axis_name=STAGE_AXIS)
        out = _on_stage_axis(run(stage, stages_in, stage_inputs,
                                 jax.tree.map(lambda value: gather(value, ids), metadata)))
        state_io = jax.lax.dynamic_update_index_in_dim(
            state_io, rotate_up(stream, out), slot, 1)
        return (state_io, shift_down(out)), None

    loop = nn.scan(iteration, variable_broadcast=view.broadcast,
                   variable_axes={True: 0}, split_rngs={True: True})
    (state_io, _), _ = loop(model, (state_io, shift), jnp.arange(count + stages - 1))
    # Microbatch 0 finished stages - 1 iterations in, so it sits that
    # many slots along; the rest follow it in order.
    order = (np.arange(slots) + (stages - 1) % slots) % slots
    finished = state_io[:, order].reshape((count, *x.shape[1:]))
    return _whole(finished, batch_axis)
