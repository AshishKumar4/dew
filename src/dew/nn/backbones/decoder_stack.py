"""How a decoder's layer stack runs, and how its variables are laid out while it does.

`run_stack` runs the layers one run at a time, each run of like layers under a
scan, and fetches the runs a store keeps banked or in host memory one layer at
a time. `run_pipeline` runs them as a GPipe pipeline over the mesh's stage
axis, one `PipelineStage` per stage; it runs any stack of like blocks so, a
DiT's, a ViT's or a UNet level's transformers (`pipelined_blocks`, one
`BlockStage` per stage). `StackView` converts the variables between
the per-layer tree a checkpoint stores and the stacked tree those loops read,
and `DecoderBank` declares where a decoder keeps its stack. Which of these a
call takes is the model's choice (`CausalTransformer.stack`).
"""

import dataclasses
import functools
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Concatenate, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.core import meta
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.sharding import NamedSharding, PartitionSpec as P

from dew.batch import Variables, merge, part
from dew.mesh import STAGE_AXIS, LayoutRefused, row_axes
from dew.nn.protocols import StagedStacks

from ..hyper_connections import Carried
from ..inspection import counted, current_positions, watched_positions
from ..kv_cache import cache_at_layer, cache_from_layer
from ..mla import INDEXER_COLLECTION
from ..moe import BANKED_EXPERTS, ROUTER_COLLECTION, at_iteration, experts_apart, holds_experts
from ..sharding import microbatched, microbatches, pipeline_stage, pipeline_stages
from .decoder_block import DecoderBlock
from .layer_names import array_rows, group_layers, group_name, stage_count
from .layer_plan import LayerSpec, run_period

Block = Callable[[int, str], DecoderBlock]
"""Layer `index`'s block under a module name: what the stack and its stages
build their layers from, so one factory describes every view of them."""

StageLayer = Callable[[int], nn.Module | tuple[nn.Module, ...]]
"""A stack of blocks' layer `j` of a stage (`BlockStage`): its block, or the
modules it repeats per layer (a factorized stack's spatial and temporal pair)."""

type RowStack[Leaf] = Callable[[list[Leaf], tuple[int, ...]], Leaf]
"""How a run's rows become one leaf (`StackView.stage_form`): the rows, stage-major, and the leading
dimensions they fill; arrays, their shapes or their shardings."""

type RowCut[Leaf] = Callable[[tuple, Leaf, tuple[int, ...]], Sequence[Leaf]]
"""How a stacked leaf at a path becomes its rows (`StackView.layer_form`), `RowStack`'s inverse."""

StageRun = Callable[[list, jax.Array, jax.Array | None], jax.Array]
"""How a stack of blocks runs a stage's layers over a microbatch, with its rows
of the inputs held by row, None where it holds none (`BlockStage`)."""

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


def on_host(tree) -> bool:
    """Whether any leaf of `tree` sits in host memory."""
    return any(jax.typeof(leaf).memory_space is jax.memory.Space.Host
               for leaf in jax.tree.leaves(tree))


def _apart(held: dict) -> tuple[dict, dict]:
    """A run's read-only variables as its layers read them from their banks: those in host memory, which
    a layer is fetched as a layer at a time, and those in device memory, which each layer reads where they
    lie (`_with`): the expert kernels whole (`BANKED_EXPERTS`, each the run's stack,
    `dew.nn.moe.experts_apart`) and the rest sliced in the iteration that reads them, where the reading
    fusion takes the slice. Fetched, a layer is a copy the iteration before stages: 1.2 GB a layer at
    Qwen3-30B-A3B's experts, half a decode step on an H100, and 0.29 ms a step of its attention's
    projections, norms and router over 8 layers (OM-1342-cache-experts-2f9c0a176)."""
    flat = flatten_dict(held)
    fetched = unflatten_dict({path: leaf for path, leaf in flat.items() if on_host(leaf)})
    in_place = unflatten_dict({path: leaf for path, leaf in flat.items() if not on_host(leaf)})
    params = in_place.get('params')
    if params is not None and holds_experts(params):
        in_place = {**in_place, **experts_apart({'params': params})}
    return fetched, in_place


def _with(read, in_place: dict, index):
    """A layer's read-only variables: those fetched for it, and those it reads in place at its `index` in
    the run (`_apart`): its slice of each, and the expert kernels whole beside the index
    (`dew.nn.moe.at_iteration`)."""
    if not in_place:
        return read
    sliced = merge(read, _layer_slice({name: tree for name, tree in in_place.items()
                                       if name != BANKED_EXPERTS}, index))
    return sliced if BANKED_EXPERTS not in in_place else {**sliced, **at_iteration(in_place, index)}


def _layer_slice(tree, index):
    return jax.tree.map(
        lambda leaf: jax.lax.dynamic_index_in_dim(leaf, index, 0, keepdims=False), tree)


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
    each pass starts from the cache as the step entered it (`SharedPassView`).
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


@dataclasses.dataclass
class SharedPassView:
    """The passes of a `Loop` with `shared_cache` over its block's own cache entries, which every
    pass reads and writes. Each later pass starts from the cache as the first pass found it
    (`first`), every entry: its cursors where the first pass's started, so it writes the step's keys
    and values over the last pass's, and each recurrent mixer's state, convolution tail and memory
    as the step entered them, so it takes the step's tokens once rather than again after the pass
    before took them. A pass that allocates the cache finds none, so the later ones allocate too and
    advance nothing (`cache_positions`). The step leaves the cache as its last pass did."""
    entering: Mapping | None = None

    def first(self, variables: Mapping[str, Mapping]) -> dict:
        self.entering = variables.get('cache')
        return dict(variables)

    def inside(self, variables: Mapping[str, Mapping]) -> dict:
        rest = {collection: tree for collection, tree in variables.items() if collection != 'cache'}
        return rest if self.entering is None else {**rest, 'cache': self.entering}

    @staticmethod
    def outside(variables: Mapping[str, Mapping]) -> dict:
        return dict(variables)


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


def _by_repeat(period: int, leaf):
    """A per-layer input's `[B, S, L, ...]` leaf as `[B, S, L / period, period, ...]`."""
    return leaf.reshape(*leaf.shape[:2], leaf.shape[2] // period, period, *leaf.shape[3:])


def _run_module(block: Block, specs: Sequence[LayerSpec], first: int, count: int,
                period: int) -> "DecoderBlock | PipelineStage":
    """The module a scanned run of `count` layers from `first` repeats: one layer for like layers,
    for a periodic run its `period` layers as a pipeline's stage holds its own (`PipelineStage`), each
    run alone, named for the run's range. No layer of a period leaves what a later one reads (a
    provider), so it threads no store."""
    if period == 1:
        return block(first, group_name(first, count))
    return PipelineStage(block=block, specs=tuple(specs[first:first + period]),
                         groups=tuple((index, 1) for index in range(period)), offset=first,
                         name=group_name(first, count))


def _scanned_runs(runs, groups: Sequence[tuple[int, int]], specs: Sequence[LayerSpec], x, *, fetching: bool,
                  train: bool, decode: bool, kv_store, per_layer_input, attention_metadata):
    """`run_stack` without the inference prefetch: each run of one layer
    called as the plain loop calls it, and each longer run under flax's
    scan, a periodic run's (a `PipelineStage` of its period) over its repeats. Training, or a
    forward that keeps intermediates, through host-resident banks scans with
    the fetch mapped onto each row under remat (`run_stack`)."""
    for run, (first, count) in zip(runs, groups, strict=True):
        inputs = None if per_layer_input is None else per_layer_input.span(first, count)
        periodic = isinstance(run, PipelineStage)
        store = kv_store if count == 1 or specs[first].kv_shared else None
        if periodic:
            store = None
            if inputs is not None:
                # The layer axis, split into repeats of the period, scans over the repeats.
                inputs = jax.tree.map(functools.partial(_by_repeat, len(run.specs)), inputs)

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

            repeats = count // len(run.specs) if periodic else count
            # A parameter its module boxed (`nn.with_logical_partitioning`) stacks with an unnamed layer
            # axis, which the run's view cuts off again per layer (`_rows`).
            x, _ = nn.scan(scanned, variable_axes={True: 0}, split_rngs={True: True},
                           in_axes=2, length=repeats,
                           metadata_params={nn.PARTITION_NAME: None})(run, x, inputs)
    return x


def run_stack(layers: Sequence[DecoderBlock], block: Block, specs: Sequence[LayerSpec],
              groups: Sequence[tuple[int, int]], x, *, train: bool, decode: bool,
              kv_store, per_layer_input, attention_metadata=None, banked: bool = False,
              periods: Sequence[int] = ()):
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
    memory, are read one layer at a time in `_prefetched_run`: a bank in device
    memory where it lies (`_apart`), and one in host memory fetched, at most two
    of a run's layers on the device, staging crossing run boundaries, and no
    fetch hoisted above the layer before it (it is issued inside that layer's
    iteration or ordered by the carry). Training uses the native Linen scan
    instead, with map_variables staging one row under remat, so the backward
    refetches the pinned bank rather than keeping a device copy. Each fetched
    layer binds its variables through Linen's `Module.bind`, which keeps the
    enclosing call's intermediate-capture policy active.

    `periods` is each run's period (`StackView.periods`): a run of a longer
    one repeats its period's layers under the scan (`_run_module`), which
    the stack runs over resident weights alone (`CausalTransformer.stack`).
    """
    if any(period > 1 for period in periods):
        periodic = [layers[first] if count == 1 else _run_module(block, specs, first, count, period)
                    for (first, count), period in zip(groups, periods, strict=True)]
        return _scanned_runs(periodic, groups, specs, x, fetching=False, train=train, decode=decode,
                             kv_store=kv_store, per_layer_input=per_layer_input,
                             attention_metadata=attention_metadata)
    runs = [layers[first] if count == 1 else block(first, group_name(first, count))
            for first, count in groups]
    streaming = any('streaming' in run.variables for run in runs)
    if streaming and train:
        raise ValueError("disk streaming is inference-only")
    fetching = streaming or banked or any(on_host(run.variables.get('params', {})) for run in runs)
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
        return _fetched(held) if groups[index][1] == 1 else _fetched_layer(_apart(held)[0], 0)

    probed: list[tuple[LayerSpec, list[str]]] = []
    staged = first_of(0, x)
    for index, (run, (first, count)) in enumerate(zip(runs, groups, strict=True)):
        inputs = None if per_layer_input is None else per_layer_input.span(first, count)
        store = kv_store if count == 1 or specs[first].kv_shared else None
        state = held(run)
        cached = (run.variables.get('cache') or None) if run.is_mutable_collection('cache') else None
        first_slice = None if inputs is None else inputs.layer(0)
        in_place = {} if count == 1 else _apart({name: run.variables[name] for name in read_only(run)})[1]
        opened = _opened(run, specs[first], probed, _with(staged, in_place, 0),
                         state if count == 1 else _layer_slice(state, 0),
                         cached if count == 1 or cached is None else _layer_slice(cached, 0), x,
                         first_slice, store=store, train=train, decode=decode,
                         attention_metadata=attention_metadata)

        layer = functools.partial(_applied, run, store, opened=opened, train=train, decode=decode,
                                  attention_metadata=attention_metadata)

        if count == 1:
            following = first_of(index + 1, x)
            x, changed = layer(staged, state, cached, x, first_slice)
        else:
            banks = _apart({name: run.variables[name] for name in read_only(run)})[0]
            x, changed, following = _prefetched_run(
                banks, staged, state, cached, x, inputs, layer, count,
                following=functools.partial(first_of, index + 1),
                reader=run.variables.get('streaming', {}).get('bank'), in_place=in_place)
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
    # The parameters, a stream's reader and the expert kernels read in place (`_apart`) are only read.
    return sorted(name for name in {*written, *state, 'intermediates'}
                  if name not in ('params', 'streaming', BANKED_EXPERTS) and run.is_mutable_collection(name))


def _prefetched_run(banks, primed, state, cache, x, inputs, layer, count: int, *, following, reader=None,
                    in_place=None):
    """Run `count` layers under `jax.lax.scan`, read one layer at a time.

    `primed` is layer 0's read-only variables, already on the device. Iteration
    `i` issues layer `i + 1`'s copy, then computes layer `i`; the last layer is
    computed after the loop and `following` stages the next run's first layer in
    place of its copy. Each layer reads its slice of `state`, the caller's
    opened collections, and their values come out stacked per iteration, as
    flax's scan hands them out. The carry is the hidden state, the staged
    variables and the cache, so a decode step holds one banked cache: each layer
    reads and writes its keys and values where the bank holds them, a step's
    tokens at their slots alone (`dew.nn.kv_cache.cache_at_layer`), and every other
    leaf as its slice, written back; a cache the layers create comes out stacked
    like `state`. `in_place` holds the variables each layer reads where they
    lie in device memory (`_apart`), handed to every iteration whole with its
    index rather than staged.
    """
    in_place = in_place or {}
    def body(carry, index):
        hidden, current, held = carry
        staged = _fetched_layer(banks, index + 1) if reader is None else reader.fetch(index + 1, hidden)
        per_layer_slice = None if inputs is None else inputs.layer(index)
        hidden, changed = layer(_with(current, in_place, index), _layer_slice(state, index),
                                None if held is None else cache_at_layer(held, index),
                                hidden, per_layer_slice)
        if held is not None:
            held = cache_from_layer(held, changed.pop('cache'), index)
        return (hidden, staged, held), changed

    (x, current, cache), sown = jax.lax.scan(body, (x, primed, cache), jnp.arange(count - 1))
    last = count - 1
    staged = following(x)
    x, changed = layer(_with(current, in_place, last), _layer_slice(state, last),
                       None if cache is None else cache_at_layer(cache, last), x,
                       None if inputs is None else inputs.layer(last))
    if cache is not None:
        cache = cache_from_layer(cache, changed.pop('cache'), last)
    changed = jax.tree.map(
        lambda rows, final: jnp.concatenate([rows, final[None]]), sown, changed)
    if cache is not None:
        changed['cache'] = cache
    return x, changed, staged


class StageSplit(NamedTuple):
    """How a stack of layers runs over a stage axis: the `prefix` layers before the run its stages
    repeat and the `suffix` layers after it run on every stage, as the plain loop runs them, and
    each stage runs `per_stage` layers of the run between. A stack on one stage has neither."""
    prefix: int
    per_stage: int
    suffix: int


def split_stages(specs: Sequence[LayerSpec], stages: int) -> StageSplit:
    """The split of the layers `specs` describe over `stages` that leaves the fewest layers on every
    stage (`StageSplit`), a shorter suffix first. The run the stages repeat holds `stages` equal
    stretches of `count` layers, layer `j` of each matching layer `j` of the first in every field of
    its spec, and none of its layers shares keys and values with another, which a stage's program
    cannot carry: what a provider leaves is its own (`LayerSpec.provider`), and a sharing layer reads
    a store no stage holds. A dense prefix before routed layers, a sharing tail and a pattern whose
    period divides no stage's length each run beside the pipeline. A stack with no such run of
    `stages` layers or more raises `LayoutRefused`."""
    layers = len(specs)
    pipelined = [spec.provider is None and not spec.kv_shared for spec in specs]
    for outside in range(layers - stages + 1):
        if (layers - outside) % stages:
            continue
        count = (layers - outside) // stages
        for suffix in range(outside + 1):
            prefix = outside - suffix
            # The stages repeat one program where the run's period divides a stage's length.
            if all(pipelined[prefix:layers - suffix]) and not count % run_period(specs, prefix,
                                                                                  layers - outside):
                return StageSplit(prefix, count, suffix)
    raise LayoutRefused(
        f"no run of {stages} or more of the {layers} layers splits into {stages} stages that repeat "
        f"one program; a pipeline of {stages} stages needs a run of layers whose pattern repeats "
        f"every run / {stages} layers and holds no layer that shares keys and values")


class PipelineStage(nn.Module):
    """Runs the layers of one pipeline stage over one microbatch, or of one period of a periodic run.

    The pipeline vmaps this module over the stage axis, so each stage runs it
    on its own slice of the stacked layer weights. Its variables therefore have
    a leading stage axis, which the `StackView` outside stacks and unstacks.
    Layer `j` of a stage is `layers_j` here and `layers_{offset + stage * count + j}`
    in the stored tree, where `offset` is the first layer the stages run and
    `count` is the number of layers per stage. `specs` and `groups` describe
    stage 0, and every stage repeats them. A periodic run's scan repeats it as
    the body of the period's layers (`_run_module`), its repeats in place of
    the stages, the stride the period's.
    """
    block: Block
    specs: tuple[LayerSpec, ...]
    groups: tuple[tuple[int, int], ...]
    train: bool = False
    offset: int = 0

    @nn.nowrap
    def _block(self, index: int, name: str):
        """The stack's block for this stage's layer `index`, the stored layer `offset + index`."""
        return self.block(self.offset + index, name)

    def setup(self):
        self.layers = [self._block(index, f'layers_{index}') for index in range(len(self.specs))]

    @nn.compact
    def __call__(self, x, per_layer_input=None, attention_metadata=None, *, train: bool | None = None,
                 decode: bool = False, kv_store=None):
        # A period's scan calls it as it calls a layer, which leaves nothing a later layer reads.
        assert kv_store is None, "a stage's or a period's layers leave nothing a later layer reads"
        # Singleton layers and scanned runs alike are built from this stage's own indices.
        return run_stack(self.layers, self._block, self.specs, self.groups, x,
                         train=self.train if train is None else train, decode=decode, kv_store=None,
                         per_layer_input=per_layer_input, attention_metadata=attention_metadata)


class BlockStage(nn.Module):
    """One pipeline stage of a stack of like blocks (`pipelined_blocks`), over one microbatch.

    `layer(j)` builds the stage's block `j` under the names its `StackView` stacks
    the stored blocks into, and `run(blocks, x, rows)` runs the stage's `count`
    blocks over the microbatch `x`, with its rows of the inputs held by row.
    A stack of blocks reads no per-layer inputs: the pipeline hands it None.
    """
    layer: StageLayer
    count: int
    run: StageRun

    @nn.compact
    def __call__(self, x, layered, rows):
        del layered
        return self.run([self.layer(index) for index in range(self.count)], x, rows)


def stack_layers(*trees):
    """Layers' trees stacked on a new leading layer axis, a bank (`with_stacked_axes`)."""
    return with_stacked_axes(jax.tree.map(lambda *leaves: jnp.stack(leaves), *trees), 1)


def with_stacked_axes(tree, count: int):
    """`tree`, its leaves stacked on `count` new leading axes, with a parameter its module boxed with its axes
    (`nn.with_logical_partitioning`) keeping its box, which names the new axes none (`meta.add_axis`), as a
    scan stacking it does (`metadata_params`)."""
    for _ in range(count):
        tree = meta.add_axis(tree, 0, {nn.PARTITION_NAME: None})
    return tree


def with_unstacked_axes(tree, count: int):
    """`tree`, its leaves a row of a stacked run's with `count` leading axes taken off, its boxes keeping the
    names of the axes the row keeps (`meta.remove_axis`), `with_stacked_axes`' inverse."""
    for _ in range(count):
        tree = meta.remove_axis(tree, 0, {nn.PARTITION_NAME: None})
    return tree


def stacked_rows(rows: Sequence[jax.Array], shape: tuple[int, ...]) -> jax.Array:
    """A run's rows (`StackView.stage_form`'s order) as one array whose leading `shape` they fill."""
    return jnp.stack(rows).reshape((*shape, *jnp.shape(rows[0])))


def _rows_on_stages(rows: Sequence[jax.Array], shape: tuple[int, ...]) -> jax.Array:
    return _on_stage_axis(stacked_rows(rows, shape))


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
    run, and both methods leave them alone, as they leave the `staged` ones,
    whose runs a stage-local state already holds stacked by stage
    (`stage_form`). `bank_names` returns the name of the
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

    `prefixes` name the stored layers, `{prefix}N` for each: a decoder's
    `layers_N`, a stack of blocks' own (`pipelined_blocks`), several where a
    stage repeats more than one module a layer (a factorized video stack's
    `spatial_N` and `temporal_N`). Only a decoder's runs scan, so a run of
    more than one layer is `group_name`'s.
    """
    groups: tuple[tuple[int, int], ...]
    stages: int = 1
    microbatches: int = 1
    broadcast: tuple[str, ...] = ()
    banked: tuple[str, ...] = ()
    batch_axis: int = 0
    prefixes: tuple[str, ...] = ('layers_',)
    offset: int = 0
    """The first stored layer a pipeline's stages run (`StageSplit.prefix`); the layers before it
    and after its stages stay as they are stored."""
    staged: tuple[str, ...] = ()
    periods: tuple[int, ...] = ()
    """Each run's period (`layer_plan.run_period`), one per group; empty for runs of like layers. A
    periodic run's leaves are stacked per position of its period, `layers_3_10/layers_0` holding
    layers 3, 5, 7 and 9 of a period of two, as its scanned body reads them (`_run_module`)."""

    def _period(self, first: int, count: int) -> int:
        """The period of the run of `count` layers from `first`, one for a run of like layers."""
        return dict(zip(self.groups, self.periods, strict=True)).get((first, count), 1) if self.periods else 1

    @staticmethod
    def _strided(base: int, stride: int, majors: int, minors: int, prefix: str) -> list[list[str]]:
        """Stored names `{prefix}{base + major * stride + minor}`, a list of minors a major: a pipeline's
        layers a stage, and transposed, a period's repeats a position."""
        return [[f'{prefix}{base + major * stride + minor}' for minor in range(minors)]
                for major in range(majors)]

    def _positions(self, first: int, count: int, period: int, prefix: str = 'layers_') -> list[list[str]]:
        """A periodic run's stored names per position of its period, each in repeat order."""
        return [list(names) for names in zip(*self._strided(first, period, count // period, period, prefix),
                                              strict=True)]

    @property
    def per_stage(self) -> int:
        return sum(count for _, count in self.groups)

    def stacked_shape(self, count: int) -> tuple[int, ...]:
        """The leading dimensions a run of `count` layers stacks into: the stages, then the run's
        own layers, each where there is more than one."""
        return tuple(size for size in (self.stages, count) if size > 1)

    def _inside_name(self, first: int, count: int, prefix: str = 'layers_') -> str | None:
        """A run's name inside, None for a layer the view leaves as it is."""
        if count > 1:
            return group_name(first, count)
        return f'{prefix}{first}' if self.stages > 1 else None

    def _outside_names(self, first: int, count: int, prefix: str = 'layers_') -> list[list[str]]:
        """The stored names of a run's layers, one list per stage."""
        return self._strided(self.offset + first, self.per_stage, self.stages, count, prefix)

    @staticmethod
    def _stacked(collection: str, tree: dict, names: list[list[str]]) -> list | None:
        """The entries `names` holds in `tree`, popped and stacked a list of them each (a single one as
        it is), or None where `tree` holds none of them; it holds every one or none."""
        held = [layer in tree for group in names for layer in group]
        if not any(held):
            return None
        every = [layer for group in names for layer in group]
        if not all(held):
            raise ValueError(f"collection {collection!r} holds some of the layers {sorted(every)} and not "
                             "the others; a run reads every one of its layers or none")
        _refuse_unlike(collection, [tree[layer] for layer in every], every)
        return [stack_layers(*[tree.pop(layer) for layer in group]) if len(group) > 1 else tree.pop(group[0])
                for group in names]

    def _runs(self) -> list[tuple[int, int, str]]:
        """Every run, `(first, count, prefix)`, once for each of `prefixes`."""
        return [(first, count, prefix) for first, count in self.groups for prefix in self.prefixes]

    def _named_runs(self) -> list[tuple[str, list[str], int]]:
        """Every run the view stacks whole, `(name, layers, count)`: its name inside, its stored layers in
        `stage_form`'s order, and how many of them each stage runs; a periodic run stacks by position
        instead (`_stacked_periods`)."""
        return [(name, [layer for stage in self._outside_names(first, count, prefix) for layer in stage],
                 count) for first, count, prefix in self._runs()
                if (name := self._inside_name(first, count, prefix)) is not None
                and self._period(first, count) == 1]

    def holds(self, tree: Mapping) -> bool:
        """Whether `tree`, a collection's subtree where the stack is stored, holds some run of the view,
        on its stored layers or stacked by stage (`stage_form`)."""
        staged = tree.get('stages', {})
        return any(name in staged or any(layer in tree for layer in layers)
                   for name, layers, _ in self._named_runs())

    def splits(self, tree: Mapping) -> bool:
        """Whether `tree` holds some of a run's stored layers and not the others, as a collection
        that keeps part of a stack (layers frozen by index) does."""
        return any(0 < sum(layer in tree for layer in layers) < len(layers)
                   for _, layers, _ in self._named_runs())

    def run_layers(self, name: str) -> tuple[list[str], int] | None:
        """The stored layers of the run named `name` inside, in `stage_form`'s order, and how many
        layers of it each stage runs; None where no run has that name."""
        return next(((layers, count) for run, layers, count in self._named_runs() if run == name), None)

    @staticmethod
    def layer_structures(variables: Mapping[str, Mapping], layers: int) -> tuple[tuple, ...]:
        """What one scanned body has to share beyond its spec, for each of `layers`
        stored layers: the runs a collection holds as one bank (`layers_3_7`) that
        hold it, and the structure of its subtree in every collection that keeps
        one per layer (`layers_N`) with each leaf's shape and dtype, as layers a
        checkpoint quantizes differently, or adapts in part or at their own ranks
        (a LoRA rank pattern), differ. A banked run stays the run its store holds
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
        def held(tree) -> tuple:
            leaves, structure = jax.tree.flatten(tree)
            return structure, tuple((jnp.shape(leaf), jnp.result_type(leaf)) for leaf in leaves)

        own = [tuple((collection, *held(tree[f'layers_{index}']))
                     for collection, tree in sorted(variables.items()) if f'layers_{index}' in tree)
               for index in range(layers)]
        return tuple((held, own[first]) for held, first in zip(banks, read, strict=True))

    def bank_names(self) -> list[str]:
        """Return every run's stored name, in order. A banked store uses these names as keys."""
        return [self._inside_name(first, count) or f'layers_{first}'
                for first, count in self.groups]

    def stack(self, variables: Mapping[str, Mapping]) -> dict:
        inside = {}
        # A box keeps its parameter's axes, the stacked layer axis named none (`stack_layers`): a draw's
        # boxes are the draw's, which a run's second trace (`nn.map_variables` at init) does not redraw.
        for collection, tree in variables.items():
            tree = dict(tree)
            if collection in self.banked:
                inside[collection] = tree
                continue
            # A periodic run's positions stack on their own (`_stacked_periods`), whatever the stages.
            self._stacked_periods(collection, tree)
            if self.stages == 1:
                for name, run in self._stacked_runs(collection, tree, stacked_rows).items():
                    # A bank already stored under the run's name holds the
                    # leaves its rows lack (`banked_collections`); the rows
                    # stack into it.
                    tree[name] = merge(tree[name], run) if name in tree else run
                inside[collection] = tree
                continue
            if collection not in self.staged:
                tree = self.stage_form(collection, tree, _rows_on_stages)
            # The layers a pipeline runs outside its stages (`StageSplit`) run beside it, so their
            # entries, a prefix's sown statistics among them, stay out of the stages' scan; the
            # entries the stages write merge back beside them.
            stored = re.compile(f"({'|'.join(map(re.escape, self.prefixes))})\\d+")
            inside[collection] = {name: value for name, value in tree.items() if not stored.fullmatch(name)}
        return inside

    def unstack(self, variables: Mapping[str, Mapping]) -> dict:
        outside = {}
        for collection, tree in variables.items():
            if collection in self.banked or collection in self.staged:
                outside[collection] = dict(tree)
                continue
            broadcast = collection in self.broadcast

            def rows(path, leaf, shape, broadcast=broadcast):
                count = math.prod(shape) // self.stages
                return [self._leaf(stage, offset if count > 1 else None, broadcast, path, leaf)
                        for stage in range(self.stages) for offset in range(count)]
            outside[collection] = self._unstacked_periods(self.layer_form(tree, rows))
        return outside

    def stage_form[Leaf](self, collection: str, tree: Mapping, stack: RowStack[Leaf]) -> dict:
        """`tree`, `collection`'s subtree where the stack is stored, with each run the stages repeat
        held as a pipeline reads it: its stored layers' leaves stacked by `stack(rows, shape)` into
        one under the run's inside name below `stages`, the rows stage-major, a run's own layers
        inside each stage, and `shape` the leading dimensions they fill (`stacked_shape`). That is a
        stage-local state's form (`staged`), which `layer_form` takes back; `stack` takes arrays,
        their shapes or their shardings alike."""
        tree = dict(tree)
        runs = self._stacked_runs(collection, tree, stack)
        if runs:
            tree['stages'] = {**tree.get('stages', {}), **runs}
        return tree

    def layer_form[Leaf](self, tree: Mapping, cut: RowCut[Leaf]) -> dict:
        """`tree` with each run back on its stored layers, `cut(path, leaf, shape)` giving a stacked
        leaf's rows in `stage_form`'s order: under `stages` in a pipeline's form, under the run's
        bank name on one stage. What a run leaves beside its layers stays."""
        tree = dict(tree)
        held = dict(tree.pop('stages', {})) if self.stages > 1 else tree
        for name, layers, count in self._named_runs():
            if name not in held:
                continue
            shape = self.stacked_shape(count)

            def rows_of(path, leaf, shape=shape, layers=layers):
                rows = tuple(cut(path, leaf, shape))
                if len(rows) != len(layers):
                    raise ValueError(f"{jax.tree_util.keystr(path)} of shape {np.shape(leaf)} under a run "
                                     f"stacked {shape} cuts into {len(rows)} rows, not its {len(layers)} "
                                     "layers'")
                return rows
            # Each layer's tree rebuilt from its rows of every leaf, the run's own structure kept: a
            # tuple the run's tree holds (an optimizer's state) is a node, not a leaf's rows.
            leaves, structure = jax.tree_util.tree_flatten_with_path(held.pop(name))
            rows = [rows_of(path, leaf) for path, leaf in leaves]
            for index, layer in enumerate(layers):
                tree[layer] = with_unstacked_axes(structure.unflatten([cuts[index] for cuts in rows]),
                                                  len(shape))
        if self.stages > 1 and held:
            tree['stages'] = held
        return tree

    def _stacked_runs[Leaf](self, collection: str, tree: dict, stack: RowStack[Leaf]) -> dict:
        """Pop each run `tree` holds out of it, its rows stacked by `stack` (`stage_form`), by the
        run's inside name. A collection holding some of a run's layers and not the others is
        refused, and so are layers whose structures differ (`_refuse_unlike`)."""
        runs = {}
        for name, layers, count in self._named_runs():
            held = [layer in tree for layer in layers]
            if not any(held):
                continue
            if not all(held):
                raise ValueError(
                    f"collection {collection!r} holds some of the layers {sorted(layers)} and not "
                    "the others; a run reads every one of its layers or none")
            _refuse_unlike(collection, [tree[layer] for layer in layers], layers)
            shape = self.stacked_shape(count)
            runs[name] = with_stacked_axes(jax.tree.map(lambda *rows, shape=shape: stack(list(rows), shape),
                                                        *[tree.pop(layer) for layer in layers]), len(shape))
        return runs

    def _stacked_periods(self, collection: str, tree: dict) -> None:
        """Each periodic run `tree` holds stacked in it by position of its period: a position's repeats
        under the name the period's stage gives its layer, below the run's inside name, as its scanned
        body reads them; a position whose layers keep nothing in the collection (a layer with no cache)
        is left out of it."""
        for first, count, prefix in self._runs():
            name = self._inside_name(first, count, prefix)
            if name is None or (period := self._period(first, count)) == 1:
                continue
            stacked = {f'{prefix}{offset}': position[0]
                       for offset, layers in enumerate(self._positions(first, count, period, prefix))
                       if (position := self._stacked(collection, tree, [layers])) is not None}
            if stacked:
                tree[name] = stacked

    def _unstacked_periods(self, tree: dict) -> dict:
        """`tree` with each periodic run back on its stored layers (`_stacked_periods`'s inverse)."""
        for first, count, prefix in self._runs():
            name = self._inside_name(first, count, prefix)
            if name is None or (period := self._period(first, count)) == 1 or name not in tree:
                continue
            view = tree.pop(name)
            for offset, layers in enumerate(self._positions(first, count, period, prefix)):
                if f'{prefix}{offset}' not in view:
                    continue
                for repeat, layer in enumerate(layers):
                    # One repeat's leaf, taken with `index_in_dim` as `_leaf` takes a layer's.
                    row = functools.partial(jax.lax.index_in_dim, index=repeat, keepdims=False)
                    tree[layer] = with_unstacked_axes(jax.tree.map(row, view[f'{prefix}{offset}']), 1)
        return tree

    def stored_layer(self, tree: Mapping, layer: int, part: str, prefix: str = 'layers_'):
        """Stored layer `layer`'s subtree `part` (its `mlp`) in `tree`, a collection's where the stack is
        stored on one stage: the layer's own entry, or its row of the bank of the run that holds it, a
        periodic run's at its position's repeat (`_stacked_periods`), taken with `index_in_dim` as `_leaf`
        takes a layer's."""
        keys, row = self._stored_at(tree, layer, prefix)
        held = functools.reduce(lambda node, key: node[key], keys, tree)[part]
        return held if row is None else jax.tree.map(
            functools.partial(jax.lax.index_in_dim, index=row, keepdims=False), held)

    def with_stored_layer(self, tree: Mapping, layer: int, part: str, value, prefix: str = 'layers_') -> dict:
        """`tree` with stored layer `layer`'s subtree `part` replaced by `value`, written where
        `stored_layer` reads it: a bank's row in place, cast to the bank's dtypes, its other rows as
        stored."""
        keys, row = self._stored_at(tree, layer, prefix)

        def written(node: Mapping, depth: int) -> dict:
            if depth < len(keys):
                return {**node, keys[depth]: written(node[keys[depth]], depth + 1)}
            put = value if row is None else jax.tree.map(
                lambda stacked, leaf: jax.lax.dynamic_update_index_in_dim(
                    stacked, leaf.astype(stacked.dtype), row, axis=0), node[part], value)
            return {**node, part: put}
        return written(tree, 0)

    def _stored_at(self, tree: Mapping, layer: int, prefix: str) -> tuple[tuple[str, ...], int | None]:
        """Where `tree` stores layer `layer` on one stage: the keys down to its entry and its row there,
        None for an entry of its own."""
        if f'{prefix}{layer}' in tree:
            return (f'{prefix}{layer}',), None
        for first, count in self.groups:
            name = self._inside_name(first, count, prefix)
            if first <= layer < first + count and name is not None and name in tree:
                period = self._period(first, count)
                if period == 1:
                    return (name,), layer - first
                return (name, f'{prefix}{(layer - first) % period}'), (layer - first) // period
        raise ValueError(f"no stored layer holds layer {layer}")

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
    """Where one stack of layers is stored, below every variables collection: a decoder's, or a stack
    of like blocks' (`block_sites`).

    `namespace` is the module path to the stack. A container module adds its
    own name in front of the namespace and leaves `view` unchanged, so the layer
    groups, module names and RNG streams stay the stack's. A shared scope is
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


def microbatch_count(rows: int) -> int:
    """The microbatches a pipeline over the stage axis in context cuts `rows` rows
    into (`microbatches`), refused (`LayoutRefused`) where the count does not
    divide the rows or is no multiple of the stages, or where it does not divide
    the rows a device holds.

    Microbatch m takes rows m, m + count, ... (`microbatched`), a share of every
    device's rows only where the count divides a block; otherwise devices
    holding none of a microbatch recompute another's (1.43 times one device's
    FLOPs for 8 rows in 4 microbatches over 4 row shards, against the bubble's
    1.25).
    """
    stages, count = pipeline_stages(), microbatches()
    if count % stages or rows % count:
        raise LayoutRefused(
            f"a batch of {rows} rows over {stages} stages needs a microbatch "
            f"count that divides the rows and is a multiple of the stages, "
            f"got {count}")
    splitting = row_axes(rows)
    shards = math.prod(jax.sharding.get_abstract_mesh().shape[axis] for axis in splitting)
    if (rows // shards) % count:
        raise LayoutRefused(
            f"a batch of {rows} rows splits {shards} ways over {' x '.join(splitting)}, "
            f"{rows // shards} rows a device, which {count} microbatches "
            f"do not divide, so the devices that hold none of a microbatch's rows would "
            f"compute another's again; use a microbatch count that divides "
            f"{rows // shards}, or a batch that is a multiple of "
            f"{count * shards} rows")
    return count


def run_pipeline(model: nn.Module, view: StackView, x, stage: Callable[[nn.Module], nn.Module], *,
                 layered=None, rows=None):
    """GPipe over the stage axis, as MaxText's `layers/pipeline.py` runs it.

    Iteration t runs stage s on microbatch t - s, the layers stacked over the
    stages under `jax.vmap` so GSPMD keeps each stage on its devices, and
    outputs shift a stage down by `ppermute`. The microbatches sit in
    `state_io`, `[stages, microbatches / stages, ...]`, stage-sharded: stage 0
    reads slot t % (microbatches / stages) and the slots rotate up a stage per
    iteration, so every microbatch reaches stage 0 and finishes in the last
    stage's slot without a gather. The first and last stages - 1 iterations are
    the bubble. Only the run of layers the stages repeat pipelines (`StageSplit`,
    from `view.offset`); embeddings, the layers outside that run, norm, head and
    loss run on the whole batch, so the loss is the plain loop's mean.

    `model` is the module whose variables `view` stacks, and `stage(model)`
    builds the module one stage runs, named `stages`: a decoder's
    `PipelineStage`, a stack of blocks' `BlockStage`. It is called on a
    microbatch, its slice of `layered`, the inputs held by row and by layer
    (a decoder's per-layer inputs, the layer at axis 2), and its rows of
    `rows`, the inputs held by row alone (an attention's metadata, a
    diffusion model's conditioning).
    """
    stages, count, batch_axis = view.stages, view.microbatches, view.batch_axis
    per_stage = view.per_stage
    micro = functools.partial(microbatched, count=count)
    x = micro(x, batch_axis)
    inputs = None if layered is None else jax.tree.map(lambda value: micro(value, 0), layered)
    # A watched summary's real positions are held by row, each stage reading its microbatch's
    # (`watched_positions`).
    held = jax.tree.map(lambda value: micro(value, 0), (rows, current_positions()))
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

    def call_stage(stage, x, layered, held, live):
        rows, positions = held
        # A bubble iteration's stage runs a microbatch no output keeps (`counted`).
        with pipeline_stage(model.path, view), watched_positions(positions), counted(live):
            return stage(x, layered, rows)

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
                view.offset + stage * per_stage, per_stage, axis=2), inputs))(ids, stage_ids)))
        # A box its module drew a parameter in carries the stage axis unnamed (`stack_layers`).
        run = nn.vmap(call_stage, variable_axes={True: 0}, split_rngs={True: True},
                      in_axes=0, out_axes=0, spmd_axis_name=STAGE_AXIS,
                      metadata_params={nn.PARTITION_NAME: None})
        live = (step - stage_ids >= 0) & (step - stage_ids < count)
        out = _on_stage_axis(run(stage(module), stages_in, stage_inputs,
                                 jax.tree.map(lambda value: gather(value, ids), held), live))
        state_io = jax.lax.dynamic_update_index_in_dim(
            state_io, rotate_up(stream, out), slot, 1)
        return (state_io, shift_down(out)), None

    loop = nn.scan(iteration, variable_broadcast=view.broadcast,
                   variable_axes={True: 0}, split_rngs={True: True},
                   metadata_params={nn.PARTITION_NAME: None})
    (state_io, _), _ = loop(model, (state_io, shift), jnp.arange(count + stages - 1))
    # Microbatch 0 finished stages - 1 iterations in, so it sits that
    # many slots along; the rest follow it in order.
    order = (np.arange(slots) + (stages - 1) % slots) % slots
    finished = state_io[:, order].reshape((count, *x.shape[1:]))
    return _whole(finished, batch_axis)


def pipelines(module: nn.Module, kinds: Sequence) -> bool:
    """Whether `module`'s stack of like blocks, block `N` of kind `kinds[N]`, runs as a pipeline
    over the stage axis of the mesh in context (`pipelined_blocks`): one the stages split into equal
    stretches that repeat one pattern of kinds, as each stage runs one program, past initialization,
    which draws the stored blocks in the plain loop. A stack they do not split so runs whole on
    every stage, as the layers outside a decoder's pipelined run do, and a model none of whose
    stacks pipelines is refused the axis (`Trainer`)."""
    return not module.is_initializing() and _splits(kinds, pipeline_stages())


def _splits(kinds: Sequence, stages: int) -> bool:
    """Whether `stages` stages split a stack of blocks of `kinds` into equal stretches that repeat one
    pattern of kinds (`pipelines`)."""
    count = len(kinds)
    return (stages > 1 and count % stages == 0
            and all(kind == kinds[index % (count // stages)] for index, kind in enumerate(kinds)))


def _block_view(prefixes: tuple[str, ...], count: int, stages: int) -> StackView:
    """How a pipeline over `stages` stages stacks a stack of `count` like blocks stored under
    `prefixes`: one block a run, stage s's from block s * count / stages on."""
    return StackView(tuple((index, 1) for index in range(count // stages)), stages=stages, prefixes=prefixes)


def blocks_off_stages[M: nn.Module, **P, R](module: M, method: Callable[Concatenate[M, P], R],
                                            prefixes: tuple[str, ...], count: int
                                            ) -> Callable[Concatenate[M, P], R] | None:
    """`off_stages` for a stack of `count` like blocks stored under `prefixes` (`pipelined_blocks`)."""
    return off_stages(module, method, functools.partial(_block_view, prefixes, count))


def block_sites(namespace: tuple[str, ...], prefixes: tuple[str, ...], kinds: Sequence,
                stages: int) -> tuple[DecoderBank, ...]:
    """Where a stack of like blocks of `kinds` stored under `prefixes` at `namespace` keeps the run a
    pipeline over `stages` stages stacks (`pipelined_blocks`): the whole stack where they split it
    (`pipelines`), nothing where it runs whole on every stage."""
    if not _splits(kinds, stages):
        return ()
    return (DecoderBank(namespace, _block_view(prefixes, len(kinds), stages)),)


def nested_sites(name: str, sites: Sequence[DecoderBank]) -> tuple[DecoderBank, ...]:
    """`sites` a module declares over its own tree, seen from the module holding it as `name`."""
    return tuple(dataclasses.replace(site, namespace=(name, *site.namespace)) for site in sites)


def part_stage_sites(name: str, module, store: Variables, stages: int) -> tuple[DecoderBank, ...]:
    """The runs a pipeline over `stages` stacks that `module`, held as `name` in every collection of
    `store`, declares (`StagedStacks.stage_sites`), seen from its holder: a headed model's backbone, a
    value head's, an actor's policy. A module that declares none has none."""
    return (nested_sites(name, module.stage_sites(part(store, name), stages))
            if isinstance(module, StagedStacks) else ())


def stored_by_stage(variables: Mapping[str, Mapping]) -> tuple[str, ...]:
    """The collections of a stack's `variables` that hold its run stacked by stage (`StackView.staged`)."""
    return tuple(name for name, tree in variables.items() if 'stages' in tree)


def off_stages[M: nn.Module, **P, R](module: M, method: Callable[Concatenate[M, P], R],
                                     view_of: Callable[[int], StackView]
                                     ) -> Callable[Concatenate[M, P], R] | None:
    """`method`, one of `module`'s, reading a stage-local store (`stored_by_stage`), its runs stacked by
    stage as a pipeline over its stage axis trains them, as the stored layers they stack: where no mesh
    in context runs them over as many stages (a model off its training mesh, a generation from the
    state `fit` returns), each run's rows are cut back onto its layers on the way in (`StackView.
    layer_form`, by the `view_of(held)` it was stacked under) and stacked again on the way out, so the
    model reads the one state its run holds, with no copy of it kept. None where `module` holds no such
    store, or the mesh in context pipelines it as stored."""
    staged = stored_by_stage(module.variables)
    held = stage_count(module.variables)
    if not staged or held == pipeline_stages():
        return None
    view = view_of(held)

    def layered(variables):
        return {name: view.layer_form(tree, array_rows) if name in staged else tree
                for name, tree in variables.items()}

    def stacked(variables):
        return {name: view.stage_form(name, tree, stacked_rows) if name in staged else tree
                for name, tree in variables.items()}

    return nn.map_variables(method, mapped_collections=True, trans_in_fn=layered, trans_out_fn=stacked,
                            init=False, mutable=True)


def pipelined_blocks(module: nn.Module, prefixes: tuple[str, ...], count: int, x: jax.Array,
                     layer: StageLayer, run: StageRun, *, rows: jax.Array | None = None) -> jax.Array:
    """`module`'s stack of `count` like blocks over the rows of `x`, as a pipeline over the
    stage axis of the mesh in context (`run_pipeline`), where `pipelines` says it runs so.

    Block `N` is stored as `{prefix}N` under `module` for those of `prefixes` its
    kind takes (a DiT's `dit_block_N`, a hybrid's SSM blocks' `ssm_block_N`),
    and stage s runs blocks s * count / stages onwards:
    `layer(j)` builds its block `j` under the same names, and `run(blocks, x,
    rows)` runs them over a microbatch, with its rows of `rows`, the inputs
    held by row. The stages hand the stream on in the dtype it settles in, which
    a stage's first block gives it (`_settled`), as the plain loop's does. A
    batch `microbatch_count` refuses raises `LayoutRefused`.
    """
    stages = pipeline_stages()
    view = dataclasses.replace(
        _block_view(prefixes, count, stages), microbatches=microbatch_count(x.shape[0]),
        # What enters the loop is read on every iteration; what
        # the loop creates (what a block sows) comes out per iteration.
        broadcast=tuple(name for name, tree in module.variables.items() if tree),
        staged=stored_by_stage(module.variables))
    x = x.astype(_settled(module, prefixes, x, layer, run, rows))
    piped = nn.map_variables(_pipelined, mapped_collections=True, trans_in_fn=view.stack,
                             trans_out_fn=view.unstack, init=False, mutable=True)
    return piped(module, view, x, functools.partial(BlockStage, layer, count // stages, run, name='stages'),
                 rows)


def _settled(module: nn.Module, prefixes: tuple[str, ...], x: jax.Array, layer: StageLayer, run: StageRun,
             rows: jax.Array | None) -> jnp.dtype:
    """The dtype `x` settles in through `module`'s first stored block: its own, promoted with the
    block's output's, evaluated abstractly (nothing sown or drawn, a placeholder `params` key flax
    draws any other stream from), as `CausalTransformer.residual_dtype` reads a decoder's. A
    stage-local store's first block is its stacked run's first row."""
    names = [f'{prefix}0' for prefix in prefixes]
    first = {}
    for collection, tree in module.variables.items():
        staged = tree.get('stages', {})
        held = {name: tree[name] if name in tree else jax.tree.map(lambda leaf: leaf[0], staged[name])
                for name in names if name in tree or name in staged}
        if held:
            first[collection] = held
    output = jax.eval_shape(lambda held: BlockStage(layer, 1, run, parent=None).apply(
        held, x, None, rows, mutable=True, rngs={'params': jax.random.key(0)})[0], first)
    return jnp.result_type(x.dtype, output.dtype)


@nn.compact
def _pipelined(module, view, x, stage, rows):
    """`run_pipeline` over `module`'s variables as `view` stacks them: compact, so the
    stages it builds are `module`'s own (`pipelined_blocks`)."""
    return run_pipeline(module, view, x, lambda _: stage(), rows=rows)


def banked_collections(groups: Sequence[tuple[int, int]],
                       variables: Mapping[str, Mapping]) -> tuple[str, ...]:
    """The collections of `variables` whose layer subtrees the store holds as banks, over the stack's
    scanned `groups` (`(first, count)` runs; `CausalTransformer.banked_collections`)."""
    runs = {group_name(first, count): [f'layers_{index}' for index in range(first, first + count)]
            for first, count in groups if count > 1}
    banked = []
    for collection, tree in variables.items():
        banks = set(tree) & set(runs)
        if not banks:
            continue
        mixed = False
        for bank, layers in runs.items():
            rows = [layer for layer in layers if layer in tree]
            if not rows:
                continue
            mixed = True
            if bank not in tree:
                continue
            shared = {tuple(entry.key for entry in path)
                      for path, _ in jax.tree_util.tree_leaves_with_path(tree[bank])}
            for layer in rows:
                overlap = shared & {tuple(entry.key for entry in path)
                                    for path, _ in jax.tree_util.tree_leaves_with_path(tree[layer])}
                if overlap:
                    raise ValueError(
                        f"collection {collection!r} holds {'/'.join(overlap.pop())} both in "
                        f"the bank {bank} and in its layer {layer}; a leaf is read from one")
        if mixed:
            continue
        if banks != set(runs):
            raise ValueError(
                f"collection {collection!r} holds the banks {sorted(banks)} "
                f"of the runs {sorted(runs)} and not the others; a store holds every "
                f"run's bank or every layer's own subtree")
        banked.append(collection)
    return tuple(banked)
