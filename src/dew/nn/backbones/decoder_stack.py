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
from jax.sharding import NamedSharding, PartitionSpec as P

from ..mla import INDEXER_COLLECTION
from ..sharding import STAGE_AXIS
from .decoder_block import DecoderBlock
from .layer_plan import LayerSpec, group_name


def _merged(bank: Mapping, rows: Mapping) -> dict:
    """`bank` with every leaf `rows` holds added beside its own."""
    merged = dict(bank)
    for key, value in rows.items():
        held = merged.get(key)
        merged[key] = (_merged(held, value) if isinstance(held, Mapping) and isinstance(value, Mapping)
                       else value)
    return merged


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
    """
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


WRITTEN = ('cache', 'router', 'qk', INDEXER_COLLECTION)
"""The collections a decoder block writes: its decode cache, and what its
router, its attention and its sparse indexer sow. A run whose parameters are
fetched is applied in a scope of its own, so these are the names whose values
the loop has to carry back out to the scope that asked for them."""


def _scanned_runs(runs, groups: Sequence[tuple[int, int]], specs: Sequence[LayerSpec], x, *, fetching: bool,
                  train: bool, decode: bool, positions, segment_ids, kv_store, per_layer_input,
                  attention_metadata):
    """`run_stack` without the inference prefetch: each run of one layer
    called as the plain loop calls it, and each longer run under flax's
    scan. Training through host-resident banks scans with the fetch mapped
    onto each row under remat (`run_stack`)."""
    for run, (first, count) in zip(runs, groups, strict=True):
        inputs = None if per_layer_input is None else per_layer_input.span(first, count)
        store = kv_store if count == 1 or specs[first].kv_shared else None

        def step(layer, carry, per_layer_input, *, store=store):
            return layer(carry, train=train, decode=decode, positions=positions,
                         segment_ids=segment_ids, kv_store=store,
                         per_layer_input=per_layer_input,
                         attention_metadata=attention_metadata), None

        if fetching:
            # Scan variables are xs, not closed-over bank slices. Native
            # transposition stacks host cotangent rows instead of adding
            # a whole-bank device accumulator. Remat retains the original
            # host operand and refetches only this layer in backward
            # (MaxText layers/decoders.py:544-565).
            step = nn.remat(nn.map_variables(
                step, mapped_collections=True, trans_in_fn=_fetched, init=False, mutable=True))
        if count == 1:
            x, _ = step(run, x, None if inputs is None else inputs.layer(0))
        else:
            x, _ = nn.scan(step, variable_axes={True: 0}, split_rngs={True: True},
                           in_axes=2, length=count)(run, x, inputs)
    return x


def run_stack(layers: Sequence[DecoderBlock], block: Block, specs: Sequence[LayerSpec],
              groups: Sequence[tuple[int, int]], x, *, train: bool, decode: bool,
              positions, segment_ids, kv_store, per_layer_input, attention_metadata=None,
              banked: bool = False):
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
    refetches the pinned bank rather than keeping a device copy.
    """
    runs = [layers[first] if count == 1 else block(first, group_name(first, count))
            for first, count in groups]
    streaming = any('streaming' in run.variables for run in runs)
    if streaming and train:
        raise ValueError("disk streaming is inference-only")
    fetching = streaming or banked or any(_on_host(run.variables.get('params', {})) for run in runs)
    if not fetching or train:
        return _scanned_runs(runs, groups, specs, x, fetching=fetching, train=train, decode=decode,
                             positions=positions, segment_ids=segment_ids, kv_store=kv_store,
                             per_layer_input=per_layer_input, attention_metadata=attention_metadata)

    def read_only(run: DecoderBlock) -> list[str]:
        """List the collections a run reads and does not write.

        These are its parameters and whatever else it was given, all of
        which a bank holds per layer.
        """
        return [name for name in run.variables
                if name != 'streaming' and (not run.is_mutable_collection(name) or name not in WRITTEN)]

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

    staged = first_of(0, x)
    for index, (run, (first, count)) in enumerate(zip(runs, groups, strict=True)):
        inputs = None if per_layer_input is None else per_layer_input.span(first, count)
        store = kv_store if count == 1 or specs[first].kv_shared else None
        mutable = [name for name in WRITTEN if run.is_mutable_collection(name)]

        def layer(read, cache, hidden, per_layer_slice, *, run=run, mutable=mutable, store=store):
            variables = dict(read) if cache is None else {**read, 'cache': cache}
            hidden, changed = run.apply(
                variables, hidden, mutable=mutable, train=train, decode=decode,
                positions=positions, segment_ids=segment_ids, kv_store=store,
                per_layer_input=per_layer_slice, attention_metadata=attention_metadata)
            return hidden, dict(changed)

        cached = (run.variables.get('cache') or None) if 'cache' in mutable else None
        if count == 1:
            following = first_of(index + 1, x)
            x, changed = layer(staged, cached, x, None if inputs is None else inputs.layer(0))
        else:
            banks = {name: run.variables[name] for name in read_only(run)}
            x, changed, following = _prefetched_run(
                banks, staged, cached, x, inputs, layer, count,
                following=functools.partial(first_of, index + 1),
                reader=run.variables.get('streaming', {}).get('bank'))
        staged = following
        for collection, tree in changed.items():
            for name, value in tree.items():
                run.put_variable(collection, name, value)
    return x


def _prefetched_run(banks, primed, cache, x, inputs, layer, count: int, *, following, reader=None):
    """Run `count` layers under `jax.lax.scan`, read one layer at a time.

    `primed` is layer 0's read-only variables, already on the device. Iteration
    `i` issues layer `i + 1`'s copy, then computes layer `i`; the last layer is
    computed after the loop and `following` stages the next run's first layer in
    place of its copy. The carry is the hidden state, the staged variables and
    the cache, written in place one layer's slice per iteration so a decode step
    holds one banked cache; a cache the layers create comes out stacked per
    iteration, as flax's scan hands them out.
    """
    def body(carry, index):
        hidden, current, held = carry
        staged = _fetched_layer(banks, index + 1) if reader is None else reader.fetch(index + 1, hidden)
        per_layer_slice = None if inputs is None else inputs.layer(index)
        hidden, changed = layer(current, None if held is None else _layer_slice(held, index),
                                hidden, per_layer_slice)
        if held is not None:
            held = _layer_written(held, changed.pop('cache'), index)
        return (hidden, staged, held), changed

    (x, current, cache), sown = jax.lax.scan(body, (x, primed, cache), jnp.arange(count - 1))
    last = count - 1
    staged = following(x)
    x, changed = layer(current, None if cache is None else _layer_slice(cache, last), x,
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
    def __call__(self, x, train: bool = False, positions=None, segment_ids=None,
                 per_layer_input=None, attention_metadata=None):
        return run_stack(self.layers, self.block, self.specs, self.groups, x,
                         train=train, decode=False, positions=positions,
                         segment_ids=segment_ids, kv_store=None,
                         per_layer_input=per_layer_input, attention_metadata=attention_metadata)


def _stack_leaves(*trees):
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *trees)


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
    iterations are its microbatches, in order.
    """
    groups: tuple[tuple[int, int], ...]
    stages: int = 1
    microbatches: int = 1
    broadcast: tuple[str, ...] = ()
    banked: tuple[str, ...] = ()

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
                runs = [_stack_leaves(*[tree.pop(layer) for layer in stage]) if count > 1
                        else tree.pop(stage[0]) for stage in names]
                if self.stages == 1:
                    # A bank already stored under the run's name holds the
                    # leaves its rows lack (`banked_collections`); the rows
                    # stack into it.
                    tree[name] = _merged(tree[name], runs[0]) if name in tree else runs[0]
                else:
                    stages[name] = jax.tree.map(_on_stage_axis, _stack_leaves(*runs))
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
                        tree[layer] = jax.tree.map(
                            functools.partial(self._leaf, stage, offset if count > 1 else None,
                                              collection in self.broadcast), view)
            if self.stages > 1 and stages:
                tree['stages'] = stages
            outside[collection] = tree
        return outside

    def _leaf(self, stage: int, offset: int | None, broadcast: bool, leaf):
        """One layer's leaf out of a run's stacked one, taken with `index_in_dim`:
        an index array would be a gather, which over a host-resident bank needs
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
        return real.reshape((-1, *real.shape[2:]))


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


def _microbatched(value, axis: int, count: int):
    """`[.., rows, ..]` as `[count, .., rows / count, ..]`: microbatch m takes
    rows m, m + count, m + 2 * count, ...

    A batch splits over its devices in blocks of consecutive rows. Strided,
    each microbatch keeps a share of every block where the block is, as long
    as `count` divides a block; cut into consecutive runs, each microbatch
    would sit in one block and move to the devices of every other."""
    shape = value.shape
    split = value.reshape((*shape[:axis], shape[axis] // count, count, *shape[axis + 1:]))
    return jnp.moveaxis(split, axis + 1, 0)


def _whole(value, axis: int):
    """The batch `_microbatched` cut, back in one piece and in its order."""
    moved = jnp.moveaxis(value, 0, axis + 1)
    shape = moved.shape
    return moved.reshape((*shape[:axis], shape[axis] * shape[axis + 1], *shape[axis + 2:]))


def run_pipeline(model: nn.Module, view: StackView, x, *, train: bool, positions, segment_ids,
                 per_layer_input, attention_metadata=None):
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
    stages, count = view.stages, view.microbatches
    per_stage = view.per_stage
    batch_axis = 1 if model.altup is not None else 0
    micro = functools.partial(_microbatched, count=count)
    x = micro(x, batch_axis)
    per_row = [None if value is None else micro(jnp.asarray(value), 0)
               for value in (positions, segment_ids)]
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

    def call_stage(stage, x, positions, segment_ids, per_layer_input, attention_metadata):
        return stage(x, train=train, positions=positions, segment_ids=segment_ids,
                     per_layer_input=per_layer_input, attention_metadata=attention_metadata)

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
        out = _on_stage_axis(run(stage, stages_in, gather(per_row[0], ids),
                                 gather(per_row[1], ids), stage_inputs,
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
