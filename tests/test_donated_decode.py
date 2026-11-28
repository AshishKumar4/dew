"""A hybrid model's decode step compiled for a GPU with its cache donated, as serving donates it: each gated
delta net's conv taps read in their checkpoint's order by either decode kernel, no launch repacking its
weight; and wherever a token's step owns the recurrent states it steps (a server's step, a generation's
loop), each launch updates its state in place where the program holds it and the program copies none
(row 14174)."""

import functools
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn
from generation_support import COMPILES_ON_CPU
from jax._src import config as jax_config
from jax.experimental import checkify
from jax.experimental.mosaic.gpu import core as mgpu_core
from jax.sharding import NamedSharding, PartitionSpec
from kernel_compile import _device, assumed, compiling_for, gpu_target

from dew.inference.serving_kernel import (
    Dense,
    StateTypes,
    decode_programs,
    opened,
    resident_formats,
    split_donated,
    state_shardings,
)
from dew.nn.inputs import ModelInputs
from dew.nn.kernels import generation, options
from dew.nn.kernels.templates.recurrence.step import donated_state
from dew.nn.linear import decode_gated_delta_rule
from dew.nn.progress import END
from dew.registry import models
from dew.sampling.decoding import Finishing
from dew.sampling.strategies import Sample
from dew.sampling.text import _checked, decode_ops, prefill_state
from dew.telemetry import hlo
from tools.benchmark_cases import LM_HYBRID

ROWS = 8
"""The rows a hybrid model's step decodes: row 14174's B8."""

_DELTA_NET = LM_HYBRID['kinds']['linear_attention']['mixer']['fields']
"""`LM_HYBRID`'s gated delta nets' fields."""

STATE = (ROWS, _DELTA_NET['linear_num_value_heads'], _DELTA_NET['linear_key_head_dim'],
         _DELTA_NET['linear_value_head_dim'])
"""A gated delta net layer's recurrent state `[rows, value heads, key width, value width]` at `LM_HYBRID`'s
widths, Qwen3.5's (4 MiB in bfloat16)."""

PERIOD = {'num_layers': 4, 'layer_types': LM_HYBRID['layer_types'][:4]}
"""`LM_HYBRID`'s first period alone: three gated delta nets, which a scanned stack runs as one bank, and an
attention layer."""


def _hybrid(**fields):
    """`LM_HYBRID` (three gated delta nets to each attention layer) at a 1024-token vocabulary in bfloat16,
    `fields` its own."""
    return models.build('causal_transformer',
                        {**LM_HYBRID, 'vocab_size': 1024, 'dtype': 'bfloat16', **fields})


def _placed(tree, shardings):
    """`tree`'s leaves as abstract values placed at `shardings`: a tree of them, or one for every leaf."""
    if isinstance(shardings, jax.sharding.Sharding):
        shardings = jax.tree.map(lambda _: shardings, tree)

    def placed(leaf, sharding):
        return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding)

    return jax.tree.map(placed, tree, shardings)


def _aliased(op) -> list[str]:
    """The operands `op` (`hlo.instructions`'s) writes in place: those its outputs alias."""
    return [op.operands[alias.operand_index] for alias in op.proto.output_operand_aliasing]


def _token(rows: int, heads: int = 16, width: int = 128) -> tuple[jax.ShapeDtypeStruct, ...]:
    """A gated delta net's decode token's operands (`decode_gated_delta_rule`'s), abstract: its queries, keys
    and values `[rows, 1, heads, width]` in bfloat16, its gates and write strengths `[rows, 1, heads]` in
    float32, and its state `[rows, heads, width, width]` in bfloat16."""
    return (*(jax.ShapeDtypeStruct((rows, 1, heads, width), jnp.bfloat16) for _ in range(3)),
            *(jax.ShapeDtypeStruct((rows, 1, heads), jnp.float32) for _ in range(2)),
            jax.ShapeDtypeStruct((rows, heads, width, width), jnp.bfloat16))


class _Stepped(NamedTuple):
    """A launch reading a recurrent state (`_steps`)."""

    name: str
    in_place: bool
    """Whether it writes its state in place: the state operand aliased to an output (`_aliased`)."""
    read: tuple
    """The producers it read each state through, back to the state the program holds (`_steps`), each a path
    whose last is an `apart` where something else made the state (a copy, a slice or gather made apart, a
    launch's output of its own)."""

    @property
    def held(self) -> bool:
        """Whether it writes in place every state it reads, each read back to the state the program holds."""
        return self.in_place and not any(path[-1].startswith('apart') for path in self.read)


Holder = Literal['program', 'loop']
"""What holds the states a program steps, which `_steps` reads each launch's state back to: the program's
parameters alone (a server's donated cache), or those and its own loop's carry (a generation's, whose
prefill made it)."""


def _steps(module, state: tuple[int, ...], *, holder: Holder) -> tuple[list[_Stepped], list[str]]:
    """`module`'s launches reading a recurrent state of `state`'s shape (a layer's, or a scanned run's bank of
    them, its trailing axes), each read back to the state its `holder` holds, and the program's copies of
    one. Read back through views (a bitcast, a tuple's element), a loop body's
    carry to the element its loop starts from (copy insertion states any copy that keeps a loop's buffer
    apart as a `copy`), what updated it in place (an aliasing launch, a slice of it written by a
    `dynamic-update-slice`, alone or a fusion's root, whose buffer XLA shares with its operand's or copies
    first, a `copy` again) and, inside the fusion around a launch (XLA's dynamic-slice fusion), a slice of a
    bank where the bank holds it."""
    every = list(hlo.instructions(module))
    found = {(op.computation, op.name): op for op in every}
    fused = {called: op for op in every if op.opcode == 'fusion' for called in op.called}
    loops = {called: op for op in every if op.opcode == 'while' for called in op.called}
    roots = {op.computation: op for op in every if op.root}
    entry = next(op.computation for op in every if op.entry)

    def stated(computation: str, name: str) -> bool:
        return tuple(found[computation, name].proto.shape.dimensions)[-len(state):] == state

    def updated_base(computation: str, name: str) -> int | None:
        """The parameter number of `computation` that `name` updates in place (a slice of it written by a
        `dynamic-update-slice`, through views), or None."""
        op = found[computation, name]
        while op.opcode in ('bitcast', 'dynamic-update-slice'):
            op = found[computation, op.operands[0]]
        return op.proto.parameter_number if op.opcode == 'parameter' else None

    def written(op, index: int | None) -> str | None:
        """The operand `op`'s result (its tuple's element `index`) is that operand updated in place, where
        it is one: an in-place launch's aliased operand, a slice of it updated (`dynamic-update-slice`, alone
        or a fusion's root)."""
        if op.opcode == 'custom-call':
            path = [] if index is None else [index]
            return next((op.operands[alias.operand_index] for alias in op.proto.output_operand_aliasing
                         if list(alias.output_shape_index) == path), None)
        if op.opcode == 'dynamic-update-slice':
            return op.operands[0]
        if op.opcode == 'fusion':
            (called,) = op.called
            root = roots[called]
            if index is not None and root.opcode == 'tuple':
                root = found[called, root.operands[index]]
            if root.opcode == 'dynamic-update-slice':
                number = updated_base(called, root.name)
                return None if number is None else op.operands[number]
        return None

    def apart(through: tuple, op) -> tuple:
        """`through` ended at `op`, which made the state apart from the one the program holds."""
        return (*through, f'apart: {op.opcode} {op.name}')

    def element(computation: str, name: str, index: int, through: tuple) -> tuple:
        """The `index`-th element of the tuple `name` holds, read back to its holder (`held`)."""
        op = found[computation, name]
        through = (*through, op.opcode)
        if op.opcode == 'tuple':
            return held(computation, op.operands[index], through)
        if op.opcode == 'parameter':
            loop = loops.get(computation)
            if computation == entry or (holder == 'loop' and loop is not None and loop.computation == entry):
                return through
            # A loop body's carry: the buffer its loop starts from.
            return (apart(through, op) if loop is None
                    else element(loop.computation, loop.operands[0], index, through))
        if op.opcode == 'while':
            # A loop's result: its carry's buffer, the one it started from.
            return element(computation, op.operands[0], index, through)
        updated = written(op, index)
        return apart(through, op) if updated is None else held(computation, updated, through)

    def held(computation: str, name: str, through: tuple = ()) -> tuple:
        """The producers `name` reads its state through back to the state the program holds, its last an
        `apart` where something else made it."""
        op = found[computation, name]
        through = (*through, op.opcode)
        if op.opcode == 'parameter':
            caller = fused.get(computation)
            if caller is not None:
                return held(caller.computation, caller.operands[op.proto.parameter_number], through)
            return through if computation == entry else apart(through, op)
        if op.opcode == 'get-tuple-element':
            return element(computation, op.operands[0], op.proto.tuple_index, through)
        if op.opcode in ('custom-call', 'fusion', 'dynamic-update-slice'):
            updated = written(op, None)
            return apart(through, op) if updated is None else held(computation, updated, through)
        if op.opcode == 'bitcast' or (op.opcode in ('dynamic-slice', 'slice') and computation in fused):
            return held(computation, op.operands[0], through)
        return apart(through, op)

    launches = [_Stepped(op.name, any(stated(op.computation, operand) for operand in _aliased(op)),
                         tuple(held(op.computation, operand) for operand in op.operands
                               if stated(op.computation, operand)))
                for op in every if op.opcode == 'custom-call'
                and any(stated(op.computation, operand) for operand in op.operands)]
    return launches, [op.name for op in every if op.opcode == 'copy' and stated(op.computation, op.name)]


def _assert_steps_in_place(executable: jax.stages.Compiled, *, holder: Holder, scanned: bool) -> None:
    """Every launch of `executable` reading a gated delta net's state (`STATE`) writes it in place, where its
    `holder` holds it unless `scanned` (`_steps`), and the program copies no such state. A scanned run's
    launches step the run's bank, which a decode over a store held a layer at a time packs from the holder's
    states each call and cuts apart after (row 14219), so no launch reads its holder's state there yet."""
    (module,) = executable.runtime_executable().hlo_modules()
    launches, copies = _steps(module, STATE, holder=holder)
    stepped = all(launch.in_place if scanned else launch.held for launch in launches)
    assert launches and stepped and not copies, (launches, copies)


def _decode_module(target: str, delta_decode: str):
    """`LM_HYBRID`'s (three gated delta nets to each attention layer) B8 decode step, its cache donated,
    compiled for a compile-only `target` at the default matmul precision, its gated delta nets' tokens on
    `delta_decode`'s kernel (`generation.chosen`): its optimized module."""
    model = _hybrid()
    ids = jnp.zeros((ROWS, 1), jnp.int32)
    mesh = _device(target)
    with (generation.chosen(layer_scan='plain', delta_decode=delta_decode),
          jax.default_matmul_precision('default')):
        variables = jax.eval_shape(lambda: nn.unbox(model.init(jax.random.key(0), ids)))
        cache = jax.eval_shape(lambda: nn.unbox(model.init(jax.random.key(0), ids, decode=True)))['cache']
        placed = _placed((variables, cache), NamedSharding(mesh, PartitionSpec()))

        def step(variables, cache):
            return model.apply({**variables, 'cache': cache}, ids, decode=True, mutable=['cache'])

        with compiling_for(target, mesh):
            compiled = jax.jit(step, donate_argnums=(1,)).trace(*placed).lower().compile()
    (module,) = compiled.runtime_executable().hlo_modules()
    return module


@COMPILES_ON_CPU
@pytest.mark.parametrize('delta_decode', ['step', 'fused'])
def test_a_donated_decode_reads_each_conv_weight_as_stored(delta_decode):
    """On an H100, each gated delta net's conv weight reaches the kernel that reads it (the short conv's
    before the step alone, the fused step's taking the conv) through no-copy views alone (`bitcast`): the
    kernels read their taps in the checkpoint's `[C, 1, K]` order (`conv.read_taps`), where read down
    `[K, C]` the step transposed each layer's weight first, weight-only work a layer a step (row 14174: 12 of
    an LM_HYBRID B8 step's), and a repack of it, a copy or a transpose of its own, would read it twice."""
    every = list(hlo.instructions(_decode_module('sm90', delta_decode)))
    users: dict[tuple[str, str], list] = {(op.computation, op.name): [] for op in every}
    for op in every:
        for operand in op.operands:
            users[op.computation, operand].append(op)
    weights = [op.name for op in every
               if op.entry and op.opcode == 'parameter' and 'conv1d____weight' in op.name]

    def parameter(computation: str, index: int) -> str:
        return next(op.name for op in every if op.computation == computation and op.opcode == 'parameter'
                    and op.proto.parameter_number == index)

    def reached(computation: str, name: str) -> tuple[set, set]:
        """The kernels `name` reaches through no-copy views (`bitcast`), into a fusion through its parameter
        (XLA's dynamic-slice fusion wrapping the fused step's launch), and the other instructions it meets."""
        kernels, others, frontier = set(), set(), [(computation, name)]
        while frontier:
            where, value = frontier.pop()
            for op in users[where, value]:
                if op.opcode == 'bitcast':
                    frontier.append((where, op.name))
                elif op.opcode == 'custom-call':
                    kernels.add(op.name)
                elif op.opcode == 'fusion':
                    (called,) = op.called
                    frontier.extend((called, parameter(called, index))
                                    for index, operand in enumerate(op.operands) if operand == value)
                else:
                    others.add(f'{op.opcode} {op.name}')
        return kernels, others

    entry = next(op.computation for op in every if op.entry)
    found = {weight: reached(entry, weight) for weight in weights}
    assert weights and all(kernels and not others for kernels, others in found.values()), found


@COMPILES_ON_CPU
@pytest.mark.parametrize('shards,checked', [(1, False), (2, False), (2, True)],
                         ids=['one device', 'split', 'in a checked map'])
def test_a_donated_state_steps_in_place_unless_a_checked_map_types_it(shards, checked):
    """A gated delta net's decode token (`decode_gated_delta_rule`) on H100s, its state donated by a program
    that owns it (`donated_state`): on one device and split over two (the step's own map, `manual_map`), its
    launch reads and writes the state in place (one operand aliased to its output), where an output of its
    own had the donated cache copy it back, 4 MiB a layer a step (row 14174); called inside a checked map
    (`jax.shard_map`, `check_vma`), whose state keeps its manual axes' type, which no launch takes as a ref,
    it takes the state's own output and compiles (head review of row 14174's in-place state). The same jitted
    function traced outside the scope, first, keeps the state's own output: the scope keys its trace."""
    mesh = _device('sm90', (('data', shards),))
    operands = _placed(_token(8), NamedSharding(mesh, PartitionSpec('data')))
    decode = decode_gated_delta_rule
    if checked:
        rows_spec = PartitionSpec('data')
        decode = jax.shard_map(decode_gated_delta_rule, in_specs=(rows_spec,) * 6, out_specs=rows_spec,
                               check_vma=True)
    stepped = jax.jit(decode, donate_argnums=(5,))

    def aliased() -> list:
        """The custom calls of `stepped`'s program whose output aliases an operand."""
        with (assumed('sm90'), jax.set_mesh(mesh),
              mgpu_core.artificial_shared_memory_limit(gpu_target('sm90').shared_limit)):
            (module,) = stepped.trace(*operands).lower().compile().runtime_executable().hlo_modules()
        return [op.name for op in hlo.instructions(module) if op.opcode == 'custom-call' and _aliased(op)]

    # The same jitted function on either side of the scope: its trace keyed on it (`_cache.DONATED_STATE`).
    kept = aliased()
    with donated_state():
        donated = aliased()
    assert not kept and bool(donated) == (not checked), (kept, donated)


@COMPILES_ON_CPU
def test_a_donated_states_step_in_a_checked_map_types_its_state_as_its_operands():
    """A decode token (`decode_gated_delta_rule`) traced in a program that donates its state (`donated_state`)
    inside a checked map over two H100s (`jax.shard_map`, `check_vma`): its initial state replicated, one
    zero row, beside a split `active`, the next state varies over the map's axis as its rows do, so a
    `psum` of it traces with no implicit cast (`jax_auto_pcast` off). Frozen from an in-place ref, the next
    state kept the initial state's replicated type, and the `psum` refused it (train 1511's K-F1)."""
    mesh = _device('sm90', (('data', 2),))
    split, whole = PartitionSpec('data'), PartitionSpec()

    def step(query, key, value, g, beta, state, active):
        _, next_state = decode_gated_delta_rule(query, key, value, g, beta, state, active)
        return jax.lax.psum(next_state.astype(jnp.float32), 'data')

    operands = (*_token(1), jax.ShapeDtypeStruct((2,), jnp.bool_))
    mapped = jax.shard_map(step, in_specs=(whole,) * 6 + (split,), out_specs=whole, check_vma=True)
    with (assumed('sm90'), jax.set_mesh(mesh), donated_state(), jax_config.auto_pcast(new_val=False),
          mgpu_core.artificial_shared_memory_limit(gpu_target('sm90').shared_limit)):
        jax.jit(mapped).trace(*operands).lower()


@COMPILES_ON_CPU
def test_a_launch_handed_a_gathered_copy_of_its_state_reads_apart():
    """The in-place witness's reading (`_steps`) of a program that hands the in-place launch a gathered copy
    of its donated state (`donated_state`): the launch aliases the copy, a buffer the program made apart
    from the state it holds, and the reading reports it so (no producer back to the state), as it would a
    cache a step sliced or moved before its launch, where no `copy` instruction states it."""
    mesh = _device('sm90')
    operands = _placed((*_token(ROWS), jax.ShapeDtypeStruct((ROWS,), jnp.int32)),
                       NamedSharding(mesh, PartitionSpec()))

    def step(query, key, value, g, beta, state, order):
        return decode_gated_delta_rule(query, key, value, g, beta, jnp.take(state, order, axis=0))

    with compiling_for('sm90', mesh), donated_state():
        executable = jax.jit(step, donate_argnums=(5,)).trace(*operands).lower().compile()
    (module,) = executable.runtime_executable().hlo_modules()
    launches, _ = _steps(module, _token(ROWS)[-1].shape, holder='program')
    assert launches and all(launch.in_place and not launch.held for launch in launches), launches


@COMPILES_ON_CPU
@pytest.mark.parametrize('layer_scan', ['scanned', 'plain'])
def test_a_servers_step_updates_each_recurrent_state_in_place(layer_scan):
    """A hybrid server's decode step (`dew.inference.serving_kernel`), eight slots and three iterations a
    call (the last two in its loop), compiled for an H100 as a server compiles it: the executable that
    chose its state's layout (`resident_formats`), which a server runs as its step where it reads the state
    in those formats, else the step compiled at them (`decode_programs`); its resident state donated, its
    stack's gated delta nets scanned as one bank (the default) or unrolled. Every launch reading a gated
    delta net's recurrent state writes it in place (`_advanced`'s `advance` in `donated_state`), on the
    donated cache (scanned, on the run's bank, which the step packs from the cache each call: row 14219),
    and the program copies no such state, where a launch writing the next state apart had the donated cache
    copy it back, 4 MiB a layer a step (row 14174)."""
    model = _hybrid(**PERIOD)
    ids = jnp.zeros((ROWS, 1), jnp.int32)
    mesh = _device('sm90')
    steps, stopping = 3, Finishing((), ())
    kept = []
    with generation.chosen(layer_scan=layer_scan), jax.default_matmul_precision('default'):
        variables = _placed(jax.eval_shape(lambda: nn.unbox(model.init(jax.random.key(0), ids))),
                            NamedSharding(mesh, PartitionSpec()))

        def prefilled(variables):
            return prefill_state(model, variables, ModelInputs(ids), decode_ops(model, variables, 0, None))[0]

        # The free slots' state as a server opens it: its cache beside the types its admissions' prefill
        # leaves (`Server._admitted_types`).
        _, prefill = jax.eval_shape(checkify.checkify(prefilled, errors=checkify.user_checks), variables)
        state = jax.eval_shape(functools.partial(opened, model, pad_id=0, slots=ROWS, capacity=64,
                                                 types=StateTypes.of(prefill)), variables)
        shardings = state_shardings(mesh, state)
        with (compiling_for('sm90', mesh),
              options.keeping(lambda program, lowered, executable: kept.append(executable))):
            policies = ((), stopping, None, END)
            formats, chosen = resident_formats(model, variables, 0, Dense(), steps, state, shardings, None,
                                               *policies)
            if chosen is None:
                # The server's step compiled at the formats chosen (`Server.__init__`).
                step, _ = decode_programs(formats, None, {})
                resident, carried = split_donated(_placed(state, shardings))
                step.compiled(model, variables, 0, Dense(), steps, resident, carried, None, *policies)
    _assert_steps_in_place(kept[-1], holder='program', scanned=layer_scan == 'scanned')


@COMPILES_ON_CPU
@pytest.mark.parametrize('layer_scan', ['scanned', 'plain'])
def test_a_generations_loop_updates_each_recurrent_state_in_place(layer_scan):
    """A hybrid model's generation, eight rows drawing four tokens each, its program compiled for an H100 as
    `generate` compiles it (`sampling.text._checked` through Dew's compile point), its prompt prefilled on
    the portable recurrence so its only launches reading a state are its tokens' steps, its gated delta nets
    scanned as one bank (the default) or unrolled: each step writes its state in place where the loop
    carries it (`_sample_rows`' `advance` in `donated_state`, each step's state the loop's own; scanned, on
    the run's bank, packed from the carry each step: row 14219), and the program copies no such state,
    where a launch writing the next state apart had the loop's carry copy it back, 4 MiB a layer a step
    (row 14174). The loop's start is the state its prefill made, read once a request."""
    model = _hybrid(**PERIOD)
    mesh = _device('sm90')
    whole = NamedSharding(mesh, PartitionSpec())
    with (generation.chosen(layer_scan=layer_scan, gated_delta_rule='xla', delta_decode='step'),
          jax.default_matmul_precision('default')):
        # A prompt of four tokens, prefilled by the portable chunked rule: a one-token prompt's took the step,
        # outside the loop, on a state of its own.
        prompt = jnp.zeros((ROWS, 4), jnp.int32)
        variables = jax.eval_shape(lambda: nn.unbox(model.init(jax.random.key(0), prompt)))
        placed = _placed((variables, ModelInputs(prompt), jax.random.split(jax.random.key(0), ROWS),
                          jnp.int32(4)), whole)
        generating = options.program(_checked, static_argnames=('model', 'trips', 'pad_id', 'n', 'packing'))
        with compiling_for('sm90', mesh):
            executable = generating.compiled(model, *placed, 4, 0, 1, (), Finishing((), ()), Sample(), END)
    _assert_steps_in_place(executable, holder='loop', scanned=layer_scan == 'scanned')
