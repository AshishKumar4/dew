"""Step FLOP measurement, MFU accounting and the persistent compilation cache.

The FLOP count is read off the optimized HLO, not out of XLA's
`cost_analysis()`. Cost analysis reports only the operations the compiler can
see arithmetic in, and a GPU backend hands its matmuls and convolutions to
cuBLAS and cuDNN custom calls, whose arithmetic it cannot see. Measured on
this repo's own benchmarks, that omitted 22.5x of the UNet step and 2.37x of
the small language-model step, and it moved between identical recompiles as
the compiler picked a visible Triton dot or an opaque cuBLAS call for the same
matmul (docs/research/benchmark-parity.md:5-9,93-102).
"""

import math
import re
import sys
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field

import jax

from dew.nn.kernels.generation import gpu_target

_cache_counts = {'/jax/compilation_cache/cache_hits': 0, '/jax/compilation_cache/cache_misses': 0}
_cache_lock = threading.Lock()


def _cache_event(event: str, **metadata) -> None:
    if event in _cache_counts:
        with _cache_lock:
            _cache_counts[event] += 1


jax.monitoring.register_event_listener(_cache_event)


def compilation_cache_counts() -> dict[str, float]:
    """Cumulative JAX persistent-cache events since Dew's instrumentation was imported.

    The listener is registered once per process. Counts include every jitted
    program, not just training steps. JAX emits cache_misses when it writes a
    cache entry, so compilations below its write thresholds do not count as
    misses. An unavailable or disabled persistent cache emits neither event.
    """
    with _cache_lock:
        return {f'train/{event.rsplit("/", 1)[-1]}': float(count)
                for event, count in _cache_counts.items()}

# One instruction: `%name = shape op(operands), attributes`, with ROOT optional.
# The shape is non-greedy so it stops at the operand list, which leaves a tuple
# result whole; the first shape inside it is the value, the rest is workspace.
_INSTRUCTION = re.compile(
    r'^\s*(?:ROOT\s+)?%(?P<name>[\w.\-]+)\s+=\s+(?P<shape>.*?)\s'
    r'(?P<op>[a-z][a-z0-9\-]*)\((?P<rest>.*)$')
_COMPUTATION = re.compile(r'^\s*(?:ENTRY\s+)?%?(?P<name>[\w.\-]+)\s*\(.*\)\s*->.*\{\s*$')
_DIMS = re.compile(r'[a-z][\w]*\[([\d,]*)\]')
_TRIP_COUNT = re.compile(r'"known_trip_count":\s*\{\s*"n":\s*"(\d+)"')
_INTEGER = re.compile(r'(-?\d+)\)')
_ELEMENT = re.compile(r'\s*([a-z]\w*)\[')
_INTEGER_TYPE = re.compile(r'([su])(8|16|32|64)$')
_INDEX = re.compile(r', index=(\d+)')
_NAME = re.compile(r'%([\w.\-]+)')
_CALLS = re.compile(r'(?:calls|to_apply|select|scatter|condition|body)=%([\w.\-]+)')
_BRANCHES = re.compile(r'branch_computations=\{([^}]*)\}')
_TARGET = re.compile(r'custom_call_target="([^"]+)"')
_TRACED = re.compile(r'op_name="([^"]*)"(?:[^}]*?source_file="([^"]*)"\s+source_line=(\d+))?')
_CONTRACTING = re.compile(r'lhs_contracting_dims=\{([\d,]*)\}')
_GEMM_CONTRACTING = re.compile(r'"lhs_contracting_dimensions":\s*\[([^\]]*)\]')
_WINDOW = re.compile(r'window=\{([^}]*)\}')
_LABELS = re.compile(r'dim_labels=(\S+?)(?:[,\s]|$)')
_GROUPS = re.compile(r'feature_group_count=(\d+)')
# A multiply-add is two FLOPs.
_MAC = 2
_CUSTOM_CALLS = (
    ("gemm", re.compile(r'^__cublas')),
    ("convolution", re.compile(r'^__cudnn\$conv')),
    ("attention", re.compile(r'^__cudnn\$fmha|^dew_flash3?_mha')),
    ("triton", re.compile(r'^__gpu\$xla\.gpu\.triton$')),
)


@dataclass(frozen=True)
class _Instruction:
    """The parts of one HLO instruction the FLOP count reads."""

    op: str
    dims: tuple[int, ...]
    operands: tuple[str, ...]
    attributes: str
    """Its attributes after the operands, its `metadata={...}` taken out."""
    recomputed: bool = False
    """Whether it recomputes the forward: JAX's remat names its recomputation
    `rematted_computation`, and XLA's own rematerialization clones an
    instruction as `<name>.remat`."""
    traced: tuple[str, ...] = ()
    """Its metadata's `op_name source_file:line`, interned: a module repeats
    each a great many times."""


@dataclass
class _Computation:
    """One HLO computation: its instructions by name, their result shapes and
    element types, its root, and the value of each integer scalar constant it
    defines."""

    instructions: dict[str, _Instruction] = field(default_factory=dict)
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    elements: dict[str, str] = field(default_factory=dict)
    root: str | None = None
    constants: dict[str, int] = field(default_factory=dict)


def _dims(text: str) -> tuple[int, ...]:
    """Dimensions of the first shape in `text`, which for a tuple is its head."""
    match = _DIMS.search(text)
    if match is None:
        return ()
    body = match.group(1).strip()
    return tuple(int(d) for d in body.split(',')) if body else ()


def _without_metadata(attributes: str) -> tuple[str, str]:
    """`attributes` with its `metadata={...}` taken out, and that metadata."""
    start = attributes.find('metadata={')
    if start < 0:
        return attributes, ''
    quoted, end = False, len(attributes)
    for index in range(start + len('metadata={'), len(attributes)):
        character = attributes[index]
        if character == '"' and attributes[index - 1] != '\\':
            quoted = not quoted
        elif character == '}' and not quoted:
            end = index + 1
            break
    return attributes[:start] + attributes[end:], attributes[start:end]


def _split_operands(rest: str) -> tuple[tuple[str, ...], str]:
    """The operand names of an instruction, and the attribute text after them."""
    depth, end = 1, len(rest)
    for index, character in enumerate(rest):
        depth += (character == '(') - (character == ')')
        if depth == 0:
            end = index
            break
    operands = tuple(
        match.group(1) for match in re.finditer(r'%([\w.\-]+)', rest[:end]))
    return operands, rest[end + 1:]


def _parse(text: str | Iterable[str]) -> tuple[dict[str, _Computation], str | None]:
    """The module's computations, and the name of its entry computation, from
    its text or its lines one at a time (a dump file read as it streams): a
    module's metadata is most of its text, and only its traced names are kept."""
    computations: dict[str, _Computation] = {}
    entry, current = None, None
    for line in text.splitlines() if isinstance(text, str) else text:
        stripped = line.strip()
        header = _COMPUTATION.match(line)
        if header and ' = ' not in stripped:
            current = header.group('name')
            computations[current] = _Computation()
            if stripped.startswith('ENTRY'):
                entry = current
            continue
        if stripped == '}':
            current = None
            continue
        match = _INSTRUCTION.match(line)
        if match is None or current is None:
            continue
        name = match.group('name')
        operands, attributes = _split_operands(match.group('rest'))
        dims = _dims(match.group('shape'))
        computation = computations[current]
        computation.shapes[name] = dims
        element = _ELEMENT.match(match.group('shape'))
        computation.elements[name] = element.group(1) if element else ''
        recomputed = '.remat' in name or '.remat' in current or 'rematted_computation' in attributes
        attributes, metadata = _without_metadata(attributes)
        traced = tuple(sys.intern(f"{op_name} {file}:{line}".strip() if file else op_name)
                       for op_name, file, line in _TRACED.findall(metadata) if op_name)
        computation.instructions[name] = _Instruction(match.group('op'), dims, operands, attributes,
                                                      recomputed, traced)
        if stripped.startswith('ROOT'):
            computation.root = name
        literal = _INTEGER.match(match.group('rest'))
        integer = _INTEGER_TYPE.match(computation.elements[name])
        if match.group('op') == 'constant' and not dims and literal and integer:
            computation.constants[name] = int(literal.group(1))
    return computations, entry


def _window(attributes: str) -> dict[str, list[int]]:
    """A convolution's window: its size and its dilations, per spatial axis."""
    window = {'size': [1], 'lhs_dilate': [1]}
    match = _WINDOW.search(attributes)
    if match is None:
        return window
    for key in window:
        found = re.search(key + r'=([0-9x]+)', match.group(1))
        if found:
            window[key] = [int(extent) for extent in found.group(1).split('x')]
    return window


def _dot_flops(instruction: _Instruction, shapes: dict[str, tuple[int, ...]],
               contracting: tuple[int, ...]) -> float:
    """Every output element costs one multiply-add per contracted element."""
    lhs = shapes.get(instruction.operands[0]) if instruction.operands else None
    if lhs is None:
        return 0.0
    contracted = math.prod(lhs[axis] for axis in contracting if axis < len(lhs))
    return _MAC * math.prod(instruction.dims) * contracted


def _convolution_flops(instruction: _Instruction,
                       shapes: dict[str, tuple[int, ...]]) -> float:
    """Output elements times the kernel window times the input features.

    Dividing by the input dilation counts a strided convolution's gradient at
    the multiply-adds its forward pass costs: the dilated positions are zeros,
    and no kernel multiplies them.
    """
    lhs = shapes.get(instruction.operands[0]) if instruction.operands else None
    labels = _LABELS.search(instruction.attributes)
    if lhs is None or labels is None:
        return 0.0
    features = lhs[labels.group(1).split('_')[0].index('f')]
    window = _window(instruction.attributes)
    groups = _GROUPS.search(instruction.attributes)
    macs = (math.prod(instruction.dims) * math.prod(window['size']) * features
            / (int(groups.group(1)) if groups else 1)
            / math.prod(window['lhs_dilate']))
    return _MAC * macs


def _cudnn_convolution_flops(target: str, instruction: _Instruction,
                             shapes: dict[str, tuple[int, ...]]) -> float:
    """A cuDNN convolution call, at the multiply-adds of its forward shape.

    XLA keeps the forward convolution's window and dim_labels on all three
    kinds, so each reduces to the same product `B Ho Wo O Kh Kw I / groups`
    read off whichever operand holds each factor: the forward call takes the
    input and the filter, the input-gradient call the output gradient and the
    filter, the filter-gradient call the input and the output gradient.
    """
    lhs = shapes.get(instruction.operands[0]) if instruction.operands else None
    rhs = shapes.get(instruction.operands[1]) if len(instruction.operands) > 1 else None
    labels = _LABELS.search(instruction.attributes)
    if lhs is None or rhs is None or labels is None:
        return 0.0
    lhs_labels, output_labels = labels.group(1).split('_')[0], labels.group(1).split('->')[1]
    kernel = math.prod(_window(instruction.attributes)['size'])
    groups = _GROUPS.search(instruction.attributes)
    dims = instruction.dims
    if 'BackwardInput' in target:
        macs = math.prod(lhs) * kernel * dims[lhs_labels.index('f')]
    elif 'BackwardFilter' in target:
        outputs = rhs[output_labels.index('f')]
        macs = math.prod(dims) * math.prod(rhs) / outputs
    else:
        macs = math.prod(dims) * kernel * lhs[lhs_labels.index('f')]
    return _MAC * macs / (int(groups.group(1)) if groups else 1)


_FLASH = {'dew_flash_mha_fwd': (0, False), 'dew_flash_mha_bwd': (1, True),
          'dew_flash3_mha_fwd': (0, False), 'dew_flash3_mha_bwd': (1, True)}
"""FlashAttention-2's and FlashAttention-3's calls (dew_flash_attn): where
the query sits among each one's operands (the backward's first is the
output's gradient), and whether it is the backward."""


def _fused_attention_flops(instruction: _Instruction, shapes: dict[str, tuple[int, ...]], *,
                           query: int, backward: bool) -> float:
    """A fused-attention call (cuDNN's or FlashAttention-2's), from the query
    and key it is given, the operands at `query` and the one after it.

    The kernel keeps the scores off memory, so nothing in the module states
    its arithmetic; the operands do. They are `[batch, sequence, heads, width]`
    (checked against the calls XLA emits for `jax.nn.dot_product_attention`,
    including grouped-query and cross attention, where the key carries its own
    head count and sequence, and against dew_flash_attn's). Both products of
    the forward pass, `Q K^T` and `P V`, cost one multiply-add per query, key,
    head and width; the backward pass runs four such products, for dV, dP, dQ
    and dK. That is the whole square: under a causal mask the kernels skip
    about half of it, so the count keeps MFU's convention and train/hfu reads
    up to twice their hardware's work there.
    """
    if len(instruction.operands) < query + 2:
        return 0.0
    queries = shapes.get(instruction.operands[query])
    key = shapes.get(instruction.operands[query + 1])
    if queries is None or key is None or len(queries) != 4 or len(key) != 4:
        return 0.0
    batch, length, heads, width = queries
    products = 4 if backward else 2
    return products * _MAC * batch * heads * length * key[1] * width


def custom_call_kind(target: str) -> str | None:
    """What a custom call computes, by its target: cuBLAS's GEMMs ('gemm'),
    cuDNN's convolutions ('convolution'), the fused attentions of cuDNN and
    dew_flash_attn ('attention'), the Triton kernels Pallas lowers to
    ('triton'); None for any other. The FLOP count and
    tools/attribute_step.py read custom calls through this one table."""
    return next((kind for kind, pattern in _CUSTOM_CALLS if pattern.search(target)), None)


@dataclass(frozen=True)
class HloInstruction:
    """One instruction of an optimized HLO module, as a profile attributes it."""

    op: str
    target: str
    """A custom call's target, empty for any other op."""
    fused: frozenset[str]
    """The ops of the computations it calls, theirs included: a fusion's body."""
    recomputed: bool
    traced: frozenset[str]
    """Where JAX traced it and what it fused: `op_name source_file:line` of
    its own metadata and of each op in the computations it calls. A
    multi-output fusion carries none of its own, its ops do."""


def hlo_instructions(text: str | Iterable[str]) -> dict[str, HloInstruction]:
    """Every instruction of the module in `text` (`compiled.as_text()`, or an XLA
    dump's after-optimizations HLO, whole or as its lines) by name, read by the
    parser the FLOP count reads (`hlo_flops`). Each computation's body is
    gathered once, so the fusions that call one share its sets."""
    computations, _ = _parse(text)
    bodies: dict[str, tuple[frozenset[str], frozenset[str]]] = {}

    def fused(name: str, seen: frozenset[str] = frozenset()) -> tuple[frozenset[str], frozenset[str]]:
        if name in bodies:
            return bodies[name]
        computation = computations.get(name)
        if computation is None or name in seen:
            return frozenset(), frozenset()
        ops = {instruction.op for instruction in computation.instructions.values()}
        names = {each for instruction in computation.instructions.values() for each in instruction.traced}
        for instruction in computation.instructions.values():
            for called in _CALLS.findall(instruction.attributes):
                inner_ops, inner_names = fused(called, seen | {name})
                ops |= inner_ops
                names |= inner_names
        bodies[name] = frozenset(ops), frozenset(names)
        return bodies[name]

    found = {}
    for computation in computations.values():
        for name, instruction in computation.instructions.items():
            target = _TARGET.search(instruction.attributes)
            calls = _CALLS.findall(instruction.attributes) if instruction.op == 'fusion' else []
            called = [fused(each) for each in calls]
            ops = called[0][0] if len(called) == 1 else frozenset().union(*(each for each, _ in called))
            names = frozenset(instruction.traced).union(*(each for _, each in called))
            found[name] = HloInstruction(instruction.op, target.group(1) if target else '', ops,
                                         instruction.recomputed, names)
    computations.clear()
    return found


def _instruction_flops(instruction: _Instruction,
                       shapes: dict[str, tuple[int, ...]]) -> float:
    """The multiply-add work of one instruction, and zero for anything else.

    Only the matmuls and convolutions count. Everything the compiler leaves
    elementwise (the optimizer, the EMA, normalization, the softmax, the loss
    reductions) is memory-bound work that a FLOP utilisation figure is not
    about, which is the convention this repo's benchmarks state
    (docs/research/benchmark-parity.md:44-48).
    """
    if instruction.op == 'dot':
        contracting = _CONTRACTING.search(instruction.attributes)
        axes = () if contracting is None else tuple(
            int(axis) for axis in contracting.group(1).split(',') if axis)
        return _dot_flops(instruction, shapes, axes)
    if instruction.op == 'convolution':
        return _convolution_flops(instruction, shapes)
    if instruction.op != 'custom-call':
        return 0.0
    target = _TARGET.search(instruction.attributes)
    target = target.group(1) if target else ''
    kind = custom_call_kind(target)
    if kind == 'gemm':
        contracting = _GEMM_CONTRACTING.search(instruction.attributes)
        axes = () if contracting is None else tuple(
            int(axis.strip(' "')) for axis in contracting.group(1).split(',')
            if axis.strip(' "'))
        return _dot_flops(instruction, shapes, axes)
    if kind == 'convolution':
        return _cudnn_convolution_flops(target, instruction, shapes)
    if kind == 'attention':
        query, backward = _FLASH.get(target, (0, target.endswith('Backward')))
        return _fused_attention_flops(instruction, shapes, query=query, backward=backward)
    return 0.0


def _call_counts(instruction: _Instruction, caller: _Computation,
                 computations: dict[str, _Computation]) -> dict[str, float]:
    """The computations this instruction runs, and how often it runs each.

    A loop body runs once per iteration. The CPU backend states the count as
    `known_trip_count`; a v6e executable states none, so a loop without it is
    read off its counter (`_counted_trips`). A loop whose count neither gives
    becomes infinite, and the caller then reports no count. A conditional
    runs one of its branches, so counting every branch bounds it.
    """
    counts = dict.fromkeys(_CALLS.findall(instruction.attributes), 1.0)
    branches = _BRANCHES.search(instruction.attributes)
    if branches is not None:
        counts.update(dict.fromkeys(_NAME.findall(branches.group(1)), 1.0))
    if instruction.op == 'while':
        body = _called(instruction, 'body')
        trip = _TRIP_COUNT.search(instruction.attributes)
        if body:
            counts[body] = (float(trip.group(1)) if trip
                            else _counted_trips(instruction, caller, computations))
    return counts


def _counted_trips(loop: _Instruction, caller: _Computation,
                   computations: dict[str, _Computation]) -> float:
    """The trip count of a loop that counts one integer tuple element from a
    constant start, by a constant positive step, while it is below a constant
    bound: the loop every `fori_loop` and `scan` with static bounds lowers
    to, and the form XLA's own trip-count analysis reads. Infinite for any
    other loop, and for one whose last step overflows the counter's type,
    since a wrapped counter need never reach the bound."""
    condition = computations.get(_called(loop, 'condition'))
    body = computations.get(_called(loop, 'body'))
    if condition is None or body is None or not loop.operands:
        return math.inf
    compare = condition.instructions.get(condition.root or '')
    if (compare is None or compare.op != 'compare' or 'direction=LT' not in compare.attributes
            or len(compare.operands) != 2):
        return math.inf
    counter = _INTEGER_TYPE.match(condition.elements.get(_origin(condition, compare.operands[0]), ''))
    index = _element_index(condition, compare.operands[0])
    bound = _constant(condition, compare.operands[1])
    if counter is None or index is None or bound is None:
        return math.inf
    start = _constant(caller, _tuple_element(caller, loop.operands[0], index))
    step = _increment(body, _tuple_element(body, body.root, index), index)
    if start is None or step is None or step <= 0:
        return math.inf
    trips = max(0, -(-(bound - start) // step))
    signed, bits = counter.group(1) == 's', int(counter.group(2))
    if start + trips * step > (1 << (bits - signed)) - 1:
        return math.inf
    return float(trips)


def _called(instruction: _Instruction, key: str) -> str:
    """The computation an instruction names under `key`, or ''."""
    match = re.search(key + r'=%([\w.\-]+)', instruction.attributes)
    return match.group(1) if match else ''


def _origin(computation: _Computation, name: str | None) -> str:
    """The name of the instruction `name` copies, through any chain of copies."""
    name = name or ''
    while name in computation.instructions and computation.instructions[name].op == 'copy':
        name = computation.instructions[name].operands[0]
    return name


def _source(computation: _Computation, name: str | None) -> _Instruction | None:
    """The instruction `name` copies, through any chain of copies."""
    return computation.instructions.get(_origin(computation, name))


def _constant(computation: _Computation, name: str | None) -> int | None:
    """The integer scalar constant `name` holds, through copies, or None."""
    return computation.constants.get(_origin(computation, name))


def _element_index(computation: _Computation, name: str | None) -> int | None:
    """Which element of the computation's tuple parameter `name` reads, or None."""
    read = _source(computation, name)
    if read is None or read.op != 'get-tuple-element' or not read.operands:
        return None
    tuple_ = computation.instructions.get(read.operands[0])
    index = _INDEX.search(read.attributes)
    return int(index.group(1)) if tuple_ is not None and tuple_.op == 'parameter' and index else None


def _tuple_element(computation: _Computation, name: str | None, index: int) -> str | None:
    """The operand at `index` of the tuple `name` builds, or None."""
    built = _source(computation, name)
    if built is None or built.op != 'tuple' or index >= len(built.operands):
        return None
    return built.operands[index]


def _increment(body: _Computation, name: str | None, index: int) -> int | None:
    """The constant the body adds to element `index` of its parameter to make
    `name`, or None when `name` is anything else."""
    added = _source(body, name)
    if added is None or added.op != 'add' or len(added.operands) != 2:
        return None
    for counter, step in (added.operands, added.operands[::-1]):
        if _element_index(body, counter) == index:
            return _constant(body, step)
    return None


def _weights(computations: dict[str, _Computation], entry: str) -> dict[str, float]:
    """How many times each computation runs per call of the entry computation.

    HLO computations cannot recurse, so the call graph is a DAG and one pass in
    topological order gives every computation the sum of its callers' counts.
    Anything the entry cannot reach keeps a count of zero.
    """
    calls = {
        name: _merged_call_counts(computation, computations)
        for name, computation in computations.items()}
    callers = dict.fromkeys(computations, 0)
    for callees in calls.values():
        for callee in callees:
            if callee in callers:
                callers[callee] += 1
    weights = dict.fromkeys(computations, 0.0)
    weights[entry] = 1.0
    ready = [name for name, count in callers.items() if count == 0]
    while ready:
        name = ready.pop()
        for callee, times in calls[name].items():
            if callee not in weights:
                continue
            weights[callee] += weights[name] * times
            callers[callee] -= 1
            if callers[callee] == 0:
                ready.append(callee)
    return weights


def _merged_call_counts(computation: _Computation,
                        computations: dict[str, _Computation]) -> dict[str, float]:
    """Per call of this computation, how often each computation it names runs."""
    merged: dict[str, float] = {}
    for instruction in computation.instructions.values():
        for name, times in _call_counts(instruction, computation, computations).items():
            merged[name] = merged.get(name, 0.0) + times
    return merged


@dataclass(frozen=True)
class StepFlops:
    """One step's FLOPs: the `model`'s, the forward a remat recomputes counted
    once, and the `hardware`'s, every matmul the device runs. MFU is read off
    the first and HFU off the second (PaLM's distinction)."""

    model: float
    hardware: float

    def __add__(self, other: "StepFlops") -> "StepFlops":
        return StepFlops(self.model + other.model, self.hardware + other.hardware)


NO_FLOPS = StepFlops(0.0, 0.0)
"""The FLOPs of no step, where an interval's sum starts."""


def step_flops(compiled: jax.stages.Compiled) -> StepFlops | None:
    """FLOPs for one call of an executable that is already compiled.

    Every matmul and convolution in the optimized module counts once per time
    the module runs it, whether it is a `dot`, a `convolution`, or one of the
    cuBLAS, cuDNN convolution and cuDNN fused-attention custom calls a GPU
    backend hands them to. Backward passes count because they are in there.
    The forward a remat recomputes counts once in `model` and twice in
    `hardware`, because the card runs it twice. None comes back when the
    module contains a loop whose length neither XLA states nor its counter
    gives, since the count would then be the body's, not the run's.
    """
    text = compiled.as_text()
    if text is None:
        return None
    model, hardware = hlo_flops(text, recomputed=False), hlo_flops(text)
    return None if model is None or hardware is None else StepFlops(model, hardware)


def compiled_flops(compiled: jax.stages.Compiled) -> float | None:
    """The hardware FLOPs of one call of a compiled executable (`step_flops`)."""
    flops = step_flops(compiled)
    return None if flops is None else flops.hardware


def hlo_flops(text: str, *, recomputed: bool = True) -> float | None:
    """Matmul and convolution FLOPs of one call of an optimized HLO module;
    without the forward a remat recomputes where `recomputed` is False."""
    computations, entry = _parse(text)
    if entry is None:
        return None
    weights = _weights(computations, entry)
    total = 0.0
    for name, computation in computations.items():
        instructions = sum(_instruction_flops(instruction, computation.shapes)
                           for instruction in computation.instructions.values()
                           if recomputed or not instruction.recomputed)
        if not instructions or not weights[name]:
            continue
        total += weights[name] * instructions
    return total if math.isfinite(total) else None


def model_flops_utilization(
    flops_per_step: float | None, step_time: float
) -> float | None:
    """Fraction of one device's dense peak achieved by its executable.

    The optimized module is the program one device runs under SPMD, so its
    shapes are per-device and the denominator is one device's peak rather
    than the mesh's.
    """
    if not flops_per_step or step_time <= 0:
        return None
    peak = gpu_target(jax.devices()[0].device_kind).bf16_peak
    if peak is None:
        return None
    return flops_per_step / step_time / peak
