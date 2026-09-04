"""Step FLOP measurement, MFU accounting and the persistent compilation cache.

The FLOP count is read off the optimized HLO rather than out of XLA's
`cost_analysis()`. Cost analysis reports only the operations the compiler can
see arithmetic in, and a GPU backend hands its matmuls and convolutions to
cuBLAS and cuDNN custom calls, whose arithmetic it cannot see. Measured on
this repo's own benchmarks, that omitted 22.5x of the UNet step and 2.37x of
the small language-model step, and it moved between identical recompiles as
the compiler picked a visible Triton dot or an opaque cuBLAS call for the same
matmul (docs/research/benchmark-parity.md:5-9,93-102).
"""

import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jax

# Dense bf16 peak per chip, from the vendors' own spec sheets. Only used to turn
# measured FLOPs into a utilisation percentage; unknown hardware just skips MFU.
PEAK_FLOPS_PER_DEVICE = {
    'TPU v2': 45e12,
    'TPU v3': 123e12,
    'TPU v4': 275e12,
    'TPU v5 lite': 197e12,
    'TPU v5e': 197e12,
    'TPU v5': 459e12,
    'TPU v5p': 459e12,
    'TPU v6 lite': 918e12,
    'TPU v6e': 918e12,
    'NVIDIA A100': 312e12,
    'NVIDIA H100': 989e12,
    'NVIDIA H200': 989e12,
    'NVIDIA GeForce RTX 4080': 97.5e12,
}


# One instruction: `%name = shape op(operands), attributes`, with ROOT optional.
# The shape is non-greedy so it stops at the operand list, which leaves a tuple
# result whole; the first shape inside it is the value, the rest is workspace.
_INSTRUCTION = re.compile(
    r'^\s*(?:ROOT\s+)?%(?P<name>[\w.\-]+)\s+=\s+(?P<shape>.*?)\s'
    r'(?P<op>[a-z][a-z0-9\-]*)\((?P<rest>.*)$')
_COMPUTATION = re.compile(r'^\s*(?:ENTRY\s+)?%?(?P<name>[\w.\-]+)\s*\(.*\)\s*->.*\{\s*$')
_DIMS = re.compile(r'[a-z][\w]*\[([\d,]*)\]')
_TRIP_COUNT = re.compile(r'"known_trip_count":\s*\{\s*"n":\s*"(\d+)"')
_NAME = re.compile(r'%([\w.\-]+)')
_CALLS = re.compile(r'(?:calls|to_apply|select|scatter|condition|body)=%([\w.\-]+)')
_BRANCHES = re.compile(r'branch_computations=\{([^}]*)\}')
_TARGET = re.compile(r'custom_call_target="([^"]+)"')
_CONTRACTING = re.compile(r'lhs_contracting_dims=\{([\d,]*)\}')
_GEMM_CONTRACTING = re.compile(r'"lhs_contracting_dimensions":\s*\[([^\]]*)\]')
_WINDOW = re.compile(r'window=\{([^}]*)\}')
_LABELS = re.compile(r'dim_labels=(\S+?)(?:[,\s]|$)')
_GROUPS = re.compile(r'feature_group_count=(\d+)')
# A multiply-add is two FLOPs.
_MAC = 2


@dataclass(frozen=True)
class _Instruction:
    """The parts of one HLO instruction the FLOP count reads."""

    op: str
    dims: Tuple[int, ...]
    operands: Tuple[str, ...]
    attributes: str


@dataclass
class _Computation:
    """One HLO computation: its instructions and their result shapes."""

    instructions: List[_Instruction]
    shapes: Dict[str, Tuple[int, ...]]


def _dims(text: str) -> Tuple[int, ...]:
    """Dimensions of the first shape in `text`, which for a tuple is its head."""
    match = _DIMS.search(text)
    if match is None:
        return ()
    body = match.group(1).strip()
    return tuple(int(d) for d in body.split(',')) if body else ()


def _split_operands(rest: str) -> Tuple[Tuple[str, ...], str]:
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


def _parse(text: str) -> Tuple[Dict[str, _Computation], Optional[str]]:
    """The module's computations, and the name of its entry computation."""
    computations: Dict[str, _Computation] = {}
    entry, current = None, None
    for line in text.splitlines():
        stripped = line.strip()
        header = _COMPUTATION.match(line)
        if header and ' = ' not in stripped:
            current = header.group('name')
            computations[current] = _Computation([], {})
            if stripped.startswith('ENTRY'):
                entry = current
            continue
        if stripped == '}':
            current = None
            continue
        match = _INSTRUCTION.match(line)
        if match is None or current is None:
            continue
        operands, attributes = _split_operands(match.group('rest'))
        dims = _dims(match.group('shape'))
        computation = computations[current]
        computation.shapes[match.group('name')] = dims
        computation.instructions.append(
            _Instruction(match.group('op'), dims, operands, attributes))
    return computations, entry


def _window(attributes: str) -> Dict[str, List[int]]:
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


def _dot_flops(instruction: _Instruction, shapes: Dict[str, Tuple[int, ...]],
               contracting: Tuple[int, ...]) -> float:
    """Every output element costs one multiply-add per contracted element."""
    lhs = shapes.get(instruction.operands[0]) if instruction.operands else None
    if lhs is None:
        return 0.0
    contracted = math.prod(lhs[axis] for axis in contracting if axis < len(lhs))
    return _MAC * math.prod(instruction.dims) * contracted


def _convolution_flops(instruction: _Instruction,
                       shapes: Dict[str, Tuple[int, ...]]) -> float:
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
                             shapes: Dict[str, Tuple[int, ...]]) -> float:
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


def _fused_attention_flops(target: str, instruction: _Instruction,
                           shapes: Dict[str, Tuple[int, ...]]) -> float:
    """A cuDNN fused-attention call, from the query and key it is given.

    The kernel keeps the scores off memory, so nothing in the module states
    its arithmetic; the operands do. They are `[batch, sequence, heads, width]`
    (checked against the calls XLA emits for `jax.nn.dot_product_attention`,
    including grouped-query and cross attention, where the key carries its own
    head count and sequence). Both products of the forward pass, `Q K^T` and
    `P V`, cost one multiply-add per query, key, head and width; the backward
    pass runs four such products, for dV, dP, dQ and dK.
    """
    if len(instruction.operands) < 2:
        return 0.0
    query = shapes.get(instruction.operands[0])
    key = shapes.get(instruction.operands[1])
    if query is None or key is None or len(query) != 4 or len(key) != 4:
        return 0.0
    batch, queries, heads, width = query
    products = 4 if target.endswith('Backward') else 2
    return products * _MAC * batch * heads * queries * key[1] * width


def _instruction_flops(instruction: _Instruction,
                       shapes: Dict[str, Tuple[int, ...]]) -> float:
    """The multiply-add work of one instruction, and zero for anything else.

    Only the matmuls and convolutions count. Everything the compiler leaves
    elementwise - the optimizer, the EMA, normalization, the softmax, the loss
    reductions - is memory-bound work that a FLOP utilisation figure is not
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
    if target.startswith('__cublas'):
        contracting = _GEMM_CONTRACTING.search(instruction.attributes)
        axes = () if contracting is None else tuple(
            int(axis.strip(' "')) for axis in contracting.group(1).split(',')
            if axis.strip(' "'))
        return _dot_flops(instruction, shapes, axes)
    if target.startswith('__cudnn$conv'):
        return _cudnn_convolution_flops(target, instruction, shapes)
    if target.startswith('__cudnn$fmha'):
        return _fused_attention_flops(target, instruction, shapes)
    return 0.0


def _call_counts(instruction: _Instruction) -> Dict[str, float]:
    """The computations this instruction runs, and how often it runs each.

    A loop body runs once per iteration, which XLA states as
    `known_trip_count` whenever the length is known, and every `jax.lax.scan`
    is such a loop. An unknown trip count becomes infinite rather than one: a
    body counted once when it runs a hundred times is a wrong MFU, not an
    approximate one, and an infinite count leaves the caller to report nothing.
    A conditional runs one of its branches, so counting every branch bounds it.
    """
    counts = {name: 1.0 for name in _CALLS.findall(instruction.attributes)}
    branches = _BRANCHES.search(instruction.attributes)
    if branches is not None:
        counts.update({name: 1.0 for name in _NAME.findall(branches.group(1))})
    if instruction.op == 'while':
        body = re.search(r'body=%([\w.\-]+)', instruction.attributes)
        trip = _TRIP_COUNT.search(instruction.attributes)
        if body is not None:
            counts[body.group(1)] = float(trip.group(1)) if trip else math.inf
    return counts


def _weights(computations: Dict[str, _Computation], entry: str) -> Dict[str, float]:
    """How many times each computation runs per call of the entry computation.

    HLO computations cannot recurse, so the call graph is a DAG and one pass in
    topological order gives every computation the sum of its callers' counts.
    Anything the entry cannot reach keeps a count of zero.
    """
    calls = {
        name: _merged_call_counts(computation)
        for name, computation in computations.items()}
    callers = {name: 0 for name in computations}
    for callees in calls.values():
        for callee in callees:
            if callee in callers:
                callers[callee] += 1
    weights = {name: 0.0 for name in computations}
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


def _merged_call_counts(computation: _Computation) -> Dict[str, float]:
    """Per call of this computation, how often each computation it names runs."""
    merged: Dict[str, float] = {}
    for instruction in computation.instructions:
        for name, times in _call_counts(instruction).items():
            merged[name] = merged.get(name, 0.0) + times
    return merged


def compiled_flops(compiled: jax.stages.Compiled) -> Optional[float]:
    """FLOPs for one call of an executable that is already compiled.

    Reading the count off the executable the loop actually runs costs nothing;
    compiling a second one to ask the same question costs a full XLA compile.

    Every matmul and convolution in the optimized module counts once per time
    the module runs it, whether it is a `dot`, a `convolution`, or one of the
    cuBLAS, cuDNN convolution and cuDNN fused-attention custom calls a GPU
    backend hands them to. Backward passes count because they are in there;
    remat counts the forward it recomputes twice, because the card runs it
    twice. None comes back when the module contains a loop whose length XLA
    does not state, since the count would be the body's rather than the run's.
    """
    text = compiled.as_text()
    return None if text is None else hlo_flops(text)


def hlo_flops(text: str) -> Optional[float]:
    """Matmul and convolution FLOPs of one call of an optimized HLO module."""
    computations, entry = _parse(text)
    if entry is None:
        return None
    weights = _weights(computations, entry)
    total = 0.0
    for name, computation in computations.items():
        instructions = sum(_instruction_flops(instruction, computation.shapes)
                           for instruction in computation.instructions)
        if not instructions or not weights[name]:
            continue
        total += weights[name] * instructions
    return total if math.isfinite(total) else None


def step_flops(jitted: jax.stages.Wrapped, *args: object, **kwargs: object) -> Optional[float]:
    """FLOPs for one call of a jitted function, straight from the compiler.

    Measured rather than derived from a hand-written parameter-count formula, so
    it stays honest across architectures, remat and gradient accumulation.

    Compiles the function, which the caller has usually already paid for: hold
    on to the compiled executable and use compiled_flops instead.
    """
    return compiled_flops(jitted.lower(*args, **kwargs).compile())


def model_flops_utilization(
    flops_per_step: Optional[float], step_time: float
) -> Optional[float]:
    """Fraction of one device's dense peak achieved by its executable.

    The optimized module is the program one device runs under SPMD, so its
    shapes are per-device and the denominator is one device's peak, not the
    whole mesh's.
    """
    if not flops_per_step or step_time <= 0:
        return None
    peak = PEAK_FLOPS_PER_DEVICE.get(jax.devices()[0].device_kind)
    if peak is None:
        return None
    return flops_per_step / step_time / peak


def default_compilation_cache_dir() -> str:
    """Where compiled executables go unless a run names somewhere else.

    Under the user's cache home, which is where a directory that can be deleted
    at any moment belongs: not in the repo and not next to the checkpoints.
    """
    home = os.environ.get('XDG_CACHE_HOME') or os.path.join('~', '.cache')
    return os.path.expanduser(os.path.join(home, 'dew', 'xla'))


def enable_compilation_cache(path: str):
    """Persist compiled executables so restarts skip XLA compilation.

    The dominant cost of a restart-heavy TPU workflow, where every run otherwise
    recompiles the same step function from scratch.
    """
    os.makedirs(path, exist_ok=True)
    jax.config.update('jax_compilation_cache_dir', path)
    # Defaults skip small/fast compilations; a training step is neither, and
    # caching everything keeps startup predictable.
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.0)
