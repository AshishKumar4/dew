"""Step FLOP measurement, MFU accounting and the persistent compilation cache's counts; the programs a jitted
function needs (`Compiles`) and the processes held to loaded programs (`executable_only`) are
`dew.telemetry.compiles`'.

The FLOP count is read off the optimized HLO, not out of XLA's
`cost_analysis()`. Cost analysis reports only the operations the compiler can
see arithmetic in, and a GPU backend hands its matmuls and convolutions to
cuBLAS and cuDNN custom calls, whose arithmetic it cannot see. Measured on
this repo's own benchmarks, that omitted 22.5x of the UNet step and 2.37x of
the small language-model step, and it moved between identical recompiles as
the compiler picked a visible Triton dot or an opaque cuBLAS call for the same
matmul (docs/research/benchmark-parity.md:5-9,93-102).
"""

import functools
import importlib
import importlib.util
import json
import math
import re
import sys
import threading
import weakref
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import NamedTuple, Protocol

import jax
import numpy as np

from dew.nn.kernels.generation import gpu_target
from dew.telemetry.trace import collective

_cache_counts = {'/jax/compilation_cache/cache_hits': 0, '/jax/compilation_cache/cache_misses': 0}
_cache_lock = threading.Lock()


def _cache_event(event: str, **metadata) -> None:
    if event in _cache_counts:
        with _cache_lock:
            _cache_counts[event] += 1


@functools.cache
def count_compilation_cache() -> None:
    """Count JAX's persistent-cache hits and misses from now on, once per process."""
    jax.monitoring.register_event_listener(_cache_event)


def compilation_cache_counts() -> dict[str, float]:
    """Cumulative JAX persistent-cache events since `count_compilation_cache` first ran.

    The trainer starts counting when a fit begins. Counts include every jitted
    program, not just training steps. JAX emits cache_misses when it writes a
    cache entry, so compilations below its write thresholds do not count as
    misses. An unavailable or disabled persistent cache emits neither event.
    """
    with _cache_lock:
        return {f'train/{event.rsplit("/", 1)[-1]}': float(count)
                for event, count in _cache_counts.items()}

# A multiply-add is two FLOPs.
_MAC = 2
_INTEGER_TYPE = re.compile(r'([su])(8|16|32|64)$')
"""An integer element type's sign and width, as HLO names its types (`s32`, `u8`)."""
_BITS = re.compile(r'(\d+)')

# The one instruction a row of XProf's hlo_stats displays (`hlo_op_expression`, cut short): XProf's display of
# it, no module, so read for its opcode and custom-call target alone (`reported_instruction`).
_REPORTED = re.compile(r'^\s*(?:ROOT\s+)?%?[\w.\-]+\s+=\s+.*?\s(?P<op>[a-z][a-z0-9\-]*)\(')
_REPORTED_TARGET = re.compile(r'custom_call_target="([^"]+)"')


def _element(shape) -> str:
    """An array shape's element type as HLO names it (`f32`, `bf16`, `pred`), read off XLA's schema's
    own enum (`xla.PrimitiveType`); empty for a tuple's."""
    if shape.tuple_shapes:
        return ''
    return shape.DESCRIPTOR.fields_by_name['element_type'].enum_type.values_by_number[
        shape.element_type].name.lower()


def _arrays(shape) -> tuple[tuple[str, tuple[int, ...], int], ...]:
    """Each array a result holds, a tuple's every element in order: its element type (`_element`), its
    dimensions and its bytes as XLA sizes them (`ShapeUtil::ByteSizeOfElements`): its layout's element
    size where it packs one, else its type's width rounded up to whole bytes (`pred` and `s4` one byte
    each), by its elements, rounded up."""
    if shape.tuple_shapes:
        return tuple(array for element in shape.tuple_shapes for array in _arrays(element))
    element, dims = _element(shape), tuple(shape.dimensions)
    width = 1 if element == 'pred' else int(next(iter(_BITS.findall(element)), 0))
    bits = shape.layout.element_size_in_bits or -(-width // 8) * 8
    return ((element, dims, -(-bits * math.prod(dims) // 8)),)


def _groups(proto) -> tuple[tuple[int, ...], ...]:
    """A collective's device groups, each the positions in the module's device assignment that take part
    together, as XLA's schema states them: listed (`collective_device_list`, or the older
    `replica_groups`), as an iota (`iota_collective_device_list`: the positions laid out, transposed and
    cut into groups), over a mesh's named axes (`mesh_axes_replica_group_list`), or a
    collective-permute's `source_target_pairs`, a pair a group. Empty where every device takes part
    together."""
    if proto.source_target_pairs:
        return tuple((pair.source, pair.target) for pair in proto.source_target_pairs)
    form = proto.WhichOneof('replica_group_list')
    if form == 'iota_collective_device_list':
        iota = proto.iota_collective_device_list
        count = iota.num_replica_groups * iota.num_devices_per_group
        positions = np.arange(count).reshape(list(iota.iota_reshape_dims) or [count])
        if iota.iota_transpose_perm:
            positions = positions.transpose(list(iota.iota_transpose_perm))
        return tuple(map(tuple, positions.reshape(iota.num_replica_groups,
                                                  iota.num_devices_per_group).tolist()))
    if form == 'mesh_axes_replica_group_list':
        return _mesh_groups(proto.mesh_axes_replica_group_list)
    listed = (proto.collective_device_list.replica_groups if form == 'collective_device_list'
              else proto.replica_groups)
    return tuple(tuple(group.replica_ids) for group in listed)


def _mesh_groups(groups) -> tuple[tuple[int, ...], ...]:
    """The device groups of a collective over a mesh's axes (`xla.MeshAxesReplicaGroupListProto`): the
    mesh's positions (its device ids, its iota or its order), each axis cut where a sub-axis it takes
    part over splits it (a sub-axis `(m)k` of an axis of size `n` the middle of its factors `m`, `k`,
    `n/(mk)`), and the positions that differ over the axes it takes part over alone grouped together."""
    mesh = groups.mesh
    sizes = [axis.size for axis in mesh.axes]
    count = math.prod(sizes)
    if mesh.device_ids:
        positions = np.asarray(mesh.device_ids)
    elif mesh.HasField('iota_transform'):
        transform = mesh.iota_transform
        positions = np.arange(count).reshape(list(transform.reshape_dims))
        positions = positions.transpose(list(transform.transpose_perm)).ravel()
    else:
        positions = np.arange(count)
    factors: list[int] = []
    taken: list[bool] = []
    for index, size in enumerate(sizes):
        refs = [ref for ref in groups.axes if ref.mesh_axis_index == index]
        cuts = sorted((ref.sub_axis_info.pre_size, ref.sub_axis_info.size) for ref in refs
                      if ref.HasField('sub_axis_info'))
        if not cuts:
            factors.append(size)
            taken.append(bool(refs))
            continue
        reached = 1
        for before, cut in cuts:
            if before > reached:
                factors.append(before // reached)
                taken.append(False)
            factors.append(cut)
            taken.append(True)
            reached = before * cut
        if size > reached:
            factors.append(size // reached)
            taken.append(False)
    moved = [index for index, take in enumerate(taken) if take]
    kept = [index for index, take in enumerate(taken) if not take]
    laid = positions.reshape(factors).transpose(kept + moved)
    return tuple(map(tuple, laid.reshape(-1, math.prod(factors[index] for index in moved)).tolist()))


class Attention(Protocol):
    """What a fused attention's call states of its operands, as a FlashAttention wheel's targets state it
    (`dew_flash_attn.TARGETS`): where its query sits among them (a backward's first is the output's
    gradient), whether it runs the backward, and whether its sequences lie between cumulative offsets
    given at run time (`packed`), whose pairs no shape states."""

    @property
    def query(self) -> int: ...

    @property
    def backward(self) -> bool: ...

    @property
    def packed(self) -> bool: ...


@dataclass(frozen=True)
class _Fused:
    """cuDNN's fused attention, as XLA calls it: its query, key and value first, batched `[B, S, H, D]`."""

    backward: bool
    query: int = 0
    packed: bool = False


@dataclass(frozen=True)
class _Call:
    """What one custom-call target computes (`custom_call_kind`), with the facts its reader needs: a fused
    attention's operands, a cuDNN convolution's direction (`forward`, `input`, `filter`: which gradient,
    if any, it computes)."""

    kind: str
    attention: Attention | None = None
    direction: str = "forward"


_XLA_CALLS: dict[str, _Call] = {
    **{target: _Call("gemm") for target in (
        "__cublas$gemm", "__cublas$lt$matmul", "__cublas$lt$matmul$f8", "__cublas$lt$matmul$mx")},
    **{target: _Call("convolution", direction=direction) for target, direction in (
        ("__cudnn$convForward", "forward"), ("__cudnn$convBiasActivationForward", "forward"),
        ("__cudnn$convForwardGraph", "forward"), ("__cudnn$convBackwardInput", "input"),
        ("__cudnn$convBackwardFilter", "filter"))},
    **{target: _Call("attention", _Fused(backward)) for target, backward in (
        ("__cudnn$fmhaSoftmax", False), ("__cudnn$fmhaSoftmaxF8", False),
        ("__cudnn$fmhaSoftmaxDropout", False),
        ("__cudnn$fmhaScaleBiasSoftmax", False), ("__cudnn$fmhaScaleBiasSoftmaxDropout", False),
        ("__cudnn$fmhaSoftmaxBackward", True), ("__cudnn$fmhaSoftmaxBackwardF8", True),
        ("__cudnn$fmhaSoftmaxDropoutBackward", True), ("__cudnn$fmhaScaleBiasSoftmaxBackward", True),
        ("__cudnn$fmhaScaleBiasSoftmaxDropoutBackward", True))},
    **{target: _Call("no_matmul") for target in (
        "__cudnn$norm", "__cudnn$convReorderFilter", "__cudnn$convReorderFilterAndBias",
        "xla.gpu.ext.cub_sort_unassigned_scratch_size", "xla.gpu.ext.cub_sort_pairs",
        "xla.gpu.ext.cub_sort_keys", "xla.gpu.ext.cub_scan_unassigned_scratch_size", "xla.gpu.ext.cub_scan",
        "Sharding", "SPMDFullToShardShape", "SPMDShardToFullShape", "annotate_device_placement",
        "MoveToHost", "MoveToDevice", "PinToDevice", "TopK",
        "xla.sdy.FuncResultSharding", "xla.sdy.ShardingGroup", "xla.sdy.PropagationBarrier",
        "xla.sdy.GlobalToLocalShape", "xla.sdy.LocalToGlobalShape")},
}
"""The custom-call targets XLA states, each by what it computes. XLA names them in C++ alone, so they are
read from its source at d867969800d4 (the revision jaxlib 0.12.0.1 builds): xla/service/gpu/cublas_cudnn.h's
cuBLAS, cuDNN and CUB targets, xla/service/memory_annotations.h's placements,
xla/service/spmd/shardy/constants.h's sharding marks and Shardy's round trip, xla/service/topk_rewriter.cc's
"TopK". JAX's own targets JAX states (`jax.extend.custom_call_targets`), which `_calls` reads. A target
this table and its owners' (`_calls`) do not hold, a cuBLAS triangular solve or grouped matmul and cuDNN's
block-scaled dot among them, is one nothing here sizes. A fused attention's backward reads its query first;
an F8 attention's form is its own target."""


@functools.cache
def _calls() -> dict[str, _Call]:
    """Every custom-call target this count knows, by what it computes: XLA's (`_XLA_CALLS`), JAX's as
    JAX states them (`jax.extend.custom_call_targets`: its Pallas kernels', its random bits', its
    callbacks' and `lax.empty`'s) and those of Dew's kernel wheels where installed, each stating its own
    (`dew_cutlass.TARGETS`, `dew_flash_attn.TARGETS`); a wheel this process lacks states none, and its
    calls go unsized."""
    from jax.extend import custom_call_targets as jax_targets

    calls = {**_XLA_CALLS, jax_targets.TRITON: _Call("triton"), jax_targets.MOSAIC_GPU: _Call("mosaic_gpu"),
             jax_targets.TPU_CUSTOM_CALL: _Call("mosaic_tpu"),
             **{target: _Call("no_matmul") for target in (
                 jax_targets.ALLOCATE_BUFFER, *jax_targets.THREEFRY2X32.values(),
                 *jax_targets.PYTHON_CALLBACK.values(), *jax_targets.BUFFER_CALLBACK.values())}}
    if importlib.util.find_spec("dew_cutlass") is not None:
        cutlass = importlib.import_module("dew_cutlass")
        calls.update({target: _Call("grouped_gemm") for target in cutlass.TARGETS})
    if importlib.util.find_spec("dew_flash_attn") is not None:
        targets: Mapping[str, Attention] = importlib.import_module("dew_flash_attn").TARGETS
        calls.update({target: _Call("packed_attention" if stated.packed else "attention", stated)
                      for target, stated in targets.items()})
    return calls


PALLAS = ("triton", "mosaic_gpu", "mosaic_tpu")
"""The kinds of custom call a Pallas kernel lowers to, whose work is the cost the kernel states."""

_STATED = re.compile(r'cost_estimate_json\s*=\s*"((?:[^"\\]|\\.)*)"')
"""A Mosaic GPU kernel's own cost as its call's backend config states it (`cost_estimate_json`), its quotes
and other bytes escaped as an MLIR string writes them (`\\22`)."""

_STATED_TPU = re.compile(r'cost_estimate\\?"\s*:\s*\{[^}]*?\bflops\\?"\s*:\s*(\d+)')
"""A Mosaic TPU kernel's own FLOPs as its call's JSON config states them (`cost_estimate`)."""


@dataclass(frozen=True)
class _Convolution:
    """A convolution's dimension numbers and window as XLA's schema states them, an HLO convolution's and a
    cuDNN convolution call's alike: where its input's and its output's features lie, its window's size and
    its input (base) dilation along each spatial axis, and its feature groups."""

    input_features: int
    output_features: int
    window: tuple[int, ...]
    dilation: tuple[int, ...]
    groups: int


@dataclass(frozen=True)
class _Instruction:
    """The parts of one HLO instruction the counts read, as XLA's schema states them
    (`jax.extend.xla.Instruction`)."""

    op: str
    dims: tuple[int, ...]
    """Its result's dimensions: a tuple's first array's."""
    operands: tuple[str, ...]
    called: tuple[str, ...] = ()
    """The computations it calls, in XLA's order: a while loop's body then its condition, a fusion's body,
    a conditional's branches."""
    target: str = ''
    """A custom call's target, empty for any other op."""
    config: str = ''
    """Its backend config as it is stored: a library call's JSON, a while loop's trip count, a kernel's
    stated cost and attributes."""
    frontend: Mapping[str, str] = field(default_factory=dict)
    """Its frontend attributes (`MODEL_FLOPS`)."""
    contracting: tuple[int, ...] = ()
    """A dot's left operand's contracting dimensions."""
    convolution: _Convolution | None = None
    direction: str = ''
    """A compare's direction (`LT`)."""
    tuple_index: int = 0
    """The element a get-tuple-element reads."""
    parameter: int = 0
    """A parameter's number."""
    recomputed: bool = False
    """Whether it recomputes the forward: JAX's remat names its recomputation
    `rematted_computation`, and XLA's own rematerialization clones an
    instruction as `<name>.remat`."""
    traced: tuple[str, ...] = ()
    """Its metadata's `op_name source_file:line`, interned: a module repeats
    each a great many times."""
    buffers: tuple[int, ...] = ()
    """A collective's or a custom call's result, each array's bytes (`_arrays`); empty for any other op."""
    shapes: tuple[tuple[int, ...], ...] = ()
    """A collective's or a custom call's result, each array's dimensions, as `buffers` sizes them; empty
    for any other op."""
    groups: tuple[tuple[int, ...], ...] = ()
    """A collective's device groups (`_groups`); empty for any other op."""
    arrays: tuple[str, ...] = ()
    """A collective's or a custom call's result, each array's element type, as `shapes` holds its
    dimensions: a cuBLAS call's `s8` workspace beside its product; empty for any other op."""


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


def _module(source: str | Iterable[str] | jax.stages.Compiled):
    """The HLO module `source` holds: a compiled executable's optimized one (its first, as JAX's own
    `Compiled.as_text` prints it), or one written as text (an XLA dump, whole or as its lines; a module a
    test writes), read by XLA's own parser (`jax.extend.xla.module_from_text`). None for an executable
    that holds no module."""
    from jax.extend import xla

    if isinstance(source, jax.stages.Compiled):
        executable = source.runtime_executable()
        modules = [] if executable is None else executable.hlo_modules()
        return modules[0] if modules else None
    return xla.module_from_text(source if isinstance(source, str) else '\n'.join(source))


def _integer(literal, element: str) -> int | None:
    """The integer a scalar constant of `element` type holds (`xla.LiteralProto`), None for any other."""
    kind = _INTEGER_TYPE.match(element)
    if kind is None:
        return None
    sign, bits = kind.groups()
    values = getattr(literal, f'{sign}{bits}s')
    if isinstance(values, bytes):
        values = np.frombuffer(values, f"<{'i' if sign == 's' else 'u'}{int(bits) // 8}").tolist()
    return int(values[0]) if len(values) == 1 else None


def _read(source: str | Iterable[str] | jax.stages.Compiled) -> tuple[dict[str, _Computation], str | None]:
    """The module's computations, and the name of its entry computation (`_module`), each instruction
    read through XLA's own schema (`jax.extend.xla.instructions`), keeping the facts the counts read: of
    its metadata, which is most of a module, only its traced name. No computations for an executable that
    holds no module."""
    from jax.extend import xla

    module = _module(source)
    computations: dict[str, _Computation] = {}
    entry = None
    if module is None:
        return computations, entry
    for read in xla.instructions(module):
        proto = read.proto
        computation = computations.setdefault(read.computation, _Computation())
        if read.entry:
            entry = read.computation
        arrays = _arrays(proto.shape)
        dims = arrays[0][1] if arrays else ()
        element = _element(proto.shape)
        computation.shapes[read.name] = dims
        computation.elements[read.name] = element
        metadata = proto.metadata
        traced = ((sys.intern(f"{metadata.op_name} {metadata.source_file}:{metadata.source_line}"
                              if metadata.source_file else metadata.op_name),) if metadata.op_name else ())
        moved = collective(read.opcode)
        sized = moved or read.opcode == 'custom-call'
        convolution = None
        if proto.HasField('convolution_dimension_numbers'):
            numbers = proto.convolution_dimension_numbers
            dimensions = proto.window.dimensions
            convolution = _Convolution(numbers.input_feature_dimension, numbers.output_feature_dimension,
                                       tuple(each.size for each in dimensions),
                                       tuple(each.base_dilation or 1 for each in dimensions),
                                       proto.feature_group_count or 1)
        computation.instructions[read.name] = _Instruction(
            read.opcode, dims, read.operands, read.called, read.custom_call_target or '',
            proto.backend_config.decode('latin-1'), dict(proto.frontend_attributes.map),
            tuple(proto.dot_dimension_numbers.lhs_contracting_dimensions), convolution,
            proto.comparison_direction, proto.tuple_index, proto.parameter_number,
            ('.remat' in read.name or '.remat' in read.computation
             or 'rematted_computation' in metadata.op_name), traced,
            tuple(size for _, _, size in arrays) if sized else (),
            tuple(each for _, each, _ in arrays) if sized else (),
            _groups(proto) if moved else (), tuple(each for each, _, _ in arrays) if sized else ())
        if read.root:
            computation.root = read.name
        if read.opcode == 'constant' and not dims:
            value = _integer(proto.literal, element)
            if value is not None:
                computation.constants[read.name] = value
    return computations, entry


def _dot_flops(instruction: _Instruction, shapes: dict[str, tuple[int, ...]],
               contracting: tuple[int, ...]) -> float | None:
    """Every output element costs one multiply-add per contracted element; None where the module states
    no shape for its first operand."""
    lhs = shapes.get(instruction.operands[0]) if instruction.operands else None
    if lhs is None:
        return None
    contracted = math.prod(lhs[axis] for axis in contracting if axis < len(lhs))
    return _MAC * math.prod(instruction.dims) * contracted


def _convolution_flops(instruction: _Instruction,
                       shapes: dict[str, tuple[int, ...]]) -> float | None:
    """Output elements times the kernel window times the input features; None where the module states
    no shape for its input.

    Dividing by the input dilation counts a strided convolution's gradient at
    the multiply-adds its forward pass costs: the dilated positions are zeros,
    and no kernel multiplies them.
    """
    lhs = shapes.get(instruction.operands[0]) if instruction.operands else None
    convolution = instruction.convolution
    if lhs is None or convolution is None:
        return None
    macs = (math.prod(instruction.dims) * math.prod(convolution.window) * lhs[convolution.input_features]
            / convolution.groups / math.prod(convolution.dilation))
    return _MAC * macs


def _grouped_gemm_flops(instruction: _Instruction, shapes: dict[str, tuple[int, ...]]) -> float | None:
    """A CUTLASS grouped GEMM's call (`dew_cutlass_grouped_gemm`, and `_k`'s weight gradient): each row of
    its first operand, the tokens, multiplies its features into the result's last axis once, whichever
    group holds it; None where the module states no shape for the tokens."""
    tokens = shapes.get(instruction.operands[0]) if instruction.operands else None
    return None if tokens is None else _MAC * math.prod(tokens) * instruction.dims[-1]


def _cudnn_convolution_flops(direction: str, instruction: _Instruction,
                             shapes: dict[str, tuple[int, ...]]) -> float | None:
    """A cuDNN convolution call computing `direction` (`_Call.direction`), at the multiply-adds of its
    forward shape; None where the module states no shape for an operand it reads.

    XLA keeps the forward convolution's window and dimension numbers on all three
    kinds, so each reduces to the same product `B Ho Wo O Kh Kw I / groups`
    read off whichever operand holds each factor: the forward call takes the
    input and the filter, the input-gradient call the output gradient and the
    filter, the filter-gradient call the input and the output gradient.
    """
    lhs = shapes.get(instruction.operands[0]) if instruction.operands else None
    rhs = shapes.get(instruction.operands[1]) if len(instruction.operands) > 1 else None
    convolution = instruction.convolution
    if lhs is None or rhs is None or convolution is None:
        return None
    kernel = math.prod(convolution.window)
    dims = instruction.dims
    if direction == 'input':
        macs = math.prod(lhs) * kernel * dims[convolution.input_features]
    elif direction == 'filter':
        macs = math.prod(dims) * math.prod(rhs) / rhs[convolution.output_features]
    else:
        macs = math.prod(dims) * kernel * lhs[convolution.input_features]
    return _MAC * macs / convolution.groups


_CAUSAL = re.compile(r'is_causal = true|"mask_type":"(?:PADDING_)?CAUSAL"')
_SIDES = re.compile(r'window_size_(left|right) = (-?\d+)')
_SLIDING = re.compile(r'"sliding_window_length":(\d+)')


def _reached(queries: int, keys: int, config: str) -> int:
    """How many (query, key) pairs a fused attention computes: each query row
    reaches the keys its call's mask leaves it, under a causal mask the keys up
    to its own position (aligned to the last key, as FlashAttention aligns
    them) and within a sliding window the ones it spans, left of the query
    and right of it. dew_flash_attn states `is_causal` and
    `window_size_left`/`_right` (-1 unbounded); cuDNN its `mask_type` and a
    `sliding_window_length` of the window's left side and the query's own
    key (0 none). Each states its mask in its call's backend config (`config`), cuDNN's as JSON and the
    FFI call's as its attributes' MLIR dictionary, which no schema of XLA's decodes."""
    causal = bool(_CAUSAL.search(config))
    sides = {side: int(size) for side, size in _SIDES.findall(config)}
    sliding = _SLIDING.search(config)
    left = sides.get('left', int(sliding.group(1)) - 1 if sliding and int(sliding.group(1)) else -1)
    right = 0 if causal else sides.get('right', -1)
    diagonal = np.arange(queries) + (keys - queries)
    low = np.maximum(diagonal - left, 0) if left >= 0 else np.zeros(queries, np.int64)
    high = np.minimum(diagonal + right, keys - 1) if right >= 0 else np.full(queries, keys - 1)
    return int(np.clip(high - low + 1, 0, None).sum())


def _fused_attention_flops(instruction: _Instruction, shapes: dict[str, tuple[int, ...]],
                           attention: Attention) -> tuple[float, float] | None:
    """A fused-attention call (cuDNN's or dew_flash_attn's) as its target states its operands
    (`Attention`), from the query, key and value it is given, the operands at its `query` and the two
    after it, and the pairs of them its mask computes (`_reached`); None where they are not batched
    `[batch, sequence, heads, width]` arrays the module states.

    The kernel keeps the scores off memory, so nothing in the module states
    its arithmetic; the operands do. They are `[batch, sequence, heads, width]`
    (checked against the calls XLA emits for `jax.nn.dot_product_attention`,
    including grouped-query and cross attention, where the key carries its own
    head count and sequence, and against dew_flash_attn's). The forward pass's
    `Q K^T` costs one multiply-add per pair, head and query width, and its
    `P V` one per pair, head and value width, which a model may set apart
    (MiMo's 192 and 128). The backward pass runs each product's two
    transposes: dP and dV at the value's width, dQ and dK at the query's. A
    causal mask leaves the kernels about half the square, which is what they
    compute: on an H100, FlashAttention-3's causal call at 2 x 16 heads x 4096
    x 128 counted the whole square, twice its work, before this read the mask.
    The model's work and the hardware's: the backward recomputes the scores
    it kept off memory, one `Q K^T` more at the query's width, the hardware's
    alone.
    """
    query = attention.query
    if len(instruction.operands) < query + 3:
        return None
    queries, key, value = (shapes.get(instruction.operands[query + offset]) for offset in range(3))
    if queries is None or key is None or value is None:
        return None
    if any(len(each) != 4 for each in (queries, key, value)):
        return None
    batch, length, heads, width = queries
    pairs = _MAC * batch * heads * _reached(length, key[1], instruction.config)
    model = (2 if attention.backward else 1) * pairs * (width + value[3])
    return model, model + (pairs * width if attention.backward else 0.0)


def custom_call_kind(target: str) -> str | None:
    """What a custom call computes, by its target (`_calls`, each name read from the owner that states
    it): cuBLAS's GEMMs ('gemm'), cuDNN's convolutions ('convolution'), the fused attentions of cuDNN and
    dew_flash_attn ('attention', and 'packed_attention' for a call over sequences between offsets given
    at run time, whose work no shape states), dew_cutlass's grouped GEMMs ('grouped_gemm'), the kernels
    Pallas lowers to (`PALLAS`), and the calls that run no matmul (sharding and placement marks, host
    copies and callbacks, random bits, sorts, scans and top-k: 'no_matmul'); None for any other, whose
    work nothing here sizes (`sizes`). The FLOP count, the kernel report (`dew.training._lowering`) and
    tools/attribute_step.py read custom calls through this one table."""
    call = _calls().get(target)
    return None if call is None else call.kind


@dataclass(frozen=True)
class HloInstruction:
    """One instruction of an optimized HLO module, as a profile attributes it."""

    op: str
    target: str
    """A custom call's target, empty for any other op."""
    operands: tuple[str, ...]
    """The names of the instructions it reads."""
    fused: frozenset[str]
    """The ops of the computations it calls, theirs included: a fusion's body."""
    recomputed: bool
    traced: frozenset[str]
    """Where JAX traced it and what it fused: `op_name source_file:line` of
    its own metadata and of each op in the computations it calls. A
    multi-output fusion carries none of its own, its ops do."""
    reads: frozenset[str]
    """The traced names of the parameters it reads, through bitcasts, copies
    and tuple elements: a jitted step's own are its arguments' paths, such
    as `current.opt_state[0].mu['head']`."""
    flops: float | None
    """The matmul and convolution work of one call, a fusion's included, as
    the step's count (`hlo_flops`) counts it; None where any of it is unsized
    (`StepFlops.unsized`)."""
    runs: float
    """Its calls per call of the module, as `hlo_flops` weighs them: a loop
    body's trip count, every branch of a conditional once."""
    buffers: tuple[int, ...]
    """A collective's or a custom call's result, each array's bytes: an async pair's `-done` holds
    what moved, its `-start` what it read as well. Empty for any other op."""
    shapes: tuple[tuple[int, ...], ...]
    """A collective's or a custom call's result, each array's dimensions, as `buffers` sizes them."""
    groups: tuple[tuple[int, ...], ...]
    """A collective's device groups, each the positions in the module's device assignment that
    take part together; empty where every device does, and for any other op (`_groups`)."""
    element: str
    """Its result's element type as HLO names it (`f32`, `bf16`); empty where its result is a tuple."""
    dims: tuple[int, ...]
    """Its result's dimensions: a tuple's first array's."""
    arrays: tuple[str, ...] = ()
    """A collective's or a custom call's result, each array's element type, as `shapes` holds its
    dimensions: a cuBLAS call's `s8` workspace beside its product."""


class Reported(NamedTuple):
    """What a row of XProf's hlo_stats displays of its instruction (`reported_instruction`)."""

    op: str
    target: str
    """A custom call's target, empty for any other op."""


def reported_instruction(expression: str) -> Reported | None:
    """The opcode and custom-call target of the one instruction `expression` displays, as a row of XProf's
    hlo_stats holds it (`hlo_op_expression`, cut short): XProf's display, no module XLA's parser reads, for
    a row whose module was not given (`hlo_instructions` reads one that was); None where it displays
    none."""
    found = _REPORTED.match(expression)
    if found is None:
        return None
    target = _REPORTED_TARGET.search(expression) if found.group('op') == 'custom-call' else None
    return Reported(found.group('op'), target.group(1) if target else '')


def hlo_instructions(source: str | Iterable[str] | jax.stages.Compiled) -> dict[str, HloInstruction]:
    """Every instruction of the module `source` holds (`_module`: a compiled executable's, or an XLA dump's
    after-optimizations HLO, whole or as its lines) by name, read as the FLOP count reads it
    (`hlo_flops`). Each computation's body is gathered once, so the fusions that call one share its
    sets."""
    computations, entry = _read(source)
    weights = {} if entry is None else _weights(computations, entry)
    bodies: dict[str, tuple[frozenset[str], frozenset[str], float | None]] = {}

    def fused(name: str,
              seen: frozenset[str] = frozenset()) -> tuple[frozenset[str], frozenset[str], float | None]:
        if name in bodies:
            return bodies[name]
        computation = computations.get(name)
        if computation is None or name in seen:
            return frozenset(), frozenset(), 0.0
        ops = {instruction.op for instruction in computation.instructions.values()}
        names = {each for instruction in computation.instructions.values() for each in instruction.traced}
        flops = _summed(_hardware(_instruction_flops(instruction, computation.shapes))
                        for instruction in computation.instructions.values())
        for instruction in computation.instructions.values():
            for called in instruction.called:
                inner_ops, inner_names, inner_flops = fused(called, seen | {name})
                ops |= inner_ops
                names |= inner_names
                flops = _summed((flops, inner_flops))
        bodies[name] = frozenset(ops), frozenset(names), flops
        return bodies[name]

    def parameters(computation: _Computation, operand: str) -> frozenset[str]:
        instruction = computation.instructions.get(operand)
        if instruction is None:
            return frozenset()
        if instruction.op == 'parameter':
            return frozenset(instruction.traced)
        if instruction.op in ('bitcast', 'copy', 'get-tuple-element'):
            return frozenset().union(*(parameters(computation, each) for each in instruction.operands))
        return frozenset()

    found = {}
    for called_as, computation in computations.items():
        for name, instruction in computation.instructions.items():
            calls = instruction.called if instruction.op == 'fusion' else ()
            called = [fused(each) for each in calls]
            ops = called[0][0] if len(called) == 1 else frozenset().union(*(each for each, _, _ in called))
            names = frozenset(instruction.traced).union(*(each for _, each, _ in called))
            reads = frozenset().union(*(parameters(computation, each) for each in instruction.operands))
            flops = _summed([_hardware(_instruction_flops(instruction, computation.shapes)),
                             *(each for _, _, each in called)])
            found[name] = HloInstruction(instruction.op, instruction.target,
                                         tuple(instruction.operands), ops, instruction.recomputed, names,
                                         reads, flops, weights.get(called_as, 0.0), instruction.buffers,
                                         instruction.shapes, instruction.groups, computation.elements[name],
                                         instruction.dims, instruction.arrays)
    computations.clear()
    return found


def _hardware(flops: tuple[float, float] | None) -> float | None:
    """The hardware's share of an instruction's work (`_instruction_flops`), None where it is unsized."""
    return None if flops is None else flops[1]


def _summed(parts: Iterable[float | None]) -> float | None:
    """The sum of `parts`, None where any is: work one part leaves unsized leaves the whole unsized."""
    parts = list(parts)
    return None if any(part is None for part in parts) else sum(part for part in parts if part is not None)


def _instruction_flops(instruction: _Instruction,
                       shapes: dict[str, tuple[int, ...]]) -> tuple[float, float] | None:
    """The multiply-add work of one instruction, the model's and the hardware's (a kernel's recomputed
    products the hardware's alone), zero for anything that runs no matmul, and None for one whose work the
    count cannot size (`sizes`): a custom call nothing here knows (`custom_call_kind`), a Pallas kernel that
    states no cost, an attention over sequences its shapes do not state, or a matmul whose operands the
    module states no shape for. Missing work stays missing, never counted as zero.

    Only the matmuls and convolutions count. Everything the compiler leaves
    elementwise (the optimizer, the EMA, normalization, the softmax, the loss
    reductions) is memory-bound work that a FLOP utilisation figure is not
    about, which is the convention this repo's benchmarks state
    (docs/research/benchmark-parity.md:44-48).
    """
    if instruction.op != 'custom-call':
        flops = _matmul_flops(instruction, shapes)
        return None if flops is None else (flops, flops)
    call = _calls().get(instruction.target)
    if call is None:
        return None
    if call.kind in PALLAS:
        # A Pallas kernel's work is the hardware's it states, the model's share where it declares one
        # (`MODEL_FLOPS`).
        hardware = _stated_flops(instruction.config)
        if hardware is None:
            return None
        declared = instruction.frontend.get(MODEL_FLOPS)
        return (hardware if declared is None else float(declared)), hardware
    if call.kind == 'no_matmul':
        return 0.0, 0.0
    if call.kind == 'attention' and call.attention is not None:
        return _fused_attention_flops(instruction, shapes, call.attention)
    flops = _library_flops(call, instruction, shapes)
    return None if flops is None else (flops, flops)


MODEL_FLOPS = "dew_model_flops"
"""The frontend attribute a kernel's launch declares the model's share of the work it states by (`declared`):
the rest its own recomputation (attention's scores, `attention_mosaic_program.stated`)."""

def _matmul_flops(instruction: _Instruction, shapes: dict[str, tuple[int, ...]]) -> float | None:
    """A `dot`'s or a `convolution`'s multiply-add work, None where the module states too little to size
    it; zero for any other instruction but a custom call (`_instruction_flops`)."""
    if instruction.op == 'dot':
        return _dot_flops(instruction, shapes, instruction.contracting)
    if instruction.op == 'convolution':
        return _convolution_flops(instruction, shapes)
    return 0.0


def _library_flops(call: _Call, instruction: _Instruction,
                   shapes: dict[str, tuple[int, ...]]) -> float | None:
    """A GEMM library call's multiply-add work, a cuBLAS GEMM's, a cuDNN convolution's or a grouped GEMM's;
    None for any other call, and where the module states too little to size it."""
    if call.kind == 'gemm':
        # cuBLAS's call states its dot's dimension numbers in its JSON config (`xla.gpu.GemmBackendConfig`).
        numbers = json.loads(instruction.config or '{}').get('gemm_backend_config', {}).get(
            'dot_dimension_numbers', {})
        return _dot_flops(instruction, shapes,
                          tuple(int(axis) for axis in numbers.get('lhs_contracting_dimensions', ())))
    if call.kind == 'convolution':
        return _cudnn_convolution_flops(call.direction, instruction, shapes)
    if call.kind == 'grouped_gemm':
        return _grouped_gemm_flops(instruction, shapes)
    return None


def _stated_flops(config: str) -> float | None:
    """The FLOPs a Pallas kernel's call states in its backend config (`_STATED`, `_STATED_TPU`), None where
    it states none."""
    if found := _STATED_TPU.search(config):
        return float(found.group(1))
    found = _STATED.search(config)
    if found is None:
        return None
    text = re.sub(r'\\([0-9A-Fa-f]{2})', lambda escaped: chr(int(escaped.group(1), 16)), found.group(1))
    flops = json.loads(text.replace('\\"', '"')).get('flops')
    return None if flops is None else float(flops)


def sizes(kind: str | None, stated: float | None) -> bool:
    """Whether the FLOP count sizes the work of a custom call of `kind` (`custom_call_kind`) whose kernel
    states `stated` FLOPs: a library's GEMM, convolution or batched attention by its shapes, a Pallas
    kernel by the cost it states, a call that runs no matmul as none; never a packed attention, whose
    sequences its shapes do not state, nor a call nothing here knows."""
    return (kind in ("gemm", "convolution", "attention", "grouped_gemm", "no_matmul")
            or (kind in PALLAS and stated is not None))


def _call_counts(instruction: _Instruction, caller: _Computation,
                 computations: dict[str, _Computation]) -> dict[str, float]:
    """The computations this instruction runs, and how often it runs each.

    A loop body runs once per iteration. The CPU backend states the count as
    `known_trip_count`; a v6e executable states none, so a loop without it is
    read off its counter (`_counted_trips`). A loop whose count neither gives
    becomes infinite, and the caller then reports no count. A conditional
    runs one of its branches, so counting every branch bounds it.
    """
    counts = dict.fromkeys(instruction.called, 1.0)
    if instruction.op == 'while':
        trip = json.loads(instruction.config or '{}').get('known_trip_count', {}).get('n')
        counts[_body(instruction)] = (float(trip) if trip is not None
                                      else _counted_trips(instruction, caller, computations))
    return counts


def _counted_trips(loop: _Instruction, caller: _Computation,
                   computations: dict[str, _Computation]) -> float:
    """The trip count of a loop that counts one integer tuple element from a
    constant start, by a constant positive step, while it is below a constant
    bound: the loop every `fori_loop` and `scan` with static bounds lowers
    to, and the form XLA's own trip-count analysis reads; where that bound is
    one side of an `and`, at most its trips, a bound as every branch of a
    conditional counted is (a loop of at most one trip, the attention guards').
    Its condition and step are read through copies and through the fusions an
    executable wraps them in (`_term`). Infinite for any other loop, and for
    one whose last step overflows the counter's type, since a wrapped counter
    need never reach the bound."""
    condition = computations.get(loop.called[1])
    body = computations.get(_body(loop))
    if condition is None or body is None or not loop.operands:
        return math.inf
    root = _term(condition, condition.root, computations)
    sides = root[2] if root is not None and root[0] == 'and' else (root,)
    return min((_compared_trips(loop, caller, body, side, computations) for side in sides),
               default=math.inf)


def _compared_trips(loop: _Instruction, caller: _Computation, body: _Computation, compare: tuple | None,
                    computations: dict[str, _Computation]) -> float:
    """`_counted_trips` of `loop` by `compare` (a `_term`), its condition's counted bound, or infinite
    where it is none."""
    if compare is None or compare[0] != 'compare' or compare[1] != 'LT' or len(compare[2]) != 2:
        return math.inf
    counter, bound = compare[2]
    if counter is None or counter[0] != 'element' or bound is None or bound[0] != 'constant':
        return math.inf
    kind = _INTEGER_TYPE.match(counter[2])
    start = _constant(caller, _tuple_element(caller, loop.operands[0], counter[1]))
    step = _step(_term(body, _tuple_element(body, body.root, counter[1]), computations), counter[1])
    if kind is None or start is None or step is None:
        return math.inf
    return counted_trips(start, bound[1], step, int(kind.group(2)), signed=kind.group(1) == 's')


def _term(computation: _Computation, name: str | None, computations: dict[str, _Computation],
          bound: tuple = ()) -> tuple | None:
    """The expression that makes `name` in `computation`, through copies and through a fusion of
    elementwise ops over its operands (an executable's loop condition and counter step): `('element',
    index, type)` for an element of the computation's tuple parameter, `('constant', value)` for an
    integer constant, `(op, direction, operands)` for any other instruction (a compare's direction), its
    operands each such a term; a fused computation's parameters bound to its fusion's operands (`bound`).
    None where it is none of these."""
    made = _source(computation, name)
    if made is None:
        return None
    if made.op == 'parameter' and bound:
        # A fused computation's parameter, bound to its fusion's operand of its number.
        return bound[made.parameter] if made.parameter < len(bound) else None
    value = _constant(computation, made_name := _origin(computation, name))
    if value is not None:
        return ('constant', value)
    index = _element_index(computation, made_name)
    if index is not None:
        return ('element', index, computation.elements.get(made_name, ''))
    operands = tuple(_term(computation, operand, computations, bound) for operand in made.operands)
    if made.op == 'fusion':
        fused = computations.get(made.called[0])
        return None if fused is None else _term(fused, fused.root, computations, operands)
    return (made.op, made.direction, operands)


def _step(stepped: tuple | None, index: int) -> int | None:
    """The constant a loop's body adds to its counter, element `index` of its parameter, to make the term
    `stepped`, or None where it adds none."""
    if stepped is None or stepped[0] != 'add' or len(stepped[2]) != 2:
        return None
    for counter, step in (stepped[2], stepped[2][::-1]):
        if counter is not None and counter[0] == 'element' and counter[1] == index and step is not None \
                and step[0] == 'constant':
            return step[1]
    return None


def counted_trips(start: int, bound: int, step: int, bits: int, *, signed: bool) -> float:
    """The trips of a loop that counts an integer of `bits` bits (`signed` or not) from `start` by `step`
    while it is below `bound`: the one rule an executable's loop (`_counted_trips`) and a lowered program's
    (`dew.nn.kernels.lowered._trips`) are both read by. Infinite for a step that is not positive, and for a
    last step that overflows the counter's type, since a wrapped counter need never reach the bound."""
    if step <= 0:
        return math.inf
    trips = max(0, -(-(bound - start) // step))
    if start + trips * step > (1 << (bits - signed)) - 1:
        return math.inf
    return float(trips)


def _body(loop: _Instruction) -> str:
    """A while loop's body, the first computation it calls (XLA's `kBodyComputationIndex`; its condition
    the second, `kConditionComputationIndex`, xla/hlo/ir/hlo_instruction.h)."""
    return loop.called[0]


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
    return read.tuple_index if tuple_ is not None and tuple_.op == 'parameter' else None


def _tuple_element(computation: _Computation, name: str | None, index: int) -> str | None:
    """The operand at `index` of the tuple `name` builds, or None."""
    built = _source(computation, name)
    if built is None or built.op != 'tuple' or index >= len(built.operands):
        return None
    return built.operands[index]


def _weights(computations: dict[str, _Computation], entry: str) -> dict[str, float]:
    """How many times each computation runs per call of the entry computation (`call_weights`)."""
    return call_weights({name: _merged_call_counts(computation, computations)
                         for name, computation in computations.items()}, entry)


def call_weights(calls: Mapping[str, Mapping[str, float]], entry: str) -> dict[str, float]:
    """How many times each function of a program runs per call of its `entry`, given how often each runs
    each function it calls (`calls`), an HLO computation's or a StableHLO function's alike.

    Neither recurses, so the call graph is a DAG and one pass in topological order gives every function
    the sum of its callers' counts. Anything the entry cannot reach keeps a count of zero.
    """
    callers = dict.fromkeys(calls, 0)
    for callees in calls.values():
        for callee in callees:
            if callee in callers:
                callers[callee] += 1
    weights = dict.fromkeys(calls, 0.0)
    weights[entry] = 1.0
    ready = [name for name, count in callers.items() if count == 0]
    while ready:
        name = ready.pop()
        for callee, times in calls[name].items():
            if callee not in weights:
                continue
            # A call that runs none adds none: a function that runs none, however many times its loops
            # would, or a call in a loop of no trips, however often its caller runs.
            weights[callee] += weights[name] * times if weights[name] and times else 0.0
            callers[callee] -= 1
            if callers[callee] == 0:
                ready.append(callee)
    return weights


def _branches(instruction: _Instruction) -> list[str]:
    """The computations a conditional `instruction` runs one of, its branches; none for another
    instruction."""
    return list(instruction.called) if instruction.op == 'conditional' else []


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
    the first and HFU off the second (PaLM's distinction). `unsized` counts
    the instructions the step runs whose work neither holds (`_instruction_flops`):
    a Pallas kernel that states no cost (JAX's Triton lowering drops the
    kernel's `CostEstimate`; Mosaic's keep theirs), an attention over
    sequences its shapes do not state (`packed_attention`), or a call nothing
    here knows, so where any runs both are lower bounds."""

    model: float
    hardware: float
    unsized: float = 0.0

    def __add__(self, other: "StepFlops") -> "StepFlops":
        return StepFlops(self.model + other.model, self.hardware + other.hardware,
                         self.unsized + other.unsized)

    def __rmul__(self, calls: float) -> "StepFlops":
        """The FLOPs of `calls` such calls (a step split into microbatches, a loop's trips)."""
        return StepFlops(calls * self.model, calls * self.hardware, calls * self.unsized)

    def utilizations(self, steps: int, step_time: float, peak: float | None = None) -> dict[str, float]:
        """The interval's HLO utilizations of `steps` steps of `step_time` seconds each, of `peak`
        (`model_flops_utilization`): train/hfu off `hardware` and train/hlo_flops_util off `model`,
        both ending `_lower_bound` beside train/unsized_kernels, the launches a step, where any
        launch's work went uncounted."""
        bound = "_lower_bound" if self.unsized else ""
        scalars = {f"train/{name}{bound}": utilization
                   for name, counted in (("hlo_flops_util", self.model), ("hfu", self.hardware))
                   if (utilization := model_flops_utilization(counted / steps, step_time, peak))
                   is not None}
        return {**scalars, "train/unsized_kernels": self.unsized / steps} if bound else scalars


NO_FLOPS = StepFlops(0.0, 0.0)
"""The FLOPs of no step, where an interval's sum starts."""


class CompiledMemory(NamedTuple):
    """A compiled step's memory as XLA analyses it (`Compiled.memory_analysis`), each in bytes: its
    arguments', its outputs', the outputs aliasing a donated argument, its temporaries' and its code's."""

    argument: int
    output: int
    alias: int
    temp: int
    generated_code: int


_ANALYSED: weakref.WeakKeyDictionary[jax.stages.Compiled, CompiledMemory | None] = (
    weakref.WeakKeyDictionary())


def compiled_memory(executable: jax.stages.Compiled) -> CompiledMemory | None:
    """`executable`'s memory as XLA analyses it, None where its backend states none, read once an
    executable: the planner's probe and forecast, the step's headroom and the run's record
    (`dew.training.memory`), a step compiled ahead, and a GPU program's admission to the host's
    communicators (`dew.communicators`, its code's bytes) each read one compiled step, and each read walked
    its program again (0.2 to 0.5 s of a CPU U-Net step's, row 3763)."""
    if executable not in _ANALYSED:
        stats = executable.memory_analysis()
        _ANALYSED[executable] = None if stats is None else CompiledMemory(
            int(stats.argument_size_in_bytes), int(stats.output_size_in_bytes),
            int(stats.alias_size_in_bytes), int(stats.temp_size_in_bytes),
            int(stats.generated_code_size_in_bytes))
    return _ANALYSED[executable]


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
    return hlo_flops(compiled)


def compiled_flops(compiled: jax.stages.Compiled) -> float | None:
    """The hardware FLOPs of one call of a compiled executable (`step_flops`)."""
    flops = step_flops(compiled)
    return None if flops is None else flops.hardware


def hlo_flops(source: str | jax.stages.Compiled) -> StepFlops | None:
    """Matmul and convolution FLOPs of one call of an optimized HLO module (`_module`: a compiled
    executable's, or one written as text), read and weighted once: the forward a remat recomputes counted
    once in `model` and each time it runs in `hardware`, each Pallas kernel at the cost
    it states, the model's share of it where it declares one (`MODEL_FLOPS`),
    beside the custom calls it runs whose work neither counts
    (`StepFlops.unsized`, `sizes`)."""
    computations, entry = _read(source)
    if entry is None:
        return None
    found = _inclusive(entry, computations, {})
    totals = (found.model, found.hardware, found.unsized)
    return found if all(math.isfinite(total) for total in totals) else None


def _inclusive(name: str, computations: dict[str, _Computation], memo: dict[str, StepFlops]) -> StepFlops:
    """The FLOPs of one call of computation `name` with every computation it runs (`hlo_flops`): its own
    instructions', each computation it calls times the calls (`_call_counts`, a loop's body its trips), and
    of each conditional's branches the one of the most hardware FLOPs, the most a run of it takes: a
    conditional runs one branch, where every branch summed counted a rare walk's kernel beside the common
    one's (the scaled attention forward's, `attention_algebra.may_scale`)."""
    if name in memo:
        return memo[name]
    computation = computations[name]
    counted = [(_instruction_flops(instruction, computation.shapes), instruction.recomputed)
               for instruction in computation.instructions.values()]
    sized = [(*flops, recomputed) for flops, recomputed in counted if flops is not None]
    total = StepFlops(sum(flops for flops, _, recomputed in sized if not recomputed),
                      sum(flops for _, flops, _ in sized), float(len(counted) - len(sized)))
    for instruction in computation.instructions.values():
        branches = _branches(instruction)
        for callee, times in _call_counts(instruction, computation, computations).items():
            if callee in computations and times and callee not in branches:
                called = _inclusive(callee, computations, memo)
                # A call that runs none adds none, however many times its loop would.
                if called != NO_FLOPS:
                    total = total + times * called
        ran = [_inclusive(branch, computations, memo) for branch in branches if branch in computations]
        if ran:
            total = total + max(ran, key=lambda flops: (flops.hardware, flops.model, flops.unsized))
    memo[name] = total
    return total


def model_flops_utilization(
    flops_per_step: float | None, step_time: float, peak: float | None = None
) -> float | None:
    """Fraction of one device's dense `peak` achieved by its executable, the
    bf16 peak of device 0's kind by default.

    The optimized module is the program one device runs under SPMD, so its
    shapes are per-device and the denominator is one device's peak rather
    than the mesh's.
    """
    if not flops_per_step or step_time <= 0:
        return None
    peak = gpu_target(jax.devices()[0].device_kind).bf16_peak if peak is None else peak
    if peak is None:
        return None
    return flops_per_step / step_time / peak
