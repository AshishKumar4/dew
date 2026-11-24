"""Whether a training step fits its devices, and the rungs that make it fit.

The trainer compiles a step, reads its memory from the executable and each
device's free bytes, and climbs the objective's recomputation ladder or
splits the step's batch until the step fits everywhere
(`dew.training.planning`); these are those measurements and the ladder.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import functools
import json
import logging
import math
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import AbstractMesh, Mesh, NamedSharding

from dew import communicators
from dew._devices import allocator_limit, device_memory, fresh_pool, gpu_free_bytes, stated_share, xla_flag
from dew._records import JSON, integer, integers, json_value, listed, record
from dew.batch import FROZEN, Batch, Variables
from dew.data.dataset import rows_of
from dew.distributed.placement import parameter_axes
from dew.mesh import batch_axes, mesh_axes
from dew.nn.backbones.layer_names import StoredRow, group_layers, stored_layers
from dew.nn.kernels.generation import backend, device_generation, gpu_target
from dew.nn.kernels.options import Option, compiler_options
from dew.nn.kernels.templates import report
from dew.nn.protocols import OffloadsResiduals, Recomputing, TritonGemm
from dew.objectives.base import Objective, head_fused_of, head_tile_of
from dew.state import TrainState
from dew.telemetry.instrumentation import CompiledMemory, compiled_memory

_log = logging.getLogger(__name__)


COLLECTIVES = {
    'xla_gpu_enable_latency_hiding_scheduler': True,
    'xla_gpu_all_reduce_combine_threshold_bytes': 1 << 30,
    'xla_gpu_all_gather_combine_threshold_bytes': 1 << 30,
    'xla_gpu_reduce_scatter_combine_threshold_bytes': 1 << 27,
    'xla_gpu_enable_all_gather_combine_by_dim': False,
    'xla_gpu_enable_reduce_scatter_combine_by_dim': False,
    'xla_disable_hlo_passes': 'rematerialization',
}
"""NVIDIA JAX-Toolbox's collective options for a GPU step over several devices: the scheduler overlaps
each collective with compute, and the collectives combine into few large ones. Measured in
docs/performance.md, "Collectives over several GPUs". A step that gathers its layers' parameters combines
its all-gathers up to one layer's instead (`gathered_layer_bytes`).

XLA's rematerialization pass is off, as MaxText runs it (row 1307): the plan fits a step to the devices'
room by its own ladder (`dew.training.planning.FitSearch`), and XLA's pass, against a limit of its own below
the plan's, recomputes what it picks at a rung that already fits, a sharded weight's all-gather among them.
Qwen3-8B's step over four compile-only H100s (fsdp 2 x tensor 2, two 4096-token rows a data shard, 48 GiB
stated) settled at `minimal` either way, 19.2 GB of temporaries; with the pass on it gathered 14
rematerialized all-gathers more, 16.8 GB a step against 11.7, 78 of them synchronous against 66."""

GATHER_THRESHOLD = 'xla_gpu_all_gather_combine_threshold_bytes'
"""The bytes XLA combines a step's all-gathers up to (`COLLECTIVES`, `step_compiler_options`)."""

def gathered_layer_bytes(objective, variables: Variables) -> int | None:
    """The bytes a device gathers for the largest layer's split parameters (`splits_parameter`), every
    collection's (a frozen split's too), each gathered whole in the dtype it is stored in: what a layer's
    gathers move at most, whatever narrower dtype its step converts a shard to first (a bf16 step over fp32
    parameters gathers half), so no layer's gathers are split, and at most two of a bf16 step's are held
    ahead. A layer is a stack module's `layers_N`, every run the objective's networks store stacked (a
    stage-local or periodic run, a host-owned state's bank of its frozen trees, a distillation teacher's
    among them) read back onto its layers first (`stored_layers`, the networks' own views, as their exports
    read them); None where no layer's leaf is split. A step whose all-gathers
    combine up to one layer's gathers each layer at once (`step_compiler_options`): on 2 x H100 at fsdp 2,
    Qwen3.5-4B's 1 GiB combined about 4.7 layers' 215 MiB each in its forward and held 5 GiB more at its
    peak than 128 MiB did, in the same time (IG-SCALING-2XH100), and a Llama-3.1-8B's 128 MiB split its 0.44
    GB layers and lost 2.6% (docs/performance.md)."""
    def layered(tree):
        # A bank's rows are read where they lie (`StoredRow`): their shapes and dtypes are what is sized.
        return jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype) if isinstance(leaf, StoredRow)
                            else leaf, stored_layers(objective, tree, banks=True))

    # Two trees cut alike: the leaves' own dtypes, and a mark of which a parameter axis splits.
    sized = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), variables)
    marks = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(
        leaf.shape, jnp.int8 if splits_parameter(leaf) else jnp.uint8), variables)
    sized, marks = jax.eval_shape(lambda sized, marks: (layered(sized), layered(marks)), sized, marks)
    layers: dict[tuple[str, ...], int] = {}
    for (path, leaf), mark in zip(jax.tree_util.tree_flatten_with_path(sized)[0], jax.tree.leaves(marks),
                                  strict=True):
        names = [str(key.key) for key in path[1:] if isinstance(key, jax.tree_util.DictKey)]
        found = [index for index, name in enumerate(names) if group_layers(name) is not None]
        if found and mark.dtype == jnp.int8:
            layer = tuple(names[:found[-1] + 1])
            layers[layer] = layers.get(layer, 0) + leaf.size * jnp.dtype(leaf.dtype).itemsize
    return max(layers.values(), default=None)


def splits_parameter(leaf) -> bool:
    """Whether a parameter axis splits `leaf`, a placed array (`parameter_axes`): a step gathers it whole
    for its use."""
    sharding = leaf.sharding
    return isinstance(sharding, NamedSharding) and any(
        sharding.mesh.shape[axis] > 1 for assignment in sharding.spec
        for axis in parameter_axes(mesh_axes(assignment)))


def layer_gathers(objective, variables: Variables, devices: int) -> int | None:
    """The bytes a step over `devices` combines its all-gathers up to (`gathered_layer_bytes`) where it
    takes the collective options (`collects`), None elsewhere: a fact of the run's plan, which its compiles'
    options read (`step_compiler_options`)."""
    return gathered_layer_bytes(objective, variables) if collects(devices) else None


def collects(devices: int) -> bool:
    """Whether a step over `devices` takes the collective options (`COLLECTIVES`): a GPU's over several."""
    return backend() == 'gpu' and devices > 1


def step_compiler_options(objective, tokens: float, frozen: bool, devices: int, *,
                          gathered: int | None = None) -> jax.stages.CompilerOptions | None:
    """Return the XLA options for this objective's training step on this device.

    Each option applies unless the run set that flag itself. `tokens` is the
    number of tokens one device steps (`device_tokens`), `frozen` says
    whether the step trains next to frozen weights, and `devices` is how
    many the step spans: over several GPUs it takes `COLLECTIVES`, its
    all-gathers combined up to the `gathered` bytes of its largest layer in
    whole MiB where it gathers its layers (`layer_gathers`, the plan's).

    On a GPU, dots that share an input (q, k and v; gate and up) run separately.
    XLA's dot merger would run them as one GEMM over their weights, concatenated
    again every step, which costs 4.0 ms of Qwen3-0.6B's step at 1 x 1024. A step
    next to frozen weights (a LoRA adapter's) with 128 tokens or fewer per device
    keeps the merger, as decoding does, because 32-token GEMMs ran 5-15% faster
    merged. On an RTX 4080, bf16, ms per step (docs/performance.md has every row):

        step                       tokens    merged   apart
        Qwen3-0.6B, LoRA r16       1 x 32     15.92   16.48
        Qwen3-0.6B, LoRA r16       4 x 32     18.64   18.99
        Qwen3-0.6B, LoRA r16       2 x 64     18.59   18.89
        Qwen3-0.6B, LoRA r16      1 x 128     20.79   19.18
        Qwen3-0.6B, LoRA r16       8 x 32     24.02   22.97
        Qwen3-0.6B, LoRA r16     1 x 1024     60.61   59.19
        Qwen3-0.6B widths, full    1 x 32     46.83   44.93
        Qwen3-0.6B widths, full  1 x 1024     97.7    94.1
        3-layer decoder, full      1 x 32      5.01    4.89

    At 128 tokens the shapes disagree: 1 x 128 runs faster apart, while 2 x 64
    and 4 x 32 run faster merged. The boundary includes 128, so no step runs
    slower than XLA's default, and 1 x 128 gives up the 1.6 ms it would gain apart.

    Only an objective whose rows hold tokens gives them (`Objective.row_tokens`), so any other
    objective's frozen step runs apart. Triton GEMM fusions are turned off where
    the GPU measured a win without them (`GpuTarget.triton_gemm_fusions`) and no module the step
    runs keeps them (`TritonGemm`), and wherever a module the step runs computes in int8
    (`compiler_options`).
    """
    generation = device_generation()
    options: dict[str, Option] = {}
    if collects(devices):
        options.update({name: value for name, value in COLLECTIVES.items() if xla_flag(name) is None})
        if gathered and xla_flag(GATHER_THRESHOLD) is None:
            options[GATHER_THRESHOLD] = -(-gathered // 2 ** 20) * 2 ** 20
    small = tokens <= 128
    if (backend() == 'gpu' and xla_flag('xla_gpu_dot_merger_threshold_mb') is None
            and not (frozen and small)):
        options['xla_gpu_dot_merger_threshold_mb'] = 0
    modules = [entry.module for entry in objective.program_key()]
    if (not gpu_target(generation).triton_gemm_fusions
            and xla_flag('xla_gpu_enable_triton_gemm') is None
            and modules and not any(isinstance(module, TritonGemm) and module.keeps_triton_gemm
                                    for module in modules)):
        options['xla_gpu_enable_triton_gemm'] = False
    options.update(compiler_options())
    return options or None


def admitted_or_missed(take: Callable[[], jax.stages.Compiled],
                       refusals: list[Exception]) -> jax.stages.Compiled | None:
    """`take()`'s executable, a program compiled or loaded at a fit candidate (`compiled_if_it_fits`,
    `dew.training.ahead.Executable.taken`), or None where its admission refuses it
    (`dew.communicators.CommunicatorRoom`: its communicators and load have no room outside XLA's pool), the
    refusal kept: the search asks the next candidate, as for a step that misses, and a search that finds
    none raises naming it (row 12131's review: a losing rung's admission ended the search)."""
    try:
        return take()
    except communicators.CommunicatorRoom as room:
        _log.info("the step's communicators and load have no room outside XLA's pool: %s", room)
        refusals.append(room)
        return None


def compiled_if_it_fits(program: jax.stages.Lowered, options: jax.stages.CompilerOptions | None,
                        refusals: list[Exception], *, relower: Callable[[], jax.stages.Lowered],
                        first: report.Attempt | None = None) -> jax.stages.Compiled | None:
    """Return `program` compiled under `options`, or None if XLA refuses it for memory (`refused_for_memory`).

    It compiles through the kernel templates' reports (`report.compiled`): where a launch's kernel
    spills, its config is refused and the step lowered afresh (`relower`), its derivation taking the
    next config, and compiled once more. A refusal is appended to `refusals`. XLA:TPU checks a
    program's temporaries against HBM while it compiles and raises RESOURCE_EXHAUSTED instead of
    returning an executable whose memory `step_fits` could read, so the refusal means the step does
    not fit. For example, on a v6e, Qwen3-0.6B at 16 x 1024
    tokens asked for 38.47G of temporaries next to 31.24G of HBM. A program its admission refuses is a miss
    too (`admitted_or_missed`). Any other compile error is raised. `first` is its first compile where
    another thread made it (`report.attempts`).
    """
    try:
        return admitted_or_missed(lambda: report.compiled(program, options, relower=relower, first=first),
                                  refusals)
    except jax.errors.JaxRuntimeError as error:
        if not refused_for_memory(error.error_code_string, str(error)):
            raise
        _log.info("XLA refused the step for memory: %s", error)
        refusals.append(error)
        return None


AUTOTUNED_FAILURES = re.compile(r"^Failures \((\d+)\):$", re.MULTILINE)
"""The head of a GPU GEMM autotuner's list of what failed (XLA's `PickBestConfig`, config_selector.cc): every
config's failure (`ConfigRunner::Failure::ToString`, config_runner.cc: `EXECUTION FAILED`, `COMPILATION
FAILED`, `REDZONE CHECK FAILED` or `WRONG RESULTS`, then its status) and every excluded backend's line, one
a line, as many as the head counts."""

EXECUTED_OUT_OF_MEMORY = re.compile(r"^EXECUTION FAILED: RESOURCE_EXHAUSTED: ", re.MULTILINE)


def refused_for_memory(code: str, message: str) -> bool:
    """Whether XLA refused a program it compiled for memory, by its error's status `code` and `message`:
    RESOURCE_EXHAUSTED, or on a GPU its GEMM autotuner finding no config it could profile (NOT_FOUND)
    where each of the failures it lists (`AUTOTUNED_FAILURES`) is a config's execution out of device
    memory. One that failed to compile, failed its redzone check or gave wrong results, or a backend
    excluded beside them, is no memory refusal, and the error is raised. An A100 with 0.30 of its memory
    to a decoder's step refused it so at the whole head's f32[65536,32000] GEMM, each of its eight configs
    7.83 GiB short (node receipt TL-a100-offload-1338-d): the step at a rung that holds less compiles."""
    if code == 'RESOURCE_EXHAUSTED':
        return True
    failures = AUTOTUNED_FAILURES.search(message)
    if code != 'NOT_FOUND' or failures is None:
        return False
    count = int(failures.group(1))
    return count > 0 and len(EXECUTED_OUT_OF_MEMORY.findall(message, failures.end())) == count


def split_share(params: Variables) -> float:
    """The share of the bytes of `params`, placed arrays, every collection a
    frozen split keeps among them, that a parameter axis splits
    (`parameter_axes`): a mesh can name fsdp or tensor and still
    split nothing of a model whose parameters all sit below `Layout`'s
    `min_shard`, and the run's banner says how much it does."""
    total = split = 0
    for leaf in jax.tree.leaves(params):
        total += leaf.nbytes
        if splits_parameter(leaf):
            split += leaf.nbytes
    return split / total if total else 0.0


def step_options[LossT, EffectsT](objective: Objective[LossT, EffectsT], micro: Batch, variables: Variables,
                                  mesh: Mesh | AbstractMesh, gathered: int | None = None
                                  ) -> jax.stages.CompilerOptions | None:
    """The options a step over microbatches of `micro` on `mesh` compiles with
    (`step_compiler_options`): by the tokens one device steps of a microbatch split over the
    mesh's batch axes, whether `variables` hold frozen weights it trains beside, and the plan's
    `gathered` bytes (`layer_gathers`)."""
    shards = math.prod(mesh.shape[axis] for axis in batch_axes(mesh))
    return step_compiler_options(objective, device_tokens(objective, micro, shards), FROZEN in variables,
                                 mesh.size, gathered=gathered)


def device_tokens[LossT, EffectsT](objective: Objective[LossT, EffectsT], batch: Batch, shards: int) -> float:
    """The tokens one device steps of `batch`, split over `shards` row shards, or infinity where its
    rows hold no token geometry (`Objective.row_tokens`), an image's. Rows are counted only then, so
    another objective's batch, one that holds no rows among them, is never asked."""
    per_row = objective.row_tokens(batch)
    return math.inf if per_row is None else rows_of(batch) // shards * per_row


def climb_to[LossT, EffectsT](objective: Objective[LossT, EffectsT], rung: Mapping[str, object]) -> None:
    """Move `objective` up to the head tile, its fusion and the remat recorded in a checkpoint's `rung`
    (`Trainer._rung`).

    Each setting is raised only where it is below the recorded one
    (`Recomputing.restore_recompute`). This takes the ladder `recompute_more`
    climbs in one move.
    """
    tile = rung.get('head_tile')
    if tile is not None and head_tile_of(objective) is None:
        rows, columns = integers(tile, 'rung head_tile')
        objective.tile_head((rows, columns))
    if rung.get('head_fused') is False:
        objective.unfuse_head()
    entries = objective.program_key()
    modules = _climbed(objective, rung)
    if any(module is not entry.module for module, entry in zip(modules, entries, strict=True)):
        objective.substitute(modules)


def _climbed[LossT, EffectsT](objective: Objective[LossT, EffectsT], rung: Mapping[str, object]) -> list:
    """The objective's modules, in program order, each trained one at the remat `rung` records where that is
    above its own (`Recomputing.restore_recompute`), the objective left as it is."""
    entries = objective.program_key()
    trained = [index for index, entry in enumerate(entries) if entry.trained]
    recorded = json_value(rung.get("remat"), "rung remat")
    # One trained module records its rung alone, several a list in program order.
    records = [recorded] if len(trained) == 1 else recorded if isinstance(recorded, list) else []
    rungs = dict(zip(trained, records, strict=False))
    return [entry.module.restore_recompute(rungs[index])
            if index in rungs and isinstance(entry.module, Recomputing) else entry.module
            for index, entry in enumerate(entries)]


def recompute_record[LossT, EffectsT](objective: Objective[LossT, EffectsT]) -> JSON:
    """What the modules the objective trains recompute in their backward
    pass, as a checkpoint records it (`Recomputing.recompute_record`): one
    module's record, a list of several in program order, None for a module
    that does not say and for an objective that trains none."""
    records = [entry.module.recompute_record() if isinstance(entry.module, Recomputing) else None
               for entry in objective.program_key() if entry.trained]
    return records[0] if len(records) == 1 else records or None


def remat_stated[LossT, EffectsT](objective: Objective[LossT, EffectsT]) -> bool:
    """Whether any module the objective trains sets its own remat (`recompute_record`), which a run
    compiles at alone: its ladder is its head's rungs (`dew.training.planning.FitSearch.ladder`)."""
    record = recompute_record(objective)
    return any(entry is not None for entry in (record if isinstance(record, list) else [record]))


def rung_record[LossT, EffectsT](objective: Objective[LossT, EffectsT], xla_defaults: bool, *,
                                 narrow: bool, ns_group: int | None = None) -> dict[str, JSON]:
    """The rung of the fit ladder a step compiles at, as a checkpoint records
    it: the objective's head tile and whether that head is fused, its trained
    modules' remat (`recompute_record`), whether the step keeps XLA's
    default options, beside each head (`dew.training.planning.FitSearch.ladder`),
    whether it reads the parameters through narrow copies
    (`dew.training.narrow`), and the stacks of one shape its Newton-Schulz joins at a time where the run
    states none (`ns_group`, `dew.newton_schulz.released`: None joins them all). Each decides the program a
    step runs."""
    tile = head_tile_of(objective)
    return {'head_tile': None if tile is None else list(tile), 'head_fused': head_fused_of(objective),
            'remat': recompute_record(objective), 'xla_defaults': xla_defaults, 'narrow': narrow,
            'ns_group': ns_group}


def step_headroom(executable: jax.stages.Compiled, devices: Sequence, held: int = 0,
                  built: int = 0, *, placed: bool = True) -> int | None:
    """Return the bytes the tightest of `devices` has to spare once the step's memory is placed.

    The step's temporaries and new outputs are placed next to `held` more bytes
    that the loop keeps outside the step. The result is None if the executable or
    a device reports no memory, a compile-only GPU reading as its stated memory holds them
    (`device_memory`). The arguments (the state and the batch) are
    already resident and counted as in use, and the donated state's buffers are
    reused for the outputs that alias them; on a stated device, `built` bytes of them, which `held`
    counts as the step's first call builds them, are not yet.

    XLA's GPU step takes all its temporaries in one allocation, so it needs one
    free block that large, not that many free bytes in total (`placeable`). Where
    the allocator `strands_temporaries`, they need room twice. Where the state is not `placed` yet, `held`
    counting it, the devices' free bytes in all hold the step (`available`).
    """
    return step_admitted(executable, devices, held, built, placed=placed)[0]


def step_admitted(executable: jax.stages.Compiled, devices: Sequence, held: int = 0, built: int = 0, *,
                  placed: bool = True) -> tuple[int | None, tuple[int, int, int] | None]:
    """`step_headroom` of `executable` and the terms it admitted by (`admission`), each None where the
    executable states no memory: one reading of its compiled memory for both."""
    stats = compiled_memory(executable)
    if stats is None:
        return None, None
    terms = admission(stats, held, built)
    return admitted(devices, *terms, placed=placed), terms


def admission(stats: CompiledMemory, held: int = 0, built: int = 0) -> tuple[int, int, int]:
    """A compiled program's terms as the allocator admits it (`admitted`): its resident bytes, its argument
    bytes less the `built` bytes of them its first call builds in place; the bytes placed beside its
    temporaries (`placed_beside`: its outputs that write over no donated argument and the `held` bytes kept
    outside it); and its temporaries, one allocation. A step's fit (`step_headroom`) and a GPU part's need
    (tests/model_flows.py's `need`) read it alike."""
    return stats.argument - built, placed_beside(stats, held), stats.temp


def placed_beside(stats: CompiledMemory, held: int) -> int:
    """The bytes a step's fit check places beside its temporaries (`step_headroom`): its outputs that write
    over no donated argument, and the `held` bytes the loop keeps outside the step (copies, accumulators, a
    state the devices hold none of yet), what a refusal states beside them (`Probe.beside`)."""
    return stats.output - stats.alias + held


MEMORY_STATS = ('bytes_limit', 'bytes_in_use', 'pool_bytes', 'largest_free_block_bytes', 'peak_bytes_in_use')


def peak_memory_gib() -> float | None:
    """The largest local device allocator high-water mark, in GiB.

    The logging tick reads this on process zero. It is the allocator's
    lifetime peak, not an interval delta or a cross-process maximum. CPU
    backends and allocators without peak_bytes_in_use report nothing.
    """
    peaks = [stats['peak_bytes_in_use'] for device in jax.local_devices()
             if (stats := device.memory_stats()) is not None and 'peak_bytes_in_use' in stats]
    return max(peaks) / 1024 ** 3 if peaks else None


def admitted(devices: Sequence, resident: int, beside: int, temporaries: int, *,
             placed: bool = True) -> int | None:
    """The allocator's admission of a compiled step (`step_headroom`): the bytes the tightest of `devices`
    has to spare once the step's `temporaries`, twice where its allocator strands them (`stranding`), and
    `beside` more bytes are placed, with `resident` bytes in use already (`_admitted`); None where a device
    states no memory. A compile-only GPU's stated memory holds `resident` as it would (`device_memory`)."""
    memory = [device_memory(device, resident) for device in devices]
    return _admitted(devices, memory, beside, temporaries, placed=placed)


def ceiling_headroom(devices: Sequence, resident: int, beside: int, temporaries: int) -> int | None:
    """The headroom `admitted` reads of a step of its admission's terms (`admission`: `resident`, `beside`,
    `temporaries`) beside a state not placed yet, at each allocator's own ceiling (`bytes_limit -
    bytes_in_use`), no GPU's free bytes capping a growing pool's (`unclaimed`, which another process may give
    back between readings): the most room a later reading of the devices can give the step while this
    process holds what it holds now."""
    memory = [device_memory(device, resident) for device in devices]
    return _admitted(devices, memory, beside, temporaries, placed=False, ceiling=True)


def _admitted(devices: Sequence, memory: Sequence[Mapping[str, int]], beside: int,
              temporaries: int, *, placed: bool = True, ceiling: bool = False) -> int | None:
    """The headroom at a fixed reading of the allocators: the least, over `devices`, of two margins. XLA
    takes a step's temporaries in one allocation, after the bytes placed beside them (the outputs no
    donated argument holds, and what the loop keeps that the allocators do not hold yet: the batches, and
    a step's copies and window where no search built them, `Building.fits`), so the temporaries need
    one free block that large once those are placed wherever they may land (`placeable_beside`); with the
    temporaries they need the free bytes in all (`available`). A search that builds what its step builds
    before it runs reads where those landed (M-MC-24c: an H100's 7.27 GB of copies took the one 21.64 GB
    block its 20.76 GB of temporaries were admitted to by a reading made before them). Where the state is
    not `placed` yet, placing it reshapes the blocks, so the free bytes in all judge the temporaries too."""
    if not stated(memory):
        return None
    margins = []
    for device, held in zip(devices, memory, strict=True):
        free = None if ceiling else unclaimed(device, held)
        needed = temporaries * stranding(device, held)
        block = placeable_beside(held, free, beside) if placed else available(held, free)
        margins.append(min(block - needed, available(held, free) - beside - needed))
    return min(margins)


def needed(device, resident: int, beside: int, temporaries: int) -> int:
    """The least bytes a GPU like `device` must hold for `admitted` to admit a step of `temporaries`, with
    `resident` bytes in use and `beside` more placed: its allocator a fresh pool at its limit, as a GPU not
    attached states it (`dew._devices.fresh_pool`), read through the one admission (`_admitted`) that
    leaves zero or more bytes to spare there and none one byte fewer: the inverse of `admitted`, which a
    pool of GPUs is sized by."""

    def headroom(memory: int) -> int:
        spare = _admitted([device], [fresh_pool(memory, resident, stated_share(device))], beside,
                          temporaries)
        assert spare is not None
        return spare

    # The headroom grows with the memory: double to a size that admits, then halve to its least.
    low, high = 0, max(1, resident + beside + temporaries)
    while headroom(high) < 0:
        low, high = high + 1, 2 * high
    while low < high:
        middle = (low + high) // 2
        low, high = (low, middle) if headroom(middle) >= 0 else (middle + 1, high)
    return low


def least_limit(device, terms: Iterable[tuple[int, int, int]]) -> int:
    """The least allocator limit admitting each of `terms` at `device`'s stated share (else this
    process's): each a program's admission triple (`admission`: resident, beside, temporaries), its least
    memory on a GPU like `device` (`needed`) read as the limit that GPU's pool holds (`allocator_limit`),
    the most of them. The one reading of a program's need the planner's compile ahead (`programs_need`),
    the GPU lane's measured units (tools/armada/measures.py) and its test share (row 8708). No terms admit
    no program, and are refused."""
    limits = [allocator_limit(needed(device, *each), stated_share(device)) for each in terms]
    if not limits:
        raise ValueError("least_limit: no admission terms, so no program to admit and no least limit")
    return max(limits)


def stated(memory: Sequence[Mapping[str, int]]) -> bool:
    """Whether every device's `memory` (`device_memory`) states its limit and its use, which a fit check
    reads; a CPU's states none, so no step's headroom on it is known ahead of its run."""
    return all('bytes_limit' in m and 'bytes_in_use' in m for m in memory)


def stranding(device, memory: Mapping[str, int]) -> int:
    """How many times its temporaries a step needs room for on `device`, its allocator stating `memory`:
    2 where it can strand them (`strands_temporaries`), else 1."""
    return 2 if strands_temporaries(device.platform, memory) else 1


def local_devices(mesh: Mesh) -> list:
    """This process's devices of `mesh`, whose memory the fit check reads."""
    process = jax.process_index()
    return [device for device in mesh.devices.flat if device.process_index == process]


class FirstRunFailed(RuntimeError):
    """A newly compiled step's first run its devices refused: what they said (`refusal`, the error it is
    raised from, its text first in its own, so a reader of its message alone tells an out-of-memory from
    any other failure), the record of the check that said it fits (`fitted`, `memory_record`) and what this
    process's devices of its mesh hold now (`now`), each GPU's bytes free outside XLA's pool
    (`outside_pool`) among it."""

    def __init__(self, step: int, refusal: BaseException, fitted: JSON, now: list[JSON]):
        self.fitted, self.now = fitted, now
        super().__init__(
            f"the training step compiled for step {step} failed on its first run: "
            f"{type(refusal).__name__}: {refusal}. It was checked to fit: "
            f"{json.dumps(fitted)}; the devices now hold {json.dumps(now)}. 'outside' is a GPU's bytes free "
            "outside XLA's pool, whose limit is its bytes_limit (XLA_PYTHON_CLIENT_MEM_FRACTION of the GPU): "
            "a collective's communicators (NCCL) take their buffers there, beside cuBLAS's workspaces and "
            "the kernels loaded, and a lower fraction leaves them more")


def first_run_failed(error: BaseException, step: int, fitted: JSON,
                     mesh: Mesh) -> FirstRunFailed:
    """A newly compiled step's first run's `error` on its devices, as the step's (`FirstRunFailed`): with
    the record of the check that said it fits (`fitted`) and what each device holds now, wherever the step
    raised it, at its own wait (`first_run`) or at a program's that waits on it before
    (`dew.communicators.called`)."""
    now: list[JSON] = [{"id": device.id, "outside": outside_pool(device),
                        **{key: value for key, value in (device.memory_stats() or {}).items()
                           if key in ('bytes_in_use', 'pool_bytes', 'largest_free_block_bytes')}}
                       for device in local_devices(mesh)]
    return FirstRunFailed(step, error, fitted, now)


def first_run(loss: jax.Array, step: int, fitted: JSON, mesh: Mesh) -> dict[str, dict[int, int]]:
    """Wait for a newly compiled step's first run, and raise its failure on the devices as the step's
    (`FirstRunFailed`). A first run that succeeds returns, and logs, each GPU's bytes free outside XLA's
    pool as the fit check read them (`fitted`) and now: what the devices took there since, every
    process's and a pool's growth counted, the step's collectives' communicators (NCCL) among it."""
    devices = local_devices(mesh)
    try:
        loss.block_until_ready()
    except jax.errors.JaxRuntimeError as error:
        raise first_run_failed(error, step, fitted, mesh) from error
    read = () if fitted is None else listed(record(fitted, "fitted")["devices"], "devices", read=record,
                                            what="each device's memory as the fit check read it")
    checked = {integer(device["id"], "id"): outside for device in read
               if isinstance(outside := device.get("outside"), int)}
    current = {device.id: free for device in devices
               if device.id in checked and (free := outside_pool(device)) is not None}
    observed = {"checked": {device: checked[device] for device in current}, "now": current}
    if current:
        _log.info("the GPUs' free bytes outside XLA's pool: %s as the step's fit check read them, %s after "
                  "its first run, which takes its collectives' communicators there", observed["checked"],
                  current)
    return observed


def outside_pool(device) -> int | None:
    """The bytes free on a GPU of this process outside XLA's pool, as its CUDA driver reports them
    (`gpu_free_bytes`), every process's allocations counted; None for any other device."""
    if device.platform != 'gpu' or device not in jax.local_devices():
        return None
    return gpu_free_bytes(device.local_hardware_id)


def memory_record(executable: jax.stages.Compiled, devices: Sequence, held: int, *, fits: bool) -> JSON:
    """What the fit check read for a compiled step, as `step_headroom` reads
    it, and its answer `fits`: XLA's memory analysis of the step, `held`, and
    each of `devices`' allocator stats and free GPU bytes, in bytes. A step
    the check passed and the devices then refused is read from it, with the
    pool Dew sized for the host's communicators (`dew.communicators.held`)."""
    stats = compiled_memory(executable)
    analysis: JSON = None if stats is None else stats._asdict()
    devices_stats: list[JSON] = []
    for device in devices:
        memory = device_memory(device)
        reported: dict[str, JSON] = {key: int(memory[key]) for key in MEMORY_STATS if key in memory}
        devices_stats.append({"id": int(device.id), **reported,
                              "unclaimed": None if not memory else unclaimed(device, memory),
                              "outside": outside_pool(device),
                              "strands": bool(memory) and strands_temporaries(device.platform, memory)})
    sizing = communicators.held()
    return {"analysis": analysis, "held": int(held), "devices": devices_stats, "fits": fits,
            **({} if sizing is None else {"pool": json_value(sizing.record(), "pool")})}


def available(memory: Mapping[str, int], unclaimed: int | None = None) -> int:
    """Return the bytes a device's allocator can give in all, given the `memory` it reports: its free
    bytes, those its pool holds free and those it can still take (`placeable`'s growth), in whatever
    blocks. A state the devices do not hold yet is judged by them (`Stepping.unplaced`): where it lands
    reshapes the free blocks, which the allocator's reading once it is placed says."""
    free = memory['bytes_limit'] - memory['bytes_in_use']
    if 'pool_bytes' not in memory or unclaimed is None:
        return free
    return min(free, memory['pool_bytes'] - memory['bytes_in_use'] + unclaimed)


def unclaimed(device, memory: Mapping[str, int]) -> int | None:
    """Return the bytes free on a GPU whose pool grows (`gpu_free_bytes`), or None for any other device.

    That is the most a new region of the pool can take.
    """
    if device.platform != 'gpu' or memory.get('pool_bytes', memory['bytes_limit']) >= memory['bytes_limit']:
        return None
    return gpu_free_bytes(device.local_hardware_id)


BFC_SPLIT = 128 << 20
"""The bytes past a request a BFC pool's free chunk must leave for the pool to split it, where it is under
twice the request (openxla/xla's tsl/framework/bfc_allocator.cc, `max_internal_fragmentation_bytes_`'s
default): a chunk short of both is handed out whole."""


def placeable_beside(memory: Mapping[str, int], unclaimed: int | None, beside: int) -> int:
    """The largest allocation a device can satisfy, its `memory` read, once `beside` bytes more are placed
    wherever its allocator may put them, a BFC pool's rules bounding where (`placeable`, none beside). A BFC
    pool takes each request from the smallest free chunk that holds it, and splits that chunk where it is
    at least twice the request or leaves `BFC_SPLIT` past it, else hands it out whole. Where the largest
    free block `L` is at least twice the bytes `B` beside, or leaves `BFC_SPLIT` past them, every request
    finds what `L` still holds twice its size or that much past it: each taken from `L` is charged its own
    bytes, `L - B` at least stays, and none grows the pool, whose growth `G` stays a block it can still
    give: `max(L - B, G)`. Otherwise one may take `L` whole (a chunk under twice a request charged under
    twice it) and grow the pool by regions it sizes apart from them, each a chunk charged as any other;
    what the growth leaves, `G - 2B` at least, lies in at most the last region's rest and the growth past
    it, so half of it is one block: `(G - 2B) / 2`. `B` is what the pool charges, each request rounded up to
    its allocation unit (`dew.distributed.placement.allocated_bytes`, as a step's held bytes count them)."""
    free = memory['bytes_limit'] - memory['bytes_in_use']
    if 'pool_bytes' not in memory:
        return free - beside
    largest = memory['largest_free_block_bytes']
    growth = memory['bytes_limit'] - memory['pool_bytes']
    if unclaimed is not None:
        growth = min(growth, unclaimed)
    if 2 * beside <= largest or largest - beside >= BFC_SPLIT:
        return min(free - beside, max(largest - beside, growth))
    return min(free - 2 * beside, (growth - min(2 * beside, growth)) // 2)


def placeable(memory: Mapping[str, int], unclaimed: int | None = None) -> int:
    """Return the largest allocation a device can satisfy, given the `memory` its allocator reports.

    That is the most a step's temporaries, and every buffer next to them, can
    take from one block (`Device.memory_stats`).

    XLA's GPU pool, its BFC allocator, reports its size (pool_bytes) and its
    largest free block. A pool that grows (XLA_PYTHON_CLIENT_PREALLOCATE=false) can
    also take a new block from the part of its limit it has not taken yet, up to
    the GPU's `unclaimed` free bytes. cuda_async and a TPU's allocator report no
    pool, so their free bytes are all there is to read.
    """
    return placeable_beside(memory, unclaimed, 0)


def strands_temporaries(platform: str, memory: Mapping[str, int]) -> bool:
    """Whether a device's allocator can leave a step's temporaries no block to return to.

    If so, the next step needs a second block as large. `platform` and `memory`
    describe the device and what its allocator reports.

    XLA's spatially partitioned BFC pool can (a preallocated pool with
    --xla_gpu_enable_allocator_spatial_partitioning left on, its default): a free
    block below a buffer serves every small allocation before the open space past
    it, so a batch prefetched past the temporaries lets the next step's outputs
    take a few bytes of their block. `import dew` turns the partitioning off.
    cuda_async can do the same inside a pool it does not report.
    """
    if 'pool_bytes' not in memory:
        return platform == 'gpu'
    partitioning = xla_flag('xla_gpu_enable_allocator_spatial_partitioning') or 'true'
    return memory['pool_bytes'] >= memory['bytes_limit'] and partitioning.lower() not in ('false', '0')


def temporaries_factor(mesh: Mesh) -> int:
    """How many times its temporaries a step needs room for on this process's tightest device of
    `mesh` (`strands_temporaries`): 2 where an allocator can strand them, else 1."""
    local = [device for device in mesh.devices.flat if device.process_index == jax.process_index()]
    return max(stranding(device, memory) if (memory := device_memory(device)) else 1
               for device in local)


@dataclasses.dataclass(frozen=True)
class MemoryMiss:
    """What a compile of the step at `split` microbatches that did not fit says of the splits above it,
    every number its own: the bytes a microbatch's rows scale (`scaled`: those its live set states descend
    from a batch row axis, with those it states no provenance of, `LiveSet`; every temporary where it states
    no live set, which errs toward a split too coarse, which its compile refuses, never one too fine, which
    every step would pay for) and the bytes it was short (`shortfall`, its stranding counted, `factor`). A
    split above it holds its share of them; the next program is XLA's to fuse and schedule again, so the
    compile at the split this names is its judge."""

    split: int
    scaled: int
    shortfall: int
    factor: int

    def fits_at(self, split: int) -> bool:
        """Whether the temporaries this miss read leave room at `split` microbatches, held at their share,
        `scaled * self.split / split`."""
        saved = self.scaled - self.scaled * self.split // split
        return self.factor * saved >= self.shortfall


LIMIT_EXIT = "xla_gpu_memory_limit_early_exit_bytes"
"""A GPU compile's limit (Dew's JAX build's, from post99): past it, the arguments, new outputs and
temporaries XLA assigns (argument + output - alias + temp), the compile stops after buffer assignment, its
executable stating its memory and its live set and refusing to run; within it, it compiles whole. Each
attempt of a plan is one compile, whole where the step can fit and stopped where it cannot."""


def compile_limit(devices: Sequence, held: int, built: int,
                  resident: Mapping[object, int] | None = None) -> int | None:
    """The bytes past which no step fits `devices` (`LIMIT_EXIT`), as XLA sums a step (its arguments, its
    outputs that write over none of them and its temporaries): the fit check's own arithmetic
    (`_admitted`'s margin over all the bytes the allocator can give, `available`) solved for that sum,
    from the allocators' reading as the compile starts (`step_room`). The check places the step's
    temporaries and new outputs beside its arguments, `built` of them not yet resident, and the `held`
    bytes the loop keeps beside it, `built` among them; so the most a fitting step's sum reaches is the
    least device's room less what is held beside the step outside its arguments (`held - built`: the
    prefetched batches). A step XLA stops past it fits nowhere the check admits; one within it is compiled
    whole and the check judges it. A reading of what else the devices hold (a borderline step beside
    another program's buffers, row 12465) stops a step the check would refuse at its buffer assignment,
    before its whole codegen. A 4.8B multimodal transformer's step on a GH200 (row 5805): the limit less all
    of `held`, the arguments it builds among it counted twice, stopped a step the check admitted, which its
    first run then refused. None where a device states no limit."""
    rooms = [step_room(device, (resident or {}).get(device, 0)) for device in devices]
    if not rooms or None in rooms:
        return None
    return min(room for room in rooms if room is not None) - (held - built)


def step_room(device, resident: int = 0) -> int | None:
    """The most a step's sum (its arguments, its new outputs, its temporaries) can take of `device` before
    what the loop holds beside it: the bytes its allocator can give (`available`, as the fit check reads
    them, `_admitted`), and the step's own arguments' `resident` bytes among those in use given back, as
    the step reuses them; None where the device states no limit. `resident` over what the allocator holds
    in use counts as all of it: a reading never gives back more than it holds."""
    memory = device_memory(device)
    if "bytes_limit" not in memory:
        return None
    return available(memory, unclaimed(device, memory)) + min(resident, memory.get("bytes_in_use", 0))


def resident_on(tree: object, devices: Sequence) -> dict[object, int]:
    """The bytes the arrays of `tree` hold on each of `devices` now, a placed state's shards on its
    devices; an abstract or donated leaf holds none."""
    held = dict.fromkeys(devices, 0)
    for leaf in jax.tree.leaves(tree):
        if isinstance(leaf, jax.Array) and not leaf.is_deleted():
            for shard in leaf.addressable_shards:
                if shard.device in held:
                    held[shard.device] += shard.data.nbytes
    return held


@dataclasses.dataclass(frozen=True)
class LiveSet:
    """What a compile's live set at its peak (`jax.stages.Compiled.peak_live`: each buffer's dimensions'
    provenance among the step's arguments) says of the step: the bytes of the buffers a microbatch's rows
    scale (`scaled`: a dimension descends from a batch argument's row axis), and those whose provenance no
    op states (`unknown`: an opaque kernel's). An argument's buffer (the state, the batch) is neither: no
    split changes it."""

    scaled: int
    unknown: int
    made: tuple[tuple[int, str, tuple[str, ...]], ...] = ()
    """Each buffer not an argument's: its bytes, the `op_name` of the op that made it and the
    `checkpoint_name` names it carries, which a search reads what a remat rung or a release frees by
    (`forward_bytes`)."""

    @staticmethod
    def of(executable: jax.stages.Compiled, rows: frozenset[tuple[int, int]]) -> LiveSet | None:
        """The live set of `executable`, `rows` the (argument, axis) pairs of the batch's rows; None where
        the backend states none."""
        live = executable.peak_live()
        if live is None:
            return None
        scaled = unknown = 0
        made = []
        for buffer in live:
            if buffer.kind == "parameter":
                continue
            made.append((buffer.bytes, buffer.op_name, tuple(buffer.names)))
            if any(dimension is None for dimension in buffer.dims):
                unknown += buffer.bytes
            elif any(dimension is not None and dimension & rows for dimension in buffer.dims):
                scaled += buffer.bytes
        return LiveSet(scaled, unknown, tuple(made))

    def forward_bytes(self, dropped: Callable[[str, tuple[str, ...]], bool]) -> int:
        """The bytes of the values its step's forward made (its op name's first scope under the step a
        `jvp(`) that `dropped` (an op name and the names its value carries) says a rung would not keep: those
        a backward reads, never the backward's own temporaries, which recomputation does not free."""
        return sum(size for size, op_name, names in self.made
                   if op_name.split("/")[1:2] and op_name.split("/")[1].startswith("jvp(")
                   and dropped(op_name, names))

    def scoped_bytes(self, scope: str) -> int:
        """The bytes of its values made within the name scope `scope` (`jax.named_scope`)."""
        return sum(size for size, op_name, _ in self.made if f"/{scope}/" in f"/{op_name}/")


def fits_everywhere(read: Callable[[], int | None]) -> tuple[bool, int | None]:
    """Whether every process's headroom, which each `read`s of its own devices alone (a compile's measure),
    is non-negative, with the same answer on every process, and this process's: a process that reads none
    (None) holds it, so a pool where none reads one fits. The read and its answer are one agreement
    (`agreed_count`): a process whose read fails ends every process's check with its error, none of them
    left in an exchange the failing one never reaches (row 8767)."""
    from dew._coordination import agreed_count

    headroom, holding = agreed_count("a step's fit", read, lambda headroom: headroom is None or headroom >= 0)
    return holding == jax.process_count(), headroom


def step_fits(executable: jax.stages.Compiled | None, mesh: Mesh, held: int = 0,
              built: int = 0, *, placed: bool = True) -> tuple[bool, int | None]:
    """Whether the compiled step fits in the free memory of every device of `mesh`,
    and this process's headroom (`step_headroom`), None where it reads none.

    Each device must also hold `held` more bytes, `built` of them among the step's arguments
    (`step_headroom`), and the processes that hold them agree on the answer. A step that XLA
    refused for memory (`compiled_if_it_fits`) does not fit, and its process still takes part in
    the agreement. Where the state is not `placed` yet, the devices' free bytes in all hold the step
    (`available`).
    """
    fit, headroom, _ = step_fits_admitted(executable, mesh, held, built, placed=placed)
    return fit, headroom


def step_fits_admitted(executable: jax.stages.Compiled | None, mesh: Mesh, held: int = 0, built: int = 0,
                       *, placed: bool = True) -> tuple[bool, int | None, tuple[int, int, int] | None]:
    """`step_fits`, with the terms this process admitted the step by (`step_admitted`), read within the
    one agreement, so a process whose read fails ends every process's check with its error."""
    local = [device for device in mesh.devices.flat if device.process_index == jax.process_index()]
    if executable is None:
        return fits_everywhere(lambda: -1)[0], None, None
    terms: list[tuple[int, int, int] | None] = [None]

    def read() -> int | None:
        headroom, terms[0] = step_admitted(executable, local, held, built, placed=placed)
        return headroom

    fit, headroom = fits_everywhere(read)
    return fit, headroom, terms[0]


def refuse_wide_floats(state: TrainState, mesh: Mesh) -> None:
    """Refuse parameters stored in float64 or complex128 on a TPU mesh.

    XLA's TPU backend has no 64-bit floats. It rewrites them into pairs of
    32-bit ones, which are not IEEE doubles (0.02 came back as
    0.01999999999999999 on a v6e), and the rewrite has no case for nextafter,
    which a truncated normal initializer runs. Parameters stored in float64
    ask for exactly the bits that rewrite loses, and their optimizer
    moments and EMA copy follow them. Other state that x64 makes float64,
    such as a dynamic loss scale's power of two, is exact in the pairs and
    trains as it did."""
    if mesh.devices.flat[0].platform != "tpu":
        return
    wide = [jax.tree_util.keystr(path) for path, leaf in jax.tree_util.tree_leaves_with_path(state.variables)
            if leaf.dtype in (jnp.float64, jnp.complex128)]
    if wide:
        others = f" and {len(wide) - 1} more parameters" if len(wide) > 1 else ""
        raise ValueError(
            f"params{wide[0]}{others} are stored in 64-bit floats, and a TPU has no float64: XLA "
            "rewrites it into pairs of float32, which are not IEEE doubles and have no "
            "nextafter. Store the parameters in float32 on a TPU, or train them on a CPU or a GPU.")


def recompute_more[LossT, EffectsT](objective: Objective[LossT, EffectsT],
                                    base: tuple[int, int] | None = None) -> bool:
    """Move the objective one rung up its ladder, and return whether there was a rung to move to.

    At each remat rung a head that keeps its whole logits for the backward
    pass, or the run's own tile `base`, comes first, then the generation's
    tile (`LMObjective.head_tile`), fused where the step takes the head's
    gradient in its forward, then that tile unfused (`unfuse_head`); after
    that, each module the objective trains recomputes one rung more
    (`remat_more`), with the head back at `base`, fused. A tiled head costs
    less than recomputing more, and more than the whole logits:
    Llama-3.1-8B at 2 x 4096 on two H100s ran 563.0 ms a step tiled under
    'minimal' and 540.0 ms whole, at the same 66.5 GiB peak. The fused head
    is the faster tiled head, its chunk holding a whole vocabulary's logits
    where a tile holds a tile's: at an 8B's head on an A100 (4096 tokens in
    one chunk) 58.4 ms against 64.9.
    """
    if head_tile_of(objective) == base and objective.tile_head() is not None:
        return True
    if objective.unfuse_head() is not None:
        return True
    if not remat_more(objective):
        return False
    objective.restore_head(base, fused=True)
    return True


def remat_more[LossT, EffectsT](objective: Objective[LossT, EffectsT]) -> bool:
    """Move each module the objective trains one rung up its own remat ladder
    where it goes on (`Recomputing.recompute_more`), and return whether any
    moved."""
    return moved(objective, lambda module: module.recompute_more() if isinstance(module, Recomputing)
                 else None)


def remat_topped[LossT, EffectsT](objective: Objective[LossT, EffectsT]) -> int:
    """How many modules the objective trains stand at the top of their remat ladders (`Recomputing`, whose
    `recompute_more` is None there): their whole recomputation, which a "selective" run's search stops short
    of (`dew.training.policy.FitPolicy.remat`)."""
    return sum(1 for entry in objective.program_key()
               if entry.trained and isinstance(entry.module, Recomputing)
               and entry.module.recompute_more() is None)


def remat_topmost[LossT, EffectsT](objective: Objective[LossT, EffectsT]) -> None:
    """Climb every module the objective trains to the top of its remat ladder (`remat_more`), its whole
    recomputation: where a "full" run's step starts (`dew.training.policy.FitPolicy.remat`), its search's and
    its lowering's alike."""
    while remat_more(objective):
        pass


def remat_admits[LossT, EffectsT](objective: Objective[LossT, EffectsT], rung: Mapping[str, object],
                                  remat: str) -> bool:
    """Whether the remat a saved `rung` records (an ahead step's plan, a checkpoint's) is one the run's
    policy takes (`dew.training.policy.FitPolicy.remat`), its objective climbed to it and not moved: any under
    "auto" and "full", which climbs on to the ladders' tops; none recomputing under "none"; no module more at
    its ladder's top than the run's configuration puts there under "selective"."""
    if remat in ("auto", "full"):
        return True
    trained = [entry.trained for entry in objective.program_key()]
    climbed = [module for module, held in zip(_climbed(objective, rung), trained, strict=True)
               if held and isinstance(module, Recomputing)]
    if remat == "none":
        return all(module.recompute_record() is None for module in climbed)
    return sum(module.recompute_more() is None for module in climbed) <= remat_topped(objective)


def remat_offloaded[LossT, EffectsT](objective: Objective[LossT, EffectsT]) -> bool:
    """Move each module the objective trains at the top of its ladder to its sibling that offloads the
    residuals it would recompute (`OffloadsResiduals.recompute_offloaded`), and return whether any
    moved."""
    return moved(objective, lambda module: module.recompute_offloaded()
                 if isinstance(module, OffloadsResiduals) else None)


def offloads[LossT, EffectsT](objective: Objective[LossT, EffectsT]) -> bool:
    """Whether a module the objective trains is at the top of its ladder with a sibling that offloads the
    residuals it recomputes (`OffloadsResiduals.recompute_offloaded`), which `remat_offloaded` moves to."""
    return any(isinstance(entry.module, OffloadsResiduals) and entry.module.recompute_offloaded() is not None
               for entry in objective.program_key() if entry.trained)


def moved[LossT, EffectsT](objective: Objective[LossT, EffectsT],
          step: Callable[[nn.Module], nn.Module | None]) -> bool:
    """Put each module the objective trains where `step` takes it (None leaves it), and return whether
    `step` took any."""
    entries = objective.program_key()
    stepped = [step(entry.module) if entry.trained else None for entry in entries]
    if all(module is None for module in stepped):
        return False
    objective.substitute([entry.module if module is None else module
                          for module, entry in zip(stepped, entries, strict=True)])
    return True




DENSE = {"bfloat16": "bf16", "float8_e4m3fn": "fp8", "float8_e5m2": "fp8"}
"""The dense peak (`GpuTarget.peaks`, by its datasheet's name) a step computing in each dtype runs its
products at; a dtype no datasheet rates (float32 among them) prices nothing."""


@dataclasses.dataclass(frozen=True)
class StepPrice:
    """A plan's step in seconds on its devices, priced from its compile (`device_price`, `streamed_price`):
    `compute`, its device's hardware FLOPs (a remat's recompute counted where it runs) over the part's dense
    peak; `link`, the bytes a step that streams its parameters from the host moves over the host link, at
    the link's rate (`GpuTarget.host_link_rate`); `host`, its update's bytes over the host's memory, the
    optimizer's on the host. The overlap rule: a streamed step fetches each layer's rows while it computes
    the last, forward and backward alike, so its link costs only what it runs past the compute; the update
    runs after the backward, beside nothing (`seconds`). Datasheet peaks, both sides alike: what ranks the
    two plans, not what either takes."""

    compute: float
    link: float = 0.0
    host: float = 0.0

    @property
    def seconds(self) -> float:
        return max(self.compute, self.link) + self.host

    def record(self) -> dict[str, JSON]:
        return {"compute": self.compute, "link": self.link, "host": self.host, "seconds": self.seconds}


class Unpriced(ValueError):
    """A plan no step price ranks (`StepPrice`): its part states no dense peak for its dtype or no host link,
    or its compile states no FLOPs. Said by name where the plans are chosen (`streamed_plan`)."""


def dense_rate(target, dtype) -> float:
    """The dense FLOPs a second `target`'s part runs a step computing in `dtype` at (`DENSE`); refused
    (`Unpriced`) where its datasheet rates none."""
    name = DENSE.get(jnp.dtype(dtype).name)
    rate = None if name is None else target.peaks.get(name)
    if rate is None:
        raise Unpriced(f"{target.device_kind} states no dense peak for {jnp.dtype(dtype).name} "
                       "(`GpuTarget.dense_peaks`)")
    return rate


def device_price(flops, target, dtype) -> StepPrice:
    """A plan's step on its devices alone (`StepPrice`): its compiled program's hardware FLOPs, a rung's
    recompute among them, over the dense rate."""
    if flops is None:
        raise Unpriced("the step's compile states no FLOPs (a loop of unstated length)")
    return StepPrice(flops.hardware / dense_rate(target, dtype))


def streamed_price(flops, target, dtype, parameters: int, update: float, *, read_once: int = 0) -> StepPrice:
    """A step that streams its `parameters` bytes, a device's shards, from the host each step
    (`Layout(host=("variables",))`): its device's program's FLOPs (`device_price`); over the link, every
    layer's rows fetched for the forward and again for the backward, which recomputes the forward from
    them, and the gradient back (three times the parameters' bytes), and `read_once` bytes more the forward
    reads and nothing differentiates (an EMA reference's weights); on the host, its `update` seconds, the
    update reading and writing its process's bytes (its parameters, their gradient and the optimizer's
    state) at its host's memory rate (`host_update_seconds`, `host_memory_rate`)."""
    link = target.host_link_rate
    if link is None:
        raise Unpriced(f"no host-link rate stated for {target.device_kind} (`GpuTarget.host_links`)")
    return dataclasses.replace(device_price(flops, target, dtype), link=(3 * parameters + read_once) / link,
                               host=update)


def host_update_seconds(updated: int, host_rate: float) -> float:
    """A streamed step's update on its host (`StepPrice.host`): its `updated` bytes read and written at
    `host_rate`. A streamed step is never priced under it, its link and compute aside (`StepPrice.seconds`):
    a device plan priced no dearer is chosen before the streamed step compiles
    (`dew.training.streaming.laid_start`)."""
    return 2 * updated / host_rate


_HOST_RATE: contextvars.ContextVar[float | None] = contextvars.ContextVar("dew_host_memory_rate",
                                                                         default=None)


def host_memory_rate() -> float:
    """This host's memory in bytes a second, read once a process: the least of three copies of a 256 MiB
    array's time, a read and a write of each byte. A host's memory is no GPU part's (the GPU's table states
    its link, `GpuTarget.host_links`), so it is measured where the run is, or the rate a block states for
    the host a run is priced for (`host_memory_stated`)."""
    stated = _HOST_RATE.get()
    return _host_memory_rate() if stated is None else stated


@contextlib.contextmanager
def host_memory_stated(rate: float) -> Iterator[None]:
    """Within the block, the host a run is priced for moves `rate` bytes a second (`host_memory_rate`): a
    host other than the one measuring, as a stated GPU is another than the one compiling
    (`dew.training.ahead.stand_in_gpus`)."""
    token = _HOST_RATE.set(rate)
    try:
        yield
    finally:
        _HOST_RATE.reset(token)


@functools.cache
def _host_memory_rate() -> float:
    import time

    import numpy as np

    source = np.ones(2 ** 25, np.float64)
    target = np.empty_like(source)
    seconds = []
    for _ in range(3):
        began = time.perf_counter()
        np.copyto(target, source)
        seconds.append(time.perf_counter() - began)
    return 2 * source.nbytes / min(seconds)
