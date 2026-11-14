"""Whether a training step fits its devices, and the rungs that make it fit.

The trainer compiles a step, reads its memory from the executable and each
device's free bytes, and climbs the objective's recomputation ladder or
splits the step's batch until the step fits everywhere
(`dew.training.planning`); these are those measurements and the ladder.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import Mesh, NamedSharding

from dew import communicators
from dew._devices import device_memory, gpu_free_bytes, xla_flag
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
from dew.objectives.base import Objective
from dew.state import TrainState
from dew.telemetry.instrumentation import compiled_memory

if TYPE_CHECKING:
    from dew.training.footprint import Footprint

_log = logging.getLogger(__name__)


COLLECTIVES = {
    'xla_gpu_enable_latency_hiding_scheduler': True,
    'xla_gpu_all_reduce_combine_threshold_bytes': 1 << 30,
    'xla_gpu_all_gather_combine_threshold_bytes': 1 << 30,
    'xla_gpu_reduce_scatter_combine_threshold_bytes': 1 << 27,
    'xla_gpu_enable_all_gather_combine_by_dim': False,
    'xla_gpu_enable_reduce_scatter_combine_by_dim': False,
}
"""NVIDIA JAX-Toolbox's collective options for a GPU step over several devices: the scheduler overlaps
each collective with compute, and the collectives combine into few large ones. Measured in
docs/performance.md, "Collectives over several GPUs". A step that gathers its layers' parameters combines
its all-gathers up to one layer's instead (`gathered_layer_bytes`)."""

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
    options.update(compiler_options(*objective.towers()))
    return options or None


def compiled_if_it_fits(program: jax.stages.Lowered, options: jax.stages.CompilerOptions | None,
                        refusals: list[RuntimeError], *, relower: Callable[[], jax.stages.Lowered],
                        first: report.Attempt | None = None) -> jax.stages.Compiled | None:
    """Return `program` compiled under `options`, or None if XLA refuses it for memory (`refused_for_memory`).

    It compiles through the kernel templates' reports (`report.compiled`): where a launch's kernel
    spills, its config is refused and the step lowered afresh (`relower`), its derivation taking the
    next config, and compiled once more. A refusal is appended to `refusals`. XLA:TPU checks a
    program's temporaries against HBM while it compiles and raises RESOURCE_EXHAUSTED instead of
    returning an executable whose memory `step_fits` could read, so the refusal means the step does
    not fit. For example, on a v6e, Qwen3-0.6B at 16 x 1024
    tokens asked for 38.47G of temporaries next to 31.24G of HBM. Any other
    compile error is raised. `first` is its first compile where another thread made it (`report.attempts`).
    """
    try:
        return report.compiled(program, options, relower=relower, first=first)
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
                                  mesh: Mesh, gathered: int | None = None
                                  ) -> jax.stages.CompilerOptions | None:
    """The options a step over microbatches of `micro` on `mesh` compiles with
    (`step_compiler_options`): by the tokens one device steps of a microbatch split over the
    mesh's batch axes, whether `variables` hold frozen weights it trains beside, and the plan's
    `gathered` bytes (`layer_gathers`)."""
    shards = math.prod(mesh.shape[axis] for axis in batch_axes(mesh))
    return step_compiler_options(objective, device_tokens(objective, micro, shards), FROZEN in variables,
                                 mesh.devices.size, gathered=gathered)


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
    trained = [index for index, entry in enumerate(entries) if entry.trained]
    recorded = json_value(rung.get("remat"), "rung remat")
    # One trained module records its rung alone, several a list in program order.
    records = [recorded] if len(trained) == 1 else recorded if isinstance(recorded, list) else []
    rungs = dict(zip(trained, records, strict=False))
    modules = [entry.module.restore_recompute(rungs[index])
               if index in rungs and isinstance(entry.module, Recomputing) else entry.module
               for index, entry in enumerate(entries)]
    if any(module is not entry.module for module, entry in zip(modules, entries, strict=True)):
        objective.substitute(modules)


def head_tile_of[LossT, EffectsT](objective: Objective[LossT, EffectsT]) -> tuple[int, int] | None:
    """The tile the vocabulary head the objective trains computes its
    backward in (`ProgramModule.head_tile`), or None for a head that keeps
    its whole logits or an objective that trains no head: what `tile_head`
    moves and `restore_head` sets."""
    return next((entry.head_tile for entry in objective.program_key()
                 if entry.trained and entry.head_tile is not None), None)


def head_fused_of[LossT, EffectsT](objective: Objective[LossT, EffectsT]) -> bool:
    """Whether the vocabulary head the objective trains takes its gradient in the forward
    (`ProgramModule.head_fused`): what `unfuse_head` moves and `restore_head` sets."""
    return any(entry.head_fused for entry in objective.program_key() if entry.trained)


def recompute_record[LossT, EffectsT](objective: Objective[LossT, EffectsT]) -> JSON:
    """What the modules the objective trains recompute in their backward
    pass, as a checkpoint records it (`Recomputing.recompute_record`): one
    module's record, a list of several in program order, None for a module
    that does not say and for an objective that trains none."""
    records = [entry.module.recompute_record() if isinstance(entry.module, Recomputing) else None
               for entry in objective.program_key() if entry.trained]
    return records[0] if len(records) == 1 else records or None


def rung_record[LossT, EffectsT](objective: Objective[LossT, EffectsT], xla_defaults: bool, *,
                                 narrow: bool) -> dict[str, JSON]:
    """The rung of the fit ladder a step compiles at, as a checkpoint records
    it: the objective's head tile and whether that head is fused, its trained
    modules' remat (`recompute_record`), whether the step keeps XLA's
    default options, the half rung above each, and whether it reads the
    parameters through narrow copies (`dew.training.narrow`), the ladder's
    first rung. Each decides the program a step runs."""
    tile = head_tile_of(objective)
    return {'head_tile': None if tile is None else list(tile), 'head_fused': head_fused_of(objective),
            'remat': recompute_record(objective), 'xla_defaults': xla_defaults, 'narrow': narrow}


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
    stats = compiled_memory(executable)
    if stats is None:
        return None
    return admitted(devices, stats.argument - built, stats.output - stats.alias + held, stats.temp,
                    placed=placed)


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


def derived_headroom(footprint: Footprint | None, devices: Sequence, built: int = 0, *,
                     placed: bool = True) -> tuple[int, int] | None:
    """The least and most bytes the step would leave free, pooled over processes: its upper and lower
    temporary bounds admitted beside its resident and held bytes by the same allocator rule as a
    compiled step (`admitted`). One memory reading serves both bounds. None where no process can judge
    a footprint. A nonnegative least fits; a negative most refuses; an interval spanning zero leaves the
    compile to decide. A footprint with one bound keeps the previous single judgment."""
    from dew._coordination import from_every_process

    bounds: tuple[int, int] | None = None
    if footprint is not None and (upper := footprint.temporaries) is not None:
        lower = footprint.lower_temporaries
        assert lower is not None
        resident = sum(footprint.resident.values()) - built
        memory = [device_memory(device, resident) for device in devices]
        least = _admitted(devices, memory, footprint.held, upper, placed=placed)
        most = least if lower == upper else _admitted(devices, memory, footprint.held, lower, placed=placed)
        if least is not None and most is not None:
            bounds = least, most
    pooled = [value for value in from_every_process(bounds) if value is not None]
    return None if not pooled else (min(value[0] for value in pooled), min(value[1] for value in pooled))


def admitted(devices: Sequence, resident: int, beside: int, temporaries: int, *,
             placed: bool = True) -> int | None:
    """The allocator's admission of a step, compiled (`step_headroom`) or derived (`derived_headroom`): the
    bytes the tightest of `devices` has to spare once the step's `temporaries`, in one block, twice where
    its allocator strands them (`stranding`), are placed beside `beside` more bytes, with `resident` bytes
    in use already; None where a device states no memory. A compile-only GPU's stated memory holds
    `resident` as it would (`device_memory`). Where the state is not `placed` yet, the devices' free bytes
    in all hold them (`available`), not one block of them."""
    memory = [device_memory(device, resident) for device in devices]
    return _admitted(devices, memory, beside, temporaries, placed=placed)


def _admitted(devices: Sequence, memory: Sequence[Mapping[str, int]], beside: int,
              temporaries: int, *, placed: bool = True) -> int | None:
    """The headroom at a fixed reading of the allocators, shared by both ends of a derived interval."""
    if not stated(memory):
        return None
    room = placeable if placed else available
    return min(room(m, unclaimed(d, m)) - beside - temporaries * stranding(d, m)
               for d, m in zip(devices, memory, strict=True))


def stated(memory: Sequence[Mapping[str, int]]) -> bool:
    """Whether every device's `memory` (`device_memory`) states its limit and its use, which a fit check
    reads; a CPU's states none, so no step's headroom on it is known ahead of its run."""
    return all('bytes_limit' in m and 'bytes_in_use' in m for m in memory)


def judged(device) -> bool:
    """Whether a step's footprint derived for `device` is judged against its memory (`derived_headroom`):
    where it states its memory (`device_memory`), which a CPU does not, and a device with none attached (a
    compile-only GPU, `dew.training.ahead.compile_only_gpus`) where its memory was stated. The plan's walk
    and the footprint's trace read this one answer (`FitSearch.walked`)."""
    return stated([device_memory(device)])


def stranding(device, memory: Mapping[str, int]) -> int:
    """How many times its temporaries a step needs room for on `device`, its allocator stating `memory`:
    2 where it can strand them (`strands_temporaries`), else 1."""
    return 2 if strands_temporaries(device.platform, memory) else 1


def local_devices(mesh: Mesh) -> list:
    """This process's devices of `mesh`, whose memory the fit check reads."""
    process = jax.process_index()
    return [device for device in mesh.devices.flat if device.process_index == process]


class FirstRunFailed(RuntimeError):
    """A newly compiled step's first run its devices refused: the record of the check that said it fits
    (`fitted`, `memory_record`) and what this process's devices of its mesh hold now (`now`), each GPU's
    bytes free outside XLA's pool (`outside_pool`) among it."""

    def __init__(self, step: int, fitted: JSON, now: list[JSON]):
        self.fitted, self.now = fitted, now
        super().__init__(
            f"the training step compiled for step {step} failed on its first run. It was checked to fit: "
            f"{json.dumps(fitted)}; the devices now hold {json.dumps(now)}. 'outside' is a GPU's bytes free "
            "outside XLA's pool, whose limit is its bytes_limit (XLA_PYTHON_CLIENT_MEM_FRACTION of the GPU): "
            "a collective's communicators (NCCL) take their buffers there, beside cuBLAS's workspaces and "
            "the kernels loaded, and a lower fraction leaves them more")


def first_run(loss: jax.Array, step: int, fitted: JSON, mesh: Mesh) -> dict[str, dict[int, int]]:
    """Wait for a newly compiled step's first run, and raise its failure on the devices as the step's
    (`FirstRunFailed`). A first run that succeeds returns, and logs, each GPU's bytes free outside XLA's
    pool as the fit check read them (`fitted`) and now: what the devices took there since, every
    process's and a pool's growth counted, the step's collectives' communicators (NCCL) among it."""
    devices = local_devices(mesh)
    try:
        loss.block_until_ready()
    except jax.errors.JaxRuntimeError as error:
        now: list[JSON] = [{"id": device.id, "outside": outside_pool(device),
                            **{key: value for key, value in (device.memory_stats() or {}).items()
                               if key in ('bytes_in_use', 'pool_bytes', 'largest_free_block_bytes')}}
                           for device in devices]
        raise FirstRunFailed(step, fitted, now) from error
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
    free = memory['bytes_limit'] - memory['bytes_in_use']
    if 'pool_bytes' not in memory:
        return free
    growth = memory['bytes_limit'] - memory['pool_bytes']
    if unclaimed is not None:
        growth = min(growth, unclaimed)
    return min(free, max(memory['largest_free_block_bytes'], growth))


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


def pooled_headroom(headroom: int | None) -> int | None:
    """The pool's headroom, the same on every process: the tightest any process states of its own devices
    (`headroom`, a compile's measure or a derivation's judgment), which reads only its own devices' memory;
    a process that states none does not affect it, and None where none states one."""
    from dew._coordination import from_every_process

    return min((value for value in from_every_process(headroom) if value is not None), default=None)


def fits_everywhere(headroom: int | None) -> bool:
    """Whether every process's headroom is non-negative, with the same answer on every process
    (`pooled_headroom`): a pool where no process states one fits."""
    pooled = pooled_headroom(headroom)
    return pooled is None or pooled >= 0


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
    local = [device for device in mesh.devices.flat if device.process_index == jax.process_index()]
    headroom = None if executable is None else step_headroom(executable, local, held=held, built=built,
                                                             placed=placed)
    return fits_everywhere(-1 if executable is None else headroom), headroom


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
    return climbed(objective, base) is not None


def headed_more[LossT, EffectsT](objective: Objective[LossT, EffectsT],
                                  base: tuple[int, int] | None = None) -> bool:
    """Move the objective one rung up its ladder where the rung is its head's (`climbed`'s without its
    remat), and return whether there was one: a run that states its remat climbs no further."""
    return climbed(objective, base, remat=False) is not None


def climbed[LossT, EffectsT](objective: Objective[LossT, EffectsT],
                             base: tuple[int, int] | None = None, *, remat: bool = True) -> str | None:
    """Move the objective one rung up its ladder as `recompute_more` does, and say what it moved to,
    None where there was no rung: the words a search that compiles again there logs
    (`dew.training.planning.FitSearch.climb`), which a derivation's walk, compiling nothing, does not.
    Without `remat` the ladder ends at its head's rungs (the generation's tile for the whole logits, then
    that tile unfused): a run that states its remat keeps it (docs/design/training.md, every choice can be
    overridden)."""
    if head_tile_of(objective) == base:
        moved = objective.tile_head()
        if moved is not None:
            return f"with the whole logits kept; compiling it again with {moved}"
    moved = objective.unfuse_head()
    if moved is not None:
        return f"with the head fused; compiling it again with {moved}"
    current = recompute_record(objective)
    if not remat or not remat_more(objective):
        return None
    objective.restore_head(base, fused=True)
    return f"under remat {current!r}; compiling it again under {recompute_record(objective)!r}"


def remat_half[LossT, EffectsT](objective: Objective[LossT, EffectsT]) -> list | None:
    """Move each module the objective trains to its half rung, where it has
    one (`Recomputing.recompute_half`), and return the modules it held, which
    `objective.substitute` restores; None where no module has one."""
    entries = objective.program_key()
    half = [entry.module.recompute_half() if entry.trained and isinstance(entry.module, Recomputing)
            else None for entry in entries]
    if all(module is None for module in half):
        return None
    held = [entry.module for entry in entries]
    objective.substitute([entry.module if module is None else module
                          for module, entry in zip(half, entries, strict=True)])
    return held


def remat_more[LossT, EffectsT](objective: Objective[LossT, EffectsT]) -> bool:
    """Move each module the objective trains one rung up its own remat ladder
    where it goes on (`Recomputing.recompute_more`), and return whether any
    moved."""
    return moved(objective, lambda module: module.recompute_more() if isinstance(module, Recomputing)
                 else None)


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


