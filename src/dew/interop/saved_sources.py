"""A saved Dew run as the source another run starts from.

A source names a saved run by its directory, and `pinned_run` resolves it, for every loader, to the
run's checkpoint at the step its selection pins. `dew export` writes that checkpoint as a release
(`export_run`, `run_plan`): a task that publishes its own released layout writes it (a decision run's,
`ReleasedTask`), any other run its model in the source layout its family reads
(`Pretrained.from_saved`), and a run whose model no layout states its native release. A language
model or decision run that starts from a saved run reads that release (`released_run`): every loader
takes a source by its directory, so the run's load options (its compute and storage dtypes, its mesh,
its source options, an adapter) apply to it as to any release, and its first weights are the export's.
Dew writes no release of a diffusion run (`samples_images`), and a diffusion run starts from the parts
and policy of the run's own sampling task instead (`pipeline_of_run`).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import jax
from etils import epath
from jax.sharding import Mesh

from dew._builds import dew_cache_dir
from dew._code import whole_digest
from dew._coordination import agreed_same, from_every_process, reading_home, seen_by_every_process
from dew._files import staged, write_atomically
from dew._records import JSON, json_value, mapped_strings, strings_of
from dew.checkpoints import RUN_FILE
from dew.registry import from_record, to_record, trust_sources, trusts_source

if TYPE_CHECKING:
    from flax import linen as nn
    from jax.typing import DTypeLike

    from dew.batch import Variables
    from dew.diffusion.process import Process
    from dew.diffusion.schedules.source import SourceGrid
    from dew.distributed.placement import Layout, MeshSpec
    from dew.inference.tasks import SavedRun
    from dew.inputs import InputSpec
    from dew.interop.pretrained import Pretrained
    from dew.interop.releases import ReleasePlan
    from dew.lora import LoRA
    from dew.nn.attention import AttentionImpl
    from dew.nn.autoencoders import AutoEncoder
    from dew.sampling.guidance import Guidance
    from dew.sampling.solvers.common import Solver


def export_run(run: SavedRun, destination: str | Path, *, ema: bool | None = None,
               mesh: MeshSpec | Mesh | None = None, layout: Layout | None = None,
               publishes: bool = True) -> None:
    """Write `run`'s pinned checkpoint (`SavedRun.pinned`) to `destination` as its release (`run_plan`),
    its EMA weights where `ema`. Its weights restore where `mesh` and `layout` place them, so no device
    holds more of them than a load does, and the writer reads them home a tensor at a time as it writes
    them, one shard of the release at a time (`save_sharded`). On a pool every process runs this same
    writer, reading each tensor that spans the pool's processes home together (`reading_home`), so each
    host holds a shard of the release, not the model (ledger 2874). One that does not `publish` reads
    every tensor as the one that does and writes no weights to `destination`, only the release's small
    files."""
    # Restored before the writer reads its tensors home, each as the writer reaches it.
    plan = run_plan(run, ema=ema, mesh=mesh, layout=layout)
    with reading_home("saved run export", writes=publishes):
        plan.write(destination)


def run_plan(run: SavedRun, *, ema: bool | None = None, mesh: MeshSpec | Mesh | None = None,
             layout: Layout | None = None) -> ReleasePlan:
    """`run`'s release (`dew.interop.releases`): its task's own where it publishes one that states the run's
    model (`ReleasedTask.unstated`, a decision task's format), else the model in its family's source layout
    where the family's layout states it (`dew.interop.layouts.source_layout`, `Pretrained.from_saved`), else
    the run's native release, its step's declaration and variables beside the files its task keeps by the
    step, which `dew.pipeline` loads as the run, and which states why no layout writes the model. Which of
    them is read off the run's record before its weights are."""
    from dew.inference.tasks import saved_model
    from dew.interop.layouts import Unwritable, source_layout
    from dew.interop.pretrained import Pretrained
    from dew.interop.releases import ReleasePlan, native_plan
    from dew.objectives.base import ReleasedTask

    task = run.objective.saved_task
    plan: ReleasePlan | Unwritable
    if task is not None and issubclass(task, ReleasedTask):
        why = task.unstated(run)
        # Under the numerics its step trained in, as every load of the run builds it (`SavedRun.formed`).
        plan = why if why is not None else run.formed(
            task.from_saved(run, ema=ema, mesh=mesh, layout=layout)).release_plan()
    elif "model" in run.record:
        stated = source_layout(saved_model(run.record, None).build())
        plan = stated if isinstance(stated, Unwritable) else Pretrained.from_saved(
            run, ema=ema, mesh=mesh, layout=layout).release_plan()
    else:
        plan = Unwritable("its run records no model of Dew's: its task rebuilds its own")
    if isinstance(plan, ReleasePlan):
        return plan
    records = step_records(json_value(run.record, "artifact"), run.directory, run.step)
    return native_plan(run.variables(ema=ema, mesh=mesh, layout=layout), run.record, run.step,
                       why=plan.reason, averaged=run.averaged(ema),
                       records=[(path, path.relative_to(run.directory)) for path in records])


RELEASES = "saved_runs"
"""The folder of Dew's cache that holds each saved run's release, one directory a key (`released_run`)."""

HELD_FILE = "dew_saved_run.json"
"""A cache release's own record of the saved run it holds (`released_run`): the run's directory, and the
step and EMA choice it was exported at, which a run that reads the release carries beside the paths it names
within it (`release_sources`)."""

RELEASE_SOURCES = "releases"
"""The field of a step's inference declaration mapping each cache release its paths lie within to the saved
run that release holds (`release_sources`), which a host lacking the release writes its own from
(`releases_here`)."""


def is_saved_run(name_or_dir: str | Path) -> bool:
    """Whether `name_or_dir` is a saved Dew run's directory, on this host or remote storage
    (`etils.epath`): one holding its run record (`RUN_FILE`)."""
    path = epath.Path(name_or_dir)
    return path.is_dir() and (path / RUN_FILE).is_file()


def step_records(artifact: dict[str, JSON] | list[JSON] | str | int | float | bool | None, directory: str,
                 step: int) -> tuple[epath.Path, ...]:
    """The files the task of the objective `artifact`, a step's inference declaration
    (`Checkpoints.artifact`), names keeps by the `step` of the run in `directory`
    (`SavedRunTask.step_records`), and the tokenizer the run holds (`dew.data.text.held_place`), which
    a release carries at the same place; none for a step that declares no objective, an objective that
    names no task, or a task that keeps none."""
    from dew.data.text import HELD_FILES, held_place
    from dew.inference.tasks import SavedRunTask, declared_objective

    if not isinstance(artifact, dict) or "objective" not in artifact:
        return ()
    held: tuple[epath.Path, ...] = ()
    files = artifact.get("tokenizer_files")
    hashed = {str(path): str(digest) for path, digest in files.items()} if isinstance(files, dict) else {}
    if hashed and (held_place(directory, hashed) / HELD_FILES).exists():
        place = held_place(directory, hashed)
        held = (*(place / path for path in hashed), place / HELD_FILES)
    task = declared_objective(artifact).saved_task
    if task is None or not issubclass(task, SavedRunTask):
        return held
    return (*task.step_records(directory, step), *held)


def _fingerprint(run: SavedRun, ema: bool | None) -> str:
    """What an export of `run`'s pinned step with `ema` is a function of: the run's location, its step,
    the EMA choice, the code that writes it (`dew._code.whole_digest`, ledger 2775; a process whose source
    changed while it ran keys an export of its own, which no other reads), the contents of the run's
    own records beside its checkpoints (its run record and a task's assets, a decision task's
    calibration among them) and those its task keeps by the step (`SavedRunTask.step_records`), and the
    step's metadata files, by size and write time, which a new save of the step rewrites."""
    root = epath.Path(run.directory)
    code = whole_digest() or f"changed source, process {os.getpid()} at {time.time_ns()}"
    digest = hashlib.sha256(json.dumps([str(root), run.step, ema, code]).encode())
    for path in sorted((path for path in root.iterdir() if path.is_file()), key=str):
        digest.update(f"{path.name}\0".encode() + path.read_bytes())
    for path in step_records(json_value(run.record, "artifact"), run.directory, run.step):
        digest.update(f"{path.relative_to(root)}\0".encode() + path.read_bytes())
    for path in sorted((root / str(run.step)).iterdir(), key=str):
        if path.is_file():
            stat = path.stat()
            digest.update(f"{path.name}\0{stat.length}\0{stat.mtime}\0".encode())
    return digest.hexdigest()[:24]


def pinned_run(name_or_dir: str | Path, *, step: int | str | None = None, ema: bool | None = None,
               trust: Sequence[str] = (), mesh: MeshSpec | Mesh | None = None,
               layout: Layout | None = None) -> SavedRun | None:
    """The saved Dew run `name_or_dir` names (`is_saved_run`) at its checkpoint `step` (`SavedRun.pinned`:
    a step, `latest`, `best`, `best:<metric>`, or the one its task loads by default), on which every
    process of a pool agrees; None for a source that is no saved run, beside which a step or an EMA
    choice is refused: it selects nothing. Every loader that takes a saved run resolves it here."""
    from dew.inference.tasks import SavedRun

    if not is_saved_run(name_or_dir):
        if step is not None or ema is not None:
            raise ValueError(f"{name_or_dir} is no saved Dew run (no {RUN_FILE}), so a step or an EMA choice "
                             "selects nothing of it; name a run's directory, or drop them")
        return None
    # The pool agrees on the step from its records alone, then writes what that step reads (`SavedRun.here`)
    # where the load places the weights.
    agreed = agreed_same("saved run step", lambda: SavedRun.selected(str(name_or_dir), step, trust).step)
    return SavedRun.pinned(str(name_or_dir), agreed, trust, mesh=mesh, layout=layout)


def samples_images(run: SavedRun) -> bool:
    """Whether `run` is a diffusion run, its task a `TextToImage`: Dew writes no release of one, and a run
    starts from its task's parts (`pipeline_of_run`)."""
    from dew.inference.images import TextToImage

    task = run.objective.saved_task
    return task is not None and issubclass(task, TextToImage)


@dataclass(frozen=True)
class RunPipeline:
    """A saved diffusion run as the source a diffusion run starts from (`pipeline_of_run`): its sampling
    task's denoiser `model`, `variables`, `process`, `inputs` and `autoencoder`, and the `solver`,
    `steps`, `guidance` and `grid` it samples by, which a `DiffusionObjective` trains as it trains a
    published pipeline (`PipelineSource`). Its variables are held as given, so an adapter's split of
    them stays split for training, where a task reads a split tree whole."""

    model: nn.Module
    variables: Variables
    process: Process
    inputs: InputSpec
    autoencoder: AutoEncoder | None
    solver: Solver
    steps: int
    guidance: Guidance | None
    grid: SourceGrid | None
    grid_process: bool
    """Whether `process` is `grid`'s own training process at its tokens, as the run's record states
    (`ImageSettings.grid_process`)."""

    def adapt(self, lora: LoRA, *, key: int | jax.Array) -> RunPipeline:
        """This source with `lora` attached to its denoiser and its factors drawn (`LoRA.apply`), as
        `Pretrained.adapt` adapts a pipeline: the factors under `params` and the denoiser's weights under
        `frozen`, so an objective over it trains the factors alone, and since B starts at zero it computes
        what this source does."""
        adapter = lora.apply(self.model, self.variables, key=key)
        return replace(self, model=adapter.model, variables=adapter.variables)


def pipeline_of_run(run: SavedRun, checkpoint: str, *, dtype: str, param_dtype: str | None,
                    revision: str | None, attention_impl: AttentionImpl, size: tuple[int, ...] | None,
                    variables: Variables | None, mesh: MeshSpec | Mesh | None, layout: Layout | None,
                    text: bool, ema: bool | None) -> RunPipeline:
    """The diffusion run `run` (`checkpoint`) as a diffusion run starts from it: its own task's parts and
    policy (`TextToImage.from_saved`), computing in `dtype`, over its tree as its checkpoint holds it
    (`restored_tree`: an adapter's split of it kept, so its base stays frozen), stored in `param_dtype` (as
    saved where None) where `mesh` and `layout` place it, or over supplied `variables`, the starting run's
    own tree, held as given; its denoiser attending through `attention_impl` where its class takes a
    kernel.

    A run of another task, a revision of a directory, and `text=False`, which the restore of a run's
    tree cannot skip, are refused, as is a kernel other than `auto` for a denoiser that takes none. A
    `size` other than the run's own binds its inputs, conditioning, grid and process at it, as a published
    pipeline's are bound (`pipeline_assembly.bound_at_size`, ledger 2792). A run that tuned its autoencoder
    (`end_to_end`) is started from with its tuned autoencoder, which the started run records and holds
    (`TunedAutoEncoder`, ledger 2791)."""
    from dew._records import record
    from dew.diffusion.schedules.source import SourceGrid
    from dew.inference.images import ImageSettings, TextToImage, restored_tree
    from dew.inference.tasks import saved_settings
    from dew.interop.pipeline_assembly import bound_at_size
    from dew.objectives.diffusion.objective import without_loss_heads

    if not samples_images(run):
        raise ValueError(f"{checkpoint} is a {run.objective.__name__} run, which Pretrained.load reads; a "
                         "diffusion run starts from a published pipeline or a saved diffusion run")
    if revision is not None:
        raise ValueError(f"{checkpoint} is a saved run's directory, which has no revision {revision!r}")
    if not text:
        raise ValueError(f"{checkpoint} is a saved run, whose tree restores its text encoder with its "
                         "denoiser; text=False skips none of it")
    settings = saved_settings(run.record, ImageSettings)
    if variables is None:
        restored = restored_tree(run, ema=ema, mesh=mesh, layout=layout, param_dtype=param_dtype)
        task = TextToImage.from_saved(run, dtype=dtype, variables=restored)
        # The tree as the checkpoint holds it (an adapter's frozen base split from its factors), its loss
        # heads the run's own; a tuned autoencoder as the task holds it, its weights and running
        # statistics under `autoencoder` (`EndToEnd.tuned`).
        tree = without_loss_heads(restored)
        if settings.end_to_end is not None:
            tree = {**tree, "autoencoder": task.variables["autoencoder"]}
    else:
        # The starting run's own tree, which holds the task's published form.
        task = TextToImage.from_saved(run, dtype=dtype, variables=variables, published=True)
        tree = variables
    model = task.model
    if "attention_impl" in {field.name for field in fields(model)}:
        model = model.clone(attention_impl=attention_impl)
    elif attention_impl != "auto":
        raise ValueError(f"{type(model).__name__} takes no attention_impl: the attention its record states "
                         "is its run's")
    # The grid as its record states it, the one `from_saved` built the task's from.
    grid = None if settings.grid is None else from_record(SourceGrid, record(settings.grid, "grid"))
    inputs, process = task.inputs, task.process
    if size is not None:
        # At the size the run's data has, as a published pipeline binds its own (ledger 2792).
        inputs, process, grid = bound_at_size(inputs, process, grid, task.autoencoder, tuple(size),
                                              grid_process=task.grid_process)
    return RunPipeline(model, tree, process, inputs, task.autoencoder, task.solver, task.steps, task.guidance,
                       grid, task.grid_process)


def released_run(name_or_dir: str | Path, *, step: int | str | None = None, ema: bool | None = None,
                 trust: Sequence[str] = (), mesh: MeshSpec | Mesh | None = None,
                 layout: Layout | None = None) -> str | Path:
    """`name_or_dir` as a language model's or a decision task's loader reads it: itself, or where it is a
    saved Dew run (`pinned_run`), the release its checkpoint at `step` with `ema` exports to
    (`export_run`), written once under Dew's cache by what the export reads (`_fingerprint`) and read
    again by every later load of it. A diffusion run, which has no release (`samples_images`), is
    refused before anything is restored.

    Every process of a pool agrees on whether any of them lacks the release, and then all of them export
    it, each writing its own cache, as a restore is collective; the export restores where the load
    would place it (`mesh`, `layout`), so it holds no more of the weights on a device than the load
    does. A load without weights (a resume's, which its restore fills) reads the release's records and
    headers, so a host whose cache lacks the release (a resumed job on another node, a cleared cache, a
    run whose records changed) writes it as the first load did (ledger 2774)."""
    run = pinned_run(name_or_dir, step=step, ema=ema, trust=trust, mesh=mesh, layout=layout)
    if run is None:
        return name_or_dir
    if samples_images(run):
        raise ValueError(f"{name_or_dir} is a diffusion run, which Dew writes no release of: a diffusion run "
                         "starts from it (load_diffusion_source), and TextToImage.from_run serves it")
    release = Path(dew_cache_dir()) / RELEASES / _fingerprint(run, ema)
    lacking = any(from_every_process(not release.is_dir()))
    if not lacking:
        if not (release / HELD_FILE).is_file():
            # A release written before releases recorded what they hold: its key is a function of the run,
            # the step and the EMA choice (`_fingerprint`), so it holds these.
            _write_held(release, run, ema)
        return release
    # One that holds the release already keeps it (`staged`). A key reads each host's own checkpoint
    # files (`_fingerprint`), which copies of one checkpoint on several hosts may stamp apart, so a pool
    # shares a release only where every process asks for the one.
    pooled_export(run, release, ema=ema, mesh=mesh, layout=layout, staging=True)
    return release


@dataclass(frozen=True)
class Held:
    """The saved run a cache release holds, as its own record states it (`HELD_FILE`): the run's directory,
    and the step and EMA choice it was exported at (`released_run`)."""

    run: str
    step: int
    ema: bool | None

    @classmethod
    def of(cls, run: SavedRun, ema: bool | None) -> Held:
        """What `run`'s release with `ema` holds, its directory absolute where it is local."""
        return cls(run.directory if "://" in run.directory else os.path.abspath(run.directory), run.step, ema)

    @classmethod
    def read(cls, value: JSON) -> Held:
        """A release's record `value` (`record`), read by its fields' declarations (`registry.from_record`,
        each scalar of its declared type), refused where it is not one."""
        return from_record(cls, value)

    def record(self) -> JSON:
        """This as a release's record (`registry.to_record`, which `read` reads)."""
        return to_record(self, Held)


def _write_held(release: Path, run: SavedRun, ema: bool | None) -> None:
    """Write `release`'s own record of the saved run it holds (`HELD_FILE`, `Held`), whole or not at all."""
    write_atomically(release / HELD_FILE, json.dumps(Held.of(run, ema).record()))


def release_sources(record: Mapping[str, object]) -> dict[str, JSON]:
    """The cache releases `record`, a step's inference declaration, names paths within, each mapped to the
    saved run it holds as its own record states it (`HELD_FILE`): a path within a directory of this host's
    cache's `RELEASES` that holds that record names it; any other path names none. A run's checkpoints
    carry it (`RELEASE_SOURCES`, `with_release_sources`), so a host lacking a release writes its own of the
    same saved run (`releases_here`)."""
    cache = Path(dew_cache_dir()) / RELEASES
    sources: dict[str, JSON] = {}
    named = {key: json_value(field, key) for key, field in record.items() if key != RELEASE_SOURCES}
    for name in strings_of(named):
        path = Path(name)
        if not path.is_absolute() or not path.is_relative_to(cache) or path == cache:
            continue
        root = cache / path.relative_to(cache).parts[0]
        if str(root) not in sources and (root / HELD_FILE).is_file():
            sources[str(root)] = Held.read(json.loads((root / HELD_FILE).read_text())).record()
    return sources


def with_release_sources(artifact: JSON) -> JSON:
    """`artifact`, a step's inference declaration, with the cache releases it names paths within mapped to
    the saved runs they hold (`release_sources`), where it names any: what a run's checkpoint carries
    (`Trainer`)."""
    match artifact:
        case {"objective": _} if sources := release_sources(artifact):
            return {**artifact, RELEASE_SOURCES: sources}
        case _:
            return artifact


def releases_here(record: Mapping[str, object], *, mesh: MeshSpec | Mesh | None = None,
                  layout: Layout | None = None, trust: Sequence[str] = ()) -> Mapping[str, object]:
    """`record`, a step's inference declaration, with each path it names within a cache release its map
    lists (`RELEASE_SOURCES`) that this host lacks named within this host's release of the saved run that
    release holds, which `released_run` writes here as the first load did, where the load places it
    (`mesh`, `layout`): a run's processor and tokenizer lie in the release of the saved run it started from,
    under its training host's cache, which a resume on another host, or a host serving it, lacks. The
    processes of a pool agree on which releases any of them lacks, and then all write theirs of those, in
    one order; a release every process holds is read where it lies. A saved run a lacking release holds
    that this host lacks too is refused, naming it; a path within a release the map does not list is the
    record's as it stands."""
    listed = record.get(RELEASE_SOURCES)
    if not isinstance(listed, dict):
        return record
    lacking = [root for root in sorted(listed) if any(from_every_process(not Path(root).is_dir()))]
    if not lacking:
        return record
    held = {root: Held.read(json_value(listed[root], root)) for root in lacking}
    missing = sorted(f"{source.run} at step {source.step} (EMA {source.ema})"
                     for source in held.values() if not is_saved_run(source.run))
    if any(from_every_process(bool(missing))):
        raise FileNotFoundError(
            f"this step reads the releases of saved runs that this host lacks, and the runs as well "
            f"({missing or 'on another process of the pool'}): place each run where its record names it")
    fields = {key: json_value(field, key) for key, field in record.items() if key != RELEASE_SOURCES}
    names = list(strings_of(fields))
    renamed: dict[str, str] = {}
    sources = {root: json_value(source, root) for root, source in listed.items() if root not in held}
    for root, source in held.items():
        # The trust its recorded release had, or its run has, taken before the release is written (an
        # absent path's trust is a Hub name's, `dew.registry.trusts_source`).
        trusted = root in trust or trusts_source(root) or trusts_source(source.run)
        here = Path(released_run(source.run, step=source.step, ema=source.ema, trust=trust, mesh=mesh,
                                 layout=layout))
        if trusted:
            trust_sources((str(here),))
        sources[str(here)] = source.record()
        if str(here) != root:
            renamed.update({name: str(here / Path(name).relative_to(root)) for name in names
                            if Path(name).is_relative_to(root)})
    def here_named(text: str) -> str:
        return renamed.get(text, text)

    return {**{key: mapped_strings(field, here_named) for key, field in fields.items()},
            RELEASE_SOURCES: sources}


def pooled_export(run: SavedRun, destination: Path, *, ema: bool | None = None,
                  mesh: MeshSpec | Mesh | None = None, layout: Layout | None = None,
                  staging: bool = False) -> None:
    """Write `run`'s release to `destination` from every process of a pool, or from one (`export_run`):
    every process restores and writes, the pool reading its tensors home together. Where the pool shares
    the destination, every process naming the one and seeing one directory, the first publishes it and
    the others read every tensor as it does, writing only the release's small files, to a directory
    discarded; elsewhere each publishes its own. A `staging` export publishes through `staged`, which
    keeps a release written already; another writes in place, as `dew export` does. Every process
    returns once the release is published (ledgers 2874, 2992)."""
    shared = (jax.process_count() > 1 and len(set(from_every_process(destination.name))) == 1
              and all(seen_by_every_process(str(destination.parent))))
    if shared and jax.process_index() != 0:
        with tempfile.TemporaryDirectory() as discarded:
            export_run(run, discarded, ema=ema, mesh=mesh, layout=layout, publishes=False)
    elif staging:
        with staged(destination) as staged_at:
            export_run(run, staged_at, ema=ema, mesh=mesh, layout=layout)
            # A cache release states the saved run it holds, which a run that reads it carries.
            _write_held(staged_at, run, ema)
    else:
        export_run(run, destination, ema=ema, mesh=mesh, layout=layout)
    from_every_process(None)  # every process reads the release once it is published


def native_release(source: str | Path) -> bool:
    """Whether `source` is a local directory holding a run's native release (`dew.inference.loading.
    released_model`), whose files no family's loader reads: a saved run's export where no family's layout
    states its model (`run_plan`) among them."""
    from dew.inference.loading import released_model

    return Path(source).is_dir() and released_model(epath.Path(source))


def native_source[Loaded: Pretrained](kind: type[Loaded], source: str | Path, *, mesh: MeshSpec | Mesh | None,
                                      layout: Layout | None, dtype: DTypeLike | None,
                                      param_dtype: DTypeLike | Literal["auto"] | None, on_host: bool,
                                      weights: bool, attention_impl: str, max_seq_len: int | None,
                                      building: Mapping[str, object]) -> Loaded:
    """The native release in `source` as the `kind` of source a run starts from: its run's step
    (`dew.inference.models.ReleasedRun`) as its task's bundle where its task declares one
    (`ReleasedTask.source_of`, a decision task's), else as its model's (`Pretrained.from_saved`: any
    token task's run, its model in its native layout where no family's writes it; row 4785), placed on
    `mesh` under `layout`, or on the host (`on_host`), every tree of it, computing in `dtype` and stored in
    `param_dtype` ("auto": as the release stores it), its weights read where `weights` asks, else their
    shapes alone from the files' headers (`ReleasedRun.shapes`). The load's `attention_impl` (but "auto")
    is the recorded model's own field (`ModelConfig.with_declared`), and its `max_seq_len` the cache the
    model is reloaded at, as every task sizes it (`dew.inference.tasks.cache_sized`: a learned position
    table keeps its rows). A load option that builds a family's model another way (`building`, set where
    neither None nor False, a 0 among them) is refused, as the release builds the model its run recorded,
    and so is a bundle of another kind than `kind`."""
    from dataclasses import replace

    from dew.inference.models import ReleasedRun
    from dew.inference.tasks import cache_sized, saved_model
    from dew.interop.pretrained import Pretrained
    from dew.objectives.base import ReleasedTask
    from dew.registry import ModelConfig

    built = [name for name, value in building.items() if value is not None and value is not False]
    if built:
        raise ValueError(f"{source} is a run's native release, which builds the model its run recorded; "
                         f"load it without {', '.join(built)}")
    run = replace(ReleasedRun.of(str(source)), shapes=not weights)
    if attention_impl != "auto" or max_seq_len is not None:
        # The recorded model as the load asks it, which every builder of the run reads off its record.
        model = saved_model(run.record, None)
        if attention_impl != "auto":
            model = model.with_declared("attention_impl", attention_impl)
        if max_seq_len is not None:
            model = ModelConfig.from_model(cache_sized(model.build(), max_seq_len))
        run = replace(run, record={**run.record, "model": to_record(model, ModelConfig)})
    task = run.objective.saved_task
    # On the host, read into host arrays from storage, so no accelerator holds the weights on the way.
    placed = {"mesh": mesh, "layout": layout, "dtype": dtype, "on_host": on_host and weights,
              "param_dtype": None if param_dtype == "auto" else param_dtype}
    if task is not None and issubclass(task, ReleasedTask):
        bundle = task.source_of(run, **placed)
    elif "model" in run.record:
        bundle = Pretrained.from_saved(run, **placed)
    else:
        raise ValueError(f"{source} is the native release of a {run.objective.__name__} run, which records "
                         "no model of Dew's and declares no task a run starts from")
    if not isinstance(bundle, kind):
        raise TypeError(f"{source} is a {type(bundle).__name__} source, not a {kind.__name__}; load it with "
                        f"{type(bundle).__name__}.load or Pretrained.load")
    return bundle


def released_source(name_or_dir: str | Path, asked: bool | None, **release) -> tuple[str | Path, bool]:
    """The source a load reads, itself or a saved run's release (`released_run`), and whether the load
    runs its own code (`_runs_code`): a trusted run's release is trusted with it, as the processor's
    record names the release."""
    runs = _runs_code(str(name_or_dir), asked)
    released = released_run(name_or_dir, **release)
    if runs and released != name_or_dir:
        trust_sources((str(released),))
    return released, runs


def _runs_code(name: str, asked: bool | None) -> bool:
    """Whether a load of `name` runs the source's own code: as `asked`, or, where None, as the process
    trusts the source (`dew.registry.trusts_source`). A load asked to run it trusts the source for the
    process, so a processor reloaded from its record runs that code too (`load_processor`)."""
    if asked is None:
        return trusts_source(name)
    if asked:
        trust_sources((name,))
    return asked
