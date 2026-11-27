"""How a run trains, as a run's record states it and its command line sets it: its trainer's options
(`TrainerConfig`) and a pretrained source's (`SourceOptions`), each a dataclass a `RunConfig` holds, and
where a run keeps its local records (`local_records`)."""

from __future__ import annotations

import dataclasses
import datetime
import functools
import hashlib
import logging
import operator
import os
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Literal

import tyro

from dew._builds import dew_cache_dir
from dew._records import duration, is_share, recorded_duration
from dew.checkpoints import Checkpoints, Keep, Resume
from dew.data.dataset import Loader, Ramp, ramped
from dew.data.plan import DataResume
from dew.distributed.placement import Layout, MeshSpec
from dew.interop.source_options import (
    SourceBuild as SourceBuild,
    SourceOptions as SourceOptions,
    SourceOptionsKeywords as SourceOptionsKeywords,
    SourcePlacement as SourcePlacement,
)
from dew.kernels import Kernels
from dew.nn.precision import DTYPE_PRESETS, NUMERICS_PRESETS, Numerics, numerics_of
from dew.nn.sharding import Link
from dew.quantization import Quantization
from dew.registry import record_argument
from dew.training.budget import Horizon, Pause, Plateau, budget_tokens, run_steps
from dew.training.cadence import Counted
from dew.training.cooldown import Cooldowns
from dew.training.diloco import DiLoCo
from dew.training.inspection import Inspection
from dew.training.policy import FitPolicy, Remat, UpdateBatch
from dew.training.profiling import ProfileWindow, RunProfile
from dew.training.selection import Best
from dew.training.tracker import Wandb

if TYPE_CHECKING:
    # tyro reads the runtime annotation, a subcommand per named numerics (`trainer.precision:bf16-true`), each
    # a `Numerics` at that preset; a type checker reads the class.
    type NumericsChoices = Numerics
else:
    NumericsChoices = functools.reduce(operator.or_, (
        Annotated[Numerics, tyro.conf.subcommand(name, default=numerics)]
        for name, numerics in {"default": Numerics(), **NUMERICS_PRESETS,
                               **{name: numerics_of(name) for name in DTYPE_PRESETS}}.items()))


_log = logging.getLogger(__name__)

PASSES = "passes"
"""The default validation and checkpoint cadence: at each pass's end where the run counts its data's passes
(as `epoch`), otherwise once, at the run's end (`100%`), as `TrainerConfig.interval` resolves it at the
start, which records and logs what it became (docs/design/training.md, cadences). `epoch`, written, is
strict: a run whose data counts no pass refuses it."""

_AT_END = "100%"
"""What `PASSES` resolves to where the data counts no pass: the run's end, a share of its planned length."""


def _cadence_argument(*, timed: bool = True):
    """A cadence's command-line form: steps, `epoch`, `passes`, a share of the run such as `5%`, None, and
    with `timed` a duration such as `30m`."""
    def read(given):
        value = given[0]
        if value == 'None':
            return None
        if value in ('epoch', PASSES) or is_share(value):
            return value
        return int(value) if value.isdecimal() or not timed else duration(value)
    return tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="STEPS|DURATION|SHARE%|epoch|passes" if timed else "STEPS|SHARE%|epoch|passes",
        instance_from_str=read,
        is_instance=lambda value: value is None or isinstance(value, (int, str, datetime.timedelta)),
        str_from_instance=lambda value: [
            recorded_duration(value) if isinstance(value, datetime.timedelta) else str(value)
        ],
    )


@dataclasses.dataclass(frozen=True)
class Diagnostics:
    """What a run pins of how its programs run, apart from what it trains and how: each op's kernel, the
    links of its split axes, the measuring of its launches and extra XLA flags, set where a run diagnoses or
    measures, every one derived where it states none (`RunConfig.diagnostics`, `--diagnostics.*`)."""

    kernels: Kernels = dataclasses.field(default_factory=Kernels)
    """The kernel the run runs for each op it names (`dew.kernels.Kernels`, `dew.kernels.use`), for every
    program it traces, its steps', evaluations' and samples' (`RunConfig.traced`):
    `--diagnostics.kernels.attention xla`. None named, each op's pick for its call (row 4649)."""
    links: Annotated[Mapping[str, Link], record_argument(Mapping[str, Link])] = dataclasses.field(
        default_factory=dict)
    """The links of the run's split axes, by axis name (`Trainer`'s `links`), stated where no process
    measures them as they are or should not: a step spreads an axis's projections as its stated link
    would, `--diagnostics.links '{"tensor": {"bytes_per_second": 4.5e11, "flops_per_second": 9.9e14}}'`.
    Unnamed, an axis's link is measured on its devices, and on compile-only ones none is: a step
    compiled ahead compiles with these and records them (`AheadOfTime.links`)."""
    tune_kernels: bool = False
    """Measure each Pallas launch of the step that has no config measured on this GPU.

    Only a process with the GPU to itself measures, and the picks persist beside
    the compilation cache for every later run that sets it too, each pick read for
    its own launch alone. It measures each of Dew's own launches that declares
    variants to time (the kernel templates, `kernels={"attention": "template"}`, the norms,
    SwiGLU and decode attention); tokamax's flash kernel (`"triton"`) runs at
    tokamax's heuristic config, unmeasured.
    Unset, nothing is measured or read: every launch takes the config it derives
    from the GPU's facts (row 4837).
    """
    xla_flags: str | None = None
    """Extra XLA_FLAGS for this run, appended to the environment by `prepare_process`.

    `dew train` sets them in XLA_FLAGS before the process loads jaxlib, so they hold
    for every runtime XLA loads (`dew._devices.hold_xla_flags`); `prepare_process`
    appends them before JAX opens a backend, unproven where jaxlib has loaded.
    Library users set XLA_FLAGS before the process starts to make them certain;
    docs/performance.md records what was measured.
    """


@dataclasses.dataclass(frozen=True)
class TrainerConfig:
    """Holds the run length, checkpointing, sharding and run tracking."""

    name: str | None = None
    checkpoint_dir: str = "./checkpoints"
    keep: Keep = dataclasses.field(default_factory=Keep)
    """Which checkpoints the run keeps besides its best (`Keep`): its latest, `--trainer.keep.latest 2`, every
    so many steps, `--trainer.keep.every 1000`, or at a wall-time spacing, `--trainer.keep.interval 1h`."""
    resume: Resume = "auto"
    """Whether the run continues its directory's latest checkpoint: where there is one, "require" for a
    job that must not start afresh, or "never" for one that must not continue another (`Trainer.fit`)."""
    data_resume: DataResume = "relaxed"
    """How a resume treats the data (docs/design/data.md, resuming data): "relaxed" reads on at its saved
    position where its plan still makes the same rows and otherwise starts it again, naming what differs in
    the log and the run's records, its state resuming either way; "exact" refuses any difference."""
    skip_nonfinite: bool = False
    """Whether a step with no loss scale whose loss or gradient norm is not finite is skipped, its optimizer
    taking one step of zero gradient, the run stopping after five in a row; off, the step is taken as it is
    formed and the run stops at its next log, naming it (training.md, a step that is not finite)."""
    best: Annotated[Best, tyro.conf.subcommand("best")] | None = None
    """The metric the run ranks its checkpoints by and how (`Best`), by its recorded name:
    `--trainer.best.metric val/accuracy --trainer.best.mode max`; None ranks by validation loss, or by
    training loss where the run validates on nothing. Several policies are `Trainer.fit`'s (`best=`)."""
    key: int = 0
    """The seed of the run key, which sets parameter init and every per-step draw."""
    steps: int | None = None
    """Train this many steps (`--trainer.steps 2000`). The run's length is named once, by `steps`, `epochs`,
    `tokens` or `examples`; none named, it is as long as its data (`dew.training.budget.Horizon`)."""
    epochs: int | None = None
    """Train this many passes over the data."""
    tokens: int | None = None
    """Train on this many tokens, its loss's targets, counted without padding as committed updates train on
    them, the final microbatch crossing it."""
    examples: int | None = None
    """Train on the data's first this many source examples, each read once with every row it makes."""
    max_tokens: int | None = None
    """End a run of steps, epochs or examples at this many tokens, counted as `tokens` counts them, the first
    of the two it reaches."""
    stop: (Annotated[Pause, tyro.conf.subcommand("pause")]
           | Annotated[Plateau, tyro.conf.subcommand("plateau")] | None) = None
    """End the run at a fraction of its length, to resume it later as if it had never stopped
    (`Pause`): `--trainer.stop.fraction 0.5`; or once a metric, by its recorded name
    as `best` names one, stops improving (`Plateau`): `trainer.stop:plateau --trainer.stop.metric
    val/loss --trainer.stop.evals 3`."""
    restore_best: bool = False
    """End on the model the run ranked best (`Trainer.fit`'s `restore_best`): the state the run hands
    on (`RunConfig.train`'s `after`) holds the best step's weights, EMA and counters, with the final
    step's optimizer state. The checkpoints stay as written; `dew export --step best` exports the
    best one."""
    cooldowns: Annotated[Cooldowns, tyro.conf.subcommand("cooldowns")] | None = None
    """Branch the run into cooldowns, each scored by its validation and discarded (`Cooldowns`):
    `--trainer.cooldowns.at 20% 40% --trainer.cooldowns.length 20%`."""
    log_every: Annotated[int | str, _cadence_argument(timed=False)] = 100
    """The steps between logs, or a share of the run such as "1%" (`eval_every`)."""
    eval_every: Annotated[int | str | None, _cadence_argument(timed=False)] = PASSES
    """The steps between validation passes.

    It is a number of steps, "epoch" for one pass over the data, "passes" (the default) for each pass's
    end where the run counts its data's passes and its end otherwise (`PASSES`), a share of the run such
    as "5%", resolved once against its planned length in steps, epochs, examples or tokens
    (`dew.training.Trainer.fit`), or None to never validate. "epoch" over data whose passes the run counts
    none of (a stream without a record count, a decision table a steps-bounded run reads) raises a
    ValueError, since the run can find no pass's end.
    """
    checkpoint_every: Annotated[int | str | datetime.timedelta | None, _cadence_argument()] = PASSES
    """The steps between checkpoints, with the same choices as `eval_every`, or a duration such as 30m.

    A stream whose iterator cannot report its read position must train with None;
    the trainer refuses any other value for it.
    """
    checkpoint_final: bool = True
    """Write the final state when training ends, independently of `checkpoint_every`.

    False disables this save; cadenced, ranked and preemption saves keep their own policy.
    """
    local_directory: str | None = None
    """A path on every host's own disk where the run also writes checkpoints every `local_every`
    steps, under its name as in `checkpoint_dir`, cheaper than the persistent ones and lost with the
    host (`Checkpoints`); a resume reads the newer of the two. Set with `local_every`, or neither."""
    local_every: int | str | None = None
    """The steps between local checkpoints, or a share of the run such as "5%" (`local_directory`)."""
    scored_to: str | None = None
    """A directory that takes every validation batch's scoring artifacts at each evaluation, a decision
    run's decisions among them, as `<scored_to>/step-<step>/<split>-<batch>.npz`, for pooling a pass's
    metrics offline (`Trainer.fit`'s `scored_to`); None writes none."""
    accumulation: int | None = None
    """The batches one optimizer update adds up (gradient accumulation), `--trainer.accumulation 4`; or,
    with `update_tokens`, none stated: one update's size is named once (`dew.training.UpdateBatch`). Neither
    named, one batch an update."""
    update_tokens: int | None = None
    """The tokens one optimizer update trains on, a whole number of the data's token batches
    (`Dataset.batch(tokens=)`), from which the run reads the batches it adds up: `--trainer.update-tokens
    524288`."""
    split: int | Literal["auto"] = "auto"
    """The microbatches each step's batch is read in, its rows split evenly; "auto", the fewest that fit the
    devices before the step recomputes anything (`Trainer.compile`). A step's tokens and its update's stay
    as stated either way; only the microbatch shrinks (`dew.training.FitPolicy`)."""
    remat: Remat = "auto"
    """What the step recomputes in its backward pass (`dew.training.Remat`): "auto", the least that fits."""
    microbatches: int | None = None
    """Under a mesh of pipeline stages, the microbatches each step feeds through them, a multiple of the
    stages; None, one a stage (`dew.training.distributed.stage_microbatches`)."""
    batch_ramp: Ramp | None = None
    """A ramp that grows `batch_size` over the run's first records instead of starting there.

    It sets the global batch the run starts at, what each stage adds, and the
    records the whole ramp spans. Unset trains at `batch_size` throughout. Either
    way there is one optimizer update a step, and the compiled step is traced
    once per stage.
    """
    precision: NumericsChoices = dataclasses.field(default_factory=Numerics)
    """The run's numerics, set once and recorded with it (`dew.nn.precision.Numerics`; `ACCURATE` for
    scientific runs), on the command line a named preset each of whose fields its flags move:
    `trainer.precision:bf16-true`, `trainer.precision:accurate --trainer.precision.forms fast`: its dtypes
    by Lightning's names, its products' precision by JAX's levels and its ops' forms, over every program it
    traces, the step and its evaluations (`dew.nn.kernels.generation.run_precision`). Its compute dtype is
    the model's where the model's record states none (`RunConfig.numerics_fields`), its parameters' a
    loaded checkpoint's storage, and a float16 compute scales the loss (`RunConfig.built`). The optimizer's
    moments are a storage format of their own (`dew.optim.Adam.state_dtype`: float32 under any weights,
    bfloat16 stochastically rounded, int8 blockwise). A dtype or precision the model states still wins, and
    a resumed run traces under the numerics it records."""
    mesh: MeshSpec = dataclasses.field(default_factory=MeshSpec)
    layout: Annotated[Layout, tyro.conf.subcommand("layout")] | None = None
    """The layout the run's trainer is given (`Trainer`'s `layout`): unset, the layout chooser decides
    (`dew.training.streaming.laid_start`); stated, as `--trainer.layout.min-shard 8`, its own, the device
    layout kept."""
    profile: ProfileWindow | None = None
    """One profiler window: the steps to trace, the warmup before them and the output directory, in place
    of the run's own (`run_profile`). Unset traces the run's own window alone.
    """
    run_profile: RunProfile | None = dataclasses.field(default_factory=RunProfile)
    """The run's own profile window, every run's by default: two of its steps traced once ten have run
    past its first compile, kept with its records (`window`), its trace and manifest alone, where the
    process can take a capture (`dew.training.profiling.capturable`): a capture's fixed cost, about a
    second on a CPU host whatever its steps (row 4112), a run of minutes does not see, and a run shorter
    than its warmup takes none. A `profile` the run names replaces it; None traces none of its own.
    """
    compilation_cache_dir: str | None = None
    """The directory of JAX's persistent compilation cache, so a restart skips recompiling the step.

    None caches under the user's cache directory wherever the run's processes share it
    (`dew.training.runtime.ready_process`); a pool across hosts without shared storage names one
    they share. JAX_ENABLE_COMPILATION_CACHE=false compiles from scratch every run.
    """
    wandb: Wandb | None = None
    """An optional W&B telemetry sink, in addition to the local tracking journal.

    Scalars, records and objective previews are tracked; the final checkpoint is uploaded where it says
    so (`Wandb.publish`), and nothing else.
    """
    multi_host: bool | None = None
    """Whether to join the JAX process pool.

    None tries, and continues alone only when no cluster is configured; True
    requires the pool; False never joins.
    """
    quantization: Quantization | None = None
    """The quantized-training spec, wrapped around the module the objective trains.

    The wrapper is applied before the run initializes the module. Unset trains in
    the compute dtype. `dew.quantization` describes what the wrapper does
    and what it keeps.
    """
    ahead: str | None = None
    """Run the step `RunConfig.lower` saved at this path as the run's first step, in place of compiling it.

    The run lowers its first step as it would and runs the saved executable where that is the same
    program, on a host of the topology it was compiled for; another program, from another config,
    data shape, Dew or a plan recorded for another rung, is refused before the run trains. A file
    compiled without its GPU kind's tuned choices (one compiled for a generation's canonical GPUs) is
    not run: the run compiles its own step, said once (`AheadTarget`).
    """
    inspection: Annotated[Inspection, tyro.conf.subcommand("inspect")] | None = None
    """What the run watches at its sampled steps (`dew.training.inspection.Inspection`): parameter paths by
    glob, the trees summarized there and every how many steps, `--trainer.inspection.paths '*norm*'
    --trainer.inspection.every 100`; unset, the step compiles unchanged."""
    diloco: Annotated[DiLoCo, tyro.conf.subcommand("diloco")] | None = None
    """DiLoCo over the mesh's diloco groups (`--trainer.mesh.diloco`, `dew.training.diloco.DiLoCo`): every
    `--trainer.diloco.every` windows of updates the groups' copies are averaged into the outer model, by
    Nesterov SGD at `--trainer.diloco.rate` with `--trainer.diloco.momentum`; unset, one copy trains."""
    def update_batches(self, dataset: Loader) -> int:
        """The steps an update pools over `dataset`: its stated batches, or its tokens over the data's token
        batches, every phase's alike (`UpdateBatch.of_tokens`)."""
        return self.update_batch.of_tokens(dataset.each_token_batch)

    def __post_init__(self):
        # A record names its numerics as a preset's name or by their fields (`numerics_of`).
        object.__setattr__(self, "precision", numerics_of(self.precision))
        if self.update_tokens is not None and self.batch_ramp is not None:
            raise ValueError("--trainer.batch-ramp grows row batches and cannot use a token budget")
        if self.wandb is not None and self.wandb.publish and not self.checkpoint_final:
            raise ValueError("trainer.wandb.publish publishes the final checkpoint, which "
                             "trainer.checkpoint_final keeps; keep it")
        if self.wandb is not None and self.wandb.publish and self.wandb.offline:
            raise ValueError("trainer.wandb.publish links the final checkpoint into W&B's registry, which an "
                             "offline W&B run cannot reach; track online to publish")
        if isinstance(self.log_every, str) and not is_share(self.log_every):
            raise ValueError(f"--trainer.log-every is steps or a share such as 1%, not {self.log_every!r}")
        if (isinstance(self.eval_every, str) and self.eval_every not in ('epoch', PASSES)
                and not is_share(self.eval_every)):
            raise ValueError(f"--trainer.eval-every is steps, epoch, passes or a share such as 5%, not "
                             f"{self.eval_every!r}")
        if (isinstance(self.checkpoint_every, str) and self.checkpoint_every not in ('epoch', PASSES)
                and not is_share(self.checkpoint_every)):
            object.__setattr__(self, 'checkpoint_every', duration(self.checkpoint_every))
        if self.stop is not None and not isinstance(self.stop, Pause | Plateau):
            # A record's stop no class reads (`registry._unnamed` keeps it as written) would train to the
            # full budget, neither paused nor stopped (row 12115).
            raise TypeError(f"trainer.stop is a Pause (fraction) or a Plateau (metric, evals), not "
                            f"{self.stop!r}")
        if self.keep.where is not None:
            raise TypeError("Keep.where is code-only; a recorded retention policy contains no callable")
        if self.best is not None and (not isinstance(self.best, Best) or self.best.score is not None):
            raise TypeError("a recorded best selector names metrics; callable scores are code-only")
        # Each refuses what it holds as it is made, a length named twice or an update sized twice.
        _ = self.horizon, self.update_batch, self.fit_policy

    @property
    def horizon(self) -> Horizon:
        """How long the run trains (`steps`, `epochs`, `tokens`, `examples`, `max_tokens`), as the fit reads
        it."""
        return Horizon(steps=self.steps, epochs=self.epochs, tokens=self.tokens, examples=self.examples,
                       max_tokens=self.max_tokens)

    @property
    def update_batch(self) -> UpdateBatch:
        """One optimizer update's size (`accumulation`, `update_tokens`), as the trainer reads it."""
        return UpdateBatch(batches=self.accumulation, tokens=self.update_tokens)

    @property
    def fit_policy(self) -> FitPolicy:
        """How the step fits the devices (`split`, `remat`, `microbatches`), as the trainer reads it."""
        return FitPolicy(split=self.split, remat=self.remat, microbatches=self.microbatches)

    def best_policies(self) -> tuple[Best, ...] | None:
        """The policies the run ranks its checkpoints by (`best`), None for the run's default."""
        return None if self.best is None else (self.best,)

    def resumes(self) -> bool:
        """Whether this run continues from a checkpoint of its own: a named run whose directory holds a
        step a resume reads (`Checkpoints.latest`). Its weights then come from that step, so the source
        it started from need not be read (`Pretrained.load`'s `weights`)."""
        if self.name is None:
            return False
        with self.checkpoints(self.name) as checkpoints:
            return checkpoints.latest is not None

    def directory(self, name: str) -> str:
        """The directory of the run `name`'s checkpoints and record, in `checkpoint_dir`."""
        return os.path.join(self.checkpoint_dir, name)

    def window(self, directory: str, name: str) -> ProfileWindow | None:
        """The profile window the run `name` under `directory` traces: `profile` as it names it, or else its
        own (`run_profile`) kept with the run's local records (`local_records`: its directory's `profile`,
        or Dew's cache for a run in a bucket) where the process can take a capture (`capturable`), none
        where it cannot; none where the run traces none. A trainer is given a window naming its directory
        alone (row 4246)."""
        from dew.training.profiling import capturable

        if self.profile is not None or self.run_profile is None:
            return self.profile
        return self.run_profile.window(local_records(directory, name, "profile")) if capturable() else None

    def checkpoints(self, name: str, directory: str | None = None) -> Checkpoints:
        """The checkpoints of the run `name` in `checkpoint_dir` (`directory`), or in `directory` where
        given (a saved run read where it lies), keeping `keep`, with its local tier under `name` in
        `local_directory`, so runs sharing a local disk resume each only its own."""
        return Checkpoints(self.directory(name) if directory is None else directory, keep=self.keep,
                           local_directory=None if self.local_directory is None else os.path.join(
                               self.local_directory, name), local_every=self.local_every)

    def source_placement(self, *, frozen: bool) -> SourcePlacement:
        """Where a run that starts from a source (`Pretrained.load`) reads it: the load's keywords,
        `frozen` where the run keeps the source as it is (an adapter trained over it). A run that
        resumes reads no weights (`resumes`), as its checkpoint holds them. A frozen source is read onto
        this run's mesh, sharded as its layout shards the state, each process reading its devices'
        shards, and the start takes those arrays as they are (`Trainer.place`): the run and its
        objective hold one copy. A source the run trains stays on the host, as its start copies the
        arrays it trains, and so does one whose state lives on the host (`Layout.host` holds the
        variables) or keeps its frozen tree whole (`frozen_whole`), which its start places itself."""
        if self.resumes():
            return {"weights": False}
        layout = Layout() if self.layout is None else self.layout
        if not frozen or "variables" in layout.host or layout.frozen_whole:
            return {"on_host": True}
        return {"mesh": self.mesh, "layout": layout}

    def resolved(self, dataset: Loader) -> tuple[Loader, int | None]:
        """`dataset` as a run of this config trains on it, and the run's steps: the dataset its batch ramp
        reads, and the step cap over that (`total_steps`), before which the ramp goes, since it reads fewer
        records a step early and a pass over the data is that many steps longer. Its updates are the steps
        over the accumulation, an update of tokens read against the loader's token batches
        (`accumulation`): the config stays as authored, which its record holds and its trainer resolves as
        it reads the loader (`Trainer._reading`)."""
        dataset = dataset if self.batch_ramp is None else ramped(dataset, self.batch_ramp)
        return dataset, self.total_steps(dataset)

    def total_steps(self, dataset: Loader) -> int | None:
        """Return the step cap from `steps`, `epochs` or `examples`, None for a run its tokens bound
        alone, its own or where its data's schedule ends (`budget_tokens`; `run_steps`, the one policy
        a direct fit applies too, which refuses a schedule to an objective counting no tokens)."""
        return run_steps(dataset, self.horizon, self.update_batches(dataset),
                         budget_tokens(dataset, self.horizon.token_cap))

    def total_updates(self, dataset: Loader) -> int | None:
        """Return the optimizer updates the run takes over `dataset` as `RunConfig.train` finalizes it,
        its token budget's accumulation and its batch ramp applied: what an average that moves once an
        update ramps over. None for a token-only run."""
        _, steps = self.resolved(dataset)
        # A window the run ends inside is never committed, as the trainer counts its updates.
        return None if steps is None else steps // self.update_batches(dataset)

    def eval_interval(self, dataset: Loader) -> int | str | Counted | None:
        """Return the steps between validation passes over `data`, or None for never."""
        return self.interval(self.eval_every, dataset, "--trainer.eval-every")

    def checkpoint_interval(self, dataset: Loader) -> int | str | Counted | datetime.timedelta | None:
        """Return the steps, or a recorded duration such as 30m, between checkpoints."""
        value = self.checkpoint_every
        if isinstance(value, datetime.timedelta):
            if value.total_seconds() <= 0:
                raise ValueError("checkpoint_every duration must be positive")
            return value
        return self.interval(self.checkpoint_every, dataset, "--trainer.checkpoint-every")

    def interval(self, value, dataset: Loader, option: str) -> int | str | Counted | None:
        """A cadence (`eval_every`, `checkpoint_every`, a split's `RunConfig.val_every`) the command
        line's `option` (`--trainer.eval-every`) sets, resolved against `dataset`: steps for `epoch`, or
        the source examples a pass holds where the run reads its epochs as passes over them
        (`Loader.epochs`, `Loader.pass_examples`); for `passes`, the same where the run counts its data's
        passes, and the run's end (`_AT_END`) where it counts none, which it logs; steps, a share of the
        run (which `fit` resolves) or None as given."""
        if value is None or isinstance(value, int) or is_share(value):
            return value
        # A pass of a read with no record count, as a run of its epochs or one counting its passes reads it
        # (`dew.data.compile`'s `count_passes`).
        passed = dataset.pass_examples
        if value == PASSES and dataset.steps_per_epoch is None and not passed:
            _log.info("%s passes: the run counts none of its data's passes (its read has no record count "
                      "and states no pass of its source examples), so it runs at the run's end (%s)", option,
                      _AT_END)
            return _AT_END
        if dataset.steps_per_epoch is None and passed:
            return Counted(passed, "examples")
        if dataset.steps_per_epoch is None:
            raise ValueError(
                f"{option} epoch needs a dataset with a record count; this "
                f"one streams without one, so give the interval in steps or None")
        return dataset.epoch_steps(window=self.update_batches(dataset))


def local_records(directory: str, name: str, kind: str) -> str:
    """Where the run `name` under `directory` keeps its local records of `kind` (its journal, `tracking`,
    or its profile's capture, `profile`), beside its checkpoints where it can be.

    A bucket URI is not a directory this process can write into, so a remote run keeps them under Dew's
    cache (`dew_cache_dir`), keyed by the run name and a digest of the bucket path.
    """
    if "://" not in directory:
        return os.path.join(directory, kind)
    return os.path.join(dew_cache_dir(), kind,
                        _artifact_name(name) + "-" + hashlib.sha256(directory.encode()).hexdigest()[:12])


def _artifact_name(name: str) -> str:
    """Return `name` with every character a path or a tracker id cannot hold replaced.

    A run name is the caller's and may hold spaces or slashes; a directory
    entry and a tracker artifact id take neither.
    """
    return re.sub(r"[^\w.-]", "-", name)
