"""How a run trains, as a run's record states it and its command line sets it: its trainer's options
(`TrainerConfig`) and a pretrained source's (`SourceOptions`), each a dataclass a `RunConfig` holds, and
where a run keeps its local records (`local_records`)."""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import os
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Self

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
from dew.nn.precision import Numerics, numerics_of
from dew.nn.sharding import Link
from dew.quantization import Quantization
from dew.registry import from_record, record_argument, to_record
from dew.training.ahead import AheadTarget
from dew.training.budget import Pause, Plateau, budget_tokens, run_length, run_steps
from dew.training.cadence import Counted
from dew.training.cooldown import Cooldowns
from dew.training.diloco import DiLoCo
from dew.training.inspection import Inspection
from dew.training.profiling import ProfileWindow, RunProfile
from dew.training.selection import Best
from dew.training.tracker import Wandb

if TYPE_CHECKING:

    pass


def _best_argument():
    """A metric name or JSON policies through one CLI argument."""
    def read(given):
        text = given[0]
        if text == 'None':
            return None
        if text.startswith(('{', '[')):
            policies = json.loads(text)
            kind = tuple[Best, ...] if isinstance(policies, list) else Best
            return from_record(kind, policies, dtypes=False)
        return Best(text)

    def write(policy):
        if policy is None:
            return ['None']
        if isinstance(policy, str):
            return [policy]
        return [
            json.dumps(
                [to_record(entry, Best) for entry in policy]
                if isinstance(policy, tuple)
                else to_record(policy, Best)
            )
        ]

    return tyro.constructors.PrimitiveConstructorSpec(
        nargs=1, metavar='METRIC|JSON', instance_from_str=read,
        is_instance=lambda policy: policy is None or isinstance(policy, (str, Best, tuple)),
        str_from_instance=write)


def _numerics_argument():
    """A run's numerics on the command line: a preset's name ("accurate", "default"), the usual dtypes'
    Lightning name ("bf16-true"), or a `Numerics` record's JSON (`dew.nn.precision.numerics_of`)."""
    return tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="NAME|JSON",
        instance_from_str=lambda given: (from_record(Numerics, json.loads(given[0]))
                                         if given[0].startswith("{") else numerics_of(given[0])),
        is_instance=lambda value: isinstance(value, Numerics),
        str_from_instance=lambda value: [json.dumps(to_record(value, Numerics))],
    )


def _keep_argument():
    return tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="LATEST|JSON",
        instance_from_str=lambda given: from_record(Keep, json.loads(given[0]), dtypes=False)
        if given[0].startswith("{")
        else int(given[0]),
        is_instance=lambda keep: isinstance(keep, (int, Keep)),
        str_from_instance=lambda keep: [
            json.dumps(to_record(keep, Keep)) if isinstance(keep, Keep) else str(keep)
        ],
    )


def _cadence_argument(*, timed: bool = True):
    """A cadence's command-line form: steps, `epoch`, a share of the run such as `5%`, None, and with
    `timed` a duration such as `30m`."""
    def read(given):
        value = given[0]
        if value == 'None':
            return None
        if value == 'epoch' or is_share(value):
            return value
        return int(value) if value.isdecimal() or not timed else duration(value)
    return tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="STEPS|DURATION|SHARE%|epoch" if timed else "STEPS|SHARE%|epoch",
        instance_from_str=read,
        is_instance=lambda value: value is None or isinstance(value, (int, str, datetime.timedelta)),
        str_from_instance=lambda value: [
            recorded_duration(value) if isinstance(value, datetime.timedelta) else str(value)
        ],
    )


@dataclasses.dataclass(frozen=True)
class TrainerConfig:
    """Holds the run length, checkpointing, sharding and run tracking."""

    name: str | None = None
    checkpoint_dir: str = "./checkpoints"
    keep: Annotated[int | Keep, _keep_argument()] = Keep.latest
    """The number of latest checkpoints kept, besides the best one (`Keep`'s own default)."""
    resume: Resume = "auto"
    """Whether the run continues its directory's latest checkpoint: where there is one, "require" for a
    job that must not start afresh, or "never" for one that must not continue another (`Trainer.fit`)."""
    data_resume: DataResume = "relaxed"
    """How a resume treats the data (docs/design/data.md, resuming data): "relaxed" reads on at its saved
    position where its plan still makes the same rows and otherwise starts it again, naming what differs in
    the log and the run's records, its state resuming either way; "exact" refuses any difference."""
    best: Annotated[str | Best | tuple[Best, ...] | None, _best_argument()] = None
    """The metric name and ranking policy; None selects validation loss or training loss."""
    update_tokens: int | None = None
    """The optimizer update's token budget, a whole number of the data's token batches (its batch stage's
    `TokenBudget`), which sets the run's accumulation (`RunConfig.train`)."""
    key: int = 0
    """The seed of the run key, which sets parameter init and every per-step draw."""
    steps: int | None = None
    epochs: int | None = None
    """The run length as passes over the data; `steps` sets it directly instead."""
    tokens: int | None = None
    """Stop after the loss has scored this many targets across all microbatches, including before a
    resume.

    Counts the tokens committed updates trained on, reported as `train/committed_tokens`, without
    padding. The final
    microbatch may cross the budget. With `steps`, `epochs` or `examples`, the
    first cap reached ends the run. A token-only run needs a constant learning
    rate or a schedule with its own explicit step horizon.
    """
    examples: int | None = None
    """The run length as the first this many source examples of the data, each read once with every
    row it makes: the records the chain makes in its own order, counted before they are packed and
    before an objective makes several rows of one (a decision layout's row a question)
    (`Trainer.fit`'s `examples`); `steps` or `epochs` set it instead."""
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
    eval_every: Annotated[int | str | None, _cadence_argument(timed=False)] = "epoch"
    """The steps between validation passes.

    It is a number of steps, "epoch" for one pass over the data, a share of the
    run such as "5%", resolved once against its planned length in steps, epochs,
    examples or tokens (`dew.training.Trainer.fit`), or None to never validate.
    "epoch" over a stream that reports no record count raises a ValueError, since
    such a stream has no pass.
    """
    checkpoint_every: Annotated[int | str | datetime.timedelta | None, _cadence_argument()] = "epoch"
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
    accumulation: int = 1
    """The number of steps per optimizer update."""
    split: int | None = None
    """The microbatches each step's batch is read in, its rows split evenly.

    None lets the trainer plan it from the compiled step's memory: the
    fewest that fit the devices before it recomputes anything, at the least
    recomputation that fits (`Trainer.compile`). A step's tokens and the
    update's stay as configured either way; only the microbatch shrinks.
    """
    batch_ramp: Ramp | None = None
    """A ramp that grows `batch_size` over the run's first records instead of starting there.

    It sets the global batch the run starts at, what each stage adds, and the
    records the whole ramp spans. Unset trains at `batch_size` throughout. Either
    way there is one optimizer update a step, and the compiled step is traced
    once per stage.
    """
    dynamic_scale: bool | None = None
    """Whether the run scales its loss dynamically (`flax.training.dynamic_scale.DynamicScale`): None
    for a float16 compute's alone, which needs it (`RunConfig.built`); a run states True or False to
    choose, as a run saved before its numerics records the one it trained under."""
    precision: Annotated[Numerics, _numerics_argument()] = dataclasses.field(default_factory=Numerics)
    """The run's numerics, set once and recorded with it (`dew.nn.precision.Numerics`; `ACCURATE` for
    scientific runs): its dtypes by Lightning's names, its products' precision by JAX's levels and its
    ops' forms, over every program it traces, the step and its evaluations
    (`dew.nn.kernels.generation.run_precision`). Its compute dtype is the model's where the model's
    record states none (`RunConfig.numerics_fields`), its parameters' a loaded checkpoint's storage, and
    a float16 compute scales the loss (`RunConfig.built`). The optimizer's moments are a storage format
    of their own (`OptimConfig.state_dtype`: float32 under any weights, bfloat16 stochastically rounded,
    int8 blockwise). A dtype or precision the model states still wins, and a resumed run traces under
    the numerics it records."""
    mesh: MeshSpec = dataclasses.field(default_factory=MeshSpec)
    microbatches: int | None = None
    """Microbatches a pipelined step feeds through the mesh's stages, a
    multiple of `mesh.stage`; unset feeds one a stage."""
    layout: Layout = dataclasses.field(default_factory=Layout)
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

    Scalars, records and objective previews are tracked; checkpoints are not uploaded.
    """
    publish: bool = False
    """Publish the final checkpoint to W&B; requires `wandb`, online, and `checkpoint_final`."""
    multi_host: bool | None = None
    """Whether to join the JAX process pool.

    None tries, and continues alone only when no cluster is configured; True
    requires the pool; False never joins.
    """
    xla_flags: str | None = None
    """Extra XLA_FLAGS for this run, appended to the environment by `prepare_process`.

    `dew train` sets them in XLA_FLAGS before the process loads jaxlib, so they hold
    for every runtime XLA loads (`dew._devices.hold_xla_flags`); `prepare_process`
    appends them before JAX opens a backend, unproven where jaxlib has loaded.
    Library users set XLA_FLAGS before the process starts to make them certain;
    docs/performance.md records what was measured.
    """
    quantization: Quantization | None = None
    """The quantized-training spec, wrapped around the module the objective trains.

    The wrapper is applied before the run initializes the module. Unset trains in
    the compute dtype. `dew.quantization` describes what the wrapper does
    and what it keeps.
    """
    lower_for: str | None = None
    """Check the run's step for a GPU on any host, a CPU too, then exit without training.

    `--trainer.lower-for sm80` (or sm75, sm86, sm89, sm90, sm120) builds the run
    as it would, traces its first step as on a GPU of that CUDA generation, with
    every kernel choice and refusal that GPU makes, and lowers it for CUDA,
    each of its Pallas kernels to Triton. A kernel that would fail the GPU's
    first compile there fails here, in seconds and without the GPU; XLA's
    compile of the whole step still needs one. The program exits with status
    0 once the step lowers, and opens no W&B run. A TPU topology's step compiles ahead
    (`compile_ahead`).
    """
    compile_ahead: Annotated[AheadTarget, tyro.conf.subcommand("ahead")] | None = None
    """Compile the run's first step for GPUs this host lacks, save it, then exit without training.

    `--trainer.compile-ahead.generation sm90 --trainer.compile-ahead.devices 8
    --trainer.compile-ahead.path step.dew-aot` builds the run as it would over eight compile-only
    H100s (`dew.training.ahead`), compiles its first step there with XLA's CUDA compiler, which needs
    the CUDA plugin and no GPU, and writes it with what it costs: each device's peak bytes and the
    step's FLOPs, read before booking the GPUs. The program exits with status 0, and writes no
    checkpoint or tracking. A TPU topology compiles the same way on any host that holds libtpu, as
    MaxText's train_compile does: `--trainer.compile-ahead.generation v5e:4x4
    --trainer.compile-ahead.devices 16`, the step loading on a host of that topology alone.

    `--trainer.compile-ahead.capacity-gib 80` states each GPU's memory, and the run's own fit search
    then plans the step there as a run would, its log and the saved step naming every program it
    compiled, each with the bytes derived for it beside XLA's, and the plan it took, with the seconds
    planning took. A run with its remat set compiles at that remat alone, and is told of no other;
    this is how to ask what fits beside it: compile it ahead once at each remat policy its model
    takes (the model's `remat` field) and compare the plans.
    """
    ahead: str | None = None
    """Run the step `compile_ahead` saved at this path as the run's first step, in place of compiling it.

    The run lowers its first step as it would and runs the saved executable where that is the same
    program, on a host of the topology it was compiled for; another program, from another config,
    data shape, Dew or a plan recorded for another rung, is refused before the run trains.
    """
    inspection: Annotated[Inspection, tyro.conf.subcommand("inspect")] | None = None
    """What the run watches at its sampled steps (`dew.training.inspection.Inspection`): parameter paths by
    glob, the trees summarized there and every how many steps, `--trainer.inspection.paths '*norm*'
    --trainer.inspection.every 100`; unset, the step compiles unchanged."""
    diloco: Annotated[DiLoCo, tyro.conf.subcommand("diloco")] | None = None
    """DiLoCo over the mesh's diloco groups (`--trainer.mesh.diloco`, `dew.training.diloco.DiLoCo`): every
    `--trainer.diloco.every` windows of updates the groups' copies are averaged into the outer model, by
    Nesterov SGD at `--trainer.diloco.rate` with `--trainer.diloco.momentum`; unset, one copy trains."""
    links: Annotated[Mapping[str, Link], record_argument(Mapping[str, Link])] = dataclasses.field(
        default_factory=dict)
    """The links of the run's split axes, by axis name (`Trainer`'s `links`), stated where no process
    measures them as they are or should not: a step spreads an axis's projections as its stated link
    would, `--set trainer.links='{"tensor": {"bytes_per_second": 4.5e11, "flops_per_second": 9.9e14}}'`.
    Unnamed, an axis's link is measured on its devices, and on compile-only ones none is: a step
    compiled ahead compiles with these and records them (`AheadOfTime.links`)."""
    tune_kernels: bool = False
    """Measure each Pallas launch of the step that has no config measured on this GPU.

    Only a process with the GPU to itself measures, and the picks persist beside
    the compilation cache for every later run that sets it too, each pick read for
    its own launch alone. It measures each of Dew's own launches that declares
    variants to time (the kernel templates, `attention_impl="template"`, the norms,
    SwiGLU and decode attention); tokamax's flash kernel (`"triton"`) runs at
    tokamax's heuristic config, unmeasured.
    Unset, nothing is measured or read: every launch takes the config it derives
    from the GPU's facts (row 4837).
    """

    def accumulating(self, dataset: Loader) -> Self:
        """This config with the accumulation its token budget sets over `dataset`'s token batches, every
        phase's alike (`Loader.token_batches`): `update_tokens` a whole number of them. Raises ValueError for
        rows, no whole number, phases of other token batches or an accumulation set otherwise."""
        if self.update_tokens is None:
            return self
        if len(set(dataset.token_batches)) > 1:
            raise ValueError(f"--trainer.update-tokens sets one accumulation; the phases batch "
                             f"{list(dataset.token_batches)} tokens: batch alike or set the accumulation")
        if dataset.batch_tokens is None or self.update_tokens % dataset.batch_tokens:
            raise ValueError(f"--trainer.update-tokens {self.update_tokens} is no whole number of the data's "
                             f"token batches ({dataset.batch_tokens}); batch the data by tokens "
                             "(Dataset.batch(tokens=)) dividing it")
        accumulation = self.update_tokens // dataset.batch_tokens
        if self.accumulation not in (1, accumulation):
            raise ValueError(f"--trainer.update-tokens sets the accumulation to {accumulation}; omit "
                             "--trainer.accumulation or match it")
        return dataclasses.replace(self, accumulation=accumulation)

    def __post_init__(self):
        # A record names its numerics as a preset's name or by their fields (`numerics_of`).
        object.__setattr__(self, "precision", numerics_of(self.precision))
        if self.update_tokens is not None and self.update_tokens < 1:
            raise ValueError("--trainer.update-tokens must be positive")
        if self.update_tokens is not None and self.batch_ramp is not None:
            raise ValueError("--trainer.batch-ramp grows row batches and cannot use a token budget")
        if self.split is not None and self.split < 1:
            raise ValueError("--trainer.split reads each step's batch in that many microbatches, at least 1")
        if self.publish and (self.wandb is None or not self.checkpoint_final):
            raise ValueError("trainer.publish requires trainer.wandb and trainer.checkpoint_final")
        if self.publish and self.wandb is not None and self.wandb.offline:
            raise ValueError("trainer.publish links the final checkpoint into W&B's registry, which an "
                             "offline W&B run cannot reach; track online to publish")
        run_length(self.steps, self.epochs, self.tokens, self.examples, required=False)
        if isinstance(self.log_every, str) and not is_share(self.log_every):
            raise ValueError(f"--trainer.log-every is steps or a share such as 1%, not {self.log_every!r}")
        if isinstance(self.eval_every, str) and self.eval_every != 'epoch' and not is_share(self.eval_every):
            raise ValueError(f"--trainer.eval-every is steps, epoch or a share such as 5%, not "
                             f"{self.eval_every!r}")
        if isinstance(self.checkpoint_every, str) and self.checkpoint_every != 'epoch' and not is_share(
                self.checkpoint_every):
            object.__setattr__(self, 'checkpoint_every', duration(self.checkpoint_every))
        if isinstance(self.keep, Mapping):
            object.__setattr__(self, 'keep', from_record(Keep, self.keep, dtypes=False))
        if isinstance(self.keep, Keep) and self.keep.where is not None:
            raise TypeError("Keep.where is code-only; a recorded retention policy contains no callable")
        if self.best is not None:
            choices = self.best if isinstance(self.best, (tuple, list)) else (self.best,)
            rebuilt = []
            for choice in choices:
                if isinstance(choice, str):
                    choice = Best(choice)
                elif isinstance(choice, Mapping):
                    choice = from_record(Best, choice, dtypes=False)
                if not isinstance(choice, Best) or choice._source is not None:
                    raise TypeError("a recorded best selector names metrics; callable scores are code-only")
                rebuilt.append(choice)
            object.__setattr__(
                self, "best", tuple(rebuilt) if isinstance(self.best, (tuple, list)) else rebuilt[0]
            )

    def best_policies(self):
        if self.best is None:
            return None
        return (Best(self.best),) if isinstance(self.best, str) else (
            self.best if isinstance(self.best, tuple) else (self.best,))

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
        if not frozen or "variables" in self.layout.host or self.layout.frozen_whole:
            return {"on_host": True}
        return {"mesh": self.mesh, "layout": self.layout}

    def resolved(self, dataset: Loader) -> tuple[Self, Loader, int | None]:
        """This config and `dataset` as a run trains on them, and the run's steps: the accumulation
        its token budget sets (`accumulating`), the dataset its batch ramp reads, and the step cap over
        that (`total_steps`), before which the ramp goes, since it reads fewer records a step early
        and a pass over the data is that many steps longer. Its updates are the steps over the
        accumulation."""
        trainer = self.accumulating(dataset)
        dataset = dataset if trainer.batch_ramp is None else ramped(dataset, trainer.batch_ramp)
        return trainer, dataset, trainer.total_steps(dataset)

    def total_steps(self, dataset: Loader) -> int | None:
        """Return the step cap from `steps`, `epochs` or `examples`, None for a run its tokens bound
        alone, its own or where its data's schedule ends (`budget_tokens`; `run_steps`, the one policy
        a direct fit applies too, which refuses a schedule to an objective counting no tokens)."""
        tokens = budget_tokens(dataset, self.tokens)
        return run_steps(dataset, self.steps, self.epochs, tokens, self.accumulation, self.examples)

    def total_updates(self, dataset: Loader) -> int | None:
        """Return the optimizer updates the run takes over `dataset` as `RunConfig.train` finalizes it,
        its token budget's accumulation and its batch ramp applied: what an average that moves once an
        update ramps over. None for a token-only run."""
        trainer, _, steps = self.resolved(dataset)
        # A window the run ends inside is never committed, as the trainer counts its updates.
        return None if steps is None else steps // trainer.accumulation

    def eval_interval(self, dataset: Loader) -> int | str | Counted | None:
        """Return the steps between validation passes over `data`, or None for never."""
        return self._interval(self.eval_every, dataset, "eval-every")

    def checkpoint_interval(self, dataset: Loader) -> int | str | Counted | datetime.timedelta | None:
        """Return the steps, or a recorded duration such as 30m, between checkpoints."""
        value = self.checkpoint_every
        if isinstance(value, datetime.timedelta):
            if value.total_seconds() <= 0:
                raise ValueError("checkpoint_every duration must be positive")
            return value
        return self._interval(self.checkpoint_every, dataset, "checkpoint-every")

    def _interval(self, value, dataset: Loader, field_name: str) -> int | str | Counted | None:
        """Steps for `epoch`, or the source examples a pass holds where the run reads its epochs as
        passes over them (`Loader.epochs`, `Loader.pass_examples`); steps, a share of the run (which `fit`
        resolves) or None as given."""
        if value is None or isinstance(value, int) or is_share(value):
            return value
        if dataset.steps_per_epoch is None and dataset.epochs is not None and dataset.pass_examples:
            return Counted(dataset.pass_examples, "examples")
        if dataset.steps_per_epoch is None:
            raise ValueError(
                f"--trainer.{field_name} epoch needs a dataset with a record count; this "
                f"one streams without one, so give the interval in steps or None")
        return dataset.epoch_steps(window=self.accumulation)


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
