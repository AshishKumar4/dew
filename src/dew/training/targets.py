"""What a run compiles or lowers its first step for in place of training (`dew.config.RunConfig.lower`,
`dew lower`): its GPUs or TPU topology, compiled ahead and saved (`AheadTarget`), or a CUDA generation it
is checked against on any host (`LowerFor`). Plain records, which `dew lower --help` reads without JAX; the
work is `dew.training.ahead`'s."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from dew._devices import DEW_SHARE

if TYPE_CHECKING:
    from dew.checkpoints import Resume
    from dew.data.dataset import Loader
    from dew.training._fit import FitPlan
    from dew.training.ahead import CompiledAhead
    from dew.training.trainer import Trainer


@dataclasses.dataclass(frozen=True)
class LowerFor:
    """Where a configured run lowers its first step for a GPU of the CUDA `generation` (sm75, sm80, sm86,
    sm89, sm90 or sm120) on any host, a CPU too, training nothing (`RunConfig.lower`, `dew lower run.json
    target:check --target.generation sm80`): the run built as it would, its first step traced as on that GPU,
    with every kernel choice and refusal it makes, and lowered for CUDA, each Pallas kernel to Triton. A
    kernel that would fail the GPU's first compile fails here, in seconds and without the GPU; XLA's compile
    of the whole step still needs one (`AheadTarget`). It opens no W&B run."""

    generation: str


@dataclasses.dataclass(frozen=True)
class AheadTarget:
    """Where a configured run compiles its first step ahead and saves it, training nothing (`RunConfig.lower`,
    `dew lower run.json target:ahead --target.generation sm90 --target.devices 8 --target.path step.dew-aot`):
    for XLA's CUDA compiler, which needs the CUDA plugin and no GPU, written with what it costs (each device's
    peak bytes and the step's FLOPs, read before booking the GPUs), for `devices` GPUs of the CUDA
    `generation` (`compile_only_gpus`), or the `devices` TPUs of the TPU topology it names ('v5e:4x4',
    `compile_only_tpus`), which this host need not have, written to `path` (`AheadOfTime.save`).
    `capacity_gib` states each GPU's memory, its total as the GPU states it, once: the run's own plan there is
    made and reported against the allocator's limit derived from it (`compile_ahead`), the offline answer to
    what fits, a run with its remat set told of no rung but its own. A host's own `topology` states its GPUs'
    memory instead, and a TPU topology its own. The links it compiles with are the run's
    (`Diagnostics.links`), which no compile-only device measures. To ask what fits beside a run's remat,
    compile it ahead once at each remat its fit policy takes (`FitPolicy.remat`) and compare the plans. Such a
    file is XLA's untuned program for those GPUs, which a run handed it does not run (`AheadOfTime.runnable`):
    a file a run takes is compiled over the GPU host's own `topology`, with that host's compilation cache once
    a run there has tuned the step."""

    generation: str
    devices: int
    path: str | None = None
    """Where the step is saved (`AheadOfTime.save`). None keeps it in the run's persistent compilation
    cache alone, with every later data phase's step (`later_phases`), where the run's compiles find them:
    a job's prepare for its node's run (`--target.topology $DEW_NODE_TOPOLOGY`, row 5460);
    or where `phases` names the steps."""
    capacity_gib: float | None = None
    frontier: bool = False
    """Whether the report lists each rung above the plan that fits at fewer microbatches than every plan
    before it (`compile_ahead`, `FitSearch.above`), and each lighter rung below it that the run's search
    passed over and that fits (`lighter`, `FitSearch.below`)."""
    multiprocessors: int | None = None
    """The multiprocessors of the GPU of `generation` it compiles for (an H100 PCIe's 114, an L40S's 142),
    where it is not the generation's own (`built_target`): what the step's launches derive for and a GPU
    loading it must have (`BuildTarget`). A TPU topology states its own."""
    topology: str | None = None
    """A host's own serialized GPU topology (`host_gpus`), which states the GPUs it compiles for whole: their
    memory and multiprocessors, so it sets neither `capacity_gib` nor `multiprocessors`. A GPU node's prepare
    is handed its node's ($DEW_NODE_TOPOLOGY), so its compiles key the cache the node's run reads."""
    phases: tuple[int, ...] = ()
    """The run's data phases whose steps are compiled in place of the first's, each over a batch of its own
    (`phase_batch`), into the run's persistent compilation cache, where that phase's first compile in the
    run finds them: none, the first step's alone."""
    share: float = DEW_SHARE
    """The share of each GPU XLA's pool takes in the run that loads the step (`dew._devices.pool_share`):
    Dew's own as Dew prepares a training process, or the run's named one. Its plan is admitted against the
    limit it derives (`StatedGpu.share`), not this process's own share, which a prepare packed beside
    another job holds of its own GPU alone (row 7644: 6.78 GiB of an 80 GiB target)."""

    def __post_init__(self) -> None:
        if not 0 < self.share <= 1:
            raise ValueError(f"a pool's share of a GPU is in (0, 1], not {self.share!r}")

    @property
    def memory(self) -> int | None:
        """Each GPU's stated memory in bytes, which its compile-only devices carry (`compile_only_gpus`)."""
        return None if self.capacity_gib is None else int(self.capacity_gib * 2**30)

    def compile_only(self) -> tuple:
        """The compile-only devices the run's mesh is built over (`dew.training.ahead.target_devices`)."""
        from dew.training.ahead import target_devices

        return target_devices(self)

    def compiled[LossT, EffectsT](self, trainer: Trainer[LossT, EffectsT], loader: Loader, steps: int | None,
                                  examples: int | None = None, resume: Resume = "auto",
                                  plan: FitPlan | None = None) -> CompiledAhead:
        """The run's first step compiled for this target and saved (`dew.training.ahead.compiled_for`)."""
        from dew.training.ahead import compiled_for

        return compiled_for(self, trainer, loader, steps, examples, resume, plan)
