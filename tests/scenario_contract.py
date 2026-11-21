"""The contracts docs/design/testing.md's scenarios check ("each scenario checks every contract its flow
touches", §Scenarios), each one function a scenario or a capability-matrix cell calls on the run it already
made: a claim its own run can show costs no second run, and a claim that needs one (a learning claim, over
independent runs) is paid once a scenario."""

import concurrent.futures
import dataclasses
import hashlib
import importlib
import json
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from process_support import device_flags, fresh

from dew._cache import enable_compilation_cache
from dew.checkpoints import Checkpoints
from dew.data.dataset import DataPartition
from dew.objectives.base import Step
from dew.telemetry.instrumentation import compilation_cache_counts, count_compilation_cache
from dew.training import Pause, Trainer

HELD_OUT = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=None)
"""The step a held-out loss is read at: its key fixed, so a run's loss before and after read one draw."""


def assert_steps_finite(scalars: Iterable[tuple[int, Mapping[str, float]]], label: str) -> None:
    """Every committed update's gradient finite: its norm (`train/grad_norm`, the norm of the gradient the
    update applies, an accumulation window's sum, so a microbatch's gradient that is not finite shows in its
    window's commit), read from the scalars a run logged; a run that logged no commit shows nothing and is
    refused. Not `train/accepted`, which a run without dynamic loss scaling sets whatever its gradient
    (`dew.training.transaction`)."""
    norms = [(step, logged["train/grad_norm"]) for step, logged in scalars if "train/grad_norm" in logged]
    assert norms, f"{label}: no committed update logged its gradient's norm"
    lost = [step for step, norm in norms if not np.isfinite(norm)]
    assert not lost, f"{label}: the updates at steps {lost} applied a gradient that was not finite"


def held_out_measures(objective, optimizer, loader_of: Callable[[int], object],
                      held_out_of: Callable[[int], Mapping], *, runs: Iterable[int], steps: int,
                      measure: Callable[[Mapping, Mapping], float] | None = None,
                      **trainer) -> tuple[np.ndarray, np.ndarray]:
    """Each of the independent runs `runs` names, its held-out measure before training and after `steps` steps
    (the runs a learning claim's gate asks for, `learning_gate.assert_learns`, trained as it asks): on the
    run's own held-out batch (`held_out_of(run)`), each run from its own initialization key and data
    (`loader_of(run)`), a Trainer of `trainer`'s options (a rollout's among them). The measure is
    `measure(variables, held_out)`, the objective's loss read at `HELD_OUT` where None, one compiled program
    for every run's reads (a released decoder's loss run eagerly dispatched each operation: 30 s its first
    read and 4 s each later, against 5 s and 0.2 s compiled). A learning claim's contrasts are formed of
    them, one a run, and judged by the learning gate (`learning_gate.assert_learns`)."""
    held_out_loss = jax.jit(lambda variables, held_out: objective.scalar_loss(variables, held_out,
                                                                              HELD_OUT)[0])

    def loss(variables: Mapping, held_out: Mapping) -> float:
        return float(held_out_loss(variables, held_out))

    measured = loss if measure is None else measure
    before, after = [], []
    for run in runs:
        fitted = Trainer(objective, optimizer, key=jax.random.key(run), **trainer)
        start = fitted.initial_state()
        state = fitted.fit(loader_of(run), steps=steps, log_every=steps)
        held_out = held_out_of(run)
        before.append(measured(start.variables, held_out))
        after.append(measured(state.variables, held_out))
    return np.asarray(before), np.asarray(after)


def covering(axes: Mapping[str, Sequence], admits: Callable[[dict], bool] | None = None) -> list[dict]:
    """A small covering set over a scenario's settings axes (docs/design/testing.md, settings and devices),
    chosen pairwise as the kernel matrix chooses its cells (`kernel_matrix.pairwise`): combinations of the
    axes' values its flow `admits` (every one, where None), in which every two values of two axes that an
    admitted combination holds meet, chosen greedily in the combinations' order, so a scenario's cells are
    the same each run; a scenario of one axis, whose values meet no other, each of its values."""
    from kernel_matrix import combinations, pairwise

    every = [combination for combination in combinations(axes) if admits is None or admits(combination)]
    return every if len(axes) == 1 else pairwise(every, [])


# --- a scenario's runs, each in a fresh process ------------------------------------------------------------

def in_fresh_process(module: str, devices: int, function: str, *arguments):
    """`module`'s `function` called on `arguments` (JSON's values) in a fresh process that sees `devices` CPU
    devices (`process_support.device_flags`, the one way a worker of the suite sets its count): what it
    returns (JSON's). A run's device count is its process's, fixed as its backend opens. Dew is imported
    first, as `dew train` imports it, so a run whose record asks XLA for flags this process lacks runs the
    process again with them before jaxlib loads (`_devices.hold_xla_flags`); faulthandler prints a native
    crash's Python stacks, as the suite's other workers do (`process_support.worker_env`)."""
    script = (f"import json, sys\nimport dew\nimport {module}\n"
              f"print(json.dumps(getattr({module}, sys.argv[1])(*json.loads(sys.argv[2]))))")
    return fresh(script, function, json.dumps(arguments), flags=device_flags(devices), platforms="cpu",
                 variables={"PYTHONFAULTHANDLER": "1"})


PAUSE = 1 / 3
"""The share of its length the resume claim's interrupted run pauses at (`Pause`)."""


@dataclass(frozen=True)
class Phase:
    """A run of a scenario's cell: its name (its run directory's), the persistent cache it compiles into (none
    for a cell's own run, which compiles as its process does), the share of its length it pauses at
    (`Pause`), and how it resumes (`TrainerConfig.resume`, `TrainerConfig.data_resume`)."""

    name: str = "cell"
    cache: str | None = None
    pause: float | None = None
    resume: str = "auto"
    data_resume: str = "relaxed"

    def settings(self) -> dict[str, str]:
        """The phase as `dew train` sets it (`--set`)."""
        return {"trainer.name": self.name, "trainer.compilation_cache_dir": str(self.cache),
                "trainer.resume": self.resume, "trainer.data_resume": self.data_resume,
                **({"trainer.stop": json.dumps({"fraction": self.pause})} if self.pause else {})}

    @property
    def stop(self) -> Pause | None:
        """Where the phase's run stops before its end, as `Trainer.fit` takes it."""
        return None if self.pause is None else Pause(self.pause)

    def entered(self) -> None:
        """Compile into the phase's cache from here on, where it names one (`enable_compilation_cache`), and
        count the process's persistent-cache events from here on (`count_compilation_cache`): a run made
        through Dew's API in this process, as `dew train` does both before its run."""
        if self.cache is not None:
            enable_compilation_cache(self.cache)
        count_compilation_cache()


CELL = Phase()
"""A cell's own run."""


def resumption(root: Path) -> tuple[Phase, Phase, Phase]:
    """The resume claim's runs, compiling into one cache under `root`: the run uninterrupted, a run paused at
    `PAUSE`, and that run resumed as one that must continue (`require`) on exactly its data (`exact`)."""
    cache = str(root / "cache")
    return (Phase("whole", cache), Phase("resumed", cache, pause=PAUSE),
            Phase("resumed", cache, resume="require", data_resume="exact"))


@dataclass(frozen=True)
class Recorded:
    """A scenario cell's run as `dew train` replays it in a fresh process that sees `devices` CPU devices
    (`run_kinds.trained_record`, `recorded_run`): its saved record, the settings its command line sets
    (`--set`, `run_kinds.cli_settings`), and the packages outside Dew its record imports, which it trusts."""

    record: Path
    settings: Mapping[str, str]
    devices: int
    trust: tuple[str, ...] = ()

    def trained(self, phase: Phase = CELL) -> tuple[Path, dict[str, float]]:
        """The run trained in `phase`: its run directory, and the persistent-cache counts of its process once
        it has ended (`recorded_run`): every program it compiled, a final validation's and a checkpoint's
        after its last logged step among them, as a `Called` run's are."""
        ran = in_fresh_process("scenario_contract", self.devices, "recorded_run", str(self.record),
                               {**self.settings, **phase.settings()}, list(self.trust))
        return Path(ran["run"]), ran["counts"]


def recorded_run(record: str, settings: Mapping[str, str], trust: Sequence[str]) -> dict:
    """The run `dew train` replays from `record` with `settings` set and `trust` trusted, in this process
    (`run_kinds.trained_record`; a fresh one, `Recorded.trained`): its run directory, and this process's
    persistent-cache counts once it has ended (`compilation_cache_counts`)."""
    from run_kinds import trained_record

    count_compilation_cache()
    run = trained_record(Path(record), settings, trust=tuple(trust))
    return {"run": str(run), "counts": compilation_cache_counts()}


@dataclass(frozen=True)
class Called:
    """A scenario cell's run made through Dew's API by `module`'s `function`, called in a fresh process that
    sees `devices` CPU devices on the cell, its root and the phase (`Phase`, a mapping), returning its
    process's persistent-cache counts; its run directory `root/runs/<the phase's name>`."""

    module: str
    function: str
    cell: Mapping
    root: Path
    devices: int

    def trained(self, phase: Phase = CELL) -> tuple[Path, dict[str, float]]:
        """The run trained in `phase`: its run directory, and its process's persistent-cache counts."""
        counts = in_fresh_process(self.module, self.devices, self.function, dict(self.cell), str(self.root),
                                  dataclasses.asdict(phase))
        return self.root / "runs" / phase.name, counts


def checkpointed(run: Path) -> dict[str, tuple]:
    """Everything the run's latest checkpoint holds, each state leaf by its path as its dtype, shape and bytes
    (a key's as its raw bits'), and the data position it resumes from, as bytes: the run's checkpoint as its
    bits, where a zero's sign is a bit too."""
    checkpoints = Checkpoints(str(run))
    state, _ = checkpoints.restore()
    held = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(state):
        keyed = jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key)
        value = np.asarray(jax.random.key_data(leaf) if keyed else leaf)
        held[jax.tree_util.keystr(path)] = (str(value.dtype), value.shape, value.tobytes())
    return {**held, "position": checkpoints.position(checkpoints.latest, DataPartition()) or b""}


def written(cache: Path) -> set[str]:
    """The files a persistent compilation cache holds: one a program a compile wrote (with every program
    cached, `enable_compilation_cache`), beside the lowerings it keeps."""
    return {str(path.relative_to(cache)) for path in cache.rglob("*") if path.is_file()} if cache.is_dir() \
        else set()


def assert_resumes_onto_its_uninterrupted_run(prepared: Recorded | Called, root: Path) -> Path:
    """Resume equals the uninterrupted run, and a resumed run reuses its compiled programs (testing.md's
    contracts): `prepared`'s run uninterrupted and, beside it, paused at `PAUSE`, then resumed
    (`resumption`), each in a fresh process compiling into one persistent cache. The run pauses before its
    end, and resumed ends at the uninterrupted run's step holding everything its final checkpoint holds
    (`checkpointed`), bit for bit; and its process, to its end (its final validation and checkpoint too),
    compiles no program the runs before it did not: it writes nothing to the cache (`written`), and found
    programs there. A checkpoint that holds an average (`TrainState.ema`) restores it from its difference to
    the weights, a program the save that wrote it compiles (`dew.checkpoints._state_trees.decoded_ahead`):
    the run resumed a second time from the same pause, its own copy of the paused run in a fresh process
    beside the first, is held alike, both against the cache the runs before them left, and each ends where
    the uninterrupted run ends. The uninterrupted run's directory, which a claim of `prepared`'s trained run
    reads after (a sampling's, `assert_samples_from_its_cached_programs`)."""
    whole_phase, paused_phase, resumed_phase = resumption(root)
    cache = Path(str(resumed_phase.cache))
    # Two processes of their own, run side by side: each its own run directory, one cache, compiling alike.
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        (whole, _), (paused, _) = pool.map(prepared.trained, (whole_phase, paused_phase))
    end = Checkpoints(str(whole)).latest
    assert end is not None and 0 < Checkpoints(str(paused)).latest < end
    resumes = [resumed_phase]
    if Checkpoints(str(paused)).stored().get("ema") is not None:
        again = dataclasses.replace(resumed_phase, name=f"{resumed_phase.name}-again")
        shutil.copytree(paused, paused.with_name(again.name))
        resumes.append(again)
    held = checkpointed(whole)
    before = written(cache)
    # Side by side, as the uninterrupted and paused runs are: each its own copy of the paused run.
    with concurrent.futures.ThreadPoolExecutor(len(resumes)) as pool:
        finished = list(pool.map(prepared.trained, resumes))
    compiled = written(cache) - before
    for phase, (resumed, counts) in zip(resumes, finished, strict=True):
        assert Checkpoints(str(resumed)).latest == end, phase.name
        ran = checkpointed(resumed)
        assert held.keys() == ran.keys(), set(held) ^ set(ran)
        off = [path for path, value in held.items() if value != ran[path]]
        assert not off, f"{phase.name}: the resumed run's leaves off the uninterrupted run's: {off}"
        # Every program read from the cache, none compiled again: a program compiled anew over an entry the
        # cache holds writes no new file (its LRU leaves the key as it is) but counts a miss.
        missed, hit = counts["train/cache_misses"], counts["train/cache_hits"]
        assert missed == 0 and hit > 0, (phase.name, counts)
    # Nothing written after the last logged step either, which its counts do not reach.
    assert not compiled, sorted(compiled)
    return whole


def cached(cache: str, module: str, function: str, *arguments) -> dict:
    """`module`'s `function` on `arguments`, compiling into the persistent cache `cache` from the start: its
    process's persistent-cache counts, and what it draws (an array's dtype, shape and its bytes' digest)."""
    enable_compilation_cache(cache)
    count_compilation_cache()
    drawn = np.asarray(getattr(importlib.import_module(module), function)(*arguments))
    return {"counts": compilation_cache_counts(),
            "drawn": [str(drawn.dtype), list(drawn.shape), hashlib.sha256(drawn.tobytes()).hexdigest()]}


def assert_samples_from_its_cached_programs(module: str, function: str, *arguments, root: Path) -> None:
    """A repeated run reuses its compiled programs, sampling (testing.md's contracts): `module`'s sampling
    `function` on `arguments`, run twice, each time in a fresh process compiling into one persistent cache
    under `root` (`cached`). The first fills it; the second compiles nothing the first did not (none missed
    the cache, and it found one) and draws what the first drew, bit for bit."""
    cache = str(root / "sampling-cache")
    first, second = (in_fresh_process("scenario_contract", 1, "cached", cache, module, function, *arguments)
                     for _ in range(2))
    assert first["counts"]["train/cache_misses"] > 0, first["counts"]
    assert second["counts"]["train/cache_misses"] == 0 and second["counts"]["train/cache_hits"] > 0, (
        second["counts"])
    assert second["drawn"] == first["drawn"]


# --- one first step under two layouts, held by check 1 (`placement_parity.layouts_agree`) -----------------

def first_steps(objective_of: Callable[[bool], object], laid: Callable[[str, int], Mapping], min_shard: int,
                reference: tuple[str, int, int]) -> Callable:
    """A scenario's first step at an arm, as `placement_parity.layouts_agree` runs one (`run(arm, seed)`): the
    loss and the gradient its trainer hands the optimizer (`layout_parity.stash`) on draw `seed`'s batch, as
    the arm's rows lay it (`laid(rows, seed)`, the scenario's own data path), over the arm's devices' fsdp
    split, read in its microbatches (`Trainer`'s `split`): `(rows, devices, split)`, its leaves sharded from
    `min_shard` up. Every arm steps from one start:
    the `reference` arm's trainer's draw of it, placed on each arm's mesh as that arm's trainer places its
    state, a truth's widened (`arm_arithmetic.float64_held`) for its model in float64 (`objective_of(wide)`).
    The batch reaches the step as `fit` hands it (`step_batch`), placed on the arm's mesh; each arm's step
    compiles once a precision."""
    from arm_arithmetic import float64_held
    from placement_parity import x64
    from tool_support import load

    from dew.distributed.batches import shard_batch
    from dew.distributed.placement import Layout, MeshSpec
    from dew.training.ahead import step_batch

    stash = load("layout_parity").stash
    built, drawn = {}, {}

    def trainer_of(rows: str, devices: int, split: int, wide: bool) -> Trainer:
        trainer = Trainer(objective_of(wide), optax.chain(stash(), optax.sgd(1e-3)), key=jax.random.key(0),
                          mesh=MeshSpec(fsdp=devices), layout=Layout(min_shard=min_shard), split=split,
                          checkpoints=None, tracker=None, param_dtype="float64" if wide else None)
        trainer.device_mesh = trainer.mesh.build(jax.devices()[:devices])
        return trainer

    def start():
        """The reference arm's start, drawn narrow, as its trainer places it."""
        if not drawn:
            with jax.enable_x64(new_val=False):
                drawn["start"] = trainer_of(*reference, wide=False).place()[0]
        return drawn["start"]

    def compiled(rows: str, devices: int, split: int):
        wide = x64()
        if (rows, devices, split, wide) not in built:
            trainer = trainer_of(rows, devices, split, wide)
            _, shardings, _ = trainer.place()
            state = jax.tree.map(lambda leaf: float64_held(leaf)
                                 if wide and jnp.issubdtype(leaf.dtype, jnp.floating) else leaf, start())

            def placed():
                # A copy a call: the step donates the state it is handed.
                return jax.tree.map(jnp.copy, jax.device_put(state, shardings))

            batch = shard_batch(trainer.device_mesh, step_batch(trainer, laid(rows, 0), DataPartition()))
            built[rows, devices, split, wide] = trainer, trainer.compile(placed(), batch), placed
        return built[rows, devices, split, wide]

    def run(arm: tuple[str, int, int], seed) -> dict:
        rows, devices, split = arm
        trainer, step, placed = compiled(rows, devices, split)
        batch = shard_batch(trainer.device_mesh, step_batch(trainer, laid(rows, int(seed)), DataPartition()))
        state, loss, *_ = step(placed(), batch)
        return {"loss": loss, "gradient": jax.device_get(trainer.stored(state).opt_state[0]["gradient"])}

    return run
