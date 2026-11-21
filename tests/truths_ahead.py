"""Check 1's float64 programs compiled ahead of the run that computes its truths (the GPU tier, row 5461;
bible §8: float64 truth runs on a GPU's FP64 units, and a GPU-free prepare holds setup and safe AOT
compilation).

Within `compiled_truths(directory, prepare=True, ...)` a check-1 site compiles every program its truths and
terms take, at each shape its draws take (`assert_as_exact`'s stages), for the devices its run computes on
(a GPU node's, compile-only, from its prepare on the node's CPU: `dew.training.ahead.compile_programs`),
serializes them under `directory` and runs nothing. Within `compiled_truths(directory, prepare=False, ...)`
the run computes each draw's truth and terms on its own devices with those executables (`called`), tracing,
lowering and compiling none of them; the site's verdict is reached as a site computing its own reaches it.
Its arms (`ARMS`: Dew's, the reference and each further one) are compiled there too, each as its run compiles
it (`jax.jit`) at each of its draws' own shapes, traced for the run's devices (`CompiledTruths.tracing`), so
a GPU run traces, lowers and compiles none of a site's programs: each compiled at a run's first call,
those ran 2,000 of a day's GPU-minutes idle (2026-11-17's census of 226 jobs). A prepare lists them from
the site's own call (`reference_error.assert_as_exact`), the one definition its run reads.
A program the store does not hold for a draw's shapes or an arm's statement, or one compiled at another
`stamp` (the commit and JAX), is a red (`MissingTruth`), never one compiled in its place; a site its
prepare did not reach, and an arm no statement holds, runs unprepared, every draw judged, recorded in the
test's store (`unprepared`); a terms walk its values steer (a branch or loop it reads as it runs,
`float64_terms.static`) is held as one, which its run walks on the GPU, named (`AT_RUN`).

A program takes its traced constants as arguments, held beside it (`held.pkl`): no executable serializes an
array it closed over. The roles are a site's truth (`truth`: its float64 program, `arm_arithmetic.Truth`),
the terms its rows are sized by (`terms`: the float64 walk of the truth, `float64_terms`, whose compiled run
the store holds for the run's first draw to decide on, `reference_error._cancelling_terms`) and a site's own
terms program (`site terms`, `reference_error._SiteTerms`). A truth the gate runs as given (NumPy, torch, a
recording) has no program, and runs as given."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import inspect
import io
import json
import logging
import os
import pickle
import shutil
import types
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import numpy as np
from store_wire import COMPILED, SITE_IN_PROGRESS
from tool_support import load


class MissingTruth(AssertionError):
    """A check-1 site under a store of programs compiled ahead (`compiled_truths`) whose prepare compiled its
    programs for another commit, for other shapes, or of another arm than the run's: a red with its reason,
    never a program compiled here in its place (the GPU tier's run compiles none of a verdict's truths, row
    5461). A site its prepare did not reach runs unprepared, recorded (`running`)."""


PROGRAMS, HELD, SITE = "programs", "held.pkl", "site.json"
"""A site's files under its store (`CompiledTruths.path`): its executables (`AheadPrograms`), each one's
constants and output tree (`held`), and its record, written last, so a prepare cut short leaves none a run
reads."""

UNPREPARED = "unprepared.json"
"""A test's record, in its store, that it ran unprepared (`unprepared`): `{"test", "where", "error",
"reasons"}`, its first reason's where and why and every reason it gave, each once (a site, an arm, a cell:
their count shows a tier drifting from its prepares), the GPU tier's list of unprepared tests read from these
alone (a ledger row each, the tier's)."""

AT_RUN = "at_run.json"
"""A test's record, in its store, of what its run compiled on its GPU as its prepare declared, as
`UNPREPARED`'s (`CompiledTruths.unprepared`'s `record`): a site's truths or arms its prepare wrote at run
(`compile_site`'s `at_run`: a choice the GPU makes that no compile without it holds, a convolution's
algorithm), and a terms walk its values steer (`float64_terms._termed`); a decision named in the run's
summary, not drift."""

DEFERRED = "deferred.json"
"""A test's record, in its store, of a GPU flow's cell its prepare deferred to its run (`unprepared`'s, as
`UNPREPARED`'s), which its run serves from the store beside it and then records unprepared."""


@dataclasses.dataclass(frozen=True)
class HeldName:
    """A function a site's held record holds that no module holds (a local hook a launch took, `Seen.held`),
    as its run reads it: by its module and qualified name (`_held_bytes`). Its run reports the launch and
    calls the program its prepare compiled, never the hook; a call of it is refused."""

    name: str

    def __call__(self, *args, **kwargs):
        raise MissingTruth(f"{self.name}: a function its prepare held by its name alone, which its run does "
                           "not call")


class _HeldPickler(pickle.Pickler):
    """A site's held record pickled as pickle pickles it, but a function pickle names by no module
    (`HeldName`): a local hook of a launch (a GEMM prologue's, a derivative's span) is no function its run
    calls (rows 9584, 9600: a held launch refused the store, its arms compiled nothing ahead)."""

    def reducer_override(self, obj):
        if not isinstance(obj, types.FunctionType):
            return NotImplemented
        try:
            pickle.dumps(obj)
        except (pickle.PicklingError, AttributeError):
            return HeldName, (f"{obj.__module__}.{obj.__qualname__}",)
        return NotImplemented


def _held_bytes(held: dict) -> bytes:
    """`held` as its site's store keeps it (`_HeldPickler`)."""
    written = io.BytesIO()
    _HeldPickler(written).dump(held)
    return written.getvalue()


def named(role: str, inputs) -> str:
    """A program's name in its site's store: its `role` and its inputs' tree, shapes, dtypes and weak types,
    as a trace keys them (`float64_terms.traced`): a weak Python scalar beside a bfloat16 array keeps it
    bfloat16, a strong one promotes it."""
    leaves, tree = jax.tree.flatten(inputs)
    shapes = (str(tree), tuple((tuple(np.shape(leaf)), str(jax.typeof(leaf).dtype),
                                jax.typeof(leaf).weak_type) for leaf in leaves))
    return f"{role} {hashlib.sha256(repr(shapes).encode()).hexdigest()[:16]}"


@dataclasses.dataclass
class CompiledTruths:
    """A store of check-1 sites' float64 programs compiled ahead (`compiled_truths`): `prepare`, whether this
    process compiles them (for `target`, a GPU's facts, `dew.nn.kernels.generation.GpuTarget`; None for its
    own devices) or reads them, at `stamp`."""

    directory: Path
    prepare: bool
    stamp: str
    target: object | None = None
    begun: set[str] = dataclasses.field(default_factory=set)
    # Each site a check's prepare began to prepare, from before its first trace, and each it finished whole,
    # its truths and its arms (`preparing`).
    tried: set[str] = dataclasses.field(default_factory=set)
    done: set[str] = dataclasses.field(default_factory=set)
    read: dict[str, tuple[dict, dict]] = dataclasses.field(default_factory=dict)
    # Each arm with no site its prepare compiled (`reference_error.ahead_arm`), by its label: its program's
    # name (`named`), its outputs' shapes and the launches its kept lowering took, which a second call of
    # the label at the same program returns, compiling nothing again.
    arms: dict[str, tuple[str, Any, Any]] = dataclasses.field(default_factory=dict)

    def path(self, label: str) -> Path:
        return self.directory / hashlib.sha256(label.encode()).hexdigest()[:32]

    @contextlib.contextmanager
    def preparing(self, *labels: str) -> Iterator[None]:
        """Within the block, a check's prepare of its sites `labels` (a site's truths and its arms, `armed`):
        recorded tried before its first trace, and finished where it ran to its end, every program it
        compiles written; a cell's later failure, its every tried site finished, checks what its prepare
        gives in their place (`gpu_flow.run_cells`)."""
        self.tried.update(labels)
        yield
        self.done.update(labels)

    def judged_its_zeros(self, error: BaseException, before: frozenset[str] = frozenset()) -> bool:
        """Whether `error`, raised in this store's prepare, checked what the prepare gave in place of the
        sites and arms it tried since `before` (its `tried` then; the test's whole where none is given): one
        at least tried, every one finished (`preparing`), and the error any but a missing program
        (`MissingTruth`). Such a check reads an `ahead_arm`'s zeros, which judges nothing, its run judging it
        (rows 8555, 10169): the one rule a flow's cell (`gpu_flow._deferred`), a test (tests/conftest.py's
        `judged_on_zeros`) and a test's child process (`of_the_parent`'s callers) pass their prepare by."""
        new = self.tried - before
        return self.prepare and bool(new) and new <= self.done and not isinstance(error, MissingTruth)

    def arm_again(self, label: str, name: str):
        """A prepare's call of an arm with no site (`reference_error.ahead_arm`) whose label it compiled
        (`arms`): its program's zeros (of the shapes its own trace gave, in the store's tracing: a jit of its
        own compiler options traced again inside another, `eval_shape`'s, was refused) and its kept
        lowering's launches, noted as its run reports them (`ran`); refused where the call is another program
        (`name`, its role and its inputs' shapes) than the one its label names."""
        from dew.nn.kernels import tuning

        compiled, shapes, launches = self.arms[label]
        if name != compiled:
            raise ValueError(f"{label}: called in one prepare at two programs (another role or other "
                             "inputs' shapes than its first call's): one label names one program")
        tuning.noted(launches)
        return jax.tree.map(lambda shape: np.zeros(shape.shape, shape.dtype), shapes)

    def holds(self, label: str) -> bool:
        """Whether the store's prepare reached site `label` (its programs written, at any stamp)."""
        return (self.path(label) / SITE).is_file()

    def at_run(self, label: str) -> str | None:
        """Why the store's prepare compiled none of site `label`'s programs, its run compiling each on its
        GPU (`compile_site`'s `at_run`), at this stamp; None for a site it compiled or did not reach."""
        site = self.path(label) / SITE
        if not site.is_file():
            return None
        stored = json.loads(site.read_text())
        return stored.get("at_run") if stored["stamp"] == self.stamp else None

    def unprepared(self, why: str, record: str = UNPREPARED) -> None:
        """Records in the store that its test ran a site, or an arm, unprepared (`unprepared`), where in its
        file this process stands; under `AT_RUN`, that its run compiled it on its GPU as its prepare
        declared."""
        test = os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0]
        unprepared(self.directory, test, [(frame.filename, frame.lineno) for frame in inspect.stack(0)], why,
                   record)

    def device(self):
        """A device of the ones the store's programs compute on, whose platform picks a branch its platform
        picks (`float64_terms.evaluated_on`): a compile-only GPU of the target's generation, else this
        process's first."""
        if self.target is None:
            return jax.devices()[0]
        from dew.nn.kernels.generation import compile_only_gpus

        return compile_only_gpus(self.target.generation, 1)[0]

    @contextlib.contextmanager
    def on_its_gpu(self) -> Iterator[None]:
        """Within the block, this process as on the store's target's GPU: that GPU's compile-only device its
        default (`device`), on which nothing computes, and every kernel chosen as on that GPU
        (`generation.assumed`: its generation and facts, which a compile-only device placed off the CPU
        backend reads as the bare platform 'gpu', every kernel it takes refused, rows 8076 and 10088); as it
        is where the store names no target. A part's prepare (`prepared`) and its child process's
        (`of_the_parent`) alike."""
        if self.target is None:
            yield
            return
        from dew.nn.kernels.generation import assumed

        with jax.default_device(self.device()), assumed(self.target):
            yield

    @contextlib.contextmanager
    def tracing(self) -> Iterator[None]:
        """Within the block, a program traces as the store's compile lowers it: for its target's GPUs
        (`dew.training.ahead.lowering_for`), or this process's devices where it names none. An arm traced in
        it holds the arrays its compile ahead holds (`reference_error._held_arm`)."""
        if self.target is None:
            yield
            return
        from dew.training.ahead import lowering_for

        with lowering_for(self.target.generation, self.target.multiprocessors):
            yield

    def compile_site(self, label: str, programs: Mapping[str, tuple[Callable, Sequence]],
                     held: Mapping[str, dict], *, compiled: Callable | None = None, arms: bool = False,
                     host: bool = False, at_run: str | None = None) -> None:
        """The site's `programs` (each a function of its constants and inputs, flat, and their shapes, by
        its name, `named`) compiled for the store's target and written with what each `held`; refused for a
        label this store compiled already, whose programs a second site of it would replace. `compiled` makes
        each program as its run's compile does (`dew.training.ahead.compile_programs`' `compiled`; Dew's
        compile point where None, a truth's, whose program is its run's); `arms` whether they are a site's
        arms (`compiled_memory`), whose state its run holds while its truths compute; `host` whether they
        are compiled for the host's CPU, whatever the store's target (an arm placed there, `ahead_arm`'s);
        `at_run` why none is compiled ahead, their run compiling each on its GPU (`at_run`), the site written
        with no programs and each one's memory read offline (`compile_programs`' `memory_only`), by which
        the lane sizes its worker (`compiled_memory`). A site its target's compile refuses
        (`report.CompiledOnDevice`: a choice its GPU makes at run time) is left unbegun, its refusal raised
        for its caller to place it otherwise."""
        from dew._cache import uncached
        from dew.nn.kernels.templates import report

        if label in self.begun:
            raise ValueError(f"{label}: two sites of one label in one store of compiled truths, where the "
                             "second would replace the first's: each needs its own")
        self.begun.add(label)
        site = self.path(label)
        if site.exists():
            shutil.rmtree(site)
        site.mkdir(parents=True)
        if at_run is not None:
            with uncached():
                read = self._compiled(programs, compiled, host, memory_only=True)
            memory = {name: executable.memory._asdict() for name, executable in read.programs.items()
                      if executable.memory is not None}
            (site / SITE).write_text(json.dumps({"stamp": self.stamp, "label": label, "memory": memory,
                                                 "arms": arms, "host": host, "at_run": at_run}))
            return
        # Compiled here, never loaded from JAX's persistent cache, whose executables serialize without their
        # kernels (`dew.training.ahead.keeping`).
        try:
            with uncached():
                built = self._compiled(programs, compiled, host)
        except report.CompiledOnDevice:
            self.begun.discard(label)
            shutil.rmtree(site)
            raise
        built.save(site / PROGRAMS)
        (site / HELD).write_bytes(_held_bytes(dict(held)))
        memory = {name: executable.memory._asdict() for name, executable in built.programs.items()
                  if executable.memory is not None}
        (site / SITE).write_text(json.dumps({"stamp": self.stamp, "label": label, "memory": memory,
                                             "arms": arms, "host": host}))

    def _compiled(self, programs: Mapping[str, tuple[Callable, Sequence]], compiled: Callable | None,
                  host: bool = False, *, memory_only: bool = False):
        """`programs` compiled for the store's target (`compile_programs`), or for this process's first device
        where it names none, or for the host's first CPU (`host`), each made by `compiled` where it is given
        (`compile_site`), else by Dew's compile point on a target and `jax.jit` here; for a target, read for
        their memory alone where `memory_only` (`compile_programs`')."""
        from dew.training.ahead import AheadPrograms, BuildTarget, Executable, compile_programs

        if self.target is not None and not host:
            return compile_programs(programs, self.target.generation, self.target.multiprocessors,
                                    memory_only=memory_only,
                                    **({} if compiled is None else {"compiled": compiled}))
        devices = jax.devices("cpu" if host else None)[:1]
        placed = jax.sharding.SingleDeviceSharding(devices[0])
        compile_one = compiled or jax.jit

        def built(function, shapes):
            placed_shapes = (jax.ShapeDtypeStruct(np.shape(shape), shape.dtype, sharding=placed,
                                                  weak_type=shape.weak_type)
                             for shape in shapes)
            return Executable.of(compile_one(function, out_shardings=placed).lower(*placed_shapes).compile())

        # Compiled on this process's devices, which made every choice themselves.
        return AheadPrograms(BuildTarget.of(devices), {
            name: built(function, shapes) for name, (function, shapes) in programs.items()},
            tuned=True, at_run=())

    def called(self, label: str, role: str, inputs):
        """The site's program `role` run on `inputs` (its draw's: a truth's widened, an arm's as drawn) and
        the constants it holds, on this process's devices, its outputs fetched to the host as a site's
        truths are held, its device's memory left to the next (a model's float64 gradients a draw, 2.9 GB
        on the sinks flows); a red (`MissingTruth`) where the store holds none for the site at this stamp or
        for these shapes."""
        programs, held = self._site(label)
        name = named(role, inputs)
        if name not in programs:
            raise MissingTruth(f"{label}: no {role} compiled ahead for these inputs' shapes (its prepare "
                               f"drew others)")
        return jax.device_get(programs[name](*held[name]["constants"], *jax.tree.leaves(inputs)))

    def ran(self, label: str, role: str, inputs):
        """The site's arm `role` run on `inputs` (`called`), reporting the routes its prepare's trace of it
        reported (`generation.reported`) and recording the launches it took and their configs
        (`tuning.noted`), as a trace of it in the run would: the run calls its executable and traces none
        (row 8374: a matrix cell's routes were empty on its GPU; row 8580: a pinned cell's launches)."""
        from dew.nn.kernels import generation, tuning

        found = self.called(label, role, inputs)
        record = self._site(label)[1][named(role, inputs)]
        generation.reported(record["routes"])
        tuning.noted(record["launches"])
        return found

    def walked(self, label: str, role: str, inputs) -> str | None:
        """Why the site's program `role` at `inputs` is one its run walks as it runs (`float64_terms`), as its
        prepare stated, having found it steered by its values, which no program compiled ahead holds; None
        for one its prepare compiled."""
        return self._site(label)[1].get(named(role, inputs), {}).get("walked")

    def tree(self, label: str, role: str, inputs):
        """The output tree the site's program `role` at `inputs` was traced with (`held`)."""
        return self._site(label)[1][named(role, inputs)]["tree"]

    def _site(self, label: str) -> tuple[dict, dict]:
        if label not in self.read:
            from dew.training.ahead import AheadPrograms

            site = self.path(label)
            if not (site / SITE).is_file():
                raise MissingTruth(f"{label}: no programs compiled ahead under {self.directory} (the tier's "
                                   f"prepare did not reach this site)")
            stored = json.loads((site / SITE).read_text())
            if stored["stamp"] != self.stamp:
                raise MissingTruth(f"{label}: its programs were compiled at {stored['stamp']}, not this "
                                   f"run's {self.stamp}")
            self.read[label] = (AheadPrograms.load(site / PROGRAMS).loaded(),
                                pickle.loads((site / HELD).read_bytes()))
        return self.read[label]


def compiled_memory(directory: Path, *, arms: bool = False) -> list[dict[str, int]]:
    """Every program of a store's sites (`compiled_truths`, prepared), its truths' or, with `arms`, its
    arms' (`CompiledTruths.compile_site`), as its compile analysed its memory
    (`dew.telemetry.instrumentation.CompiledMemory`, as its fields name it), its sites' record's; none for
    a program whose backend stated none, nor for one compiled for the host's CPU (`compile_site`'s `host`),
    which holds none of the devices' memory."""
    records = (json.loads(site.read_text()) for site in Path(directory).glob(f"*/{SITE}"))
    return [memory for record in records
            if record.get("arms", False) == arms and not record.get("host", False)
            for memory in record.get("memory", {}).values()]


def admitted_needs(arms: Sequence[dict[str, int]], truths: Sequence[dict[str, int]]) -> list[dict[str, int]]:
    """A run's programs as the allocator admits them (`dew.training.memory.admitted`), each its compile's
    memory (`compiled_memory`'s, `CompiledMemory`'s fields) in the admission's own terms
    (`dew.training.memory.admission`): an arm's alone, a float64 truth's or terms program's with the arms'
    state held beside it (the most any arm takes as arguments, which the run keeps while its truths compute;
    every program's outputs are fetched to the host, `CompiledTruths.called`). A part's worker fits a slot
    where every program of its units is admitted there (`tier_wire.unit_needs`)."""
    from dew.telemetry.instrumentation import CompiledMemory
    from dew.training.memory import admission

    arms_memory = [CompiledMemory(**memory) for memory in arms]
    state = max((memory.argument for memory in arms_memory), default=0)

    def admitted_as(memory: CompiledMemory, held: int) -> dict[str, int]:
        return dict(zip(("resident", "beside", "temporaries"), admission(memory, held), strict=True))

    return [*(admitted_as(memory, 0) for memory in arms_memory),
            *(admitted_as(CompiledMemory(**memory), state) for memory in truths)]


READ: list[Path] = []
"""Each store this process's runs read, in order (`compiled_truths`): a test's own and a flow's alike, whose
records of what its run compiled on its GPU (`AT_RUN`, `UNPREPARED`) the run's summary names
(tests/conftest.py's `compiled_ahead`)."""


@contextlib.contextmanager
def compiled_truths(directory: str | Path, *, prepare: bool, stamp: str,
                    target=None) -> Iterator[CompiledTruths]:
    """Within the block, check 1's sites compile their float64 programs ahead into `directory` (`prepare`)
    or compute their truths from those programs (the module's account)."""
    store = CompiledTruths(Path(directory), prepare, stamp, target)
    if not prepare:
        READ.append(store.directory)
    token = COMPILED.set(store)
    try:
        yield store
    finally:
        COMPILED.reset(token)


@contextlib.contextmanager
def prepared(directory: Path, *, stamp: str, target) -> Iterator[CompiledTruths]:
    """A GPU tier part's test's prepare (`tests/conftest.py`'s `compiled_ahead`): its check-1 sites compiled
    into `directory` for the node's GPU (`target`, its `GpuTarget`: `generation_support.node_target`), this
    process as on that GPU (`CompiledTruths.on_its_gpu`), and their programs' needs (`admitted_needs`) filed
    under the part's unit as it closes, whatever its test ended in (`tier_wire.unit_needs`, as a flow's are,
    row 7900), by which the lane sizes the unit's worker."""
    import tier_wire

    try:
        with (compiled_truths(directory, prepare=True, stamp=stamp, target=target) as store,
              store.on_its_gpu()):
            yield store
    finally:
        arms, truths = compiled_memory(directory, arms=True), compiled_memory(directory)
        tier_wire.unit_needs(admitted_needs(arms, truths))


PARENT = "DEW_PARENT_STORE"
"""The environment variable a child process of a test opens its test's store by (`for_a_child`)."""

PARENT_STORE = "parent.pkl"
"""A store's own record of how its test's process opened it, which a child process reads (`of_the_parent`)."""


def for_a_child() -> dict[str, str]:
    """The environment a child process of this test opens its test's store with (`of_the_parent`): the
    store's directory, beside its sites its own record (whether it prepares, its stamp and its target), so a
    computation that needs a process of its own (its XLA flags, a backend option set once a process: the
    approximate divider's) prepares and runs from its test's store, its sites the test's; none outside a
    part's prepare or run."""
    store = COMPILED.get()
    if store is None:
        return {}
    store.directory.mkdir(parents=True, exist_ok=True)
    (store.directory / PARENT_STORE).write_bytes(pickle.dumps((store.prepare, store.stamp, store.target)))
    return {PARENT: str(store.directory)}


@contextlib.contextmanager
def of_the_parent() -> Iterator[CompiledTruths | None]:
    """In a child process of a test (`for_a_child`), its test's store as the test's process opened it:
    preparing it, the process as on its target's GPU as a part's prepare's (`CompiledTruths.on_its_gpu`,
    row 10765: a child's prepare read the bare platform 'gpu'), or reading it; outside one, none."""
    directory = os.environ.get(PARENT)
    if directory is None:
        yield None
        return
    prepare, stamp, target = pickle.loads((Path(directory) / PARENT_STORE).read_bytes())
    with (compiled_truths(directory, prepare=prepare, stamp=stamp, target=target) as store,
          store.on_its_gpu() if prepare else contextlib.nullcontext()):
        yield store


def reading() -> tuple[CompiledTruths, str] | None:
    """The store a run reads the site in progress from, and its label: where a run reads a store
    (`compiled_truths(prepare=False)`) within a site (`store_wire.judged`) its prepare reached, else None. A
    site its prepare did not reach (a cell its prepare skipped, one computing past an earlier site) runs
    unprepared, recorded (`CompiledTruths.unprepared`); one prepared at another stamp is a red
    (`MissingTruth`)."""
    store, label = COMPILED.get(), SITE_IN_PROGRESS.get()
    if store is None or store.prepare or label is None:
        return None
    if not store.holds(label):
        store.unprepared(f"{label}: its prepare did not reach this site")
        return None
    return store, label


def running() -> tuple[CompiledTruths, str] | None:
    """The store whose programs compute the truths of the site in progress, and its label (`reading`), but
    None where its prepare wrote its truths at run (`compile_site`'s `at_run`): its run compiles them on its
    GPU, their float64 on its FP64 units, as a site outside a part does, named (`AT_RUN`)."""
    ran = reading()
    if ran is not None and (why := ran[0].at_run(ran[1])) is not None:
        ran[0].unprepared(f"{ran[1]}: its truths: {why}", AT_RUN)
        return None
    return ran


def unprepared(directory: Path, test: str, frames: Sequence[tuple[str, int]], why: str,
               record: str = UNPREPARED) -> None:
    """The writer of a test's record that it ran unprepared (`UNPREPARED`, in its store `directory`),
    compiling a site or an arm on its devices with every site judged: its node id and each reason it gives,
    where (`frames`, innermost first, `(file, line)`: the first of the test's own file, else the test's file)
    and why (`reasoned`). Under `DEFERRED`, a GPU flow's cell its prepare deferred to its run
    (`gpu_flow.run_cells`), which its run's store still serves, and which the run names unprepared
    (tests/conftest.py's `compiled_ahead`)."""
    # The test's file as CI's grading reads a node id (`tools/armada/grading.py`'s `owning`).
    path = Path(load("grading", "tools/armada").owning(test))
    lines = [line for file, line in frames if Path(file).name == path.name]
    reasoned(directory / record, test, f"{path}:{lines[0]}" if lines else str(path), why)


def reasoned(record: Path, test: str, where: str, why: str) -> None:
    """Into a test's `record` (`UNPREPARED`, `DEFERRED`), a reason it ran unprepared, `where` and `why`, each
    reason once in the order given, its first's its record's own `where` and `error` (a prepare's, before
    its run's): a test with three unprepared arms names three (Main)."""
    reason = {"where": where, "error": why}
    if record.exists():
        held = json.loads(record.read_text())
        if reason not in held["reasons"]:
            held["reasons"].append(reason)
            record.write_text(json.dumps(held))
        return
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"test": test, **reason, "reasons": [reason]}))


_log = logging.getLogger(__name__)


def arm_role(role: str, fn, label: str) -> str | None:
    """An arm's role in its site's store (`arm_store._compile_arms`) with what it computes as Dew states
    an identity (`dew._stated.stated`, one form in every process that holds an equal value): a run that passes
    the site another computation under that role finds none (`MissingTruth`), never the prepare's run in its
    place. None for an arm no statement holds (`Unestablished`), which runs unprepared, named
    (`CompiledTruths.unprepared`)."""
    from dew._stated import Unestablished, stated

    at = f"{label}: its {role} arm"
    try:
        known: dict[int, Any] = {}
        _mutable_known(fn, known, set(), at)
        statement = stated(fn, at, known)
    except Unestablished as unstated:
        _log.info("an arm no statement holds runs unprepared: %s", unstated)
        return None
    return f"{role} {hashlib.sha256(json.dumps(statement, sort_keys=True).encode()).hexdigest()[:16]}"


def _mutable_known(value, known: dict[int, Any], walking: set[int], at: str) -> None:
    """Into `known` (`dew._stated.stated`'s values its caller states), each list, dict or set `value`
    reaches through partials, records, tuples, jits, containers and what each function reads, stated by its
    contents as this process first named an arm reading it (`_FIRST_NAMED`): such a container may be state an
    arm keeps as it runs (a test's record of its calls, a cache a trace fills), which a prepare's trace of the
    arm fills after its first naming and its run, calling the arm's executable, never does; and it may hold
    what the arm computes by (a test's options read off its backend, another in a GPU's run than in its CPU's
    prepare), which names another arm (row 7674: stated by its type alone, the run computed the prepare's
    program). Each container's parts are walked before it is stated, through the one `known` every reached
    container's statement is in, so one a trace filled since its first naming is stated as it was then
    wherever it lies, inside another or beside an alias of itself (the review's: a dict read its record
    trace-filled in a prepare, empty in its run)."""
    from dew._sources import function_reads
    from dew._stated import library_function, stated

    if id(value) in walking:
        return
    walking.add(id(value))
    container = isinstance(value, (list, dict, set))
    if container:
        parts = tuple(value.values()) if isinstance(value, dict) else tuple(value)
    elif isinstance(value, functools.partial):
        parts = (value.func, *value.args, *value.keywords.values())
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        parts = tuple(getattr(value, field.name) for field in dataclasses.fields(value))
    elif isinstance(value, tuple):
        parts = value
    elif isinstance(value, types.FunctionType):
        # A library's function, its globals its module's (`dew._stated.library_function`), as its statement
        # walks it: its closure and defaults, a user's function and what it reads among them.
        cells, read = function_reads(value)
        parts = (*cells.values(), *(() if library_function(value) else read.values()),
                 *(value.__defaults__ or ()), *(value.__kwdefaults__ or {}).values())
    elif (wrapped := getattr(value, "__wrapped__", None)) is not None:
        parts = (wrapped,)
    else:
        return
    for part in parts:
        _mutable_known(part, known, walking, at)
    if container:
        named = _FIRST_NAMED.get(id(value))
        if named is None:
            named = _FIRST_NAMED[id(value)] = (value, ["mutable", stated(value, at, known)])
        known[id(value)] = named[1]

_FIRST_NAMED: dict[int, tuple[Any, Any]] = {}
"""Each list, dict or set an arm reads (`_mutable_known`), by its `id`, with its statement as this process
first named an arm reading it, before any trace of that arm wrote to it, as a prepare names it and a run, a
process whose executables write to none, alike; held, so no other container takes its `id`."""
