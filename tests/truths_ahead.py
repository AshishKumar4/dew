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
A program the store does not hold for a draw's shapes, or one compiled at another `stamp` (the commit and
JAX), is a red (`MissingTruth`), never one compiled in its place; a terms walk its values steer (a branch
or loop it reads as it runs, `float64_terms.static`) is held as one, which its run walks on its device.

A program takes its traced constants as arguments, held beside it (`held.pkl`): no executable serializes an
array it closed over. The roles are a site's truth (`truth`: its float64 program, `arm_arithmetic.Truth`),
the terms its rows are sized by (`terms`: the float64 walk of the truth, `float64_terms`, whose compiled run
the store holds for the run's first draw to decide on, `reference_error._cancelling_terms`) and a site's own
terms program (`site terms`, `reference_error._SiteTerms`). A truth the gate runs as given (NumPy, torch, a
recording) has no program, and runs as given."""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import hashlib
import json
import pickle
import shutil
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

import jax
import numpy as np


class MissingTruth(AssertionError):
    """A check-1 site under a store of programs compiled ahead (`compiled_truths`) that its prepare did not
    compile, or compiled for another commit or other shapes: a red with its reason, never a program compiled
    here in its place (the GPU tier's run compiles none of a verdict's truths, row 5461)."""


PROGRAMS, HELD, SITE = "programs", "held.pkl", "site.json"
"""A site's files under its store (`CompiledTruths.path`): its executables (`AheadPrograms`), each one's
constants and output tree (`held`), and its record, written last, so a prepare cut short leaves none a run
reads."""


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
    read: dict[str, tuple[dict, dict]] = dataclasses.field(default_factory=dict)

    def path(self, label: str) -> Path:
        return self.directory / hashlib.sha256(label.encode()).hexdigest()[:32]

    def device(self):
        """A device of the ones the store's programs compute on, whose platform picks a branch its platform
        picks (`float64_terms.evaluated_on`): a compile-only GPU of the target's generation, else this
        process's first."""
        if self.target is None:
            return jax.devices()[0]
        from dew.nn.kernels.generation import compile_only_gpus

        return compile_only_gpus(self.target.generation, 1)[0]

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
                     held: Mapping[str, dict], *, compiled: Callable | None = None) -> None:
        """The site's `programs` (each a function of its constants and inputs, flat, and their shapes, by
        its name, `named`) compiled for the store's target and written with what each `held`; refused for a
        label this store compiled already, whose programs a second site of it would replace. `compiled` makes
        each program as its run's compile does (`dew.training.ahead.compile_programs`' `compiled`; Dew's
        compile point where None, a truth's, whose program is its run's)."""
        from dew._cache import uncached

        if label in self.begun:
            raise ValueError(f"{label}: two sites of one label in one store of compiled truths, where the "
                             "second would replace the first's: each needs its own")
        self.begun.add(label)
        site = self.path(label)
        if site.exists():
            shutil.rmtree(site)
        site.mkdir(parents=True)
        # Compiled here, never loaded from JAX's persistent cache, whose executables serialize without their
        # kernels (`dew.training.ahead.keeping`).
        with uncached():
            built = self._compiled(programs, compiled)
        built.save(site / PROGRAMS)
        (site / HELD).write_bytes(pickle.dumps(dict(held)))
        memory = {name: executable.memory._asdict() for name, executable in built.programs.items()
                  if executable.memory is not None}
        (site / SITE).write_text(json.dumps({"stamp": self.stamp, "label": label, "memory": memory}))

    def _compiled(self, programs: Mapping[str, tuple[Callable, Sequence]], compiled: Callable | None):
        """`programs` compiled for the store's target (`compile_programs`), or for this process's first device
        where it names none, each made by `compiled` where it is given (`compile_site`), else by Dew's compile
        point on a target and `jax.jit` here."""
        from dew.training.ahead import AheadPrograms, BuildTarget, Executable, compile_programs

        if self.target is not None:
            return compile_programs(programs, self.target.generation, self.target.multiprocessors,
                                    **({} if compiled is None else {"compiled": compiled}))
        devices = jax.devices()[:1]
        placed = jax.sharding.SingleDeviceSharding(devices[0])
        compile_one = compiled or jax.jit

        def built(function, shapes):
            placed_shapes = (jax.ShapeDtypeStruct(np.shape(shape), shape.dtype, sharding=placed,
                                                  weak_type=shape.weak_type)
                             for shape in shapes)
            return Executable.of(compile_one(function, out_shardings=placed).lower(*placed_shapes).compile())

        return AheadPrograms(BuildTarget.of(devices), {
            name: built(function, shapes) for name, (function, shapes) in programs.items()})

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

    def walked(self, label: str, role: str, inputs) -> bool:
        """Whether the site's program `role` at `inputs` is one its run walks as it runs (`float64_terms`),
        its prepare having found it steered by its values, which no program compiled ahead holds."""
        return "walked" in self._site(label)[1].get(named(role, inputs), {})

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


def compiled_memory(directory: Path) -> list[dict[str, int]]:
    """Every program of a store's sites (`compiled_truths`, prepared) as its compile analysed its memory
    (`dew.telemetry.instrumentation.CompiledMemory`, as its fields name it), its sites' record's; none for
    a program whose backend stated none."""
    return [memory for site in Path(directory).glob(f"*/{SITE}")
            for memory in json.loads(site.read_text()).get("memory", {}).values()]


COMPILED: contextvars.ContextVar[CompiledTruths | None] = contextvars.ContextVar("truths_ahead_store",
                                                                                default=None)
"""The store open (`compiled_truths`), None outside one."""

SITE_IN_PROGRESS: contextvars.ContextVar[str | None] = contextvars.ContextVar("truths_ahead_site",
                                                                              default=None)
"""The label of the check-1 site whose truths are computed (`judged`), None outside one."""


@contextlib.contextmanager
def compiled_truths(directory: str | Path, *, prepare: bool, stamp: str,
                    target=None) -> Iterator[CompiledTruths]:
    """Within the block, check 1's sites compile their float64 programs ahead into `directory` (`prepare`)
    or compute their truths from those programs (the module's account)."""
    store = CompiledTruths(Path(directory), prepare, stamp, target)
    token = COMPILED.set(store)
    try:
        yield store
    finally:
        COMPILED.reset(token)


@contextlib.contextmanager
def judged(label: str) -> Iterator[None]:
    """Within the block, site `label`'s truths are computed (`running`)."""
    token = SITE_IN_PROGRESS.set(label)
    try:
        yield
    finally:
        SITE_IN_PROGRESS.reset(token)


def running() -> tuple[CompiledTruths, str] | None:
    """The store whose programs compute the truths of the site in progress, and its label: where a run reads
    a store (`compiled_truths(prepare=False)`) within a site (`judged`), else None."""
    store, label = COMPILED.get(), SITE_IN_PROGRESS.get()
    return None if store is None or store.prepare or label is None else (store, label)
