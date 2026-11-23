"""A training step compiled ahead of time for GPUs this CPU host does not have (`dew.training.ahead`).

The step is compiled by XLA's CUDA backend for compile-only GPUs (the `kernel-compile` extra), the
step `fit` would compile on them, and is held to the step the same trainer compiles on this host's
eight CPU devices: the bytes each device reads and writes are the state's and the batch's shares
under one placement, whichever backend compiles it. What it costs and its executable are XLA's own;
a GPU loads and runs the saved executable (TL-*-compile-ahead-881)."""

import contextlib
import dataclasses
import functools
import itertools
import json
import logging
import os
import pickle
import shutil
import subprocess
import sys

import lane_environment
import jax
import jax.numpy as jnp
import numpy as np
import optax
import process_support
import pytest
from lane_environment import without_flags
from generation_support import COMPILES_ON_CPU
from jax.extend import custom_call_targets, xla
from kernel_compile import gate_generations
from model_support import TINY_DECODER, memory_limited_to
from token_support import token_rows

from dew._devices import allocator_limit, device_memory
from dew._records import listed, record
from dew.data import Phase, Steps
from dew.nn.kernels.generation import capability
from dew.nn.sharding import Link
from dew.objectives.lm import LMObjective
from dew.optim import AdamW, OptimConfig
from dew.registry import models
from dew.telemetry.compiles import Compiles
from dew.training import Layout, MeshSpec, Trainer, TrainerAdvanced
from dew.training.targets import AheadTarget, LowerFor

VOCAB, SEQ_LEN, BATCH = 96, 16, 8
TINY_SHARD = 256


def this_gpu() -> tuple[str, int | None, int]:
    """This GPU's own target, whose programs its loader takes (`BuildTarget.held`): its generation, its
    multiprocessors (an H100 PCIe's 114, not the canonical H100's 132) and the memory whose allocator's
    limit is its own (`memory_limited_to`)."""
    from dew.training.ahead import BuildTarget

    own = BuildTarget.of(jax.devices()[:1])
    assert own.memory is not None
    return own.generation, own.multiprocessors, memory_limited_to(own.memory)


def this_gpus(count: int = 1) -> tuple:
    """`count` compile-only GPUs of this GPU's own target (`this_gpu`)."""
    from dew.nn.kernels.generation import compile_only_gpus

    generation, multiprocessors, memory = this_gpu()
    return compile_only_gpus(generation, count, memory, multiprocessors=multiprocessors)


def this_gpus_ahead(path):
    """The target compiling a run's step ahead into `path` for one GPU of this GPU's own target
    (`this_gpu`), with no topology."""
    from dew.training.targets import AheadTarget

    generation, multiprocessors, memory = this_gpu()
    return AheadTarget(generation=generation, devices=1, path=str(path), capacity_gib=memory / 2**30,
                       multiprocessors=multiprocessors)


def gpus(generation: str, count: int = 1, limit: int = 2**36) -> tuple:
    """`count` compile-only GPUs of `generation` whose stated memory gives their allocator `limit` bytes
    (`memory_limited_to`), 64 GiB unless a test asks less: XLA's own scheduling reads the memory a target
    states, which a GPU's of that size leaves to the step's own needs."""
    from dew.nn.kernels.generation import compile_only_gpus

    return compile_only_gpus(generation, count, memory_limited_to(limit))


def fine_tune(mesh, *, ahead=None, width: int | None = None, xla: bool = False, split=1, remat=None,
              inspection=None, tracker=None, dtype: str | None = None):
    fields = {**TINY_DECODER, **({} if width is None else {"emb_features": width}),
              **({"attention_impl": "xla"} if xla else {}), **({} if remat is None else {"remat": remat}),
              **({} if dtype is None else {"dtype": dtype})}
    model = models.build("causal_transformer", vocab_size=VOCAB, **fields, num_kv_heads=2)
    return Trainer(LMObjective(model, SEQ_LEN), optax.adam(1e-3), key=jax.random.key(0), mesh=mesh,
                   advanced=TrainerAdvanced(layout=Layout(min_shard=TINY_SHARD), ahead=ahead,
                                            inspection=inspection), split=split, tracker=tracker)


@functools.cache
def need_ahead(generation: str, *, split: int, remat: str | None, tiled: bool = False) -> int:
    """The bytes the fine-tuned step (`fine_tune`) compiled ahead for a compile-only GPU of `generation`
    needs at `split` microbatches and `remat`, its head tiled where asked: the headroom its compile left of
    a room past all (`gpus`' 64 GiB, which states exactly that allocator limit, `memory_limited_to`), taken
    from that room. A cell states its room from such needs, so the plan it claims of a room holds whatever
    the step's GEMM route or workspace makes them; XLA schedules a tighter room's compile for less, so a cell
    states the least room its claim allows and reads its premise from the build's own probes too. Measured
    once a process: the cells that read one need share it (`shared_setup("sm90_full_tiled_needs")`)."""
    from dew.training.ahead import compile_ahead

    room = 2**36
    trainer = fine_tune(MeshSpec().build(gpus(generation, limit=room)), split=split, remat=remat)
    if tiled:
        trainer.objective.tile_head()
    ahead = compile_ahead(trainer, token_rows(VOCAB, BATCH, SEQ_LEN + 1))
    return room - record(ahead.plan, "plan")["probes"][0]["headroom"]


def need_run(root, generation: str, split: int) -> int:
    """The bytes a configured run's step (`run_kinds.lm`) compiled ahead for a GPU of `generation` needs at
    `split` microbatches: its compile's headroom of a room past all (64 GiB its allocator's), taken from the
    room its build states."""
    from run_kinds import lm

    from dew.config.lm import LMRunConfig
    from dew.training.ahead import AheadOfTime, CompiledAhead

    root.mkdir()
    path = root / "step.dew-aot"
    run = LMRunConfig.cli(["--trainer.name", "need", "--trainer.checkpoint-dir", str(root / "runs"),
                           "--trainer.steps", "2", "--trainer.split", str(split),
                           "--trainer.eval-every",
                           "None", "--trainer.checkpoint-every", "None"], default=lm(root, drawn=False))
    with pytest.raises(CompiledAhead):
        run.lower(AheadTarget(generation=generation, devices=1, path=str(path),
                              capacity_gib=capacity_gib(2**36)))
    ahead = AheadOfTime.load(path)
    return ahead.capacity - record(ahead.plan, "plan")["probes"][0]["headroom"]


def capacity_gib(room: int) -> float:
    """`AheadTarget.capacity_gib` for a GPU whose allocator is limited to `room` bytes."""
    return memory_limited_to(room) / 2**30


def test_a_background_compile_takes_no_slot_a_step_waits_for():
    """One compile slot (`report.CompileSlots`), held: a background compile (a start's program, `options
    .compiling`) and a step's then wait for it, and when it frees the step's takes it first, the background
    one only once no step's waits."""
    import threading
    import time

    from dew.nn.kernels.templates.report import CompileSlots

    slots, order = CompileSlots(1), []

    def compile_(name, background):
        with slots.held(background=background):
            order.append(name)

    with slots.held(background=False):
        waiting = [threading.Thread(target=compile_, args=(name, name == "start"))
                   for name in ("start", "step")]
        for thread in waiting:
            thread.start()
        # The step's compile waiting before the slot frees: its order is the slots', not the threads'.
        deadline = time.monotonic() + 10
        while not slots._waiting and time.monotonic() < deadline:
            time.sleep(0.01)
    for thread in waiting:
        thread.join(timeout=10)
    assert order == ["step", "start"], order


def test_a_program_compiled_ahead_runs_in_place_of_compiling():
    """A program compiled ahead over its arguments' shapes runs where a call of it lowers that program
    (`dew.nn.kernels.options.taking`, by its digest): the call compiles nothing and answers as the program
    compiled for it does; a call of another program compiles its own (Main 11-07: a run compiled ahead
    compiled its placement on the GPU)."""
    from dew.nn.kernels.options import owned_program, program_digest, taking

    devices = jax.devices()[:1]
    placed = jax.sharding.SingleDeviceSharding(devices[0])
    built = owned_program(lambda values: values * 2 + 1, out_shardings=placed)
    shapes = jax.ShapeDtypeStruct((4,), jnp.float32, sharding=placed)
    lowered = built.lower(shapes)
    compiled = lowered.compile()
    loads = []

    def loaded():
        loads.append(True)
        return compiled

    values = jax.device_put(jnp.arange(4, dtype=jnp.float32), placed)
    with taking({program_digest(lowered): loaded}), Compiles(None) as compiles:
        answered = owned_program(lambda values: values * 2 + 1, out_shardings=placed)(values)
    assert [program.name for program in compiles.programs if program.kind == "compiled"] == []
    assert loads == [True]
    np.testing.assert_array_equal(np.asarray(answered), np.arange(4) * 2 + 1)
    with taking({program_digest(lowered): loaded}), Compiles(None) as compiles:
        owned_program(lambda values: values * 3, out_shardings=placed)(values)
    assert [program.kind for program in compiles.programs].count("compiled") == 1


ORPHANED_START = """
import atexit
import os
import threading

import jax
import jax.numpy as jnp

from dew.nn.kernels import options
from dew.nn.kernels.templates import report

# A start's compile, XLA's own, of a program long enough to compile that it is in flight when the run is
# refused: its thread is shown to have entered XLA's compile.
begun, ended = threading.Event(), threading.Event()
compiled = report.attempted


def attempted(lowered, held, **given):
    begun.set()
    try:
        return compiled(lowered, held, **given)
    finally:
        ended.set()


report.attempted = attempted
# Registered before the compile begins, so the exit runs it after its wait for the compile and before JAX's
# own cleanup: 3 where the compile had ended by then, 5 where the exit went on with it running, and 4 where
# it had ended before the refusal, whose exit then says nothing of the wait.
early = threading.Event()
atexit.register(lambda: os._exit(4 if early.is_set() else 3 if ended.is_set() else 5))


def long(x):
    for index in range(20000):
        x = jnp.sin(x) * (index % 7 + 1)
    return x


options.compiling(jax.jit(long), jnp.ones(2))
begun.wait()
# Refused before its start was taken, its compile still running (else the exit's timing says nothing of it,
# and the exit hook says so by its own status).
if ended.wait(0.5):
    early.set()
raise SystemExit(1)
"""
"""A run that begins its start's compile beside its planning and is refused before it takes the start."""


def test_a_filed_executable_states_the_memory_its_compile_analysed():
    """A program compiled ahead (`Executable.of`), filed and read back, states the memory XLA analysed for it
    where it compiled (`compiled_memory`: its arguments', temporaries', outputs' and live peak), so a run's
    need is read before any device loads it (the GPU tier's workers sized by their parts' programs)."""
    from dew.telemetry.instrumentation import compiled_memory
    from dew.training.ahead import Executable, filed

    compiled = jax.jit(lambda values: jnp.cumsum(jnp.sin(values) @ values.T, axis=0)).lower(
        jax.ShapeDtypeStruct((64, 32), jnp.float32)).compile()
    read = pickle.loads(filed(Executable.of(compiled)))
    assert read.memory == compiled_memory(compiled) is not None


def test_a_run_refused_before_its_start_is_taken_exits_once_its_compile_ends():
    """A run whose start's compile is begun beside its planning (`options.compiling`) and which is refused
    before it takes the start, its compile still running, exits once that compile ends, saying it waits:
    JAX's exit cleanup and the C runtime's exit tore XLA down under the running compile, and 12 of 48 such
    exits ended by a segmentation fault (row 6837). The child's own exit hook, run after the wait and before
    JAX's cleanup, states whether the compile had ended: deterministically, where a crash came 1 in 4."""
    import process_support

    environment = {**os.environ, "PYTHONPATH": os.pathsep.join((str(process_support.REPO_ROOT / "src"),
                                                                str(process_support.REPO_ROOT / "tests"))),
                   "JAX_ENABLE_COMPILATION_CACHE": "false"}
    exited = subprocess.run([sys.executable, "-c", ORPHANED_START], env=environment, capture_output=True,
                            text=True, timeout=300, check=False)
    assert exited.returncode == 3, (exited.returncode, exited.stderr[-3000:])


def _begun(case: str) -> tuple:
    """`case`'s program, the input it compiles over, what its compile is begun over, and an input of
    another abstract value: arrays, begun over themselves; a fresh `jax.random.key`, no one committed to
    its devices, begun over its shape (`shapes_of`, as a fresh fit begins its start), the other a key
    committed to a device."""
    if case == "arrays":
        return lambda x: x * 2, jnp.ones(4), jnp.ones(4), jnp.ones(8)
    from dew.nn.kernels.options import shapes_of

    key = jax.random.key(0)
    return (lambda key: jax.random.normal(key, (4,)), key, shapes_of(key),
            jax.device_put(jax.random.key(1), jax.devices()[0]))


@pytest.mark.parametrize("case", ["arrays", "a fresh key's shape"])
def test_a_program_whose_compile_was_begun_passes_once_where_it_compiles_over_the_same(case):
    """A program whose first compile was begun in a thread of its own (`options.compiling`) and which then
    compiles within `attempting` that compile over the same abstract inputs takes the begun lowering: its
    algebra pass runs once (`report.passes`), not again only to match the begun compile (a DiT start's
    1.66 s of 11.5 s of passes, simple-dit-b). A fresh key's call matches its begun shape, as a fresh fit's
    start placed with its key matches the start begun over the key's shape: an uncommitted array lowers as
    an unplaced shape (train 912's F5, train 914's F2). Over another abstract value it lowers, passes and
    compiles its own."""
    from dew.nn.kernels.options import attempting, compiling, owned_program
    from dew.nn.kernels.templates import report

    function, given, begun_over, other = _begun(case)
    program = owned_program(function)
    with report.recording() as begun_passes:
        begun = compiling(program, begun_over)
    with attempting(begun):
        with report.recording() as same:
            program.compiled(given)
        with report.recording() as elsewhere:
            program.compiled(other)
    assert len(begun_passes.passes) == 1 and not same.passes and len(elsewhere.passes) == 1


def test_a_begun_compile_is_not_taken_by_a_call_over_another_layout():
    """A program whose compile was begun over a committed array of its devices' default layout, called within
    `attempting` over a committed array of another layout on the same sharding, lowers and compiles its own
    and computes what the program does: the begun lowering's layout is not the call's, which JAX's
    dispatcher holds a program to (the morning audit of 2026-11-16: the begun executable taken, the call
    refused with a ValueError)."""
    from jax.experimental.layout import Format, Layout
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    from dew.nn.kernels.options import attempting, compiling, owned_program

    sharding = NamedSharding(Mesh(jax.devices()[:1], ("x",)), PartitionSpec())
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    default, other = (jax.device_put(values, Format(Layout(major_to_minor=order), sharding))
                      for order in ((0, 1), (1, 0)))
    program = owned_program(lambda x: x * 2)
    begun = compiling(program, default)
    with attempting(begun):
        np.testing.assert_array_equal(np.asarray(program(other)), values * 2)


def counting_state(values: jax.Array):
    """The state of an optimizer whose factory makes its state's class at each call (`counting`), begun for
    `values`: a program whose results' tree no file of programs states."""
    from optim_support import counting

    return counting(1e-2).init(values)


def test_a_program_whose_trees_no_file_states_compiles_where_programs_are_handed_and_is_kept_by_none(
        tmp_path):
    """A program whose results are an optimizer's state of a class its factory makes (`counting`), a node
    no process names, is one no file of programs holds: a block that keeps programs refuses it, and a block
    handed a file, or programs compiled ahead, compiles and runs it; a training step over that state
    compiled ahead is refused (`Unestablished`, which names it). The keeping block pickled the class's
    tree before stating it, and raised on Python 3.12 (train 964)."""
    from optim_support import counting

    from dew._stated import Unestablished
    from dew.nn.kernels.options import owned_program, taking
    from dew.training.ahead import compile_ahead, keeping, kept

    values = jnp.ones(3)
    path = tmp_path / "programs.dew-aot"
    with keeping() as programs:
        owned_program(counting_state)(values)
    programs.save(path)
    assert len(programs.refused) == 1 and programs.kept == {}
    with kept(path):
        assert int(owned_program(counting_state)(values).steps) == 0
    with taking({}):
        assert int(owned_program(counting_state)(values).steps) == 0
    model = models.build("causal_transformer", vocab_size=VOCAB, **TINY_DECODER)
    trainer = Trainer(LMObjective(model, SEQ_LEN), counting(1e-2), key=jax.random.key(0),
                      mesh=MeshSpec().build(gpus("sm80")))
    with pytest.raises(Unestablished):
        compile_ahead(trainer, token_rows(VOCAB, BATCH, SEQ_LEN + 1))


def test_a_program_called_over_shapes_while_collecting_compiles_there_and_runs_where_handed():
    """A program called within `collecting` over shapes compiles there, keyed by its program's digest, and
    answers with its results' shapes, running nothing; handed to a call of the program over arrays of those
    shapes (`taking`), its executable runs in place of compiling: how a run compiled ahead compiles what
    its steps run on their results, called as the run calls it (row 4752)."""
    from dew.nn.kernels.options import collecting, owned_program, taking

    placed = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    values = jax.device_put(np.arange(4, dtype=np.float32), placed)
    with collecting() as collected, Compiles(None) as compiles:
        shapes = owned_program(lambda values: values * 2 + 1)(jax.ShapeDtypeStruct((4,), jnp.float32,
                                                                                    sharding=placed))
    assert (shapes.shape, shapes.dtype, shapes.sharding) == ((4,), jnp.float32, placed)
    assert len(collected) == [program.kind for program in compiles.programs].count("compiled") == 1
    handed = {digest: lambda executable=executable: executable for digest, executable in collected.items()}
    with taking(handed), Compiles(None) as compiles:
        answered = owned_program(lambda values: values * 2 + 1)(values)
    assert [program.kind for program in compiles.programs].count("compiled") == 0
    np.testing.assert_array_equal(np.asarray(answered), np.arange(4) * 2 + 1)


def test_a_program_compiled_over_a_zero_tangents_shape_runs_for_its_host_value():
    """A program compiled within `collecting` over a float0 shape (a zero tangent's, which a pullback reads
    for an integer statistic) runs for the host value JAX hands such a tangent back as, tracing nothing
    (`executable_only`): no device holds a float0, so a call knows it by its shape alone."""
    from dew.nn.kernels.options import collecting, owned_program
    from dew.telemetry.compiles import executable_only

    program = owned_program(lambda values, zero: values + 1)
    placed = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    with collecting():
        program(jax.ShapeDtypeStruct((4,), jnp.float32, sharding=placed),
                jax.ShapeDtypeStruct((3,), jax.dtypes.float0))
    with executable_only():
        answered = program(jax.device_put(np.zeros(4, np.float32), placed), np.zeros((3,), jax.dtypes.float0))
    np.testing.assert_array_equal(np.asarray(answered), np.ones(4))


def test_a_program_holds_at_most_its_bound_kept_and_compiled_over_shapes_together():
    """A program called over more distinct shapes than it keeps (its `CACHED`, three here), some of them
    compiled over shapes first (`collecting`), holds no more executables than its bound across its kept
    and its shaped stores together: one eviction policy (`_bounded`)."""
    from dew.nn.kernels.options import collecting, owned_program

    program = owned_program(lambda values: values + 1)
    program.CACHED = 3
    with collecting():
        for size in (1, 2):
            program(jax.ShapeDtypeStruct((size,), jnp.float32))
    for size in range(3, 8):
        program(jnp.zeros((size,), jnp.float32))
    assert len(program._executables) + len(program._shaped) <= 3
    # The latest kept, which the bound never lets go: called again, it compiles nothing.
    with Compiles(None) as compiles:
        program(jnp.zeros((7,), jnp.float32))
    assert compiles.count == 0


def test_each_collection_in_a_process_holds_every_program_it_called():
    """Two collections in one process over the same shapes (two equivalent runs compiled ahead) each hold
    every program their calls reached, the second's taken from the first's compile, not left out: each
    file its loader reads (`AheadOfTime.starts`) is whole."""
    from dew.nn.kernels.options import collecting, owned_program

    program = owned_program(lambda values: values * 3 - 1)
    placed = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    shape = jax.ShapeDtypeStruct((4,), jnp.float32, sharding=placed)
    held = []
    for _ in range(2):
        with collecting() as collected:
            program(shape)
        held.append(collected)
    first, second = held
    assert len(first) == 1 and second.keys() == first.keys()


def test_a_program_compiled_over_shapes_while_collecting_runs_for_arrays_alike_tracing_nothing():
    """A program called within `collecting` over shapes runs for a later call over arrays alike in shape,
    dtype and placement with nothing traced, lowered or compiled (`executable_only` allows none): how a run
    whose startup compiles its host step's programs over its state's shapes steps on them. Arrays placed
    otherwise than the shapes stated miss it: that call compiles its own program, placed where they are."""
    from dew.nn.kernels.options import collecting, owned_program
    from dew.telemetry.compiles import executable_only

    devices = jax.devices()
    placed = jax.sharding.SingleDeviceSharding(devices[0])
    program = owned_program(lambda values: values * 2 + 1)
    with collecting():
        program(jax.ShapeDtypeStruct((4,), jnp.float32, sharding=placed))
    with executable_only():
        answered = program(jax.device_put(np.arange(4, dtype=np.float32), placed))
    np.testing.assert_array_equal(np.asarray(answered), np.arange(4) * 2 + 1)
    elsewhere = jax.sharding.SingleDeviceSharding(devices[-1])
    with Compiles(None) as compiles:
        moved = program(jax.device_put(np.arange(4, dtype=np.float32), elsewhere))
    assert [program.kind for program in compiles.programs].count("compiled") == 1
    assert moved.sharding == elsewhere
    np.testing.assert_array_equal(np.asarray(moved), np.arange(4) * 2 + 1)


@pytest.mark.parametrize("asked", ["midway", "at its end", "scored to"])
def test_a_runs_evaluations_called_over_its_first_states_shapes_are_the_programs_its_fit_runs(
        asked, tmp_path):
    """A run's evaluations' programs, called over its first state's shapes where a run compiled ahead
    collects them (`Evaluator.warmed_ahead`, `dew.training.evaluation.warmed_ahead`), are the ones its fit
    compiles: a fit evaluating midway, at its end alone, or writing what it scores (`scored_to`, which
    reports its artifacts beside its loss) compiles them, and a fit handed the collection (`taking`) takes
    every program in it and compiles none of what the first fit compiled but its step's own, each fit in a
    process whose programs hold no executable before it (`release`), and scores and ends on what the
    first fit scored and ended on, bit for bit. Which executables it ran is counted where they are handed,
    as XLA's persistent cache would hand the second fit what the first compiled, which compiles nothing
    either (row 4752)."""
    from recording import RecordingTracker

    from dew.data import Memory
    from dew.distributed.placement import placed_shapes
    from dew.nn.kernels.options import collecting, release, taking
    from dew.training.ahead import fit_plan

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)

    # A split of two batches, whose loss's statistics the pass sums.
    validation = {"val": Memory(token_rows(VOCAB, 2 * BATCH, SEQ_LEN + 1, seed=1)).batch(BATCH)}
    fit = {"steps": 4, "log_every": 2, "validation": validation,
           **{"midway": {"eval_every": 2}, "at its end": {},
              "scored to": {"eval_every": 2, "scored_to": str(tmp_path / "scores")}}[asked]}
    def compiled(handed) -> tuple[set[str], list, list]:
        """The programs the fit compiled, the validation scores it logged and the weights it ended on."""
        release()
        jax.clear_caches()
        written = tmp_path / "scores"
        if written.exists():
            shutil.rmtree(written)
        tracker = RecordingTracker()
        run = fine_tune(MeshSpec().build(jax.devices()[:1]), xla=True, tracker=tracker)
        with taking(handed), Compiles(None) as compiles:
            state = run.fit(batch, **fit)
        # A fit asked to write what it scores writes it, handed its programs or not.
        assert (asked == "scored to") == any(written.rglob("*"))
        scores = [(step, {name: value for name, value in scalars.items() if name.startswith("val/")})
                  for step, scalars in tracker.scalars if any(name.startswith("val/") for name in scalars)]
        return ({program.name for program in compiles.programs if program.kind == "compiled"}, scores,
                [np.asarray(leaf) for leaf in jax.tree.leaves(state.variables)])

    alone, scored, ended = compiled({})
    ahead = fine_tune(MeshSpec().build(jax.devices()[:1]), xla=True)
    plan = fit_plan(ahead, batch, fit)
    start = ahead._started()
    with collecting() as collected:
        ahead._evaluator().warmed_ahead(plan, placed_shapes(start.abstract, start.shardings), start.shardings,
                                        ahead._compiler.links(ahead.device_mesh))
    served: set[str] = set()

    def serving(digest, executable):
        served.add(digest)
        return executable

    handed = {digest: functools.partial(serving, digest, executable)
              for digest, executable in collected.items()}
    taken, rescored, reended = compiled(handed)
    # Every program the run collected ahead is one its fit took, and none it took is one it compiled: what
    # the first fit compiled beside its step's own programs, the second compiled none of; its scores and
    # weights are the first fit's.
    assert served == set(collected) and served
    assert len(alone - taken) >= len(collected) and taken < alone
    assert scored and rescored == scored
    assert all(np.array_equal(left, right) for left, right in zip(reended, ended, strict=True))


def test_two_calls_whose_texts_agree_as_jax_prunes_an_unread_argument_each_take_their_own_executable():
    """Two calls of one program whose lowered texts agree only as JAX prunes an argument neither reads,
    held in other trees (a step's commit with its sample's summaries and without), compile apart while
    collecting and each runs its own executable where handed (`program_digest` keys the trees too; row
    4752's review: the second overwrote the first, whose call it then refused)."""
    from dew.nn.kernels.options import collecting, owned_program, taking

    placed = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    shape = jax.ShapeDtypeStruct((4,), jnp.float32, sharding=placed)

    def counted(values, unread):
        return values + 1

    with collecting() as collected:
        owned_program(counted)(shape, None)
        owned_program(counted)(shape, {"summaries": shape})
    assert len(collected) == 2
    values = jax.device_put(np.arange(4, dtype=np.float32), placed)
    handed = {digest: lambda executable=executable: executable for digest, executable in collected.items()}
    with taking(handed), Compiles(None) as compiles:
        bare = owned_program(counted)(values, None)
        held = owned_program(counted)(values, {"summaries": values})
    assert [program.kind for program in compiles.programs].count("compiled") == 0
    np.testing.assert_array_equal(np.asarray(bare), np.asarray(held))


def test_a_run_resumed_mid_window_compiles_its_step_ahead_building_no_window_of_its_own(tmp_path):
    """A run whose checkpoint holds a part-filled accumulation window compiles its step ahead: the state it
    resumes holds its window, so it builds none ahead, and its count of microbatches, a shape there, is
    never read (row 4752's review: it was read as a number and compiling ahead raised TypeError)."""
    from dew.checkpoints import Checkpoints
    from dew.training.ahead import compile_ahead

    def accumulating(advanced=None, **mesh):
        model = models.build("causal_transformer", vocab_size=VOCAB, **TINY_DECODER, num_kv_heads=2)
        return Trainer(LMObjective(model, SEQ_LEN), optax.adam(1e-3), key=jax.random.key(0), **mesh,
                       advanced=dataclasses.replace(advanced or TrainerAdvanced(),
                                                    layout=Layout(min_shard=TINY_SHARD)),
                       accumulation=2,
                       checkpoints=Checkpoints(str(tmp_path / "run")))

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    run = accumulating()
    run.fit(batch, steps=1, checkpoint_every=1)
    run.checkpoints.wait()
    assert run.checkpoints.accumulation_template(1) is not None
    ahead = compile_ahead(accumulating(mesh=MeshSpec().build(gpus("sm90"))), batch)
    assert ahead.starts


def test_a_fresh_runs_start_compiles_once_beside_its_steps_planning():
    """A fresh run's start (`start_compiling`) compiles in a thread of its own while its step is planned,
    and the start compiled ahead with the step is that compile: its program compiled once, its executable
    among the run's starts, where it compiled after the step, in turn (a Qwen3.5-4B's 97 s after its
    step's 110 s for an H100)."""
    from dew.training.ahead import compile_ahead

    trainer = fine_tune(MeshSpec().build(gpus("sm90")))
    with Compiles("joined") as compiles:
        ahead = compile_ahead(trainer, token_rows(VOCAB, BATCH, SEQ_LEN + 1))
    assert compiles.count == 1 and ahead.starts


def test_a_resume_compiled_ahead_steps_at_the_split_its_checkpoint_was_saved_at(tmp_path):
    """A run saved at an update's end after updates of two microbatches, its window empty and so not kept
    (`dew.state.without_empty_window`), resumes at that split, which a run stating none takes from its
    state (`dew.training.compiler.resumed_stepping`): its step compiled ahead plans split 2, the program
    the resume compiles, where it planned a split of its own (docs/design/training.md: a resumed run's at
    its checkpoint's rung)."""
    from dew.checkpoints import Checkpoints
    from dew.training.ahead import compile_ahead

    def run(advanced=None, **options):
        model = models.build("causal_transformer", vocab_size=VOCAB, **TINY_DECODER, num_kv_heads=2)
        return Trainer(LMObjective(model, SEQ_LEN), optax.adam(1e-3), key=jax.random.key(0),
                       advanced=dataclasses.replace(advanced or TrainerAdvanced(),
                                                    layout=Layout(min_shard=TINY_SHARD)),
                       checkpoints=Checkpoints(str(tmp_path / "run")),
                       **options)

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    saved = run(split=2, mesh=MeshSpec().build(jax.devices()[:1]))
    saved.fit(batch, steps=1, checkpoint_every=1)
    saved.checkpoints.wait()
    assert saved.checkpoints.window_size(1) == 2 and saved.checkpoints.accumulation_template(1) is None
    ahead = compile_ahead(run(mesh=MeshSpec().build(gpus("sm90"))), batch)
    assert record(ahead.plan, "plan")["split"] == 2


def test_a_bfloat16_decoders_step_compiles_ahead_for_an_a100():
    """A bfloat16 decoder's training step compiles ahead for an A100 (`compile_only_gpus("sm80")`): its
    compile point keys each body's recognition by its equations' parameters, the custom VJP's rules its
    recognized bodies hold among them, which JAX holds unhashable (row 4783's head: "unhashable type:
    'Unhashable'")."""
    from dew.training.ahead import compile_ahead

    model = models.build("causal_transformer", vocab_size=VOCAB, **TINY_DECODER, num_kv_heads=2,
                         dtype="bfloat16")
    trainer = Trainer(LMObjective(model, SEQ_LEN), optax.adam(1e-3), key=jax.random.key(0),
                      mesh=MeshSpec().build(gpus("sm80")),
                      advanced=TrainerAdvanced(layout=Layout(min_shard=TINY_SHARD)))
    assert compile_ahead(trainer, token_rows(VOCAB, BATCH, SEQ_LEN + 1)).starts


def test_a_step_compiled_ahead_over_phases_of_other_batches_takes_a_split_that_cuts_each():
    """A run over phases of 12, then 8 rows a step compiles its step ahead in a split that cuts every
    phase's batch (`Trainer.reading`'s `Loader.batches`), as its fit plans it: a fixed split of 2 compiles,
    and one of 3, which cuts the first phase's 12 rows and not the second's 8, is refused, where it was
    compiled ahead for a run that then plans another program (row 4812's review)."""
    from dew.data import Memory, phases
    from dew.data.compile import compile
    from dew.data.pipeline import Pipeline
    from dew.mesh import LayoutRefused
    from dew.training.ahead import compile_ahead, first_step_batch

    corpus = Pipeline(Memory(token_rows(VOCAB, 64, SEQ_LEN + 1))).repeat()
    loader = compile(phases(Phase(corpus.batch(12), Steps(2)), Phase(corpus.batch(8))), read="memory")
    for split, refused in ((2, False), (3, True)):
        trainer = fine_tune(MeshSpec().build(gpus("sm90")), split=split)
        with pytest.raises(LayoutRefused) if refused else contextlib.nullcontext():
            compile_ahead(trainer, first_step_batch(trainer, loader, 4))


def test_compiling_ahead_plans_once_and_saves_the_selected_program(tmp_path, monkeypatch):
    """A compile-only GPU with stated memory takes the run's own fit search, whose one fitting
    program is the saved executable, not another compile outside the report. The report records
    that compile's time and memory, and planning creates no device state or runner to execute."""
    from dew.training.ahead import AheadOfTime, compile_ahead

    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "rungs"))
    trainer = fine_tune(MeshSpec().build(gpus("sm90", limit=2**30)))
    with Compiles("step") as compiles:
        ahead = compile_ahead(trainer, token_rows(VOCAB, BATCH, SEQ_LEN + 1))
    assert compiles.count == 1
    # Its fresh state's build and placement compiled beside the step, for a run handed it to compile none of
    # it on its GPU (`AheadOfTime.starts`).
    assert ahead.starts
    plan = record(ahead.plan, "plan")
    (probe,) = listed(plan["probes"], "probes", read=record, what="probes")
    assert probe["fits"] is True and probe["temp"] == ahead.temp_bytes
    assert probe["argument"] == ahead.argument_bytes
    # The plan was made against the allocator's limit the GPU's stated memory gives (`allocator_limit`); the
    # build states what its programs need of it (`programs_need`).
    assert ahead.capacity == 2**30 and 0 < ahead.built.memory <= ahead.capacity
    stated = device_memory(trainer.device_mesh.devices.flat[0], ahead.argument_bytes)
    assert stated["pool_bytes"] == ahead.capacity
    assert stated["largest_free_block_bytes"] == ahead.capacity - ahead.argument_bytes
    seconds = probe["compile_seconds"]
    assert isinstance(seconds, float) and ahead.planned_seconds is not None
    assert ahead.planned_seconds >= seconds > 0
    # Its seconds by part, as the perf gate reports them (row 5339): the step was traced, passed through the
    # algebra, lowered and compiled, and its compile is at least its probe's; the fresh state's start,
    # compiled beside it, is a part of its own, none of the step's (row 6740).
    parts = ahead.planned_parts
    assert all(parts[part] > 0 for part in ("traced", "algebra", "lowered", "start"))
    assert parts["compiled"] >= seconds
    destination = tmp_path / "planned.dew-aot"
    ahead.save(destination)
    saved = AheadOfTime.load(destination)
    assert saved.plan == ahead.plan and saved.program == ahead.program


def at_its_plan(plan, probes) -> None:
    """What a user gets of a plan: the fewest microbatches that fit at its rung, which fits. Its rung's
    probes at that split fit and at fewer were refused, whatever order the search compiled them in."""
    rung = ("remat", "head_tile", "head_fused", "xla_defaults", "narrow")
    own = [probe for probe in probes if all(probe[name] == plan[name] for name in rung)]
    assert any(probe["split"] == plan["split"] and probe["fits"] is True for probe in own), own
    assert not any(probe["split"] < plan["split"] and probe["fits"] is True for probe in own), own


ROOMY = 2**36
"""The bytes a compile-only GPU states (`gpus`' default) where a step's need is measured: room for any of
`fine_tune`'s compiles, so its need does not depend on it (`memory.step_fits`)."""


def _need(split: int, *, up: bool = False) -> int:
    """What `fine_tune`'s step needs beside what a device holds, as its plan's search reads it (`limit -
    headroom`, `memory.step_fits`): compiled ahead at `split` microbatches on its own rung, or with `up` the
    rung above it, its search's first climb (`memory.recompute_more`: the head's tile, before any remat),
    under `ROOMY`."""
    from dew.training.ahead import compile_ahead
    from dew.training.memory import recompute_more

    trainer = fine_tune(MeshSpec().build(gpus("sm90")), split=split)
    if up:
        assert recompute_more(trainer.objective)
    plan = record(compile_ahead(trainer, token_rows(VOCAB, BATCH, SEQ_LEN + 1)).plan, "plan")
    return ROOMY - listed(plan["probes"], "probes", read=record, what="probes")[0]["headroom"]


@pytest.fixture(scope="module")
def needing_microbatches(tmp_path_factory) -> int:
    """Bytes a compile-only GPU states for `fine_tune`'s step to need several microbatches at its own rung
    with a rung above it fitting at fewer, read from the step's own compiles (`_need`): the larger of its
    own rung's need at two microbatches and the rung above's whole, where its own rung's need whole is more
    (the scenario's condition, which the step's figures state or the cell says they do not). The needs are
    measured under `ROOMY`, where XLA's scheduler, which reads the memory a target states, may need more
    than under the limit; the admitted search under the limit is the judge (`plan["split"] > 1`, a frontier),
    and where it goes red the three needs are measured again under the limit, never a margin added."""
    with pytest.MonkeyPatch.context() as patch:
        lane_environment.per_test_settings(patch, tmp_path_factory.mktemp)
        patch.setenv("DEW_RUNG_RECORDS", str(tmp_path_factory.mktemp("needs")))
        whole, halves, above = _need(1), _need(2), _need(1, up=True)
    limit = max(halves, above)
    assert whole > limit, (whole, halves, above)
    return limit


@pytest.fixture(scope="module")
def admitted(tmp_path_factory, needing_microbatches):
    """`fine_tune`'s step compiled ahead on GPUs of `needing_microbatches`, its plan kept beside the
    compilation cache as every search keeps its own (`DEW_RUNG_RECORDS`): the records' directory, the
    trainer and its `AheadOfTime`, admitted once for the module, and the limit."""
    from dew.training.ahead import compile_ahead

    records = tmp_path_factory.mktemp("rungs")
    with pytest.MonkeyPatch.context() as patch:
        # The module's own data and Hub caches, as each test's (`own_settings`), its rung records `records`.
        lane_environment.per_test_settings(patch, tmp_path_factory.mktemp)
        patch.setenv("DEW_RUNG_RECORDS", str(records))
        trainer = fine_tune(MeshSpec().build(gpus("sm90", limit=needing_microbatches)), split="auto")
        ahead = compile_ahead(trainer, token_rows(VOCAB, BATCH, SEQ_LEN + 1))
    return records, trainer, ahead, needing_microbatches


def cache_key_of(lowered: jax.stages.Lowered, device, overrides: dict | None = None) -> str:
    """The key JAX's persistent cache keeps `lowered`'s compile for one `device` under, with the compile
    options `overrides` given (`jax._src.cache_key.get`)."""
    from jax._src import cache_key, compiler

    options = compiler.get_compile_options(num_replicas=1, num_partitions=1, device_assignment=[[device]],
                                           backend=device.client, env_options_overrides=overrides)
    return cache_key.get(lowered.compiler_ir("stablehlo"), np.array([device]), options, device.client)


def test_a_program_compiled_over_a_hosts_serialized_topology_keys_the_cache_as_over_its_own_gpus(tmp_path):
    """A host's GPUs, serialized as their topology (`get_attached_topology(...).serialize()`, here of
    compile-only H100s at a stated 40 GiB) and read back elsewhere (`host_gpus`), are the GPUs XLA compiles
    for: a program compiled over them keys the persistent cache with the host's own platform version and
    topology fingerprint, as over the devices serialized, where a canonical target of the generation
    (`compile_only_gpus`, its memory unstated) keys another entry."""
    from jax.experimental import topologies

    from dew.nn.kernels.generation import compile_only_gpus
    from dew.training.ahead import host_gpus

    host = compile_only_gpus("sm90", 2, 40 * 2**30)
    serialized = tmp_path / "topology.bin"
    serialized.write_bytes(topologies.TopologyDescription(list(host)).serialize())

    def key(devices) -> str:
        lowered = jax.jit(lambda x: x * 2 + 1).trace(jax.ShapeDtypeStruct(
            (8, 128), jnp.float32, sharding=jax.sharding.SingleDeviceSharding(devices[0]))).lower()
        return cache_key_of(lowered, devices[0])

    read = host_gpus("sm90", serialized, 2)
    assert [device.device_kind for device in read] == [device.device_kind for device in host]
    assert key(read) == key(host)
    assert key(read) != key(compile_only_gpus("sm90", 2))
    # The topology states its GPUs' memory, which the devices read back carry as the ones serialized do.
    assert [device_memory(device)["bytes_limit"] for device in read] == [allocator_limit(40 * 2**30)] * 2


def test_a_step_prepared_for_a_hosts_gpus_keys_the_cache_as_the_hosts_run_compiles_it():
    """A program compiled for a host's GPUs within a prepare (`report.for_devices`), each tuned choice read
    from the autotune cache and every one required, keys the persistent cache as the host's run's compile
    of it does, under that run's own options (`report.targeted` on a GPU that runs: a CUB sort's choice
    kept off its key), so the run reads the step the prepare compiled (row 6046)."""
    from dew.nn.kernels.generation import compile_only_gpus
    from dew.nn.kernels.templates.report import CUB_NO_FALLBACK, for_devices, targeted

    (device,) = compile_only_gpus("sm90", 1, 40 * 2**30)
    # A bf16 product, which keeps XLA's Triton GEMM at any precision: the cache's key alone.
    lowered = jax.jit(lambda x: x @ x).trace(jax.ShapeDtypeStruct(
        (64, 64), jnp.bfloat16, sharding=jax.sharding.SingleDeviceSharding(device))).lower()
    with for_devices():
        prepared = targeted(lowered, None)
    assert cache_key_of(lowered, device, prepared) == cache_key_of(
        lowered, device, {"xla_gpu_deviceless_cub_mode": CUB_NO_FALLBACK})


def test_a_prepares_compiles_on_dews_compile_threads_take_its_target(monkeypatch):
    """A program compiled on one of Dew's compile threads within a prepare (`report.for_devices`), a run's
    start begun beside its planning (`options.compiling`) and a search's programs compiled together
    (`report.attempts`), is compiled as one in the prepare's own thread, for a device to run, its tuned
    choices required (`report.targeted`): on a host no H100's run left them beside, not preparable
    (`report.Untuned`), where one read offline compiles untuned."""
    from dew.nn.kernels import options
    from dew.nn.kernels.generation import compile_only_gpus
    from dew.nn.kernels.templates import report

    monkeypatch.setenv("XLA_FLAGS", without_flags(os.environ.get("XLA_FLAGS", ""),
                                                  "--xla_gpu_autotune_level"))
    (device,) = compile_only_gpus("sm90", 1, 40 * 2**30)
    program = jax.jit(lambda x, w: jnp.tanh(x @ w) @ w)
    placed = jax.sharding.SingleDeviceSharding(device)
    shapes = [jax.ShapeDtypeStruct((512, 512), jnp.bfloat16, sharding=placed)] * 2
    lowered = program.trace(*shapes).lower()
    assert report.attempts([(lowered, None), (lowered, None)])
    with report.for_devices():
        begun = options.compiling(program, *shapes)
        together = report.attempts([(lowered, None), (lowered, None)])
    # Each program's refusal its own attempt's, raised where it is settled, in its caller's turn.
    for attempt in (*together, begun.attempt.result()):
        assert isinstance(attempt.error, report.Untuned)
        with pytest.raises(report.Untuned):
            report.settled(lowered, None, lambda: lowered, attempt)


@pytest.mark.parametrize("host, memory", [("a100-sxm4-40gb", 42_406_903_808), ("gh200", 101_468_602_368)])
def test_a_hosts_serialized_topology_states_its_gpus_memory_and_their_allocators_limit(monkeypatch, host,
                                                                                         memory):
    """A topology a GPU host serialized of its own GPUs (`get_attached_topology("cuda").serialize()`, an
    A100 SXM4 40 GB's and a GH200's, kept as the hosts wrote them, in base64) states each GPU's memory, its
    total as XLA reads it for its target (`topology_memory`); at Dew's share of a GPU the allocator's limit
    derived from it (`allocator_limit`) is at most XLA's 2 MiB rounding under the one the GPU states: the
    GH200's run stated 96,395,591,680 (M-IG-6), a compile ahead that states its memory loads there."""
    from pathlib import Path

    from dew.training.ahead import topology_memory

    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
    import base64

    # Kept as base64 text: the bytes the host's `serialize()` wrote.
    written = Path(__file__).parent / "fixtures" / "topologies" / f"{host}.b64"
    serialized = base64.b64decode(written.read_text())
    assert topology_memory(serialized) == memory
    if host == "gh200":
        assert 0 <= 96_395_591_680 - allocator_limit(memory) < 2 * 2**20
    assert topology_memory(b"") is None


@pytest.mark.shared_setup("admitted")
def test_compiling_ahead_on_gpus_its_plan_needs_microbatches_for_admits_it(admitted):
    """A step compiled ahead on GPUs whose memory its plan needs several microbatches for searches for its
    plan, the fewest microbatches that fit, which fits (`at_its_plan`), and saves it, the run left at its
    plan; the plan states the limit it was made against, its GPUs' stated memory at the share their loading
    run takes (`pool_record`). Compiled on compile-only GPUs outside a prepare into the cache, it is XLA's
    untuned program, which its record says (`AheadOfTime.tuned`, `report.autotuned`)."""
    from dew._devices import pool_share
    from dew.training.memory import recompute_record

    _, trainer, ahead, limit = admitted
    plan = record(ahead.plan, "plan")
    probes = listed(plan["probes"], "probes", read=record, what="probes")
    assert plan["split"] > 1
    at_its_plan(plan, probes)
    assert trainer.plan == ahead.plan and recompute_record(trainer.objective) == plan["remat"]
    pool = record(plan["pool"], "pool")
    assert pool["source"] == "compile_only" and pool["share"] == pool_share()
    assert pool["limit"] == limit
    assert not ahead.tuned and plan["autotuned"] is False


@pytest.mark.shared_setup("admitted")
def test_compiling_ahead_with_its_frontier_lists_each_rung_above_its_plan_at_fewer_microbatches(
        admitted, monkeypatch):
    """With its frontier asked, a step compiled ahead on GPUs whose memory its plan needs several
    microbatches for reports, past the plan it takes and saves, each rung above it that fits at fewer
    microbatches than every plan before it, at the fewest that fit there (rows 2115, 3108, 3110): the plans
    a run's speed is chosen among on its GPUs (tools/benchmark_step.py times each), the run naming the one it
    takes. A rung at as many microbatches as one below it is none of them. The run is left at its plan. Its
    search starts at the plan admitted before it (`admitted`), kept beside the compilation cache, as a run
    after another starts at its recorded plan: that plan's compile alone, the frontier read above it."""
    from dew.training.ahead import compile_ahead
    from dew.training.memory import recompute_record

    records, _, earlier, limit = admitted
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(records))
    trainer = fine_tune(MeshSpec().build(gpus("sm90", limit=limit)), split="auto")
    ahead = compile_ahead(trainer, token_rows(VOCAB, BATCH, SEQ_LEN + 1), frontier=True)
    plan = record(ahead.plan, "plan")
    probes = listed(plan["probes"], "probes", read=record, what="probes")
    # A sibling that offloads what the ladder's top recomputes sits at that top's split (row 1932).
    from dew.nn.backbones.decoder_block import REMAT_POLICIES, remat_record

    offloaded = remat_record(REMAT_POLICIES["minimal_offloaded"])
    frontier = [entry for entry in listed(plan["frontier"], "frontier", read=record, what="frontier")
                if entry["remat"] != offloaded]
    rung = ("split", "remat", "head_tile", "head_fused", "xla_defaults", "narrow")
    admitted_plan = record(earlier.plan, "admitted plan")
    assert {name: plan[name] for name in rung} == {name: admitted_plan[name] for name in rung}
    assert len(probes) == 1
    splits = [plan["split"], *(entry["split"] for entry in frontier)]
    assert frontier and all(entry["fits"] is True for entry in frontier)
    assert all(isinstance(more, int) and isinstance(fewer, int) and fewer < more
               for more, fewer in itertools.pairwise(splits))
    rungs = [json.dumps({name: entry[name] for name in ("remat", "head_tile", "head_fused")}, sort_keys=True)
             for entry in (plan, *frontier)]
    assert len(set(rungs)) == len(rungs)
    assert trainer.plan == ahead.plan and recompute_record(trainer.objective) == plan["remat"]


@pytest.mark.shared_setup("sm90_full_tiled_needs")
@pytest.mark.parametrize("several", [False, True], ids=["one microbatch", "several"])
def test_a_frontier_at_the_ladders_top_lists_its_offloaded_sibling_at_its_last_fit(tmp_path, monkeypatch,
                                                                                    several):
    """A step whose plan is the remat ladder's top ('full') lists, past it, the sibling that keeps what
    'minimal' saves in pinned host memory in place of recomputing it ('minimal_offloaded'): as many device
    bytes as 'full', so no fit rung, but another speed, which the GPUs' timing chooses (row 1932). It is
    compiled at the last configuration at 'full' that fit, its head policy put back, whatever head policies
    the frontier then tried at fewer microbatches; and the run, which times nothing, is left at its plan.
    With room for one microbatch, and with room for what two microbatches and the sibling at two need at
    'full', their head tiled, and less than one needs (each measured, `need_ahead`), where the plan's
    compile at one did not fit. Tiled, as the plan's fit there takes it: on cuBLAS's GEMMs the untiled
    head's workspace makes the offloaded sibling at two need more than 'full' at one, so only the tiled
    head has a room where 'full' takes two microbatches and the sibling fits."""
    from dew.nn.backbones.decoder_block import REMAT_POLICIES, remat_record
    from dew.objectives.base import head_fused_of, head_tile_of
    from dew.training.ahead import compile_ahead
    from dew.training.memory import recompute_record

    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "rungs"))
    capacity = 2**30
    if several:
        one, two = (need_ahead("sm90", split=split, remat="full", tiled=True) for split in (1, 2))
        sibling = need_ahead("sm90", split=2, remat="minimal_offloaded", tiled=True)
        capacity = max(two, sibling)
        assert capacity < one, (one, two, sibling)
    trainer = fine_tune(MeshSpec().build(gpus("sm90", limit=capacity)), split="auto", remat="full")
    ahead = compile_ahead(trainer, token_rows(VOCAB, BATCH, SEQ_LEN + 1), frontier=True)
    plan = record(ahead.plan, "plan")
    frontier = listed(plan["frontier"], "frontier", read=record, what="frontier")
    full, offloaded = (remat_record(REMAT_POLICIES[name]) for name in ("full", "minimal_offloaded"))
    if several:
        whole = [probe for probe in plan["probes"]
                 if probe["split"] == 1 and probe["remat"] == full and probe["head_tile"] is not None]
        assert whole and not any(probe["fits"] for probe in whole), plan["probes"]
        assert plan["head_tile"] is not None, plan
    *rungs, sibling = frontier
    last = [entry for entry in (plan, *rungs) if entry["remat"] == full][-1]
    assert plan["remat"] == full
    assert sibling["remat"] == offloaded and sibling["fits"] is True
    assert (sibling["split"], sibling["head_tile"], sibling["head_fused"]) == (
        last["split"], last["head_tile"], last["head_fused"])
    assert recompute_record(trainer.objective) == plan["remat"]
    tile = head_tile_of(trainer.objective)
    assert ([*tile] if tile else None, head_fused_of(trainer.objective)) == (plan["head_tile"],
                                                                             plan["head_fused"])


@pytest.mark.shared_setup("sm90_full_tiled_needs")
def test_a_fused_tiled_heads_plan_compiled_ahead_with_its_frontier_restores_the_admitted_plan(tmp_path,
                                                                                              monkeypatch):
    """A fused tiled head's plan compiled ahead with its frontier (`FitSearch.above`, which moves the
    objective up the ladder and to its offloaded sibling) leaves the objective at the admitted plan: its
    remat, its head's tile and whether it is fused, the trainer's plan the one compiled ahead: in room for
    what two microbatches need with the head tiled at 'full' and less than one needs (each measured,
    `need_ahead`), where the plan's compile at one did not fit."""
    from dew.objectives.base import head_fused_of, head_tile_of
    from dew.training.ahead import compile_ahead
    from dew.training.memory import recompute_record

    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "rungs"))
    one, room = (need_ahead("sm90", split=split, remat="full", tiled=True) for split in (1, 2))
    assert room < one, (one, room)
    trainer = fine_tune(MeshSpec().build(gpus("sm90", limit=room)), split="auto", remat="full")
    trainer.objective.tile_head()
    ahead = compile_ahead(trainer, token_rows(VOCAB, BATCH, SEQ_LEN + 1), frontier=True)
    plan = record(ahead.plan, "plan")
    whole = [probe for probe in plan["probes"]
             if probe["split"] == 1 and probe["head_tile"] is not None]
    assert whole and not any(probe["fits"] for probe in whole), plan["probes"]
    assert plan["remat"] == "full"
    assert plan["head_fused"] and plan["head_tile"] is not None
    assert recompute_record(trainer.objective) == plan["remat"]
    tile = head_tile_of(trainer.objective)
    assert tile is not None and list(tile) == plan["head_tile"]
    assert head_fused_of(trainer.objective) == plan["head_fused"]
    assert trainer.plan == ahead.plan


def test_a_program_compiled_for_gpus_none_attached_skips_the_autotuning_nothing_can_time(monkeypatch):
    """Every program compiled for compile-only GPUs takes autotuning off at Dew's compile point
    (`report.targeted`), a step and the programs a run compiles ahead beside it alike, its programs' bytes
    the same: XLA's config assigner gives each matmul its default config where no device is attached, while
    its autotuner would still compile every candidate it cannot time, and fails a GEMM fusion whose
    candidates are cuBLASLt's on a GPU host. A run's own XLA_FLAGS setting, and devices that run, keep the
    options as given; the fit search compiles each rung under the step's own options, the target's left to
    the compile point."""
    from dew.nn.kernels.generation import compile_only_gpus
    from dew.nn.kernels.options import compiler_options
    from dew.nn.kernels.templates.report import AUTOTUNE_CACHE_READ, CUB_NO_FALLBACK, for_devices, targeted
    from dew.training.memory import LIMIT_EXIT
    from dew.training.planning import FitSearch

    monkeypatch.setenv("XLA_FLAGS", without_flags(os.environ.get("XLA_FLAGS", ""),
                                                  "--xla_gpu_autotune_level"))
    deviceless = MeshSpec().build(compile_only_gpus("sm90", 1))
    running = MeshSpec().build(jax.devices()[:1])

    def lowered(mesh):
        # A bf16 product, which keeps XLA's Triton GEMM at any precision: the autotuner's options alone.
        placed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        shape = jax.ShapeDtypeStruct((64, 64), jnp.bfloat16, sharding=placed)
        return jax.jit(lambda x: x @ x).lower(shape)

    # One read offline skips the autotuner; one compiled for a device takes the tuned choices, every one
    # required, and keeps a CUB sort's choice off its key (`CUB_NO_FALLBACK`), as a device's own does.
    sorts = {"xla_gpu_deviceless_cub_mode": CUB_NO_FALLBACK}
    assert targeted(lowered(deviceless), None) == {"xla_gpu_autotune_level": 0}
    assert targeted(lowered(deviceless), {"xla_gpu_enable_triton_gemm": False}) == {
        "xla_gpu_enable_triton_gemm": False, "xla_gpu_autotune_level": 0}
    with for_devices():
        assert targeted(lowered(deviceless), None) == {
            **sorts, "xla_gpu_experimental_autotune_cache_mode": AUTOTUNE_CACHE_READ,
            "xla_gpu_require_complete_aot_autotune_results": True}
    assert targeted(lowered(running), {"xla_gpu_enable_triton_gemm": False}) == {
        "xla_gpu_enable_triton_gemm": False}
    # A step whose options are the run's own compiles as the half rung of XLA's defaults does, one
    # program (row 3506).
    objective = fine_tune(deviceless, split="auto").objective
    search = FitSearch(lower=None, options=lambda candidate: compiler_options() or None,
                       objective=objective, mesh=deviceless, xla_defaults=False, narrow=False, ns_group=None,
                       base=None, resumed=None)
    assert search.options(None) == search.required == (compiler_options() or None)
    # Its compile's stop past the devices' limit is no option of its program's: the ladder asks no rung of
    # XLA's defaults beside each place, the same program (train 993's F3), and does where its options differ.
    start = search.rung_now()

    def twins(chosen) -> list:
        search.chosen = chosen
        ladder = search.ladder(start, narrow=False, ns_group=None, remat_set=search.remat_set,
                               defaults=search.own_defaults())
        return [rung for rungs in ladder for rung in rungs if rung.xla_defaults]

    assert not twins({**(search.required or {}), LIMIT_EXIT: 2**30})
    assert twins({**(search.required or {}), "xla_gpu_enable_triton_gemm": True, LIMIT_EXIT: 2**30})
    monkeypatch.setenv("XLA_FLAGS", f"{os.environ['XLA_FLAGS']} --xla_gpu_autotune_level=2")
    assert targeted(lowered(deviceless), None) is None


@pytest.mark.parametrize("frontier", [False, True])
def test_compiling_ahead_on_gpus_of_no_stated_memory_is_refused_before_anything_traces(tmp_path, monkeypatch,
                                                                                       frontier):
    """A step compiled ahead for compile-only GPUs that state no memory, which every plan for them is made
    against, is refused before anything traces or compiles, its frontier or not (row 3107; PF03: it
    planned by compiling every rung, fourteen minutes of probes on a B200's step), naming the two ways to
    state it once: each GPU's memory, or a host's own serialized topology."""
    from dew.nn.kernels.generation import compile_only_gpus
    from dew.training.ahead import compile_ahead

    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "rungs"))
    trainer = fine_tune(MeshSpec().build(compile_only_gpus("sm90", 1)))
    with Compiles(None) as compiles, pytest.raises(ValueError):
        compile_ahead(trainer, token_rows(VOCAB, BATCH, SEQ_LEN + 1), frontier=frontier)
    assert compiles.count == 0


def test_a_configured_run_compiling_ahead_for_gpus_of_no_stated_memory_is_refused_before_it_reads(tmp_path):
    """A configured run compiling ahead for GPUs whose memory it does not state (no `capacity-gib`, no
    host topology) is refused before its first batch is read or its step planned (`stated_limit`), as a
    step compiled ahead for them is (`compile_ahead`): no step compiles and nothing is written."""
    from run_kinds import lm

    from dew.config.lm import LMRunConfig
    from dew.training.ahead import stated_limit

    run = LMRunConfig.cli(["--trainer.name", "ahead", "--trainer.checkpoint-dir", str(tmp_path / "runs"),
                           "--trainer.steps", "2"], default=lm(tmp_path, drawn=False))
    with Compiles("step") as compiles, pytest.raises(ValueError):
        run.lower(AheadTarget(generation="sm80", devices=1, path=str(tmp_path / "step.dew-aot")))
    assert compiles.count == 0 and not (tmp_path / "step.dew-aot").exists()
    assert stated_limit(gpus("sm80", limit=2**34)) == 2**34


def test_a_step_built_for_a_whole_gpu_loads_where_its_programs_fit_and_is_refused_naming_their_need(
        tmp_path, monkeypatch):
    """A step compiled ahead for a GPU of 64 GiB states what its programs need of a device
    (`programs_need`): a GPU whose allocator holds that much takes it, however much less than the room its
    plan was made in (the GPU tier's part, its pool sized by its tests' peaks, refused a tiny step built for
    80 GiB, rows 8094 and 8371), and one holding a byte less is refused, naming the need."""
    import dataclasses

    from dew.training.ahead import BuildTarget, compile_ahead

    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path))
    ahead = compile_ahead(fine_tune(MeshSpec().build(gpus("sm90", limit=2**36))), token_rows(VOCAB, BATCH,
                                                                                         SEQ_LEN + 1))
    need = ahead.built.memory
    assert need is not None and need < 2**30
    assert not ahead.built.differences(BuildTarget.of(gpus("sm90", limit=2**30)))
    held = BuildTarget.of(gpus("sm90"))
    assert not ahead.built.differences(dataclasses.replace(held, memory=need))
    refused = ahead.built.differences(dataclasses.replace(held, memory=need - 1))
    assert refused == [f"memory {need} bytes, not {need - 1}"], refused


def test_a_build_of_compile_only_gpus_holds_their_stated_memory_and_refuses_a_device_holding_less():
    """A build of compile-only GPUs (`BuildTarget.of`) holds the allocator's limit their stated memory gives
    (`device_memory`), as a build of attached GPUs holds theirs: a device of a lower limit is refused naming
    both, one of a higher limit takes it."""
    from dew.training.ahead import BuildTarget

    small, large = BuildTarget.of(gpus("sm90", limit=2**34)), BuildTarget.of(gpus("sm90", limit=2**35))
    assert (small.memory, large.memory) == (2**34, 2**35)
    assert large.differences(small) and not small.differences(large)


def test_an_ahead_plan_that_fits_no_device_reports_every_refused_program(tmp_path, monkeypatch):
    """An absent GPU with less memory than even the arguments gets no executable to run. Its
    refusal keeps every candidate's compile evidence, so the offline gate can check refusals too. The
    step starts at its ladder's last remat rung and its split is pinned, so the refusal comes after the
    few rungs left rather than after every policy's compile."""
    from dew.nn.backbones.decoder_block import REMAT_POLICIES
    from dew.training.ahead import compile_ahead
    from dew.training.planning import PlanRefused

    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path))
    # GPUs that hold the state and the batch between steps, and not a step's temporaries beside them.
    rows = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    held = max(fine_tune(MeshSpec().build(gpus("sm90")), remat=REMAT_POLICIES["full"]).validate(
        rows).device_bytes.values())
    trainer = fine_tune(MeshSpec().build(gpus("sm90", limit=held)), remat=REMAT_POLICIES["full"])
    with pytest.raises(PlanRefused) as refused:
        compile_ahead(trainer, rows)
    probes = refused.value.probes
    assert probes and all(not probe.fits for probe in probes)
    assert all(probe.compile_seconds > 0 for probe in probes)
    assert any(probe.headroom is not None and probe.headroom < 0 for probe in probes)
    # The allocator its closest's headroom read, stated beside it: the GPU's limit.
    (allocator,) = refused.value.allocators
    assert allocator["bytes_limit"] == held
    # What each probe placed beside its temporaries, which its headroom read.
    assert all(probe.beside is not None and probe.beside >= 0 for probe in probes)


def test_a_compile_only_gpu_holds_the_accumulator_a_step_builds_once(tmp_path, monkeypatch):
    """On a compile-only GPU of stated memory, a step read in two microbatches, whose first call builds
    its window's gradient sum, needs room for that sum once: the stated pool holds the step's arguments,
    the sum among them, and beside them the outputs that alias none, the temporaries and the batches
    prefetched next to the step (the 2113 review: the sum was counted among the arguments and again
    beside them, so an offline plan could refuse a step that fits). The pool charges what the first call
    builds by allocation, each shard rounded up (`allocated_bytes`), past what its arguments count; the
    same step run here builds the same buffers."""
    from dew.distributed.batches import batch_shardings, prefetched_bytes
    from dew.distributed.placement import allocated_bytes, placed_bytes
    from dew.training.ahead import compile_ahead, step_batch
    from dew.training.memory import temporaries_factor

    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path))
    capacity = 2**30
    trainer = fine_tune(MeshSpec().build(gpus("sm90", limit=capacity)), split=2)
    rows = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    ahead = compile_ahead(trainer, rows)
    (probe,) = listed(record(ahead.plan, "plan")["probes"], "probes", read=record, what="probes")
    batch = step_batch(trainer, rows)
    prefetched = prefetched_bytes(batch, batch_shardings(trainer.device_mesh, batch))
    temporaries = ahead.temp_bytes * temporaries_factor(trainer.device_mesh)
    run = fine_tune(MeshSpec().build(jax.devices()[:1]), split=2).fit(rows, steps=1)
    built = [part for part in (run.compute, run.window) if part is not None]
    assert probe["fits"] is True
    assert probe["headroom"] == (capacity - ahead.argument_bytes - (ahead.output_bytes - ahead.alias_bytes)
                                 - prefetched - temporaries - (allocated_bytes(built) - placed_bytes(built)))


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("spec", [MeshSpec(fsdp=8), MeshSpec(), MeshSpec(fsdp=4, tensor=2)],
                         ids=["fsdp8", "data8", "fsdp4_tensor2"])
def test_a_step_compiled_ahead_for_eight_gpus_reads_and_writes_what_the_run_on_eight_devices_does(spec):
    """Compiled ahead for eight compile-only H100s, the step reads and writes on each device the bytes
    the same trainer's step reads and writes on eight CPU devices, under each layout: its share of
    the state, whole or split, and of the batch. XLA states what it costs and every byte count of its
    memory, and its peak holds its arguments, its outputs that alias none, and its temporaries."""
    from dew.training.ahead import compile_ahead, step_batch

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    absent = fine_tune(spec.build(gpus("sm90", 8)))
    ahead = compile_ahead(absent, batch)
    run = fine_tune(spec.build())
    state, _, _ = run.place()
    run.compile(state, step_batch(run, batch))
    assert run.executable is not None
    here = run.executable.memory_analysis()
    assert here is not None
    assert (ahead.argument_bytes, ahead.output_bytes, ahead.alias_bytes) == (
        here.argument_size_in_bytes, here.output_size_in_bytes, here.alias_size_in_bytes)
    assert (ahead.generation, ahead.devices, ahead.mesh) == ("sm90", 8, dict(run.device_mesh.shape))
    assert ahead.executable and ahead.temp_bytes > 0
    assert ahead.flops is not None and ahead.flops.hardware > 0
    beside = ahead.output_bytes - ahead.alias_bytes + ahead.temp_bytes
    assert ahead.peak_bytes == ahead.argument_bytes + beside


def test_a_step_compiled_ahead_saves_and_loads_whole_and_runs_only_on_its_own_gpus(tmp_path):
    """A step saved is the step loaded, executable and what XLA stated of it, with the target it was built
    for (`BuildTarget`), the one rule every loader holds: this CPU host, which is not the topology the step
    was compiled for, refuses to load it, and a run reproducing it derives its launches for its GPU only on
    one of as many multiprocessors, as a balanced tail's grid past the GPU's residency can wait on a
    program that never starts (row 3606)."""
    import pickle

    from dew.training.ahead import AheadOfTime, compile_ahead

    ahead = compile_ahead(fine_tune(MeshSpec(fsdp=2).build(gpus("sm80", 2))),
                          token_rows(VOCAB, BATCH, SEQ_LEN + 1))
    ahead.save(tmp_path / "step.dew-aot")
    loaded = AheadOfTime.load(tmp_path / "step.dew-aot")
    assert loaded == ahead
    with pytest.raises(ValueError):
        loaded.loaded()
    # A step saved before steps carried their start's programs is refused as it loads, never an
    # AttributeError as its run places its state.
    earlier = dataclasses.replace(ahead)
    vars(earlier).pop("starts")
    (tmp_path / "earlier.dew-aot").write_bytes(pickle.dumps(earlier))
    with pytest.raises(ValueError):
        AheadOfTime.load(tmp_path / "earlier.dew-aot")
    assert loaded.derived_for(loaded.target) == loaded.target
    fewer = dataclasses.replace(loaded.target, multiprocessors=loaded.target.multiprocessors - 6)
    with pytest.raises(ValueError):
        loaded.derived_for(fewer)


GATED_DELTA_AHEAD = """
import dataclasses, json, sys
from pathlib import Path
from run_kinds import lm
from dew.config.lm import LMRunConfig
from dew.nn.kernels.generation import routes
from dew.registry import ModelConfig
from dew.training.ahead import AheadOfTime, CompiledAhead
from dew.training.targets import AheadTarget

root = Path(sys.argv[1])
model = ModelConfig("causal_transformer", {
    "emb_features": 128, "num_layers": 1, "num_heads": 2, "head_dim": 64, "mlp_features": 256,
    "dtype": "bfloat16", "layer_types": ["linear_attention"], "kinds": {"linear_attention": {"mixer": {
        "class": "gated_delta_net", "fields": {"linear_num_key_heads": 2, "linear_num_value_heads": 2,
                                               "linear_key_head_dim": 64, "linear_value_head_dim": 64}}}}})
run = LMRunConfig.cli(["--trainer.name", "ahead", "--trainer.checkpoint-dir", str(root / "runs"),
                       "--trainer.steps", "2"],
                      default=dataclasses.replace(lm(root, drawn=False), model=model))
with routes() as taken:
    try:
        run.lower(AheadTarget(generation="sm80", devices=1, path=str(root / "step"), capacity_gib=32.0))
    except CompiledAhead as ended:
        assert ended.code == 0
saved = AheadOfTime.load(root / "step")
print(json.dumps({"generation": saved.generation, "kernels": sorted({(route.choice, route.kernel)
                                                                      for route in taken})}))
"""
"""A bf16 gated-delta decoder's run compiled ahead for a compile-only A100 by its command's options, as a user
asks it, in a process of its own (`process_support.fresh`): its directory is its argument; it prints the
saved step's generation and every kernel its routes took."""


def test_a_gated_delta_steps_mosaic_kernels_compile_ahead_for_the_gpu_it_targets(tmp_path):
    """A bf16 gated-delta decoder's run compiled ahead for a compile-only A100 by its command
    (`AheadTarget(generation="sm80")`, rows 3916, 3977) saves a step for sm80 whose gated delta
    rule runs its Mosaic GPU recurrence, lowered for the architecture its trace's abstract device states,
    the A100's 8.0 (row 3885): a mesh's abstract device stated none, so Mosaic lowered for an H100's, and
    its `nvvm.elect.sync` failed sm80's LLVM instruction selection and aborted the process, every A100
    prepare of a model with a Mosaic route among them. A process of its own, as Mosaic compiles for one
    capability a process (`kernel_compile.compile_mosaic_kernels`)."""
    found = process_support.fresh(GATED_DELTA_AHEAD, str(tmp_path), platforms="cpu")
    assert found["generation"] == "sm80"
    assert ("gated_delta_rule", "mosaic_gpu") in {tuple(kernel) for kernel in found["kernels"]}, found


def test_a_gemm_asked_to_interpret_on_a_cpu_traced_ahead_for_a_gpu_holds_no_interpreter():
    """A GEMM asked to interpret its kernel on a CPU (`interpret_on_cpu`), traced on this CPU host for a GPU
    as a prepare traces a check-1 arm (`lowering_for`): its program holds no interpreted branch, whose host
    callbacks' ordered effect gave its executable a token parameter its run on the GPU never passed (row
    8099: a balanced dense kernel gradient's arm, "Execution supplied 2 arguments but compiled program
    expected 3")."""
    from dew.nn.kernels import gemm, generation
    from dew.training.ahead import lowering_for

    def gradient(rows, cotangent):
        return gemm.gemm_t(rows, cotangent, None, out_dtype=jnp.bfloat16, interpret_on_cpu=True)[0]

    with lowering_for("sm90"), generation.chosen(grouped_matmul="mosaic"):
        traced = jax.make_jaxpr(gradient)(jax.ShapeDtypeStruct((320, 128), jnp.bfloat16),
                                          jax.ShapeDtypeStruct((320, 2816), jnp.bfloat16))
    assert not traced.effects, traced.effects


EVERY_TEMPLATE_AHEAD = """
import json, sys
import jax
import optax
from token_support import token_rows

from dew.nn.kernels.generation import routes
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.training import Layout, MeshSpec, Trainer, TrainerAdvanced
from dew.nn.kernels.generation import compile_only_gpus
from dew.training.ahead import compile_ahead

model = models.build(
    "causal_transformer", vocab_size=1024, emb_features=1024, num_layers=2, num_heads=8, num_kv_heads=2,
    head_dim=128, mlp_features=8192, max_seq_len=512, dtype="bfloat16", attention_impl="template",
    layer_types=("linear_attention", "attention"),
    kinds={"attention": {}, "linear_attention": {"mixer": {"class": "gated_delta_net", "fields": {
        "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 64,
        "linear_value_head_dim": 64, "linear_conv_kernel_dim": 4}}}},
    mixture={"experts": 4, "top_k": 2, "layers": (1,)})
mesh = MeshSpec(fsdp=2).build(compile_only_gpus(sys.argv[1], 2, 2**36))
trainer = Trainer(LMObjective(model, 512), optax.adam(1e-3), key=jax.random.key(0), mesh=mesh,
                  advanced=TrainerAdvanced(layout=Layout(min_shard=256)))
with routes() as taken:
    ahead = compile_ahead(trainer, token_rows(1024, 8, 513))
print(json.dumps({"kernels": sorted({(route.choice, route.kernel) for route in taken})}))
"""
"""A hybrid MoE decoder whose step runs every template, compiled ahead for two compile-only GPUs of a
generation, in a process of its own (`process_support.fresh`): the generation is its argument; it prints
every kernel its routes took."""


@pytest.mark.parametrize("generation", gate_generations())
def test_a_step_running_every_template_compiles_ahead_over_two_compile_only_gpus(generation):
    """A bf16 hybrid decoder's training step, a gated delta net's layer beside attention and a routed
    experts' feed-forward under RMS norms and SwiGLU, its dense feed-forward wide enough that a kernel
    gradient's last wave balances (`tail.balanced`), compiled ahead for two compile-only GPUs of
    `generation` over an FSDP mesh: every template it runs (the attention template's mainloop, the gated
    delta rule's Mosaic GPU recurrence, the grouped matmul's Mosaic GPU mainloop) derives, lowers and
    compiles there. A derivation, a selector or a rewrite that
    computes a constant eagerly under the step's mesh places it on devices that run nothing here and fails
    the compile ("'Executable' has no 'local_devices'": the balanced tail's walk, 6be12fce5b1; the
    algebra's masks, b520fda9c59). Its own process, as Mosaic compiles for one capability a process."""
    found = process_support.fresh(EVERY_TEMPLATE_AHEAD, generation, platforms="cpu")
    kernels = {tuple(kernel) for kernel in found["kernels"]}
    assert {("attention", "template"), ("gated_delta_rule", "mosaic_gpu"),
            ("grouped_matmul", "mosaic")} <= kernels, kernels


@pytest.mark.parametrize("generation", ["sm80", "sm90"])
def test_tokamaxs_triton_attention_compiles_ahead_for_a_gpu_with_none_attached(generation):
    """tokamax's Triton attention, named, its forward and gradient traced for a compile-only GPU of
    `generation` on this host, which has no GPU attached: lowered and compiled as the GPU runs it, its route
    the kernel itself, not XLA's attention in its place; tokamax's check of the device a kernel runs on is
    the assumed GPU's (`generation.assumed`), where it read this host's ("Not supported on cpu")."""
    from dew.nn.attention import scaled_dot_product_attention
    from dew.nn.kernels.generation import assumed, compile_only_gpus, routes

    mesh = jax.sharding.Mesh(np.asarray(compile_only_gpus(generation, 1)), ("devices",))
    placed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    shapes = [jax.ShapeDtypeStruct(shape, jnp.bfloat16, sharding=placed)
              for shape in ((2, 256, 4, 64), (2, 256, 2, 64), (2, 256, 2, 64))]

    def loss(query, key, value):
        return jnp.sum(scaled_dot_product_attention(query, key, value, causal=True, implementation="triton")
                       .astype(jnp.float32))

    with assumed(generation), jax.sharding.use_abstract_mesh(mesh.abstract_mesh), routes() as taken:
        lowered = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2))).trace(*shapes).lower()
    assert {route.kernel for route in taken} == {"triton"}
    targets = {instruction.custom_call_target for instruction in xla.instructions(lowered.compiler_ir("hlo"))}
    assert custom_call_targets.TRITON in targets
    lowered.compile()


@pytest.mark.parametrize("implementation", ["auto", "template"])
@pytest.mark.parametrize("generation", gate_generations(
    scoped=True, tpu=True, needs=lambda family, architecture: family.issues_in_consumers))
def test_a_step_compiled_ahead_with_no_gpu_attached_runs_the_attention_its_gpu_runs(generation,
                                                                                    implementation):
    """A training step compiled ahead for a compile-only GPU of `generation` on this host, which has no GPU
    attached: its attention the route the GPU takes ('auto': FlashAttention where the cell declares it, its
    FFI targets lowered for the target the wheel is told, `attention_kernels.flash_target`; the template
    where named), lowered here, where a kernel this host could not lower would refuse the step
    (`selection.Unlowerable`; row 4616: every compile ahead's attention once ran as XLA's, and each GPU job
    compiled its own step at run time)."""
    from dew.training.ahead import compile_ahead

    model = models.build("causal_transformer", vocab_size=VOCAB, **TINY_DECODER, num_kv_heads=2,
                         attention_impl=implementation)
    trainer = Trainer(LMObjective(model, SEQ_LEN), optax.adam(1e-3), key=jax.random.key(0),
                      mesh=MeshSpec().build(gpus(generation)),
                      advanced=TrainerAdvanced(layout=Layout(min_shard=TINY_SHARD)))
    from dew.nn.kernels.generation import routes

    with routes() as taken:
        compile_ahead(trainer, token_rows(VOCAB, BATCH, SEQ_LEN + 1))
    if implementation == "template":
        assert {route.kernel for route in taken if route.choice == "attention"} == {"template"}


@pytest.mark.parametrize("kernel", ["cudnn", "flash"])
def test_a_program_holding_a_kernel_this_host_cannot_lower_is_refused_ahead_naming_it(kernel):
    """A program whose attention names a CUDA library kernel this host cannot lower for the target, compiled
    ahead for a canonical GPU: refused, the kernel, why and the program named, never compiled with another
    kernel in its place (row 4616). cuDNN's plan holds to its GPU's model, so no canonical target takes it
    (`generation.Assumption.portable`), here an sm90; FlashAttention for a generation its installed wheel
    holds no code for (`attention_kernels.flash_built_for`), any where no wheel is installed."""
    from dew.nn.attention import scaled_dot_product_attention
    from dew.nn.attention_kernels import flash_architectures, flash_built_for
    from dew.nn.kernels.selection import Unlowerable
    from dew.training.ahead import compile_programs

    built = flash_architectures() or ()
    target = "sm90" if kernel == "cudnn" else next(
        name for name in gate_generations() if not flash_built_for(name, built))

    def attended(query, key, value):
        return scaled_dot_product_attention(query, key, value, causal=True, implementation=kernel)

    shapes = [jax.ShapeDtypeStruct(shape, jnp.bfloat16)
              for shape in ((2, 256, 4, 64), (2, 256, 2, 64), (2, 256, 2, 64))]
    with pytest.raises(Unlowerable) as refused:
        compile_programs({"attention": (attended, shapes)}, target)
    assert (refused.value.choice, refused.value.kernel, refused.value.program) == ("attention", kernel,
                                                                                  "attention")


@COMPILES_ON_CPU
@pytest.mark.parametrize("generation", [name for name in gate_generations() if capability(name) >= (9, 0)])
def test_a_compile_only_gpu_assembles_mosaic_gpus_architecture_specific_code(generation):
    """A Mosaic GPU kernel compiled for a compile-only GPU of each generation from Hopper on, in a process
    that assembles for that generation alone (`compile_mosaic_kernels`): its PTX targets the
    architecture-specific variant (sm_90a, sm_100a, sm_120a), which the device's capability states, so it
    assembles. An sm120 stating no feature extension assembled for plain sm_120 and refused its own PTX
    ('.target sm_120a cannot be compiled for architecture sm_120')."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, staged_copy

    kernel = staged_copy((64, 64), jnp.bfloat16)
    spec = jax.ShapeDtypeStruct((64, 64), jnp.bfloat16)
    record, = compile_mosaic_kernels([KernelCase("copy", lambda: (kernel, (spec,)), {})], generation)
    assert "error" not in record, record["error"]


@pytest.mark.mesh(devices=8)
def test_programs_saved_before_executables_were_one_type_are_refused_to_compile_again(tmp_path):
    """A file of programs compiled ahead whose programs are not executables of this Dew's (an `AheadPrograms`
    saved before row 3670 made `Executable` its programs' one form) is refused as it loads, asking for them
    compiled again (row 3671), never an AttributeError as it runs."""
    import pickle

    from dew.training.ahead import AheadPrograms, BuildTarget

    target = BuildTarget("sm90", 132, 1, "0", "0", (), "x86_64")
    path = tmp_path / "old.dew-aot"
    path.write_bytes(pickle.dumps(AheadPrograms(target, {"step": ("program", b"executable", ())}, tuned=True,
                                                at_run=())))
    with pytest.raises(ValueError):
        AheadPrograms.load(path)


def test_programs_saved_before_their_files_said_how_they_were_tuned_are_refused(tmp_path):
    """A file of programs saved before it said whether XLA tuned them and which leave a choice to run time
    (`AheadPrograms.tuned`, `at_run`) is refused as it loads, asking for them compiled again, where the class
    default took it as tuned (row 8459)."""
    import pickle

    from dew.training.ahead import AheadPrograms, BuildTarget

    old = object.__new__(AheadPrograms)
    for field, value in (("target", BuildTarget("sm90", 132, 1, "0", "0", (), "x86_64")), ("programs", {}),
                         ("kept", None)):
        object.__setattr__(old, field, value)
    path = tmp_path / "old.dew-aot"
    path.write_bytes(pickle.dumps(old))
    with pytest.raises(ValueError):
        AheadPrograms.load(path)


@pytest.mark.parametrize("generation", ["sm80", "sm90", "sm100"])
def test_a_generations_abstract_mesh_is_its_compile_only_gpus(generation):
    """A mesh traced for a GPU on any host (`generation.gpu_mesh`) is the abstract mesh of compile-only GPUs
    of that generation with the same axes: one statement of the GPU in every key reading it (the Pallas
    lowering cache's, a program lowered for the GPU)."""
    from dew.nn.kernels.generation import compile_only_gpus, gpu_mesh

    devices = compile_only_gpus(generation)
    host = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",))
    assert gpu_mesh(host, generation) == jax.sharding.Mesh(np.array(devices[:1]), ("data",)).abstract_mesh


def test_compiling_ahead_takes_a_mesh_of_compile_only_gpus():
    """A trainer whose mesh is this host's own devices runs its step here; compiling it ahead is
    refused, as XLA's CPU executable, loaded again, finds none of its fused functions."""
    from dew.training.ahead import compile_ahead

    with pytest.raises(ValueError):
        compile_ahead(fine_tune(MeshSpec(fsdp=8).build()), token_rows(VOCAB, BATCH, SEQ_LEN + 1))


def test_a_step_compiled_ahead_is_one_program_and_a_run_of_another_refuses_it(tmp_path):
    """Two trainers of one config compile ahead one program, the digest a run lowers its first step
    to (`AheadOfTime.program`); a wider model's is another. A run handed another program's step, of its
    own target and tuned (as a GPU host's prepare compiles one; here this CPU's, a CPU being no target a
    step compiles ahead for), takes the file, then refuses its step before it trains, and again on its next
    fit, rather than compiling its own (row 1915)."""
    from dew.nn.kernels.generation import assumed
    from dew.training.ahead import AheadOfTime, BuildTarget, NotTheStepCompiledAhead, compile_ahead

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)

    def ahead(width=None):
        return compile_ahead(fine_tune(MeshSpec().build(gpus("sm80")), width=width), batch)

    first, again, wide = ahead(), ahead(), ahead(width=48)
    assert first.program == again.program != wide.program
    here = jax.devices()[:1]
    tuned = {**record(wide.plan, "plan"), "autotuned": True}
    own = dataclasses.replace(wide, built=BuildTarget.of(here), plan=tuned)
    own.save(tmp_path / "wide.dew-aot")
    run = fine_tune(MeshSpec().build(here), ahead=AheadOfTime.load(tmp_path / "wide.dew-aot"))
    # On the GPU it was compiled for, which a CPU is not, its program is what refuses it (train 332's review).
    with assumed(own.target):
        for _ in range(2):
            with pytest.raises(NotTheStepCompiledAhead):
                run.fit(batch, steps=1, log_every=1)


def test_a_configured_run_compiles_its_step_ahead_for_gpus_it_lacks_and_trains_nothing(tmp_path):
    """A run lowered for an `AheadTarget` of a generation, devices and path (`RunConfig.lower`,
    its default the run's own config) compiles its first step for one compile-only A100, writes it
    where it names, and ends with status 0, the step loadable and what it costs stated; it writes no
    checkpoint or tracking, as it trains nothing (row 1915)."""
    from run_kinds import lm

    from dew.config.lm import LMRunConfig
    from dew.training.ahead import AheadOfTime, CompiledAhead

    runs, path = tmp_path / "runs", tmp_path / "step.dew-aot"
    default = lm(tmp_path, drawn=False)
    assert isinstance(default, LMRunConfig)
    run = LMRunConfig.cli(["--trainer.name", "ahead", "--trainer.checkpoint-dir", str(runs),
                           "--trainer.steps", "2"], default=default)
    with pytest.raises(CompiledAhead) as ended:
        run.lower(AheadTarget(generation="sm80", devices=1, path=str(path), capacity_gib=1.0))
    assert ended.value.code == 0
    saved = AheadOfTime.load(path)
    assert (saved.generation, saved.devices) == ("sm80", 1) and saved.peak_bytes > 0
    assert saved.capacity == allocator_limit(2**30) and saved.plan is not None
    assert not runs.exists()


def test_a_configured_fresh_run_that_pauses_compiles_ahead_planned_as_it_trains_and_makes_no_directory(
        tmp_path):
    """A configured run that pauses (`Pause`, which its checkpoints resume) and keeps a local tier compiles
    its step ahead in a directory that holds nothing yet, planned with the checkpointer the run trains
    with, and creates neither its directory nor its local tier's: the compile ahead's trainer held no
    checkpointer where no directory was, so its plan refused the pause its run takes (review of row 4752),
    and asking a local tier for its step opened it (`Checkpoints.latest`)."""
    from run_kinds import lm

    from dew.training.ahead import AheadOfTime, CompiledAhead
    from dew.training.budget import Pause

    runs, local, path = tmp_path / "runs", tmp_path / "local", tmp_path / "step.dew-aot"
    default = lm(tmp_path, drawn=False)
    target = AheadTarget(generation="sm80", devices=1, path=str(path), capacity_gib=1)
    run = dataclasses.replace(default, trainer=dataclasses.replace(default.trainer, name="ahead",
                                                                   checkpoint_dir=str(runs),
                                                                   local_directory=str(local), local_every=2,
                                                                   steps=4, stop=Pause(0.5)))
    with pytest.raises(CompiledAhead) as ended:
        run.lower(target)
    assert ended.value.code == 0 and AheadOfTime.load(path).plan is not None
    assert not runs.exists() and not local.exists()


def test_a_configured_run_compiles_ahead_for_the_multiprocessors_it_states_and_loads_only_there(
        tmp_path, monkeypatch):
    """`AheadTarget.multiprocessors` states the GPU a step is compiled ahead for within its
    generation (an H100 PCIe's 114 against the canonical H100's 132, rows 3606 and 3669): XLA compiles the
    step and its sampled step over compile-only GPUs of 114 (as they state), and the saved build target and
    the facts its launches derived for are that GPU's, so a GPU of 114 runs it and the canonical one is
    refused; the rung its plan recorded is that GPU's, which a second build for it reuses, apart from the
    canonical build's beside it in one record directory. Its runs score no split midway, whose programs a
    build compiles too (row 4752), so its three builds are their steps' alone (row 3981); the first alone
    inspects, so compiles a sampled step, which no rung record's key holds. Its GPUs' room is what the step
    needs at two microbatches, less than one needs (each measured, `need_run`)."""
    import dataclasses

    from run_kinds import lm

    from dew.config.lm import LMRunConfig
    from dew.nn.kernels.generation import gpu_target
    from dew.training.ahead import AheadOfTime, CompiledAhead

    one, room = (need_run(tmp_path / f"need{split}", "sm90", split) for split in (1, 2))
    assert room < one, (one, room)
    records = tmp_path / "rungs"
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(records))

    builds = iter(range(3))

    def compiled(multiprocessors: int | None = None, inspected: bool = False) -> AheadOfTime:
        root = tmp_path / f"run{next(builds)}"
        root.mkdir()
        path = root / "step.dew-aot"
        run = LMRunConfig.cli(["--trainer.name", "ahead", "--trainer.checkpoint-dir", str(root / "runs"),
                               "--trainer.steps", "2",
                               *(["--trainer.inspection.every", "1"] if inspected else []),
                               "--trainer.eval-every", "None", "--trainer.checkpoint-every", "None"],
                              default=lm(root, drawn=False))
        with pytest.raises(CompiledAhead):
            run.lower(AheadTarget(generation="sm90", devices=1, path=str(path),
                                  capacity_gib=capacity_gib(room), multiprocessors=multiprocessors))
        return AheadOfTime.load(path)

    pcie = compiled(114, inspected=True)
    again = compiled(114)
    canonical = compiled()
    assert (pcie.built.multiprocessors, pcie.target.multiprocessors) == (114, 114)
    assert pcie.target.memory == allocator_limit(int(capacity_gib(room) * 2**30))
    assert 0 < pcie.built.memory <= pcie.target.memory
    # One target, its need the programs each carries: the inspected build's sampled step beside its step.
    assert pcie.sampled is not None
    assert again.built == dataclasses.replace(pcie.built, memory=again.built.memory)
    assert again.built.memory <= pcie.built.memory
    assert canonical.built.multiprocessors == gpu_target("sm90").multiprocessors
    held = dataclasses.replace(gpu_target("sm90"), device_kind="NVIDIA H100 PCIe", multiprocessors=114)
    assert pcie.derived_for(held) == pcie.target
    with pytest.raises(ValueError):
        pcie.derived_for(gpu_target("sm90"))
    assert len(list(records.rglob("*.json"))) == 2


def test_a_build_holds_a_call_to_the_memory_it_states_else_to_its_least_part_and_loads_where_that_is_held():
    """A program compiled ahead for an A100 holds a call's footprint (`device_memory`) to the memory its build
    states, else to the least an A100 holds, 40 GB, not the 80 GB part's (a build that kept an attention only
    that part holds failed to load on the other); a device holding less than a build's stated memory is
    refused it, one holding more or stating none runs it."""
    from dew.nn.kernels.generation import assumed, built_target, device_memory
    from dew.training.ahead import BuildTarget

    with assumed(built_target("sm80")):
        assert device_memory() == 40 * 10**9
    with assumed(built_target("sm80", memory=36 * 2**30)):
        assert device_memory() == 36 * 2**30
    build = BuildTarget("sm80", 108, 1, "0", "0", (), "x86_64", 36 * 2**30)
    assert build.differences(dataclasses.replace(build, memory=30 * 2**30))
    assert not build.differences(dataclasses.replace(build, memory=76 * 2**30))
    assert not build.differences(dataclasses.replace(build, memory=None))


def test_a_build_binds_its_loader_to_its_plugin_and_linked_libraries_and_not_to_its_topology():
    """Programs compiled ahead carry the CUDA plugin that compiled them and the cuBLAS and cuDNN they loaded
    (a cuDNN graph is serialized at compile), so a process of another plugin build, or of another version
    of a library the compile loaded, is refused them, naming which; a library the compile could not load
    wrote none of its state and binds no loader. The topology they were compiled over names the compile
    alone: a host's own GPUs run what a generation's canonical ones built."""
    from dew.training.ahead import BuildTarget

    build = BuildTarget("sm90", 132, 1, "0", "0", (), "x86_64", plugin="-cuda12-0.12.0.1.post73",
                        libraries=(("cublas", 1209), ("cudnn", 927)), topology=None)
    plugin = dataclasses.replace(build, plugin="-cuda13-0.12.0.1.post73")
    cudnn = dataclasses.replace(build, libraries=(("cublas", 1209), ("cudnn", 928)))
    assert [line.split()[0] for line in build.differences(plugin)] == ["plugin"]
    assert [line.split()[:2] for line in build.differences(cudnn)] == [["libraries", "cudnn"]]
    unloaded = dataclasses.replace(build, libraries=(("cublas", 1209), ("cudnn", None)))
    assert not unloaded.differences(cudnn)
    assert not build.differences(dataclasses.replace(build, topology="0" * 64))


def test_a_configured_run_compiling_ahead_with_its_frontier_records_it_in_the_saved_plan(tmp_path):
    """`AheadTarget.frontier` (`RunConfig.lower`) asks the step compiled ahead for its frontier,
    which its saved plan records (rows 3108, 3147): on GPUs whose room is what the step needs at two
    microbatches, less than one needs (each measured, `need_run`), where the plan's compile at one did not
    fit and the plan takes two, a rung above it that fits at one."""
    from run_kinds import lm

    from dew.config.lm import LMRunConfig
    from dew.training.ahead import AheadOfTime, CompiledAhead

    one, room = (need_run(tmp_path / f"need{split}", "sm80", split) for split in (1, 2))
    assert room < one, (one, room)
    runs, path = tmp_path / "runs", tmp_path / "step.dew-aot"
    run = LMRunConfig.cli(["--trainer.name", "ahead", "--trainer.checkpoint-dir", str(runs),
                           "--trainer.steps", "2"], default=lm(tmp_path, drawn=False))
    with pytest.raises(CompiledAhead) as ended:
        run.lower(AheadTarget(generation="sm80", devices=1, path=str(path), capacity_gib=capacity_gib(room),
                              frontier=True))
    assert ended.value.code == 0
    plan = record(AheadOfTime.load(path).plan, "plan")
    whole = [probe for probe in plan["probes"]
             if probe["split"] == 1 and probe["remat"] == plan["remat"]]
    assert whole and not any(probe["fits"] for probe in whole), plan["probes"]
    frontier = listed(plan["frontier"], "frontier", read=record, what="frontier")
    assert plan["split"] == 2 and frontier
    assert all(entry["fits"] is True and entry["split"] < plan["split"] for entry in frontier)


def test_a_configured_run_compiles_its_step_ahead_for_a_tpu_topology_with_its_stated_links(tmp_path):
    """`AheadTarget.generation` names a TPU topology as well as a CUDA generation: the run
    compiles its first step for that topology's compile-only devices here, through the plan a run takes,
    and writes it as any step compiled ahead (ledger 2551). The links it compiles with are the run's own
    (`diagnostics.links`), which the step records (ledger 2552)."""
    from run_kinds import lm

    from dew.config.lm import LMRunConfig
    from dew.training.ahead import AheadOfTime, CompiledAhead

    runs, path = tmp_path / "runs", tmp_path / "step.dew-aot"
    default = lm(tmp_path, drawn=False)
    assert isinstance(default, LMRunConfig)
    links = {"tensor": {"bytes_per_second": 4.5e11, "flops_per_second": 9.9e14}}
    run = LMRunConfig.cli(["--trainer.name", "ahead", "--trainer.checkpoint-dir", str(runs),
                           "--trainer.steps", "2", "--trainer.mesh.tensor", "2",
                           "--diagnostics.links",
                           json.dumps(links)], default=default)
    with pytest.raises(CompiledAhead) as ended:
        run.lower(AheadTarget(generation="v5e:2x2", devices=4, path=str(path)))
    assert ended.value.code == 0
    saved = AheadOfTime.load(path)
    assert (saved.generation, saved.devices) == ("v5e", 4) and saved.peak_bytes > 0
    assert saved.links["tensor"].bytes_per_second == 4.5e11
    assert not runs.exists()


def test_a_run_on_the_examples_clock_compiles_its_step_ahead_reading_its_progress(tmp_path):
    """A decision run bounded in examples, which its objective expands after its bound so no pass counts
    its steps, keeps its progress on the examples clock: compiled ahead for a GPU it lacks, its step reads
    that progress as fit's does (`Trainer.reading`), under a schedule over the run (`Linear`), and the run
    ends with the step saved, where it was compiled over no clock and refused the schedule (row 2434). Its
    tickets stream, of no record count, so its evaluations and checkpoints are counted in steps, as its run
    reads them (`CLOCKED`), which its compile reads too to compile its evaluations (row 4752)."""
    from run_kinds import decision

    from dew.decision.config import DecisionRunConfig
    from dew.training.ahead import AheadOfTime, CompiledAhead

    runs, path = tmp_path / "runs", tmp_path / "step.dew-aot"
    default = decision(tmp_path, drawn=False)
    assert isinstance(default, DecisionRunConfig)
    run = DecisionRunConfig.cli(["--trainer.name", "ahead", "--trainer.checkpoint-dir", str(runs),
                                 "--trainer.examples", "16", "--trainer.eval-every", "3",
                                 "--trainer.checkpoint-every", "3", "--optim.lr", "0.001",
                                 "optim.schedule:linear"], default=default)
    with pytest.raises(CompiledAhead) as ended:
        run.lower(AheadTarget(generation="sm80", devices=1, path=str(path), capacity_gib=4.0))
    assert ended.value.code == 0
    assert AheadOfTime.load(path).peak_bytes > 0


CLOCKED = ["--trainer.examples", "48", "--trainer.eval-every", "3", "--trainer.checkpoint-every", "3",
           "--trainer.multi-host", "False", "--trainer.compilation-cache-dir", "None",
           "--trainer.cooldowns.at", "50%", "--trainer.cooldowns.length", "25%",
           "--optim.lr", "0.001", "optim.schedule:linear"]
"""The clocked decision run's flags (the clocked GPU cell's): its tickets stream, of no record count, so its
evaluations and checkpoints are counted in steps (ledger 3153)."""


def test_a_decision_runs_step_compiled_ahead_is_the_program_its_run_lowers_as_fit_hands_its_step(tmp_path):
    """The clocked GPU cell's property, read where a host has no GPU (ledger 3153): the decision run compiled
    ahead for eight sm90 GPUs (`AheadTarget`) and the same run lowered for them as `fit` hands it
    its step, its batch placed on the devices (`LowerFor`), are one program, so a run handed the
    step it compiled ahead takes it. A step lowered over the host batch the loader gave and over that batch
    placed came to two module texts, a helper call's function told apart by its arguments' shardings."""
    from run_kinds import decision

    from dew.decision.config import DecisionRunConfig
    from dew.nn.kernels.options import program_digest
    from dew.training._lowering import Lowered
    from dew.training.ahead import AheadOfTime, CompiledAhead

    path = tmp_path / "ahead.dew-aot"
    stated = ["--trainer.checkpoint-dir", str(tmp_path / "runs"), *CLOCKED]
    ahead = DecisionRunConfig.cli(["--trainer.name", "ahead", *stated],
                                  default=decision(tmp_path, drawn=False))
    with pytest.raises(CompiledAhead):
        ahead.lower(AheadTarget(generation="sm90", devices=8, path=str(path), capacity_gib=80.0))
    lowering = DecisionRunConfig.cli(["--trainer.name", "lowered", *stated],
                                     default=decision(tmp_path, drawn=False))
    with pytest.raises(Lowered) as lowered:
        lowering.lower(LowerFor("sm90"))
    assert program_digest(lowered.value.lowered) == AheadOfTime.load(path).program


def a_clocked_run_with_a_cooldown_trains_from_its_step_compiled_ahead_compiling_no_other(tmp_path):
    """A GPU flow's cell (tests/test_gpu_training.py; it trains from a step compiled ahead on this host's own
    GPU): a decision run on the examples clock under a schedule over the run, with a cooldown, compiled
    ahead for this GPU's generation (`RunConfig.lower`) and then handed
    the step it saved (`--trainer.ahead`), trains from that step through its cooldown's branch, its last
    window's too, and compiles no step program of its own: each batch it steps closes its window as the
    one it compiled ahead over did (ledgers 2434, 2527)."""
    from run_kinds import decision

    from dew.decision.config import DecisionRunConfig
    from dew.training.ahead import CompiledAhead

    runs, path = tmp_path / "runs", tmp_path / "step.dew-aot"
    default = decision(tmp_path, drawn=False)
    stated = ["--trainer.checkpoint-dir", str(runs), *CLOCKED]
    # This GPU's own target, its multiprocessors as it counts them (an H100 PCIe's 114).
    compiling = DecisionRunConfig.cli(["--trainer.name", "ahead", *stated], default=default)
    with pytest.raises(CompiledAhead):
        compiling.lower(this_gpus_ahead(path))
    handed = DecisionRunConfig.cli(["--trainer.name", "handed", *stated, "--trainer.ahead", str(path)],
                                   default=default)
    with Compiles("step") as compiles:
        state = handed.run()
    assert compiles.count == 0 and int(state.updates) > 0


def a_run_handed_an_untuned_file_by_its_command_compiles_its_own(tmp_path, monkeypatch):
    """A GPU flow's cell (tests/test_gpu_training.py): a decision run compiled ahead through its command line
    for this GPU's own target (its generation, multiprocessors and memory; no topology: XLA's untuned
    program) and then handed the file (`--trainer.ahead`), its XLA_FLAGS leaving the autotuner on as a user's
    do, does not run the file's programs: it compiles its own step on the GPU and trains, said once as an
    untuned file (`AheadOfTime.runnable`, `UNTUNED`, UB-4792-AHEAD-TIME)."""
    from lane_environment import without_flags
    from run_kinds import decision

    from dew._devices import xla_flag
    from dew.decision.config import DecisionRunConfig
    from dew.training.ahead import UNTUNED, CompiledAhead

    runs, path = tmp_path / "runs", tmp_path / "step.dew-aot"
    default = decision(tmp_path, drawn=False)
    stated = ["--trainer.checkpoint-dir", str(runs), *CLOCKED]
    # The autotuner left on from the compile ahead on, as a user's flags leave it: XLA's flags are part of a
    # step's target (`BuildTarget.flags`), which the run handed it holds to its own.
    flags = without_flags(os.environ.get("XLA_FLAGS", ""), "--xla_gpu_autotune_level")
    monkeypatch.setenv("XLA_FLAGS", flags)
    assert xla_flag("xla_gpu_autotune_level") is None
    # This GPU's own target, so the file differs from what the run compiles by its tuning alone.
    with pytest.raises(CompiledAhead) as compiled:
        DecisionRunConfig.cli(["--trainer.name", "ahead", *stated],
                              default=default).lower(this_gpus_ahead(path))
    assert not compiled.value.ahead.tuned
    handed = DecisionRunConfig.cli(["--trainer.name", "handed", *stated, "--trainer.ahead", str(path)],
                                   default=default)
    said: list[logging.LogRecord] = []
    heard = logging.Handler(logging.WARNING)
    heard.emit = said.append
    logging.getLogger("dew.training.ahead").addHandler(heard)
    try:
        with Compiles("step") as compiles:
            state = handed.run()
    finally:
        logging.getLogger("dew.training.ahead").removeHandler(heard)
    assert compiles.count >= 1 and int(state.updates) > 0
    assert [record.msg for record in said] == [UNTUNED]


ACCUMULATION_CELLS = [("sm80", "bfloat16"), ("sm90", "float32")]
"""Each generation and each compute dtype once (testing.md's pairwise settings): a bfloat16 step reads
its parameters through narrow copies on a GPU (`dew.training.narrow`), a float32 step reads them whole."""

WIDE_MLP = {**TINY_DECODER, "emb_features": 64, "mlp_features": 2048}
"""A decoder whose parameters, 3.3 MB in float32, dwarf what a row of 16 tokens keeps for its backward:
each buffer of a value a parameter is then larger than one microbatch's activations."""


@pytest.mark.parametrize(("generation", "dtype"), ACCUMULATION_CELLS)
def test_an_accumulating_step_holds_its_gradient_sum_and_no_other_buffer_a_parameter(generation, dtype):
    """Row 1900: a step that pools four microbatches into a commit, compiled ahead for a GPU, holds
    beyond the step that commits each one its window's gradient sum, a value a parameter in the
    parameter's dtype, among its arguments, and no more temporaries than one microbatch's activations,
    nor more at its peak, outputs included, as XLA's memory analysis of both programs states them. A
    resident copy of the parameters, a second gradient buffer or a narrow copy kept beside the sum,
    each two bytes a parameter or more, exceeds that: row 1661's 14 GB on a 4B run surfaced only on a
    GPU, and this finds its kind on a CPU host."""
    from dew.training.ahead import compile_ahead

    model = models.build("causal_transformer", vocab_size=VOCAB, **WIDE_MLP, num_kv_heads=2, dtype=dtype)

    def ahead(accumulation: int, rows: int):
        trainer = Trainer(LMObjective(model, SEQ_LEN), OptimConfig(optimizer=AdamW(), lr=1e-3),
                          key=jax.random.key(0), mesh=MeshSpec().build(gpus(generation)),
                          advanced=TrainerAdvanced(layout=Layout(min_shard=TINY_SHARD)),
                          accumulation=accumulation,
                          split=1)
        trainer.validate(steps=8)
        return compile_ahead(trainer, token_rows(VOCAB, rows, SEQ_LEN + 1))

    one, window, two_rows = ahead(1, 1), ahead(4, 1), ahead(1, 2)
    shapes = jax.eval_shape(model.init, jax.random.key(0), jnp.zeros((1, SEQ_LEN), jnp.int32))["params"]
    gradient = sum(leaf.size * leaf.dtype.itemsize for leaf in jax.tree.leaves(shapes))
    held = 4 * token_rows(VOCAB, 1, SEQ_LEN + 1)["text"].nbytes
    activations = two_rows.temp_bytes - one.temp_bytes
    # The cell can tell: half the sum, a narrow copy's two bytes a parameter, is more than it allows.
    assert 0 < activations < gradient / 2
    # The window's arguments: its gradient sum, and at most a token row a microbatch it holds to replay.
    assert window.argument_bytes - one.argument_bytes <= gradient + held
    # Its temporaries beside the committing step's: fewer than the smallest buffer the cell catches, a narrow
    # copy's two bytes a parameter. XLA schedules for a target of stated memory, which may hold a
    # microbatch's working set a little longer, so they are not held to one microbatch's activations.
    narrow = 2 * sum(leaf.size for leaf in jax.tree.leaves(shapes))
    assert window.temp_bytes - one.temp_bytes < narrow
    # Its whole peak, outputs that alias no argument among it: the sum and the rows beside the committing
    # step's, and less than a narrow copy more.
    assert window.peak_bytes - one.peak_bytes < gradient + held + narrow


def a_step_compiled_ahead_and_saved_runs_on_its_gpu_in_the_run_s_place(tmp_path):
    """A step compiled ahead for one compile-only GPU of this GPU's own target (`this_gpus`), saved,
    and loaded again, runs on this GPU in the place of the run's own compile of it: handed the arguments the
    run's step is (the state the trainer places, the batch, the first part), it makes outputs of
    the same tree, shapes, dtypes and placements, the same step and microstep counters, a finite
    loss, and parameters it has updated. Its floats are not held to the run's own bit for bit: XLA
    tunes a GEMM's algorithm on the GPU it compiles on and takes its default with none attached."""
    from dew.nn.kernels.generation import chosen
    from dew.state import apart
    from dew.training.ahead import AheadOfTime, compile_ahead, step_batch

    # A step that reads its parameters whole, so its arguments are the placed state's own.
    with chosen(narrow_copies="off"):
        compile_ahead(fine_tune(MeshSpec().build(this_gpus())),
                      token_rows(VOCAB, BATCH, SEQ_LEN + 1)).save(tmp_path / "step.dew-aot")
        run = fine_tune(MeshSpec().build(jax.devices()[:1]))
        batch = step_batch(run, token_rows(VOCAB, BATCH, SEQ_LEN + 1))
        state, _, _ = run.place()
        run.compile(state, batch)
    ahead = AheadOfTime.load(tmp_path / "step.dew-aot")
    assert run.executable is not None
    here = run.executable.memory_analysis()
    assert here is not None
    assert (ahead.argument_bytes, ahead.output_bytes, ahead.alias_bytes) == (
        here.argument_size_in_bytes, here.output_size_in_bytes, here.alias_size_in_bytes)
    shardings, _ = run.executable.input_shardings

    def arguments():
        # Each call takes the state it is handed, so each is handed its own placing of the same one.
        given, _, _ = run.place()
        return jax.device_put((*apart(dataclasses.replace(given, compute=None)), batch, np.int32(0)),
                              shardings)

    theirs, ours = run.executable(*arguments()), ahead.loaded()(*arguments())
    assert jax.tree.structure(ours) == jax.tree.structure(theirs)
    for left, right in zip(jax.tree.leaves(ours), jax.tree.leaves(theirs), strict=True):
        assert (left.shape, left.dtype, left.sharding) == (right.shape, right.dtype, right.sharding)
    moving, *outputs = ours
    np.testing.assert_array_equal(np.asarray(moving.step), np.asarray(theirs[0].step))
    np.testing.assert_array_equal(np.asarray(moving.microstep), np.asarray(theirs[0].microstep))
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(outputs)
               if jnp.issubdtype(leaf.dtype, jnp.floating))
    initial = jax.tree.leaves(state.variables["params"])
    assert any(not np.array_equal(np.asarray(new), np.asarray(old))
               for new, old in zip(jax.tree.leaves(moving.variables["params"]), initial, strict=True))


def a_run_handed_its_step_compiled_ahead_trains_from_it_on_its_gpu(tmp_path):
    """A step compiled ahead for one compile-only GPU of this GPU's own target (`this_gpus`), saved and handed
    to the run (`TrainerAdvanced(ahead=...)`), is the program the run lowers its first step to, and the run
    trains from it, its plan's first compile the step compiled ahead: its steps advance and its parameters
    move (row 1915). A wider model's step for this GPU is another program, refused. The model's attention
    takes XLA's path, which a step compiled with no GPU attached traces too."""
    from dew.nn.kernels.generation import chosen
    from dew.training.ahead import AheadOfTime, compile_ahead

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    with chosen(narrow_copies="off"):
        for name, width in (("step", None), ("wide", 48)):
            compile_ahead(fine_tune(MeshSpec().build(this_gpus()), xla=True,
                                    width=width), batch).save(tmp_path / f"{name}.dew-aot")
        here = MeshSpec().build(jax.devices()[:1])
        run = fine_tune(here, ahead=AheadOfTime.load(tmp_path / "step.dew-aot"), xla=True)
        initial = jax.tree.leaves(run.initial_state().variables["params"])
        state = run.fit(batch, steps=2, log_every=1)
        other = fine_tune(here, ahead=AheadOfTime.load(tmp_path / "wide.dew-aot"), xla=True)
        with pytest.raises(ValueError):
            other.fit(batch, steps=1, log_every=1)
    first, *_ = listed(record(run.plan, "plan")["probes"], "probes", read=record, what="probes")
    assert int(state.step) == 2 and first["ahead"] is True
    assert any(not np.array_equal(np.asarray(new), np.asarray(old))
               for new, old in zip(jax.tree.leaves(state.variables["params"]), initial, strict=True))


def test_a_run_takes_an_untuned_step_only_where_its_own_compile_tunes_nothing(tmp_path, caplog, monkeypatch):
    """A run takes a step compiled ahead (`AheadOfTime.runnable`) only for its own target, refused for any
    other, tuned or not; for its own, a step compiled without its GPU kind's tuned choices (`tuned`, read
    off its plan: compiled on compile-only GPUs outside a prepare), whose XLA products ran up to 48 times
    slower on a GH200 (UB-4792-AHEAD-TIME), is not taken where the run's own compile tunes (its step
    compiled on its devices as without a file, said once), and is where it tunes nothing either (XLA_FLAGS
    setting the autotuner's level 0, as a repeatable lane's do), its products then the run's own, unless
    it leaves a convolution's algorithm to run time (`at_run`). A step of this host's own target stands for
    the GPU host's here: a CPU is no target a step compiles ahead for."""
    from dew._cache import uncached
    from dew.training.ahead import AheadOfTime, BuildTarget, Executable, compile_ahead

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    compile_ahead(fine_tune(MeshSpec().build(gpus("sm90"))), batch).save(tmp_path / "step.dew-aot")
    handed = AheadOfTime.load(tmp_path / "step.dew-aot")
    here = jax.devices()[:1]
    assert not handed.tuned
    with pytest.raises(ValueError):
        handed.runnable(here)
    own = dataclasses.replace(handed, built=BuildTarget.of(here))
    caplog.clear()
    with caplog.at_level("WARNING", logger="dew.training.ahead"):
        assert own.runnable(here) is None
    assert len([entry for entry in caplog.records if entry.name == "dew.training.ahead"]) == 1
    tuned = dataclasses.replace(own, plan={**record(own.plan, "plan"), "autotuned": True})
    assert tuned.tuned and tuned.runnable(here) is tuned
    # Loaded direct, as a tool or a caller of `taken` loads it, the untuned step is refused alike and the
    # tuned one loads (row 8986): a step this host loads as it is, over its every device.
    every = jax.devices()
    replicated = jax.sharding.NamedSharding(jax.sharding.Mesh(np.asarray(every), ("devices",)),
                                            jax.sharding.PartitionSpec())
    with uncached():
        doubled = jax.jit(lambda x: x * 2.0, out_shardings=replicated).lower(
            jax.ShapeDtypeStruct((4,), jnp.float32, sharding=replicated)).compile()
    loads = dataclasses.replace(own, built=BuildTarget.of(every), executable=Executable.of(doubled))
    with pytest.raises(ValueError):
        loads.loaded()
    assert dataclasses.replace(loads, plan=tuned.plan).loaded() is not None
    # Built under the flags it runs under: XLA's flags are part of a step's target (`BuildTarget.flags`).
    monkeypatch.setenv("XLA_FLAGS", f"{os.environ.get('XLA_FLAGS', '')} --xla_gpu_autotune_level=0")
    level_zero = dataclasses.replace(handed, built=BuildTarget.of(here))
    assert level_zero.runnable(here) is level_zero
    # One leaving a convolution's algorithm to run time is never taken: the run's own compile picks it.
    at_run = dataclasses.replace(level_zero, at_run=True)
    assert at_run.runnable(here) is None
    with pytest.raises(ValueError):
        own.runnable(here)



def test_a_watched_runs_step_compiled_ahead_carries_its_sampled_step():
    """A run that inspects compiles its sampled step beside its step (`Candidate.sampled`), and a step
    compiled ahead for it carries both (`AheadOfTime.sampled`), each its own program, so the run handed it
    compiles neither on its target (train 332's review)."""
    from dew.training import Inspection
    from dew.training.ahead import compile_ahead

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    devices = gpus("sm90")
    watched = compile_ahead(fine_tune(MeshSpec().build(devices), inspection=Inspection(every=1)), batch)
    plain = compile_ahead(fine_tune(MeshSpec().build(devices)), batch)
    assert watched.sampled is not None and plain.sampled is None
    assert watched.sampled.program != watched.program and watched.program == plain.program


def a_watched_run_handed_its_steps_compiled_ahead_compiles_neither(tmp_path):
    """A GPU flow's cell (tests/test_gpu_training.py): a run that inspects, handed its step and sampled step
    compiled ahead for this host's own GPU, takes both (`taken`) and compiles no step program on its GPU
    (`Compiles`): its first plan's compile is the step compiled ahead, and it trains from it (train 332's
    review)."""
    from dew.nn.kernels.generation import chosen
    from dew.telemetry.compiles import Compiles
    from dew.training import Inspection
    from dew.training.ahead import AheadOfTime, compile_ahead

    batch, watched = token_rows(VOCAB, BATCH, SEQ_LEN + 1), Inspection(every=1)
    with chosen(narrow_copies="off"):
        compile_ahead(fine_tune(MeshSpec().build(this_gpus()), xla=True,
                                inspection=watched), batch).save(tmp_path / "step.dew-aot")
        run = fine_tune(MeshSpec().build(jax.devices()[:1]), xla=True, inspection=watched,
                        ahead=AheadOfTime.load(tmp_path / "step.dew-aot"))
        with Compiles("step") as compiles:
            state = run.fit(batch, steps=1, log_every=1)
    first, *_ = listed(record(run.plan, "plan")["probes"], "probes", read=record, what="probes")
    assert int(state.step) == 1 and first["ahead"] is True and compiles.count == 0


@pytest.mark.tpu(reason="runs a step compiled ahead for this host's TPU topology as a run's first step")
def test_a_run_handed_its_step_compiled_ahead_for_a_tpu_topology_trains_from_it(tmp_path):
    """A step compiled ahead for this host's one-chip TPU topology (`compile_only_tpus`), saved and handed
    to the run on the TPU itself, is the program the run lowers its first step to, and the run trains
    from it, its plan's first compile the step compiled ahead: the TPU run row 2090 asks for."""
    from dew.nn.kernels.generation import gpu_target
    from dew.training.ahead import AheadOfTime, compile_ahead, compile_only_tpus

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    chip = gpu_target(jax.devices()[0].device_kind).generation
    compile_ahead(fine_tune(MeshSpec().build(compile_only_tpus(f"{chip}:1x1")), xla=True),
                  batch).save(tmp_path / "step.dew-aot")
    run = fine_tune(MeshSpec().build(jax.devices()[:1]), ahead=AheadOfTime.load(tmp_path / "step.dew-aot"),
                    xla=True)
    state = run.fit(batch, steps=2, log_every=1)
    first, *_ = listed(record(run.plan, "plan")["probes"], "probes", read=record, what="probes")
    assert int(state.step) == 2 and first["ahead"] is True


def test_a_run_naming_its_input_alone_compiles_its_step_ahead(tmp_path):
    """An objective that names its input field alone (`Supervised(inputs="x")`) compiles its step ahead
    (`AheadTarget.compiled`) as `fit` would start it: its input declared at the shape the loader's first
    batch gives (`Trainer._declare_input`) before the run is checked (train 325's review of ledger
    2551)."""
    from flax import linen as nn

    from dew.data.dataset import Loader
    from dew.objectives.supervised import Supervised
    from dew.training.targets import AheadTarget

    data = Loader(train=lambda partition: iter([{"x": np.ones((BATCH, 8), np.float32),
                                                  "y": np.zeros((BATCH, 4), np.float32)}] * 4),
                  val=None, batch=BATCH, records=None)
    trainer = Trainer(Supervised(nn.Dense(4), lambda outputs, batch: (outputs - batch["y"]) ** 2, inputs="x"),
                      optax.sgd(0.1), key=jax.random.key(0),
                      mesh=MeshSpec().build(gpus("sm90")))
    ended = AheadTarget("sm90", 1, str(tmp_path / "step.dew-aot")).compiled(trainer, data, steps=1)
    assert ended.ahead.devices == 1 and trainer.objective.inputs is not None
    # Into the cache, over canonical GPUs and not a host's own topology, its compiles key entries no run
    # reads (`host_gpus`): refused, naming the two forms that are read.
    with pytest.raises(ValueError):
        AheadTarget("sm90", 1).compiled(trainer, data, steps=1)


FAST = {"tensor": Link(1e18, 1e9)}
"""A tensor axis whose link moves a projection's bytes in no time beside its FLOPs: any projection
spreads over it, as over a measured NVLink one may."""


def latent(mesh, *, ahead=None, links=None, checkpoints=None):
    """A decoder of latent attention, whose down projection a tensor axis's link may spread
    (`dew.nn.sharding.down_projection`), trained over `mesh`."""
    from dew.nn.mla import MLAMixer

    model = models.build("causal_transformer", vocab_size=VOCAB, **TINY_DECODER, attention_impl="xla",
                         mixer=MLAMixer(q_lora_rank=8, kv_lora_rank=8, qk_nope_head_dim=4,
                                        qk_rope_head_dim=4, v_head_dim=4))
    return Trainer(LMObjective(model, SEQ_LEN), optax.adam(1e-3), key=jax.random.key(0), mesh=mesh,
                   advanced=TrainerAdvanced(layout=Layout(min_shard=TINY_SHARD), ahead=ahead, links=links),
                   checkpoints=checkpoints)


def test_a_step_compiled_ahead_records_the_links_its_program_was_decided_by():
    """A latent-attention step compiled ahead over a tensor axis of two compile-only GPUs, which no
    process holds to measure, spreads no projection over it and records that link
    (`AheadOfTime.links`); the same step over a link that spreads one is another program, which a run
    on GPUs that measure such a link would lower in its place (row 2229)."""
    from dew.training.ahead import compile_ahead

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    target = MeshSpec(tensor=2).build(gpus("sm90", 2))
    saved = compile_ahead(latent(target), batch)
    assert saved.links["tensor"].bytes_per_second is None
    assert compile_ahead(latent(target, links=FAST), batch).program != saved.program


def test_a_stated_link_is_the_one_a_step_compiled_ahead_records_and_a_handed_run_traces_with(caplog):
    """A step compiled ahead over a link its trainer states, as `diagnostics.links` states one, records
    that link whole, bandwidth and peak, and is the program a stated link spreads by; a run
    handed it that states another traces with the step's, and says so once (train 300's review: the
    step recorded the bandwidth alone, so a run stating the same link traced with its own peak)."""
    from dew.training.ahead import compile_ahead

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    target = MeshSpec(tensor=2).build(gpus("sm90", 2))
    stated = compile_ahead(latent(target, links=FAST), batch)
    (recorded,) = stated.links.values()
    assert (recorded.bytes_per_second, recorded.flops_per_second, recorded.spread) == (1e18, 1e9, True)
    assert stated.program != compile_ahead(latent(target), batch).program
    other = {"tensor": Link(1.0, 1.0)}
    with caplog.at_level("WARNING", logger="dew.training.ahead"):
        assert stated.over(other) == stated.over(FAST) == stated.links
    assert len(caplog.records) == 1


def a_run_handed_a_step_compiled_ahead_traces_with_the_links_it_was_compiled_with(tmp_path):
    """A run on two GPUs whose link would spread a projection (`FAST`), handed a step compiled ahead
    over compile-only GPUs that spread none, traces with the link the step records and trains from it,
    rather than lowering another program and refusing it; resumed from its checkpoint without the step,
    it traces with that link again, not one it measures (row 2229)."""
    from dew.checkpoints import Checkpoints
    from dew.nn.kernels.generation import chosen
    from dew.training.ahead import AheadOfTime, compile_ahead

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    spec, path = MeshSpec(tensor=2), tmp_path / "step.dew-aot"
    here = spec.build(jax.devices()[:2])
    with chosen(narrow_copies="off"):
        target = spec.build(this_gpus(2))
        compile_ahead(latent(target), batch).save(path)
        saved = str(tmp_path / "run")
        run = latent(here, ahead=AheadOfTime.load(path), links=FAST, checkpoints=Checkpoints(saved))
        state = run.fit(batch, steps=2, log_every=1)
        resumed = latent(here, checkpoints=Checkpoints(saved))
        again = resumed.fit(batch, steps=3, log_every=1)
    first, *_ = listed(record(run.plan, "plan")["probes"], "probes", read=record, what="probes")
    assert int(state.step) == 2 and first["ahead"] is True
    assert int(again.step) == 3 and resumed.links["tensor"].bytes_per_second is None


def a_run_takes_the_saved_offline_split_before_loading_its_executable(tmp_path):
    """An offline plan for two microbatches is handed to a live run with its split unpinned. The
    live planner takes the saved split before lowering, so the artifact's program is reused and its
    two donated microbatch calls produce one committed update, not a mismatch or a recompile."""
    from dew.nn.kernels.generation import chosen
    from dew.training.ahead import AheadOfTime, compile_ahead

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    with chosen(narrow_copies="off"):
        target = MeshSpec().build(this_gpus())
        saved = compile_ahead(fine_tune(target, xla=True, split=2), batch)
        path = tmp_path / "split.dew-aot"
        saved.save(path)
        here = MeshSpec().build(jax.devices()[:1])
        run = fine_tune(here, ahead=AheadOfTime.load(path), xla=True, split="auto")
        state = run.fit(batch, steps=1, log_every=1)
    plan = record(run.plan, "plan")
    first, *_ = listed(plan["probes"], "probes", read=record, what="probes")
    assert int(state.step) == 1 and int(state.updates) == 1
    assert plan["split"] == 2 and first["ahead"] is True


def test_programs_compiled_ahead_save_and_load_whole_and_load_only_on_their_own_target(tmp_path):
    """Named programs compiled ahead for a GPU this CPU host lacks (`compile_programs`), one of them reading
    no argument, are the programs loaded after saving, each an executable this host's CPU is no target of:
    loading them here is refused, naming the GPU kind it is not (ledger 2716: an H100 node loads what a CPU
    host built, and runs it alone)."""
    from dew.training.ahead import AheadPrograms, BuildTarget, compile_programs

    square = jax.ShapeDtypeStruct((128, 128), jnp.float32)
    built = compile_programs({"dot": (lambda x, y: x @ y, [square, square]),
                              "iota": (lambda: jnp.arange(8.0), [])}, "sm90")
    assert set(built.programs) == {"dot", "iota"}
    assert (built.target.generation, built.target.devices) == ("sm90", 1)
    built.save(tmp_path / "programs.dew-aot")
    loaded = AheadPrograms.load(tmp_path / "programs.dew-aot")
    assert loaded == built
    with pytest.raises(ValueError):
        loaded.loaded()
    here = BuildTarget.of(jax.devices()[:1])
    # This host's CPU is another generation, of no multiprocessors, a processor a GPU's target has none
    # of (`BuildTarget.cpu`), and no CUDA plugin, which compiled the GPU's build (`BuildTarget.plugin`).
    assert [line.split()[0] for line in built.target.differences(here)] == ["generation", "multiprocessors",
                                                                             "cpu", "plugin"]
    # A kind of the generation with its multiprocessors is its target, whatever its memory or its name: the
    # build's target is sm90's facts, 132 multiprocessors.
    assert (built.target.generation, built.target.multiprocessors) == ("sm90", 132)



def test_programs_compiled_ahead_for_a_generation_are_untuned_and_refused_where_a_compile_tunes(
        tmp_path, monkeypatch):
    """Named programs compiled for a generation's canonical GPUs (`compile_programs` with no topology) are
    XLA's untuned programs, which their file says (`AheadPrograms.tuned`), and a process whose own compile
    tunes refuses to load them (row 8422: untuned convolutions left cuDNN's runtime heuristic to pick a
    non-deterministic algorithm), where one compiled tuned, or one under the autotuner's level 0, loads.
    Over a node's own topology the compile takes the GPU kind's tuned choices or is refused, naming the
    program, where no run has tuned one yet. A program of this host's own target stands for the GPU
    host's here."""
    from jax.experimental import topologies

    from dew.nn.kernels.generation import compile_only_gpus
    from dew.training.ahead import AheadPrograms, BuildTarget, compile_programs

    square = jax.ShapeDtypeStruct((128, 128), jnp.float32)
    built = compile_programs({"dot": (lambda x, y: x @ y, [square, square])}, "sm90")
    assert not built.tuned
    here = jax.devices()[:1]
    with pytest.raises(ValueError):
        AheadPrograms(BuildTarget.of(here), {}, tuned=False, at_run=()).loaded()
    assert AheadPrograms(BuildTarget.of(here), {}, tuned=True, at_run=()).loaded() == {}
    serialized = tmp_path / "topology.bin"
    serialized.write_bytes(topologies.TopologyDescription(list(compile_only_gpus("sm90", 1))).serialize())
    with pytest.raises(ValueError):
        compile_programs({"dot": (lambda x, y: x @ y, [square, square])}, "sm90", topology=serialized)
    monkeypatch.setenv("XLA_FLAGS", f"{os.environ.get('XLA_FLAGS', '')} --xla_gpu_autotune_level=0")
    assert AheadPrograms(BuildTarget.of(here), {}, tuned=False, at_run=()).loaded() == {}


def test_a_devices_own_target_is_its_kinds_with_the_facts_it_reports():
    """A device's own target, what a program compiled ahead is checked against, is the one `gpu_target`
    resolves for it unassumed: an H100 PCIe its own kind and peaks with the multiprocessors it reports, and an
    sm90 kind the table names no SKU of its own kind, its peaks unknown, never the generation's canonical
    device's."""
    from dew.nn.kernels.generation import device_target

    # Hashable, as a JAX device is: a device's target is resolved once a device and generation.
    @dataclasses.dataclass(frozen=True)
    class Device:
        device_kind: str
        core_count: int
        platform: str = "gpu"
        compute_capability: str = "9.0"
        local_hardware_id: int = 0

    def device(kind, cores):
        return Device(kind, cores)

    pcie = device_target(device("NVIDIA H100 PCIe", 114))
    unnamed = device_target(device("NVIDIA H20", 78))
    assert (pcie.generation, pcie.device_kind, pcie.bf16_peak, pcie.multiprocessors) == (
        "sm90", "NVIDIA H100 PCIe", 756e12, 114)
    assert (unnamed.generation, unnamed.device_kind, unnamed.bf16_peak, unnamed.multiprocessors) == (
        "sm90", "NVIDIA H20", None, 78)


def test_a_runs_compilers_dumps_hold_what_it_compiled_and_nothing_where_it_compiled_nothing(tmp_path,
                                                                                           monkeypatch):
    """A run given `dumps` writes its compiles under that directory, a child process's too from another
    working directory, whatever the environment said of XLA's dumping (elsewhere, or filtered to no
    module), and nothing where it compiled nothing (`compiled_in`); a directory dumps did not make is
    refused, never read as empty: the gate of a run of programs compiled ahead (ledger 2716)."""
    import subprocess
    import sys
    from pathlib import Path

    from dew.training.ahead import compiled_in, dumps

    child = "import jax; jax.jit(lambda x: x + 1)(1.0)"
    elsewhere = f"[sys.executable, '-c', {child!r}], cwd='/', check=True"
    compiling = f"import subprocess, sys; subprocess.run({elsewhere})"
    monkeypatch.chdir(tmp_path)
    given = (("elsewhere", f"--xla_dump_to={tmp_path / 'other'}"),
             ("filtered", "--xla_dump_hlo_module_re=^never$"))
    for name, flags in given:
        environment = {**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": flags}
        subprocess.run([sys.executable, "-c", compiling], env=dumps(Path(name), environment), check=True)
        assert compiled_in(tmp_path / name), name
    computing = dumps(Path("computes"), {**os.environ, "JAX_PLATFORMS": "cpu"})
    subprocess.run([sys.executable, "-c", "import numpy; numpy.arange(8.0).sum()"], env=computing, check=True)
    assert compiled_in(tmp_path / "computes") == []
    with pytest.raises(FileNotFoundError):
        compiled_in(tmp_path / "never-made")
    # A dump directory that is a link elsewhere is refused at both ends, and one that takes no file at the
    # start.
    (tmp_path / "linked").mkdir()
    (tmp_path / "link-target").mkdir()
    (tmp_path / "linked" / "xla").symlink_to(tmp_path / "link-target")
    with pytest.raises(ValueError):
        dumps(tmp_path / "linked", {})
    (tmp_path / "linked" / "mosaic").mkdir()
    with pytest.raises(ValueError):
        compiled_in(tmp_path / "linked")
    (tmp_path / "locked" / "xla").mkdir(parents=True)
    (tmp_path / "locked" / "xla").chmod(0o500)
    with pytest.raises(PermissionError):
        dumps(tmp_path / "locked", {})


def test_a_flag_given_twice_is_read_by_its_order():
    """XLA reads a flag given twice by its last value, so the two orders of one pair are two targets
    (`computation_flags`, review)."""
    from dew.training.ahead import computation_flags

    first = "--xla_allow_excess_precision=false --xla_allow_excess_precision=true --xla_dump_to=/tmp/x"
    second = "--xla_allow_excess_precision=true --xla_allow_excess_precision=false"
    assert computation_flags(first) != computation_flags(second)
    excess = "--xla_allow_excess_precision"
    assert computation_flags(first) == (f"{excess}=false", f"{excess}=true")


@pytest.mark.parametrize(("generation", "capability"),
                         [("sm100", (10, 0)), ("sm90", (9, 0)), ("sm80", (8, 0))])
def test_cudnn_attentions_gates_read_the_capability_of_the_gpus_a_program_is_traced_for(generation,
                                                                                         capability):
    """The gates of cuDNN's attention (its head dims past 128, packed and latent forms and score functions
    from Hopper on, a 10.3's cuDNN version) read the capability of the GPUs a program is traced for: in this
    GPU-free process, those of compile-only GPUs (`compile_only_gpus`) whose mesh it is traced under
    (`jax.set_mesh`), as a step compiled ahead is, and none outside it, its backend the CPU. JAX read the
    process's backend alone, so a step compiled ahead for a B200 refused its d256 attention (row 5894)."""
    from jax._src.cudnn import fused_attention_stablehlo as cudnn

    from dew.training.ahead import compile_only_gpus

    mesh = jax.sharding.Mesh(np.asarray(compile_only_gpus(generation, 1)), ("data",))
    with jax.set_mesh(mesh):
        assert cudnn.check_compute_capability("9.0") == (capability >= (9, 0))
        assert cudnn.is_cuda_compute_capability_equal("10.0") == (capability == (10, 0))
    assert not cudnn.check_compute_capability("8.0")


@COMPILES_ON_CPU
def test_a_step_compiled_ahead_is_admitted_at_the_loading_runs_pool_share_not_the_compilers(monkeypatch):
    """A prepare whose own process holds a sliver of its GPU (a job packed beside others, its pool's
    fraction 0.08) admits the run it compiles for at the share that run holds (`AheadTarget.share`): Dew's
    own by default, the run's named one where it names another. It admitted an 80 GiB GPU's step against
    6.78 GiB, its own share of its own GPU, and refused a state that fits the run (row 7644). Targets of two
    shares made in one process each keep their own, built before either is read, and so does a direct
    `compile_only_gpus` (the library's path), where the last target's share was every alike GPU's and a
    direct one took the process's (train 1026's review, F1)."""
    from dew._devices import DEW_SHARE, FRACTIONS, device_memory
    from dew.training.ahead import compile_only_gpus
    from dew.training.targets import AheadTarget

    for name in FRACTIONS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(FRACTIONS[1], "0.08")
    memory = 80 * 2**30
    made = [(target.compile_only(), share) for target, share in (
        (AheadTarget("sm90", 2, capacity_gib=80), DEW_SHARE),
        (AheadTarget("sm90", 2, capacity_gib=80, share=0.5), 0.5))]
    made.append((compile_only_gpus("sm90", 2, memory), DEW_SHARE))
    for devices, share in made:
        assert {device_memory(device)["bytes_limit"] for device in devices} == {int(memory * share)}
    with pytest.raises(ValueError):
        AheadTarget("sm90", 2, capacity_gib=80, share=1.5)


@COMPILES_ON_CPU
def test_a_program_compiled_for_a_compile_only_gpu_is_refused_by_the_keeper_not_crashed_on():
    """A program a process compiles at Dew's compile point for a compile-only GPU (`gpus`), as a GPU cell's
    prepare does on its host's CPU, is no program of the process's default backend: the keeper
    (`keeping`) refuses it by name, its executable never loaded, where reading its loaded devices raised
    AttributeError and failed the A100 tier's recurrence cells (rows 7323, 7866, 7867)."""
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    from dew.nn.kernels.options import owned_program
    from dew.training.ahead import keeping

    placed = jax.ShapeDtypeStruct((3,), jnp.float32, sharding=NamedSharding(
        Mesh(np.asarray(gpus("sm80")), ("data",)), PartitionSpec()))
    with keeping() as programs:
        owned_program(lambda values: values * 2).executable(placed)
    assert programs.kept == {} and len(programs.refused) == 1


@pytest.mark.mesh(devices=2)
def test_programs_built_for_the_host_cpu_load_onto_it_whatever_the_process_lists_first(monkeypatch):
    """Programs compiled ahead for the host's CPU (`BuildTarget.cpu`) load onto the host's first CPU
    (`BuildTarget.own_devices`) in a process whose devices list another first, as a GPU's process lists its
    GPUs before the CPUs it holds beside them: loaded onto the process's first devices, a CPU's program went
    to them (row 8376's CPU-placed arms, `reference_error.ahead_arm(host=True)`)."""
    from dew._cache import uncached
    from dew.training.ahead import AheadPrograms, BuildTarget, Executable

    host = jax.devices("cpu")
    x = np.arange(4.0, dtype=np.float32)
    placed = jax.sharding.SingleDeviceSharding(host[0])
    # Compiled here, as every program compiled ahead is (`compile_programs`): one XLA loaded from the
    # persistent cache serializes without its kernels, and loading it fails ("Function ... not found").
    with uncached():
        doubled = jax.jit(lambda x: x * 2.0, out_shardings=placed).lower(
            jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=placed)).compile()
    programs = AheadPrograms(BuildTarget.of(host[:1]), {"doubled": Executable.of(doubled)}, tuned=True,
                             at_run=())
    listed = jax.devices
    # A process listing another device first, the host's first CPU among its CPUs.
    monkeypatch.setattr(jax, "devices", lambda backend=None: listed(backend) if backend else listed()[1:])
    ran = programs.loaded()["doubled"](x)
    assert ran.devices() == {host[0]} and np.asarray(ran).tolist() == [0.0, 2.0, 4.0, 6.0]
