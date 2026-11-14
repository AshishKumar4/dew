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
import os
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
from jax.extend import custom_call_targets, xla
from kernel_compile import gate_generations
from model_support import TINY_DECODER, memory_limited_to
from token_support import token_rows

from dew._devices import allocator_limit, device_memory
from dew._records import listed, record
from dew.nn.kernels.generation import capability
from dew.nn.sharding import Link
from dew.objectives.lm import LMObjective
from dew.optim import OptimConfig
from dew.registry import models
from dew.telemetry.compiles import Compiles
from dew.training import Layout, MeshSpec, Trainer

VOCAB, SEQ_LEN, BATCH = 96, 16, 8
TINY_SHARD = 256


def this_gpus(count: int = 1) -> tuple:
    """`count` compile-only GPUs of this GPU's own target, its generation, its multiprocessors (an H100
    PCIe's 114, not the canonical H100's 132) and the memory whose allocator's limit is its own, whose
    programs this GPU's loader takes (`BuildTarget.held`)."""
    from dew.nn.kernels.generation import compile_only_gpus
    from dew.training.ahead import BuildTarget

    own = BuildTarget.of(jax.devices()[:1])
    assert own.memory is not None
    return compile_only_gpus(own.generation, count, memory_limited_to(own.memory),
                             multiprocessors=own.multiprocessors)


def gpus(generation: str, count: int = 1, limit: int = 2**36) -> tuple:
    """`count` compile-only GPUs of `generation` whose stated memory gives their allocator `limit` bytes
    (`memory_limited_to`), 64 GiB unless a test asks less: XLA's own scheduling reads the memory a target
    states, which a GPU's of that size leaves to the step's own needs."""
    from dew.nn.kernels.generation import compile_only_gpus

    return compile_only_gpus(generation, count, memory_limited_to(limit))


def fine_tune(mesh, *, ahead=None, width: int | None = None, xla: bool = False, split=1, remat=None,
              inspection=None, tracker=None):
    fields = {**TINY_DECODER, **({} if width is None else {"emb_features": width}),
              **({"attention_impl": "xla"} if xla else {}), **({} if remat is None else {"remat": remat})}
    model = models.build("causal_transformer", vocab_size=VOCAB, **fields, num_kv_heads=2)
    return Trainer(LMObjective(model, SEQ_LEN), optax.adam(1e-3), key=jax.random.key(0), mesh=mesh,
                   layout=Layout(min_shard=TINY_SHARD), split=split, ahead=ahead, inspection=inspection,
                   tracker=tracker)


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


KEPT_MODULE = '''
import dataclasses
import os

import jax.numpy as jnp

from dew.nn.kernels.generation import traced_fact

OFFSET = 0


@dataclasses.dataclass(frozen=True)
class Scale:
    by: int


def scaled(scale, values):
    return jnp.tanh(values) * scale.by


def shifted(values):
    return values + OFFSET


def chosen(flag, values):
    return values + (1 if flag is None else 2)


def _by(values):
    return values * OFFSET


def helped(values):
    return _by(values)


def nested(values):
    return (lambda held: held - OFFSET)(values)


@dataclasses.dataclass(frozen=True)
class Factor:
    factor = 1

    def __call__(self, values):
        return values * self.factor


def applied(factor, values):
    return factor(values)


@traced_fact
def kept_fact():
    return os.environ.get("KEPT_FACT")
'''
"""The module the kept programs' functions and their static argument's class are defined in
(tests/kept_programs_worker.py), which each test writes beside its file: a user's own, outside any installed
distribution, with a setting its code changes at run time (`OFFSET`) and a fact of the process it declares
its traces read (`kept_fact`)."""


def kept_worker(directory, mode: str, scale: int = 2, offset: int = 1,
                environment=None) -> subprocess.CompletedProcess:
    """tests/kept_programs_worker.py in a fresh process: `mode` over the programs file in `directory` at
    `scale` and `offset`, `kept_module` read from `directory`, with `environment`'s variables beside this
    process's."""
    path = os.pathsep.join((str(directory), str(process_support.REPO_ROOT / "src"),
                            str(process_support.REPO_ROOT / "tests")))
    return subprocess.run(
        [sys.executable, str(process_support.REPO_ROOT / "tests" / "kept_programs_worker.py"), mode,
         str(directory / "programs.dew-aot"), str(scale), str(offset)],
        env={**os.environ, "PYTHONPATH": path, **(environment or {})}, capture_output=True, text=True,
        cwd=process_support.REPO_ROOT)


def kept_answer(run: subprocess.CompletedProcess) -> dict:
    assert run.returncode == 0, run.stderr[-3000:]
    return json.loads(run.stdout.splitlines()[-1])


def kept_in(directory, scale: int = 2, offset: int = 1) -> dict:
    """What a fresh process keeping its programs in `directory` at `scale` and `offset` answers and keeps."""
    directory.mkdir(exist_ok=True)
    (directory / "kept_module.py").write_text(KEPT_MODULE)
    return kept_answer(kept_worker(directory, "keep", scale, offset))


def test_a_program_one_process_kept_runs_in_another_that_traces_nothing(tmp_path):
    """A process keeps the executable of each program it compiles at Dew's compile point
    (`dew.training.ahead.keeping`) and saves them; a later process of the same target, code and settings,
    handed the file (`dew.training.ahead.kept`), answers every call exactly as the first did and needs no
    trace and no compile of any of them. Each program is kept for its own call alone: a static argument
    that is None and one that is an empty tuple, and two functions written on one line that differ in a
    constant, each answer as their own. A call that differs in what its trace reads is another call,
    which the handed process compiles and answers as a process compiling it does: at another static
    argument, and at another value of a global its function names, a function of the user's it calls
    names or code nested in it names, or of a class attribute a static argument's `__call__` reads, each
    of which its module's code changed at run time. With `verify` each handed call also lowers its
    own program and finds the kept one's. A CPU's file states its host's processor; a function that
    reads an array outside its arguments, and a
    call made inside one of Dew's own contexts, have no identity a file can state and are named as not
    kept (row 5265: a warm scoring server paid its programs' traces)."""
    saved = kept_in(tmp_path / "kept")
    answers = saved["answers"]
    # Every answered call kept, and the calls of no identity the worker makes besides, those alone, named
    # as not.
    assert saved["kept"] == len(answers)
    refused = sorted(name.rsplit(" (", 1)[0] for name in saved["refused"])
    assert refused == saved["unidentified"], saved["reasons"]
    assert (saved["cpu"] is not None) == (jax.default_backend() == "cpu")
    assert answers["none"] != answers["empty"] and answers["one"] != answers["two"]
    assert kept_answer(kept_worker(tmp_path / "kept", "take")) == {"answers": answers, "needed": []}
    assert kept_answer(kept_worker(tmp_path / "kept", "verify"))["answers"] == answers
    other = kept_answer(kept_worker(tmp_path / "kept", "take", scale=3, offset=4))
    assert other["needed"] == ["applied", "helped", "nested", "scaled", "shifted"]
    compiled = kept_in(tmp_path / "other", scale=3, offset=4)["answers"]
    assert other["answers"] == compiled
    differing = {name for name in answers if answers[name] != compiled[name]}
    assert differing == {"applied", "helped", "nested", "scaled", "shifted"}


@pytest.mark.parametrize(("changed", "process", "target"), [
    ("source", ["source kept_module"], []), ("environment", ["environment DEW_KEPT_PROGRAMS_TEST"], []),
    ("fact", ["fact kept_module.kept_fact"], []), ("setting", ["setting jax_default_prng_impl"], []),
    ("flags", [], ["flags"]), ("target", [], ["host"])])
def test_programs_kept_under_other_code_settings_or_target_are_refused_by_name(tmp_path, changed, process,
                                                                                target):
    """A file of kept programs is refused before any call where the process handed it is not the one that
    kept it, and its records name what differs (`Process.differences`, `BuildTarget.differences`): a
    module's source edited since (the functions' own, a user's module no installed distribution
    versions), an environment variable of Dew's, a fact of the process its code declares its traces read
    (`traced_fact`), a setting of JAX's that a trace reads and JAX's own trace context does not hold (the
    default PRNG implementation), an XLA flag that decides what a program computes (deterministic ops,
    which Dew's kernels' routes read), or a target that is not this host's. A kept executable is run for
    nothing but the call, code and settings it was compiled from."""
    from dew.training.ahead import AheadPrograms

    kept_in(tmp_path)
    flags = f"{os.environ.get('XLA_FLAGS', '')} --xla_gpu_deterministic_ops=true".strip()
    environment = {"environment": {"DEW_KEPT_PROGRAMS_TEST": "1"}, "fact": {"KEPT_FACT": "another"},
                   "setting": {"JAX_DEFAULT_PRNG_IMPL": "rbg"}, "flags": {"XLA_FLAGS": flags}}.get(changed)
    if changed == "source":
        (tmp_path / "kept_module.py").write_text(KEPT_MODULE.replace("jnp.tanh(values)", "jnp.sin(values)"))
    elif changed == "target":
        path = tmp_path / "programs.dew-aot"
        programs = AheadPrograms.load(path)
        dataclasses.replace(programs, target=dataclasses.replace(programs.target, host="another")).save(path)
    assert kept_worker(tmp_path, "take", environment=environment).returncode != 0
    assert kept_answer(kept_worker(tmp_path, "differences", environment=environment)) == {
        "process": process, "target": target}


CHECKED_MODULE = '''
import jax.numpy as jnp
from jax.experimental import checkify


def _doubled(values):
    checkify.check(jnp.all(values > 0), "the values must be positive")
    return values * 2


def checked(values):
    return checkify.checkify(_doubled)(values)
'''
"""A program whose check a run keeps (`checked`): its error tree holds the check's traceback."""

CHECKED_RUN = """
import json, sys
import jax.numpy as jnp
from jax.experimental import checkify
import checked_module
from dew.nn.kernels.options import owned_program
from dew.training.ahead import keeping, kept

mode, path = sys.argv[1], sys.argv[2]
program = owned_program(checked_module.checked)


def refused():
    error, _ = program(-jnp.ones(3))
    try:
        error.throw()
    except checkify.JaxRuntimeError as raised:
        return [type(raised).__name__, str(raised)]
    return None


if mode == "keep":
    with keeping() as programs:
        program(jnp.ones(3))
    programs.save(path)
    print(json.dumps({"refused": refused(), "kept": len(programs.kept)}))
else:
    with kept(path):
        print(json.dumps({"refused": refused()}))
"""
"""A process that keeps a checked program, or is handed it, and reports what its failed check raises."""


def test_a_kept_programs_failed_check_raises_where_handed_as_where_kept(tmp_path):
    """A checked program kept by one process (its error tree's tracebacks held as their frames,
    `HeldTraceback`) and handed to another raises its failed check there as the keeping process does, the
    check's own error and message, not an error of the held traceback's."""
    import process_support

    (tmp_path / "checked_module.py").write_text(CHECKED_MODULE)
    path = str(tmp_path / "checked.dew-aot")
    # The module beside the file, on the path as a user's own.
    found = {"PYTHONPATH": os.pathsep.join((str(tmp_path), str(process_support.REPO_ROOT / "src"),
                                            str(process_support.REPO_ROOT / "tests")))}
    keep = process_support.fresh(CHECKED_RUN, "keep", path, variables=found)
    take = process_support.fresh(CHECKED_RUN, "take", path, variables=found)
    assert keep["kept"] == 1 and keep["refused"] is not None
    assert take["refused"] == keep["refused"]


def test_a_kept_program_that_is_not_its_calls_is_found_by_verifying(tmp_path):
    """`kept(path, verify=True)` lowers each handed call's own program and holds it to the kept one's
    digest: a file whose program for a call is another program (here its recorded digest changed) is
    refused at that call (`NotTheKeptProgram`), where the same file handed without verifying runs. And a
    file of programs compiled ahead by name, none kept by its call, is refused as it is handed."""
    from dew.training.ahead import AheadProgram, AheadPrograms, kept

    answers = kept_in(tmp_path)["answers"]
    path = tmp_path / "programs.dew-aot"
    programs = AheadPrograms.load(path)
    other = {digest: dataclasses.replace(each, program=AheadProgram("0" * 64, each.program.executable))
             for digest, each in programs.kept.items()}
    dataclasses.replace(programs, kept=other).save(path)
    assert kept_answer(kept_worker(tmp_path, "take")) == {"answers": answers, "needed": []}
    assert kept_answer(kept_worker(tmp_path, "verify")) == {"refused": "NotTheKeptProgram"}
    AheadPrograms(programs.target, {}).save(path)
    with pytest.raises(ValueError), kept(path):
        pass


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

    def accumulating(**mesh):
        model = models.build("causal_transformer", vocab_size=VOCAB, **TINY_DECODER, num_kv_heads=2)
        return Trainer(LMObjective(model, SEQ_LEN), optax.adam(1e-3), key=jax.random.key(0), **mesh,
                       layout=Layout(min_shard=TINY_SHARD), accumulation=2,
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

    def run(**options):
        model = models.build("causal_transformer", vocab_size=VOCAB, **TINY_DECODER, num_kv_heads=2)
        return Trainer(LMObjective(model, SEQ_LEN), optax.adam(1e-3), key=jax.random.key(0),
                       layout=Layout(min_shard=TINY_SHARD), checkpoints=Checkpoints(str(tmp_path / "run")),
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
                      layout=Layout(min_shard=TINY_SHARD))
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
    loader = compile(phases((corpus.batch(12), 2), (corpus.batch(8), None)), read="memory")
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
    # The plan was made against the allocator's limit the GPU's stated memory gives (`allocator_limit`).
    assert ahead.capacity == ahead.built.memory == 2**30
    stated = device_memory(trainer.device_mesh.devices.flat[0], ahead.argument_bytes)
    assert stated["pool_bytes"] == ahead.capacity
    assert stated["largest_free_block_bytes"] == ahead.capacity - ahead.argument_bytes
    seconds = probe["compile_seconds"]
    assert isinstance(seconds, float) and ahead.planned_seconds is not None
    assert ahead.planned_seconds >= seconds > 0
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


NEEDING_MICROBATCHES = 600_000
"""Bytes a compile-only GPU states for `fine_tune`'s step to need several microbatches: four, with a rung
above it at two (a sweep at base/kernels e4d702a55 with the split computed from the compile that missed,
2cb8d0bb4: from 500 kB to 600 kB the plan takes four or eight, at 650 kB and 700 kB two, at 800 kB one)."""


@pytest.fixture(scope="module")
def admitted(tmp_path_factory):
    """`fine_tune`'s step compiled ahead on GPUs of `NEEDING_MICROBATCHES`, its plan kept beside the
    compilation cache as every search keeps its own (`DEW_RUNG_RECORDS`): the records' directory, the
    trainer and its `AheadOfTime`, admitted once for the module."""
    from dew.training.ahead import compile_ahead

    records = tmp_path_factory.mktemp("rungs")
    with pytest.MonkeyPatch.context() as patch:
        # The module's own data and Hub caches, as each test's (`own_settings`), its rung records `records`.
        lane_environment.per_test_settings(patch, tmp_path_factory.mktemp)
        patch.setenv("DEW_RUNG_RECORDS", str(records))
        trainer = fine_tune(MeshSpec().build(gpus("sm90", limit=NEEDING_MICROBATCHES)), split=None)
        ahead = compile_ahead(trainer, token_rows(VOCAB, BATCH, SEQ_LEN + 1))
    return records, trainer, ahead


def test_a_program_compiled_over_a_hosts_serialized_topology_keys_the_cache_as_over_its_own_gpus(tmp_path):
    """A host's GPUs, serialized as their topology (`get_attached_topology(...).serialize()`, here of
    compile-only H100s at a stated 40 GiB) and read back elsewhere (`host_gpus`), are the GPUs XLA compiles
    for: a program compiled over them keys the persistent cache with the host's own platform version and
    topology fingerprint, as over the devices serialized, where a canonical target of the generation
    (`compile_only_gpus`, its memory unstated) keys another entry."""
    from jax._src import cache_key, compiler
    from jax.experimental import topologies

    from dew.nn.kernels.generation import compile_only_gpus
    from dew.training.ahead import host_gpus

    host = compile_only_gpus("sm90", 2, 40 * 2**30)
    serialized = tmp_path / "topology.bin"
    serialized.write_bytes(topologies.TopologyDescription(list(host)).serialize())

    def key(devices) -> str:
        lowered = jax.jit(lambda x: x * 2 + 1).trace(jax.ShapeDtypeStruct(
            (8, 128), jnp.float32, sharding=jax.sharding.SingleDeviceSharding(devices[0]))).lower()
        backend = devices[0].client
        options = compiler.get_compile_options(num_replicas=1, num_partitions=1,
                                               device_assignment=[[devices[0]]], backend=backend)
        return cache_key.get(lowered.compiler_ir("stablehlo"), np.array(devices[:1]), options, backend)

    read = host_gpus("sm90", serialized, 2)
    assert [device.device_kind for device in read] == [device.device_kind for device in host]
    assert key(read) == key(host)
    assert key(read) != key(compile_only_gpus("sm90", 2))
    # The topology states its GPUs' memory, which the devices read back carry as the ones serialized do.
    assert [device_memory(device)["bytes_limit"] for device in read] == [allocator_limit(40 * 2**30)] * 2


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
    run takes (`pool_record`)."""
    from dew._devices import pool_share
    from dew.training.memory import recompute_record

    _, trainer, ahead = admitted
    plan = record(ahead.plan, "plan")
    probes = listed(plan["probes"], "probes", read=record, what="probes")
    assert plan["split"] > 1
    at_its_plan(plan, probes)
    assert trainer.plan == ahead.plan and recompute_record(trainer.objective) == plan["remat"]
    pool = record(plan["pool"], "pool")
    assert pool["source"] == "compile_only" and pool["share"] == pool_share()
    assert pool["limit"] == NEEDING_MICROBATCHES


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

    records, _, earlier = admitted
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(records))
    trainer = fine_tune(MeshSpec().build(gpus("sm90", limit=NEEDING_MICROBATCHES)), split=None)
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


@pytest.mark.parametrize("capacity", [2**30, 650_000], ids=["one microbatch", "several"])
def test_a_frontier_at_the_ladders_top_lists_its_offloaded_sibling_at_its_last_fit(tmp_path, monkeypatch,
                                                                                    capacity):
    """A step whose plan is the remat ladder's top ('full') lists, past it, the sibling that keeps what
    'minimal' saves in pinned host memory in place of recomputing it ('minimal_offloaded'): as many device
    bytes as 'full', so no fit rung, but another speed, which the GPUs' timing chooses (row 1932). It is
    compiled at the last configuration at 'full' that fit, its head policy put back, whatever head policies
    the frontier then tried at fewer microbatches; and the run, which times nothing, is left at its plan.
    At 650 kB the plan at 'full' takes two microbatches and the sibling fits there (a sweep at base/kernels
    e4d702a55 with 2cb8d0bb4: from 500 kB to 600 kB the sibling fits at no plan's split on a device with no
    host memory of its own, and from 700 kB the plan takes one)."""
    from dew.nn.backbones.decoder_block import REMAT_POLICIES, remat_record
    from dew.training.ahead import compile_ahead
    from dew.training.memory import head_fused_of, head_tile_of, recompute_record

    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "rungs"))
    trainer = fine_tune(MeshSpec().build(gpus("sm90", limit=capacity)), split=None, remat="full")
    ahead = compile_ahead(trainer, token_rows(VOCAB, BATCH, SEQ_LEN + 1), frontier=True)
    plan = record(ahead.plan, "plan")
    frontier = listed(plan["frontier"], "frontier", read=record, what="frontier")
    full, offloaded = (remat_record(REMAT_POLICIES[name]) for name in ("full", "minimal_offloaded"))
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
    from dew.training.planning import FitSearch

    monkeypatch.setenv("XLA_FLAGS", without_flags(os.environ.get("XLA_FLAGS", ""),
                                                  "--xla_gpu_autotune_level"))
    deviceless = MeshSpec().build(compile_only_gpus("sm90", 1))
    running = MeshSpec().build(jax.devices()[:1])

    def lowered(mesh):
        placed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        shape = jax.ShapeDtypeStruct((64, 64), jnp.float32, sharding=placed)
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
    # A step whose options are its towers' own compiles as the half rung of XLA's defaults does, one
    # program (row 3506).
    objective = fine_tune(deviceless, split=None).objective
    search = FitSearch(lower=None, options=lambda candidate: compiler_options(*objective.towers()) or None,
                       objective=objective, mesh=deviceless, xla_defaults=False, narrow=False, base=None,
                       resumed=None)
    assert search.compiled_options(None) == search.required == (compiler_options(*objective.towers()) or None)
    assert dataclasses.replace(search, xla_defaults=True).compiled_options(None) == search.required
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
                           "--trainer.steps", "2", "--trainer.compile-ahead.generation", "sm80",
                           "--trainer.compile-ahead.devices", "1",
                           "--trainer.compile-ahead.path", str(tmp_path / "step.dew-aot")],
                          default=lm(tmp_path, drawn=False))
    with Compiles("step") as compiles, pytest.raises(ValueError):
        run.run()
    assert compiles.count == 0 and not (tmp_path / "step.dew-aot").exists()
    assert stated_limit(gpus("sm80", limit=2**34)) == 2**34


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
    beside them, so an offline plan could refuse a step that fits)."""
    from dew.distributed.batches import batch_shardings, prefetched_bytes
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
    assert probe["fits"] is True
    assert probe["headroom"] == (capacity - ahead.argument_bytes - (ahead.output_bytes - ahead.alias_bytes)
                                 - prefetched - temporaries)


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
import sys
sys.path[:0] = [f"{sys.argv[1]}/src", f"{sys.argv[1]}/tests"]
import jax
import optax
from token_support import token_rows

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.mixers.gated_delta_net import GatedDeltaNetMixer
from dew.objectives.lm import LMObjective
from dew.training import Layout, MeshSpec, Trainer
from dew.nn.kernels.generation import compile_only_gpus
from dew.training.ahead import compile_ahead

mixer = GatedDeltaNetMixer(linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=64,
                           linear_value_head_dim=64)
model = CausalTransformer(vocab_size=64, emb_features=128, num_layers=1, num_heads=2, head_dim=64,
                          mlp_features=256, max_seq_len=128, dtype="bfloat16", mixer=mixer)
trainer = Trainer(LMObjective(model, 128), optax.adam(1e-3), key=jax.random.key(0),
                  mesh=MeshSpec().build(compile_only_gpus("sm80", 1, 2**35)), layout=Layout(min_shard=256))
compile_ahead(trainer, token_rows(64, 2, 129))
"""
"""A bf16 gated-delta decoder's step compiled ahead for a compile-only A100, in a process of its own: its
checkout is its argument."""


def test_a_gated_delta_steps_mosaic_kernels_compile_ahead_for_the_gpu_it_targets():
    """A bf16 gated-delta decoder's step compiled ahead for a compile-only A100 (`compile_only_gpus("sm80")`)
    lowers its Mosaic GPU kernels for the architecture its trace's abstract device states, the A100's 8.0
    (row 3885): a mesh's abstract device stated none, so Mosaic lowered for an H100's, and its
    `nvvm.elect.sync` failed sm80's LLVM instruction selection and aborted the process, every A100 prepare
    of a model with a Mosaic route among them. The compile runs in a process of its own, as Mosaic compiles
    for one capability a process (`kernel_compile.compile_mosaic_kernels`)."""
    import subprocess
    import sys
    from pathlib import Path

    root = str(Path(__file__).resolve().parents[1])
    # Its compile took 45 to 75 s on armada; past ten minutes it is stuck, not slow.
    ran = subprocess.run([sys.executable, "-c", GATED_DELTA_AHEAD, root], capture_output=True, text=True,
                         timeout=600, check=False)
    assert ran.returncode == 0, ran.stderr[-3000:]


EVERY_TEMPLATE_AHEAD = """
import json, sys
sys.path[:0] = [f"{sys.argv[1]}/src", f"{sys.argv[1]}/tests"]
import jax
import optax
from token_support import token_rows

from dew.nn.kernels.generation import routes
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.training import Layout, MeshSpec, Trainer
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
mesh = MeshSpec(fsdp=2).build(compile_only_gpus(sys.argv[2], 2, 2**36))
trainer = Trainer(LMObjective(model, 512), optax.adam(1e-3), key=jax.random.key(0), mesh=mesh,
                  layout=Layout(min_shard=256))
with routes() as taken:
    ahead = compile_ahead(trainer, token_rows(1024, 8, 513))
print(json.dumps({"kernels": sorted({(route.choice, route.kernel) for route in taken})}))
"""
"""A hybrid MoE decoder whose step runs every template, compiled ahead for two compile-only GPUs of a
generation, in a process of its own: its checkout and the generation are its arguments; it prints every
kernel its routes took."""


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
    import subprocess
    import sys
    from pathlib import Path

    root = str(Path(__file__).resolve().parents[1])
    ran = subprocess.run([sys.executable, "-c", EVERY_TEMPLATE_AHEAD, root, generation], capture_output=True,
                         text=True, timeout=1500, check=False)
    assert ran.returncode == 0, ran.stderr[-3000:]
    found = json.loads(ran.stdout.strip().splitlines()[-1])
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
                      layout=Layout(min_shard=TINY_SHARD))
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
    path.write_bytes(pickle.dumps(AheadPrograms(target, {"step": ("program", b"executable", ())})))
    with pytest.raises(ValueError):
        AheadPrograms.load(path)


def test_compiling_ahead_takes_a_mesh_of_compile_only_gpus():
    """A trainer whose mesh is this host's own devices runs its step here; compiling it ahead is
    refused, as XLA's CPU executable, loaded again, finds none of its fused functions."""
    from dew.training.ahead import compile_ahead

    with pytest.raises(ValueError):
        compile_ahead(fine_tune(MeshSpec(fsdp=8).build()), token_rows(VOCAB, BATCH, SEQ_LEN + 1))


def test_a_step_compiled_ahead_is_one_program_and_a_run_of_another_refuses_it(tmp_path):
    """Two trainers of one config compile ahead one program, the digest a run lowers its first step
    to (`AheadOfTime.program`); a wider model's is another. A run handed another program's step
    refuses it before it trains, and refuses it again on its next fit, rather than compiling its own
    (row 1915)."""
    from dew.nn.kernels.generation import assumed
    from dew.training.ahead import AheadOfTime, compile_ahead

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)

    def ahead(width=None):
        return compile_ahead(fine_tune(MeshSpec().build(gpus("sm80")), width=width), batch)

    first, again, wide = ahead(), ahead(), ahead(width=48)
    assert first.program == again.program != wide.program
    wide.save(tmp_path / "wide.dew-aot")
    run = fine_tune(MeshSpec().build(jax.devices()[:1]), ahead=AheadOfTime.load(tmp_path / "wide.dew-aot"))
    # On the GPU it was compiled for, which a CPU is not, its program is what refuses it (train 332's review).
    with assumed(wide.target):
        for _ in range(2):
            with pytest.raises(ValueError):
                run.fit(batch, steps=1, log_every=1)


def test_a_configured_run_compiles_its_step_ahead_for_gpus_it_lacks_and_trains_nothing(tmp_path):
    """A run parsed with `--trainer.compile-ahead.generation`, `.devices` and `.path` (`RunConfig.cli`,
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
                           "--trainer.steps", "2", "--trainer.compile-ahead.generation", "sm80",
                           "--trainer.compile-ahead.devices", "1", "--trainer.compile-ahead.path", str(path),
                           "--trainer.compile-ahead.capacity-gib", "1"],
                          default=default)
    with pytest.raises(CompiledAhead) as ended:
        run.run()
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

    from dew.training.ahead import AheadOfTime, AheadTarget, CompiledAhead
    from dew.training.budget import Pause

    runs, local, path = tmp_path / "runs", tmp_path / "local", tmp_path / "step.dew-aot"
    default = lm(tmp_path, drawn=False)
    target = AheadTarget(generation="sm80", devices=1, path=str(path), capacity_gib=1)
    run = dataclasses.replace(default, trainer=dataclasses.replace(
        default.trainer, name="ahead", checkpoint_dir=str(runs), local_directory=str(local), local_every=2,
        steps=4, stop=Pause(0.5), compile_ahead=target))
    with pytest.raises(CompiledAhead) as ended:
        run.run()
    assert ended.value.code == 0 and AheadOfTime.load(path).plan is not None
    assert not runs.exists() and not local.exists()


def test_a_configured_run_compiles_ahead_for_the_multiprocessors_it_states_and_loads_only_there(
        tmp_path, monkeypatch):
    """`--trainer.compile-ahead.multiprocessors` states the GPU a step is compiled ahead for within its
    generation (an H100 PCIe's 114 against the canonical H100's 132, rows 3606 and 3669): XLA compiles the
    step and its sampled step over compile-only GPUs of 114 (as they state), and the saved build target and
    the facts its launches derived for are that GPU's, so a GPU of 114 runs it and the canonical one is
    refused; the rung its plan recorded is that GPU's, which a second build for it reuses, apart from the
    canonical build's beside it in one record directory. Its runs score no split midway, whose programs a
    build compiles too (row 4752), so its three builds are their steps' alone (row 3981)."""
    import dataclasses

    from run_kinds import lm

    from dew.config.lm import LMRunConfig
    from dew.nn.kernels.generation import gpu_target
    from dew.training.ahead import AheadOfTime, CompiledAhead

    records = tmp_path / "rungs"
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(records))

    builds = iter(range(3))

    def compiled(*stated: str) -> AheadOfTime:
        root = tmp_path / f"run{next(builds)}"
        root.mkdir()
        path = root / "step.dew-aot"
        run = LMRunConfig.cli(["--trainer.name", "ahead", "--trainer.checkpoint-dir", str(root / "runs"),
                               "--trainer.steps", "2", "--trainer.compile-ahead.generation", "sm90",
                               "--trainer.compile-ahead.devices", "1",
                               "--trainer.compile-ahead.path", str(path),
                               "--trainer.compile-ahead.capacity-gib", "0.0007",
                               "--trainer.inspection.every", "1", "--trainer.eval-every", "None",
                               "--trainer.checkpoint-every", "None", *stated],
                              default=lm(root, drawn=False))
        with pytest.raises(CompiledAhead):
            run.run()
        return AheadOfTime.load(path)

    pcie = compiled("--trainer.compile-ahead.multiprocessors", "114")
    again = compiled("--trainer.compile-ahead.multiprocessors", "114")
    canonical = compiled()
    assert (pcie.built.multiprocessors, pcie.target.multiprocessors) == (114, 114)
    assert pcie.built.memory == pcie.target.memory == allocator_limit(int(0.0007 * 2**30))
    assert pcie.sampled is not None and again.built == pcie.built
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
    """`--trainer.compile-ahead.frontier` (`RunConfig.cli`) asks the step compiled ahead for its frontier,
    which its saved plan records (rows 3108, 3147): on GPUs of 0.0007 GiB, where the plan needs two
    microbatches, a rung above it that fits at one."""
    from run_kinds import lm

    from dew.config.lm import LMRunConfig
    from dew.training.ahead import AheadOfTime, CompiledAhead

    runs, path = tmp_path / "runs", tmp_path / "step.dew-aot"
    run = LMRunConfig.cli(["--trainer.name", "ahead", "--trainer.checkpoint-dir", str(runs),
                           "--trainer.steps", "2", "--trainer.compile-ahead.generation", "sm80",
                           "--trainer.compile-ahead.devices", "1", "--trainer.compile-ahead.path", str(path),
                           "--trainer.compile-ahead.capacity-gib", "0.0007",
                           "--trainer.compile-ahead.frontier"],
                          default=lm(tmp_path, drawn=False))
    with pytest.raises(CompiledAhead) as ended:
        run.run()
    assert ended.value.code == 0
    plan = record(AheadOfTime.load(path).plan, "plan")
    frontier = listed(plan["frontier"], "frontier", read=record, what="frontier")
    assert plan["split"] == 2 and frontier
    assert all(entry["fits"] is True and entry["split"] < plan["split"] for entry in frontier)


def test_a_configured_run_compiles_its_step_ahead_for_a_tpu_topology_with_its_stated_links(tmp_path):
    """`--trainer.compile-ahead.generation` names a TPU topology as well as a CUDA generation: the run
    compiles its first step for that topology's compile-only devices here, through the plan a run takes,
    and writes it as any step compiled ahead (ledger 2551). The links it compiles with are the run's own
    (`trainer.links`), which the step records (ledger 2552)."""
    from run_kinds import lm

    from dew.config.lm import LMRunConfig
    from dew.training.ahead import AheadOfTime, CompiledAhead

    runs, path = tmp_path / "runs", tmp_path / "step.dew-aot"
    default = lm(tmp_path, drawn=False)
    assert isinstance(default, LMRunConfig)
    links = {"tensor": {"bytes_per_second": 4.5e11, "flops_per_second": 9.9e14}}
    run = LMRunConfig.cli(["--trainer.name", "ahead", "--trainer.checkpoint-dir", str(runs),
                           "--trainer.steps", "2", "--trainer.compile-ahead.generation", "v5e:2x2",
                           "--trainer.compile-ahead.devices", "4", "--trainer.compile-ahead.path", str(path),
                           "--trainer.mesh.tensor", "2", "--set", f"trainer.links={json.dumps(links)}"],
                          default=default)
    with pytest.raises(CompiledAhead) as ended:
        run.run()
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
                                 "--trainer.checkpoint-every", "3", "--optim.learning-rate",
                                 '{"class": "linear", "fields": {"peak_value": 0.001}}',
                                 "--trainer.compile-ahead.generation", "sm80",
                                 "--trainer.compile-ahead.devices", "1",
                                 "--trainer.compile-ahead.path", str(path),
                                 "--trainer.compile-ahead.capacity-gib", "4"],
                                default=default)
    with pytest.raises(CompiledAhead) as ended:
        run.run()
    assert ended.value.code == 0
    assert AheadOfTime.load(path).peak_bytes > 0


CLOCKED = ["--trainer.examples", "48", "--trainer.eval-every", "3", "--trainer.checkpoint-every", "3",
           "--trainer.multi-host", "False", "--trainer.compilation-cache-dir", "None",
           "--trainer.cooldowns.at", "50%", "--trainer.cooldowns.length", "25%",
           "--optim.learning-rate", '{"class": "linear", "fields": {"peak_value": 0.001}}']
"""The clocked decision run's flags (the clocked GPU cell's): its tickets stream, of no record count, so its
evaluations and checkpoints are counted in steps (ledger 3153)."""


def test_a_decision_runs_step_compiled_ahead_is_the_program_its_run_lowers_as_fit_hands_its_step(tmp_path):
    """The clocked GPU cell's property, read where a host has no GPU (ledger 3153): the decision run compiled
    ahead for eight sm90 GPUs (`--trainer.compile-ahead`) and the same run lowered for them as `fit` hands it
    its step, its batch placed on the devices (`--trainer.lower-for`), are one program, so a run handed the
    step it compiled ahead takes it. A step lowered over the host batch the loader gave and over that batch
    placed came to two module texts, a helper call's function told apart by its arguments' shardings."""
    from run_kinds import decision

    from dew.decision.config import DecisionRunConfig
    from dew.nn.kernels.options import program_digest
    from dew.training._lowering import Lowered
    from dew.training.ahead import AheadOfTime, CompiledAhead

    path = tmp_path / "ahead.dew-aot"
    stated = ["--trainer.checkpoint-dir", str(tmp_path / "runs"), *CLOCKED]
    target = ["--trainer.compile-ahead.generation", "sm90", "--trainer.compile-ahead.devices", "8",
              "--trainer.compile-ahead.path", str(path), "--trainer.compile-ahead.capacity-gib", "80"]
    ahead = DecisionRunConfig.cli(["--trainer.name", "ahead", *stated, *target],
                                  default=decision(tmp_path, drawn=False))
    with pytest.raises(CompiledAhead):
        ahead.run()
    lowering = DecisionRunConfig.cli(["--trainer.name", "lowered", *stated, "--trainer.lower-for", "sm90"],
                                     default=decision(tmp_path, drawn=False))
    with pytest.raises(Lowered) as lowered:
        lowering.run()
    assert program_digest(lowered.value.lowered) == AheadOfTime.load(path).program


def a_clocked_run_with_a_cooldown_trains_from_its_step_compiled_ahead_compiling_no_other(tmp_path):
    """A GPU flow's cell (tests/test_gpu_training.py; it trains from a step compiled ahead on this host's own
    GPU): a decision run on the examples clock under a schedule over the run, with a cooldown, compiled
    ahead through its command line for this GPU's generation (`--trainer.compile-ahead`) and then handed
    the step it saved (`--trainer.ahead`), trains from that step through its cooldown's branch, its last
    window's too, and compiles no step program of its own: each batch it steps closes its window as the
    one it compiled ahead over did (ledgers 2434, 2527)."""
    from run_kinds import decision

    from dew.decision.config import DecisionRunConfig
    from dew.training.ahead import BuildTarget, CompiledAhead

    runs, path = tmp_path / "runs", tmp_path / "step.dew-aot"
    default = decision(tmp_path, drawn=False)
    stated = ["--trainer.checkpoint-dir", str(runs), *CLOCKED]
    # This GPU's own target, its multiprocessors as it counts them (an H100 PCIe's 114).
    own = BuildTarget.of(jax.devices()[:1])
    assert own.memory is not None
    ahead = ["--trainer.compile-ahead.generation", own.generation,
             "--trainer.compile-ahead.multiprocessors", str(own.multiprocessors),
             "--trainer.compile-ahead.capacity-gib", str(memory_limited_to(own.memory) / 2**30),
             "--trainer.compile-ahead.devices", "1", "--trainer.compile-ahead.path", str(path)]
    compiling = DecisionRunConfig.cli(["--trainer.name", "ahead", *stated, *ahead], default=default)
    with pytest.raises(CompiledAhead):
        compiling.run()
    handed = DecisionRunConfig.cli(["--trainer.name", "handed", *stated, "--trainer.ahead", str(path)],
                                   default=default)
    with Compiles("step") as compiles:
        state = handed.run()
    assert compiles.count == 0 and int(state.updates) > 0


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
        trainer = Trainer(LMObjective(model, SEQ_LEN), OptimConfig(optimizer="adamw", learning_rate=1e-3),
                          key=jax.random.key(0), mesh=MeshSpec().build(gpus(generation)),
                          layout=Layout(min_shard=TINY_SHARD), split=1, accumulation=accumulation)
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
    """A step compiled ahead for one compile-only GPU of this GPU's own target (`this_gpus`), saved and
    handed to the run (`Trainer(ahead=...)`), is the program the run lowers its first step to, and the run
    trains from it, its plan's first compile the step compiled ahead: its steps advance and its
    parameters move (row 1915). A wider model's step for this GPU is another program, refused. The
    model's attention takes XLA's path, which a step compiled with no GPU attached traces too."""
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
    from dew.training.ahead import AheadTarget

    data = Loader(train=lambda partition: iter([{"x": np.ones((BATCH, 8), np.float32),
                                                  "y": np.zeros((BATCH, 4), np.float32)}] * 4),
                  val=None, batch=BATCH, records=None)
    trainer = Trainer(Supervised(nn.Dense(4), lambda outputs, batch: (outputs - batch["y"]) ** 2, inputs="x"),
                      optax.sgd(0.1), key=jax.random.key(0),
                      mesh=MeshSpec().build(gpus("sm90")))
    ended = AheadTarget("sm90", 1, str(tmp_path / "step.dew-aot")).compiled(trainer, data, steps=1)
    assert ended.ahead.devices == 1 and trainer.objective.inputs is not None


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
                   layout=Layout(min_shard=TINY_SHARD), ahead=ahead, links=links, checkpoints=checkpoints)


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
    """A step compiled ahead over a link its trainer states, as `trainer.links` states one, records
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
        run = fine_tune(here, ahead=AheadOfTime.load(path), xla=True, split=None)
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
