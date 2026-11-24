"""Programs compiled ahead for this GPU's generation by a process that sees no GPU load in a fresh process on
the GPU and run there with nothing compiled, bit for bit as the same programs compiled here through Dew's
compile point for the generation's facts, as the build compiles them (ledger 2716: a GPU node spends its time
on its device's work alone; 2026-11-02, an H100 ran a CPU host's sm90 build of a dot and the template
attention's forward so)."""

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
from lane_environment import without_flags
from ahead_programs import arguments
from jax.experimental import topologies

from dew._devices import pool_share
from dew.nn.kernels.generation import assumed
from dew.training.ahead import BuildTarget, compiled_in, dumps

pytestmark = pytest.mark.gpu(reason="a program compiled ahead runs on the GPU it was built for")

HERE = """
import pickle
import sys

import numpy as np
from ahead_programs import programs

from dew.nn.kernels.generation import assumed, built_target
from dew.nn.kernels.options import owned_program

generation, multiprocessors = sys.argv[1], int(sys.argv[2])
with open(sys.argv[3], "rb") as file:
    arrays = pickle.load(file)
here = {}
for name, (function, _) in programs(generation).items():
    # Under the facts the build used: this GPU's generation and its own multiprocessors.
    with assumed(built_target(generation, multiprocessors)):
        here[name] = np.asarray(owned_program(function)(*arrays[name]))
with open(sys.argv[4], "wb") as file:
    pickle.dump(here, file)
"""
"""A fresh process compiling the programs for argv[1] on this GPU at argv[2] multiprocessors, as the build
compiles them, and running them on the arguments at argv[3], their outputs written to argv[4]."""
UNTUNED = 3
"""The status the build exits with where its compile is refused for a choice no run of the GPU tuned
(`report.Untuned`), apart from any other failure's."""
BUILD = f"""
import sys

from dew.nn.kernels.templates import report
from dew.training.ahead import compile_programs
from ahead_programs import programs

try:
    built = compile_programs(programs(sys.argv[1]), sys.argv[1], int(sys.argv[3]), topology=sys.argv[4])
except ValueError as refused:
    if isinstance(refused.__cause__, report.Untuned):
        sys.exit({UNTUNED})
    raise
built.save(sys.argv[2])
"""
RUN = """
import pickle
import sys

import jax
import numpy as np

from dew.telemetry.compiles import executable_only
from dew.training.ahead import AheadPrograms

loaded = AheadPrograms.load(sys.argv[1]).loaded()
with open(sys.argv[2], "rb") as file:
    arrays = pickle.load(file)
with executable_only():
    results = {name: np.asarray(jax.block_until_ready(program(*arrays[name])))
               for name, program in loaded.items()}
with open(sys.argv[3], "wb") as file:
    pickle.dump(results, file)
"""
REFUSED = """
import sys

from dew.training.ahead import AheadPrograms

try:
    AheadPrograms.load(sys.argv[1]).loaded()
except ValueError:
    sys.exit(0)
sys.exit("its programs loaded in a process whose XLA computes otherwise")
"""
"""A fresh process loading the programs at argv[1], which its loader must refuse."""
TRAIN_REFUSED = """
import sys

import jax
from test_compile_ahead import BATCH, SEQ_LEN, VOCAB, fine_tune
from token_support import token_rows

from dew.training import Inspection, MeshSpec
from dew.training.ahead import AheadOfTime

from dew.nn.kernels.generation import chosen

# The fine-tune's attention XLA's (`chosen`), as `fine_tune(xla=True)` named it before row 11938.
with chosen(attention="xla"):
    run = fine_tune(MeshSpec().build(jax.devices()[:1]), ahead=AheadOfTime.load(sys.argv[1]), split=2,
                    inspection=Inspection(every=2), dtype="bfloat16")
    try:
        run.fit(token_rows(VOCAB, BATCH, SEQ_LEN + 1), steps=1, log_every=1)
    except ValueError:
        sys.exit(0)
    sys.exit("the step loaded in a process whose XLA computes otherwise")
"""
"""A fresh process whose trainer trains a step from the step at argv[1] (`QUIET_BUILD`'s), which its loader
must refuse as it loads it: a trainer handed a step loads it at its first step's compile, not as it is
made."""
QUIET_BUILD = """
import sys

from test_compile_ahead import BATCH, SEQ_LEN, VOCAB, fine_tune
from token_support import token_rows

from dew.data import Memory
from dew.training import Inspection, MeshSpec
from dew.nn.kernels.generation import compile_only_gpus
from dew.training.ahead import compile_ahead

generation, path, multiprocessors, memory = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
from dew.nn.kernels.generation import chosen

# The fine-tune's attention XLA's (`chosen`), as `fine_tune(xla=True)` named it before row 11938.
with chosen(attention="xla"):
    # Planned at the loading run's share (argv[5]), which its pool takes of the GPU's memory.
    target = MeshSpec().build(compile_only_gpus(generation, 1, memory, multiprocessors=multiprocessors,
                                                share=float(sys.argv[5])))
    trainer = fine_tune(target, split=2, inspection=Inspection(every=2), dtype="bfloat16")
    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    validation = {"val": Memory(token_rows(VOCAB, 2 * BATCH, SEQ_LEN + 1, seed=1)).batch(BATCH)}
    fit = {"steps": 6, "log_every": 5, "eval_every": 3, "validation": validation}
    compile_ahead(trainer, batch, fit=fit).save(path)
"""
"""A step of two microbatches, sampled every second step, compiled ahead for the GPU argv[1] names, with
argv[3] multiprocessors and argv[4] bytes of memory, with the programs of its fit's evaluations, and saved
at argv[2], in a process that sees no GPU."""
SAMPLED_BUILD = """
import sys

import jax
import optax
from test_loop_sampled import TRAINED, token_draw
from test_pass_weights import looped

from dew.nn.kernels.generation import compile_only_gpus
from dew.objectives.lm.objective import LMObjective
from dew.training import Layout, MeshSpec, Trainer, TrainerAdvanced
from dew.training.ahead import compile_ahead

generation, path, multiprocessors, memory = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
# Planned at the loading run's share (argv[5]), which its pool takes of the GPU's memory.
target = MeshSpec().build(compile_only_gpus(generation, 1, memory, multiprocessors=multiprocessors,
                                            share=float(sys.argv[5])))
trainer = Trainer(LMObjective(looped(steps=TRAINED.cap), 16), optax.sgd(0.1), key=jax.random.key(0),
                  mesh=target, advanced=TrainerAdvanced(layout=Layout(min_shard=1, tolerance=1.0),
                                                        sampled_passes=TRAINED))
compile_ahead(trainer, {"text": token_draw()}).save(path)
"""
"""A run that draws its looped model's pass count per microbatch, its step compiled ahead for the GPU of the
generation argv[1] names (as `QUIET_BUILD`'s), every count's beside the cap's, saved at argv[2], in a
process that sees no GPU."""

SAMPLED = """
import sys

import jax
import optax
from test_loop_sampled import TRAINED, token_draw
from test_pass_weights import looped

from dew.objectives.lm.objective import LMObjective
from dew.telemetry.compiles import Compiles
from dew.training import Layout, MeshSpec, Trainer, TrainerAdvanced
from dew.training.ahead import AheadOfTime

run = Trainer(LMObjective(looped(steps=TRAINED.cap), 16), optax.sgd(0.1), key=jax.random.key(0),
              mesh=MeshSpec().build(jax.devices()[:1]),
              advanced=TrainerAdvanced(layout=Layout(min_shard=1, tolerance=1.0), sampled_passes=TRAINED,
                                       ahead=AheadOfTime.load(sys.argv[1])))
with Compiles(None) as compiles:
    state = run.fit({"text": token_draw()}, steps=4, log_every=2)
assert int(state.step) == 4
compiled = [(program.name, program.where) for program in compiles.programs if program.kind == "compiled"]
assert not compiled, compiled
"""
"""A fresh process whose sampled run trains four steps from the step at argv[1] (`SAMPLED_BUILD`'s), each
microbatch at its drawn count, compiling nothing."""

QUIET = """
import sys

import jax
from test_compile_ahead import BATCH, SEQ_LEN, VOCAB, fine_tune
from token_support import token_rows

from dew.data import Memory
from dew.telemetry.compiles import Compiles
from dew.training import Inspection, MeshSpec
from dew.training.ahead import AheadOfTime

from dew.nn.kernels.generation import chosen

# The fine-tune's attention XLA's (`chosen`), as `fine_tune(xla=True)` named it before row 11938.
with chosen(attention="xla"):
    run = fine_tune(MeshSpec().build(jax.devices()[:1]), ahead=AheadOfTime.load(sys.argv[1]), split=2,
                    inspection=Inspection(every=2), dtype="bfloat16")
    validation = {"val": Memory(token_rows(VOCAB, 2 * BATCH, SEQ_LEN + 1, seed=1)).batch(BATCH)}
    with Compiles(None) as compiles:
        state = run.fit(token_rows(VOCAB, BATCH, SEQ_LEN + 1), steps=6, log_every=5, eval_every=3,
                        validation=validation)
    assert int(state.step) == 6 and run.plan["narrow"]
    compiled = [(program.name, program.where) for program in compiles.programs if program.kind == "compiled"]
    assert not compiled, compiled
"""
"""A fresh process training six steps of two microbatches from the step at argv[1], sampled every second
step, two samples to a log, evaluated every third step, compiling nothing on its GPU: its bf16 step's plan
keeps its narrow copies, which the programs compiled with it make (row 4777)."""
COMPUTING = "--xla_gpu_enable_fast_min_max=true"
"""An XLA flag that changes what a program computes, which a process loading programs compiled without it is
refused for (`BuildTarget.held`)."""


ALIASED = """
import json
import sys

import jax

mode, cache, topology = sys.argv[1], sys.argv[2], sys.argv[3]
jax.config.update("jax_compilation_cache_dir", cache)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
read = []
jax.monitoring.register_event_listener(
    lambda event, **metadata: read.append(event) if event == "/jax/compilation_cache/cache_hits" else None)
import jax.numpy as jnp
from jax.experimental import topologies

from dew.training.ahead import BuildTarget, host_gpus


def program(values, weights):
    # A product XLA tunes on a GPU and leaves untuned where it has none: a 4096-square bf16 GEMM fusion.
    return jax.nn.relu(values @ weights) * 2 + 1


SHAPE = (4096, 4096)


if mode == "topology":
    # This host's GPU's own topology and generation, written for a process that sees no GPU.
    with open(topology, "wb") as file:
        file.write(topologies.get_attached_topology("cuda").serialize())
    print(json.dumps({"generation": BuildTarget.of(jax.devices()[:1]).generation}))
elif mode == "compile":
    # No GPU in this process (none visible, JAX_PLATFORMS=cpu): XLA compiles over the topology with none to
    # tune on.
    (compile_only,) = host_gpus(sys.argv[4], topology, 1)
    there = jax.sharding.SingleDeviceSharding(compile_only)
    placed = jax.ShapeDtypeStruct(SHAPE, jnp.bfloat16, sharding=there)
    jax.jit(program).lower(placed, placed).compile()
    print(json.dumps({"read": len(read)}))
else:
    values = jnp.ones(SHAPE, jnp.bfloat16)
    jax.block_until_ready(jax.jit(program)(values, values))
    print(json.dumps({"read": len(read)}))
"""
"""A fresh process, by its mode (argv[1]), over a persistent cache (argv[2]) and a topology file (argv[3]):
`topology` writes this host's GPU's topology and prints its generation; `compile` compiles one program over
that topology for the generation (argv[4]) with no GPU attached, outside Dew's compile point; `run` runs it
on the GPU. Each prints how many programs it read from the cache."""


def environment_of(tmp_path: Path) -> dict[str, str]:
    """This process's environment for a fresh one running a script under `tmp_path`, with this suite's
    modules importable."""
    tests = str(Path(__file__).resolve().parent)
    paths = os.pathsep.join([str(tmp_path), tests, os.environ.get("PYTHONPATH", "")])
    return {**os.environ, "PYTHONPATH": paths}


def computing_otherwise(environment: dict[str, str]) -> dict[str, str]:
    """`environment` with `COMPUTING` among its XLA flags."""
    return {**environment, "XLA_FLAGS": f"{environment.get('XLA_FLAGS', '')} {COMPUTING}".strip()}


def test_programs_built_with_no_gpu_run_on_this_gpu_with_nothing_compiled_bit_for_bit_as_compiled_here(
        tmp_path):
    own = BuildTarget.of(jax.devices()[:1])
    generation = own.generation
    # A target is the device's own, whatever generation this process assumes.
    with assumed("sm80" if generation != "sm80" else "sm90"):
        assert BuildTarget.of(jax.devices()[:1]) == own
    for name, text in (("here.py", HERE), ("build.py", BUILD), ("run.py", RUN), ("refused.py", REFUSED)):
        (tmp_path / name).write_text(text)
    # Every process the autotuner on and measuring, the lane's level 0 and deterministic ops left out (under
    # which XLA measures nothing and keeps no picks), every other flag kept, so the build and the run agree
    # on their compile options; one store of picks of its own, written as well as read (a packed job's
    # read-only mode left out: row 8783), which here's compile writes and the build reads; and a compilation
    # cache of its own, which no other job's executable serves (row 8457: at level 0 nothing tuned, so a
    # node's compile never met an untuned choice and its file never said one).
    environment = environment_of(tmp_path)
    picks, mode = "--xla_gpu_per_fusion_autotune_cache_dir", "--xla_gpu_experimental_autotune_cache_mode"
    environment["XLA_FLAGS"] = " ".join((without_flags(
        environment.get("XLA_FLAGS", ""), "--xla_gpu_autotune_level", "--xla_gpu_deterministic_ops", picks,
        mode), f"{picks}={tmp_path / 'picks'}"))
    environment["JAX_COMPILATION_CACHE_DIR"] = str(tmp_path / "cache")
    arrays = {name: [np.asarray(array) for array in drawn] for name, drawn in arguments(generation).items()}
    (tmp_path / "arguments.pkl").write_bytes(pickle.dumps(arrays))
    topology = tmp_path / "topology.bin"
    topology.write_bytes(topologies.get_attached_topology("cuda").serialize())
    # Built where no GPU is visible, as a CPU host builds them, over this GPU's serialized topology.
    # For this GPU: its generation and its own multiprocessors. Refused before this GPU tuned them: their
    # picks are none yet.
    build = [sys.executable, str(tmp_path / "build.py"), generation, str(tmp_path / "programs.dew-aot"),
             str(own.multiprocessors), str(topology)]
    deviceless = {**environment, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"}
    assert subprocess.run(build, env=deviceless).returncode == UNTUNED
    # Compiled on this GPU, which tunes them and leaves its choices in the store the build reads (row 8422:
    # a node's programs compile tuned or are refused); then built.
    subprocess.run([sys.executable, str(tmp_path / "here.py"), generation, str(own.multiprocessors),
                    str(tmp_path / "arguments.pkl"), str(tmp_path / "here.pkl")], env=environment, check=True)
    here = pickle.loads((tmp_path / "here.pkl").read_bytes())
    subprocess.run(build, env=deviceless, check=True)
    # Loaded and run in a fresh process whose compilers' dumps must stay empty; refused in one whose XLA
    # computes otherwise.
    run = [sys.executable, str(tmp_path / "run.py"), str(tmp_path / "programs.dew-aot"),
           str(tmp_path / "arguments.pkl"), str(tmp_path / "outputs.pkl")]
    subprocess.run(run, env=dumps(tmp_path / "dumps", environment), check=True)
    assert compiled_in(tmp_path / "dumps") == []
    subprocess.run([sys.executable, str(tmp_path / "refused.py"), str(tmp_path / "programs.dew-aot")],
                   env=computing_otherwise(environment), check=True)
    ahead = pickle.loads((tmp_path / "outputs.pkl").read_bytes())
    for name, compiled in here.items():
        assert ahead[name].dtype == compiled.dtype and ahead[name].tobytes() == compiled.tobytes(), name


def test_a_run_compiled_ahead_for_this_gpu_compiles_nothing_on_it_and_is_refused_where_xla_computes_otherwise(
        tmp_path):
    """A run handed its step compiled ahead for this GPU where no GPU is visible, two microbatches a step and
    sampled every second step, trains six steps logged twice and evaluated twice in a fresh process
    compiling nothing on the GPU: its start, its window's allocation, its split step's whole, its counters,
    its latest sample, its log's copies, its logged tokens' sum and its evaluations' keys, loss, sums and
    reduction were compiled with the step (`AheadOfTime.starts`; row 4752), and its bf16 compute's narrow
    copies, which it makes, were too (row 4777). A fresh process whose XLA
    computes otherwise is refused that step by the trainer's loader: the one rule every loader holds (row
    3606)."""
    own = BuildTarget.of(jax.devices()[:1])
    for name, text in (("quiet_build.py", QUIET_BUILD), ("quiet.py", QUIET), ("refused.py", TRAIN_REFUSED)):
        (tmp_path / name).write_text(text)
    environment = environment_of(tmp_path)
    # Its memory stated as the GPU's own (`CompileAhead.memory`, a capacity), which the build takes the
    # loading run's share of as the GPU's allocator does: the attached GPU's allocator limit (`own.memory`) is
    # that share already, so the GPU's memory is it over the share (row 6717: stated as the limit, the build
    # planned for the share of the share), and the build is told that share, not Dew's own (row 7935: at
    # Dew's 0.95 it planned past a tier's packed fraction, which the loader refused).
    assert own.memory is not None
    build = [sys.executable, str(tmp_path / "quiet_build.py"), own.generation, str(tmp_path / "step.dew-aot"),
             str(own.multiprocessors), str(round(own.memory / pool_share())), str(pool_share())]
    subprocess.run(build, env={**environment, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"}, check=True)
    subprocess.run([sys.executable, str(tmp_path / "quiet.py"), str(tmp_path / "step.dew-aot")],
                   env=environment, check=True)
    subprocess.run([sys.executable, str(tmp_path / "refused.py"), str(tmp_path / "step.dew-aot")],
                   env=computing_otherwise(environment), check=True)


def test_a_sampled_run_compiled_ahead_for_this_gpu_compiles_none_of_its_counts_on_it(tmp_path):
    """A run that draws its looped model's pass count per microbatch (`SampledPasses`), handed its step
    compiled ahead for this GPU where no GPU is visible, trains in a fresh process compiling nothing on the
    GPU: the file holds every count's step at its cap's plan beside the cap's own (`AheadOfTime.counts`, row
    7924), which the run takes in place of compiling them. Its memory stated as
    `test_a_run_compiled_ahead_for_this_gpu_compiles_nothing_on_it_and_is_refused_where_xla_computes_otherwise`
    states it, at the loading run's share."""
    own = BuildTarget.of(jax.devices()[:1])
    for name, text in (("sampled_build.py", SAMPLED_BUILD), ("sampled.py", SAMPLED)):
        (tmp_path / name).write_text(text)
    environment = environment_of(tmp_path)
    assert own.memory is not None
    build = [sys.executable, str(tmp_path / "sampled_build.py"), own.generation,
             str(tmp_path / "step.dew-aot"), str(own.multiprocessors), str(round(own.memory / pool_share())),
             str(pool_share())]
    subprocess.run(build, env={**environment, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"}, check=True)
    subprocess.run([sys.executable, str(tmp_path / "sampled.py"), str(tmp_path / "step.dew-aot")],
                   env=environment, check=True)


def test_a_deviceless_compile_stores_nothing_a_device_run_reads(tmp_path):
    """The contract between the persistent cache's key and the autotuner: a program compiled in a process that
    sees no GPU (none visible, `JAX_PLATFORMS=cpu`) over this host's own topology (`host_gpus`), as a user's
    compile-only process compiles one (a 4096-square bf16 GEMM, outside Dew's compile point, with no
    process settings handed down), is no program of the GPU's own: XLA had no device to tune it on. So the
    GPU's run of the same program on the same persistent cache reads nothing the deviceless compile wrote and
    compiles its own: an untuned program is never read under a tuned key (rows 4792 and 6046). JAX post87
    read it (UB-4792-AHEAD-TIME-gh200); since post90's key it does not, read 0 on GH200 on post94 and post95
    (MC-UNTUNEDKEY4-post94-f74e55896, MC-UNTUNEDKEY4-post95-16d207cc1), so a JAX that loses that key turns
    this red."""
    (tmp_path / "aliased.py").write_text(ALIASED)
    # None of this process's JAX settings or XLA flags handed down (a lane's name its own tuning and cache),
    # as a user's process runs; its client's memory settings kept (`XLA_PYTHON_CLIENT_*`: a packed job's
    # share of a GPU others share).
    environment = {name: value for name, value in environment_of(tmp_path).items()
                   if not name.startswith("JAX_") and name != "XLA_FLAGS"}
    cache, topology = str(tmp_path / "cache"), str(tmp_path / "topology.bin")

    def ran(mode: str, *arguments: str, **variables: str) -> dict:
        done = subprocess.run([sys.executable, str(tmp_path / "aliased.py"), mode, cache, topology,
                               *arguments], env={**environment, **variables}, capture_output=True, text=True,
                              check=True)
        return json.loads(done.stdout.splitlines()[-1])

    generation = ran("topology")["generation"]
    # The GPU hidden as well: a process that sees one of the topology's own compiles through it, tuned.
    assert ran("compile", generation, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="") == {"read": 0}
    assert ran("run") == {"read": 0}


FRESH_WEIGHT_SEEDS = 8
"""Check 1's declared draws of a fresh state's weights on the GPU, placed against drawn under JAX's default
lowering (`seeds_for`: the two programs share every bit and differ only in how they round from them)."""


def test_a_fresh_state_placed_on_a_gpu_draws_the_bits_of_jaxs_default_lowering():
    """A fresh run's start program on the GPU draws through loops over threefry's rounds
    (`dew.objectives.drawn.drawing`): the raw bits it draws are bit for bit those of JAX's default lowering,
    every round unrolled, and the weights it builds from them (`Trainer.place`) are on par (check 1) with
    the state its initializer draws jitted under the default lowering, truth the initializer's float64
    replay over the same bits (`float64_replay`): the looped draw changes the start program's compile alone
    (row 6740), and floats two programs round from those bits are held on par, not bit for bit (row 9045,
    docs/design/testing.md's layout invariance)."""
    import jax.numpy as jnp
    import optax
    from float64_replay import float64_replay
    from placement_parity import floats
    from reference_error import as_given, assert_as_exact

    from dew.objectives.drawn import drawing
    from dew.objectives.lm import LMObjective
    from dew.registry import models
    from dew.training import Trainer

    model = models.build("causal_transformer", vocab_size=64, emb_features=32, num_layers=2, num_heads=2,
                         mlp_features=64, max_seq_len=16)
    objective = LMObjective(model, 15)

    def trainer(key) -> Trainer:
        return Trainer(objective, optax.adam(1e-3), key=key)

    # Its key formed on the host's CPU, as a site's draws are: a GPU tier's prepare computes on no other one.
    with jax.default_device(jax.devices("cpu")[0]):
        drawer = trainer(jax.random.key(0))

    def initial(data):
        # The initializer's draws from the key of `data`, as a fresh run's start makes them
        # (`Trainer.initial_state`): traceable, its floats read outside the trace.
        return drawer.initial_state(None, jax.random.wrap_key_data(data)).variables

    def drawn(seed):
        # A seed's key as its raw data, formed on the host's CPU, which a check-1 site compares bit for bit.
        with jax.default_device(jax.devices("cpu")[0]):
            return (np.asarray(jax.random.key_data(jax.random.key(seed))),)

    def placed(data):
        key = jax.random.wrap_key_data(data)
        return floats(trainer(key).place()[0].variables)

    drawn_here, replayed = jax.jit(initial), float64_replay(initial)
    assert_as_exact(as_given(placed), as_given(lambda data: floats(drawn_here(data))), drawn,
                    "a fresh state's weights placed and drawn", seeds=FRESH_WEIGHT_SEEDS,
                    truth_fn=as_given(lambda data: floats(replayed(data))))

    # The raw bits, after the check-1 site, which a GPU tier's prepare reaches computing on no device before
    # it (row 7872).
    def bits(key):
        return jax.random.bits(key, (64, 33), jnp.uint32)

    for seed in range(3):
        key = jax.random.key(seed)
        with drawing():
            rolled = jax.jit(bits)(key)
        assert np.array_equal(np.asarray(rolled), np.asarray(jax.jit(lambda key: bits(key))(key)))


CERTIFIED = """
import json
import sys

import jax
import jax.numpy as jnp

from dew.nn.kernels.generation import compile_only_gpus, device_generation
from dew.nn.kernels.templates import attention as template, attention_mosaic, report

stated = []
compile_ = jax.stages.Lowered.compile


def recorded(lowered, compiler_options=None, *args, **kwargs):
    # A certification's own program (`report.named`'s `certified_` name), the options JAX compiles it under.
    if "certified_" in lowered._lowering.stablehlo().operation.attributes["sym_name"].value:
        stated.append(compiler_options)
    return compile_(lowered, compiler_options, *args, **kwargs)


jax.stages.Lowered.compile = recorded
device = jax.devices("gpu")[0] if sys.argv[1] == "here" else compile_only_gpus(device_generation())[0]
tokens, heads, kv_heads, width = 640, 4, 2, 128
spans = template.spans(tokens, tokens, 1, causal=True)
placed = jax.sharding.NamedSharding(jax.sharding.Mesh([device], ("x",)), jax.sharding.PartitionSpec())
query = jax.ShapeDtypeStruct((1, tokens, heads, width), jnp.bfloat16, sharding=placed)
key = jax.ShapeDtypeStruct((1, tokens, kv_heads, width), jnp.bfloat16, sharding=placed)
with report.certifying(device):
    jax.jit(lambda q, k, v: attention_mosaic.forward(q, k, v, spans, None)).lower(query, key, key)
# Each distinct set of options a certification compiled under.
print(json.dumps(sorted({json.dumps(each, sort_keys=True) for each in stated})))
"""


def test_a_certification_compiles_alike_on_this_gpu_and_a_compile_only_one_of_its_generation(tmp_path):
    """A launch's certification, its kernels compiled alone to be read and run nowhere (`report.certified`),
    compiles under the same options on this GPU as on a compile-only GPU of its generation, so a run reads a
    deviceless build's certifications from the cache they share: the run's took a CUB sort's choice off
    its key (`xla_gpu_deviceless_cub_mode`) and the build's did not, and a GH200 run of a step built on a
    compile-only H100 compiled five of its certifications again (row 8372, M-MC-27-3e). Each side certifies
    in a process of its own: in one, the second reads the first's certification, keyed by its generation
    alone, and compiles none (row 9608). Each reads and writes a compilation cache of its own too, beside
    which its verdicts are stored (`tuning.store_verdict`, by generation, `report.verdict_target`): in one
    they share, the second reads the first's verdict and compiles nothing to compare (row 10104)."""
    (tmp_path / "certified.py").write_text(CERTIFIED)
    environment = environment_of(tmp_path)

    def options(where: str) -> set[str]:
        ran = subprocess.run([sys.executable, str(tmp_path / "certified.py"), where],
                             env={**environment, "JAX_COMPILATION_CACHE_DIR": str(tmp_path / where)},
                             check=True, capture_output=True, text=True)
        return set(json.loads(ran.stdout.strip().splitlines()[-1]))

    here = options("here")
    assert here and here == options("compile-only"), here


@pytest.mark.users_process
def test_a_program_compiled_ahead_for_compile_only_gpus_of_no_stated_memory_computes_as_compiled_here():
    """Dew's portable recurrence gradient (test_recurrence_stability's scalar delta at aligned keys), compiled
    ahead for a compile-only GPU of this one's generation and size that states no memory, as a tier's
    prepare compiles (`compile_programs`), loaded here, computes bit for bit what its compile on this GPU
    computes, one program under one set of options (`report.targeted`): XLA's memory-driven passes compiled
    it for a GPU of 0 bytes to values O(1) off, a key's gradient 5.44 off where it is 41.8 (row 8768,
    DC-p15-ahead-ab-gh200)."""
    import functools

    import test_recurrence_stability as stability
    from arm_arithmetic import summed_as_run
    from arm_store import _arm_program
    from reference_error import with_gradients
    from truths_ahead import named

    from dew.nn.kernels import generation
    from dew.nn.kernels.templates import report
    from dew.training.ahead import Executable

    hooks = stability.PORTABLE_SITES["scalar delta"]
    dew = with_gradients(stability.restarting(functools.partial(
        stability.chunk_on, "xla", hooks, chunk_size=stability.CHUNK, **stability.bounded("aligned keys")),
        "aligned keys"))
    inputs = stability.cotangents(hooks, "aligned keys", 0)
    leaves = jax.tree.leaves(inputs)
    with summed_as_run():
        programs, held = {}, {}
        _arm_program(programs, held, "dew", dew, inputs, reference=False)
        function, shapes = programs[named("dew", inputs)]
        here = jax.jit(function).lower(*leaves)
        expected = jax.device_get(here.compile(compiler_options=report.targeted(here, None))(*leaves))
        target = generation.built_target(generation.gpu_target().generation,
                                         generation.gpu_target().multiprocessors)
        (gpu,) = generation.compile_only_gpus(target.generation, 1, multiprocessors=target.multiprocessors)
        mesh = jax.sharding.Mesh(np.asarray([gpu]), ("devices",))
        placed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        with (generation.assumed(target, portable=True), jax.sharding.use_abstract_mesh(mesh.abstract_mesh),
              report.certifying(gpu)):
            lowered = jax.jit(function, out_shardings=placed).lower(
                *(jax.ShapeDtypeStruct(shape.shape, shape.dtype, sharding=placed) for shape in shapes))
            ahead = lowered.compile(compiler_options=report.targeted(lowered, None))
        got = jax.device_get(Executable.of(ahead).loaded(jax.devices()[:1])(*leaves))
    for value, wanted in zip(jax.tree.leaves(got), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(np.asarray(value), np.asarray(wanted))


SERVED_KEPT = """
import json
import sys

import jax
import numpy as np
from jax.experimental import pallas as pl

from dew.nn.kernels.options import owned_program
from dew.telemetry.compiles import Compiles
from dew.training.ahead import kept, keeping

# The persistent cache's reads: each program served from it, and each it had none of.
events = []
jax.monitoring.register_event_listener(lambda event, **_: events.append(event))


def kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...] * 2.0 + 1.0


def doubled(x):
    return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype))(x)


# Drawn on the host, so the only program a process needs is `doubled`'s.
x = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
mode, path = sys.argv[1], sys.argv[2]
with Compiles(None) as compiles:
    if mode == "cache":
        out = owned_program(doubled)(x)
    elif mode == "keep":
        with keeping() as programs:
            out = owned_program(doubled)(x)
        programs.save(path)
    else:
        with kept(path):
            out = owned_program(doubled)(x)
print(json.dumps({"ok": bool(np.array_equal(np.asarray(out), x * 2 + 1)),
                  "hits": sum(event == "/jax/compilation_cache/cache_hits" for event in events),
                  "misses": sum(event == "/jax/compilation_cache/cache_misses" for event in events),
                  "compiled": len(compiles.seconds)}))
"""
"""A Pallas program on this GPU, in a process of its own sharing one cache: compiled into the persistent cache
(`cache`), kept by `keeping` and saved to `path` (`keep`), or run from the file (`run`); whether it answered
exactly, how many programs its cache served and had none of, and how many it compiled (`Compiles`, which
hears a compile whether or not the cache keeps it)."""


def test_a_gpu_process_keeps_a_program_its_cache_serves_and_another_runs_it_compiling_nothing(tmp_path):
    """On a GPU a block keeping its programs (`ahead.keeping`) reads JAX's persistent cache as any compile
    does: a Pallas program the cache serves is kept and saved whole, its kernel among it, and a third
    process handed the file runs it exactly, compiling nothing (row 6641; a CPU's served executable loses its
    kernels, so a CPU block compiles each again)."""
    from process_support import fresh

    path = str(tmp_path / "served.dew-aot")
    assert fresh(SERVED_KEPT, "cache", path, cache_dir=tmp_path / "cache")["ok"]
    keep = fresh(SERVED_KEPT, "keep", path, cache_dir=tmp_path / "cache")
    run = fresh(SERVED_KEPT, "run", path, cache_dir=tmp_path / "other")
    assert keep["ok"] and keep["hits"] and not keep["misses"], keep
    assert run["ok"] and not run["hits"] and not run["misses"] and not run["compiled"], run

