"""The kernel configs' whole path: a step's launches found in its lowered
program, keyed by bucketed shape, resolved from a pin, a stored pick or a
heuristic, stored and read back in a fresh process, and measured once on a
GPU that a process has to itself (dew.nn.kernels.tuning)."""

import ast
import dataclasses
import functools
import json
import os
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
import tokamax
from lane_environment import DETERMINISTIC_OPS
from process_support import READ_ONLY, REPO_ROOT, fresh, keeping_picks, run_pool
from reference_error import equations

from dew.nn.activations import gated_product
from dew.nn.attention import rms_normalized, stream_rms_normalized
from dew.nn.kernels import generation, norm_swiglu, tuning


def loss(x, branch, scale, gate, up):
    stream, normed = stream_rms_normalized(x, branch, scale, 1e-6, x.dtype, scale_offset=True,
                                           scale_after_cast=False, fp32_statistics=True,
                                           implementation="pallas")
    # The product feeds the rest of the step, as a down projection reads it: a step whose gradient
    # never reads a fused kernel's output drops its forward launch (`generation.linear_tangent`).
    product = gated_product("swiglu", implementation="pallas")(gate, up)
    return jnp.sum(stream * normed) + jnp.sum(product * product)


def plain(x, scale):
    return rms_normalized(x, scale, 1e-6, x.dtype, scale_offset=True, scale_after_cast=False,
                          fp32_statistics=True, implementation="pallas")


def step():
    return jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4)))


def operands(rows: int, width: int = 64, hidden: int = 96):
    keys = jax.random.split(jax.random.key(0), 5)
    return (jax.random.normal(keys[0], (rows, width)), jax.random.normal(keys[1], (rows, width)),
            jax.random.normal(keys[2], (width,)), jax.random.normal(keys[3], (rows, hidden)),
            jax.random.normal(keys[4], (rows, hidden)))


def launches(function, *args) -> dict[str, tuple[tuple[int, ...], int]]:
    """Each Pallas launch's grid and warps in `function`'s program, by kernel name."""
    return {name: (tuple(grid), warps) for name, grid, warps in launched(jax.make_jaxpr(function)(*args))}


def launched(program) -> list[list]:
    """Each Pallas launch of the traced `program` (a closed jaxpr), its sub-programs' included, as
    `[kernel name, grid, warps]`, in the order traced."""
    return [[str(eqn.params["name"]), list(eqn.params["grid_mapping"].grid),
             eqn.params["compiler_params"].num_warps]
            for eqn in equations(program.jaxpr) if eqn.primitive.name == "pallas_call"]




STORED = """
import json, os, sys
from dew import cache
from dew.files import write_atomically
path = cache.kernel_picks_path(json.loads(sys.argv[1])["device_kind"])
os.makedirs(os.path.dirname(path), exist_ok=True)
write_atomically(path, sys.argv[1])
print(json.dumps([path]))
"""

FOREIGN = "dew.nn.kernels.another_branch:Kernel"
"""A kernel another checkout stores and this one cannot read."""


def with_foreign(text: str) -> str:
    """A store's JSON with two more launches this checkout cannot read: one of a kernel it lacks, and
    one of a kernel it has whose stored arguments lack an operand (another checkout's signature of
    it, as before a launch gained an operand). The first is last."""
    stored = json.loads(text)
    entry = json.loads(json.dumps(stored["data"][0]))
    entry["op"]["dataclass"] = FOREIGN
    unbound = json.loads(json.dumps(stored["data"][0]))
    del unbound["arguments"][next(iter(unbound["arguments"]))]
    return json.dumps({**stored, "data": [*stored["data"], unbound, entry]})


READ = """
import json
from test_tuning import launches, operands, step
from dew.nn.kernels import tuning
print(json.dumps([{rows: launches(step(), *operands(rows)) for rows in (13, 15, 40)},
                  tuning.settled(tuning.tune(lambda: step().lower(*operands(15)))[1])]))
"""


def test_a_steps_launches_take_pins_stored_picks_and_heuristics(tmp_path):
    """The four launches of a norm seam and SwiGLU step are found in its
    lowered program, keyed with their token axis at a power of two. Each
    takes its heuristic, a pinned config its space holds (one it does not
    raises, naming the config and why), and in a fresh
    process the picks stored for its device beside its compilation cache,
    for every token count in the bucket, and a token count stored for no
    bucket the nearest bucket's, beside a launch of a kernel only another
    checkout has. Nothing measures on a CPU or beside other work on a GPU.
    A process under deterministic ops reads none of them and launches its
    heuristics, the same in every process while other runs store picks."""
    with tuning.launches():
        found = tokamax.autotuning.get_bound_args(step(), *operands(15))
        thirteen = tokamax.autotuning.get_bound_args(step(), *operands(13))
    ops = {type(ba.op): ba for ba in found}
    assert set(ops) == {norm_swiglu.RmsForward, norm_swiglu.RmsBackward, norm_swiglu.SwigluForward,
                        norm_swiglu.SwigluBackward}
    keys = {kind: dict(ba.autotuning_cache_key) for kind, ba in ops.items()}
    assert keys[norm_swiglu.RmsForward]["x"].shape == (16, 64)
    assert keys[norm_swiglu.RmsBackward]["dy"].shape == (16, 64)
    assert keys[norm_swiglu.SwigluBackward]["gate"].shape == (2048,)
    assert keys == {type(ba.op): dict(ba.autotuning_cache_key) for ba in thirteen}

    usual = launches(step(), *operands(15))
    assert usual["rms_norm_forward"] == ((1,), 4) and usual["swiglu_forward"] == ((2,), 4)
    # A pinned config its space holds wins; one it does not hold (64 rows of a 64-wide norm) raises.
    with tuning.pinned({norm_swiglu.RmsForward: norm_swiglu.RowsConfig(1, 1),
                        norm_swiglu.RmsBackward: norm_swiglu.ReverseConfig(1, 256, 2),
                        norm_swiglu.SwigluForward: norm_swiglu.BlockConfig(4096, 8)}):
        pinned = launches(step(), *operands(15))
    assert pinned["rms_norm_forward"] == ((15,), 1) and pinned["rms_norm_backward"] == ((15,), 2)
    assert pinned["swiglu_forward"] == ((1,), 8) and pinned["swiglu_backward"] == usual["swiglu_backward"]
    with (tuning.pinned({norm_swiglu.RmsForward: norm_swiglu.RowsConfig(64, 2)}),
          pytest.raises(tuning.Unpinnable) as refused):
        launches(step(), *operands(15))
    assert (refused.value.kernel, refused.value.config) == ("RmsForward", norm_swiglu.RowsConfig(64, 2))

    # The picks a GPU measured, stored for this device, as a fresh process finds them.
    picked = {norm_swiglu.RmsForward: norm_swiglu.RowsConfig(4, 2),
              norm_swiglu.RmsBackward: norm_swiglu.ReverseConfig(2, 256, 8),
              norm_swiglu.SwigluForward: norm_swiglu.BlockConfig(512, 2),
              norm_swiglu.SwigluBackward: norm_swiglu.BlockConfig(256, 1)}
    timing = tokamax.BenchmarkData(compile_time_ms=1.0, lower_time_ms=1.0, evaluation_times_ms=(1.0,),
                                   metadata={})
    # A plain norm's pick beside them: a launch with no branch, which the seam's nearest search passes over.
    with tuning.launches():
        [alone] = tokamax.autotuning.get_bound_args(jax.jit(plain), operands(15)[0], operands(15)[2])
    picks = tuning.dumps_picks(jax.devices()[0].device_kind, (
        *((ba, tuning.Timings({picked[type(ba.op)]: timing})) for ba in found),
        (alone, tuning.Timings({norm_swiglu.RowsConfig(2, 1): timing}))))
    # Stored as every checkout writes its picks: each op as its picks name it, without the device check
    # a describing launch bypasses (`tuning._identity`).
    written = json.loads(picks)
    assert all(entry["op"]["fields"]["bypass_device_check"] is False for entry in written["data"])
    flags = f"{keeping_picks()} {READ_ONLY}".strip()
    [path] = fresh(STORED, with_foreign(json.dumps(written)), cache_dir=tmp_path / "cache", flags=flags)
    assert path.startswith(str(tmp_path / "cache" / "autotune" / "jaxlib-")) and path.endswith(".json")
    stored, tuned = fresh(READ, cache_dir=tmp_path / "cache", flags=flags)
    for rows in ("13", "15", "40"):
        assert stored[rows]["rms_norm_forward"] == [[-(-int(rows) // 4)], 2]
        assert stored[rows]["swiglu_backward"] == [[-(-int(rows) * 96 // 256)], 1]
    for rows in ("13", "15"):
        assert stored[rows]["rms_norm_backward"] == [[-(-int(rows) // 2)], 8]
    assert tuned is True
    assert os.listdir(os.path.dirname(path)) == [os.path.basename(path)]
    repeatable, _ = fresh(READ, cache_dir=tmp_path / "cache", flags=f"{flags} {DETERMINISTIC_OPS}")
    heuristics = {str(rows): launches(step(), *operands(rows)) for rows in (13, 15, 40)}
    assert repeatable == json.loads(json.dumps(heuristics))
    assert repeatable != stored


def test_a_lowering_records_each_launch_it_traces_as_a_lowering_made_to_describe_it_reads_it():
    """A step's lowering records each launch whose config picks decide as it traces it (`seen`),
    keyed as tokamax's description of it in a lowering made to read it is (`launches`), and again in
    a lowering whose trace was cached before: a step needs no lowering of its own to be tuned."""
    def keyed(op, ba):
        return type(op).__name__, repr(sorted(op.keyed(ba).autotuning_cache_key.items()))

    with tuning.launches():
        found = tokamax.autotuning.get_bound_args(step(), *operands(15))
    described = sorted(keyed(ba.op, ba) for ba in found)
    function = step()
    function.lower(*operands(15))
    for _ in range(2):
        with tuning.seen() as found:
            function.lower(*operands(15))
        assert sorted(keyed(op, ba) for op, ba in found) == described


def test_a_launch_is_resolved_and_described_without_asking_a_traced_array_for_its_devices(monkeypatch):
    """Each launch of a step's lowering, the plain one and the one that
    describes its launches to `tune`, reads its config and its GPU model
    without asking a traced array for its devices. That question raises an
    error whose message walks the whole trace so far to say where the array
    came from, once per array of each launch: quadratic in the step, it was
    111 s of the packed Qwen3.5-0.8B step's 330 s compile on an A100."""
    raised = []
    made = jax.errors.ConcretizationTypeError.__init__

    def counted(self, *args, **kwargs):
        raised.append(type(self))
        made(self, *args, **kwargs)

    monkeypatch.setattr(jax.errors.ConcretizationTypeError, "__init__", counted)
    with tuning.launches():
        step().lower(*operands(15))
    step().lower(*operands(15))
    assert raised == []


DISTINCT = """
import json, logging
import jax, tokamax
from test_tuning import loss, operands, step
from dew.nn.kernels import generation, tuning


def layers(x, branch, scale, gate, up):
    return loss(x, branch, scale, gate, up) + loss(2 * x, branch, scale, gate, up)


def described(launches):
    return sorted([type(ba.op).__name__, repr(sorted(ba.autotuning_cache_key.items()))]
                  for ba in tuning.loads_launches(launches))


class Told(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append([record.levelname, record.getMessage()])


told = Told()
logging.getLogger(tuning.__name__).addHandler(told)
logging.getLogger(tuning.__name__).setLevel(logging.DEBUG)
with generation.assumed("sm80"):
    _, two = tuning.tune(lambda: jax.make_jaxpr(jax.grad(layers, argnums=(0, 1, 2, 3, 4)))(*operands(15)))
    _, one = tuning.tune(lambda: jax.make_jaxpr(step())(*operands(15)))
print(json.dumps([described(two), described(one), told.records]))
"""


def test_a_step_settles_each_distinct_launch_once(tmp_path):
    """Two layers of one shape launch the same four kernels at the same keys.
    Tuned as on a GPU, the two-layer step's launches are the one layer's,
    each once: the pool decides each distinct launch once, and a measuring
    pass measures it once, not once a layer. With no picks stored, the process
    says so once, as information: a heuristic config is a working default,
    which a notebook's output should not flag as a fault."""
    two, one, told = fresh(DISTINCT, cache_dir=tmp_path / "cache", flags=keeping_picks())
    assert len(one) == 4 and two == one
    assert [level for level, message in told if "have no config measured" in message] == ["INFO"]


@pytest.mark.parametrize("stores, rows, norms", [
    ({0: {8: (4, 2)}}, (15,), [[[4], 2]]),
    ({1: {8: (4, 2)}}, (15,), [[[1], 4]]),
    ({0: {16: (4, 2)}, 1: {8: (4, 2), 64: (8, 4)}}, (16, 32), [[[4], 2], [[8], 2]]),
], ids=["process 0 holds", "process 1 holds", "one pick moves another's nearest"])
def test_every_process_launches_process_zeros_config_for_a_launch_without_a_pick(tmp_path, stores, rows,
                                                                                   norms):
    """A launch without a pick of its own takes the nearest stored pick, or its
    heuristic, as process 0 resolves it, whatever another process stores: the
    program `tune` returns on each of two processes launches each norm alike.
    Where process 0 stores an 8-token norm's pick, 4 rows a program and 2
    warps, a 15-token step's norm takes it on both; where process 1 does, the
    heuristic's one program of 4 warps. Where process 0 stores a 16-token pick
    and process 1 an 8- and a 64-token one, a step at 16 and 32 tokens takes
    process 0's pick for both on both: process 1's 32-token norm, first traced
    with its 64-token neighbour's 8 rows and 4 warps, traces again once the
    16-token pick that becomes its nearest is installed."""
    timing = tokamax.BenchmarkData(compile_time_ms=1.0, lower_time_ms=1.0, evaluation_times_ms=(1.0,),
                                   metadata={})

    def keyed(count):
        # The launch as a step traced for the GPU keys it, which is how a measurement stores it.
        with generation.assumed("sm80"), tuning.seen() as found:
            jax.make_jaxpr(step())(*operands(count))
        [norm] = [ba for op, ba in found if type(op) is norm_swiglu.RmsForward]
        return norm

    picks = tmp_path / "picks"
    picks.mkdir()
    for process, held in stores.items():
        stored = tuple((keyed(count), tuning.Timings({norm_swiglu.RowsConfig(*config): timing}))
                       for count, config in held.items())
        (picks / f"process{process}.json").write_text(tuning.dumps_picks(jax.devices()[0].device_kind,
                                                                         stored))
    reports = run_pool("settle", tmp_path / "pool", 2, script=Path(__file__).with_name("tuning_worker.py"),
                       cache=tmp_path / "cache", picks=picks, rows=",".join(map(str, rows)))
    assert reports[0] == reports[1]
    assert [[grid, warps] for name, grid, warps in reports[0] if name == "rms_norm_forward"] == norms


TRAINED = """
import json, os, sys
import jax, numpy as np, optax
import dew.nn.backbones
from dew import cache
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.training import Trainer
from dew.training.distributed import shard_batch

seq = int(sys.argv[1])
model = models.build("causal_transformer", vocab_size=4096, emb_features=1024, num_layers=2, num_heads=8,
                     mlp_features=3584, max_seq_len=seq)
trainer = Trainer(LMObjective(model, seq), optax.adam(1e-3), key=jax.random.key(0))
state, _, _ = trainer.place()
tokens = np.random.default_rng(0).integers(0, 4096, size=(4, seq + 1)).astype(np.int32)
trainer.compile(state, shard_batch(trainer.device_mesh, {"text": tokens}))
path = cache.kernel_picks_path(jax.devices()[0].device_kind)
print(json.dumps([path, os.stat(path).st_mtime_ns if os.path.exists(path) else None]))
"""


MEASURED = """
import json, os, sys
import jax
from test_tuning import operands, step
from dew import cache
from dew.nn.kernels import tuning
lowerings = []


def lower():
    lowerings.append(None)
    return step().lower(*operands(int(sys.argv[1])))


_, launches = tuning.tune(lower, measure=True)
path = cache.kernel_picks_path(jax.devices()[0].device_kind)
stamp = os.stat(path).st_mtime_ns if os.path.exists(path) else None
print(json.dumps([path, stamp, tuning.settled(launches), len(lowerings)]))
"""


@pytest.mark.gpu
def test_only_an_explicit_pass_on_a_gpu_to_itself_measures_and_it_measures_each_launch_once(tmp_path):
    """A language model's step compiled by the Trainer measures nothing. An
    explicit pass (`tune(..., measure=True)`) on a GPU it has to itself
    measures each launch over every config of its space, stores every time
    and keeps the fastest; the next pass measures nothing, and one beside
    other work on its GPU (gpu-run -n 0.5) adds no picks for a new shape."""
    inherited = keeping_picks()
    if READ_ONLY in inherited:
        pytest.skip("measuring needs the GPU to itself: gpu-run without -n")
    path, untouched = fresh(TRAINED, "2048", cache_dir=tmp_path / "cache", flags=inherited)
    assert untouched is None
    # Another checkout's launch, which the pass reads past and keeps.
    with tuning.launches():
        found = tokamax.autotuning.get_bound_args(step(), *operands(15))
    timing = tokamax.BenchmarkData(compile_time_ms=1.0, lower_time_ms=1.0, evaluation_times_ms=(1.0,),
                                   metadata={})
    seed = tuning.dumps_picks(jax.devices()[0].device_kind,
                              ((found[0], tuning.Timings({found[0].heuristics_config: timing})),))
    foreign = json.loads(with_foreign(seed))["data"][-2:]
    fresh(STORED, json.dumps({**json.loads(seed), "data": foreign}), cache_dir=tmp_path / "cache",
          flags=inherited)
    path, measured, settled, _ = fresh(MEASURED, "2048", cache_dir=tmp_path / "cache", flags=inherited)
    assert settled is True
    with open(path) as file:
        written = json.load(file)
    assert all(entry in written["data"] for entry in foreign)
    stored, others = tuning._read(path)
    assert others == foreign
    kinds = {type(ba.op) for ba, _ in stored.values()}
    assert {norm_swiglu.RmsForward, norm_swiglu.RmsBackward} <= kinds
    for ba, timings in stored.values():
        assert len(timings) > 1, type(ba.op)
        assert all(data.median_evaluation_time_ms > 0 for data in timings.values())
        assert timings.fastest_config in ba.autotuning_configs
    # A pass whose launches all have picks lowers its step once, recording them as it traces.
    assert fresh(MEASURED, "2048", cache_dir=tmp_path / "cache", flags=inherited) == [path, measured, True, 1]
    assert fresh(MEASURED, "1024", cache_dir=tmp_path / "cache",
                 flags=f"{inherited} {READ_ONLY}".strip()) == [path, measured, False, 1]


class KernelBug(Exception):
    """A kernel's own failure, which no GPU limit explains."""


@pytest.mark.gpu(reason="the tuner measures on CUDA")
def test_a_config_failing_for_more_than_its_shared_memory_stops_the_measurement():
    """A launch one of whose configs raises as it lowers, a kernel's bug and not its GPU's shared
    memory: measuring the launch propagates the error, where the tuner had counted it one more
    failed config and kept the others' fastest."""
    with tuning.launches():
        found = tokamax.autotuning.get_bound_args(step(), *operands(15))
    [norm] = [ba for ba in found if type(ba.op) is norm_swiglu.RmsForward]
    broken_config = sorted(norm.autotuning_configs, key=repr)[0]

    class Broken(norm_swiglu.RmsForward):
        """The norm's forward, raising for one config of its space. Its signature is the norm's, which
        tokamax reads to bind the launch's operands."""

        @functools.wraps(norm_swiglu.RmsForward._fwd)
        def _fwd(self, *args, config, **kwargs):
            if config == broken_config:
                raise KernelBug
            return super()._fwd(*args, config=config, **kwargs)

    fields = {field.name: getattr(norm.op, field.name) for field in dataclasses.fields(norm.op) if field.init}
    broken = tokamax.BoundArguments(Broken(**fields), norm.arguments)
    with pytest.raises(KernelBug):
        tuning._measured(broken.op, broken)


@pytest.mark.gpu(reason="the tuner measures on CUDA")
def test_a_launch_measured_beside_a_profile_capture_keeps_its_heuristic(tmp_path):
    """A profile capture holds CUPTI, which takes one subscriber: measuring a launch beside it times
    no config, returns each as failed, so the launch keeps its heuristic, and leaves CUPTI to the
    capture without raising; once the capture ends, the same launch measures every config."""
    with tuning.launches():
        found = tokamax.autotuning.get_bound_args(step(), *operands(15))
    [norm] = [ba for ba in found if type(ba.op) is norm_swiglu.RmsForward]
    jax.profiler.start_trace(str(tmp_path / "trace"))
    try:
        beside = tuning._measured(norm.op, norm)
    finally:
        jax.profiler.stop_trace()
    assert set(beside) == set(norm.autotuning_configs) and not beside.prune_errors()
    alone = tuning._measured(norm.op, norm)
    assert set(alone.prune_errors()) == set(norm.autotuning_configs)


STORES = """
import json, sys
import jax
import tokamax
from test_tuning import operands, step
from dew import cache
from dew.nn.kernels import norm_swiglu, tuning
with tuning.launches():
    found = tokamax.autotuning.get_bound_args(step(), *operands(15))
[norm] = [ba for ba in found if type(ba.op) is norm_swiglu.RmsForward]
timing = tokamax.BenchmarkData(compile_time_ms=1.0, lower_time_ms=1.0, evaluation_times_ms=(1.0,),
                               metadata={})
config = norm_swiglu.RowsConfig(*json.loads(sys.argv[1]))
pick = tuning.Timings({config: timing})
tuning._store(jax.devices()[0].device_kind, ((norm, pick),))
path = cache.kernel_picks_path(jax.devices()[0].device_kind)
read, foreign = tuning._read(path)
picks = {json.dumps(sorted(ba.arguments)): [timings.fastest_config.rows, timings.fastest_config.warps]
         for ba, timings in read.values() if type(ba.op) is norm_swiglu.RmsForward}
print(json.dumps([path, picks, len(foreign)]))
"""


def widened_checkout(root: Path) -> Path:
    """A copy of this checkout's package whose norm launch takes one operand more (`shift`), as
    another branch's would."""
    shutil.copytree(REPO_ROOT / "src" / "dew", root / "dew")
    path = root / "dew" / "nn" / "kernels" / "norm_swiglu.py"
    tree = ast.parse(path.read_text())
    [kernel] = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RmsForward"]
    [fwd] = [node for node in kernel.body if isinstance(node, ast.FunctionDef) and node.name == "_fwd"]
    fwd.args.kwonlyargs.insert(0, ast.arg("shift"))
    fwd.args.kw_defaults.insert(0, ast.Constant(None))
    path.write_text(ast.unparse(tree))
    return root


def test_a_checkout_whose_kernel_drops_an_operand_keeps_the_others_picks_of_it(tmp_path):
    """Two checkouts share a store: another branch's, whose norm launch takes an operand more, and
    this one. Each stores its pick of the launch through the store's writer and reads back only its
    own, and each write leaves the other's entry as it was."""
    flags = keeping_picks()
    cache_dir = tmp_path / "cache"
    other = widened_checkout(tmp_path / "other")
    path, theirs, _ = fresh(STORES, "[4, 2]", cache_dir=cache_dir, flags=flags, source=other)
    [(names, pick)] = theirs.items()
    assert "shift" in json.loads(names) and pick == [4, 2]
    with open(path) as file:
        [entry] = json.load(file)["data"]
    _, ours, foreign = fresh(STORES, "[2, 8]", cache_dir=cache_dir, flags=flags)
    [(names, pick)] = ours.items()
    assert "shift" not in json.loads(names) and pick == [2, 8] and foreign == 1
    with open(path) as file:
        assert entry in json.load(file)["data"]
    _, theirs, foreign = fresh(STORES, "[4, 2]", cache_dir=cache_dir, flags=flags, source=other)
    [(names, pick)] = theirs.items()
    assert "shift" in json.loads(names) and pick == [4, 2] and foreign == 1


def test_a_launch_is_chosen_for_the_gpu_its_operands_or_its_mesh_name():
    """`tuning._kind`, read through public jax as tokamax reads a launch's GPU for its cache: a traced
    launch under a mesh of abstract A100s (`generation.gpu_mesh`, `Trainer.lower_for`) is the A100's,
    whatever this host is; a concrete one is its operands' device's; neither, the default device's."""
    mesh = jax.make_mesh((1,), ("data",), axis_types=(jax.sharding.AxisType.Explicit,))
    shape = jax.ShapeDtypeStruct((8, 64), jnp.bfloat16)
    keyed = tokamax.BoundArguments(norm_swiglu.RmsForward(), {"x": shape})
    with jax.sharding.use_abstract_mesh(generation.gpu_mesh(mesh, "sm80")):
        assert tuning._kind(keyed) == generation.gpu_target("sm80").device_kind
    assert tuning._kind(keyed) == jax.devices()[0].device_kind
    concrete = tokamax.BoundArguments(norm_swiglu.RmsForward(), {"x": jnp.ones((8, 64), jnp.bfloat16)})
    assert tuning._kind(concrete) == jax.devices()[0].device_kind
