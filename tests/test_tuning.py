"""The kernel configs' whole path: a step's launches recorded as it traces,
keyed by bucketed shape, resolved from a pin, a stored pick or a
heuristic, stored and read back in a fresh process, and measured once on a
GPU that a process has to itself (dew.nn.kernels.tuning)."""

import dataclasses
import functools
import gc
import json
import operator
import os
import threading
import weakref
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

import jax
import jax.numpy as jnp
import pytest
import tokamax
from lane_environment import DETERMINISTIC_OPS, without_flags
from process_support import READ_ONLY, fresh, keeping_picks, run_pool
from reference_error import equations
from tokamax._src import config as tokamax_config
from tuned import recorded

from dew import _cache
from dew.nn.activations import gated_product
from dew.nn.kernels import generation, norm_swiglu, tuning
from dew.nn.kernels.templates import assembled
from dew.nn.norms import stream_rms_normalized
from dew.telemetry import compiles


def seam(x, branch, scale):
    """The fused pre-norm seam called itself (`norm_swiglu.stream_rms_norm`), interpreted on a CPU, XLA's
    seam its portable arithmetic."""
    def portable(x, branch, scale):
        return stream_rms_normalized(x, branch, scale, 1e-6, x.dtype, scale_offset=True,
                                     scale_after_cast=False, fp32_statistics=True, implementation="xla")

    return norm_swiglu.stream_rms_norm(x, branch, scale, 1e-6, x.dtype, offset=True, after_cast=False,
                                       portable=portable)


def loss(x, branch, scale, gate, up):
    stream, normed = seam(x, branch, scale)
    # The product feeds the rest of the step, as a down projection reads it: a step whose gradient
    # never reads a fused kernel's output drops its forward launch (`generation.linear_tangent`).
    product = norm_swiglu.swiglu(gate, up, gated_product("swiglu", implementation="xla"))
    return jnp.sum(stream * normed) + jnp.sum(product * product)


def plain(x, scale):
    return seam(x, None, scale)[1]


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
    # Each kernel's name less its launch's digest (`report.named`).
    return [[assembled.kernel_of(str(eqn.params["name"]))[0], list(eqn.params["grid_mapping"].grid),
             eqn.params["compiler_params"].num_warps]
            for eqn in equations(program.jaxpr) if eqn.primitive.name == "pallas_call"]




STORED = """
import json, os, sys
from dew import _cache
from dew._files import write_atomically
path = _cache.kernel_picks_path(json.loads(sys.argv[1])["device_kind"])
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
with tuning.reading():
    found = {rows: launches(step(), *operands(rows)) for rows in (13, 15, 40)}
    # Whether every launch of the step has a stored pick, every process taking process 0's (`_settle`).
    with tuning.seen() as traced:
        step().lower(*operands(15))
    settled = tuning._settle(traced, measurer=None) == 0
print(json.dumps([found, settled]))
"""


def test_a_pin_of_another_kind_within_a_pinned_block_keeps_the_blocks_pins():
    """A block pinning one kind of launch (`tuning.pinned`) within another pinning another: each launch of
    either kind takes its pin, the inner block's adding to the outer's (M-NA-5: a decode's inner pin of its
    attention forward dropped a lab's outer pin of its merged launch, and every arm the lab timed ran
    Triton's)."""
    with (tuning.pinned({norm_swiglu.RmsForward: norm_swiglu.RowsConfig(1, 1)}),
          tuning.pinned({norm_swiglu.SwigluForward: norm_swiglu.BlockConfig(4096, 8)})):
        nested = launches(step(), *operands(15))
    assert nested["rms_norm_forward"] == ((15,), 1) and nested["swiglu_forward"] == ((1, 1), 8), nested


def test_a_pin_at_a_launchs_key_holds_that_launch_alone():
    """A pin of a launch as the store keys it (`tuning.pinned`'s `(op, key)`) holds the launch at that key and
    no other of its kind: the 15-row norm takes it, the 40-row one its own heuristic, as unpinned (a config
    one launch of a kernel refuses is gated at the launch that takes it, `tests/tuned.each_config`)."""
    with tuning.seen() as traced:
        jax.make_jaxpr(step())(*operands(15))
    pin = next(tuning._launch_key(observed.keyed) for observed in traced
               if type(observed.op) is norm_swiglu.RmsForward)
    with tuning.pinned({pin: norm_swiglu.RowsConfig(1, 1)}):
        at_key, elsewhere = launches(step(), *operands(15)), launches(step(), *operands(40))
    assert at_key["rms_norm_forward"] == ((15,), 1) and elsewhere == launches(step(), *operands(40))


def test_a_pin_of_a_launch_holds_no_other_op_of_its_kind_at_its_key():
    """A pin of a launch as the store keys it (`tuning.pinned`'s `(op, key)`: its op whole) holds that op's
    launch alone: another op of its kind at the same key (a scaled attention forward beside its common one;
    here the norm's next revision) takes its own config, and the gate's pins (`tests/tuned.configs_by_kernel`)
    are each op's apart, so no config one refuses is dropped from the other's (Train 1426: pins keyed by
    the op's type held both, their spaces intersected)."""
    from tuned import configs_by_kernel

    x, scale = jax.ShapeDtypeStruct((15, 64), jnp.float32), jax.ShapeDtypeStruct((64,), jnp.float32)
    with tuning.seen() as traced:
        jax.make_jaxpr(plain)(x, scale)
    [observed] = [each for each in traced if type(each.op) is norm_swiglu.RmsForward]
    statics = {name: value for name, value in observed.arguments.arguments.items()
               if name not in ("x", "branch", "scale", "gate")}
    revised = observed.op.replace(revision=observed.op.revision + 1)

    def both(x, scale):
        return [op(x, None, scale, None, **statics) for op in (observed.op, revised)]

    pins = configs_by_kernel(both, x, scale, kernels=(norm_swiglu.RmsForward,))[norm_swiglu.RmsForward]
    assert {pin[0] for pin, _ in pins} == {tuning._identity(observed.op), tuning._identity(revised)}
    pin = tuning._launch_key(revised.keyed(observed.arguments))
    with tuning.pinned({pin: norm_swiglu.RowsConfig(1, 1)}), tuning.seen() as pinned:
        jax.make_jaxpr(both)(x, scale)
    assert [each.config for each in pinned] == [observed.config, norm_swiglu.RowsConfig(1, 1)]


def test_a_steps_observed_launches_are_rebuilt_whole_in_another_process():
    """A trace's launches kept past it (`Seen.held`), as a prepare's record keeps them for its GPU run
    (`arm_store`), pickled and read back: each launch, its config, arguments and digest as observed, its
    placement's abstract mesh alone (its concrete devices this process's), as a fresh process reads them."""
    import pickle

    mesh = jax.make_mesh((1,), ("data",), devices=jax.devices("cpu")[:1],
                         axis_types=(jax.sharding.AxisType.Explicit,))
    with tuning.seen() as traced, jax.set_mesh(mesh):
        jax.make_jaxpr(step())(*operands(15))
    held = traced.held()
    read = pickle.loads(pickle.dumps(held))
    assert len(read) == len(held) > 0
    assert [observed._replace(placed=None) for observed in read] == [
        observed._replace(placed=None) for observed in held]
    assert [observed.placed.abstract for observed in read] == [observed.placed.abstract for observed in held]
    assert all(observed.placed.concrete.empty for observed in read)
    assert not any(observed.placed.concrete.empty for observed in held)


def test_a_pools_processes_agree_on_their_reach_as_the_store_names_it():
    """A tuning run's processes agree on what their lowerings reach before any settles (`_agreed_reach`,
    through `_coordination.agreed_same`, which refuses a reach that differs, or a process failing to state
    its own, on every process alike): its launches as the store names them (their ops' whole identities and
    arguments: another revision of an op names another), their GPU models and the foreign kernels. A launch
    no JSON names (an object its static) fails as its reach is stated, within the agreement."""
    with tuning.seen() as traced:
        jax.make_jaxpr(step())(*operands(15))
    reached = tuning.Reach(tuple(traced), ("a foreign kernel",))
    stated = []
    tuning._agreed_reach(reached, agree=lambda phase, find: stated.append(find()))
    [(launches, models, foreign)] = stated
    launched = [named for named, _ in launches]
    op, ba = reached.launches[0].launch
    revised = tokamax.BoundArguments(op.replace(revision=op.revision + 1), ba.arguments)
    assert tuning._launch_json(ba) in launched and tuning._launch_json(revised) not in launched
    assert models == [jax.devices()[0].device_kind] and foreign == ["a foreign kernel"]
    static = next(name for name, value in ba.arguments.items() if name not in (*op.HOOKS, *op.ROWS)
                  and not isinstance(value, jax.ShapeDtypeStruct))
    unnamed = tokamax.BoundArguments(op, {**ba.arguments, static: object()})
    with pytest.raises(TypeError):
        tuning._agreed_reach(tuning.Reach((reached.launches[0]._replace(keyed=unnamed),), ()))


def test_a_launch_reached_for_two_gpu_models_is_refused_as_one_settlements():
    """One keyed launch traced for an A100 and for an H100 (each under its model's abstract mesh) is two
    distinct launches (`tuning._distinct`, a store's each), which one settlement refuses (`tuning._settle`'s
    `_one_kind`) on every process alike, where keeping the first alone settled one model's store for both."""
    x, scale = jax.ShapeDtypeStruct((15, 64), jnp.float32), jax.ShapeDtypeStruct((64,), jnp.float32)
    with tuning.seen() as traced:
        jax.make_jaxpr(plain)(x, scale)
    [observed] = [each for each in traced if type(each.op) is norm_swiglu.RmsForward]
    mesh = jax.make_mesh((1,), ("data",), axis_types=(jax.sharding.AxisType.Explicit,))
    both = [observed._replace(placed=observed.placed._replace(abstract=generation.gpu_mesh(mesh, name)))
            for name in ("sm80", "sm90")]
    assert len(tuning._distinct(both)) == 2 and len(tuning._distinct(both[:1] * 2)) == 1
    with pytest.raises(ValueError):
        tuning._settle(both, measurer=None)


def test_a_steps_launches_take_pins_stored_picks_and_heuristics(tmp_path):
    """The four launches of a norm seam and SwiGLU step are recorded as it
    traces (`tuning.seen`), keyed with their token axis at a power of two. Each
    takes its heuristic, a pinned config of its type (one of another type
    raises, naming the config and why, or within `pinned_where_taken` is
    set aside), and in a fresh
    process reading picks as an opt-in tuning run does (`tuning.reading`)
    the picks stored for its device beside its compilation cache (each a
    variant of its launch, `KernelOp.variants`), for every token count in
    the bucket, and at a token count of a bucket nothing stores its
    heuristic, a pick serving its own key alone, beside a launch of a kernel
    only another checkout has. Nothing measures on a CPU or beside other
    work on a GPU.
    A process under deterministic ops reads none of them and launches its
    heuristics, the same in every process while other runs store picks."""
    found, thirteen = recorded(step(), *operands(15)), recorded(step(), *operands(13))
    ops = {type(ba.op): ba for ba in found}
    assert set(ops) == {norm_swiglu.RmsForward, norm_swiglu.RmsBackward, norm_swiglu.SwigluForward,
                        norm_swiglu.SwigluBackward}
    keys = {kind: dict(ba.autotuning_cache_key) for kind, ba in ops.items()}
    assert keys[norm_swiglu.RmsForward]["x"].shape == (16, 64)
    assert keys[norm_swiglu.RmsBackward]["dy"].shape == (16, 64)
    assert keys[norm_swiglu.SwigluBackward]["gate"].shape == (16, 96)
    assert keys == {type(ba.op): dict(ba.autotuning_cache_key) for ba in thirteen}

    usual = launches(step(), *operands(15))
    # A pinned config of the launch's type wins; one of another type raises.
    with tuning.pinned({norm_swiglu.RmsForward: norm_swiglu.RowsConfig(1, 1),
                        norm_swiglu.RmsBackward: norm_swiglu.ReverseConfig(1, 256, 2),
                        norm_swiglu.SwigluForward: norm_swiglu.BlockConfig(4096, 8)}):
        pinned = launches(step(), *operands(15))
    assert pinned["rms_norm_forward"] == ((15,), 1) and pinned["rms_norm_backward"] == ((15,), 2)
    assert pinned["swiglu_forward"] == ((1, 1), 8) and pinned["swiglu_backward"] == usual["swiglu_backward"]
    with (tuning.pinned({norm_swiglu.RmsForward: norm_swiglu.BlockConfig(1024, 4)}),
          pytest.raises(tuning.Unpinnable) as refused):
        launches(step(), *operands(15))
    assert (refused.value.kernel, refused.value.config) == ("RmsForward", norm_swiglu.BlockConfig(1024, 4))
    # A launch made under a pin it cannot take (`tuning.pinned_where_taken`: a scaled attention forward
    # beside its common one) takes its own config, every other kind's pin kept; one it can take, the pin.
    rms = ops[norm_swiglu.RmsForward]
    for pin, rows in ((norm_swiglu.BlockConfig(1024, 4), usual["rms_norm_forward"]),
                      (norm_swiglu.RowsConfig(1, 1), ((15,), 1))):
        with (tuning.pinned({norm_swiglu.RmsForward: pin,
                             norm_swiglu.SwigluForward: norm_swiglu.BlockConfig(4096, 8)}),
              tuning.pinned_where_taken(rms.op, rms)):
            taken = launches(step(), *operands(15))
        assert taken["rms_norm_forward"] == rows and taken["swiglu_forward"] == ((1, 1), 8)
    # The op's own config (a measured candidate's) likewise: kept where it can take it, else its own.
    for own, kept in ((norm_swiglu.BlockConfig(1024, 4), None), (norm_swiglu.RowsConfig(1, 1),
                                                                 norm_swiglu.RowsConfig(1, 1))):
        with tuning.pinned_where_taken(rms.op.replace(config=own), rms) as taken_op:
            assert taken_op.config == kept

    # The picks a GPU measured, stored for this device, as a fresh process finds them: each launch's
    # variant beside its heuristic (16 rows and 4 warps a norm program, 4096 elements and 4 warps a SwiGLU's).
    picked = {norm_swiglu.RmsForward: norm_swiglu.RowsConfig(8, 4),
              norm_swiglu.RmsBackward: norm_swiglu.ReverseConfig(16, 1, 8),
              norm_swiglu.SwigluForward: norm_swiglu.BlockConfig(4096, 2),
              norm_swiglu.SwigluBackward: norm_swiglu.BlockConfig(4096, 8)}
    assert all(picked[type(ba.op)] in ba.op.keyed(ba).autotuning_configs for ba in found)
    timing = tokamax.BenchmarkData(compile_time_ms=1.0, lower_time_ms=1.0, evaluation_times_ms=(1.0,),
                                   metadata={})
    # A plain norm's pick beside them: a launch with no branch, which the seam's nearest search passes over.
    [alone] = recorded(plain, operands(15)[0], operands(15)[2])
    picks = tuning.dumps_picks(jax.devices()[0].device_kind, (
        *((ba, tuning.Timings({picked[type(ba.op)]: timing})) for ba in found),
        (alone, tuning.Timings({norm_swiglu.RowsConfig(16, 8): timing}))))
    flags = f"{keeping_picks()} {READ_ONLY}".strip()
    [path] = fresh(STORED, with_foreign(picks), cache_dir=tmp_path / "cache", flags=flags)
    assert path.startswith(str(tmp_path / "cache" / "autotune" / "jaxlib-")) and path.endswith(".json")
    stored, tuned = fresh(READ, cache_dir=tmp_path / "cache", flags=flags)
    heuristics = {str(rows): launches(step(), *operands(rows)) for rows in (13, 15, 40)}
    heuristics = json.loads(json.dumps(heuristics))
    for rows in ("13", "15"):
        assert stored[rows]["rms_norm_forward"] == [[-(-int(rows) // 8)], 4]
        assert stored[rows]["swiglu_forward"] == [[1, 1], 2]
        assert stored[rows]["swiglu_backward"] == [[1, 1], 8]
        assert stored[rows]["rms_norm_backward"] == [[1], 8]
    assert stored["40"] == heuristics["40"]
    assert tuned is True
    assert os.listdir(os.path.dirname(path)) == [os.path.basename(path)]
    repeatable, _ = fresh(READ, cache_dir=tmp_path / "cache", flags=f"{flags} {DETERMINISTIC_OPS}")
    assert repeatable == heuristics
    assert repeatable != stored


def test_a_lowering_records_each_launch_it_traces_again_where_its_trace_was_cached():
    """A step's lowering records each launch whose config picks decide as it traces it (`seen`), and
    a lowering whose trace was cached before records the same launches as a fresh step's: a step needs
    no lowering of its own to be tuned."""
    def recorded_by(function) -> list:
        with tuning.seen() as found:
            function.lower(*operands(15))
        return sorted((type(observed.op).__name__, repr(sorted(observed.keyed.autotuning_cache_key.items())))
                      for observed in found)

    fresh_step = recorded_by(step())
    function = step()
    function.lower(*operands(15))
    assert fresh_step and recorded_by(function) == recorded_by(function) == fresh_step


def test_a_launch_is_resolved_and_recorded_without_asking_a_traced_array_for_its_devices(monkeypatch):
    """Each launch of a step's lowering, the plain one and the one that
    records its launches for `tune` (`seen`), reads its config and its GPU model
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
    with tuning.seen():
        step().lower(*operands(15))
    step().lower(*operands(15))
    assert raised == []


DISTINCT = """
import json
import jax
from test_tuning import loss, operands, step
from dew.nn.kernels import generation, tuning


def layers(x, branch, scale, gate, up):
    return loss(x, branch, scale, gate, up) + loss(2 * x, branch, scale, gate, up)


def described(trace):
    # The distinct launches the traced program makes, as the pool decides them (`tuning._distinct`).
    with tuning.seen() as found:
        trace()
    return sorted([type(observed.op).__name__, repr(sorted(observed.keyed.autotuning_cache_key.items()))]
                  for observed in tuning._distinct(found))


with generation.assumed("sm80"):
    two = described(lambda: jax.make_jaxpr(jax.grad(layers, argnums=(0, 1, 2, 3, 4)))(*operands(15)))
    one = described(lambda: jax.make_jaxpr(step())(*operands(15)))
    with tuning.seen() as traced:
        jax.make_jaxpr(step())(*operands(15))
    distinct = sorted({(type(observed.op).__name__, repr(sorted(observed.keyed.autotuning_cache_key.items())))
                       for observed in traced})
print(json.dumps([two, one, [list(launch) for launch in distinct]]))
"""


def test_a_step_settles_each_distinct_launch_once(tmp_path):
    """Two layers of one shape launch the same kernels at the same keys.
    Tuned as on a GPU, the two-layer step's launches are the one layer's,
    and the one layer's are the distinct launches its trace makes, each
    once: the pool decides each distinct launch once, and a measuring pass
    measures it once, not once a layer."""
    two, one, distinct = fresh(DISTINCT, cache_dir=tmp_path / "cache", flags=keeping_picks())
    assert one == distinct and two == one


@pytest.mark.parametrize("stores, rows, norms", [
    ({0: {15: (8, 4)}}, (15,), [[[2], 4]]),
    ({1: {15: (8, 4)}}, (15,), [[[1], 4]]),
    ({0: {16: (8, 4)}, 1: {64: (32, 4)}}, (16, 64), [[[2], 4], [[4], 4]]),
], ids=["process 0 holds", "process 1 holds", "each key its own"])
def test_every_process_launches_process_zeros_config_for_a_launch_without_a_pick(tmp_path, stores, rows,
                                                                                   norms):
    """A launch takes its own key's pick as process 0 resolves it, else its
    heuristic, whatever another process stores: the program `tune` returns on
    each of two processes launches each norm alike. Where process 0 stores a
    15-token norm's pick, 8 rows a program (a variant of the heuristic's 16),
    the step's norm takes it on both; where process 1 does, the heuristic's
    one program of 4 warps. Where process 0 stores a 16-token pick and process
    1 a 64-token one, a step at 16 and 64 tokens takes process 0's pick at 16
    and the heuristic at 64 on both: process 1's 64-token norm, first traced
    with its own pick's 32 rows, traces again with process 0's decision."""
    timing = tokamax.BenchmarkData(compile_time_ms=1.0, lower_time_ms=1.0, evaluation_times_ms=(1.0,),
                                   metadata={})

    def keyed(count):
        # The launch as a step traced for the GPU keys it, which is how a measurement stores it.
        with generation.assumed("sm80"), tuning.seen() as found:
            jax.make_jaxpr(step())(*operands(count))
        [norm] = [observed.keyed for observed in found if type(observed.op) is norm_swiglu.RmsForward]
        return norm

    picks = tmp_path / "picks"
    picks.mkdir()
    for process, held in stores.items():
        stored = tuple((keyed(count), tuning.Timings({norm_swiglu.RowsConfig(*config): timing}))
                       for count, config in held.items())
        (picks / f"process{process}.json").write_text(tuning.dumps_picks(
            generation.gpu_target("sm80").device_kind, stored))
    reports = run_pool("settle", tmp_path / "pool", 2, script=Path(__file__).with_name("tuning_worker.py"),
                       cache=tmp_path / "cache", picks=picks, rows=",".join(map(str, rows)))
    assert reports[0] == reports[1]
    assert [[grid, warps] for name, grid, warps in reports[0] if name == "rms_norm_forward"] == norms


TRAINED = """
import json, os, sys
import jax, numpy as np, optax
import dew.nn.backbones
from dew import _cache
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.distributed.batches import shard_batch
from dew.training import Trainer

seq = int(sys.argv[1])
model = models.build("causal_transformer", vocab_size=4096, emb_features=1024, num_layers=2, num_heads=8,
                     mlp_features=3584, max_seq_len=seq)
trainer = Trainer(LMObjective(model, seq), optax.adam(1e-3), key=jax.random.key(0))
state, _, _ = trainer.place()
tokens = np.random.default_rng(0).integers(0, 4096, size=(4, seq + 1)).astype(np.int32)
trainer.compile(state, shard_batch(trainer.device_mesh, {"text": tokens}))
path = _cache.kernel_picks_path(jax.devices()[0].device_kind)
print(json.dumps([path, os.stat(path).st_mtime_ns if os.path.exists(path) else None]))
"""


MEASURED = """
import json, os, sys
import jax
from test_tuning import operands, step
from dew import _cache
from dew.nn.kernels import tuning
lowerings = []


def lower():
    lowerings.append(None)
    return step().lower(*operands(int(sys.argv[1])))


tuning.tune(lower, measure=True)
lowered = len(lowerings)
path = _cache.kernel_picks_path(jax.devices()[0].device_kind)
stamp = os.stat(path).st_mtime_ns if os.path.exists(path) else None
# Whether every launch of the step now has a stored pick, every process taking process 0's (`_settle`).
with tuning.reading():
    with tuning.seen() as traced:
        step().lower(*operands(int(sys.argv[1])))
    settled = tuning._settle(traced, measurer=None) == 0
print(json.dumps([path, stamp, settled, lowered]))
"""


def only_an_explicit_pass_on_a_gpu_to_itself_measures_and_it_measures_each_launch_once(tmp_path):
    """A language model's step compiled by the Trainer measures nothing. An
    explicit pass (`tune(..., measure=True)`) on a GPU it has to itself
    measures each launch over every variant (`KernelOp.variants`), stores every time
    and keeps the fastest; the next pass measures nothing, and one beside
    other work on its GPU (gpu-run -n 0.5) adds no picks for a new shape."""
    inherited = keeping_picks()
    if READ_ONLY in inherited:
        pytest.skip("measuring needs the GPU to itself: gpu-run without -n")
    path, untouched = fresh(TRAINED, "2048", cache_dir=tmp_path / "cache", flags=inherited)
    assert untouched is None
    # Another checkout's launch, which the pass reads past and keeps.
    found = recorded(step(), *operands(15))
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


class KernelRefusal(ValueError):
    """A kernel's own failure raised as a ValueError, as a product refusing the tiles it was given does."""


def a_config_failing_for_more_than_its_shared_memory_stops_the_measurement():
    """A launch one of whose configs raises as it lowers, a kernel's bug and not its GPU's shared
    memory: measuring the launch propagates the error, where the tuner had counted it one more
    failed config and kept the others' fastest; a ValueError too, which the tuner refuses only where
    CUPTI read none of a config's kernels (row 4446)."""
    [norm] = [ba for ba in recorded(step(), *operands(15)) if type(ba.op) is norm_swiglu.RmsForward]
    broken_config = sorted(norm.autotuning_configs, key=repr)[0]
    fields = {field.name: getattr(norm.op, field.name) for field in dataclasses.fields(norm.op) if field.init}
    for failure in (KernelBug, KernelRefusal):

        class Broken(norm_swiglu.RmsForward):
            """The norm's forward, raising for one of its variants. Its signature is the norm's, which
            tokamax reads to bind the launch's operands."""

            @functools.wraps(norm_swiglu.RmsForward._fwd)
            def _fwd(self, *args, config, failure=failure, **kwargs):
                if config == broken_config:
                    raise failure
                return super()._fwd(*args, config=config, **kwargs)

        broken = tokamax.BoundArguments(Broken(**fields), norm.arguments)
        with pytest.raises(failure):
            tuning._measured(broken.op, broken)


def a_launch_measured_beside_a_profile_capture_keeps_its_heuristic(tmp_path):
    """A profile capture holds CUPTI, which takes one subscriber: measuring a launch beside it times
    no config, returns each as failed, so the launch keeps its heuristic, and leaves CUPTI to the
    capture without raising; once the capture ends, the same launch measures every config."""
    [norm] = [ba for ba in recorded(step(), *operands(15)) if type(ba.op) is norm_swiglu.RmsForward]
    jax.profiler.start_trace(str(tmp_path / "trace"))
    try:
        beside = tuning._measured(norm.op, norm)
    finally:
        jax.profiler.stop_trace()
    assert set(beside) == set(norm.autotuning_configs) and not beside.prune_errors()
    alone = tuning._measured(norm.op, norm)
    assert set(alone.prune_errors()) == set(norm.autotuning_configs)


def a_pulled_launch_is_timed_with_its_backward():
    """The selective scan's launch measured on this GPU (`tuning._measured`) as a gradient calls it (its op
    pulled, `KernelOp.moded`) and as a forward alone: every config's timed kernels hold its backward's
    (`selective_scan_backward`) pulled and none forward, so a pulled pick weighs the backward its tokens a
    thread and residency set (row 12340: the tuner timed the forward alone)."""
    from dew.nn.kernels.templates import report
    from dew.nn.kernels.templates.recurrence import selective

    rows, length, channels, states = 2, 2048, 1024, 16
    specs = (jax.ShapeDtypeStruct((rows, length, channels), jnp.bfloat16),) * 2 + (
        jax.ShapeDtypeStruct((channels, states), jnp.float32),
        *(jax.ShapeDtypeStruct((rows, length, states), jnp.bfloat16),) * 2,
        jax.ShapeDtypeStruct((channels,), jnp.float32))
    with report.pulled():
        pulled = selective.SELECTIVE.moded()
    assert pulled.pulled
    for op, backward in ((selective.SELECTIVE, False), (pulled, True)):
        timings = tuning._measured(op, op.keyed(op.bind(*specs)))
        ran = {config: any("selective_scan_backward" in name for name in timed.metadata["kernels"])
               for config, timed in timings.items() if not isinstance(timed, Exception)}
        assert ran and set(ran.values()) == {backward}, (op.pulled, ran)


TWO_CPU_MEASURE = """
import json, os, threading, time
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
import jax
from tuned import recorded
from test_tuning import operands, step
from dew.nn.kernels import norm_swiglu, tuning

compile_ = jax.stages.Lowered.compile
held, running, most = threading.Lock(), [0], [0]


def compiled(lowered, *args, **kwargs):
    with held:
        running[0] += 1
        most[0] = max(most[0], running[0])
    try:
        time.sleep(0.05)
        return compile_(lowered, *args, **kwargs)
    finally:
        with held:
            running[0] -= 1


jax.stages.Lowered.compile = compiled
[norm] = [ba for ba in recorded(step(), *operands(15)) if type(ba.op) is norm_swiglu.RmsForward]
measured = tuning._measured(norm.op, norm)
print(json.dumps({"configs": len(norm.autotuning_configs), "measured": len(measured), "most": most[0]}))
"""
"""A launch measured in a process of two CPUs (`os.sched_setaffinity`), each of its compiles counted as JAX
runs it: the candidates' count, those measured and the most compiling at once."""


def a_launchs_measurement_compiles_its_candidates_in_the_processs_compile_slots():
    """A launch measured on this GPU (`tuning._measured`) in a process of two CPUs, whose compile slots are
    one (half its CPUs, `report.compile_threads`), compiles one candidate config at a time, as every compile
    of Dew's takes a slot: its pool compiled them raw, a compile a CPU the process may use, beside the slots
    a step's compiles take (base review of 58fdadba6, F4: four compiles at once on a four-vCPU quota, past
    its two slots)."""
    from process_support import fresh

    found = fresh(TWO_CPU_MEASURE)
    assert found["configs"] > 1 and found["measured"] == found["configs"], found
    assert found["most"] == 1, found


VERSIONED_OP = """
import dataclasses
from typing import ClassVar

from dew.nn.kernels import tuning

OPERANDS = {operands!r}


@dataclasses.dataclass(frozen=True)
class Rows:
    rows: int
    warps: int


@dataclasses.dataclass(frozen=True)
class Versioned(tuning.KernelOp):
    config_cls: ClassVar[type] = Rows

    def _fwd(self, {parameters}, *, config, return_residuals=False):
        return x, None

    def _get_heuristics_config(self, ba):
        return Rows(1, 4)
"""
"""One checkout's version of a kernel launch (`versioned_op.Versioned`): its operands (`OPERANDS`), which
another checkout's version of the same launch takes one more or fewer of."""


def versioned_op(root: Path, operands: tuple[str, ...]) -> Path:
    """A directory holding a version of `versioned_op` whose launch takes `operands`."""
    root.mkdir(parents=True)
    (root / "versioned_op.py").write_text(VERSIONED_OP.format(operands=operands,
                                                              parameters=", ".join(operands)))
    return root


STORES = """
import json, sys
sys.path.insert(0, sys.argv[2])
import jax
import tokamax
import versioned_op
from dew import _cache
from dew.nn.kernels import tuning
shape = jax.ShapeDtypeStruct((16, 64), jax.numpy.float32)
launch = versioned_op.Versioned().bind(*(shape for _ in versioned_op.OPERANDS))
timing = tokamax.BenchmarkData(compile_time_ms=1.0, lower_time_ms=1.0, evaluation_times_ms=(1.0,),
                               metadata={})
pick = tuning.Timings({versioned_op.Rows(*json.loads(sys.argv[1])): timing})
tuning._store(jax.devices()[0].device_kind, {tuning._launch_key(launch): (tuning._described(launch), pick)})
path = _cache.kernel_picks_path(jax.devices()[0].device_kind)
read, foreign = tuning._read(path)
picks = {json.dumps(sorted(ba.arguments)): [timings.fastest_config.rows, timings.fastest_config.warps]
         for ba, timings in read.values() if type(ba.op) is versioned_op.Versioned}
print(json.dumps([path, picks, len(foreign)]))
"""


def test_a_checkout_whose_kernel_drops_an_operand_keeps_the_others_picks_of_it(tmp_path):
    """Two checkouts share a store, each with its version of one launch: another branch's, whose kernel
    takes an operand more, and this one's. Each stores its pick of the launch through the store's writer
    and reads back only its own, and each write leaves the other's entry as it was."""
    flags = keeping_picks()
    cache_dir = tmp_path / "cache"
    wide = str(versioned_op(tmp_path / "wide", ("x", "shift")))
    narrow = str(versioned_op(tmp_path / "narrow", ("x",)))
    path, theirs, _ = fresh(STORES, "[4, 2]", wide, cache_dir=cache_dir, flags=flags)
    assert theirs == {json.dumps(["return_residuals", "shift", "x"]): [4, 2]}
    with open(path) as file:
        [entry] = json.load(file)["data"]
    _, ours, foreign = fresh(STORES, "[2, 8]", narrow, cache_dir=cache_dir, flags=flags)
    assert (ours, foreign) == ({json.dumps(["return_residuals", "x"]): [2, 8]}, 1)
    with open(path) as file:
        assert entry in json.load(file)["data"]
    _, theirs, foreign = fresh(STORES, "[4, 2]", wide, cache_dir=cache_dir, flags=flags)
    assert (theirs, foreign) == ({json.dumps(["return_residuals", "shift", "x"]): [4, 2]}, 1)


def test_a_launch_is_chosen_for_the_gpu_its_mesh_names():
    """`tuning._mesh_kind` of the context's placement, read through public jax as tokamax reads a launch's GPU
    for its cache: a launch traced under a mesh of abstract A100s (`generation.gpu_mesh`, `Trainer.lower_for`)
    is the A100's, whatever this host is; under none, the default device's."""
    mesh = jax.make_mesh((1,), ("data",), axis_types=(jax.sharding.AxisType.Explicit,))
    with jax.sharding.use_abstract_mesh(generation.gpu_mesh(mesh, "sm80")):
        assert tuning._mesh_kind(tuning.Placement.current()) == generation.gpu_target("sm80").device_kind
    assert tuning._mesh_kind(tuning.Placement.current()) == jax.devices()[0].device_kind


def test_a_launch_is_settled_against_the_target_it_traced_for():
    """A launch traced for a GPU of facts its own (`generation.assumed` of its target, `Trainer.lower_for`'s:
    an RTX 4080's 76 multiprocessors where its catalog model states 58) under that model's abstract mesh is
    settled at its placement (`tuning._placed_in`, where `_settle` reads and checks its picks and derives its
    config) under that target once the assumption has ended, and keyed by its model alone
    (`tuning._mesh_kind`); its placement rebuilt in another process (a prepare's record) keeps it. Restored
    to its mesh alone, it was held to its catalog model's facts (Train 1426)."""
    import pickle

    catalog = generation.gpu_target("NVIDIA GeForce RTX 4080")
    own = dataclasses.replace(catalog, multiprocessors=76)
    mesh = jax.make_mesh((1,), ("data",), axis_types=(jax.sharding.AxisType.Explicit,))
    with generation.assumed(own), jax.sharding.use_abstract_mesh(generation.gpu_mesh(mesh, own)):
        placed = tuning.Placement.current()
    assert catalog.multiprocessors != 76 and generation.assumption() is None
    for at in (placed, pickle.loads(pickle.dumps(placed))):
        with tuning._placed_in(at):
            assert generation.gpu_target().multiprocessors == 76
        assert tuning._mesh_kind(at) == catalog.device_kind


def _rtx_4080_placements() -> tuple[tuning.Placement, tuning.Placement]:
    """An RTX 4080's placement as its catalog states it (58 multiprocessors) and as a target assumed with 76
    (`Trainer.lower_for`'s own facts), both under its model's abstract mesh: one model, two GPUs' facts."""
    catalog = generation.gpu_target("NVIDIA GeForce RTX 4080")
    own = dataclasses.replace(catalog, multiprocessors=76)
    mesh = jax.make_mesh((1,), ("data",), axis_types=(jax.sharding.AxisType.Explicit,))
    placed = []
    for target in (catalog, own):
        with generation.assumed(target), jax.sharding.use_abstract_mesh(generation.gpu_mesh(mesh, target)):
            placed.append(tuning.Placement.current())
    return placed[0], placed[1]


class _Facted(norm_swiglu.RmsForward):
    """The norm's forward, its space and its configs read at the GPU in effect: on a GPU of 76
    multiprocessors its space lacks one of its configs (`UNSPACED`) and it refuses another (`REFUSED`) as a
    pin, a tile one GPU's budget holds and another's of the same model does not; each GPU its space and its
    configs were read at recorded (`asked`)."""

    REFUSED: ClassVar = None
    UNSPACED: ClassVar = None
    asked: ClassVar[list] = []

    def _get_autotuning_configs(self, ba):
        target = generation.gpu_target()
        _Facted.asked.append(target.multiprocessors)
        lacking = {_Facted.UNSPACED} if target.multiprocessors == 76 else set()
        return super()._get_autotuning_configs(ba) - lacking

    def pinnable(self, ba, config):
        target = generation.gpu_target()
        _Facted.asked.append(target.multiprocessors)
        refused = config == _Facted.REFUSED and target.multiprocessors == 76
        return not refused and super().pinnable(ba, config)


def _facted_calls(*placements: tuning.Placement, rows: int = 15) -> list[tuning.Observed]:
    """One call of `_Facted` at each of `placements`, of a step of `rows` rows (its key's 16 of 13 to 16):
    its keyed arguments its key's, its own the call's."""
    [norm] = [ba for ba in recorded(step(), *operands(16)) if type(ba.op) is norm_swiglu.RmsForward]
    with tuning.seen() as traced:
        jax.make_jaxpr(step())(*operands(rows))
    [own] = [observed.arguments for observed in traced if type(observed.op) is norm_swiglu.RmsForward]
    fields = {field.name: getattr(norm.op, field.name) for field in dataclasses.fields(norm.op) if field.init}
    op = _Facted(**fields)
    # Two of the configs an RTX 4080's catalog placement spaces, neither placement's heuristic (which every
    # space holds, tokamax's `autotuning_configs`), restricted at 76 multiprocessors.
    _Facted.REFUSED = _Facted.UNSPACED = None
    spaces, heuristics = [], set()
    for placed in _rtx_4080_placements():
        with tuning._placed_in(placed):
            probe = tokamax.BoundArguments(op, norm.arguments)
            spaces.append(probe.autotuning_configs)
            heuristics.add(probe.heuristics_config)
    _Facted.REFUSED, _Facted.UNSPACED = sorted(spaces[0] - heuristics, key=repr)[:2]
    keyed, arguments = tokamax.BoundArguments(op, norm.arguments), tokamax.BoundArguments(op, own.arguments)
    return [tuning.Observed(op, keyed, None, arguments, None, placed) for placed in placements]


def _timed(ms: float) -> tokamax.BenchmarkData:
    return tokamax.BenchmarkData(compile_time_ms=0.0, lower_time_ms=0.0, evaluation_times_ms=(ms,),
                                 metadata={})



def test_a_launch_measured_for_some_calls_keeps_the_timings_its_other_calls_take():
    """A launch whose held timings give no pick every call of a group takes, measured for that group
    (`tuning._settle`, `_measured`: the configs each of its calls takes), merges what it measures into what is
    held in this process (`_TIMINGS`), its store given what it measured alone (`Measurer.store`, which merges
    it into what the store holds as it writes, `_store`), so a later settlement of its other calls still
    takes their pick from the timings that give it (Train 1444's review: a narrower remeasurement replaced
    the whole entry, its other calls' fastest lost)."""
    catalog, own = _rtx_4080_placements()
    calls = _facted_calls(catalog, own)
    others = [config for config in calls[0].keyed.autotuning_configs
              if config not in (_Facted.REFUSED, _Facted.UNSPACED)]
    kind, (op, key) = tuning._mesh_kind(catalog), tuning._launch_key(calls[0].keyed)
    held = tuning._TIMINGS.setdefault((op, kind), tuning._Picks())
    # The fastest absent from the restricted placement's space, the next refused there: the pair of calls
    # takes none held.
    held[key] = tuning.Timings({_Facted.UNSPACED: _timed(1.0), _Facted.REFUSED: _timed(1.5)})
    # A GPU of the launch's model, its measurement timing one config its calls take, its store recorded.
    stored = []
    with tuning._placed_in(catalog):
        attached = generation.gpu_target()
    measurer = tuning.Measurer(kind, attached, lambda *_: tuning.Timings({others[0]: _timed(2.0)}),
                               lambda _, fresh: stored.append(fresh))
    picks = calls[0].op.get_autotuning_cache(kind)
    try:
        assert tuning._settle(calls, measurer=measurer) == 0
        assert picks[key].fastest_config == others[0]
        times = {config: timing.median_evaluation_time_ms for config, timing in held[key].items()}
        assert times == {_Facted.UNSPACED: 1.0, _Facted.REFUSED: 1.5, others[0]: 2.0}
        [fresh] = stored
        assert set(fresh[(op, key)][1]) == {others[0]}
        tuning._settle(calls[:1], measurer=None)
        assert picks[key].fastest_config == _Facted.UNSPACED
    finally:
        held.pop(key, None)
        picks.pop(key, None)


def test_a_launch_an_assumed_device_chose_is_timed_on_no_other_gpu(monkeypatch):
    """A launch traced under an assumed device and no mesh naming it (an RTX 4080's tiles traced on this
    host, `generation.assumed` alone) keys its picks by the attached device's model (`_mesh_kind`), and so
    reaches the attached GPU's measurer, but was chosen for the assumed GPU: it is timed on none other
    (`tuning._chosen_for`), nothing measured or stored, its launch left without a pick; the attached GPU's
    own launches still are (Train 1478's review: an sm80 launch was timed on a Hopper and stored in its
    picks)."""
    with generation.assumed(generation.gpu_target("NVIDIA GeForce RTX 4080")):
        assumed = tuning.Placement.current()
    attached = tuning.Placement.current()
    with generation.reassumed(None):
        here = generation.gpu_target()
    [call] = _facted_calls(assumed)
    measured, stored = [], []
    measurer = tuning.Measurer(tuning._mesh_kind(assumed), here,
                               lambda *args: measured.append(args) or tuning.Timings(),
                               lambda _, fresh: stored.append(fresh))
    assert tuning._mesh_kind(assumed) == tuning._mesh_kind(attached)
    assert tuning._settle([call], measurer=measurer) == 1
    assert (measured, stored) == ([], [])
    assert tuning._chosen_for(attached, here) and not tuning._chosen_for(assumed, here)
    # A process's measurer is the device that runs the timing, whatever GPU the caller's mesh names: under an
    # RTX 4080's abstract mesh, this host's own device's facts, which that mesh's launches were not chosen
    # for (the k01-69 review: the mesh's catalog facts were read as the attached GPU's).
    catalog, _ = _rtx_4080_placements()
    # The measuring arm's flags: picks kept (no deterministic ops) and the GPU its own (no read-only cache).
    monkeypatch.setenv("XLA_FLAGS", without_flags(keeping_picks(), READ_ONLY.split("=")[0]))
    with tuning._placed_in(catalog):
        measurer = tuning._measurer(measure=True)
    device = jax.devices()[0]
    assert measurer is not None and measurer.target == generation.device_target(device)
    assert not tuning._chosen_for(catalog, measurer.target)

def test_a_launchs_candidates_are_read_taken_and_compiled_at_its_placement():
    """A launch measured at its call's placement (`tuning._candidate_compiles`, `_measured`'s): its space is
    read there and each candidate asked of there (`KernelOp.pinnable`), against its GPU's facts, so a config
    that GPU's space lacks or refuses is not timed; and each compile runs on a worker of the compile pool at
    it, the placement's target assumed and its mesh in context, where they were read at the caller's
    context and compiled with none (Train 1444's review)."""
    catalog, own = _rtx_4080_placements()
    [call] = _facted_calls(own)
    _Facted.asked.clear()

    def compiled(config):
        return threading.get_ident(), generation.assumption(), jax.sharding.get_abstract_mesh()

    futures = tuning._candidate_compiles(call.op, call.keyed, [call], compiled)
    ran = [future.result() for future in futures.values()]
    assert ran and set(_Facted.asked) == {76}
    assert all(thread != threading.get_ident() for thread, _, _ in ran)
    assert all((assumed, mesh) == (own.assumption, own.abstract) for _, assumed, mesh in ran)
    assert {_Facted.REFUSED, _Facted.UNSPACED}.isdisjoint(futures)
    [plain] = _facted_calls(catalog)
    assert {_Facted.REFUSED, _Facted.UNSPACED} <= set(tuning._measurable(plain.op, plain.keyed, [plain]))


def test_a_pick_is_one_every_call_of_its_launch_takes_and_the_pool_agrees_on_their_calls():
    """One launch called at two placements of one GPU model, its catalog's and a target assumed with other
    facts (`_rtx_4080_placements`): settled from every timing its store holds (every config that ran, as its
    store is read, `_kernel_launches`), whatever the first call's space lacks, its pick is the fastest each
    call takes (`tuning._settle`, `_usable`) in either order of its calls, none where only one takes it, so
    no call falls back to its own derivation; a pick that changes at any call, the first's none before and
    after, lowers the step again (`KERNEL_TUNES`); and the pool's agreement on a reach (`_agreed_reach`)
    states each launch's calls, each by its own arguments and its placement's facts, so two processes calling
    it otherwise (at another placement, or another row count its key buckets) disagree before settling
    (Train 1444's review: a tile one placement's budget refused was picked for both, one rank falling
    back)."""
    catalog, own = _rtx_4080_placements()
    calls = _facted_calls(catalog, own)
    others = [config for config in calls[0].keyed.autotuning_configs
              if config not in (_Facted.REFUSED, _Facted.UNSPACED)]
    # The fastest absent from the restricted placement's space, the next refused there.
    stored = tuning.Timings({_Facted.UNSPACED: _timed(1.0), _Facted.REFUSED: _timed(1.5),
                             **{config: _timed(2.0) for config in others}})
    [(_, key, read)] = tuning._kernel_launches({tuning._launch_key(calls[0].keyed): (calls[0].keyed, stored)})
    assert set(read) == set(stored)
    assert tuning._usable(read, calls[:1]).fastest_config == _Facted.UNSPACED
    assert tuning._usable(tuning.Timings({_Facted.REFUSED: _timed(1.0)}), calls) is None
    picks = calls[0].op.get_autotuning_cache(tuning._mesh_kind(catalog))
    for ordered in (calls, calls[::-1]):
        picks[key] = read
        try:
            assert tuning._settle(ordered, measurer=None) == 0
            assert picks[key].fastest_config in others
        finally:
            picks.pop(key, None)
    # The restricted call first: it reads no pick of the fastest alone before or after, its peer one.
    tunes = _cache.KERNEL_TUNES.get_global()
    picks[key] = tuning.Timings({_Facted.UNSPACED: _timed(1.0)})
    try:
        assert tuning._settle(calls[::-1], measurer=None) == 1
        assert key not in picks and _cache.KERNEL_TUNES.get_global() == tunes + 1
    finally:
        picks.pop(key, None)

    # Two settlements of one key in turn, the second's calls taking none of the first's pick: each chooses
    # from every timing its store held (`_TIMINGS`), not the pick the first installed.
    held = tuning._TIMINGS.setdefault((tuning._launch_key(calls[0].keyed)[0], tuning._mesh_kind(catalog)),
                                      tuning._Picks())
    held[key] = read
    try:
        tuning._settle(calls[:1], measurer=None)
        assert picks[key].fastest_config == _Facted.UNSPACED
        tuning._settle(calls, measurer=None)
        assert picks[key].fastest_config in others
    finally:
        held.pop(key, None)
        picks.pop(key, None)

    stated = []
    for reached in ([catalog], [catalog, own], [catalog, catalog]):
        rows = [15, 13][:len(reached)]
        observed = [call for placed, count in zip(reached, rows, strict=True)
                    for call in _facted_calls(placed, rows=count)]
        tuning._agreed_reach(tuning.Reach(tuple(observed), ()), agree=lambda _, state: stated.append(state()))
    assert stated[0] != stated[1] and stated[0] != stated[2] and stated[1] != stated[2]


def _scaled_by(factor: float):
    """A norm epilogue scaling its rows by `factor`: each made a function of its own, at its own address."""
    def epilogue(rows):
        return rows * factor

    return epilogue


def test_a_calls_signature_names_its_hooks_as_their_programs_the_same_in_every_process():
    """A call as the pool's processes agree on it (`tuning._call_signature`, `_agreed_reach`'s): its hooks
    named by what they compute, as its key names them (`KernelOp.keyed`), so two calls of equivalent
    epilogues made apart (each a function at its own address, as each process makes its own) are named
    alike and another epilogue otherwise (Train 1444's review: a hook's `repr`, its address, made every
    rank's plan its own)."""
    [call] = _facted_calls(_rtx_4080_placements()[0])

    def hooked(factor):
        hooked = {**call.arguments.arguments, "epilogue": _scaled_by(factor)}
        arguments = tokamax.BoundArguments(call.op, hooked)
        return tuning._call_signature(call._replace(keyed=call.op.keyed(arguments), arguments=arguments))

    assert hooked(2.0) == hooked(2.0) != hooked(3.0)


def test_a_calls_signature_holds_its_nested_operands_at_their_own_shapes():
    """A call's signature (`tuning._call_signature`) holds an operand nested in a tuple at the call's own
    shape (a grouped product's gathered table, `gather=(table, None)`), so two calls of one key whose tables
    differ in length are two calls a pick is held to, not one (Train 1444's review)."""
    [call] = _facted_calls(_rtx_4080_placements()[0])

    def gathered(rows):
        keyed = {**call.keyed.arguments, "gate": (jax.ShapeDtypeStruct((16, 64), jnp.float32), None)}
        own = {**call.arguments.arguments, "gate": (jax.ShapeDtypeStruct((rows, 64), jnp.float32), None)}
        return tuning._call_signature(call._replace(keyed=tokamax.BoundArguments(call.keyed.op, keyed),
                                                    arguments=tokamax.BoundArguments(call.op, own)))

    assert gathered(13) == gathered(13) != gathered(15)


def test_a_settlement_reads_no_timing_where_tokamaxs_cache_is_ignored():
    """With tokamax's cache ignored (`--tokamax_ignore_autotuning_cache`), as a launch's own read takes no
    pick, a settlement chooses none from the timings its store holds (`tuning._timings_of`): its calls keep
    their own derivations, and a measuring run measures them, where history was chosen from (Train 1444's
    review)."""
    catalog, _ = _rtx_4080_placements()
    [call] = _facted_calls(catalog)
    identity, key = tuning._launch_key(call.keyed)
    held = tuning._TIMINGS.setdefault((identity, tuning._mesh_kind(catalog)), tuning._Picks())
    held[key] = tuning.Timings({config: _timed(1.0) for config in call.keyed.autotuning_configs})
    try:
        assert tuning._timings_of(call) is not None
        with tokamax_config.ignore_autotuning_cache("true"):
            assert tuning._timings_of(call) is None
            assert tuning._settle([call], measurer=None) == 1
    finally:
        held.pop(key, None)


class _Holding:
    """A hook object whose call reads an array it holds: a cache keyed on it would keep the array alive."""

    def __init__(self, array):
        self.array = array

    def __call__(self, x):
        return x * self.array.shape[0]


@dataclasses.dataclass(frozen=True)
class _Scaled:
    """A hook of a value alone: equal for equal factors, which a cache may hold."""

    by: float

    def __call__(self, x):
        return x * self.by


@dataclasses.dataclass(frozen=True)
class _Peaked:
    """A hook of a value and a mapping its hash leaves out, unchangeable as a GPU target's peaks are
    (`generation.Frozen`), or not."""

    by: float
    peaks: Mapping = dataclasses.field(default_factory=lambda: generation.Frozen({"bf16": 1.0}), hash=False,
                                       compare=False)

    def __call__(self, x):
        return x * self.by


def _doubled(x):
    return 2 * x


def _defaulted(array, *, keyword: bool, listed: bool = False):
    """A hook function of no closure (each name it reads its own) holding `array` as a default of its own,
    positional or keyword-only, alone or in a list."""
    if listed:
        held = [array]

        def hook(x, *, by=held):
            return x * by[0].shape[0]
    elif keyword:
        def hook(x, *, by=array):
            return x * by.shape[0]
    else:
        def hook(x, by=array):
            return x * by.shape[0]
    return hook


@dataclasses.dataclass
class _Options:
    """A hook's options, a parent among them, which may be the options themselves."""

    factor: float
    parent: object = None


@dataclasses.dataclass
class _Lazy:
    """A hook's options, one field set only once it is computed (`init=False`), unset here."""

    factor: float
    computed: float = dataclasses.field(init=False, repr=False, compare=False)


@dataclasses.dataclass(frozen=True)
class _LazyHook:
    """A hook object, one field set only once it is computed (`init=False`), unset here."""

    factor: float
    computed: float = dataclasses.field(init=False, repr=False, compare=False)

    def __call__(self, x):
        return x * self.factor


def _lazily_defaulted():
    """A hook function of no closure whose keyword-only default leaves a field of its own unset."""
    options = _Lazy(2.0)

    def hook(x, *, by=options):
        return x * by.factor

    return hook


def _cyclic(*, listed: bool):
    """A hook function of no closure whose default holds itself: a list holding itself, read by its length,
    or options whose parent is themselves, read by their factor."""
    if listed:
        ring: list = []
        ring.append(ring)

        def hook(x, *, by=ring):
            return x * len(by)
    else:
        options = _Options(2.0)
        options.parent = options

        def hook(x, *, by=options):
            return x * by.factor
    return hook


@pytest.mark.parametrize(("made", "cached", "made_here"),
                         [(lambda: _doubled, True, False), (lambda: _Scaled(2.0), True, True),
                          (lambda: _Holding(jnp.ones(3)), False, True), (lambda: jnp.tanh, False, False),
                          (lambda: _defaulted(jnp.ones(3), keyword=False), False, True),
                          (lambda: _defaulted(jnp.ones(3), keyword=True), False, True),
                          (lambda: _defaulted(jnp.ones(3), keyword=True, listed=True), False, True),
                          (lambda: _cyclic(listed=True), False, True),
                          (lambda: _cyclic(listed=False), False, True), (_lazily_defaulted, False, True),
                          (lambda: _LazyHook(2.0), False, True), (lambda: _Peaked(2.0), True, True),
                          (lambda: _Peaked(2.0, generation.Frozen({"bf16": jnp.ones(3)})), False, True),
                          (lambda: _Peaked(2.0, {"bf16": 1.0}), False, True),
                          (lambda: _Peaked(2.0, MappingProxyType({"bf16": 1.0})), False, True)],
                         ids=["function", "dataclass", "array-holding-object", "jitted-callable",
                              "array-default", "array-keyword-default", "listed-array-default",
                              "cyclic-list-default", "cyclic-dataclass-default", "unset-field-default",
                              "unset-field-object", "unchangeable-mapping-field", "array-mapping-field",
                              "mutable-mapping-field", "proxied-mapping-field"])
def test_a_launch_lowered_once_keeps_its_program_only_for_a_hook_it_may_hold(made, cached, made_here):
    """`tuning.lowered_once` keeps a launch's program for a static hook it may hold (a function of no closure,
    a dataclass of values, an unchangeable mapping of values a field holds among them, as a GPU target's
    peaks: Train 1429's red, a decoder's layers each lowering their own recurrence kernels), holding the
    hook: its second call traces nothing (JAX's trace events). One that may reach an array or change after
    it is held (a callable object's attributes, which no check sees, a jitted function's among them; a
    function's default, keyword-only or not, alone or in a list; a mapping holding an array, a mutable one or
    a view of one; a default holding itself, or one leaving a field of its own unset, or such an object
    itself) runs the launch itself, and once the caller lets it go nothing the launch kept holds it. Every
    call computes the hook's value. (Each hook made in the test: an array drawn as the module loads would
    start JAX's backend before a worker process joins its pool, `tuning_worker`.)"""
    hook = made()

    @tuning.lowered_once(("hook",))
    def launch(x, *, hook):
        return hook(x)

    x = jnp.arange(4.0)
    first = launch(x, hook=hook)
    with compiles.Compiles(None) as again:
        second = jax.block_until_ready(launch(x, hook=hook))
    assert jnp.array_equal(first, hook(x)) and jnp.array_equal(second, first)
    if cached:
        assert not [program for program in again.programs if program.kind == "traced"]
    if made_here:
        # Let go by the caller: the kept program holds a hook it keys on; a launch run uncached holds none.
        kept = weakref.ref(hook)
        del hook
        gc.collect()
        assert (kept() is not None) == cached


@pytest.mark.parametrize("viewed", [False, True], ids=["list", "read-only view of a dict"])
def test_a_launch_whose_hooks_mutable_default_changes_computes_it_as_it_is_then(viewed):
    """A hook function of no closure reading a default that may change (a list, or a read-only view of a
    dict its caller keeps), changed after the launch's first call (`tuning.lowered_once`): the second call
    computes with it as it is then, the launch run itself at each call (`holdable` refuses what may change
    after it is held), where a held trace computed the first call's value (Train 1429's review: a walk
    admitting a list's values, or a view's, held it)."""
    scales = {0: 2.0} if viewed else [2.0]
    default = MappingProxyType(scales) if viewed else scales

    def hook(x, *, by=default):
        return x * by[0]

    @tuning.lowered_once(("hook",))
    def launch(x, *, hook):
        return hook(x)

    x = jnp.arange(4.0)
    assert jnp.array_equal(launch(x, hook=hook), 2 * x)
    scales[0] = 3.0
    assert jnp.array_equal(launch(x, hook=hook), 3 * x)

def test_an_unchangeable_mapping_is_changed_by_nothing_it_gives_out():
    """`generation.Frozen`, the one mapping a launch's cache holds as a value (`tuning.holdable`): its
    reconstruction's data (`__reduce__`, a pickle's) is a copy, which changed leaves it as it was; its items
    are read-only, its backing never set again or deleted, and a second `__init__` leaves it as it was (Train
    1444's review: a slot set anew changed a held mapping); and a pickle of it rebuilds it equal."""
    import pickle

    frozen = generation.Frozen({0: 2.0})
    _, (data,) = frozen.__reduce__()
    data[0] = 3.0
    for change in (lambda: operator.setitem(frozen._items, 0, 3.0),
                   lambda: setattr(frozen, "_items", {0: 3.0}), lambda: delattr(frozen, "_items")):
        with pytest.raises(TypeError):
            change()
    frozen.__init__({0: 3.0})
    assert frozen[0] == 2.0 and pickle.loads(pickle.dumps(frozen)) == frozen == {0: 2.0}

def test_a_launch_whose_trace_reads_a_value_of_the_trace_around_it_runs_itself():
    """A launch whose trace reads a value of the trace around it that is none of its operands (one its caller
    left where the launch reads it) is no function of its key, so no trace of it is held: it runs itself, each
    call reading its own, where its held trace ran without the value it read and raised ("foreach() argument
    2 is shorter than argument 1": an sm120 attention training step's fused backward, Main's G4 run)."""
    around = []

    @tuning.lowered_once(())
    def launch(x):
        return x + around[-1]

    def stepped(scale: float):
        @jax.jit
        def step(x, y):
            around.append(y * scale)
            return launch(x)
        return step

    # Two steps traced apart over the same operands: the second takes the verdict the first held.
    x = jnp.arange(4.0)
    for scale in (1.0, 2.0):
        assert jnp.array_equal(stepped(scale)(x, jnp.float32(1.0)), x + scale)


def _halved_score(scores):
    """A module's score hook (the prepared-hook witness's)."""
    return scores * 0.5 + 1.0


def test_a_prepared_hook_pickles_as_its_source_and_is_rebuilt_by_the_constructor_that_made_it():
    """A prepared hook pickles as its source, rebuilt by the constructor that made it (`Prepared.__reduce__`):
    one the pass rewrites (`prepared`) runs the pass's form again, one taken as prepared (`prepared_as`) its
    source itself, each equal to the hook pickled and computing what it did, so a launch a GPU tier part's
    prepare held is its run's (rows 9582 on: a held launch's score hook refused the truths store)."""
    import pickle

    scores = jnp.arange(8.0).reshape(2, 4)
    rewritten, taken = tuning.prepared(_halved_score), tuning.prepared_as(_halved_score)
    loaded_rewritten, loaded_taken = (pickle.loads(pickle.dumps(made)) for made in (rewritten, taken))
    assert loaded_rewritten == rewritten and loaded_taken == taken
    assert loaded_rewritten.run is not loaded_rewritten.source and loaded_taken.run is loaded_taken.source
    for loaded, made in ((loaded_rewritten, rewritten), (loaded_taken, taken)):
        assert jnp.array_equal(loaded(scores), made(scores))


def _foreign(lowered) -> tuple[str, ...]:
    # The foreign kernels a run of `lowered` may reach, no launch traced (`tuning.reach`).
    return tuning.reach(tuning.Seen(), lowered).foreign


def test_a_tpu_step_of_another_librarys_kernels_alone_is_named_untunable_by_its_compiler():
    """A TPU step none of whose kernel launches is Dew's, another library's Pallas TPU kernel alone (JAX's
    Splash attention's kind), lowered for a TPU here: a measuring tuning run of it would measure nothing it
    asked for, so its kernels are named by their compiler (`tuning.reach`'s foreign, which `tune` refuses
    with as `Untunable`), never passed by as a GPU's Triton kernels alone were (row 13534); in a branch of a
    conditional a run may take as in the step's own flow, not in one its trace decided against (whose arms JAX
    lowers both). The same kernel behind a while loop of no trips (a
    counted one's, or one whose condition its trace decided false), which no run reaches though JAX lowers
    it, beside a step's own XLA work, names none (row 13559). A step of
    Dew's own Mosaic GPU launches (the attention template's, lowered for an H100) names none: each call
    carries the launch that traced it (`KernelCall.launched`), whatever its kernel's name reads."""
    from jax.experimental import pallas as pl
    from kernel_compile import lowered_for

    from dew.nn.attention import scaled_dot_product_attention
    from dew.nn.kernels.lowered import kernel_calls

    def doubled(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2

    def step(x):
        return pl.pallas_call(doubled, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype))(x)

    rows = jax.ShapeDtypeStruct((8, 128), jnp.float32)

    def foreign_of(fn):
        return _foreign(jax.jit(fn).trace(rows).lower(lowering_platforms=("tpu",)))

    foreign = foreign_of(step)
    assert foreign and all(name.startswith("mosaic_tpu: ") for name in foreign), foreign
    assert "mosaic_tpu: " in str(tuning.Untunable(foreign))
    # Taken where the data says: a run may take either branch.
    assert foreign_of(lambda x: jax.lax.cond(x[0, 0] > 0, step, lambda y: y + 1, x)) == foreign
    # A conditional its trace decided (a Python predicate's), both its arms lowered: its selected arm alone.
    for taken in (False, True):
        decided = jax.jit(lambda x, taken=taken: jax.lax.cond(taken, step, lambda y: y + 1, x)).trace(
            rows).lower(lowering_platforms=("tpu",))
        assert any(call.kind == "mosaic_tpu" for call in kernel_calls(decided, unrun=True))
        assert _foreign(decided) == (foreign if taken else ())
    # Behind a loop of no trips, a counted one's (from 0 while below 0) or one whose condition its trace
    # decided false, beside the step's own work: lowered all the same, no run launches it. (A fori_loop or
    # scan of no trips JAX lowers no body for at all.)
    def counted(x):
        return jax.lax.while_loop(lambda held: held[0] < 0, lambda held: (held[0] + 1, step(held[1])),
                                  (0, x))[1] * 2 + 1

    for never in (counted, lambda x: jax.lax.while_loop(lambda _: False, step, x) * 2 + 1):
        lowered_never = jax.jit(never).trace(rows).lower(lowering_platforms=("tpu",))
        assert any(call.kind == "mosaic_tpu" for call in kernel_calls(lowered_never, unrun=True))
        assert _foreign(lowered_never) == ()

    def attended(query, key, value):
        return scaled_dot_product_attention(query, key, value, causal=True, implementation="template")

    shapes = [jax.ShapeDtypeStruct(shape, jnp.bfloat16) for shape in ((1, 512, 4, 64), (1, 512, 2, 64),
                                                                       (1, 512, 2, 64))]
    owned = lowered_for(attended, shapes, "sm90")
    assert any(call.kind == "mosaic_gpu" for call in kernel_calls(owned, unrun=True))
    assert _foreign(owned) == ()


def test_a_steps_reach_holds_its_live_launch_and_not_its_dead_one():
    """A step tracing the fused RMS norm's launch at two shapes, one in its own flow and one in a loop no run
    enters, reaches the live one alone (`tuning.reach`, what `tune` settles and tools/kernel_picks.py
    counts), though both traced, and no foreign kernel."""
    from kernel_compile import live_and_dead, lowered_for

    shapes = [jax.ShapeDtypeStruct((256, 64), jnp.float32), jax.ShapeDtypeStruct((64,), jnp.float32),
              jax.ShapeDtypeStruct((256, 128), jnp.float32), jax.ShapeDtypeStruct((128,), jnp.float32)]
    with generation.assumed("sm90"), tuning.seen() as found:
        lowered = lowered_for(live_and_dead(plain), shapes, "sm90")
        # The live launch's own step, its key the one a reach holds.
        with tuning.seen() as alone:
            lowered_for(plain, shapes[:2], "sm90")
    reached = tuning.reach(found, lowered)
    assert len(tuning._distinct(found)) == 2 and reached.foreign == ()
    assert {observed.placed.abstract.abstract_device.device_kind for observed in reached.launches} == {
        generation.gpu_target("sm90").device_kind}
    assert [observed.keyed.autotuning_cache_key for observed in reached.launches] == [
        observed.keyed.autotuning_cache_key for observed in tuning._distinct(alone)]


def test_a_steps_reach_holds_its_live_forward_and_not_its_dead_reverse_launch_of_one_config():
    """A step running the selective scan's launch (a Mosaic GPU template's, lowered for a compile-only H100)
    and tracing, in a loop no run enters, the same launch as reverse mode runs it (`report.pulled`: its op
    pulled, its digest its mode's), both pinned to one config: its reach (`tuning.reach`) holds the forward
    alone, its call stating its launch whole (its caller, mode and all), where a join on the launches'
    identity and config, which their modes share, held the dead reverse launch too, for `tune` to time and
    store (Train 1426)."""
    from kernel_compile import lowered_for

    from dew.nn.kernels.templates import report
    from dew.nn.kernels.templates.recurrence import selective

    rows, length, channels, states = 1, 2048, 256, 16
    operands = (jax.ShapeDtypeStruct((rows, length, channels), jnp.bfloat16),) * 2 + (
        jax.ShapeDtypeStruct((channels, states), jnp.float32),
        *(jax.ShapeDtypeStruct((rows, length, states), jnp.bfloat16),) * 2,
        jax.ShapeDtypeStruct((channels,), jnp.float32))

    def pulled(held):
        with report.pulled():
            return (selective.SELECTIVE(*held)[0].astype(held[0].dtype), *held[1:])

    def stepped(*given):
        dead = jax.lax.while_loop(lambda _: False, pulled, given)
        return selective.SELECTIVE(*given)[0], dead[0]

    with generation.assumed("sm90"):
        tiles = selective.SELECTIVE.derived(selective.SELECTIVE.bind(*operands), generation.gpu_target())
        with tuning.pinned({selective.Selective: tiles}), tuning.seen() as found:
            lowered = lowered_for(stepped, operands, "sm90")
    joined = {(report.traced(observed.digest).identity, observed.config) for observed in found}
    assert len(joined) == 1 and {observed.op.pulled for observed in tuning._distinct(found)} == {False, True}
    assert [observed.op.pulled for observed in tuning.reach(found, lowered).launches] == [False]


def test_a_dew_launch_no_run_reaches_leaves_a_steps_foreign_kernels_untunable(monkeypatch):
    """A measuring tuning run of a step (`tuning.tune`, on an assumed H100) whose Dew launch, the fused RMS
    norm's Triton kernel (named for its launch by `launches.pallas_call`), is traced only in a loop no run
    enters, beside another library's Triton kernel the step runs, is refused as the foreign kernel's alone
    (`Untunable`), where counting every traced launch passed it by (row 13715); the same launch in the step's
    own flow is reached and the step lowered. Measurement off at the device's boundary (XLA's read-only
    autotune cache, `dew._cache.measures_picks`)."""
    from jax.experimental import pallas as pl
    from kernel_compile import lowered_for

    monkeypatch.setenv("XLA_FLAGS", f"{without_flags(keeping_picks(), READ_ONLY.split('=')[0])} {READ_ONLY}")

    def doubled(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2

    def foreign(x):
        return pl.pallas_call(doubled, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype))(x)

    def unreached(x, scale):
        held = jax.lax.while_loop(lambda _: False, lambda held: (plain(*held), held[1]), (x, scale))
        return foreign(held[0])

    def reached(x, scale):
        return foreign(plain(x, scale))

    shapes = [jax.ShapeDtypeStruct((256, 64), jnp.float32), jax.ShapeDtypeStruct((64,), jnp.float32)]
    with generation.assumed("sm90"):
        with pytest.raises(tuning.Untunable):
            tuning.tune(lambda: lowered_for(unreached, shapes, "sm90"), measure=True)
        # Its program and its reach, the launch a run makes (`tuning.tuned`), no further lowering to count it.
        tuned = tuning.tuned(lambda: lowered_for(reached, shapes, "sm90"), measure=True)
        assert tuned.lowered is not None and len(tuned.reached.launches) == 1 and tuned.reached.foreign


