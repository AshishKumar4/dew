"""The kernel configs' whole path: a step's launches found in its lowered
program, keyed by bucketed shape, resolved from a pin, a stored pick or a
heuristic, stored and read back in a fresh process, and measured once on a
GPU that a process has to itself (dew.nn.kernels.tuning)."""

import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import pytest
import tokamax
from process_support import REPO_ROOT
from reference_error import equations

from dew import cache
from dew.nn.attention import rms_normalized, stream_rms_normalized
from dew.nn.kernels import norm_swiglu, tuning
from dew.nn.moe import gated_product

READ_ONLY = "--xla_gpu_experimental_autotune_cache_mode=AUTOTUNE_CACHE_MODE_READ"
"""The flag `gpu-run -n 0.5` adds beside other work on a GPU, under which no process measures."""


def loss(x, branch, scale, gate, up):
    stream, normed = stream_rms_normalized(x, branch, scale, 1e-6, x.dtype, scale_offset=True,
                                           scale_after_cast=False, fp32_statistics=True,
                                           implementation="pallas")
    return jnp.sum(stream * normed) + jnp.sum(gated_product("swiglu", implementation="pallas")(gate, up))


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
    graph = jax.make_jaxpr(function)(*args).jaxpr
    return {str(eqn.params["name"]): (tuple(eqn.params["grid_mapping"].grid),
                                      eqn.params["compiler_params"].num_warps)
            for eqn in equations(graph) if eqn.primitive.name == "pallas_call"}


def fresh(script: str, *arguments: str, cache_dir, flags: str) -> list:
    """`script`'s last printed JSON, run by a fresh interpreter on this checkout
    and these tests, compiling into `cache_dir` under XLA `flags`."""
    environment = {**os.environ, "JAX_COMPILATION_CACHE_DIR": str(cache_dir), "XLA_FLAGS": flags,
                   "PYTHONPATH": os.pathsep.join((str(REPO_ROOT / "src"), str(REPO_ROOT / "tests")))}
    run = subprocess.run([sys.executable, "-c", script, *arguments], env=environment, check=False,
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-3000:]
    return json.loads(run.stdout.splitlines()[-1])


STORED = """
import json, os, sys
from dew import cache
from dew.files import write_atomically
path = cache.kernel_picks_path(json.loads(sys.argv[1])["device_kind"])
os.makedirs(os.path.dirname(path), exist_ok=True)
write_atomically(path, sys.argv[1])
print(json.dumps([path]))
"""

FOREIGN = "dew.nn.kernels.another_branch.Kernel"
"""A kernel another checkout stores and this one cannot read."""


def with_foreign(text: str) -> str:
    """A store's JSON with two more launches this checkout cannot read: one of a kernel it lacks, and
    one of a kernel it has whose stored arguments lack an operand (another checkout's signature of
    it, as before a launch gained an operand). The first is last."""
    stored = json.loads(text)
    entry = json.loads(json.dumps(stored["data"][0]))
    entry[0]["op"]["__type"] = FOREIGN
    unbound = json.loads(json.dumps(stored["data"][0]))
    del unbound[0]["arguments"][next(iter(unbound[0]["arguments"]))]
    return json.dumps({**stored, "data": [*stored["data"], unbound, entry]})


READ = """
import json
from test_tuning import launches, operands, step
from dew.nn.kernels import tuning
print(json.dumps([{rows: launches(step(), *operands(rows)) for rows in (13, 15, 40)},
                  tuning.settled(tuning.tune(lambda: step().lower(*operands(15))))]))
"""


def test_a_steps_launches_take_pins_stored_picks_and_heuristics(tmp_path):
    """The four launches of a norm seam and SwiGLU step are found in its
    lowered program, keyed with their token axis at a power of two. Each
    takes its heuristic, a pinned config its space holds, and in a fresh
    process the picks stored for its device beside its compilation cache,
    for every token count in the bucket, and a token count stored for no
    bucket the nearest bucket's, beside a launch of a kernel only another
    checkout has. Nothing measures on a CPU or beside other work on a GPU."""
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
    # A pinned config its space holds wins; one it does not hold (64 rows of a 64-wide norm) is passed over.
    with tuning.pinned({norm_swiglu.RmsForward: norm_swiglu.RowsConfig(1, 1),
                        norm_swiglu.RmsBackward: norm_swiglu.ReverseConfig(1, 256, 2),
                        norm_swiglu.SwigluForward: norm_swiglu.BlockConfig(4096, 8)}):
        pinned = launches(step(), *operands(15))
    assert pinned["rms_norm_forward"] == ((15,), 1) and pinned["rms_norm_backward"] == ((15,), 2)
    assert pinned["swiglu_forward"] == ((1,), 8) and pinned["swiglu_backward"] == usual["swiglu_backward"]
    with tuning.pinned({norm_swiglu.RmsForward: norm_swiglu.RowsConfig(64, 2)}):
        assert launches(step(), *operands(15)) == usual

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
    picks = tokamax.AutotuningResult(jax.devices()[0].device_kind, (
        *((ba, tuning.AutotuningData({picked[type(ba.op)]: timing})) for ba in found),
        (alone, tuning.AutotuningData({norm_swiglu.RowsConfig(2, 1): timing}))))
    flags = f"{os.environ.get('XLA_FLAGS', '')} {READ_ONLY}".strip()
    [path] = fresh(STORED, with_foreign(picks.dumps()), cache_dir=tmp_path / "cache", flags=flags)
    assert path.startswith(str(tmp_path / "cache" / "autotune" / "jaxlib-")) and path.endswith(".json")
    stored, tuned = fresh(READ, cache_dir=tmp_path / "cache", flags=flags)
    for rows in ("13", "15", "40"):
        assert stored[rows]["rms_norm_forward"] == [[-(-int(rows) // 4)], 2]
        assert stored[rows]["swiglu_backward"] == [[-(-int(rows) * 96 // 256)], 1]
    for rows in ("13", "15"):
        assert stored[rows]["rms_norm_backward"] == [[-(-int(rows) // 2)], 8]
    assert tuned is True
    assert os.listdir(os.path.dirname(path)) == [os.path.basename(path)]


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
launches = tuning.tune(lambda: step().lower(*operands(int(sys.argv[1]))), measure=True)
path = cache.kernel_picks_path(jax.devices()[0].device_kind)
stamp = os.stat(path).st_mtime_ns if os.path.exists(path) else None
print(json.dumps([path, stamp, tuning.settled(launches)]))
"""


@pytest.mark.gpu
def test_only_an_explicit_pass_on_a_gpu_to_itself_measures_and_it_measures_each_launch_once(tmp_path):
    """A language model's step compiled by the Trainer measures nothing. An
    explicit pass (`tune(..., measure=True)`) on a GPU it has to itself
    measures each launch over every config of its space, stores every time
    and keeps the fastest; the next pass measures nothing, and one beside
    other work on its GPU (gpu-run -n 0.5) adds no picks for a new shape."""
    if not cache.measures_picks():
        pytest.skip("measuring needs the GPU to itself: gpu-run without -n")
    inherited = os.environ.get("XLA_FLAGS", "")
    path, untouched = fresh(TRAINED, "2048", cache_dir=tmp_path / "cache", flags=inherited)
    assert untouched is None
    # Another checkout's launch, which the pass reads past and keeps.
    with tuning.launches():
        found = tokamax.autotuning.get_bound_args(step(), *operands(15))
    timing = tokamax.BenchmarkData(compile_time_ms=1.0, lower_time_ms=1.0, evaluation_times_ms=(1.0,),
                                   metadata={})
    seed = tokamax.AutotuningResult(jax.devices()[0].device_kind, (
        (found[0], tuning.AutotuningData({found[0].heuristics_config: timing})),))
    foreign = json.loads(with_foreign(seed.dumps()))["data"][-2:]
    fresh(STORED, json.dumps({**json.loads(seed.dumps()), "data": foreign}), cache_dir=tmp_path / "cache",
          flags=inherited)
    path, measured, settled = fresh(MEASURED, "2048", cache_dir=tmp_path / "cache", flags=inherited)
    assert settled is True
    with open(path) as file:
        written = json.load(file)
    assert all(entry in written["data"] for entry in foreign)
    stored = tokamax.AutotuningResult.loads(json.dumps(
        {**written, "data": [entry for entry in written["data"] if entry not in foreign]}))
    kinds = {type(ba.op) for ba, _ in stored.data}
    assert {norm_swiglu.RmsForward, norm_swiglu.RmsBackward} <= kinds
    for ba, timings in stored.data:
        assert len(timings) > 1, type(ba.op)
        assert all(data.median_evaluation_time_ms > 0 for data in timings.values())
        assert timings.fastest_config in ba.autotuning_configs
    assert fresh(MEASURED, "2048", cache_dir=tmp_path / "cache", flags=inherited) == [path, measured, True]
    assert fresh(MEASURED, "1024", cache_dir=tmp_path / "cache",
                 flags=f"{inherited} {READ_ONLY}".strip()) == [path, measured, False]
