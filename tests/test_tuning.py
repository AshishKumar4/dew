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
from dew.nn.attention import stream_rms_normalized
from dew.nn.kernels import norm_swiglu, tuning
from dew.nn.moe import gated_product

READ_ONLY = "--xla_gpu_experimental_autotune_cache_mode=AUTOTUNE_CACHE_MODE_READ"
"""The flag `gpu-run -n 0.5` adds beside other work on a GPU, under which no process measures."""


def loss(x, branch, scale, gate, up):
    stream, normed = stream_rms_normalized(x, branch, scale, 1e-6, x.dtype, scale_offset=True,
                                           scale_after_cast=False, fp32_statistics=True,
                                           implementation="pallas")
    return jnp.sum(stream * normed) + jnp.sum(gated_product("swiglu", implementation="pallas")(gate, up))


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
import tokamax
from dew import cache
from dew.files import write_atomically
picks = tokamax.AutotuningResult.loads(sys.argv[1])
path = cache.kernel_picks_path(picks.device_kind)
os.makedirs(os.path.dirname(path), exist_ok=True)
write_atomically(path, sys.argv[1])
print(json.dumps([path]))
"""

READ = """
import json
from test_tuning import launches, operands, step
from dew.nn.kernels import tuning
print(json.dumps([{rows: launches(step(), *operands(rows)) for rows in (13, 15)},
                  tuning.tune(lambda: step().lower(*operands(15)))]))
"""


def test_a_steps_launches_take_pins_stored_picks_and_heuristics(tmp_path):
    """The four launches of a norm seam and SwiGLU step are found in its
    lowered program, keyed with their token axis at a power of two. Each
    takes its heuristic, a pinned config its space holds, and in a fresh
    process the picks stored for its device beside its compilation cache,
    for every token count in the bucket. Nothing measures on a CPU or beside
    other work on a GPU."""
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
    picks = tokamax.AutotuningResult(jax.devices()[0].device_kind, tuple(
        (ba, tuning.AutotuningData({picked[type(ba.op)]: timing})) for ba in found))
    flags = f"{os.environ.get('XLA_FLAGS', '')} {READ_ONLY}".strip()
    [path] = fresh(STORED, picks.dumps(), cache_dir=tmp_path / "cache", flags=flags)
    assert path.startswith(str(tmp_path / "cache" / "autotune" / "jaxlib-")) and path.endswith(".json")
    stored, tuned = fresh(READ, cache_dir=tmp_path / "cache", flags=flags)
    for rows in ("13", "15"):
        assert stored[rows]["rms_norm_forward"] == [[-(-int(rows) // 4)], 2]
        assert stored[rows]["rms_norm_backward"] == [[-(-int(rows) // 2)], 8]
        assert stored[rows]["swiglu_backward"] == [[-(-int(rows) * 96 // 256)], 1]
    assert tuned is False
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


@pytest.mark.gpu
def test_a_trainer_on_a_gpu_to_itself_measures_its_launches_once_and_a_shared_one_reads(tmp_path):
    """A language model's step compiled by the Trainer on a GPU it has to
    itself measures each of its launches over every config of its space,
    stores every time and keeps the fastest, and compiles with them. The next
    process compiling that step measures nothing, and one beside other work
    on its GPU (gpu-run -n 0.5) compiles a new shape and adds no picks."""
    if not cache.measures_picks():
        pytest.skip("measuring needs the GPU to itself: gpu-run without -n")
    inherited = os.environ.get("XLA_FLAGS", "")
    path, measured = fresh(TRAINED, "2048", cache_dir=tmp_path / "cache", flags=inherited)
    with open(path) as file:
        stored = tokamax.AutotuningResult.load(file)
    kinds = {type(ba.op) for ba, _ in stored.data}
    assert {norm_swiglu.RmsForward, norm_swiglu.RmsBackward} <= kinds
    for ba, timings in stored.data:
        assert len(timings) > 1, type(ba.op)
        assert all(data.median_evaluation_time_ms > 0 for data in timings.values())
        assert timings.fastest_config in ba.autotuning_configs
    assert fresh(TRAINED, "2048", cache_dir=tmp_path / "cache", flags=inherited) == [path, measured]
    assert fresh(TRAINED, "1024", cache_dir=tmp_path / "cache",
                 flags=f"{inherited} {READ_ONLY}".strip()) == [path, measured]
