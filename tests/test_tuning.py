"""The kernel configs' whole path: a step's launches found in its lowered
program, keyed by bucketed shape, resolved from a pin, a stored pick or a
heuristic, stored and read back in a fresh process, and measured once on a
GPU that a process has to itself (dew.nn.kernels.tuning)."""

import json
import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import pytest
import tokamax

from dew.nn.attention import stream_rms_normalized
from dew.nn.kernels import norm_swiglu, tuning
from dew.nn.moe import gated_product

STEP = """
import jax, jax.numpy as jnp
from dew.nn.attention import stream_rms_normalized
from dew.nn.moe import gated_product

def loss(x, branch, scale, gate, up):
    stream, normed = stream_rms_normalized(x, branch, scale, 1e-6, x.dtype, True, False, True,
                                           implementation="pallas")
    return jnp.sum(stream * normed) + jnp.sum(gated_product("swiglu", implementation="pallas")(gate, up))

step = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4)))
"""


def loss(x, branch, scale, gate, up):
    stream, normed = stream_rms_normalized(x, branch, scale, 1e-6, x.dtype, scale_offset=True,
                                           scale_after_cast=False, fp32_statistics=True,
                                           implementation="pallas")
    return jnp.sum(stream * normed) + jnp.sum(gated_product("swiglu", implementation="pallas")(gate, up))


def operands(rows: int, width: int = 64, hidden: int = 96):
    keys = jax.random.split(jax.random.key(0), 5)
    return (jax.random.normal(keys[0], (rows, width)), jax.random.normal(keys[1], (rows, width)),
            jax.random.normal(keys[2], (width,)), jax.random.normal(keys[3], (rows, hidden)),
            jax.random.normal(keys[4], (rows, hidden)))


def launches(function, *args) -> dict[str, tuple[tuple[int, ...], int]]:
    """Each Pallas launch's grid and warps in `function`'s program, by kernel name."""
    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "pallas_call":
                yield eqn
            for value in eqn.params.values():
                for inner in value if isinstance(value, (list, tuple)) else [value]:
                    inner = getattr(inner, "jaxpr", inner)
                    if hasattr(inner, "eqns"):
                        yield from walk(inner)

    return {str(eqn.params["name"]): (tuple(eqn.params["grid_mapping"].grid),
                                      eqn.params["compiler_params"].num_warps)
            for eqn in walk(jax.make_jaxpr(function)(*args).jaxpr)}


def test_a_steps_launches_take_pins_stored_picks_and_heuristics(tmp_path):
    """The four launches of a norm seam and SwiGLU step are found in its
    lowered program, keyed with their token axis at a power of two. Each
    takes its heuristic, then each end of its space when pinned, and a
    fresh process takes the picks stored for its device beside its
    compilation cache, for every token count in the bucket. A CPU measures
    nothing."""
    step = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4)))
    found = tokamax.autotuning.get_bound_args(step, *operands(15))
    ops = {type(ba.op): ba for ba in found}
    assert set(ops) == {norm_swiglu.RmsForward, norm_swiglu.RmsBackward, norm_swiglu.SwigluForward,
                        norm_swiglu.SwigluBackward}
    keys = {kind: dict(ba.autotuning_cache_key) for kind, ba in ops.items()}
    assert keys[norm_swiglu.RmsForward]["x"].shape == (16, 64)
    assert keys[norm_swiglu.RmsBackward]["dy"].shape == (16, 64)
    assert keys[norm_swiglu.SwigluBackward]["gate"].shape == (2048,)
    assert keys == {type(ba.op): dict(ba.autotuning_cache_key)
                    for ba in tokamax.autotuning.get_bound_args(step, *operands(13))}

    usual = launches(jax.grad(loss, argnums=(0, 1, 2, 3, 4)), *operands(15))
    assert usual["rms_norm_forward"] == ((1,), 4) and usual["swiglu_forward"] == ((2,), 4)
    # A pinned config its space holds wins; one it does not hold (64 rows of a 64-wide norm) is passed over.
    with tuning.pinned({norm_swiglu.RmsForward: norm_swiglu.RowsConfig(1, 1),
                        norm_swiglu.RmsBackward: norm_swiglu.ReverseConfig(1, 256, 2),
                        norm_swiglu.SwigluForward: norm_swiglu.BlockConfig(4096, 8)}):
        pinned = launches(jax.grad(loss, argnums=(0, 1, 2, 3, 4)), *operands(15))
    assert pinned["rms_norm_forward"] == ((15,), 1) and pinned["rms_norm_backward"] == ((15,), 2)
    assert pinned["swiglu_forward"] == ((1,), 8) and pinned["swiglu_backward"] == usual["swiglu_backward"]
    with tuning.pinned({norm_swiglu.RmsForward: norm_swiglu.RowsConfig(64, 2)}):
        assert launches(jax.grad(loss, argnums=(0, 1, 2, 3, 4)), *operands(15)) == usual

    # The picks a GPU measured, stored for this device, as a fresh process finds them.
    picked = {norm_swiglu.RmsForward: norm_swiglu.RowsConfig(4, 2),
              norm_swiglu.RmsBackward: norm_swiglu.ReverseConfig(2, 256, 8),
              norm_swiglu.SwigluForward: norm_swiglu.BlockConfig(512, 2),
              norm_swiglu.SwigluBackward: norm_swiglu.BlockConfig(256, 1)}
    timing = tokamax.BenchmarkData(compile_time_ms=1.0, lower_time_ms=1.0, evaluation_times_ms=(1.0,),
                                   metadata={})
    kind = jax.devices()[0].device_kind
    picks = tokamax.AutotuningResult(kind, tuple(
        (ba, tuning.AutotuningData({picked[type(ba.op)]: timing})) for ba in found))
    # Read only, as beside other work on a GPU: a process that lowers the step measures nothing.
    environment = {**os.environ, "JAX_COMPILATION_CACHE_DIR": str(tmp_path / "cache"),
                   "XLA_FLAGS": "--xla_gpu_experimental_autotune_cache_mode=AUTOTUNE_CACHE_MODE_READ"}
    store = textwrap.dedent("""
        import json, os, tokamax
        from dew import cache
        path = cache.kernel_picks_path(tokamax.AutotuningResult.loads(TEXT).device_kind)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").write(TEXT)
        print(json.dumps(path))
    """).replace("TEXT", repr(picks.dumps()))
    path = json.loads(subprocess.run([sys.executable, "-c", store], env=environment, check=True,
                                     capture_output=True, text=True).stdout.splitlines()[-1])
    assert path.startswith(str(tmp_path / "cache" / "autotune" / "jaxlib-")) and path.endswith(".json")
    read = STEP + textwrap.dedent("""
        import json, sys
        sys.path.insert(0, %r)
        from test_tuning import launches, operands
        from dew.nn.kernels import tuning
        print(json.dumps({rows: launches(step, *operands(rows)) for rows in (13, 15)}
                         | {"tuned": tuning.tune(step, *operands(15))}))
    """) % os.path.dirname(__file__)
    stored = json.loads(subprocess.run([sys.executable, "-c", read], env=environment, check=True,
                                       capture_output=True, text=True).stdout.splitlines()[-1])
    for rows in ("13", "15"):
        assert stored[rows]["rms_norm_forward"] == [[-(-int(rows) // 4)], 2]
        assert stored[rows]["rms_norm_backward"] == [[-(-int(rows) // 2)], 8]
        assert stored[rows]["swiglu_backward"] == [[-(-int(rows) * 96 // 256)], 1]
    assert stored["tuned"] is False
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
    environment = {**os.environ, "JAX_COMPILATION_CACHE_DIR": str(tmp_path / "cache"), "XLA_FLAGS": ""}

    def trained(seq: int, **extra):
        run = subprocess.run([sys.executable, "-c", TRAINED, str(seq)], env={**environment, **extra},
                             check=True, capture_output=True, text=True)
        return json.loads(run.stdout.splitlines()[-1])

    path, measured = trained(2048)
    with open(path) as file:
        stored = tokamax.AutotuningResult.load(file)
    kinds = {type(ba.op) for ba, _ in stored.data}
    assert {norm_swiglu.RmsForward, norm_swiglu.RmsBackward} <= kinds
    for ba, timings in stored.data:
        assert len(timings) > 1, type(ba.op)
        assert all(data.median_evaluation_time_ms > 0 for data in timings.values())
        assert timings.fastest_config in ba.autotuning_configs
    assert trained(2048) == [path, measured]
    assert trained(1024, XLA_FLAGS="--xla_gpu_experimental_autotune_cache_mode=AUTOTUNE_CACHE_MODE_READ") == [
        path, measured]
