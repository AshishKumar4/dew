"""A step built ahead for compile-only GPUs certifies and lowers its programs as the run on those GPUs does
(`dew.training.ahead`, `dew.nn.kernels.templates.report`), so the run compiles none of them again."""

import os

from generation_support import COMPILES_ON_CPU
from process_support import fresh


def test_a_build_for_compile_only_gpus_starts_with_no_launch_to_certify(tmp_path):
    """A step built for a compile-only GPU in a process that sees none traces its start (`Trainer._started`)
    for that GPU, as the run on it does: its model's draw of its variables selects no kernel
    (`generation.drawing`), so neither the build nor the run on its GPU certifies a launch at the draw's
    shapes, which no step runs (rows 7322, 8299: five init-shape launches certified at the run; row 11803:
    the draw's certifications were 29% of an every-template build's main thread)."""
    script = """
import json
from test_compile_ahead import BATCH, SEQ_LEN, VOCAB, fine_tune
from token_support import token_rows
from dew.nn.kernels.generation import compile_only_gpus
from dew.nn.kernels.templates import report
from dew.training import MeshSpec
from dew.training.ahead import _targeted

target = MeshSpec().build(compile_only_gpus("sm90", 1, 80 * 2**30))
trainer = fine_tune(target, xla=True, split=2, dtype="bfloat16")
batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
with _targeted(trainer, batch, what="test", platforms=(frozenset({"gpu"}),)):
    ops = [c.op for c in report.certifications()]
print(json.dumps({"ops": ops}))
"""
    result = fresh(script, cache_dir=tmp_path / "cache", platforms="cpu",
                   variables={"CUDA_VISIBLE_DEVICES": ""})
    assert result["ops"] == [], result["ops"]


def test_a_program_lowered_while_its_launches_certify_aside_lowers_as_a_warm_one(tmp_path):
    """Two equal layers lowered while their launches certify aside (`report.certifying_aside`), the first
    capture pending as the second traces, lower to the program a warm process lowers, one function each:
    a build's cold lowering and a run's warm one share their digest, so the run takes what the build compiled
    (rows 7322, 8299, 8372: `validated_loss` and `jit__trainable_cotangent` compiled again at the run)."""
    script = """
import json
import jax
import jax.numpy as jnp
from dew.nn.kernels.generation import assumed, built_target, compile_only_gpus
from dew.nn.kernels.options import owned_program, program_digest
from dew.nn.kernels.templates import report
from dew.nn.norms import rms_normalized

(gpu,) = compile_only_gpus("sm90", 1, 80 * 2**30, multiprocessors=132)
mesh = jax.sharding.Mesh([gpu], ("x",))
placed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
x = jax.ShapeDtypeStruct((128, 32), jnp.bfloat16, sharding=placed)
w = jax.ShapeDtypeStruct((32,), jnp.float32, sharding=placed)
stated = built_target("sm90", gpu.core_count, 80 * 2**30)

def two_layers(x, w):
    y = rms_normalized(x, w, 1e-6, jnp.bfloat16, False, False, True)
    return rms_normalized(y, w, 1e-6, jnp.bfloat16, False, False, True)

with assumed(stated, portable=True), report.certifying(gpu), jax.set_mesh(mesh):
    with report.certifying_aside() as aside:
        cold = owned_program(two_layers).trace(x, w).lower()
    report.settled_aside(aside)
    with report.certifying_aside() as aside2:
        warm = owned_program(two_layers).trace(x, w).lower()
    report.settled_aside(aside2)

print(json.dumps({"cold_digest": program_digest(cold), "warm_digest": program_digest(warm),
                  "certified": len(report.certifications())}))
"""
    untuned = f"{os.environ.get('XLA_FLAGS', '')} --xla_gpu_autotune_level=0".strip()
    result = fresh(script, cache_dir=tmp_path / "cache", platforms="cpu",
                   variables={"CUDA_VISIBLE_DEVICES": ""}, flags=untuned)
    assert result["certified"] > 0 and result["cold_digest"] == result["warm_digest"], result


@COMPILES_ON_CPU
def test_a_steps_planner_certifies_its_launches_beside_the_trace_that_reaches_them(tmp_path):
    """A step compiled ahead for a compile-only H100 (`compile_ahead`: its planner's trace of each candidate,
    `FitSearch`, and its fit check's compile), in a process of its own over a cache of its own, certifies its
    attention launches beside the trace that reaches them (`report.certified`, as a program traces): each
    launch's own compile, a `certified_` program, runs on a certification thread (`dew-certify`), none on a
    thread that traces. Certified within the planner's trace, a 4x1024 MoE step's 16 certifications compiled
    one after another (MC-6740-POOL), and an sm90 plan certified 64 of its 82 seconds in turn (FS-P5324-MAIN,
    row 5324)."""
    script = """
import json
import threading
import jax
import optax
from model_support import TINY_DECODER
from test_compile_ahead import BATCH, TINY_SHARD, VOCAB, gpus
from token_support import token_rows
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.training import Layout, MeshSpec, Trainer
from dew.training.ahead import compile_ahead

compiled_on = []

def heard(event, seconds, *, fun_name="", **_):
    if event == "/jax/core/compile/backend_compile_duration":
        compiled_on.append((threading.current_thread().name, fun_name))

tokens = 256
model = models.build("causal_transformer", vocab_size=VOCAB,
                     **{**TINY_DECODER, "emb_features": 256, "max_seq_len": tokens}, num_kv_heads=2,
                     dtype="bfloat16")
trainer = Trainer(LMObjective(model, tokens), optax.adam(1e-3), key=jax.random.key(0),
                  mesh=MeshSpec().build(gpus("sm90")), layout=Layout(min_shard=TINY_SHARD))
jax.monitoring.register_event_duration_secs_listener(heard)
started = bool(compile_ahead(trainer, token_rows(VOCAB, BATCH, tokens + 1)).starts)
print(json.dumps({"started": started, "compiled_on": compiled_on}))
"""
    result = fresh(script, cache_dir=tmp_path / "cache", platforms="cpu",
                   variables={"CUDA_VISIBLE_DEVICES": ""})
    certifying = [thread for thread, name in result["compiled_on"] if "certified_" in name]
    assert result["started"], result
    assert certifying and all(thread.startswith("dew-certify") for thread in certifying), result
