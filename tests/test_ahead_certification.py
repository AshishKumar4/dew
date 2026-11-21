"""A step built ahead for compile-only GPUs certifies and lowers its programs as the run on those GPUs does
(`dew.training.ahead`, `dew.nn.kernels.templates.report`), so the run compiles none of them again."""

import os

from process_support import fresh


def test_a_build_for_compile_only_gpus_certifies_its_starts_launches(tmp_path):
    """A step built for a compile-only GPU in a process that sees none traces its start (`Trainer._started`)
    for that GPU, as the run on it does: the initializer's norm and SwiGLU launches are certified in the
    build, so the run reads their verdicts and certifies nothing on its GPU (rows 7322, 8299: five init-shape
    launches certified at the run)."""
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
    ops = result["ops"]
    assert "RmsForward" in ops and "SwigluForward" in ops, ops


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
    result = fresh(script, cache_dir=tmp_path / "cache", platforms="cpu",
                   variables={"CUDA_VISIBLE_DEVICES": ""}, flags=f"{os.environ.get('XLA_FLAGS', '')} --xla_gpu_autotune_level=0".strip())
    assert result["certified"] > 0 and result["cold_digest"] == result["warm_digest"], result
