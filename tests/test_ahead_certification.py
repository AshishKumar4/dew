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

from dew.nn.kernels.generation import chosen

# The fine-tune's attention XLA's (`chosen`), as `fine_tune(xla=True)` named it before row 11938.
with chosen(attention="xla"):
    target = MeshSpec().build(compile_only_gpus("sm90", 1, 80 * 2**30))
    trainer = fine_tune(target, split=2, dtype="bfloat16")
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
    with report.certifying_aside():
        cold = owned_program(two_layers).trace(x, w).lower()
    with report.certifying_aside():
        warm = owned_program(two_layers).trace(x, w).lower()

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
    row 5324). Its start traced first alone, as a trainer's start traces it (`Trainer._started`, targeting
    its mesh, no device named to certify on), then the plan: every launch certified on the one GPU the plan
    certifies on, its mesh's first, where a start's launches traced outside the plan's certifying block were
    certified again on another (rows 6791, 6824, 7212)."""
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
from dew.training import Layout, MeshSpec, Trainer, TrainerAdvanced
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
                  mesh=MeshSpec().build(gpus("sm90")),
                  advanced=TrainerAdvanced(layout=Layout(min_shard=TINY_SHARD)))
jax.monitoring.register_event_duration_secs_listener(heard)
from dew.nn.kernels.generation import assumed
from dew.nn.kernels.templates import report
with assumed("sm90"):
    trainer._started()
started = bool(compile_ahead(trainer, token_rows(VOCAB, BATCH, tokens + 1)).starts)
certified = [[each.op, str(each.device)] for each in report.certifications()]
print(json.dumps({"started": started, "compiled_on": compiled_on, "certified": certified,
                  "first": str(trainer.device_mesh.devices.flat[0])}))
"""
    result = fresh(script, cache_dir=tmp_path / "cache", platforms="cpu",
                   variables={"CUDA_VISIBLE_DEVICES": ""})
    certifying = [thread for thread, name in result["compiled_on"] if "certified_" in name]
    assert result["started"], result
    assert certifying and all(thread.startswith("dew-certify") for thread in certifying), result
    # Every launch certified on one device, the mesh's first compile-only GPU, the start's and the plan's
    # alike, never the host's: a certification is one a launch and device, so none twice (rows 6791, 6824,
    # 7212).
    assert result["certified"] and {each[1] for each in result["certified"]} == {result["first"]}, result

def test_a_scanned_compile_ahead_carries_the_unstack_its_fit_returns_through(tmp_path):
    """A one-stage scanned run compiled ahead (`compile_ahead` with `fit=`, as the GPU test's QUIET_BUILD
    builds it) carries in `AheadOfTime.starts` the donated `_layered_entries` unstack programs
    `Trainer.fit` runs as it returns its state on its stored layers, so a run loading it compiles none of them
    (row 7322: a loading A100 run compiled them on its GPU)."""
    script = """
import dataclasses
import json
import jax
from test_compile_ahead import BATCH, SEQ_LEN, VOCAB, fine_tune
from token_support import token_rows
from dew.data import Memory
from dew.distributed.placement import placed_shapes
from dew.nn.kernels import options
from dew.nn.kernels.generation import assumed, chosen, compile_only_gpus
from dew.training import Inspection, MeshSpec
from dew.training.ahead import compile_ahead

gen, sms, mem, share = "sm80", 108, 40 * 2**30, 0.75
with chosen(attention="xla"):
    target = MeshSpec().build(compile_only_gpus(gen, 1, mem, multiprocessors=sms, share=share))
    trainer = fine_tune(target, split=2, inspection=Inspection(every=2), dtype="bfloat16")
    validation = {"val": Memory(token_rows(VOCAB, 2 * BATCH, SEQ_LEN + 1, seed=1)).batch(BATCH)}
    fit = {"steps": 6, "log_every": 5, "eval_every": 3, "validation": validation}
    ahead = compile_ahead(trainer, token_rows(VOCAB, BATCH, SEQ_LEN + 1), fit=fit)
    a100 = dataclasses.replace(ahead.target, device_kind="NVIDIA A100-SXM4-40GB")
    with assumed(a100, portable=False):
        start = trainer._started()
        placed = placed_shapes(start.abstract, start.shardings)
        with options.lowering(("cuda",)) as lowered:
            trainer._staging.stored(dataclasses.replace(placed, compute=None), consumed=True)
digests = [options.stated_digest(program) for _, program in lowered]
print(json.dumps({"unstacks": len(digests), "carried": [digest in ahead.starts for digest in digests]}))
"""
    untuned = f"{os.environ.get('XLA_FLAGS', '')} --xla_gpu_autotune_level=0".strip()
    result = fresh(script, cache_dir=tmp_path / "cache", platforms="cpu",
                   variables={"CUDA_VISIBLE_DEVICES": ""}, flags=untuned)
    assert result["unstacks"] >= 2 and all(result["carried"]), result


def test_a_step_compiled_ahead_is_refused_by_a_run_that_compiles_under_other_options():
    """A step compiled ahead is held to the XLA options it compiled under (a compile's limit aside) by its
    one fingerprint (`program_digest` under them): a run of its program that compiles under an option it
    took none of, a scoped `compiled_with` one, refuses it, as its own compile would have sent XLA that
    option (row 13553: it loaded it, the option dropped); under its own options, a compile's limit beside
    them, the run goes on to load it (here refused as no process's to load, `built` None)."""
    import jax
    import jax.numpy as jnp
    import pytest

    from dew.nn.kernels.options import program_digest
    from dew.training.ahead import AheadOfTime, Executable, NotTheStepCompiledAhead
    from dew.training.memory import LIMIT_EXIT

    lowered = jax.jit(lambda x: jnp.sin(x) * 2).lower(jnp.ones(4))
    options = {"xla_gpu_enable_triton_gemm": False}
    ahead = AheadOfTime("sm90", 1, {}, program_digest(lowered, options), Executable(b"", ()), None, 0, 0, 0,
                        0, 0, {})
    with pytest.raises(NotTheStepCompiledAhead):
        ahead.taken(lowered, {**options, "xla_gpu_enable_command_buffer": ""})
    with pytest.raises(ValueError) as loading:
        ahead.taken(lowered, {**options, LIMIT_EXIT: 1 << 30})
    assert not isinstance(loading.value, NotTheStepCompiledAhead), loading.value


def test_a_step_saved_with_its_options_apart_from_its_program_is_compiled_again(tmp_path):
    """A step an earlier Dew compiled ahead, its program's fingerprint without its options and the options
    saved beside it (train 1349's review F7), is refused as it loads, asking for it compiled again, its step
    or its sampled step: its fingerprint matches a run of the program under no options whatever it compiled
    under. The same step saved without them loads."""
    import dataclasses
    import pickle

    import jax
    import jax.numpy as jnp
    import pytest

    from dew.nn.kernels.options import program_digest
    from dew.training.ahead import AheadOfTime, AheadProgram, Executable

    lowered = jax.jit(lambda x: jnp.sin(x) * 2).lower(jnp.ones(4))
    options = {"xla_gpu_enable_triton_gemm": False}
    sampled = AheadProgram(program_digest(lowered, options), Executable(b"", ()))
    current = AheadOfTime("sm90", 1, {}, program_digest(lowered, options), Executable(b"", ()), None, 0, 0, 0,
                          0, 0, {}, plan={"autotuned": True}, sampled=sampled)

    def saved(step: AheadOfTime, name: str):
        path = tmp_path / f"{name}.dew-aot"
        path.write_bytes(pickle.dumps(step))
        return path

    assert AheadOfTime.load(saved(current, "current")) == current
    earlier = dataclasses.replace(current)
    object.__setattr__(earlier, "options", options)
    earlier_sampled = dataclasses.replace(sampled)
    object.__setattr__(earlier_sampled, "options", options)
    for step, name in ((earlier, "earlier"),
                       (dataclasses.replace(current, sampled=earlier_sampled), "sampled")):
        with pytest.raises(ValueError):
            AheadOfTime.load(saved(step, name))


def test_a_lowerings_program_is_rendered_once_for_its_key_and_every_fingerprint(monkeypatch):
    """A lowering's program is rendered once for as long as it lives: its process-local key
    (`program_key`) and its fingerprints a file of programs holds it by, under any options
    (`program_digest`), read the one rendering (row 13555: a step handed an executable compiled ahead
    rendered and hashed its whole program twice as it started)."""
    import jax
    import jax.numpy as jnp

    from dew.nn.kernels.options import program_digest, program_key

    rendered = []
    render = jax.stages.Lowered.as_text

    def counted(lowered, *args, **kwargs):
        rendered.append(lowered)
        return render(lowered, *args, **kwargs)

    monkeypatch.setattr(jax.stages.Lowered, "as_text", counted)
    lowered = jax.jit(lambda x: jnp.sin(x) * 2).lower(jnp.ones(4))
    program_key(lowered)
    plain, optioned = program_digest(lowered), program_digest(lowered, {"xla_gpu_enable_command_buffer": ""})
    assert plain != optioned and len(rendered) == 1
