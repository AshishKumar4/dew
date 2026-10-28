"""Row 2078: whether the attention programs lower and compile on the mma.sync bundle (sm80) once qualified,
and at what registers (a probe, never lands)."""
import os, sys
os.environ["MOSAIC_GPU_DUMP_CONSTRAINT_SYSTEM"] = "1"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ.setdefault("LAUNCHES", "forward,training,backward,fused")
os.environ.setdefault("CONFIGS", "64x64c1")
from dew.nn.kernels.templates import attention_mosaic_program, mosaic_sm80
attention_mosaic_program.QUALIFIED = attention_mosaic_program.QUALIFIED | {mosaic_sm80.Mma.name}
"""Off-GPU ptxas registers, spills and register need (kernel_compile.register_need) of Mosaic attention launches (a probe, never lands).
Run from a worktree root on the VM: python ptxas_probe.py LAUNCHES CONFIGS, LAUNCHES a comma list of
forward,training,backward,accurate,fused with an 'h' suffix for hooked (e.g. training,trainingh), CONFIGS a comma
list of BQxBKcC (e.g. 64x64c1,64x128c2)."""
import json
import sys

sys.path.insert(0, "tests")
import fla_gdn_support as support  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental.mosaic.gpu import layout_inference as li
from jax.experimental.mosaic.gpu import utils as mgutils
_dump = mgutils.dump_to_file_or_stdout
def brief_dump(content, *args, **kwargs):
    print("DUMP", args, "LINES", len(str(content).splitlines()), flush=True)
mgutils.dump_to_file_or_stdout = brief_dump
_reduce_constraint = li.cs.reduce_constraint
_seen_bad = set()
def reduce_debug(constraint, assignments):
    result = _reduce_constraint(constraint, assignments)
    if isinstance(result, li.cs.Equals):
        a,b = result.lhs,result.rhs
        if isinstance(b, li.cs.Variable): a,b = b,a
        if isinstance(a, li.cs.Variable) and isinstance(b, li.cs.Constant):
            if a in assignments and assignments[a] != b:
                print("CONFLICT ASSIGNMENT", constraint, result, assignments[a], flush=True)
    if isinstance(result, li.cs.Unsatisfiable) or result.holds() is False:
        key = str(result)
        if key not in _seen_bad and len(_seen_bad) < 15:
            _seen_bad.add(key)
            print("BAD CONSTRAINT", constraint, "\nREDUCED", result, flush=True)
    return result
li.cs.reduce_constraint = reduce_debug
_isvalid = li.cs.is_valid_assignment
def valid_debug(var, value):
    result = _isvalid(var, value)
    if not result and len(_seen_bad) < 25:
        key = "invalid" + str(var) + str(value)
        if key not in _seen_bad:
            _seen_bad.add(key)
            print("INVALID ASSIGNMENT", var, value, flush=True)
    return result
li.cs.is_valid_assignment = valid_debug
_shortcut = li.cs._has_relayout_of_non_splat_to_splat
def shortcut_debug(constraints):
    result = _shortcut(constraints)
    if result:
        print("SHORTCUT", " | ".join(map(str, constraints)), flush=True)
    return result
li.cs._has_relayout_of_non_splat_to_splat = shortcut_debug
_conjure = li.conjure_assignment
def conjure_debug(unknowns, system, arch):
    count = 0
    for assignment in _conjure(unknowns, system, arch):
        count += 1
        yield assignment
    if count == 0:
        print("NO CANDIDATES", " | ".join(map(str, unknowns)),
              "CONSTRAINTS", " | ".join(map(str, system.constraints)), flush=True)
li.conjure_assignment = conjure_debug
_and = li.cs.ConstraintSystem.__and__
def and_debug(self, other):
    result = _and(self, other)
    if isinstance(result, li.cs.Unsatisfiable):
        print("UNSAT CONJUNCTION conflicts", [(str(k), str(v), str(other.assignments[k]))
              for k,v in self.assignments.items()
              if k in other.assignments and other.assignments[k] != v] if hasattr(other, "assignments") else [],
              flush=True)
    return result
li.cs.ConstraintSystem.__and__ = and_debug
for opname, rule in tuple(li._constraint_system_derivation_rules.items()):
    def wrapped(*args, _rule=rule, _name=opname, **kwargs):
        result = _rule(*args, **kwargs)
        if isinstance(result[0], li.cs.Unsatisfiable):
            print("UNSAT OP", _name, args[1], "LOCATION", args[1].location, flush=True)
        return result
    li._constraint_system_derivation_rules[opname] = wrapped
from kernel_compile import compile_mosaic_kernels  # noqa: E402

from dew.nn.kernels import tuning  # noqa: E402
from dew.nn.kernels.templates import attention as template, attention_mosaic  # noqa: E402
_stored = mosaic_sm80.Mma.stored
def stored_vector(self, parts, views, *, smem, wait=True):
    if smem is not None and len(smem.shape) == 1:
        for part, view in zip(parts, views, strict=True):
            mosaic_sm80.plgpu.store(view, part.astype(view.dtype), optimized=False)
    else:
        return _stored(self, parts, views, smem=smem, wait=wait)
if os.environ.get("FIX_VECTOR", "0") == "1":
    mosaic_sm80.Mma.stored = stored_vector
from dew.nn.kernels.templates import attention_mosaic_forward as af, attention_mosaic_backward as ab
_kernel = mosaic_sm80.Mma.kernel
def kernel_packed(self, entry, **kwargs):
    raw = _kernel(self, entry, **kwargs)
    name = kwargs.get("name", "")
    skipped = (CURRENT == "training" and name.startswith("attention_peaks")
               or CURRENT == "dq" and name.startswith("attention_dkv")
               or CURRENT == "dkv" and name.startswith("attention_dq"))
    if skipped:
        return lambda *args: tuple(jnp.zeros(out.shape, out.dtype) for out in kwargs["out_type"])
    def call(*args):
        args = list(args)
        if os.environ.get("PACKED", "0") == "1":
            for index in ((0, 3) if name.startswith("attention_dkv") else (1, 2)):
                args[index] = jnp.swapaxes(args[index], 1, 2)
        return raw(*args)
    return call
mosaic_sm80.Mma.kernel = kernel_packed
def views_packed(refs, row, head):
    return {name: (ref.at[row, head], lambda j: (j, 0)) for name,ref in refs.items()}
if os.environ.get("PACKED", "0") == "1":
    af._views_of = ab._views_of = views_packed
af.guarded_columns = lambda family, width, dtype: min(width, 32)
af.algebra.walked_launches = lambda spec, launch, kept: launch(spec, None)[0]

import os
ROWS, TOKENS, HEADS, KV = 1, 1024, 8, 2
WIDTH = 128
SPANS = template.spans(TOKENS, TOKENS, ROWS, causal=True)
QUERY = jax.ShapeDtypeStruct((ROWS, TOKENS, HEADS, WIDTH), jnp.bfloat16)
KEY = jax.ShapeDtypeStruct((ROWS, TOKENS, KV, WIDTH), jnp.bfloat16)
STATS = jax.ShapeDtypeStruct((ROWS, HEADS, TOKENS), jnp.float32)
RESIDUALS = (STATS, STATS, jax.ShapeDtypeStruct((ROWS, HEADS, TOKENS), jnp.int32),
             template.Split(jax.ShapeDtypeStruct((ROWS, HEADS, TOKENS, WIDTH), jnp.float32), STATS, STATS, STATS))
SLOPES = jax.ShapeDtypeStruct((HEADS,), jnp.float32)
for op in (attention_mosaic.MosaicForward, attention_mosaic.MosaicBackward):
    op.register_budget = lambda self, config: 10 ** 6
    op.legal = lambda *args: True


def hooked(slope):
    def score_mod(s, q_pos, k_pos, alibi):
        return 30.0 * jnp.tanh(s / 30.0) + alibi * k_pos.astype(s.dtype)
    return template.Hooks(score_mod, (template.Param(slope, template.Kind.HEAD),),
                          template.sinks(jnp.zeros((HEADS,), jnp.float32), ROWS, TOKENS, WIDTH))


def case(launch, config):
    hooks = launch.endswith("h")
    kind = launch.rstrip("h")
    bq, rest = config.split("x")
    bk, consumers = rest.split("c")
    bq, bk, consumers = int(bq), int(bk), int(consumers)

    def call():
        if kind in ("forward", "training"):
            tiles = attention_mosaic.Tiles(bq, bk, 2, consumers, family="mma.sync")

            def run(q, k, v, slope):
                with tuning.pinned({attention_mosaic.MosaicForward: tiles}):
                    return attention_mosaic.forward(q, k, v, SPANS, hooked(slope) if hooks else None,
                                                    residuals=kind == "training")
            return run, (QUERY, KEY, KEY, SLOPES)
        tiles = attention_mosaic.BackwardTiles(bq, bk, 2, consumers, fused=kind == "fused", family="mma.sync")

        def run(q, k, v, dout, slope, residuals):
            with tuning.pinned({attention_mosaic.MosaicBackward: tiles}):
                grads = attention_mosaic.backward(q, k, v, SPANS, hooked(slope) if hooks else None, dout,
                                                  residuals=residuals, accurate=kind == "accurate")
            return grads.query, grads.key, grads.value
        return run, (QUERY, KEY, KEY, QUERY, SLOPES, RESIDUALS)
    return support.KernelCase(f"{launch}|{config}", call)


LAUNCHES = os.environ.get("LAUNCHES", "forward,training,backward,fused,accurate")
CONFIGS = os.environ.get("CONFIGS", "64x64c1,64x128c1,64x64c2")
import traceback
from dew.nn.kernels import generation
from jax.sharding import Mesh
import numpy as np
results = []
for launch in ("plain", "peaks", "training", "dq", "dkv", "fused"):
    CURRENT = launch
    alias = {"plain": "forward", "peaks": "training", "dq": "backward", "dkv": "backward"}.get(launch, launch)
    fn, args = case(alias, "64x64c1").call()
    try:
        mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm80")
        with generation.assumed("sm80"), jax.sharding.use_abstract_mesh(mesh):
            print("ENV", jax.__version__, generation.gpu_target(), generation.family("sm80"),
                  attention_mosaic_program.QUALIFIED,
                  attention_mosaic_program.bundle("sm80", jnp.bfloat16), flush=True)
            jax.jit(fn).trace(*args).lower(lowering_platforms=("cuda",))
        print("LOWERED", launch)
        results.append((launch, "LOWERED"))
    except Exception:
        print("FAILED", launch)
        print("ERROR", type(sys.exception()).__name__, str(sys.exception()), flush=True)
        results.append((launch, type(sys.exception()).__name__, str(sys.exception())))
for result in results:
    print("FINAL_RESULT", result, flush=True)
