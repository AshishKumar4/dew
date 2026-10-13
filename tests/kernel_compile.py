"""Compile Pallas kernels for a GPU on this CPU with Triton's own compiler: the CPU suite's first filter
for declared kernel configs (tests/test_kernel_configs.py). A GPU's tests are the verdict."""

import concurrent.futures
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import jax
import numpy as np
from jax.sharding import Mesh

from dew.nn.kernels.generation import assumed, gpu_mesh, gpu_target

HUNG = 600.0
"""Wall seconds after which a compile's child session is stopped as hung: the process boundary's
guard, not a bound on the kernel. How long a compile takes on a shared CPU says nothing about it."""

_TRITON_CALL = "__gpu$xla.gpu.triton"
_COMPILE = """import json, sys
import triton
from triton.backends.compiler import GPUTarget

entry = json.loads(sys.argv[1])
kernel = triton.compile(entry["path"], target=GPUTarget("cuda", entry["capability"], 32),
                        options={"num_warps": entry["warps"], "num_stages": entry["stages"]})
print(json.dumps({"shared": kernel.metadata.shared}))
"""


def _triton_calls(lowered: jax.stages.Lowered) -> list[dict]:
    """Each Pallas-Triton call of a lowered program: its Triton IR as text, its warps and stages."""
    # The custom call's Triton IR is reachable only through JAX's MLIR bindings, jaxlib's and its own.
    from jax._src.interpreters import mlir
    from jax._src.lib.triton import dialect
    from jaxlib.mlir import ir

    found = []

    def visit(operation) -> None:
        if (operation.name == "stablehlo.custom_call"
                and ir.StringAttr(operation.attributes["call_target_name"]).value == _TRITON_CALL):
            config = ir.DictAttr(operation.attributes["mhlo.backend_config"])
            found.append({"ir": ir.StringAttr(config["ir"]).value_bytes,
                          "warps": ir.IntegerAttr(config["num_warps"]).value,
                          "stages": ir.IntegerAttr(config["num_stages"]).value})
        for region in operation.regions:
            for block in region.blocks:
                for inner in block.operations:
                    visit(inner.operation)

    module = lowered.compiler_ir("stablehlo")
    if module is None:
        raise ValueError("the lowered program has no StableHLO module")
    visit(module.operation)
    context = mlir.make_ir_context()
    dialect.register_dialect(context)
    with context:
        for call in found:
            call["ir"] = ir.Module.parse(call["ir"]).operation.get_asm(enable_debug_info=False)
    return found


def _compiled(entry: dict) -> dict:
    """One Triton call compiled by pip's Triton in a child of its own session: its shared bytes, or why
    it has none. A session still running at HUNG seconds is stopped."""
    child = subprocess.Popen([sys.executable, "-c", _COMPILE, json.dumps(entry)], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        output, _ = child.communicate(timeout=HUNG)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.communicate()
        return {"error": f"its compile hung past {HUNG:.0f} s"}
    if child.returncode:
        return {"error": f"Triton's compile failed ({child.returncode}): {output.strip()[-800:]}"}
    return json.loads(output.strip().splitlines()[-1])


def failures(found: list[dict]) -> list[str]:
    """Each compiled record's failure, if any: its error, or its shared bytes past its block's limit."""
    return [f"{record['name']}: {record['error']}" if "error" in record else
            f"{record['name']}: {record['shared']} shared bytes, limit {record['limit']}"
            for record in found if "error" in record or not record["shared"] <= record["limit"]]


def compile_kernels(cases, generation: str) -> list[dict]:
    """Each case (`name`, `call()` giving the kernel and abstract operands) on the CUDA `generation`,
    lowered as on that GPU on this host (`assumed`, `gpu_mesh`) and compiled by pip's Triton for its
    capability, never run: its name, the generation's block limit, and the shared bytes Triton allocates (the
    most of any call it makes), or the error.

    A cheap first filter, not a GPU's verdict: a kernel that does not compile or takes more shared
    memory than its GPU's block holds fails here, before a GPU is paid for. Identical IR compiles once;
    compiles run one to each CPU this process may use."""
    found: list[dict] = []
    calls: dict[str, tuple[dict, list[dict]]] = {}
    with tempfile.TemporaryDirectory(prefix="dew-compile-") as scratch:
        mesh = gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), generation)
        for case in cases:
            record: dict = {"name": case.name, "generation": generation,
                            "limit": gpu_target(generation).shared_limit, "shared": 0}
            found.append(record)
            fn, args = case.call()
            try:
                with assumed(generation), jax.sharding.use_abstract_mesh(mesh):
                    lowered = jax.jit(fn).trace(*args).lower(lowering_platforms=("cuda",))
            except Exception as error:  # A case that does not lower is its own failure.
                record["error"] = f"it does not lower: {str(error)[-800:]}"
                continue
            for call in _triton_calls(lowered):
                named = f"{generation}|{call['warps']}|{call['stages']}|{call['ir']}"
                key = hashlib.sha256(named.encode()).hexdigest()
                if key not in calls:
                    path = Path(scratch) / f"{len(calls)}.ttir"
                    path.write_text(call["ir"])
                    calls[key] = ({"path": str(path), "capability": int(generation.removeprefix("sm")),
                                   "warps": call["warps"], "stages": call["stages"]}, [])
                calls[key][1].append(record)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
            entries = [entry for entry, _ in calls.values()]
            outcomes = list(pool.map(_compiled, entries))
    for (_, records), outcome in zip(calls.values(), outcomes, strict=True):
        for record in records:
            if "error" in outcome:
                record.setdefault("error", outcome["error"])
            else:
                record["shared"] = max(record["shared"], outcome["shared"])
    return found
