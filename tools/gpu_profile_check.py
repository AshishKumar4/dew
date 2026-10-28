"""Whether profiling a GPU still works: run first in every GPU session, and on every JAX bump.

    python tools/gpu_profile_check.py --out DIR

About a minute. A small step (a scan of dense layers and its gradient) is
warmed up, then three of its runs are captured under `dew.Profiler`, as a
Trainer's profile window captures a warmed-up step. That happens twice, each
in its own process, since XLA reads its flags once per process:

- `default`: the session's own XLA_FLAGS;
- `attributed`: with `--xla_gpu_enable_command_buffer=` added, the flag
  `dew.telemetry.profile.COMMAND_BUFFERS` names.

Each records how many HLO ops XProf's hlo_stats attributes device time to,
the manifest's `op_attribution`, whether XLA logged that it ran a command
buffer op by op for the session (TF_CPP_VMODULE=command_buffer_thunk=1),
the device busy time per step from the trace against the step's wall time,
and the versions. DIR/check.json holds both. The check fails (exit 1) unless
the `attributed` capture names ops, and each capture's per-step busy time is
positive and at most its wall time. `default` is recorded, not judged: it
attributes no op on JAX 0.11.2, so a release that changes that shows here.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

COMMAND_BUFFERS_OFF = "--xla_gpu_enable_command_buffer="
FALLBACK = "as a regular thunk sequence because we detected active profiling session"
STEPS = 3


def capture(out: Path) -> dict:
    """One variant, in this process: warm the step, time it, capture it."""
    import jax
    import jax.numpy as jnp
    from dew.telemetry.trace import busy_time, device_events

    import dew
    from dew.telemetry.benchmark import samples
    from dew.telemetry.profile import attributed_ops

    def layers(x, weights):
        return jax.lax.scan(lambda h, w: (jnp.tanh(h @ w), None), x, weights)[0]

    def loss(weights, x):
        return jnp.sum(layers(x, weights) ** 2)

    step = jax.jit(jax.grad(loss))
    weights = jax.random.normal(jax.random.key(0), (4, 256, 256), jnp.float32) * 0.05
    x = jax.random.normal(jax.random.key(1), (64, 256), jnp.float32)
    walls = samples(step, weights, x, warmup=5, repeats=10)
    with dew.Profiler(out) as profiler:
        for _ in range(STEPS):
            jax.block_until_ready(step(weights, x))
    # The capture just made: its reports sit beside its manifest.
    (directory,) = profiler.captures
    manifest = json.loads((directory / "manifest.json").read_text())
    planes, _ = device_events(directory)
    events = [event for name, found in planes.items() if name.startswith("/device:GPU") for event in found]
    busy = busy_time([(event.start_ns, event.start_ns + event.duration_ns) for event in events])
    return {"device": jax.devices()[0].device_kind, "platform": jax.default_backend(),
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "attributed_ops": attributed_ops(directory / "reports"),
            "op_attribution": manifest.get("op_attribution"),
            "kernels_per_step": len(events) / STEPS,
            "busy_ms_per_step": busy / 1e6 / STEPS,
            "wall_ms_per_step": statistics.median(walls) * 1e3}


def variant(name: str, out: Path) -> dict:
    """`capture` in a child process with its variant's flags, and what XLA logged."""
    flags = os.environ.get("XLA_FLAGS", "")
    env = {**os.environ, "TF_CPP_VMODULE": "command_buffer_thunk=1", "TF_CPP_MIN_LOG_LEVEL": "0",
           "XLA_FLAGS": f"{flags} {COMMAND_BUFFERS_OFF}".strip() if name == "attributed" else flags,
           "PYTHONPATH": os.pathsep.join(filter(None, (str(Path(__file__).parent),
                                                       os.environ.get("PYTHONPATH"))))}
    done = subprocess.run([sys.executable, __file__, "--capture", str(out / name)], env=env,
                          capture_output=True, text=True, timeout=600)
    (out / f"{name}.log").write_text(done.stdout + done.stderr)
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    if done.returncode != 0 or not lines:
        return {"error": f"exit {done.returncode}: {(done.stderr or done.stdout)[-1500:]}"}
    return {**json.loads(lines[-1]), "fallback_logged": FALLBACK in done.stderr}


def judged(result: dict) -> list[str]:
    """What fails the check: an attributed capture that names no op, or a
    capture whose per-step device time is not in (0, wall time]."""
    failures = []
    for name, found in result["variants"].items():
        if "error" in found:
            failures.append(f"{name}: {found['error'][-300:]}")
            continue
        if not 0 < found["busy_ms_per_step"] <= found["wall_ms_per_step"]:
            failures.append(f"{name}: {found['busy_ms_per_step']:.3f} ms of device time a step "
                            f"against {found['wall_ms_per_step']:.3f} ms of wall time")
    attributed = result["variants"].get("attributed", {})
    if "error" not in attributed and not attributed.get("attributed_ops"):
        failures.append("attributed: hlo_stats names no op with command buffers off")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--capture", type=Path, help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.capture is not None:
        print(json.dumps(capture(options.capture)))
        return 0
    out = options.out
    out.mkdir(parents=True, exist_ok=True)
    installed = {distribution.metadata["Name"] for distribution in importlib.metadata.distributions()}
    packages = ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda13-plugin")
    versions = {package: importlib.metadata.version(package) for package in packages if package in installed}
    result = {"versions": versions,
              "variants": {name: variant(name, out) for name in ("default", "attributed")}}
    result["failures"] = judged(result)
    (out / "check.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
