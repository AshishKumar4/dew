"""Scratch plugin (`-p tests.hlo_diff --hlo-show=step|lambda`): the run's optimized HLO dumps
(/tmp/dew-hlo), the fp32 objective-gradient module (`jit__lambda` with no f64) against the `jit_step`
module whose fusions overlap it most, both normalized (names, layouts, metadata dropped). Prints the
fused computations with a reduce, dot, divide, log or exp that only the shown side has, and the
unfused dots and custom calls only it has."""
import collections
import glob
import re
from pathlib import Path


def pytest_addoption(parser):
    parser.addoption("--hlo-show", default="step")


def _normal(line: str) -> str:
    line = re.sub(r", (metadata|backend_config|frontend_attributes|sharding|statistics)=.*$", "", line)
    line = re.sub(r"\]\{[^}]*\}", "]", line)
    line = re.sub(r"%[\w.\-]+", "%", line)
    return line.strip().removeprefix("ROOT ")


def _computations(text: str) -> dict[str, list[str]]:
    held, name, body = {}, None, []
    for line in text.splitlines():
        if name is None:
            match = re.match(r"^(ENTRY )?%([\w.\-]+) ", line)
            if match and line.rstrip().endswith("{"):
                name, body = ("ENTRY" if match.group(1) else match.group(2)), []
        elif line.strip() == "}":
            held[name] = body
            name = None
        else:
            body.append(_normal(line))
    return held


def _fusions(computations):
    return collections.Counter(tuple(sorted(body)) for name, body in computations.items()
                               if name.startswith("fused") and any(
                                   f" {op}(" in line for line in body
                                   for op in ("reduce", "dot", "divide", "log", "exponential")))


def _loose(computations):
    return collections.Counter(line for name, body in computations.items() if not name.startswith("fused")
                               for line in body if " dot(" in line or " custom-call(" in line)


def _summary(signature) -> str:
    ops = [line for line in signature if any(f" {op}(" in line for op in
                                             ("reduce", "dot", "divide", "log", "exponential", "convert"))]
    return " | ".join(re.sub(r"\s+", " ", line)[:110] for line in ops)[:600]


def pytest_terminal_summary(terminalreporter, config):
    paths = glob.glob("/tmp/dew-hlo/*cpu_after_optimizations.txt") or glob.glob(
        "/tmp/dew-hlo/*after_optimizations.txt")
    modules = {path: Path(path).read_text() for path in paths}

    def entry(text):
        return next((line for line in text.splitlines() if line.startswith("ENTRY")), "")

    # The fp32 arm's parameters are fp32; the float64 truth's are f64 (a CPU's fp32 program holds f64 sums).
    lambdas = [path for path, text in modules.items() if "jit__lambda" in path and "f64[" not in entry(text)]
    steps = [path for path in modules if ".jit_step." in path]
    if not lambdas or not steps:
        terminalreporter.write_line(f"HLODIFF none: {len(lambdas)} lambdas {len(steps)} steps")
        return
    lam = max(lambdas, key=lambda path: len(modules[path]))
    lam_c = _computations(modules[lam])
    lam_f = _fusions(lam_c)

    def overlap(path):
        return sum((_fusions(_computations(modules[path])) & lam_f).values())

    step = max(steps, key=overlap)
    step_c = _computations(modules[step])
    step_f = _fusions(step_c)
    terminalreporter.write_line(f"HLODIFF lambdas {len(lambdas)} steps {len(steps)}")
    terminalreporter.write_line(f"HLODIFF lambda {lam.split('/')[-1]} step {step.split('/')[-1]} shared "
                                f"{sum((step_f & lam_f).values())} step-only {sum((step_f - lam_f).values())} "
                                f"lambda-only {sum((lam_f - step_f).values())}")
    shown = step_f - lam_f if config.getoption("--hlo-show") == "step" else lam_f - step_f
    for signature, count in sorted(shown.items(), key=lambda item: -len(item[0]))[:14]:
        terminalreporter.write_line(f"HLODIFF {count}x {_summary(signature)}")
    loose = (_loose(step_c) - _loose(lam_c)) if config.getoption("--hlo-show") == "step" else (
        _loose(lam_c) - _loose(step_c))
    for line, count in list(loose.items())[:8]:
        terminalreporter.write_line(f"HLODIFF loose {count}x {re.sub(chr(32) + '+', ' ', line)[:200]}")
