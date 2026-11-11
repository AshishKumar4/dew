"""tools/compiled_kernels.py: the kernels a run compiled, read from XLA's dump."""

import json
import subprocess
import sys
from pathlib import Path

TOOL = str(Path(__file__).resolve().parents[1] / "tools" / "compiled_kernels.py")
GEMM = ('%d = f32[4,4] custom-call(%a, %b), custom_call_target="__cublas$gemm", '
        'backend_config={"operation_queue_id":"QUEUE","gemm_backend_config":{"dot_dimension_numbers":{"lhs":[1]},'
        '"alpha_real":ALPHA}}')
TRITON = ('%t = f32[4] custom-call(%a), custom_call_target="__gpu$xla.gpu.triton", '
          'backend_config={debug = false, grid_x = GRID : i32, ir = "IR", name = "attend", '
          'num_warps = 4 : i32}')


def listed(tmp_path: Path, name: str, modules: dict[str, str]) -> Path:
    dump = tmp_path / name
    dump.mkdir()
    for module, text in modules.items():
        (dump / f"{module}.after_optimizations.txt").write_text(text)
    out = tmp_path / f"{name}.json"
    subprocess.run([sys.executable, TOOL, "list", str(dump), str(out)], check=True, capture_output=True)
    return out


def missing(before: Path, after: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, TOOL, "missing", str(before), str(after)], capture_output=True,
                          text=True)


def test_a_run_that_keeps_every_kernel_holds_a_superset_and_any_config_changed_is_missing(tmp_path):
    """A later run that compiles every kernel an earlier one did, and more, misses none, whatever queue it
    schedules them on and whatever locations its Triton modules serialize; one whose gemm differs only
    past its first nested object (its alpha) or in its result's shape, or whose Triton kernel runs another
    grid, misses that kernel, each configuration read whole, and one compiling a kernel at one tile where
    the earlier compiled it at two misses one."""
    gemm = GEMM.replace("QUEUE", "0")
    first = gemm.replace("ALPHA", "1") + "\n" + TRITON.replace("IR", "a").replace("GRID", "8")
    before = listed(tmp_path, "before", {"a": first})
    kernels = json.loads(before.read_text())
    assert {(kernel["target"], kernel["name"]) for kernel in kernels} == {("__cublas$gemm", ""),
                                                                         ("__gpu$xla.gpu.triton", "attend")}
    # A kernel compiled at a second tile of its grid counts twice: losing that tile loses one.
    twice = listed(tmp_path, "twice", {"a": first, "c": TRITON.replace("IR", "c").replace("GRID", "8")})
    assert missing(twice, before).returncode == 1
    # Another run of the same program schedules its calls elsewhere and serializes its Triton module's
    # source locations otherwise: the same kernels.
    kept = listed(tmp_path, "kept", {"a": first.replace('ir = "a"', 'ir = "b"'),
                                     "b": GEMM.replace("ALPHA", "3").replace("QUEUE", "7")})
    assert missing(before, kept).returncode == 0
    triton = TRITON.replace("IR", "a")
    for name, text, lost in (("alpha", gemm.replace("ALPHA", "2") + "\n" + triton.replace("GRID", "8"),
                              "__cublas$gemm"),
                             ("grid", gemm.replace("ALPHA", "1") + "\n" + triton.replace("GRID", "16"),
                              "__gpu$xla.gpu.triton"),
                             ("shape", gemm.replace("ALPHA", "1").replace("f32[4,4]", "f32[8,4]") + "\n"
                              + triton.replace("GRID", "8"), "__cublas$gemm")):
        found = missing(before, listed(tmp_path, name, {"a": text}))
        assert found.returncode == 1 and found.stdout.startswith(lost), found.stdout
