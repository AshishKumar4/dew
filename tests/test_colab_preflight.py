"""The Colab preflight: shell facts, Python symbol chains, arch-aware sources and the receipt gate."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.colab import colab_batch as batch, preflight


def write(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def repository(root: Path, files: dict[str, str]) -> str:
    write(root, files)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    identity = ["-c", "user.name=Test", "-c", "user.email=test@example.invalid"]
    subprocess.run(["git", "-C", str(root), *identity, "commit", "-qm", "tree"], check=True)
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def test_installs_need_exact_pins_and_redirections_are_not_packages():
    found = preflight.shell_facts(
        "stage torch-install /usr/bin/python3 -m pip install --no-deps fla-core==0.5.2 einops > log 2>&1\n"
        "uv pip install -q flash_qla[cp]==0.1.3 -c constraints.txt\n",
        {},
    )
    assert found["pins"] == {"fla-core": "0.5.2", "flash-qla": "0.1.3"}
    assert found["failures"] == ["pip installs 'einops' without an exact version"]


def test_a_script_is_read_with_the_variables_its_command_sets():
    script = (
        "set -u\nout=/content/out\n"
        'git checkout -q "$C56_REF"\n'
        'stage gdn env PYTHONPATH=src timeout 2400 python /content/c56/rules.py --out "$out/x"\n'
        'stage profile bash -c "$C56_PROFILE"\n'
    )
    command = 'C56_REF=abc C56_PROFILE="python -m dew.train --steps 2" bash /content/c56/run.sh'
    found = preflight.shell(command, {"run.sh": script, "rules.py": ""})
    assert found["checkouts"] == ["abc"]
    assert found["runs"] == ["/content/c56/run.sh", "/content/c56/rules.py"]
    assert found["modules"] == ["dew.train"]
    unset = preflight.shell("bash /content/c56/run.sh", {"run.sh": script})
    assert unset["checkouts"] == ["$C56_REF"]
    assert "runs $C56_PROFILE, which the command does not set" in unset["failures"]


def test_chains_follow_aliases_scopes_and_module_assignments():
    source = (
        "import torch\n"
        "import jax.numpy as jnp\n"
        "def jax_rules():\n"
        "    from dew.nn.kernels import fla_gdn\n"
        "    return fla_gdn.chunk_gated_delta_rule, jnp.zeros\n"
        "def shadowed(torch):\n"
        "    return torch.not_a_module_attribute\n"
        "def shim():\n"
        "    import flash_qla.ops.gated_delta_rule.chunk as chunk\n"
        "    chunk.aggregate_card_state = None\n"
    )
    chains, bound = preflight.uses(source, "rules.py")
    found = {(chain["module"], tuple(chain["attrs"])) for chain in chains}
    assert ("dew.nn.kernels", ("fla_gdn", "chunk_gated_delta_rule")) in found
    assert ("jax.numpy", ("zeros",)) in found
    assert not any("not_a_module_attribute" in attrs for _, attrs in found)
    assert bound == {("flash_qla.ops.gated_delta_rule.chunk", "aggregate_card_state")}


# FlashQLA 0.1.3's shape: the chunk package binds aggregate_card_state on sm90 only, and a function
# in a submodule imports it on every call.
PACKAGE = {
    "fq/__init__.py": "from .chunk import run\n",
    "fq/arch.py": "def get_target_compute_version():\n    raise RuntimeError('no GPU')\n",
    "fq/chunk/__init__.py": (
        "from fq import arch\n"
        "if arch.get_target_compute_version() == '9.0':\n"
        "    from .hopper import kernel, aggregate\n"
        "elif arch.get_target_compute_version() in ['12.0', '12.1']:\n"
        "    from .sm120 import kernel\n"
        "from .cp import prepare\n"
        "def run():\n"
        "    return prepare()\n"
    ),
    "fq/chunk/cp.py": (
        "try:\n    from fq.optional import extra\nexcept ImportError:\n    extra = None\n"
        "def prepare():\n    from fq.chunk import kernel, aggregate\n    return kernel, aggregate\n"
    ),
    "fq/chunk/hopper.py": "kernel = aggregate = None\n",
    "fq/chunk/sm120.py": "kernel = None\n",
}


@pytest.mark.parametrize(("arch", "failing"), [("12.0", True), ("9.0", False)])
def test_sources_follow_the_target_gpu_into_function_imports(tmp_path, arch, failing):
    write(tmp_path, PACKAGE)
    sources = preflight.Sources([tmp_path], arch, {"fq"})
    failures = sources.closure({"fq"}, set())
    assert bool(failures) == failing
    if failing:
        assert failures == ["fq.chunk.cp:6: from fq.chunk import aggregate: unbound on sm12.0"]
    assert sources.check("fq", ["run"], set()) is None
    assert sources.check("fq.chunk", ["kernel"], set()) is None
    assert (sources.check("fq.chunk", ["aggregate"], set()) is not None) == failing
    assert sources.closure({"fq"}, {("fq.chunk", "aggregate")}) == []


def test_runtime_resolution_names_missing_modules_attributes_and_import_errors(tmp_path):
    write(tmp_path, {"good.py": "x = 1\n", "needs_gpu.py": "raise RuntimeError('no CUDA device')\n"})
    chains = [
        {"module": "good", "attrs": ["x"]},
        {"module": "good", "attrs": ["y"]},
        {"module": "good", "attrs": ["z"]},
        {"module": "absent_module", "attrs": []},
        {"module": "needs_gpu", "attrs": []},
    ]
    found = preflight.resolve(sys.executable, chains, {("good", "z")}, cwd=tmp_path)
    assert [kind for kind, _ in found] == ["ok", "attr", "ok", "absent", "import"]
    assert found[1][1] == "good has no y"


def test_a_ref_without_the_called_module_fails_the_preflight(tmp_path, monkeypatch):
    tree = tmp_path / "dew"
    commit = repository(tree, {"src/dew/__init__.py": "", "src/dew/kernels.py": "rule = 1\n"})
    monkeypatch.setattr(preflight, "REPO", tree)
    script = tmp_path / "rules.py"
    script.write_text("def jax_rules():\n    from dew.kernels import rule, fla_gdn\n")
    args = argparse.Namespace(
        ref=commit, hardware="gpu:G4", command="python /content/c56/rules.py", inputs=[str(script)], pin=[]
    )
    args.origin = "origin"
    args.python, args.dew_python = preflight.PYTHON, sys.executable
    receipt = preflight.preflight(args, tmp_path / "work")
    assert receipt["commit"] == commit and not receipt["ok"]
    assert receipt["failures"] == ["rules.py:2: dew.kernels.fla_gdn: dew.kernels has no fla_gdn"]
    assert receipt["inputs"] == {"rules.py": hashlib.sha256(script.read_bytes()).hexdigest()}


def receipt_for(tmp_path, head, **changes):
    script = tmp_path / "rules.py"
    script.write_text("print(1)\n")
    receipt = {
        "ok": True,
        "commit": head,
        "hardware": "gpu:T4",
        "command": "python /content/c56/rules.py",
        "inputs": {"rules.py": hashlib.sha256(script.read_bytes()).hexdigest()},
        "failures": [],
    }
    receipt.update(changes)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    return path, script


def request(receipt: Path, script: Path) -> dict:
    fields = {
        "lane": "test",
        "command": "python /content/c56/rules.py",
        "outputs": ["out/"],
        "ref": "HEAD",
        "inputs": f"{script} to /content/c56/",
        "gpu_reason": "test",
        "expected_cu": "0.1",
        "decision": "test",
        "memory_estimate": "1GiB",
        "preflight": str(receipt),
    }
    return {"text": "- lane: test job\n", "fields": fields}


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({}, None),
        ({"ok": False, "failures": ["dew.kernels has no fla_gdn"]}, "it failed: dew.kernels has no fla_gdn"),
        ({"commit": "0" * 40}, "it checked 0000"),
        ({"hardware": "gpu:G4"}, "it checked gpu:G4, not gpu:T4"),
        ({"command": "python other.py"}, "a different command"),
        ({"inputs": {}}, "it did not check /content/c56/rules.py"),
    ],
)
def test_plan_requires_a_passing_receipt_of_this_exact_request(tmp_path, changes, reason):
    commit = repository(tmp_path / "repo", {"README": "x\n"})
    path, script = receipt_for(tmp_path, commit, **changes)
    entry = request(path, script)
    if reason is None:
        plan = batch.plan_jobs([entry], "gpu", "gpu:T4", commit, tmp_path / "repo")
        assert plan["jobs"][0]["preflight"] == hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        with pytest.raises(ValueError, match=reason):
            batch.plan_jobs([entry], "gpu", "gpu:T4", commit, tmp_path / "repo")


def test_selection_refuses_a_request_without_a_preflight(tmp_path, monkeypatch):
    queue = tmp_path / "queue.md"
    queue.write_text(
        "## Pending\n- lane: test job\n  device: gpu\n  ref: HEAD\n  command: run\n  outputs: out/\n"
        "  gpu_reason: test\n  expected_cu: 0.1\n  decision: test\n  memory_estimate: 1GiB\n\n## Done\n"
    )
    monkeypatch.setattr(batch, "QUEUE", queue)
    with pytest.raises(ValueError, match="GPU request lacks preflight"):
        batch.select("gpu", "test", "gpu:T4", [])
