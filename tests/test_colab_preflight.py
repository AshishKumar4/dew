"""The Colab preflight: shell facts, Python symbol chains, arch-aware sources and the receipt gate."""

import argparse
import ast
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
    assert found["interpreters"] == ["system", "venv"]


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


def test_chains_follow_aliases_scopes_module_assignments_and_path_additions():
    source = (
        "import sys\n"
        "from pathlib import Path\n"
        "RECIPES = Path('recipes/decision')\n"
        "sys.path.insert(0, str(RECIPES))\n"
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
        "def trace(name):\n"
        "    sys.path.insert(0, str(Path.cwd() / 'tools'))\n"
        "    sys.path.append(__file__)\n"
        "    import importlib\n"
        "    return importlib.import_module(name)\n"
    )
    chains, bound, paths, unchecked = preflight.uses(source, "rules.py")
    found = {(chain["module"], tuple(chain["attrs"])) for chain in chains}
    assert ("dew.nn.kernels", ("fla_gdn", "chunk_gated_delta_rule")) in found
    assert ("jax.numpy", ("zeros",)) in found
    assert not any("not_a_module_attribute" in attrs for _, attrs in found)
    assert bound == {("flash_qla.ops.gated_delta_rule.chunk", "aggregate_card_state")}
    assert paths == {"recipes/decision", "tools"}
    assert unchecked == ["rules.py:19: importlib.import_module of a computed name"]


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
    assert set(sources.guarded) == {"fq.chunk.cp:2 imports fq.optional, which is not installed"}


@pytest.mark.parametrize(
    ("test", "value"),
    [
        ("torch.cuda.get_device_capability()[0] >= 9", True),
        ("get_target_compute_version() in ('10.0', '10.3')", False),
        ("sys.version_info[:2] >= (3, 13) and sys.platform == 'linux'", True),
        ("torch.version.hip or IS_SM120", True),
        ("torch.version.hip and IS_SM120", preflight.UNKNOWN),
    ],
)
def test_conditions_are_known_only_from_the_vm(test, value):
    constants = {"IS_SM120": True}
    assert preflight.evaluate(ast.parse(test, mode="eval").body, constants, "12.0", (3, 13)) is value


def test_an_accepted_failure_needs_a_match_and_a_reason():
    failures = ["a:1: x unbound on sm12.0", "b:2: y unbound on sm12.0"]
    accepted, kept = preflight.accept(failures, ["a:1::never on CUDA", "zzz::stale", "b:2"])
    assert accepted == [{"failure": failures[0], "reason": "never on CUDA"}]
    assert kept == [
        failures[1],
        "--accept 'zzz::stale' matches no failure or gives no reason",
        "--accept 'b:2' matches no failure or gives no reason",
    ]


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
    monkeypatch.setattr(preflight, "VENV_PYTHON", "{}.{}".format(*sys.version_info[:2]))
    script = tmp_path / "rules.py"
    script.write_text("def jax_rules():\n    from dew.kernels import rule, fla_gdn\n")
    args = argparse.Namespace(
        ref=commit, hardware="gpu:G4", command="python /content/c56/rules.py", inputs=[str(script)], pin=[]
    )
    args.origin, args.accept, args.also, args.dew_python = "origin", [], [], sys.executable
    receipt = preflight.preflight(args, tmp_path / "work")
    assert receipt["commit"] == commit and not receipt["ok"]
    assert receipt["failures"] == ["rules.py:2: dew.kernels.fla_gdn: dew.kernels has no fla_gdn"]
    assert receipt["inputs"] == {"rules.py": hashlib.sha256(script.read_bytes()).hexdigest()}
    assert (receipt["interpreter"], receipt["tool"]) == ("system", preflight.TOOL)


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
        "tool": preflight.TOOL,
        "runtime": ["torch==2.11.0"],
        "interpreter": "system",
        "python": preflight.INTERPRETERS["system"],
        "dew_python": preflight.VENV_PYTHON,
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
        "runtime_pins": "torch==2.11.0",
    }
    return {"text": "- lane: test job\n", "fields": fields}


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({}, None),
        ({"ok": False, "failures": ["dew.kernels has no fla_gdn"]}, "it failed: dew.kernels has no fla_gdn"),
        ({"commit": "0" * 40}, "its commit is '0000"),
        ({"hardware": "gpu:G4"}, "its hardware is 'gpu:G4', not 'gpu:T4'"),
        ({"command": "python other.py"}, "its command is 'python other.py'"),
        ({"tool": "0" * 64}, "its tool is"),
        ({"runtime": []}, "its runtime is"),
        ({"interpreter": "venv"}, "its python is '3.13', not '3.12'"),
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


KERNELS_SOURCE = "KERNELS = {'gated_delta_rule': {'sm80': 'fla'}, 'attention': {'sm80': 'flash'}}\n"


def kernel_tree(root: Path, kernels: str = KERNELS_SOURCE) -> Path:
    write(root, {
        "src/dew/__init__.py": "", "src/dew/nn/__init__.py": "", "src/dew/nn/kernels/__init__.py": "",
        "src/dew/nn/kernels/generation.py": kernels,
        "src/dew/nn/kernels/rule.py": "def rule(): return 1\n",
        "src/dew/nn/linear.py": "from dew.nn.kernels import rule\n",
        "src/dew/data.py": "x = 1\n",
    })
    return root


def test_the_kernel_digest_covers_the_kernels_and_their_callers_only(tmp_path):
    tree = kernel_tree(tmp_path)
    first = preflight.kernel_digest(tree)
    (tree / "src/dew/data.py").write_text("x = 2\n")
    assert preflight.kernel_digest(tree) == first
    for changed in ("src/dew/nn/linear.py", "src/dew/nn/kernels/rule.py"):
        (tree / changed).write_text((tree / changed).read_text() + "# changed\n")
        assert preflight.kernel_digest(tree) != first
        first = preflight.kernel_digest(tree)


def test_pallas_kernel_choices_need_the_labs_record_of_their_exact_code(tmp_path, monkeypatch):
    """A ref whose KERNELS run Pallas kernels on the target GPU needs the kernel lab's record for its
    kernels' digest; a ref naming none (here the A100's flash attention is CUDA, not Pallas) needs
    none, and a G4 with no row runs no Pallas kernel."""
    tree = kernel_tree(tmp_path)
    records = {}
    monkeypatch.setattr(preflight, "store_read", lambda key: (records.get(key), "absent"))
    failures: list[str] = []
    found = preflight.lab_verified(tree, "gpu:A100", failures)
    assert found["pallas"] == ["gated_delta_rule=fla"]
    assert failures and "not verified on a100" in failures[0]
    records[f"lab/verified/a100/{found['digest']}.json"] = {"ok": True, "ref": "abc", "gpu": "a100"}
    failures.clear()
    assert preflight.lab_verified(tree, "gpu:A100", failures)["record"]["ref"] == "abc" and not failures
    assert preflight.lab_verified(tree, "gpu:G4", failures) == {"generation": "sm120", "pallas": []}
    assert not failures


def test_a_second_ref_is_checked_out_only_when_declared_and_fetched(tmp_path, monkeypatch):
    """c58's fallback checkout failed on the VM, which had fetched only the job's ref, and the
    script went on at the job's ref. A script's other ref must be declared (`--also`), and the
    entry's `refs:` must fetch it."""
    tree = tmp_path / "dew"
    commit = repository(tree, {"src/dew/__init__.py": "", "src/dew/kernels.py": "rule = 1\n"})
    subprocess.run(["git", "-C", str(tree), "-c", "user.name=T", "-c", "user.email=t@e.invalid",
                    "commit", "--allow-empty", "-qm", "fallback"], check=True)
    fallback = subprocess.check_output(["git", "-C", str(tree), "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setattr(preflight, "REPO", tree)
    monkeypatch.setattr(preflight, "VENV_PYTHON", "{}.{}".format(*sys.version_info[:2]))
    script = tmp_path / "run.sh"
    script.write_text(f"git checkout -q {commit}\ngit checkout -q {fallback}\n")
    args = argparse.Namespace(ref=commit, hardware="gpu:A100", command="bash /content/c/run.sh",
                              inputs=[str(script)], pin=[], origin="origin", accept=[], also=[],
                              dew_python=sys.executable)
    undeclared = preflight.preflight(args, tmp_path / "work")
    assert not undeclared["ok"] and "nor a declared --also ref" in undeclared["failures"][0]
    args.also = [fallback]
    declared = preflight.preflight(args, tmp_path / "work2")
    assert declared["ok"] and declared["refs"] == [fallback]
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(declared))
    request = {"commit": commit, "hardware": "gpu:A100", "command": args.command, "runtime": []}
    with pytest.raises(ValueError, match="refs: does not fetch"):
        preflight.verify(path, request, [])
    assert preflight.verify(path, {**request, "refs": [fallback]}, [])
