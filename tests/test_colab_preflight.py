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
    # The script runs Dew's code, and this ref predates the lowering dry run: both fail, by name.
    assert receipt["failures"][0] == "rules.py:2: dew.kernels.fla_gdn: dew.kernels has no fla_gdn"
    assert len(receipt["failures"]) == 2 and "predates the lowering dry run" in receipt["failures"][1]
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
    monkeypatch.setattr(preflight, "store_read", lambda key, config=None: (records.get(key), "absent"))
    failures: list[str] = []
    found = preflight.lab_verified(tree, "gpu:A100", failures)
    assert found["pallas"] == ["gated_delta_rule=fla"]
    assert failures and "not verified on a100" in failures[0]
    records[f"lab/verified/a100/{found['digest']}.json"] = {"ok": True, "ref": "abc", "gpu": "a100"}
    failures.clear()
    assert preflight.lab_verified(tree, "gpu:A100", failures)["record"]["ref"] == "abc" and not failures
    assert preflight.lab_verified(tree, "gpu:G4", failures) == {"generation": "sm120", "pallas": []}
    assert not failures
    # A script that names a Pallas implementation needs the record even where KERNELS names none.
    script = {"ab.py": "rule(q, implementation='fla')\n"}
    sites = preflight.explicit_pallas(script, "clef --delta-rule-impl pallas")
    assert sites == ["<command>:1", "ab.py:1"]
    preflight.lab_verified(tree, "gpu:G4", failures, sites)
    assert failures and "not verified on g4" in failures[0]
    # The pinned JAX is part of the digest: a JAX bump needs the lab again.
    digest = preflight.kernel_digest(tree)
    (tree / "constraints.txt").write_text("jax==0.11.2\njaxlib==0.11.2\nnumpy==2.0\n")
    bumped = preflight.kernel_digest(tree)
    (tree / "constraints.txt").write_text("jax==0.11.3\njaxlib==0.11.3\nnumpy==2.0\n")
    newer = preflight.kernel_digest(tree)
    assert len({digest, bumped, newer}) == 3
    (tree / "constraints.txt").write_text("jax==0.11.3\njaxlib==0.11.3\nnumpy==2.1\n")
    assert preflight.kernel_digest(tree) == newer


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


def test_a_setting_known_to_break_gpu_runs_fails_the_preflight():
    texts = {"drive.py": "env = {**os.environ, 'JAX_EXPLAIN_CACHE_MISSES': '1'}\n"}
    found = preflight.hazards(texts, "python drive.py")
    assert len(found) == 1 and found[0].startswith("drive.py:1 sets JAX_EXPLAIN_CACHE_MISSES")
    assert preflight.hazards({"drive.py": "print(1)\n"}, "python drive.py") == []
    made = {"run.sh": "JAX_EXPLAIN_CACHE_MISSES=1 python x.py\n",
            "x.py": "os.environ['JAX_EXPLAIN_CACHE_MISSES'] = '1'\n"
                    "jax.config.update('jax_explain_cache_misses', True)\n"}
    assert [hazard.split(" sets ")[0] for hazard in preflight.hazards(made, "bash run.sh")] == [
        "run.sh:1", "x.py:1", "x.py:2"]
    named = {"c50_profile.py": '"""JAX\'s cache-miss explainer (JAX_EXPLAIN_CACHE_MISSES) is not used."""\n'}
    assert preflight.hazards(named, "python c50_profile.py") == []



def test_every_request_that_runs_dews_code_has_a_step_to_lower(tmp_path):
    """c58's command reached the Clef recipe through the profile script it staged. A kernel bench and
    the lab's verify run Dew's kernels without training and lower as much: two short verifies once
    launched on a kernel whose inline assembly did not lower. Only a request running no Dew code,
    a torch reference, has nothing to lower."""
    texts = {"c58.sh": 'bash -c "$C58_PROFILE"\n',
             "c50_profile.py": "import subprocess\nsubprocess.run(['python', 'recipes/decision/clef.py'])\n"}
    assert preflight.trains(texts, "C58_PROFILE='python c50_profile.py' bash c58.sh") == ["c50_profile.py:2"]
    assert preflight.trains({"run.py": "from dew.training import Trainer\n"}, "python run.py") == ["run.py:1"]
    bench, _, _, _ = preflight.uses("from dew.nn.kernels import fla_gdn\n", "c54_fla_port.py")
    assert preflight.trains({"c54_fla_port.py": ""}, "python c54_fla_port.py") == []
    assert preflight.runs_dew(bench) == ["c54_fla_port.py:1"]
    verify = [{"module": "tools.kernel_lab.verify", "attrs": [], "where": "<shell>"}]
    assert preflight.runs_dew(verify) == ["<shell>"]
    torch, _, _, _ = preflight.uses("import torch\n", "reference.py")
    assert preflight.runs_dew(torch) == []
    failures: list[str] = []
    assert preflight.lowered(tmp_path, "gpu:A100", sys.executable, [], {}, failures) == {
        "skipped": "the request runs no Dew code"}
    assert not failures
    preflight.lowered(tmp_path, "gpu:A100", sys.executable, ["<shell>"], {}, failures)
    assert len(failures) == 1 and "runs Dew's code (<shell>)" in failures[0]


FAKE_STEP = """import json, os, sys
print("lowering", file=sys.stderr)
print(json.dumps({"kernels": {"args": " ".join(sys.argv[1:-1]), "platforms": os.environ["JAX_PLATFORMS"],
                              "flags": os.environ["XLA_FLAGS"], "path": os.environ["PYTHONPATH"]}}))
"""


def test_a_training_request_lowers_the_refs_clef_step_under_its_own_jax_settings(tmp_path):
    """The ref's tools/clef_step.py lowers its Clef step for the GPU's generation over one device, on
    this CPU whatever platform the request names, under the JAX and XLA settings the request's
    command and shell scripts assign; the receipt keeps the generation, the device count, the
    settings, the script's digest and the kernels it lowered. A ref without the script, or one
    whose step does not lower, fails."""
    tree = tmp_path / "dew"
    failures: list[str] = []
    found = preflight.lowered(tree, "gpu:A100", sys.executable, ["<command>:1"], {}, failures)
    assert len(failures) == 1 and "predates the lowering dry run" in failures[0] and "kernels" not in found
    write(tree, {"tools/clef_step.py": FAKE_STEP})
    shell = {"run.sh": "export XLA_FLAGS='--xla_a=1 --xla_b=2'\nJAX_PLATFORMS=cuda python recipes/d/clef.py\n"
                       "JAX_Z=$W true\n"}
    env = preflight.lowering_env(shell, "JAX_TRACEBACK_FILTERING=off bash run.sh")
    assert env == {"JAX_TRACEBACK_FILTERING": "off", "XLA_FLAGS": "--xla_a=1 --xla_b=2",
                   "JAX_PLATFORMS": "cuda"}
    failures.clear()
    found = preflight.lowered(tree, "gpu:G4", sys.executable, ["run.sh:2"], env, failures)
    assert not failures
    assert found == {"generation": "sm120", "devices": 1, "env": env, "sites": ["run.sh:2"], "model": "tiny",
                     "geometry": {}, "script": hashlib.sha256(FAKE_STEP.encode()).hexdigest(),
                     "kernels": {"args": "--lower-for sm120 --devices 1 --work", "platforms": "cpu",
                                 "flags": "--xla_a=1 --xla_b=2", "path": str(tree / "src")}}
    shape = {"like": "Qwen/Qwen3.5-9B", "max-len": "3072"}
    found = preflight.lowered(tree, "gpu:G4", sys.executable, ["run.sh:2"], env, failures, shape)
    assert found["model"] == "Qwen/Qwen3.5-9B" and found["geometry"] == shape and not failures
    assert found["kernels"]["args"] == ("--lower-for sm120 --devices 1 --like=Qwen/Qwen3.5-9B --max-len=3072 "
                                        "--work")
    write(tree, {"tools/clef_step.py": "import sys\nsys.exit('ValueError: not enough values to unpack')\n"})
    found = preflight.lowered(tree, "gpu:A100", sys.executable, ["run.sh:2"], {}, failures)
    assert len(failures) == 1 and "does not lower for sm80" in failures[0]
    assert "not enough values to unpack" in failures[0] and found["kernels"] is None


LAB_SCRIPT = 'git checkout -q "$LAB_REF"\npython -m tools.kernel_lab.server --gpu a100 --work /content/lab\n'


def test_the_kernel_labs_own_session_needs_no_record_it_exists_to_write(tmp_path, monkeypatch):
    """A session that runs the kernel lab alone, its server, client and gate, produces the lab's
    record, so its preflight records the bootstrap; a session that also runs a script of its own or
    trains is no lab session, and still needs the record."""
    command = "bash /content/kernel-lab/kernel-lab.sh"
    texts = {"kernel-lab.sh": LAB_SCRIPT}
    assert preflight.lab_session(preflight.shell(command, texts), texts, command)
    bench = {**texts, "c54.py": "from dew.nn.kernels import fla_gdn\n"}
    benched = command + " && python /content/c54/c54.py"
    assert not preflight.lab_session(preflight.shell(benched, bench), bench, benched)
    training = {"kernel-lab.sh": LAB_SCRIPT + "python -m tools.kernel_lab.submit --verify --ref X\n"
                                              "python -m dew.training.launch recipes/decision/clef.py\n"}
    assert not preflight.lab_session(preflight.shell(command, training), training, command)
    tree = kernel_tree(tmp_path / "dew")
    write(tree, {"tools/__init__.py": "", "tools/kernel_lab/__init__.py": "",
                 "tools/kernel_lab/server.py": "",
                 "tools/clef_step.py": "import json\nprint(json.dumps({'kernels': {'rms_norm': 'xla'}}))\n"})
    commit = repository(tree, {})
    monkeypatch.setattr(preflight, "REPO", tree)
    monkeypatch.setattr(preflight, "VENV_PYTHON", "{}.{}".format(*sys.version_info[:2]))
    monkeypatch.setattr(preflight, "store_read", lambda key, config=None: (None, "absent"))
    script = tmp_path / "kernel-lab.sh"
    script.write_text(LAB_SCRIPT)
    args = argparse.Namespace(ref=commit, hardware="gpu:A100", command=f"LAB_REF={commit} {command}",
                              inputs=[str(script)], pin=[], origin="origin", accept=[], also=[],
                              dew_python=sys.executable)
    receipt = preflight.preflight(args, tmp_path / "work")
    assert receipt["ok"], receipt["failures"]
    assert receipt["kernel_lab"]["bootstrap"] == "this session produces the record"
    # The lab's server compiles Dew's kernels on the GPU, so its step lowers here first.
    assert receipt["lowered"]["generation"] == "sm80" and receipt["lowered"]["kernels"]


def test_a_session_that_trains_only_after_the_lab_gate_stops_it_is_gated_at_runtime():
    """The gate must stop the session where it fails, by `set -e` before it or `|| exit` on its line,
    and stand before everything that trains, here a staged driver that runs the Clef recipe."""
    driver = {"train.py": "subprocess.run([python, 'recipes/decision/clef.py'])\n"}
    verified = ("set -eu\n# then train.py, once the gate holds\n"
                "python -m tools.kernel_lab.verify --gpu a100 --ref X\n")
    gated = {"run.sh": verified + "python -m tools.colab.lab_gate --gpu a100\npython train.py\n", **driver}
    assert preflight.runtime_gate(gated, "bash run.sh") == "run.sh:4"
    stopped = {"run.sh": "python -m tools.colab.lab_gate --gpu a100 || exit 1\npython train.py\n", **driver}
    assert preflight.runtime_gate(stopped, "bash run.sh") == "run.sh:1"
    for unsafe in ("python -m tools.colab.lab_gate --gpu a100\npython train.py\n",
                   "python train.py\npython -m tools.colab.lab_gate --gpu a100 || exit 1\n",
                   "python -m tools.colab.lab_gate --gpu a100 || exit 1\necho done\n"):
        assert preflight.runtime_gate({"run.sh": unsafe, **driver}, "bash run.sh") is None, unsafe


def test_the_lab_gate_reads_what_the_preflight_reads(tmp_path, monkeypatch):
    """The gate fails a ref whose KERNELS row names Pallas kernels on the GPU and whose digest has no
    ok record, passes one that has it, and passes a GPU whose row names none."""
    from tools.colab import lab_gate

    tree = kernel_tree(tmp_path)
    records: dict = {}
    monkeypatch.setattr(preflight, "store_read", lambda key, config=None: (records.get(key), "absent"))
    found, failures = lab_gate.gate(tree, "a100")
    assert failures and "not verified on a100" in failures[0]
    records[f"lab/verified/a100/{found['digest']}.json"] = {"ok": True, "ref": "abc", "gpu": "a100"}
    assert lab_gate.gate(tree, "a100")[1] == []
    assert lab_gate.gate(tree, "g4") == ({"generation": "sm120", "pallas": []}, [])


def test_the_lowered_step_takes_the_checkpoint_and_rows_the_request_trains():
    """c58's profile script handed the recipe its checkpoint in a Python list and its command its batch:
    the step lowers with that checkpoint's widths at those rows. A checkpoint on the VM's disk, which
    this host lacks, leaves the tiny model's widths and is named."""
    texts = {"c50_profile.py": 'flags = ["--pretrained", "Qwen/Qwen3.5-0.8B", "--revision", "abc123",\n'
                               '         "--max-len", "3072"]\n',
             "c58.sh": "python c50_profile.py --trainer.batch-size 8\n"}
    assert preflight.geometry(texts, "bash c58.sh") == {"like": "Qwen/Qwen3.5-0.8B", "revision": "abc123",
                                                        "max-len": "3072", "batch": "8"}
    local = "python recipes/decision/clef.py --pretrained /content/ckpt --revision None"
    assert preflight.geometry({}, local) == {"local": "/content/ckpt"}
    computed = {"c50.py": 'run(["--pretrained", backbone[0], "--max-len", str(row)])\n'}
    assert preflight.geometry(computed, "python c50.py") == {}
    looped = "python recipes/decision/clef.py --loop-layers 2 6 --model.attention-impl xla --remat minimal"
    assert preflight.geometry({}, looped) == {"not applied": "--loop-layers, --model.attention-impl, --remat"}
