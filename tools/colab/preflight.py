"""Prove a Colab request's scripts on CPU before a GPU is paid for.

Every symbol a request's Python inputs import must resolve at the request's
exact Dew ref and at the exact package versions its shell inputs install. A
package that cannot import without a GPU (FlashQLA, whose import asks TileLang
for the device's compute capability) is checked against its installed sources
instead, following the branches the target GPU takes, including imports made
inside functions. `colab_batch.plan_jobs` requires the receipt this writes.

    python -m tools.colab.preflight --ref SHA --hardware gpu:G4 \
        --command 'C56_REF=SHA bash /content/c56/c56-g4.sh' --pin torch==2.11.0 INPUT...
"""

import argparse
import ast
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from tools.colab.common import REPO, VENV_PYTHON, git
from tools.python_uses import dotted, uses

ARCH = {"gpu:T4": "7.5", "gpu:L4": "8.9", "gpu:A100": "8.0", "gpu:G4": "12.0"}
# The VM's interpreters: Colab's /usr/bin/python3 (c56's tracebacks, /usr/local/lib/python3.13), and
# the job's venv, which `python` and `pip` name (colab_batch_vm).
INTERPRETERS = {"system": "3.13", "venv": VENV_PYTHON}
# A receipt holds for the preflight that wrote it: this file and the syntax reader it uses.
TOOL = hashlib.sha256(
    b"".join(path.read_bytes() for path in (Path(__file__), REPO / "tools/python_uses.py"))
).hexdigest()
LOCAL = ("dew", "tools")
SCRIPTS = (".py", ".sh", ".bash")
VALUED = {"-c", "--constraint", "-r", "--requirement", "-i", "--index-url", "--extra-index-url", "-t"}
VALUED |= {"--target", "-f", "--find-links", "--python", "--prefix", "--root", "--torch-backend"}
UNKNOWN = object()


def facts() -> dict:
    keys = ("checkouts", "modules", "runs", "sources", "failures", "interpreters")
    return {"pins": {}, "environments": {}, **{key: [] for key in keys}}


def merge(into: dict, found: dict) -> None:
    for key, value in found.items():
        into[key].update(value) if isinstance(value, dict) else into[key].extend(value)


def shell_facts(text: str, env: dict[str, str], depth: int = 0) -> dict:
    """The pins, checkouts, Python runs and inline programs a shell text names."""
    found = facts()
    env = dict(env)
    for line in re.sub(r"\\\n", " ", text).splitlines():
        lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        try:
            words = list(lexer)
        except ValueError as error:
            found["failures"].append(f"cannot parse shell line {line.strip()!r}: {error}")
            continue
        local, command, skip = dict(env), [], False
        for word in [*words, ";"]:
            if skip:
                skip = False
            elif word in {";", "&&", "||", "|", "&", "(", ")"}:
                if not command or command[0] == "export":
                    env.update(local)
                program(command, local, found, depth)
                local, command = dict(env), []
            elif re.fullmatch(r"[<>&]*[<>][<>&]*", word):
                if command and command[-1].isdigit():
                    command.pop()
                skip = True
            elif (not command or command == ["export"]) and re.fullmatch(r"[A-Za-z_]\w*=.*", word):
                key, _, value = word.partition("=")
                local[key] = expand(value, local)
            else:
                command.append(expand(word, local))
    return found


def expand(word: str, env: dict[str, str]) -> str:
    return re.sub(r"\$\{?([A-Za-z_]\w*)\}?", lambda match: env.get(match[1], match[0]), word)


def program(words: list[str], env: dict[str, str], found: dict, depth: int) -> None:
    names = [Path(word).name for word in words]
    pattern = r"python[\d.]*|pip\d?|uv|git|bash|sh"
    start = next((i for i, item in enumerate(names) if re.fullmatch(pattern, item)), None)
    if start is None:
        return
    head, rest = names[start], words[start + 1 :]
    system = words[start].startswith("/usr/")
    if head.startswith("pip") or (head == "uv" and rest[:2] == ["pip", "install"]):
        pins(rest[1:] if head == "uv" else rest, found, system="--system" in rest)
    elif head == "uv" and rest[:1] == ["run"]:
        python = next((i for i, word in enumerate(rest) if Path(word).name.startswith("python")), None)
        if python is not None:
            program(rest[python:], env, found, depth)
    elif head.startswith("python"):
        options = iter(rest)
        for word in options:
            if word in ("-X", "-W"):
                next(options, None)
            elif word == "-m":
                module = next(options, "")
                if module == "pip":
                    pins(list(options), found, system)
                else:
                    found["modules"].append(module)
                return
            elif word == "-c":
                found["sources"].append(("<python -c>", next(options, "")))
                return
            elif not word.startswith("-"):
                found["runs"].append(word)
                return
    elif head == "git" and rest[:1] == ["checkout"]:
        found["checkouts"].extend(word for word in rest[1:] if not word.startswith("-"))
    elif head in ("bash", "sh") and rest[:1] == ["-c"] and len(rest) > 1:
        if re.fullmatch(r"\$\{?\w+\}?", rest[1]):
            found["failures"].append(f"runs {rest[1]}, which the command does not set")
        elif depth < 4:
            merge(found, shell_facts(rest[1], env, depth + 1))
    elif head in ("bash", "sh") and rest and not rest[0].startswith("-"):
        found["runs"].append(rest[0])
        found["environments"][Path(rest[0]).name] = env


def pins(words: list[str], found: dict, system: bool) -> None:
    if words[:1] != ["install"]:
        return
    found["interpreters"].append("system" if system else "venv")
    options = iter(words[1:])
    for word in options:
        if word in VALUED:
            if word in ("-r", "--requirement"):
                found["failures"].append("a requirements file is not preflighted; name each pin")
            next(options, None)
        elif word.startswith(("-", ".", "/")):
            continue
        elif match := re.fullmatch(r"([A-Za-z0-9][\w.-]*)(?:\[[\w,.-]+\])?==([\w.+!-]+)", word):
            found["pins"][match[1].lower().replace("_", "-")] = match[2]
        else:
            found["failures"].append(f"pip installs {word!r} without an exact version")


RESOLVE = r"""
import importlib, inspect, json, sys
chains, bound = json.load(sys.stdin)
bound = {tuple(item) for item in bound}
found = []
for chain in chains:
    module = chain["module"]
    try:
        value = importlib.import_module(module)
    except ModuleNotFoundError as error:
        missing = error.name or ""
        absent = module == missing or module.startswith(missing + ".")
        found.append(["absent" if absent else "import", f"{type(error).__name__}: {error}"])
        continue
    except BaseException as error:
        found.append(["import", f"{type(error).__name__}: {error}"])
        continue
    name, problem = module, None
    for attr in chain["attrs"]:
        if hasattr(value, attr):
            value = getattr(value, attr)
        elif inspect.ismodule(value):
            try:
                value = importlib.import_module(value.__name__ + "." + attr)
            except ModuleNotFoundError:
                problem = None if (name, attr) in bound else f"{name} has no {attr}"
                break
        else:
            problem = None if (name, attr) in bound else f"{name} has no {attr}"
            break
        name = name + "." + attr
    root = sys.modules.get(module.split(".")[0])
    found.append(["attr", problem] if problem else ["ok", getattr(root, "__file__", "")])
print(json.dumps(found))
"""


def resolve(python: str, chains: list[dict], bound: set, env: dict | None = None, cwd: Path | None = None):
    """Each chain's [kind, detail], importing one top-level package an interpreter, as one can kill it."""
    groups: dict[str, list[int]] = {}
    for index, chain in enumerate(chains):
        groups.setdefault(top(chain), []).append(index)
    found: list = [None] * len(chains)
    for name, members in groups.items():
        request = json.dumps([[chains[index] for index in members], sorted(bound)])
        try:
            result = subprocess.run(
                [python, "-c", RESOLVE],
                input=request,
                capture_output=True,
                text=True,
                env=env,
                cwd=cwd,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            answers = [["import", f"importing {name} took over 300 s"]] * len(members)
        else:
            answers = answered(result, name, len(members))
        for index, answer in zip(members, answers, strict=True):
            found[index] = answer
    return found


def answered(result: subprocess.CompletedProcess[str], name: str, count: int) -> list:
    """The resolver's answers, or, where importing `name` killed it, that for each of its chains."""
    try:
        return json.loads(result.stdout.splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        lines = result.stderr.strip().splitlines() or ["no output"]
        last = next((line for line in lines if line.startswith("Fatal Python error")), lines[-1])
        return [["import", f"importing {name} ended the interpreter ({result.returncode}): {last}"]] * count


class Sources:
    """Static name resolution over installed sources, taking the target GPU's branches."""

    def __init__(self, roots: list[Path], arch: str, scope: set[str], python: str = INTERPRETERS["system"]):
        self.roots, self.arch, self.scope = roots, arch, scope
        self.python = tuple(int(part) for part in python.split("."))
        self.guarded: list[str] = []
        self.tables: dict[str, tuple[dict, bool] | None] = {}
        self.trees: dict[Path, ast.Module] = {}

    def path(self, module: str) -> Path | None:
        for root in self.roots:
            base = root.joinpath(*module.split("."))
            for candidate in (base / "__init__.py", base.with_suffix(".py")):
                if candidate.is_file():
                    return candidate
            if base.is_dir() or any(base.parent.glob(base.name + ".*.so")):
                return base
        return None

    def table(self, module: str) -> tuple[dict, bool] | None:
        """Names the module binds at import, and whether it answers any name (a module __getattr__)."""
        if module not in self.tables:
            self.tables[module] = None
            path = self.path(module)
            if path is None or path.suffix != ".py":
                self.tables[module] = ({}, path is not None)
                return self.tables[module]
            names: dict = {}
            # A module that writes its globals() or answers __getattr__ binds names no parse can list.
            wildcard = "globals()" in path.read_text(errors="replace")
            package = module if path.name == "__init__.py" else module.rpartition(".")[0]
            for node, _ in self.statements(path, functions=False):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        names[alias.asname or alias.name.split(".")[0]] = (
                            ("module", alias.name) if alias.asname else ("module", alias.name.split(".")[0])
                        )
                elif isinstance(node, ast.ImportFrom):
                    base = absolute(node, package)
                    for alias in node.names:
                        if alias.name == "*":
                            star = self.table(base)
                            if star:
                                names.update({key: item for key, item in star[0].items() if key[0] != "_"})
                                wildcard |= star[1]
                        else:
                            names[alias.asname or alias.name] = ("from", base, alias.name)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names[node.name] = ("value",)
                    wildcard |= node.name == "__getattr__"
                elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For, ast.With)):
                    if isinstance(node, ast.With):
                        targets = [item.optional_vars for item in node.items if item.optional_vars]
                    else:
                        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        for name in ast.walk(target):
                            if isinstance(name, ast.Name):
                                names[name.id] = ("value",)
            self.tables[module] = (names, wildcard)
        return self.tables[module]

    def lookup(self, module: str, name: str, seen: frozenset = frozenset()) -> tuple | None:
        if module.split(".")[0] not in self.scope:
            return ("value",)
        table = self.table(module)
        if table is None:
            return ("value",)
        names, wildcard = table
        binding = names.get(name)
        if binding is None:
            if self.path(f"{module}.{name}"):
                return ("module", f"{module}.{name}")
            return ("value",) if wildcard else None
        if binding[0] == "from":
            base, attr = binding[1], binding[2]
            if (base, attr) in seen:
                return ("value",)
            return self.lookup(base, attr, seen | {(base, attr)})
        return binding

    def check(self, module: str, attrs: list[str], bound: set) -> str | None:
        if not self.path(module):
            return f"no module {module}"
        for attr in attrs:
            binding = self.lookup(module, attr)
            if binding is None:
                return None if (module, attr) in bound else f"{module} does not bind {attr} on sm{self.arch}"
            if binding[0] != "module":
                return None
            module = binding[1]
        return None

    def closure(self, starts: set[str], bound: set) -> list[str]:
        """Check every import the reachable modules of the packages in scope make, inside functions too."""
        failures, pending, seen = [], sorted(starts), set()
        while pending:
            module = pending.pop()
            path = self.path(module)
            if module in seen or module.split(".")[0] not in self.scope or not path or path.suffix != ".py":
                continue
            seen.add(module)
            package = module if path.name == "__init__.py" else module.rpartition(".")[0]
            for node, guarded in self.statements(path, functions=True):
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                where = f"{module}:{node.lineno}"
                # An ImportError handler expects its import may miss: noted, not failed.
                found = self.guarded if guarded else failures
                if isinstance(node, ast.Import):
                    targets = [(alias.name, []) for alias in node.names]
                else:
                    names = [alias.name for alias in node.names if alias.name != "*"]
                    targets = [(absolute(node, package), names)]
                for target, names in targets:
                    if target.split(".")[0] not in self.scope:
                        continue
                    if not self.path(target):
                        found.append(f"{where} imports {target}, which is not installed")
                        continue
                    pending.append(target)
                    for name in names:
                        binding = self.lookup(target, name)
                        if binding is None and (target, name) not in bound:
                            found.append(f"{where}: from {target} import {name}: unbound on sm{self.arch}")
                        elif binding and binding[0] == "module":
                            pending.append(binding[1])
        return failures

    def statements(self, path: Path, functions: bool):
        """Statements that run on the target GPU, each with whether an ImportError handler guards it."""
        if path not in self.trees:
            self.trees[path] = ast.parse(path.read_text(errors="replace"), str(path))
        tree = self.trees[path]
        constants: dict = {}

        def walk(body: list, guarded: bool, module: bool):
            for node in body:
                yield node, guarded
                target = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None
                if module and isinstance(target, ast.Name):
                    constants[target.id] = evaluate(node.value, constants, self.arch, self.python)
                if isinstance(node, ast.If):
                    taken = evaluate(node.test, constants, self.arch, self.python)
                    if taken is UNKNOWN or taken:
                        yield from walk(node.body, guarded, module)
                    if taken is UNKNOWN or not taken:
                        yield from walk(node.orelse, guarded, module)
                elif isinstance(node, ast.Try):
                    caught = guarded or any(catches_import(handler) for handler in node.handlers)
                    yield from walk(node.body, caught, module)
                    for part in [*(handler.body for handler in node.handlers), node.orelse, node.finalbody]:
                        yield from walk(part, guarded, module)
                elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                    yield from walk(node.body + node.orelse, guarded, module)
                elif isinstance(node, (ast.With, ast.AsyncWith)):
                    yield from walk(node.body, guarded, module)
                elif isinstance(node, ast.Match):
                    for case in node.cases:
                        yield from walk(case.body, guarded, module)
                elif functions and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    yield from walk(node.body, guarded, module=False)

        yield from walk(tree.body, guarded=False, module=True)


def absolute(node: ast.ImportFrom, package: str) -> str:
    if not node.level:
        return node.module or ""
    parts = package.split(".")
    base = parts[: len(parts) - node.level + 1]
    return ".".join([*base, *([node.module] if node.module else [])])


def catches_import(handler: ast.ExceptHandler) -> bool:
    names = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}
    if handler.type is None:
        return True
    kinds = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(dotted(kind).split(".")[-1] in names for kind in kinds)


def evaluate(node: ast.AST, constants: dict, arch: str, python: tuple = (3, 13)):
    """A condition's value on the target VM, or UNKNOWN: literals, the device's compute capability,
    the platform and the Python version are known, nothing else."""

    def value(item: ast.AST):
        return evaluate(item, constants, arch, python)

    known = {"sys.platform": "linux", "sys.version_info": (*python, 0)}
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id, UNKNOWN)
    if isinstance(node, ast.Attribute):
        return known.get(dotted(node), UNKNOWN)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        items = [value(item) for item in node.elts]
        return UNKNOWN if any(item is UNKNOWN for item in items) else tuple(items)
    if isinstance(node, ast.Call):
        name = dotted(node.func)
        if name.endswith("get_target_compute_version"):
            return arch
        if name.endswith("get_device_capability"):
            return tuple(int(part) for part in arch.split("."))
        return True if name.endswith("cuda.is_available") else UNKNOWN
    if isinstance(node, ast.Subscript):
        index = node.slice
        if isinstance(index, ast.Slice):
            bounds = [None if part is None else value(part) for part in (index.lower, index.upper)]
            index = UNKNOWN if UNKNOWN in bounds or index.step else slice(*bounds)
        else:
            index = value(index)
        container = value(node.value)
        try:
            return UNKNOWN if UNKNOWN in (index, container) else container[index]
        except (IndexError, KeyError, TypeError):
            return UNKNOWN
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        operand = value(node.operand)
        return UNKNOWN if operand is UNKNOWN else not operand
    if isinstance(node, ast.BoolOp):
        values = [value(item) for item in node.values]
        decisive = isinstance(node.op, ast.Or)
        if any(item is not UNKNOWN and bool(item) == decisive for item in values):
            return decisive
        return UNKNOWN if any(item is UNKNOWN for item in values) else not decisive
    if isinstance(node, ast.Compare):
        left = value(node.left)
        for operator, comparator in zip(node.ops, node.comparators, strict=True):
            right = value(comparator)
            if left is UNKNOWN or right is UNKNOWN:
                return UNKNOWN
            try:
                held = COMPARE[type(operator)](left, right)
            except (KeyError, TypeError):
                return UNKNOWN
            if not held:
                return False
            left = right
        return True
    return UNKNOWN


COMPARE = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


def run(args: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, timeout=1800, **kwargs)


def install(pinned: dict[str, str], work: Path, python: str) -> tuple[str | None, list[Path], dict, list]:
    """A CPU environment holding the pins; a pin that cannot install there keeps its sources only."""
    if not pinned:
        return None, [], {}, []
    failures: list[str] = []
    venv = work / "venv"
    made = run(["uv", "venv", "-q", "--python", python, str(venv)])
    if made.returncode:
        return None, [], {}, [f"cannot make a Python {python} environment: {made.stderr[-500:]}"]
    interpreter = str(venv / "bin" / "python")
    specs = [f"{name}=={version}" for name, version in sorted(pinned.items())]
    base = ["uv", "pip", "install", "-q", "--python", interpreter, "--torch-backend", "cpu"]
    sources = work / "sources"
    if run([*base, *specs]).returncode:
        for spec in specs:
            if run([*base, spec]).returncode:
                kept = run(["uv", "pip", "install", "-q", "--no-deps", "--target", str(sources), spec])
                if kept.returncode:
                    failures.append(f"{spec} cannot be fetched: {kept.stderr[-500:]}")
    listed = run(["uv", "pip", "list", "--python", interpreter, "--format", "json"])
    listing = json.loads(listed.stdout or "[]")
    versions = {item["name"].lower().replace("_", "-"): item["version"] for item in listing}
    for source in sources.glob("*.dist-info"):
        name, _, version = source.name.removesuffix(".dist-info").rpartition("-")
        versions.setdefault(name.lower().replace("_", "-"), version + " (sources only)")
    for name, version in pinned.items():
        if versions.get(name, "").split(" ")[0].split("+")[0] != version.split("+")[0]:
            failures.append(f"{name}=={version} is pinned but {versions.get(name, 'nothing')} is installed")
    query = "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    purelib = run([interpreter, "-c", query]).stdout.strip()
    return interpreter, [Path(purelib), sources], versions, failures


def top_levels(roots: list[Path], names: set[str]) -> set[str]:
    """The import names the pinned distributions install."""
    found = set()
    for root in roots:
        for record in root.glob("*.dist-info/RECORD"):
            name = record.parent.name.removesuffix(".dist-info").rpartition("-")[0].lower().replace("_", "-")
            if name not in names:
                continue
            for line in record.read_text().splitlines():
                first = line.split(",")[0].split("/")[0]
                if first.endswith(".py"):
                    first = first[:-3]
                if first and not first.endswith((".dist-info", ".data")) and first.isidentifier():
                    found.add(first)
    return found


def checkout(ref: str, origin: str, work: Path) -> tuple[str, Path]:
    try:
        commit = git(REPO, "rev-parse", f"{ref}^{{commit}}")
    except RuntimeError:
        git(REPO, "fetch", "-q", origin, ref)
        commit = git(REPO, "rev-parse", f"{ref}^{{commit}}")
    clean = not git(REPO, "status", "--porcelain", "--untracked-files=no")
    if clean and git(REPO, "rev-parse", "HEAD") == commit:
        return commit, REPO
    tree = work / "ref"
    git(REPO, "worktree", "add", "-q", "--detach", str(tree), commit)
    return commit, tree


def shell(command: str, texts: dict[str, str]) -> dict:
    """Facts of the command and of every shell input, each script read with the variables its caller sets."""
    found, done = facts(), set()

    def visit(name: str, text: str, env: dict[str, str]) -> None:
        done.add(name)
        own = shell_facts(text, env)
        merge(found, own)
        for path in own["runs"]:
            script = Path(path).name
            if script.endswith(SCRIPTS[1:]) and script in texts and script not in done:
                visit(script, texts[script], own["environments"].get(script, env))

    visit("<command>", command, {})
    for name, text in texts.items():
        if name.endswith(SCRIPTS[1:]) and name not in done:
            visit(name, text, {})
    return found


def preflight(args: argparse.Namespace, work: Path) -> dict:
    arch = ARCH[args.hardware]
    commit, tree = checkout(args.ref, args.origin, work)
    receipt: dict = {"schema": 1, "commit": commit, "hardware": args.hardware, "arch": arch}
    receipt.update(tool=TOOL, preflight=git(REPO, "rev-parse", "HEAD"), command=args.command, inputs={})
    texts: dict[str, str] = {}
    for item in map(Path, args.inputs):
        data = item.read_bytes()
        receipt["inputs"][item.name] = hashlib.sha256(data).hexdigest()
        if item.name.endswith(SCRIPTS):
            texts[item.name] = data.decode()
    found = shell(args.command, texts)
    failures = list(found["failures"])
    # A ref beside the job's own reaches the VM only through the entry's `refs:` (colab_batch adds
    # them to the commits it fetches): a checkout of any other one fails there, and silently in a
    # script that runs on (c58's fallback).
    declared = {resolved(ref, args.origin): ref for ref in args.also}
    receipt["refs"] = sorted(declared)
    for ref in found["checkouts"]:
        target = resolved(ref, args.origin)
        if target != commit and target not in declared:
            failures.append(f"a script checks out {ref}, which is neither the preflighted {commit} "
                            "nor a declared --also ref")
    targets = set(found["interpreters"]) or {"system"}
    if len(targets) > 1:
        failures.append("pins go to both the system Python and the job's venv; preflight one interpreter")
    receipt["interpreter"] = interpreter = min(targets)
    receipt["python"] = INTERPRETERS[interpreter]
    receipt["dew_python"] = version(args.dew_python)
    if receipt["dew_python"] != VENV_PYTHON:
        failures.append(f"Dew resolved under Python {receipt['dew_python']}, not the VM venv's {VENV_PYTHON}")
    chains, bound, unchecked = symbols(found, texts, (tree, commit, work), args.dew_python, failures)
    receipt["unchecked"] = unchecked
    runtime = sorted(pin.lower() for pin in args.pin)
    pinned = {**found["pins"], **dict(pin.split("==", 1) for pin in runtime)}
    receipt.update(runtime=runtime, pins=pinned)
    interpreter, roots, versions, problems = install(pinned, work, receipt["python"])
    failures += problems
    receipt["installed"] = versions
    other = [chain for chain in chains if not chain.get("local")]
    results = resolve(interpreter, other, bound) if interpreter else [["absent", "no pins"]] * len(other)
    absent = [chain for chain, result in zip(other, results, strict=True) if result[0] == "absent"]
    retried = iter(resolve(args.dew_python, absent, bound))
    results = [next(retried) if result[0] == "absent" else result for result in results]
    # Only a pinned package that cannot import on CPU is read from its sources; the closure stays
    # inside it, as what it imports from importable packages was proven by importing them.
    failed = {top(chain): why for chain, (kind, why) in zip(other, results, strict=True) if kind == "import"}
    static = Sources(roots, arch, set(failed) & top_levels(roots, set(pinned)), receipt["python"])
    for chain, (kind, detail) in zip(other, results, strict=True):
        if kind == "import" and top(chain) in static.scope:
            problem = static.check(chain["module"], chain["attrs"], bound)
            if problem:
                failures.append(f"{label(chain)}: {problem} (from sources; on CPU: {detail})")
        elif kind == "absent":
            failures.append(f"{label(chain)}: no pin or Dew dependency provides it ({detail})")
        elif kind != "ok":
            failures.append(f"{label(chain)}: {detail}")
    failures += static.closure({chain["module"] for chain in other if top(chain) in static.scope}, bound)
    receipt.update(static={name: failed[name] for name in sorted(static.scope)}, guarded=static.guarded)
    failures += hazards(texts, args.command)
    receipt["kernel_lab"] = lab_verified(tree, args.hardware, failures, explicit_pallas(texts, args.command))
    receipt["accepted"], failures = accept(failures, args.accept)
    receipt.update(checked=len(chains), failures=failures, ok=not failures)
    return receipt


def symbols(found: dict, texts: dict, where: tuple, python: str, failures: list) -> tuple:
    """The chains the scripts read, Dew's and the scripts' own (marked local) already resolved at the
    ref; the attributes they assign on modules; the imports no parse can name."""
    tree, commit, work = where
    sources = [(name, text) for name, text in texts.items() if name.endswith(".py")] + found["sources"]
    for path in found["runs"]:
        if Path(path).name in texts:
            continue
        if Path(path).is_absolute() or not (tree / path).is_file():
            failures.append(f"runs {path}, which is neither an input nor in {commit[:12]}")
        elif path.endswith(".py"):
            sources.append((path, (tree / path).read_text()))
    chains = [{"module": module, "attrs": [], "where": "<shell>"} for module in found["modules"]]
    bound: set = set()
    paths: set[Path] = set()
    unchecked: list[str] = []
    for name, text in sources:
        try:
            more, assigned, added, dynamic = uses(text, name)
        except SyntaxError as error:
            failures.append(f"{name} does not parse: {error}")
            continue
        chains += more
        bound |= assigned
        unchecked += dynamic
        paths |= {tree / path for path in added if not os.path.isabs(path) and (tree / path).is_dir()}
    unique = {(chain["module"], tuple(chain["attrs"])): chain for chain in chains}
    chains = [chain for chain in unique.values() if top(chain) not in sys.stdlib_module_names]
    # A script imports its sibling inputs by name, from the directory it runs in.
    siblings = work / "inputs"
    siblings.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        if name.endswith(".py"):
            (siblings / name).write_text(text)
    ours = {*LOCAL, *(Path(name).stem for name in texts if name.endswith(".py"))}
    for folder in paths:
        ours |= {item.stem for item in folder.iterdir() if item.suffix == ".py" or item.is_dir()}
    local = [chain for chain in chains if top(chain) in ours]
    path = [str(siblings), *map(str, sorted(paths)), str(tree / "src"), str(tree)]
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(path)}
    for chain, (kind, detail) in zip(local, resolve(python, local, bound, environment, tree), strict=True):
        chain["local"] = True
        if kind != "ok":
            failures.append(f"{label(chain)}: {detail}")
        elif top(chain) == "dew" and not str(detail).startswith(str(tree)):
            failures.append(f"dew resolved from {detail}, not from {tree}")
    return chains, bound, unchecked


def accept(failures: list[str], accepted: list[str]) -> tuple[list[dict], list[str]]:
    """Failures a request accepts as `TEXT::REASON`, each matching at least one; the rest stand."""
    kept, recorded = list(failures), []
    for item in accepted:
        text, _, reason = item.partition("::")
        matched = [failure for failure in kept if text in failure]
        if not matched or not reason.strip():
            kept.append(f"--accept {item!r} matches no failure or gives no reason")
            continue
        kept = [failure for failure in kept if failure not in matched]
        recorded += [{"failure": failure, "reason": reason.strip()} for failure in matched]
    return recorded, kept


def version(python: str) -> str:
    query = "import sys; print('%d.%d' % sys.version_info[:2])"
    return subprocess.run([python, "-c", query], capture_output=True, text=True, check=True).stdout.strip()


def top(chain: dict) -> str:
    return chain["module"].split(".")[0]


def resolved(ref: str, origin: str) -> str:
    """`ref` as a commit of this clone, fetched from `origin` if it lacks it; `ref` itself if it
    names none."""

    def known() -> bool:
        return subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{ref}^{{commit}}"],
                              capture_output=True, check=False).returncode == 0

    if not known():
        subprocess.run(["git", "-C", str(REPO), "fetch", "-q", origin, ref], capture_output=True, check=False)
    return git(REPO, "rev-parse", f"{ref}^{{commit}}") if known() else ref


# Kernel choices that run Dew's own Pallas (Triton) kernels, whose lowering only a GPU proves.
PALLAS_CHOICES = {"pallas", "fla", "triton"}
GENERATIONS = {"gpu:T4": "sm75", "gpu:L4": "sm89", "gpu:A100": "sm80", "gpu:G4": "sm120"}
LAB_GPUS = {"gpu:A100": "a100", "gpu:G4": "g4", "gpu:L4": "l4", "gpu:T4": "t4"}


def kernel_digest(tree: Path) -> str:
    """The content of Dew's kernels and of every source file that calls them, at `tree`, and the
    jax and jaxlib its constraints pin, whose Triton lowering is half of a kernel's fate: a ref's
    kernels are the kernel lab's verified ones exactly when this matches the lab's record."""
    source = tree / "src" / "dew"
    calls = re.compile(r"(?:from|import)\s+(?:dew\.nn\.)?(?:\.+)?kernels\b|dew\.nn\.kernels")
    files = sorted(path for path in source.rglob("*.py")
                   if "kernels" in path.relative_to(source).parts or calls.search(path.read_text()))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(tree)).encode() + b"\0" + path.read_bytes() + b"\0")
    constraints = tree / "constraints.txt"
    pinned = constraints.read_text() if constraints.exists() else ""
    pins = re.findall(r"^(?:jax|jaxlib|jax-cuda\d+-\w+)==\S+", pinned, re.M)
    digest.update("\n".join(sorted(pins)).encode())
    return digest.hexdigest()


# An explicit Pallas choice in a staged script or the command, which `auto`'s KERNELS row does not see.
EXPLICIT = re.compile(r"""(?:implementation|_impl)["']?\s*[=:]\s*["'](pallas|fla|triton)["']"""
                      r"""|--[\w.-]*impl[\w.-]*[= ](pallas|fla|triton)\b""")


def explicit_pallas(texts: dict[str, str], command: str) -> list[str]:
    """Each site that asks for a Pallas implementation by name, as `file:line`."""
    sites = []
    for name, text in [("<command>", command), *texts.items()]:
        for number, line in enumerate(text.splitlines(), 1):
            if EXPLICIT.search(line):
                sites.append(f"{name}:{number}")
    return sites


# Settings that break a GPU run whatever the code: each with what it breaks.
HAZARDS = {
    re.compile(r"JAX_EXPLAIN_CACHE_MISSES|jax_explain_cache_misses"):
        "JAX 0.11.2's cache-miss explainer crashes on Pallas-Triton lowerings that pass debug_info as "
        "a keyword (partial_eval.explain -> diff_tracing_cache_keys): c58's and c57's runs died "
        "before their first compile",
}


def hazards(texts: dict[str, str], command: str) -> list[str]:
    """A failure for each known-bad setting a staged script or the command makes."""
    return [f"{name}:{number} sets {match.group(0)}: {why}"
            for name, text in [("<command>", command), *texts.items()]
            for number, line in enumerate(text.splitlines(), 1)
            for pattern, why in HAZARDS.items() if (match := pattern.search(line))]


def lab_verified(tree: Path, hardware: str, failures: list[str], explicit: Sequence[str] = ()) -> dict:
    """Where the ref's measured kernel choices on `hardware` name Pallas kernels, or a script asks for
    one by name (`explicit`), the kernel lab's record that it compiled and ran them on that GPU,
    kernels, their callers and the pinned JAX byte for byte: CPU interpretation cannot catch a
    Triton lowering failure (c58, c57). None needed, none read."""
    if not (tree / "src/dew/nn/kernels/generation.py").exists():
        named = [f"explicit at {site}" for site in explicit]
        return lab_record(tree, hardware, GENERATIONS[hardware], named, failures)
    choices = subprocess.run(
        [sys.executable, "-c", "import json; from dew.nn.kernels.generation import KERNELS; "
         "print(json.dumps(KERNELS))"],
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(tree / "src")}, cwd=tree,
        timeout=300)
    if choices.returncode:
        failures.append(f"cannot read the ref's KERNELS: {choices.stderr[-300:]}")
        return {}
    generation = GENERATIONS[hardware]
    named = sorted(f"{choice}={row[generation]}" for choice, row in json.loads(choices.stdout).items()
                   if row.get(generation) in PALLAS_CHOICES)
    named += [f"explicit at {site}" for site in explicit]
    return lab_record(tree, hardware, generation, named, failures)


def lab_record(tree: Path, hardware: str, generation: str, named: list[str], failures: list[str]) -> dict:
    """The kernel lab's record for `tree`'s kernels on `hardware`, where `named` Pallas kernels run."""
    found: dict = {"generation": generation, "pallas": named}
    if not named:
        return found
    found["digest"] = digest = kernel_digest(tree)
    key = f"lab/verified/{LAB_GPUS[hardware]}/{digest}.json"
    record, missing = store_read(key)
    if record is None or not record.get("ok"):
        failures.append(f"KERNELS on {generation} names Pallas kernels ({', '.join(named)}) whose code "
                        f"the kernel lab has not verified on {LAB_GPUS[hardware]} ({key}: "
                        f"{missing or 'not ok'})")
        return found
    found["record"] = {k: record.get(k) for k in ("ref", "gpu", "versions", "at")}
    return found


def store_read(key: str) -> tuple[dict | None, str]:
    """A JSON object from Dew's R2 store (dew-data) with the read token in DEW_DATA_READ_TOKEN or
    ~/.config/dew/dew-data.json's read_token_file, and why there is none where there is none."""
    config = Path.home() / ".config/dew/dew-data.json"
    url = "https://dew-data.ashishkmr472.workers.dev"
    token = os.environ.get("DEW_DATA_READ_TOKEN")
    if config.exists():
        settings = json.loads(config.read_text())
        url = settings.get("url", url)
        if token is None and settings.get("read_token_file"):
            token = Path(settings["read_token_file"]).expanduser().read_text().strip()
    if not token:
        return None, "no read token for the store (DEW_DATA_READ_TOKEN)"
    request = urllib.request.Request(f"{url.rstrip('/')}/{key}", headers={
        "Authorization": f"Bearer {token}", "User-Agent": "dew-preflight"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            value = json.loads(response.read())
    except (urllib.error.URLError, json.JSONDecodeError) as error:
        return None, f"{type(error).__name__}: {error}"
    return (value, "") if isinstance(value, dict) else (None, "the record is not a JSON object")


def label(chain: dict) -> str:
    return f"{chain['where']}: {'.'.join([chain['module'], *chain['attrs']])}"


def verify(path: Path, request: dict, inputs: list[tuple[str, str]]) -> str:
    """The receipt's digest, if it is a passing preflight, by this preflight, of exactly this request:
    its commit, hardware, verbatim command, runtime pins and every script it stages."""
    data = path.read_bytes()
    receipt = json.loads(data)
    problems = []
    if not receipt.get("ok"):
        problems.append("it failed: " + "; ".join(receipt.get("failures", ["no result"]))[:2000])
    expected = {key: request[key] for key in ("commit", "hardware", "command")}
    fetched = set(request.get("refs", ()))
    problems.extend(f"its scripts check out {ref[:12]}, which the entry's refs: does not fetch"
                    for ref in receipt.get("refs", []) if ref not in fetched)
    expected.update(
        tool=TOOL, runtime=sorted(pin.lower() for pin in request["runtime"]), dew_python=VENV_PYTHON
    )
    expected["python"] = INTERPRETERS.get(receipt.get("interpreter", ""), "a known interpreter")
    for key, value in expected.items():
        if receipt.get(key) != value:
            problems.append(f"its {key} is {str(receipt.get(key))[:80]!r}, not {str(value)[:80]!r}")
    covered = set(receipt.get("inputs", {}).values())
    for target, digest in inputs:
        if target.endswith(SCRIPTS) and digest not in covered:
            problems.append(f"it did not check {target} at {digest[:12]}")
    if problems:
        raise ValueError(f"preflight receipt {path} does not clear this request: " + "; ".join(problems))
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prove a Colab request's scripts on CPU before a GPU is paid for."
    )
    parser.add_argument("inputs", nargs="*", help="the request's staged scripts")
    parser.add_argument("--ref", required=True)
    parser.add_argument("--hardware", choices=sorted(ARCH), required=True)
    parser.add_argument("--command", default="", help="the queue entry's command, verbatim")
    parser.add_argument("--pin", action="append", default=[], help="NAME==VERSION the runtime provides")
    parser.add_argument(
        "--accept", action="append", default=[], help="TEXT::REASON for a failure that stands"
    )
    parser.add_argument("--also", action="append", default=[],
                        help="another ref the scripts check out, which the entry's refs: names")
    parser.add_argument("--origin", default="origin", help="where to fetch a ref this clone lacks")
    parser.add_argument("--dew-python", default=sys.executable)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    for pin in args.pin:
        if not re.fullmatch(r"[\w.-]+==[\w.+!-]+", pin):
            parser.error(f"--pin needs NAME==VERSION, not {pin}")
    with tempfile.TemporaryDirectory(prefix="dew-preflight-") as work:
        try:
            receipt = preflight(args, Path(work))
        finally:
            subprocess.run(["git", "-C", str(REPO), "worktree", "prune"], capture_output=True, check=False)
    text = json.dumps(receipt, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    for failure in receipt["failures"]:
        print("PREFLIGHT FAIL " + failure, file=sys.stderr)
    print(f"PREFLIGHT {'ok' if receipt['ok'] else 'failed'} {receipt['commit'][:12]}", file=sys.stderr)
    raise SystemExit(0 if receipt["ok"] else 1)


if __name__ == "__main__":
    main()
