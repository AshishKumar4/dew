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
from pathlib import Path

from tools.colab.common import REPO, git

ARCH = {"gpu:T4": "7.5", "gpu:L4": "8.9", "gpu:A100": "8.0", "gpu:G4": "12.0"}
# Colab's Python; the pinned packages are installed for it.
PYTHON = "3.13"
LOCAL = ("dew", "tools")
SCRIPTS = (".py", ".sh", ".bash")
VALUED = {"-c", "--constraint", "-r", "--requirement", "-i", "--index-url", "--extra-index-url", "-t"}
VALUED |= {"--target", "-f", "--find-links", "--python", "--prefix", "--root", "--torch-backend"}
UNKNOWN = object()


def facts() -> dict:
    keys = ("checkouts", "modules", "runs", "sources", "failures")
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
    if head.startswith("pip") or (head == "uv" and rest[:2] == ["pip", "install"]):
        pins(rest[1:] if head == "uv" else rest, found)
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
                    pins(list(options), found)
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


def pins(words: list[str], found: dict) -> None:
    if words[:1] != ["install"]:
        return
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


def uses(source: str, label: str) -> tuple[list[dict], set[tuple[str, str]]]:
    """Every module attribute chain a Python source reads, and the attributes it assigns on modules."""
    tree = ast.parse(source, label)
    chains: list[dict] = []
    bound: set[tuple[str, str]] = set()

    def scope(node: ast.AST, inherited: dict[str, tuple[str, tuple[str, ...]]]) -> None:
        aliases = dict(inherited)
        own, nested = [], []
        pending = list(ast.iter_child_nodes(node))
        while pending:
            child = pending.pop()
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                nested.append(child)
                pending.extend(child.decorator_list)
                continue
            own.append(child)
            pending.extend(ast.iter_child_nodes(child))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
                aliases.pop(argument.arg, None)
        imported = set()
        for child in own:
            where = f"{label}:{getattr(child, 'lineno', 0)}"
            if isinstance(child, ast.Import):
                for alias in child.names:
                    chains.append({"module": alias.name, "attrs": [], "where": where})
                    bind = alias.asname or alias.name.split(".")[0]
                    aliases[bind] = (alias.name if alias.asname else bind, ())
                    imported.add(bind)
            elif isinstance(child, ast.ImportFrom) and not child.level and child.module:
                for alias in child.names:
                    attrs = [] if alias.name == "*" else [alias.name]
                    chains.append({"module": child.module, "attrs": attrs, "where": where})
                    if attrs:
                        aliases[alias.asname or alias.name] = (child.module, (alias.name,))
                        imported.add(alias.asname or alias.name)
            elif isinstance(child, ast.Call) and dotted(child.func).endswith("import_module"):
                named = child.args[0] if child.args else None
                if isinstance(named, ast.Constant) and isinstance(named.value, str):
                    chains.append({"module": named.value, "attrs": [], "where": where})
        for child in own:
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store) and child.id not in imported:
                aliases.pop(child.id, None)
        for child in own:
            if not isinstance(child, ast.Attribute):
                continue
            path = dotted(child)
            root, *attrs = path.split(".") if path else [""]
            if root not in aliases:
                continue
            module, prefix = aliases[root]
            attrs = [*prefix, *attrs]
            if isinstance(child.ctx, ast.Load):
                chains.append({"module": module, "attrs": attrs, "where": f"{label}:{child.lineno}"})
            else:
                bound.add((".".join([module, *attrs[:-1]]), attrs[-1]))
                chains.append({"module": module, "attrs": attrs[:-1], "where": f"{label}:{child.lineno}"})
        for child in nested:
            scope(child, aliases)

    scope(tree, {})
    return chains, bound


def dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return ".".join([node.id, *reversed(parts)]) if isinstance(node, ast.Name) else ""


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
        result = subprocess.run(
            [python, "-c", RESOLVE],
            input=json.dumps([[chains[index] for index in members], sorted(bound)]),
            capture_output=True,
            text=True,
            env=env,
            cwd=cwd,
            timeout=1200,
        )
        try:
            answers = json.loads(result.stdout.splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            lines = result.stderr.strip().splitlines() or ["no output"]
            last = next((line for line in lines if line.startswith("Fatal Python error")), lines[-1])
            answers = [["import", f"importing {name} ended the interpreter ({result.returncode}): {last}"]]
            answers *= len(members)
        for index, answer in zip(members, answers, strict=True):
            found[index] = answer
    return found


class Sources:
    """Static name resolution over installed sources, taking the target GPU's branches."""

    def __init__(self, roots: list[Path], arch: str, scope: set[str]):
        self.roots, self.arch, self.scope = roots, arch, scope
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

    def lookup(self, module: str, name: str, seen: frozenset = frozenset()):
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
                if guarded or not isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                where = f"{module}:{node.lineno}"
                if isinstance(node, ast.Import):
                    targets = [(alias.name, []) for alias in node.names]
                else:
                    names = [alias.name for alias in node.names if alias.name != "*"]
                    targets = [(absolute(node, package), names)]
                for target, names in targets:
                    if target.split(".")[0] not in self.scope:
                        continue
                    if not self.path(target):
                        failures.append(f"{where} imports {target}, which is not installed")
                        continue
                    pending.append(target)
                    for name in names:
                        binding = self.lookup(target, name)
                        if binding is None and (target, name) not in bound:
                            failures.append(f"{where}: from {target} import {name}: unbound on sm{self.arch}")
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
                    constants[target.id] = evaluate(node.value, constants, self.arch)
                if isinstance(node, ast.If):
                    taken = evaluate(node.test, constants, self.arch)
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


def evaluate(node: ast.AST, constants: dict, arch: str):
    """A condition's value on the target GPU, or UNKNOWN; only device queries and literals are known."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id, UNKNOWN)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        items = [evaluate(item, constants, arch) for item in node.elts]
        return UNKNOWN if any(item is UNKNOWN for item in items) else tuple(items)
    if isinstance(node, ast.Call):
        name = dotted(node.func)
        if name.endswith("get_target_compute_version"):
            return arch
        if name.endswith("get_device_capability"):
            return tuple(int(part) for part in arch.split("."))
        if name.endswith("cuda.is_available"):
            return True
        return UNKNOWN
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        value = evaluate(node.value, constants, arch)
        try:
            return UNKNOWN if value is UNKNOWN else value[node.slice.value]
        except (IndexError, KeyError, TypeError):
            return UNKNOWN
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = evaluate(node.operand, constants, arch)
        return UNKNOWN if value is UNKNOWN else not value
    if isinstance(node, ast.BoolOp):
        values = [evaluate(value, constants, arch) for value in node.values]
        decisive = isinstance(node.op, ast.Or)
        if any(value is not UNKNOWN and bool(value) == decisive for value in values):
            return decisive
        return UNKNOWN if any(value is UNKNOWN for value in values) else not decisive
    if isinstance(node, ast.Compare):
        left = evaluate(node.left, constants, arch)
        for operator, comparator in zip(node.ops, node.comparators, strict=True):
            right = evaluate(comparator, constants, arch)
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
    receipt.update(python=args.python, command=args.command, inputs={})
    texts: dict[str, str] = {}
    for item in map(Path, args.inputs):
        data = item.read_bytes()
        receipt["inputs"][item.name] = hashlib.sha256(data).hexdigest()
        if item.name.endswith(SCRIPTS):
            texts[item.name] = data.decode()
    found = shell(args.command, texts)
    failures = list(found["failures"])
    for ref in found["checkouts"]:
        try:
            target = git(REPO, "rev-parse", f"{ref}^{{commit}}")
        except RuntimeError:
            target = ref
        if target != commit:
            failures.append(f"a script checks out {ref}, not the preflighted {commit}")
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
    for name, text in sources:
        try:
            more, assigned = uses(text, name)
        except SyntaxError as error:
            failures.append(f"{name} does not parse: {error}")
            continue
        chains += more
        bound |= assigned
    unique = {(chain["module"], tuple(chain["attrs"])): chain for chain in chains}
    chains = [chain for chain in unique.values() if top(chain) not in sys.stdlib_module_names]
    # A script imports its sibling inputs by name, from the directory it runs in.
    siblings = work / "inputs"
    siblings.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        if name.endswith(".py"):
            (siblings / name).write_text(text)
    ours = {*LOCAL, *(Path(name).stem for name in texts if name.endswith(".py"))}
    local = [chain for chain in chains if top(chain) in ours]
    other = [chain for chain in chains if top(chain) not in ours]
    path = [str(siblings), str(tree / "src"), str(tree)]
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(path)}
    found_local = resolve(args.dew_python, local, bound, environment, tree)
    for chain, (kind, detail) in zip(local, found_local, strict=True):
        if kind != "ok":
            failures.append(f"{label(chain)}: {detail}")
        elif top(chain) == "dew" and not str(detail).startswith(str(tree)):
            failures.append(f"dew resolved from {detail}, not from {tree}")
    pinned = {**found["pins"], **dict(pin.lower().split("==", 1) for pin in args.pin)}
    interpreter, roots, versions, problems = install(pinned, work, args.python)
    failures += problems
    receipt.update(pins=pinned, installed=versions)
    scope = top_levels(roots, set(pinned))
    static = Sources(roots, arch, scope)
    results = resolve(interpreter, other, bound) if interpreter else [["absent", "no pins"]] * len(other)
    absent = [chain for chain, result in zip(other, results, strict=True) if result[0] == "absent"]
    retried = iter(resolve(args.dew_python, absent, bound))
    for chain, result in zip(other, results, strict=True):
        kind, detail = next(retried) if result[0] == "absent" else result
        if kind == "import" and top(chain) in scope:
            problem = static.check(chain["module"], chain["attrs"], bound)
            if problem:
                failures.append(f"{label(chain)}: {problem} (from sources; on CPU: {detail})")
        elif kind == "absent":
            failures.append(f"{label(chain)}: no pin or Dew dependency provides it ({detail})")
        elif kind != "ok":
            failures.append(f"{label(chain)}: {detail}")
    failures += static.closure({chain["module"] for chain in other if top(chain) in scope}, bound)
    receipt.update(checked=len(chains), failures=failures, ok=not failures)
    return receipt


def top(chain: dict) -> str:
    return chain["module"].split(".")[0]


def label(chain: dict) -> str:
    return f"{chain['where']}: {'.'.join([chain['module'], *chain['attrs']])}"


def verify(path: Path, commit: str, hardware: str, command: str, inputs: list[tuple[str, str]]) -> str:
    """The receipt's digest, if it is a passing preflight of exactly this request."""
    data = path.read_bytes()
    receipt = json.loads(data)
    problems = []
    if not receipt.get("ok"):
        problems.append("it failed: " + "; ".join(receipt.get("failures", ["no result"]))[:2000])
    if receipt.get("commit") != commit:
        problems.append(f"it checked {receipt.get('commit')}, not {commit}")
    if receipt.get("hardware") != hardware:
        problems.append(f"it checked {receipt.get('hardware')}, not {hardware}")
    if receipt.get("command") != command:
        problems.append("it checked a different command")
    covered = set(receipt.get("inputs", {}).values())
    for target, digest in inputs:
        if target.endswith(SCRIPTS) and digest not in covered:
            problems.append(f"it did not check {target} at {digest[:12]}")
    if problems:
        raise ValueError(f"preflight receipt {path} does not clear this request: " + "; ".join(problems))
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("inputs", nargs="*", help="the request's staged scripts")
    parser.add_argument("--ref", required=True)
    parser.add_argument("--hardware", choices=sorted(ARCH), required=True)
    parser.add_argument("--command", default="", help="the queue entry's command, verbatim")
    parser.add_argument("--pin", action="append", default=[], help="NAME==VERSION the runtime provides")
    parser.add_argument("--origin", default="origin", help="where to fetch a ref this clone lacks")
    parser.add_argument("--python", default=PYTHON)
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
