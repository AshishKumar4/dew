"""Every program the repository hands a child interpreter compiles, and the Dew names it imports exist.

A test or tool that runs `python -c SOURCE` only proves SOURCE when it runs, and a GPU-only test
or a GPU session's tool never runs on CPU. A rename once rewrote one such child's import into a
syntax error (`from dew.interop import Pretrained.load`) that waited for a paid GPU lane to find
it. This reads the `-c` programs of every tracked Python file under tests, tools, src, site/scripts,
recipes and examples (literals, names bound to literal concatenations, dedents), compiles them and
resolves their `dew` imports here, on CPU. A program built at run time cannot be read; its site is
named as unchecked in a failure's message.
"""

import ast
import importlib
import subprocess
import textwrap
from pathlib import Path

from tools.python_uses import dotted, uses

ROOT = Path(__file__).resolve().parents[1]
SCOPE = ("tests", "tools", "src", "site/scripts", "recipes", "examples")


def literal(node: ast.AST, strings: dict[str, str]) -> str | None:
    """A string expression's value where it is made of literals, else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return strings.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = literal(node.left, strings), literal(node.right, strings)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.Call) and dotted(node.func) in ("textwrap.dedent", "dedent") and node.args:
        text = literal(node.args[0], strings)
        return None if text is None else textwrap.dedent(text)
    return None


def interpreter(node: ast.AST) -> bool:
    """Whether an argument names a Python interpreter: sys.executable, a name or attribute holding
    `python` or `interpreter`, or str() of one."""
    if isinstance(node, ast.Call) and dotted(node.func) == "str" and node.args:
        return interpreter(node.args[0])
    name = dotted(node).lower()
    return name == "sys.executable" or any(
        word in name.rsplit(".", 1)[-1] for word in ("python", "interpreter")
    )


def sites(tree: ast.AST):
    """`(source node, scope's literal strings)` for each `-c` an interpreter takes: after one in a
    list, tuple or call, or first in a list a program constant follows (`[*python, "-c", PROGRAM]`)."""
    module: dict[str, str] = {}
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for scope in [tree, *functions]:
        strings = dict(module)
        for node in ast.walk(scope):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                text = literal(node.value, strings)
                if text is not None:
                    strings[node.targets[0].id] = text
        if scope is tree:
            module = strings
        for node in ast.walk(scope):
            items = node.args if isinstance(node, ast.Call) else getattr(node, "elts", None)
            if not isinstance(node, (ast.Call, ast.List, ast.Tuple)) or not items:
                continue
            for index, flag in enumerate(items[:-1]):
                if not (isinstance(flag, ast.Constant) and flag.value == "-c"):
                    continue
                source = items[index + 1]
                if (index > 0 and interpreter(items[index - 1])) or (
                    index == 0 and isinstance(source, ast.Name)
                ):
                    yield source, strings


def programs() -> tuple[dict[str, str], list[str]]:
    """`{file:line: source}` for each child program read, and the sites whose source is not literal."""
    tracked = subprocess.run(
        ["git", "ls-files", "--", *SCOPE], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    found, unread = {}, []
    for name in sorted(item for item in tracked if item.endswith(".py")):
        tree = ast.parse((ROOT / name).read_text(), name)
        for source, strings in sites(tree):
            text = literal(source, strings)
            if text is None:
                unread.append(f"{name}:{source.lineno}")
            else:
                found[f"{name}:{source.lineno}"] = text
    return found, unread


def test_child_programs_are_found():
    """The reader sees the programs it guards: the test whose import broke, a tool's, src's own."""
    found, _ = programs()
    assert any("from dew.interop import Pretrained" in text for text in found.values())
    assert any(where.startswith("tools/") for where in found)
    assert any(where.startswith("src/dew/data/sources/tfds.py") for where in found)


def test_every_child_program_compiles_and_its_dew_imports_exist():
    found, unread = programs()
    problems = []
    for where, text in found.items():
        try:
            compile(text, where, "exec")
        except SyntaxError as error:
            problems.append(f"{where}: {error.msg} in {error.text!r}")
            continue
        chains, *_ = uses(text, where)
        for chain in chains:
            if chain["module"].split(".")[0] != "dew":
                continue
            name = chain["module"]
            try:
                value = importlib.import_module(name)
                for attr in chain["attrs"]:
                    value = (
                        getattr(value, attr)
                        if hasattr(value, attr)
                        else importlib.import_module(f"{name}.{attr}")
                    )
                    name = f"{name}.{attr}"
            except ImportError as error:
                problems.append(f"{where}: {error}")
    unchecked = f"\nunchecked (built at run time): {', '.join(unread) or 'none'}"
    assert not problems, "\n".join(problems) + unchecked
