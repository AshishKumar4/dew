"""Every program a test hands a child interpreter compiles, and the Dew names it imports exist.

A test that runs `sys.executable -c SOURCE` only proves SOURCE when it runs, and a GPU-only test never
runs on CPU. A rename once rewrote one such child's import into a syntax error
(`from dew.interop import Pretrained.load`) that waited for a paid GPU lane to find it. This reads each
test file's `-c` sources (literals, and names bound to literal concatenations), compiles them and
resolves their `dew` imports here, on CPU.
"""

import ast
import importlib
import textwrap
from pathlib import Path

from tools.colab.preflight import dotted, uses

TESTS = Path(__file__).parent


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


def programs() -> dict[str, str]:
    """`{file:line: source}` for each `sys.executable, "-c", SOURCE` whose SOURCE is literal."""
    found = {}
    for path in sorted(TESTS.glob("*.py")):
        tree = ast.parse(path.read_text(), str(path))
        module: dict[str, str] = {}
        functions = (
            node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        scopes = [tree, *functions]
        for scope in scopes:
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
                for before, flag, source in zip(items, items[1:], items[2:], strict=False):
                    if (
                        dotted(before) == "sys.executable"
                        and isinstance(flag, ast.Constant)
                        and flag.value == "-c"
                    ):
                        text = literal(source, strings)
                        if text is not None:
                            found[f"{path.name}:{source.lineno}"] = text
    return found


def test_child_programs_are_found():
    """The reader sees the programs it guards, among them the one whose import broke."""
    found = programs()
    assert len(found) > 20
    assert any("from dew.interop import Pretrained" in text for text in found.values())


def test_every_child_program_compiles_and_its_dew_imports_exist():
    problems = []
    for where, text in programs().items():
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
                    if not hasattr(value, attr):
                        value = importlib.import_module(f"{name}.{attr}")
                    else:
                        value = getattr(value, attr)
                    name = f"{name}.{attr}"
            except ImportError as error:
                problems.append(f"{where}: {error}")
    assert not problems, "\n".join(problems)
