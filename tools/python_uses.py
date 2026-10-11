"""Which modules and attributes a Python source reads, by its syntax alone.

`uses` walks a source's scopes, following aliases (`import a.b as c`, `from a import b`) into the
attribute chains each name reads; it also names the attributes the source assigns on modules,
the directories it puts on sys.path, and the imports no parse can name. The Colab preflight
resolves these chains against a pinned environment; the child-program test resolves the ones a
test hands `python -c`.
"""

from __future__ import annotations

import ast
import os


def uses(source: str, label: str) -> tuple[list[dict], set[tuple[str, str]], set[str], list[str]]:
    """Every module attribute chain a Python source reads, the attributes it assigns on modules, the
    directories, relative to the Dew tree it runs in, it puts on sys.path, and the imports no parse
    can name (a computed module name, a star import)."""
    found = Uses(ast.parse(source, label), label)
    return found.chains, found.bound, found.paths, found.unchecked


class Uses:
    def __init__(self, tree: ast.Module, label: str):
        self.label = label
        self.chains: list[dict] = []
        self.bound: set[tuple[str, str]] = set()
        self.paths: set[str] = set()
        self.unchecked: list[str] = []
        self.constants: dict[str, str] = {}
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                value = directory(node.value, self.constants)
                if value is not None:
                    self.constants[node.targets[0].id] = value
        self.scope(tree, {})

    def chain(self, module: str, attrs: list[str], node: ast.AST) -> None:
        where = f"{self.label}:{getattr(node, 'lineno', 0)}"
        self.chains.append({"module": module, "attrs": list(attrs), "where": where})

    def scope(self, node: ast.AST, inherited: dict[str, tuple[str, tuple[str, ...]]]) -> None:
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
            imported |= self.record(child, aliases)
        for child in own:
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store) and child.id not in imported:
                aliases.pop(child.id, None)
        for child in own:
            if isinstance(child, ast.Attribute):
                self.attribute(child, aliases)
        for child in nested:
            self.scope(child, aliases)

    def record(self, node: ast.AST, aliases: dict) -> set[str]:
        """An import, sys.path addition or import_module call; the names an import binds."""
        names = set()
        if isinstance(node, ast.Import):
            for alias in node.names:
                self.chain(alias.name, [], node)
                bind = alias.asname or alias.name.split(".")[0]
                aliases[bind] = (alias.name if alias.asname else bind, ())
                names.add(bind)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            for alias in node.names:
                attrs = [] if alias.name == "*" else [alias.name]
                self.chain(node.module, attrs, node)
                if attrs:
                    aliases[alias.asname or alias.name] = (node.module, (alias.name,))
                    names.add(alias.asname or alias.name)
                else:
                    self.unchecked.append(f"{self.label}:{node.lineno}: from {node.module} import *")
        elif isinstance(node, ast.Call) and node.args:
            name, argument = dotted(node.func), node.args[-1]
            if name in ("sys.path.insert", "sys.path.append"):
                value = directory(argument, self.constants)
                if value is not None:
                    self.paths.add(value)
            elif name.endswith("import_module") or name == "__import__":
                if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    self.chain(node.args[0].value, [], node)
                else:
                    self.unchecked.append(f"{self.label}:{node.lineno}: {name} of a computed name")
        return names

    def attribute(self, node: ast.Attribute, aliases: dict) -> None:
        root, *attrs = (dotted(node) or ".").split(".")
        if root not in aliases:
            return
        module, prefix = aliases[root]
        attrs = [*prefix, *attrs]
        if isinstance(node.ctx, ast.Load):
            self.chain(module, attrs, node)
        else:
            self.bound.add((".".join([module, *attrs[:-1]]), attrs[-1]))
            self.chain(module, attrs[:-1], node)


def directory(node: ast.AST, constants: dict[str, str]) -> str | None:
    """A path expression's value relative to the working directory, where it is a literal one."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return os.path.normpath(node.value)
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left, right = directory(node.left, constants), directory(node.right, constants)
        return None if left is None or right is None else os.path.normpath(os.path.join(left, right))
    if isinstance(node, ast.Call):
        name = dotted(node.func)
        if name in ("Path.cwd", "pathlib.Path.cwd", "os.getcwd") or (name.endswith("Path") and not node.args):
            return "."
        if name in ("Path", "pathlib.Path", "str", "os.fspath", "os.path.join") and node.args:
            parts = [directory(argument, constants) for argument in node.args]
            known = [part for part in parts if part is not None]
            return os.path.normpath(os.path.join(*known)) if len(known) == len(parts) else None
    return None


def dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return ".".join([node.id, *reversed(parts)]) if isinstance(node, ast.Name) else ""
