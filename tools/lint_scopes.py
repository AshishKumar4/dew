"""What the names of a Python file stand for, for tools/lint_slop.py: the
nodes of each lexical scope, a signature's parameters, and the imports each
scope reads through its own bindings.

Scopes are read flow-insensitively: a name a scope both imports and assigns
counts as assigned throughout it, as one the scope binds by a parameter and
then imports counts as imported.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping, Sequence

SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def own(scope: ast.AST) -> Iterator[ast.AST]:
    """Every node in one scope, a nested scope's own node but not its contents."""
    for node in ast.iter_child_nodes(scope):
        yield node
        if not isinstance(node, SCOPES):
            yield from own(node)


def arguments(function: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> list[ast.arg]:
    """Every parameter in a signature."""
    arguments = function.args
    return [argument for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,
                                      arguments.vararg, arguments.kwarg) if argument is not None]


def annotations(function: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterator[tuple[str, ast.expr]]:
    """Every annotation in a signature, paired with the name it belongs to."""
    for argument in arguments(function):
        if argument.annotation is not None:
            yield argument.arg, argument.annotation
    if function.returns is not None:
        yield "return", function.returns


def bound_by(node: ast.Import | ast.ImportFrom, parent: Sequence[str]) -> dict[str, str]:
    """What each name one import statement binds stands for, as a dotted path."""
    if isinstance(node, ast.Import):
        heads = {alias: alias.name.split(".")[0] for alias in node.names}
        return {alias.asname or head: alias.name if alias.asname else head for alias, head in heads.items()}
    base = node.module or ""
    if node.level:
        base = ".".join([*parent[:len(parent) - node.level + 1], *([base] if base else [])])
    return {alias.asname or alias.name: f"{base}.{alias.name}" for alias in node.names if alias.name != "*"}


def imports(tree: ast.Module, parent: Sequence[str]) -> dict[str, str]:
    """What each name a module imports stands for, as a dotted path: every
    import in it, those inside functions and type-checking blocks included.
    `parent` is the package its relative imports start from (`_parent`)."""
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            bound |= bound_by(node, parent)
    return bound


def outside(scope: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.ClassDef) -> list[ast.AST]:
    """The parts of a nested scope's statement its enclosing scope evaluates:
    decorators, defaults, annotations, a class's bases and keywords."""
    if isinstance(scope, ast.ClassDef):
        return [*scope.decorator_list, *scope.bases, *scope.keywords]
    defaults = [*scope.args.defaults, *(default for default in scope.args.kw_defaults if default is not None)]
    if isinstance(scope, ast.Lambda):
        return defaults
    return [*scope.decorator_list, *defaults, *(annotation for _, annotation in annotations(scope))]


def comprehended(nodes: Sequence[ast.AST], table: Mapping[str, str], body: Mapping[str, str]
                  ) -> dict[int, Mapping[str, str]]:
    """The table each node inside a comprehension reads: `body`, the table a
    nested function would read (in a class body, the class's surroundings'),
    less its targets, which bind there and in comprehensions inside it, but
    not in its first iterable, which reads the scope's own `table`. Keyed by
    node identity."""
    tables: dict[int, Mapping[str, str]] = {}
    for node in nodes:
        if isinstance(node, ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp):
            around = tables.get(id(node), body)
            names = {name.id for generator in node.generators for name in ast.walk(generator.target)
                     if isinstance(name, ast.Name)}
            inside = {name: target for name, target in around.items() if name not in names}
            first = {id(child) for child in ast.walk(node.generators[0].iter)}
            for child in ast.walk(node):
                if child is not node and id(child) not in first:
                    tables[id(child)] = inside
    return tables


def scoped(body: Sequence[ast.AST], visible: Mapping[str, str], parent: Sequence[str],
           parameters: frozenset[str] = frozenset(), class_body: bool = False
           ) -> Iterator[tuple[ast.AST, Mapping[str, str]]]:
    """Every node of a scope's `body` and of the scopes in it, with what each
    name imported there stands for (`visible` around it). A scope's own
    import replaces its `parameters`; a name it otherwise binds (an
    assignment, a definition) stands for no import. A comprehension's
    targets are its own, and a class body's names reach no scope inside it."""
    nodes = [node for statement in body
             for node in (statement, *(() if isinstance(statement, SCOPES) else own(statement)))]
    targets = {id(name) for node in nodes if isinstance(node, ast.comprehension)
               for name in ast.walk(node.target)}
    local = {node.id for node in nodes
             if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and id(node) not in targets}
    local |= {node.name for node in nodes
              if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)}
    # An exception's name and a pattern's capture bind without a stored Name.
    local |= {node.name for node in nodes if isinstance(node, ast.ExceptHandler | ast.MatchAs | ast.MatchStar)
              and node.name}
    local |= {node.rest for node in nodes if isinstance(node, ast.MatchMapping) and node.rest}
    imported: dict[str, str] = {}
    for node in nodes:
        if isinstance(node, ast.Import | ast.ImportFrom):
            imported |= bound_by(node, parent)
    around = {name: target for name, target in visible.items() if name not in parameters}
    table = {name: target for name, target in {**around, **imported}.items() if name not in local}
    inside = comprehended(nodes, table, visible if class_body else table)
    for node in nodes:
        here = inside.get(id(node), table)
        yield node, here
        if isinstance(node, SCOPES):
            yield from scoped(outside(node), here, parent)
            inner = node.body if isinstance(node.body, list) else [node.body]
            bound = frozenset() if isinstance(node, ast.ClassDef) else frozenset(
                argument.arg for argument in arguments(node))
            yield from scoped(inner, visible if class_body else here, parent, bound,
                              isinstance(node, ast.ClassDef))


def resolved(node: ast.expr, table: Mapping[str, str]) -> str:
    """The dotted path an attribute chain from a name spells, its name read as
    the import `table` gives it (`scoped`), or "" for anything else."""
    attributes = []
    while isinstance(node, ast.Attribute):
        attributes.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or node.id not in table:
        return ""
    return ".".join([table[node.id], *reversed(attributes)])
