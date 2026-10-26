"""SLOP022's reading of a file, for tools/lint_slop.py: a torch module promoted to float64 after its
construction (ledger 1619). A float64 truth built in float32 and cast afterwards keeps what its
constructor computed rounded to float32 (a rotary `inv_freq`, Gemma's `embed_scale`; ledger 1601), so a
truth's twin is built inside tools/torch_precision.py's `float64`, by its `float64_twin`.

A promotion is `.double()`, `.to(...)` or `.type(...)` at float64 (`torch.float64`, `torch.double`, or a
name bound only to one) on a module, or the unbound `torch.nn.Module.double(module)` and its `to` and
`type`, or a call of a function of the file that promotes its parameter to another
(`walk(model, ..., dtype)`), given a module there and float64 as the dtype. A module is what the file
proves one, flow by flow in each scope: a constructor's result (a class the file imports, `torch.nn`'s
and transformers' among them, a capitalized attribute of anything, or a class the file defines on such
a base), a library's reader (`from_pretrained`, `from_config`, `from_single_file`), a deep copy of a
module, a module-preserving call on one (`eval()`, `cpu()`, `to(device)`), a function of the file that
returns one, and a name or attribute chain bound to any of these until it is bound to something else.
A forward's result, an attribute of a module (a tensor or a submodule) and an item of one are not, nor
is a module built inside `float64` or `widened` (or a context a condition picks from them, whose branch
the promotion's own condition is read to share), or at float64 by its `dtype`. Blind spots: a module
the file receives (a parameter, a factory it is passed, an attribute another method set), a loop's
target, and a submodule promoted by its attribute.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Sequence

import lint_scopes as scopes

FLOAT64 = frozenset({"torch.float64", "torch.double", "torch.DoubleTensor"})
"""The dtypes a promotion casts to, with `type`'s tensor class for float64, by name or spelled."""

PROMOTIONS = frozenset({"double", "to", "type"})
"""The methods that cast a module: `double()` always, `to` and `type` at a FLOAT64 dtype."""

PRESERVING = frozenset({"eval", "train", "cpu", "cuda", "to", "type", "float", "half", "bfloat16",
                        "double", "requires_grad_", "apply", "share_memory"})
"""The methods whose result is the module they are called on."""

READERS = frozenset({"from_pretrained", "from_config", "from_single_file"})
"""A library's public readers, each of which returns a model built from its arguments."""

TENSORS = frozenset({"Parameter", "Buffer", "Size"})
"""Classes whose instances are tensors or shapes, not modules: with every name ending in `Tensor`."""

WIDE = ("torch_precision.float64", "torch_precision.widened")
"""The contexts a module built in computes its constructor's values in float64."""


class _File:
    """What SLOP022 reads of one file: each node's import table, the classes it defines on an imported
    base (or on another such), the functions of it that return a module and those that promote a
    parameter (`_promoter`), and the nodes built inside a float64 context."""

    def __init__(self, tree: ast.Module, parent: Sequence[str]):
        self.tables = {id(node): table for node, table in scopes.scoped(tree.body, {}, list(parent))}
        self.wide = {id(node) for block in ast.walk(tree) if isinstance(block, ast.With | ast.AsyncWith)
                     and any(self._wide(item.context_expr) for item in block.items)
                     for statement in block.body for node in ast.walk(statement)}
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        self.classes: set[str] = set()
        functions = {node.name: node for node in ast.walk(tree)
                     if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)}
        self.factories: set[str] = set()
        self.promoters = {name: found for name, function in functions.items()
                          if (found := _promoter(function)) is not None}
        grown = True
        while grown:
            before = (len(self.classes), len(self.factories))
            self.classes |= {node.name for node in classes if any(
                self._constructs(base) or (isinstance(base, ast.Name) and base.id in self.classes)
                for base in node.bases)}
            self.factories |= {name for name, function in functions.items() if any(
                self.module(returned.value, _modules(self, function.body, frozenset()))
                for returned in scopes.own(function) if isinstance(returned, ast.Return) and returned.value)}
            grown = (len(self.classes), len(self.factories)) != before

    def resolved(self, node: ast.expr) -> str:
        return scopes.resolved(node, self.tables.get(id(node), {}))

    def _wide(self, node: ast.expr) -> bool:
        """Whether a `with` item is a float64 context (WIDE), or one where a condition picks it, as
        `float64() if wide else nullcontext()` does for the arm the condition widens."""
        if isinstance(node, ast.IfExp):
            return self._wide(node.body) or self._wide(node.orelse)
        called = self.resolved(node.func) if isinstance(node, ast.Call) else ""
        return any(called == path or called.endswith("." + path) for path in WIDE)

    def _constructs(self, node: ast.expr) -> bool:
        """Whether `node` names a class: an imported name, or an attribute of anything
        (`model_module.Transformer`), whose last name is capitalized, and not a tensor's."""
        spelled = node.attr if isinstance(node, ast.Attribute) else ""
        name = self.resolved(node).rpartition(".")[2] or spelled
        return bool(name) and name[0].isupper() and name not in TENSORS and not name.endswith("Tensor")

    def float64(self, node: ast.expr, aliases: frozenset[str]) -> bool:
        spelled = isinstance(node, ast.Constant) and node.value in FLOAT64
        aliased = isinstance(node, ast.Name) and node.id in aliases
        return spelled or aliased or self.resolved(node) in FLOAT64

    def module(self, node: ast.expr | None, held: frozenset[str]) -> bool:
        """Whether `node` evaluates to a module the file proves one (`held`: the names and attribute
        chains bound to one where `node` is)."""
        if node is None:
            return False
        if isinstance(node, ast.IfExp):
            return self.module(node.body, held) or self.module(node.orelse, held)
        if isinstance(node, ast.NamedExpr):
            return self.module(node.value, held)
        key = _key(node)
        if key is not None:
            return key in held
        if not isinstance(node, ast.Call):
            return False
        function = node.func
        if self.resolved(function) == "copy.deepcopy":
            return bool(node.args) and self.module(node.args[0], held)
        method = function.attr if isinstance(function, ast.Attribute) else None
        if method in PRESERVING and self.module(function.value, held):
            return True
        built = ((isinstance(function, ast.Name) and function.id in self.classes | self.factories)
                 or self._constructs(function) or method in READERS)
        # A module built in float64, inside its context or at its dtype, computed its constructor's values so.
        return built and id(node) not in self.wide and not _at_float64(self, node)


def _key(node: ast.expr) -> str | None:
    """The name or attribute chain of names `node` spells (`self.model`), or None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return ".".join([node.id, *reversed(parts)]) if isinstance(node, ast.Name) else None


Promoter = tuple[int, str, int | None, str | None]
"""A function's promoted parameter, by position and name, and the parameter it is promoted to (None
for `double()`)."""


def _promoter(function: ast.FunctionDef | ast.AsyncFunctionDef) -> Promoter | None:
    """The first parameter a function promotes (`Promoter`): by `double()`, or by `to` or `type` to
    another of its parameters, positionally or as `dtype`."""
    names = [argument.arg for argument in (*function.args.posonlyargs, *function.args.args)]
    named = {argument.arg for argument in scopes.arguments(function)}
    for node in scopes.own(function):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in PROMOTIONS and isinstance(node.func.value, ast.Name)
                and node.func.value.id in named):
            continue
        promoted = node.func.value.id
        place = names.index(promoted) if promoted in names else -1
        if node.func.attr == "double":
            return place, promoted, None, None
        for dtype in (*node.args, *(keyword.value for keyword in node.keywords if keyword.arg == "dtype")):
            if isinstance(dtype, ast.Name) and dtype.id in named:
                return place, promoted, names.index(dtype.id) if dtype.id in names else -1, dtype.id
    return None


def _argument(call: ast.Call, place: int | None, name: str | None) -> ast.expr | None:
    """The argument a call passes a parameter, by its position or its name."""
    if place is not None and 0 <= place < len(call.args):
        return call.args[place]
    return next((keyword.value for keyword in call.keywords if keyword.arg == name), None)


def _at_float64(file: _File, call: ast.Call) -> bool:
    """Whether a constructor or reader is given a float64 `dtype` (or transformers' `torch_dtype`)."""
    return any(keyword.arg in ("dtype", "torch_dtype") and file.float64(keyword.value, frozenset())
               for keyword in call.keywords)


def _bound(target: ast.expr, value: ast.expr | None) -> list[tuple[str, ast.expr | None]]:
    """Each name or attribute chain an assignment binds, with the value it binds it to where an
    unpacking pairs them, else None."""
    if isinstance(target, ast.Tuple | ast.List):
        values = value.elts if isinstance(value, ast.Tuple | ast.List) and len(value.elts) == len(
            target.elts) else [None] * len(target.elts)
        return [pair for part, part_value in zip(target.elts, values, strict=True)
                for pair in _bound(part, part_value)]
    if isinstance(target, ast.Starred):
        return _bound(target.value, None)
    key = _key(target)
    return [] if key is None else [(key, value)]


def _rebinds(node: ast.AST) -> list[str]:
    """The names a statement other than an assignment rebinds to what the file cannot prove a module:
    a loop's or a `with`'s target, an import."""
    if isinstance(node, ast.For | ast.AsyncFor):
        return [key for key, _ in _bound(node.target, None)]
    if isinstance(node, ast.withitem) and node.optional_vars is not None:
        return [key for key, _ in _bound(node.optional_vars, None)]
    if isinstance(node, ast.Import | ast.ImportFrom):
        return list(scopes.bound_by(node, ()))
    return []


def _events(body: Sequence[ast.stmt]) -> list[ast.AST]:
    """A scope's assignments, rebinding statements and calls, in the order their effect takes place: a
    binding once its statement's value has been read, at its end; a call where it starts."""
    events = []
    for node in scopes.nodes_of(body):
        if isinstance(node, ast.For | ast.AsyncFor | ast.Call):
            events.append(((node.lineno, node.col_offset), id(node), node))
        elif isinstance(node, ast.withitem):
            ended = node.context_expr
            events.append(((ended.end_lineno or ended.lineno, ended.end_col_offset or 0), id(node), node))
        elif isinstance(node, ast.Assign | ast.AnnAssign | ast.Import | ast.ImportFrom):
            events.append(((node.end_lineno or node.lineno, node.end_col_offset or 0), id(node), node))
    return [node for *_, node in sorted(events, key=lambda event: event[:2])]


def _aliases(file: _File, body: Sequence[ast.stmt], around: frozenset[str]) -> frozenset[str]:
    """The names a scope binds only to a FLOAT64 dtype, with those around it it does not bind."""
    values: dict[str, list[ast.expr | None]] = {}
    for node in scopes.nodes_of(body):
        if isinstance(node, ast.Assign | ast.AnnAssign):
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                for key, value in _bound(target, node.value):
                    values.setdefault(key, []).append(value)
    bound = set(scopes.binders(scopes.nodes_of(body)))
    mine = {name for name, assigned in values.items()
            if all(value is not None and file.float64(value, frozenset()) for value in assigned)}
    return frozenset(mine | (around - bound))


def _modules(file: _File, body: Sequence[ast.stmt], around: frozenset[str],
             promotions: list[ast.Call] | None = None, aliases: frozenset[str] = frozenset()
             ) -> frozenset[str]:
    """The names and attribute chains bound to a module at the end of a scope's `body` (`around`: those
    of the scopes around it the scope does not bind), each promotion of one appended to `promotions`."""
    bound = set(scopes.binders(scopes.nodes_of(body)))
    held = {key for key in around if key.split(".")[0] not in bound}
    for node in _events(body):
        if isinstance(node, ast.Call):
            if promotions is not None and _promotes(file, node, frozenset(held), aliases):
                promotions.append(node)
            continue
        if isinstance(node, ast.Assign | ast.AnnAssign):
            pairs = [pair for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
                     for pair in _bound(target, node.value)]
        else:
            pairs = [(key, None) for key in _rebinds(node)]
        now = frozenset(held)
        for key, value in pairs:
            held = {kept for kept in held if kept != key and not kept.startswith(key + ".")}
            if file.module(value, now):
                held.add(key)
    return frozenset(held)


def _promotes(file: _File, call: ast.Call, held: frozenset[str], aliases: frozenset[str]) -> bool:
    """Whether `call` promotes a module to float64: bound, on a module the file proves one; unbound, a
    class's `double`, `to` or `type` given the module first; or a function of the file that promotes
    its parameter (`_promoter`), given a module there and float64 as the dtype it promotes to."""
    function = call.func
    if isinstance(function, ast.Name) and function.id in file.promoters:
        place, promoted, dtype_place, dtype = file.promoters[function.id]
        if not file.module(_argument(call, place, promoted), held):
            return False
        given = None if dtype is None else _argument(call, dtype_place, dtype)
        return dtype is None or (given is not None and file.float64(given, aliases))
    if not isinstance(function, ast.Attribute) or function.attr not in PROMOTIONS:
        return False
    arguments = list(call.args)
    if file._constructs(function.value) and not file.module(function.value, held):
        arguments = arguments[1:]
        if len(arguments) == len(call.args):
            return False
    elif not file.module(function.value, held):
        return False
    if function.attr == "double":
        return True
    dtypes = [*arguments, *(keyword.value for keyword in call.keywords if keyword.arg == "dtype")]
    return any(file.float64(dtype, aliases) for dtype in dtypes)


def _scopes(body: Sequence[ast.stmt]) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef]:
    """The functions directly in a scope's `body`, a class's methods included."""
    for node in scopes.nodes_of(body):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            yield node
        elif isinstance(node, ast.ClassDef):
            yield from _scopes(node.body)


def promotions(tree: ast.Module, parent: Sequence[str]) -> Iterator[ast.Call]:
    """Every promotion of a module to float64 in a file (the module's docstring), in code order."""
    file = _File(tree, parent)
    found: list[ast.Call] = []

    def walk(body: Sequence[ast.stmt], around: frozenset[str], around_aliases: frozenset[str]) -> None:
        aliases = _aliases(file, body, around_aliases)
        held = _modules(file, body, around, found, aliases)
        for function in _scopes(body):
            walk(function.body, held, aliases)

    walk(tree.body, frozenset(), frozenset())
    yield from sorted(found, key=lambda node: (node.lineno, node.col_offset))


def message(call: ast.Call) -> str:
    """SLOP022's finding for one promotion."""
    called = f".{call.func.attr}" if isinstance(call.func, ast.Attribute) else _key(call.func)
    return (f"promotes a torch module built in float32 to float64 (`{called}`), which keeps what "
            "its constructor computed rounded to float32 (ledger 1601): build its twin inside "
            "tools/torch_precision.py's `float64`, by `float64_twin`")
