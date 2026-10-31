"""Code identity for tier 2 from parts (`dew.interop.parts`): whether a source's function computes a known
one by its code (`same_code`), the same function or the same syntax tree under the source's names, which
may differ from the known class's in the model's class-name prefix and in a one-to-one renaming of the
attributes the code reads off `self` (`Renaming`), each free name it reads bound alike; and whether a
source's class computes a known class by its forward and every method that forward reads
(`matched_class`).
"""

from __future__ import annotations

import ast
import inspect
import linecache
import textwrap
from collections import ChainMap
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cache
from types import FunctionType, MethodType, NoneType

# --- code identity ---------------------------------------------------------------------------------

_POSITIONS = frozenset({'lineno', 'col_offset', 'end_lineno', 'end_col_offset', 'type_comment'})
_ANNOTATIONS = frozenset({'annotation', 'returns', 'type_params'})
"""Syntax fields that do not change what a function computes: positions, and annotations (which name
the model's own classes)."""


class _Unannotated(ast.NodeTransformer):
    """A function's statements with their annotations dropped: `x: T = v` is `x = v`, and a bare
    `x: T` states nothing."""

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
        if node.value is None:
            return None
        return ast.copy_location(ast.Assign(targets=[node.target], value=node.value), node)


def _source_tree(function: FunctionType) -> ast.FunctionDef | None:
    """`function`'s definition without its docstring or annotations, or None where its module's loader
    holds no source for it."""
    if not linecache.getlines(function.__code__.co_filename, function.__globals__):
        return None
    node = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return None
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        node.body = body[1:]
    return _Unannotated().visit(node)


_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.ListComp, ast.SetComp,
           ast.DictComp, ast.GeneratorExp)
"""The nodes that open a scope of their own inside a function."""


def _bound(node: ast.FunctionDef) -> tuple[set[str], set[str]]:
    """The names a function binds in its own scope (its arguments and the names its body stores, past
    any `global` or `nonlocal` it declares), and the names a scope nested in it binds (a
    comprehension's targets, a lambda's or an inner function's arguments and locals)."""
    arguments = node.args
    own = {argument.arg for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)}
    own |= {argument.arg for argument in (arguments.vararg, arguments.kwarg) if argument is not None}
    nested: set[str] = set()
    declared: set[str] = set()

    def visit(child: ast.AST, names: set[str]) -> None:
        if isinstance(child, ast.Global | ast.Nonlocal):
            declared.update(child.names)
        elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store | ast.Del):
            names.add(child.id)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            names.add(child.name)
        elif isinstance(child, ast.alias):
            names.add(child.asname or child.name.split('.')[0])
        elif isinstance(child, _SCOPES):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                names.add(child.name)
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
                inner = child.args
                nested.update(argument.arg for argument in
                              (*inner.posonlyargs, *inner.args, *inner.kwonlyargs, inner.vararg, inner.kwarg)
                              if argument is not None)
            for grandchild in ast.iter_child_nodes(child):
                visit(grandchild, nested)
            return
        for grandchild in ast.iter_child_nodes(child):
            visit(grandchild, names)

    for statement in node.body:
        visit(statement, own)
    return own - declared, nested - own


def prefixes(candidate: str, known: str) -> tuple[str, str]:
    """The model-name prefixes of two class names that share their part's name: `MixedDecoderLayer`
    and `LlamaDecoderLayer` give ('Mixed', 'Llama'); identical names give empty ones."""
    shared = 0
    while shared < min(len(candidate), len(known)) and candidate[-1 - shared] == known[-1 - shared]:
        shared += 1
    # The shared part starts at a capital, as a class name's parts do.
    while shared and not candidate[len(candidate) - shared].isupper():
        shared -= 1
    return candidate[:len(candidate) - shared], known[:len(known) - shared]


@dataclass
class Renaming:
    """What two functions' code differs by: the source's attribute names off `self` against the known
    class's (`attributes`, one to one), and the model-name prefixes of their class names."""
    candidate_prefix: str = ''
    known_prefix: str = ''
    attributes: dict[str, str] = field(default_factory=dict)
    """The source's attribute name by the known class's."""

    def attribute(self, candidate: str, known: str) -> bool:
        if self.attributes.get(known, candidate) != candidate:
            return False
        if any(theirs == candidate and ours != known for ours, theirs in self.attributes.items()):
            return False
        self.attributes[known] = candidate
        return True

    def identifier(self, candidate: str, known: str) -> bool:
        if candidate == known:
            return True
        ours, theirs = self.candidate_prefix, self.known_prefix
        return bool(ours or theirs) and candidate.startswith(ours) and known.startswith(theirs) \
            and candidate[len(ours):] == known[len(theirs):]


class _Comparison:
    """Two functions compared node by node (`same_code`), collecting the free names each reads. Their
    own names (arguments and locals) may differ one to one; a name a nested scope binds is compared as
    written and its free reading too (`nested`), since outside that scope it is the module's."""

    def __init__(self, renaming: Renaming, candidate: ast.FunctionDef, known: ast.FunctionDef):
        self.renaming = renaming
        (ours, ours_nested), (theirs, theirs_nested) = _bound(candidate), _bound(known)
        self.locals = (ours, theirs)
        self.nested = (ours_nested, theirs_nested)
        self.local: dict[str, str] = {}
        self.names: list[tuple[str, str]] = []
        self.optional: list[tuple[str, str]] = []
        """Free names a nested scope binds too: unresolved on both sides is that scope's own."""

    def own(self, a: str, b: str) -> bool:
        """Whether the local `a` is the local `b`, one to one."""
        if self.local.get(a, b) != b or any(theirs == b and ours != a for ours, theirs in self.local.items()):
            return False
        self.local[a] = b
        return True

    def same(self, a, b) -> bool:
        if type(a) is not type(b):
            return False
        if isinstance(a, list):
            return len(a) == len(b) and all(self.same(x, y) for x, y in zip(a, b, strict=True))
        if not isinstance(a, ast.AST):
            return a == b
        if (isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name) and a.value.id == 'self'
                and isinstance(b.value, ast.Name) and b.value.id == 'self'):
            return type(a.ctx) is type(b.ctx) and self.renaming.attribute(a.attr, b.attr)
        if isinstance(a, ast.arg):
            return self.own(a.arg, b.arg)
        if isinstance(a, ast.Name):
            if type(a.ctx) is not type(b.ctx):
                return False
            if a.id in self.locals[0] or b.id in self.locals[1]:
                return a.id in self.locals[0] and b.id in self.locals[1] and self.own(a.id, b.id)
            if a.id in self.nested[0] or b.id in self.nested[1]:
                if a.id != b.id:
                    return False
                self.optional.append((a.id, b.id))
                return True
            if not self.renaming.identifier(a.id, b.id):
                return False
            self.names.append((a.id, b.id))
            return True
        # A definition's own name is the binding its callers match it by (an attribute off `self`, a
        # local), already paired where they read it.
        named = isinstance(a, ast.FunctionDef | ast.AsyncFunctionDef)
        return all(self.same(getattr(a, name), getattr(b, name)) for name in a._fields
                   if name not in _POSITIONS and name not in _ANNOTATIONS and not (named and name == 'name'))


_MISSING = object()
"""A free name neither the function's closure, its module nor builtins hold."""
_WRAPPED = object()
"""A decorator's wrapper's binding of the function it wraps, which is compared on its own."""


def _bindings(function: FunctionType, wrapped: FunctionType | None = None) -> Mapping[str, object]:
    """Every name `function` reads free, as it holds them when it runs: its closure's cells, over its
    module's globals, over its own builtins; a wrapper's binding of `wrapped` as `_WRAPPED`."""
    cells = {name: _WRAPPED if cell.cell_contents is wrapped else cell.cell_contents
             for name, cell in zip(function.__code__.co_freevars, function.__closure__ or (), strict=True)}
    return ChainMap(cells, function.__globals__, function.__builtins__)


def _wrappers(function: FunctionType) -> tuple[list[tuple[FunctionType, FunctionType]], FunctionType]:
    """The decorators' wrappers around `function` (`functools.wraps`'s `__wrapped__` chain, which it
    stores in the wrapper's own attributes), outermost first, each with the function it wraps, and the
    innermost function, whose source `inspect.getsource` reads."""
    chain: list[tuple[FunctionType, FunctionType]] = []
    while isinstance(wrapped := vars(function).get('__wrapped__'), FunctionType):
        chain.append((function, wrapped))
        function = wrapped
    return chain, function


def same_code(candidate: FunctionType, known: FunctionType, renaming: Renaming, seen: set | None = None, *,
              collect: bool = False) -> bool:
    """Whether `candidate` computes what `known` does by its code: the same function, or the same syntax
    tree under `renaming` (extended with the attributes off `self` the two read), each free name it
    reads bound alike (`_same_bindings`). A function whose source is not Python's is the same only as
    itself. With `collect`, one function compared with itself is still read, so `renaming` holds the
    attributes it reads."""
    if candidate is known and not collect:
        return True
    seen = set() if seen is None else seen
    if (id(candidate), id(known)) in seen:
        return True
    seen.add((id(candidate), id(known)))
    unattributed = Renaming(renaming.candidate_prefix, renaming.known_prefix)
    # Each decorator's wrapper runs the same code over the same bindings, the wrapped function aside.
    (ours_chain, candidate), (theirs_chain, known) = _wrappers(candidate), _wrappers(known)
    if len(ours_chain) != len(theirs_chain):
        return False
    for (ours, ours_inner), (theirs, theirs_inner) in zip(ours_chain, theirs_chain, strict=True):
        # Its closure, every global or builtin its code names (an attribute name among them binds
        # nothing either side), and its defaults.
        names = [(name, name) for name in (*ours.__code__.co_freevars, *ours.__code__.co_names)]
        if ours.__code__ is not theirs.__code__ or not _same_bindings(
                _bindings(ours, ours_inner), _bindings(theirs, theirs_inner), names, unattributed, seen,
                unbound=True) or not _same_defaults(ours, theirs, unattributed, seen):
            return False
    ours_tree, known_tree = _source_tree(candidate), _source_tree(known)
    if ours_tree is None or known_tree is None:
        return candidate is known
    comparison = _Comparison(renaming, ours_tree, known_tree)
    if not comparison.same(ours_tree, known_tree):
        return False
    ours_bound, known_bound = _bindings(candidate), _bindings(known)
    return (_same_bindings(ours_bound, known_bound, comparison.optional, unattributed, seen, unbound=True)
            and _same_bindings(ours_bound, known_bound, comparison.names, unattributed, seen)
            and _same_defaults(candidate, known, unattributed, seen))


def _same_defaults(ours: FunctionType, theirs: FunctionType, renaming: Renaming, seen: set) -> bool:
    """Whether two functions hold alike defaults when they run (`__defaults__`, `__kwdefaults__`),
    whatever their source writes: a default bound when a factory made the function, or set after."""
    positional = (ours.__defaults__ or (), theirs.__defaults__ or ())
    keywords = (ours.__kwdefaults__ or {}, theirs.__kwdefaults__ or {})
    if len(positional[0]) != len(positional[1]) or keywords[0].keys() != keywords[1].keys():
        return False
    held = ({**{str(index): value for index, value in enumerate(positional[0])}, **keywords[0]},
            {**{str(index): value for index, value in enumerate(positional[1])}, **keywords[1]})
    return _same_bindings(*held, [(name, name) for name in held[0]], renaming, seen)


@cache
def _grad_modes() -> tuple[type, ...]:
    """torch's grad-mode context managers, which decorate a function (`torch.no_grad()`) by their
    attributes alone."""
    import torch

    return (torch.no_grad, torch.enable_grad, torch.inference_mode, torch.set_grad_enabled)


def _same_bindings(ours: Mapping[str, object], theirs: Mapping[str, object], names: Iterable[tuple[str, str]],
                   renaming: Renaming, seen: set, *, unbound: bool = False) -> bool:
    """Whether each pair of `names` is bound alike in `ours` and `theirs`: the same object, functions of
    the same code (`same_code`; a module-level helper reads no `self`, so `renaming` holds no
    attributes), methods of the same function bound to instances alike, equal constants, tuples whose
    items are bound alike (a kernel decorator's parameter names, which LFM2-MoE's copy of LFM2's
    convolution helpers holds), or instances of one class whose own state is alike (a decorator's
    context manager, `torch.no_grad()`). A name
    neither binds is a NameError when it runs, not evidence of the same code, unless `unbound` (a name
    a nested scope binds, which reads its own)."""
    for our_name, their_name in names:
        a, b = ours.get(our_name, _MISSING), theirs.get(their_name, _MISSING)
        if a is b and (a is not _MISSING or unbound):
            continue
        if isinstance(a, FunctionType) and isinstance(b, FunctionType):
            if not same_code(a, b, renaming, seen):
                return False
        elif isinstance(a, MethodType) and isinstance(b, MethodType):
            if a.__func__ is not b.__func__ or not _same_bindings(
                    {'self': a.__self__}, {'self': b.__self__}, [('self', 'self')], renaming, seen):
                return False
        elif a is _MISSING or b is _MISSING or type(a) is not type(b) or isinstance(a, type):
            return False
        elif isinstance(a, int | float | str | bool | bytes | NoneType):
            if a != b:
                return False
        elif isinstance(a, tuple) and isinstance(b, tuple):
            if len(a) != len(b) or not _same_bindings(
                    {str(index): item for index, item in enumerate(a)},
                    {str(index): item for index, item in enumerate(b)},
                    [(str(index), str(index)) for index in range(len(a))], renaming, seen):
                return False
        # A grad-mode decorator keeps its whole state in its instance's attributes; any other object
        # computes alike only as itself.
        elif not (isinstance(a, _grad_modes()) and vars(a).keys() == vars(b).keys() and _same_bindings(
                vars(a), vars(b), [(name, name) for name in vars(a)], renaming, seen)):
            return False
    return True


def matched_class(candidate: type, known: type, *, whole: bool = False) -> Renaming | None:
    """The renaming under which `candidate`'s forward computes `known`'s (`same_code`), every method of
    its own the forward reads off `self` included, or None; with `whole`, every function the known
    class itself defines, its constructor's included. A subclass that keeps the known forward matches
    under no renaming."""
    candidate_prefix, known_prefix = prefixes(candidate.__name__, known.__name__)
    renaming = Renaming(candidate_prefix, known_prefix)
    if not same_code(candidate.forward, known.forward, renaming, collect=True):
        return None
    if whole:
        for name, theirs in vars(known).items():
            theirs = theirs.__func__ if isinstance(theirs, staticmethod | classmethod) else theirs
            if isinstance(theirs, FunctionType) and name != 'forward':
                ours = inspect.getattr_static(candidate, renaming.attributes.get(name, name), None)
                ours = ours.__func__ if isinstance(ours, staticmethod | classmethod) else ours
                if not (isinstance(ours, FunctionType) and same_code(ours, theirs, renaming, collect=True)):
                    return None
    compared: set[str] = set()
    while pending := [name for name in renaming.attributes if name not in compared]:
        for name in pending:
            compared.add(name)
            theirs = inspect.getattr_static(known, name, None)
            ours = inspect.getattr_static(candidate, renaming.attributes[name], None)
            if inspect.isfunction(theirs):
                if not (inspect.isfunction(ours) and same_code(ours, theirs, renaming, collect=True)):
                    return None
            elif inspect.isfunction(ours):
                return None
    return renaming
