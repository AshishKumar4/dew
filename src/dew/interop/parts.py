"""Tier 2 from parts: an unregistered decoder assembled from the parts Dew's registered families are
built from, each read from the source's own modeling code (ledger 1607).

Every module of the source's causal LM is matched to a known part, a class of a registered family's
transformers 5.16.1 modeling (`KNOWN`), by its code (`same_code`): its forward is the known class's own,
inherited, or the same syntax tree under the source's names, which may differ from the known class's in
the model's class-name prefix and in a one-to-one renaming of the attributes the code reads off `self`.
The known part then reads, off the instance, the values that code reads (a head width, a norm's epsilon,
an activation, a router's top-k, whether the head is the embedding's own parameter), so a part's
semantics are the source's own, never a guess from its names or shapes. A module no known part matches,
or two parts that state one field differently, leave the decoder to tier 3 (`Unassembled`, naming it).

The renamings map the source's module paths onto the canonical ones `decoder_parts.dew_path` reads
(`Assembly.weight_path`), so a renamed checkpoint loads by its structure. `dew.interop.verify` confirms
an assembly on its shrunken probe before any weight downloads.
"""

from __future__ import annotations

import ast
import functools
import inspect
import linecache
import textwrap
from collections import ChainMap
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import cache
from types import FunctionType, MethodType, NoneType
from typing import TYPE_CHECKING

from dew import _records
from dew.interop import decoder_parts
from dew.interop.config_records import native_fields
from dew.interop.decoder_parts import ACTIVATIONS, PLAIN_SPELLING, DecoderFields
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.mla import MultiHeadLatentAttention

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel


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
    attributes), methods of the same function bound to instances alike, equal constants, or instances
    of one class whose own state is alike (a decorator's context manager, `torch.no_grad()`). A name
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


# --- the instances, read through the known names --------------------------------------------------

class PartView:
    """A source module read by the known class's attribute names (`Renaming.attributes`)."""

    def __init__(self, module: torch.nn.Module, renaming: Renaming):
        self.module, self.renaming = module, renaming

    def name(self, known: str) -> str:
        return self.renaming.attributes.get(known, known)

    def __getattr__(self, known: str):
        return getattr(self.module, self.name(known))


class Ambiguous(ValueError):
    """A part the source's code does not let tier 2 read, which leaves the decoder to tier 3."""


def gated_activation(module: torch.nn.Module) -> str:
    """The gated MLP Dew computes for the activation module a source part holds: the `hidden_act` name
    transformers' ACT2FN builds that same module for, read back by its class and state."""
    from transformers.activations import ACT2FN

    def state(activation: torch.nn.Module) -> dict[str, tuple]:
        """Each public attribute as it computes: a partial by its function and arguments, a bound
        method by its function, any other value by its class and value."""
        held = {}
        for key, value in vars(activation).items():
            if key.startswith('_') or key == 'training':
                continue
            if isinstance(value, functools.partial):
                held[key] = (value.func, value.args, tuple(sorted(value.keywords.items())))
            elif isinstance(value, MethodType):
                held[key] = (value.__func__,)
            else:
                held[key] = (type(value), value)
        return held

    for name, gated in ACTIVATIONS.items():
        built = ACT2FN[name]
        if type(built) is type(module) and state(built) == state(module):
            return gated
    raise Ambiguous(f"the activation {type(module).__name__} is none of {sorted(ACTIVATIONS)}, the gated "
                    "MLP's")


def _biased(*linears: torch.nn.Module) -> bool:
    biased = {linear.bias is not None for linear in linears}
    if len(biased) != 1:
        raise Ambiguous("projections that Dew biases together are biased apart")
    return biased.pop()


def _rope(config) -> Mapping[str, object]:
    """The rotary a `LlamaRotaryEmbedding`-shaped class builds from its config's standardized record
    (its rope_type through ROPE_INIT_FUNCTIONS, `decoder_parts.read_rope`), over the whole head."""
    parameters = dict(config.rope_parameters or {})
    if parameters.get('partial_rotary_factor', 1.0) != 1.0:
        raise Ambiguous("the rotary turns part of each head, which this rotation applies whole")
    entry = {key: value for key, value in parameters.items() if key != 'partial_rotary_factor'}
    try:
        ropes = decoder_parts.read_rope({'rope_parameters': entry,
                                         'max_position_embeddings': config.max_position_embeddings},
                                        set(), PLAIN_SPELLING, per_kind=False)
    except ValueError as error:
        raise Ambiguous(f"the rotary's record is not one this rotation reads ({error})") from error
    fields: dict[str, object] = {'rope_theta': ropes.theta}
    if ropes.scaling is not None:
        fields['rope_scaling'] = ropes.scaling
    return fields


# Readers: each takes the part's view and returns the CausalTransformer fields its code states. A
# field several parts state must agree (`assemble`); per-layer facts go under `_LAYER`.
_LAYER = '_layer'


def _read_causal_lm(view: PartView) -> Mapping[str, object]:
    if view.lm_head.bias is not None:
        raise Ambiguous("the head is biased, which this head's code does not add")
    return {}


def _read_scaled_causal_lm(view: PartView) -> Mapping[str, object]:
    """Granite's head: Llama's, its logits divided by the config's `logits_scaling`."""
    return {**_read_causal_lm(view), 'logits_scaling': float(view.config.logits_scaling)}


def _read_scaled_model(view: PartView) -> Mapping[str, object]:
    """Granite's model: Llama's, its embeddings times its `embedding_multiplier`."""
    return {**_read_llama_model(view), 'embedding_multiplier': float(view.embedding_multiplier)}


def _read_scaled_layer(view: PartView) -> Mapping[str, object]:
    """Granite's layer: Llama's, each sublayer's output times its `residual_multiplier` into the residual."""
    return {'residual_multiplier': float(view.residual_multiplier)}


def _read_llama_model(view: PartView) -> Mapping[str, object]:
    layers = len(view.layers)
    if view.config.num_hidden_layers != layers:
        raise Ambiguous(f"the forward runs {view.config.num_hidden_layers} of its {layers} layers")
    return {'num_layers': layers}


def _read_windowed_model(view: PartView) -> Mapping[str, object]:
    """Mixtral's and Mistral's model: every layer attends a window where the config states one
    (`create_sliding_window_causal_mask`)."""
    window = view.config.sliding_window
    fields = dict(_read_llama_model(view))
    if window is not None:
        layers = _records.integer(fields['num_layers'], 'num_layers')
        fields.update(layer_types=('sliding_attention',) * layers,
                      kinds={'sliding_attention': native_fields(LayerKind)(
                          window=_records.integer(window, 'sliding_window'))})
    return fields


def _read_layer_typed_model(view: PartView) -> Mapping[str, object]:
    """Qwen2's model: each layer attends under its config layer type's mask, a window where it is
    `sliding_attention` (`create_sliding_window_causal_mask` at the config's sliding_window)."""
    fields = dict(_read_llama_model(view))
    types = tuple(_records.strings(view.config.layer_types, 'layer_types'))[:fields['num_layers']]
    if set(types) - {'full_attention', 'sliding_attention'}:
        raise Ambiguous(f"the layer types {sorted(set(types))} are not attention's full or sliding masks")
    fields['layer_types'] = types
    if 'sliding_attention' in types:
        fields['kinds'] = {'sliding_attention': native_fields(LayerKind)(
            window=_records.integer(view.config.sliding_window, 'sliding_window'))}
    return fields


def _read_attention(view: PartView) -> Mapping[str, object]:
    width = view.head_dim
    if not view.is_causal:
        raise Ambiguous("the attention is not causal")
    heads, kv_heads = view.q_proj.out_features // width, view.k_proj.out_features // width
    if heads * width != view.q_proj.out_features or kv_heads * width != view.k_proj.out_features:
        raise Ambiguous("the projections hold no whole number of heads")
    biased = _biased(view.q_proj, view.k_proj, view.v_proj)
    fields: dict[str, object] = {
        'num_heads': heads, 'num_kv_heads': kv_heads, 'head_dim': width, 'qk_norm': False,
        'attention_bias': biased, 'attention_dropout_rate': float(view.attention_dropout)}
    if (view.o_proj.bias is not None) != biased:
        # Qwen2 biases its queries, keys and values and not its output.
        fields['o_proj_bias'] = view.o_proj.bias is not None
    if view.scaling != width ** -0.5:
        fields['attention_scale'] = float(view.scaling)
    return fields


def _read_qk_norm_attention(view: PartView) -> Mapping[str, object]:
    """Qwen3's attention: Llama's with an RMSNorm over each head's queries and keys (`q_norm`, `k_norm`,
    read as norms like the block's)."""
    return {**_read_attention(view), 'qk_norm': True}


def _read_nope_attention(view: PartView) -> Mapping[str, object]:
    """SmolLM3's attention: Llama's, rotating its queries and keys only where `use_rope` (each fourth layer
    attends without positions, `no_rope_layers`), a per-layer fact `assemble` makes a kind of."""
    return {**_read_attention(view), _LAYER: {'nope': not bool(view.use_rope)}}


def _read_projection_qk_norm_attention(view: PartView) -> Mapping[str, object]:
    """OLMo 2's attention: Llama's with an RMSNorm over the whole query and key projections, before the
    heads split (`qk_norm_scope='projection'`, OLMo 3's)."""
    return {**_read_attention(view), 'qk_norm': True, 'qk_norm_scope': 'projection'}


def _read_output_normed_layer(view: PartView) -> Mapping[str, object]:
    """OLMo 2's layer: each sublayer's output normed before it joins the residual, its input not
    (`sandwich_norms` without `pre_norms`, OLMo 3's)."""
    return {'sandwich_norms': True, 'pre_norms': False}


def _read_scale_first_norm(view: PartView) -> Mapping[str, object]:
    """OLMo 2's RMSNorm: its weight times the normalized input in float32, cast after."""
    return {'norm_eps': float(view.variance_epsilon), 'scale_after_cast': False, 'scale_offset': False}


def _read_mlp(view: PartView) -> Mapping[str, object]:
    return {'mlp': gated_activation(view.act_fn), 'mlp_features': view.gate_proj.out_features,
            'mlp_bias': _biased(view.gate_proj, view.up_proj, view.down_proj)}


def _read_mixtral_moe(view: PartView) -> Mapping[str, object]:
    if view.jitter_noise:
        raise Ambiguous("the router jitters its inputs while training")
    return {}


def _read_router(view: PartView, *, normalized: bool | None) -> Mapping[str, object]:
    norm = normalized if normalized is not None else bool(view.norm_topk_prob)
    return {_LAYER: {'top_k': int(view.top_k), 'experts': int(view.num_experts), 'norm_topk_prob': norm}}


def _read_sigmoid_router(view: PartView) -> Mapping[str, object]:
    """HY-3's router: sigmoid scores in float32, chosen by score plus the block's correction bias,
    weighted by score renormalized over the chosen and times `router_scaling_factor` (DeepSeek V3's)."""
    return {_LAYER: {'top_k': int(view.top_k), 'experts': int(view.num_experts), 'norm_topk_prob': True,
                     'score_function': 'sigmoid', 'scaling': float(view.router_scaling_factor), 'bias': True}}


def _read_grouped_router(view: PartView) -> Mapping[str, object]:
    """DeepSeek V3's router (GLM-4.7-Flash's): sigmoid scores in float32 plus its correction bias,
    experts chosen within the `topk_group` best of `num_group` groups (each scored by its two best),
    weighted by score, renormalized where `norm_topk_prob`, times `routed_scaling_factor`."""
    return {_LAYER: {'top_k': int(view.top_k), 'experts': int(view.num_experts),
                     'norm_topk_prob': bool(view.norm_topk_prob), 'score_function': 'sigmoid',
                     'scaling': float(view.routed_scaling_factor), 'bias': True,
                     'groups': int(view.num_group), 'groups_per_token': int(view.topk_group)}}


def _read_latent_attention(view: PartView) -> Mapping[str, object]:
    """DeepSeek V3's multi-head latent attention (GLM-4.7-Flash's): queries through a low-rank latent
    where `q_lora_rank` (whole otherwise), keys and values from one normed latent beside a shared rotated
    key, each head's query and key a rotated and an unrotated part (the `mla` mixer). Its latents are
    normed at their own epsilon, which the mixer holds at `MultiHeadLatentAttention.norm_eps`."""
    if not view.is_causal:
        raise Ambiguous("the attention is not causal")
    nope, rope = int(view.qk_nope_head_dim), int(view.qk_rope_head_dim)
    rank = view.q_lora_rank
    norms = [view.kv_a_layernorm] + ([] if rank is None else [view.q_a_layernorm])
    epsilon = MultiHeadLatentAttention.norm_eps
    if any(float(norm.variance_epsilon) != epsilon for norm in norms):
        raise Ambiguous(f"the latents are normed at an epsilon other than {epsilon}")
    heads = int(view.num_heads)
    return {'num_heads': heads, 'num_kv_heads': heads // int(view.num_key_value_groups),
            'head_dim': nope + rope, 'qk_norm': False,
            'attention_bias': view.kv_a_proj_with_mqa.bias is not None,
            'attention_dropout_rate': float(view.attention_dropout),
            'mixer': {'class': 'mla', 'fields': {
                'q_lora_rank': None if rank is None else int(rank), 'kv_lora_rank': int(view.kv_lora_rank),
                'qk_nope_head_dim': nope, 'qk_rope_head_dim': rope, 'v_head_dim': int(view.v_head_dim),
                'rope_interleave': bool(view.config.rope_interleave)}}}


def _read_shared_moe(view: PartView) -> Mapping[str, object]:
    """A routed block beside always-on shared experts (`shared_experts`, an MLP read as `shared`), the two
    summed: DeepSeek V3's, and HY-3's, whose float32 sum of two values of the compute dtype rounds as
    their sum there does (`enable_moe_fp32_combine`), so it reads no field."""
    return {}


def _read_shared_mlp(view: PartView) -> Mapping[str, object]:
    """A routed block's shared experts: Llama's MLP, its width the block's (`Mixture.shared_features`)."""
    width = int(view.gate_proj.out_features)
    return {'mlp': gated_activation(view.act_fn), _LAYER: {'shared_features': width}}


def _read_experts(view: PartView) -> Mapping[str, object]:
    return {'mlp': gated_activation(view.act_fn), _LAYER: {'expert_features': int(view.intermediate_dim)}}


def _read_rms_norm(view: PartView) -> Mapping[str, object]:
    return {'norm_eps': float(view.variance_epsilon), 'scale_after_cast': True, 'scale_offset': False}


def _read_offset_norm(view: PartView) -> Mapping[str, object]:
    """Gemma's norm: (1 + w) times the normalized input, in float32, cast after."""
    return {'norm_eps': float(view.eps), 'scale_after_cast': False, 'scale_offset': True}


def _read_embedding(view: PartView) -> Mapping[str, object]:
    """A lookup's table sizes and pad row. `nn.Embedding.forward` also renormalizes rows past
    `max_norm` and scales a token's gradient by its frequency (`scale_grad_by_freq`), which Dew's
    lookup does not."""
    if view.max_norm is not None or view.scale_grad_by_freq:
        raise Ambiguous("the embedding renormalizes its rows or scales its gradient by token frequency")
    return {'vocab_size': view.num_embeddings, 'emb_features': view.embedding_dim,
            'padding_idx': view.padding_idx}


def _read_scaled_embedding(view: PartView) -> Mapping[str, object]:
    """Gemma's scaled embedding: the lookup times sqrt(width), the factor cast to the table's dtype,
    which Dew's `embedding_scale` computes; any other factor is not Dew's."""
    if float(view.scalar_embed_scale) != view.embedding_dim ** 0.5:
        raise Ambiguous(f"the embedding scales by {view.scalar_embed_scale}, not by sqrt(width)")
    return {**_read_embedding(view), 'embedding_scale': True}


def _read_rotary(view: PartView) -> Mapping[str, object]:
    return _rope(view.config)


def _read_nothing(view: PartView) -> Mapping[str, object]:
    return {}


@dataclass(frozen=True)
class KnownPart:
    """A known class's place in a decoder: its role, the roles of the children its code calls (by the
    known class's attribute names, which are also their canonical module names under it), and the
    reader of what its code states."""
    role: str
    location: str
    """`module:qualname` of the known class."""
    children: Mapping[str, tuple[str, ...]]
    read: Callable[[PartView], Mapping[str, object]]
    parameters: tuple[str, ...] = ()
    """The parameters it holds itself, by the known class's names (a norm's weight)."""
    whole: bool = False
    """Whether every method of the known class is compared, its constructor's included: a part whose
    buffers its constructor computes (a rotary's frequencies), which the forward reads and the reader
    takes from the config."""
    state: Mapping[str, str] = field(default_factory=dict)
    """The state a checkpoint stores beside its parameters (a router's correction bias, a buffer), by the
    known class's names, onto its canonical name under the part (HY-3's MoE holds the bias its router
    reads, which Dew keeps in the router, `gate.e_score_correction_bias`)."""

    def known(self) -> type:
        module, _, qualname = self.location.partition(':')
        return getattr(__import__(module, fromlist=[qualname]), qualname)


_LLAMA = 'transformers.models.llama.modeling_llama'
_MIXTRAL = 'transformers.models.mixtral.modeling_mixtral'
_QWEN3 = 'transformers.models.qwen3.modeling_qwen3'
_QWEN3_MOE = 'transformers.models.qwen3_moe.modeling_qwen3_moe'
_GEMMA = 'transformers.models.gemma.modeling_gemma'
_MISTRAL = 'transformers.models.mistral.modeling_mistral'
_QWEN2 = 'transformers.models.qwen2.modeling_qwen2'
_GRANITE = 'transformers.models.granite.modeling_granite'
_SMOLLM3 = 'transformers.models.smollm3.modeling_smollm3'
_OLMO2 = 'transformers.models.olmo2.modeling_olmo2'
_HY_V3 = 'transformers.models.hy_v3.modeling_hy_v3'
_DEEPSEEK_V3 = 'transformers.models.deepseek_v3.modeling_deepseek_v3'
_GLM4_MOE = 'transformers.models.glm4_moe.modeling_glm4_moe'
Children = Mapping[str, tuple[str, ...]]
_MODEL_CHILDREN: Children = {'embed_tokens': ('embedding',), 'layers': ('layers',), 'norm': ('norm',),
                             'rotary_emb': ('rotary',)}
_LAYER_CHILDREN: Children = {'self_attn': ('attention',), 'mlp': ('mlp', 'moe'), 'input_layernorm': ('norm',),
                             'post_attention_layernorm': ('norm',)}
_ATTENTION_CHILDREN: Children = dict.fromkeys(('q_proj', 'k_proj', 'v_proj', 'o_proj'), ('linear',))
_SHARED_MOE_CHILDREN: Children = {'gate': ('router',), 'experts': ('experts',), 'shared_experts': ('shared',)}
_LATENT_CHILDREN: Children = {
    **dict.fromkeys(('q_proj', 'q_a_proj', 'q_b_proj', 'kv_a_proj_with_mqa', 'kv_b_proj', 'o_proj'),
                    ('linear',)),
    **dict.fromkeys(('q_a_layernorm', 'kv_a_layernorm'), ('latent_norm',))}

KNOWN: tuple[KnownPart, ...] = (
    KnownPart('causal_lm', f'{_LLAMA}:LlamaForCausalLM', {'model': ('model',), 'lm_head': ('linear',)},
         _read_causal_lm),
    KnownPart('causal_lm', f'{_MIXTRAL}:MixtralForCausalLM', {'model': ('model',), 'lm_head': ('linear',)},
         _read_causal_lm),
    KnownPart('causal_lm', f'{_GRANITE}:GraniteForCausalLM', {'model': ('model',), 'lm_head': ('linear',)},
              _read_scaled_causal_lm),
    KnownPart('causal_lm', f'{_HY_V3}:HYV3ForCausalLM', {'model': ('model',), 'lm_head': ('linear',)},
              _read_causal_lm),
    KnownPart('model', f'{_HY_V3}:HYV3Model', _MODEL_CHILDREN, _read_llama_model),
    KnownPart('model', f'{_LLAMA}:LlamaModel', _MODEL_CHILDREN, _read_llama_model),
    KnownPart('model', f'{_GRANITE}:GraniteModel', _MODEL_CHILDREN, _read_scaled_model),
    KnownPart('model', f'{_MIXTRAL}:MixtralModel', _MODEL_CHILDREN, _read_windowed_model),
    KnownPart('model', f'{_GEMMA}:GemmaModel', _MODEL_CHILDREN, _read_llama_model),
    KnownPart('model', f'{_MISTRAL}:MistralModel', _MODEL_CHILDREN, _read_windowed_model),
    KnownPart('model', f'{_QWEN2}:Qwen2Model', _MODEL_CHILDREN, _read_layer_typed_model),
    KnownPart('layer', f'{_LLAMA}:LlamaDecoderLayer', _LAYER_CHILDREN, _read_nothing),
    KnownPart('layer', f'{_MIXTRAL}:MixtralDecoderLayer', _LAYER_CHILDREN, _read_nothing),
    KnownPart('layer', f'{_GRANITE}:GraniteDecoderLayer', _LAYER_CHILDREN, _read_scaled_layer),
    KnownPart('layer', f'{_OLMO2}:Olmo2DecoderLayer',
              {'self_attn': ('attention',), 'mlp': ('mlp', 'moe'), 'post_attention_layernorm': ('norm',),
               'post_feedforward_layernorm': ('norm',)}, _read_output_normed_layer),
    KnownPart('attention', f'{_LLAMA}:LlamaAttention', _ATTENTION_CHILDREN, _read_attention),
    KnownPart('attention', f'{_MIXTRAL}:MixtralAttention', _ATTENTION_CHILDREN, _read_attention),
    KnownPart('attention', f'{_QWEN2}:Qwen2Attention', _ATTENTION_CHILDREN, _read_attention),
    KnownPart('attention', f'{_SMOLLM3}:SmolLM3Attention', _ATTENTION_CHILDREN, _read_nope_attention),
    KnownPart('attention', f'{_DEEPSEEK_V3}:DeepseekV3Attention', _LATENT_CHILDREN, _read_latent_attention),
    KnownPart('attention', f'{_HY_V3}:HYV3Attention',
              {**_ATTENTION_CHILDREN, 'q_norm': ('norm',), 'k_norm': ('norm',)}, _read_qk_norm_attention),
    KnownPart('attention', f'{_OLMO2}:Olmo2Attention',
              {**_ATTENTION_CHILDREN, 'q_norm': ('norm',), 'k_norm': ('norm',)},
              _read_projection_qk_norm_attention),
    KnownPart('attention', f'{_QWEN3}:Qwen3Attention',
         {**_ATTENTION_CHILDREN, 'q_norm': ('norm',), 'k_norm': ('norm',)}, _read_qk_norm_attention),
    KnownPart('mlp', f'{_LLAMA}:LlamaMLP', dict.fromkeys(('gate_proj', 'up_proj', 'down_proj'), ('linear',)),
         _read_mlp),
    KnownPart('moe', f'{_MIXTRAL}:MixtralSparseMoeBlock', {'gate': ('router',), 'experts': ('experts',)},
         _read_mixtral_moe),
    KnownPart('moe', f'{_QWEN3_MOE}:Qwen3MoeSparseMoeBlock', {'gate': ('router',), 'experts': ('experts',)},
         _read_nothing),
    KnownPart('moe', f'{_HY_V3}:HYV3MoE', _SHARED_MOE_CHILDREN, _read_shared_moe,
              state={'e_score_correction_bias': 'gate.e_score_correction_bias'}),
    KnownPart('shared', f'{_LLAMA}:LlamaMLP',
              dict.fromkeys(('gate_proj', 'up_proj', 'down_proj'), ('linear',)), _read_shared_mlp),
    KnownPart('router', f'{_HY_V3}:HYV3TopKRouter', {}, _read_sigmoid_router, ('weight',)),
    KnownPart('moe', f'{_DEEPSEEK_V3}:DeepseekV3MoE', _SHARED_MOE_CHILDREN, _read_shared_moe),
    KnownPart('router', f'{_DEEPSEEK_V3}:DeepseekV3TopkRouter', {}, _read_grouped_router,
              ('weight',), state={'e_score_correction_bias': 'e_score_correction_bias'}),
    KnownPart('latent_norm', f'{_LLAMA}:LlamaRMSNorm', {}, _read_nothing, ('weight',)),
    KnownPart('router', f'{_MIXTRAL}:MixtralTopKRouter', {}, lambda view: _read_router(view, normalized=True),
         ('weight',)),
    KnownPart('router', f'{_QWEN3_MOE}:Qwen3MoeTopKRouter', {},
              lambda view: _read_router(view, normalized=None), ('weight',)),
    KnownPart('experts', f'{_MIXTRAL}:MixtralExperts', {}, _read_experts, ('gate_up_proj', 'down_proj')),
    KnownPart('norm', f'{_LLAMA}:LlamaRMSNorm', {}, _read_rms_norm, ('weight',)),
    KnownPart('norm', f'{_GEMMA}:GemmaRMSNorm', {}, _read_offset_norm, ('weight',)),
    KnownPart('norm', f'{_OLMO2}:Olmo2RMSNorm', {}, _read_scale_first_norm, ('weight',)),
    KnownPart('rotary', f'{_LLAMA}:LlamaRotaryEmbedding', {}, _read_rotary, whole=True),
    KnownPart('rotary', f'{_QWEN2}:Qwen2RotaryEmbedding', {}, _read_rotary, whole=True),
    KnownPart('rotary', f'{_OLMO2}:Olmo2RotaryEmbedding', {}, _read_rotary, whole=True),
    KnownPart('rotary', f'{_GLM4_MOE}:Glm4MoeRotaryEmbedding', {}, _read_rotary, whole=True),
    KnownPart('embedding', 'torch.nn:Embedding', {}, _read_embedding, ('weight',)),
    KnownPart('embedding', f'{_GEMMA}:GemmaTextScaledWordEmbedding', {}, _read_scaled_embedding, ('weight',)),
    KnownPart('linear', 'torch.nn:Linear', {}, _read_nothing, ('weight', 'bias')),
)
"""The known parts: transformers' classes whose computation Dew's decoder states by the fields their readers
return, a registered family's or a type's own (Granite's, OLMo 2's), each assembly of them still verified
against its source's code (`dew.interop.verify`). A class that computes one of these by its code (Qwen2's
MLP is Llama's, Qwen3-MoE's experts Mixtral's, GLM-4.7-Flash's attention DeepSeek V3's) matches it through
`same_code`, so each entry stands for every copy of its code."""


@cache
def _known_by_role() -> dict[str, tuple[tuple[KnownPart, type], ...]]:
    table: dict[str, list[tuple[KnownPart, type]]] = {}
    for part in KNOWN:
        table.setdefault(part.role, []).append((part, part.known()))
    return {role: tuple(entries) for role, entries in table.items()}


def identify(module: torch.nn.Module, roles: tuple[str, ...]) -> tuple[KnownPart, Renaming]:
    """The known part `module` is in one of `roles`, and the renaming its code differs by, or
    `Ambiguous` naming the module's class. Its invocation is part of its code: an instance's own
    `forward` is none, and its class's `__call__` (transformers' `GradientCheckpointingLayer`'s, or
    torch's own) is the known class's."""
    cls = type(module)
    if 'forward' in vars(module):
        raise Ambiguous(f"{cls.__qualname__} is called through its instance's own forward")
    for role in roles:
        for part, known in _known_by_role().get(role, ()):
            renaming = matched_class(cls, known, whole=part.whole)
            if renaming is not None and _same_invocation(cls, known, renaming):
                return part, renaming
    raise Ambiguous(f"{cls.__module__}.{cls.__qualname__} computes no known {' or '.join(roles)} part by its "
                    "code")


def _same_invocation(candidate: type, known: type, renaming: Renaming) -> bool:
    """Whether calling a `candidate` runs what calling a `known` does around their forwards: the same
    `__call__`, or one of the same code."""
    ours, theirs = inspect.getattr_static(candidate, '__call__'), inspect.getattr_static(known, '__call__')
    return ours is theirs or (isinstance(ours, FunctionType) and isinstance(theirs, FunctionType)
                              and same_code(ours, theirs, renaming))


# --- the assembly ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Assembly:
    """A decoder assembled from known parts: its CausalTransformer fields and the canonical module path
    of each of the source's (`weight_path`)."""
    fields: DecoderFields
    names: Mapping[str, tuple[str, Mapping[str, str]]]
    """Each source module path that holds parameters: its canonical path, and its parameters' known
    names by their own."""
    packed: tuple[decoder_parts.Packed, ...] = ()
    """The stored tensors that hold several the path map reads, by their stored names: each experts
    module's fused gate and up projection and its down projection (`decoder_parts.FUSED_EXPERTS`'s)."""
    tied: tuple[str, str] = ('lm_head.weight', 'model.embed_tokens.weight')
    """The head's and the table's stored names, which a tied checkpoint stores two copies of."""
    dropouts: tuple[tuple[str, str], ...] = ()
    """Each attention module's path and the attribute its code reads its dropout rate from, which a
    deterministic training forward sets to zero (`dew.interop.verify`'s training qualification)."""
    loaded: Callable[[str], tuple[str, tuple[str, str] | None]] = lambda name: (name, None)
    """A stored tensor's name as the source's load names it, and for one expert's own tensor, which the
    load stacks into its experts module's fused one, the expert's index and projection
    (`_load_names`)."""

    def canonical(self, name: str) -> str | None:
        """A stored tensor's canonical name: its module's canonical path and its parameter's known
        name, a part a fused tensor splits into (`_SPLIT`) by its own, one expert's tensor of a
        checkpoint stored per expert by its index and projection; None where no part holds it."""
        loaded, expert = self.loaded(name)
        module, _, parameter = loaded.rpartition('.')
        held = self.names.get(module)
        if held is None:
            return None
        path, parameters = held
        if parameter.startswith(_SPLIT):
            return f'{path}.{parameter.removeprefix(_SPLIT)}'
        if parameter not in parameters:
            return None
        if expert is not None:
            index, projection = expert
            return f'{path}.{index}.{projection}.weight'
        return f'{path}.{parameters[parameter]}'


_PER_EXPERT = {'gate_up_proj': ('gate_proj', 'up_proj'), 'down_proj': ('down_proj',)}
"""Each fused experts tensor (`_fused_experts`) by the projections a checkpoint stored per expert holds
it as, in the order the fused one concatenates them."""


def _load_names(model: PreTrainedModel) -> Callable[[str], tuple[str, tuple[str, str] | None]]:
    """The name `model`'s load gives a stored tensor, by the conversion mapping transformers keeps for
    its model_type (`get_model_conversion_mapping`): its renamings applied (HY-3's `mlp.expert_bias` is
    its `mlp.e_score_correction_bias`), and an expert's own tensor `{experts}.{index}.{projection}.weight`
    named as the fused one the load stacks it into, with its index and projection. A merge is read only
    where it stacks experts on the first axis and concatenates `_PER_EXPERT`'s projections in order on
    the second, as the fused tensor `_fused_experts` splits holds them; any other conversion's tensor
    keeps its own name, which no part holds."""
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.core_model_loading import (
        Concatenate,
        MergeModulelist,
        WeightConverter,
        WeightRenaming,
        rename_source_key,
    )

    transforms = get_model_conversion_mapping(model)
    renamings = [transform for transform in transforms if isinstance(transform, WeightRenaming)]
    converters = [transform for transform in transforms if isinstance(transform, WeightConverter)]

    def stacks(converter: WeightConverter, position: int, fused: str, projection: str) -> bool:
        projections = _PER_EXPERT.get(fused, ())
        operations = [(type(operation), operation.dim) for operation in converter.operations
                      if isinstance(operation, (MergeModulelist, Concatenate))]
        return (len(converter.source_patterns) == len(projections) > position
                and projections[position] == projection
                and len(operations) == len(converter.operations)
                and operations == [(MergeModulelist, 0)] + [(Concatenate, 1)] * (len(projections) > 1))

    def loaded(name: str) -> tuple[str, tuple[str, str] | None]:
        renamed, _ = rename_source_key(name, renamings, [])
        for converter in converters:
            target, pattern = converter.rename_source_key(renamed)
            if pattern is None:
                continue
            module, _, fused = target.rpartition('.')
            index, _, projection = renamed.removeprefix(f'{module}.').removesuffix('.weight').partition('.')
            if (renamed == f'{module}.{index}.{projection}.weight' and index.isdigit()
                    and stacks(converter, converter.source_patterns.index(pattern), fused, projection)):
                return target, (index, projection)
            return renamed, None
        return renamed, None

    return loaded



class _Assembler:
    def __init__(self):
        self.fields: dict[str, object] = {}
        self.layers: dict[int, dict[str, object]] = {}
        self.names: dict[str, tuple[str, Mapping[str, str]]] = {}
        self.tables: dict[str, tuple[str, object]] = {}
        self.packed: list[decoder_parts.Packed] = []
        self.dropouts: list[tuple[str, str]] = []

    def state(self, fields: Mapping[str, object], where: str, layer: int | None) -> None:
        for key, value in fields.items():
            if key == _LAYER:
                if layer is None:
                    raise Ambiguous(f"{where} routes outside a layer")
                self.layers.setdefault(layer, {}).update(_records.record(value, _LAYER))
            elif key in self.fields and self.fields[key] != value:
                raise Ambiguous(f"{where} states {key}={value!r} where another part states "
                                f"{self.fields[key]!r}")
            else:
                self.fields[key] = value

    def walk(self, module: torch.nn.Module, roles: tuple[str, ...], path: str, canonical: str,
             layer: int | None) -> str:
        """Identify `module` (at `path`, canonically `canonical`), read it and its children; return its
        role."""
        import torch

        if roles == ('layers',):
            if not isinstance(module, torch.nn.ModuleList):
                raise Ambiguous(f"{path} holds its layers in a {type(module).__name__}")
            for index, child in enumerate(module):
                self.walk(child, ('layer',), f'{path}.{index}', f'{canonical}.{index}', index)
            return 'layers'
        part, renaming = identify(module, roles)
        view = PartView(module, renaming)
        self.state(part.read(view), path or type(module).__name__, layer)
        # The head is tied where it is the table's own parameter, as the source holds it.
        if part.role == 'embedding' or (part.role == 'linear' and canonical == 'lm_head'):
            table = 'embedding' if part.role == 'embedding' else 'head'
            self.tables[table] = (f'{path}.{view.name("weight")}', view.weight)
        if part.role == 'experts':
            self.packed.extend(_fused_experts(path.rpartition('.')[2], view))
        if part.role == 'attention':
            self.dropouts.append((path, view.name('attention_dropout')))
        held = {name for name, _ in module.named_parameters(recurse=False)}
        expected = {view.name(name): name for name in part.parameters}
        if held - set(expected):
            raise Ambiguous(f"{path} holds parameters {sorted(held - set(expected))} its code does not read")
        stored = {view.name(name): known for name, known in part.state.items()
                  if view.name(name) in dict(module.named_buffers(recurse=False))}
        if held or stored:
            self.names[path] = (canonical, {**{name: expected[name] for name in held}, **stored})
        children = dict(module.named_children())
        claimed = set()
        for known, child_roles in part.children.items():
            name = view.name(known)
            claimed.add(name)
            child = children.get(name)
            if child is None:
                continue
            role = self.walk(child, child_roles, f'{path}.{name}' if path else name,
                             f'{canonical}.{known}' if canonical else known, layer)
            if part.role == 'layer' and known == 'mlp' and layer is not None:
                self.layers.setdefault(layer, {})['routed'] = role == 'moe'
        unclaimed = [name for name, child in children.items() if name not in claimed
                     and any(True for _ in child.parameters())]
        if unclaimed:
            raise Ambiguous(f"{path or type(module).__name__} holds {unclaimed}, which its code does not "
                            "call")
        return part.role


def assemble(model: PreTrainedModel) -> Assembly:
    """`model`, a source's causal LM, assembled from known parts, or `Ambiguous` naming the module that
    is not one."""
    assembler = _Assembler()
    assembler.walk(model, ('causal_lm',), '', '', None)
    (head, head_weight), (table, table_weight) = assembler.tables['head'], assembler.tables['embedding']
    fields = {**assembler.fields, 'tie_embeddings': head_weight is table_weight}
    routed = sorted(layer for layer, facts in assembler.layers.items() if facts.get('routed'))
    if routed:
        routes = {tuple(sorted((key, value) for key, value in assembler.layers[layer].items()
                               if key not in ('routed', 'nope'))) for layer in routed}
        if len(routes) != 1:
            raise Ambiguous("the routed layers route differently")
        mixture = dict(routes.pop())
        width = _records.integer(mixture['expert_features'], 'expert_features')
        # A router's scores, scaling and correction bias and a block's shared experts, where its parts
        # state them (DeepSeek's routing, HY-3's).
        stated = {key: mixture[key] for key in ('score_function', 'scaling', 'bias', 'shared_features',
                                                'groups', 'groups_per_token') if key in mixture}
        fields['mixture'] = {**native_fields(Mixture)(
            experts=_records.integer(mixture['experts'], 'experts'),
            top_k=_records.integer(mixture['top_k'], 'top_k'),
            norm_topk_prob=bool(mixture['norm_topk_prob']), expert_features=width,
            layers=None if len(routed) == fields['num_layers'] else tuple(routed)), **stated}
        fields.setdefault('mlp_features', width)
    for required in ('vocab_size', 'num_layers', 'num_heads', 'mlp', 'mlp_features', 'norm_eps',
                     'rope_theta'):
        if required not in fields:
            raise Ambiguous(f"no part states {required}")
    unrotated = sorted(layer for layer, facts in assembler.layers.items() if facts.get('nope'))
    fields.update(_without_positions(fields, unrotated))
    capacity, context = decoder_parts.context_lengths(
        {'max_position_embeddings': model.config.to_dict().get('max_position_embeddings')})
    fields.update(max_seq_len=capacity, context_length=context)
    kinds = fields.pop('kinds', {})
    record = native_fields(CausalTransformer)(**fields)
    record['kinds'] = kinds
    return Assembly(record, assembler.names, tuple(dict.fromkeys(assembler.packed)), (head, table),
                    tuple(assembler.dropouts), _load_names(model))


def _without_positions(fields: Mapping[str, object], layers: list[int]) -> Mapping[str, object]:
    """The layer types and kinds that make each of `layers` (the ones whose attention rotates nothing,
    SmolLM3's NoPE layers) a kind of its own beside its layer type's, its attention taking no positions,
    as Llama 4's NoPE layers do."""
    if not layers:
        return {}
    count = _records.integer(fields['num_layers'], 'num_layers')
    stated_types = fields.get('layer_types')
    types = (list(_records.strings(stated_types, 'layer_types')) if stated_types
             else ['full_attention'] * count)
    kinds = dict(_records.record(fields.get('kinds', {}), 'kinds'))
    for layer in layers:
        stated = types[layer]
        unrotated = f'{stated}_nope'
        if unrotated not in kinds:
            kind = dict(_records.record(kinds.get(stated) or native_fields(LayerKind)(), stated))
            kind['mixer'] = {'class': 'attention', 'fields': {'nope': True}}
            kinds[unrotated] = kind
        types[layer] = unrotated
    return {'layer_types': tuple(types), 'kinds': kinds}


def _fused_experts(module: str, view: PartView) -> tuple[decoder_parts.Packed, ...]:
    """An experts module's fused tensors by their stored names (`module`, its attribute; the view, its
    parameters'), split into the parts the path map reads as `decoder_parts.FUSED_EXPERTS` splits
    Mixtral's."""
    fused, down = view.name('gate_up_proj'), view.name('down_proj')
    halves = (f'.{module}.{_SPLIT}gate_proj', f'.{module}.{_SPLIT}up_proj')
    return (decoder_parts.Packed(f'.{module}.{fused}', halves, -1, (0, 2, 1)),
            decoder_parts.Packed(f'.{module}.{down}', (f'.{module}.{_SPLIT}down_proj',), -1, (0, 2, 1)))


_SPLIT = '~'
"""The mark of a part a fused tensor splits into (`_fused_experts`), already canonical: no Python
attribute holds it, so it names no stored parameter, whatever the source renames its own to."""


_STACKED_EXPERTS: decoder_parts.Leaves = {
    f'mlp.experts.{name}': ('mlp', 'experts', name, 'kernel') for name in decoder_parts.MOE_SHARED}
"""The canonical experts, each projection one tensor over every expert (`decoder_parts.FUSED_EXPERTS`)."""


def family(model_type: str, assembly: Assembly) -> decoder_parts.DecoderFamily:
    """The decoder family a type verified from parts loads through (`dew.interop.verify`): `assembly`'s
    fields for its config, every field of which the source's own code read, and each stored tensor at
    the path of its canonical name (`Assembly.canonical`), a fused expert tensor split as Mixtral's
    first (`DecoderFamily.packed`), so an export writes each back under its stored name."""
    def translated(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
        used.update(hf_config)
        return assembly.fields.copy()

    def weight_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
        canonical = assembly.canonical(name)
        if canonical is None:
            raise ValueError(f"the checkpoint's {name!r} is held by no part of the assembled decoder")
        return decoder_parts.renamed_path((), canonical, config, layer_leaves=_STACKED_EXPERTS)

    return decoder_parts.DecoderFamily(
        (model_type,), translated, lambda fields: False, model_type, 'ForCausalLM', lambda model: {},
        preserve_source_layout=True, weight_path=weight_path, packed=assembly.packed,
        tied_head_names=assembly.tied)
