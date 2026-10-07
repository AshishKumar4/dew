"""Low-rank adapters (LoRA) for Dew models.

LoRA is from arXiv 2106.09685. Next to a target kernel `W` of shape
`[in..., out...]`, the tree holds `lora_A` of shape `[in..., r]` and `lora_B` of
shape `[r, out...]`, and the module computes `x W + scale * (x A) B` with
`scale = alpha / r` (`alpha / sqrt(r)` for rsLoRA, arXiv 2312.03732). Merging
adds `scale * A B` into the kernel and removes the factors. A target is its
module's path in the tree, such as `("params", "layers_0", "self_attn", "q_proj")`,
so one description works for every model.

`LoRA` is the adapter spec a user writes and a run records, with PEFT's own
fields. `LoRA.apply` attaches it to a model and its variables, and `LoRA.load`
reads one from disk. Both return an `Adapter`, which holds the adapted model;
the variables, with the factors under `params` and every base weight under
`FROZEN`, so any objective trains only the factors; and the bindings that
`merge` and `save` use. The adapted model wraps `apply` and `init` in a Flax
method interceptor instead of swapping modules, and adds `FROZEN` back into
`params` itself. It also handles what PEFT's wrapper layers do: the factor
shapes of a `DenseGeneral` with several contracted axes, the kernel path's
compute dtype, dropout on the branch input, and the parameter names that the
merge and the export agree on.

The file formats are the references' own. PEFT's directory
(`adapter_config.json`, `adapter_model.safetensors`, keys
`base_model.model.<module>.lora_A.weight`) is what Transformers loads. The
Diffusers file (`pytorch_lora_weights.safetensors`, keys
`<component>.<module>.lora_A.weight`, with each component's PEFT config in the
header's `lora_adapter_metadata`) is what a pipeline's `load_lora_weights`
reads. Kohya/sgm keys are not accepted. Module names resolve to tree paths
through a loaded source's `Pretrained.layouts`; a model built from the registry
passes none, and `bound_layouts` reads the names and shapes from its own
kernels.
"""
from __future__ import annotations

import dataclasses
import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path as FilePath
from typing import ClassVar, Protocol, TypedDict, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.dtypes import promote_dtype
from flax.linen.module import Interceptor

from dew.interop.safetensors_io import read_file, write_file
from dew.interop.streaming import WeightLayout
from dew.nn.backbones.layer_plan import group_layers
from dew.objectives.base import Path, Variables, freeze, merge as overlay, select, thaw

PEFT_CONFIG = "adapter_config.json"
PEFT_WEIGHTS = "adapter_model.safetensors"
PEFT_PREFIX = "base_model.model."
DIFFUSERS_WEIGHTS = "pytorch_lora_weights.safetensors"
DIFFUSERS_METADATA = "lora_adapter_metadata"

FACTORS = ("lora_A", "lora_B")
# PEFT's init: A as torch's Linear default, kaiming-uniform with a=sqrt(5),
# which is uniform on +-1/sqrt(fan_in); B zero, so a fresh adapter is the
# identity.
INIT_A = nn.initializers.variance_scaling(1 / 3, "fan_in", "uniform")
INIT_B = nn.initializers.zeros


@dataclass(frozen=True)
class Target:
    """One adapted kernel's rank and alpha."""

    rank: int
    alpha: float


@dataclass(frozen=True)
class LoRA:
    """A low-rank adapter spec with PEFT's own fields.

    A user writes it and a run records it (`RunConfig.lora`,
    `--lora.rank 16 --lora.modules q_proj v_proj`).

    `modules` are PEFT's `target_modules`. A projection matches when its name
    relative to the model (`model.layers.0.self_attn.q_proj`, or `to_q` under a
    pipeline component) equals an entry or ends in `.` followed by the entry.
    `alpha` None means twice the rank. `rslora` scales by `alpha / sqrt(rank)`, and
    `dropout` drops the branch's input in a training forward pass. `rank` must be a
    positive int, `modules` must name at least one projection, and `dropout` must be
    in [0, 1).
    """

    rank: int
    modules: tuple[str, ...]
    alpha: float | None = None
    rslora: bool = False
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.modules, str):
            raise ValueError("modules is a sequence of module names, not one string")
        object.__setattr__(self, "modules", tuple(self.modules))
        if type(self.rank) is not int or self.rank < 1:
            raise ValueError(f"rank must be a positive integer, not {self.rank!r}")
        if not self.modules or any(not isinstance(module, str) or not module for module in self.modules):
            raise ValueError("modules names at least one projection")
        if self.alpha is not None and (isinstance(self.alpha, bool) or not math.isfinite(self.alpha)):
            raise ValueError(f"alpha must be a finite number, not {self.alpha!r}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), not {self.dropout!r}")

    def apply(self, model: nn.Module, variables: Variables, *, key: int | jax.Array,
              layouts: Mapping[str, WeightLayout] | None = None) -> Adapter:
        """Attach this adapter to `model` over `variables` and draw its factors.

        `layouts` maps module names to tree paths: `Pretrained.layouts` for a loaded
        source, or None for a model built from its class, whose own module paths are
        its names. An entry of `modules` that matches no projection is refused. A is
        drawn per module from `key` in sorted name order, the way PEFT draws it, and B
        is zero, so the adapted model computes exactly what the source does.
        """
        from dew.nn.inputs import request_key

        if isinstance(type(model), _Adapted):
            raise ValueError("The model is already adapted")
        variables = thaw(variables)
        bound = _named(bound_layouts(model, variables, layouts or {}), self.modules)
        scaling = 2.0 * self.rank if self.alpha is None else float(self.alpha)
        targets: dict[Path, Target] = {}
        leaves: dict = {}
        keys = jax.random.split(request_key(key), len(bound))
        for name, factor_key in zip(sorted(bound), keys, strict=True):
            factors = _factors(name, bound[name], variables, self.rank)
            shape_a, shape_b = factors.shapes(self.rank)
            targets[factors.module] = Target(self.rank, scaling)
            _insert_factors(leaves, factors.module, INIT_A(factor_key, shape_a, jnp.float32),
                            INIT_B(factor_key, shape_b, jnp.float32))
        return Adapter.bound(model, overlay(variables, leaves), targets, self.rslora, self.dropout, bound)

    @staticmethod
    def load(model: nn.Module, variables: Variables, path: str | FilePath, *,
             layouts: Mapping[str, WeightLayout] | None = None) -> Adapter:
        """Return the adapter a PEFT directory or Diffusers file holds, attached to `model`.

        The adapter goes over `variables` with its factors in place. `path` is a PEFT
        adapter directory or a Diffusers file (or the directory that holds one), and
        `layouts` is read as in `apply`. Targets the model does not have, tensors
        whose shapes do not fit the bound weight, ranks that disagree with the config,
        and PEFT features this loader does not support are refused by name.
        """
        if isinstance(type(model), _Adapted):
            raise ValueError("The model is already adapted")
        variables = thaw(variables)
        bound = bound_layouts(model, variables, layouts or {})
        path = FilePath(path)
        if (path / PEFT_WEIGHTS).is_file():
            where = str(path / PEFT_CONFIG)
            config = _Config.read(json.loads(FilePath(where).read_text()), where)
            tensors, _ = read_file(path / PEFT_WEIGHTS)
            return _place(model, bound, variables, _entries(_components(bound), tensors, {"": config},
                                                            PEFT_PREFIX))
        file = path if path.is_file() else path / DIFFUSERS_WEIGHTS
        if not file.is_file():
            raise FileNotFoundError(f"{path} holds neither {PEFT_WEIGHTS} nor {DIFFUSERS_WEIGHTS}")
        tensors, metadata = read_file(file)
        configs = _diffusers_configs(tensors, metadata.get(DIFFUSERS_METADATA), str(file))
        return _place(model, bound, variables, _entries(_components(bound), tensors, configs, ""))


@dataclass(frozen=True, eq=False)
class Adapter:
    """A low-rank adapter bound to one model, as `LoRA.apply` and `LoRA.load` return it.

    Two adapters compare equal only if they are the same object; compare their
    `targets` and `layouts` to compare what they bind.

    `model` computes the adapter branch in every target module and adds `FROZEN`
    back into `params` itself, so any objective can apply it to the split tree.
    `variables` holds the factors under `params` and every base weight under
    `FROZEN`, so the optimizer updates only the factors. `targets` lists the
    adapted modules' paths with their rank and alpha. `layouts` holds the
    bindings, keyed by the name a file stores each module under, that `save`
    writes the factors back through.
    """

    model: nn.Module
    variables: Variables
    targets: Mapping[Path, Target]
    rslora: bool = False
    dropout: float = 0.0
    layouts: Mapping[str, WeightLayout] = dataclasses.field(default_factory=dict)

    @classmethod
    def bound(cls, model: nn.Module, variables: Variables, targets: Mapping[Path, Target], rslora: bool,
              dropout: float, layouts: Mapping[str, WeightLayout]) -> Adapter:
        """Adapt `model` to `targets` and split `variables` into factors and frozen weights.

        The factors go under `params` and everything else under `FROZEN`.
        """
        adapted = _adapted(model, _Branch(targets, rslora, dropout), layouts)
        factors = {(*path, factor) for path in targets for factor in FACTORS}
        return cls(adapted, freeze(thaw(variables), lambda path: path in factors), targets, rslora, dropout,
                   layouts)

    @classmethod
    def from_run(cls, directory: str | FilePath, *, step: int | str | None = None,
                 ema: bool | None = None, trust: Sequence[str] = ()) -> Adapter:
        """Return the adapter a run trained, rebuilt from the run's own record and checkpoint.

        It holds the adapted model, the checkpoint's variables and the bindings the
        run recorded, so `save` writes the factors under the source's names without
        the source at hand. A recorded binding whose shapes do not fit the restored
        factors is refused by name. `trust` is as `TextGeneration.from_run` takes it.
        """
        from dew.config import ModelConfig
        from dew.inference.tasks import run_record
        from dew.records import record, text
        from dew.registry import objectives

        declaration = run_record(str(directory), step, trust)
        config = ModelConfig.from_dict(record(declaration['model'], 'model'))
        if config.adapter is None:
            raise ValueError(f"{directory} trained no adapter")
        variables = objectives[text(declaration['objective'], 'objective')]._saved_variables(
            str(directory), step=step, ema=ema)
        return cls.recorded(config.build(), variables, config.adapter)

    @classmethod
    def recorded(cls, model: nn.Module, variables: Variables, adapter: Mapping) -> Adapter:
        """Return the adapter a model record's `adapter` describes.

        It goes over the adapted `model` and a checkpoint's `variables`.
        """
        if not isinstance(type(model), _Adapted):
            raise ValueError(f"{type(model).__name__} is not adapted, so it carries no adapter")
        targets, rslora, dropout, layouts = _read_record(adapter)
        whole = thaw(variables)
        for name, layout in layouts.items():
            module = layout.paths[0][:-1]
            rank = targets[module].rank
            try:
                expected = _factors(name, layout, whole, rank).shapes(rank)
            except (KeyError, ValueError) as error:
                raise ValueError(f"the recorded binding {name} does not fit the checkpoint: "
                                 f"{error}") from None
            node = _node(whole, module)
            found = tuple(tuple(np.shape(node[factor])) for factor in FACTORS)
            if found != expected:
                raise ValueError(f"the recorded binding {name} expects factors {expected}, and the "
                                 f"checkpoint holds {found}")
        return cls(model, variables, targets, rslora, dropout, layouts)

    def scale(self, target: Target) -> float:
        return _scale(target, self.rslora)

    def merge(self, variables: Variables) -> Variables:
        """Return `variables` with every delta added into its kernel and the factors removed.

        `variables` is the adapted tree, whole or split as a trainer keeps it. The sum
        runs in at least fp32 at full precision and is stored in the kernel's dtype,
        as PEFT's `merge_and_unload` does. The result is one `params` collection.
        """
        variables = thaw(variables)
        merged: dict = {}
        for path, target in self.targets.items():
            node = _node(variables, path)
            kernel, a, b = node["kernel"], node["lora_A"], node["lora_B"]
            dtype = jnp.promote_types(kernel.dtype, jnp.float32)
            delta = jnp.tensordot(a.astype(dtype), b.astype(dtype), axes=1,
                                  precision=jax.lax.Precision.HIGHEST)
            _insert(
                merged,
                (*path, "kernel"),
                (kernel.astype(dtype) + self.scale(target) * delta).astype(kernel.dtype),
            )
        factors = {(*path, factor) for path in self.targets for factor in FACTORS}
        return select(overlay(variables, merged), lambda path: path not in factors)

    def save(self, variables: Variables, path: str | FilePath) -> None:
        """Write the adapter's factors from `variables` under its own module names.

        The names are the ones this adapter bound, so a run saves what it trained from
        the tree it trained, split or whole. With a single unnamed component (a
        decoder source, or a model built from its class), it writes PEFT's directory.
        A pipeline source, whose weights are named under several components, gets the
        Diffusers file with each component's PEFT config in its header.
        """
        variables = thaw(variables)
        components = _components(self.layouts)
        names = {layout.paths[0][:-1]: name for name, layout in self.layouts.items()
                 if layout.paths[0][-1] == "kernel"}
        tensors: dict[str, np.ndarray] = {}
        named: dict[str, dict[str, Target]] = {}
        for module, target in self.targets.items():
            name = names.get(module)
            if name is None:
                raise ValueError(f"{'/'.join(module)} is not a projection this adapter bound")
            component, relative = _component(components, name)
            factors = _factors(name, self.layouts[name], variables, target.rank)
            for factor in (factors.a, factors.b):
                tensors[factor.name] = factor.export(variables)
            named.setdefault(component, {})[relative] = target
        path = FilePath(path)
        path.mkdir(parents=True, exist_ok=True)
        if not components:
            (path / PEFT_CONFIG).write_text(json.dumps(_config(self, named[""]), indent=2) + "\n")
            write_file({PEFT_PREFIX + name: tensor for name, tensor in tensors.items()},
                       path / PEFT_WEIGHTS, {"format": "pt"})
            return
        metadata = {f"{component}.{field}": value
                    for component, targets in sorted(named.items())
                    for field, value in _config(self, targets).items()}
        write_file(tensors, path / DIFFUSERS_WEIGHTS,
                   {"format": "pt", DIFFUSERS_METADATA: json.dumps(metadata, indent=2, sort_keys=True)})


def _scale(target: Target, rslora: bool) -> float:
    return target.alpha / (math.sqrt(target.rank) if rslora else target.rank)


def _target_at(targets: Mapping[Path, Target], path: Path) -> Target | None:
    """Return the target a module path names, seen through the stack that runs
    it: a scanned run's module `layers_3_7` stands for layers 3 through 7,
    whose targets must agree, since the run's kernels share one stacked
    factor pair."""
    if path in targets:
        return targets[path]
    for depth, name in enumerate(path):
        layers = group_layers(name)
        if layers is None or len(layers) == 1:
            continue
        found = {targets.get((*path[:depth], f"layers_{index}", *path[depth + 1:])) for index in layers}
        if found == {None}:
            return None
        if len(found) != 1:
            raise ValueError(f"{'/'.join(path)} runs layers whose adapter targets differ: "
                             f"a scanned run stacks one factor pair over all of them")
        return found.pop()
    return None


@dataclass(frozen=True)
class _Branch:
    """The interceptor an adapted model runs under: the adapter branch on
    every target module's `__call__`."""

    targets: Mapping[Path, Target]
    rslora: bool
    dropout: float
    root: Path = ("params",)

    def __call__(self, next_fun, args, kwargs, context):
        module = context.module
        root = self.root
        if context.method_name == "stochastic_input":
            # A layer asks whether its submodule's input is drawn on this
            # call (`MultiHeadLatentAttention.stochastic_input`): the
            # branch's dropout is, on a target under a dropout stream.
            name = args[0] if args else kwargs["name"]
            return next_fun(*args, **kwargs) or bool(
                self.dropout and module.has_rng("dropout")
                and _target_at(self.targets, root + module.path + (name,)) is not None)
        target = _target_at(self.targets, root + module.path) if context.method_name == "__call__" else None
        if target is None:
            return next_fun(*args, **kwargs)
        if type(module) not in (nn.Dense, nn.DenseGeneral):
            raise TypeError(
                f"{'/'.join(root + module.path)} is a {type(module).__name__}; an adapter "
                "targets nn.Dense and nn.DenseGeneral kernels")
        x = args[0] if args else kwargs["inputs"]
        output = next_fun(*args, **kwargs)
        if isinstance(module, nn.DenseGeneral):
            if module.batch_dims:
                raise TypeError(f"{'/'.join(root + module.path)} contracts with batch dims")
            axes = (module.axis,) if isinstance(module.axis, int) else tuple(module.axis)
        else:
            axes = (-1,)
        axes = tuple(sorted(axis % x.ndim for axis in axes))
        kernel = module.get_variable("params", "kernel")
        a = module.param("lora_A", INIT_A, (*kernel.shape[:len(axes)], target.rank), module.param_dtype)
        b = module.param("lora_B", INIT_B, (target.rank, *kernel.shape[len(axes):]), module.param_dtype)
        # The branch drops out its input when the forward carries the
        # dropout stream, which is how a training forward is marked; an
        # evaluation forward carries none.
        if self.dropout and module.has_rng("dropout"):
            x = nn.Dropout(self.dropout, deterministic=False)(x)
        x, a, b = promote_dtype(x, a, b, dtype=module.dtype)
        hidden = jax.lax.dot_general(x, a, ((axes, tuple(range(len(axes)))), ((), ())),
                                     precision=module.precision)
        delta = jax.lax.dot_general(hidden, b, (((hidden.ndim - 1,), (0,)), ((), ())),
                                    precision=module.precision)
        return output + jnp.asarray(_scale(target, self.rslora), delta.dtype) * delta


def _adapted(model: nn.Module, branch: _Branch, layouts: Mapping[str, WeightLayout]) -> nn.Module:
    """Return `model` computing `branch` in every target module.

    The result is an instance of a subclass of `model`'s class with the
    same fields and methods, so it builds, checks and generates as the model
    does; only `apply` and `init` change. `apply` folds `FROZEN` back into
    `params` and runs under the interceptor; `init` (Flax's dispatches
    through `init_with_output`) runs under it and splits what it draws, the
    factors under `params` and the rest under `FROZEN`. `layouts` are the
    bindings the class's record (`ModelConfig.adapter`) writes beside the
    targets; the record is written when a run asks for it, so an adapter a
    run cannot record (a file's mixed ranks) still loads and computes.
    """
    base = type(model)
    if isinstance(base, _Adapted):
        raise ValueError("The model is already adapted")
    factors = {(*path, factor) for path in branch.targets for factor in FACTORS}

    class Adapted(base):
        # A staticmethod, or Flax would wrap the callable as a module method.
        _dew_lora_interceptor: ClassVar[Interceptor] = staticmethod(branch)
        _dew_lora_base: ClassVar[type[nn.Module]] = base

        @classmethod
        def _dew_lora_record(cls) -> dict:
            return _record(branch.targets, branch.rslora, branch.dropout, layouts)

        def apply(self, variables, *args, **kwargs):
            with nn.intercept_methods(self._dew_lora_interceptor):
                return super().apply(thaw(variables), *args, **kwargs)

        def init_with_output(self, *args, **kwargs):
            with nn.intercept_methods(self._dew_lora_interceptor):
                output, variables = super().init_with_output(*args, **kwargs)
            return output, dict(freeze(variables, lambda path: path in factors))

    Adapted.__name__ = Adapted.__qualname__ = base.__name__
    return Adapted(**{field.name: getattr(model, field.name)
                      for field in dataclasses.fields(model)
                      if field.init and field.name not in ("parent", "name")})


def _record(targets: Mapping[Path, Target], rslora: bool, dropout: float,
            layouts: Mapping[str, WeightLayout]) -> dict:
    """The bound adapter as a model record writes it: one rank and alpha, the
    targets' module paths, and each target's binding (the name a file writes
    it under, the weight's stored shape and transpose), so the run's adapter
    saves with no source at hand. A mixed-rank adapter is refused."""
    ranks = {target.rank for target in targets.values()}
    alphas = {target.alpha for target in targets.values()}
    if len(ranks) != 1 or len(alphas) != 1:
        raise ValueError("a recorded LoRA requires one rank and alpha across its modules")
    bindings = {layout.paths[0][:-1]: layout for layout in layouts.values()}

    def binding(layout: WeightLayout) -> dict:
        transpose = layout.transpose
        return {'name': layout.name, 'shape': list(layout.shape),
                'transpose': None if transpose is None else list(transpose)}

    return {'rank': next(iter(ranks)), 'alpha': next(iter(alphas)), 'rslora': rslora, 'dropout': dropout,
            'modules': ['/'.join(path) for path in sorted(targets)],
            'layouts': {'/'.join(path): binding(bindings[path])
                        for path in sorted(targets) if path in bindings}}


def _read_record(record: Mapping) -> tuple[dict[Path, Target], bool, float, dict[str, WeightLayout]]:
    """Read back what `_record` wrote."""
    from dew.records import integer, text

    if set(record) != {'rank', 'alpha', 'rslora', 'dropout', 'modules', 'layouts'}:
        raise ValueError("a LoRA record needs rank, alpha, rslora, dropout, modules and layouts")
    rank = integer(record['rank'], 'adapter rank')
    alpha = record['alpha']
    if rank < 1 or isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha):
        raise ValueError("adapter rank must be positive and alpha must be a finite number")
    if type(record['rslora']) is not bool:
        raise ValueError("adapter rslora must be boolean")
    dropout = record['dropout']
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
        raise ValueError("adapter dropout must be in [0, 1)")
    modules = record['modules']
    if not isinstance(modules, list) or not modules:
        raise ValueError("adapter modules must be a nonempty list of native module names")
    targets: dict[Path, Target] = {}
    for module in modules:
        path = tuple(text(module, 'adapter module').split('/'))
        if any(not name or name in ('.', '..') for name in path) or path in targets:
            raise ValueError("adapter modules must be distinct native module names")
        targets[path] = Target(rank, float(alpha))
    bindings = record['layouts']
    if not isinstance(bindings, dict) or not set(bindings) <= set(modules):
        raise ValueError("adapter layouts must be keyed by the adapter's own modules")
    layouts: dict[str, WeightLayout] = {}
    for module, binding in bindings.items():
        if not isinstance(binding, dict) or set(binding) != {'name', 'shape', 'transpose'}:
            raise ValueError(f"the adapter's binding of {module} needs name, shape and transpose")
        name = text(binding['name'], f'binding of {module}')
        shape = tuple(integer(size, f'binding of {module}') for size in binding['shape'])
        transpose = binding['transpose']
        layout = WeightLayout(name, ((*module.split('/'), 'kernel'),), shape,
                              None if transpose is None else tuple(integer(axis, f'binding of {module}')
                                                                   for axis in transpose))
        layouts[name.removesuffix('.weight').replace('/', '.')] = layout
    return targets, record['rslora'], float(dropout), layouts


def adapted(model: nn.Module, record: Mapping) -> nn.Module:
    """`model` adapted as the model record `record` says, which is how a
    recorded run's model is rebuilt (`ModelConfig.build`); the checkpoint
    supplies the factors."""
    targets, rslora, dropout, layouts = _read_record(record)
    return _adapted(model, _Branch(targets, rslora, dropout), layouts)


def _node(tree: Mapping, path: Path) -> Mapping:
    node = tree
    for name in path:
        node = node[name]
    return node


def _insert(tree: dict, path: Path, value) -> None:
    for name in path[:-1]:
        tree = tree.setdefault(name, {})
    tree[path[-1]] = value



# --------------------------------------------------------------------------
# Source modules and their factor layouts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Factors:
    """Holds the source-layout bindings of one target's factors.

    `a` stores `[r, in]` and `b` `[out, r]`, PEFT's `lora_A.weight` and
    `lora_B.weight`, as transposed views of the tree's `[in..., r]` and
    `[r, out...]` leaves, derived from how the kernel itself is stored.
    `contracted` is how many leading kernel axes are the input.
    """

    module: Path
    kernel_shape: tuple[int, ...]
    contracted: int
    a: WeightLayout
    b: WeightLayout

    def shapes(self, rank: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return the tree's `lora_A` and `lora_B` shapes at `rank`."""
        return (*self.kernel_shape[:self.contracted], rank), (rank, *self.kernel_shape[self.contracted:])


def _insert_factors(leaves: dict, module: Path, a, b) -> None:
    """Write one target's two factor leaves into `leaves`, under `FACTORS`."""
    for name, value in zip(FACTORS, (a, b), strict=True):
        _insert(leaves, (*module, name), value)


def _factors(name: str, layout: WeightLayout, variables: Variables, rank: int) -> _Factors:
    """Return the factor layouts of the kernel `layout` binds in `variables`."""
    if layout.paths[0][-1] != "kernel":
        raise ValueError(f"{name} is not a projection weight, so it takes no low-rank delta")
    if len(layout.paths) != 1 or layout.concatenate is not None or layout.expert_index is not None:
        raise ValueError(
            f"{name} is assembled from several leaves of the model, and a low-rank "
            "delta on the assembled tensor does not split into one per leaf")
    if len(layout.shape) != 2:
        raise ValueError(f"{name} stores {layout.shape}, not a [out, in] matrix")
    kernel_shape = _node(variables, layout.paths[0][:-1])["kernel"].shape
    out, inner = layout.shape
    transpose = layout.transpose or tuple(range(len(kernel_shape)))
    stored = tuple(kernel_shape[axis] for axis in transpose)
    split = next((k for k in range(len(stored) + 1)
                  if math.prod(stored[:k]) == out and math.prod(stored[k:]) == inner), None)
    contracted = len(stored) - (split or 0)
    if (split is None or sorted(transpose[split:]) != list(range(contracted))
            or sorted(transpose[:split]) != list(range(contracted, len(stored)))):
        raise ValueError(
            f"{name} stores [out, in] {layout.shape} from a kernel of {kernel_shape} "
            f"transposed {transpose}, whose input axes do not lead the kernel")
    module = layout.paths[0][:-1]
    prefix = name.removesuffix(".weight")
    return _Factors(
        module, kernel_shape, contracted,
        WeightLayout(f"{prefix}.lora_A.weight", ((*module, "lora_A"),), (rank, inner),
                     (contracted, *transpose[split:])),
        WeightLayout(f"{prefix}.lora_B.weight", ((*module, "lora_B"),), (out, rank),
                     (*(1 + axis - contracted for axis in transpose[:split]), 0)))


def bound_layouts(model: nn.Module, variables: Variables,
                  layouts: Mapping[str, WeightLayout]) -> Mapping[str, WeightLayout]:
    """Return the projections an adapter can bind on `model`, by the name a file uses.

    A loaded source publishes `Pretrained.layouts`, the bindings its export
    runs backwards, and those names are the ones PEFT and Diffusers write.
    A model built from the registry has no published file, so its own module
    path under `params` is the name and its kernel in `variables` is the
    shape; that is the mapping this derives when none is given.

    A derived layout covers a two-axis kernel, which stores `[in, out]` and
    exports as PEFT's `[out, in]`. A kernel of more axes splits into
    contracted and feature axes that the module decides and the tree does
    not record, so it carries no derived name and a target that asks for it
    is refused as unbound.
    """
    if layouts:
        return layouts
    derived = {".".join(module): _kernel_layout(module, kernel)
               for module, kernel in _kernels(variables.get("params", {}), ())
               if kernel.ndim == 2}
    if not derived:
        raise ValueError(
            f"this {type(model).__name__}'s variables hold no two-axis kernel to adapt; "
            f"pass the layouts of a loaded source, or train a model with Dense projections")
    return derived


def _kernels(node: Mapping, path: Path) -> Iterator[tuple[Path, np.ndarray | jax.Array]]:
    """Yield every `kernel` leaf under `node`, with the module path that holds it."""
    for name, child in node.items():
        if not isinstance(child, Mapping):
            continue
        kernel = child.get("kernel")
        if isinstance(kernel, (np.ndarray, jax.Array)):
            yield (*path, name), kernel
        yield from _kernels(child, (*path, name))


def _kernel_layout(module: Path, kernel: np.ndarray | jax.Array) -> WeightLayout:
    """Return one `[in, out]` kernel as the `[out, in]` weight a PEFT file names."""
    inner, out = kernel.shape
    return WeightLayout(f"{'.'.join(module)}.weight", (("params", *module, "kernel"),),
                        (out, inner), (1, 0))


def _components(layouts: Mapping[str, WeightLayout]) -> frozenset[str]:
    """Return the named components these layouts bind.

    A pipeline source names each weight under the component that holds it,
    `unet/down_blocks...`; a decoder and a registry-built model name theirs
    under the model's own root, which is the one unnamed component the
    reference injects into and matches its patterns against.
    """
    return frozenset(layout.name.partition("/")[0]
                     for layout in layouts.values() if "/" in layout.name)


def _component(components: frozenset[str], module: str) -> tuple[str, str]:
    """Split one module name of a file into (component, name relative to it)."""
    component, _, relative = module.partition(".")
    return (component, relative) if component in components else ("", module)


# --------------------------------------------------------------------------
# PEFT configuration
# --------------------------------------------------------------------------

_REFUSED_FLAGS = ("fan_in_fan_out", "use_dora", "lora_bias", "modules_to_save",
                  "trainable_token_indices", "target_parameters")


@dataclass(frozen=True)
class _Config:
    """Holds the fields of a PEFT `LoraConfig` an adapter's numerics depend on."""

    rank: int
    alpha: float
    rank_pattern: Mapping[str, int]
    alpha_pattern: Mapping[str, float]
    rslora: bool
    dropout: float

    @classmethod
    def read(cls, config: Mapping[str, object], where: str) -> _Config:
        if config.get("peft_type", "LORA") != "LORA":
            raise ValueError(f"{where} is a {config['peft_type']} adapter, not LoRA")
        refused = [name for name in _REFUSED_FLAGS if config.get(name)]
        if config.get("bias", "none") != "none":
            refused.append("bias")
        if refused:
            raise ValueError(f"{where} sets {', '.join(refused)}, which this loader does not carry")
        rank, alpha = config.get("r"), config.get("lora_alpha")
        rank_pattern, alpha_pattern = config.get("rank_pattern", {}), config.get("alpha_pattern", {})
        rslora, dropout = config.get("use_rslora", False), config.get("lora_dropout", 0.0)
        if (type(rank) is not int or not isinstance(alpha, (int, float))
                or not isinstance(rank_pattern, Mapping) or not isinstance(alpha_pattern, Mapping)
                or not isinstance(rslora, bool) or not isinstance(dropout, (int, float))
                or any(type(value) is not int for value in rank_pattern.values())
                or any(not isinstance(value, (int, float)) for value in alpha_pattern.values())):
            raise ValueError(f"{where} needs an integer r, numeric lora_alpha and per-module patterns")
        return cls(rank, float(alpha), dict(rank_pattern),
                   {key: float(value) for key, value in alpha_pattern.items()}, rslora, float(dropout))

    def rank_of(self, relative: str) -> int:
        return self.rank_pattern.get(_pattern(self.rank_pattern, relative), self.rank)

    def alpha_of(self, relative: str) -> float:
        return self.alpha_pattern.get(_pattern(self.alpha_pattern, relative), self.alpha)


def _pattern(patterns: Mapping[str, object], relative: str) -> str:
    """Return PEFT's `get_pattern_key`: the first pattern that names this module,
    else the name itself, which no pattern table holds."""
    return next((key for key in patterns if re.match(rf"(.*\.)?({key})$", relative)), relative)


class PeftConfig(TypedDict):
    """The contents of `adapter_config.json`, as PEFT writes and reads it.

    It holds the defaults every target uses and the per-module exceptions.
    `fan_in_fan_out`, `bias`, `init_lora_weights` and `inference_mode` are fixed
    because Dew builds its adapters one way. Nothing here reads those four back;
    they are written so that a PEFT reader finds the keys it expects.
    """

    peft_type: str
    r: int
    lora_alpha: float
    rank_pattern: dict[str, int]
    alpha_pattern: dict[str, float]
    target_modules: list[str]
    use_rslora: bool
    lora_dropout: float
    fan_in_fan_out: bool
    bias: str
    init_lora_weights: bool
    inference_mode: bool


def _config(lora: Adapter, named: Mapping[str, Target]) -> PeftConfig:
    """Build the PEFT config of `named` targets, keyed by the relative module name.

    The commonest rank and alpha become the defaults and the rest the patterns.
    """
    ranks = [target.rank for target in named.values()]
    alphas = [target.alpha for target in named.values()]
    rank = max(set(ranks), key=ranks.count)
    alpha = max(set(alphas), key=alphas.count)
    return {
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": alpha,
        "rank_pattern": {name: target.rank for name, target in named.items() if target.rank != rank},
        "alpha_pattern": {name: target.alpha for name, target in named.items() if target.alpha != alpha},
        "target_modules": sorted(named),
        "use_rslora": lora.rslora,
        "lora_dropout": lora.dropout,
        "fan_in_fan_out": False,
        "bias": "none",
        "init_lora_weights": True,
        "inference_mode": True,
    }


# --------------------------------------------------------------------------
# What LoRA.apply, LoRA.load and Adapter.save read and write the files with
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Entry:
    """Holds one adapted source module as a file names it, with its component's config."""

    module: str
    a: np.ndarray
    b: np.ndarray
    config: _Config


def _entries(components: frozenset[str], tensors: Mapping[str, np.ndarray],
             configs: Mapping[str, _Config], prefix: str) -> list[_Entry]:
    """Return every module's factor pair. `configs` is keyed by component."""
    factors: dict[str, dict[str, np.ndarray]] = {}
    for key, tensor in tensors.items():
        module, _, factor = key.removeprefix(prefix).rpartition(".lora_")
        if not key.startswith(prefix) or not module or factor not in ("A.weight", "B.weight"):
            raise ValueError(
                f"{key} is not a LoRA tensor ({prefix}<module>.lora_A.weight or lora_B.weight); "
                "kohya/sgm keys and embedding adapters are not accepted")
        factors.setdefault(module, {})[factor[0]] = tensor
    entries = []
    for module, pair in factors.items():
        if pair.keys() != {"A", "B"}:
            raise ValueError(f"{module} has lora_{next(iter(pair))} without its partner")
        component = _component(components, module)[0]
        if component not in configs:
            raise ValueError(f"the file carries no config for {module}'s component {component!r}")
        entries.append(_Entry(module, pair["A"], pair["B"], configs[component]))
    return entries


def _place(model: nn.Module, layouts: Mapping[str, WeightLayout], variables: Variables,
           entries: Sequence[_Entry]) -> Adapter:
    """Build the adapter the entries describe, its factors restored into the model's tree."""
    components = _components(layouts)
    settings = {(entry.config.rslora, entry.config.dropout) for entry in entries}
    if len(settings) != 1:
        raise ValueError(
            "the components disagree on use_rslora or lora_dropout, which one adapter carries once"
        )
    (rslora, dropout), = settings
    targets: dict[Path, Target] = {}
    bound: dict[str, WeightLayout] = {}
    leaves: dict = {}
    for entry in entries:
        layout = layouts.get(entry.module)
        if layout is None:
            raise ValueError(f"the adapter targets {entry.module}, which this source does not bind")
        if entry.a.ndim != 2 or entry.b.ndim != 2 or entry.a.shape[0] != entry.b.shape[1]:
            raise ValueError(
                f"{entry.module} carries lora_A {entry.a.shape} and lora_B {entry.b.shape}, "
                "not [r, in] and [out, r] of one rank")
        rank = entry.a.shape[0]
        relative = _component(components, entry.module)[1]
        declared = entry.config.rank_of(relative)
        if declared != rank:
            raise ValueError(f"{entry.module} stores rank {rank} but its config declares {declared}")
        factors = _factors(entry.module, layout, variables, rank)
        if (entry.b.shape[0], entry.a.shape[1]) != layout.shape:
            raise ValueError(
                f"{entry.module} carries a delta of {(entry.b.shape[0], entry.a.shape[1])} "
                f"on a weight the source stores as {layout.shape}")
        shape_a, shape_b = factors.shapes(rank)
        targets[factors.module] = Target(rank, entry.config.alpha_of(relative))
        bound[entry.module] = layout
        _insert_factors(leaves, factors.module, factors.a.restore(entry.a, shape_a),
                        factors.b.restore(entry.b, shape_b))
    return Adapter.bound(model, overlay(thaw(variables), leaves), targets, rslora, dropout, bound)


def _diffusers_configs(tensors: Mapping[str, np.ndarray], metadata: str | None,
                       where: str) -> dict[str, _Config]:
    """Read per-component PEFT configs from the header, or what Diffusers derives
    without one (utils.peft_utils.get_peft_kwargs): every module's rank from
    its tensors, and one alpha for the component, the rank of its first
    module in file order."""
    if metadata is not None:
        fields: dict[str, dict[str, object]] = {}
        for key, value in json.loads(metadata).items():
            component, _, field = key.partition(".")
            fields.setdefault(component, {})[field] = value
        return {
            component: _Config.read(config, f"{where} ({component})") for component, config in fields.items()
        }
    ranks: dict[str, dict[str, int]] = {}
    for key, tensor in tensors.items():
        component, _, rest = key.partition(".")
        module, _, factor = rest.rpartition(".lora_")
        if factor == "B.weight":
            ranks.setdefault(component, {})[module] = tensor.shape[1]
    return {component: _Config(next(iter(found.values())), float(next(iter(found.values()))),
                               found, {}, rslora=False, dropout=0.0)
            for component, found in ranks.items()}


def _named(layouts: Mapping[str, WeightLayout], wanted: Sequence[str]) -> dict[str, WeightLayout]:
    """Return the projections PEFT's `target_modules` selects: a name relative to
    the model that is an entry, or ends in `.` and an entry."""
    components = _components(layouts)
    matched: dict[str, WeightLayout] = {}
    hit = set()
    for name, layout in layouts.items():
        relative = _component(components, name)[1]
        entries = {entry for entry in wanted if relative == entry or relative.endswith("." + entry)}
        if entries and layout.paths[0][-1] == "kernel":
            matched[name] = layout
            hit |= entries
    missed = [entry for entry in wanted if entry not in hit]
    if missed:
        raise ValueError(f"{', '.join(missed)} match no projection of this source")
    return matched


@runtime_checkable
class _Adapted(Protocol):
    """Marks a module class an adapter already wrapped.

    The wrapper subclass declares the interceptor its `apply` and `init` run
    under, which is the whole record that a class was adapted: a second
    adapter over it would run one branch inside the other.
    """

    _dew_lora_interceptor: ClassVar[Interceptor]
    _dew_lora_base: ClassVar[type[nn.Module]]

    @classmethod
    def _dew_lora_record(cls) -> dict: ...


__all__ = ["Adapter", "LoRA", "PeftConfig", "Target"]
