"""Adapt a variables tree with low-rank deltas (LoRA, arXiv 2106.09685).

An adapter is a set of rank-`r` deltas on the kernels of Dense modules.
Beside a target kernel `W`, `[in..., out...]`, the tree holds `lora_A`,
`[in..., r]`, and `lora_B`, `[r, out...]`, and the module computes
`x W + scale * (x A) B` with `scale = alpha / r` (`alpha / sqrt(r)` for
rsLoRA, arXiv 2312.03732). Merged, `scale * A B` is added into the kernel
and the factors are gone. A target is the path of its module's leaves in
the tree, `("params", "layers_0", "self_attn", "q_proj")`, so one
description serves every model and every collection.

One object carries all of it. `LoRA.fresh` draws an adapter on the
projections a model binds, `LoRA.load` reads one off disk onto them, and the
adapter that comes back adapts the model (`adapt`), says what trains
(`trainable`), folds itself in (`merge`) and writes itself out (`save`),
since it holds the bindings it was built over.

`LoRA.adapt` makes a model compute the branch without a module of its own
for every projection: it wraps `apply` and `init` in a Flax method
interceptor that adds the branch to each target `Dense`'s output and reads
or creates the factors as that module's own parameters. What it hides is
the per-module bookkeeping PEFT does with wrapper layers: the factor shapes
of a `DenseGeneral` with several contracted axes, the compute dtype the
kernel path uses, dropout on the branch input, and the parameter naming
the merge, the export and the trainable filter agree on.

Two files are read and written, both what the references produce natively.
PEFT's directory (`adapter_config.json`, `adapter_model.safetensors`, keys
`base_model.model.<module>.lora_A.weight`) is what Transformers loads; the
Diffusers file (`pytorch_lora_weights.safetensors`, keys
`<component>.<module>.lora_A.weight`, the PEFT config per component in the
header's `lora_adapter_metadata`) is what a pipeline's `load_lora_weights`
reads. Kohya/sgm keys (`lora_unet_...`, `.alpha` tensors) are not accepted.
Source module names resolve to tree paths through `Pretrained.layouts`, the
bindings a source's export runs backwards, so an adapter is placed exactly
where the base tensor it modifies went. An adapter attaches to a model and
its variables, not to a loader: a model built from the registry passes no
layouts and `bound_layouts` reads the names and shapes off its own kernels,
so a run that never touched a published checkpoint adapts the same way.
`RunConfig.lora` is that path from a config: the run adapts the module its
objective trains and freezes everything but the factors.
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

from dew.interop.pretrained import WeightLayout
from dew.interop.safetensors_io import read_file, write_file
from dew.nn.backbones.causal_transformer import group_layers
from dew.objectives.base import Path, PathFilter, Variables, merge as overlay, select

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
    """Holds one adapted kernel's rank and alpha."""

    rank: int
    alpha: float


@dataclass(frozen=True)
class LoRA:
    """Says which kernels carry a low-rank delta, and how the delta scales."""

    targets: Mapping[Path, Target]
    rslora: bool = False
    dropout: float = 0.0
    layouts: Mapping[str, WeightLayout] = dataclasses.field(
        default_factory=dict, compare=False, metadata={"record": False})
    """The bindings the targets were bound over, by the name a file writes
    them under, which is what `save` writes the factors back through.
    `fresh` and `load` fill it with the source's own projections; an adapter
    a run config declares carries its targets alone, so it compares by them
    and cannot save itself. A run record leaves it out: the source the
    record names is where the bindings come from."""

    def scale(self, target: Target) -> float:
        return target.alpha / (math.sqrt(target.rank) if self.rslora else target.rank)

    def trainable(self, path: Path) -> bool:
        """Return the adapter's own leaves: the `PathFilter` a partial run trains."""
        return path[-1] in FACTORS and path[:-1] in self.targets

    def target_at(self, path: Path) -> Target | None:
        """Return the target a module path names, seen through the stack that runs
        it: a scanned run's module `layers_3_7` stands for layers 3 through
        7, whose targets must agree, since the run's kernels share one
        stacked factor pair."""
        if path in self.targets:
            return self.targets[path]
        for depth, name in enumerate(path):
            layers = group_layers(name)
            if layers is None or len(layers) == 1:
                continue
            found = {self.targets.get((*path[:depth], f"layers_{index}", *path[depth + 1:]))
                     for index in layers}
            if found == {None}:
                return None
            if len(found) != 1:
                raise ValueError(f"{'/'.join(path)} runs layers whose adapter targets differ: "
                                 f"a scanned run stacks one factor pair over all of them")
            return found.pop()
        return None

    def adapt(self, model: nn.Module, root: Path = ("params",)) -> nn.Module:
        """Return `model` computing the adapter branch in every target module.

        The result is an instance of a subclass of `model`'s class with the
        same fields and methods, so it builds, checks and generates as the
        model does; only `apply` and `init` change, running under the
        interceptor (Flax's `init` dispatches through `init_with_output`).
        Adapt the module that is applied: a target inside a submodule is
        reached through its parent's `apply`. `root` is where the model's
        own `params` sit in the tree the targets are named in: the whole
        tree's `params` for a model applied on it, deeper for a tower
        applied on a subtree.
        """
        base = type(model)
        if isinstance(base, _Adapted):
            raise ValueError("The model is already adapted")
        branch = self._branch(root)

        class Adapted(base):
            _dew_lora_interceptor: ClassVar[Interceptor] = staticmethod(branch)

            def apply(self, *args, **kwargs):
                with nn.intercept_methods(self._dew_lora_interceptor):
                    return super().apply(*args, **kwargs)

            def init_with_output(self, *args, **kwargs):
                with nn.intercept_methods(self._dew_lora_interceptor):
                    return super().init_with_output(*args, **kwargs)

        Adapted.__name__ = Adapted.__qualname__ = base.__name__
        return Adapted(**{field.name: getattr(model, field.name)
                          for field in dataclasses.fields(model)
                          if field.init and field.name not in ("parent", "name")})

    def _branch(self, root: Path):
        def branch(next_fun, args, kwargs, context):
            module = context.module
            if context.method_name == "stochastic_input":
                # A layer asks whether its submodule's input is drawn on this
                # call (`MultiHeadLatentAttention.stochastic_input`): the
                # branch's dropout is, on a target under a dropout stream.
                name = args[0] if args else kwargs["name"]
                return next_fun(*args, **kwargs) or bool(
                    self.dropout and module.has_rng("dropout")
                    and self.target_at(root + module.path + (name,)) is not None)
            target = self.target_at(root + module.path) if context.method_name == "__call__" else None
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
            return output + jnp.asarray(self.scale(target), delta.dtype) * delta

        return branch

    def merge(self, variables: Variables) -> Variables:
        """Return `variables` with every delta added into its kernel and the factors removed.

        The sum runs in at least fp32 at full precision and lands in the
        kernel's dtype, PEFT's `merge_and_unload`.
        """
        merged: dict = {}
        for path, target in self.targets.items():
            node = _node(variables, path)
            kernel, a, b = node["kernel"], node["lora_A"], node["lora_B"]
            dtype = jnp.promote_types(kernel.dtype, jnp.float32)
            delta = jnp.tensordot(a.astype(dtype), b.astype(dtype), axes=1,
                                  precision=jax.lax.Precision.HIGHEST)
            _insert(merged, (*path, "kernel"), (kernel.astype(dtype) + self.scale(target) * delta).astype(kernel.dtype))
        return select(overlay(variables, merged), lambda path: not self.trainable(path))

    @classmethod
    def fresh(cls, model: nn.Module, variables: Variables, layouts: Mapping[str, WeightLayout], *,
              rank: int, modules: Sequence[str], key: jax.Array, alpha: float | None = None,
              rslora: bool = False, dropout: float = 0.0) -> tuple[LoRA, Variables]:
        """Build a new adapter on the projections `modules` name, and add its factors.

        `modules` are PEFT's `target_modules`: a projection matches when its
        name relative to the model (`model.layers.0.self_attn.q_proj`, or `to_q`
        under a pipeline component) is the entry or ends in `.` and the entry.
        An entry that matches no projection is refused. `alpha` unset is
        PEFT's own default for a rank, twice it. A is drawn from `key` the
        way PEFT draws it and B is zero, so the fresh adapter is the identity.
        """
        bound = _named(bound_layouts(model, variables, layouts), modules)
        scaling = 2.0 * rank if alpha is None else alpha
        targets: dict[Path, Target] = {}
        leaves: dict = {}
        for name, factor_key in zip(sorted(bound), jax.random.split(key, len(bound)), strict=True):
            factors = _factors(name, bound[name], variables, rank)
            shape_a, shape_b = factors.shapes(rank)
            targets[factors.module] = Target(rank, scaling)
            _insert_factors(leaves, factors.module, INIT_A(factor_key, shape_a, jnp.float32),
                            INIT_B(factor_key, shape_b, jnp.float32))
        return cls(targets, rslora, dropout, bound), overlay(variables, leaves)

    @classmethod
    def load(cls, model: nn.Module, variables: Variables, layouts: Mapping[str, WeightLayout],
             path: str | FilePath) -> tuple[LoRA, Variables]:
        """Load an adapter for `model` and `variables` with the factors in place.

        `layouts` are the bindings a file's module names resolve through:
        `Pretrained.layouts` for a loaded source, an empty mapping for a model
        built from the registry, whose own module paths are its names.

        `path` is a PEFT adapter directory or a Diffusers file (or the directory
        holding one). Targets the model does not bind, tensors whose shapes do
        not fit the bound weight, ranks that disagree with the config, and PEFT
        features this loader does not carry are refused by name.
        """
        bound = bound_layouts(model, variables, layouts)
        path = FilePath(path)
        if (path / PEFT_WEIGHTS).is_file():
            where = str(path / PEFT_CONFIG)
            config = _Config.read(json.loads(FilePath(where).read_text()), where)
            tensors, _ = read_file(path / PEFT_WEIGHTS)
            return _place(bound, variables, _entries(_components(bound), tensors, {"": config}, PEFT_PREFIX))
        file = path if path.is_file() else path / DIFFUSERS_WEIGHTS
        if not file.is_file():
            raise FileNotFoundError(f"{path} holds neither {PEFT_WEIGHTS} nor {DIFFUSERS_WEIGHTS}")
        tensors, metadata = read_file(file)
        configs = _diffusers_configs(tensors, metadata.get(DIFFUSERS_METADATA), str(file))
        return _place(bound, variables, _entries(_components(bound), tensors, configs, ""))

    def save(self, variables: Variables, path: str | FilePath) -> None:
        """Write the adapter's factors from `variables` under its own module names.

        The names are the ones this adapter bound at construction, so a run
        saves what it trained with the tree it trained it in. One unnamed
        component writes PEFT's directory, which is a decoder source and a
        registry-built model; a pipeline source, whose weights are named
        under several components, writes the Diffusers file with each
        component's PEFT config in its header.
        """
        if not self.layouts:
            raise ValueError(
                "this adapter binds no source names to write its factors under; "
                "LoRA.fresh and LoRA.load bind them, a declared target set does not")
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
    """Describes `adapter_config.json` as PEFT writes and reads it.

    It carries the defaults every target takes and the per-module exceptions.
    `fan_in_fan_out`, `bias`, `init_lora_weights` and `inference_mode` are
    fixed because dew builds its adapters one way. Nothing here reads those
    four back; they are written so a PEFT reader finds the keys it expects.
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


def _config(lora: LoRA, named: Mapping[str, Target]) -> PeftConfig:
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
# What LoRA.fresh, LoRA.load and LoRA.save read and write the files with
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


def _place(layouts: Mapping[str, WeightLayout], variables: Variables,
           entries: Sequence[_Entry]) -> tuple[LoRA, Variables]:
    """Build the adapter the entries describe, its factors restored into the model's tree."""
    components = _components(layouts)
    settings = {(entry.config.rslora, entry.config.dropout) for entry in entries}
    if len(settings) != 1:
        raise ValueError("the components disagree on use_rslora or lora_dropout, which one adapter carries once")
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
    return LoRA(targets, rslora, dropout, bound), overlay(variables, leaves)


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
        return {component: _Config.read(config, f"{where} ({component})") for component, config in fields.items()}
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
    """Marks a module class `adapt` already wrapped.

    The wrapper subclass declares the interceptor its `apply` and `init` run
    under, which is the whole record that a class was adapted: a second
    adapter over it would run one branch inside the other.
    """

    _dew_lora_interceptor: ClassVar[Interceptor]


@runtime_checkable
class Adaptable(Protocol):
    """Declares an objective an adapter attaches to.

    It trains one module, which is its `model`, and it takes the filter that
    says which of that module's leaves the optimizer moves. `LMObjective`
    and `BlockDiffusionObjective` are the two; an objective that keeps no
    model or selects what trains some other way is refused by name.
    """

    model: nn.Module
    trainable: PathFilter | None


def attach(objective: object, adapter: LoRA) -> None:
    """Adapt the module `objective` trains and freeze all but the factors.

    `RunConfig.train` calls this once, after a recipe has built the
    objective and before anything initialises it, so the adapted module is
    what the run traces and the adapter's own leaves are the only ones the
    optimizer moves. The adapted module is a subclass of the same class with
    the same fields, so what the objective read off the model at
    construction still holds.
    """
    if not isinstance(objective, Adaptable):
        raise ValueError(
            f"--lora adapts the module an objective trains and freezes the rest, and "
            f"{type(objective).__name__} keeps no `model` it can select leaves of; train "
            f"an LMObjective or a BlockDiffusionObjective, or leave the adapter unset")
    if objective.trainable is not None:
        raise ValueError(
            f"{type(objective).__name__} already selects what trains, and an adapter "
            f"freezes everything but its own factors; pass one filter, not both")
    objective.model = adapter.adapt(objective.model)
    objective.trainable = adapter.trainable
