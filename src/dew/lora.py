"""Low-rank adapters (LoRA, arXiv 2106.09685) over a variables tree.

An adapter is a set of rank-`r` deltas on the kernels of Dense modules.
Beside a target kernel `W`, `[in..., out...]`, the tree holds `lora_A`,
`[in..., r]`, and `lora_B`, `[r, out...]`, and the module computes
`x W + scale * (x A) B` with `scale = alpha / r` (`alpha / sqrt(r)` for
rsLoRA, arXiv 2312.03732). Merged, `scale * A B` is added into the kernel
and the factors are gone. A target is the path of its module's leaves in
the tree, `("params", "layers_0", "self_attn", "q_proj")`, so one
description serves every model and every collection.

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
Source module names resolve to tree paths through the source's own
`weight_layouts`, the bindings its export runs backwards, so an adapter is
placed exactly where the base tensor it modifies went.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path as FilePath
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.dtypes import promote_dtype
from flax.linen.module import Interceptor

from dew.interop.pretrained import Pretrained, WeightLayout
from dew.interop.safetensors_io import read_file, write_file
from dew.objectives.base import Path, Variables, merge as overlay, select

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
    """Which kernels carry a low-rank delta, and how the delta scales."""

    targets: Mapping[Path, Target]
    rslora: bool = False
    dropout: float = 0.0

    def scale(self, target: Target) -> float:
        return target.alpha / (math.sqrt(target.rank) if self.rslora else target.rank)

    def trainable(self, path: Path) -> bool:
        """The adapter's own leaves: the `PathFilter` a partial run trains."""
        return path[-1] in FACTORS and path[:-1] in self.targets

    def adapt(self, model: nn.Module, root: Path = ("params",)) -> nn.Module:
        """`model` computing the adapter branch in every target module.

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
        if hasattr(base, "_dew_lora_interceptor"):
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
            target = self.targets.get(root + module.path) if context.method_name == "__call__" else None
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
        """`variables` with every delta added into its kernel and the factors removed.

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
    """The source-layout bindings of one target's factors.

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
        """The tree's `lora_A` and `lora_B` shapes at `rank`."""
        return (*self.kernel_shape[:self.contracted], rank), (rank, *self.kernel_shape[self.contracted:])


def _factors(name: str, layout: WeightLayout, variables: Variables, rank: int) -> _Factors:
    """The factor layouts of the kernel `layout` binds in `variables`."""
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


def _layouts(source: Pretrained) -> dict[str, WeightLayout]:
    """Source module name to the layout of its weight: `model.<module>` for
    a decoder, `unet.<module>` for a pipeline component."""
    return {layout.name.removesuffix(".weight").replace("/", "."): layout
            for layout in source.weight_layouts if layout.name.endswith(".weight")}


def _component(source: Pretrained, module: str) -> tuple[str, str]:
    """A pipeline source's `(component, name relative to it)`; a decoder is
    one component, `""`, with names relative to its root, which is what the
    reference injects into and matches its patterns against."""
    if source.process is None:
        return "", module
    component, _, relative = module.partition(".")
    return component, relative


# --------------------------------------------------------------------------
# PEFT configuration
# --------------------------------------------------------------------------

_REFUSED_FLAGS = ("fan_in_fan_out", "use_dora", "lora_bias", "modules_to_save",
                  "trainable_token_indices", "target_parameters")


@dataclass(frozen=True)
class _Config:
    """The fields of a PEFT `LoraConfig` an adapter's numerics depend on."""

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
    """PEFT's `get_pattern_key`: the first pattern that names this module,
    else the name itself, which no pattern table holds."""
    return next((key for key in patterns if re.match(rf"(.*\.)?({key})$", relative)), relative)


def _config(lora: LoRA, named: Mapping[str, Target]) -> dict[str, object]:
    """The PEFT config of `named` targets, keyed by the relative module name;
    the commonest rank and alpha are the defaults, the rest the patterns."""
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
# Loading, creating and saving
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Entry:
    """One adapted source module as a file names it, with its component's config."""

    module: str
    a: np.ndarray
    b: np.ndarray
    config: _Config


def _entries(source: Pretrained, tensors: Mapping[str, np.ndarray],
             configs: Mapping[str, _Config], prefix: str) -> list[_Entry]:
    """Every module's factor pair; `configs` is keyed by component."""
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
        component = _component(source, module)[0]
        if component not in configs:
            raise ValueError(f"the file carries no config for {module}'s component {component!r}")
        entries.append(_Entry(module, pair["A"], pair["B"], configs[component]))
    return entries


def _place(source: Pretrained, entries: Sequence[_Entry]) -> tuple[LoRA, Variables]:
    """The adapter the entries describe, with its factors restored into the source's tree."""
    layouts = _layouts(source)
    settings = {(entry.config.rslora, entry.config.dropout) for entry in entries}
    if len(settings) != 1:
        raise ValueError("the components disagree on use_rslora or lora_dropout, which one adapter carries once")
    (rslora, dropout), = settings
    targets: dict[Path, Target] = {}
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
        relative = _component(source, entry.module)[1]
        declared = entry.config.rank_of(relative)
        if declared != rank:
            raise ValueError(f"{entry.module} stores rank {rank} but its config declares {declared}")
        factors = _factors(entry.module, layout, source.variables, rank)
        if (entry.b.shape[0], entry.a.shape[1]) != layout.shape:
            raise ValueError(
                f"{entry.module} carries a delta of {(entry.b.shape[0], entry.a.shape[1])} "
                f"on a weight the source stores as {layout.shape}")
        shape_a, shape_b = factors.shapes(rank)
        targets[factors.module] = Target(rank, entry.config.alpha_of(relative))
        _insert(leaves, (*factors.module, "lora_A"), factors.a.restore(entry.a, shape_a))
        _insert(leaves, (*factors.module, "lora_B"), factors.b.restore(entry.b, shape_b))
    return LoRA(targets, rslora, dropout), overlay(source.variables, leaves)


def load(source: Pretrained, path: str | FilePath) -> tuple[LoRA, Variables]:
    """An adapter for `source` and the source's variables with the factors in place.

    `path` is a PEFT adapter directory or a Diffusers file (or the directory
    holding one). Targets the source does not bind, tensors whose shapes do
    not fit the bound weight, ranks that disagree with the config, and PEFT
    features this loader does not carry are refused by name.
    """
    path = FilePath(path)
    if (path / PEFT_WEIGHTS).is_file():
        where = str(path / PEFT_CONFIG)
        config = _Config.read(json.loads(FilePath(where).read_text()), where)
        tensors, _ = read_file(path / PEFT_WEIGHTS)
        return _place(source, _entries(source, tensors, {"": config}, PEFT_PREFIX))
    file = path if path.is_file() else path / DIFFUSERS_WEIGHTS
    if not file.is_file():
        raise FileNotFoundError(f"{path} holds neither {PEFT_WEIGHTS} nor {DIFFUSERS_WEIGHTS}")
    tensors, metadata = read_file(file)
    configs = _diffusers_configs(tensors, metadata.get(DIFFUSERS_METADATA), str(file))
    return _place(source, _entries(source, tensors, configs, ""))


def _diffusers_configs(tensors: Mapping[str, np.ndarray], metadata: str | None,
                       where: str) -> dict[str, _Config]:
    """Per-component PEFT configs from the header, or what Diffusers derives
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


def _named(source: Pretrained, wanted: Sequence[str]) -> dict[str, WeightLayout]:
    """The projections PEFT's `target_modules` selects: a name relative to
    the model that is an entry, or ends in `.` and an entry."""
    matched: dict[str, WeightLayout] = {}
    hit = set()
    for name, layout in _layouts(source).items():
        relative = _component(source, name)[1]
        entries = {entry for entry in wanted if relative == entry or relative.endswith("." + entry)}
        if entries and layout.paths[0][-1] == "kernel":
            matched[name] = layout
            hit |= entries
    missed = [entry for entry in wanted if entry not in hit]
    if missed:
        raise ValueError(f"{', '.join(missed)} match no projection of this source")
    return matched


def fresh(source: Pretrained, *, rank: int, alpha: float, modules: Sequence[str], key: jax.Array,
          rslora: bool = False, dropout: float = 0.0) -> tuple[LoRA, Variables]:
    """A new adapter on the projections `modules` name, and the source's variables with its factors.

    `modules` are PEFT's `target_modules`: a projection matches when its
    name relative to the model (`model.layers.0.self_attn.q_proj`, or `to_q`
    under a pipeline component) is the entry or ends in `.` and the entry.
    An entry that matches no projection is refused. A is drawn from `key`
    the way PEFT draws it and B is zero, so the fresh adapter is the identity.
    """
    matched = _named(source, modules)
    targets: dict[Path, Target] = {}
    leaves: dict = {}
    for name, factor_key in zip(sorted(matched), jax.random.split(key, len(matched)), strict=True):
        factors = _factors(name, matched[name], source.variables, rank)
        shape_a, shape_b = factors.shapes(rank)
        targets[factors.module] = Target(rank, alpha)
        _insert(leaves, (*factors.module, "lora_A"), INIT_A(factor_key, shape_a, jnp.float32))
        _insert(leaves, (*factors.module, "lora_B"), INIT_B(factor_key, shape_b, jnp.float32))
    return LoRA(targets, rslora, dropout), overlay(source.variables, leaves)


def save(source: Pretrained, lora: LoRA, variables: Variables, path: str | FilePath) -> None:
    """Write the adapter's factors from `variables` under the source's module names.

    A decoder source writes PEFT's directory; a pipeline source writes the
    Diffusers file with each component's PEFT config in its header.
    """
    layouts = _layouts(source)
    names = {layout.paths[0][:-1]: name for name, layout in layouts.items() if layout.paths[0][-1] == "kernel"}
    tensors: dict[str, np.ndarray] = {}
    named: dict[str, dict[str, Target]] = {}
    for module, target in lora.targets.items():
        name = names.get(module)
        if name is None:
            raise ValueError(f"{'/'.join(module)} is not a projection this source binds")
        component, relative = _component(source, name)
        factors = _factors(name, layouts[name], variables, target.rank)
        for factor in (factors.a, factors.b):
            tensors[factor.name] = factor.export(variables)
        named.setdefault(component, {})[relative] = target
    path = FilePath(path)
    path.mkdir(parents=True, exist_ok=True)
    if source.process is None:
        (path / PEFT_CONFIG).write_text(json.dumps(_config(lora, named[""]), indent=2) + "\n")
        write_file({PEFT_PREFIX + name: tensor for name, tensor in tensors.items()},
                   path / PEFT_WEIGHTS, {"format": "pt"})
        return
    metadata = {f"{component}.{field}": value
                for component, targets in sorted(named.items())
                for field, value in _config(lora, targets).items()}
    write_file(tensors, path / DIFFUSERS_WEIGHTS,
               {"format": "pt", DIFFUSERS_METADATA: json.dumps(metadata, indent=2, sort_keys=True)})
