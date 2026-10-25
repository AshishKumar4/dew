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
import functools
import json
import math
import numbers
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path as FilePath
from typing import ClassVar, NotRequired, Protocol, TypedDict, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, traverse_util
from flax.linen.dtypes import promote_dtype
from flax.linen.module import Interceptor
from flax.nnx import bridge

from dew.interop.safetensors_io import read_file, write_file
from dew.interop.streaming import WeightLayout
from dew.nn.backbones.layer_plan import group_layers
from dew.nn.dense import Dense, DenseGeneral
from dew.nn.dropout import Dropout
from dew.nn.flops import (
    MissingFlopsCount,
    counted_flops,
    frozen_training_flops,
    parameter_flops,
    projection_flops,
)
from dew.nn.precision import accumulated, at_least_fp32, float32_dot_general
from dew.nn.protocols import (
    Counted,
    CountedConditionedSpatial,
    CountedImages,
    CountedLogits,
    CountedLogitsTraining,
    CountedMatrixUsage,
    CountedMultimodal,
    CountedParameters,
    CountedPredictionParameters,
    CountedPredictions,
    CountedPredictionTraining,
    CountedSequences,
    CountedSpatial,
    CountedTraining,
    CountedVideo,
)
from dew.nn.sharding import LOW_RANK_FACTORS
from dew.objectives.base import FROZEN, Path, Variables, freeze, merge as overlay, part, select, thaw
from dew.quantization import (
    FROZEN_FP8,
    QUANTIZATION,
    InputQuantization,
    frozen_fp8_kernel,
    measured_fp8_sites,
)
from dew.records import finite
from dew.registry import rebuilt, wrapper_layers

PEFT_CONFIG = "adapter_config.json"
PEFT_WEIGHTS = "adapter_model.safetensors"
PEFT_PREFIX = "base_model.model."
DIFFUSERS_WEIGHTS = "pytorch_lora_weights.safetensors"
DIFFUSERS_METADATA = "lora_adapter_metadata"

FACTORS = LOW_RANK_FACTORS
# PEFT's init: A as torch's Linear default, kaiming-uniform with a=sqrt(5),
# which is uniform on +-1/sqrt(fan_in); B zero, so a fresh adapter is the
# identity.
INIT_A = nn.initializers.variance_scaling(1 / 3, "fan_in", "uniform")
INIT_B = nn.initializers.zeros
FACTOR_DTYPE = jnp.float32
"""What an adapter's factors are stored in, whatever its base's storage: attached, loaded or
initialized by the model rebuilt from its record."""


@dataclass(frozen=True)
class Target:
    """One adapted kernel's rank and alpha."""

    rank: int
    alpha: float


def _rank(owner: str, rank: object) -> int:
    """`rank`, refused unless a positive int. This and `_alpha` and `_dropout`
    hold the numbers every description of an adapter states (a `LoRA`, an
    `AdapterRecord`, a PEFT or Diffusers file's config) to one rule."""
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError(f"{owner}: rank must be a positive integer, not {rank!r}")
    return rank


def _dropout(owner: str, dropout: object) -> float:
    """`dropout` as a float, refused outside [0, 1)."""
    if isinstance(dropout, bool) or not isinstance(dropout, numbers.Real) or not 0 <= float(dropout) < 1:
        raise ValueError(f"{owner}: dropout must be in [0, 1), not {dropout!r}")
    return float(dropout)


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
        _rank("LoRA", self.rank)
        if self.alpha is not None:
            finite(self.alpha, "LoRA: alpha")
        _dropout("LoRA", self.dropout)
        if not self.modules or any(not isinstance(module, str) or not module for module in self.modules):
            raise ValueError("modules names at least one projection")

    def apply(self, model: nn.Module, variables: Variables, *, key: int | jax.Array,
              layouts: Mapping[str, WeightLayout] | None = None,
              fp8_base: InputQuantization | None = None,
              base: AdapterBase | None = None) -> Adapter:
        """Attach this adapter to `model` over `variables` and draw its factors.

        `layouts` maps module names to tree paths: `Pretrained.layouts` for a loaded
        source, or None for a model built from its class, whose own module paths are
        its names. `base` is the published model `model` was loaded from
        (`Pretrained.adapt` passes it), which each component's PEFT config names. An
        entry of `modules` that matches no projection is refused. A is drawn per
        module from `key` in sorted name order, the way PEFT draws it, and B is
        zero, so the adapted model computes exactly what the source does.

        `fp8_base` (`fp8_base`'s measurement) runs its frozen sites' products in
        FP8: their kernels are packed from the base this adapter freezes, and the
        model computes them under the record, inside the adapter's branch. FP8
        changes the function, by about E4M3's rounding of each operand.
        """
        from dew.nn.inputs import request_key

        _refuse_unadaptable(model)
        if fp8_base is not None:
            model = fp8_base.joined(model)
        source, variables = variables, thaw(variables)
        bound = _named(bound_layouts(model, variables, layouts or {}), self.modules)
        scaling = 2.0 * self.rank if self.alpha is None else float(self.alpha)
        targets: dict[Path, Target] = {}
        leaves: dict = {}
        keys = jax.random.split(request_key(key), len(bound))
        for name, factor_key in zip(sorted(bound), keys, strict=True):
            factors = _factors(name, bound[name], variables, self.rank)
            shape_a, shape_b = factors.shapes(self.rank)
            targets[factors.module] = Target(self.rank, scaling)
            _insert_factors(leaves, factors.module, INIT_A(factor_key, shape_a, FACTOR_DTYPE),
                            INIT_B(factor_key, shape_b, FACTOR_DTYPE))
        bases = {} if base is None else dict.fromkeys(_components(bound) or {""}, base)
        return Adapter.bound(model, overlay(source, leaves), targets, self.rslora, self.dropout, bound,
                             bases)

    def fp8_base(self, model: nn.Module, variables: Variables, *, tokens: int,
                 modules: Sequence[str] | None = None,
                 layouts: Mapping[str, WeightLayout] | None = None) -> InputQuantization:
        """Measure which projections of the base this adapter freezes run
        faster in FP8 at `tokens` rows a step, on process 0's GPU for the
        whole pool, and return the record `apply` takes as its `fp8_base`
        (`measured_fp8_sites`).

        `modules` names the projections as `modules` does (this adapter's own
        when None) and `layouts` resolves them as `apply` does. The
        measurement holds the GPU; a run records its result, so a rerun
        reproduces the choice without measuring.
        """
        bound = _named(bound_layouts(model, thaw(variables), layouts or {}),
                       self.modules if modules is None else tuple(modules))
        sites = tuple("/".join(layout.paths[0][1:-1]) for layout in bound.values())
        return measured_fp8_sites(thaw(variables), sites, tokens)

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
        _refuse_unadaptable(model)
        bound = bound_layouts(model, thaw(variables), layouts or {})
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


@dataclass(frozen=True)
class AdapterBase:
    """The published model an adapter was bound to: its Hub id and the commit the Hub resolved it
    to. PEFT's config names them (`base_model_name_or_path`, `revision`), so a reader on any machine
    loads the adapter onto the same weights. A config another tool wrote may name a base without a
    commit, which is read and written back as it is. An adapter holds one for each component a
    config names, since a pipeline file's components each name their own (`Adapter.bases`), and a
    run's record holds them (`AdapterRecord`), as it holds each target's binding."""

    name: str
    revision: str | None


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

    @property
    def _binding(self) -> _Bound:
        """The settings `model` was adapted with (`_Bound`), the one record its branch computes from and
        `merge` and `save` read."""
        adapted = type(self.model)
        if not isinstance(adapted, AdaptedClass):
            raise TypeError(f"{adapted.__name__} is not adapted, so it carries no adapter")
        return adapted._dew_lora_bound

    @property
    def targets(self) -> Mapping[Path, Target]:
        return self._binding.targets

    @property
    def rslora(self) -> bool:
        return self._binding.rslora

    @property
    def dropout(self) -> float:
        return self._binding.dropout

    @property
    def layouts(self) -> Mapping[str, WeightLayout]:
        return self._binding.layouts

    @property
    def bases(self) -> Mapping[str, AdapterBase]:
        return self._binding.bases

    @classmethod
    def bound(cls, model: nn.Module, variables: Variables, targets: Mapping[Path, Target], rslora: bool,
              dropout: float, layouts: Mapping[str, WeightLayout],
              bases: Mapping[str, AdapterBase] | None = None) -> Adapter:
        """Adapt `model` to `targets` and split `variables` into factors and frozen weights.

        The factors go under `params` and everything else under `FROZEN`,
        but what a split tree already trains, such as a cross-attention
        block, keeps training (`freeze`). `bases` are the published models its
        components were bound to (`Adapter.bases`).
        """
        adapted = _adapted(model, _Bound(targets, rslora, dropout, layouts, dict(bases or {})))
        factors = {(*path, factor) for path in targets for factor in FACTORS}
        return cls(adapted, _fp8_packed(model, freeze(variables, lambda path: path in factors)))

    @classmethod
    def from_run(cls, directory: str | FilePath, *, step: int | str | None = None,
                 ema: bool | None = None, trust: Sequence[str] = ()) -> Adapter:
        """Return the adapter a run trained, rebuilt from the run's own record and checkpoint.

        It holds the adapted model, the checkpoint's variables and the bindings the
        run recorded, so `save` writes the factors under the source's names without
        the source at hand: the recorded model's own part of a tree that nests it
        beside others (`Objective._recorded_part`, a decision's backbone). A recorded
        binding whose shapes do not fit the restored factors is refused by name.
        `trust` is as `TextGeneration.from_run` takes it.
        """
        from dew.config import ModelConfig
        from dew.inference.tasks import run_record
        from dew.records import record, text
        from dew.registry import objectives

        declaration, step = run_record(str(directory), step, trust)
        config = ModelConfig.from_dict(record(declaration['model'], 'model'))
        if config.adapter is None:
            raise ValueError(f"{directory} trained no adapter")
        objective = objectives[text(declaration['objective'], 'objective')]
        variables = objective._saved_variables(str(directory), step=step, ema=ema)
        # The recorded model's own tree, where the objective's nests it beside parts the record leaves out.
        if objective._recorded_part is not None:
            variables = part(variables, objective._recorded_part)
        return cls.recorded(config.build(), variables, config.adapter)

    @classmethod
    def recorded(cls, model: nn.Module, variables: Variables, adapter: AdapterRecord) -> Adapter:
        """Return the adapter a model record's `adapter` describes.

        It goes over the adapted `model` and a checkpoint's `variables`. The model holds the adapter's
        settings (`_Bound`), so a record that describes other settings than the ones it was adapted
        with is refused rather than read past.
        """
        adapted = type(model)
        if not isinstance(adapted, AdaptedClass):
            raise ValueError(f"{adapted.__name__} is not adapted, so it carries no adapter")
        if _record(adapted._dew_lora_bound) != adapter:
            raise ValueError(f"the record's adapter is not the one {adapted.__name__} was adapted with")
        targets, layouts = adapter.bound()
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
        return cls(model, variables)

    def scale(self, target: Target) -> float:
        return _scale(target, self.rslora)

    def merge(self, variables: Variables) -> Variables:
        """Return `variables` with every delta added into its kernel and the factors removed.

        `variables` is the adapted tree, whole or split as a trainer keeps it. The sum
        runs in at least fp32 at full precision and is stored in the kernel's dtype,
        as PEFT's `merge_and_unload` does. A frozen FP8 site's kernel is the one its
        codes decode to (`frozen_fp8_kernel`), targeted or not, and the codes go,
        since a model under the record would multiply them in the merged kernel's
        place. The result is one `params` collection.
        """
        variables = thaw(variables)
        coded = {path[:-1] for path in traverse_util.flatten_dict(variables.get(QUANTIZATION, {}))
                 if path[-1] == FROZEN_FP8[0]}
        merged: dict = {}
        for site in coded:
            _insert(merged, ("params", *site, "kernel"), _base_kernel(variables, ("params", *site)))
        for path, target in self.targets.items():
            node = _node(variables, path)
            kernel = _updated(_base_kernel(variables, path), node["lora_A"], node["lora_B"],
                              self.scale(target))
            _insert(merged, (*path, "kernel"), kernel)
        dropped = ({(*path, factor) for path in self.targets for factor in FACTORS}
                   | {(QUANTIZATION, *site, leaf) for site in coded for leaf in FROZEN_FP8})
        return select(overlay(variables, merged), lambda path: path not in dropped)

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
            (path / PEFT_CONFIG).write_text(json.dumps(_config(self, named[""], self.bases.get("")), indent=2)
                                            + "\n")
            write_file({PEFT_PREFIX + name: tensor for name, tensor in tensors.items()},
                       path / PEFT_WEIGHTS, {"format": "pt"})
            return
        metadata = {f"{component}.{field}": value
                    for component, targets in sorted(named.items())
                    for field, value in _config(self, targets, self.bases.get(component)).items()}
        write_file(tensors, path / DIFFUSERS_WEIGHTS,
                   {"format": "pt", DIFFUSERS_METADATA: json.dumps(metadata, indent=2, sort_keys=True)})


def _refuse_unadaptable(model: nn.Module) -> None:
    """Refuse a model already adapted, and an NNX model behind Flax's bridge,
    whose `nnx.Linear` makes no `Dense` call the branch could join: its
    factors would train while its forward never read them."""
    if isinstance(type(model), AdaptedClass):
        raise ValueError("The model is already adapted")
    if isinstance(model, bridge.ToLinen):
        raise TypeError(f"{model.nnx_class.__name__} is an NNX model behind flax.nnx.bridge.ToLinen; an "
                        "adapter joins flax.linen.Dense calls, and its nnx.Linear layers make none. Adapt "
                        "it in NNX before the bridge: flax.nnx.LoRALinear in place of nnx.Linear, or "
                        "flax.nnx.LoRA beside it")


def _scale(target: Target, rslora: bool) -> float:
    return target.alpha / (math.sqrt(target.rank) if rslora else target.rank)


def _updated(kernel: jax.Array, a: jax.Array, b: jax.Array, scale: float) -> jax.Array:
    """`kernel` plus `scale` times the factors' product, summed in at least fp32 at full precision
    and stored in the kernel's dtype, as PEFT's `merge_and_unload` does."""
    dtype = at_least_fp32(kernel.dtype)
    delta = accumulated(functools.partial(jnp.tensordot, axes=1, precision=jax.lax.Precision.HIGHEST),
                        a.astype(dtype), b.astype(dtype))
    return (kernel.astype(dtype) + scale * delta).astype(kernel.dtype)


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
class _Bound:
    """An adapter's settings, held once by the model it adapts (its branch, `_Branch`) and read by its
    `Adapter`, `merge`, `save` and the model's record: `targets` the adapted modules' paths with their
    rank and alpha, `rslora`, `dropout`, `layouts` the bindings by the name a file stores each module
    under, and `bases` the published model each component was bound to (`""` for a decoder's or a
    registry-built model's one unnamed component), which `save` names in that component's PEFT config;
    a component bound to a local directory or a model built from its class has none."""

    targets: Mapping[Path, Target]
    rslora: bool = False
    dropout: float = 0.0
    layouts: Mapping[str, WeightLayout] = dataclasses.field(default_factory=dict)
    bases: Mapping[str, AdapterBase] = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class _Branch:
    """The interceptor an adapted model runs under: the adapter branch `bound` describes on every
    target module's `__call__`."""

    bound: _Bound
    root: Path = ("params",)

    @property
    def targets(self) -> Mapping[Path, Target]:
        return self.bound.targets

    @property
    def rslora(self) -> bool:
        return self.bound.rslora

    @property
    def dropout(self) -> float:
        return self.bound.dropout

    def __call__(self, next_fun, args, kwargs, context):
        module = context.module
        root = self.root
        if context.method_name == "nonlinear_input":
            # A layer asks whether its submodule is linear in its input on
            # this call (`MultiHeadLatentAttention.nonlinear_input`): the
            # branch's dropout draws on it, on a target under a dropout stream.
            name = args[0] if args else kwargs["name"]
            return next_fun(*args, **kwargs) or bool(
                self.dropout and module.has_rng("dropout")
                and _target_at(self.targets, root + module.path + (name,)) is not None)
        if context.method_name == "applied_kernel":
            # A model asks for the kernel its submodule multiplies by
            # (`CausalTransformer.applied_kernel`): a target's carries the
            # branch's update, as `LoRA.merge` sums it.
            name = args[0] if args else kwargs["name"]
            kernel = next_fun(*args, **kwargs)
            target = _target_at(self.targets, root + module.path + (name,))
            if target is None:
                return kernel
            child = getattr(module, name)
            a, b = child.get_variable("params", "lora_A"), child.get_variable("params", "lora_B")
            return _updated(kernel, a, b, _scale(target, self.rslora))
        target = _target_at(self.targets, root + module.path) if context.method_name == "__call__" else None
        if target is None:
            return next_fun(*args, **kwargs)
        # Dew's layers are flax's with the bias summed at fp32; a model built on flax's own adapts alike.
        if type(module) not in (Dense, DenseGeneral, nn.Dense, nn.DenseGeneral):
            raise TypeError(
                f"{'/'.join(root + module.path)} is a {type(module).__name__}; an adapter "
                "targets Dense and DenseGeneral kernels")
        x = args[0] if args else kwargs["inputs"]
        output = next_fun(*args, **kwargs)
        if isinstance(module, nn.DenseGeneral):
            if module.batch_dims:
                raise TypeError(f"{'/'.join(root + module.path)} contracts with batch dims")
            axes = (module.axis,) if isinstance(module.axis, int) else tuple(module.axis)
        else:
            axes = (-1,)
        axes = tuple(sorted(axis % x.ndim for axis in axes))
        if module.has_variable("params", "lora_A"):
            # A frozen FP8 site holds no kernel past init (`_fp8_packed`); its factors hold their shapes.
            a, b = module.get_variable("params", "lora_A"), module.get_variable("params", "lora_B")
        else:
            kernel = module.get_variable("params", "kernel")
            a = module.param("lora_A", INIT_A, (*kernel.shape[:len(axes)], target.rank), FACTOR_DTYPE)
            b = module.param("lora_B", INIT_B, (target.rank, *kernel.shape[len(axes):]), FACTOR_DTYPE)
        # The branch drops out its input when the forward carries the
        # dropout stream, which is how a training forward is marked; an
        # evaluation forward carries none.
        if self.dropout and module.has_rng("dropout"):
            x = Dropout(self.dropout, deterministic=False)(x)
        x, a, b = promote_dtype(x, a, b, dtype=module.dtype)
        hidden = float32_dot_general(x, a, ((axes, tuple(range(len(axes)))), ((), ())),
                                     precision=module.precision)
        delta = float32_dot_general(hidden, b, (((hidden.ndim - 1,), (0,)), ((), ())),
                                    precision=module.precision)
        return output + jnp.asarray(_scale(target, self.rslora), delta.dtype) * delta


def _refusing_counts(model: nn.Module, base: type[nn.Module], *protocols) -> type[nn.Module]:
    """Inherited geometry describes the native forward, not the added factors."""
    def missing(self, *args, **kwargs) -> float:
        raise MissingFlopsCount(f"{type(model).__name__} needs declared adapter matrix usage "
                                "and parameter work for this geometry")

    methods = {name: nn.nowrap(missing) for protocol in protocols for name in vars(protocol)
               if name.endswith('_flops') and hasattr(model, name)}
    return type('UncountedAdapter', (base,), methods)


def _counted_adapter(model: nn.Module, branch: _Branch) -> type[nn.Module]:
    """An adapter adds analytical work only to capabilities its wrapped model declares."""
    base = _refusing_counts(model, type(model), CountedImages, CountedMultimodal, CountedSequences,
                            CountedSpatial, CountedConditionedSpatial, CountedVideo)
    bindings = {layout.paths[0][:-1]: layout for layout in branch.bound.layouts.values()
                if len(layout.paths) == 1}

    class CountedAdapter(base):
        @nn.nowrap
        def _adapter_forward(self, lengths, *, logits: bool = False,
                              usage: Callable[[Path, np.ndarray], np.ndarray] | None = None) -> float:
            original = unadapted(self)
            if not isinstance(original, CountedMatrixUsage):
                raise MissingFlopsCount(f"{type(original).__name__} has no declared adapter matrix usage")
            total = 0.0
            for path, target in branch.targets.items():
                layout = bindings.get(path)
                if layout is None or len(layout.shape) != 2:
                    raise MissingFlopsCount(f"adapter {'/'.join(path)} needs bound two-axis matrix geometry")
                shape = layout.shape
                if layout.transpose is not None:
                    if len(layout.transpose) != 2:
                        raise MissingFlopsCount(f"adapter {'/'.join(path)} has nonmatrix native axes")
                    shape = tuple(shape[layout.transpose.index(axis)] for axis in range(2))
                positions = (original.matrix_usage(path, lengths, logits=logits) if usage is None
                             else usage(path, lengths))
                total += projection_flops(positions, shape[0] + shape[1], target.rank)
            return total

        @nn.nowrap
        def model_flops(self, lengths) -> float:
            """Actual forward work, including factors, distinct from training work."""
            return counted_flops(unadapted(self), lengths) + self._adapter_forward(lengths)

        @nn.nowrap
        def parameter_flops(self, lengths) -> float:
            return parameter_flops(unadapted(self), lengths) + self._adapter_forward(lengths)

        @nn.nowrap
        def training_flops(self, lengths) -> float:
            """Frozen base maps get 4N, factors 6N; state/attention retain 3*forward.

            Frozen matrices are charged 4N throughout, including the first
            layer's input projections, whose input gradient a frozen
            embedding never needs.
            """
            original = unadapted(self)
            forward = counted_flops(original, lengths)
            learned = parameter_flops(original, lengths)
            return frozen_training_flops(forward, learned, self._adapter_forward(lengths))

    class LogitsAdapter(CountedAdapter):
        @nn.nowrap
        def prediction_flops(self, lengths, *, prediction_lengths=None) -> float:
            original = unadapted(self)
            if not isinstance(original, CountedPredictionParameters):
                raise MissingFlopsCount(f"{type(original).__name__} has no declared prediction matrix work")
            forward = original.prediction_flops(lengths, prediction_lengths=prediction_lengths)
            usage = partial(original.prediction_matrix_usage, prediction_lengths=prediction_lengths)
            return forward + self._adapter_forward(lengths, usage=usage)

        @nn.nowrap
        def prediction_parameter_flops(self, lengths, *, prediction_lengths=None) -> float:
            original = unadapted(self)
            if not isinstance(original, CountedPredictionParameters):
                raise MissingFlopsCount(f"{type(original).__name__} has no declared prediction matrix work")
            learned = original.prediction_parameter_flops(lengths, prediction_lengths=prediction_lengths)
            usage = partial(original.prediction_matrix_usage, prediction_lengths=prediction_lengths)
            return learned + self._adapter_forward(lengths, usage=usage)

        @nn.nowrap
        def prediction_training_flops(self, lengths, *, prediction_lengths=None) -> float:
            original = unadapted(self)
            if not isinstance(original, CountedPredictionParameters):
                raise MissingFlopsCount(f"{type(original).__name__} has no declared prediction matrix work")
            forward = original.prediction_flops(lengths, prediction_lengths=prediction_lengths)
            learned = original.prediction_parameter_flops(lengths, prediction_lengths=prediction_lengths)
            usage = partial(original.prediction_matrix_usage, prediction_lengths=prediction_lengths)
            return frozen_training_flops(forward, learned, self._adapter_forward(lengths, usage=usage))

        @nn.nowrap
        def logits_flops(self, lengths) -> float:
            original = unadapted(self)
            if not isinstance(original, CountedLogits):
                raise MissingFlopsCount(f"{type(original).__name__} has no declared vocabulary-head work")
            return original.logits_flops(lengths) + self._adapter_forward(lengths, logits=True)

        @nn.nowrap
        def logits_training_flops(self, lengths) -> float:
            original = unadapted(self)
            if not isinstance(original, CountedLogits):
                raise MissingFlopsCount(f"{type(original).__name__} has no declared vocabulary-head work")
            # The vocabulary head is frozen whole: all its forward work takes no weight gradient.
            head = original.logits_flops(lengths)
            return frozen_training_flops(head, head, self._adapter_forward(lengths, logits=True))

    if isinstance(model, CountedLogits):
        return LogitsAdapter
    if isinstance(model, Counted):
        return CountedAdapter
    return _refusing_counts(model, base, CountedParameters, CountedTraining, CountedLogitsTraining,
                            CountedPredictions, CountedPredictionParameters, CountedPredictionTraining)


def _fp8_packed(model: nn.Module, variables: Variables) -> dict:
    """`variables` with the frozen FP8 kernels of the record `model` computes
    under packed from its `FROZEN` base (`InputQuantization.packed`), the
    only kernels an adapter never trains, in place of those kernels: the
    product never reads them (`frozen_fp8_dense`), and a merge decodes them
    (`frozen_fp8_kernel`)."""
    records = [layer._dew_wrapper() for layer in wrapper_layers(type(model))[1]]
    record = next((wrapper for wrapper in records if isinstance(wrapper, InputQuantization)), None)
    if record is None or not record.frozen_fp8:
        return dict(variables)
    packed = record.packed(variables[FROZEN])
    sites = {tuple(site.split("/")) for site in record.frozen_fp8}
    frozen = select(variables[FROZEN], lambda path: not (path[-1] == "kernel" and path[:-1] in sites))
    return {**variables, FROZEN: frozen, QUANTIZATION: overlay(variables.get(QUANTIZATION, {}), packed)}


def _adapted(model: nn.Module, bound: _Bound) -> nn.Module:
    """Return `model` computing the adapter branch `bound` describes in every target module.

    The result is an instance of a subclass of `model`'s class with the
    same fields and methods, so it builds, checks and generates as the model
    does. Its count adds the branch only where the model supplies geometry.
    `apply` folds `FROZEN` back into `params` and runs under the interceptor;
    `init` (Flax's dispatches through `init_with_output`) runs under it and
    splits what it draws, the factors under `params` and the rest under
    `FROZEN`. The class's record (`AdapterRecord`) is `bound`'s, written when a run asks for it.
    """
    base = type(model)
    if isinstance(base, AdaptedClass):
        raise ValueError("The model is already adapted")
    branch = _Branch(bound)
    factors = {(*path, factor) for path in bound.targets for factor in FACTORS}
    adapted_base = _counted_adapter(model, branch)

    class Adapted(adapted_base):
        # Staticmethods, or Flax would wrap the callables as module methods.
        _dew_lora_interceptor: ClassVar[Interceptor] = staticmethod(branch)
        _dew_lora_bound: ClassVar[_Bound] = bound
        _dew_wrapped: ClassVar[type[nn.Module]] = base

        @staticmethod
        def _dew_wrapper() -> AdapterRecord:
            return _record(bound)

        def apply(self, variables, *args, **kwargs):
            with nn.intercept_methods(branch):
                return super().apply(thaw(variables), *args, **kwargs)

        def init_with_output(self, *args, **kwargs):
            with nn.intercept_methods(branch):
                output, variables = super().init_with_output(*args, **kwargs)
            return output, _fp8_packed(model, dict(freeze(variables, lambda path: path in factors)))

    Adapted.__name__ = Adapted.__qualname__ = base.__name__
    return rebuilt(Adapted, model)


@dataclass(frozen=True)
class AdapterBinding:
    """Where a file writes one target's weight: the name, the stored shape and
    the transpose from the native kernel."""

    name: str
    shape: tuple[int, ...]
    transpose: tuple[int, ...] | None


@dataclass(frozen=True)
class AdapterRecord:
    """A bound adapter as a model's wrapper chain records it
    (`ModelConfig.wrappers`): each target's rank and alpha by its native
    module path, as `Target` holds them, and each target's binding, so the
    run's adapter, mixed ranks and all, saves with no source at hand. The
    checkpoint stores the factors."""

    targets: Mapping[str, Target]
    rslora: bool
    dropout: float
    layouts: Mapping[str, AdapterBinding]
    bases: Mapping[str, AdapterBase] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        for module, target in self.targets.items():
            _rank(f"the adapter record's {module}", target.rank)
            finite(target.alpha, f"the adapter record's {module}: alpha")
        _dropout("the adapter record", self.dropout)
        if not self.targets or any(not name or name in ('.', '..')
                                   for module in self.targets for name in module.split('/')):
            raise ValueError("adapter targets must be keyed by native module names, at least one")
        if not set(self.layouts) <= set(self.targets):
            raise ValueError("adapter layouts must be keyed by the adapter's own modules")

    def apply(self, model: nn.Module) -> nn.Module:
        """`model` adapted as this record says, which is how a recorded run's
        model is rebuilt (`ModelConfig.build`); the checkpoint supplies the factors."""
        targets, layouts = self.bound()
        return _adapted(model, _Bound(targets, self.rslora, self.dropout, layouts, dict(self.bases)))

    def bound(self) -> tuple[dict[Path, Target], dict[str, WeightLayout]]:
        """The targets by native path and the bindings by the name a file writes them under."""
        targets = {tuple(module.split('/')): target for module, target in self.targets.items()}
        layouts = {binding.name.removesuffix('.weight').replace('/', '.'): WeightLayout(
                       binding.name, ((*module.split('/'), 'kernel'),), binding.shape, binding.transpose)
                   for module, binding in self.layouts.items()}
        return targets, layouts


def _record(bound: _Bound) -> AdapterRecord:
    """The bound adapter as a model record holds it."""
    targets = bound.targets
    bindings = {layout.paths[0][:-1]: layout for layout in bound.layouts.values()}
    return AdapterRecord({'/'.join(path): targets[path] for path in sorted(targets)}, bound.rslora,
                         bound.dropout,
                         {'/'.join(path): AdapterBinding(bindings[path].name, bindings[path].shape,
                                                         bindings[path].transpose)
                          for path in sorted(targets) if path in bindings}, dict(bound.bases))


def _site_codes(variables: Variables, path: Path) -> Mapping | None:
    """A frozen FP8 site's packed leaves for the module at `path` (under
    `params`), or None where the module holds its own kernel."""
    leaves = variables.get(QUANTIZATION, {})
    for name in path[1:]:
        leaves = leaves.get(name, {})
    return leaves if FROZEN_FP8[0] in leaves else None


def _base_kernel(variables: Variables, path: Path) -> jax.Array:
    """The kernel of the module at `path`: its own, or a frozen FP8 site's
    decoded from its packed codes (`frozen_fp8_kernel`)."""
    codes = _site_codes(variables, path)
    return _node(variables, path)["kernel"] if codes is None else jnp.asarray(frozen_fp8_kernel(codes))


def _kernel_shape(variables: Variables, path: Path) -> tuple[int, ...]:
    """The shape of `_base_kernel`, read without decoding a frozen FP8 site."""
    codes = _site_codes(variables, path)
    return tuple(np.shape(_node(variables, path)["kernel"] if codes is None else codes[FROZEN_FP8[0]].T))


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
    """Write one target's two factor leaves into `leaves`, under `FACTORS`, stored as `FACTOR_DTYPE`."""
    for name, value in zip(FACTORS, (a, b), strict=True):
        _insert(leaves, (*module, name), jnp.asarray(value, FACTOR_DTYPE))


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
    kernel_shape = _kernel_shape(variables, layout.paths[0][:-1])
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
    base: AdapterBase | None

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
        if (not isinstance(rank_pattern, Mapping) or not isinstance(alpha_pattern, Mapping)
                or not isinstance(rslora, bool)):
            raise ValueError(f"{where} needs per-module patterns and a boolean use_rslora")
        name, revision = config.get("base_model_name_or_path"), config.get("revision")
        if not isinstance(name, str | None) or not isinstance(revision, str | None):
            raise ValueError(f"{where} names its base model and revision as strings or not at all")
        return cls(_rank(where, rank), finite(alpha, f"{where}: alpha"),
                   {module: _rank(f"{where} rank_pattern {module}", value)
                    for module, value in rank_pattern.items()},
                   {module: finite(value, f"{where} alpha_pattern {module}: alpha")
                    for module, value in alpha_pattern.items()},
                   rslora, _dropout(where, dropout), None if name is None else AdapterBase(name, revision))

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
    they are written so that a PEFT reader finds the keys it expects. An adapter
    bound to a Hub model names it and its commit (`AdapterBase`), which PEFT
    loads the base from; one bound to a local directory or a model built from
    its class names none, since no other machine resolves it.
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
    base_model_name_or_path: NotRequired[str]
    revision: NotRequired[str]


def _config(lora: Adapter, named: Mapping[str, Target], base: AdapterBase | None) -> PeftConfig:
    """Build the PEFT config of `named` targets, keyed by the relative module name.

    The commonest rank and alpha become the defaults and the rest the patterns.
    """
    ranks = [target.rank for target in named.values()]
    alphas = [target.alpha for target in named.values()]
    rank = max(set(ranks), key=ranks.count)
    alpha = max(set(alphas), key=alphas.count)
    config: PeftConfig = {
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
    if base is not None:
        config["base_model_name_or_path"] = base.name
        if base.revision is not None:
            config["revision"] = base.revision
    return config


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
    """Build the adapter the entries describe, its factors restored into the model's tree, which
    keeps training what it trained where it came split (`freeze`)."""
    source, variables = variables, thaw(variables)
    components = _components(layouts)
    settings = {(entry.config.rslora, entry.config.dropout) for entry in entries}
    if len(settings) != 1:
        raise ValueError(
            "the components disagree on use_rslora or lora_dropout, which one adapter carries once"
        )
    (rslora, dropout), = settings
    # A pipeline file's components each name their own base, as the tool that wrote it bound each
    # (Diffusers writes each component's config apart).
    bases = {_component(components, entry.module)[0]: entry.config.base for entry in entries
             if entry.config.base is not None}
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
    return Adapter.bound(model, overlay(source, leaves), targets, rslora, dropout, bound, bases)


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
            ranks.setdefault(component, {})[module] = _rank(f"{where} {component}.{module}", tensor.shape[1])
    return {component: _Config(next(iter(found.values())), float(next(iter(found.values()))),
                               found, {}, rslora=False, dropout=0.0, base=None)
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


def unadapted(model: nn.Module) -> nn.Module:
    """`model` without its adapter: its base class over the same fields, in
    every other wrapper its class wraps that base in (a quantization), in
    order. A model no adapter wraps is itself."""
    base, layers = wrapper_layers(type(model))
    if not any("_dew_lora_interceptor" in vars(layer) for layer in layers):
        return model
    model = rebuilt(base, model)
    for layer in reversed(layers):
        if "_dew_lora_interceptor" not in vars(layer):
            model = layer._dew_wrapper().apply(model)
    return model


@runtime_checkable
class AdaptedClass(Protocol):
    """Marks a module class an adapter already wrapped, at any depth of its wrappers.

    The wrapper subclass declares the interceptor its `apply` and `init` run
    under, which is the whole record that a class was adapted: a second
    adapter over it would run one branch inside the other.
    """

    _dew_lora_interceptor: ClassVar[Interceptor]
    _dew_lora_bound: ClassVar[_Bound]


__all__ = ["Adapter", "AdapterRecord", "LoRA", "PeftConfig", "Target", "unadapted"]
