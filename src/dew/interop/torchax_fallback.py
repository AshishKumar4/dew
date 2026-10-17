"""Tier 3: a transformers PyTorch causal LM, lowered to JAX by torchax.

`Pretrained.load(source, fallback="torchax")` builds the model with
transformers on the host and hands its forward to torchax, which runs every
torch op as a JAX op, so the model jits, differentiates and shards like any
other JAX function. Dew knows nothing about the architecture: no Dew
kernels, no logical sharding declarations, no cached decoding. What Dew
supplies is the seam: a Flax module over the torch forward, the parameters
split from the buffers, and a `Layout` that places tensors by their torch
names.

The split is `JittableModule`'s. Its `params` are the trainable
`nn.Parameter`s and go under the `params` collection, the one the optimizer
moves; every other tensor (rotary `inv_freq`, causal masks) goes under
`buffers`, which the trainer carries as state and never updates. A tied
output head is one parameter under its first name, as `JittableModule`
deduplicates it.

This module imports neither torch nor torchax until a load asks for them.
"""

from __future__ import annotations

import contextlib
import fnmatch
import importlib.metadata
import json
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from dew.interop import sources
from dew.interop.generation_config import read_generation_config
from dew.interop.pickles import host_view
from dew.interop.pretrained import PretrainedFallback, source_processor
from dew.interop.weights import AUTO
from dew.nn.protocols import OutputTable
from dew.nn.sharding import LogicalAxes, logical_spec, parameter_path
from dew.registry import resolve_dtype
from dew.training.distributed import Layout, Placement, parameter_axes

if TYPE_CHECKING:
    import torch
    from torchax.interop import JittableModule
    from transformers import PreTrainedModel

    from dew.nn.backbones.decoder_stack import DecoderBank

INSTALL = "pip install 'dewml[torchax]'"

TORCH_AXES: tuple[tuple[str, LogicalAxes], ...] = (
    # nn.Linear stores [out, in]; its bias is the output side alone.
    ("*.q_proj.weight", ("heads", "embed")), ("*.q_proj.bias", ("heads",)),
    ("*.k_proj.weight", ("kv", "embed")), ("*.k_proj.bias", ("kv",)),
    ("*.v_proj.weight", ("kv", "embed")), ("*.v_proj.bias", ("kv",)),
    ("*.o_proj.weight", ("embed", "heads")),
    ("*.self_attn.out_proj.weight", ("embed", "heads")),
    ("*.attention.query_key_value.weight", ("heads", "embed")),
    ("*.attention.query_key_value.bias", ("heads",)),
    ("*.attention.dense.weight", ("embed", "heads")),
    ("*.gate_proj.weight", ("mlp", "embed")), ("*.up_proj.weight", ("mlp", "embed")),
    ("*.down_proj.weight", ("embed", "mlp")),
    ("*.fc1.weight", ("mlp", "embed")), ("*.fc1.bias", ("mlp",)),
    ("*.fc2.weight", ("embed", "mlp")),
    ("*.dense_h_to_4h.weight", ("mlp", "embed")), ("*.dense_h_to_4h.bias", ("mlp",)),
    ("*.dense_4h_to_h.weight", ("embed", "mlp")),
    # GPT-2's Conv1D stores [in, out].
    ("*.attn.c_attn.weight", ("embed", "heads")), ("*.attn.c_attn.bias", ("heads",)),
    ("*.attn.c_proj.weight", ("heads", "embed")),
    ("*.mlp.c_fc.weight", ("embed", "mlp")), ("*.mlp.c_fc.bias", ("mlp",)),
    ("*.mlp.c_proj.weight", ("mlp", "embed")),
    # Token tables and output heads are [vocab, embed], positions [positions, embed].
    ("*embed_tokens.weight", ("vocab", "embed")), ("*.wte.weight", ("vocab", "embed")),
    ("*embed_in.weight", ("vocab", "embed")), ("*word_embeddings.weight", ("vocab", "embed")),
    ("lm_head.weight", ("vocab", "embed")), ("embed_out.weight", ("vocab", "embed")),
    ("*.wpe.weight", (None, "embed")), ("*embed_positions.weight", (None, "embed")),
)
"""The logical axes of the tensors the common transformers conventions name,
in their torch orientation. A rule applies to a tensor its glob matches and
whose rank equals the length of its axes; the first such rule wins."""


@dataclass(frozen=True)
class TorchLayout(Layout):
    """A `Layout` for a tree keyed by torch parameter names.

    `names` maps a glob over a tensor's torch name to its logical axes;
    `rules` then map those onto the mesh exactly as they do for a native
    model, so the tensor axis, fsdp, `min_shard` and `check` mean what they
    mean there. A tensor no name matches takes `Layout`'s shape heuristic.
    Optimizer moments and EMA copies end in their parameter's name, so they
    take its placement.

    The placement decides where GSPMD keeps each tensor, not what the forward
    computes: a spec that suits the model badly costs collectives, never
    correctness.
    """

    names: tuple[tuple[str, LogicalAxes], ...] = TORCH_AXES

    def axes(self, name: str, ndim: int) -> LogicalAxes | None:
        """The logical axes the first matching rule gives `name`, or None."""
        for pattern, axes in self.names:
            if len(axes) == ndim and fnmatch.fnmatchcase(name, pattern):
                return axes
        return None

    def shardings[TreeT](self, mesh: Mesh, tree: TreeT) -> Placement[TreeT]:
        placed = super().shardings(mesh, tree)
        sharded_devices = math.prod(mesh.shape[axis] for axis in parameter_axes(mesh.axis_names))

        def leaf_sharding(path, value, heuristic: NamedSharding) -> NamedSharding:
            names = parameter_path(path)
            axes = self.axes(names[-1], len(value.shape)) if names else None
            if axes is None:
                return heuristic
            if sharded_devices == 1 or math.prod(value.shape) < self.min_shard:
                return NamedSharding(mesh, P())
            return NamedSharding(
                mesh, logical_spec(axes, tuple(value.shape), rules=self.axis_rules, mesh=mesh)
            )

        return jax.tree_util.tree_map_with_path(leaf_sharding, nn.unbox(tree), placed)


@dataclass(frozen=True, eq=False)
class TorchGraph:
    """The transformers module torchax runs, as a function of its tensors.

    `head` names the output head's weight when the logits are exactly
    `hidden @ head.T` and train as that product, checked at load
    (`_plain_head`); None when the model adds anything after it (a bias,
    zero or not, a scale, a softcap). `persistent` names the buffers the source
    checkpoint stores. A training forward runs the module's own training
    forward, its dropout and other draws taking `key` as torchax's random
    key; one that draws on the host instead (OPT's layer drop,
    `torch.rand([])`) is decided once at trace time, and is refused.
    """

    module: JittableModule
    head: str | None
    persistent: tuple[str, ...]

    def __call__(self, params: Mapping[str, jax.Array], buffers: Mapping[str, jax.Array],
                 tokens: jax.Array, *, train: bool, key: jax.Array | None = None
                 ) -> tuple[jax.Array, jax.Array]:
        """The logits and the output head's input states for `tokens`."""
        import torchax

        env = torchax.default_env()
        draws = _Draws()
        arguments = env.j2t_iso((dict(params), dict(buffers), tokens))
        self.module.train(train)
        try:
            with env, env.override_property(prng=key) if key is not None else contextlib.nullcontext(), draws:
                outputs = self.module.functional_call(_forward, *arguments)
        finally:
            self.module.eval()
        if draws.host:
            raise ValueError(
                f"fallback='torchax' traces this model's training forward once, and it draws on the host "
                f"({draws.host} draws, e.g. a layer drop), which tracing would fix for every step; set "
                f"that rate to 0 in its config, or train a family Dew loads natively")
        if draws.device and key is None:
            raise ValueError("this model's training forward draws (dropout) and was given no 'dropout' rng")
        logits, hidden = env.t2j_iso(outputs)
        return logits, hidden

    def export(self, model: nn.Module, variables: Mapping[str, object],
               config: Mapping[str, object]) -> dict[str, np.ndarray]:
        """The source's tensors under its own names: the parameters and the
        buffers its checkpoint stores. `Pretrained.save` writes them."""
        params = variables["params"]
        buffers = variables.get("buffers", {})
        if not isinstance(params, Mapping) or not isinstance(buffers, Mapping):
            raise TypeError("a torchax model's variables are {'params': ..., 'buffers': ...} "
                            "keyed by torch names, as Pretrained.load returns them")
        tensors = {name: np.asarray(leaf) for name, leaf in params.items()}
        tensors.update({name: np.asarray(buffers[name]) for name in self.persistent})
        return tensors


def _Draws():
    """A torch function mode counting the calls of torch's random functions,
    on torchax's device (`device`) and on the host (`host`)."""
    import torch
    import torchax
    from torch.overrides import TorchFunctionMode

    drawing = {torch.rand, torch.rand_like, torch.randn, torch.randn_like, torch.randint, torch.randint_like,
               torch.randperm, torch.bernoulli, torch.multinomial, torch.normal, torch.Tensor.uniform_,
               torch.Tensor.normal_, torch.Tensor.bernoulli_, torch.Tensor.random_, torch.Tensor.exponential_,
               torch.Tensor.cauchy_, torch.Tensor.geometric_, torch.Tensor.log_normal_}

    class Draws(TorchFunctionMode):
        device = host = 0

        def __torch_function__(self, func, types, args=(), kwargs=None):
            tensor = func(*args, **(kwargs or {}))
            if func in drawing:
                if isinstance(tensor, torchax.tensor.Tensor):
                    self.device += 1
                else:
                    self.host += 1
            return tensor

    return Draws()


def _forward(model: PreTrainedModel, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run `model` over `tokens`, keeping what its output head reads."""
    import torch

    states: list[torch.Tensor] = []
    head = model.get_output_embeddings()
    if not isinstance(head, torch.nn.Module):
        raise ValueError(f"{type(model).__name__} has no output embeddings for a causal LM's logits")
    hook = head.register_forward_pre_hook(lambda _, inputs: states.append(inputs[0]))
    try:
        logits = model(input_ids=tokens, use_cache=False).logits
    finally:
        hook.remove()
    return logits, states[0]


class TorchCausalLM(nn.Module):
    """A transformers causal LM as a Flax module, run by torchax.

    `apply(variables, tokens)` returns fp32 logits. The variables are the
    source's own tensors under their torch names, `params` and `buffers`;
    `init` has no initializer to draw from, so the module starts from the
    loaded variables only. Floating parameters are cast to `dtype` for the
    forward, which keeps them in their stored dtype as masters; buffers keep
    theirs. A training forward runs the module's own, its dropout drawing
    from the 'dropout' rng (`TorchGraph`).

    A loss scores its whole logits (`Logits`). A model whose logits are the
    plain head product loads as `TorchTableLM` instead, which `LMObjective`
    scores through its head input and head weight. The reads take plain
    token rows: transformers' forward runs with no packing columns and no
    cache, so a packing field is refused by name and there is no cached decode.
    """

    graph: TorchGraph
    vocab_size: int
    dtype: Dtype | None = None
    final_logit_softcap: float | None = None
    precision: PrecisionLike = None

    @property
    def bank_sites(self) -> tuple[DecoderBank, ...]:
        return ()

    @property
    def causal(self) -> bool:
        """True: `load` builds the model with `AutoModelForCausalLM`, whose
        architectures all predict each token from the ones before it."""
        return True

    @property
    def mask_token_id(self) -> int | None:
        """None: a causal LM is not trained to fill masked tokens."""
        return None

    def __call__(self, tokens, train: bool = False) -> jax.Array:
        logits, _ = self._run(tokens, train)
        return logits.astype(jnp.float32)

    def logits(self, tokens, *, train: bool = False, **packing) -> jax.Array:
        """The fp32 logits of the whole rows, as `__call__` returns them."""
        _refuse_packing(packing)
        return self(tokens, train=train)

    def _run(self, tokens, train: bool) -> tuple[jax.Array, jax.Array]:
        if self.is_initializing():
            raise ValueError(
                "a torchax model has no initializer; apply it to the variables "
                "Pretrained.load returned, or pass them to LMObjective as variables=")
        held = self.variables
        params = {name: self._compute(leaf) for name, leaf in held["params"].items()}
        buffers = {name: jnp.asarray(leaf) for name, leaf in held.get("buffers", {}).items()}
        key = self.make_rng("dropout") if train and self.has_rng("dropout") else None
        return self.graph(params, buffers, jnp.asarray(tokens), train=train, key=key)

    def _compute(self, leaf) -> jax.Array:
        value = jnp.asarray(leaf)
        if self.dtype is None or not jnp.issubdtype(value.dtype, jnp.floating):
            return value
        return value.astype(self.dtype)


class TorchTableLM(TorchCausalLM):
    """A `TorchCausalLM` whose logits are its head input times its head weight
    (`TorchGraph.head`), the pair `LMObjective` scores through, so the
    vocabulary-sized logits are never built whole."""

    def hidden_states(self, tokens, train: bool = False, **packing) -> jax.Array:
        """The states the output head reads, `[B, S, features]`."""
        _refuse_packing(packing)
        _, hidden = self._run(tokens, train)
        return hidden

    def output_table(self) -> OutputTable:
        """The output head's `[vocab, features]` torch weight as the bound params
        hold it, the parameter a loss keeps, with the forward's compute dtype,
        which casts the product's operands as the forward casts them whatever
        dtype the states arrive in (`head_product`)."""
        if self.graph.head is None:
            raise ValueError("this model's logits are not its plain head product; load it as TorchCausalLM")
        return OutputTable(self.variables["params"][self.graph.head], vocab_major=True,
                           precision=self.precision, compute=self.dtype)


def _refuse_packing(packing: Mapping[str, object]) -> None:
    """Refuse every field past the tokens: transformers' forward takes whole
    plain rows, with no packing columns and no cache."""
    if packing:
        raise ValueError(
            f"a torchax model runs transformers' forward over whole plain token rows, with no cache, "
            f"which takes no {sorted(packing)}; train it on unpacked rows without packing columns")


def load(name_or_dir: str | Path, directory: Path, revision: str | None, *, dtype: str,
         param_dtype: str, attention_impl: str, max_seq_len: int | None) -> PretrainedFallback:
    """Load `directory`'s causal LM through transformers and torchax.

    `directory` is the metadata snapshot `Pretrained.load` resolved and
    `revision` its commit (None for a local directory); the weights come
    from the same commit through `sources.snapshot`, so the file
    selection is the native loader's.
    """
    if attention_impl != "auto":
        raise ValueError(
            f"attention_impl={attention_impl!r} selects a Dew kernel, and fallback='torchax' "
            "runs transformers' own attention; leave attention_impl at 'auto'")
    if max_seq_len is not None:
        raise ValueError(
            "max_seq_len sizes Dew's decode cache and rotary table, which fallback='torchax' "
            "does not build; leave it unset and feed rows no longer than the model's "
            "max_position_embeddings")
    if not (directory / "config.json").is_file():
        raise FileNotFoundError(
            f"fallback='torchax' builds the model a transformers config.json describes, and "
            f"{name_or_dir} has none; load a repo that ships config.json and its weights")
    config = json.loads((directory / "config.json").read_text())
    quantization = config.get("quantization_config")
    if quantization is not None:
        method = quantization.get("quant_method") if isinstance(quantization, Mapping) else None
        raise ValueError(
            f"fallback='torchax' reads float weights only, and this source declares a "
            f"{method or 'quantization'} quantization_config; load the model's BF16 or FP32 "
            f"release, or drop fallback for a registered family whose codec Dew reads")
    try:
        import torch
        from torchax.interop import JittableModule, extract_all_buffers
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise ImportError(f"fallback='torchax' needs torch, torchax and transformers: {INSTALL}") from error
    directory = sources.snapshot(str(name_or_dir), revision)
    # Before transformers, whose reader fails on a malformed file without naming it.
    generation_config = read_generation_config(directory)
    storage = "auto" if param_dtype == AUTO else getattr(torch, param_dtype)
    model = AutoModelForCausalLM.from_pretrained(str(directory), dtype=storage, local_files_only=True)
    model.eval()
    jittable = JittableModule(model)
    head = model.get_output_embeddings()
    if not isinstance(head, torch.nn.Module):
        raise ValueError(f"{type(model).__name__} has no output embeddings for a causal LM's logits")
    head_weight = head.weight
    if not isinstance(head_weight, torch.Tensor):
        raise ValueError(f"{type(model).__name__}'s output head holds no weight tensor")
    head_name = next((name for name, value in jittable.params.items() if value is head_weight), None)
    if head_name is not None and not _plain_head(model, head_weight):
        head_name = None
    # Host views of torch's storage: nothing reaches a device before the
    # caller's placement.
    params = {name: host_view(leaf, name) for name, leaf in jittable.params.items()}
    buffers = {name: host_view(leaf, name) for name, leaf in extract_all_buffers(model)[1].items()}
    stored = model.state_dict()
    persistent = tuple(name for name in buffers if name in stored)
    vocab = int(head_weight.shape[0])
    # Torch's copies are no longer read: the forward runs on the arrays above.
    torch.nn.Module.to(model, "meta")
    graph = TorchGraph(jittable, head_name, persistent)
    module = (TorchCausalLM if head_name is None else TorchTableLM)(graph, vocab, dtype=resolve_dtype(dtype))
    source = str(name_or_dir) if revision is None else f"{name_or_dir}@{revision}"
    processor = source_processor(directory, config, config, module, source=source)
    versions = {name: importlib.metadata.version(name) for name in ("transformers", "torchax", "torch")}
    warnings.warn(
        f"{name_or_dir} loads as tier 3 (fallback='torchax'): transformers {versions['transformers']}'s "
        f"PyTorch forward lowered by torchax {versions['torchax']}, pinned to torch "
        f"{versions['torch']}; no Dew kernels, sharding rules beyond TorchLayout's name table, "
        f"or KV-cache generation; the load holds the torch model on the host, at least 2x "
        f"the checkpoint", stacklevel=3)
    built = {**config, "dtype": dtype, "fallback": "torchax", **versions}
    return PretrainedFallback(module, {"params": params, "buffers": buffers}, processor, config, directory,
                              built, generation_config, export_adapter=graph.export, revision=revision)


def _plain_head(model: PreTrainedModel, weight: torch.Tensor) -> bool:
    """Whether `model`'s logits are its head input times the head weight, in
    value and in training.

    Both sides run the same torch matmul on the same states, so a scale or a
    cap the model applies breaks the values' equality. A term whose value
    happens to vanish does not (Phi's lm_head.bias starts at zero), so the
    gradients of one random weighting of the two must be equal too, for every
    parameter: a bias, a trainable scale or anything past the head input
    takes gradient through the logits and none through the product.
    """
    import torch

    tokens = torch.arange(min(8, int(weight.shape[0])))[None]
    trained = [parameter for parameter in model.parameters() if parameter.requires_grad]
    with torch.enable_grad():
        logits, hidden = _forward(model, tokens)
        product = torch.nn.functional.linear(hidden, weight)
        if not torch.equal(logits, product):
            return False
        cotangent = torch.randn(logits.shape, generator=torch.Generator().manual_seed(0), dtype=logits.dtype)
        through = torch.autograd.grad((logits * cotangent).sum(), trained, retain_graph=True,
                                      allow_unused=True)
        plain = torch.autograd.grad((product * cotangent).sum(), trained, allow_unused=True)
    return all((a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))
               for a, b in zip(through, plain, strict=True))
