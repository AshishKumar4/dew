"""Tier 3: a transformers PyTorch causal LM, lowered to JAX by torchax.

`load_pretrained(source, fallback="torchax")` builds the model with
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

from dew.interop import hf_decoders as decoders
from dew.interop.pickles import host_view
from dew.interop.pretrained import AUTO, Pretrained, _source_processor
from dew.nn.sharding import LogicalAxes, parameter_path
from dew.registry import resolve_dtype
from dew.training.distributed import PARAMETER_AXES, Layout, Placement, _mesh_spec, _rule_table

if TYPE_CHECKING:
    import torch
    from torchax.interop import JittableModule
    from transformers import PreTrainedModel

    from dew.nn.backbones.causal_transformer import DecoderBank

INSTALL = "pip install 'dew-ml[torchax]'"

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
        rules = _rule_table(self.rules)
        sharded_devices = math.prod(mesh.shape[axis] for axis in PARAMETER_AXES)

        def leaf_sharding(path, value, heuristic: NamedSharding) -> NamedSharding:
            names = parameter_path(path)
            axes = self.axes(names[-1], len(value.shape)) if names else None
            if axes is None:
                return heuristic
            if sharded_devices == 1 or math.prod(value.shape) < self.min_shard:
                return NamedSharding(mesh, P())
            return NamedSharding(mesh, _mesh_spec(value.shape, axes, rules, mesh))

        return jax.tree_util.tree_map_with_path(leaf_sharding, nn.unbox(tree), placed)


@dataclass(frozen=True, eq=False)
class TorchGraph:
    """The transformers module torchax runs, as a function of its tensors.

    `head` names the output head's weight when the logits are exactly
    `hidden @ head.T`, checked at load; None when the model adds anything
    (a bias, a scale, a softcap). `persistent` names the buffers the source
    checkpoint stores.
    """

    module: JittableModule
    head: str | None
    persistent: tuple[str, ...]

    def __call__(self, params: Mapping[str, jax.Array], buffers: Mapping[str, jax.Array],
                 tokens: jax.Array) -> tuple[jax.Array, jax.Array]:
        """The logits and the output head's input states for `tokens`."""
        import torchax

        env = torchax.default_env()
        arguments = env.j2t_iso((dict(params), dict(buffers), tokens))
        with env:
            outputs = self.module.functional_call(_forward, *arguments)
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
                            "keyed by torch names, as load_pretrained returns them")
        tensors = {name: np.asarray(leaf) for name, leaf in params.items()}
        tensors.update({name: np.asarray(buffers[name]) for name in self.persistent})
        return tensors


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
    theirs. The torch module runs in eval mode, so dropout is off whatever
    `train` says.

    `hidden_states` and `head_weight` are the pair `LMObjective` scores
    through, so the vocabulary-sized logits are never built whole. Only a
    model whose logits are the plain head product has them (`TorchGraph`).
    """

    graph: TorchGraph
    vocab_size: int
    dtype: Dtype | None = None
    final_logit_softcap: float | None = None
    precision: PrecisionLike = None

    @property
    def bank_sites(self) -> tuple[DecoderBank, ...]:
        return ()

    def __call__(self, tokens, train: bool = False) -> jax.Array:
        logits, _ = self._run(tokens)
        return logits.astype(jnp.float32)

    def hidden_states(self, tokens, train: bool = False, **packing) -> jax.Array:
        """The states the output head reads, `[B, S, features]`."""
        if packing:
            raise ValueError(
                f"a torchax model runs transformers' forward over plain token rows, which "
                f"takes no {sorted(packing)}; train it on unpacked rows without packing columns")
        _, hidden = self._run(tokens)
        return hidden

    def head_weight(self, params) -> jax.Array:
        """The `[features, vocab]` output head, the transposed torch weight."""
        if self.graph.head is None:
            raise ValueError(
                "this model's logits are not its output head's plain product (the head has a "
                "bias, a scale or a cap), so LMObjective cannot score it in vocabulary chunks; "
                "train it with a loss over model.apply's logits instead")
        return jnp.asarray(params[self.graph.head]).T

    def _run(self, tokens) -> tuple[jax.Array, jax.Array]:
        if self.is_initializing():
            raise ValueError(
                "a torchax model has no initializer; apply it to the variables "
                "load_pretrained returned, or pass them to LMObjective as pretrained=")
        held = self.variables
        params = {name: self._compute(leaf) for name, leaf in held["params"].items()}
        buffers = {name: jnp.asarray(leaf) for name, leaf in held.get("buffers", {}).items()}
        return self.graph(params, buffers, jnp.asarray(tokens))

    def _compute(self, leaf) -> jax.Array:
        value = jnp.asarray(leaf)
        if self.dtype is None or not jnp.issubdtype(value.dtype, jnp.floating):
            return value
        return value.astype(self.dtype)


def load(name_or_dir: str | Path, directory: Path, revision: str | None, *, dtype: str,
         param_dtype: str, attention_impl: str, max_seq_len: int | None) -> Pretrained:
    """Load `directory`'s causal LM through transformers and torchax.

    `directory` is the metadata snapshot `load_pretrained` resolved and
    `revision` its commit (None for a local directory); the weights come
    from the same commit through `hf_decoders._snapshot`, so the file
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
    directory = decoders._snapshot(str(name_or_dir), revision)
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
    module = TorchCausalLM(graph, vocab, dtype=resolve_dtype(dtype))
    processor = _source_processor(directory, config, config, module)
    generation_path = directory / "generation_config.json"
    generation_config = json.loads(generation_path.read_text()) if generation_path.exists() else {}
    versions = {name: importlib.metadata.version(name) for name in ("transformers", "torchax", "torch")}
    warnings.warn(
        f"{name_or_dir} loads as tier 3 (fallback='torchax'): transformers {versions['transformers']}'s "
        f"PyTorch forward lowered by torchax {versions['torchax']}, pinned to torch "
        f"{versions['torch']}; no Dew kernels, sharding rules beyond TorchLayout's name table, "
        f"or KV-cache generation; the load holds the torch model on the host, at least 2x "
        f"the checkpoint", stacklevel=3)
    built = {**config, "dtype": dtype, "fallback": "torchax", **versions}
    return Pretrained(module, {"params": params, "buffers": buffers}, processor, config, directory,
                      built, generation_config, export_adapter=graph.export, revision=revision)


def _plain_head(model: PreTrainedModel, weight: torch.Tensor) -> bool:
    """Whether `model`'s logits are exactly its head input times the head weight.

    Both sides run the same torch matmul on the same states, so any bias,
    scale or cap the model applies breaks equality, and nothing else does.
    """
    import torch

    tokens = torch.arange(min(8, int(weight.shape[0])))[None]
    with torch.no_grad():
        logits, hidden = _forward(model, tokens)
        return bool(torch.equal(logits, torch.nn.functional.linear(hidden, weight)))
