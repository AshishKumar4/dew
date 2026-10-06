"""Map Hugging Face decoder tensor names onto CausalTransformer paths, and back.

`_dew_path` reads a source name into a variables path and `_hf_name` writes
one back. A family whose names differ respells them through its `Renames`, and
a `Packed` tensor is split on load and packed again on export, so one table
holds both directions.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from dew import records
from dew.interop.decoder_config import families
from dew.interop.streaming import LazyTree, SourceLeaf, WeightLayout

# Where a layer's norms sit in the two trees. Without the sandwich the names
# are the same; with it three of the four move, because HF names its norms
# after the sublayer they follow while dew names them after what they
# normalize. HF's post_attention_layernorm normalizes the attention output
# (our attention_output_norm); its pre_feedforward_layernorm is the MLP's
# pre-norm (our post_attention_layernorm); and its post_feedforward_layernorm
# normalizes the MLP output (our mlp_output_norm).
_PRE_NORMS = {
    'input_layernorm': 'input_layernorm',
    'post_attention_layernorm': 'post_attention_layernorm',
}
_SANDWICH_NORMS = {
    'input_layernorm': 'input_layernorm',
    'post_attention_layernorm': 'attention_output_norm',
    'pre_feedforward_layernorm': 'post_attention_layernorm',
    'post_feedforward_layernorm': 'mlp_output_norm',
}
_PROJECTIONS = {'self_attn': ('q_proj', 'k_proj', 'v_proj', 'o_proj'),
                'mlp': ('gate_proj', 'up_proj', 'down_proj', 'gate')}
_HEAD_NORMS = ('q_norm', 'k_norm')
# The MLA projections and norms live under self_attn beside the standard
# ones, with no counterpart in another family, so they extend the map by
# pattern. A tensor that is present maps, whatever the family.
_MLA_PROJECTIONS = ('q_a_proj', 'q_b_proj', 'kv_a_proj_with_mqa',
                    'kv_b_proj', 'o_proj')
_MLA_NORMS = ('q_a_layernorm', 'kv_a_layernorm')
# One leaf per projection for the router and the shared experts; the routed
# experts stack per-expert tensors (see _stack_experts).
_MOE_SHARED = ('gate_proj', 'up_proj', 'down_proj')
# Qwen3.5's linear_attn is the block's mixer, so it lands where self_attn
# does; its Linear leaves transpose like any other, and the rest keep the
# checkpoint's names and shapes (GatedDeltaNet in dew.nn.linear). Qwen3-Next
# stores the same projections fused as in_proj_qkvz and in_proj_ba.
_LINEAR_PROJECTIONS = ('in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a',
                       'in_proj_qkvz', 'in_proj_ba', 'out_proj')
_LINEAR_LEAVES: frozenset[tuple[str, ...]] = frozenset(
    (('conv1d', 'weight'), ('norm', 'weight'), ('A_log',), ('dt_bias',)))
# DeepSeek V4's attention leaves, nested as its modules are: the layer holds
# the query LoRA, the shared key/value head, the grouped output projection
# and the sinks (modeling_deepseek_v4.py:777-786); a compressor its two
# projections, position bias and entry norm (:379-382); the indexer those
# and its query projection (:490-496); and the indexer's scorer the head
# weights (:446-450). mHC's mixing tensors keep the reference's layout, so
# they are leaves of their own rather than kernels.
_V4_PROJECTIONS = ('q_a_proj', 'q_b_proj', 'kv_proj', 'gate_proj', 'o_a_proj',
                   'o_b_proj', 'weights_proj')
_V4_NORMS = ('q_a_norm', 'kv_norm')
_V4_MODULES = ('compressor', 'indexer', 'scorer')
_V4_TENSORS = ('sinks', 'position_bias')
_V4_HC = ('fn', 'base', 'scale')
_V4_HEAD = ('hc_fn', 'hc_base', 'hc_scale')
# The router state a training step moves (V4.1's image-span bias too), beside
# the frozen table a hash router selects by; none is a parameter, so each
# lands where `Router` keeps it (modeling_deepseek_v4.py:1033, :1062).
_MOE_STATE = ('e_score_correction_bias', 'media_bias', 'tid2eid')


def _norm_names(sandwich: bool) -> dict[str, str]:
    return _SANDWICH_NORMS if sandwich else _PRE_NORMS


def _dew_path(hf_name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """Map one HF tensor name to its path in a CausalTransformer's variables.

    The first name is the collection: `params` for a weight, `moe` for
    DeepSeek's balancing bias. That bias is router state a training step
    moves, not a parameter, so it lands where `Router` keeps it. None means
    the tensor is the tied lm_head copy. Prediction layers use their own
    family path rather than being discarded. An unexplained tensor name
    raises before any checkpoint is accepted.
    """
    parts = hf_name.split('.')
    if (len(parts) == 6 and parts[:2] == ['model', 'layers'] and parts[2].isdigit()
            and parts[3:5] == ['mlp', 'gate'] and parts[5] in _MOE_STATE):
        return ('moe', f'layers_{parts[2]}', 'mlp', 'gate', parts[5])
    path = _param_path(parts, config)
    return None if path is None else ('params', *path)


# The decoder's tensors outside its layers, read one way on load and the
# other on export.
_TRUNK: Mapping[str, tuple[str, ...]] = {
    'model.norm.weight': ('norm', 'scale'), 'model.norm.bias': ('norm', 'bias'),
    'model.embed_tokens.weight': ('embed_tokens', 'embedding'),
    'model.embed_positions.weight': ('embed_positions', 'embedding'),
    'model.embedding_layernorm.weight': ('embedding_layernorm', 'scale'),
    'model.embedding_layernorm.bias': ('embedding_layernorm', 'bias'),
    'model.embed_tokens_per_layer.weight': ('embed_tokens_per_layer', 'embedding'),
    'model.per_layer_model_projection.weight': ('per_layer_model_projection', 'kernel'),
    'model.per_layer_projection_norm.weight': ('per_layer_projection_norm', 'scale'),
}
_TRUNK_NAMES: Mapping[tuple[str, ...], str] = {path: name for name, path in _TRUNK.items()}
# Gemma 4's per-layer residual. Gate and projection are kernels, the post
# norm is a scale. The values norm carries no weight, so it maps nothing.
_PER_LAYER_INPUTS = {'per_layer_input_gate': 'kernel', 'per_layer_projection': 'kernel',
                     'post_per_layer_input_norm': 'scale'}


type Renames = tuple[tuple[str, str], ...]
"""A family's own names onto the ones `_dew_path` reads, as (source, shared)
pairs of dotted fragments: a load respells left to right, an export right to
left, so one table holds both directions."""


def _renamed(name: str, renames: Renames, *, export: bool = False) -> str:
    """Respell `name` through `renames` in one pass over its dotted parts.

    At each part the first pair whose fragment starts there replaces it and
    the pass moves past it, so a respelled part is never read again and the
    reverse pass undoes the forward one.
    """
    pairs = [(old.split('.'), new) for old, new in
             ((shared, source) if export else (source, shared) for source, shared in renames)]
    parts, spelled, index = name.split('.'), [], 0
    while index < len(parts):
        for old, new in pairs:
            if parts[index:index + len(old)] == old:
                spelled.append(new)
                index += len(old)
                break
        else:
            spelled.append(parts[index])
            index += 1
    return '.'.join(spelled)


def _renamed_path(renames: Renames, name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """`_dew_path` of a family whose names differ from the shared ones by `renames`."""
    return _dew_path(_renamed(name, renames), config)


def _renamed_name(renames: Renames, dew_name: str, config: Mapping[str, object]) -> str | None:
    """`_hf_name` respelled in the family's own names: `_renamed_path` backwards."""
    name = _hf_name(dew_name, config)
    return None if name is None else _renamed(name, renames, export=True)


def _param_path(parts: list[str], config: Mapping[str, object]) -> tuple[str, ...] | None:
    """Return the params-tree path of a split HF tensor name, or None for the tied head."""
    hf_name = '.'.join(parts)
    if hf_name in _TRUNK:
        return _TRUNK[hf_name]
    if len(parts) == 3 and parts[:2] == ['model', 'hc_head'] and parts[2] in _V4_HEAD:
        return ('hc_head', parts[2])
    if parts == ['lm_head', 'weight']:
        return None if config['tie_embeddings'] else ('lm_head', 'kernel')
    if parts == ['lm_head', 'bias']:
        return ('head_bias',)

    if len(parts) >= 4 and parts[:2] == ['model', 'layers'] and parts[2].isdigit():
        path = _layer_param_path(parts, config)
        if path is not None:
            return (f'layers_{parts[2]}', *path)
    raise ValueError(f"unknown tensor name {hf_name!r}")


def _layer_param_path(parts: list[str], config: Mapping[str, object]) -> tuple[str, ...] | None:
    """The path within one decoder layer, before adding its layer-bank name."""
    module, leaf = parts[3], parts[-1]
    if config.get('hyper_connections') is not None:
        path = _hyper_connection_param_path(parts)
        if path is not None:
            return path
    if module in _PROJECTIONS and len(parts) == 6:
        sublayer = parts[4]
        if sublayer in _PROJECTIONS[module] and leaf in ('weight', 'bias'):
            # torch Linear holds [out, in]; nn.Dense keeps [in, out]
            return (module, sublayer, 'kernel' if leaf == 'weight' else 'bias')
        if module == 'self_attn' and sublayer in _HEAD_NORMS and leaf == 'weight':
            return (module, sublayer, 'scale')
        if module == 'self_attn' and sublayer in _MLA_PROJECTIONS and leaf in ('weight', 'bias'):
            return (module, sublayer, 'kernel' if leaf == 'weight' else 'bias')
        if module == 'self_attn' and sublayer in _MLA_NORMS and leaf == 'weight':
            return (module, sublayer, 'scale')
    if module == 'self_attn':
        return _attention_param_path(parts)
    if len(parts) == 5 and module in ('attn_hc', 'ffn_hc') and leaf in _V4_HC:
        # mHC's residual mapping around each sublayer, its tensors in
        # the reference's own layout (modeling_deepseek_v4.py:902-913).
        return (module, leaf)
    if (len(parts) == 8 and module == 'mlp' and parts[4] == 'experts'
            and parts[5].isdigit() and parts[6] in _MOE_SHARED and leaf == 'weight'):
        # model.layers.N.mlp.experts.K.{gate,up,down}_proj.weight, one
        # tensor per expert, stacked by _stack_experts below.
        return ('mlp', 'experts', parts[5], parts[6], 'kernel')
    if (len(parts) == 7 and module == 'mlp' and parts[4] == 'shared_experts'
            and parts[5] in _MOE_SHARED and leaf == 'weight'):
        # The dense shared experts beside them, one MLP however many the
        # config counts.
        return ('mlp', 'shared_experts', parts[5], 'kernel')
    if (module == 'linear_attn'
            and records.strings(config['layer_types'], 'layer_types')[int(parts[2])] == 'linear_attention'):
        tail = tuple(parts[4:])
        if len(tail) == 2 and tail[0] in _LINEAR_PROJECTIONS and leaf == 'weight':
            return ('self_attn', tail[0], 'kernel')
        if tail in _LINEAR_LEAVES:
            return ('self_attn', *tail)
    if len(parts) == 5 and leaf == 'weight' and module in _PER_LAYER_INPUTS:
        return (module, _PER_LAYER_INPUTS[module])
    norms = _norm_names(bool(config.get('sandwich_norms')))
    if len(parts) == 5 and module in norms and leaf in ('weight', 'bias'):
        return (norms[module], 'scale' if leaf == 'weight' else 'bias')
    return None


def _hyper_connection_param_path(parts: list[str]) -> tuple[str, ...] | None:
    """The V4 stream mappings and attention leaves in a hyper-connected layer."""
    module = parts[3]
    if len(parts) == 4 and module.startswith(('hc_attn_', 'hc_ffn_')):
        site, suffix = module[3:].split('_', 1)
        if suffix in ('fn', 'base', 'scale'):
            return (f'{site}_hc', suffix)
    if module == 'self_attn':
        tail = tuple(parts[4:])
        if tail in (('A_log',), ('dt_bias',), ('o_norm', 'weight')):
            return (module, *tail)
        if len(tail) == 2 and tail[1] == 'weight':
            if tail[0] in ('q_conv1d', 'k_conv1d', 'v_conv1d'):
                return (module, *tail)
            if tail[0] in ('f_a_proj', 'f_b_proj', 'b_proj', 'g_a_proj', 'g_b_proj'):
                return (module, tail[0], 'kernel')
        if len(tail) == 2 and tail[0] == 'indexer' and tail[1] in (
                'index_kpool_compress_ape', 'index_kpool_compress_gate'):
            return (module, *tail)
    return None


def _attention_param_path(parts: list[str]) -> tuple[str, ...] | None:
    """The sparse selector's leaves, followed by DeepSeek V4's nested leaves."""
    if len(parts) == 7 and parts[4] == 'indexer':
        # model.layers.N.self_attn.indexer.{wq_b,wk,weights_proj}.weight
        # and k_norm.{weight,bias}: the sparse selector's own tensors.
        sublayer, leaf = parts[5], parts[6]
        if sublayer in ('wq_b', 'wk', 'weights_proj') and leaf == 'weight':
            return ('self_attn', 'indexer', sublayer, 'kernel')
        if sublayer == 'k_norm' and leaf in ('weight', 'bias'):
            return ('self_attn', 'indexer', sublayer, 'scale' if leaf == 'weight' else 'bias')
    tail = _v4_attention_leaf(parts[4:])
    return None if tail is None else ('self_attn', *tail)


def _v4_attention_leaf(tail: list[str]) -> tuple[str, ...] | None:
    """Return DeepSeek V4's own leaf under `self_attn`, or None for another family's.

    The reference nests a compressor under the layer, an indexer under a
    compressor and a scorer under the indexer, each holding projections and
    norms of the same names, so the nesting is read off the name and the
    leaf under it decides the kind (_V4_PROJECTIONS, _V4_NORMS, _V4_TENSORS).
    """
    prefix: tuple[str, ...] = ()
    while len(tail) > 1 and tail[0] in _V4_MODULES:
        prefix, tail = (*prefix, tail[0]), tail[1:]
    if len(tail) == 1 and tail[0] in _V4_TENSORS:
        return (*prefix, tail[0])
    if len(tail) == 2 and tail[1] == 'weight':
        if tail[0] in _V4_PROJECTIONS:
            return (*prefix, tail[0], 'kernel')
        if tail[0] in _V4_NORMS:
            return (*prefix, tail[0], 'scale')
    return None


def _stack_experts(params: LazyTree) -> None:
    """Stack per-expert `experts/K/projection` dicts into `[E, ...]` leaves.

    A checkpoint names one tensor per expert while the tree keeps one leaf
    per projection stacked on an expert dimension, so after the flat map
    each sparse layer's digit-keyed dicts stack in expert order. A layer
    whose experts do not form a dense `0..E-1` run refuses.
    """
    blocks = [(layer, block) for layer, block in params.items()
              if isinstance(block, dict) and layer.startswith('layers_')]
    # An MTP depth's block routes like the layer before it.
    for depth, block in params.items():
        nested = block.get('block') if isinstance(block, dict) else None
        if depth.startswith(('mtp_', 'dspark_')) and isinstance(nested, dict):
            blocks.append((depth, nested))
    slots = [(f'{layer}.{name}', slot) for layer, block in blocks for name, slot in block.items()
             if name in ('mlp', 'self_attn') and isinstance(slot, dict)]
    for layer, mlp in slots:
        experts = mlp.get('experts')
        if not isinstance(experts, dict):
            continue
        if not any(index.isdigit() for index in experts):
            continue
        indices = sorted(experts, key=int)
        if ([int(index) for index in indices]
                != list(range(len(indices)))):
            raise ValueError(
                f"{layer} experts {indices} are not a dense 0..E-1 run")
        stacked: LazyTree = {}
        first = experts[indices[0]]
        if not isinstance(first, dict):
            raise ValueError(f"{layer} expert {indices[0]} is a tensor, not projections")
        for projection in first:
            leaves = []
            for index in indices:
                expert = experts[index]
                node = expert.get(projection) if isinstance(expert, dict) else None
                leaf = node.get('kernel') if isinstance(node, dict) else None
                if not isinstance(leaf, SourceLeaf):
                    raise ValueError(f"{layer} expert {index} has no {projection} kernel")
                leaves.append(leaf)
            stacked[projection] = {'kernel': SourceLeaf.stack(leaves, f"{layer} experts' {projection}")}
        mlp['experts'] = stacked


def _hf_name(dew_name: str, config: Mapping[str, object]) -> str | None:
    """Map one flattened dew param path to its HF tensor name, or None.

    None is the tied lm_head, whose embedding copy is written instead.
    """
    parts = dew_name.split('.')
    if tuple(parts) in _TRUNK_NAMES:
        return _TRUNK_NAMES[tuple(parts)]
    if parts == ['lm_head', 'kernel']:
        return None if config['tie_word_embeddings'] else 'lm_head.weight'
    if parts == ['head_bias']:
        return 'lm_head.bias'

    if parts[0].startswith('layers_'):
        index = parts[0].removeprefix('layers_')
        module, leaf = parts[1], parts[-1]
        if len(parts) == 3 and _PER_LAYER_INPUTS.get(module) == leaf:
            return f'model.layers.{index}.{module}.weight'
        if len(parts) == 4 and module in _PROJECTIONS:
            if parts[2] in _PROJECTIONS[module] and leaf in ('kernel', 'bias'):
                return (f'model.layers.{index}.{module}.{parts[2]}.'
                        + ('weight' if leaf == 'kernel' else 'bias'))
            if module == 'self_attn' and parts[2] in _HEAD_NORMS and leaf == 'scale':
                return f'model.layers.{index}.self_attn.{parts[2]}.weight'
        theirs = {ours: hf for hf, ours in
                  _norm_names(families()[records.text(config['model_type'],
                                             'model_type')].sandwich_norms).items()}
        if len(parts) == 3 and module in theirs and leaf in ('scale', 'bias'):
            return f'model.layers.{index}.{theirs[module]}.' + ('weight' if leaf == 'scale' else 'bias')
    raise ValueError(f"unknown parameter path {dew_name!r}")


@dataclass(frozen=True)
class Packed:
    """One source tensor that holds several the path map reads: `parts`,
    concatenated on `axis`, then permuted by `transpose`.

    Each name is a suffix after the stem a tensor and its parts share. A load
    splits the tensor into views of its parts (`DecoderFamily.prepare_weights`)
    and an export packs their leaves back (`layout`), so the one entry is
    both directions.
    """

    name: str
    parts: tuple[str, ...]
    axis: int = -1
    transpose: tuple[int, ...] | None = None
    widths: Callable[[Mapping[str, object]], tuple[int, ...]] | None = None
    """Config-derived unequal part widths, as in Phi-3's grouped-query qkv."""

    def split(self, name: str, tensor: np.ndarray,
              config: Mapping[str, object] | None = None) -> dict[str, np.ndarray]:
        """The parts of the source tensor `name`, as views of it."""
        stem = name.removesuffix(self.name)
        stored = tensor if self.transpose is None else tensor.transpose(self.transpose)
        sections: int | np.ndarray = len(self.parts)
        if self.widths is not None:
            if config is None:
                raise ValueError(f'{name} needs translated geometry to split its projections')
            widths = self.widths(config)
            if len(widths) != len(self.parts) or min(widths) < 1 or sum(widths) != stored.shape[self.axis]:
                raise ValueError(f'{name} has shape {stored.shape}, incompatible with part widths {widths}')
            sections = np.cumsum(widths[:-1])
        return {stem + part: piece for part, piece in
                zip(self.parts, np.split(stored, sections, axis=self.axis), strict=True)}

    def layout(self, name: str, parts: Sequence[WeightLayout]) -> WeightLayout:
        """The layout of the source tensor `name`, from each part's one-leaf layout."""
        ndim = len(parts[0].shape)
        # A part's axis k is its leaf's axis order[k].
        order = parts[0].transpose or tuple(range(ndim))
        axis = self.axis % ndim
        shape = [*parts[0].shape]
        shape[axis] = sum(part.shape[axis] for part in parts)
        back = tuple(range(ndim)) if self.transpose is None else tuple(
            int(k) for k in np.argsort(self.transpose))
        transpose = tuple(order[k] for k in back)
        return WeightLayout(name, tuple(path for part in parts for path in part.paths),
                            tuple(shape[k] for k in back),
                            None if transpose == tuple(range(ndim)) else transpose,
                            None if len(parts) == 1 else order[axis] - ndim)


# Gemma 4, Qwen3-Next and Qwen 3.5 MoE hold their routed experts as torch
# Linears, `gate_up_proj` `[E, 2 * expert, hidden]` with the gate in the first
# rows and `down_proj` `[E, hidden, expert]`, where dew stacks `[E, in, out]`.
_FUSED_EXPERTS = (Packed('.experts.gate_up_proj', ('.experts.gate_proj', '.experts.up_proj'), -1, (0, 2, 1)),
                  Packed('.experts.down_proj', ('.experts.down_proj',), -1, (0, 2, 1)))
