"""The decoder's field validation: each field, or pair of fields, a `CausalTransformer` cannot build
is refused when it sets up, by `refuse_unbuildable`, naming the field the caller set."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..mixers import AttentionMixer
from ..mixers.mamba2 import Mamba2Mixer
from ..mixers.mlp import MLPMixer

if TYPE_CHECKING:
    from .causal_transformer import CausalTransformer
    from .layer_plan import ResolvedKind


def _lm_engine(model: CausalTransformer, kinds: Mapping[str, ResolvedKind]):
    """Raise `ValueError` if lm-engine's multipliers or initialisation are set on a model that
    cannot apply them.

    `embedding_multiplier` and `logits_scaling` act at the root, on the embeddings before the
    first layer (`scaled_embeddings`) and on the final states, so any stack of blocks takes
    them; prediction depths and the DSpark drafter embed their own tokens unscaled and score
    their own states. `residual_multiplier` and `initializer_range` act inside lm-engine's
    dense and routed blocks."""
    root = model.embedding_multiplier != 1.0 or model.logits_scaling != 1.0
    block = model.residual_multiplier != 1.0 or model.initializer_range is not None
    if (root or block) and (model.num_nextn_predict_layers or model.dspark is not None):
        raise ValueError(
            "embedding_multiplier, residual_multiplier, logits_scaling and initializer_range "
            "are lm-engine's; prediction depths and the DSpark drafter do not carry them")
    if block and (model.hyper_connections is not None
                  or (model.mixture is not None and model.mixture.parallel)
                  or model.mlp == 'swigluoai' or model.laurel_rank is not None
                  or model.altup is not None or model.per_layer_input_dim):
        raise ValueError(
            "residual_multiplier and initializer_range are lm-engine's dense and routed "
            "blocks; hyper-connection streams, Gemma 4's parallel experts, gpt-oss experts, "
            "LAuReL, AltUp and per-layer inputs do not carry them")
    if model.logits_scaling <= 0 or (model.initializer_range is not None
                                    and model.initializer_range <= 0):
        raise ValueError("logits_scaling and initializer_range are positive")
    if model.initializer_range is not None:
        for layer_type, kind in sorted(kinds.items()):
            mixer = kind.mixer or model.mixer
            if mixer is not None and not isinstance(mixer, (AttentionMixer, Mamba2Mixer)):
                raise ValueError(
                    f"initializer_range draws the attention and mamba2 mixers' "
                    f"projections; {layer_type!r} runs {type(mixer).__name__}, "
                    f"which keeps its own initializers")
    elif model.depth_scaled_init:
        raise ValueError("depth_scaled_init scales initializer_range's std; set it")

def _dropout(model: CausalTransformer, kinds: Mapping[str, ResolvedKind]) -> None:
    """Raise `ValueError` for a dropout rate outside [0, 1), or for attention dropout with a mixer that
    cannot drop probabilities."""
    for field in ("embedding_dropout_rate", "attention_dropout_rate"):
        rate = getattr(model, field)
        if not 0 <= rate < 1:
            raise ValueError(f"{field} must be within [0, 1), got {rate}")
    if model.attention_dropout_rate:
        for kind in kinds.values():
            if not isinstance(kind.mixer or model.mixer or AttentionMixer(), AttentionMixer):
                raise ValueError("attention_dropout_rate requires ordinary attention mixers")

def _classic(model: CausalTransformer):
    """Raise `ValueError` for norm, position and feed-forward combinations that no supported
    model uses."""
    if model.norm_type not in ('rms', 'layer'):
        raise ValueError(f'norm_type must be rms or layer, got {model.norm_type!r}')
    if model.norm_type == 'rms' and model.norm_bias:
        raise ValueError('norm_bias requires LayerNorm')
    if not model.first_attention_norm and not model.pre_norms:
        raise ValueError('first_attention_norm=False drops a pre-norm, so it requires pre_norms')
    if model.head_transform not in (None, 'gelu', 'gelu_exact'):
        raise ValueError("head_transform must be None, 'gelu' or 'gelu_exact'")
    if model.norm_type == 'layer' and model.scale_offset:
        raise ValueError('scale_offset describes RMSNorm weights')
    if model.position_embedding not in ('rotary', 'learned', 'alibi'):
        raise ValueError('position_embedding must be rotary, learned or alibi')
    if model.position_embedding_size is not None and (
            model.position_embedding != 'learned' or model.position_embedding_size < model.max_seq_len):
        raise ValueError('position_embedding_size requires learned positions and covers max_seq_len')
    if model.position_embedding_offset < 0 or (model.position_embedding_offset and (
            model.position_embedding != 'learned' or model.position_embedding_size is None
            or model.position_embedding_size < model.max_seq_len + model.position_embedding_offset)):
        raise ValueError('position_embedding_offset requires learned table rows past max_seq_len')
    if model.position_embedding != "rotary" and (
        model.partial_rotary_factor is not None or model.rope_scaling is not None
    ):
        raise ValueError('rotary scaling requires rotary positions')
    if model.mixture is not None and (model.mlp_bias or model.mlp in ('gelu', 'gelu_exact', 'relu')):
        raise ValueError('the routed experts require a bias-free gated MLP')
    if model.num_nextn_predict_layers and (model.norm_type != 'rms' or model.position_embedding != 'rotary'):
        raise ValueError('prediction depths require RMSNorm and rotary positions')

def _kinds(model: CausalTransformer, kinds: Mapping[str, ResolvedKind]):
    """Raise `ValueError` for a kind with invalid rotary, local-mask or grouped-head geometry."""
    for layer_type, kind in sorted(kinds.items()):
        if model.position_embedding == 'rotary' and kind.head_dim % 2:
            raise ValueError(
                "rotary positions rotate pairs, so the head dim of "
                f"{layer_type!r} must be even, got {kind.head_dim}")
        if kind.window is not None and kind.window < 1:
            raise ValueError(
                f"the window of {layer_type!r} must be positive, got {kind.window}")
        if kind.chunk is not None and kind.chunk < 1:
            raise ValueError(
                f"the chunk of {layer_type!r} must be positive, got {kind.chunk}")
        if kind.chunk is not None and kind.window is not None:
            raise ValueError(
                f"{layer_type!r} sets a window and a chunk; a local layer "
                f"reads one or the other")
        if kind.chunk is not None and not model.causal:
            raise ValueError(
                f"{layer_type!r} chunks a causal layer's keys, and this model "
                f"is not causal")
        if kind.bidirectional_window and (model.causal or kind.window is None):
            raise ValueError(
                f"{layer_type!r} keeps a window on both sides of a query, which "
                f"needs a window and a model that is not causal")
        if kind.num_kv_heads < 1 or model.num_heads % kind.num_kv_heads:
            raise ValueError(
                f"num_heads ({model.num_heads}) must be a multiple of the key/value "
                f"heads of {layer_type!r} ({kind.num_kv_heads})")

def _ngpt(model: CausalTransformer, kinds: Mapping[str, ResolvedKind]):
    """Raise `ValueError` for what nGPT's arithmetic does not reach: its gains and unit norms
    live in the attention and the dense gated feed-forward (an MLP mixer's too), so routed or
    parallel experts, another mixer, prediction depths and the DSpark drafter, which would
    compute without them, are refused."""
    if model.ngpt is None:
        return
    if model.mixture is not None or model.num_nextn_predict_layers or model.dspark is not None:
        raise ValueError("nGPT unit-norms the attention and the dense gated feed-forward; routed or "
                         "parallel experts (mixture), prediction depths and the DSpark drafter take "
                         "none of it")
    for layer_type, kind in sorted(kinds.items()):
        mixer = kind.mixer or model.mixer
        dense = isinstance(mixer, MLPMixer) and mixer.mixture is None
        if mixer is not None and not (isinstance(mixer, AttentionMixer) or dense):
            raise ValueError(f"nGPT's gains and unit norms are the attention's and the dense gated "
                             f"feed-forward's; {layer_type!r} runs {mixer!r}")

def refuse_unbuildable(model: CausalTransformer, kinds: Mapping[str, ResolvedKind]):
    """Raise `ValueError` for a field, or a pair of fields, that this model cannot build.

    Each error names the field the caller set and what a model without it
    looks like, so whoever translated the config can see which entry to fix.
    """
    _lm_engine(model, kinds)
    _dropout(model, kinds)
    _classic(model)
    _ngpt(model, kinds)
    mtp_hc = model.mtp_hyper_connections
    if mtp_hc is not None:
        if model.num_nextn_predict_layers != 1:
            raise ValueError("mtp_hyper_connections requires a single prediction depth")
        if mtp_hc.head != 'weighted':
            raise ValueError("mtp_hyper_connections requires the V4 weighted stream-collapse head")
        if model.hyper_connections is None or mtp_hc.hc_mult != model.hyper_connections.hc_mult:
            raise ValueError("mtp_hyper_connections must match the trunk's residual stream count")
    _kinds(model, kinds)
    carried = [name for name in ("altup", "hyper_connections", "attention_residuals", "dspark")
               if getattr(model, name) is not None]
    if model.loop is not None and carried:
        raise ValueError(f"a loop norms the one residual between its passes, and {carried} carry "
                         "others beside it; a model with them runs its stack once (loop=None)")
    if model.loop is not None and model.loop.layers is not None:
        if model.loop.layers[1] > model.num_layers:
            raise ValueError(f"the loop's block {model.loop.layers} ends past layer {model.num_layers}")
        if model.sharing_layers:
            raise ValueError("a loop over a block reads its layers' keys and values apart from the rest, "
                             "and kv_shared_layers reads them across it; loop the whole stack")
    if model.num_heads % model.kv_heads:
        raise ValueError(
            f"num_heads ({model.num_heads}) must be a multiple of num_kv_heads "
            f"({model.kv_heads})")
    if model.partial_rotary_factor is not None and not 0 < model.partial_rotary_factor <= 1:
        raise ValueError(
            "partial_rotary_factor must be within (0, 1], "
            f"got {model.partial_rotary_factor}")
    if model.partial_rotary_type not in ('proportional', 'default'):
        raise ValueError(
            "partial_rotary_type names the convention of a partial rotary, "
            f"'proportional' or 'default', got {model.partial_rotary_type!r}")
    sparse = model.sparse_layers
    outside = sorted(index for index in sparse
                     if not 0 <= index < model.num_layers)
    if outside:
        raise ValueError(
            f"the mixture's layers {outside} are outside the "
            f"{model.num_layers} layers of this model")
    hashed = model.hash_layers
    if hashed - set(sparse):
        raise ValueError(
            f"the mixture's hash_layers {sorted(hashed - set(sparse))} are not "
            "among its sparse layers, which are the ones with experts to route")
    sharing = model.kv_sharing
    if model.use_double_wide_mlp and not sharing:
        raise ValueError(
            "use_double_wide_mlp widens the MLP of the layers that share "
            "their keys and values, so it needs kv_shared_layers set")
    ple = model.per_layer_input_dim
    if ple is not None and ple < 1:
        raise ValueError(
            f"per_layer_input_dim is the width of a layer's own input, got "
            f"{ple}; None is a model without them")
    if model.num_nextn_predict_layers < 0:
        raise ValueError(
            f"num_nextn_predict_layers counts prediction depths, got "
            f"{model.num_nextn_predict_layers}; 0 is a model without them")
    widths = model.mlp_widths
    if len(widths) != model.num_layers or min(widths) < 0:
        raise ValueError(
            f"mlp_features names one width per layer of {model.num_layers}, "
            f"0 for a layer without a feed-forward, got {model.mlp_features}")
    sparsity = model.activation_sparsity_pattern
    if sparsity is not None and (len(sparsity) != model.num_layers
                                 or not all(0 <= fraction < 1 for fraction in sparsity)):
        raise ValueError(
            f"activation_sparsity_pattern names one fraction within [0, 1) per "
            f"layer of {model.num_layers}, got {sparsity}")
    if model.altup is not None and model.num_nextn_predict_layers:
        raise ValueError(
            "altup carries a stack of residual copies through the layers and "
            "the prediction depths read one, so a model has one or the other")
    if model.altup is not None and model.hyper_connections is not None:
        raise ValueError(
            "altup and hyper_connections each carry their own stack of residual "
            "copies through the layers, so a model has one or the other")
    _streams(model)
    _cross_attention(model)
    if model.swiglu_limit is not None and model.swiglu_limit <= 0:
        raise ValueError(
            f"swiglu_limit caps the gate and up projections, so it is positive, "
            f"got {model.swiglu_limit}; None leaves them unclamped")
    mask = model.mask_token_id
    if mask is not None and (isinstance(mask, bool) or not isinstance(mask, int) or mask < 0):
        raise ValueError(
            f"mask_token_id names a vocabulary id, got {mask!r}; None is a "
            "model trained on plain next-token prediction")

def _cross_attention(model: CausalTransformer):
    """Raise `ValueError` where `cross_attention` names a layer this model lacks, or one whose block
    it cannot build (`CrossBlocks.block`)."""
    cross = model.cross_attention
    if cross is None:
        return
    outside = sorted(set(cross.layer_ids) - set(range(model.num_layers)))
    if outside:
        raise ValueError(f"cross-attention layers {outside} are outside the {model.num_layers} layers")
    for index in cross.layer_ids:
        cross.block(model, index)

def _streams(model: CausalTransformer):
    """Raise `ValueError` for settings on mHC's residual streams that cannot be
    built: engram's layers, DSpark's stages and the Single-Pass schedule."""
    engram, hc = model.engram, model.hyper_connections
    if engram is not None:
        if hc is None:
            raise ValueError("engram gates its lookup into each of mHC's residual streams, "
                             "so it needs hyper_connections")
        outside = sorted(set(engram.layer_ids) - set(range(model.num_layers)))
        if outside:
            raise ValueError(f"engram layers {outside} are outside the {model.num_layers} layers")
    if model.dspark is not None:
        # The stages are the trunk's mHC blocks, V4.1's Single-Pass ones
        # (v41:1100-1156) or V4's with a learned head on the last stage
        # (V4-Flash-0731 model.py:818-874); a drafter for another trunk's
        # residual is not built here.
        if hc is None or not (hc.single_pass or hc.head == 'weighted') or model.mixture is None:
            raise ValueError("the DSpark drafter chains mHC blocks over routed experts and "
                             "collapses them by Single-Pass's carried pre (V4.1) or by a "
                             "learned head (V4), as the trunk it drafts for does")
        outside = sorted(set(model.dspark.target_layers) - set(range(model.num_layers)))
        if outside:
            raise ValueError(f"DSpark target layers {outside} are outside the {model.num_layers} layers")
        if model.num_nextn_predict_layers:
            raise ValueError("a model drafts with DSpark or with MTP depths, not both")
    if hc is not None and hc.single_pass and model.mtp_hyper_connections:
        raise ValueError("Single-Pass mHC carries the next sublayer's collapse beside the "
                         "streams, which no stream prediction depth reads")
