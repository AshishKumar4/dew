"""CPU Qwen MTP reference composition over the actual Transformers decoder.

vLLM qwen3_5_mtp.py at 51da0ca66c8065619c79e35dff97aa99aeaf5644,
Qwen3_5MultiTokenPredictor.forward, defines embedding norm, hidden norm,
embedding-first concatenation, fc, one full-attention decoder layer and norm;
qwen3_next_mtp.py's Qwen3NextMultiTokenPredictor.forward is the same
composition over the Qwen3-Next classes (its rotary takes one position row,
Qwen3.5's the three mRoPE rows). The linears use torch.nn and the
layer/rotary/norm classes come directly from Transformers 5.16.1. This
executes the published composition on CPU; it does not claim to execute
vLLM's tensor-parallel kernels or scheduler.
"""

import copy

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextDecoderLayer, Qwen3NextRMSNorm, Qwen3NextRotaryEmbedding,
)

# (decoder layer, norm, rotary, position rows the rotary takes) by model_type.
_CLASSES = {
    "qwen3_5_text": (Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding, 3),
    "qwen3_5_moe_text": (Qwen3_5MoeDecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding, 3),
    "qwen3_next": (Qwen3NextDecoderLayer, Qwen3NextRMSNorm, Qwen3NextRotaryEmbedding, 1),
}


class QwenMTP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg = copy.deepcopy(config)
        cfg.layer_types = ["full_attention"]
        cfg.num_hidden_layers = 1
        cfg._attn_implementation = "eager"
        layer_type, norm, rotary, self.position_rows = _CLASSES[cfg.model_type]
        self.fc = torch.nn.Linear(cfg.hidden_size * 2, cfg.hidden_size, bias=False)
        self.pre_fc_norm_embedding = norm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = norm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.layers = torch.nn.ModuleList([layer_type(cfg, 0)])
        self.norm = norm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rotary = rotary(cfg)

    def forward(self, hidden, embeds, positions, valid):
        merged = self.fc(torch.cat([self.pre_fc_norm_embedding(embeds),
                                   self.pre_fc_norm_hidden(hidden)], dim=-1))
        if positions.ndim == 2 and self.position_rows == 3:
            positions = positions[None].expand(3, -1, -1)
        position_embeddings = self.rotary(merged, positions)
        length = merged.shape[1]
        causal = torch.arange(length)[:, None] >= torch.arange(length)[None]
        allowed = causal[None, None] & valid[:, None, None, :]
        mask = torch.zeros(allowed.shape, dtype=merged.dtype).masked_fill(~allowed, torch.finfo(merged.dtype).min)
        return self.norm(self.layers[0](merged, position_embeddings=position_embeddings,
                                        attention_mask=mask, use_cache=False))


def prediction_inputs(model, encoded):
    """Capture the actual decoder embeddings, including projected image features."""
    text = model.model.language_model if hasattr(model.model, "language_model") else model.model
    captured = {}

    def capture(module, args, kwargs):
        embeddings = kwargs.get("inputs_embeds")
        captured["embeddings"] = embeddings if embeddings is not None else module.get_input_embeddings()(kwargs["input_ids"])
        captured["positions"] = kwargs.get("position_ids")

    handle = text.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        outputs = model(**encoded, use_cache=False, output_hidden_states=True)
    finally:
        handle.remove()
    hidden = outputs.hidden_states[-1]
    positions = captured["positions"]
    if positions is None:
        positions = encoded["attention_mask"].long().cumsum(-1) - 1
    return outputs.logits, hidden, captured["embeddings"], positions
