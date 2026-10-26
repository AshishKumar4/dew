"""Measure `dew.interop.verify._GRADIENT_PARITY`: how much farther transformers' float32 gradient of
every parameter is from the float64 truth than Dew's on tier 2's probe (`verify.gradient_parity`), over
the causal LM types transformers builds whose every part Dew knows (`dew.interop.parts`) and the
compositions of those parts tier 2 admits (`COMPOSITIONS`).

Each is built at the probe's shrunken sizes with the probe's weights (`verify.shrink`,
`verify.scatter_weights`), assembled from parts, and its gradients compared as the probe compares them
(`verify.gradient_parity`). Prints each one's measure and the largest; the probe's bound is twice the
largest, as its logits' is twice `_ROUNDING`.

    python tools/gradient_parity.py
"""
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from dew.interop import hf_decoders, parts, sources, verify

TYPES = ("llama", "mistral", "mixtral", "qwen2", "qwen3", "qwen3_moe", "gemma", "olmo", "helium", "seed_oss",
         "smollm3", "cwm", "olmoe")
"""The types measured: transformers' classes of the registered families' parts and those built of them."""

FIELDS = {"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
          "head_dim": 16, "intermediate_size": 96, "vocab_size": 256, "max_position_embeddings": 64,
          "tie_word_embeddings": False, "pad_token_id": None, "bos_token_id": 1, "eos_token_id": 2}
"""Each type's config before the probe shrinks it, its special ids within its vocabulary."""


def _mixed(config) -> torch.nn.Module:
    """Llama's attention, Mixtral's routed experts and Gemma's (1 + w) norms in one decoder, as
    tests/test_tier2_parts.py's remote source composes them."""
    from transformers.models.gemma.modeling_gemma import GemmaRMSNorm
    from transformers.models.llama.modeling_llama import LlamaForCausalLM
    from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

    config.num_local_experts, config.num_experts_per_tok, config.router_jitter_noise = 4, 2, 0.0
    model = LlamaForCausalLM(config)
    for layer in model.model.layers:
        layer.mlp = MixtralSparseMoeBlock(config)
        layer.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        layer.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    model.model.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    return model


COMPOSITIONS = {"llama+mixtral+gemma": ("llama", _mixed)}
"""Each composition of known parts measured: its base config's model_type and its builder."""


def measured(model_type: str, built=None) -> float | str:
    """The type's measure (`built` of its shrunken config, transformers' own model where None), or why
    it is not one Dew assembles from parts."""
    config = verify.shrink({"model_type": model_type, **FIELDS})
    shrunk = AutoConfig.for_model(model_type, **config)
    reference = (AutoModelForCausalLM.from_config(shrunk, dtype=torch.float32) if built is None
                 else built(shrunk).float())
    verify.scatter_weights(reference)
    reference.set_attn_implementation("eager")
    try:
        assembly = parts.assemble(reference)
    except parts.Ambiguous as ambiguous:
        return str(ambiguous)
    family = parts.family(model_type, assembly)
    with tempfile.TemporaryDirectory() as folder:
        # The module's own names, which the assembly reads, not a released format's
        # (`save_pretrained` writes Mixtral's experts one by one, as its releases stored them).
        reference.save_pretrained(folder, save_original_format=False)
        variables = hf_decoders.translate_weights(sources.load_shards(Path(folder)), assembly.fields, family)
    return verify.gradient_parity(reference, family, assembly.fields, variables)


def main() -> None:
    measures = {}
    for name, (model_type, built) in {**{name: (name, None) for name in TYPES}, **COMPOSITIONS}.items():
        result = measured(model_type, built)
        print(f"{name}: {result}")
        if isinstance(result, float):
            measures[name] = result
    worst = max(measures, key=measures.__getitem__)
    print(f"largest: {measures[worst]:.3f} ({worst}) over {len(measures)} types")


if __name__ == "__main__":
    sys.exit(main())
