"""Measure `dew.interop.verify._GRADIENT_PARITY` and `_LOGITS_PARITY`: how much farther transformers'
float32 gradient of every parameter, and its float32 logits, are from the float64 truth than Dew's on
tier 2's probe (`verify.gradient_parity`, `verify.logits_parity`).

The gradients are measured over the causal LM types transformers builds whose every part Dew knows
(`dew.interop.parts`) and the compositions of those parts tier 2 admits (`COMPOSITIONS`): each built at
the probe's shrunken sizes with the probe's weights (`verify.shrink`, `verify.scatter_weights`),
assembled from parts, and its gradients compared as the probe compares them. The logits are measured
over the same types and compositions as the probe compares them, each read as tier 2 reads it: from
parts, or a type whose attention no part is by the registered family it verifies as (`CONVENTIONS`).
Prints each one's measures and the largest; each bound is twice its largest.

    python tools/gradient_parity.py
"""
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from dew.interop import decoder_parts, hf_decoders, parts, sources, verify
from dew.interop.decoder_families import ENTRIES
from dew.registry import models

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

CONVENTIONS = {"seed_oss": "qwen2"}
"""Each type that assembles from no parts by the registered family the probe verifies it as."""


def measured(model_type: str, built=None) -> tuple[float, float] | str:
    """The type's gradient and logits measures (`built` of its shrunken config, transformers' own model
    where None), the logits read by its `CONVENTIONS` family where it has one, or why it is not one Dew
    reads."""
    config = verify.shrink({"model_type": model_type, **FIELDS})
    shrunk = AutoConfig.for_model(model_type, **config)
    reference = (AutoModelForCausalLM.from_config(shrunk, dtype=torch.float32) if built is None
                 else built(shrunk).float())
    verify.scatter_weights(reference)
    reference.set_attn_implementation("eager")
    # Its eval forward, as the probe runs it: Seed-OSS drops its sublayers' outputs in training.
    reference.eval()
    ids = verify.probe_ids(shrunk.vocab_size)
    with torch.no_grad():
        expected = reference(torch.as_tensor(ids, dtype=torch.long), use_cache=False).logits.numpy()
    if model_type in CONVENTIONS:
        family = next(entry for entry in ENTRIES if CONVENTIONS[model_type] in entry.model_types)
        record = decoder_parts.translate_family_config({**config, "model_type": model_type}, family)[0]
    else:
        try:
            assembly = parts.assemble(reference)
        except parts.Ambiguous as ambiguous:
            return str(ambiguous)
        family, record = parts.family(model_type, assembly), assembly.fields
    with tempfile.TemporaryDirectory() as folder:
        # As a release stores it (Mixtral's experts one by one), which the assembly reads through the
        # load's conversion, as verify's training qualification does.
        reference.save_pretrained(folder)
        variables = hf_decoders.translate_weights(sources.load_shards(Path(folder)), record, family)
    model = models.build("causal_transformer", {**record, "dtype": "float32", "attention_impl": "reference"})
    return (verify.gradient_parity(reference, family, record, variables),
            verify.logits_parity(expected, model, record, variables, ids))


def main() -> None:
    measures = {}
    for name, (model_type, built) in {**{name: (name, None) for name in TYPES}, **COMPOSITIONS}.items():
        result = measured(model_type, built)
        print(f"{name}: {result}")
        if isinstance(result, tuple):
            measures[name] = result
    for index, kind in enumerate(("gradients", "logits")):
        worst = max(measures, key=lambda name: measures[name][index])
        print(f"largest {kind}: {measures[worst][index]:.3f} ({worst}) over {len(measures)} types")


if __name__ == "__main__":
    sys.exit(main())
