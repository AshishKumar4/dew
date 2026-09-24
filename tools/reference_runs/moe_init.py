#!/usr/bin/env python3
"""Draw a small Qwen3-MoE-shaped decoder from Dew's own initializers and
write it in the Hugging Face layout, so torch and Dew start from one set of
weights.

The config is a Qwen3-MoE config shrunk to a model that trains in minutes:
every layer routed, 8 experts, top 2 with the renormalised weights Qwen3-MoE
uses, grouped-query attention with per-head q/k norms, and the embedding
tied to the head, which keeps the fp32 file under 400 MB at Qwen3's
151,936-id vocabulary.

Dew writes a routed family back over a source checkpoint's own tensor
layout (`Pretrained.save`; qwen3_moe preserves the source layout), so the
layout comes from a scaffold: transformers builds the config and saves it
once, Dew loads that scaffold, draws its own variables from the seeded key
with the model it built (`model.init`), and saves those over the scaffold's
layout. The scaffold's own weights are never read by either run, and it is
deleted. The saved directory is read back and must hold Dew's draw exactly.

    PYTHONPATH=src:<torch cpu site> python tools/reference_runs/moe_init.py --out <dir> --seed 0

A checkpoint drawn before Dew drew runs of like layers under one scan
(docs/concepts/distributed.md) holds different weights at the same seed.
"""

import argparse
import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

CONFIG = {
    "vocab_size": 151936,
    "hidden_size": 384,
    "intermediate_size": 1024,
    "moe_intermediate_size": 512,
    "num_hidden_layers": 8,
    "num_attention_heads": 6,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "norm_topk_prob": True,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
    "router_aux_loss_coef": 0.01,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "max_position_embeddings": 4096,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "use_sliding_window": False,
    "tie_word_embeddings": True,
    "initializer_range": 0.02,
    "bos_token_id": 151643,
    "eos_token_id": 151643,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    from dew.interop import load_pretrained

    scaffold = Path(f"{args.out}-scaffold")
    Qwen3MoeForCausalLM(Qwen3MoeConfig.from_dict(CONFIG)).save_pretrained(scaffold, safe_serialization=True)
    source = load_pretrained(str(scaffold), dtype="float32", param_dtype="float32", attention_impl="xla")
    variables = source.model.init(jax.random.key(args.seed), jnp.zeros((1, 8), jnp.int32))
    if jax.tree.structure(variables) != jax.tree.structure(source.variables):
        raise ValueError("Dew's init and the scaffold's variables are different trees")
    source.save(args.out, variables=variables)
    shutil.rmtree(scaffold)

    written = load_pretrained(args.out, dtype="float32", param_dtype="float32", attention_impl="xla")
    for (path, mine), theirs in zip(jax.tree_util.tree_leaves_with_path(variables),
                                    jax.tree.leaves(written.variables), strict=True):
        if not np.array_equal(np.asarray(mine), np.asarray(theirs)):
            raise ValueError(f"{jax.tree_util.keystr(path)} did not survive the export")
    count = sum(leaf.size for leaf in jax.tree.leaves(variables["params"]))
    print(json.dumps({"out": args.out, "seed": args.seed, "parameters": int(count),
                      "config": json.loads(Path(args.out, "config.json").read_text())}, indent=1))


if __name__ == "__main__":
    main()
