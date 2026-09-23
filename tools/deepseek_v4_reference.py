"""Released-shaped DeepSeek V4 fixture generation; no pretrained weights."""

import json
import tempfile
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
import transformers
from safetensors import safe_open
from safetensors.numpy import load_file, save_file
from transformers import DeepseekV4Config, DeepseekV4ForCausalLM
from transformers.masking_utils import create_sliding_window_causal_mask
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4HashRouter, DeepseekV4RMSNorm, DeepseekV4TopKRouter,
)


# deepseek-ai/DeepSeek-V4-Flash at toy width. The release's proportions on a
# hidden width of 32: moe_intermediate_size half the width (16), the query
# LoRA and the grouped output LoRA a quarter of it (8 each), 4 attention
# heads of the release's 64 over one key/value head, and o_groups of 8 over
# those 64 heads becomes 2 over 4. The release's own values, unscaled:
# scoring_func sqrtsoftplus, routed_scaling_factor 1.5 (V4-Pro's is 2.5),
# rope_theta 10000 on the sliding kind against compress_rope_theta 160000
# under YaRN factor 16 on the compressed kinds, 20 Sinkhorn iterations at
# hc_eps 1e-6, rms_norm_eps 1e-6, three hash layers, one prediction depth,
# and bos/eos 0 and 1. 256 experts with 6 per token become 8 with 2, 64
# indexer heads of 128 keeping 512 compressed entries become 2 heads of 8
# keeping 2, sliding_window 128 becomes 4 and the vocabulary 129280
# becomes 256. hc_mult is 2 rather than the release's 4, the smallest
# expansion that still mixes streams.
#
# Two proportions do not survive the scaling. head_dim is an eighth of the
# release's hidden width (512 of 4096), which here would be 4, too narrow
# to carry a key/value latent under a rope slice, so it stays at 16, half
# the width. And the release ropes an eighth of head_dim (64 of 512):
# a quarter is roped here (4 of 16) because an eighth would leave the rope
# slice 2 wide, one frequency, which YaRN's correction range extrapolates
# whole; the interpolating branch would never run. At a quarter the two
# frequencies come out fully extrapolated and half interpolated, so both
# branches of the ramp move a frequency.
#
# The schedule holds every kind the config allows: V4-Pro's 2x heavily
# compressed bootstrap and the CSA/HCA interleave both releases run
# (V4-Flash bootstraps on two sliding layers instead), plus one sliding
# layer, so all three of DEEPSEEK_V4_LAYER_TYPES appear. Rates of 2 and 4
# over twelve positions leave a CSA layer six compressed entries and an
# HCA layer three, and a sliding window of 4 keeps the local branch
# narrower than the sequence.
#
# swiglu_limit 2.0 against these weights clamps 4.6% of the gate and 8.7%
# of the up pre-activations the routed and shared experts compute (6.6% of
# 13824 values), and another 4.0% of the gate values sit below -2.0, where
# the reference does not clamp (modeling_deepseek_v4.py:978-979,
# :1019-1020): the asymmetry is observable, and dropping the clamp
# altogether moves the logits by 4.0.
#
# Seed 2070 is searched: on both CSA layers and every query row with more
# causally allowed compressed entries than index_topk, the indexer's
# second and third index scores stay at least 0.0220 apart, so the
# fixture's top-k is one selection and not a tie torch and jax break
# differently. The ReLU in the scorer (:448) zeroes a compressed entry
# whenever both indexer heads disagree with the query, and the head
# weights carry either sign (:449), so the scores collect at exactly zero
# and on both sides of it; 17 seeds of 4000 clear 1e-2 on all 28 rows.
DEEPSEEK_V4_SEED = 2070
DEEPSEEK_V4_YARN = {
    "beta_fast": 32, "beta_slow": 1, "factor": 16,
    "original_max_position_embeddings": 65536, "type": "yarn",
}
DEEPSEEK_V4_TINY = dict(
    vocab_size=256, hidden_size=32, num_hidden_layers=6,
    num_attention_heads=4, num_key_value_heads=1, head_dim=16,
    partial_rotary_factor=0.25, q_lora_rank=8, o_groups=2, o_lora_rank=8,
    layer_types=["heavily_compressed_attention", "heavily_compressed_attention",
                 "compressed_sparse_attention", "heavily_compressed_attention",
                 "sliding_attention", "compressed_sparse_attention"],
    mlp_layer_types=["hash_moe"] * 3 + ["moe"] * 3, sliding_window=4,
    compress_rates={"compressed_sparse_attention": 2,
                    "heavily_compressed_attention": 4},
    index_n_heads=2, index_head_dim=8, index_topk=2,
    n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1,
    moe_intermediate_size=16, scoring_func="sqrtsoftplus",
    routed_scaling_factor=1.5, swiglu_limit=2.0, norm_topk_prob=True,
    hc_mult=2, hc_sinkhorn_iters=20, hc_eps=1e-6,
    rope_theta=10000, compress_rope_theta=160000,
    rope_scaling=dict(DEEPSEEK_V4_YARN), max_position_embeddings=64,
    rms_norm_eps=1e-6, tie_word_embeddings=False, num_nextn_predict_layers=1,
    hidden_act="silu", initializer_range=0.02, attention_bias=False,
    attention_dropout=0.0, mlp_bias=False, use_cache=True,
    bos_token_id=0, eos_token_id=1,
)

# The same config as V4-Flash and V4-Pro spell it. Both releases ship the
# schedule as one compress_ratio per layer plus a trailing one for the
# prediction depth, keyed by the release's own rates rather than the
# fixture's (configuration_deepseek_v4.py:28-32 maps 0, 4 and 128 to the
# three kinds and :276 truncates to num_hidden_layers), the fixture's rates
# beside it as compress_rate_csa and compress_rate_hca (:261-264), the hash
# count as num_hash_layers (:279-282), the rope slice as qk_rope_head_dim
# (:254, :286-292) and YaRN flat under rope_scaling (:302-321), where
# `save_pretrained` writes layer_types, mlp_layer_types, compress_rates,
# partial_rotary_factor and the nested rope_parameters it derived from
# them. The trailing 0 is the depth, which V4-Flash ships without a
# compressor (mtp.0.attn.* carries no compressor.* tensors). The releases
# leave the rates at their defaults, so their ratios and their rates read
# the same 4 and 128; here they diverge, and a reader that took the ratio
# for the rate would compress the heavy layers at 128 rather than 4.
DEEPSEEK_V4_RELEASED = dict(
    compress_ratios=[128, 128, 4, 128, 0, 4, 0],
    compress_rate_csa=2, compress_rate_hca=4, num_hash_layers=3,
    qk_rope_head_dim=4, rope_scaling=dict(DEEPSEEK_V4_YARN),
    topk_method="noaux_tc",
)


def tiny_deepseek_v4() -> DeepseekV4ForCausalLM:
    """V4-Flash's shape at toy width, with its hash tables filled.

    `DeepseekV4HashRouter.tid2eid` is a persistent buffer the checkpoint
    carries and the reference initialises to zeros
    (modeling_deepseek_v4.py:1062, :1228-1229), which would route every
    token to expert 0 twice over. Each row here is a fresh permutation of
    the experts truncated to num_experts_per_tok, so the two experts a
    token takes are distinct and the three hash layers disagree with each
    other and with the top-k layers.
    """
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(DeepseekV4Config.from_dict(dict(DEEPSEEK_V4_TINY)))
    generator = torch.Generator().manual_seed(DEEPSEEK_V4_SEED)
    experts = model.config.n_routed_experts
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            if name.endswith("tid2eid"):
                rows, width = buffer.shape
                buffer.copy_(torch.stack([
                    torch.randperm(experts, generator=generator)[:width]
                    for _ in range(rows)]))
    return model


# Every tensor name V4-Flash's model.safetensors.index.json carries, over
# the layer index (N), the expert index (K) and the depth index (J). Read
# off the released index and the safetensors headers of shards 1, 2, 4 and
# 46 (their leading length word and JSON header alone, never a tensor
# byte); the released shapes those headers give, at hidden 4096, head_dim
# 512, 64 heads, q_lora 1024, o_groups 8 of o_lora 1024, 256 experts of
# 2048 and hc_mult 4, are the ones this fixture scales.
#
# `save_pretrained` does not write these names. It reverses the conversion
# mapping, and two of those renames do not come back to the release's
# spelling: `^embed\.weight$`, `^head\.weight$` and the three `^hc_head_*$`
# patterns are anchored (conversion_mapping.py:489-493), so a key stored
# under the base model's own `model.` prefix never matches them, and the
# broad `\.norm\.` -> `\.kv_norm\.` rename (:508) reverses over the
# attention's latent norm, which the release already ships as
# `attn.kv_norm.weight`. Both spellings load, since `.kv_norm.` does not
# contain `.norm.`, but only one is the release's.
DEEPSEEK_V4_SOURCE_NAMES = (
    "embed.weight", "norm.weight", "head.weight",
    "hc_head_fn", "hc_head_base", "hc_head_scale",
    "layers.N.attn_norm.weight", "layers.N.ffn_norm.weight",
    "layers.N.hc_attn_fn", "layers.N.hc_attn_base", "layers.N.hc_attn_scale",
    "layers.N.hc_ffn_fn", "layers.N.hc_ffn_base", "layers.N.hc_ffn_scale",
    "layers.N.attn.attn_sink", "layers.N.attn.q_norm.weight",
    "layers.N.attn.kv_norm.weight", "layers.N.attn.wq_a.weight",
    "layers.N.attn.wq_b.weight", "layers.N.attn.wkv.weight",
    "layers.N.attn.wo_a.weight", "layers.N.attn.wo_b.weight",
    "layers.N.attn.compressor.wkv.weight", "layers.N.attn.compressor.wgate.weight",
    "layers.N.attn.compressor.ape", "layers.N.attn.compressor.norm.weight",
    "layers.N.attn.indexer.compressor.wkv.weight",
    "layers.N.attn.indexer.compressor.wgate.weight",
    "layers.N.attn.indexer.compressor.ape",
    "layers.N.attn.indexer.compressor.norm.weight",
    "layers.N.attn.indexer.weights_proj.weight", "layers.N.attn.indexer.wq_b.weight",
    "layers.N.ffn.gate.weight", "layers.N.ffn.gate.tid2eid", "layers.N.ffn.gate.bias",
    "layers.N.ffn.shared_experts.w1.weight", "layers.N.ffn.shared_experts.w2.weight",
    "layers.N.ffn.shared_experts.w3.weight",
    "layers.N.ffn.experts.K.w1.weight", "layers.N.ffn.experts.K.w2.weight",
    "layers.N.ffn.experts.K.w3.weight",
    "mtp.J.enorm.weight", "mtp.J.hnorm.weight", "mtp.J.norm.weight",
    "mtp.J.e_proj.weight", "mtp.J.h_proj.weight",
    "mtp.J.attn_norm.weight", "mtp.J.ffn_norm.weight",
    "mtp.J.hc_attn_fn", "mtp.J.hc_attn_base", "mtp.J.hc_attn_scale",
    "mtp.J.hc_ffn_fn", "mtp.J.hc_ffn_base", "mtp.J.hc_ffn_scale",
    "mtp.J.hc_head_fn", "mtp.J.hc_head_base", "mtp.J.hc_head_scale",
    "mtp.J.attn.attn_sink", "mtp.J.attn.q_norm.weight", "mtp.J.attn.kv_norm.weight",
    "mtp.J.attn.wq_a.weight", "mtp.J.attn.wq_b.weight", "mtp.J.attn.wkv.weight",
    "mtp.J.attn.wo_a.weight", "mtp.J.attn.wo_b.weight",
    "mtp.J.ffn.gate.weight", "mtp.J.ffn.gate.bias",
    "mtp.J.ffn.shared_experts.w1.weight", "mtp.J.ffn.shared_experts.w2.weight",
    "mtp.J.ffn.shared_experts.w3.weight",
    "mtp.J.ffn.experts.K.w1.weight", "mtp.J.ffn.experts.K.w2.weight",
    "mtp.J.ffn.experts.K.w3.weight",
)


def deepseek_v4_source_name(key: str) -> str:
    """One `save_pretrained` key as V4-Flash's index.json spells it."""
    bare = key.removeprefix("model.")
    bare = bare.replace("embed_tokens.weight", "embed.weight")
    bare = bare.replace("hc_head.hc_", "hc_head_")
    return bare.replace(".attn.norm.", ".attn.kv_norm.")


def deepseek_v4_name_family(key: str) -> str:
    """A tensor name over its layer, depth and expert indices."""
    parts = key.split(".")
    for index, part in enumerate(parts):
        if part.isdigit():
            previous = parts[index - 1]
            parts[index] = {"layers": "N", "mtp": "J"}.get(previous, "K")
    return ".".join(parts)


def write_deepseek_v4_source(name: str, seed: int = DEEPSEEK_V4_SEED) -> None:
    """The checkpoint under V4-Flash's own tensor names, with its depth.

    The released prediction depth is a sliding layer over a top-k routed
    MLP, with two normalisations and two square projections of its own
    ahead of the block and an mHC head of its own after it: V4-Flash's
    mtp.0.* carries no compressor.* and no indexer.* tensor, which is what
    its trailing compress_ratio of 0 says, and its attention shapes are a
    trunk layer's. So the depth here is the fixture's own sliding layer,
    which is also its first top-k routed one, renamed and redrawn, beside
    enorm, hnorm and norm shaped like the trunk's final norm, e_proj and
    h_proj square in the hidden width, and a copy of the trunk's mHC head.

    transformers instantiates no depth (configuration_deepseek_v4.py:173)
    and ignores the whole prefix on load (modeling_deepseek_v4.py:1212), so
    the generic reference loader leaves these tensors unused.
    `load_mtp_reference` composes them into the depth's reference logits
    (`write_deepseek_v4_mtp_reference` writes mtp_logits.npy). The draw
    follows `scatter_weights`: norms around one, everything else at a
    fifth, and the depth's balancing bias on the same linspace the trunk's
    top-k layers carry.

    This runs on the config `save_pretrained` wrote, before
    `write_deepseek_v4_config` folds `layer_types` back into the release's
    `compress_ratios`, and says so if the order is ever swapped.
    """
    from hf_reference import FIXTURES

    directory = FIXTURES / name
    tensors = {deepseek_v4_source_name(key): value
               for key, value in load_file(str(directory / "model.safetensors")).items()}
    kinds = json.loads((directory / "config.json").read_text()).get("layer_types")
    if kinds is None:
        raise SystemExit(f"{directory}: run this before write_deepseek_v4_config, "
                         "which rewrites layer_types into the release's compress_ratios")
    sliding = f"layers.{kinds.index('sliding_attention')}."
    depth = {"mtp.0." + key.removeprefix(sliding): value
             for key, value in tensors.items() if key.startswith(sliding)}
    width = tensors["embed.weight"].shape[1]
    for leaf in ("enorm.weight", "hnorm.weight", "norm.weight"):
        depth["mtp.0." + leaf] = tensors["norm.weight"]
    for leaf in ("e_proj.weight", "h_proj.weight"):
        depth["mtp.0." + leaf] = np.empty((width, width), np.float32)
    for leaf in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
        depth["mtp.0." + leaf] = tensors[leaf]

    generator = torch.Generator().manual_seed(seed)
    for key in sorted(depth):
        shape = depth[key].shape
        if key.endswith("gate.bias"):
            drawn = torch.linspace(-0.4, 0.4, shape[0])
        else:
            noise = torch.randn(tuple(shape), generator=generator) * 0.05
            drawn = 1.0 + noise if "norm" in key else noise * 4.0
        depth[key] = drawn.numpy().astype(np.float32)
    tensors.update(depth)
    save_file(tensors, str(directory / "model.safetensors"), metadata={"format": "pt"})
    print(f"{directory}: {len(tensors)} tensors under the release's names, "
          f"{len(depth)} of them the mtp.0 depth")


def write_deepseek_v4_config(name: str, repo: str) -> None:
    """The fixture's config as DeepSeek releases one, and its provenance.

    The release also ships an fp8 `quantization_config` and the
    `expert_dtype` naming the experts' own format; neither is written here,
    since the fixture's weights are floating.

    The reload fails the fixture if the released spelling reaches other
    logits than the saved config did, or if any tensor outside the depth
    goes unread.
    """
    from huggingface_hub import model_info
    from hf_reference import FIXTURES
    from dew.interop.verify import reference_logits

    directory = FIXTURES / name
    saved = json.loads((directory / "config.json").read_text())
    for field in ("layer_types", "mlp_layer_types", "compress_rates",
                  "partial_rotary_factor", "rope_parameters", "qk_rope_head_dim"):
        saved.pop(field, None)
    config = {"architectures": ["DeepseekV4ForCausalLM"], **saved, **DEEPSEEK_V4_RELEASED,
              "model_type": "deepseek_v4", "transformers_version": transformers.__version__}
    (directory / "config.json").write_text(
        json.dumps(dict(sorted(config.items())), indent=1) + "\n")
    (directory / "source.json").write_text(json.dumps({
        "released": {"repo": repo, "revision": model_info(repo).sha},
        "transformers": {"version": transformers.__version__,
                         "revision": "93c8b7b485963a10800c91f55304db6be211c2bd"},
    }, indent=1) + "\n")

    loaded = DeepseekV4ForCausalLM.from_pretrained(
        str(directory), dtype=torch.float32, local_files_only=True, output_loading_info=True)
    if not isinstance(loaded, tuple):
        raise SystemExit("output_loading_info returns the model and its report")
    model, report = loaded
    unread = {key: sorted(report[key]) for key in
              ("missing_keys", "mismatched_keys", "error_msgs") if report[key]}
    # The depth's tensors are `_keys_to_ignore_on_load_unexpected`
    # (modeling_deepseek_v4.py:1212): transformers builds no depth, so they
    # are checked by the separate MTP composition; no other key may go unread.
    stray = sorted(key for key in report["unexpected_keys"] if "mtp." not in key)
    if unread or stray:
        raise SystemExit(f"{directory}: the released config does not read its own weights: "
                         f"{unread} {stray}")
    ids = np.load(directory / "input_ids.npy")
    difference = float(np.max(np.abs(reference_logits(model, ids)
                                     - np.load(directory / "logits.npy"))))
    if difference != 0.0:
        raise SystemExit(f"{directory}: the released spelling moved the logits by {difference:.3e}")
    print(f"{directory}: model_type {config['model_type']}, {len(config)} fields, "
          f"released spelling reloads bit for bit")
    print_deepseek_v4_tensors(directory, model)


def print_deepseek_v4_tensors(directory: Path, model: DeepseekV4ForCausalLM) -> None:
    """The saved tensor names, and the proof they are V4-Flash's own.

    Every name in the file has to be one DEEPSEEK_V4_SOURCE_NAMES lists and
    every trunk name in that table has to appear, so a `save_pretrained`
    spelling left unrewritten and a depth tensor left out both fail here.
    conversion_mapping.py:521-533 stacks `experts.K.w1` and `experts.K.w3`
    into one `experts.gate_up_proj` and `experts.K.w2` into
    `experts.down_proj`, and :517 renames `gate.bias` to
    `e_score_correction_bias`, so the per-expert counts are checked against
    the routers the reference actually built: a hash layer whose table went
    missing, or a top-k layer whose balancing bias did, is a failure. The
    names print per layer, depth and expert collapsed, since the six
    layers, one depth and eight experts repeat.
    """
    tensors = load_file(str(directory / "model.safetensors"))
    families: dict[str, list[str]] = {}
    for key, tensor in tensors.items():
        families.setdefault(deepseek_v4_name_family(key), []).append(
            f"{list(tensor.shape)} {tensor.dtype}")
    unknown = sorted(set(families) - set(DEEPSEEK_V4_SOURCE_NAMES))
    absent = sorted(key for key in DEEPSEEK_V4_SOURCE_NAMES
                    if key not in families and not key.startswith("mtp."))
    if unknown or absent:
        raise SystemExit(f"{directory}: the saved names are not the release's, "
                         f"unknown {unknown} absent {absent}")
    routers = [module for module in model.modules()
               if isinstance(module, (DeepseekV4HashRouter, DeepseekV4TopKRouter))]
    experts = model.config.n_routed_experts * model.config.num_hidden_layers
    required = {
        "layers.N.ffn.experts.K.w1.weight": experts,
        "layers.N.ffn.experts.K.w2.weight": experts,
        "layers.N.ffn.experts.K.w3.weight": experts,
        "layers.N.ffn.gate.tid2eid":
            sum(1 for router in routers if isinstance(router, DeepseekV4HashRouter)),
        "layers.N.ffn.gate.bias":
            sum(1 for router in routers if isinstance(router, DeepseekV4TopKRouter)),
    }
    wrong = {key: (len(families.get(key, ())), count)
             for key, count in required.items() if len(families.get(key, ())) != count}
    if wrong:
        raise SystemExit(f"{directory}: the routed tensors do not match the routers "
                         f"the reference built, found and wanted {wrong}")
    print(f"{directory}: {len(tensors)} tensors over {len(families)} names")
    for key, shapes in sorted(families.items()):
        print(f"  {len(shapes):4d}  {key:52s} {', '.join(sorted(set(shapes)))}")


class DeepseekV4MTP(torch.nn.Module):
    """The V4 prediction depth as the release composes it.

    The official inference model runs it in MTPBlock.forward
    (deepseek-v4-pro-b5968e9-model.py:756-766): the shifted token's
    embedding is normalised and projected, added to the normalised and
    projected raw trunk streams ([B, S, H, D]; the depth never reads the
    trunk's collapsed, final-normed state), run through one unchanged
    decoder layer, collapsed by the depth's own mHC head, normed by its
    own norm and mapped to logits by the trunk's head. ParallelHead
    scores only the last position for generation; the fixture scores
    every position, like the trunk's own logits.

    `reference_model` owns the loaded block, norm and collapse head of the
    one-layer checkpoint constructed from the released depth tensors. Its
    original module names also drive reference gradient conversion.
    """

    def __init__(self, config: DeepseekV4Config,
                 reference_model: DeepseekV4ForCausalLM) -> None:
        super().__init__()
        self.config = config
        self.e_proj = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.h_proj = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.enorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.reference_model = reference_model

    def forward(self, model: DeepseekV4ForCausalLM, streams: torch.Tensor,
                ids: torch.Tensor) -> torch.Tensor:
        embeddings = model.model.embed_tokens(ids)
        fused = self.e_proj(self.enorm(embeddings))[:, :, None, :] + self.h_proj(self.hnorm(streams))
        positions = torch.arange(fused.shape[1], device=fused.device)[None]
        mask = create_sliding_window_causal_mask(
            config=self.config, inputs_embeds=embeddings, attention_mask=None,
            past_key_values=None, position_ids=positions)
        out = self.reference_model.model.layers[0](
            fused,
            position_embeddings={
                "main": model.model.rotary_emb(embeddings, position_ids=positions,
                                               layer_type="main"),
            },
            position_ids=positions, attention_mask=mask, input_ids=ids)
        return model.lm_head(self.reference_model.model.norm(self.reference_model.model.hc_head(out)))


def load_mtp_reference(directory: Path, config: DeepseekV4Config) -> DeepseekV4MTP:
    """The fixture's released mtp.0.* tensors as a runnable depth.

    The release ships the depth under its own flat names and transformers
    builds none, so the block is loaded through the standard
    `from_pretrained` conversion instead of a name map: the depth's block
    keys are re-prefixed mtp.0.* -> layers.0.*, its own final norm and mHC
    head are moved to the checkpoint's top level (norm.weight and
    hc_head_fn/hc_head_base/hc_head_scale, where the trunk's live), and
    the trunk's embed.weight and head.weight complete the one-layer
    source. e_proj, h_proj, enorm and hnorm have no trunk counterpart, so
    they are bound straight onto the wrapper's four own modules.

    The temporary config is a copy of the trunk's with the schedule
    overridden rather than translated: explicit `layer_types` and
    `mlp_layer_types` take precedence over the legacy compress_ratios and
    num_hash_layers spellings (configuration_deepseek_v4.py:266-282), so
    the one sliding layer over a top-k routed MLP is spelled directly and
    the released rope_parameters carry over untouched.

    Reads the depth tensors off the index when the checkpoint is sharded
    (trained exports are), so it composes the same way without redrawing
    any weight.
    """
    directory = Path(directory)
    fields = config.to_dict()
    fields.update(architectures=["DeepseekV4ForCausalLM"], num_hidden_layers=1,
                  layer_types=["sliding_attention"], mlp_layer_types=["moe"])
    for legacy in ("compress_ratios", "num_hash_layers"):
        fields.pop(legacy, None)

    index = directory / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        wanted = [key for key in weight_map
                  if key.startswith("mtp.0.") or key in ("embed.weight", "head.weight")]
        tensors = {}
        for shard in sorted({weight_map[key] for key in wanted}):
            with safe_open(str(directory / shard), framework="pt") as opened:
                tensors.update({key: opened.get_tensor(key) for key in wanted
                                if weight_map[key] == shard})
    else:
        tensors = {key: value for key, value in
                   safetensors.torch.load_file(str(directory / "model.safetensors")).items()
                   if key.startswith("mtp.0.") or key in ("embed.weight", "head.weight")}

    renamed, direct = {}, {}
    for key, value in tensors.items():
        if key.startswith("mtp.0."):
            leaf = key.removeprefix("mtp.0.")
            if leaf in ("e_proj.weight", "h_proj.weight", "enorm.weight", "hnorm.weight"):
                direct[leaf] = value
            elif leaf.startswith("norm.") or leaf.startswith("hc_head_"):
                renamed[leaf] = value
            else:
                renamed[f"layers.0.{leaf}"] = value
        else:
            renamed[key] = value

    with tempfile.TemporaryDirectory() as temporary:
        Path(temporary, "config.json").write_text(json.dumps(fields))
        safetensors.torch.save_file(renamed, str(Path(temporary, "model.safetensors")))
        loaded = DeepseekV4ForCausalLM.from_pretrained(
            temporary, dtype=torch.float32, local_files_only=True, output_loading_info=True)
    if not isinstance(loaded, tuple):
        raise SystemExit("output_loading_info returns the model and its report")
    model, report = loaded
    unread = {key: sorted(report[key]) for key in
              ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
              if report[key]}
    if unread:
        raise SystemExit(f"{directory}: the mtp.0 depth does not load cleanly: {unread}")
    model.eval()
    model.set_attn_implementation('eager')

    # Loader-only copies are not part of the MTP computation: forward calls
    # the trunk's shared embedding and head, as the released MTPBlock does.
    model.model.embed_tokens.requires_grad_(False)
    model.lm_head.requires_grad_(False)
    depth = DeepseekV4MTP(model.config, model)
    for name in ("e_proj", "h_proj", "enorm", "hnorm"):
        getattr(depth, name).weight = torch.nn.Parameter(direct[f"{name}.weight"])
    return depth


def write_deepseek_v4_mtp_reference(name: str) -> None:
    """The depth's reference logits over the fixture's shifted tokens.

    A forward hook on the last decoder layer captures the trunk's raw
    [B, S, H, D] streams, the tensor the released depth reads before
    the trunk's hc_head and final norm touch it, and the depth then
    scores the next token from each position, the same shift the GLM
    fixtures use. The hook is removed in `finally` and nothing detaches
    the streams, so `load_mtp_reference`'s forward stays differentiable
    when it is called outside no_grad.
    """
    from hf_reference import FIXTURES

    directory = FIXTURES / name
    model = DeepseekV4ForCausalLM.from_pretrained(
        str(directory), dtype=torch.float32, local_files_only=True)
    model.eval()
    model.set_attn_implementation("eager")
    ids = torch.from_numpy(np.load(directory / "input_ids.npy").astype(np.int64))
    captured = []

    def capture(_module, _inputs, output):
        captured.append(output)

    hook = model.model.layers[-1].register_forward_hook(capture)
    try:
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
    finally:
        hook.remove()
    depth = load_mtp_reference(directory, model.config)
    with torch.no_grad():
        logits = depth(model, captured[0][:, :-1], ids[:, 1:])
    np.save(directory / "mtp_logits.npy", logits.to(torch.float32).numpy())
    source = json.loads((directory / "source.json").read_text())
    source["mtp_reference"] = {
        "repo": "deepseek-ai/DeepSeek-V4-Pro",
        "revision": "b5968e9190ef611bbf34a7229255be88a0e937c1",
        "path": "inference/model.py",
        "sha256": "ce962f1face79d4f633d36436576214057a7e11443c9789935e1deb5c6cd1d71",
    }
    (directory / "source.json").write_text(json.dumps(source, indent=1) + "\n")
    print(f"{directory}: depth mtp.0.* composed, mtp logits {tuple(logits.shape)}")
