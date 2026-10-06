"""DeepSeek-V4-Flash-0731 tiny fixture: the release's DSpark drafter over V4's trunk.

V4-Flash-0731 ships V4's trunk with three DSpark stages under `mtp.*`; its
config.json, inference/model.py and weight index are byte for byte
DeepSeek-V4-Flash-DSpark's. transformers' DeepseekV4 builds the trunk and no
drafter, so the drafter's reference is the release's inference/model.py at
DEEPSEEK_V4_0731_REVISION. This tool downloads it, installs
`tools/deepseek_v41_kernels.py` as its `kernel` module, as
tools/deepseek_v41_reference.py does for V4.1 (the two releases' kernel.py
export the same functions, and their sparse_attn differs only on a row with
no valid key, which no row here has), and runs its `Transformer` in fp32 on
CPU over dense weights. The trunk's logits come from transformers 5.16.1's
DeepseekV4ForCausalLM over the same checkpoint. Run it in the project's
environment:

    PYTHONPATH=src:.:tools python tools/deepseek_v4_dspark_reference.py [--seed N] [--released]

It writes tests/fixtures/hf/deepseek-v4-dspark-tiny: config.json in the
release's spelling at toy width, model.safetensors under the release's
tensor names, source.json, and

    reference.npz      input_ids [2, 16]; logits, transformers' trunk over
                       them; decode_prompt, the prefill length of the cached
                       run; draft_ids, draft_logits, draft_confidence, the
                       release's forward_spec after every teacher-forced step
                       past the prefill
    reference_f64.npz  the same outputs in float64: transformers with its
                       weights widened, the release under
                       tools/deepseek_v41_reference.py's `widened`

The tool refuses a seed where either reference, in fp32, picks another
expert, indexer entry or draft token than its float64 run does, or where
the two references' float64 trunks part by more than float64 rounding. It
counts the tied selections the release met, top-k rows whose k-th and next
scores are equal, which the exact zeros of the indexer's ReLU make
(model.py:427) and only the tie order decides; source.json records the
count. SEED is the first seed tried, not one searched for.

`--released` writes tests/fixtures/hf/deepseek-v4-flash-0731 instead: the
release's config.json and tensor_names.json, its weight index with the
`.scale` partners dropped and the expert index as K.

Where the reference departs from the release it says so. The quantizers
are off: act_quant and fp4_act_quant are the identity, since neither Dew's
V4 mixer nor transformers' DeepseekV4 rounds the cache. With them off,
rotate_activation, the indexer's Hadamard rotation of its queries and keys
alike, leaves the scores it ranks unchanged in exact arithmetic, so it is
the identity (model.py:253-257, :374-376, :420-422), and the head returns
every position's logits (`_full_head`). Beyond the quantizers, the tie order
is the one documented difference from the release: each indexer and router
breaks equal scores toward the lower entry, as jax.lax.top_k does, where
torch.topk leaves the order unspecified (`lower_index_ties` of
tools/decoder_export_reference.py, on the release's Indexer and Gate and
transformers' indexer and router alike).
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import deepseek_v41_reference as v41  # noqa: E402
from decoder_export_reference import lower_index_ties  # noqa: E402

REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEEPSEEK_V4_0731_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
FIXTURE = ROOT / "tests" / "fixtures" / "hf" / "deepseek-v4-dspark-tiny"
RELEASED = ROOT / "tests" / "fixtures" / "hf" / "deepseek-v4-flash-0731"
SEED = 0
LENGTH = 16
# Past the 4-token window, so the prefill wraps the trunk's and DSpark's
# rings, and a multiple of neither compress ratio, so each compressor
# carries an open group from the prefill into the decode.
PROMPT = 9

# V4-Flash-0731 at toy width, one entry per ModelArgs field the release's
# inference/config.json sets. The release stacks two sliding layers, then
# CSA at ratio 4 (with the indexer, which model.py:474-477 builds for ratio
# 4 alone) and HCA at 128 alternating, three hash-routed layers first, and
# three DSpark stages over the last three layers' outputs. The toy keeps
# each: sliding 0-1, CSA 2 and 4 keeping 2 of up to 4 entries, HCA 3 at
# ratio 8 so that it pools inside 16 tokens, sliding 5, hash routing on
# layer 0, and the stages over layers 3-5 with the trunk's experts, as the
# release names no drafter experts of its own. hc_mult is 2, the smallest
# that mixes streams.
TINY = {
    "max_batch_size": 2, "max_seq_len": 32, "temperature": 0.0, "dtype": "bf16", "scale_fmt": None,
    "expert_dtype": None, "scale_dtype": "fp32",
    "vocab_size": 256, "dim": 32, "moe_inter_dim": 16, "n_layers": 6, "n_hash_layers": 1, "n_mtp_layers": 3,
    "n_heads": 4, "n_routed_experts": 8, "n_shared_experts": 1, "n_activated_experts": 2,
    "score_func": "sqrtsoftplus", "route_scale": 1.5, "swiglu_limit": 2.0,
    "q_lora_rank": 8, "head_dim": 16, "rope_head_dim": 4, "norm_eps": 1e-6, "o_groups": 2, "o_lora_rank": 8,
    "window_size": 4, "compress_ratios": (0, 0, 4, 8, 4, 0, 0, 0, 0),
    "compress_rope_theta": 160000.0, "original_seq_len": 65536, "rope_theta": 10000.0,
    "rope_factor": 16, "beta_fast": 32, "beta_slow": 1,
    "index_n_heads": 2, "index_head_dim": 8, "index_topk": 2,
    "hc_mult": 2, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
    "dspark_block_size": 5, "dspark_noise_token_id": 255, "dspark_target_layer_ids": (3, 4, 5),
    "dspark_markov_rank": 8,
}


def config_json() -> dict:
    """TINY as the release's config.json spells it. A ratio names its kind
    there (0, 4 or 128, configuration_deepseek_v4.py:28-32), so the HCA
    layer's 8 is compress_rate_hca beside a 128; num_nextn_predict_layers
    keeps the release's misstated 1 against three stages."""
    ratios = [128 if rate == 8 else rate for rate in TINY["compress_ratios"]]
    return {
        "architectures": ["DeepseekV4ForCausalLM"], "attention_bias": False, "attention_dropout": 0.0,
        "bos_token_id": 0, "eos_token_id": 1, "hc_eps": TINY["hc_eps"], "hc_mult": TINY["hc_mult"],
        "hc_sinkhorn_iters": TINY["hc_sinkhorn_iters"], "head_dim": TINY["head_dim"], "hidden_act": "silu",
        "hidden_size": TINY["dim"], "index_head_dim": TINY["index_head_dim"],
        "index_n_heads": TINY["index_n_heads"], "index_topk": TINY["index_topk"], "initializer_range": 0.02,
        "max_position_embeddings": 64, "model_type": "deepseek_v4",
        "moe_intermediate_size": TINY["moe_inter_dim"], "n_routed_experts": TINY["n_routed_experts"],
        "n_shared_experts": 1, "norm_topk_prob": True, "num_attention_heads": TINY["n_heads"],
        "num_experts_per_tok": TINY["n_activated_experts"], "num_hidden_layers": TINY["n_layers"],
        "num_hash_layers": TINY["n_hash_layers"], "num_key_value_heads": 1, "num_nextn_predict_layers": 1,
        "o_groups": TINY["o_groups"], "o_lora_rank": TINY["o_lora_rank"], "q_lora_rank": TINY["q_lora_rank"],
        "qk_rope_head_dim": TINY["rope_head_dim"], "rms_norm_eps": TINY["norm_eps"],
        "rope_scaling": {"beta_fast": TINY["beta_fast"], "beta_slow": TINY["beta_slow"],
                         "factor": TINY["rope_factor"],
                         "original_max_position_embeddings": TINY["original_seq_len"], "type": "yarn"},
        "rope_theta": int(TINY["rope_theta"]), "routed_scaling_factor": TINY["route_scale"],
        "scoring_func": TINY["score_func"], "sliding_window": TINY["window_size"],
        "swiglu_limit": TINY["swiglu_limit"], "tie_word_embeddings": False, "topk_method": "noaux_tc",
        "torch_dtype": "float32", "use_cache": True, "vocab_size": TINY["vocab_size"],
        "compress_rope_theta": int(TINY["compress_rope_theta"]), "compress_ratios": ratios,
        "compress_rate_hca": 8,
        "dspark_block_size": TINY["dspark_block_size"], "dspark_markov_rank": TINY["dspark_markov_rank"],
        "dspark_noise_token_id": TINY["dspark_noise_token_id"],
        "dspark_target_layer_ids": list(TINY["dspark_target_layer_ids"]),
    }


def import_reference():
    """The pinned model.py over the torch kernels, its quantizers and
    rotation the identity (the module doc)."""
    from huggingface_hub import hf_hub_download

    sys.modules["kernel"] = importlib.import_module("tools.deepseek_v41_kernels")
    path = hf_hub_download(REPO, "inference/model.py", revision=DEEPSEEK_V4_0731_REVISION)
    spec = importlib.util.spec_from_file_location("deepseek_v4_0731_model", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    v41.set_quantizers(module, enabled=False)
    module.rotate_activation = lambda x: x
    return module


def draw(name: str, shape, generator) -> torch.Tensor:
    """One parameter's toy values: V4.1's rules (`v41.draw`), the Markov
    embedding at unit scale as V4.1's, and every mHC head in the ranges of
    the sites' own."""
    leaf = name.rsplit(".", 1)[-1]
    if leaf in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
        return v41.draw(name.replace("hc_head_", "x.hc_attn_"), shape, generator)
    if name.endswith("markov_w1.weight"):
        return torch.randn(tuple(shape), generator=generator, dtype=torch.float32)
    return v41.draw(name, shape, generator)


def build(module, seed: int, widen: bool = False):
    """The toy Transformer in fp32, or fp64 when `widen`, with its fp32
    weights drawn and each hash row a fresh permutation of the experts
    truncated to the top-k, so a token takes two distinct experts."""
    torch.manual_seed(seed)
    net = module.Transformer(module.ModelArgs(**TINY))
    net = net.double() if widen else net.float()
    generator = torch.Generator().manual_seed(seed)
    for name, tensor in net.named_parameters():
        tensor.requires_grad_(requires_grad=False)
        if name.endswith("tid2eid"):
            rows, width = tensor.shape
            tensor.copy_(torch.stack([torch.randperm(TINY["n_routed_experts"], generator=generator)[:width]
                                      for _ in range(rows)]))
        else:
            tensor.copy_(draw(name, tensor.shape, generator))
    net.head.forward = functools.partial(v41._full_head, net.head)
    return net


def reset(net):
    """Zero every cache and compressor state a previous run wrote."""
    for name, buffer in net.named_buffers():
        if name.endswith("score_state"):
            buffer.fill_(float("-inf"))
        elif name.endswith(("kv_cache", "kv_state")):
            buffer.zero_()


def forward(net, ids, start_pos=0):
    """Transformer.forward (model.py:912-926), every position's logits."""
    return type(net).forward.__wrapped__(net, ids, start_pos)


def draft(net, ids, main_hidden, start_pos):
    """Transformer.forward_spec (model.py:928-936)."""
    return type(net).forward_spec.__wrapped__(net, ids, main_hidden, start_pos)


def cached(net, ids) -> dict[str, np.ndarray]:
    """The prefill of the first PROMPT tokens, which seeds DSpark's windows,
    then a teacher-forced step and DSpark's draft after each."""
    draft_ids, draft_logits, draft_confidence = [], [], []
    reset(net)
    next_ids, _, main_hidden = forward(net, ids[:, :PROMPT])
    draft(net, next_ids[:, -1], main_hidden, 0)
    for position in range(PROMPT, LENGTH):
        next_ids, _, main_hidden = forward(net, ids[:, position:position + 1], position)
        drafted = draft(net, next_ids[:, -1], main_hidden, position)
        draft_ids.append(drafted[0])
        draft_logits.append(drafted[1])
        draft_confidence.append(drafted[2])
    return {"draft_ids": torch.stack(draft_ids, 1).numpy().astype(np.int32),
            "draft_logits": torch.stack(draft_logits, 1).numpy(),
            "draft_confidence": torch.stack(draft_confidence, 1).numpy()}


def inputs(seed: int) -> torch.Tensor:
    """The fixture's token ids at `seed`, clear of the noise token."""
    generator = torch.Generator().manual_seed(seed + 1)
    return torch.randint(2, TINY["vocab_size"] - 1, (2, LENGTH), generator=generator)


def tied(found: list) -> int:
    """How many recorded selections the tie order decided: rows whose k-th
    and next finite scores are equal."""
    count = 0
    for rows, picks in found:
        k = picks.size(-1)
        if k < rows.size(-1):
            ordered = rows.sort(-1, descending=True).values
            kth, after = ordered[:, k - 1], ordered[:, k]
            count += int((torch.isfinite(kth) & (kth == after)).sum())
    return count


def differ(found: list, wide: list) -> int:
    """How many picks an fp32 run made otherwise than its float64 run."""
    if len(found) != len(wide):
        return max(len(found), len(wide))
    return sum(int((picks != other).sum()) for (_, picks), (_, other) in zip(found, wide, strict=True))


def released_outputs(module, seed: int, ids: torch.Tensor, widen: bool = False):
    """The release's full-forward logits and its cached run, the net, and
    every top-k its indexers and routers made, each breaking ties toward the
    lower index (`lower_index_ties`)."""
    found: list = []
    with torch.no_grad():
        net = build(module, seed, widen)
        with lower_index_ties(net, classes=(module.Indexer, module.Gate), found=found):
            reset(net)
            _, logits, _ = forward(net, ids)
            outputs = {"release_logits": logits.numpy(), **cached(net, ids)}
    return net, outputs, found


@contextlib.contextmanager
def widened_transformers(module):
    """transformers in float64: `v41.widened`'s `.float()` and the pins
    transformers adds, `.to(torch.float32)` and the compressors'
    `softmax(dtype=torch.float32)` (modeling_deepseek_v4.py:57, :408, :540,
    :665), each made float64."""
    to, softmax = torch.Tensor.to, torch.Tensor.softmax

    def wide(dtype):
        return torch.float64 if dtype is torch.float32 else dtype

    def wide_to(self, *args, **kwargs):
        if "dtype" in kwargs:
            kwargs["dtype"] = wide(kwargs["dtype"])
        return to(self, *(wide(arg) for arg in args), **kwargs)

    torch.Tensor.to = wide_to
    torch.Tensor.softmax = lambda self, *args, dtype=None, **kwargs: softmax(
        self, *args, dtype=wide(dtype), **kwargs)
    try:
        with v41.widened(module):
            yield
    finally:
        torch.Tensor.to, torch.Tensor.softmax = to, softmax


def transformers_logits(directory: Path, ids: torch.Tensor, dtype: torch.dtype):
    """transformers' logits over `ids` from the checkpoint in `directory`, and
    every top-k its indexers and routers made under the lower-index tie order
    (`lower_index_ties`); refused if it leaves a trunk tensor unread."""
    from transformers import DeepseekV4ForCausalLM
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Indexer, DeepseekV4TopKRouter

    # The eager experts, since torch's grouped_mm takes no float64.
    loaded = DeepseekV4ForCausalLM.from_pretrained(str(directory), dtype=dtype, local_files_only=True,
                                                   output_loading_info=True, experts_implementation="eager")
    model, report = loaded
    unread = {key: sorted(report[key]) for key in ("missing_keys", "mismatched_keys", "error_msgs")
              if report[key]}
    # transformers builds no drafter and ignores `mtp.*` on load
    # (modeling_deepseek_v4.py:1212); nothing else may go unread.
    stray = sorted(key for key in report["unexpected_keys"] if "mtp." not in key)
    if unread or stray:
        raise SystemExit(f"{directory}: transformers does not read the checkpoint: {unread} {stray}")
    model.eval()
    model.set_attn_implementation("eager")
    found: list = []
    selecting = (DeepseekV4Indexer, DeepseekV4TopKRouter)
    with torch.no_grad(), lower_index_ties(model, classes=selecting, found=found):
        logits = model(input_ids=ids, use_cache=False).logits.numpy()
    return logits, found


def write(seed: int):
    from dew.interop.safetensors_io import write_file

    module = import_reference()
    ids = inputs(seed)
    net, outputs, found = released_outputs(module, seed, ids)
    with v41.widened(module):
        _, wide, wide_found = released_outputs(module, seed, ids, widen=True)
    FIXTURE.mkdir(parents=True, exist_ok=True)
    # Each stage holds the trunk's embedding and head as attributes
    # (model.py:903-904), which the release's index does not repeat.
    tensors = {name: tensor.numpy() for name, tensor in net.state_dict().items()
               if not re.fullmatch(r"mtp\.\d+\.(embed|head)\.weight", name)}
    released = {family(name) for name in json.loads((RELEASED / "tensor_names.json").read_text())["names"]}
    unshipped = sorted({family(name) for name in tensors} - released)
    if unshipped:
        raise SystemExit(f"names the release does not ship: {unshipped}")
    write_file(tensors, FIXTURE / "model.safetensors", {"format": "pt"})
    (FIXTURE / "config.json").write_text(json.dumps(config_json(), indent=1) + "\n")
    logits, picks = transformers_logits(FIXTURE, ids, torch.float32)
    with widened_transformers(module):
        truth, truth_picks = transformers_logits(FIXTURE, ids, torch.float64)

    differing = {"release picks": differ(found, wide_found), "transformers picks": differ(picks, truth_picks),
                 "draft_ids": int(np.sum(outputs["draft_ids"] != wide["draft_ids"]))}
    trunks = float(np.max(np.abs(wide["release_logits"] - truth)))
    ties = tied(found)
    print(json.dumps({"seed": seed, "differing": differing, "float64 trunks apart": trunks,
                      "tied selections": ties, "selections": sum(len(picks) for _, picks in found)}))
    if any(differing.values()) or trunks > 1e-9:
        raise SystemExit(f"seed {seed}: a reference decides otherwise than its float64 run, or the "
                         "release's trunk is not transformers'")
    names = ("draft_ids", "draft_logits", "draft_confidence")
    np.savez(FIXTURE / "reference.npz", input_ids=ids.numpy().astype(np.int32), logits=logits,
             decode_prompt=np.int32(PROMPT), **{name: outputs[name] for name in names})
    np.savez(FIXTURE / "reference_f64.npz", logits=truth, draft_logits=wide["draft_logits"],
             draft_confidence=wide["draft_confidence"])
    (FIXTURE / "source.json").write_text(json.dumps({
        "release": {"repo": REPO, "revision": DEEPSEEK_V4_0731_REVISION, "path": "inference/model.py"},
        "transformers": {"version": "5.16.1"}, "seed": seed, "tied_selections": ties}, indent=1) + "\n")
    print(f"{FIXTURE}: {len(tensors)} tensors, seed {seed}")


def family(name: str) -> str:
    """A tensor name over its layer, stage and expert indices."""
    name = re.sub(r"^layers\.\d+\.", "layers.N.", name)
    name = re.sub(r"^mtp\.\d+\.", "mtp.J.", name)
    return re.sub(r"\.experts\.(\d+|K)\.", ".experts.K.", name)


def write_released():
    """The release's config.json and its weight index's tensor names at the
    pinned revision."""
    from huggingface_hub import hf_hub_download

    RELEASED.mkdir(parents=True, exist_ok=True)
    config = Path(hf_hub_download(REPO, "config.json", revision=DEEPSEEK_V4_0731_REVISION)).read_text()
    (RELEASED / "config.json").write_text(config)
    index = json.loads(Path(hf_hub_download(REPO, "model.safetensors.index.json",
                                            revision=DEEPSEEK_V4_0731_REVISION)).read_text())
    names = sorted({re.sub(r"\.experts\.\d+\.", ".experts.K.", name)
                    for name in index["weight_map"] if not name.endswith(".scale")})
    (RELEASED / "tensor_names.json").write_text(json.dumps({
        "revision": DEEPSEEK_V4_0731_REVISION,
        "source": "model.safetensors.index.json weight_map, .scale partners dropped, expert index as K",
        "names": names}, indent=1) + "\n")
    print(f"{RELEASED}: {len(names)} names of {len(index['weight_map'])} tensors")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--released", action="store_true")
    args = parser.parse_args()
    if args.released:
        write_released()
    else:
        write(args.seed)


if __name__ == "__main__":
    main()
