"""Write the tiny Kimi K3 fixture with the released remote code.

The model is `KimiK3ForConditionalGeneration` from moonshotai/Kimi-K3 at
REVISION: configuration_kimi_k3.py, modeling_kimi_k3.py and
modeling_kimi_linear.py, fetched at that revision into CACHE. Their KDA
layers call fla-core's Triton kernels (chunk_kda, fused_recurrent_kda,
ShortConvolution, FusedRMSNormGated), so this runs on a CUDA device, in fp32,
with TRITON_F32_DEFAULT=ieee so no kernel rounds through TF32. The routed
experts are encoded by compressed-tensors' own MXFP4 compressor and the
reference computes with its own decompression of them, so the packed file
and the logits describe one model.

Environment (~/.cache/dew/reference-venvs/kimi-k3): torch 2.8.0+cu128,
transformers 4.56.2, fla-core 0.5.2, compressed-tensors 0.17.1.

Two things change how the reference runs, neither what it computes:

- KimiLinearModel pins `flash_attention_2`; MLA runs the file's own
  `eager_attention_forward` instead (flash-attn is not installed).
- `KimiSparseMoeBlock.moe_infer` is decorated `torch.no_grad`, which would
  cut the routed experts out of the backward pass; the gradient step calls
  the undecorated function. The gate asserts eval mode, so the model stays
  in eval mode, where nothing in the text model is stochastic.

The tiny config keeps the release's fields and layer pattern at small widths:
7 layers, KDA on 1-3 and 5-6 and MLA on 4 and 7 (1-based, as the release
names them, the last layer MLA), attention-residual blocks of 3 so the last
block is short, a dense first layer, 8 experts of which 2 route, latent
experts, SiTU with the release's betas. Like the release, `A_log` is stored
zero-padded from the KDA heads to the KDA head dim (96 heads to 128 there).
"""

import copy
import importlib
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

REPO = "moonshotai/Kimi-K3"
REVISION = "f831ab66814297da540d832a5235f8e904f29d06"
REMOTE = ("configuration_kimi_k3.py", "modeling_kimi_k3.py", "modeling_kimi_linear.py")
CACHE = Path.home() / ".cache" / "dew" / "research" / "kimi-k3" / f"remote-{REVISION[:7]}"
ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf"
SOURCE = ROOT / "kimi-k3-source"
DESTINATION = ROOT / "kimi-k3-tiny"
LEARNING_RATE = 1e-2


def remote_package():
    """Import the pinned remote code as the package `kimi_k3_remote`."""
    package = CACHE / "kimi_k3_remote"
    package.mkdir(parents=True, exist_ok=True)
    for name in REMOTE:
        path = package / name
        if not path.exists():
            url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
            path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    (package / "__init__.py").touch()
    sys.path.insert(0, str(CACHE))
    return (importlib.import_module("kimi_k3_remote.configuration_kimi_k3"),
            importlib.import_module("kimi_k3_remote.modeling_kimi_k3"),
            importlib.import_module("kimi_k3_remote.modeling_kimi_linear"))


def tiny_config() -> dict:
    config = json.loads((SOURCE / "config.json").read_text())
    text = config["text_config"]
    text.update(hidden_size=64, intermediate_size=96, num_hidden_layers=7,
                num_attention_heads=4, num_key_value_heads=4, vocab_size=512,
                q_lora_rank=32, kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=8,
                v_head_dim=16, num_experts=8, num_experts_per_token=2, moe_intermediate_size=32,
                routed_expert_hidden_size=32, attn_res_block_size=3, max_position_embeddings=512,
                bos_token_id=1, eos_token_id=2, pad_token_id=0)
    text["linear_attn_config"].update(kda_layers=[1, 2, 3, 5, 6], full_attn_layers=[4, 7],
                                      num_heads=2, head_dim=16)
    config["vision_config"].update(vt_hidden_size=32, vt_intermediate_size=64, vt_num_attention_heads=2,
                                   vt_num_hidden_layers=1, mm_hidden_size=32, text_hidden_size=64,
                                   qkv_hidden_size=48, init_pos_emb_height=4, init_pos_emb_width=4,
                                   init_pos_emb_time=2, _attn_implementation="eager")
    config.update(bos_token_id=1, eos_token_id=2, pad_token_id=0, media_placeholder_token_id=5)
    return config


def scatter(model: torch.nn.Module) -> None:
    """Nondegenerate values for every parameter: routing, norms, gates, depth scores."""
    generator = torch.Generator().manual_seed(3181)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            noise = torch.randn(parameter.shape, generator=generator)
            if name.endswith("A_log"):
                parameter.copy_(torch.rand(parameter.shape, generator=generator) * 2 - 1)
            elif name.endswith("dt_bias"):
                parameter.copy_(noise * 0.5)
            elif name.endswith("e_score_correction_bias"):
                parameter.copy_(noise * 0.05)
            elif parameter.ndim == 1:
                parameter.copy_(1 + noise * 0.1)
            else:
                parameter.copy_(noise * 0.08)


def mxfp4(weight: torch.Tensor):
    """compressed-tensors' MXFP4 encoding of a Linear weight and its decoding."""
    from compressed_tensors.compressors.mxfp4.base import MXFP4PackedCompressor
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    from compressed_tensors.quantization.utils.mxfp_utils import generate_mx_scales

    args = QuantizationArgs(num_bits=4, type="float", strategy="group", group_size=32,
                            symmetric=True, scale_dtype=torch.uint8)
    scheme = QuantizationScheme(targets=["Linear"], weights=args)
    rows, columns = weight.shape
    peaks = weight.detach().float().cpu().reshape(rows, columns // 32, 32).abs().amax(-1)
    scale = 2.0 ** (generate_mx_scales(peaks) - 127)
    packed = MXFP4PackedCompressor.compress({"weight": weight.detach().float().cpu(), "weight_scale": scale}, scheme)
    decoded = MXFP4PackedCompressor.decompress(dict(packed), scheme)["weight"].float()
    return packed["weight_packed"].contiguous(), packed["weight_scale"].contiguous(), decoded


def main() -> None:
    if os.environ.get("TRITON_F32_DEFAULT") != "ieee":
        raise SystemExit("run with TRITON_F32_DEFAULT=ieee so fla's kernels keep fp32")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    configuration, modeling, linear = remote_package()
    config = tiny_config()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(3180)
    model = modeling.KimiK3ForConditionalGeneration(configuration.KimiK3Config(**copy.deepcopy(config))).float()
    model.language_model.config._attn_implementation = "eager"
    scatter(model)
    tensors = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".block_sparse_moe.experts." in name:
                packed, scale, decoded = mxfp4(parameter)
                parameter.copy_(decoded)
                stem = name.removesuffix(".weight")
                tensors[stem + ".weight_packed"], tensors[stem + ".weight_scale"] = packed, scale
            elif name.endswith(".self_attn.A_log"):
                width = config["text_config"]["linear_attn_config"]["head_dim"]
                tensors[name] = torch.nn.functional.pad(parameter.detach().cpu(), (0, width - parameter.numel()))
            else:
                tensors[name] = parameter.detach().cpu().contiguous().clone()
    save_file(tensors, str(DESTINATION / "model.safetensors"))
    (DESTINATION / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    generation = {"bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0}
    (DESTINATION / "generation_config.json").write_text(json.dumps(generation, indent=2) + "\n")

    model = model.cuda().eval()
    moe = linear.KimiSparseMoeBlock
    moe.moe_infer = moe.moe_infer.__wrapped__
    rng = np.random.default_rng(3182)
    ids = rng.integers(3, 512, (2, 70))
    mask = np.ones_like(ids)
    ids[1, :9], mask[1, :9] = 0, 0
    input_ids = torch.tensor(ids, device="cuda")
    attention_mask = torch.tensor(mask, device="cuda")
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    valid = attention_mask.bool()
    target_valid = valid[:, 1:] & valid[:, :-1]
    ce = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                           input_ids[:, 1:].reshape(-1), reduction="none")
    loss = (ce.reshape(target_valid.shape) * target_valid).sum() / target_valid.sum()
    loss.backward()
    with torch.no_grad():
        moved = 0
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and name.startswith("language_model."):
                parameter.add_(parameter.grad, alpha=-LEARNING_RATE)
                moved += 1
        updated = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    print("parameters moved", moved, "loss", float(loss),
          "update changed logits by", float((updated - logits).abs()[valid].max()))
    model.zero_grad(set_to_none=True)
    for name, parameter in model.named_parameters():
        if name in tensors:
            source = tensors[name]
            with torch.no_grad():
                parameter.copy_(source[:parameter.numel()].to(parameter) if name.endswith("A_log") else source.to(parameter))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".block_sparse_moe.experts." in name:
                stem = name.removesuffix(".weight")
                parameter.copy_(mxfp4_decode(tensors[stem + ".weight_packed"], tensors[stem + ".weight_scale"]).to(parameter))
        again = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        assert torch.equal(again, logits), "restoring the written weights must restore the logits"
        # The wrapper carries no GenerationMixin under transformers 4.56.2;
        # without pixels it runs the language model on its embeddings
        # (modeling_kimi_k3.py:1145-1218), so that model generates.
        generated = model.language_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=4,
                                   do_sample=False, eos_token_id=None, pad_token_id=0,
                                   output_logits=True, return_dict_in_generate=True)
    grid = torch.linspace(-80, 80, 4001, dtype=torch.float32)
    situ = linear.SituAndMul(config["text_config"]["activation_situ_beta"],
                             config["text_config"]["activation_situ_linear_beta"])
    gate, up = torch.meshgrid(grid[::40], grid[::40], indexing="ij")
    situ_out = situ(torch.cat([gate.reshape(-1, 1), up.reshape(-1, 1)], dim=-1))
    np.savez(DESTINATION / "reference.npz", input_ids=ids, attention_mask=mask,
             logits=logits.detach().cpu().numpy(), loss=np.float32(loss.detach().cpu()),
             learning_rate=np.float32(LEARNING_RATE), updated_logits=updated.cpu().numpy(),
             generated=generated.sequences.cpu().numpy(),
             step_logits=torch.stack(generated.logits, 1).cpu().numpy(),
             situ_gate=gate.reshape(-1).numpy(), situ_up=up.reshape(-1).numpy(),
             situ=situ_out.reshape(-1).numpy())
    print("wrote", DESTINATION, "tensors", len(tensors))


def mxfp4_decode(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """compressed-tensors' decompression of one packed pair."""
    from compressed_tensors.compressors.mxfp4.base import MXFP4PackedCompressor
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    args = QuantizationArgs(num_bits=4, type="float", strategy="group", group_size=32,
                            symmetric=True, scale_dtype=torch.uint8)
    scheme = QuantizationScheme(targets=["Linear"], weights=args)
    return MXFP4PackedCompressor.decompress({"weight_packed": packed, "weight_scale": scale}, scheme)["weight"].float()


if __name__ == "__main__":
    main()
