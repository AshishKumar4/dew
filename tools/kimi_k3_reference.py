"""Write the tiny Kimi K3 fixture with the released remote code.

The model is `KimiK3ForConditionalGeneration` from moonshotai/Kimi-K3 at
REVISION: configuration_kimi_k3.py, modeling_kimi_k3.py and
modeling_kimi_linear.py, fetched at that revision through the hub's cache.
Their KDA layers call fla-core's Triton kernels (chunk_kda,
fused_recurrent_kda, ShortConvolution, FusedRMSNormGated), so this runs on a
CUDA device, in fp32, with TRITON_F32_DEFAULT=ieee for every dot that leaves
its precision unset and IEEE forced on the one that sets it (the third item below). The routed
experts are encoded by compressed-tensors' MXFP4 compressor over the scales
its `generate_mx_scales` gives each group's largest magnitude, and the
reference computes with the library's own decompression of them, so the
packed file and the logits describe one model. `calculate_qparams` clamps
the scale exponents and substitutes a finite scale for an all-zero group.

Environment (~/.cache/dew/reference-venvs/kimi-k3): torch 2.8.0+cu128,
transformers 4.56.2, fla-core 0.5.2, compressed-tensors 0.17.1.

Three things change how the reference runs, neither what it computes:

- KimiLinearModel pins `flash_attention_2`; MLA runs the file's own
  `eager_attention_forward` instead (flash-attn is not installed).
- `KimiSparseMoeBlock.moe_infer` is decorated `torch.no_grad`, which would
  cut the routed experts out of the backward pass; the gradient step calls
  the undecorated function. The gate asserts eval mode, so the model stays
  in eval mode, where nothing in the text model is stochastic.
- fla-core 0.5.2's chunk_kda multiplies the blocks of each chunk's inverted
  triangular system with `input_precision='tf32'` on any GPU that has TF32
  (fla/ops/kda/chunk_intra.py:20-23 and 314-351), which TRITON_F32_DEFAULT
  does not reach. `main` sets that constant to 'ieee', the value fla takes
  on a GPU without TF32. At TF32 the logits of tokens early in one of the
  kernel's 16-token sub-chunks missed a float64 forward by up to 1.3e-4,
  five times the 2.3e-5 the IEEE run misses by. Triton keys its disk cache
  on a kernel's source and not on the globals it reads, so these kernels
  compile into a cache directory of their own, which no TF32 build can enter.

The tiny config keeps the release's fields and layer pattern at small widths:
7 layers, KDA on 1-3 and 5-6 and MLA on 4 and 7 (1-based, as the release
names them, the last layer MLA), attention-residual blocks of 3 so the last
block is short, a dense first layer, 8 experts of which 2 route, latent
experts, SiTU with the release's betas. Like the release, `A_log` is stored
zero-padded from the KDA heads to the KDA head dim (96 heads to 128 there).
"""

import copy
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from reference_source import config_lookups, hub_modules, record_padding
from safetensors.torch import save_file

REPO = "moonshotai/Kimi-K3"
REVISION = "f831ab66814297da540d832a5235f8e904f29d06"
REMOTE = ("configuration_kimi_k3.py", "modeling_kimi_k3.py", "modeling_kimi_linear.py")
CACHE = Path.home() / ".cache" / "dew" / "research" / "kimi-k3" / f"remote-{REVISION[:7]}"
ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf"
SOURCE = ROOT / "kimi-k3-source"
DESTINATION = ROOT / "kimi-k3-tiny"
LEARNING_RATE = 1e-2


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


def fixture_batch() -> tuple[np.ndarray, np.ndarray]:
    """Two rows of 70 ids, past one 64-token KDA chunk, the second left-padded by 9."""
    rng = np.random.default_rng(3182)
    ids = rng.integers(3, 512, (2, 70))
    mask = np.ones_like(ids)
    ids[1, :9], mask[1, :9] = 0, 0
    return ids, mask


def sgd_step(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor,
             trained: Callable[[str], bool]):
    """The logits, the mean next-token loss over targets whose input and
    target are valid, and the logits after one SGD step at LEARNING_RATE on
    each parameter `trained` names that the loss reaches. The step stays in
    the model's parameters."""
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
            if parameter.grad is not None and trained(name):
                parameter.add_(parameter.grad, alpha=-LEARNING_RATE)
                moved += 1
        updated = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    print("parameters moved", moved, "loss", float(loss),
          "update changed logits by", float((updated - logits).abs()[valid].max()))
    model.zero_grad(set_to_none=True)
    return logits, loss, updated


def greedy(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Four greedy tokens after each row, and the logits each decode step read."""
    generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=4,
                               do_sample=False, eos_token_id=None, pad_token_id=0,
                               output_logits=True, return_dict_in_generate=True)
    return generated.sequences.cpu().numpy(), torch.stack(generated.logits, 1).cpu().numpy()


def mxfp4(weight: torch.Tensor):
    """compressed-tensors' MXFP4 encoding of a Linear weight and its decoding (`reference_mxfp4`)."""
    import reference_mxfp4

    packed, scale = reference_mxfp4.encoded(weight.detach().float().cpu())
    return packed, scale, reference_mxfp4.decoded(packed, scale)


def record_released_padding() -> None:
    """padding.json for kimi-k3-tiny (`reference_source.record_padding`): the rows the release's own
    token lookup leaves untrained under the fixture's config with its text config's pad id stated, left
    out and null, the language model being built of that text config (modeling_kimi_k3.py:919). It
    builds the model and runs no kernel, so it needs no CUDA device."""
    configuration, modeling, _ = hub_modules("kimi_k3_remote", REPO, REVISION, REMOTE)
    record_padding(*config_lookups(lambda fields: configuration.KimiK3Config(**fields),
                                   modeling.KimiK3ForConditionalGeneration, ("text_config",)),
                   json.loads((DESTINATION / "config.json").read_text()), DESTINATION, ("text_config",))


def released_modules():
    """The pinned configuration, modeling and linear modules, with fla's
    intra-chunk solve held to IEEE fp32."""
    if os.environ.get("TRITON_F32_DEFAULT") != "ieee":
        raise SystemExit("run with TRITON_F32_DEFAULT=ieee so fla's kernels keep fp32")
    os.environ["TRITON_CACHE_DIR"] = str(CACHE / "triton-ieee")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    configuration, modeling, linear = hub_modules("kimi_k3_remote", REPO, REVISION, REMOTE)
    from fla.ops.kda import chunk_intra
    from triton.language import constexpr

    chunk_intra.SOLVE_TRIL_DOT_PRECISION = constexpr("ieee")
    return configuration, modeling, linear


def matmul_precision() -> str:
    """What each array file records of this run's matmuls (`recorded_matmuls`), fla's intra-chunk
    solve among them."""
    from fla.ops.kda import chunk_intra
    from torch_precision import recorded_matmuls

    return recorded_matmuls(**{"the intra-chunk solve": str(getattr(chunk_intra.SOLVE_TRIL_DOT_PRECISION, "value",
                                                                    chunk_intra.SOLVE_TRIL_DOT_PRECISION))})


def load_weights(model: torch.nn.Module, tensors: dict[str, torch.Tensor]) -> None:
    """Copy stored tensors into `model`: MXFP4 experts decoded, A_log cut to its heads."""
    import reference_mxfp4

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".block_sparse_moe.experts." in name:
                stem = name.removesuffix(".weight")
                packed, scale = tensors[stem + ".weight_packed"], tensors[stem + ".weight_scale"]
                source = reference_mxfp4.decoded(packed, scale)
            elif name.endswith("A_log"):
                source = tensors[name].reshape(-1)[:parameter.numel()].reshape(parameter.shape)
            else:
                source = tensors[name]
            parameter.copy_(source.to(parameter))


def main() -> None:
    import reference_mxfp4

    configuration, modeling, linear = released_modules()
    config = tiny_config()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(3180)
    model = modeling.KimiK3ForConditionalGeneration(
        configuration.KimiK3Config(**copy.deepcopy(config))).float()
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
                tensors[name] = torch.nn.functional.pad(
                    parameter.detach().cpu(), (0, width - parameter.numel()))
            else:
                tensors[name] = parameter.detach().cpu().contiguous().clone()
    save_file(tensors, str(DESTINATION / "model.safetensors"))
    (DESTINATION / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    generation = {"bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0}
    (DESTINATION / "generation_config.json").write_text(json.dumps(generation, indent=2) + "\n")

    model = model.cuda().eval()
    moe = linear.KimiSparseMoeBlock
    moe.moe_infer = moe.moe_infer.__wrapped__
    ids, mask = fixture_batch()
    input_ids = torch.tensor(ids, device="cuda")
    attention_mask = torch.tensor(mask, device="cuda")
    logits, loss, updated = sgd_step(
        model, input_ids, attention_mask, lambda name: name.startswith("language_model."))
    for name, parameter in model.named_parameters():
        if name in tensors:
            source = tensors[name]
            with torch.no_grad():
                parameter.copy_(source[:parameter.numel()].to(parameter)
                                if name.endswith("A_log") else source.to(parameter))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".block_sparse_moe.experts." in name:
                stem = name.removesuffix(".weight")
                parameter.copy_(reference_mxfp4.decoded(
                    tensors[stem + ".weight_packed"], tensors[stem + ".weight_scale"]).to(parameter))
        again = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        assert torch.equal(again, logits), "restoring the written weights must restore the logits"
        # The wrapper carries no GenerationMixin under transformers 4.56.2;
        # without pixels it runs the language model on its embeddings
        # (modeling_kimi_k3.py:1145-1218), so that model generates.
        generated, step_logits = greedy(model.language_model, input_ids, attention_mask)
    grid = torch.linspace(-80, 80, 4001, dtype=torch.float32)
    situ = linear.SituAndMul(config["text_config"]["activation_situ_beta"],
                             config["text_config"]["activation_situ_linear_beta"])
    gate, up = torch.meshgrid(grid[::40], grid[::40], indexing="ij")
    situ_out = situ(torch.cat([gate.reshape(-1, 1), up.reshape(-1, 1)], dim=-1))
    np.savez(DESTINATION / "reference.npz", input_ids=ids, attention_mask=mask,
             logits=logits.detach().cpu().numpy(), loss=np.float32(loss.detach().cpu()),
             learning_rate=np.float32(LEARNING_RATE), updated_logits=updated.cpu().numpy(),
             generated=generated, step_logits=step_logits,
             situ_gate=gate.reshape(-1).numpy(), situ_up=up.reshape(-1).numpy(),
             situ=situ_out.reshape(-1).numpy(), precision=matmul_precision())
    print("wrote", DESTINATION, "tensors", len(tensors))


def orders(drawn: Path) -> None:
    """The reference's own rounding draws over the residual orders
    tools/rounding_orders.py wrote under `drawn`, in MXFP4's groups of 32
    (`record_orders`)."""
    configuration, modeling, linear = released_modules()
    config = json.loads((DESTINATION / "config.json").read_text())
    model = modeling.KimiK3ForConditionalGeneration(
        configuration.KimiK3Config(**copy.deepcopy(config))).float()
    model.language_model.config._attn_implementation = "eager"
    model = model.cuda().eval()
    moe = linear.KimiSparseMoeBlock
    moe.moe_infer = moe.moe_infer.__wrapped__
    record_orders(model, drawn, DESTINATION, load_weights, lambda name: name.startswith("language_model."),
                  matmul_precision())


def record_orders(model: torch.nn.Module, drawn: Path, destination: Path,
                  load: Callable[[torch.nn.Module, dict[str, torch.Tensor]], None],
                  trainable: Callable[[str], bool], precision: str) -> None:
    """The SGD step of `model`'s `trainable` parameters over each residual
    order under `drawn`, each order's weights read in by `load`: its updated
    logits' RMS distance from float64 written to `destination`/orders.npz
    beside the orders (tests/reference_error.py's K-order rule). Order 0 is
    the fixture's own run and must reproduce its updated logits bit for bit."""
    from safetensors.torch import load_file

    ids, mask = fixture_batch()
    input_ids, attention_mask = torch.tensor(ids, device="cuda"), torch.tensor(mask, device="cuda")
    valid = mask.astype(bool)
    with np.load(destination / "numerics.npz") as exact, np.load(destination / "reference.npz") as stored:
        truth, recorded = exact["updated_logits_f64"][valid], stored["updated_logits"]
    drawn_orders = np.load(drawn / "orders.npy")
    distances = []
    for k in range(len(drawn_orders)):
        load(model, load_file(str(drawn / str(k) / "model.safetensors")))
        _, _, updated = sgd_step(model, input_ids, attention_mask, trainable)
        updated = updated.cpu().numpy()
        if k == 0:
            assert np.array_equal(updated, recorded), "order 0 must reproduce the fixture's run"
        distances.append(np.sqrt(np.mean(np.square(updated[valid].astype(np.float64) - truth))))
    np.savez(destination / "orders.npz", orders=drawn_orders.astype(np.min_scalar_type(drawn_orders.max())),
             updated_logits=np.asarray(distances), precision=precision)
    print("orders", len(distances), "distances", np.round(np.asarray(distances) / distances[0], 2))


if __name__ == "__main__":
    import sys

    if sys.argv[1:2] == ["orders"]:
        orders(Path(sys.argv[2]))
    elif sys.argv[1:] == ["padding"]:
        record_released_padding()
    else:
        main()
