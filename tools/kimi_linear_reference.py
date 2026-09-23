"""Write the tiny Kimi Linear fixture with the released remote code.

The model is `KimiLinearForCausalLM` from moonshotai/Kimi-Linear-48B-A3B-Instruct
at REVISION: configuration_kimi.py and modeling_kimi.py, fetched at that
revision into CACHE. Its KDA layers call fla-core's Triton kernels, whose
0.4.0 `fused_kda_gate` signature the release calls (0.5 changed it), so this
runs on a CUDA device, in fp32, with TRITON_F32_DEFAULT=ieee for every dot
that leaves its precision unset.

Environment (~/.cache/dew/reference-venvs/kimi-linear): torch 2.8.0+cu128,
transformers 4.57.1, fla-core 0.4.0.

Four things change how the reference runs, not what it computes:

- `transformers.utils.auto_docstring`, which only writes docstrings, is the
  identity while the file imports: transformers 4.57.1 cannot format the
  file's `X | None` annotations (a types.UnionType has no __name__), and
  K3's revision of the file comments the decorator out
  (modeling_kimi_linear.py:1137, 1254 of moonshotai/Kimi-K3 at f831ab6).
- MLA runs the file's own `eager_attention_forward` (flash-attn is not
  installed).
- `KimiSparseMoeBlock.moe_infer` is decorated `torch.no_grad`, which would
  cut the routed experts out of the backward pass; the gradient step calls
  the undecorated function. The gate asserts eval mode, so the model stays
  in eval mode, where nothing in it is stochastic.
- Two fla-core 0.4.0 kernels set their dots' precision. The intra-chunk
  solve reads FLA_TRIL_PRECISION, 'ieee' unless the environment says
  otherwise (fla/ops/utils/solve_tril.py:15), and `recompute_w_u_fwd_kernel`
  autotunes between 'tf32x3' and 'ieee' on a GPU with TF32
  (fla/ops/kda/wy_fast.py:20-29). `main` requires the first and holds the
  second, and any other wy_fast kernel that tunes it, to 'ieee'.

One change to what the reference computes: the released gate adds the
balancing bias to its scores in place (modeling_kimi.py:664-665), so the
routing weights it gathers from those scores (:689) carry the bias. K3's
revision of the same file adds it out of place (modeling_kimi_linear.py:723
of moonshotai/Kimi-K3 at f831ab6), and vLLM, the engine the model card
serves with, weighs by the unshifted scores (vllm 0.30.0,
model_executor/layers/fused_moe/router/grouped_topk_router.py:123-124, 150).
`unbiased_gate` keeps the released selection and weighs it by the
unshifted scores, as Dew's router does, and checks on every call that the
released weights differ from these by the gathered bias and nothing else.

The tiny config keeps the release's fields and layer pattern at small widths:
7 layers, KDA on 1-3 and 5-6 and MLA on 4 and 7 (1-based, as the release
names them, the last layer MLA), a dense first layer, then 8 routed experts
of which 2 route, beside the release's one shared expert and its routed
scaling of 2.446. `A_log` is stored in the release's `[1, 1, heads, 1]`.
"""

import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
from kimi_k3_reference import LEARNING_RATE, fixture_batch, greedy, remote_modules, scatter, sgd_step
from safetensors.torch import save_file

REPO = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
REVISION = "e1df551a447157d4658b573f9a695d57658590e9"
REMOTE = ("configuration_kimi.py", "modeling_kimi.py")
CACHE = Path.home() / ".cache" / "dew" / "research" / "kimi-linear" / f"remote-{REVISION[:7]}"
ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf"
SOURCE = ROOT / "kimi-linear-source"
DESTINATION = ROOT / "kimi-linear-tiny"


def tiny_config() -> dict:
    config = json.loads((SOURCE / "config.json").read_text())
    config.update(hidden_size=64, intermediate_size=96, num_hidden_layers=7, num_attention_heads=4,
                  num_key_value_heads=4, head_dim=16, vocab_size=512, kv_lora_rank=16,
                  qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16, num_experts=8,
                  num_experts_per_token=2, moe_intermediate_size=32,
                  bos_token_id=1, eos_token_id=2, pad_token_id=0)
    config["linear_attn_config"].update(kda_layers=[1, 2, 3, 5, 6], full_attn_layers=[4, 7],
                                        num_heads=2, head_dim=16)
    return config


def ieee_only(module) -> int:
    """Hold each autotuned kernel of `module` that tries dot precisions to its
    'ieee' configurations; the count of kernels held."""
    from triton.runtime.autotuner import Autotuner, Heuristics

    held = 0
    for kernel in vars(module).values():
        while isinstance(kernel, Heuristics):
            kernel = kernel.fn
        if isinstance(kernel, Autotuner) and any("DOT_PRECISION" in config.kwargs for config in kernel.configs):
            kernel.configs = [config for config in kernel.configs if config.kwargs["DOT_PRECISION"] == "ieee"]
            held += 1
    return held


def unbiased_gate(released):
    """`KimiMoEGate.forward` with the released selection weighed by the
    unshifted scores.

    Each call runs the released forward without its renormalization and
    scaling and asserts that it chose the same experts (it is what chooses
    them) with weights that are the unshifted ones plus the gathered bias,
    bit for bit: the patch changes the routing weights by the bias shift
    and nothing else.
    """

    def forward(self, hidden_states):
        renormalize, scaling = self.moe_renormalize, self.routed_scaling_factor
        self.moe_renormalize, self.routed_scaling_factor = False, 1.0
        try:
            with torch.no_grad():
                topk_idx, shifted = released(self, hidden_states)
        finally:
            self.moe_renormalize, self.routed_scaling_factor = renormalize, scaling
        logits = torch.nn.functional.linear(
            hidden_states.reshape(-1, hidden_states.shape[-1]).type(torch.float32), self.weight.type(torch.float32))
        scores = logits.sigmoid() if self.moe_router_activation_func == "sigmoid" else logits.softmax(dim=1)
        weight = scores.gather(1, topk_idx)
        with torch.no_grad():
            assert torch.equal(shifted, weight + self.e_score_correction_bias[topk_idx]), \
                "the released weights must be the unshifted ones plus the gathered bias"
        if self.top_k > 1 and self.moe_renormalize:
            weight = weight / (weight.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_idx, weight * self.routed_scaling_factor

    return forward


def main() -> None:
    if os.environ.get("TRITON_F32_DEFAULT") != "ieee":
        raise SystemExit("run with TRITON_F32_DEFAULT=ieee so fla's kernels keep fp32")
    if os.environ.get("FLA_TRIL_PRECISION", "ieee") != "ieee":
        raise SystemExit("leave FLA_TRIL_PRECISION at its 'ieee' default")
    os.environ["TRITON_CACHE_DIR"] = str(CACHE / "triton-ieee")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import transformers.utils

    documented = transformers.utils.auto_docstring
    transformers.utils.auto_docstring = lambda obj=None, **kwargs: obj if obj is not None else (lambda inner: inner)
    try:
        configuration, modeling = remote_modules(REPO, REVISION, CACHE, "kimi_linear_remote", REMOTE)
    finally:
        transformers.utils.auto_docstring = documented
    from fla.ops.kda import wy_fast

    assert ieee_only(wy_fast), "fla's recompute_w_u no longer autotunes its dot precision"
    modeling.KimiMoEGate.forward = unbiased_gate(modeling.KimiMoEGate.forward)
    moe = modeling.KimiSparseMoeBlock
    moe.moe_infer = moe.moe_infer.__wrapped__
    config = tiny_config()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(3280)
    model = modeling.KimiLinearForCausalLM(configuration.KimiLinearConfig(**copy.deepcopy(config))).float()
    model.config._attn_implementation = "eager"
    scatter(model)
    tensors = {name: parameter.detach().cpu().contiguous().clone() for name, parameter in model.named_parameters()}
    save_file(tensors, str(DESTINATION / "model.safetensors"))
    (DESTINATION / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    generation = {"bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0}
    (DESTINATION / "generation_config.json").write_text(json.dumps(generation, indent=2) + "\n")

    model = model.cuda().eval()
    ids, mask = fixture_batch()
    input_ids = torch.tensor(ids, device="cuda")
    attention_mask = torch.tensor(mask, device="cuda")
    logits, loss, updated = sgd_step(model, input_ids, attention_mask, lambda name: True)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(tensors[name].to(parameter))
        again = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        assert torch.equal(again, logits), "restoring the written weights must restore the logits"
        generated, step_logits = greedy(model, input_ids, attention_mask)
    np.savez(DESTINATION / "reference.npz", input_ids=ids, attention_mask=mask,
             logits=logits.detach().cpu().numpy(), loss=np.float32(loss.detach().cpu()),
             learning_rate=np.float32(LEARNING_RATE), updated_logits=updated.cpu().numpy(),
             generated=generated, step_logits=step_logits)
    print("wrote", DESTINATION, "tensors", len(tensors))


if __name__ == "__main__":
    main()
