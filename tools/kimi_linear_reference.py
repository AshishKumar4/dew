"""Write the tiny Kimi Linear fixture with the released remote code.

The model is `KimiLinearForCausalLM` from moonshotai/Kimi-Linear-48B-A3B-Instruct
at REVISION: configuration_kimi.py and modeling_kimi.py, fetched at that
revision through the hub's cache. Its KDA layers call fla-core's Triton
kernels, whose 0.4.0 `fused_kda_gate` signature the release calls (0.5
changed it), so this runs on a CUDA device, in fp32, with
TRITON_F32_DEFAULT=ieee for every dot that leaves its precision unset.

Environment: tools/kimi_linear_reference.requirements.txt (torch 2.8.0+cu128,
transformers 4.57.1, fla-core 0.4.0).

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
tests/test_router_source.py holds Dew's router to vLLM's own grouped_topk
(tools/vllm_router_reference.py) on Kimi Linear's published router.

The tiny config keeps the release's fields and layer pattern at small widths:
7 layers, KDA on 1-3 and 5-6 and MLA on 4 and 7 (1-based, as the release
names them, the last layer MLA), a dense first layer, then 8 routed experts
of which 2 route, beside the release's one shared expert and its routed
scaling of 2.446. `A_log` is stored in the release's `[1, 1, heads, 1]`.

`truth` writes numerics.npz as kimi_k3_reference.py's does, the released
model as `patched` changes it, in float64 on the CPU, fla 0.4.0's kernels
answered by its own torch references, its separate gate by `kda_gate_ref`.
"""

import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
from kimi_k3_reference import (
    LEARNING_RATE,
    fixture_on_gpu,
    float64_truth,
    greedy,
    record_orders,
    record_picks,
    scatter,
    sgd_step,
    write_fixture,
    write_truth,
)
from reference_source import config_lookups, hub_modules, record_padding

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


def matmul_precision() -> str:
    """What each array file records of this run's matmuls (`recorded_matmuls`), fla's triangular
    solve among them. `released_modules` holds fla's autotuned recompute to its 'ieee'
    configurations."""
    from torch_precision import recorded_matmuls

    return recorded_matmuls(FLA_TRIL_PRECISION=os.environ.get("FLA_TRIL_PRECISION", "ieee"))


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


def remote_modules():
    """The pinned configuration and modeling modules as the release writes them, transformers'
    `auto_docstring` the identity while they import (the first of the changes above)."""
    import transformers.utils

    documented = transformers.utils.auto_docstring
    transformers.utils.auto_docstring = lambda obj=None, **kwargs: obj if obj is not None else (lambda inner: inner)
    try:
        return hub_modules("kimi_linear_remote", REPO, REVISION, REMOTE)
    finally:
        transformers.utils.auto_docstring = documented


def record_released_padding() -> None:
    """padding.json for kimi-linear-tiny (`reference_source.record_padding`): the rows the release's own
    token lookup leaves untrained under the fixture's config with its pad id stated, left out and
    null. It builds the model and runs no kernel, so it needs no CUDA device."""
    configuration, modeling = remote_modules()
    record_padding(*config_lookups(lambda fields: configuration.KimiLinearConfig(**fields),
                                   modeling.KimiLinearForCausalLM),
                   json.loads((DESTINATION / "config.json").read_text()), DESTINATION)


def patched():
    """The pinned modules (`remote_modules`) with the gate weighing by unshifted scores and the routed experts
    in the backward pass."""
    configuration, modeling = remote_modules()
    modeling.KimiMoEGate.forward = unbiased_gate(modeling.KimiMoEGate.forward)
    moe = modeling.KimiSparseMoeBlock
    moe.moe_infer = moe.moe_infer.__wrapped__
    return configuration, modeling


def released_modules():
    """The pinned modules as `patched` changes them, with fla's kernels held to IEEE fp32."""
    if os.environ.get("TRITON_F32_DEFAULT") != "ieee":
        raise SystemExit("run with TRITON_F32_DEFAULT=ieee so fla's kernels keep fp32")
    if os.environ.get("FLA_TRIL_PRECISION", "ieee") != "ieee":
        raise SystemExit("leave FLA_TRIL_PRECISION at its 'ieee' default")
    os.environ["TRITON_CACHE_DIR"] = str(CACHE / "triton-ieee")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    configuration, modeling = patched()
    from fla.ops.kda import wy_fast

    assert ieee_only(wy_fast), "fla's recompute_w_u no longer autotunes its dot precision"
    return configuration, modeling


def main() -> None:
    configuration, modeling = released_modules()
    config = tiny_config()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(3280)
    model = modeling.KimiLinearForCausalLM(configuration.KimiLinearConfig(**copy.deepcopy(config))).float()
    model.config._attn_implementation = "eager"
    scatter(model)
    tensors = {name: parameter.detach().cpu().contiguous().clone() for name, parameter in model.named_parameters()}
    write_fixture(DESTINATION, tensors, config)
    model, ids, mask, input_ids, attention_mask = fixture_on_gpu(model)
    logits, loss, updated = sgd_step(model, input_ids, attention_mask, lambda name: True)
    with torch.no_grad():
        load_weights(model, tensors)
        again = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        assert torch.equal(again, logits), "restoring the written weights must restore the logits"
        generated, step_logits = greedy(model, input_ids, attention_mask)
    np.savez(DESTINATION / "reference.npz", input_ids=ids, attention_mask=mask,
             logits=logits.detach().cpu().numpy(), loss=np.float32(loss.detach().cpu()),
             learning_rate=np.float32(LEARNING_RATE), updated_logits=updated.cpu().numpy(),
             generated=generated, step_logits=step_logits, precision=matmul_precision())
    record_picks(DESTINATION)
    print("wrote", DESTINATION, "tensors", len(tensors))


def load_weights(model: torch.nn.Module, tensors: dict[str, torch.Tensor]) -> None:
    """Copy stored tensors into `model`'s parameters, each in its parameter's dtype and device."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(tensors[name].to(parameter))


def truth(directory: Path) -> dict[str, np.ndarray]:
    """The float64 truth of the Kimi Linear fixture at `directory` (`kimi_k3_reference.float64_truth`): the
    released model as `patched` changes it, built and run in float64 on the CPU (`torch_precision.float64`),
    fla's kernels answered by its own references at their call boundaries (`fla_float64`), holding the
    fixture's weights, every parameter trained."""
    from fla_float64 import installed
    from safetensors.torch import load_file
    from torch_precision import float64

    configuration, modeling = patched()
    config = json.loads((directory / "config.json").read_text())
    tensors = load_file(str(directory / "model.safetensors"))
    with installed(modeling), float64():
        model = modeling.KimiLinearForCausalLM(
            configuration.KimiLinearConfig(**copy.deepcopy(config))).double()
        model.config._attn_implementation = "eager"
        load_weights(model, tensors)
        return float64_truth(model.eval(), directory, lambda name: True)


def orders(drawn: Path) -> None:
    """The reference's own rounding draws over the residual orders
    tools/rounding_orders.py wrote under `drawn` (`record_orders`), every
    parameter trained."""
    configuration, modeling = released_modules()
    config = json.loads((DESTINATION / "config.json").read_text())
    model = modeling.KimiLinearForCausalLM(configuration.KimiLinearConfig(**copy.deepcopy(config))).float()
    model.config._attn_implementation = "eager"
    record_orders(model, drawn, DESTINATION, load_weights, lambda name: True, matmul_precision())

if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["padding"]:
        record_released_padding()
    elif sys.argv[1:2] == ["orders"]:
        orders(Path(sys.argv[2]))
    elif sys.argv[1:2] == ["truth"]:
        write_truth(Path(sys.argv[2]) if len(sys.argv) > 2 else DESTINATION, truth)
    else:
        main()
