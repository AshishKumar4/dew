#!/usr/bin/env python3
"""The lm-engine fixtures tests/test_lm_engine_parity.py checks against, from
lm-engine 45b6b57b (https://github.com/open-lm-engine/lm-engine) on CPU in
float64.

    git clone https://github.com/open-lm-engine/lm-engine /tmp/lm-engine
    git -C /tmp/lm-engine checkout 45b6b57b
    uv venv -p 3.12 venv && VIRTUAL_ENV=venv uv pip install torch==2.10.0 \\
        numpy pydantic==2.13.4 transformers==4.57.3 gitpython \\
        --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
    VIRTUAL_ENV=venv uv pip install --no-deps -e /tmp/lm-engine
    cd /tmp/lm-engine && venv/bin/python <dew>/tools/lm_engine_reference.py

What lands in tests/fixtures/lm_engine/:

- hybrid.npz: a four-layer `gpt_base` shaped like Rigel (three Mamba-2
  layers and one grouped-query attention layer with exclusive self attention
  and no positional encoding, a top-2 softmax MoE of four SwiGLU experts on
  every layer, tied embeddings, RMSNorm) under muP multipliers m_emb 12,
  m_residual 0.22, m_width 4. Its weights under lm-engine's names, a batch,
  the logits, the LM loss, the summed Switch-plus-0.1-z aux loss, and every
  parameter's gradient of `lm + 0.01 * aux`. Then three steps of torch AdamW
  (betas 0.9/0.95, eps 1e-10, weight decay 0.1) over lm-engine's muP
  parameter groups (configs/param-groups/mup.yml) on its PowerScheduler: the
  gradient each step read (`step<k>/grad:<name>`, step 0's being `grad:`),
  the parameters after it (`step<k>/param:<name>`), and the group each
  parameter fell in. Arrays are stored in float32, scalars in float64.
- schedule.npz: lm-engine's PowerScheduler and LinearScheduler factors at
  every step of short runs, with the arguments that drew them.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from lm_engine.training.arguments import ParamsGroup
from lm_engine.training.hf_adapter import get_causal_lm_class
from lm_engine.training.models import GPTBaseConfig
from lm_engine.training.optimization.lr_scheduler import LinearScheduler, PowerScheduler
from lm_engine.training.optimization.params_group import get_param_groups_with_names

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "lm_engine"
M_WIDTH = 4.0
COEF = 0.01


def hybrid_config() -> GPTBaseConfig:
    attention = {"sequence_mixer_type": "softmax_attention", "add_bias": False,
                 "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                 "softmax_dropout": 0, "dropout": 0, "attention_multiplier": None,
                 "attention_multiplier_method": "1 / sqrt(head_dim)", "attention_gate": False,
                 "exclusive_self_attention": True, "sliding_window": None}
    mamba = {"sequence_mixer_type": "mamba2", "state_size": 8, "intermediate_size": 32,
             "num_heads": 4, "conv_kernel_size": 4, "activation_function": "silu",
             "num_groups": 1, "chunk_size": 8, "normalization_function": "rmsnorm"}
    moe = {"mlp_type": "MoE", "num_experts": 4, "num_experts_per_tok": 2, "normalized_topk": True,
           "activation_function": "swiglu", "add_bias": False, "intermediate_size": 8,
           "dropout": 0, "shared_intermediate_size": None, "shared_expert_gating": False}
    return GPTBaseConfig(
        vocab_size=64, max_position_embeddings=64, hidden_size=32, num_layers=4,
        position_embedding_type="nope", normalization_function="rmsnorm", layer_norm_epsilon=1e-5,
        embedding_dropout=0, initializer_range=0.02, rope_theta=10000, rope_scaling=None,
        init_method="mup", embedding_init_method="mup", use_depth_scaled_init=True,
        router_aux_loss_coef=COEF, rope_dim=None, tie_word_embeddings=True,
        bos_token_id=0, eos_token_id=1, pad_token_id=2, m_emb=12.0, m_width=M_WIDTH,
        m_residual=0.22, sequence_mixer_blocks=[mamba, mamba, mamba, attention],
        mlp_blocks=[moe] * 4)


class _Wrapped:
    """What `get_param_groups_with_names` reads off a ModelWrapper."""

    def __init__(self, model):
        self.model, self.config = model, model.config

    def has_teacher_model(self) -> bool:
        return False

    def named_parameters(self):
        return self.model.named_parameters()

    def named_modules(self):
        return self.model.named_modules()


def loss_of(model, tokens):
    out = model(input_ids=tokens)
    labels = tokens[:, 1:]
    lm = torch.nn.functional.cross_entropy(
        out.logits[:, :-1].reshape(-1, out.logits.size(-1)), labels.reshape(-1))
    return out, lm, lm + COEF * out.aux_loss


def write_hybrid() -> None:
    torch.manual_seed(0)
    config = hybrid_config()
    model = get_causal_lm_class(config.model_type)(config, use_padding_free_transformer=False).double()
    with torch.no_grad():
        # Weights eight times lm-engine's init, so exclusion, routing and the
        # scan act on values well away from zero.
        for name, parameter in model.named_parameters():
            if parameter.ndim >= 2 and "wte" not in name:
                parameter.mul_(8.0)
    model.train()
    tokens = torch.randint(0, config.vocab_size, (2, 24))
    dump = {"tokens": tokens.numpy()}
    for name, parameter in model.named_parameters():
        dump["param:" + name] = parameter.detach().numpy().astype(np.float32)
    out, lm, total = loss_of(model, tokens)
    total.backward()
    dump.update(logits=out.logits.detach().numpy().astype(np.float32), lm_loss=lm.item(),
                aux_loss=out.aux_loss.item(), total=total.item())
    for name, parameter in model.named_parameters():
        dump["grad:" + name] = parameter.grad.numpy().astype(np.float32)

    groups_file = Path("configs/param-groups/mup.yml")
    groups = [ParamsGroup(**group) for group in yaml.safe_load(groups_file.read_text())]
    args = {"lr": 0.01, "betas": (0.9, 0.95), "eps": 1e-10, "weight_decay": 0.1}
    named = get_param_groups_with_names(_Wrapped(model), args, groups)
    membership = {name: group.name for group in named.params_groups
                  for name in group.parameter_name_map}
    optimizer = torch.optim.AdamW(named.to_torch_compatible_params_groups(), **args)
    schedule = PowerScheduler(optimizer, num_warmup_steps=2, num_constant_steps=0,
                              num_decay_steps=None, num_training_steps=3, lr_decay_factor=0.0,
                              a=0.05, b=-0.51, c=16.0)
    rates = []
    for step in range(3):
        optimizer.zero_grad()
        _, _, total = loss_of(model, tokens)
        total.backward()
        rates.append(optimizer.param_groups[0]["lr"] * 1.0)
        for name, parameter in model.named_parameters():
            if step:
                dump[f"step{step}/grad:{name}"] = parameter.grad.numpy().astype(np.float32)
        optimizer.step()
        schedule.step()
        for name, parameter in model.named_parameters():
            dump[f"step{step}/param:{name}"] = parameter.detach().numpy().astype(np.float32)
    dump["base_rates"] = np.asarray(rates)
    dump["groups"] = np.asarray(json.dumps(membership))
    np.savez_compressed(OUT / "hybrid.npz", **dump)
    print("hybrid: lm", lm.item(), "aux", float(out.aux_loss))


def write_schedules() -> None:
    parameter = torch.nn.Parameter(torch.zeros(1))
    runs = {}
    power = {"lr": 0.01, "num_warmup_steps": 50, "a": 4 * 1152.0, "b": -0.51, "c": 4096.0 * 1152}
    optimizer = torch.optim.SGD([parameter], lr=power["lr"])
    scheduler = PowerScheduler(optimizer, num_warmup_steps=power["num_warmup_steps"],
                               num_constant_steps=0, num_decay_steps=None,
                               num_training_steps=4000, lr_decay_factor=0.0,
                               a=power["a"], b=power["b"], c=power["c"])
    runs["power"] = np.asarray([power["lr"] * scheduler._lr_lambda(step) for step in range(4001)])
    linear = {"lr": 5e-5, "num_warmup_steps": 25, "num_constant_steps": 40, "num_decay_steps": 200}
    optimizer = torch.optim.SGD([parameter], lr=linear["lr"])
    scheduler = LinearScheduler(optimizer, num_warmup_steps=linear["num_warmup_steps"],
                                num_constant_steps=linear["num_constant_steps"],
                                num_decay_steps=linear["num_decay_steps"],
                                num_training_steps=300, lr_decay_factor=0.0)
    runs["linear"] = np.asarray([linear["lr"] * scheduler._lr_lambda(step) for step in range(301)])
    np.savez(OUT / "schedule.npz", **runs, power_args=np.asarray(json.dumps(power)),
             linear_args=np.asarray(json.dumps(linear)))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    write_hybrid()
    write_schedules()
    sys.exit(0)
