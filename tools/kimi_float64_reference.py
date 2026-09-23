#!/usr/bin/env python3
"""Write the float64 truth that tests/test_kimi_k3.py and tests/test_kimi_linear.py
measure Dew and the reference from.

tests/reference_error.py measures both runs from the reference evaluated in
float64. The Kimi references compute KDA in fla-core's Triton kernels, which
run in fp32 and bf16 only, so this evaluates each tiny fixture's model in
float64 from the pinned modeling files instead, op for op:
modeling_kimi_linear.py of moonshotai/Kimi-K3 at f831ab6 for kimi-k3-tiny
and modeling_kimi.py of moonshotai/Kimi-Linear-48B-A3B-Instruct at e1df551
for kimi-linear-tiny. KDA is fla's exact token recurrence
(fla/ops/kda/naive.py, `naive_recurrent_kda`) with the gate, l2 norms, beta
sigmoid and gated output norm fla-core computes around it (0.5.2 for K3,
0.4.0 for Kimi Linear). Routing is transcribed for one expert group, which
both fixtures use, and the Kimi Linear gate weighs its experts by the
unbiased scores, as tools/kimi_linear_reference.py patches the reference to.
Nothing here imports Dew or fla. Each row runs over its valid tokens alone,
which is what the reference's unpadded KDA and masked NoPE attention compute.

What lands in numerics.npz beside each fixture's reference.npz, in float64:

- `logits_f64` [rows, tokens, vocab]: the forward over `input_ids`, NaN at
  the padded positions.
- `loss_f64`: the mean next-token cross entropy over the targets whose
  input and target are both valid, as the fixture's `loss`.
- `updated_logits_f64`: the same forward after one SGD step at the
  fixture's learning rate, its gradient taken in float64 over every weight.
  The balancing bias only chooses experts, so its gradient is zero.
- `step_logits_f64` [rows, steps, vocab]: the logits teacher-forced over
  `generated` at the positions its greedy decode scored, as `step_logits`.

Run from the checkout, in Dew's own environment:

    JAX_PLATFORMS=cpu python tools/kimi_float64_reference.py
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file

jax.config.update("jax_enable_x64", val=True)

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf"
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
"""compressed-tensors' E2M1 values by code, the sign in bit 3."""
LATENT_EPS = 1e-6
"""KimiRMSNorm's default epsilon, which MLA's latent norms keep (modeling_kimi_linear.py:368, 383)."""


def load(directory: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """The text config and the decoder's weights in float64 by their names below
    the vision wrapper, each compressed-tensors MXFP4 pair decoded (low nibble
    first, E8M0 scales over 32 inputs) and each `A_log` as its heads."""
    config = json.loads((directory / "config.json").read_text())
    text = config.get("text_config", config)
    prefix = "language_model." if "text_config" in config else ""
    heads = text["linear_attn_config"]["num_heads"]
    raw = load_file(str(directory / "model.safetensors"))
    weights = {}
    for name, value in raw.items():
        if not name.startswith(prefix) or name.endswith(".weight_scale"):
            continue
        name = name.removeprefix(prefix)
        if name.endswith(".weight_packed"):
            name = name.removesuffix("_packed")
            codes = np.stack([value & 15, value >> 4], -1).reshape(value.shape[0], -1)
            exponents = raw[prefix + name + "_scale"].astype(np.float64) - 127
            value = E2M1[codes] * np.exp2(np.repeat(exponents, 32, axis=1))
        elif name.endswith(".A_log"):
            # K3 stores the heads zero-padded to the head dim, Kimi Linear as [1, 1, heads, 1].
            value = value.reshape(-1)[:heads]
        weights[name] = np.asarray(value, np.float64)
    return text, weights


def rms(x, weight, eps):
    return weight * x / jnp.sqrt(jnp.mean(x * x, -1, keepdims=True) + eps)


def short_conv(x, weight):
    """fla's ShortConvolution with silu from a fresh history: weight [channels, 1, width]."""
    width = weight.shape[-1]
    padded = jnp.pad(x, ((width - 1, 0), (0, 0)))
    return jax.nn.silu(sum(weight[:, 0, j] * padded[j:j + x.shape[0]] for j in range(width)))


def delta_rule(q, k, v, g, beta):
    """`naive_recurrent_kda`: each token decays the state by exp(g), writes
    beta k (v - k S) into it, and reads it with q / sqrt(K)."""
    def step(state, token):
        q, k, v, g, beta = token
        state = state * jnp.exp(g)[..., None]
        state = state + jnp.einsum("hk,hv->hkv", beta[:, None] * k, v - jnp.einsum("hk,hkv->hv", k, state))
        return state, jnp.einsum("hk,hkv->hv", q, state)

    heads, dim = q.shape[1:]
    _, out = jax.lax.scan(step, jnp.zeros((heads, dim, v.shape[-1])), (q * dim ** -0.5, k, v, g, beta))
    return out


def kda(x, w, p, text):
    """`KimiDeltaAttention.forward` (modeling_kimi_linear.py:543-664, modeling_kimi.py:505-606)."""
    linear = text["linear_attn_config"]
    tokens, heads, dim = x.shape[0], linear["num_heads"], linear["head_dim"]

    def split(y):
        return y.reshape(tokens, heads, dim)

    q, k, v = (split(short_conv(x @ w[f"{p}{n}_proj.weight"].T, w[f"{p}{n}_conv1d.weight"])) for n in "qkv")
    g = split((x @ w[p + "f_a_proj.weight"].T) @ w[p + "f_b_proj.weight"].T + w[p + "dt_bias"])
    decay = jnp.exp(w[p + "A_log"])[:, None]
    bound = linear.get("gate_lower_bound")
    # fla's naive_kda_gate and naive_kda_lowerbound_gate (fla/ops/kda/gate.py).
    g = -decay * jax.nn.softplus(g) if bound is None else bound * jax.nn.sigmoid(decay * g)
    beta = jax.nn.sigmoid(x @ w[p + "b_proj.weight"].T)
    q, k = (y / jnp.sqrt(jnp.sum(y * y, -1, keepdims=True) + 1e-6) for y in (q, k))
    out = delta_rule(q, k, v, g, beta)
    if linear.get("use_full_rank_gate"):
        gate = x @ w[p + "g_proj.weight"].T
    else:
        gate = (x @ w[p + "g_a_proj.weight"].T) @ w[p + "g_b_proj.weight"].T
    out = rms(out, w[p + "o_norm.weight"], text["rms_norm_eps"]) * jax.nn.sigmoid(split(gate))
    return out.reshape(tokens, heads * dim) @ w[p + "o_proj.weight"].T


def mla(x, w, p, text):
    """`KimiMLAAttention.forward` with NoPE (modeling_kimi_linear.py:405-475, modeling_kimi.py:380-442)."""
    tokens, heads = x.shape[0], text["num_attention_heads"]
    nope, rope, width, rank = (text[key] for key in ("qk_nope_head_dim", "qk_rope_head_dim",
                                                     "v_head_dim", "kv_lora_rank"))
    if text.get("q_lora_rank") is None:
        query = x @ w[p + "q_proj.weight"].T
    else:
        query = rms(x @ w[p + "q_a_proj.weight"].T, w[p + "q_a_layernorm.weight"], LATENT_EPS) @ w[p + "q_b_proj.weight"].T
    query = query.reshape(tokens, heads, nope + rope)
    compressed = x @ w[p + "kv_a_proj_with_mqa.weight"].T
    kv = rms(compressed[:, :rank], w[p + "kv_a_layernorm.weight"], LATENT_EPS) @ w[p + "kv_b_proj.weight"].T
    kv = kv.reshape(tokens, heads, nope + width)
    key = jnp.concatenate([kv[..., :nope], jnp.broadcast_to(compressed[:, None, rank:], (tokens, heads, rope))], -1)
    scores = jnp.einsum("qhd,khd->hqk", query, key) * (nope + rope) ** -0.5
    scores = jnp.where(jnp.tril(jnp.ones((tokens, tokens), bool)), scores, -jnp.inf)
    out = jnp.einsum("hqk,khd->qhd", jax.nn.softmax(scores, -1), kv[..., nope:]).reshape(tokens, heads * width)
    if text.get("mla_use_output_gate"):
        out = out * jax.nn.sigmoid(x @ w[p + "g_proj.weight"].T)
    return out @ w[p + "o_proj.weight"].T


def mlp(x, w, p, text, names=("gate_proj", "up_proj", "down_proj")):
    """A gated MLP under the config's activation: SiTU (`SituAndMul`,
    modeling_kimi_linear.py:64-82) or SwiGLU."""
    gate, up = (x @ w[f"{p}{name}.weight"].T for name in names[:2])
    if text.get("hidden_act") == "situ":
        beta, cap = text.get("activation_situ_beta") or 1.0, text.get("activation_situ_linear_beta")
        product = beta * jnp.tanh(gate / beta) * jax.nn.sigmoid(gate) * (up if cap is None else cap * jnp.tanh(up / cap))
    else:
        product = jax.nn.silu(gate) * up
    return product @ w[f"{p}{names[2]}.weight"].T


def moe(x, w, p, text):
    """`KimiSparseMoeBlock.forward` with `KimiMoEGate` (modeling_kimi_linear.py:703-838):
    the experts chosen on the scores plus the balancing bias, weighed by the
    scores alone, renormalized and scaled; K3's run at the latent width and
    are normed before the up projection; the shared experts read x."""
    top_k = text["num_experts_per_token"]
    logits = x @ w[p + "gate.weight"].T
    sigmoid = text.get("moe_router_activation_func", "sigmoid") == "sigmoid"
    scores = jax.nn.sigmoid(logits) if sigmoid else jax.nn.softmax(logits, -1)
    chosen = jnp.argsort(-(scores + w[p + "gate.e_score_correction_bias"]), -1)[:, :top_k]
    weights = jnp.take_along_axis(scores, chosen, -1)
    if top_k > 1 and text.get("moe_renormalize", True):
        weights = weights / (weights.sum(-1, keepdims=True) + 1e-20)
    weights = weights * text["routed_scaling_factor"]
    latent = text.get("routed_expert_hidden_size") is not None
    inner = x @ w[p + "routed_expert_down_proj.weight"].T if latent else x
    routed = sum(jnp.sum(weights * (chosen == expert), -1, keepdims=True)
                 * mlp(inner, w, f"{p}experts.{expert}.", text, ("w1", "w3", "w2"))
                 for expert in range(text["num_experts"]))
    if latent:
        if text.get("latent_moe_use_norm"):
            routed = rms(routed, w[p + "routed_expert_norm.weight"], text["rms_norm_eps"])
        routed = routed @ w[p + "routed_expert_up_proj.weight"].T
    return routed + mlp(x, w, p + "shared_experts.", text)


def depth_attention(partial, blocks, w, site, eps):
    """`_apply_attn_res` (modeling_kimi_linear.py:1075-1088): the site's
    pseudo-query scores each RMS-normed finished block and the partial sum,
    and their softmax mixes them."""
    values = jnp.stack([*blocks, partial], 1)
    keys = values / jnp.sqrt(jnp.mean(values * values, -1, keepdims=True) + eps)
    query = w[site + "_norm.weight"] * w[site + "_proj.weight"][0]
    return jnp.einsum("tn,tnd->td", jax.nn.softmax(keys @ query, -1), values)


def forward(w, ids, text):
    """The logits over one row's valid ids: pre-norm residuals, or K3's
    Attention Residuals over blocks of `attn_res_block_size` layers
    (`KimiDecoderLayer._forward_attn_residual`, modeling_kimi_linear.py:973-1046,
    and `_apply_output_attn_res`, :1226-1233)."""
    eps, size = text["rms_norm_eps"], text.get("attn_res_block_size")
    hidden = w["model.embed_tokens.weight"][ids]
    blocks = []
    for index in range(text["num_hidden_layers"]):
        p = f"model.layers.{index}."
        mixer = kda if index + 1 in text["linear_attn_config"]["kda_layers"] else mla
        routed = (text.get("num_experts") is not None and index >= text.get("first_k_dense_replace", 0)
                  and index % text.get("moe_layer_freq", 1) == 0)
        feedforward, inner = (moe, p + "block_sparse_moe.") if routed else (mlp, p + "mlp.")
        read = hidden
        if size is not None and blocks:
            read = depth_attention(hidden, blocks, w, p + "self_attention_res", eps)
        opens = size is not None and index % size == 0
        if opens:
            blocks.append(hidden)
        out = mixer(rms(read, w[p + "input_layernorm.weight"], eps), w, p + "self_attn.", text)
        hidden = out if opens else hidden + out
        read = hidden if size is None else depth_attention(hidden, blocks, w, p + "mlp_res", eps)
        hidden = hidden + feedforward(rms(read, w[p + "post_attention_layernorm.weight"], eps), w, inner, text)
    if size is not None:
        hidden = depth_attention(hidden, blocks, w, "model.output_attn_res", eps)
    return rms(hidden, w["model.norm.weight"], eps) @ w["lm_head.weight"].T


def loss(w, rows, text):
    """The mean next-token cross entropy over every row's valid ids."""
    total = sum(-jnp.take_along_axis(jax.nn.log_softmax(forward(w, ids, text)[:-1], -1), ids[1:, None], -1).sum()
                for ids in rows)
    return total / sum(ids.shape[0] - 1 for ids in rows)


def truth(directory: Path) -> dict[str, np.ndarray]:
    text, weights = load(directory)
    with np.load(directory / "reference.npz") as stored:
        fixture = {name: stored[name] for name in stored.files}
    valid = fixture["attention_mask"].astype(bool)
    rows = tuple(jnp.asarray(ids[mask]) for ids, mask in zip(fixture["input_ids"], valid, strict=True))
    run = jax.jit(lambda w, ids: forward(w, ids, text))

    def grid(w):
        logits = np.full((*valid.shape, weights["lm_head.weight"].shape[0]), np.nan)
        for row, ids in enumerate(rows):
            logits[row, valid[row]] = run(w, ids)
        return logits

    value, gradient = jax.jit(jax.value_and_grad(lambda w: loss(w, rows, text)))(weights)
    rate = np.float64(fixture["learning_rate"])
    stepped = {name: weights[name] - rate * np.asarray(gradient[name]) for name in weights}
    steps = fixture["step_logits"].shape[1]
    generated = fixture["generated"]
    reached = np.concatenate([valid, np.ones((valid.shape[0], steps), bool)], 1)
    step_logits = np.stack([np.asarray(run(weights, jnp.asarray(ids[mask][:-1])))[-steps:]
                            for ids, mask in zip(generated, reached, strict=True)])
    return {"logits_f64": grid(weights), "loss_f64": np.float64(value),
            "updated_logits_f64": grid(stepped), "step_logits_f64": step_logits}


def main() -> None:
    for name in ("kimi-k3-tiny", "kimi-linear-tiny"):
        directory = FIXTURES / name
        arrays = truth(directory)
        with np.load(directory / "reference.npz") as stored:
            valid = stored["attention_mask"].astype(bool)
            for key in ("logits", "updated_logits", "step_logits"):
                exact = arrays[f"{key}_f64"]
                reference = stored[key]
                if key != "step_logits":
                    exact, reference = exact[valid], reference[valid]
                print(f"{name} {key}: the reference sits {np.max(np.abs(reference - exact)):.3e} (max) and "
                      f"{np.sqrt(np.mean(np.square(reference - exact))):.3e} (rms) from float64")
            print(f"{name} loss: float64 {arrays['loss_f64']:.9f}, the reference {float(stored['loss']):.9f}")
        np.savez(directory / "numerics.npz", **arrays)


if __name__ == "__main__":
    main()
