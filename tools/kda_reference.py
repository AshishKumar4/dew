"""Record torch CPU KDA outputs and gradients from Transformers' published functions over check 1's
seeds, and transcribe them in JAX at any precision.

modeling_glm5_next.py at 93c8b7b485963a10800c91f55304db6be211c2bd (v5.16.1)
supplies l2norm and both rules verbatim, without the optional hub-kernel decorators.
Each seed's fp32 inputs (`rule_inputs`) cross two chunks, end with padding and carry a nonzero
initial state. The recording holds check 1's evidence of each rule's output, final state and six
operand gradients over its seeds (`record_case`'s summary), the float32 run's from the float64 one,
which runs under `float64` so the bodies' float32 casts widen too; tests/test_kda.py draws the
inputs again and recomputes the truth with `oracle_chunk` and `oracle_recurrent`, the same bodies
transcribed into JAX, in float64.
Run with torch and NumPy installed: python tools/kda_reference.py [output.npz].
"""

import sys
from functools import partial
from pathlib import Path

import numpy as np

REVISION = "93c8b7b485963a10800c91f55304db6be211c2bd"
SOURCE = f"https://raw.githubusercontent.com/huggingface/transformers/{REVISION}/src/transformers/models/glm5_next/modeling_glm5_next.py"
RECORDED = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "kda" / "recorded_rules.npz"
INPUTS = ("query", "key", "value", "g", "beta", "state")
RULES = ("chunk", "recurrent")
ROWS = 16
"""A draw's rows: a gradient of a few values a row takes many to resolve."""


def rule_inputs(seed: int) -> dict[str, np.ndarray]:
    """A seed's fp32 operands and cotangents, the same on every run: ROWS rows of seven tokens of two
    heads, keys of three and values of four, decays below zero, gates in (0, 1) and a small initial
    state."""
    rng = np.random.default_rng([713, seed])
    arrays = {name: rng.normal(size=shape).astype(np.float32) for name, shape in (
        ("query", (ROWS, 7, 2, 3)), ("key", (ROWS, 7, 2, 3)), ("value", (ROWS, 7, 2, 4)),
        ("g", (ROWS, 7, 2, 3)), ("beta", (ROWS, 7, 2)), ("state", (ROWS, 2, 3, 4)))}
    arrays["g"] = -np.abs(arrays["g"]) * np.float32(.7)
    arrays["beta"] = 1 / (1 + np.exp(-arrays["beta"]))
    arrays["state"] *= np.float32(.3)
    arrays["output_cotangent"] = rng.normal(size=(ROWS, 7, 2, 4)).astype(np.float32)
    arrays["state_cotangent"] = rng.normal(size=(ROWS, 2, 3, 4)).astype(np.float32)
    return arrays


def l2norm(x, eps=1e-6):
    """modeling_glm5_next.py's l2norm, in `x`'s dtype."""
    import jax.numpy as jnp

    return x / jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def oracle_recurrent(query, key, value, g, beta, state):
    """`recurrent_kimi_delta_attention` (modeling_glm5_next.py:428-478) in JAX at its operands' dtype,
    operands `[B, S, H, D]`, `g` `[B, S, H, Dk]`."""
    import jax.numpy as jnp

    query, key = l2norm(query), l2norm(key)
    query = query / np.sqrt(query.shape[-1])
    outputs, s = [], state
    for i in range(query.shape[1]):
        s = s * jnp.exp(g[:, i])[..., None]
        kv_mem = (s * key[:, i][..., None]).sum(-2)
        delta = (value[:, i] - kv_mem) * beta[:, i][..., None]
        s = s + key[:, i][..., None] * delta[..., None, :]
        outputs.append((s * query[:, i][..., None]).sum(-2))
    return jnp.stack(outputs, axis=1), s


def oracle_chunk(query, key, value, g, beta, state, chunk_size=4):
    """`chunk_kimi_delta_attention` (modeling_glm5_next.py:482-578) in JAX at its operands' dtype, its
    row correction loop included."""
    import jax.numpy as jnp

    query, key, value, beta, g = (jnp.swapaxes(x, 1, 2) for x in (query, key, value, beta, g))
    query, key = l2norm(query), l2norm(key)
    batch, heads, length, dk = key.shape
    dv = value.shape[-1]
    pad = (chunk_size - length % chunk_size) % chunk_size
    total = length + pad
    query = jnp.pad(query, ((0, 0), (0, 0), (0, pad), (0, 0))) / np.sqrt(dk)
    key, value, g = (jnp.pad(x, ((0, 0), (0, 0), (0, pad), (0, 0))) for x in (key, value, g))
    beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad)))
    v_beta, k_beta = value * beta[..., None], key * beta[..., None]
    query, key, value, g, k_beta, v_beta = (
        x.reshape(batch, heads, -1, chunk_size, x.shape[-1]) for x in (query, key, value, g, k_beta, v_beta))
    g = jnp.cumsum(g, axis=-2)
    decay_mask = jnp.exp(g[..., :, None, :] - g[..., None, :, :])
    attn = -(k_beta[..., :, None, :] * key[..., None, :, :] * decay_mask).sum(-1)
    attn = jnp.where(np.triu(np.ones((chunk_size, chunk_size), bool), 0), 0, attn)
    for i in range(1, chunk_size):
        row, sub = attn[..., i, :i], attn[..., :i, :i]
        attn = attn.at[..., i, :i].set(row + (row[..., None] * sub).sum(-2))
    attn = attn + jnp.eye(chunk_size, dtype=attn.dtype)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * jnp.exp(g))
    s, core = state, []
    upper = np.triu(np.ones((chunk_size, chunk_size), bool), 1)
    for i in range(total // chunk_size):
        q_i, k_i, g_i = query[:, :, i], key[:, :, i], g[:, :, i]
        attn_intra = (q_i[..., :, None, :] * k_i[..., None, :, :] * decay_mask[:, :, i]).sum(-1)
        attn_intra = jnp.where(upper, 0, attn_intra)
        v_new = value[:, :, i] - k_cumdecay[:, :, i] @ s
        core.append((q_i * jnp.exp(g_i)) @ s + attn_intra @ v_new)
        s = (s * jnp.exp(g_i[:, :, -1])[..., None]
             + jnp.swapaxes(k_i * jnp.exp(g_i[:, :, -1:] - g_i), -1, -2) @ v_new)
    core = jnp.stack(core, axis=2).reshape(batch, heads, -1, dv)[:, :, :length]
    return jnp.swapaxes(core, 1, 2), s


def with_gradients(rule, inputs: dict) -> dict:
    """`rule`'s output and final state on `inputs`' operands, and the six operands' gradients of their
    sum against `inputs`' cotangents, by name, in JAX at the operands' dtype."""
    import jax
    import jax.numpy as jnp

    def loss(*operands):
        out, state = rule(*operands)
        value = jnp.sum(out * inputs["output_cotangent"]) + jnp.sum(state * inputs["state_cotangent"])
        return value, (out, state)

    (_, (out, state)), gradients = jax.value_and_grad(loss, tuple(range(len(INPUTS))), has_aux=True)(
        *(jnp.asarray(inputs[name]) for name in INPUTS))
    return {"output": out, "state": state} | {f"grad/{name}": grad for name, grad in zip(INPUTS, gradients,
                                                                                         strict=True)}


def oracle_rule(rule: str):
    """`rule`'s oracle (`oracle_chunk` in chunks of four, or `oracle_recurrent`)."""
    return partial(oracle_chunk, chunk_size=4) if rule == "chunk" else oracle_recurrent


def reference_functions() -> dict:
    import torch
    import torch.nn.functional as F
    from reference_source import fetched, ran, selected

    functions = selected(fetched(SOURCE), {"l2norm", "chunk_kimi_delta_attention",
                                           "recurrent_kimi_delta_attention"}, filename=SOURCE)
    for node in functions:
        node.decorator_list = []
    return ran(functions, {"torch": torch, "F": F}, SOURCE)


def torch_rule(reference: dict, rule: str, inputs: dict, dtype) -> dict:
    """The published `rule` on `inputs` in torch at `dtype`: its output, final state and the six
    operand gradients, by name."""
    import torch

    operands = [torch.tensor(inputs[name], dtype=dtype, requires_grad=True) for name in INPUTS]
    options = {"chunk_size": 4} if rule == "chunk" else {}
    out, state = reference[f"{rule}_kimi_delta_attention"](
        *operands[:5], initial_state=operands[5], output_final_state=True, use_qk_l2norm_in_kernel=True,
        **options)
    loss = (out * torch.tensor(inputs["output_cotangent"], dtype=dtype)).sum()
    loss = loss + (state * torch.tensor(inputs["state_cotangent"], dtype=dtype)).sum()
    gradients = torch.autograd.grad(loss, operands)
    return {"output": out.detach().numpy(), "state": state.detach().numpy()} | {
        f"grad/{name}": grad.numpy() for name, grad in zip(INPUTS, gradients, strict=True)}


def main(destination: Path) -> None:
    import jax
    import torch
    from diffusers_wan_reference import float64
    from recorded_references import provenance, record_case, save

    torch.set_num_threads(1)
    reference = reference_functions()
    recorded = {}
    with jax.enable_x64(new_val=True):
        for rule in RULES:
            def evaluate(inputs, rule=rule):
                found = torch_rule(reference, rule, inputs, torch.float32)
                with float64():
                    truth = torch_rule(reference, rule, inputs, torch.float64)
                return found, truth

            def proxy(inputs, rule=rule):
                return with_gradients(oracle_rule(rule), inputs)

            recorded.update(record_case(f"{rule} rule", rule_inputs, evaluate, proxy, truth_proxy=proxy,
                                        store_inputs=False, summary=True))
            recorded[f"{rule} rule.provenance"] = provenance(
                "Transformers modeling_glm5_next", REVISION, "cpu", "python tools/kda_reference.py",
                torch=torch.__version__)
    written = save(destination, recorded, reference=f"huggingface/transformers {REVISION}",
                   license="Apache-2.0")
    print(f"{written}: {written.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else RECORDED)
