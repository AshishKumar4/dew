"""mamba_ssm's Mamba-3 checkpoints (`MambaLMHeadModel`, its `ssm_cfg` layer Mamba3) read as the mamba3 family:
a tiny checkpoint in mamba_ssm's names and config, loaded (`Pretrained.load`), on par with mamba_ssm's model
as its modules compute it on the stored tensors (check 1)."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import as_given, assert_as_exact
from safetensors.numpy import save_file

from dew.interop.pretrained import Pretrained
from dew.nn.mixers.mamba3 import recurrent_mamba3

WIDTH, INTERMEDIATE, LAYERS, VOCAB, PADDED = 32, 48, 2, 50, 64
HEAD_DIM, STATE = 16, 16
ANGLES = STATE // 4
"""The pairs that turn at `rope_fraction` 0.5 (mamba_ssm's `num_rope_angles`)."""
SEEDS = 8
VARIANTS = {"siso": {"is_mimo": False, "is_outproj_norm": False, "chunk_size": 16},
            "mimo-normed": {"is_mimo": True, "mimo_rank": 2, "is_outproj_norm": True, "chunk_size": 8}}
"""The layers the checkpoints hold: the 187m release's (SISO, no out-projection norm), and MIMO of rank two
under the out-projection norm."""


def checkpoint_config(variant: str) -> dict:
    """The checkpoint's config.json as mamba_ssm writes it (mamba_ssm/models/config_mamba.py)."""
    return {"d_model": WIDTH, "d_intermediate": INTERMEDIATE, "n_layer": LAYERS, "vocab_size": VOCAB,
            "ssm_cfg": {"layer": "Mamba3", "d_state": STATE, "expand": 2, "headdim": HEAD_DIM, "ngroups": 1,
                        "rope_fraction": 0.5, "A_floor": 1e-4, **VARIANTS[variant]},
            "attn_layer_idx": [], "attn_cfg": {}, "rms_norm": True, "residual_in_fp32": True,
            "fused_add_norm": True, "pad_vocab_size_multiple": 16, "tie_embeddings": True}


def rank_of(variant: str) -> int:
    return VARIANTS[variant].get("mimo_rank", 1) if VARIANTS[variant]["is_mimo"] else 1


def checkpoint_tensors(variant: str) -> dict:
    """The checkpoint's fp32 tensors in mamba_ssm's names and shapes (mamba_ssm/modules/mamba3.py, its
    `Block` and `GatedMLP`), drawn about their init's scale; the head the embedding's copy."""
    normal = np.random.default_rng([3, len(variant)]).standard_normal
    rank, heads, inner = rank_of(variant), 2 * WIDTH // HEAD_DIM, 2 * WIDTH
    projected = 2 * inner + 2 * STATE * rank + 3 * heads + ANGLES
    tensors = {"backbone.embedding.weight": normal((PADDED, WIDTH)),
               "backbone.norm_f.weight": 1 + 0.1 * normal((WIDTH,))}
    for layer in range(LAYERS):
        mixer = {"in_proj.weight": normal((projected, WIDTH)) / np.sqrt(WIDTH),
                 "dt_bias": np.log(np.expm1(np.exp(np.random.default_rng(layer).uniform(-6.9, -2.3, heads)))),
                 **{f"{name}_bias": 1 + 0.1 * normal((heads, rank, STATE)) for name in ("B", "C")},
                 "B_norm.weight": 1 + 0.1 * normal((STATE,)), "C_norm.weight": 1 + 0.1 * normal((STATE,)),
                 "D": 1 + 0.1 * normal((heads,)), "out_proj.weight": normal((WIDTH, inner)) / np.sqrt(inner)}
        if VARIANTS[variant]["is_mimo"]:
            mixer.update({name: scale + 0.1 * normal((heads, rank, HEAD_DIM))
                          for name, scale in (("mimo_x", 1 / rank), ("mimo_z", 1.0), ("mimo_o", 1 / rank))})
        if VARIANTS[variant]["is_outproj_norm"]:
            mixer["norm.weight"] = 1 + 0.1 * normal((inner,))
        block = {"norm.weight": 1 + 0.1 * normal((WIDTH,)), "norm2.weight": 1 + 0.1 * normal((WIDTH,)),
                 "mlp.fc1.weight": normal((2 * INTERMEDIATE, WIDTH)) / np.sqrt(WIDTH),
                 "mlp.fc2.weight": normal((WIDTH, INTERMEDIATE)) / np.sqrt(INTERMEDIATE),
                 **{f"mixer.{name}": value for name, value in mixer.items()}}
        tensors.update({f"backbone.layers.{layer}.{name}": value for name, value in block.items()})
    tensors["lm_head.weight"] = tensors["backbone.embedding.weight"]
    return {name: np.asarray(value, np.float32) for name, value in tensors.items()}


def _rms(x, weight, groups: int = 1):
    """mamba_ssm's RMSNorm at eps 1e-5 over `groups` equal parts of the last axis."""
    parts = x.reshape(*x.shape[:-1], groups, -1)
    normed = parts * jax.lax.rsqrt(jnp.mean(parts * parts, axis=-1, keepdims=True) + 1e-5)
    return normed.reshape(x.shape) * weight


def mamba_ssm_logits(tensors: dict, variant: str, ids, dtype):
    """mamba_ssm's `MambaLMHeadModel` forward on the stored tensors in `dtype`, as its modules compute it:
    each block `h + Mamba3(norm(h))`, then `h + GatedMLP(norm2(h))`; the layer's split of `in_proj`, its
    activations (`softplus`, `heavy_tail`, `sigmoid`, `tanh` times pi), its norms of B and C before their
    biases, its ranks' `x` and `z` weights, the recurrence on the complex state (`recurrent_mamba3`), `D x`,
    the gate (or the out-projection norm before it), the ranks' sum."""
    t = {name: jnp.asarray(value, dtype) for name, value in tensors.items()}
    rank, heads, inner = rank_of(variant), 2 * WIDTH // HEAD_DIM, 2 * WIDTH
    mimo, normed = VARIANTS[variant]["is_mimo"], VARIANTS[variant]["is_outproj_norm"]
    h = t["backbone.embedding.weight"][ids]
    batch, length = ids.shape
    for layer in range(LAYERS):
        p = f"backbone.layers.{layer}."
        u = _rms(h, t[p + "norm.weight"]) @ t[p + "mixer.in_proj.weight"].T
        sizes = np.cumsum([inner, inner, STATE * rank, STATE * rank, heads, heads, heads])
        z, x, B, C, dt, A, trap, angles = jnp.split(u, sizes, axis=-1)
        z, x = (v.reshape(batch, length, 1, heads, HEAD_DIM) for v in (z, x))
        dt = jax.nn.softplus(dt + t[p + "mixer.dt_bias"])
        A = jnp.minimum(-(jnp.maximum(A, 0) + 1 / (1 - jnp.minimum(A, 0))), -1e-4)
        keys = []
        for v, name in ((B, "B"), (C, "C")):
            v = _rms(v.reshape(batch, length, rank, 1, STATE), t[p + f"mixer.{name}_norm.weight"])
            keys.append(jnp.repeat(v, heads, axis=3) + jnp.swapaxes(t[p + f"mixer.{name}_bias"], 0, 1))
        if mimo:
            x, z = (v * jnp.swapaxes(t[p + f"mixer.mimo_{name}"], 0, 1) for v, name in ((x, "x"), (z, "z")))
        theta = (jnp.tanh(angles) * jnp.pi)[:, :, None]
        y, _ = recurrent_mamba3(x, dt, A, keys[0], keys[1], jax.nn.sigmoid(trap), theta)
        y = y + t[p + "mixer.D"][:, None] * x
        if normed:
            weight = t[p + "mixer.norm.weight"].reshape(heads, HEAD_DIM)
            y = _rms(y, 1.0, 1) * weight * jax.nn.silu(z)
        else:
            y = y * jax.nn.silu(z)
        y = jnp.einsum("bsrhp,hrp->bshp", y, t[p + "mixer.mimo_o"]) if mimo else y[:, :, 0]
        h = h + y.reshape(batch, length, inner) @ t[p + "mixer.out_proj.weight"].T
        up, gate = jnp.split(_rms(h, t[p + "norm2.weight"]) @ t[p + "mlp.fc1.weight"].T, 2, axis=-1)
        h = h + (up * jax.nn.silu(gate)) @ t[p + "mlp.fc2.weight"].T
    return _rms(h, t["backbone.norm_f.weight"]) @ t["lm_head.weight"].T


def tokens(seed: int):
    """Two rows of 40 token ids of the unpadded vocabulary."""
    return (np.random.default_rng([seed, 2006]).integers(0, VOCAB, (2, 40)).astype(np.int32),)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_a_mamba_ssm_mamba3_checkpoint_loads_as_mamba_ssm_computes_it(tmp_path, variant):
    """mamba_ssm's config and names read through its one reader (`mamba2.config_from_mamba_ssm`, its layer
    Mamba3) into the mamba3 family: the load's logits on par with mamba_ssm's model of the checkpoint
    (`mamba_ssm_logits`), its truth that model in float64 (check 1), SISO as the 187m release is and MIMO
    of rank two under the out-projection norm."""
    tensors = checkpoint_tensors(variant)
    (tmp_path / "config.json").write_text(json.dumps(checkpoint_config(variant)))
    save_file({name: value for name, value in tensors.items() if name != "lm_head.weight"},
              tmp_path / "model.safetensors")
    loaded = Pretrained.load(tmp_path, dtype="float32", attention_impl="reference")

    @as_given
    def reference(ids):
        return np.asarray(mamba_ssm_logits(tensors, variant, jnp.asarray(ids), jnp.float32))

    @as_given
    def truth(ids):
        with jax.enable_x64(new_val=True):
            return np.asarray(mamba_ssm_logits(tensors, variant, jnp.asarray(ids), jnp.float64))

    assert_as_exact(jax.jit(lambda ids: loaded.model.apply(loaded.variables, ids)), reference, tokens,
                    f"mamba3 interop {variant}", seeds=SEEDS, truth_fn=truth)
