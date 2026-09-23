"""Same-weight parity with lm-engine 45b6b57b, the codebase Rigel was trained
with: a Rigel-shaped hybrid's forward, losses and gradients, AdamW steps over
its muP parameter groups, and its learning-rate schedulers.

The fixtures are float64 torch runs stored as float32
(tools/lm_engine_reference.py). Dew runs here in float64, but the router's gate, the Mamba-2 scan internals and the
head contract in float32 by design (`Router.logits`, `Mamba2`, `_logits`), so
agreement is held to float32 resolution, a few ulps of 1.2e-7 relative to each
tensor's largest entry, and not to float64's. Each feature under test moves
the logits by 0.8% to 300% when it is switched off, so these tolerances leave
no room for a missing one.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.config import OptimConfig
from dew.nn.backbones.causal_transformer import CausalTransformer, LayerKind, Mixture
from dew.nn.mixers import AttentionMixer
from dew.nn.mixers.mamba2 import Mamba2Mixer
from dew.nn.moe import global_router_loss, router_moments
from dew.objectives.lm.objective import _router_scores, router_z_terms
from dew.training.optim import (
    build_optimizer,
    linear_schedule,
    mup_param_groups,
    param_labels,
    power_schedule,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "lm_engine"
COEF = 0.01  # the fixture's router_aux_loss_coef


@pytest.fixture(scope="module")
def reference():
    return dict(np.load(FIXTURES / "hybrid.npz"))


def hybrid(dtype=jnp.float64) -> CausalTransformer:
    """The fixture's model in Dew's fields."""
    return CausalTransformer(
        vocab_size=64, emb_features=32, num_layers=4, num_heads=4, num_kv_heads=2, head_dim=8,
        qk_norm=False, max_seq_len=64,
        layer_types=("mamba", "mamba", "mamba", "attention"),
        kinds={"mamba": LayerKind(mixer=Mamba2Mixer(
            num_heads=4, head_dim=8, state_size=8, n_groups=1, chunk_size=8)),
               "attention": LayerKind(mixer=AttentionMixer(nope=True, exclusive_self_attention=True))},
        mixture=Mixture(experts=4, top_k=2, expert_features=8),
        embedding_multiplier=12.0, residual_multiplier=0.22, logits_scaling=4.0,
        dtype=dtype, attention_impl="xla")


def leaves(reference, prefix: str) -> list[tuple[str, tuple[str, ...], np.ndarray]]:
    """Every Dew leaf as (lm-engine source name, Dew path, value), read from
    the fixture's `prefix` entries. c_attn holds each KV head's queries, key
    and value in turn; c_fc interleaves each expert's gate (even rows) and up
    (odd rows) projections (mlp_blocks/mlp/utils.py)."""
    def get(name):
        return np.asarray(reference[prefix + name], np.float64)

    out = [("transformer.wte.weight", ("embed_tokens", "embedding"), get("transformer.wte.weight")),
           ("transformer.ln_f.weight", ("norm", "scale"), get("transformer.ln_f.weight"))]
    for index in range(4):
        h, layer = f"transformer.h.{index}.", f"layers_{index}"
        out += [(h + "ln_1.weight", (layer, "input_layernorm", "scale"), get(h + "ln_1.weight")),
                (h + "ln_2.weight", (layer, "post_attention_layernorm", "scale"), get(h + "ln_2.weight"))]
        fc = get(h + "mlp_block.c_fc.weight")
        out += [(h + "mlp_block.gate.weight", (layer, "mlp", "gate", "kernel"),
                 get(h + "mlp_block.gate.weight").T),
                (h + "mlp_block.c_fc.weight", (layer, "mlp", "experts", "gate_proj", "kernel"),
                 fc[:, ::2].transpose(0, 2, 1)),
                (h + "mlp_block.c_fc.weight", (layer, "mlp", "experts", "up_proj", "kernel"),
                 fc[:, 1::2].transpose(0, 2, 1)),
                (h + "mlp_block.c_proj.weight", (layer, "mlp", "experts", "down_proj", "kernel"),
                 get(h + "mlp_block.c_proj.weight").transpose(0, 2, 1))]
        s = h + "sequence_mixer."
        mixer = (layer, "self_attn")
        if index < 3:
            out += [(s + "in_proj.weight", (*mixer, "in_proj", "kernel"), get(s + "in_proj.weight").T),
                    (s + "conv1d.weight", (*mixer, "conv1d", "weight"), get(s + "conv1d.weight")),
                    (s + "conv1d.bias", (*mixer, "conv1d", "bias"), get(s + "conv1d.bias")),
                    (s + "decay_gate.A_log", (*mixer, "A_log"), get(s + "decay_gate.A_log")),
                    (s + "decay_gate.dt_bias", (*mixer, "dt_bias"), get(s + "decay_gate.dt_bias")),
                    (s + "D", (*mixer, "D"), get(s + "D")),
                    (s + "norm.weight", (*mixer, "norm", "weight"), get(s + "norm.weight")),
                    (s + "out_proj.weight", (*mixer, "out_proj", "kernel"), get(s + "out_proj.weight").T)]
        else:
            fused = get(s + "c_attn.weight").reshape(2, 32, 32)
            out += [(s + "c_attn.weight", (*mixer, "q_proj", "kernel"), fused[:, :16].reshape(32, 32).T),
                    (s + "c_attn.weight", (*mixer, "k_proj", "kernel"), fused[:, 16:24].reshape(16, 32).T),
                    (s + "c_attn.weight", (*mixer, "v_proj", "kernel"), fused[:, 24:].reshape(16, 32).T),
                    (s + "c_proj.weight", (*mixer, "o_proj", "kernel"), get(s + "c_proj.weight").T)]
    return out


def tree(entries) -> dict:
    out: dict = {}
    for _, path, value in entries:
        node = out
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = jnp.asarray(value)
    return out


def worst(ours, theirs) -> float:
    """The largest leaf error relative to that leaf's largest entry."""
    def relative(a, b):
        b = np.asarray(b)
        return float(np.max(np.abs(np.asarray(a) - b)) / max(np.max(np.abs(b)), 1e-300))
    return max(jax.tree.leaves(jax.tree.map(relative, ours, theirs)))


def objective(model, tokens):
    """lm-engine's training loss: the mean next-token cross entropy plus the
    coefficient times the summed Switch loss and 0.1 of the router z-loss."""
    def loss(params):
        logits, sown = model.apply({"params": params}, tokens, mutable=["router"])
        logp = jax.nn.log_softmax(logits[:, :-1], axis=-1)
        lm = -jnp.mean(jnp.take_along_axis(logp, tokens[:, 1:, None], axis=-1))
        routing = sown["router"]
        switch = sum(global_router_loss(router_moments(s, i), 1.0) for s, i in _router_scores(routing))
        z = sum(term.total / term.mass for term in router_z_terms(routing, 1.0))
        aux = switch + 0.1 * z
        return lm + COEF * aux, (logits, lm, aux)
    return loss


def test_rigel_shaped_hybrid_matches_lm_engine_forward_losses_and_gradients(reference):
    with jax.enable_x64(True):
        model = hybrid()
        tokens = jnp.asarray(reference["tokens"], jnp.int32)
        params = tree(leaves(reference, "param:"))
        assert (jax.tree.map(jnp.shape, model.init(jax.random.key(0), tokens)["params"])
                == jax.tree.map(jnp.shape, params))
        (total, (logits, lm, aux)), grads = jax.value_and_grad(
            objective(model, tokens), has_aux=True)(params)
        assert worst(logits, reference["logits"]) < 1e-6
        np.testing.assert_allclose(float(lm), reference["lm_loss"], rtol=1e-6)
        np.testing.assert_allclose(float(aux), reference["aux_loss"], rtol=1e-6)
        np.testing.assert_allclose(float(total), reference["total"], rtol=1e-6)
        assert worst(grads, tree(leaves(reference, "grad:"))) < 1e-5


def test_mup_groups_and_adamw_steps_match_lm_engine(reference):
    """lm-engine's mup.yml groups, torch AdamW and its PowerScheduler, fed the
    same gradients: the group of every parameter and the parameters after
    each of three steps agree."""
    membership = json.loads(str(reference["groups"]))
    groups = mup_param_groups(4.0)
    params0 = leaves(reference, "param:")
    labels = param_labels(groups)(tree(params0))
    for source, path, _ in params0:
        node = labels
        for key in path:
            node = node[key]
        assert node == membership[source], (path, node, membership[source])

    config = OptimConfig(
        optimizer="adamw", optimizer_opts={"b1": 0.9, "b2": 0.95, "eps": 1e-10},
        learning_rate_schedule="power", learning_rate_peak=0.01, learning_rate_warmup_steps=2,
        power_a=0.05, power_b=-0.51, power_c=16.0, weight_decay=0.1, param_groups=groups)
    with jax.enable_x64(True):
        solver = build_optimizer(config, steps=3)
        params = tree(params0)
        state = solver.init(params)
        for step in range(3):
            grads = tree(leaves(reference, f"step{step}/grad:" if step else "grad:"))
            updates, state = solver.update(grads, state, params)
            params = jax.tree.map(lambda p, u: p + u, params, updates)
            # The rates are float32 schedule values; the step moves each entry
            # by at most the rate, so the parameters agree to a float32 ulp
            # of the rate on the largest entry.
            assert worst(params, tree(leaves(reference, f"step{step}/param:"))) < 1e-6, step


def test_power_and_linear_schedules_match_lm_engine():
    runs = np.load(FIXTURES / "schedule.npz")
    power = json.loads(str(runs["power_args"]))
    schedule = power_schedule(power["lr"], power["num_warmup_steps"], power["a"], power["b"], power["c"])
    steps = np.arange(len(runs["power"]))
    np.testing.assert_allclose(np.asarray(jax.vmap(schedule)(steps)), runs["power"], rtol=2e-6, atol=0)
    linear = json.loads(str(runs["linear_args"]))
    warmup, constant = linear["num_warmup_steps"], linear["num_constant_steps"]
    schedule = linear_schedule(linear["lr"], warmup, warmup + constant,
                               warmup + constant + linear["num_decay_steps"])
    steps = np.arange(len(runs["linear"]))
    # optax evaluates in float32: near zero the tail's error is a float32 ulp
    # of the peak it interpolates from, not of the value.
    np.testing.assert_allclose(np.asarray(jax.vmap(schedule)(steps)), runs["linear"],
                               rtol=2e-6, atol=2e-6 * linear["lr"])
