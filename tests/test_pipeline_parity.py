"""A pipelined decoder's full training state stays on par with one device's (check 1).

Each draw initializes its own four-layer decoder with 32-wide heads and takes two AdamW steps
on 64 rows of 64 tokens. The larger heads and batches give the q/k norm scale gradients enough
elements and terms for paired sizing. The two-stage layout places the rows over eight CPU devices
beside fsdp; its parameters, EMA and optimizer moments are measured against the same initial state
widened to float64.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import TINY_DECODER
from reference_error import as_given, assert_as_exact, widened
from token_support import token_rows

from dew.distributed.batches import shard_batch
from dew.objectives.lm import LMObjective
from dew.optim import OptimConfig
from dew.registry import models
from dew.training import Layout, MeshSpec, Trainer

pytestmark = pytest.mark.mesh

VOCAB, BATCH, SEQUENCE, STEPS = 64, 64, 64, 2
HEAD_DIM = 32
STAGED = MeshSpec(fsdp=2, stage=2)
SEEDS = 512
"""Sized paired by `seeds_for`, pipeline against one device, on armada at 4a21a396e9
(TR-3369dbg-19). Both disjoint pilot blocks resolve at 512, and the standard verdict passes at
its first stage over 512 fresh draws, with no product change or layout-specific tolerance."""
LABEL = "two-stage decoder AdamW full state"


def decoder():
    """Four layers, which two pipeline stages split into two layers each."""
    return models.build("causal_transformer", **{
        "vocab_size": VOCAB, **TINY_DECODER, "num_layers": 4, "num_heads": 4,
        "num_kv_heads": 2, "emb_features": 4 * HEAD_DIM, "mlp_features": 8 * HEAD_DIM,
        "max_seq_len": SEQUENCE})


def draw(seed):
    """A fresh batch and the independent key that draws its decoder's initial variables."""
    return token_rows(VOCAB, BATCH, SEQUENCE + 1, seed=seed)["text"], np.int64(seed)


def optimizer():
    """Adam's epsilon keeps its update smooth above a gradient's FP32 rounding."""
    return OptimConfig(optimizer="adamw", learning_rate=1e-2, weight_decay=0.1,
                       optimizer_opts={"eps": 1e-4})


@functools.cache
def initializer():
    return jax.jit(decoder().init)


@functools.lru_cache(maxsize=64)
def variables_at(seed):
    """Every layout starts from these same FP32 variables, drawn once per seed."""
    with jax.enable_x64(new_val=False):
        return initializer()(jax.random.key(seed), jnp.zeros((1, SEQUENCE), jnp.int32))


def trainer(model, spec, devices):
    run = Trainer(LMObjective(model, SEQUENCE, ema_decay=0.9), optimizer(), key=jax.random.key(0),
                  mesh=spec, layout=Layout(min_shard=256), tracker=None)
    run.device_mesh = spec.build(jax.devices()[:devices])
    run.validate(steps=STEPS)
    return run


@functools.cache
def initial_template():
    """The same optimizer state in all arms, including its already-rounded FP32 hyperparameters."""
    with jax.enable_x64(new_val=False):
        return trainer(decoder(), MeshSpec(), 1).initial_state()


def widened_state(state):
    """A truth's copy of the exact initial state, its typed PRNG key kept as a key."""
    return jax.tree.map(lambda leaf: jnp.copy(leaf) if jax.dtypes.issubdtype(
        leaf.dtype, jax.dtypes.prng_key) else widened(leaf), state)


def trained(spec, devices, *, exact=False):
    """Two compiled steps over a layout, returning every parameter, EMA and optimizer-state leaf."""
    built = []

    @as_given
    def run(tokens, seed):
        if not built:
            model = decoder().clone(dtype=jnp.float64) if exact else decoder()
            built.append(trainer(model, spec, devices))
        steps = built[0]
        template, variables = initial_template(), variables_at(int(seed))
        assert template.ema is not None
        moving = {name: jax.tree.map(jnp.copy, variables[name]) for name in template.ema}
        start = dataclasses.replace(template, variables=variables, ema=moving)
        start = widened_state(start) if exact else jax.tree.map(jnp.copy, start)
        state = jax.device_put(start, steps.shardings(start))
        batch = shard_batch(steps.device_mesh, {"text": tokens})
        if len(built) == 1:
            built.append(steps.compile(state, batch))
        for _ in range(STEPS):
            state, *_ = built[1](state, batch)
        return jax.device_get({"variables": state.variables, "opt_state": state.opt_state,
                               "ema": state.ema})
    return run


def arms():
    """The pipeline, one device and float64, each reading the same per-draw initial state."""
    return trained(STAGED, 8), trained(MeshSpec(), 1), trained(MeshSpec(), 1, exact=True)


@pytest.mark.mesh(devices=8)
def test_a_pipelined_decoder_trains_as_one_device_through_adamw():
    """Two AdamW steps over stages leave the full state on par with one device, from float64.

    The gate runs both arms with actual FP32 product sums (`summed_as_run`), and sizes each row
    and leaf by the standard check 1 alone, with no fixed-draw or layout-specific tolerance.
    """
    staged, one, truth = arms()
    assert_as_exact(staged, one, draw, LABEL, seeds=SEEDS, truth_fn=truth)
