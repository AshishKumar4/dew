"""A decoder's logits and every parameter gradient over stages stay on par with one device's.

Each draw initializes its own four-layer decoder and scores eight rows of sixteen tokens.
The two-stage layout runs beside fsdp over eight CPU devices; both arms are measured against
the same variables widened to float64, with actual FP32 product sums under check 1.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P
from model_support import TINY_DECODER
from reference_error import as_given, assert_as_exact, widened, without_ynn
from token_support import token_rows

from dew.mesh import batch_axes
from dew.nn.sharding import pipeline_microbatches
from dew.registry import models
from dew.training import Layout, MeshSpec

pytestmark = pytest.mark.mesh

VOCAB, BATCH, SEQUENCE, HEAD_DIM = 64, 8, 16, 8
STAGED = MeshSpec(fsdp=2, stage=2)
SEEDS = 8
"""Sized paired by `seeds_for`, pipeline against one device, on armada at a887cfb3d0
(TR-3369dbg-20). Both pilot blocks resolve at eight seeds, and check 1 passes at its first stage."""
LABEL = "decoder logits and parameter gradients over stages"


def decoder():
    """Four layers, which two pipeline stages split into two layers each."""
    return models.build("causal_transformer", **{
        "vocab_size": VOCAB, **TINY_DECODER, "num_layers": 4, "num_kv_heads": 2})


def draw(seed):
    """A fresh batch and the independent key that draws its decoder's initial variables."""
    return token_rows(VOCAB, BATCH, SEQUENCE + 1, seed=seed)["text"], np.int64(seed)


@functools.cache
def initializer():
    return jax.jit(decoder().init)


@functools.lru_cache(maxsize=64)
def variables_at(seed):
    """Every layout starts from these same FP32 variables, drawn once per seed."""
    with jax.enable_x64(new_val=False):
        return initializer()(jax.random.key(seed), jnp.zeros((1, SEQUENCE), jnp.int32))


def logits_and_gradients(model, variables, tokens):
    """The logits and all parameter gradients of the mean next-token cross entropy."""
    def loss(params):
        logits = model.apply({"params": params}, tokens[:, :-1], train=True)
        log_probs = jax.nn.log_softmax(logits)
        value = -jnp.take_along_axis(log_probs, tokens[:, 1:, None], axis=-1).mean()
        return value, logits

    (_, logits), gradients = jax.value_and_grad(loss, has_aux=True)(variables["params"])
    return {"logits": logits, "gradients": gradients}


def evaluated(spec, devices, *, exact=False):
    """A layout's compiled forward and backward, placed as the stage-block parity checks place them."""
    model = decoder().clone(dtype=jnp.float64) if exact else decoder()
    mesh = spec.build(jax.devices()[:devices])
    rows = NamedSharding(mesh, P(batch_axes(mesh)))
    compiled = without_ynn(functools.partial(logits_and_gradients, model))

    @as_given
    def run(tokens, seed):
        variables = variables_at(int(seed))
        if exact:
            variables = jax.tree.map(jnp.asarray, widened(variables))
        placement = Layout(min_shard=256).shardings(mesh, variables)
        placed = jax.device_put(variables, placement)
        with jax.set_mesh(mesh), pipeline_microbatches(None):
            result = compiled(placed, jax.device_put(tokens, rows))
        return jax.device_get(result)
    return run


def arms():
    """The pipeline, one device and float64, each reading the same per-draw initial variables."""
    return evaluated(STAGED, 8), evaluated(MeshSpec(), 1), evaluated(MeshSpec(), 1, exact=True)


@pytest.mark.mesh(devices=8)
def test_a_decoder_over_stages_computes_logits_and_gradients_on_par_with_one_device():
    """Both layouts' logits and every parameter gradient stay on par from float64, by check 1.

    The gate runs both arms with actual FP32 product sums (`summed_as_run`), with no optimizer
    steps, fixed-draw guard or layout-specific tolerance.
    """
    staged, one, truth = arms()
    assert_as_exact(staged, one, draw, LABEL, seeds=SEEDS, truth_fn=truth)
