"""mHC's streams over a range of layers (`HyperConnections.hc_layers`), composed with what reads them.

A range is entered at its first layer as the stack enters its streams (`enter_streams`) and left at its
last by the head the stack's end would collapse them by (`leave_streams`), or, where the stack ends in
the range and no loop runs it again, by the stack's own collapse (`HyperConnections.ends_stack`). So a
range spanning the stack computes what streams over the whole stack do, whatever reads them: engram
layers, DSpark's targets, V4's prediction depths, a learned head and Single-Pass mHC. Each released
fixture's whole-stack decoder is the reference arm, check 1 against it in float64.
"""

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import as_given, assert_as_exact

from dew.interop import Pretrained
from dew.nn.backbones.decoder_stack import Loop

FIXTURES = Path(__file__).parent / "fixtures" / "hf"

SPANNING = ("deepseek-v4-tiny", "deepseek-v4-dspark-tiny", "deepseek-v41-tiny", "qwen4-exp-tiny",
            "glm5-next-tiny")
"""The released fixtures streaming their whole stack: V4's weighted head and prediction depths, V4's
DSpark, V4.1's Single-Pass schedule, carried head, engram layers and DSpark, Qwen4-Exp's gated residual
and engram layers, and GLM-5.3-Flash's mean."""

SEEDS = {**dict.fromkeys(SPANNING, 8), **{f"{name} left by its last layer": 8 for name in SPANNING}}
"""Check 1's declared seeds for each site of the file, sized by `seeds_for` on armada."""


def _loaded(name: str) -> Pretrained:
    return Pretrained.load(FIXTURES / name, dtype="float32", attention_impl="reference")


def _spanning(model):
    """`model` with its streams over the range of layers spanning its stack, [0, num_layers)."""
    hc = model.hyper_connections
    return model.clone(hyper_connections=dataclasses.replace(hc, hc_layers=(0, model.num_layers)))


def _drawn(loaded: Pretrained):
    """Check 1's draw: the fixture's weights and two rows of sixteen drawn ids."""
    vocab = loaded.model.vocab_size
    return lambda seed: (loaded.variables,
                         np.random.default_rng(seed).integers(0, vocab, (2, 16)).astype(np.int32))


def _read(model):
    """What a decoder streaming its layers hands its readers, jitted: its logits, V4's prediction depths'
    logits where it has depths (from the streams its trunk ends with), and DSpark's context where it
    drafts (its targets' streams)."""
    def run(variables, ids):
        prediction, logits = model.apply(variables, ids, method="states_and_logits")
        read = {"logits": logits}
        if model.mtp_hyper_connections is not None:
            read["depths"] = model.apply(variables, prediction, ids, method="mtp_logits")
        if model.dspark is not None:
            _, sown = model.apply(variables, ids, mutable=["prediction_inputs"])
            read["draft context"] = model.apply(variables, sown["prediction_inputs"], method="draft_context")
        return read
    return jax.jit(run)


@pytest.mark.parametrize("name", SPANNING)
def test_streams_over_a_range_spanning_the_stack_compute_the_whole_stacks(name):
    """A released decoder's streams over the range of layers spanning its stack, entered by its first
    layer and left by the stack's own collapse, are on par with its streams over the whole stack (check
    1, against them in float64): its logits, its engram layers', learned head's and Single-Pass
    schedule's included, V4's prediction depths reading the streams the trunk ends with, and DSpark's
    context, its targets' streams."""
    loaded = _loaded(name)
    whole = loaded.model
    assert_as_exact(_read(_spanning(whole)), _read(whole), _drawn(loaded), f"{name} over a spanning range",
                    seeds=SEEDS[name], truth_fn=_read(whole.clone(dtype=jnp.float64)))


@pytest.mark.parametrize("name", [name for name in SPANNING if name != "deepseek-v4-tiny"])
def test_a_range_its_last_layer_leaves_collapses_as_the_stacks_head_does(name):
    """Under a loop, which runs the range again, the range's last layer leaves the streams itself, by the
    head the stack's end would collapse them by, which it holds (`hc_head`): V4's learned head,
    Qwen4-Exp's gated read, V4.1's carried pre or the mean. A loop of one pass computes the stack
    once, so the decoder is on par with its streams over the whole stack, the stack's head's weights
    moved to its last layer (check 1, against the whole stack in float64). DSpark, which reads beside
    a loop's residual, is left out of both, and V4's depths, which read the streams the trunk ends with,
    are refused under a loop."""
    loaded = _loaded(name)
    whole = loaded.model.clone(dspark=None)
    last = f"layers_{whole.num_layers - 1}"
    looped = _spanning(whole).clone(loop=Loop(1, step_norm=False))

    def moved(variables):
        params = dict(variables["params"])
        if "hc_head" in params:
            params[last] = {**params[last], "hc_head": params.pop("hc_head")}
        return {**variables, "params": params}

    def logits(model, held=lambda variables: variables):
        return jax.jit(lambda variables, ids: model.apply(held(variables), ids))

    assert_as_exact(logits(looped, moved), logits(whole), _drawn(loaded), f"{name} left by its last layer",
                    seeds=SEEDS[f"{name} left by its last layer"],
                    truth_fn=logits(whole.clone(dtype=jnp.float64)))


def test_a_dspark_target_outside_the_range_reads_its_residual():
    """A DSpark target layer before streams over a later range of layers reads its plain residual, the
    mean of its one copy, as a target inside reads its streams' mean: V4.1's context at its first
    target, layer 9 of 12 under streams over [10, 12), is that layer's input, layer 8's output, held
    without arithmetic. Its engram layers, 1 and 4, gate into streams, which layers outside the range
    carry none of, so they are left out."""
    loaded = _loaded("deepseek-v41-tiny")
    whole = loaded.model.clone(engram=None)
    target = whole.dspark.target_layers[0]
    ranged = whole.clone(hyper_connections=dataclasses.replace(whole.hyper_connections,
                                                              hc_layers=(target + 1, whole.num_layers)))
    _, ids = _drawn(loaded)(0)
    _, sown = ranged.apply(loaded.variables, ids, mutable=["prediction_inputs", *ranged.layer_collections],
                           capture_intermediates=ranged.capture_layers)
    context = ranged.apply(loaded.variables, sown["prediction_inputs"], method="draft_context")
    np.testing.assert_array_equal(context[..., :whole.emb_features], ranged.layer_state(sown, target - 1))


PIPELINED = {"vocab_size": 64, "emb_features": 32, "num_layers": 4, "num_heads": 4, "num_kv_heads": 2,
             "mlp_features": 64, "max_seq_len": 8}
"""A four-layer decoder two stages of a pipeline over the stage axis split into a two-layer prefix and two
layers they repeat (`split_stages`)."""


def _single_pass(layers: tuple[int, int]):
    from dew.nn.hyper_connections import HyperConnections
    from dew.registry import models

    spec = HyperConnections(hc_mult=2, single_pass=True, head="carried", hc_layers=layers)
    return models.build("causal_transformer", **PIPELINED, hyper_connections=spec)


def _forward_on(model, variables, stage: int):
    """`model`'s logits over token rows on a mesh of `stage` stages and as many row shards as make eight
    devices, the pipeline's schedule of four microbatches in context."""
    from dew.distributed import MeshSpec
    from dew.distributed.batches import shard_batch
    from dew.nn.sharding import pipeline_microbatches

    mesh = MeshSpec(fsdp=8 // stage, stage=stage).build()
    forward = jax.jit(model.apply)

    @as_given
    def run(ids):
        rows = shard_batch(mesh, {"ids": ids})["ids"]
        with jax.set_mesh(mesh), pipeline_microbatches(4 if stage > 1 else None):
            return np.asarray(forward(variables, rows))
    return run


PIPELINED_SEEDS = 8
"""Check 1's declared seeds for Single-Pass streams beside a pipeline, sized by `seeds_for` on armada."""


@pytest.mark.mesh
def test_single_pass_streams_the_pipelines_prefix_holds_run_beside_its_stages():
    """Single-Pass streams over layers [0, 1) of four, which a two-stage pipeline's two-layer prefix holds,
    leave the residual plain before the stages: the pipeline runs, its logits on par with the same
    decoder's off it (check 1, against it in float64). Streams over [0, 4), whose layers the stages
    repeat, carry each sublayer's collapse beside them, which the pipeline's microbatch buffers do not
    hold: refused."""
    model = _single_pass((0, 1))
    variables = jax.tree.map(np.asarray, model.init(jax.random.key(0), jnp.ones((1, 8), jnp.int32)))

    def draw(seed):
        return (np.random.default_rng(seed).integers(0, 64, (16, 8)).astype(np.int32),)

    @as_given
    def float64(ids):
        wide = jax.tree.map(lambda leaf: np.asarray(leaf, np.float64), variables)
        return np.asarray(model.clone(dtype=jnp.float64).apply(wide, ids))

    assert_as_exact(_forward_on(model, variables, 2), _forward_on(model, variables, 1), draw,
                    "Single-Pass streams in the pipeline's prefix", seeds=PIPELINED_SEEDS, truth_fn=float64)
    whole = _single_pass((0, 4))
    with pytest.raises(ValueError):
        _forward_on(whole, variables, 2)(draw(0)[0])
