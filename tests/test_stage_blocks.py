"""A stack of like blocks over the stage axis: a DiT's, a hybrid DiT's, a JEPA encoder's (an image's
and a clip's factorized one) and a UNet level's transformers each run as a pipeline over the mesh's
stages (`dew.nn.backbones.decoder_stack.pipelined_blocks`), and compute through it what one device
computes: each model's output and the gradients of its parameters and inputs under a drawn cotangent
are on par with one device's, both measured from float64 (check 1).

Each model is the smallest that still splits into two stages of like blocks: a DiT of four blocks,
a hybrid of four alternating SSM and attention blocks (two stages of one of each), an encoder of
four, a clip's encoder of two spatial and temporal pairs, and a UNet of one level over 8 x 8
latents, whose attentions down, through the middle and up each hold two transformers. The layouts
put the stages beside fsdp and the rows over data (eight devices, two rows a device, one microbatch
a stage), or beside a sequence axis (four rows a device over data, four microbatches), the DiT and
the UNet on both."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P
from reference_error import as_given, assert_as_exact, widened, without_ynn

from dew.diffusion.process import DenoisingCondition
from dew.mesh import batch_axes
from dew.nn.conv import Conv
from dew.nn.sharding import pipeline_microbatches
from dew.registry import models
from dew.training import Layout, MeshSpec

pytestmark = pytest.mark.mesh

BATCH = 8
RES = 16
TINY_SHARD = 256
LATENT_RES = 8
"""The UNet's latents' side: its one level attends over all 64 positions, which a sequence axis of two
splits."""
CONTEXT, CONTEXT_FEATURES = 4, 16

LAYOUTS = {
    "fsdp2_stage2": (MeshSpec(fsdp=2, stage=2), None),
    "stage2_sequence2": (MeshSpec(stage=2, sequence=2), 4),
}
"""Each layout's mesh over the eight devices and the microbatches its step feeds the stages."""


def diffusion_call(model, variables, images, times):
    return model.apply(variables, images, times)


def unet_call(model, variables, latents, times, context):
    return model.apply(variables, latents, times, conditioning=DenoisingCondition(context=context))


def encoder_call(model, variables, images):
    return model.apply(variables, images)


VIT = {"patch_size": 4, "emb_features": 32, "num_heads": 4, "mlp_ratio": 2}

CASES = {
    "dit": ("simple_dit", {**VIT, "num_layers": 4}, "images"),
    "hybrid_dit": ("hybrid_dit", {**VIT, "num_layers": 4, "ssm_state_dim": 8, "ssm_attention_ratio": "1:1"},
                   "images"),
    "jepa": ("jepa_encoder", {**VIT, "num_layers": 4}, "encoded"),
    "jepa_video": ("jepa_video_encoder", {**VIT, "num_layers": 2}, "clips"),
    "unet": ("unet_2d_condition", {"stages": [{"features": 32, "heads": 4, "depth": 2}],
                                   "blocks_per_level": 1, "in_channels": 4, "out_channels": 4}, "latents"),
}
"""Each model by its registry name and fields, and the inputs it reads."""

CELLS = [("dit", "fsdp2_stage2"), ("dit", "stage2_sequence2"), ("hybrid_dit", "fsdp2_stage2"),
         ("jepa", "stage2_sequence2"), ("jepa_video", "fsdp2_stage2"), ("unet", "fsdp2_stage2"),
         ("unet", "stage2_sequence2")]

SEEDS = {"dit": 32, "hybrid_dit": 32, "jepa": 32, "jepa_video": 32, "unet": 128, "patch_conv": 8}
"""Each model's check-1 seeds, the most any of its layouts took, sized paired by `seeds_for` (`dew_fn`, the
layout's arm) on armada, at 9efb77a3e and the UNet's with the confirming block at 8ce8072b5 (64 over
fsdp2_stage2, 128 beside the sequence axis): the arms share their inputs and variables."""


def drawn(kind: str):
    """A draw's inputs and the cotangent of the model's output, for a model reading `kind`."""
    def draw(seed):
        rng = np.random.default_rng(seed)
        if kind == "images":
            inputs = (rng.normal(size=(BATCH, RES, RES, 3)).astype(np.float32),
                      rng.uniform(size=(BATCH,)).astype(np.float32))
            out = (BATCH, RES, RES, 3)
        elif kind == "latents":
            inputs = (rng.normal(size=(BATCH, LATENT_RES, LATENT_RES, 4)).astype(np.float32),
                      rng.uniform(0, 1000, size=(BATCH,)).astype(np.float32),
                      rng.normal(size=(BATCH, CONTEXT, CONTEXT_FEATURES)).astype(np.float32))
            out = (BATCH, LATENT_RES, LATENT_RES, 4)
        elif kind == "clips":
            inputs = (rng.normal(size=(BATCH, 2, RES, RES, 3)).astype(np.float32),)
            out = (BATCH, 2, (RES // 4) ** 2, 32)
        else:
            inputs = (rng.normal(size=(BATCH, RES, RES, 3)).astype(np.float32),)
            out = (BATCH, (RES // 4) ** 2, 32)
        return inputs, rng.normal(size=out).astype(np.float32)
    return draw


def call_of(kind: str):
    return {"images": diffusion_call, "latents": unet_call}.get(kind, encoder_call)


def initialized(model, kind: str, inputs):
    """The model's variables, drawn on one device from the inputs' shapes, each leaf its initializer
    leaves all zeros drawn from a normal of std 0.02 instead: a DiT zero-initializes its output
    projection and its modulations, and its gradient would reach no block below them."""
    if kind == "latents":
        latents, times, context = inputs
        variables = jax.jit(lambda key: model.init(key, latents, times, conditioning=DenoisingCondition(
            context=context)))(jax.random.key(0))
    else:
        variables = jax.jit(lambda key: model.init(key, *inputs))(jax.random.key(0))
    rng = np.random.default_rng(1)
    return jax.tree.map(lambda leaf: rng.normal(0, 0.02, leaf.shape).astype(leaf.dtype)
                        if jnp.issubdtype(leaf.dtype, jnp.floating) and not np.any(leaf) else leaf, variables)


def pulled(model, call):
    """The model's output on its inputs and the gradients of its variables and inputs under the
    cotangent, one function of the variables, the inputs and the cotangent."""
    def run(variables, inputs, cotangent):
        out, pull = jax.vjp(lambda held, given: call(model, held, *given), variables, inputs)
        return out, *pull(cotangent)
    return run


def on_mesh(spec: MeshSpec, devices: int, model, variables, call, microbatches=None):
    """`pulled` on `spec`'s mesh over the first `devices` devices, the variables placed by the layout
    and the rows of the inputs and the cotangent over the batch's axes, with the pipeline's
    microbatches in context as a trainer's step puts them there, compiled once (`without_ynn`)."""
    mesh = spec.build(jax.devices()[:devices])
    placed = jax.device_put(variables, Layout(min_shard=TINY_SHARD).shardings(mesh, variables))
    rows = NamedSharding(mesh, P(batch_axes(mesh)))
    compiled = without_ynn(pulled(model, call))

    @as_given
    def run(inputs, cotangent):
        with jax.set_mesh(mesh), pipeline_microbatches(microbatches):
            return jax.device_get(compiled(placed, *jax.device_put((inputs, cotangent), rows)))
    return run


def float64_pulled(model, variables, call):
    """`pulled` of the model computing in float64 over its variables widened: the truth, one program the
    gate compiles for every draw. Returned to the host, the gate would run it as a NumPy truth, op by
    op, which took a UNet cell's draw 1.2 s against the arms' 0.16 s."""
    wide = model.clone(dtype=jnp.float64)
    with jax.enable_x64(new_val=True):
        held = jax.tree.map(jnp.asarray, widened(variables))
    run = pulled(wide, call)
    return lambda inputs, cotangent: run(held, inputs, cotangent)


def runs_a_pipeline(spec: MeshSpec, model, variables, call, inputs, microbatches=None) -> bool:
    """Whether the model's forward on `spec`'s mesh runs a pipeline over the stage axis
    (`pipeline_microbatches`' schedule), traced abstractly."""
    mesh = spec.build()
    with jax.set_mesh(mesh), pipeline_microbatches(microbatches) as schedule:
        jax.eval_shape(lambda held, given: call(model, held, *given), variables, inputs)
    return schedule.pipelined


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize(("case", "layout"), [
    pytest.param(case, layout, id=f"{case}-{layout}",
                 marks=pytest.mark.whole_suite if (case, layout) == ("unet", "fsdp2_stage2") else ())
    for case, layout in CELLS])
def test_a_stack_of_like_blocks_over_stages_computes_what_one_device_computes(case, layout):
    """The model runs its blocks as a pipeline over the stage axis, and its output and gradients are
    on par with one device's, both measured from float64 (check 1). Before, every model but a
    decoder ran its whole stack on every stage, which the trainer refused."""
    name, fields, kind = CASES[case]
    spec, microbatches = LAYOUTS[layout]
    model = models.build(name, **fields)
    draw, call = drawn(kind), call_of(kind)
    inputs, _ = draw(0)
    variables = initialized(model, kind, inputs)
    assert runs_a_pipeline(spec, model, variables, call, inputs, microbatches)
    assert_as_exact(on_mesh(spec, 8, model, variables, call, microbatches),
                    on_mesh(MeshSpec(), 1, model, variables, call), draw, f"{case} over {layout}",
                    seeds=SEEDS[case], truth_fn=float64_pulled(model, variables, call))


@pytest.mark.mesh(devices=8)
def test_a_convolution_beside_a_pipeline_computes_its_kernel_gradient_as_one_device_does():
    """A convolution outside a pipeline runs on every stage. A patch embedding over stage 2 beside
    sequence 2, its output's rows split over sequence and the stage axis left replicating it, had XLA's
    partitioner double its kernel gradient (`dew.nn.conv`'s first bug; the JEPA encoder's above);
    split over the stage axis too, its output and gradients are on par with one device's (check 1)."""
    model = Conv(32, (4, 4), strides=(4, 4))
    variables = initialized(model, "encoded", drawn("encoded")(0)[0])

    def draw(seed):
        rng = np.random.default_rng(seed)
        return ((rng.normal(size=(BATCH, RES, RES, 3)).astype(np.float32),),
                rng.normal(size=(BATCH, RES // 4, RES // 4, 32)).astype(np.float32))

    spec, microbatches = LAYOUTS["stage2_sequence2"]
    assert_as_exact(on_mesh(spec, 8, model, variables, encoder_call, microbatches),
                    on_mesh(MeshSpec(), 1, model, variables, encoder_call), draw,
                    "patch conv over stage2_sequence2", seeds=SEEDS["patch_conv"],
                    truth_fn=float64_pulled(model, variables, encoder_call))
