"""Invariants of the model architectures that a training run does not check.

Every registered architecture trains through `fit` in test_architectures.py;
what is here is the behaviour a forward pass has to have beyond running: the
scan orders, the position signal, the mixing across frames, the inflation of
a 2D UNet into a 3D one, and the stage values the UNets are configured with.
"""

import contextlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from flax.core import freeze, unfreeze
from model_support import moved
from reference_error import as_given, assert_as_exact, torch_with_gradients, widened, with_gradients

from dew.interop.autoencoders.vae import translate_vae_weights
from dew.nn.attention_blocks import Stage
from dew.nn.autoencoders.vae import FlaxEncoder
from dew.nn.backbones.dit import SimpleDiT
from dew.nn.backbones.joint import DoubleStreamBlock, JointAttention
from dew.nn.backbones.mmdit import SimpleMMDiT
from dew.nn.backbones.unet import Unet
from dew.nn.conv import Conv
from dew.nn.dit import ModulatedBlock, TextContext
from dew.nn.norms import GroupNorm, LayerNorm
from dew.nn.scan_orders import hilbert_indices, zigzag_indices
from dew.registry import models

RES = 32


def text(batch=2, tokens=77, features=768):
    """A fully real text context, the shape CLIP-L/14 gives."""
    return TextContext(jnp.ones((batch, tokens, features), jnp.float32), jnp.ones((batch, tokens)))


def small_inputs(rng, res=RES, channels=3):
    x = jax.random.normal(rng, (2, res, res, channels))
    temb = jnp.ones((2,))
    return x, temb, text()


VAE_BLOCK_SEEDS = {(32, 32): 8, (32, 64): 8}
"""Each bf16 residual block's declared draws against Diffusers' Flax block (check 1), sized by `seeds_for`
paired with Dew's block on the CPU and an A100."""


@pytest.mark.parametrize("channels", [(32, 32), (32, 64)], ids=["same", "widening"])
def test_a_bf16_vae_block_rounds_where_diffusers_flax_vae_does(channels):
    """In bf16, the VAE's residual block, its two norms ahead of swish, is
    on par with Diffusers' own Flax block on the same weights from the
    float64 block over drawn inputs (check 1): each norm keeps its fp32
    result into swish and the next convolution casts (vae_flax.py:154-209)."""
    from diffusers.models import vae_flax

    from dew.nn.autoencoders.vae import _resnet

    ours = _resnet(channels[1], 8, jnp.float32)
    theirs = vae_flax.FlaxResnetBlock2D(*channels, groups=8)
    shape = (2, 8, 8, channels[0])
    params = theirs.init(jax.random.key(0), jnp.zeros(shape, jnp.float32))
    assert jax.tree_util.tree_structure(ours.init(jax.random.key(0), jnp.zeros(shape, jnp.float32))) == \
        jax.tree_util.tree_structure(params)
    params = jax.tree.map(lambda leaf: leaf + 0.05 * jax.random.normal(jax.random.key(1), leaf.shape),
                          params)

    def draw(seed):
        return (np.random.default_rng(seed).standard_normal(shape).astype(np.float32) * 3,)

    def block(module):
        return lambda x: module.clone(dtype=jnp.bfloat16).apply(params, jnp.asarray(x, jnp.bfloat16))

    assert_as_exact(block(ours), block(theirs), draw, "bf16 VAE residual block",
                    seeds=VAE_BLOCK_SEEDS[channels],
                    truth_fn=lambda x: ours.clone(dtype=jnp.float64).apply(widened(params), x))


ENCODER_SEEDS = 8
"""The VAE encoder's declared draws, the device's run against the host's (check 1), sized by `seeds_for`
paired with the device's run on the CPU."""


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("backward", [False, True], ids=["encode", "vjp"])
def test_a_vae_encoder_keeps_the_host_reference_precision(batch, backward):
    """The full jitted encoder, not an isolated downsampler, and its VJP.

    XLA:TPU's stride-two rewrite moved the batch-one output by 40%. The
    device's run is on par with the host's fp32 run from the host's float64
    run over drawn images and cotangents (check 1).
    """
    model = FlaxEncoder(out_channels=4, block_out_channels=(32, 64), layers_per_block=1,
                        norm_num_groups=8)
    host, device = jax.devices("cpu")[0], jax.devices()[0]
    with jax.default_device(host):
        variables = jax.tree.map(np.asarray, model.init(jax.random.key(0), np.zeros((1, 17, 17, 3),
                                                                                   np.float32)))

    def draw(seed):
        rng = np.random.default_rng(seed)
        return (rng.standard_normal((batch, 17, 17, 3)).astype(np.float32),
                rng.standard_normal((batch, 8, 8, 8)).astype(np.float32))

    def evaluate(dtype, target):
        network = model.clone(dtype=dtype)

        @jax.jit
        def run(params, x, cot):
            if backward:
                return jax.grad(lambda p, a: jnp.sum(network.apply(p, a) * cot),
                                argnums=(0, 1))(params, x)
            return network.apply(params, x)

        def evaluated(image, cotangent):
            arguments = jax.tree.map(lambda x: jax.device_put(np.asarray(x, dtype), target),
                                     (variables, image, cotangent))
            with jax.default_device(target):
                value = jax.tree.map(np.asarray, run(*arguments))
            # The complete parameter and input VJP as one vector, not a cancelling bias leaf scaled by
            # its near-zero norm.
            return np.concatenate([x.reshape(-1) for x in jax.tree.leaves(value)]) if backward else value

        return as_given(evaluated)

    assert_as_exact(evaluate(np.float32, device), evaluate(np.float32, host), draw,
                    "VAE encoder VJP" if backward else "VAE encode", seeds=ENCODER_SEEDS,
                    truth_fn=evaluate(np.float64, host))


VAE_ENCODER = Path(__file__).resolve().parent / "fixtures" / "vae" / "encoder.npz"
VAE_ENCODER_SEEDS = {False: 8, True: 64}
"""Diffusers' encoder site's declared check-1 seeds, the moments' and the gradients', sized by `seeds_for`
paired on armada."""


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("backward", [False, True], ids=["encode", "vjp"])
def test_a_vae_encoder_is_as_exact_as_diffusers(batch, backward):
    """The encoder above against Diffusers' own `Encoder` at its geometry
    (tools/vae_encoder_reference.py's weights), the moments, and the gradients
    of the image and every parameter, on whichever backend runs the test, on par
    with Diffusers' over drawn images and probes (check 1, the truth its float64
    twin's). The host-precision test holds the device to the host;
    this holds both to the published model. On the CPU the ratios are 1.62
    (moments) and 1.60 (gradients) at a batch of one, where XLA convolves
    another way, and 0.99 and 1.01 at four."""
    import ml_dtypes
    import torch
    from diffusers import AutoencoderKL

    from _dew_torch.precision import float64_twin, widened as widened_torch
    from tools.vae_encoder_reference import CONFIG, SIZE, walk

    with np.load(VAE_ENCODER) as loaded:
        arrays = dict(loaded)
    names = [key.removeprefix("param.") for key in arrays if key.startswith("param.")]
    source = {name: arrays[f"param.{name}"].view(ml_dtypes.bfloat16).astype(np.float32) for name in names}
    params = jax.tree.map(jnp.asarray, translate_vae_weights(
        {f"encoder.{name}": value for name, value in source.items()})["encoder"])
    model = FlaxEncoder(out_channels=4, block_out_channels=(32, 64), layers_per_block=1, norm_num_groups=8)
    encoder = AutoencoderKL.from_config(CONFIG).encoder.eval()
    encoder.load_state_dict({name: torch.from_numpy(value) for name, value in source.items()})
    wide = float64_twin(encoder, lambda: AutoencoderKL.from_config(CONFIG).encoder)
    shape = jax.eval_shape(model.apply, {"params": params}, jnp.zeros((batch, SIZE, SIZE, 3))).shape

    def draw(seed):
        rng = np.random.default_rng([67, batch, seed])
        return (rng.standard_normal((batch, 3, SIZE, SIZE)).astype(np.float32),
                rng.standard_normal((shape[0], shape[3], shape[1], shape[2])).astype(np.float32))

    def source_layout(tree, name: str):
        """A Dew gradient in Diffusers' layout: the one tensor translated alone names its place, and
        kernels transpose back."""
        [(path, _)] = jax.tree_util.tree_flatten_with_path(translate_vae_weights(
            {f"encoder.{name}": np.zeros(source[name].shape, np.float32)})["encoder"])[0]
        value = tree
        for key in path:
            value = value[key.key]
        if path[-1].key == "kernel":
            return jnp.transpose(value, (3, 2, 0, 1)) if value.ndim == 4 else value.T
        return value

    def dew(image, probe):
        image, probe = jnp.moveaxis(image, 1, -1), jnp.moveaxis(probe, 1, -1)
        moments = model.apply({"params": params}, image)
        if not backward:
            return {"moments": jnp.moveaxis(moments, -1, 1)}
        grad_params, grad_image = jax.grad(lambda p, x: jnp.sum(model.apply({"params": p}, x) * probe),
                                           argnums=(0, 1))(params, image)
        return {"grad_image": jnp.moveaxis(grad_image, -1, 1),
                **{f"grad_param.{name}": source_layout(grad_params, name) for name in names}}

    def diffusers_walk(model, dtype, context):
        @as_given
        def run(image, probe):
            with context():
                found = walk(model, torch.from_numpy(np.asarray(image)), torch.from_numpy(np.asarray(probe)),
                             dtype)
            return ({"moments": found["moments"]} if not backward else
                    {key: value for key, value in found.items() if key != "moments"})
        return run

    assert_as_exact(dew, diffusers_walk(encoder, torch.float32, contextlib.nullcontext), draw,
                    f"batch {batch} {'gradients' if backward else 'moments'}",
                    seeds=VAE_ENCODER_SEEDS[backward],
                    truth_fn=diffusers_walk(wide, torch.float64, widened_torch))


LINEARIZATION_SEEDS = 8
"""The strided convolutions' declared draws for each linearization, the device's run against the host's
(check 1), sized by `seeds_for` paired with the device's run on the CPU."""


@pytest.mark.parametrize("transform", ["jvp", "transpose", "forward_over_reverse", "reverse_over_forward"])
def test_strided_convolution_linearizations_keep_host_precision(transform):
    """Random filters expose a tangent chain just as they expose the primal.

    A boundary whose JVP drops the barrier leaves the same miscompiled
    convolution chain in the tangent-only program.
    """
    def network(dtype):
        return nn.Sequential([
            Conv(12, (1, 1), use_bias=False, dtype=dtype, precision=jax.lax.Precision.HIGHEST),
            Conv(8, (3, 3), strides=2, padding="VALID", use_bias=False,
                 dtype=dtype, precision=jax.lax.Precision.HIGHEST),
        ])

    host, device = jax.devices("cpu")[0], jax.devices()[0]
    with jax.default_device(host):
        variables = jax.tree.map(np.asarray, network(np.float32).init(jax.random.key(0),
                                                                      np.zeros((1, 17, 17, 3), np.float32)))

    def draw(seed):
        rng = np.random.default_rng(seed)
        image, direction = (rng.standard_normal((1, 17, 17, 3)).astype(np.float32) for _ in range(2))
        return image, direction, rng.standard_normal((1, 8, 8, 8)).astype(np.float32)

    def evaluate(dtype, target):
        model = network(dtype)

        @jax.jit
        def run(params, x, tangent, cot):
            def forward(value):
                return model.apply(params, value)
            if transform == "jvp":
                return jax.jvp(forward, (x,), (tangent,))[1]
            if transform == "transpose":
                return jax.linear_transpose(forward, x)(cot)[0]
            if transform == "forward_over_reverse":
                return jax.jvp(jax.grad(lambda value: jnp.sum(forward(value)**2)),
                               (x,), (tangent,))[1]
            return jax.grad(lambda value: jnp.sum(
                jax.jvp(forward, (value,), (tangent,))[1] * forward(value)))(x)

        def evaluated(image, direction, cotangent):
            args = jax.tree.map(lambda x: jax.device_put(np.asarray(x, dtype), target),
                                (variables, image, direction, cotangent))
            with jax.default_device(target):
                return np.asarray(run(*args))

        return as_given(evaluated)

    assert_as_exact(evaluate(np.float32, device), evaluate(np.float32, host), draw, transform,
                    seeds=LINEARIZATION_SEEDS, truth_fn=evaluate(np.float64, host))


def test_strided_convolutions_keep_nested_vmap_and_its_vjp():
    """Small integer operands make every product/sum exact in fp32.

    The largest possible sum in this two-convolution VJP is below 2**24,
    so a mapped axis or tangent lost by the TPU barrier cannot hide in a
    tolerance. The host evaluates the same six images as one batch.
    """
    model = nn.Sequential([
        Conv(4, (1, 1), use_bias=False, precision=jax.lax.Precision.HIGHEST),
        Conv(2, (3, 3), strides=2, padding="VALID", use_bias=False,
             precision=jax.lax.Precision.HIGHEST),
    ])
    image = (np.arange(2 * 3 * 17 * 17 * 3).reshape(2, 3, 17, 17, 3) % 7).astype(np.float32)
    cotangent = (np.arange(2 * 3 * 8 * 8 * 2).reshape(2, 3, 8, 8, 2) % 5 - 2).astype(np.float32)
    host, device = jax.devices("cpu")[0], jax.devices()[0]
    with jax.default_device(host):
        variables = jax.tree.map(jnp.ones_like, model.init(jax.random.key(0), image[0, 0]))

    def mapped(params, x):
        return jax.vmap(jax.vmap(lambda item: model.apply(params, item)))(x)

    results = []
    for forward, target in ((model.apply, host), (mapped, device)):
        arguments = jax.tree.map(
            lambda x, target=target: jax.device_put(x, target), (variables, image, cotangent)
        )

        def loss(variables, x, cot, *, forward=forward):
            return jnp.sum(forward(variables, x) * cot)

        with jax.default_device(target):
            values = jax.jit(forward)(*arguments[:2])
            gradients = jax.jit(jax.grad(loss, argnums=(0, 1)))(*arguments)
            results.append(jax.tree.map(np.asarray, (values, gradients)))
    for actual, expected in zip(jax.tree.leaves(results[1]), jax.tree.leaves(results[0]), strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("architecture, extra", [
    ("simple_dit", {}),
    ("simple_dit", {"patch_bottleneck": 16}),
    ("hybrid_dit", {"ssm_attention_ratio": "all-attn"}),
    ("simple_mmdit", {}),
], ids=["simple_dit", "simple_dit_bottleneck", "hybrid_dit", "simple_mmdit"])
def test_a_scan_order_is_a_permutation_joint_attention_cannot_see(rng, architecture, extra):
    """The hilbert and zigzag orders share one parameter tree (the patch
    embedding's) and differ only in the permutation the tokens travel in.
    Attention is permutation-equivariant and the MLP is per token, so on the
    same weights the two orders must agree once each is unpermuted back to the
    image: the sincos signal has to be permuted with the tokens, the rotation
    has to stay off, and the inverse permutation has to be the inverse. The
    grid is non-square, which a transposed permutation fails on."""
    x = jax.random.normal(rng, (2, 16, 32, 3))
    temb = jnp.ones((2,))
    textcontext = text(features=64)
    config = dict(patch_size=4, emb_features=64, num_layers=2, num_heads=2, mlp_ratio=2, **extra)
    hilbert = models.build(architecture, scan_order="hilbert", **config)
    zigzag = models.build(architecture, scan_order="zigzag", **config)
    params = jax.tree.map(lambda p: p + 0.05 * jax.random.normal(rng, p.shape),
                          hilbert.init(rng, x, temb, textcontext))
    out_hilbert = hilbert.apply(params, x, temb, textcontext)
    out_zigzag = zigzag.apply(params, x, temb, textcontext)
    assert float(jnp.max(jnp.abs(out_hilbert))) > 0.1, "the perturbed weights produce no signal"
    # The two orders sum the same softmax in a different order: 2e-6 observed.
    assert float(jnp.max(jnp.abs(out_hilbert - out_zigzag))) < 1e-5


UVIT_SEEDS = 8
"""The Hilbert UViT's declared draws against the raster one (check 1), sized by `seeds_for` paired with
it on the CPU and an A100."""


def test_a_hilbert_uvit_computes_the_raster_uvits_image(rng):
    """UViT's learned patch positions follow their patches into the scan
    order (`scan_ordered`), as the patches' tokens do, so on the same weights
    a Hilbert-ordered UViT computes the raster one's image, as exact as the
    raster model computes it (check 1 over drawn images, the truth the raster
    model in float64), where positions added in raster order give each moved
    patch another's (audit NN-A3). The grid is non-square, which a
    transposed permutation fails on."""
    shape = (2, 16, 32, 3)
    temb = jnp.ones((2,))
    context = text(tokens=8, features=32)
    config = {"patch_size": 4, "emb_features": 64, "num_layers": 2, "num_heads": 2, "mlp_ratio": 2,
              "image_size": 32, "text_tokens": 8}
    raster = models.build("uvit", **config)
    hilbert = models.build("uvit", scan_order="hilbert", **config)
    params = jax.tree.map(lambda p: p + 0.05 * jax.random.normal(rng, p.shape),
                          raster.init(rng, jnp.zeros(shape), temb, context))

    def draw(seed):
        return (jax.random.normal(jax.random.key(seed), shape),)

    def image(model):
        return lambda x: model.apply(params, x, temb, context)

    def truth(x):
        held, steps, given = widened((params, temb, context))
        return raster.clone(dtype=jnp.float64).apply(held, x, steps, given)

    assert_as_exact(image(hilbert), image(raster), draw, "a Hilbert UViT's image", seeds=UVIT_SEEDS,
                    truth_fn=truth)


@pytest.mark.parametrize("scan_order", ["hilbert", "zigzag"])
def test_a_scan_order_model_traces_under_jit(rng, scan_order):
    """The permutation is host data that rides into the compiled step as a
    constant, so a scan-order model compiles like a raster one and computes
    the same thing compiled as eager."""
    model = SimpleDiT(patch_size=4, emb_features=64, num_layers=2, num_heads=2, mlp_ratio=2,
                      scan_order=scan_order)
    x, temb, textcontext = small_inputs(rng)
    params = jax.jit(model.init)(rng, x, temb, textcontext)
    params = jax.tree.map(lambda p: p + 0.05, params)
    compiled = jax.jit(model.apply)(params, x, temb, textcontext)
    # 7.5e-7 observed: XLA fuses the compiled graph differently.
    assert jnp.allclose(compiled, model.apply(params, x, temb, textcontext), atol=1e-5)


@pytest.mark.parametrize("scan_order", ["raster", "hilbert", "zigzag"])
def test_2d_fusion_convolves_the_grid_not_the_scan(rng, scan_order):
    """The spatial fusion of an SSM block sees the row-major grid whatever
    order the tokens arrive in: a depthwise kernel that reads the neighbour to
    the right shifts the grid by one column, and the result comes back in the
    scan order it was given."""
    H_P = W_P = 4
    block = ModulatedBlock(features=3, num_heads=1, mixer='ssm', ssm_state_dim=2,
                           use_2d_fusion=True, scan_order=scan_order)
    grid = jax.random.normal(rng, (2, H_P, W_P, 3))
    order = {"raster": np.arange(H_P * W_P), "hilbert": hilbert_indices(H_P, W_P),
             "zigzag": zigzag_indices(H_P, W_P)}[scan_order]
    scanned = grid.reshape(2, H_P * W_P, 3)[:, order, :]
    variables = block.init(rng, scanned, jnp.zeros((2, 3)), None)
    # One 3x3 depthwise tap at (row 1, column 2): out[h, w] = in[h, w + 1].
    kernels = jax.tree.map(jnp.zeros_like, variables["params"]["spatial_fusion"])
    kernels["dwconv_dil1"]["kernel"] = kernels["dwconv_dil1"]["kernel"].at[1, 2, 0, :].set(1.0)
    variables = {"params": {**variables["params"], "spatial_fusion": kernels}}
    fused = block.apply(variables, scanned, method=ModulatedBlock._apply_2d_fusion)

    shifted = jnp.pad(grid, ((0, 0), (0, 0), (0, 1), (0, 0)))[:, :, 1:, :]
    expected = (grid + shifted).reshape(2, H_P * W_P, 3)[:, order, :]
    assert jnp.allclose(fused, expected, atol=1e-6)


def test_hilbert_patchify_roundtrip(rng):
    """Patches gathered into hilbert order come with the inverse permutation;
    unpatchify with that permutation must be an exact identity, on a grid
    that is not square and not a power of two on either side."""
    from dew.nn.dit import scan_ordered
    from dew.nn.scan_orders import patchify, unscanned

    x = jax.random.normal(rng, (2, 12, 20, 3))
    patches, inv_idx = scan_ordered(patchify(x, 4), "hilbert", 3, 5)
    assert inv_idx is not None
    rec = unscanned(patches, inv_idx, 4, 12, 20, 3)
    assert jnp.array_equal(rec, x)


@pytest.mark.parametrize("stack", [SimpleDiT, SimpleMMDiT], ids=["simple_dit", "simple_mmdit"])
def test_dropout_is_active_in_train_mode(rng, stack):
    """Dropout is applied in train mode, so `dropout_rate` reaches the blocks."""
    model = stack(patch_size=4, emb_features=64, num_layers=2, num_heads=2,
                  mlp_ratio=2, dropout_rate=0.5)
    x, temb, textcontext = small_inputs(rng)
    params = model.init(rng, x, temb, textcontext)
    # nudge every param off init by a drawn amount: the zero-initialized adaLN gates and final projection
    # make a fresh DiT output exactly zero, hiding dropout, and one nudge for every weight leaves each
    # row's features alike, which a layer norm maps to zero whatever was dropped
    leaves, tree = jax.tree.flatten(params)
    keys = jax.random.split(rng, len(leaves))
    params = jax.tree.unflatten(tree, [leaf + 0.02 * jax.random.normal(key, leaf.shape, leaf.dtype)
                                       for leaf, key in zip(leaves, keys, strict=True)])

    d0 = model.apply(params, x, temb, textcontext, train=True, rngs={"dropout": jax.random.PRNGKey(1)})
    d1 = model.apply(params, x, temb, textcontext, train=True, rngs={"dropout": jax.random.PRNGKey(2)})
    # Exact inequality: dropout zeroes different units, so the outputs must
    # differ bitwise. A tolerance-based check is too weak here because the
    # zero-init output head keeps the magnitudes small.
    assert not jnp.array_equal(d0, d1), "different dropout rngs must give different outputs"

    e0 = model.apply(params, x, temb, textcontext)
    e1 = model.apply(params, x, temb, textcontext)
    assert jnp.array_equal(e0, e1), "eval mode must be deterministic"


def test_the_second_attention_of_a_dual_block_takes_its_dropout():
    """With every gate closed but the second attention's, a dual-attention block's training outputs
    differ between dropout keys and its evaluation ones do not: the second attention's output is
    dropped as the block's other projected outputs are."""
    block = DoubleStreamBlock(8, 2, 4, mlp_hidden=16, attention_impl="reference", dual_attention=True,
                              shared_modulation=True, dropout_rate=0.5)
    image, context = jnp.ones((1, 3, 8)), jnp.ones((1, 2, 8))
    ones, zeros = jnp.ones((1, 8)), jnp.zeros((1, 8))
    image_mods = (zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, ones)
    condition = (image_mods, (zeros,) * 6)
    variables = block.init(jax.random.key(0), image, context, condition)
    variables = jax.tree.map(lambda p: p + 0.02, variables)

    def run(key=None):
        rngs = None if key is None else {"dropout": jax.random.key(key)}
        return block.apply(variables, image, context, condition, train=key is not None, rngs=rngs)[0]

    assert not jnp.array_equal(run(1), run(2))
    assert jnp.array_equal(run(), run())


def test_a_joint_attention_over_an_empty_context_attends_as_the_image_alone():
    """A context of no tokens joins no keys, so the image attends as it does
    with no context at all, and the context's output is as empty as it is."""
    attention = JointAttention(2, 4, attention_impl="reference")
    image, context = jax.random.normal(jax.random.key(0), (1, 3, 8)), jnp.zeros((1, 0, 8))
    variables = attention.init(jax.random.key(1), image, jnp.ones((1, 2, 8)))
    joined, added = attention.apply(variables, image, context)
    alone, _ = attention.apply(variables, image)
    assert added.shape == (1, 0, 8)
    np.testing.assert_array_equal(joined, alone)


def test_mmdit_is_dual_stream(rng):
    """Text must participate in the token sequence: zeroing the text context
    must change the image output through joint attention, beyond the change
    through the pooled conditioning vector."""
    model = SimpleMMDiT(patch_size=4, emb_features=64, num_layers=2, num_heads=2, mlp_ratio=2)
    x, temb, textcontext = small_inputs(rng)
    params = model.init(rng, x, temb, textcontext)
    params = jax.tree.map(lambda p: p + 0.02, params)

    text_a = textcontext
    text_b = TextContext(
        jnp.concatenate([jnp.ones_like(textcontext.hidden[:, :38]) * 2.0, text_a.hidden[:, 38:]], axis=1),
        textcontext.mask)
    out_a = model.apply(params, x, temb, text_a)
    out_b = model.apply(params, x, temb, text_b)
    assert not jnp.allclose(out_a, out_b), "text tokens do not reach the image stream"


def open_the_gates(params, key, scale=0.5):
    """A DiT at init is adaLN-Zero gated with a zero output head, so its
    output barely depends on its input: adding 1.0 to a whole frame moves that
    frame's own prediction by 5e-7. A test of information flow has to open the
    gates and the head first, and only those. The head follows a LayerNorm,
    whose output sums to zero over features, so its kernel is set to random
    values; a constant kernel would cancel to nothing."""
    def open_(path, value):
        name = jax.tree_util.keystr(path)
        if "ada_proj" in name and "kernel" in name:
            return value + scale
        if "final_proj" in name and "kernel" in name:
            return scale * jax.random.normal(key, value.shape, value.dtype)
        return value
    return jax.tree_util.tree_map_with_path(open_, params)


def test_video_dit_temporal_mixing(rng):
    """Temporal blocks must actually mix across frames: perturbing frame 0
    must change the prediction for frame 2, by an amount comparable to what
    it does to frame 0 itself, which is a ratio no summation order moves."""
    from dew.nn.backbones.video_dit import VideoDiT
    model = VideoDiT(patch_size=4, emb_features=64, num_layers=2, num_heads=2, mlp_ratio=2)
    x = jax.random.normal(rng, (1, 3, 16, 16, 3))
    temb = jnp.ones((1,))
    params = open_the_gates(model.init(rng, x, temb, None), rng)

    out_a = model.apply(params, x, temb, None)
    out_b = model.apply(params, x.at[:, 0].add(1.0), temb, None)
    same_frame = jnp.max(jnp.abs(out_a[:, 0] - out_b[:, 0]))
    other_frame = jnp.max(jnp.abs(out_a[:, 2] - out_b[:, 2]))
    assert same_frame > 1e-2, "the perturbation did not reach the model"
    assert other_frame > 1e-3 * same_frame, "no information flow across frames"


def test_unet3d_inflation_reproduces_2d_unet(rng):
    """A UNet3D inflated from a 2D Unet checkpoint must reproduce the 2D model
    frame by frame exactly - the temporal blocks are zero-initialized, so
    training starts from the pretrained image model and only learns motion.
    The checkpoint's Fourier table comes along with its weights."""
    from dew.nn.backbones.unet3d import UNet3D, inflate_unet_variables

    config = {
        "emb_features": 64,
        "feature_depths": [16, 32],
        "attention_configs": [None, Stage(heads=2, dtype=jnp.float32,
                                       use_projection=False, use_self_and_cross=False)],
        "num_res_blocks": 1,
        "num_middle_res_blocks": 1,
    }
    model_2d = Unet(**config)
    model_3d = UNet3D(**config, temporal_heads=2)

    x = jax.random.normal(rng, (2, 3, 16, 16, 3))
    temb = jnp.ones((2,))
    textcontext = text()

    variables_2d = model_2d.init(rng, x[:, 0], temb, textcontext)
    # A converted checkpoint carries a table no fresh init draws.
    variables_2d = {**variables_2d,
                    "constants": jax.tree.map(lambda table: table * 1.5, variables_2d["constants"])}
    variables_3d = model_3d.init(jax.random.PRNGKey(7), x, temb, textcontext)
    inflated = inflate_unet_variables(variables_2d, variables_3d)
    # A checkpoint read as a FrozenDict inflates alike, its temporal blocks the 3D init's.
    frozen = unfreeze(inflate_unet_variables(freeze(variables_2d), variables_3d))
    assert jax.tree.all(jax.tree.map(jnp.array_equal, frozen, unfreeze(inflated)))

    out_3d = model_3d.apply(inflated, x, temb, textcontext)
    frames_2d = jnp.stack(
        [model_2d.apply(variables_2d, x[:, t], temb, textcontext) for t in range(3)], axis=1)
    assert jnp.max(jnp.abs(out_3d - frames_2d)) < 1e-5, "inflated UNet3D does not match the 2D model"


def test_unet3d_temporal_mixing_after_training_signal(rng):
    """Once the temporal gate is nudged off zero, frames must exchange information."""
    from dew.nn.backbones.unet3d import UNet3D

    model = UNet3D(emb_features=64, feature_depths=[16, 32],
                   attention_configs=[None, None], num_res_blocks=1,
                   num_middle_res_blocks=1, temporal_heads=2)
    x = jax.random.normal(rng, (1, 3, 16, 16, 3))
    temb = jnp.ones((1,))
    textcontext = text(batch=1)
    params = model.init(rng, x, temb, textcontext)
    params = jax.tree.map(lambda p: p + 0.02, params)

    out_a = model.apply(params, x, temb, textcontext)
    out_b = model.apply(params, x.at[:, 0].add(1.0), temb, textcontext)
    same_frame = jnp.max(jnp.abs(out_a[:, 0] - out_b[:, 0]))
    other_frame = jnp.max(jnp.abs(out_a[:, 2] - out_b[:, 2]))
    assert other_frame > 1e-3 * same_frame, "no information flow across frames"


def test_a_stage_names_the_dials_the_block_supports(rng):
    """Every `TransformerBlock` dial a stage names reaches the block the unet
    builds from it: `norm_epsilon` changes the output when set."""
    x = jax.random.normal(rng, (2, 16, 16, 3))
    temb = jnp.ones((2,))
    context = text(features=64)

    def output(**stage):
        model = Unet(output_channels=3, emb_features=32, feature_depths=(8, 16),
                     num_res_blocks=1, norm_groups=4,
                     attention_configs=(None, Stage(heads=2, **stage)))
        return model.apply(model.init(rng, x, temb, context), x, temb, context)

    projected = output(use_projection=True)
    assert not jnp.allclose(projected, output(use_projection=True, norm_epsilon=1.0), atol=1e-5)


@pytest.mark.parametrize("dtype, inputs", [(None, jnp.float32), (jnp.float32, jnp.float32),
                                           (jnp.bfloat16, jnp.bfloat16), (None, jnp.float64)],
                         ids=["unset", "float32", "bfloat16", "unset-float64"])
def test_a_stage_without_dtype_computes_in_the_models_dtype(dtype, inputs, rng):
    """An unchanged stage inherits the model dtype, or, when both leave it unset, at least fp32 of its
    input's: a float64 run attends in float64, as FlaxDiff's float64 twin does."""
    from dew.nn.attention_blocks import stage_attention

    expected_dtype = dtype or inputs
    stage = Stage(heads=2)
    with jax.enable_x64(new_val=inputs == jnp.float64):
        block = stage_attention(stage, 8, "reference", dtype, None, "attention")
        explicit = stage_attention(
            Stage(heads=2, dtype=expected_dtype), 8, "reference", dtype, None, "attention")
        x = jnp.ones((1, 4, 8), inputs)
        variables = block.init(rng, x)
        output = block.apply(variables, x)
        assert stage.dtype is None and stage.force_fp32_for_softmax is True
        assert output.dtype == expected_dtype
        np.testing.assert_array_equal(output, explicit.apply(variables, x))


GROUP_NORM_SEEDS = 32
"""The group norm's declared seeds for check 1 (`assert_as_exact`), sized by `seeds_for` on armada."""


@pytest.mark.parametrize("dtype", [None, jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("shape", [(2, 6, 16), (2, 4, 5, 16)])
def test_group_norm_is_flax_s(rng, dtype, shape):
    """Without an activation, `dew.nn.norms.GroupNorm` is flax's
    `nn.GroupNorm`, with its parameter tree: its statistics in the module's
    dtype, else the input's, at least fp32, and its result cast by flax's dtype
    rule, as near float64 over drawn inputs and parameters as flax's (check 1,
    `assert_as_exact`)."""
    ours, reference = GroupNorm(4, epsilon=1e-5, dtype=dtype), nn.GroupNorm(4, epsilon=1e-5, dtype=dtype)
    x = jax.random.normal(rng, shape, jnp.float32)
    params, flax_params = ours.init(rng, x), reference.init(rng, x)
    assert jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(flax_params)
    assert ours.apply(params, x.astype(dtype or jnp.float32)).dtype == reference.apply(
        params, x.astype(dtype or jnp.float32)).dtype

    def draw(seed):
        """Inputs about 1 apart from zero at a spread of 3, in the module's dtype, and parameters
        drawn about flax's initial ones."""
        keys = jax.random.split(jax.random.key(seed), 3)
        drawn = (jax.random.normal(keys[0], shape, jnp.float32) * 3 + 1).astype(dtype or jnp.float32)
        return np.asarray(drawn), jax.tree.map(np.asarray, moved(params, keys[1], 0.1))

    def truth(drawn, moved):
        return nn.GroupNorm(4, epsilon=1e-5, dtype=jnp.float64).apply(moved, drawn)

    assert_as_exact(jax.jit(lambda drawn, moved: ours.apply(moved, drawn)),
                    jax.jit(lambda drawn, moved: reference.apply(moved, drawn)), draw,
                    f"group norm {dtype} {shape}", seeds=GROUP_NORM_SEEDS, truth_fn=truth)


def offset_draw(shape):
    """Check 1's draw for a norm over `shape`: activations a hundred from zero at a spread of one, where
    E[x^2] - E[x]^2 cancels in fp32, a scale and bias about the initial ones, and a cotangent."""
    def draw(seed):
        keys = jax.random.split(jax.random.key(seed), 4)
        return (np.asarray(jax.random.normal(keys[0], shape, jnp.float32) + 100.0),
                np.asarray(1.0 + 0.1 * jax.random.normal(keys[1], shape[-1:], jnp.float32)),
                np.asarray(0.1 * jax.random.normal(keys[2], shape[-1:], jnp.float32)),
                np.asarray(jax.random.normal(keys[3], shape, jnp.float32)))
    return draw


def normalized(module):
    """A norm `module`'s output and its gradients along a cotangent, of its input, scale and bias."""
    def apply(x, scale, bias):
        return module.apply({"params": {"scale": scale, "bias": bias}}, x)
    return with_gradients(apply)


OFFSET_GROUP_NORM_SEEDS = 64
"""The offset group norm's declared seeds for check 1 (`assert_as_exact`), sized by `seeds_for` on armada."""


def test_group_norm_is_as_exact_as_torch_s_where_a_group_s_mean_dwarfs_its_spread():
    """A group's variance is its centered mean square, as torch's GroupNorm takes it: over activations a
    hundred from zero at a spread of one, where E[x^2] - E[x]^2 cancels in fp32, the forward and the
    gradients of the input, scale and bias along a drawn cotangent are on par (check 1) with torch's
    float32 GroupNorm from flax's in float64."""
    import torch

    shape, groups = (2, 4, 4, 16), 4

    def torch_s(x, scale, bias):
        return torch.nn.functional.group_norm(x.movedim(-1, 1), groups, scale, bias, eps=1e-6).movedim(1, -1)

    assert_as_exact(normalized(GroupNorm(groups, epsilon=1e-6)), torch_with_gradients(torch_s),
                    offset_draw(shape), "offset group norm", seeds=OFFSET_GROUP_NORM_SEEDS,
                    truth_fn=normalized(nn.GroupNorm(groups, epsilon=1e-6, dtype=jnp.float64)))


OFFSET_LAYER_NORM_SEEDS = 64
"""The offset layer norm's declared seeds for check 1 (`assert_as_exact`), sized by `seeds_for` on armada
(cfbb134bf)."""


def test_a_layer_norms_variance_is_its_centered_mean_square():
    """A layer's variance is its centered mean square, as torch's LayerNorm takes it (`dew.nn.norms.
    LayerNorm`): over activations a hundred from zero at a spread of one, where E[x^2] - E[x]^2 cancels in
    fp32, the forward and the gradients of the input, scale and bias along a drawn cotangent are on par
    (check 1) with torch's float32 LayerNorm from flax's in float64 (row 5907)."""
    import torch

    shape = (2, 6, 16)

    def torch_s(x, scale, bias):
        return torch.nn.functional.layer_norm(x, shape[-1:], scale, bias, eps=1e-5)

    assert_as_exact(normalized(LayerNorm(epsilon=1e-5)), torch_with_gradients(torch_s), offset_draw(shape),
                    "offset layer norm", seeds=OFFSET_LAYER_NORM_SEEDS,
                    truth_fn=normalized(nn.LayerNorm(epsilon=1e-5, dtype=jnp.float64)))


TOWER_NORM_SEEDS = 8
"""The tower norms' declared seeds for check 1 (`assert_as_exact`)."""

WIDTH = 64
"""The towers' width: RAE's encoders take a head of 64 (`HEAD_WIDTH`)."""


def _tower_layout(tower):
    """`tower`'s tiny layout (`ViTLayout`) of one layer `WIDTH` wide, as its module builds it; None for
    CLIP's text tower, whose block is `clip_layer`'s own."""
    from dew.nn.autoencoders.rae import RepresentationEncoder
    from dew.nn.vision.llama4 import Llama4Vision
    from dew.nn.vision.qwen35 import Qwen35Vision
    from dew.nn.vision.siglip import SiglipVision

    sizes = {"hidden_size": WIDTH, "intermediate_size": 2 * WIDTH, "num_heads": 2, "patch_size": 8}
    return {"clip text": lambda: None,
            "siglip": lambda: SiglipVision(num_layers=1, image_size=16, **sizes).layout(),
            "llama4": lambda: Llama4Vision(num_layers=1, image_size=16, **sizes).layout(),
            "qwen35": lambda: Qwen35Vision(depth=1, num_position_embeddings=4, **sizes).layout(),
            "rae": lambda: RepresentationEncoder("siglip2", width=WIDTH, layers=1, patch=8,
                                                 input_size=16).layout()}[tower]()


@pytest.mark.parametrize("tower", ["clip text", "siglip", "llama4", "qwen35", "rae"])
def test_a_towers_layer_norms_read_rows_as_torch_s_do(tower):
    """Each tower's block normalizes its input as torch's LayerNorm does, every tower's reference being
    torch's: over rows a hundred from zero at a spread of one, where flax's one-pass E[x^2] - E[x]^2
    cancels in fp32, the block's first norm's output (`input_layernorm`) is on par (check 1) with torch's
    float32 LayerNorm of the same rows at that norm's parameters, from their float64."""
    import torch

    from dew.nn.inputs import AttentionMetadata
    from dew.nn.text_encoders import clip_layer
    from dew.nn.vision.vit import vit_block

    layout = _tower_layout(tower)
    block = (clip_layer(WIDTH, 2, 2 * WIDTH, 8) if layout is None
             else vit_block(layout, 0, dtype=jnp.float32, precision=None))
    eps = 1e-5 if layout is None else layout.norm_eps
    rotary = layout is not None and layout.rotary is not None
    metadata = AttentionMetadata(rotary_positions=jnp.zeros((2, 8, 2), jnp.int32)) if rotary else None
    rows = jnp.zeros((2, 8, WIDTH), jnp.float32)
    params = block.init(jax.random.key(0), rows, attention_metadata=metadata)["params"]

    def draw(seed):
        keys = jax.random.split(jax.random.key(seed), 3)
        return (np.asarray(jax.random.normal(keys[0], rows.shape, jnp.float32) + 100.0),
                np.asarray(1.0 + 0.1 * jax.random.normal(keys[1], (WIDTH,), jnp.float32)),
                np.asarray(0.1 * jax.random.normal(keys[2], (WIDTH,), jnp.float32)))

    def normalized(x, scale, bias):
        held = {**params, "input_layernorm": {"scale": scale, "bias": bias}}
        _, state = block.apply({"params": held}, x, attention_metadata=metadata, capture_intermediates=True,
                               mutable=["intermediates"])
        return state["intermediates"]["input_layernorm"]["__call__"][0]

    def torch_s(x, scale, bias):
        x, scale, bias = (torch.from_numpy(np.asarray(each)) for each in (x, scale, bias))
        return torch.nn.functional.layer_norm(x, (WIDTH,), scale, bias, eps=eps).numpy()

    def truth(x, scale, bias):
        x = np.asarray(x, np.float64)
        centered = x - x.mean(-1, keepdims=True)
        return centered / np.sqrt((centered ** 2).mean(-1, keepdims=True) + eps) * scale + bias

    assert_as_exact(normalized, as_given(torch_s), draw, f"{tower} tower norm", seeds=TOWER_NORM_SEEDS,
                    truth_fn=as_given(truth))


def test_a_group_norm_narrower_than_its_input_normalizes_at_its_own_width(rng):
    """A float64 input under a float32 group norm is normalized at the module's width, statistics and
    affine result alike, as flax's GroupNorm of that dtype normalizes it (its centered variance), the
    activation reading that result: the module's output is flax's activated, bit for bit (row 3465)."""
    with jax.enable_x64(new_val=True):
        x = jax.random.normal(rng, (2, 4, 4, 16), jnp.float64) * 3 + 1
        ours = GroupNorm(4, activation=nn.swish, dtype=jnp.float32)
        params = ours.init(rng, x)
        reference = nn.GroupNorm(4, epsilon=1e-6, use_fast_variance=False, dtype=jnp.float32)
        output = ours.apply(params, x)
        assert output.dtype == jnp.float32
        np.testing.assert_array_equal(np.asarray(output), np.asarray(nn.swish(reference.apply(params, x))))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_group_norm_activates_its_wide_output_before_its_one_cast(rng, dtype):
    """The activation reads the fp32 affine result, as Diffusers' Flax VAE
    and UNet compute it (vae_flax.py:154-163, a GroupNorm with no dtype
    before swish), and the result rounds once to the module's dtype; the
    gradient keeps the input as it arrived."""
    x = (jax.random.normal(rng, (2, 4, 4, 16)) * 3).astype(dtype)
    ours = GroupNorm(4, activation=nn.swish, dtype=dtype)
    params = ours.init(rng, x)
    wide = nn.swish(nn.GroupNorm(4, use_fast_variance=False).apply(params, x.astype(jnp.float32)))
    assert jnp.array_equal(ours.apply(params, x), wide.astype(dtype))
    # The pullback holds the input as it arrived, no wider activation.
    _, pullback = jax.vjp(lambda x: ours.apply(params, x), x)
    held = [leaf for leaf in jax.tree.leaves(pullback) if leaf.size == x.size]
    assert held and {leaf.dtype for leaf in held} == {x.dtype}


LAYER_NORM_SEEDS = 64
"""The layer norm's declared seeds for check 1 (`assert_as_exact`), sized by `seeds_for` on armada."""


def test_every_production_layer_norm_is_dews_rematerialized_one():
    """The decoder block's exact-variance layer norm, the offset layer norm's core and Gemma 4 Unified's
    patch norm are Dew's `LayerNorm`, rematerialized saving nothing, as every other is (ledger 3010):
    each one's differentiated program checkpoints its normalization."""
    from dew.nn.backbones.decoder_block import decoder_norm
    from dew.nn.norms import OffsetLayerNorm
    from dew.nn.vision.gemma4_unified import Gemma4UnifiedVision

    def checkpoints(module: nn.Module, x: jax.Array) -> bool:
        variables = module.init(jax.random.key(0), x)
        traced = jax.make_jaxpr(jax.grad(lambda v: jnp.sum(module.apply(v, x) ** 2)))(variables)
        return "checkpoint" in str(traced) or "remat" in str(traced)

    x = jnp.ones((2, 8), jnp.float32)
    built = decoder_norm("layer", epsilon=1e-5, bias=True, scale_offset=False, scale_after_cast=False,
                         dtype=jnp.float32)
    assert checkpoints(built(name="norm"), x)
    assert checkpoints(OffsetLayerNorm(), x)
    patches = Gemma4UnifiedVision().build()
    assert patches.norm is not None and isinstance(patches.norm(), LayerNorm)


@pytest.mark.parametrize('use_scale,use_bias', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('dtype', [None, jnp.bfloat16, jnp.float32])
def test_layer_norm_is_flax_s_keeping_only_its_input_for_the_backward(rng, use_scale, use_bias, dtype):
    """`dew.nn.norms.LayerNorm` is flax's `nn.LayerNorm` under `nn.remat` saving nothing, its variance
    centered: the same parameter tree and auto-named scope; its output and the gradients of its input and
    parameters as near float64 over drawn inputs and parameters as flax's own (check 1); and its
    pullback holds the input as it arrived, no wider copy of the activations."""
    x = jax.random.normal(rng, (2, 6, 16), jnp.float32)
    x = x.astype(jnp.bfloat16) if dtype is jnp.bfloat16 else x
    fields = {"epsilon": 1e-5, "use_scale": use_scale, "use_bias": use_bias, "dtype": dtype}
    ours, reference = LayerNorm(**fields), nn.LayerNorm(**fields)
    params, flax_params = ours.init(rng, x), reference.init(rng, x)
    assert jax.tree.map(lambda leaf: (leaf.shape, leaf.dtype), params) == jax.tree.map(
        lambda leaf: (leaf.shape, leaf.dtype), flax_params)

    class Named(nn.Module):
        norm: type[nn.Module]

        @nn.compact
        def __call__(self, x):
            return self.norm(**fields)(x)

    assert jax.tree.structure(Named(LayerNorm).init(rng, x)) == jax.tree.structure(
        Named(nn.LayerNorm).init(rng, x))

    def draw(seed):
        """Inputs about 1 apart from zero at a spread of 3, in the input's dtype, flax's initial
        parameters moved about them, and a cotangent for the output."""
        keys = jax.random.split(jax.random.key(seed), 3)
        drawn = (jax.random.normal(keys[0], x.shape, jnp.float32) * 3 + 1).astype(x.dtype)
        cotangent = jax.random.normal(keys[2], x.shape, jnp.float32)
        return np.asarray(drawn), jax.tree.map(np.asarray, moved(params, keys[1], 0.1)), \
            np.asarray(cotangent)

    def observed(module):
        """The module's output and the gradients of its input and parameters, under a cotangent of the
        output's own dtype."""
        run = with_gradients(lambda value, variables: module.apply(variables, value))

        def step(drawn, moved, cotangent):
            return run(drawn, moved, cotangent.astype(jax.eval_shape(module.apply, moved, drawn).dtype))
        return step

    assert ours.apply(params, x).dtype == reference.apply(flax_params, x).dtype
    wide = nn.LayerNorm(**{**fields, "dtype": jnp.float64})
    assert_as_exact(jax.jit(observed(ours)), jax.jit(observed(reference)), draw,
                    f"layer norm {dtype} scale={use_scale} bias={use_bias}", seeds=LAYER_NORM_SEEDS,
                    truth_fn=observed(wide))
    _, pullback = jax.vjp(lambda value: ours.apply(params, value), x)
    held = [leaf for leaf in jax.tree.leaves(pullback) if np.size(leaf) == x.size]
    assert held and {leaf.dtype for leaf in held} == {x.dtype}


MIXED_ROW_SEEDS = 64
"""The layer norm's declared seeds for check 1 over bf16 rows of mixed magnitudes, sized by `seeds_for` on
armada."""


def test_a_layer_norm_centers_a_bf16_row_in_float32():
    """A bf16 row of mixed magnitudes (outliers of 256 either side among values about one, as an
    activation's channels hold them) is centered in float32, as flax's `nn.LayerNorm` centers it: run as
    given, op by op, where no compiler keeps a bf16 difference wide, Dew's output and the gradients of its
    input, scale and bias are on par with flax's (check 1) from float64, where a difference taken in bf16
    rounds the values about one to its 8 bits."""
    shape = (2, 6, 16)

    def draw(seed):
        keys = jax.random.split(jax.random.key(seed), 6)
        outliers = 256.0 * jnp.sign(jax.random.normal(keys[1], shape)) * (
            jax.random.uniform(keys[2], shape) < 0.125)
        return (np.asarray((jax.random.normal(keys[0], shape) + outliers).astype(jnp.bfloat16)),
                np.asarray(1.0 + 0.1 * jax.random.normal(keys[3], shape[-1:], jnp.float32)),
                np.asarray(0.1 * jax.random.normal(keys[4], shape[-1:], jnp.float32)),
                np.asarray(jax.random.normal(keys[5], shape, jnp.float32)))

    assert_as_exact(as_given(normalized(LayerNorm(epsilon=1e-5))),
                    as_given(normalized(nn.LayerNorm(epsilon=1e-5))), draw, "bf16 mixed-row layer norm",
                    seeds=MIXED_ROW_SEEDS,
                    truth_fn=normalized(nn.LayerNorm(epsilon=1e-5, dtype=jnp.float64)))


SHARDED_LAYER_NORM_SEEDS = 64
"""The sharded layer norm's declared seeds for check 1, sized by `seeds_for` on armada."""


def test_a_layer_norm_over_sharded_features_centers_each_row_by_one_mean():
    """A layer norm whose features lie in shards (flax's `axis_name`, its statistics over every shard) reads
    each row less one mean, the same on every shard: over rows whose two shards' means differ (one about
    zero, one about ten), mapped over the shards under that name, its output and the gradients of its
    input, scale and bias are on par with flax's (check 1) from float64, where each shard less its own
    mean would center it alone."""
    # Two shards of four rows, eight features a shard.
    shape = (2, 4, 8)

    def sharded(module):
        def apply(x, scale, bias):
            def shard(x, scale, bias):
                return module.apply({"params": {"scale": scale, "bias": bias}}, x)
            return jax.vmap(shard, axis_name="shards")(x, scale, bias)
        return with_gradients(apply)

    def draw(seed):
        keys = jax.random.split(jax.random.key(seed), 4)
        offsets = np.array([0.0, 10.0], np.float32)[:, None, None]
        return (np.asarray(jax.random.normal(keys[0], shape, jnp.float32)) + offsets,
                np.asarray(1.0 + 0.1 * jax.random.normal(keys[1], (shape[0], shape[-1]), jnp.float32)),
                np.asarray(0.1 * jax.random.normal(keys[2], (shape[0], shape[-1]), jnp.float32)),
                np.asarray(jax.random.normal(keys[3], shape, jnp.float32)))

    fields = {"epsilon": 1e-5, "axis_name": "shards"}
    assert_as_exact(sharded(LayerNorm(**fields)), sharded(nn.LayerNorm(**fields)), draw,
                    "sharded layer norm", seeds=SHARDED_LAYER_NORM_SEEDS,
                    truth_fn=sharded(nn.LayerNorm(**fields, dtype=jnp.float64)))
