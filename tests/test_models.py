"""Invariants of the model architectures that a training run does not check.

Every registered architecture trains through `fit` in test_architectures.py;
what is here is the behaviour a forward pass has to have beyond running: the
scan orders, the position signal, the mixing across frames, the inflation of
a 2D UNet into a 3D one, and the stage values the UNets are configured with.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from flax.core import freeze, unfreeze
from reference_error import (
    assert_as_exact,
    assert_as_exact_as_the_reference,
    float64_terms,
    widened,
    with_gradients,
)

from dew.nn.attention import GroupNorm, LayerNorm, Stage
from dew.nn.autoencoders.vae import FlaxEncoder, translate_vae_weights
from dew.nn.backbones.dit import SimpleDiT
from dew.nn.backbones.joint import DoubleStreamBlock, JointAttention
from dew.nn.backbones.mmdit import SimpleMMDiT
from dew.nn.backbones.unet import Unet
from dew.nn.conv import Conv
from dew.nn.dit import ModulatedBlock, TextContext
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


@pytest.mark.parametrize("channels", [(32, 32), (32, 64)], ids=["same", "widening"])
def test_a_bf16_vae_block_rounds_where_diffusers_flax_vae_does(channels):
    """In bf16, the VAE's residual block, its two norms ahead of swish, is
    within the float64 gate of Diffusers' own Flax block on the same
    weights: each norm keeps its fp32 result into swish and the next
    convolution casts (vae_flax.py:154-209)."""
    from diffusers.models import vae_flax

    from dew.nn.autoencoders.vae import _resnet

    ours = _resnet(channels[1], 8, jnp.float32)
    theirs = vae_flax.FlaxResnetBlock2D(*channels, groups=8)
    x = np.random.default_rng(5).standard_normal((2, 8, 8, channels[0])).astype(np.float32) * 3
    params = theirs.init(jax.random.key(0), jnp.asarray(x))
    assert jax.tree_util.tree_structure(ours.init(jax.random.key(0), jnp.asarray(x))) == \
        jax.tree_util.tree_structure(params)
    params = jax.tree.map(lambda leaf: leaf + 0.05 * jax.random.normal(jax.random.key(1), leaf.shape),
                          params)
    narrow = jnp.asarray(x, jnp.bfloat16)
    actual = np.asarray(ours.clone(dtype=jnp.bfloat16).apply(params, narrow), np.float32)
    reference = np.asarray(theirs.clone(dtype=jnp.bfloat16).apply(params, narrow), np.float32)
    with jax.enable_x64():
        wide = jax.tree.map(lambda leaf: jnp.asarray(np.asarray(leaf), jnp.float64), params)
        truth = np.asarray(ours.clone(dtype=jnp.float64).apply(wide, jnp.asarray(x, jnp.float64)))
    assert_as_exact_as_the_reference(actual, reference, truth, "bf16 VAE residual block")


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("backward", [False, True], ids=["encode", "vjp"])
def test_a_vae_encoder_keeps_the_host_reference_precision(batch, backward):
    """The full jitted encoder, not an isolated downsampler, and its VJP.

    XLA:TPU's stride-two rewrite moved the batch-one output by 40%. Compare
    both paths to a float64 host evaluation, allowing only the host fp32
    reference's rounding (the shared reference_error rule).
    """
    model = FlaxEncoder(out_channels=4, block_out_channels=(32, 64), layers_per_block=1,
                        norm_num_groups=8)
    host, device = jax.devices("cpu")[0], jax.devices()[0]
    rng = np.random.default_rng(43)
    image = rng.standard_normal((batch, 17, 17, 3)).astype(np.float32)
    cotangent = rng.standard_normal((batch, 8, 8, 8)).astype(np.float32)
    with jax.default_device(host):
        variables = jax.tree.map(np.asarray, model.init(jax.random.key(0), image[:1]))

    def evaluate(dtype, target):
        network = model.clone(dtype=dtype)
        arguments = jax.tree.map(lambda x: jax.device_put(np.asarray(x, dtype), target),
                                  (variables, image, cotangent))

        def run(params, x, cot):
            if backward:
                return jax.grad(lambda p, a: jnp.sum(network.apply(p, a) * cot),
                                argnums=(0, 1))(params, x)
            return network.apply(params, x)

        with jax.default_device(target):
            return jax.tree.map(np.asarray, jax.jit(run)(*arguments))

    reference = evaluate(np.float32, host)
    with jax.enable_x64():
        truth = evaluate(np.float64, host)
    actual = evaluate(np.float32, device)
    if backward:
        # Treat the complete parameter/input VJP as one vector rather than
        # scaling a cancelling bias leaf by its near-zero norm.
        actual, reference, truth = [np.concatenate([x.reshape(-1) for x in jax.tree.leaves(value)])
                                    for value in (actual, reference, truth)]
    assert_as_exact_as_the_reference(
        actual, reference, truth, "VAE encoder VJP" if backward else "VAE encode"
    )


VAE_ENCODER = Path(__file__).resolve().parent / "fixtures" / "vae" / "encoder.npz"


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("backward", [False, True], ids=["encode", "vjp"])
def test_a_vae_encoder_is_as_exact_as_diffusers(batch, backward):
    """The encoder above against Diffusers' own `Encoder` at its geometry
    (tools/vae_encoder_reference.py): the moments, and the gradients of the
    image and every parameter, on whichever backend runs the test, by the
    float64 rule. The host-precision test holds the device to the host;
    this holds both to the published model. On the CPU the ratios are 1.62
    (moments) and 1.60 (gradients) at a batch of one, where XLA convolves
    another way, and 0.99 and 1.01 at four."""
    import ml_dtypes

    with np.load(VAE_ENCODER) as loaded:
        arrays = dict(loaded)
    names = [key.removeprefix("param.") for key in arrays if key.startswith("param.")]
    weights = {f"encoder.{name}": arrays[f"param.{name}"].view(ml_dtypes.bfloat16).astype(np.float32)
               for name in names}
    params = jax.tree.map(jnp.asarray, translate_vae_weights(weights)["encoder"])
    model = FlaxEncoder(out_channels=4, block_out_channels=(32, 64), layers_per_block=1, norm_num_groups=8)
    image = jnp.asarray(np.moveaxis(arrays["image"][:batch], 1, -1))
    probe = jnp.asarray(np.moveaxis(arrays[f"{batch}/probe"], 1, -1))

    def part(precision: str, name: str) -> np.ndarray:
        return arrays[f"{batch}/{precision}.{name}"]

    if not backward:
        moments = jax.jit(model.apply)({"params": params}, image)
        assert_as_exact_as_the_reference(np.moveaxis(np.asarray(moments), -1, 1), part("fp32", "moments"),
                                         part("fp64", "moments"), f"batch {batch} moments")
        return
    def gradients(p, x):
        return jax.grad(lambda p, x: jnp.sum(model.apply({"params": p}, x) * probe), argnums=(0, 1))(p, x)

    grad_params, grad_image = jax.jit(gradients)(params, image)
    # Dew's own path in float64 on the CPU: each gradient entry's terms, what its sums add
    # (`float64_terms`).
    with jax.enable_x64(new_val=True), jax.default_device(jax.devices("cpu")[0]):
        _, (param_terms, image_terms) = float64_terms(gradients, *widened((params, image)))

    def source_layout(tree, name: str) -> np.ndarray:
        """A Dew gradient in Diffusers' layout: the one tensor translated
        alone names its place, and kernels transpose back."""
        shape = arrays[f"param.{name}"].shape
        [(path, _)] = jax.tree_util.tree_flatten_with_path(
            translate_vae_weights({f"encoder.{name}": np.zeros(shape, np.float32)})["encoder"])[0]
        value = tree
        for key in path:
            value = value[key.key]
        value = np.asarray(value)
        if path[-1].key == "kernel":
            return value.transpose(3, 2, 0, 1) if value.ndim == 4 else value.T
        return value

    def native_order(grad_params, grad_image) -> np.ndarray:
        native = [np.moveaxis(np.asarray(grad_image), -1, 1),
                  *(source_layout(grad_params, name) for name in names)]
        return np.concatenate([np.ravel(leaf) for leaf in native])

    def source_order(precision: str) -> np.ndarray:
        leaves = [part(precision, "grad_image"), *(part(precision, f"grad_param.{name}") for name in names)]
        return np.concatenate([np.ravel(leaf) for leaf in leaves])

    assert_as_exact_as_the_reference(native_order(grad_params, grad_image), source_order("fp32"),
                                     source_order("fp64"), f"batch {batch} gradients",
                                     terms=native_order(param_terms, image_terms))


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
    rng = np.random.default_rng(23)
    image, direction = (rng.standard_normal((1, 17, 17, 3)).astype(np.float32) for _ in range(2))
    cotangent = rng.standard_normal((1, 8, 8, 8)).astype(np.float32)
    with jax.default_device(host):
        variables = jax.tree.map(np.asarray, network(np.float32).init(jax.random.key(0), image))

    def evaluate(dtype, target):
        model = network(dtype)
        args = jax.tree.map(lambda x: jax.device_put(np.asarray(x, dtype), target),
                             (variables, image, direction, cotangent))

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

        with jax.default_device(target):
            return np.asarray(jax.jit(run)(*args))

    reference = evaluate(np.float32, host)
    with jax.enable_x64():
        truth = evaluate(np.float64, host)
    actual = evaluate(np.float32, device)
    assert_as_exact_as_the_reference(actual, reference, truth, transform)


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
    # nudge every param off init: the zero-initialized adaLN gates and final
    # projection make a fresh DiT output exactly zero, hiding dropout
    params = jax.tree.map(lambda p: p + 0.02, params)

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


@pytest.mark.parametrize("dtype", [None, jnp.float32, jnp.bfloat16])
def test_a_stage_without_dtype_computes_in_the_models_dtype(dtype, rng):
    """An unchanged stage inherits the model dtype, or fp32 when both leave it unset."""
    from dew.nn.attention import stage_attention

    expected_dtype = dtype or jnp.float32
    stage = Stage(heads=2)
    block = stage_attention(stage, 8, "reference", dtype, None, "attention")
    explicit = stage_attention(
        Stage(heads=2, dtype=expected_dtype), 8, "reference", dtype, None, "attention")
    x = jnp.ones((1, 4, 8), expected_dtype)
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
    """Without an activation, `dew.nn.attention.GroupNorm` is flax's
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
        leaves, tree = jax.tree.flatten(params)
        moved = [leaf + 0.1 * jax.random.normal(key, leaf.shape) for leaf, key
                 in zip(leaves, jax.random.split(keys[1], len(leaves)), strict=True)]
        return np.asarray(drawn), jax.tree.unflatten(tree, [np.asarray(leaf) for leaf in moved])

    def truth(drawn, moved):
        return nn.GroupNorm(4, epsilon=1e-5, dtype=jnp.float64).apply(moved, drawn)

    assert_as_exact(jax.jit(lambda drawn, moved: ours.apply(moved, drawn)),
                    jax.jit(lambda drawn, moved: reference.apply(moved, drawn)), draw,
                    f"group norm {dtype} {shape}", seeds=GROUP_NORM_SEEDS, truth_fn=truth)


OFFSET_GROUP_NORM_SEEDS = 32
"""The offset group norm's declared seeds for check 1 (`assert_as_exact`), sized by `seeds_for` on armada."""


def test_group_norm_is_as_exact_as_torch_s_where_a_group_s_mean_dwarfs_its_spread():
    """A group's variance is its centered mean square, as torch's GroupNorm takes it: over activations a
    hundred from zero at a spread of one, where E[x^2] - E[x]^2 cancels in fp32, the forward and the
    input's gradient are on par (check 1) with torch's float32 GroupNorm from flax's in float64."""
    import torch

    shape, groups = (2, 4, 4, 16), 4
    ours = GroupNorm(groups, epsilon=1e-6)
    params = ours.init(jax.random.key(0), jnp.zeros(shape))
    cotangent = np.asarray(jax.random.normal(jax.random.key(1), shape, jnp.float32))

    def draw(seed):
        keys = jax.random.split(jax.random.key(seed), 3)
        drawn = jax.random.normal(keys[0], shape, jnp.float32) + 100.0
        moved = jax.tree.map(lambda leaf, key: np.asarray(leaf + 0.1 * jax.random.normal(key, leaf.shape)),
                             params, {"params": dict(zip(params["params"], jax.random.split(keys[1], 2),
                                                          strict=True))})
        return np.asarray(drawn), moved

    def dew(drawn, moved):
        value, pullback = jax.vjp(lambda x: ours.apply(moved, x), drawn)
        return {"value": np.asarray(value), "gradient": np.asarray(pullback(cotangent)[0])}

    def torch_s(drawn, moved):
        x = torch.tensor(np.moveaxis(drawn, -1, 1), requires_grad=True)
        y = torch.nn.functional.group_norm(x, groups, torch.tensor(moved["params"]["scale"]),
                                           torch.tensor(moved["params"]["bias"]), eps=1e-6)
        (y * torch.tensor(np.moveaxis(cotangent, -1, 1))).sum().backward()
        return {"value": np.moveaxis(y.detach().numpy(), 1, -1),
                "gradient": np.moveaxis(x.grad.numpy(), 1, -1)}

    def truth(drawn, moved):
        wide = nn.GroupNorm(groups, epsilon=1e-6, dtype=jnp.float64)
        value, pullback = jax.vjp(lambda x: wide.apply(moved, x), drawn)
        return {"value": np.asarray(value), "gradient": np.asarray(pullback(cotangent.astype(np.float64))[0])}

    assert_as_exact(jax.jit(dew), torch_s, draw, "offset group norm", seeds=OFFSET_GROUP_NORM_SEEDS,
                    truth_fn=truth)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_group_norm_activates_its_wide_output_before_its_one_cast(rng, dtype):
    """The activation reads the fp32 affine result, as Diffusers' Flax VAE
    and UNet compute it (vae_flax.py:154-163, a GroupNorm with no dtype
    before swish), and the result rounds once to the module's dtype; the
    gradient keeps the input as it arrived."""
    x = (jax.random.normal(rng, (2, 4, 4, 16)) * 3).astype(dtype)
    ours = GroupNorm(4, activation=nn.swish, dtype=dtype)
    params = ours.init(rng, x)
    wide = nn.swish(nn.GroupNorm(4).apply(params, x.astype(jnp.float32)))
    assert jnp.array_equal(ours.apply(params, x), wide.astype(dtype))
    # The pullback holds the input as it arrived, no wider activation.
    _, pullback = jax.vjp(lambda x: ours.apply(params, x), x)
    held = [leaf for leaf in jax.tree.leaves(pullback) if leaf.size == x.size]
    assert held and {leaf.dtype for leaf in held} == {x.dtype}


LAYER_NORM_SEEDS = 64
"""The layer norm's declared seeds for check 1 (`assert_as_exact`), sized by `seeds_for` on armada."""


@pytest.mark.parametrize('use_scale,use_bias', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('dtype', [None, jnp.bfloat16, jnp.float32])
def test_layer_norm_is_flax_s_keeping_only_its_input_for_the_backward(rng, use_scale, use_bias, dtype):
    """`dew.nn.attention.LayerNorm` is flax's `nn.LayerNorm` under `nn.remat` saving nothing: the
    same parameter tree and auto-named scope; its output and the gradients of its input and
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
        leaves, tree = jax.tree.flatten(params)
        moved = [leaf + 0.1 * jax.random.normal(key, leaf.shape) for leaf, key
                 in zip(leaves, jax.random.split(keys[1], len(leaves)), strict=True)]
        cotangent = jax.random.normal(keys[2], x.shape, jnp.float32)
        return np.asarray(drawn), jax.tree.unflatten(tree, [np.asarray(leaf) for leaf in moved]), \
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
