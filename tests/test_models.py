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
from flax.traverse_util import flatten_dict
from reference_error import assert_as_exact_as_the_reference

from dew.diffusion.process import DenoisingCondition
from dew.nn.attention import LayerNorm, Stage
from dew.nn.autoencoders.vae import FlaxDecoder, FlaxEncoder, translate_vae_weights
from dew.nn.backbones.dit import SimpleDiT
from dew.nn.backbones.mmdit import SimpleMMDiT
from dew.nn.backbones.ssm_dit import HybridSSMAttentionDiT
from dew.nn.backbones.unet import Unet
from dew.nn.backbones.unet_condition import UNet2DCondition, UNetStage
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


def test_a_bf16_dit_predicts_in_fp32(rng):
    """The output head runs in fp32 whatever the model's compute dtype, so the
    loss the objective takes from the prediction is an fp32 one. A head that
    computed in bf16 and cast the result up keeps the dtype and loses the
    mantissa, so the prediction has to carry bits bf16 cannot hold."""
    model = SimpleDiT(patch_size=4, emb_features=64, num_layers=2, num_heads=2,
                      mlp_ratio=2, dtype=jnp.bfloat16)
    x, temb, textcontext = small_inputs(rng)
    # The zero-initialized head predicts exactly zero, a value every dtype
    # holds, so the weights are nudged off init first.
    params = jax.tree.map(lambda p: p + 0.02, model.init(rng, x, temb, textcontext))
    out = model.apply(params, x, temb, textcontext)
    assert out.dtype == jnp.float32
    assert not jnp.array_equal(out, out.astype(jnp.bfloat16).astype(jnp.float32))


def _unet(norm_groups):
    stage = Stage(heads=2, dtype=jnp.bfloat16, force_fp32_for_softmax=True)
    model = Unet(output_channels=3, emb_features=32, feature_depths=(16, 32),
                 attention_configs=(None, stage), num_res_blocks=1, norm_groups=norm_groups,
                 dtype=jnp.bfloat16)
    return model, (jnp.ones((2, 16, 16, 3), jnp.bfloat16), jnp.ones((2,))), {}


def _unet_condition():
    model = UNet2DCondition(stages=(UNetStage(32, 2), UNetStage(64, 2)), blocks_per_level=1,
                            norm_groups=8, dtype=jnp.bfloat16)
    context = DenoisingCondition(context=jnp.ones((2, 8, 32), jnp.bfloat16))
    return model, (jnp.ones((2, 16, 16, 4), jnp.bfloat16), jnp.ones((2,))), {"conditioning": context}


def _vae_encoder():
    model = FlaxEncoder(out_channels=4, block_out_channels=(32, 64), layers_per_block=1, norm_num_groups=8,
                        dtype=jnp.bfloat16)
    return model, (jnp.ones((2, 16, 16, 3), jnp.bfloat16),), {}


def _vae_decoder():
    model = FlaxDecoder(out_channels=3, block_out_channels=(32, 64), layers_per_block=1, norm_num_groups=8,
                        dtype=jnp.bfloat16)
    return model, (jnp.ones((2, 8, 8, 4), jnp.bfloat16),), {}


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
    grad_params, grad_image = jax.jit(jax.grad(
        lambda p, x: jnp.sum(model.apply({"params": p}, x) * probe), argnums=(0, 1)))(params, image)

    def source_layout(name: str) -> np.ndarray:
        """A Dew gradient in Diffusers' layout: the one tensor translated
        alone names its place, and kernels transpose back."""
        shape = arrays[f"param.{name}"].shape
        [(path, _)] = jax.tree_util.tree_flatten_with_path(
            translate_vae_weights({f"encoder.{name}": np.zeros(shape, np.float32)})["encoder"])[0]
        value = grad_params
        for key in path:
            value = value[key.key]
        value = np.asarray(value)
        if path[-1].key == "kernel":
            return value.transpose(3, 2, 0, 1) if value.ndim == 4 else value.T
        return value

    native = [np.moveaxis(np.asarray(grad_image), -1, 1), *(source_layout(name) for name in names)]

    def source_order(precision: str) -> np.ndarray:
        leaves = [part(precision, "grad_image"), *(part(precision, f"grad_param.{name}") for name in names)]
        return np.concatenate([np.ravel(leaf) for leaf in leaves])

    native_order = np.concatenate([np.ravel(leaf) for leaf in native])
    assert_as_exact_as_the_reference(native_order, source_order("fp32"), source_order("fp64"),
                                     f"batch {batch} gradients")


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


@pytest.mark.parametrize("build", [
    lambda: _unet(8), lambda: _unet(0), _unet_condition, _vae_encoder, _vae_decoder,
], ids=["unet_group_norm", "unet_rms_norm", "unet_condition", "vae_encoder", "vae_decoder"])
def test_a_bf16_convolutional_model_keeps_its_activations_in_bf16(rng, build):
    """With fp32 parameters and a bf16 compute dtype, every image-shaped
    activation the convolutional models produce is bf16, the norms' outputs
    included. A norm that leaves its dtype to promotion returns fp32 against
    its fp32 scale, and the activation and convolution after it then read
    fp32 activations the step keeps for the backward pass."""
    model, inputs, kwargs = build()
    params = model.init(rng, *inputs, **kwargs)
    _, state = model.apply(params, *inputs, **kwargs, capture_intermediates=True)
    activations = {"/".join(path): leaf.dtype
                   for path, outputs in flatten_dict(state["intermediates"]).items()
                   for leaf in jax.tree.leaves(outputs) if leaf.ndim == 4}
    assert any("norm" in path for path in activations)
    assert {path: dtype for path, dtype in activations.items() if dtype != jnp.bfloat16} == {}


@pytest.mark.parametrize("architecture, extra", [
    ("simple_dit", {}),
    ("hybrid_dit", {"ssm_attention_ratio": "all-attn"}),
    ("simple_mmdit", {}),
], ids=["simple_dit", "hybrid_dit", "simple_mmdit"])
def test_a_scan_order_is_a_permutation_joint_attention_cannot_see(rng, architecture, extra):
    """The hilbert and zigzag orders share one parameter tree (`hilbert_projection`
    over raw patches) and differ only in the permutation the tokens travel in.
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
    """patchify returns patches in hilbert order plus the inverse permutation;
    unpatchify with that permutation must be an exact identity, on a grid
    that is not square and not a power of two on either side."""
    from dew.nn.scan_orders import hilbert_patchify, hilbert_unpatchify

    x = jax.random.normal(rng, (2, 12, 20, 3))
    patches, inv_idx = hilbert_patchify(x, 4)
    rec = hilbert_unpatchify(patches, inv_idx, 4, 12, 20, 3)
    assert jnp.array_equal(rec, x)


@pytest.mark.parametrize("options, forwarded, stack_only", [
    ("_TransformerOptions", "_BlockOptions", set()),
    ("_AttentionStackOptions", "_AttentionBlockOptions", set()),
    ("_DiTStackOptions", "_AttentionBlockOptions",
     {"output_channels", "patch_size", "emb_features", "num_layers", "num_heads", "remat", "scan_order"}),
    ("_JepaStackOptions", "_TokenStackOptions", set()),
])
def test_a_stack_option_class_forwards_every_control_it_declares(options, forwarded, stack_only):
    """The DiT stacks hand their shared controls to blocks through a
    TypedDict, whose literal pyright checks; a control added to the class and
    not to its TypedDict would silently not reach the blocks."""
    import dataclasses

    import dew.nn.dit as dit

    declared = {field.name for field in dataclasses.fields(getattr(dit, options))} - {"parent", "name"}
    keys = getattr(dit, forwarded).__required_keys__ | getattr(dit, forwarded).__optional_keys__
    assert declared - stack_only == keys


def test_dropout_is_active_in_train_mode(rng):
    """Dropout is applied in train mode, so `dropout_rate` reaches the blocks."""
    model = SimpleDiT(patch_size=4, emb_features=64, num_layers=2, num_heads=2,
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


def test_attention_impl_parity(rng):
    """Every attention implementation must share one param tree and produce
    the same outputs. 'xla' (the jax.nn fused entrypoint) is verifiable on
    CPU; cudnn/tpu dispatch to the same wrapper."""
    ref = SimpleDiT(patch_size=4, emb_features=64, num_layers=2, num_heads=2, mlp_ratio=2)
    xla = SimpleDiT(patch_size=4, emb_features=64, num_layers=2, num_heads=2, mlp_ratio=2,
                    attention_impl="xla")
    x, temb, textcontext = small_inputs(rng)
    params = ref.init(rng, x, temb, textcontext)
    out_ref = ref.apply(params, x, temb, textcontext)
    out_xla = xla.apply(params, x, temb, textcontext)
    assert jnp.max(jnp.abs(out_ref - out_xla)) < 1e-4


def test_fused_attention_rejects_a_dtype_it_cannot_honor(rng):
    """The fused kernels compute in the inputs' dtype, so a dtype asking for
    anything else raises ValueError, as a HIGH precision and a False softmax
    flag already do. The reference path keeps honoring it."""
    from dew.nn.attention import scaled_dot_product_attention
    query = jax.random.normal(rng, (2, 8, 4, 16), jnp.bfloat16)
    key = jax.random.normal(jax.random.fold_in(rng, 1), (2, 8, 4, 16), jnp.bfloat16)
    value = jax.random.normal(jax.random.fold_in(rng, 2), (2, 8, 4, 16), jnp.bfloat16)
    with pytest.raises(ValueError, match="cannot honor"):
        scaled_dot_product_attention(query, key, value, dtype=jnp.float32,
                                     implementation="xla")
    assert scaled_dot_product_attention(query, key, value, dtype=jnp.bfloat16,
                                        implementation="xla").dtype == jnp.bfloat16
    assert scaled_dot_product_attention(query, key, value, dtype=jnp.float32,
                                        implementation="reference").dtype == jnp.float32


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


def test_non_symmetric_attention_configs_place_attention_on_that_stage_alone(rng):
    """attention_configs is per stage and need not be symmetric: attention on
    the first of two stages builds one block on the way down and its mirror on
    the way up, and the middle block follows the deepest stage, which names
    none. A stage that is None must not decide anything for the stages that
    are not, in either the image or the video stack."""
    from dew.nn.backbones.unet3d import UNet3D

    config = {
        "emb_features": 64,
        "feature_depths": [16, 32],
        "attention_configs": [Stage(heads=2, dtype=jnp.float32,
                                 use_projection=False, use_self_and_cross=False), None],
        "num_res_blocks": 1,
        "num_middle_res_blocks": 1,
    }
    temb = jnp.ones((2,))
    textcontext = text()

    def attention_blocks(variables):
        return sorted(name for name in variables["params"] if "attention" in name)

    image = jax.random.normal(rng, (2, 16, 16, 3))
    video = jax.random.normal(rng, (2, 3, 16, 16, 3))
    expected = ["down_0_attention_0", "up_1_attention_0"]
    assert attention_blocks(Unet(**config).init(rng, image, temb, textcontext)) == expected
    assert attention_blocks(UNet3D(**config, temporal_heads=2).init(
        rng, video, temb, textcontext)) == expected


def test_stages_that_do_not_match_the_feature_depths_are_refused(rng):
    """attention_configs names one stage per feature depth. The decoder walks
    the two reversed, so a list of the wrong length does not lose its odd end:
    it offsets the levels the decoder attends in from the ones the encoder
    does, and the halves stop mirroring with nothing raised. Narrowing
    feature_depths and leaving the stages alone is how a run reaches it, so
    both UNets refuse the disagreement instead of building either half.
    """
    from dew.nn.backbones.unet3d import UNet3D

    config = {"emb_features": 64, "feature_depths": [16, 32], "num_res_blocks": 1,
                  "num_middle_res_blocks": 1,
                  "attention_configs": [None, Stage(heads=2), Stage(heads=2)]}
    temb, textcontext = jnp.ones((2,)), text()
    with pytest.raises(ValueError, match="3 stages for 2 depths"):
        Unet(**config).init(rng, jax.random.normal(rng, (2, 16, 16, 3)), temb, textcontext)
    with pytest.raises(ValueError, match="3 stages for 2 depths"):
        UNet3D(**config, temporal_heads=2).init(
            rng, jax.random.normal(rng, (2, 3, 16, 16, 3)), temb, textcontext)


@pytest.mark.parametrize("video", [False, True])
def test_upsampling_reaches_the_next_decoder_stage_width(rng, video):
    from dew.nn.backbones.unet3d import UNet3D
    from dew.nn.blocks import Upsample

    depths = (8, 16, 32)
    model_type = UNet3D if video else Unet
    model = model_type(emb_features=16, feature_depths=depths,
                       attention_configs=(None,) * len(depths),
                       num_res_blocks=1, num_middle_res_blocks=1, dtype=jnp.float32)
    shape = (1, 2, 16, 16, 3) if video else (1, 16, 16, 3)
    _, variables = model.init_with_output(
        rng, jax.random.normal(rng, shape), jnp.ones((1,)),
        capture_intermediates=lambda module, method: isinstance(module, Upsample))
    transitions = [values["__call__"][0].shape[-3:]
                   for _, values in sorted(variables["intermediates"].items())]
    assert transitions == [(8, 8, 16), (16, 16, 8)]


def test_benchmark_unet_resampling_matches_the_native_model(rng, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from dew.nn.blocks import Upsample

    for name in ("TMPDIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        monkeypatch.setenv(name, str(tmp_path / name))
    from tools.benchmark_torch import Unet as TorchUnet

    depths = (8, 16, 32)
    native = Unet(emb_features=16, feature_depths=depths,
                  attention_configs=(None,) * len(depths),
                  num_res_blocks=1, num_middle_res_blocks=1, dtype=jnp.float32)
    image = jnp.ones((1, 16, 16, 3))
    _, variables = native.init_with_output(
        rng, image, jnp.ones((1,)),
        capture_intermediates=lambda module, method: isinstance(module, Upsample))
    expected = [values["__call__"][0].shape[-3:]
                for _, values in sorted(variables["intermediates"].items())]
    twin = TorchUnet({"feature_depths": depths, "attention_heads": (None,) * len(depths),
                      "emb_features": 16, "num_res_blocks": 1, "num_middle_res_blocks": 1},
                     "reference")
    observed = []

    def record_shape(module, arguments, output):
        observed.append((output.shape[2], output.shape[3], output.shape[1]))

    handles = [layer.register_forward_hook(record_shape) for layer in twin.upsamples]
    try:
        with torch.no_grad():
            twin(torch.ones(1, 16, 16, 3), torch.ones(1), None)
    finally:
        for handle in handles:
            handle.remove()
    assert observed == expected


def test_a_stage_with_an_unknown_field_is_refused():
    """Design rule 6: an unknown field raises ValueError naming it, so a
    misspelled dial fails at build and the dial it meant is never left at
    its default."""
    stages = [None, {"heads": 2, "use_projeciton": True}]
    with pytest.raises(ValueError, match="use_projeciton"):
        models.build("unet", feature_depths=(8, 16), attention_configs=stages,
                     num_res_blocks=1, norm_groups=4)


def test_a_stage_record_builds_the_value():
    """A stage arrives as a dict from a command line or a run record and is
    the declared value by the time the model holds it."""
    model = models.build("unet", feature_depths=(8, 16), num_res_blocks=1, norm_groups=4,
                         attention_configs=[None, {"heads": 2, "use_projection": True}])
    assert model.attention_configs[0] is None
    assert model.attention_configs[1] == Stage(heads=2, use_projection=True)
    assert models.build("unet", feature_depths=(8, 16), num_res_blocks=1, norm_groups=4,
                        attention_configs=[None, Stage(heads=2, use_projection=True)]
                        ).attention_configs[1] == model.attention_configs[1]


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


def test_a_block_pattern_and_a_ratio_together_are_refused():
    """`block_pattern` names every layer's mixer and `ssm_attention_ratio`
    names them by ratio; the hybrid DiT raises ValueError on the pair and
    takes either alone."""
    model = HybridSSMAttentionDiT(patch_size=4, emb_features=32, num_layers=2, num_heads=2,
                                  block_pattern=("ssm", "attn"), ssm_attention_ratio="1:1")
    with pytest.raises(ValueError, match="ssm_attention_ratio"):
        model.init(jax.random.PRNGKey(0), jnp.zeros((1, 8, 8, 3)), jnp.ones((1,)))
    for alone in ({"block_pattern": ("ssm", "attn")}, {"ssm_attention_ratio": "1:1"}):
        HybridSSMAttentionDiT(patch_size=4, emb_features=32, num_layers=2, num_heads=2,
                              **alone).init(jax.random.PRNGKey(0), jnp.zeros((1, 8, 8, 3)),
                                            jnp.ones((1,)))


def test_a_block_pattern_that_misses_a_layer_is_refused():
    """`block_pattern` names every layer's mixer, so one shorter than
    `num_layers` is refused rather than deciding the depth. The hybrid DiT
    built one block per named layer, so a short pattern silently returned a
    shallower model than the config asked for."""
    hybrid = HybridSSMAttentionDiT(patch_size=4, emb_features=32, num_layers=4,
                                   num_heads=2, block_pattern=("ssm", "attn"))
    with pytest.raises(ValueError, match="2 entries for 4 layers"):
        hybrid.init(jax.random.PRNGKey(0), jnp.zeros((1, 8, 8, 3)), jnp.ones((1,)))


@pytest.mark.parametrize('use_scale,use_bias', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('dtype', [None, jnp.bfloat16, jnp.float32])
def test_layer_norm_matches_flax_bit_for_bit(rng, use_scale, use_bias, dtype):
    """`dew.nn.attention.LayerNorm` exists to change which values the backward
    pass keeps, not which values the forward pass produces, so it computes
    flax's `nn.LayerNorm` op for op: E[x] and E[x^2] in fp32, the variance
    from the pair and clipped at zero, the weight folded into the inverse
    deviation before it meets the centered activations, the result cast by
    flax's own dtype rule. A flax release that changes any of those moves a
    checkpoint's outputs, and this is where that shows."""
    x = jax.random.normal(rng, (2, 6, 16), jnp.float32)
    x = x.astype(jnp.bfloat16) if dtype is jnp.bfloat16 else x
    fields = {"epsilon": 1e-5, "use_scale": use_scale, "use_bias": use_bias, "dtype": dtype}
    ours, reference = LayerNorm(**fields), nn.LayerNorm(**fields)
    params, flax_params = ours.init(rng, x), reference.init(rng, x)

    assert jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(flax_params)
    for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(flax_params), strict=True):
        assert a.shape == b.shape and a.dtype == b.dtype
    found, expected = ours.apply(params, x), reference.apply(flax_params, x)
    assert found.dtype == expected.dtype
    assert jnp.array_equal(found, expected)
