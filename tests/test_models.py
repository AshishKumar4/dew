"""Invariants of the model architectures that a training run does not check.

Every registered architecture trains through `fit` in test_architectures.py;
what is here is the behaviour a forward pass has to have beyond running: the
scan orders, the position signal, the mixing across frames, the inflation of
a 2D UNet into a 3D one, and the stage values the UNets are configured with.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.backbones.unet import Unet
from dew.nn.backbones.dit import SimpleDiT
from dew.nn.dit import ModulatedBlock, TextContext
from dew.nn.backbones.mmdit import SimpleMMDiT
from dew.nn.backbones.ssm_dit import HybridSSMAttentionDiT
from dew.nn.attention import Stage
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
    loss the objective takes from the prediction is an fp32 one."""
    model = SimpleDiT(patch_size=4, emb_features=64, num_layers=2, num_heads=2,
                      mlp_ratio=2, dtype=jnp.bfloat16)
    x, temb, textcontext = small_inputs(rng)
    params = model.init(rng, x, temb, textcontext)
    assert model.apply(params, x, temb, textcontext).dtype == jnp.float32


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
    has to stay off, and the inverse permutation has to be the inverse. A
    non-square grid, so a transposed permutation cannot pass."""
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
    must change the image output through joint attention, not just through
    the pooled conditioning vector."""
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
    anything else is refused rather than silently dropped, as precision and
    the softmax flag already are. The reference path keeps honoring it."""
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
                                        implementation=None).dtype == jnp.float32


def open_the_gates(params, key, scale=0.5):
    """A DiT at init is adaLN-Zero gated with a zero output head, so its
    output barely depends on its input: adding 1.0 to a whole frame moves that
    frame's own prediction by 5e-7. A test of information flow has to open the
    gates and the head first, and only those. The head follows a LayerNorm,
    whose output sums to zero over features, so its kernel is set to random
    values rather than a constant, which would cancel to nothing."""
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
    training starts from the pretrained image model and only learns motion."""
    from dew.nn.backbones.unet3d import UNet3D, inflate_unet_params

    config = dict(
        emb_features=64,
        feature_depths=[16, 32],
        attention_configs=[None, Stage(heads=2, dtype=jnp.float32,
                                       use_projection=False, use_self_and_cross=False)],
        num_res_blocks=1,
        num_middle_res_blocks=1,
    )
    model_2d = Unet(**config)
    model_3d = UNet3D(**config, temporal_heads=2)

    x = jax.random.normal(rng, (2, 3, 16, 16, 3))
    temb = jnp.ones((2,))
    textcontext = text()

    params_2d = model_2d.init(rng, x[:, 0], temb, textcontext)
    params_3d = model_3d.init(jax.random.PRNGKey(7), x, temb, textcontext)
    inflated = {"params": inflate_unet_params(params_2d["params"], params_3d["params"])}

    out_3d = model_3d.apply(inflated, x, temb, textcontext)
    frames_2d = jnp.stack(
        [model_2d.apply(params_2d, x[:, t], temb, textcontext) for t in range(3)], axis=1)
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


def test_non_symmetric_attention_configs_init(rng):
    """attention_configs is per stage and need not be symmetric: a stage that
    is None must not decide anything for the stages that are not."""
    from dew.nn.backbones.unet3d import UNet3D

    config = dict(
        emb_features=64,
        feature_depths=[16, 32],
        attention_configs=[Stage(heads=2, dtype=jnp.float32,
                                 use_projection=False, use_self_and_cross=False), None],
        num_res_blocks=1,
        num_middle_res_blocks=1,
    )
    temb = jnp.ones((2,))
    textcontext = text()

    image = jax.random.normal(rng, (2, 16, 16, 3))
    Unet(**config).init(rng, image, temb, textcontext)

    video = jax.random.normal(rng, (2, 3, 16, 16, 3))
    UNet3D(**config, temporal_heads=2).init(rng, video, temb, textcontext)


def test_a_stage_with_an_unknown_field_is_refused():
    """Design rule 6: an unknown field raises, so a misspelled dial cannot
    leave the dial it meant at its default in silence."""
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
    builds from it: `use_linear_attention` and `norm_epsilon` each change the
    output when set, so the unet cannot drop one on the way."""
    x = jax.random.normal(rng, (2, 16, 16, 3))
    temb = jnp.ones((2,))
    context = text(features=64)

    def output(**stage):
        model = Unet(output_channels=3, emb_features=32, feature_depths=(8, 16),
                     num_res_blocks=1, norm_groups=4,
                     attention_configs=(None, Stage(heads=2, **stage)))
        return model.apply(model.init(rng, x, temb, context), x, temb, context)

    projected = output(use_projection=True)
    assert not jnp.allclose(projected, output(use_projection=True,
                                              use_linear_attention=False), atol=1e-5)
    assert not jnp.allclose(projected, output(use_projection=True, norm_epsilon=1.0), atol=1e-5)


def test_with_precision_fills_a_stage_whichever_shape_it_arrives_in():
    """The run's dtype and the fused-kernel softmax reach into every stage,
    which is what `with_precision` exists for, whether the stage is a value
    or the record of one from a logged config."""
    from dew.registry import with_precision

    record, value = ({"heads": 2}, Stage(heads=2))
    from_record = with_precision("unet", {"attention_configs": [None, record]},
                                 dtype="bfloat16", attention_impl="xla")
    from_value = with_precision("unet", {"attention_configs": [None, value]},
                                dtype="bfloat16", attention_impl="xla")
    # A record keeps the dtype's name, which `build` resolves with every
    # other field; a value is resolved where it is written, since nothing
    # resolves it afterwards. The build boundary makes the two agree.
    assert from_record["attention_configs"][1] == {
        "heads": 2, "dtype": "bfloat16", "force_fp32_for_softmax": True}
    assert from_value["attention_configs"][1] == Stage(
        heads=2, dtype=jnp.bfloat16, force_fp32_for_softmax=True)
    built = [models.build("unet", feature_depths=(8, 16), num_res_blocks=1, norm_groups=4,
                          **fields) for fields in (from_record, from_value)]
    assert built[0].attention_configs == built[1].attention_configs
    assert built[0].attention_configs[1].dtype == jnp.bfloat16
    assert built[0].attention_configs[1].force_fp32_for_softmax is True


def test_a_block_pattern_and_a_ratio_together_are_refused():
    """`block_pattern` names every layer's mixer and `ssm_attention_ratio`
    names them by ratio; set together they would have to disagree in silence,
    so the hybrid DiT refuses the pair and takes either alone."""
    model = HybridSSMAttentionDiT(patch_size=4, emb_features=32, num_layers=2, num_heads=2,
                                  block_pattern=("ssm", "attn"), ssm_attention_ratio="1:1")
    with pytest.raises(ValueError, match="ssm_attention_ratio"):
        model.init(jax.random.PRNGKey(0), jnp.zeros((1, 8, 8, 3)), jnp.ones((1,)))
    for alone in (dict(block_pattern=("ssm", "attn")), dict(ssm_attention_ratio="1:1")):
        HybridSSMAttentionDiT(patch_size=4, emb_features=32, num_layers=2, num_heads=2,
                              **alone).init(jax.random.PRNGKey(0), jnp.zeros((1, 8, 8, 3)),
                                            jnp.ones((1,)))
