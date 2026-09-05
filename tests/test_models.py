"""Forward-pass tests for every model architecture.

Every architecture must build, run a forward pass at a small resolution,
and return the input shape back in float32. This is the first line of
defense against dead configs and shape bugs.
"""

import jax
import jax.numpy as jnp
import pytest

from dew.nn.backbones.unet import Unet
from dew.nn.backbones.dit import SimpleDiT
from dew.nn.dit import TextContext
from dew.nn.backbones.mmdit import SimpleMMDiT
from dew.nn.backbones.uvit import UViT
from dew.nn.backbones.ssm_dit import HybridSSMAttentionDiT
from dew.nn.attention import Stage
from dew.registry import models

RES = 32


def run_forward(model, rng, x, temb, textcontext):
    params = model.init(rng, x, temb, textcontext)
    return model.apply(params, x, temb, textcontext)


def text(batch=2, tokens=77, features=768):
    """A fully real text context, the shape CLIP-L/14 gives."""
    return TextContext(jnp.ones((batch, tokens, features), jnp.float32), jnp.ones((batch, tokens)))


def small_inputs(rng, res=RES, channels=3):
    x = jax.random.normal(rng, (2, res, res, channels))
    temb = jnp.ones((2,))
    return x, temb, text()


def test_unet_forward(rng):
    model = Unet(
        emb_features=64,
        feature_depths=[16, 32],
        attention_configs=[None, Stage(heads=2, dtype=jnp.float32,
                                       use_projection=False, use_self_and_cross=False)],
        num_res_blocks=1,
        num_middle_res_blocks=1,
    )
    x, temb, textcontext = small_inputs(rng)
    out = run_forward(model, rng, x, temb, textcontext)
    assert out.shape == x.shape
    assert out.dtype == jnp.float32


def test_simple_dit_forward(rng):
    model = SimpleDiT(patch_size=4, emb_features=64, num_layers=2, num_heads=2, mlp_ratio=2)
    x, temb, textcontext = small_inputs(rng)
    out = run_forward(model, rng, x, temb, textcontext)
    assert out.shape == x.shape
    assert out.dtype == jnp.float32


def test_simple_dit_bf16_outputs_fp32(rng):
    model = SimpleDiT(patch_size=4, emb_features=64, num_layers=2, num_heads=2,
                      mlp_ratio=2, dtype=jnp.bfloat16)
    x, temb, textcontext = small_inputs(rng)
    out = run_forward(model, rng, x, temb, textcontext)
    assert out.dtype == jnp.float32


def test_simple_dit_non_square(rng):
    model = SimpleDiT(patch_size=4, emb_features=64, num_layers=2, num_heads=2, mlp_ratio=2)
    x = jax.random.normal(rng, (2, 16, 64, 3))
    temb = jnp.ones((2,))
    textcontext = text()
    out = run_forward(model, rng, x, temb, textcontext)
    assert out.shape == x.shape


def test_simple_mmdit_forward(rng):
    model = SimpleMMDiT(patch_size=4, emb_features=64, num_layers=2, num_heads=2, mlp_ratio=2)
    x, temb, textcontext = small_inputs(rng)
    out = run_forward(model, rng, x, temb, textcontext)
    assert out.shape == x.shape


def test_uvit_forward(rng):
    model = UViT(patch_size=4, emb_features=64, num_layers=4, num_heads=2)
    x, temb, textcontext = small_inputs(rng)
    out = run_forward(model, rng, x, temb, textcontext)
    assert out.shape == x.shape


@pytest.mark.parametrize("kwargs", [
    {},
    {"scan_order": "hilbert"},
    {"scan_order": "zigzag"},
    {"scan_order": "zigzag", "use_2d_fusion": True},
    {"scan_order": "hilbert", "use_2d_fusion": True},
])
def test_hybrid_dit_forward(rng, kwargs):
    model = HybridSSMAttentionDiT(patch_size=4, emb_features=64, num_layers=4,
                                  num_heads=2, mlp_ratio=2, ssm_state_dim=8, **kwargs)
    x, temb, textcontext = small_inputs(rng)
    out = run_forward(model, rng, x, temb, textcontext)
    assert out.shape == x.shape


def test_uvit_hilbert_forward(rng):
    """UViT takes the hilbert order and returns the shape it was given."""
    model = UViT(patch_size=4, emb_features=64, num_layers=4, num_heads=2, scan_order="hilbert")
    x, temb, textcontext = small_inputs(rng)
    out = run_forward(model, rng, x, temb, textcontext)
    assert out.shape == x.shape


def test_hilbert_patchify_roundtrip(rng):
    """patchify returns patches in hilbert order plus the inverse permutation;
    unpatchify with that permutation must be an exact identity."""
    from dew.nn.scan_orders import hilbert_patchify, hilbert_unpatchify

    x = jax.random.normal(rng, (2, 16, 16, 3))
    patches, inv_idx = hilbert_patchify(x, 4)
    rec = hilbert_unpatchify(patches, inv_idx, 4, 16, 16, 3)
    assert jnp.allclose(rec, x)


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


def test_hierarchical_mmdit_forward(rng):
    from dew.nn.backbones.mmdit import HierarchicalMMDiT
    model = HierarchicalMMDiT(base_patch_size=2, emb_features=(32, 64, 96),
                              num_layers=(1, 1, 1), num_heads=(2, 2, 2), mlp_ratio=2)
    x, temb, textcontext = small_inputs(rng)
    out = run_forward(model, rng, x, temb, textcontext)
    assert out.shape == x.shape


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


def test_video_dit_forward(rng):
    """Factorized ST video model over (B, T, H, W, C), the replacement for
    the never-wired diffusers-derived UNet3D."""
    from dew.nn.backbones.video_dit import VideoDiT
    model = VideoDiT(patch_size=4, emb_features=64, num_layers=2, num_heads=2, mlp_ratio=2)
    x = jax.random.normal(rng, (2, 3, 16, 16, 3))
    temb = jnp.ones((2,))
    textcontext = text()
    out = run_forward(model, rng, x, temb, textcontext)
    assert out.shape == x.shape
    assert jnp.all(jnp.isfinite(out))


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
    assert out_3d.shape == x.shape
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

############################################################################################################
# A unet stage is a declared value
############################################################################################################

def test_a_stage_with_an_unknown_field_is_refused():
    """Design rule 6: an unknown field raises. A stage used to be an untyped
    dict read with `.get`, so a typo turned the dial it named off in silence."""
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
    """`use_linear_attention` and `norm_epsilon` are TransformerBlock dials no
    config could name while a stage was a dict: the unets passed neither, so
    the projection kind and the norm epsilon were unreachable from a run."""
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
    which is what `with_precision` exists for; a stage is now a value, and a
    record of one still arrives from a logged config."""
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
    """`block_pattern` used to win over `ssm_attention_ratio` in silence
    (ssm_dit.py's own docstring said "overrides"). The ratio stays: its "3:1"
    default is length-independent and a pattern cannot express that, so the
    two are refused together rather than one being dropped."""
    model = HybridSSMAttentionDiT(patch_size=4, emb_features=32, num_layers=2, num_heads=2,
                                  block_pattern=("ssm", "attn"), ssm_attention_ratio="1:1")
    with pytest.raises(ValueError, match="ssm_attention_ratio"):
        model.init(jax.random.PRNGKey(0), jnp.zeros((1, 8, 8, 3)), jnp.ones((1,)))
    # Either alone still builds.
    for alone in (dict(block_pattern=("ssm", "attn")), dict(ssm_attention_ratio="1:1")):
        HybridSSMAttentionDiT(patch_size=4, emb_features=32, num_layers=2, num_heads=2,
                              **alone).init(jax.random.PRNGKey(0), jnp.zeros((1, 8, 8, 3)),
                                            jnp.ones((1,)))

def test_a_stage_keeps_the_defaults_the_dict_read_had():
    """No default moved in the cutover: `dtype` is float32 and not the model's,
    which is why `with_precision` writes into every stage, and a stage that
    names no `precision` takes the model's, as `.get("precision",
    self.precision)` did."""
    stage = Stage(heads=8)
    assert stage.dtype is jnp.float32 and stage.precision is None
    assert (stage.use_linear_attention, stage.use_projection, stage.use_self_and_cross,
            stage.only_pure_attention, stage.force_fp32_for_softmax,
            stage.norm_inputs, stage.explicitly_add_residual, stage.norm_epsilon) == \
        (True, False, True, True, False, True, True, 1e-4)

    model = Unet(emb_features=32, feature_depths=(8, 16), num_res_blocks=1, norm_groups=4,
                 precision="highest", attention_configs=(None, Stage(heads=2)))
    x = jnp.zeros((1, 16, 16, 3))
    params = model.init(jax.random.PRNGKey(0), x, jnp.ones((1,)), text(batch=1, features=64))
    assert jnp.all(jnp.isfinite(model.apply(params, x, jnp.ones((1,)),
                                            text(batch=1, features=64))))
