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
        attention_configs=[None, {"heads": 2, "dtype": jnp.float32,
                                  "use_projection": False, "use_self_and_cross": False}],
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
        attention_configs=[None, {"heads": 2, "dtype": jnp.float32,
                                  "use_projection": False, "use_self_and_cross": False}],
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
        attention_configs=[{"heads": 2, "dtype": jnp.float32,
                            "use_projection": False, "use_self_and_cross": False}, None],
        num_res_blocks=1,
        num_middle_res_blocks=1,
    )
    temb = jnp.ones((2,))
    textcontext = text()

    image = jax.random.normal(rng, (2, 16, 16, 3))
    Unet(**config).init(rng, image, temb, textcontext)

    video = jax.random.normal(rng, (2, 3, 16, 16, 3))
    UNet3D(**config, temporal_heads=2).init(rng, video, temb, textcontext)
