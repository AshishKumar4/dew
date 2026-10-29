"""Independent counts of input maps, mixed sequences and visible architectural slots."""

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_model_flops import forward_products

from dew.batch import TEXT_KEY
from dew.diffusion import presets
from dew.diffusion.process import DenoisingCondition
from dew.distributed.batches import shard_batch
from dew.inputs import Condition, Field, InputSpec
from dew.inputs.condition import ConditionWork
from dew.inputs.diffusion import DiffusionConditioner, T5Segment
from dew.inputs.encoders import CharTable, T5Text
from dew.nn.attention_blocks import Stage
from dew.nn.attention_residuals import AttentionResiduals
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import BlockWiring, MTPBlock
from dew.nn.backbones.edm2 import EDM2UNet
from dew.nn.backbones.flux import FluxSingleBlock, FluxTransformer
from dew.nn.backbones.flux2 import Flux2SingleBlock, Flux2Transformer
from dew.nn.backbones.joint import DoubleStreamBlock
from dew.nn.backbones.mmdit import HierarchicalMMDiT, SimpleMMDiT
from dew.nn.backbones.qwen_image import QwenImageTransformer
from dew.nn.backbones.sd3 import SD3Transformer
from dew.nn.backbones.unet import Unet
from dew.nn.backbones.unet3d import UNet3D
from dew.nn.backbones.unet_condition import UNet2DCondition, UNetStage
from dew.nn.backbones.uvit import SimpleUDiT, UViT
from dew.nn.backbones.video_dit import VideoDiT
from dew.nn.backbones.wan import WanBlock, WanTransformer
from dew.nn.backbones.z_image import ZImageBlock, ZImageTransformer
from dew.nn.dit import TextContext
from dew.nn.flops import MissingFlopsCount, patch_lengths
from dew.nn.gemma3n import AltUp
from dew.nn.hyper_connections import HyperConnections
from dew.nn.mixer_base import MixerContext
from dew.nn.mixers.attention import CausalSelfAttention
from dew.nn.mlp import GatedMLP
from dew.nn.text_encoders import CLIPTextTransformer, CLIPVisionTransformer, T5EncoderTransformer
from dew.objectives.diffusion import DiffusionObjective
from dew.objectives.lm import LMObjective
from dew.training import MeshSpec
from dew.training.model_flops import TRAINING_FLOPS, prepare_counts, step_counts

IMAGE = np.array([2, 5], np.int64)
TEXT = np.array([3, 4], np.int64)


def test_joint_block_counts_each_streams_maps_and_the_joint_square():
    block = DoubleStreamBlock(8, 2, 4)
    joined = IMAGE + TEXT
    maps = (8 * 8 * 8 + 4 * 8 * 32) * joined.sum()
    attention = 4 * 8 * np.sum(joined**2)
    modulation = 2 * 8 * 12 * 8 * 2
    assert block.multisequence_flops(IMAGE, TEXT) == maps + attention + modulation
    last = block.clone(context_pre_only=True)
    omitted = (2 * 8 * 8 + 4 * 8 * 32) * TEXT.sum() + 2 * 8 * 4 * 8 * 2
    assert last.multisequence_flops(IMAGE, TEXT) == block.multisequence_flops(IMAGE, TEXT) - omitted
    extra = 8 * 8 * 8 * IMAGE.sum() + 4 * 8 * np.sum(IMAGE**2) + 2 * 8 * 3 * 8 * 2
    assert block.clone(dual_attention=True).multisequence_flops(IMAGE, TEXT) == (
        maps + attention + modulation + extra)


def test_single_flux_blocks_share_output_without_double_counting_it():
    lengths = IMAGE + TEXT
    first = (8 * 8 * 8 + 4 * 8 * 32) * lengths.sum() + 4 * 8 * np.sum(lengths**2)
    assert FluxSingleBlock(8, 2, 4).model_flops(lengths) == first + 2 * 8 * 3 * 8 * 2
    second = (8 * 8 * 8 + 6 * 8 * 24) * lengths.sum() + 4 * 8 * np.sum(lengths**2)
    assert Flux2SingleBlock(8, 2, 4, 24).model_flops(lengths) == second


@pytest.mark.parametrize("model", [
    FluxTransformer(in_channels=12, out_channels=12, heads=2, head_dim=4, num_layers=2,
                    num_single_layers=1, joint_attention_dim=16, pooled_projection_dim=12),
    Flux2Transformer(in_channels=4, out_channels=4, heads=2, head_dim=4, num_layers=2,
                     num_single_layers=1, joint_attention_dim=16),
    SD3Transformer(in_channels=4, out_channels=4, heads=2, head_dim=4, num_layers=2,
                   joint_attention_dim=16, caption_projection_dim=8, pooled_projection_dim=12),
    WanTransformer(in_channels=4, out_channels=4, num_attention_heads=2, attention_head_dim=4,
                   num_layers=2, text_dim=16, ffn_dim=24),
])
def test_joint_families_have_additive_rows_and_refuse_missing_text_geometry(model):
    with pytest.raises(MissingFlopsCount):
        model.model_flops(IMAGE)
    widths = {"input_channels": 1, "text_features": 6, "pooled_features": 5}
    complete = model.multisequence_flops(IMAGE, TEXT, **widths)
    split = sum(model.multisequence_flops(IMAGE[index:index + 1], TEXT[index:index + 1], **widths)
                for index in range(2))
    assert complete == split and complete > 0
    assert model.multisequence_flops(IMAGE, TEXT * 2, **widths) > complete


def test_flux2_shared_modulation_is_not_counted_for_every_layer():
    model = Flux2Transformer(heads=2, head_dim=4, num_layers=1, num_single_layers=0,
                             in_channels=4, out_channels=4, joint_attention_dim=16)
    block = DoubleStreamBlock(8, 2, 4, mlp_hidden=24, swiglu=True, shared_modulation=True)
    assert model.clone(num_layers=3).multisequence_flops(
        IMAGE, TEXT, input_channels=4, text_features=6) - model.multisequence_flops(
            IMAGE, TEXT, input_channels=4, text_features=6) == 2 * block.multisequence_flops(IMAGE, TEXT)


def test_wan_cross_attention_uses_video_times_text_not_the_joined_square():
    block = WanBlock(8, 2, 24)
    maps = 12 * 8 * 8 * IMAGE.sum() + 4 * 8 * 8 * TEXT.sum() + 4 * 8 * 24 * IMAGE.sum()
    products = 4 * 8 * np.sum(IMAGE**2 + IMAGE * TEXT)
    assert block.multisequence_flops(IMAGE, TEXT) == maps + products


def test_zimage_visible_pad_slots_count_in_blocks_but_not_patch_or_output_maps():
    model = ZImageTransformer(in_channels=4, dim=24, n_heads=3, n_layers=1,
                              n_refiner_layers=1, cap_feat_dim=16)
    difference = model.multimodal_flops(np.array([33]), np.array([3]), 4, 16) - model.multimodal_flops(
        np.array([32]), np.array([3]), 4, 16)
    block = ZImageBlock(24, 3)
    expected = (2 * 16 * 24 + 2 * 24 * 16
                + block.model_flops(np.array([64])) - block.model_flops(np.array([32]))
                + block.model_flops(np.array([96])) - block.model_flops(np.array([64])))
    assert difference == expected


def test_qwen_shared_time_zero_row_is_counted_once_for_the_batch():
    model = QwenImageTransformer(in_channels=4, out_channels=4, heads=2, head_dim=4,
                                 num_layers=1, context_in_dim=16)
    together = model.multisequence_flops(IMAGE, TEXT, input_channels=4, text_features=6)
    separate = sum(model.multisequence_flops(IMAGE[index:index + 1], TEXT[index:index + 1],
                                            input_channels=4, text_features=6)
                   for index in range(2))
    shared = 2 * 256 * 8 + 2 * 8 * 5 * 8
    assert separate - together == shared


@pytest.mark.mesh(devices=2)
def test_sharded_objective_counts_sum_to_one_qwen_batch_not_one_per_row():
    model = QwenImageTransformer(in_channels=4, out_channels=4, heads=2, head_dim=4,
                                 num_layers=1, context_in_dim=16)
    encoder = CharTable.from_pretrained(tokens=6, features=16)
    objective = DiffusionObjective(model, presets.Flow(),
        InputSpec(Field("images", (2, 3, 4)), conditions={"conditioning": Condition(encoder)}),
        autoencoder=None, ema_decay=None, unconditional_prob=0)
    batch = {"images": np.zeros((2, 2, 3, 4), np.float32), "text": encoder.tokenize(["a", "abcd"])}
    whole = objective.training_flops(None, batch)
    counted = prepare_counts(objective, batch, count_model=True)
    assert counted[TRAINING_FLOPS].sum() == whole
    mesh = MeshSpec(fsdp=2).build(jax.devices()[:2])
    placed = shard_batch(mesh, counted)
    assert float(jax.jit(step_counts)(placed)[TRAINING_FLOPS]) == whole
    per_shard = sum(np.asarray(shard.data).sum() for shard in placed[TRAINING_FLOPS].addressable_shards)
    assert per_shard == whole


def test_text_and_vision_towers_count_all_projection_maps():
    clip = CLIPTextTransformer(hidden_size=8, intermediate_size=16, num_layers=2, num_heads=2)
    assert clip.model_flops(TEXT) == 2 * ((8 * 8 * 8 + 4 * 8 * 16) * 7 + 2 * 8 * 25)
    t5 = T5EncoderTransformer(d_model=8, d_ff=16, num_layers=2, num_heads=2, head_dim=4)
    expected = 2 * ((8 * 8 * 8 + 4 * 8 * 16) * 7 + 4 * 8 * 25)
    assert t5.model_flops(TEXT) == expected
    assert t5.clone(feed_forward_proj="gated-gelu").model_flops(TEXT) == expected + 2 * 2 * 8 * 16 * 7
    vision = CLIPVisionTransformer(hidden_size=8, intermediate_size=16, num_layers=2, num_heads=2,
                                   patch_size=2, num_channels=3)
    expected = 2 * 12 * 8 * IMAGE.sum() + 2 * ((8 * 8 * 8 + 4 * 8 * 16) * 9 + 4 * 8 * 45)
    assert vision.model_flops(IMAGE) == expected


@pytest.mark.parametrize("fixture", ["gemma-3n-audio-tiny", "gemma-4-audio-tiny",
                                     "gemma4-unified-native-tiny"])
def test_a_media_wrapper_counts_its_decoder_and_refuses_an_audio_encoders_work(fixture):
    """A media wrapper's model FLOPs are its language model's over the real tokens. An audio encoder's
    work over its frames has no count, so a wrapper with one refuses, Gemma 3n's and Gemma 4's alike;
    Gemma 4 Unified's audio is a projector of raw frames, left out of the count as a vision projector's
    is, so it counts its language model's work."""
    from dew.interop import Pretrained

    model = Pretrained.load(Path(__file__).parent / "fixtures" / "hf" / fixture, weights=False,
                            dtype="float32").model
    lengths = np.array([5, 3], np.int64)
    if fixture == "gemma4-unified-native-tiny":
        assert model.model_flops(lengths) == model.language_model.model_flops(lengths) > 0
    else:
        with pytest.raises(MissingFlopsCount):
            model.model_flops(lengths)


def test_patch_lengths_retain_video_frames_and_reject_partial_patches():
    np.testing.assert_array_equal(patch_lengths((2, 4, 8, 3), (1, 2, 2), 2), [16, 16])
    with pytest.raises(MissingFlopsCount):
        patch_lengths((2, 3, 8, 3), (1, 2, 2), 2)


@pytest.mark.parametrize("family", ["unet", "unet3d", "edm2", "sd_unet", "uvit", "udit", "video",
                                    "mmdit", "hierarchical", "flux", "flux2", "sd3", "wan"])
def test_each_family_matches_every_traced_forward_product(family):
    """Unpadded noncausal fixtures: exact dot/conv counts, no elementwise tolerance."""
    image = jnp.ones((1, 4, 4, 3))
    time = jnp.ones((1,))
    text = TextContext(jnp.ones((1, 2, 8)), jnp.ones((1, 2), bool))
    condition = DenoisingCondition(text.hidden, pooled=jnp.ones((1, 8)))
    method_kwargs = {}
    if family in ("unet", "unet3d"):
        kind = UNet3D if family == "unet3d" else Unet
        model = kind(feature_depths=(8, 16), attention_configs=(None, None), num_res_blocks=1,
                     emb_features=8, norm_groups=1, attention_impl="reference")
        if family == "unet3d":
            model = model.clone(temporal_heads=2)
            image = jnp.ones((1, 2, 4, 4, 3))
        operands = (image, time)
        count = model.spatial_flops(image.shape[1:], 1)
    elif family == "edm2":
        model = EDM2UNet(model_channels=8, channel_mult=(1, 2), channels_per_head=4,
                         num_blocks=1, attn_resolutions=(2,), attention_impl="reference")
        operands = (image, time)
        count = model.spatial_flops(image.shape[1:], 1)
    elif family == "sd_unet":
        model = UNet2DCondition((UNetStage(8, 2), UNetStage(16, 2)), in_channels=3,
                                out_channels=3, blocks_per_level=1, norm_groups=1,
                                attention_impl="reference")
        operands, method_kwargs = (image, time), {"conditioning": condition}
        count = model.spatial_flops(image.shape[1:], 1, np.array([2]), 8)
    elif family in ("uvit", "udit"):
        kind = UViT if family == "uvit" else SimpleUDiT
        model = kind(emb_features=8, num_heads=2, num_layers=2, patch_size=2, attention_impl="reference")
        operands = (image, time)
        count = model.image_flops(np.array([4]), 3)
    elif family == "video":
        model = VideoDiT(emb_features=8, num_heads=2, num_layers=1, patch_size=2, attention_impl="reference")
        image = jnp.ones((1, 2, 4, 4, 3))
        operands = (image, time)
        count = model.spatiotemporal_flops(np.array([2]), np.array([4]), 3)
    elif family in ("mmdit", "hierarchical"):
        model = (SimpleMMDiT(emb_features=8, num_heads=2, num_layers=2, patch_size=2,
                             attention_impl="reference") if family == "mmdit" else
                 HierarchicalMMDiT(emb_features=(8, 16), num_heads=(2, 2), num_layers=(1, 1),
                                    base_patch_size=2, attention_impl="reference"))
        operands = (image, time, text)
        count = model.multimodal_flops(np.array([4]), np.array([2]), 3, 8)
    else:
        image, model, condition = diffusion_family(family)
        operands = (image, time, condition)
        pooled = None if condition.pooled is None else condition.pooled.shape[-1]
        count = model.multisequence_flops(model.image_lengths(image.shape[1:], 1), np.array([2]),
                                          input_channels=image.shape[-1],
                                          text_features=condition.context.shape[-1],
                                          pooled_features=pooled)
    variables = model.init(jax.random.key(310), *operands, **method_kwargs)
    def forward(variables, *operands):
        return model.apply(variables, *operands, **method_kwargs)
    assert count == forward_products(forward, variables, *operands), family


def diffusion_family(family):
    condition = DenoisingCondition(jnp.ones((1, 2, 8)), pooled=jnp.ones((1, 8)), guidance=jnp.ones((1,)))
    image = jnp.ones((1, 4, 4, 3))
    if family == "flux":
        model = FluxTransformer(in_channels=12, out_channels=12, heads=1, head_dim=8,
                                 num_layers=1, num_single_layers=1, joint_attention_dim=8,
                                 pooled_projection_dim=8, axes_dims_rope=(2, 2, 4),
                                 guidance_embeds=True, attention_impl="reference")
    elif family == "flux2":
        model = Flux2Transformer(in_channels=3, out_channels=3, heads=1, head_dim=8,
                                  num_layers=1, num_single_layers=1, joint_attention_dim=8,
                                  axes_dims_rope=(2, 2, 2, 2), attention_impl="reference")
    elif family == "sd3":
        model = SD3Transformer(in_channels=3, out_channels=3, heads=1, head_dim=8, num_layers=2,
                                joint_attention_dim=8, caption_projection_dim=8, pooled_projection_dim=8,
                                dual_attention_layers=(0,), attention_impl="reference")
    else:
        image = jnp.ones((1, 1, 4, 4, 3))
        model = WanTransformer(in_channels=3, out_channels=3, num_attention_heads=1, attention_head_dim=8,
                                num_layers=1, text_dim=8, ffn_dim=16, attention_impl="reference")
    return image, model, condition


@pytest.mark.parametrize("fields", [
    {"altup": AltUp(num_inputs=2), "laurel_rank": 2},
    {"altup": AltUp(num_inputs=3), "per_layer_input_dim": 4},
    {"hyper_connections": HyperConnections(hc_mult=2)},
    {"hyper_connections": HyperConnections(hc_mult=2, head="weighted")},
    {"hyper_connections": HyperConnections(hc_mult=2, head="carried", single_pass=True)},
    {"attention_residuals": AttentionResiduals(block_size=2)},
    {"attention_residuals": AttentionResiduals(block_size=1)},
    {"attention_residuals": AttentionResiduals(block_size=3)},
])
def test_multistream_projection_and_mixing_counts_match_the_forward(fields):
    model = CausalTransformer(vocab_size=16, emb_features=8, num_heads=2, num_kv_heads=2,
                              head_dim=4, num_layers=3, mlp_features=16, causal=False,
                              attention_impl="reference", **fields)
    tokens = jnp.ones((1, 4), jnp.int32)
    variables = model.init(jax.random.key(320), tokens)
    count = model.model_flops(np.array([4])) + model.logits_flops(np.array([4]))
    assert count == forward_products(model.apply, variables, tokens)


@pytest.mark.parametrize("family", ["qwen_image", "z_image"])
def test_published_single_stream_families_match_forward_products(family):
    image, time = jnp.ones((1, 4, 4, 3)), jnp.ones((1,))
    if family == "qwen_image":
        model = QwenImageTransformer(in_channels=3, out_channels=3, heads=1, head_dim=8,
                                     num_layers=2, context_in_dim=8, axes_dims_rope=(2, 2, 4),
                                     attention_impl="reference")
        text = 2
        # The reference dot executes the upper causal text half too; MFU excludes it.
        upper = model.num_layers * 2 * model.features * text**2
    else:
        model = ZImageTransformer(in_channels=3, dim=24, n_heads=3, n_layers=1, n_refiner_layers=1,
                                  cap_feat_dim=8, axes_dims=(2, 2, 4), axes_lens=(64, 16, 16),
                                  attention_impl="reference")
        # No masked caption padding: the rounded image slots are architectural inputs.
        text, upper = 32, 0
    condition = DenoisingCondition(jnp.ones((1, text, 8)), mask=jnp.ones((1, text), bool))
    variables = model.init(jax.random.key(321), image, time, condition)
    count = (model.multisequence_flops(model.image_lengths(image.shape[1:], 1), np.array([text]),
                                       input_channels=image.shape[-1],
                                       text_features=condition.context.shape[-1])
             if family == "qwen_image" else model.multimodal_flops(
                 model.image_lengths(image.shape[1:], 1), np.array([text]),
                 image.shape[-1], condition.context.shape[-1]))
    assert count == forward_products(model.apply, variables, image, time, condition) - upper


@pytest.mark.parametrize("fields", [{}, {"context_pre_only": True}, {"dual_attention": True},
                                    {"swiglu": True}, {"shared_modulation": True, "swiglu": True}])
def test_joint_block_variants_match_every_forward_product(fields):
    model = DoubleStreamBlock(8, 2, 4, mlp_hidden=16, attention_impl="reference", **fields)
    image, context = jnp.ones((1, 3, 8)), jnp.ones((1, 2, 8))
    modulation = (tuple(jnp.ones((1, 8)) for _ in range(6)),) * 2
    condition = modulation if fields.get("shared_modulation") else jnp.ones((1, 8))
    variables = model.init(jax.random.key(322), image, context, condition)
    count = model.multisequence_flops(np.array([3]), np.array([2]))
    assert count == forward_products(model.apply, variables, image, context, condition)


@pytest.mark.parametrize("family", ["clip_text", "clip_vision", "t5", "t5_gated"])
def test_condition_towers_match_forward_products(family):
    length, upper = 4, 0
    if family == "clip_text":
        model = CLIPTextTransformer(vocab_size=16, hidden_size=8, intermediate_size=16,
                                    num_layers=2, num_heads=2, max_position_embeddings=8)
        operand = jnp.ones((1, length), jnp.int32)
        upper = model.num_layers * 2 * model.hidden_size * length**2
    elif family == "clip_vision":
        model = CLIPVisionTransformer(hidden_size=8, intermediate_size=16, num_layers=2,
                                      num_heads=2, patch_size=2, image_size=4, num_channels=3)
        operand = jnp.ones((1, 3, 4, 4))
    else:
        model = T5EncoderTransformer(vocab_size=16, d_model=8, d_ff=16, num_layers=2,
                                     num_heads=2, head_dim=6,
                                     feed_forward_proj="gated-gelu" if family == "t5_gated" else "relu")
        operand = jnp.ones((1, length), jnp.int32)
    variables = model.init(jax.random.key(323), operand)
    assert model.model_flops(np.array([length])) == forward_products(model.apply, variables, operand) - upper


def test_diffusion_objective_counts_one_real_encoder_and_the_conditioned_denoiser():
    directory = Path(__file__).parent / "fixtures" / "t5" / "tiny"
    encoder = T5Text.from_pretrained(str(directory), max_length=4)
    model = SimpleUDiT(emb_features=8, num_heads=2, num_layers=2, patch_size=2, attention_impl="reference")
    objective = DiffusionObjective(model, presets.Flow(),
        InputSpec(Field("images", (4, 4, 3)), conditions={"textcontext": Condition(encoder)}),
        autoencoder=None, ema_decay=None)
    tokens = {"input_ids": np.ones((1, 3), np.int32), "attention_mask": np.ones((1, 3), np.int32)}
    image, time = jnp.ones((1, 4, 4, 3)), jnp.ones((1,))
    condition = encoder.encode(encoder.params, tokens)
    variables = model.init(jax.random.key(324), image, time, condition)

    def forward(variables, encoder_params, image, time, tokens):
        return model.apply(variables, image, time, encoder.encode(encoder_params, tokens))

    count = objective.model_flops(None, {"images": np.asarray(image), "text": tokens})
    assert count == forward_products(forward, variables, encoder.params, image, time, tokens)
    encoded = encoder.conditioning_flops(tokens)
    denoised = model.image_flops(np.array([4]), 3, encoded.features)
    training = objective.training_flops(None, {"images": np.asarray(image), "text": tokens})
    assert training == 3 * denoised + encoded.flops


def test_multiple_conditions_refuse_the_unknown_composition():
    encoder = CharTable.from_pretrained(tokens=4, features=8)
    model = SimpleUDiT(emb_features=8, num_heads=2, num_layers=2, patch_size=2)
    objective = DiffusionObjective(model, presets.Flow(),
        InputSpec(Field("images", (4, 4, 3)), conditions={"textcontext": Condition(encoder),
                                                       "other": Condition(encoder, field="other_text")}),
        autoencoder=None, ema_decay=None)
    with pytest.raises(MissingFlopsCount):
        objective.model_flops(None, {"images": np.ones((1, 4, 4, 3)), **objective.inputs.tokenize(["a"])})


def test_unet_counts_all_stage_attention_branches_and_repeated_middle_blocks():
    stage = Stage(heads=2, use_projection=True, only_pure_attention=False)
    model = Unet(feature_depths=(8, 16, 24), attention_configs=(stage, stage, stage),
                  num_res_blocks=1, num_middle_res_blocks=2, emb_features=8, norm_groups=1,
                  attention_impl="reference")
    image, time = jnp.ones((1, 8, 8, 3)), jnp.ones((1,))
    text = TextContext(jnp.ones((1, 3, 8)), jnp.ones((1, 3), bool))
    variables = model.init(jax.random.key(325), image, time, text)
    count = model.spatial_flops(image.shape[1:], 1, np.array([3]), 8)
    assert count == forward_products(model.apply, variables, image, time, text)


@pytest.mark.parametrize("inpaint", [False, True])
def test_conditional_unet_counts_odd_resizing_pooled_time_ids_and_cross_only_attention(inpaint):
    model = UNet2DCondition((UNetStage(8, 2, cross_only=True), UNetStage(16, 2)),
                            in_channels=7 if inpaint else 3, out_channels=3, blocks_per_level=1,
                            additional_time_features=2, linear_projection=True, norm_groups=1,
                            attention_impl="reference")
    image, time = jnp.ones((1, 5, 7, 3)), jnp.ones((1,))
    condition = DenoisingCondition(jnp.ones((1, 2, 8)), pooled=jnp.ones((1, 5)),
                                   time_ids=jnp.ones((1, 6)))
    extra = {"mask": jnp.ones((1, 5, 7, 1)), "masked_image": image} if inpaint else {}
    variables = model.init(jax.random.key(326), image, time, conditioning=condition, **extra)

    def forward(variables, image, time, condition):
        return model.apply(variables, image, time, conditioning=condition, **extra)

    count = model.conditioned_spatial_flops(image.shape[1:], 1,
        ConditionWork(0, np.array([2]), np.array([2]), 8, 5, 6))
    assert count == forward_products(forward, variables, image, time, condition)


@pytest.mark.parametrize("depths", [1, 2])
def test_prediction_depths_and_the_lm_objective_match_every_forward_product(depths):
    model = CausalTransformer(vocab_size=16, emb_features=8, num_heads=2, num_kv_heads=2,
                              head_dim=4, num_layers=2, mlp_features=16, causal=True,
                              num_nextn_predict_layers=depths, attention_impl="reference")
    tokens = jnp.ones((1, 6), jnp.int32)
    variables = model.init(jax.random.key(327), tokens)

    def forward(variables, tokens):
        hidden = model.apply(variables, tokens, method="hidden_states")
        logits = model.apply(variables, hidden, method="logits_from_hidden")
        return logits, model.apply(variables, hidden, tokens, method="mtp_logits")

    objective = LMObjective(model, seq_len=6, mtp_weight=1.0)
    batch = {TEXT_KEY: np.ones((1, 7), np.int32)}
    count = objective.model_flops(None, batch)
    # The reference dots include the causal upper halves that model work excludes.
    upper = 2 * model.num_heads * model.head_dim * (model.num_layers * 6**2
                + sum((6 - depth)**2 for depth in range(1, depths + 1)))
    assert count == forward_products(forward, variables, tokens) - upper
    assert objective.training_flops(None, batch) == 3 * count


def test_multistream_prediction_projects_each_carried_stream_and_collapses_its_output():
    context = MixerContext(emb_features=8, num_heads=2, num_kv_heads=2, head_dim=4, max_seq_len=8,
                           causal=False, attention_impl="reference")
    attention = partial(CausalSelfAttention, context, nope=True)
    model = MTPBlock(attention, partial(GatedMLP, hidden_features=16, out_features=8), 8, BlockWiring(),
                     hyper_connections=HyperConnections(hc_mult=2, head="weighted"))
    streams, embeds = jnp.ones((1, 4, 2, 8)), jnp.ones((1, 4, 8))
    variables = model.init(jax.random.key(328), streams, embeds)
    assert model.model_flops(np.array([4])) == forward_products(model.apply, variables, streams, embeds)


def test_uvit_counts_time_mlp_text_tokens_and_hilbert_skip_projections():
    model = UViT(emb_features=8, num_heads=2, num_layers=2, patch_size=2, mlp_time_embed=True,
                  scan_order="hilbert", conv=False, image_size=4, text_tokens=2, attention_impl="reference")
    image, time = jnp.ones((1, 4, 4, 3)), jnp.ones((1,))
    text = TextContext(jnp.ones((1, 2, 8)), jnp.ones((1, 2), bool))
    variables = model.init(jax.random.key(329), image, time, text)
    count = model.multimodal_flops(np.array([4]), np.array([2]), 3, 8)
    assert count == forward_products(model.apply, variables, image, time, text)


@pytest.mark.parametrize("family", ["unet", "unet3d", "uvit", "mmdit", "hierarchical", "flux2"])
def test_unmasked_denoisers_count_the_conditioners_physical_padded_span(family):
    encoder = CharTable.from_pretrained(tokens=4, features=8)
    tokens = encoder.tokenize(["a"])
    work = encoder.conditioning_flops(tokens)
    np.testing.assert_array_equal(work.lengths, [2])
    np.testing.assert_array_equal(work.spans, [4])
    image, time = jnp.ones((1, 4, 4, 3)), jnp.ones((1,))
    text = encoder.encode(encoder.params, tokens)
    if family in ("unet", "unet3d"):
        kind = UNet3D if family == "unet3d" else Unet
        model = kind(feature_depths=(8, 16), attention_configs=(Stage(heads=2), Stage(heads=2)),
                     emb_features=8, num_res_blocks=1, norm_groups=1, attention_impl="reference")
        if family == "unet3d":
            model = model.clone(temporal_heads=2)
            image = jnp.ones((1, 2, 4, 4, 3))
    elif family == "uvit":
        model = UViT(emb_features=8, num_heads=2, num_layers=2, patch_size=2,
                      image_size=4, text_tokens=4, attention_impl="reference")
    elif family == "mmdit":
        model = SimpleMMDiT(emb_features=8, num_heads=2, num_layers=1, patch_size=2,
                            attention_impl="reference")
    elif family == "hierarchical":
        model = HierarchicalMMDiT(emb_features=(8, 16), num_heads=(2, 2), num_layers=(1, 1),
                                  base_patch_size=2, attention_impl="reference")
    else:
        model = Flux2Transformer(in_channels=3, out_channels=3, heads=1, head_dim=8,
                                  num_layers=1, num_single_layers=1, joint_attention_dim=8,
                                  guidance_embeds=False, axes_dims_rope=(2, 2, 2, 2),
                                  attention_impl="reference")
        text = DenoisingCondition(text.hidden, mask=text.mask)
    keyword = "conditioning" if family == "flux2" else "textcontext"
    objective = DiffusionObjective(model, presets.Flow(),
        InputSpec(Field("images", image.shape[1:]), conditions={keyword: Condition(encoder)}),
        autoencoder=None, ema_decay=None, unconditional_prob=0)
    variables = model.init(jax.random.key(330), image, time, text)
    count = objective.model_flops(None, {"images": np.asarray(image), "text": tokens})
    assert count == forward_products(model.apply, variables, image, time, text)


@pytest.mark.parametrize("probability", [0, 0.12, 1])
def test_masked_condition_dropout_counts_blank_geometry_or_refuses_an_unknown_draw(probability):
    encoder = CharTable.from_pretrained(tokens=4, features=8)
    model = QwenImageTransformer(in_channels=3, out_channels=3, heads=1, head_dim=8,
                                 num_layers=1, context_in_dim=8, axes_dims_rope=(2, 2, 4),
                                 attention_impl="reference")
    objective = DiffusionObjective(model, presets.Flow(),
        InputSpec(Field("images", (2, 2, 3)), conditions={"conditioning": Condition(encoder)}),
        autoencoder=None, ema_decay=None, unconditional_prob=probability)
    batch = {"images": np.ones((1, 2, 2, 3)), "text": encoder.tokenize(["abcd"])}
    if probability == 0.12:
        with pytest.raises(MissingFlopsCount):
            objective.model_flops(None, batch)
    else:
        lengths = np.array([1 if probability == 1 else 4])
        assert objective.model_flops(None, batch) == model.multisequence_flops(
            np.array([4]), lengths, input_channels=3, text_features=8)


def test_zimage_counts_actual_image_and_caption_widths_not_unused_config_defaults():
    model = ZImageTransformer(in_channels=4, cap_feat_dim=16, dim=24, n_heads=3, n_layers=1,
                              n_refiner_layers=1, axes_dims=(2, 2, 4), axes_lens=(64, 16, 16),
                              attention_impl="reference")
    image, time = jnp.ones((1, 4, 4, 3)), jnp.ones((1,))
    text = DenoisingCondition(jnp.ones((1, 32, 8)), mask=jnp.ones((1, 32), bool))
    variables = model.init(jax.random.key(331), image, time, text)
    count = model.multimodal_flops(np.array([4]), np.array([32]), 3, 8)
    assert count == forward_products(model.apply, variables, image, time, text)


def test_zero_block_conditional_unet_counts_the_middle_width_change_and_decoder_skip():
    model = UNet2DCondition((UNetStage(8, 2), UNetStage(16, 2)), in_channels=3, out_channels=3,
                            blocks_per_level=0, norm_groups=1, attention_impl="reference")
    image, time = jnp.ones((1, 4, 4, 3)), jnp.ones((1,))
    text = DenoisingCondition(jnp.ones((1, 2, 8)))
    variables = model.init(jax.random.key(332), image, time, conditioning=text)

    def forward(variables, image, time, text):
        return model.apply(variables, image, time, conditioning=text)

    assert model.spatial_flops(image.shape[1:], 1, np.array([2]), 8) == forward_products(
        forward, variables, image, time, text)


@pytest.mark.parametrize("family", ["flux", "flux2", "sd3", "wan", "qwen_image"])
def test_joint_family_counts_use_actual_dense_inputs_when_config_defaults_disagree(family):
    if family == "qwen_image":
        image = jnp.ones((1, 4, 4, 3))
        model = QwenImageTransformer(in_channels=5, out_channels=3, heads=1, head_dim=8,
                                     context_in_dim=16, num_layers=1, axes_dims_rope=(2, 2, 4),
                                     attention_impl="reference")
        condition = DenoisingCondition(jnp.ones((1, 2, 8)), mask=jnp.ones((1, 2), bool))
        upper = model.num_layers * 2 * model.features * 2**2
    else:
        image, model, condition = diffusion_family(family)
        if family in ("flux", "sd3"):
            model = model.clone(in_channels=5, joint_attention_dim=16, pooled_projection_dim=12)
        elif family == "flux2":
            model = model.clone(in_channels=5, joint_attention_dim=16)
        else:
            model = model.clone(in_channels=5, text_dim=16)
        upper = 0
    time = jnp.ones((1,))
    variables = model.init(jax.random.key(333), image, time, condition)
    pooled = None if condition.pooled is None else condition.pooled.shape[-1]
    count = model.multisequence_flops(model.image_lengths(image.shape[1:], 1), np.array([2]),
                                      input_channels=image.shape[-1],
                                      text_features=condition.context.shape[-1],
                                      pooled_features=pooled)
    assert count == forward_products(model.apply, variables, image, time, condition) - upper


def test_flux_conditioner_pooled_width_is_unprojected_and_its_composition_matches_the_forward():
    clip = CLIPTextTransformer(vocab_size=16, hidden_size=8, intermediate_size=16,
                               num_layers=1, num_heads=2, max_position_embeddings=4)
    ids = np.ones((1, 2), np.int32)
    clip_variables = clip.init(jax.random.key(334), jnp.asarray(ids))
    directory = Path(__file__).parent / "fixtures" / "t5" / "tiny"
    t5 = T5Text.from_pretrained(str(directory), max_length=2)
    conditioner = DiffusionConditioner((clip,), (t5.tokenizer,), ("clip",),
        {"clip": {"text_model": clip_variables["params"]}, "t5": t5.params["params"]},
        "tiny-flux", 4, 4, t5.transformer.emb_features, composition="flux",
        t5=T5Segment(t5.transformer, t5.tokenizer, "t5", 2))
    tokens = {"input_ids": ids[:, None, :], "t5_input_ids": ids,
              "zero_condition": np.zeros((1,), bool)}
    assert "text_projection" not in conditioner.params["clip"]
    work = conditioner.conditioning_flops(tokens)
    condition = conditioner.encode(conditioner.params, tokens)
    assert work.pooled_features == condition.pooled.shape[-1] == clip.emb_features
    np.testing.assert_array_equal(work.spans, [2])
    model = FluxTransformer(in_channels=12, out_channels=12, heads=1, head_dim=8,
                             num_layers=1, num_single_layers=1, joint_attention_dim=16,
                             pooled_projection_dim=32, guidance_embeds=False,
                             axes_dims_rope=(2, 2, 4), attention_impl="reference")
    image, time = jnp.ones((1, 4, 4, 3)), jnp.ones((1,))
    variables = model.init(jax.random.key(335), image, time, condition)
    objective = DiffusionObjective(model, presets.Flow(),
        InputSpec(Field("images", image.shape[1:]), conditions={"conditioning": Condition(conditioner)}),
        autoencoder=None, ema_decay=None, unconditional_prob=0)

    def forward(variables, encoder_params, image, time, tokens):
        encoded = conditioner.encode(encoder_params, tokens)
        return model.apply(variables, image, time, encoded)

    count = objective.model_flops(None, {"images": np.asarray(image), "text": tokens})
    upper = clip.num_layers * 2 * clip.hidden_size * ids.shape[1]**2
    assert count == forward_products(forward, variables, conditioner.params, image, time, tokens) - upper
    native = model.multisequence_flops(np.array([4]), work.spans, input_channels=3,
                                       text_features=work.features, pooled_features=work.pooled_features)
    training = objective.training_flops(None, {"images": np.asarray(image), "text": tokens})
    assert training == 3 * native + work.flops
