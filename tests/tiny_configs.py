"""Every registered architecture's tiny configuration."""

TINY = {"emb_features": 32}
DIT = {"output_channels": 3, "patch_size": 4, "num_layers": 1, "num_heads": 2}
UNET = {"output_channels": 3, "feature_depths": [8, 16],
        "attention_configs": [None, {"heads": 2}],
        "num_res_blocks": 1, "num_middle_res_blocks": 1, "norm_groups": 4}
LM = {"num_layers": 1, "num_heads": 2, "vocab_size": 32, "max_seq_len": 16}
PER_ARCH = {
    "unet": UNET,
    "unet_2d_condition": {"stages": [{"features": 32, "heads": 2}, {"features": 64, "heads": 4}],
                           "blocks_per_level": 1},
    "unet_3d": {**UNET, "temporal_heads": 2},
    "edm2_unet": {"output_channels": 3, "model_channels": 8, "channel_mult": [1, 2], "num_blocks": 1,
                  "attn_resolutions": [8], "channels_per_head": 8},
    "uvit": {**DIT, "num_layers": 2},
    "simple_udit": {**DIT, "num_layers": 2},
    "simple_dit": DIT,
    "simple_mmdit": DIT,
    "hybrid_dit": DIT,
    "video_dit": DIT,
    "hierarchical_mmdit": {"output_channels": 3, "emb_features": (16, 32),
                           "num_layers": (1, 1), "num_heads": (2, 2), "base_patch_size": 2},
    "jepa_encoder": {"patch_size": 4, "num_layers": 1, "num_heads": 2},
    "jepa_video_encoder": {"patch_size": 4, "num_layers": 1, "num_heads": 2},
    "jepa_predictor": {"num_layers": 1, "num_heads": 2, "grid": (4, 4), "predictor_features": 16},
    "causal_transformer": LM,
    # The two published transformer families, at one block each.
    "sd3_transformer": {"patch_size": 2, "in_channels": 4, "out_channels": 4, "num_layers": 1,
                        "heads": 2, "head_dim": 8, "joint_attention_dim": 12,
                        "caption_projection_dim": 16, "pooled_projection_dim": 10,
                        "sample_size": 8, "pos_embed_max_size": 8},
    "flux_transformer": {"patch_size": 1, "in_channels": 16, "out_channels": 16, "num_layers": 1,
                         "num_single_layers": 1, "heads": 2, "head_dim": 12,
                         "joint_attention_dim": 16, "pooled_projection_dim": 10,
                         "guidance_embeds": True, "axes_dims_rope": (4, 4, 4)},
    "qwen_image_transformer": {"in_channels": 4, "out_channels": 4, "num_layers": 1, "heads": 2,
                               "head_dim": 12, "context_in_dim": 16, "axes_dims_rope": (4, 4, 4)},
    "flux2_transformer": {"in_channels": 16, "out_channels": 16, "num_layers": 1, "num_single_layers": 1,
                          "heads": 2, "head_dim": 16, "joint_attention_dim": 16,
                          "timestep_guidance_channels": 32, "axes_dims_rope": (4, 4, 4, 4)},
    "z_image_transformer": {"in_channels": 4, "dim": 32, "n_layers": 1, "n_refiner_layers": 1, "n_heads": 2,
                            "cap_feat_dim": 16, "axes_dims": (4, 6, 6), "axes_lens": (64, 16, 16)},
    "wan_transformer": {"num_attention_heads": 2, "attention_head_dim": 12, "in_channels": 4,
                        "out_channels": 4, "text_dim": 16, "freq_dim": 16, "ffn_dim": 32,
                        "num_layers": 1, "rope_max_seq_len": 16},
}
COMPOSITES = ("diffusion_gemma", "multimodal_transformer")
RES, FRAMES = 16, 2
