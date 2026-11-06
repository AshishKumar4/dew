"""The tiny FlaxDiff models tools/flaxdiff_family_worker.py runs and tests/test_flaxdiff_family_source.py
holds Dew's to, two per model, each `case: (module, class, config, image [(T,) H, W, C], text [tokens,
width])`, data its worker and its test both read:

- `published`: the constructor's defaults, narrowed (raster scan, no qk-norm, MLP ratio 4; the Unet with
  attention at every level);
- `variant`: the other paths: a zigzag or Hilbert scan, qk-norm, MLP ratio 2, odd head counts, a
  rectangular image, three hierarchy stages, and UNets with a level without attention, RMS norms, and
  stages that project in and out and run the full self, cross and feed-forward block (whose middle stage
  FlaxDiff projects with 1x1 convolutions).

The video models take three-frame clips in `published` and two in `variant`. Attention runs FlaxDiff's
reference path (`attention_impl=None`) with `force_fp32_for_softmax=False`: flax's forced softmax casts to
float32 whenever the dtype is not, which would round the float64 walk, and in float32 it changes nothing.
"""

ATTENTION = {"attention_impl": None, "force_fp32_for_softmax": False}
# case: (module, class, config, image [(T,) H, W, C], text [tokens, width])
CASES = {
    "simple_dit/published": ("simple_dit", "SimpleDiT", {
        "output_channels": 4, "patch_size": 2, "emb_features": 16, "num_layers": 2, "num_heads": 2,
        "mlp_ratio": 4, **ATTENTION}, (8, 8, 4), (5, 12)),
    "simple_dit/variant": ("simple_dit", "SimpleDiT", {
        "output_channels": 3, "patch_size": 2, "emb_features": 12, "num_layers": 2, "num_heads": 3,
        "mlp_ratio": 2, "qk_norm": True, "use_zigzag": True, **ATTENTION}, (8, 12, 3), (4, 7)),
    "simple_udit/published": ("simple_vit", "SimpleUDiT", {
        "output_channels": 4, "patch_size": 2, "emb_features": 16, "num_layers": 4, "num_heads": 2,
        "mlp_ratio": 4, **ATTENTION}, (8, 8, 4), (5, 12)),
    "simple_udit/variant": ("simple_vit", "SimpleUDiT", {
        "output_channels": 3, "patch_size": 2, "emb_features": 12, "num_layers": 2, "num_heads": 3,
        "mlp_ratio": 2, "use_hilbert": True, **ATTENTION}, (8, 8, 3), (4, 7)),
    "simple_mmdit/published": ("simple_mmdit", "SimpleMMDiT", {
        "output_channels": 4, "patch_size": 2, "emb_features": 16, "num_layers": 2, "num_heads": 2,
        "mlp_ratio": 4, **ATTENTION}, (8, 8, 4), (5, 12)),
    "simple_mmdit/variant": ("simple_mmdit", "SimpleMMDiT", {
        "output_channels": 3, "patch_size": 2, "emb_features": 12, "num_layers": 2, "num_heads": 3,
        "mlp_ratio": 2, "qk_norm": True, "use_zigzag": True, **ATTENTION}, (8, 12, 3), (4, 7)),
    "hierarchical_mmdit/published": ("simple_mmdit", "HierarchicalMMDiT", {
        "output_channels": 4, "base_patch_size": 2, "emb_features": (12, 16), "num_layers": (1, 2),
        "num_heads": (2, 2), "mlp_ratio": 4, **ATTENTION}, (8, 8, 4), (5, 12)),
    "hierarchical_mmdit/variant": ("simple_mmdit", "HierarchicalMMDiT", {
        "output_channels": 3, "base_patch_size": 1, "emb_features": (8, 12, 16), "num_layers": (1, 1, 1),
        "num_heads": (2, 3, 4), "mlp_ratio": 2, "qk_norm": True, **ATTENTION}, (8, 12, 3), (4, 7)),
    # FlaxDiff's decoder upsamples level i to feature_depths[-i], Dew's to
    # feature_depths[-i - 2]: the two agree for two levels, or equal depths.
    "unet/published": ("simple_unet", "Unet", {
        "output_channels": 4, "emb_features": 16, "feature_depths": (8, 16),
        "attention_configs": ({"heads": 2}, {"heads": 2}), "num_res_blocks": 2, "num_middle_res_blocks": 1,
        "norm_groups": 4, "attention_impl": None}, (8, 8, 4), (5, 12)),
    "unet/variant": ("simple_unet", "Unet", {
        "output_channels": 3, "emb_features": 12, "feature_depths": (8, 8, 8),
        "attention_configs": ({"heads": 2, "use_projection": True, "only_pure_attention": False}, None,
                              {"heads": 4, "use_projection": True, "only_pure_attention": False}),
        "num_res_blocks": 1, "num_middle_res_blocks": 2, "norm_groups": 0, "attention_impl": None},
        (8, 8, 3), (4, 7)),
    "video_dit/published": ("video_dit", "VideoDiT", {
        "output_channels": 4, "patch_size": 2, "emb_features": 16, "num_layers": 1, "num_heads": 2,
        "mlp_ratio": 4, **ATTENTION}, (3, 8, 8, 4), (5, 12)),
    "video_dit/variant": ("video_dit", "VideoDiT", {
        "output_channels": 3, "patch_size": 2, "emb_features": 12, "num_layers": 2, "num_heads": 3,
        "mlp_ratio": 2, "qk_norm": True, "use_zigzag": True, **ATTENTION}, (2, 8, 12, 3), (4, 7)),
    "unet_3d/published": ("unet_3d", "UNet3D", {
        "output_channels": 4, "emb_features": 16, "feature_depths": (8, 16),
        "attention_configs": ({"heads": 2}, {"heads": 2}), "num_res_blocks": 2, "num_middle_res_blocks": 1,
        "norm_groups": 4, "temporal_heads": 2, "attention_impl": None}, (3, 8, 8, 4), (5, 12)),
    "unet_3d/variant": ("unet_3d", "UNet3D", {
        "output_channels": 3, "emb_features": 12, "feature_depths": (8, 8, 8),
        "attention_configs": ({"heads": 2, "use_projection": True, "only_pure_attention": False}, None,
                              {"heads": 4, "use_projection": True, "only_pure_attention": False}),
        "num_res_blocks": 1, "num_middle_res_blocks": 2, "norm_groups": 0, "temporal_heads": 4,
        "attention_impl": None}, (2, 8, 8, 3), (4, 7)),
}
