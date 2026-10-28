"""Every architecture at the smallest size that still exercises it (`CASES`, tests/test_architectures.py's
cases), as plain data: a test that reads a case's configuration imports no part of Dew."""

from dataclasses import dataclass

RES = 16
FRAMES = 2
PATCH = 4
GRID = (RES // PATCH, RES // PATCH)


@dataclass(frozen=True)
class Case:
    """One architecture at the smallest size that still exercises it."""

    architecture: str
    config: dict
    frames: int = 0
    """Video architectures take (frames, H, W, C) samples; 0 means images."""
    predictor: dict | None = None
    """Set for JEPA: `architecture` is the encoder and this builds its predictor."""
    seq_len: int = 0
    """Set for language models: batches are token windows, not images."""
    label: str = ""
    """A second case of one architecture, named apart from the first."""

    @property
    def is_jepa(self) -> bool:
        return self.predictor is not None

    @property
    def is_lm(self) -> bool:
        return self.seq_len > 0

    @property
    def sample_shape(self) -> tuple:
        square = (RES, RES, 3)
        return square if self.frames == 0 else (self.frames, *square)

    @property
    def sample_key(self) -> str:
        return "video" if self.frames else "image"

    @property
    def name(self) -> str:
        return self.architecture + (f"+{self.label}" if self.label else "")
DIT = {"patch_size": PATCH, "emb_features": 64, "num_layers": 2, "num_heads": 2,
       "mlp_ratio": 2}
UNET = {"emb_features": 64, "feature_depths": [16, 32],
        "attention_configs": [None, {"heads": 2, "use_projection": False,
                                     "use_self_and_cross": False}],
        "num_res_blocks": 1, "num_middle_res_blocks": 1}
ENCODER = {"patch_size": PATCH, "emb_features": 32, "num_layers": 2, "num_heads": 2,
           "mlp_ratio": 2}
PREDICTOR = {"grid": GRID, "emb_features": 32, "predictor_features": 16,
             "num_layers": 1, "num_heads": 2, "mlp_ratio": 2}
VOCAB = 64
SEQ_LEN = 16
LONG_SEQ_LEN = 128
"""The rows of a cell whose float32 error is heavy-tailed over draws, a few draws holding most of it
(Gemma 3n's and Qwen 3.5's): longer rows put more such events in each draw,
so check 1's pooled ratio resolves over a sized count of draws, where SEQ_LEN's rows left the
reference compared with itself unresolved within 512. Qwen 3.5's rows are twice as long: at 128 its
embedding's momentum still did not resolve within 512. Gemma 3n's draws also clear the sparsity
cutoff by a margin (`selection_margin`)."""
LM = {"vocab_size": VOCAB, "emb_features": 32, "num_layers": 2, "num_heads": 2,
      "num_kv_heads": 1, "mlp_features": 64, "max_seq_len": SEQ_LEN}
CASES = [
    Case("unet", UNET),
    # Both U-shaped stacks split their layers into a down and an up half
    Case("uvit", {"patch_size": PATCH, "emb_features": 64, "num_layers": 4,
                  "num_heads": 2}),
    Case("simple_udit", {**DIT, "num_layers": 2}),
    Case("simple_dit", DIT),
    Case("simple_mmdit", DIT),
    Case("hierarchical_mmdit", {"base_patch_size": 2, "emb_features": (32, 64, 96),
                                "num_layers": (1, 1, 1), "num_heads": (2, 2, 2),
                                "mlp_ratio": 2}),
    Case("hybrid_dit", {**DIT, "num_layers": 4, "ssm_state_dim": 8,
                        "ssm_attention_ratio": "3:1"}),
    Case("video_dit", {**DIT, "num_layers": 1}, frames=FRAMES),
    Case("unet_3d", {**UNET, "attention_configs": [None, None], "temporal_heads": 2},
         frames=FRAMES),
    Case("jepa_encoder", ENCODER, predictor=PREDICTOR),
    Case("jepa_video_encoder", {**ENCODER, "num_layers": 1},
         predictor={**PREDICTOR, "factorized": True}, frames=FRAMES),
    Case("causal_transformer", LM, seq_len=SEQ_LEN),
    # Eight experts so the expert dimension divides every mesh this file
    # builds, on the second layer only, so one dense and one sparse
    # feed-forward go through the same run.
    Case("causal_transformer", {**LM, "mixture": {"experts": 8, "top_k": 2, "layers": (1,)}},
         seq_len=SEQ_LEN, label="moe"),
    # GPT OSS at toy width: sink attention on both kinds, the fused biased
    # experts on every layer, and YaRN over grouped-query heads.
    Case("causal_transformer", {
        **LM, "mlp": "swigluoai", "attention_sinks": True, "qk_norm": False,
        "layer_types": ("sliding_attention", "full_attention"),
        "kinds": {"sliding_attention": {"window": 4}},
        "yarn": {"factor": 4.0, "original_max_position_embeddings": 8, "truncate": False},
        "mixture": {"experts": 8, "top_k": 2},
    }, seq_len=SEQ_LEN, label="gpt_oss"),
    # DeepSeek V3.2's stack at toy width: the mla mixer with its sparse
    # indexer on every layer, and the routed layer with the balancing bias,
    # the group limit and a shared expert. The low ranks are 32 wide, so the
    # latent norms' scales, whose rounding over a draw varies most, hold
    # enough values a draw for the layout cell to resolve.
    Case("causal_transformer", {
        **LM, "head_dim": 16,
        "mixer": {"class": "mla", "fields": {"q_lora_rank": 32, "kv_lora_rank": 32,
                  "qk_nope_head_dim": 8, "qk_rope_head_dim": 8, "v_head_dim": 8,
                  "index_topk": 4, "index_n_heads": 2, "index_head_dim": 16}},
        "mixture": {"experts": 8, "top_k": 2, "layers": (1,), "score_function": "sigmoid",
                    "groups": 4, "groups_per_token": 2, "bias": True,
                    "expert_features": 16, "shared_features": 16},
    }, seq_len=SEQ_LEN, label="mla"),
    # DeepSeek V2 Lite's stack at toy width: the mla mixer without the query
    # LoRA, and a routed layer under softmax group-limited routing scored by
    # the best expert, unnormalised, with a shared expert and no bias.
    Case("causal_transformer", {
        **LM, "head_dim": 16,
        "mixer": {"class": "mla", "fields": {"kv_lora_rank": 8, "qk_nope_head_dim": 8,
                  "qk_rope_head_dim": 8, "v_head_dim": 8}},
        "mixture": {"experts": 8, "top_k": 4, "layers": (1,), "score_function": "softmax",
                    "norm_topk_prob": False, "groups": 4, "groups_per_token": 2,
                    "group_score": "max", "expert_features": 16, "shared_features": 32},
    }, seq_len=SEQ_LEN, label="deepseek_v2"),
    # GLM 4.5's stack at toy width: biased q/k/v over a bias-free o_proj, a
    # half rotary, DeepSeek V3 routing with a shared expert on the last
    # layer, and one MTP depth that routes like it.
    Case("causal_transformer", {
        **LM, "attention_bias": True, "o_proj_bias": False, "qk_norm": False,
        "head_dim": 8, "partial_rotary_factor": 0.5, "partial_rotary_type": "default",
        "mixture": {"experts": 8, "top_k": 2, "layers": (1,), "score_function": "sigmoid",
                    "bias": True, "expert_features": 16, "shared_features": 16},
        "num_nextn_predict_layers": 1,
    }, seq_len=SEQ_LEN, label="glm4_moe"),
    # Llama 4's stack at toy width: chunked rotated layers, adjacent pairs under the weightless L2
    # norm, around a global layer without positions tuning its temperature; the routed layers scale
    # their inputs.
    Case("causal_transformer", {
        **LM, "head_dim": 8, "qk_norm": False,
        "layer_types": ("chunked_attention", "full_attention"),
        "kinds": {"chunked_attention": {"chunk": 4, "qk_norm": True, "mixer": {
                      "class": "attention", "fields": {"rotary_pairs": "adjacent", "qk_norm_weight": False}}},
                  "full_attention": {"mixer": {"class": "attention", "fields": {
                      "nope": True, "temperature_tuning": (4.0, 0.1)}}}},
        "mixture": {"experts": 8, "top_k": 1, "layers": (1,), "score_function": "sigmoid",
                    "norm_topk_prob": False, "scale_inputs": True, "expert_features": 16,
                    "shared_features": 16},
    }, seq_len=SEQ_LEN, label="llama4"),
    # Gemma 4's 26B-A4B at toy width: the routed branch beside every layer's
    # dense MLP, the global layer reading its values off the keys with its
    # own head dim and key/value count, and the per-layer output scalar.
    Case("causal_transformer", {
        **LM, "num_heads": 4, "num_kv_heads": 2, "head_dim": 8, "mlp": "geglu",
        "sandwich_norms": True, "qk_norm": True, "v_norm": True,
        "embedding_scale": True, "attention_scale": 1.0, "partial_rotary_factor": 0.25,
        "layer_types": ("sliding_attention", "full_attention"),
        "kinds": {"sliding_attention": {"window": 4},
                  "full_attention": {"head_dim": 16, "num_kv_heads": 1}},
        "attention_k_eq_v": True, "layer_scalar": "frozen",
        "mixture": {"experts": 8, "top_k": 2, "expert_features": 16, "parallel": True},
    }, seq_len=SEQ_LEN, label="gemma4_moe"),
    # Gemma 3n at toy width: three copies of the residual stream under
    # AltUp, the LAuReL block, gaussian top-k on the first layer, widths
    # that differ per layer, per-layer inputs and the last layer sharing the
    # first's keys and values. The first layer's 192 units keep about ten a
    # token at 0.95 sparsity, where the released models keep hundreds: at 48,
    # about two, a token whose kept units all sit just above the cutoff gives
    # the feed-forward's post-norm a near-zero input, whose direction float32
    # resolves poorly, and four draws in 320 held 60% of a layer-0 leaf's
    # pooled error (at 192, at most 22%; at 384, 6%).
    Case("causal_transformer", {
        **LM, "num_heads": 4, "num_kv_heads": 2, "head_dim": 8, "mlp": "geglu",
        "mlp_features": (192, 64), "activation_sparsity_pattern": (0.95, 0.0),
        "sandwich_norms": True, "qk_norm": True, "v_norm": True,
        "embedding_scale": True, "attention_scale": 1.0, "final_logit_softcap": 30.0,
        "layer_types": ("sliding_attention", "sliding_attention"),
        "kinds": {"sliding_attention": {"window": 4, "rope_theta": 1e4}},
        "altup": {"num_inputs": 3}, "laurel_rank": 8,
        "per_layer_input_dim": 8, "kv_shared_layers": (1,), "max_seq_len": LONG_SEQ_LEN,
    }, seq_len=LONG_SEQ_LEN, label="gemma3n"),
    # Qwen3.5's stack: gated delta net layers on the linear_attention kind,
    # one gated full-attention layer with the sliced partial rotary. The
    # delta net's projections are wide enough to cross the shard threshold,
    # so its declarations and its heuristic conv taps are what the layout
    # places here.
    Case("causal_transformer", {
        **LM, "num_layers": 4, "head_dim": 8, "output_gate": True,
        "partial_rotary_factor": 0.5, "partial_rotary_type": "default",
        "layer_types": ("linear_attention",) * 3 + ("full_attention",),
        "kinds": {"linear_attention": {"mixer": {
            "class": "gated_delta_net", "fields": {"linear_num_key_heads": 2,
            "linear_num_value_heads": 4, "linear_key_head_dim": 8,
            "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4}}}},
        "max_seq_len": 2 * LONG_SEQ_LEN}, seq_len=2 * LONG_SEQ_LEN, label="qwen35"),
]
IDS = [case.name for case in CASES]

TEXT_TOKENS = 8
TEXT_FEATURES = 32
TEXT_VOCAB = 16
