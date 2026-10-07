"""Benchmark cases and presets shared without importing either framework."""

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

# The CLIP-L/14 context's shape, from the library's table encoder: a benchmark
# of the model should not spend its first minute downloading a text tower, and
# the step cost depends only on the context's shape.
TEXT_TOKENS = 77
TEXT_FEATURES = 768

@dataclass(frozen=True)
class Case:
    """One measurement: what to build, how much to feed it, how to shard it."""

    architecture: str
    config: dict[str, object] = field(default_factory=dict)
    dtype: str | None = None
    """Compute dtype, written into the model config by the precision policy;
    None takes the run's --dtype."""
    matmul_precision: str | None = None
    """What every matmul asks XLA for (`ModelConfig.matmul_precision`), written
    into a model that declares `precision`; None keeps the model's own."""
    orders: bool = False
    """Layout comparisons use 52 exact residual orders against fp64 when True.
    Other measurements and rows retain their ordinary single-draw path."""
    batch_size: int = 8
    accumulation: int = 1
    """Microbatches of `batch_size` rows the trainer pools into one optimizer
    step; a timed step is one microbatch."""
    mesh: dict[str, int] = field(default_factory=dict)
    """`MeshSpec` fields, `{"fsdp": 2, "tensor": 2}`; the empty record is
    data parallelism over every device."""
    device_order: list[int] | None = None
    """The global device ids, in the order the mesh lays them out; None is
    `jax.devices()`'s. The innermost axis takes consecutive entries, so
    `[0, 2, 1, 3]` puts a size-2 inner axis across the pairs `[0, 1]` would
    keep together."""
    image_size: int = 32
    channels: int = 3
    """Input channels, including four-channel latent diffusion inputs."""
    frames: int = 0
    """Video models take (frames, H, W, C) samples; 0 means images."""
    predictor: dict[str, object] | None = None
    """Set for JEPA: the architecture is an encoder and this builds its predictor."""
    canvas: dict[str, object] | None = None
    """Set for a block-diffusion decoder: `prompt_length` and `canvas_size`.
    The architecture is the shared encoder/decoder trunk `config` builds, and
    the row of `seq_len + 1` tokens splits into a clean prompt and whole
    canvases the way recipes/lm/train.py's block objective splits it."""
    media: dict[str, object] | None = None
    """Set for a vision-conditioned decoder: the `tower` and `projector`
    records, the `family` and `image_token_id` the wrapper carries, `images`
    per row and the `pixels` shape the checkpoint's processor emits. The
    architecture is the decoder `config` builds and the images ride the batch
    beside the tokens, as they do on the loaded multimodal path."""
    seq_len: int = 0
    """Set for language models: batches are token windows of this length, not images."""
    packed_documents: int = 0
    """Set for language models: documents packed into every row, with the
    segment ids and positions the packed loader emits; 0 feeds a fixed
    window, the form a stream of tokens gives."""
    head_chunks: int | None = None
    """For language models, the vocabulary slices the loss scores a batch in;
    None is the objective's own default."""
    objective: dict[str, object] = field(default_factory=dict)
    """For language models, further keywords of the decoder's objective, such
    as LMObjective's balance terms `aux_loss_alpha` and `balance_rate`."""
    decoder_objective: Literal["lm", "sft", "dpo", "grpo", "mdlm"] = "lm"
    """For language models, what the decoder trains: next-token cross entropy
    ("lm"), the same over the assistant's turns only ("sft"), DPO over
    preference pairs ("dpo"), GRPO's clipped surrogate over rollouts
    ("grpo"), or masked diffusion's negative ELBO over a bidirectional
    decoder ("mdlm", MDLM's process)."""
    fsdp_min_param_size: int = 2 ** 16

    @property
    def is_jepa(self) -> bool:
        return self.predictor is not None

    @property
    def is_lm(self) -> bool:
        return self.seq_len > 0

    @property
    def packs_documents(self) -> bool:
        """Whether --packed-documents means anything for this case: a plain
        token window packs, a canvas row is the objective's own fixed
        geometry, a media row is what a processor emitted, and pairs,
        rollouts and a masked-diffusion row have their own columns."""
        return (self.is_lm and self.canvas is None and self.media is None
                and self.decoder_objective in ("lm", "sft"))

    @property
    def sample_shape(self) -> tuple[int, ...]:
        square = (self.image_size, self.image_size, self.channels)
        return square if self.frames == 0 else (self.frames, *square)

    @property
    def label(self) -> str:
        mixture = self.config.get("mixture")
        experts = f" x{mixture['experts']}experts" if isinstance(mixture, dict) else ""
        canvases = f" x{canvas_split(self)[2]}canvas" if self.canvas else ""
        images = f" x{images_per_row(self)}img" if self.media else ""
        order = "" if self.device_order is None else " order" + "".join(map(str, self.device_order))
        pooled = "" if self.accumulation == 1 else f" x{self.accumulation}acc"
        return (f"{self.architecture}{experts}{canvases}{images} b{self.batch_size}{pooled} "
                f"{mesh_label(self.mesh)}{order}")


def mesh_label(mesh: Mapping[str, int]) -> str:
    """`fsdp2-tensor2`, the mesh's sharded axes in MeshSpec order; `data` for none."""
    named = [f"{name}{mesh[name]}"
             for name in ("fsdp", "expert", "tensor", "sequence", "stage", "microbatches", "replicas")
             if mesh.get(name) not in (None, 1)]
    return "-".join(named) or "data"


def _count(record: Mapping[str, object], key: str, owner: str) -> int:
    value = record.get(key)
    if type(value) is not int or value < 1:
        raise ValueError(f"{owner} needs a positive integer {key}, got {value!r}")
    return value


def canvas_split(case: Case) -> tuple[int, int, int]:
    """A block-diffusion row as (prompt_length, canvas_size, num_canvases).

    The row holds `seq_len + 1` tokens, the width the token-windows loader
    emits, and splits the way recipes/lm/train.py splits it for the official
    fine-tuning objective: a clean prompt, then whole canvases.
    """
    canvas = case.canvas or {}
    prompt = _count(canvas, "prompt_length", "a canvas case")
    width = _count(canvas, "canvas_size", "a canvas case")
    response = case.seq_len + 1 - prompt
    if response < width or response % width:
        raise ValueError(
            f"{case.architecture}'s row of {case.seq_len + 1} tokens is not "
            f"{prompt} prompt tokens followed by whole canvases of {width}")
    return prompt, width, response // width


def images_per_row(case: Case) -> int:
    return _count(case.media or {}, "images", "a media case")


def media_pixels(case: Case) -> tuple[int, ...]:
    """One processed image's (channels, height, width), as a processor emits it."""
    shape = (case.media or {}).get("pixels")
    if (not isinstance(shape, (list, tuple)) or len(shape) != 3
            or any(type(size) is not int or size < 1 for size in shape)):
        raise ValueError(
            f"a media case's pixels is [channels, height, width], got {shape!r}")
    return tuple(shape)


def cpu_smoke_cases() -> list[Case]:
    """Tiny enough to run anywhere, real enough to compile the same step.

    The last two are the composites at a size a CPU compiles in seconds: the
    same wrappers, the same objectives and the same media and canvas work as
    --preset small, on a two-layer decoder.
    """
    tiny_decoder: dict[str, object] = {"vocab_size": 256, "emb_features": 32,
                                       "num_layers": 2, "num_heads": 2,
                                       "mlp_features": 64, "max_seq_len": 16}
    return [
        Case("simple_dit", {"patch_size": 4, "emb_features": 64, "num_layers": 2,
                            "num_heads": 2, "mlp_ratio": 2},
             batch_size=8, image_size=16, fsdp_min_param_size=256),
        Case("unet_2d_condition", {"stages": [{"features": 32, "heads": 2}, {"features": 64, "heads": 4}],
                                    "blocks_per_level": 1, "in_channels": 4, "out_channels": 4},
             batch_size=8, image_size=16, channels=4, fsdp_min_param_size=256),
        Case("sd3_transformer", {"in_channels": 4, "out_channels": 4, "num_layers": 2,
                                  "heads": 2, "head_dim": 8, "joint_attention_dim": 16,
                                  "caption_projection_dim": 16, "pooled_projection_dim": 32,
                                  "sample_size": 8, "pos_embed_max_size": 4},
             batch_size=8, image_size=8, channels=4, fsdp_min_param_size=256),
        Case("flux_transformer", {"in_channels": 16, "out_channels": 16,
                                   "num_layers": 1, "num_single_layers": 1, "heads": 2,
                                   "head_dim": 12, "joint_attention_dim": 16,
                                   "pooled_projection_dim": 8, "axes_dims_rope": (4, 4, 4),
                                   "guidance_embeds": True},
             batch_size=8, image_size=8, channels=4, fsdp_min_param_size=256),
        Case("jepa_encoder", {"patch_size": 4, "emb_features": 32, "num_layers": 2,
                              "num_heads": 2, "mlp_ratio": 2},
             predictor={"grid": (4, 4), "emb_features": 32, "predictor_features": 16,
                        "num_layers": 1, "num_heads": 2, "mlp_ratio": 2},
             batch_size=8, image_size=16, fsdp_min_param_size=256),
        Case("causal_transformer", tiny_decoder,
             batch_size=8, seq_len=16, fsdp_min_param_size=256),
        Case("multimodal_transformer", tiny_decoder,
             media={"family": "gemma3", "image_token_id": 255, "images": 1,
                    "pixels": [3, 16, 16],
                    "tower": {"class": "siglip", "fields": {"hidden_size": 32, "intermediate_size": 64,
                              "num_layers": 1, "num_heads": 2, "image_size": 16,
                              "patch_size": 8}},
                    "projector": {"class": "gemma", "fields": {"text_width": 32,
                                  "patches_per_side": 2, "tokens_per_side": 2}}},
             batch_size=8, seq_len=15, fsdp_min_param_size=256),
        Case("diffusion_gemma", {**tiny_decoder, "layer_scalar": "frozen"},
             canvas={"prompt_length": 8, "canvas_size": 4},
             batch_size=8, seq_len=15, fsdp_min_param_size=256),
    ]


def small_cases(dtype: str) -> list[Case]:
    """Every registry architecture at a size that fits one 16 GB card in bf16.

    Sized so the whole sweep takes minutes: real token counts (256 image
    tokens at 64px/patch 4) and real widths, but few layers.
    """
    dit: dict[str, object] = {"patch_size": 4, "emb_features": 384, "num_layers": 6,
                              "num_heads": 6, "mlp_ratio": 4}
    unet: dict[str, object] = {"emb_features": 256, "feature_depths": [64, 128, 256],
                               "attention_configs": [None, {"heads": 4}, {"heads": 4}],
                               "num_res_blocks": 2, "num_middle_res_blocks": 1}
    encoder: dict[str, object] = {"patch_size": 4, "emb_features": 384, "num_layers": 6,
                                  "num_heads": 6, "mlp_ratio": 4}
    predictor: dict[str, object] = {"grid": (16, 16), "emb_features": 384,
                                    "predictor_features": 192, "num_layers": 3,
                                    "num_heads": 6, "mlp_ratio": 4}
    # GPT-2 small's width and heads at a quarter of its depth, on 512-token
    # rows. The two composites wrap this same decoder, so their rows are the
    # plain decoder's rows plus the work their wrapper adds.
    decoder: dict[str, object] = {"vocab_size": 50304, "emb_features": 768, "num_layers": 3,
                                  "num_heads": 12, "mlp_features": 3072, "max_seq_len": 512}

    cases = [
        Case("unet", unet, batch_size=16, image_size=64),
        # EDM2's magnitude-preserving U-Net at the plain U-Net's widths.
        Case("edm2_unet", {"model_channels": 64, "channel_mult": [1, 2, 4], "num_blocks": 2,
                           "attn_resolutions": [16], "channels_per_head": 64},
             batch_size=16, image_size=64),
        Case("unet_2d_condition", {"stages": [{"features": 64, "heads": 4}, {"features": 128, "heads": 4},
                                              {"features": 256, "heads": 8, "cross_attention": False}],
                                    "blocks_per_level": 1, "in_channels": 4, "out_channels": 4},
             batch_size=4, image_size=32, channels=4),
        Case("sd3_transformer", {"num_layers": 6, "heads": 6, "head_dim": 64,
                                  "caption_projection_dim": 384, "sample_size": 32,
                                  "pos_embed_max_size": 16},
             batch_size=4, image_size=32, channels=16),
        Case("flux_transformer", {"num_layers": 3, "num_single_layers": 3, "heads": 6,
                                   "head_dim": 64, "axes_dims_rope": (16, 24, 24),
                                   "guidance_embeds": True},
             batch_size=4, image_size=32, channels=16),
        Case("qwen_image_transformer", {"in_channels": 16, "out_channels": 16, "num_layers": 3,
                                         "heads": 6, "head_dim": 64, "axes_dims_rope": (16, 24, 24)},
             batch_size=4, image_size=32, channels=16),
        # FLUX.2 reads three stacked encoder layers, so its context is three
        # text widths wide. Each rotary splits the 64 head channels.
        Case("flux2_transformer", {"num_layers": 3, "num_single_layers": 3, "heads": 6, "head_dim": 64,
                                    "joint_attention_dim": 3 * TEXT_FEATURES,
                                    "axes_dims_rope": (16, 16, 16, 16)},
             batch_size=4, image_size=32, channels=128),
        Case("z_image_transformer", {"dim": 384, "n_layers": 3, "n_refiner_layers": 1, "n_heads": 6,
                                      "cap_feat_dim": TEXT_FEATURES, "axes_dims": (16, 24, 24)},
             batch_size=4, image_size=32, channels=16),
        # Wan 2.1's text-to-video transformer at the 1.3B's head width and a
        # tenth of its depth, over 1 + 4k latent frames of its VAE's 16 channels.
        Case("wan_transformer", {"num_attention_heads": 6, "attention_head_dim": 64,
                                  "text_dim": TEXT_FEATURES, "ffn_dim": 1536, "num_layers": 3},
             batch_size=4, image_size=32, channels=16, frames=5),
        Case("uvit", dit,
             batch_size=16, image_size=64),
        Case("simple_udit", {**dit, "num_layers": 6}, batch_size=16, image_size=64),
        Case("simple_dit", dit, batch_size=16, image_size=64),
        Case("simple_mmdit", dit, batch_size=16, image_size=64),
        Case("hierarchical_mmdit",
             {"base_patch_size": 2, "emb_features": (192, 384, 576),
              "num_layers": (2, 2, 2), "num_heads": (3, 6, 9), "mlp_ratio": 4},
             batch_size=16, image_size=64),
        Case("hybrid_dit", {**dit, "ssm_state_dim": 64, "ssm_attention_ratio": "3:1"},
             batch_size=16, image_size=64),
        Case("video_dit", {**dit, "num_layers": 4}, batch_size=4, image_size=64, frames=8),
        Case("unet_3d", {**unet, "temporal_heads": 4},
             batch_size=4, image_size=64, frames=8),
        Case("jepa_encoder", encoder, predictor=predictor, batch_size=16, image_size=64),
        Case("jepa_video_encoder", {**encoder, "num_layers": 4},
             predictor={**predictor, "num_layers": 2, "factorized": True},
             batch_size=4, image_size=64, frames=8),
        Case("causal_transformer", decoder, batch_size=16, seq_len=512),
        # The same decoder with an 8-expert, top-2 feed-forward on every second
        # layer, which is the sparse shape the 4.7 acceptance run trains
        Case("causal_transformer",
             {**decoder, "mixture": {"experts": 8, "top_k": 2, "layers": (1,)}},
             batch_size=16, seq_len=512),
        # The same decoder as a vision-conditioned one: SigLIP-so400m's widths
        # at four layers over a 448px crop, pooled to the 256 soft tokens
        # Gemma 3 gives an image, so half of every 512-token row is media. One
        # image a row keeps the tower's cost the per-row cost a caption batch
        # pays, and the batch is halved because each row carries one. The trunk
        # is the plain decoder, so the row is the tower, the projector, the
        # fusion and a causal decoder; Gemma 3's bidirectional-over-image mask
        # is a per-family attention setting this case does not turn on.
        Case("multimodal_transformer", decoder,
             media={"family": "gemma3", "image_token_id": 50303, "images": 1,
                    "pixels": [3, 448, 448],
                    "tower": {"class": "siglip", "fields": {"hidden_size": 1152,
                              "intermediate_size": 4304, "num_layers": 4,
                              "num_heads": 16, "image_size": 448, "patch_size": 14}},
                    "projector": {"class": "gemma", "fields": {"text_width": 768,
                                  "patches_per_side": 32, "tokens_per_side": 16}}},
             batch_size=8, seq_len=512),
        # The same decoder read both ways by the official DiffusionGemma
        # fine-tuning loss: one encoder pass over the whole row into the
        # cache, then two decoder passes over the response for the
        # self-conditioning branch. `frozen` is the published scalar policy
        # the objective migrates to a trained one. The published vocabulary is
        # 262144 and this loss holds whole `[B, S, vocab]` logit tensors, so
        # the case keeps the other rows' vocabulary and a quarter of their
        # batch.
        Case("diffusion_gemma", {**decoder, "layer_scalar": "frozen"},
             canvas={"prompt_length": 256, "canvas_size": 128},
             batch_size=4, seq_len=511),
    ]
    return [dataclasses.replace(case, dtype=dtype) for case in cases]


