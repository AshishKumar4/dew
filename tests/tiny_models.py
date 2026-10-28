"""Every registered architecture, and the decoder's mixers past its default, at a tiny size with an
explicit compute dtype (`build_model`), with inputs for each (`tiny_inputs`): the catalogue the
precision, capability and protocol tests build their models from."""

import json
from importlib import import_module
from pathlib import Path

import jax
import jax.numpy as jnp
from architecture_cases import CASES as ARCHITECTURE_CASES
from tiny_configs import COMPOSITES, FRAMES, LM, PER_ARCH, RES, TINY

from dew.diffusion.process import DenoisingCondition
from dew.interop.hf_decoders import translate_config
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.dit import TextContext
from dew.nn.multimodal import MultimodalTransformer
from dew.nn.vision import GemmaProjector, SiglipVision
from dew.registry import models, resolve_dtype

import_module("dew.nn.multimodal")  # registers the fixture kind


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"
FIXTURE_DECODERS = ("mamba-tiny", "mamba2-tiny", "deepseek-v4-tiny", "glm5-next-tiny", "kimi-k3-tiny")
"""Tiny released configs whose mixers no toy case reaches: Mamba, Mamba-2,
DeepSeek V4's compressed attention and hyper-connections, GLM-5 Next's
k-pool sparse attention and Kimi's delta attention, depth attention and
SiTU. DeepSeek V4.1's KV quantizers round through float32 bits on
purpose (`dew.nn.fake_quant`), so its float64 twin is
tools/deepseek_v41_numerics.py's, not this one."""


def _translated(name):
    """A fixture's decoder config, leaving compute settings to the caller."""
    config = translate_config(json.loads((FIXTURES / name / "config.json").read_text()))
    return {key: value for key, value in config.items() if key not in ("dtype", "attention_impl")}


DECODERS = ({case.name: case.config for case in ARCHITECTURE_CASES
             if case.architecture == "causal_transformer" and case.label}
            | {f"causal_transformer+{name}": _translated(name) for name in FIXTURE_DECODERS})
"""The decoder's mixers and mixtures past its default, at toy width
(tests/test_architectures.py): the MoE routers, Gemma 4's, GPT OSS's, MLA,
Llama 4's, gemma3n's and the gated delta net, by case name; and the
fixtures' (`FIXTURE_DECODERS`)."""


def build_model(architecture, dtype="bfloat16"):
    """Every registered architecture at a tiny size with an explicit compute dtype.

    Float64 references use an actual dtype (not a recorded run dtype name),
    under x64. Composite trunks receive the same dtype as leaf models.
    """
    def resolved(name, fields):
        return {**fields, "dtype": jnp.float64 if dtype == "float64" else dtype}

    if architecture in DECODERS:
        return models.build("causal_transformer", **resolved("causal_transformer", DECODERS[architecture]))
    if architecture not in COMPOSITES:
        own = (
            "unet_2d_condition",
            "sd3_transformer",
            "flux_transformer",
            "qwen_image_transformer",
            "edm2_unet",
            "flux2_transformer",
            "z_image_transformer",
            "wan_transformer",
        )
        fields = PER_ARCH[architecture] if architecture in own else {**TINY, **PER_ARCH[architecture]}
        return models.build(architecture, **resolved(architecture, fields))
    text = models.build("causal_transformer", **resolved(
        "causal_transformer", {**TINY, **LM, "mlp_features": 64}))
    if architecture == "diffusion_gemma":
        return DiffusionGemma(text, canvas_length=4)
    return MultimodalTransformer(
        text, SiglipVision(hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2,
                           image_size=8, patch_size=4),
        GemmaProjector(text_width=TINY["emb_features"],
                       patches_per_side=2, tokens_per_side=1),
        family="gemma3", image_token_id=1,
        dtype=jnp.float64 if dtype == "float64" else resolve_dtype(dtype))


def tiny_inputs(architecture, rng):
    """What each architecture's __call__ takes, at the smallest useful size."""
    image = jax.random.normal(rng, (1, RES, RES, 3))
    video = jax.random.normal(rng, (1, FRAMES, RES, RES, 3))
    text = TextContext(jnp.ones((1, 7, 768)), jnp.ones((1, 7), bool))
    if architecture in ("unet_3d", "video_dit"):
        return video, jnp.ones((1,)), text
    if architecture == "unet_2d_condition":
        latents = jax.random.normal(rng, (1, RES, RES, 4))
        return (latents, jnp.ones((1,))), {"conditioning": DenoisingCondition(text.hidden)}
    if architecture == "sd3_transformer":
        latents = jax.random.normal(rng, (1, 8, 8, 4))
        return (latents, jnp.ones((1,))), {"conditioning": DenoisingCondition(
            text.hidden[:, :, :12], jnp.ones((1, 10)))}
    if architecture == "flux_transformer":
        latents = jax.random.normal(rng, (1, 8, 8, 4))
        return (latents, jnp.ones((1,))), {"conditioning": DenoisingCondition(
            text.hidden[:, :, :16], jnp.ones((1, 10)), guidance=jnp.full((1,), 3.5))}
    if architecture == "flux2_transformer":
        latents = jax.random.normal(rng, (1, 4, 4, 16))
        return (latents, jnp.ones((1,))), {"conditioning": DenoisingCondition(
            text.hidden[:, :, :16], guidance=jnp.full((1,), 3.5))}
    if architecture in ("z_image_transformer", "qwen_image_transformer"):
        latents = jax.random.normal(rng, (1, 4, 4, 4))
        return (latents, jnp.ones((1,))), {"conditioning": DenoisingCondition(
            text.hidden[:, :, :16], mask=text.mask)}
    if architecture == "wan_transformer":
        # 32 patch tokens: the source's time embedder is fp32 on purpose, and
        # fewer tokens would leave it a toy-sized share of the matmuls.
        latents = jax.random.normal(rng, (1, FRAMES, 8, 8, 4))
        return (latents, jnp.ones((1,))), {"conditioning": DenoisingCondition(text.hidden[:, :, :16])}
    if architecture == "jepa_encoder":
        return (image,)
    if architecture == "causal_transformer" or architecture in DECODERS:
        return (jnp.zeros((1, 8), jnp.int32),)
    if architecture == "diffusion_gemma":
        return (jnp.zeros((1, 4), jnp.int32),)
    if architecture == "multimodal_transformer":
        tokens = jnp.array([[2, 1, 3, 4, 5, 6, 7, 2]], jnp.int32)
        return (tokens,), {"image_indices": jnp.where(tokens == 1, 0, -1),
                           "conditioning": {"pixel_values": jax.random.normal(rng, (1, 1, 3, 8, 8))}}
    if architecture == "jepa_video_encoder":
        return (video,)
    if architecture == "jepa_predictor":
        return (jax.random.normal(rng, (1, 8, 32)),
                jnp.arange(8)[None], jnp.arange(8, 12)[None])
    return image, jnp.ones((1,)), text


def forward(architecture, dtype, rng):
    """`architecture` at `dtype`, its variables and inputs, and its forward
    pass as a function of them: DiffusionGemma's encodes its prompt first.
    At float64 the parameters and the float inputs are float64 too."""
    model = build_model(architecture, dtype)
    inputs = tiny_inputs(architecture, rng)
    args, kwargs = inputs if isinstance(inputs[0], tuple) else (inputs, {})
    variables = model.init(rng, *args, **kwargs)
    if dtype == "float64":
        variables, args, kwargs = jax.tree.map(
            lambda leaf: leaf.astype(jnp.float64) if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf,
            (variables, args, kwargs))

    def run(variables, args, kwargs):
        if architecture == "diffusion_gemma":
            cache = model.apply(variables, 1, method=model.init_cache, mutable=["cache"])[1]["cache"]
            cache = model.apply({**variables, "cache": cache}, jnp.zeros((1, 8), jnp.int32),
                                method=model.encode, mutable=["cache"])[1]["cache"]
            variables = {**variables, "cache": cache}
        return model.apply(variables, *args, **kwargs)

    return run, (variables, args, kwargs)
