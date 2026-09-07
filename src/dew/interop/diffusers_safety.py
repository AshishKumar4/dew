"""The Diffusers SD checker with Dew's native CLIP vision tower.

Score equations and three-decimal thresholds follow Diffusers 0.34.0
safety_checker_flax.py. The processor and all checkpoint tensors survive save.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from transformers.models.clip.image_processing_pil_clip import CLIPImageProcessorPil

from dew.nn.text_encoders import CLIPTowerOutput, CLIPVisionTransformer, translate_clip_weights, translate_vision_config
from dew.objectives.base import Variables


def _vision_from_flax(vision):
    return {**vision["embeddings"], "pre_layernorm": vision["pre_layrnorm"],
            "post_layernorm": vision["post_layernorm"],
            **{f"layers_{i}": layer for i, layer in vision["encoder"]["layers"].items()}}


def _vision_to_flax(vision):
    return {"embeddings": {key: vision[key] for key in ("class_embedding", "patch_embedding", "position_embedding")},
            "pre_layrnorm": vision["pre_layernorm"], "post_layernorm": vision["post_layernorm"],
            "encoder": {"layers": {key.removeprefix("layers_"): value for key, value in vision.items()
                                   if key.startswith("layers_")}}}


@dataclass(eq=False)
class SafetyChecker:
    tower: CLIPVisionTransformer
    processor: CLIPImageProcessorPil
    config: dict
    params: Variables

    @classmethod
    def load(cls, directory: Path, *, from_pt=False, dtype=jnp.float32):
        folder = directory / "safety_checker"
        config = json.loads((folder / "config.json").read_text())
        if from_pt:
            from safetensors.numpy import load_file
            tensors = load_file(folder / "model.safetensors")
            heads = {name: value for name, value in tensors.items()
                     if name.startswith(("concept_embeds", "special_care_embeds"))}
            clip = {name.removeprefix("vision_model.") if name.startswith("vision_model.") else name: value
                    for name, value in tensors.items() if name not in heads}
            params = {**translate_clip_weights(clip), **heads}
        else:
            raw = serialization.msgpack_restore((folder / "flax_model.msgpack").read_bytes())
            if not isinstance(raw, dict):
                raise ValueError("Safety checker parameters must form a mapping")
            params = {**raw, "vision_model": _vision_from_flax(raw["vision_model"]["vision_model"])}
        tower = CLIPVisionTransformer(**translate_vision_config(config), dtype=dtype)
        processor = CLIPImageProcessorPil.from_pretrained(directory / "feature_extractor", local_files_only=True)
        return cls(tower, processor, config, params)

    def features(self, params, pixel_values):
        output = self.tower.apply({"params": params["vision_model"]}, pixel_values)
        assert isinstance(output, CLIPTowerOutput)
        return output.pooler_output @ params["visual_projection"]["kernel"]

    def __call__(self, params, pixel_values):
        embeddings = self.features(params, pixel_values)

        def cosine(concepts):
            images = (embeddings.T / jnp.maximum(jnp.linalg.norm(embeddings, axis=1), 1e-12)).T
            concepts = (concepts.T / jnp.maximum(jnp.linalg.norm(concepts, axis=1), 1e-12)).T
            return images @ concepts.T

        special = jnp.round(cosine(params["special_care_embeds"]) - params["special_care_embeds_weights"], 3)
        adjustment = jnp.any(special > 0, axis=1, keepdims=True) * 0.01
        concepts = jnp.round(cosine(params["concept_embeds"]) - params["concept_embeds_weights"] + adjustment, 3)
        return jnp.any(concepts > 0, axis=1)

    def filter(self, params, images):
        pixels = np.rint((np.asarray(images) + 1) * 127.5).clip(0, 255).astype(np.uint8)
        features = self.processor(list(pixels), return_tensors="np").pixel_values
        flagged = self(params, features)
        if np.any(flagged):
            warnings.warn("The checkpoint safety checker flagged generated images; returning black images.", stacklevel=2)
        return jnp.where(flagged[:, None, None, None], -jnp.ones_like(images), images)

    def save(self, directory: Path, params):
        folder = directory / "safety_checker"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text(json.dumps(self.config, indent=2))
        raw = {**params, "vision_model": {"vision_model": _vision_to_flax(params["vision_model"])}}
        (folder / "flax_model.msgpack").write_bytes(serialization.to_bytes(raw))
        self.processor.save_pretrained(directory / "feature_extractor")
