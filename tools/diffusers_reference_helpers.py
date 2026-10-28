"""Replayed draws, saved pipelines and tiny text encoders for Diffusers fixtures."""

from __future__ import annotations

import contextlib
import json
import math
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import torch

if TYPE_CHECKING:
    from recorded_draws import Tape


def restore_transformers_names() -> None:
    """Put back the two names diffusers 0.34.0's pipeline modules import from
    `transformers.utils`, which transformers 5 dropped (`FLAX_WEIGHTS_NAME`,
    `WEIGHTS_INDEX_NAME`), so a tool walking those pipelines under
    transformers 5 can import them. A tool calls it before its first pipeline
    import; nothing it would compute is copied in their place."""
    import transformers.utils

    for name, value in (("FLAX_WEIGHTS_NAME", "flax_model.msgpack"),
                        ("WEIGHTS_INDEX_NAME", "pytorch_model.bin.index.json")):
        if not hasattr(transformers.utils, name):
            setattr(transformers.utils, name, value)


def frequency_exponent(half: int, shift: float, max_period: float) -> torch.Tensor:
    """`get_timestep_embedding`'s float32 exponent, its own arithmetic."""
    exponent = -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32)
    return exponent / (half - shift)


def torch_exp_disagreements(half: int) -> int:
    """How many of a `half`-entry table's frequencies torch's float32 `exp`
    rounds differently from the float64 one; the attribution a record carries."""
    exponent = frequency_exponent(half, 0, 10000)
    return int((torch.exp(exponent) != torch.exp(exponent.double()).float()).sum())


def step_noise(shape: tuple[int, ...], count: int,
               dtype: type[np.float32] | type[np.float64]) -> list[np.ndarray]:
    """Draw float32 normals under the walk's folded key, then store in `dtype`."""
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(0)
    return [np.asarray(jax.random.normal(jax.random.fold_in(key, index), shape, jnp.float32),
                       dtype) for index in range(count)]


@contextlib.contextmanager
def fed_noise(module: ModuleType, noises: list[np.ndarray]) -> Iterator[Tape]:
    """Replay draws in the requested dtype, on a tape that counts them.

    An omitted dtype keeps the input array's dtype. Modules without
    `randn_tensor` take no draws; the original function is restored on exit.
    """
    from recorded_draws import Tape

    tape = Tape(noises)

    def draw(shape, generator=None, device=None, dtype=None, layout=None):
        return torch.tensor(tape.next(shape=shape), dtype=dtype)

    if getattr(module, "randn_tensor", None) is None:
        yield tape
        return
    with patch.object(module, "randn_tensor", draw):
        yield tape


def recording(destination: str, cohort: str) -> Path:
    """The directory a generator records into, refused unless the installed
    releases are `cohort`'s, the ones its fixtures were recorded against
    (`reference_releases.require`), with torch on the two threads they were
    recorded on."""
    from reference_releases import require

    require(cohort)
    torch.set_num_threads(2)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    return root


def perturb(generator: torch.Generator, scale: float, *modules: torch.nn.Module) -> None:
    """Move every parameter of `modules` off its initialization by `scale`
    times a standard normal draw from `generator`, in order."""
    with torch.no_grad():
        for module in modules:
            for parameter in module.parameters():
                parameter.add_(scale * torch.randn(parameter.shape, generator=generator))


def saved_pipeline(pipe, directory: Path, height: int, width: int):
    """`pipe` saved to `directory` as safetensors, its progress bar off, and
    the geometry it is read at declared in its model_index.json (`dew_height`,
    `dew_width`), which the pipeline classes do not declare."""
    pipe.save_pretrained(directory, safe_serialization=True)
    pipe.set_progress_bar_config(disable=True)
    index = json.loads((directory / "model_index.json").read_text())
    index.update(dew_height=height, dew_width=width)
    (directory / "model_index.json").write_text(json.dumps(index, indent=2))
    return pipe


def drawn_wan_vae(model_class, config: dict, seed: int, directory: Path):
    """A Wan-family VAE `model_class` of `config`, its latent statistics, its
    norm gains and biases drawn from `seed` (every branch live), each weight
    rounded to bfloat16's grid so a bfloat16 run reads it exactly, saved to
    `directory`. Returns the model and the generator, to draw on from."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    z_dim = config["z_dim"]
    latents_mean = torch.randn(z_dim, generator=generator).mul(0.5).tolist()
    latents_std = torch.rand(z_dim, generator=generator).add(0.5).tolist()
    model = model_class(**config, latents_mean=latents_mean, latents_std=latents_std).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(("gamma", "bias")):
                parameter.add_(0.1 * torch.randn(parameter.shape, generator=generator))
            parameter.copy_(parameter.to(torch.bfloat16).float())
    model.save_pretrained(directory, safe_serialization=True)
    return model, generator


def tiny_qwen3_encoder(source: Path, tok, *, hidden_size: int, layers: int, seed: int):
    """The published Qwen3 text encoder's config under `source`, narrowed to `hidden_size` wide and
    `layers` deep with two query heads over one key head, every field it does not name the
    release's own, its weights drawn at `seed + 5` and perturbed at `seed + 6`."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = json.loads((source / "text_encoder" / "config.json").read_text())
    config.pop("torch_dtype", None)
    config.pop("dtype", None)
    config.update(hidden_size=hidden_size, intermediate_size=32, num_hidden_layers=layers,
                  layer_types=["full_attention"] * layers, max_window_layers=layers, num_attention_heads=2,
                  num_key_value_heads=1, head_dim=8, vocab_size=len(tok), max_position_embeddings=1024,
                  bos_token_id=None, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    torch.manual_seed(seed + 5)
    model = Qwen3ForCausalLM(Qwen3Config(**config)).eval()
    perturb(torch.Generator().manual_seed(seed + 6), 0.05, model)
    return model


def flux2_ids(rows: int, columns: int, tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`Flux2Pipeline._prepare_text_ids` and `_prepare_latent_ids`: the four-axis ids the pipeline lays
    out for a text of `tokens` and a latent of `rows` by `columns` positions."""
    text = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(tokens))
    image = torch.cartesian_prod(torch.arange(1), torch.arange(rows), torch.arange(columns), torch.arange(1))
    return text, image
