"""The absolute sinusoidal tables (`dew.nn.rope.sinusoid_frequencies`, `sinusoids` and `sincos_grid`)
against the references that define them: each axis of the 2D patch grid as MAE's
`get_1d_sincos_pos_embed_from_grid` builds it in float64, the rows' half first, and SD3's position
buffer as Diffusers' builds it, each rounded once to float32 and equal to the bit; Gemma 3n's audio
relative positions on par with transformers' float32 table from the float64 one (check 1,
tests/reference_error.py), where Dew rounds the frequencies once from float64 rather than taking
float32's exp."""

import math

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from reference_error import as_given, assert_as_exact

from dew.nn.audio import _position_signal
from dew.nn.backbones.sd3 import sincos_position
from dew.nn.scan_orders import build_2d_sincos_pos_embed

AUDIO_SEEDS = 8
"""Check 1's declared seeds for the audio table, sized by `seeds_for` on armada."""


def mae_grid(emb_dim: int, rows: int, columns: int) -> np.ndarray:
    """Each axis as MAE's `get_1d_sincos_pos_embed_from_grid` (util/pos_embed.py) builds it, float64
    throughout: its positions times `1 / 10000 ** (i / (d / 2))`, sines then cosines; the rows' half
    before the columns', Dew's order (MAE's 2D table puts the columns first)."""
    def axis(width, positions):
        omega = np.arange(width // 2, dtype=np.float64) / (width / 2.0)
        angles = np.einsum("m,d->md", positions.reshape(-1), 1.0 / 10000 ** omega)
        return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)

    grid_w, grid_h = np.meshgrid(np.arange(columns, dtype=np.float32), np.arange(rows, dtype=np.float32))
    return np.concatenate([axis(emb_dim // 2, grid_h), axis(emb_dim // 2, grid_w)], axis=1)


@pytest.mark.parametrize("emb_dim, rows, columns", [(64, 8, 8), (384, 16, 16), (768, 14, 10)])
def test_each_patch_grid_axis_is_maes_float64_table_rounded_once(emb_dim, rows, columns):
    np.testing.assert_array_equal(build_2d_sincos_pos_embed(emb_dim, rows, columns).view(np.uint32),
                                  mae_grid(emb_dim, rows, columns).astype(np.float32).view(np.uint32))


@pytest.mark.parametrize("channels, grid, base_size", [(64, 8, 8), (1536, 192, 64), (1152, 64, 64)])
def test_sd3s_position_buffer_is_diffusers_bit_for_bit(channels, grid, base_size):
    from diffusers.models.embeddings import get_2d_sincos_pos_embed

    expected = get_2d_sincos_pos_embed(channels, grid, base_size=base_size, output_type="pt").float()
    found = np.asarray(sincos_position(channels, grid, base_size=base_size))[0]
    np.testing.assert_array_equal(found.view(np.uint32), expected.numpy().view(np.uint32))


@pytest.mark.parametrize("width", [64, 1536])
def test_gemma3n_audio_positions_are_on_par_with_the_reference(width):
    """Dew's relative-position table against `Gemma3nAudioRelativePositionEmbedding`'s float32 one
    over drawn positions, from the same table in float64 (check 1). The reference's frequencies are
    its `inv_timescales` buffer; its sines and cosines are taken as its timing signal takes them, in
    float32, sines first."""
    from transformers import Gemma3nAudioConfig
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nAudioRelativePositionEmbedding

    reference = Gemma3nAudioRelativePositionEmbedding(Gemma3nAudioConfig(
        hidden_size=width, conf_num_attention_heads=8))
    increment = math.log(1.0e4) / max(width // 2 - 1, 1)
    inverse = np.exp(np.arange(width // 2, dtype=np.float64) * -increment)

    def draw(seed):
        return (np.random.default_rng(seed).integers(-64, 64, 48).astype(np.int32),)

    def truth(positions):
        angles = np.asarray(positions, np.float64)[:, None] * inverse[None, :]
        return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)

    @as_given
    def theirs(positions):
        angles = torch.from_numpy(np.asarray(positions)).float()[:, None] * reference.inv_timescales[0]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).numpy()

    def ours(positions):
        return _position_signal(width, jnp.asarray(positions), jnp.float32)

    assert_as_exact(ours, theirs, draw, f"gemma3n audio positions {width}", seeds=AUDIO_SEEDS, truth_fn=truth)
