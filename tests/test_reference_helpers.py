"""Reference monkeypatches restore upstream functions when a walk fails."""

import numpy as np
import pytest


def test_kimi_mxfp4_zero_groups_have_finite_scales_and_decode_to_zero():
    torch = pytest.importorskip("torch")
    pytest.importorskip("compressed_tensors")
    from tools.kimi_k3_reference import mxfp4

    weight = torch.zeros((2, 64))
    weight[0, :32] = torch.linspace(-6, 6, 32)
    weight[1, 17] = -0.0
    packed, scales, decoded = mxfp4(weight)

    assert torch.isfinite(decoded).all()
    assert torch.equal(scales[:, 1], torch.full((2,), 127, dtype=torch.uint8))
    assert scales[1, 0] == 127
    assert torch.count_nonzero(packed[:, 16:]) == torch.count_nonzero(packed[1, :16]) == 0
    torch.testing.assert_close(decoded[:, 32:], weight[:, 32:], rtol=0, atol=0)
    torch.testing.assert_close(decoded[1, :32], weight[1, :32], rtol=0, atol=0)
    assert decoded[0, :32].abs().max() == 6


def test_rounded_frequencies_restore_the_source_after_an_exception():
    """A failed transformer walk leaves the original timestep embedding callable."""
    pytest.importorskip("diffusers")
    from diffusers.models import embeddings

    from tools.diffusers_reference_helpers import rounded_frequency_table, rounded_timestep_embedding

    original = embeddings.get_timestep_embedding
    with pytest.raises(RuntimeError, match="walk failed"), rounded_frequency_table():
        assert embeddings.get_timestep_embedding is rounded_timestep_embedding
        raise RuntimeError("walk failed")
    assert embeddings.get_timestep_embedding is original


def test_replayed_noise_keeps_requested_dtypes_and_restores_after_an_exception():
    """Successful draws are counted in order and a failed walk restores `randn_tensor`."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from diffusers.schedulers import scheduling_euler_ancestral_discrete as module

    from tools.diffusers_reference_helpers import fed_noise

    original = module.randn_tensor
    noises = [np.asarray([0.25, -0.5], np.float32), np.asarray([1.5, -2.0], np.float64)]
    with pytest.raises(RuntimeError, match="walk failed"), fed_noise(module, noises) as taken:
        for noise, dtype in zip(noises, (torch.float64, torch.float32), strict=True):
            drawn = module.randn_tensor(noise.shape, dtype=dtype)
            assert drawn.dtype == dtype
            np.testing.assert_array_equal(drawn.numpy(), noise)
        assert taken == [1, 1]
        raise RuntimeError("walk failed")
    assert module.randn_tensor is original
