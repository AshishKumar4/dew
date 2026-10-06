"""Reference monkeypatches restore upstream functions when a walk fails."""

import numpy as np
import pytest


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
