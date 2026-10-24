"""dew.interop.weights' indexed-name folding, the rule every autoencoder family's tensor names share."""

import pytest

from dew.interop.weights import indexed_path


def test_retained_tensors_stay_unread_until_their_export_shard_is_written(tmp_path):
    """Composing retained source tensors into an export reads metadata only; the shard writer
    subsequently requests their values and writes a checkpoint that reloads unchanged."""
    import jax
    import numpy as np

    from dew.interop.decoder_parts import layout_tensors
    from dew.interop.safetensors_io import LazyTensors, read_weights, save_sharded

    values = {"a.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
              "b.weight": np.arange(20, dtype=np.float32).reshape(4, 5)}
    reads = []

    def read(name):
        reads.append(name)
        return values[name].copy()

    retained = LazyTensors({name: jax.ShapeDtypeStruct(value.shape, value.dtype)
                            for name, value in values.items()}, read)
    export = layout_tensors({}, {}, retained=retained)
    assert not reads
    save_sharded(export, tmp_path, max_shard_size=80)
    assert set(reads) == set(values)
    restored = read_weights(tmp_path)
    for name, expected in values.items():
        np.testing.assert_array_equal(restored[name], expected)


@pytest.mark.parametrize(("name", "indexed", "path"), [
    ("down_blocks.0.resnets.1.conv1", None, ("down_blocks_0", "resnets_1", "conv1")),
    ("decoder.up_blocks.0.0.conv", None, ("decoder", "up_blocks_0_0", "conv")),
    ("0.proj", None, ("0", "proj")),
    ("decoder.up_blocks.2.resample.1", {"up_blocks"}, ("decoder", "up_blocks_2", "resample", "1")),
    ("a.0.1", {"a"}, ("a_0", "1")),
    ("a.0.1", {"a", "a_0"}, ("a_0_1",)),
    ("a.0", set(), ("a", "0")),
])
def test_a_numeric_token_joins_the_indexed_token_before_it(name, indexed, path):
    """A digit joins the token before it where that token, as folded so far, is indexed (any,
    unrestricted; none, where empty): unrestricted, consecutive digits all join it; a leading digit, or
    one after a token left out, stays its own."""
    assert indexed_path(name.split("."), indexed=indexed) == path


def test_no_tokens_fold_to_no_path():
    assert indexed_path(()) == ()
