"""dew.interop.weights' indexed-name folding, the rule every autoencoder family's tensor names share."""

import pytest

from dew.interop.weights import indexed_path


@pytest.mark.parametrize(("name", "indexed", "path"), [
    ("down_blocks.0.resnets.1.conv1", None, ("down_blocks_0", "resnets_1", "conv1")),
    ("decoder.up_blocks.0.0.conv", None, ("decoder", "up_blocks_0_0", "conv")),
    ("0.proj", None, ("0", "proj")),
    ("decoder.up_blocks.2.resample.1", {"up_blocks"}, ("decoder", "up_blocks_2", "resample", "1")),
    ("a.0.1", {"a"}, ("a_0", "1")),
])
def test_a_numeric_token_joins_the_indexed_token_before_it(name, indexed, path):
    """A digit joins the token before it where that token is indexed (any, unrestricted), so
    consecutive digits all join it; a leading digit, or one after a token left out, stays its own."""
    assert indexed_path(name.split("."), indexed=indexed) == path
