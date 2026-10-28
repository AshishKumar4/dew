"""A latent attention's decode cache, which it keeps in the attention cache (`open_kv_cache`)."""

import jax
import jax.numpy as jnp
import pytest

from dew.mesh import SEQUENCE_AXIS, LayoutRefused
from dew.nn.inputs import MixerCall
from dew.nn.mla import MultiHeadLatentAttention


@pytest.mark.mesh
@pytest.mark.parametrize("index_topk", [None, 4], ids=["dense", "sparse"])
def test_a_latent_attention_refuses_to_decode_under_a_sequence_axis_as_attention_does(index_topk):
    """A decode cache holds whole sequences, so a latent attention, its dense latents and its sparse
    expanded keys alike, refuses to decode under a sequence axis, as attention's own cache does."""
    module = MultiHeadLatentAttention(
        emb_features=32, num_heads=2, max_seq_len=16, q_lora_rank=8, kv_lora_rank=8, qk_nope_head_dim=8,
        qk_rope_head_dim=4, v_head_dim=8, index_topk=index_topk, index_n_heads=2 if index_topk else None,
        index_head_dim=8 if index_topk else None)
    with jax.set_mesh(jax.make_mesh((2,), (SEQUENCE_AXIS,))), pytest.raises(LayoutRefused):
        module.init(jax.random.key(0), jnp.zeros((1, 1, 32)), MixerCall(decode=True))
