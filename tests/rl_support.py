"""What the policy-gradient suites share: a position-wise policy with the
causal stack's scoring contract, and verl's token-mean reduction of the
surrogate terms they check `dew.rl` against."""

import jax.numpy as jnp
from flax import linen as nn

from dew.rl import surrogate


class TinyHead(nn.Module):
    """A position-wise map with the backbone's scoring contract, standing in
    for the causal stack: int32 ids in, float32 logits out, the head split
    off behind `hidden_states` and `head_weight`. It takes the packing
    keywords and the decoding cache the causal stack does and reads neither:
    the trunk mixes nothing across positions."""

    vocab_size: int
    final_logit_softcap = None
    precision = None

    def setup(self):
        self.lm_head = nn.Dense(self.vocab_size, use_bias=False)

    @nn.compact
    def hidden_states(self, tokens, train: bool = False, **packing):
        x = nn.Embed(self.vocab_size, 8)(tokens)
        h = nn.LayerNorm()(x)
        return nn.LayerNorm()(x + nn.Dense(8)(nn.gelu(nn.Dense(16)(h))))

    @nn.compact
    def init_cache(self, batch_size):
        """A placeholder cache: incremental decoding keeps no state, but `generate` threads one."""
        self.variable("cache", "index", lambda: jnp.zeros((batch_size,), jnp.int32))

    def __call__(self, tokens, train: bool = False, decode: bool = False, attention_mask=None):
        return self.lm_head(
            self.hidden_states(tokens, train=train)).astype(jnp.float32)

    def head_weight(self, params):
        return params["lm_head"]["kernel"].astype(jnp.float32)


def token_mean(x, mask):
    """verl's token-mean aggregation, `agg_loss(loss_agg_mode="token-mean")`,
    which is how GRPO reduces these terms: masked values out before the
    multiply, so a nan behind the mask stays out, over the exact token count."""
    weights = mask.astype(x.dtype)
    return jnp.sum(jnp.where(weights != 0, x, 0) * weights) / jnp.sum(weights)


def clipped_surrogate(log_ratio, advantages, mask, module=surrogate, **clip):
    """The token-mean reduction of the dual-clipped policy terms."""
    terms, aux = module.clipped_surrogate_terms(log_ratio, advantages, mask, **clip)
    return token_mean(terms, mask), aux
