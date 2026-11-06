"""What the policy-gradient suites share: a position-wise policy with the
causal stack's scoring contract, verl's token-mean reduction of the
surrogate terms they check `dew.rl` against, and an asynchronous scheduler's
scripted source and publisher."""

from concurrent.futures import Future

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.nn.protocols import OutputTable
from dew.rl import surrogate


class TinyHead(nn.Module):
    """A position-wise map with the backbone's scoring contract, standing in
    for the causal stack: int32 ids in, float32 logits out, the head split
    off behind `hidden_states` and `output_table`. It takes the packing
    keywords and the decoding cache the causal stack does and reads neither:
    the trunk mixes nothing across positions."""

    vocab_size: int

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

    def __call__(self, tokens, train: bool = False, decode: bool = False, coordinates=None):
        return self.lm_head(
            self.hidden_states(tokens, train=train)).astype(jnp.float32)

    def output_table(self):
        return OutputTable(self.lm_head.variables["params"]["kernel"], vocab_major=False)


class TinyLogits(nn.Module):
    """`TinyHead`'s map as a model that gives its logits from its tokens
    alone (`Logits`): no states and no head over them to score."""

    vocab_size: int

    def setup(self):
        self.policy = TinyHead(self.vocab_size)

    def logits(self, tokens, train: bool = False, **packing):
        return self.policy(tokens, train=train)

    def __call__(self, tokens, train: bool = False, **packing):
        return self.logits(tokens, train=train)


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


def prompt_inputs(records):
    """A batch's left-padded prompts (`prompt`, `prompt_length`) as model inputs, each row's real
    tokens the last `prompt_length` and the rest masked out."""
    from dew.nn.inputs import ModelInputs

    tokens = jnp.asarray(records["prompt"])
    padding = tokens.shape[1] - jnp.asarray(records["prompt_length"])[:, None]
    return ModelInputs(tokens, {"attention_mask": jnp.arange(tokens.shape[1])[None, :] >= padding})





def completion_length(text: str, record) -> float:
    """A reward of a completion's length, which varies within a group, so every group has an advantage."""
    return float(len(text))


class ScriptedSource:
    """An asynchronous scheduler's source (`dew.objectives.rl.scheduler.Asynchronous`) whose sessions
    `drawn(task, sample, version)` draws at once, each submitted call's future done; a cancel cancels
    them."""

    def __init__(self, drawn):
        self.drawn = drawn

    def submit(self, task, samples, *, version, key):
        futures = []
        for sample in range(samples):
            future = Future()
            future.set_result(self.drawn(task, sample, version))
            futures.append(future)
        return futures

    def cancel(self, futures):
        for future in futures:
            future.cancel()


class Published:
    """An asynchronous scheduler's publisher that serves no weights: it takes each version published as
    loaded."""

    version = 0

    def load(self, variables, version):
        self.version = version


def rollout_prompts() -> dict:
    """Eight left-padded prompts whose lengths and EOS timing differ by rank.

    Rows 0-3 belong to process zero and rows 4-7 to process one under the
    trainer's halving. Process zero holds one real length, process one holds
    three, so the ranks would run different bucket counts on their own, and
    the ranks stop at different steps. Eight rows split over the eight-device
    mesh.
    """
    prompts = np.array([[3, 4, 5, 6], [9, 10, 5, 6], [1, 2, 3, 4], [7, 8, 9, 1],
                        [0, 0, 0, 2], [0, 0, 7, 8], [0, 6, 5, 4], [0, 3, 8, 1]], np.int32)
    return {"prompt": prompts, "prompt_length": np.array([4, 4, 4, 4, 1, 2, 3, 3], np.int32),
            "data_source": np.full((8, 1), 97, np.int32),
            "ground_truth": np.full((8, 1), 49, np.int32),
            "extra_info": np.zeros((8, 1), np.int32)}


def pooled_rollout() -> tuple:
    """The tiny decoder a pool's rollout tests draw from, its parameters, `rollout_prompts` whole and this
    process's share of them, under the trainer's halving."""
    from dew.nn.backbones.causal_transformer import CausalTransformer

    rank, processes = jax.process_index(), jax.process_count()
    model = CausalTransformer(vocab_size=13, emb_features=16, num_layers=1, num_heads=2,
                              head_dim=8, mlp_features=32, max_seq_len=12,
                              dtype="float32", attention_impl="xla")
    params = nn.unbox(model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32)))
    batch = rollout_prompts()
    rows = len(batch["prompt"]) // processes
    local = {name: value[rank * rows:(rank + 1) * rows] for name, value in batch.items()}
    return model, params, batch, local
