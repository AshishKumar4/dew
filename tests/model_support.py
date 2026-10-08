"""The tiny decoder and the parameter-tree view several model tests share."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.inference import RunProcessor, TextGeneration
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.mixers.gated_delta_net import GatedDeltaNetMixer
from dew.nn.mla import MLAMixer
from dew.sampling import Sampling

TINY_DECODER = {"emb_features": 32, "num_layers": 2, "num_heads": 4, "mlp_features": 64, "max_seq_len": 16}
"""A causal transformer 32 wide, two layers of four heads; each test file adds
its own vocabulary and any field its claim turns on."""


def flat_tree(tree) -> dict:
    """`tree`'s leaves by their dictionary keys joined with dots, the way a
    checkpoint names them: 'params.layers_0.mlp.gate.kernel'."""
    return {".".join(str(entry.key) for entry in path): leaf
            for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]}


def decoder(kind: str = "attention") -> CausalTransformer:
    mixer = None
    if kind == "mla":
        mixer = MLAMixer(q_lora_rank=8, kv_lora_rank=8, qk_nope_head_dim=4,
                         qk_rope_head_dim=4, v_head_dim=8)
    elif kind == "recurrent":
        mixer = GatedDeltaNetMixer(linear_num_key_heads=2, linear_num_value_heads=2,
                                  linear_key_head_dim=8, linear_value_head_dim=8)
    return CausalTransformer(vocab_size=13, emb_features=16, num_layers=1, num_heads=2,
                             head_dim=8, mlp_features=32, max_seq_len=12,
                             dtype="float32", mixer=mixer)


class Digits:
    def encode(self, text):
        return [int(character) for character in text]

    def decode(self, ids):
        return "".join(str(int(token)) for token in ids)


_DEFAULT_TASK_SAMPLING = Sampling(temperature=0, eos_id=12)


def serving_task(sampling: Sampling = _DEFAULT_TASK_SAMPLING, capacity: int = 128) -> TextGeneration:
    model = decoder().clone(max_seq_len=capacity)
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    return TextGeneration(model, params, RunProcessor(Digits()), sampling=sampling)


def greedy_walk(model: nn.Module, params: dict, prompt, steps: int,
                transform: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None) -> np.ndarray:
    """Greedy continuation with a fresh full forward for every token."""
    sequence = np.asarray(prompt)
    for _ in range(steps):
        logits = np.asarray(model.apply(params, jnp.asarray(sequence))[:, -1])
        if transform is not None:
            logits = transform(sequence, logits)
        sequence = np.concatenate([sequence, logits.argmax(-1)[:, None].astype(np.int32)], axis=1)
    return sequence
