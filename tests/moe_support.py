"""Router parity readings the MoE tests share.

The routers stay with their tests: V2 scores by softmax and the best expert of
a group without renormalising, V3 by sigmoid and a group's two best with the
selection bias. What they share is how a reference's choice is read and how
its gate weight becomes a `Router` tree.
"""

import jax.numpy as jnp
import numpy as np


def by_expert(indices, weights):
    """One token's slots ordered by expert id, indices and weights together."""
    order = np.argsort(np.asarray(indices), axis=-1)
    return (np.take_along_axis(np.asarray(indices), order, axis=-1),
            np.take_along_axis(np.asarray(weights), order, axis=-1))


def router_variables(tensors, bias=False):
    """The reference gate weight as a `Router` parameter tree.

    torch Linear holds [out, in] and Dew keeps [in, out], the transpose every
    kernel takes in dew.interop.hf_decoders.
    """
    variables = {"params": {"kernel": jnp.asarray(tensors["mlp.gate.weight"].T)}}
    if bias:
        variables["moe"] = {"e_score_correction_bias": jnp.asarray(
            tensors["mlp.gate.e_score_correction_bias"])}
    return variables

