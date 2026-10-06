"""Router parity readings and the two-process expert-exchange launch the MoE
tests share. The routers stay with their tests: V2's softmax and best-expert
group score and V3's sigmoid, two-best group score and bias differ."""

import subprocess
import sys

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


def exchange_worker(script, out, processes, process_id, coordinator):
    """One rank of an expert-exchange script, as run_pool's `start`: one CPU
    device, its own session, and the rank, coordinator and report as argv."""
    from test_multiprocess import worker_env

    return subprocess.Popen(
        [sys.executable, str(script), str(process_id), coordinator, str(out)],
        env=worker_env(1), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True)
