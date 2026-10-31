"""Muon's opt-in variants, each against its paper's own math: Kimi K3's per-head Muon (arXiv 2607.24653
2.5), Polar Express's coefficients (arXiv 2505.16932) and NorMuon's neuron-wise normalization (arXiv
2510.05491)."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from optim_support import LR, decoder_params, drawn_tree
from reference_error import as_given, assert_as_exact
from test_optim import UPDATE_SEEDS, assert_updates_as, at, muon_solver, stacked_draw

from dew.newton_schulz import NEWTON_SCHULZ, POLAR_EXPRESS_TABLE, orthogonalized, polar_express

QUERY = ('params', 'layers_0', 'self_attn', 'q_proj', 'kernel')
"""The decoder's first query projection, `[embed 32, heads 2 * head_dim 16]`."""

HEADS = {"*q_proj/kernel": 16, "*k_proj/kernel": 16, "*v_proj/kernel": 16}
"""The decoder's query, key and value projections by pattern, each head 16 wide (`per_head`)."""


def one_matrix(tree):
    """Muon's dimension numbers over `tree`, each leaf one matrix, its first axis the reduction and its
    second the output, any other a batch of them."""
    return jax.tree.map(lambda _: optax.contrib.MuonDimensionNumbers(0, 1), tree)


def test_per_head_muon_orthogonalizes_each_head_alone():
    """Per-head Muon (`per_head`, Kimi K3's, arXiv 2607.24653 2.5): a query projection's update, its
    output side two heads 16 wide, is optax's Muon of each head's block alone, `[embed, head_dim]`, its
    momentum and its shape scaling the block's (check 1)."""
    def heads(path, value):
        # `[embed, heads * head_dim]` as `[embed, head_dim, heads]`: each head's block one matrix.
        return value.reshape(32, 2, 16).transpose(0, 2, 1) if path == QUERY else value

    assert_updates_as(muon_solver(optimizer_opts={"per_head": HEADS}),
                      optax.contrib.muon(LR, muon_weight_dimension_numbers=one_matrix), decoder_params(),
                      [QUERY], "Muon per head", flatten=heads)


def test_per_head_muonclip_clips_each_head_after_its_own_update():
    """MuonClip with per-head Muon: a step under sown maxima past the threshold stores each query and key
    head's weights as the paper's QK-Clip of per-head Muon's step (tools/muonclip_reference.py), its
    heads' rescale following the heads' own updates (check 1)."""
    from test_optim import CLIP_SEEDS, clip_step, qk_stats, qk_tree

    from tools.muonclip_reference import clip_qk_kernel, clip_scale

    stats = qk_stats(values=[[200.0, 150.0, 10.0, 5.0], [10.0] * 4, [10.0] * 4])
    per_head = {"*q_proj/kernel": 8, "*k_proj/kernel": 8}
    applied = clip_step(muon_solver(optimizer="muonclip", optimizer_opts={"per_head": per_head}), stats)
    muon = muon_solver(optimizer_opts={"per_head": per_head})

    def paper(params, grads):
        updates, _ = muon.update(grads, muon.init(params), params)
        s_max = jnp.max(jnp.asarray(stats['layers_0']['self_attn']['max_logits'][0]), axis=0)
        gamma = clip_scale(s_max, 100.0, guard=0.0)
        return {'layers_0': {'self_attn': {projection: {'kernel': clip_qk_kernel(
            params['layers_0']['self_attn'][projection]['kernel']
            + at(updates, ('layers_0', 'self_attn', projection, 'kernel')), gamma, 4)}
            for projection in ('q_proj', 'k_proj')}}}

    assert_as_exact(applied, paper, lambda seed: qk_tree(seed=seed), "MuonClip per head", seeds=CLIP_SEEDS,
                    truth_fn=paper)


def test_per_head_muon_orthogonalizes_each_latent_attention_head_alone():
    """Per-head Muon over multi-head latent attention's per-head projections: the query's up-projection,
    each head its `qk_nope + qk_rope` columns, and the key-value up-projection, each head its `qk_nope + v`
    columns, each head's block optax's Muon of it alone (check 1); its latent down-projections, shared by
    every head, one matrix each as Muon's."""
    from dew.nn.mla import MultiHeadLatentAttention

    mla = MultiHeadLatentAttention(emb_features=32, num_heads=2, max_seq_len=8, q_lora_rank=24,
                                   kv_lora_rank=16, qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8,
                                   attention_impl="reference")
    params = mla.init(jax.random.key(0), jnp.ones((1, 8, 32), jnp.float32))
    query, value = ('params', 'q_b_proj', 'kernel'), ('params', 'kv_b_proj', 'kernel')
    widths = {query: 12, value: 16}

    def heads(path, leaf):
        # `[latent, heads * width]` as `[latent, width, heads]`: each head's block one matrix.
        if path not in widths:
            return leaf
        return leaf.reshape(leaf.shape[0], 2, widths[path]).transpose(0, 2, 1)

    per_head = {"*q_b_proj/kernel": 12, "*kv_b_proj/kernel": 16}
    assert_updates_as(muon_solver(optimizer_opts={"per_head": per_head}),
                      optax.contrib.muon(LR, muon_weight_dimension_numbers=one_matrix), params,
                      [query, value], "latent attention per head", flatten=heads)


def test_per_head_muon_refuses_heads_its_output_side_does_not_hold():
    """A head width that does not divide the projection's output side names no stack of heads: refused
    by name."""
    params = decoder_params()
    solver = muon_solver(optimizer_opts={"per_head": {"*q_proj/kernel": 12}})
    with pytest.raises(ValueError):
        solver.update(jax.tree.map(jnp.ones_like, params), solver.init(params), params)


def polar_express_reference(matrix: np.ndarray, steps: int, dtype) -> np.ndarray:
    """Polar Express's `PolarExpress(G, steps)` as its paper's Appendix A writes it, in `dtype`: the matrix
    the wide way round over `1.01 ||X||_F + 1e-7`, then `aX + (bA + cA A) X` at each step's triple, the
    table's (`POLAR_EXPRESS_TABLE`) but its last divided by `(1.01, 1.01^3, 1.01^5)`, its last repeated."""
    safe = [(a / 1.01, b / 1.01 ** 3, c / 1.01 ** 5) for a, b, c in POLAR_EXPRESS_TABLE[:-1]]
    table = [*safe, POLAR_EXPRESS_TABLE[-1]]
    triples = table[:steps] + [table[-1]] * (steps - len(table))
    # Its code's own arithmetic: bfloat16 throughout, or float64 as the truth's.
    xp = np if dtype == np.float64 else jnp
    x = xp.asarray(matrix, dtype)
    tall = x.shape[0] > x.shape[1]
    x = x.T if tall else x
    x = x / (xp.linalg.norm(x.astype(np.float32)).astype(dtype) * 1.01 + 1e-7)
    for a, b, c in triples:
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return np.asarray(x.T if tall else x, np.float64)


@pytest.mark.parametrize("steps", [5, 10])
def test_polar_express_orthogonalizes_as_its_paper_does(steps):
    """Muon's orthogonalization at Polar Express's coefficients (`ns_coeffs="polar_express"`,
    `polar_express`) over a wide and a tall stack, in the default numerics' bfloat16, on par from float64
    with the paper's own iteration in bfloat16 (check 1): its input's safety factor folded into the first
    step's polynomial, its last triple repeated past the table."""
    numbers = {"wide": optax.contrib.MuonDimensionNumbers((1,), (2,)),
               "tall": optax.contrib.MuonDimensionNumbers((1,), (2,))}

    def dew(wide, tall):
        # The stacks in the tree's own order, "tall" before "wide".
        return jax.tree.leaves(orthogonalized({"wide": wide, "tall": tall}, numbers,
                                              jnp.asarray(polar_express(steps)), steps, 1e-7))

    def paper(dtype):
        @as_given
        def run(wide, tall):
            return [np.stack([polar_express_reference(np.asarray(matrix), steps, dtype) for matrix in stack])
                    for stack in (tall, wide)]
        return run

    assert_as_exact(dew, paper(jnp.bfloat16), stacked_draw, f"Polar Express, {steps} steps",
                    seeds=UPDATE_SEEDS, truth_fn=paper(np.float64))


@pytest.mark.parametrize("steps", [3, 5])
def test_polar_express_is_a_muon_preset(steps):
    """`ns_coeffs="polar_express"` names Polar Express's coefficients to Muon: its update is optax's Muon
    at those coefficients (`polar_express`), its momentum and shape scaling Muon's (check 1)."""
    assert_updates_as(muon_solver(optimizer_opts={"ns_coeffs": "polar_express", "ns_steps": steps}),
                      optax.contrib.muon(LR, ns_coeffs=polar_express(steps), ns_steps=steps),
                      decoder_params(), [QUERY, ('params', 'layers_1', 'mlp', 'down_proj', 'kernel')],
                      f"Polar Express {steps}")


def neuron_normed(update: np.ndarray, moment: np.ndarray, beta2: float) -> tuple[np.ndarray, np.ndarray]:
    """NorMuon's `normuon_update` lines past the orthogonalization, as its code writes them, for a matrix
    laid `[reduction, output]` (each output neuron a column here, a row in its `[out, in]`): the neuron's
    mean square decayed into `moment`, the update divided by its root, scaled back to its own Frobenius
    norm. The normed update and the moment."""
    moment = beta2 * moment + (1 - beta2) * np.mean(update * update, axis=0)
    divided = update / (np.sqrt(moment) + 1e-10)
    return divided * np.linalg.norm(update) / (np.linalg.norm(divided) + 1e-10), moment


def test_normuon_normalizes_each_neuron_as_its_paper_does():
    """NorMuon (`neuron_beta2`, arXiv 2510.05491): over three steps of drawn gradients, a projection's
    update is NorMuon's normalization of optax's orthogonalized momentum, each output neuron divided by
    the root of its decayed mean square and the matrix kept at the orthogonal update's norm, then Jordan's
    shape scaling and the rate, on par from float64 with its code's arithmetic (check 1)."""
    params = decoder_params()
    path = ('params', 'layers_0', 'mlp', 'down_proj', 'kernel')
    fan_in, fan_out = 64, 32

    def draw(seed):
        grads = [drawn_tree(params, seed * 3 + index)[1] for index in range(3)]
        return tuple(at(grad, path) for grad in grads)

    def dew(*grads):
        solver = muon_solver(optimizer_opts={"neuron_beta2": 0.95})
        tree = jax.tree.map(np.zeros_like, params)
        state, updates = solver.init(tree), tree
        for grad in grads:
            given = jax.tree_util.tree_map_with_path(lambda key, leaf, grad=grad: grad if tuple(
                entry.key for entry in key) == path else np.zeros_like(leaf), tree)
            updates, state = solver.update(given, state, tree)
        return at(updates, path)

    @as_given
    def paper(*grads):
        muon = optax.contrib.scale_by_muon(NEWTON_SCHULZ)
        state, moment, normed = muon.init(grads[0]), np.zeros(fan_out), np.zeros((fan_in, fan_out))
        for grad in grads:
            orthogonal, state = muon.update(grad, state)
            normed, moment = neuron_normed(np.asarray(orthogonal, np.float64), moment, 0.95)
        return -LR * np.sqrt(max(1.0, fan_out / fan_in)) * normed

    assert_as_exact(dew, paper, draw, "NorMuon's update", seeds=UPDATE_SEEDS, truth_fn=paper)


def test_normuon_normalizes_a_half_precision_update_in_float32():
    """NorMuon's normalization of a float16 update works in float32 (`scale_by_neuron_norm`): a zero
    update stays zero, its neurons' guard held, and a block whose squared sum is past float16's range
    keeps its norm, finite."""
    from dew.optim import scale_by_neuron_norm

    norm = scale_by_neuron_norm(0.95)
    zero = {"w": jnp.zeros((64, 32), jnp.float16)}
    updated, _ = norm.update(zero, norm.init(zero))
    assert not np.any(np.asarray(at(updated, ("w",))))
    large = {"w": jnp.full((512, 512), 100.0, jnp.float16)}
    updated, _ = norm.update(large, norm.init(large))
    assert np.all(np.isfinite(np.asarray(at(updated, ("w",)), np.float32)))
    assert np.linalg.norm(np.asarray(at(updated, ("w",)), np.float64)) == np.linalg.norm(
        np.asarray(large["w"], np.float64))
