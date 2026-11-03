"""Fresh SSD inputs and exact head, channel and state-coordinate orbits, and the recurrence's float64
oracle (`stepwise_scan`)."""

import contextlib
import functools
import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import with_gradients
from tool_support import load

from dew.nn.mixers.mamba2 import Mamba2, chunk_ssd, recurrent_ssd

_mamba_ssm = load("mamba_ssd_reference")

LAYER_SHAPES = {"in_proj.weight": (72, 12), "conv1d.weight": (44, 1, 4), "conv1d.bias": (44,),
                "dt_bias": (4,), "A_log": (4,), "D": (4,), "norm.weight": (24,), "out_proj.weight": (12, 24),
                "hidden": (2, 70, 12)}
"""One transformers `Mamba2Mixer`'s weights (torch's names and layouts) and hidden states at the layer
fixture's geometry (tests/fixtures/mamba2/config.json: a hidden width of 12, four heads of six, two groups of
five states, a 4-wide conv), two rows of 70 tokens."""

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mamba2"

LAYER_GEOMETRY = json.loads((FIXTURES / "config.json").read_text())
"""The layer fixture's geometry (tests/fixtures/mamba2/config.json)."""


def ssd_orbit(arrays, seed):
    """Permute heads with their groups and parameters, P channels and N coordinates together.

    Cotangents follow their primal's coordinates; neither outputs nor gradients are unpermuted.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    heads, width = arrays["x"].shape[2:]
    groups, state = arrays["B"].shape[2:]
    group_order = rng.permutation(groups)
    per_group = heads // groups
    head_order = np.concatenate([group * per_group + rng.permutation(per_group) for group in group_order])
    channels, states = rng.permutation(width), rng.permutation(state)
    axes = {
        "x": ((2, head_order), (3, channels)), "dy": ((2, head_order), (3, channels)),
        "dt": ((2, head_order),), "a": ((2, head_order),),
        "A": ((0, head_order),), "D": ((0, head_order),), "dt_bias": ((0, head_order),),
        "B": ((2, group_order), (3, states)), "C": ((2, group_order), (3, states)),
        "h0": ((1, head_order), (2, channels), (3, states)),
        "dfinal": ((1, head_order), (2, channels), (3, states)),
    }
    result = {}
    for name, value in arrays.items():
        value = np.asarray(value)
        for axis, order in axes.get(name, ()):
            value = np.take(value, order, axis=axis)
        result[name] = value
    return result


def mamba2_draw(seed, shape=(2, 70, 4, 6, 2, 5)):
    """The transformers scan fixture's distribution and geometry, with a fresh input and orbit seed.

    `dt` is raw, before softplus and its per-head bias, as the upstream reference takes it.
    """
    batch, length, heads, width, groups, state = shape
    rng = np.random.default_rng(seed)

    def normal(shape):
        return rng.standard_normal(shape).astype(np.float32)

    return ssd_orbit({
        "x": normal((batch, length, heads, width)), "dt": normal((batch, length, heads)),
        "A": -np.exp(normal((heads,))), "B": normal((batch, length, groups, state)),
        "C": normal((batch, length, groups, state)), "D": normal((heads,)),
        "dt_bias": normal((heads,)), "h0": normal((batch, heads, width, state)),
    }, seed)


def mamba_ssd_operands(rng, shape, dtype, documents):
    """The upstream recorder's native operands without torch, including its bf16-to-fp32 storage."""
    return _mamba_ssm.operands(rng, *shape, lambda value: value.astype(dtype).astype(np.float32), documents)


def mamba_ssd_draw(seed, shape, dtype, documents):
    """Regenerate a seeded mamba_ssm recording's native inputs and exact coordinate orbit."""
    return ssd_orbit(mamba_ssd_operands(np.random.default_rng(seed), shape, dtype, documents), seed)


def mamba2_run(site, arrays):
    """Dew's scan or selective step on the native inputs recorded for transformers."""
    x, dt, A, B, C, D, bias, state = (jnp.asarray(arrays[name]) for name in
                                     ("x", "dt", "A", "B", "C", "D", "dt_bias", "h0"))
    dt = jax.nn.softplus(dt + bias)
    if site == "step":
        output, final = recurrent_ssd(x[:, :1], dt[:, :1], A, B[:, :1], C[:, :1], D, state)
        output = output[:, 0]
    else:
        output, final = chunk_ssd(x, dt, A, B, C, D, state if site == "carried" else None, 32)
    return {"output": output, "final": final}


def ssd_value_and_grad(scan, operands, dy, dfinal):
    """A scan's output, final state and all seven operand gradients under both cotangents, each cast to
    its output's dtype, pulled back through the gate (`reference_error.with_gradients`)."""
    out, final = jax.eval_shape(scan, *operands)
    (out, final), *grads = with_gradients(scan)(*operands, (dy.astype(out.dtype), dfinal.astype(final.dtype)))
    return {"y": out, "final": final, **dict(zip(("x", "dt", "A", "B", "C", "D", "h0"),
                                               grads, strict=True))}


def mamba_ssd_xla(arrays, chunk, dtype=None):
    """The recorder's XLA sizing proxy and float64 cross-check, with optional native input casts."""
    operands = tuple(jnp.asarray(arrays[name], dtype if name in ("x", "B", "C") else None)
                     for name in ("x", "dt", "A", "B", "C", "D", "h0"))
    seq = arrays.get("seq")
    starts = (None if seq is None else
              jnp.concatenate([seq[:, :1] > 0, seq[:, 1:] > seq[:, :-1]], axis=1))
    return ssd_value_and_grad(lambda *args: chunk_ssd(*args, chunk_size=chunk, starts=starts), operands,
                              jnp.asarray(arrays["dy"], dtype), jnp.asarray(arrays["dfinal"]))


def stepwise_scan(x, b, c, a, state):
    """The recurrence one step at a time, no chunks: `h_t = exp(a_t) h_{t-1}
    + x_t b_t^T`, `y_t = h_t c_t`. Written apart from both chunked paths, so
    in float64 it is an oracle neither of them shares a line with."""
    def step(h, inputs):
        x, b, c, a = inputs
        h = jnp.exp(a)[..., None, None] * h + x[..., :, None] * b[..., None, :]
        return h, jnp.einsum('bhpn,bhn->bhp', h, c)

    final, ys = jax.lax.scan(step, state, tuple(jnp.moveaxis(t, 1, 0) for t in (x, b, c, a)))
    return jnp.moveaxis(ys, 0, 1), final


def mamba2_layer_draw(seed):
    """A fresh `Mamba2Mixer` (`LAYER_SHAPES`): its weights and hidden states standard normal, the weights
    halved, as the layer fixture draws them (tools/mamba2_reference.py's `layer_fixture`)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    return {name: (rng.standard_normal(shape) * (1.0 if name == "hidden" else 0.5)).astype(np.float32)
            for name, shape in LAYER_SHAPES.items()}


def mamba2_layer(dtype=jnp.float32) -> Mamba2:
    """Dew's Mamba-2 mixer at the layer fixture's geometry (`LAYER_GEOMETRY`) computing in `dtype`."""
    g = LAYER_GEOMETRY
    return Mamba2(emb_features=g["hidden_size"], num_heads=g["num_heads"], head_dim=g["head_dim"],
                  state_size=g["state_size"], n_groups=g["n_groups"], conv_kernel=g["conv_kernel"],
                  chunk_size=g["chunk_size"], norm_eps=g["layer_norm_epsilon"], dtype=dtype)


def mamba2_layer_params(arrays) -> dict:
    """A transformers `Mamba2Mixer`'s weights (torch's names) under Dew's names: kernels transposed from
    torch's [out, in], every other weight as stored."""
    return {"params": {
        "in_proj": {"kernel": jnp.asarray(arrays["in_proj.weight"]).T},
        "conv1d": {"weight": jnp.asarray(arrays["conv1d.weight"]),
                   "bias": jnp.asarray(arrays["conv1d.bias"])},
        "A_log": jnp.asarray(arrays["A_log"]), "dt_bias": jnp.asarray(arrays["dt_bias"]),
        "D": jnp.asarray(arrays["D"]), "norm": {"weight": jnp.asarray(arrays["norm.weight"])},
        "out_proj": {"kernel": jnp.asarray(arrays["out_proj.weight"]).T}}}


def mamba2_layer_run(site, arrays):
    """Dew's mixer on a drawn layer (`mamba2_layer_draw`): 'layer' in the arrays' dtype, 'layer_bf16'
    computing in bfloat16 on fp32 weights and bf16 hidden states, its output in fp32."""
    hidden = jnp.asarray(arrays["hidden"])
    if site == "layer_bf16":
        out = mamba2_layer(jnp.bfloat16).apply(mamba2_layer_params(arrays), hidden.astype(jnp.bfloat16))
        return {"output": out.astype(jnp.float32)}
    return {"output": mamba2_layer(hidden.dtype).apply(mamba2_layer_params(arrays), hidden)}


SCAN_STARTS = np.zeros((2, 70), bool)
SCAN_STARTS[0, 32] = SCAN_STARTS[1, 7] = SCAN_STARTS[1, 45] = True
"""The 'starts' site's document starts over the scan fixture's rows: one on the first token of a chunk, one
inside a chunk."""

PAUSED = np.asarray([[1] * 12, [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1]], bool)
"""The 'padded' site's two rows of 12 slots, the second paused on padding at five of them."""

CUTS = (0, 20, 22, 70)
"""The 'packed' site's row of three documents of 20, 2 (shorter than the conv's three-token history) and 48
tokens, the cuts off the chunk grid."""

PREFILL = 5
"""The 'decoded' site's prefill: its first five tokens at once, then one a step."""

TANGENT_TOKENS = 24
"""The 'tangent' site's tokens, one row of them."""


def documents_of(starts) -> list[list[tuple[int, int]]]:
    """Each row's documents as `(start, end)` spans from its `starts` `[B, S]` (a row's first document at
    token 0)."""
    spans = []
    for row in np.asarray(starts):
        cuts = [0, *np.flatnonzero(row[1:]) + 1, row.shape[0]]
        spans.append([(start, end) for start, end in itertools.pairwise(cuts) if end > start])
    return spans


def mamba2_starts_run(arrays):
    """Dew's scan over the scan fixture's rows with documents starting at `SCAN_STARTS`, a document's first
    token from a zero state, the first from the drawn `h0`, as transformers' scan runs each document alone
    (the 'starts' site's truth, `chunk_ssd` in the arrays' dtype)."""
    x, dt, A, B, C, D, bias, state = (jnp.asarray(arrays[name]) for name in
                                     ("x", "dt", "A", "B", "C", "D", "dt_bias", "h0"))
    dt = jax.nn.softplus(dt + bias)
    output, final = chunk_ssd(x, dt, A, B, C, D, state, 32, starts=jnp.asarray(SCAN_STARTS))
    return {"output": output, "final": final}


def mamba2_layer_site_run(site, arrays):
    """Dew's mixer on a drawn layer as transformers' runs the site's comparison (the site's truth in float64):
    'decoded' its forwards at the widths a cached decode computes (`model_support.as_stepped`), 'padded'
    each row with its padding closed, 'packed' each document alone, 'tangent' its JVP along the drawn
    direction over `TANGENT_TOKENS` tokens of the first row."""
    from model_support import as_stepped

    hidden = jnp.asarray(arrays["hidden"])
    module = mamba2_layer(hidden.dtype)
    variables = mamba2_layer_params(arrays)

    def mixer(x):
        return module.apply(variables, x)

    if site == "decoded":
        return {"output": as_stepped(mixer, hidden, PREFILL)}
    if site == "padded":
        return {"first": mixer(hidden[:1, :12])[0], "second": mixer(hidden[1, :12][PAUSED[1]][None])[0]}
    if site == "packed":
        return {"output": jnp.concatenate([mixer(hidden[:1, start:end])
                                           for start, end in itertools.pairwise(CUTS)], axis=1)}
    if site == "tangent":
        x = hidden[:1, :TANGENT_TOKENS]
        return {"output": jax.jvp(mixer, (x,), (jnp.asarray(arrays["direction"]).astype(x.dtype),))[1]}
    raise ValueError(f"no layer site {site!r}")


def mamba2_layer_site_draw(seed):
    """A drawn layer (`mamba2_layer_draw`) with the 'tangent' site's direction beside it, standard normal over
    `TANGENT_TOKENS` tokens of one row."""
    arrays = mamba2_layer_draw(seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1933]))
    shape = (1, TANGENT_TOKENS, arrays["hidden"].shape[-1])
    arrays["direction"] = rng.standard_normal(shape).astype(np.float32)
    return arrays


MAMBA2_130M = "3a5aea0c25d0fb43cc360e2c2aac82c26e3eed49"
"""state-spaces/mamba2-130m's revision (mamba_ssm's config and names, pickles on main; SFconvertbot's
refs/pr/1)."""

MAMBA2_130M_HF = "05e8773fc4ac1cd067e8a18a5c45372ce5178405"
"""AntonV/mamba2-130m-hf's revision: transformers' port of state-spaces/mamba2-130m."""

INTEROP_TOKENS = (2, 13)
"""The 'interop' site's rows of token ids."""

INTEROP_VOCAB = 50000
"""The ids the 'interop' site draws its tokens below: every one a GPT-NeoX tokenizer's own (its 50254, which
the recorder checks)."""


def mamba2_interop_draw(seed):
    """The 'interop' site's rows of token ids (`INTEROP_TOKENS`), each below `INTEROP_VOCAB`."""
    return {"ids": np.random.default_rng(np.random.SeedSequence([seed, 130])).integers(
        0, INTEROP_VOCAB, INTEROP_TOKENS).astype(np.int32)}


def mamba2_130m(dtype):
    """state-spaces/mamba2-130m through Dew's interop (`Pretrained.load`), computing and stored in `dtype`
    (float64 loaded under `jax.enable_x64`, which its runs take too)."""
    from dew.interop.pretrained import Pretrained

    wide = jnp.dtype(dtype) == jnp.float64
    with jax.enable_x64(new_val=True) if wide else contextlib.nullcontext():
        return Pretrained.load("state-spaces/mamba2-130m", revision=MAMBA2_130M, dtype=dtype,
                               param_dtype=dtype)


def mamba2_130m_logits(loaded):
    """The 'interop' site's computation on `loaded` (`mamba2_130m`): a draw's logits, fp32 where narrower."""
    def run(inputs):
        logits = loaded.model.apply(loaded.variables, jnp.asarray(inputs["ids"]))
        return {"logits": logits.astype(jnp.promote_types(logits.dtype, jnp.float32))}

    return run


HF_TINY = Path(__file__).resolve().parent / "fixtures" / "hf" / "mamba2-tiny"
"""A tiny transformers `Mamba2ForCausalLM` (two layers, a hidden width of 8, a vocabulary of 32): the model
sites' architecture, its weights drawn afresh at each seed."""

HOLES = (2, 7, 8, 13)
"""The padded rows' padding slots: two documents of six real tokens each, the second starting after the gap
at 7 and 8."""


@functools.cache
def mamba2_model():
    """Dew's model of `HF_TINY` through its interop (`Pretrained.load`), fp32, its loaded weights a template
    the model draws replace."""
    from dew.interop.pretrained import Pretrained

    return Pretrained.load(HF_TINY, dtype="float32", attention_impl="reference")


def padded_layout(length: int) -> tuple[np.ndarray, np.ndarray]:
    """A row of `length` slots, padding at `HOLES`: its validity `[1, length]` and segment ids, two documents
    of six real tokens and the rest of the row a third."""
    valid = np.ones((1, length), bool)
    valid[0, list(HOLES)] = False
    real = np.flatnonzero(valid[0])
    segments = np.zeros((1, length), np.int32)
    segments[0, real[:6]], segments[0, real[6:]] = 1, 2
    return valid, segments


def mamba2_model_draw(seed):
    """A fresh model of `HF_TINY`'s architecture: each weight of Dew's tree (by its path, `variables/...`)
    standard normal halved, and token ids `[2, 17]` in 1 to 31."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
    leaves = jax.tree_util.tree_flatten_with_path(mamba2_model().variables)[0]
    arrays = {f"variables{jax.tree_util.keystr(path)}":
              (rng.standard_normal(np.shape(leaf)) * 0.5).astype(np.float32) for path, leaf in leaves}
    arrays["ids"] = rng.integers(1, 32, (2, 17)).astype(np.int32)
    return arrays


def mamba2_model_variables(arrays):
    """A model draw's weights (`mamba2_model_draw`) as Dew's tree."""
    leaves, tree = jax.tree_util.tree_flatten_with_path(mamba2_model().variables)
    return jax.tree.unflatten(tree, [jnp.asarray(arrays[f"variables{jax.tree_util.keystr(path)}"])
                                     for path, _ in leaves])


def mamba2_model_hidden(model, variables, tokens, **fields):
    """The model's final normalized states over `tokens` `[B, S]`, its token `fields` laid out once."""
    from dew.batch import field_coordinates
    from dew.nn.backbones.causal_transformer import CausalTransformer

    return model.apply(variables, tokens, method=CausalTransformer.hidden_and_mtp_inputs,
                       coordinates=field_coordinates(fields, tokens.shape))[0]


def mamba2_model_site_run(site, arrays):
    """Dew's model on a draw as transformers' runs the site's comparison (the site's truth in float64):
    'model_padded' and 'model_scored' the final states of each document of a padded row (16 and 17 slots,
    `padded_layout`) alone, over its real tokens, the scored row's first 12 of them (its inputs);
    'model_decoded' the logits at the widths a cached decode computes from a prefill of four (`as_stepped`),
    from the fourth."""
    from model_support import as_stepped, float64_twin

    variables = mamba2_model_variables(arrays)
    model = mamba2_model().model
    model = float64_twin(model) if jax.tree.leaves(variables)[0].dtype == jnp.float64 else model
    ids = jnp.asarray(arrays["ids"])
    if site == "model_decoded":
        return {"logits": as_stepped(lambda held: model.apply(variables, held), ids[:, :12], 4)[:, 3:]}
    length = {"model_padded": 16, "model_scored": 17}[site]
    valid, segments = padded_layout(length)
    real = ids[:1, :length][:, valid[0]]
    closed = segments[:, valid[0]]
    if site == "model_scored":
        real, closed = real[:, :12], closed[:, :12]
    spans = documents_of(np.concatenate([[True], closed[0, 1:] != closed[0, :-1]])[None])[0]
    return {"hidden": jnp.concatenate([mamba2_model_hidden(model, variables, real[:, start:end])[0]
                                       for start, end in spans])}


def mamba2_scan_twin(site, arrays):
    """Dew's scan as a scan site's truth runs it: `mamba2_starts_run` for 'starts', else `mamba2_run`."""
    return mamba2_starts_run(arrays) if site == "starts" else mamba2_run(site, arrays)


def mamba2_layer_twin(site, arrays):
    """Dew's mixer as a layer site's truth runs it: the mixer itself for 'layer' and 'layer_bf16' (both in
    float64 there), else `mamba2_layer_site_run`."""
    if site in ("layer", "layer_bf16"):
        return mamba2_layer_run("layer", arrays)
    return mamba2_layer_site_run(site, arrays)


SCAN_SITES = ("scan", "carried", "step", "starts")
LAYER_SITES = ("layer", "layer_bf16", "decoded", "padded", "packed", "tangent")
MODEL_SITES = ("model_padded", "model_scored", "model_decoded")
"""The seeded recording's sites (tools/mamba2_reference.py's `write_seeded`), by their draws."""


def mamba2_site(site):
    """A seeded site's draw and Dew's float64 twin, its truth: `(inputs_of, twin)`."""
    if site in SCAN_SITES:
        return mamba2_draw, mamba2_scan_twin
    if site in LAYER_SITES:
        return mamba2_layer_site_draw, mamba2_layer_twin
    return mamba2_model_draw, mamba2_model_site_run
