"""Fresh SSD inputs and exact head, channel and state-coordinate orbits."""

import numpy as np


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
