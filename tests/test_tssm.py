"""TransDreamer's dynamics (`dew.objectives.rl.tssm.TSSM`), held to a transformer written out here from its
weights apart from Dew's modules (`written_states`): a window's states, a step at a time over the acting
cache and imagination from each step of a window over its cache, each to float64 rounding, and the
window's states in bfloat16 on par with the written transformer's (check 1). A state reads its episode up
to itself and nothing after, bit for bit."""

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import assert_as_exact, assert_computes_the_oracle, chain_roundings, seeds_for

from dew.objectives.rl.tssm import TSSM

ROWS, LENGTH, WIDTH, HEADS = 3, 7, 16, 2


def tiny(dtype=jnp.float32) -> TSSM:
    return TSSM(action_size=2, stoch=3, classes=4, width=WIDTH, layers=2, heads=HEADS, dtype=dtype)


def variables(model: TSSM, seed: int = 0):
    """The model's weights from its initialization, each moved by a draw so every norm's scale and every
    bias matters."""
    held = model.init(jax.random.key(seed), jnp.zeros((1, 2, 5)), jnp.zeros((1, 2, 2)),
                      jnp.zeros((1, 2), bool))
    rng = np.random.default_rng([2202, 9, seed])
    return jax.tree.map(lambda leaf: leaf + rng.normal(0, 0.1, leaf.shape).astype(leaf.dtype), held)


def window(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A window's inputs `[3, 7, width]` and where episodes start: row 1 at step 3, row 2 at step 0."""
    first = np.zeros((ROWS, LENGTH), bool)
    first[1, 3] = first[2, 0] = True
    return np.random.default_rng([2202, seed]).normal(size=(ROWS, LENGTH, WIDTH)), first


def written_states(params, x, first):
    """TransDreamer's states of a window's inputs `x` `[B, T, width]`, written out from the weights in
    `x`'s dtype, norms, rotary angles and softmax in at least float32: each pre-norm block's rotary,
    RMS-normed attention over the steps of each step's episode up to itself, then its SiLU-gated
    feed-forward layer, each added to its input; a final RMS norm."""
    dtype = x.dtype
    wide = jnp.promote_types(dtype, jnp.float32)
    params = params["params"]

    def norm(values, scale):
        values = values.astype(wide)
        return (values / jnp.sqrt(jnp.mean(values * values, -1, keepdims=True) + 1e-6) * scale).astype(dtype)

    def dense(values, layer, contract=1):
        kernel = layer["kernel"].astype(dtype)
        axes = (list(range(values.ndim - contract, values.ndim)), list(range(contract)))
        return jnp.tensordot(values, kernel, axes) + layer["bias"].astype(dtype)

    rows, length = first.shape
    head = WIDTH // HEADS
    episode = np.zeros((rows, length), int)
    position = np.zeros((rows, length), int)
    for row in range(rows):
        for step in range(length):
            starts = step == 0 or first[row, step]
            episode[row, step] = episode[row, step - 1] + starts if step else 0
            position[row, step] = 0 if starts else position[row, step - 1] + 1
    mask = (episode[:, :, None] == episode[:, None, :]) & (np.arange(length)[None, :, None]
                                                           >= np.arange(length)[None, None, :])
    angles = position[..., None].astype(np.float64) / 10000.0 ** (np.arange(0, head, 2) / head)
    cos, sin = (jnp.asarray(f(angles), wide)[:, :, None] for f in (np.cos, np.sin))

    def rotated(values):
        values = values.astype(wide)
        first_half, second_half = values[..., :head // 2], values[..., head // 2:]
        return jnp.concatenate([first_half * cos - second_half * sin, second_half * cos + first_half * sin],
                               -1).astype(dtype)

    for layer in range(2):
        block = params[f"blocks_{layer}"]
        h = norm(x, block["attention_norm"]["scale"])
        query = rotated(norm(dense(h, block["query"]), block["query_norm"]["scale"]))
        key = rotated(norm(dense(h, block["key"]), block["key_norm"]["scale"]))
        value = dense(h, block["value"])
        logits = jnp.einsum("btnh,bsnh->bnts", query.astype(wide), key.astype(wide)) / np.sqrt(head)
        logits = jnp.where(jnp.asarray(mask)[:, None], logits, -jnp.inf)
        weights = jax.nn.softmax(logits, -1).astype(dtype)
        attended = jnp.einsum("bnts,bsnh->btnh", weights, value)
        x = x + dense(attended, block["out"], contract=2)
        h = norm(x, block["feed_norm"]["scale"])
        x = x + dense(jax.nn.silu(dense(h, block["gate"])) * dense(h, block["up"]), block["down"])
    return norm(x, params["out_norm"]["scale"])


def test_a_windows_states_are_the_written_transformers():
    """A window's states (`TSSM.states`), its episodes restarting where they start, are the written
    transformer's to float64 rounding."""
    with jax.enable_x64(new_val=True):
        model = tiny(jnp.float64)
        held = variables(model)
        x, first = window(0)
        states = jax.jit(lambda x: model.apply(held, x, jnp.asarray(first), method=TSSM.states))
        roundings = chain_roundings(jax.make_jaxpr(states)(x))
        written = np.asarray(written_states(held, jnp.asarray(x), first))
        assert_computes_the_oracle(np.asarray(states(x)), written, "a TSSM's states", roundings=roundings)


def test_a_windows_states_in_bfloat16_are_on_par_with_the_written_transformers():
    """In bfloat16 over weights drawn in bfloat16, a window's states are on par from float64 with the
    written transformer's in bfloat16 (check 1)."""
    model = tiny(jnp.bfloat16)
    first = window(0)[1]

    def draw(seed):
        held = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.bfloat16), variables(tiny(), seed))
        return held, jnp.asarray(window(100 + seed)[0], jnp.bfloat16)

    def dew(held, x):
        return model.apply(held, x, jnp.asarray(first), method=TSSM.states)

    def written(held, x):
        return written_states(held, x, first)

    label = "a TSSM's states in bfloat16"
    assert_as_exact(dew, written, draw, label, seeds=seeds_for(written, draw, label, dew_fn=dew))


def test_a_state_reads_its_episode_up_to_itself_and_nothing_after():
    """Changing a window's inputs from step t on leaves every state before t as it was, bit for bit, and
    changes the states from t; changing the steps before an episode's start leaves that episode's states
    as they were."""
    model = tiny()
    held = variables(model)
    x, first = window(0)
    states = jax.jit(lambda x: model.apply(held, x, jnp.asarray(first), method=TSSM.states))
    base = np.asarray(states(x))
    for t in range(LENGTH):
        changed = x.copy()
        changed[:, t:] += 1.0
        moved = np.asarray(states(changed))
        np.testing.assert_array_equal(moved[:, :t], base[:, :t])
        assert not np.array_equal(moved[:, t], base[:, t])
    changed = x.copy()
    changed[1, :3] += 1.0
    np.testing.assert_array_equal(np.asarray(states(changed))[1, 3:], base[1, 3:])


def test_a_step_at_a_time_over_its_cache_computes_the_written_transformers_states():
    """Acting's path, a step at a time from an empty cache, its episodes restarting where they start,
    computes the written transformer's states over the window, to float64 rounding."""
    with jax.enable_x64(new_val=True):
        model = tiny(jnp.float64)
        held = variables(model)
        x, first = window(1)

        def stepped(x):
            cache = model.apply(held, ROWS, LENGTH, method=TSSM.empty_cache)
            found = []
            for t in range(LENGTH):
                h, cache = model.apply(held, x[:, t], jnp.asarray(first[:, t]), cache, method=TSSM.step)
                found.append(h)
            return jnp.stack(found, 1)

        roundings = chain_roundings(jax.make_jaxpr(stepped)(x))
        written = np.asarray(written_states(held, jnp.asarray(x), first))
        assert_computes_the_oracle(np.asarray(stepped(x)), written, "a TSSM's states a step at a time",
                                   roundings=roundings)


def test_imagination_from_each_step_reads_the_window_up_to_it():
    """Imagination's first two steps from each step k of a window, over the window's cache (the window held
    once for every start, beside each start's own imagined steps), compute the written transformer's states
    over the window up to k extended by the imagined steps' inputs, to float64 rounding."""
    with jax.enable_x64(new_val=True):
        model = tiny(jnp.float64)
        held = variables(model)
        x, first = window(2)
        following = np.random.default_rng([2202, 3]).normal(size=(2, ROWS * LENGTH, WIDTH))

        def imagined(following):
            cache = model.apply(held, x, first, 2, method=TSSM.window_cache)
            found = []
            for step in range(2):
                h, cache = model.apply(held, following[step], jnp.zeros(ROWS * LENGTH, bool), cache,
                                       method=TSSM.step)
                found.append(h)
            return jnp.stack(found, 1)

        expected = []
        for row in range(ROWS):
            for k in range(LENGTH):
                extended = np.concatenate([x[row, :k + 1], following[:, row * LENGTH + k]])[None]
                starts = np.concatenate([first[row, :k + 1], [False, False]])[None]
                expected.append(np.asarray(written_states(held, jnp.asarray(extended), starts))[0, -2:])
        roundings = chain_roundings(jax.make_jaxpr(imagined)(following))
        assert_computes_the_oracle(np.asarray(imagined(following)), np.stack(expected),
                                   "imagination from a window's steps", roundings=roundings)


def test_imagination_that_restarts_reads_nothing_of_its_window():
    """Imagination from each step of a window where every third start restarts its episode at the first
    imagined step: those compute the written transformer's states over the imagined inputs alone, at that
    step and the next, the window's history read no more; the rest over the window up to their start
    extended by the imagined inputs; to float64 rounding."""
    with jax.enable_x64(new_val=True):
        model = tiny(jnp.float64)
        held = variables(model)
        x, first = window(4)
        following = np.random.default_rng([2202, 5]).normal(size=(2, ROWS * LENGTH, WIDTH))
        restarts = np.arange(ROWS * LENGTH) % 3 == 0

        def imagined(following):
            cache = model.apply(held, x, first, 2, method=TSSM.window_cache)
            found = []
            for step, starts in enumerate((restarts, np.zeros_like(restarts))):
                h, cache = model.apply(held, following[step], jnp.asarray(starts), cache, method=TSSM.step)
                found.append(h)
            return jnp.stack(found, 1)

        expected = []
        for row in range(ROWS):
            for k in range(LENGTH):
                start = row * LENGTH + k
                if restarts[start]:
                    extended, starts = following[:, start][None], np.asarray([[True, False]])
                else:
                    extended = np.concatenate([x[row, :k + 1], following[:, start]])[None]
                    starts = np.concatenate([first[row, :k + 1], [False, False]])[None]
                expected.append(np.asarray(written_states(held, jnp.asarray(extended), starts))[0, -2:])
        roundings = chain_roundings(jax.make_jaxpr(imagined)(following))
        assert_computes_the_oracle(np.asarray(imagined(following)), np.stack(expected),
                                   "imagination that restarts", roundings=roundings)
