"""The chunked op's compositions with an auxiliary scan (`dew.nn.kernels.slots`): LightNet's and gated slot
attention's (GSA's, and ABC's where the slots' decay is their own running normalizer), each on par from
float64 with its recurrence a token at a time at the call's dtype (check 1), as fla's naive references
write them (fla/ops/gsa/naive.py, fla/ops/abc/naive.py and fla/ops/lightnet at 413ad2c)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import FEWEST_SEEDS, assert_as_exact

from dew.nn.kernels import slots

ROWS, LENGTH, HEADS, KEYS, SLOTS, VALUES = 2, 80, 2, 32, 16, 16
"""A chunk of the op's 64 tokens and a part one, over two rows and two heads."""


def _logcumsumexp(x, entering):
    """`additive_gate`'s running normalizer a token at a time, from `entering`."""
    def step(z, x_t):
        z = jnp.logaddexp(z, x_t)
        return z, z

    final, zs = jax.lax.scan(step, entering, jnp.moveaxis(x, 1, 0))
    return jnp.moveaxis(zs, 0, 1), final


def _gate(x, rounded=None):
    """fla's additive gate a token at a time: the weights `exp(x - z)` (rounded to `rounded` where given, as
    fla's gate stores them in its input's dtype, gsa/gate.py) and the log decays `z_{t-1} - z_t`, zero at a
    row's first token."""
    z, _ = _logcumsumexp(x, jnp.full(x[:, 0].shape, -jnp.inf, x.dtype))
    before = jnp.concatenate([z[:, :1], z[:, :-1]], axis=1)
    return _round(jnp.exp(x - z), rounded), before - z


def _round(x, rounded):
    """`x` rounded to `rounded` and back, or as it is where None."""
    return x if rounded is None else x.astype(rounded).astype(x.dtype)


def _recurrence(query, key, value, log_decay, along: str, scale, state):
    """`S_t = D_t S_{t-1} + k_t v_t^T`, read `scale q_t^T S_t`, a token at a time: `D_t` the decay along the
    keys (`along` 'keys') or along the values ('values')."""
    def step(memory, token):
        q, k, v, g = token
        decay = jnp.exp(g)
        memory = memory * (decay[..., :, None] if along == 'keys' else decay[..., None, :])
        memory = memory + k[..., :, None] * v[..., None, :]
        return memory, jnp.einsum('bhk,bhkv->bhv', q * scale, memory)

    tokens = tuple(jnp.moveaxis(x, 1, 0) for x in (query, key, value, log_decay))
    final, out = jax.lax.scan(step, state, tokens)
    return jnp.moveaxis(out, 0, 1), final


def lightnet_by_token(q, k, v, state, rounded=None):
    """LightNet's recurrence a token at a time (`slots.lightnet`'s), its keys' weights rounded to `rounded`
    where given (`_gate`)."""
    weights, log_decay = _gate(k, rounded)
    return _recurrence(q, weights, v, log_decay, 'keys', KEYS ** -0.5, state)


def slots_by_token(q, k, v, s, keys_memory, values_memory, g=None, rounded=None):
    """Gated slot attention a token at a time (fla's `naive_recurrent_gsa`; ABC's where `g` is None), the
    slots' weights and the softmax read rounded to `rounded` where given (as fla's chunked forms store
    them)."""
    if g is None:
        s, g = _gate(s, rounded)
    read, keys_final = _recurrence(q, k, s, g, 'values', KEYS ** -0.5, keys_memory)
    weights = _round(jax.nn.softmax(read, axis=-1), rounded)
    out, values_final = _recurrence(weights, s, v, g, 'keys', 1.0, values_memory)
    return out, (keys_final, values_final)


def differentiated(fn, count: int):
    """`fn` as `run(*operands, cotangents)`: its outputs and the gradients of its first `count` operands."""
    def run(*given):
        *operands, cotangents = given
        found, pull = jax.vjp(lambda *diff: fn(*diff, *operands[count:]), *operands[:count])
        return found, *pull(cotangents)

    return run


def at_dtype(fn):
    """`fn` with its operands as the call gives them, its outputs rounded to the values' dtype and its
    memories fp32, its arithmetic fp32, the weights it forms between its passes rounded to the call's dtype
    as fla's chunked forms store them: the token recurrence at the call's precision."""
    def run(*operands):
        widened = [x if x is None else x.astype(jnp.float32) for x in operands]
        dtype = operands[2].dtype
        out, finals = fn(*widened, rounded=None if dtype == jnp.float32 else dtype)
        return out.astype(dtype), finals

    return run


def _draws(dtype, *, form: str):
    """A seed's operands (unit-scale queries, keys and values in `dtype`; slots' writes normal and their log
    decays `log sigmoid(n + 2)` in fp32, GSA's; memories entering) and its outputs' cotangents last."""
    def draw(seed):
        rng = np.random.default_rng([6127, seed, len(form)])

        def normal(*shape, scale=1.0):
            return rng.standard_normal(shape) * scale

        q, k = (normal(ROWS, LENGTH, HEADS, KEYS, scale=KEYS ** -0.5) for _ in range(2))
        v = normal(ROWS, LENGTH, HEADS, VALUES)
        cast = (q.astype(dtype), k.astype(dtype), v.astype(dtype))
        if form == 'lightnet':
            state = normal(ROWS, HEADS, KEYS, VALUES).astype(np.float32)
            return (*cast, state, (normal(*v.shape).astype(dtype), normal(*state.shape).astype(np.float32)))
        s = normal(ROWS, LENGTH, HEADS, SLOTS).astype(dtype)
        g = (-np.logaddexp(0.0, -(normal(ROWS, LENGTH, HEADS, SLOTS) + 2.0))).astype(np.float32)
        memories = (normal(ROWS, HEADS, KEYS, SLOTS).astype(np.float32),
                    normal(ROWS, HEADS, SLOTS, VALUES).astype(np.float32))
        cotangents = (normal(*v.shape).astype(dtype),
                      tuple(normal(*m.shape).astype(np.float32) for m in memories))
        return (*cast, s, *memories, *((g,) if form == 'gsa' else ()), cotangents)
    return draw


def _lightnet(q, k, v, state):
    out, (final, _) = slots.lightnet(q, k, v, state=state)
    return out, final


def _slot_attention(q, k, v, s, keys_memory, values_memory, g=None):
    out, finals = slots.slot_attention(q, k, v, s, g, state=(keys_memory, values_memory))
    return out, finals[:2]


@pytest.mark.parametrize('dtype', [np.float32, jnp.bfloat16], ids=['fp32', 'bf16'])
def test_lightnet_is_as_exact_as_its_recurrence_token_by_token(dtype):
    """LightNet's recurrence (`slots.lightnet`): its output, final memory and the gradients of its queries,
    keys, values and entering memory on par from float64 with its recurrence a token at a time at the call's
    dtype (check 1)."""
    assert_as_exact(jax.jit(differentiated(_lightnet, 4)),
                    jax.jit(differentiated(at_dtype(lightnet_by_token), 4)), _draws(dtype, form='lightnet'),
                    f"LightNet in {jnp.dtype(dtype).name}", seeds=FEWEST_SEEDS,
                    truth_fn=jax.jit(differentiated(lightnet_by_token, 4)))


@pytest.mark.parametrize('form', ['gsa', 'abc'])
@pytest.mark.parametrize('dtype', [np.float32, jnp.bfloat16], ids=['fp32', 'bf16'])
def test_slot_attention_is_as_exact_as_its_recurrences_token_by_token(dtype, form):
    """Gated slot attention (`slots.slot_attention`), GSA's with its slots' log decays given and ABC's with
    their own running normalizer: its output, both final memories and the gradients of its queries, keys,
    values, slots' writes, both entering memories and (GSA's) slots' log decays on par from float64 with its
    two recurrences a token at a time at the call's dtype (check 1)."""
    count = 7 if form == 'gsa' else 6
    assert_as_exact(jax.jit(differentiated(_slot_attention, count)),
                    jax.jit(differentiated(at_dtype(slots_by_token), count)), _draws(dtype, form=form),
                    f"{form.upper()} in {jnp.dtype(dtype).name}", seeds=FEWEST_SEEDS,
                    truth_fn=jax.jit(differentiated(slots_by_token, count)))


def _equal(found, expected) -> bool:
    """Whether two trees' leaves are equal bit for bit (a normalizer's pair among them)."""
    pairs = zip(jax.tree.leaves(found), jax.tree.leaves(expected), strict=True)
    return all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in pairs)


NONE = (-jnp.inf, 0.0)
"""A normalizer of no writes (`additive_gate`'s pair)."""


def test_the_additive_gate_starts_each_row_at_its_own_write():
    """`slots.additive_gate` with no normalizer entering: a row's first token is its own normalizer, so its
    weight is exactly one and its log decay exactly zero (fla's convention: no decay before a first write),
    and a normalizer entering as none is none."""
    x = jnp.asarray(np.random.default_rng(6127).standard_normal((2, 40, 3)), jnp.float32)
    found = slots.additive_gate(x)
    weights, log_decay, _ = found
    assert np.all(np.asarray(weights[:, 0]) == 1.0) and np.all(np.asarray(log_decay[:, 0]) == 0.0)
    assert _equal(slots.additive_gate(x, tuple(jnp.full((2, 3), part) for part in NONE)), found)


def test_the_additive_gate_takes_absent_writes_with_zero_gradients():
    """`slots.additive_gate` of absent writes (`-inf` logits) from a normalizer entering at zero: their
    weights and log decays zero, the normalizer the one entering, and its log's gradient one to the
    normalizer's largest write entering and zero to every absent write (a NaN where its scan combined two
    absent writes before the seed)."""
    x = jnp.full((1, 2, 1), -jnp.inf, jnp.float32)
    entering = (jnp.zeros((1, 1), jnp.float32), jnp.zeros((1, 1), jnp.float32))
    weights, log_decay, last = slots.additive_gate(x, entering)
    assert np.all(np.asarray(weights) == 0.0) and np.all(np.asarray(log_decay) == 0.0)
    assert _equal(last, entering)

    def logged(x, largest):
        return slots.log_normalizer(slots.additive_gate(x, (largest, entering[1]))[2]).sum()

    d_x, d_entering = jax.grad(logged, argnums=(0, 1))(x, entering[0])
    assert np.all(np.asarray(d_x) == 0.0) and np.all(np.asarray(d_entering) == 1.0)


def test_a_call_of_no_tokens_returns_its_entering_states():
    """The gate and both compositions over no tokens: empty outputs, the normalizer and memories entering."""
    entering = (jnp.full((ROWS, HEADS, KEYS), 0.5, jnp.float32), jnp.full((ROWS, HEADS, KEYS), 0.125))
    weights, log_decay, last = slots.additive_gate(jnp.zeros((ROWS, 0, HEADS, KEYS)), entering)
    assert weights.shape == log_decay.shape == (ROWS, 0, HEADS, KEYS) and _equal(last, entering)
    empty = {width: jnp.zeros((ROWS, 0, HEADS, width), jnp.float32) for width in (KEYS, SLOTS, VALUES)}
    memory = jnp.ones((ROWS, HEADS, KEYS, VALUES), jnp.float32)
    out, finals = slots.lightnet(empty[KEYS], empty[KEYS], empty[VALUES], state=memory, normalizer=entering)
    assert out.shape == (ROWS, 0, HEADS, VALUES) and _equal(finals, (memory, entering))
    memories = (jnp.ones((ROWS, HEADS, KEYS, SLOTS)), jnp.ones((ROWS, HEADS, SLOTS, VALUES)),
                (jnp.full((ROWS, HEADS, SLOTS), 0.25), jnp.full((ROWS, HEADS, SLOTS), 0.5)))
    out, finals = slots.slot_attention(empty[KEYS], empty[KEYS], empty[VALUES], empty[SLOTS], state=memories)
    assert out.shape == (ROWS, 0, HEADS, VALUES) and _equal(finals, memories)


def test_the_additive_gate_takes_an_unseeded_absent_write_with_no_gradient():
    """An absent write (`-inf`) with no normalizer entering: the normalizer stays none, the write weighs
    zero, and the normalizer's gradient into it is zero (a both-absent combine a constant, not the larger of
    its two)."""
    x = jnp.full((1, 2, 1), -jnp.inf, jnp.float32)
    weights, log_decay, last = slots.additive_gate(x)
    assert np.all(np.asarray(weights) == 0.0) and np.all(np.asarray(log_decay) == 0.0)
    assert _equal(last, tuple(jnp.full((1, 1), part) for part in NONE))
    d_x = jax.grad(lambda x: slots.additive_gate(x)[2][1].sum())(x)
    assert np.all(np.asarray(d_x) == 0.0)


def test_the_additive_gate_keeps_a_faint_increment():
    """Unseeded writes `[0, -20]`: the normalizer's relative mass `log(1 + e^-20)`, about `2.061e-9`, and the
    second token's log decay its negation, each exactly as `jnp.logaddexp` (its `log1p`) forms them, where
    `log` of the rounded sum `1 + e^-20` reads zero."""
    x = jnp.asarray([[[0.0], [-20.0]]], jnp.float32)
    _, log_decay, (largest, mass) = slots.additive_gate(x)
    expected = np.asarray(jnp.logaddexp(jnp.float32(0.0), jnp.float32(-20.0)))
    assert expected > 0.0 and np.asarray(largest)[0, 0] == 0.0
    assert np.asarray(mass)[0, 0] == expected and np.asarray(log_decay)[0, 1, 0] == -expected


def test_an_abc_call_split_in_two_is_the_call_whole():
    """ABC's call of two tokens split into calls of one, each entering the state the first returns (its
    memories and its slots' normalizer), is the call whole bit for bit: two slots of zero logits (a softmax
    over them a half each), values 2 and 4 in both channels, so the second token weighs a half and the
    first decays by a half, both reading 3. Its normalizer dropped, the second call's gate started afresh
    and read 6 (a review's P0 of 1a60ffc17)."""
    q = jnp.ones((1, 2, 1, 2), jnp.float32)
    s = jnp.zeros((1, 2, 1, 2), jnp.float32)
    v = jnp.asarray([[2.0, 2.0], [4.0, 4.0]], jnp.float32).reshape(1, 2, 1, 2)
    whole, _ = slots.slot_attention(q, q, v, s)
    first, state = slots.slot_attention(q[:, :1], q[:, :1], v[:, :1], s[:, :1])
    second, _ = slots.slot_attention(q[:, 1:], q[:, 1:], v[:, 1:], s[:, 1:], state=state)
    assert np.array_equal(np.asarray(whole), np.asarray(jnp.concatenate([first, second], axis=1)))
    assert np.all(np.asarray(whole)[0, 1] == 3.0)


def _split(rule, operands: tuple, at: int):
    """`rule`'s reads of `operands` (time on axis 1) in two calls, split at token `at`, the second entering
    the state the first returns."""
    first, state = rule(*(x[:, :at] for x in operands))
    second, _ = rule(*(x[:, at:] for x in operands), state=state)
    return jnp.concatenate([first, second], axis=1)


def test_the_slot_forms_are_their_writes_shifted_by_a_common_large_offset_split_or_whole():
    """Integer writes and their copies shifted by 2^24's neighbourhood (2^20 for drawn integers, each shift
    exact in fp32; three equal writes at 2^24 itself), through the gate, LightNet and ABC, whole and split
    into a call of two tokens and one: the shifted reads are the unshifted's bit for bit, split or whole.
    The gate's relative masses keep the increments a normalizer in absolute scale rounds off: two equal
    writes at 2^24 weighed one where they weigh a half (a review's P0 of 1a60ffc17), and returned as one
    number between calls, `2^24 + log 2` rounded to `2^24`, a third token weighed a half where it weighs a
    third, ABC reading about 4.5 where it reads 4 (one of 969895771)."""
    drawn = np.random.default_rng(13500).integers(-4, 4, (ROWS, LENGTH, HEADS, KEYS))
    small = jnp.asarray(drawn, jnp.float32)
    shifted = small + 2.0 ** 20
    assert _equal(slots.additive_gate(small)[:2], slots.additive_gate(shifted)[:2])
    weights, _, _ = slots.additive_gate(jnp.full((1, 2, 1), 2.0 ** 24, jnp.float32))
    assert np.asarray(weights)[0, 1, 0] == 0.5
    q, v = (jnp.asarray(np.random.default_rng(seed).standard_normal((ROWS, LENGTH, HEADS, width)),
                        jnp.float32) for seed, width in ((1, KEYS), (2, VALUES)))

    def lightnet(q, k, v, state=None):
        memory, normalizer = (None, None) if state is None else state
        return slots.lightnet(q, k, v, state=memory, normalizer=normalizer)

    for read in (lambda k: lightnet(q, k, v)[0], lambda k: _split(lightnet, (q, k, v), LENGTH // 2)):
        assert _equal(read(small), read(shifted))
    ones = jnp.ones((1, 3, 1, 2), jnp.float32)
    values = jnp.asarray([[2.0, 2.0], [4.0, 4.0], [6.0, 6.0]], jnp.float32).reshape(1, 3, 1, 2)

    def abc(q, k, v, s, state=None):
        return slots.slot_attention(q, k, v, s, state=state)

    for logit in (0.0, 2.0 ** 24):
        logits = jnp.full((1, 3, 1, 2), logit, jnp.float32)
        found = [abc(ones, ones, values, logits)[0], _split(abc, (ones, ones, values, logits), 2)]
        if logit == 0.0:
            unshifted = found
        else:
            assert all(_equal(a, b) for a, b in zip(found, unshifted, strict=True))


