"""An optimizer's moments: their bias correction formed in double (`bias_corrected`), which `dew.optim`'s
Adam and `dew.laprop`'s LaProp both divide by; Adam's scaling over it (`scale_by_adam`); and the moments'
storage below float32, in bfloat16 with stochastic rounding (`bf16_moments`) or blockwise in 8 bits
(`int8_moments`), each read per stored layer of a stage-local run (`dew.nn.sharding.layers_of`)."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew._hashing import fmix32
from dew.nn.precision import stochastic_bf16
from dew.nn.sharding import layers_of


def bias_corrected(moment, decay, count):
    """`optax.tree.bias_correction`, with the correction of a Python `decay` in (0, 1) formed as
    -expm1(count log decay), its log taken in double before the trace.

    Optax forms 1 - decay**count inside a jit, where a Python decay is already rounded to float32:
    0.999 becomes 0.99900001, so the correction at count 1 is 1.3e-5 off (2.2e-4 at 0.9999), where
    torch forms it in double. A decay given as an array, a schedule's (`scheduled_solver`), takes
    optax's own. google-deepmind/optax#1827 proposed this for optax and closed unmerged; should a release
    form the correction so, delete this and `scale_by_adam` (optax's own then), and have
    `dew.laprop.scale_by_laprop`, which optax has no form of, divide by `optax.tree.bias_correction`.
    """
    if isinstance(decay, (int, float)) and 0 < decay < 1:
        correction = -jnp.expm1(count * math.log(decay))
        return jax.tree.map(lambda leaf: leaf / correction.astype(leaf.dtype), moment)
    return optax.tree.bias_correction(moment, decay, count)


def stochastic_round_bf16(x: jax.Array, seed: jax.Array) -> jax.Array:
    """`x` rounded to bf16 up or down with the probability of its distance to
    each, from a hash of `seed` and each element's flat index.

    Adding 16 uniform bits below the kept mantissa and truncating rounds up
    exactly when the discarded bits plus the noise carry, which is the
    discarded fraction's probability. The noise is a counter-based hash, so a
    step's rounding is a pure function of the seed and the position: no key
    is split or carried, and nothing is read from memory. jax.random.bits
    (threefry) made the whole update 1.75x slower than fp32 state on a TPU
    v6e and 1.05x on an L4; this costs a few integer ops per element. NaN
    stays NaN (`precision.stochastic_bf16`, its rounding).
    """
    x = x.astype(jnp.float32)
    index = jax.lax.iota(jnp.uint32, x.size).reshape(x.shape)
    return stochastic_bf16(x, fmix32(index * jnp.uint32(0x9E3779B1) + seed.astype(jnp.uint32)))


def moment_dtype(param, dtype):
    """The dtype a first moment of `param` is stored in under a requested `dtype`: the parameter's own
    where None, and for a complex parameter the complex type that holds `dtype`'s values in each
    component (complex64 for float32 or narrower), so its imaginary part is never dropped."""
    if dtype is None:
        return None
    if jnp.issubdtype(jnp.asarray(param).dtype, jnp.complexfloating):
        return jnp.promote_types(dtype, jnp.complex64)
    return dtype


def scale_by_adam(b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0, mu_dtype=None, *,
                  nesterov: bool = False, nu_dtype=None) -> optax.GradientTransformation:
    """`optax.scale_by_adam` (optax 0.2.8) with its bias corrections formed in double
    (`bias_corrected`): optax's moments, state layout (`optax.ScaleByAdamState`) and update,
    composed from optax's public tree math. `nu_dtype` stores the second moment as `mu_dtype` does
    the first, each the parameter's dtype where None; a gradient narrower than its moment is
    widened before either reads it, so an fp16 gradient's square does not underflow first."""
    mu_dtype = None if mu_dtype is None else jax.dtypes.canonicalize_dtype(mu_dtype)
    nu_dtype = None if nu_dtype is None else jax.dtypes.canonicalize_dtype(nu_dtype)

    def init_fn(params):
        # A complex parameter's first moment keeps both components (`moment_dtype`); its second, of the
        # gradient's squared magnitude, is real.
        return optax.ScaleByAdamState(count=jnp.zeros([], jnp.int32),
                                      mu=jax.tree.map(lambda p: jnp.zeros_like(p, moment_dtype(p, mu_dtype)),
                                                      params),
                                      nu=optax.tree.zeros_like(params, dtype=nu_dtype))

    def update_fn(updates, state, params=None):
        del params  # Adam reads only the gradients
        updates = jax.tree.map(lambda update, moment: None if update is None else update.astype(
            jnp.promote_types(update.dtype, moment.dtype)), updates, state.nu,
            is_leaf=lambda leaf: leaf is None)
        mu = optax.tree.update_moment(updates, state.mu, b1, 1)
        nu = optax.tree.update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count = optax.safe_increment(state.count)
        if nesterov:
            mu_hat = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g,
                                  bias_corrected(mu, b1, optax.safe_increment(count)),
                                  bias_corrected(updates, b1, count))
        else:
            mu_hat = bias_corrected(mu, b1, count)
        nu_hat = bias_corrected(nu, b2, count)
        scaled = jax.tree.map(lambda m, v: None if m is None else m / (jnp.sqrt(v + eps_root) + eps),
                              mu_hat, nu_hat, is_leaf=lambda leaf: leaf is None)
        # A requested dtype is each leaf's own (`moment_dtype`): a complex leaf's first moment stays
        # complex. None keeps the promoted moment, as Optax's does.
        mu = mu if mu_dtype is None else optax.tree.cast_like(mu, state.mu)
        return scaled, optax.ScaleByAdamState(count=count, mu=mu, nu=optax.tree.cast_like(nu, state.nu))

    return optax.GradientTransformation(init_fn, update_fn)


@dataclasses.dataclass(frozen=True)
class _Store:
    """How `_stored_moments` keeps Adam's moments: `zeros(params, signed)` a moment's stored zeros (the signed
    one Adam's first); `parts(structure, moment)` its stored leaves in the gradient's order; `decoded(part,
    signed)` one as float32; `encoded(moment, part, signed, salt)` a float32 one stored as `part` is, `salt` a
    uint32 of the step, the leaf and the moment; `whole(structure, parts)` the moment from its leaves."""
    zeros: Callable
    parts: Callable
    decoded: Callable
    encoded: Callable
    whole: Callable


def _stored_moments(b1: float, b2: float, eps: float, eps_root: float, nesterov: bool,
                    store: _Store) -> optax.GradientTransformation:
    """`scale_by_adam` with both moments kept as `store` keeps them. Each update runs optax's own update one
    leaf at a time, on that leaf's moments decoded to fp32, and stores the new ones: the step is optax's,
    only the storage is Dew's. Leaf by leaf keeps one leaf's fp32 moments live at a time; widening a bf16
    state whole first held all of them and raised the update's peak from 4.49 to 7.37 GB on the lm-dense
    tree (RTX 4080). The state keeps optax's `ScaleByAdamState`, its moments in `store`'s form."""
    inner = scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root, nesterov=nesterov)

    def init_fn(params):
        return optax.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=store.zeros(params, signed=True),
                                      nu=store.zeros(params, signed=False))

    def update_fn(updates, state, params=None):
        del params  # scale_by_adam reads none
        step = fmix32(jnp.asarray(state.count).astype(jnp.uint32) * jnp.uint32(0x27D4EB2F))
        leaves, structure = jax.tree.flatten(updates)
        mus, nus = store.parts(structure, state.mu), store.parts(structure, state.nu)
        scaled, new_mu, new_nu, count = [], [], [], state.count
        for index, (gradient, mu, nu) in enumerate(zip(leaves, mus, nus, strict=True)):
            # One array is a pytree of one leaf, so optax updates the leaf alone.
            update, one = inner.update(gradient, optax.ScaleByAdamState(
                count=state.count, mu=store.decoded(mu, signed=True), nu=store.decoded(nu, signed=False)))
            if not isinstance(one, optax.ScaleByAdamState):
                raise TypeError(f"scale_by_adam returned {type(one).__name__}")
            count = one.count
            salt = fmix32(step + jnp.uint32(2 * index + 1) * jnp.uint32(0x165667B1))
            scaled.append(update)
            new_mu.append(store.encoded(jnp.asarray(one.mu), mu, signed=True, salt=salt))
            new_nu.append(store.encoded(jnp.asarray(one.nu), nu, signed=False,
                                        salt=fmix32(salt + jnp.uint32(0x9E3779B9))))
        return structure.unflatten(scaled), optax.ScaleByAdamState(
            count=count, mu=store.whole(structure, new_mu), nu=store.whole(structure, new_nu))

    return optax.GradientTransformation(init_fn, update_fn)


def bf16_moments(b1: float, b2: float, eps: float, eps_root: float,
                 nesterov: bool) -> optax.GradientTransformation:
    """`scale_by_adam` with both moments stored in bf16 (`_stored_moments`), written back stochastically
    rounded (`stochastic_round_bf16`), seeded by the step count, the leaf and the moment. Round to nearest
    would lose every increment of the second moment smaller than half its bf16 spacing, which at b2 =
    0.999 is most of them; a stochastic rounding keeps each one in expectation. The moments are trees of
    the parameters' shapes, so sharding and checkpoints read them as they read fp32 state."""
    def zeros(params, signed: bool):
        # A complex parameter's first moment is kept whole (`_complex_moment`).
        return jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, _complex_moment(leaf, signed) or jnp.bfloat16),
                            params)

    def encoded(moment, part, signed, salt):
        return moment.astype(part.dtype) if _is_complex(part) else stochastic_round_bf16(moment, salt)

    return _stored_moments(b1, b2, eps, eps_root, nesterov, _Store(
        zeros=zeros,
        parts=lambda structure, moment: structure.flatten_up_to(moment),
        decoded=lambda part, signed: part if _is_complex(part) else part.astype(jnp.float32),
        encoded=encoded,
        whole=lambda structure, parts: structure.unflatten(parts)))


def _is_complex(leaf) -> bool:
    return jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.complexfloating)


def _complex_moment(leaf, signed: bool):
    """complex64 for a complex parameter's first moment (`signed`), whose imaginary part a real store would
    drop, kept whole by every reduced store; None otherwise. Its second moment, of the gradient's squared
    magnitude, is real and stored as any."""
    return jnp.complex64 if signed and _is_complex(leaf) else None


def dynamic_map(*, signed: bool) -> np.ndarray:
    """The 256 values an 8-bit moment's codes stand for, ascending: Dettmers' dynamic tree map
    (arXiv 1511.04561), bitsandbytes' and torchao's `create_dynamic_map(signed)` at 7 exponent bits, the
    signed one for Adam's first moment and the unsigned one for its second. Each decade from 1e-6 to 1
    holds a fraction's midpoints, more the larger the decade, beside 0 and 1."""
    values, exponent_bits, fraction_bits = [], 7, 7
    for exponent in range(exponent_bits):
        count = 2 ** (exponent + fraction_bits - exponent_bits + (0 if signed else 1)) + 1
        bounds = np.linspace(0.1, 1, count, dtype=np.float32)
        means = (bounds[:-1] + bounds[1:]) / np.float32(2)
        decade = np.float32(10.0 ** (exponent - exponent_bits + 1))
        values += list(decade * means) + (list(-decade * means) if signed else [])
    values += [0.0, 1.0]
    assert len(values) == 256
    return np.sort(np.asarray(values, np.float32))


class QuantizedMoments(NamedTuple):
    """A moment of an optimizer's state in 8 bits (`int8_moments`), as two trees of the parameters' own
    paths, so each leaf is placed as its parameter is: `codes`, each value's code, the index of its nearest
    `dynamic_map` value, in the parameter's shape (a parameter `int8_moments` keeps whole, its float32
    moment), and `scales`, each block's largest magnitude along the last axis, which the codes' values are
    fractions of (None where the moment is kept whole)."""
    codes: object
    scales: object


QUANTIZED_SIZE = 4096
"""The fewest values a moment holds in 8 bits (`int8_moments`), bitsandbytes' and torchao's: a smaller
one's scales cost more than its codes save, and it stays float32."""


def int8_moments(b1: float, b2: float, eps: float, eps_root: float, nesterov: bool,
                 block: int = 256) -> optax.GradientTransformation:
    """`scale_by_adam` with both moments stored in 8 bits, blockwise (Dettmers et al., arXiv 2110.02861, as
    torchao's `AdamW8bit` stores them, through `_stored_moments`): each block of `block` values along a
    leaf's last axis keeps its largest magnitude in float32 and each value the code of the nearest
    `dynamic_map` value to its fraction of it (`quantized`), the first moment's map signed and the
    second's unsigned. A leaf of fewer than `QUANTIZED_SIZE` values, or whose last axis `block` does not
    divide, keeps float32 moments. A moment's bytes fall from 4 a value to 1 and its block's share of a
    float32 scale, 1/64 a value at the default block."""
    maps = {signed: jnp.asarray(dynamic_map(signed=signed)) for signed in (True, False)}

    def held(path: jax.tree_util.KeyPath, leaf) -> bool:
        # A stage-local run's leaf is held as each of its stored layers would be (`layers_of`); a complex
        # parameter's moments are kept whole (`_complex_moment`).
        layers, stacked = layers_of(path)
        return (leaf.size // len(layers) >= QUANTIZED_SIZE and leaf.ndim > stacked
                and leaf.shape[-1] % block == 0 and not _is_complex(leaf))

    def zeros(params, signed: bool) -> QuantizedMoments:
        zero = _zero_code(dynamic_map(signed=signed))
        return QuantizedMoments(
            jax.tree_util.tree_map_with_path(
                lambda path, leaf: jnp.full(leaf.shape, zero, jnp.uint8) if held(path, leaf)
                else jnp.zeros(leaf.shape, _complex_moment(leaf, signed) or jnp.float32), params),
            jax.tree_util.tree_map_with_path(
                lambda path, leaf: jnp.zeros((*leaf.shape[:-1], leaf.shape[-1] // block), jnp.float32)
                if held(path, leaf) else None, params))

    def encoded(moment, part, signed, salt):
        return (moment, None) if part[1] is None else quantized(moment, block, signed=signed)

    return _stored_moments(b1, b2, eps, eps_root, nesterov, _Store(
        zeros=zeros,
        parts=lambda structure, moment: list(zip(structure.flatten_up_to(moment.codes),
                                                 structure.flatten_up_to(moment.scales), strict=True)),
        decoded=lambda part, signed: (part[0] if part[1] is None
                                      else dequantized(part[0], part[1], maps[signed], block)),
        encoded=encoded,
        whole=lambda structure, parts: QuantizedMoments(structure.unflatten([part[0] for part in parts]),
                                                        structure.unflatten([part[1] for part in parts]))))


def _zero_code(values: np.ndarray) -> int:
    """The code of 0 in a `dynamic_map`."""
    return int(np.flatnonzero(values == 0)[0])


def dequantized(codes: jax.Array, scales: jax.Array, values: jax.Array, block: int) -> jax.Array:
    """A moment stored in 8 bits as float32: its `codes`' `values` times their blocks' `scales`."""
    fractions = values[codes.astype(jnp.int32)]
    blocked = fractions.reshape(*fractions.shape[:-1], -1, block)
    return (blocked * scales[..., None]).reshape(fractions.shape)


CODE_WINDOW = 6
"""The codes around a value's estimated one (`_estimated_codes`) that `quantized` compares it with: the
estimate is within two of the code below the value, whose search they settle."""


def _estimated_codes(fractions: jax.Array, *, signed: bool) -> jax.Array:
    """Where each of `fractions` falls in `dynamic_map(signed=signed)`, to within two codes: its decade
    by its base-10 logarithm, and its place among the decade's evenly spaced values, the map's own
    layout (a decade `e` of 0 to 6 holds `2 ** e` values each side of 0, `2 ** (e + 1)` unsigned, the
    means of as many even steps from a tenth of the decade's top to its top)."""
    magnitude = jnp.abs(fractions)
    decade = jnp.clip(jnp.floor(jnp.log10(magnitude)) + 7, 0, 6).astype(jnp.int32)
    count = jnp.left_shift(1, decade + (0 if signed else 1))
    top = jnp.power(jnp.float32(10), (decade - 6).astype(jnp.float32))
    step = jnp.float32(0.9) / count.astype(jnp.float32)
    place = jnp.clip(jnp.floor((magnitude / top - jnp.float32(0.1)) / step), 0, count - 1).astype(jnp.int32)
    if not signed:
        return count - 1 + place  # past 0 and the 2 ** (e + 1) - 2 values of the decades below
    # 0 is code 127, each decade's positive values follow the 2 ** e - 1 of the decades below it, and
    # its negative values mirror them below 0.
    return jnp.where(fractions < 0, 127 - count - place, 127 + count + place)


def quantized(moment: jax.Array, block: int, *, signed: bool) -> tuple[jax.Array, jax.Array]:
    """`moment` in 8 bits, its codes and its blocks' scales: each block of `block` values along its last
    axis scaled by its largest magnitude (at least 1e-12), and each value coded by the nearest value of
    `dynamic_map(signed=signed)`, a value halfway between two taking the larger, as torchao's
    `quantize_8bit_with_qmap` does. The code below each value is found among the `CODE_WINDOW` around
    its estimate (`_estimated_codes`), each compared apart from the others: a search of the whole map
    ran its eight halvings one after another (docs/performance.md)."""
    values = jnp.asarray(dynamic_map(signed=signed))
    last = values.shape[0] - 1
    blocked = moment.reshape(*moment.shape[:-1], -1, block)
    scales = jnp.maximum(jnp.max(jnp.abs(blocked), axis=-1), 1e-12)
    fractions = (blocked / scales[..., None]).reshape(moment.shape)
    # The codes at most each value: those of the window's that are, past the codes below it. One code at
    # a time, the comparisons fuse with the coding; a window axis held six of each value's at once.
    first = _estimated_codes(fractions, signed=signed) - CODE_WINDOW // 2 - 1
    below = first
    for code in (first + offset for offset in range(1, CODE_WINDOW + 1)):
        below = below + ((code < 0) | ((code <= last) & (values[jnp.clip(code, 0, last)] <= fractions)))
    below = jnp.clip(below, 0, last)
    above = jnp.minimum(below + 1, last)
    nearest = jnp.where(fractions - values[below] >= (values[above] - values[below]) * 0.5, above, below)
    return nearest.astype(jnp.uint8), scales
