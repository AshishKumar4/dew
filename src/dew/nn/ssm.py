"""Mix tokens with S5 state-space layers and a 2D state fusion convolution.

The S5 layer is a diagonal SSM from the S4D-Lin poles, run in chunks of
pole-power products (`diagonal_recurrence`) on a GPU and the CPU, and by
`associative_scan` on a TPU (`_directions`). The fusion convolution is
Spatial-Mamba's, and the two together are the SSM mixer of `ModulatedBlock`.
"""


import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from .conv import Conv
from .precision import at_least_fp32


def hippo_a_imag_init(key, shape, dtype=jnp.float32):
    """S4D-Lin imaginary part: A_imag_n = pi * n."""
    state_dim = shape[0]
    n = jnp.arange(state_dim, dtype=dtype)
    return (jnp.pi * n).astype(dtype)


# Positions a chunk of `diagonal_recurrence` covers.
SCAN_CHUNK = 16


def _complex_blocks(re: jax.Array, im: jax.Array) -> jax.Array:
    """The real form `[[re, -im], [im, re]]` of complex matrices `[M, K, ...]`:
    their product with `[re; im]` stacked along K is the complex product,
    stacked the same way along M."""
    return jnp.concatenate([jnp.concatenate([re, -im], 1), jnp.concatenate([im, re], 1)], 0)


def _powers(pole: jax.Array, length: int) -> jax.Array:
    """`pole ** t` for t = 0 .. length, `[length + 1, N]`, by doubling: the
    powers so far times the next squared power, so each is a product of at
    most 2 log2(length) roundings rather than t, in log2(length) elementwise
    steps rather than a serial running product."""
    powers = jnp.ones((1, pole.shape[0]), pole.dtype)
    factor = pole
    while powers.shape[0] <= length:
        powers = jnp.concatenate([powers, powers * factor])
        factor = factor * factor
    return powers[:length + 1]


def _carried(pole: jax.Array, last: jax.Array) -> jax.Array:
    """The state each chunk starts from, `[B, 2, C, N]` in real form: zero
    for the first chunk, then `e_(c-1)` of `e_c = pole * e_(c-1) + last_c`,
    run by `associative_scan` over the chunks' own last states `last`
    `[B, 2, C, N]` with `pole` the `[N]` complex power that spans one chunk."""
    def combine(earlier, later):
        a_re, a_im, b_re, b_im = earlier
        c_re, c_im, d_re, d_im = later
        return (a_re * c_re - a_im * c_im, a_re * c_im + a_im * c_re,
                c_re * b_re - c_im * b_im + d_re, c_re * b_im + c_im * b_re + d_im)

    chunks = last.shape[2]
    span = (jnp.broadcast_to(pole.real, (1, chunks, pole.shape[0])),
            jnp.broadcast_to(pole.imag, (1, chunks, pole.shape[0])))
    *_, end_re, end_im = jax.lax.associative_scan(combine, (*span, last[:, 0], last[:, 1]), axis=1)
    ends = jnp.stack([end_re, end_im], axis=1)
    return jnp.pad(ends[:, :, :-1], ((0, 0), (0, 0), (1, 0), (0, 0)))


def diagonal_recurrence(pole: jax.Array, inputs: jax.Array, chunk: int = SCAN_CHUNK) -> jax.Array:
    """The states `x_k = pole * x_(k-1) + v_k`, from `x = 0` before the
    first position, of N diagonal complex recurrences over axis 1, in real
    arithmetic, in chunks of `chunk` positions.

    `pole` is `[N]` complex. `inputs` is `v` `[B, S, 2N]` in real form, its
    real parts and then its imaginary parts on the last axis, and the states
    come back the same way. Inside a chunk the states are one product of the
    powers `pole^(k - j)` with the chunk's inputs, at fp32's full precision,
    and across chunks `associative_scan` carries each chunk's last state
    forward by `pole^chunk`. The powers come by doubling (`_powers`), and
    the states stay within the fp32 running-error bound of the recurrence
    (tests/test_ssm.py).
    """
    batch, steps, width = inputs.shape
    states = width // 2
    length = min(chunk, steps)
    chunks = -(-steps // length)
    highest = jax.lax.Precision.HIGHEST
    powers = _powers(pole, length)
    blocks = jnp.pad(inputs, ((0, 0), (0, chunks * length - steps), (0, 0)))
    blocks = blocks.reshape(batch, chunks, length, 2, states).transpose(0, 1, 3, 2, 4)
    # Entry (k, j) is pole^(k - j) where k >= j, and zero above the diagonal:
    # a one-hot product, exact at full precision, whose transpose is a
    # product too rather than a scatter or a stack of shifted copies.
    lag = np.arange(length)[:, None] - np.arange(length)[None, :]
    picks = (lag[..., None] == np.arange(length)).astype(np.float32)
    within_re, within_im = (jnp.einsum("kjt,tn->kjn", picks, part, precision=highest)
                            for part in (powers[:length].real, powers[:length].imag))
    local = jnp.einsum("kjn,bcjn->bckn", _complex_blocks(within_re, within_im),
                       blocks.reshape(batch, chunks, 2 * length, states), precision=highest)
    local = local.reshape(batch, chunks, 2, length, states)
    if chunks > 1:
        carry = _carried(powers[-1], local[:, :, :, -1].transpose(0, 2, 1, 3))[:, :, :, None]
        # The state a chunk starts from, carried to each of its positions by pole^(k + 1).
        lift = powers[1:]
        local = local + jnp.stack([lift.real * carry[:, 0] - lift.imag * carry[:, 1],
                                   lift.real * carry[:, 1] + lift.imag * carry[:, 0]], axis=2)
    return local.transpose(0, 1, 3, 2, 4).reshape(batch, chunks * length, width)[:, :steps]


class S5Layer(nn.Module):
    """Run a diagonal complex state-space recurrence over `[B, S, F]` inputs.

        x_k = A x_{k-1} + B u_k
        y_k = Re(C x_k) + D u_k

    `A` is `state_dim` complex poles, stored as the log of the negative
    real part so the recurrence cannot grow. `dt` discretizes them per
    pole, and the scan runs in at least fp32 whatever dtype the input
    carries (`at_least_fp32`).

    The parameters are the poles `log_A_real`/`A_imag`, the input map
    `B_re`/`B_im`, the output map `C_re`/`C_im`, the skip `D` and the
    per-pole step `log_dt`.
    """
    features: int
    state_dim: int = 64
    dtype: Dtype | None = None

    def setup(self):
        # A: diagonal complex state matrix, S4D-Lin init, parameterized as
        # log of the negative real part for stability: A_real_n = -1/2 for
        # every state (Gu, Gupta, Goel and Re 2022, section 4; S5's HiPPO-N
        # poles share it, Smith et al. 2023, section 4.1)
        self.log_A_real = self.param('log_A_real', nn.initializers.constant(jnp.log(0.5), jnp.float32),
                                     (self.state_dim,))
        self.A_imag = self.param('A_imag', hippo_a_imag_init, (self.state_dim,))
        # B: input-to-state projection [state_dim, F]
        self.B_re = self.param('B_re', nn.initializers.lecun_normal(), (self.state_dim, self.features))
        self.B_im = self.param('B_im', nn.initializers.lecun_normal(), (self.state_dim, self.features))
        # C: state-to-output projection [F, state_dim], lecun_normal as in S5
        self.C_re = self.param('C_re', nn.initializers.lecun_normal(), (self.features, self.state_dim))
        self.C_im = self.param('C_im', nn.initializers.lecun_normal(), (self.features, self.state_dim))
        # D: skip connection, N(0,1) per channel as in S5
        self.D = self.param('D', nn.initializers.normal(stddev=1.0), (self.features,))
        # dt: discretization timestep, learned per state dim so each state
        # channel can model its own time scale, drawn log-uniform in [0.001, 0.1]
        self.log_dt = self.param(
            'log_dt',
            lambda key, shape: jax.random.uniform(key, shape, minval=jnp.log(0.001), maxval=jnp.log(0.1)),
            (self.state_dim,))

    def operators(self) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """The zero-order hold of the recurrence, `(A_bar, B_bar, C, D)`:
        `A_bar = exp(A dt)` the `[N]` complex poles, `B_bar = (A_bar - 1) /
        A * B` `[N, F]` and `C` `[F, N]` complex, and `D` the `[F]` skip."""
        dt = jnp.exp(self.log_dt)
        A_diag = -jnp.exp(self.log_A_real) + 1j * self.A_imag
        A_bar = jnp.exp(A_diag * dt)
        B_bar = ((A_bar[:, None] - 1.0) / (A_diag[:, None] + 1e-8)) * (self.B_re + 1j * self.B_im)
        return A_bar, B_bar, self.C_re + 1j * self.C_im, self.D

    def __call__(self, u):
        """Scan `u` `[B, S, F]` through the discretized poles into `[B, S, F]`."""
        F = u.shape[-1]
        assert self.features == F, f"S5Layer built for {self.features} features, got {F}"
        y, = _directions(u, (self.operators(),), (False,), self.dtype if self.dtype is not None else u.dtype)
        return y


def _directions(u: jax.Array, operators, reversed_: tuple[bool, ...], dtype) -> list[jax.Array]:
    """Each direction's `Re(C x_k) + D u_k` in `dtype` over `u` `[B, S, F]`,
    read in reverse where `reversed_` says, computed in at least fp32 by
    backend. The split is measured, not chosen (docs/performance.md, the S5
    table).

    A GPU and the CPU run `_chunked_directions`, which won every shape
    measured there. A TPU runs `_scanned_directions`, the layer's first form,
    unchanged: on a v6e the chunks were faster at a batch of 32 and sampling
    one image but slower at 16 and sampling 4, and the real input product
    alone left the step at 16 1% slower.
    """
    return jax.lax.platform_dependent(
        u, tpu=lambda u: _scanned_directions(u, operators, reversed_, dtype),
        default=lambda u: _chunked_directions(u, operators, reversed_, dtype))


def _chunked_directions(u: jax.Array, operators, reversed_: tuple[bool, ...], dtype) -> list[jax.Array]:
    """`_directions` by `diagonal_recurrence`, every direction's complex
    input product as one real one, the backward recurrence over the reversed
    positions of its projected inputs rather than over a reversed copy of
    `u`."""
    u = u.astype(at_least_fp32(u.dtype))
    width = 2 * operators[0][0].shape[0]
    stacked = jnp.concatenate([jnp.concatenate([b.real, b.imag]) for _, b, _, _ in operators])
    projected = jnp.einsum('bsf,nf->bsn', u, stacked)
    out = []
    for index, ((pole, _, c, d), backwards) in enumerate(zip(operators, reversed_, strict=True)):
        v = projected[..., index * width:(index + 1) * width]
        x = (jnp.flip(diagonal_recurrence(pole, jnp.flip(v, axis=1)), axis=1) if backwards
             else diagonal_recurrence(pole, v))
        # The k-th output is Re(C x_k), the real form's [C_re, -C_im] product.
        y = jnp.einsum('bsn,fn->bsf', x, jnp.concatenate([c.real, -c.imag], -1)) + d * u
        out.append(y.astype(dtype))
    return out


def _scanned_directions(u: jax.Array, operators, reversed_: tuple[bool, ...], dtype) -> list[jax.Array]:
    """`_directions` as the layer first ran it: per direction, a complex input
    product over `u` (reversed where the direction reads backwards) and an
    `associative_scan` over complex states, `(a1, b1) * (a2, b2) = (a1 a2,
    a2 b1 + b2)`."""
    out = []
    for (pole, b, c, d), backwards in zip(operators, reversed_, strict=True):
        read = jnp.flip(u, axis=1) if backwards else u
        read = read.astype(at_least_fp32(read.dtype))
        bu = jnp.einsum('bsf,nf->bsn', read, b)
        _, x = jax.lax.associative_scan(lambda e1, e2: (e1[0] * e2[0], e2[0] * e1[1] + e2[1]),
                                        (jnp.broadcast_to(pole[None, None, :], bu.shape), bu), axis=1)
        y = (jnp.einsum('fn,bsn->bsf', c, x).real + d[None, None, :] * read).astype(dtype)
        out.append(jnp.flip(y, axis=1) if backwards else y)
    return out


class BidirectionalS5Layer(nn.Module):
    """Runs forward and backward S5 scans, concats and projects back to features.
    Patches have no inherent direction, so scan both ways.

    The directions run by backend (`_directions`).
    """
    features: int
    state_dim: int = 64
    dtype: Dtype | None = None

    @nn.compact
    def __call__(self, u):
        # The input u has shape [B, S, F].
        operators = tuple(S5Layer(features=self.features, state_dim=self.state_dim, dtype=self.dtype,
                                  name=name).operators() for name in ("s5_forward", "s5_backward"))
        dtype = self.dtype if self.dtype is not None else u.dtype
        y_fwd, y_bwd = _directions(u, operators, (False, True), dtype)
        y_cat = jnp.concatenate([y_fwd, y_bwd], axis=-1)  # [B, S, 2F]
        return nn.Dense(features=self.features, dtype=self.dtype, name="out_proj")(y_cat)


class SpatialFusionConv(nn.Module):
    """Multi-dilation depthwise 2D convs summed as a residual over the SSM output grid.
    The 1D scan scrambles 2D locality; this recovers a direction-balanced local
    receptive field. Kernels are zero-init so the fusion starts as a pass-through.
    """
    features: int
    dilations: tuple[int, ...] = (1, 2, 3)
    kernel_size: int = 3
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, y_2d):
        # y_2d: [B, H_P, W_P, F], SSM output reshaped to a row-major grid
        out = y_2d
        for dil in self.dilations:
            dw = Conv(
                features=self.features,
                kernel_size=(self.kernel_size, self.kernel_size),
                strides=(1, 1),
                padding='SAME',
                kernel_dilation=(dil, dil),
                feature_group_count=self.features,  # depthwise
                use_bias=False,
                kernel_init=nn.initializers.zeros,
                dtype=self.dtype,
                precision=self.precision,
                name=f"dwconv_dil{dil}",
            )(y_2d)
            out = out + dw
        return out
