"""Mix tokens with S5 state-space layers and a 2D state fusion convolution.

The S5 layer is a diagonal SSM from the S4D-Lin poles, run in chunks of
pole-power products (`diagonal_recurrence`). The fusion convolution is
Spatial-Mamba's, and the two together are the SSM mixer of `ModulatedBlock`.
"""


import jax
import jax.numpy as jnp
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


def _powers(log_pole: jax.Array, exponents: jax.Array) -> jax.Array:
    """`pole ** exponents` for each of the N poles, `[*exponents.shape, N]`,
    as `exp(exponent * log(pole))`: elementwise, so a TPU spends no serial
    product or shifted-copy assembly on it, and each power rounds once
    rather than once per factor of a running product."""
    return jnp.exp(exponents[..., None].astype(log_pole.real.dtype) * log_pole)


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
    first position, of N diagonal complex recurrences over axis 1.

    `pole` is `[N]` complex. `inputs` is `v` `[B, S, 2N]` in real form, its
    real parts and then its imaginary parts on the last axis, and the states
    come back the same way, computed in real arithmetic. Positions run in
    chunks of `chunk`: inside a chunk the states are one product of the
    powers `pole^(k - j)` with the chunk's inputs, at fp32's full precision,
    and across chunks `associative_scan` carries each chunk's last state
    forward by `pole^chunk`. The powers are `exp(t log(pole))`, and the
    states stay within the fp32 running-error bound of the scan the layer
    ran before (tests/test_ssm.py).
    """
    batch, steps, width = inputs.shape
    states = width // 2
    length = min(chunk, steps)
    chunks = -(-steps // length)
    highest = jax.lax.Precision.HIGHEST
    log_pole = jnp.log(pole)
    blocks = jnp.pad(inputs, ((0, 0), (0, chunks * length - steps), (0, 0)))
    blocks = blocks.reshape(batch, chunks, length, 2, states).transpose(0, 1, 3, 2, 4)
    # Entry (k, j) is pole^(k - j) where k >= j, and zero above the diagonal.
    lag = jnp.arange(length)[:, None] - jnp.arange(length)[None, :]
    within = jnp.where((lag >= 0)[..., None], _powers(log_pole, jnp.maximum(lag, 0)), 0)
    local = jnp.einsum("kjn,bcjn->bckn", _complex_blocks(within.real, within.imag),
                       blocks.reshape(batch, chunks, 2 * length, states), precision=highest)
    local = local.reshape(batch, chunks, 2, length, states)
    if chunks > 1:
        carry = _carried(_powers(log_pole, jnp.asarray(length)), local[:, :, :, -1].transpose(0, 2, 1, 3))
        carry = carry[:, :, :, None]
        # The state a chunk starts from, carried to each of its positions by pole^(k + 1).
        lift = _powers(log_pole, jnp.arange(1, length + 1))
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
        """The zero-order hold of the recurrence, `(A_bar, B, C, D)`.

        `A_bar = exp(A dt)` is the `[N]` complex poles. The complex products
        with u and with the states run as real ones over stacked real and
        imaginary parts, so `B` is `[Re B_bar; Im B_bar]` `[2N, F]`, with
        `B_bar = (A_bar - 1) / A * B`, and `C` is `[C_re, -C_im]` `[F, 2N]`,
        whose product with real-form states is `Re(C x)`. `D` is the `[F]`
        skip.
        """
        dt = jnp.exp(self.log_dt)
        A_diag = -jnp.exp(self.log_A_real) + 1j * self.A_imag
        A_bar = jnp.exp(A_diag * dt)
        B_bar = ((A_bar[:, None] - 1.0) / (A_diag[:, None] + 1e-8)) * (self.B_re + 1j * self.B_im)
        return (A_bar, jnp.concatenate([B_bar.real, B_bar.imag]),
                jnp.concatenate([self.C_re, -self.C_im], -1), self.D)

    def __call__(self, u):
        """Scan `u` `[B, S, F]` through the discretized poles into `[B, S, F]`."""
        F = u.shape[-1]
        assert self.features == F, f"S5Layer built for {self.features} features, got {F}"
        A_bar, B_bar, C, D = self.operators()
        u_float = u.astype(at_least_fp32(u.dtype))
        x = diagonal_recurrence(A_bar, jnp.einsum('bsf,nf->bsn', u_float, B_bar))
        # The k-th output is Re(C x_k) plus the skip D u_k.
        y = jnp.einsum('bsn,fn->bsf', x, C) + D * u_float
        return y.astype(self.dtype) if self.dtype is not None else y.astype(u.dtype)


class BidirectionalS5Layer(nn.Module):
    """Runs forward and backward S5 scans, concats and projects back to features.
    Patches have no inherent direction, so scan both ways.

    Both directions read `u` in its own order through one input product,
    and the backward recurrence runs over the reversed positions of its
    projected inputs, the narrow `[B, S, 2N]` stream, rather than over a
    reversed copy of `u`.
    """
    features: int
    state_dim: int = 64
    dtype: Dtype | None = None

    @nn.compact
    def __call__(self, u):
        # The input u has shape [B, S, F].
        (pole_fwd, b_fwd, c_fwd, d_fwd), (pole_bwd, b_bwd, c_bwd, d_bwd) = (
            S5Layer(features=self.features, state_dim=self.state_dim, dtype=self.dtype,
                    name=name).operators()
            for name in ("s5_forward", "s5_backward"))
        width = 2 * self.state_dim
        u_float = u.astype(at_least_fp32(u.dtype))
        projected = jnp.einsum('bsf,nf->bsn', u_float, jnp.concatenate([b_fwd, b_bwd]))
        x_fwd = diagonal_recurrence(pole_fwd, projected[..., :width])
        x_bwd = jnp.flip(diagonal_recurrence(pole_bwd, jnp.flip(projected[..., width:], axis=1)), axis=1)
        dtype = self.dtype if self.dtype is not None else u.dtype
        y_fwd = (jnp.einsum('bsn,fn->bsf', x_fwd, c_fwd) + d_fwd * u_float).astype(dtype)
        y_bwd = (jnp.einsum('bsn,fn->bsf', x_bwd, c_bwd) + d_bwd * u_float).astype(dtype)
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
