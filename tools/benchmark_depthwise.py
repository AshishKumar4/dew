#!/usr/bin/env python3
"""Measure 3x3 depthwise forward/VJP and the published 176M DiT's step.

RTX 4080: ~/.cache/dew/dew-gpu-run env PYTHONPATH=src python tools/benchmark_depthwise.py kernels
Run `step --implementation lax` and `step --implementation production` in
separate processes. The diagnostic lax choice runs lax's dilated
convolution where production runs its dilated forms (polyphase in bf16, the
materialized shifted products in fp32): each Dew convolution traced within
`dilated_form` takes the form as its argument (`dew.nn.conv`), and
production dispatch has no performance flag.
"""

import argparse
import contextlib
import json
import statistics
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

import dew.nn.conv as conv
from dew.accuracy import reduction_bound, within
from dew.telemetry.benchmark import samples


def reference(x, kernel, dilation, *, layout='NHWC', precision=jax.lax.Precision.HIGHEST):
    if layout == 'NCHW':
        x, kernel = x.transpose(0, 3, 1, 2), kernel.transpose(3, 2, 0, 1)
    out = jax.lax.conv_general_dilated(
        x, kernel, (1, 1), 'SAME', rhs_dilation=(dilation, dilation),
        dimension_numbers=(layout, 'HWIO' if layout == 'NHWC' else 'OIHW', layout),
        feature_group_count=kernel.shape[-1] if layout == 'NHWC' else kernel.shape[0],
        precision=precision)
    return out if layout == 'NHWC' else out.transpose(0, 2, 3, 1)


@contextlib.contextmanager
def dilated_form(form):
    """Each Dew convolution applied within runs a dilated 3x3 depthwise one on
    CUDA as `form(x, kernel, dilation)`, in place of production's form."""
    convolve = partial(conv._conv_general_dilated, dilated=form)

    def substituted(call, args, kwargs, context):
        module = context.module
        if (context.method_name != '__call__' or not isinstance(module, conv.Conv)
                or module.conv_general_dilated is not conv._conv_general_dilated):
            return call(*args, **kwargs)
        # The same layer in its own scope, rewound so a layer called twice reads
        # its parameters again, the form its convolution's argument.
        return module.clone(parent=module.scope.rewound(), conv_general_dilated=convolve)(*args, **kwargs)

    with nn.intercept_methods(substituted):
        yield


def vjp(operation, x, kernel, cotangent):
    output, backward = jax.vjp(operation, x, kernel)
    return output, backward(cotangent)


def timed(operation, args, repeats):
    executable = jax.jit(operation).lower(*args).compile()
    measured = [seconds * 1e3 for seconds in samples(executable, *args, warmup=5, repeats=5, calls=repeats)]
    return {'median_ms': statistics.median(measured), 'min_ms': min(measured), 'max_ms': max(measured)}


def kernels(args):
    rng = np.random.default_rng(17)
    rows = []
    for batch in ((16, 32) if args.batch_size is None else (args.batch_size,)):
        for dtype in (jnp.float32, jnp.bfloat16):
            x = jnp.asarray(rng.normal(size=(batch, 16, 16, 768)), dtype)
            kernel = jnp.asarray(rng.normal(size=(3, 3, 1, 768)), dtype)
            cotangent = jnp.asarray(rng.normal(size=x.shape), dtype)
            for dilation in args.dilations:
                expected = jax.jit(partial(vjp, partial(reference, dilation=dilation)))(
                    x, kernel, cotangent)
                magnitudes = jax.jit(partial(vjp, partial(reference, dilation=dilation)))(
                    jnp.abs(x).astype(jnp.float32), jnp.abs(kernel).astype(jnp.float32),
                    jnp.abs(cotangent).astype(jnp.float32))
                for name, operation in (
                        ('lax-NHWC', partial(reference, dilation=dilation)),
                        ('lax-NCHW', partial(reference, dilation=dilation, layout='NCHW')),
                        ('polyphase', partial(conv._polyphase_depthwise_3x3, dilation=dilation)),
                        ('materialized', partial(conv._materialized_depthwise_3x3, dilation=dilation))):
                    actual = jax.jit(partial(vjp, operation))(x, kernel, cotangent)
                    errors = [float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))
                              for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True)]
                    ratios = []
                    for got, want, magnitude, terms in zip(
                            jax.tree.leaves(actual), jax.tree.leaves(expected), jax.tree.leaves(magnitudes),
                            (9, 9, np.prod(x.shape[:-1])), strict=True):
                        bound = reduction_bound(magnitude, terms)
                        error = np.abs(np.asarray(got, np.float64) - np.asarray(want, np.float64))
                        assert within(error, bound), (name, dtype, dilation, float(error.max()))
                        ratios.append(float(np.max(np.divide(error, bound, out=np.zeros_like(error), where=bound > 0))))
                    row = {'batch': batch, 'dtype': str(jnp.dtype(dtype)), 'dilation': dilation,
                           'implementation': name, 'forward': timed(operation, (x, kernel), args.repeats),
                           'backward': timed(lambda x, w, dy: jax.vjp(operation, x, w)[1](dy),
                                             (x, kernel, cotangent), args.repeats),
                           'forward_and_backward': timed(partial(vjp, operation),
                                                         (x, kernel, cotangent), args.repeats),
                           'max_abs_error_output_dx_dw': errors,
                           'max_fp32_reduction_bound_ratio_output_dx_dw': ratios,
                           'jax': jax.__version__, 'device': jax.devices()[0].device_kind}
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    args.output.write_text(json.dumps(rows, indent=2))
    return rows


def lax_convolution():
    """The model's depthwise convolutions as lax's dilated form, which a step benchmark runs inside
    (benchmark_step's `within`), its peak processes too."""
    return dilated_form(partial(reference, precision=None))


def step(args):
    import benchmark_step
    from benchmark_cases import Case

    config = {'emb_features': 768, 'mlp_ratio': 4, 'norm_epsilon': 1e-5,
              'num_heads': 12, 'num_layers': 16, 'patch_size': 2, 'scan_order': 'zigzag',
              'ssm_attention_ratio': '3:1', 'ssm_state_dim': 64, 'text_pooling': 'all',
              'use_2d_fusion': True, 'adaln_silu': False, 'output_channels': 4}
    if args.remat:
        config['remat'] = True
    cases = [Case('hybrid_dit', config, batch_size=batch, image_size=32, channels=4)
             for batch in ((16, 32) if args.batch_size is None else (args.batch_size,))]
    rows = benchmark_step.main(benchmark_step.BenchmarkConfig(
        cases=cases, steps=args.repeats, warmup=5,
        profile_dir=str(args.output.parent / args.implementation), profile_steps=5,
        json_out=str(args.output), dtype=args.dtype,
        within='benchmark_depthwise:lax_convolution' if args.implementation == 'lax' else None))
    for row, case in zip(rows, cases, strict=True):
        profile = args.output.parent / args.implementation / case.label.replace(' ', '_') / 'process0'
        row.update(convolution_profile(profile))
    args.output.write_text(json.dumps(rows, indent=2))
    return rows


def convolution_profile(directory):
    from trace_window import device_events

    native = 0.0
    for events in device_events(directory)[0].values():
        for event in events:
            stats = {name: str(value) for name, value in event.stats}
            milliseconds = (event.end_ns - event.start_ns) / 1e6 / 5
            if stats.get('hlo_op', '').startswith('cudnn-conv'):
                native += milliseconds
    # The grids' interleaving can fuse with surrounding ops, so this counts
    # only native convolutions and never calls their absence zero fusion work.
    return {'native_cudnn_conv_ms': native}


def checkpoint(args):
    import dew
    from dew.nn.autoencoders.sd_vae import StableDiffusionVAE

    pipeline = dew.pipeline('dewml/hybrid-dit-176m', revision='8d891d967319e51cd0e73d3e5b7c29d68a238b52')
    prompt = 'a watercolor painting of a mountain lake at sunrise'

    def sample(dilated=None):
        jax.clear_caches()
        with contextlib.nullcontext() if dilated is None else dilated_form(dilated):
            return jax.block_until_ready(pipeline(prompt, key=17, steps=20))

    with jax.default_matmul_precision('highest'):
        before = sample(partial(reference, precision=None))
        after = sample()
        rounded = sample(one_rounding)
        vae, params = pipeline.autoencoder, pipeline.variables['autoencoder']
        float_vae = StableDiffusionVAE(
            model=vae.model.clone(dtype=jnp.float32), params=params,
            dtype=jnp.float32, latent_shift=vae.latent_shift, latent_scale=vae.latent_scale)
        decode = jax.jit(lambda params, z: jnp.clip(float_vae.decode(params, z), -1, 1))
        before_fp32, after_fp32, rounded_fp32 = (
            decode(params, z) for z in (before.latents, after.latents, rounded.latents))
        jax.block_until_ready((before_fp32, after_fp32, rounded_fp32))
        delta = np.asarray(after.latents, np.float64) - np.asarray(before.latents, np.float64)
        noise = np.random.default_rng(29).normal(size=delta.shape).astype(np.float32)
        noise *= np.sqrt(np.mean(delta ** 2)) / np.sqrt(np.mean(noise ** 2))
        noise.flat[0] = np.max(np.abs(delta))
        perturbed = before.latents + jnp.asarray(noise)
        decode_bf16 = jax.jit(lambda params, z: jnp.clip(vae.decode(params, z), -1, 1))
        noisy_images = decode_bf16(params, perturbed)
        original_images = decode_bf16(params, before.latents)
        jax.block_until_ready((noisy_images, original_images))
    report = {'latents': errors(before.latents, after.latents),
              'fp32_images': errors(before_fp32, after_fp32),
              'one_rounding_latents': errors(before.latents, rounded.latents),
              'one_rounding_fp32_images': errors(before_fp32, rounded_fp32),
              'bf16_images': errors(before.images, after.images),
              'independent_latent_noise': errors(before.latents, perturbed),
              'bf16_decoder_noise_sensitivity': errors(original_images, noisy_images)}
    print(json.dumps(report), flush=True)
    args.output.write_text(json.dumps(report, indent=2))
    for name in ('latents', 'fp32_images'):
        for statistic in ('max_abs_error', 'rms_error'):
            assert report[name][statistic] <= 2 * report[f'one_rounding_{name}'][statistic], (name, report)
    return report


def one_rounding(x, kernel, dilation):
    """lax's dilated convolution with every other output feature moved up
    one ulp: one rounding's worth of difference in half the outputs."""
    out = reference(x, kernel, dilation, precision=None)
    moved = jnp.nextafter(out, jnp.asarray(jnp.inf, out.dtype))
    return jnp.where(jax.lax.broadcasted_iota(jnp.int32, out.shape, out.ndim - 1) % 2 == 0, moved, out)


def errors(lhs, rhs):
    difference = np.asarray(lhs, np.float64) - np.asarray(rhs, np.float64)
    return {'max_abs_error': float(np.max(np.abs(difference))),
            'rms_error': float(np.sqrt(np.mean(difference ** 2)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('kernels', 'step', 'checkpoint'))
    parser.add_argument('--implementation', choices=('lax', 'production'), default='production')
    parser.add_argument('--dtype', choices=('float32', 'bfloat16'), default='bfloat16',
                        help='Training step compute dtype; checkpoint keeps the published dtypes.')
    parser.add_argument('--batch-size', type=int, choices=(16, 32), help='One step case in a fresh process.')
    parser.add_argument('--dilations', type=int, nargs='+', choices=(1, 2, 3), default=(1, 2, 3),
                        help='Depthwise kernel dilations to measure.')
    parser.add_argument('--remat', action='store_true', help='The same rematerialization policy for both paths.')
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--output', type=Path, default=Path('depthwise.json'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return {'kernels': kernels, 'step': step, 'checkpoint': checkpoint}[args.mode](args)


if __name__ == '__main__':
    main()
