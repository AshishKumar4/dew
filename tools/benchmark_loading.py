#!/usr/bin/env python3
"""Time imports, checkpoint restore, model construction and an optional first sample.

Run in a fresh process on the serving container, with the published source
already cached. Sampling is opt-in; local CPU probes should omit it.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def imports():
    rows = {}
    for name, code in (
            ('dew', 'import dew'),
            ('sampling', 'from dew.sampling import TextToImage'),
            ('hub', 'from dew.interop.hub import pull_from_hub')):
        start = time.perf_counter()
        run = subprocess.run([sys.executable, '-X', 'importtime', '-c', code],
                             capture_output=True, text=True, check=True)
        rows[name] = {'process_s': time.perf_counter() - start, 'importtime': run.stderr.splitlines()}
    return rows


def phases(args):
    start = time.perf_counter()
    import jax
    from dew.objectives.diffusion import DiffusionRunConfig
    from dew.sampling import CFG, DPMSolverMultistep, TextToImage
    from dew.sampling import pipelines

    timings = {'imports_s': time.perf_counter() - start}

    def measured(name, operation):
        def run(*values, **fields):
            begin = time.perf_counter()
            result = operation(*values, **fields)
            if name == 'read_s':
                jax.block_until_ready(result)
            timings[name] = timings.get(name, 0.0) + time.perf_counter() - begin
            return result
        return run

    pipelines.restore_variables = measured('read_s', pipelines.restore_variables)
    DiffusionRunConfig.build = measured('build_s', DiffusionRunConfig.build)
    pipelines._with_drawn_tables = measured('build_s', pipelines._with_drawn_tables)
    load = time.perf_counter()
    with jax.default_matmul_precision('highest'):
        pipe = (TextToImage.from_run(args.source) if os.path.isdir(args.source) else
                TextToImage.from_pretrained(args.source))
        timings['from_pretrained_s'] = time.perf_counter() - load
        if args.sample:
            begin = time.perf_counter()
            result = pipe(['a red fox in a snowy forest'], seed=0, steps=args.steps,
                          sampler=DPMSolverMultistep(), guidance=CFG(5.0)).host()
            timings['first_sample_s'] = time.perf_counter() - begin
            timings['from_pretrained_to_first_sample_s'] = time.perf_counter() - load
            timings['process_to_first_sample_s'] = time.perf_counter() - start
            timings['image_shape'] = list(result.images.shape)
            if args.arrays is not None:
                import numpy as np

                args.arrays.parent.mkdir(parents=True, exist_ok=True)
                np.savez(args.arrays, latents=result.latents, images=result.images)
    timings['bytes'] = sum(leaf.nbytes for leaf in jax.tree.leaves(pipe.params))
    timings['jax'] = jax.__version__
    timings['orbax'] = importlib.metadata.version('orbax-checkpoint')
    timings['device'] = jax.devices()[0].device_kind
    timings['platform'] = platform.platform()
    return timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='Cached Hub repo or a local run directory.')
    parser.add_argument('--sample', action='store_true', help='Also run the first 15-step image sample.')
    parser.add_argument('--steps', type=int, default=15)
    parser.add_argument('--imports', action='store_true', help='Profile imports in fresh subprocesses.')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--arrays', type=Path, help='Keep the first sample for a before/after parity check.')
    args = parser.parse_args()
    report = {'imports': imports()} if args.imports else {}
    report['phases'] = phases(args)
    output = json.dumps(report, indent=2)
    print(output, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)


if __name__ == '__main__':
    main()
