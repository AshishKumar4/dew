"""Scratch KDA stage measurements; invoked only by the exclusive diagnostic job."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def torch_stages(source, destination):
    import torch
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
    from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
    from fla.ops.utils import chunk_local_cumsum

    inputs = np.load(source)
    saved = {}
    for seed in range(int(inputs['seeds'])):
        q, k, v, g, beta, h0 = [torch.tensor(inputs[f'{seed}/{name}'], device='cuda')
                                for name in ('q', 'k', 'v', 'g', 'beta', 'h0')]
        cg = chunk_local_cumsum(g, chunk_size=64, scale=1.4426950216)
        w, u, qg, kg, aqk, akk = chunk_kda_fwd_intra(
            q=q, k=k, v=v, gk=cg, beta=beta, scale=0.5, chunk_size=64, disable_recompute=True)
        h, vn, ht = chunk_gated_delta_rule_fwd_h(
            k=kg, w=w, u=u, gk=cg, initial_state=h0, output_final_state=True, chunk_size=64)
        o = chunk_gla_fwd_o_gk(q=q, v=vn, g=cg, A=aqk, h=h, scale=0.5, chunk_size=64)
        for name, value in dict(cg=cg, aqk=aqk, akk=akk, w=w, u=u, kg=kg, qg=qg,
                                vn=vn, ht=ht, o=o).items():
            host = value.detach().cpu().numpy()
            if name == 'aqk':
                causal = np.arange(host.shape[-1])[None, :] <= np.arange(host.shape[1])[:, None]
                host = np.where(causal[:, None, :], host, 0.0)
            saved[f'{seed}/{name}'] = host
    np.savez(destination, **saved)


def truth(q, k, v, g, beta, h0, *, cumulative=False, aqk=None, akk=None, w=None, u=None, kg=None):
    q, k, v, g, beta, h0 = [np.asarray(x, np.float64) for x in (q, k, v, g, beta, h0)]
    cg = g if cumulative else np.cumsum(g, axis=1) / np.log(2.0)
    scores = np.einsum('bihk,bjhk,bijhk->bihj', q, k,
                       np.exp2(cg[:, :, None] - cg[:, None, :])) * 0.5
    raw = np.einsum('bihk,bjhk,bijhk->bihj', k * beta[..., None], k,
                    np.exp2(cg[:, :, None] - cg[:, None, :]))
    length = q.shape[1]
    scores *= np.tril(np.ones((length, length)))[:, None, :]
    raw *= np.tril(np.ones((length, length)), -1)[:, None, :]
    inverse = np.linalg.inv(np.moveaxis(raw, 2, 1) + np.eye(length))
    inverse = np.moveaxis(inverse, 1, 2)
    used_inverse = inverse if akk is None else np.asarray(akk, np.float64)[..., :length]
    wt = np.einsum('bihj,bjhk->bihk', used_inverse, k * beta[..., None] * np.exp2(cg))
    ut = np.einsum('bihj,bjhv->bihv', used_inverse, v * beta[..., None])
    kgt = k * np.exp2(cg[:, -1:] - cg)
    qgt = q * np.exp2(cg)
    wt, ut, kgt = (wt if w is None else w, ut if u is None else u, kgt if kg is None else kg)
    vn = ut - np.einsum('bihk,bhkv->bihv', wt, h0)
    ht = h0 * np.exp2(cg[:, -1])[..., None] + np.einsum('bihk,bihv->bhkv', kgt, vn)
    scores = scores if aqk is None else np.asarray(aqk, np.float64)[..., :length]
    out = np.einsum('bihk,bhkv->bihv', qgt, h0) * 0.5 + np.einsum('bihj,bjhv->bihv', scores, vn)
    return dict(cg=cg, aqk=scores, akk=inverse, w=wt, u=ut, kg=kgt, qg=qgt, vn=vn, ht=ht, o=out)


def main():
    import jax
    import jax.numpy as jnp
    from dew.nn.kernels import fla_gdn, fla_kda, recurrence
    from dew.nn.precision import l2norm
    from test_kda import operands
    from kda_precision_experiment import install_dot

    seeds = 40
    arrays = {'seeds': np.array(seeds)}
    for seed in range(seeds):
        q, k, v, g, beta, h0 = operands(seed)
        q, k = [np.asarray(l2norm(jnp.asarray(x))) for x in (q, k)]
        q, k, v, g = [np.pad(x, ((0, 0), (0, 0), (0, 0), (0, 64 - x.shape[-1])))
                       for x in (q, k, v, g)]
        h0 = np.pad(h0, ((0, 0), (0, 0), (0, 60), (0, 60)))
        for name, value in zip(('q', 'k', 'v', 'g', 'beta', 'h0'), (q, k, v, g, beta, h0), strict=True):
            arrays[f'{seed}/{name}'] = value

    def port(q, k, v, g, beta, h0, *, mode):
        if mode.startswith('prefix64'):
            with jax.enable_x64():
                cg = (jnp.cumsum(g.astype(jnp.float64), axis=1) / np.log(2.0)).astype(jnp.float32)
        else:
            cg = recurrence.cumulated(g, scale=fla_gdn.RCP_LN2)
        aqk, akk = fla_kda.intra_solve(q, k, cg, beta, scale=0.5)
        w, u, kg, qg = fla_kda.recompute_w_u(k, v, beta, akk, cg, q)
        h, vn, ht = fla_gdn.chunk_fwd_h(kg, w, u, cg, h0)
        out = fla_kda.chunk_fwd_o(q, vn, cg, aqk, h, scale=0.5)
        return dict(cg=cg, aqk=aqk, akk=akk, w=w, u=u, kg=kg, qg=qg, vn=vn, ht=ht, o=out)

    with tempfile.TemporaryDirectory() as directory:
        source, destination = [str(Path(directory) / name) for name in ('inputs.npz', 'fla.npz')]
        np.savez(source, **arrays)
        subprocess.run([os.environ['DEW_FLA_PYTHON'], __file__, '--fla', source, destination], check=True)
        upstream = np.load(destination)
        for mode in ('baseline', 'prefix64', 'bf16x6', 'prefix64-bf16x6'):
            if mode == 'bf16x6':
                install_dot()
            measured = {}
            run = jax.jit(lambda *args: port(*args, mode=mode))
            for seed in range(seeds):
                args = [arrays[f'{seed}/{name}'] for name in ('q', 'k', 'v', 'g', 'beta', 'h0')]
                found = {name: np.asarray(x, np.float64) for name, x in run(*args).items()}
                reference = {name: upstream[f'{seed}/{name}'].astype(np.float64) for name in found}
                wanted = truth(*args)
                for arm, result in (('dew', found), ('fla', reference)):
                    for name, value in result.items():
                        if name in ('aqk', 'akk'):
                            value = value[..., :args[0].shape[1]]
                        error = np.mean((value - wanted[name]) ** 2)
                        measured.setdefault(f'{arm}/{name}', []).append(error)
                    local = truth(*args[:3], result['cg'], *args[4:], cumulative=True)
                    for name in ('aqk', 'akk'):
                        value = result[name][..., :args[0].shape[1]]
                        measured.setdefault(f'{arm}/local-{name}', []).append(
                            np.mean((value - local[name]) ** 2))
                    local = truth(*args[:3], result['cg'], *args[4:], cumulative=True, akk=result['akk'])
                    for name in ('w', 'u', 'kg', 'qg'):
                        measured.setdefault(f'{arm}/local-{name}', []).append(
                            np.mean((result[name] - local[name]) ** 2))
                    local = truth(*args[:3], result['cg'], *args[4:], cumulative=True,
                                  akk=result['akk'], w=result['w'], u=result['u'], kg=result['kg'],
                                  aqk=result['aqk'])
                    for name in ('vn', 'ht', 'o'):
                        measured.setdefault(f'{arm}/local-{name}', []).append(
                            np.mean((result[name] - local[name]) ** 2))
            rms = {name: float(np.sqrt(np.mean(values))) for name, values in measured.items()}
            ratios = {name.removeprefix('dew/'): rms[name] / rms[name.replace('dew/', 'fla/')]
                      for name in rms if name.startswith('dew/')}
            print(json.dumps({'mode': mode, 'rms': rms, 'ratios': ratios}, sort_keys=True), flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--fla':
        torch_stages(*sys.argv[2:])
    else:
        main()
