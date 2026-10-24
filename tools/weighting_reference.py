#!/usr/bin/env python3
"""Record P2's and min-SNR's loss weights, as their authors compute them, over check 1's seeds, to
tests/fixtures/weighting/recorded_weights.npz.

Each repo's `guided_diffusion` package (gaussian_diffusion.py with the nn.py
and losses.py it imports) is fetched at a pinned commit and imported as it
is: jychoi118/P2-weighting (Choi et al. 2022) and
TiankaiHang/Min-SNR-Diffusion-Training (Hang et al. 2023). Its own
`GaussianDiffusion.training_losses` then runs at every timestep of a beta
table on a zero clean sample with zero noise and a stand-in model that
predicts one everywhere: every target is zero there, so the squared error
is exactly one and `terms["mse"]` is the weight the authors apply at that
step.

The cases: P2 on an epsilon model at (k, gamma) = (1, 1), (1, 0.5) and
(2, 1), and min-SNR-5 on epsilon (`min_snr_5`), v (`vmin_snr_5`) and x_0
(`min_snr_5` with START_X) models, each over two families of beta tables a
seed draws (`table_inputs`): Ho et al.'s linear table between drawn ends, and
improved-diffusion's cosine table at a drawn offset with its last beta one,
whose zero terminal SNR min-SNR's code weights as one.

Each case runs as written, in the fp32 the authors' `_extract_into_tensor`
casts to, and again with that one helper keeping float64, which gives the
exact weights the fp32 run is measured from. The recording keeps check 1's
evidence of each case over its seeds (`record_case`'s summary), and
tests/test_schedulers.py draws the tables again.

Run with Torch 2.14.0 CPU and NumPy:
  python tools/weighting_reference.py
"""

import argparse
import importlib
from functools import partial
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
RECORDED = FIXTURES / "weighting" / "recorded_weights.npz"
REPOS = {
    "p2": ("jychoi118/P2-weighting", "3da0947ac350072e457c211401218175bc94e137"),
    "min_snr": ("TiankaiHang/Min-SNR-Diffusion-Training", "5189997d6bbde5a1301bcf31a54d2b3589f7309c"),
}
FILES = {
    "p2": {"gaussian_diffusion.py": "19b3a2a9612b5d658bc6f169958cdebe14a42e9ec1a1173ee1aec6b494a5ebae",
           "nn.py": "d9547fb22b421a16dc079cb82a9d98630100b5e1b2a710d343f4d3f90b8661f3",
           "losses.py": "0b0f5a5532910fbc7f16d30f2b891d8f1e20aa88881589ce4eacc2cc6ce06026"},
    "min_snr": {"gaussian_diffusion.py": "8a58f1e4f42dc4ec5908fa7688cc9b9e8b1bb530b4cc8448a82f6ef45be7ed5d",
                "nn.py": "d9547fb22b421a16dc079cb82a9d98630100b5e1b2a710d343f4d3f90b8661f3",
                "losses.py": "0b0f5a5532910fbc7f16d30f2b891d8f1e20aa88881589ce4eacc2cc6ce06026"},
}
"""Each repo's guided_diffusion files its gaussian_diffusion module reads, to their SHA-256."""
CASES = {
    "p2": (("EPSILON", {"p2_k": 1, "p2_gamma": 1}), ("EPSILON", {"p2_k": 1, "p2_gamma": 0.5}),
           ("EPSILON", {"p2_k": 2, "p2_gamma": 1})),
    "min_snr": (("EPSILON", {"mse_loss_weight_type": "min_snr_5"}),
                ("VELOCITY", {"mse_loss_weight_type": "vmin_snr_5"}),
                ("START_X", {"mse_loss_weight_type": "min_snr_5"})),
}
TABLES = ("linear", "zero_terminal_cosine")
STEPS = 1000
"""Each drawn table's steps."""


def label_of(name: str, mean: str, options: dict, table: str) -> str:
    """A case's recording label, as tests/test_schedulers.py names it."""
    words = [name, mean.lower(), *(f"{key}{value}" for key, value in options.items()), table]
    return "_".join(words).replace(".", "p")


def table_inputs(table: str, seed: int) -> dict[str, np.ndarray]:
    """A seed's float64 beta table of `table`'s family, STEPS long, the same on every run: Ho et al.'s
    linear betas from a drawn first one in [5e-5, 2e-4] to a drawn last one in [0.01, 0.03], or
    improved-diffusion's cosine betas (clipped at 0.999) at a drawn offset in [0.004, 0.016] with the
    last beta one."""
    from dew.diffusion.schedules.cosine import cosine_beta_schedule

    rng = np.random.default_rng([160, TABLES.index(table), seed])
    if table == "linear":
        return {"betas": np.linspace(rng.uniform(5e-5, 2e-4), rng.uniform(0.01, 0.03), STEPS)}
    # improved-diffusion's arithmetic to the bit (`cosine_beta_schedule`), at the drawn offset.
    betas = cosine_beta_schedule(STEPS, start_angle=rng.uniform(0.004, 0.016))
    betas[-1] = 1.0
    return {"betas": betas}


def closed_form(name: str, mean: str, options: dict, inputs: dict, dtype) -> dict:
    """The authors' weight at every step of `inputs`' betas in jnp, their float64 tables cast to
    `dtype` where their `_extract_into_tensor` casts (float32 in their run, kept float64 in the exact
    one): the proxy a recording plans its count on, and in float64 tests/test_schedulers.py's truth."""
    import jax.numpy as jnp

    cumulative = jnp.cumprod(1 - jnp.asarray(inputs["betas"], jnp.float64))
    if name == "p2":
        snr = 1.0 / (1 - cumulative) - 1
        return {"weight": (1 / (options["p2_k"] + snr) ** options["p2_gamma"]).astype(dtype)}
    alpha, sigma = jnp.sqrt(cumulative).astype(dtype), jnp.sqrt(1.0 - cumulative).astype(dtype)
    snr = (alpha / sigma) ** 2
    weight = jnp.minimum(snr, float(options["mse_loss_weight_type"].rsplit("_", 1)[1]))
    if mean != "START_X":
        weight = weight / (snr + 1 if mean == "VELOCITY" else snr)
    return {"weight": jnp.where(snr == 0, 1.0, weight).astype(dtype)}


def guided_diffusion(name: str):
    """The repo's own gaussian_diffusion module, imported from its files."""
    from reference_source import package, published_files

    files = {f"guided_diffusion/{file}": digest for file, digest in FILES[name].items()}
    package(f"guided_diffusion_{name}", published_files(*REPOS[name], files) / "guided_diffusion")
    return importlib.import_module(f"guided_diffusion_{name}.gaussian_diffusion")


def weights(module, betas: np.ndarray, mean: str, options: dict, exact: bool) -> np.ndarray:
    """The authors' weight at every step of `betas`, read off `terms["mse"]`, at its own dtype."""
    import torch
    from torch_precision import replaced

    def keep_float64(array, timesteps, shape):
        result = torch.from_numpy(array)[timesteps]
        while result.dim() < len(shape):
            result = result[..., None]
        return result.expand(shape)

    with replaced(*[(module, "_extract_into_tensor", keep_float64)] if exact else []):
        diffusion = module.GaussianDiffusion(
            betas=betas, model_mean_type=module.ModelMeanType[mean],
            model_var_type=module.ModelVarType.FIXED_SMALL, loss_type=module.LossType.MSE, **options)
        dtype = torch.float64 if exact else torch.float32
        steps = torch.arange(len(betas))
        zero = torch.zeros((len(betas), 1, 1, 1), dtype=dtype)
        terms = diffusion.training_losses(lambda x, t, **_: torch.ones_like(x), zero, steps, noise=zero)
        return terms["mse"].numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RECORDED)
    out = parser.parse_args().out
    import jax
    import jax.numpy as jnp
    import torch
    from recorded_references import provenance, record_case, save

    recorded = {}
    with jax.enable_x64(new_val=True):
        for name, cases in CASES.items():
            module = guided_diffusion(name)
            for mean, options in cases:
                for table in TABLES:
                    label = label_of(name, mean, options, table)

                    def evaluate(inputs, module=module, mean=mean, options=options):
                        return tuple({"weight": weights(module, inputs["betas"], mean, options, exact)}
                                     for exact in (False, True))

                    form = partial(closed_form, name, mean, options)
                    recorded.update(record_case(label, partial(table_inputs, table), evaluate,
                                                partial(form, dtype=jnp.float32),
                                                truth_proxy=partial(form, dtype=jnp.float64),
                                                store_inputs=False, summary=True))
                    recorded[f"{label}.provenance"] = provenance(
                        REPOS[name][0], REPOS[name][1], "cpu", "python tools/weighting_reference.py",
                        torch=torch.__version__)
    written = save(out, recorded, reference=" ".join(f"{repo} {commit}" for repo, commit in REPOS.values()),
                   license="MIT (P2-weighting); Min-SNR-Diffusion-Training states none")
    print(f"{written}: {written.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    main()
