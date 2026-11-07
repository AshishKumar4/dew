"""FlaxDiff's own models (AshishKumar4/FlaxDiff at COMMIT, its model modules checked against their
SHA-256), the project Dew grew out of, run on drawn weights and inputs, in float32 and float64: the
reference arm of tools/venv_reference.py's `flaxdiff-<model>-<config>` sites (its protocol
tools/reference_worker.py), each a case of tools/flaxdiff_family_cases.py's CASES, run in Dew's own
environment (its JAX), on the CPU.

    python tools/flaxdiff_family_worker.py flaxdiff-MODEL-CONFIG    `IN OUT` lines on stdin, each `done`
    python tools/flaxdiff_family_worker.py --layout OUT.json        every case's shapes and Fourier table

IN holds each draw's `<draw>.param.<path>` by its '/'-joined path in FlaxDiff's parameter tree, its `image`
(or clip), `times` and `text` states, and `probe`, a cotangent of the output. OUT gets
`<draw>.reference.<leaf>` and `<draw>.truth.<leaf>` in float32 and float64: the model's `output`, and the
`gradients` of `sum(output * probe)` with respect to the image, the text and every parameter in sorted path
order, concatenated (`answered`). The float64 walk runs a second copy of the same files with every
`jnp.float32` and `np.float32` reading float64 (FlaxDiff pins its Fourier input, time input, RoPE and sincos
tables, output heads and UNet attention to float32), under `jax.enable_x64`, but the Fourier frequencies,
which FlaxDiff draws as a float32 normal and the float64 copy draws so too, widened: the model's constant.
Every one of those is a dtype (a cast, a default, an initializer's dtype) but one comparison, common.py's
weight-standardized convolution taking `eps = 1e-5 if self.dtype == jnp.float32 else 1e-3`: each walk
compares against its own dtype, so the branch stays the same (and no case here builds that convolution).
One thread and no XNNPACK: a reduction split over threads can sum in another order from call to call.
"""

import os

os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_cpu_use_xnnpack=false"
                           " --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import contextlib
import functools
import json
import sys
import types
from collections.abc import Iterator

import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict
from flaxdiff_family_cases import CASES
from reference_source import github
from reference_worker import Applied, answered, serve

REPO, COMMIT = "AshishKumar4/FlaxDiff", "15c55b001304604147a2a8002a76cf5d4c32092f"
# In import order: each module's own imports come before it.
FILES = {
    "hilbert": "24b7c9bf77539c0b15ca54ac372d93eba4cdd4fd8ce85e5e95a3ab3d9f7faff7",
    "s5": "531ac17e10cef3255ab03d607ff3abb977711b55b62a5379cc0347a993045093",
    "common": "d06be6c24cb015696d8621dd6d2a55b73f86817e0e3afdf86474740b60ad048b",
    "attention": "4954deaa1d05d25b7e551ddafb1687a2ece2d5abd6add54cbb74a645d62c4c7c",
    "vit_common": "665d8e2d248855b662a90d4a960b105171f02d3bed351bad0ca4128ef83dfb13",
    "dit_common": "9beefef6e13f06be1b955be880a77e21b7528d6eb21c729bc9053dc57e50c7b4",
    "simple_unet": "9964f37d44362e99343f0e1d198837e0d38007046b800d27a9d8064a32220d31",
    "simple_vit": "579d6e981142d8aab267e8e7d30ef50f90332090148b925feeda465aa17dbab6",
    "simple_dit": "c3064f67c0da98add4656ea74e3aa3dd4fbbdc52d96f053dc90c4abb44263b79",
    "simple_mmdit": "1d67ca79eb12bdb5cd4ca58994b3909a0c8c642e6073c21cfba507b2eb891a7b",
    "video_dit": "cecd0c05fa53f2078cf5dc7fcc5e55232be3cc9bb722de43429997616d377bac",
    "unet_3d": "ec1323014ec4cedcccac84c7dfe6af46a1dbed9bb1fd8a5a9493a4063916ea6a",
}
FOURIER = "(self.features // 2, ), dtype=jnp.float64)"
"""common.py's Fourier draw as the float64 copy reads it, which draws in float32 and widens instead."""


def flaxdiff(wide: bool) -> dict[str, types.ModuleType]:
    """FlaxDiff's model modules at `COMMIT` as the package `flaxdiff.models`,
    with every `jnp.float32` read as `jnp.float64` when `wide`."""
    for name in [name for name in sys.modules if name.split(".")[0] == "flaxdiff"]:
        del sys.modules[name]
    for package in ("flaxdiff", "flaxdiff.models"):
        module = types.ModuleType(package)
        module.__path__ = []
        sys.modules[package] = module
    modules = {}
    for stem, digest in FILES.items():
        local = github(REPO, COMMIT, f"flaxdiff/models/{stem}.py", digest)
        source = local.read_text()
        if wide:
            source = source.replace("jnp.float32", "jnp.float64").replace("np.float32", "np.float64")
            source = source.replace(FOURIER, "(self.features // 2, ), dtype=jnp.float32).astype(jnp.float64)")
        name = f"flaxdiff.models.{stem}"
        module = types.ModuleType(name)
        module.__package__, module.__file__ = "flaxdiff.models", str(local)
        sys.modules[name] = module
        exec(compile(source, str(local), "exec"), module.__dict__)
        setattr(sys.modules["flaxdiff.models"], stem, module)
        modules[stem] = module
    return modules


def case_of(site: str) -> str:
    """The case of CASES a `flaxdiff-<model>-<config>` site names."""
    model, config = site.removeprefix("flaxdiff-").rsplit("-", 1)
    return f"{model}/{config}"


@contextlib.contextmanager
def applying(case: str, dtype: str) -> Iterator[Applied]:
    """The case's model, in float64 for a float64 run, as `answered` runs it: its output, and the gradients
    of `sum(output * probe)` with respect to the image, the text and every parameter by sorted path."""
    module, cls, config, _, _ = CASES[case]
    model = getattr(flaxdiff(dtype == "float64")[module], cls)(**config)

    def apply(params, inputs):
        image, text, times, probe = (jnp.asarray(inputs[name], dtype) for name in ("image", "text", "times",
                                                                                   "probe"))
        tree = unflatten_dict({tuple(path.split("/")): leaf for path, leaf in params.items()})

        def forward(tree, image, text):
            return model.apply({"params": tree}, image, times, text)

        output, pullback = jax.vjp(forward, tree, image, text)
        grad_tree, grad_image, grad_text = pullback(probe)
        grads = flatten_dict(grad_tree)
        ordered = [grad_image, grad_text, *(grads[path] for path in sorted(grads, key="/".join))]
        return {"output": output, "gradients": jnp.concatenate([jnp.ravel(leaf) for leaf in ordered])}
    yield apply


def layout() -> dict[str, dict]:
    """Every case's parameters' shapes by '/'-joined path, from its model's own initialization, and the
    Fourier table its FourierEmbedding draws (`fourier`), the model's constant."""
    modules = flaxdiff(wide=False)
    found = {}
    for case, (module, cls, config, image, text) in CASES.items():
        model = getattr(modules[module], cls)(**config)
        # Eagerly: FlaxDiff's scan orders read their permutations as NumPy.
        variables = model.init(jax.random.key(0), jnp.zeros((2, *image)), jnp.zeros(2), jnp.zeros((2, *text)))
        width = config["emb_features"]
        table = modules["common"].FourierEmbedding(
            features=width[0] if isinstance(width, tuple) else width).bind({}).freqs
        found[case] = {"params": {"/".join(path): list(leaf.shape)
                                  for path, leaf in sorted(flatten_dict(variables["params"]).items())},
                       "fourier": [float(value) for value in table]}
    return found


if __name__ == "__main__":
    if sys.argv[1:2] == ["--layout"]:
        shapes = layout()
        with open(sys.argv[2], "w") as out:
            json.dump(shapes, out, indent=1)
    elif len(sys.argv) == 2 and case_of(sys.argv[1]) in CASES:
        serve(lambda cases: answered(cases, functools.partial(applying, case_of(sys.argv[1]))))
    else:
        raise SystemExit(__doc__)
