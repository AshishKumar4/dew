"""Convert saved Flax oracle parameters to canonical safetensors, without model math.

Run in the isolated reference environment. Existing oracle arrays stay intact.
  python tools/native_diffusion_reference_files.py DIRECTORY
The runtime loader consumes only checkpoint/config/tokenizer files afterward.
"""
import argparse
from pathlib import Path
import re

from flax import serialization
from flax.traverse_util import flatten_dict
import numpy as np
from safetensors.numpy import save_file


def convert(directory):
    directory = Path(directory)
    for component in ("unet", "vae", "text_encoder", "text_encoder_2", "safety_checker"):
        folder = directory / component
        flax_file = folder / ("diffusion_flax_model.msgpack" if component in ("unet", "vae") else "flax_model.msgpack")
        if not flax_file.is_file():
            continue
        tensors = {}
        for path, value in flatten_dict(serialization.msgpack_restore(flax_file.read_bytes())).items():
            parts = list(path)
            leaf = parts.pop()
            if leaf in ("kernel", "scale", "embedding"):
                parts.append("weight")
            else:
                parts.append(leaf)
            if leaf == "kernel":
                value = value.transpose(3, 2, 0, 1) if value.ndim == 4 else value.T
            names = []
            for part in parts:
                if part not in ("linear_1", "linear_2"):
                    part = re.sub(r"_(\d+)$", r".\1", part)
                if component == "vae":
                    part = {"query": "to_q", "key": "to_k", "value": "to_v", "proj_attn": "to_out.0"}.get(part, part)
                names.append(part)
            tensors[".".join(names)] = np.ascontiguousarray(value)
        filename = "diffusion_pytorch_model.safetensors" if component in ("unet", "vae") else "model.safetensors"
        save_file(tensors, folder / filename)
    print(directory, "canonical safetensors ready")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    convert(parser.parse_args().directory)
