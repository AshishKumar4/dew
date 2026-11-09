"""Convert saved Flax oracle parameters to canonical safetensors, without model math.

Run in the isolated reference environment. Existing oracle arrays stay intact.
  python tools/native_diffusion_reference_files.py DIRECTORY
The runtime loader consumes only checkpoint/config/tokenizer files afterward.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from flax import serialization
from flax.traverse_util import flatten_dict


def convert(directory):
    directory = Path(directory)
    from safetensors.numpy import load_file

    from dew.interop.diffusion import flax_weights_file, weights_stem, write_flax_component
    from dew.interop.safetensors_io import write_file
    index = json.loads((directory / "model_index.json").read_text())
    # Each component the pipeline declares, its weight files named as the library its entry names writes them.
    for component, declared in index.items():
        folder = directory / component
        if not (isinstance(declared, list) and len(declared) == 2 and isinstance(declared[1], str)
                and declared[1].startswith("Flax") and folder.is_dir()):
            continue
        flax_file = folder / flax_weights_file(declared)
        filename = f"{weights_stem(declared)}.safetensors"
        if not flax_file.is_file():
            write_flax_component(directory, component, declared, load_file(folder / filename))
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
                    renamed = {"query": "to_q", "key": "to_k", "value": "to_v", "proj_attn": "to_out.0"}
                    part = renamed.get(part, part)
                names.append(part)
            tensors[".".join(names)] = np.ascontiguousarray(value)
        write_file(tensors, folder / filename, {"format": "pt"})
    print(directory, "canonical safetensors ready")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    convert(parser.parse_args().directory)
