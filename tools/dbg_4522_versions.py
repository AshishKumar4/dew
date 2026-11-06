"""Report installed CPU compiler provenance without opening a JAX backend."""
import importlib.metadata as metadata
import json
import subprocess

import jaxlib.version

for package in ("jax", "jaxlib"):
    distribution = metadata.distribution(package)
    print("DIST", package, distribution.version, flush=True)
    for name in ("direct_url.json", "WHEEL"):
        print("METADATA", package, name, distribution.read_text(name), flush=True)
print("JAXLIB_VERSION_FIELDS", json.dumps(
    {name: str(value) for name, value in vars(jaxlib.version).items()
     if any(part in name.lower() for part in ("version", "git", "commit"))}), flush=True)
distribution = metadata.distribution("jaxlib")
for file in distribution.files or ():
    if file.name == "libjax_common.so":
        binary = distribution.locate_file(file)
        symbols = subprocess.run(["nm", "-C", str(binary)], capture_output=True, text=True)
        print("GENERIC_DOT_SYMBOLS", symbols.returncode,
              [line for line in symbols.stdout.splitlines()
               if "dot_fp32_1xNx1_1x1x1" in line], flush=True)
