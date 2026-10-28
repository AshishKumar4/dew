"""Probe: what a compile-only GPU device states."""


def test_probe():
    import jax

    from dew.training.ahead import compile_only_gpus

    (device,) = compile_only_gpus("sm90", 1)
    lines = [repr(device), str(type(device.client)), str(device.client.platform),
             str([n for n in dir(device) if not n.startswith("_")]),
             str([n for n in dir(device.client) if not n.startswith("_")][:60]),
             f"in jax.devices(): {device in jax.devices()}", f"process {device.process_index} {jax.process_index()}"]
    raise SystemExit("PROBE\n" + "\n".join(lines))
