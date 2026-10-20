"""The time a Server's warm compiles take, for a full-depth decoder of a released family's shape.

    python tools/benchmark_serving_compile.py --family dory-1.0-4b --slots 16 --json-out compile.json

Builds the family's committed config (tests/fixtures/hf/<family>/config.json), every layer at its
released width, over random weights, and times the Server's warm (first) compile of every step
program a run of `--slots` slots compiles: its construction's decode step (`Server.__init__`
compiles it to choose the resident layouts), then each power-of-two admission width's prefill
(`Server.admission`). The record holds the wall seconds of construction (`construction_seconds`),
of each admission width's first round (`by_admission_width`) and of the whole warm
(`warm_seconds`), which compiles fill (the steps themselves run in a fraction of a second), and
the device they compiled for. Nothing is read from or written to a compilation cache.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def narrow(config: dict, hidden: int) -> dict:
    """The family's config at `hidden` width, every layer kept: four heads of `hidden / 4`, two of
    them key-value heads, a feed-forward twice the width and a vocabulary of 1024."""
    head = hidden // 4
    return {**config, "hidden_size": hidden, "intermediate_size": 2 * hidden, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": head, "vocab_size": 1024}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="dory-1.0-4b")
    parser.add_argument("--hidden", type=int, default=None,
                        help="a width other than the released one, in four heads; a narrow width hides "
                             "fusion pathologies the released one shows (ledger 635)")
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--prompt", type=int, default=512)
    parser.add_argument("--output", type=int, default=16)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_compilation_cache", val=False)
    from dew.inference import TextGeneration
    from dew.inference.serving import Server
    from dew.interop import hf_decoders
    from dew.registry import models
    from dew.sampling import Sampling

    source = ROOT / "tests/fixtures/hf" / args.family / "config.json"
    released = json.loads(source.read_text())
    config = released if args.hidden is None else narrow(released, args.hidden)
    # The model's length leaves room for the Server's rounding of its capacity to its cache's pages.
    model = models.build("causal_transformer", hf_decoders.translate_config(config)).clone(
        dtype=jnp.bfloat16, max_seq_len=2 * (args.prompt + args.output))
    variables = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    variables = jax.tree.map(
        lambda leaf: leaf.astype(jnp.bfloat16) if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf,
        variables)
    task = TextGeneration(model, variables, None, sampling=Sampling(temperature=0, eos_token_ids=None))
    began = time.perf_counter()
    server = Server.from_task(task, slots=args.slots, capacity=args.prompt + args.output)
    constructed = time.perf_counter() - began
    widths = sorted({min(server.admission, 1 << bit) for bit in range(server.admission.bit_length())})
    rng = np.random.default_rng(0)
    timed = {}
    for rows in widths:
        start = time.perf_counter()
        tickets = [server.submit(rng.integers(1, 1000, args.prompt).astype(np.int32), args.output,
                                 key=jax.random.key(row)) for row in range(rows)]
        server.run()
        for ticket in tickets:
            ticket.result()
        timed[f"admission {rows}"] = time.perf_counter() - start
    record = {"family": args.family, "hidden": config["hidden_size"],
              "layers": config.get("num_hidden_layers"), "slots": args.slots, "admission": server.admission,
              "device": jax.devices()[0].device_kind,
              "construction_seconds": constructed, "by_admission_width": timed,
              "warm_seconds": time.perf_counter() - began}
    args.json_out.write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


if __name__ == "__main__":
    main()
