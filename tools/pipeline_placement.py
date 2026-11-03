"""Write tests/fixtures/placement/flux_pipeline.txt: the placement of the tiny Flux pipeline's training state
(its denoiser, frozen CLIP and T5 encoders and autoencoder, and its optimizer state) and of its load placed
on a mesh, one line a leaf, `case path shape spec memory_kind`; tests/test_placement_snapshot.py holds a
change to it. Run from a checkout's root (the suite's eight host devices) on a machine the project's rules
allow:

    python tools/pipeline_placement.py tests/fixtures/placement/flux_pipeline.txt
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "tools")]


def lines(name, tree, placement):
    """`tree`'s leaves placed as `placement` places them, a line each."""
    import jax

    found = []
    for (path, leaf), sharding in zip(jax.tree_util.tree_leaves_with_path(tree), jax.tree.leaves(placement),
                                      strict=True):
        found.append("\t".join((name, jax.tree_util.keystr(path), "x".join(map(str, leaf.shape)) or "-",
                                str(tuple(sharding.spec)), str(sharding.memory_kind))))
    return found


def main() -> None:
    import os
    import tempfile

    from lane_environment import configure_lane

    # The suite's environment, its eight simulated devices among it, set before JAX is imported.
    configure_lane(os.environ)
    import jax
    import optax
    from diffusion_support import source_objective
    from interop_support import extract_fixture

    from dew.distributed.placement import Layout, MeshSpec
    from dew.interop.pretrained import Pretrained
    from dew.training import Trainer

    layout = MeshSpec(fsdp=2, expert=2, tensor=2)
    source = extract_fixture(Path("tests/fixtures/flux_source.tar.xz"), Path(tempfile.mkdtemp()))
    _, objective = source_objective(source)
    mesh = layout.build(jax.devices()[:8])
    trainer = Trainer(objective, optax.adamw(1e-3), key=jax.random.key(0), mesh=layout,
                      layout=Layout(min_shard=16), tracker=None)
    trainer.device_mesh = mesh
    state = jax.eval_shape(trainer.initial_state)
    placed = trainer.shardings(state)
    out = lines("flux:state", {"variables": state.variables, "opt_state": state.opt_state},
                {"variables": placed.variables, "opt_state": placed.opt_state})
    on_mesh = Pretrained.load(str(source / "pipeline"), dtype="float32", attention_impl="xla", mesh=layout,
                              layout=Layout(min_shard=16))
    out += lines("flux:load", on_mesh.variables, jax.tree.map(lambda leaf: leaf.sharding, on_mesh.variables))
    Path(sys.argv[1]).write_text("\n".join(out) + "\n")
    print(len(out), "lines")


if __name__ == "__main__":
    main()
