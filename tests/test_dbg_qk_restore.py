import inspect
import importlib.metadata

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest

from dew.checkpoints import Checkpoints, FROZEN_STORE, _digest
from dew.checkpoints._state_trees import _item_metadata


@pytest.mark.parametrize("kind", ["decision", "supervised", "diffusion"])
def test_trace_objective(kind, tmp_path, monkeypatch):
    from test_optim import test_every_objective_carries_live_qk_statistics_to_muonclip

    original_save = Checkpoints._frozen
    original_read = Checkpoints._frozen_values

    def save(self, state_tree):
        for field, tree in state_tree.items():
            paths = [jax.tree_util.keystr(path) for path, _ in
                     jax.tree_util.tree_flatten_with_path(tree)[0]
                     if any(word in jax.tree_util.keystr(path)
                            for word in ("qk", "head_dim", "max_logits"))]
            print("DBG_STATE", kind, field, paths, flush=True)
        return original_save(self, state_tree)

    def read(self, digests, templates):
        for name, digest in digests.items():
            path = self.directory + "/" + FROZEN_STORE + "/" + digest
            with ocp.PyTreeCheckpointer() as checkpointer:
                stored = _item_metadata(checkpointer.metadata(path))
            print("DBG_METADATA", kind, name, repr(stored), flush=True)
            print("DBG_TEMPLATE", kind, name,
                  jax.tree.structure(templates[name]), flush=True)
        return original_read(self, digests, templates)

    monkeypatch.setattr(Checkpoints, "_frozen", save)
    monkeypatch.setattr(Checkpoints, "_frozen_values", read)
    test_every_objective_carries_live_qk_statistics_to_muonclip(kind, tmp_path)


def test_frozen_sown_tuples_round_trip(tmp_path):
    """A nonparameter Flax collection is allowed to contain tuples, including nested ones."""
    written = {"variables": {
        "params": {"weight": jnp.ones((2,))},
        "observations": {"attention": {
            "head_dim": (jnp.asarray(4),),
            "nested": ({"value": (jnp.arange(2.),)}, jnp.asarray(3)),
        }},
    }}
    checkpoints = Checkpoints(str(tmp_path))
    checkpoints.save(1, written)
    checkpoints.wait()
    restored, _ = checkpoints.restore(written)
    assert jax.tree.structure(restored) == jax.tree.structure(written)
    for got, want in zip(jax.tree.leaves(restored), jax.tree.leaves(written), strict=True):
        np.testing.assert_array_equal(got, want)


def test_orbax_tuple_metadata(tmp_path):
    print("DBG_ORBAX_VERSION", importlib.metadata.version("orbax-checkpoint"))
    print("DBG_FROM_FLAT_SOURCE", inspect.getsource(ocp.tree.from_flat_dict))
    print("DBG_SERIALIZE_SOURCE", inspect.getsource(ocp.tree.serialize_tree))
    value = {"attention": {"head_dim": (jnp.asarray(4),)}}
    path = tmp_path / "orbax"
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(path, value)
        metadata = _item_metadata(checkpointer.metadata(path))
        print("DBG_ORBAX_METADATA", repr(metadata))
        restored = checkpointer.restore(path)
        print("DBG_ORBAX_RESTORED", repr(restored))
        print("DBG_DIGEST", _digest(value), _digest(restored))
        repacked = ocp.tree.from_flat_dict(ocp.tree.to_flat_dict(metadata), target=value)
        print("DBG_REPACKED", jax.tree.structure(repacked), jax.tree.structure(value))
        assert isinstance(metadata["attention"]["head_dim"], list)
        assert isinstance(repacked["attention"]["head_dim"], tuple)
