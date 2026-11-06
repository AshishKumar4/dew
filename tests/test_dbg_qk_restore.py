import inspect
import importlib.metadata
import os
import shutil
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest
from flax import linen as nn

from dew.checkpoints import Checkpoints, FROZEN_STORE, _digest
from dew.checkpoints._state_trees import _item_metadata
from dew.state import TrainState


def train_state(variables):
    count = jnp.zeros((), jnp.int32)
    return TrainState(step=count, microstep=count, updates=count, variables=variables,
                      opt_state={"count": count}, ema=None,
                      key=jax.random.key_data(jax.random.key(0)), scale=None,
                      window_size=jnp.ones((), jnp.int32))


@pytest.fixture(autouse=True)
def prototype_source_fix(monkeypatch):
    """Apply the proposed source diff at runtime only; the product files stay untouched."""
    if os.environ.get("DBG_REPACK_FROZEN") != "1":
        return
    source = textwrap.dedent(inspect.getsource(Checkpoints._frozen_values))
    source = source.replace(
        "jax.tree.structure(template) == jax.tree.structure(stored)",
        "_paths(template) == _paths(stored)")
    source = source.replace(
        '_refuse_reshaped(template, stored, f"{path} ({name})")',
        '_refuse_reshaped(template, stored, f"{path} ({name})")\n'
        '                stored = ocp.tree.from_flat_dict(ocp.tree.to_flat_dict(stored), target=template)')
    namespace = {}
    exec(source, Checkpoints._frozen_values.__globals__, namespace)
    monkeypatch.setattr(Checkpoints, "_frozen_values", namespace["_frozen_values"])


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
                            for word in ("['qk']", "head_dim", "max_logits"))]
            if paths or field in ("variables", "opt_state", "accumulation"):
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
    written = train_state({
        "params": {"weight": jnp.ones((2,))},
        "observations": {"attention": {
            "head_dim": (jnp.asarray(4),),
            "nested": ({"value": (jnp.arange(2.),)}, jnp.asarray(3)),
        }},
    })
    checkpoints = Checkpoints(str(tmp_path))
    checkpoints.save(1, written)
    checkpoints.wait()
    restored, _ = checkpoints.restore(written)
    assert jax.tree.structure(restored) == jax.tree.structure(written)
    for got, want in zip(jax.tree.leaves(restored), jax.tree.leaves(written), strict=True):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("collection", ["qk", "custom"])
def test_flax_sown_collection_round_trip(collection, tmp_path):
    class Sowing(nn.Module):
        @nn.compact
        def __call__(self, value):
            self.sow(collection, "head_dim", jnp.asarray(value.shape[-1]))
            return nn.Dense(2)(value)

    variables = Sowing().init(jax.random.key(0), jnp.ones((1, 4)))
    written = train_state(variables)
    checkpoints = Checkpoints(str(tmp_path))
    checkpoints.save(1, written)
    checkpoints.wait()
    restored, _ = checkpoints.restore(written)
    assert jax.tree.structure(restored) == jax.tree.structure(written)
    assert isinstance(restored.variables[collection]["head_dim"], tuple)


def test_corrupt_tuple_collection_refused(tmp_path):
    value = {"head_dim": (jnp.asarray(4),)}
    checkpoints = Checkpoints(str(tmp_path))
    _, digests = checkpoints._frozen({"variables": {"observations": value}})
    digest = digests["variables"]["observations"]
    wrong = "0" * 64
    shutil.copytree(tmp_path / FROZEN_STORE / digest, tmp_path / FROZEN_STORE / wrong)
    with pytest.raises(ValueError, match="holds content with digest"):
        checkpoints._frozen_values({"observations": wrong}, {"observations": value})


def test_orbax_tuple_metadata(tmp_path):
    print("DBG_ORBAX_VERSION", importlib.metadata.version("orbax-checkpoint"))
    print("DBG_FROM_FLAT_SOURCE", inspect.getsource(ocp.tree.from_flat_dict).replace("\n", " "))
    print("DBG_SERIALIZE_SOURCE", inspect.getsource(ocp.tree.serialize_tree).replace("\n", " "))
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
