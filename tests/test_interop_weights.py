"""dew.interop.weights' indexed-name folding, the rule every autoencoder family's tensor names share."""

import pytest

from dew.interop.weights import indexed_path


@pytest.mark.parametrize("geometry", ["expert_chunks", "padded", "scalar"])
def test_a_source_shaped_gradient_keeps_compute_precision_before_checkpoint_storage(tmp_path, geometry):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from dew.interop.safetensors_io import read_weights, save_sharded
    from dew.interop.streaming import WeightLayout

    if geometry == "expert_chunks":
        shape = (2, 3, 4, 4)
        layout = WeightLayout("projection.weight", (("params", "first"), ("params", "second")),
                              (2, 16), transpose=(1, 0), concatenate=0, expert_index=1,
                              dtype=np.dtype("float16"), chunks=2)
        def mapped(values):
            selected = values[:, 1].reshape(2, 2, 2, 4)
            return selected.transpose(1, 0, 2, 3).reshape(8, 4).T.reshape(2, 16)
        def variables(values):
            return {"params": {"first": values[0], "second": values[1]}}
    elif geometry == "padded":
        shape = (4,)
        layout = WeightLayout("bias", (("params", "bias"),), (7,), padded=7,
                              dtype=np.dtype("float16"))
        def mapped(values):
            return np.pad(values, (0, 3))
        def variables(values):
            return {"params": {"bias": values}}
    else:
        shape = ()
        layout = WeightLayout("scale", (("params", "scale"),), (), dtype=np.dtype("float16"))
        def mapped(values):
            return values
        def variables(values):
            return {"params": {"scale": values}}

    with jax.enable_x64(new_val=True):
        values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + 2**-30
        probe = np.arange(np.prod(layout.shape), dtype=np.float64).reshape(layout.shape) + 1
        view = jax.jit(lambda value: layout.source_array(variables(value)))
        actual = view(jnp.asarray(values))
        np.testing.assert_array_equal(actual, mapped(values))
        assert actual.dtype == np.float64
        gradient = jax.jit(jax.grad(lambda value: jnp.sum(view(value) * probe)))(jnp.asarray(values))
        expected = np.zeros(values.size, np.float64)
        indices = mapped(np.arange(values.size).reshape(shape))
        valid = mapped(np.ones(shape, bool))
        np.add.at(expected, indices[valid], probe[valid])
        np.testing.assert_array_equal(gradient, expected.reshape(shape))
        exported = layout.export(variables(jnp.asarray(values)))
    save_sharded({layout.name: exported}, tmp_path)
    restored = read_weights(tmp_path)[layout.name]
    assert restored.dtype == np.float16
    np.testing.assert_array_equal(restored, mapped(values).astype(np.float16), strict=True)


def test_retained_tensors_stay_unread_until_their_export_shard_is_written(tmp_path):
    """Composing retained source tensors into an export reads metadata only; the shard writer
    subsequently requests their values and writes a checkpoint that reloads unchanged."""
    import jax
    import numpy as np

    from dew.interop.decoder_parts import layout_tensors
    from dew.interop.safetensors_io import LazyTensors, read_weights, save_sharded

    values = {"a.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
              "b.weight": np.arange(20, dtype=np.float32).reshape(4, 5)}
    reads = []

    def read(name):
        reads.append(name)
        return values[name].copy()

    retained = LazyTensors({name: jax.ShapeDtypeStruct(value.shape, value.dtype)
                            for name, value in values.items()}, read)
    export = layout_tensors({}, {}, retained=retained)
    assert not reads
    save_sharded(export, tmp_path, max_shard_size=80)
    assert set(reads) == set(values)
    restored = read_weights(tmp_path)
    for name, expected in values.items():
        np.testing.assert_array_equal(restored[name], expected)


@pytest.mark.parametrize(("name", "indexed", "path"), [
    ("down_blocks.0.resnets.1.conv1", None, ("down_blocks_0", "resnets_1", "conv1")),
    ("decoder.up_blocks.0.0.conv", None, ("decoder", "up_blocks_0_0", "conv")),
    ("0.proj", None, ("0", "proj")),
    ("decoder.up_blocks.2.resample.1", {"up_blocks"}, ("decoder", "up_blocks_2", "resample", "1")),
    ("a.0.1", {"a"}, ("a_0", "1")),
    ("a.0.1", {"a", "a_0"}, ("a_0_1",)),
    ("a.0", set(), ("a", "0")),
])
def test_a_numeric_token_joins_the_indexed_token_before_it(name, indexed, path):
    """A digit joins the token before it where that token, as folded so far, is indexed (any,
    unrestricted; none, where empty): unrestricted, consecutive digits all join it; a leading digit, or
    one after a token left out, stays its own."""
    assert indexed_path(name.split("."), indexed=indexed) == path


def test_no_tokens_fold_to_no_path():
    assert indexed_path(()) == ()


def test_a_host_leafs_export_transposes_on_the_host():
    """A host leaf's transposed export (an export gathered home, a NumPy bfloat16 kernel) transposes
    where it is, moving nothing to a device and back, which would need it to fit there (ledger 1875);
    a device leaf's transposes on its device, as before, the same values either way."""
    import jax
    import ml_dtypes
    import numpy as np

    from dew.interop.streaming import WeightLayout

    layout = WeightLayout("proj.weight", (("params", "kernel"),), (3, 2), transpose=(1, 0))
    kernel = np.arange(6, dtype=np.float32).reshape(2, 3).astype(ml_dtypes.bfloat16)
    with jax.transfer_guard("disallow_explicit"):
        exported = layout.export({"params": {"kernel": kernel}})
    np.testing.assert_array_equal(exported, kernel.T)
    on_device = layout.export({"params": {"kernel": jax.device_put(kernel)}})
    np.testing.assert_array_equal(on_device, kernel.T)


PACKS = {
    "a transposed qkv (GPT-2's c_attn)": ("gpt2", ".attn.c_attn.weight", (8, 24), {}),
    "unequal widths (Phi-3's qkv)": ("phi3", ".self_attn.qkv_proj.weight", (24, 5),
                                     {"num_heads": 4, "num_kv_heads": 2, "head_dim": 3}),
    "runs a key head (MiMo-V2's grouped qkv)": ("mimo_v2", ".self_attn.qkv_proj.weight", (24, 5),
                                                {"num_heads": 4, "num_kv_heads": 2, "head_dim": 3}),
    "transposed experts (Qwen 3.5 MoE's gate_up_proj)": (
        "qwen3_5_moe_text", ".experts.gate_up_proj", (2, 6, 4), {}),
}
"""Packed source tensors of each kind the export joins: one transposed after its parts are joined, one of
unequal parts, one interleaved in runs, and stacked experts transposed on their last two axes; each by its
family, a stored shape and the translated geometry its parts' widths and runs read."""


@pytest.mark.parametrize("kind", PACKS)
def test_a_packed_tensor_split_into_its_parts_and_joined_by_its_layout_is_the_stored_tensor(kind):
    """A source tensor a family packs, split into its parts as a load reads them (`packed_readings`) and
    joined again from them held in memory by the layout an export writes it by (`packed_layouts`), is the
    stored tensor bit for bit: the one plan reads a pack and writes it back, whether its parts are a
    decoder's leaves or tensors in memory (ledger 1027)."""
    import numpy as np

    from dew.interop.decoder_parts import packed_layouts, packed_readings, packing_of
    from dew.interop.hf_decoders import family_entry
    from dew.interop.streaming import WeightLayout

    family, suffix, shape, config = PACKS[kind]
    packed = family_entry(family).packed
    name = f"model.layers.0{suffix}"
    assert packing_of(packed, name) is not None, (family, suffix)
    stored = np.random.default_rng(1027).normal(size=shape).astype(np.float32)
    (reading,) = packed_readings({name: stored}, packed, config)
    parts = {part.name: np.asarray(part.tensor) for part in reading.parts}
    assert reading.packing is not None and len(parts) == len(reading.packing.parts)
    held = {part: WeightLayout(part, ((part,),), value.shape) for part, value in parts.items()}
    layouts = packed_layouts(held, packed, config)
    assert list(layouts) == [name]
    np.testing.assert_array_equal(layouts[name].export(parts), stored)
