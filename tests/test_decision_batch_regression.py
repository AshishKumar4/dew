"""Diagnostic draws of a decision row alone and beside mixed rows."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew import algebra
from dew.decision import Choice, Decide, DecisionInputs


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("kind", ["laya", "joint"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_mixed_rows_keep_each_probability(tmp_path, kind, dtype):
    source = FIXTURES / "laya" / "tiny"
    if kind == "joint":
        source = tmp_path / "release"
        source.mkdir()
        backbone = FIXTURES / "hf" / "qwen38-dense-tiny"
        joint = FIXTURES / "joint_checkpoint" / "tiny"
        for item in [*backbone.iterdir(), joint / "joint_head_config.json",
                     joint / "joint_head.safetensors"]:
            (source / item.name).symlink_to(item.resolve())
    task = Decide.from_pretrained(source, attention_impl="xla", dtype=dtype)
    task = replace(task, layout=replace(task.layout, max_len=256 if kind == "laya" else 512))
    short = ("refund", {"q": Choice("team", ["pay", "tech", "sales"])})
    long = ("invoice " * 20, {"q": Choice("which department", ["billing", "technical", "sales",
                                                             "support", "account", "other"])})
    requests = [short, long]
    rows = [own[0] for own in task.rows(requests)]
    print("DIAGNOSTIC", kind, dtype, "lengths", [len(row.tokens) for row in rows], flush=True)
    for enabled in (False, True):
        with algebra.passes(enabled):
            together = task(requests)
            apart = [task(state, questions) for state, questions in requests]
            print("PUBLIC", kind, dtype, enabled,
                  [(np.asarray(batch["q"].probabilities).tolist(),
                    np.asarray(lone["q"].probabilities).tolist())
                   for batch, lone in zip(together, apart, strict=True)], flush=True)
            batched = np.asarray(task._forward(
                task.variables, DecisionInputs.collate(rows, task.specials.pad)))
            for index, row in enumerate(rows):
                own = np.asarray(task._forward(
                    task.variables, DecisionInputs.collate([row], task.specials.pad)))
                width = len(row.questions[0].options)
                print("COLLATED", kind, dtype, enabled, index,
                      batched[index, 0, :width].tolist(), own[0, 0, :width].tolist(), flush=True)
    print("ENV", jax.config.jax_default_matmul_precision, jnp.dtype(dtype), str(task.layout), flush=True)


def test_locate_the_first_changed_states(tmp_path):
    from dew.batch import part, thaw

    source = tmp_path / "release"
    source.mkdir()
    backbone = FIXTURES / "hf" / "qwen38-dense-tiny"
    joint = FIXTURES / "joint_checkpoint" / "tiny"
    for item in [*backbone.iterdir(), joint / "joint_head_config.json",
                 joint / "joint_head.safetensors"]:
        (source / item.name).symlink_to(item.resolve())
    task = Decide.from_pretrained(source, attention_impl="xla", dtype="bfloat16")
    short = ("refund", {"q": Choice("team", ["pay", "tech", "sales"])})
    long = ("invoice " * 20, {"q": Choice("which department", ["billing", "technical", "sales",
                                                             "support", "account", "other"])})
    rows = [own[0] for own in task.rows([short, long])]
    single = DecisionInputs.collate(rows[:1], task.specials.pad)
    mixed = DecisionInputs.collate(rows, task.specials.pad)
    padded = jax.tree.map(lambda leaf: leaf[:1], mixed)
    changed = replace(mixed, tokens=mixed.tokens.at[1].set(mixed.tokens[1][::-1]))

    def captured(inputs):
        return task.model.backbone.apply(
            thaw(part(task.variables, "backbone")), inputs.tokens, train=False,
            coordinates=inputs.coordinates(), method=task.model.states_method,
            capture_intermediates=True, mutable=["intermediates"])

    with algebra.passes(False):
        captured = jax.jit(captured)
        outputs = {name: captured(inputs) for name, inputs in
                   (("single", single), ("mixed", mixed), ("padded", padded), ("changed", changed))}
        alone = dict(jax.tree_util.tree_flatten_with_path(outputs["single"])[0])
        for name in ("padded", "mixed", "changed"):
            leaves = dict(jax.tree_util.tree_flatten_with_path(outputs[name])[0])
            for path, old in alone.items():
                new = leaves[path]
                if not old.shape or old.shape[0] != 1 or old.ndim < 3:
                    continue
                slices = tuple(slice(0, width) for width in old.shape)
                first, second = np.asarray(old), np.asarray(new[slices])
                different = np.count_nonzero(first != second)
                if different:
                    print("STATES", name, str(path), old.shape, new.shape, different,
                          float(np.max(np.abs(first.astype(float) - second.astype(float)))), flush=True)
        for first, second in (("padded", "mixed"), ("mixed", "changed")):
            left = np.asarray(outputs[first][0])[0, :single.tokens.shape[1]]
            right = np.asarray(outputs[second][0])[0, :single.tokens.shape[1]]
            print("FIXED_LENGTH", first, second, int(np.count_nonzero(left != right)),
                  float(np.max(np.abs(left.astype(float) - right.astype(float)))), flush=True)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_mixed_joint_probabilities_against_float64(tmp_path, dtype):
    from reference_error import as_given, assert_as_exact, widened

    from dew.decision.model import DecisionModel

    source = tmp_path / "release"
    source.mkdir()
    backbone = FIXTURES / "hf" / "qwen38-dense-tiny"
    joint = FIXTURES / "joint_checkpoint" / "tiny"
    for item in [*backbone.iterdir(), joint / "joint_head_config.json",
                 joint / "joint_head.safetensors"]:
        (source / item.name).symlink_to(item.resolve())
    task = Decide.from_pretrained(source, attention_impl="xla", dtype=dtype)
    task = replace(task, layout=replace(task.layout, max_len=512))
    short = ("refund", {"q": Choice("team", ["pay", "tech", "sales"])})
    long = ("invoice " * 20, {"q": Choice("which department", ["billing", "technical", "sales",
                                                             "support", "account", "other"])})
    rows = [own[0] for own in task.rows([short, long])]
    wide_model = DecisionModel(task.model.backbone.clone(dtype=jnp.float64),
                               task.model.head.clone(dtype=jnp.float64))
    wide = replace(task, model=wide_model)

    def probabilities(held, variables, together):
        scores = held.scored(rows if together else rows[:1], variables)[0][0, :3]
        return np.asarray(held.calibration.probabilities(short[1]["q"], scores))

    def draw(seed):
        rng = np.random.default_rng(seed)
        return (jax.tree.map(lambda leaf: np.asarray(
            leaf * (1 + rng.normal(0, 0.05, np.shape(leaf))), np.asarray(leaf).dtype), task.variables),)

    def truth(variables):
        with jax.enable_x64(new_val=True), algebra.passes(False):
            return probabilities(wide, widened(variables), False)

    assert_as_exact(as_given(lambda variables: probabilities(task, variables, True)),
                    as_given(lambda variables: probabilities(task, variables, False)), draw,
                    "mixed joint probabilities", seeds=16, truth_fn=truth)


def test_isolate_the_recurrence_roundings(tmp_path, monkeypatch):
    import dew.nn.linear as linear
    from dew.nn.kernels import recurrence_algebra
    from dew.nn.precision import float64_sum

    source = tmp_path / "release"
    source.mkdir()
    backbone = FIXTURES / "hf" / "qwen38-dense-tiny"
    joint = FIXTURES / "joint_checkpoint" / "tiny"
    for item in [*backbone.iterdir(), joint / "joint_head_config.json",
                 joint / "joint_head.safetensors"]:
        (source / item.name).symlink_to(item.resolve())
    task = Decide.from_pretrained(source, attention_impl="xla", dtype="bfloat16")
    task = replace(task, layout=replace(task.layout, max_len=512))
    short = ("refund", {"q": Choice("team", ["pay", "tech", "sales"])})
    long = ("invoice " * 20, {"q": Choice("which department", ["billing", "technical", "sales",
                                                             "support", "account", "other"])})
    rows = [own[0] for own in task.rows([short, long])]
    single = DecisionInputs.collate(rows[:1], task.specials.pad)
    padded = jax.tree.map(lambda leaf: leaf[:1], DecisionInputs.collate(rows, task.specials.pad))
    original = linear.chunk_gated_delta_rule

    def captured(inputs):
        records = []

        def record(*args, **kwargs):
            result = original(*args, **kwargs)
            records.append((*args[:5], result[0]))
            return result

        with monkeypatch.context() as patch:
            patch.setattr(linear, "chunk_gated_delta_rule", record)
            states = task.model.states(task.variables, inputs)
        return states, records[0]

    def show(mode):
        compiled = jax.jit(lambda inputs: captured(inputs))
        alone, extra = compiled(single), compiled(padded)
        for index, (left, right) in enumerate(zip(jax.tree.leaves(alone), jax.tree.leaves(extra),
                                                strict=True)):
            left, right = np.asarray(left), np.asarray(right)
            right = right[tuple(slice(0, width) for width in left.shape)]
            hits = np.argwhere(left != right)
            print("RECURRENCE", mode, index, left.shape, len(hits),
                  hits[:3].tolist(), float(np.max(np.abs(left.astype(float) - right.astype(float)))),
                  flush=True)

    with algebra.passes(False):
        show("fp32")
        jax.clear_caches()
        with monkeypatch.context() as patch:
            def rounded_once(left, right, precision=None):
                return float64_sum(jnp.matmul, left, right, dtype=jnp.result_type(left, right))

            patch.setattr(recurrence_algebra, "float32_matmul", rounded_once)
            show("wide_products")
