"""One run of the training qualification (tests/test_training_qualification.py).

It trains a tiny transformers Llama loaded through `Pretrained.load`, with
dropout, EMA, two-micro-step accumulation, dynamic loss scaling and a clipped
AdamW, and checkpoints every CHECKPOINT_STEP. The `interrupted` run blocks
after step INTERRUPT_STEP, once that checkpoint has landed, for the test to
SIGKILL it. The `resumed` run restores that checkpoint in the middle of an
accumulation. Each finished run writes its whole state, its export and the
export's fp32 logits; the baseline also writes the fp32 loss and the gradient
of every source tensor, for the test to hold to transformers.

    python tests/qualification_worker.py <directory> baseline|interrupted|resumed
"""

import json
import os
import sys
import threading
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew.checkpoints import Checkpoints
from dew.data import DataPartition, Loading, TokenWindows
from dew.data.dataset import GlobalStream
from dew.interop import Pretrained
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.training import Trainer
from dew.training.tracker import LocalTracker

STEPS = 24
CHECKPOINT_STEP = 5
INTERRUPT_STEP = 7
SEQUENCE = 32
BATCH = 8


def mid_accumulation(state) -> bool:
    """Whether a restored state sits between the two micro-steps of an
    update, holding the first one's gradient."""
    return (int(state.microstep) % 2 == 1 and state.accumulation is not None
            and state.accumulation.mass is not None and float(state.accumulation.mass) > 0)


def write_reference(path: Path, source, variables, probe, step) -> None:
    """The fp32 loss of the probe batch and each source tensor's gradient,
    from the model at its reference attention, with fp32 matmuls on any
    backend (a GPU's default is TF32-class)."""
    parity = LMObjective(source.model.clone(dtype=jnp.float32, attention_impl='reference'),
                         SEQUENCE, ema_decay=None)

    def reference_loss(params):
        stats, _ = parity.loss({"params": params}, probe, step)
        return parity.reduce_loss(stats)[0]

    with jax.default_matmul_precision("highest"):
        loss, gradients = jax.value_and_grad(reference_loss)(variables["params"])
    arrays = {"ids": np.asarray(probe["text"]), "loss": np.asarray(loss)}
    for layout in source.weight_layouts:
        arrays["gradient/" + layout.name] = layout.export({"params": gradients})
    np.savez(path, **arrays)


def state_arrays(state, restored) -> dict[str, np.ndarray]:
    """The final state's leaves by path, each finite, after checking that the
    checkpoint restores every one bit for bit in its own dtype."""
    arrays = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
        value = jax.random.key_data(leaf) if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key) else leaf
        array = np.asarray(value)
        assert np.isfinite(array).all(), f"nonfinite state at {jax.tree_util.keystr(path)}"
        arrays[jax.tree_util.keystr(path)] = array
    for left, right in zip(jax.tree.leaves(state), jax.tree.leaves(restored), strict=True):
        assert left.dtype == right.dtype, "the checkpoint changed a state dtype"
        if jnp.issubdtype(left.dtype, jax.dtypes.prng_key):
            left, right = jax.random.key_data(left), jax.random.key_data(right)
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    return arrays


def export(run: Path, source, variables, probe) -> None:
    """Save the trained variables in the source's format, reload them, and
    hold the reload's fp32 logits on the probe to the trained model's."""
    source.save(run / "export", variables=variables)
    reloaded = Pretrained.load(run / "export", dtype="float32", attention_impl="reference")
    ids = jnp.asarray(probe["text"][:, :-1])
    with jax.default_matmul_precision("highest"):
        expected = source.model.clone(dtype=jnp.float32, attention_impl='reference').apply(variables, ids)
        actual = reloaded.model.apply(reloaded.variables, ids)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-4, rtol=0)
    np.savez(run / "export_logits.npz", ids=np.asarray(ids), logits=np.asarray(actual))


def run(directory: Path, mode: str) -> None:
    out = directory / ("baseline" if mode == "baseline" else "restarted")
    checkpoints = Checkpoints(str(out / "checkpoints"), keep=2)

    class Interrupting(LocalTracker):
        def log(self, scalars, step):
            super().log(scalars, step)
            if mode == "interrupted" and step == INTERRUPT_STEP:
                checkpoints.wait()
                assert checkpoints.latest == CHECKPOINT_STEP, "the interruption missed its checkpoint"
                (directory / "ready-to-kill").write_text(str(os.getpid()))
                threading.Event().wait()

    source = Pretrained.load(directory / "source", dtype="float32", attention_impl="xla")
    objective = LMObjective(source.model.clone(dropout_rate=0.1), SEQUENCE,
                            pretrained=source.variables, ema_decay=0.9)
    data = TokenWindows(path=str(directory / "tokens"), seq_len=SEQUENCE, seed=17, val_batches=1,
                        loading=Loading(workers=0, threads=1)).load(batch=BATCH)
    trainer = Trainer(objective, optax.chain(optax.clip_by_global_norm(0.5),
                                             optax.adamw(0.003, weight_decay=0.01)),
                      key=jax.random.key(11), accumulation=2, dynamic_scale=True,
                      checkpoints=checkpoints, tracker=Interrupting(out / "metrics"))
    before, _, before_position = trainer.place()
    assert mode != "resumed" or mid_accumulation(before), "the resume lost the accumulated gradient"
    stream = data.train(DataPartition.of(trainer.device_mesh))
    assert isinstance(stream, GlobalStream), "the token loader has no global-position stream"
    try:
        probe = next(stream)
    finally:
        stream.close()
    step = Step(step=jnp.int32(0), key=jax.random.key(19), ema=None)
    stats, _, _ = objective.predict(before.variables, probe, step, train=False)
    initial_loss = float(objective.reduce_loss(stats)[0])
    if mode == "baseline":
        write_reference(out / "reference.npz", source, before.variables, probe, step)
    state = trainer.fit(data, steps=STEPS, log_every=1, checkpoint_every=CHECKPOINT_STEP)
    stats, _, _ = objective.predict(state.variables, probe, step, train=False)
    final_loss = float(objective.reduce_loss(stats)[0])
    restored, _, position = trainer.place()
    assert position, "the final checkpoint has no data position"
    np.savez(out / "state.npz", **state_arrays(state, restored))
    export(out, source, state.variables, probe)
    (out / "result.json").write_text(json.dumps({
        "initial_loss": initial_loss, "final_loss": final_loss,
        "restored_step": int(before.step), "restored_updates": int(before.updates),
        "restored_position": None if before_position is None else before_position.decode(),
        "step": int(state.step), "microstep": int(state.microstep), "updates": int(state.updates),
        "position": position.decode(), "state_structure": str(jax.tree.structure(state)),
    }))


if __name__ == "__main__":
    run(Path(sys.argv[1]), sys.argv[2])
