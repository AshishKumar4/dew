"""Gemma 3n's AltUp, LAuReL block and sparse MLP against transformers 5.16.1.

The block fixtures come from tools/hf_reference_b.py: one `Gemma3nTextAltUp`
predicting and correcting a stream of four copies, one
`Gemma3nTextLaurelBlock` and one `Gemma3nTextMLP` at sparsity 0.95, each on
random weights. Everything runs at fp32 on CPU, and each parity test states
its tolerance and the largest difference observed.
"""

import contextlib
from pathlib import Path
from statistics import NormalDist

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from interop_support import fixture_arrays
from reference_error import as_given, assert_as_exact, torch_with_gradients, with_gradients

from dew.nn.gemma3n import AltUp, AltUpLayer, LaurelBlock
from dew.nn.mlp import GatedMLP, gaussian_topk

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "gemma3n"
HIDDEN, COPIES = 32, 4
ALTUP_SEEDS, PADDED_SEEDS = 128, 256
SPARSITY_SEEDS = {0.9: 8, 0.95: 8, 0.99: 32}
"""Each site's declared seeds for check 1 (`assert_as_exact`), sized by `seeds_for` (on armada)."""


def fixture() -> dict:
    return fixture_arrays(FIXTURES / "blocks.npz")


class Steps(nn.Module):
    """Both AltUp steps in one call, so one init holds every leaf."""
    spec: AltUp

    def setup(self):
        self.altup = AltUpLayer(spec=self.spec, emb_features=HIDDEN, name="altup")

    def __call__(self, stream, activated, train: bool = False):
        predictions = self.altup.predict(stream, train=train)
        corrected = self.altup.correct(predictions, activated, train=train)
        return predictions, corrected, self.altup.scale_corrected_output(
            corrected[self.spec.active_idx])


def steps(spec: AltUp, variables: dict, stream, activated, train: bool = False) -> tuple:
    outputs = Steps(spec).apply(variables, stream, activated, train=train)
    return tuple(np.asarray(output) for output in outputs)


def altup_variables(tensors: dict) -> dict:
    return {"params": {"altup": {
        "correct_output_scale": jnp.asarray(tensors["altup.correct_output_scale"]),
        "correction_coefs": {"kernel": jnp.asarray(tensors["altup.correction_coefs.weight"].T)},
        "prediction_coefs": {"kernel": jnp.asarray(tensors["altup.prediction_coefs.weight"].T)},
        "modality_router": {"kernel": jnp.asarray(tensors["altup.modality_router.weight"].T)},
        "router_norm": {"scale": jnp.asarray(tensors["altup.router_norm.weight"])},
    }}}


def test_altup_predicts_and_corrects_like_the_reference():
    """`predict`: the copies mixed by a per-token matrix of the active
    copy's modalities, added back; `correct`: the innovation of the block's
    output over the active prediction, scaled per copy by one plus a
    coefficient, added to every prediction; then the output scale on the
    active copy. Tolerance 1e-5; observed 1.2e-07, 4.8e-07 and 1.5e-08."""
    tensors = fixture()
    predictions, corrected, scaled = steps(
        AltUp(num_inputs=COPIES), altup_variables(tensors),
        jnp.asarray(tensors["stream"]), jnp.asarray(tensors["activated"]))
    assert float(np.max(np.abs(predictions - tensors["predictions"]))) < 1e-5
    assert float(np.max(np.abs(corrected - tensors["corrected"]))) < 1e-5
    assert float(np.max(np.abs(scaled - tensors["scaled"]))) < 1e-5


def test_altup_trains_past_its_coefficient_clip_as_the_reference_does():
    """A training step with half of the prediction coefficients past
    `coef_clip`, two of them at 1e10. The reference clamps the stored
    prediction weight before its product, so the forward reads the clamped
    values and the gradient reaches every one of them, while the correction
    clamps inside its product, which zeroes the clamped ones' gradient. The
    outputs and the gradients of all five AltUp leaves are as exact as the
    reference's, both measured from the reference in float64."""
    import contextlib

    import torch
    from transformers import Gemma3nTextConfig
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nTextAltUp

    from tools.torch_precision import float64

    tensors = fixture()
    clip = float(np.median(np.abs(tensors["altup.prediction_coefs.weight"])))
    assert np.abs(tensors["altup.correction_coefs.weight"]).max() > clip
    tensors["altup.prediction_coefs.weight"][[0, 1], [0, 1]] = 1e10, -1e10
    config = Gemma3nTextConfig(hidden_size=HIDDEN, num_hidden_layers=1, altup_num_inputs=COPIES,
                               altup_coef_clip=clip)
    names = ("correction_coefs", "prediction_coefs", "modality_router")

    def draw(seed):
        """The fixture's stream and activations at seed 0, else each drawn normal at its own scale,
        and the outputs' cotangents standard normal."""
        draws = np.random.default_rng(seed)
        inputs = (tensors["stream"], tensors["activated"]) if seed == 0 else tuple(
            (np.std(tensors[name]) * draws.standard_normal(tensors[name].shape)).astype(np.float32)
            for name in ("stream", "activated"))
        stream, activated = inputs
        if seed == 0:
            draws = np.random.default_rng(0)
        cotangents = tuple(draws.standard_normal(shape).astype(np.float32)
                           for shape in (stream.shape, stream.shape, activated.shape))
        return (stream, activated, *cotangents)

    def reference(dtype, stream, activated, *cotangents):
        # The float64 truth runs under `float64`, the norm's and the router's float32 pins included.
        with float64() if dtype == torch.float64 else contextlib.nullcontext():
            altup = Gemma3nTextAltUp(config).to(dtype).train()
            with torch.no_grad():
                for name in names:
                    getattr(altup, name).weight.copy_(torch.from_numpy(tensors[f"altup.{name}.weight"]))
                altup.router_norm.weight.copy_(torch.from_numpy(tensors["altup.router_norm.weight"]))
                altup.correct_output_scale.copy_(torch.from_numpy(tensors["altup.correct_output_scale"]))
            class Stepped(torch.nn.Module):
                """AltUp's prediction, correction and scaled correction in one forward, which
                `torch.func.functional_call` runs on the pullback's weights (the prediction weight
                clamped in place, as the reference's training forward clamps it)."""

                def __init__(self):
                    super().__init__()
                    self.altup = altup

                def forward(self, stream, activated):
                    predictions = self.altup.predict(stream)
                    corrected = self.altup.correct(predictions, activated)
                    return predictions, corrected, self.altup.scale_corrected_output(corrected[0])

            stepped = Stepped()
            held = {name: leaf.detach().numpy() for name, leaf in stepped.named_parameters()}
            host = held["altup.router_norm.weight"].dtype
            # The draw's cotangents, pulled back through the gate's torch twin.
            outputs, gradients, _, _ = torch_with_gradients(
                lambda stored, *inputs: torch.func.functional_call(stepped, stored, inputs))(
                held, np.asarray(stream, host), np.asarray(activated, host), cotangents)
        return {"outputs": list(outputs), "params": {"altup": {
            **{name: {"kernel": gradients[f"altup.{name}.weight"].T} for name in names},
            "router_norm": {"scale": gradients["altup.router_norm.weight"]},
            "correct_output_scale": gradients["altup.correct_output_scale"]}}}

    params = altup_variables(tensors)["params"]

    @jax.jit
    def stepped(stream, activated, *cotangents):
        def run(params):
            return Steps(AltUp(num_inputs=COPIES, coef_clip=clip)).apply(
                {"params": params}, stream, activated, train=True)

        # The draw's cotangents, pulled back through the gate.
        outputs, gradient = with_gradients(run)(params, tuple(cotangents))
        return {"outputs": list(outputs), "params": gradient}

    assert_as_exact(stepped, as_given(lambda *inputs: reference(torch.float32, *inputs)), draw,
                    "an AltUp step past its clip", seeds=ALTUP_SEEDS,
                    truth_fn=lambda *inputs: reference(torch.float64, *inputs))


STEP_ROWS, STEP_LENGTH = 4, 32
"""A training step's draw: rows a device and ids a row, within the checkpoint's 64 positions."""
UPDATE_SEEDS = {0.0: 8, 0.1: 8}
"""Each decay's declared seeds for the update's check 1, sized by `seeds_for` (on armada)."""
DENSE_ROWS, DENSE_ORDERS, DENSE_SEEDS = 16, 4, 512
"""The dense decoder's gradient draw's rows, the orders of them its rows are judged over (an orbit of
the mean loss's exact symmetry, `assert_as_exact`'s `orders`), and check 1's declared seeds, sized by
`seeds_for` (on armada). At STEP_ROWS AltUp's output scale in its third layer, a leaf of 32 values each
summed over a draw's tokens, did not resolve within 512 seeds. Over one draw's orders the worst row of
AltUp's correction gradient in its third layer, a sum that nearly cancels, ranges tenfold for the
reference alone (2.2e-3 to 2.0e-2 of its size at one draw, on armada): one order is no measure of it."""


def clipped_gemma3n():
    """The gemma3n-tiny decoder with its AltUp clip at the median prediction
    coefficient, so half of every layer's coefficients start past it, and its
    loaded source and clip."""
    import dataclasses

    from dew.interop import Pretrained

    directory = Path(__file__).resolve().parent / "fixtures" / "hf" / "gemma3n-tiny"
    loaded = Pretrained.load(str(directory), dtype="float32", attention_impl="reference")
    layers = [tree["altup"]["prediction_coefs"]["kernel"] for name, tree in loaded.variables["params"].items()
              if name.startswith("layers_")]
    clip = float(np.median(np.abs(np.concatenate([np.ravel(kernel) for kernel in layers]))))
    return loaded, loaded.model.clone(altup=dataclasses.replace(loaded.model.altup, coef_clip=clip)), clip


def exported(loaded, params) -> dict:
    """`params`, a tree of the loaded decoder's parameters' shape (its parameters or their gradient), as
    the source's float32 tensors by name (`Pretrained.export`)."""
    tensors = loaded.export({**loaded.variables, "params": params})
    return {name: np.asarray(value, np.float32) for name, value in tensors.items()}


def reference_decoder(directory, clip, dtype, sparsity=None):
    """The source's Gemma3nForCausalLM with its AltUp clip at `clip`, in `dtype` (float64 built under
    `float64`, the norm's and the router's float32 pins widened), its gaussian top-k at `sparsity` where
    given, and the checkpoint's weights."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from tools.torch_precision import float64

    config = AutoConfig.from_pretrained(directory)
    config.altup_coef_clip = clip
    if sparsity is not None:
        config.activation_sparsity_pattern = list(sparsity)
    loaded = AutoModelForCausalLM.from_pretrained(directory, config=config,
                                                  attn_implementation="eager").float()
    if dtype == torch.float32:
        return loaded.train()
    with float64():
        wide = AutoModelForCausalLM.from_config(config, dtype=torch.float64, attn_implementation="eager")
    wide.load_state_dict(loaded.state_dict())
    return wide.train()


@pytest.mark.parametrize("decay", [0.0, 0.1])
def test_a_training_step_updates_the_clamped_altup_coefficients_as_the_reference_does(decay):
    """A training step's update of the gemma3n-tiny decoder, with half of every layer's AltUp
    prediction coefficients past `coef_clip`, against the reference's: torch's SGD over
    Gemma3nForCausalLM, whose training forward clamps the stored prediction weight in place, so its
    update, weight decay included, applies to the clamped weight (a coefficient at 2 under a clip of 1,
    gradient 1 and SGD at 0.1 stores 0.9, not 1.9). Both take one drawn gradient (the draw's, in Dew's
    layout and exported to the source's), so the update's own arithmetic is what check 1 holds: every
    stored parameter after it on par with the reference's float32 step from its float64 step.

    The step's gradient is held apart, where it resolves. In this decoder it does not: over 1024 drawn
    rows the reference's own float32 gradient, against its float64 one, carries a heavy tail in every
    leaf (a few rows' errors 1e5 times the median), which no seed count or pooling resolves; layer 1's
    backward amplifies it by cancellation at the gaussian top-k (its 0.95 sparsity) and at AltUp's
    modality router. So the composed gradient is held by check 1 on this decoder without the gaussian
    top-k (`test_the_dense_decoders_gradient_is_the_references`), and the cancelling pieces at their
    own sizes, sparsities included, by AltUp's step past its clip
    (`test_altup_trains_past_its_coefficient_clip_as_the_reference_does`) and the sparse MLP's
    (`test_the_sparse_mlp_trains_on_padded_rows_as_the_reference_does`,
    `test_the_sparse_mlp_is_as_exact_as_the_reference_at_each_sparsity`). The Trainer's step is this
    update of the clamped coefficients, under SGD and AdamW, bit for bit
    (`test_steps_from_coefficients_past_the_clip_are_the_steps_from_the_clamped_ones`); an AdamW
    step itself is not held to torch's here, since optax forms Adam's bias corrections in float32
    (`1 - 0.999 ** t`), 6.4e-6 off at the first step, where torch forms them in float64."""
    import dataclasses

    import optax
    import torch

    from dew.objectives.base import Aux
    from dew.objectives.lm import LMObjective
    from dew.state import TrainState
    from dew.training.transaction import Transaction
    from tools.torch_precision import float64

    loaded, model, clip = clipped_gemma3n()
    params = loaded.variables["params"]
    rate = 0.1
    tx = optax.chain(optax.add_decayed_weights(decay), optax.sgd(rate))
    names = list(loaded.export(loaded.variables))
    # The layouts export a gradient as they export the parameters: by position alone, no offset.
    doubled = loaded.export({**loaded.variables, "params": jax.tree.map(lambda leaf: 2 * leaf, params)})
    for name, value in loaded.export(loaded.variables).items():
        np.testing.assert_array_equal(doubled[name], 2 * value)

    def draw(seed):
        """Rows of ids, which the reference's training forward reads to clamp its coefficients, and a
        gradient for every parameter, standard normal at its leaf's scale."""
        rng = np.random.default_rng([0x3E, seed])
        rows = rng.integers(0, int(model.vocab_size), (STEP_ROWS, STEP_LENGTH)).astype(np.int32)
        gradient = jax.tree.map(lambda leaf: (np.sqrt(np.mean(np.square(np.asarray(leaf, np.float64))))
                                              * rng.standard_normal(leaf.shape)).astype(np.float32), params)
        return rows, gradient

    # The Trainer's own commit of a drawn gradient (`Transaction.commit`): the constraint's
    # projection, the optimizer's update and the projection again, over the clipped decoder's state.
    objective = LMObjective(dataclasses.replace(loaded, model=model), STEP_LENGTH, ema_decay=None)
    transaction = Transaction(objective, tx, 1, (jax.ShapeDtypeStruct((), jnp.float32), Aux({})))
    zero = jnp.zeros((), jnp.int32)
    state = TrainState(step=zero, microstep=zero, updates=zero, variables=loaded.variables,
                       opt_state=tx.init(params), ema=None, key=jax.random.key(0), scale=None,
                       window_size=jnp.ones((), jnp.int32))

    @jax.jit
    def updated(rows, gradient):
        committed, _, _ = transaction.commit(state, gradient, (), None)
        return committed.variables["params"]

    @as_given
    def stepped(rows, gradient):
        return exported(loaded, updated(rows, gradient))

    references = {dtype: reference_decoder(loaded.source, clip, dtype)
                  for dtype in (torch.float32, torch.float64)}
    weights = {key: value.clone() for key, value in references[torch.float32].state_dict().items()}

    def reference(dtype, rows, gradient):
        # Each draw steps from the checkpoint's weights; its training forward clamps them in place.
        reference = references[dtype]
        reference.load_state_dict(weights)
        tokens = torch.from_numpy(np.asarray(rows)).long()
        exported = loaded.export({**loaded.variables, "params": gradient})
        with float64() if dtype == torch.float64 else contextlib.nullcontext(), torch.no_grad():
            reference(input_ids=tokens)
            for name, leaf in reference.named_parameters():
                leaf.grad = torch.as_tensor(np.asarray(exported[name]), dtype=dtype)
            torch.optim.SGD(reference.parameters(), lr=rate, weight_decay=decay).step()
        state_dict = reference.state_dict()
        return {name: state_dict[name].detach().numpy() for name in names}

    label = f"a clamped AltUp update, decay {decay}"
    assert_as_exact(stepped, as_given(lambda *inputs: reference(torch.float32, *inputs)), draw, label,
                    seeds=UPDATE_SEEDS[decay], truth_fn=lambda *inputs: reference(torch.float64, *inputs))


def test_the_dense_decoders_gradient_is_the_references():
    """The gemma3n-tiny decoder's loss gradient, its AltUp coefficients past the clip and its MLPs
    without the gaussian top-k (sparsity 0 in every layer), on par with the reference's from its
    float64 one over drawn rows (check 1): the composition of every layer's backward, AltUp's and the
    per-layer inputs' included, where the reference's own float32 gradient resolves. With the 0.95
    top-k it does not (`test_a_training_step_updates_the_clamped_altup_coefficients_as_the_reference_does`
    says why), and the top-k's backward is held by the sparse MLP's sites."""
    import optax
    import torch

    from tools.torch_precision import float64

    loaded, model, clip = clipped_gemma3n()
    dense = model.clone(activation_sparsity_pattern=(0.0,) * len(model.activation_sparsity_pattern))
    names = list(loaded.export(loaded.variables))
    stored = dense.projected(loaded.variables["params"], updated=False)

    sparsity = (0.0,) * len(model.activation_sparsity_pattern)
    references = {dtype: reference_decoder(loaded.source, clip, dtype, sparsity)
                  for dtype in (torch.float32, torch.float64)}
    pad = references[torch.float32].config.pad_token_id

    assert pad is not None

    def draw(seed):
        """Rows of ids drawn uniformly from the vocabulary, each reading the pad id at its sixth place,
        and the loss's cotangent, one: both embeddings are the reference's
        `nn.Embedding(padding_idx=pad)` and Dew's `padding_idx`, whose pad rows take no gradient from
        what reads them."""
        rows = np.random.default_rng([0xDE, seed]).integers(
            0, int(dense.vocab_size), (DENSE_ROWS, STEP_LENGTH))
        rows[:, 5] = pad
        return rows.astype(np.int32), np.float32(1)

    def orders(inputs):
        """The draw and DENSE_ORDERS - 1 orders of its rows, under which the mean loss is the same."""
        rows, one = inputs
        return [inputs, *((rows[np.random.default_rng([0x0D, order]).permutation(len(rows))], one)
                          for order in range(1, DENSE_ORDERS))]

    def loss(params, rows):
        # The source's loss: the mean cross entropy of each next id.
        logits = dense.apply({**loaded.variables, "params": params}, rows[:, :-1], train=True)
        return optax.softmax_cross_entropy_with_integer_labels(logits, rows[:, 1:]).mean()

    @jax.jit
    def gradient_of(rows, one):
        _, gradient = with_gradients(lambda params: loss(params, rows))(stored, one)
        return gradient

    layouts = {layout.name: layout for layout in loaded.weight_layouts}
    if set(names) != set(layouts):
        raise AssertionError(f"the export writes {sorted(set(names) ^ set(layouts))} beside its layouts")

    @jax.jit
    def differentiated(rows, one):
        """The gradient in the source's tensors by name, each its layout's source array (`export`'s
        tensor before the host copy), in the program that computes it."""
        gradient = {**loaded.variables, "params": gradient_of(rows, one)}
        return {name: layouts[name].source_array(gradient).astype(jnp.float32) for name in names}

    def clamped(dtype):
        """The reference's parameters after its training forward, which clamps the stored coefficients
        in place, as Dew's projection does: the same values whatever rows that forward reads, and
        clamping them again leaves them, so one forward clamps them for every call."""
        reference = references[dtype]
        with float64() if dtype == torch.float64 else contextlib.nullcontext(), torch.no_grad():
            reference(input_ids=torch.from_numpy(draw(0)[0]).long())
        return {name: leaf.detach().numpy() for name, leaf in reference.named_parameters()}

    held = {dtype: clamped(dtype) for dtype in references}

    def reference(dtype, rows, one):
        reference = references[dtype]
        tokens = torch.from_numpy(np.asarray(rows)).long()
        with float64() if dtype == torch.float64 else contextlib.nullcontext():
            def run(weights):
                return torch.func.functional_call(reference, weights, (),
                                                  {"input_ids": tokens, "labels": tokens}).loss

            _, gradients = torch_with_gradients(run)(held[dtype], one)
        return {name: gradients[name] for name in names}

    # The reference on one thread: its tiny steps gain nothing from more, and Dew's program, dispatched
    # before it runs, takes the host's other cores meanwhile.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert_as_exact(differentiated, as_given(lambda *inputs: reference(torch.float32, *inputs)), draw,
                        "the dense decoder's gradient", seeds=DENSE_SEEDS,
                        truth_fn=lambda *inputs: reference(torch.float64, *inputs), orders=orders)
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("optimizer", ["sgd", "adamw"])
@pytest.mark.parametrize("objective", ["lm", "grpo", "ppo"])
def test_steps_from_coefficients_past_the_clip_are_the_steps_from_the_clamped_ones(objective, optimizer):
    """The reference clamps the stored prediction weight in place before each forward, so
    training steps from AltUp coefficients past `coef_clip` are the steps from the clamped ones,
    every later step reading the clamp of what the step before it stored. Dew's two steps, SGD's
    and AdamW's, its decay included, are, bit for bit, under an LM loss, whose forward trains
    the model, under GRPO's, whose token scores read it without training, and under PPO's, whose
    critic is a value head over its own Gemma 3n: each step reads the coefficients and updates
    them as each network's constraint projects them (`Objective.projected`)."""
    import dataclasses

    import optax
    from objective_cases import SEQ_LEN, packed

    from dew.objectives.lm import LMObjective
    from dew.objectives.rl import GRPOObjective, PPOObjective, ValueHead
    from dew.objectives.rl.ppo import OLD_VALUES_KEY, RETURNS_KEY
    from dew.objectives.rl.sessions import RESPONSE_MASK_KEY
    from dew.training import Trainer

    loaded, model, clip = clipped_gemma3n()

    def stepped(start):
        if objective == "lm":
            task = LMObjective(model, SEQ_LEN, head_chunks=1, variables=loaded.variables)
            batch = {"text": np.tile(np.arange(1, SEQ_LEN + 2, dtype=np.int32), (jax.device_count(), 1))}
        else:
            def build(seq_len):
                if objective == "grpo":
                    return GRPOObjective(model, seq_len, variables=loaded.variables)
                return PPOObjective(model, seq_len, critic=ValueHead(model), variables=loaded.variables)

            task, rows = packed(build)
            if objective == "ppo":
                # The rollout's critic values and returns, on the response tokens.
                mask = np.asarray(rows[RESPONSE_MASK_KEY], np.float32)
                rows = {**rows, OLD_VALUES_KEY: 0.5 * mask, RETURNS_KEY: mask}
            # Every device takes a row: the packed rows repeat, as a loader cycles them.
            every = np.arange(jax.device_count())
            batch = jax.tree.map(lambda leaf: np.asarray(leaf)[every % len(leaf)], rows)
        tx = optax.sgd(0.1) if optimizer == "sgd" else optax.adamw(1e-2, weight_decay=0.1)
        trainer = Trainer(task, tx, key=jax.random.key(0))
        state, _, _ = trainer.place()

        def started(path, leaf):
            names = tuple(getattr(key, "key", None) for key in path[-3:])
            return (jax.device_put(start(leaf), leaf.sharding)
                    if names == ("altup", "prediction_coefs", "kernel") else leaf)

        moved = jax.tree_util.tree_map_with_path(started, state.variables)
        state = dataclasses.replace(state, variables=moved)
        step = trainer.compile(state, batch)
        for _ in range(2):
            state, *_ = step(state, batch)
        return jax.tree.map(np.asarray, state.variables["params"])

    beyond = stepped(lambda kernel: 10 * kernel)
    clamped = stepped(lambda kernel: jnp.clip(10 * kernel, -clip, clip))
    assert jax.tree.structure(beyond) == jax.tree.structure(clamped)
    for ours, theirs in zip(jax.tree.leaves(beyond), jax.tree.leaves(clamped), strict=True):
        np.testing.assert_array_equal(ours, theirs)


def test_the_laurel_block_matches_the_reference():
    """`x + post_laurel_norm(linear_right(linear_left(x)))`. Tolerance
    1e-5; observed 4.8e-07."""
    tensors = fixture()
    variables = {"params": {
        "linear_left": {"kernel": jnp.asarray(tensors["laurel.linear_left.weight"].T)},
        "linear_right": {"kernel": jnp.asarray(tensors["laurel.linear_right.weight"].T)},
        "post_laurel_norm": {"scale": jnp.asarray(tensors["laurel.post_laurel_norm.weight"])}}}
    output = LaurelBlock(rank=8, emb_features=HIDDEN).apply(
        variables, jnp.asarray(tensors["activated"]))
    assert float(np.max(np.abs(np.asarray(output) - tensors["laurel_output"]))) < 1e-5


def test_the_sparse_mlp_matches_the_reference():
    """`Gemma3nTextMLP` at sparsity 0.95: the gate's gaussian top-k before
    its tanh-gelu. Tolerance 1e-5; observed 3.0e-07; the same weights with
    the gate left dense disagree."""
    tensors = fixture()
    variables = {"params": {
        name: {"kernel": jnp.asarray(tensors[f"mlp.{name}.weight"].T)}
        for name in ("gate_proj", "up_proj", "down_proj")}}
    hidden = jnp.asarray(tensors["activated"])
    sparse = GatedMLP(hidden_features=48, out_features=HIDDEN, activation="geglu",
                      activation_sparsity=0.95).apply(variables, hidden)
    assert float(np.max(np.abs(np.asarray(sparse) - tensors["mlp_output"]))) < 1e-5
    dense = GatedMLP(hidden_features=48, out_features=HIDDEN, activation="geglu").apply(
        variables, hidden)
    assert float(np.max(np.abs(np.asarray(dense) - tensors["mlp_output"]))) > 1e-2


def test_the_sparse_mlp_trains_on_padded_rows_as_the_reference_does():
    """A padded row of zeros has a constant gate row, whose deviation is
    zero. The reference's deviation takes a zero gradient there (torch's std
    backward masks it), and so must the port's: sqrt's infinite derivative
    times the masked loss's zero cotangent made every weight's gradient NaN.
    The output and the gradients of the input and the three kernels, the
    padding masked out of the loss, are as exact as the reference's, both
    measured from the reference in float64."""
    import torch
    from transformers import Gemma3nTextConfig
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nTextMLP

    tensors = fixture()
    names = ("gate_proj", "up_proj", "down_proj")
    shape = np.shape(tensors["activated"])
    real = np.arange(shape[1]) < 4

    def draw(seed):
        """The fixture's activations at seed 0, else drawn normal at their scale, rows past the
        fourth padded to zero; the cotangent standard normal, zero on the padding."""
        draws = np.random.default_rng(seed)
        hidden = np.array(tensors["activated"] if seed == 0
                          else np.std(tensors["activated"]) * draws.standard_normal(shape), np.float32)
        hidden[:, 4:] = 0
        if seed == 0:
            draws = np.random.default_rng(0)
        cotangent = draws.standard_normal(hidden.shape).astype(np.float32) * real[:, None]
        return hidden, cotangent

    config = Gemma3nTextConfig(
        vocab_size=64, hidden_size=HIDDEN, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, sliding_window=4,
        hidden_size_per_layer_input=8, altup_num_inputs=COPIES, laurel_rank=8,
        activation_sparsity_pattern=[0.95, 0.0], num_kv_shared_layers=0)

    def reference(dtype, hidden, cotangent):
        mlp = Gemma3nTextMLP(config, layer_idx=0).to(dtype)
        with torch.no_grad():
            for name in names:
                getattr(mlp, name).weight.copy_(torch.from_numpy(tensors[f"mlp.{name}.weight"]))
        return pulled(mlp, names, np.asarray(hidden, torch.empty(0, dtype=dtype).numpy().dtype), cotangent)

    params = {name: {"kernel": jnp.asarray(tensors[f"mlp.{name}.weight"].T)} for name in names}

    mlp = GatedMLP(hidden_features=48, out_features=HIDDEN, activation="geglu", activation_sparsity=0.95)

    @jax.jit
    def stepped(hidden, cotangent):
        return differentiated(mlp, params, hidden, cotangent)

    assert_as_exact(stepped, as_given(lambda *inputs: reference(torch.float32, *inputs)), draw,
                    "a sparse MLP step over padded rows", seeds=PADDED_SEEDS,
                    truth_fn=lambda *inputs: reference(torch.float64, *inputs))


def pulled(mlp, names, rows, cotangent) -> dict:
    """The reference MLP's output and its input's and projections' gradients (kernels, transposed) under
    the draw's `cotangent`, pulled back through the gate's torch twin from the weights `mlp` holds."""
    import torch

    def run(stored, x):
        return torch.func.functional_call(mlp, stored, (x,))

    stored = {f"{name}.weight": getattr(mlp, name).weight.detach().numpy() for name in names}
    output, gradients, at_input = torch_with_gradients(run)(stored, rows, cotangent)
    return {"output": output, "input": at_input, **{name: gradients[f"{name}.weight"].T for name in names}}


def differentiated(mlp: GatedMLP, params, rows, cotangent) -> dict:
    """Dew's MLP's output and its input's and projections' gradients under the draw's `cotangent`,
    pulled back through the gate, by `pulled`'s names."""
    output, kernels, at_input = with_gradients(lambda params, x: mlp.apply({"params": params}, x))(
        params, rows, cotangent)
    return {"output": output, "input": at_input,
            **{name: kernel["kernel"] for name, kernel in kernels.items()}}


def sparse_mlps(sparsity: float, width: int, rows: int, dtype, hidden: int = HIDDEN):
    """The reference's public `Gemma3nTextMLP` and Dew's `GatedMLP` at `sparsity` over the
    same random weights, `width` gate activations a row of `hidden` features, and `rows`
    random inputs; the reference takes other rows (`rows`) as well."""
    import torch
    from transformers import Gemma3nTextConfig
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nTextMLP

    draws = np.random.default_rng(0)
    names = ("gate_proj", "up_proj", "down_proj")
    shapes = {"gate_proj": (width, hidden), "up_proj": (width, hidden), "down_proj": (hidden, width)}
    weights = {name: (draws.standard_normal(shape) / np.sqrt(shape[1])).astype(np.float32)
               for name, shape in shapes.items()}
    inputs = draws.standard_normal((rows, hidden)).astype(np.float32)
    config = Gemma3nTextConfig(hidden_size=hidden, intermediate_size=width, num_hidden_layers=1,
                               num_attention_heads=1, num_key_value_heads=1, head_dim=max(hidden, 2),
                               activation_sparsity_pattern=[sparsity])

    def reference(torch_dtype, cotangent=None, rows=None):
        mlp = Gemma3nTextMLP(config, layer_idx=0).to(torch_dtype)
        with torch.no_grad():
            for name in names:
                getattr(mlp, name).weight.copy_(torch.from_numpy(weights[name]))
        x = torch.tensor(inputs if rows is None else rows).to(torch_dtype)
        if cotangent is None:
            return {"output": mlp(x).detach().float().numpy()}
        return pulled(mlp, names, x.numpy(), cotangent)

    model = GatedMLP(hidden_features=width, out_features=hidden, activation="geglu",
                     activation_sparsity=sparsity, dtype=dtype)
    params = {name: {"kernel": jnp.asarray(weights[name].T, dtype)} for name in names}
    return reference, model, params, jnp.asarray(inputs, dtype)


@pytest.mark.parametrize("sparsity", [0.9, 0.95, 0.99])
def test_the_sparse_mlp_is_as_exact_as_the_reference_at_each_sparsity(sparsity):
    """Gemma 3n's sparse MLP through the reference's public forward and backward, 256 rows of
    2048 gate activations, at each sparsity a released pattern uses or bounds: Dew's fp32
    output and gradients are within tests/reference_error.py's rule of the reference's fp32,
    both from the reference in float64. The gate's cutoff multiplier is the reference's (its
    fp32 icdf); jax's own fp32 erfinv lands two ulps from it at 0.95. Four features a row keep
    the projections' own rounding below the cutoff's, which a wide row hides: that multiplier
    put the down projection's gradient 3.4 and 2.4 times the reference's distance at 0.9 and
    0.95, and the right one 1.0 and 1.1."""
    import torch

    reference, model, params, hidden = sparse_mlps(sparsity, 2048, 256, jnp.float32, hidden=4)

    def draw(seed):
        """The shared rows and a cotangent from seed 1 at seed 0, else both standard normal."""
        if seed == 0:
            return np.asarray(hidden), np.random.default_rng(1).standard_normal((256, 4)).astype(np.float32)
        draws = np.random.default_rng(seed)
        return tuple(draws.standard_normal((256, 4)).astype(np.float32) for _ in range(2))

    @jax.jit
    def stepped(rows, cotangent):
        return differentiated(model, params, rows, cotangent)

    assert_as_exact(stepped, as_given(lambda rows, cotangent: reference(torch.float32, cotangent, rows)),
                    draw, f"a sparse MLP step at sparsity {sparsity}", seeds=SPARSITY_SEEDS[sparsity],
                    truth_fn=lambda rows, cotangent: reference(torch.float64, cotangent, rows))


@pytest.mark.parametrize("sparsity", [1e-9, 1 - 1e-9])
def test_the_sparse_mlp_takes_the_references_cutoff_at_the_ends_of_its_domain(sparsity):
    """A sparsity whose fp32 argument 2p - 1 rounds to -1 or 1 takes the reference's infinite
    cutoff multiplier (torch's icdf there): above, the gate keeps nothing and the MLP outputs
    zeros; below, it keeps every activation as infinity, which the down projection sums to
    NaN. Dew's MLP outputs the reference's, bit for bit, in fp32 and bf16, which both take the
    fp32 argument. A float64 model forms its own float64 argument, which these do not end."""
    import torch

    for torch_dtype, dtype in ((torch.float32, jnp.float32), (torch.bfloat16, jnp.bfloat16)):
        reference, model, params, hidden = sparse_mlps(sparsity, 64, 4, dtype)
        dew = jax.jit(lambda params, x, model=model: model.apply({"params": params}, x))(params, hidden)
        np.testing.assert_array_equal(np.asarray(dew, np.float32), reference(torch_dtype)["output"])


def test_gaussian_topk_carries_a_nan_row_as_relu_does():
    """A gate row holding a NaN has a NaN cutoff, so it keeps no value and every value is NaN, as relu of
    its distance from the cutoff is; a finite row beside it is untouched."""
    rows = jnp.asarray([[0.5, np.nan, -1.0, 2.0], [0.5, 1.0, -1.0, 2.0]], jnp.float32)
    out = np.asarray(jax.jit(gaussian_topk, static_argnums=1)(rows, 0.5))
    assert np.all(np.isnan(out[0])) and np.all(np.isfinite(out[1]))


@pytest.mark.parametrize("sparsity", [0.5, 0.95])
def test_gaussian_topk_has_the_population_cutoff_jacobian(sparsity):
    # Both active and inactive coordinates, separated from the cutoff.
    x = np.array([-2, -1, 0, 1, 5], dtype=np.float64)
    multiplier = NormalDist().inv_cdf(sparsity)
    mean, std = x.mean(), x.std()
    margin = x - mean - multiplier * std
    assert np.min(np.abs(margin)) > 0.05
    cutoff_gradient = (1 + multiplier * (x - mean) / std) / x.size
    expected = (np.eye(x.size) - cutoff_gradient) * (margin > 0)[:, None]
    def apply(row):
        return gaussian_topk(row, sparsity)
    actual = jax.jit(jax.jacrev(apply))(jnp.asarray(x, jnp.float32))
    # Five-term population moments and the normal quantile at fp32.
    # A stopped cutoff gradient or sample deviation misses by >1e-2.
    np.testing.assert_allclose(actual, expected, atol=5e-7, rtol=0)
    # Forward cancellation at the cutoff is bounded in the input scale,
    # not by relative error in the small positive remainder.
    unit = np.finfo(np.float32).eps / 2
    bound = 8 * unit * (np.abs(x) + abs(mean) + abs(multiplier) * std)
    output = np.asarray(apply(jnp.asarray(x, jnp.float32)), dtype=np.float64)
    assert np.all(np.abs(output - np.maximum(margin, 0)) <= bound)
