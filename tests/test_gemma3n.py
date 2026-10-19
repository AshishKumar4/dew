"""Gemma 3n's AltUp, LAuReL block and sparse MLP against transformers 5.16.1.

The block fixtures come from tools/hf_reference_b.py: one `Gemma3nTextAltUp`
predicting and correcting a stream of four copies, one
`Gemma3nTextLaurelBlock` and one `Gemma3nTextMLP` at sparsity 0.95, each on
random weights. Everything runs at fp32 on CPU, and each parity test states
its tolerance and the largest difference observed.
"""

from pathlib import Path
from statistics import NormalDist

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from reference_error import assert_as_exact_as_the_reference, assert_leaves_as_exact

from dew.nn.gemma3n import AltUp, AltUpLayer, LaurelBlock, gaussian_topk
from dew.nn.mlp import GatedMLP

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "gemma3n"
HIDDEN, COPIES = 32, 4


def fixture() -> dict:
    with np.load(FIXTURES / "blocks.npz") as data:
        return {key: np.asarray(value) for key, value in data.items()}


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
    import torch
    from transformers import Gemma3nTextConfig
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nTextAltUp

    tensors = fixture()
    clip = float(np.median(np.abs(tensors["altup.prediction_coefs.weight"])))
    assert np.abs(tensors["altup.correction_coefs.weight"]).max() > clip
    tensors["altup.prediction_coefs.weight"][[0, 1], [0, 1]] = 1e10, -1e10
    stream, activated = tensors["stream"], tensors["activated"]
    draws = np.random.default_rng(0)
    cotangents = [draws.standard_normal(shape).astype(np.float32)
                  for shape in (stream.shape, stream.shape, activated.shape)]
    config = Gemma3nTextConfig(hidden_size=HIDDEN, num_hidden_layers=1, altup_num_inputs=COPIES,
                               altup_coef_clip=clip)
    names = ("correction_coefs", "prediction_coefs", "modality_router")

    def reference(dtype):
        altup = Gemma3nTextAltUp(config).to(dtype).train()
        with torch.no_grad():
            for name in names:
                getattr(altup, name).weight.copy_(torch.from_numpy(tensors[f"altup.{name}.weight"]))
            altup.router_norm.weight.copy_(torch.from_numpy(tensors["altup.router_norm.weight"]))
            altup.correct_output_scale.copy_(torch.from_numpy(tensors["altup.correct_output_scale"]))
        predictions = altup.predict(torch.tensor(stream, dtype=dtype))
        corrected = altup.correct(predictions, torch.tensor(activated, dtype=dtype))
        outputs = (predictions, corrected, altup.scale_corrected_output(corrected[0]))
        sum((output * torch.tensor(cotangent, dtype=dtype)).sum()
            for output, cotangent in zip(outputs, cotangents, strict=True)).backward()
        return {"outputs": [output.detach().numpy() for output in outputs], "params": {"altup": {
            **{name: {"kernel": getattr(altup, name).weight.grad.numpy().T} for name in names},
            "router_norm": {"scale": altup.router_norm.weight.grad.numpy()},
            "correct_output_scale": altup.correct_output_scale.grad.numpy()}}}

    def loss(params):
        outputs = Steps(AltUp(num_inputs=COPIES, coef_clip=clip)).apply(
            {"params": params}, jnp.asarray(stream), jnp.asarray(activated), train=True)
        terms = [jnp.sum(output * cotangent) for output, cotangent in zip(outputs, cotangents, strict=True)]
        return sum(terms), outputs

    (_, outputs), gradient = jax.value_and_grad(loss, has_aux=True)(altup_variables(tensors)["params"])
    dew = jax.tree.map(np.asarray, {"outputs": list(outputs), "params": gradient})
    assert_leaves_as_exact(dew, reference(torch.float32), reference(torch.float64),
                           reductions=stream.shape[1] * stream.shape[2])


@pytest.mark.parametrize("optimizer", ["sgd", "adamw"])
def test_training_steps_update_the_clamped_altup_coefficients_as_the_reference_does(optimizer):
    """Two training steps of the gemma3n-tiny decoder through the Trainer, with half of every
    layer's AltUp prediction coefficients past `coef_clip`, against the reference's two steps
    from the same weights: torch's optimizer over Gemma3nForCausalLM in train mode, whose forward
    clamps the stored prediction weight in place, so the update applies to the clamped weight
    (a coefficient at 2 under a clip of 1, gradient 1 and SGD at 0.1 stores 0.9, not 1.9).
    Every stored parameter after the steps is as exact as the reference's float32 run, both
    measured from its float64 run; each tensor alone, since the second step reads the first's
    rounding (`assert_leaves_as_exact`)."""
    import dataclasses
    import json

    import optax
    import torch
    from safetensors.numpy import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    from dew.interop import Pretrained
    from dew.interop.hf_decoders import translate_config, translate_weights
    from dew.objectives.lm import LMObjective
    from dew.training import Trainer
    from tools.diffusers_wan_reference import float64

    directory = Path(__file__).resolve().parent / "fixtures" / "hf" / "gemma3n-tiny"
    loaded = Pretrained.load(str(directory), dtype="float32", attention_impl="reference")
    layers = [tree["altup"]["prediction_coefs"]["kernel"] for name, tree in loaded.variables["params"].items()
              if name.startswith("layers_")]
    clip = float(np.median(np.abs(np.concatenate([np.ravel(kernel) for kernel in layers]))))
    model = loaded.model.clone(altup=dataclasses.replace(loaded.model.altup, coef_clip=clip))
    ids = np.load(directory / "input_ids.npy").astype(np.int32)
    rows = ids[np.arange(jax.device_count()) % len(ids)]
    rate, decay = (0.1, 0.0) if optimizer == "sgd" else (1e-2, 0.1)
    tx = optax.sgd(rate) if optimizer == "sgd" else optax.adamw(rate, eps=1e-8, weight_decay=decay)

    trainer = Trainer(LMObjective(model, rows.shape[1] - 1, head_chunks=1, variables=loaded.variables), tx,
                      key=jax.random.key(0))
    state, _, _ = trainer.place()
    step = trainer.compile(state, {"text": rows})
    for _ in range(2):
        state, *_ = step(state, {"text": rows})

    stored = set(load_file(str(directory / "model.safetensors")))
    config = json.loads((directory / "config.json").read_text())

    def reference(dtype):
        hf_config = AutoConfig.from_pretrained(directory)
        hf_config.altup_coef_clip = clip
        reference = AutoModelForCausalLM.from_pretrained(directory, config=hf_config,
                                                         attn_implementation="eager").float()
        if dtype == torch.float64:
            with float64():
                wide = AutoModelForCausalLM.from_config(hf_config, dtype=torch.float64,
                                                        attn_implementation="eager")
                wide.load_state_dict(reference.state_dict())
            reference = wide
        reference.train()
        if optimizer == "sgd":
            solver = torch.optim.SGD(reference.parameters(), lr=rate)
        else:
            solver = torch.optim.AdamW(reference.parameters(), lr=rate, eps=1e-8, weight_decay=decay)
        tokens = torch.from_numpy(rows).long()
        with float64() if dtype == torch.float64 else torch.enable_grad():
            for _ in range(2):
                reference(input_ids=tokens, labels=tokens).loss.backward()
                solver.step()
                solver.zero_grad()
        tensors = {name: value.detach().numpy() for name, value in reference.state_dict().items()
                   if name in stored}
        return translate_weights(tensors, translate_config(config), config["model_type"],
                                 param_dtype="float64" if dtype == torch.float64 else "float32")["params"]

    dew = jax.tree.map(np.asarray, state.variables["params"])
    with jax.enable_x64(new_val=True):
        truth = jax.tree.map(np.asarray, reference(torch.float64))
    assert_leaves_as_exact(dew, jax.tree.map(np.asarray, reference(torch.float32)), truth, rows=False)


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
    hidden = np.array(tensors["activated"], np.float32)
    hidden[:, 4:] = 0
    real = np.arange(hidden.shape[1]) < 4
    cotangent = np.random.default_rng(0).standard_normal(hidden.shape).astype(np.float32) * real[:, None]
    config = Gemma3nTextConfig(
        vocab_size=64, hidden_size=HIDDEN, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, sliding_window=4,
        hidden_size_per_layer_input=8, altup_num_inputs=COPIES, laurel_rank=8,
        activation_sparsity_pattern=[0.95, 0.0], num_kv_shared_layers=0)

    def reference(dtype):
        mlp = Gemma3nTextMLP(config, layer_idx=0).to(dtype)
        with torch.no_grad():
            for name in names:
                getattr(mlp, name).weight.copy_(torch.from_numpy(tensors[f"mlp.{name}.weight"]))
        x = torch.tensor(hidden, dtype=dtype, requires_grad=True)
        output = mlp(x)
        (output * torch.tensor(cotangent, dtype=dtype)).sum().backward()
        return {"output": output.detach().numpy(), "input": x.grad.numpy(),
                **{name: getattr(mlp, name).weight.grad.numpy().T for name in names}}

    def loss(params, x):
        output = GatedMLP(hidden_features=48, out_features=HIDDEN, activation="geglu",
                          activation_sparsity=0.95).apply({"params": params}, x)
        return jnp.sum(output * cotangent), output

    params = {name: {"kernel": jnp.asarray(tensors[f"mlp.{name}.weight"].T)} for name in names}
    (_, output), (kernels, inputs) = jax.value_and_grad(loss, (0, 1), has_aux=True)(
        params, jnp.asarray(hidden))
    dew = {"output": output, "input": inputs, **{name: kernels[name]["kernel"] for name in names}}
    assert_leaves_as_exact(jax.tree.map(np.asarray, dew), reference(torch.float32), reference(torch.float64),
                           reductions=hidden.shape[0] * hidden.shape[1])


@pytest.mark.parametrize("sparsity", [0.9, 0.95, 0.99])
def test_gaussian_topk_is_as_exact_as_the_references(sparsity):
    """Gemma 3n's sparse gate over 256 rows of 2048 normal activations: Dew's
    fp32 is within tests/reference_error.py's rule of the reference's fp32
    (Gemma3nTextMLP._gaussian_topk), both from the reference in float64, which
    keeps the reference's fp32 cutoff multiplier. Dew's multiplier from jax's
    fp32 erfinv, two ulps off at 0.95, was 3.77 and 6.33 times the reference's
    distance at 0.95 and 0.9, and 1.10 and 1.15 once formed as the reference's."""
    import torch
    from transformers import Gemma3nTextConfig
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nTextMLP

    rows = np.random.default_rng(0).standard_normal((256, 2048)).astype(np.float32)
    config = Gemma3nTextConfig(hidden_size=HIDDEN, intermediate_size=rows.shape[1], num_hidden_layers=1,
                               activation_sparsity_pattern=[sparsity])
    mlp = Gemma3nTextMLP(config, layer_idx=0)

    def reference(dtype):
        return mlp._gaussian_topk(torch.tensor(rows, dtype=dtype)).numpy()

    dew = jax.jit(lambda x: gaussian_topk(x, sparsity))(jnp.asarray(rows))
    assert_as_exact_as_the_reference(np.asarray(dew), reference(torch.float32), reference(torch.float64),
                                     f"the gate at sparsity {sparsity}")


@pytest.mark.parametrize("sparsity", [1e-9, 1 - 1e-9])
def test_gaussian_topk_takes_the_references_cutoff_at_the_ends_of_its_domain(sparsity):
    """A sparsity whose fp32 argument 2p - 1 rounds to -1 or 1 takes the
    reference's infinite cutoff multiplier (torch's icdf there): every value
    kept as infinity below, and none kept above, bit for bit as the
    reference's gate in fp32 and bf16, which both take the fp32 argument. A
    float64 model forms its own float64 argument, which these do not end."""
    import torch
    from transformers import Gemma3nTextConfig
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nTextMLP

    rows = np.random.default_rng(0).standard_normal((4, 64)).astype(np.float32)
    config = Gemma3nTextConfig(hidden_size=HIDDEN, intermediate_size=rows.shape[1], num_hidden_layers=1,
                               activation_sparsity_pattern=[sparsity])
    mlp = Gemma3nTextMLP(config, layer_idx=0)
    for torch_dtype, dtype in ((torch.float32, jnp.float32), (torch.bfloat16, jnp.bfloat16)):
        dew = jax.jit(lambda x: gaussian_topk(x, sparsity))(jnp.asarray(rows, dtype))
        theirs = mlp._gaussian_topk(torch.tensor(rows).to(torch_dtype)).float().numpy()
        np.testing.assert_array_equal(np.asarray(dew, np.float32), theirs)


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
