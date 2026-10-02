"""Check the native CausalTransformer against the saved exact nanoGPT probe.

PYTHONPATH=src python tools/nanogpt_parity.py --probe-directory DIRECTORY
reads the shared-state checkpoints, logits, gradients and AdamW updates
written by training-parity/torch_training_probe.py at steps 0, 40 and 92.
--train-smoke instead runs the bias-free dropout architecture through Trainer
for three updates at the probe's batch size and sequence length.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew.interop.hf_decoders import translate_weights
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.base import Step, scalar_loss
from dew.objectives.lm import LMObjective


def converted(state):
    """nanoGPT Linear tensors as GPT-2 Conv1D tensors, then the public HF map."""
    tensors = {name: value.T if name.endswith('.weight') and any(part in name for part in (
        '.attn.c_attn.', '.attn.c_proj.', '.mlp.c_fc.', '.mlp.c_proj.')) else value
        for name, value in state.items()}
    config = {'tie_embeddings': True}
    variables = translate_weights(tensors, config, 'gpt2')
    return jax.tree.map(jnp.asarray, variables)


def train_smoke(model, directory, starts, tokens):
    """The six-layer, 384-wide, bias-free dropout architecture in Dew's trainer."""
    from dew import Trainer
    from dew.data import Dataset

    model = model.clone(norm_bias=False, attention_bias=False, mlp_bias=False,
                        dropout_rate=.2, embedding_dropout_rate=.2, attention_dropout_rate=.2)
    weights = converted(dict(np.load(directory / 'probe-a-step-zero.npz')))
    rows = [np.stack([tokens[start:start + 257] for start in batch]).astype(np.int32)
            for batch in starts[:3]]
    objective = LMObjective(model, seq_len=256, ema_decay=None, pretrained=weights)
    mask = jax.tree.map(lambda value: value.ndim >= 2, weights['params'])
    rate = optax.warmup_cosine_decay_schedule(0., 1e-3, 100, 5000, end_value=1e-4)
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(
        rate, b1=.9, b2=.99, weight_decay=.1, mask=mask))
    data = Dataset(train=lambda partition: iter({'text': batch} for batch in rows),
                   val=None, records=3 * 64, batch=64)
    state = Trainer(objective, optimizer, key=jax.random.key(1337)).fit(data, steps=3)
    assert int(state.step) == 3
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(state.params))
    movement = max(float(jnp.max(jnp.abs(before - after))) for before, after in
                   zip(jax.tree.leaves(weights), jax.tree.leaves(state.params), strict=True))
    assert movement > 0
    return {'steps': int(state.step), 'dropout': .2, 'bias': False, 'max_weight_movement': movement}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--train-smoke', action='store_true')
    args = parser.parse_args()
    directory = args.probe_directory
    reference = json.loads((directory / 'torch-result.json').read_text())
    tokens = np.fromfile(directory.parent / 'tokens' / 'train.bin', np.uint16)
    starts = np.load(directory / 'batch-starts.npy')
    model = CausalTransformer(vocab_size=256, emb_features=384, num_layers=6, num_heads=6,
                              max_seq_len=256, position_embedding='learned', mlp='gelu_exact',
                              mlp_bias=True, norm_type='layer', norm_bias=True,
                              attention_bias=True, qk_norm=False, attention_impl='reference',
                              precision=jax.lax.Precision.HIGHEST)
    objective = LMObjective(model, seq_len=256, ema_decay=None)
    step = Step(step=jnp.asarray(0), key=jax.random.key(0), ema=None)
    score = jax.jit(jax.value_and_grad(lambda params, batch: scalar_loss(objective, params, batch, step)[0]))
    forward = jax.jit(model.apply)
    observations = []
    for index in (() if args.train_smoke else (0, 40, 92)):
        weights = converted(dict(np.load(directory / f'shared-{index}-weights.npz')))
        gradients = converted(dict(np.load(directory / f'shared-{index}-grads.npz')))
        rows = np.stack([tokens[start:start + 257] for start in starts[index]]).astype(np.int32)
        loss, actual_grads = score(weights, {'text': jnp.asarray(rows)})
        logits = np.asarray(forward(weights, jnp.asarray(rows[:, :-1])))
        expected_logits = np.load(directory / f'shared-{index}-logits.npy')
        gradient_gap = max(float(jnp.max(jnp.abs(left - right))) for left, right in
                           zip(jax.tree.leaves(actual_grads), jax.tree.leaves(gradients), strict=True))
        mask = jax.tree.map(lambda value: value.ndim >= 2, weights['params'])
        optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(
            reference['learning_rates'][index], b1=.9, b2=.99, weight_decay=.1, mask=mask))
        state = optimizer.init(weights['params'])
        if index:
            mu = converted(dict(np.load(directory / f'shared-{index}-mu.npz')))['params']
            nu = converted(dict(np.load(directory / f'shared-{index}-nu.npz')))['params']
            state = (state[0], (state[1][0]._replace(count=jnp.asarray(index, jnp.int32), mu=mu, nu=nu),
                                *state[1][1:]))
        updates, _ = optimizer.update(actual_grads['params'], state, weights['params'])
        updated = optax.apply_updates(weights['params'], updates)
        expected = converted(dict(np.load(directory / f'shared-{index}-updated.npz')))['params']
        update_gap = max(float(jnp.max(jnp.abs(left - right))) for left, right in
                         zip(jax.tree.leaves(updated), jax.tree.leaves(expected), strict=True))
        record = {'step': index, 'loss_gap': abs(float(loss) - reference['losses'][index]),
                  'max_logit_gap': float(np.max(np.abs(logits - expected_logits))),
                  'max_gradient_gap': gradient_gap, 'max_updated_weight_gap': update_gap}
        observations.append(record)
        print(json.dumps(record), flush=True)
        assert record['loss_gap'] < 1e-5 and record['max_logit_gap'] < 1e-4
        assert gradient_gap < 1e-5 and update_gap < 1e-5
    result = {'model': 'native CausalTransformer', 'dtype': 'float32', 'precision': 'highest',
              'jax': jax.__version__, 'device': jax.devices()[0].device_kind,
              'reference': 'nanoGPT shared-state probe', 'observations': observations}
    if args.train_smoke:
        result['training'] = train_smoke(model, directory, starts, tokens)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
