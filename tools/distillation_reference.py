#!/usr/bin/env python3
"""Write the distillation fixtures tests/test_distillation.py checks against,
and hold the z-loss oracle tests/test_lm_objective.py reads.

The reference is MaxText 0.2.4 (commit 538fe7a3). Its distillation trainer
mixes the student's cross entropy with the KL from a frozen teacher and a
feature distance, `CombinedDistillationStrategy.compute_loss`
(src/maxtext/trainers/post_train/distillation/distillation_utils.py:462-566),
with the knobs `distill_alpha`, `distill_temperature`, `distill_beta`,
`distill_feature_loss_type` and `distill_layer_indices`
(src/maxtext/configs/base.yml:1386-1393, configs/types.py:1626-1633) and
their `*_end`/`*_schedule` anneals (base.yml:1396-1401, types.py:1634-1654,
`compute_schedule` distillation_utils.py:165-194). The teacher runs in eval
mode under a stop_gradient (train_distill.py:705, :309) and a vocabulary
mismatch is refused (train_distill.py:678-682). PaLM's z-loss is
`cross_entropy_with_logits` (src/maxtext/utils/max_utils.py:649-656), masked
and normalised with the cross entropy (trainers/pre_train/train.py:227-234,
:337-340) under `z_loss_multiplier` (base.yml:404, types.py:585).

The loss is transcribed in float64 torch, so autograd gives its exact
gradients, and evaluated on the logits and layer outputs of two tiny
fixed-seed Dew decoders over one batch with padding. What lands under
tests/fixtures/distillation is the models' variables, the batch, both
models' outputs and the reference loss with its terms and its gradients
with respect to the student's logits and projected features; the test drives
Dew's objective on the same variables and chains the gradients through the
models with `jax.vjp`. The suite needs no maxtext install.

    PYTHONPATH=src python tools/distillation_reference.py

What it writes:

- fixture.npz: the variables (`student/`, `teacher/`, `projection_k`
  prefixed, path-joined), the batch, the logits, the features, and per
  feature loss the reference values (`cosine/loss`, `l2/...`).
- meta.json: the reference release and lines, the configuration.
"""

import json
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "distillation"

COMMIT = "538fe7a3f3376d94cf3f04e77741aa6d7e8efa45"
SOURCE = {
    "compute_loss": "src/maxtext/trainers/post_train/distillation/distillation_utils.py:462-566",
    "mask_from_labels": "distillation_utils.py:492-494 with create_labels :635-642",
    "student_log_softmax_at_T": "distillation_utils.py:499",
    "teacher_softmax_at_T": "distillation_utils.py:526",
    "kl_teacher_to_student": "distillation_utils.py:529 (optax.kl_divergence(log_s_T, t_p_T))",
    "temperature_squared": "distillation_utils.py:537-538",
    "hard_loss": "distillation_utils.py:541-543",
    "mix": "distillation_utils.py:547",
    "feature_cast_fp32": "distillation_utils.py:559-560",
    "feature_cosine": "distillation_utils.py:418-427",
    "feature_l2": "distillation_utils.py:430-436",
    "feature_layers": "distillation_utils.py:552-557",
    "total": "distillation_utils.py:562-566",
    "teacher_loss": "distillation_utils.py:532, :545, :582",
    "compute_schedule": "distillation_utils.py:165-194",
    "knobs": "src/maxtext/configs/base.yml:1386-1401, configs/types.py:1626-1654",
    "teacher_frozen": "train_distill.py:705 (eval), :309 (stop_gradient)",
    "vocab_check": "train_distill.py:678-682",
    "z_loss": "src/maxtext/utils/max_utils.py:649-656",
    "z_loss_mask_and_mean": "src/maxtext/trainers/pre_train/train.py:227-234, :337-340",
    "z_loss_knob": "src/maxtext/configs/base.yml:404, configs/types.py:585",
}

BATCH, SEQ, VOCAB, PAD = 2, 8, 37, 0
STUDENT = dict(vocab_size=VOCAB, emb_features=16, num_layers=2, num_heads=2, mlp_features=32,
               max_seq_len=16)
TEACHER = dict(vocab_size=VOCAB, emb_features=24, num_layers=3, num_heads=2, mlp_features=48,
               max_seq_len=16)
PAIRS = ((0, 0), (2, 1))
"""`(teacher layer, student layer)`; the widths differ, so the student's
states go through a projection the fixture also fixes."""
ALPHA, TEMPERATURE, BETA = 0.6, 2.0, 0.7
SEED = 11


def compute_schedule(step: np.ndarray, max_steps: int, start_value: float,
                     end_value: float | None, schedule_type: str) -> np.ndarray:
    """`compute_schedule` (:165-194): the anneal from `start_value` to
    `end_value` over `max_steps`, constant when there is no end."""
    if end_value is None or schedule_type == "constant":
        return np.full(np.shape(step), start_value, np.float32)
    progress = np.clip(np.asarray(step, np.float32) / max_steps, 0.0, 1.0)
    if schedule_type == "linear":
        return start_value + (end_value - start_value) * progress
    if schedule_type == "cosine":
        return end_value + (start_value - end_value) * 0.5 * (1.0 + np.cos(np.pi * progress))
    raise ValueError(f"Unsupported schedule_type: {schedule_type!r}")


def z_loss_reference(logits: np.ndarray, targets: np.ndarray, weights: np.ndarray,
                     coefficient: float) -> tuple[float, float, np.ndarray]:
    """`cross_entropy_with_logits` (max_utils.py:649-656) summed over the
    counted targets and divided by their count (train.py:227-234, :337-340),
    in float64: the loss, the z term alone, and the loss's gradient with
    respect to the logits (the custom vjp's `1 + 2 * z_loss * log_z` factor,
    max_utils.py:706)."""
    logits = np.asarray(logits, np.float64)
    weights = np.asarray(weights, np.float64)
    largest = logits.max(axis=-1, keepdims=True)
    log_z = np.squeeze(largest + np.log(np.exp(logits - largest).sum(axis=-1, keepdims=True)), -1)
    picked = np.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    xent = log_z - picked
    z_term = coefficient * np.square(log_z)
    count = weights.sum()
    loss = float(((xent + z_term) * weights).sum() / count)
    softmax = np.exp(logits - log_z[..., None])
    one_hot = np.zeros_like(logits)
    np.put_along_axis(one_hot, targets[..., None], 1.0, axis=-1)
    gradient = (weights / count)[..., None] * ((1 + 2 * coefficient * log_z)[..., None] * softmax - one_hot)
    return loss, float((z_term * weights).sum() / count), gradient


def distillation_loss(student_logits, teacher_logits, targets, mask, student_features,
                      teacher_features, alpha: float, temperature: float, beta: float,
                      feature_loss_type: str):
    """`CombinedDistillationStrategy.compute_loss` (:462-566) in float64
    torch over one batch: the total and its reported terms.

    `student_features` and `teacher_features` are the `[L, B, T, D]` stacks
    the reference slices with `distill_layer_indices` (:552-557), here the
    pairs already in order.
    """
    import torch

    s_logits = student_logits.to(torch.float64)
    t_logits = teacher_logits.to(torch.float64)
    mask = mask.to(torch.float64)
    valid_count = mask.sum()
    safe_count = torch.clamp(valid_count, min=1.0)
    labels = torch.nn.functional.one_hot(targets, s_logits.shape[-1]).to(torch.float64) * mask[..., None]

    log_s_T = torch.log_softmax(s_logits / temperature, dim=-1)
    t_p_T = torch.softmax(t_logits / temperature, dim=-1)
    log_t_p_T = torch.log_softmax(t_logits / temperature, dim=-1)
    kl_softened_per_pos = (t_p_T * (log_t_p_T - log_s_T)).sum(-1)
    kl_softened_sum = (kl_softened_per_pos * mask).sum()
    soft_loss_sum_scaled = kl_softened_sum * temperature ** 2
    soft_loss_mean = soft_loss_sum_scaled / safe_count

    ce_student_per_pos = -(labels * torch.log_softmax(s_logits, dim=-1)).sum(-1)
    hard_loss_mean = (ce_student_per_pos * mask).sum() / safe_count
    ce_teacher_per_pos = -(labels * torch.log_softmax(t_logits, dim=-1)).sum(-1)
    teacher_loss_mean = (ce_teacher_per_pos * mask).sum() / safe_count

    base_logit_loss = alpha * soft_loss_mean + (1.0 - alpha) * hard_loss_mean

    s_features = student_features.to(torch.float64)
    t_features = teacher_features.to(torch.float64)
    if feature_loss_type == "cosine":
        s_norm = torch.sqrt(torch.clamp(s_features.square().sum(-1, keepdim=True), min=1e-6))
        t_norm = torch.sqrt(torch.clamp(t_features.square().sum(-1, keepdim=True), min=1e-6))
        per = 1.0 - ((s_features / s_norm) * (t_features / t_norm)).sum(-1)
    elif feature_loss_type == "l2":
        per = (s_features - t_features).square().mean(-1)
    else:
        raise ValueError(f"Unsupported feature_loss_type: {feature_loss_type!r}")
    num_valid_terms = torch.clamp(mask.sum(), min=1.0) * per.shape[0]
    feature_mean = (per * mask[None]).sum() / num_valid_terms
    feature_loss = beta * feature_mean

    total_loss = base_logit_loss + feature_loss
    return total_loss, {
        "hard_loss": hard_loss_mean, "soft_loss": soft_loss_mean,
        "kl_div_at_T": kl_softened_sum / safe_count, "feature": feature_mean,
        "teacher_loss": teacher_loss_mean,
    }


def dew_outputs(model, variables, ids, layers):
    """The model's logits and the named layers' outputs under Dew's forward."""
    import jax
    from dew.nn.backbones.causal_transformer import INTERMEDIATES, layer_output, layer_outputs

    logits = model.apply(variables, ids)
    _, state = model.apply(variables, ids, method=type(model).hidden_states,
                           capture_intermediates=layer_outputs, mutable=[INTERMEDIATES])
    kept = [layer_output(state[INTERMEDIATES], index) for index in layers]
    return np.asarray(logits, np.float32), [np.asarray(x, np.float32) for x in kept]


def flat(tree, prefix: str) -> dict[str, np.ndarray]:
    from flax.traverse_util import flatten_dict
    return {prefix + "/".join(path): np.asarray(leaf) for path, leaf in flatten_dict(tree).items()}


def main() -> None:
    import jax
    import jax.numpy as jnp
    import torch
    from flax import linen as nn
    from dew.nn.backbones.causal_transformer import CausalTransformer

    student = CausalTransformer(vocab_size=VOCAB, emb_features=STUDENT["emb_features"],
                                num_layers=STUDENT["num_layers"], num_heads=STUDENT["num_heads"],
                                mlp_features=STUDENT["mlp_features"], max_seq_len=STUDENT["max_seq_len"])
    teacher = CausalTransformer(vocab_size=VOCAB, emb_features=TEACHER["emb_features"],
                                num_layers=TEACHER["num_layers"], num_heads=TEACHER["num_heads"],
                                mlp_features=TEACHER["mlp_features"], max_seq_len=TEACHER["max_seq_len"])
    keys = jax.random.split(jax.random.key(SEED), 4)
    tokens = np.array(jax.random.randint(keys[0], (BATCH, SEQ + 1), 1, VOCAB), np.int32)
    # Padding the loss must leave out: the second row's tail.
    tokens[1, -3:] = PAD
    ids = jnp.asarray(tokens[:, :-1])
    targets = tokens[:, 1:]
    student_vars = student.init(keys[1], ids)
    teacher_vars = teacher.init(keys[2], ids)
    projections = {
        f"projection_{index}": np.asarray(nn.initializers.lecun_normal()(
            jax.random.fold_in(keys[3], index), (STUDENT["emb_features"], TEACHER["emb_features"]),
            jnp.float32))
        for index in range(len(PAIRS))}

    student_logits, student_states = dew_outputs(student, student_vars, ids, [s for _, s in PAIRS])
    teacher_logits, teacher_states = dew_outputs(teacher, teacher_vars, ids, [t for t, _ in PAIRS])
    projected = [state.astype(np.float64) @ projections[f"projection_{index}"].astype(np.float64)
                 for index, state in enumerate(student_states)]
    mask = (targets != PAD).astype(np.float32)

    arrays = {**flat(student_vars, "student/"), **flat(teacher_vars, "teacher/"), **projections,
              "tokens": tokens, "student_logits": student_logits, "teacher_logits": teacher_logits}
    for index, (own, other) in enumerate(zip(student_states, teacher_states)):
        arrays[f"student_feature_{index}"] = own
        arrays[f"teacher_feature_{index}"] = other
    for kind in ("cosine", "l2"):
        s_logits = torch.tensor(student_logits, dtype=torch.float64, requires_grad=True)
        s_features = torch.tensor(np.stack(projected), dtype=torch.float64, requires_grad=True)
        loss, terms = distillation_loss(
            s_logits, torch.tensor(teacher_logits), torch.tensor(targets, dtype=torch.int64),
            torch.tensor(mask), s_features, torch.tensor(np.stack(teacher_states)),
            ALPHA, TEMPERATURE, BETA, kind)
        d_logits, d_features = torch.autograd.grad(loss, (s_logits, s_features))
        arrays[f"{kind}/loss"] = np.float64(loss.item())
        for name, value in terms.items():
            arrays[f"{kind}/{name}"] = np.float64(value.item())
        arrays[f"{kind}/d_student_logits"] = d_logits.numpy()
        arrays[f"{kind}/d_student_features"] = d_features.numpy()
    steps = np.arange(0, 101, 25)
    arrays["schedule/steps"] = steps
    for kind in ("linear", "cosine"):
        arrays[f"schedule/{kind}"] = compute_schedule(steps, 100, ALPHA, 0.1, kind)

    FIXTURES.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURES / "fixture.npz", **arrays)
    meta = {
        "maxtext": "0.2.4", "commit": COMMIT, "source": SOURCE,
        "student": STUDENT, "teacher": TEACHER, "pairs": list(map(list, PAIRS)),
        "alpha": ALPHA, "temperature": TEMPERATURE, "beta": BETA,
        "schedule": {"start": ALPHA, "end": 0.1, "max_steps": 100},
        "batch": BATCH, "seq_len": SEQ, "vocab": VOCAB, "pad_id": PAD, "seed": SEED,
    }
    (FIXTURES / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    for name, value in sorted(arrays.items()):
        if name.startswith(("student/", "teacher/")):
            continue
        print(f"{name}: {value.shape} {value.dtype}"
              + (f" = {float(value):.6f}" if value.ndim == 0 else ""))


if __name__ == "__main__":
    main()
