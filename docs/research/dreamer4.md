# Dreamer 4: a specification from the paper

Dreamer 4 is Hafner, Yan and Lillicrap, "Training Agents Inside of Scalable World Models",
arXiv 2509.24527v1 (29 September 2025). No official code exists: the project page
(danijar.com/project/dreamer4) links only the paper and videos, and github.com/danijar/dreamer4
does not exist. Everything below is the paper's, by equation and section; what the paper leaves
open is listed last and is chosen, never inferred, by an implementation.

Dew implements the world model's core in `dew.objectives.world_model`: a causal tokenizer
(`CausalTokenizer`, trained by `TokenizerObjective`) and the dynamics over its latents (`Dynamics`), trained
by shortcut forcing (`WorldModelObjective`) or, with `shortcut=False`, by diffusion forcing on the same model
and data. `python tools/rl_acceptance.py dreamer4 OUT` compares the two on bouncing balls: both are trained
for the same steps, and their open-loop predictions after held-out context frames are scored at 4 sampling
steps and at 64. This tests the paper's claim that shortcut forcing generates at four steps what diffusion
forcing needs many steps for. The comparison is a tool to run; its result is not recorded here. The agent
(behaviour cloning, reward model and PMPO in imagination) is specified here and reuses DreamerV3's two-hot
heads (`dew.rl.distributions`).

## Pipeline (Algorithm 1)

1. Train the tokenizer on video (Eq. 5), then the world model on tokenized video with actions
   where present (Eq. 7).
2. Add agent tokens and task embeddings; train behaviour cloning and the reward model (Eq. 9)
   beside the video loss, which keeps the dynamics.
3. Train the policy and value heads in imagination (Eqs. 10-11), the transformer frozen.

Every loss term is divided by a running RMS estimate of itself (Section 3); its decay and epsilon
are not stated.

## Causal tokenizer (Section 3.1, Appendix A)

Each frame's patches and `N_b = 512` learned latent tokens pass a transformer; the latents are
projected to `D_b = 16` channels through a tanh, the bottleneck. In the encoder the latents attend
to everything and the patches only to their own modality; in the decoder the latents, projected
back up beside learned readout tokens, attend only to latents, and the readout tokens to
themselves and the latents. The tokenizer is causal in time, so frames decode one at a time.
Training masks each patch with probability `p ~ U(0, 0.9)` drawn per image (MAE), and none at
inference:

    L_tok = L_MSE + 0.2 L_LPIPS                                                         (5)

## Interactive dynamics (Sections 2, 3.2, 3.4)

A timestep's tokens are `S_a` action tokens (each action component embedded and summed with a
learned embedding; an unlabeled video gets the learned embedding alone), one shortcut token (the
step index of `tau` and of `d`, embedded and concatenated), `S_z` spatial tokens (a linear map of
the tokenizer's latents, 512 x 16 regrouped to 256 x 32) and `S_r` register tokens. The
transformer is block-causal over time, space and time attention separate, time attention every
fourth layer, with pre-RMSNorm, RoPE, SwiGLU, QK-norm, logit soft-capping and grouped-query
attention.

Shortcut forcing gives every frame its own signal level `tau` and step size `d`, as diffusion
forcing gives it its own level:

    d ~ 1 / U({1, 2, 4, ..., K_max}),   tau ~ U({0, 1/d, ..., 1 - 1/d}),   d_min = 1 / K_max   (4)

    z_0 ~ N(0, 1),   z_1 ~ D,   z~ = (1 - tau) z_0 + tau z_1,   z^_1 = f(z~, tau, d, a)        (6)

The network predicts the clean latent (x-prediction). At the smallest step it regresses the
latent; at a larger one it matches, in velocity space, the average of two half steps it takes
itself (the bootstrap):

    b'  = (f(z~, tau, d/2, a) - z~) / (1 - tau)
    z'  = z~ + b' d/2
    b'' = (f(z', tau + d/2, d/2, a) - z') / (1 - (tau + d/2))

    L = || z^_1 - z_1 ||^2                                                  if d = d_min
    L = (1 - tau)^2 || (z^_1 - z~) / (1 - tau) - sg(b' + b'') / 2 ||^2       otherwise        (7)

The `(1 - tau)^2` converts the velocity-space error back to x-space (footnote 1). A ramp weight
focuses capacity on the high signal levels:

    w(tau) = 0.9 tau + 0.1                                                                  (8)

The paper does not print `w` inside Eq. 7; Dew multiplies each frame's loss by it.

Generation is autoregressive over frames, each frame denoised in `K = 4` shortcut steps of
`d = 1/4`, the context frames re-noised to `tau_ctx = 0.1`. Section 4 reports that diffusion
forcing at 4 steps loses the quality it has at 64 (Table 2), which is the claim Dew's reproduction
tests.

## Behaviour cloning and reward (Section 3.3)

Agent tokens carry the task's embedding, attend to every modality and are attended by none. Small
policy and reward heads read them, one output layer per multi-token-prediction distance up to
`L = 8`; the reward head is DreamerV3's symexp two-hot, the policy categorical or binary by the
action space:

    L_BC+R = - sum_{n=0..L} ln p(a_{t+n} | h_t) - sum_{n=0..L} ln p(r_{t+n} | h_t)          (9)

Half the batch is uniform sequences, half task-relevant ones; Eq. 9 applies to the relevant ones
and the dynamics loss to the uniform ones.

## Imagination (Section 3.3)

The transformer is frozen and the policy and value heads train on one imagined rollout per
dataset context, `z` drawn from the flow, `a` from the policy, `r` from the reward head:

    R_t = r_t + gamma c_t ((1 - lambda) v_t + lambda R_{t+1}),   R_T = v_T,
    L_val = - sum_t ln p(R_t | s_t),   gamma = 0.997                                          (10)

With `A_t = R_t - v_t`, the states split into those with a non-negative advantage `D+` and the rest
`D-`, pooled over batch and time, and the policy follows PMPO with the KL to a frozen copy of the
behaviour-cloned policy reversed:

    L_pol = (1 - alpha) / |D-| sum_{D-} ln pi(a|s) - alpha / |D+| sum_{D+} ln pi(a|s)
            + beta / N sum_i KL[pi(.|s_i) || pi_prior(.|s_i)],   alpha = 0.5, beta = 0.3     (11)

## Stated sizes

2B parameters (tokenizer 400M, dynamics 1.6B) on 256 to 1024 TPU-v5p at one sequence a device.
Minecraft: `N_z = 256`, context 192 frames, batch lengths 64 and 256 alternated, then 256 alone;
30% of videos taken as single images. Real-world video: `N_z = 512`, context 96, lengths 32 and
128.

## What the paper leaves open

The optimizer, learning rate, schedule, weight decay and gradient clipping; the widths, depths,
head and register counts, RoPE base, GQA groups and soft-cap; `K_max`; `lambda` and the imagination
horizon; the batch size in sequences; the two-hot bin count; the RMS normalizer's decay; the LPIPS
network; how a KV cache serves context frames re-noised to `tau_ctx`; and where the ramp weight
sits in Eq. 7.
